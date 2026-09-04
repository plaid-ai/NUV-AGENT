from __future__ import annotations

import base64
import copy
import hashlib
import json
import tempfile
import unittest
import uuid
from pathlib import Path

from nuvion_app.inference.command_inbox import DurableCommandInbox
from nuvion_app.inference.command_observation import DurableCommandObservationOutbox
from nuvion_app.inference.effect_reconciler import (
    FleetEffectCoordinator,
    ReconcileDeferred,
    ReconcilerRegistry,
)
from nuvion_app.inference.fleet_command import VerifiedFleetCommand
from nuvion_app.inference.reconcile_store import (
    JOB_PHASE_PENDING,
    JOB_PHASE_SUPERSEDED,
    JOB_PHASE_WAITING_RESTART,
    DurableReconcileStore,
)
from nuvion_app.inference.settings_reconciler import (
    AtomicSettingsStore,
    SettingsReconciler,
    UnsupportedSettingsEffect,
    canonical_settings_digest,
    config_env_updates,
)
from nuvion_app.inference.stream_policy import (
    StreamPolicyReconciler,
    StreamRuntimeEvidence,
)
from nuvion_app.runtime.settings_boot_guard import (
    SettingsBootGuardError,
    run_settings_boot_guard,
)
from nuvion_app.runtime.telemetry import verify_model_artifact_identity


def _stream_reconciler(encoder: object) -> StreamPolicyReconciler:
    return StreamPolicyReconciler(
        encoder,
        runtime_evidence=lambda: StreamRuntimeEvidence(True, 0.0),
        health_clock=lambda: 0.0,
    )


def _healthy_event_outbox() -> dict[str, object]:
    return {
        "capacityState": "HEALTHY",
        "unsavedCriticalEvents": 0,
        "safetyStop": False,
        "protocolStop": False,
        "durableSafetyRetained": False,
        "blockedRows": 0,
        "dlqRows": 0,
    }


def _healthy_command_outbox() -> dict[str, object]:
    return {
        "capacityState": "HEALTHY",
        "retentionPressure": False,
        "dlqBlockedRows": 0,
        "dlqRows": 0,
    }


def _command(
    sequence: int,
    *,
    config_version: int | None = None,
    activation: str = "IMMEDIATE",
    sections: dict[str, object] | None = None,
) -> VerifiedFleetCommand:
    payload: dict[str, object] = {
        "configVersion": config_version if config_version is not None else sequence,
        "activation": activation,
        **(
            sections
            or {
                "video": {
                    "width": 640,
                    "height": 480,
                    "fps": 30,
                    "bitrateKbps": 1500,
                }
            }
        ),
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return VerifiedFleetCommand(
        command_id=str(uuid.uuid4()),
        device_id="device-1",
        space_id=1,
        command_type="CONFIG_APPLY",
        schema_version=1,
        issued_at="2026-09-01T00:00:00Z",
        expires_at="2026-09-01T01:00:00Z",
        sequence=sequence,
        payload_base64=base64.urlsafe_b64encode(encoded).decode().rstrip("="),
        payload_hash=hashlib.sha256(encoded).hexdigest(),
        payload=payload,
        actor="operator@example.com",
        authorization_context="SPACE_ADMIN",
        key_id="test",
        required_capability="command.config.apply",
        compact_jws=f"header.{sequence}.signature",
    )


def _stream_command(sequence: int, target: int = 1200) -> VerifiedFleetCommand:
    payload = {
        "policyVersion": sequence,
        "mode": "FIXED",
        "targetBitrateKbps": target,
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return VerifiedFleetCommand(
        command_id=str(uuid.uuid4()),
        device_id="device-1",
        space_id=1,
        command_type="STREAM_POLICY",
        schema_version=1,
        issued_at="2026-09-01T00:00:00Z",
        expires_at="2026-09-01T01:00:00Z",
        sequence=sequence,
        payload_base64=base64.urlsafe_b64encode(encoded).decode().rstrip("="),
        payload_hash=hashlib.sha256(encoded).hexdigest(),
        payload=payload,
        actor="operator@example.com",
        authorization_context="SPACE_ADMIN",
        key_id="test",
        required_capability="command.stream.policy",
        compact_jws=f"header.{sequence}.signature",
    )


class _Runtime:
    def __init__(self, state: dict[str, object] | None = None) -> None:
        self.state = copy.deepcopy(
            state
            or {
                "model": {
                    "pointer": "anomalyclip/prod",
                    "digest": "sha256:" + "a" * 64,
                },
                "labels": {
                    "inspection": ["normal", "defect"],
                    "anomaly": ["defect"],
                },
                "clip": {"enabled": True, "preSeconds": 5, "postSeconds": 5},
                "video": {
                    "width": 640,
                    "height": 480,
                    "fps": 30,
                    "bitrateKbps": 1000,
                },
            }
        )
        self.healthy = True
        self.unsupported = False
        self.restore_calls = 0
        self.apply_calls = 0

    def snapshot(self) -> dict[str, object]:
        return copy.deepcopy(self.state)

    def apply_immediate(self, desired) -> dict[str, object]:
        self.apply_calls += 1
        if self.unsupported:
            raise UnsupportedSettingsEffect("effect is not live-reconfigurable")
        for section in ("model", "labels", "clip", "video"):
            if section in desired:
                self.state[section] = copy.deepcopy(desired[section])
        return self.snapshot()

    def restore(self, snapshot) -> None:
        self.restore_calls += 1
        self.state = copy.deepcopy(dict(snapshot))

    def functional_health(self) -> bool:
        return self.healthy

    def verify_model(self, desired) -> dict[str, str]:
        return copy.deepcopy(self.state["model"])

    def verify_labels(self, desired) -> dict[str, object]:
        actual = self.state["labels"]
        return {key: copy.deepcopy(actual[key]) for key in desired}


class _Encoder:
    name = "x264enc"

    def __init__(self) -> None:
        self.bitrate = 1000

    def read_bitrate_kbps(self) -> int:
        return self.bitrate

    def set_bitrate_kbps(self, value: int) -> int:
        self.bitrate = int(value)
        return self.bitrate


class SettingsReconcilerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.config_path = self.root / "agent.env"
        self.original = (
            b"NUVION_DEVICE_PASSWORD=keep-this-secret\n"
            b"NUVION_VIDEO_WIDTH=640\n"
            b"NUVION_VIDEO_HEIGHT=480\n"
            b"NUVION_VIDEO_FPS=30\n"
            b"NUVION_VIDEO_BITRATE_KBPS=1000\n"
        )
        self.config_path.write_bytes(self.original)

    def _reconciler(
        self,
        runtime: _Runtime,
        process_id: str,
        *,
        event_outbox_health_provider=_healthy_event_outbox,
        command_outbox_health_provider=_healthy_command_outbox,
    ) -> SettingsReconciler:
        return SettingsReconciler(
            store=AtomicSettingsStore(self.config_path, self.root / "state"),
            runtime=runtime,
            process_instance_id=process_id,
            event_outbox_health_provider=event_outbox_health_provider,
            command_outbox_health_provider=command_outbox_health_provider,
        )

    def test_immediate_apply_proves_readback_and_signed_payload_superset(self) -> None:
        command = _command(7)
        runtime = _Runtime()

        outcome = self._reconciler(runtime, "process-a").reconcile(command)

        self.assertEqual(outcome.status, "SUCCEEDED")
        self.assertEqual(outcome.reported_state["video"], command.payload["video"])
        self.assertEqual(outcome.reported_state["health"], "FUNCTIONAL_HEALTHY")
        self.assertEqual(
            outcome.reported_state["settingsDigest"],
            canonical_settings_digest(command.payload),
        )
        self.assertEqual(outcome.reported_state["configSchema"], "12")
        self.assertEqual(self.config_path.read_bytes(), self.original)
        active = self.root / "state" / "active.env"
        self.assertIn("NUVION_VIDEO_BITRATE_KBPS=1500", active.read_text())
        self.assertNotIn("NUVION_DEVICE_PASSWORD", active.read_text())

    def test_base_secret_file_inode_mode_and_content_are_preserved(self) -> None:
        self.config_path.chmod(0o640)
        before = self.config_path.stat()

        outcome = self._reconciler(_Runtime(), "process-a").reconcile(_command(6))

        after = self.config_path.stat()
        self.assertEqual(outcome.status, "SUCCEEDED")
        self.assertEqual(after.st_ino, before.st_ino)
        self.assertEqual(after.st_mode & 0o777, 0o640)
        self.assertEqual(self.config_path.read_bytes(), self.original)

    def test_label_arrays_are_encoded_losslessly_for_env_storage(self) -> None:
        labels = ["scratch,edge", "한글 label", "line\\nbreak"]
        encoded = config_env_updates(
            {
                "labels": {
                    "inspection": labels,
                    "anomaly": ["scratch,edge"],
                }
            }
        )["NUVION_ZERO_SHOT_LABELS_B64"]
        padding = "=" * ((4 - len(encoded) % 4) % 4)

        self.assertEqual(
            json.loads(base64.urlsafe_b64decode(encoded + padding).decode()),
            labels,
        )

    def test_unsupported_immediate_effect_restores_runtime_and_lkg(self) -> None:
        command = _command(8)
        runtime = _Runtime()
        before = runtime.snapshot()
        runtime.unsupported = True

        outcome = self._reconciler(runtime, "process-a").reconcile(command)

        self.assertEqual(outcome.status, "FAILED")
        self.assertEqual(outcome.code, "UNSUPPORTED_ACTUAL_EFFECT")
        self.assertEqual(runtime.state, before)
        self.assertEqual(self.config_path.read_bytes(), self.original)
        self.assertEqual((self.root / "state" / "active.env").read_bytes(), b"")

    def test_failed_health_rolls_back_runtime_and_lkg(self) -> None:
        command = _command(9)
        runtime = _Runtime()
        before = runtime.snapshot()
        runtime.healthy = False

        outcome = self._reconciler(runtime, "process-a").reconcile(command)

        self.assertEqual(outcome.status, "ROLLED_BACK")
        self.assertEqual(outcome.code, "FUNCTIONAL_HEALTH_ROLLBACK")
        self.assertEqual(runtime.state, before)
        self.assertEqual(self.config_path.read_bytes(), self.original)
        self.assertEqual((self.root / "state" / "active.env").read_bytes(), b"")

    def test_outbox_health_is_a_fail_closed_functional_success_gate(self) -> None:
        event_health = _healthy_event_outbox()
        event_health["durableSafetyRetained"] = True
        runtime = _Runtime()

        outcome = self._reconciler(
            runtime,
            "process-a",
            event_outbox_health_provider=lambda: event_health,
        ).reconcile(_command(91))

        self.assertEqual(outcome.status, "ROLLED_BACK")
        self.assertEqual(outcome.code, "FUNCTIONAL_HEALTH_ROLLBACK")
        self.assertIn("CRITICAL_SAFETY_RETAINED", outcome.message)
        self.assertEqual(runtime.apply_calls, 1)
        self.assertEqual(runtime.restore_calls, 1)

    def test_functional_health_reasons_require_every_outbox_safety_field(self) -> None:
        event_failures = (
            ({"capacityState": "PRESSURE"}, "EVENT_OUTBOX_UNHEALTHY"),
            ({"unsavedCriticalEvents": 1}, "CRITICAL_EVENT_NOT_DURABLE"),
            ({"safetyStop": True}, "CRITICAL_EVENT_SAFETY_STOP"),
            ({"protocolStop": True}, "EVENT_PROTOCOL_STOP"),
            ({"durableSafetyRetained": True}, "CRITICAL_SAFETY_RETAINED"),
            ({"blockedRows": 1}, "EVENT_OUTBOX_BLOCKED"),
            ({"dlqRows": 1}, "EVENT_OUTBOX_DLQ_PRESENT"),
        )
        command_failures = (
            ({"capacityState": "BACKPRESSURE"}, "COMMAND_OUTBOX_UNHEALTHY"),
            ({"retentionPressure": True}, "COMMAND_OUTBOX_RETENTION_PRESSURE"),
            ({"dlqBlockedRows": 1}, "COMMAND_OUTBOX_BLOCKED"),
            ({"dlqRows": 1}, "COMMAND_OUTBOX_DLQ_PRESENT"),
        )

        for update, reason in event_failures:
            with self.subTest(event_reason=reason):
                health = _healthy_event_outbox()
                health.update(update)
                reconciler = self._reconciler(
                    _Runtime(),
                    f"event-{reason}",
                    event_outbox_health_provider=lambda health=health: health,
                )
                self.assertEqual(reconciler._functional_health_failure_reason(), reason)

        for update, reason in command_failures:
            with self.subTest(command_reason=reason):
                health = _healthy_command_outbox()
                health.update(update)
                reconciler = self._reconciler(
                    _Runtime(),
                    f"command-{reason}",
                    command_outbox_health_provider=lambda health=health: health,
                )
                self.assertEqual(reconciler._functional_health_failure_reason(), reason)

    def test_missing_outbox_health_provider_fails_closed(self) -> None:
        reconciler = SettingsReconciler(
            store=AtomicSettingsStore(self.config_path, self.root / "missing-health"),
            runtime=_Runtime(),
            process_instance_id="process-a",
        )

        self.assertEqual(
            reconciler._functional_health_failure_reason(),
            "EVENT_OUTBOX_HEALTH_UNAVAILABLE",
        )

    def test_config_version_rejects_higher_sequence_before_external_effect_and_restart(
        self,
    ) -> None:
        db_path = self.root / "monotonic.sqlite3"
        inbox = DurableCommandInbox(db_path)
        store = DurableReconcileStore(inbox)
        runtime = _Runtime()
        registry = ReconcilerRegistry()
        registry.register(self._reconciler(runtime, "process-a"))
        coordinator = FleetEffectCoordinator(
            inbox=inbox,
            store=store,
            registry=registry,
            process_instance_id="process-a",
        )
        current = _command(100, config_version=500)
        inbox.accept(current)
        inbox.transition(current.command_id, "IN_PROGRESS")
        inbox.run_transactional_effect(
            current.command_id,
            lambda connection: store.stage_verified(current, connection),
        )
        self.assertEqual(coordinator.run_once().terminal_acks[0].status, "SUCCEEDED")
        self.assertEqual(runtime.apply_calls, 1)

        reopened_inbox = DurableCommandInbox(db_path)
        reopened_store = DurableReconcileStore(reopened_inbox)
        stale = _command(101, config_version=500)
        reopened_inbox.accept(stale)
        reopened_inbox.transition(stale.command_id, "IN_PROGRESS")
        ack, effect_applied = reopened_inbox.run_transactional_effect(
            stale.command_id,
            lambda connection: reopened_store.stage_verified(stale, connection),
        )

        # The short SQLite handler transaction commits the terminal rejection,
        # but no reconcile job is staged, so no external runtime effect exists.
        self.assertTrue(effect_applied)
        self.assertEqual(ack.status, "FAILED")
        self.assertEqual(ack.code, "STALE_CONFIG_VERSION")
        self.assertEqual(ack.reported_state["currentConfigVersion"], 500)
        self.assertIsNone(reopened_store.get_job(stale.command_id))
        self.assertEqual(
            reopened_store.applied_state("CONFIG_APPLY").payload["configVersion"],
            500,
        )
        self.assertTrue(reopened_inbox.accept(stale).duplicate)
        self.assertEqual(reopened_inbox.get(stale.command_id).code, "STALE_CONFIG_VERSION")

    def test_restart_requires_new_process_and_actual_model_digest(self) -> None:
        requested_model = {
            "pointer": "anomalyclip/prod-v2",
            "digest": "sha256:" + "b" * 64,
        }
        command = _command(
            10,
            activation="RESTART",
            sections={"model": requested_model},
        )
        first = self._reconciler(_Runtime(), "process-a").reconcile(command)
        self.assertIsInstance(first, ReconcileDeferred)
        self.assertEqual(first.reported_state["health"], "RESTART_REQUIRED")

        restarted_runtime = _Runtime()
        restarted_runtime.state["model"] = copy.deepcopy(requested_model)
        succeeded = self._reconciler(restarted_runtime, "process-b").reconcile(command)

        self.assertEqual(succeeded.status, "SUCCEEDED")
        self.assertEqual(succeeded.reported_state["model"], requested_model)
        self.assertEqual(succeeded.reported_state["health"], "FUNCTIONAL_HEALTHY")

    def test_restart_wrong_model_digest_stages_lkg_rollback(self) -> None:
        requested_model = {
            "pointer": "anomalyclip/prod-v2",
            "digest": "sha256:" + "b" * 64,
        }
        command = _command(
            11,
            activation="RESTART",
            sections={"model": requested_model},
        )
        self._reconciler(_Runtime(), "process-a").reconcile(command)
        wrong_runtime = _Runtime()
        wrong_runtime.state["model"] = {
            **requested_model,
            "digest": "sha256:" + "c" * 64,
        }

        rollback = self._reconciler(wrong_runtime, "process-b").reconcile(command)

        self.assertIsInstance(rollback, ReconcileDeferred)
        self.assertEqual(rollback.reported_state["health"], "ROLLBACK_RESTART_REQUIRED")
        self.assertEqual(self.config_path.read_bytes(), self.original)

        recovered = self._reconciler(_Runtime(), "process-c").reconcile(command)
        self.assertEqual(recovered.status, "ROLLED_BACK")
        self.assertEqual(recovered.reported_state["health"], "LKG_RESTORED")
        self.assertEqual(
            AtomicSettingsStore(self.config_path, self.root / "state").marker()["phase"],
            "ROLLED_BACK",
        )
        self.assertEqual(
            run_settings_boot_guard(
                {"NUVION_SETTINGS_STATE_DIR": str(self.root / "state")},
                base_config_path=self.config_path,
            ),
            "ROLLED_BACK",
        )

    def test_corrupt_restart_marker_recovers_through_lkg_restart(self) -> None:
        command = _command(11, activation="RESTART")
        first_reconciler = self._reconciler(_Runtime(), "process-a")
        first_reconciler.reconcile(command)
        first_reconciler.store.marker_path.write_text("not-json", encoding="utf-8")

        rollback = self._reconciler(_Runtime(), "process-b").reconcile(command)

        self.assertIsInstance(rollback, ReconcileDeferred)
        self.assertEqual(rollback.reported_state["health"], "ROLLBACK_RESTART_REQUIRED")
        self.assertEqual(self.config_path.read_bytes(), self.original)
        repaired = self._reconciler(_Runtime(), "process-c").reconcile(command)
        self.assertEqual(repaired.status, "ROLLED_BACK")

    def test_waiting_restart_uses_lifecycle_until_succeeded_observation(self) -> None:
        command = _command(12, activation="RESTART")
        db_path = self.root / "commands.sqlite3"
        inbox = DurableCommandInbox(db_path)
        observations = DurableCommandObservationOutbox(inbox)
        store = DurableReconcileStore(inbox, observation_outbox=observations)
        inbox.accept(command)
        inbox.transition(command.command_id, "IN_PROGRESS")
        inbox.run_transactional_effect(
            command.command_id,
            lambda connection: store.stage_verified(command, connection),
        )
        registry = ReconcilerRegistry()
        registry.register(self._reconciler(_Runtime(), "process-a"))
        first = FleetEffectCoordinator(
            inbox=inbox,
            store=store,
            registry=registry,
            process_instance_id="process-a",
            restart_requester=lambda: True,
        ).run_once()
        self.assertEqual(first.processed, 1)
        self.assertEqual(store.get_job(command.command_id).phase, JOB_PHASE_WAITING_RESTART)
        self.assertEqual(inbox.get(command.command_id).status, "IN_PROGRESS")
        self.assertEqual(observations.pending(), [])

        reopened_inbox = DurableCommandInbox(db_path)
        reopened_observations = DurableCommandObservationOutbox(reopened_inbox)
        reopened_store = DurableReconcileStore(
            reopened_inbox,
            observation_outbox=reopened_observations,
        )
        restarted = _Runtime()
        restarted.state["video"] = copy.deepcopy(command.payload["video"])
        restarted_registry = ReconcilerRegistry()
        restarted_registry.register(self._reconciler(restarted, "process-b"))
        second = FleetEffectCoordinator(
            inbox=reopened_inbox,
            store=reopened_store,
            registry=restarted_registry,
            process_instance_id="process-b",
            restart_requester=lambda: True,
        ).run_once()

        self.assertEqual(second.processed, 1)
        self.assertEqual(second.terminal_acks[0].status, "SUCCEEDED")
        pending = reopened_observations.pending()
        self.assertEqual([item.revision for item in pending], [1])
        self.assertEqual(pending[-1].reported_state["health"], "FUNCTIONAL_HEALTHY")

    def test_manual_observation_is_ignored_until_command_succeeds(self) -> None:
        command = _command(121)
        inbox = DurableCommandInbox(self.root / "observation-guard.sqlite3")
        observations = DurableCommandObservationOutbox(inbox)
        store = DurableReconcileStore(inbox, observation_outbox=observations)
        inbox.accept(command)
        inbox.transition(command.command_id, "IN_PROGRESS")

        store.observe_state(command=command, reported_state={"health": "EARLY"})
        self.assertEqual(observations.pending(), [])

        inbox.transition(
            command.command_id,
            "SUCCEEDED",
            reported_state={"health": "FUNCTIONAL_HEALTHY"},
        )
        store.observe_state(
            command=command,
            reported_state={"health": "FUNCTIONAL_HEALTHY"},
        )
        self.assertEqual(len(observations.pending()), 1)

    def test_restart_without_supervisor_support_fails_before_settings_mutation(self) -> None:
        command = _command(13, activation="RESTART")
        inbox = DurableCommandInbox(self.root / "unsupported-restart.sqlite3")
        store = DurableReconcileStore(inbox)
        inbox.accept(command)
        inbox.transition(command.command_id, "IN_PROGRESS")
        inbox.run_transactional_effect(
            command.command_id,
            lambda connection: store.stage_verified(command, connection),
        )
        registry = ReconcilerRegistry()
        reconciler = self._reconciler(_Runtime(), "process-a")
        registry.register(reconciler)

        result = FleetEffectCoordinator(
            inbox=inbox,
            store=store,
            registry=registry,
            process_instance_id="process-a",
        ).run_once()

        self.assertEqual(result.terminal_acks[0].code, "RESTART_UNSUPPORTED")
        self.assertFalse(reconciler.store.active_path.exists())
        self.assertEqual(self.config_path.read_bytes(), self.original)

    def test_restart_callback_runs_only_after_waiting_restart_is_durable(self) -> None:
        command = _command(14, activation="RESTART")
        inbox = DurableCommandInbox(self.root / "restart-order.sqlite3")
        store = DurableReconcileStore(inbox)
        inbox.accept(command)
        inbox.transition(command.command_id, "IN_PROGRESS")
        inbox.run_transactional_effect(
            command.command_id,
            lambda connection: store.stage_verified(command, connection),
        )
        registry = ReconcilerRegistry()
        registry.register(self._reconciler(_Runtime(), "process-a"))
        observed_phases: list[str] = []

        def request_restart() -> bool:
            observed_phases.append(store.get_job(command.command_id).phase)
            return True

        FleetEffectCoordinator(
            inbox=inbox,
            store=store,
            registry=registry,
            process_instance_id="process-a",
            restart_requester=request_restart,
        ).run_once()

        self.assertEqual(observed_phases, [JOB_PHASE_WAITING_RESTART])

    def test_waiting_restart_is_terminally_superseded_without_losing_lkg(self) -> None:
        first = _command(20, activation="RESTART")
        second = _command(21, activation="IMMEDIATE")
        inbox = DurableCommandInbox(self.root / "supersession.sqlite3")
        store = DurableReconcileStore(inbox)

        def stage(command: VerifiedFleetCommand) -> None:
            inbox.accept(command)
            inbox.transition(command.command_id, "IN_PROGRESS")
            inbox.run_transactional_effect(
                command.command_id,
                lambda connection: store.stage_verified(command, connection),
            )

        stage(first)
        registry = ReconcilerRegistry()
        registry.register(self._reconciler(_Runtime(), "process-a"))
        FleetEffectCoordinator(
            inbox=inbox,
            store=store,
            registry=registry,
            process_instance_id="process-a",
            restart_requester=lambda: True,
        ).run_once()
        stage(second)

        self.assertEqual(inbox.get(first.command_id).status, "FAILED")
        self.assertEqual(inbox.get(first.command_id).code, "SUPERSEDED")
        self.assertEqual(store.get_job(first.command_id).phase, JOB_PHASE_SUPERSEDED)
        self.assertEqual(store.get_job(second.command_id).phase, JOB_PHASE_PENDING)

        second_runtime = _Runtime()
        registry.register(self._reconciler(second_runtime, "process-a"))
        second_runtime.healthy = False
        outcome = FleetEffectCoordinator(
            inbox=inbox,
            store=store,
            registry=registry,
            process_instance_id="process-a",
        ).run_once()
        self.assertEqual(outcome.terminal_acks[0].status, "ROLLED_BACK")
        self.assertEqual(self.config_path.read_bytes(), self.original)

    def test_stream_policy_owns_shared_encoder_over_config_video(self) -> None:
        inbox = DurableCommandInbox(self.root / "encoder-ownership.sqlite3")
        store = DurableReconcileStore(inbox)
        registry = ReconcilerRegistry()
        encoder = _Encoder()
        registry.register(_stream_reconciler(encoder))
        settings = self._reconciler(_Runtime(), "process-a")
        registry.register(settings)
        coordinator = FleetEffectCoordinator(
            inbox=inbox,
            store=store,
            registry=registry,
            owner="shared-encoder-worker",
        )

        def stage(command: VerifiedFleetCommand) -> None:
            inbox.accept(command)
            inbox.transition(command.command_id, "IN_PROGRESS")
            inbox.run_transactional_effect(
                command.command_id,
                lambda connection: store.stage_verified(command, connection),
            )

        stream = _stream_command(1, 1400)
        stage(stream)
        self.assertEqual(coordinator.run_once().terminal_acks[0].status, "SUCCEEDED")
        config = _command(2)
        stage(config)

        rejected = coordinator.run_once().terminal_acks[0]

        self.assertEqual(rejected.status, "FAILED")
        self.assertEqual(rejected.code, "ENCODER_OWNED_BY_STREAM_POLICY")
        self.assertEqual(encoder.bitrate, 1400)
        self.assertFalse(settings.store.active_path.exists())

    def test_model_identity_hashes_actual_artifact_and_rejects_wrong_digest(self) -> None:
        model_dir = self.root / "model"
        metadata = model_dir / "metadata"
        metadata.mkdir(parents=True)
        manifest = metadata / "gcs_manifest.json"
        manifest.write_bytes(b"authenticated manifest")
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        (metadata / "server_presign_response.json").write_text(
            json.dumps({"pointer": "anomalyclip/prod-v2"}),
            encoding="utf-8",
        )
        (metadata / "downloaded_from_server.json").write_text(
            json.dumps(
                [
                    {
                        "key": "manifest",
                        "dst": str(manifest),
                        "sha256": digest,
                    }
                ]
            ),
            encoding="utf-8",
        )

        self.assertEqual(
            verify_model_artifact_identity(
                model_dir,
                expected_pointer="anomalyclip/prod-v2",
                expected_digest="sha256:" + digest,
            ),
            {
                "pointer": "anomalyclip/prod-v2",
                "digest": "sha256:" + digest,
            },
        )
        self.assertIsNone(
            verify_model_artifact_identity(
                model_dir,
                expected_pointer="anomalyclip/prod-v2",
                expected_digest="sha256:" + "f" * 64,
            )
        )
        manifest.write_bytes(b"tampered")
        self.assertIsNone(
            verify_model_artifact_identity(
                model_dir,
                expected_pointer="anomalyclip/prod-v2",
                expected_digest="sha256:" + digest,
            )
        )

    def test_crash_boundaries_recover_without_overwriting_lkg(self) -> None:
        initial = _command(30)
        initial_runtime = _Runtime()
        initial_reconciler = self._reconciler(initial_runtime, "initial-process")
        self.assertEqual(initial_reconciler.reconcile(initial).status, "SUCCEEDED")
        active_before = initial_reconciler.store.active_path.read_bytes()
        replacement = _command(
            31,
            activation="RESTART",
            sections={
                "model": {
                    "pointer": "anomalyclip/prod-v31",
                    "digest": "sha256:" + "d" * 64,
                }
            },
        )

        for boundary, candidate_should_recover in (
            ("LKG_FSYNCED", False),
            ("CANDIDATE_FSYNCED", False),
            ("PREPARED_FSYNCED", True),
            ("ACTIVE_SWITCHED", True),
            ("ACTIVATED_FSYNCED", True),
        ):
            with self.subTest(boundary=boundary):
                state_dir = self.root / f"crash-{boundary.lower()}"
                seed = AtomicSettingsStore(self.config_path, state_dir)
                seed._atomic_write(seed.active_path, active_before)

                def crash(current: str, target: str = boundary) -> None:
                    if current == target:
                        raise RuntimeError("simulated power loss")

                crashing = AtomicSettingsStore(
                    self.config_path,
                    state_dir,
                    fault_hook=crash,
                )
                with self.assertRaises(RuntimeError):
                    crashing.stage_and_activate(
                        command=replacement,
                        process_instance_id="crashing-process",
                        settings_digest=canonical_settings_digest(
                            replacement.payload
                        ),
                    )

                recovered = AtomicSettingsStore(self.config_path, state_dir)
                marker = recovered.recover_prepared()
                self.assertEqual(recovered.lkg_path.read_bytes(), active_before)
                if candidate_should_recover:
                    self.assertEqual(marker["phase"], "ACTIVATED")
                    self.assertIn(
                        "NUVION_MODEL_POINTER=anomalyclip/prod-v31",
                        recovered.active_path.read_text(),
                    )
                else:
                    self.assertEqual(recovered.active_path.read_bytes(), active_before)

    def test_successor_patch_does_not_inherit_uncommitted_predecessor(self) -> None:
        state_dir = self.root / "partial-leakage"
        store = AtomicSettingsStore(self.config_path, state_dir)
        predecessor = _command(
            50,
            activation="RESTART",
            sections={
                "model": {
                    "pointer": "anomalyclip/uncommitted",
                    "digest": "sha256:" + "e" * 64,
                }
            },
        )
        successor = _command(51)
        store.stage_and_activate(
            command=predecessor,
            process_instance_id="process-a",
            settings_digest=canonical_settings_digest(predecessor.payload),
        )

        store.stage_and_activate(
            command=successor,
            process_instance_id="process-b",
            settings_digest=canonical_settings_digest(successor.payload),
        )

        active = store.active_path.read_text()
        self.assertNotIn("NUVION_MODEL_POINTER", active)
        self.assertNotIn("NUVION_MODEL_DIGEST", active)
        self.assertIn("NUVION_VIDEO_BITRATE_KBPS=1500", active)

    def test_boot_guard_restores_lkg_before_second_candidate_boot(self) -> None:
        command = _command(40, activation="RESTART")
        reconciler = self._reconciler(_Runtime(), "staging-process")
        deferred = reconciler.reconcile(command)
        self.assertIsInstance(deferred, ReconcileDeferred)
        environment = {"NUVION_SETTINGS_STATE_DIR": str(self.root / "state")}

        self.assertEqual(
            run_settings_boot_guard(
                environment,
                base_config_path=self.config_path,
            ),
            "CANDIDATE_BOOT_ATTEMPT",
        )
        self.assertEqual(
            run_settings_boot_guard(
                environment,
                base_config_path=self.config_path,
            ),
            "LKG_RESTORED",
        )
        store = AtomicSettingsStore(self.config_path, self.root / "state")
        self.assertTrue(store.lkg_is_active())
        self.assertEqual(store.marker()["phase"], "ROLLBACK_STAGED")

        with self.assertRaises(SettingsBootGuardError):
            run_settings_boot_guard(
                environment,
                base_config_path=self.config_path,
            )

    def test_boot_guard_never_loads_superseded_uncommitted_candidate(self) -> None:
        inbox_path = self.root / "boot-supersession.sqlite3"
        inbox = DurableCommandInbox(inbox_path)
        store = DurableReconcileStore(inbox)
        predecessor = _command(60, activation="RESTART")
        successor = _command(61)

        def stage(command: VerifiedFleetCommand) -> None:
            inbox.accept(command)
            inbox.transition(command.command_id, "IN_PROGRESS")
            inbox.run_transactional_effect(
                command.command_id,
                lambda connection: store.stage_verified(command, connection),
            )

        stage(predecessor)
        registry = ReconcilerRegistry()
        reconciler = self._reconciler(_Runtime(), "process-a")
        registry.register(reconciler)
        FleetEffectCoordinator(
            inbox=inbox,
            store=store,
            registry=registry,
            process_instance_id="process-a",
            restart_requester=lambda: True,
        ).run_once()
        stage(successor)
        environment = {
            "NUVION_SETTINGS_STATE_DIR": str(self.root / "state"),
            "NUVION_COMMAND_INBOX_PATH": str(inbox_path),
        }

        result = run_settings_boot_guard(
            environment,
            base_config_path=self.config_path,
        )

        self.assertEqual(result, "SUPERSEDED_LKG_RESTORED")
        self.assertTrue(reconciler.store.lkg_is_active())
        self.assertEqual(
            reconciler.store.marker()["recoveryReason"],
            "SUPERSEDED_BEFORE_BOOT",
        )


if __name__ == "__main__":
    unittest.main()
