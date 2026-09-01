from __future__ import annotations

import base64
import hashlib
import json
import tempfile
import unittest
import uuid
from pathlib import Path

from nuvion_app.inference.command_inbox import (
    CommandEffectOutcome,
    DurableCommandInbox,
)
from nuvion_app.inference.effect_reconciler import (
    FleetEffectCoordinator,
    ReconcileDeferred,
    ReconcilerRegistry,
)
from nuvion_app.inference.fleet_command import VerifiedFleetCommand
from nuvion_app.inference.reconcile_store import (
    JOB_PHASE_APPLYING,
    JOB_PHASE_CLAIMED,
    JOB_PHASE_PENDING,
    JOB_PHASE_SUPERSEDED,
    JOB_PHASE_VERIFYING,
    DurableReconcileStore,
    EffectFenceStale,
)
from nuvion_app.inference.stream_policy import StreamPolicyReconciler


def _stream_command(sequence: int, target: int = 1000) -> VerifiedFleetCommand:
    payload = {
        "policyVersion": sequence,
        "mode": "FIXED",
        "targetBitrateKbps": target,
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    command_id = str(uuid.uuid4())
    return VerifiedFleetCommand(
        command_id=command_id,
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


class _WritingReconciler:
    command_type = "STREAM_POLICY"
    capability = "command.stream.policy"

    def __init__(self, inbox: DurableCommandInbox) -> None:
        self.inbox = inbox
        self.calls: list[str] = []
        self.restore_calls: list[str] = []

    def reconcile(self, command: VerifiedFleetCommand) -> CommandEffectOutcome:
        # This independent write proves the coordinator did not retain its claim
        # transaction across the external effect callback.
        with self.inbox.transaction(immediate=True) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS external_effect_probe(value TEXT)"
            )
            connection.execute(
                "INSERT INTO external_effect_probe(value) VALUES (?)",
                (command.command_id,),
            )
        self.calls.append(command.command_id)
        return CommandEffectOutcome.succeeded(
            {
                "policyVersion": command.payload["policyVersion"],
                "appliedBitrateKbps": command.payload["targetBitrateKbps"],
            }
        )

    def restore_applied(
        self,
        command: VerifiedFleetCommand,
        _persisted_state: dict[str, object],
    ) -> dict[str, object]:
        self.restore_calls.append(command.command_id)
        outcome = self.reconcile(command)
        return dict(outcome.reported_state or {})


class _RetryingReconciler:
    command_type = "STREAM_POLICY"
    capability = "command.stream.policy"

    def __init__(self) -> None:
        self.calls = 0

    def reconcile(
        self, command: VerifiedFleetCommand
    ) -> CommandEffectOutcome | ReconcileDeferred:
        self.calls += 1
        if self.calls == 1:
            return ReconcileDeferred(
                reported_state={**command.payload, "health": "HELPER_UNAVAILABLE"},
                checkpoint={
                    "nextAction": "RETRY_EFFECT",
                    "restartRequired": False,
                },
            )
        return CommandEffectOutcome.succeeded(
            {**command.payload, "health": "STREAM_CONTINUOUS"}
        )


class _Encoder:
    name = "x264enc"

    def __init__(self) -> None:
        self.bitrate = 1000
        self.fail_target: int | None = None

    def read_bitrate_kbps(self) -> int:
        return self.bitrate

    def set_bitrate_kbps(self, bitrate_kbps: int) -> int:
        self.bitrate = int(bitrate_kbps)
        if self.fail_target == self.bitrate:
            raise RuntimeError("simulated encoder readback failure")
        return self.bitrate


class DurableReconcileStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.inbox = DurableCommandInbox(Path(self.tempdir.name) / "commands.sqlite3")
        self.store = DurableReconcileStore(self.inbox)

    def _stage(self, command: VerifiedFleetCommand) -> None:
        self.inbox.accept(command)
        self.inbox.transition(command.command_id, "IN_PROGRESS")
        ack, applied = self.inbox.run_transactional_effect(
            command.command_id,
            lambda connection: self.store.stage_verified(command, connection),
        )
        self.assertEqual(ack.status, "IN_PROGRESS")
        self.assertTrue(applied)

    def test_new_desired_state_terminally_supersedes_older_job(self) -> None:
        first = _stream_command(1, 900)
        second = _stream_command(2, 1500)

        self._stage(first)
        self._stage(second)

        first_record = self.inbox.get(first.command_id)
        self.assertEqual(first_record.status, "FAILED")
        self.assertEqual(first_record.code, "SUPERSEDED")
        for key, expected in first.payload.items():
            self.assertEqual(first_record.reported_state[key], expected)
        self.assertEqual(first_record.reported_state["health"], "SUPERSEDED")
        self.assertEqual(
            first_record.reported_state["supersededByCommandId"],
            second.command_id,
        )
        self.assertEqual(first_record.reported_state["supersededBySequence"], 2)
        self.assertEqual(self.store.get_job(first.command_id).phase, JOB_PHASE_SUPERSEDED)
        self.assertEqual(self.store.get_job(second.command_id).phase, JOB_PHASE_PENDING)
        self.assertEqual(
            [ack.status for ack in self.inbox.ack_transitions(first.command_id)],
            ["RECEIVED", "IN_PROGRESS", "FAILED"],
        )
        self.assertEqual(
            self.store.history(first.command_id)[-1]["toPhase"],
            JOB_PHASE_SUPERSEDED,
        )

    def test_coordinator_applies_outside_transaction_and_commits_applied_state(self) -> None:
        command = _stream_command(1, 1800)
        self._stage(command)
        reconciler = _WritingReconciler(self.inbox)
        registry = ReconcilerRegistry()
        registry.register(reconciler)
        coordinator = FleetEffectCoordinator(
            inbox=self.inbox,
            store=self.store,
            registry=registry,
            owner="worker-1",
            max_jobs_per_run=1,
        )

        result = coordinator.run_once()

        self.assertEqual(result.processed, 1)
        self.assertEqual(result.terminal_acks[0].status, "SUCCEEDED")
        self.assertEqual(reconciler.calls, [command.command_id])
        self.assertEqual(self.inbox.get(command.command_id).status, "SUCCEEDED")
        applied = self.store.applied_state("STREAM_POLICY")
        self.assertEqual(applied.command_id, command.command_id)
        self.assertEqual(applied.reported_state["appliedBitrateKbps"], 1800)
        self.assertEqual(
            [item["toPhase"] for item in self.store.history(command.command_id)],
            ["PENDING", "CLAIMED", "APPLYING", "SUCCEEDED"],
        )

    def test_expired_lease_is_reclaimed_with_checkpoint_history(self) -> None:
        ticks = {"now": 10.0}
        inbox = DurableCommandInbox(Path(self.tempdir.name) / "lease.sqlite3")
        store = DurableReconcileStore(
            inbox,
            monotonic_clock=lambda: ticks["now"],
            wall_clock=lambda: f"2026-09-01T00:00:{int(ticks['now']):02d}Z",
        )
        command = _stream_command(1)
        inbox.accept(command)
        inbox.transition(command.command_id, "IN_PROGRESS")
        inbox.run_transactional_effect(
            command.command_id,
            lambda connection: store.stage_verified(command, connection),
        )

        first = store.claim_next(owner="worker-a", lease_seconds=5)
        self.assertEqual(first.phase, JOB_PHASE_CLAIMED)
        applying = store.checkpoint(
            command.command_id,
            owner="worker-a",
            expected_phase=JOB_PHASE_CLAIMED,
            next_phase=JOB_PHASE_APPLYING,
            lease_seconds=5,
        )
        self.assertEqual(applying.phase, JOB_PHASE_APPLYING)

        ticks["now"] = 16.0
        reclaimed = store.claim_next(owner="worker-b", lease_seconds=5)
        self.assertEqual(reclaimed.phase, JOB_PHASE_CLAIMED)
        self.assertEqual(reclaimed.lease_owner, "worker-b")
        self.assertEqual(reclaimed.attempts, 2)

    def test_effect_retry_uses_backoff_without_requesting_restart(self) -> None:
        ticks = {"now": 10.0}
        inbox = DurableCommandInbox(Path(self.tempdir.name) / "retry.sqlite3")
        store = DurableReconcileStore(
            inbox,
            monotonic_clock=lambda: ticks["now"],
        )
        command = _stream_command(1)
        inbox.accept(command)
        inbox.transition(command.command_id, "IN_PROGRESS")
        inbox.run_transactional_effect(
            command.command_id,
            lambda connection: store.stage_verified(command, connection),
        )
        reconciler = _RetryingReconciler()
        registry = ReconcilerRegistry()
        registry.register(reconciler)
        restart_calls: list[bool] = []
        coordinator = FleetEffectCoordinator(
            inbox=inbox,
            store=store,
            registry=registry,
            owner="retry-worker",
            restart_requester=lambda: restart_calls.append(True) or True,
        )

        first = coordinator.run_once()

        self.assertEqual(first.processed, 1)
        self.assertEqual(first.terminal_acks, ())
        self.assertEqual(store.get_job(command.command_id).phase, JOB_PHASE_VERIFYING)
        self.assertEqual(restart_calls, [])
        self.assertEqual(coordinator.run_once().processed, 0)

        ticks["now"] = 12.1
        completed = coordinator.run_once()
        self.assertEqual(completed.terminal_acks[0].status, "SUCCEEDED")
        self.assertEqual(reconciler.calls, 2)
        self.assertEqual(restart_calls, [])

    def test_expired_stale_fence_cannot_mutate_encoder(self) -> None:
        ticks = {"now": 10.0}
        inbox = DurableCommandInbox(Path(self.tempdir.name) / "stale-fence.sqlite3")
        store = DurableReconcileStore(inbox, monotonic_clock=lambda: ticks["now"])
        command = _stream_command(1, 1900)
        inbox.accept(command)
        inbox.transition(command.command_id, "IN_PROGRESS")
        inbox.run_transactional_effect(
            command.command_id,
            lambda connection: store.stage_verified(command, connection),
        )
        store.claim_next(owner="stale-worker", lease_seconds=2)
        _job, fence = store.begin_effect(
            command,
            owner="stale-worker",
            lease_seconds=2,
        )
        encoder = _Encoder()
        reconciler = StreamPolicyReconciler(encoder)
        reconciler.set_effect_fence(fence.assert_current)

        ticks["now"] = 13.0
        with self.assertRaises(EffectFenceStale):
            reconciler.reconcile(command)

        self.assertEqual(encoder.bitrate, 1000)

    def test_newer_desired_invalidates_stale_fence_before_encoder_callback(self) -> None:
        first = _stream_command(1, 900)
        second = _stream_command(2, 1800)
        self._stage(first)
        self.store.claim_next(owner="old-worker", lease_seconds=30)
        _job, fence = self.store.begin_effect(
            first,
            owner="old-worker",
            lease_seconds=30,
        )
        self._stage(second)
        encoder = _Encoder()
        reconciler = StreamPolicyReconciler(encoder)
        reconciler.set_effect_fence(fence.assert_current)

        with self.assertRaises(EffectFenceStale):
            reconciler.reconcile(first)

        self.assertEqual(encoder.bitrate, 1000)

    def test_terminal_ack_contains_signed_fixed_policy_and_runtime_readback(self) -> None:
        command = _stream_command(1, 1600)
        self._stage(command)
        registry = ReconcilerRegistry()
        registry.register(StreamPolicyReconciler(_Encoder()))

        result = FleetEffectCoordinator(
            inbox=self.inbox,
            store=self.store,
            registry=registry,
            owner="stream-worker",
        ).run_once()

        ack = result.terminal_acks[0]
        self.assertEqual(ack.status, "SUCCEEDED")
        for key, expected in command.payload.items():
            self.assertEqual(ack.reported_state[key], expected)
        self.assertEqual(ack.reported_state["requestedBitrateKbps"], 1600)
        self.assertEqual(ack.reported_state["appliedBitrateKbps"], 1600)
        self.assertEqual(ack.reported_state["encoder"], "x264enc")
        self.assertEqual(ack.reported_state["health"], "STREAM_CONTINUOUS")

    def test_restart_restores_long_lived_controller_from_applied_state(self) -> None:
        command = _stream_command(1, 1700)
        self._stage(command)
        initial = _WritingReconciler(self.inbox)
        initial_registry = ReconcilerRegistry()
        initial_registry.register(initial)
        FleetEffectCoordinator(
            inbox=self.inbox,
            store=self.store,
            registry=initial_registry,
            owner="worker-before-restart",
        ).run_once()

        restarted = _WritingReconciler(self.inbox)
        restarted_registry = ReconcilerRegistry()
        restarted_registry.register(restarted)
        result = FleetEffectCoordinator(
            inbox=self.inbox,
            store=self.store,
            registry=restarted_registry,
            owner="worker-after-restart",
        ).run_once()

        self.assertEqual(result.processed, 0)
        self.assertEqual(restarted.restore_calls, [command.command_id])
        self.assertEqual(restarted.calls, [command.command_id])
        self.assertEqual(self.inbox.get(command.command_id).status, "SUCCEEDED")

    def test_failed_replacement_restores_previous_applied_policy_next_run(self) -> None:
        encoder = _Encoder()
        registry = ReconcilerRegistry()
        registry.register(StreamPolicyReconciler(encoder))
        coordinator = FleetEffectCoordinator(
            inbox=self.inbox,
            store=self.store,
            registry=registry,
            owner="stream-worker",
        )
        first = _stream_command(1, 1200)
        self._stage(first)
        coordinator.run_once()
        self.assertEqual(encoder.bitrate, 1200)

        second = _stream_command(2, 1800)
        self._stage(second)
        encoder.fail_target = 1800
        failed = coordinator.run_once()
        self.assertEqual(failed.terminal_acks[0].status, "FAILED")
        self.assertEqual(encoder.bitrate, 1800)

        encoder.fail_target = None
        restored = coordinator.run_once()
        self.assertEqual(restored.processed, 0)
        self.assertEqual(encoder.bitrate, 1200)
        self.assertEqual(
            self.store.applied_state("STREAM_POLICY").command_id,
            first.command_id,
        )


if __name__ == "__main__":
    unittest.main()
