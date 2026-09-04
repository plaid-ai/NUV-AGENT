from __future__ import annotations

import base64
import hashlib
import json
import tempfile
import unittest
import uuid
from pathlib import Path

from nuvion_app.inference.command_inbox import CommandEffectOutcome, DurableCommandInbox
from nuvion_app.inference.command_observation import (
    COMMAND_OBSERVED_DESTINATION,
    CommandObservationError,
    DurableCommandObservationOutbox,
    build_command_observation_payload,
)
from nuvion_app.inference.command_runtime import FleetCommandRuntime
from nuvion_app.inference.effect_reconciler import (
    FleetEffectCoordinator,
    ReconcilerRegistry,
)
from nuvion_app.inference.fleet_command import VerifiedFleetCommand
from nuvion_app.inference.reconcile_store import DurableReconcileStore
from nuvion_app.inference.stream_policy import (
    StreamPolicyReconciler,
    StreamRuntimeEvidence,
)


def _stream_reconciler(
    encoder: object,
    *,
    clock_ms=None,
) -> StreamPolicyReconciler:
    return StreamPolicyReconciler(
        encoder,
        clock_ms=clock_ms,
        runtime_evidence=lambda: StreamRuntimeEvidence(True, 0.0),
        health_clock=lambda: 0.0,
    )


def _command(sequence: int = 1) -> VerifiedFleetCommand:
    payload = {
        "policyVersion": sequence,
        "mode": "FIXED",
        "targetBitrateKbps": 1200,
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
        compact_jws="header.claims.signature",
    )


class _HttpClient:
    async def pull_after(self, _after_sequence: int, _limit: int):
        raise AssertionError("not used")


class _Encoder:
    name = "x264enc"

    def __init__(self) -> None:
        self.bitrate = 1000
        self.set_calls: list[int] = []

    def read_bitrate_kbps(self) -> int:
        return self.bitrate

    def set_bitrate_kbps(self, bitrate_kbps: int) -> int:
        self.bitrate = int(bitrate_kbps)
        self.set_calls.append(self.bitrate)
        return self.bitrate


class CommandObservationOutboxTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.inbox = DurableCommandInbox(Path(temporary.name) / "commands.sqlite3")
        self.command = _command()
        self.inbox.accept(self.command)
        self.retry_ticks = {"now": 0.0}
        ids = iter(
            (
                "a1974c8c-c916-4990-9956-15e299c44a78",
                "ca3ccf86-253c-4802-9ac5-a1731d5c16a4",
                "df54376f-51f9-4864-ac6d-392626364f2e",
            )
        )
        self.outbox = DurableCommandObservationOutbox(
            self.inbox,
            clock=lambda: "2026-09-01T00:00:00Z",
            id_factory=lambda: next(ids),
            retry_clock=lambda: self.retry_ticks["now"],
        )

    def test_revision_is_monotonic_and_identical_latest_state_is_idempotent(self) -> None:
        state = {**self.command.payload, "appliedBitrateKbps": 1200}
        first = self.outbox.enqueue(
            command_id=self.command.command_id,
            sequence=1,
            command_type="STREAM_POLICY",
            reported_state=state,
        )
        duplicate = self.outbox.enqueue(
            command_id=self.command.command_id,
            sequence=1,
            command_type="STREAM_POLICY",
            reported_state=state,
        )
        changed = self.outbox.enqueue(
            command_id=self.command.command_id,
            sequence=1,
            command_type="STREAM_POLICY",
            reported_state={**state, "appliedBitrateKbps": 900},
        )

        self.assertEqual(duplicate.observation_id, first.observation_id)
        self.assertEqual(duplicate.revision, 1)
        self.assertEqual(changed.revision, 2)
        payload = build_command_observation_payload(changed)
        self.assertEqual(
            set(payload),
            {
                "observationId",
                "commandId",
                "revision",
                "observedAt",
                "reportedState",
            },
        )
        self.assertNotIn("sequence", payload)
        self.assertNotIn("commandType", payload)

    def test_retry_reuses_observation_id_until_ack_and_survives_reopen(self) -> None:
        self.inbox.transition(self.command.command_id, "IN_PROGRESS")
        self.inbox.transition(self.command.command_id, "SUCCEEDED")
        observation = self.outbox.enqueue(
            command_id=self.command.command_id,
            sequence=1,
            command_type="STREAM_POLICY",
            reported_state={**self.command.payload, "appliedBitrateKbps": 1200},
        )
        attempts: list[dict[str, object]] = []
        outcomes = iter((False, True))
        runtime = FleetCommandRuntime(
            inbox=self.inbox,
            processor=object(),
            http_client=_HttpClient(),
            ack_sender=lambda destination, payload: (
                attempts.append({"destination": destination, **payload})
                or next(outcomes)
            ),
            observation_outbox=self.outbox,
        )

        self.assertEqual(runtime.replay_observations(), 0)
        self.retry_ticks["now"] = 1.0
        reopened = DurableCommandObservationOutbox(
            self.inbox,
            retry_clock=lambda: self.retry_ticks["now"],
        )
        runtime.observation_outbox = reopened
        self.assertEqual(runtime.replay_observations(), 1)
        self.assertEqual(
            [item["observationId"] for item in attempts],
            [observation.observation_id, observation.observation_id],
        )
        self.assertTrue(
            all(item["destination"] == COMMAND_OBSERVED_DESTINATION for item in attempts)
        )

        body = json.dumps(
            {
                "observationId": observation.observation_id,
                "commandId": self.command.command_id,
                "revision": 1,
                "status": "ACCEPTED",
                "processedAt": "2026-09-01T00:00:02Z",
                "retryable": None,
                "code": None,
                "reason": None,
            }
        )
        ack, removed = runtime.acknowledge_observation(body)
        self.assertEqual(ack.status, "ACCEPTED")
        self.assertTrue(removed)
        self.assertEqual(reopened.pending(), [])
        self.assertFalse(runtime.acknowledge_observation(body)[1])

    def test_non_succeeded_legacy_observation_is_discarded_without_send(self) -> None:
        self.inbox.transition(self.command.command_id, "IN_PROGRESS")
        observation = self.outbox.enqueue(
            command_id=self.command.command_id,
            sequence=1,
            command_type="STREAM_POLICY",
            reported_state={"updatePhase": "ROLLBACK_FAILED"},
        )
        self.inbox.transition(
            self.command.command_id,
            "FAILED",
            code="EXPIRED",
            reported_state={"updatePhase": "ROLLBACK_FAILED"},
        )
        attempts: list[dict[str, object]] = []
        runtime = FleetCommandRuntime(
            inbox=self.inbox,
            processor=object(),
            http_client=_HttpClient(),
            ack_sender=lambda _destination, payload: not attempts.append(payload),
            observation_outbox=self.outbox,
        )

        self.assertEqual(runtime.replay_observations(), 0)
        self.assertEqual(attempts, [])
        self.assertIsNone(self.outbox.get(observation.observation_id))

    def test_ack_identity_collision_cannot_delete_pending_observation(self) -> None:
        observation = self.outbox.enqueue(
            command_id=self.command.command_id,
            sequence=1,
            command_type="STREAM_POLICY",
            reported_state={**self.command.payload, "health": "STREAM_CONTINUOUS"},
        )
        body = json.dumps(
            {
                "observationId": observation.observation_id,
                "commandId": str(uuid.uuid4()),
                "revision": 1,
                "status": "DUPLICATE",
                "processedAt": "2026-09-01T00:00:02Z",
            }
        )
        with self.assertRaises(CommandObservationError) as raised:
            self.outbox.acknowledge_body(body)
        self.assertEqual(raised.exception.code, "OBSERVATION_ACK_COLLISION")
        self.assertEqual(len(self.outbox.pending()), 1)

    def test_retryable_reject_stays_pending_and_permanent_reject_moves_to_dlq(self) -> None:
        observation = self.outbox.enqueue(
            command_id=self.command.command_id,
            sequence=1,
            command_type="STREAM_POLICY",
            reported_state={**self.command.payload, "health": "STREAM_CONTINUOUS"},
        )

        def rejection(retryable: bool) -> str:
            return json.dumps(
                {
                    "observationId": observation.observation_id,
                    "commandId": self.command.command_id,
                    "revision": 1,
                    "status": "REJECTED",
                    "processedAt": "2026-09-01T00:00:02Z",
                    "retryable": retryable,
                    "code": "OBSERVATION_BUSY" if retryable else "INVALID_STATE",
                    "reason": "try again" if retryable else "permanent schema error",
                }
            )

        unspecified = json.loads(rejection(True))
        unspecified.pop("retryable")
        self.assertFalse(
            self.outbox.acknowledge_body(json.dumps(unspecified))[1]
        )
        self.assertEqual(len(self.outbox.pending()), 0)

        self.assertFalse(self.outbox.acknowledge_body(rejection(True))[1])
        self.assertEqual(len(self.outbox.pending()), 0)
        self.retry_ticks["now"] = 1.0
        self.assertEqual(len(self.outbox.pending()), 1)
        self.assertIn("try again", self.outbox.get(observation.observation_id).last_error)

        self.assertTrue(self.outbox.acknowledge_body(rejection(False))[1])
        self.assertEqual(self.outbox.pending(), [])
        dead_letters = self.outbox.dead_letters()
        self.assertEqual(len(dead_letters), 1)
        self.assertEqual(dead_letters[0]["observationId"], observation.observation_id)
        self.assertEqual(dead_letters[0]["code"], "INVALID_STATE")

    def test_stream_adaptive_changes_emit_monotonic_desired_superset_only(self) -> None:
        payload = {
            "policyVersion": 9,
            "mode": "ADAPTIVE",
            "minBitrateKbps": 500,
            "maxBitrateKbps": 2000,
            "initialBitrateKbps": 1000,
            "decreaseFactor": 0.5,
            "increaseStepKbps": 100,
            "congestionSamples": 1,
            "recoverySamples": 1,
            "cooldownSeconds": 1,
        }
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        command = VerifiedFleetCommand(
            **{
                **self.command.__dict__,
                "command_id": str(uuid.uuid4()),
                "sequence": 2,
                "payload": payload,
                "payload_base64": base64.urlsafe_b64encode(encoded)
                .decode()
                .rstrip("="),
                "payload_hash": hashlib.sha256(encoded).hexdigest(),
            }
        )
        self.inbox.accept(command)
        self.inbox.transition(command.command_id, "IN_PROGRESS")
        outbox = DurableCommandObservationOutbox(self.inbox, max_rows=1)
        store = DurableReconcileStore(self.inbox, observation_outbox=outbox)
        self.inbox.run_transactional_effect(
            command.command_id,
            lambda connection: store.stage_verified(command, connection),
        )
        registry = ReconcilerRegistry()
        registry.register(
            _stream_reconciler(
                _Encoder(),
                clock_ms=iter((0.0, 2000.0, 4000.0)).__next__,
            )
        )
        coordinator = FleetEffectCoordinator(
            inbox=self.inbox,
            store=store,
            registry=registry,
        )

        coordinator.run_once()
        terminal = next(
            item
            for item in outbox.pending()
            if item.command_id == command.command_id
        )
        with self.assertRaises(CommandObservationError) as raised:
            coordinator.observe_connectivity({"quality": "POOR"})
        self.assertEqual(raised.exception.code, "OBSERVATION_OUTBOX_FULL")
        encoder = registry.get("STREAM_POLICY").encoder
        calls_before_retry = list(encoder.set_calls)
        with self.assertRaises(CommandObservationError):
            coordinator.observe_connectivity({"quality": "GOOD"})
        self.assertEqual(encoder.set_calls, calls_before_retry)
        outbox.acknowledge_body(
            json.dumps(
                {
                    "observationId": terminal.observation_id,
                    "commandId": command.command_id,
                    "revision": 1,
                    "status": "ACCEPTED",
                    "processedAt": "2026-09-01T00:00:02Z",
                }
            )
        )
        # No new bitrate transition occurs here; the failed durable write itself
        # forces replay of the current state.
        self.assertEqual(coordinator.observe_connectivity({"quality": "POOR"}), 1)
        second = outbox.pending()[0]
        with self.assertRaises(CommandObservationError):
            coordinator.observe_connectivity({"quality": "POOR"})
        outbox.acknowledge_body(
            json.dumps(
                {
                    "observationId": second.observation_id,
                    "commandId": command.command_id,
                    "revision": 2,
                    "status": "ACCEPTED",
                    "processedAt": "2026-09-01T00:00:03Z",
                }
            )
        )
        # The bitrate remains at the minimum, but the reason-only transition
        # from connectivity_poor to at_minimum is still durable and observable.
        self.assertEqual(coordinator.observe_connectivity({"quality": "POOR"}), 1)
        third = outbox.pending()[0]
        self.assertEqual(coordinator.observe_connectivity({"quality": "POOR"}), 0)

        observations = [terminal, second, third]
        self.assertEqual([item.revision for item in observations], [1, 2, 3])
        self.assertEqual(observations[0].reported_state["appliedBitrateKbps"], 1000)
        self.assertEqual(observations[1].reported_state["appliedBitrateKbps"], 500)
        self.assertEqual(observations[2].reported_state["appliedBitrateKbps"], 500)
        self.assertEqual(
            observations[2].reported_state["lastAdjustmentReason"],
            "at_minimum",
        )
        for observation in observations:
            for key, expected in payload.items():
                self.assertEqual(observation.reported_state[key], expected)
            self.assertEqual(observation.reported_state["encoder"], "x264enc")
            self.assertEqual(
                observation.reported_state["health"],
                "STREAM_CONTINUOUS",
            )
            self.assertIn("requestedBitrateKbps", observation.reported_state)

    def test_stream_health_transitions_are_durable_without_encoder_mutation(self) -> None:
        self.inbox.transition(self.command.command_id, "IN_PROGRESS")
        outbox = DurableCommandObservationOutbox(self.inbox)
        store = DurableReconcileStore(self.inbox, observation_outbox=outbox)
        self.inbox.run_transactional_effect(
            self.command.command_id,
            lambda connection: store.stage_verified(self.command, connection),
        )
        evidence = {"last_frame": 100.0, "now": 100.0}
        encoder = _Encoder()
        reconciler = StreamPolicyReconciler(
            encoder,
            runtime_evidence=lambda: StreamRuntimeEvidence(
                pipeline_running=True,
                last_frame_monotonic=evidence["last_frame"],
            ),
            health_clock=lambda: evidence["now"],
            max_frame_age_seconds=5.0,
        )
        registry = ReconcilerRegistry()
        registry.register(reconciler)
        coordinator = FleetEffectCoordinator(
            inbox=self.inbox,
            store=store,
            registry=registry,
        )

        coordinator.run_once()
        evidence["now"] = 106.0
        self.assertEqual(coordinator.observe_stream_health(), 1)
        self.assertEqual(coordinator.observe_stream_health(), 0)

        # Simulate a process restart after the stale observation persisted but
        # before its recovery.  The fresh replacement must preserve that
        # degraded local state until it can emit the ordered recovery revision.
        evidence["last_frame"] = 106.0
        restarted_encoder = _Encoder()
        restarted_reconciler = StreamPolicyReconciler(
            restarted_encoder,
            runtime_evidence=lambda: StreamRuntimeEvidence(
                pipeline_running=True,
                last_frame_monotonic=evidence["last_frame"],
            ),
            health_clock=lambda: evidence["now"],
            max_frame_age_seconds=5.0,
        )
        restarted_registry = ReconcilerRegistry()
        restarted_registry.register(restarted_reconciler)
        restarted = FleetEffectCoordinator(
            inbox=self.inbox,
            store=store,
            registry=restarted_registry,
        )
        self.assertEqual(restarted.run_once().processed, 0)
        self.assertEqual(
            store.applied_state("STREAM_POLICY").reported_state["health"],
            "STREAM_FRAME_STALE",
        )
        self.assertEqual(restarted.observe_stream_health(), 1)

        observations = outbox.pending()
        self.assertEqual([item.revision for item in observations], [1, 2, 3])
        self.assertEqual(observations[1].reported_state["health"], "STREAM_FRAME_STALE")
        self.assertEqual(
            observations[1].reported_state["lastAdjustmentReason"],
            "STREAM_FRAME_STALE",
        )
        self.assertEqual(observations[2].reported_state["health"], "STREAM_CONTINUOUS")
        self.assertEqual(
            observations[2].reported_state["lastAdjustmentReason"],
            "stream_health_recovered",
        )
        self.assertEqual(encoder.set_calls, [1200])
        self.assertEqual(restarted_encoder.set_calls, [1200])

    def test_row_and_byte_quota_fail_closed(self) -> None:
        outbox = DurableCommandObservationOutbox(
            self.inbox,
            max_rows=10,
            max_bytes=64 * 1024,
        )
        outbox.enqueue(
            command_id=self.command.command_id,
            sequence=1,
            command_type="STREAM_POLICY",
            reported_state={"value": "a" * 40_000},
        )

        with self.assertRaises(CommandObservationError) as raised:
            outbox.enqueue(
                command_id=self.command.command_id,
                sequence=1,
                command_type="STREAM_POLICY",
                reported_state={"value": "b" * 40_000},
            )

        self.assertEqual(raised.exception.code, "OBSERVATION_OUTBOX_FULL")
        health = outbox.health_snapshot()
        self.assertEqual(health.pending_rows, 1)
        self.assertGreater(health.pending_bytes, 40_000)

    def test_full_dlq_preserves_original_as_non_replayable_blocked_record(self) -> None:
        second = _command(2)
        self.inbox.accept(second)
        outbox = DurableCommandObservationOutbox(
            self.inbox,
            dlq_max_rows=1,
        )
        first_observation = outbox.enqueue(
            command_id=self.command.command_id,
            sequence=1,
            command_type="STREAM_POLICY",
            reported_state={"value": "first"},
        )
        second_observation = outbox.enqueue(
            command_id=second.command_id,
            sequence=2,
            command_type="STREAM_POLICY",
            reported_state={"value": "second"},
        )

        def permanent(observation, command_id, revision):
            return json.dumps(
                {
                    "observationId": observation.observation_id,
                    "commandId": command_id,
                    "revision": revision,
                    "status": "REJECTED",
                    "processedAt": "2026-09-01T00:00:02Z",
                    "retryable": False,
                    "code": "INVALID_STATE",
                    "reason": "permanent",
                }
            )

        self.assertTrue(
            outbox.acknowledge_body(
                permanent(first_observation, self.command.command_id, 1)
            )[1]
        )
        self.assertTrue(
            outbox.acknowledge_body(
                permanent(second_observation, second.command_id, 1)
            )[1]
        )

        self.assertEqual(len(outbox.dead_letters()), 1)
        self.assertEqual(outbox.pending(), [])
        blocked = outbox.blocked()
        self.assertEqual([item.observation_id for item in blocked], [second_observation.observation_id])
        self.assertEqual(blocked[0].delivery_state, "DLQ_BLOCKED")
        health = outbox.health_snapshot()
        self.assertEqual(health.dlq_blocked_rows, 1)
        self.assertTrue(health.retention_pressure)

    def test_dlq_age_retention_is_bounded_and_counted(self) -> None:
        ticks = {"now": 0.0}
        outbox = DurableCommandObservationOutbox(
            self.inbox,
            retry_clock=lambda: ticks["now"],
            dlq_max_age_seconds=60,
        )
        observation = outbox.enqueue(
            command_id=self.command.command_id,
            sequence=1,
            command_type="STREAM_POLICY",
            reported_state={"value": "old-rejection"},
        )
        outbox.acknowledge_body(
            json.dumps(
                {
                    "observationId": observation.observation_id,
                    "commandId": self.command.command_id,
                    "revision": 1,
                    "status": "REJECTED",
                    "processedAt": "1970-01-01T00:00:00Z",
                    "retryable": False,
                    "code": "INVALID_STATE",
                }
            )
        )
        self.assertEqual(len(outbox.dead_letters()), 1)

        ticks["now"] = 61.0
        health = outbox.health_snapshot()

        self.assertEqual(outbox.dead_letters(), [])
        self.assertEqual(health.expired_dlq_pruned, 1)

    def test_terminal_reservation_converts_to_observation_without_reexecuting_effect(self) -> None:
        self.inbox.transition(self.command.command_id, "IN_PROGRESS")
        outbox = DurableCommandObservationOutbox(self.inbox, max_rows=1)
        store = DurableReconcileStore(self.inbox, observation_outbox=outbox)
        _ack, staged = self.inbox.run_transactional_effect(
            self.command.command_id,
            lambda connection: store.stage_verified(self.command, connection),
        )
        self.assertTrue(staged)
        self.assertEqual(outbox.health_snapshot().reserved_rows, 1)

        encoder = _Encoder()
        registry = ReconcilerRegistry()
        registry.register(_stream_reconciler(encoder))
        coordinator = FleetEffectCoordinator(
            inbox=self.inbox,
            store=store,
            registry=registry,
        )
        first = coordinator.run_once()
        second = coordinator.run_once()

        self.assertEqual(first.terminal_acks[0].status, "SUCCEEDED")
        self.assertEqual(second.processed, 0)
        self.assertEqual(outbox.health_snapshot().reserved_rows, 0)
        self.assertEqual(len(outbox.pending()), 1)
        self.assertEqual(encoder.set_calls, [1200])

    def test_failed_terminal_uses_lifecycle_ack_and_releases_observation_quota(
        self,
    ) -> None:
        self.inbox.transition(self.command.command_id, "IN_PROGRESS")
        outbox = DurableCommandObservationOutbox(self.inbox, max_rows=1)
        store = DurableReconcileStore(self.inbox, observation_outbox=outbox)
        self.inbox.run_transactional_effect(
            self.command.command_id,
            lambda connection: store.stage_verified(self.command, connection),
        )
        self.assertIsNotNone(store.claim_next(owner="worker", lease_seconds=30))

        ack = store.finish(
            self.command,
            owner="worker",
            outcome=CommandEffectOutcome(
                status="FAILED",
                code="EXPIRED",
                reported_state={"updatePhase": "ROLLBACK_FAILED"},
            ),
        )

        self.assertIsNotNone(ack)
        self.assertEqual(ack.status, "FAILED")
        self.assertEqual(ack.code, "EXPIRED")
        self.assertEqual(outbox.pending(), [])
        self.assertEqual(outbox.health_snapshot().reserved_rows, 0)

    def test_capacity_exhaustion_rejects_before_external_effect(self) -> None:
        outbox = DurableCommandObservationOutbox(self.inbox, max_rows=1)
        outbox.enqueue(
            command_id=self.command.command_id,
            sequence=1,
            command_type="STREAM_POLICY",
            reported_state={"health": "existing"},
        )
        second = _command(2)
        self.inbox.accept(second)
        self.inbox.transition(second.command_id, "IN_PROGRESS")
        store = DurableReconcileStore(self.inbox, observation_outbox=outbox)

        ack, applied = self.inbox.run_transactional_effect(
            second.command_id,
            lambda connection: store.stage_verified(second, connection),
        )

        self.assertTrue(applied)
        self.assertEqual(ack.status, "FAILED")
        self.assertEqual(ack.code, "OBSERVATION_CAPACITY_UNAVAILABLE")
        self.assertIsNone(store.get_job(second.command_id))


if __name__ == "__main__":
    unittest.main()
