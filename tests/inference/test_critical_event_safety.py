from __future__ import annotations

import unittest

from nuvion_app.inference.critical_event_safety import (
    CriticalEventBackpressureError,
    CriticalEventSafetyGate,
    PendingCriticalEvent,
)
from nuvion_app.inference.durable_events import (
    DELIVERY_CLASS_CRITICAL,
    DurableEvent,
    DurableEventCapacityError,
)


def sample_event(
    event_id: str = "334aab50-3cf6-49c4-8362-f3cb26a6994e",
) -> PendingCriticalEvent:
    return PendingCriticalEvent.create(
        event_type="PRODUCTION",
        destination="/app/device/production",
        payload={"count": 1, "labels": ["양품"]},
        event_id=event_id,
        occurred_at="2026-09-01T00:00:00Z",
    )


def persisted(event: PendingCriticalEvent) -> DurableEvent:
    return DurableEvent(
        event_id=event.event_id,
        event_type=event.event_type,
        destination=event.destination,
        payload={
            **event.payload,
            "eventId": event.event_id,
            "occurredAt": event.occurred_at,
        },
        occurred_at=event.occurred_at,
        delivery_class=DELIVERY_CLASS_CRITICAL,
    )


class CriticalEventSafetyGateTest(unittest.TestCase):
    def test_capacity_failure_uses_bounded_retries_and_retains_exact_event(
        self,
    ) -> None:
        sleeps: list[float] = []
        attempted: list[PendingCriticalEvent] = []
        gate = CriticalEventSafetyGate(
            max_attempts=3,
            retry_delay_seconds=0.25,
            sleep=sleeps.append,
        )
        event = sample_event()

        def fail(candidate: PendingCriticalEvent) -> DurableEvent:
            attempted.append(candidate)
            raise DurableEventCapacityError("quota full")

        with self.assertRaises(CriticalEventBackpressureError) as raised:
            gate.persist(event, fail)

        self.assertEqual(raised.exception.event_id, event.event_id)
        self.assertEqual(attempted, [event, event, event])
        self.assertEqual(sleeps, [0.25, 0.25])
        self.assertEqual(gate.pending_event(), event)
        self.assertTrue(gate.is_stopped())
        self.assertTrue(gate.replay_allowed())
        self.assertEqual(gate.health_overlay()["unsavedCriticalEvents"], 1)

    def test_retained_event_retries_with_same_identity_but_stop_requires_operator(
        self,
    ) -> None:
        gate = CriticalEventSafetyGate(max_attempts=1, retry_delay_seconds=0)
        event = sample_event()
        attempts: list[PendingCriticalEvent] = []

        with self.assertRaises(CriticalEventBackpressureError):
            gate.persist(
                event,
                lambda candidate: (_ for _ in ()).throw(
                    DurableEventCapacityError(candidate.event_id)
                ),
            )

        self.assertTrue(
            gate.retry_retained(
                lambda candidate: attempts.append(candidate) or persisted(candidate)
            )
        )
        self.assertEqual(attempts, [event])
        self.assertIsNone(gate.pending_event())
        self.assertTrue(gate.is_stopped())
        self.assertTrue(gate.health_overlay()["retainedEventPersisted"])
        self.assertTrue(gate.clear_operator_stop())
        self.assertFalse(gate.is_stopped())

    def test_operator_stop_never_replaces_retained_event_with_new_observation(
        self,
    ) -> None:
        gate = CriticalEventSafetyGate(max_attempts=1, retry_delay_seconds=0)
        first = sample_event()
        second = sample_event("7e4066d4-c752-44f6-a795-033465f354a6")

        with self.assertRaises(CriticalEventBackpressureError):
            gate.persist(
                first,
                lambda _candidate: (_ for _ in ()).throw(
                    DurableEventCapacityError("full")
                ),
            )
        with self.assertRaises(CriticalEventBackpressureError):
            gate.persist(second, persisted)

        self.assertEqual(gate.pending_event(), first)

    def test_uncorrelated_permanent_protocol_error_stops_replay(self) -> None:
        gate = CriticalEventSafetyGate()

        gate.enter_protocol_stop("permanent payload has no usable eventId")

        self.assertTrue(gate.is_stopped())
        self.assertFalse(gate.replay_allowed())
        self.assertEqual(gate.health_overlay()["unsavedCriticalEvents"], 0)
        self.assertTrue(gate.health_overlay()["protocolStop"])

    def test_anomaly_outbox_init_and_safety_slot_failure_enters_degraded_stop(
        self,
    ) -> None:
        gate = CriticalEventSafetyGate(max_attempts=1, retry_delay_seconds=0)
        event = PendingCriticalEvent.create(
            event_type="ANOMALY",
            destination="/app/device/anomaly",
            payload={"anomalyStatus": "DEFECT", "message": "camera observation"},
            event_id="334aab50-3cf6-49c4-8362-f3cb26a6994e",
            occurred_at="2026-09-01T00:00:00Z",
        )

        with self.assertRaises(CriticalEventBackpressureError) as raised:
            gate.persist(
                event,
                lambda _candidate: (_ for _ in ()).throw(
                    DurableEventCapacityError("critical outbox is unavailable")
                ),
                lambda _candidate, _reason: (_ for _ in ()).throw(
                    DurableEventCapacityError("safety reserve unavailable")
                ),
            )

        self.assertIn("durable safety retention failed", str(raised.exception))
        self.assertEqual(gate.pending_event(), event)
        self.assertTrue(gate.is_stopped())
        self.assertFalse(gate.health_overlay()["durableSafetyRetained"])


if __name__ == "__main__":
    unittest.main()
