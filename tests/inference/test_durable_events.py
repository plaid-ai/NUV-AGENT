from __future__ import annotations

import sqlite3
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nuvion_app.inference.critical_event_safety import (
    CriticalEventBackpressureError,
    CriticalEventSafetyGate,
    PendingCriticalEvent,
)
from nuvion_app.inference.durable_events import (
    ACK_STATUS_ACCEPTED,
    ACK_STATUS_DUPLICATE,
    ACK_STATUS_REJECTED,
    DELIVERY_CLASS_CRITICAL,
    DELIVERY_CLASS_METRIC,
    DELIVERY_CLASS_STATE,
    EVENT_TYPE_ANOMALY,
    EVENT_TYPE_DEVICE_STATE,
    EVENT_TYPE_METRIC,
    EVENT_TYPE_PRODUCTION,
    DurableEvent,
    DurableEventCapacityError,
    DurableEventDelivery,
    DurableEventOutbox,
    is_uncorrelated_permanent_event_rejection,
    parse_event_ack,
    parse_permanent_event_rejection,
    resolve_default_outbox_path,
)


class DurableEventOutboxTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "events.sqlite3"
        self.outbox = DurableEventOutbox(self.db_path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_resolving_default_path_is_lazy_and_does_not_touch_filesystem(self) -> None:
        state_home = Path(self.tmp.name) / "state"

        path = resolve_default_outbox_path({"XDG_STATE_HOME": str(state_home)})

        self.assertEqual(path, (state_home / "nuvion" / "events.sqlite3").resolve())
        self.assertFalse(state_home.exists())

    def test_existing_parent_permissions_are_not_changed(self) -> None:
        existing_parent = Path(self.tmp.name) / "shared"
        existing_parent.mkdir(mode=0o755)

        DurableEventOutbox(existing_parent / "events.sqlite3")

        mode = stat.S_IMODE(existing_parent.stat().st_mode)
        self.assertEqual(mode, 0o755)

    def test_existing_outbox_schema_is_migrated_in_place(self) -> None:
        legacy_path = Path(self.tmp.name) / "legacy.sqlite3"
        event_id = "334aab50-3cf6-49c4-8362-f3cb26a6994e"
        with sqlite3.connect(legacy_path) as connection:
            connection.execute(
                """
                CREATE TABLE durable_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    destination TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    last_attempt_at TEXT
                )
                """
            )
            connection.execute(
                """
                INSERT INTO durable_events (
                    event_id, event_type, destination, payload_json, occurred_at, created_at
                ) VALUES (?, 'PRODUCTION', '/app/device/production', ?, ?, ?)
                """,
                (
                    event_id,
                    '{"count":1,"eventId":"334aab50-3cf6-49c4-8362-f3cb26a6994e","occurredAt":"2026-08-31T01:02:03Z"}',
                    "2026-08-31T01:02:03Z",
                    "2026-08-31T01:02:03Z",
                ),
            )

        migrated = DurableEventOutbox(legacy_path)

        self.assertEqual(migrated.count(), 1)
        self.assertEqual(migrated.pending()[0].event_id, event_id)
        self.assertEqual(migrated.dead_letter_count(), 0)
        self.assertGreater(migrated.record_bytes(), 0)
        with sqlite3.connect(legacy_path) as connection:
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(durable_events)")
            }
        self.assertIn("record_size_bytes", columns)
        self.assertIn("delivery_state", columns)
        self.assertIn("rejection_reason", columns)

    def test_cross_instance_initialization_serializes_in_place_migration(self) -> None:
        legacy_path = Path(self.tmp.name) / "legacy-concurrent.sqlite3"
        with sqlite3.connect(legacy_path) as connection:
            connection.execute(
                """
                CREATE TABLE durable_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    destination TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    last_attempt_at TEXT
                )
                """
            )

        start = threading.Barrier(4)
        errors: list[Exception] = []

        def initialize() -> None:
            try:
                start.wait(timeout=2.0)
                DurableEventOutbox(legacy_path)
            except (RuntimeError, ValueError, sqlite3.Error) as exc:
                errors.append(exc)

        threads = [threading.Thread(target=initialize) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5.0)

        self.assertEqual(errors, [])
        with sqlite3.connect(legacy_path) as connection:
            columns = [
                row[1]
                for row in connection.execute("PRAGMA table_info(durable_events)")
            ]
        self.assertEqual(columns.count("record_size_bytes"), 1)
        self.assertEqual(columns.count("delivery_state"), 1)

    def test_cross_process_same_event_id_is_idempotent(self) -> None:
        database_path = Path(self.tmp.name) / "cross-process.sqlite3"
        gate_path = Path(self.tmp.name) / "start-gate"
        event_id = "334aab50-3cf6-49c4-8362-f3cb26a6994e"
        script = """
import sys
import time
from pathlib import Path
from nuvion_app.inference.durable_events import DurableEventOutbox, EVENT_TYPE_ANOMALY

database_path = Path(sys.argv[1])
gate_path = Path(sys.argv[2])
while not gate_path.exists():
    time.sleep(0.005)
DurableEventOutbox(database_path).persist(
    event_type=EVENT_TYPE_ANOMALY,
    destination="/app/device/anomaly",
    payload={"value": 1},
    event_id=sys.argv[3],
    occurred_at="2026-09-01T00:00:00Z",
)
"""
        processes = [
            subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    script,
                    str(database_path),
                    str(gate_path),
                    event_id,
                ],
                cwd=Path(__file__).parents[2],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for _ in range(2)
        ]
        gate_path.touch()
        outputs = [process.communicate(timeout=10.0) for process in processes]

        self.assertEqual(
            [
                (process.returncode, stdout, stderr)
                for process, (stdout, stderr) in zip(processes, outputs)
            ],
            [(0, "", ""), (0, "", "")],
        )
        self.assertEqual(DurableEventOutbox(database_path).count(), 1)

    def test_persist_adds_contract_fields_and_survives_reopen(self) -> None:
        event = self.outbox.persist(
            event_type=EVENT_TYPE_ANOMALY,
            destination="/app/device/anomaly",
            payload={"anomalyStatus": "DEFECT"},
            event_id="334aab50-3cf6-49c4-8362-f3cb26a6994e",
            occurred_at="2026-08-31T01:02:03Z",
        )

        reopened = DurableEventOutbox(self.db_path)
        pending = reopened.pending()

        self.assertEqual(event.payload["eventId"], event.event_id)
        self.assertEqual(event.payload["occurredAt"], "2026-08-31T01:02:03Z")
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0], event)

    def test_queue_or_send_failure_cannot_delete_persisted_event(self) -> None:
        event = self.outbox.persist(
            event_type=EVENT_TYPE_PRODUCTION,
            destination="/app/device/production",
            payload={"count": 1},
        )

        self.outbox.mark_attempt(event.event_id)

        self.assertEqual(self.outbox.count(), 1)
        self.assertEqual(self.outbox.pending()[0].attempt_count, 1)

    def test_ack_deletes_only_matching_accepted_or_duplicate_event(self) -> None:
        event = self.outbox.persist(
            event_type=EVENT_TYPE_ANOMALY,
            destination="/app/device/anomaly",
            payload={"anomalyStatus": "DEFECT"},
        )

        self.assertFalse(
            self.outbox.acknowledge(
                event.event_id, EVENT_TYPE_PRODUCTION, ACK_STATUS_ACCEPTED
            )
        )
        self.assertEqual(self.outbox.count(), 1)
        self.assertTrue(
            self.outbox.acknowledge(
                event.event_id, EVENT_TYPE_ANOMALY, ACK_STATUS_DUPLICATE
            )
        )
        self.assertEqual(self.outbox.count(), 0)

    def test_wait_for_ack_unblocks_clip_finalize_after_server_ack(self) -> None:
        event = self.outbox.persist(
            event_type=EVENT_TYPE_ANOMALY,
            destination="/app/device/anomaly",
            payload={"anomalyStatus": "DEFECT"},
        )
        result: list[bool] = []
        waiter = threading.Thread(
            target=lambda: result.append(self.outbox.wait_for_ack(event.event_id, 1.0))
        )
        waiter.start()

        self.outbox.acknowledge(event.event_id, EVENT_TYPE_ANOMALY, ACK_STATUS_ACCEPTED)
        waiter.join(timeout=2.0)

        self.assertEqual(result, [True])

    def test_parse_event_ack_enforces_contract(self) -> None:
        parsed = parse_event_ack(
            '{"eventId":"334aab50-3cf6-49c4-8362-f3cb26a6994e",'
            '"eventType":"ANOMALY","status":"ACCEPTED","processedAt":"2026-08-31T01:02:04Z"}'
        )

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.event_type, EVENT_TYPE_ANOMALY)
        self.assertEqual(parsed.status, ACK_STATUS_ACCEPTED)
        self.assertIsNone(
            parse_event_ack('{"eventId":"x","eventType":"ANOMALY","status":"FAILED"}')
        )
        self.assertIsNone(
            parse_event_ack('{"eventType":"ANOMALY","status":"ACCEPTED"}')
        )

    def test_permanent_rejection_moves_pending_event_to_dlq(self) -> None:
        delivery = DurableEventDelivery(self.outbox)
        event = delivery.publish(
            event_type=EVENT_TYPE_ANOMALY,
            destination="/app/device/anomaly",
            payload={"anomalyStatus": "DEFECT"},
            event_id="334aab50-3cf6-49c4-8362-f3cb26a6994e",
            occurred_at="2026-08-31T01:02:03Z",
            sender=lambda _event: True,
        )
        delivery.mark_sent(event.event_id)

        ack, changed = delivery.acknowledge_body(
            '{"eventId":"334aab50-3cf6-49c4-8362-f3cb26a6994e",'
            '"eventType":"ANOMALY","status":"REJECTED","retryable":false,'
            '"code":"AGENTWS_400_001","reason":"invalid payload",'
            '"processedAt":"2026-08-31T01:02:04Z"}'
        )

        self.assertIsNotNone(ack)
        self.assertEqual(ack.status, ACK_STATUS_REJECTED)
        self.assertTrue(changed)
        self.assertEqual(self.outbox.count(), 0)
        self.assertEqual(self.outbox.dead_letter_count(), 1)
        dead = self.outbox.dead_letters()[0]
        self.assertEqual(dead.event_id, event.event_id)
        self.assertEqual(dead.rejection_code, "AGENTWS_400_001")
        self.assertEqual(dead.reason, "invalid payload")
        self.assertEqual(dead.attempt_count, 1)

    def test_agent_error_requires_event_id_and_explicit_permanent_classification(
        self,
    ) -> None:
        rejection = parse_permanent_event_rejection(
            {
                "eventId": "334aab50-3cf6-49c4-8362-f3cb26a6994e",
                "path": "/app/device/anomaly",
                "status": 400,
                "retryable": False,
                "code": "AGENTWS_400_001",
                "detail": "invalid payload",
            }
        )

        self.assertIsNotNone(rejection)
        self.assertEqual(rejection.event_type, EVENT_TYPE_ANOMALY)
        self.assertIsNone(
            parse_permanent_event_rejection(
                {
                    "path": "/app/device/anomaly",
                    "status": 400,
                    "retryable": False,
                }
            )
        )
        collision = parse_permanent_event_rejection(
            {
                "eventId": "334aab50-3cf6-49c4-8362-f3cb26a6994e",
                "path": "/app/device/production",
                "status": 409,
                "retryable": False,
                "terminal": True,
                "failureClass": "PERMANENT",
                "code": "EVENT_ID_COLLISION",
            }
        )
        self.assertIsNotNone(collision)
        self.assertEqual(collision.event_type, EVENT_TYPE_PRODUCTION)
        self.assertIsNone(
            parse_permanent_event_rejection(
                {
                    "eventId": "334aab50-3cf6-49c4-8362-f3cb26a6994e",
                    "path": "/app/device/anomaly",
                    "status": 500,
                    "retryable": True,
                }
            )
        )

    def test_retryable_rejection_stays_pending_for_replay(self) -> None:
        delivery = DurableEventDelivery(self.outbox)
        delivery.publish(
            event_type=EVENT_TYPE_PRODUCTION,
            destination="/app/device/production",
            payload={"count": 1},
            event_id="334aab50-3cf6-49c4-8362-f3cb26a6994e",
            occurred_at="2026-08-31T01:02:03Z",
            sender=lambda _event: True,
        )

        ack, changed = delivery.acknowledge_body(
            '{"eventId":"334aab50-3cf6-49c4-8362-f3cb26a6994e",'
            '"eventType":"PRODUCTION","status":"REJECTED","retryable":true,'
            '"reason":"temporary dependency failure"}'
        )

        self.assertIsNotNone(ack)
        self.assertFalse(changed)
        self.assertEqual(self.outbox.count(), 1)
        self.assertEqual(self.outbox.dead_letter_count(), 0)

    def test_rejection_without_explicit_retryability_stays_pending(self) -> None:
        delivery = DurableEventDelivery(self.outbox)
        event = delivery.publish(
            event_type=EVENT_TYPE_PRODUCTION,
            destination="/app/device/production",
            payload={"count": 1},
            event_id="334aab50-3cf6-49c4-8362-f3cb26a6994e",
            occurred_at="2026-08-31T01:02:03Z",
            sender=lambda _event: True,
        )

        ack, changed = delivery.acknowledge_body(
            '{"eventId":"334aab50-3cf6-49c4-8362-f3cb26a6994e",'
            '"eventType":"PRODUCTION","status":"REJECTED",'
            '"reason":"classification missing"}'
        )

        self.assertIsNotNone(ack)
        self.assertIsNone(ack.retryable)
        self.assertFalse(changed)
        self.assertTrue(self.outbox.is_pending(event.event_id))
        self.assertEqual(self.outbox.dead_letter_count(), 0)

    def test_low_level_acknowledge_never_deletes_rejected_event(self) -> None:
        event = self.outbox.persist(
            event_type=EVENT_TYPE_ANOMALY,
            destination="/app/device/anomaly",
            payload={"anomalyStatus": "DEFECT"},
        )

        changed = self.outbox.acknowledge(
            event.event_id,
            EVENT_TYPE_ANOMALY,
            ACK_STATUS_REJECTED,
        )

        self.assertFalse(changed)
        self.assertTrue(self.outbox.is_pending(event.event_id))

    def test_success_ack_deletes_without_creating_dead_letter(self) -> None:
        event = self.outbox.persist(
            event_type=EVENT_TYPE_ANOMALY,
            destination="/app/device/anomaly",
            payload={"anomalyStatus": "DEFECT"},
        )

        self.assertTrue(
            self.outbox.acknowledge(
                event.event_id, EVENT_TYPE_ANOMALY, ACK_STATUS_ACCEPTED
            )
        )
        self.assertEqual(self.outbox.count(), 0)
        self.assertEqual(self.outbox.dead_letter_count(), 0)

    def test_row_quota_applies_backpressure_without_evicting_critical_pending(
        self,
    ) -> None:
        limited = DurableEventOutbox(self.db_path, max_rows=2, max_bytes=1_000_000)
        first = limited.persist(
            event_type=EVENT_TYPE_PRODUCTION,
            destination="/app/device/production",
            payload={"count": 1},
        )
        second = limited.persist(
            event_type=EVENT_TYPE_PRODUCTION,
            destination="/app/device/production",
            payload={"count": 2},
        )
        sender_calls: list[str] = []
        with self.assertRaises(DurableEventCapacityError):
            DurableEventDelivery(limited).publish(
                event_type=EVENT_TYPE_PRODUCTION,
                destination="/app/device/production",
                payload={"count": 3},
                event_id="334aab50-3cf6-49c4-8362-f3cb26a6994e",
                occurred_at="2026-08-31T01:02:03Z",
                sender=lambda event: sender_calls.append(event.event_id) is None,
            )

        self.assertFalse(first.dead_lettered)
        self.assertFalse(second.dead_lettered)
        self.assertEqual(sender_calls, [])
        self.assertEqual(
            [event.event_id for event in limited.pending()],
            [first.event_id, second.event_id],
        )
        self.assertEqual(limited.dead_letters(), [])

    def test_byte_quota_rejects_oversized_critical_event(self) -> None:
        limited = DurableEventOutbox(self.db_path, max_rows=100, max_bytes=64)

        with self.assertRaises(DurableEventCapacityError):
            limited.persist(
                event_type=EVENT_TYPE_ANOMALY,
                destination="/app/device/anomaly",
                payload={"message": "x" * 200},
            )

        self.assertEqual(limited.count(), 0)
        self.assertEqual(limited.dead_letter_count(), 0)

    def test_dlq_row_cap_purges_oldest_dead_letter(self) -> None:
        limited = DurableEventOutbox(
            self.db_path, max_rows=10, max_bytes=1_000_000, max_dead_letters=2
        )
        dead_ids: list[str] = []
        for count in range(3):
            event = limited.persist(
                event_type=EVENT_TYPE_PRODUCTION,
                destination="/app/device/production",
                payload={"count": count},
            )
            limited.dead_letter(
                event.event_id,
                EVENT_TYPE_PRODUCTION,
                reason="permanent rejection",
                rejection_code="TEST_REJECTED",
                source="test",
            )
            dead_ids.append(event.event_id)

        remaining_ids = [event.event_id for event in limited.dead_letters(limit=10)]
        self.assertEqual(remaining_ids, dead_ids[-2:])

    def test_state_compaction_keeps_only_latest_value_for_key(self) -> None:
        first = self.outbox.persist(
            event_type=EVENT_TYPE_DEVICE_STATE,
            destination="/app/device/state",
            payload={"runtimeStatus": "STARTING"},
            delivery_class=DELIVERY_CLASS_STATE,
            compaction_key="runtime",
        )
        second = self.outbox.persist(
            event_type=EVENT_TYPE_DEVICE_STATE,
            destination="/app/device/state",
            payload={"runtimeStatus": "RUNNING"},
            delivery_class=DELIVERY_CLASS_STATE,
            compaction_key="runtime",
        )

        pending = self.outbox.pending()

        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].event_id, second.event_id)
        self.assertNotEqual(first.event_id, second.event_id)
        self.assertEqual(pending[0].payload["runtimeStatus"], "RUNNING")

    def test_critical_event_evicts_oldest_metric_when_capacity_is_full(self) -> None:
        limited = DurableEventOutbox(self.db_path, max_rows=2, max_bytes=1_000_000)
        old_metric = limited.persist(
            event_type=EVENT_TYPE_METRIC,
            destination="/app/device/metric",
            payload={"cpuPct": 10},
            delivery_class=DELIVERY_CLASS_METRIC,
        )
        kept_metric = limited.persist(
            event_type=EVENT_TYPE_METRIC,
            destination="/app/device/metric",
            payload={"cpuPct": 20},
            delivery_class=DELIVERY_CLASS_METRIC,
        )

        critical = limited.persist(
            event_type=EVENT_TYPE_ANOMALY,
            destination="/app/device/anomaly",
            payload={"anomalyStatus": "DEFECT"},
            delivery_class=DELIVERY_CLASS_CRITICAL,
        )

        pending_ids = [event.event_id for event in limited.pending()]
        self.assertNotIn(old_metric.event_id, pending_ids)
        self.assertIn(kept_metric.event_id, pending_ids)
        self.assertIn(critical.event_id, pending_ids)
        self.assertFalse(critical.dead_lettered)

    def test_metric_overflow_is_dropped_instead_of_consuming_dlq(self) -> None:
        limited = DurableEventOutbox(self.db_path, max_rows=1, max_bytes=1_000_000)
        limited.persist(
            event_type=EVENT_TYPE_ANOMALY,
            destination="/app/device/anomaly",
            payload={"anomalyStatus": "DEFECT"},
        )

        metric = limited.persist(
            event_type=EVENT_TYPE_METRIC,
            destination="/app/device/metric",
            payload={"cpuPct": 42},
            delivery_class=DELIVERY_CLASS_METRIC,
        )

        self.assertTrue(metric.dropped)
        self.assertEqual(limited.count(), 1)
        self.assertEqual(limited.dead_letter_count(), 0)

    def test_age_policy_drops_metric_but_keeps_critical_pending_until_ack(self) -> None:
        current = {"value": datetime(2026, 9, 1, tzinfo=timezone.utc)}
        limited = DurableEventOutbox(
            self.db_path,
            max_age_seconds=60,
            wall_clock=lambda: current["value"],
        )
        metric = limited.persist(
            event_type=EVENT_TYPE_METRIC,
            destination="/app/device/metric",
            payload={"cpuPct": 42},
            delivery_class=DELIVERY_CLASS_METRIC,
        )
        critical = limited.persist(
            event_type=EVENT_TYPE_PRODUCTION,
            destination="/app/device/production",
            payload={"count": 1},
        )

        current["value"] += timedelta(seconds=61)
        pending = limited.pending()

        self.assertEqual([event.event_id for event in pending], [critical.event_id])
        self.assertEqual(limited.dead_letters(), [])
        self.assertFalse(limited.is_pending(metric.event_id))
        self.assertNotEqual(metric.event_id, critical.event_id)

    def test_failed_state_replacement_keeps_previous_state_pending(self) -> None:
        limited = DurableEventOutbox(self.db_path, max_rows=10, max_bytes=700)
        previous = limited.persist(
            event_type=EVENT_TYPE_DEVICE_STATE,
            destination="/app/device/state",
            payload={"runtimeStatus": "RUNNING"},
            compaction_key="runtime",
        )

        replacement = limited.persist(
            event_type=EVENT_TYPE_DEVICE_STATE,
            destination="/app/device/state",
            payload={"runtimeStatus": "ERROR", "detail": "x" * 1_000},
            compaction_key="runtime",
        )

        self.assertTrue(replacement.dropped)
        self.assertFalse(replacement.dead_lettered)
        self.assertEqual(
            [event.event_id for event in limited.pending()], [previous.event_id]
        )

    def test_oversized_critical_does_not_evict_lower_priority_backlog(self) -> None:
        limited = DurableEventOutbox(self.db_path, max_rows=10, max_bytes=700)
        metric = limited.persist(
            event_type=EVENT_TYPE_METRIC,
            destination="/app/device/metric",
            payload={"cpuPct": 42},
        )

        with self.assertRaises(DurableEventCapacityError):
            limited.persist(
                event_type=EVENT_TYPE_ANOMALY,
                destination="/app/device/anomaly",
                payload={"message": "x" * 1_000},
            )

        self.assertEqual(
            [event.event_id for event in limited.pending()], [metric.event_id]
        )

    def test_dlq_capacity_failure_keeps_original_event_blocked_without_replay(
        self,
    ) -> None:
        limited = DurableEventOutbox(
            self.db_path,
            max_rows=10,
            max_bytes=10_000,
            max_dead_letter_bytes=1,
        )
        event = limited.persist(
            event_type=EVENT_TYPE_ANOMALY,
            destination="/app/device/anomaly",
            payload={"message": "retain me"},
        )

        changed = limited.dead_letter(
            event.event_id,
            EVENT_TYPE_ANOMALY,
            reason="permanent rejection",
            source="test",
        )

        self.assertFalse(changed)
        self.assertFalse(limited.is_pending(event.event_id))
        self.assertEqual(limited.dead_letter_count(), 0)
        self.assertEqual(limited.blocked_count(), 1)
        blocked = limited.blocked_events()[0]
        self.assertEqual(blocked.event_id, event.event_id)
        sent_ids: list[str] = []
        self.assertEqual(
            DurableEventDelivery(limited).replay(
                lambda item: sent_ids.append(item.event_id) is None,
                limit=10,
            ),
            0,
        )
        self.assertEqual(sent_ids, [])

    def test_sql_dlq_insert_failure_rolls_back_and_keeps_source_pending(self) -> None:
        event = self.outbox.persist(
            event_type=EVENT_TYPE_ANOMALY,
            destination="/app/device/anomaly",
            payload={"message": "retain on SQL failure"},
        )
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TRIGGER reject_dead_letter_insert
                BEFORE INSERT ON dead_letter_events
                BEGIN
                    SELECT RAISE(ABORT, 'simulated DLQ write failure');
                END
                """
            )

        with self.assertRaises(sqlite3.IntegrityError):
            self.outbox.dead_letter(
                event.event_id,
                EVENT_TYPE_ANOMALY,
                reason="permanent rejection",
                source="test",
            )

        self.assertTrue(self.outbox.is_pending(event.event_id))
        self.assertEqual(self.outbox.dead_letter_count(), 0)

    def test_unretainable_critical_raises_without_evicting_existing_metric(
        self,
    ) -> None:
        limited = DurableEventOutbox(
            self.db_path,
            max_rows=10,
            max_bytes=700,
            max_dead_letter_bytes=1,
        )
        metric = limited.persist(
            event_type=EVENT_TYPE_METRIC,
            destination="/app/device/metric",
            payload={"cpuPct": 42},
        )

        with self.assertRaises(DurableEventCapacityError):
            limited.persist(
                event_type=EVENT_TYPE_ANOMALY,
                destination="/app/device/anomaly",
                payload={"message": "x" * 1_000},
            )

        self.assertEqual(
            [event.event_id for event in limited.pending()], [metric.event_id]
        )
        self.assertEqual(limited.dead_letter_count(), 0)

    def test_sql_outbox_insert_failure_does_not_apply_planned_evictions(self) -> None:
        limited = DurableEventOutbox(self.db_path, max_rows=1, max_bytes=10_000)
        metric = limited.persist(
            event_type=EVENT_TYPE_METRIC,
            destination="/app/device/metric",
            payload={"cpuPct": 42},
        )
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TRIGGER reject_critical_insert
                BEFORE INSERT ON durable_events
                WHEN NEW.delivery_class = 'CRITICAL'
                BEGIN
                    SELECT RAISE(ABORT, 'simulated outbox write failure');
                END
                """
            )

        with self.assertRaises(sqlite3.IntegrityError):
            limited.persist(
                event_type=EVENT_TYPE_ANOMALY,
                destination="/app/device/anomaly",
                payload={"message": "critical"},
            )

        self.assertEqual(
            [event.event_id for event in limited.pending()], [metric.event_id]
        )

    def test_commit_failure_rolls_back_database_and_terminal_state(self) -> None:
        class CommitFailingConnection(sqlite3.Connection):
            fail_commit = False

            def commit(self) -> None:
                if self.fail_commit:
                    raise sqlite3.OperationalError("simulated commit failure")
                super().commit()

        class CommitFailingOutbox(DurableEventOutbox):
            def _connect(self) -> sqlite3.Connection:
                connection = sqlite3.connect(
                    str(self.path),
                    timeout=30.0,
                    factory=CommitFailingConnection,
                )
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA busy_timeout = 30000")
                return connection

        outbox = CommitFailingOutbox(self.db_path)
        first = outbox.persist(
            event_type=EVENT_TYPE_DEVICE_STATE,
            destination="/app/device/state",
            payload={"runtimeStatus": "STARTING"},
            compaction_key="runtime",
        )

        CommitFailingConnection.fail_commit = True
        try:
            with self.assertRaisesRegex(sqlite3.OperationalError, "commit failure"):
                outbox.persist(
                    event_type=EVENT_TYPE_DEVICE_STATE,
                    destination="/app/device/state",
                    payload={"runtimeStatus": "RUNNING"},
                    compaction_key="runtime",
                )
        finally:
            CommitFailingConnection.fail_commit = False

        self.assertTrue(outbox.is_pending(first.event_id))
        self.assertIsNone(outbox.terminal_result(first.event_id))

    def test_dlq_that_cannot_fit_full_payload_blocks_and_preserves_source(self) -> None:
        event_id = "334aab50-3cf6-49c4-8362-f3cb26a6994e"
        limited = DurableEventOutbox(
            self.db_path,
            max_rows=10,
            max_bytes=10_000,
            max_dead_letter_bytes=640,
        )
        arguments = {
            "event_type": EVENT_TYPE_ANOMALY,
            "destination": "/app/device/anomaly",
            "payload": {"message": "x" * 1_000},
            "event_id": event_id,
            "occurred_at": "2026-09-01T00:00:00Z",
        }

        pending = limited.persist(**arguments)
        self.assertFalse(
            limited.dead_letter(
                pending.event_id,
                EVENT_TYPE_ANOMALY,
                reason="permanent rejection",
            )
        )
        self.assertEqual(limited.dead_letter_count(), 0)
        self.assertEqual(limited.blocked_count(), 1)
        with sqlite3.connect(limited.path) as connection:
            payload_json, delivery_state = connection.execute(
                "SELECT payload_json, delivery_state FROM durable_events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        self.assertIn('"message":"' + "x" * 1_000 + '"', payload_json)
        self.assertEqual(delivery_state, "DLQ_BLOCKED")

        duplicate = limited.persist(**arguments)
        self.assertFalse(duplicate.dead_lettered)
        self.assertEqual(duplicate.delivery_state, "DLQ_BLOCKED")

    def test_event_type_priority_cannot_be_downgraded_by_caller(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be downgraded"):
            self.outbox.persist(
                event_type=EVENT_TYPE_ANOMALY,
                destination="/app/device/anomaly",
                payload={"message": "critical"},
                delivery_class=DELIVERY_CLASS_METRIC,
            )

    def test_metric_overflow_replaces_oldest_metric_with_freshest_sample(self) -> None:
        limited = DurableEventOutbox(self.db_path, max_rows=2, max_bytes=100_000)
        first = limited.persist(
            event_type=EVENT_TYPE_METRIC,
            destination="/app/device/metric",
            payload={"sample": 1},
        )
        second = limited.persist(
            event_type=EVENT_TYPE_METRIC,
            destination="/app/device/metric",
            payload={"sample": 2},
        )
        third = limited.persist(
            event_type=EVENT_TYPE_METRIC,
            destination="/app/device/metric",
            payload={"sample": 3},
        )

        self.assertFalse(third.dropped)
        self.assertEqual(
            [event.payload["sample"] for event in limited.pending()], [2, 3]
        )
        self.assertFalse(limited.is_pending(first.event_id))
        self.assertTrue(limited.is_pending(second.event_id))

    def test_cross_instance_capacity_plan_is_serialized(self) -> None:
        plan_barrier = threading.Barrier(2)

        class InterleavingOutbox(DurableEventOutbox):
            def _plan_capacity_locked(self, *args, **kwargs):
                result = super()._plan_capacity_locked(*args, **kwargs)
                try:
                    plan_barrier.wait(timeout=0.1)
                except threading.BrokenBarrierError:
                    pass
                return result

        seed = DurableEventOutbox(self.db_path, max_rows=2, max_bytes=100_000)
        seed.persist(
            event_type=EVENT_TYPE_ANOMALY,
            destination="/app/device/anomaly",
            payload={"seed": True},
        )
        outboxes = [
            InterleavingOutbox(self.db_path, max_rows=2, max_bytes=100_000),
            InterleavingOutbox(self.db_path, max_rows=2, max_bytes=100_000),
        ]
        errors: list[Exception] = []

        def persist_metric(outbox: DurableEventOutbox, sample: int) -> None:
            try:
                outbox.persist(
                    event_type=EVENT_TYPE_METRIC,
                    destination="/app/device/metric",
                    payload={"sample": sample},
                )
            except (RuntimeError, ValueError, sqlite3.Error) as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=persist_metric, args=(outbox, index))
            for index, outbox in enumerate(outboxes, start=1)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5.0)

        self.assertEqual(errors, [])
        self.assertEqual(DurableEventOutbox(self.db_path).count(), 2)

    def test_cross_instance_same_event_id_is_idempotent_and_collision_is_value_error(
        self,
    ) -> None:
        event_id = "334aab50-3cf6-49c4-8362-f3cb26a6994e"
        start_barrier = threading.Barrier(2)
        outboxes = [DurableEventOutbox(self.db_path), DurableEventOutbox(self.db_path)]
        results: list[DurableEvent] = []
        errors: list[Exception] = []

        def persist(outbox: DurableEventOutbox, payload: dict) -> None:
            try:
                start_barrier.wait(timeout=2.0)
                results.append(
                    outbox.persist(
                        event_type=EVENT_TYPE_ANOMALY,
                        destination="/app/device/anomaly",
                        payload=payload,
                        event_id=event_id,
                        occurred_at="2026-09-01T00:00:00Z",
                    )
                )
            except (RuntimeError, ValueError, sqlite3.Error) as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=persist, args=(outbox, {"value": 1}))
            for outbox in outboxes
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5.0)

        self.assertEqual(errors, [])
        self.assertEqual([event.event_id for event in results], [event_id, event_id])
        self.assertEqual(DurableEventOutbox(self.db_path).count(), 1)

        collision_errors: list[Exception] = []

        def collide() -> None:
            try:
                DurableEventOutbox(self.db_path).persist(
                    event_type=EVENT_TYPE_ANOMALY,
                    destination="/app/device/anomaly",
                    payload={"value": 2},
                    event_id=event_id,
                    occurred_at="2026-09-01T00:00:00Z",
                )
            except ValueError as exc:
                collision_errors.append(exc)

        collide()
        self.assertEqual(len(collision_errors), 1)
        self.assertIsInstance(collision_errors[0], ValueError)

    def test_logical_record_quota_counts_rejection_metadata_and_bounds_external_strings(
        self,
    ) -> None:
        limited = DurableEventOutbox(
            self.db_path,
            max_bytes=100_000,
            max_dead_letter_bytes=4_096,
        )
        event = limited.persist(
            event_type=EVENT_TYPE_ANOMALY,
            destination="/app/device/anomaly",
            payload={"value": 1},
        )

        self.assertTrue(
            limited.dead_letter(
                event.event_id,
                EVENT_TYPE_ANOMALY,
                reason="x" * 1_000_000,
                rejection_code="Y" * 10_000,
                source="Z" * 10_000,
            )
        )
        dead = limited.dead_letters()[0]
        self.assertLessEqual(len(dead.reason.encode("utf-8")), 2_048)
        self.assertLessEqual(len(dead.rejection_code.encode("utf-8")), 128)
        self.assertLessEqual(len(dead.source.encode("utf-8")), 128)
        self.assertLessEqual(limited.dead_letter_record_bytes(), 4_096)

        with self.assertRaisesRegex(ValueError, "destination exceeds"):
            limited.persist(
                event_type=EVENT_TYPE_METRIC,
                destination="/app/device/" + "x" * 600,
                payload={"value": 2},
            )
        with self.assertRaisesRegex(ValueError, "compaction_key exceeds"):
            limited.persist(
                event_type=EVENT_TYPE_DEVICE_STATE,
                destination="/app/device/state",
                payload={"value": 2},
                compaction_key="x" * 300,
            )

    def test_lower_priority_dlq_cannot_purge_critical_and_ack_can_retry_blocked_move(
        self,
    ) -> None:
        limited = DurableEventOutbox(
            self.db_path,
            max_bytes=100_000,
            max_dead_letters=1,
            max_dead_letter_bytes=10_000,
        )
        critical = limited.persist(
            event_type=EVENT_TYPE_ANOMALY,
            destination="/app/device/anomaly",
            payload={"value": "critical"},
        )
        self.assertTrue(
            limited.dead_letter(
                critical.event_id, EVENT_TYPE_ANOMALY, reason="critical reject"
            )
        )
        state = limited.persist(
            event_type=EVENT_TYPE_DEVICE_STATE,
            destination="/app/device/state",
            payload={"value": "state"},
        )

        self.assertFalse(
            limited.dead_letter(
                state.event_id, EVENT_TYPE_DEVICE_STATE, reason="state reject"
            )
        )
        self.assertEqual(limited.dead_letters()[0].event_id, critical.event_id)
        self.assertEqual(limited.blocked_events()[0].event_id, state.event_id)
        self.assertEqual(
            limited.blocked_events()[0].rejection_code, "PERMANENT_REJECTION"
        )

        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "DELETE FROM dead_letter_events WHERE event_id = ?",
                (critical.event_id,),
            )

        self.assertTrue(
            limited.dead_letter(
                state.event_id, EVENT_TYPE_DEVICE_STATE, reason="state reject"
            )
        )
        self.assertEqual(limited.blocked_count(), 0)
        self.assertEqual(limited.dead_letters()[0].event_id, state.event_id)

    def test_transport_send_is_terminal_for_state_but_critical_waits_for_application_ack(
        self,
    ) -> None:
        delivery = DurableEventDelivery(self.outbox)
        state = delivery.publish(
            event_type=EVENT_TYPE_DEVICE_STATE,
            destination="/app/device/state",
            payload={"runtimeStatus": "RUNNING"},
            event_id="334aab50-3cf6-49c4-8362-f3cb26a6994e",
            occurred_at="2026-09-01T00:00:00Z",
            sender=lambda _event: True,
            compaction_key="runtime",
        )

        self.assertTrue(delivery.mark_sent(state.event_id))
        self.assertFalse(self.outbox.is_pending(state.event_id))
        self.assertEqual(delivery.inflight_count(), 0)
        self.assertTrue(delivery.wait_for_ack(state.event_id, 0.0))

        metric = delivery.publish(
            event_type=EVENT_TYPE_METRIC,
            destination="/app/device/metric",
            payload={"cpuPct": 42},
            event_id="e69cb8a4-48c9-4a4b-8e31-8dcf7bd31d69",
            occurred_at="2026-09-01T00:00:00Z",
            sender=lambda _event: True,
        )
        self.assertTrue(delivery.mark_sent(metric.event_id))
        self.assertFalse(self.outbox.is_pending(metric.event_id))

        critical = delivery.publish(
            event_type=EVENT_TYPE_ANOMALY,
            destination="/app/device/anomaly",
            payload={"value": "critical"},
            event_id="2b395cd2-ff9c-4660-b55a-8e3fbe0059ad",
            occurred_at="2026-09-01T00:00:01Z",
            sender=lambda _event: True,
        )
        self.assertTrue(delivery.mark_sent(critical.event_id))
        self.assertTrue(self.outbox.is_pending(critical.event_id))
        self.assertTrue(
            self.outbox.acknowledge(
                critical.event_id, EVENT_TYPE_ANOMALY, ACK_STATUS_ACCEPTED
            )
        )

    def test_state_compaction_releases_superseded_delivery_inflight(self) -> None:
        delivery = DurableEventDelivery(self.outbox)
        first = delivery.publish(
            event_type=EVENT_TYPE_DEVICE_STATE,
            destination="/app/device/state",
            payload={"runtimeStatus": "STARTING"},
            event_id="334aab50-3cf6-49c4-8362-f3cb26a6994e",
            occurred_at="2026-09-01T00:00:00Z",
            sender=lambda _event: True,
            compaction_key="runtime",
        )
        second = delivery.publish(
            event_type=EVENT_TYPE_DEVICE_STATE,
            destination="/app/device/state",
            payload={"runtimeStatus": "RUNNING"},
            event_id="2b395cd2-ff9c-4660-b55a-8e3fbe0059ad",
            occurred_at="2026-09-01T00:00:01Z",
            sender=lambda _event: True,
            compaction_key="runtime",
        )

        self.assertFalse(self.outbox.is_pending(first.event_id))
        self.assertTrue(self.outbox.is_pending(second.event_id))
        self.assertEqual(delivery.inflight_count(), 1)

        with self.assertRaisesRegex(ValueError, "cannot be downgraded"):
            self.outbox.persist(
                event_type=EVENT_TYPE_DEVICE_STATE,
                destination="/app/device/state",
                payload={"runtimeStatus": "RUNNING"},
                delivery_class=DELIVERY_CLASS_METRIC,
            )

    def test_dlq_byte_cap_keeps_logical_payload_storage_bounded(self) -> None:
        limited = DurableEventOutbox(
            self.db_path,
            max_rows=10,
            max_bytes=1_000_000,
            max_dead_letters=10,
            max_dead_letter_bytes=800,
        )
        for count in range(3):
            event = limited.persist(
                event_type=EVENT_TYPE_ANOMALY,
                destination="/app/device/anomaly",
                payload={"message": f"{count}-" + "x" * 180},
            )
            limited.dead_letter(
                event.event_id,
                EVENT_TYPE_ANOMALY,
                reason="permanent rejection",
                source="test",
            )

        self.assertLessEqual(limited.dead_letter_record_bytes(), 800)
        self.assertGreater(limited.dead_letter_count(), 0)
        self.assertLess(limited.dead_letter_count(), 3)

    def test_health_snapshot_reports_capacity_and_oldest_critical_age(self) -> None:
        now = datetime(2026, 9, 1, 0, 10, tzinfo=timezone.utc)
        limited = DurableEventOutbox(
            self.db_path,
            max_rows=2,
            max_bytes=100_000,
            wall_clock=lambda: now,
        )
        limited.persist(
            event_type=EVENT_TYPE_ANOMALY,
            destination="/app/device/anomaly",
            payload={"message": "critical"},
            occurred_at="2026-09-01T00:00:00Z",
        )

        health = limited.health_snapshot().to_telemetry()

        self.assertEqual(health["pendingRows"], 1)
        self.assertGreater(health["pendingBytes"], 0)
        self.assertEqual(health["oldestCriticalAgeSeconds"], 0)
        self.assertEqual(health["dlqRows"], 0)
        self.assertEqual(health["blockedRows"], 0)
        self.assertEqual(health["capacityState"], "HEALTHY")

    def test_critical_safety_slot_survives_restart_and_retries_exact_payload(
        self,
    ) -> None:
        outbox = DurableEventOutbox(
            self.db_path,
            max_rows=1,
            max_bytes=10_000,
            max_critical_safety_bytes=10_000,
        )
        outbox.persist(
            event_type=EVENT_TYPE_ANOMALY,
            destination="/app/device/anomaly",
            payload={"message": "occupies normal quota"},
        )
        retained = PendingCriticalEvent.create(
            event_type=EVENT_TYPE_PRODUCTION,
            destination="/app/device/production",
            payload={"count": 7},
            event_id="334aab50-3cf6-49c4-8362-f3cb26a6994e",
            occurred_at="2026-09-01T00:00:00Z",
        )
        gate = CriticalEventSafetyGate(max_attempts=1, retry_delay_seconds=0)

        def persist_once(candidate: PendingCriticalEvent) -> DurableEvent:
            return outbox.persist(
                event_type=candidate.event_type,
                destination=candidate.destination,
                payload=candidate.payload,
                event_id=candidate.event_id,
                occurred_at=candidate.occurred_at,
            )

        with self.assertRaises(CriticalEventBackpressureError):
            gate.persist(
                retained,
                persist_once,
                lambda candidate, reason: outbox.retain_critical_safety_event(
                    event_type=candidate.event_type,
                    destination=candidate.destination,
                    payload=candidate.payload,
                    event_id=candidate.event_id,
                    occurred_at=candidate.occurred_at,
                    last_error=reason,
                ),
                outbox.clear_critical_safety_event,
            )

        reopened = DurableEventOutbox(
            self.db_path,
            max_rows=1,
            max_bytes=10_000,
            max_critical_safety_bytes=10_000,
        )
        restored = reopened.critical_safety_event()
        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertEqual(restored.event_id, retained.event_id)
        self.assertEqual(restored.payload, retained.payload)
        health = reopened.health_snapshot().to_telemetry()
        self.assertEqual(health["criticalSafetyRows"], 1)
        self.assertEqual(health["capacityState"], "OPERATOR_STOP")

        first_pending = reopened.pending()[0]
        reopened.acknowledge(
            first_pending.event_id,
            first_pending.event_type,
            ACK_STATUS_ACCEPTED,
        )
        restored_gate = CriticalEventSafetyGate(max_attempts=1, retry_delay_seconds=0)
        restored_pending = PendingCriticalEvent.create(
            event_type=restored.event_type,
            destination=restored.destination,
            payload=restored.payload,
            event_id=restored.event_id,
            occurred_at=restored.occurred_at,
        )
        restored_gate.restore_retained(restored_pending, restored.last_error or "")

        self.assertTrue(
            restored_gate.retry_retained(
                lambda candidate: reopened.persist(
                    event_type=candidate.event_type,
                    destination=candidate.destination,
                    payload=candidate.payload,
                    event_id=candidate.event_id,
                    occurred_at=candidate.occurred_at,
                ),
                reopened.clear_critical_safety_event,
            )
        )
        self.assertIsNone(reopened.critical_safety_event())
        self.assertEqual(reopened.pending()[0].event_id, retained.event_id)
        self.assertTrue(restored_gate.is_stopped())

    def test_critical_safety_slot_rejects_oversize_without_bypassing_hard_cap(
        self,
    ) -> None:
        limited = DurableEventOutbox(
            self.db_path,
            max_rows=1,
            max_bytes=10_000,
            max_critical_safety_bytes=256,
        )

        with self.assertRaises(DurableEventCapacityError):
            limited.retain_critical_safety_event(
                event_type=EVENT_TYPE_ANOMALY,
                destination="/app/device/anomaly",
                payload={"message": "x" * 1_000},
                event_id="334aab50-3cf6-49c4-8362-f3cb26a6994e",
                occurred_at="2026-09-01T00:00:00Z",
                last_error="normal quota full",
            )

        self.assertIsNone(limited.critical_safety_event())
        self.assertLessEqual(
            limited.health_snapshot().to_telemetry()["criticalSafetyBytes"],
            256,
        )

    def test_uncorrelated_permanent_error_requires_explicit_identity_absence(
        self,
    ) -> None:
        self.assertTrue(
            is_uncorrelated_permanent_event_rejection(
                {
                    "path": "/app/device/anomaly",
                    "status": 400,
                    "retryable": False,
                    "permanent": True,
                    "eventIdentityAvailable": False,
                }
            )
        )
        self.assertFalse(
            is_uncorrelated_permanent_event_rejection(
                {
                    "path": "/app/device/anomaly",
                    "status": 500,
                    "retryable": True,
                    "eventIdentityAvailable": False,
                }
            )
        )
        self.assertTrue(
            is_uncorrelated_permanent_event_rejection(
                {
                    "path": "/app/device/production",
                    "status": 409,
                    "retryable": False,
                    "terminal": True,
                    "failureClass": "PERMANENT_NO_EVENT_IDENTITY",
                    "eventIdentityAvailable": False,
                }
            )
        )

    def test_persist_send_restart_replay_ack_delete_flow(self) -> None:
        sent_ids: list[str] = []
        delivery = DurableEventDelivery(self.outbox)
        event = delivery.publish(
            event_type=EVENT_TYPE_ANOMALY,
            destination="/app/device/anomaly",
            payload={"anomalyStatus": "DEFECT"},
            event_id="334aab50-3cf6-49c4-8362-f3cb26a6994e",
            occurred_at="2026-08-31T01:02:03Z",
            sender=lambda item: sent_ids.append(item.event_id) is None,
        )
        delivery.mark_sent(event.event_id)

        restarted = DurableEventDelivery(DurableEventOutbox(self.db_path))
        replayed = restarted.replay(
            lambda item: sent_ids.append(item.event_id) is None, limit=10
        )
        ack, removed = restarted.acknowledge_body(
            '{"eventId":"334aab50-3cf6-49c4-8362-f3cb26a6994e",'
            '"eventType":"ANOMALY","status":"DUPLICATE","processedAt":"2026-08-31T01:02:04Z"}'
        )

        self.assertEqual(replayed, 1)
        self.assertEqual(sent_ids, [event.event_id, event.event_id])
        self.assertIsNotNone(ack)
        self.assertTrue(removed)
        self.assertEqual(restarted.outbox.count(), 0)

    def test_replay_never_duplicates_queued_backlog_and_retries_sent_without_ack(
        self,
    ) -> None:
        now = {"value": 0.0}
        sent_ids: list[str] = []
        delivery = DurableEventDelivery(self.outbox, clock=lambda: now["value"])
        event = delivery.publish(
            event_type=EVENT_TYPE_PRODUCTION,
            destination="/app/device/production",
            payload={"count": 1},
            event_id="334aab50-3cf6-49c4-8362-f3cb26a6994e",
            occurred_at="2026-08-31T01:02:03Z",
            sender=lambda item: sent_ids.append(item.event_id) is None,
        )

        now["value"] = 100.0
        delivery.replay(
            lambda item: sent_ids.append(item.event_id) is None,
            limit=10,
            retry_after_seconds=5.0,
        )
        self.assertEqual(sent_ids, [event.event_id])

        delivery.mark_sent(event.event_id)
        now["value"] = 104.0
        delivery.replay(
            lambda item: sent_ids.append(item.event_id) is None,
            limit=10,
            retry_after_seconds=5.0,
        )
        self.assertEqual(sent_ids, [event.event_id])

        now["value"] = 106.0
        delivery.replay(
            lambda item: sent_ids.append(item.event_id) is None,
            limit=10,
            retry_after_seconds=5.0,
        )
        self.assertEqual(sent_ids, [event.event_id, event.event_id])


if __name__ == "__main__":
    unittest.main()
