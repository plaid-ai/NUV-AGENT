from __future__ import annotations

import sqlite3
import stat
import tempfile
import threading
import unittest
from pathlib import Path

from nuvion_app.inference.durable_events import (
    ACK_STATUS_ACCEPTED,
    ACK_STATUS_DUPLICATE,
    ACK_STATUS_REJECTED,
    EVENT_TYPE_ANOMALY,
    EVENT_TYPE_PRODUCTION,
    DurableEventDelivery,
    DurableEventOutbox,
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

        self.assertFalse(self.outbox.acknowledge(event.event_id, EVENT_TYPE_PRODUCTION, ACK_STATUS_ACCEPTED))
        self.assertEqual(self.outbox.count(), 1)
        self.assertTrue(self.outbox.acknowledge(event.event_id, EVENT_TYPE_ANOMALY, ACK_STATUS_DUPLICATE))
        self.assertEqual(self.outbox.count(), 0)

    def test_wait_for_ack_unblocks_clip_finalize_after_server_ack(self) -> None:
        event = self.outbox.persist(
            event_type=EVENT_TYPE_ANOMALY,
            destination="/app/device/anomaly",
            payload={"anomalyStatus": "DEFECT"},
        )
        result: list[bool] = []
        waiter = threading.Thread(target=lambda: result.append(self.outbox.wait_for_ack(event.event_id, 1.0)))
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
        self.assertIsNone(parse_event_ack('{"eventId":"x","eventType":"ANOMALY","status":"FAILED"}'))
        self.assertIsNone(parse_event_ack('{"eventType":"ANOMALY","status":"ACCEPTED"}'))

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

    def test_agent_error_requires_event_id_and_explicit_permanent_classification(self) -> None:
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

        self.assertTrue(self.outbox.acknowledge(event.event_id, EVENT_TYPE_ANOMALY, ACK_STATUS_ACCEPTED))
        self.assertEqual(self.outbox.count(), 0)
        self.assertEqual(self.outbox.dead_letter_count(), 0)

    def test_row_quota_quarantines_new_event_without_evicting_pending(self) -> None:
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
        overflow = DurableEventDelivery(limited).publish(
            event_type=EVENT_TYPE_PRODUCTION,
            destination="/app/device/production",
            payload={"count": 3},
            event_id="334aab50-3cf6-49c4-8362-f3cb26a6994e",
            occurred_at="2026-08-31T01:02:03Z",
            sender=lambda event: sender_calls.append(event.event_id) is None,
        )

        self.assertFalse(first.dead_lettered)
        self.assertFalse(second.dead_lettered)
        self.assertTrue(overflow.dead_lettered)
        self.assertEqual(sender_calls, [])
        self.assertEqual([event.event_id for event in limited.pending()], [first.event_id, second.event_id])
        self.assertEqual(limited.dead_letters()[0].reason, "outbox row quota exceeded")

    def test_byte_quota_quarantines_oversized_new_event(self) -> None:
        limited = DurableEventOutbox(self.db_path, max_rows=100, max_bytes=64)

        overflow = limited.persist(
            event_type=EVENT_TYPE_ANOMALY,
            destination="/app/device/anomaly",
            payload={"message": "x" * 200},
        )

        self.assertTrue(overflow.dead_lettered)
        self.assertEqual(limited.count(), 0)
        self.assertEqual(limited.dead_letter_count(), 1)
        self.assertEqual(limited.dead_letters()[0].rejection_code, "OUTBOX_CAPACITY_EXCEEDED")

    def test_dlq_row_cap_purges_oldest_dead_letter(self) -> None:
        limited = DurableEventOutbox(self.db_path, max_rows=10, max_bytes=1_000_000, max_dead_letters=2)
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
        replayed = restarted.replay(lambda item: sent_ids.append(item.event_id) is None, limit=10)
        ack, removed = restarted.acknowledge_body(
            '{"eventId":"334aab50-3cf6-49c4-8362-f3cb26a6994e",'
            '"eventType":"ANOMALY","status":"DUPLICATE","processedAt":"2026-08-31T01:02:04Z"}'
        )

        self.assertEqual(replayed, 1)
        self.assertEqual(sent_ids, [event.event_id, event.event_id])
        self.assertIsNotNone(ack)
        self.assertTrue(removed)
        self.assertEqual(restarted.outbox.count(), 0)

    def test_replay_never_duplicates_queued_backlog_and_retries_sent_without_ack(self) -> None:
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
        delivery.replay(lambda item: sent_ids.append(item.event_id) is None, limit=10, retry_after_seconds=5.0)
        self.assertEqual(sent_ids, [event.event_id])

        delivery.mark_sent(event.event_id)
        now["value"] = 104.0
        delivery.replay(lambda item: sent_ids.append(item.event_id) is None, limit=10, retry_after_seconds=5.0)
        self.assertEqual(sent_ids, [event.event_id])

        now["value"] = 106.0
        delivery.replay(lambda item: sent_ids.append(item.event_id) is None, limit=10, retry_after_seconds=5.0)
        self.assertEqual(sent_ids, [event.event_id, event.event_id])


if __name__ == "__main__":
    unittest.main()
