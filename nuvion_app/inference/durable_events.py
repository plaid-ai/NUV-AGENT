from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVENT_TYPE_ANOMALY = "ANOMALY"
EVENT_TYPE_PRODUCTION = "PRODUCTION"
EVENT_TYPES = frozenset({EVENT_TYPE_ANOMALY, EVENT_TYPE_PRODUCTION})

ACK_STATUS_ACCEPTED = "ACCEPTED"
ACK_STATUS_DUPLICATE = "DUPLICATE"
ACK_STATUS_REJECTED = "REJECTED"
ACK_STATUSES = frozenset({ACK_STATUS_ACCEPTED, ACK_STATUS_DUPLICATE, ACK_STATUS_REJECTED})
ACK_SUCCESS_STATUSES = frozenset({ACK_STATUS_ACCEPTED, ACK_STATUS_DUPLICATE})

DEFAULT_OUTBOX_MAX_ROWS = 10_000
DEFAULT_OUTBOX_MAX_BYTES = 64 * 1024 * 1024
DEFAULT_DLQ_MAX_ROWS = 10_000

log = logging.getLogger(__name__)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def resolve_default_outbox_path(environ: Mapping[str, str] | None = None) -> Path:
    values = os.environ if environ is None else environ
    explicit = str(values.get("NUVION_EVENT_OUTBOX_PATH") or "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()

    xdg_state_home = str(values.get("XDG_STATE_HOME") or "").strip()
    state_root = Path(xdg_state_home).expanduser() if xdg_state_home else Path.home() / ".local" / "state"
    return (state_root / "nuvion" / "events.sqlite3").resolve()


def _normalize_uuid(value: str | None = None) -> str:
    candidate = str(uuid.uuid4()) if value is None else value
    try:
        return str(uuid.UUID(candidate))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError("event_id must be a UUID string") from exc


@dataclass(frozen=True)
class DurableEvent:
    event_id: str
    event_type: str
    destination: str
    payload: dict[str, Any]
    occurred_at: str
    attempt_count: int = 0
    dead_lettered: bool = False


@dataclass(frozen=True)
class DeadLetterEvent:
    event_id: str
    event_type: str
    destination: str
    payload: dict[str, Any]
    occurred_at: str
    created_at: str
    attempt_count: int
    dead_lettered_at: str
    reason: str
    rejection_code: str
    source: str


@dataclass(frozen=True)
class EventAck:
    event_id: str
    event_type: str
    status: str
    processed_at: str | None
    retryable: bool | None = None
    code: str | None = None
    reason: str | None = None

    @property
    def successful(self) -> bool:
        return self.status in ACK_SUCCESS_STATUSES

    @property
    def permanent_rejection(self) -> bool:
        return self.status == ACK_STATUS_REJECTED and self.retryable is False


@dataclass(frozen=True)
class PermanentEventRejection:
    event_id: str
    event_type: str
    reason: str
    rejection_code: str


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def parse_permanent_event_rejection(payload: Mapping[str, Any]) -> PermanentEventRejection | None:
    if _parse_bool(payload.get("retryable")):
        return None
    path = str(payload.get("path") or "").strip()
    event_type_by_path = {
        "/app/device/anomaly": EVENT_TYPE_ANOMALY,
        "/app/device/production": EVENT_TYPE_PRODUCTION,
    }
    event_type = event_type_by_path.get(path)
    if event_type is None:
        return None
    try:
        event_id = _normalize_uuid(str(payload.get("eventId") or ""))
    except ValueError:
        return None

    status_value = payload.get("status")
    try:
        status = int(status_value)
    except (TypeError, ValueError):
        status = None
    explicitly_permanent = _parse_bool(payload.get("permanent"))
    if not explicitly_permanent and status not in {400, 422}:
        return None
    return PermanentEventRejection(
        event_id=event_id,
        event_type=event_type,
        reason=str(payload.get("detail") or payload.get("message") or "permanent rejection"),
        rejection_code=str(payload.get("code") or "PERMANENT_REJECTION"),
    )


def parse_event_ack(body: str) -> EventAck | None:
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None

    event_type = str(payload.get("eventType") or "").strip().upper()
    status = str(payload.get("status") or "").strip().upper()
    if event_type not in EVENT_TYPES or status not in ACK_STATUSES:
        return None
    try:
        event_id = _normalize_uuid(str(payload.get("eventId") or ""))
    except ValueError:
        return None
    processed_at = str(payload.get("processedAt") or "").strip() or None
    code = str(payload.get("code") or "").strip() or None
    reason = str(payload.get("reason") or payload.get("message") or "").strip() or None
    retryable = _parse_bool(payload["retryable"]) if "retryable" in payload else None
    return EventAck(
        event_id=event_id,
        event_type=event_type,
        status=status,
        processed_at=processed_at,
        retryable=retryable,
        code=code,
        reason=reason,
    )


class DurableEventOutbox:
    def __init__(
        self,
        path: str | Path,
        *,
        max_rows: int = DEFAULT_OUTBOX_MAX_ROWS,
        max_bytes: int = DEFAULT_OUTBOX_MAX_BYTES,
        max_dead_letters: int = DEFAULT_DLQ_MAX_ROWS,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.max_rows = max(1, int(max_rows))
        self.max_bytes = max(1, int(max_bytes))
        self.max_dead_letters = max(1, int(max_dead_letters))
        self._lock = threading.RLock()
        self._ack_condition = threading.Condition(self._lock)
        self._terminal_results: dict[str, bool] = {}
        self._terminal_order: deque[str] = deque(maxlen=2_000)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        parent_existed = self.path.parent.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not parent_existed:
            try:
                os.chmod(self.path.parent, 0o700)
            except OSError:
                pass
        with self._lock, self._connection() as connection:
            connection.execute("PRAGMA journal_mode = DELETE")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS durable_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    destination TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_size_bytes INTEGER NOT NULL DEFAULT 0,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    last_attempt_at TEXT
                )
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(durable_events)").fetchall()
            }
            if "payload_size_bytes" not in columns:
                connection.execute(
                    "ALTER TABLE durable_events ADD COLUMN payload_size_bytes INTEGER NOT NULL DEFAULT 0"
                )
            connection.execute(
                """
                UPDATE durable_events
                SET payload_size_bytes = length(CAST(payload_json AS BLOB))
                WHERE payload_size_bytes <= 0
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS dead_letter_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    event_type TEXT NOT NULL,
                    destination TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_size_bytes INTEGER NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    last_attempt_at TEXT,
                    dead_lettered_at TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    rejection_code TEXT NOT NULL,
                    source TEXT NOT NULL
                )
                """
            )
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def persist(
        self,
        *,
        event_type: str,
        destination: str,
        payload: Mapping[str, Any],
        event_id: str | None = None,
        occurred_at: str | None = None,
    ) -> DurableEvent:
        normalized_type = str(event_type or "").strip().upper()
        if normalized_type not in EVENT_TYPES:
            raise ValueError(f"unsupported event_type: {event_type}")
        normalized_destination = str(destination or "").strip()
        if not normalized_destination.startswith("/app/device/"):
            raise ValueError("critical event destination must be under /app/device/")

        payload_event_id = str(payload.get("eventId") or "").strip() or None
        normalized_id = _normalize_uuid(event_id or payload_event_id)
        if event_id and payload_event_id and _normalize_uuid(event_id) != _normalize_uuid(payload_event_id):
            raise ValueError("event_id does not match payload.eventId")
        normalized_occurred_at = str(occurred_at or payload.get("occurredAt") or utc_now_iso()).strip()
        if not normalized_occurred_at:
            raise ValueError("occurred_at is required")

        durable_payload = json.loads(json.dumps(dict(payload)))
        durable_payload["eventId"] = normalized_id
        durable_payload["occurredAt"] = normalized_occurred_at
        payload_json = json.dumps(durable_payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        created_at = utc_now_iso()
        payload_size_bytes = len(payload_json.encode("utf-8"))

        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM durable_events WHERE event_id = ?",
                (normalized_id,),
            ).fetchone()
            if row is not None:
                event = self._row_to_event(row)
                self._assert_same_event(event, normalized_type, normalized_destination, durable_payload)
                return event

            dead_row = connection.execute(
                "SELECT * FROM dead_letter_events WHERE event_id = ?",
                (normalized_id,),
            ).fetchone()
            if dead_row is not None:
                dead = self._row_to_dead_letter(dead_row)
                event = DurableEvent(
                    event_id=dead.event_id,
                    event_type=dead.event_type,
                    destination=dead.destination,
                    payload=dead.payload,
                    occurred_at=dead.occurred_at,
                    attempt_count=dead.attempt_count,
                    dead_lettered=True,
                )
                self._assert_same_event(event, normalized_type, normalized_destination, durable_payload)
                return event

            stats = connection.execute(
                """
                SELECT COUNT(*) AS row_count,
                       COALESCE(SUM(payload_size_bytes), 0) AS total_bytes
                FROM durable_events
                """
            ).fetchone()
            quota_reason = None
            if int(stats["row_count"]) >= self.max_rows:
                quota_reason = "outbox row quota exceeded"
            elif int(stats["total_bytes"]) + payload_size_bytes > self.max_bytes:
                quota_reason = "outbox byte quota exceeded"

            if quota_reason is not None:
                self._insert_dead_letter_values(
                    connection,
                    event_id=normalized_id,
                    event_type=normalized_type,
                    destination=normalized_destination,
                    payload_json=payload_json,
                    occurred_at=normalized_occurred_at,
                    created_at=created_at,
                    payload_size_bytes=payload_size_bytes,
                    attempt_count=0,
                    last_attempt_at=None,
                    reason=quota_reason,
                    rejection_code="OUTBOX_CAPACITY_EXCEEDED",
                    source="capacity-policy",
                )
                log.error(
                    "outbox capacity exceeded; event quarantined eventId=%s reason=%s",
                    normalized_id,
                    quota_reason,
                )
                self._record_terminal_locked(normalized_id, successful=False)
                self._ack_condition.notify_all()
                return DurableEvent(
                    event_id=normalized_id,
                    event_type=normalized_type,
                    destination=normalized_destination,
                    payload=durable_payload,
                    occurred_at=normalized_occurred_at,
                    dead_lettered=True,
                )

            connection.execute(
                """
                INSERT INTO durable_events (
                    event_id, event_type, destination, payload_json, occurred_at,
                    created_at, payload_size_bytes
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_id,
                    normalized_type,
                    normalized_destination,
                    payload_json,
                    normalized_occurred_at,
                    created_at,
                    payload_size_bytes,
                ),
            )
            row = connection.execute(
                "SELECT * FROM durable_events WHERE event_id = ?",
                (normalized_id,),
            ).fetchone()

        event = self._row_to_event(row)
        self._assert_same_event(event, normalized_type, normalized_destination, durable_payload)
        return event

    def pending(self, limit: int = 200) -> list[DurableEvent]:
        safe_limit = max(1, min(int(limit), 10_000))
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM durable_events
                ORDER BY created_at ASC, rowid ASC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def mark_attempt(self, event_id: str) -> bool:
        normalized_id = _normalize_uuid(event_id)
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE durable_events
                SET attempt_count = attempt_count + 1, last_attempt_at = ?
                WHERE event_id = ?
                """,
                (utc_now_iso(), normalized_id),
            )
            return cursor.rowcount == 1

    def is_pending(self, event_id: str) -> bool:
        try:
            normalized_id = _normalize_uuid(event_id)
        except ValueError:
            return False
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM durable_events WHERE event_id = ?",
                (normalized_id,),
            ).fetchone()
        return row is not None

    def dead_letter(
        self,
        event_id: str,
        event_type: str,
        *,
        reason: str,
        rejection_code: str = "PERMANENT_REJECTION",
        source: str = "server",
    ) -> bool:
        try:
            normalized_id = _normalize_uuid(event_id)
        except ValueError:
            return False
        normalized_type = str(event_type or "").strip().upper()
        if normalized_type not in EVENT_TYPES:
            return False

        with self._ack_condition:
            with self._connection() as connection:
                row = connection.execute(
                    "SELECT * FROM durable_events WHERE event_id = ? AND event_type = ?",
                    (normalized_id, normalized_type),
                ).fetchone()
                if row is None:
                    existing = connection.execute(
                        "SELECT 1 FROM dead_letter_events WHERE event_id = ? AND event_type = ?",
                        (normalized_id, normalized_type),
                    ).fetchone()
                    if existing is None:
                        return False
                    self._record_terminal_locked(normalized_id, successful=False)
                    self._ack_condition.notify_all()
                    return True
                self._insert_dead_letter_values(
                    connection,
                    event_id=str(row["event_id"]),
                    event_type=str(row["event_type"]),
                    destination=str(row["destination"]),
                    payload_json=str(row["payload_json"]),
                    occurred_at=str(row["occurred_at"]),
                    created_at=str(row["created_at"]),
                    payload_size_bytes=int(row["payload_size_bytes"]),
                    attempt_count=int(row["attempt_count"]),
                    last_attempt_at=str(row["last_attempt_at"]) if row["last_attempt_at"] else None,
                    reason=str(reason or "permanent rejection"),
                    rejection_code=str(rejection_code or "PERMANENT_REJECTION"),
                    source=str(source or "server"),
                )
                connection.execute(
                    "DELETE FROM durable_events WHERE event_id = ? AND event_type = ?",
                    (normalized_id, normalized_type),
                )
            self._record_terminal_locked(normalized_id, successful=False)
            self._ack_condition.notify_all()
            return True

    def dead_letters(self, limit: int = 200) -> list[DeadLetterEvent]:
        safe_limit = max(1, min(int(limit), 10_000))
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM dead_letter_events
                ORDER BY id ASC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        return [self._row_to_dead_letter(row) for row in rows]

    def dead_letter_count(self) -> int:
        with self._lock, self._connection() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM dead_letter_events").fetchone()
        return int(row["count"])

    def acknowledge(self, event_id: str, event_type: str, status: str) -> bool:
        try:
            normalized_id = _normalize_uuid(event_id)
        except ValueError:
            return False
        normalized_type = str(event_type or "").strip().upper()
        normalized_status = str(status or "").strip().upper()
        if normalized_type not in EVENT_TYPES or normalized_status not in ACK_SUCCESS_STATUSES:
            return False

        with self._ack_condition:
            with self._connection() as connection:
                cursor = connection.execute(
                    "DELETE FROM durable_events WHERE event_id = ? AND event_type = ?",
                    (normalized_id, normalized_type),
                )
                deleted = cursor.rowcount == 1
            if deleted:
                self._record_terminal_locked(normalized_id, successful=True)
                self._ack_condition.notify_all()
            return deleted

    def wait_for_ack(self, event_id: str, timeout: float) -> bool:
        normalized_id = _normalize_uuid(event_id)
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._ack_condition:
            while normalized_id not in self._terminal_results:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._ack_condition.wait(timeout=remaining)
            return self._terminal_results[normalized_id]

    def count(self) -> int:
        with self._lock, self._connection() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM durable_events").fetchone()
        return int(row["count"])

    def _insert_dead_letter_values(
        self,
        connection: sqlite3.Connection,
        *,
        event_id: str,
        event_type: str,
        destination: str,
        payload_json: str,
        occurred_at: str,
        created_at: str,
        payload_size_bytes: int,
        attempt_count: int,
        last_attempt_at: str | None,
        reason: str,
        rejection_code: str,
        source: str,
    ) -> None:
        count_row = connection.execute("SELECT COUNT(*) AS count FROM dead_letter_events").fetchone()
        if int(count_row["count"]) >= self.max_dead_letters:
            connection.execute(
                """
                DELETE FROM dead_letter_events
                WHERE id = (SELECT id FROM dead_letter_events ORDER BY id ASC LIMIT 1)
                """
            )
            log.error("DLQ row cap reached; purged oldest dead letter")
        connection.execute(
            """
            INSERT OR IGNORE INTO dead_letter_events (
                event_id, event_type, destination, payload_json, occurred_at,
                created_at, payload_size_bytes, attempt_count, last_attempt_at,
                dead_lettered_at, reason, rejection_code, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                event_type,
                destination,
                payload_json,
                occurred_at,
                created_at,
                payload_size_bytes,
                attempt_count,
                last_attempt_at,
                utc_now_iso(),
                reason,
                rejection_code,
                source,
            ),
        )

    def _record_terminal_locked(self, event_id: str, *, successful: bool) -> None:
        if len(self._terminal_order) == self._terminal_order.maxlen:
            oldest = self._terminal_order.popleft()
            self._terminal_results.pop(oldest, None)
        self._terminal_results[event_id] = successful
        self._terminal_order.append(event_id)

    @staticmethod
    def _assert_same_event(
        event: DurableEvent,
        event_type: str,
        destination: str,
        payload: dict[str, Any],
    ) -> None:
        if event.event_type != event_type or event.destination != destination or event.payload != payload:
            raise ValueError(f"event_id collision with different content: {event.event_id}")

    @staticmethod
    def _row_to_event(row: sqlite3.Row | None) -> DurableEvent:
        if row is None:
            raise RuntimeError("durable event row not found")
        return DurableEvent(
            event_id=str(row["event_id"]),
            event_type=str(row["event_type"]),
            destination=str(row["destination"]),
            payload=json.loads(str(row["payload_json"])),
            occurred_at=str(row["occurred_at"]),
            attempt_count=int(row["attempt_count"]),
        )

    @staticmethod
    def _row_to_dead_letter(row: sqlite3.Row) -> DeadLetterEvent:
        return DeadLetterEvent(
            event_id=str(row["event_id"]),
            event_type=str(row["event_type"]),
            destination=str(row["destination"]),
            payload=json.loads(str(row["payload_json"])),
            occurred_at=str(row["occurred_at"]),
            created_at=str(row["created_at"]),
            attempt_count=int(row["attempt_count"]),
            dead_lettered_at=str(row["dead_lettered_at"]),
            reason=str(row["reason"]),
            rejection_code=str(row["rejection_code"]),
            source=str(row["source"]),
        )


class DurableEventDelivery:
    """Coordinates at-least-once delivery while SQLite remains the source of truth."""

    def __init__(self, outbox: DurableEventOutbox, clock: Callable[[], float] | None = None) -> None:
        self.outbox = outbox
        self._clock = clock or time.monotonic
        # None means queued but not yet written to the WebSocket. A timestamp
        # means transport send completed and the application ACK is pending.
        self._inflight: dict[str, float | None] = {}
        self._lock = threading.Lock()

    def reset_for_reconnect(self) -> None:
        with self._lock:
            self._inflight.clear()

    def enqueue(
        self,
        event: DurableEvent,
        sender: Callable[[DurableEvent], bool],
        retry_after_seconds: float | None = None,
    ) -> bool:
        with self._lock:
            if event.event_id in self._inflight:
                sent_at = self._inflight[event.event_id]
                if sent_at is None:
                    return True
                retry_after = max(0.0, float(retry_after_seconds or 0.0))
                if retry_after_seconds is None or self._clock() - sent_at < retry_after:
                    return True
            self._inflight[event.event_id] = None
        sent = False
        try:
            sent = sender(event)
            return sent
        finally:
            if not sent:
                self.release(event.event_id)

    def publish(
        self,
        *,
        event_type: str,
        destination: str,
        payload: Mapping[str, Any],
        event_id: str,
        occurred_at: str,
        sender: Callable[[DurableEvent], bool],
    ) -> DurableEvent:
        event = self.outbox.persist(
            event_type=event_type,
            destination=destination,
            payload=payload,
            event_id=event_id,
            occurred_at=occurred_at,
        )
        if not event.dead_lettered:
            self.enqueue(event, sender)
        return event

    def replay(
        self,
        sender: Callable[[DurableEvent], bool],
        limit: int,
        retry_after_seconds: float = 5.0,
    ) -> int:
        enqueued = 0
        for event in self.outbox.pending(limit=limit):
            if not self.enqueue(event, sender, retry_after_seconds=retry_after_seconds):
                break
            enqueued += 1
        return enqueued

    def mark_sent(self, event_id: str) -> bool:
        marked = self.outbox.mark_attempt(event_id)
        if marked:
            with self._lock:
                if event_id in self._inflight:
                    self._inflight[event_id] = self._clock()
        return marked

    def release(self, event_id: str) -> None:
        with self._lock:
            self._inflight.pop(event_id, None)

    def acknowledge_body(self, body: str) -> tuple[EventAck | None, bool]:
        ack = parse_event_ack(body)
        if ack is None:
            return None, False
        if ack.successful:
            changed = self.outbox.acknowledge(ack.event_id, ack.event_type, ack.status)
        elif ack.permanent_rejection:
            changed = self.outbox.dead_letter(
                ack.event_id,
                ack.event_type,
                reason=ack.reason or "server permanently rejected event",
                rejection_code=ack.code or "PERMANENT_REJECTION",
                source="event.ack",
            )
        else:
            changed = False
        self.release(ack.event_id)
        return ack, changed

    def reject_event(
        self,
        event_id: str,
        event_type: str,
        *,
        reason: str,
        rejection_code: str,
        source: str,
    ) -> bool:
        changed = self.outbox.dead_letter(
            event_id,
            event_type,
            reason=reason,
            rejection_code=rejection_code,
            source=source,
        )
        self.release(event_id)
        return changed

    def wait_for_ack(self, event_id: str, timeout: float) -> bool:
        return self.outbox.wait_for_ack(event_id, timeout)
