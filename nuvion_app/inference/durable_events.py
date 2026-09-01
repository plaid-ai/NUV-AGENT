from __future__ import annotations

import hashlib
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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

EVENT_TYPE_ANOMALY = "ANOMALY"
EVENT_TYPE_PRODUCTION = "PRODUCTION"
EVENT_TYPE_DEVICE_STATE = "DEVICE_STATE"
EVENT_TYPE_CONNECTIVITY = "CONNECTIVITY"
EVENT_TYPE_UPDATE = "UPDATE"
EVENT_TYPE_SECURITY = "SECURITY"
EVENT_TYPE_METRIC = "METRIC"
EVENT_TYPES = frozenset(
    {
        EVENT_TYPE_ANOMALY,
        EVENT_TYPE_PRODUCTION,
        EVENT_TYPE_DEVICE_STATE,
        EVENT_TYPE_CONNECTIVITY,
        EVENT_TYPE_UPDATE,
        EVENT_TYPE_SECURITY,
        EVENT_TYPE_METRIC,
    }
)

DELIVERY_CLASS_CRITICAL = "CRITICAL"
DELIVERY_CLASS_STATE = "STATE"
DELIVERY_CLASS_METRIC = "METRIC"
DELIVERY_CLASSES = frozenset(
    {DELIVERY_CLASS_CRITICAL, DELIVERY_CLASS_STATE, DELIVERY_CLASS_METRIC}
)
DELIVERY_CLASS_PRIORITY = {
    DELIVERY_CLASS_METRIC: 0,
    DELIVERY_CLASS_STATE: 1,
    DELIVERY_CLASS_CRITICAL: 2,
}
DEFAULT_DELIVERY_CLASS_BY_EVENT_TYPE = {
    EVENT_TYPE_ANOMALY: DELIVERY_CLASS_CRITICAL,
    EVENT_TYPE_PRODUCTION: DELIVERY_CLASS_CRITICAL,
    EVENT_TYPE_UPDATE: DELIVERY_CLASS_CRITICAL,
    EVENT_TYPE_SECURITY: DELIVERY_CLASS_CRITICAL,
    EVENT_TYPE_DEVICE_STATE: DELIVERY_CLASS_STATE,
    EVENT_TYPE_CONNECTIVITY: DELIVERY_CLASS_STATE,
    EVENT_TYPE_METRIC: DELIVERY_CLASS_METRIC,
}

ACK_STATUS_ACCEPTED = "ACCEPTED"
ACK_STATUS_DUPLICATE = "DUPLICATE"
ACK_STATUS_REJECTED = "REJECTED"
ACK_STATUSES = frozenset(
    {ACK_STATUS_ACCEPTED, ACK_STATUS_DUPLICATE, ACK_STATUS_REJECTED}
)
ACK_SUCCESS_STATUSES = frozenset({ACK_STATUS_ACCEPTED, ACK_STATUS_DUPLICATE})

DEFAULT_OUTBOX_MAX_ROWS = 10_000
DEFAULT_OUTBOX_MAX_BYTES = 64 * 1024 * 1024
DEFAULT_CRITICAL_SAFETY_MAX_BYTES = 64 * 1024 * 1024
DEFAULT_DLQ_MAX_ROWS = 10_000
DEFAULT_DLQ_MAX_BYTES = 64 * 1024 * 1024
DEFAULT_OUTBOX_MAX_AGE_SECONDS = 30 * 24 * 60 * 60

DELIVERY_STATE_PENDING = "PENDING"
DELIVERY_STATE_DLQ_BLOCKED = "DLQ_BLOCKED"

MAX_DESTINATION_BYTES = 512
MAX_COMPACTION_KEY_BYTES = 256
MAX_OCCURRED_AT_BYTES = 96
MAX_REJECTION_REASON_BYTES = 2_048
MAX_REJECTION_CODE_BYTES = 128
MAX_REJECTION_SOURCE_BYTES = 128

log = logging.getLogger(__name__)


class DurableEventCapacityError(RuntimeError):
    """Raised when neither the outbox nor its DLQ can durably retain an event."""


def utc_now_iso(value: datetime | None = None) -> str:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return (
        current.astimezone(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _utf8_size(*values: Any) -> int:
    return sum(len(str(value).encode("utf-8")) for value in values if value is not None)


def _require_bounded_text(value: Any, *, field: str, max_bytes: int) -> str:
    normalized = str(value or "").strip()
    if len(normalized.encode("utf-8")) > max_bytes:
        raise ValueError(f"{field} exceeds {max_bytes} UTF-8 bytes")
    return normalized


def _truncate_utf8(value: Any, max_bytes: int) -> str:
    raw = str(value or "").strip().encode("utf-8")
    if len(raw) <= max_bytes:
        return raw.decode("utf-8")
    return raw[:max_bytes].decode("utf-8", errors="ignore")


def resolve_default_outbox_path(environ: Mapping[str, str] | None = None) -> Path:
    values = os.environ if environ is None else environ
    explicit = str(values.get("NUVION_EVENT_OUTBOX_PATH") or "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()

    xdg_state_home = str(values.get("XDG_STATE_HOME") or "").strip()
    state_root = (
        Path(xdg_state_home).expanduser()
        if xdg_state_home
        else Path.home() / ".local" / "state"
    )
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
    delivery_class: str = DELIVERY_CLASS_CRITICAL
    compaction_key: str | None = None
    dropped: bool = False
    delivery_state: str = DELIVERY_STATE_PENDING


@dataclass(frozen=True)
class BlockedDurableEvent:
    event_id: str
    event_type: str
    destination: str
    occurred_at: str
    delivery_class: str
    rejection_code: str | None
    rejection_reason: str | None
    rejection_source: str | None
    rejected_at: str | None


@dataclass(frozen=True)
class DurableEventOutboxHealth:
    pending_rows: int
    pending_bytes: int
    oldest_critical_age_seconds: int | None
    dlq_rows: int
    dlq_bytes: int
    blocked_rows: int
    capacity_state: str
    outbox_rows: int
    outbox_bytes: int
    max_rows: int
    max_bytes: int
    dlq_max_rows: int
    dlq_max_bytes: int
    critical_safety_rows: int
    critical_safety_bytes: int
    critical_safety_max_bytes: int

    def to_telemetry(self) -> dict[str, Any]:
        return {
            "pendingRows": self.pending_rows,
            "pendingBytes": self.pending_bytes,
            "oldestCriticalAgeSeconds": self.oldest_critical_age_seconds,
            "dlqRows": self.dlq_rows,
            "dlqBytes": self.dlq_bytes,
            "blockedRows": self.blocked_rows,
            "capacityState": self.capacity_state,
            "outboxRows": self.outbox_rows,
            "outboxBytes": self.outbox_bytes,
            "maxRows": self.max_rows,
            "maxBytes": self.max_bytes,
            "dlqMaxRows": self.dlq_max_rows,
            "dlqMaxBytes": self.dlq_max_bytes,
            "criticalSafetyRows": self.critical_safety_rows,
            "criticalSafetyBytes": self.critical_safety_bytes,
            "criticalSafetyMaxBytes": self.critical_safety_max_bytes,
        }


@dataclass(frozen=True)
class CriticalSafetyEvent:
    event_id: str
    event_type: str
    destination: str
    payload: dict[str, Any]
    occurred_at: str
    retained_at: str
    last_error: str | None


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
    delivery_class: str = DELIVERY_CLASS_CRITICAL
    compaction_key: str | None = None
    payload_sha256: str = ""


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


def _has_explicit_permanent_terminal_classification(
    payload: Mapping[str, Any],
) -> bool:
    """Recognize the BE poison-event terminal contract without guessing by status."""
    if not _parse_bool(payload.get("terminal")):
        return False
    failure_class = str(payload.get("failureClass") or "").strip().upper()
    return failure_class in {"PERMANENT", "PERMANENT_NO_EVENT_IDENTITY"}


def parse_permanent_event_rejection(
    payload: Mapping[str, Any],
) -> PermanentEventRejection | None:
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
    explicitly_permanent = _parse_bool(
        payload.get("permanent")
    ) or _has_explicit_permanent_terminal_classification(payload)
    if not explicitly_permanent and status not in {400, 422}:
        return None
    return PermanentEventRejection(
        event_id=event_id,
        event_type=event_type,
        reason=str(
            payload.get("detail") or payload.get("message") or "permanent rejection"
        ),
        rejection_code=str(payload.get("code") or "PERMANENT_REJECTION"),
    )


def is_uncorrelated_permanent_event_rejection(payload: Mapping[str, Any]) -> bool:
    """Recognize a poison-event protocol stop that cannot identify a durable row."""
    if _parse_bool(payload.get("retryable")):
        return False
    if payload.get("eventIdentityAvailable") is not False:
        return False
    path = str(payload.get("path") or "").strip()
    if path not in {"/app/device/anomaly", "/app/device/production"}:
        return False
    try:
        status = int(payload.get("status"))
    except (TypeError, ValueError):
        status = None
    return (
        _parse_bool(payload.get("permanent"))
        or _has_explicit_permanent_terminal_classification(payload)
        or status in {400, 422}
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


def _outbox_record_size(
    *,
    event_id: str,
    event_type: str,
    destination: str,
    payload_json: str,
    occurred_at: str,
    created_at: str,
    delivery_class: str,
    compaction_key: str | None,
    attempt_count: int = 0,
    last_attempt_at: str | None = None,
    delivery_state: str = DELIVERY_STATE_PENDING,
    rejection_code: str | None = None,
    rejection_reason: str | None = None,
    rejection_source: str | None = None,
    rejected_at: str | None = None,
) -> int:
    # Reserve the longest delivery-state token so PENDING -> DLQ_BLOCKED can
    # always be persisted without exceeding an already-full logical quota.
    delivery_state_size = max(
        _utf8_size(delivery_state),
        _utf8_size(DELIVERY_STATE_DLQ_BLOCKED),
    )
    mutable_attempt_size = max(_utf8_size(attempt_count), 20)
    mutable_last_attempt_size = max(_utf8_size(last_attempt_at), MAX_OCCURRED_AT_BYTES)
    return (
        _utf8_size(
            event_id,
            event_type,
            destination,
            payload_json,
            occurred_at,
            created_at,
            delivery_class,
            compaction_key,
            rejection_code,
            rejection_reason,
            rejection_source,
            rejected_at,
        )
        + delivery_state_size
        + mutable_attempt_size
        + mutable_last_attempt_size
    )


def _dead_letter_record_size(
    *,
    event_id: str,
    event_type: str,
    destination: str,
    payload_json: str,
    occurred_at: str,
    created_at: str,
    delivery_class: str,
    compaction_key: str | None,
    payload_sha256: str,
    attempt_count: int,
    last_attempt_at: str | None,
    dead_lettered_at: str,
    reason: str,
    rejection_code: str,
    source: str,
) -> int:
    return _utf8_size(
        event_id,
        event_type,
        destination,
        payload_json,
        occurred_at,
        created_at,
        delivery_class,
        compaction_key,
        payload_sha256,
        attempt_count,
        last_attempt_at,
        dead_lettered_at,
        reason,
        rejection_code,
        source,
    )


def _critical_safety_record_size(
    *,
    event_id: str,
    event_type: str,
    destination: str,
    payload_json: str,
    occurred_at: str,
    retained_at: str,
    last_error: str | None,
) -> int:
    return _utf8_size(
        1,
        event_id,
        event_type,
        destination,
        payload_json,
        occurred_at,
        retained_at,
        last_error,
    )


class DurableEventOutbox:
    def __init__(
        self,
        path: str | Path,
        *,
        max_rows: int = DEFAULT_OUTBOX_MAX_ROWS,
        max_bytes: int = DEFAULT_OUTBOX_MAX_BYTES,
        max_dead_letters: int = DEFAULT_DLQ_MAX_ROWS,
        max_dead_letter_bytes: int = DEFAULT_DLQ_MAX_BYTES,
        max_critical_safety_bytes: int | None = None,
        max_age_seconds: int = DEFAULT_OUTBOX_MAX_AGE_SECONDS,
        wall_clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.max_rows = max(1, int(max_rows))
        self.max_bytes = max(1, int(max_bytes))
        self.max_dead_letters = max(1, int(max_dead_letters))
        self.max_dead_letter_bytes = max(1, int(max_dead_letter_bytes))
        self.max_critical_safety_bytes = max(
            1,
            int(
                max_critical_safety_bytes
                if max_critical_safety_bytes is not None
                else self.max_bytes
            ),
        )
        self.max_age_seconds = max(1, int(max_age_seconds))
        self._wall_clock = wall_clock or (lambda: datetime.now(timezone.utc))
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
    def _connection(
        self,
        *,
        immediate: bool = False,
        configure_journal: bool = False,
        terminal_updates: list[tuple[str, bool]] | None = None,
    ):
        connection = self._connect()
        committed = False
        try:
            if configure_journal:
                connection.execute("PRAGMA journal_mode = DELETE")
                connection.execute("PRAGMA synchronous = FULL")
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield connection
            connection.commit()
            committed = True
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        if committed and terminal_updates:
            self._apply_terminal_updates_locked(terminal_updates)

    def _initialize(self) -> None:
        parent_existed = self.path.parent.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not parent_existed:
            try:
                os.chmod(self.path.parent, 0o700)
            except OSError:
                pass
        with (
            self._lock,
            self._connection(immediate=True, configure_journal=True) as connection,
        ):
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
                    record_size_bytes INTEGER NOT NULL DEFAULT 0,
                    delivery_class TEXT NOT NULL DEFAULT 'CRITICAL',
                    compaction_key TEXT,
                    delivery_state TEXT NOT NULL DEFAULT 'PENDING',
                    rejection_code TEXT,
                    rejection_reason TEXT,
                    rejection_source TEXT,
                    rejected_at TEXT,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    last_attempt_at TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS critical_safety_slot (
                    slot_id INTEGER PRIMARY KEY CHECK (slot_id = 1),
                    event_id TEXT NOT NULL UNIQUE,
                    event_type TEXT NOT NULL,
                    destination TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    retained_at TEXT NOT NULL,
                    last_error TEXT,
                    record_size_bytes INTEGER NOT NULL
                )
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(durable_events)"
                ).fetchall()
            }
            if "payload_size_bytes" not in columns:
                connection.execute(
                    "ALTER TABLE durable_events ADD COLUMN payload_size_bytes INTEGER NOT NULL DEFAULT 0"
                )
            if "record_size_bytes" not in columns:
                connection.execute(
                    "ALTER TABLE durable_events ADD COLUMN record_size_bytes INTEGER NOT NULL DEFAULT 0"
                )
            if "delivery_class" not in columns:
                connection.execute(
                    "ALTER TABLE durable_events ADD COLUMN delivery_class TEXT NOT NULL DEFAULT 'CRITICAL'"
                )
            if "compaction_key" not in columns:
                connection.execute(
                    "ALTER TABLE durable_events ADD COLUMN compaction_key TEXT"
                )
            if "delivery_state" not in columns:
                connection.execute(
                    "ALTER TABLE durable_events ADD COLUMN delivery_state TEXT NOT NULL DEFAULT 'PENDING'"
                )
            for column in (
                "rejection_code",
                "rejection_reason",
                "rejection_source",
                "rejected_at",
            ):
                if column not in columns:
                    connection.execute(
                        f"ALTER TABLE durable_events ADD COLUMN {column} TEXT"
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
                    record_size_bytes INTEGER NOT NULL DEFAULT 0,
                    delivery_class TEXT NOT NULL DEFAULT 'CRITICAL',
                    compaction_key TEXT,
                    payload_sha256 TEXT NOT NULL DEFAULT '',
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    last_attempt_at TEXT,
                    dead_lettered_at TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    rejection_code TEXT NOT NULL,
                    source TEXT NOT NULL
                )
                """
            )
            dead_letter_columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(dead_letter_events)"
                ).fetchall()
            }
            if "delivery_class" not in dead_letter_columns:
                connection.execute(
                    "ALTER TABLE dead_letter_events ADD COLUMN delivery_class TEXT NOT NULL DEFAULT 'CRITICAL'"
                )
            if "record_size_bytes" not in dead_letter_columns:
                connection.execute(
                    "ALTER TABLE dead_letter_events ADD COLUMN record_size_bytes INTEGER NOT NULL DEFAULT 0"
                )
            if "compaction_key" not in dead_letter_columns:
                connection.execute(
                    "ALTER TABLE dead_letter_events ADD COLUMN compaction_key TEXT"
                )
            if "payload_sha256" not in dead_letter_columns:
                connection.execute(
                    "ALTER TABLE dead_letter_events ADD COLUMN payload_sha256 TEXT NOT NULL DEFAULT ''"
                )
            unhashed_dead_letters = connection.execute(
                "SELECT id, payload_json FROM dead_letter_events WHERE payload_sha256 = ''"
            ).fetchall()
            for dead_letter in unhashed_dead_letters:
                connection.execute(
                    "UPDATE dead_letter_events SET payload_sha256 = ? WHERE id = ?",
                    (
                        hashlib.sha256(
                            str(dead_letter["payload_json"]).encode("utf-8")
                        ).hexdigest(),
                        int(dead_letter["id"]),
                    ),
                )
            unmeasured_events = connection.execute(
                "SELECT * FROM durable_events WHERE record_size_bytes <= 0"
            ).fetchall()
            for event in unmeasured_events:
                record_size_bytes = _outbox_record_size(
                    event_id=str(event["event_id"]),
                    event_type=str(event["event_type"]),
                    destination=str(event["destination"]),
                    payload_json=str(event["payload_json"]),
                    occurred_at=str(event["occurred_at"]),
                    created_at=str(event["created_at"]),
                    delivery_class=str(
                        event["delivery_class"] or DELIVERY_CLASS_CRITICAL
                    ),
                    compaction_key=str(event["compaction_key"])
                    if event["compaction_key"]
                    else None,
                    attempt_count=int(event["attempt_count"]),
                    last_attempt_at=str(event["last_attempt_at"])
                    if event["last_attempt_at"]
                    else None,
                    delivery_state=str(
                        event["delivery_state"] or DELIVERY_STATE_PENDING
                    ),
                    rejection_code=str(event["rejection_code"])
                    if event["rejection_code"]
                    else None,
                    rejection_reason=str(event["rejection_reason"])
                    if event["rejection_reason"]
                    else None,
                    rejection_source=str(event["rejection_source"])
                    if event["rejection_source"]
                    else None,
                    rejected_at=str(event["rejected_at"])
                    if event["rejected_at"]
                    else None,
                )
                connection.execute(
                    "UPDATE durable_events SET record_size_bytes = ? WHERE event_id = ?",
                    (record_size_bytes, str(event["event_id"])),
                )
            unmeasured_dead_letters = connection.execute(
                "SELECT * FROM dead_letter_events WHERE record_size_bytes <= 0"
            ).fetchall()
            for dead_letter in unmeasured_dead_letters:
                record_size_bytes = _dead_letter_record_size(
                    event_id=str(dead_letter["event_id"]),
                    event_type=str(dead_letter["event_type"]),
                    destination=str(dead_letter["destination"]),
                    payload_json=str(dead_letter["payload_json"]),
                    occurred_at=str(dead_letter["occurred_at"]),
                    created_at=str(dead_letter["created_at"]),
                    delivery_class=str(
                        dead_letter["delivery_class"] or DELIVERY_CLASS_CRITICAL
                    ),
                    compaction_key=(
                        str(dead_letter["compaction_key"])
                        if dead_letter["compaction_key"]
                        else None
                    ),
                    payload_sha256=str(dead_letter["payload_sha256"]),
                    attempt_count=int(dead_letter["attempt_count"]),
                    last_attempt_at=(
                        str(dead_letter["last_attempt_at"])
                        if dead_letter["last_attempt_at"]
                        else None
                    ),
                    dead_lettered_at=str(dead_letter["dead_lettered_at"]),
                    reason=str(dead_letter["reason"]),
                    rejection_code=str(dead_letter["rejection_code"]),
                    source=str(dead_letter["source"]),
                )
                connection.execute(
                    "UPDATE dead_letter_events SET record_size_bytes = ? WHERE id = ?",
                    (record_size_bytes, int(dead_letter["id"])),
                )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_durable_events_delivery_created
                ON durable_events(delivery_class, created_at)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_durable_events_compaction
                ON durable_events(delivery_class, compaction_key)
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
        delivery_class: str | None = None,
        compaction_key: str | None = None,
    ) -> DurableEvent:
        normalized_type = str(event_type or "").strip().upper()
        if normalized_type not in EVENT_TYPES:
            raise ValueError(f"unsupported event_type: {event_type}")
        default_delivery_class = DEFAULT_DELIVERY_CLASS_BY_EVENT_TYPE[normalized_type]
        normalized_delivery_class = (
            str(delivery_class or default_delivery_class).strip().upper()
        )
        if normalized_delivery_class not in DELIVERY_CLASSES:
            raise ValueError(f"unsupported delivery_class: {delivery_class}")
        if (
            DELIVERY_CLASS_PRIORITY[normalized_delivery_class]
            < DELIVERY_CLASS_PRIORITY[default_delivery_class]
        ):
            raise ValueError(
                f"event_type {normalized_type} cannot be downgraded from "
                f"{default_delivery_class} to {normalized_delivery_class}"
            )
        normalized_destination = _require_bounded_text(
            destination,
            field="destination",
            max_bytes=MAX_DESTINATION_BYTES,
        )
        if not normalized_destination.startswith("/app/device/"):
            raise ValueError("event destination must be under /app/device/")
        normalized_compaction_key = (
            _require_bounded_text(
                compaction_key,
                field="compaction_key",
                max_bytes=MAX_COMPACTION_KEY_BYTES,
            )
            or None
        )
        if (
            normalized_delivery_class == DELIVERY_CLASS_STATE
            and normalized_compaction_key is None
        ):
            normalized_compaction_key = normalized_destination

        payload_event_id = str(payload.get("eventId") or "").strip() or None
        normalized_id = _normalize_uuid(event_id or payload_event_id)
        if (
            event_id
            and payload_event_id
            and _normalize_uuid(event_id) != _normalize_uuid(payload_event_id)
        ):
            raise ValueError("event_id does not match payload.eventId")
        normalized_occurred_at = _require_bounded_text(
            occurred_at or payload.get("occurredAt") or utc_now_iso(),
            field="occurred_at",
            max_bytes=MAX_OCCURRED_AT_BYTES,
        )
        if not normalized_occurred_at:
            raise ValueError("occurred_at is required")

        durable_payload = json.loads(json.dumps(dict(payload)))
        durable_payload["eventId"] = normalized_id
        durable_payload["occurredAt"] = normalized_occurred_at
        payload_json = json.dumps(
            durable_payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        payload_sha256 = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        created_at = utc_now_iso(self._wall_clock())
        payload_size_bytes = len(payload_json.encode("utf-8"))
        record_size_bytes = _outbox_record_size(
            event_id=normalized_id,
            event_type=normalized_type,
            destination=normalized_destination,
            payload_json=payload_json,
            occurred_at=normalized_occurred_at,
            created_at=created_at,
            delivery_class=normalized_delivery_class,
            compaction_key=normalized_compaction_key,
        )

        terminal_updates: list[tuple[str, bool]] = []
        with (
            self._lock,
            self._connection(
                immediate=True,
                terminal_updates=terminal_updates,
            ) as connection,
        ):
            row = connection.execute(
                "SELECT * FROM durable_events WHERE event_id = ?",
                (normalized_id,),
            ).fetchone()
            if row is not None:
                event = self._row_to_event(row)
                self._assert_same_event(
                    event,
                    normalized_type,
                    normalized_destination,
                    durable_payload,
                    normalized_delivery_class,
                    normalized_compaction_key,
                )
                terminal_updates.extend(
                    self._prune_expired_locked(
                        connection, exclude_event_ids={normalized_id}
                    )
                )
                return event

            dead_row = connection.execute(
                "SELECT * FROM dead_letter_events WHERE event_id = ?",
                (normalized_id,),
            ).fetchone()
            if dead_row is not None:
                dead = self._row_to_dead_letter(dead_row)
                self._assert_same_dead_letter(
                    dead,
                    event_type=normalized_type,
                    destination=normalized_destination,
                    occurred_at=normalized_occurred_at,
                    payload_sha256=payload_sha256,
                    delivery_class=normalized_delivery_class,
                    compaction_key=normalized_compaction_key,
                )
                event = DurableEvent(
                    event_id=dead.event_id,
                    event_type=dead.event_type,
                    destination=dead.destination,
                    payload=dead.payload,
                    occurred_at=dead.occurred_at,
                    attempt_count=dead.attempt_count,
                    dead_lettered=True,
                    delivery_class=dead.delivery_class,
                    compaction_key=dead.compaction_key,
                )
                terminal_updates.extend(self._prune_expired_locked(connection))
                return event

            terminal_updates.extend(self._prune_expired_locked(connection))

            replacement_rows: list[sqlite3.Row] = []
            if (
                normalized_delivery_class == DELIVERY_CLASS_STATE
                and normalized_compaction_key
            ):
                replacement_rows = connection.execute(
                    """
                    SELECT event_id FROM durable_events
                    WHERE delivery_class = ? AND compaction_key = ? AND delivery_state = ?
                    """,
                    (
                        normalized_delivery_class,
                        normalized_compaction_key,
                        DELIVERY_STATE_PENDING,
                    ),
                ).fetchall()
            replacement_ids = {
                str(replacement["event_id"]) for replacement in replacement_rows
            }

            quota_reason, planned_evictions = self._plan_capacity_locked(
                connection,
                incoming_class=normalized_delivery_class,
                incoming_size_bytes=record_size_bytes,
                replacement_ids=replacement_ids,
            )

            if quota_reason is not None:
                if normalized_delivery_class == DELIVERY_CLASS_CRITICAL:
                    raise DurableEventCapacityError(
                        f"{quota_reason}; critical eventId={normalized_id} requires backpressure"
                    )
                log.warning(
                    "%s dropped by outbox capacity policy eventId=%s reason=%s",
                    normalized_delivery_class.lower(),
                    normalized_id,
                    quota_reason,
                )
                return DurableEvent(
                    event_id=normalized_id,
                    event_type=normalized_type,
                    destination=normalized_destination,
                    payload=durable_payload,
                    occurred_at=normalized_occurred_at,
                    delivery_class=normalized_delivery_class,
                    compaction_key=normalized_compaction_key,
                    dropped=True,
                )

            superseded_ids: set[str] = set()
            for replacement_id in replacement_ids:
                connection.execute(
                    "DELETE FROM durable_events WHERE event_id = ?",
                    (replacement_id,),
                )
                superseded_ids.add(replacement_id)
            for evicted_id, evicted_class in planned_evictions:
                connection.execute(
                    "DELETE FROM durable_events WHERE event_id = ?",
                    (evicted_id,),
                )
                superseded_ids.add(evicted_id)
            connection.execute(
                """
                INSERT INTO durable_events (
                    event_id, event_type, destination, payload_json, occurred_at,
                    created_at, payload_size_bytes, record_size_bytes, delivery_class,
                    compaction_key, delivery_state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_id,
                    normalized_type,
                    normalized_destination,
                    payload_json,
                    normalized_occurred_at,
                    created_at,
                    payload_size_bytes,
                    record_size_bytes,
                    normalized_delivery_class,
                    normalized_compaction_key,
                    DELIVERY_STATE_PENDING,
                ),
            )
            for evicted_id, evicted_class in planned_evictions:
                log.warning(
                    "outbox capacity policy evicted %s eventId=%s for incoming=%s",
                    evicted_class,
                    evicted_id,
                    normalized_delivery_class,
                )
            for superseded_id in superseded_ids:
                terminal_updates.append((superseded_id, False))
            row = connection.execute(
                "SELECT * FROM durable_events WHERE event_id = ?",
                (normalized_id,),
            ).fetchone()

        event = self._row_to_event(row)
        self._assert_same_event(
            event,
            normalized_type,
            normalized_destination,
            durable_payload,
            normalized_delivery_class,
            normalized_compaction_key,
        )
        return event

    def pending(self, limit: int = 200) -> list[DurableEvent]:
        safe_limit = max(1, min(int(limit), 10_000))
        terminal_updates: list[tuple[str, bool]] = []
        with (
            self._lock,
            self._connection(
                immediate=True,
                terminal_updates=terminal_updates,
            ) as connection,
        ):
            terminal_updates.extend(self._prune_expired_locked(connection))
            rows = connection.execute(
                """
                SELECT * FROM durable_events
                WHERE delivery_state = ?
                ORDER BY CASE delivery_class
                    WHEN 'CRITICAL' THEN 0
                    WHEN 'STATE' THEN 1
                    ELSE 2
                END ASC, created_at ASC, rowid ASC
                LIMIT ?
                """,
                (DELIVERY_STATE_PENDING, safe_limit),
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def mark_attempt(self, event_id: str) -> bool:
        normalized_id = _normalize_uuid(event_id)
        terminal_updates: list[tuple[str, bool]] = []
        with (
            self._lock,
            self._connection(
                immediate=True,
                terminal_updates=terminal_updates,
            ) as connection,
        ):
            row = connection.execute(
                "SELECT * FROM durable_events WHERE event_id = ? AND delivery_state = ?",
                (normalized_id, DELIVERY_STATE_PENDING),
            ).fetchone()
            if row is None:
                return False
            delivery_class = str(row["delivery_class"] or DELIVERY_CLASS_CRITICAL)
            if delivery_class != DELIVERY_CLASS_CRITICAL:
                cursor = connection.execute(
                    "DELETE FROM durable_events WHERE event_id = ? AND delivery_state = ?",
                    (normalized_id, DELIVERY_STATE_PENDING),
                )
                if cursor.rowcount == 1:
                    terminal_updates.append((normalized_id, True))
                    return True
                return False
            last_attempt_at = utc_now_iso(self._wall_clock())
            attempt_count = int(row["attempt_count"]) + 1
            record_size_bytes = self._outbox_record_size_from_row(
                row,
                attempt_count=attempt_count,
                last_attempt_at=last_attempt_at,
            )
            cursor = connection.execute(
                """
                UPDATE durable_events SET attempt_count = ?, last_attempt_at = ?,
                    record_size_bytes = ?
                WHERE event_id = ? AND delivery_state = ?
                """,
                (
                    attempt_count,
                    last_attempt_at,
                    record_size_bytes,
                    normalized_id,
                    DELIVERY_STATE_PENDING,
                ),
            )
            return cursor.rowcount == 1

    def is_pending(self, event_id: str) -> bool:
        try:
            normalized_id = _normalize_uuid(event_id)
        except ValueError:
            return False
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM durable_events WHERE event_id = ? AND delivery_state = ?",
                (normalized_id, DELIVERY_STATE_PENDING),
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
        normalized_reason = _truncate_utf8(
            reason or "permanent rejection",
            MAX_REJECTION_REASON_BYTES,
        )
        normalized_code = _truncate_utf8(
            rejection_code or "PERMANENT_REJECTION",
            MAX_REJECTION_CODE_BYTES,
        )
        normalized_source = _truncate_utf8(
            source or "server", MAX_REJECTION_SOURCE_BYTES
        )

        terminal_updates: list[tuple[str, bool]] = []
        with self._ack_condition:
            with self._connection(
                immediate=True,
                terminal_updates=terminal_updates,
            ) as connection:
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
                    terminal_updates.append((normalized_id, False))
                    return True
                payload_json = str(row["payload_json"])
                moved_to_dead_letter = self._insert_dead_letter_values(
                    connection,
                    event_id=str(row["event_id"]),
                    event_type=str(row["event_type"]),
                    destination=str(row["destination"]),
                    payload_json=payload_json,
                    occurred_at=str(row["occurred_at"]),
                    created_at=str(row["created_at"]),
                    payload_size_bytes=int(row["payload_size_bytes"]),
                    attempt_count=int(row["attempt_count"]),
                    last_attempt_at=str(row["last_attempt_at"])
                    if row["last_attempt_at"]
                    else None,
                    reason=normalized_reason,
                    rejection_code=normalized_code,
                    source=normalized_source,
                    delivery_class=str(
                        row["delivery_class"] or DELIVERY_CLASS_CRITICAL
                    ),
                    compaction_key=str(row["compaction_key"])
                    if row["compaction_key"]
                    else None,
                    payload_sha256=hashlib.sha256(
                        payload_json.encode("utf-8")
                    ).hexdigest(),
                )
                if not moved_to_dead_letter:
                    self._mark_dlq_blocked_locked(
                        connection,
                        row,
                        rejection_code=normalized_code,
                        rejection_reason=normalized_reason,
                        rejection_source=normalized_source,
                    )
                    terminal_updates.append((normalized_id, False))
                    log.error(
                        "DLQ cannot retain permanently rejected event; replay blocked eventId=%s",
                        normalized_id,
                    )
                    return False
                connection.execute(
                    "DELETE FROM durable_events WHERE event_id = ? AND event_type = ?",
                    (normalized_id, normalized_type),
                )
                terminal_updates.append((normalized_id, False))
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
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM dead_letter_events"
            ).fetchone()
        return int(row["count"])

    def dead_letter_payload_bytes(self) -> int:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT COALESCE(SUM(payload_size_bytes), 0) AS total_bytes FROM dead_letter_events"
            ).fetchone()
        return int(row["total_bytes"])

    def dead_letter_record_bytes(self) -> int:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT COALESCE(SUM(record_size_bytes), 0) AS total_bytes "
                "FROM dead_letter_events"
            ).fetchone()
        return int(row["total_bytes"])

    def blocked_events(self, limit: int = 200) -> list[BlockedDurableEvent]:
        safe_limit = max(1, min(int(limit), 10_000))
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM durable_events
                WHERE delivery_state = ?
                ORDER BY rejected_at ASC, created_at ASC, rowid ASC
                LIMIT ?
                """,
                (DELIVERY_STATE_DLQ_BLOCKED, safe_limit),
            ).fetchall()
        return [self._row_to_blocked_event(row) for row in rows]

    def blocked_count(self) -> int:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM durable_events WHERE delivery_state = ?",
                (DELIVERY_STATE_DLQ_BLOCKED,),
            ).fetchone()
        return int(row["count"])

    def acknowledge(self, event_id: str, event_type: str, status: str) -> bool:
        try:
            normalized_id = _normalize_uuid(event_id)
        except ValueError:
            return False
        normalized_type = str(event_type or "").strip().upper()
        normalized_status = str(status or "").strip().upper()
        if (
            normalized_type not in EVENT_TYPES
            or normalized_status not in ACK_SUCCESS_STATUSES
        ):
            return False

        terminal_updates: list[tuple[str, bool]] = []
        with self._ack_condition:
            with self._connection(
                immediate=True,
                terminal_updates=terminal_updates,
            ) as connection:
                cursor = connection.execute(
                    "DELETE FROM durable_events WHERE event_id = ? AND event_type = ?",
                    (normalized_id, normalized_type),
                )
                deleted = cursor.rowcount == 1
                if deleted:
                    terminal_updates.append((normalized_id, True))
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

    def terminal_result(self, event_id: str) -> bool | None:
        try:
            normalized_id = _normalize_uuid(event_id)
        except ValueError:
            return None
        with self._ack_condition:
            return self._terminal_results.get(normalized_id)

    def count(self) -> int:
        terminal_updates: list[tuple[str, bool]] = []
        with (
            self._lock,
            self._connection(
                immediate=True,
                terminal_updates=terminal_updates,
            ) as connection,
        ):
            terminal_updates.extend(self._prune_expired_locked(connection))
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM durable_events"
            ).fetchone()
        return int(row["count"])

    def record_bytes(self) -> int:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT COALESCE(SUM(record_size_bytes), 0) AS total_bytes "
                "FROM durable_events"
            ).fetchone()
        return int(row["total_bytes"])

    def retain_critical_safety_event(
        self,
        *,
        event_type: str,
        destination: str,
        payload: Mapping[str, Any],
        event_id: str,
        occurred_at: str,
        last_error: str,
    ) -> CriticalSafetyEvent:
        """Durably reserve the single fail-closed slot outside normal outbox quota."""
        normalized_type = str(event_type or "").strip().upper()
        if (
            DEFAULT_DELIVERY_CLASS_BY_EVENT_TYPE.get(normalized_type)
            != DELIVERY_CLASS_CRITICAL
        ):
            raise ValueError("critical safety slot only accepts CRITICAL event types")
        normalized_id = _normalize_uuid(event_id)
        normalized_destination = _require_bounded_text(
            destination,
            field="destination",
            max_bytes=MAX_DESTINATION_BYTES,
        )
        normalized_occurred_at = _require_bounded_text(
            occurred_at,
            field="occurred_at",
            max_bytes=MAX_OCCURRED_AT_BYTES,
        )
        payload_json = json.dumps(
            dict(payload),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        retained_at = utc_now_iso(self._wall_clock())
        normalized_error: str | None = (
            _truncate_utf8(
                last_error,
                MAX_REJECTION_REASON_BYTES,
            )
            or None
        )
        record_size_bytes = _critical_safety_record_size(
            event_id=normalized_id,
            event_type=normalized_type,
            destination=normalized_destination,
            payload_json=payload_json,
            occurred_at=normalized_occurred_at,
            retained_at=retained_at,
            last_error=normalized_error,
        )
        if record_size_bytes > self.max_critical_safety_bytes:
            normalized_error = None
            record_size_bytes = _critical_safety_record_size(
                event_id=normalized_id,
                event_type=normalized_type,
                destination=normalized_destination,
                payload_json=payload_json,
                occurred_at=normalized_occurred_at,
                retained_at=retained_at,
                last_error=None,
            )
        if record_size_bytes > self.max_critical_safety_bytes:
            raise DurableEventCapacityError(
                "critical event exceeds the hard safety-slot byte quota"
            )

        with self._lock, self._connection(immediate=True) as connection:
            existing = connection.execute(
                "SELECT * FROM critical_safety_slot WHERE slot_id = 1"
            ).fetchone()
            if existing is not None:
                same_event = (
                    str(existing["event_id"]) == normalized_id
                    and str(existing["event_type"]) == normalized_type
                    and str(existing["destination"]) == normalized_destination
                    and str(existing["payload_json"]) == payload_json
                    and str(existing["occurred_at"]) == normalized_occurred_at
                )
                if not same_event:
                    raise DurableEventCapacityError(
                        "critical safety slot is already occupied by another event"
                    )
                return self._row_to_critical_safety_event(existing)
            connection.execute(
                """
                INSERT INTO critical_safety_slot (
                    slot_id, event_id, event_type, destination, payload_json,
                    occurred_at, retained_at, last_error, record_size_bytes
                ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_id,
                    normalized_type,
                    normalized_destination,
                    payload_json,
                    normalized_occurred_at,
                    retained_at,
                    normalized_error,
                    record_size_bytes,
                ),
            )
            row = connection.execute(
                "SELECT * FROM critical_safety_slot WHERE slot_id = 1"
            ).fetchone()
        return self._row_to_critical_safety_event(row)

    def critical_safety_event(self) -> CriticalSafetyEvent | None:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM critical_safety_slot WHERE slot_id = 1"
            ).fetchone()
        return self._row_to_critical_safety_event(row) if row is not None else None

    def clear_critical_safety_event(self, event_id: str) -> bool:
        normalized_id = _normalize_uuid(event_id)
        with self._lock, self._connection(immediate=True) as connection:
            cursor = connection.execute(
                "DELETE FROM critical_safety_slot WHERE slot_id = 1 AND event_id = ?",
                (normalized_id,),
            )
            return cursor.rowcount == 1

    def health_snapshot(self) -> DurableEventOutboxHealth:
        """Return a bounded operational projection after applying the age policy."""
        terminal_updates: list[tuple[str, bool]] = []
        with (
            self._lock,
            self._connection(
                immediate=True,
                terminal_updates=terminal_updates,
            ) as connection,
        ):
            terminal_updates.extend(self._prune_expired_locked(connection))
            outbox = connection.execute(
                """
                SELECT
                    COUNT(*) AS outbox_rows,
                    COALESCE(SUM(record_size_bytes), 0) AS outbox_bytes,
                    COALESCE(SUM(CASE WHEN delivery_state = ? THEN 1 ELSE 0 END), 0)
                        AS pending_rows,
                    COALESCE(SUM(CASE WHEN delivery_state = ? THEN record_size_bytes ELSE 0 END), 0)
                        AS pending_bytes,
                    COALESCE(SUM(CASE WHEN delivery_state = ? THEN 1 ELSE 0 END), 0)
                        AS blocked_rows,
                    MIN(CASE
                        WHEN delivery_state = ? AND delivery_class = ? THEN created_at
                        ELSE NULL
                    END) AS oldest_critical_created_at
                FROM durable_events
                """,
                (
                    DELIVERY_STATE_PENDING,
                    DELIVERY_STATE_PENDING,
                    DELIVERY_STATE_DLQ_BLOCKED,
                    DELIVERY_STATE_PENDING,
                    DELIVERY_CLASS_CRITICAL,
                ),
            ).fetchone()
            dead_letter = connection.execute(
                """
                SELECT COUNT(*) AS dlq_rows,
                    COALESCE(SUM(record_size_bytes), 0) AS dlq_bytes
                FROM dead_letter_events
                """
            ).fetchone()
            critical_safety = connection.execute(
                """
                SELECT COUNT(*) AS safety_rows,
                    COALESCE(SUM(record_size_bytes), 0) AS safety_bytes
                FROM critical_safety_slot
                """
            ).fetchone()

        outbox_rows = int(outbox["outbox_rows"])
        outbox_bytes = int(outbox["outbox_bytes"])
        blocked_rows = int(outbox["blocked_rows"])
        dlq_rows = int(dead_letter["dlq_rows"])
        dlq_bytes = int(dead_letter["dlq_bytes"])
        critical_safety_rows = int(critical_safety["safety_rows"])
        critical_safety_bytes = int(critical_safety["safety_bytes"])
        oldest_critical_age_seconds: int | None = None
        oldest_created_at = outbox["oldest_critical_created_at"]
        if oldest_created_at:
            try:
                parsed = datetime.fromisoformat(
                    str(oldest_created_at).replace("Z", "+00:00")
                )
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                oldest_critical_age_seconds = max(
                    0,
                    int(
                        (
                            self._wall_clock().astimezone(timezone.utc)
                            - parsed.astimezone(timezone.utc)
                        ).total_seconds()
                    ),
                )
            except ValueError:
                log.error(
                    "invalid oldest critical created_at in outbox: %s",
                    oldest_created_at,
                )

        row_ratio = outbox_rows / self.max_rows
        byte_ratio = outbox_bytes / self.max_bytes
        dlq_row_ratio = dlq_rows / self.max_dead_letters
        dlq_byte_ratio = dlq_bytes / self.max_dead_letter_bytes
        if critical_safety_rows > 0:
            capacity_state = "OPERATOR_STOP"
        elif blocked_rows > 0:
            capacity_state = "BLOCKED"
        elif max(row_ratio, byte_ratio, dlq_row_ratio, dlq_byte_ratio) >= 1.0:
            capacity_state = "SATURATED"
        elif max(row_ratio, byte_ratio, dlq_row_ratio, dlq_byte_ratio) >= 0.8:
            capacity_state = "PRESSURE"
        else:
            capacity_state = "HEALTHY"

        return DurableEventOutboxHealth(
            pending_rows=int(outbox["pending_rows"]),
            pending_bytes=int(outbox["pending_bytes"]),
            oldest_critical_age_seconds=oldest_critical_age_seconds,
            dlq_rows=dlq_rows,
            dlq_bytes=dlq_bytes,
            blocked_rows=blocked_rows,
            capacity_state=capacity_state,
            outbox_rows=outbox_rows,
            outbox_bytes=outbox_bytes,
            max_rows=self.max_rows,
            max_bytes=self.max_bytes,
            dlq_max_rows=self.max_dead_letters,
            dlq_max_bytes=self.max_dead_letter_bytes,
            critical_safety_rows=critical_safety_rows,
            critical_safety_bytes=critical_safety_bytes,
            critical_safety_max_bytes=self.max_critical_safety_bytes,
        )

    def _plan_capacity_locked(
        self,
        connection: sqlite3.Connection,
        *,
        incoming_class: str,
        incoming_size_bytes: int,
        replacement_ids: set[str],
    ) -> tuple[str | None, list[tuple[str, str]]]:
        if incoming_size_bytes > self.max_bytes:
            return "outbox byte quota exceeded", []

        rows = connection.execute(
            """
            SELECT event_id, delivery_class, delivery_state, record_size_bytes
            FROM durable_events
            ORDER BY CASE delivery_class
                WHEN 'METRIC' THEN 0
                WHEN 'STATE' THEN 1
                ELSE 2
            END ASC, created_at ASC, rowid ASC
            """
        ).fetchall()
        retained_rows = [
            row for row in rows if str(row["event_id"]) not in replacement_ids
        ]
        row_count = len(retained_rows)
        total_bytes = sum(int(row["record_size_bytes"]) for row in retained_rows)

        def fits() -> bool:
            return (
                row_count + 1 <= self.max_rows
                and total_bytes + incoming_size_bytes <= self.max_bytes
            )

        if fits():
            return None, []

        if incoming_class == DELIVERY_CLASS_CRITICAL:
            evictable_classes = {DELIVERY_CLASS_METRIC, DELIVERY_CLASS_STATE}
        elif incoming_class == DELIVERY_CLASS_STATE:
            evictable_classes = {DELIVERY_CLASS_METRIC}
        else:
            evictable_classes = {DELIVERY_CLASS_METRIC}

        planned_evictions: list[tuple[str, str]] = []
        for row in retained_rows:
            delivery_class = str(row["delivery_class"] or DELIVERY_CLASS_CRITICAL)
            delivery_state = str(row["delivery_state"] or DELIVERY_STATE_PENDING)
            if (
                delivery_class not in evictable_classes
                or delivery_state != DELIVERY_STATE_PENDING
            ):
                continue
            planned_evictions.append((str(row["event_id"]), delivery_class))
            row_count -= 1
            total_bytes -= int(row["record_size_bytes"])
            if fits():
                return None, planned_evictions

        row_overflow = row_count + 1 > self.max_rows
        reason = (
            "outbox row quota exceeded"
            if row_overflow
            else "outbox byte quota exceeded"
        )
        return reason, []

    def _prune_expired_locked(
        self,
        connection: sqlite3.Connection,
        *,
        exclude_event_ids: set[str] | None = None,
    ) -> list[tuple[str, bool]]:
        cutoff = utc_now_iso(
            self._wall_clock() - timedelta(seconds=self.max_age_seconds)
        )
        rows = connection.execute(
            """
            SELECT * FROM durable_events
            WHERE created_at < ? AND delivery_state = ?
            ORDER BY created_at ASC, rowid ASC
            """,
            (cutoff, DELIVERY_STATE_PENDING),
        ).fetchall()
        terminal_updates: list[tuple[str, bool]] = []
        excluded = exclude_event_ids or set()
        for row in rows:
            if str(row["event_id"]) in excluded:
                continue
            delivery_class = str(row["delivery_class"] or DELIVERY_CLASS_CRITICAL)
            if delivery_class == DELIVERY_CLASS_CRITICAL:
                continue
            connection.execute(
                "DELETE FROM durable_events WHERE event_id = ?",
                (str(row["event_id"]),),
            )
            terminal_updates.append((str(row["event_id"]), False))
        return terminal_updates

    @staticmethod
    def _outbox_record_size_from_row(
        row: sqlite3.Row,
        *,
        attempt_count: int,
        last_attempt_at: str | None,
    ) -> int:
        return _outbox_record_size(
            event_id=str(row["event_id"]),
            event_type=str(row["event_type"]),
            destination=str(row["destination"]),
            payload_json=str(row["payload_json"]),
            occurred_at=str(row["occurred_at"]),
            created_at=str(row["created_at"]),
            delivery_class=str(row["delivery_class"] or DELIVERY_CLASS_CRITICAL),
            compaction_key=str(row["compaction_key"])
            if row["compaction_key"]
            else None,
            attempt_count=attempt_count,
            last_attempt_at=last_attempt_at,
            delivery_state=str(row["delivery_state"] or DELIVERY_STATE_PENDING),
            rejection_code=str(row["rejection_code"])
            if row["rejection_code"]
            else None,
            rejection_reason=(
                str(row["rejection_reason"]) if row["rejection_reason"] else None
            ),
            rejection_source=(
                str(row["rejection_source"]) if row["rejection_source"] else None
            ),
            rejected_at=str(row["rejected_at"]) if row["rejected_at"] else None,
        )

    def _mark_dlq_blocked_locked(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        rejection_code: str,
        rejection_reason: str,
        rejection_source: str,
    ) -> None:
        total = connection.execute(
            "SELECT COALESCE(SUM(record_size_bytes), 0) AS total_bytes FROM durable_events"
        ).fetchone()
        current_total = int(total["total_bytes"])
        current_size = int(row["record_size_bytes"])
        rejected_at = utc_now_iso(self._wall_clock())

        metadata_candidates: tuple[
            tuple[str | None, str | None, str | None, str | None], ...
        ] = (
            (rejection_code, rejection_reason, rejection_source, rejected_at),
            (rejection_code, None, rejection_source, rejected_at),
            (None, None, None, None),
        )
        for code, reason, source, timestamp in metadata_candidates:
            record_size_bytes = _outbox_record_size(
                event_id=str(row["event_id"]),
                event_type=str(row["event_type"]),
                destination=str(row["destination"]),
                payload_json=str(row["payload_json"]),
                occurred_at=str(row["occurred_at"]),
                created_at=str(row["created_at"]),
                delivery_class=str(row["delivery_class"] or DELIVERY_CLASS_CRITICAL),
                compaction_key=(
                    str(row["compaction_key"]) if row["compaction_key"] else None
                ),
                attempt_count=int(row["attempt_count"]),
                last_attempt_at=(
                    str(row["last_attempt_at"]) if row["last_attempt_at"] else None
                ),
                delivery_state=DELIVERY_STATE_DLQ_BLOCKED,
                rejection_code=code,
                rejection_reason=reason,
                rejection_source=source,
                rejected_at=timestamp,
            )
            if current_total - current_size + record_size_bytes > self.max_bytes:
                continue
            connection.execute(
                """
                UPDATE durable_events
                SET delivery_state = ?, rejection_code = ?, rejection_reason = ?,
                    rejection_source = ?, rejected_at = ?, record_size_bytes = ?
                WHERE event_id = ?
                """,
                (
                    DELIVERY_STATE_DLQ_BLOCKED,
                    code,
                    reason,
                    source,
                    timestamp,
                    record_size_bytes,
                    str(row["event_id"]),
                ),
            )
            return
        raise DurableEventCapacityError(
            f"cannot retain DLQ_BLOCKED state for eventId={row['event_id']}"
        )

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
        delivery_class: str,
        compaction_key: str | None,
        payload_sha256: str,
    ) -> bool:
        normalized_payload_sha256 = (
            payload_sha256 or hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        )
        normalized_reason = _truncate_utf8(reason, MAX_REJECTION_REASON_BYTES)
        normalized_code = _truncate_utf8(rejection_code, MAX_REJECTION_CODE_BYTES)
        normalized_source = _truncate_utf8(source, MAX_REJECTION_SOURCE_BYTES)
        stored_payload_json = payload_json
        stored_payload_size_bytes = payload_size_bytes
        dead_lettered_at = utc_now_iso(self._wall_clock())

        def record_size() -> int:
            return _dead_letter_record_size(
                event_id=event_id,
                event_type=event_type,
                destination=destination,
                payload_json=stored_payload_json,
                occurred_at=occurred_at,
                created_at=created_at,
                delivery_class=delivery_class,
                compaction_key=compaction_key,
                payload_sha256=normalized_payload_sha256,
                attempt_count=attempt_count,
                last_attempt_at=last_attempt_at,
                dead_lettered_at=dead_lettered_at,
                reason=normalized_reason,
                rejection_code=normalized_code,
                source=normalized_source,
            )

        stored_record_size_bytes = record_size()
        if stored_record_size_bytes > self.max_dead_letter_bytes:
            log.error(
                "DLQ byte cap too small to retain full event payload eventId=%s",
                event_id,
            )
            return False

        existing = connection.execute(
            "SELECT * FROM dead_letter_events WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        if existing is not None:
            existing_compaction_key = (
                str(existing["compaction_key"]) if existing["compaction_key"] else None
            )
            if (
                str(existing["event_type"]) != event_type
                or str(existing["destination"]) != destination
                or str(existing["payload_sha256"]) != normalized_payload_sha256
                or str(existing["delivery_class"] or DELIVERY_CLASS_CRITICAL)
                != delivery_class
                or existing_compaction_key != compaction_key
            ):
                raise ValueError(
                    f"event_id collision with different dead letter: {event_id}"
                )
            return True

        rows = connection.execute(
            """
            SELECT id, record_size_bytes, delivery_class
            FROM dead_letter_events
            ORDER BY CASE delivery_class
                WHEN 'METRIC' THEN 0
                WHEN 'STATE' THEN 1
                ELSE 2
            END ASC, id ASC
            """
        ).fetchall()
        row_count = len(rows)
        total_bytes = sum(int(row["record_size_bytes"]) for row in rows)
        planned_purges: list[int] = []
        for row in rows:
            if (
                row_count + 1 <= self.max_dead_letters
                and total_bytes + stored_record_size_bytes <= self.max_dead_letter_bytes
            ):
                break
            existing_class = str(row["delivery_class"] or DELIVERY_CLASS_CRITICAL)
            if (
                DELIVERY_CLASS_PRIORITY[existing_class]
                > DELIVERY_CLASS_PRIORITY[delivery_class]
            ):
                continue
            planned_purges.append(int(row["id"]))
            row_count -= 1
            total_bytes -= int(row["record_size_bytes"])

        if (
            row_count + 1 > self.max_dead_letters
            or total_bytes + stored_record_size_bytes > self.max_dead_letter_bytes
        ):
            return False
        for dead_letter_id in planned_purges:
            connection.execute(
                "DELETE FROM dead_letter_events WHERE id = ?", (dead_letter_id,)
            )
            log.error("DLQ capacity reached; purged dead letter id=%s", dead_letter_id)
        inserted = connection.execute(
            """
            INSERT INTO dead_letter_events (
                event_id, event_type, destination, payload_json, occurred_at,
                created_at, payload_size_bytes, record_size_bytes, delivery_class,
                compaction_key, payload_sha256, attempt_count, last_attempt_at,
                dead_lettered_at, reason, rejection_code, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                event_type,
                destination,
                stored_payload_json,
                occurred_at,
                created_at,
                stored_payload_size_bytes,
                stored_record_size_bytes,
                delivery_class,
                compaction_key,
                normalized_payload_sha256,
                attempt_count,
                last_attempt_at,
                dead_lettered_at,
                normalized_reason,
                normalized_code,
                normalized_source,
            ),
        )
        return inserted.rowcount == 1

    def _apply_terminal_updates_locked(self, updates: list[tuple[str, bool]]) -> None:
        for event_id, successful in updates:
            self._record_terminal_locked(event_id, successful=successful)
        if updates:
            self._ack_condition.notify_all()

    def _record_terminal_locked(self, event_id: str, *, successful: bool) -> None:
        if event_id in self._terminal_results:
            try:
                self._terminal_order.remove(event_id)
            except ValueError:
                pass
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
        delivery_class: str,
        compaction_key: str | None,
    ) -> None:
        if (
            event.event_type != event_type
            or event.destination != destination
            or event.payload != payload
            or event.delivery_class != delivery_class
            or event.compaction_key != compaction_key
        ):
            raise ValueError(
                f"event_id collision with different content: {event.event_id}"
            )

    @staticmethod
    def _assert_same_dead_letter(
        event: DeadLetterEvent,
        *,
        event_type: str,
        destination: str,
        occurred_at: str,
        payload_sha256: str,
        delivery_class: str,
        compaction_key: str | None,
    ) -> None:
        if (
            event.event_type != event_type
            or event.destination != destination
            or event.occurred_at != occurred_at
            or event.payload_sha256 != payload_sha256
            or event.delivery_class != delivery_class
            or event.compaction_key != compaction_key
        ):
            raise ValueError(
                f"event_id collision with different dead letter: {event.event_id}"
            )

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
            delivery_class=str(row["delivery_class"] or DELIVERY_CLASS_CRITICAL),
            compaction_key=str(row["compaction_key"])
            if row["compaction_key"]
            else None,
            delivery_state=str(row["delivery_state"] or DELIVERY_STATE_PENDING),
        )

    @staticmethod
    def _row_to_blocked_event(row: sqlite3.Row) -> BlockedDurableEvent:
        return BlockedDurableEvent(
            event_id=str(row["event_id"]),
            event_type=str(row["event_type"]),
            destination=str(row["destination"]),
            occurred_at=str(row["occurred_at"]),
            delivery_class=str(row["delivery_class"] or DELIVERY_CLASS_CRITICAL),
            rejection_code=str(row["rejection_code"])
            if row["rejection_code"]
            else None,
            rejection_reason=(
                str(row["rejection_reason"]) if row["rejection_reason"] else None
            ),
            rejection_source=(
                str(row["rejection_source"]) if row["rejection_source"] else None
            ),
            rejected_at=str(row["rejected_at"]) if row["rejected_at"] else None,
        )

    @staticmethod
    def _row_to_critical_safety_event(row: sqlite3.Row) -> CriticalSafetyEvent:
        return CriticalSafetyEvent(
            event_id=str(row["event_id"]),
            event_type=str(row["event_type"]),
            destination=str(row["destination"]),
            payload=json.loads(str(row["payload_json"])),
            occurred_at=str(row["occurred_at"]),
            retained_at=str(row["retained_at"]),
            last_error=str(row["last_error"]) if row["last_error"] else None,
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
            delivery_class=str(row["delivery_class"] or DELIVERY_CLASS_CRITICAL),
            compaction_key=str(row["compaction_key"])
            if row["compaction_key"]
            else None,
            payload_sha256=str(row["payload_sha256"]),
        )


class DurableEventDelivery:
    """Coordinates at-least-once delivery while SQLite remains the source of truth."""

    def __init__(
        self, outbox: DurableEventOutbox, clock: Callable[[], float] | None = None
    ) -> None:
        self.outbox = outbox
        self._clock = clock or time.monotonic
        # None means queued but not yet written to the WebSocket. A timestamp
        # means transport send completed and the application ACK is pending.
        self._inflight: dict[str, float | None] = {}
        self._lock = threading.Lock()

    def reset_for_reconnect(self) -> None:
        with self._lock:
            self._inflight.clear()

    def release_non_pending(self) -> int:
        with self._lock:
            candidates = tuple(self._inflight)
        released = 0
        for event_id in candidates:
            if self.outbox.is_pending(event_id):
                continue
            self.release(event_id)
            released += 1
        return released

    def inflight_count(self) -> int:
        with self._lock:
            return len(self._inflight)

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
        delivery_class: str | None = None,
        compaction_key: str | None = None,
    ) -> DurableEvent:
        event = self.outbox.persist(
            event_type=event_type,
            destination=destination,
            payload=payload,
            event_id=event_id,
            occurred_at=occurred_at,
            delivery_class=delivery_class,
            compaction_key=compaction_key,
        )
        self.release_non_pending()
        if (
            not event.dead_lettered
            and not event.dropped
            and event.delivery_state == DELIVERY_STATE_PENDING
        ):
            self.enqueue(event, sender)
        return event

    def replay(
        self,
        sender: Callable[[DurableEvent], bool],
        limit: int,
        retry_after_seconds: float = 5.0,
    ) -> int:
        enqueued = 0
        pending_events = self.outbox.pending(limit=limit)
        self.release_non_pending()
        for event in pending_events:
            if not self.enqueue(event, sender, retry_after_seconds=retry_after_seconds):
                break
            enqueued += 1
        return enqueued

    def mark_sent(self, event_id: str) -> bool:
        marked = self.outbox.mark_attempt(event_id)
        if marked:
            if self.outbox.is_pending(event_id):
                with self._lock:
                    if event_id in self._inflight:
                        self._inflight[event_id] = self._clock()
            else:
                self.release(event_id)
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
