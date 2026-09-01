from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from nuvion_app.inference.command_inbox import (
    MAX_REPORTED_STATE_BYTES,
    DurableCommandInbox,
    utc_now_iso,
)
from nuvion_app.inference.signaling_contract import (
    COMMAND_OBSERVED_ACK_QUEUE_DEST,
)

COMMAND_OBSERVED_DESTINATION = "/app/device/command.observed"
COMMAND_OBSERVED_ACK_DESTINATION = COMMAND_OBSERVED_ACK_QUEUE_DEST
OBSERVATION_ACK_STATUSES = frozenset({"ACCEPTED", "DUPLICATE", "REJECTED"})
DEFAULT_OBSERVATION_OUTBOX_MAX_ROWS = 10_000
DEFAULT_OBSERVATION_OUTBOX_MAX_BYTES = 128 * 1024 * 1024
DEFAULT_OBSERVATION_TERMINAL_SAFETY_ROWS = 8
DEFAULT_OBSERVATION_TERMINAL_SAFETY_BYTES = 8 * MAX_REPORTED_STATE_BYTES
DEFAULT_OBSERVATION_DLQ_MAX_ROWS = 1_000
DEFAULT_OBSERVATION_DLQ_MAX_BYTES = 32 * 1024 * 1024
DEFAULT_OBSERVATION_DLQ_MAX_AGE_SECONDS = 30 * 24 * 60 * 60
DEFAULT_OBSERVATION_RETRY_BASE_SECONDS = 1.0
DEFAULT_OBSERVATION_RETRY_MAX_SECONDS = 300.0


class CommandObservationError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class CommandObservation:
    observation_id: str
    command_id: str
    sequence: int
    revision: int
    command_type: str
    observed_at: str
    reported_state: dict[str, Any]
    attempts: int
    last_attempt_at: str | None = None
    last_error: str | None = None
    acknowledged_at: str | None = None
    delivery_state: str = "PENDING"


@dataclass(frozen=True)
class ObservationOutboxHealth:
    pending_rows: int
    pending_bytes: int
    reserved_rows: int
    reserved_bytes: int
    dlq_rows: int
    dlq_bytes: int
    dlq_blocked_rows: int
    max_pending_rows: int
    max_pending_bytes: int
    retention_pressure: bool
    expired_dlq_pruned: int

    def to_telemetry(self) -> dict[str, Any]:
        return {
            "pendingRows": self.pending_rows,
            "pendingBytes": self.pending_bytes,
            "reservedRows": self.reserved_rows,
            "reservedBytes": self.reserved_bytes,
            "dlqRows": self.dlq_rows,
            "dlqBytes": self.dlq_bytes,
            "dlqBlockedRows": self.dlq_blocked_rows,
            "maxPendingRows": self.max_pending_rows,
            "maxPendingBytes": self.max_pending_bytes,
            "retentionPressure": self.retention_pressure,
            "expiredDlqPruned": self.expired_dlq_pruned,
            "capacityState": (
                "DLQ_BLOCKED"
                if self.dlq_blocked_rows
                else "BACKPRESSURE" if self.retention_pressure else "HEALTHY"
            ),
        }


@dataclass(frozen=True)
class CommandObservationAck:
    observation_id: str
    command_id: str
    revision: int
    status: str
    processed_at: str
    retryable: bool | None = None
    code: str | None = None
    reason: str | None = None


def _canonical_uuid(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise CommandObservationError("INVALID_OBSERVATION_ACK", f"{field} must be UUID text")
    try:
        normalized = str(uuid.UUID(value))
    except ValueError as exc:
        raise CommandObservationError(
            "INVALID_OBSERVATION_ACK", f"{field} must be UUID text"
        ) from exc
    if normalized != value.lower():
        raise CommandObservationError(
            "INVALID_OBSERVATION_ACK", f"{field} must use canonical UUID form"
        )
    return normalized


def _positive_int(value: Any, field: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        or value > 2**63 - 1
    ):
        raise CommandObservationError(
            "INVALID_OBSERVATION_ACK",
            f"{field} must be a positive signed 64-bit integer",
        )
    return value


def _rfc3339(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise CommandObservationError(
            "INVALID_OBSERVATION_ACK", f"{field} must be RFC3339 text"
        )
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise CommandObservationError(
            "INVALID_OBSERVATION_ACK", f"{field} must be RFC3339 text"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CommandObservationError(
            "INVALID_OBSERVATION_ACK", f"{field} must include a timezone"
        )
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON member: {key}")
        result[key] = value
    return result


def parse_command_observation_ack(body: str) -> CommandObservationAck:
    try:
        payload = json.loads(body, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise CommandObservationError(
            "INVALID_OBSERVATION_ACK", "command observation ACK is not valid JSON"
        ) from exc
    required = {"observationId", "commandId", "revision", "status", "processedAt"}
    optional = {"retryable", "code", "reason"}
    if (
        not isinstance(payload, dict)
        or not required.issubset(payload)
        or not set(payload).issubset(required | optional)
    ):
        raise CommandObservationError(
            "INVALID_OBSERVATION_ACK",
            "command observation ACK must contain exactly observationId, commandId, "
            "revision, status and processedAt plus optional retryable/code/reason",
        )
    status = payload.get("status")
    if status not in OBSERVATION_ACK_STATUSES:
        raise CommandObservationError(
            "INVALID_OBSERVATION_ACK",
            f"unsupported command observation ACK status: {status}",
        )
    retryable = payload.get("retryable")
    if retryable is not None and not isinstance(retryable, bool):
        raise CommandObservationError(
            "INVALID_OBSERVATION_ACK", "retryable must be boolean or null"
        )
    code = payload.get("code")
    reason = payload.get("reason")
    if code is not None and (not isinstance(code, str) or len(code) > 100):
        raise CommandObservationError(
            "INVALID_OBSERVATION_ACK", "code must be text up to 100 characters"
        )
    if reason is not None and (not isinstance(reason, str) or len(reason) > 1000):
        raise CommandObservationError(
            "INVALID_OBSERVATION_ACK", "reason must be text up to 1000 characters"
        )
    return CommandObservationAck(
        observation_id=_canonical_uuid(payload.get("observationId"), "observationId"),
        command_id=_canonical_uuid(payload.get("commandId"), "commandId"),
        revision=_positive_int(payload.get("revision"), "revision"),
        status=str(status),
        processed_at=_rfc3339(payload.get("processedAt"), "processedAt"),
        retryable=retryable,
        code=code,
        reason=reason,
    )


def build_command_observation_payload(
    observation: CommandObservation,
) -> dict[str, Any]:
    """Strict BE wire adapter; internal sequence/type never leak top-level."""

    if not isinstance(observation.reported_state, dict):
        raise TypeError("reportedState must be an object")
    return {
        "observationId": _canonical_uuid(
            observation.observation_id, "observationId"
        ),
        "commandId": _canonical_uuid(observation.command_id, "commandId"),
        "revision": _positive_int(observation.revision, "revision"),
        "observedAt": _rfc3339(observation.observed_at, "observedAt"),
        "reportedState": dict(observation.reported_state),
    }


class DurableCommandObservationOutbox:
    def __init__(
        self,
        inbox: DurableCommandInbox,
        *,
        clock: Callable[[], str] | None = None,
        id_factory: Callable[[], str] | None = None,
        max_rows: int = DEFAULT_OBSERVATION_OUTBOX_MAX_ROWS,
        retry_clock: Callable[[], float] | None = None,
        retry_base_seconds: float = DEFAULT_OBSERVATION_RETRY_BASE_SECONDS,
        retry_max_seconds: float = DEFAULT_OBSERVATION_RETRY_MAX_SECONDS,
        max_bytes: int = DEFAULT_OBSERVATION_OUTBOX_MAX_BYTES,
        terminal_safety_rows: int = DEFAULT_OBSERVATION_TERMINAL_SAFETY_ROWS,
        terminal_safety_bytes: int = DEFAULT_OBSERVATION_TERMINAL_SAFETY_BYTES,
        dlq_max_rows: int = DEFAULT_OBSERVATION_DLQ_MAX_ROWS,
        dlq_max_bytes: int = DEFAULT_OBSERVATION_DLQ_MAX_BYTES,
        dlq_max_age_seconds: float = DEFAULT_OBSERVATION_DLQ_MAX_AGE_SECONDS,
    ) -> None:
        self.inbox = inbox
        self._clock = clock or utc_now_iso
        self._id_factory = id_factory or (lambda: str(uuid.uuid4()))
        self.max_rows = max(1, min(int(max_rows), 1_000_000))
        self.max_bytes = max(MAX_REPORTED_STATE_BYTES, int(max_bytes))
        self.terminal_safety_rows = max(1, min(int(terminal_safety_rows), 1000))
        self.terminal_safety_bytes = max(
            MAX_REPORTED_STATE_BYTES,
            int(terminal_safety_bytes),
        )
        self.dlq_max_rows = max(1, min(int(dlq_max_rows), 1_000_000))
        self.dlq_max_bytes = max(MAX_REPORTED_STATE_BYTES, int(dlq_max_bytes))
        self.dlq_max_age_seconds = max(60.0, float(dlq_max_age_seconds))
        self._retry_clock = retry_clock or time.time
        self.retry_base_seconds = max(0.1, float(retry_base_seconds))
        self.retry_max_seconds = max(
            self.retry_base_seconds,
            float(retry_max_seconds),
        )
        self._initialize()

    def _initialize(self) -> None:
        with self.inbox.transaction(immediate=True) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS fleet_command_observation (
                    observation_id TEXT PRIMARY KEY,
                    command_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    revision INTEGER NOT NULL,
                    command_type TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    reported_state_json TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_attempt_at TEXT,
                    last_error TEXT,
                    next_attempt_at REAL NOT NULL DEFAULT 0,
                    acknowledged_at TEXT,
                    payload_bytes INTEGER NOT NULL DEFAULT 0,
                    delivery_state TEXT NOT NULL DEFAULT 'PENDING',
                    safety_slot INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(command_id, revision),
                    FOREIGN KEY(command_id) REFERENCES command_inbox(command_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS fleet_command_observation_dlq (
                    observation_id TEXT PRIMARY KEY,
                    command_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_bytes INTEGER NOT NULL DEFAULT 0,
                    code TEXT,
                    reason TEXT,
                    rejected_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS fleet_command_observation_reservation (
                    command_id TEXT PRIMARY KEY,
                    slots INTEGER NOT NULL,
                    reserved_bytes INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(command_id) REFERENCES command_inbox(command_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS fleet_command_observation_health (
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    expired_dlq_pruned INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO fleet_command_observation_health "
                "(singleton, expired_dlq_pruned) VALUES (1, 0)"
            )
            columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(fleet_command_observation)"
                ).fetchall()
            }
            if "next_attempt_at" not in columns:
                connection.execute(
                    "ALTER TABLE fleet_command_observation "
                    "ADD COLUMN next_attempt_at REAL NOT NULL DEFAULT 0"
                )
            if "payload_bytes" not in columns:
                connection.execute(
                    "ALTER TABLE fleet_command_observation "
                    "ADD COLUMN payload_bytes INTEGER NOT NULL DEFAULT 0"
                )
            if "delivery_state" not in columns:
                connection.execute(
                    "ALTER TABLE fleet_command_observation "
                    "ADD COLUMN delivery_state TEXT NOT NULL DEFAULT 'PENDING'"
                )
            if "safety_slot" not in columns:
                connection.execute(
                    "ALTER TABLE fleet_command_observation "
                    "ADD COLUMN safety_slot INTEGER NOT NULL DEFAULT 0"
                )
            connection.execute(
                "UPDATE fleet_command_observation SET payload_bytes = "
                "length(CAST(reported_state_json AS BLOB)) WHERE payload_bytes = 0"
            )
            dlq_columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(fleet_command_observation_dlq)"
                ).fetchall()
            }
            if "payload_bytes" not in dlq_columns:
                connection.execute(
                    "ALTER TABLE fleet_command_observation_dlq "
                    "ADD COLUMN payload_bytes INTEGER NOT NULL DEFAULT 0"
                )
            connection.execute(
                "UPDATE fleet_command_observation_dlq SET payload_bytes = "
                "length(CAST(payload_json AS BLOB)) WHERE payload_bytes = 0"
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_fleet_observation_pending
                ON fleet_command_observation(
                    acknowledged_at, delivery_state, command_id, revision
                )
                """
            )

    @staticmethod
    def _json(reported_state: Mapping[str, Any]) -> str:
        encoded = json.dumps(
            dict(reported_state),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        if len(encoded.encode("utf-8")) > MAX_REPORTED_STATE_BYTES:
            raise ValueError(
                f"reportedState exceeds {MAX_REPORTED_STATE_BYTES} UTF-8 bytes"
            )
        return encoded

    def enqueue(
        self,
        *,
        command_id: str,
        sequence: int,
        command_type: str,
        reported_state: Mapping[str, Any],
        terminal: bool = False,
        use_reservation: bool = False,
    ) -> CommandObservation:
        with self.inbox.transaction(immediate=True) as connection:
            return self.enqueue_in_transaction(
                connection,
                command_id=command_id,
                sequence=sequence,
                command_type=command_type,
                reported_state=reported_state,
                terminal=terminal,
                use_reservation=use_reservation,
            )

    def reserve_terminal_in_transaction(
        self,
        connection: sqlite3.Connection,
        *,
        command_id: str,
        slots: int = 1,
    ) -> None:
        """Reserve quota before an external effect can become observable."""

        safe_slots = max(1, min(int(slots), 4))
        existing = connection.execute(
            "SELECT slots FROM fleet_command_observation_reservation "
            "WHERE command_id = ?",
            (command_id,),
        ).fetchone()
        if existing is not None:
            if int(existing["slots"]) < safe_slots:
                raise CommandObservationError(
                    "OBSERVATION_RESERVATION_MISMATCH",
                    "existing observation reservation is smaller than required",
                )
            return
        pending_rows, pending_bytes, reserved_rows, reserved_bytes = (
            self._pending_usage_in_transaction(connection)
        )
        requested_bytes = safe_slots * MAX_REPORTED_STATE_BYTES
        if (
            pending_rows + reserved_rows + safe_slots > self.max_rows
            or pending_bytes + reserved_bytes + requested_bytes > self.max_bytes
        ):
            raise CommandObservationError(
                "OBSERVATION_CAPACITY_UNAVAILABLE",
                "no durable observation quota is available before effect execution",
            )
        connection.execute(
            """
            INSERT INTO fleet_command_observation_reservation (
                command_id, slots, reserved_bytes, created_at
            ) VALUES (?, ?, ?, ?)
            """,
            (command_id, safe_slots, requested_bytes, self._clock()),
        )

    @staticmethod
    def _pending_usage_in_transaction(
        connection: sqlite3.Connection,
    ) -> tuple[int, int, int, int]:
        pending = connection.execute(
            """
            SELECT COUNT(*) AS rows, COALESCE(SUM(payload_bytes), 0) AS bytes
            FROM fleet_command_observation
            WHERE acknowledged_at IS NULL
            """
        ).fetchone()
        reserved = connection.execute(
            """
            SELECT COALESCE(SUM(slots), 0) AS rows,
                   COALESCE(SUM(reserved_bytes), 0) AS bytes
            FROM fleet_command_observation_reservation
            """
        ).fetchone()
        return (
            int(pending["rows"]),
            int(pending["bytes"]),
            int(reserved["rows"]),
            int(reserved["bytes"]),
        )

    def enqueue_in_transaction(
        self,
        connection: sqlite3.Connection,
        *,
        command_id: str,
        sequence: int,
        command_type: str,
        reported_state: Mapping[str, Any],
        terminal: bool = False,
        use_reservation: bool = False,
    ) -> CommandObservation:
        state_json = self._json(reported_state)
        state_bytes = len(state_json.encode("utf-8"))
        latest = connection.execute(
            """
            SELECT * FROM fleet_command_observation
            WHERE command_id = ?
            ORDER BY revision DESC
            LIMIT 1
            """,
            (command_id,),
        ).fetchone()
        if latest is not None and str(latest["reported_state_json"]) == state_json:
            if terminal and use_reservation:
                connection.execute(
                    "DELETE FROM fleet_command_observation_reservation "
                    "WHERE command_id = ?",
                    (command_id,),
                )
            return self._row_to_observation(latest)

        reservation = connection.execute(
            "SELECT slots, reserved_bytes FROM fleet_command_observation_reservation "
            "WHERE command_id = ?",
            (command_id,),
        ).fetchone()
        consumed_reservation = bool(use_reservation and reservation is not None)
        safety_slot = 0
        if consumed_reservation:
            if int(reservation["slots"]) < 1 or int(reservation["reserved_bytes"]) < state_bytes:
                raise CommandObservationError(
                    "OBSERVATION_RESERVATION_EXHAUSTED",
                    "durable observation reservation cannot hold reported state",
                )
        else:
            pending_rows, pending_bytes, reserved_rows, reserved_bytes = (
                self._pending_usage_in_transaction(connection)
            )
            within_normal = (
                pending_rows + reserved_rows + 1 <= self.max_rows
                and pending_bytes + reserved_bytes + state_bytes <= self.max_bytes
            )
            within_terminal_safety = terminal and (
                pending_rows + 1 <= self.max_rows + self.terminal_safety_rows
                and pending_bytes + state_bytes
                <= self.max_bytes + self.terminal_safety_bytes
            )
            if not within_normal and not within_terminal_safety:
                raise CommandObservationError(
                    "OBSERVATION_OUTBOX_FULL",
                    "command observation outbox row/byte quota is full",
                )
            safety_slot = int(not within_normal and within_terminal_safety)
        revision = int(latest["revision"]) + 1 if latest is not None else 1
        _positive_int(revision, "revision")
        observation_id = _canonical_uuid(self._id_factory(), "observationId")
        observed_at = self._clock()
        connection.execute(
            """
            INSERT INTO fleet_command_observation (
                observation_id, command_id, sequence, revision, command_type,
                observed_at, reported_state_json, payload_bytes, delivery_state,
                safety_slot
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)
            """,
            (
                observation_id,
                command_id,
                sequence,
                revision,
                command_type,
                observed_at,
                state_json,
                state_bytes,
                safety_slot,
            ),
        )
        if consumed_reservation:
            remaining_slots = int(reservation["slots"]) - 1
            remaining_bytes = int(reservation["reserved_bytes"]) - MAX_REPORTED_STATE_BYTES
            if remaining_slots <= 0 or terminal:
                connection.execute(
                    "DELETE FROM fleet_command_observation_reservation "
                    "WHERE command_id = ?",
                    (command_id,),
                )
            else:
                connection.execute(
                    """
                    UPDATE fleet_command_observation_reservation
                    SET slots = ?, reserved_bytes = ?
                    WHERE command_id = ?
                    """,
                    (remaining_slots, remaining_bytes, command_id),
                )
        row = connection.execute(
            "SELECT * FROM fleet_command_observation WHERE observation_id = ?",
            (observation_id,),
        ).fetchone()
        return self._row_to_observation(row)

    def pending(self, *, limit: int = 100) -> list[CommandObservation]:
        safe_limit = max(1, min(int(limit), 1000))
        with self.inbox.transaction(immediate=True) as connection:
            self._prune_expired_dlq_in_transaction(connection)
            rows = connection.execute(
                """
                SELECT * FROM fleet_command_observation
                WHERE acknowledged_at IS NULL
                  AND delivery_state = 'PENDING'
                  AND next_attempt_at <= ?
                ORDER BY rowid ASC
                LIMIT ?
                """,
                (self._retry_clock(), safe_limit),
            ).fetchall()
        return [self._row_to_observation(row) for row in rows]

    def mark_attempt(
        self,
        observation_id: str,
        *,
        error: str | None = None,
    ) -> None:
        with self.inbox.transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT attempts FROM fleet_command_observation "
                "WHERE observation_id = ? AND acknowledged_at IS NULL "
                "AND delivery_state = 'PENDING'",
                (observation_id,),
            ).fetchone()
            if row is None:
                return
            next_attempt = self._retry_clock() + self._retry_delay(
                int(row["attempts"]) + 1
            )
            connection.execute(
                """
                UPDATE fleet_command_observation
                SET attempts = attempts + 1, last_attempt_at = ?, last_error = ?,
                    next_attempt_at = ?
                WHERE observation_id = ? AND acknowledged_at IS NULL
                  AND delivery_state = 'PENDING'
                """,
                (
                    self._clock(),
                    str(error)[:1000] if error else None,
                    next_attempt,
                    observation_id,
                ),
            )

    def acknowledge(self, ack: CommandObservationAck) -> bool:
        with self.inbox.transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT * FROM fleet_command_observation WHERE observation_id = ?",
                (ack.observation_id,),
            ).fetchone()
            if row is None:
                raise CommandObservationError(
                    "OBSERVATION_NOT_FOUND",
                    f"unknown observationId: {ack.observation_id}",
                )
            if (
                str(row["command_id"]) != ack.command_id
                or int(row["revision"]) != ack.revision
            ):
                raise CommandObservationError(
                    "OBSERVATION_ACK_COLLISION",
                    "observation ACK identity does not match durable outbox",
                )
            if row["acknowledged_at"] is not None:
                return False
            if ack.status == "REJECTED" and ack.retryable is not False:
                detail = ": ".join(
                    value for value in (ack.code, ack.reason) if value
                ) or (
                    "retryable observation rejection"
                    if ack.retryable is True
                    else "observation rejection omitted retryability"
                )
                connection.execute(
                    """
                    UPDATE fleet_command_observation
                    SET last_error = ?, next_attempt_at = ?
                    WHERE observation_id = ?
                    """,
                    (
                        detail[:1000],
                        self._retry_clock()
                        + self._retry_delay(max(1, int(row["attempts"]))),
                        ack.observation_id,
                    ),
                )
                return False
            if ack.status == "REJECTED":
                self._prune_expired_dlq_in_transaction(connection)
                dlq_usage = connection.execute(
                    """
                    SELECT COUNT(*) AS rows,
                           COALESCE(SUM(payload_bytes), 0) AS bytes
                    FROM fleet_command_observation_dlq
                    """
                ).fetchone()
                payload_bytes = int(row["payload_bytes"])
                if (
                    int(dlq_usage["rows"]) + 1 > self.dlq_max_rows
                    or int(dlq_usage["bytes"]) + payload_bytes > self.dlq_max_bytes
                ):
                    detail = ": ".join(
                        value for value in (ack.code, ack.reason) if value
                    ) or "permanent observation rejection"
                    connection.execute(
                        """
                        UPDATE fleet_command_observation
                        SET delivery_state = 'DLQ_BLOCKED', last_error = ?,
                            next_attempt_at = 0
                        WHERE observation_id = ?
                        """,
                        (
                            ("DLQ full: " + detail)[:1000],
                            ack.observation_id,
                        ),
                    )
                    return True
                connection.execute(
                    """
                    INSERT OR IGNORE INTO fleet_command_observation_dlq (
                        observation_id, command_id, revision, payload_json,
                        payload_bytes, code, reason, rejected_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ack.observation_id,
                        ack.command_id,
                        ack.revision,
                        str(row["reported_state_json"]),
                        payload_bytes,
                        ack.code,
                        ack.reason,
                        ack.processed_at,
                    ),
                )
            connection.execute(
                """
                UPDATE fleet_command_observation
                SET acknowledged_at = ?, last_error = NULL,
                    delivery_state = 'ACKED'
                WHERE observation_id = ?
                """,
                (ack.processed_at, ack.observation_id),
            )
            # Keep only the latest acknowledged row as the durable revision/
            # idempotency anchor. Pending older revisions are never discarded.
            connection.execute(
                """
                DELETE FROM fleet_command_observation
                WHERE command_id = ?
                  AND acknowledged_at IS NOT NULL
                  AND revision < (
                      SELECT MAX(revision)
                      FROM fleet_command_observation
                      WHERE command_id = ? AND acknowledged_at IS NOT NULL
                  )
                """,
                (ack.command_id, ack.command_id),
            )
        return True

    def _prune_expired_dlq_in_transaction(
        self,
        connection: sqlite3.Connection,
    ) -> int:
        cutoff = datetime.fromtimestamp(
            self._retry_clock(), tz=timezone.utc
        ) - timedelta(seconds=self.dlq_max_age_seconds)
        rows = connection.execute(
            "SELECT observation_id, rejected_at "
            "FROM fleet_command_observation_dlq"
        ).fetchall()
        expired: list[str] = []
        for row in rows:
            raw = str(row["rejected_at"])
            candidate = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
            try:
                rejected_at = datetime.fromisoformat(candidate)
            except ValueError:
                continue
            if rejected_at.tzinfo is not None and rejected_at <= cutoff:
                expired.append(str(row["observation_id"]))
        for observation_id in expired:
            connection.execute(
                "DELETE FROM fleet_command_observation_dlq "
                "WHERE observation_id = ?",
                (observation_id,),
            )
        if expired:
            connection.execute(
                """
                UPDATE fleet_command_observation_health
                SET expired_dlq_pruned = expired_dlq_pruned + ?
                WHERE singleton = 1
                """,
                (len(expired),),
            )
        return len(expired)

    def _retry_delay(self, attempts: int) -> float:
        exponent = min(max(int(attempts) - 1, 0), 20)
        return min(
            self.retry_base_seconds * (2**exponent),
            self.retry_max_seconds,
        )

    def dead_letters(self, *, limit: int = 100) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 1000))
        with self.inbox.transaction() as connection:
            rows = connection.execute(
                """
                SELECT * FROM fleet_command_observation_dlq
                ORDER BY rowid ASC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        return [
            {
                "observationId": str(row["observation_id"]),
                "commandId": str(row["command_id"]),
                "revision": int(row["revision"]),
                "reportedState": json.loads(str(row["payload_json"])),
                "code": str(row["code"]) if row["code"] else None,
                "reason": str(row["reason"]) if row["reason"] else None,
                "rejectedAt": str(row["rejected_at"]),
            }
            for row in rows
        ]

    def blocked(self, *, limit: int = 100) -> list[CommandObservation]:
        safe_limit = max(1, min(int(limit), 1000))
        with self.inbox.transaction() as connection:
            rows = connection.execute(
                """
                SELECT * FROM fleet_command_observation
                WHERE acknowledged_at IS NULL
                  AND delivery_state = 'DLQ_BLOCKED'
                ORDER BY rowid ASC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        return [self._row_to_observation(row) for row in rows]

    def health_snapshot(self) -> ObservationOutboxHealth:
        with self.inbox.transaction(immediate=True) as connection:
            self._prune_expired_dlq_in_transaction(connection)
            pending = connection.execute(
                """
                SELECT COUNT(*) AS rows,
                       COALESCE(SUM(payload_bytes), 0) AS bytes,
                       COALESCE(SUM(CASE WHEN delivery_state = 'DLQ_BLOCKED'
                                        THEN 1 ELSE 0 END), 0) AS blocked
                FROM fleet_command_observation
                WHERE acknowledged_at IS NULL
                """
            ).fetchone()
            reserved = connection.execute(
                """
                SELECT COALESCE(SUM(slots), 0) AS rows,
                       COALESCE(SUM(reserved_bytes), 0) AS bytes
                FROM fleet_command_observation_reservation
                """
            ).fetchone()
            dlq = connection.execute(
                """
                SELECT COUNT(*) AS rows,
                       COALESCE(SUM(payload_bytes), 0) AS bytes
                FROM fleet_command_observation_dlq
                """
            ).fetchone()
            counters = connection.execute(
                "SELECT expired_dlq_pruned FROM fleet_command_observation_health "
                "WHERE singleton = 1"
            ).fetchone()
        pending_rows = int(pending["rows"])
        pending_bytes = int(pending["bytes"])
        reserved_rows = int(reserved["rows"])
        reserved_bytes = int(reserved["bytes"])
        pressure = (
            pending_rows + reserved_rows >= int(self.max_rows * 0.8)
            or pending_bytes + reserved_bytes >= int(self.max_bytes * 0.8)
            or int(pending["blocked"]) > 0
            or int(dlq["rows"]) >= int(self.dlq_max_rows * 0.8)
            or int(dlq["bytes"]) >= int(self.dlq_max_bytes * 0.8)
        )
        return ObservationOutboxHealth(
            pending_rows=pending_rows,
            pending_bytes=pending_bytes,
            reserved_rows=reserved_rows,
            reserved_bytes=reserved_bytes,
            dlq_rows=int(dlq["rows"]),
            dlq_bytes=int(dlq["bytes"]),
            dlq_blocked_rows=int(pending["blocked"]),
            max_pending_rows=self.max_rows,
            max_pending_bytes=self.max_bytes,
            retention_pressure=pressure,
            expired_dlq_pruned=int(counters["expired_dlq_pruned"]),
        )

    def acknowledge_body(self, body: str) -> tuple[CommandObservationAck, bool]:
        ack = parse_command_observation_ack(body)
        return ack, self.acknowledge(ack)

    def get(self, observation_id: str) -> CommandObservation | None:
        with self.inbox.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM fleet_command_observation WHERE observation_id = ?",
                (observation_id,),
            ).fetchone()
        return self._row_to_observation(row) if row is not None else None

    @staticmethod
    def _row_to_observation(row: sqlite3.Row) -> CommandObservation:
        return CommandObservation(
            observation_id=str(row["observation_id"]),
            command_id=str(row["command_id"]),
            sequence=int(row["sequence"]),
            revision=int(row["revision"]),
            command_type=str(row["command_type"]),
            observed_at=str(row["observed_at"]),
            reported_state=json.loads(str(row["reported_state_json"])),
            attempts=int(row["attempts"]),
            last_attempt_at=(
                str(row["last_attempt_at"]) if row["last_attempt_at"] else None
            ),
            last_error=str(row["last_error"]) if row["last_error"] else None,
            acknowledged_at=(
                str(row["acknowledged_at"]) if row["acknowledged_at"] else None
            ),
            delivery_state=str(row["delivery_state"]),
        )
