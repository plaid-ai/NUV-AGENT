from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nuvion_app.inference.fleet_command import (
    COMMAND_CAPABILITY_BY_TYPE,
    VerifiedFleetCommand,
)

COMMAND_STATUS_RECEIVED = "RECEIVED"
COMMAND_STATUS_IN_PROGRESS = "IN_PROGRESS"
COMMAND_STATUS_SUCCEEDED = "SUCCEEDED"
COMMAND_STATUS_FAILED = "FAILED"
COMMAND_STATUS_REJECTED = "REJECTED"
COMMAND_STATUS_ROLLED_BACK = "ROLLED_BACK"
MAX_REPORTED_STATE_BYTES = 64 * 1024
COMMAND_TRUST_DOMAINS = frozenset({"production", "macos-dev", "iq9075-dev"})

COMMAND_STATUSES = frozenset(
    {
        COMMAND_STATUS_RECEIVED,
        COMMAND_STATUS_IN_PROGRESS,
        COMMAND_STATUS_SUCCEEDED,
        COMMAND_STATUS_FAILED,
        COMMAND_STATUS_REJECTED,
        COMMAND_STATUS_ROLLED_BACK,
    }
)
TERMINAL_COMMAND_STATUSES = frozenset(
    {
        COMMAND_STATUS_SUCCEEDED,
        COMMAND_STATUS_FAILED,
        COMMAND_STATUS_REJECTED,
        COMMAND_STATUS_ROLLED_BACK,
    }
)
_ALLOWED_TRANSITIONS = {
    COMMAND_STATUS_RECEIVED: frozenset({COMMAND_STATUS_IN_PROGRESS}),
    COMMAND_STATUS_IN_PROGRESS: TERMINAL_COMMAND_STATUSES,
}


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def deterministic_ack_id(command_id: str, status: str) -> str:
    normalized_status = str(status or "").strip().upper()
    if normalized_status not in COMMAND_STATUSES:
        raise ValueError(f"unsupported command ACK status: {status}")
    try:
        namespace = uuid.UUID(command_id)
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError("command_id must be a UUID") from exc
    return str(uuid.uuid5(namespace, normalized_status))


def resolve_default_command_inbox_path(
    environ: Mapping[str, str] | None = None,
) -> Path:
    values = os.environ if environ is None else environ
    explicit = str(values.get("NUVION_COMMAND_INBOX_PATH") or "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    xdg_state_home = str(values.get("XDG_STATE_HOME") or "").strip()
    state_root = (
        Path(xdg_state_home).expanduser()
        if xdg_state_home
        else Path.home() / ".local" / "state"
    )
    return (state_root / "nuvion" / "commands.sqlite3").resolve()


class CommandInboxError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class CommandAck:
    ack_id: str
    command_id: str
    sequence: int
    status: str
    observed_at: str
    code: str | None = None
    message: str | None = None
    reported_state: dict[str, Any] | None = None


@dataclass(frozen=True)
class CommandRecord:
    command_id: str
    device_id: str
    space_id: int
    command_type: str
    schema_version: int
    sequence: int
    payload_base64: str
    payload_hash: str
    payload: dict[str, Any]
    compact_jws: str
    key_id: str
    actor: str
    authorization_context: str
    issued_at: str
    expires_at: str
    status: str
    received_at: str
    updated_at: str
    code: str | None = None
    message: str | None = None
    reported_state: dict[str, Any] | None = None

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_COMMAND_STATUSES


@dataclass(frozen=True)
class AcceptResult:
    record: CommandRecord
    ack: CommandAck
    duplicate: bool


@dataclass(frozen=True)
class CommandEffectOutcome:
    status: str
    reported_state: dict[str, Any] | None = None
    code: str | None = None
    message: str | None = None

    @classmethod
    def succeeded(
        cls, reported_state: Mapping[str, Any] | None = None
    ) -> CommandEffectOutcome:
        return cls(
            status=COMMAND_STATUS_SUCCEEDED,
            reported_state=dict(reported_state) if reported_state is not None else None,
        )

    @classmethod
    def deferred(cls) -> CommandEffectOutcome:
        """Checkpoint an external desired-state effect without claiming completion."""

        return cls(status=COMMAND_STATUS_IN_PROGRESS)


class DurableCommandInbox:
    """Crash-safe command journal and ACK transition store.

    A callback executed by :meth:`run_transactional_effect` may mutate this same
    SQLite connection, making that database effect atomic with the terminal ACK.
    Effects outside SQLite cannot be made exactly-once by this class: filesystem,
    process, device, and updater handlers must be idempotent/convergent and resume
    safely when an ``IN_PROGRESS`` command is replayed after a crash.
    """

    def __init__(
        self, path: str | Path, *, clock: Callable[[], str] | None = None
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self._clock = clock or utc_now_iso
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _transaction(self, *, immediate: bool = False):
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def _read_connection(self):
        connection = self._connect()
        try:
            yield connection
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
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("PRAGMA journal_mode = DELETE")
                connection.execute("PRAGMA synchronous = FULL")
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS command_inbox (
                        command_id TEXT PRIMARY KEY,
                        device_id TEXT NOT NULL,
                        space_id INTEGER NOT NULL,
                        command_type TEXT NOT NULL,
                        schema_version INTEGER NOT NULL,
                        sequence INTEGER NOT NULL UNIQUE,
                        payload_base64 TEXT NOT NULL,
                        payload_hash TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        compact_jws TEXT NOT NULL,
                        key_id TEXT NOT NULL,
                        actor TEXT NOT NULL,
                        authorization_context TEXT NOT NULL,
                        issued_at TEXT NOT NULL,
                        expires_at TEXT NOT NULL,
                        status TEXT NOT NULL,
                        code TEXT,
                        message TEXT,
                        reported_state_json TEXT,
                        received_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS command_ack_transitions (
                        ack_id TEXT PRIMARY KEY,
                        command_id TEXT NOT NULL,
                        sequence INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        observed_at TEXT NOT NULL,
                        code TEXT,
                        message TEXT,
                        reported_state_json TEXT,
                        UNIQUE(command_id, status),
                        FOREIGN KEY(command_id) REFERENCES command_inbox(command_id)
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS command_sequence (
                        scope TEXT PRIMARY KEY,
                        last_sequence INTEGER NOT NULL
                    )
                    """
                )
                connection.execute(
                    "INSERT OR IGNORE INTO command_sequence(scope, last_sequence) VALUES ('fleet', 0)"
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS command_identity_scope (
                        scope TEXT PRIMARY KEY,
                        device_id TEXT NOT NULL,
                        space_id INTEGER NOT NULL,
                        trust_domain TEXT NOT NULL
                    )
                    """
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    @staticmethod
    def _json(value: Mapping[str, Any]) -> str:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )

    @staticmethod
    def _optional_json(value: Mapping[str, Any] | None) -> str | None:
        return DurableCommandInbox._json(value) if value is not None else None

    @staticmethod
    def _normalize_transition_fields(
        code: str | None,
        message: str | None,
        reported_state: Mapping[str, Any] | None,
    ) -> tuple[str | None, str | None, dict[str, Any] | None]:
        normalized_code = str(code).strip() if code is not None else None
        normalized_message = str(message).strip() if message is not None else None
        normalized_code = normalized_code or None
        normalized_message = normalized_message or None
        if normalized_code is not None and len(normalized_code) > 100:
            raise ValueError("ACK code exceeds 100 characters")
        if normalized_message is not None and len(normalized_message) > 1000:
            raise ValueError("ACK message exceeds 1000 characters")
        normalized_state = dict(reported_state) if reported_state is not None else None
        if normalized_state is not None:
            encoded_state = DurableCommandInbox._json(normalized_state).encode("utf-8")
            if len(encoded_state) > MAX_REPORTED_STATE_BYTES:
                raise ValueError(
                    f"reportedState exceeds {MAX_REPORTED_STATE_BYTES} UTF-8 bytes"
                )
        return normalized_code, normalized_message, normalized_state

    def accept(self, command: VerifiedFleetCommand) -> AcceptResult:
        with self._lock, self._transaction(immediate=True) as connection:
            return self._accept_locked(connection, command)

    def accept_rejected(
        self,
        command: VerifiedFleetCommand,
        *,
        code: str,
        message: str,
    ) -> AcceptResult:
        """Atomically journal an authenticated command as terminally rejected.

        A command rejected before initial acceptance (for example because its signed
        validity window expired while the device was offline) must never be left as a
        resumable ``RECEIVED``/``IN_PROGRESS`` row. The command, sequence cursor and
        complete lifecycle are therefore committed by one ``BEGIN IMMEDIATE``.
        """

        normalized_code, normalized_message, _ = self._normalize_transition_fields(
            code,
            message,
            None,
        )
        with self._lock, self._transaction(immediate=True) as connection:
            accepted = self._accept_locked(connection, command)
            record = accepted.record
            if record.terminal:
                return accepted
            if record.status == COMMAND_STATUS_RECEIVED:
                self._transition_locked(
                    connection,
                    command.command_id,
                    COMMAND_STATUS_IN_PROGRESS,
                    code=None,
                    message=None,
                    reported_state=None,
                )
            elif record.status != COMMAND_STATUS_IN_PROGRESS:
                raise CommandInboxError(
                    "INVALID_REJECTION_STATE",
                    f"cannot reject command from status={record.status}",
                )
            ack = self._transition_locked(
                connection,
                command.command_id,
                COMMAND_STATUS_REJECTED,
                code=normalized_code,
                message=normalized_message,
                reported_state=None,
            )
            row = connection.execute(
                "SELECT * FROM command_inbox WHERE command_id = ?",
                (command.command_id,),
            ).fetchone()
            return AcceptResult(
                record=self._row_to_record(row),
                ack=ack,
                duplicate=accepted.duplicate,
            )

    def _accept_locked(
        self,
        connection: sqlite3.Connection,
        command: VerifiedFleetCommand,
    ) -> AcceptResult:
        existing = connection.execute(
            "SELECT * FROM command_inbox WHERE command_id = ?",
            (command.command_id,),
        ).fetchone()
        if existing is not None:
            self._assert_same_command(existing, command)
            record = self._row_to_record(existing)
            return AcceptResult(
                record=record,
                ack=self._ack_for_status(connection, command.command_id, record.status),
                duplicate=True,
            )

        last_sequence = int(
            connection.execute(
                "SELECT last_sequence FROM command_sequence WHERE scope = 'fleet'"
            ).fetchone()[0]
        )
        if command.sequence <= last_sequence:
            raise CommandInboxError(
                "SEQUENCE_REPLAY",
                f"sequence={command.sequence} is not greater than lastSequence={last_sequence}",
            )

        now = self._clock()
        connection.execute(
            """
            INSERT INTO command_inbox (
                command_id, device_id, space_id, command_type, schema_version,
                sequence, payload_base64, payload_hash, payload_json, compact_jws,
                key_id, actor, authorization_context, issued_at, expires_at,
                status, received_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                command.command_id,
                command.device_id,
                command.space_id,
                command.command_type,
                command.schema_version,
                command.sequence,
                command.payload_base64,
                command.payload_hash,
                self._json(command.payload),
                command.compact_jws,
                command.key_id,
                command.actor,
                command.authorization_context,
                command.issued_at,
                command.expires_at,
                COMMAND_STATUS_RECEIVED,
                now,
                now,
            ),
        )
        connection.execute(
            "UPDATE command_sequence SET last_sequence = ? WHERE scope = 'fleet'",
            (command.sequence,),
        )
        ack = self._insert_ack(
            connection,
            command_id=command.command_id,
            sequence=command.sequence,
            status=COMMAND_STATUS_RECEIVED,
            observed_at=now,
        )
        row = connection.execute(
            "SELECT * FROM command_inbox WHERE command_id = ?",
            (command.command_id,),
        ).fetchone()
        return AcceptResult(record=self._row_to_record(row), ack=ack, duplicate=False)

    def get(self, command_id: str) -> CommandRecord | None:
        with self._lock, self._read_connection() as connection:
            row = connection.execute(
                "SELECT * FROM command_inbox WHERE command_id = ?",
                (command_id,),
            ).fetchone()
        return self._row_to_record(row) if row is not None else None

    def bind_identity(
        self,
        *,
        device_id: str,
        space_id: int,
        trust_domain: str,
    ) -> None:
        """Bind one inbox file to one provisioned device/space trust scope."""

        normalized_device = str(device_id or "").strip()
        normalized_domain = str(trust_domain or "").strip()
        if not normalized_device:
            raise ValueError("device_id must be non-empty")
        if isinstance(space_id, bool) or not isinstance(space_id, int) or space_id < 1:
            raise ValueError("space_id must be a positive integer")
        if normalized_domain not in COMMAND_TRUST_DOMAINS:
            raise ValueError(
                "trust_domain must be production, macos-dev, or iq9075-dev"
            )

        with self._lock, self._transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT * FROM command_identity_scope WHERE scope = 'fleet'"
            ).fetchone()
            if row is None:
                historic_count = int(
                    connection.execute("SELECT COUNT(*) FROM command_inbox").fetchone()[
                        0
                    ]
                )
                last_sequence = int(
                    connection.execute(
                        "SELECT last_sequence FROM command_sequence WHERE scope = 'fleet'"
                    ).fetchone()[0]
                )
                if historic_count > 0 or last_sequence > 0:
                    raise CommandInboxError(
                        "IDENTITY_SCOPE_UNBOUND_LEGACY",
                        "legacy command inbox has no trusted identity scope; "
                        "archive or explicitly migrate it before enabling Fleet commands",
                    )
                connection.execute(
                    """
                    INSERT INTO command_identity_scope (
                        scope, device_id, space_id, trust_domain
                    ) VALUES ('fleet', ?, ?, ?)
                    """,
                    (normalized_device, space_id, normalized_domain),
                )
                return
            if (
                str(row["device_id"]) != normalized_device
                or int(row["space_id"]) != space_id
                or str(row["trust_domain"]) != normalized_domain
            ):
                raise CommandInboxError(
                    "IDENTITY_SCOPE_MISMATCH",
                    "command inbox identity scope changed; archive it before reprovisioning",
                )

    def last_sequence(self) -> int:
        with self._lock, self._read_connection() as connection:
            row = connection.execute(
                "SELECT last_sequence FROM command_sequence WHERE scope = 'fleet'"
            ).fetchone()
        return int(row[0])

    def pending(self, limit: int = 200) -> list[CommandRecord]:
        safe_limit = max(1, min(int(limit), 10_000))
        return self.pending_page(
            after_sequence=0,
            through_sequence=2**63 - 1,
            limit=safe_limit,
        )

    def pending_high_watermark(self) -> int:
        with self._lock, self._read_connection() as connection:
            row = connection.execute(
                """
                SELECT COALESCE(MAX(sequence), 0)
                FROM command_inbox
                WHERE status IN (?, ?)
                """,
                (COMMAND_STATUS_RECEIVED, COMMAND_STATUS_IN_PROGRESS),
            ).fetchone()
        return int(row[0])

    def pending_page(
        self,
        *,
        after_sequence: int,
        through_sequence: int,
        limit: int,
    ) -> list[CommandRecord]:
        if after_sequence < 0 or through_sequence < 0:
            raise ValueError("pending sequence bounds must be non-negative")
        if through_sequence < after_sequence:
            return []
        safe_limit = max(1, min(int(limit), 1_000))
        with self._lock, self._read_connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM command_inbox
                WHERE status IN (?, ?)
                  AND sequence > ?
                  AND sequence <= ?
                ORDER BY sequence ASC
                LIMIT ?
                """,
                (
                    COMMAND_STATUS_RECEIVED,
                    COMMAND_STATUS_IN_PROGRESS,
                    after_sequence,
                    through_sequence,
                    safe_limit,
                ),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def recent_command_ids(self, limit: int = 200) -> list[str]:
        """Return recent command IDs in sequence order for deterministic ACK replay."""

        safe_limit = max(1, min(int(limit), 10_000))
        with self._lock, self._read_connection() as connection:
            rows = connection.execute(
                """
                SELECT command_id FROM (
                    SELECT command_id, sequence
                    FROM command_inbox
                    ORDER BY sequence DESC
                    LIMIT ?
                ) recent
                ORDER BY sequence ASC
                """,
                (safe_limit,),
            ).fetchall()
        return [str(row["command_id"]) for row in rows]

    @staticmethod
    def rehydrate(record: CommandRecord) -> VerifiedFleetCommand:
        required_capability = COMMAND_CAPABILITY_BY_TYPE.get(record.command_type)
        if required_capability is None:
            raise CommandInboxError(
                "UNSUPPORTED_STORED_COMMAND",
                f"stored command type is unsupported: {record.command_type}",
            )
        return VerifiedFleetCommand(
            command_id=record.command_id,
            device_id=record.device_id,
            space_id=record.space_id,
            command_type=record.command_type,
            schema_version=record.schema_version,
            issued_at=record.issued_at,
            expires_at=record.expires_at,
            sequence=record.sequence,
            payload_base64=record.payload_base64,
            payload_hash=record.payload_hash,
            payload=record.payload,
            actor=record.actor,
            authorization_context=record.authorization_context,
            key_id=record.key_id,
            required_capability=required_capability,
            compact_jws=record.compact_jws,
        )

    def transition(
        self,
        command_id: str,
        status: str,
        *,
        code: str | None = None,
        message: str | None = None,
        reported_state: Mapping[str, Any] | None = None,
    ) -> CommandAck:
        normalized_status = str(status or "").strip().upper()
        if normalized_status not in COMMAND_STATUSES:
            raise ValueError(f"unsupported command status: {status}")
        normalized_code, normalized_message, normalized_state = (
            self._normalize_transition_fields(
                code,
                message,
                reported_state,
            )
        )
        with self._lock, self._transaction(immediate=True) as connection:
            return self._transition_locked(
                connection,
                command_id,
                normalized_status,
                code=normalized_code,
                message=normalized_message,
                reported_state=normalized_state,
            )

    def run_transactional_effect(
        self,
        command_id: str,
        effect: Callable[[sqlite3.Connection], CommandEffectOutcome],
    ) -> tuple[CommandAck, bool]:
        """Run an effect against this SQLite DB and persist its durable checkpoint.

        The callback is invoked only while the command is ``IN_PROGRESS``. If a
        duplicate worker reaches a terminal row, the saved terminal ACK is returned
        without invoking the callback. A deferred outcome commits the callback's
        SQLite checkpoint and retains ``IN_PROGRESS`` for a later product-specific
        reconciler; terminal outcomes are committed atomically with their ACK.
        """

        with self._lock, self._transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT * FROM command_inbox WHERE command_id = ?",
                (command_id,),
            ).fetchone()
            if row is None:
                raise CommandInboxError(
                    "COMMAND_NOT_FOUND", f"command not found: {command_id}"
                )
            status = str(row["status"])
            if status in TERMINAL_COMMAND_STATUSES:
                return self._ack_for_status(connection, command_id, status), False
            if status != COMMAND_STATUS_IN_PROGRESS:
                raise CommandInboxError(
                    "INVALID_TRANSITION", f"effect requires IN_PROGRESS, got {status}"
                )

            outcome = effect(connection)
            if not isinstance(outcome, CommandEffectOutcome):
                raise TypeError("command effect must return CommandEffectOutcome")
            normalized_status = str(outcome.status or "").strip().upper()
            if normalized_status == COMMAND_STATUS_IN_PROGRESS:
                if (
                    outcome.code is not None
                    or outcome.message is not None
                    or outcome.reported_state is not None
                ):
                    raise ValueError(
                        "deferred command effect cannot report terminal fields"
                    )
                return self._ack_for_status(
                    connection,
                    command_id,
                    COMMAND_STATUS_IN_PROGRESS,
                ), True
            if normalized_status not in TERMINAL_COMMAND_STATUSES:
                raise ValueError(
                    "command effect must return IN_PROGRESS or a terminal status"
                )
            normalized_code, normalized_message, normalized_state = (
                self._normalize_transition_fields(
                    outcome.code,
                    outcome.message,
                    outcome.reported_state,
                )
            )
            ack = self._transition_locked(
                connection,
                command_id,
                normalized_status,
                code=normalized_code,
                message=normalized_message,
                reported_state=normalized_state,
            )
            return ack, True

    def ack_transitions(self, command_id: str) -> list[CommandAck]:
        with self._lock, self._read_connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM command_ack_transitions
                WHERE command_id = ?
                ORDER BY rowid ASC
                """,
                (command_id,),
            ).fetchall()
        return [self._row_to_ack(row) for row in rows]

    def _transition_locked(
        self,
        connection: sqlite3.Connection,
        command_id: str,
        target_status: str,
        *,
        code: str | None,
        message: str | None,
        reported_state: Mapping[str, Any] | None,
    ) -> CommandAck:
        row = connection.execute(
            "SELECT * FROM command_inbox WHERE command_id = ?",
            (command_id,),
        ).fetchone()
        if row is None:
            raise CommandInboxError(
                "COMMAND_NOT_FOUND", f"command not found: {command_id}"
            )
        current_status = str(row["status"])
        if current_status == target_status:
            return self._ack_for_status(connection, command_id, target_status)
        if current_status in TERMINAL_COMMAND_STATUSES:
            raise CommandInboxError(
                "TERMINAL_STATE",
                f"terminal command cannot transition from {current_status} to {target_status}",
            )
        if target_status not in _ALLOWED_TRANSITIONS.get(current_status, frozenset()):
            raise CommandInboxError(
                "INVALID_TRANSITION",
                f"invalid command transition {current_status} -> {target_status}",
            )

        now = self._clock()
        connection.execute(
            """
            UPDATE command_inbox
            SET status = ?, code = ?, message = ?, reported_state_json = ?, updated_at = ?
            WHERE command_id = ?
            """,
            (
                target_status,
                code,
                message,
                self._optional_json(reported_state),
                now,
                command_id,
            ),
        )
        return self._insert_ack(
            connection,
            command_id=command_id,
            sequence=int(row["sequence"]),
            status=target_status,
            observed_at=now,
            code=code,
            message=message,
            reported_state=reported_state,
        )

    def _insert_ack(
        self,
        connection: sqlite3.Connection,
        *,
        command_id: str,
        sequence: int,
        status: str,
        observed_at: str,
        code: str | None = None,
        message: str | None = None,
        reported_state: Mapping[str, Any] | None = None,
    ) -> CommandAck:
        ack_id = deterministic_ack_id(command_id, status)
        connection.execute(
            """
            INSERT INTO command_ack_transitions (
                ack_id, command_id, sequence, status, observed_at,
                code, message, reported_state_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ack_id,
                command_id,
                sequence,
                status,
                observed_at,
                code,
                message,
                self._optional_json(reported_state),
            ),
        )
        return CommandAck(
            ack_id=ack_id,
            command_id=command_id,
            sequence=sequence,
            status=status,
            observed_at=observed_at,
            code=code,
            message=message,
            reported_state=dict(reported_state) if reported_state is not None else None,
        )

    def _ack_for_status(
        self,
        connection: sqlite3.Connection,
        command_id: str,
        status: str,
    ) -> CommandAck:
        row = connection.execute(
            """
            SELECT * FROM command_ack_transitions
            WHERE command_id = ? AND status = ?
            """,
            (command_id, status),
        ).fetchone()
        if row is None:
            raise CommandInboxError(
                "ACK_NOT_FOUND", f"ACK not found for {command_id}/{status}"
            )
        return self._row_to_ack(row)

    @staticmethod
    def _assert_same_command(row: sqlite3.Row, command: VerifiedFleetCommand) -> None:
        expected = (
            command.device_id,
            command.space_id,
            command.command_type,
            command.schema_version,
            command.sequence,
            command.payload_base64,
            command.payload_hash,
            command.compact_jws,
        )
        actual = (
            str(row["device_id"]),
            int(row["space_id"]),
            str(row["command_type"]),
            int(row["schema_version"]),
            int(row["sequence"]),
            str(row["payload_base64"]),
            str(row["payload_hash"]),
            str(row["compact_jws"]),
        )
        if actual != expected:
            raise CommandInboxError(
                "COMMAND_ID_COLLISION",
                f"commandId reused with different immutable content: {command.command_id}",
            )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> CommandRecord:
        reported = (
            json.loads(str(row["reported_state_json"]))
            if row["reported_state_json"]
            else None
        )
        return CommandRecord(
            command_id=str(row["command_id"]),
            device_id=str(row["device_id"]),
            space_id=int(row["space_id"]),
            command_type=str(row["command_type"]),
            schema_version=int(row["schema_version"]),
            sequence=int(row["sequence"]),
            payload_base64=str(row["payload_base64"]),
            payload_hash=str(row["payload_hash"]),
            payload=json.loads(str(row["payload_json"])),
            compact_jws=str(row["compact_jws"]),
            key_id=str(row["key_id"]),
            actor=str(row["actor"]),
            authorization_context=str(row["authorization_context"]),
            issued_at=str(row["issued_at"]),
            expires_at=str(row["expires_at"]),
            status=str(row["status"]),
            received_at=str(row["received_at"]),
            updated_at=str(row["updated_at"]),
            code=str(row["code"]) if row["code"] is not None else None,
            message=str(row["message"]) if row["message"] is not None else None,
            reported_state=reported,
        )

    @staticmethod
    def _row_to_ack(row: sqlite3.Row) -> CommandAck:
        reported = (
            json.loads(str(row["reported_state_json"]))
            if row["reported_state_json"]
            else None
        )
        return CommandAck(
            ack_id=str(row["ack_id"]),
            command_id=str(row["command_id"]),
            sequence=int(row["sequence"]),
            status=str(row["status"]),
            observed_at=str(row["observed_at"]),
            code=str(row["code"]) if row["code"] is not None else None,
            message=str(row["message"]) if row["message"] is not None else None,
            reported_state=reported,
        )
