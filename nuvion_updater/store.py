from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import threading
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from nuvion_updater.errors import UpdaterError, UpdaterSecurityError
from nuvion_updater.util import ensure_directory


class UpdatePhase(str, Enum):
    AUTHORIZED = "AUTHORIZED"
    DOWNLOADING = "DOWNLOADING"
    STAGING = "STAGING"
    VERIFIED = "VERIFIED"
    ACTIVATING = "ACTIVATING"
    BOOT_HEALTHY = "BOOT_HEALTHY"
    FUNCTIONAL_HEALTHY = "FUNCTIONAL_HEALTHY"
    COMMITTED = "COMMITTED"
    ROLLING_BACK = "ROLLING_BACK"
    ROLLED_BACK = "ROLLED_BACK"
    ROLLBACK_FAILED = "ROLLBACK_FAILED"
    FAILED = "FAILED"


TERMINAL_PHASES = frozenset(
    {
        UpdatePhase.COMMITTED,
        UpdatePhase.ROLLED_BACK,
        UpdatePhase.ROLLBACK_FAILED,
        UpdatePhase.FAILED,
    }
)


@dataclass(frozen=True)
class UpdateState:
    command_id: str
    sequence: int
    compact_jws: str
    compact_jws_sha256: str
    target_version: str
    bom_digest: str
    phase: UpdatePhase
    candidate_slot: str | None
    previous_slot: str | None
    previous_version: str | None
    release_sequence: int | None
    artifact_digest: str | None
    component_sha: str | None
    config_schema: str | None
    bom_verification_status: str | None
    publisher_key_id: str | None
    health_deadline: str | None
    error_code: str | None
    message: str | None
    created_at: str
    updated_at: str

    @property
    def terminal(self) -> bool:
        return self.phase in TERMINAL_PHASES

    def public_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "commandId": self.command_id,
            "sequence": self.sequence,
            "targetVersion": self.target_version,
            "bomDigest": self.bom_digest,
            "phase": self.phase.value,
            "updatePhase": self.phase.value,
            "updatedAt": self.updated_at,
        }
        optional = {
            "candidateSlot": self.candidate_slot,
            "previousSlot": self.previous_slot,
            "previousVersion": self.previous_version,
            "releaseSequence": self.release_sequence,
            "artifactDigest": self.artifact_digest,
            "componentSha": self.component_sha,
            "configSchema": self.config_schema,
            "bomVerificationStatus": self.bom_verification_status,
            "publisherKeyId": self.publisher_key_id,
            "healthDeadline": self.health_deadline,
            "errorCode": self.error_code,
            "message": self.message,
        }
        if self.phase == UpdatePhase.ROLLED_BACK and self.previous_slot is not None:
            result["slot"] = self.previous_slot
            result["rollbackSlot"] = self.previous_slot
            if self.previous_version is not None:
                result["rollbackVersion"] = self.previous_version
        elif self.candidate_slot is not None:
            result["slot"] = f"releases/{self.bom_digest[7:]}"
        if self.phase in {UpdatePhase.FUNCTIONAL_HEALTHY, UpdatePhase.COMMITTED}:
            result["health"] = "FUNCTIONAL_HEALTHY"
            result["functionalHealth"] = "FUNCTIONAL_HEALTHY"
        elif self.phase == UpdatePhase.BOOT_HEALTHY:
            result["health"] = "BOOT_HEALTHY"
            result["functionalHealth"] = "FUNCTIONAL_UNHEALTHY"
        elif self.phase == UpdatePhase.ROLLED_BACK:
            result["health"] = "LKG_RESTORED"
            result["functionalHealth"] = "FUNCTIONAL_UNHEALTHY"
        else:
            result["functionalHealth"] = "FUNCTIONAL_UNHEALTHY"
        result.update({key: value for key, value in optional.items() if value is not None})
        return result


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


class UpdaterStore:
    """Root-only crash-safe update journal and replay boundary."""

    def __init__(
        self,
        path: str | Path,
        *,
        require_root_owner: bool = True,
        clock: Callable[[], str] | None = None,
    ) -> None:
        self.path = Path(path)
        self.require_root_owner = require_root_owner
        self._clock = clock or utc_now
        self._lock = threading.RLock()
        self.state_directory = ensure_directory(
            self.path.parent,
            mode=0o700,
            require_root_owner=require_root_owner,
        )
        if self.path.exists():
            metadata = self.path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise UpdaterSecurityError(
                    "UNSAFE_JOURNAL", "updater journal must be a regular file"
                )
            if metadata.st_mode & 0o022:
                raise UpdaterSecurityError(
                    "UNSAFE_JOURNAL",
                    "updater journal must not be group/other writable",
                )
            if require_root_owner and metadata.st_uid != 0:
                raise UpdaterSecurityError(
                    "UNSAFE_JOURNAL", "updater journal must be owned by root"
                )
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA synchronous = FULL")
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

    def _initialize(self) -> None:
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("PRAGMA journal_mode = WAL")
                schema_version = int(
                    connection.execute("PRAGMA user_version").fetchone()[0]
                )
                if schema_version not in {0, 1, 2}:
                    raise UpdaterSecurityError(
                        "UNSUPPORTED_JOURNAL_SCHEMA",
                        f"unsupported updater journal schema: {schema_version}",
                    )
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS updater_command (
                        command_id TEXT PRIMARY KEY,
                        sequence INTEGER NOT NULL UNIQUE CHECK (sequence > 0),
                        compact_jws TEXT NOT NULL,
                        compact_jws_sha256 TEXT NOT NULL,
                        target_version TEXT NOT NULL,
                        bom_digest TEXT NOT NULL,
                        phase TEXT NOT NULL,
                        candidate_slot TEXT,
                        previous_slot TEXT,
                        previous_version TEXT,
                        release_sequence INTEGER,
                        artifact_digest TEXT,
                        component_sha TEXT,
                        config_schema TEXT,
                        bom_verification_status TEXT,
                        publisher_key_id TEXT,
                        health_deadline TEXT,
                        error_code TEXT,
                        message TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS updater_transition (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        command_id TEXT NOT NULL,
                        from_phase TEXT,
                        to_phase TEXT NOT NULL,
                        observed_at TEXT NOT NULL,
                        error_code TEXT,
                        message TEXT,
                        FOREIGN KEY (command_id) REFERENCES updater_command(command_id)
                    );

                    CREATE TABLE IF NOT EXISTS updater_meta (
                        meta_key TEXT PRIMARY KEY,
                        meta_value TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_updater_command_sequence
                    ON updater_command(sequence DESC);
                    """
                )
                columns = {
                    str(row[1])
                    for row in connection.execute(
                        "PRAGMA table_info(updater_command)"
                    ).fetchall()
                }
                migrations = {
                    "previous_version": "TEXT",
                    "artifact_digest": "TEXT",
                    "component_sha": "TEXT",
                    "config_schema": "TEXT",
                    "bom_verification_status": "TEXT",
                    "publisher_key_id": "TEXT",
                }
                for column, declaration in migrations.items():
                    if column not in columns:
                        connection.execute(
                            f"ALTER TABLE updater_command ADD COLUMN {column} {declaration}"
                        )
                connection.execute("PRAGMA user_version = 2")
                connection.commit()
            finally:
                connection.close()
        try:
            os.chmod(self.path, 0o600)
        except OSError as exc:
            raise UpdaterSecurityError(
                "UNSAFE_JOURNAL", f"cannot secure updater journal: {exc}"
            ) from exc

    def authorize(
        self,
        *,
        command_id: str,
        sequence: int,
        compact_jws: str,
        target_version: str,
        bom_digest: str,
    ) -> tuple[UpdateState, bool]:
        digest = hashlib.sha256(compact_jws.encode("ascii")).hexdigest()
        now = self._clock()
        with self._lock, self._transaction(immediate=True) as connection:
            existing = connection.execute(
                "SELECT * FROM updater_command WHERE command_id = ?", (command_id,)
            ).fetchone()
            if existing is not None:
                state = self._row(existing)
                if (
                    state.sequence != sequence
                    or state.compact_jws_sha256 != digest
                    or state.target_version != target_version
                    or state.bom_digest != bom_digest
                ):
                    raise UpdaterSecurityError(
                        "COMMAND_COLLISION",
                        "commandId was already authorized with different signed content",
                    )
                return state, True

            maximum = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) FROM updater_command"
            ).fetchone()[0]
            if sequence <= int(maximum):
                raise UpdaterSecurityError(
                    "COMMAND_REPLAY",
                    "command sequence is not greater than the durable updater sequence",
                )

            active = connection.execute(
                """
                SELECT * FROM updater_command
                WHERE phase NOT IN (
                    'COMMITTED', 'ROLLED_BACK', 'ROLLBACK_FAILED', 'FAILED'
                )
                ORDER BY sequence DESC LIMIT 1
                """
            ).fetchone()
            if active is not None:
                raise UpdaterError(
                    "UPDATE_IN_PROGRESS",
                    f"another update is active in phase {active['phase']}",
                )

            connection.execute(
                """
                INSERT INTO updater_command (
                    command_id, sequence, compact_jws, compact_jws_sha256,
                    target_version, bom_digest, phase, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    command_id,
                    sequence,
                    compact_jws,
                    digest,
                    target_version,
                    bom_digest,
                    UpdatePhase.AUTHORIZED.value,
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO updater_transition (
                    command_id, from_phase, to_phase, observed_at
                ) VALUES (?, NULL, ?, ?)
                """,
                (command_id, UpdatePhase.AUTHORIZED.value, now),
            )
            row = connection.execute(
                "SELECT * FROM updater_command WHERE command_id = ?", (command_id,)
            ).fetchone()
            return self._row(row), False

    def bind_device_identity(self, identity: dict[str, object]) -> str:
        """Permanently bind this journal to one root-provisioned device tuple."""

        expected = {
            "trustDomain",
            "deviceId",
            "spaceId",
            "productModel",
            "platformProfile",
            "hardwareRevision",
            "architecture",
            "dockerRequired",
        }
        if set(identity) != expected:
            raise ValueError("updater device identity fields are incomplete")
        canonical = json.dumps(
            identity,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        fingerprint = hashlib.sha256(canonical).hexdigest()
        with self._lock, self._transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT meta_value FROM updater_meta "
                "WHERE meta_key = 'deviceBindingFingerprint'"
            ).fetchone()
            if row is not None and str(row[0]) != fingerprint:
                raise UpdaterSecurityError(
                    "DEVICE_BINDING_MISMATCH",
                    "updater journal belongs to another provisioned device identity",
                )
            connection.execute(
                """
                INSERT INTO updater_meta(meta_key, meta_value)
                VALUES ('deviceBindingFingerprint', ?)
                ON CONFLICT(meta_key) DO NOTHING
                """,
                (fingerprint,),
            )
        return fingerprint

    def transition(
        self,
        command_id: str,
        phase: UpdatePhase,
        *,
        allowed_from: Iterable[UpdatePhase],
        candidate_slot: str | None = None,
        previous_slot: str | None = None,
        previous_version: str | None = None,
        release_sequence: int | None = None,
        artifact_digest: str | None = None,
        component_sha: str | None = None,
        config_schema: str | None = None,
        bom_verification_status: str | None = None,
        publisher_key_id: str | None = None,
        health_deadline: str | None = None,
        error_code: str | None = None,
        message: str | None = None,
    ) -> UpdateState:
        allowed = frozenset(allowed_from)
        now = self._clock()
        with self._lock, self._transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT * FROM updater_command WHERE command_id = ?", (command_id,)
            ).fetchone()
            if row is None:
                raise UpdaterError("UPDATE_NOT_FOUND", "update command is not journaled")
            current = self._row(row)
            if current.phase == phase:
                return current
            if current.phase not in allowed:
                raise UpdaterError(
                    "INVALID_PHASE",
                    f"cannot transition {current.phase.value} to {phase.value}",
                )
            connection.execute(
                """
                UPDATE updater_command SET
                    phase = ?,
                    candidate_slot = COALESCE(?, candidate_slot),
                    previous_slot = COALESCE(?, previous_slot),
                    previous_version = COALESCE(?, previous_version),
                    release_sequence = COALESCE(?, release_sequence),
                    artifact_digest = COALESCE(?, artifact_digest),
                    component_sha = COALESCE(?, component_sha),
                    config_schema = COALESCE(?, config_schema),
                    bom_verification_status = COALESCE(?, bom_verification_status),
                    publisher_key_id = COALESCE(?, publisher_key_id),
                    health_deadline = ?,
                    error_code = ?,
                    message = ?,
                    updated_at = ?
                WHERE command_id = ?
                """,
                (
                    phase.value,
                    candidate_slot,
                    previous_slot,
                    previous_version,
                    release_sequence,
                    artifact_digest,
                    component_sha,
                    config_schema,
                    bom_verification_status,
                    publisher_key_id,
                    health_deadline,
                    error_code,
                    message,
                    now,
                    command_id,
                ),
            )
            connection.execute(
                """
                INSERT INTO updater_transition (
                    command_id, from_phase, to_phase, observed_at, error_code, message
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    command_id,
                    current.phase.value,
                    phase.value,
                    now,
                    error_code,
                    message,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM updater_command WHERE command_id = ?", (command_id,)
            ).fetchone()
            return self._row(updated)

    def get(self, command_id: str) -> UpdateState | None:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM updater_command WHERE command_id = ?", (command_id,)
            ).fetchone()
        return self._row(row) if row is not None else None

    def checkpoint(
        self,
        command_id: str,
        *,
        required_phase: UpdatePhase,
        candidate_slot: str | None = None,
        previous_slot: str | None = None,
        previous_version: str | None = None,
        release_sequence: int | None = None,
        artifact_digest: str | None = None,
        component_sha: str | None = None,
        config_schema: str | None = None,
        bom_verification_status: str | None = None,
        publisher_key_id: str | None = None,
        health_deadline: str | None = None,
    ) -> UpdateState:
        """Durably enrich the current phase without fabricating a transition."""

        now = self._clock()
        with self._lock, self._transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT * FROM updater_command WHERE command_id = ?", (command_id,)
            ).fetchone()
            if row is None:
                raise UpdaterError("UPDATE_NOT_FOUND", "update command is not journaled")
            current = self._row(row)
            if current.phase != required_phase:
                raise UpdaterError(
                    "INVALID_PHASE",
                    f"checkpoint requires {required_phase.value}, got {current.phase.value}",
                )
            connection.execute(
                """
                UPDATE updater_command SET
                    candidate_slot = COALESCE(?, candidate_slot),
                    previous_slot = COALESCE(?, previous_slot),
                    previous_version = COALESCE(?, previous_version),
                    release_sequence = COALESCE(?, release_sequence),
                    artifact_digest = COALESCE(?, artifact_digest),
                    component_sha = COALESCE(?, component_sha),
                    config_schema = COALESCE(?, config_schema),
                    bom_verification_status = COALESCE(?, bom_verification_status),
                    publisher_key_id = COALESCE(?, publisher_key_id),
                    health_deadline = COALESCE(?, health_deadline),
                    updated_at = ?
                WHERE command_id = ?
                """,
                (
                    candidate_slot,
                    previous_slot,
                    previous_version,
                    release_sequence,
                    artifact_digest,
                    component_sha,
                    config_schema,
                    bom_verification_status,
                    publisher_key_id,
                    health_deadline,
                    now,
                    command_id,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM updater_command WHERE command_id = ?", (command_id,)
            ).fetchone()
            return self._row(updated)

    def current(self) -> UpdateState | None:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM updater_command ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
        return self._row(row) if row is not None else None

    def current_release_sequence(self) -> int:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT meta_value FROM updater_meta WHERE meta_key = 'currentReleaseSequence'"
            ).fetchone()
        return int(row[0]) if row is not None else 0

    def current_bom_digest(self) -> str | None:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT meta_value FROM updater_meta WHERE meta_key = 'currentBomDigest'"
            ).fetchone()
        return str(row[0]) if row is not None else None

    def commit_release_sequence(self, sequence: int) -> None:
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
            raise ValueError("release sequence must be a positive integer")
        with self._lock, self._transaction(immediate=True) as connection:
            current = connection.execute(
                "SELECT meta_value FROM updater_meta WHERE meta_key = 'currentReleaseSequence'"
            ).fetchone()
            if current is not None and int(current[0]) > sequence:
                raise UpdaterSecurityError(
                    "DOWNGRADE_REJECTED", "cannot lower the committed release sequence"
                )
            connection.execute(
                """
                INSERT INTO updater_meta(meta_key, meta_value)
                VALUES ('currentReleaseSequence', ?)
                ON CONFLICT(meta_key) DO UPDATE SET meta_value = excluded.meta_value
                """,
                (str(sequence),),
            )

    def commit_release(self, *, sequence: int, bom_digest: str) -> None:
        """Atomically advance both anti-downgrade sequence and release identity."""

        if not bom_digest.startswith("sha256:") or len(bom_digest) != 71:
            raise ValueError("bom_digest must use sha256:<64 hex>")
        int(bom_digest[7:], 16)
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
            raise ValueError("release sequence must be a positive integer")
        with self._lock, self._transaction(immediate=True) as connection:
            current = connection.execute(
                "SELECT meta_value FROM updater_meta WHERE meta_key = 'currentReleaseSequence'"
            ).fetchone()
            if current is not None and int(current[0]) > sequence:
                raise UpdaterSecurityError(
                    "DOWNGRADE_REJECTED", "cannot lower the committed release sequence"
                )
            connection.executemany(
                """
                INSERT INTO updater_meta(meta_key, meta_value) VALUES (?, ?)
                ON CONFLICT(meta_key) DO UPDATE SET meta_value = excluded.meta_value
                """,
                (
                    ("currentReleaseSequence", str(sequence)),
                    ("currentBomDigest", bom_digest),
                ),
            )

    def commit_command(self, command_id: str) -> UpdateState:
        """Commit command phase and installed-release identity in one transaction."""

        now = self._clock()
        with self._lock, self._transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT * FROM updater_command WHERE command_id = ?", (command_id,)
            ).fetchone()
            if row is None:
                raise UpdaterError("UPDATE_NOT_FOUND", "update command is not journaled")
            current = self._row(row)
            if current.phase == UpdatePhase.COMMITTED:
                return current
            if current.phase != UpdatePhase.FUNCTIONAL_HEALTHY:
                raise UpdaterError(
                    "INVALID_PHASE",
                    "FUNCTIONAL_HEALTHY is required before commit",
                )
            if current.release_sequence is None:
                raise UpdaterError("INVALID_JOURNAL", "release sequence is missing")

            committed_sequence = connection.execute(
                "SELECT meta_value FROM updater_meta "
                "WHERE meta_key = 'currentReleaseSequence'"
            ).fetchone()
            committed_digest = connection.execute(
                "SELECT meta_value FROM updater_meta "
                "WHERE meta_key = 'currentBomDigest'"
            ).fetchone()
            if (
                committed_sequence is not None
                and int(committed_sequence[0]) > current.release_sequence
            ):
                raise UpdaterSecurityError(
                    "DOWNGRADE_REJECTED", "cannot lower the committed release sequence"
                )
            if (
                committed_sequence is not None
                and int(committed_sequence[0]) == current.release_sequence
                and committed_digest is not None
                and str(committed_digest[0]) != current.bom_digest
            ):
                raise UpdaterSecurityError(
                    "RELEASE_SEQUENCE_COLLISION",
                    "release sequence is already bound to another BOM",
                )

            connection.executemany(
                """
                INSERT INTO updater_meta(meta_key, meta_value) VALUES (?, ?)
                ON CONFLICT(meta_key) DO UPDATE SET meta_value = excluded.meta_value
                """,
                (
                    ("currentReleaseSequence", str(current.release_sequence)),
                    ("currentBomDigest", current.bom_digest),
                ),
            )
            connection.execute(
                """
                UPDATE updater_command SET
                    phase = ?, health_deadline = NULL,
                    error_code = NULL, message = NULL, updated_at = ?
                WHERE command_id = ?
                """,
                (UpdatePhase.COMMITTED.value, now, command_id),
            )
            connection.execute(
                """
                INSERT INTO updater_transition (
                    command_id, from_phase, to_phase, observed_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    command_id,
                    UpdatePhase.FUNCTIONAL_HEALTHY.value,
                    UpdatePhase.COMMITTED.value,
                    now,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM updater_command WHERE command_id = ?", (command_id,)
            ).fetchone()
            return self._row(updated)

    def transitions(self, command_id: str) -> list[tuple[str | None, str]]:
        with self._transaction() as connection:
            rows = connection.execute(
                """
                SELECT from_phase, to_phase FROM updater_transition
                WHERE command_id = ? ORDER BY id
                """,
                (command_id,),
            ).fetchall()
        return [(row[0], str(row[1])) for row in rows]

    @staticmethod
    def _row(row: sqlite3.Row) -> UpdateState:
        return UpdateState(
            command_id=str(row["command_id"]),
            sequence=int(row["sequence"]),
            compact_jws=str(row["compact_jws"]),
            compact_jws_sha256=str(row["compact_jws_sha256"]),
            target_version=str(row["target_version"]),
            bom_digest=str(row["bom_digest"]),
            phase=UpdatePhase(str(row["phase"])),
            candidate_slot=row["candidate_slot"],
            previous_slot=row["previous_slot"],
            previous_version=row["previous_version"],
            release_sequence=(
                int(row["release_sequence"])
                if row["release_sequence"] is not None
                else None
            ),
            artifact_digest=row["artifact_digest"],
            component_sha=row["component_sha"],
            config_schema=row["config_schema"],
            bom_verification_status=row["bom_verification_status"],
            publisher_key_id=row["publisher_key_id"],
            health_deadline=row["health_deadline"],
            error_code=row["error_code"],
            message=row["message"],
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )
