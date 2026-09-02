from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from nuvion_app.inference.command_inbox import (
    COMMAND_STATUS_FAILED,
    COMMAND_STATUS_IN_PROGRESS,
    COMMAND_STATUS_ROLLED_BACK,
    COMMAND_STATUS_SUCCEEDED,
    TERMINAL_COMMAND_STATUSES,
    CommandAck,
    CommandEffectOutcome,
    DurableCommandInbox,
    utc_now_iso,
)
from nuvion_app.inference.command_observation import (
    CommandObservationError,
    DurableCommandObservationOutbox,
)
from nuvion_app.inference.fleet_command import VerifiedFleetCommand

JOB_PHASE_PENDING = "PENDING"
JOB_PHASE_CLAIMED = "CLAIMED"
JOB_PHASE_APPLYING = "APPLYING"
JOB_PHASE_VERIFYING = "VERIFYING"
JOB_PHASE_WAITING_RESTART = "WAITING_RESTART"
JOB_PHASE_SUCCEEDED = COMMAND_STATUS_SUCCEEDED
JOB_PHASE_FAILED = COMMAND_STATUS_FAILED
JOB_PHASE_ROLLED_BACK = COMMAND_STATUS_ROLLED_BACK
JOB_PHASE_SUPERSEDED = "SUPERSEDED"

ACTIVE_JOB_PHASES = frozenset(
    {
        JOB_PHASE_PENDING,
        JOB_PHASE_CLAIMED,
        JOB_PHASE_APPLYING,
        JOB_PHASE_VERIFYING,
        JOB_PHASE_WAITING_RESTART,
    }
)
TERMINAL_JOB_PHASES = frozenset(
    {
        JOB_PHASE_SUCCEEDED,
        JOB_PHASE_FAILED,
        JOB_PHASE_ROLLED_BACK,
        JOB_PHASE_SUPERSEDED,
    }
)


class EffectFenceStale(RuntimeError):
    """Raised before an actuator mutation when the durable lease lost authority."""


def effect_domains(command: VerifiedFleetCommand) -> tuple[str, ...]:
    """Return stable actuator domains shared by otherwise independent commands."""

    domains: set[str] = set()
    if command.command_type == "STREAM_POLICY":
        domains.add("video_encoder")
    elif command.command_type == "CONFIG_APPLY":
        domains.add("settings")
        if isinstance(command.payload.get("video"), dict):
            domains.add("video_encoder")
        if isinstance(command.payload.get("model"), dict):
            domains.add("inference_model")
    else:
        domains.add(command.command_type.lower())
    return tuple(sorted(domains))


@dataclass(frozen=True)
class ReconcileJob:
    command_id: str
    command_type: str
    sequence: int
    phase: str
    attempts: int
    lease_owner: str | None
    lease_expires_at: float | None
    checkpoint: dict[str, Any] | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class AppliedState:
    command_type: str
    command_id: str
    sequence: int
    payload_hash: str
    payload: dict[str, Any]
    reported_state: dict[str, Any]
    applied_at: str
    updated_at: str


@dataclass(frozen=True)
class EffectFence:
    store: DurableReconcileStore
    command_id: str
    owner: str
    sequence: int
    generations: tuple[tuple[str, int], ...]

    @property
    def domains(self) -> tuple[str, ...]:
        return tuple(domain for domain, _generation in self.generations)

    def assert_current(self) -> None:
        self.store.assert_effect_fence(self)


class DurableReconcileStore:
    """Per-command desired-state jobs sharing the durable inbox database.

    Only short SQLite mutations live here.  A coordinator claims a job, releases
    the database transaction, performs the external effect, and then checkpoints
    or completes it in a new short transaction.
    """

    def __init__(
        self,
        inbox: DurableCommandInbox,
        *,
        monotonic_clock: Callable[[], float] | None = None,
        wall_clock: Callable[[], str] | None = None,
        observation_outbox: DurableCommandObservationOutbox | None = None,
    ) -> None:
        self.inbox = inbox
        self._monotonic_clock = monotonic_clock or time.time
        self._wall_clock = wall_clock or utc_now_iso
        self.observation_outbox = observation_outbox
        self._initialize()

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
        return DurableReconcileStore._json(value) if value is not None else None

    def _initialize(self) -> None:
        with self.inbox.transaction(immediate=True) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS fleet_desired_state (
                    command_type TEXT PRIMARY KEY,
                    command_id TEXT NOT NULL UNIQUE,
                    sequence INTEGER NOT NULL UNIQUE,
                    payload_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    accepted_at TEXT NOT NULL,
                    FOREIGN KEY(command_id) REFERENCES command_inbox(command_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS fleet_reconcile_job (
                    command_id TEXT PRIMARY KEY,
                    command_type TEXT NOT NULL,
                    sequence INTEGER NOT NULL UNIQUE,
                    phase TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    lease_owner TEXT,
                    lease_expires_at REAL,
                    checkpoint_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(command_id) REFERENCES command_inbox(command_id)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_fleet_reconcile_claim
                ON fleet_reconcile_job(phase, lease_expires_at, sequence)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS fleet_reconcile_history (
                    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    command_id TEXT NOT NULL,
                    from_phase TEXT,
                    to_phase TEXT NOT NULL,
                    code TEXT,
                    message TEXT,
                    state_json TEXT,
                    observed_at TEXT NOT NULL,
                    FOREIGN KEY(command_id) REFERENCES command_inbox(command_id)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_fleet_reconcile_history_command
                ON fleet_reconcile_history(command_id, history_id)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS fleet_applied_state (
                    command_type TEXT PRIMARY KEY,
                    command_id TEXT NOT NULL UNIQUE,
                    sequence INTEGER NOT NULL UNIQUE,
                    payload_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    reported_state_json TEXT NOT NULL,
                    applied_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(command_id) REFERENCES command_inbox(command_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS fleet_actuator_fence (
                    domain TEXT PRIMARY KEY,
                    generation INTEGER NOT NULL,
                    command_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    lease_owner TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(command_id) REFERENCES command_inbox(command_id)
                )
                """
            )

    def stage_verified(
        self,
        command: VerifiedFleetCommand,
        connection: sqlite3.Connection,
    ) -> CommandEffectOutcome:
        """Stage one desired state atomically with its IN_PROGRESS lifecycle."""

        now = self._wall_clock()
        current = connection.execute(
            "SELECT command_id, sequence FROM fleet_desired_state WHERE command_type = ?",
            (command.command_type,),
        ).fetchone()
        if current is not None and str(current["command_id"]) == command.command_id:
            return CommandEffectOutcome.deferred()

        if current is not None and int(current["sequence"]) >= command.sequence:
            self._insert_job_if_missing(
                connection,
                command,
                phase=JOB_PHASE_SUPERSEDED,
                now=now,
            )
            return CommandEffectOutcome(
                status=COMMAND_STATUS_FAILED,
                code="SUPERSEDED",
                message=(
                    f"newer {command.command_type} sequence={int(current['sequence'])} "
                    "is already desired"
                ),
                reported_state={
                    "supersededByCommandId": str(current["command_id"]),
                    "supersededBySequence": int(current["sequence"]),
                },
            )

        if self.observation_outbox is not None:
            try:
                self.observation_outbox.reserve_terminal_in_transaction(
                    connection,
                    command_id=command.command_id,
                    slots=1,
                )
            except CommandObservationError as exc:
                return CommandEffectOutcome(
                    status=COMMAND_STATUS_FAILED,
                    code=exc.code,
                    message=str(exc)[:1000],
                    reported_state={**command.payload, "health": "NOT_APPLIED"},
                )

        active_rows = connection.execute(
            """
            SELECT command_id, phase, sequence
            FROM fleet_reconcile_job
            WHERE command_type = ?
              AND command_id != ?
              AND phase IN (?, ?, ?, ?, ?)
            ORDER BY sequence ASC
            """,
            (
                command.command_type,
                command.command_id,
                JOB_PHASE_PENDING,
                JOB_PHASE_CLAIMED,
                JOB_PHASE_APPLYING,
                JOB_PHASE_VERIFYING,
                JOB_PHASE_WAITING_RESTART,
            ),
        ).fetchall()
        for row in active_rows:
            superseded_id = str(row["command_id"])
            inbox_row = connection.execute(
                "SELECT status FROM command_inbox WHERE command_id = ?",
                (superseded_id,),
            ).fetchone()
            superseded_state = {
                "supersededByCommandId": command.command_id,
                "supersededBySequence": command.sequence,
                "health": "SUPERSEDED",
            }
            payload_row = connection.execute(
                "SELECT payload_json FROM command_inbox WHERE command_id = ?",
                (superseded_id,),
            ).fetchone()
            if payload_row is not None:
                superseded_state = {
                    **json.loads(str(payload_row["payload_json"])),
                    **superseded_state,
                }
            if inbox_row is not None and str(inbox_row["status"]) == COMMAND_STATUS_IN_PROGRESS:
                self.inbox.transition_in_transaction(
                    connection,
                    superseded_id,
                    COMMAND_STATUS_FAILED,
                    code="SUPERSEDED",
                    message=(
                        f"superseded by {command.command_type} "
                        f"sequence={command.sequence}"
                    ),
                    reported_state=superseded_state,
                )
                if self.observation_outbox is not None:
                    self.observation_outbox.discard_command_in_transaction(
                        connection,
                        command_id=superseded_id,
                    )
            connection.execute(
                """
                UPDATE fleet_reconcile_job
                SET phase = ?, lease_owner = NULL, lease_expires_at = NULL,
                    checkpoint_json = ?, updated_at = ?
                WHERE command_id = ?
                """,
                (
                    JOB_PHASE_SUPERSEDED,
                    self._json(
                        {
                            "supersededByCommandId": command.command_id,
                            "supersededBySequence": command.sequence,
                        }
                    ),
                    now,
                    superseded_id,
                ),
            )
            self._append_history(
                connection,
                superseded_id,
                from_phase=str(row["phase"]),
                to_phase=JOB_PHASE_SUPERSEDED,
                code="SUPERSEDED",
                message=f"superseded by sequence={command.sequence}",
                state={
                    "supersededByCommandId": command.command_id,
                    "supersededBySequence": command.sequence,
                },
                now=now,
            )

        payload_json = self._json(command.payload)
        connection.execute(
            """
            INSERT INTO fleet_desired_state (
                command_type, command_id, sequence, payload_hash, payload_json,
                accepted_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(command_type) DO UPDATE SET
                command_id = excluded.command_id,
                sequence = excluded.sequence,
                payload_hash = excluded.payload_hash,
                payload_json = excluded.payload_json,
                accepted_at = excluded.accepted_at
            WHERE excluded.sequence > fleet_desired_state.sequence
            """,
            (
                command.command_type,
                command.command_id,
                command.sequence,
                command.payload_hash,
                payload_json,
                now,
            ),
        )
        inserted = self._insert_job_if_missing(
            connection,
            command,
            phase=JOB_PHASE_PENDING,
            now=now,
        )
        if inserted:
            self._append_history(
                connection,
                command.command_id,
                from_phase=None,
                to_phase=JOB_PHASE_PENDING,
                now=now,
            )
        return CommandEffectOutcome.deferred()

    def _insert_job_if_missing(
        self,
        connection: sqlite3.Connection,
        command: VerifiedFleetCommand,
        *,
        phase: str,
        now: str,
    ) -> bool:
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO fleet_reconcile_job (
                command_id, command_type, sequence, phase, attempts,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, 0, ?, ?)
            """,
            (
                command.command_id,
                command.command_type,
                command.sequence,
                phase,
                now,
                now,
            ),
        )
        return cursor.rowcount == 1

    def claim_next(self, *, owner: str, lease_seconds: float) -> ReconcileJob | None:
        normalized_owner = str(owner or "").strip()
        if not normalized_owner:
            raise ValueError("reconcile lease owner must be non-empty")
        safe_lease = max(1.0, min(float(lease_seconds), 3600.0))
        now_epoch = self._monotonic_clock()
        expires_at = now_epoch + safe_lease
        now = self._wall_clock()
        with self.inbox.transaction(immediate=True) as connection:
            row = connection.execute(
                """
                SELECT * FROM fleet_reconcile_job
                WHERE phase = ?
                   OR (
                        phase IN (?, ?, ?)
                        AND lease_expires_at IS NOT NULL
                        AND lease_expires_at <= ?
                   )
                ORDER BY sequence ASC
                LIMIT 1
                """,
                (
                    JOB_PHASE_PENDING,
                    JOB_PHASE_CLAIMED,
                    JOB_PHASE_APPLYING,
                    JOB_PHASE_VERIFYING,
                    now_epoch,
                ),
            ).fetchone()
            if row is None:
                return None
            previous_phase = str(row["phase"])
            connection.execute(
                """
                UPDATE fleet_reconcile_job
                SET phase = ?, attempts = attempts + 1, lease_owner = ?,
                    lease_expires_at = ?, updated_at = ?
                WHERE command_id = ?
                """,
                (
                    JOB_PHASE_CLAIMED,
                    normalized_owner,
                    expires_at,
                    now,
                    str(row["command_id"]),
                ),
            )
            self._append_history(
                connection,
                str(row["command_id"]),
                from_phase=previous_phase,
                to_phase=JOB_PHASE_CLAIMED,
                state={"leaseOwner": normalized_owner, "leaseExpiresAt": expires_at},
                now=now,
            )
            claimed = connection.execute(
                "SELECT * FROM fleet_reconcile_job WHERE command_id = ?",
                (str(row["command_id"]),),
            ).fetchone()
        return self._row_to_job(claimed)

    def checkpoint(
        self,
        command_id: str,
        *,
        owner: str,
        expected_phase: str,
        next_phase: str,
        state: Mapping[str, Any] | None = None,
        lease_seconds: float = 30.0,
    ) -> ReconcileJob:
        if next_phase not in ACTIVE_JOB_PHASES:
            raise ValueError(f"invalid active reconcile phase: {next_phase}")
        now_epoch = self._monotonic_clock()
        now = self._wall_clock()
        with self.inbox.transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT * FROM fleet_reconcile_job WHERE command_id = ?",
                (command_id,),
            ).fetchone()
            if row is None:
                raise LookupError(f"reconcile job not found: {command_id}")
            if str(row["phase"]) != expected_phase or str(row["lease_owner"] or "") != owner:
                raise RuntimeError("reconcile lease or phase changed before checkpoint")
            connection.execute(
                """
                UPDATE fleet_reconcile_job
                SET phase = ?, checkpoint_json = ?, lease_expires_at = ?,
                    updated_at = ?
                WHERE command_id = ?
                """,
                (
                    next_phase,
                    self._optional_json(state),
                    now_epoch + max(1.0, min(float(lease_seconds), 3600.0)),
                    now,
                    command_id,
                ),
            )
            self._append_history(
                connection,
                command_id,
                from_phase=expected_phase,
                to_phase=next_phase,
                state=state,
                now=now,
            )
            updated = connection.execute(
                "SELECT * FROM fleet_reconcile_job WHERE command_id = ?",
                (command_id,),
            ).fetchone()
        return self._row_to_job(updated)

    def begin_effect(
        self,
        command: VerifiedFleetCommand,
        *,
        owner: str,
        lease_seconds: float,
    ) -> tuple[ReconcileJob, EffectFence]:
        """Atomically enter APPLYING and issue per-actuator generation fences."""

        now_epoch = self._monotonic_clock()
        now = self._wall_clock()
        expires_at = now_epoch + max(1.0, min(float(lease_seconds), 3600.0))
        domains = effect_domains(command)
        generations: list[tuple[str, int]] = []
        with self.inbox.transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT * FROM fleet_reconcile_job WHERE command_id = ?",
                (command.command_id,),
            ).fetchone()
            if row is None:
                raise LookupError(f"reconcile job not found: {command.command_id}")
            if (
                str(row["phase"]) != JOB_PHASE_CLAIMED
                or str(row["lease_owner"] or "") != owner
                or row["lease_expires_at"] is None
                or float(row["lease_expires_at"]) <= now_epoch
            ):
                raise EffectFenceStale(
                    "reconcile lease changed or expired before effect fencing"
                )
            desired = connection.execute(
                "SELECT command_id FROM fleet_desired_state WHERE command_type = ?",
                (command.command_type,),
            ).fetchone()
            if desired is None or str(desired["command_id"]) != command.command_id:
                raise EffectFenceStale("newer desired state replaced this command")
            connection.execute(
                """
                UPDATE fleet_reconcile_job
                SET phase = ?, checkpoint_json = ?, lease_expires_at = ?,
                    updated_at = ?
                WHERE command_id = ?
                """,
                (
                    JOB_PHASE_APPLYING,
                    self._json({"attempt": int(row["attempts"])}),
                    expires_at,
                    now,
                    command.command_id,
                ),
            )
            for domain in domains:
                previous = connection.execute(
                    "SELECT generation FROM fleet_actuator_fence WHERE domain = ?",
                    (domain,),
                ).fetchone()
                generation = int(previous["generation"]) + 1 if previous else 1
                connection.execute(
                    """
                    INSERT INTO fleet_actuator_fence (
                        domain, generation, command_id, sequence, lease_owner,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(domain) DO UPDATE SET
                        generation = excluded.generation,
                        command_id = excluded.command_id,
                        sequence = excluded.sequence,
                        lease_owner = excluded.lease_owner,
                        updated_at = excluded.updated_at
                    """,
                    (
                        domain,
                        generation,
                        command.command_id,
                        command.sequence,
                        owner,
                        now,
                    ),
                )
                generations.append((domain, generation))
            self._append_history(
                connection,
                command.command_id,
                from_phase=JOB_PHASE_CLAIMED,
                to_phase=JOB_PHASE_APPLYING,
                state={
                    "attempt": int(row["attempts"]),
                    "effectFence": dict(generations),
                },
                now=now,
            )
            updated = connection.execute(
                "SELECT * FROM fleet_reconcile_job WHERE command_id = ?",
                (command.command_id,),
            ).fetchone()
        return self._row_to_job(updated), EffectFence(
            store=self,
            command_id=command.command_id,
            owner=owner,
            sequence=command.sequence,
            generations=tuple(generations),
        )

    def assert_effect_fence(self, fence: EffectFence) -> None:
        now_epoch = self._monotonic_clock()
        with self.inbox.transaction() as connection:
            job = connection.execute(
                "SELECT phase, lease_owner, lease_expires_at, command_type "
                "FROM fleet_reconcile_job WHERE command_id = ?",
                (fence.command_id,),
            ).fetchone()
            if job is None:
                raise EffectFenceStale("effect lease is no longer current")
            phase = str(job["phase"])
            applying_is_current = (
                phase == JOB_PHASE_APPLYING
                and str(job["lease_owner"] or "") == fence.owner
                and job["lease_expires_at"] is not None
                and float(job["lease_expires_at"]) > now_epoch
            )
            applied_is_current = False
            if phase == JOB_PHASE_SUCCEEDED:
                applied = connection.execute(
                    "SELECT command_id, sequence FROM fleet_applied_state "
                    "WHERE command_type = ?",
                    (str(job["command_type"]),),
                ).fetchone()
                applied_is_current = (
                    applied is not None
                    and str(applied["command_id"]) == fence.command_id
                    and int(applied["sequence"]) == fence.sequence
                )
            if not applying_is_current and not applied_is_current:
                raise EffectFenceStale("effect lease is no longer current")
            desired = connection.execute(
                "SELECT command_id, sequence FROM fleet_desired_state "
                "WHERE command_type = ?",
                (str(job["command_type"]),),
            ).fetchone()
            desired_matches = (
                desired is not None
                and str(desired["command_id"]) == fence.command_id
                and int(desired["sequence"]) == fence.sequence
            )
            desired_failed = False
            if desired is not None and not desired_matches and applied_is_current:
                desired_job = connection.execute(
                    "SELECT phase FROM fleet_reconcile_job WHERE command_id = ?",
                    (str(desired["command_id"]),),
                ).fetchone()
                desired_failed = desired_job is not None and str(
                    desired_job["phase"]
                ) in {
                    JOB_PHASE_FAILED,
                    JOB_PHASE_ROLLED_BACK,
                    JOB_PHASE_SUPERSEDED,
                }
            if not desired_matches and not desired_failed:
                raise EffectFenceStale("effect was superseded by newer desired state")
            for domain, generation in fence.generations:
                current = connection.execute(
                    "SELECT generation, command_id, lease_owner "
                    "FROM fleet_actuator_fence WHERE domain = ?",
                    (domain,),
                ).fetchone()
                if (
                    current is None
                    or int(current["generation"]) != generation
                    or str(current["command_id"]) != fence.command_id
                    or str(current["lease_owner"]) != fence.owner
                ):
                    raise EffectFenceStale(
                        f"actuator generation changed for domain={domain}"
                    )

    def release_effect_for_retry(
        self,
        command_id: str,
        *,
        owner: str,
        code: str,
    ) -> bool:
        now = self._wall_clock()
        with self.inbox.transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT phase, lease_owner FROM fleet_reconcile_job WHERE command_id = ?",
                (command_id,),
            ).fetchone()
            if (
                row is None
                or str(row["phase"]) not in {JOB_PHASE_CLAIMED, JOB_PHASE_APPLYING}
                or str(row["lease_owner"] or "") != owner
            ):
                return False
            previous = str(row["phase"])
            connection.execute(
                """
                UPDATE fleet_reconcile_job
                SET phase = ?, lease_owner = NULL, lease_expires_at = NULL,
                    updated_at = ?
                WHERE command_id = ?
                """,
                (JOB_PHASE_PENDING, now, command_id),
            )
            self._append_history(
                connection,
                command_id,
                from_phase=previous,
                to_phase=JOB_PHASE_PENDING,
                code=code,
                message="actuator ownership was not available; retrying",
                now=now,
            )
        return True

    def fence_applied_state(
        self,
        command: VerifiedFleetCommand,
        *,
        owner: str,
    ) -> EffectFence:
        """Issue a new actuator generation for idempotent controller restore."""

        now = self._wall_clock()
        generations: list[tuple[str, int]] = []
        with self.inbox.transaction(immediate=True) as connection:
            applied = connection.execute(
                "SELECT command_id, sequence FROM fleet_applied_state "
                "WHERE command_type = ?",
                (command.command_type,),
            ).fetchone()
            desired = connection.execute(
                """
                SELECT d.command_id, d.sequence, j.phase
                FROM fleet_desired_state d
                JOIN fleet_reconcile_job j ON j.command_id = d.command_id
                WHERE d.command_type = ?
                """,
                (command.command_type,),
            ).fetchone()
            desired_matches = (
                desired is not None
                and str(desired["command_id"]) == command.command_id
                and int(desired["sequence"]) == command.sequence
            )
            desired_failed = desired is not None and str(desired["phase"]) in {
                JOB_PHASE_FAILED,
                JOB_PHASE_ROLLED_BACK,
                JOB_PHASE_SUPERSEDED,
            }
            if (
                applied is None
                or str(applied["command_id"]) != command.command_id
                or int(applied["sequence"]) != command.sequence
                or (not desired_matches and not desired_failed)
            ):
                raise EffectFenceStale("applied controller is no longer authoritative")
            for domain in effect_domains(command):
                previous = connection.execute(
                    "SELECT generation FROM fleet_actuator_fence WHERE domain = ?",
                    (domain,),
                ).fetchone()
                generation = int(previous["generation"]) + 1 if previous else 1
                connection.execute(
                    """
                    INSERT INTO fleet_actuator_fence (
                        domain, generation, command_id, sequence, lease_owner,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(domain) DO UPDATE SET
                        generation = excluded.generation,
                        command_id = excluded.command_id,
                        sequence = excluded.sequence,
                        lease_owner = excluded.lease_owner,
                        updated_at = excluded.updated_at
                    """,
                    (
                        domain,
                        generation,
                        command.command_id,
                        command.sequence,
                        owner,
                        now,
                    ),
                )
                generations.append((domain, generation))
        return EffectFence(
            store=self,
            command_id=command.command_id,
            owner=owner,
            sequence=command.sequence,
            generations=tuple(generations),
        )

    def encoder_owned_by_stream_policy(self) -> bool:
        with self.inbox.transaction() as connection:
            row = connection.execute(
                """
                SELECT d.payload_json, j.phase
                FROM fleet_desired_state d
                JOIN fleet_reconcile_job j ON j.command_id = d.command_id
                WHERE d.command_type = 'STREAM_POLICY'
                """
            ).fetchone()
        if row is None or str(row["phase"]) in {
            JOB_PHASE_FAILED,
            JOB_PHASE_ROLLED_BACK,
            JOB_PHASE_SUPERSEDED,
        }:
            return False
        payload = json.loads(str(row["payload_json"]))
        return payload.get("mode") in {"FIXED", "ADAPTIVE"}

    def finish(
        self,
        command: VerifiedFleetCommand,
        *,
        owner: str,
        outcome: CommandEffectOutcome,
    ) -> CommandAck | None:
        status = str(outcome.status or "").strip().upper()
        if status not in {
            COMMAND_STATUS_SUCCEEDED,
            COMMAND_STATUS_FAILED,
            COMMAND_STATUS_ROLLED_BACK,
        }:
            raise ValueError("reconciler outcome must be terminal")
        now = self._wall_clock()
        with self.inbox.transaction(immediate=True) as connection:
            job = connection.execute(
                "SELECT * FROM fleet_reconcile_job WHERE command_id = ?",
                (command.command_id,),
            ).fetchone()
            if job is None:
                raise LookupError(f"reconcile job not found: {command.command_id}")
            phase = str(job["phase"])
            if phase in TERMINAL_JOB_PHASES:
                return None
            if str(job["lease_owner"] or "") != owner:
                raise RuntimeError("reconcile lease changed before completion")
            inbox_row = connection.execute(
                "SELECT status FROM command_inbox WHERE command_id = ?",
                (command.command_id,),
            ).fetchone()
            if inbox_row is None:
                raise LookupError(f"command disappeared: {command.command_id}")
            if str(inbox_row["status"]) in TERMINAL_COMMAND_STATUSES:
                return None

            reported_state = {
                **command.payload,
                **dict(outcome.reported_state or {}),
            }
            connection.execute(
                """
                UPDATE fleet_reconcile_job
                SET phase = ?, checkpoint_json = ?, lease_owner = NULL,
                    lease_expires_at = NULL, updated_at = ?
                WHERE command_id = ?
                """,
                (
                    status,
                    self._json(reported_state),
                    now,
                    command.command_id,
                ),
            )
            self._append_history(
                connection,
                command.command_id,
                from_phase=phase,
                to_phase=status,
                code=outcome.code,
                message=outcome.message,
                state=reported_state,
                now=now,
            )
            if status == COMMAND_STATUS_SUCCEEDED:
                connection.execute(
                    """
                    INSERT INTO fleet_applied_state (
                        command_type, command_id, sequence, payload_hash,
                        payload_json, reported_state_json, applied_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(command_type) DO UPDATE SET
                        command_id = excluded.command_id,
                        sequence = excluded.sequence,
                        payload_hash = excluded.payload_hash,
                        payload_json = excluded.payload_json,
                        reported_state_json = excluded.reported_state_json,
                        applied_at = excluded.applied_at,
                        updated_at = excluded.updated_at
                    WHERE excluded.sequence >= fleet_applied_state.sequence
                    """,
                    (
                        command.command_type,
                        command.command_id,
                        command.sequence,
                        command.payload_hash,
                        self._json(command.payload),
                        self._json(reported_state),
                        now,
                        now,
                    ),
                )
            ack = self.inbox.transition_in_transaction(
                connection,
                command.command_id,
                status,
                code=outcome.code,
                message=outcome.message,
                reported_state=reported_state,
            )
            if self.observation_outbox is not None:
                if status == COMMAND_STATUS_SUCCEEDED:
                    self.observation_outbox.enqueue_in_transaction(
                        connection,
                        command_id=command.command_id,
                        sequence=command.sequence,
                        command_type=command.command_type,
                        reported_state=reported_state,
                        terminal=True,
                        use_reservation=True,
                    )
                else:
                    self.observation_outbox.discard_command_in_transaction(
                        connection,
                        command_id=command.command_id,
                    )
            return ack

    def defer_for_restart(
        self,
        command: VerifiedFleetCommand,
        *,
        owner: str,
        reported_state: Mapping[str, Any],
        checkpoint: Mapping[str, Any],
    ) -> None:
        now = self._wall_clock()
        state = {**command.payload, **dict(reported_state)}
        with self.inbox.transaction(immediate=True) as connection:
            job = connection.execute(
                "SELECT * FROM fleet_reconcile_job WHERE command_id = ?",
                (command.command_id,),
            ).fetchone()
            if job is None:
                raise LookupError(f"reconcile job not found: {command.command_id}")
            if str(job["phase"]) in TERMINAL_JOB_PHASES:
                return
            if str(job["lease_owner"] or "") != owner:
                raise RuntimeError("reconcile lease changed before restart deferral")
            connection.execute(
                """
                UPDATE fleet_reconcile_job
                SET phase = ?, checkpoint_json = ?, lease_owner = NULL,
                    lease_expires_at = NULL, updated_at = ?
                WHERE command_id = ?
                """,
                (
                    JOB_PHASE_WAITING_RESTART,
                    self._json(checkpoint),
                    now,
                    command.command_id,
                ),
            )
            self._append_history(
                connection,
                command.command_id,
                from_phase=str(job["phase"]),
                to_phase=JOB_PHASE_WAITING_RESTART,
                state=state,
                now=now,
            )

    def defer_for_retry(
        self,
        command: VerifiedFleetCommand,
        *,
        owner: str,
        reported_state: Mapping[str, Any],
        checkpoint: Mapping[str, Any],
        retry_after_seconds: float,
    ) -> None:
        """Persist a non-restart effect retry behind a capped lease deadline."""

        now_epoch = self._monotonic_clock()
        now = self._wall_clock()
        retry_at = now_epoch + max(1.0, min(float(retry_after_seconds), 300.0))
        state = {**command.payload, **dict(reported_state)}
        with self.inbox.transaction(immediate=True) as connection:
            job = connection.execute(
                "SELECT * FROM fleet_reconcile_job WHERE command_id = ?",
                (command.command_id,),
            ).fetchone()
            if job is None:
                raise LookupError(f"reconcile job not found: {command.command_id}")
            if str(job["phase"]) in TERMINAL_JOB_PHASES:
                return
            if str(job["lease_owner"] or "") != owner:
                raise RuntimeError("reconcile lease changed before retry deferral")
            connection.execute(
                """
                UPDATE fleet_reconcile_job
                SET phase = ?, checkpoint_json = ?, lease_owner = NULL,
                    lease_expires_at = ?, updated_at = ?
                WHERE command_id = ?
                """,
                (
                    JOB_PHASE_VERIFYING,
                    self._json(checkpoint),
                    retry_at,
                    now,
                    command.command_id,
                ),
            )
            self._append_history(
                connection,
                command.command_id,
                from_phase=str(job["phase"]),
                to_phase=JOB_PHASE_VERIFYING,
                code="EFFECT_RETRY_DEFERRED",
                message="external effect will retry without restarting the Agent",
                state={**state, "retryAt": retry_at},
                now=now,
            )

    def requeue_waiting_restart(self, *, process_instance_id: str) -> int:
        now = self._wall_clock()
        requeued = 0
        with self.inbox.transaction(immediate=True) as connection:
            rows = connection.execute(
                """
                SELECT command_id, checkpoint_json
                FROM fleet_reconcile_job
                WHERE phase = ?
                ORDER BY sequence ASC
                """,
                (JOB_PHASE_WAITING_RESTART,),
            ).fetchall()
            for row in rows:
                checkpoint = (
                    json.loads(str(row["checkpoint_json"]))
                    if row["checkpoint_json"]
                    else {}
                )
                if checkpoint.get("processInstanceId") == process_instance_id:
                    continue
                connection.execute(
                    """
                    UPDATE fleet_reconcile_job
                    SET phase = ?, updated_at = ?
                    WHERE command_id = ? AND phase = ?
                    """,
                    (
                        JOB_PHASE_PENDING,
                        now,
                        str(row["command_id"]),
                        JOB_PHASE_WAITING_RESTART,
                    ),
                )
                self._append_history(
                    connection,
                    str(row["command_id"]),
                    from_phase=JOB_PHASE_WAITING_RESTART,
                    to_phase=JOB_PHASE_PENDING,
                    state={"resumedByProcessInstanceId": process_instance_id},
                    now=now,
                )
                requeued += 1
        return requeued

    def retry_restart_request(self, command_id: str) -> bool:
        now = self._wall_clock()
        with self.inbox.transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT phase FROM fleet_reconcile_job WHERE command_id = ?",
                (command_id,),
            ).fetchone()
            if row is None or str(row["phase"]) != JOB_PHASE_WAITING_RESTART:
                return False
            connection.execute(
                """
                UPDATE fleet_reconcile_job
                SET phase = ?, lease_owner = NULL, lease_expires_at = NULL,
                    updated_at = ?
                WHERE command_id = ? AND phase = ?
                """,
                (
                    JOB_PHASE_PENDING,
                    now,
                    command_id,
                    JOB_PHASE_WAITING_RESTART,
                ),
            )
            self._append_history(
                connection,
                command_id,
                from_phase=JOB_PHASE_WAITING_RESTART,
                to_phase=JOB_PHASE_PENDING,
                code="RESTART_REQUEST_RETRY",
                message="supervisor restart request was not accepted",
                now=now,
            )
        return True

    def update_applied_reported_state(
        self,
        *,
        command_type: str,
        command_id: str,
        reported_state: Mapping[str, Any],
    ) -> bool:
        now = self._wall_clock()
        with self.inbox.transaction(immediate=True) as connection:
            row = connection.execute(
                """
                SELECT sequence, payload_json, reported_state_json
                FROM fleet_applied_state
                WHERE command_type = ? AND command_id = ?
                """,
                (command_type, command_id),
            ).fetchone()
            if row is None:
                return False
            state = {
                **json.loads(str(row["payload_json"])),
                **dict(reported_state),
            }
            cursor = connection.execute(
                """
                UPDATE fleet_applied_state
                SET reported_state_json = ?, updated_at = ?
                WHERE command_type = ? AND command_id = ?
                """,
                (
                    self._json(state),
                    now,
                    command_type,
                    command_id,
                ),
            )
            if cursor.rowcount == 1 and self.observation_outbox is not None:
                self.observation_outbox.enqueue_in_transaction(
                    connection,
                    command_id=command_id,
                    sequence=int(row["sequence"]),
                    command_type=command_type,
                    reported_state=state,
                )
        return cursor.rowcount == 1

    def observe_state(
        self,
        *,
        command: VerifiedFleetCommand,
        reported_state: Mapping[str, Any],
    ) -> None:
        if self.observation_outbox is None:
            return
        state = {**command.payload, **dict(reported_state)}
        with self.inbox.transaction(immediate=True) as connection:
            inbox_row = connection.execute(
                "SELECT status FROM command_inbox WHERE command_id = ?",
                (command.command_id,),
            ).fetchone()
            if (
                inbox_row is None
                or str(inbox_row["status"]) != COMMAND_STATUS_SUCCEEDED
            ):
                self.observation_outbox.discard_command_in_transaction(
                    connection,
                    command_id=command.command_id,
                )
                return
            self.observation_outbox.enqueue_in_transaction(
                connection,
                command_id=command.command_id,
                sequence=command.sequence,
                command_type=command.command_type,
                reported_state=state,
            )

    def get_job(self, command_id: str) -> ReconcileJob | None:
        with self.inbox.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM fleet_reconcile_job WHERE command_id = ?",
                (command_id,),
            ).fetchone()
        return self._row_to_job(row) if row is not None else None

    def history(self, command_id: str) -> list[dict[str, Any]]:
        with self.inbox.transaction() as connection:
            rows = connection.execute(
                """
                SELECT from_phase, to_phase, code, message, state_json, observed_at
                FROM fleet_reconcile_history
                WHERE command_id = ?
                ORDER BY history_id ASC
                """,
                (command_id,),
            ).fetchall()
        return [
            {
                "fromPhase": str(row["from_phase"]) if row["from_phase"] else None,
                "toPhase": str(row["to_phase"]),
                "code": str(row["code"]) if row["code"] else None,
                "message": str(row["message"]) if row["message"] else None,
                "state": json.loads(str(row["state_json"])) if row["state_json"] else None,
                "observedAt": str(row["observed_at"]),
            }
            for row in rows
        ]

    def applied_state(self, command_type: str) -> AppliedState | None:
        with self.inbox.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM fleet_applied_state WHERE command_type = ?",
                (command_type,),
            ).fetchone()
        if row is None:
            return None
        return AppliedState(
            command_type=str(row["command_type"]),
            command_id=str(row["command_id"]),
            sequence=int(row["sequence"]),
            payload_hash=str(row["payload_hash"]),
            payload=json.loads(str(row["payload_json"])),
            reported_state=json.loads(str(row["reported_state_json"])),
            applied_at=str(row["applied_at"]),
            updated_at=str(row["updated_at"]),
        )

    def _append_history(
        self,
        connection: sqlite3.Connection,
        command_id: str,
        *,
        from_phase: str | None,
        to_phase: str,
        code: str | None = None,
        message: str | None = None,
        state: Mapping[str, Any] | None = None,
        now: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO fleet_reconcile_history (
                command_id, from_phase, to_phase, code, message, state_json,
                observed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                command_id,
                from_phase,
                to_phase,
                str(code)[:100] if code else None,
                str(message)[:1000] if message else None,
                self._optional_json(state),
                now,
            ),
        )

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> ReconcileJob:
        return ReconcileJob(
            command_id=str(row["command_id"]),
            command_type=str(row["command_type"]),
            sequence=int(row["sequence"]),
            phase=str(row["phase"]),
            attempts=int(row["attempts"]),
            lease_owner=str(row["lease_owner"]) if row["lease_owner"] else None,
            lease_expires_at=(
                float(row["lease_expires_at"])
                if row["lease_expires_at"] is not None
                else None
            ),
            checkpoint=(
                json.loads(str(row["checkpoint_json"]))
                if row["checkpoint_json"]
                else None
            ),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )
