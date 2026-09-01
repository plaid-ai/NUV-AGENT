from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
import tempfile
import unittest
import uuid
from dataclasses import replace
from pathlib import Path
from unittest import mock

from nuvion_app.inference.command_inbox import (
    COMMAND_STATUS_FAILED,
    COMMAND_STATUS_IN_PROGRESS,
    COMMAND_STATUS_SUCCEEDED,
    CommandEffectOutcome,
    CommandInboxError,
    DurableCommandInbox,
    deterministic_ack_id,
)
from nuvion_app.inference.command_processor import DurableCommandProcessor
from nuvion_app.inference.command_transport import build_lifecycle_ack_payloads
from nuvion_app.inference.fleet_command import VerifiedFleetCommand


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _command(
    *, sequence: int = 1, command_id: str | None = None
) -> VerifiedFleetCommand:
    payload = {"configVersion": sequence}
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    normalized_id = command_id or str(uuid.uuid4())
    return VerifiedFleetCommand(
        command_id=normalized_id,
        device_id="sp-3-nuvion-test",
        space_id=3,
        command_type="CONFIG_APPLY",
        schema_version=1,
        issued_at="2026-09-01T02:00:00Z",
        expires_at="2026-09-01T02:10:00Z",
        sequence=sequence,
        payload_base64=_b64url(payload_bytes),
        payload_hash=hashlib.sha256(payload_bytes).hexdigest(),
        payload=payload,
        actor="operator@example.com",
        authorization_context="SPACE_ADMIN",
        key_id="test-only-key",
        required_capability="command.config.apply",
        compact_jws=f"header.claims.signature-{normalized_id}-{sequence}",
    )


class StubVerifier:
    def __init__(self, command: VerifiedFleetCommand) -> None:
        self.command = command
        self.calls = 0

    def verify(self, compact_jws: str) -> VerifiedFleetCommand:
        self.calls += 1
        if compact_jws != self.command.compact_jws:
            raise AssertionError("unexpected compact JWS")
        return self.command


class MustNotVerifyAgain:
    def verify(self, compact_jws: str) -> VerifiedFleetCommand:
        raise AssertionError(
            "durably accepted command must not be reverified on restart"
        )


class SimulatedProcessCrash(BaseException):
    pass


class DurableCommandInboxTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.path = Path(self.temporary_directory.name) / "commands.sqlite3"
        self.inbox = DurableCommandInbox(
            self.path,
            clock=lambda: "2026-09-01T02:00:03Z",
        )

    def test_accept_is_idempotent_and_rejects_collision_and_sequence_replay(
        self,
    ) -> None:
        command = _command(sequence=7)

        accepted = self.inbox.accept(command)
        duplicate = self.inbox.accept(command)

        self.assertFalse(accepted.duplicate)
        self.assertTrue(duplicate.duplicate)
        self.assertEqual(accepted.ack.status, "RECEIVED")
        self.assertEqual(duplicate.ack.ack_id, accepted.ack.ack_id)
        self.assertEqual(
            accepted.ack.ack_id,
            deterministic_ack_id(command.command_id, "RECEIVED"),
        )
        self.assertEqual(self.inbox.last_sequence(), 7)

        with self.assertRaises(CommandInboxError) as collision:
            self.inbox.accept(
                replace(command, compact_jws=command.compact_jws + "-changed")
            )
        self.assertEqual(collision.exception.code, "COMMAND_ID_COLLISION")

        with self.assertRaises(CommandInboxError) as replay:
            self.inbox.accept(_command(sequence=7))
        self.assertEqual(replay.exception.code, "SEQUENCE_REPLAY")

    def test_inbox_identity_scope_cannot_silently_change_on_reprovision(self) -> None:
        self.inbox.bind_identity(
            device_id="sp-3-nuvion-test",
            space_id=3,
            trust_domain="macos-dev",
        )
        self.inbox.bind_identity(
            device_id="sp-3-nuvion-test",
            space_id=3,
            trust_domain="macos-dev",
        )

        with self.assertRaises(CommandInboxError) as mismatch:
            self.inbox.bind_identity(
                device_id="sp-4-nuvion-reassigned",
                space_id=4,
                trust_domain="production",
            )

        self.assertEqual(mismatch.exception.code, "IDENTITY_SCOPE_MISMATCH")

    def test_iq9075_dev_identity_scope_is_supported_and_stable(self) -> None:
        self.inbox.bind_identity(
            device_id="sp-3-iq9075-dev",
            space_id=3,
            trust_domain="iq9075-dev",
        )

        restarted = DurableCommandInbox(self.path)
        restarted.bind_identity(
            device_id="sp-3-iq9075-dev",
            space_id=3,
            trust_domain="iq9075-dev",
        )

        self.assertEqual(restarted.last_sequence(), 0)

    def test_legacy_rows_without_identity_scope_require_explicit_migration(
        self,
    ) -> None:
        self.inbox.accept(_command(sequence=1))

        with self.assertRaises(CommandInboxError) as legacy:
            self.inbox.bind_identity(
                device_id="sp-3-nuvion-test",
                space_id=3,
                trust_domain="macos-dev",
            )

        self.assertEqual(legacy.exception.code, "IDENTITY_SCOPE_UNBOUND_LEGACY")

    def test_transactional_effect_runs_once_and_replays_all_lifecycle_acks(
        self,
    ) -> None:
        command = _command(sequence=1)
        verifier = StubVerifier(command)
        handler_calls: list[str] = []

        def handler(
            verified: VerifiedFleetCommand, connection: sqlite3.Connection
        ) -> CommandEffectOutcome:
            handler_calls.append(verified.command_id)
            connection.execute(
                "CREATE TABLE IF NOT EXISTS applied_effects (command_id TEXT PRIMARY KEY, value INTEGER NOT NULL)"
            )
            connection.execute(
                "INSERT INTO applied_effects(command_id, value) VALUES (?, ?)",
                (verified.command_id, verified.payload["configVersion"]),
            )
            return CommandEffectOutcome.succeeded(
                {"configVersion": verified.payload["configVersion"]}
            )

        processor = DurableCommandProcessor(
            inbox=self.inbox,
            verifier=verifier,
            handlers={"CONFIG_APPLY": handler},
        )

        first = processor.process(command.compact_jws)
        duplicate = processor.process(command.compact_jws)

        self.assertTrue(first.effect_applied)
        self.assertFalse(first.duplicate)
        self.assertFalse(duplicate.effect_applied)
        self.assertTrue(duplicate.duplicate)
        self.assertEqual(handler_calls, [command.command_id])
        self.assertEqual(first.ack.status, COMMAND_STATUS_SUCCEEDED)
        self.assertEqual(duplicate.ack.ack_id, first.ack.ack_id)
        self.assertEqual(
            [ack.status for ack in first.lifecycle_acks],
            ["RECEIVED", "IN_PROGRESS", "SUCCEEDED"],
        )

        with sqlite3.connect(self.path) as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM applied_effects"
            ).fetchone()[0]
        self.assertEqual(count, 1)

        reopened = DurableCommandInbox(self.path)
        replay_payloads = build_lifecycle_ack_payloads(reopened, command.command_id)
        self.assertEqual(
            [payload["status"] for payload in replay_payloads],
            ["RECEIVED", "IN_PROGRESS", "SUCCEEDED"],
        )
        self.assertEqual(
            [payload["ackId"] for payload in replay_payloads],
            [
                deterministic_ack_id(command.command_id, status)
                for status in ("RECEIVED", "IN_PROGRESS", "SUCCEEDED")
            ],
        )
        self.assertEqual(replay_payloads[-1]["reportedState"], {"configVersion": 1})

    def test_deferred_effect_commits_checkpoint_without_claiming_success(self) -> None:
        command = _command(sequence=8)

        def handler(
            verified: VerifiedFleetCommand,
            connection: sqlite3.Connection,
        ) -> CommandEffectOutcome:
            connection.execute(
                "CREATE TABLE deferred_effect (command_id TEXT PRIMARY KEY)"
            )
            connection.execute(
                "INSERT INTO deferred_effect(command_id) VALUES (?)",
                (verified.command_id,),
            )
            return CommandEffectOutcome.deferred()

        result = DurableCommandProcessor(
            inbox=self.inbox,
            verifier=StubVerifier(command),
            handlers={"CONFIG_APPLY": handler},
        ).process(command.compact_jws)

        self.assertEqual(result.ack.status, COMMAND_STATUS_IN_PROGRESS)
        self.assertEqual(
            [ack.status for ack in result.lifecycle_acks],
            ["RECEIVED", "IN_PROGRESS"],
        )
        self.assertEqual(
            self.inbox.get(command.command_id).status, COMMAND_STATUS_IN_PROGRESS
        )
        with sqlite3.connect(self.path) as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM deferred_effect"
            ).fetchone()[0]
        self.assertEqual(count, 1)

    def test_authenticated_rejection_advances_sequence_without_running_effect(
        self,
    ) -> None:
        command = _command(sequence=9)
        processor = DurableCommandProcessor(
            inbox=self.inbox,
            verifier=MustNotVerifyAgain(),
            handlers={},
        )

        result = processor.reject_verified(
            command,
            code="EXPIRED",
            message="expired before acceptance",
        )

        self.assertFalse(result.effect_applied)
        self.assertEqual(self.inbox.last_sequence(), 9)
        self.assertEqual(
            [ack.status for ack in result.lifecycle_acks],
            ["RECEIVED", "IN_PROGRESS", "REJECTED"],
        )
        self.assertEqual(result.ack.code, "EXPIRED")

    def test_authenticated_rejection_is_atomic_across_simulated_process_crash(
        self,
    ) -> None:
        command = _command(sequence=10)
        processor = DurableCommandProcessor(
            inbox=self.inbox,
            verifier=MustNotVerifyAgain(),
            handlers={},
        )
        original_insert_ack = self.inbox._insert_ack

        def crash_before_terminal_ack(connection, **kwargs):
            if kwargs["status"] == "REJECTED":
                raise SimulatedProcessCrash()
            return original_insert_ack(connection, **kwargs)

        with (
            mock.patch.object(
                self.inbox,
                "_insert_ack",
                side_effect=crash_before_terminal_ack,
            ),
            self.assertRaises(SimulatedProcessCrash),
        ):
            processor.reject_verified(
                command,
                code="EXPIRED",
                message="expired before acceptance",
            )

        self.assertIsNone(self.inbox.get(command.command_id))
        self.assertEqual(self.inbox.last_sequence(), 0)
        self.assertEqual(self.inbox.ack_transitions(command.command_id), [])
        self.assertEqual(processor.resume_pending(), [])

        retried = processor.reject_verified(
            command,
            code="EXPIRED",
            message="expired before acceptance",
        )
        self.assertEqual(retried.ack.status, "REJECTED")
        self.assertEqual(self.inbox.pending(), [])

    def test_transaction_rolls_back_effect_on_crash_and_resume_applies_once(
        self,
    ) -> None:
        command = _command(sequence=2)

        def crashing_handler(
            verified: VerifiedFleetCommand,
            connection: sqlite3.Connection,
        ) -> CommandEffectOutcome:
            connection.execute(
                "CREATE TABLE crash_effect (command_id TEXT PRIMARY KEY)"
            )
            connection.execute(
                "INSERT INTO crash_effect(command_id) VALUES (?)",
                (verified.command_id,),
            )
            raise SimulatedProcessCrash()

        crashing_processor = DurableCommandProcessor(
            inbox=self.inbox,
            verifier=StubVerifier(command),
            handlers={"CONFIG_APPLY": crashing_handler},
        )

        with self.assertRaises(SimulatedProcessCrash):
            crashing_processor.process(command.compact_jws)
        self.assertEqual(
            self.inbox.get(command.command_id).status, COMMAND_STATUS_IN_PROGRESS
        )
        with sqlite3.connect(self.path) as connection:
            table = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='crash_effect'"
            ).fetchone()
        self.assertIsNone(table)

        resume_calls: list[str] = []

        def resume_handler(
            verified: VerifiedFleetCommand,
            connection: sqlite3.Connection,
        ) -> CommandEffectOutcome:
            resume_calls.append(verified.command_id)
            connection.execute(
                "CREATE TABLE crash_effect (command_id TEXT PRIMARY KEY)"
            )
            connection.execute(
                "INSERT INTO crash_effect(command_id) VALUES (?)",
                (verified.command_id,),
            )
            return CommandEffectOutcome.succeeded({"resumed": True})

        resumed = DurableCommandProcessor(
            inbox=DurableCommandInbox(self.path),
            verifier=MustNotVerifyAgain(),
            handlers={"CONFIG_APPLY": resume_handler},
        ).resume_pending()

        self.assertEqual(len(resumed), 1)
        self.assertEqual(resumed[0].ack.status, COMMAND_STATUS_SUCCEEDED)
        self.assertTrue(resumed[0].effect_applied)
        self.assertEqual(resume_calls, [command.command_id])
        with sqlite3.connect(self.path) as connection:
            count = connection.execute("SELECT COUNT(*) FROM crash_effect").fetchone()[
                0
            ]
        self.assertEqual(count, 1)

    def test_state_machine_requires_in_progress_before_terminal_and_terminal_is_final(
        self,
    ) -> None:
        command = _command(sequence=3)
        self.inbox.accept(command)

        with self.assertRaises(CommandInboxError) as skipped:
            self.inbox.transition(command.command_id, COMMAND_STATUS_SUCCEEDED)
        self.assertEqual(skipped.exception.code, "INVALID_TRANSITION")

        in_progress = self.inbox.transition(
            command.command_id, COMMAND_STATUS_IN_PROGRESS
        )
        terminal = self.inbox.transition(command.command_id, COMMAND_STATUS_SUCCEEDED)
        self.assertEqual(in_progress.status, COMMAND_STATUS_IN_PROGRESS)
        self.assertEqual(terminal.status, COMMAND_STATUS_SUCCEEDED)

        with self.assertRaises(CommandInboxError) as after_terminal:
            self.inbox.transition(command.command_id, COMMAND_STATUS_FAILED)
        self.assertEqual(after_terminal.exception.code, "TERMINAL_STATE")

    def test_oversized_reported_state_is_rejected_before_terminal_commit(self) -> None:
        command = _command(sequence=11)
        self.inbox.accept(command)
        self.inbox.transition(command.command_id, COMMAND_STATUS_IN_PROGRESS)

        with self.assertRaisesRegex(ValueError, "reportedState exceeds"):
            self.inbox.transition(
                command.command_id,
                COMMAND_STATUS_SUCCEEDED,
                reported_state={"blob": "한" * 65_536},
            )

        record = self.inbox.get(command.command_id)
        self.assertEqual(record.status, COMMAND_STATUS_IN_PROGRESS)
        self.assertEqual(
            [ack.status for ack in self.inbox.ack_transitions(command.command_id)],
            ["RECEIVED", "IN_PROGRESS"],
        )

    def test_restart_resumes_already_accepted_command_without_expiry_reverification(
        self,
    ) -> None:
        expired_now_but_previously_verified = _command(sequence=4)
        self.inbox.accept(expired_now_but_previously_verified)

        calls: list[str] = []

        def handler(
            command: VerifiedFleetCommand,
            connection: sqlite3.Connection,
        ) -> CommandEffectOutcome:
            calls.append(command.command_id)
            return CommandEffectOutcome.succeeded({"configVersion": 4})

        processor = DurableCommandProcessor(
            inbox=DurableCommandInbox(self.path),
            verifier=MustNotVerifyAgain(),
            handlers={"CONFIG_APPLY": handler},
        )

        results = processor.resume_pending()

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].ack.status, COMMAND_STATUS_SUCCEEDED)
        self.assertEqual(calls, [expired_now_but_previously_verified.command_id])
        self.assertEqual(
            [ack.status for ack in results[0].lifecycle_acks],
            ["RECEIVED", "IN_PROGRESS", "SUCCEEDED"],
        )


if __name__ == "__main__":
    unittest.main()
