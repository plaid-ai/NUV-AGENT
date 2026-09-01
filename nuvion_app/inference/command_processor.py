from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from nuvion_app.inference.command_inbox import (
    COMMAND_STATUS_FAILED,
    COMMAND_STATUS_IN_PROGRESS,
    COMMAND_STATUS_RECEIVED,
    COMMAND_STATUS_REJECTED,
    CommandAck,
    CommandEffectOutcome,
    DurableCommandInbox,
)
from nuvion_app.inference.fleet_command import VerifiedFleetCommand

CommandEffectResult = CommandEffectOutcome
CommandHandler = Callable[
    [VerifiedFleetCommand, sqlite3.Connection], CommandEffectResult
]


class CommandVerifier(Protocol):
    def verify(self, compact_jws: str) -> VerifiedFleetCommand: ...


@dataclass(frozen=True)
class ProcessResult:
    command: VerifiedFleetCommand
    ack: CommandAck
    lifecycle_acks: tuple[CommandAck, ...]
    duplicate: bool
    effect_applied: bool


class DurableCommandProcessor:
    """Verify, journal and execute a durable command.

    Terminal duplicates never invoke their handler and instead return the persisted
    deterministic lifecycle ACK history for ordered reconnect replay. A handler gets
    the inbox SQLite connection, so a SQLite-backed effect can commit atomically with
    the terminal ACK. External effects are fundamentally outside that transaction and
    therefore must use an idempotent/convergent handler contract (for example desired
    state reconciliation or an A/B updater checkpoint).
    """

    def __init__(
        self,
        *,
        inbox: DurableCommandInbox,
        verifier: CommandVerifier,
        handlers: Mapping[str, CommandHandler],
    ) -> None:
        self.inbox = inbox
        self.verifier = verifier
        self.handlers = dict(handlers)

    def process(self, compact_jws: str) -> ProcessResult:
        command = self.verifier.verify(compact_jws)
        return self.process_verified(command)

    def process_verified(self, command: VerifiedFleetCommand) -> ProcessResult:
        """Journal a command already authenticated by this processor's verifier."""

        accepted = self.inbox.accept(command)
        return self._execute_verified(command, duplicate=accepted.duplicate)

    def resume_pending(self, limit: int = 200) -> list[ProcessResult]:
        """Resume already-verified RECEIVED/IN_PROGRESS rows after process restart.

        Expiry controls initial acceptance. Once a signed command is durably accepted,
        its inbox row is the trusted source for crash recovery even if ``expiresAt``
        passes while the device is rebooting.
        """

        results: list[ProcessResult] = []
        for record in self.inbox.pending(limit=limit):
            results.append(
                self._execute_verified(
                    self.inbox.rehydrate(record),
                    duplicate=True,
                )
            )
        return results

    def resume_pending_page(
        self,
        *,
        after_sequence: int,
        through_sequence: int,
        limit: int,
    ) -> list[ProcessResult]:
        """Resume one keyset page from a fixed pending snapshot."""

        return [
            self._execute_verified(
                self.inbox.rehydrate(record),
                duplicate=True,
            )
            for record in self.inbox.pending_page(
                after_sequence=after_sequence,
                through_sequence=through_sequence,
                limit=limit,
            )
        ]

    def reject_verified(
        self,
        command: VerifiedFleetCommand,
        *,
        code: str,
        message: str,
    ) -> ProcessResult:
        """Advance a fully authenticated but non-executable command without an effect."""

        accepted = self.inbox.accept_rejected(
            command,
            code=code,
            message=message,
        )
        lifecycle = tuple(self.inbox.ack_transitions(command.command_id))
        return ProcessResult(
            command=command,
            ack=accepted.ack,
            lifecycle_acks=lifecycle,
            duplicate=accepted.duplicate,
            effect_applied=False,
        )

    def _execute_verified(
        self, command: VerifiedFleetCommand, *, duplicate: bool
    ) -> ProcessResult:
        record = self.inbox.get(command.command_id)
        if record is None:
            raise RuntimeError(
                f"accepted command disappeared from inbox: {command.command_id}"
            )

        if record.terminal:
            lifecycle = tuple(self.inbox.ack_transitions(command.command_id))
            return ProcessResult(
                command=command,
                ack=lifecycle[-1],
                lifecycle_acks=lifecycle,
                duplicate=True,
                effect_applied=False,
            )

        if record.status == COMMAND_STATUS_RECEIVED:
            self.inbox.transition(command.command_id, COMMAND_STATUS_IN_PROGRESS)
        elif record.status != COMMAND_STATUS_IN_PROGRESS:
            raise RuntimeError(
                f"unexpected non-terminal command status: {record.status}"
            )

        handler = self.handlers.get(command.command_type)
        if handler is None:
            ack = self.inbox.transition(
                command.command_id,
                COMMAND_STATUS_REJECTED,
                code="HANDLER_NOT_REGISTERED",
                message=f"No handler registered for {command.command_type}",
            )
            return ProcessResult(
                command=command,
                ack=ack,
                lifecycle_acks=tuple(self.inbox.ack_transitions(command.command_id)),
                duplicate=duplicate,
                effect_applied=False,
            )

        try:
            ack, effect_applied = self.inbox.run_transactional_effect(
                command.command_id,
                lambda connection: handler(command, connection),
            )
        except Exception as exc:  # noqa: BLE001 - handler failures become durable FAILED ACKs.
            message = f"{type(exc).__name__}: {exc}"[:1000]
            ack = self.inbox.transition(
                command.command_id,
                COMMAND_STATUS_FAILED,
                code="HANDLER_ERROR",
                message=message,
            )
            effect_applied = False

        return ProcessResult(
            command=command,
            ack=ack,
            lifecycle_acks=tuple(self.inbox.ack_transitions(command.command_id)),
            duplicate=duplicate,
            effect_applied=effect_applied,
        )
