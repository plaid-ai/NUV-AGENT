from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

import aiohttp

from nuvion_app.inference.command_inbox import (
    COMMAND_STATUSES,
    CommandAck,
    DurableCommandInbox,
)
from nuvion_app.inference.fleet_command import VerifiedFleetCommand

COMMAND_WAKE_DESTINATION = "/user/queue/fleet.command"
COMMAND_ACK_DESTINATION = "/app/device/command.ack"
DEFAULT_COMMAND_PULL_LIMIT = 100
MAX_COMMAND_PULL_LIMIT = 100


class FleetCommandTransportError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool,
        status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.status = status


@dataclass(frozen=True)
class CommandWakeup:
    command_id: str
    sequence: int


@dataclass(frozen=True)
class PulledCommand:
    command_id: str
    sequence: int
    compact_jws: str

    def verify(self, verifier: PulledCommandVerifier) -> VerifiedFleetCommand:
        command = verifier.verify(self.compact_jws)
        self.validate_envelope(command)
        return command

    def validate_envelope(self, command: VerifiedFleetCommand) -> None:
        if command.command_id != self.command_id or command.sequence != self.sequence:
            raise FleetCommandTransportError(
                "JOURNAL_ENVELOPE_MISMATCH",
                "journal metadata does not match signed command claims",
                retryable=False,
            )


@dataclass(frozen=True)
class PulledCommandPage:
    commands: tuple[PulledCommand, ...]
    next_after_sequence: int
    has_more: bool


class PulledCommandVerifier(Protocol):
    def verify(self, compact_jws: str) -> VerifiedFleetCommand: ...


def _canonical_uuid(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise FleetCommandTransportError(
            "INVALID_RESPONSE",
            f"{field} must be a UUID string",
            retryable=False,
        )
    try:
        normalized = str(uuid.UUID(value))
    except ValueError as exc:
        raise FleetCommandTransportError(
            "INVALID_RESPONSE",
            f"{field} must be a UUID string",
            retryable=False,
        ) from exc
    if normalized != value.lower():
        raise FleetCommandTransportError(
            "INVALID_RESPONSE",
            f"{field} must use canonical UUID form",
            retryable=False,
        )
    return normalized


def _positive_sequence(value: Any, field: str = "sequence") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise FleetCommandTransportError(
            "INVALID_RESPONSE",
            f"{field} must be a positive integer",
            retryable=False,
        )
    return value


def parse_command_wakeup(body: str) -> CommandWakeup:
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, TypeError) as exc:
        raise FleetCommandTransportError(
            "INVALID_WAKEUP",
            "fleet command wake-up is not valid JSON",
            retryable=False,
        ) from exc
    if not isinstance(payload, dict):
        raise FleetCommandTransportError(
            "INVALID_WAKEUP",
            "fleet command wake-up must be an object",
            retryable=False,
        )
    try:
        return CommandWakeup(
            command_id=_canonical_uuid(payload.get("commandId"), "commandId"),
            sequence=_positive_sequence(payload.get("sequence")),
        )
    except FleetCommandTransportError as exc:
        raise FleetCommandTransportError(
            "INVALID_WAKEUP",
            str(exc),
            retryable=False,
        ) from exc


def build_command_ack_payload(ack: CommandAck) -> dict[str, Any]:
    if ack.status not in COMMAND_STATUSES:
        raise ValueError(f"unsupported command ACK status: {ack.status}")
    observed_at = str(ack.observed_at or "").strip()
    try:
        parsed_observed_at = datetime.fromisoformat(
            observed_at[:-1] + "+00:00" if observed_at.endswith("Z") else observed_at
        )
    except ValueError as exc:
        raise ValueError("observedAt must be RFC3339 date-time") from exc
    if parsed_observed_at.tzinfo is None or parsed_observed_at.utcoffset() is None:
        raise ValueError("observedAt must include a timezone")
    if ack.code is not None and len(ack.code) > 100:
        raise ValueError("ACK code exceeds 100 characters")
    if ack.message is not None and len(ack.message) > 1000:
        raise ValueError("ACK message exceeds 1000 characters")
    if ack.reported_state is not None and not isinstance(ack.reported_state, dict):
        raise ValueError("reportedState must be an object or null")
    return {
        "ackId": _canonical_uuid(ack.ack_id, "ackId"),
        "commandId": _canonical_uuid(ack.command_id, "commandId"),
        "sequence": _positive_sequence(ack.sequence),
        "status": ack.status,
        "observedAt": observed_at,
        "code": ack.code,
        "message": ack.message,
        "reportedState": ack.reported_state,
    }


def build_lifecycle_ack_payloads(
    inbox: DurableCommandInbox,
    command_id: str,
) -> list[dict[str, Any]]:
    """Return every persisted deterministic ACK in lifecycle order for reconnect replay."""

    return [build_command_ack_payload(ack) for ack in inbox.ack_transitions(command_id)]


class FleetCommandHttpClient:
    def __init__(
        self,
        *,
        base_url: str,
        access_token_provider: Callable[[], str],
        session: Any | None = None,
        timeout_seconds: float = 15.0,
    ) -> None:
        normalized_base = str(base_url or "").strip().rstrip("/")
        if not normalized_base.startswith(("https://", "http://")):
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.base_url = normalized_base
        self.access_token_provider = access_token_provider
        self.session = session
        self.timeout_seconds = timeout_seconds

    async def pull_after(
        self,
        after_sequence: int,
        limit: int = DEFAULT_COMMAND_PULL_LIMIT,
    ) -> PulledCommandPage:
        if (
            isinstance(after_sequence, bool)
            or not isinstance(after_sequence, int)
            or after_sequence < 0
        ):
            raise ValueError("after_sequence must be a non-negative integer")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= MAX_COMMAND_PULL_LIMIT
        ):
            raise ValueError(f"limit must be between 1 and {MAX_COMMAND_PULL_LIMIT}")
        token = str(self.access_token_provider() or "").strip()
        if not token:
            raise FleetCommandTransportError(
                "AUTH_TOKEN_MISSING",
                "device access token is unavailable",
                retryable=True,
            )

        try:
            if self.session is not None:
                return await self._pull(self.session, token, after_sequence, limit)

            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                return await self._pull(session, token, after_sequence, limit)
        except FleetCommandTransportError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            raise FleetCommandTransportError(
                "COMMAND_PULL_UNAVAILABLE",
                f"command journal transport unavailable: {type(exc).__name__}",
                retryable=True,
            ) from exc

    async def _pull(
        self,
        session: Any,
        token: str,
        after_sequence: int,
        limit: int,
    ) -> PulledCommandPage:
        url = f"{self.base_url}/devices/me/commands"
        async with session.get(
            url,
            params={"afterSequence": after_sequence, "limit": limit},
            headers={"Authorization": f"Bearer {token}"},
        ) as response:
            if response.status < 200 or response.status >= 300:
                detail = (await response.text())[:500]
                retryable = response.status in {408, 425, 429} or response.status >= 500
                raise FleetCommandTransportError(
                    "COMMAND_PULL_FAILED",
                    f"command journal pull failed status={response.status} detail={detail}",
                    retryable=retryable,
                    status=response.status,
                )
            try:
                payload = await response.json(content_type=None)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise FleetCommandTransportError(
                    "INVALID_RESPONSE",
                    "command journal response is not JSON",
                    retryable=False,
                    status=response.status,
                ) from exc

        items: Any = (
            payload.get("data")
            if isinstance(payload, dict) and "data" in payload
            else payload
        )
        if isinstance(items, dict):
            items = items.get("commands", items.get("content"))
        if not isinstance(items, list):
            raise FleetCommandTransportError(
                "INVALID_RESPONSE",
                "command journal response data must be an array",
                retryable=False,
                status=200,
            )
        if len(items) > limit:
            raise FleetCommandTransportError(
                "INVALID_RESPONSE",
                "command journal response exceeds the requested page limit",
                retryable=False,
                status=200,
            )

        commands: list[PulledCommand] = []
        previous_sequence = after_sequence
        for item in items:
            if not isinstance(item, dict):
                raise FleetCommandTransportError(
                    "INVALID_RESPONSE",
                    "command journal entry must be an object",
                    retryable=False,
                    status=200,
                )
            command_id = _canonical_uuid(item.get("commandId"), "commandId")
            sequence = _positive_sequence(item.get("sequence"))
            compact_jws = item.get("compactJws")
            if not isinstance(compact_jws, str) or compact_jws.count(".") != 2:
                raise FleetCommandTransportError(
                    "INVALID_RESPONSE",
                    "command journal entry compactJws is invalid",
                    retryable=False,
                    status=200,
                )
            if sequence <= previous_sequence:
                raise FleetCommandTransportError(
                    "NON_MONOTONIC_RESPONSE",
                    "command journal response is not strictly sequence ordered",
                    retryable=False,
                    status=200,
                )
            previous_sequence = sequence
            commands.append(
                PulledCommand(
                    command_id=command_id,
                    sequence=sequence,
                    compact_jws=compact_jws,
                )
            )
        next_after_sequence = commands[-1].sequence if commands else after_sequence
        if commands and next_after_sequence <= after_sequence:
            raise FleetCommandTransportError(
                "NON_PROGRESSING_RESPONSE",
                "command journal page did not advance its sequence cursor",
                retryable=False,
                status=200,
            )
        return PulledCommandPage(
            commands=tuple(commands),
            next_after_sequence=next_after_sequence,
            has_more=len(commands) == limit,
        )
