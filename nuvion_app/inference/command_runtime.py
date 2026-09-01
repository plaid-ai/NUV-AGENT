from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import stat
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from nuvion_app.inference.command_inbox import (
    CommandEffectOutcome,
    CommandInboxError,
    DurableCommandInbox,
    resolve_default_command_inbox_path,
)
from nuvion_app.inference.command_processor import (
    DurableCommandProcessor,
    ProcessResult,
)
from nuvion_app.inference.command_transport import (
    COMMAND_ACK_DESTINATION,
    DEFAULT_COMMAND_PULL_LIMIT,
    FleetCommandHttpClient,
    PulledCommand,
    PulledCommandPage,
    build_command_ack_payload,
    build_lifecycle_ack_payloads,
    parse_command_wakeup,
)
from nuvion_app.inference.fleet_command import (
    AUTHENTICATED_REJECTION_CODES,
    CommandValidationError,
    Ed25519Keyring,
    FleetCommandEvaluation,
    FleetCommandVerifier,
    VerifiedFleetCommand,
)
from nuvion_app.runtime.platform_identity import (
    IDENTITY_STATUS_DEV,
    IDENTITY_STATUS_VERIFIED,
    PROFILE_IQ9075_DEV,
    PROFILE_MACOS_DEV,
    PlatformIdentity,
    resolve_platform_identity,
)

KEYRING_SCHEMA_VERSION = 1
MAX_KEYRING_BYTES = 64 * 1024
DEFAULT_ACK_REPLAY_LIMIT = 10_000
DEFAULT_MAX_PULL_BATCHES = 100

AckSender = Callable[[str, dict[str, Any]], bool]


class FleetCommandRuntimeError(RuntimeError):
    pass


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON member: {key}")
        result[key] = value
    return result


def _decode_canonical_base64(value: Any, *, kid: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise FleetCommandRuntimeError(f"public key for kid={kid} must be base64 text")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise FleetCommandRuntimeError(
            f"public key for kid={kid} is not valid base64"
        ) from exc
    if base64.b64encode(decoded).decode("ascii") != value:
        raise FleetCommandRuntimeError(
            f"public key for kid={kid} must use canonical padded base64"
        )
    return decoded


def _read_integrity_file(path: Path, *, require_root_owner: bool) -> bytes:
    try:
        if stat.S_ISLNK(path.lstat().st_mode):
            raise FleetCommandRuntimeError(
                "Fleet command keyring must not be a symlink"
            )
    except OSError as exc:
        raise FleetCommandRuntimeError(
            f"cannot inspect Fleet command keyring: {exc}"
        ) from exc
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FleetCommandRuntimeError(
            f"cannot open Fleet command keyring: {exc}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise FleetCommandRuntimeError(
                "Fleet command keyring must be a regular file"
            )
        if require_root_owner and metadata.st_uid != 0:
            raise FleetCommandRuntimeError(
                "Fleet command keyring for this trust domain must be owned by root"
            )
        if metadata.st_mode & 0o022:
            raise FleetCommandRuntimeError(
                "Fleet command keyring must not be group/other writable"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(8192, MAX_KEYRING_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_KEYRING_BYTES:
                raise FleetCommandRuntimeError("Fleet command keyring exceeds 64 KiB")
        final_metadata = os.fstat(descriptor)
        if (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
        ) != (
            final_metadata.st_dev,
            final_metadata.st_ino,
            final_metadata.st_size,
            final_metadata.st_mtime_ns,
        ):
            raise FleetCommandRuntimeError(
                "Fleet command keyring changed while being read"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def load_fleet_command_keyring(
    path: str | Path,
    *,
    expected_trust_domain: str,
    require_root_owner: bool,
) -> Ed25519Keyring:
    raw = _read_integrity_file(
        Path(path).expanduser(), require_root_owner=require_root_owner
    )
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise FleetCommandRuntimeError(
            "Fleet command keyring is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schemaVersion",
        "trustDomain",
        "keys",
    }:
        raise FleetCommandRuntimeError(
            "Fleet command keyring must contain exactly schemaVersion, trustDomain and keys"
        )
    if payload["schemaVersion"] != KEYRING_SCHEMA_VERSION:
        raise FleetCommandRuntimeError(
            "unsupported Fleet command keyring schemaVersion"
        )
    if payload["trustDomain"] != expected_trust_domain:
        raise FleetCommandRuntimeError(
            "Fleet command keyring trustDomain does not match this platform"
        )
    keys = payload["keys"]
    if not isinstance(keys, dict) or not keys:
        raise FleetCommandRuntimeError(
            "Fleet command keyring keys must be a non-empty object"
        )
    if len(keys) > 32:
        raise FleetCommandRuntimeError(
            "Fleet command keyring may contain at most 32 keys"
        )
    decoded = {
        str(kid): _decode_canonical_base64(value, kid=str(kid))
        for kid, value in keys.items()
    }
    try:
        return Ed25519Keyring(decoded)
    except (TypeError, ValueError) as exc:
        raise FleetCommandRuntimeError(str(exc)) from exc


def desired_state_handler(
    command: VerifiedFleetCommand,
    connection: Any,
) -> CommandEffectOutcome:
    """Atomically checkpoint verified desired state with the terminal ACK.

    NUV-436/437/439 reconcilers consume this table and own external effects. This
    common transport reports durable acceptance, not camera/package activation.
    """

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS fleet_desired_state (
            command_type TEXT PRIMARY KEY,
            command_id TEXT NOT NULL UNIQUE,
            sequence INTEGER NOT NULL UNIQUE,
            payload_hash TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            accepted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    payload_json = json.dumps(
        command.payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    connection.execute(
        """
        INSERT INTO fleet_desired_state (
            command_type, command_id, sequence, payload_hash, payload_json
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(command_type) DO UPDATE SET
            command_id = excluded.command_id,
            sequence = excluded.sequence,
            payload_hash = excluded.payload_hash,
            payload_json = excluded.payload_json,
            accepted_at = CURRENT_TIMESTAMP
        WHERE excluded.sequence > fleet_desired_state.sequence
        """,
        (
            command.command_type,
            command.command_id,
            command.sequence,
            command.payload_hash,
            payload_json,
        ),
    )
    return CommandEffectOutcome.deferred()


class FleetCommandRuntime:
    def __init__(
        self,
        *,
        inbox: DurableCommandInbox,
        processor: DurableCommandProcessor,
        http_client: FleetCommandHttpClient,
        ack_sender: AckSender,
        ack_replay_limit: int = DEFAULT_ACK_REPLAY_LIMIT,
        max_pull_batches: int = DEFAULT_MAX_PULL_BATCHES,
        pull_page_size: int = DEFAULT_COMMAND_PULL_LIMIT,
        max_resume_batches: int = DEFAULT_MAX_PULL_BATCHES,
        resume_page_size: int = DEFAULT_COMMAND_PULL_LIMIT,
    ) -> None:
        self.inbox = inbox
        self.processor = processor
        self.http_client = http_client
        self.ack_sender = ack_sender
        self.ack_replay_limit = max(1, min(int(ack_replay_limit), 10_000))
        self.max_pull_batches = max(1, min(int(max_pull_batches), 1_000))
        self.pull_page_size = max(
            1, min(int(pull_page_size), DEFAULT_COMMAND_PULL_LIMIT)
        )
        self.max_resume_batches = max(1, min(int(max_resume_batches), 1_000))
        self.resume_page_size = max(1, min(int(resume_page_size), 1_000))
        self._active_scan_cursor = 0
        self._pending_resume_initialized = False
        self._pending_resume_cursor = 0
        self._pending_resume_through: int | None = None
        self._sync_lock = asyncio.Lock()

    def _send_payloads(self, payloads: list[dict[str, Any]]) -> int:
        sent = 0
        for payload in payloads:
            if self.ack_sender(COMMAND_ACK_DESTINATION, payload):
                sent += 1
        return sent

    def _send_result(self, result: ProcessResult) -> int:
        return self._send_payloads(
            [build_command_ack_payload(ack) for ack in result.lifecycle_acks]
        )

    def replay_recent_acks(self) -> int:
        sent = 0
        for command_id in self.inbox.recent_command_ids(self.ack_replay_limit):
            sent += self._send_payloads(
                build_lifecycle_ack_payloads(self.inbox, command_id)
            )
        return sent

    async def on_connected(self) -> int:
        async with self._sync_lock:
            if not self._pending_resume_initialized:
                self._pending_resume_through = await asyncio.to_thread(
                    self.inbox.pending_high_watermark
                )
                self._pending_resume_initialized = True
            sent = await self._resume_pending_locked()
            return sent + await self._reconcile_active_locked()

    async def poll(self) -> int:
        """Repair lost wake-ups and lost ACKs from BE's non-terminal journal view."""

        async with self._sync_lock:
            sent = await self._resume_pending_locked()
            return sent + await self._reconcile_active_locked()

    async def on_wakeup(self, body: str) -> int:
        wakeup = parse_command_wakeup(body)
        async with self._sync_lock:
            existing = self.inbox.get(wakeup.command_id)
            if existing is not None:
                if existing.sequence != wakeup.sequence:
                    raise FleetCommandRuntimeError(
                        "wake-up metadata collides with the durable inbox"
                    )
                return self._send_payloads(
                    build_lifecycle_ack_payloads(self.inbox, wakeup.command_id)
                )
            return await self._reconcile_active_locked()

    async def _resume_pending_locked(self) -> int:
        through_sequence = self._pending_resume_through
        if through_sequence is None or through_sequence <= self._pending_resume_cursor:
            self._pending_resume_through = None
            return 0

        sent = 0
        for _ in range(self.max_resume_batches):
            results = await asyncio.to_thread(
                self.processor.resume_pending_page,
                after_sequence=self._pending_resume_cursor,
                through_sequence=through_sequence,
                limit=self.resume_page_size,
            )
            if not results:
                self._pending_resume_through = None
                return sent
            for result in results:
                sequence = result.command.sequence
                if sequence <= self._pending_resume_cursor:
                    raise FleetCommandRuntimeError(
                        "pending resume page did not advance its sequence cursor"
                    )
                self._pending_resume_cursor = sequence
                sent += self._send_result(result)
                await asyncio.sleep(0)
            if self._pending_resume_cursor >= through_sequence:
                self._pending_resume_through = None
                return sent
        return sent

    async def _reconcile_active_locked(self) -> int:
        sent = 0
        journal_cursor = self._active_scan_cursor
        for _ in range(self.max_pull_batches):
            page = await self.http_client.pull_after(
                journal_cursor,
                self.pull_page_size,
            )
            if not isinstance(page, PulledCommandPage):
                raise FleetCommandRuntimeError(
                    "command pull must return a bounded page"
                )
            if not page.commands:
                if page.has_more or page.next_after_sequence != journal_cursor:
                    raise FleetCommandRuntimeError(
                        "empty command page contains a non-progressing continuation"
                    )
                self._active_scan_cursor = 0
                return sent
            if (
                page.next_after_sequence <= journal_cursor
                or page.next_after_sequence != page.commands[-1].sequence
            ):
                raise FleetCommandRuntimeError(
                    "command page did not advance its sequence cursor"
                )
            for delivery in page.commands:
                existing = self.inbox.get(delivery.command_id)
                if existing is not None:
                    if existing.sequence != delivery.sequence:
                        raise FleetCommandRuntimeError(
                            "journal metadata collides with the durable inbox"
                        )
                    sent += self._send_payloads(
                        build_lifecycle_ack_payloads(
                            self.inbox,
                            delivery.command_id,
                        )
                    )
                else:
                    result = await asyncio.to_thread(
                        self._process_delivery,
                        delivery,
                    )
                    sent += self._send_result(result)
                await asyncio.sleep(0)
            journal_cursor = page.next_after_sequence
            self._active_scan_cursor = journal_cursor
            if not page.has_more:
                self._active_scan_cursor = 0
                return sent
        return sent

    def _process_delivery(self, delivery: PulledCommand) -> ProcessResult:
        evaluator = getattr(self.processor.verifier, "evaluate", None)
        if callable(evaluator):
            evaluation = evaluator(delivery.compact_jws)
            if not isinstance(evaluation, FleetCommandEvaluation):
                raise FleetCommandRuntimeError(
                    "command verifier returned an invalid evaluation"
                )
            delivery.validate_envelope(evaluation.command)
            if evaluation.executable:
                return self.processor.process_verified(evaluation.command)
            code = str(evaluation.rejection_code)
            if code not in AUTHENTICATED_REJECTION_CODES:
                raise FleetCommandRuntimeError(
                    f"command verifier exposed a non-terminal policy code: {code}"
                )
            return self.processor.reject_verified(
                evaluation.command,
                code=code,
                message=str(evaluation.rejection_message),
            )

        # Compatibility path for injected/test verifiers that implement the v1
        # ``verify`` protocol but not the phase-separated evaluation API.
        try:
            verified = delivery.verify(self.processor.verifier)
            return self.processor.process_verified(verified)
        except CommandValidationError as exc:
            if exc.code != "EXPIRED":
                raise
            rejection_verifier = getattr(
                self.processor.verifier,
                "verify_expired_for_rejection",
                None,
            )
            if not callable(rejection_verifier):
                raise
            expired = rejection_verifier(delivery.compact_jws)
            if (
                expired.command_id != delivery.command_id
                or expired.sequence != delivery.sequence
            ):
                raise FleetCommandRuntimeError(
                    "expired journal metadata does not match signed command claims"
                )
            return self.processor.reject_verified(
                expired,
                code="EXPIRED",
                message="signed command expired before durable acceptance",
            )


def build_fleet_command_runtime(
    *,
    base_url: str,
    access_token_provider: Callable[[], str],
    ack_sender: AckSender,
    device_id: str,
    space_id: int,
    keyring_path: str | Path,
    inbox_path: str | Path,
    platform_identity: PlatformIdentity,
) -> FleetCommandRuntime:
    if platform_identity.identity_status not in {
        IDENTITY_STATUS_VERIFIED,
        IDENTITY_STATUS_DEV,
    }:
        raise FleetCommandRuntimeError(
            "Fleet command runtime requires verified platform identity, "
            f"got {platform_identity.identity_status}"
        )
    if platform_identity.identity_status == IDENTITY_STATUS_VERIFIED:
        trust_domain = "production"
        require_root_owner = True
    elif platform_identity.platform_profile == PROFILE_MACOS_DEV:
        trust_domain = "macos-dev"
        require_root_owner = False
    elif platform_identity.platform_profile == PROFILE_IQ9075_DEV:
        trust_domain = "iq9075-dev"
        require_root_owner = True
    else:
        raise FleetCommandRuntimeError(
            "Fleet command runtime does not support DEV platform profile "
            f"{platform_identity.platform_profile}"
        )
    keyring = load_fleet_command_keyring(
        keyring_path,
        expected_trust_domain=trust_domain,
        require_root_owner=require_root_owner,
    )
    inbox = DurableCommandInbox(inbox_path)
    try:
        inbox.bind_identity(
            device_id=device_id,
            space_id=space_id,
            trust_domain=trust_domain,
        )
    except CommandInboxError as exc:
        raise FleetCommandRuntimeError(
            f"Fleet command inbox identity binding failed ({exc.code}): {exc}"
        ) from exc
    verifier = FleetCommandVerifier(
        keyring=keyring,
        expected_device_id=device_id,
        expected_space_id=space_id,
        capabilities=platform_identity.capabilities,
    )
    processor = DurableCommandProcessor(
        inbox=inbox,
        verifier=verifier,
        handlers={
            "CONFIG_APPLY": desired_state_handler,
            "STREAM_POLICY": desired_state_handler,
            "AGENT_UPDATE": desired_state_handler,
        },
    )
    return FleetCommandRuntime(
        inbox=inbox,
        processor=processor,
        http_client=FleetCommandHttpClient(
            base_url=base_url,
            access_token_provider=access_token_provider,
        ),
        ack_sender=ack_sender,
    )


def build_fleet_command_runtime_from_env(
    *,
    base_url: str,
    access_token_provider: Callable[[], str],
    ack_sender: AckSender,
    environ: Mapping[str, str] | None = None,
) -> FleetCommandRuntime | None:
    values = os.environ if environ is None else environ
    enabled = str(values.get("NUVION_FLEET_COMMAND_ENABLED") or "false").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return None
    device_id = str(
        values.get("NUVION_DEVICE_ID") or values.get("NUVION_DEVICE_USERNAME") or ""
    ).strip()
    if not device_id:
        raise FleetCommandRuntimeError(
            "NUVION_DEVICE_ID or NUVION_DEVICE_USERNAME is required"
        )
    try:
        space_id = int(str(values.get("NUVION_SPACE_ID") or ""))
    except ValueError as exc:
        raise FleetCommandRuntimeError(
            "NUVION_SPACE_ID must be a positive integer"
        ) from exc
    if space_id < 1:
        raise FleetCommandRuntimeError("NUVION_SPACE_ID must be a positive integer")
    keyring_path = str(values.get("NUVION_FLEET_COMMAND_KEYRING_PATH") or "").strip()
    if not keyring_path:
        raise FleetCommandRuntimeError("NUVION_FLEET_COMMAND_KEYRING_PATH is required")
    identity = resolve_platform_identity(environ=values)
    return build_fleet_command_runtime(
        base_url=base_url,
        access_token_provider=access_token_provider,
        ack_sender=ack_sender,
        device_id=device_id,
        space_id=space_id,
        keyring_path=keyring_path,
        inbox_path=resolve_default_command_inbox_path(values),
        platform_identity=identity,
    )
