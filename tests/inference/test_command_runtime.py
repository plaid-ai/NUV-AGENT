from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
import tempfile
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from nuvion_app.inference.command_inbox import (
    CommandEffectOutcome,
    CommandInboxError,
    DurableCommandInbox,
)
from nuvion_app.inference.command_processor import DurableCommandProcessor
from nuvion_app.inference.command_runtime import (
    FleetCommandRuntime,
    FleetCommandRuntimeError,
    build_fleet_command_runtime,
    desired_state_handler,
    load_fleet_command_keyring,
)
from nuvion_app.inference.command_transport import (
    FleetCommandTransportError,
    PulledCommand,
    PulledCommandPage,
)
from nuvion_app.inference.fleet_command import (
    CommandValidationError,
    Ed25519Keyring,
    FleetCommandVerifier,
    VerifiedFleetCommand,
)


def _command(sequence: int) -> VerifiedFleetCommand:
    payload = {"configVersion": sequence}
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    command_id = str(uuid.uuid4())
    return VerifiedFleetCommand(
        command_id=command_id,
        device_id="sp-3-nuvion-test",
        space_id=3,
        command_type="CONFIG_APPLY",
        schema_version=1,
        issued_at="2026-09-01T02:00:00Z",
        expires_at="2026-09-01T02:10:00Z",
        sequence=sequence,
        payload_base64=base64.urlsafe_b64encode(payload_bytes)
        .decode("ascii")
        .rstrip("="),
        payload_hash=hashlib.sha256(payload_bytes).hexdigest(),
        payload=payload,
        actor="operator@example.com",
        authorization_context="SPACE_ADMIN",
        key_id="test-only",
        required_capability="command.config.apply",
        compact_jws=f"header{sequence}.claims{sequence}.signature{sequence}",
    )


def _signed_delivery(
    private_key: Ed25519PrivateKey,
    *,
    sequence: int,
    claims_overrides: dict[str, object] | None = None,
    kid: str = "runtime-key",
    signing_key: Ed25519PrivateKey | None = None,
) -> PulledCommand:
    payload = json.dumps(
        {"configVersion": sequence},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    command_id = str(uuid.uuid4())
    claims: dict[str, object] = {
        "commandId": command_id,
        "deviceId": "sp-3-nuvion-test",
        "spaceId": 3,
        "type": "CONFIG_APPLY",
        "schemaVersion": 1,
        "issuedAt": "2026-09-01T02:00:00Z",
        "expiresAt": "2026-09-01T02:10:00Z",
        "sequence": sequence,
        "payloadBase64": base64.urlsafe_b64encode(payload).decode("ascii").rstrip("="),
        "payloadHash": hashlib.sha256(payload).hexdigest(),
        "actor": "operator@example.com",
        "authorizationContext": "SPACE_ADMIN",
    }
    claims.update(claims_overrides or {})
    protected = {"alg": "EdDSA", "kid": kid, "typ": "nuvion-command+jws"}
    protected_segment = (
        base64.urlsafe_b64encode(
            json.dumps(protected, separators=(",", ":"), sort_keys=True).encode("utf-8")
        )
        .decode("ascii")
        .rstrip("=")
    )
    claims_segment = (
        base64.urlsafe_b64encode(
            json.dumps(claims, separators=(",", ":"), sort_keys=True).encode("utf-8")
        )
        .decode("ascii")
        .rstrip("=")
    )
    signing_input = f"{protected_segment}.{claims_segment}".encode("ascii")
    signature_segment = (
        base64.urlsafe_b64encode((signing_key or private_key).sign(signing_input))
        .decode("ascii")
        .rstrip("=")
    )
    return PulledCommand(
        command_id=command_id,
        sequence=sequence,
        compact_jws=f"{protected_segment}.{claims_segment}.{signature_segment}",
    )


class MappingVerifier:
    def __init__(self, commands: list[VerifiedFleetCommand]) -> None:
        self.commands = {command.compact_jws: command for command in commands}

    def verify(self, compact_jws: str) -> VerifiedFleetCommand:
        return self.commands[compact_jws]


class BatchHttpClient:
    def __init__(self, batches: list[list[PulledCommand]]) -> None:
        self.batches = list(batches)
        self.after_sequences: list[int] = []

    async def pull_after(
        self,
        after_sequence: int,
        limit: int = 100,
    ) -> PulledCommandPage:
        self.after_sequences.append(after_sequence)
        commands = tuple(self.batches.pop(0) if self.batches else [])
        return PulledCommandPage(
            commands=commands,
            next_after_sequence=commands[-1].sequence if commands else after_sequence,
            has_more=len(commands) == limit,
        )


class ExpiredVerifier:
    def __init__(self, command: VerifiedFleetCommand) -> None:
        self.command = command

    def verify(self, compact_jws: str) -> VerifiedFleetCommand:
        if compact_jws != self.command.compact_jws:
            raise AssertionError("unexpected compact JWS")
        raise CommandValidationError("EXPIRED", "command has expired")

    def verify_expired_for_rejection(self, compact_jws: str) -> VerifiedFleetCommand:
        if compact_jws != self.command.compact_jws:
            raise AssertionError("unexpected compact JWS")
        return self.command


class FleetCommandRuntimeTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)

    def test_keyring_is_strict_domain_bound_and_symlink_safe(self) -> None:
        raw_public_key = (
            Ed25519PrivateKey.generate()
            .public_key()
            .public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        )
        keyring_path = self.root / "keyring.json"
        keyring_path.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "trustDomain": "macos-dev",
                    "keys": {
                        "test-only": base64.b64encode(raw_public_key).decode("ascii")
                    },
                }
            ),
            encoding="utf-8",
        )
        os.chmod(keyring_path, 0o600)

        keyring = load_fleet_command_keyring(
            keyring_path,
            expected_trust_domain="macos-dev",
            require_root_owner=False,
        )
        self.assertEqual(keyring.public_key_der("test-only")[:2], b"0*")

        with self.assertRaises(FleetCommandRuntimeError):
            load_fleet_command_keyring(
                keyring_path,
                expected_trust_domain="production",
                require_root_owner=False,
            )

        symlink = self.root / "keyring-link.json"
        symlink.symlink_to(keyring_path)
        with self.assertRaises(FleetCommandRuntimeError):
            load_fleet_command_keyring(
                symlink,
                expected_trust_domain="macos-dev",
                require_root_owner=False,
            )

        keyring_path.write_text(
            '{"schemaVersion":1,"trustDomain":"macos-dev",'
            '"trustDomain":"production","keys":{}}',
            encoding="utf-8",
        )
        with self.assertRaises(FleetCommandRuntimeError):
            load_fleet_command_keyring(
                keyring_path,
                expected_trust_domain="production",
                require_root_owner=False,
            )

    async def test_reconnect_pull_effect_and_ack_replay_are_end_to_end_idempotent(
        self,
    ) -> None:
        first = _command(1)
        second = _command(2)
        commands = [first, second]
        deliveries = [
            PulledCommand(command.command_id, command.sequence, command.compact_jws)
            for command in commands
        ]
        inbox = DurableCommandInbox(
            self.root / "commands.sqlite3",
            clock=lambda: "2026-09-01T02:00:03Z",
        )
        verifier = MappingVerifier(commands)
        processor = DurableCommandProcessor(
            inbox=inbox,
            verifier=verifier,
            handlers={"CONFIG_APPLY": desired_state_handler},
        )
        http_client = BatchHttpClient([deliveries, []])
        sent: list[tuple[str, dict[str, object]]] = []
        runtime = FleetCommandRuntime(
            inbox=inbox,
            processor=processor,
            http_client=http_client,
            ack_sender=lambda destination, payload: (
                not sent.append((destination, payload))
            ),
        )

        processed = await runtime.on_connected()

        self.assertEqual(processed, 4)
        self.assertEqual(http_client.after_sequences, [0])
        self.assertEqual(
            [payload["status"] for _, payload in sent],
            [
                "RECEIVED",
                "IN_PROGRESS",
                "RECEIVED",
                "IN_PROGRESS",
            ],
        )
        with sqlite3.connect(inbox.path) as connection:
            row = connection.execute(
                "SELECT command_id, sequence, payload_json FROM fleet_desired_state"
            ).fetchone()
        self.assertEqual(row, (second.command_id, 2, '{"configVersion":2}'))

        sent.clear()
        http_client.batches = [deliveries]
        replayed = await runtime.on_connected()
        self.assertEqual(replayed, 4)
        self.assertEqual(len(sent), 4)

        sent.clear()
        wake_body = json.dumps(
            {"commandId": second.command_id, "sequence": second.sequence}
        )
        duplicate_acks = await runtime.on_wakeup(wake_body)
        self.assertEqual(duplicate_acks, 2)
        self.assertEqual(
            [payload["status"] for _, payload in sent],
            ["RECEIVED", "IN_PROGRESS"],
        )

    async def test_on_connected_counts_only_successfully_enqueued_pending_acks(
        self,
    ) -> None:
        command = _command(3)
        inbox = DurableCommandInbox(
            self.root / "pending.sqlite3",
            clock=lambda: "2026-09-01T02:00:03Z",
        )
        inbox.accept(command)
        inbox.transition(command.command_id, "IN_PROGRESS")
        attempted: list[str] = []

        def selective_sender(_destination: str, payload: dict[str, object]) -> bool:
            status = str(payload["status"])
            attempted.append(status)
            return status == "IN_PROGRESS"

        runtime = FleetCommandRuntime(
            inbox=inbox,
            processor=DurableCommandProcessor(
                inbox=inbox,
                verifier=MappingVerifier([command]),
                handlers={"CONFIG_APPLY": desired_state_handler},
            ),
            http_client=BatchHttpClient([[]]),
            ack_sender=selective_sender,
        )

        sent = await runtime.on_connected()

        self.assertEqual(attempted, ["RECEIVED", "IN_PROGRESS"])
        self.assertEqual(sent, 1)

    async def test_poll_from_zero_replays_local_lifecycle_for_server_active_command(
        self,
    ) -> None:
        command = _command(5)
        inbox = DurableCommandInbox(
            self.root / "active-reconcile.sqlite3",
            clock=lambda: "2026-09-01T02:00:03Z",
        )
        processor = DurableCommandProcessor(
            inbox=inbox,
            verifier=MappingVerifier([command]),
            handlers={
                "CONFIG_APPLY": lambda verified, _connection: (
                    CommandEffectOutcome.succeeded(
                        {"configVersion": verified.payload["configVersion"]}
                    )
                )
            },
        )
        processor.process(command.compact_jws)
        http_client = BatchHttpClient(
            [
                [
                    PulledCommand(
                        command.command_id, command.sequence, command.compact_jws
                    )
                ],
                [],
            ]
        )
        sent: list[str] = []
        runtime = FleetCommandRuntime(
            inbox=inbox,
            processor=processor,
            http_client=http_client,
            ack_sender=lambda _destination, payload: (
                not sent.append(str(payload["status"]))
            ),
        )

        replayed = await runtime.poll()

        self.assertEqual(http_client.after_sequences, [0])
        self.assertEqual(replayed, 3)
        self.assertEqual(sent, ["RECEIVED", "IN_PROGRESS", "SUCCEEDED"])

    async def test_active_scan_continues_across_bounded_polls_without_starvation(
        self,
    ) -> None:
        commands = [_command(sequence) for sequence in range(1, 6)]
        deliveries = [
            PulledCommand(command.command_id, command.sequence, command.compact_jws)
            for command in commands
        ]
        inbox = DurableCommandInbox(self.root / "paged-active.sqlite3")
        http_client = BatchHttpClient([deliveries[:2], deliveries[2:4], deliveries[4:]])
        runtime = FleetCommandRuntime(
            inbox=inbox,
            processor=DurableCommandProcessor(
                inbox=inbox,
                verifier=MappingVerifier(commands),
                handlers={"CONFIG_APPLY": desired_state_handler},
            ),
            http_client=http_client,
            ack_sender=lambda _destination, _payload: True,
            max_pull_batches=1,
            pull_page_size=2,
        )

        self.assertEqual(await runtime.poll(), 4)
        self.assertEqual(await runtime.poll(), 4)
        self.assertEqual(await runtime.poll(), 2)

        self.assertEqual(http_client.after_sequences, [0, 2, 4])
        self.assertEqual(inbox.last_sequence(), 5)

    async def test_pending_snapshot_resumes_each_deferred_row_once_across_polls(
        self,
    ) -> None:
        commands = [_command(sequence) for sequence in range(1, 6)]
        inbox = DurableCommandInbox(self.root / "paged-pending.sqlite3")
        for command in commands:
            inbox.accept(command)
            inbox.transition(command.command_id, "IN_PROGRESS")
        resumed: list[int] = []

        def deferred_handler(verified, _connection):
            resumed.append(verified.sequence)
            return CommandEffectOutcome.deferred()

        runtime = FleetCommandRuntime(
            inbox=inbox,
            processor=DurableCommandProcessor(
                inbox=inbox,
                verifier=MappingVerifier(commands),
                handlers={"CONFIG_APPLY": deferred_handler},
            ),
            http_client=BatchHttpClient([[], [], []]),
            ack_sender=lambda _destination, _payload: True,
            max_resume_batches=1,
            resume_page_size=2,
        )

        self.assertEqual(await runtime.on_connected(), 4)
        self.assertEqual(resumed, [1, 2])
        self.assertEqual(await runtime.poll(), 4)
        self.assertEqual(resumed, [1, 2, 3, 4])
        self.assertEqual(await runtime.poll(), 2)
        self.assertEqual(resumed, [1, 2, 3, 4, 5])
        self.assertEqual(await runtime.poll(), 0)
        self.assertEqual(resumed, [1, 2, 3, 4, 5])

    async def test_authenticated_expired_command_is_rejected_and_cursor_advances(
        self,
    ) -> None:
        command = _command(7)
        inbox = DurableCommandInbox(
            self.root / "expired.sqlite3",
            clock=lambda: "2026-09-01T02:10:03Z",
        )
        processor = DurableCommandProcessor(
            inbox=inbox,
            verifier=ExpiredVerifier(command),
            handlers={"CONFIG_APPLY": desired_state_handler},
        )
        sent: list[dict[str, object]] = []
        runtime = FleetCommandRuntime(
            inbox=inbox,
            processor=processor,
            http_client=BatchHttpClient(
                [
                    [PulledCommand(command.command_id, 7, command.compact_jws)],
                    [],
                ]
            ),
            ack_sender=lambda _destination, payload: not sent.append(payload),
        )

        processed = await runtime.on_connected()

        self.assertEqual(processed, 3)
        self.assertEqual(inbox.last_sequence(), 7)
        self.assertEqual(
            [payload["status"] for payload in sent],
            ["RECEIVED", "IN_PROGRESS", "REJECTED"],
        )
        self.assertEqual(sent[-1]["code"], "EXPIRED")

    async def test_authenticated_policy_rejection_advances_to_later_command(
        self,
    ) -> None:
        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        unsupported = _signed_delivery(
            private_key,
            sequence=7,
            claims_overrides={"authorizationContext": "VIEWER"},
        )
        executable = _signed_delivery(private_key, sequence=8)
        inbox = DurableCommandInbox(self.root / "policy-rejection.sqlite3")
        sent: list[dict[str, object]] = []
        runtime = FleetCommandRuntime(
            inbox=inbox,
            processor=DurableCommandProcessor(
                inbox=inbox,
                verifier=FleetCommandVerifier(
                    keyring=Ed25519Keyring({"runtime-key": public_key}),
                    expected_device_id="sp-3-nuvion-test",
                    expected_space_id=3,
                    capabilities={"command.config.apply"},
                    clock=lambda: datetime(2026, 9, 1, 2, 5, tzinfo=timezone.utc),
                    allowed_clock_skew=timedelta(seconds=30),
                ),
                handlers={"CONFIG_APPLY": desired_state_handler},
            ),
            http_client=BatchHttpClient([[unsupported, executable], []]),
            ack_sender=lambda _destination, payload: not sent.append(payload),
            pull_page_size=2,
        )

        processed = await runtime.on_connected()

        self.assertEqual(processed, 5)
        self.assertEqual(inbox.last_sequence(), 8)
        self.assertEqual(inbox.get(unsupported.command_id).status, "REJECTED")
        self.assertEqual(inbox.get(executable.command_id).status, "IN_PROGRESS")
        self.assertEqual(
            [payload["status"] for payload in sent],
            ["RECEIVED", "IN_PROGRESS", "REJECTED", "RECEIVED", "IN_PROGRESS"],
        )
        self.assertEqual(sent[2]["code"], "UNSUPPORTED_AUTHORIZATION_CONTEXT")

    async def test_policy_disguise_cannot_advance_unauthenticated_delivery(
        self,
    ) -> None:
        private_key = Ed25519PrivateKey.generate()
        attacker_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        cases = (
            (
                "UNKNOWN_KEY_ID",
                _signed_delivery(
                    private_key,
                    sequence=11,
                    claims_overrides={"authorizationContext": "VIEWER"},
                    kid="unknown-key",
                ),
            ),
            (
                "INVALID_SIGNATURE",
                _signed_delivery(
                    private_key,
                    sequence=12,
                    claims_overrides={"authorizationContext": "VIEWER"},
                    signing_key=attacker_key,
                ),
            ),
            (
                "DEVICE_MISMATCH",
                _signed_delivery(
                    private_key,
                    sequence=13,
                    claims_overrides={
                        "authorizationContext": "VIEWER",
                        "deviceId": "another-device",
                    },
                ),
            ),
            (
                "SPACE_MISMATCH",
                _signed_delivery(
                    private_key,
                    sequence=14,
                    claims_overrides={
                        "authorizationContext": "VIEWER",
                        "spaceId": 999,
                    },
                ),
            ),
            (
                "INVALID_CLAIMS",
                _signed_delivery(
                    private_key,
                    sequence=15,
                    claims_overrides={
                        "authorizationContext": "VIEWER",
                        "sequence": True,
                    },
                ),
            ),
            (
                "INVALID_PAYLOAD_HASH",
                _signed_delivery(
                    private_key,
                    sequence=16,
                    claims_overrides={
                        "authorizationContext": "VIEWER",
                        "payloadHash": "0" * 64,
                    },
                ),
            ),
            (
                "NOT_YET_VALID",
                _signed_delivery(
                    private_key,
                    sequence=17,
                    claims_overrides={
                        "authorizationContext": "VIEWER",
                        "issuedAt": "2026-09-01T02:06:00Z",
                    },
                ),
            ),
        )

        for index, (expected_code, delivery) in enumerate(cases):
            with self.subTest(expected_code=expected_code):
                inbox = DurableCommandInbox(
                    self.root / f"policy-disguise-{index}.sqlite3"
                )
                sent: list[dict[str, object]] = []
                runtime = FleetCommandRuntime(
                    inbox=inbox,
                    processor=DurableCommandProcessor(
                        inbox=inbox,
                        verifier=FleetCommandVerifier(
                            keyring=Ed25519Keyring({"runtime-key": public_key}),
                            expected_device_id="sp-3-nuvion-test",
                            expected_space_id=3,
                            capabilities={"command.config.apply"},
                            clock=lambda: datetime(
                                2026, 9, 1, 2, 5, tzinfo=timezone.utc
                            ),
                            allowed_clock_skew=timedelta(seconds=30),
                        ),
                        handlers={"CONFIG_APPLY": desired_state_handler},
                    ),
                    http_client=BatchHttpClient([[delivery]]),
                    ack_sender=lambda _destination, payload, _sent=sent: (
                        not _sent.append(payload)
                    ),
                )

                with self.assertRaises(CommandValidationError) as raised:
                    await runtime.on_connected()

                self.assertEqual(raised.exception.code, expected_code)
                self.assertEqual(inbox.last_sequence(), 0)
                self.assertIsNone(inbox.get(delivery.command_id))
                self.assertEqual(sent, [])

    async def test_policy_rejection_cannot_bypass_signed_journal_envelope(self) -> None:
        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        signed = _signed_delivery(
            private_key,
            sequence=18,
            claims_overrides={"schemaVersion": 2},
        )
        mismatched = PulledCommand(
            command_id=str(uuid.uuid4()),
            sequence=signed.sequence,
            compact_jws=signed.compact_jws,
        )
        inbox = DurableCommandInbox(self.root / "policy-envelope-mismatch.sqlite3")
        runtime = FleetCommandRuntime(
            inbox=inbox,
            processor=DurableCommandProcessor(
                inbox=inbox,
                verifier=FleetCommandVerifier(
                    keyring=Ed25519Keyring({"runtime-key": public_key}),
                    expected_device_id="sp-3-nuvion-test",
                    expected_space_id=3,
                    capabilities={"command.config.apply"},
                    clock=lambda: datetime(2026, 9, 1, 2, 5, tzinfo=timezone.utc),
                    allowed_clock_skew=timedelta(seconds=30),
                ),
                handlers={"CONFIG_APPLY": desired_state_handler},
            ),
            http_client=BatchHttpClient([[mismatched]]),
            ack_sender=lambda _destination, _payload: True,
        )

        with self.assertRaises(FleetCommandTransportError) as raised:
            await runtime.on_connected()

        self.assertEqual(raised.exception.code, "JOURNAL_ENVELOPE_MISMATCH")
        self.assertEqual(inbox.last_sequence(), 0)
        self.assertIsNone(inbox.get(signed.command_id))

    def test_identity_scope_mismatch_is_wrapped_as_runtime_configuration_error(
        self,
    ) -> None:
        identity = SimpleNamespace(
            identity_status="DEV",
            platform_profile="macos_dev",
            capabilities=frozenset({"command.config.apply"}),
        )
        with (
            mock.patch(
                "nuvion_app.inference.command_runtime.load_fleet_command_keyring",
                return_value=object(),
            ),
            mock.patch.object(
                DurableCommandInbox,
                "bind_identity",
                side_effect=CommandInboxError(
                    "IDENTITY_SCOPE_MISMATCH",
                    "scope changed",
                ),
            ),
            self.assertRaises(FleetCommandRuntimeError) as raised,
        ):
            build_fleet_command_runtime(
                base_url="https://api.example.test",
                access_token_provider=lambda: "token",
                ack_sender=lambda _destination, _payload: True,
                device_id="sp-3-nuvion-test",
                space_id=3,
                keyring_path=self.root / "unused-keyring.json",
                inbox_path=self.root / "mismatch.sqlite3",
                platform_identity=identity,
            )

        self.assertIn("IDENTITY_SCOPE_MISMATCH", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
