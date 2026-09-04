from __future__ import annotations

import base64
import hashlib
import io
import json
import shutil
import sqlite3
import tarfile
import tempfile
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from nuvion_app.inference.fleet_command import Ed25519Keyring, FleetCommandVerifier
from nuvion_app.runtime.release_bom import (
    ReleaseKeyring,
    ReleaseTarget,
    build_release_bom_signature,
    build_release_bom_v2_payload,
    canonical_release_bom_json,
    canonical_release_bom_signature_json,
)
from nuvion_updater.controller import UpdaterController
from nuvion_updater.errors import UpdaterError, UpdaterSecurityError
from nuvion_updater.health_attestation import (
    CommitProcessIdentity,
    HealthAttestationVerifier,
)
from nuvion_updater.protocol import UpdaterProtocol
from nuvion_updater.repository import ContentAddressedReleaseRepository
from nuvion_updater.slots import ReleaseSlotManager
from nuvion_updater.store import CommitGate, UpdatePhase, UpdaterStore
from nuvion_updater.trust import DeviceBinding


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


class UpdaterCoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.remote = self.root / "remote"
        self.remote.mkdir(mode=0o755)
        self.install_root = self.root / "opt" / "nuv-agent"
        self.download_root = self.root / "var" / "lib" / "nuvion-updater" / "downloads"
        self.state_path = self.root / "var" / "lib" / "nuvion-updater" / "updater.sqlite3"
        self.agent_state = self.root / "var" / "lib" / "nuv-agent"
        self.agent_state.mkdir(parents=True)
        (self.agent_state / "events.sqlite3").write_bytes(b"durable-event-sentinel")

        self.command_key = Ed25519PrivateKey.generate()
        command_public = self.command_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        self.release_key = Ed25519PrivateKey.generate()
        release_public = self.release_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        self.release_keyring = ReleaseKeyring({"release-test": release_public})
        self.health_key = Ed25519PrivateKey.generate()
        health_public = self.health_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        self.health_verifier = HealthAttestationVerifier(
            keyring=Ed25519Keyring({"health-test": health_public}),
            clock=lambda: datetime.now(timezone.utc),
        )
        self.binding = DeviceBinding(
            trust_domain="production",
            device_id="sp-3-device-1",
            space_id=3,
            product_model="NUVION",
            platform_profile="rpi5_deepx_dx_m1",
            hardware_revision="REV_A",
            architecture="arm64",
            docker_required=False,
        )
        self.verifier = FleetCommandVerifier(
            keyring=Ed25519Keyring({"command-test": command_public}),
            expected_device_id=self.binding.device_id,
            expected_space_id=self.binding.space_id,
            capabilities={"command.agent.update"},
            clock=lambda: datetime.now(timezone.utc),
        )
        self.store = UpdaterStore(self.state_path, require_root_owner=False)
        self.slots = ReleaseSlotManager(self.install_root, require_root_owner=False)
        self._seed_current_slot()
        self.store.commit_release(
            sequence=1,
            bom_digest="sha256:" + "0" * 64,
        )

    def _seed_current_slot(self) -> None:
        old_slot = self.slots.releases_root / ("0" * 64)
        (old_slot / "bin").mkdir(parents=True)
        (old_slot / ".nuvion").mkdir()
        entrypoint = old_slot / "bin" / "nuv-agent"
        entrypoint.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        entrypoint.chmod(0o755)
        (old_slot / ".nuvion" / "release.json").write_text(
            json.dumps(
                {
                    "schemaVersion": 2,
                    "bomDigest": "sha256:" + "0" * 64,
                    "agentVersion": "0.1.115",
                    "releaseSequence": 1,
                    "artifactDigest": "sha256:" + "1" * 64,
                    "componentSha": "1" * 40,
                    "configSchema": "12",
                    "publisherKeyId": "release-test",
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        old_slot.chmod(0o755)
        (self.install_root / "current").symlink_to("releases/" + "0" * 64)

    def _bundle(self, *, unsafe_symlink: bool = False) -> Path:
        artifact = self.root / f"nuv-agent-{uuid.uuid4().hex}.bundle.tar"
        with tarfile.open(artifact, "w") as archive:
            directory = tarfile.TarInfo("bin")
            directory.type = tarfile.DIRTYPE
            directory.mode = 0o755
            archive.addfile(directory)
            if unsafe_symlink:
                link = tarfile.TarInfo("bin/nuv-agent")
                link.type = tarfile.SYMTYPE
                link.linkname = "/bin/sh"
                link.mode = 0o777
                archive.addfile(link)
            else:
                payload = b"#!/bin/sh\nexit 0\n"
                executable = tarfile.TarInfo("bin/nuv-agent")
                executable.size = len(payload)
                executable.mode = 0o755
                archive.addfile(executable, io.BytesIO(payload))
        return artifact

    def _publish(
        self,
        *,
        release_sequence: int = 2,
        version: str = "0.1.116",
        target: ReleaseTarget | None = None,
        unsafe_symlink: bool = False,
    ) -> tuple[dict[str, object], Path]:
        artifact = self._bundle(unsafe_symlink=unsafe_symlink)
        payload = build_release_bom_v2_payload(
            bom_id=f"nuv-agent-{version}-arm64",
            release_sequence=release_sequence,
            agent_version=version,
            component_sha="a" * 40,
            config_schema="12",
            min_updater_version="0.1.0",
            targets=[
                target
                or ReleaseTarget(
                    product_model="NUVION",
                    platform_profile="rpi5_deepx_dx_m1",
                    hardware_revision="REV_A",
                    architecture="arm64",
                )
            ],
            artifact_path=artifact,
            artifact_kind="agent-bundle",
            built_at="2026-09-01T10:00:00Z",
        )
        signature = build_release_bom_signature(
            payload,
            key_id="release-test",
            private_key=self.release_key,
        )
        digest = str(payload["bomDigest"])[7:]
        release_dir = self.remote / "releases" / "by-bom-sha256" / digest
        release_dir.mkdir(parents=True)
        (release_dir / "release-bom.json").write_text(
            canonical_release_bom_json(payload), encoding="utf-8"
        )
        (release_dir / "release-bom.json.sig").write_text(
            canonical_release_bom_signature_json(signature), encoding="utf-8"
        )
        shutil.copyfile(artifact, release_dir / artifact.name)
        return payload, release_dir

    def _command(
        self,
        payload: dict[str, object],
        *,
        sequence: int = 10,
        device_id: str | None = None,
        expires_at: datetime | None = None,
        extra_payload: dict[str, object] | None = None,
    ) -> tuple[str, str]:
        command_id = str(uuid.uuid4())
        update_payload: dict[str, object] = {
            "targetVersion": payload["agentVersion"],
            "bomDigest": payload["bomDigest"],
        }
        update_payload.update(extra_payload or {})
        encoded_payload = json.dumps(
            update_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        now = datetime.now(timezone.utc)
        claims = {
            "commandId": command_id,
            "deviceId": device_id or self.binding.device_id,
            "spaceId": self.binding.space_id,
            "type": "AGENT_UPDATE",
            "schemaVersion": 1,
            "issuedAt": now.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "expiresAt": (expires_at or now + timedelta(minutes=10))
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "sequence": sequence,
            "payloadBase64": _b64url(encoded_payload),
            "payloadHash": hashlib.sha256(encoded_payload).hexdigest(),
            "actor": "release-manager@example.test",
            "authorizationContext": "SPACE_ADMIN",
        }
        protected = {"alg": "EdDSA", "kid": "command-test", "typ": "nuvion-command+jws"}
        protected_segment = _b64url(
            json.dumps(protected, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        claims_segment = _b64url(
            json.dumps(claims, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        signature = self.command_key.sign(
            f"{protected_segment}.{claims_segment}".encode("ascii")
        )
        return command_id, f"{protected_segment}.{claims_segment}.{_b64url(signature)}"

    def _health_attestation(
        self,
        gate: CommitGate,
        *,
        claims_override: dict[str, object] | None = None,
        protected_override: dict[str, object] | None = None,
        issued_at: datetime | None = None,
        expires_at: datetime | None = None,
    ) -> str:
        now = issued_at or datetime.now(timezone.utc)
        claims = {
            "schemaVersion": 1,
            "jti": str(uuid.uuid4()),
            "aud": "nuvion-updater",
            "purpose": "agent-update-commit",
            "trustDomain": self.binding.trust_domain,
            "gateId": gate.gate_id,
            "challenge": gate.challenge,
            "deviceId": self.binding.device_id,
            "commandId": gate.command_id,
            "commandExpiresAt": gate.command_expires_at,
            "bomDigest": gate.bom_digest,
            "componentSha": gate.component_sha,
            "releaseSequence": gate.release_sequence,
            "productModel": self.binding.product_model,
            "platformProfile": self.binding.platform_profile,
            "hardwareRevision": self.binding.hardware_revision,
            "architecture": self.binding.architecture,
            "health": "HEALTHY",
            "issuedAt": now.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "expiresAt": (expires_at or now + timedelta(seconds=30))
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
        }
        protected = {
            "alg": "EdDSA",
            "kid": "health-test",
            "typ": "nuvion-update-health+jws",
        }
        claims.update(claims_override or {})
        protected.update(protected_override or {})
        protected_segment = _b64url(
            json.dumps(protected, sort_keys=True, separators=(",", ":")).encode()
        )
        claims_segment = _b64url(
            json.dumps(claims, sort_keys=True, separators=(",", ":")).encode()
        )
        signature = self.health_key.sign(
            f"{protected_segment}.{claims_segment}".encode("ascii")
        )
        return f"{protected_segment}.{claims_segment}.{_b64url(signature)}"

    def _attested_commit(
        self,
        controller: UpdaterController,
        command_id: str,
        *,
        peer_pid: int = 4242,
    ):
        gate = controller.begin_commit_gate(command_id, peer_pid=peer_pid)
        return controller.commit(
            command_id,
            gate_id=gate.gate_id,
            health_attestation_jws=self._health_attestation(gate),
            peer_pid=peer_pid,
        )

    def _controller(
        self,
        *,
        boot_health: bool = True,
        functional_health: bool = True,
        store: UpdaterStore | None = None,
        slots: ReleaseSlotManager | None = None,
        activation_callback: object | None = None,
        boot_callback: object | None = None,
        functional_callback: object | None = None,
        rollback_boot_health: bool = True,
        safe_stop_callback: object | None = None,
        clock: object | None = None,
        activation_timeout_seconds: int = 300,
        boot_timeout_seconds: int = 120,
        functional_timeout_seconds: int = 300,
        commit_timeout_seconds: int = 120,
    ) -> UpdaterController:
        repository = ContentAddressedReleaseRepository(
            base_url=self.remote.as_uri(),
            download_root=self.download_root,
            require_root_owner=False,
            allow_file_url=True,
            disk_reserve_bytes=0,
        )
        active_slots = slots or self.slots
        return UpdaterController(
            store=store or self.store,
            slots=slots or self.slots,
            repository=repository,
            command_verifier=self.verifier,
            release_keyring=self.release_keyring,
            binding=self.binding,
            updater_version="0.1.0",
            activation_callback=(
                activation_callback
                if activation_callback is not None
                else lambda _slot: None
            ),  # type: ignore[arg-type]
            boot_health_check=(
                boot_callback
                if boot_callback is not None
                else lambda _state: (boot_health, "boot-fake")
            ),  # type: ignore[arg-type]
            functional_health_check=(
                functional_callback
                if functional_callback is not None
                else lambda _state: (functional_health, "functional-fake")
            ),  # type: ignore[arg-type]
            commit_process_check=lambda state, pid: CommitProcessIdentity(
                pid=pid,
                start_ticks=123456,
                boot_id="00000000-0000-4000-8000-000000000123",
                active_slot=active_slots.relative_target(state.candidate_slot),
            ),
            health_attestation_verifier=self.health_verifier,
            rollback_boot_health_check=lambda _slot: (
                rollback_boot_health,
                "rollback-boot-fake",
            ),
            safe_stop_callback=(
                safe_stop_callback
                if safe_stop_callback is not None
                else lambda: None
            ),  # type: ignore[arg-type]
            clock=clock,  # type: ignore[arg-type]
            activation_timeout_seconds=activation_timeout_seconds,
            boot_timeout_seconds=boot_timeout_seconds,
            functional_timeout_seconds=functional_timeout_seconds,
            commit_timeout_seconds=commit_timeout_seconds,
        )

    def test_signed_release_atomic_lifecycle_and_agent_state_preserved(self) -> None:
        payload, _ = self._publish()
        command_id, command = self._command(payload)
        controller = self._controller()

        staged = controller.authorize_and_stage(command)
        self.assertEqual(staged.phase, UpdatePhase.VERIFIED)
        self.assertFalse((self.download_root / str(payload["bomDigest"])[7:]).exists())
        self.assertEqual(self.slots.current_slot(), "releases/" + "0" * 64)
        activated = controller.activate(command_id)
        self.assertEqual(activated.phase, UpdatePhase.ACTIVATING)
        self.assertNotEqual(self.slots.current_slot(), "releases/" + "0" * 64)
        self.assertEqual(self.slots.previous_slot(), "releases/" + "0" * 64)
        self.assertEqual(
            controller.report_boot_health(command_id, healthy=True).phase,
            UpdatePhase.BOOT_HEALTHY,
        )
        self.assertEqual(
            controller.report_functional_health(command_id, healthy=True).phase,
            UpdatePhase.FUNCTIONAL_HEALTHY,
        )
        committed = self._attested_commit(controller, command_id)

        self.assertEqual(committed.phase, UpdatePhase.COMMITTED)
        evidence = committed.public_dict()
        self.assertEqual(evidence["releaseSequence"], 2)
        self.assertEqual(
            evidence["artifactDigest"],
            "sha256:" + str(payload["artifact"]["sha256"]),  # type: ignore[index]
        )
        self.assertEqual(evidence["componentSha"], "a" * 40)
        self.assertEqual(evidence["configSchema"], "12")
        self.assertEqual(evidence["bomVerificationStatus"], "VERIFIED")
        self.assertEqual(evidence["health"], "FUNCTIONAL_HEALTHY")
        self.assertEqual(evidence["functionalHealth"], "FUNCTIONAL_HEALTHY")
        self.assertEqual(evidence["previousVersion"], "0.1.115")
        self.assertEqual(self.store.current_release_sequence(), 2)
        self.assertEqual(self.store.current_bom_digest(), payload["bomDigest"])
        with self.assertRaises(UpdaterError) as rollback_after_commit:
            controller.rollback(command_id, reason="UNSIGNED_DOWNGRADE_ATTEMPT")
        self.assertEqual(rollback_after_commit.exception.code, "INVALID_PHASE")
        self.assertEqual(self.slots.current_slot(), evidence["slot"])
        self.assertEqual(
            (self.agent_state / "events.sqlite3").read_bytes(),
            b"durable-event-sentinel",
        )

    def test_commit_gate_is_process_bound_idempotent_and_single_use(self) -> None:
        payload, _ = self._publish()
        command_id, command = self._command(payload)
        controller = self._controller()
        controller.authorize_and_stage(command)
        controller.activate(command_id)
        controller.report_boot_health(command_id, healthy=True)
        functional = controller.report_functional_health(command_id, healthy=True)

        with self.assertRaises(UpdaterSecurityError) as missing:
            controller.commit(
                command_id,
                gate_id=str(uuid.uuid4()),
                health_attestation_jws="a.b.c",
                peer_pid=4242,
            )
        self.assertEqual(missing.exception.code, "COMMIT_GATE_REQUIRED")

        gate = controller.begin_commit_gate(command_id, peer_pid=4242)
        self.assertEqual(len(gate.challenge), 43)
        self.assertEqual(len(base64.urlsafe_b64decode(gate.challenge + "=")), 32)
        self.assertEqual(gate.health_deadline, functional.health_deadline)
        self.assertEqual(gate.command_expires_at, self.store.get(command_id).command_expires_at)
        self.assertEqual(
            controller.begin_commit_gate(command_id, peer_pid=4242), gate
        )
        with self.assertRaises(UpdaterSecurityError) as wrong_peer:
            controller.begin_commit_gate(command_id, peer_pid=4243)
        self.assertEqual(
            wrong_peer.exception.code, "COMMIT_GATE_BINDING_MISMATCH"
        )

        compact_jws = self._health_attestation(gate)
        committed = controller.commit(
            command_id,
            gate_id=gate.gate_id,
            health_attestation_jws=compact_jws,
            peer_pid=4242,
        )
        self.assertEqual(committed.phase, UpdatePhase.COMMITTED)
        consumed = self.store.commit_gate(command_id)
        assert consumed is not None
        self.assertIsNotNone(consumed.consumed_at)
        self.assertEqual(
            controller.commit(
                command_id,
                gate_id=gate.gate_id,
                health_attestation_jws=compact_jws,
                peer_pid=4242,
            ).phase,
            UpdatePhase.COMMITTED,
        )
        with self.assertRaises(UpdaterSecurityError) as replay:
            controller.commit(
                command_id,
                gate_id=gate.gate_id,
                health_attestation_jws=self._health_attestation(gate),
                peer_pid=4242,
            )
        self.assertEqual(replay.exception.code, "HEALTH_ATTESTATION_REPLAY")

    def test_commit_rejects_domain_identity_challenge_and_ttl_mismatch(self) -> None:
        payload, _ = self._publish()
        command_id, command = self._command(payload)
        controller = self._controller()
        controller.authorize_and_stage(command)
        controller.activate(command_id)
        controller.report_boot_health(command_id, healthy=True)
        controller.report_functional_health(command_id, healthy=True)
        gate = controller.begin_commit_gate(command_id, peer_pid=4242)
        valid_for_tamper = self._health_attestation(gate)
        protected, claims, encoded_signature = valid_for_tamper.split(".")
        signature = bytearray(
            base64.urlsafe_b64decode(encoded_signature + "=" * (-len(encoded_signature) % 4))
        )
        signature[0] ^= 1
        tampered_signature = f"{protected}.{claims}.{_b64url(bytes(signature))}"

        cases = (
            (
                "tampered-signature",
                tampered_signature,
                "INVALID_HEALTH_ATTESTATION_SIGNATURE",
            ),
            (
                "wrong-domain",
                self._health_attestation(
                    gate, claims_override={"aud": "some-other-service"}
                ),
                "INVALID_HEALTH_ATTESTATION_DOMAIN",
            ),
            (
                "wrong-challenge",
                self._health_attestation(
                    gate, claims_override={"challenge": "A" * 43}
                ),
                "HEALTH_ATTESTATION_CHALLENGE_MISMATCH",
            ),
            (
                "wrong-component",
                self._health_attestation(
                    gate, claims_override={"componentSha": "b" * 40}
                ),
                "HEALTH_ATTESTATION_MISMATCH",
            ),
            (
                "wrong-command-expiry",
                self._health_attestation(
                    gate,
                    claims_override={
                        "commandExpiresAt": (
                            datetime.now(timezone.utc) + timedelta(hours=1)
                        ).isoformat(timespec="seconds").replace("+00:00", "Z")
                    },
                ),
                "HEALTH_ATTESTATION_MISMATCH",
            ),
            (
                "cross-protocol-jws",
                self._health_attestation(
                    gate, protected_override={"typ": "nuvion-command+jws"}
                ),
                "INVALID_HEALTH_ATTESTATION_DOMAIN",
            ),
            (
                "expired",
                self._health_attestation(
                    gate,
                    issued_at=datetime.now(timezone.utc) - timedelta(minutes=2),
                    expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
                ),
                "HEALTH_ATTESTATION_EXPIRED",
            ),
        )
        for name, compact_jws, code in cases:
            with self.subTest(name=name), self.assertRaises(
                UpdaterSecurityError
            ) as raised:
                controller.commit(
                    command_id,
                    gate_id=gate.gate_id,
                    health_attestation_jws=compact_jws,
                    peer_pid=4242,
                )
            self.assertEqual(raised.exception.code, code)
        self.assertEqual(
            self.store.get(command_id).phase, UpdatePhase.FUNCTIONAL_HEALTHY
        )

    def test_duplicate_command_is_idempotent_and_cross_device_is_rejected(self) -> None:
        payload, _ = self._publish()
        command_id, command = self._command(payload)
        controller = self._controller()

        first = controller.authorize_and_stage(command)
        second = controller.authorize_and_stage(command)
        self.assertEqual(first, second)
        self.assertEqual(self.store.get(command_id).phase, UpdatePhase.VERIFIED)

        _, replay = self._command(payload, sequence=9)
        with self.assertRaises(UpdaterSecurityError) as replay_error:
            controller.authorize_and_stage(replay)
        self.assertEqual(replay_error.exception.code, "COMMAND_REPLAY")

        _, wrong_device = self._command(
            payload, sequence=11, device_id="sp-3-another-device"
        )
        with self.assertRaises(UpdaterSecurityError) as raised:
            controller.authorize_and_stage(wrong_device)
        self.assertEqual(raised.exception.code, "DEVICE_MISMATCH")

    def test_commit_phase_and_release_identity_roll_back_together_on_failure(self) -> None:
        payload, _ = self._publish(version="0.1.133", release_sequence=13)
        command_id, command = self._command(payload, sequence=49)
        controller = self._controller()
        controller.authorize_and_stage(command)
        controller.activate(command_id)
        controller.report_boot_health(command_id, healthy=True)
        functional = controller.report_functional_health(command_id, healthy=True)
        self.assertEqual(functional.phase, UpdatePhase.FUNCTIONAL_HEALTHY)

        connection = sqlite3.connect(self.state_path)
        connection.execute(
            """
            CREATE TRIGGER abort_atomic_commit
            BEFORE UPDATE OF phase ON updater_command
            WHEN NEW.phase = 'COMMITTED'
            BEGIN
                SELECT RAISE(ABORT, 'simulated power loss');
            END
            """
        )
        connection.commit()
        connection.close()

        with self.assertRaises(sqlite3.DatabaseError):
            self._attested_commit(controller, command_id)

        reopened = UpdaterStore(self.state_path, require_root_owner=False)
        persisted = reopened.get(command_id)
        assert persisted is not None
        self.assertEqual(persisted.phase, UpdatePhase.FUNCTIONAL_HEALTHY)
        self.assertEqual(reopened.current_release_sequence(), 1)
        self.assertEqual(reopened.current_bom_digest(), "sha256:" + "0" * 64)
        persisted_gate = reopened.commit_gate(command_id)
        assert persisted_gate is not None
        self.assertIsNone(persisted_gate.attestation_id)
        self.assertIsNone(persisted_gate.consumed_at)

    def test_tampered_signature_wrong_target_and_downgrade_fail_before_activation(self) -> None:
        cases: list[tuple[str, dict[str, object], Path]] = []
        tampered_payload, tampered_dir = self._publish(version="0.1.117")
        signature_path = tampered_dir / "release-bom.json.sig"
        signature = json.loads(signature_path.read_text(encoding="utf-8"))
        encoded = signature["signature"]
        signature["signature"] = ("A" if encoded[0] != "A" else "B") + encoded[1:]
        signature_path.write_text(json.dumps(signature), encoding="utf-8")
        cases.append(("tampered", tampered_payload, tampered_dir))

        wrong_payload, wrong_dir = self._publish(
            version="0.1.118",
            target=ReleaseTarget(
                product_model="NUVION_ULTRA",
                platform_profile="jetson_orin_nx",
                hardware_revision="REV_A",
                architecture="arm64",
            ),
        )
        cases.append(("wrong-target", wrong_payload, wrong_dir))

        self.store.commit_release(sequence=4, bom_digest="sha256:" + "f" * 64)
        downgrade_payload, downgrade_dir = self._publish(
            version="0.1.119", release_sequence=3
        )
        cases.append(("downgrade", downgrade_payload, downgrade_dir))

        for index, (name, payload, _directory) in enumerate(cases, start=20):
            with self.subTest(name=name):
                command_id, command = self._command(payload, sequence=index)
                with self.assertRaises((UpdaterError, UpdaterSecurityError)):
                    self._controller().authorize_and_stage(command)
                self.assertEqual(self.store.get(command_id).phase, UpdatePhase.FAILED)
                self.assertEqual(self.slots.current_slot(), "releases/" + "0" * 64)

    def test_unsafe_bundle_and_disk_full_fail_closed(self) -> None:
        unsafe_payload, _ = self._publish(unsafe_symlink=True)
        unsafe_id, unsafe_command = self._command(unsafe_payload, sequence=30)
        with self.assertRaises(UpdaterSecurityError) as raised:
            self._controller().authorize_and_stage(unsafe_command)
        self.assertEqual(raised.exception.code, "UNSAFE_BUNDLE_TYPE")
        self.assertEqual(self.store.get(unsafe_id).phase, UpdatePhase.FAILED)

        disk_payload, _ = self._publish(version="0.1.120", release_sequence=5)
        _, disk_command = self._command(disk_payload, sequence=31)
        with (
            mock.patch(
                "nuvion_updater.repository.shutil.disk_usage",
                return_value=SimpleNamespace(free=0),
            ),
            self.assertRaises(UpdaterError) as disk_error,
        ):
            self._controller().authorize_and_stage(disk_command)
        self.assertEqual(disk_error.exception.code, "INSUFFICIENT_DISK")

        install_payload, _ = self._publish(
            version="0.1.130",
            release_sequence=10,
        )
        _, install_command = self._command(install_payload, sequence=32)
        constrained_slots = ReleaseSlotManager(
            self.install_root,
            require_root_owner=False,
            disk_reserve_bytes=10**18,
        )
        with self.assertRaises(UpdaterError) as install_error:
            self._controller(slots=constrained_slots).authorize_and_stage(
                install_command
            )
        self.assertEqual(
            install_error.exception.code,
            "INSUFFICIENT_INSTALL_DISK",
        )

    def test_slot_count_is_bounded_and_stale_incoming_is_recovered(self) -> None:
        stale = self.slots.releases_root / (".incoming-" + "a" * 64 + "-" + "b" * 32)
        stale.mkdir(mode=0o700)
        (stale / "partial").write_bytes(b"partial")
        stale.chmod(0o755)
        second = self.slots.releases_root / ("1" * 64)
        second.mkdir(mode=0o755)
        bounded = ReleaseSlotManager(
            self.install_root,
            require_root_owner=False,
            disk_reserve_bytes=0,
            max_release_slots=2,
        )
        self.assertFalse(stale.exists())

        payload, _ = self._publish(version="0.1.131", release_sequence=11)
        _, command = self._command(payload, sequence=33)
        with self.assertRaises(UpdaterError) as capacity:
            self._controller(slots=bounded).authorize_and_stage(command)
        self.assertEqual(capacity.exception.code, "SLOT_CAPACITY_EXHAUSTED")

    def test_tampered_bom_digest_fails_closed(self) -> None:
        payload, release_dir = self._publish(version="0.1.121")
        bom_path = release_dir / "release-bom.json"
        tampered = json.loads(bom_path.read_text(encoding="utf-8"))
        tampered["componentSha"] = "b" * 40
        bom_path.write_text(json.dumps(tampered), encoding="utf-8")
        command_id, command = self._command(payload, sequence=35)

        with self.assertRaises(UpdaterSecurityError) as raised:
            self._controller().authorize_and_stage(command)

        self.assertEqual(raised.exception.code, "RELEASE_VERIFICATION_FAILED")
        self.assertEqual(self.store.get(command_id).phase, UpdatePhase.FAILED)
        self.assertEqual(self.slots.current_slot(), "releases/" + "0" * 64)
        self.assertFalse((self.download_root / str(payload["bomDigest"])[7:]).exists())

    def test_restart_reopens_activating_phase_and_bad_functional_health_rolls_back(self) -> None:
        payload, _ = self._publish()
        command_id, command = self._command(payload, sequence=40)
        controller = self._controller(functional_health=False)
        controller.authorize_and_stage(command)
        controller.activate(command_id)

        reopened_store = UpdaterStore(self.state_path, require_root_owner=False)
        reopened_slots = ReleaseSlotManager(
            self.install_root, require_root_owner=False
        )
        recovered = self._controller(
            functional_health=False,
            store=reopened_store,
            slots=reopened_slots,
        ).recover()
        self.assertEqual(recovered.phase, UpdatePhase.ACTIVATING)
        recovered_controller = self._controller(
            functional_health=False,
            store=reopened_store,
            slots=reopened_slots,
        )
        recovered_controller.report_boot_health(command_id, healthy=True)
        rolled_back = recovered_controller.report_functional_health(
            command_id, healthy=True
        )
        self.assertEqual(rolled_back.phase, UpdatePhase.ROLLED_BACK)
        self.assertEqual(reopened_slots.current_slot(), "releases/" + "0" * 64)

    def test_restart_resumes_download_from_root_journal(self) -> None:
        payload, _ = self._publish(version="0.1.123")
        command_id, compact = self._command(payload, sequence=39)
        verified = self.verifier.verify(compact)
        self.store.authorize(
            command_id=command_id,
            sequence=verified.sequence,
            compact_jws=compact,
            target_version=str(verified.payload["targetVersion"]),
            bom_digest=str(verified.payload["bomDigest"]),
            command_expires_at=verified.expires_at,
        )
        self.store.transition(
            command_id,
            UpdatePhase.DOWNLOADING,
            allowed_from={UpdatePhase.AUTHORIZED},
        )

        reopened = self._controller(
            store=UpdaterStore(self.state_path, require_root_owner=False),
            slots=ReleaseSlotManager(self.install_root, require_root_owner=False),
        ).recover()

        self.assertEqual(reopened.phase, UpdatePhase.VERIFIED)
        self.assertEqual(reopened.command_id, command_id)
        self.assertEqual(reopened.command_expires_at, verified.expires_at)

    def test_restart_refuses_an_expired_download_from_the_root_journal(self) -> None:
        payload, _ = self._publish(version="0.1.135", release_sequence=15)
        command_expiry = datetime.now(timezone.utc) + timedelta(minutes=5)
        command_id, compact = self._command(
            payload,
            sequence=51,
            expires_at=command_expiry,
        )
        verified = self.verifier.verify(compact)
        self.store.authorize(
            command_id=command_id,
            sequence=verified.sequence,
            compact_jws=compact,
            target_version=str(verified.payload["targetVersion"]),
            bom_digest=str(verified.payload["bomDigest"]),
            command_expires_at=verified.expires_at,
        )
        self.store.transition(
            command_id,
            UpdatePhase.DOWNLOADING,
            allowed_from={UpdatePhase.AUTHORIZED},
        )

        recovered = self._controller(
            store=UpdaterStore(self.state_path, require_root_owner=False),
            slots=ReleaseSlotManager(self.install_root, require_root_owner=False),
            clock=lambda: command_expiry,
        ).recover()

        assert recovered is not None
        self.assertEqual(recovered.phase, UpdatePhase.FAILED)
        self.assertEqual(recovered.error_code, "COMMAND_EXPIRED")

    def test_commit_wall_clock_expiry_rolls_back_before_attestation_consumption(self) -> None:
        payload, _ = self._publish(version="0.1.136", release_sequence=16)
        command_expiry = datetime.now(timezone.utc) + timedelta(minutes=5)
        command_id, command = self._command(
            payload,
            sequence=52,
            expires_at=command_expiry,
        )
        controller = self._controller()
        controller.authorize_and_stage(command)
        controller.activate(command_id)
        controller.report_boot_health(command_id, healthy=True)
        controller.report_functional_health(command_id, healthy=True)
        gate = controller.begin_commit_gate(command_id, peer_pid=4242)
        compact_jws = self._health_attestation(gate)

        expired = self._controller(clock=lambda: command_expiry)
        with self.assertRaises(UpdaterError) as raised:
            expired.commit(
                command_id,
                gate_id=gate.gate_id,
                health_attestation_jws=compact_jws,
                peer_pid=4242,
            )

        self.assertEqual(raised.exception.code, "INVALID_PHASE")
        rolled_back = self.store.get(command_id)
        assert rolled_back is not None
        self.assertEqual(rolled_back.phase, UpdatePhase.ROLLED_BACK)
        self.assertEqual(rolled_back.message, "COMMAND_EXPIRED")
        persisted_gate = self.store.commit_gate(command_id)
        assert persisted_gate is not None
        self.assertIsNone(persisted_gate.consumed_at)

    def test_atomic_commit_rechecks_all_deadlines_after_process_inspection(self) -> None:
        cases = (
            ("command", 5, 120, 5, "COMMAND_EXPIRED"),
            ("health", 300, 5, 30, "COMMIT_TIMEOUT"),
            ("attestation", 300, 120, 5, "HEALTH_ATTESTATION_EXPIRED"),
        )
        for index, (
            name,
            command_ttl,
            commit_ttl,
            attestation_ttl,
            expected_code,
        ) in enumerate(cases, start=1):
            with self.subTest(deadline=name):
                now = [datetime.now(timezone.utc)]
                atomic_store = UpdaterStore(
                    self.state_path,
                    require_root_owner=False,
                    clock=lambda: now[0]
                    .isoformat(timespec="milliseconds")
                    .replace("+00:00", "Z"),
                )
                self.health_verifier.clock = lambda: now[0]
                payload, _ = self._publish(
                    version=f"0.1.{140 + index}",
                    release_sequence=20 + index,
                )
                command_id, command = self._command(
                    payload,
                    sequence=60 + index,
                    expires_at=now[0] + timedelta(seconds=command_ttl),
                )
                controller = self._controller(
                    store=atomic_store,
                    clock=lambda: now[0],
                    commit_timeout_seconds=commit_ttl,
                )
                controller.authorize_and_stage(command)
                controller.activate(command_id)
                controller.report_boot_health(command_id, healthy=True)
                controller.report_functional_health(command_id, healthy=True)
                gate = controller.begin_commit_gate(command_id, peer_pid=4242)
                attestation = self._health_attestation(
                    gate,
                    issued_at=now[0],
                    expires_at=now[0] + timedelta(seconds=attestation_ttl),
                )

                process_checks = 0

                def process_after_delay(state, pid):
                    nonlocal process_checks
                    process_checks += 1
                    identity = CommitProcessIdentity(
                        pid=pid,
                        start_ticks=123456,
                        boot_id="00000000-0000-4000-8000-000000000123",
                        active_slot=self.slots.relative_target(state.candidate_slot),
                    )
                    # The first call in commit validates the peer before JWS
                    # verification. Advance only in the second inspection so
                    # the signed proof is initially valid but stale at the
                    # atomic journal boundary.
                    if process_checks == 2:
                        now[0] += timedelta(seconds=10)
                    return identity

                controller.commit_process_check = process_after_delay
                with self.assertRaises(UpdaterError) as raised:
                    controller.commit(
                        command_id,
                        gate_id=gate.gate_id,
                        health_attestation_jws=attestation,
                        peer_pid=4242,
                    )

                self.assertEqual(raised.exception.code, expected_code)
                rolled_back = atomic_store.get(command_id)
                assert rolled_back is not None
                self.assertEqual(rolled_back.phase, UpdatePhase.ROLLED_BACK)
                self.assertEqual(rolled_back.message, expected_code)
                persisted_gate = atomic_store.commit_gate(command_id)
                assert persisted_gate is not None
                self.assertIsNone(persisted_gate.attestation_id)
                self.assertIsNone(persisted_gate.consumed_at)
                self.assertEqual(atomic_store.current_release_sequence(), 1)

    def test_recovery_cleans_cache_left_after_verified_transition_crash(self) -> None:
        payload, _ = self._publish(version="0.1.134", release_sequence=14)
        command_id, command = self._command(payload, sequence=50)
        verified = self._controller().authorize_and_stage(command)
        self.assertEqual(verified.phase, UpdatePhase.VERIFIED)
        stale_cache = self.download_root / str(payload["bomDigest"])[7:]
        stale_cache.mkdir(mode=0o700)
        (stale_cache / "crash-window").write_bytes(b"stale")

        recovered = self._controller(
            store=UpdaterStore(self.state_path, require_root_owner=False),
            slots=ReleaseSlotManager(self.install_root, require_root_owner=False),
        ).recover()

        assert recovered is not None
        self.assertEqual(recovered.command_id, command_id)
        self.assertEqual(recovered.phase, UpdatePhase.VERIFIED)
        self.assertFalse(stale_cache.exists())

    def test_bad_boot_health_rolls_back(self) -> None:
        payload, _ = self._publish(version="0.1.122")
        command_id, command = self._command(payload, sequence=41)
        controller = self._controller(boot_health=False)
        controller.authorize_and_stage(command)
        controller.activate(command_id)

        rolled_back = controller.report_boot_health(command_id, healthy=True)

        self.assertEqual(rolled_back.phase, UpdatePhase.ROLLED_BACK)
        self.assertEqual(self.slots.current_slot(), "releases/" + "0" * 64)

    def test_watchdog_rolls_back_without_agent_traffic_and_restarts_old_slot(self) -> None:
        payload, _ = self._publish(version="0.1.124")
        command_id, command = self._command(payload, sequence=42)
        now = [datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)]
        restarted: list[str] = []
        controller = self._controller(
            activation_callback=restarted.append,
            clock=lambda: now[0],
            boot_timeout_seconds=1,
        )
        controller.authorize_and_stage(command)
        controller.activate(command_id)

        now[0] += timedelta(seconds=2)
        rolled_back = controller.watchdog_tick()

        self.assertEqual(rolled_back.phase, UpdatePhase.ROLLED_BACK)
        self.assertEqual(self.slots.current_slot(), "releases/" + "0" * 64)
        self.assertEqual(len(restarted), 2)
        self.assertEqual(restarted[-1], "releases/" + "0" * 64)

    def test_verified_activation_lease_expires_without_blocking_future_updates(
        self,
    ) -> None:
        payload, _ = self._publish(version="0.1.128", release_sequence=8)
        _command_id, command = self._command(payload, sequence=46)
        now = [datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)]
        controller = self._controller(
            clock=lambda: now[0],
            activation_timeout_seconds=1,
        )
        verified = controller.authorize_and_stage(command)
        self.assertEqual(verified.phase, UpdatePhase.VERIFIED)

        now[0] += timedelta(seconds=2)
        expired = controller.watchdog_tick()

        self.assertEqual(expired.phase, UpdatePhase.FAILED)
        self.assertEqual(expired.error_code, "ACTIVATION_TIMEOUT")
        self.assertEqual(self.slots.current_slot(), "releases/" + "0" * 64)

    def test_functional_commit_lease_rolls_back_without_agent_traffic(self) -> None:
        payload, _ = self._publish(version="0.1.129", release_sequence=9)
        command_id, command = self._command(payload, sequence=47)
        now = [datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)]
        restarted: list[str] = []
        controller = self._controller(
            activation_callback=restarted.append,
            clock=lambda: now[0],
            commit_timeout_seconds=1,
        )
        controller.authorize_and_stage(command)
        controller.activate(command_id)
        controller.report_boot_health(command_id, healthy=True)
        functional = controller.report_functional_health(command_id, healthy=True)
        self.assertEqual(functional.phase, UpdatePhase.FUNCTIONAL_HEALTHY)
        gate = controller.begin_commit_gate(command_id, peer_pid=4242)
        self.assertEqual(gate.health_deadline, functional.health_deadline)

        now[0] += timedelta(seconds=2)
        rolled_back = controller.watchdog_tick()

        self.assertEqual(rolled_back.phase, UpdatePhase.ROLLED_BACK)
        self.assertEqual(rolled_back.message, "COMMIT_TIMEOUT")
        self.assertEqual(self.slots.current_slot(), "releases/" + "0" * 64)
        self.assertEqual(restarted[-1], "releases/" + "0" * 64)

    def test_functional_probe_cannot_overrun_persisted_deadline(self) -> None:
        payload, _ = self._publish(version="0.1.127")
        command_id, command = self._command(payload, sequence=45)
        now = [datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)]

        def slow_probe(_state: object) -> tuple[bool, str]:
            now[0] += timedelta(seconds=2)
            return True, "probe-finished-too-late"

        controller = self._controller(
            clock=lambda: now[0],
            functional_callback=slow_probe,
            functional_timeout_seconds=1,
        )
        controller.authorize_and_stage(command)
        controller.activate(command_id)
        controller.report_boot_health(command_id, healthy=True)

        rolled_back = controller.report_functional_health(command_id, healthy=True)

        self.assertEqual(rolled_back.phase, UpdatePhase.ROLLED_BACK)
        self.assertEqual(rolled_back.message, "HEALTH_TIMEOUT")

    def test_boot_probe_cannot_overrun_persisted_deadline(self) -> None:
        payload, _ = self._publish(version="0.1.132", release_sequence=12)
        command_id, command = self._command(payload, sequence=48)
        now = [datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)]

        def slow_boot(_state: object) -> tuple[bool, str]:
            now[0] += timedelta(seconds=2)
            return True, "boot-check-finished-too-late"

        controller = self._controller(
            clock=lambda: now[0],
            boot_callback=slow_boot,
            boot_timeout_seconds=1,
        )
        controller.authorize_and_stage(command)
        controller.activate(command_id)

        rolled_back = controller.report_boot_health(command_id, healthy=True)

        self.assertEqual(rolled_back.phase, UpdatePhase.ROLLED_BACK)
        self.assertEqual(rolled_back.message, "HEALTH_TIMEOUT")

    def test_rollback_restart_failure_safe_stops_and_is_terminal(self) -> None:
        payload, _ = self._publish(version="0.1.125")
        command_id, command = self._command(payload, sequence=43)
        restarted: list[str] = []
        safe_stops: list[bool] = []

        def restart(slot: str) -> None:
            restarted.append(slot)
            if slot == "releases/" + "0" * 64:
                raise RuntimeError("old service failed to restart")

        controller = self._controller(
            boot_health=False,
            activation_callback=restart,
            safe_stop_callback=lambda: safe_stops.append(True),
        )
        controller.authorize_and_stage(command)
        controller.activate(command_id)

        failed = controller.report_boot_health(command_id, healthy=True)

        self.assertEqual(failed.phase, UpdatePhase.ROLLBACK_FAILED)
        self.assertEqual(failed.error_code, "ROLLBACK_FAILED")
        self.assertEqual(safe_stops, [True])
        self.assertEqual(self.slots.current_slot(), "releases/" + "0" * 64)

    def test_power_loss_after_rollback_symlink_is_idempotently_recovered(self) -> None:
        payload, _ = self._publish(version="0.1.126")
        command_id, command = self._command(payload, sequence=44)
        controller = self._controller()
        controller.authorize_and_stage(command)
        activated = controller.activate(command_id)
        rolling = self.store.transition(
            command_id,
            UpdatePhase.ROLLING_BACK,
            allowed_from={UpdatePhase.ACTIVATING},
            error_code="HEALTH_GATE_FAILED",
            message="simulated power loss",
        )
        assert rolling.previous_slot is not None
        self.slots.restore(rolling.previous_slot)

        reopened = self._controller(
            store=UpdaterStore(self.state_path, require_root_owner=False),
            slots=ReleaseSlotManager(self.install_root, require_root_owner=False),
        ).recover()

        self.assertEqual(reopened.phase, UpdatePhase.ROLLED_BACK)
        self.assertEqual(self.slots.current_slot(), "releases/" + "0" * 64)
        self.assertNotEqual(activated.candidate_slot, str(self.install_root / self.slots.current_slot()))

    def test_v1_journal_migrates_release_evidence_columns(self) -> None:
        legacy_path = self.root / "legacy" / "updater.sqlite3"
        legacy_path.parent.mkdir()
        connection = sqlite3.connect(legacy_path)
        connection.execute(
            """
            CREATE TABLE updater_command (
                command_id TEXT PRIMARY KEY,
                sequence INTEGER NOT NULL UNIQUE,
                compact_jws TEXT NOT NULL,
                compact_jws_sha256 TEXT NOT NULL,
                target_version TEXT NOT NULL,
                bom_digest TEXT NOT NULL,
                phase TEXT NOT NULL,
                candidate_slot TEXT,
                previous_slot TEXT,
                release_sequence INTEGER,
                health_deadline TEXT,
                error_code TEXT,
                message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute("PRAGMA user_version = 1")
        connection.commit()
        connection.close()
        legacy_path.chmod(0o600)

        UpdaterStore(legacy_path, require_root_owner=False)

        connection = sqlite3.connect(legacy_path)
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(updater_command)")
        }
        gate_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(updater_commit_gate)")
        }
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        connection.close()
        self.assertEqual(version, 4)
        self.assertIn("updater_commit_gate", tables)
        self.assertIn("command_expires_at", gate_columns)
        self.assertTrue(
            {
                "artifact_digest",
                "component_sha",
                "config_schema",
                "bom_verification_status",
                "previous_version",
                "command_expires_at",
            }.issubset(columns)
        )

    def test_protocol_rejects_arbitrary_url_path_shell_and_unknown_fields(self) -> None:
        protocol = UpdaterProtocol(self._controller())
        for dangerous in (
            {"url": "https://attacker.invalid/a"},
            {"path": "/etc/shadow"},
            {"shell": "id"},
        ):
            with self.subTest(dangerous=dangerous), self.assertRaises(
                UpdaterSecurityError
            ):
                protocol.dispatch(
                    {
                        "schemaVersion": 1,
                        "operation": "STATUS",
                        **dangerous,
                    }
                )

    def test_journal_is_private_and_docker_profile_without_helper_is_fail_closed(self) -> None:
        self.assertEqual(self.state_path.stat().st_mode & 0o777, 0o600)
        docker_binding = DeviceBinding(
            **{
                **self.binding.__dict__,
                "docker_required": True,
            }
        )
        controller = UpdaterController(
            store=self.store,
            slots=self.slots,
            repository=ContentAddressedReleaseRepository(
                base_url=self.remote.as_uri(),
                download_root=self.download_root,
                require_root_owner=False,
                allow_file_url=True,
                disk_reserve_bytes=0,
            ),
            command_verifier=self.verifier,
            release_keyring=self.release_keyring,
            binding=docker_binding,
            updater_version="0.1.0",
            activation_callback=lambda _slot: None,
            boot_health_check=lambda _state: (True, "ok"),
            functional_health_check=lambda _state: (True, "ok"),
            commit_process_check=lambda state, pid: CommitProcessIdentity(
                pid=pid,
                start_ticks=123456,
                boot_id="00000000-0000-4000-8000-000000000123",
                active_slot=self.slots.relative_target(state.candidate_slot),
            ),
            health_attestation_verifier=self.health_verifier,
            rollback_boot_health_check=lambda _slot: (True, "ok"),
            safe_stop_callback=lambda: None,
        )
        self.assertFalse(controller.capability_available)


if __name__ == "__main__":
    unittest.main()
