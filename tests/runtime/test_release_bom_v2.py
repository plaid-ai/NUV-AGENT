from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from nuvion_app.runtime.release_bom import (
    ReleaseBomValidationError,
    ReleaseKeyring,
    ReleaseTarget,
    assert_minimum_updater_version,
    assert_release_compatible,
    assert_release_sequence_allowed,
    build_release_bom_signature,
    build_release_bom_v2_payload,
    canonical_release_bom_json,
    canonical_release_bom_signature_json,
    compute_bom_digest,
    load_signed_release_bom,
    verify_signed_release_bom,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "tests" / "runtime" / "fixtures" / "release-bom-v2-ed25519.json"


class ReleaseBomV2Test(unittest.TestCase):
    def setUp(self) -> None:
        self.private_key = Ed25519PrivateKey.generate()
        public_key = self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        self.keyring = ReleaseKeyring({"release-prod-2026": public_key})

    def _build(
        self,
        root: Path,
        *,
        release_sequence: int = 42,
    ) -> tuple[dict, dict]:
        artifact = root / "nuv-agent_0.2.0_arm64.tar.zst"
        artifact.write_bytes(b"self-contained-agent-slot-bundle")
        payload = build_release_bom_v2_payload(
            bom_id="nuv-agent-0.2.0-arm64",
            release_sequence=release_sequence,
            agent_version="0.2.0",
            component_sha="a" * 40,
            config_schema="12",
            min_updater_version="1.4.0",
            targets=[
                ReleaseTarget(
                    product_model="NUVION_ULTRA",
                    platform_profile="jetson_orin_nx",
                    hardware_revision="orin-nx-rev-a",
                    architecture="aarch64",
                ),
                ReleaseTarget(
                    product_model="NUVION",
                    platform_profile="rpi5_deepx_dx_m1",
                    hardware_revision="rpi5-dxm1-rev-a",
                    architecture="aarch64",
                ),
            ],
            artifact_path=artifact,
            artifact_kind="agent-bundle",
            built_at="2026-09-01T03:00:00Z",
        )
        signature = build_release_bom_signature(
            payload,
            key_id="release-prod-2026",
            private_key=self.private_key,
        )
        return payload, signature

    def test_signed_v2_load_authenticates_publisher_and_exposes_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload, signature = self._build(root)
            bom_path = root / "release-bom.json"
            signature_path = root / "release-bom.json.sig"
            bom_path.write_text(canonical_release_bom_json(payload), encoding="utf-8")
            signature_path.write_text(
                canonical_release_bom_signature_json(signature), encoding="utf-8"
            )

            verified = load_signed_release_bom(
                bom_path,
                signature_path,
                release_keyring=self.keyring,
                expected_bom_digest=payload["bomDigest"],
            )

        self.assertEqual(verified.schema_version, 2)
        self.assertEqual(verified.release_sequence, 42)
        self.assertEqual(verified.min_updater_version, "1.4.0")
        self.assertEqual(verified.artifact_kind, "agent-bundle")
        self.assertIsNone(verified.updater_version)
        self.assertEqual(verified.publisher_key_id, "release-prod-2026")
        self.assertEqual(
            verified.platform_profiles,
            ("jetson_orin_nx", "rpi5_deepx_dx_m1"),
        )
        self.assertNotIn("updaterVersion", verified.to_telemetry())
        self.assertEqual(verified.to_telemetry()["releaseSequence"], 42)

    def test_content_and_detached_signature_tampering_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload, signature = self._build(Path(tmp))

        tampered_payload = copy.deepcopy(payload)
        tampered_payload["agentVersion"] = "0.2.1"
        tampered_payload["bomDigest"] = (
            f"sha256:{compute_bom_digest(tampered_payload)}"
        )
        with self.assertRaisesRegex(
            ReleaseBomValidationError, "publisher signature verification failed"
        ):
            verify_signed_release_bom(
                tampered_payload,
                signature,
                release_keyring=self.keyring,
            )

        tampered_signature = copy.deepcopy(signature)
        raw_signature = bytearray(base64.b64decode(signature["signature"]))
        raw_signature[0] ^= 1
        tampered_signature["signature"] = base64.b64encode(raw_signature).decode(
            "ascii"
        )
        with self.assertRaisesRegex(
            ReleaseBomValidationError, "publisher signature verification failed"
        ):
            verify_signed_release_bom(
                payload,
                tampered_signature,
                release_keyring=self.keyring,
            )

    def test_unknown_signer_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload, signature = self._build(Path(tmp))
        attacker = Ed25519PrivateKey.generate()
        attacker_public = attacker.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        with self.assertRaisesRegex(ReleaseBomValidationError, "not trusted"):
            verify_signed_release_bom(
                payload,
                signature,
                release_keyring=ReleaseKeyring({"another-key": attacker_public}),
            )

    def test_exact_product_profile_hardware_and_architecture_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload, signature = self._build(Path(tmp))
        verified = verify_signed_release_bom(
            payload, signature, release_keyring=self.keyring
        )

        matched = assert_release_compatible(
            verified,
            product_model="NUVION",
            platform_profile="rpi5_deepx_dx_m1",
            hardware_revision="rpi5-dxm1-rev-a",
            architecture="aarch64",
            current_updater_version="1.4.0",
        )
        self.assertEqual(matched.product_model, "NUVION")

        with self.assertRaises(ReleaseBomValidationError):
            assert_release_compatible(
                verified,
                product_model="NUVION",
                platform_profile="ventuno_q",
                hardware_revision="rpi5-dxm1-rev-a",
                architecture="aarch64",
                current_updater_version="1.4.0",
            )
        with self.assertRaisesRegex(ReleaseBomValidationError, "no exact target"):
            assert_release_compatible(
                verified,
                product_model="NUVION",
                platform_profile="rpi5_deepx_dx_m1",
                hardware_revision="rpi5-dxm1-rev-b",
                architecture="aarch64",
                current_updater_version="1.4.0",
            )
        with self.assertRaisesRegex(ReleaseBomValidationError, "no exact target"):
            assert_release_compatible(
                verified,
                product_model="NUVION",
                platform_profile="rpi5_deepx_dx_m1",
                hardware_revision="rpi5-dxm1-rev-a",
                architecture="arm64",
                current_updater_version="1.4.0",
            )

    def test_minimum_updater_and_sequence_prevent_unsafe_activation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload, signature = self._build(Path(tmp))
        verified = verify_signed_release_bom(
            payload, signature, release_keyring=self.keyring
        )

        assert_minimum_updater_version(
            verified, current_updater_version="1.4.0+host.1"
        )
        with self.assertRaisesRegex(ReleaseBomValidationError, "below"):
            assert_minimum_updater_version(
                verified, current_updater_version="1.3.99"
            )

        assert_release_sequence_allowed(
            verified, current_release_sequence=41
        )
        assert_release_sequence_allowed(
            verified,
            current_release_sequence=42,
            current_bom_digest=payload["bomDigest"],
        )
        with self.assertRaisesRegex(ReleaseBomValidationError, "digest is required"):
            assert_release_sequence_allowed(
                verified,
                current_release_sequence=42,
            )
        with self.assertRaisesRegex(ReleaseBomValidationError, "downgrade"):
            assert_release_sequence_allowed(
                verified, current_release_sequence=43
            )
        with self.assertRaisesRegex(ReleaseBomValidationError, "cannot reuse"):
            assert_release_sequence_allowed(
                verified,
                current_release_sequence=42,
                current_bom_digest="b" * 64,
            )

    def test_v2_generator_reads_private_key_from_environment_and_writes_sidecar(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "nuv-agent_0.2.0_arm64.tar.zst"
            artifact.write_bytes(b"self-contained-agent-slot-bundle")
            output = root / "release-bom.json"
            private_material = self.private_key.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption(),
            )
            encoded_private_key = base64.b64encode(private_material).decode("ascii")
            environment = os.environ.copy()
            environment["NUV_TEST_RELEASE_SIGNING_KEY"] = encoded_private_key
            result = subprocess.run(
                [
                    sys.executable,
                    str(
                        ROOT
                        / "packaging"
                        / "release"
                        / "generate-release-bom.py"
                    ),
                    "--schema-version",
                    "2",
                    "--bom-id",
                    "nuv-agent-0.2.0-arm64",
                    "--version",
                    "0.2.0",
                    "--component-sha",
                    "a" * 40,
                    "--config-schema",
                    "12",
                    "--release-sequence",
                    "42",
                    "--min-updater-version",
                    "1.4.0",
                    "--target",
                    "NUVION:rpi5_deepx_dx_m1:rpi5-dxm1-rev-a:aarch64",
                    "--artifact",
                    str(artifact),
                    "--artifact-kind",
                    "agent-bundle",
                    "--built-at",
                    "2026-09-01T03:00:00Z",
                    "--signing-key-id",
                    "release-prod-2026",
                    "--signing-private-key-env",
                    "NUV_TEST_RELEASE_SIGNING_KEY",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn(encoded_private_key, result.stdout)
            signature_path = Path(f"{output}.sig")
            self.assertTrue(signature_path.is_file())
            verified = load_signed_release_bom(
                output,
                signature_path,
                release_keyring=self.keyring,
            )

        self.assertEqual(verified.release_sequence, 42)

    def test_cross_language_signature_fixture_locks_byte_contract(self) -> None:
        fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        bom = fixture["bom"]
        expected = fixture["expected"]
        canonical_bom = json.dumps(
            bom,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        prefix = bytes.fromhex(expected["domainSeparationPrefixHex"])
        signing_input = prefix + canonical_bom

        self.assertEqual(
            hashlib.sha256(canonical_bom).hexdigest(),
            expected["canonicalBomSha256"],
        )
        self.assertEqual(
            hashlib.sha256(signing_input).hexdigest(),
            expected["signingInputSha256"],
        )
        public_key = fixture["publicKey"]
        self.assertEqual(public_key["encoding"], "raw-base64")
        public_material = base64.b64decode(public_key["value"], validate=True)
        self.assertEqual(len(public_material), 32)
        self.assertEqual(
            base64.b64encode(public_material).decode("ascii"),
            public_key["value"],
        )
        verified = verify_signed_release_bom(
            bom,
            fixture["signatureEnvelope"],
            release_keyring=ReleaseKeyring(
                {public_key["keyId"]: public_material}
            ),
        )
        self.assertEqual(verified.publisher_key_id, "fixture-rfc8032-1")


if __name__ == "__main__":
    unittest.main()
