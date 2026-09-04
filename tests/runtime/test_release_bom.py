from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from nuvion_app.runtime.release_bom import (
    ReleaseBomValidationError,
    build_release_bom_payload,
    canonical_release_bom_json,
    compute_bom_digest,
    load_release_bom,
    verify_release_artifact,
    verify_release_bom,
)

ROOT = Path(__file__).resolve().parents[2]


class ReleaseBomTest(unittest.TestCase):
    def _build(self, root: Path) -> dict:
        artifact = root / "nuv-agent_0.1.113_arm64.deb"
        artifact.write_bytes(b"immutable-agent-artifact")
        return build_release_bom_payload(
            bom_id="nuv-agent-0.1.113-arm64",
            agent_version="0.1.113",
            component_sha="a" * 40,
            config_schema="11",
            updater_version="0.1.113",
            platform_profiles=[
                "ventuno_q",
                "rpi5_deepx_dx_m1",
                "jetson_orin_nx",
                "iq9075_dev",
            ],
            artifact_path=artifact,
            artifact_kind="deb",
            built_at="2026-09-01T03:00:00Z",
        )

    def test_generated_bom_is_canonical_content_addressed_and_loadable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self._build(root)
            path = root / "release-bom.json"
            document = canonical_release_bom_json(payload)
            self.assertEqual(
                document,
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
            )
            path.write_text(document, encoding="utf-8")

            verified = load_release_bom(path, expected_bom_digest=payload["bomDigest"])

        self.assertEqual(verified.bom_id, "nuv-agent-0.1.113-arm64")
        self.assertEqual(verified.component_sha, "a" * 40)
        self.assertEqual(
            verified.platform_profiles,
            ("iq9075_dev", "jetson_orin_nx", "rpi5_deepx_dx_m1", "ventuno_q"),
        )
        self.assertEqual(verified.to_telemetry()["bomDigest"], payload["bomDigest"])
        self.assertTrue(verified.to_telemetry()["artifactDigest"].startswith("sha256:"))

    def test_artifact_verification_binds_name_digest_and_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = self._build(root)
            verified = verify_release_bom(payload)
            artifact = root / "nuv-agent_0.1.113_arm64.deb"
            verify_release_artifact(verified, artifact)

            artifact.write_bytes(b"changed-release-artifact")
            with self.assertRaisesRegex(ReleaseBomValidationError, "digest or size"):
                verify_release_artifact(verified, artifact)

    def test_artifact_symlink_and_unsafe_names_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "real.deb"
            artifact.write_bytes(b"release")
            link = root / "linked.deb"
            link.symlink_to(artifact)
            with self.assertRaisesRegex(ReleaseBomValidationError, "symbolic link"):
                build_release_bom_payload(
                    bom_id="nuv-agent-0.1.113-arm64",
                    agent_version="0.1.113",
                    component_sha="a" * 40,
                    config_schema="11",
                    updater_version="0.1.113",
                    platform_profiles=["rpi5_deepx_dx_m1"],
                    artifact_path=link,
                    artifact_kind="deb",
                    built_at="2026-09-01T03:00:00Z",
                )

            payload = self._build(root)
            payload["artifact"]["name"] = "unsafe name.deb"
            payload["bomDigest"] = f"sha256:{compute_bom_digest(payload)}"
            with self.assertRaisesRegex(ReleaseBomValidationError, "safe basename"):
                verify_release_bom(payload)

    def test_content_or_trusted_digest_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._build(Path(tmp))
        payload["agentVersion"] = "0.1.114"

        with self.assertRaisesRegex(ReleaseBomValidationError, "bomDigest"):
            verify_release_bom(payload)
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._build(Path(tmp))
            with self.assertRaisesRegex(
                ReleaseBomValidationError, "trusted expected digest"
            ):
                verify_release_bom(payload, expected_bom_digest="b" * 64)

    def test_duplicate_json_fields_and_unknown_fields_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            duplicate = root / "duplicate.json"
            duplicate.write_text(
                '{"schemaVersion":1,"schemaVersion":1}', encoding="utf-8"
            )
            with self.assertRaises(ReleaseBomValidationError):
                load_release_bom(duplicate)

            payload = self._build(root)
            payload["mutableChannel"] = "latest"
            with self.assertRaisesRegex(ReleaseBomValidationError, "fields"):
                verify_release_bom(payload)

            payload = self._build(root)
            payload["schemaVersion"] = True
            payload["bomDigest"] = f"sha256:{compute_bom_digest(payload)}"
            with self.assertRaisesRegex(ReleaseBomValidationError, "schemaVersion"):
                verify_release_bom(payload)

    def test_bom_symlink_and_oversize_input_are_rejected_before_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "release-bom.json"
            target.write_text("{}", encoding="utf-8")
            link = root / "release-bom-link.json"
            link.symlink_to(target)
            with self.assertRaisesRegex(ReleaseBomValidationError, "symbolic link"):
                load_release_bom(link)

            target.write_bytes(b"x" * (1024 * 1024 + 1))
            with self.assertRaisesRegex(ReleaseBomValidationError, "size limit"):
                load_release_bom(target)

            target.write_text("[" * 2000 + "0" + "]" * 2000, encoding="utf-8")
            with self.assertRaises(ReleaseBomValidationError):
                load_release_bom(target)

    def test_artifact_and_profile_constraints_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._build(Path(tmp))
        payload["artifact"]["name"] = "../agent.deb"
        payload["bomDigest"] = "sha256:" + "0" * 64
        with self.assertRaises(ReleaseBomValidationError):
            verify_release_bom(payload)

    def test_generator_is_atomic_and_refuses_different_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "nuv-agent_0.1.113_arm64.deb"
            artifact.write_bytes(b"immutable-agent-artifact")
            output = root / "release-bom.json"
            base_command = [
                sys.executable,
                str(ROOT / "packaging" / "release" / "generate-release-bom.py"),
                "--bom-id",
                "nuv-agent-0.1.113-arm64",
                "--version",
                "0.1.113",
                "--component-sha",
                "a" * 40,
                "--config-schema",
                "11",
                "--updater-version",
                "0.1.113",
                "--platform-profile",
                "rpi5_deepx_dx_m1",
                "--artifact",
                str(artifact),
                "--artifact-kind",
                "deb",
                "--output",
                str(output),
            ]

            first = subprocess.run(
                [*base_command, "--built-at", "2026-09-01T03:00:00Z"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            original = output.read_bytes()

            identical = subprocess.run(
                [*base_command, "--built-at", "2026-09-01T03:00:00Z"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(identical.returncode, 0, identical.stderr)

            different = subprocess.run(
                [*base_command, "--built-at", "2026-09-01T03:00:01Z"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(different.returncode, 0)
            self.assertIn("refusing to overwrite", different.stderr)
            self.assertEqual(output.read_bytes(), original)

            linked_artifact = root / "linked-agent.deb"
            linked_artifact.symlink_to(artifact)
            linked_output = root / "linked-release-bom.json"
            linked_command = list(base_command)
            linked_command[linked_command.index(str(artifact))] = str(linked_artifact)
            linked_command[linked_command.index(str(output))] = str(linked_output)
            linked = subprocess.run(
                [*linked_command, "--built-at", "2026-09-01T03:00:00Z"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(linked.returncode, 0)
            self.assertIn("symbolic link", linked.stderr)
            self.assertFalse(linked_output.exists())


if __name__ == "__main__":
    unittest.main()
