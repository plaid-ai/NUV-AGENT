from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/iq9075-candidate-evidence.yml"
RUNBOOK = ROOT / "packaging/release/v0.1.121-release-runbook.md"


class Iq9075CandidateEvidenceWorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = WORKFLOW.read_text(encoding="utf-8")
        cls.header, cls.jobs = cls.workflow.split("jobs:", maxsplit=1)
        cls.build, remaining = cls.jobs.split("  sign:", maxsplit=1)
        cls.sign, cls.stage = remaining.split("  stage:", maxsplit=1)

    def test_is_manual_current_main_only_and_read_only(self) -> None:
        self.assertIn("workflow_dispatch:", self.header)
        self.assertNotIn("pull_request:", self.header)
        self.assertNotIn("push:", self.header)
        self.assertNotIn("workflow_run:", self.header)
        self.assertNotIn("contents: write", self.workflow)
        self.assertIn("github.ref == 'refs/heads/main'", self.build)
        self.assertIn('[ "$GITHUB_REF" = refs/heads/main ]', self.sign)
        self.assertIn('[ "$GITHUB_SHA" = "$REQUESTED_COMPONENT_SHA" ]', self.workflow)
        self.assertGreaterEqual(
            self.workflow.count(
                '[ "$EXECUTING_WORKFLOW_SHA" = "$REQUESTED_COMPONENT_SHA" ]'
            ),
            2,
        )
        self.assertGreaterEqual(self.workflow.count("git/ref/heads/main"), 2)

    def test_secretless_native_arm_build_is_separate_from_signing(self) -> None:
        self.assertIn("runs-on: ubuntu-24.04-arm", self.build)
        self.assertIn("build-agent-bundle.sh", self.build)
        self.assertIn("stamp-build-info.py", self.build)
        self.assertIn("SOURCE_DATE_EPOCH", self.build)
        self.assertNotIn("IQ9075_RELEASE_SIGNING_PRIVATE_KEY", self.build)
        self.assertNotIn("environment:", self.build)
        self.assertIn("environment: iq9075-release", self.sign)
        self.assertEqual(
            self.sign.count("secrets.IQ9075_RELEASE_SIGNING_PRIVATE_KEY"), 1
        )
        for unrelated_secret in (
            "APT_GPG_PRIVATE_KEY",
            "HOMEBREW_TAP_TOKEN",
        ):
            self.assertNotIn(unrelated_secret, self.workflow)
        self.assertNotIn("GCP_SA_KEY", self.sign)
        self.assertNotIn("GCP_PROJECT_ID", self.sign)
        self.assertEqual(self.stage.count("secrets.GCP_SA_KEY"), 1)
        self.assertEqual(self.stage.count("secrets.GCP_PROJECT_ID"), 1)

    def test_signing_is_bound_to_policy_key_and_downloaded_artifact(self) -> None:
        revalidate = self.sign.index(
            "Revalidate source and artifact before signer access"
        )
        signer = self.sign.index("Sign exact canonical BOM")
        self.assertLess(revalidate, signer)
        self.assertIn("generate-release-bom.py", self.sign)
        self.assertIn("--schema-version 2", self.sign)
        self.assertIn("--signing-private-key-env", self.sign)
        self.assertIn("trusted-release-keyrings/iq9075-dev.json", self.sign)
        self.assertIn("load_signed_release_bom", self.sign)
        self.assertIn("verify_release_artifact", self.sign)
        self.assertIn("candidate-build-manifest.json", self.sign)
        self.assertIn("sha256sum", self.sign)

    def test_workflow_stages_only_content_addressed_objects(self) -> None:
        forbidden = (
            "git push",
            "git tag",
            "gh release",
            "publish-immutable",
            "update-homebrew",
            "generate-release-promotion",
            "sequence-reservation",
            "aptly",
            "contents: write",
        )
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, self.workflow)
        self.assertIn("OTA_CONTENT_ONLY=true", self.stage)
        self.assertIn("SKIP_APT_PUBLISH=true", self.stage)
        self.assertIn("packaging/apt/publish-gcs.sh", self.stage)
        self.assertIn("releases/by-bom-sha256/", self.sign)
        self.assertIn("retention-days: 2", self.workflow)
        self.assertIn("cancel-in-progress: false", self.workflow)

    def test_every_external_action_is_full_sha_pinned(self) -> None:
        uses = re.findall(r"^\s+uses:\s+([^\s]+)", self.workflow, flags=re.MULTILINE)
        self.assertTrue(uses)
        for action in uses:
            with self.subTest(action=action):
                self.assertRegex(action, r"^[^@]+@[0-9a-f]{40}$")

    def test_runbook_uses_the_signed_content_only_candidate(self) -> None:
        runbook = RUNBOOK.read_text(encoding="utf-8")
        self.assertIn("gh workflow run iq9075-candidate-evidence.yml", runbook)
        self.assertIn("iq9075-signed-evidence-${candidate_build_run_id}", runbook)
        self.assertIn("releases/by-bom-sha256/<bomhex>/", runbook)
        self.assertIn("CANDIDATE_BOM_SIGNATURE", runbook)
        self.assertIn("load_signed_release_bom", runbook)
        self.assertIn("verify_release_artifact", runbook)

    def test_content_only_staging_never_creates_version_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            artifact = (
                root
                / "nuv-agent_0.1.121_iq9075-aarch64.agent-bundle.tar.gz"
            )
            artifact.write_bytes(b"exact candidate evidence bundle")
            private_key = Ed25519PrivateKey.generate()
            private_raw = private_key.private_bytes(
                serialization.Encoding.Raw,
                serialization.PrivateFormat.Raw,
                serialization.NoEncryption(),
            )
            public_der = private_key.public_key().public_bytes(
                serialization.Encoding.DER,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            keyring = root / "keyring.json"
            keyring.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "trustDomain": "test-iq9075",
                        "keys": {
                            "test-release": base64.b64encode(public_der).decode(
                                "ascii"
                            )
                        },
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            bom = root / "release-bom.json"
            signature = root / "release-bom.json.sig"
            generation_environment = {
                **os.environ,
                "TEST_RELEASE_PRIVATE_KEY": base64.b64encode(private_raw).decode(
                    "ascii"
                ),
            }
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "packaging/release/generate-release-bom.py"),
                    "--schema-version",
                    "2",
                    "--bom-id",
                    "nuv-agent-0.1.121-iq9075-aarch64",
                    "--version",
                    "0.1.121",
                    "--component-sha",
                    "a" * 40,
                    "--config-schema",
                    "12",
                    "--release-sequence",
                    "2",
                    "--min-updater-version",
                    "0.2.0",
                    "--target",
                    "IQ9075_DEV:iq9075_dev:QCS9075-EVK:aarch64",
                    "--artifact",
                    str(artifact),
                    "--artifact-kind",
                    "agent-bundle",
                    "--built-at",
                    "2026-09-03T12:00:00+09:00",
                    "--output",
                    str(bom),
                    "--signature-output",
                    str(signature),
                    "--signing-key-id",
                    "test-release",
                    "--signing-private-key-env",
                    "TEST_RELEASE_PRIVATE_KEY",
                ],
                check=True,
                capture_output=True,
                env=generation_environment,
            )
            bom_digest = json.loads(bom.read_text(encoding="utf-8"))[
                "bomDigest"
            ][7:]
            binary = root / "bin"
            remote = root / "remote"
            public = root / "public"
            binary.mkdir()
            remote.mkdir()
            fake_gcloud = binary / "gcloud"
            fake_gcloud.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
[ "$1" = storage ]
args=("$@")
remote_arg="${args[${#args[@]}-1]}"
relative="${remote_arg#gs://test-bucket/}"
target="$FAKE_GCLOUD_REMOTE/$relative"
case "$2" in
  cp)
    [ ! -e "$target" ] || exit 1
    mkdir -p "$(dirname "$target")"
    cp "${args[${#args[@]}-2]}" "$target"
    ;;
  cat) cat "$target" ;;
  *) exit 2 ;;
esac
""",
                encoding="utf-8",
            )
            fake_gcloud.chmod(0o755)
            environment = {
                **os.environ,
                "PATH": f"{binary}:{Path(sys.executable).parent}:{os.environ['PATH']}",
                "FAKE_GCLOUD_REMOTE": str(remote),
                "VERSION": "0.1.121",
                "BUCKET": "test-bucket",
                "SKIP_APT_PUBLISH": "true",
                "OTA_CONTENT_ONLY": "true",
                "APT_PUBLIC_DIR": str(public),
                "APT_RUNTIME_ROOT": str(root / "runtime"),
                "RELEASE_KEYRING_PATH": str(keyring),
                "RELEASE_TRUST_DOMAIN": "test-iq9075",
            }
            command = [
                str(ROOT / "packaging/apt/publish-gcs.sh"),
                str(artifact),
                str(bom),
                str(signature),
                str(artifact),
            ]
            for _ in range(2):
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    env=environment,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
            objects = {
                path.relative_to(remote).as_posix()
                for path in remote.rglob("*")
                if path.is_file()
            }
            prefix = f"releases/by-bom-sha256/{bom_digest}"
            self.assertEqual(
                objects,
                {
                    f"{prefix}/{artifact.name}",
                    f"{prefix}/release-bom.json",
                    f"{prefix}/release-bom.json.sig",
                },
            )
            self.assertFalse((remote / "releases/0.1.121").exists())


if __name__ == "__main__":
    unittest.main()
