from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/iq9075-candidate-evidence.yml"
TRUSTED_WORKFLOW = (
    ROOT / ".github/workflows/iq9075-candidate-trusted-publish.yml"
)
RUNBOOK = ROOT / "packaging/release/v0.1.121-release-runbook.md"
TRUSTED_PUBLISHER_SHA = "59a073eaecbdbfdc79b8be728ac1dc778947410d"


class Iq9075CandidateEvidenceWorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = WORKFLOW.read_text(encoding="utf-8")
        cls.header, cls.jobs = cls.workflow.split("jobs:", maxsplit=1)
        _, build_and_trusted_call = cls.jobs.split("  build:", maxsplit=1)
        cls.build, cls.trusted_call = build_and_trusted_call.split(
            "  trusted-sign-and-stage:", maxsplit=1
        )
        cls.trusted_workflow = TRUSTED_WORKFLOW.read_text(encoding="utf-8")
        cls.trusted_header, trusted_jobs = cls.trusted_workflow.split(
            "jobs:", maxsplit=1
        )
        cls.sign, cls.stage = trusted_jobs.split("  stage:", maxsplit=1)
        verifier_marker = '$GITHUB_RUN_ID" "$GITHUB_RUN_ATTEMPT" <<\'PY\'\n'
        verifier_scripts: list[str] = []
        offset = 0
        while True:
            marker = cls.trusted_workflow.find(verifier_marker, offset)
            if marker < 0:
                break
            start = marker + len(verifier_marker)
            end = cls.trusted_workflow.index("\n          PY", start)
            verifier_scripts.append(
                textwrap.dedent(cls.trusted_workflow[start:end])
            )
            offset = end + 1
        if len(verifier_scripts) != 2:
            raise AssertionError("expected one OIDC verifier in each privileged job")
        if verifier_scripts[0] != verifier_scripts[1]:
            raise AssertionError("sign and stage OIDC verifiers diverged")
        cls.oidc_verifier = verifier_scripts[0]

    def test_is_manual_current_main_only_and_read_only(self) -> None:
        self.assertIn("workflow_dispatch:", self.header)
        self.assertNotIn("pull_request:", self.header)
        self.assertNotIn("push:", self.header)
        self.assertNotIn("workflow_run:", self.header)
        self.assertIn("workflow_call:", self.trusted_header)
        self.assertNotIn("workflow_dispatch:", self.trusted_header)
        self.assertNotIn("contents: write", self.workflow + self.trusted_workflow)
        self.assertIn("github.ref == 'refs/heads/main'", self.build)
        self.assertIn('[ "$GITHUB_REF" = refs/heads/main ]', self.sign)
        self.assertGreaterEqual(
            self.trusted_workflow.count(
                '[ "$GITHUB_SHA" = "$REQUESTED_COMPONENT_SHA" ]'
            ), 2
        )
        self.assertIn(
            '[ "$EXECUTING_WORKFLOW_SHA" = "$REQUESTED_COMPONENT_SHA" ]',
            self.build,
        )
        self.assertGreaterEqual(
            self.trusted_workflow.count("git/ref/heads/main"), 2
        )

    def test_dispatch_uses_only_the_literal_trusted_publisher_pin(self) -> None:
        self.assertNotIn("trusted-publisher-pin-required:", self.workflow)
        self.assertNotIn("needs: trusted-publisher-pin-required", self.build)
        reusable_call = re.search(
            r"uses: plaid-ai/NUV-AGENT/\.github/workflows/"
            r"iq9075-candidate-trusted-publish\.yml@([0-9a-f]{40})",
            self.trusted_call,
        )
        self.assertIsNotNone(reusable_call)
        assert reusable_call is not None
        self.assertEqual(reusable_call.group(1), TRUSTED_PUBLISHER_SHA)
        self.assertEqual(self.workflow.count(TRUSTED_PUBLISHER_SHA), 2)
        self.assertIn(
            f"trusted_workflow_sha: {TRUSTED_PUBLISHER_SHA}",
            self.trusted_call,
        )
        self.assertIn("needs: build", self.trusted_call)
        self.assertIn("id-token: write", self.trusted_call)
        self.assertNotIn("runs-on:", self.trusted_call)
        self.assertNotIn("steps:", self.trusted_call)
        self.assertNotIn("environment:", self.workflow)
        sentinel = "${{ secrets.GITHUB_TOKEN }}"
        self.assertEqual(self.workflow.count(sentinel), 3)
        caller_without_sentinels = self.workflow.replace(sentinel, "")
        self.assertNotRegex(
            caller_without_sentinels,
            r"\$\{\{[^}]*\bsecrets\b",
        )
        self.assertEqual(
            re.findall(
                r"\$\{\{\s*secrets\.([A-Z0-9_]+)\s*\}\}",
                self.workflow,
            ),
            ["GITHUB_TOKEN", "GITHUB_TOKEN", "GITHUB_TOKEN"],
        )
        for secret_name in (
            "IQ9075_RELEASE_SIGNING_PRIVATE_KEY",
            "GCP_PROJECT_ID",
            "GCP_SA_KEY",
        ):
            self.assertIn(
                f"{secret_name}: ${{{{ secrets.GITHUB_TOKEN }}}}",
                self.trusted_call,
            )
        bindings = self.trusted_call.split("    secrets:\n", maxsplit=1)[1]
        self.assertEqual(
            bindings.strip().splitlines(),
            [
                "IQ9075_RELEASE_SIGNING_PRIVATE_KEY: ${{ secrets.GITHUB_TOKEN }}",
                "      GCP_PROJECT_ID: ${{ secrets.GITHUB_TOKEN }}",
                "      GCP_SA_KEY: ${{ secrets.GITHUB_TOKEN }}",
            ],
        )
        for output_name in (
            "bundle_name",
            "bundle_sha256",
            "bundle_size",
            "deb_name",
            "deb_sha256",
            "deb_size",
            "built_at",
            "config_schema",
            "min_updater_version",
        ):
            self.assertIn(
                f"${{{{ needs.build.outputs.{output_name} }}}}",
                self.trusted_call,
            )

    def test_secretless_native_arm_build_is_separate_from_signing(self) -> None:
        self.assertIn("runs-on: ubuntu-24.04-arm", self.build)
        self.assertIn("build-agent-bundle.sh", self.build)
        self.assertIn("packaging/deb/build-deb.sh", self.build)
        self.assertIn('BOOTSTRAP_BUNDLE_PATH="$bundle_path"', self.build)
        self.assertIn('deb_name="nuv-agent_${VERSION}_arm64.deb"', self.build)
        self.assertIn('"bootstrapDeb": {', self.build)
        self.assertIn("stamp-build-info.py", self.build)
        self.assertIn("SOURCE_DATE_EPOCH", self.build)
        self.assertNotIn("IQ9075_RELEASE_SIGNING_PRIVATE_KEY", self.build)
        self.assertNotIn("environment:", self.build)
        self.assertIn("environment: iq9075-candidate-sign", self.sign)
        self.assertIn("environment: iq9075-candidate-stage", self.stage)
        self.assertEqual(
            self.sign.count("secrets.IQ9075_RELEASE_SIGNING_PRIVATE_KEY"), 1
        )
        for unrelated_secret in (
            "APT_GPG_PRIVATE_KEY",
            "HOMEBREW_TAP_TOKEN",
        ):
            self.assertNotIn(
                unrelated_secret, self.workflow + self.trusted_workflow
            )
        self.assertNotIn("GCP_SA_KEY", self.sign)
        self.assertNotIn("GCP_PROJECT_ID", self.sign)
        self.assertEqual(self.stage.count("secrets.GCP_SA_KEY"), 1)
        self.assertEqual(self.stage.count("secrets.GCP_PROJECT_ID"), 1)
        self.assertNotIn("actions/upload-artifact", self.stage)
        isolated_sign = self.sign.split(
            "- name: Sign exact canonical BOM with the isolated key", maxsplit=1
        )[1].split("- name: Verify the signed candidate", maxsplit=1)[0]
        self.assertNotIn("GH_TOKEN", isolated_sign)
        self.assertNotIn("gh api", isolated_sign)
        self.assertEqual(isolated_sign.count("python3 "), 1)
        credentialed_stage = self.stage.split(
            "- name: Authenticate source GCP credential for downscoping", maxsplit=1
        )[1]
        self.assertNotIn("GH_TOKEN", credentialed_stage)
        self.assertNotIn("gh api", credentialed_stage)

    def test_called_workflow_declares_only_fixed_environment_secret_names(self) -> None:
        declarations = self.trusted_header.split("    secrets:", maxsplit=1)[1]
        self.assertEqual(declarations.count("        required: false"), 3)
        for name in (
            "IQ9075_RELEASE_SIGNING_PRIVATE_KEY",
            "GCP_PROJECT_ID",
            "GCP_SA_KEY",
        ):
            self.assertEqual(declarations.count(f"      {name}:"), 1)
        self.assertNotIn("secrets: inherit", self.workflow + self.trusted_workflow)
        self.assertEqual(self.trusted_call.count("    secrets:"), 1)

    def test_signing_is_bound_to_policy_key_and_downloaded_artifact(self) -> None:
        revalidate = self.sign.index(
            "Revalidate source and artifact before signer access"
        )
        signer = self.sign.index("Sign exact canonical BOM with the isolated key")
        self.assertLess(revalidate, signer)
        self.assertIn("generate-release-bom.py", self.sign)
        self.assertIn("--schema-version 2", self.sign)
        self.assertIn("--signing-private-key-env", self.sign)
        self.assertIn("trusted-release-keyrings/iq9075-dev.json", self.sign)
        self.assertIn("load_signed_release_bom", self.sign)
        self.assertIn("verify_release_artifact", self.sign)
        self.assertIn("candidate-build-manifest.json", self.sign)
        self.assertIn("DEB_SHA256: ${{ inputs.deb_sha256 }}", self.sign)
        self.assertIn('"bootstrapDeb": {', self.sign)
        self.assertIn("dist/${{ inputs.deb_name }}", self.sign)
        self.assertIn("sha256sum", self.sign)

    def test_privileged_jobs_verify_signed_called_workflow_before_checkout(self) -> None:
        for job in (self.sign, self.stage):
            with self.subTest(job="sign" if job is self.sign else "stage"):
                install = job.index("Install the trusted identity verifier")
                verify = job.index(
                    "Verify the signed GitHub reusable-workflow identity"
                )
                checkout = job.index(
                    "Check out the immutable trusted workflow source"
                )
                self.assertLess(install, verify)
                self.assertLess(verify, checkout)
                self.assertIn("id-token: write", job)
                self.assertIn("ACTIONS_ID_TOKEN_REQUEST_URL", job)
                self.assertIn("ACTIONS_ID_TOKEN_REQUEST_TOKEN", job)
                self.assertIn(".well-known/openid-configuration", job)
                self.assertIn(".well-known/jwks", job)
                self.assertIn("public_key.verify", job)
                self.assertIn('header.get("alg") != "RS256"', job)
                self.assertIn('"job_workflow_sha": trusted_sha', job)
                self.assertIn(
                    '"job_workflow_ref": "plaid-ai/NUV-AGENT/.github/workflows/iq9075-candidate-trusted-publish.yml@" + trusted_sha',
                    job,
                )
                self.assertIn(
                    '"sub": "repo:plaid-ai/NUV-AGENT:environment:" + environment',
                    job,
                )
                self.assertIn("repository: plaid-ai/NUV-AGENT", job)
                self.assertIn(
                    "ref: ${{ steps.trusted-identity.outputs.workflow_sha }}",
                    job,
                )
                self.assertNotIn("job.workflow_sha", job)
                self.assertNotIn("job.workflow_repository", job)
                self.assertNotIn("ref: ${{ inputs.component_sha }}", job)

    @staticmethod
    def _base64url(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")

    def _run_oidc_verifier(
        self,
        root: Path,
        *,
        claim_override: dict[str, object] | None = None,
        corrupt_signature: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048
        )
        numbers = private_key.public_key().public_numbers()
        now = int(time.time())
        trusted_sha = "1" * 40
        component_sha = "2" * 40
        environment = "iq9075-candidate-sign"
        audience = "nuvion-iq9075-candidate-trusted-publisher"
        claims: dict[str, object] = {
            "iss": "https://token.actions.githubusercontent.com",
            "aud": audience,
            "repository": "plaid-ai/NUV-AGENT",
            "repository_id": "1149331364",
            "repository_owner": "plaid-ai",
            "repository_owner_id": "199492120",
            "repository_visibility": "public",
            "sub": f"repo:plaid-ai/NUV-AGENT:environment:{environment}",
            "ref": "refs/heads/main",
            "ref_type": "branch",
            "event_name": "workflow_dispatch",
            "runner_environment": "github-hosted",
            "sha": component_sha,
            "workflow_ref": "plaid-ai/NUV-AGENT/.github/workflows/iq9075-candidate-evidence.yml@refs/heads/main",
            "workflow_sha": component_sha,
            "run_id": "12345",
            "run_attempt": "2",
            "environment": environment,
            "job_workflow_ref": "plaid-ai/NUV-AGENT/.github/workflows/iq9075-candidate-trusted-publish.yml@"
            + trusted_sha,
            "job_workflow_sha": trusted_sha,
            "nbf": now - 5,
            "iat": now - 5,
            "exp": now + 295,
        }
        claims.update(claim_override or {})
        header = {"alg": "RS256", "kid": "test-github-key", "typ": "JWT"}
        encoded_header = self._base64url(
            json.dumps(header, separators=(",", ":")).encode()
        )
        encoded_payload = self._base64url(
            json.dumps(claims, separators=(",", ":")).encode()
        )
        signing_input = f"{encoded_header}.{encoded_payload}".encode()
        signature = private_key.sign(
            signing_input, padding.PKCS1v15(), hashes.SHA256()
        )
        if corrupt_signature:
            signature = bytes([signature[0] ^ 1]) + signature[1:]
        token = f"{encoded_header}.{encoded_payload}.{self._base64url(signature)}"
        token_path = root / "token.json"
        configuration_path = root / "openid.json"
        jwks_path = root / "jwks.json"
        token_path.write_text(json.dumps({"value": token}), encoding="utf-8")
        configuration_path.write_text(
            json.dumps(
                {
                    "issuer": "https://token.actions.githubusercontent.com",
                    "jwks_uri": "https://token.actions.githubusercontent.com/.well-known/jwks",
                }
            ),
            encoding="utf-8",
        )
        jwks_path.write_text(
            json.dumps(
                {
                    "keys": [
                        {
                            "kid": "test-github-key",
                            "kty": "RSA",
                            "alg": "RS256",
                            "use": "sig",
                            "n": self._base64url(
                                numbers.n.to_bytes(
                                    (numbers.n.bit_length() + 7) // 8, "big"
                                )
                            ),
                            "e": self._base64url(
                                numbers.e.to_bytes(
                                    (numbers.e.bit_length() + 7) // 8, "big"
                                )
                            ),
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return subprocess.run(
            [
                sys.executable,
                "-c",
                self.oidc_verifier,
                str(token_path),
                str(configuration_path),
                str(jwks_path),
                trusted_sha,
                component_sha,
                environment,
                audience,
                "12345",
                "2",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_oidc_verifier_accepts_only_signed_exact_claims(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            valid = self._run_oidc_verifier(root)
            self.assertEqual(valid.returncode, 0, valid.stderr)
            wrong_called_sha = self._run_oidc_verifier(
                root, claim_override={"job_workflow_sha": "3" * 40}
            )
            self.assertNotEqual(wrong_called_sha.returncode, 0)
            self.assertIn("job_workflow_sha", wrong_called_sha.stderr)
            wrong_environment = self._run_oidc_verifier(
                root, claim_override={"environment": "iq9075-release"}
            )
            self.assertNotEqual(wrong_environment.returncode, 0)
            self.assertIn("environment", wrong_environment.stderr)
            forged = self._run_oidc_verifier(root, corrupt_signature=True)
            self.assertNotEqual(forged.returncode, 0)

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
                self.assertNotIn(
                    token, self.workflow + self.trusted_workflow
                )
        self.assertIn("mint-candidate-gcs-cab-token.py", self.stage)
        self.assertIn("publish-iq9075-candidate-gcs.py", self.stage)
        self.assertIn("iq9075-candidate-gcs-cab.json", self.stage)
        self.assertNotIn("packaging/apt/publish-gcs.sh", self.stage)
        self.assertNotIn("gcloud storage", self.stage)
        publish_step = self.stage.split(
            "Publish exact candidate objects with downscoped token", maxsplit=1
        )[1]
        self.assertNotIn("--deb", publish_step)
        self.assertIn("ifGenerationMatch=0", (ROOT / "packaging/release/publish-iq9075-candidate-gcs.py").read_text(encoding="utf-8"))
        self.assertIn("releases/by-bom-sha256/", self.sign)
        self.assertIn("retention-days: 2", self.trusted_workflow)
        self.assertIn("cancel-in-progress: false", self.workflow)

    def test_stage_destroys_broad_adc_before_direct_json_api_publisher(self) -> None:
        setup = self.stage.index("Set up gcloud before cloud credentials")
        authenticate = self.stage.index("Authenticate source GCP credential for downscoping")
        mint = self.stage.index("Mint prefix-bound token and destroy broad ADC")
        publish = self.stage.index("Publish exact candidate objects with downscoped token")
        self.assertLess(setup, authenticate)
        self.assertLess(authenticate, mint)
        self.assertLess(mint, publish)
        mint_step = self.stage[mint:publish]
        for variable in (
            "GOOGLE_APPLICATION_CREDENTIALS",
            "CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE",
            "GOOGLE_GHA_CREDS_PATH",
        ):
            self.assertIn(variable, mint_step)
            self.assertIn(f"{variable}=", mint_step)
        publish_step = self.stage[publish:]
        self.assertNotIn("secrets.GCP_SA_KEY", publish_step)
        self.assertNotIn("google-github-actions", publish_step)
        self.assertIn("CANDIDATE_CAB_TOKEN_FILE", publish_step)
        self.assertIn('env -i PATH="$PATH" LC_ALL=C PYTHONDONTWRITEBYTECODE=1', mint_step)
        self.assertIn('env -i PATH="$PATH" LC_ALL=C PYTHONDONTWRITEBYTECODE=1', publish_step)

    def test_every_external_action_is_full_sha_pinned(self) -> None:
        uses = re.findall(
            r"^\s+uses:\s+([^\s]+)",
            self.workflow + self.trusted_workflow,
            flags=re.MULTILINE,
        )
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
        self.assertIn("Credential Access Boundary", runbook)
        self.assertIn("storage.objects.create", runbook)
        self.assertIn("storage.objects.get", runbook)
        self.assertIn("ifGenerationMatch=0", runbook)
        self.assertIn("broad legacy service-account", runbook)
        self.assertIn("Workload Identity Federation", runbook)
        self.assertIn("same descriptors", runbook)
        self.assertIn("job_workflow_sha", runbook)
        self.assertIn("literal full commit SHA", runbook)

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
            gcloud_log = root / "gcloud.log"
            binary.mkdir()
            remote.mkdir()
            fake_gcloud = binary / "gcloud"
            fake_gcloud.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
echo "$*" >> "$FAKE_GCLOUD_LOG"
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
                "FAKE_GCLOUD_LOG": str(gcloud_log),
                "VERSION": "0.1.121",
                "BUCKET": "test-bucket",
                "SKIP_APT_PUBLISH": "true",
                "OTA_CONTENT_ONLY": "true",
                "APT_PUBLIC_DIR": str(public),
                "APT_RUNTIME_ROOT": str(root / "runtime"),
                "RELEASE_KEYRING_PATH": str(keyring),
                "RELEASE_TRUST_DOMAIN": "test-iq9075",
                "EXPECTED_OTA_COMPONENT_SHA": "a" * 40,
                "EXPECTED_OTA_RELEASE_SEQUENCE": "2",
                "EXPECTED_OTA_ARTIFACT_SHA256": hashlib.sha256(
                    artifact.read_bytes()
                ).hexdigest(),
                "EXPECTED_OTA_BOM_SHA256": hashlib.sha256(
                    bom.read_bytes()
                ).hexdigest(),
                "EXPECTED_OTA_SIGNATURE_SHA256": hashlib.sha256(
                    signature.read_bytes()
                ).hexdigest(),
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
            copy_calls = [
                line
                for line in gcloud_log.read_text(encoding="utf-8").splitlines()
                if line.startswith("storage cp ")
            ]
            self.assertTrue(copy_calls)
            self.assertTrue(
                all("--if-generation-match=0" in call for call in copy_calls)
            )
            cat_calls = [
                line
                for line in gcloud_log.read_text(encoding="utf-8").splitlines()
                if line.startswith("storage cat ")
            ]
            self.assertGreaterEqual(len(cat_calls), 6)
            for expected_object in objects:
                self.assertGreaterEqual(
                    sum(expected_object in call for call in cat_calls), 2
                )

            remote_bom = remote / prefix / "release-bom.json"
            remote_bom.write_bytes(b"different immutable bytes\n")
            collision = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertNotEqual(collision.returncode, 0)
            self.assertIn("Refusing to overwrite", collision.stderr)

            linked_artifact = root / "linked-agent-bundle.tar.gz"
            linked_artifact.symlink_to(artifact)
            linked = subprocess.run(
                [
                    str(ROOT / "packaging/apt/publish-gcs.sh"),
                    str(linked_artifact),
                    str(bom),
                    str(signature),
                    str(linked_artifact),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertNotEqual(linked.returncode, 0)
            self.assertIn("symbolic link", linked.stderr)

            wrong_component = {**environment, "EXPECTED_OTA_COMPONENT_SHA": "b" * 40}
            mismatched = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                env=wrong_component,
            )
            self.assertNotEqual(mismatched.returncode, 0)
            self.assertIn("exact candidate identity", mismatched.stderr)


if __name__ == "__main__":
    unittest.main()
