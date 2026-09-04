from __future__ import annotations

import base64
import importlib.util
import json
import re
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEGACY_WORKFLOW = ROOT / ".github/workflows/iq9075-candidate-evidence.yml"
PUBLISHER_WORKFLOW = (
    ROOT / ".github/workflows/iq9075-candidate-trusted-publish.yml"
)
OIDC_PATH = ROOT / "packaging/release/verify-github-oidc.py"
RUNBOOK = ROOT / "packaging/release/v0.1.121-release-runbook.md"

SPEC = importlib.util.spec_from_file_location("verify_github_oidc", OIDC_PATH)
assert SPEC is not None and SPEC.loader is not None
OIDC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OIDC)


class Iq9075CandidateEvidenceWorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.legacy = LEGACY_WORKFLOW.read_text(encoding="utf-8")
        cls.publisher = PUBLISHER_WORKFLOW.read_text(encoding="utf-8")
        cls.header, jobs = cls.publisher.split("jobs:", maxsplit=1)
        _, build_and_rest = jobs.split("  build:", maxsplit=1)
        cls.build, sign_and_stage = build_and_rest.split("  sign:", maxsplit=1)
        cls.sign, cls.stage = sign_and_stage.split("  stage:", maxsplit=1)

    def test_legacy_main_entry_point_is_permanently_deny_only(self) -> None:
        header, job = self.legacy.split("jobs:", maxsplit=1)
        self.assertIn("workflow_dispatch:", header)
        for trigger in ("workflow_call:", "push:", "pull_request:", "workflow_run:"):
            self.assertNotIn(trigger, header)
        self.assertIn("permissions: {}", header)
        self.assertIn("exit 1", job)
        for forbidden in (
            "uses:",
            "secrets.",
            "environment:",
            "checkout",
            "component_sha",
            "id-token: write",
        ):
            self.assertNotIn(forbidden, job)

    def test_publisher_is_standalone_manual_exact_tag_only(self) -> None:
        self.assertIn("workflow_dispatch:", self.header)
        for trigger in ("workflow_call:", "push:", "pull_request:", "workflow_run:"):
            self.assertNotIn(trigger, self.header)
        self.assertIn("permissions: {}", self.header)
        self.assertNotIn("secrets:", self.header)
        self.assertEqual(
            self.publisher.count("github.ref == 'refs/tags/candidate-publisher-v1'"),
            3,
        )
        self.assertNotIn("refs/heads/main", self.publisher)
        self.assertNotIn(
            "uses: plaid-ai/NUV-AGENT/.github/workflows/", self.publisher
        )
        self.assertIn("cancel-in-progress: false", self.header)

    def test_candidate_runs_are_attempt_one_only_and_never_reused(self) -> None:
        self.assertEqual(
            self.publisher.count('[ "$GITHUB_RUN_ATTEMPT" = "1" ]'),
            4,
        )
        for job in (self.build, self.sign, self.stage):
            self.assertIn("Reject partial or whole workflow rerun", job)
            self.assertIn("dispatch a new candidate run instead of rerunning", job)
        runbook = RUNBOOK.read_text(encoding="utf-8")
        self.assertIn("never use GitHub **Re-run jobs**", runbook)
        self.assertIn("Every build/sign/stage artifact name is bound to one", runbook)
        self.assertIn("`run_id` and `run_attempt`", runbook)
        self.assertIn("A failed run's\n  artifacts are not inputs", runbook)

    def test_secretless_build_executes_only_exact_current_main_component(self) -> None:
        self.assertIn("runs-on: ubuntu-24.04-arm", self.build)
        self.assertIn("ref: ${{ inputs.component_sha }}", self.build)
        self.assertIn("persist-credentials: false", self.build)
        self.assertIn("git/ref/heads/main", self.build)
        self.assertIn('live_main_sha" = "$REQUESTED_COMPONENT_SHA', self.build)
        self.assertIn('git merge-base --is-ancestor "$PUBLISHER_SHA"', self.build)
        self.assertIn('[ "$GITHUB_SHA" = "$PUBLISHER_SHA" ]', self.build)
        self.assertNotIn('[ "$GITHUB_SHA" = "$REQUESTED_COMPONENT_SHA" ]', self.build)
        self.assertIn("build-agent-bundle.sh", self.build)
        self.assertIn("packaging/deb/build-deb.sh", self.build)
        self.assertIn("stamp-build-info.py", self.build)
        self.assertNotIn("environment:", self.build)
        self.assertNotIn("id-token: write", self.build)
        self.assertNotIn("secrets.", self.build)
        self.assertNotIn("GCP_", self.build)
        self.assertNotIn("RELEASE_SIGNING", self.build)
        self.assertIn(
            "iq9075-candidate-build-${{ github.run_id }}-${{ github.run_attempt }}",
            self.build,
        )

    def test_privileged_jobs_execute_only_signed_publisher_code(self) -> None:
        for name, job, environment in (
            ("sign", self.sign, "iq9075-candidate-sign"),
            ("stage", self.stage, "iq9075-candidate-stage"),
        ):
            with self.subTest(job=name):
                self.assertIn(f"environment: {environment}", job)
                self.assertIn("id-token: write", job)
                self.assertIn("ref: ${{ github.workflow_sha }}", job)
                self.assertIn("fetch-tags: true", job)
                self.assertIn("verify-candidate-publisher-tag.py", job)
                self.assertIn("--main-ref refs/remotes/origin/main", job)
                self.assertIn("verify-github-oidc.py", job)
                self.assertIn(
                    "refs/tags/candidate-publisher-v1", job
                )
                self.assertEqual(
                    OIDC.WORKFLOW_REF,
                    "plaid-ai/NUV-AGENT/.github/workflows/iq9075-candidate-trusted-publish.yml@refs/tags/candidate-publisher-v1",
                )
                self.assertNotIn("ref: ${{ inputs.component_sha }}", job)
                self.assertNotIn("stamp-build-info.py", job)
                self.assertNotIn("build-agent-bundle.sh", job)
                self.assertNotIn("packaging/deb/build-deb.sh", job)
                self.assertNotIn("apt-get", job)
        self.assertEqual(
            self.sign.count("secrets.IQ9075_RELEASE_SIGNING_PRIVATE_KEY"), 1
        )
        self.assertNotIn("secrets.GCP", self.sign)
        self.assertEqual(self.stage.count("secrets.GCP_PROJECT_ID"), 1)
        self.assertEqual(self.stage.count("secrets.GCP_SA_KEY"), 1)
        self.assertNotIn("RELEASE_SIGNING_PRIVATE_KEY", self.stage)

    def test_oidc_identity_is_standalone_p_and_not_component_a(self) -> None:
        verifier = OIDC_PATH.read_text(encoding="utf-8")
        expected = {
            '"ref": TAG_REF',
            '"ref_type": "tag"',
            '"sha": publisher_sha',
            '"workflow_ref": WORKFLOW_REF',
            '"workflow_sha": publisher_sha',
            '"event_name": "workflow_dispatch"',
        }
        for fragment in expected:
            self.assertIn(fragment, verifier)
        self.assertIn('if "job_workflow_ref" in claims', verifier)
        self.assertIn('or "job_workflow_sha" in claims', verifier)
        self.assertNotIn("component_sha", verifier)
        self.assertNotIn("cryptography", verifier)
        self.assertIn('shutil.which("openssl")', verifier)

    def test_candidate_identity_is_locked_again_before_each_secret_use(self) -> None:
        sign_preflight = self.sign.index(
            "Recheck exact source immediately before signer access"
        )
        sign_secret = self.sign.index("secrets.IQ9075_RELEASE_SIGNING_PRIVATE_KEY")
        self.assertLess(sign_preflight, sign_secret)
        self.assertIn('live_main_sha" = "$COMPONENT_SHA', self.sign[sign_preflight:])

        stage_preflight = self.stage.index(
            "Revalidate source and signed bytes before cloud credentials"
        )
        stage_secret = self.stage.index("secrets.GCP_PROJECT_ID")
        self.assertLess(stage_preflight, stage_secret)
        self.assertIn(
            'live_main_sha" = "$REQUESTED_COMPONENT_SHA',
            self.stage[stage_preflight:stage_secret],
        )

    def test_candidate_v1_prevents_downgrade_or_scope_expansion(self) -> None:
        self.assertGreaterEqual(self.publisher.count('= "0.1.121" ]'), 3)
        self.assertGreaterEqual(self.publisher.count('= "2" ]'), 3)
        self.assertGreaterEqual(self.publisher.count('= "12" ]'), 2)
        self.assertGreaterEqual(self.publisher.count('= "0.2.0" ]'), 2)
        forbidden = (
            "git push",
            "git tag",
            "gh release",
            "publish-immutable",
            "generate-release-promotion",
            "sequence-reservation",
            "aptly",
            "contents: write",
        )
        for token in forbidden:
            self.assertNotIn(token, self.publisher)
        self.assertIn("mint-candidate-gcs-cab-token.py", self.stage)
        self.assertIn("publish-iq9075-candidate-gcs.py", self.stage)
        self.assertIn("releases/by-bom-sha256/", self.sign)
        publish = self.stage.split(
            "Publish exact candidate objects with downscoped token", maxsplit=1
        )[1]
        self.assertNotIn("--deb", publish)
        self.assertNotIn("gcloud storage", self.stage)

    def test_every_external_action_is_full_sha_pinned(self) -> None:
        actions = re.findall(
            r"^\s+uses:\s+([^\s]+)", self.legacy + self.publisher, re.MULTILINE
        )
        self.assertTrue(actions)
        for action in actions:
            self.assertRegex(action, r"^[^@]+@[0-9a-f]{40}$")

    def test_policy_and_runbook_name_the_same_trusted_tag(self) -> None:
        policy = json.loads(
            (ROOT / "packaging/release/release-security-policy.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            policy["candidatePublisher"],
            {
                "tag": "candidate-publisher-v1",
                "tagRef": "refs/tags/candidate-publisher-v1",
                "workflow": ".github/workflows/iq9075-candidate-trusted-publish.yml",
                "agentVersion": "0.1.121",
                "releaseSequence": 2,
                "configSchema": "12",
                "minUpdaterVersion": "0.2.0",
                "rulesetName": "protected-candidate-publisher",
            },
        )
        for name in ("iq9075-candidate-sign", "iq9075-candidate-stage"):
            self.assertEqual(
                policy["requiredEnvironments"][name]["deploymentBranchPolicies"],
                [{"name": "candidate-publisher-v1", "type": "tag"}],
            )
            self.assertFalse(
                policy["requiredEnvironments"][name]["canAdminsBypass"]
            )
        runbook = RUNBOOK.read_text(encoding="utf-8")
        self.assertIn(
            "gh workflow run iq9075-candidate-trusted-publish.yml", runbook
        )
        self.assertIn("--ref candidate-publisher-v1", runbook)
        self.assertIn(
            "Reusable-only\njob_workflow_sha and job_workflow_ref claims are rejected",
            runbook,
        )
        self.assertGreaterEqual(runbook.count("set -euo pipefail"), 2)
        self.assertIn('"can_admins_bypass": false', runbook)
        self.assertIn("expected exactly one numeric main branch policy", runbook)
        self.assertIn("locked_candidate_ruleset", runbook)
        self.assertIn("remote ref is not the exact annotated tag", runbook)


@unittest.skipUnless(shutil.which("openssl"), "openssl is required")
class StandalonePublisherOidcTest(unittest.TestCase):
    @staticmethod
    def _base64url(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")

    def _fixture(
        self,
        root: Path,
        *,
        overrides: dict[str, object] | None = None,
        corrupt_signature: bool = False,
    ) -> tuple[dict[str, Path], str]:
        private_key = root / "private.pem"
        subprocess.run(
            [
                "openssl",
                "genpkey",
                "-algorithm",
                "RSA",
                "-pkeyopt",
                "rsa_keygen_bits:2048",
                "-out",
                str(private_key),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        modulus_output = subprocess.check_output(
            ["openssl", "rsa", "-in", str(private_key), "-noout", "-modulus"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        self.assertTrue(modulus_output.startswith("Modulus="))
        modulus = bytes.fromhex(modulus_output.removeprefix("Modulus="))
        publisher_sha = "1" * 40
        now = int(time.time())
        claims: dict[str, object] = {
            "iss": OIDC.ISSUER,
            "aud": "nuvion-iq9075-candidate-trusted-publisher",
            "repository": "plaid-ai/NUV-AGENT",
            "repository_id": "1149331364",
            "repository_owner": "plaid-ai",
            "repository_owner_id": "199492120",
            "repository_visibility": "public",
            "sub": "repo:plaid-ai/NUV-AGENT:environment:iq9075-candidate-sign",
            "ref": OIDC.TAG_REF,
            "ref_type": "tag",
            "event_name": "workflow_dispatch",
            "runner_environment": "github-hosted",
            "sha": publisher_sha,
            "workflow_ref": OIDC.WORKFLOW_REF,
            "workflow_sha": publisher_sha,
            "run_id": "12345",
            "run_attempt": "2",
            "environment": "iq9075-candidate-sign",
            "nbf": now - 5,
            "iat": now - 5,
            "exp": now + 295,
        }
        claims.update(overrides or {})
        header = {"alg": "RS256", "kid": "github-test-key", "typ": "JWT"}
        encoded_header = self._base64url(
            json.dumps(header, separators=(",", ":")).encode()
        )
        encoded_claims = self._base64url(
            json.dumps(claims, separators=(",", ":")).encode()
        )
        signing_input = f"{encoded_header}.{encoded_claims}".encode()
        signing_input_path = root / "signing-input"
        signature_path = root / "signature"
        signing_input_path.write_bytes(signing_input)
        subprocess.run(
            [
                "openssl",
                "dgst",
                "-sha256",
                "-sign",
                str(private_key),
                "-out",
                str(signature_path),
                str(signing_input_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        signature = signature_path.read_bytes()
        if corrupt_signature:
            signature = bytes([signature[0] ^ 1]) + signature[1:]
        token = f"{encoded_header}.{encoded_claims}.{self._base64url(signature)}"
        paths = {
            "token": root / "token.json",
            "configuration": root / "configuration.json",
            "jwks": root / "jwks.json",
        }
        paths["token"].write_text(json.dumps({"value": token}), encoding="utf-8")
        paths["configuration"].write_text(
            json.dumps({"issuer": OIDC.ISSUER, "jwks_uri": OIDC.JWKS_URI}),
            encoding="utf-8",
        )
        paths["jwks"].write_text(
            json.dumps(
                {
                    "keys": [
                        {
                            "kid": "github-test-key",
                            "kty": "RSA",
                            "alg": "RS256",
                            "use": "sig",
                            "n": self._base64url(
                                modulus
                            ),
                            "e": self._base64url(
                                (65537).to_bytes(3, "big")
                            ),
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return paths, publisher_sha

    def _verify(self, paths: dict[str, Path], publisher_sha: str) -> dict[str, str]:
        return OIDC.verify(
            token_response_path=paths["token"],
            configuration_path=paths["configuration"],
            jwks_path=paths["jwks"],
            publisher_sha=publisher_sha,
            environment="iq9075-candidate-sign",
            audience="nuvion-iq9075-candidate-trusted-publisher",
            run_id="12345",
            run_attempt="2",
        )

    def test_accepts_signed_exact_standalone_claims(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            paths, publisher_sha = self._fixture(Path(raw_root))
            result = self._verify(paths, publisher_sha)
            self.assertEqual(result["publisherSha"], publisher_sha)

    def test_rejects_component_or_reusable_identity_and_forged_signature(self) -> None:
        cases = (
            ({"sha": "2" * 40}, False),
            ({"workflow_sha": "2" * 40}, False),
            ({"ref": "refs/heads/main"}, False),
            ({"job_workflow_sha": "1" * 40}, False),
            ({}, True),
        )
        for overrides, corrupt in cases:
            with self.subTest(overrides=overrides, corrupt=corrupt):
                with tempfile.TemporaryDirectory() as raw_root:
                    paths, publisher_sha = self._fixture(
                        Path(raw_root),
                        overrides=overrides,
                        corrupt_signature=corrupt,
                    )
                    with self.assertRaises(OIDC.OidcVerificationError):
                        self._verify(paths, publisher_sha)


if __name__ == "__main__":
    unittest.main()
