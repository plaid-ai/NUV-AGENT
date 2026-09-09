from __future__ import annotations

import base64
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
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
        cls.authorize, build_and_rest = jobs.split("  build:", maxsplit=1)
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
            self.publisher.count("github.ref == 'refs/tags/candidate-publisher-v18'"),
            3,
        )
        self.assertNotRegex(self.publisher, r"refs/tags/candidate-publisher-v1(?![0-9])")
        self.assertNotIn("refs/tags/candidate-publisher-v2", self.publisher)
        self.assertNotIn("refs/tags/candidate-publisher-v3", self.publisher)
        self.assertNotIn("refs/tags/candidate-publisher-v4", self.publisher)
        self.assertNotIn("refs/heads/main", self.publisher)
        self.assertNotIn(
            "uses: plaid-ai/NUV-AGENT/.github/workflows/", self.publisher
        )
        self.assertIn("cancel-in-progress: false", self.header)

    @unittest.skipUnless(shutil.which("bash"), "bash is required")
    def test_authorization_accepts_only_exact_v18_ref_and_workflow_identity(self) -> None:
        script = textwrap.dedent(
            self.authorize.split("        run: |\n", maxsplit=1)[1]
        )
        workflow = (
            "plaid-ai/NUV-AGENT/.github/workflows/"
            "iq9075-candidate-trusted-publish.yml@"
        )
        active_ref = "refs/tags/candidate-publisher-v18"
        cases = [
            (active_ref, active_ref, "15", True),
            (active_ref, active_ref, "2", False),
            (active_ref, active_ref, "3", False),
            (active_ref, active_ref, "4", False),
            (active_ref, active_ref, "5", False),
            (active_ref, active_ref, "6", False),
            (active_ref, active_ref, "7", False),
            (active_ref, active_ref, "8", False),
            (active_ref, active_ref, "9", False),
            (active_ref, active_ref, "10", False),
            (active_ref, active_ref, "11", False),
            (active_ref, active_ref, "12", False),
            (active_ref, active_ref, "13", False),
            (active_ref, active_ref, "14", False),
            (active_ref, active_ref, "16", False),
            (active_ref + "0", active_ref + "0", "15", False),
        ]
        for retired_ref in (
            "refs/tags/candidate-publisher-v1",
            "refs/tags/candidate-publisher-v2",
            "refs/tags/candidate-publisher-v3",
            "refs/tags/candidate-publisher-v4",
            "refs/tags/candidate-publisher-v5",
            "refs/tags/candidate-publisher-v6",
            "refs/tags/candidate-publisher-v7",
            "refs/tags/candidate-publisher-v8",
            "refs/tags/candidate-publisher-v9",
            "refs/tags/candidate-publisher-v10",
            "refs/tags/candidate-publisher-v11",
            "refs/tags/candidate-publisher-v12",
            "refs/tags/candidate-publisher-v13",
            "refs/tags/candidate-publisher-v14",
            "refs/tags/candidate-publisher-v15",
            "refs/tags/candidate-publisher-v16",
            "refs/tags/candidate-publisher-v17",
        ):
            cases.extend(
                (
                    (retired_ref, retired_ref, "15", False),
                    (retired_ref, active_ref, "15", False),
                    (active_ref, retired_ref, "15", False),
                )
            )
        for ref, workflow_ref, sequence, accepted in cases:
            with self.subTest(ref=ref, workflow_ref=workflow_ref, sequence=sequence):
                result = subprocess.run(
                    ["bash", "-c", script],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    env={
                        **os.environ,
                        "RELEASE_ACTOR_ID": "57535980",
                        "GITHUB_REPOSITORY": "plaid-ai/NUV-AGENT",
                        "GITHUB_EVENT_NAME": "workflow_dispatch",
                        "GITHUB_REF": ref,
                        "GITHUB_RUN_ATTEMPT": "1",
                        "GITHUB_SHA": "1" * 40,
                        "PUBLISHER_SHA": "1" * 40,
                        "PUBLISHER_REF": workflow + workflow_ref,
                        "COMPONENT_SHA": "2" * 40,
                        "VERSION": "0.1.121",
                        "RELEASE_SEQUENCE": sequence,
                    },
                )
                self.assertEqual(result.returncode == 0, accepted, result.stderr)

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
                    "refs/tags/candidate-publisher-v18", job
                )
                self.assertEqual(
                    OIDC.WORKFLOW_REF,
                    "plaid-ai/NUV-AGENT/.github/workflows/iq9075-candidate-trusted-publish.yml@refs/tags/candidate-publisher-v18",
                )
                self.assertNotIn("ref: ${{ inputs.component_sha }}", job)
                self.assertNotIn("stamp-build-info.py", job)
                self.assertNotIn("build-agent-bundle.sh", job)
                self.assertNotIn("packaging/deb/build-deb.sh", job)
                self.assertNotIn("apt-get", job)
        self.assertEqual(
            self.sign.count("secrets.IQ9075_RELEASE_SIGNING_PRIVATE_KEY"), 0
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
        self.assertIn('"job_workflow_ref"', verifier)
        self.assertIn('"job_workflow_sha"', verifier)
        self.assertNotIn("component_sha", verifier)
        self.assertNotIn("cryptography", verifier)
        self.assertIn('shutil.which("openssl")', verifier)

    def test_privileged_jobs_probe_ed25519_before_secret_access(self) -> None:
        step_name = "Verify runner Ed25519 support before secret access"
        for name, job in (("sign", self.sign), ("stage", self.stage)):
            with self.subTest(job=name):
                self.assertLess(job.index("verify-github-oidc.py"), job.index(step_name))
                self.assertLess(job.index(step_name), job.index("Authenticate exact immutable candidate to Cloud KMS") if name == "sign" else job.index("secrets."))
                step = job.split(step_name, maxsplit=1)[1].split("      - name:", maxsplit=1)[0]
                self.assertNotIn("env:", step)
                self.assertNotRegex(step, r"\bpip[0-9]*\s+install\b")
                script = textwrap.dedent(step.split("        run: |\n", maxsplit=1)[1])
                python_script = script.split("python3 - <<'PY'\n", maxsplit=1)[1].rsplit("\nPY", maxsplit=1)[0]
                result = subprocess.run(
                    [sys.executable, "-c", python_script],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Ed25519 runtime verified: cryptography=", result.stdout)

    def test_candidate_identity_is_locked_again_before_each_secret_use(self) -> None:
        sign_preflight = self.sign.index(
            "Recheck exact source immediately before signer access"
        )
        sign_secret = self.sign.index("Authenticate exact immutable candidate to Cloud KMS")
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

    def test_candidate_v18_prevents_downgrade_or_scope_expansion(self) -> None:
        self.assertGreaterEqual(self.publisher.count('= "0.1.121" ]'), 3)
        self.assertGreaterEqual(self.publisher.count('= "15" ]'), 3)
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
        self.assertIn(
            '--bom-id "nuv-agent-${VERSION}-iq9075-aarch64-seq${RELEASE_SEQUENCE}"',
            self.sign,
        )
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
                "tag": "candidate-publisher-v18",
                "tagRef": "refs/tags/candidate-publisher-v18",
                "retiredTagRefs": [
                    "refs/tags/candidate-publisher-v1",
                    "refs/tags/candidate-publisher-v2",
                    "refs/tags/candidate-publisher-v3",
                    "refs/tags/candidate-publisher-v4",
                    "refs/tags/candidate-publisher-v5",
                    "refs/tags/candidate-publisher-v6",
                    "refs/tags/candidate-publisher-v7",
                    "refs/tags/candidate-publisher-v8",
                    "refs/tags/candidate-publisher-v9",
                    "refs/tags/candidate-publisher-v10",
                    "refs/tags/candidate-publisher-v11",
                    "refs/tags/candidate-publisher-v12",
                    "refs/tags/candidate-publisher-v13",
                    "refs/tags/candidate-publisher-v14",
                    "refs/tags/candidate-publisher-v15",
                    "refs/tags/candidate-publisher-v16",
                    "refs/tags/candidate-publisher-v17",
                ],
                "workflow": ".github/workflows/iq9075-candidate-trusted-publish.yml",
                "agentVersion": "0.1.121",
                "releaseSequence": 15,
                "configSchema": "12",
                "minUpdaterVersion": "0.2.0",
                "rulesetName": "protected-candidate-publisher",
            },
        )
        for name in ("iq9075-candidate-sign", "iq9075-candidate-stage"):
            self.assertEqual(
                policy["requiredEnvironments"][name]["deploymentBranchPolicies"],
                [{"name": "candidate-publisher-v18", "type": "tag"}],
            )
            self.assertFalse(
                policy["requiredEnvironments"][name]["canAdminsBypass"]
            )
        current_runbook = (ROOT / "packaging/release/cloud-kms/ota-cutover.md").read_text()
        self.assertIn("--ref candidate-publisher-v18", current_runbook)
        self.assertIn("kms-approve-release.yml", current_runbook)
        # The old runbook is retained as historical v10 migration evidence.
        runbook = RUNBOOK.read_text(encoding="utf-8")
        self.assertIn(
            "gh workflow run iq9075-candidate-trusted-publish.yml", runbook
        )
        self.assertIn("--ref candidate-publisher-v10", runbook)
        self.assertNotRegex(runbook, r"--ref candidate-publisher-v1(?![0-9])")
        self.assertNotIn("--ref candidate-publisher-v2", runbook)
        self.assertNotIn("--ref candidate-publisher-v3", runbook)
        self.assertNotIn("--ref candidate-publisher-v4", runbook)
        self.assertIn("job_workflow_sha", runbook)
        self.assertIn("job_workflow_ref", runbook)
        self.assertGreaterEqual(runbook.count("set -euo pipefail"), 2)
        self.assertIn('"can_admins_bypass": false', runbook)
        self.assertIn("expected exactly one retired v9 tag policy", runbook)
        self.assertIn("locked_candidate_ruleset", runbook)
        self.assertIn("remote ref is not the exact annotated tag", runbook)
        for version in (1, 2, 3, 4, 5, 6, 7, 8, 9):
            self.assertGreaterEqual(
                runbook.count(f"retired_v{version}_tag_object_sha"), 2
            )
        self.assertIn("7059eb508b40b156939dc70ab6db0fd5ae08d579", runbook)
        self.assertIn("d1e6febe683be07a7cecbf48dc383125c0bfe54b", runbook)
        migration = runbook.split("updated_policy=", maxsplit=1)[1]
        migration = migration.split("~~~", maxsplit=1)[0]
        self.assertIn('"$(gh api --method PUT', migration)
        self.assertIn("deployment-branch-policies/$policy_id", migration)
        self.assertNotIn("--method DELETE", migration)
        self.assertNotIn("--method POST", migration)
        self.assertIn('.id == $id and .name == "candidate-publisher-v10"', migration)
        self.assertIn('.type == "tag"', migration)


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
            self.assertEqual(
                result["workflowRef"],
                "plaid-ai/NUV-AGENT/.github/workflows/"
                "iq9075-candidate-trusted-publish.yml@refs/tags/candidate-publisher-v18",
            )

    def test_accepts_signed_exact_optional_job_workflow_pair(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            paths, publisher_sha = self._fixture(
                Path(raw_root),
                overrides={
                    "job_workflow_ref": OIDC.WORKFLOW_REF,
                    "job_workflow_sha": "1" * 40,
                },
            )
            self.assertEqual(
                self._verify(paths, publisher_sha)["publisherSha"], publisher_sha
            )

    def test_rejects_partial_null_or_different_job_workflow_identity(self) -> None:
        cases = (
            {"job_workflow_ref": OIDC.WORKFLOW_REF},
            {"job_workflow_sha": "1" * 40},
            {"job_workflow_ref": None, "job_workflow_sha": None},
            {"job_workflow_ref": None, "job_workflow_sha": "1" * 40},
            {"job_workflow_ref": OIDC.WORKFLOW_REF, "job_workflow_sha": None},
            {"job_workflow_ref": OIDC.WORKFLOW_REF, "job_workflow_sha": "2" * 40},
            {
                "job_workflow_ref": OIDC.WORKFLOW_REF.replace(
                    "candidate-publisher-v18", "candidate-publisher-v2"
                ),
                "job_workflow_sha": "1" * 40,
            },
            {
                "job_workflow_ref": OIDC.WORKFLOW_REF.replace(
                    "candidate-publisher-v18", "candidate-publisher-v3"
                ),
                "job_workflow_sha": "1" * 40,
            },
            {
                "job_workflow_ref": "other/repo/.github/workflows/reusable.yml@refs/heads/main",
                "job_workflow_sha": "1" * 40,
            },
            {
                "job_workflow_ref": OIDC.WORKFLOW_REF.replace(
                    "candidate-publisher-v18", "candidate-publisher-v4"
                ),
                "job_workflow_sha": "1" * 40,
            },
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with tempfile.TemporaryDirectory() as raw_root:
                    paths, publisher_sha = self._fixture(
                        Path(raw_root), overrides=overrides
                    )
                    with self.assertRaises(OIDC.OidcVerificationError):
                        self._verify(paths, publisher_sha)

    def test_rejects_validly_signed_retired_oidc_identity(self) -> None:
        cases = []
        for tag in (
            "candidate-publisher-v1",
            "candidate-publisher-v2",
            "candidate-publisher-v3",
            "candidate-publisher-v4",
        ):
            retired = {
                "ref": f"refs/tags/{tag}",
                "workflow_ref": (
                    "plaid-ai/NUV-AGENT/.github/workflows/"
                    f"iq9075-candidate-trusted-publish.yml@refs/tags/{tag}"
                ),
            }
            cases.extend((retired, {"workflow_ref": retired["workflow_ref"]}))
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with tempfile.TemporaryDirectory() as raw_root:
                    paths, publisher_sha = self._fixture(
                        Path(raw_root), overrides=overrides
                    )
                    with self.assertRaisesRegex(
                        OIDC.OidcVerificationError,
                        r"claim (ref|workflow_ref) is not protected",
                    ):
                        self._verify(paths, publisher_sha)

    def test_rejects_component_identity_and_forged_signature(self) -> None:
        cases = (
            ({"sha": "2" * 40}, False),
            ({"workflow_sha": "2" * 40}, False),
            ({"ref": "refs/heads/main"}, False),
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
