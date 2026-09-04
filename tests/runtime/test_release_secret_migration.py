from __future__ import annotations

import base64
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = ROOT / "packaging/release/verify-secret-migration-material.py"
RESUME_HELPER_PATH = ROOT / "packaging/release/resume-release-secret-migration.py"
MANAGER_PATH = ROOT / "packaging/release/manage-secret-migration-environment.sh"
WORKFLOW_PATH = ROOT / ".github/workflows/migrate-release-secrets.yml"
RELEASE_WORKFLOW_PATH = ROOT / ".github/workflows/release-publish.yml"
POLICY_PATH = ROOT / "packaging/release/release-security-policy.json"
RUNBOOK_PATH = ROOT / "packaging/release/v0.1.121-release-runbook.md"
PACKAGING_README_PATH = ROOT / "packaging/README.md"
HOMEBREW_BOOTSTRAP_PATH = ROOT / "packaging/release/bootstrap-homebrew-tap.sh"
CANONICAL_HOMEBREW_REPOSITORY = "plaid-ai/homebrew-NUV-AGENT-HOMEBREW"


def load_helper():
    specification = importlib.util.spec_from_file_location(
        "verify_secret_migration_material", HELPER_PATH
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


HELPER = load_helper()


def load_resume_helper():
    specification = importlib.util.spec_from_file_location(
        "resume_release_secret_migration", RESUME_HELPER_PATH
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


RESUME_HELPER = load_resume_helper()


def gcp_key(project: str = "nuvion-project") -> str:
    return json.dumps(
        {
            "type": "service_account",
            "project_id": project,
            "private_key_id": "key-12345678",
            "private_key": "-----BEGIN PRIVATE KEY-----\ntest\n-----END PRIVATE KEY-----\n",
            "client_email": f"publisher@{project}.iam.gserviceaccount.com",
        }
    )


class ReleaseSecretMigrationTests(unittest.TestCase):
    def test_gcp_auth_uses_isolated_config_and_suppresses_subprocess_output(self) -> None:
        completed = subprocess.CompletedProcess([], 0, stdout="access-token\n")
        environment = {
            "PROJECT": "nuvion-project",
            "KEY": gcp_key(),
            "UNRELATED_SECRET": "must-not-reach-child",
        }
        with mock.patch.dict(os.environ, environment, clear=False):
            with mock.patch.object(HELPER.subprocess, "run", return_value=completed) as run:
                HELPER.verify_gcp_auth("KEY", "PROJECT")
        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            child_environment = call.kwargs["env"]
            self.assertIn("CLOUDSDK_CONFIG", child_environment)
            self.assertEqual(child_environment["CLOUDSDK_CORE_DISABLE_PROMPTS"], "1")
            self.assertNotIn("PROJECT", child_environment)
            self.assertNotIn("KEY", child_environment)
            self.assertNotIn("UNRELATED_SECRET", child_environment)
            self.assertIs(call.kwargs["stderr"], subprocess.DEVNULL)

    def test_gcp_errors_never_include_credential_identity(self) -> None:
        secret = gcp_key(project="unexpected-project")
        with mock.patch.dict(
            os.environ,
            {"PROJECT": "nuvion-project", "KEY": secret},
            clear=False,
        ):
            with self.assertRaises(HELPER.MaterialError) as raised:
                HELPER.verify_gcp_auth("KEY", "PROJECT")
        message = str(raised.exception)
        self.assertNotIn("publisher@", message)
        self.assertNotIn("key-12345678", message)
        self.assertNotIn("unexpected-project", message)

    def test_iq_private_key_must_match_raw_or_spki_public_keyring(self) -> None:
        private_key = Ed25519PrivateKey.generate()
        private_raw = private_key.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
        public_raw = private_key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
        public_spki = private_key.public_key().public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "policy.json"
            policy.write_text(
                json.dumps(
                    {
                        "iq9075": {
                            "publisherKeyId": "release-key",
                            "publicKeyringFile": "keyring.json",
                        }
                    }
                ),
                encoding="utf-8",
            )
            for keyring_material in (public_raw, public_spki):
                with self.subTest(keyring_length=len(keyring_material)):
                    (root / "keyring.json").write_text(
                        json.dumps(
                            {
                                "keys": {
                                    "release-key": base64.b64encode(
                                        keyring_material
                                    ).decode("ascii")
                                }
                            }
                        ),
                        encoding="utf-8",
                    )
                    with mock.patch.dict(
                        os.environ,
                        {
                            "IQ_KEY": base64.b64encode(private_raw).decode("ascii"),
                            "UNRELATED_SECRET": "must-not-reach-child",
                        },
                        clear=False,
                    ):
                        with mock.patch.object(
                            HELPER.subprocess,
                            "run",
                            return_value=subprocess.CompletedProcess(
                                [], 0, stdout=public_spki
                            ),
                        ) as run:
                            HELPER.verify_iq_signing_key("IQ_KEY", policy)
                    child_environment = run.call_args.kwargs["env"]
                    self.assertNotIn("IQ_KEY", child_environment)
                    self.assertNotIn("UNRELATED_SECRET", child_environment)

            (root / "keyring.json").write_text(
                json.dumps(
                    {
                        "keys": {
                            "release-key": base64.b64encode(
                                (b"\x00" * 12) + public_raw
                            ).decode("ascii")
                        }
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.dict(
                os.environ,
                {"IQ_KEY": base64.b64encode(private_raw).decode("ascii")},
                clear=False,
            ):
                with self.assertRaisesRegex(
                    HELPER.MaterialError, "IQ9075 public key is invalid"
                ):
                    HELPER.verify_iq_signing_key("IQ_KEY", policy)

    def test_workflow_is_a_protected_main_only_one_shot(self) -> None:
        workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
        self.assertIn("github.ref == 'refs/heads/main'", workflow)
        self.assertIn("inputs.confirmation == 'MIGRATE_AND_DELETE", workflow)
        self.assertIn("EXECUTING_WORKFLOW_SHA: ${{ github.workflow_sha }}", workflow)
        self.assertIn("[ \"$GITHUB_SHA\" = \"$main_sha\" ]", workflow)
        self.assertIn("[ \"$protected\" = true ]", workflow)
        self.assertIn("[ \"$token_login\" = \"$GITHUB_ACTOR\" ]", workflow)
        self.assertIn("releaseAdminTeamId", workflow)
        self.assertNotIn("pull_request:", workflow)
        self.assertNotIn("push:", workflow)
        self.assertNotIn("macos-", workflow)

    def test_workflow_copies_only_scoped_target_secrets_via_stdin(self) -> None:
        workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
        self.assertEqual(workflow.count("gh secret set HOMEBREW_TAP_TOKEN"), 1)
        self.assertEqual(
            workflow.count("gh secret set IQ9075_RELEASE_SIGNING_PRIVATE_KEY"), 2
        )
        self.assertEqual(workflow.count("gh secret set GCP_PROJECT_ID"), 1)
        self.assertEqual(workflow.count("gh secret set GCP_SA_KEY"), 1)
        self.assertNotIn("gh secret set APT_", workflow)
        self.assertIn("printf '%s' \"$SOURCE_HOMEBREW_TAP_TOKEN\"", workflow)
        self.assertIn("printf '%s' \"$SOURCE_IQ_SIGNING_PRIVATE_KEY\"", workflow)
        self.assertIn("printf '%s' \"$SOURCE_GCP_PROJECT_ID\"", workflow)
        self.assertIn("printf '%s' \"$SOURCE_GCP_SA_KEY\"", workflow)
        self.assertNotIn("--body", workflow)
        homebrew_copy = workflow.split(
            "- name: Copy Homebrew source only", maxsplit=1
        )[1].split("- name: Copy IQ9075 signing source only", maxsplit=1)[0]
        signing_copy = workflow.split(
            "- name: Copy IQ9075 signing source only", maxsplit=1
        )[1].split("- name: Copy GCP source only", maxsplit=1)[0]
        gcp_copy = workflow.split("- name: Copy GCP source only", maxsplit=1)[
            1
        ].split("  verify-homebrew:", maxsplit=1)[0]
        self.assertNotIn("SOURCE_IQ_SIGNING", homebrew_copy)
        self.assertNotIn("SOURCE_GCP_", homebrew_copy)
        self.assertNotIn("SOURCE_HOMEBREW", signing_copy)
        self.assertNotIn("SOURCE_GCP_", signing_copy)
        self.assertNotIn("SOURCE_HOMEBREW", gcp_copy)
        self.assertNotIn("SOURCE_IQ_SIGNING", gcp_copy)
        for copy_step in (homebrew_copy, signing_copy, gcp_copy):
            self.assertIn("env -i", copy_step)

    def test_homebrew_repository_is_canonical_across_release_surfaces(self) -> None:
        workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
        release_workflow = RELEASE_WORKFLOW_PATH.read_text(encoding="utf-8")
        runbook = RUNBOOK_PATH.read_text(encoding="utf-8")
        packaging_readme = PACKAGING_README_PATH.read_text(encoding="utf-8")
        bootstrap = HOMEBREW_BOOTSTRAP_PATH.read_text(encoding="utf-8")
        canonical_api_path = f"repos/{CANONICAL_HOMEBREW_REPOSITORY}"
        self.assertEqual(workflow.count(canonical_api_path), 2)
        self.assertIn(f"TAP_REPO: {CANONICAL_HOMEBREW_REPOSITORY}", release_workflow)
        self.assertIn(f"`{CANONICAL_HOMEBREW_REPOSITORY}`", runbook)
        self.assertIn(f"`{CANONICAL_HOMEBREW_REPOSITORY}`", packaging_readme)
        self.assertIn("REPO=${REPO:-homebrew-NUV-AGENT-HOMEBREW}", bootstrap)
        self.assertNotIn("repos/plaid-ai/NUV-agent-homebrew", workflow)
        self.assertNotIn("push access to `plaid-ai/NUV-agent-homebrew`", packaging_readme)

    def test_missing_source_metadata_selects_target_only_resume(self) -> None:
        forbidden = RESUME_HELPER.EXPECTED_FORBIDDEN
        self.assertEqual(
            RESUME_HELPER.classify_source_state(set(forbidden)), "copy"
        )
        for missing in forbidden:
            with self.subTest(missing=missing):
                self.assertEqual(
                    RESUME_HELPER.classify_source_state(set(forbidden) - {missing}),
                    "target-only-resume",
                )
        self.assertEqual(
            RESUME_HELPER.classify_source_state(set()), "target-only-resume"
        )

    def test_metadata_client_does_not_forward_ambient_secret_environment(self) -> None:
        completed = subprocess.CompletedProcess([], 0, stdout="GCP_SA_KEY\n")
        with mock.patch.dict(
            os.environ,
            {"UNRELATED_SECRET": "must-not-reach-gh", "SOURCE_GCP_SA_KEY": "source"},
            clear=False,
        ):
            client = RESUME_HELPER.GitHubRepositorySecrets(
                RESUME_HELPER.EXPECTED_REPOSITORY, "administrator-token"
            )
            with mock.patch.object(
                RESUME_HELPER.subprocess, "run", return_value=completed
            ) as run:
                self.assertEqual(client.names(), {"GCP_SA_KEY"})

        child_environment = run.call_args.kwargs["env"]
        self.assertEqual(
            set(child_environment),
            {"PATH", "HOME", "LC_ALL", "GH_PROMPT_DISABLED", "GH_TOKEN"},
        )
        self.assertEqual(child_environment["GH_TOKEN"], "administrator-token")
        self.assertNotIn("UNRELATED_SECRET", child_environment)
        self.assertNotIn("SOURCE_GCP_SA_KEY", child_environment)
        self.assertIs(run.call_args.kwargs["stdin"], subprocess.DEVNULL)
        self.assertIs(run.call_args.kwargs["stderr"], subprocess.DEVNULL)

    def test_partial_deletion_failure_is_resumable_without_restoring_sources(self) -> None:
        class FakeRepositorySecrets:
            def __init__(self) -> None:
                self.current = set(RESUME_HELPER.EXPECTED_FORBIDDEN) | {
                    "UNRELATED_REPOSITORY_SECRET"
                }
                self.fail_name = RESUME_HELPER.EXPECTED_FORBIDDEN[1]
                self.failures_remaining = RESUME_HELPER.DELETE_ATTEMPTS
                self.attempts: list[str] = []

            def names(self) -> set[str]:
                return set(self.current)

            def delete(self, name: str) -> bool:
                self.attempts.append(name)
                if name == self.fail_name and self.failures_remaining:
                    self.failures_remaining -= 1
                    return False
                self.current.discard(name)
                return True

        repository = FakeRepositorySecrets()
        with self.assertRaises(RESUME_HELPER.MigrationError):
            RESUME_HELPER.delete_remaining_sources(
                repository, sleeper=lambda _seconds: None
            )
        first = RESUME_HELPER.EXPECTED_FORBIDDEN[0]
        self.assertNotIn(first, repository.current)
        self.assertIn(repository.fail_name, repository.current)

        repository.fail_name = ""
        RESUME_HELPER.delete_remaining_sources(
            repository, sleeper=lambda _seconds: None
        )
        self.assertEqual(repository.current, {"UNRELATED_REPOSITORY_SECRET"})
        self.assertEqual(repository.attempts.count(first), 1)
        self.assertNotIn("UNRELATED_REPOSITORY_SECRET", repository.attempts)

    def test_delete_accepts_concurrent_absence_but_not_a_remaining_source(self) -> None:
        class ConcurrentDelete:
            def __init__(self, *, disappears: bool) -> None:
                self.current = {RESUME_HELPER.EXPECTED_FORBIDDEN[0]}
                self.disappears = disappears

            def names(self) -> set[str]:
                return set(self.current)

            def delete(self, name: str) -> bool:
                if self.disappears:
                    self.current.discard(name)
                return False

        RESUME_HELPER.delete_remaining_sources(
            ConcurrentDelete(disappears=True), sleeper=lambda _seconds: None
        )
        with self.assertRaises(RESUME_HELPER.MigrationError):
            RESUME_HELPER.delete_remaining_sources(
                ConcurrentDelete(disappears=False), sleeper=lambda _seconds: None
            )

    def test_source_values_are_skipped_during_target_only_resume(self) -> None:
        workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
        migrate = workflow.split("  migrate:", maxsplit=1)[1].split(
            "  verify-homebrew:", maxsplit=1
        )[0]
        self.assertIn("resume-release-secret-migration.py plan", migrate)
        source_steps = (
            "Verify source Homebrew authorization in isolation",
            "Verify source IQ9075 signing identity in isolation",
            "Set up pinned gcloud CLI before isolated source GCP verification",
            "Verify source GCP identity in isolation",
            "Copy Homebrew source only to its target via stdin",
            "Copy IQ9075 signing source only to signer targets via stdin",
            "Copy GCP source only to candidate stager via stdin",
        )
        for index, step_name in enumerate(source_steps):
            start = migrate.index(f"- name: {step_name}")
            next_start = (
                migrate.index("- name:", start + 8)
                if index + 1 < len(source_steps)
                else len(migrate)
            )
            with self.subTest(step=step_name):
                self.assertIn(
                    "if: steps.source-state.outputs.mode == 'copy'",
                    migrate[start:next_start],
                )

    def test_deletion_requires_material_jobs_and_exact_environment_metadata(self) -> None:
        workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
        deletion = workflow.split("  delete-repository-copies:", maxsplit=1)[1]
        for prerequisite in (
            "migrate",
            "verify-homebrew",
            "verify-apt",
            "verify-iq9075",
            "verify-candidate-sign",
            "verify-candidate-stage",
            "verify-face",
        ):
            self.assertIn(f"      - {prerequisite}", deletion)
        for environment in (
            "homebrew-release",
            "apt-release",
            "iq9075-release",
            "iq9075-candidate-sign",
            "iq9075-candidate-stage",
            "face-artifacts-release",
        ):
            self.assertIn(f"assert_exact_names {environment}", deletion)
        self.assertIn("resume-release-secret-migration.py delete", deletion)
        self.assertIn("verify-github-release-settings.py", deletion)
        self.assertIn("protected main changed before repository-secret deletion", deletion)
        self.assertIn("no longer a repository administrator", deletion)
        self.assertIn("no longer an active Platform-Admin member", deletion)
        helper = RESUME_HELPER_PATH.read_text(encoding="utf-8")
        self.assertIn("DELETE_ATTEMPTS = 3", helper)
        self.assertIn("if name not in present:", helper)
        self.assertLess(
            deletion.index("assert_exact_names"),
            deletion.index("resume-release-secret-migration.py delete"),
        )

    def test_forbidden_repository_secret_policy_is_the_expected_exact_set(self) -> None:
        policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            policy["forbiddenRepositorySecrets"],
            [
                "APT_GPG_PASSPHRASE",
                "APT_GPG_PRIVATE_KEY",
                "GCP_PROJECT_ID",
                "GCP_SA_KEY",
                "HOMEBREW_TAP_TOKEN",
                "IQ9075_RELEASE_PUBLIC_KEYRING_JSON",
                "IQ9075_RELEASE_SIGNING_KEY_ID",
                "IQ9075_RELEASE_SIGNING_PRIVATE_KEY",
            ],
        )

    def test_environment_manager_accepts_admin_token_only_on_stdin(self) -> None:
        manager = MANAGER_PATH.read_text(encoding="utf-8")
        setup = manager.split("setup_environment()", maxsplit=1)[1].split(
            "cleanup_environment()", maxsplit=1
        )[0]
        self.assertIn("IFS= read -r admin_token", setup)
        self.assertIn("printf '%s' \"$admin_token\"", setup)
        self.assertIn("RELEASE_SECRET_MIGRATION_ADMIN_TOKEN", setup)
        self.assertNotIn("gh auth token", setup)
        self.assertNotIn("credential", setup.lower())
        self.assertIn("git/trees/main?recursive=1", manager)
        self.assertIn("iq9075-candidate-sign", manager)
        self.assertIn("iq9075-candidate-stage", manager)
        self.assertIn("forbidden repository secret remains", manager)

    def test_one_shot_removal_requires_exact_five_file_set_to_leave_main(self) -> None:
        one_shot_paths = {
            ".github/workflows/migrate-release-secrets.yml",
            "packaging/release/manage-secret-migration-environment.sh",
            "packaging/release/verify-secret-migration-material.py",
            "packaging/release/resume-release-secret-migration.py",
            "tests/runtime/test_release_secret_migration.py",
        }
        manager = MANAGER_PATH.read_text(encoding="utf-8")
        setup = manager.split("setup_environment()", maxsplit=1)[1].split(
            "cleanup_environment()", maxsplit=1
        )[0]
        cleanup = manager.split("cleanup_environment()", maxsplit=1)[1]
        runbook = RUNBOOK_PATH.read_text(encoding="utf-8")
        removal = runbook.split("whose exact delete\nset is:", maxsplit=1)[1].split(
            "After all five paths", maxsplit=1
        )[0]

        constants = {
            "WORKFLOW_PATH": ".github/workflows/migrate-release-secrets.yml",
            "MANAGER_PATH": "packaging/release/manage-secret-migration-environment.sh",
            "MATERIAL_VERIFIER_PATH": "packaging/release/verify-secret-migration-material.py",
            "HELPER_PATH": "packaging/release/resume-release-secret-migration.py",
            "TEST_PATH": "tests/runtime/test_release_secret_migration.py",
        }
        for name, path in constants.items():
            arg_name = (
                "verifier"
                if name == "MATERIAL_VERIFIER_PATH"
                else name.removesuffix("_PATH").lower()
            )
            self.assertIn(f'readonly {name}="{path}"', manager)
            self.assertIn(f'--arg {arg_name} "${name}"', setup)
            self.assertIn(f'--arg {arg_name} "${name}"', cleanup)
        self.assertIn("protected-main tree metadata is truncated", cleanup)
        self.assertIn("exact five-file one-shot migration set", setup)
        self.assertIn("remove the exact five-file one-shot migration set", cleanup)
        self.assertEqual(
            {
                line
                for line in removal.splitlines()
                if line.startswith((".github/", "packaging/", "tests/"))
            },
            one_shot_paths,
        )
        self.assertIn(
            '"$migration_control_root/packaging/release/'
            'manage-secret-migration-environment.sh" cleanup',
            runbook,
        )
        self.assertIn(
            'git worktree add --detach "$migration_control_root" '
            '"$migration_publisher_sha"',
            runbook,
        )

    def test_cleanup_refuses_when_any_one_shot_path_remains_on_remote_main(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fake_gh = Path(directory) / "gh"
            fake_gh.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
case "$*" in
  "api user --jq .login") echo admin ;;
  *"/collaborators/admin/permission"*) echo admin ;;
  *"teams/16128529/memberships/admin"*) printf 'active\\tmember\\n' ;;
  *"git/trees/main?recursive=1"*)
    printf '%s\\n' '{"truncated":false,"tree":[{"path":"packaging/release/resume-release-secret-migration.py"}]}'
    ;;
  *) exit 97 ;;
esac
""",
                encoding="utf-8",
            )
            fake_gh.chmod(0o700)
            environment = os.environ.copy()
            environment["PATH"] = f"{directory}:{environment['PATH']}"
            completed = subprocess.run(
                ["bash", str(MANAGER_PATH), "cleanup"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
                check=False,
                env=environment,
            )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "remove the exact five-file one-shot migration set from protected main",
            completed.stderr,
        )
        self.assertNotIn("forbidden repository secret remains", completed.stderr)


if __name__ == "__main__":
    unittest.main()
