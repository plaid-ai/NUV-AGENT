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
MANAGER_PATH = ROOT / "packaging/release/manage-secret-migration-environment.sh"
WORKFLOW_PATH = ROOT / ".github/workflows/migrate-release-secrets.yml"
POLICY_PATH = ROOT / "packaging/release/release-security-policy.json"


def load_helper():
    specification = importlib.util.spec_from_file_location(
        "verify_secret_migration_material", HELPER_PATH
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


HELPER = load_helper()


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
        environment = {"PROJECT": "nuvion-project", "KEY": gcp_key()}
        with mock.patch.dict(os.environ, environment, clear=False):
            with mock.patch.object(HELPER.subprocess, "run", return_value=completed) as run:
                HELPER.verify_gcp_auth("KEY", "PROJECT")
        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            child_environment = call.kwargs["env"]
            self.assertIn("CLOUDSDK_CONFIG", child_environment)
            self.assertEqual(child_environment["CLOUDSDK_CORE_DISABLE_PROMPTS"], "1")
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

    def test_iq_private_key_must_match_committed_public_keyring(self) -> None:
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
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "keyring.json").write_text(
                json.dumps(
                    {
                        "keys": {
                            "release-key": base64.b64encode(public_raw).decode("ascii")
                        }
                    }
                ),
                encoding="utf-8",
            )
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
            with mock.patch.dict(
                os.environ,
                {"IQ_KEY": base64.b64encode(private_raw).decode("ascii")},
                clear=False,
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

    def test_workflow_copies_only_currently_missing_target_secrets_via_stdin(self) -> None:
        workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
        self.assertEqual(workflow.count("gh secret set HOMEBREW_TAP_TOKEN"), 1)
        self.assertEqual(
            workflow.count("gh secret set IQ9075_RELEASE_SIGNING_PRIVATE_KEY"), 1
        )
        self.assertNotIn("gh secret set GCP_", workflow)
        self.assertNotIn("gh secret set APT_", workflow)
        self.assertIn("printf '%s' \"$SOURCE_HOMEBREW_TAP_TOKEN\"", workflow)
        self.assertIn("printf '%s' \"$SOURCE_IQ_SIGNING_PRIVATE_KEY\"", workflow)
        self.assertNotIn("--body", workflow)

    def test_deletion_requires_material_jobs_and_exact_environment_metadata(self) -> None:
        workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
        deletion = workflow.split("  delete-repository-copies:", maxsplit=1)[1]
        for prerequisite in (
            "migrate",
            "verify-homebrew",
            "verify-apt",
            "verify-iq9075",
            "verify-face",
        ):
            self.assertIn(f"      - {prerequisite}", deletion)
        for environment in (
            "homebrew-release",
            "apt-release",
            "iq9075-release",
            "face-artifacts-release",
        ):
            self.assertIn(f"assert_exact_names {environment}", deletion)
        self.assertIn("gh secret delete \"$name\"", deletion)
        self.assertIn("protected main changed before repository-secret deletion", deletion)
        self.assertIn("for _ in 1 2 3", deletion)
        self.assertLess(deletion.index("assert_exact_names"), deletion.index("gh secret delete"))

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
        self.assertIn("forbidden repository secret remains", manager)


if __name__ == "__main__":
    unittest.main()
