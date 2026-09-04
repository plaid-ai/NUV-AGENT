from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "packaging/release/verify-candidate-publisher-tag.py"
SPEC = importlib.util.spec_from_file_location("verify_candidate_publisher_tag", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


@unittest.skipUnless(shutil.which("git") and shutil.which("gpg"), "git and gpg are required")
class CandidatePublisherTagTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        cls.support = Path(cls._temporary.name)
        cls.gpg_home = cls.support / "gpg"
        cls.gpg_home.mkdir(mode=0o700)
        cls.gpg_environment = {**os.environ, "GNUPGHOME": str(cls.gpg_home)}
        subprocess.run(
            [
                "gpg",
                "--batch",
                "--passphrase",
                "",
                "--quick-generate-key",
                "Candidate Publisher Test <candidate@example.invalid>",
                "ed25519",
                "cert",
                "1d",
            ],
            check=True,
            capture_output=True,
            env=cls.gpg_environment,
        )
        listing = subprocess.check_output(
            ["gpg", "--batch", "--with-colons", "--list-keys"],
            text=True,
            env=cls.gpg_environment,
        )
        cls.fingerprint = next(
            line.split(":")[9]
            for line in listing.splitlines()
            if line.startswith("fpr:")
        )
        subprocess.run(
            [
                "gpg",
                "--batch",
                "--passphrase",
                "",
                "--quick-add-key",
                cls.fingerprint,
                "ed25519",
                "sign",
                "1d",
            ],
            check=True,
            capture_output=True,
            env=cls.gpg_environment,
        )
        cls.public_key = subprocess.check_output(
            ["gpg", "--batch", "--armor", "--export", cls.fingerprint],
            env=cls.gpg_environment,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def setUp(self) -> None:
        self._test_root = tempfile.TemporaryDirectory()
        self.root = Path(self._test_root.name)

    def tearDown(self) -> None:
        self._test_root.cleanup()

    def _git(self, repository: Path, *arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(repository), *arguments],
            text=True,
            env=self.gpg_environment,
        ).strip()

    def _run_git(self, repository: Path, *arguments: str) -> None:
        subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            env=self.gpg_environment,
        )

    def _fixture(
        self, *, policy_fingerprint: str | None = None
    ) -> tuple[Path, str, str, Path, Path]:
        repository = self.root / "repository"
        repository.mkdir()
        self._run_git(repository, "init", "--initial-branch=main")
        self._run_git(repository, "config", "user.name", "Candidate Publisher Test")
        self._run_git(repository, "config", "user.email", "candidate@example.invalid")
        self._run_git(repository, "config", "user.signingkey", self.fingerprint)
        self._run_git(repository, "config", "gpg.program", "gpg")
        (repository / ".gitignore").write_text("*.cache\n", encoding="utf-8")
        (repository / "publisher.txt").write_text("trusted publisher\n", encoding="utf-8")
        policy = repository / VERIFIER.POLICY_RELATIVE_PATH
        policy.parent.mkdir(parents=True)
        policy_payload = self._policy_payload()
        if policy_fingerprint is not None:
            policy_payload["trustedTagSignerFingerprints"] = [policy_fingerprint]
        self._write_policy(policy, policy_payload)
        signers = repository / VERIFIER.SIGNER_RELATIVE_PATH
        signers.mkdir()
        (signers / "candidate-publisher.asc").write_bytes(self.public_key)
        self._run_git(repository, "add", ".")
        self._run_git(repository, "commit", "-m", "publisher")
        publisher_sha = self._git(repository, "rev-parse", "HEAD")
        self._run_git(
            repository,
            "tag",
            "--sign",
            "--message",
            "trusted candidate publisher",
            VERIFIER.EXPECTED_TAG,
            publisher_sha,
        )
        (repository / "component.txt").write_text("component\n", encoding="utf-8")
        self._run_git(repository, "add", "component.txt")
        self._run_git(repository, "commit", "-m", "component")
        component_sha = self._git(repository, "rev-parse", "HEAD")
        self._run_git(
            repository, "update-ref", "refs/remotes/origin/main", component_sha
        )
        self._run_git(repository, "checkout", "--detach", publisher_sha)

        return repository, publisher_sha, component_sha, policy, signers

    def _policy_payload(self) -> dict[str, object]:
        return {
            "schemaVersion": 1,
            "defaultBranch": "main",
            "trustedTagSignerFingerprints": [self.fingerprint],
            "candidatePublisher": {
                "tag": "candidate-publisher-v1",
                "tagRef": "refs/tags/candidate-publisher-v1",
                "workflow": ".github/workflows/iq9075-candidate-trusted-publish.yml",
                "agentVersion": "0.1.121",
                "releaseSequence": 2,
                "configSchema": "12",
                "minUpdaterVersion": "0.2.0",
                "rulesetName": "protected-candidate-publisher",
            },
        }

    def _write_policy(
        self, path: Path, payload: dict[str, object] | None = None
    ) -> None:
        path.write_text(
            json.dumps(payload or self._policy_payload(), sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _verify(
        self,
        repository: Path,
        publisher_sha: str,
        component_sha: str,
        policy: Path,
        signers: Path,
    ) -> dict[str, str]:
        return VERIFIER.verify_candidate_publisher_tag(
            repository=repository,
            publisher_sha=publisher_sha,
            component_sha=component_sha,
            main_ref="refs/remotes/origin/main",
            policy_path=policy,
            signer_directory=signers,
        )

    def test_accepts_exact_signed_direct_tag_for_clean_ancestor(self) -> None:
        repository, publisher_sha, component_sha, policy, signers = self._fixture()
        result = self._verify(
            repository, publisher_sha, component_sha, policy, signers
        )
        self.assertEqual(result["candidate_publisher_tag"], VERIFIER.EXPECTED_TAG)
        self.assertEqual(result["candidate_publisher_sha"], publisher_sha)
        self.assertEqual(result["component_sha"], component_sha)
        self.assertEqual(result["tag_signer_fingerprint"], self.fingerprint)
        self.assertRegex(result["candidate_publisher_tag_object_sha"], r"^[0-9a-f]{40}$")

    def test_rejects_lightweight_tag(self) -> None:
        repository, publisher_sha, component_sha, policy, signers = self._fixture()
        self._run_git(repository, "tag", "--delete", VERIFIER.EXPECTED_TAG)
        self._run_git(repository, "tag", VERIFIER.EXPECTED_TAG, publisher_sha)
        with self.assertRaises(VERIFIER.CandidatePublisherVerificationError):
            self._verify(repository, publisher_sha, component_sha, policy, signers)

    def test_rejects_unsigned_annotated_tag(self) -> None:
        repository, publisher_sha, component_sha, policy, signers = self._fixture()
        self._run_git(repository, "tag", "--delete", VERIFIER.EXPECTED_TAG)
        self._run_git(
            repository,
            "tag",
            "--annotate",
            "--message",
            "unsigned",
            VERIFIER.EXPECTED_TAG,
            publisher_sha,
        )
        with self.assertRaises(VERIFIER.CandidatePublisherVerificationError):
            self._verify(repository, publisher_sha, component_sha, policy, signers)

    def test_rejects_signed_tag_chain(self) -> None:
        repository, publisher_sha, component_sha, policy, signers = self._fixture()
        self._run_git(repository, "tag", "--delete", VERIFIER.EXPECTED_TAG)
        self._run_git(
            repository,
            "tag",
            "--sign",
            "--message",
            "inner",
            "candidate-publisher-inner",
            publisher_sha,
        )
        self._run_git(
            repository,
            "tag",
            "--sign",
            "--message",
            "outer",
            VERIFIER.EXPECTED_TAG,
            "candidate-publisher-inner",
        )
        with self.assertRaisesRegex(
            VERIFIER.CandidatePublisherVerificationError, "directly annotate a commit"
        ):
            self._verify(repository, publisher_sha, component_sha, policy, signers)

    def test_rejects_signed_tag_that_targets_a_different_commit(self) -> None:
        repository, publisher_sha, component_sha, policy, signers = self._fixture()
        self._run_git(repository, "tag", "--delete", VERIFIER.EXPECTED_TAG)
        self._run_git(
            repository,
            "tag",
            "--sign",
            "--message",
            "wrong publisher",
            VERIFIER.EXPECTED_TAG,
            component_sha,
        )
        with self.assertRaisesRegex(
            VERIFIER.CandidatePublisherVerificationError,
            "does not identify the supplied publisher SHA",
        ):
            self._verify(repository, publisher_sha, component_sha, policy, signers)

    def test_rejects_policy_drift(self) -> None:
        repository, publisher_sha, component_sha, policy, signers = self._fixture()
        payload = self._policy_payload()
        candidate = payload["candidatePublisher"]
        assert isinstance(candidate, dict)
        candidate["unexpected"] = True
        self._write_policy(policy, payload)
        with self.assertRaisesRegex(
            VERIFIER.CandidatePublisherVerificationError, "policy fields"
        ):
            VERIFIER._candidate_policy(policy)

        for field, value in (
            ("tag", "candidate-publisher-v2"),
            ("tagRef", "refs/tags/candidate-publisher-v2"),
        ):
            with self.subTest(field=field):
                payload = self._policy_payload()
                candidate = payload["candidatePublisher"]
                assert isinstance(candidate, dict)
                candidate[field] = value
                self._write_policy(policy, payload)
                with self.assertRaisesRegex(
                    VERIFIER.CandidatePublisherVerificationError, "policy values"
                ):
                    VERIFIER._candidate_policy(policy)

    def test_rejects_policy_or_signer_paths_outside_publisher_checkout(self) -> None:
        repository, publisher_sha, component_sha, policy, signers = self._fixture()
        outside_policy = self.root / "policy.json"
        outside_policy.write_bytes(policy.read_bytes())
        with self.assertRaisesRegex(
            VERIFIER.CandidatePublisherVerificationError, "policy path"
        ):
            self._verify(
                repository,
                publisher_sha,
                component_sha,
                outside_policy,
                signers,
            )

        outside_signers = self.root / "signers"
        shutil.copytree(signers, outside_signers)
        with self.assertRaisesRegex(
            VERIFIER.CandidatePublisherVerificationError, "signer directory"
        ):
            self._verify(
                repository,
                publisher_sha,
                component_sha,
                policy,
                outside_signers,
            )

    def test_rejects_nonexact_signer_keyring(self) -> None:
        repository, publisher_sha, component_sha, policy, signers = self._fixture(
            policy_fingerprint="A" * 40
        )
        with self.assertRaisesRegex(
            VERIFIER.CandidatePublisherVerificationError, "exactly match"
        ):
            self._verify(repository, publisher_sha, component_sha, policy, signers)

    def test_rejects_dirty_checkout_including_ignored_file(self) -> None:
        repository, publisher_sha, component_sha, policy, signers = self._fixture()
        (repository / "forged.cache").write_text("executable payload\n", encoding="utf-8")
        with self.assertRaisesRegex(
            VERIFIER.CandidatePublisherVerificationError, "not clean"
        ):
            self._verify(repository, publisher_sha, component_sha, policy, signers)

    def test_rejects_noncurrent_or_non_descendant_component(self) -> None:
        repository, publisher_sha, component_sha, policy, signers = self._fixture()
        with self.assertRaisesRegex(
            VERIFIER.CandidatePublisherVerificationError, "differs from current"
        ):
            self._verify(repository, publisher_sha, publisher_sha, policy, signers)

        tree = self._git(repository, "rev-parse", f"{publisher_sha}^{{tree}}")
        unrelated_sha = subprocess.check_output(
            ["git", "-C", str(repository), "commit-tree", tree],
            input="unrelated component\n",
            text=True,
            env={
                **self.gpg_environment,
                "GIT_AUTHOR_NAME": "Candidate Publisher Test",
                "GIT_AUTHOR_EMAIL": "candidate@example.invalid",
                "GIT_COMMITTER_NAME": "Candidate Publisher Test",
                "GIT_COMMITTER_EMAIL": "candidate@example.invalid",
            },
        ).strip()
        self._run_git(
            repository, "update-ref", "refs/remotes/origin/main", unrelated_sha
        )
        with self.assertRaisesRegex(
            VERIFIER.CandidatePublisherVerificationError, "not an ancestor"
        ):
            self._verify(repository, publisher_sha, unrelated_sha, policy, signers)

    def test_rejects_tag_ref_change_during_verification(self) -> None:
        repository, publisher_sha, component_sha, policy, signers = self._fixture()

        def move_tag(*_args: object, **_kwargs: object) -> str:
            self._run_git(
                repository,
                "update-ref",
                VERIFIER.EXPECTED_TAG_REF,
                component_sha,
            )
            return self.fingerprint

        with mock.patch.object(VERIFIER, "_verify_signed_tag", side_effect=move_tag):
            with self.assertRaisesRegex(
                VERIFIER.CandidatePublisherVerificationError,
                "changed during verification",
            ):
                self._verify(
                    repository, publisher_sha, component_sha, policy, signers
                )


if __name__ == "__main__":
    unittest.main()
