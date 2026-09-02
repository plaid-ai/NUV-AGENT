from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str, relative: str):
    specification = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


VERIFY_SOURCE = load_script(
    "verify_release_source", "packaging/release/verify-release-source.py"
)
PLAN_OTA = load_script("plan_iq9075_ota", "packaging/release/plan-iq9075-ota.py")
PREPARE_APT = load_script(
    "prepare_apt_rollback", "packaging/release/prepare-apt-rollback.py"
)
PROMOTION = load_script(
    "generate_release_promotion", "packaging/release/generate-release-promotion.py"
)
SETTINGS = load_script(
    "verify_github_release_settings",
    "packaging/release/verify-github-release-settings.py",
)
GITHUB_RELEASE = load_script(
    "publish_github_release", "packaging/release/publish-github-release.py"
)
READINESS = load_script(
    "verify_release_readiness", "packaging/release/verify-release-readiness.py"
)
SETTINGS_ATTESTATION = load_script(
    "verify_release_settings_attestation",
    "packaging/release/verify-release-settings-attestation.py",
)


class ReleaseSecurityWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.publish = (
            ROOT / ".github/workflows/release-publish.yml"
        ).read_text(encoding="utf-8")
        self.request = (
            ROOT / ".github/workflows/release-request.yml"
        ).read_text(encoding="utf-8")

    def test_tag_push_is_secret_zero_and_default_branch_starts_publisher(self) -> None:
        self.assertIn("on:\n  push:\n    tags:", self.request)
        self.assertNotIn("${{ secrets.", self.request)
        self.assertNotIn("contents: write", self.request)
        self.assertNotIn("environment:", self.request)
        self.assertIn('workflows: ["release-request"]', self.publish)
        self.assertIn("group: release-publisher-global", self.publish)
        trigger = self.publish.split("jobs:", maxsplit=1)[0]
        self.assertNotIn("  push:\n", trigger)
        self.assertIn('publisher workflow_dispatch must run from main', self.publish)
        self.assertIn("github.event.workflow_run.head_sha", self.publish)
        self.assertIn("ref: ${{ github.workflow_sha }}", self.publish)
        self.assertIn("path: settings-evidence", self.publish)
        self.assertIn('--trusted-publisher-sha "$TRUSTED_PUBLISHER_SHA"', self.publish)

    def test_every_credential_job_uses_environment_and_trusted_checkout(self) -> None:
        sections = {}
        job_names = [
            "github-release-publish",
            "homebrew-publish",
            "apt-publish",
            "finalize-distribution",
            "iq9075-ota-publish",
        ]
        for index, name in enumerate(job_names):
            start = self.publish.index(f"  {name}:")
            following = [
                self.publish.find(f"  {candidate}:", start + 1)
                for candidate in job_names[index + 1 :]
            ]
            following = [value for value in following if value >= 0]
            end = min(following) if following else len(self.publish)
            sections[name] = self.publish[start:end]
        expected_environment = {
            "github-release-publish": "homebrew-release",
            "homebrew-publish": "homebrew-release",
            "apt-publish": "apt-release",
            "finalize-distribution": "homebrew-release",
            "iq9075-ota-publish": "iq9075-release",
        }
        for name, section in sections.items():
            with self.subTest(job=name):
                self.assertIn(
                    f"environment: {expected_environment[name]}", section
                )
                self.assertIn("Checkout trusted publisher only", section)
                self.assertIn(
                    "ref: ${{ needs.release-preflight.outputs.trusted_publisher_sha }}",
                    section,
                )
                self.assertIn("path: publisher", section)
                self.assertNotIn("ref: ${{ needs.release-preflight.outputs.release_tag }}", section)
                self.assertNotIn("build-agent-bundle.sh", section)

    def test_github_release_is_draft_until_final_promotion_then_immutable(self) -> None:
        stage = self.publish.split("  github-release-publish:", maxsplit=1)[1].split(
            "  homebrew-publish:", maxsplit=1
        )[0]
        finalize = self.publish.split("  finalize-distribution:", maxsplit=1)[1].split(
            "  iq9075-ota-publish:", maxsplit=1
        )[0]
        self.assertIn("--phase stage", stage)
        self.assertNotIn("--phase finalize", stage)
        self.assertIn("--phase finalize", finalize)
        self.assertIn('NAME="nuv_agent-${VERSION}-distribution-promotion.json"', finalize)
        self.assertIn('publisher/packaging/release/publish-github-release.py', stage)
        self.assertIn('publisher/packaging/release/publish-github-release.py', finalize)
        self.assertNotIn("softprops/action-gh-release", self.publish)

    def test_v121_dependency_readiness_is_enforced_before_build(self) -> None:
        preflight = self.publish.split("  release-preflight:", maxsplit=1)[1].split(
            "  release-build:", maxsplit=1
        )[0]
        self.assertIn("verify-release-readiness.py", preflight)
        self.assertIn("release-readiness.json", preflight)
        self.assertLess(
            self.publish.index("verify-release-readiness.py"),
            self.publish.index("  release-build:"),
        )
        READINESS.verify_readiness(
            ROOT / "packaging/release/release-readiness.json",
            version="0.1.121",
            allow_legacy=False,
        )
        with tempfile.TemporaryDirectory() as raw_root:
            blocked = Path(raw_root) / "readiness.json"
            blocked.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "releases": {
                            "0.1.121": {
                                "status": "BLOCKED",
                                "blockers": [{"id": "TRANSFORMERS-REGRESSION"}],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(READINESS.ReadinessError):
                READINESS.verify_readiness(
                    blocked,
                    version="0.1.121",
                    allow_legacy=False,
                )
        READINESS.verify_readiness(
            ROOT / "packaging/release/release-readiness.json",
            version="0.1.120",
            allow_legacy=True,
        )

    def test_ota_sequence_failure_precedes_private_key_and_uses_global_cas(self) -> None:
        ota = self.publish.split("  iq9075-ota-publish:", maxsplit=1)[1]
        self.assertIn("group: iq9075-ota-global-publisher", ota)
        self.assertLess(
            ota.index("Independently verify latest sequence and version absence"),
            ota.index("IQ9075_RELEASE_SIGNING_PRIVATE_KEY"),
        )
        self.assertLess(
            ota.index("Atomically reserve exact release sequence"),
            ota.index("Sign exact bundle BOM with trusted signer"),
        )
        immutable = (
            ROOT / "packaging/release/publish-immutable-gcs-file.sh"
        ).read_text(encoding="utf-8")
        apt = (ROOT / "packaging/apt/publish-gcs.sh").read_text(encoding="utf-8")
        for source in (immutable, apt):
            self.assertIn("--if-generation-match=0", source)
            self.assertNotIn(" cp -n ", source)
            self.assertIn("gcloud storage cat", source)

    def test_ota_verifier_uses_only_policy_pinned_public_keyring(self) -> None:
        policy_path = ROOT / "packaging/release/release-security-policy.json"
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        keyring_path = policy_path.parent / policy["iq9075"]["publicKeyringFile"]
        self.assertEqual(
            hashlib.sha256(keyring_path.read_bytes()).hexdigest(),
            policy["iq9075"]["publicKeyringSha256"],
        )
        self.assertEqual(
            policy["iq9075"]["publicKeyringSha256"],
            "2d72a28745e14014d5988ecf7970dc6f09c2f077be35105b3ad233cda0d0969a",
        )
        self.assertEqual(
            policy["iq9075"]["publisherKeyId"],
            "release-iq9075-dev-2026-09-01",
        )
        public_map = json.loads(keyring_path.read_text(encoding="utf-8"))["keys"]
        self.assertEqual(
            hashlib.sha256(
                json.dumps(public_map, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest(),
            "fe087dd340fbec31604a8c7910bc95a5c1615c5157c526cae5b4e18090a774c7",
        )
        self.assertNotIn("IQ9075_RELEASE_PUBLIC_KEYRING_JSON", self.publish)
        self.assertNotIn("secrets.IQ9075_RELEASE_SIGNING_KEY_ID", self.publish)
        self.assertNotIn(
            "IQ9075_RELEASE_PUBLIC_KEYRING_JSON",
            policy_path.read_text(encoding="utf-8"),
        )

    def test_all_external_actions_are_full_sha_pinned(self) -> None:
        for path in (
            ROOT / ".github/workflows/agent-release-gate.yml",
            ROOT / ".github/workflows/release-request.yml",
            ROOT / ".github/workflows/release-publish.yml",
        ):
            for line in path.read_text(encoding="utf-8").splitlines():
                match = re.search(r"\buses:\s*([^\s#]+)", line)
                if match is None or match.group(1).startswith("./"):
                    continue
                self.assertRegex(match.group(1), r"@[0-9a-f]{40}$")

    def test_required_main_context_exists_and_is_secret_zero(self) -> None:
        gate = (ROOT / ".github/workflows/agent-release-gate.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("name: agent-release-gate", gate)
        self.assertIn("  agent-release-gate:\n    name: agent-release-gate", gate)
        self.assertIn("runs-on: ubuntu-24.04-arm", gate)
        self.assertIn("needs: arm64-release-prerequisite", gate)
        self.assertIn("if: always()", gate)
        self.assertIn("needs.arm64-release-prerequisite.result", gate)
        self.assertIn("requirements-agent-bundle-arm64.txt", gate)
        self.assertIn("packaging/release/run-isolated-tests.py", gate)
        self.assertIn("actionlint", gate)
        self.assertIn("shellcheck", gate)
        self.assertNotIn("${{ secrets.", gate)
        self.assertNotIn("contents: write", gate)


class ReleaseSourceVerificationTest(unittest.TestCase):
    def _git(self, repository: Path, *arguments: str, environment=None) -> str:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        return result.stdout.strip()

    def _repository(self, root: Path) -> tuple[Path, str]:
        repository = root / "repository"
        repository.mkdir()
        self._git(repository, "init", "-b", "main")
        self._git(repository, "config", "user.name", "Release Test")
        self._git(repository, "config", "user.email", "release@example.invalid")
        (repository / "README").write_text("release\n", encoding="utf-8")
        self._git(repository, "add", "README")
        self._git(repository, "commit", "-m", "release")
        return repository, self._git(repository, "rev-parse", "HEAD")

    def _policy(self, root: Path, *, fingerprint: str, legacy: dict[str, str]) -> Path:
        payload = json.loads(
            (ROOT / "packaging/release/release-security-policy.json").read_text(
                encoding="utf-8"
            )
        )
        payload["trustedTagSignerFingerprints"] = [fingerprint]
        payload["legacyUnsignedReruns"] = legacy
        path = root / "policy.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_exact_legacy_tag_is_manual_rerun_only(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            repository, commit = self._repository(root)
            self._git(repository, "tag", "-a", "v1.2.3", "-m", "legacy")
            policy = self._policy(
                root,
                fingerprint="A" * 40,
                legacy={"v1.2.3": commit},
            )
            signers = root / "signers"
            signers.mkdir()
            verified = VERIFY_SOURCE.verify_release_source(
                repository=repository,
                tag="v1.2.3",
                origin_main_ref="refs/heads/main",
                trusted_publisher_sha=commit,
                event_name="workflow_dispatch",
                policy_path=policy,
                signer_directory=signers,
            )
            self.assertEqual(verified["legacy_rerun"], "true")
            with self.assertRaises(VERIFY_SOURCE.VerificationError):
                VERIFY_SOURCE.verify_release_source(
                    repository=repository,
                    tag="v1.2.3",
                    origin_main_ref="refs/heads/main",
                    trusted_publisher_sha=commit,
                    event_name="workflow_run",
                    policy_path=policy,
                    signer_directory=signers,
                )

    @unittest.skipUnless(shutil.which("gpg"), "gpg is required")
    def test_signed_tag_requires_exact_allowlisted_primary_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            repository, commit = self._repository(root)
            gpg_home = root / "gpg"
            gpg_home.mkdir(mode=0o700)
            environment = {**os.environ, "GNUPGHOME": str(gpg_home)}
            subprocess.run(
                [
                    "gpg",
                    "--batch",
                    "--passphrase",
                    "",
                    "--quick-generate-key",
                    "Release Test <release@example.invalid>",
                    "ed25519",
                    "sign",
                    "1d",
                ],
                check=True,
                capture_output=True,
                env=environment,
            )
            listing = subprocess.check_output(
                ["gpg", "--batch", "--with-colons", "--list-keys"],
                text=True,
                env=environment,
            )
            fingerprint = next(
                line.split(":")[9]
                for line in listing.splitlines()
                if line.startswith("fpr:")
            )
            self._git(repository, "config", "user.signingkey", fingerprint)
            self._git(repository, "config", "gpg.program", "gpg")
            self._git(
                repository,
                "tag",
                "-s",
                "v1.2.3",
                "-m",
                "signed",
                environment=environment,
            )
            signers = root / "signers"
            signers.mkdir()
            public_key = subprocess.check_output(
                ["gpg", "--batch", "--armor", "--export", fingerprint],
                env=environment,
            )
            (signers / "test.asc").write_bytes(public_key)
            policy = self._policy(root, fingerprint=fingerprint, legacy={})
            verified = VERIFY_SOURCE.verify_release_source(
                repository=repository,
                tag="v1.2.3",
                origin_main_ref="refs/heads/main",
                trusted_publisher_sha=commit,
                event_name="workflow_run",
                policy_path=policy,
                signer_directory=signers,
            )
            self.assertEqual(verified["tag_signer_fingerprint"], fingerprint)


class SequenceAndPromotionTest(unittest.TestCase):
    def test_new_sequence_must_be_latest_plus_one(self) -> None:
        from nuvion_app.runtime.release_bom import ReleaseTarget, VerifiedReleaseBom

        target = ReleaseTarget(
            product_model="IQ9075_DEV",
            platform_profile="iq9075_dev",
            hardware_revision="QCS9075-EVK",
            architecture="aarch64",
        )
        published = VerifiedReleaseBom(
            schema_version=2,
            bom_id="nuv-agent-0.1.120-iq9075-aarch64",
            bom_digest="26a7f1674bdd4a24bfe26fa37c681798244990408fe7d858ca76957a88bdb9f1",
            agent_version="0.1.120",
            component_sha="b354026f73d63a82ad4c64923f46dc400a73efcb",
            config_schema="12",
            updater_version=None,
            release_sequence=1,
            min_updater_version="0.1.0",
            targets=(target,),
            publisher_key_id="release-test",
            platform_profiles=(),
            artifact_name="nuv-agent_0.1.120_iq9075-aarch64.agent-bundle.tar.gz",
            artifact_kind="agent-bundle",
            artifact_sha256="1" * 64,
            artifact_size_bytes=10,
            built_at="2026-09-01T12:00:00+00:00",
        )
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            artifact = root / "nuv-agent_0.1.121_iq9075-aarch64.agent-bundle.tar.gz"
            artifact.write_bytes(b"new release")
            keyring = (
                ROOT
                / "packaging/release/trusted-release-keyrings/iq9075-dev.json"
            )
            with mock.patch.object(
                PLAN_OTA, "list_version_boms", return_value={"0.1.120": "1"}
            ), mock.patch.object(
                PLAN_OTA, "_load_remote_signed_bom", return_value=published
            ):
                reservation, output = PLAN_OTA.plan_sequence(
                    policy_path=ROOT / "packaging/release/release-security-policy.json",
                    keyring_path=keyring,
                    artifact_path=artifact,
                    version="0.1.121",
                    component_sha="a" * 40,
                    requested_sequence=2,
                    config_schema="12",
                    min_updater_version="0.2.0",
                    built_at="2026-09-02T00:00:00+00:00",
                )
                self.assertEqual(reservation["releaseSequence"], 2)
                self.assertEqual(output["latest_sequence"], "1")
                self.assertEqual(output["reservation_object"], "releases/reservations/iq9075/2.json")
                with self.assertRaises(PLAN_OTA.SequencePlanError):
                    PLAN_OTA.plan_sequence(
                        policy_path=ROOT / "packaging/release/release-security-policy.json",
                        keyring_path=keyring,
                        artifact_path=artifact,
                        version="0.1.121",
                        component_sha="a" * 40,
                        requested_sequence=3,
                        config_schema="12",
                        min_updater_version="0.2.0",
                        built_at="2026-09-02T00:00:00+00:00",
                    )

    def test_distribution_promotion_is_deterministic_and_binds_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            paths = {}
            for name in ("sdist", "bom", "bundle", "deb"):
                path = root / name
                path.write_bytes(name.encode())
                paths[name] = path
            arguments = argparse.Namespace(
                version="0.1.121",
                tag="v0.1.121",
                component_sha="a" * 40,
                trusted_publisher_sha="b" * 40,
                sdist=paths["sdist"],
                sdist_bom=paths["bom"],
                bundle=paths["bundle"],
                deb=paths["deb"],
                rollback_version="0.1.120",
                rollback_sha256="c" * 64,
            )
            first = PROMOTION.build_distribution(arguments)
            second = PROMOTION.build_distribution(arguments)
            self.assertEqual(first, second)
            self.assertEqual(first["status"], "PROMOTED")
            self.assertEqual(
                first["rollbackPackage"],
                {"agentVersion": "0.1.120", "sha256": "c" * 64},
            )

    def test_ota_promotion_binds_distribution_bundle_to_signed_bom(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            artifacts = {}
            names = {
                "sdist": "nuv_agent-0.1.121.tar.gz",
                "sdist_bom": "nuv_agent-0.1.121-sdist.release-bom.json",
                "bundle": "nuv-agent_0.1.121_iq9075-aarch64.agent-bundle.tar.gz",
                "deb": "nuv-agent_0.1.121_arm64.deb",
            }
            for label, name in names.items():
                path = root / name
                path.write_bytes(label.encode("ascii"))
                artifacts[label] = path
            distribution_arguments = argparse.Namespace(
                version="0.1.121",
                tag="v0.1.121",
                component_sha="a" * 40,
                trusted_publisher_sha="b" * 40,
                sdist=artifacts["sdist"],
                sdist_bom=artifacts["sdist_bom"],
                bundle=artifacts["bundle"],
                deb=artifacts["deb"],
                rollback_version="0.1.120",
                rollback_sha256="c" * 64,
            )
            manifest = PROMOTION.build_distribution(distribution_arguments)
            manifest_path = root / "nuv_agent-0.1.121-distribution-promotion.json"
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            bundle_identity = manifest["artifacts"]["iq9075Bundle"]
            bom = mock.Mock(
                release_sequence=2,
                min_updater_version="0.2.0",
                agent_version="0.1.121",
                component_sha="a" * 40,
                bom_digest="d" * 64,
                artifact_name=bundle_identity["name"],
                artifact_sha256=bundle_identity["sha256"],
                artifact_size_bytes=bundle_identity["sizeBytes"],
                publisher_key_id="release-test",
            )
            ota_arguments = argparse.Namespace(
                distribution_promotion=manifest_path,
                bom=root / "release-bom.json",
                signature=root / "release-bom.json.sig",
                keyring=root / "keyring.json",
                trust_domain="iq9075-dev",
            )
            with mock.patch.object(PROMOTION, "load_release_keyring"), mock.patch.object(
                PROMOTION, "load_signed_release_bom", return_value=bom
            ):
                result = PROMOTION.build_ota(ota_arguments)
                self.assertEqual(result["releaseSequence"], 2)
                manifest["artifacts"]["iq9075Bundle"]["sha256"] = "e" * 64
                manifest_path.write_text(
                    json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaises(PROMOTION.PromotionError):
                    PROMOTION.build_ota(ota_arguments)


class AptRollbackAndCasTest(unittest.TestCase):
    def test_selects_highest_authenticated_lower_version(self) -> None:
        packages = """
Package: nuv-agent
Version: 0.1.119
Architecture: arm64
Filename: pool/main/n/nuv-agent/nuv-agent_0.1.119_arm64.deb
Size: 10
SHA256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa

Package: nuv-agent
Version: 0.1.120
Architecture: arm64
Filename: pool/main/n/nuv-agent/nuv-agent_0.1.120_arm64.deb
Size: 11
SHA256: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
""".strip()
        selected = PREPARE_APT.select_rollback_record(
            PREPARE_APT.parse_packages(packages), current_version="0.1.121"
        )
        assert selected is not None
        self.assertEqual(selected["Version"], "0.1.120")
        release = (
            "Origin: NUV\nSHA256:\n "
            + "c" * 64
            + " 123 main/binary-arm64/Packages.gz\n"
        )
        self.assertEqual(
            PREPARE_APT.parse_release_sha256(
                release, "main/binary-arm64/Packages.gz"
            ),
            ("c" * 64, 123),
        )

    def _fake_gcloud(self, root: Path) -> tuple[Path, Path, Path]:
        binary = root / "bin"
        binary.mkdir()
        remote = root / "remote"
        log = root / "log"
        script = binary / "gcloud"
        script.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
echo "$*" >> "$FAKE_GCLOUD_LOG"
[ "$1" = storage ]
case "$2" in
  cp)
    [ "${FAKE_GCLOUD_CP_RC:-0}" = 0 ] || exit "$FAKE_GCLOUD_CP_RC"
    args=("$@")
    cp "${args[${#args[@]}-2]}" "$FAKE_GCLOUD_REMOTE"
    ;;
  cat) cat "$FAKE_GCLOUD_REMOTE" ;;
  *) exit 2 ;;
esac
""",
            encoding="utf-8",
        )
        script.chmod(0o755)
        return binary, remote, log

    def test_generation_zero_cas_accepts_only_identical_concurrent_writer(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            binary, remote, log = self._fake_gcloud(root)
            source = root / "reservation.json"
            source.write_text('{"releaseSequence":2}\n', encoding="utf-8")
            environment = {
                **os.environ,
                "PATH": f"{binary}:{os.environ['PATH']}",
                "FAKE_GCLOUD_REMOTE": str(remote),
                "FAKE_GCLOUD_LOG": str(log),
                "FAKE_GCLOUD_CP_RC": "0",
            }
            command = [
                str(ROOT / "packaging/release/publish-immutable-gcs-file.sh"),
                str(source),
                "apt.plaidai.io",
                "releases/reservations/iq9075/2.json",
            ]
            subprocess.run(command, check=True, capture_output=True, env=environment)
            self.assertEqual(remote.read_bytes(), source.read_bytes())
            self.assertIn("--if-generation-match=0", log.read_text(encoding="utf-8"))
            environment["FAKE_GCLOUD_CP_RC"] = "1"
            subprocess.run(command, check=True, capture_output=True, env=environment)
            remote.write_text("different\n", encoding="utf-8")
            failed = subprocess.run(command, check=False, capture_output=True, env=environment)
            self.assertNotEqual(failed.returncode, 0)


class SettingsPolicyTest(unittest.TestCase):
    def test_required_ruleset_matching_is_exact(self) -> None:
        rulesets = [
            {
                "target": "tag",
                "enforcement": "active",
                "conditions": {"ref_name": {"include": ["refs/tags/v*"]}},
                "rules": [
                    {"type": "creation"},
                    {"type": "deletion"},
                    {"type": "non_fast_forward"},
                ],
            }
        ]
        self.assertTrue(
            SETTINGS._ruleset_covers(
                rulesets,
                target="tag",
                include="refs/tags/v*",
                required_rules={"creation", "deletion", "non_fast_forward"},
            )
        )
        rulesets[0]["enforcement"] = "evaluate"
        self.assertFalse(
            SETTINGS._ruleset_covers(
                rulesets,
                target="tag",
                include="refs/tags/v*",
                required_rules={"creation", "deletion", "non_fast_forward"},
            )
        )

    def test_main_ruleset_requires_real_agent_release_gate_context(self) -> None:
        rulesets = [
            {
                "target": "branch",
                "enforcement": "active",
                "conditions": {"ref_name": {"include": ["refs/heads/main"]}},
                "rules": [
                    {"type": "deletion"},
                    {"type": "non_fast_forward"},
                    {"type": "pull_request"},
                    {
                        "type": "required_status_checks",
                        "parameters": {
                            "required_status_checks": [
                                {"context": "agent-release-gate"}
                            ]
                        },
                    },
                ],
            }
        ]
        arguments = {
            "target": "branch",
            "include": "refs/heads/main",
            "required_rules": {
                "deletion",
                "non_fast_forward",
                "pull_request",
                "required_status_checks",
            },
            "required_status_context": "agent-release-gate",
        }
        self.assertTrue(SETTINGS._ruleset_covers(rulesets, **arguments))
        rulesets[0]["rules"][-1]["parameters"]["required_status_checks"][0][
            "context"
        ] = "nonexistent-check"
        self.assertFalse(SETTINGS._ruleset_covers(rulesets, **arguments))

    def test_short_lived_settings_attestation_binds_policy_and_expiry(self) -> None:
        settings_verifier = (
            ROOT / "packaging/release/verify-github-release-settings.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '"RELEASE_TRUSTED_PUBLISHER_SHA": trusted_publisher_sha',
            settings_verifier,
        )
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            policy = ROOT / "packaging/release/release-security-policy.json"
            now = dt.datetime(2026, 9, 2, 0, 0, tzinfo=dt.timezone.utc)
            attestation = {
                "schemaVersion": 1,
                "kind": "nuvion-release-settings-attestation",
                "repository": "plaid-ai/NUV-AGENT",
                "trustedPublisherSha": "a" * 40,
                "policySha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
                "verifiedAt": "2026-09-02T00:00:00Z",
                "expiresAt": "2026-09-03T00:00:00Z",
                "settings": {
                    "defaultBranch": "main",
                    "secretScopesChecked": True,
                    "status": "VERIFIED",
                },
            }
            path = root / "attestation.json"
            path.write_text(
                json.dumps(attestation, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            signature = root / "attestation.json.asc"
            signature.write_text("test-signature\n", encoding="utf-8")
            with mock.patch.object(
                SETTINGS_ATTESTATION,
                "_verify_signature",
                return_value="9A07D327F3ADF6F452A4BF0055E5CAF706571888",
            ):
                result = SETTINGS_ATTESTATION.verify_attestation(
                    attestation_path=path,
                    signature_path=signature,
                    policy_path=policy,
                    signer_directory=ROOT / "packaging/release/trusted-tag-signers",
                    repository="plaid-ai/NUV-AGENT",
                    trusted_publisher_sha="a" * 40,
                    now=now,
                )
                self.assertEqual(result["status"], "VERIFIED")
                self.assertEqual(result["trustedPublisherSha"], "a" * 40)
                with self.assertRaises(SETTINGS_ATTESTATION.AttestationError):
                    SETTINGS_ATTESTATION.verify_attestation(
                        attestation_path=path,
                        signature_path=signature,
                        policy_path=policy,
                        signer_directory=ROOT / "packaging/release/trusted-tag-signers",
                        repository="plaid-ai/NUV-AGENT",
                        trusted_publisher_sha="b" * 40,
                        now=now,
                    )
                with self.assertRaises(SETTINGS_ATTESTATION.AttestationError):
                    SETTINGS_ATTESTATION.verify_attestation(
                        attestation_path=path,
                        signature_path=signature,
                        policy_path=policy,
                        signer_directory=ROOT / "packaging/release/trusted-tag-signers",
                        repository="plaid-ai/NUV-AGENT",
                        trusted_publisher_sha="a" * 40,
                        now=now + dt.timedelta(days=1),
                    )

    @unittest.skipUnless(shutil.which("gpg"), "gpg is required")
    def test_settings_attestation_requires_allowlisted_gpg_signature(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            gpg_home = root / "gpg"
            gpg_home.mkdir(mode=0o700)
            environment = {**os.environ, "GNUPGHOME": str(gpg_home)}
            subprocess.run(
                [
                    "gpg",
                    "--batch",
                    "--passphrase",
                    "",
                    "--quick-generate-key",
                    "Settings Auditor <settings@example.invalid>",
                    "ed25519",
                    "cert",
                    "1d",
                ],
                check=True,
                capture_output=True,
                env=environment,
            )
            listing = subprocess.check_output(
                ["gpg", "--batch", "--with-colons", "--list-keys"],
                text=True,
                env=environment,
            )
            fingerprint = next(
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
                    fingerprint,
                    "ed25519",
                    "sign",
                    "1d",
                ],
                check=True,
                capture_output=True,
                env=environment,
            )
            policy_payload = json.loads(
                (ROOT / "packaging/release/release-security-policy.json").read_text(
                    encoding="utf-8"
                )
            )
            policy_payload["trustedTagSignerFingerprints"] = [fingerprint]
            policy = root / "policy.json"
            policy.write_text(
                json.dumps(policy_payload, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            attestation_payload = {
                "schemaVersion": 1,
                "kind": "nuvion-release-settings-attestation",
                "repository": "plaid-ai/NUV-AGENT",
                "trustedPublisherSha": "a" * 40,
                "policySha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
                "verifiedAt": "2026-09-02T00:00:00Z",
                "expiresAt": "2026-09-02T12:00:00Z",
                "settings": {
                    "defaultBranch": "main",
                    "secretScopesChecked": True,
                    "status": "VERIFIED",
                },
            }
            attestation = root / "attestation.json"
            attestation.write_text(
                json.dumps(attestation_payload, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            signature = root / "attestation.json.asc"
            subprocess.run(
                [
                    "gpg",
                    "--batch",
                    "--armor",
                    "--detach-sign",
                    "--local-user",
                    fingerprint,
                    "--output",
                    str(signature),
                    str(attestation),
                ],
                check=True,
                capture_output=True,
                env=environment,
            )
            signers = root / "signers"
            signers.mkdir()
            (signers / "auditor.asc").write_bytes(
                subprocess.check_output(
                    ["gpg", "--batch", "--armor", "--export", fingerprint],
                    env=environment,
                )
            )
            result = SETTINGS_ATTESTATION.verify_attestation(
                attestation_path=attestation,
                signature_path=signature,
                policy_path=policy,
                signer_directory=signers,
                repository="plaid-ai/NUV-AGENT",
                trusted_publisher_sha="a" * 40,
                now=dt.datetime(2026, 9, 2, 1, 0, tzinfo=dt.timezone.utc),
            )
            self.assertEqual(result["signerFingerprint"], fingerprint)

    def test_ruleset_bypass_is_exact_release_admin_team(self) -> None:
        rulesets = [
            {
                "target": "tag",
                "enforcement": "active",
                "conditions": {"ref_name": {"include": ["refs/tags/v*"]}},
                "rules": [
                    {"type": "creation"},
                    {"type": "deletion"},
                    {"type": "non_fast_forward"},
                ],
                "bypass_actors": [
                    {
                        "actor_id": 16128529,
                        "actor_type": "Team",
                        "bypass_mode": "always",
                    }
                ],
            }
        ]
        arguments = {
            "target": "tag",
            "include": "refs/tags/v*",
            "required_rules": {"creation", "deletion", "non_fast_forward"},
            "required_bypass_team_id": 16128529,
            "required_bypass_mode": "always",
        }
        self.assertTrue(SETTINGS._ruleset_covers(rulesets, **arguments))
        rulesets[0]["bypass_actors"].append(
            {"actor_id": 1, "actor_type": "User", "bypass_mode": "always"}
        )
        self.assertFalse(SETTINGS._ruleset_covers(rulesets, **arguments))


class ImmutableGitHubReleaseTest(unittest.TestCase):
    class FakeApi:
        repository = "plaid-ai/NUV-AGENT"
        token = "test-token"

        def __init__(self, release: dict[str, object]) -> None:
            self.value = release

        def release(self, tag: str):
            return self.value

        def request(self, method: str, path: str, payload=None):
            if path.endswith("/immutable-releases"):
                return {"enabled": True, "enforced_by_owner": False}
            if method == "PATCH":
                self.value["draft"] = False
                self.value["immutable"] = True
                return self.value
            raise AssertionError((method, path, payload))

    @staticmethod
    def _release(*, draft: bool, immutable: bool) -> dict[str, object]:
        return {
            "id": 123,
            "tag_name": "v0.1.121",
            "draft": draft,
            "immutable": immutable,
            "assets": [],
        }

    def test_finalize_uploads_all_assets_before_immutable_publication(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            paths = []
            for name in ("nuv_agent-0.1.121.tar.gz", "release-bom.json", "promotion.json"):
                path = root / name
                path.write_bytes(name.encode("ascii"))
                paths.append(path)
            api = self.FakeApi(self._release(draft=True, immutable=False))

            def upload(fake_api, *, tag, local):
                fake_api.value["assets"].append(
                    {
                        "name": local["name"],
                        "size": local["size"],
                        "digest": local["digest"],
                        "state": "uploaded",
                    }
                )

            with mock.patch.object(GITHUB_RELEASE, "_upload_asset", side_effect=upload):
                result = GITHUB_RELEASE.publish_release(
                    api=api,
                    tag="v0.1.121",
                    component_sha="a" * 40,
                    phase="finalize",
                    asset_paths=paths,
                    allow_legacy_mutable=False,
                )
            self.assertFalse(result["draft"])
            self.assertTrue(result["immutable"])
            self.assertEqual(len(result["assets"]), 3)

    def test_mutable_published_release_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            asset = Path(raw_root) / "asset.tar.gz"
            asset.write_bytes(b"asset")
            api = self.FakeApi(self._release(draft=False, immutable=False))
            with self.assertRaises(GITHUB_RELEASE.GitHubReleaseError):
                GITHUB_RELEASE.publish_release(
                    api=api,
                    tag="v0.1.121",
                    component_sha="a" * 40,
                    phase="stage",
                    asset_paths=[asset],
                    allow_legacy_mutable=False,
                )


if __name__ == "__main__":
    unittest.main()
