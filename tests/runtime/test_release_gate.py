from __future__ import annotations

import re
import unittest
from pathlib import Path

from nuvion_app import build_info

ROOT = Path(__file__).resolve().parents[2]


class ReleaseGateTest(unittest.TestCase):
    def test_release_tests_source_and_installed_sdist_before_publish(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "release-publish.yml").read_text(
            encoding="utf-8"
        )

        source_test = workflow.index("- name: Test source in clean environment")
        build = workflow.index("- name: Build sdist")
        smoke = workflow.index("- name: Install and smoke-test built sdist")
        publish = workflow.index("- name: Create GitHub release")

        self.assertLess(source_test, build)
        self.assertLess(build, smoke)
        self.assertLess(smoke, publish)
        self.assertIn("packaging/release/run-isolated-tests.py", workflow)
        self.assertIn('pip install --no-cache-dir "$TARBALL"', workflow)
        self.assertIn(
            'stamp-build-info.py --sha "$COMPONENT_SHA" --version "$VERSION"', workflow
        )
        self.assertIn("REQUESTED_TAG: ${{ inputs.tag }}", workflow)
        self.assertIn('[[ ! "$TAG" =~ ^v[0-9]+\\.[0-9]+\\.[0-9]+$ ]]', workflow)
        self.assertNotIn('TAG="${{ inputs.tag }}"', workflow)
        self.assertIn(
            'COMPONENT_SHA=$(git rev-parse "refs/tags/$TAG^{commit}")', workflow
        )
        self.assertIn('git rev-parse "refs/tags/$TAG^{commit}"', workflow)
        self.assertIn("Release checkout must be clean before stamping", workflow)
        self.assertIn("Release tests modified the source tree", workflow)
        self.assertIn("APT checkout does not match the release component SHA", workflow)
        self.assertIn("overwrite_files: false", workflow)
        self.assertIn("verify-github-release-assets.py", workflow)
        self.assertIn("if git diff --cached --quiet; then", workflow)
        self.assertIn("rerun is idempotent", workflow)
        self.assertIn("group: release-${{ inputs.tag || github.ref_name }}", workflow)
        self.assertIn("cancel-in-progress: false", workflow)
        self.assertIn("normalize-sdist.py", workflow)
        self.assertIn('SOURCE_DATE_EPOCH=$(git show -s --format=%ct "$COMPONENT_SHA")', workflow)
        self.assertEqual(workflow.count('--built-at "$BUILT_AT"'), 2)
        self.assertGreaterEqual(workflow.count("stamp-build-info.py"), 2)
        self.assertGreaterEqual(
            workflow.count(
                "NUV_AGENT_CONFIG: ${{ runner.temp }}/nuv-agent-release-test-"
            ),
            2,
        )

        stamp = (ROOT / "packaging" / "release" / "stamp-build-info.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", stamp)

    def test_release_workflow_publishes_content_addressed_bom_sidecars(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "release-publish.yml").read_text(
            encoding="utf-8"
        )

        self.assertEqual(workflow.count("generate-release-bom.py"), 2)
        self.assertIn("${{ steps.sdist_bom.outputs.path }}", workflow)
        self.assertIn(
            '--target "IQ9075_DEV:iq9075_dev:QCS9075-EVK:aarch64"',
            workflow,
        )
        for unsupported_target in (
            "rpi5_deepx_dx_m1",
            "ventuno_q",
            "jetson_orin_nx",
        ):
            self.assertNotIn(unsupported_target, workflow)
        self.assertIn("--platform-profile macos_dev", workflow)
        self.assertIn('--component-sha "$COMPONENT_SHA"', workflow)
        self.assertIn("--artifact-kind agent-bundle", workflow)
        self.assertIn('packaging/apt/publish-gcs.sh "$DEB_PATH"', workflow)
        self.assertIn(
            '"$BUNDLE_PATH" "$BOM_PATH" "$SIGNATURE_PATH" "$BUNDLE_PATH"',
            workflow,
        )
        self.assertIn("RELEASE_TRUST_DOMAIN=iq9075-dev", workflow)
        self.assertIn("SKIP_APT_PUBLISH=true", workflow)
        self.assertIn("IQ9075_RELEASE_SIGNING_PRIVATE_KEY", workflow)
        self.assertIn("IQ9075_RELEASE_PUBLIC_KEYRING_JSON", workflow)
        self.assertIn("--signing-private-key-env NUVION_IQ9075_RELEASE_SIGNING_KEY", workflow)
        self.assertIn("build-agent-bundle.sh", workflow)
        apt_and_release, ota_jobs = workflow.split(
            "  iq9075-ota-build:", maxsplit=1
        )
        ota_build, ota_publish = ota_jobs.split(
            "  iq9075-ota-publish:", maxsplit=1
        )
        self.assertIn("APT_GPG_PRIVATE_KEY", apt_and_release)
        self.assertNotIn("IQ9075_RELEASE_SIGNING_PRIVATE_KEY", apt_and_release)
        self.assertNotIn("IQ9075_RELEASE_SIGNING_PRIVATE_KEY", ota_build)
        self.assertNotIn("GCP_SA_KEY", ota_build)
        self.assertRegex(ota_build, r"actions/upload-artifact@[0-9a-f]{40} # v4")
        self.assertIn("IQ9075_RELEASE_SIGNING_PRIVATE_KEY", ota_publish)
        self.assertNotIn("APT_GPG_PRIVATE_KEY", ota_publish)
        self.assertNotIn("build-agent-bundle.sh", ota_publish)
        self.assertNotIn("aptly", ota_publish)
        self.assertIn("python3-cryptography", ota_publish)
        self.assertRegex(ota_publish, r"actions/download-artifact@[0-9a-f]{40} # v4")
        private_sign_step, publish_step = ota_publish.split(
            "      - name: Publish verified exact bundle without signing key",
            maxsplit=1,
        )
        self.assertIn("RELEASE_SIGNING_PRIVATE_KEY", private_sign_step)
        self.assertNotIn("RELEASE_SIGNING_PRIVATE_KEY", publish_step)
        self.assertIn("BOOTSTRAP_BUNDLE_PATH=\"$BUNDLE_PATH\"", ota_build)
        self.assertIn("deb_sha256", ota_build)

        apt_publish = (ROOT / "packaging" / "apt" / "publish-gcs.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("releases/by-bom-sha256/$BOM_DIGEST", apt_publish)
        self.assertIn(
            "Refusing to overwrite existing immutable release bytes", apt_publish
        )
        self.assertIn("release-bom.json.sig", apt_publish)
        self.assertIn('$(basename "$BOM_ARTIFACT_PATH")', apt_publish)
        self.assertIn('gsutil cat "gs://$BUCKET/$relative_path" | cmp -s', apt_publish)
        self.assertIn('cp -n "$published_release" "$remote_path"', apt_publish)
        self.assertIn("-x '^(releases/|pool/)'", apt_publish)
        self.assertNotIn("setmeta", apt_publish)
        self.assertIn("RELEASE_TRUST_DOMAIN is required", apt_publish)

        bundle = (
            ROOT / "packaging" / "release" / "build-agent-bundle.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("python@sha256:", bundle)
        self.assertIn('--target "$site_packages"', bundle)
        self.assertIn('NUVION_SYSTEM_PYTHON:-/usr/bin/python3', bundle)
        self.assertIn("requirements-agent-bundle-arm64.txt", bundle)
        self.assertIn("--no-build-isolation", bundle)
        self.assertGreaterEqual(bundle.count("--require-hashes"), 2)
        self.assertIn("EXPECTED_COMPONENT_SHA", bundle)
        self.assertIn("build_info.COMPONENT_SHA != expected_sha", bundle)
        self.assertIn("package_path.is_relative_to(slot / \"venv\")", bundle)
        self.assertIn("requirements-depthai-arm64.txt", bundle)
        self.assertIn("agent-bundle must not contain symbolic links", bundle)

        lock = (
            ROOT
            / "packaging"
            / "release"
            / "requirements-agent-bundle-arm64.txt"
        ).read_text(encoding="utf-8")
        requirements = [
            line for line in lock.splitlines() if line and not line.startswith("#")
        ]
        self.assertGreaterEqual(len(requirements), 20)
        self.assertTrue(all("==" in line and "--hash=sha256:" in line for line in requirements))

    def test_release_candidate_version_is_consistent(self) -> None:
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(
            r'^version = "([0-9]+\.[0-9]+\.[0-9]+)"$', pyproject, re.MULTILINE
        )
        self.assertIsNotNone(match)
        version = match.group(1)

        deb = (ROOT / "packaging" / "deb" / "build-deb.sh").read_text(encoding="utf-8")
        homebrew = (ROOT / "packaging" / "homebrew" / "nuv-agent.rb").read_text(
            encoding="utf-8"
        )

        self.assertEqual(build_info.AGENT_VERSION, version)
        self.assertIn(f'VERSION="${{VERSION:-{version}}}"', deb)
        self.assertIn(f'version "{version}"', homebrew)

    def test_command_verifier_dependencies_are_pinned_for_homebrew(self) -> None:
        homebrew = (ROOT / "packaging" / "homebrew" / "nuv-agent.rb").read_text(
            encoding="utf-8"
        )

        for resource in ("cryptography", "cffi", "pycparser"):
            self.assertIn(f'resource "{resource}" do', homebrew)
        self.assertIn("depends_on :macos", homebrew)
        self.assertIn("depends_on arch: :arm64", homebrew)
        self.assertIn(
            "cryptography-46.0.4-cp311-abi3-macosx_10_9_universal2.whl", homebrew
        )
        self.assertIn("cffi-2.0.0-cp314-cp314-macosx_11_0_arm64.whl", homebrew)

    def test_debian_bootstrap_is_prebuilt_and_excludes_checkout_source(self) -> None:
        deb = (ROOT / "packaging" / "deb" / "build-deb.sh").read_text(encoding="utf-8")

        self.assertIn("BOOTSTRAP_BUNDLE_PATH", deb)
        self.assertIn("BOOTSTRAP_BUNDLE_PATH is required", deb)
        self.assertIn("DEB_BUILDER_IMAGE=\"ubuntu@sha256:", deb)
        self.assertIn("bootstrap-agent-bundle.sha256", deb)
        self.assertNotIn('SRC_DIR="$PKG_DIR/opt/nuv-agent/src"', deb)
        self.assertNotIn('"$ROOT_DIR/" \\\n', deb)
        self.assertNotIn(".gnupg", deb)
        self.assertNotIn("gha-creds-", deb)

    def test_systemd_persists_both_durable_queues_in_state_directory(self) -> None:
        unit = (ROOT / "packaging" / "systemd" / "nuv-agent.service").read_text(
            encoding="utf-8"
        )

        self.assertIn("User=nuvion", unit)
        self.assertIn("StateDirectory=nuv-agent", unit)
        self.assertIn(
            "Environment=NUVION_EVENT_OUTBOX_PATH=/var/lib/nuv-agent/events.sqlite3",
            unit,
        )
        self.assertIn(
            "Environment=NUVION_COMMAND_INBOX_PATH=/var/lib/nuv-agent/commands.sqlite3",
            unit,
        )
        self.assertIn(
            "Environment=NUVION_SETTINGS_STATE_DIR=/var/lib/nuv-agent/settings",
            unit,
        )
        self.assertIn("Environment=NUVION_SUPERVISOR_RESTART_ENABLED=true", unit)
        self.assertIn("-m nuvion_app.runtime.settings_boot_guard", unit)
        self.assertEqual(unit.count("/opt/nuv-agent/current/venv/bin/python"), 2)
        self.assertNotIn("ExecStartPre=/opt/nuv-agent/venv/bin/python", unit)
        self.assertIn(
            "ExecStartPre=/opt/nuv-agent/current/venv/bin/python -s "
            "-m nuvion_app.runtime.settings_boot_guard",
            unit,
        )
        self.assertNotIn("PermissionsStartOnly", unit)
        self.assertNotIn("/usr/sbin/runuser", unit)
        self.assertNotIn("docker.service", unit)
        self.assertLess(
            unit.index("-m nuvion_app.runtime.settings_boot_guard"),
            unit.index("from nuvion_app.runtime.bootstrap import ensure_ready"),
        )
        self.assertIn("Restart=always", unit)
        self.assertIn("StartLimitIntervalSec=300", unit)
        self.assertIn("StartLimitBurst=3", unit)

        postinst = (ROOT / "packaging" / "deb" / "postinst").read_text(
            encoding="utf-8"
        )
        self.assertIn('readonly BOOTSTRAP_ROOT="$INSTALL_ROOT/bootstrap"', postinst)
        self.assertIn("atomic_slot_link current", postinst)
        self.assertNotIn('readonly NUVION_CURRENT="/opt/nuv-agent/current"', postinst)
        self.assertNotIn('ln -s /opt/nuv-agent "$NUVION_CURRENT"', postinst)


if __name__ == "__main__":
    unittest.main()
