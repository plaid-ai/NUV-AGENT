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
        for profile in (
            "rpi5_deepx_dx_m1",
            "ventuno_q",
            "jetson_orin_nx",
            "iq9075_dev",
            "macos_dev",
        ):
            self.assertIn(f"--platform-profile {profile}", workflow)
        self.assertIn('--component-sha "$COMPONENT_SHA"', workflow)
        self.assertIn("--artifact-kind deb", workflow)
        self.assertIn('packaging/apt/publish-gcs.sh "$DEB_PATH" "$BOM_PATH"', workflow)

        apt_publish = (ROOT / "packaging" / "apt" / "publish-gcs.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("releases/by-bom-sha256/$BOM_DIGEST", apt_publish)
        self.assertIn("Refusing to overwrite an existing remote BOM", apt_publish)
        self.assertIn('gsutil cat "gs://$BUCKET/$relative_path" | cmp -s', apt_publish)

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

    def test_debian_source_copy_excludes_local_credentials(self) -> None:
        deb = (ROOT / "packaging" / "deb" / "build-deb.sh").read_text(encoding="utf-8")

        self.assertIn('"$ROOT_DIR/pyproject.toml" "$SRC_DIR/pyproject.toml"', deb)
        self.assertIn('"$ROOT_DIR/README.md" "$SRC_DIR/README.md"', deb)
        self.assertIn('"$ROOT_DIR/nuvion_app/"', deb)
        self.assertIn('"$SRC_DIR/nuvion_app/"', deb)
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
            "ExecStartPre=/usr/sbin/runuser -u nuvion -- "
            "/opt/nuv-agent/current/venv/bin/python -s "
            "-m nuvion_app.runtime.settings_boot_guard",
            unit,
        )
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
        self.assertIn('readonly NUVION_CURRENT="/opt/nuv-agent/current"', postinst)
        self.assertIn('ln -s /opt/nuv-agent "$NUVION_CURRENT"', postinst)


if __name__ == "__main__":
    unittest.main()
