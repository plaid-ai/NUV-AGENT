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
        request = (ROOT / ".github" / "workflows" / "release-request.yml").read_text(
            encoding="utf-8"
        )

        source_test = workflow.index("- name: Test source in clean environment")
        build = workflow.index("- name: Build normalized sdist and BOM")
        smoke = workflow.index("- name: Install and smoke-test built sdist")
        publish = workflow.index(
            "- name: Finalize exact immutable GitHub release before live channels"
        )

        self.assertLess(source_test, build)
        self.assertLess(build, smoke)
        self.assertLess(smoke, publish)
        self.assertIn('workflows: ["release-request"]', workflow)
        self.assertNotIn("  push:\n", workflow.split("jobs:", maxsplit=1)[0])
        self.assertIn('on:\n  push:\n    tags:', request)
        self.assertNotIn("${{ secrets.", request)
        self.assertNotIn("contents: write", request)
        self.assertIn("packaging/release/run-isolated-tests.py", workflow)
        self.assertIn("RELEASE_TEST_VENV: ${{ runner.temp }}/", workflow)
        self.assertIn(
            '"$RELEASE_TEST_VENV/bin/python" packaging/release/generate-release-bom.py',
            workflow,
        )
        self.assertIn('pip install --no-cache-dir "$SDIST"', workflow)
        self.assertIn(
            '--sha "$COMPONENT_SHA" --version "$VERSION"', workflow
        )
        self.assertIn("REQUESTED_TAG: ${{ inputs.tag }}", workflow)
        self.assertIn('[[ "$TAG" =~ ^v[0-9]+\\.[0-9]+\\.[0-9]+$ ]]', workflow)
        self.assertIn("verify-release-source.py", workflow)
        self.assertIn("verify-release-settings-attestation.py", workflow)
        self.assertIn("trusted_publisher_sha", workflow)
        self.assertIn('path: publisher', workflow)
        self.assertIn('path: release-source', workflow)
        self.assertIn("publish-github-release.py", workflow)
        self.assertNotIn("--phase stage", workflow)
        self.assertIn("--phase finalize", workflow)
        self.assertNotIn("softprops/action-gh-release", workflow)
        self.assertIn("if git diff --cached --quiet; then", workflow)
        self.assertIn("rerun is idempotent", workflow)
        self.assertIn("group: release-publisher-global", workflow)
        self.assertIn("cancel-in-progress: false", workflow)
        self.assertIn("normalize-sdist.py", workflow)
        self.assertIn("requirements-release-build.txt", workflow)
        self.assertIn("--sdist --no-isolation --outdir dist", workflow)
        self.assertIn('SOURCE_DATE_EPOCH=$(git show -s --format=%ct "$COMPONENT_SHA")', workflow)
        self.assertGreaterEqual(workflow.count('--built-at "$BUILT_AT"'), 2)
        self.assertGreaterEqual(workflow.count("stamp-build-info.py"), 2)
        self.assertGreaterEqual(workflow.count("NUV_AGENT_CONFIG: ${{ runner.temp }}/"), 2)

        stamp = (ROOT / "packaging" / "release" / "stamp-build-info.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", stamp)

    def test_release_workflow_publishes_content_addressed_bom_sidecars(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "release-publish.yml").read_text(
            encoding="utf-8"
        )

        self.assertEqual(workflow.count("generate-release-bom.py"), 2)
        self.assertIn("sdist_bom_sha256", workflow)
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
        self.assertIn('publisher/packaging/apt/publish-gcs.sh "$DEB_PATH"', workflow)
        self.assertIn(
            '"$BUNDLE_PATH" "$BOM_PATH" "$BOM_PATH.sig" "$BUNDLE_PATH"',
            workflow,
        )
        self.assertIn("RELEASE_TRUST_DOMAIN=iq9075-dev", workflow)
        self.assertIn("SKIP_APT_PUBLISH=true", workflow)
        self.assertIn("IQ9075_RELEASE_SIGNING_PRIVATE_KEY", workflow)
        self.assertNotIn("secrets.IQ9075_RELEASE_PUBLIC_KEYRING_JSON", workflow)
        self.assertIn("trusted-release-keyrings/iq9075-dev.json", workflow)
        self.assertIn("--signing-private-key-env NUVION_IQ9075_RELEASE_SIGNING_KEY", workflow)
        self.assertIn("build-agent-bundle.sh", workflow)
        ota_build = workflow.split("  iq9075-ota-build:", maxsplit=1)[1].split(
            "  github-release-publish:", maxsplit=1
        )[0]
        apt_publish = workflow.split("  apt-publish:", maxsplit=1)[1].split(
            "  iq9075-ota-publish:", maxsplit=1
        )[0]
        ota_publish = workflow.split("  iq9075-ota-publish:", maxsplit=1)[1]
        self.assertIn("APT_GPG_PRIVATE_KEY", apt_publish)
        self.assertNotIn("IQ9075_RELEASE_SIGNING_PRIVATE_KEY", apt_publish)
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
            "      - name: Publish verified exact bundle with trusted publisher",
            maxsplit=1,
        )
        self.assertIn("RELEASE_SIGNING_PRIVATE_KEY", private_sign_step)
        self.assertNotIn("RELEASE_SIGNING_PRIVATE_KEY", publish_step)
        self.assertIn("BOOTSTRAP_BUNDLE_PATH=\"$BUNDLE_PATH\"", ota_build)
        self.assertIn("deb_sha256", ota_build)
        self.assertIn("plan-iq9075-ota.py", ota_publish)
        self.assertIn("iq9075-ota-global-publisher", ota_publish)
        self.assertLess(
            ota_publish.index("Independently verify latest sequence"),
            ota_publish.index("IQ9075_RELEASE_SIGNING_PRIVATE_KEY"),
        )
        self.assertIn("publish-immutable-gcs-file.sh", ota_publish)
        self.assertIn("generate-release-promotion.py ota", ota_publish)
        self.assertIn("APT_PREVIOUS_DEB_PATH", apt_publish)
        self.assertIn("prepare-apt-rollback.py", apt_publish)

        apt_script = (ROOT / "packaging" / "apt" / "publish-gcs.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("releases/by-bom-sha256/$BOM_DIGEST", apt_script)
        self.assertIn(
            "Refusing to overwrite existing immutable release bytes", apt_script
        )
        self.assertIn("release-bom.json.sig", apt_script)
        self.assertIn('$(basename "$BOM_ARTIFACT_PATH")', apt_script)
        self.assertIn('gcloud storage cat "gs://$BUCKET/$relative_path" | cmp -s', apt_script)
        self.assertIn("--if-generation-match=0", apt_script)
        self.assertNotIn("gsutil", apt_script)
        self.assertIn("-acquire-by-hash", apt_script)
        self.assertIn("Acquire-By-Hash: yes", apt_script)
        self.assertIn("APT_BY_HASH_PATHS", apt_script)
        self.assertIn("APT_DISCOVERY_PATH", apt_script)
        self.assertLess(
            apt_script.index('for apt_metadata in "${APT_MUTABLE_METADATA_PATHS[@]}"'),
            apt_script.index('upload_mutable_metadata "$APT_DISCOVERY_PATH"'),
        )
        self.assertNotIn("setmeta", apt_script)
        self.assertIn("RELEASE_TRUST_DOMAIN is required", apt_script)
        self.assertIn('aptly -config="$APTLY_CONFIG" repo add "$REPO_NAME" "$ROLLBACK_DEB_PATH"', apt_script)

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
        self.assertIn(
            "cryptography==50.0.0 --hash=sha256:07949c449a1abcf60d1ee6e88956d89404c7df3c8258f46589e912988e551987",
            lock,
        )
        self.assertIn("setuptools==83.0.0", lock)
        self.assertIn("wheel==0.46.2", lock)
        build_lock = (
            ROOT / "packaging/release/requirements-release-build.txt"
        ).read_text(encoding="utf-8")
        self.assertIn("build==1.6.0 --hash=sha256:", build_lock)
        self.assertIn("setuptools==83.0.0 --hash=sha256:", build_lock)
        self.assertIn("wheel==0.46.2 --hash=sha256:", build_lock)

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
            "cryptography-50.0.0-cp311-abi3-macosx_11_0_arm64.whl", homebrew
        )
        self.assertIn("cffi-2.0.0-cp314-cp314-macosx_11_0_arm64.whl", homebrew)
        for artifact in (
            "transformers-5.16.1-py3-none-any.whl",
            "huggingface_hub-1.29.0-py3-none-any.whl",
            "hf_xet-1.6.0-cp38-abi3-macosx_11_0_arm64.whl",
            "tokenizers-0.23.1-cp310-abi3-macosx_11_0_arm64.whl",
            "safetensors-0.8.0-cp310-abi3-macosx_11_0_arm64.whl",
            "packaging-26.3-py3-none-any.whl",
            "setuptools-83.0.0-py3-none-any.whl",
            "wheel-0.46.2-py3-none-any.whl",
            "typer-0.27.2-py3-none-any.whl",
            "rich-15.0.0-py3-none-any.whl",
            "annotated_doc-0.0.5-py3-none-any.whl",
        ):
            self.assertIn(artifact, homebrew)
        self.assertIn('"--no-deps", "--no-build-isolation"', homebrew)

        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        zsad = (ROOT / "nuvion_app/inference/requirements-zsad.txt").read_text(
            encoding="utf-8"
        )
        for requirement in (
            "torch==2.10.0",
            "numpy==2.4.2",
            "transformers==5.16.1",
            "huggingface-hub==1.29.0",
            "hf-xet==1.6.0",
            "tokenizers==0.23.1",
            "safetensors==0.8.0",
            "Pillow==12.1.0",
            "sentencepiece==0.2.2",
            "protobuf==7.36.1",
        ):
            self.assertIn(requirement, pyproject)
            self.assertIn(requirement, zsad)
        iq_lock = (
            ROOT / "packaging/release/requirements-agent-bundle-arm64.txt"
        ).read_text(encoding="utf-8")
        self.assertNotIn("transformers", iq_lock)
        self.assertNotIn("torch", iq_lock)

    def test_macos_arm64_formula_and_fixed_revision_mps_gate_is_fail_closed(self) -> None:
        gate = (ROOT / ".github/workflows/agent-release-gate.yml").read_text(
            encoding="utf-8"
        )
        macos = gate.split(
            "  macos-arm64-release-prerequisite:", maxsplit=1
        )[1].split("  agent-release-gate:", maxsplit=1)[0]
        self.assertIn("runs-on: macos-14", macos)
        self.assertIn('brew tap-new --no-git "$TAP_NAME"', macos)
        self.assertIn('brew trust --formula "$FORMULA_NAME"', macos)
        self.assertIn(
            "brew untrust --formula nuvion/release-gate/nuv-agent", macos
        )
        self.assertIn('$TAP_ROOT/Formula/nuv-agent.rb', macos)
        self.assertNotIn(
            'brew install --formula --build-from-source "${RUNNER_TEMP}/nuv-agent.rb"',
            macos,
        )
        self.assertIn("brew install --formula --build-from-source", macos)
        self.assertIn('platform.machine() == "arm64"', macos)
        self.assertIn("torch.backends.mps.is_available()", macos)
        self.assertEqual(
            macos.count("PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0"), 1
        )
        self.assertIn(
            'os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.0"', macos
        )
        self.assertIn("detector._inference_dtype == torch.float16", macos)
        self.assertIn("detector._model.text_model is None", macos)
        self.assertIn(
            "torch.mps.current_allocated_memory() < 512 * 1024**2", macos
        )
        self.assertIn('PYTHONNOUSERSITE=1 "$FORMULA_PYTHON" -I -', macos)
        self.assertEqual(
            macos.count('PYTHONNOUSERSITE=1 "$FORMULA_PYTHON" -I -'), 2
        )
        self.assertIn('package_path.is_relative_to(formula_prefix / "libexec")', macos)
        self.assertIn("reference_outputs.logits_per_image", macos)
        self.assertIn("observed_scores, reference_scores", macos)
        self.assertIn('REFERENCE_PATH="${RUNNER_TEMP}/siglip-reference-v1.json"', macos)
        self.assertIn("os.O_WRONLY | os.O_CREAT | os.O_EXCL", macos)
        self.assertIn('stat.S_IMODE(reference_stat.st_mode) == 0o600', macos)
        self.assertIn('type(reference["schemaVersion"]) is int', macos)
        self.assertIn("and 0.0 <= score <= 1.0", macos)
        reference = macos.index("reference_outputs.logits_per_image")
        post_reference_purge = macos.index("sudo purge", reference)
        mps_detector = macos.index("detector = ZeroShotAnomalyDetector")
        self.assertLess(reference, post_reference_purge)
        self.assertLess(post_reference_purge, mps_detector)
        self.assertIn("for _ in range(16):", macos)
        self.assertIn("stable_bytes + 16 * 1024**2", macos)
        self.assertIn("sudo purge", macos)
        zero_shot = (
            ROOT / "nuvion_app" / "inference" / "zero_shot.py"
        ).read_text(encoding="utf-8")
        self.assertIn("low_cpu_mem_usage=True", zero_shot)
        self.assertIn("dtype=self._inference_dtype", zero_shot)
        self.assertIn(
            "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2", macos
        )
        self.assertIn("local_files_only=True", macos)
        self.assertIn("HF_HUB_OFFLINE", macos)
        self.assertIn("ZeroShotAnomalyDetector", macos)
        for identity in (
            '"torch": "2.10.0"',
            '"numpy": "2.4.2"',
            '"transformers": "5.16.1"',
            '"huggingface-hub": "1.29.0"',
            '"hf-xet": "1.6.0"',
            '"tokenizers": "0.23.1"',
            '"safetensors": "0.8.0"',
            '"Pillow": "12.1.0"',
            '"sentencepiece": "0.2.2"',
            '"protobuf": "7.36.1"',
        ):
            self.assertIn(identity, macos)
        aggregator = gate.split("  agent-release-gate:", maxsplit=1)[1]
        self.assertIn("macos-arm64-release-prerequisite", aggregator)
        self.assertIn("needs.macos-arm64-release-prerequisite.result", aggregator)

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
