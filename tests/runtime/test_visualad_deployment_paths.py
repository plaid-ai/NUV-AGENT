"""Exercise launcher setup only; never run Python, camera, model or systemd."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "packaging/dev/run-iq9075-visualad-htp-demo.sh"
DEPLOYMENT_ROOT = "NUVION_VISUALAD_DEPLOYMENT_ROOT"
HTP_STATE = "NUVION_VISUALAD_DEPLOYMENT_HTP_STATE_DIR"
SETTINGS = "NUVION_VISUALAD_DEPLOYMENT_SETTINGS_DIR"


class VisualADDeploymentPathTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.deployments = self.root / "deployments"
        self.states = self.root / "states"
        self.deployments.mkdir()
        self.states.mkdir()
        self.default = self._deployment("20260910-visualad-htp")
        (self.states / "visualad-htp").mkdir()
        (self.states / "visualad-htp-settings").mkdir()

    def _deployment(self, name):
        root = self.deployments / name
        for relative in (
            "src/nuvion_app/runtime/visualad_htp.py",
            "model/manifest.json",
            "python/lib/python3.12/site-packages/onnxruntime_qnn/libQnnHtp.so",
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        return root

    def _run(self, overrides=None):
        # Execute the actual shell setup, replacing only fixed test roots. The
        # Python exec boundary is excluded, so optional imports never execute.
        source = LAUNCHER.read_text(encoding="utf-8")
        setup, marker, _python = source.partition("\nexec /usr/bin/python3 -s -c ")
        self.assertTrue(marker, "launcher's no-execution test boundary moved")
        setup = setup.replace("/opt/nuvion-demo", str(self.deployments)).replace(
            "/var/lib/nuv-agent", str(self.states)
        )
        output_names = (
            "demo_root",
            "NUVION_VISUALAD_HTP_STATE_DIR",
            "NUVION_SETTINGS_STATE_DIR",
            "PYTHONPATH",
            "LD_LIBRARY_PATH",
            "ADSP_LIBRARY_PATH",
            "NUVION_FLEET_COMMAND_ENABLED",
            "NUVION_ZERO_SHOT_SAMPLE_SEC",
            "NUVION_ZSAD_BACKEND",
            "NUVION_VISUALAD_HTP_MANIFEST",
            "NUVION_VISUALAD_HTP_MANIFEST_SHA256",
            "NUVION_VISUALAD_THRESHOLD",
            "NUVION_MODEL_POINTER",
            "NUVION_MODEL_DIGEST",
        )
        output = "\n".join(f'printf "%s\\n" "${name}"' for name in output_names)
        environment = {
            "PATH": os.environ.get("PATH", os.defpath),
            "NUVION_VISUALAD_HTP_MANIFEST_SHA256": "a" * 64,
            "NUVION_MODEL_DIGEST": "",
            # The real unit injects this; existing demo isolation must survive.
            "NUVION_SETTINGS_STATE_DIR": "/outside/original-fleet-settings",
            **(overrides or {}),
        }
        result = subprocess.run(
            ["bash", "-s"],
            input='hostname() { printf "iq9075\\n"; }\n' + setup + "\n" + output,
            env=environment,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
        values = dict(zip(output_names, result.stdout.splitlines()))
        return result, values

    def test_defaults_preserve_demo_isolation_and_inference_contract(self):
        result, values = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(values["demo_root"], str(self.default))
        self.assertEqual(
            values["NUVION_VISUALAD_HTP_STATE_DIR"], str(self.states / "visualad-htp")
        )
        self.assertEqual(
            values["NUVION_SETTINGS_STATE_DIR"],
            str(self.states / "visualad-htp-settings"),
        )
        self.assertEqual(values["NUVION_FLEET_COMMAND_ENABLED"], "false")
        self.assertEqual(values["NUVION_ZERO_SHOT_SAMPLE_SEC"], "0")
        self.assertEqual(values["NUVION_ZSAD_BACKEND"], "visualad_htp")
        self.assertEqual(values["NUVION_VISUALAD_HTP_MANIFEST_SHA256"], "a" * 64)
        self.assertEqual(values["NUVION_VISUALAD_THRESHOLD"], "0.0")

    def test_scoped_overrides_rebind_source_model_sdk_and_both_state_paths(self):
        deployment = self._deployment("integration-source")
        htp, settings = self.states / "new-htp", self.states / "settings"
        htp.mkdir()
        settings.mkdir()
        result, values = self._run(
            {
                DEPLOYMENT_ROOT: str(deployment),
                HTP_STATE: str(htp),
                SETTINGS: str(settings),
            }
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(values["demo_root"], str(deployment))
        self.assertEqual(values["NUVION_VISUALAD_HTP_STATE_DIR"], str(htp))
        self.assertEqual(values["NUVION_SETTINGS_STATE_DIR"], str(settings))
        self.assertEqual(
            values["NUVION_VISUALAD_HTP_MANIFEST"],
            str(deployment / "model/manifest.json"),
        )
        self.assertTrue(values["PYTHONPATH"].startswith(str(deployment / "src") + ":"))
        bundle = str(deployment / "python/lib/python3.12/site-packages/onnxruntime_qnn")
        self.assertEqual(values["LD_LIBRARY_PATH"], bundle)
        self.assertEqual(values["ADSP_LIBRARY_PATH"], bundle)

    def test_invalid_or_noncanonical_overrides_fail_before_runtime_execution(self):
        outside = self.root / "outside"
        outside.mkdir()
        for variable, scope, existing in (
            (DEPLOYMENT_ROOT, self.deployments, self.default),
            (HTP_STATE, self.states, self.states / "visualad-htp"),
            (SETTINGS, self.states, self.states / "visualad-htp-settings"),
        ):
            file = scope / "not-directory"
            file.touch(exist_ok=True)
            link = scope / (variable + "-link")
            link.symlink_to(existing, target_is_directory=True)
            outside_link = scope / (variable + "-outside")
            outside_link.symlink_to(outside, target_is_directory=True)
            child = existing / "child"
            child.mkdir(exist_ok=True)
            for value in (
                "",
                "relative",
                "/",
                str(scope),
                str(outside),
                str(scope / "missing"),
                str(scope) + "-sibling",
                str(file),
                str(link),
                str(outside_link),
                str(link / "child"),
                str(existing) + "/",
                str(existing) + "/.",
                str(existing) + "/child/..",
                str(existing) + "//child",
                str(scope) + "/../outside",
            ):
                with self.subTest(variable=variable, value=value):
                    result, values = self._run({variable: value})
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(values)

    def test_fleet_opt_in_preserves_bootstrap_model_and_requires_explicit_dependencies(
        self,
    ):
        opted_in = {
            "NUVION_VISUALAD_HTP_FLEET_ENABLED": "true",
            "NUVION_VISUALAD_HTP_FLEET_STORE": str(self.root / "models"),
            "NUVION_MODEL_POINTER": "visualad/iq9075-htp-demo",
            "NUVION_MODEL_DIGEST": "sha256:" + "b" * 64,
            "NUVION_COMMAND_INBOX_PATH": str(self.root / "inbox.sqlite3"),
        }
        result, values = self._run(opted_in)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(values["NUVION_FLEET_COMMAND_ENABLED"], "true")
        self.assertEqual(values["NUVION_MODEL_DIGEST"], opted_in["NUVION_MODEL_DIGEST"])
        self.assertEqual(
            values["NUVION_MODEL_POINTER"], opted_in["NUVION_MODEL_POINTER"]
        )
        self.assertEqual(values["NUVION_ZERO_SHOT_SAMPLE_SEC"], "0")
        for key in (
            "NUVION_VISUALAD_HTP_FLEET_STORE",
            "NUVION_MODEL_POINTER",
            "NUVION_MODEL_DIGEST",
            "NUVION_COMMAND_INBOX_PATH",
        ):
            with self.subTest(key=key):
                result, _ = self._run({**opted_in, key: ""})
                self.assertNotEqual(result.returncode, 0)
        for flag in ("", "TRUE", "typo", "1"):
            with self.subTest(flag=flag):
                result, _ = self._run(
                    {**opted_in, "NUVION_VISUALAD_HTP_FLEET_ENABLED": flag}
                )
                self.assertNotEqual(result.returncode, 0)

    def test_real_fleet_startup_guard_precedes_pipeline_import_but_not_import_check(
        self,
    ):
        source = LAUNCHER.read_text()
        guard = source.index("prepare_fleet_startup()")
        self.assertLess(
            guard, source.index("from nuvion_app.inference import pipeline")
        )
        self.assertIn(
            'if sys.argv[5] == "true" and sys.argv[4] != "--import-check":', source
        )


if __name__ == "__main__":
    unittest.main()
