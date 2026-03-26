from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nuvion_app.runtime.config_guard import CURRENT_CONFIG_SCHEMA_VERSION, guard_config


class ConfigGuardTest(unittest.TestCase):
    def test_guard_applies_legacy_migrations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "agent.env"
            config_path.write_text(
                "\n".join(
                    [
                        "NUVION_SERVER_BASE_URL=https://api.example.com",
                        "NUVION_DEVICE_USERNAME=device-1",
                        "NUVION_DEVICE_PASSWORD=secret",
                        "NUVION_TRITON_INPUT=images",
                        "NUVION_TRITON_INPUT_FORMAT=INVALID",
                        "NUVION_MODEL_SOURCE=invalid",
                        "",
                    ]
                )
            )

            report = guard_config(config_path=config_path, apply_fixes=True)

            self.assertTrue(report.ok)
            self.assertEqual(report.values["NUVION_TRITON_INPUT"], "image")
            self.assertEqual(report.values["NUVION_TRITON_INPUT_FORMAT"], "NCHW")
            self.assertEqual(report.values["NUVION_MODEL_SOURCE"], "server")
            self.assertEqual(report.values["NUVION_CONFIG_SCHEMA_VERSION"], CURRENT_CONFIG_SCHEMA_VERSION)
            self.assertGreater(len(report.changed), 0)

    def test_guard_detects_required_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "agent.env"
            config_path.write_text(
                "\n".join(
                    [
                        "NUVION_SERVER_BASE_URL=https://api.example.com",
                        "NUVION_DEVICE_USERNAME=device-1",
                        "NUVION_DEVICE_PASSWORD=***",
                        "",
                    ]
                )
            )
            report = guard_config(config_path=config_path, apply_fixes=True)
            self.assertFalse(report.ok)
            self.assertTrue(any(issue.key == "NUVION_DEVICE_PASSWORD" for issue in report.errors))

    def test_guard_accepts_webrtc_only_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "agent.env"
            config_path.write_text(
                "\n".join(
                    [
                        "NUVION_SERVER_BASE_URL=https://api.example.com",
                        "NUVION_DEVICE_USERNAME=device-1",
                        "NUVION_DEVICE_PASSWORD=secret",
                        "NUVION_ZSAD_BACKEND=none",
                        "",
                    ]
                )
            )
            report = guard_config(config_path=config_path, apply_fixes=True)
            self.assertTrue(report.ok)

    def test_guard_reports_env_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "agent.env"
            config_path.write_text(
                "\n".join(
                    [
                        "NUVION_SERVER_BASE_URL=https://api.example.com",
                        "NUVION_DEVICE_USERNAME=device-1",
                        "NUVION_DEVICE_PASSWORD=secret",
                        "NUVION_TRITON_INPUT=image",
                        "",
                    ]
                )
            )
            with mock.patch.dict(os.environ, {"NUVION_TRITON_INPUT": "images"}, clear=False):
                report = guard_config(config_path=config_path, apply_fixes=False)
            self.assertIn("NUVION_TRITON_INPUT", report.env_overrides)

    def test_guard_accepts_mvtec_demo_defaults_when_demo_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "agent.env"
            config_path.write_text(
                "\n".join(
                    [
                        "NUVION_SERVER_BASE_URL=https://api.example.com",
                        "NUVION_DEVICE_USERNAME=device-1",
                        "NUVION_DEVICE_PASSWORD=secret",
                        "NUVION_DEMO_MODE=true",
                        "",
                    ]
                )
            )
            report = guard_config(config_path=config_path, apply_fixes=True)
            self.assertTrue(report.ok)

    def test_guard_rejects_invalid_mvtec_base_url_when_demo_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "agent.env"
            config_path.write_text(
                "\n".join(
                    [
                        "NUVION_SERVER_BASE_URL=https://api.example.com",
                        "NUVION_DEVICE_USERNAME=device-1",
                        "NUVION_DEVICE_PASSWORD=secret",
                        "NUVION_DEMO_MODE=true",
                        "NUVION_DEMO_MVTEC_BASE_URL=ftp://invalid",
                        "",
                    ]
                )
            )
            report = guard_config(config_path=config_path, apply_fixes=True)
            self.assertFalse(report.ok)
            self.assertTrue(any(issue.key == "NUVION_DEMO_MVTEC_BASE_URL" for issue in report.errors))


if __name__ == "__main__":
    unittest.main()
