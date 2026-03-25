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
                        "NUVION_RTP_REMOTE_IP=1.2.3.4",
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
                        "NUVION_RTP_REMOTE_IP=1.2.3.4",
                        "",
                    ]
                )
            )
            report = guard_config(config_path=config_path, apply_fixes=True)
            self.assertFalse(report.ok)
            self.assertTrue(any(issue.key == "NUVION_DEVICE_PASSWORD" for issue in report.errors))

    def test_guard_accepts_missing_rtp_target_in_webrtc_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "agent.env"
            config_path.write_text(
                "\n".join(
                    [
                        "NUVION_SERVER_BASE_URL=https://api.example.com",
                        "NUVION_DEVICE_USERNAME=device-1",
                        "NUVION_DEVICE_PASSWORD=secret",
                        "NUVION_UPLINK_MODE=webrtc",
                        "NUVION_RTP_REMOTE_IP=",
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
                        "NUVION_RTP_REMOTE_IP=1.2.3.4",
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
                        "NUVION_RTP_REMOTE_IP=1.2.3.4",
                        "NUVION_DEMO_MODE=true",
                        "NUVION_DEMO_VIDEO_PATH=",
                        "",
                    ]
                )
            )
            report = guard_config(config_path=config_path, apply_fixes=True)
            self.assertTrue(report.ok)

    def test_guard_accepts_demo_video_path_when_demo_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            demo_file = Path(tmp) / "demo.mp4"
            demo_file.write_bytes(b"fake")

            config_path = Path(tmp) / "agent.env"
            config_path.write_text(
                "\n".join(
                    [
                        "NUVION_SERVER_BASE_URL=https://api.example.com",
                        "NUVION_DEVICE_USERNAME=device-1",
                        "NUVION_DEVICE_PASSWORD=secret",
                        "NUVION_RTP_REMOTE_IP=1.2.3.4",
                        "NUVION_DEMO_MODE=true",
                        f"NUVION_DEMO_VIDEO_PATH={demo_file}",
                        "",
                    ]
                )
            )
            report = guard_config(config_path=config_path, apply_fixes=True)
            self.assertTrue(report.ok)

    def test_guard_accepts_fallback_demo_video_path_when_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            demo_file = Path(tmp) / "demo.webm"
            demo_file.write_bytes(b"fake")

            config_path = Path(tmp) / "agent.env"
            config_path.write_text(
                "\n".join(
                    [
                        "NUVION_SERVER_BASE_URL=https://api.example.com",
                        "NUVION_DEVICE_USERNAME=device-1",
                        "NUVION_DEVICE_PASSWORD=secret",
                        "NUVION_RTP_REMOTE_IP=1.2.3.4",
                        "NUVION_DEMO_MODE=true",
                        "NUVION_DEMO_VIDEO_PATH=",
                        "",
                    ]
                )
            )
            with mock.patch.dict(os.environ, {"NUVION_DEMO_VIDEO_FALLBACK_PATHS": str(demo_file)}, clear=False):
                report = guard_config(config_path=config_path, apply_fixes=True)
            self.assertTrue(report.ok)


if __name__ == "__main__":
    unittest.main()
