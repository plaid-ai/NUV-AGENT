from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nuvion_app.runtime.config_guard import (
    CURRENT_CONFIG_SCHEMA_VERSION,
    ensure_runtime_config,
    guard_config,
)


class ConfigGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        # Production intentionally treats process environment as an operator override. Keep this
        # module hermetic when the full unittest suite has imported runtime code that populated
        # NUVION_* values earlier in the same interpreter.
        environment = mock.patch.dict(os.environ, {}, clear=False)
        environment.start()
        self.addCleanup(environment.stop)
        for key in list(os.environ):
            if key == "NUV_AGENT_CONFIG" or key.startswith("NUVION_"):
                os.environ.pop(key, None)

    def test_current_schema_is_version_12_for_depthai_camera_contract(self) -> None:
        self.assertEqual(CURRENT_CONFIG_SCHEMA_VERSION, "12")
        template = (
            Path(__file__).parents[2] / "nuvion_app" / "config_template.env"
        ).read_text(encoding="utf-8")
        self.assertIn("NUVION_CONFIG_SCHEMA_VERSION=12", template)
        example = (Path(__file__).parents[2] / ".env.example").read_text(
            encoding="utf-8"
        )
        self.assertIn("NUVION_CONFIG_SCHEMA_VERSION=12", example)

    def test_guard_normalizes_invalid_outbox_limits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "agent.env"
            config_path.write_text(
                "\n".join(
                    [
                        "NUVION_SERVER_BASE_URL=https://api.example.com",
                        "NUVION_DEVICE_USERNAME=device-1",
                        "NUVION_DEVICE_PASSWORD=secret",
                        "NUVION_ZSAD_BACKEND=none",
                        "NUVION_EVENT_OUTBOX_MAX_ROWS=0",
                        "NUVION_EVENT_OUTBOX_MAX_BYTES=invalid",
                        "NUVION_EVENT_CRITICAL_SAFETY_MAX_BYTES=1",
                        "NUVION_EVENT_OUTBOX_MAX_AGE_SECONDS=0",
                        "NUVION_EVENT_DLQ_MAX_ROWS=-1",
                        "NUVION_EVENT_DLQ_MAX_BYTES=invalid",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            report = guard_config(config_path=config_path, apply_fixes=True)

            self.assertEqual(report.values["NUVION_EVENT_OUTBOX_MAX_ROWS"], "10000")
            self.assertEqual(report.values["NUVION_EVENT_OUTBOX_MAX_BYTES"], "67108864")
            self.assertEqual(
                report.values["NUVION_EVENT_CRITICAL_SAFETY_MAX_BYTES"],
                "67108864",
            )
            self.assertEqual(report.values["NUVION_EVENT_OUTBOX_MAX_AGE_SECONDS"], "2592000")
            self.assertEqual(report.values["NUVION_EVENT_DLQ_MAX_ROWS"], "10000")
            self.assertEqual(report.values["NUVION_EVENT_DLQ_MAX_BYTES"], "67108864")

    def test_guard_normalizes_non_finite_depthai_timeouts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "agent.env"
            config_path.write_text(
                "\n".join(
                    [
                        "NUVION_SERVER_BASE_URL=https://api.example.com",
                        "NUVION_DEVICE_USERNAME=device-1",
                        "NUVION_DEVICE_PASSWORD=secret",
                        "NUVION_ZSAD_BACKEND=none",
                        "NUVION_DEPTHAI_STARTUP_TIMEOUT_SEC=nan",
                        "NUVION_DEPTHAI_READ_TIMEOUT_SEC=inf",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            report = guard_config(config_path=config_path, apply_fixes=True)

            self.assertTrue(report.ok)
            self.assertEqual(report.values["NUVION_DEPTHAI_STARTUP_TIMEOUT_SEC"], "15.0")
            self.assertEqual(report.values["NUVION_DEPTHAI_READ_TIMEOUT_SEC"], "2.0")

    def test_guard_rejects_fractional_depthai_timeout_threshold_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "agent.env"
            config_path.write_text(
                "\n".join(
                    [
                        "NUVION_SERVER_BASE_URL=https://api.example.com",
                        "NUVION_DEVICE_USERNAME=device-1",
                        "NUVION_DEVICE_PASSWORD=secret",
                        "NUVION_ZSAD_BACKEND=none",
                        "NUVION_DEPTHAI_MAX_CONSECUTIVE_TIMEOUTS=3",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            with mock.patch.dict(
                os.environ,
                {"NUVION_DEPTHAI_MAX_CONSECUTIVE_TIMEOUTS": "1.5"},
                clear=False,
            ):
                report = guard_config(config_path=config_path, apply_fixes=True)

            self.assertFalse(report.ok)
            self.assertTrue(
                any(
                    issue.key == "NUVION_DEPTHAI_MAX_CONSECUTIVE_TIMEOUTS"
                    for issue in report.errors
                )
            )

    def test_guard_rejects_conflicting_depthai_device_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "agent.env"
            config_path.write_text(
                "\n".join(
                    [
                        "NUVION_SERVER_BASE_URL=https://api.example.com",
                        "NUVION_DEVICE_USERNAME=device-1",
                        "NUVION_DEVICE_PASSWORD=secret",
                        "NUVION_ZSAD_BACKEND=none",
                        "NUVION_VIDEO_SOURCE=oak:inline-mxid",
                        "NUVION_DEPTHAI_DEVICE_ID=configured-mxid",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            report = guard_config(config_path=config_path, apply_fixes=True)

            self.assertFalse(report.ok)
            self.assertTrue(
                any(
                    issue.key == "NUVION_DEPTHAI_DEVICE_ID"
                    and "conflict" in issue.message
                    for issue in report.errors
                )
            )

    def test_runtime_migration_replaces_dotenv_origin_values_in_current_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "agent.env"
            original_values = {
                "NUVION_CONFIG_SCHEMA_VERSION": "8",
                "NUVION_SERVER_BASE_URL": "https://api.nuvion-dev.plaidai.io",
                "NUVION_MODEL_SERVER_BASE_URL": "https://api.nuvion-dev.plaidai.io",
                "NUVION_DEVICE_USERNAME": "device-1",
                "NUVION_DEVICE_PASSWORD": "secret",
                "NUVION_ZSAD_BACKEND": "none",
            }
            config_path.write_text(
                "\n".join(f"{key}={value}" for key, value in original_values.items()) + "\n",
                encoding="utf-8",
            )

            with mock.patch.dict(os.environ, original_values, clear=True):
                report = ensure_runtime_config(config_path, stage="test", apply_fixes=True)

                self.assertEqual(os.environ["NUVION_CONFIG_SCHEMA_VERSION"], CURRENT_CONFIG_SCHEMA_VERSION)
                self.assertEqual(os.environ["NUVION_SERVER_BASE_URL"], "https://api.nuvion-dev.plaidlabs.ai")
                self.assertNotIn("NUVION_CONFIG_SCHEMA_VERSION", report.env_overrides)
                self.assertNotIn("NUVION_SERVER_BASE_URL", report.env_overrides)

    def test_runtime_migration_preserves_explicit_external_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "agent.env"
            config_path.write_text(
                "\n".join(
                    [
                        "NUVION_CONFIG_SCHEMA_VERSION=8",
                        "NUVION_SERVER_BASE_URL=https://api.nuvion-dev.plaidai.io",
                        "NUVION_DEVICE_USERNAME=device-1",
                        "NUVION_DEVICE_PASSWORD=secret",
                        "NUVION_ZSAD_BACKEND=none",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            environment = {
                "NUVION_CONFIG_SCHEMA_VERSION": "8",
                "NUVION_SERVER_BASE_URL": "https://edge.override.example.com",
                "NUVION_DEVICE_USERNAME": "device-1",
                "NUVION_DEVICE_PASSWORD": "secret",
                "NUVION_ZSAD_BACKEND": "none",
            }

            with mock.patch.dict(os.environ, environment, clear=True):
                report = ensure_runtime_config(config_path, stage="test", apply_fixes=True)

                self.assertEqual(os.environ["NUVION_CONFIG_SCHEMA_VERSION"], CURRENT_CONFIG_SCHEMA_VERSION)
                self.assertEqual(os.environ["NUVION_SERVER_BASE_URL"], "https://edge.override.example.com")
                self.assertIn("NUVION_SERVER_BASE_URL", report.env_overrides)

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

    def test_guard_migrates_legacy_nuvion_hostnames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "agent.env"
            config_path.write_text(
                "\n".join(
                    [
                        "NUVION_CONFIG_SCHEMA_VERSION=8",
                        "NUVION_SERVER_BASE_URL=https://api.nuvion-dev.plaidai.io",
                        "NUVION_MODEL_SERVER_BASE_URL=https://api.nuvion-dev.plaidai.io",
                        "NUVION_CONNECTIVITY_TARGET_HOST=api.nuvion-dev.plaidai.io",
                        "NUVION_DEVICE_USERNAME=device-1",
                        "NUVION_DEVICE_PASSWORD=secret",
                        "NUVION_ZSAD_BACKEND=none",
                        "",
                    ]
                )
            )

            report = guard_config(config_path=config_path, apply_fixes=True)

            self.assertTrue(report.ok)
            self.assertEqual(report.values["NUVION_CONFIG_SCHEMA_VERSION"], CURRENT_CONFIG_SCHEMA_VERSION)
            self.assertEqual(report.values["NUVION_SERVER_BASE_URL"], "https://api.nuvion-dev.plaidlabs.ai")
            self.assertEqual(report.values["NUVION_MODEL_SERVER_BASE_URL"], "https://api.nuvion-dev.plaidlabs.ai")
            self.assertEqual(report.values["NUVION_CONNECTIVITY_TARGET_HOST"], "api.nuvion-dev.plaidlabs.ai")

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

    def test_guard_migrates_legacy_tracking_responsiveness_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "agent.env"
            config_path.write_text(
                "\n".join(
                    [
                        "NUVION_CONFIG_SCHEMA_VERSION=5",
                        "NUVION_SERVER_BASE_URL=https://api.example.com",
                        "NUVION_DEVICE_USERNAME=device-1",
                        "NUVION_DEVICE_PASSWORD=secret",
                        "NUVION_TRACKING_SAMPLE_SEC=0.1",
                        "NUVION_TRACKING_DEADZONE_PCT=0.12",
                        "NUVION_MOTOR_COMMAND_INTERVAL_SEC=0.1",
                        "",
                    ]
                )
            )

            report = guard_config(config_path=config_path, apply_fixes=True)

            self.assertTrue(report.ok)
            self.assertEqual(report.values["NUVION_TRACKING_SAMPLE_SEC"], "0.05")
            self.assertEqual(report.values["NUVION_TRACKING_DEADZONE_PCT"], "0.08")
            self.assertEqual(report.values["NUVION_MOTOR_COMMAND_INTERVAL_SEC"], "0.05")

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

    def test_guard_normalizes_invalid_video_rotation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "agent.env"
            config_path.write_text(
                "\n".join(
                    [
                        "NUVION_SERVER_BASE_URL=https://api.example.com",
                        "NUVION_DEVICE_USERNAME=device-1",
                        "NUVION_DEVICE_PASSWORD=secret",
                        "NUVION_VIDEO_ROTATION=45",
                        "",
                    ]
                )
            )

            report = guard_config(config_path=config_path, apply_fixes=True)

            self.assertTrue(report.ok)
            self.assertEqual(report.values["NUVION_VIDEO_ROTATION"], "0")

    def test_guard_normalizes_invalid_motor_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "agent.env"
            config_path.write_text(
                "\n".join(
                    [
                        "NUVION_SERVER_BASE_URL=https://api.example.com",
                        "NUVION_DEVICE_USERNAME=device-1",
                        "NUVION_DEVICE_PASSWORD=secret",
                        "NUVION_MOTOR_BACKEND=weird",
                        "",
                    ]
                )
            )

            report = guard_config(config_path=config_path, apply_fixes=True)

            self.assertTrue(report.ok)
            self.assertEqual(report.values["NUVION_MOTOR_BACKEND"], "auto")

    def test_guard_normalizes_invalid_face_tracking_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "agent.env"
            config_path.write_text(
                "\n".join(
                    [
                        "NUVION_SERVER_BASE_URL=https://api.example.com",
                        "NUVION_DEVICE_USERNAME=device-1",
                        "NUVION_DEVICE_PASSWORD=secret",
                        "NUVION_FACE_TRACKING_BACKEND=weird",
                        "",
                    ]
                )
            )

            report = guard_config(config_path=config_path, apply_fixes=True)

            self.assertTrue(report.ok)
            self.assertEqual(report.values["NUVION_FACE_TRACKING_BACKEND"], "auto")

    def test_guard_normalizes_invalid_camera_preferences_and_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "agent.env"
            config_path.write_text(
                "\n".join(
                    [
                        "NUVION_SERVER_BASE_URL=https://api.example.com",
                        "NUVION_DEVICE_USERNAME=device-1",
                        "NUVION_DEVICE_PASSWORD=secret",
                        "NUVION_CAMERA_PREFERENCE=sideways",
                        "NUVION_CAMERA_WB_MODE=moonlight",
                        "NUVION_CAMERA_BRIGHTNESS=4",
                        "NUVION_CAMERA_CONTRAST=-1",
                        "NUVION_CAMERA_SATURATION=4",
                        "NUVION_CAMERA_EXPOSURE_COMPENSATION=9",
                        "",
                    ]
                )
            )

            report = guard_config(config_path=config_path, apply_fixes=True)

            self.assertTrue(report.ok)
            self.assertEqual(report.values["NUVION_CAMERA_PREFERENCE"], "auto")
            self.assertEqual(report.values["NUVION_CAMERA_WB_MODE"], "auto")
            self.assertEqual(report.values["NUVION_CAMERA_BRIGHTNESS"], "1.0")
            self.assertEqual(report.values["NUVION_CAMERA_CONTRAST"], "0.0")
            self.assertEqual(report.values["NUVION_CAMERA_SATURATION"], "2.0")
            self.assertEqual(report.values["NUVION_CAMERA_EXPOSURE_COMPENSATION"], "2.0")

    def test_guard_requires_device_credentials_or_access_token_for_model_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "agent.env"
            config_path.write_text(
                "\n".join(
                    [
                        "NUVION_SERVER_BASE_URL=https://api.example.com",
                        "NUVION_DEVICE_USERNAME=device-1",
                        "NUVION_DEVICE_PASSWORD=***",
                        "NUVION_ZSAD_BACKEND=triton",
                        "",
                    ]
                )
            )

            report = guard_config(config_path=config_path, apply_fixes=True)

            self.assertFalse(report.ok)
            self.assertTrue(any("device credential" in issue.message for issue in report.errors))


if __name__ == "__main__":
    unittest.main()
