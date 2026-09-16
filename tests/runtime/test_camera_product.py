from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from nuvion_app.runtime.camera_product import (
    apply_camera_product_preset,
    build_camera_qualification_report,
    camera_product_updates,
)


class CameraProductPresetTest(unittest.TestCase):
    def test_base_preset_preserves_credentials_and_clears_stale_i2c_bus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "agent.env"
            config_path.write_text(
                """NUVION_DEVICE_PASSWORD=keep-me
NUVION_VIDEO_SOURCE=jetson
NUVION_CAMERA_PROFILE=arducam_b0273
NUVION_CAMERA_I2C_BUS=9
CUSTOM_LOCAL_KEY=keep-too
""",
                encoding="utf-8",
            )

            apply_camera_product_preset(config_path, product="base")

            content = config_path.read_text(encoding="utf-8")
            self.assertIn("NUVION_DEVICE_PASSWORD=keep-me", content)
            self.assertIn("CUSTOM_LOCAL_KEY=keep-too", content)
            self.assertIn("NUVION_VIDEO_SOURCE=rpi", content)
            self.assertIn("NUVION_CAMERA_PROFILE=arducam_b0272", content)
            self.assertIn("NUVION_CAMERA_FOCUS_REQUIRED=true", content)
            self.assertIn("NUVION_CAMERA_I2C_BUS=", content)

    def test_ultra_preset_requires_explicit_i2c_bus_before_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "agent.env"
            config_path.write_text("NUVION_VIDEO_SOURCE=auto\n", encoding="utf-8")
            before = config_path.read_bytes()

            with self.assertRaisesRegex(ValueError, "I2C bus"):
                apply_camera_product_preset(config_path, product="ultra")

            self.assertEqual(config_path.read_bytes(), before)

    def test_ultra_preset_binds_b0273_and_selected_i2c_bus(self) -> None:
        updates = camera_product_updates(product="ultra", i2c_bus=10)

        self.assertEqual(updates["NUVION_VIDEO_SOURCE"], "jetson")
        self.assertEqual(updates["NUVION_CAMERA_PROFILE"], "arducam_b0273")
        self.assertEqual(updates["NUVION_CAMERA_FOCUS_MODE"], "startup-lock")
        self.assertEqual(updates["NUVION_CAMERA_FOCUS_REQUIRED"], "true")
        self.assertEqual(updates["NUVION_CAMERA_I2C_BUS"], "10")

    def test_qualification_report_contains_no_unrelated_config_or_secrets(self) -> None:
        report = build_camera_qualification_report(
            config_path=Path("/etc/nuv-agent/agent.env"),
            values={
                "NUVION_DEVICE_PASSWORD": "never-export",
                "NUVION_SERVER_BASE_URL": "https://api.example.com",
                "NUVION_VIDEO_SOURCE": "jetson",
                "NUVION_CAMERA_PROFILE": "arducam_b0273",
                "NUVION_CAMERA_FOCUS_MODE": "startup-lock",
                "NUVION_CAMERA_FOCUS_REQUIRED": "true",
                "NUVION_CAMERA_I2C_BUS": "9",
            },
            config_ok=True,
            hardware_checks=[
                {
                    "name": "Camera product profile",
                    "status": "pass",
                    "detail": "arducam_b0273 hardware identity verified",
                }
            ],
        )

        encoded = json.dumps(report)
        self.assertEqual(report["schemaVersion"], "nuvion.camera-qualification.v1")
        self.assertEqual(report["product"], "ultra")
        self.assertEqual(report["result"], "PASS")
        self.assertEqual(report["camera"]["i2cBus"], "9")
        self.assertNotIn("never-export", encoded)
        self.assertNotIn("NUVION_SERVER_BASE_URL", encoded)

    def test_qualification_report_does_not_pass_skipped_hardware_probe(self) -> None:
        report = build_camera_qualification_report(
            config_path=Path("/etc/nuv-agent/agent.env"),
            values={"NUVION_CAMERA_PROFILE": "arducam_b0272"},
            config_ok=True,
            hardware_checks=[
                {
                    "name": "Camera probe",
                    "status": "skip",
                    "detail": "gst-launch-1.0 not found",
                }
            ],
        )

        self.assertEqual(report["result"], "FAIL")


if __name__ == "__main__":
    unittest.main()
