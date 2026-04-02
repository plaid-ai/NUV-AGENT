from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nuvion_app import config as config_module


class QrSetupTest(unittest.TestCase):
    def test_run_qr_setup_prompts_camera_settings_and_saves(self) -> None:
        fields = [
            {"key": "NUVION_SERVER_BASE_URL", "default": "https://api.example.com", "comment": "Server URL"},
            {"key": "NUVION_DEVICE_USERNAME", "default": "", "comment": "Device username"},
            {"key": "NUVION_DEVICE_PASSWORD", "default": "", "comment": "Device password"},
            {"key": "NUVION_VIDEO_SOURCE", "default": "auto", "comment": "Video source"},
            {"key": "NUVION_VIDEO_ROTATION", "default": "0", "comment": "Video rotation"},
            {"key": "NUVION_VIDEO_FLIP_HORIZONTAL", "default": "false", "comment": "Flip horizontal"},
            {"key": "NUVION_VIDEO_FLIP_VERTICAL", "default": "false", "comment": "Flip vertical"},
        ]
        lines = [f"{field['key']}={field['default']}" for field in fields]
        existing = {"NUVION_SERVER_BASE_URL": "https://api.example.com"}
        prompted_values = {
            "NUVION_SERVER_BASE_URL": "https://api.example.com",
            "NUVION_DEVICE_USERNAME": "device-1",
            "NUVION_DEVICE_PASSWORD": "secret-1",
            "NUVION_VIDEO_SOURCE": "jetson",
            "NUVION_VIDEO_ROTATION": "180",
            "NUVION_VIDEO_FLIP_HORIZONTAL": "true",
            "NUVION_VIDEO_FLIP_VERTICAL": "false",
        }

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "agent.env"
            with mock.patch.object(config_module, "load_template", return_value=(lines, fields)):
                with mock.patch.object(config_module, "read_env", return_value=existing):
                    with mock.patch.object(
                        config_module,
                        "_init_pairing",
                        return_value={
                            "pairingId": "pairing-1",
                            "pairingUrl": "https://example.com/pair",
                            "pairingCode": "123456",
                        },
                    ):
                        with mock.patch.object(
                            config_module,
                            "_wait_for_pairing",
                            return_value={"deviceUsername": "device-1", "devicePassword": "secret-1"},
                        ):
                            with mock.patch.object(config_module, "_print_qr"):
                                with mock.patch.object(
                                    config_module,
                                    "_prompt_camera_setup",
                                    return_value=prompted_values,
                                ) as prompt_camera_setup:
                                    with mock.patch.object(config_module, "write_env") as write_env:
                                        config_module.run_qr_setup(config_path, advanced=False)

        prompt_camera_setup.assert_called_once()
        write_env.assert_called_once()
        saved_values = write_env.call_args.args[2]
        self.assertEqual(saved_values["NUVION_VIDEO_SOURCE"], "jetson")
        self.assertEqual(saved_values["NUVION_VIDEO_ROTATION"], "180")
        self.assertEqual(saved_values["NUVION_VIDEO_FLIP_HORIZONTAL"], "true")


if __name__ == "__main__":
    unittest.main()
