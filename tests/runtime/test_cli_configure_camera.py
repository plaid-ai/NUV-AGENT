from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

from nuvion_app import cli


class CliConfigureCameraTest(unittest.TestCase):
    def test_configure_ultra_passes_explicit_bus_to_product_preset(self) -> None:
        with (
            mock.patch.object(
                sys,
                "argv",
                [
                    "nuv-agent",
                    "configure-camera",
                    "--product",
                    "ultra",
                    "--i2c-bus",
                    "9",
                    "--config",
                    "/tmp/agent.env",
                ],
            ),
            mock.patch("nuvion_app.cli.load_env"),
            mock.patch(
                "nuvion_app.cli.resolve_config_path",
                return_value=Path("/tmp/agent.env"),
            ),
            mock.patch("nuvion_app.cli.apply_camera_product_preset") as apply_preset,
        ):
            cli.main()

        apply_preset.assert_called_once_with(
            Path("/tmp/agent.env"), product="ultra", i2c_bus=9
        )

    def test_configure_ultra_without_bus_exits_without_applying(self) -> None:
        with (
            mock.patch.object(
                sys,
                "argv",
                [
                    "nuv-agent",
                    "configure-camera",
                    "--product",
                    "ultra",
                    "--config",
                    "/tmp/agent.env",
                ],
            ),
            mock.patch("nuvion_app.cli.load_env"),
            mock.patch("nuvion_app.cli.apply_camera_product_preset") as apply_preset,
            self.assertRaisesRegex(SystemExit, "2"),
        ):
            cli.main()

        apply_preset.assert_not_called()


if __name__ == "__main__":
    unittest.main()
