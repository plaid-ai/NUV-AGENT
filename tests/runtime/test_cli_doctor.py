from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

from nuvion_app import cli


class CliDoctorTest(unittest.TestCase):
    def test_hardware_doctor_fails_when_camera_probe_fails(self) -> None:
        report = mock.Mock(ok=True, values={"NUVION_VIDEO_SOURCE": "oak"})
        checks = [
            {"name": "Camera source", "status": "pass", "detail": "oak"},
            {"name": "Camera probe", "status": "fail", "detail": "runtime missing"},
        ]
        with (
            mock.patch.object(
                sys,
                "argv",
                ["nuv-agent", "doctor", "--hardware", "--config", "/tmp/agent.env"],
            ),
            mock.patch("nuvion_app.cli.load_env"),
            mock.patch(
                "nuvion_app.cli.resolve_config_path",
                return_value=Path("/tmp/agent.env"),
            ),
            mock.patch("nuvion_app.cli.guard_config", return_value=report),
            mock.patch("nuvion_app.cli.print_report"),
            mock.patch(
                "nuvion_app.cli.run_camera_health_checks",
                return_value=checks,
            ),
        ):
            with self.assertRaisesRegex(SystemExit, "2"):
                cli.main()

    def test_config_only_doctor_does_not_touch_hardware(self) -> None:
        report = mock.Mock(ok=True, values={"NUVION_VIDEO_SOURCE": "oak"})
        with (
            mock.patch.object(
                sys,
                "argv",
                ["nuv-agent", "doctor", "--config", "/tmp/agent.env"],
            ),
            mock.patch("nuvion_app.cli.load_env"),
            mock.patch(
                "nuvion_app.cli.resolve_config_path",
                return_value=Path("/tmp/agent.env"),
            ),
            mock.patch("nuvion_app.cli.guard_config", return_value=report),
            mock.patch("nuvion_app.cli.print_report"),
            mock.patch("nuvion_app.cli.run_camera_health_checks") as hardware_mock,
        ):
            cli.main()

        hardware_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
