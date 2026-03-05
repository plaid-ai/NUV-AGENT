from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from nuvion_app import cli


class CliDemoModeTest(unittest.TestCase):
    def setUp(self) -> None:
        for key in ("NUVION_DEMO_MODE", "NUVION_DEMO_VIDEO_PATH"):
            os.environ.pop(key, None)

    def _run_cli(self, argv: list[str]) -> None:
        fake_inference_main = types.ModuleType("nuvion_app.inference.main")
        fake_inference_main.main = mock.Mock()

        with mock.patch.dict(sys.modules, {"nuvion_app.inference.main": fake_inference_main}):
            with mock.patch.object(sys, "argv", argv):
                with mock.patch("nuvion_app.cli.resolve_config_path", return_value=Path("/tmp/agent.env")):
                    with mock.patch("nuvion_app.cli.load_env", return_value=Path("/tmp/agent.env")):
                        with mock.patch("nuvion_app.cli.ensure_runtime_config") as ensure_runtime_config:
                            ensure_runtime_config.return_value = mock.Mock(ok=True)
                            cli.main()

        fake_inference_main.main.assert_called_once()

    def test_run_demo_sets_demo_mode_env(self) -> None:
        self._run_cli(["nuv-agent", "run", "--demo"])
        self.assertEqual(os.getenv("NUVION_DEMO_MODE"), "true")

    def test_run_demo_video_overrides_env(self) -> None:
        with mock.patch.dict(os.environ, {"NUVION_DEMO_VIDEO_PATH": "/tmp/old.mp4"}, clear=False):
            self._run_cli(["nuv-agent", "run", "--demo", "--demo-video", "/tmp/new.mp4"])
            self.assertEqual(os.getenv("NUVION_DEMO_VIDEO_PATH"), "/tmp/new.mp4")


if __name__ == "__main__":
    unittest.main()
