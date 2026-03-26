from __future__ import annotations

import unittest
from unittest import mock

from nuvion_app.config import _parse_gst_device_monitor_output, discover_video_source_options


class ConfigVideoSourceTest(unittest.TestCase):
    def test_parse_gst_device_monitor_output_macos(self) -> None:
        sample = """
Device found:

\tname  : OBS Virtual Camera
\tgst-launch-1.0 avfvideosrc device-index=1 ! ...

Device found:

\tname  : MacBook Pro Camera
\tgst-launch-1.0 avfvideosrc device-index=2 ! ...
"""
        options = _parse_gst_device_monitor_output(sample)
        self.assertEqual(
            options,
            [
                {"value": "avf:1", "label": "OBS Virtual Camera", "detail": "macOS camera #1"},
                {"value": "avf:2", "label": "MacBook Pro Camera", "detail": "macOS camera #2"},
            ],
        )

    def test_discover_video_source_options_linux_lists_video_devices(self) -> None:
        fake_paths = [mock.Mock(name="video2"), mock.Mock(name="video0")]
        fake_paths[0].name = "video2"
        fake_paths[0].__str__ = lambda self=fake_paths[0]: "/dev/video2"
        fake_paths[1].name = "video0"
        fake_paths[1].__str__ = lambda self=fake_paths[1]: "/dev/video0"

        with mock.patch("nuvion_app.config.Path.glob", return_value=fake_paths):
            options = discover_video_source_options(platform_name="linux")

        self.assertEqual(options[0]["value"], "/dev/video0")
        self.assertEqual(options[1]["value"], "/dev/video2")
        self.assertEqual(options[-1]["value"], "rpi")


if __name__ == "__main__":
    unittest.main()
