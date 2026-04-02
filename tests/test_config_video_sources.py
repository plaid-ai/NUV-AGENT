from __future__ import annotations

import unittest
from unittest import mock

from nuvion_app.config import _check_camera_source, _parse_gst_device_monitor_output, _render_form, discover_video_source_options


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
            with mock.patch("nuvion_app.config._is_jetson_platform", return_value=False):
                options = discover_video_source_options(platform_name="linux")

        self.assertEqual(options[0]["value"], "/dev/video0")
        self.assertEqual(options[1]["value"], "/dev/video2")
        self.assertEqual(options[-1]["value"], "rpi")

    def test_discover_video_source_options_linux_includes_jetson_option(self) -> None:
        with mock.patch("nuvion_app.config.Path.glob", return_value=[]):
            with mock.patch("nuvion_app.config._is_jetson_platform", return_value=True):
                with mock.patch("nuvion_app.config._gst_element_available", return_value=True):
                    options = discover_video_source_options(platform_name="linux")

        self.assertEqual(options[0]["value"], "jetson")
        self.assertEqual(options[0]["detail"], "Uses nvarguscamerasrc.")

    def test_check_camera_source_accepts_auto(self) -> None:
        check = _check_camera_source({"NUVION_VIDEO_SOURCE": "auto"})
        self.assertEqual(check["status"], "pass")

    def test_render_form_groups_advanced_fields_and_renders_boolean_selects(self) -> None:
        fields = [
            {"key": "NUVION_SERVER_BASE_URL", "default": "https://api.example.com", "comment": "Server URL"},
            {"key": "NUVION_DEVICE_USERNAME", "default": "device-1", "comment": "Device username"},
            {"key": "NUVION_DEVICE_PASSWORD", "default": "***", "comment": "Device password"},
            {"key": "NUVION_VIDEO_SOURCE", "default": "/dev/video0", "comment": "Video source"},
            {"key": "NUVION_VIDEO_ROTATION", "default": "0", "comment": "Video rotation"},
            {"key": "NUVION_VIDEO_FLIP_HORIZONTAL", "default": "false", "comment": "Flip horizontal"},
            {"key": "NUVION_VIDEO_FLIP_VERTICAL", "default": "false", "comment": "Flip vertical"},
            {"key": "NUVION_DEMO_MODE", "default": "false", "comment": "Demo mode"},
            {"key": "NUVION_WEBRTC_FORCE_RELAY", "default": "true", "comment": "Force relay"},
            {"key": "NUVION_CLIP_ENABLED", "default": "true", "comment": "Clip enabled"},
            {"key": "NUVION_ZSAD_BACKEND", "default": "triton", "comment": "Backend"},
            {"key": "NUVION_ZERO_SHOT_DEVICE", "default": "auto", "comment": "SigLIP device"},
        ]
        values = {
            "NUVION_SERVER_BASE_URL": "https://api.example.com",
            "NUVION_DEVICE_USERNAME": "device-1",
            "NUVION_DEVICE_PASSWORD": "secret",
            "NUVION_VIDEO_SOURCE": "avf:2",
            "NUVION_VIDEO_ROTATION": "90",
            "NUVION_VIDEO_FLIP_HORIZONTAL": "true",
            "NUVION_VIDEO_FLIP_VERTICAL": "false",
            "NUVION_DEMO_MODE": "false",
            "NUVION_WEBRTC_FORCE_RELAY": "true",
            "NUVION_CLIP_ENABLED": "true",
            "NUVION_ZSAD_BACKEND": "triton",
            "NUVION_ZERO_SHOT_DEVICE": "auto",
        }

        html = _render_form(
            fields,
            values,
            missing=[],
            device_name="nuvion-pro-1",
            video_source_options=[
                {"value": "avf:2", "label": "MacBook Pro Camera", "detail": "macOS camera #2"},
            ],
        )

        self.assertIn("Quick Start", html)
        self.assertIn("Advanced Options", html)
        self.assertIn("Most devices only need the server address", html)
        self.assertIn("Use detected camera names instead of typing /dev/video0 or avf:2 manually.", html)
        self.assertIn('name="NUVION_VIDEO_ROTATION"', html)
        self.assertIn('value="90" selected', html)
        self.assertIn("Live sessions can be overridden by the backend", html)
        self.assertIn('<option value="true" selected>On</option>', html)
        self.assertIn('<option value="false" selected>Off</option>', html)


if __name__ == "__main__":
    unittest.main()
