from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nuvion_app.inference.video_source import build_video_source_pipeline
from nuvion_app.inference.video_source import resolve_demo_video_path


class VideoSourceTest(unittest.TestCase):
    def test_build_camera_source_linux(self) -> None:
        pipeline = build_video_source_pipeline(
            "/dev/video0",
            640,
            480,
            30,
            platform_name="linux",
        )
        self.assertIn("v4l2src device=/dev/video0", pipeline)
        self.assertIn("video/x-raw,format=RGB", pipeline)

    def test_build_camera_source_without_output_format_keeps_raw_caps(self) -> None:
        pipeline = build_video_source_pipeline(
            "/dev/video0",
            640,
            480,
            30,
            platform_name="darwin",
            output_format=None,
        )
        self.assertIn("avfvideosrc", pipeline)
        self.assertNotIn("video/x-raw,format=RGB", pipeline)

    def test_build_camera_source_macos_auto(self) -> None:
        pipeline = build_video_source_pipeline(
            "auto",
            640,
            480,
            30,
            platform_name="darwin",
        )
        self.assertIn("avfvideosrc", pipeline)

    def test_demo_mode_accepts_valid_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            demo_file = Path(tmp) / "demo.mp4"
            demo_file.write_bytes(b"fake")

            pipeline = build_video_source_pipeline(
                "/dev/video0",
                640,
                480,
                30,
                demo_mode=True,
                demo_video_path=str(demo_file),
                platform_name="linux",
            )

            self.assertIn("uridecodebin", pipeline)
            self.assertIn(demo_file.resolve().as_uri(), pipeline)

    def test_demo_mode_requires_path(self) -> None:
        with mock.patch("nuvion_app.inference.video_source.DEFAULT_DEMO_VIDEO_PATHS", tuple()):
            with mock.patch.dict("os.environ", {"NUVION_DEMO_VIDEO_FALLBACK_PATHS": ""}, clear=False):
                with self.assertRaises(ValueError):
                    build_video_source_pipeline(
                        "/dev/video0",
                        640,
                        480,
                        30,
                        demo_mode=True,
                        demo_video_path="",
                        platform_name="linux",
                    )

    def test_gst_override_takes_priority(self) -> None:
        pipeline = build_video_source_pipeline(
            "/dev/video0",
            640,
            480,
            30,
            gst_source_override="videotestsrc pattern=smpte",
            demo_mode=True,
            demo_video_path="/tmp/demo.mp4",
            platform_name="linux",
        )
        self.assertEqual(pipeline, "videotestsrc pattern=smpte")

    def test_demo_mode_uses_fallback_path_when_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            demo_file = Path(tmp) / "demo.webm"
            demo_file.write_bytes(b"fake")
            with mock.patch.dict(
                "os.environ",
                {"NUVION_DEMO_VIDEO_FALLBACK_PATHS": str(demo_file)},
                clear=False,
            ):
                resolved = resolve_demo_video_path("")
            self.assertEqual(resolved, demo_file.resolve())


if __name__ == "__main__":
    unittest.main()
