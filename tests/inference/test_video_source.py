from __future__ import annotations

import tempfile
import unittest
from unittest import mock

from nuvion_app.inference.video_source import build_video_source_pipeline


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

    def test_build_camera_source_macos_auto(self) -> None:
        pipeline = build_video_source_pipeline(
            "auto",
            640,
            480,
            30,
            platform_name="darwin",
        )
        self.assertIn("avfvideosrc", pipeline)

    def test_demo_mode_without_video_uses_mvtec_slideshow(self) -> None:
        fake_source = mock.Mock(
            stage_pattern="/tmp/mvtec/slides/screw/%05d.png",
            slideshow_caps="image/png,framerate=1/1",
            decoder="pngdec",
        )
        with mock.patch("nuvion_app.inference.video_source.prepare_mvtec_demo_source", return_value=fake_source):
            pipeline = build_video_source_pipeline(
                "/dev/video0",
                640,
                480,
                30,
                demo_mode=True,
                platform_name="linux",
            )
        self.assertIn("multifilesrc", pipeline)
        self.assertIn(fake_source.stage_pattern, pipeline)
        self.assertIn(fake_source.decoder, pipeline)

    def test_gst_override_takes_priority(self) -> None:
        pipeline = build_video_source_pipeline(
            "/dev/video0",
            640,
            480,
            30,
            gst_source_override="videotestsrc pattern=smpte",
            demo_mode=True,
            platform_name="linux",
        )
        self.assertEqual(pipeline, "videotestsrc pattern=smpte")


if __name__ == "__main__":
    unittest.main()
