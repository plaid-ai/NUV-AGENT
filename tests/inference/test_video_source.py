from __future__ import annotations

import tempfile
import unittest
from unittest import mock

from nuvion_app.inference.video_source import LinuxVideoDeviceInfo
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

    def test_build_camera_source_linux_auto_uses_first_v4l2_device(self) -> None:
        fake_devices = [
            LinuxVideoDeviceInfo(path="/dev/video2", name="USB Camera"),
            LinuxVideoDeviceInfo(path="/dev/video4", name="Other Camera"),
        ]
        with mock.patch("nuvion_app.inference.video_source._linux_video_devices", return_value=fake_devices):
            with mock.patch("nuvion_app.inference.video_source._is_jetson_platform", return_value=False):
                pipeline = build_video_source_pipeline(
                    "auto",
                    640,
                    480,
                    30,
                    platform_name="linux",
                )

        self.assertIn("v4l2src device=/dev/video2", pipeline)

    def test_build_camera_source_linux_auto_prefers_jetson_argus_for_csi(self) -> None:
        fake_devices = [LinuxVideoDeviceInfo(path="/dev/video0", name="vi-output, imx477 9-001a")]
        with mock.patch("nuvion_app.inference.video_source._linux_video_devices", return_value=fake_devices):
            with mock.patch("nuvion_app.inference.video_source._is_jetson_platform", return_value=True):
                with mock.patch("nuvion_app.inference.video_source._gst_element_available", return_value=True):
                    pipeline = build_video_source_pipeline(
                        "auto",
                        640,
                        480,
                        30,
                        platform_name="linux",
                    )

        self.assertIn("nvarguscamerasrc sensor-id=0", pipeline)
        self.assertIn("nvvidconv", pipeline)

    def test_build_camera_source_linux_auto_prefers_usb_when_requested(self) -> None:
        fake_devices = [
            LinuxVideoDeviceInfo(path="/dev/video0", name="vi-output, imx477 9-001a"),
            LinuxVideoDeviceInfo(path="/dev/video2", name="USB Camera"),
        ]
        with mock.patch.dict("os.environ", {"NUVION_CAMERA_PREFERENCE": "usb"}, clear=False):
            with mock.patch("nuvion_app.inference.video_source._linux_video_devices", return_value=fake_devices):
                with mock.patch("nuvion_app.inference.video_source._is_jetson_platform", return_value=True):
                    with mock.patch("nuvion_app.inference.video_source._gst_element_available", return_value=True):
                        pipeline = build_video_source_pipeline(
                            "auto",
                            640,
                            480,
                            30,
                            platform_name="linux",
                        )

        self.assertIn("v4l2src device=/dev/video2", pipeline)
        self.assertNotIn("nvarguscamerasrc", pipeline)

    def test_build_camera_source_linux_explicit_jetson_csi_device_uses_argus(self) -> None:
        fake_device = LinuxVideoDeviceInfo(path="/dev/video0", name="vi-output, imx477 9-001a")
        with mock.patch("nuvion_app.inference.video_source._is_jetson_platform", return_value=True):
            with mock.patch("nuvion_app.inference.video_source._gst_element_available", return_value=True):
                with mock.patch("nuvion_app.inference.video_source._find_linux_video_device", return_value=fake_device):
                    pipeline = build_video_source_pipeline(
                        "/dev/video0",
                        640,
                        480,
                        30,
                        platform_name="linux",
                    )

        self.assertIn("nvarguscamerasrc sensor-id=0", pipeline)

    def test_build_camera_source_linux_explicit_usb_device_stays_v4l2_on_jetson(self) -> None:
        fake_device = LinuxVideoDeviceInfo(path="/dev/video0", name="USB Camera")
        with mock.patch("nuvion_app.inference.video_source._is_jetson_platform", return_value=True):
            with mock.patch("nuvion_app.inference.video_source._gst_element_available", return_value=True):
                with mock.patch("nuvion_app.inference.video_source._find_linux_video_device", return_value=fake_device):
                    pipeline = build_video_source_pipeline(
                        "/dev/video0",
                        640,
                        480,
                        30,
                        platform_name="linux",
                    )

        self.assertIn("v4l2src device=/dev/video0", pipeline)

    def test_build_camera_source_linux_supports_v4l_alias_path(self) -> None:
        pipeline = build_video_source_pipeline(
            "/dev/v4l/by-id/usb-test-camera",
            640,
            480,
            30,
            platform_name="linux",
        )

        self.assertIn("v4l2src device=/dev/v4l/by-id/usb-test-camera", pipeline)

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

    def test_demo_mode_uses_provided_demo_source(self) -> None:
        fake_source = mock.Mock(
            stage_pattern="/tmp/mvtec/slides/cable/%05d.png",
            slideshow_caps="image/png,framerate=1/2",
            decoder="pngdec",
        )
        with mock.patch("nuvion_app.inference.video_source.prepare_mvtec_demo_source") as prepare_mock:
            pipeline = build_video_source_pipeline(
                "/dev/video0",
                640,
                480,
                30,
                demo_mode=True,
                platform_name="linux",
                demo_source=fake_source,
            )
        prepare_mock.assert_not_called()
        self.assertIn(fake_source.stage_pattern, pipeline)

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

    def test_video_transforms_are_applied_to_standard_pipeline(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                "NUVION_VIDEO_ROTATION": "90",
                "NUVION_VIDEO_FLIP_HORIZONTAL": "true",
                "NUVION_VIDEO_FLIP_VERTICAL": "false",
            },
            clear=False,
        ):
            pipeline = build_video_source_pipeline(
                "/dev/video0",
                640,
                480,
                30,
                platform_name="linux",
            )

        self.assertIn("videoflip method=horizontal-flip", pipeline)
        self.assertIn("videoflip method=clockwise", pipeline)

    def test_video_transforms_are_applied_to_gst_override(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                "NUVION_VIDEO_ROTATION": "180",
                "NUVION_VIDEO_FLIP_HORIZONTAL": "false",
                "NUVION_VIDEO_FLIP_VERTICAL": "true",
            },
            clear=False,
        ):
            pipeline = build_video_source_pipeline(
                "/dev/video0",
                640,
                480,
                30,
                gst_source_override="videotestsrc pattern=smpte",
                platform_name="linux",
            )

        self.assertEqual(
            pipeline,
            "videotestsrc pattern=smpte ! videoconvert ! videoflip method=vertical-flip ! videoflip method=rotate-180 ! video/x-raw,format=RGB",
        )

    def test_jetson_pipeline_applies_camera_controls_and_balance(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                "NUVION_CAMERA_AUTO_EXPOSURE": "false",
                "NUVION_CAMERA_EXPOSURE_COMPENSATION": "0.5",
                "NUVION_CAMERA_AUTO_WHITE_BALANCE": "false",
                "NUVION_CAMERA_WB_MODE": "daylight",
                "NUVION_CAMERA_BRIGHTNESS": "0.1",
                "NUVION_CAMERA_CONTRAST": "1.2",
                "NUVION_CAMERA_SATURATION": "1.3",
            },
            clear=False,
        ):
            with mock.patch("nuvion_app.inference.video_source._gst_element_available", return_value=True):
                pipeline = build_video_source_pipeline(
                    "jetson",
                    640,
                    480,
                    30,
                    platform_name="linux",
                )

        self.assertIn("aelock=true", pipeline)
        self.assertIn("awblock=true", pipeline)
        self.assertIn("exposurecompensation=0.500", pipeline)
        self.assertIn("wbmode=daylight", pipeline)
        self.assertIn("videobalance brightness=0.100 contrast=1.200 saturation=1.300", pipeline)


if __name__ == "__main__":
    unittest.main()
