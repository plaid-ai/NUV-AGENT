from __future__ import annotations

import unittest

import numpy as np

try:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    GST_AVAILABLE = True
except (ImportError, ValueError):
    Gst = None  # type: ignore[assignment]
    GST_AVAILABLE = False

from nuvion_app.inference.depthai_gst import DepthAIGStreamerBridge
from nuvion_app.inference.video_source import DEPTHAI_APPSRC_NAME
from nuvion_app.inference.video_source import build_video_source_pipeline


class _UnusedFrameSource:
    def start(self) -> None:
        return None

    def close(self) -> None:
        return None


@unittest.skipUnless(GST_AVAILABLE, "native GStreamer bindings are unavailable")
class DepthAIGStreamerSoakTest(unittest.TestCase):
    def test_real_appsrc_soak_remains_within_buffer_and_byte_caps(self) -> None:
        assert Gst is not None
        Gst.init(None)
        width = 320
        height = 240
        frame_bytes = width * height * 3
        source_pipeline = build_video_source_pipeline(
            "oak",
            width,
            height,
            30,
            platform_name="linux",
        )
        pipeline = Gst.parse_launch(
            f"{source_pipeline} ! identity sleep-time=100000 ! fakesink sync=false"
        )
        appsrc = pipeline.get_by_name(DEPTHAI_APPSRC_NAME)
        self.assertIsNotNone(appsrc)
        assert appsrc is not None
        bridge = DepthAIGStreamerBridge(
            frame_source=_UnusedFrameSource(),
            appsrc=appsrc,
            gst=Gst,
            width=width,
            height=height,
            metrics_interval_sec=0,
        )
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        try:
            result = pipeline.set_state(Gst.State.PLAYING)
            self.assertNotEqual(result, Gst.StateChangeReturn.FAILURE)
            for sequence in range(5_000):
                frame[0, 0, 0] = sequence & 0xFF
                bridge._push(frame)

            stats = bridge.stats_snapshot()
            self.assertEqual(stats.push_attempts, 5_000)
            self.assertEqual(stats.push_succeeded, 5_000)
            self.assertEqual(stats.push_failed, 0)
            self.assertIsNotNone(stats.appsrc_current_level_buffers)
            self.assertIsNotNone(stats.appsrc_current_level_bytes)
            assert stats.appsrc_current_level_buffers is not None
            assert stats.appsrc_current_level_bytes is not None
            self.assertLessEqual(stats.appsrc_current_level_buffers, 2)
            self.assertLessEqual(stats.appsrc_current_level_bytes, 2 * frame_bytes)
            if stats.appsrc_dropped is not None:
                self.assertGreater(stats.appsrc_dropped, 0)
        finally:
            pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    unittest.main()
