from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


WORKER_ENV = "NUVION_NATIVE_GST_SOAK_WORKER"


class DepthAIGStreamerSoakTest(unittest.TestCase):
    def test_real_appsrc_soak_remains_within_buffer_and_byte_caps(self) -> None:
        repository = Path(__file__).resolve().parents[2]
        environment = os.environ.copy()
        python_path = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = str(repository) + (
            os.pathsep + python_path if python_path else ""
        )
        environment[WORKER_ENV] = "1"
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve())],
            cwd=repository,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            check=False,
        )
        if completed.returncode == 77:
            self.skipTest("native GStreamer bindings are unavailable")
        self.assertEqual(completed.returncode, 0, completed.stdout[-4000:])


def _run_native_worker() -> int:
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
    except (ImportError, ValueError):
        return 77

    import numpy as np

    from nuvion_app.inference.depthai_gst import DepthAIGStreamerBridge
    from nuvion_app.inference.video_source import DEPTHAI_APPSRC_NAME
    from nuvion_app.inference.video_source import build_video_source_pipeline

    class UnusedFrameSource:
        def start(self) -> None:
            return None

        def close(self) -> None:
            return None

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
    if appsrc is None:
        raise RuntimeError("native DepthAI appsrc is missing")
    bridge = DepthAIGStreamerBridge(
        frame_source=UnusedFrameSource(),
        appsrc=appsrc,
        gst=Gst,
        width=width,
        height=height,
        metrics_interval_sec=0,
    )
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    try:
        result = pipeline.set_state(Gst.State.PLAYING)
        if result == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("native GStreamer soak failed to enter PLAYING")
        for sequence in range(5_000):
            frame[0, 0, 0] = sequence & 0xFF
            bridge._push(frame)

        stats = bridge.stats_snapshot()
        if (stats.push_attempts, stats.push_succeeded, stats.push_failed) != (
            5_000,
            5_000,
            0,
        ):
            raise RuntimeError(f"unexpected native bridge counters: {stats}")
        if stats.appsrc_current_level_buffers is None:
            raise RuntimeError("native appsrc buffer level is unavailable")
        if stats.appsrc_current_level_bytes is None:
            raise RuntimeError("native appsrc byte level is unavailable")
        if stats.appsrc_current_level_buffers > 2:
            raise RuntimeError(f"native appsrc buffer cap exceeded: {stats}")
        if stats.appsrc_current_level_bytes > 2 * frame_bytes:
            raise RuntimeError(f"native appsrc byte cap exceeded: {stats}")
        if stats.appsrc_dropped is not None and stats.appsrc_dropped <= 0:
            raise RuntimeError(f"native appsrc did not report downstream drops: {stats}")
    finally:
        pipeline.set_state(Gst.State.NULL)
    return 0


if __name__ == "__main__":
    if os.environ.get(WORKER_ENV) == "1":
        raise SystemExit(_run_native_worker())
    unittest.main()
