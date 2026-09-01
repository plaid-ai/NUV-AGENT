from __future__ import annotations

import threading
import time
import unittest
from unittest import mock

import numpy as np

from nuvion_app.inference.depthai_gst import DepthAIGStreamerBridge
from nuvion_app.inference.depthai_gst import DepthAIGStreamerBridgeError
from nuvion_app.inference.depthai_source import DepthAITimeoutError


class _Buffer:
    def __init__(self, size: int) -> None:
        self.data = bytearray(size)

    def fill(self, offset: int, data: bytes) -> None:
        self.data[offset : offset + len(data)] = data


class _Gst:
    class FlowReturn:
        OK = "OK"

    class Buffer:
        @staticmethod
        def new_allocate(_allocator, size: int, _params) -> _Buffer:
            return _Buffer(size)


class _AppSrc:
    def __init__(self, flow: str = "OK") -> None:
        self.flow = flow
        self.buffers: list[_Buffer] = []
        self.pushed = threading.Event()

    def emit(self, signal: str, buffer: _Buffer) -> str:
        if signal != "push-buffer":
            raise AssertionError(signal)
        self.buffers.append(buffer)
        self.pushed.set()
        return self.flow


class _Source:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = list(frames)
        self.started = False
        self.closed = threading.Event()

    def start(self) -> None:
        self.started = True

    def read(self, timeout: float) -> np.ndarray:
        if self.frames:
            return self.frames.pop(0)
        self.closed.wait(timeout)
        raise DepthAITimeoutError("no frame")

    def close(self) -> None:
        self.closed.set()


class _StuckSource:
    def __init__(self) -> None:
        self.entered_read = threading.Event()
        self.release_read = threading.Event()

    def start(self) -> None:
        return None

    def read(self, timeout: float) -> np.ndarray:
        self.entered_read.set()
        self.release_read.wait()
        raise DepthAITimeoutError("released")

    def close(self) -> None:
        return None


class DepthAIGStreamerBridgeTest(unittest.TestCase):
    def test_thread_start_failure_closes_started_source(self) -> None:
        source = _Source([])
        bridge = DepthAIGStreamerBridge(
            frame_source=source,
            appsrc=_AppSrc(),
            gst=_Gst,
            width=4,
            height=3,
        )
        fake_thread = mock.Mock()
        fake_thread.start.side_effect = RuntimeError("thread unavailable")

        with mock.patch(
            "nuvion_app.inference.depthai_gst.threading.Thread",
            return_value=fake_thread,
        ):
            with self.assertRaisesRegex(RuntimeError, "thread unavailable"):
                bridge.start()

        self.assertTrue(source.started)
        self.assertTrue(source.closed.is_set())
        self.assertFalse(bridge.running)

    def test_pushes_contiguous_rgb_frame_and_closes_source(self) -> None:
        frame = np.arange(4 * 3 * 3, dtype=np.uint8).reshape(3, 4, 3)
        source = _Source([frame])
        appsrc = _AppSrc()
        bridge = DepthAIGStreamerBridge(
            frame_source=source,
            appsrc=appsrc,
            gst=_Gst,
            width=4,
            height=3,
            read_timeout=0.1,
            max_consecutive_timeouts=10,
        )

        bridge.start()
        self.assertTrue(appsrc.pushed.wait(1.0))
        bridge.close()

        self.assertTrue(source.started)
        self.assertTrue(source.closed.is_set())
        self.assertEqual(bytes(appsrc.buffers[0].data), frame.tobytes())
        self.assertFalse(bridge.running)
        self.assertIsNone(bridge.failure)

    def test_invalid_frame_surfaces_failure_callback(self) -> None:
        source = _Source([np.zeros((2, 2, 3), dtype=np.uint8)])
        failure = threading.Event()
        seen: list[BaseException] = []
        bridge = DepthAIGStreamerBridge(
            frame_source=source,
            appsrc=_AppSrc(),
            gst=_Gst,
            width=4,
            height=3,
            read_timeout=0.1,
            on_failure=lambda exc: (seen.append(exc), failure.set()),
        )

        bridge.start()
        self.assertTrue(failure.wait(1.0))
        bridge.close()

        self.assertIsInstance(seen[0], DepthAIGStreamerBridgeError)
        self.assertIs(bridge.failure, seen[0])

    def test_repeated_timeouts_fail_instead_of_hanging_forever(self) -> None:
        source = _Source([])
        failure = threading.Event()
        bridge = DepthAIGStreamerBridge(
            frame_source=source,
            appsrc=_AppSrc(),
            gst=_Gst,
            width=4,
            height=3,
            read_timeout=0.01,
            max_consecutive_timeouts=2,
            on_failure=lambda _exc: failure.set(),
        )

        bridge.start()
        self.assertTrue(failure.wait(1.0))
        bridge.close()

        self.assertIsInstance(bridge.failure, DepthAIGStreamerBridgeError)

    def test_stuck_reader_is_not_reported_stopped(self) -> None:
        source = _StuckSource()
        bridge = DepthAIGStreamerBridge(
            frame_source=source,
            appsrc=_AppSrc(),
            gst=_Gst,
            width=4,
            height=3,
            read_timeout=0.01,
        )

        bridge.start()
        self.assertTrue(source.entered_read.wait(1.0))
        bridge.close(join_timeout=0.01)
        self.assertTrue(bridge.running)

        source.release_read.set()
        deadline = time.monotonic() + 1.0
        while bridge.running and time.monotonic() < deadline:
            time.sleep(0.01)
        bridge.close(join_timeout=0.1)
        self.assertFalse(bridge.running)


if __name__ == "__main__":
    unittest.main()
