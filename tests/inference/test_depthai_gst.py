from __future__ import annotations

import threading
import time
import unittest
import weakref
from unittest import mock

import numpy as np

from nuvion_app.inference.depthai_gst import DepthAIGStreamerBridge
from nuvion_app.inference.depthai_gst import DepthAIGStreamerBridgeError
from nuvion_app.inference.depthai_source import DepthAITimeoutError


class _Buffer:
    def __init__(self, size: int) -> None:
        self.data = bytearray(size)

    def fill(self, offset: int, data: bytes | memoryview) -> int:
        self.data[offset : offset + len(data)] = data
        return len(data)


class _Gst:
    class FlowReturn:
        OK = "OK"

    class Buffer:
        @staticmethod
        def new_allocate(_allocator, size: int, _params) -> _Buffer:
            return _Buffer(size)

        @staticmethod
        def new_wrapped(data: bytes) -> _Buffer:
            buffer = _Buffer(len(data))
            buffer.fill(0, data)
            return buffer


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


class _NonRetainingAppSrc:
    def __init__(self) -> None:
        self.buffer_refs: list[weakref.ReferenceType[_Buffer]] = []
        self.properties = {
            "current-level-buffers": 2,
            "current-level-bytes": 72,
            "current-level-time": 0,
            "in": 5,
            "out": 3,
            "dropped": 2,
        }

    def emit(self, signal: str, buffer: _Buffer) -> str:
        if signal != "push-buffer":
            raise AssertionError(signal)
        self.buffer_refs.append(weakref.ref(buffer))
        return "OK"

    def find_property(self, name: str) -> object | None:
        return object() if name in self.properties else None

    def get_property(self, name: str) -> int:
        return self.properties[name]


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
    def test_metrics_interval_rejects_invalid_values(self) -> None:
        for value in (-1, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    DepthAIGStreamerBridge(
                        frame_source=_Source([]),
                        appsrc=_AppSrc(),
                        gst=_Gst,
                        width=4,
                        height=3,
                        metrics_interval_sec=value,
                    )

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

    def test_push_wraps_one_owned_frame_and_releases_wrapper(self) -> None:
        class _WrappedOnlyGst:
            FlowReturn = _Gst.FlowReturn

            class Buffer:
                @staticmethod
                def new_allocate(*_args: object) -> _Buffer:
                    raise AssertionError("separate native allocation is forbidden")

                @staticmethod
                def new_wrapped(data: bytes) -> _Buffer:
                    buffer = _Buffer(len(data))
                    buffer.fill(0, data)
                    return buffer

        frame = np.arange(4 * 3 * 3, dtype=np.uint8).reshape(3, 4, 3)
        appsrc = _NonRetainingAppSrc()
        bridge = DepthAIGStreamerBridge(
            frame_source=_Source([]),
            appsrc=appsrc,
            gst=_WrappedOnlyGst,
            width=4,
            height=3,
            metrics_interval_sec=0,
        )

        bridge._push(frame)
        stats = bridge.stats_snapshot()

        self.assertEqual(stats.push_attempts, 1)
        self.assertEqual(stats.push_succeeded, 1)
        self.assertEqual(stats.push_failed, 0)
        self.assertEqual(stats.appsrc_current_level_buffers, 2)
        self.assertEqual(stats.appsrc_current_level_bytes, 72)
        self.assertEqual(stats.appsrc_dropped, 2)
        self.assertIsNone(appsrc.buffer_refs[0]())

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
