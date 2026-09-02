from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from nuvion_app.inference.depthai_source import DepthAIConfig
from nuvion_app.inference.depthai_source import DepthAIFrameSource
from nuvion_app.inference.depthai_source import DepthAIPermissionError
from nuvion_app.inference.depthai_source import DepthAIReadError
from nuvion_app.inference.depthai_source import DepthAIStartupError
from nuvion_app.inference.depthai_source import DepthAIStateError
from nuvion_app.inference.depthai_source import DepthAITimeoutError
from nuvion_app.inference.depthai_source import DepthAIUnavailableError


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class _FakePacket:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame

    def getCvFrame(self) -> np.ndarray:
        return self.frame


class _FakeNativePacket:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame

    def getFrame(self) -> np.ndarray:
        return self.frame


class _FakeOutputQueue:
    def __init__(self, packets: list[_FakePacket] | None = None) -> None:
        self.packets = list(packets or [])
        self.close_calls = 0

    def tryGetAll(self) -> list[_FakePacket]:
        packets = list(self.packets)
        self.packets.clear()
        return packets

    def close(self) -> None:
        self.close_calls += 1


class _FakeCamera:
    def __init__(self) -> None:
        self.preview = SimpleNamespace(link=mock.Mock())
        self.preview_size: tuple[int, int] | None = None
        self.interleaved: bool | None = None
        self.fps: float | None = None
        self.color_order: object | None = None
        self.board_socket: object | None = None

    def setPreviewSize(self, width: int, height: int) -> None:
        self.preview_size = (width, height)

    def setInterleaved(self, interleaved: bool) -> None:
        self.interleaved = interleaved

    def setFps(self, fps: float) -> None:
        self.fps = fps

    def setColorOrder(self, color_order: object) -> None:
        self.color_order = color_order

    def setBoardSocket(self, board_socket: object) -> None:
        self.board_socket = board_socket


class _FakeXLinkOut:
    def __init__(self) -> None:
        self.input = object()
        self.stream_name: str | None = None

    def setStreamName(self, stream_name: str) -> None:
        self.stream_name = stream_name


class _FakePipeline:
    def __init__(self, owner: _FakeDepthAI) -> None:
        self.owner = owner
        self.camera = _FakeCamera()
        self.output = _FakeXLinkOut()

    def create(self, node_type: object) -> object:
        if node_type is self.owner.node.ColorCamera:
            return self.camera
        if node_type is self.owner.node.XLinkOut:
            return self.output
        raise AssertionError(f"unexpected fake node type: {node_type!r}")


class _FakeDevice:
    def __init__(self, owner: _FakeDepthAI, args: tuple[object, ...]) -> None:
        self.owner = owner
        self.args = args
        self.close_calls = 0
        self.queue_request: dict[str, object] | None = None

    def getOutputQueue(self, **kwargs: object) -> _FakeOutputQueue:
        self.queue_request = kwargs
        return self.owner.output_queue

    def close(self) -> None:
        self.close_calls += 1


class _FakeDepthAI:
    __version__ = "2.32.0.0"

    def __init__(
        self,
        packets: list[_FakePacket] | None = None,
        *,
        device_error: Exception | None = None,
    ) -> None:
        self.node = SimpleNamespace(ColorCamera=object(), XLinkOut=object())
        self.ColorCameraProperties = SimpleNamespace(ColorOrder=SimpleNamespace(RGB="rgb"))
        self.CameraBoardSocket = SimpleNamespace(CAM_A="cam-a", RGB="rgb-socket")
        self.output_queue = _FakeOutputQueue(packets)
        self.device_error = device_error
        self.pipeline: _FakePipeline | None = None
        self.devices: list[_FakeDevice] = []
        self.device_info_calls: list[str] = []
        self.device_created = threading.Event()

    def Pipeline(self) -> _FakePipeline:
        self.pipeline = _FakePipeline(self)
        return self.pipeline

    def DeviceInfo(self, device_id: str) -> object:
        self.device_info_calls.append(device_id)
        return SimpleNamespace(device_id=device_id)

    def Device(self, *args: object) -> _FakeDevice:
        if self.device_error is not None:
            raise self.device_error
        device = _FakeDevice(self, args)
        self.devices.append(device)
        self.device_created.set()
        return device


def _packet(value: int, *, width: int = 4, height: int = 3) -> _FakePacket:
    return _FakePacket(np.full((height, width, 3), value, dtype=np.uint8))


def _config(**overrides: object) -> DepthAIConfig:
    values: dict[str, object] = {
        "width": 4,
        "height": 3,
        "fps": 25,
        "startup_timeout": 0.1,
        "read_timeout": 0.1,
        "poll_interval": 0.01,
    }
    values.update(overrides)
    return DepthAIConfig(**values)  # type: ignore[arg-type]


class DepthAIFrameSourceTest(unittest.TestCase):
    def test_depthai_import_is_lazy_and_missing_runtime_is_typed(self) -> None:
        with mock.patch(
            "nuvion_app.inference.depthai_source.importlib.import_module",
            side_effect=ModuleNotFoundError("private installation detail"),
        ) as import_module:
            source = DepthAIFrameSource(_config())
            import_module.assert_not_called()

            with self.assertRaises(DepthAIUnavailableError) as raised:
                source.start()

        import_module.assert_called_once_with("depthai")
        self.assertNotIn("private installation detail", str(raised.exception))
        self.assertTrue(source.is_closed)

    def test_start_configures_v2_rgb_pipeline_and_specific_device(self) -> None:
        fake = _FakeDepthAI([_packet(7)])
        config = _config(device_id="  oak-mxid-1  ", queue_size=2)
        source = DepthAIFrameSource(config, depthai_module=fake)

        returned = source.start()

        self.assertIs(returned, source)
        self.assertTrue(source.is_running)
        self.assertEqual(config.device_id, "oak-mxid-1")
        self.assertEqual(fake.device_info_calls, ["oak-mxid-1"])
        self.assertIsNotNone(fake.pipeline)
        assert fake.pipeline is not None
        self.assertEqual(fake.pipeline.camera.preview_size, (4, 3))
        self.assertFalse(fake.pipeline.camera.interleaved)
        self.assertEqual(fake.pipeline.camera.fps, 25)
        self.assertEqual(fake.pipeline.camera.color_order, "rgb")
        self.assertEqual(fake.pipeline.camera.board_socket, "cam-a")
        self.assertEqual(fake.pipeline.output.stream_name, DepthAIFrameSource.STREAM_NAME)
        fake.pipeline.camera.preview.link.assert_called_once_with(fake.pipeline.output.input)
        self.assertEqual(len(fake.devices[0].args), 2)
        self.assertEqual(
            fake.devices[0].queue_request,
            {"name": "nuvion_rgb", "maxSize": 2, "blocking": False},
        )

    def test_default_device_does_not_construct_device_info(self) -> None:
        fake = _FakeDepthAI([_packet(1)])
        source = DepthAIFrameSource(_config(), depthai_module=fake)

        source.start()

        self.assertEqual(fake.device_info_calls, [])
        self.assertEqual(len(fake.devices[0].args), 1)
        source.close()

    def test_read_returns_newest_frame_from_bounded_queue(self) -> None:
        fake = _FakeDepthAI([_packet(0)])
        source = DepthAIFrameSource(_config(queue_size=3), depthai_module=fake)
        source.start()
        startup_frame = source.read()
        self.assertTrue(np.all(startup_frame == 0))

        fake.output_queue.packets.extend([_packet(1), _packet(2), _packet(3)])
        newest = source.read()

        self.assertTrue(np.all(newest == 3))
        self.assertEqual(fake.output_queue.packets, [])
        source.close()

    def test_returned_frame_does_not_alias_sdk_packet_memory(self) -> None:
        packet = _packet(5)
        fake = _FakeDepthAI([packet])
        source = DepthAIFrameSource(_config(), depthai_module=fake)
        source.start()

        frame = source.read()
        packet.frame[:] = 9

        self.assertTrue(np.all(frame == 5))
        source.close()

    def test_get_cv_frame_bgr_channels_are_converted_to_contiguous_rgb(self) -> None:
        bgr = np.zeros((3, 4, 3), dtype=np.uint8)
        bgr[..., 0] = 11
        bgr[..., 1] = 22
        bgr[..., 2] = 33
        fake = _FakeDepthAI([_FakePacket(bgr)])
        source = DepthAIFrameSource(_config(), depthai_module=fake)
        source.start()

        rgb = source.read()

        self.assertTrue(rgb.flags.c_contiguous)
        self.assertTrue(np.all(rgb[..., 0] == 33))
        self.assertTrue(np.all(rgb[..., 1] == 22))
        self.assertTrue(np.all(rgb[..., 2] == 11))
        source.close()

    def test_native_planar_rgb_frame_avoids_opencv_dependency(self) -> None:
        planar_rgb = np.zeros((3, 3, 4), dtype=np.uint8)
        planar_rgb[0, ...] = 11
        planar_rgb[1, ...] = 22
        planar_rgb[2, ...] = 33
        packet = _FakeNativePacket(planar_rgb)
        fake = _FakeDepthAI([packet])
        source = DepthAIFrameSource(_config(), depthai_module=fake)
        source.start()

        rgb = source.read()

        self.assertEqual(rgb.shape, (3, 4, 3))
        self.assertTrue(rgb.flags.c_contiguous)
        self.assertTrue(np.all(rgb[..., 0] == 11))
        self.assertTrue(np.all(rgb[..., 1] == 22))
        self.assertTrue(np.all(rgb[..., 2] == 33))
        source.close()

    def test_runtime_native_interleaved_frame_does_not_alias_sdk_packet_memory(self) -> None:
        startup = _FakeNativePacket(np.zeros((3, 3, 4), dtype=np.uint8))
        runtime = _FakeNativePacket(np.full((3, 4, 3), 7, dtype=np.uint8))
        fake = _FakeDepthAI([startup])
        source = DepthAIFrameSource(_config(), depthai_module=fake)
        source.start()
        source.read()
        fake.output_queue.packets.append(runtime)

        frame = source.read()
        runtime.frame[:] = 9

        self.assertTrue(frame.flags.owndata)
        self.assertFalse(np.shares_memory(frame, runtime.frame))
        self.assertTrue(np.all(frame == 7))
        source.close()

    def test_read_timeout_is_deterministic_and_typed(self) -> None:
        clock = _FakeClock()
        fake = _FakeDepthAI([_packet(1)])
        source = DepthAIFrameSource(
            _config(read_timeout=0.025),
            depthai_module=fake,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
        source.start()
        source.read()

        with self.assertRaises(DepthAITimeoutError) as raised:
            source.read()

        self.assertEqual(raised.exception.code, "depthai_frame_timeout")
        self.assertGreaterEqual(clock.now, 0.025)
        source.close()

    def test_startup_timeout_releases_queue_and_device(self) -> None:
        clock = _FakeClock()
        fake = _FakeDepthAI([])
        source = DepthAIFrameSource(
            _config(startup_timeout=0.025),
            depthai_module=fake,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )

        with self.assertRaises(DepthAITimeoutError):
            source.start()

        self.assertTrue(source.is_closed)
        self.assertEqual(fake.output_queue.close_calls, 1)
        self.assertEqual(fake.devices[0].close_calls, 1)

    def test_permission_error_is_typed_and_does_not_expose_original_detail(self) -> None:
        fake = _FakeDepthAI(
            [_packet(1)],
            device_error=PermissionError("LIBUSB_ERROR_ACCESS token=very-secret"),
        )
        source = DepthAIFrameSource(_config(), depthai_module=fake)

        with self.assertRaises(DepthAIPermissionError) as raised:
            source.start()

        self.assertEqual(raised.exception.code, "depthai_permission_denied")
        self.assertNotIn("very-secret", str(raised.exception))

    def test_missing_device_error_is_typed_and_sanitized(self) -> None:
        fake = _FakeDepthAI(
            [_packet(1)],
            device_error=RuntimeError("X_LINK_DEVICE_NOT_FOUND serial=private"),
        )
        source = DepthAIFrameSource(_config(), depthai_module=fake)

        with self.assertRaises(DepthAIUnavailableError) as raised:
            source.start()

        self.assertNotIn("private", str(raised.exception))

    def test_invalid_startup_frame_is_typed_and_resources_are_released(self) -> None:
        malformed = _FakePacket(np.zeros((2, 2), dtype=np.uint8))
        fake = _FakeDepthAI([malformed])
        source = DepthAIFrameSource(_config(), depthai_module=fake)

        with self.assertRaises(DepthAIStartupError):
            source.start()

        self.assertEqual(fake.output_queue.close_calls, 1)
        self.assertEqual(fake.devices[0].close_calls, 1)

    def test_invalid_runtime_frame_is_a_typed_read_error(self) -> None:
        fake = _FakeDepthAI([_packet(1)])
        source = DepthAIFrameSource(_config(), depthai_module=fake)
        source.start()
        source.read()
        fake.output_queue.packets.append(
            _FakePacket(np.zeros((3, 4, 3), dtype=np.float32))
        )

        with self.assertRaises(DepthAIReadError):
            source.read()

        source.close()

    def test_lifecycle_checks_and_idempotent_close(self) -> None:
        fake = _FakeDepthAI([_packet(1)])
        source = DepthAIFrameSource(_config(), depthai_module=fake)

        with self.assertRaises(DepthAIStateError):
            source.read()

        source.start()
        self.assertIs(source.start(), source)
        source.close()
        source.close()

        self.assertEqual(fake.output_queue.close_calls, 1)
        self.assertEqual(fake.devices[0].close_calls, 1)
        with self.assertRaises(DepthAIStateError):
            source.read()
        with self.assertRaises(DepthAIStateError):
            source.start()

    def test_close_interrupts_waiting_reader_without_thread_leak(self) -> None:
        fake = _FakeDepthAI([_packet(1)])
        source = DepthAIFrameSource(
            _config(read_timeout=2.0, poll_interval=0.005),
            depthai_module=fake,
        )
        source.start()
        source.read()
        errors: list[BaseException] = []

        def wait_for_frame() -> None:
            try:
                source.read()
            except BaseException as exc:  # capture the worker result for assertion
                errors.append(exc)

        reader = threading.Thread(target=wait_for_frame, name="depthai-test-reader")
        reader.start()
        time.sleep(0.02)
        source.close()
        reader.join(timeout=0.5)

        self.assertFalse(reader.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], DepthAIStateError)

    def test_source_is_not_running_until_ready_and_close_interrupts_startup(self) -> None:
        fake = _FakeDepthAI([])
        source = DepthAIFrameSource(
            _config(startup_timeout=2.0, poll_interval=0.005),
            depthai_module=fake,
        )
        errors: list[BaseException] = []

        def start_source() -> None:
            try:
                source.start()
            except BaseException as exc:  # capture the worker result for assertion
                errors.append(exc)

        starter = threading.Thread(target=start_source, name="depthai-test-starter")
        starter.start()
        self.assertTrue(fake.device_created.wait(0.5))
        self.assertFalse(source.is_running)
        with self.assertRaises(DepthAIStateError):
            source.read()
        with self.assertRaises(DepthAIStateError):
            source.start()

        source.close()
        starter.join(timeout=0.5)

        self.assertFalse(starter.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], DepthAIStateError)
        self.assertEqual(fake.output_queue.close_calls, 1)
        self.assertEqual(fake.devices[0].close_calls, 1)

    def test_context_manager_closes_resources(self) -> None:
        fake = _FakeDepthAI([_packet(4)])

        with DepthAIFrameSource(_config(), depthai_module=fake) as source:
            self.assertTrue(source.is_running)
            self.assertTrue(np.all(source.read() == 4))

        self.assertTrue(source.is_closed)
        self.assertEqual(fake.devices[0].close_calls, 1)

    def test_config_rejects_invalid_bounds(self) -> None:
        invalid_values = (
            {"width": 0},
            {"height": 0},
            {"fps": 0},
            {"queue_size": 0},
            {"startup_timeout": 0},
            {"read_timeout": -1},
            {"poll_interval": 0},
            {"fps": float("nan")},
            {"startup_timeout": float("inf")},
            {"read_timeout": float("nan")},
            {"poll_interval": float("inf")},
        )
        for values in invalid_values:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    _config(**values)

    def test_read_rejects_non_finite_timeout_override(self) -> None:
        fake = _FakeDepthAI([_packet(1)])
        source = DepthAIFrameSource(_config(), depthai_module=fake)
        source.start()

        for timeout in (float("nan"), float("inf"), -1.0):
            with self.subTest(timeout=timeout):
                with self.assertRaises(ValueError):
                    source.read(timeout=timeout)
        source.close()


if __name__ == "__main__":
    unittest.main()
