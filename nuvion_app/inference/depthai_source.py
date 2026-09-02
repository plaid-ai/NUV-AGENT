"""DepthAI-backed RGB frame source for Luxonis OAK cameras.

The :mod:`depthai` dependency is deliberately imported only by ``start()`` so
the rest of the inference package remains importable on hosts without OAK
camera support.  The host-side DepthAI output queue is bounded and
non-blocking; ``read()`` drains the current backlog and returns the newest
frame.
"""

from __future__ import annotations

import errno
import importlib
import math
import threading
import time
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable

import numpy as np


class DepthAIError(RuntimeError):
    """Base class for safe, typed DepthAI source failures."""

    code = "depthai_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)


class DepthAIUnavailableError(DepthAIError):
    """DepthAI is not installed or the requested OAK device is unavailable."""

    code = "depthai_unavailable"


class DepthAIPermissionError(DepthAIError):
    """The process does not have permission to access the OAK USB device."""

    code = "depthai_permission_denied"


class DepthAIStartupError(DepthAIError):
    """The OAK pipeline could not be configured or started."""

    code = "depthai_startup_failed"


class DepthAITimeoutError(DepthAIError):
    """A frame was not received before the configured deadline."""

    code = "depthai_frame_timeout"


class DepthAIStateError(DepthAIError):
    """The frame source lifecycle is being used in an invalid state."""

    code = "depthai_invalid_state"


class DepthAIReadError(DepthAIError):
    """A running OAK source failed to produce a valid RGB frame."""

    code = "depthai_read_failed"


@dataclass(frozen=True)
class DepthAIConfig:
    """Configuration for an OAK RGB frame stream."""

    width: int = 640
    height: int = 480
    fps: float = 30.0
    device_id: str | None = None
    queue_size: int = 1
    startup_timeout: float = 5.0
    read_timeout: float = 1.0
    poll_interval: float = 0.01

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError("width must be greater than zero")
        if self.height <= 0:
            raise ValueError("height must be greater than zero")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be greater than zero")
        if self.queue_size <= 0:
            raise ValueError("queue_size must be greater than zero")
        if not math.isfinite(self.startup_timeout) or self.startup_timeout <= 0:
            raise ValueError("startup_timeout must be greater than zero")
        if not math.isfinite(self.read_timeout) or self.read_timeout < 0:
            raise ValueError("read_timeout must be finite and not negative")
        if not math.isfinite(self.poll_interval) or self.poll_interval <= 0:
            raise ValueError("poll_interval must be greater than zero")

        normalized_device_id = (self.device_id or "").strip() or None
        object.__setattr__(self, "device_id", normalized_device_id)


@dataclass
class _DepthAIResources:
    pipeline: Any
    output_queue: Any
    device: Any | None = None


class DepthAIFrameSource:
    """Explicit-lifecycle, latest-frame RGB source for an OAK camera.

    ``start()`` waits for the first valid frame, making successful return a
    useful readiness signal. ``read()`` returns a copied ``H x W x 3`` RGB
    array and raises :class:`DepthAITimeoutError` instead of returning a stale
    frame. No Python worker thread is created by this class.
    """

    STREAM_NAME = "nuvion_rgb"

    def __init__(
        self,
        config: DepthAIConfig | None = None,
        *,
        depthai_module: ModuleType | Any | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config or DepthAIConfig()
        self._depthai_module = depthai_module
        self._monotonic = monotonic
        self._sleep = sleep
        self._lifecycle_lock = threading.RLock()
        self._read_lock = threading.Lock()
        self._closed_event = threading.Event()
        self._resources: _DepthAIResources | None = None
        self._pending_frame: np.ndarray | None = None
        self._starting = False
        self._started = False
        self._closed = False

    @property
    def is_running(self) -> bool:
        with self._lifecycle_lock:
            return self._started and not self._closed

    @property
    def is_closed(self) -> bool:
        with self._lifecycle_lock:
            return self._closed

    def start(self) -> DepthAIFrameSource:
        """Open the selected device and wait for its first RGB frame."""

        with self._lifecycle_lock:
            if self._closed:
                raise DepthAIStateError("DepthAI source is closed")
            if self._started:
                return self
            if self._starting:
                raise DepthAIStateError("DepthAI source startup is already in progress")
            self._starting = True

        try:
            depthai = self._load_depthai()
            resources = self._open_resources(depthai)
        except DepthAIError:
            self._mark_failed_start()
            raise
        except Exception as exc:
            self._mark_failed_start()
            raise self._classify_startup_error(exc) from None

        with self._lifecycle_lock:
            if self._closed:
                self._release_resources(resources)
                raise DepthAIStateError("DepthAI source was closed during startup")
            self._resources = resources

        try:
            packet = self._wait_for_packet(
                resources.output_queue,
                timeout=self.config.startup_timeout,
                operation="startup",
            )
            first_frame = self._packet_to_rgb(packet, startup=True)
        except DepthAIError:
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise self._classify_startup_error(exc) from None

        with self._lifecycle_lock:
            if self._closed:
                raise DepthAIStateError("DepthAI source was closed during startup")
            self._pending_frame = first_frame
            self._starting = False
            self._started = True
        return self

    def read(self, timeout: float | None = None) -> np.ndarray:
        """Return the newest RGB frame available before ``timeout`` seconds."""

        effective_timeout = self.config.read_timeout if timeout is None else timeout
        if not math.isfinite(effective_timeout) or effective_timeout < 0:
            raise ValueError("timeout must be finite and not negative")

        with self._read_lock:
            with self._lifecycle_lock:
                if self._closed:
                    raise DepthAIStateError("DepthAI source is closed")
                if not self._started or self._resources is None:
                    raise DepthAIStateError("DepthAI source has not been started")
                if self._pending_frame is not None:
                    frame = self._pending_frame
                    self._pending_frame = None
                    return frame.copy()
                output_queue = self._resources.output_queue

            try:
                packet = self._wait_for_packet(
                    output_queue,
                    timeout=effective_timeout,
                    operation="read",
                )
                return self._packet_to_rgb(packet, startup=False)
            except (DepthAITimeoutError, DepthAIStateError):
                raise
            except DepthAIError:
                raise
            except Exception:
                raise DepthAIReadError("DepthAI frame read failed") from None

    def close(self) -> None:
        """Stop the pipeline and release all device resources; idempotent."""

        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            self._starting = False
            self._started = False
            self._pending_frame = None
            self._closed_event.set()
            resources = self._resources
            self._resources = None

        if resources is not None:
            self._release_resources(resources)

    def __enter__(self) -> DepthAIFrameSource:
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _load_depthai(self) -> Any:
        if self._depthai_module is not None:
            return self._depthai_module
        try:
            depthai = importlib.import_module("depthai")
        except (ImportError, ModuleNotFoundError):
            raise DepthAIUnavailableError(
                "DepthAI support is unavailable; install the optional depthai runtime"
            ) from None
        self._depthai_module = depthai
        return depthai

    def _open_resources(self, depthai: Any) -> _DepthAIResources:
        if self._depthai_major_version(depthai) >= 3:
            return self._open_v3_resources(depthai)
        return self._open_v2_resources(depthai)

    def _open_v2_resources(self, depthai: Any) -> _DepthAIResources:
        pipeline = depthai.Pipeline()
        camera = pipeline.create(depthai.node.ColorCamera)
        camera.setPreviewSize(self.config.width, self.config.height)
        camera.setInterleaved(False)
        camera.setFps(self.config.fps)

        color_order = depthai.ColorCameraProperties.ColorOrder.RGB
        camera.setColorOrder(color_order)
        self._set_rgb_socket(camera, depthai)

        output = pipeline.create(depthai.node.XLinkOut)
        output.setStreamName(self.STREAM_NAME)
        camera.preview.link(output.input)

        if self.config.device_id is None:
            device = depthai.Device(pipeline)
        else:
            device_info = depthai.DeviceInfo(self.config.device_id)
            device = depthai.Device(pipeline, device_info)

        try:
            output_queue = device.getOutputQueue(
                name=self.STREAM_NAME,
                maxSize=self.config.queue_size,
                blocking=False,
            )
        except Exception:
            self._call_quietly(device, "close")
            raise
        return _DepthAIResources(pipeline=pipeline, output_queue=output_queue, device=device)

    def _open_v3_resources(self, depthai: Any) -> _DepthAIResources:
        device = None
        pipeline = None
        try:
            if self.config.device_id is None:
                pipeline = depthai.Pipeline()
            else:
                device = depthai.Device(depthai.DeviceInfo(self.config.device_id))
                pipeline = depthai.Pipeline(device)

            camera_builder = pipeline.create(depthai.node.Camera)
            socket = self._rgb_socket(depthai)
            camera = camera_builder.build(socket) if socket is not None else camera_builder.build()

            image_type = self._rgb_image_type(depthai)
            request_kwargs: dict[str, Any] = {"fps": self.config.fps}
            if image_type is not None:
                request_kwargs["type"] = image_type
            camera_output = camera.requestOutput(
                (self.config.width, self.config.height),
                **request_kwargs,
            )
            output_queue = camera_output.createOutputQueue(
                maxSize=self.config.queue_size,
                blocking=False,
            )
            pipeline.start()
        except Exception:
            self._call_quietly(pipeline, "stop")
            self._call_quietly(pipeline, "wait")
            self._call_quietly(device, "close")
            raise
        return _DepthAIResources(pipeline=pipeline, output_queue=output_queue, device=device)

    def _wait_for_packet(self, output_queue: Any, *, timeout: float, operation: str) -> Any:
        deadline = self._monotonic() + timeout
        while True:
            if self._closed_event.is_set():
                raise DepthAIStateError("DepthAI source is closed")
            try:
                packet = self._try_get_latest(output_queue)
            except Exception:
                if self._closed_event.is_set():
                    raise DepthAIStateError("DepthAI source is closed") from None
                if operation == "startup":
                    raise DepthAIStartupError("DepthAI startup frame read failed") from None
                raise DepthAIReadError("DepthAI frame read failed") from None

            if packet is not None:
                return packet

            remaining = deadline - self._monotonic()
            if remaining <= 0:
                if operation == "startup":
                    raise DepthAITimeoutError("DepthAI startup timed out waiting for an RGB frame")
                raise DepthAITimeoutError("DepthAI frame read timed out")
            self._sleep(min(self.config.poll_interval, remaining))

    def _try_get_latest(self, output_queue: Any) -> Any | None:
        try_get_all = getattr(output_queue, "tryGetAll", None)
        if callable(try_get_all):
            packets = try_get_all()
            return packets[-1] if packets else None

        try_get = getattr(output_queue, "tryGet", None)
        if not callable(try_get):
            raise AttributeError("DepthAI output queue does not support non-blocking reads")

        latest = try_get()
        if latest is None:
            return None
        for _ in range(self.config.queue_size - 1):
            packet = try_get()
            if packet is None:
                break
            latest = packet
        return latest

    def _packet_to_rgb(self, packet: Any, *, startup: bool) -> np.ndarray:
        try:
            get_frame = getattr(packet, "getFrame", None)
            if callable(get_frame):
                frame = np.asarray(get_frame())
                if frame.shape == (3, self.config.height, self.config.width):
                    frame = np.transpose(frame, (1, 2, 0))
                elif frame.shape != (self.config.height, self.config.width, 3):
                    raise ValueError("unexpected native RGB frame shape")
                if frame.dtype != np.uint8:
                    raise ValueError("unexpected native RGB frame dtype")
                # The camera is explicitly configured as RGB888p/RGB888i.
                # getFrame() does not require opencv-python, unlike getCvFrame().
                # Always detach from the SDK packet. ``getFrame()`` may expose
                # packet-owned memory and ``ascontiguousarray`` is allowed to
                # return that same allocation for interleaved RGB frames.
                return np.array(frame, dtype=np.uint8, copy=True, order="C")

            frame = np.asarray(packet.getCvFrame())
            if (
                frame.shape != (self.config.height, self.config.width, 3)
                or frame.dtype != np.uint8
            ):
                raise ValueError("unexpected OpenCV BGR frame")
            # Compatibility fallback for SDK packet doubles and older APIs:
            # getCvFrame() is OpenCV-ready BGR, so convert it to RGB.
            return np.array(frame[..., ::-1], dtype=np.uint8, copy=True, order="C")
        except Exception:
            if startup:
                raise DepthAIStartupError("DepthAI produced an invalid startup RGB frame") from None
            raise DepthAIReadError("DepthAI produced an invalid RGB frame") from None

    def _mark_failed_start(self) -> None:
        with self._lifecycle_lock:
            self._closed = True
            self._starting = False
            self._started = False
            self._closed_event.set()

    def _release_resources(self, resources: _DepthAIResources) -> None:
        self._call_quietly(resources.output_queue, "close")
        self._call_quietly(resources.pipeline, "stop")
        self._call_quietly(resources.pipeline, "wait")
        self._call_quietly(resources.device, "close")

    @staticmethod
    def _call_quietly(resource: Any, method_name: str) -> None:
        if resource is None:
            return
        method = getattr(resource, method_name, None)
        if not callable(method):
            return
        try:
            method()
        except Exception:
            # Every release hook is attempted. Close is intentionally idempotent
            # and must remain safe during interpreter/service shutdown.
            pass

    @staticmethod
    def _depthai_major_version(depthai: Any) -> int:
        raw_version = str(getattr(depthai, "__version__", "2"))
        try:
            return int(raw_version.split(".", 1)[0])
        except ValueError:
            return 2

    @classmethod
    def _rgb_socket(cls, depthai: Any) -> Any | None:
        sockets = getattr(depthai, "CameraBoardSocket", None)
        if sockets is None:
            return None
        return getattr(sockets, "CAM_A", getattr(sockets, "RGB", None))

    @classmethod
    def _set_rgb_socket(cls, camera: Any, depthai: Any) -> None:
        socket = cls._rgb_socket(depthai)
        setter = getattr(camera, "setBoardSocket", None)
        if socket is not None and callable(setter):
            setter(socket)

    @staticmethod
    def _rgb_image_type(depthai: Any) -> Any | None:
        image_types = getattr(getattr(depthai, "ImgFrame", None), "Type", None)
        if image_types is None:
            return None
        return getattr(image_types, "RGB888i", getattr(image_types, "RGB888p", None))

    @staticmethod
    def _classify_startup_error(exc: Exception) -> DepthAIError:
        message = str(exc).lower()
        os_errno = getattr(exc, "errno", None)
        permission_markers = (
            "permission",
            "access denied",
            "libusb_error_access",
            "udev",
            "operation not permitted",
        )
        if (
            isinstance(exc, PermissionError)
            or os_errno in {errno.EACCES, errno.EPERM}
            or any(marker in message for marker in permission_markers)
        ):
            return DepthAIPermissionError(
                "DepthAI USB access was denied; verify the Luxonis udev permissions"
            )

        unavailable_markers = (
            "no available devices",
            "no device found",
            "device not found",
            "failed to find",
            "x_link_device_not_found",
            "xlink_device_not_found",
        )
        if any(marker in message for marker in unavailable_markers):
            return DepthAIUnavailableError("No matching DepthAI device is available")
        return DepthAIStartupError("DepthAI camera startup failed")


__all__ = [
    "DepthAIConfig",
    "DepthAIError",
    "DepthAIFrameSource",
    "DepthAIPermissionError",
    "DepthAIReadError",
    "DepthAIStartupError",
    "DepthAIStateError",
    "DepthAITimeoutError",
    "DepthAIUnavailableError",
]
