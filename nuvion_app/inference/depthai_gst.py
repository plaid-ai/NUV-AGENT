from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

import numpy as np

from nuvion_app.inference.depthai_source import DepthAITimeoutError


class DepthAIGStreamerBridgeError(RuntimeError):
    """Raised when a DepthAI frame cannot be delivered to GStreamer."""


class DepthAIGStreamerBridge:
    """Pump bounded DepthAI RGB frames into a live GStreamer appsrc."""

    def __init__(
        self,
        *,
        frame_source: Any,
        appsrc: Any,
        gst: Any,
        width: int,
        height: int,
        read_timeout: float = 2.0,
        max_consecutive_timeouts: int = 3,
        on_failure: Callable[[BaseException], None] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.frame_source = frame_source
        self.appsrc = appsrc
        self.gst = gst
        self.width = int(width)
        self.height = int(height)
        self.read_timeout = max(float(read_timeout), 0.1)
        self.max_consecutive_timeouts = max(int(max_consecutive_timeouts), 1)
        self.on_failure = on_failure
        self.log = logger or logging.getLogger(__name__)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._failure: BaseException | None = None
        self._lock = threading.Lock()

    @property
    def failure(self) -> BaseException | None:
        with self._lock:
            return self._failure

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        with self._lock:
            self._failure = None
        self.frame_source.start()
        try:
            thread = threading.Thread(
                target=self._pump,
                name="nuvion-depthai-appsrc",
                daemon=True,
            )
            self._thread = thread
            thread.start()
        except BaseException:
            self._thread = None
            self.frame_source.close()
            raise

    def close(self, *, join_timeout: float = 5.0) -> None:
        self._stop.set()
        try:
            self.frame_source.close()
        finally:
            thread = self._thread
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=max(float(join_timeout), 0.0))
                if thread.is_alive():
                    self.log.warning("[DEPTHAI] capture thread did not stop before timeout")
            if thread is None or not thread.is_alive():
                self._thread = None

    def _record_failure(self, exc: BaseException) -> None:
        with self._lock:
            if self._failure is not None:
                return
            self._failure = exc
        self._stop.set()
        if self.on_failure is not None:
            try:
                self.on_failure(exc)
            except Exception as callback_error:  # noqa: BLE001 - preserve original failure.
                self.log.error("[DEPTHAI] failure callback failed: %s", callback_error)

    def _validate_frame(self, frame: Any) -> np.ndarray:
        array = np.asarray(frame)
        expected_shape = (self.height, self.width, 3)
        if array.dtype != np.uint8 or array.shape != expected_shape:
            raise DepthAIGStreamerBridgeError(
                f"DepthAI RGB frame must be uint8 {expected_shape}, got "
                f"dtype={array.dtype} shape={array.shape}"
            )
        return np.ascontiguousarray(array)

    def _push(self, frame: np.ndarray) -> None:
        buffer = self.gst.Buffer.new_allocate(None, int(frame.nbytes), None)
        buffer.fill(0, frame.tobytes())
        flow = self.appsrc.emit("push-buffer", buffer)
        if flow != self.gst.FlowReturn.OK:
            raise DepthAIGStreamerBridgeError(f"GStreamer appsrc push failed: {flow}")

    def _pump(self) -> None:
        consecutive_timeouts = 0
        try:
            while not self._stop.is_set():
                try:
                    frame = self.frame_source.read(timeout=self.read_timeout)
                except DepthAITimeoutError as exc:
                    if self._stop.is_set():
                        return
                    consecutive_timeouts += 1
                    if consecutive_timeouts >= self.max_consecutive_timeouts:
                        raise DepthAIGStreamerBridgeError(
                            "DepthAI camera stopped producing frames"
                        ) from exc
                    continue
                consecutive_timeouts = 0
                self._push(self._validate_frame(frame))
        except BaseException as exc:  # noqa: BLE001 - thread boundary must surface failures.
            if not self._stop.is_set():
                self._record_failure(exc)
