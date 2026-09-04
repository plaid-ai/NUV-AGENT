from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from nuvion_app.inference.depthai_source import DepthAITimeoutError


class DepthAIGStreamerBridgeError(RuntimeError):
    """Raised when a DepthAI frame cannot be delivered to GStreamer."""


@dataclass(frozen=True)
class DepthAIGStreamerBridgeStats:
    """Bounded bridge/appsrc counters safe to sample from another thread."""

    push_attempts: int
    push_succeeded: int
    push_failed: int
    appsrc_current_level_buffers: int | None
    appsrc_current_level_bytes: int | None
    appsrc_current_level_time: int | None
    appsrc_in: int | None
    appsrc_out: int | None
    appsrc_dropped: int | None


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
        metrics_interval_sec: float = 30.0,
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
        self.metrics_interval_sec = float(metrics_interval_sec)
        if (
            not math.isfinite(self.metrics_interval_sec)
            or self.metrics_interval_sec < 0
        ):
            raise ValueError("metrics_interval_sec must be finite and not negative")
        self.on_failure = on_failure
        self.log = logger or logging.getLogger(__name__)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._failure: BaseException | None = None
        self._lock = threading.Lock()
        self._push_attempts = 0
        self._push_succeeded = 0
        self._push_failed = 0
        self._next_metrics_at = 0.0

    @property
    def failure(self) -> BaseException | None:
        with self._lock:
            return self._failure

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    @staticmethod
    def _optional_uint_property(element: Any, name: str) -> int | None:
        try:
            find_property = getattr(element, "find_property", None)
            if callable(find_property) and find_property(name) is None:
                return None
            get_property = getattr(element, "get_property", None)
            if not callable(get_property):
                return None
            value = get_property(name)
            return int(value) if value is not None else None
        except Exception:  # noqa: BLE001 - diagnostics must not stop capture.
            return None

    def stats_snapshot(self) -> DepthAIGStreamerBridgeStats:
        """Return bridge counters plus optional version-dependent appsrc levels."""

        with self._lock:
            attempts = self._push_attempts
            succeeded = self._push_succeeded
            failed = self._push_failed
        return DepthAIGStreamerBridgeStats(
            push_attempts=attempts,
            push_succeeded=succeeded,
            push_failed=failed,
            appsrc_current_level_buffers=self._optional_uint_property(
                self.appsrc,
                "current-level-buffers",
            ),
            appsrc_current_level_bytes=self._optional_uint_property(
                self.appsrc,
                "current-level-bytes",
            ),
            appsrc_current_level_time=self._optional_uint_property(
                self.appsrc,
                "current-level-time",
            ),
            appsrc_in=self._optional_uint_property(self.appsrc, "in"),
            appsrc_out=self._optional_uint_property(self.appsrc, "out"),
            appsrc_dropped=self._optional_uint_property(self.appsrc, "dropped"),
        )

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        with self._lock:
            self._failure = None
            self._push_attempts = 0
            self._push_succeeded = 0
            self._push_failed = 0
        self._next_metrics_at = time.monotonic() + self.metrics_interval_sec
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
        timeout = max(float(join_timeout), 0.0)
        thread = self._thread
        current_thread = threading.current_thread()

        # In the normal path, let the bounded/non-blocking DepthAI read return
        # after observing _stop before closing its native queue/device. Closing
        # those resources while tryGetAll() is active can race the SDK teardown.
        if thread is not None and thread is not current_thread:
            thread.join(timeout=timeout)

        close_error: BaseException | None = None
        try:
            self.frame_source.close()
        except BaseException as exc:  # Rejoin before preserving the close failure.
            close_error = exc

        # A reader that ignored the cooperative stop gets one forced close to
        # interrupt native I/O and one more bounded opportunity to finish.
        if thread is not None and thread is not current_thread and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():
                self.log.warning(
                    "[DEPTHAI] capture thread did not stop after forced source close"
                )
        if thread is None or not thread.is_alive():
            self._thread = None
        if close_error is not None:
            raise close_error

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
        with self._lock:
            self._push_attempts += 1
        try:
            # A single owned bytes object crosses the PyGObject boundary.
            # ``new_allocate`` + ``fill`` creates a second full-size native
            # allocation/copy; passing a memoryview avoids that allocation but
            # PyGObject marshals it byte-by-byte and cannot sustain 30 fps.
            buffer = self.gst.Buffer.new_wrapped(frame.tobytes())
            flow = self.appsrc.emit("push-buffer", buffer)
            # appsrc keeps its own native reference when needed. Do not retain
            # the PyGObject wrapper across pump iterations.
            del buffer
        except BaseException:
            with self._lock:
                self._push_failed += 1
            raise
        if flow != self.gst.FlowReturn.OK:
            with self._lock:
                self._push_failed += 1
            raise DepthAIGStreamerBridgeError(f"GStreamer appsrc push failed: {flow}")
        with self._lock:
            self._push_succeeded += 1
        self._maybe_log_metrics()

    def _maybe_log_metrics(self) -> None:
        if self.metrics_interval_sec <= 0:
            return
        now = time.monotonic()
        if now < self._next_metrics_at:
            return
        self._next_metrics_at = now + self.metrics_interval_sec
        stats = self.stats_snapshot()
        self.log.info(
            "[DEPTHAI] bridge push=%d ok=%d failed=%d "
            "appsrc_buffers=%s appsrc_bytes=%s appsrc_in=%s "
            "appsrc_out=%s appsrc_dropped=%s",
            stats.push_attempts,
            stats.push_succeeded,
            stats.push_failed,
            stats.appsrc_current_level_buffers,
            stats.appsrc_current_level_bytes,
            stats.appsrc_in,
            stats.appsrc_out,
            stats.appsrc_dropped,
        )

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
