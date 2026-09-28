"""Volatile WebSocket camera intent. Never persist or replay movement."""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Any

from nuvion_app.inference.camera_position import CameraPositionReconciler, _DIRECTION_COMMANDS
from nuvion_app.inference.motor import MotorController, MotorLimitReached

log = logging.getLogger(__name__)

CAMERA_CONTROL_QUEUE = "/user/queue/camera.control"
CAMERA_TELEMETRY_DESTINATION = "/app/device/camera.telemetry"


class CameraRealtimeControl:
    def __init__(self, controller: MotorController, publish: Callable[[dict], None], *, start=True):
        self.controller = controller
        self.publish = publish
        self._lock = threading.Lock()
        self._intent: dict[str, Any] | None = None
        self._generation = 0
        self._seen: dict[str, tuple[int, float]] = {}
        self._moving = False
        self._blocked_sessions: set[str] = set()
        self._stop_requested = False
        self._closed = threading.Event()
        self._thread = None
        if start:
            self._thread = threading.Thread(target=self._run, name="camera-control", daemon=True)
            self._thread.start()

    def accept(self, frame: dict) -> bool:
        now = time.time()
        if not isinstance(frame, dict):
            return False
        session, sequence, expires = frame.get("sessionId"), frame.get("sequence"), frame.get("expiresAtMs")
        action = frame.get("action")
        if (not isinstance(session, str) or not 1 <= len(session) <= 128
                or type(sequence) is not int or sequence < 1
                or type(expires) is not int or not 0 < expires / 1000 - now <= 1.0
                or action not in {*_DIRECTION_COMMANDS, "STATUS", "STOP"}
                or not isinstance(frame.get("requestId"), str)):
            return False
        with self._lock:
            self._seen = {k: v for k, v in self._seen.items() if now - v[1] < 30}
            self._blocked_sessions.intersection_update(self._seen)
            previous = self._seen.get(session)
            if previous and sequence <= previous[0]:
                return False
            if len(self._seen) >= 128 and session not in self._seen:
                return False
            self._seen[session] = (sequence, now)
            if (self._intent and self._intent["sessionId"] != session
                    and self._intent["expiresAtMs"] > now * 1000
                    and self._intent["action"] in _DIRECTION_COMMANDS and action != "STOP"):
                self.publish({"requestId": frame["requestId"], "sequence": sequence,
                              "status": "BUSY", "message": "다른 사용자가 카메라를 조작 중입니다."})
                return False
            if session in self._blocked_sessions and action in _DIRECTION_COMMANDS:
                return False
            if action == "STOP":
                if self._intent and self._intent["sessionId"] != session:
                    self._blocked_sessions.add(self._intent["sessionId"])
                self._blocked_sessions.discard(session)
            self._intent = dict(frame)
            self._generation += 1
            if action == "STOP":
                self._stop_requested = True
            # Manual control always outranks automatic tracking during the lease.
            if action != "STATUS":
                self.controller.manual_control_until = expires / 1000 + 0.5
        return True

    def disconnect(self) -> None:
        with self._lock:
            if self._intent:
                self._blocked_sessions.add(self._intent["sessionId"])
            self._intent = None
            self._generation += 1
            self._stop_requested = True

    def tick(self) -> None:
        with self._lock:
            frame = dict(self._intent) if self._intent else None
            generation = self._generation
            stop = self._stop_requested
            self._stop_requested = False
        expired = frame is None or frame["expiresAtMs"] <= time.time() * 1000
        if expired and not self._moving and not stop:
            return
        if expired and self._moving and frame:
            with self._lock:
                self._blocked_sessions.add(frame["sessionId"])
        action = "STOP" if expired or stop else frame["action"]
        status = "STOPPED" if action == "STOP" else "MOVING" if action in _DIRECTION_COMMANDS else "IDLE"
        try:
            if action in _DIRECTION_COMMANDS:
                self._moving = True  # even a UART exception requires a best-effort stop
                def is_active():
                    with self._lock:
                        return (self._intent is not None and not self._stop_requested
                                and self._intent["sessionId"] == frame["sessionId"]
                                and self._intent["action"] == action)
                response = self.controller.realtime_step(
                    _DIRECTION_COMMANDS[action], expires_at=frame["expiresAtMs"] / 1000,
                    is_active=is_active)
            else:
                response = self.controller.position_action(action)
                if action == "STOP":
                    self._moving = False
            state = CameraPositionReconciler._reported_state(action, response)
            state.update(controlVersion=2, limits=dict(self.controller.position_limits),
                         observedAt=datetime.now(timezone.utc).isoformat())
            result = {"status": status, "telemetry": state}
            if expired and frame and frame["action"] in _DIRECTION_COMMANDS:
                result.update(status="ERROR", message="조작 입력이 만료되어 정지했습니다.")
        except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
            # Never turn a partially executed move into a retry. Stop and report
            # the cause; a later explicit press is a new intent.
            state = None
            try:
                response = self.controller.position_action("STOP")
                state = CameraPositionReconciler._reported_state("STOP", response)
                state.update(controlVersion=2, limits=dict(self.controller.position_limits),
                             observedAt=datetime.now(timezone.utc).isoformat())
                self._moving = False
            except (OSError, RuntimeError, ValueError, TimeoutError):
                pass
            result = {"status": "LIMIT_REACHED" if isinstance(exc, MotorLimitReached) else "ERROR",
                      "message": str(exc)[:300], "telemetry": state}
            with self._lock:
                if frame:
                    self._blocked_sessions.add(frame["sessionId"])
                if generation == self._generation:
                    self._intent = None
            self._stop_requested = self._moving
        if frame:
            result.update(requestId=frame["requestId"], sequence=frame["sequence"])
            self.publish(result)
        if action not in _DIRECTION_COMMANDS:
            with self._lock:
                if generation == self._generation:
                    self._intent = None

    @property
    def ready(self) -> bool:
        return not self._closed.is_set() and self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        try:
            while not self._closed.wait(0.08):
                self.tick()
        except Exception:
            log.exception("camera realtime worker stopped")
            self._closed.set()
            try:
                self.controller.position_action("STOP")
            except Exception:
                log.exception("camera realtime emergency hold failed")

    def close(self) -> None:
        self.disconnect()
        self._closed.set()
        if self._thread:
            self._thread.join(timeout=3)
        self.controller.position_action("STOP")
