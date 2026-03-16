from __future__ import annotations

import threading
from typing import Any, Callable


DEVICE_STATE_RUNNING = "RUNNING"
DEVICE_STATE_ERROR = "ERROR"
DEVICE_STATE_NETWORK_ISSUE = "NETWORK_ISSUE"

CONNECTIVITY_QUALITY_GOOD = "GOOD"
CONNECTIVITY_QUALITY_POOR = "POOR"

STATE_MESSAGE_BY_STATUS = {
    DEVICE_STATE_RUNNING: "heartbeat",
    DEVICE_STATE_ERROR: "불량 감지됨",
    DEVICE_STATE_NETWORK_ISSUE: "통신 상태 확인 필요",
}


class DeviceStateCoordinator:
    def __init__(
        self,
        *,
        send_message: Callable[[dict[str, Any]], bool],
        line_id: int | None,
        process_id: int | None,
    ) -> None:
        self._send_message = send_message
        self._line_id = line_id
        self._process_id = process_id
        self._lock = threading.Lock()
        self._detection_state = DEVICE_STATE_RUNNING
        self._connectivity_quality = CONNECTIVITY_QUALITY_GOOD

    def emit_heartbeat(self) -> bool:
        return self._send_message(self.current_payload())

    def current_payload(self) -> dict[str, Any]:
        with self._lock:
            status = self._effective_state_locked()
            return self._build_payload(status)

    def set_detection_state(self, detection_state: str) -> None:
        normalized = detection_state.strip().upper()
        if normalized not in {DEVICE_STATE_RUNNING, DEVICE_STATE_ERROR}:
            return
        self._update_state(detection_state=normalized)

    def set_connectivity_quality(self, quality: str) -> None:
        normalized = quality.strip().upper()
        if normalized not in {CONNECTIVITY_QUALITY_GOOD, CONNECTIVITY_QUALITY_POOR}:
            return
        self._update_state(connectivity_quality=normalized)

    def _update_state(
        self,
        *,
        detection_state: str | None = None,
        connectivity_quality: str | None = None,
    ) -> None:
        payload: dict[str, Any] | None = None
        with self._lock:
            previous_effective_state = self._effective_state_locked()
            if detection_state is not None:
                self._detection_state = detection_state
            if connectivity_quality is not None:
                self._connectivity_quality = connectivity_quality
            next_effective_state = self._effective_state_locked()
            if next_effective_state != previous_effective_state:
                payload = self._build_payload(next_effective_state)

        if payload is not None:
            self._send_message(payload)

    def _effective_state_locked(self) -> str:
        if self._detection_state == DEVICE_STATE_ERROR:
            return DEVICE_STATE_ERROR
        if self._connectivity_quality == CONNECTIVITY_QUALITY_POOR:
            return DEVICE_STATE_NETWORK_ISSUE
        return DEVICE_STATE_RUNNING

    def _build_payload(self, status: str) -> dict[str, Any]:
        return {
            "status": status,
            "message": STATE_MESSAGE_BY_STATUS[status],
            "lineId": self._line_id,
            "processId": self._process_id,
        }
