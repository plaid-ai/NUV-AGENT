from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from typing import Any

DEVICE_STATE_RUNNING = "RUNNING"
DEVICE_STATE_ERROR = "ERROR"
DEVICE_STATE_NETWORK_ISSUE = "NETWORK_ISSUE"

CONNECTIVITY_QUALITY_GOOD = "GOOD"
CONNECTIVITY_QUALITY_POOR = "POOR"

RUNTIME_STATUS_RUNNING = "RUNNING"
RUNTIME_STATUS_ERROR = "ERROR"
INSPECTION_STATUS_NORMAL = "NORMAL"
INSPECTION_STATUS_DEFECT = "DEFECT"

STATE_MESSAGE_BY_STATUS = {
    DEVICE_STATE_RUNNING: "heartbeat",
    DEVICE_STATE_NETWORK_ISSUE: "통신 상태 확인 필요",
}
RUNTIME_ERROR_MESSAGE = "Agent runtime error"
INSPECTION_DEFECT_MESSAGE = "불량 감지됨"


class DeviceStateCoordinator:
    def __init__(
        self,
        *,
        send_message: Callable[[dict[str, Any]], bool],
        line_id: int | None,
        process_id: int | None,
        telemetry: Mapping[str, Any] | None = None,
        runtime_telemetry_provider: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        self._send_message = send_message
        self._line_id = line_id
        self._process_id = process_id
        self._lock = threading.Lock()
        self._runtime_status = RUNTIME_STATUS_RUNNING
        self._inspection_status = INSPECTION_STATUS_NORMAL
        self._connectivity_status = CONNECTIVITY_QUALITY_GOOD
        self._telemetry = dict(telemetry or {})
        self._runtime_telemetry_provider = runtime_telemetry_provider

    def emit_heartbeat(self) -> bool:
        return self._send_message(self.current_payload())

    def current_payload(self) -> dict[str, Any]:
        with self._lock:
            status = self._effective_state_locked()
            return self._build_payload(status)

    def set_detection_state(self, detection_state: str) -> None:
        normalized = detection_state.strip().upper()
        mapping = {
            DEVICE_STATE_RUNNING: INSPECTION_STATUS_NORMAL,
            DEVICE_STATE_ERROR: INSPECTION_STATUS_DEFECT,
            INSPECTION_STATUS_NORMAL: INSPECTION_STATUS_NORMAL,
            INSPECTION_STATUS_DEFECT: INSPECTION_STATUS_DEFECT,
        }
        if normalized not in mapping:
            return
        self.set_inspection_status(mapping[normalized])

    def set_connectivity_quality(self, quality: str) -> None:
        self.set_connectivity_status(quality)

    def set_runtime_status(self, status: str) -> None:
        normalized = status.strip().upper()
        if normalized not in {RUNTIME_STATUS_RUNNING, RUNTIME_STATUS_ERROR}:
            return
        self._update_state(runtime_status=normalized)

    def set_inspection_status(self, status: str) -> None:
        normalized = status.strip().upper()
        if normalized not in {INSPECTION_STATUS_NORMAL, INSPECTION_STATUS_DEFECT}:
            return
        self._update_state(inspection_status=normalized)

    def set_connectivity_status(self, quality: str) -> None:
        normalized = quality.strip().upper()
        if normalized not in {CONNECTIVITY_QUALITY_GOOD, CONNECTIVITY_QUALITY_POOR}:
            return
        self._update_state(connectivity_status=normalized)

    def _update_state(
        self,
        *,
        runtime_status: str | None = None,
        inspection_status: str | None = None,
        connectivity_status: str | None = None,
    ) -> None:
        payload: dict[str, Any] | None = None
        with self._lock:
            previous_state = self._state_tuple_locked()
            if runtime_status is not None:
                self._runtime_status = runtime_status
            if inspection_status is not None:
                self._inspection_status = inspection_status
            if connectivity_status is not None:
                self._connectivity_status = connectivity_status
            if self._state_tuple_locked() != previous_state:
                payload = self._build_payload(self._effective_state_locked())

        if payload is not None:
            self._send_message(payload)

    def _effective_state_locked(self) -> str:
        if self._runtime_status == RUNTIME_STATUS_ERROR:
            return DEVICE_STATE_ERROR
        if self._inspection_status == INSPECTION_STATUS_DEFECT:
            return DEVICE_STATE_ERROR
        if self._connectivity_status == CONNECTIVITY_QUALITY_POOR:
            return DEVICE_STATE_NETWORK_ISSUE
        return DEVICE_STATE_RUNNING

    def _state_tuple_locked(self) -> tuple[str, str, str]:
        return self._runtime_status, self._inspection_status, self._connectivity_status

    def _build_payload(self, status: str) -> dict[str, Any]:
        payload = {
            "status": status,
            "message": self._message_locked(status),
            "lineId": self._line_id,
            "processId": self._process_id,
            "runtimeStatus": self._runtime_status,
            "inspectionStatus": self._inspection_status,
            "connectivityStatus": self._connectivity_status,
        }
        payload.update(self._telemetry)
        if self._runtime_telemetry_provider is not None:
            dynamic_telemetry = dict(self._runtime_telemetry_provider())
            runtime_telemetry = payload.get("runtimeTelemetry")
            if isinstance(runtime_telemetry, Mapping):
                merged_runtime_telemetry = dict(runtime_telemetry)
                merged_runtime_telemetry.update(dynamic_telemetry)
                payload["runtimeTelemetry"] = merged_runtime_telemetry
            else:
                payload["runtimeTelemetry"] = dynamic_telemetry
        return payload

    def _message_locked(self, status: str) -> str:
        if self._runtime_status == RUNTIME_STATUS_ERROR:
            return RUNTIME_ERROR_MESSAGE
        if self._inspection_status == INSPECTION_STATUS_DEFECT:
            return INSPECTION_DEFECT_MESSAGE
        return STATE_MESSAGE_BY_STATUS[status]
