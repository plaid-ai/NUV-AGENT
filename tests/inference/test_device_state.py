from __future__ import annotations

import unittest

from nuvion_app.inference.device_state import (
    CONNECTIVITY_QUALITY_GOOD,
    CONNECTIVITY_QUALITY_POOR,
    DEVICE_STATE_ERROR,
    DEVICE_STATE_NETWORK_ISSUE,
    DEVICE_STATE_RUNNING,
    DeviceStateCoordinator,
)


class DeviceStateCoordinatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sent_payloads: list[dict[str, object]] = []
        self.coordinator = DeviceStateCoordinator(
            send_message=self._capture,
            line_id=1,
            process_id=2,
        )

    def _capture(self, payload: dict[str, object]) -> bool:
        self.sent_payloads.append(payload)
        return True

    def test_heartbeat_uses_current_running_state(self) -> None:
        self.coordinator.emit_heartbeat()

        self.assertEqual(
            self.sent_payloads,
            [{"status": DEVICE_STATE_RUNNING, "message": "heartbeat", "lineId": 1, "processId": 2}],
        )

    def test_detection_error_emits_error_state_immediately(self) -> None:
        self.coordinator.set_detection_state(DEVICE_STATE_ERROR)

        self.assertEqual(
            self.sent_payloads,
            [{"status": DEVICE_STATE_ERROR, "message": "불량 감지됨", "lineId": 1, "processId": 2}],
        )

    def test_connectivity_poor_emits_network_issue_state_immediately(self) -> None:
        self.coordinator.set_connectivity_quality(CONNECTIVITY_QUALITY_POOR)

        self.assertEqual(
            self.sent_payloads,
            [{"status": DEVICE_STATE_NETWORK_ISSUE, "message": "통신 상태 확인 필요", "lineId": 1, "processId": 2}],
        )

    def test_error_recovery_falls_back_to_network_issue_when_connectivity_is_still_poor(self) -> None:
        self.coordinator.set_detection_state(DEVICE_STATE_ERROR)
        self.coordinator.set_connectivity_quality(CONNECTIVITY_QUALITY_POOR)
        self.coordinator.set_detection_state(DEVICE_STATE_RUNNING)

        self.assertEqual(
            self.sent_payloads,
            [
                {"status": DEVICE_STATE_ERROR, "message": "불량 감지됨", "lineId": 1, "processId": 2},
                {"status": DEVICE_STATE_NETWORK_ISSUE, "message": "통신 상태 확인 필요", "lineId": 1, "processId": 2},
            ],
        )

    def test_network_issue_recovery_emits_running_when_detection_is_normal(self) -> None:
        self.coordinator.set_connectivity_quality(CONNECTIVITY_QUALITY_POOR)
        self.coordinator.set_connectivity_quality(CONNECTIVITY_QUALITY_GOOD)

        self.assertEqual(
            self.sent_payloads,
            [
                {"status": DEVICE_STATE_NETWORK_ISSUE, "message": "통신 상태 확인 필요", "lineId": 1, "processId": 2},
                {"status": DEVICE_STATE_RUNNING, "message": "heartbeat", "lineId": 1, "processId": 2},
            ],
        )

    def test_error_has_priority_over_network_issue(self) -> None:
        self.coordinator.set_connectivity_quality(CONNECTIVITY_QUALITY_POOR)
        self.coordinator.set_detection_state(DEVICE_STATE_ERROR)

        self.assertEqual(self.coordinator.current_payload()["status"], DEVICE_STATE_ERROR)
        self.assertEqual(
            self.sent_payloads,
            [
                {"status": DEVICE_STATE_NETWORK_ISSUE, "message": "통신 상태 확인 필요", "lineId": 1, "processId": 2},
                {"status": DEVICE_STATE_ERROR, "message": "불량 감지됨", "lineId": 1, "processId": 2},
            ],
        )


if __name__ == "__main__":
    unittest.main()
