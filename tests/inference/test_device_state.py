from __future__ import annotations

import unittest

from nuvion_app.inference.device_state import (
    CONNECTIVITY_QUALITY_GOOD,
    CONNECTIVITY_QUALITY_POOR,
    DEVICE_STATE_ERROR,
    DEVICE_STATE_NETWORK_ISSUE,
    DEVICE_STATE_RUNNING,
    INSPECTION_STATUS_DEFECT,
    INSPECTION_STATUS_NORMAL,
    RUNTIME_STATUS_ERROR,
    RUNTIME_STATUS_RUNNING,
    DeviceStateCoordinator,
)


class DeviceStateCoordinatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sent_payloads: list[dict[str, object]] = []
        self.coordinator = DeviceStateCoordinator(
            send_message=self._capture,
            line_id=1,
            process_id=2,
            telemetry={
                "agentVersion": "0.1.113",
                "componentSha": "abc123",
                "configSchema": "11",
                "modelPointer": "anomalyclip/prod",
                "modelVersion": "v0001",
            },
        )

    def _capture(self, payload: dict[str, object]) -> bool:
        self.sent_payloads.append(payload)
        return True

    def test_heartbeat_uses_current_running_state(self) -> None:
        self.coordinator.emit_heartbeat()

        self.assertEqual(
            self.sent_payloads,
            [
                {
                    "status": DEVICE_STATE_RUNNING,
                    "message": "heartbeat",
                    "lineId": 1,
                    "processId": 2,
                    "runtimeStatus": RUNTIME_STATUS_RUNNING,
                    "inspectionStatus": INSPECTION_STATUS_NORMAL,
                    "connectivityStatus": CONNECTIVITY_QUALITY_GOOD,
                    "agentVersion": "0.1.113",
                    "componentSha": "abc123",
                    "configSchema": "11",
                    "modelPointer": "anomalyclip/prod",
                    "modelVersion": "v0001",
                    "functionalHealth": "FUNCTIONAL_HEALTHY",
                }
            ],
        )

    def test_detection_error_emits_error_state_immediately(self) -> None:
        self.coordinator.set_detection_state(DEVICE_STATE_ERROR)

        self.assertEqual(
            self.sent_payloads,
            [
                {
                    **self.coordinator.current_payload(),
                    "status": DEVICE_STATE_ERROR,
                    "message": "불량 감지됨",
                }
            ],
        )

    def test_connectivity_poor_emits_network_issue_state_immediately(self) -> None:
        self.coordinator.set_connectivity_quality(CONNECTIVITY_QUALITY_POOR)

        self.assertEqual(
            self.sent_payloads,
            [
                {
                    **self.coordinator.current_payload(),
                    "status": DEVICE_STATE_NETWORK_ISSUE,
                    "message": "통신 상태 확인 필요",
                }
            ],
        )

    def test_error_recovery_falls_back_to_network_issue_when_connectivity_is_still_poor(self) -> None:
        self.coordinator.set_detection_state(DEVICE_STATE_ERROR)
        self.coordinator.set_connectivity_quality(CONNECTIVITY_QUALITY_POOR)
        self.coordinator.set_detection_state(DEVICE_STATE_RUNNING)

        self.assertEqual(
            [payload["status"] for payload in self.sent_payloads],
            [DEVICE_STATE_ERROR, DEVICE_STATE_ERROR, DEVICE_STATE_NETWORK_ISSUE],
        )
        self.assertEqual(self.sent_payloads[1]["connectivityStatus"], CONNECTIVITY_QUALITY_POOR)
        self.assertEqual(self.sent_payloads[-1]["inspectionStatus"], INSPECTION_STATUS_NORMAL)
        self.assertEqual(self.sent_payloads[-1]["connectivityStatus"], CONNECTIVITY_QUALITY_POOR)

    def test_network_issue_recovery_emits_running_when_detection_is_normal(self) -> None:
        self.coordinator.set_connectivity_quality(CONNECTIVITY_QUALITY_POOR)
        self.coordinator.set_connectivity_quality(CONNECTIVITY_QUALITY_GOOD)

        self.assertEqual([payload["status"] for payload in self.sent_payloads], [DEVICE_STATE_NETWORK_ISSUE, DEVICE_STATE_RUNNING])

    def test_error_has_priority_over_network_issue(self) -> None:
        self.coordinator.set_connectivity_quality(CONNECTIVITY_QUALITY_POOR)
        self.coordinator.set_detection_state(DEVICE_STATE_ERROR)

        self.assertEqual(self.coordinator.current_payload()["status"], DEVICE_STATE_ERROR)
        self.assertEqual([payload["status"] for payload in self.sent_payloads], [DEVICE_STATE_NETWORK_ISSUE, DEVICE_STATE_ERROR])

    def test_runtime_inspection_and_connectivity_are_independent(self) -> None:
        self.coordinator.set_inspection_status(INSPECTION_STATUS_DEFECT)
        self.coordinator.set_connectivity_status(CONNECTIVITY_QUALITY_POOR)
        self.coordinator.set_runtime_status(RUNTIME_STATUS_ERROR)

        payload = self.coordinator.current_payload()

        self.assertEqual(payload["runtimeStatus"], RUNTIME_STATUS_ERROR)
        self.assertEqual(payload["inspectionStatus"], INSPECTION_STATUS_DEFECT)
        self.assertEqual(payload["connectivityStatus"], CONNECTIVITY_QUALITY_POOR)
        self.assertEqual(payload["status"], DEVICE_STATE_ERROR)
        self.assertEqual(payload["message"], "Agent runtime error")

    def test_inspection_defect_message_is_not_runtime_error(self) -> None:
        self.coordinator.set_inspection_status(INSPECTION_STATUS_DEFECT)

        payload = self.coordinator.current_payload()

        self.assertEqual(payload["status"], DEVICE_STATE_ERROR)
        self.assertEqual(payload["message"], "불량 감지됨")

    def test_running_heartbeat_does_not_clear_defect(self) -> None:
        self.coordinator.set_inspection_status(INSPECTION_STATUS_DEFECT)
        self.sent_payloads.clear()

        self.coordinator.emit_heartbeat()

        self.assertEqual(self.sent_payloads[0]["status"], DEVICE_STATE_ERROR)
        self.assertEqual(self.sent_payloads[0]["inspectionStatus"], INSPECTION_STATUS_DEFECT)

    def test_runtime_telemetry_provider_refreshes_outbox_health_each_heartbeat(self) -> None:
        health = {"pendingRows": 1, "capacityState": "HEALTHY"}
        coordinator = DeviceStateCoordinator(
            send_message=self._capture,
            line_id=1,
            process_id=2,
            telemetry={"runtimeTelemetry": {"agentVersion": "0.1.113"}},
            runtime_telemetry_provider=lambda: {
                "eventOutbox": dict(health),
                "functionalHealth": "FUNCTIONAL_HEALTHY",
                "capabilities": ["command.stream.policy"],
            },
        )

        coordinator.emit_heartbeat()
        health.update({"pendingRows": 2, "capacityState": "OPERATOR_STOP"})
        coordinator.emit_heartbeat()

        first = self.sent_payloads[0]["runtimeTelemetry"]
        second = self.sent_payloads[1]["runtimeTelemetry"]
        self.assertEqual(first["eventOutbox"]["pendingRows"], 1)
        self.assertEqual(second["eventOutbox"]["pendingRows"], 2)
        self.assertEqual(second["eventOutbox"]["capacityState"], "OPERATOR_STOP")
        self.assertEqual(second["agentVersion"], "0.1.113")
        self.assertEqual(
            self.sent_payloads[1]["functionalHealth"],
            "FUNCTIONAL_HEALTHY",
        )
        self.assertEqual(
            self.sent_payloads[1]["capabilities"],
            ["command.stream.policy"],
        )
        self.assertEqual(second["functionalHealth"], "FUNCTIONAL_HEALTHY")

        coordinator.set_runtime_status(RUNTIME_STATUS_ERROR)
        unhealthy = self.sent_payloads[-1]
        self.assertEqual(unhealthy["functionalHealth"], "FUNCTIONAL_UNHEALTHY")
        self.assertEqual(
            unhealthy["runtimeTelemetry"]["functionalHealth"],
            "FUNCTIONAL_UNHEALTHY",
        )

    def test_updater_version_is_promoted_only_with_authenticated_live_status(self) -> None:
        dynamic: dict[str, object] = {
            "updaterVersion": "99.0.0",
            "agentUpdate": {
                "capabilityAvailable": False,
                "authenticatedHelper": False,
                "reason": "UPDATER_UNAVAILABLE",
            },
        }
        coordinator = DeviceStateCoordinator(
            send_message=self._capture,
            line_id=1,
            process_id=2,
            telemetry={
                "updaterVersion": "7.7.7",
                "runtimeTelemetry": {"updaterVersion": "7.7.7"},
            },
            runtime_telemetry_provider=lambda: dict(dynamic),
        )

        unavailable = coordinator.current_payload()
        self.assertEqual(unavailable["updaterVersion"], "unknown")
        self.assertEqual(
            unavailable["runtimeTelemetry"]["updaterVersion"], "unknown"
        )

        dynamic["updaterVersion"] = "0.1.0"
        dynamic["agentUpdate"] = {
            "capabilityAvailable": True,
            "authenticatedHelper": True,
            "reason": "READY",
            "updaterVersion": "0.1.0",
        }
        live = coordinator.current_payload()
        self.assertEqual(live["updaterVersion"], "0.1.0")
        self.assertEqual(live["runtimeTelemetry"]["updaterVersion"], "0.1.0")


if __name__ == "__main__":
    unittest.main()
