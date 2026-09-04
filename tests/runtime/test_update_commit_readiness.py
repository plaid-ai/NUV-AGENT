from __future__ import annotations

import unittest

from nuvion_app.runtime.update_commit_readiness import (
    evaluate_update_commit_readiness,
)


class UpdateCommitReadinessTest(unittest.TestCase):
    def _evaluate(self, **overrides: object) -> dict[str, object]:
        arguments: dict[str, object] = {
            "now_monotonic": 130.0,
            "signaling_ready_since": 100.0,
            "stomp_last_send_at": 125.0,
            "min_stable_seconds": 15.0,
            "max_evidence_age_seconds": 20.0,
            "pipeline_running": True,
            "pipeline_last_frame_at": 129.0,
            "webrtc_health": {
                "hasPipeline": True,
                "sessionId": "session-1",
                "generation": 2,
                "connectionState": "connected",
                "iceConnectionState": "completed",
                "connectedSince": 100.0,
                "iceConnectedSince": 100.0,
                "outboundProgressSamples": 2,
                "lastOutboundProgressAt": 128.0,
            },
            "stomp_blocked": False,
            "event_outbox_health": {
                "capacityState": "HEALTHY",
                "blockedRows": 0,
                "unsavedCriticalEvents": 0,
                "safetyStop": False,
                "protocolStop": False,
            },
            "command_outbox_health": {
                "capacityState": "HEALTHY",
                "dlqBlockedRows": 0,
                "retentionPressure": False,
            },
        }
        arguments.update(overrides)
        return evaluate_update_commit_readiness(**arguments)  # type: ignore[arg-type]

    def test_ready_requires_stable_stomp_pipeline_webrtc_and_outboxes(self) -> None:
        self.assertEqual(
            self._evaluate(),
            {
                "ready": True,
                "reason": "READY",
                "stableSeconds": 30,
                "webrtcSessionId": "session-1",
                "webrtcGeneration": 2,
            },
        )

    def test_each_missing_runtime_proof_fails_closed(self) -> None:
        cases = (
            ({"pipeline_running": False}, "PIPELINE_NOT_RUNNING"),
            ({"pipeline_last_frame_at": None}, "CAMERA_FRAME_UNAVAILABLE"),
            ({"pipeline_last_frame_at": 100.0}, "CAMERA_FRAME_STALE"),
            (
                {"webrtc_health": {"hasPipeline": False}},
                "WEBRTC_PIPELINE_NOT_ATTACHED",
            ),
            ({"signaling_ready_since": None}, "STOMP_NOT_CONNECTED"),
            ({"stomp_blocked": True}, "STOMP_UPLINK_BLOCKED"),
            (
                {"now_monotonic": 110.0, "pipeline_last_frame_at": 109.0},
                "STOMP_SOAK_PENDING",
            ),
            ({"stomp_last_send_at": None}, "STOMP_SEND_UNPROVEN"),
            ({"stomp_last_send_at": 100.0}, "STOMP_SEND_STALE"),
            (
                {"webrtc_health": {"hasPipeline": True}},
                "WEBRTC_SESSION_UNAVAILABLE",
            ),
            (
                {
                    "webrtc_health": {
                        **self._evaluate_arguments_webrtc(),
                        "connectionState": "connecting",
                    }
                },
                "WEBRTC_NOT_CONNECTED",
            ),
            (
                {
                    "webrtc_health": {
                        **self._evaluate_arguments_webrtc(),
                        "iceConnectionState": "checking",
                    }
                },
                "WEBRTC_ICE_NOT_CONNECTED",
            ),
            (
                {
                    "webrtc_health": {
                        **self._evaluate_arguments_webrtc(),
                        "connectedSince": 120.0,
                    }
                },
                "WEBRTC_SOAK_PENDING",
            ),
            (
                {
                    "webrtc_health": {
                        **self._evaluate_arguments_webrtc(),
                        "outboundProgressSamples": 1,
                    }
                },
                "WEBRTC_RTP_PROGRESS_UNPROVEN",
            ),
            (
                {
                    "webrtc_health": {
                        **self._evaluate_arguments_webrtc(),
                        "lastOutboundProgressAt": 100.0,
                    }
                },
                "WEBRTC_RTP_PROGRESS_STALE",
            ),
            (
                {"event_outbox_health": {"capacityState": "PRESSURE"}},
                "EVENT_OUTBOX_UNHEALTHY",
            ),
            (
                {
                    "event_outbox_health": {
                        "capacityState": "HEALTHY",
                        "blockedRows": 1,
                    }
                },
                "EVENT_OUTBOX_BLOCKED",
            ),
            (
                {
                    "event_outbox_health": {
                        "capacityState": "HEALTHY",
                        "unsavedCriticalEvents": 1,
                    }
                },
                "CRITICAL_EVENT_NOT_DURABLE",
            ),
            (
                {
                    "event_outbox_health": {
                        "capacityState": "HEALTHY",
                        "unsavedCriticalEvents": 0,
                        "safetyStop": True,
                    }
                },
                "CRITICAL_EVENT_SAFETY_STOP",
            ),
            (
                {
                    "event_outbox_health": {
                        "capacityState": "HEALTHY",
                        "unsavedCriticalEvents": 0,
                        "protocolStop": True,
                    }
                },
                "EVENT_PROTOCOL_STOP",
            ),
            (
                {"command_outbox_health": {"capacityState": "BACKPRESSURE"}},
                "COMMAND_OUTBOX_UNHEALTHY",
            ),
            (
                {
                    "command_outbox_health": {
                        "capacityState": "HEALTHY",
                        "dlqBlockedRows": 1,
                    }
                },
                "COMMAND_OUTBOX_DLQ_BLOCKED",
            ),
            (
                {
                    "command_outbox_health": {
                        "capacityState": "HEALTHY",
                        "retentionPressure": True,
                    }
                },
                "COMMAND_OUTBOX_RETENTION_PRESSURE",
            ),
        )
        for overrides, reason in cases:
            with self.subTest(reason=reason):
                self.assertEqual(self._evaluate(**overrides)["reason"], reason)
                self.assertFalse(self._evaluate(**overrides)["ready"])

    def test_invalid_soak_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one second"):
            self._evaluate(min_stable_seconds=0.0)
        with self.assertRaisesRegex(ValueError, "at least one second"):
            self._evaluate(max_evidence_age_seconds=0.0)

    @staticmethod
    def _evaluate_arguments_webrtc() -> dict[str, object]:
        return {
            "hasPipeline": True,
            "sessionId": "session-1",
            "generation": 2,
            "connectionState": "connected",
            "iceConnectionState": "completed",
            "connectedSince": 100.0,
            "iceConnectedSince": 100.0,
            "outboundProgressSamples": 2,
            "lastOutboundProgressAt": 128.0,
        }


if __name__ == "__main__":
    unittest.main()
