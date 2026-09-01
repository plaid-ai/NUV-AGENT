from __future__ import annotations

import asyncio
import json
import sys
import types
import unittest
from unittest import mock


def _install_gi_stub_when_native_bindings_are_unavailable() -> None:
    """Keep the pure safety-boundary tests independent from native GStreamer."""
    try:
        import gi

        return
    except ModuleNotFoundError:
        pass

    gi = types.ModuleType("gi")
    gi.require_version = lambda *_args, **_kwargs: None
    repository = types.ModuleType("gi.repository")
    repository.GLib = types.SimpleNamespace()
    repository.Gst = types.SimpleNamespace(
        Pipeline=object,
        Element=object,
        Promise=object,
    )
    repository.GstSdp = types.SimpleNamespace()
    repository.GstWebRTC = types.SimpleNamespace()
    gi.repository = repository
    sys.modules["gi"] = gi
    sys.modules["gi.repository"] = repository


_install_gi_stub_when_native_bindings_are_unavailable()

from nuvion_app.inference import pipeline
from nuvion_app.inference.critical_event_safety import (
    CriticalEventBackpressureError,
    CriticalEventSafetyGate,
)
from nuvion_app.inference.durable_events import EVENT_TYPE_ANOMALY


class _Coordinator:
    def __init__(self) -> None:
        self.runtime_statuses: list[str] = []
        self.inspection_statuses: list[str] = []

    def set_runtime_status(self, status: str) -> None:
        self.runtime_statuses.append(status)

    def set_inspection_status(self, status: str) -> None:
        self.inspection_statuses.append(status)


class PipelineDurableSafetyTest(unittest.TestCase):
    def test_unavailable_outbox_retains_anomaly_in_gate_and_enters_stop(self) -> None:
        gate = CriticalEventSafetyGate(max_attempts=1, retry_delay_seconds=0)
        coordinator = _Coordinator()
        with (
            mock.patch.object(pipeline, "critical_event_safety_gate", gate),
            mock.patch.object(
                pipeline,
                "initialize_durable_event_outbox",
                return_value=None,
            ),
            mock.patch.object(
                pipeline,
                "get_device_state_coordinator",
                return_value=coordinator,
            ),
            self.assertRaises(CriticalEventBackpressureError),
        ):
            pipeline.persist_critical_event(
                EVENT_TYPE_ANOMALY,
                "/app/device/anomaly",
                {"anomalyStatus": "DEFECT", "message": "observation"},
                "334aab50-3cf6-49c4-8362-f3cb26a6994e",
                "2026-09-01T00:00:00Z",
            )

        self.assertTrue(gate.is_stopped())
        self.assertIsNotNone(gate.pending_event())
        self.assertFalse(gate.health_overlay()["durableSafetyRetained"])
        self.assertEqual(coordinator.runtime_statuses, [pipeline.RUNTIME_STATUS_ERROR])

    def test_send_status_never_returns_before_critical_safety_boundary(self) -> None:
        state = object.__new__(pipeline.NuvionEventState)
        state.last_sent_status = None
        state.last_status = None
        state.last_sent_at = 0.0
        state.demo_mode = False
        coordinator = _Coordinator()
        failure = CriticalEventBackpressureError(
            "334aab50-3cf6-49c4-8362-f3cb26a6994e",
            "outbox unavailable",
        )

        with (
            mock.patch.object(
                pipeline,
                "initialize_durable_event_outbox",
                return_value=None,
            ),
            mock.patch.object(
                pipeline,
                "get_device_state_coordinator",
                return_value=coordinator,
            ),
            mock.patch.object(
                pipeline,
                "persist_critical_event",
                side_effect=failure,
            ) as persist,
            self.assertRaises(CriticalEventBackpressureError),
        ):
            state.send_status(
                "DEFECT",
                "scratch",
                "detected",
                "WARNING",
                snapshot_object="anomalies/1/device/snapshot.jpg",
                clip_object="anomalies/1/device/clip.mp4",
                clip_status="UPLOADING",
            )

        persist.assert_called_once()

    def test_uncorrelated_terminal_409_stops_replay_instead_of_poison_loop(
        self,
    ) -> None:
        gate = CriticalEventSafetyGate()
        coordinator = _Coordinator()
        body = json.dumps(
            {
                "path": "/app/device/production",
                "status": 409,
                "retryable": False,
                "terminal": True,
                "failureClass": "PERMANENT_NO_EVENT_IDENTITY",
                "eventIdentityAvailable": False,
                "code": "COMMON_409_002",
            }
        )

        with (
            mock.patch.object(pipeline, "critical_event_safety_gate", gate),
            mock.patch.object(
                pipeline,
                "get_device_state_coordinator",
                return_value=coordinator,
            ),
        ):
            asyncio.run(pipeline.handle_agent_error(body))

        self.assertTrue(gate.is_stopped())
        self.assertFalse(gate.replay_allowed())
        self.assertEqual(coordinator.runtime_statuses, [pipeline.RUNTIME_STATUS_ERROR])

    def test_correlated_terminal_409_is_quarantined_without_protocol_stop(self) -> None:
        gate = CriticalEventSafetyGate()
        delivery = mock.Mock()
        delivery.reject_event.return_value = False
        event_id = "334aab50-3cf6-49c4-8362-f3cb26a6994e"
        body = json.dumps(
            {
                "path": "/app/device/anomaly",
                "eventId": event_id,
                "status": 409,
                "retryable": False,
                "terminal": True,
                "failureClass": "PERMANENT",
                "eventIdentityAvailable": True,
                "code": "EVENT_ID_COLLISION",
            }
        )

        with (
            mock.patch.object(pipeline, "critical_event_safety_gate", gate),
            mock.patch.object(pipeline, "critical_event_delivery", delivery),
        ):
            asyncio.run(pipeline.handle_agent_error(body))

        delivery.reject_event.assert_called_once_with(
            event_id,
            EVENT_TYPE_ANOMALY,
            reason="permanent rejection",
            rejection_code="EVENT_ID_COLLISION",
            source="agent.error",
        )
        self.assertFalse(gate.is_stopped())
        self.assertTrue(gate.replay_allowed())


if __name__ == "__main__":
    unittest.main()
