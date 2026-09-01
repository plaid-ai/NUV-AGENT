from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
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
from nuvion_app.inference.settings_reconciler import (
    UnsupportedSettingsEffect,
    config_env_updates,
)


class _Coordinator:
    def __init__(self) -> None:
        self.runtime_statuses: list[str] = []
        self.inspection_statuses: list[str] = []

    def set_runtime_status(self, status: str) -> None:
        self.runtime_statuses.append(status)

    def set_inspection_status(self, status: str) -> None:
        self.inspection_statuses.append(status)


class PipelineDurableSafetyTest(unittest.TestCase):
    def test_config_label_array_storage_round_trips_without_csv_loss(self) -> None:
        labels = ["scratch,edge", "한글 label"]
        encoded = config_env_updates(
            {"labels": {"inspection": labels}}
        )["NUVION_ZERO_SHOT_LABELS_B64"]

        self.assertEqual(
            pipeline.parse_label_array(encoded, "legacy,fallback"),
            labels,
        )

    def test_clip_and_webrtc_use_independent_named_encoders(self) -> None:
        description = pipeline.build_uplink_pipeline(
            rtp_ssrc=1234,
            clip_enabled=True,
            clip_segment_sec=2.0,
            clip_max_segments=20,
            clip_segments_dir="/var/lib/nuvion/segments",
            video_bitrate_kbps=1750,
        )

        self.assertIn("tee name=stream_split", description)
        self.assertEqual(description.count("x264enc"), 2)
        self.assertIn("x264enc name=video_encoder", description)
        self.assertIn("name=video_encoder tune=zerolatency speed-preset=faster bitrate=1750", description)
        self.assertIn("x264enc name=clip_encoder", description)
        self.assertLess(
            description.index("stream_split. ! queue ! x264enc")
            if "stream_split. ! queue ! x264enc" in description
            else description.index("name=video_encoder"),
            description.index("name=clip_encoder"),
        )

    def test_config_model_evidence_requires_exact_siglip_loaded_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary) / "desired-model"
            metadata = model_dir / "metadata"
            metadata.mkdir(parents=True)
            manifest = metadata / "manifest.json"
            manifest.write_bytes(b"loaded model manifest")
            digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
            (metadata / "server_presign_response.json").write_text(
                json.dumps({"pointer": "anomalyclip/prod-v2"}),
                encoding="utf-8",
            )
            (metadata / "downloaded_from_server.json").write_text(
                json.dumps(
                    [
                        {
                            "key": "manifest",
                            "dst": str(manifest),
                            "sha256": digest,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            class Detector:
                enabled = True
                ready = True

                def __init__(self, source: Path) -> None:
                    self.source = source

                def loaded_model_source(self) -> str:
                    return str(self.source)

            user_data = types.SimpleNamespace(
                backend="siglip",
                zero_shot=Detector(model_dir),
            )
            app = types.SimpleNamespace(user_data=user_data)
            adapter = pipeline.PipelineSettingsRuntimeAdapter(
                app=app,
                encoder=object(),
                model_pointer="anomalyclip/prod-v2",
                model_dir=model_dir,
            )
            desired = {
                "pointer": "anomalyclip/prod-v2",
                "digest": "sha256:" + digest,
            }

            self.assertEqual(adapter.verify_model(desired), desired)

            old_model = Path(temporary) / "old-model"
            old_model.mkdir()
            user_data.zero_shot = Detector(old_model)
            with self.assertRaisesRegex(RuntimeError, "old or different"):
                adapter.verify_model(desired)

            user_data.backend = "triton"
            with self.assertRaises(UnsupportedSettingsEffect):
                adapter.verify_model(desired)

    def test_dynamic_telemetry_merges_updater_rollback_and_functional_health(self) -> None:
        app = types.SimpleNamespace(
            pipeline=object(),
            user_data=types.SimpleNamespace(running=True),
        )
        pipeline.register_updater_public_state_provider(
            lambda: {
                "updatePhase": "ROLLED_BACK",
                "updateEvidence": {
                    "rolledBackToVersion": "0.1.119",
                    "reason": "health gate failed",
                },
            }
        )
        self.addCleanup(pipeline.register_updater_public_state_provider, None)

        with mock.patch.object(pipeline, "g_app", app):
            telemetry = pipeline.build_dynamic_runtime_telemetry()

        self.assertEqual(telemetry["functionalHealth"], "FUNCTIONAL_HEALTHY")
        self.assertEqual(telemetry["updatePhase"], "ROLLED_BACK")
        self.assertIn("updateEvidence", telemetry)
        self.assertIn("commandObservationOutbox", telemetry)

    def test_stale_updater_cache_hides_capability_but_keeps_terminal_evidence(self) -> None:
        trusted = {
            "agentUpdate": {
                "capabilityAvailable": True,
                "authenticatedHelper": True,
                "reason": "READY",
            },
            "updaterVersion": "0.1.0",
            "updatePhase": "ROLLED_BACK",
            "updateEvidence": {"commandId": "update-1", "phase": "ROLLED_BACK"},
        }
        with (
            mock.patch.object(pipeline, "updater_telemetry_cache", trusted),
            mock.patch.object(
                pipeline,
                "updater_telemetry_cache_updated_at",
                pipeline.time.monotonic() - 100.0,
            ),
            mock.patch.object(pipeline, "UPDATER_TELEMETRY_TTL_SEC", 1.0),
        ):
            stale = pipeline.get_cached_updater_runtime_telemetry()

        self.assertEqual(stale["updaterVersion"], "unknown")
        self.assertFalse(stale["agentUpdate"]["capabilityAvailable"])
        self.assertEqual(stale["updatePhase"], "ROLLED_BACK")
        self.assertEqual(stale["updateEvidence"], trusted["updateEvidence"])

    def test_supervisor_restart_is_enabled_only_inside_systemd_linux_service(self) -> None:
        with mock.patch.object(pipeline.sys, "platform", "linux"):
            self.assertTrue(
                pipeline.systemd_restart_enabled(
                    {
                        "NUVION_SUPERVISOR_RESTART_ENABLED": "true",
                        "INVOCATION_ID": "systemd-invocation",
                    }
                )
            )
            self.assertFalse(
                pipeline.systemd_restart_enabled(
                    {"NUVION_SUPERVISOR_RESTART_ENABLED": "true"}
                )
            )
        with mock.patch.object(pipeline.sys, "platform", "darwin"):
            self.assertFalse(
                pipeline.systemd_restart_enabled(
                    {
                        "NUVION_SUPERVISOR_RESTART_ENABLED": "true",
                        "INVOCATION_ID": "systemd-invocation",
                    }
                )
            )

    def test_supervisor_restart_requests_graceful_main_loop_shutdown(self) -> None:
        app = object.__new__(pipeline.GStreamerInferenceApp)
        app._supervisor_restart_lock = threading.Lock()
        app._supervisor_restart_requested = False
        shutdown_calls: list[str] = []
        app.shutdown = lambda: shutdown_calls.append("shutdown")
        callbacks = []

        with mock.patch.object(
            pipeline.GLib,
            "idle_add",
            side_effect=lambda callback: callbacks.append(callback) or 1,
            create=True,
        ):
            self.assertTrue(app.request_supervisor_restart())
            self.assertTrue(app.request_supervisor_restart())

        self.assertEqual(len(callbacks), 1)
        callbacks[0]()
        self.assertEqual(shutdown_calls, ["shutdown"])

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
