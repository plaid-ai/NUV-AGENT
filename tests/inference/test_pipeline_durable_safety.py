from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


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
    repository.GLibUnix = types.SimpleNamespace()
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
from nuvion_app.inference.command_runtime import build_fleet_command_runtime
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
    def _trusted_runtime(self, root: Path, registry=None):
        public_key = (
            Ed25519PrivateKey.generate()
            .public_key()
            .public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        )
        keyring = root / "test-keyring.json"
        keyring.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "trustDomain": "macos-dev",
                    "keys": {"test-only": base64.b64encode(public_key).decode("ascii")},
                }
            )
        )
        keyring.chmod(0o600)
        return build_fleet_command_runtime(
            base_url="https://api.example.test",
            access_token_provider=lambda: "test",
            ack_sender=lambda *_args: True,
            device_id="test-device",
            space_id=3,
            keyring_path=keyring,
            inbox_path=root / "inbox.sqlite3",
            platform_identity=types.SimpleNamespace(
                identity_status="DEV",
                platform_profile="macos_dev",
                capabilities=frozenset(),
            ),
            reconciler_registry=registry,
        )

    def test_runtime_authority_capability_updates_flat_and_nested_heartbeats(
        self,
    ) -> None:
        capability = "fleet.auth.platform_admin.v1"
        with tempfile.TemporaryDirectory() as temporary:
            runtime = self._trusted_runtime(Path(temporary))
            coordinator = pipeline.DeviceStateCoordinator(
                send_message=lambda _payload: True,
                line_id=None,
                process_id=None,
                telemetry={
                    "capabilities": [capability],
                    "runtimeTelemetry": {"capabilities": [capability]},
                },
                runtime_telemetry_provider=lambda: (
                    pipeline.build_dynamic_runtime_telemetry({capability})
                ),
            )
            with mock.patch.object(pipeline, "fleet_command_runtime", runtime):
                payload = coordinator.current_payload()
                self.assertIn(capability, payload["capabilities"])
                self.assertIn(capability, payload["runtimeTelemetry"]["capabilities"])
            with mock.patch.object(pipeline, "fleet_command_runtime", None):
                payload = coordinator.current_payload()
                self.assertNotIn(capability, payload["capabilities"])
                self.assertNotIn(
                    capability, payload["runtimeTelemetry"]["capabilities"]
                )

    def test_unready_runtime_and_static_or_effect_claims_cannot_grant_authority(
        self,
    ) -> None:
        capabilities = {
            "fleet.auth.platform_admin.v1",
            "command.config.model.siglip.v1",
        }
        for attempted in (False, True):
            for runtime in (
                None,
                types.SimpleNamespace(authorization_capabilities=capabilities),
            ):
                with (
                    self.subTest(attempted=attempted, runtime=runtime),
                    mock.patch.object(pipeline, "fleet_command_runtime", runtime),
                    mock.patch.object(
                        pipeline, "fleet_command_runtime_init_attempted", attempted
                    ),
                    mock.patch.object(
                        pipeline,
                        "fleet_effect_registry",
                        types.SimpleNamespace(capabilities=capabilities),
                    ),
                    mock.patch.object(
                        pipeline,
                        "build_command_observation_runtime_health",
                        return_value={},
                    ),
                ):
                    telemetry = pipeline.build_dynamic_runtime_telemetry(capabilities)
                    self.assertTrue(capabilities.isdisjoint(telemetry["capabilities"]))

    def test_failed_authority_runtime_property_withdraws_capability(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = self._trusted_runtime(Path(temporary))
            with (
                mock.patch.object(pipeline, "fleet_command_runtime", runtime),
                mock.patch.object(
                    type(runtime),
                    "authorization_capabilities",
                    new_callable=mock.PropertyMock,
                    side_effect=RuntimeError("unavailable"),
                ),
            ):
                telemetry = pipeline.build_dynamic_runtime_telemetry(
                    {"fleet.auth.platform_admin.v1"}
                )
                self.assertNotIn(
                    "fleet.auth.platform_admin.v1", telemetry["capabilities"]
                )

    def test_siglip_model_capability_requires_live_matching_runtime_without_hashing(
        self,
    ) -> None:
        capability = "command.config.model.siglip.v1"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "model"
            model_dir.mkdir()
            detector = types.SimpleNamespace(
                enabled=True, ready=True, loaded_model_source=lambda: str(model_dir)
            )
            app = types.SimpleNamespace(
                pipeline=object(),
                user_data=types.SimpleNamespace(
                    backend="siglip", running=True, zero_shot=detector
                ),
            )
            adapter = pipeline.PipelineSettingsRuntimeAdapter(
                app=app,
                encoder=object(),
                model_pointer="siglip/test",
                model_dir=model_dir,
            )
            registry = pipeline.ReconcilerRegistry()
            registry.register(
                pipeline.SettingsReconciler(store=object(), runtime=adapter)
            )
            runtime = self._trusted_runtime(root, registry)
            with (
                mock.patch.object(pipeline, "fleet_command_runtime", runtime),
                mock.patch.object(pipeline, "fleet_effect_registry", registry),
                mock.patch.object(pipeline, "g_app", app),
                mock.patch.object(
                    pipeline,
                    "verify_model_artifact_identity",
                    side_effect=AssertionError("heartbeat must not hash weights"),
                ) as verify,
            ):
                self.assertIn(
                    capability,
                    pipeline.build_dynamic_runtime_telemetry()["capabilities"],
                )
                verify.assert_not_called()
                for backend in ("none", "triton", "visualad", "visualad_htp"):
                    with (
                        self.subTest(backend=backend),
                        mock.patch.object(app.user_data, "backend", backend),
                    ):
                        self.assertEqual(
                            pipeline.build_model_config_capabilities(), frozenset()
                        )
                for flag in ("enabled", "ready"):
                    with (
                        self.subTest(flag=flag),
                        mock.patch.object(detector, flag, False),
                    ):
                        self.assertEqual(
                            pipeline.build_model_config_capabilities(), frozenset()
                        )
                old_model = root / "old-model"
                old_model.mkdir()
                for source in (None, str(old_model), "remote/huggingface-model"):
                    with (
                        self.subTest(source=source),
                        mock.patch.object(
                            detector,
                            "loaded_model_source",
                            lambda source=source: source,
                        ),
                    ):
                        self.assertEqual(
                            pipeline.build_model_config_capabilities(), frozenset()
                        )
                for field in ("can_verify_model", "verify_model"):
                    with (
                        self.subTest(field=field),
                        mock.patch.object(adapter, field, None),
                    ):
                        self.assertEqual(
                            pipeline.build_model_config_capabilities(), frozenset()
                        )
                with mock.patch.object(app.user_data, "running", False):
                    self.assertEqual(
                        pipeline.build_model_config_capabilities(), frozenset()
                    )
                with mock.patch.object(pipeline, "g_app", object()):
                    self.assertEqual(
                        pipeline.build_model_config_capabilities(), frozenset()
                    )
                with mock.patch.object(pipeline, "fleet_command_runtime", None):
                    self.assertNotIn(
                        capability,
                        pipeline.build_dynamic_runtime_telemetry({capability})[
                            "capabilities"
                        ],
                    )
                registry.unregister("CONFIG_APPLY")
                self.assertEqual(
                    pipeline.build_model_config_capabilities(), frozenset()
                )

    def test_visualad_model_capability_and_identity_require_actual_live_adapter(self):
        capability = "command.config.model.visualad_htp.v1"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            model_dir = root / "model"
            model_dir.mkdir()
            proof = {
                "pointer": "visualad/iq9075-htp-demo",
                "digest": "sha256:" + "c" * 64,
            }
            detector = object.__new__(pipeline.FleetVisualADHTPAnomalyDetector)
            detector.selection = types.SimpleNamespace(
                directory=model_dir, pointer=proof["pointer"]
            )
            detector.loaded_model_proof = mock.Mock(return_value=proof)
            detector.verify_model = mock.Mock(return_value=proof)
            detector.ready = True
            app = types.SimpleNamespace(
                pipeline=object(),
                user_data=types.SimpleNamespace(
                    backend="visualad_htp",
                    running=True,
                    zero_shot=detector,
                    inference_failed=False,
                    last_inference_at=pipeline.time.monotonic(),
                ),
            )
            adapter = pipeline.PipelineSettingsRuntimeAdapter(
                app=app,
                encoder=object(),
                model_pointer=proof["pointer"],
                model_dir=model_dir,
            )
            registry = pipeline.ReconcilerRegistry()
            registry.register(
                pipeline.SettingsReconciler(store=object(), runtime=adapter)
            )
            runtime = self._trusted_runtime(root, registry)
            with (
                mock.patch.object(pipeline, "fleet_command_runtime", runtime),
                mock.patch.object(pipeline, "fleet_effect_registry", registry),
                mock.patch.object(pipeline, "g_app", app),
            ):
                telemetry = pipeline.build_dynamic_runtime_telemetry()
                self.assertIn(capability, telemetry["capabilities"])
                self.assertNotIn(
                    "command.config.model.siglip.v1", telemetry["capabilities"]
                )
                self.assertEqual(telemetry["modelObservedPointer"], proof["pointer"])
                self.assertEqual(telemetry["modelDigest"], proof["digest"])
                self.assertEqual(adapter.verify_model(proof), proof)
                for field, value in (("inference_failed", True), ("running", False)):
                    with mock.patch.object(app.user_data, field, value):
                        telemetry = pipeline.build_dynamic_runtime_telemetry(
                            {capability}
                        )
                        self.assertNotIn(capability, telemetry["capabilities"])
                        self.assertEqual(telemetry["modelDigest"], "unknown")
                detector.loaded_model_proof.return_value = None
                telemetry = pipeline.build_dynamic_runtime_telemetry({capability})
                self.assertNotIn(capability, telemetry["capabilities"])
                self.assertEqual(telemetry["modelObservedPointer"], "unknown")
                self.assertEqual(telemetry["modelDigest"], "unknown")
                detector.loaded_model_proof.return_value = proof
                with mock.patch.object(adapter, "model_dir", root):
                    self.assertNotIn(
                        capability, pipeline.build_model_config_capabilities()
                    )
                with mock.patch.object(pipeline, "fleet_command_runtime", None):
                    self.assertNotIn(
                        capability,
                        pipeline.build_dynamic_runtime_telemetry({capability})[
                            "capabilities"
                        ],
                    )
                registry.unregister("CONFIG_APPLY")
                self.assertNotIn(capability, pipeline.build_model_config_capabilities())

    def _run_visualad_frame(
        self,
        result=None,
        failure=None,
        stop_during_inference=False,
        backend="visualad",
        frame_input=None,
        safety_stop_during_inference=False,
    ):
        state = object.__new__(pipeline.NuvionEventState)
        state.running = True
        state.backend = backend
        state.inference_failed = False
        state.demo_mode = False
        state._set_anomaly_overlay = mock.Mock()
        state.send_status = mock.Mock()
        state.zero_shot_queue = mock.Mock()

        frames = iter((frame_input if frame_input is not None else object(),))
        stopped = False

        def get_frame(**_kwargs):
            frame = next(frames, None)
            if frame is None:
                state.running = False
            return frame

        def infer(_frame):
            nonlocal stopped
            stopped = safety_stop_during_inference
            if stop_during_inference:
                state.running = False
            if failure:
                raise failure
            return result

        def safety_stopped():
            if stopped:
                # End this finite worker test after the publication safety fence.
                state.running = False
            return stopped

        state.zero_shot_queue.get.side_effect = get_frame
        state.zero_shot = types.SimpleNamespace(
            enabled=True,
            threshold=0.0,
            is_anomaly=mock.Mock(side_effect=infer),
        )
        coordinator = _Coordinator()
        with (
            mock.patch.object(
                pipeline, "get_device_state_coordinator", return_value=coordinator
            ),
            mock.patch.object(
                pipeline.critical_event_safety_gate,
                "is_stopped",
                side_effect=safety_stopped,
            ),
        ):
            state._zsad_worker()
        return state, coordinator

    def test_continuous_htp_keeps_only_latest_frame_without_sample_delay(self):
        state = object.__new__(pipeline.NuvionEventState)
        state.backend = "visualad_htp"
        state.zero_shot_last_sample = 100.0
        state.zero_shot_queue = pipeline.queue.Queue(maxsize=1)
        state.face_tracking_enabled = False
        frames = [object() for _ in range(30)]
        with (
            mock.patch.object(pipeline, "ZERO_SHOT_SAMPLE_SEC", 0.0),
            mock.patch.object(pipeline.time, "time", return_value=100.0),
            mock.patch.object(pipeline.time, "monotonic", return_value=500.0),
        ):
            for frame in frames:
                state.maybe_enqueue_frame(frame)
        self.assertEqual(state.zero_shot_queue.qsize(), 1)
        newest = state.zero_shot_queue.get_nowait()
        self.assertIs(newest.pixels, frames[-1])
        self.assertEqual(newest.arrived_at_monotonic, 500.0)

    def test_continuous_htp_worker_unwraps_frame_and_reports_arrival_age(self):
        pixels = object()
        with mock.patch.object(pipeline.time, "monotonic", return_value=100.0):
            state, _ = self._run_visualad_frame(
                (
                    False,
                    {
                        "label": "normal",
                        "score": -0.2,
                        "inference_seconds": 0.61,
                    },
                ),
                backend="visualad_htp",
                frame_input=pipeline.TimedInferenceFrame(pixels, 99.975),
            )
        state.zero_shot.is_anomaly.assert_called_once_with(pixels)
        self.assertEqual(state.last_inference_frame_arrived_at, 99.975)
        self.assertIn("queueWaitSeconds=0.025", state.send_status.call_args.args[2])
        self.assertIn("frameToResultSeconds=0.025", state.send_status.call_args.args[2])

    def test_sampled_backends_keep_existing_sampling_contract(self):
        for backend in ("visualad_htp", "visualad", "siglip", "triton"):
            state = object.__new__(pipeline.NuvionEventState)
            state.backend = backend
            state.zero_shot_last_sample = 100.0
            state.zero_shot_queue = pipeline.queue.Queue(maxsize=1)
            state.face_tracking_enabled = False
            with mock.patch.object(pipeline, "ZERO_SHOT_SAMPLE_SEC", 2.0):
                with mock.patch.object(pipeline.time, "time", return_value=101.0):
                    state.maybe_enqueue_frame(object())
                    self.assertTrue(state.zero_shot_queue.empty())
                pixels = object()
                with mock.patch.object(pipeline.time, "time", return_value=102.0):
                    state.maybe_enqueue_frame(pixels)
                self.assertIs(state.zero_shot_queue.get_nowait(), pixels)

    def test_continuous_result_cannot_publish_after_safety_stop(self):
        for result, failure in (
            ((True, {"label": "defect", "score": 0.8}), None),
            ((False, None), None),
            (None, RuntimeError("late hardware failure")),
        ):
            state, coordinator = self._run_visualad_frame(
                result,
                failure,
                backend="visualad_htp",
                frame_input=pipeline.TimedInferenceFrame(object(), 1.0),
                safety_stop_during_inference=True,
            )
            state.send_status.assert_not_called()
            state._set_anomaly_overlay.assert_not_called()
            self.assertEqual(coordinator.runtime_statuses, [])

    def test_visualad_failure_never_generates_normal_or_anomaly(self):
        for result, failure in (
            ((False, None), None),
            (None, RuntimeError("load failed")),
        ):
            with self.subTest(failure=failure):
                state, coordinator = self._run_visualad_frame(result, failure)
                state.send_status.assert_not_called()
                self.assertTrue(state.inference_failed)
                self.assertEqual(
                    coordinator.runtime_statuses, [pipeline.RUNTIME_STATUS_ERROR]
                )

    def test_visualad_actual_result_uses_existing_durable_event_path(self):
        state, _ = self._run_visualad_frame(
            (
                True,
                {
                    "label": "defect",
                    "score": 0.8,
                    "model_sha256": "f" * 64,
                },
            )
        )
        args = state.send_status.call_args.args
        self.assertEqual(args[0], "DEFECT")
        self.assertIn("VisualAD", args[2])
        self.assertIn("model=" + "f" * 64, args[2])
        self.assertIn("rawAnomalyScore=0.8000", args[2])
        self.assertEqual(state.last_inference_score, 0.8)

    def test_slow_inference_cannot_emit_after_shutdown(self):
        for result, failure in (
            ((True, {"label": "defect", "score": 0.8}), None),
            ((False, None), None),
            (None, RuntimeError("late failure")),
        ):
            with self.subTest(result=result, failure=failure):
                state, coordinator = self._run_visualad_frame(
                    result, failure, stop_during_inference=True
                )
                state.send_status.assert_not_called()
                state._set_anomaly_overlay.assert_not_called()
                self.assertEqual(coordinator.runtime_statuses, [])

    def test_htp_result_keeps_provider_and_latency_in_durable_event_message(self):
        state, _ = self._run_visualad_frame(
            (
                True,
                {
                    "label": "defect",
                    "score": 0.3,
                    "model_sha256": "f" * 64,
                    "inference_seconds": 0.61,
                    "graph_sha256": "a" * 64,
                },
            ),
            backend="visualad_htp",
        )
        args = state.send_status.call_args.args
        self.assertEqual(args[0], "DEFECT")
        self.assertIn("provider=QNN/HTP", args[2])
        self.assertIn("graph=" + "a" * 64, args[2])
        self.assertIn("inferenceSeconds=0.610", args[2])

    def test_htp_unavailable_is_not_a_normal_inspection(self):
        state, coordinator = self._run_visualad_frame(
            (False, None), backend="visualad_htp"
        )
        state.send_status.assert_not_called()
        self.assertTrue(state.inference_failed)
        self.assertEqual(coordinator.runtime_statuses, [pipeline.RUNTIME_STATUS_ERROR])

    def test_experimental_htp_result_is_not_presented_as_validated(self):
        with mock.patch.dict(
            pipeline.os.environ,
            {
                "NUVION_VISUALAD_EXPERIMENTAL": "true",
                "NUVION_VISUALAD_VALIDATION_STATUS": "PARITY_FAILED",
                "NUVION_VISUALAD_THRESHOLD_STATUS": "UNCALIBRATED",
            },
        ):
            state, _ = self._run_visualad_frame(
                (
                    False,
                    {
                        "label": "normal",
                        "score": -0.2,
                        "inference_seconds": 0.6,
                    },
                ),
                backend="visualad_htp",
            )
        message = state.send_status.call_args.args[2]
        self.assertIn("experimental=true validation=PARITY_FAILED", message)
        self.assertIn("thresholdStatus=UNCALIBRATED", message)
        self.assertIn("processingSeconds=", message)
        state._set_anomaly_overlay.assert_called_once_with(
            "EXP / UNCALIBRATED NORMAL -0.20"
        )

    def test_htp_compiles_then_discards_startup_frame_before_fresh_inference(self):
        state = object.__new__(pipeline.NuvionEventState)
        state.running = True
        state.backend = "visualad_htp"
        state.send_status = mock.Mock()
        stale, fresh = object(), object()
        steps = []

        def prepare():
            steps.append("compile")
            return True

        def discard():
            steps.append("discard_startup_frame")
            return stale

        def get_frame(**_kwargs):
            steps.append("capture_fresh_frame")
            return fresh

        def infer(frame):
            self.assertIs(frame, fresh)
            steps.append("infer_fresh_frame")
            state.running = False
            return False, None

        state.zero_shot_queue = types.SimpleNamespace(get=get_frame, get_nowait=discard)
        state.zero_shot = types.SimpleNamespace(
            enabled=True, prepare=prepare, is_anomaly=infer
        )
        with mock.patch.object(
            pipeline.critical_event_safety_gate, "is_stopped", return_value=False
        ):
            state._zsad_worker()
        self.assertEqual(
            steps,
            [
                "compile",
                "discard_startup_frame",
                "capture_fresh_frame",
                "infer_fresh_frame",
            ],
        )
        state.send_status.assert_not_called()

    def test_stream_runtime_evidence_reads_playing_state_and_frame_without_webrtc(
        self,
    ) -> None:
        playing = object()

        class _Pipeline:
            def __init__(self, state: object) -> None:
                self.state = state

            def get_state(self, timeout: int):
                self.timeout = timeout
                return object(), self.state, object()

        app = object.__new__(pipeline.GStreamerInferenceApp)
        app.pipeline = _Pipeline(playing)
        app.user_data = types.SimpleNamespace(
            running=True,
            last_frame_monotonic=123.0,
        )

        with mock.patch.object(
            pipeline,
            "Gst",
            types.SimpleNamespace(State=types.SimpleNamespace(PLAYING=playing)),
        ):
            evidence = app._stream_runtime_evidence()

        self.assertTrue(evidence.pipeline_running)
        self.assertEqual(evidence.last_frame_monotonic, 123.0)
        self.assertEqual(app.pipeline.timeout, 0)

    def test_update_commit_readiness_uses_live_pipeline_stomp_rtp_and_outboxes(
        self,
    ) -> None:
        class _Controller:
            @staticmethod
            def runtime_health_snapshot() -> dict[str, object]:
                return {
                    "hasPipeline": True,
                    "sessionId": "canary-session",
                    "generation": 4,
                    "connectionState": "connected",
                    "iceConnectionState": "completed",
                    "connectedSince": 100.0,
                    "iceConnectedSince": 100.0,
                    "outboundProgressSamples": 3,
                    "lastOutboundProgressAt": 128.0,
                }

        app = types.SimpleNamespace(
            pipeline=object(),
            user_data=types.SimpleNamespace(
                running=True,
                last_frame_monotonic=129.0,
            ),
            webrtc_uplink=_Controller(),
        )
        event_health = {
            "capacityState": "HEALTHY",
            "blockedRows": 0,
            "unsavedCriticalEvents": 0,
            "safetyStop": False,
            "protocolStop": False,
        }
        command_health = {
            "capacityState": "HEALTHY",
            "dlqBlockedRows": 0,
            "retentionPressure": False,
        }
        with (
            mock.patch.object(pipeline, "g_app", app),
            mock.patch.object(
                pipeline,
                "update_commit_signaling_ready_since",
                100.0,
            ),
            mock.patch.object(
                pipeline,
                "update_commit_stomp_last_send_at",
                125.0,
            ),
            mock.patch.object(pipeline, "agent_uplink_blocked", False),
            mock.patch.object(
                pipeline,
                "build_event_outbox_runtime_health",
                return_value=event_health,
            ),
            mock.patch.object(
                pipeline,
                "build_command_observation_runtime_health",
                return_value=command_health,
            ),
            mock.patch.object(pipeline.time, "monotonic", return_value=130.0),
        ):
            readiness = pipeline.build_update_commit_readiness()

        self.assertTrue(readiness["ready"])
        self.assertEqual(readiness["webrtcSessionId"], "canary-session")

    def test_stomp_send_evidence_belongs_to_current_connection(self) -> None:
        with mock.patch.object(
            pipeline.time,
            "monotonic",
            side_effect=(100.0, 105.0, 110.0),
        ):
            pipeline._set_update_commit_signaling_ready(True)
            pipeline._mark_update_commit_stomp_send()
            self.assertEqual(pipeline.update_commit_signaling_ready_since, 100.0)
            self.assertEqual(pipeline.update_commit_stomp_last_send_at, 105.0)
            pipeline._set_update_commit_signaling_ready(False)

        self.assertIsNone(pipeline.update_commit_signaling_ready_since)
        self.assertIsNone(pipeline.update_commit_stomp_last_send_at)

    def test_signaling_reset_purges_only_volatile_webrtc_transport_state(self) -> None:
        async def scenario() -> None:
            token = pipeline.WebRTCSignalingToken(7, "session-old")
            pending: asyncio.Queue[pipeline._OutboundMessage] = asyncio.Queue()
            pending.put_nowait(
                pipeline._OutboundMessage(
                    pipeline.WEBRTC_UPLINK_OFFER_DEST,
                    {"sessionId": "session-old"},
                    signaling_token=token,
                )
            )
            durable = pipeline._OutboundMessage(
                "/app/device/state",
                {"state": "RUNNING"},
                event_id="durable-event-1",
            )
            pending.put_nowait(durable)
            retry = asyncio.create_task(asyncio.sleep(60))
            controller = types.SimpleNamespace(
                reset_count=0,
                on_signaling_reset=lambda: setattr(
                    controller, "reset_count", controller.reset_count + 1
                ),
            )
            with (
                mock.patch.object(
                    pipeline, "g_app", types.SimpleNamespace(webrtc_uplink=controller)
                ),
                mock.patch.object(pipeline, "outbound_queue", pending),
                mock.patch.object(pipeline, "last_sent_payloads", {}),
                mock.patch.object(
                    pipeline,
                    "webrtc_retry_tasks",
                    {
                        pipeline._agent_retry_key(
                            pipeline.WEBRTC_UPLINK_OFFER_DEST,
                            token,
                        ): retry
                    },
                ),
            ):
                pipeline._remember_last_payload(
                    pipeline.WEBRTC_UPLINK_OFFER_DEST,
                    {"sessionId": "session-old"},
                    token,
                )
                pipeline._remember_last_payload(
                    "/app/device/state",
                    {"state": "RUNNING"},
                )

                pipeline._reset_webrtc_signaling_transport()
                await asyncio.sleep(0)

                self.assertEqual(controller.reset_count, 1)
                self.assertTrue(retry.cancelled())
                self.assertNotIn(
                    pipeline.WEBRTC_UPLINK_OFFER_DEST,
                    pipeline.last_sent_payloads,
                )
                self.assertIn("/app/device/state", pipeline.last_sent_payloads)
                self.assertIs(pending.get_nowait(), durable)
                self.assertTrue(pending.empty())

        asyncio.run(scenario())

    def test_unscoped_webrtc_signaling_cannot_bypass_generation_validation(
        self,
    ) -> None:
        self.assertFalse(
            pipeline.enqueue_stomp_message(
                pipeline.WEBRTC_UPLINK_OFFER_DEST,
                {"sessionId": "unscoped"},
            )
        )

    def test_correlated_terminal_offer_rejection_disposes_exact_session(self) -> None:
        token = pipeline.WebRTCSignalingToken(7, "session-current")
        controller = mock.Mock()
        controller.reject_signaling.return_value = True
        cached = {
            pipeline.WEBRTC_UPLINK_OFFER_DEST: pipeline._CachedPayload(
                {"sessionId": "session-current"},
                token,
            )
        }
        body = json.dumps(
            {
                "path": pipeline.WEBRTC_UPLINK_OFFER_DEST,
                "sessionId": "session-current",
                "status": 400,
                "retryable": False,
                "code": "WEBRTC_OFFER_REJECTED",
            }
        )

        with (
            mock.patch.object(
                pipeline,
                "g_app",
                types.SimpleNamespace(webrtc_uplink=controller),
            ),
            mock.patch.object(pipeline, "last_sent_payloads", cached),
            mock.patch.object(pipeline, "agent_retry_attempts", {}),
            mock.patch.object(pipeline, "webrtc_retry_tasks", {}),
        ):
            asyncio.run(pipeline.handle_agent_error(body))

        controller.reject_signaling.assert_called_once_with(
            token,
            reason="non-retryable server rejection: WEBRTC_OFFER_REJECTED status=400",
        )
        self.assertEqual(cached, {})

    def test_uncorrelated_or_stale_offer_rejection_waits_for_exact_watchdog(
        self,
    ) -> None:
        token = pipeline.WebRTCSignalingToken(9, "session-current")
        controller = mock.Mock()
        cached = {
            pipeline.WEBRTC_UPLINK_OFFER_DEST: pipeline._CachedPayload(
                {"sessionId": "session-current"},
                token,
            )
        }
        base = {
            "path": pipeline.WEBRTC_UPLINK_OFFER_DEST,
            "status": 400,
            "retryable": False,
            "code": "WEBRTC_OFFER_REJECTED",
        }

        with (
            mock.patch.object(
                pipeline,
                "g_app",
                types.SimpleNamespace(webrtc_uplink=controller),
            ),
            mock.patch.object(pipeline, "last_sent_payloads", cached),
            mock.patch.object(pipeline, "agent_retry_attempts", {}),
            mock.patch.object(pipeline, "webrtc_retry_tasks", {}),
        ):
            asyncio.run(pipeline.handle_agent_error(json.dumps(base)))
            asyncio.run(
                pipeline.handle_agent_error(
                    json.dumps({**base, "sessionId": "session-stale"})
                )
            )

        controller.reject_signaling.assert_not_called()
        self.assertIn(pipeline.WEBRTC_UPLINK_OFFER_DEST, cached)

    def test_stale_retryable_webrtc_errors_cannot_exhaust_new_session_budget(
        self,
    ) -> None:
        async def scenario() -> None:
            current = pipeline.WebRTCSignalingToken(10, "session-new")
            controller = mock.Mock()
            controller.is_signaling_token_current.side_effect = lambda token: (
                token == current
            )
            cached = {
                pipeline.WEBRTC_UPLINK_OFFER_DEST: pipeline._CachedPayload(
                    {"sessionId": "session-new", "sdp": "v=0\r\n"},
                    current,
                )
            }
            retry_started = asyncio.Event()
            release_retry = asyncio.Event()

            async def pending_retry(*_args: object, **_kwargs: object) -> None:
                retry_started.set()
                await release_retry.wait()

            error = {
                "path": pipeline.WEBRTC_UPLINK_OFFER_DEST,
                "status": 503,
                "retryable": True,
                "code": "WEBRTC_OFFER_TEMPORARY_FAILURE",
            }
            attempts: dict[pipeline._AgentRetryKey, int] = {}
            tasks: dict[
                pipeline._AgentRetryKey,
                asyncio.Task[None],
            ] = {}
            with (
                mock.patch.object(
                    pipeline,
                    "g_app",
                    types.SimpleNamespace(webrtc_uplink=controller),
                ),
                mock.patch.object(pipeline, "last_sent_payloads", cached),
                mock.patch.object(pipeline, "agent_retry_attempts", attempts),
                mock.patch.object(pipeline, "webrtc_retry_tasks", tasks),
                mock.patch.object(
                    pipeline,
                    "_enqueue_retry_after_delay",
                    side_effect=pending_retry,
                ),
            ):
                for _ in range(pipeline.AGENT_ERROR_MAX_RETRIES + 1):
                    await pipeline.handle_agent_error(
                        json.dumps({**error, "sessionId": "session-old"})
                    )

                self.assertEqual(attempts, {})
                self.assertEqual(tasks, {})
                controller.reject_signaling.assert_not_called()

                await pipeline.handle_agent_error(
                    json.dumps({**error, "sessionId": "session-new"})
                )
                await asyncio.wait_for(retry_started.wait(), timeout=1)

                current_key = pipeline._agent_retry_key(
                    pipeline.WEBRTC_UPLINK_OFFER_DEST,
                    current,
                )
                self.assertEqual(attempts, {current_key: 1})
                self.assertEqual(tuple(tasks), (current_key,))
                first_retry = tasks[current_key]
                for _ in range(pipeline.AGENT_ERROR_MAX_RETRIES + 1):
                    await pipeline.handle_agent_error(
                        json.dumps({**error, "sessionId": "session-new"})
                    )
                self.assertEqual(attempts, {current_key: 1})
                self.assertIs(tasks[current_key], first_retry)
                controller.reject_signaling.assert_not_called()
                release_retry.set()
                await asyncio.gather(*tasks.values())

        asyncio.run(scenario())

    def test_uncached_webrtc_auth_rejection_blocks_transport_and_tears_down(
        self,
    ) -> None:
        controller = mock.Mock()
        attempts = {
            pipeline._agent_retry_key(
                pipeline.WEBRTC_UPLINK_ICE_CANDIDATE_DEST,
                pipeline.WebRTCSignalingToken(4, "session-old"),
            ): 2
        }
        with (
            mock.patch.object(
                pipeline,
                "g_app",
                types.SimpleNamespace(webrtc_uplink=controller),
            ),
            mock.patch.object(pipeline, "last_sent_payloads", {}),
            mock.patch.object(pipeline, "agent_retry_attempts", attempts),
            mock.patch.object(pipeline, "webrtc_retry_tasks", {}),
            mock.patch.object(pipeline, "agent_uplink_blocked", False),
            mock.patch.object(pipeline, "agent_uplink_block_reason", ""),
        ):
            asyncio.run(
                pipeline.handle_agent_error(
                    json.dumps(
                        {
                            "path": pipeline.WEBRTC_UPLINK_ICE_CANDIDATE_DEST,
                            "status": 403,
                            "retryable": False,
                            "code": "FORBIDDEN",
                            "message": "token rejected",
                        }
                    )
                )
            )

            self.assertTrue(pipeline.agent_uplink_blocked)
            self.assertEqual(
                pipeline.agent_uplink_block_reason, "FORBIDDEN token rejected"
            )
            self.assertEqual(attempts, {})
            controller.on_signaling_reset.assert_called_once_with()

    def test_exact_uncached_candidate_rejection_disposes_current_generation(
        self,
    ) -> None:
        token = pipeline.WebRTCSignalingToken(12, "session-current")
        controller = mock.Mock()
        controller.signaling_token_for_session.return_value = token
        controller.is_signaling_token_current.side_effect = lambda value: value == token
        controller.reject_signaling.return_value = True
        body = json.dumps(
            {
                "path": pipeline.WEBRTC_UPLINK_ICE_CANDIDATE_DEST,
                "sessionId": "session-current",
                "status": 429,
                "retryable": True,
                "code": "WEBRTC_SIGNALING_CAPACITY",
            }
        )

        with (
            mock.patch.object(
                pipeline,
                "g_app",
                types.SimpleNamespace(webrtc_uplink=controller),
            ),
            mock.patch.object(pipeline, "last_sent_payloads", {}),
            mock.patch.object(pipeline, "agent_retry_attempts", {}),
            mock.patch.object(pipeline, "webrtc_retry_tasks", {}),
        ):
            asyncio.run(pipeline.handle_agent_error(body))

        controller.signaling_token_for_session.assert_called_once_with(
            "session-current",
            terminal=False,
        )
        controller.reject_signaling.assert_called_once_with(
            token,
            reason="signaling frame rejected: WEBRTC_SIGNALING_CAPACITY status=429",
        )

    def test_stale_uncached_candidate_rejection_cannot_dispose_current_session(
        self,
    ) -> None:
        controller = mock.Mock()
        controller.signaling_token_for_session.return_value = None
        with (
            mock.patch.object(
                pipeline,
                "g_app",
                types.SimpleNamespace(webrtc_uplink=controller),
            ),
            mock.patch.object(pipeline, "last_sent_payloads", {}),
            mock.patch.object(pipeline, "agent_retry_attempts", {}),
            mock.patch.object(pipeline, "webrtc_retry_tasks", {}),
        ):
            asyncio.run(
                pipeline.handle_agent_error(
                    json.dumps(
                        {
                            "path": pipeline.WEBRTC_UPLINK_ICE_CANDIDATE_DEST,
                            "sessionId": "session-stale",
                            "status": 400,
                            "retryable": False,
                            "code": "INVALID_ICE",
                        }
                    )
                )
            )

        controller.reject_signaling.assert_not_called()

    def test_outbound_sender_revalidates_token_without_dropping_durable_event(
        self,
    ) -> None:
        async def scenario() -> None:
            stale = pipeline.WebRTCSignalingToken(3, "session-old")
            pending: asyncio.Queue[pipeline._OutboundMessage] = asyncio.Queue()
            pending.put_nowait(
                pipeline._OutboundMessage(
                    pipeline.WEBRTC_UPLINK_ICE_CANDIDATE_DEST,
                    {"sessionId": "session-old"},
                    signaling_token=stale,
                )
            )
            pending.put_nowait(
                pipeline._OutboundMessage(
                    "/app/device/state",
                    {"state": "RUNNING"},
                    event_id="durable-event-1",
                )
            )
            sent = asyncio.Event()

            class _WebSocket:
                def __init__(self) -> None:
                    self.frames: list[str] = []

                async def send(self, frame: str) -> None:
                    self.frames.append(frame)
                    sent.set()

            class _Outbox:
                @staticmethod
                def is_pending(_event_id: str) -> bool:
                    return True

            class _Delivery:
                outbox = _Outbox()

                def __init__(self) -> None:
                    self.marked: list[str] = []

                def mark_sent(self, event_id: str) -> None:
                    self.marked.append(event_id)

                @staticmethod
                def release(_event_id: str) -> None:
                    return None

            delivery = _Delivery()
            controller = types.SimpleNamespace(
                is_signaling_token_current=lambda _token: False
            )
            websocket = _WebSocket()
            with (
                mock.patch.object(
                    pipeline, "g_app", types.SimpleNamespace(webrtc_uplink=controller)
                ),
                mock.patch.object(pipeline, "outbound_queue", pending),
                mock.patch.object(pipeline, "critical_event_delivery", delivery),
            ):
                sender = asyncio.create_task(pipeline.outbound_sender(websocket))
                await asyncio.wait_for(sent.wait(), timeout=1)
                sender.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await sender

            self.assertEqual(len(websocket.frames), 1)
            self.assertIn("/app/device/state", websocket.frames[0])
            self.assertNotIn("session-old", websocket.frames[0])
            self.assertEqual(delivery.marked, ["durable-event-1"])

        asyncio.run(scenario())

    def test_config_label_array_storage_round_trips_without_csv_loss(self) -> None:
        labels = ["scratch,edge", "한글 label"]
        encoded = config_env_updates({"labels": {"inspection": labels}})[
            "NUVION_ZERO_SHOT_LABELS_B64"
        ]

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
        self.assertIn(
            "tee name=webrtc_uplink_tee allow-not-linked=true",
            description,
        )
        self.assertNotIn("webrtcbin", description)
        self.assertEqual(description.count("x264enc"), 2)
        self.assertIn("x264enc name=video_encoder", description)
        self.assertIn(
            "name=video_encoder tune=zerolatency speed-preset=faster bitrate=1750",
            description,
        )
        self.assertIn("x264enc name=clip_encoder", description)
        self.assertEqual(description.count("level=(string)3.1"), 2)
        self.assertEqual(description.count("max-size-buffers=2"), 2)
        self.assertEqual(description.count("leaky=downstream"), 2)
        self.assertLess(
            description.index("name=video_encoder"),
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

    def test_dynamic_telemetry_merges_updater_rollback_and_functional_health(
        self,
    ) -> None:
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

    def test_visualad_health_requires_recent_successful_real_inference(self) -> None:
        self._assert_visualad_health("visualad")
        self._assert_visualad_health("visualad_htp")

    def _assert_visualad_health(self, backend) -> None:
        for ready, failed, sampled_at, expected in (
            (False, False, None, "FUNCTIONAL_UNHEALTHY"),
            (True, False, None, "FUNCTIONAL_UNHEALTHY"),
            (True, True, 95.0, "FUNCTIONAL_UNHEALTHY"),
            (True, False, 10.0, "FUNCTIONAL_UNHEALTHY"),
            (True, False, 95.0, "FUNCTIONAL_HEALTHY"),
        ):
            with self.subTest(
                backend=backend, ready=ready, failed=failed, sampled_at=sampled_at
            ):
                app = types.SimpleNamespace(
                    pipeline=object(),
                    user_data=types.SimpleNamespace(
                        running=True,
                        backend=backend,
                        inference_failed=failed,
                        last_inference_at=sampled_at,
                        zero_shot=types.SimpleNamespace(ready=ready, threshold=0.0),
                    ),
                )
                with (
                    mock.patch.object(pipeline, "g_app", app),
                    mock.patch.object(pipeline.time, "monotonic", return_value=100.0),
                ):
                    telemetry = pipeline.build_dynamic_runtime_telemetry()
                self.assertEqual(telemetry["functionalHealth"], expected)
                self.assertEqual(telemetry["inference"]["ready"], ready)
                self.assertEqual(telemetry["inference"]["backend"], backend)
                self.assertEqual(
                    telemetry["inference"]["scoreKind"], "raw_top_1_percent_mean"
                )
                self.assertEqual(
                    telemetry["inference"]["thresholdStatus"], "UNCALIBRATED"
                )
                self.assertEqual(
                    telemetry["inference"]["sampleAgeSeconds"],
                    telemetry["inference"]["lastResultAgeSeconds"],
                )

    def test_stale_updater_cache_hides_capability_but_keeps_terminal_evidence(
        self,
    ) -> None:
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
                1.0,
            ),
            mock.patch.object(pipeline, "UPDATER_TELEMETRY_TTL_SEC", 1.0),
            mock.patch.object(pipeline.time, "monotonic", return_value=200.0),
        ):
            stale = pipeline.get_cached_updater_runtime_telemetry()

        self.assertEqual(stale["updaterVersion"], "unknown")
        self.assertFalse(stale["agentUpdate"]["capabilityAvailable"])
        self.assertEqual(stale["updatePhase"], "ROLLED_BACK")
        self.assertEqual(stale["updateEvidence"], trusted["updateEvidence"])

    def test_supervisor_restart_is_enabled_only_inside_systemd_linux_service(
        self,
    ) -> None:
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

    def test_sigterm_quits_main_loop_and_runs_graceful_shutdown(self) -> None:
        class _SignalSource:
            def __init__(self) -> None:
                self.callback = None
                self.context = None
                self.destroy_calls = 0

            def set_callback(self, callback) -> None:
                self.callback = callback

            def attach(self, context) -> int:
                self.context = context
                return 7

            def destroy(self) -> None:
                self.destroy_calls += 1

        class _Loop:
            def __init__(self) -> None:
                self.running = True
                self.quit_calls = 0

            def run(self) -> None:
                self.running = True
                self.source_callback_result = source.callback()

            def is_running(self) -> bool:
                return self.running

            def quit(self) -> None:
                self.quit_calls += 1
                self.running = False

            def get_context(self):
                return context

        success = object()
        failure = object()
        context = object()
        source = _SignalSource()
        loop = _Loop()
        app = object.__new__(pipeline.GStreamerInferenceApp)
        app.pipeline = types.SimpleNamespace(set_state=lambda _state: success)
        app.depthai_bridge = None
        app.loop = loop
        app.shutdown = mock.Mock()

        with (
            mock.patch.object(pipeline, "LOCAL_DISPLAY", False),
            mock.patch.object(
                pipeline.GLibUnix,
                "signal_source_new",
                return_value=source,
                create=True,
            ),
            mock.patch.object(
                # GI enum attributes are lazily materialized native objects;
                # patch the dependency reference, not GI's enum type cache.
                pipeline,
                "Gst",
                types.SimpleNamespace(
                    State=types.SimpleNamespace(PLAYING=object()),
                    StateChangeReturn=types.SimpleNamespace(FAILURE=failure),
                ),
            ),
            mock.patch.object(
                pipeline.threading,
                "Thread",
                return_value=types.SimpleNamespace(start=lambda: None),
            ),
            mock.patch.object(pipeline, "get_device_state_coordinator"),
        ):
            app.run()

        self.assertEqual(loop.quit_calls, 1)
        self.assertTrue(loop.source_callback_result)
        self.assertIs(source.context, context)
        self.assertEqual(source.destroy_calls, 1)
        app.shutdown.assert_called_once_with()

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
