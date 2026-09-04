from __future__ import annotations

import base64
import hashlib
import json
import threading
import time
import unittest
import uuid
from pathlib import Path

from nuvion_app.inference.fleet_command import VerifiedFleetCommand
from nuvion_app.inference.stream_policy import (
    AdaptiveBitrateController,
    GlibMainContextDispatcher,
    StreamPolicy,
    StreamPolicyReconciler,
    StreamRuntimeEvidence,
    X264EncoderAdapter,
)


def _adaptive_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "policyVersion": 7,
        "mode": "ADAPTIVE",
        "minBitrateKbps": 200,
        "maxBitrateKbps": 2000,
        "initialBitrateKbps": 1000,
        "congestionSamples": 2,
        "recoverySamples": 2,
        "cooldownSeconds": 1,
        "increaseStepKbps": 200,
        "decreaseFactor": 0.5,
    }
    payload.update(overrides)
    return payload


def _command(payload: dict[str, object], sequence: int = 1) -> VerifiedFleetCommand:
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return VerifiedFleetCommand(
        command_id=str(uuid.uuid4()),
        device_id="device-1",
        space_id=1,
        command_type="STREAM_POLICY",
        schema_version=1,
        issued_at="2026-09-01T00:00:00Z",
        expires_at="2026-09-01T01:00:00Z",
        sequence=sequence,
        payload_base64=base64.urlsafe_b64encode(encoded).decode().rstrip("="),
        payload_hash=hashlib.sha256(encoded).hexdigest(),
        payload=dict(payload),
        actor="operator@example.com",
        authorization_context="SPACE_ADMIN",
        key_id="test",
        required_capability="command.stream.policy",
        compact_jws=f"header.{sequence}.signature",
    )


class _FakeEncoder:
    name = "x264enc"

    def __init__(self, bitrate: int = 1000) -> None:
        self.bitrate = bitrate
        self.set_calls: list[int] = []

    def read_bitrate_kbps(self) -> int:
        return self.bitrate

    def set_bitrate_kbps(self, bitrate_kbps: int) -> int:
        self.bitrate = int(bitrate_kbps)
        self.set_calls.append(self.bitrate)
        return self.bitrate


class _FakeElement:
    def __init__(self, bitrate: int = 1000) -> None:
        self.bitrate = bitrate

    def get_property(self, name: str) -> int:
        if name != "bitrate":
            raise KeyError(name)
        return self.bitrate

    def set_property(self, name: str, value: int) -> None:
        if name != "bitrate":
            raise KeyError(name)
        self.bitrate = int(value)


def _healthy_runtime_evidence() -> StreamRuntimeEvidence:
    return StreamRuntimeEvidence(
        pipeline_running=True,
        last_frame_monotonic=time.monotonic(),
    )


class AdaptiveBitrateControllerTest(unittest.TestCase):
    def test_aimd_uses_hysteresis_cooldown_and_clamps(self) -> None:
        controller = AdaptiveBitrateController(
            StreamPolicy.from_payload(_adaptive_payload())
        )

        first = controller.observe(
            {
                "outboundPacketLossPct": 10.0,
                "outboundRttMs": 40,
                "outboundPacketsDelta": 10,
                "outboundBytesDelta": 1000,
            },
            now_ms=0,
        )
        second = controller.observe(
            {
                "outboundPacketLossPct": 10.0,
                "outboundRttMs": 40,
                "outboundPacketsDelta": 10,
                "outboundBytesDelta": 1000,
            },
            now_ms=1,
        )
        cooling = controller.observe(
            {
                "outboundPacketLossPct": 0.0,
                "outboundRttMs": 40,
                "outboundPacketsDelta": 10,
                "outboundBytesDelta": 1000,
            },
            now_ms=500,
        )
        decaying = controller.observe(
            {
                "outboundPacketLossPct": 0.0,
                "outboundRttMs": 40,
                "outboundPacketsDelta": 10,
                "outboundBytesDelta": 1000,
            },
            now_ms=1500,
        )
        stable = controller.observe(
            {
                "outboundPacketLossPct": 0.0,
                "outboundRttMs": 40,
                "outboundPacketsDelta": 10,
                "outboundBytesDelta": 1000,
            },
            now_ms=2500,
        )
        recovered = controller.observe(
            {
                "outboundPacketLossPct": 0.0,
                "outboundRttMs": 40,
                "outboundPacketsDelta": 10,
                "outboundBytesDelta": 1000,
            },
            now_ms=3500,
        )

        self.assertFalse(first.changed)
        self.assertEqual(
            first.reason,
            "awaiting_hysteresis:packet_loss_high",
        )
        self.assertTrue(second.changed)
        self.assertEqual(second.bitrate_kbps, 500)
        self.assertEqual(cooling.reason, "cooldown:packet_loss_high")
        self.assertEqual(decaying.reason, "packet_loss_high")
        self.assertEqual(decaying.bitrate_kbps, 250)
        self.assertFalse(stable.changed)
        self.assertTrue(recovered.changed)
        self.assertEqual(recovered.bitrate_kbps, 450)
        self.assertEqual(recovered.reason, "healthy_recovery")

    def test_link_bitrate_is_auxiliary_congestion_signal(self) -> None:
        controller = AdaptiveBitrateController(
            StreamPolicy.from_payload(
                _adaptive_payload(congestionSamples=1)
            )
        )

        decision = controller.observe(
            {
                "quality": "GOOD",
                "packetLossPct": 0.0,
                "rttMs": 20,
                "uplinkKbps": 800,
            },
            now_ms=0,
        )

        self.assertTrue(decision.changed)
        self.assertEqual(decision.bitrate_kbps, 500)
        self.assertEqual(decision.reason, "link_capacity_low")

    def test_unknown_sample_holds_without_drift(self) -> None:
        controller = AdaptiveBitrateController(
            StreamPolicy.from_payload(_adaptive_payload())
        )
        decision = controller.observe({}, now_ms=0)
        self.assertEqual(decision.state, "HOLD")
        self.assertEqual(decision.reason, "insufficient_signal")
        self.assertEqual(decision.bitrate_kbps, 1000)

    def test_webrtc_feedback_takes_priority_over_auxiliary_connectivity(self) -> None:
        controller = AdaptiveBitrateController(
            StreamPolicy.from_payload(_adaptive_payload(congestionSamples=1))
        )

        healthy_primary = controller.observe(
            {
                "outboundPacketLossPct": 0.0,
                "outboundRttMs": 30,
                "nackDelta": 0,
                "pliDelta": 0,
                "quality": "POOR",
                "packetLossPct": 50.0,
                "rttMs": 900,
                "uplinkKbps": 100,
            },
            now_ms=0,
        )
        nack = controller.observe(
            {
                "outboundPacketLossPct": 0.0,
                "outboundRttMs": 30,
                "nackDelta": 2,
                "pliDelta": 0,
            },
            now_ms=6000,
        )

        self.assertFalse(healthy_primary.changed)
        self.assertNotIn("connectivity_poor", healthy_primary.reason)
        self.assertTrue(nack.changed)
        self.assertEqual(nack.reason, "nack_increase")

    def test_fresh_primary_samples_prevent_parallel_auxiliary_source_thrashing(self) -> None:
        controller = AdaptiveBitrateController(
            StreamPolicy.from_payload(_adaptive_payload(congestionSamples=2))
        )

        first_primary = controller.observe(
            {
                "outboundPacketLossPct": 10.0,
                "outboundRttMs": 40,
            },
            now_ms=0,
        )
        ignored_auxiliary = controller.observe(
            {
                "quality": "POOR",
                "packetLossPct": 30.0,
                "rttMs": 500,
            },
            now_ms=1000,
        )
        second_primary = controller.observe(
            {
                "outboundPacketLossPct": 10.0,
                "outboundRttMs": 40,
            },
            now_ms=2000,
        )

        self.assertFalse(first_primary.changed)
        self.assertEqual(ignored_auxiliary.reason, "primary_signal_fresh")
        self.assertTrue(second_primary.changed)
        self.assertEqual(second_primary.bitrate_kbps, 500)

    def test_auxiliary_fallback_starts_once_after_primary_stale_window(self) -> None:
        controller = AdaptiveBitrateController(
            StreamPolicy.from_payload(_adaptive_payload(congestionSamples=2)),
            primary_stale_ms=5000,
        )
        controller.observe(
            {
                "outboundPacketLossPct": 0.0,
                "outboundRttMs": 30,
            },
            now_ms=0,
        )

        fresh = controller.observe(
            {"quality": "POOR", "packetLossPct": 20.0},
            now_ms=4000,
        )
        first_stale = controller.observe(
            {"quality": "POOR", "packetLossPct": 20.0},
            now_ms=6000,
        )
        second_stale = controller.observe(
            {"quality": "POOR", "packetLossPct": 20.0},
            now_ms=7000,
        )

        self.assertEqual(fresh.reason, "primary_signal_fresh")
        self.assertFalse(first_stale.changed)
        self.assertTrue(second_stale.changed)
        self.assertEqual(second_stale.bitrate_kbps, 500)

    def test_idle_primary_stats_cannot_accumulate_recovery_samples(self) -> None:
        controller = AdaptiveBitrateController(
            StreamPolicy.from_payload(
                _adaptive_payload(
                    congestionSamples=1,
                    recoverySamples=2,
                    cooldownSeconds=1,
                )
            )
        )
        degraded = controller.observe({"quality": "POOR"}, now_ms=0)

        first_idle = controller.observe(
            {
                "outboundPacketLossPct": 0.0,
                "outboundRttMs": 30,
                "outboundPacketsDelta": 0,
                "outboundBytesDelta": 0,
            },
            now_ms=2000,
        )
        second_idle = controller.observe(
            {
                "outboundPacketLossPct": 0.0,
                "outboundRttMs": 30,
                "outboundPacketsDelta": 0,
                "outboundBytesDelta": 0,
            },
            now_ms=3000,
        )
        first_progress = controller.observe(
            {
                "outboundPacketLossPct": 0.0,
                "outboundRttMs": 30,
                "outboundPacketsDelta": 10,
                "outboundBytesDelta": 1000,
            },
            now_ms=4000,
        )
        recovered = controller.observe(
            {
                "outboundPacketLossPct": 0.0,
                "outboundRttMs": 30,
                "outboundPacketsDelta": 10,
                "outboundBytesDelta": 1000,
            },
            now_ms=5000,
        )

        self.assertTrue(degraded.changed)
        self.assertEqual(degraded.bitrate_kbps, 500)
        self.assertEqual(first_idle.reason, "outbound_progress_idle")
        self.assertEqual(second_idle.reason, "outbound_progress_idle")
        self.assertFalse(first_progress.changed)
        self.assertTrue(recovered.changed)
        self.assertEqual(recovered.bitrate_kbps, 700)

    def test_auxiliary_good_without_outbound_proof_cannot_raise_bitrate(self) -> None:
        controller = AdaptiveBitrateController(
            StreamPolicy.from_payload(
                _adaptive_payload(
                    congestionSamples=1,
                    recoverySamples=2,
                    cooldownSeconds=1,
                )
            )
        )

        degraded = controller.observe({"quality": "POOR"}, now_ms=0)
        first_good = controller.observe({"quality": "GOOD"}, now_ms=2000)
        second_good = controller.observe({"quality": "GOOD"}, now_ms=3000)

        self.assertTrue(degraded.changed)
        self.assertEqual(degraded.bitrate_kbps, 500)
        self.assertEqual(first_good.reason, "outbound_progress_unproven")
        self.assertEqual(second_good.reason, "outbound_progress_unproven")
        self.assertEqual(second_good.bitrate_kbps, 500)


class EncoderAdapterTest(unittest.TestCase):
    def test_named_x264_adapter_mutates_and_reads_back_on_dispatcher(self) -> None:
        element = _FakeElement()
        dispatch_calls: list[str] = []

        def idle_add(callback):
            dispatch_calls.append("scheduled")
            callback()
            return 1

        adapter = X264EncoderAdapter(
            element,
            dispatch=GlibMainContextDispatcher(idle_add),
        )

        self.assertEqual(adapter.set_bitrate_kbps(1750), 1750)
        self.assertEqual(adapter.read_bitrate_kbps(), 1750)
        self.assertEqual(dispatch_calls, ["scheduled", "scheduled"])

    def test_timed_out_glib_callback_is_cancelled_before_late_mutation(self) -> None:
        element = _FakeElement()
        callbacks = []

        def idle_add(callback):
            callbacks.append(callback)
            return 1

        adapter = X264EncoderAdapter(
            element,
            dispatch=GlibMainContextDispatcher(
                idle_add,
                timeout_seconds=0.01,
            ),
        )

        with self.assertRaises(TimeoutError):
            adapter.set_bitrate_kbps(1750)
        callbacks[0]()

        self.assertEqual(element.bitrate, 1000)

    def test_glib_callback_rechecks_fence_at_actual_mutation_boundary(self) -> None:
        element = _FakeElement()
        callback_ready = threading.Event()
        callbacks = []
        authorized = {"value": True}

        def idle_add(callback):
            callbacks.append(callback)
            callback_ready.set()
            return 1

        adapter = X264EncoderAdapter(
            element,
            dispatch=GlibMainContextDispatcher(idle_add, timeout_seconds=1),
        )

        def fence() -> None:
            if not authorized["value"]:
                raise RuntimeError("stale fence")

        adapter.set_effect_fence(fence)
        errors: list[BaseException] = []

        def mutate() -> None:
            try:
                adapter.set_bitrate_kbps(1750)
            except BaseException as exc:  # noqa: BLE001 - asserted below.
                errors.append(exc)

        worker = threading.Thread(target=mutate)
        worker.start()
        self.assertTrue(callback_ready.wait(1))
        authorized["value"] = False
        callbacks[0]()
        worker.join(1)

        self.assertFalse(worker.is_alive())
        self.assertEqual(element.bitrate, 1000)
        self.assertRegex(str(errors[0]), "stale fence")


class StreamPolicyReconcilerTest(unittest.TestCase):
    def test_reported_state_matches_fleet_effect_v2_fixture(self) -> None:
        fixture = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "fixtures"
                / "fleet-effect-v2-stream-policy.json"
            ).read_text(encoding="utf-8")
        )
        payload = fixture["validPayloads"][2]
        encoder = _FakeEncoder()
        outcome = StreamPolicyReconciler(
            encoder,
            runtime_evidence=_healthy_runtime_evidence,
        ).reconcile(_command(payload))

        self.assertEqual(
            set(outcome.reported_state),
            set(fixture["adaptiveReportedStateKeys"]),
        )
        for key, expected in fixture["adaptiveDefaults"].items():
            self.assertEqual(outcome.reported_state[key], expected)
        self.assertEqual(outcome.reported_state["requestedBitrateKbps"], 1200)
        self.assertEqual(outcome.reported_state["encoder"], "x264enc")
        self.assertEqual(outcome.reported_state["health"], "STREAM_CONTINUOUS")
        for key, expected in payload.items():
            self.assertEqual(outcome.reported_state[key], expected)

    def test_fixed_and_adaptive_succeeded_state_preserve_signed_desired_fields(self) -> None:
        fixture = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "fixtures"
                / "fleet-effect-v2-stream-policy.json"
            ).read_text(encoding="utf-8")
        )
        cases = (
            (fixture["validPayloads"][0], fixture["fixedReportedStateKeys"]),
            (fixture["validPayloads"][1], fixture["adaptiveReportedStateKeys"]),
        )

        for payload, expected_keys in cases:
            with self.subTest(mode=payload["mode"]):
                outcome = StreamPolicyReconciler(
                    _FakeEncoder(),
                    runtime_evidence=_healthy_runtime_evidence,
                ).reconcile(_command(payload))
                self.assertEqual(outcome.status, "SUCCEEDED")
                self.assertEqual(set(outcome.reported_state), set(expected_keys))
                for key, expected in payload.items():
                    self.assertEqual(outcome.reported_state[key], expected)
                self.assertEqual(
                    outcome.reported_state["requestedBitrateKbps"],
                    payload.get(
                        "targetBitrateKbps",
                        payload.get("initialBitrateKbps"),
                    ),
                )

    def test_fixed_policy_reports_encoder_readback(self) -> None:
        encoder = _FakeEncoder()
        reconciler = StreamPolicyReconciler(
            encoder,
            runtime_evidence=_healthy_runtime_evidence,
        )
        command = _command(
            {
                "policyVersion": 3,
                "mode": "FIXED",
                "targetBitrateKbps": 1400,
            }
        )

        outcome = reconciler.reconcile(command)

        self.assertEqual(outcome.status, "SUCCEEDED")
        self.assertEqual(outcome.reported_state["mode"], "FIXED")
        self.assertEqual(outcome.reported_state["appliedBitrateKbps"], 1400)
        self.assertEqual(outcome.reported_state["targetBitrateKbps"], 1400)
        self.assertEqual(encoder.set_calls, [1400])

    def test_adaptive_policy_updates_reported_state_from_connectivity(self) -> None:
        ticks = iter([0.0, 1.0])
        encoder = _FakeEncoder()
        reconciler = StreamPolicyReconciler(
            encoder,
            clock_ms=lambda: next(ticks),
            runtime_evidence=_healthy_runtime_evidence,
        )
        command = _command(_adaptive_payload(congestionSamples=1))
        reconciler.reconcile(command)

        update = reconciler.observe_connectivity(
            {"quality": "POOR", "packetLossPct": 12.0, "rttMs": 300}
        )

        self.assertEqual(update.command_id, command.command_id)
        self.assertEqual(
            update.reported_state["lastAdjustmentReason"],
            "connectivity_poor,packet_loss_high,round_trip_time_high",
        )
        self.assertEqual(update.reported_state["appliedBitrateKbps"], 500)
        self.assertEqual(encoder.set_calls, [1000, 500])

    def test_adaptive_recovery_reason_reaches_reported_observation(self) -> None:
        ticks = iter([0.0, 2000.0])
        encoder = _FakeEncoder()
        reconciler = StreamPolicyReconciler(
            encoder,
            clock_ms=lambda: next(ticks),
            runtime_evidence=_healthy_runtime_evidence,
        )
        command = _command(
            _adaptive_payload(
                congestionSamples=1,
                recoverySamples=1,
                cooldownSeconds=1,
            )
        )
        reconciler.reconcile(command)

        poor = reconciler.observe_connectivity({"quality": "POOR"})
        reconciler.observation_committed(poor)
        recovered = reconciler.observe_stream_metrics(
            {
                "outboundPacketLossPct": 0.0,
                "outboundRttMs": 30,
                "outboundPacketsDelta": 10,
                "outboundBytesDelta": 1000,
            }
        )

        self.assertEqual(poor.reported_state["appliedBitrateKbps"], 500)
        self.assertEqual(recovered.reported_state["appliedBitrateKbps"], 700)
        self.assertEqual(
            recovered.reported_state["lastAdjustmentReason"],
            "healthy_recovery",
        )
        self.assertEqual(encoder.set_calls, [1000, 500, 700])

    def test_hold_reason_change_is_reported_without_bitrate_mutation(self) -> None:
        encoder = _FakeEncoder()
        reconciler = StreamPolicyReconciler(
            encoder,
            clock_ms=lambda: 0.0,
            runtime_evidence=_healthy_runtime_evidence,
        )
        command = _command(_adaptive_payload())
        reconciler.reconcile(command)

        update = reconciler.observe_connectivity({})

        self.assertIsNotNone(update)
        self.assertEqual(update.command_id, command.command_id)
        self.assertEqual(update.reported_state["appliedBitrateKbps"], 1000)
        self.assertEqual(
            update.reported_state["lastAdjustmentReason"],
            "insufficient_signal",
        )
        self.assertEqual(encoder.set_calls, [1000])

    def test_missing_runtime_evidence_fails_before_encoder_mutation(self) -> None:
        encoder = _FakeEncoder()

        outcome = StreamPolicyReconciler(encoder).reconcile(
            _command(_adaptive_payload())
        )

        self.assertEqual(outcome.status, "FAILED")
        self.assertEqual(outcome.code, "STREAM_HEALTH_EVIDENCE_UNAVAILABLE")
        self.assertEqual(outcome.reported_state, {"health": "NOT_APPLIED"})
        self.assertEqual(encoder.set_calls, [])

    def test_pipeline_and_recent_frame_prove_health_without_webrtc(self) -> None:
        evidence = {"last_frame": 99.0}
        reconciler = StreamPolicyReconciler(
            _FakeEncoder(),
            runtime_evidence=lambda: StreamRuntimeEvidence(
                pipeline_running=True,
                last_frame_monotonic=evidence["last_frame"],
            ),
            health_clock=lambda: 100.0,
            max_frame_age_seconds=5.0,
        )

        outcome = reconciler.reconcile(_command(_adaptive_payload()))

        self.assertEqual(outcome.status, "SUCCEEDED")
        self.assertEqual(outcome.reported_state["health"], "STREAM_CONTINUOUS")
        self.assertTrue(reconciler.ready())

    def test_stale_frame_fails_closed_and_resets_recovery_hysteresis(self) -> None:
        evidence = {"last_frame": 99.0}
        now = {"value": 100.0}
        encoder = _FakeEncoder()
        reconciler = StreamPolicyReconciler(
            encoder,
            clock_ms=lambda: now["value"] * 1000.0,
            runtime_evidence=lambda: StreamRuntimeEvidence(
                pipeline_running=True,
                last_frame_monotonic=evidence["last_frame"],
            ),
            health_clock=lambda: now["value"],
            max_frame_age_seconds=5.0,
        )
        reconciler.reconcile(
            _command(
                _adaptive_payload(
                    recoverySamples=2,
                    congestionSamples=1,
                    cooldownSeconds=1,
                )
            )
        )
        poor = reconciler.observe_connectivity({"quality": "POOR"})
        reconciler.observation_committed(poor)
        now["value"] = 102.0
        first = reconciler.observe_stream_metrics(
            {
                "outboundRttMs": 20,
                "outboundPacketsDelta": 5,
                "outboundBytesDelta": 500,
            }
        )
        reconciler.observation_committed(first)

        now["value"] = 110.0
        stale = reconciler.observe_stream_metrics(
            {
                "outboundRttMs": 20,
                "outboundPacketsDelta": 5,
                "outboundBytesDelta": 500,
            }
        )
        self.assertIsNotNone(stale)
        self.assertEqual(stale.reported_state["health"], "STREAM_FRAME_STALE")
        self.assertEqual(
            stale.reported_state["lastAdjustmentReason"], "STREAM_FRAME_STALE"
        )
        reconciler.observation_committed(stale)
        self.assertFalse(reconciler.ready())
        evidence["last_frame"] = 110.0
        resumed = reconciler.observe_stream_metrics(
            {
                "outboundRttMs": 20,
                "outboundPacketsDelta": 5,
                "outboundBytesDelta": 500,
            }
        )

        self.assertIsNotNone(resumed)
        self.assertEqual(resumed.reported_state["health"], "STREAM_CONTINUOUS")
        self.assertEqual(
            resumed.reported_state["lastAdjustmentReason"],
            "stream_health_recovered",
        )
        reported = reconciler.reported_state()
        self.assertEqual(reported["appliedBitrateKbps"], 500)
        self.assertNotEqual(
            reported["lastAdjustmentReason"],
            "healthy_recovery",
        )
        self.assertEqual(encoder.set_calls, [1000, 500])

    def test_disabled_policy_is_acknowledged_without_live_frame_evidence(self) -> None:
        encoder = _FakeEncoder(1300)
        reconciler = StreamPolicyReconciler(
            encoder,
            runtime_evidence=lambda: StreamRuntimeEvidence(
                pipeline_running=True,
                last_frame_monotonic=0.0,
            ),
            health_clock=lambda: 10.0,
        )
        disabled = _command({"policyVersion": 8, "mode": "DISABLED"})

        self.assertFalse(reconciler.ready())
        self.assertTrue(reconciler.admit_when_unready(disabled))
        outcome = reconciler.reconcile(disabled)

        self.assertEqual(outcome.status, "SUCCEEDED")
        self.assertEqual(outcome.reported_state["mode"], "DISABLED")
        self.assertEqual(outcome.reported_state["health"], "STREAM_FRAME_STALE")
        self.assertEqual(
            outcome.reported_state["lastAdjustmentReason"], "STREAM_FRAME_STALE"
        )
        self.assertEqual(encoder.set_calls, [])

    def test_runtime_probe_failures_publish_health_reason_without_encoder_mutation(
        self,
    ) -> None:
        state: dict[str, object] = {
            "evidence": StreamRuntimeEvidence(True, 100.0),
        }

        def runtime_evidence() -> StreamRuntimeEvidence:
            evidence = state["evidence"]
            if isinstance(evidence, BaseException):
                raise evidence
            return evidence  # type: ignore[return-value]

        encoder = _FakeEncoder()
        reconciler = StreamPolicyReconciler(
            encoder,
            runtime_evidence=runtime_evidence,
            health_clock=lambda: 100.0,
        )
        reconciler.reconcile(_command(_adaptive_payload()))
        expected = (
            (StreamRuntimeEvidence(False, 100.0), "STREAM_PIPELINE_NOT_RUNNING"),
            (RuntimeError("probe unavailable"), "STREAM_HEALTH_PROBE_FAILED"),
        )

        for evidence, reason in expected:
            with self.subTest(reason=reason):
                state["evidence"] = evidence
                update = reconciler.observe_runtime_health()
                self.assertIsNotNone(update)
                self.assertEqual(update.reported_state["health"], reason)
                self.assertEqual(update.reported_state["lastAdjustmentReason"], reason)
                reconciler.observation_committed(update)

        self.assertEqual(encoder.set_calls, [1000])


if __name__ == "__main__":
    unittest.main()
