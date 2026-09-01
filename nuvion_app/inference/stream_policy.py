from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from nuvion_app.inference.command_inbox import CommandEffectOutcome
from nuvion_app.inference.effect_reconciler import (
    ObservedStateUpdate,
)
from nuvion_app.inference.fleet_command import (
    VerifiedFleetCommand,
    _validate_command_payload,
)

STREAM_POLICY_COMMAND_TYPE = "STREAM_POLICY"
STREAM_POLICY_CAPABILITY = "command.stream.policy"
STREAM_POLICY_DEFAULT_DECREASE_FACTOR = 0.75
STREAM_POLICY_DEFAULT_INCREASE_STEP_KBPS = 100
STREAM_POLICY_DEFAULT_CONGESTION_SAMPLES = 3
STREAM_POLICY_DEFAULT_RECOVERY_SAMPLES = 8
STREAM_POLICY_DEFAULT_COOLDOWN_SECONDS = 5
STREAM_POLICY_PRIMARY_STALE_SECONDS = 5


@dataclass(frozen=True)
class StreamPolicy:
    policy_version: int
    mode: str
    target_bitrate_kbps: int | None
    initial_bitrate_kbps: int | None
    min_bitrate_kbps: int | None
    max_bitrate_kbps: int | None
    congestion_samples: int
    recovery_samples: int
    cooldown_seconds: int
    increase_step_kbps: int
    decrease_factor: float
    ewma_alpha: float
    congestion_loss_pct: float
    congestion_rtt_ms: int
    uplink_utilization_pct: int

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> StreamPolicy:
        _validate_command_payload(STREAM_POLICY_COMMAND_TYPE, payload)
        mode = str(payload["mode"])
        if mode == "DISABLED":
            return cls(
                policy_version=int(payload["policyVersion"]),
                mode=mode,
                target_bitrate_kbps=None,
                initial_bitrate_kbps=None,
                min_bitrate_kbps=None,
                max_bitrate_kbps=None,
                congestion_samples=1,
                recovery_samples=1,
                cooldown_seconds=1,
                increase_step_kbps=0,
                decrease_factor=0.75,
                ewma_alpha=0.25,
                congestion_loss_pct=5.0,
                congestion_rtt_ms=250,
                uplink_utilization_pct=80,
            )
        if mode == "FIXED":
            target = int(payload["targetBitrateKbps"])
            return cls(
                policy_version=int(payload["policyVersion"]),
                mode=mode,
                target_bitrate_kbps=target,
                initial_bitrate_kbps=None,
                min_bitrate_kbps=target,
                max_bitrate_kbps=target,
                congestion_samples=1,
                recovery_samples=1,
                cooldown_seconds=1,
                increase_step_kbps=0,
                decrease_factor=0.75,
                ewma_alpha=0.25,
                congestion_loss_pct=5.0,
                congestion_rtt_ms=250,
                uplink_utilization_pct=80,
            )
        minimum = int(payload["minBitrateKbps"])
        maximum = int(payload["maxBitrateKbps"])
        return cls(
            policy_version=int(payload["policyVersion"]),
            mode=mode,
            target_bitrate_kbps=None,
            initial_bitrate_kbps=int(payload["initialBitrateKbps"]),
            min_bitrate_kbps=minimum,
            max_bitrate_kbps=maximum,
            congestion_samples=int(
                payload.get(
                    "congestionSamples",
                    STREAM_POLICY_DEFAULT_CONGESTION_SAMPLES,
                )
            ),
            recovery_samples=int(
                payload.get("recoverySamples", STREAM_POLICY_DEFAULT_RECOVERY_SAMPLES)
            ),
            cooldown_seconds=int(
                payload.get(
                    "cooldownSeconds",
                    STREAM_POLICY_DEFAULT_COOLDOWN_SECONDS,
                )
            ),
            increase_step_kbps=int(
                payload.get(
                    "increaseStepKbps",
                    STREAM_POLICY_DEFAULT_INCREASE_STEP_KBPS,
                )
            ),
            decrease_factor=float(
                payload.get(
                    "decreaseFactor",
                    STREAM_POLICY_DEFAULT_DECREASE_FACTOR,
                )
            ),
            ewma_alpha=0.25,
            congestion_loss_pct=5.0,
            congestion_rtt_ms=250,
            uplink_utilization_pct=80,
        )


@dataclass(frozen=True)
class AdaptiveDecision:
    bitrate_kbps: int
    changed: bool
    state: str
    reason: str
    ewma_loss_pct: float | None
    ewma_rtt_ms: float | None
    ewma_uplink_kbps: float | None


def _optional_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


class AdaptiveBitrateController:
    """Deterministic EWMA classification with AIMD bitrate changes."""

    def __init__(
        self,
        policy: StreamPolicy,
        *,
        primary_stale_ms: float = STREAM_POLICY_PRIMARY_STALE_SECONDS * 1000.0,
    ) -> None:
        if policy.mode != "ADAPTIVE":
            raise ValueError("adaptive controller requires ADAPTIVE policy")
        self.policy = policy
        self.current_bitrate_kbps = int(policy.initial_bitrate_kbps or 0)
        self.ewma_loss_pct: float | None = None
        self.ewma_rtt_ms: float | None = None
        self.ewma_uplink_kbps: float | None = None
        self._congested_samples = 0
        self._stable_samples = 0
        self._last_change_ms: float | None = None
        self._signal_source: str | None = None
        self._last_primary_ms: float | None = None
        self._primary_stale_ms = max(100.0, float(primary_stale_ms))
        self.last_state = "STARTUP"
        self.last_reason = "policy_activated"

    def set_current_bitrate(self, bitrate_kbps: int) -> None:
        self.current_bitrate_kbps = int(bitrate_kbps)

    def _ewma(self, previous: float | None, current: float | None) -> float | None:
        if current is None:
            return previous
        if previous is None:
            return current
        alpha = self.policy.ewma_alpha
        return (alpha * current) + ((1.0 - alpha) * previous)

    def observe(
        self,
        sample: Mapping[str, Any],
        *,
        now_ms: float,
    ) -> AdaptiveDecision:
        primary_loss = _optional_number(sample.get("outboundPacketLossPct"))
        primary_rtt = _optional_number(sample.get("outboundRttMs"))
        nack_delta = _optional_number(sample.get("nackDelta"))
        pli_delta = _optional_number(sample.get("pliDelta"))
        queue_pressure = _optional_number(sample.get("queuePressurePct"))
        has_primary_signal = any(
            value is not None
            for value in (
                primary_loss,
                primary_rtt,
                nack_delta,
                pli_delta,
                queue_pressure,
            )
        )
        auxiliary_loss = _optional_number(sample.get("packetLossPct"))
        auxiliary_rtt = _optional_number(sample.get("rttMs"))
        auxiliary_uplink = _optional_number(sample.get("uplinkKbps"))
        loss = primary_loss if has_primary_signal else auxiliary_loss
        rtt = primary_rtt if has_primary_signal else auxiliary_rtt
        uplink = None if has_primary_signal else auxiliary_uplink
        quality_raw = str(sample.get("quality") or "").strip().upper()
        quality = (
            None
            if has_primary_signal
            else quality_raw if quality_raw in {"GOOD", "POOR"} else None
        )

        has_auxiliary_signal = any(
            value is not None
            for value in (quality, auxiliary_loss, auxiliary_rtt, auxiliary_uplink)
        )
        if has_primary_signal:
            self._last_primary_ms = now_ms
        elif (
            has_auxiliary_signal
            and self._last_primary_ms is not None
            and now_ms - self._last_primary_ms < self._primary_stale_ms
        ):
            # Connectivity and WebRTC samplers run concurrently. A fresh
            # primary RTP signal owns the loop; auxiliary samples must not
            # thrash EWMA/hysteresis by toggling the source every interval.
            return self._decision(False, "HOLD", "primary_signal_fresh")
        signal_source = (
            "PRIMARY"
            if has_primary_signal
            else "AUXILIARY" if has_auxiliary_signal else None
        )
        if (
            signal_source is not None
            and self._signal_source is not None
            and self._signal_source != signal_source
        ):
            self.ewma_loss_pct = None
            self.ewma_rtt_ms = None
            self.ewma_uplink_kbps = None
            self._congested_samples = 0
            self._stable_samples = 0
        if signal_source is not None:
            self._signal_source = signal_source

        self.ewma_loss_pct = self._ewma(self.ewma_loss_pct, loss)
        self.ewma_rtt_ms = self._ewma(self.ewma_rtt_ms, rtt)
        self.ewma_uplink_kbps = self._ewma(self.ewma_uplink_kbps, uplink)

        reasons: list[str] = []
        if quality == "POOR":
            reasons.append("connectivity_poor")
        if (
            self.ewma_loss_pct is not None
            and self.ewma_loss_pct >= self.policy.congestion_loss_pct
        ):
            reasons.append("packet_loss_high")
        if (
            self.ewma_rtt_ms is not None
            and self.ewma_rtt_ms >= self.policy.congestion_rtt_ms
        ):
            reasons.append("round_trip_time_high")
        if not has_primary_signal and self.ewma_uplink_kbps is not None:
            safe_uplink = (
                self.ewma_uplink_kbps * self.policy.uplink_utilization_pct / 100.0
            )
            if self.current_bitrate_kbps > safe_uplink:
                reasons.append("link_capacity_low")
        if has_primary_signal and nack_delta is not None and nack_delta > 0:
            reasons.append("nack_increase")
        if has_primary_signal and pli_delta is not None and pli_delta > 0:
            reasons.append("pli_increase")
        if (
            has_primary_signal
            and queue_pressure is not None
            and queue_pressure >= 80.0
        ):
            reasons.append("queue_pressure_high")

        has_signal = any(
            value is not None
            for value in (
                quality,
                loss,
                rtt,
                uplink,
                nack_delta,
                pli_delta,
                queue_pressure,
            )
        )
        congested = bool(reasons)
        if not has_signal:
            self._congested_samples = 0
            self._stable_samples = 0
            return self._decision(False, "HOLD", "insufficient_signal")

        if congested:
            self._congested_samples += 1
            self._stable_samples = 0
        else:
            self._stable_samples += 1
            self._congested_samples = 0

        in_cooldown = (
            self._last_change_ms is not None
            and now_ms - self._last_change_ms
            < self.policy.cooldown_seconds * 1000.0
        )
        if in_cooldown:
            return self._decision(
                False,
                "HOLD",
                "cooldown:" + (",".join(reasons) if reasons else "stable"),
            )

        if congested and self._congested_samples >= self.policy.congestion_samples:
            previous = self.current_bitrate_kbps
            self.current_bitrate_kbps = max(
                int(self.policy.min_bitrate_kbps or previous),
                math.floor(previous * self.policy.decrease_factor),
            )
            self._congested_samples = 0
            changed = self.current_bitrate_kbps != previous
            if changed:
                self._last_change_ms = now_ms
            return self._decision(
                changed,
                "DECREASE" if changed else "HOLD",
                ",".join(reasons) if changed else "at_minimum",
            )

        if not congested and self._stable_samples >= self.policy.recovery_samples:
            previous = self.current_bitrate_kbps
            self.current_bitrate_kbps = min(
                int(self.policy.max_bitrate_kbps or previous),
                previous + self.policy.increase_step_kbps,
            )
            self._stable_samples = 0
            changed = self.current_bitrate_kbps != previous
            if changed:
                self._last_change_ms = now_ms
            return self._decision(
                changed,
                "INCREASE" if changed else "HOLD",
                "stable" if changed else "at_maximum",
            )

        return self._decision(
            False,
            "HOLD",
            "awaiting_hysteresis:" + (",".join(reasons) if reasons else "stable"),
        )

    def _decision(self, changed: bool, state: str, reason: str) -> AdaptiveDecision:
        self.last_state = state
        self.last_reason = reason
        return AdaptiveDecision(
            bitrate_kbps=self.current_bitrate_kbps,
            changed=changed,
            state=state,
            reason=reason,
            ewma_loss_pct=self.ewma_loss_pct,
            ewma_rtt_ms=self.ewma_rtt_ms,
            ewma_uplink_kbps=self.ewma_uplink_kbps,
        )


class EncoderAdapter(Protocol):
    name: str

    def read_bitrate_kbps(self) -> int: ...

    def set_bitrate_kbps(self, bitrate_kbps: int) -> int: ...


_T = TypeVar("_T")


class GlibMainContextDispatcher:
    """Synchronously marshal one short encoder property mutation to GLib."""

    def __init__(
        self,
        idle_add: Callable[..., int],
        *,
        timeout_seconds: float = 3.0,
    ) -> None:
        self._idle_add = idle_add
        self._timeout_seconds = max(0.1, float(timeout_seconds))

    def __call__(self, operation: Callable[[], _T]) -> _T:
        completed = threading.Event()
        cancelled = threading.Event()
        result: list[_T] = []
        errors: list[BaseException] = []

        def invoke() -> bool:
            try:
                if cancelled.is_set():
                    return False
                result.append(operation())
            except BaseException as exc:  # noqa: BLE001 - re-raised to worker.
                errors.append(exc)
            finally:
                completed.set()
            return False

        source_id = self._idle_add(invoke)
        if not source_id:
            raise RuntimeError("failed to schedule encoder mutation on GLib context")
        if not completed.wait(self._timeout_seconds):
            cancelled.set()
            raise TimeoutError("GLib encoder mutation timed out")
        if errors:
            raise errors[0]
        return result[0]


class X264EncoderAdapter:
    name = "x264enc"

    def __init__(
        self,
        element: Any,
        *,
        dispatch: Callable[[Callable[[], Any]], Any] | None = None,
    ) -> None:
        if element is None:
            raise ValueError("named x264 encoder element is required")
        self._element = element
        self._dispatch = dispatch or (lambda operation: operation())
        self._fence_lock = threading.Lock()
        self._effect_fence: Callable[[], None] = lambda: None

    def set_effect_fence(self, fence_check: Callable[[], None]) -> None:
        if not callable(fence_check):
            raise TypeError("encoder effect fence must be callable")
        with self._fence_lock:
            self._effect_fence = fence_check

    def _ensure_fence(self) -> None:
        with self._fence_lock:
            fence = self._effect_fence
        fence()

    def read_bitrate_kbps(self) -> int:
        def read() -> Any:
            self._ensure_fence()
            return self._element.get_property("bitrate")

        value = self._dispatch(read)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("x264 bitrate property did not return a number")
        return int(value)

    def set_bitrate_kbps(self, bitrate_kbps: int) -> int:
        target = int(bitrate_kbps)
        if target < 100 or target > 20_000:
            raise ValueError("x264 bitrate must be in [100, 20000] Kbps")

        def apply_and_read() -> int:
            self._ensure_fence()
            self._element.set_property("bitrate", target)
            return int(self._element.get_property("bitrate"))

        applied = int(self._dispatch(apply_and_read))
        if applied != target:
            raise RuntimeError(
                f"x264 bitrate readback mismatch requested={target} applied={applied}"
            )
        return applied


class StreamPolicyReconciler:
    command_type = STREAM_POLICY_COMMAND_TYPE
    capability = STREAM_POLICY_CAPABILITY

    def __init__(
        self,
        encoder: EncoderAdapter,
        *,
        clock_ms: Callable[[], float] | None = None,
    ) -> None:
        self.encoder = encoder
        self._clock_ms = clock_ms or (lambda: time.monotonic() * 1000.0)
        self._lock = threading.RLock()
        self._command_id: str | None = None
        self._policy: StreamPolicy | None = None
        self._controller: AdaptiveBitrateController | None = None
        self._reported_state: dict[str, Any] | None = None
        self._observation_pending = False
        self._effect_fence: Callable[[], None] = lambda: None

    def set_effect_fence(self, fence_check: Callable[[], None]) -> None:
        if not callable(fence_check):
            raise TypeError("stream effect fence must be callable")
        with self._lock:
            self._effect_fence = fence_check
            bind_encoder_fence = getattr(self.encoder, "set_effect_fence", None)
            if callable(bind_encoder_fence):
                bind_encoder_fence(fence_check)

    def _ensure_fence(self) -> None:
        self._effect_fence()

    def reconcile(self, command: VerifiedFleetCommand) -> CommandEffectOutcome:
        policy = StreamPolicy.from_payload(command.payload)
        with self._lock:
            self._ensure_fence()
            self._policy = policy
            if policy.mode == "DISABLED":
                self._ensure_fence()
                applied = self.encoder.read_bitrate_kbps()
                self._controller = None
                reason = "policy_disabled"
            else:
                self._ensure_fence()
                applied = self.encoder.set_bitrate_kbps(
                    int(
                        policy.target_bitrate_kbps
                        or policy.initial_bitrate_kbps
                        or 0
                    )
                )
                if policy.mode == "ADAPTIVE":
                    self._controller = AdaptiveBitrateController(policy)
                    self._controller.set_current_bitrate(applied)
                    reason = "policy_activated"
                else:
                    self._controller = None
                    reason = "fixed_target_applied"
            self._command_id = command.command_id
            self._reported_state = self._snapshot(
                applied_bitrate_kbps=applied,
                reason=reason,
            )
            self._observation_pending = False
            return CommandEffectOutcome.succeeded(self._reported_state)

    def observe_connectivity(
        self, sample: Mapping[str, Any]
    ) -> ObservedStateUpdate | None:
        with self._lock:
            if self._controller is None or self._command_id is None:
                return None
            self._ensure_fence()
            if self._observation_pending and self._reported_state is not None:
                # Preserve every applied transition: do not mutate the encoder
                # again until the previous runtime state is durably enqueued.
                return ObservedStateUpdate(
                    command_type=self.command_type,
                    command_id=self._command_id,
                    reported_state=dict(self._reported_state),
                )
            previous_reported = dict(self._reported_state or {})
            previous = self._controller.current_bitrate_kbps
            decision = self._controller.observe(sample, now_ms=self._clock_ms())
            applied = previous
            if decision.changed:
                try:
                    self._ensure_fence()
                    applied = self.encoder.set_bitrate_kbps(decision.bitrate_kbps)
                except Exception:
                    self._controller.set_current_bitrate(previous)
                    raise
                self._controller.set_current_bitrate(applied)
            else:
                self._ensure_fence()
                applied = self.encoder.read_bitrate_kbps()
                self._controller.set_current_bitrate(applied)
            self._reported_state = self._snapshot(
                applied_bitrate_kbps=applied,
                reason=decision.reason,
            )
            if (
                previous_reported.get("appliedBitrateKbps")
                == self._reported_state.get("appliedBitrateKbps")
                and previous_reported.get("health")
                == self._reported_state.get("health")
                and not self._observation_pending
            ):
                return None
            return ObservedStateUpdate(
                command_type=self.command_type,
                command_id=self._command_id,
                reported_state=dict(self._reported_state),
            )

    def observation_failed(self, update: ObservedStateUpdate) -> None:
        with self._lock:
            if update.command_id == self._command_id:
                self._observation_pending = True

    def observation_committed(self, update: ObservedStateUpdate) -> None:
        with self._lock:
            if (
                update.command_id == self._command_id
                and self._reported_state == update.reported_state
            ):
                self._observation_pending = False

    def observe_stream_metrics(
        self, sample: Mapping[str, Any]
    ) -> ObservedStateUpdate | None:
        return self.observe_connectivity(sample)

    def reported_state(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._reported_state) if self._reported_state else None

    def restore_applied(
        self,
        command: VerifiedFleetCommand,
        _persisted_state: Mapping[str, Any],
    ) -> dict[str, Any]:
        outcome = self.reconcile(command)
        if outcome.reported_state is None:
            raise RuntimeError("restored stream policy did not report encoder state")
        restored = dict(outcome.reported_state)
        if (
            _persisted_state.get("appliedBitrateKbps")
            == restored.get("appliedBitrateKbps")
            and _persisted_state.get("health") == restored.get("health")
        ):
            restored["lastAdjustmentReason"] = _persisted_state.get(
                "lastAdjustmentReason",
                "restored_after_restart",
            )
        with self._lock:
            self._reported_state = restored
        return restored

    def _snapshot(
        self,
        *,
        applied_bitrate_kbps: int,
        reason: str,
    ) -> dict[str, Any]:
        if self._policy is None:
            raise RuntimeError("stream policy is not active")
        reported: dict[str, Any] = {
            "policyVersion": self._policy.policy_version,
            "mode": self._policy.mode,
            "encoder": self.encoder.name,
            "requestedBitrateKbps": (
                self._policy.target_bitrate_kbps
                or self._policy.initial_bitrate_kbps
                or int(applied_bitrate_kbps)
            ),
            "appliedBitrateKbps": int(applied_bitrate_kbps),
            "lastAdjustmentReason": reason,
            "health": "STREAM_CONTINUOUS",
        }
        if self._policy.mode == "ADAPTIVE":
            reported.update(
                {
                    "minBitrateKbps": self._policy.min_bitrate_kbps,
                    "maxBitrateKbps": self._policy.max_bitrate_kbps,
                    "initialBitrateKbps": self._policy.initial_bitrate_kbps,
                    "decreaseFactor": self._policy.decrease_factor,
                    "increaseStepKbps": self._policy.increase_step_kbps,
                    "congestionSamples": self._policy.congestion_samples,
                    "recoverySamples": self._policy.recovery_samples,
                    "cooldownSeconds": self._policy.cooldown_seconds,
                }
            )
        elif self._policy.mode == "FIXED":
            reported["targetBitrateKbps"] = self._policy.target_bitrate_kbps
        return reported
