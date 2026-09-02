from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def evaluate_update_commit_readiness(
    *,
    now_monotonic: float,
    signaling_ready_since: float | None,
    stomp_last_send_at: float | None,
    min_stable_seconds: float,
    max_evidence_age_seconds: float,
    pipeline_running: bool,
    pipeline_last_frame_at: float | None,
    webrtc_health: Mapping[str, Any],
    stomp_blocked: bool,
    event_outbox_health: Mapping[str, Any],
    command_outbox_health: Mapping[str, Any],
) -> dict[str, object]:
    """Return fail-closed live candidate evidence required before OTA commit.

    Merely constructing a pipeline or connecting STOMP is insufficient.  A
    candidate must have recent camera frames, a current-process STOMP send, a
    connected ICE/WebRTC session, and two increasing outbound RTP samples.
    """

    if min_stable_seconds < 1.0:
        raise ValueError("min_stable_seconds must be at least one second")
    if max_evidence_age_seconds < 1.0:
        raise ValueError("max_evidence_age_seconds must be at least one second")
    if not math.isfinite(now_monotonic) or now_monotonic <= 0.0:
        raise ValueError("now_monotonic must be a positive finite number")

    def age(value: object) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        timestamp = float(value)
        if not math.isfinite(timestamp) or timestamp <= 0.0 or timestamp > now_monotonic:
            return None
        return now_monotonic - timestamp

    if not pipeline_running:
        return {"ready": False, "reason": "PIPELINE_NOT_RUNNING"}
    frame_age = age(pipeline_last_frame_at)
    if frame_age is None:
        return {"ready": False, "reason": "CAMERA_FRAME_UNAVAILABLE"}
    if frame_age > max_evidence_age_seconds:
        return {"ready": False, "reason": "CAMERA_FRAME_STALE"}
    if webrtc_health.get("hasPipeline") is not True:
        return {"ready": False, "reason": "WEBRTC_PIPELINE_NOT_ATTACHED"}
    if signaling_ready_since is None or signaling_ready_since <= 0.0:
        return {"ready": False, "reason": "STOMP_NOT_CONNECTED"}
    if stomp_blocked:
        return {"ready": False, "reason": "STOMP_UPLINK_BLOCKED"}

    stable_seconds = max(0.0, now_monotonic - signaling_ready_since)
    if stable_seconds < min_stable_seconds:
        return {
            "ready": False,
            "reason": "STOMP_SOAK_PENDING",
            "stableSeconds": int(stable_seconds),
        }
    stomp_send_age = age(stomp_last_send_at)
    if stomp_send_age is None or float(stomp_last_send_at or 0.0) < signaling_ready_since:
        return {"ready": False, "reason": "STOMP_SEND_UNPROVEN"}
    if stomp_send_age > max_evidence_age_seconds:
        return {"ready": False, "reason": "STOMP_SEND_STALE"}

    session_id = webrtc_health.get("sessionId")
    generation = webrtc_health.get("generation")
    if (
        not isinstance(session_id, str)
        or not session_id
        or isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 1
    ):
        return {"ready": False, "reason": "WEBRTC_SESSION_UNAVAILABLE"}
    if str(webrtc_health.get("connectionState") or "").lower() != "connected":
        return {"ready": False, "reason": "WEBRTC_NOT_CONNECTED"}
    if str(webrtc_health.get("iceConnectionState") or "").lower() not in {
        "connected",
        "completed",
    }:
        return {"ready": False, "reason": "WEBRTC_ICE_NOT_CONNECTED"}

    connection_age = age(webrtc_health.get("connectedSince"))
    ice_age = age(webrtc_health.get("iceConnectedSince"))
    if connection_age is None or ice_age is None:
        return {"ready": False, "reason": "WEBRTC_DWELL_UNPROVEN"}
    webrtc_stable_seconds = min(connection_age, ice_age)
    if webrtc_stable_seconds < min_stable_seconds:
        return {
            "ready": False,
            "reason": "WEBRTC_SOAK_PENDING",
            "stableSeconds": int(webrtc_stable_seconds),
        }

    progress_samples = webrtc_health.get("outboundProgressSamples")
    if (
        isinstance(progress_samples, bool)
        or not isinstance(progress_samples, int)
        or progress_samples < 2
    ):
        return {"ready": False, "reason": "WEBRTC_RTP_PROGRESS_UNPROVEN"}
    progress_age = age(webrtc_health.get("lastOutboundProgressAt"))
    if progress_age is None or progress_age > max_evidence_age_seconds:
        return {"ready": False, "reason": "WEBRTC_RTP_PROGRESS_STALE"}

    if event_outbox_health.get("capacityState") != "HEALTHY":
        return {"ready": False, "reason": "EVENT_OUTBOX_UNHEALTHY"}
    if int(event_outbox_health.get("blockedRows") or 0) > 0:
        return {"ready": False, "reason": "EVENT_OUTBOX_BLOCKED"}
    if event_outbox_health.get("unsavedCriticalEvents") not in {0, None}:
        return {"ready": False, "reason": "CRITICAL_EVENT_NOT_DURABLE"}
    if event_outbox_health.get("safetyStop") is True:
        return {"ready": False, "reason": "CRITICAL_EVENT_SAFETY_STOP"}
    if event_outbox_health.get("protocolStop") is True:
        return {"ready": False, "reason": "EVENT_PROTOCOL_STOP"}
    if command_outbox_health.get("capacityState") != "HEALTHY":
        return {"ready": False, "reason": "COMMAND_OUTBOX_UNHEALTHY"}
    if int(command_outbox_health.get("dlqBlockedRows") or 0) > 0:
        return {"ready": False, "reason": "COMMAND_OUTBOX_DLQ_BLOCKED"}
    if command_outbox_health.get("retentionPressure") is True:
        return {"ready": False, "reason": "COMMAND_OUTBOX_RETENTION_PRESSURE"}

    return {
        "ready": True,
        "reason": "READY",
        "stableSeconds": int(min(stable_seconds, webrtc_stable_seconds)),
        "webrtcSessionId": session_id,
        "webrtcGeneration": generation,
    }
