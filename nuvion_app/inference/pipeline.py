# nuvion_app/inference/pipeline.py
#
# USB/Webcam -> GStreamer -> H.264(RTP) -> mediasoup plain transport
# Zero-shot anomaly detection (SigLIP) or Triton backend (optional)

import base64
import os
import sys
import json
import time
import queue
import warnings
import random
import string
import asyncio
import logging
import threading
import glob
import math
import shutil
import sqlite3
import subprocess
import urllib.request
import urllib.error
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
from nuvion_app.config import load_env, resolve_config_path
import aiohttp
import websockets
from nuvion_app.inference.connectivity import ConnectivityReporter
from nuvion_app.inference.connectivity import ConnectivityThresholds
from nuvion_app.inference.clip_segments import list_stable_segments
from nuvion_app.inference.command_runtime import (
    FleetCommandRuntime,
    FleetCommandRuntimeError,
    build_fleet_command_runtime_from_env,
)
from nuvion_app.inference.agent_update import AgentUpdateReconciler
from nuvion_app.inference.effect_reconciler import ReconcilerRegistry
from nuvion_app.inference.fleet_command import (
    COMMAND_CAPABILITY_BY_TYPE,
    VerifiedFleetCommand,
)
from nuvion_app.inference.demo_mvtec import MvtecDemoSource
from nuvion_app.inference.demo_mvtec import prepare_mvtec_demo_source
from nuvion_app.inference.depthai_gst import DepthAIGStreamerBridge
from nuvion_app.inference.depthai_source import DepthAIConfig
from nuvion_app.inference.depthai_source import DepthAIFrameSource
from nuvion_app.inference.device_state import (
    CONNECTIVITY_QUALITY_GOOD,
    INSPECTION_STATUS_DEFECT,
    INSPECTION_STATUS_NORMAL,
    RUNTIME_STATUS_ERROR,
    RUNTIME_STATUS_RUNNING,
    DeviceStateCoordinator,
)
from nuvion_app.inference.durable_events import (
    DEFAULT_DLQ_MAX_BYTES,
    DEFAULT_DLQ_MAX_ROWS,
    DEFAULT_CRITICAL_SAFETY_MAX_BYTES,
    DEFAULT_DELIVERY_CLASS_BY_EVENT_TYPE,
    DEFAULT_OUTBOX_MAX_AGE_SECONDS,
    DEFAULT_OUTBOX_MAX_BYTES,
    DEFAULT_OUTBOX_MAX_ROWS,
    DELIVERY_CLASS_CRITICAL,
    EVENT_TYPE_ANOMALY,
    EVENT_TYPE_CONNECTIVITY,
    EVENT_TYPE_DEVICE_STATE,
    EVENT_TYPE_METRIC,
    EVENT_TYPE_PRODUCTION,
    DurableEvent,
    DurableEventCapacityError,
    DurableEventDelivery,
    DurableEventOutbox,
    is_uncorrelated_permanent_event_rejection,
    parse_permanent_event_rejection,
    resolve_default_outbox_path,
    utc_now_iso,
)
from nuvion_app.inference.critical_event_safety import (
    CriticalEventBackpressureError,
    CriticalEventSafetyGate,
    PendingCriticalEvent,
)
from nuvion_app.inference.face_tracking import FaceTrackingController
from nuvion_app.inference.face_tracking import TrackingOverlaySnapshot
from nuvion_app.inference.face_tracking import TrackingOverlayState
from nuvion_app.inference.face_tracking import build_face_detector
from nuvion_app.inference.face_tracking import build_overlay_snapshot
from nuvion_app.inference.face_tracking import draw_tracking_overlay
from nuvion_app.inference.motor import MotorController
from nuvion_app.inference.motor import motor_config_from_env
from nuvion_app.inference.snapshot import LatestFrameBuffer
from nuvion_app.inference.snapshot import capture_and_upload_snapshot
from nuvion_app.inference.stream_policy import (
    GlibMainContextDispatcher,
    StreamPolicyReconciler,
    X264EncoderAdapter,
)
from nuvion_app.inference.settings_reconciler import (
    AtomicSettingsStore,
    SettingsReconciler,
    UnsupportedSettingsEffect,
)
from nuvion_app.inference.signaling_contract import (
    AGENT_COMMAND_QUEUE_DEST,
    AGENT_ERROR_QUEUE_DEST,
    COMMAND_OBSERVED_ACK_QUEUE_DEST,
    EVENT_ACK_QUEUE_DEST,
    FLEET_COMMAND_QUEUE_DEST,
    REQUIRED_AGENT_SUBSCRIPTIONS,
)
from nuvion_app.inference.webrtc_signaling import (
    WEBRTC_UPLINK_ANSWER,
    WEBRTC_UPLINK_ICE_CANDIDATE,
    WEBRTC_UPLINK_OFFER_DEST,
    WEBRTC_UPLINK_ICE_CANDIDATE_DEST,
    WEBRTC_UPLINK_START,
    WEBRTC_UPLINK_STATE,
    WEBRTC_UPLINK_STOP_DEST,
    h264_level_from_profile_level_id,
    negotiate_stomp_send_interval_ms,
    parse_command_payload,
)
from nuvion_app.inference.webrtc_uplink import (
    WebRTCSignalingToken,
    WebRTCUplinkController,
)

# Python 3.14에서 third-party stomper 패키지의 legacy regex 문자열로
# SyntaxWarning(invalid escape sequence)가 발생한다. 런타임 동작에는 영향이 없어
# 해당 모듈 경고만 제한적으로 숨긴다.
warnings.filterwarnings(
    "ignore",
    message=r".*invalid escape sequence.*",
    category=SyntaxWarning,
)
import stomper

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

from nuvion_app.inference.zero_shot import ZeroShotAnomalyDetector
from nuvion_app.inference.video_source import build_video_source_pipeline
from nuvion_app.inference.video_source import DEPTHAI_APPSRC_NAME
from nuvion_app.inference.video_source import resolve_depthai_device_id
from nuvion_app.inference.video_source import should_use_depthai_source
from nuvion_app.inference.video_source import is_truthy
from nuvion_app.runtime.inference_mode import (
    apply_inference_runtime_defaults,
    normalize_backend,
    normalize_siglip_device,
)
from nuvion_app.model_store import DEFAULT_MODEL_POINTER
from nuvion_app.runtime.model_guard import resolve_effective_profile, resolve_model_dir
from nuvion_app.runtime.platform_identity import (
    IDENTITY_STATUS_DEV,
    IDENTITY_STATUS_VERIFIED,
    resolve_platform_identity,
)
from nuvion_app.runtime.telemetry import (
    build_runtime_telemetry,
    merge_runtime_public_state,
    verify_model_artifact_identity,
)
from nuvion_app.runtime.settings_overlay import resolve_settings_state_dir
from nuvion_app.runtime.updater_client import (
    UpdaterClient,
    build_updater_capability_telemetry,
)
from nuvion_app.runtime.update_commit_readiness import (
    evaluate_update_commit_readiness,
)
from nuvion_app.runtime.update_health_attestation import (
    build_health_attestation_request,
    request_health_attestation,
)

try:
    from nuvion_app.agent.triton_client import TritonAnomalyClient
except Exception:
    TritonAnomalyClient = None

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

OVERLAY_COLOR_WHITE = 0xFFFFFFFF
OVERLAY_COLOR_GREEN = 0xFF00FF00
OVERLAY_COLOR_RED = 0xFFFF0000
DEMO_OVERLAY_STATUS_XPAD = 25
DEMO_OVERLAY_LABEL_XPAD = 175
DEMO_OVERLAY_SCORE_XPAD = 360
DEMO_OVERLAY_GT_XPAD = 470


@dataclass(frozen=True)
class OverlayPayload:
    status: str
    label: str
    score: float
    ground_truth: str | None = None

    @property
    def score_text(self) -> str:
        return f"{self.score:.2f}"

    @property
    def matches_ground_truth(self) -> bool | None:
        if self.ground_truth == "normal":
            return self.status == "NORMAL"
        if self.ground_truth == "defect":
            return self.status == "DEFECT"
        return None


def parse_csv(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_label_array(encoded_value: str | None, csv_value: str) -> list[str]:
    encoded = str(encoded_value or "").strip()
    if not encoded:
        return parse_csv(csv_value)
    try:
        padding = "=" * ((4 - len(encoded) % 4) % 4)
        decoded = base64.b64decode(
            encoded + padding,
            altchars=b"-_",
            validate=True,
        )
        payload = json.loads(decoded.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        log.error("[CONFIG-APPLY] invalid encoded label array; using legacy CSV")
        return parse_csv(csv_value)
    if (
        not isinstance(payload, list)
        or not 1 <= len(payload) <= 100
        or any(
            not isinstance(item, str)
            or not item
            or item != item.strip()
            or len(item) > 100
            for item in payload
        )
        or len({item.lower() for item in payload}) != len(payload)
    ):
        log.error("[CONFIG-APPLY] invalid label array payload; using legacy CSV")
        return parse_csv(csv_value)
    return list(payload)


def parse_int(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_int_with_default(value: str | None, default: int) -> int:
    parsed = parse_int(value)
    return parsed if parsed is not None else default


def parse_float(value: str | None, default: float) -> float:
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def systemd_restart_enabled(environ) -> bool:
    return (
        sys.platform.startswith("linux")
        and is_truthy(environ.get("NUVION_SUPERVISOR_RESTART_ENABLED", "false"))
        and bool(str(environ.get("INVOCATION_ID") or "").strip())
    )


load_env()
apply_inference_runtime_defaults()

SERVER_BASE_URL = os.getenv("NUVION_SERVER_BASE_URL", "http://localhost:8080")
DEVICE_USERNAME = os.getenv("NUVION_DEVICE_USERNAME", "device")
DEVICE_PASSWORD = os.getenv("NUVION_DEVICE_PASSWORD", "password")

VIDEO_SOURCE_ENV = os.getenv("NUVION_VIDEO_SOURCE", "auto")
GST_SOURCE_OVERRIDE = os.getenv("NUVION_GST_SOURCE")
DEPTHAI_DEVICE_ID = (os.getenv("NUVION_DEPTHAI_DEVICE_ID", "") or "").strip() or None
DEPTHAI_STARTUP_TIMEOUT_SEC = parse_float(
    os.getenv("NUVION_DEPTHAI_STARTUP_TIMEOUT_SEC"),
    15.0,
)
DEPTHAI_READ_TIMEOUT_SEC = parse_float(
    os.getenv("NUVION_DEPTHAI_READ_TIMEOUT_SEC"),
    2.0,
)
DEPTHAI_MAX_CONSECUTIVE_TIMEOUTS = max(
    parse_int_with_default(os.getenv("NUVION_DEPTHAI_MAX_CONSECUTIVE_TIMEOUTS"), 3),
    1,
)
DEMO_MODE = is_truthy(os.getenv("NUVION_DEMO_MODE", "false"))
DEMO_LOOP = is_truthy(os.getenv("NUVION_DEMO_LOOP", "true"))
DEMO_TAG = ((os.getenv("NUVION_DEMO_TAG", "[DEMO]") or "").strip() or "[DEMO]")

WEBRTC_FORCE_RELAY = is_truthy(os.getenv("NUVION_WEBRTC_FORCE_RELAY", "false"))
RTP_SSRC_ENV = os.getenv("NUVION_RTP_SSRC", None)
H264_PROFILE_LEVEL_ID_ENV = os.getenv("NUVION_H264_PROFILE_LEVEL_ID", "42e01f")
H264_PROFILE_ENV = os.getenv("NUVION_H264_PROFILE", "constrained-baseline")
H264_PACKETIZATION_MODE_ENV = os.getenv("NUVION_H264_PACKETIZATION_MODE", "1")
H264_LEVEL_ASYMMETRY_ALLOWED_ENV = os.getenv("NUVION_H264_LEVEL_ASYMMETRY_ALLOWED", "1")

ANOMALY_LABELS = {label.lower() for label in parse_csv(os.getenv("NUVION_ANOMALY_LABELS", ""))}
PRODUCTION_LABELS = {label.lower() for label in parse_csv(os.getenv("NUVION_PRODUCTION_LABELS", ""))}

ANOMALY_CONFIDENCE_THRESHOLD = parse_float(os.getenv("NUVION_ANOMALY_CONFIDENCE_THRESHOLD"), 0.5)
PRODUCTION_CONFIDENCE_THRESHOLD = parse_float(os.getenv("NUVION_PRODUCTION_CONFIDENCE_THRESHOLD"), 0.5)
ANOMALY_MIN_INTERVAL_SEC = parse_float(os.getenv("NUVION_ANOMALY_MIN_INTERVAL_SEC"), 5.0)
PRODUCTION_DEDUP_SEC = parse_float(os.getenv("NUVION_PRODUCTION_DEDUP_SEC"), 3.0)

ZERO_SHOT_ENABLED = is_truthy(os.getenv("NUVION_ZERO_SHOT_ENABLED", "true"))
ZERO_SHOT_MODEL = os.getenv("NUVION_ZERO_SHOT_MODEL", "google/siglip2-base-patch16-224")
ZERO_SHOT_DEVICE = normalize_siglip_device(os.getenv("NUVION_ZERO_SHOT_DEVICE", "auto"), default="auto")
ZERO_SHOT_LABELS = parse_label_array(
    os.getenv("NUVION_ZERO_SHOT_LABELS_B64"),
    os.getenv("NUVION_ZERO_SHOT_LABELS", "normal,defect"),
)
ZERO_SHOT_ANOMALY_LABELS = parse_label_array(
    os.getenv("NUVION_ZERO_SHOT_ANOMALY_LABELS_B64"),
    os.getenv(
        "NUVION_ZERO_SHOT_ANOMALY_LABELS",
        "defect,broken,crack,scratch",
    ),
)
ZERO_SHOT_THRESHOLD = parse_float(os.getenv("NUVION_ZERO_SHOT_THRESHOLD"), 0.7)
ZERO_SHOT_SAMPLE_SEC = parse_float(os.getenv("NUVION_ZERO_SHOT_SAMPLE_SEC"), 2.0)
FACE_TRACKING_ENABLED = is_truthy(os.getenv("NUVION_FACE_TRACKING_ENABLED", "false"))
FACE_TRACKING_SHOW_BBOX = is_truthy(os.getenv("NUVION_FACE_TRACKING_SHOW_BBOX", "true"))
TRACKING_SAMPLE_SEC = parse_float(os.getenv("NUVION_TRACKING_SAMPLE_SEC"), 0.05)
TRACKING_BATCH_SIZE = max(int(os.getenv("NUVION_FACE_TRACKING_BATCH_SIZE", "2") or "2"), 1)
TRACKING_DEADZONE_PCT = parse_float(os.getenv("NUVION_TRACKING_DEADZONE_PCT"), 0.08)
TRACKING_HYSTERESIS_PCT = parse_float(os.getenv("NUVION_TRACKING_HYSTERESIS_PCT"), 0.05)
TRACKING_LOST_TIMEOUT_SEC = parse_float(os.getenv("NUVION_TRACKING_LOST_TIMEOUT_SEC"), 1.0)

LOCAL_DISPLAY = is_truthy(os.getenv("NUVION_LOCAL_DISPLAY", "false"))

TRITON_THRESHOLD = parse_float(os.getenv("NUVION_TRITON_THRESHOLD"), 0.7)
ZSAD_BACKEND = normalize_backend(os.getenv("NUVION_ZSAD_BACKEND", "triton"), default="triton")

CLIP_ENABLED = is_truthy(os.getenv("NUVION_CLIP_ENABLED", "true"))
CLIP_PRE_SEC = parse_float(os.getenv("NUVION_CLIP_PRE_SEC"), 5.0)
CLIP_POST_SEC = parse_float(os.getenv("NUVION_CLIP_POST_SEC"), 5.0)
CLIP_SEGMENT_SEC = parse_float(os.getenv("NUVION_CLIP_SEGMENT_SEC"), 2.0)
CLIP_MAX_SEGMENTS = int(os.getenv("NUVION_CLIP_MAX_SEGMENTS", "30"))
CLIP_OUTPUT_DIR = os.getenv("NUVION_CLIP_OUTPUT_DIR", "/tmp/nuvion_clips")
CLIP_COOLDOWN_SEC = parse_float(os.getenv("NUVION_CLIP_COOLDOWN_SEC"), 10.0)
CLIP_CONTENT_TYPE = os.getenv("NUVION_CLIP_CONTENT_TYPE", "video/mp4")
SNAPSHOT_ENABLED = is_truthy(os.getenv("NUVION_SNAPSHOT_ENABLED", "true"))
SNAPSHOT_CONTENT_TYPE = os.getenv("NUVION_SNAPSHOT_CONTENT_TYPE", "image/jpeg")
CLIP_WARMUP_SEC = parse_float(
    os.getenv("NUVION_CLIP_WARMUP_SEC"),
    CLIP_PRE_SEC + CLIP_SEGMENT_SEC + 1.0,
)
CLIP_SEGMENT_MIN_AGE_SEC = parse_float(
    os.getenv("NUVION_CLIP_SEGMENT_MIN_AGE_SEC"),
    max(CLIP_SEGMENT_SEC * 1.5, 2.0),
)
CLIP_SEGMENT_MIN_SIZE_BYTES = parse_int_with_default(
    os.getenv("NUVION_CLIP_SEGMENT_MIN_SIZE_BYTES"),
    64 * 1024,
)
VIDEO_WIDTH = parse_int_with_default(os.getenv("NUVION_VIDEO_WIDTH"), 640)
VIDEO_HEIGHT = parse_int_with_default(os.getenv("NUVION_VIDEO_HEIGHT"), 480)
VIDEO_FPS = parse_int_with_default(os.getenv("NUVION_VIDEO_FPS"), 30)
VIDEO_BITRATE_KBPS = parse_int_with_default(
    os.getenv("NUVION_VIDEO_BITRATE_KBPS"),
    1000,
)
MODEL_POINTER = (
    os.getenv("NUVION_MODEL_POINTER", DEFAULT_MODEL_POINTER) or DEFAULT_MODEL_POINTER
).strip()

LINE_ID = parse_int(os.getenv("NUVION_LINE_ID"))
PROCESS_ID = parse_int(os.getenv("NUVION_PROCESS_ID"))
DEVICE_STATE_INTERVAL_SEC = parse_float(os.getenv("NUVION_DEVICE_STATE_INTERVAL_SEC"), 30.0)

CONNECTIVITY_ENABLED = is_truthy(os.getenv("NUVION_CONNECTIVITY_ENABLED", "true"))
CONNECTIVITY_INTERVAL_SEC = parse_float(os.getenv("NUVION_CONNECTIVITY_INTERVAL_SEC"), 10.0)
CONNECTIVITY_MIN_SEND_INTERVAL_SEC = parse_float(os.getenv("NUVION_CONNECTIVITY_MIN_SEND_INTERVAL_SEC"), 30.0)
CONNECTIVITY_POOR_RSSI_DBM = parse_int_with_default(os.getenv("NUVION_CONNECTIVITY_POOR_RSSI_DBM"), -80)
CONNECTIVITY_POOR_PACKET_LOSS_PCT = parse_float(os.getenv("NUVION_CONNECTIVITY_POOR_PACKET_LOSS_PCT"), 8.0)
CONNECTIVITY_POOR_RTT_MS = parse_int_with_default(os.getenv("NUVION_CONNECTIVITY_POOR_RTT_MS"), 250)
CONNECTIVITY_TARGET_HOST = (os.getenv("NUVION_CONNECTIVITY_TARGET_HOST", "") or "").strip()
CONNECTIVITY_WIFI_INTERFACE = (os.getenv("NUVION_WIFI_INTERFACE", "") or "").strip()

OUTBOUND_QUEUE_MAX = int(os.getenv("NUVION_STOMP_QUEUE_MAX", "200"))
EVENT_REPLAY_INTERVAL_SEC = parse_float(os.getenv("NUVION_EVENT_REPLAY_INTERVAL_SEC"), 5.0)
FLEET_COMMAND_POLL_INTERVAL_SEC = parse_float(
    os.getenv("NUVION_FLEET_COMMAND_POLL_INTERVAL_SEC"),
    30.0,
)
FLEET_EFFECT_RECONCILE_INTERVAL_SEC = parse_float(
    os.getenv("NUVION_FLEET_EFFECT_RECONCILE_INTERVAL_SEC"),
    1.0,
)
FLEET_OBSERVATION_REPLAY_INTERVAL_SEC = parse_float(
    os.getenv("NUVION_FLEET_OBSERVATION_REPLAY_INTERVAL_SEC"),
    2.0,
)
WEBRTC_STATS_INTERVAL_SEC = parse_float(
    os.getenv("NUVION_WEBRTC_STATS_INTERVAL_SEC"),
    2.0,
)
UPDATER_TELEMETRY_REFRESH_SEC = parse_float(
    os.getenv("NUVION_UPDATER_TELEMETRY_REFRESH_SEC"),
    5.0,
)
UPDATER_TELEMETRY_TTL_SEC = parse_float(
    os.getenv("NUVION_UPDATER_TELEMETRY_TTL_SEC"),
    15.0,
)
UPDATE_COMMIT_STABLE_SEC = max(
    10.0,
    min(
        parse_float(os.getenv("NUVION_UPDATE_COMMIT_STABLE_SEC"), 15.0),
        60.0,
    ),
)
UPDATE_COMMIT_MAX_EVIDENCE_AGE_SEC = max(
    5.0,
    min(
        parse_float(
            os.getenv("NUVION_UPDATE_COMMIT_MAX_EVIDENCE_AGE_SEC"),
            20.0,
        ),
        60.0,
    ),
)
EVENT_OUTBOX_MAX_ROWS = parse_int_with_default(
    os.getenv("NUVION_EVENT_OUTBOX_MAX_ROWS"),
    DEFAULT_OUTBOX_MAX_ROWS,
)
EVENT_OUTBOX_MAX_BYTES = parse_int_with_default(
    os.getenv("NUVION_EVENT_OUTBOX_MAX_BYTES"),
    DEFAULT_OUTBOX_MAX_BYTES,
)
EVENT_CRITICAL_SAFETY_MAX_BYTES = parse_int_with_default(
    os.getenv("NUVION_EVENT_CRITICAL_SAFETY_MAX_BYTES"),
    DEFAULT_CRITICAL_SAFETY_MAX_BYTES,
)
EVENT_DLQ_MAX_ROWS = parse_int_with_default(
    os.getenv("NUVION_EVENT_DLQ_MAX_ROWS"),
    DEFAULT_DLQ_MAX_ROWS,
)
EVENT_DLQ_MAX_BYTES = parse_int_with_default(
    os.getenv("NUVION_EVENT_DLQ_MAX_BYTES"),
    DEFAULT_DLQ_MAX_BYTES,
)
EVENT_OUTBOX_MAX_AGE_SECONDS = parse_int_with_default(
    os.getenv("NUVION_EVENT_OUTBOX_MAX_AGE_SECONDS"),
    DEFAULT_OUTBOX_MAX_AGE_SECONDS,
)
CLIP_EVENT_ACK_WAIT_SEC = parse_float(os.getenv("NUVION_CLIP_EVENT_ACK_WAIT_SEC"), 60.0)
CLIP_STATUS_MAX_RETRIES = parse_int_with_default(os.getenv("NUVION_CLIP_STATUS_MAX_RETRIES"), 5)
CLIP_STATUS_RETRY_BASE_SEC = parse_float(os.getenv("NUVION_CLIP_STATUS_RETRY_BASE_SEC"), 1.0)
AGENT_ERROR_MAX_RETRIES = int(os.getenv("NUVION_AGENT_ERROR_MAX_RETRIES", "3"))
AGENT_ERROR_BACKOFF_BASE_SEC = parse_float(os.getenv("NUVION_AGENT_ERROR_BACKOFF_BASE_SEC"), 1.0)
AGENT_ERROR_BACKOFF_MAX_SEC = parse_float(os.getenv("NUVION_AGENT_ERROR_BACKOFF_MAX_SEC"), 15.0)

AGENT_RETRY_DESTINATIONS = {
    "/app/device/log",
    "/app/device/state",
    "/app/device/connectivity",
    WEBRTC_UPLINK_OFFER_DEST,
    WEBRTC_UPLINK_ICE_CANDIDATE_DEST,
    WEBRTC_UPLINK_STOP_DEST,
}
WEBRTC_SIGNALING_DESTINATIONS = frozenset(
    {
        WEBRTC_UPLINK_OFFER_DEST,
        WEBRTC_UPLINK_ICE_CANDIDATE_DEST,
        WEBRTC_UPLINK_STOP_DEST,
    }
)

websocket: websockets.WebSocketClientProtocol | None = None
g_app = None
signaling_loop: asyncio.AbstractEventLoop | None = None
@dataclass(frozen=True)
class _OutboundMessage:
    destination: str
    payload: dict
    event_id: str | None = None
    signaling_token: WebRTCSignalingToken | None = None


@dataclass(frozen=True)
class _CachedPayload:
    payload: dict
    signaling_token: WebRTCSignalingToken | None = None


outbound_queue: asyncio.Queue[_OutboundMessage] | None = None
auth_token: str | None = None
auth_token_lock = threading.Lock()
agent_uplink_blocked = False
agent_uplink_block_reason = ""
agent_uplink_lock = threading.Lock()
_AgentRetryKey = tuple[str, WebRTCSignalingToken | None]


agent_retry_attempts: dict[_AgentRetryKey, int] = {}
agent_retry_lock = threading.Lock()
last_sent_payloads: dict[str, _CachedPayload] = {}
last_sent_payloads_lock = threading.Lock()
webrtc_retry_tasks: dict[_AgentRetryKey, asyncio.Task[None]] = {}
critical_event_delivery: DurableEventDelivery | None = None
critical_event_outbox_init_attempted = False
critical_event_outbox_lock = threading.Lock()
critical_event_safety_gate = CriticalEventSafetyGate()
device_state_coordinator: DeviceStateCoordinator | None = None
device_state_coordinator_lock = threading.Lock()
fleet_command_runtime: FleetCommandRuntime | None = None
fleet_command_runtime_init_attempted = False
fleet_command_runtime_lock = threading.Lock()
fleet_effect_registry = ReconcilerRegistry()
fleet_updater_client = UpdaterClient()
updater_telemetry_cache: dict[str, object] = {
    "agentUpdate": {
        "capabilityAvailable": False,
        "authenticatedHelper": False,
        "reason": "INITIALIZING",
    },
    "updaterVersion": "unknown",
    "updatePhase": "IDLE",
}
updater_telemetry_cache_updated_at = 0.0
updater_telemetry_cache_lock = threading.Lock()
updater_public_state_provider: Callable[[], Mapping[str, object]] | None = None
updater_public_state_lock = threading.Lock()
update_commit_signaling_ready_since: float | None = None
update_commit_stomp_last_send_at: float | None = None
update_commit_signaling_lock = threading.Lock()
agent_update_identity_cache: dict[str, object] | None = None
agent_update_identity_lock = threading.Lock()
FLEET_PROCESS_INSTANCE_ID = str(uuid.uuid4())
CLIP_SEGMENTS_DIR = os.path.join(CLIP_OUTPUT_DIR, "segments")
CLIP_CLIPS_DIR = os.path.join(CLIP_OUTPUT_DIR, "clips")


def _ensure_clip_dirs() -> None:
    global CLIP_OUTPUT_DIR, CLIP_SEGMENTS_DIR, CLIP_CLIPS_DIR
    if not CLIP_ENABLED:
        return

    def init_dirs(base_dir: str) -> tuple[str, str]:
        segments = os.path.join(base_dir, "segments")
        clips = os.path.join(base_dir, "clips")
        os.makedirs(segments, exist_ok=True)
        os.makedirs(clips, exist_ok=True)
        return segments, clips

    def touch_test(path: str) -> bool:
        test_path = os.path.join(path, ".write_test")
        try:
            with open(test_path, "w") as handle:
                handle.write("1")
            os.remove(test_path)
            return True
        except OSError:
            return False

    def ensure_segments_writable(path: str) -> bool:
        pattern = os.path.join(path, "segment_*.mp4")
        for seg in glob.glob(pattern):
            try:
                os.remove(seg)
            except OSError:
                return False
        return True

    try:
        CLIP_SEGMENTS_DIR, CLIP_CLIPS_DIR = init_dirs(CLIP_OUTPUT_DIR)
        if not touch_test(CLIP_SEGMENTS_DIR):
            raise PermissionError
        if not ensure_segments_writable(CLIP_SEGMENTS_DIR):
            raise PermissionError
        clip_base_mode = 0o1777 if CLIP_OUTPUT_DIR.startswith(("/tmp", "/var/tmp")) else 0o770
        try:
            os.chmod(CLIP_OUTPUT_DIR, clip_base_mode)
        except OSError:
            pass
        try:
            os.chmod(CLIP_SEGMENTS_DIR, 0o777)
            os.chmod(CLIP_CLIPS_DIR, 0o777)
        except OSError:
            pass
        return
    except PermissionError:
        fallback_dir = f"/tmp/nuvion_clips_{os.getuid()}_{int(time.time())}"
        log.warning(
            "[CLIP] No write access to %s. Falling back to %s",
            CLIP_OUTPUT_DIR,
            fallback_dir,
        )
        CLIP_OUTPUT_DIR = fallback_dir
        CLIP_SEGMENTS_DIR, CLIP_CLIPS_DIR = init_dirs(CLIP_OUTPUT_DIR)
    except Exception as exc:
        log.warning("[CLIP] Failed to initialize clip dirs: %s", exc)


_ensure_clip_dirs()

_FFMPEG_PATH: str | None = None


def resolve_ffmpeg_path() -> str | None:
    global _FFMPEG_PATH
    if _FFMPEG_PATH is not None:
        return _FFMPEG_PATH

    custom = os.getenv("NUVION_FFMPEG_PATH", "").strip()
    if custom:
        if os.path.isfile(custom) and os.access(custom, os.X_OK):
            _FFMPEG_PATH = custom
            log.info("[CLIP] Using ffmpeg from NUVION_FFMPEG_PATH=%s", custom)
            return _FFMPEG_PATH
        log.warning("[CLIP] NUVION_FFMPEG_PATH is not executable: %s", custom)

    candidate = shutil.which("ffmpeg")
    if candidate:
        _FFMPEG_PATH = candidate
        log.info("[CLIP] Using ffmpeg at %s", candidate)
        return _FFMPEG_PATH

    fallback_paths = (
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/usr/bin/ffmpeg",
        "/bin/ffmpeg",
    )
    for path in fallback_paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            _FFMPEG_PATH = path
            log.info("[CLIP] Using ffmpeg at %s", path)
            return _FFMPEG_PATH

    return None


def resolve_ffprobe_path() -> str | None:
    custom = os.getenv("NUVION_FFPROBE_PATH", "").strip()
    if custom:
        if os.path.isfile(custom) and os.access(custom, os.X_OK):
            log.info("[CLIP] Using ffprobe from NUVION_FFPROBE_PATH=%s", custom)
            return custom
        log.warning("[CLIP] NUVION_FFPROBE_PATH is not executable: %s", custom)

    candidate = shutil.which("ffprobe")
    if candidate:
        log.info("[CLIP] Using ffprobe at %s", candidate)
        return candidate

    ffmpeg_path = resolve_ffmpeg_path()
    if ffmpeg_path:
        sibling = os.path.join(os.path.dirname(ffmpeg_path), "ffprobe")
        if os.path.isfile(sibling) and os.access(sibling, os.X_OK):
            log.info("[CLIP] Using ffprobe at %s", sibling)
            return sibling

    return None


def extract_host_from_server_url(url: str) -> str:
    parsed = urlparse(url)
    return parsed.hostname or "127.0.0.1"


def get_rtp_ssrc() -> int:
    if RTP_SSRC_ENV:
        try:
            return int(RTP_SSRC_ENV)
        except ValueError:
            log.warning("[RTP] Invalid NUVION_RTP_SSRC='%s', using random ssrc.", RTP_SSRC_ENV)
    return random.randint(100000, 4294967295)


def build_x264_encoder_pipeline(element_name: str, *, bitrate_kbps: int = 1000) -> str:
    normalized_name = str(element_name or "").strip()
    if not normalized_name.replace("_", "").isalnum():
        raise ValueError("GStreamer encoder element name must be alphanumeric/underscore")
    target = int(bitrate_kbps)
    if target < 100 or target > 20_000:
        raise ValueError("encoder bitrate must be in [100, 20000] Kbps")
    h264_level = h264_level_from_profile_level_id(H264_PROFILE_LEVEL_ID_ENV)
    return (
        "videoconvert ! "
        "video/x-raw,format=I420 ! "
        f"x264enc name={normalized_name} "
        "tune=zerolatency "
        "speed-preset=faster "
        f"bitrate={target} "
        "vbv-buf-capacity=10000 "
        "key-int-max=30 "
        "bframes=0 "
        "threads=4 "
        "sliced-threads=true "
        "pass=cbr "
        "! "
        f"video/x-h264,profile={H264_PROFILE_ENV},level=(string){h264_level} ! "
    )


def build_bounded_live_queue(
    *, max_buffers: int = 2, element_name: str | None = None
) -> str:
    """Bound one live raw-video branch independently and prefer fresh frames."""

    buffers = int(max_buffers)
    if buffers <= 0 or buffers > 60:
        raise ValueError("live queue max_buffers must be in [1, 60]")
    name_property = ""
    if element_name is not None:
        normalized_name = str(element_name or "").strip()
        if not normalized_name.replace("_", "").isalnum():
            raise ValueError("live queue element_name must be alphanumeric/underscore")
        name_property = f" name={normalized_name}"
    return (
        f"queue{name_property} max-size-buffers={buffers} "
        "max-size-bytes=0 max-size-time=0 leaky=downstream"
    )


def build_uplink_pipeline(
    *,
    rtp_ssrc: int,
    clip_enabled: bool,
    clip_segment_sec: float,
    clip_max_segments: int,
    clip_segments_dir: str,
    video_bitrate_kbps: int = 1000,
) -> str:
    encoder_pipeline = build_x264_encoder_pipeline(
        "video_encoder",
        bitrate_kbps=video_bitrate_kbps,
    )
    if not clip_enabled:
        return (
            f"{encoder_pipeline}"
            f"rtph264pay name=webrtc_pay config-interval=1 pt=96 mtu=1200 ssrc={int(rtp_ssrc)} ! "
            "application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000 ! "
            "tee name=webrtc_uplink_tee allow-not-linked=true"
        )

    segment_ns = int(float(clip_segment_sec) * 1_000_000_000)
    segment_location = os.path.join(clip_segments_dir, "segment_%05d.mp4")
    clip_encoder_pipeline = build_x264_encoder_pipeline("clip_encoder")
    live_queue = build_bounded_live_queue(element_name="uplink_live_queue")
    clip_queue = build_bounded_live_queue(element_name="clip_live_queue")
    return (
        "tee name=stream_split "
        f"stream_split. ! {live_queue} ! "
        f"{encoder_pipeline}"
        "h264parse config-interval=1 ! "
        f"rtph264pay name=webrtc_pay config-interval=1 pt=96 mtu=1200 ssrc={int(rtp_ssrc)} ! "
        "application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000 ! "
        "tee name=webrtc_uplink_tee allow-not-linked=true "
        f"stream_split. ! {clip_queue} ! "
        f"{clip_encoder_pipeline}"
        "h264parse config-interval=1 ! "
        f"splitmuxsink name=clip_sink muxer=mp4mux max-size-time={segment_ns} "
        f"max-files={int(clip_max_segments)} location=\"{segment_location}\""
    )


async def login() -> str | None:
    log.info("[SIGNALING] Attempting to login as '%s'...", DEVICE_USERNAME)
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                f"{SERVER_BASE_URL}/auth/login",
                json={"username": DEVICE_USERNAME, "password": DEVICE_PASSWORD},
                timeout=10,
            ) as response:
                response.raise_for_status()
                data = await response.json()
                token = data.get("data", {}).get("accessToken")
                if token:
                    log.info("[SIGNALING] ✅ Login successful.")
                    return token
                log.error("[SIGNALING] ❌ Login OK, but 'accessToken' not found.")
        except Exception as exc:
            log.error("[SIGNALING] ❌ Login error: %s", exc)
    return None


def set_auth_token(token: str | None) -> None:
    global auth_token
    with auth_token_lock:
        auth_token = token


def get_auth_token() -> str | None:
    with auth_token_lock:
        return auth_token


def refresh_auth_token() -> str | None:
    try:
        token = asyncio.run(login())
    except RuntimeError:
        return None
    if token:
        set_auth_token(token)
    return token


def api_request(
    method: str,
    path: str,
    payload: dict | None = None,
    timeout: int = 10,
    retry: bool = True,
) -> dict | None:
    url = f"{SERVER_BASE_URL}{path}"
    token = get_auth_token() or refresh_auth_token()
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else None
    except urllib.error.HTTPError as exc:
        if exc.code == 401 and retry:
            set_auth_token(None)
            token = refresh_auth_token()
            if token:
                return api_request(method, path, payload, timeout, False)
        log.warning("[HTTP] %s %s failed: %s", method, path, exc)
    except Exception as exc:
        log.warning("[HTTP] %s %s error: %s", method, path, exc)
    return None


def request_upload_url(media_type: str = "CLIP", content_type: str | None = None) -> dict | None:
    payload = {"type": media_type, "contentType": content_type or CLIP_CONTENT_TYPE}
    response = api_request("POST", "/devices/media/upload-url", payload)
    if not response:
        return None
    return response.get("data")


def update_clip_status(object_name: str, status: str) -> bool:
    payload = {"objectName": object_name, "status": status}
    return api_request("PATCH", "/devices/media/clip-status", payload) is not None


def update_clip_status_with_retry(object_name: str, status: str) -> bool:
    attempts = max(1, CLIP_STATUS_MAX_RETRIES)
    for attempt in range(1, attempts + 1):
        if update_clip_status(object_name, status):
            return True
        if attempt < attempts:
            delay = max(0.1, CLIP_STATUS_RETRY_BASE_SEC) * (2 ** (attempt - 1))
            log.warning(
                "[CLIP] finalize retry object=%s status=%s attempt=%d/%d delay=%.1fs",
                object_name,
                status,
                attempt,
                attempts,
                delay,
            )
            time.sleep(delay)
    log.error("[CLIP] finalize failed object=%s status=%s", object_name, status)
    return False


def upload_bytes_to_url(upload_url: str, data: bytes, content_type: str) -> bool:
    try:
        server_host = urlparse(SERVER_BASE_URL).netloc
        upload_host = urlparse(upload_url).netloc
        req = urllib.request.Request(upload_url, data=data, method="PUT")
        req.add_header("Content-Type", content_type)
        if server_host and upload_host == server_host:
            token = get_auth_token()
            if token:
                req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req, timeout=60) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError as exc:
        log.warning("[UPLOAD] Failed: %s", exc)
    except Exception as exc:
        log.warning("[UPLOAD] Error: %s", exc)
    return False


def upload_file_to_url(upload_url: str, file_path: str, content_type: str) -> bool:
    try:
        with open(file_path, "rb") as file_handle:
            data = file_handle.read()
    except Exception as exc:
        log.warning("[UPLOAD] Error reading %s: %s", file_path, exc)
        return False
    return upload_bytes_to_url(upload_url, data, content_type)


def build_send_frame(destination: str, payload: dict) -> str:
    return (
        "SEND\n"
        f"destination:{destination}\n"
        "content-type:application/json\n\n"
        f"{json.dumps(payload)}\x00"
    )


def _clone_payload(payload: dict) -> dict:
    return json.loads(json.dumps(payload))


def _remember_last_payload(
    destination: str,
    payload: dict,
    signaling_token: WebRTCSignalingToken | None = None,
) -> None:
    if destination not in AGENT_RETRY_DESTINATIONS:
        return
    with last_sent_payloads_lock:
        last_sent_payloads[destination] = _CachedPayload(
            _clone_payload(payload),
            signaling_token,
        )
    if destination in WEBRTC_SIGNALING_DESTINATIONS and signaling_token is not None:
        _retire_stale_webrtc_retries(destination, signaling_token)


def _get_last_payload(destination: str) -> _CachedPayload | None:
    with last_sent_payloads_lock:
        cached = last_sent_payloads.get(destination)
        if cached is None:
            return None
        return _CachedPayload(
            _clone_payload(cached.payload),
            cached.signaling_token,
        )


def _is_signaling_token_current(token: WebRTCSignalingToken | None) -> bool:
    if token is None:
        return True
    controller = getattr(g_app, "webrtc_uplink", None) if g_app else None
    validator = getattr(controller, "is_signaling_token_current", None)
    return bool(callable(validator) and validator(token))


def _agent_retry_key(
    destination: str,
    signaling_token: WebRTCSignalingToken | None = None,
) -> _AgentRetryKey:
    return destination, signaling_token


def _cancel_webrtc_retry(key: _AgentRetryKey) -> None:
    task = webrtc_retry_tasks.pop(key, None)
    if task is not None:
        task.cancel()


def _retire_stale_webrtc_retries(
    destination: str,
    signaling_token: WebRTCSignalingToken,
) -> None:
    current_key = _agent_retry_key(destination, signaling_token)
    with agent_retry_lock:
        stale_attempt_keys = tuple(
            key
            for key in agent_retry_attempts
            if key[0] == destination and key != current_key
        )
        for key in stale_attempt_keys:
            agent_retry_attempts.pop(key, None)
    for key in tuple(webrtc_retry_tasks):
        if key[0] == destination and key != current_key:
            _cancel_webrtc_retry(key)


def _correlated_webrtc_error(
    payload: dict,
    path: str,
) -> _CachedPayload | None:
    """Return only the live generation explicitly named by an error envelope."""

    error_session_id = str(payload.get("sessionId") or "").strip()
    if not error_session_id:
        log.warning(
            "[WEBRTC-UPLINK] ignored uncorrelated signaling error path=%s",
            path,
        )
        return None
    cached = _get_last_payload(path)
    if cached is not None and cached.signaling_token is not None:
        cached_session_id = str(cached.payload.get("sessionId") or "").strip()
        if error_session_id != cached_session_id:
            log.warning(
                "[WEBRTC-UPLINK] ignored stale signaling error "
                "path=%s errorSessionId=%s currentSessionId=%s",
                path,
                error_session_id,
                cached_session_id or "<missing>",
            )
            return None
    else:
        controller = getattr(g_app, "webrtc_uplink", None) if g_app else None
        resolver = getattr(controller, "signaling_token_for_session", None)
        token = (
            resolver(
                error_session_id,
                terminal=path == WEBRTC_UPLINK_STOP_DEST,
            )
            if callable(resolver)
            else None
        )
        if not isinstance(token, WebRTCSignalingToken):
            log.warning(
                "[WEBRTC-UPLINK] ignored signaling error without current generation "
                "path=%s sessionId=%s",
                path,
                error_session_id,
            )
            return None
        cached = _CachedPayload({"sessionId": error_session_id}, token)
    if not _is_signaling_token_current(cached.signaling_token):
        log.warning(
            "[WEBRTC-UPLINK] ignored signaling error for inactive generation "
            "path=%s sessionId=%s generation=%s",
            path,
            error_session_id,
            cached.signaling_token.generation,
        )
        return None
    return cached


def _purge_webrtc_outbound_queue() -> int:
    """Drop only volatile WebRTC frames; durable event envelopes are retained."""

    pending = outbound_queue
    if pending is None:
        return 0
    retained: list[_OutboundMessage] = []
    dropped = 0
    while True:
        try:
            item = pending.get_nowait()
        except asyncio.QueueEmpty:
            break
        if item.destination in WEBRTC_SIGNALING_DESTINATIONS:
            dropped += 1
        else:
            retained.append(item)
        pending.task_done()
    for item in retained:
        pending.put_nowait(item)
    return dropped


def _reset_webrtc_signaling_transport() -> None:
    controller = getattr(g_app, "webrtc_uplink", None) if g_app else None
    if controller is not None:
        controller.on_signaling_reset()
    with last_sent_payloads_lock:
        for destination in WEBRTC_SIGNALING_DESTINATIONS:
            last_sent_payloads.pop(destination, None)
    for task in tuple(webrtc_retry_tasks.values()):
        task.cancel()
    webrtc_retry_tasks.clear()
    dropped = _purge_webrtc_outbound_queue()
    if dropped:
        log.info("[WEBRTC-UPLINK] discarded stale queued signaling frames=%d", dropped)


def _reject_current_webrtc_offer(
    payload: dict,
    *,
    reason: str,
    correlated: _CachedPayload | None = None,
) -> bool:
    """Bind a server rejection to the exact active signaling generation."""

    path = str(payload.get("path") or "")
    if path not in {WEBRTC_UPLINK_OFFER_DEST, WEBRTC_UPLINK_ICE_CANDIDATE_DEST}:
        return False
    cached = correlated or _get_last_payload(path)
    if cached is None or cached.signaling_token is None:
        return False
    rejected_session_id = str(payload.get("sessionId") or "").strip()
    cached_session_id = str(cached.payload.get("sessionId") or "").strip()
    if not rejected_session_id:
        log.warning(
            "[WEBRTC-UPLINK] uncorrelated signaling rejection deferred to watchdog"
        )
        return False
    if rejected_session_id != cached_session_id:
        log.warning(
            "[WEBRTC-UPLINK] ignored stale correlated signaling rejection "
            "rejectedSessionId=%s currentSessionId=%s",
            rejected_session_id,
            cached_session_id,
        )
        return False
    controller = getattr(g_app, "webrtc_uplink", None) if g_app else None
    reject = getattr(controller, "reject_signaling", None)
    if not callable(reject) or not reject(cached.signaling_token, reason=reason):
        return False
    with last_sent_payloads_lock:
        current = last_sent_payloads.get(path)
        if current is not None and current.signaling_token == cached.signaling_token:
            last_sent_payloads.pop(path, None)
    retry_key = _agent_retry_key(path, cached.signaling_token)
    _cancel_webrtc_retry(retry_key)
    _reset_agent_retry_attempt(retry_key)
    return True


def _set_agent_uplink_blocked(blocked: bool, reason: str = "") -> None:
    global agent_uplink_blocked, agent_uplink_block_reason
    with agent_uplink_lock:
        agent_uplink_blocked = blocked
        agent_uplink_block_reason = reason


def _is_agent_uplink_blocked(destination: str) -> bool:
    if destination not in AGENT_RETRY_DESTINATIONS:
        return False
    with agent_uplink_lock:
        if not agent_uplink_blocked:
            return False
        log.error(
            "[STOMP] outbound blocked by non-retryable agent error. destination=%s reason=%s",
            destination,
            agent_uplink_block_reason,
        )
        return True


def _reset_agent_retry_attempt(key: _AgentRetryKey) -> None:
    with agent_retry_lock:
        agent_retry_attempts.pop(key, None)


def _reset_agent_retry_attempts_for_destination(destination: str) -> None:
    with agent_retry_lock:
        for key in tuple(agent_retry_attempts):
            if key[0] == destination:
                agent_retry_attempts.pop(key, None)


def _next_agent_retry_attempt(key: _AgentRetryKey) -> int:
    with agent_retry_lock:
        attempt = agent_retry_attempts.get(key, 0) + 1
        agent_retry_attempts[key] = attempt
        return attempt


def _reset_agent_ws_state() -> None:
    _set_update_commit_signaling_ready(False)
    _set_agent_uplink_blocked(False, "")
    with agent_retry_lock:
        agent_retry_attempts.clear()
    if critical_event_delivery is not None:
        critical_event_delivery.reset_for_reconnect()
    _reset_webrtc_signaling_transport()


def _set_update_commit_signaling_ready(ready: bool) -> None:
    global update_commit_signaling_ready_since, update_commit_stomp_last_send_at
    with update_commit_signaling_lock:
        update_commit_signaling_ready_since = time.monotonic() if ready else None
        update_commit_stomp_last_send_at = None


def _mark_update_commit_stomp_send() -> None:
    global update_commit_stomp_last_send_at
    with update_commit_signaling_lock:
        if update_commit_signaling_ready_since is not None:
            update_commit_stomp_last_send_at = time.monotonic()


def build_update_commit_readiness() -> dict[str, object]:
    with update_commit_signaling_lock:
        signaling_ready_since = update_commit_signaling_ready_since
        stomp_last_send_at = update_commit_stomp_last_send_at
    controller = getattr(g_app, "webrtc_uplink", None) if g_app else None
    user_data = getattr(g_app, "user_data", None) if g_app else None
    health_provider = getattr(controller, "runtime_health_snapshot", None)
    with agent_uplink_lock:
        stomp_blocked = agent_uplink_blocked
    return evaluate_update_commit_readiness(
        now_monotonic=time.monotonic(),
        signaling_ready_since=signaling_ready_since,
        stomp_last_send_at=stomp_last_send_at,
        min_stable_seconds=UPDATE_COMMIT_STABLE_SEC,
        max_evidence_age_seconds=UPDATE_COMMIT_MAX_EVIDENCE_AGE_SEC,
        pipeline_running=(
            g_app is not None
            and getattr(g_app, "pipeline", None) is not None
            and bool(getattr(user_data, "running", False))
        ),
        pipeline_last_frame_at=getattr(user_data, "last_frame_monotonic", None),
        webrtc_health=(
            health_provider() if callable(health_provider) else {}
        ),
        stomp_blocked=stomp_blocked,
        event_outbox_health=build_event_outbox_runtime_health(),
        command_outbox_health=build_command_observation_runtime_health(),
    )


def _agent_update_platform_identity() -> dict[str, object]:
    global agent_update_identity_cache
    with agent_update_identity_lock:
        if agent_update_identity_cache is None:
            identity = resolve_platform_identity()
            if identity.identity_status not in {
                IDENTITY_STATUS_VERIFIED,
                IDENTITY_STATUS_DEV,
            }:
                raise RuntimeError("platform identity is not verified for Agent update")
            agent_update_identity_cache = {
                "productModel": identity.product_model,
                "platformProfile": identity.platform_profile,
                "hardwareRevision": identity.hardware_revision,
                "architecture": identity.architecture,
            }
        return dict(agent_update_identity_cache)


def request_agent_update_health_attestation(
    command: VerifiedFleetCommand,
    update: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> Mapping[str, Any]:
    request = build_health_attestation_request(
        device_id=command.device_id,
        command_id=command.command_id,
        expected_bom_digest=str(command.payload.get("bomDigest") or ""),
        expected_component_sha=str(update.get("componentSha") or ""),
        expected_release_sequence=update.get("releaseSequence"),
        gate=gate,
        identity=_agent_update_platform_identity(),
    )
    return request_health_attestation(
        request,
        transport=lambda payload: api_request(
            "POST",
            "/devices/me/agent-update-health-attestations",
            payload,
            timeout=10,
        ),
    )


def initialize_durable_event_outbox() -> DurableEventDelivery | None:
    global critical_event_delivery, critical_event_outbox_init_attempted
    with critical_event_outbox_lock:
        if critical_event_outbox_init_attempted:
            return critical_event_delivery
        critical_event_outbox_init_attempted = True
        try:
            outbox = DurableEventOutbox(
                resolve_default_outbox_path(),
                max_rows=EVENT_OUTBOX_MAX_ROWS,
                max_bytes=EVENT_OUTBOX_MAX_BYTES,
                max_dead_letters=EVENT_DLQ_MAX_ROWS,
                max_dead_letter_bytes=EVENT_DLQ_MAX_BYTES,
                max_critical_safety_bytes=EVENT_CRITICAL_SAFETY_MAX_BYTES,
                max_age_seconds=EVENT_OUTBOX_MAX_AGE_SECONDS,
            )
            critical_event_delivery = DurableEventDelivery(outbox)
            retained = outbox.critical_safety_event()
            if retained is not None:
                critical_event_safety_gate.restore_retained(
                    PendingCriticalEvent.create(
                        event_type=retained.event_type,
                        destination=retained.destination,
                        payload=retained.payload,
                        event_id=retained.event_id,
                        occurred_at=retained.occurred_at,
                    ),
                    retained.last_error or "restored after process restart",
                )
            log.info(
                "[OUTBOX] ready path=%s retained=%d blocked=%d",
                outbox.path,
                outbox.count(),
                outbox.blocked_count(),
            )
        except Exception as exc:
            critical_event_delivery = None
            log.error("[OUTBOX] unavailable; critical events fail closed: %s", exc)
        return critical_event_delivery


def _try_enqueue_outbound(
    destination: str,
    payload: dict,
    event_id: str | None = None,
    *,
    signaling_token: WebRTCSignalingToken | None = None,
    remember: bool = False,
) -> bool:
    if outbound_queue is None or signaling_loop is None:
        return False
    completed = threading.Event()
    cancelled = threading.Event()
    enqueued: list[bool] = []

    def _enqueue() -> None:
        try:
            if cancelled.is_set() or not _is_signaling_token_current(signaling_token):
                enqueued.append(False)
                return
            outbound_queue.put_nowait(
                _OutboundMessage(
                    destination,
                    _clone_payload(payload),
                    event_id,
                    signaling_token,
                )
            )
            if remember:
                _remember_last_payload(destination, payload, signaling_token)
            enqueued.append(True)
        except asyncio.QueueFull:
            enqueued.append(False)
        finally:
            completed.set()

    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None
    if current_loop is signaling_loop:
        _enqueue()
    else:
        try:
            signaling_loop.call_soon_threadsafe(_enqueue)
        except RuntimeError:
            return False
        if not completed.wait(timeout=0.5):
            cancelled.set()
            return False
    return bool(enqueued and enqueued[0])


def enqueue_stomp_message(
    destination: str,
    payload: dict,
    remember: bool = True,
    signaling_token: WebRTCSignalingToken | None = None,
) -> bool:
    if _is_agent_uplink_blocked(destination):
        return False
    if destination in WEBRTC_SIGNALING_DESTINATIONS and signaling_token is None:
        log.error("[STOMP] unscoped WebRTC signaling frame rejected=%s", destination)
        return False
    if signaling_token is not None and destination not in WEBRTC_SIGNALING_DESTINATIONS:
        log.error("[STOMP] signaling token used for non-WebRTC destination=%s", destination)
        return False
    if not _try_enqueue_outbound(
        destination,
        payload,
        signaling_token=signaling_token,
        remember=remember,
    ):
        log.warning("[STOMP] outbound unavailable or full for %s", destination)
        return False
    return True


def initialize_fleet_command_runtime() -> FleetCommandRuntime | None:
    global fleet_command_runtime, fleet_command_runtime_init_attempted
    with fleet_command_runtime_lock:
        if fleet_command_runtime_init_attempted:
            return fleet_command_runtime
        fleet_command_runtime_init_attempted = True
        try:
            if fleet_effect_registry.get("AGENT_UPDATE") is None:
                fleet_effect_registry.register(
                    AgentUpdateReconciler(
                        fleet_updater_client,
                        readiness_provider=_cached_agent_update_status,
                        commit_readiness_provider=build_update_commit_readiness,
                        health_attestation_provider=(
                            request_agent_update_health_attestation
                        ),
                    )
                )
            fleet_command_runtime = build_fleet_command_runtime_from_env(
                base_url=SERVER_BASE_URL,
                access_token_provider=lambda: get_auth_token() or "",
                ack_sender=lambda destination, payload: enqueue_stomp_message(
                    destination,
                    payload,
                    remember=False,
                ),
                reconciler_registry=fleet_effect_registry,
                process_instance_id=FLEET_PROCESS_INSTANCE_ID,
                restart_requester=(
                    (lambda: bool(g_app and g_app.request_supervisor_restart()))
                    if systemd_restart_enabled(os.environ)
                    else None
                ),
            )
        except (FleetCommandRuntimeError, OSError, sqlite3.Error, ValueError) as exc:
            log.error("[FLEET-COMMAND] disabled by invalid fail-closed configuration: %s", exc)
            fleet_command_runtime = None
        if fleet_command_runtime is None:
            log.info("[FLEET-COMMAND] runtime disabled")
        else:
            log.info(
                "[FLEET-COMMAND] durable inbox ready path=%s lastSequence=%d",
                fleet_command_runtime.inbox.path,
                fleet_command_runtime.inbox.last_sequence(),
            )
        return fleet_command_runtime


def _send_durable_event(event: DurableEvent) -> bool:
    return _try_enqueue_outbound(event.destination, event.payload, event.event_id)


def persist_durable_event(
    event_type: str,
    destination: str,
    payload: dict,
    event_id: str | None = None,
    occurred_at: str | None = None,
    compaction_key: str | None = None,
) -> DurableEvent | None:
    delivery = initialize_durable_event_outbox()
    if delivery is None:
        return None
    normalized_event_id = event_id or str(uuid.uuid4())
    normalized_occurred_at = occurred_at or utc_now_iso()
    try:
        event = delivery.publish(
            event_type=event_type,
            destination=destination,
            payload=payload,
            event_id=normalized_event_id,
            occurred_at=normalized_occurred_at,
            sender=_send_durable_event,
            compaction_key=compaction_key,
        )
        if event.dead_lettered:
            log.error(
                "[OUTBOX] duplicate event is already in DLQ type=%s eventId=%s",
                event_type,
                normalized_event_id,
            )
        elif event.dropped:
            log.warning(
                "[OUTBOX] non-critical event dropped type=%s eventId=%s",
                event_type,
                normalized_event_id,
            )
        return event
    except DurableEventCapacityError:
        if (
            DEFAULT_DELIVERY_CLASS_BY_EVENT_TYPE.get(str(event_type).strip().upper())
            == DELIVERY_CLASS_CRITICAL
        ):
            raise
        log.error(
            "[OUTBOX] non-critical capacity failure type=%s eventId=%s",
            event_type,
            normalized_event_id,
        )
        return None
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        sqlite3.Error,
    ) as exc:
        log.error(
            "[OUTBOX] persist failed type=%s eventId=%s: %s",
            event_type,
            normalized_event_id,
            exc,
        )
        return None


def persist_critical_event(
    event_type: str,
    destination: str,
    payload: dict,
    event_id: str,
    occurred_at: str,
) -> DurableEvent | None:
    retained = PendingCriticalEvent.create(
        event_type=event_type,
        destination=destination,
        payload=payload,
        event_id=event_id,
        occurred_at=occurred_at,
    )
    try:
        return critical_event_safety_gate.persist(
            retained,
            _persist_pending_critical_once,
            _retain_failed_critical_event,
            _clear_retained_critical_event,
        )
    except CriticalEventBackpressureError:
        get_device_state_coordinator().set_runtime_status(RUNTIME_STATUS_ERROR)
        raise


def _persist_pending_critical_once(event: PendingCriticalEvent) -> DurableEvent:
    delivery = initialize_durable_event_outbox()
    if delivery is None:
        raise DurableEventCapacityError("critical outbox is unavailable")
    try:
        persisted = delivery.publish(
            event_type=event.event_type,
            destination=event.destination,
            payload=event.payload,
            event_id=event.event_id,
            occurred_at=event.occurred_at,
            sender=_send_durable_event,
        )
    except DurableEventCapacityError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError, sqlite3.Error) as exc:
        raise DurableEventCapacityError(
            f"critical outbox persistence failed: {exc}"
        ) from exc
    if persisted.dead_lettered or persisted.dropped:
        raise DurableEventCapacityError(
            "critical event did not enter the pending durable outbox"
        )
    return persisted


def _retain_failed_critical_event(
    event: PendingCriticalEvent,
    reason: str,
) -> None:
    delivery = initialize_durable_event_outbox()
    if delivery is None:
        raise DurableEventCapacityError("critical safety slot is unavailable")
    delivery.outbox.retain_critical_safety_event(
        event_type=event.event_type,
        destination=event.destination,
        payload=event.payload,
        event_id=event.event_id,
        occurred_at=event.occurred_at,
        last_error=reason,
    )


def _clear_retained_critical_event(event_id: str) -> bool:
    if critical_event_delivery is None:
        return False
    return critical_event_delivery.outbox.clear_critical_safety_event(event_id)


def retry_retained_critical_event() -> bool:
    return critical_event_safety_gate.retry_retained(
        _persist_pending_critical_once,
        _clear_retained_critical_event,
    )


def build_event_outbox_runtime_health() -> dict:
    if critical_event_delivery is None:
        health = {
            "pendingRows": 0,
            "pendingBytes": 0,
            "oldestCriticalAgeSeconds": None,
            "dlqRows": 0,
            "dlqBytes": 0,
            "blockedRows": 0,
            "capacityState": (
                "UNAVAILABLE" if critical_event_outbox_init_attempted else "INITIALIZING"
            ),
        }
    else:
        try:
            health = critical_event_delivery.outbox.health_snapshot().to_telemetry()
        except (OSError, RuntimeError, sqlite3.Error) as exc:
            health = {
                "pendingRows": 0,
                "pendingBytes": 0,
                "oldestCriticalAgeSeconds": None,
                "dlqRows": 0,
                "dlqBytes": 0,
                "blockedRows": 0,
                "capacityState": "UNAVAILABLE",
                "healthError": str(exc)[:500],
            }
    safety = critical_event_safety_gate.health_overlay()
    health.update(safety)
    health["unsavedCriticalEvents"] = max(
        int(health.get("criticalSafetyRows") or 0),
        int(safety["unsavedCriticalEvents"]),
    )
    if safety["safetyStop"]:
        health["capacityState"] = "OPERATOR_STOP"
    elif safety["unsavedCriticalEvents"]:
        health["capacityState"] = "BACKPRESSURE"
    return health


def register_updater_public_state_provider(
    provider: Callable[[], Mapping[str, object]] | None,
) -> None:
    """Integration hook for the authenticated root updater public state."""

    if provider is not None and not callable(provider):
        raise TypeError("updater public state provider must be callable")
    global updater_public_state_provider
    with updater_public_state_lock:
        updater_public_state_provider = provider


def _stale_updater_telemetry(
    trusted: Mapping[str, object] | None = None,
) -> dict[str, object]:
    stale: dict[str, object] = {
        "agentUpdate": {
            "capabilityAvailable": False,
            "authenticatedHelper": False,
            "reason": "STALE_UPDATER_STATUS" if trusted else "UPDATER_UNAVAILABLE",
        },
        "updaterVersion": "unknown",
        "updatePhase": "IDLE",
    }
    if trusted is not None:
        for key in ("updatePhase", "updateEvidence"):
            if key in trusted:
                stale[key] = trusted[key]
    return stale


def get_cached_updater_runtime_telemetry() -> dict[str, object]:
    with updater_telemetry_cache_lock:
        cached = dict(updater_telemetry_cache)
        updated_at = updater_telemetry_cache_updated_at
    ttl = max(1.0, min(UPDATER_TELEMETRY_TTL_SEC, 300.0))
    if updated_at <= 0.0 or time.monotonic() - updated_at > ttl:
        return _stale_updater_telemetry(cached if updated_at > 0.0 else None)
    return cached


def _cached_agent_update_status() -> Mapping[str, object]:
    status = get_cached_updater_runtime_telemetry().get("agentUpdate")
    return dict(status) if isinstance(status, Mapping) else {}


async def refresh_updater_runtime_telemetry() -> dict[str, object]:
    global updater_telemetry_cache_updated_at
    try:
        refreshed = await asyncio.to_thread(
            build_updater_capability_telemetry,
            fleet_updater_client,
        )
    except Exception as exc:  # noqa: BLE001 - cache remains fail closed.
        log.warning("[UPDATER] status refresh failed: %s", str(exc)[:200])
        refreshed = _stale_updater_telemetry(
            get_cached_updater_runtime_telemetry()
        )
    with updater_telemetry_cache_lock:
        updater_telemetry_cache.clear()
        updater_telemetry_cache.update(refreshed)
        updater_telemetry_cache_updated_at = time.monotonic()
    return dict(refreshed)


async def updater_telemetry_refresh_sender() -> None:
    interval = max(1.0, min(UPDATER_TELEMETRY_REFRESH_SEC, 60.0))
    while True:
        await asyncio.sleep(interval)
        await refresh_updater_runtime_telemetry()


def build_command_observation_runtime_health() -> dict:
    runtime = fleet_command_runtime
    if runtime is None or runtime.observation_outbox is None:
        return {
            "pendingRows": 0,
            "pendingBytes": 0,
            "reservedRows": 0,
            "reservedBytes": 0,
            "dlqRows": 0,
            "dlqBytes": 0,
            "dlqBlockedRows": 0,
            "retentionPressure": False,
            "capacityState": (
                "UNAVAILABLE"
                if fleet_command_runtime_init_attempted
                else "INITIALIZING"
            ),
        }
    try:
        return runtime.observation_outbox.health_snapshot().to_telemetry()
    except (OSError, RuntimeError, sqlite3.Error) as exc:
        return {
            "pendingRows": 0,
            "pendingBytes": 0,
            "reservedRows": 0,
            "reservedBytes": 0,
            "dlqRows": 0,
            "dlqBytes": 0,
            "dlqBlockedRows": 0,
            "retentionPressure": True,
            "capacityState": "UNAVAILABLE",
            "healthError": str(exc)[:500],
        }


def build_dynamic_runtime_telemetry(
    base_capabilities: set[str] | frozenset[str] = frozenset(),
) -> dict:
    functional_health = (
        "FUNCTIONAL_HEALTHY"
        if g_app is not None
        and getattr(g_app, "pipeline", None) is not None
        and bool(getattr(getattr(g_app, "user_data", None), "running", False))
        else "FUNCTIONAL_UNHEALTHY"
    )
    updater_telemetry = get_cached_updater_runtime_telemetry()
    public_state: dict[str, object] = {"functionalHealth": functional_health}
    for key in ("updatePhase", "updateEvidence"):
        if key in updater_telemetry:
            public_state[key] = updater_telemetry[key]
    with updater_public_state_lock:
        provider = updater_public_state_provider
    if provider is not None:
        try:
            public_state.update(dict(provider()))
        except Exception as exc:  # noqa: BLE001 - telemetry must stay available.
            log.error("[UPDATER] public state unavailable: %s", exc)
            public_state["functionalHealth"] = "FUNCTIONAL_UNHEALTHY"
    if functional_health == "FUNCTIONAL_UNHEALTHY":
        public_state["functionalHealth"] = functional_health
    try:
        merged = merge_runtime_public_state({}, public_state)
    except (TypeError, ValueError) as exc:
        log.error("[UPDATER] invalid public state rejected: %s", exc)
        merged = {"functionalHealth": "FUNCTIONAL_UNHEALTHY"}
    merged["eventOutbox"] = build_event_outbox_runtime_health()
    merged["commandObservationOutbox"] = build_command_observation_runtime_health()
    merged["agentUpdate"] = updater_telemetry["agentUpdate"]
    merged["updaterVersion"] = updater_telemetry["updaterVersion"]
    merged["capabilities"] = sorted(
        set(base_capabilities) | set(fleet_effect_registry.capabilities)
    )
    return merged


def persist_state_event(payload: dict) -> bool:
    event = persist_durable_event(
        EVENT_TYPE_DEVICE_STATE,
        "/app/device/state",
        payload,
        compaction_key="device-state",
    )
    return event is not None and not event.dropped and not event.dead_lettered


def persist_connectivity_event(payload: dict) -> bool:
    event = persist_durable_event(
        EVENT_TYPE_CONNECTIVITY,
        "/app/device/connectivity",
        payload,
        compaction_key="connectivity",
    )
    return event is not None and not event.dropped and not event.dead_lettered


def persist_metric_event(payload: dict) -> bool:
    """Durably stage one best-effort metric; transport completion is its terminal ACK."""
    event = persist_durable_event(EVENT_TYPE_METRIC, "/app/device/metric", payload)
    return event is not None and not event.dropped and not event.dead_lettered


def get_device_state_coordinator() -> DeviceStateCoordinator:
    global device_state_coordinator
    with device_state_coordinator_lock:
        if device_state_coordinator is None:
            telemetry = build_runtime_telemetry(
                effect_capabilities=fleet_effect_registry.capabilities,
            )
            base_capabilities = frozenset(telemetry.get("capabilities", ())) - set(
                COMMAND_CAPABILITY_BY_TYPE.values()
            )
            device_state_coordinator = DeviceStateCoordinator(
                send_message=persist_state_event,
                line_id=LINE_ID,
                process_id=PROCESS_ID,
                telemetry=telemetry,
                runtime_telemetry_provider=lambda: build_dynamic_runtime_telemetry(
                    base_capabilities
                ),
            )
        return device_state_coordinator


async def outbound_sender(ws: websockets.WebSocketClientProtocol):
    if outbound_queue is None:
        return
    while True:
        message = await outbound_queue.get()
        destination = message.destination
        payload = message.payload
        event_id = message.event_id
        if not _is_signaling_token_current(message.signaling_token):
            outbound_queue.task_done()
            log.info(
                "[WEBRTC-UPLINK] dropped stale signaling frame before send destination=%s",
                destination,
            )
            continue
        if (
            event_id
            and critical_event_delivery is not None
            and not critical_event_delivery.outbox.is_pending(event_id)
        ):
            critical_event_delivery.release(event_id)
            outbound_queue.task_done()
            continue
        frame_str = build_send_frame(destination, payload)
        try:
            await ws.send(json.dumps([frame_str]))
            _mark_update_commit_stomp_send()
            if event_id and critical_event_delivery is not None:
                critical_event_delivery.mark_sent(event_id)
        except Exception as exc:
            if event_id and critical_event_delivery is not None:
                critical_event_delivery.release(event_id)
            log.warning("[STOMP] send failed: %s", exc)
            await ws.close()
            raise
        finally:
            outbound_queue.task_done()


async def durable_event_replay_sender() -> None:
    interval = max(1.0, EVENT_REPLAY_INTERVAL_SEC)
    while True:
        if critical_event_safety_gate.pending_event() is not None:
            if retry_retained_critical_event():
                log.error(
                    "[OUTBOX] retained critical event is now durable; "
                    "operator stop remains until explicit recovery"
                )
        if (
            critical_event_delivery is not None
            and critical_event_safety_gate.replay_allowed()
        ):
            critical_event_delivery.replay(
                _send_durable_event,
                limit=OUTBOUND_QUEUE_MAX,
                retry_after_seconds=interval,
            )
        await asyncio.sleep(interval)


async def handle_event_ack(body: str) -> None:
    if critical_event_delivery is None:
        return
    ack, removed = critical_event_delivery.acknowledge_body(body)
    if ack is None:
        log.warning("[OUTBOX] invalid ACK: %s", body)
        return
    if removed:
        if ack.successful:
            log.info("[OUTBOX] ACK eventId=%s type=%s status=%s", ack.event_id, ack.event_type, ack.status)
        else:
            log.error(
                "[OUTBOX] permanent rejection moved to DLQ eventId=%s type=%s code=%s reason=%s",
                ack.event_id,
                ack.event_type,
                ack.code,
                ack.reason,
            )


async def stomp_heartbeat_sender(
    ws: websockets.WebSocketClientProtocol,
    send_interval_ms: int | None,
):
    if not send_interval_ms or send_interval_ms <= 0:
        log.info("[SIGNALING] STOMP heartbeat sender disabled.")
        return

    interval_sec = max(send_interval_ms / 1000.0, 1.0)
    log.info("[SIGNALING] STOMP heartbeat sender enabled. interval_ms=%s", send_interval_ms)

    while True:
        await asyncio.sleep(interval_sec)
        try:
            await ws.send(json.dumps(["\n"]))
            _mark_update_commit_stomp_send()
        except Exception as exc:
            log.warning("[SIGNALING] STOMP heartbeat send failed: %s", exc)
            raise


async def device_state_heartbeat_sender():
    interval = max(1.0, DEVICE_STATE_INTERVAL_SEC)
    coordinator = get_device_state_coordinator()
    while True:
        coordinator.emit_heartbeat()
        await asyncio.sleep(interval)


async def device_connectivity_sender(reporter: ConnectivityReporter):
    interval = max(1.0, CONNECTIVITY_INTERVAL_SEC)
    coordinator = get_device_state_coordinator()
    while True:
        sample = reporter.collect_sample_payload()
        runtime = fleet_command_runtime
        if sample and runtime is not None:
            try:
                await runtime.observe_connectivity(sample)
            except Exception as exc:  # noqa: BLE001 - next sample keeps the loop alive.
                log.error(
                    "[STREAM-POLICY] connectivity observation failed type=%s detail=%s",
                    type(exc).__name__,
                    str(exc)[:500],
                )
        payload = reporter.build_transition_payload(sample)
        if payload:
            coordinator.set_connectivity_status(str(payload.get("quality") or CONNECTIVITY_QUALITY_GOOD))
        if payload and persist_connectivity_event(payload):
            log.info(
                "[CONNECTIVITY] sent quality=%s reason=%s rssi=%s loss=%s rtt=%s",
                payload.get("quality"),
                payload.get("reason"),
                payload.get("rssiDbm"),
                payload.get("packetLossPct"),
                payload.get("rttMs"),
            )
        await asyncio.sleep(interval)


async def _enqueue_retry_after_delay(
    destination: str,
    payload: dict,
    delay_sec: float,
    attempt: int,
    max_attempts: int,
    code: str,
    signaling_token: WebRTCSignalingToken | None = None,
) -> None:
    await asyncio.sleep(delay_sec)
    enqueued = enqueue_stomp_message(
        destination,
        payload,
        remember=False,
        signaling_token=signaling_token,
    )
    if enqueued:
        log.warning(
            "[AGENT-ERROR] retry sent destination=%s code=%s attempt=%d/%d",
            destination,
            code,
            attempt,
            max_attempts,
        )


async def handle_agent_error(body: str) -> None:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        log.warning("[AGENT-ERROR] invalid payload: %s", body)
        return

    code = str(payload.get("code") or "UNKNOWN")
    message = str(payload.get("message") or "")
    detail = str(payload.get("detail") or "")
    path = str(payload.get("path") or "")
    retryable = bool(payload.get("retryable"))
    status = payload.get("status")

    status_int: int | None = None
    if isinstance(status, int):
        status_int = status
    elif isinstance(status, str) and status.isdigit():
        status_int = int(status)

    rejection = parse_permanent_event_rejection(payload)
    if rejection is not None and critical_event_delivery is not None:
        quarantined = critical_event_delivery.reject_event(
            rejection.event_id,
            rejection.event_type,
            reason=rejection.reason,
            rejection_code=rejection.rejection_code,
            source="agent.error",
        )
        if quarantined:
            log.error(
                "[OUTBOX] agent error moved event to DLQ eventId=%s type=%s code=%s",
                rejection.event_id,
                rejection.event_type,
                rejection.rejection_code,
            )
        return

    if rejection is None and is_uncorrelated_permanent_event_rejection(payload):
        reason = (
            f"uncorrelated permanent event rejection path={path} code={code}: "
            f"{detail or message}"
        )
        critical_event_safety_gate.enter_protocol_stop(reason)
        get_device_state_coordinator().set_runtime_status(RUNTIME_STATUS_ERROR)
        log.error("[OUTBOX] %s; critical replay stopped for operator action", reason)
        return

    if not retryable and status_int in (401, 403):
        # Authentication is transport-wide, unlike a WebRTC negotiation
        # failure. Preserve the existing fail-closed behavior even when a
        # candidate/stop frame was intentionally not cached for retry.
        _reset_agent_retry_attempts_for_destination(path)
        if path in WEBRTC_SIGNALING_DESTINATIONS:
            _reset_webrtc_signaling_transport()
        reason = f"{code} {message}".strip()
        _set_agent_uplink_blocked(True, reason)
        log.error(
            "[AGENT-ERROR][auth] uplink blocked. code=%s status=%s path=%s message=%s detail=%s",
            code,
            status_int,
            path,
            message,
            detail,
        )
        return

    cached: _CachedPayload | None = None
    retry_key = _agent_retry_key(path)
    if path in WEBRTC_SIGNALING_DESTINATIONS:
        # Server errors are delivered asynchronously and can outlive the
        # generation that emitted the frame. A path-only or stale error must
        # never consume the retry budget or tear down a newer session; the
        # controller's bounded watchdog owns uncorrelated failures.
        cached = _correlated_webrtc_error(payload, path)
        if cached is None or cached.signaling_token is None:
            return
        retry_key = _agent_retry_key(path, cached.signaling_token)
        if path != WEBRTC_UPLINK_OFFER_DEST:
            # ICE candidates are intentionally volatile and STOP is already a
            # terminal local transition. Neither can be replayed safely. An
            # exact active candidate rejection tears down its branch; a STOP
            # rejection is logged after the branch has already been released.
            _reset_agent_retry_attempt(retry_key)
            rejected = _reject_current_webrtc_offer(
                payload,
                reason=f"signaling frame rejected: {code} status={status_int}",
                correlated=cached,
            )
            log.warning(
                "[WEBRTC-UPLINK] terminal signaling rejection path=%s "
                "sessionId=%s disposed=%s code=%s status=%s",
                path,
                str(payload.get("sessionId") or ""),
                rejected,
                code,
                status_int,
            )
            return

    if retryable:
        if path not in AGENT_RETRY_DESTINATIONS:
            log.warning(
                "[AGENT-ERROR][retryable] unsupported path. code=%s path=%s message=%s",
                code,
                path,
                message,
            )
            return

        if cached is None:
            cached = _get_last_payload(path)
        if cached is None:
            log.warning(
                "[AGENT-ERROR][retryable] no cached payload. code=%s path=%s message=%s",
                code,
                path,
                message,
            )
            return

        if path == WEBRTC_UPLINK_OFFER_DEST:
            pending_retry = webrtc_retry_tasks.get(retry_key)
            if pending_retry is not None and not pending_retry.done():
                log.warning(
                    "[AGENT-ERROR][retryable] coalesced duplicate WebRTC error "
                    "path=%s sessionId=%s",
                    path,
                    cached.signaling_token.session_id
                    if cached.signaling_token is not None
                    else "<missing>",
                )
                return
            if pending_retry is not None:
                webrtc_retry_tasks.pop(retry_key, None)

        attempt = _next_agent_retry_attempt(retry_key)
        if attempt > AGENT_ERROR_MAX_RETRIES:
            log.error(
                "[AGENT-ERROR] retry exhausted. code=%s path=%s max=%d detail=%s",
                code,
                path,
                AGENT_ERROR_MAX_RETRIES,
                detail,
            )
            _reject_current_webrtc_offer(
                payload,
                reason=f"offer retry exhausted: {code}",
                correlated=cached,
            )
            return

        delay_sec = min(AGENT_ERROR_BACKOFF_BASE_SEC * (2 ** (attempt - 1)), AGENT_ERROR_BACKOFF_MAX_SEC)
        log.warning(
            "[AGENT-ERROR][retryable] code=%s status=%s path=%s attempt=%d/%d delay=%.1fs message=%s detail=%s",
            code,
            status_int,
            path,
            attempt,
            AGENT_ERROR_MAX_RETRIES,
            delay_sec,
            message,
            detail,
        )
        retry_task = asyncio.create_task(
            _enqueue_retry_after_delay(
                destination=path,
                payload=cached.payload,
                delay_sec=delay_sec,
                attempt=attempt,
                max_attempts=AGENT_ERROR_MAX_RETRIES,
                code=code,
                signaling_token=cached.signaling_token,
            )
        )
        if path in WEBRTC_SIGNALING_DESTINATIONS:
            previous = webrtc_retry_tasks.get(retry_key)
            if previous is not None and previous is not retry_task:
                previous.cancel()
            webrtc_retry_tasks[retry_key] = retry_task

            def _remove_completed_webrtc_retry(
                completed: asyncio.Task[None],
                *,
                key: _AgentRetryKey = retry_key,
            ) -> None:
                if webrtc_retry_tasks.get(key) is completed:
                    webrtc_retry_tasks.pop(key, None)

            retry_task.add_done_callback(_remove_completed_webrtc_retry)
        return

    _reset_agent_retry_attempt(retry_key)
    offer_rejected = _reject_current_webrtc_offer(
        payload,
        reason=f"non-retryable server rejection: {code} status={status_int}",
        correlated=cached,
    )
    if offer_rejected:
        log.warning(
            "[WEBRTC-UPLINK] disposed locally rejected offer path=%s code=%s status=%s",
            path,
            code,
            status_int,
        )
    if status_int is not None and 400 <= status_int < 500:
        log.warning(
            "[AGENT-ERROR][client] dropped. code=%s status=%s path=%s message=%s detail=%s",
            code,
            status_int,
            path,
            message,
            detail,
        )
        return

    log.error(
        "[AGENT-ERROR][server] non-retryable. code=%s status=%s path=%s message=%s detail=%s",
        code,
        status_int,
        path,
        message,
        detail,
    )


async def handle_webrtc_uplink_command(data: dict) -> bool:
    global g_app

    if not g_app or not getattr(g_app, "webrtc_uplink", None):
        return False

    command_type = str(data.get("type") or "").strip()
    if command_type == WEBRTC_UPLINK_START:
        g_app.webrtc_uplink.start(data)
        return True
    if command_type == WEBRTC_UPLINK_ANSWER:
        g_app.webrtc_uplink.apply_answer(data)
        return True
    if command_type == WEBRTC_UPLINK_ICE_CANDIDATE:
        g_app.webrtc_uplink.add_remote_ice_candidate(data)
        return True
    if command_type == WEBRTC_UPLINK_STATE:
        g_app.webrtc_uplink.handle_remote_state(data)
        return True
    return False


async def handle_command_message(body: str):
    data = parse_command_payload(body)
    if not data:
        return

    await handle_webrtc_uplink_command(data)


async def handle_fleet_command_wakeup(body: str) -> None:
    runtime = initialize_fleet_command_runtime()
    if runtime is None:
        log.warning("[FLEET-COMMAND] ignored wake-up because runtime is disabled")
        return
    try:
        sent = await runtime.on_wakeup(body)
        log.info("[FLEET-COMMAND] wake-up reconciled lifecycleAcks=%d", sent)
    except Exception as exc:  # noqa: BLE001 - command remains durable for the next pull.
        log.error(
            "[FLEET-COMMAND] wake-up reconciliation failed type=%s detail=%s",
            type(exc).__name__,
            str(exc)[:500],
        )


async def fleet_command_poll_sender(runtime: FleetCommandRuntime) -> None:
    interval = max(5.0, FLEET_COMMAND_POLL_INTERVAL_SEC)
    while True:
        await asyncio.sleep(interval)
        try:
            reconciled = await runtime.poll()
            if reconciled:
                log.info(
                    "[FLEET-COMMAND] periodic reconciliation count=%d",
                    reconciled,
                )
        except Exception as exc:  # noqa: BLE001 - durable journal retries next interval.
            log.error(
                "[FLEET-COMMAND] periodic reconciliation failed type=%s detail=%s",
                type(exc).__name__,
                str(exc)[:500],
            )


async def fleet_effect_reconcile_sender(runtime: FleetCommandRuntime) -> None:
    interval = max(0.25, FLEET_EFFECT_RECONCILE_INTERVAL_SEC)
    while True:
        try:
            processed = await runtime.reconcile_effects()
            if processed:
                log.info(
                    "[FLEET-EFFECT] reconciled durable jobs count=%d",
                    processed,
                )
        except Exception as exc:  # noqa: BLE001 - durable leases permit retry.
            log.error(
                "[FLEET-EFFECT] reconciliation failed type=%s detail=%s",
                type(exc).__name__,
                str(exc)[:500],
            )
        await asyncio.sleep(interval)


async def fleet_observation_sender(runtime: FleetCommandRuntime) -> None:
    interval = max(0.5, FLEET_OBSERVATION_REPLAY_INTERVAL_SEC)
    while True:
        try:
            sent = await asyncio.to_thread(runtime.replay_observations)
            if sent:
                log.debug("[FLEET-OBSERVED] replayed count=%d", sent)
        except Exception as exc:  # noqa: BLE001 - durable outbox retries next interval.
            log.error(
                "[FLEET-OBSERVED] replay failed type=%s detail=%s",
                type(exc).__name__,
                str(exc)[:500],
            )
        await asyncio.sleep(interval)


async def handle_command_observation_ack(body: str) -> None:
    runtime = initialize_fleet_command_runtime()
    if runtime is None:
        return
    try:
        ack, removed = await asyncio.to_thread(
            runtime.acknowledge_observation,
            body,
        )
        log.debug(
            "[FLEET-OBSERVED] ACK observationId=%s revision=%d status=%s removed=%s",
            ack.observation_id,
            ack.revision,
            ack.status,
            removed,
        )
    except Exception as exc:  # noqa: BLE001 - malformed ACK cannot delete durable row.
        log.error(
            "[FLEET-OBSERVED] invalid ACK type=%s detail=%s",
            type(exc).__name__,
            str(exc)[:500],
        )


async def webrtc_stats_sender(runtime: FleetCommandRuntime) -> None:
    interval = max(0.5, WEBRTC_STATS_INTERVAL_SEC)
    while True:
        controller = getattr(g_app, "webrtc_uplink", None) if g_app else None
        if controller is not None:
            sample = controller.take_latest_outbound_stats()
            if sample:
                try:
                    await runtime.observe_stream_metrics(sample)
                except Exception as exc:  # noqa: BLE001 - next stats sample retries.
                    log.error(
                        "[STREAM-POLICY] WebRTC stats observation failed type=%s detail=%s",
                        type(exc).__name__,
                        str(exc)[:500],
                    )
            controller.request_outbound_stats()
        await asyncio.sleep(interval)


async def signaling_client_main():
    global websocket, signaling_loop, outbound_queue

    if signaling_loop is None:
        signaling_loop = asyncio.get_running_loop()
    if outbound_queue is None:
        outbound_queue = asyncio.Queue(maxsize=OUTBOUND_QUEUE_MAX)
    initialize_durable_event_outbox()
    await refresh_updater_runtime_telemetry()
    updater_refresh_task = asyncio.create_task(updater_telemetry_refresh_sender())
    command_runtime = initialize_fleet_command_runtime()
    if command_runtime is not None:
        try:
            # Boot verification/rollback is local and must not wait for login or
            # WebSocket availability. Lifecycle ACK/observations remain durable.
            await command_runtime.reconcile_effects()
        except Exception as exc:  # noqa: BLE001 - durable lease retries on connect.
            log.error(
                "[FLEET-EFFECT] startup reconciliation failed type=%s detail=%s",
                type(exc).__name__,
                str(exc)[:500],
            )

    while True:
        token = await login()
        if not token:
            log.error("[SIGNALING] Login failed. Retrying in 10s...")
            await asyncio.sleep(10)
            continue
        set_auth_token(token)
        _reset_agent_ws_state()

        rand_num = "".join(random.choices(string.digits, k=3))
        rand_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
        ws_url = f"{SERVER_BASE_URL.replace('http', 'ws')}/signaling/{rand_num}/{rand_id}/websocket"

        try:
            async with websockets.connect(ws_url) as ws:
                websocket = ws

                open_frame = await ws.recv()
                if open_frame != "o":
                    raise ConnectionError(f"SockJS error: {open_frame}")
                log.info("[SIGNALING] SockJS open.")

                headers = {
                    "accept-version": "1.2,1.1,1.0",
                    "heart-beat": "10000,10000",
                    "Authorization": f"Bearer {token}",
                }
                h_lines = "\n".join([f"{k}:{v}" for k, v in headers.items()])
                connect_frame_str = f"CONNECT\n{h_lines}\n\n\x00"
                await ws.send(json.dumps([connect_frame_str]))

                msg = await ws.recv()
                if not msg.startswith("a["):
                    raise ConnectionError(f"Unexpected msg: {msg}")

                raw = json.loads(msg[1:])[0]
                if "CONNECTED" not in raw:
                    raise ConnectionError("STOMP CONNECT failed")

                connected_frame = stomper.unpack_frame(raw)
                connected_headers = connected_frame.get("headers", {}) if isinstance(connected_frame, dict) else {}
                send_interval_ms = negotiate_stomp_send_interval_ms(
                    10_000,
                    connected_headers.get("heart-beat"),
                )

                log.info("[SIGNALING] ✅ STOMP CONNECTED.")

                for index, destination in enumerate(REQUIRED_AGENT_SUBSCRIPTIONS):
                    await ws.send(json.dumps([stomper.subscribe(destination, f"sub-agent-{index}")]))

                sender_task = asyncio.create_task(outbound_sender(ws))
                event_replay_task = asyncio.create_task(durable_event_replay_sender())
                stomp_heartbeat_task = asyncio.create_task(stomp_heartbeat_sender(ws, send_interval_ms))
                heartbeat_task = asyncio.create_task(device_state_heartbeat_sender())
                connectivity_task = None
                fleet_command_poll_task = None
                fleet_effect_task = None
                webrtc_stats_task = None
                fleet_observation_task = None
                if CONNECTIVITY_ENABLED:
                    connectivity_target_host = CONNECTIVITY_TARGET_HOST or extract_host_from_server_url(SERVER_BASE_URL)
                    connectivity_thresholds = ConnectivityThresholds(
                        poor_rssi_dbm=CONNECTIVITY_POOR_RSSI_DBM,
                        poor_packet_loss_pct=CONNECTIVITY_POOR_PACKET_LOSS_PCT,
                        poor_rtt_ms=CONNECTIVITY_POOR_RTT_MS,
                    )
                    reporter = ConnectivityReporter(
                        target_host=connectivity_target_host,
                        wifi_interface=CONNECTIVITY_WIFI_INTERFACE or None,
                        thresholds=connectivity_thresholds,
                        min_send_interval_sec=CONNECTIVITY_MIN_SEND_INTERVAL_SEC,
                    )
                    connectivity_task = asyncio.create_task(device_connectivity_sender(reporter))

                if command_runtime is not None:
                    command_runtime_connected = False
                    try:
                        reconciled = await command_runtime.on_connected()
                        command_runtime_connected = True
                        log.info(
                            "[FLEET-COMMAND] reconnect reconciliation complete count=%d",
                            reconciled,
                        )
                    except Exception as exc:  # noqa: BLE001 - keep signaling online for retry/wake.
                        log.error(
                            "[FLEET-COMMAND] reconnect reconciliation failed type=%s detail=%s",
                            type(exc).__name__,
                            str(exc)[:500],
                        )
                    if command_runtime_connected:
                        _set_update_commit_signaling_ready(True)
                    fleet_command_poll_task = asyncio.create_task(
                        fleet_command_poll_sender(command_runtime)
                    )
                    fleet_effect_task = asyncio.create_task(
                        fleet_effect_reconcile_sender(command_runtime)
                    )
                    webrtc_stats_task = asyncio.create_task(
                        webrtc_stats_sender(command_runtime)
                    )
                    fleet_observation_task = asyncio.create_task(
                        fleet_observation_sender(command_runtime)
                    )

                async for message in ws:
                    if not message.startswith("a["):
                        continue
                    frame_list = json.loads(message[1:])
                    for frame_str in frame_list:
                        if isinstance(frame_str, str) and not frame_str.strip():
                            continue
                        frame = stomper.unpack_frame(frame_str)
                        destination = frame["headers"].get("destination")
                        body = frame["body"]

                        if destination and COMMAND_OBSERVED_ACK_QUEUE_DEST in destination:
                            await handle_command_observation_ack(body)
                        elif destination and FLEET_COMMAND_QUEUE_DEST in destination:
                            await handle_fleet_command_wakeup(body)
                        elif destination and AGENT_COMMAND_QUEUE_DEST in destination:
                            await handle_command_message(body)
                        elif destination and AGENT_ERROR_QUEUE_DEST in destination:
                            await handle_agent_error(body)
                        elif destination and EVENT_ACK_QUEUE_DEST in destination:
                            await handle_event_ack(body)

        except Exception as exc:
            log.error("[SIGNALING] WebSocket error: %s", exc)
        finally:
            _set_update_commit_signaling_ready(False)
            # A transport reconnect is an exact WebRTC generation boundary.
            # Invalidate the media branch and every volatile signaling envelope,
            # cache entry and delayed retry before a new socket can consume them.
            _reset_webrtc_signaling_transport()
            websocket = None
            if "sender_task" in locals():
                sender_task.cancel()
            if "event_replay_task" in locals():
                event_replay_task.cancel()
            if "stomp_heartbeat_task" in locals():
                stomp_heartbeat_task.cancel()
            if "heartbeat_task" in locals():
                heartbeat_task.cancel()
            if "connectivity_task" in locals() and connectivity_task is not None:
                connectivity_task.cancel()
            if "fleet_command_poll_task" in locals() and fleet_command_poll_task is not None:
                fleet_command_poll_task.cancel()
            if "fleet_effect_task" in locals() and fleet_effect_task is not None:
                fleet_effect_task.cancel()
            if "webrtc_stats_task" in locals() and webrtc_stats_task is not None:
                webrtc_stats_task.cancel()
            if "fleet_observation_task" in locals() and fleet_observation_task is not None:
                fleet_observation_task.cancel()

        log.info("[SIGNALING] Reconnecting in 10s...")
        await asyncio.sleep(10)


class NuvionEventState:
    def __init__(self, overlay_callback=None, demo_source: MvtecDemoSource | None = None):
        self.pipeline_started_at = time.time()
        self.running = True
        self.last_anomaly_at = 0.0
        self.last_production_at = 0.0
        self.zero_shot_last_sample = 0.0
        self.zero_shot_queue = queue.Queue(maxsize=1)
        self.tracking_last_sample = 0.0
        self.tracking_queue = queue.Queue(maxsize=TRACKING_BATCH_SIZE)
        self.overlay_callback = overlay_callback
        self.overlay_lock = threading.Lock()
        self.last_anomaly_overlay: str | OverlayPayload | None = None
        self.tracking_status_text = ""
        self.last_status = None
        self.last_sent_status = None
        self.last_sent_at = 0.0
        self.latest_frame = LatestFrameBuffer()
        self.last_frame_monotonic: float | None = None
        self.clip_enabled = CLIP_ENABLED
        self.demo_mode = DEMO_MODE
        self.demo_tag = DEMO_TAG
        self.demo_source = demo_source
        self.demo_ground_truth_labels = tuple(demo_source.ground_truth_labels) if demo_source else ()
        self.demo_image_duration_sec = float(demo_source.image_duration_sec) if demo_source else 0.0
        self.demo_started_at = time.time()
        self.current_demo_ground_truth = self.demo_ground_truth_labels[0] if self.demo_ground_truth_labels else None
        self.clip_in_progress = False
        self.clip_last_started = 0.0
        self.clip_lock = threading.Lock()
        self.face_tracking_enabled = FACE_TRACKING_ENABLED
        self.face_tracking_show_bbox = FACE_TRACKING_SHOW_BBOX
        self.tracking_overlay_state = TrackingOverlayState()
        self.tracking_overlay_state.update(
            TrackingOverlaySnapshot(
                enabled=self.face_tracking_enabled,
                show_bbox=self.face_tracking_show_bbox,
                status_text="TRACK disabled" if not self.face_tracking_enabled else "TRACK idle",
                updated_at=time.time(),
            )
        )
        self.face_detector = build_face_detector() if self.face_tracking_enabled else None
        self.motor_controller = MotorController(motor_config_from_env())
        self.tracking_controller = None

        self.backend = ZSAD_BACKEND
        self.zero_shot = None
        self.triton_client = None
        self._triton_client_thread_id = None

        if self.backend == "siglip":
            self.zero_shot = ZeroShotAnomalyDetector(
                enabled=ZERO_SHOT_ENABLED,
                model_name=ZERO_SHOT_MODEL,
                labels=ZERO_SHOT_LABELS,
                anomaly_labels=ZERO_SHOT_ANOMALY_LABELS,
                threshold=ZERO_SHOT_THRESHOLD,
                device_preference=ZERO_SHOT_DEVICE,
            )
            if not self.zero_shot.enabled:
                self.backend = "none"
        elif self.backend == "triton":
            if TritonAnomalyClient is None:
                log.warning("[TRITON] Triton client unavailable. Disable backend.")
                self.backend = "none"
        else:
            self.backend = "none"

        if self.face_tracking_enabled:
            if self.face_detector is None or not self.face_detector.ready:
                reason = self.face_detector.error if self.face_detector is not None else "detector unavailable"
                log.warning("[TRACK] Face tracking disabled: %s", reason)
                self.face_tracking_enabled = False
                self.tracking_status_text = f"TRACK unavailable: {reason}"
                self.tracking_overlay_state.update(
                    TrackingOverlaySnapshot(
                        enabled=False,
                        show_bbox=False,
                        status_text=self.tracking_status_text,
                        updated_at=time.time(),
                    )
                )
            else:
                self.tracking_controller = FaceTrackingController(
                    detector=self.face_detector,
                    deadzone_pct=TRACKING_DEADZONE_PCT,
                    hysteresis_pct=TRACKING_HYSTERESIS_PCT,
                    lost_timeout_sec=TRACKING_LOST_TIMEOUT_SEC,
                )
                self.tracking_status_text = "TRACK idle"

        self.worker_thread = threading.Thread(target=self._zsad_worker, daemon=True)
        self.worker_thread.start()
        self.tracking_thread = threading.Thread(target=self._tracking_worker, daemon=True)
        self.tracking_thread.start()

    def reset_demo_timing(self) -> None:
        self.demo_started_at = time.time()
        if self.demo_ground_truth_labels:
            self.current_demo_ground_truth = self.demo_ground_truth_labels[0]

    def update_demo_ground_truth(self, pts_ns: int | None) -> None:
        if not self.demo_mode or not self.demo_ground_truth_labels:
            return

        index: int | None = None
        if pts_ns is not None and pts_ns != Gst.CLOCK_TIME_NONE and self.demo_image_duration_sec > 0:
            duration_ns = max(1, int(self.demo_image_duration_sec * Gst.SECOND))
            index = int(pts_ns // duration_ns)
        elif self.demo_image_duration_sec > 0:
            elapsed = max(0.0, time.time() - self.demo_started_at)
            index = int(elapsed / self.demo_image_duration_sec)

        if index is None:
            return

        self.current_demo_ground_truth = self.demo_ground_truth_labels[index % len(self.demo_ground_truth_labels)]

    def _resolve_tracking_status(self) -> str:
        if not self.face_tracking_enabled:
            return self.tracking_status_text
        if self.tracking_status_text:
            return self.tracking_status_text
        if self.motor_controller.available:
            return "TRACK ready"
        if self.motor_controller.config.enabled:
            return f"TRACK overlay only ({self.motor_controller.reason or 'motor unavailable'})"
        return "TRACK overlay only"

    def _emit_combined_overlay(self) -> None:
        if not self.overlay_callback:
            return

        with self.overlay_lock:
            anomaly_overlay = self.last_anomaly_overlay
            tracking_status = self._resolve_tracking_status()

        if self.demo_mode:
            payload = anomaly_overlay
            if payload is None:
                return
            self._emit_overlay(payload)
            return

        lines: list[str] = []
        if isinstance(anomaly_overlay, OverlayPayload):
            lines.append(f"{anomaly_overlay.status} {anomaly_overlay.label} {anomaly_overlay.score_text}")
        elif isinstance(anomaly_overlay, str) and anomaly_overlay.strip():
            lines.append(anomaly_overlay.strip())

        if tracking_status:
            lines.append(tracking_status)

        if not lines:
            return

        self._emit_overlay("\n".join(lines))

    def _set_anomaly_overlay(self, overlay: str | OverlayPayload) -> None:
        with self.overlay_lock:
            self.last_anomaly_overlay = overlay
        self._emit_combined_overlay()

    def _set_tracking_status(self, status: str) -> None:
        resolved = status
        if self.face_tracking_enabled and not self.motor_controller.available:
            if self.motor_controller.config.enabled:
                resolved = f"{status} | motor unavailable"
            else:
                resolved = f"{status} | overlay only"
        with self.overlay_lock:
            self.tracking_status_text = resolved
        self._emit_combined_overlay()

    def _get_or_create_triton_client(self):
        if TritonAnomalyClient is None:
            raise RuntimeError("Triton client unavailable")

        current_thread_id = threading.get_ident()
        if self.triton_client is None or self._triton_client_thread_id != current_thread_id:
            self.triton_client = TritonAnomalyClient()
            self._triton_client_thread_id = current_thread_id
        return self.triton_client

    def send_status(
        self,
        status: str,
        anomaly_type: str,
        message: str,
        severity: str,
        snapshot_object: str | None = None,
        clip_object: str | None = None,
        clip_status: str | None = None,
    ):
        now = time.time()
        inspection_status = INSPECTION_STATUS_DEFECT if status == "DEFECT" else INSPECTION_STATUS_NORMAL
        get_device_state_coordinator().set_inspection_status(inspection_status)
        prev_sent_status = self.last_sent_status
        status_changed = (prev_sent_status is None) or (status != prev_sent_status)
        self.last_status = status

        if prev_sent_status is None and status == "NORMAL":
            return

        if status_changed:
            pass
        elif status == "DEFECT" and now - self.last_sent_at >= ANOMALY_MIN_INTERVAL_SEC:
            pass
        else:
            return

        event_id = str(uuid.uuid4())
        occurred_at = utc_now_iso()
        if status == "DEFECT" and status_changed and snapshot_object is None:
            snapshot_object = self.capture_snapshot_upload()

        if status == "DEFECT" and status_changed and clip_object is None and clip_status is None:
            clip_object = self.start_clip_upload(event_id=event_id)
            if clip_object:
                clip_status = "UPLOADING"

        tagged_message = self._apply_demo_tag(message)
        payload = {
            "anomalyType": anomaly_type,
            "anomalyStatus": status,
            "message": tagged_message,
            "severity": severity,
            "lineId": LINE_ID,
            "processId": PROCESS_ID,
            "snapshotObject": snapshot_object,
            "clipObject": clip_object,
            "clipStatus": clip_status,
        }
        event = persist_critical_event(
            EVENT_TYPE_ANOMALY,
            "/app/device/anomaly",
            payload,
            event_id,
            occurred_at,
        )
        if event is None:
            return
        self.last_sent_status = status
        self.last_sent_at = now
        if status_changed:
            log.info("[ZSAD] Sent %s status (change): %s", status, tagged_message)
        else:
            log.info("[ZSAD] Sent %s status (repeat): %s", status, tagged_message)

    def _apply_demo_tag(self, message: str) -> str:
        if not self.demo_mode:
            return message
        if message.lstrip().startswith(self.demo_tag):
            return message
        return f"{self.demo_tag} {message}"

    def remember_latest_frame(self, frame_rgb: np.ndarray) -> None:
        self.latest_frame.remember(frame_rgb)
        self.last_frame_monotonic = time.monotonic()

    def capture_snapshot_upload(self) -> str | None:
        if not SNAPSHOT_ENABLED:
            return None

        frame = self.latest_frame.copy()
        try:
            return capture_and_upload_snapshot(
                frame,
                request_upload_url=request_upload_url,
                upload_bytes_to_url=upload_bytes_to_url,
                preferred_content_type=SNAPSHOT_CONTENT_TYPE,
            )
        except Exception as exc:
            log.warning("[SNAPSHOT] Failed to upload snapshot: %s", exc)
            return None

    def start_clip_upload(self, event_id: str | None = None) -> str | None:
        if not self.clip_enabled or not CLIP_ENABLED:
            return None
        now = time.time()
        if now - self.pipeline_started_at < CLIP_WARMUP_SEC:
            log.info(
                "[CLIP] Warm-up guard active. Skip clip creation for %.1fs after startup.",
                CLIP_WARMUP_SEC,
            )
            return None
        with self.clip_lock:
            if self.clip_in_progress:
                return None
            if now - self.clip_last_started < CLIP_COOLDOWN_SEC:
                return None
            self.clip_in_progress = True
            self.clip_last_started = now

        meta = request_upload_url()
        if not meta:
            with self.clip_lock:
                self.clip_in_progress = False
            return None

        object_name = meta.get("objectName")
        upload_url = meta.get("uploadUrl")
        if not object_name or not upload_url:
            with self.clip_lock:
                self.clip_in_progress = False
            return None

        threading.Thread(
            target=self._capture_and_upload_clip,
            args=(object_name, upload_url, now, event_id),
            daemon=True,
        ).start()
        return object_name

    def _capture_and_upload_clip(
        self,
        object_name: str,
        upload_url: str,
        detected_at: float,
        event_id: str | None,
    ):
        clip_path = None
        final_status = "FAILED"
        try:
            clip_path = self._build_clip_from_segments(detected_at)
            if clip_path:
                ok = upload_file_to_url(upload_url, clip_path, CLIP_CONTENT_TYPE)
                final_status = "READY" if ok else "FAILED"
        except Exception as exc:
            log.warning("[CLIP] capture/upload failed object=%s: %s", object_name, exc)
        finally:
            if clip_path:
                try:
                    os.remove(clip_path)
                except OSError:
                    pass
            with self.clip_lock:
                self.clip_in_progress = False
        self._finalize_clip_status(object_name, final_status, event_id)

    @staticmethod
    def _finalize_clip_status(object_name: str, status: str, event_id: str | None) -> None:
        if event_id and critical_event_delivery is not None:
            acknowledged = critical_event_delivery.wait_for_ack(event_id, CLIP_EVENT_ACK_WAIT_SEC)
            if not acknowledged:
                log.warning(
                    "[CLIP] event ACK timeout before finalize eventId=%s object=%s",
                    event_id,
                    object_name,
                )
        update_clip_status_with_retry(object_name, status)

    def _list_segments(self) -> list[str]:
        now = time.time()
        return list_stable_segments(
            CLIP_SEGMENTS_DIR,
            settle_sec=CLIP_SEGMENT_MIN_AGE_SEC,
            probe_func=lambda segment: self._is_stable_segment(segment, now),
            now_ts=now,
        )

    def _is_stable_segment(self, path: str, now: float | None = None) -> bool:
        current = now if now is not None else time.time()
        try:
            stat = os.stat(path)
        except OSError:
            return False

        if current - stat.st_mtime < CLIP_SEGMENT_MIN_AGE_SEC:
            return False

        if stat.st_size < CLIP_SEGMENT_MIN_SIZE_BYTES:
            return False

        ffprobe_path = resolve_ffprobe_path()
        if not ffprobe_path:
            return True

        cmd = [
            ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            log.warning("[CLIP] Skip unstable segment %s: %s", path, result.stderr.strip())
            return False

        try:
            duration = float((result.stdout or "").strip())
        except ValueError:
            return False
        return duration > 0

    def _collect_segments(self, before: float | None = None, after: float | None = None, count: int = 5) -> list[str]:
        segments = self._list_segments()
        if before is not None:
            segments = [s for s in segments if os.path.getmtime(s) <= before]
            return segments[-count:]
        if after is not None:
            segments = [s for s in segments if os.path.getmtime(s) >= after]
            return segments[:count]
        return segments[-count:]

    def _build_clip_from_segments(self, detected_at: float) -> str | None:
        ffmpeg_path = resolve_ffmpeg_path()
        if not ffmpeg_path:
            log.warning("[CLIP] ffmpeg not found. Skip clip creation.")
            return None

        pre_count = max(1, int(math.ceil(CLIP_PRE_SEC / CLIP_SEGMENT_SEC)))
        post_count = max(1, int(math.ceil(CLIP_POST_SEC / CLIP_SEGMENT_SEC)))

        pre_segments = self._collect_segments(before=detected_at, count=pre_count)
        time.sleep(CLIP_POST_SEC + CLIP_SEGMENT_SEC)
        post_segments = self._collect_segments(after=detected_at, count=post_count)

        segments = pre_segments + [s for s in post_segments if s not in pre_segments]
        minimum_segments = max(2, pre_count)
        if len(segments) < minimum_segments:
            log.warning(
                "[CLIP] Not enough stable segments for clip. need>=%s actual=%s",
                minimum_segments,
                len(segments),
            )
            return None

        ts = int(detected_at)
        list_file = os.path.join(CLIP_CLIPS_DIR, f"concat_{ts}.txt")
        output_path = os.path.join(CLIP_CLIPS_DIR, f"clip_{ts}.mp4")

        try:
            with open(list_file, "w") as f:
                for seg in segments:
                    f.write(f"file '{seg}'\n")

            cmd = [
                ffmpeg_path,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_file,
                "-c",
                "copy",
                output_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                log.warning("[CLIP] ffmpeg failed: %s", result.stderr.strip())
                return None
        finally:
            try:
                os.remove(list_file)
            except OSError:
                pass

        return output_path

    def report_production(self, count: int) -> bool:
        event_id = str(uuid.uuid4())
        occurred_at = utc_now_iso()
        payload = {
            "count": int(count),
            "lineId": LINE_ID,
            "processId": PROCESS_ID,
        }
        event = persist_critical_event(
            EVENT_TYPE_PRODUCTION,
            "/app/device/production",
            payload,
            event_id,
            occurred_at,
        )
        return event is not None and not event.dead_lettered and not event.dropped

    def _emit_overlay(self, text: str):
        if self.overlay_callback:
            try:
                self.overlay_callback(text)
            except Exception:
                pass

    def maybe_enqueue_frame(self, frame_rgb):
        now = time.time()
        if self.backend != "none":
            if now - self.zero_shot_last_sample >= ZERO_SHOT_SAMPLE_SEC:
                self.zero_shot_last_sample = now
                if not self.zero_shot_queue.full():
                    try:
                        self.zero_shot_queue.put_nowait(frame_rgb)
                    except queue.Full:
                        pass

        if self.face_tracking_enabled and self.tracking_controller is not None:
            if now - self.tracking_last_sample >= TRACKING_SAMPLE_SEC:
                self.tracking_last_sample = now
                if not self.tracking_queue.full():
                    try:
                        self.tracking_queue.put_nowait(frame_rgb)
                    except queue.Full:
                        pass

    def _zsad_worker(self):
        while self.running:
            if critical_event_safety_gate.is_stopped():
                time.sleep(0.25)
                continue
            try:
                frame = self.zero_shot_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if frame is None:
                continue

            if self.backend == "siglip" and self.zero_shot and self.zero_shot.enabled:
                is_anomaly, result = self.zero_shot.is_anomaly(frame)
                if result:
                    label = result.get("label", "ZSAD")
                    score = float(result.get("score", 0.0))
                    status = "DEFECT" if is_anomaly else "NORMAL"
                    overlay = OverlayPayload(
                        status=status,
                        label=label,
                        score=score,
                        ground_truth=self.current_demo_ground_truth if self.demo_mode else None,
                    )
                    self._set_anomaly_overlay(overlay if self.demo_mode else f"{status} {label} {score:.2f}")
                    try:
                        if status == "DEFECT":
                            self.send_status("DEFECT", label, f"Zero-shot anomaly: {label} ({score:.2f})", "WARNING")
                        else:
                            self.send_status("NORMAL", label, f"Recovered to normal: {label} ({score:.2f})", "INFO")
                    except CriticalEventBackpressureError as exc:
                        log.critical("[SAFETY-STOP] %s", exc)
                        continue

                    if PRODUCTION_LABELS and label.lower() in PRODUCTION_LABELS and score >= PRODUCTION_CONFIDENCE_THRESHOLD:
                        now = time.time()
                        if now - self.last_production_at >= PRODUCTION_DEDUP_SEC:
                            try:
                                if self.report_production(1):
                                    self.last_production_at = now
                            except CriticalEventBackpressureError as exc:
                                log.critical("[SAFETY-STOP] %s", exc)
                                continue

            elif self.backend == "triton":
                try:
                    triton_client = self._get_or_create_triton_client()
                    result = triton_client.predict(frame)
                except Exception as exc:
                    log.warning("[TRITON] inference failed: %s", exc)
                    continue

                if result is None:
                    continue

                label = result.get("label", "ZSAD")
                score = float(result.get("score", 0.0))
                is_anomaly = score >= TRITON_THRESHOLD
                status = "DEFECT" if is_anomaly else "NORMAL"
                overlay = OverlayPayload(
                    status=status,
                    label=label,
                    score=score,
                    ground_truth=self.current_demo_ground_truth if self.demo_mode else None,
                )
                self._set_anomaly_overlay(overlay if self.demo_mode else f"{status} {label} {score:.2f}")

                try:
                    if status == "DEFECT":
                        self.send_status("DEFECT", label, f"Triton anomaly score={score:.2f}", "WARNING")
                    else:
                        self.send_status("NORMAL", label, f"Triton recovered: {label} ({score:.2f})", "INFO")
                except CriticalEventBackpressureError as exc:
                    log.critical("[SAFETY-STOP] %s", exc)
                    continue

                if PRODUCTION_LABELS and label.lower() in PRODUCTION_LABELS and score >= PRODUCTION_CONFIDENCE_THRESHOLD:
                    now = time.time()
                    if now - self.last_production_at >= PRODUCTION_DEDUP_SEC:
                        try:
                            if self.report_production(1):
                                self.last_production_at = now
                        except CriticalEventBackpressureError as exc:
                            log.critical("[SAFETY-STOP] %s", exc)
                            continue

    @staticmethod
    def _drain_queue_batch(work_queue: queue.Queue, first_item, max_items: int) -> list:
        batch = [first_item]
        while len(batch) < max_items:
            try:
                batch.append(work_queue.get_nowait())
            except queue.Empty:
                break
        return batch

    def _tracking_worker(self):
        while self.running:
            try:
                frame = self.tracking_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if frame is None or not self.face_tracking_enabled or self.tracking_controller is None:
                continue

            try:
                detector = getattr(self.tracking_controller, "detector", None)
                batch_limit = max(1, int(getattr(detector, "max_batch_size", TRACKING_BATCH_SIZE)))
                frame_batch = self._drain_queue_batch(self.tracking_queue, frame, min(TRACKING_BATCH_SIZE, batch_limit))
                if detector is not None and hasattr(detector, "detect_many"):
                    faces_batch = detector.detect_many(frame_batch)
                else:
                    faces_batch = [self.tracking_controller.detector.detect(item) for item in frame_batch]
            except Exception as exc:
                log.warning("[TRACK] face tracking inference failed: %s", exc)
                error_text = f"TRACK error: {exc}"
                self.tracking_overlay_state.update(
                    TrackingOverlaySnapshot(
                        enabled=self.face_tracking_enabled,
                        show_bbox=False,
                        status_text=error_text,
                        updated_at=time.time(),
                    )
                )
                self._set_tracking_status(error_text)
                continue

            for batch_frame, faces in zip(frame_batch, faces_batch):
                decision = self.tracking_controller.process_detections(batch_frame.shape[:2], list(faces))
                snapshot = build_overlay_snapshot(
                    decision,
                    enabled=self.face_tracking_enabled,
                    show_bbox=self.face_tracking_show_bbox,
                )
                self.tracking_overlay_state.update(snapshot)
                self._set_tracking_status(decision.status_text)

                if decision.primary_face is None or decision.centered:
                    continue

                if decision.pan_command is not None:
                    self.motor_controller.send_pan(decision.pan_command)
                if decision.tilt_command is not None:
                    self.motor_controller.send_tilt(decision.tilt_command)

def on_new_sample(appsink, user_data: NuvionEventState):
    sample = appsink.emit("pull-sample")
    if sample is None:
        return Gst.FlowReturn.OK

    buffer = sample.get_buffer()
    caps = sample.get_caps()
    if buffer is None or caps is None:
        return Gst.FlowReturn.OK

    structure = caps.get_structure(0)
    width = structure.get_value("width")
    height = structure.get_value("height")

    success, mapinfo = buffer.map(Gst.MapFlags.READ)
    if not success:
        return Gst.FlowReturn.OK

    try:
        frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape(height, width, 3).copy()
    except Exception:
        buffer.unmap(mapinfo)
        return Gst.FlowReturn.OK

    buffer.unmap(mapinfo)
    user_data.update_demo_ground_truth(int(buffer.pts) if buffer.pts != Gst.CLOCK_TIME_NONE else None)
    user_data.remember_latest_frame(frame)
    user_data.maybe_enqueue_frame(frame)
    return Gst.FlowReturn.OK


class PipelineSettingsRuntimeAdapter:
    """Read back only settings that the running pipeline can prove and restore."""

    def __init__(
        self,
        *,
        app,
        encoder: X264EncoderAdapter,
        model_pointer: str,
        model_dir,
    ) -> None:
        self.app = app
        self.encoder = encoder
        self.model_pointer = str(model_pointer)
        self.model_dir = model_dir

    def set_effect_fence(self, fence_check: Callable[[], None]) -> None:
        self.encoder.set_effect_fence(fence_check)

    def snapshot(self) -> dict[str, object]:
        return {
            "labels": {
                "inspection": list(ZERO_SHOT_LABELS),
                "anomaly": list(ZERO_SHOT_ANOMALY_LABELS),
            },
            "clip": {
                "enabled": bool(self.app.user_data.clip_enabled),
                "preSeconds": int(CLIP_PRE_SEC),
                "postSeconds": int(CLIP_POST_SEC),
            },
            "video": {
                "width": int(self.app.video_width),
                "height": int(self.app.video_height),
                "fps": int(self.app.frame_rate),
                "bitrateKbps": self.encoder.read_bitrate_kbps(),
            },
        }

    def apply_immediate(self, desired) -> dict[str, object]:
        global CLIP_PRE_SEC, CLIP_POST_SEC

        if "model" in desired:
            raise UnsupportedSettingsEffect(
                "model changes require activation=RESTART"
            )
        if "labels" in desired:
            current_labels = self.snapshot()["labels"]
            if any(
                current_labels.get(key) != value
                for key, value in desired["labels"].items()
            ):
                raise UnsupportedSettingsEffect(
                    "label changes require activation=RESTART"
                )
            self.verify_labels(desired["labels"])
        clip = desired.get("clip")
        if isinstance(clip, dict):
            if bool(clip["enabled"]) != bool(self.app.user_data.clip_enabled):
                raise UnsupportedSettingsEffect(
                    "clip topology changes require activation=RESTART"
                )
            CLIP_PRE_SEC = float(clip["preSeconds"])
            CLIP_POST_SEC = float(clip["postSeconds"])
        video = desired.get("video")
        if isinstance(video, dict):
            if (
                int(video["width"]) != int(self.app.video_width)
                or int(video["height"]) != int(self.app.video_height)
                or int(video["fps"]) != int(self.app.frame_rate)
            ):
                raise UnsupportedSettingsEffect(
                    "video geometry changes require activation=RESTART"
                )
            self.encoder.set_bitrate_kbps(int(video["bitrateKbps"]))
        return self.snapshot()

    def restore(self, snapshot) -> None:
        global CLIP_PRE_SEC, CLIP_POST_SEC

        clip = snapshot.get("clip")
        if isinstance(clip, dict):
            CLIP_PRE_SEC = float(clip["preSeconds"])
            CLIP_POST_SEC = float(clip["postSeconds"])
        video = snapshot.get("video")
        if isinstance(video, dict):
            self.encoder.set_bitrate_kbps(int(video["bitrateKbps"]))

    def functional_health(self) -> bool:
        if self.app.pipeline is None or not self.app.user_data.running:
            return False
        try:
            state_result, current_state, _pending_state = self.app.pipeline.get_state(0)
            if state_result == Gst.StateChangeReturn.FAILURE:
                return False
            if current_state != Gst.State.PLAYING:
                return False
            return 100 <= self.encoder.read_bitrate_kbps() <= 20_000
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False

    def verify_model(self, desired) -> dict[str, str]:
        detector = getattr(self.app.user_data, "zero_shot", None)
        source_provider = getattr(detector, "loaded_model_source", None)
        if (
            self.app.user_data.backend != "siglip"
            or detector is None
            or not detector.enabled
            or not detector.ready
            or not callable(source_provider)
        ):
            raise UnsupportedSettingsEffect(
                "active inference backend cannot prove exact loaded model identity"
            )
        loaded_source = source_provider()
        if not loaded_source:
            raise UnsupportedSettingsEffect(
                "SigLIP was not loaded from the authenticated local model store"
            )
        try:
            expected_source = Path(self.model_dir).expanduser().resolve(strict=True)
            actual_source = Path(loaded_source).expanduser().resolve(strict=True)
        except OSError as exc:
            raise RuntimeError("loaded model source is no longer resolvable") from exc
        if actual_source != expected_source:
            raise RuntimeError("active runtime loaded an old or different model source")
        if str(desired.get("pointer") or "") != self.model_pointer:
            raise RuntimeError("active configured model pointer mismatch")
        verified = verify_model_artifact_identity(
            actual_source,
            expected_pointer=str(desired.get("pointer") or ""),
            expected_digest=str(desired.get("digest") or ""),
        )
        if verified is None:
            raise RuntimeError("loaded model manifest/digest identity mismatch")
        return verified

    def verify_labels(self, desired) -> dict[str, object]:
        detector = getattr(self.app.user_data, "zero_shot", None)
        if self.app.user_data.backend != "siglip" or detector is None or not detector.enabled:
            raise UnsupportedSettingsEffect(
                "labels are not active in the selected inference backend"
            )
        inspection = list(desired.get("inspection") or [])
        anomaly = list(desired.get("anomaly") or [])
        if "inspection" in desired and list(detector.labels) != inspection:
            raise RuntimeError("active inspection labels mismatch")
        if "anomaly" in desired and {
            value.lower() for value in anomaly
        } != set(detector.anomaly_labels):
            raise RuntimeError("active anomaly labels mismatch")
        return dict(desired)


class GStreamerInferenceApp:
    def __init__(self, video_source: str):
        self.video_width = VIDEO_WIDTH
        self.video_height = VIDEO_HEIGHT
        self.frame_rate = VIDEO_FPS
        self.video_source = video_source
        self.demo_mode = DEMO_MODE
        self.demo_loop = DEMO_LOOP
        self.demo_source = self._prepare_demo_source() if self.demo_mode else None
        self.rtp_ssrc = get_rtp_ssrc()
        self.overlay = None
        self.tracking_overlay = None
        self.status_overlay = None
        self.label_overlay = None
        self.score_overlay = None
        self.gt_overlay = None
        self._overlay_update_lock = threading.Lock()
        self._pending_overlay_text: str | OverlayPayload | None = None
        self.user_data = NuvionEventState(self.update_overlay_text, demo_source=self.demo_source)
        self.webrtc_uplink = WebRTCUplinkController(
            send_message=self.send_webrtc_signal,
            default_force_relay=WEBRTC_FORCE_RELAY,
            h264_profile_level_id=H264_PROFILE_LEVEL_ID_ENV,
            h264_packetization_mode=H264_PACKETIZATION_MODE_ENV,
            h264_level_asymmetry_allowed=H264_LEVEL_ASYMMETRY_ALLOWED_ENV,
            on_fatal_cleanup=self._on_webrtc_cleanup_failure,
        )

        self.pipeline = None
        self.loop = None
        self.encoder_adapter: X264EncoderAdapter | None = None
        self.depthai_bridge: DepthAIGStreamerBridge | None = None
        self._demo_restarting = False
        self._demo_last_restart_at = 0.0
        self._supervisor_restart_lock = threading.Lock()
        self._supervisor_restart_requested = False

        self.create_pipeline()

        global g_app
        g_app = self

    def _prepare_demo_source(self) -> MvtecDemoSource | None:
        return prepare_mvtec_demo_source(
            base_url=os.getenv("NUVION_DEMO_MVTEC_BASE_URL"),
            categories=os.getenv("NUVION_DEMO_MVTEC_CATEGORIES"),
            cache_dir=os.getenv("NUVION_DEMO_MVTEC_CACHE_DIR"),
            image_duration_sec=float(os.getenv("NUVION_DEMO_IMAGE_DURATION_SEC", "1.0")),
        )

    def create_pipeline(self):
        Gst.init(None)
        source_pipeline = build_video_source_pipeline(
            self.video_source,
            self.video_width,
            self.video_height,
            self.frame_rate,
            gst_source_override=GST_SOURCE_OVERRIDE,
            demo_mode=self.demo_mode,
            demo_source=self.demo_source,
        )

        tracking_overlay_pipeline = ""
        if (
            self.user_data.face_tracking_enabled
            and self.user_data.face_tracking_show_bbox
            and Gst.ElementFactory.find("cairooverlay") is not None
        ):
            tracking_overlay_pipeline = (
                "videoconvert ! "
                "video/x-raw,format=BGRx ! "
                "cairooverlay name=tracking_overlay ! "
                "videoconvert ! "
            )

        if self.demo_mode:
            overlay_pipeline = (
                f"{tracking_overlay_pipeline}"
                "videoconvert ! "
                "textoverlay name=zsad_overlay "
                "font-desc=\"Sans 24\" "
                "halignment=left valignment=top "
                "shaded-background=true "
                "xpad=25 "
                "text=\"\" "
                "! "
                "textoverlay name=zsad_status_overlay "
                "font-desc=\"Monospace 24\" "
                "halignment=left valignment=top "
                f"xpad={DEMO_OVERLAY_STATUS_XPAD} "
                "shaded-background=true "
                "color=4294967295 "
                "text=\"\" "
                "! "
                "textoverlay name=zsad_label_overlay "
                "font-desc=\"Monospace 24\" "
                "halignment=left valignment=top "
                f"xpad={DEMO_OVERLAY_LABEL_XPAD} "
                "shaded-background=true "
                "color=4294967295 "
                "text=\"\" "
                "! "
                "textoverlay name=zsad_score_overlay "
                "font-desc=\"Monospace 24\" "
                "halignment=left valignment=top "
                f"xpad={DEMO_OVERLAY_SCORE_XPAD} "
                "shaded-background=true "
                "color=4294967295 "
                "text=\"\" "
                "! "
                "textoverlay name=zsad_gt_overlay "
                "font-desc=\"Monospace 24\" "
                "halignment=left valignment=top "
                f"xpad={DEMO_OVERLAY_GT_XPAD} "
                "shaded-background=true "
                "color=4294967295 "
                "text=\"\" "
                "! "
            )
        else:
            overlay_pipeline = (
                f"{tracking_overlay_pipeline}"
                "videoconvert ! "
                "textoverlay name=zsad_overlay "
                "font-desc=\"Sans 24\" "
                "halignment=left valignment=top "
                "shaded-background=true "
                "text=\"\" "
                "! "
            )

        uplink_pipeline = build_uplink_pipeline(
            rtp_ssrc=self.rtp_ssrc,
            clip_enabled=CLIP_ENABLED,
            clip_segment_sec=CLIP_SEGMENT_SEC,
            clip_max_segments=CLIP_MAX_SEGMENTS,
            clip_segments_dir=CLIP_SEGMENTS_DIR,
            video_bitrate_kbps=VIDEO_BITRATE_KBPS,
        )
        live_queue = build_bounded_live_queue()

        if LOCAL_DISPLAY:
            pipeline_string = (
                f"{source_pipeline} ! "
                "tee name=t "
                f"t. ! {live_queue} ! "
                "appsink name=zsad_sink emit-signals=true max-buffers=1 drop=true sync=false "
                f"t. ! {live_queue} ! "
                f"{overlay_pipeline}"
                "tee name=dt "
                f"dt. ! {live_queue} ! "
                f"{uplink_pipeline} "
                f"dt. ! {live_queue} ! videoconvert ! autovideosink sync=false"
            )
        else:
            pipeline_string = (
                f"{source_pipeline} ! "
                "tee name=t "
                f"t. ! {live_queue} ! "
                "appsink name=zsad_sink emit-signals=true max-buffers=1 drop=true sync=false "
                f"t. ! {live_queue} ! "
                f"{overlay_pipeline}"
                f"{uplink_pipeline}"
            )

        log.info("[PIPELINE] %s", pipeline_string)
        self.pipeline = Gst.parse_launch(pipeline_string)
        self.loop = GLib.MainLoop()

        if should_use_depthai_source(
            self.video_source,
            gst_source_override=GST_SOURCE_OVERRIDE,
            demo_mode=self.demo_mode,
        ):
            depthai_appsrc = self.pipeline.get_by_name(DEPTHAI_APPSRC_NAME)
            if depthai_appsrc is None:
                raise RuntimeError("DepthAI GStreamer appsrc is missing")
            configured_device_id = resolve_depthai_device_id(
                self.video_source,
                DEPTHAI_DEVICE_ID,
            )
            depthai_source = DepthAIFrameSource(
                DepthAIConfig(
                    width=self.video_width,
                    height=self.video_height,
                    fps=self.frame_rate,
                    device_id=configured_device_id,
                    queue_size=1,
                    startup_timeout=DEPTHAI_STARTUP_TIMEOUT_SEC,
                    read_timeout=DEPTHAI_READ_TIMEOUT_SEC,
                )
            )
            self.depthai_bridge = DepthAIGStreamerBridge(
                frame_source=depthai_source,
                appsrc=depthai_appsrc,
                gst=Gst,
                width=self.video_width,
                height=self.video_height,
                read_timeout=DEPTHAI_READ_TIMEOUT_SEC,
                max_consecutive_timeouts=DEPTHAI_MAX_CONSECUTIVE_TIMEOUTS,
                on_failure=self._on_depthai_failure,
                logger=log,
            )

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, self.loop)

        appsink = self.pipeline.get_by_name("zsad_sink")
        if appsink:
            appsink.connect("new-sample", on_new_sample, self.user_data)
        else:
            log.warning("[PIPELINE] zsad_sink not found.")

        self.overlay = self.pipeline.get_by_name("zsad_overlay")
        if not self.overlay:
            log.warning("[PIPELINE] zsad_overlay not found.")
        else:
            self._queue_overlay_text(self._default_overlay_text())
            self._flush_pending_overlay_text()
            GLib.timeout_add(100, self._flush_pending_overlay_text)
        self.tracking_overlay = self.pipeline.get_by_name("tracking_overlay")
        if self.tracking_overlay:
            self.tracking_overlay.connect("draw", self._draw_tracking_overlay)
        elif self.user_data.face_tracking_enabled and self.user_data.face_tracking_show_bbox:
            log.warning("[TRACK] cairooverlay not available. Bounding boxes will be disabled.")
        self.status_overlay = self.pipeline.get_by_name("zsad_status_overlay")
        self.label_overlay = self.pipeline.get_by_name("zsad_label_overlay")
        self.score_overlay = self.pipeline.get_by_name("zsad_score_overlay")
        self.gt_overlay = self.pipeline.get_by_name("zsad_gt_overlay")

        video_encoder = self.pipeline.get_by_name("video_encoder")
        if video_encoder is None:
            fleet_effect_registry.unregister("STREAM_POLICY")
            log.error(
                "[STREAM-POLICY] named video_encoder missing; capability disabled"
            )
        else:
            self.encoder_adapter = X264EncoderAdapter(
                video_encoder,
                dispatch=GlibMainContextDispatcher(GLib.idle_add),
            )
            fleet_effect_registry.register(
                StreamPolicyReconciler(self.encoder_adapter)
            )
            log.info("[STREAM-POLICY] x264 reconciler registered")

            try:
                settings_state_dir = resolve_settings_state_dir(os.environ)
                settings_runtime = PipelineSettingsRuntimeAdapter(
                    app=self,
                    encoder=self.encoder_adapter,
                    model_pointer=MODEL_POINTER,
                    model_dir=resolve_model_dir(resolve_effective_profile()),
                )
                fleet_effect_registry.register(
                    SettingsReconciler(
                        store=AtomicSettingsStore(
                            resolve_config_path(),
                            settings_state_dir,
                        ),
                        runtime=settings_runtime,
                        process_instance_id=FLEET_PROCESS_INSTANCE_ID,
                    )
                )
                log.info("[CONFIG-APPLY] transactional reconciler registered")
            except (OSError, RuntimeError, ValueError) as exc:
                fleet_effect_registry.unregister("CONFIG_APPLY")
                log.error(
                    "[CONFIG-APPLY] capability disabled: %s",
                    exc,
                )

        if self.webrtc_uplink and self.pipeline and not self.webrtc_uplink.attach_pipeline(self.pipeline):
            self.webrtc_uplink = None

    def _restart_demo_pipeline(self, reason: str) -> bool:
        if not self.pipeline:
            return False
        now = time.time()
        if self._demo_restarting:
            return False
        # Avoid tight restart loops when upstream keeps failing.
        if now - self._demo_last_restart_at < 0.5:
            return False

        self._demo_restarting = True
        self._demo_last_restart_at = now
        try:
            self.pipeline.set_state(Gst.State.NULL)
            # Wait state transition to settle before replay.
            self.pipeline.get_state(2 * Gst.SECOND)
            restart_result = self.pipeline.set_state(Gst.State.PLAYING)
            if restart_result == Gst.StateChangeReturn.FAILURE:
                return False
            self.user_data.reset_demo_timing()
            self.update_overlay_text(self._default_overlay_text())
            log.info("[DEMO] Restarted demo video (%s).", reason)
            return True
        finally:
            self._demo_restarting = False

    def request_supervisor_restart(self) -> bool:
        """Gracefully end the process; systemd Restart=always owns relaunch."""

        with self._supervisor_restart_lock:
            if self._supervisor_restart_requested:
                return True

            def _graceful_stop() -> bool:
                self.shutdown()
                return False

            try:
                source_id = GLib.idle_add(_graceful_stop)
            except Exception:  # noqa: BLE001 - caller will retry the request.
                return False
            if not source_id:
                return False
            self._supervisor_restart_requested = True
            log.warning(
                "[CONFIG-APPLY] graceful shutdown requested for supervisor restart"
            )
            return True

    def _on_webrtc_cleanup_failure(self, reason: str) -> bool:
        get_device_state_coordinator().set_runtime_status(RUNTIME_STATUS_ERROR)
        log.critical(
            "[WEBRTC-UPLINK] unreleased media branch requires process recovery: %s",
            str(reason)[:500],
        )
        return self.request_supervisor_restart()

    def _on_depthai_failure(self, exc: BaseException) -> None:
        log.error(
            "[DEPTHAI] capture failed type=%s detail=%s",
            type(exc).__name__,
            str(exc)[:500],
        )
        get_device_state_coordinator().set_runtime_status(RUNTIME_STATUS_ERROR)

        def _stop_failed_pipeline() -> bool:
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
            if self.loop and self.loop.is_running():
                self.loop.quit()
            return False

        try:
            GLib.idle_add(_stop_failed_pipeline)
        except Exception:  # noqa: BLE001 - fallback during GLib teardown.
            _stop_failed_pipeline()

    def bus_call(self, bus, message, loop):
        msg_type = message.type
        if msg_type == Gst.MessageType.EOS:
            if self.demo_mode and self.demo_loop and self.pipeline:
                if self._restart_demo_pipeline("eos"):
                    return True
                log.error("[DEMO] Failed to restart demo video on EOS.")
            log.info("End-of-stream")
            self.shutdown()
        elif msg_type == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            get_device_state_coordinator().set_runtime_status(RUNTIME_STATUS_ERROR)
            err_text = str(err).lower()
            dbg_text = (dbg or "").lower()
            if self.demo_mode and self.demo_loop and (
                "not-linked" in err_text or "not-linked" in dbg_text
            ):
                log.warning("[DEMO] GStreamer not-linked error detected. Trying pipeline restart.")
                if self._restart_demo_pipeline("not-linked-error"):
                    return True
            log.error("GStreamer Error: %s, %s", err, dbg)
            self.shutdown()
        return True

    def send_webrtc_signal(
        self,
        destination: str,
        payload: dict,
        remember: bool,
        signaling_token: WebRTCSignalingToken,
    ) -> bool:
        return enqueue_stomp_message(
            destination,
            payload,
            remember=remember,
            signaling_token=signaling_token,
        )

    def _draw_tracking_overlay(self, _overlay, context, _timestamp, _duration) -> None:
        snapshot = self.user_data.tracking_overlay_state.snapshot()
        draw_tracking_overlay(context, snapshot)

    def _set_overlay_field(self, overlay, text: str, color: int = OVERLAY_COLOR_WHITE) -> None:
        if overlay is None:
            return
        overlay.set_property("text", text)
        overlay.set_property("color", color)

    def _queue_overlay_text(self, text: str | OverlayPayload) -> None:
        with self._overlay_update_lock:
            self._pending_overlay_text = text

    def _flush_pending_overlay_text(self) -> bool:
        if not self.overlay:
            return True

        with self._overlay_update_lock:
            text = self._pending_overlay_text
            self._pending_overlay_text = None

        if text is None:
            return True

        if self.demo_mode and isinstance(text, OverlayPayload):
            match = text.matches_ground_truth
            match_color = OVERLAY_COLOR_WHITE
            if match is True:
                match_color = OVERLAY_COLOR_GREEN
            elif match is False:
                match_color = OVERLAY_COLOR_RED

            self.overlay.set_property("text", "")
            self._set_overlay_field(self.status_overlay, text.status, match_color)
            self._set_overlay_field(self.label_overlay, text.label)
            self._set_overlay_field(self.score_overlay, text.score_text)
            self._set_overlay_field(self.gt_overlay, text.ground_truth or "", match_color)
            return True

        resolved_text = text if isinstance(text, str) else f"{text.status} {text.label} {text.score_text}"
        self.overlay.set_property("text", resolved_text)
        self._set_overlay_field(self.status_overlay, "")
        self._set_overlay_field(self.label_overlay, "")
        self._set_overlay_field(self.score_overlay, "")
        self._set_overlay_field(self.gt_overlay, "")
        return True

    def update_overlay_text(self, text: str | OverlayPayload):
        self._queue_overlay_text(text)

    def _default_overlay_text(self) -> str:
        backend = getattr(self.user_data, "backend", "none")
        prefix = "DEMO | " if self.demo_mode else ""
        tracking_suffix = " | TRACK" if self.user_data.face_tracking_enabled else ""
        if backend == "triton":
            return f"{prefix}ZSAD TRITON ON | WEBRTC{tracking_suffix}"
        if backend == "siglip":
            return f"{prefix}ZSAD ON | WEBRTC{tracking_suffix}"
        return f"{prefix}ZSAD OFF | WEBRTC{tracking_suffix}"

    def run(self):
        def _start():
            log.info("Starting GStreamer main loop...")
            try:
                state_result = self.pipeline.set_state(Gst.State.PLAYING)
                if state_result == Gst.StateChangeReturn.FAILURE:
                    raise RuntimeError("GStreamer pipeline failed to enter PLAYING")
                if self.depthai_bridge is not None:
                    self.depthai_bridge.start()
                    log.info("[DEPTHAI] RGB source started")

                get_device_state_coordinator().set_runtime_status(RUNTIME_STATUS_RUNNING)
                log.info("Starting signaling thread...")
                signaling_thread = threading.Thread(
                    target=lambda: asyncio.run(signaling_client_main()),
                    daemon=True,
                )
                signaling_thread.start()
                self.loop.run()
            except KeyboardInterrupt:
                log.info("KeyboardInterrupt received.")
            except Exception:
                get_device_state_coordinator().set_runtime_status(RUNTIME_STATUS_ERROR)
                raise
            finally:
                self.shutdown()

        if LOCAL_DISPLAY and sys.platform == "darwin":
            log.info("Using Gst.macos_main() for local display on macOS...")
            def _macos_main(_argc, _argv, _data):
                _start()
                return 0
            Gst.macos_main(_macos_main, sys.argv, "")
        else:
            _start()

    def shutdown(self):
        self.user_data.running = False
        self.user_data.motor_controller.close()
        if self.depthai_bridge is not None:
            self.depthai_bridge.close()
        if self.webrtc_uplink:
            self.webrtc_uplink.stop(send_signal=True)
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        if self.loop and self.loop.is_running():
            self.loop.quit()


def main():
    video_source = os.getenv("NUVION_VIDEO_SOURCE", "auto")
    app = GStreamerInferenceApp(video_source)
    app.run()


if __name__ == "__main__":
    main()
