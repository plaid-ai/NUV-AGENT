# nuvion_app/inference/pipeline.py
#
# USB/Webcam -> GStreamer -> H.264(RTP) -> mediasoup plain transport
# Zero-shot anomaly detection (SigLIP) or Triton backend (optional)

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
import subprocess
import urllib.request
import urllib.error
from dataclasses import dataclass
from urllib.parse import urlparse

import numpy as np
from nuvion_app.config import load_env
import aiohttp
import websockets
from nuvion_app.inference.connectivity import ConnectivityReporter
from nuvion_app.inference.connectivity import ConnectivityThresholds
from nuvion_app.inference.demo_mvtec import MvtecDemoSource
from nuvion_app.inference.demo_mvtec import prepare_mvtec_demo_source
from nuvion_app.inference.face_tracking import FaceTrackingController
from nuvion_app.inference.face_tracking import TrackingOverlaySnapshot
from nuvion_app.inference.face_tracking import TrackingOverlayState
from nuvion_app.inference.face_tracking import build_face_detector
from nuvion_app.inference.face_tracking import build_overlay_snapshot
from nuvion_app.inference.face_tracking import draw_tracking_overlay
from nuvion_app.inference.motor import MotorController
from nuvion_app.inference.motor import motor_config_from_env
from nuvion_app.inference.webrtc_signaling import (
    WEBRTC_UPLINK_ANSWER,
    WEBRTC_UPLINK_ICE_CANDIDATE,
    WEBRTC_UPLINK_OFFER_DEST,
    WEBRTC_UPLINK_ICE_CANDIDATE_DEST,
    WEBRTC_UPLINK_START,
    WEBRTC_UPLINK_STATE,
    WEBRTC_UPLINK_STOP_DEST,
    parse_command_payload,
)
from nuvion_app.inference.webrtc_uplink import WebRTCUplinkController

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
from nuvion_app.inference.video_source import is_truthy
from nuvion_app.runtime.inference_mode import (
    apply_inference_runtime_defaults,
    normalize_backend,
    normalize_siglip_device,
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


load_env()
apply_inference_runtime_defaults()

SERVER_BASE_URL = os.getenv("NUVION_SERVER_BASE_URL", "http://localhost:8080")
DEVICE_USERNAME = os.getenv("NUVION_DEVICE_USERNAME", "device")
DEVICE_PASSWORD = os.getenv("NUVION_DEVICE_PASSWORD", "password")

VIDEO_SOURCE_ENV = os.getenv("NUVION_VIDEO_SOURCE", "auto")
GST_SOURCE_OVERRIDE = os.getenv("NUVION_GST_SOURCE")
DEMO_MODE = is_truthy(os.getenv("NUVION_DEMO_MODE", "false"))
DEMO_LOOP = is_truthy(os.getenv("NUVION_DEMO_LOOP", "true"))
DEMO_TAG = ((os.getenv("NUVION_DEMO_TAG", "[DEMO]") or "").strip() or "[DEMO]")

WEBRTC_FORCE_RELAY = is_truthy(os.getenv("NUVION_WEBRTC_FORCE_RELAY", "true"))
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
ZERO_SHOT_LABELS = parse_csv(os.getenv("NUVION_ZERO_SHOT_LABELS", "normal,defect"))
ZERO_SHOT_ANOMALY_LABELS = parse_csv(os.getenv("NUVION_ZERO_SHOT_ANOMALY_LABELS", "defect,broken,crack,scratch"))
ZERO_SHOT_THRESHOLD = parse_float(os.getenv("NUVION_ZERO_SHOT_THRESHOLD"), 0.7)
ZERO_SHOT_SAMPLE_SEC = parse_float(os.getenv("NUVION_ZERO_SHOT_SAMPLE_SEC"), 2.0)
FACE_TRACKING_ENABLED = is_truthy(os.getenv("NUVION_FACE_TRACKING_ENABLED", "false"))
FACE_TRACKING_SHOW_BBOX = is_truthy(os.getenv("NUVION_FACE_TRACKING_SHOW_BBOX", "true"))
TRACKING_SAMPLE_SEC = parse_float(os.getenv("NUVION_TRACKING_SAMPLE_SEC"), 0.05)
TRACKING_BATCH_SIZE = max(int(os.getenv("NUVION_FACE_TRACKING_BATCH_SIZE", "2") or "2"), 1)
TRACKING_DEADZONE_PCT = parse_float(os.getenv("NUVION_TRACKING_DEADZONE_PCT"), 0.08)
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
AGENT_ERROR_MAX_RETRIES = int(os.getenv("NUVION_AGENT_ERROR_MAX_RETRIES", "3"))
AGENT_ERROR_BACKOFF_BASE_SEC = parse_float(os.getenv("NUVION_AGENT_ERROR_BACKOFF_BASE_SEC"), 1.0)
AGENT_ERROR_BACKOFF_MAX_SEC = parse_float(os.getenv("NUVION_AGENT_ERROR_BACKOFF_MAX_SEC"), 15.0)

AGENT_ERROR_QUEUE_DEST = "/user/queue/agent/error"
AGENT_COMMAND_QUEUE_DEST = "/user/queue/command"
AGENT_RETRY_DESTINATIONS = {
    "/app/device/anomaly",
    "/app/device/production",
    "/app/device/log",
    "/app/device/state",
    "/app/device/connectivity",
    WEBRTC_UPLINK_OFFER_DEST,
    WEBRTC_UPLINK_ICE_CANDIDATE_DEST,
    WEBRTC_UPLINK_STOP_DEST,
}

websocket: websockets.WebSocketClientProtocol | None = None
g_app = None
signaling_loop: asyncio.AbstractEventLoop | None = None
outbound_queue: asyncio.Queue | None = None
auth_token: str | None = None
auth_token_lock = threading.Lock()
agent_uplink_blocked = False
agent_uplink_block_reason = ""
agent_uplink_lock = threading.Lock()
agent_retry_attempts: dict[str, int] = {}
agent_retry_lock = threading.Lock()
last_sent_payloads: dict[str, dict] = {}
last_sent_payloads_lock = threading.Lock()
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


def request_upload_url() -> dict | None:
    payload = {"type": "CLIP", "contentType": CLIP_CONTENT_TYPE}
    response = api_request("POST", "/devices/media/upload-url", payload)
    if not response:
        return None
    return response.get("data")


def update_clip_status(object_name: str, status: str) -> None:
    payload = {"objectName": object_name, "status": status}
    api_request("PATCH", "/devices/media/clip-status", payload)


def upload_file_to_url(upload_url: str, file_path: str, content_type: str) -> bool:
    try:
        server_host = urlparse(SERVER_BASE_URL).netloc
        upload_host = urlparse(upload_url).netloc
        with open(file_path, "rb") as f:
            data = f.read()
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


def build_send_frame(destination: str, payload: dict) -> str:
    return (
        "SEND\n"
        f"destination:{destination}\n"
        "content-type:application/json\n\n"
        f"{json.dumps(payload)}\x00"
    )


def _clone_payload(payload: dict) -> dict:
    return json.loads(json.dumps(payload))


def _remember_last_payload(destination: str, payload: dict) -> None:
    if destination not in AGENT_RETRY_DESTINATIONS:
        return
    with last_sent_payloads_lock:
        last_sent_payloads[destination] = _clone_payload(payload)


def _get_last_payload(destination: str) -> dict | None:
    with last_sent_payloads_lock:
        payload = last_sent_payloads.get(destination)
        if payload is None:
            return None
        return _clone_payload(payload)


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


def _reset_agent_retry_attempt(destination: str) -> None:
    with agent_retry_lock:
        if destination in agent_retry_attempts:
            agent_retry_attempts[destination] = 0


def _next_agent_retry_attempt(destination: str) -> int:
    with agent_retry_lock:
        attempt = agent_retry_attempts.get(destination, 0) + 1
        agent_retry_attempts[destination] = attempt
        return attempt


def _reset_agent_ws_state() -> None:
    _set_agent_uplink_blocked(False, "")
    with agent_retry_lock:
        agent_retry_attempts.clear()
    if g_app and getattr(g_app, "webrtc_uplink", None):
        g_app.webrtc_uplink.on_signaling_reset()


def enqueue_stomp_message(destination: str, payload: dict, remember: bool = True) -> bool:
    if outbound_queue is None or signaling_loop is None:
        log.warning("[STOMP] outbound not ready, dropping message to %s", destination)
        return False
    if _is_agent_uplink_blocked(destination):
        return False
    if remember:
        _remember_last_payload(destination, payload)

    def _enqueue():
        try:
            outbound_queue.put_nowait((destination, payload))
        except asyncio.QueueFull:
            log.warning("[STOMP] outbound queue full, dropping message to %s", destination)

    signaling_loop.call_soon_threadsafe(_enqueue)
    return True


async def outbound_sender(ws: websockets.WebSocketClientProtocol):
    if outbound_queue is None:
        return
    while True:
        destination, payload = await outbound_queue.get()
        frame_str = build_send_frame(destination, payload)
        try:
            await ws.send(json.dumps([frame_str]))
        except Exception as exc:
            log.warning("[STOMP] send failed: %s", exc)


async def device_state_heartbeat_sender():
    interval = max(1.0, DEVICE_STATE_INTERVAL_SEC)
    while True:
        payload = {
            "status": "RUNNING",
            "message": "heartbeat",
            "lineId": LINE_ID,
            "processId": PROCESS_ID,
        }
        enqueue_stomp_message("/app/device/state", payload)
        await asyncio.sleep(interval)


async def device_connectivity_sender(reporter: ConnectivityReporter):
    interval = max(1.0, CONNECTIVITY_INTERVAL_SEC)
    while True:
        payload = reporter.build_transition_payload()
        if payload and enqueue_stomp_message("/app/device/connectivity", payload):
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
) -> None:
    await asyncio.sleep(delay_sec)
    enqueued = enqueue_stomp_message(destination, payload, remember=False)
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

    if retryable:
        if path not in AGENT_RETRY_DESTINATIONS:
            log.warning(
                "[AGENT-ERROR][retryable] unsupported path. code=%s path=%s message=%s",
                code,
                path,
                message,
            )
            return

        last_payload = _get_last_payload(path)
        if last_payload is None:
            log.warning(
                "[AGENT-ERROR][retryable] no cached payload. code=%s path=%s message=%s",
                code,
                path,
                message,
            )
            return

        attempt = _next_agent_retry_attempt(path)
        if attempt > AGENT_ERROR_MAX_RETRIES:
            log.error(
                "[AGENT-ERROR] retry exhausted. code=%s path=%s max=%d detail=%s",
                code,
                path,
                AGENT_ERROR_MAX_RETRIES,
                detail,
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
        asyncio.create_task(
            _enqueue_retry_after_delay(
                destination=path,
                payload=last_payload,
                delay_sec=delay_sec,
                attempt=attempt,
                max_attempts=AGENT_ERROR_MAX_RETRIES,
                code=code,
            )
        )
        return

    _reset_agent_retry_attempt(path)
    if status_int in (401, 403):
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


async def signaling_client_main():
    global websocket, signaling_loop, outbound_queue

    if signaling_loop is None:
        signaling_loop = asyncio.get_running_loop()
    if outbound_queue is None:
        outbound_queue = asyncio.Queue(maxsize=OUTBOUND_QUEUE_MAX)

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

                log.info("[SIGNALING] ✅ STOMP CONNECTED.")

                await ws.send(json.dumps([stomper.subscribe(AGENT_COMMAND_QUEUE_DEST, "sub-command")]))
                await ws.send(json.dumps([stomper.subscribe(AGENT_ERROR_QUEUE_DEST, "sub-agent-error")]))

                sender_task = asyncio.create_task(outbound_sender(ws))
                heartbeat_task = asyncio.create_task(device_state_heartbeat_sender())
                connectivity_task = None
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

                async for message in ws:
                    if not message.startswith("a["):
                        continue
                    frame_list = json.loads(message[1:])
                    for frame_str in frame_list:
                        frame = stomper.unpack_frame(frame_str)
                        destination = frame["headers"].get("destination")
                        body = frame["body"]

                        if destination and AGENT_COMMAND_QUEUE_DEST in destination:
                            await handle_command_message(body)
                        elif destination and AGENT_ERROR_QUEUE_DEST in destination:
                            await handle_agent_error(body)

        except Exception as exc:
            log.error("[SIGNALING] WebSocket error: %s", exc)
        finally:
            websocket = None
            if "sender_task" in locals():
                sender_task.cancel()
            if "heartbeat_task" in locals():
                heartbeat_task.cancel()
            if "connectivity_task" in locals() and connectivity_task is not None:
                connectivity_task.cancel()

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
        clip_object: str | None = None,
        clip_status: str | None = None,
    ):
        now = time.time()
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

        if status == "DEFECT" and status_changed and clip_object is None and clip_status is None:
            clip_object = self.start_clip_upload()
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
            "snapshotObject": None,
            "clipObject": clip_object,
            "clipStatus": clip_status,
        }
        enqueued = enqueue_stomp_message("/app/device/anomaly", payload)
        if not enqueued:
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

    def start_clip_upload(self) -> str | None:
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
            args=(object_name, upload_url, now),
            daemon=True,
        ).start()
        return object_name

    def _capture_and_upload_clip(self, object_name: str, upload_url: str, detected_at: float):
        clip_path = None
        try:
            clip_path = self._build_clip_from_segments(detected_at)
            if not clip_path:
                update_clip_status(object_name, "FAILED")
                return

            ok = upload_file_to_url(upload_url, clip_path, CLIP_CONTENT_TYPE)
            update_clip_status(object_name, "READY" if ok else "FAILED")
        finally:
            if clip_path:
                try:
                    os.remove(clip_path)
                except OSError:
                    pass
            with self.clip_lock:
                self.clip_in_progress = False

    def _list_segments(self) -> list[str]:
        pattern = os.path.join(CLIP_SEGMENTS_DIR, "segment_*.mp4")
        segments = glob.glob(pattern)
        segments.sort(key=os.path.getmtime)
        if len(segments) > 1:
            segments = segments[:-1]
        now = time.time()
        return [segment for segment in segments if self._is_stable_segment(segment, now)]

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

    def report_production(self, count: int):
        payload = {
            "count": int(count),
            "lineId": LINE_ID,
            "processId": PROCESS_ID,
        }
        enqueue_stomp_message("/app/device/production", payload)

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
                    if status == "DEFECT":
                        self.send_status("DEFECT", label, f"Zero-shot anomaly: {label} ({score:.2f})", "WARNING")
                    else:
                        self.send_status("NORMAL", label, f"Recovered to normal: {label} ({score:.2f})", "INFO")

                    if PRODUCTION_LABELS and label.lower() in PRODUCTION_LABELS and score >= PRODUCTION_CONFIDENCE_THRESHOLD:
                        now = time.time()
                        if now - self.last_production_at >= PRODUCTION_DEDUP_SEC:
                            self.last_production_at = now
                            self.report_production(1)

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

                if status == "DEFECT":
                    self.send_status("DEFECT", label, f"Triton anomaly score={score:.2f}", "WARNING")
                else:
                    self.send_status("NORMAL", label, f"Triton recovered: {label} ({score:.2f})", "INFO")

                if PRODUCTION_LABELS and label.lower() in PRODUCTION_LABELS and score >= PRODUCTION_CONFIDENCE_THRESHOLD:
                        now = time.time()
                        if now - self.last_production_at >= PRODUCTION_DEDUP_SEC:
                            self.last_production_at = now
                            self.report_production(1)

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
    user_data.maybe_enqueue_frame(frame)
    return Gst.FlowReturn.OK


class GStreamerInferenceApp:
    def __init__(self, video_source: str):
        self.video_width = 640
        self.video_height = 480
        self.frame_rate = 30
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
        )

        self.pipeline = None
        self.loop = None
        self._demo_restarting = False
        self._demo_last_restart_at = 0.0

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

        encoder_pipeline = (
            "videoconvert ! "
            "video/x-raw,format=I420 ! "
            "x264enc "
            "tune=zerolatency "
            "speed-preset=faster "
            "bitrate=1000 "
            "vbv-buf-capacity=10000 "
            "key-int-max=30 "
            "bframes=0 "
            "threads=4 "
            "sliced-threads=true "
            "pass=cbr "
            "! "
            f"video/x-h264,profile={H264_PROFILE_ENV} ! "
        )

        if CLIP_ENABLED:
            segment_ns = int(CLIP_SEGMENT_SEC * 1_000_000_000)
            segment_location = os.path.join(CLIP_SEGMENTS_DIR, "segment_%05d.mp4")
            uplink_pipeline = (
                f"{encoder_pipeline}"
                "tee name=enc_t "
                "enc_t. ! queue ! h264parse config-interval=1 ! "
                f"rtph264pay name=webrtc_pay config-interval=1 pt=96 mtu=1200 ssrc={self.rtp_ssrc} ! "
                "application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000 ! "
                "webrtc_uplink. "
                "enc_t. ! queue ! h264parse config-interval=1 ! "
                f"splitmuxsink name=clip_sink muxer=mp4mux max-size-time={segment_ns} "
                f"max-files={CLIP_MAX_SEGMENTS} location=\"{segment_location}\""
            )
        else:
            uplink_pipeline = (
                f"{encoder_pipeline}"
                f"rtph264pay name=webrtc_pay config-interval=1 pt=96 mtu=1200 ssrc={self.rtp_ssrc} ! "
                "application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000 ! "
                "webrtc_uplink."
            )

        if LOCAL_DISPLAY:
            pipeline_string = (
                f"{source_pipeline} ! "
                "tee name=t "
                "t. ! queue ! "
                "appsink name=zsad_sink emit-signals=true max-buffers=1 drop=true sync=false "
                "t. ! queue ! "
                f"{overlay_pipeline}"
                "tee name=dt "
                "dt. ! queue ! "
                f"{uplink_pipeline} "
                "dt. ! queue ! videoconvert ! autovideosink sync=false"
            )
        else:
            pipeline_string = (
                f"{source_pipeline} ! "
                "tee name=t "
                "t. ! queue ! "
                "appsink name=zsad_sink emit-signals=true max-buffers=1 drop=true sync=false "
                "t. ! queue ! "
                f"{overlay_pipeline}"
                f"{uplink_pipeline}"
            )

        pipeline_string = f"{pipeline_string} webrtcbin name=webrtc_uplink bundle-policy=max-bundle latency=0"

        log.info("[PIPELINE] %s", pipeline_string)
        self.pipeline = Gst.parse_launch(pipeline_string)
        self.loop = GLib.MainLoop()

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

    def send_webrtc_signal(self, destination: str, payload: dict, remember: bool) -> bool:
        return enqueue_stomp_message(destination, payload, remember=remember)

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
            log.info("Starting signaling thread...")
            signaling_thread = threading.Thread(target=lambda: asyncio.run(signaling_client_main()), daemon=True)
            signaling_thread.start()

            log.info("Starting GStreamer main loop...")
            self.pipeline.set_state(Gst.State.PLAYING)
            try:
                self.loop.run()
            except KeyboardInterrupt:
                log.info("KeyboardInterrupt received.")
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
