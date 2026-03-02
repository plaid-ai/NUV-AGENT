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
from urllib.parse import urlparse

import numpy as np
from nuvion_app.config import load_env
import aiohttp
import websockets

# Python 3.14에서 third-party stomper 패키지의 legacy regex 문자열로
# SyntaxWarning(invalid escape sequence)가 발생한다. 런타임 동작에는 영향이 없어
# 해당 모듈 경고만 제한적으로 숨긴다.
warnings.filterwarnings(
    "ignore",
    message=r".*invalid escape sequence.*",
    category=SyntaxWarning,
    module=r"stomper\.stompbuffer",
)
import stomper

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

from nuvion_app.inference.zero_shot import ZeroShotAnomalyDetector
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

VIDEO_SOURCE_ENV = os.getenv("NUVION_VIDEO_SOURCE", "/dev/video0")
GST_SOURCE_OVERRIDE = os.getenv("NUVION_GST_SOURCE")

RTP_REMOTE_IP_ENV = os.getenv("NUVION_RTP_REMOTE_IP", "")
RTP_SSRC_ENV = os.getenv("NUVION_RTP_SSRC", None)
H264_PROFILE_LEVEL_ID_ENV = os.getenv("NUVION_H264_PROFILE_LEVEL_ID", "64001f")
H264_PROFILE_ENV = os.getenv("NUVION_H264_PROFILE", "baseline")
H264_PACKETIZATION_MODE_ENV = os.getenv("NUVION_H264_PACKETIZATION_MODE", "1")
H264_LEVEL_ASYMMETRY_ALLOWED_ENV = os.getenv("NUVION_H264_LEVEL_ASYMMETRY_ALLOWED", "1")

ANOMALY_LABELS = {label.lower() for label in parse_csv(os.getenv("NUVION_ANOMALY_LABELS", ""))}
PRODUCTION_LABELS = {label.lower() for label in parse_csv(os.getenv("NUVION_PRODUCTION_LABELS", ""))}

ANOMALY_CONFIDENCE_THRESHOLD = parse_float(os.getenv("NUVION_ANOMALY_CONFIDENCE_THRESHOLD"), 0.5)
PRODUCTION_CONFIDENCE_THRESHOLD = parse_float(os.getenv("NUVION_PRODUCTION_CONFIDENCE_THRESHOLD"), 0.5)
ANOMALY_MIN_INTERVAL_SEC = parse_float(os.getenv("NUVION_ANOMALY_MIN_INTERVAL_SEC"), 5.0)
PRODUCTION_DEDUP_SEC = parse_float(os.getenv("NUVION_PRODUCTION_DEDUP_SEC"), 3.0)

ZERO_SHOT_ENABLED = os.getenv("NUVION_ZERO_SHOT_ENABLED", "true").lower() in ("1", "true", "yes")
ZERO_SHOT_MODEL = os.getenv("NUVION_ZERO_SHOT_MODEL", "google/siglip2-base-patch16-224")
ZERO_SHOT_DEVICE = normalize_siglip_device(os.getenv("NUVION_ZERO_SHOT_DEVICE", "auto"), default="auto")
ZERO_SHOT_LABELS = parse_csv(os.getenv("NUVION_ZERO_SHOT_LABELS", "normal,defect"))
ZERO_SHOT_ANOMALY_LABELS = parse_csv(os.getenv("NUVION_ZERO_SHOT_ANOMALY_LABELS", "defect,broken,crack,scratch"))
ZERO_SHOT_THRESHOLD = parse_float(os.getenv("NUVION_ZERO_SHOT_THRESHOLD"), 0.7)
ZERO_SHOT_SAMPLE_SEC = parse_float(os.getenv("NUVION_ZERO_SHOT_SAMPLE_SEC"), 2.0)

LOCAL_DISPLAY = os.getenv("NUVION_LOCAL_DISPLAY", "false").lower() in ("1", "true", "yes")

TRITON_THRESHOLD = parse_float(os.getenv("NUVION_TRITON_THRESHOLD"), 0.7)
ZSAD_BACKEND = normalize_backend(os.getenv("NUVION_ZSAD_BACKEND", "triton"), default="triton")

CLIP_ENABLED = os.getenv("NUVION_CLIP_ENABLED", "true").lower() in ("1", "true", "yes")
CLIP_PRE_SEC = parse_float(os.getenv("NUVION_CLIP_PRE_SEC"), 5.0)
CLIP_POST_SEC = parse_float(os.getenv("NUVION_CLIP_POST_SEC"), 5.0)
CLIP_SEGMENT_SEC = parse_float(os.getenv("NUVION_CLIP_SEGMENT_SEC"), 1.0)
CLIP_MAX_SEGMENTS = int(os.getenv("NUVION_CLIP_MAX_SEGMENTS", "30"))
CLIP_OUTPUT_DIR = os.getenv("NUVION_CLIP_OUTPUT_DIR", "/tmp/nuvion_clips")
CLIP_COOLDOWN_SEC = parse_float(os.getenv("NUVION_CLIP_COOLDOWN_SEC"), 10.0)
CLIP_CONTENT_TYPE = os.getenv("NUVION_CLIP_CONTENT_TYPE", "video/mp4")

LINE_ID = parse_int(os.getenv("NUVION_LINE_ID"))
PROCESS_ID = parse_int(os.getenv("NUVION_PROCESS_ID"))

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
    "/app/broadcast/start",
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


def extract_host_from_server_url(url: str) -> str:
    parsed = urlparse(url)
    return parsed.hostname or "127.0.0.1"


def parse_rtp_sdp(sdp: str) -> tuple[str, int, int]:
    ip = "0.0.0.0"
    port = 5004
    pt = 96

    lines = [line.strip() for line in sdp.splitlines() if line.strip()]

    for line in lines:
        if line.startswith("c="):
            parts = line.split()
            if len(parts) >= 3:
                ip = parts[2]
            break

    for line in lines:
        if line.startswith("m=video"):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    port = int(parts[1])
                except ValueError:
                    port = 5004
                try:
                    pt = int(parts[3])
                except ValueError:
                    pt = 96
            break

    for line in lines:
        if line.startswith("a=rtpmap:") and "H264" in line:
            try:
                mid = line.split()[0]
                pt_str = mid.split(":")[1]
                pt = int(pt_str)
            except Exception:
                pass
            break

    log.info("[RTP] Parsed SDP: ip=%s, port=%s, pt=%s", ip, port, pt)
    return ip, port, pt


def get_rtp_ssrc() -> int:
    if RTP_SSRC_ENV:
        try:
            return int(RTP_SSRC_ENV)
        except ValueError:
            log.warning("[RTP] Invalid NUVION_RTP_SSRC='%s', using random ssrc.", RTP_SSRC_ENV)
    return random.randint(100000, 4294967295)


def build_rtp_parameters(payload_type: int, ssrc: int) -> dict:
    try:
        packetization_mode = int(H264_PACKETIZATION_MODE_ENV)
    except ValueError:
        packetization_mode = 1
    try:
        level_asymmetry_allowed = int(H264_LEVEL_ASYMMETRY_ALLOWED_ENV)
    except ValueError:
        level_asymmetry_allowed = 1

    return {
        "codecs": [
            {
                "mimeType": "video/H264",
                "payloadType": int(payload_type),
                "clockRate": 90000,
                "parameters": {
                    "packetization-mode": packetization_mode,
                    "profile-level-id": H264_PROFILE_LEVEL_ID_ENV,
                    "level-asymmetry-allowed": level_asymmetry_allowed,
                },
                "rtcpFeedback": [
                    {"type": "nack"},
                    {"type": "nack", "parameter": "pli"},
                    {"type": "ccm", "parameter": "fir"},
                    {"type": "goog-remb"},
                ],
            }
        ],
        "encodings": [
            {"ssrc": int(ssrc)}
        ],
        "headerExtensions": [],
        "rtcp": {
            "cname": f"nuvion-{DEVICE_USERNAME}",
            "reducedSize": True,
        },
    }


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


async def notify_broadcast_started(payload_type: int, ssrc: int):
    global websocket
    if not websocket:
        log.error("[RTP] Cannot notify broadcast started: WebSocket is None.")
        return

    destination = "/app/broadcast/start"
    if _is_agent_uplink_blocked(destination):
        return

    payload = {
        "broadcastId": DEVICE_USERNAME,
        "kind": "video",
        "rtpParameters": build_rtp_parameters(payload_type, ssrc),
    }
    _remember_last_payload(destination, payload)

    frame_str = build_send_frame(destination, payload)
    try:
        await websocket.send(json.dumps([frame_str]))
        log.info("[RTP] Notified server that broadcast has started.")
    except Exception as exc:
        log.warning("[RTP] Failed to notify broadcast started directly: %s", exc)
        enqueue_stomp_message(destination, payload, remember=False)


async def handle_rtp_endpoint_ready(body: str):
    global g_app

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return

    if data.get("type") != "RTP_ENDPOINT_READY":
        return

    broadcast_id = data.get("broadcastId")
    sdp = data.get("sdp")

    if broadcast_id and broadcast_id != DEVICE_USERNAME:
        return

    ip = data.get("ip")
    port = data.get("port")
    pt = data.get("payloadType")
    rtcp_port = data.get("rtcpPort")
    rtcp_mux = data.get("rtcpMux")
    comedia = data.get("comedia")

    if ip is None or port is None or pt is None:
        if not sdp:
            return
        sdp_ip, sdp_port, sdp_pt = parse_rtp_sdp(sdp)
        if ip is None:
            ip = sdp_ip
        if port is None:
            port = sdp_port
        if pt is None:
            pt = sdp_pt

    if RTP_REMOTE_IP_ENV:
        ip = RTP_REMOTE_IP_ENV
        log.info("[RTP] Override RTP IP via NUVION_RTP_REMOTE_IP: %s", ip)
    elif ip == "0.0.0.0":
        ip = extract_host_from_server_url(SERVER_BASE_URL)
        log.info("[RTP] Using fallback RTP IP: %s", ip)

    if not g_app:
        log.error("[RTP] GStreamer app is not initialized.")
        return

    log.info("[RTP] Update Sink -> ip=%s, port=%s, pt=%s, rtcpMux=%s, comedia=%s", ip, port, pt, rtcp_mux, comedia)
    g_app.configure_rtp_sink(ip, int(port), int(pt))

    await notify_broadcast_started(int(pt), g_app.rtp_ssrc)


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

                async for message in ws:
                    if not message.startswith("a["):
                        continue
                    frame_list = json.loads(message[1:])
                    for frame_str in frame_list:
                        frame = stomper.unpack_frame(frame_str)
                        destination = frame["headers"].get("destination")
                        body = frame["body"]

                        if destination and AGENT_COMMAND_QUEUE_DEST in destination:
                            await handle_rtp_endpoint_ready(body)
                        elif destination and AGENT_ERROR_QUEUE_DEST in destination:
                            await handle_agent_error(body)

        except Exception as exc:
            log.error("[SIGNALING] WebSocket error: %s", exc)
        finally:
            websocket = None
            if "sender_task" in locals():
                sender_task.cancel()

        log.info("[SIGNALING] Reconnecting in 10s...")
        await asyncio.sleep(10)


class NuvionEventState:
    def __init__(self, overlay_callback=None):
        self.running = True
        self.last_anomaly_at = 0.0
        self.last_production_at = 0.0
        self.zero_shot_last_sample = 0.0
        self.zero_shot_queue = queue.Queue(maxsize=1)
        self.overlay_callback = overlay_callback
        self.last_status = None
        self.last_sent_status = None
        self.last_sent_at = 0.0
        self.clip_enabled = CLIP_ENABLED
        self.clip_in_progress = False
        self.clip_last_started = 0.0
        self.clip_lock = threading.Lock()

        self.backend = ZSAD_BACKEND
        self.zero_shot = None
        self.triton_client = None

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
                self.triton_client = TritonAnomalyClient()
        else:
            self.backend = "none"

        self.worker_thread = threading.Thread(target=self._zsad_worker, daemon=True)
        self.worker_thread.start()

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

        payload = {
            "anomalyType": anomaly_type,
            "anomalyStatus": status,
            "message": message,
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
            log.info("[ZSAD] Sent %s status (change): %s", status, message)
        else:
            log.info("[ZSAD] Sent %s status (repeat): %s", status, message)

    def start_clip_upload(self) -> str | None:
        if not self.clip_enabled or not CLIP_ENABLED:
            return None
        now = time.time()
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
        return segments

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
        if not segments:
            log.warning("[CLIP] No segments available for clip.")
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
        if self.backend == "none":
            return
        now = time.time()
        if now - self.zero_shot_last_sample < ZERO_SHOT_SAMPLE_SEC:
            return
        self.zero_shot_last_sample = now

        if self.zero_shot_queue.full():
            return
        try:
            self.zero_shot_queue.put_nowait(frame_rgb)
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
                    self._emit_overlay(f"{status} {label} {score:.2f}")
                    if status == "DEFECT":
                        self.send_status("DEFECT", label, f"Zero-shot anomaly: {label} ({score:.2f})", "WARNING")
                    else:
                        self.send_status("NORMAL", label, f"Recovered to normal: {label} ({score:.2f})", "INFO")

                    if PRODUCTION_LABELS and label.lower() in PRODUCTION_LABELS and score >= PRODUCTION_CONFIDENCE_THRESHOLD:
                        now = time.time()
                        if now - self.last_production_at >= PRODUCTION_DEDUP_SEC:
                            self.last_production_at = now
                            self.report_production(1)

            elif self.backend == "triton" and self.triton_client:
                try:
                    result = self.triton_client.predict(frame)
                except Exception as exc:
                    log.warning("[TRITON] inference failed: %s", exc)
                    continue

                if result is None:
                    continue

                label = result.get("label", "ZSAD")
                score = float(result.get("score", 0.0))
                is_anomaly = score >= TRITON_THRESHOLD
                status = "DEFECT" if is_anomaly else "NORMAL"
                self._emit_overlay(f"{status} {label} {score:.2f}")

                if status == "DEFECT":
                    self.send_status("DEFECT", label, f"Triton anomaly score={score:.2f}", "WARNING")
                else:
                    self.send_status("NORMAL", label, f"Triton recovered: {label} ({score:.2f})", "INFO")

                if PRODUCTION_LABELS and label.lower() in PRODUCTION_LABELS and score >= PRODUCTION_CONFIDENCE_THRESHOLD:
                    now = time.time()
                    if now - self.last_production_at >= PRODUCTION_DEDUP_SEC:
                        self.last_production_at = now
                        self.report_production(1)


def build_source_pipeline(video_source: str, width: int, height: int, fps: int) -> str:
    if GST_SOURCE_OVERRIDE:
        return GST_SOURCE_OVERRIDE

    if not video_source or video_source == "auto":
        video_source = "avf" if sys.platform == "darwin" else "/dev/video0"

    if video_source.startswith("/dev/video"):
        if sys.platform == "darwin":
            source = "avfvideosrc"
        else:
            source = f"v4l2src device={video_source}"
    elif video_source.lower() in {"rpi", "libcamera"}:
        source = "libcamerasrc"
    elif video_source.lower().startswith(("avf", "avfoundation", "mac")):
        device_index = None
        if ":" in video_source:
            _, maybe_index = video_source.split(":", 1)
            if maybe_index.isdigit():
                device_index = int(maybe_index)
        source = f"avfvideosrc device-index={device_index}" if device_index is not None else "avfvideosrc"
    else:
        source = "autovideosrc"

    return (
        f"{source} ! "
        f"video/x-raw,width={width},height={height},framerate={fps}/1 ! "
        "videoconvert ! "
        "video/x-raw,format=RGB"
    )


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
    user_data.maybe_enqueue_frame(frame)
    return Gst.FlowReturn.OK


class GStreamerInferenceApp:
    def __init__(self, video_source: str):
        self.video_width = 640
        self.video_height = 480
        self.frame_rate = 30
        self.video_source = video_source
        self.rtp_ssrc = get_rtp_ssrc()
        self.overlay = None
        self.user_data = NuvionEventState(self.update_overlay_text)

        self.pipeline = None
        self.loop = None

        self.create_pipeline()

        global g_app
        g_app = self

    def create_pipeline(self):
        Gst.init(None)
        source_pipeline = build_source_pipeline(self.video_source, self.video_width, self.video_height, self.frame_rate)

        overlay_pipeline = (
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
            "bitrate=8000 "
            "vbv-buf-capacity=12000 "
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
            rtp_pipeline = (
                f"{encoder_pipeline}"
                "tee name=enc_t "
                "enc_t. ! queue ! "
                f"rtph264pay name=rtp_pay config-interval=1 pt=96 mtu=1200 ssrc={self.rtp_ssrc} ! "
                "udpsink name=rtp_sink host=0.0.0.0 port=5004 async=false sync=false "
                "enc_t. ! queue ! h264parse config-interval=1 ! "
                f"splitmuxsink name=clip_sink muxer=mp4mux max-size-time={segment_ns} "
                f"max-files={CLIP_MAX_SEGMENTS} location=\"{segment_location}\""
            )
        else:
            rtp_pipeline = (
                f"{encoder_pipeline}"
                f"rtph264pay name=rtp_pay config-interval=1 pt=96 mtu=1200 ssrc={self.rtp_ssrc} ! "
                "udpsink name=rtp_sink host=0.0.0.0 port=5004 async=false sync=false"
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
                f"{rtp_pipeline} "
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
                f"{rtp_pipeline}"
            )

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
            self.update_overlay_text(self._default_overlay_text())

    def bus_call(self, bus, message, loop):
        msg_type = message.type
        if msg_type == Gst.MessageType.EOS:
            log.info("End-of-stream")
            self.shutdown()
        elif msg_type == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            log.error("GStreamer Error: %s, %s", err, dbg)
            self.shutdown()
        return True

    def configure_rtp_sink(self, host: str, port: int, pt: int):
        if not self.pipeline:
            log.error("[RTP] Pipeline not initialized.")
            return

        rtp_sink = self.pipeline.get_by_name("rtp_sink")
        rtp_pay = self.pipeline.get_by_name("rtp_pay")

        if rtp_sink and rtp_pay:
            log.info("[RTP] Reconfiguring -> host=%s, port=%s, pt=%s", host, port, pt)
            rtp_sink.set_property("host", host)
            rtp_sink.set_property("port", port)
            rtp_pay.set_property("pt", pt)
            rtp_pay.set_property("ssrc", self.rtp_ssrc)

    def update_overlay_text(self, text: str):
        if not self.overlay:
            return
        def _apply():
            if self.overlay:
                self.overlay.set_property("text", text)
            return False
        GLib.idle_add(_apply)

    def _default_overlay_text(self) -> str:
        backend = getattr(self.user_data, "backend", "none")
        if backend == "triton":
            return "ZSAD TRITON ON"
        if backend == "siglip":
            return "ZSAD ON"
        return "ZSAD OFF"

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
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        if self.loop and self.loop.is_running():
            self.loop.quit()


def main():
    video_source = os.getenv("NUVION_VIDEO_SOURCE", "/dev/video0")
    app = GStreamerInferenceApp(video_source)
    app.run()


if __name__ == "__main__":
    main()
