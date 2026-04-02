from __future__ import annotations

import html
import json
import os
import platform
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import urllib.error
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import dotenv_values, load_dotenv
from nuvion_app.inference.demo_mvtec import validate_mvtec_demo_settings
DEFAULT_PORT = 8088
SECRET_KEY_MARKERS = ("PASSWORD",)
DEVICE_TYPE = "NUV_AGENT"
PAIRING_POLL_INTERVAL_SEC = int(os.getenv("NUVION_PAIRING_POLL_INTERVAL_SEC", "5"))
PAIRING_TIMEOUT_SEC = int(os.getenv("NUVION_PAIRING_TIMEOUT_SEC", "600"))
PROVISION_ENDPOINT = os.getenv("NUVION_DEVICE_PROVISION_ENDPOINT", "/devices/provision")
PAIRING_INIT_ENDPOINT = os.getenv("NUVION_PAIRING_INIT_ENDPOINT", "/devices/pairings/init")
PAIRING_STATUS_ENDPOINT = os.getenv("NUVION_PAIRING_STATUS_ENDPOINT", "/devices/pairings/{pairing_id}")
BASE_REQUIRED_KEYS = {
    "NUVION_SERVER_BASE_URL",
    "NUVION_DEVICE_USERNAME",
    "NUVION_DEVICE_PASSWORD",
}
PLACEHOLDER_VALUES = {"***"}
ADVANCED_SECTION_ORDER = [
    "streaming",
    "detection",
    "connectivity",
    "clips",
    "runtime",
]
ADVANCED_SECTION_META = {
    "streaming": (
        "Streaming & Video",
        "H264, relay fallback, and camera-related transport settings.",
    ),
    "detection": (
        "Detection & Labels",
        "Inference thresholds, labels, and optional line/process overrides.",
    ),
    "connectivity": (
        "Health & Connectivity",
        "Heartbeat cadence and network quality reporting.",
    ),
    "clips": (
        "Clip Capture",
        "Pre/post buffer recording and ffmpeg-based clip upload.",
    ),
    "runtime": (
        "Runtime & Models",
        "Bootstrap, Triton, model bundle, and local runtime behavior.",
    ),
}

_LOADED = False
_LOADED_PATH: Optional[Path] = None


def _video_source_sort_key(value: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", value)
    if match:
        return int(match.group(1)), value
    return 10_000, value


def _parse_gst_device_monitor_output(output: str) -> List[Dict[str, str]]:
    options: List[Dict[str, str]] = []
    for block in output.split("Device found:"):
        name_match = re.search(r"^\s*name\s*:\s*(.+)$", block, re.MULTILINE)
        launch_match = re.search(r"gst-launch-1\.0\s+avfvideosrc(?:\s+device-index=(\d+))?", block)
        if not name_match or not launch_match:
            continue
        name = name_match.group(1).strip()
        device_index = launch_match.group(1)
        if device_index is None:
            continue
        options.append(
            {
                "value": f"avf:{device_index}",
                "label": name,
                "detail": f"macOS camera #{device_index}",
            }
        )
    return options


def _is_jetson_platform() -> bool:
    return Path("/etc/nv_tegra_release").exists()


def _gst_element_available(element_name: str) -> bool:
    try:
        result = subprocess.run(
            ["gst-inspect-1.0", element_name],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def discover_video_source_options(platform_name: Optional[str] = None) -> List[Dict[str, str]]:
    current_platform = platform_name or sys.platform
    options: List[Dict[str, str]] = []

    if current_platform == "darwin":
        options.append(
            {
                "value": "avf",
                "label": "Default macOS camera",
                "detail": "Uses AVFoundation default device.",
            }
        )
        try:
            result = subprocess.run(
                ["gst-device-monitor-1.0", "Video/Source"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            options.extend(_parse_gst_device_monitor_output(result.stdout))
        except (FileNotFoundError, subprocess.SubprocessError):
            pass
        return options

    video_devices = sorted(Path("/dev").glob("video*"), key=lambda path: _video_source_sort_key(path.name))
    for device in video_devices:
        options.append(
            {
                "value": str(device),
                "label": device.name,
                "detail": "Linux V4L2 camera device",
            }
        )

    if current_platform.startswith("linux"):
        if _is_jetson_platform() and _gst_element_available("nvarguscamerasrc"):
            options.insert(
                0,
                {
                    "value": "jetson",
                    "label": "Jetson CSI camera",
                    "detail": "Uses nvarguscamerasrc.",
                },
            )
        options.append(
            {
                "value": "rpi",
                "label": "Raspberry Pi camera",
                "detail": "Uses libcamerasrc.",
            }
        )

    return options


def _prompt_video_source(default: str, platform_name: Optional[str] = None) -> str:
    options = discover_video_source_options(platform_name)
    if options:
        print("Available camera sources:")
        for index, option in enumerate(options, start=1):
            suffix = " (current)" if option["value"] == default else ""
            detail = f" - {option['detail']}" if option.get("detail") else ""
            print(f"  {index}. {option['label']} [{option['value']}] {detail}{suffix}")
        print("Enter a number to select a camera, or type a custom source value.")

    prompt = "Camera source (NUVION_VIDEO_SOURCE)"
    if default:
        prompt += f" [{default}]"
    prompt += ": "

    while True:
        entered = input(prompt).strip()
        if not entered:
            return default
        if entered.isdigit():
            selected_index = int(entered) - 1
            if 0 <= selected_index < len(options):
                return options[selected_index]["value"]
            print("Invalid selection number.")
            continue
        return entered


def _is_placeholder(value: Optional[str]) -> bool:
    if value is None:
        return True
    stripped = value.strip()
    if not stripped:
        return True
    if stripped in PLACEHOLDER_VALUES:
        return True
    if stripped.startswith("<") and stripped.endswith(">"):
        return True
    return False


def _is_secret_key(key: str) -> bool:
    return any(marker in key for marker in SECRET_KEY_MARKERS)


def _is_basic_setup_field(key: str) -> bool:
    return key in {
        "NUVION_SERVER_BASE_URL",
        "NUVION_DEVICE_USERNAME",
        "NUVION_DEVICE_PASSWORD",
        "NUVION_VIDEO_SOURCE",
        "NUVION_VIDEO_ROTATION",
        "NUVION_VIDEO_FLIP_HORIZONTAL",
        "NUVION_VIDEO_FLIP_VERTICAL",
        "NUVION_FACE_TRACKING_ENABLED",
        "NUVION_FACE_TRACKING_SHOW_BBOX",
        "NUVION_MOTOR_ENABLED",
        "NUVION_DEMO_MODE",
    }


def _field_section(key: str) -> str:
    if key.startswith("NUVION_CLIP_"):
        return "clips"
    if key.startswith("NUVION_CONNECTIVITY_") or key == "NUVION_DEVICE_STATE_INTERVAL_SEC":
        return "connectivity"
    if key.startswith("NUVION_ZERO_SHOT_") or key.startswith("NUVION_FACE_TRACKING_"):
        return "detection"
    if key.startswith("NUVION_MOTOR_"):
        return "runtime"
    if key.startswith("NUVION_TRITON_"):
        return "runtime"
    if key.startswith("NUVION_MODEL_") or key.startswith("NUVION_RUNTIME_") or key.startswith("NUVION_BOOTSTRAP_"):
        return "runtime"
    if key.startswith("NUVION_DOCKER_") or key.startswith("NUVION_HOMEBREW_"):
        return "runtime"
    if key.startswith("NUVION_ANOMALY_") or key.startswith("NUVION_PRODUCTION_"):
        return "detection"
    if key.startswith("NUVION_H264_") or key.startswith("NUVION_WEBRTC_") or key.startswith("NUVION_VIDEO_"):
        return "streaming"
    if key in {"NUVION_LINE_ID", "NUVION_PROCESS_ID"}:
        return "detection"
    return "runtime"


def _field_note(key: str) -> str:
    notes = {
        "NUVION_DEVICE_USERNAME": "Usually auto-filled by Auto Provision. You rarely need to type this manually.",
        "NUVION_DEVICE_PASSWORD": "Usually auto-filled by Auto Provision. Leave blank on save to keep the current secret.",
        "NUVION_DEMO_MODE": "Turn this on only when testing without a real camera.",
        "NUVION_VIDEO_SOURCE": "Use auto-detect unless you need to force a specific USB/CSI source.",
        "NUVION_VIDEO_ROTATION": "Allowed values: 0, 90, 180, 270.",
        "NUVION_VIDEO_FLIP_HORIZONTAL": "Mirror the image left-to-right after source capture.",
        "NUVION_VIDEO_FLIP_VERTICAL": "Flip the image upside-down after source capture.",
        "NUVION_FACE_TRACKING_ENABLED": "Track the face closest to the frame center and optionally steer the motor.",
        "NUVION_FACE_TRACKING_SHOW_BBOX": "Draw detected face boxes and deadzone overlay on the stream.",
        "NUVION_MOTOR_ENABLED": "Enable motor control if the device has a supported motor backend.",
        "NUVION_MOTOR_BACKEND": "Allowed values: auto, uart, pwm, none.",
        "NUVION_MOTOR_UART_PORT": "UART serial device for the external motor controller.",
        "NUVION_MOTOR_UART_BAUD": "Serial baud rate for UART motor control.",
        "NUVION_MOTOR_PAN_INVERT": "Invert left/right commands when motor wiring is reversed.",
        "NUVION_MOTOR_TILT_INVERT": "Invert up/down commands when motor wiring is reversed.",
        "NUVION_WEBRTC_FORCE_RELAY": "Fallback only. Live sessions can be overridden by the backend, so most users should leave this as-is.",
        "NUVION_CLIP_ENABLED": "Stores short video evidence around anomaly events. Disable only when debugging clip-related issues.",
    }
    return notes.get(key, "")


def _is_rotation_field(key: str) -> bool:
    return key == "NUVION_VIDEO_ROTATION"


def _is_motor_backend_field(key: str) -> bool:
    return key == "NUVION_MOTOR_BACKEND"


def _prompt_boolean_setting(label: str, key: str, default: str) -> str:
    normalized_default = "true" if _is_truthy(default) else "false"
    prompt = f"{label} ({key}) ["
    prompt += "On" if normalized_default == "true" else "Off"
    prompt += "]: "

    while True:
        entered = input(prompt).strip().lower()
        if not entered:
            return normalized_default
        if entered in {"1", "true", "yes", "y", "on"}:
            return "true"
        if entered in {"0", "false", "no", "n", "off"}:
            return "false"
        print("Enter on/off, yes/no, or true/false.")


def _prompt_rotation_setting(label: str, key: str, default: str) -> str:
    options = ("0", "90", "180", "270")
    prompt = f"{label} ({key}) [{default or '0'}]: "
    while True:
        entered = input(prompt).strip()
        if not entered:
            return default or "0"
        if entered in options:
            return entered
        print("Enter one of: 0, 90, 180, 270.")


def _prompt_choice_setting(label: str, key: str, default: str, choices: tuple[str, ...]) -> str:
    prompt = f"{label} ({key}) [{default or choices[0]}]: "
    while True:
        entered = input(prompt).strip().lower()
        if not entered:
            return default or choices[0]
        if entered in choices:
            return entered
        print(f"Enter one of: {', '.join(choices)}.")


def _parse_int_or_default(value: str, default: int) -> int:
    try:
        parsed = int(str(value).strip())
    except Exception:
        return default
    return parsed if parsed > 0 else default


def _parse_float_or_default(value: str, default: float) -> float:
    try:
        parsed = float(str(value).strip())
    except Exception:
        return default
    return parsed if parsed > 0 else default


def _build_motor_config_from_values(values: Dict[str, str]):
    from nuvion_app.inference.motor import MotorConfig

    return MotorConfig(
        enabled=_is_truthy(values.get("NUVION_MOTOR_ENABLED", "false")),
        backend=(values.get("NUVION_MOTOR_BACKEND") or "auto").strip().lower() or "auto",
        uart_port=(values.get("NUVION_MOTOR_UART_PORT") or "/dev/ttyTHS1").strip() or "/dev/ttyTHS1",
        uart_baud=_parse_int_or_default(values.get("NUVION_MOTOR_UART_BAUD", "115200"), 115200),
        uart_timeout_sec=_parse_float_or_default(values.get("NUVION_MOTOR_UART_TIMEOUT_SEC", "1.0"), 1.0),
        pan_invert=_is_truthy(values.get("NUVION_MOTOR_PAN_INVERT", "false")),
        tilt_invert=_is_truthy(values.get("NUVION_MOTOR_TILT_INVERT", "false")),
        command_interval_sec=_parse_float_or_default(values.get("NUVION_MOTOR_COMMAND_INTERVAL_SEC", "0.1"), 0.1),
    )


def _prompt_camera_setup(fields: List[Dict[str, str]], existing: Dict[str, str]) -> Dict[str, str]:
    values = dict(existing)
    camera_fields = {
        "NUVION_VIDEO_SOURCE",
        "NUVION_VIDEO_ROTATION",
        "NUVION_VIDEO_FLIP_HORIZONTAL",
        "NUVION_VIDEO_FLIP_VERTICAL",
    }

    print("Camera setup (press Enter to keep current value)")
    for field in fields:
        key = field["key"]
        if key not in camera_fields:
            continue

        default = values.get(key, field["default"])
        label = field["comment"] or key

        if key == "NUVION_VIDEO_SOURCE":
            values[key] = _prompt_video_source(default)
            continue
        if _is_rotation_field(key):
            values[key] = _prompt_rotation_setting(label, key, default)
            continue
        if _is_boolean_like(field["default"], default):
            values[key] = _prompt_boolean_setting(label, key, default)
            continue

    return values


def _maybe_run_motor_test(values: Dict[str, str]) -> None:
    from nuvion_app.inference.motor import MotorController, run_motor_test

    answer = input("Run motor test now? [y/N]: ").strip().lower()
    if answer not in {"y", "yes"}:
        return

    controller = MotorController(_build_motor_config_from_values(values))
    run_motor_test(controller)


def _prompt_tracking_motor_setup(fields: List[Dict[str, str]], existing: Dict[str, str]) -> Dict[str, str]:
    values = dict(existing)
    field_map = {field["key"]: field for field in fields}

    print("Tracking + motor setup (press Enter to keep current value)")
    values["NUVION_FACE_TRACKING_ENABLED"] = _prompt_boolean_setting(
        field_map["NUVION_FACE_TRACKING_ENABLED"]["comment"] or "Face tracking",
        "NUVION_FACE_TRACKING_ENABLED",
        values.get("NUVION_FACE_TRACKING_ENABLED", field_map["NUVION_FACE_TRACKING_ENABLED"]["default"]),
    )

    if _is_truthy(values.get("NUVION_FACE_TRACKING_ENABLED", "false")):
        values["NUVION_FACE_TRACKING_SHOW_BBOX"] = _prompt_boolean_setting(
            field_map["NUVION_FACE_TRACKING_SHOW_BBOX"]["comment"] or "Show tracking bbox",
            "NUVION_FACE_TRACKING_SHOW_BBOX",
            values.get("NUVION_FACE_TRACKING_SHOW_BBOX", field_map["NUVION_FACE_TRACKING_SHOW_BBOX"]["default"]),
        )

    values["NUVION_MOTOR_ENABLED"] = _prompt_boolean_setting(
        field_map["NUVION_MOTOR_ENABLED"]["comment"] or "Motor enabled",
        "NUVION_MOTOR_ENABLED",
        values.get("NUVION_MOTOR_ENABLED", field_map["NUVION_MOTOR_ENABLED"]["default"]),
    )

    if not _is_truthy(values.get("NUVION_MOTOR_ENABLED", "false")):
        return values

    values["NUVION_MOTOR_BACKEND"] = _prompt_choice_setting(
        field_map["NUVION_MOTOR_BACKEND"]["comment"] or "Motor backend",
        "NUVION_MOTOR_BACKEND",
        values.get("NUVION_MOTOR_BACKEND", field_map["NUVION_MOTOR_BACKEND"]["default"]),
        ("auto", "uart", "pwm", "none"),
    )

    for key in (
        "NUVION_MOTOR_UART_PORT",
        "NUVION_MOTOR_UART_BAUD",
        "NUVION_MOTOR_UART_TIMEOUT_SEC",
        "NUVION_MOTOR_PAN_INVERT",
        "NUVION_MOTOR_TILT_INVERT",
    ):
        field = field_map[key]
        default = values.get(key, field["default"])
        if _is_boolean_like(field["default"], default):
            values[key] = _prompt_boolean_setting(field["comment"] or key, key, default)
            continue
        if _is_motor_backend_field(key):
            values[key] = _prompt_choice_setting(field["comment"] or key, key, default, ("auto", "uart", "pwm", "none"))
            continue
        prompt = f"{field['comment'] or key} ({key})"
        if default:
            prompt += f" [{default}]"
        prompt += ": "
        entered = input(prompt).strip()
        values[key] = entered or default

    _maybe_run_motor_test(values)
    return values


def _is_boolean_like(default_value: str, value: str) -> bool:
    allowed = {"true", "false"}
    default_norm = (default_value or "").strip().lower()
    value_norm = (value or "").strip().lower()
    return default_norm in allowed and (not value_norm or value_norm in allowed)


def effective_required_keys(values: Optional[Dict[str, str]] = None) -> set[str]:
    _ = values
    return set(BASE_REQUIRED_KEYS)


def template_path() -> Path:
    path = Path(__file__).resolve().parent / "config_template.env"
    if path.exists():
        return path
    fallback = Path(__file__).resolve().parents[1] / ".env.example"
    return fallback


def load_template() -> Tuple[List[str], List[Dict[str, str]]]:
    lines = template_path().read_text().splitlines()
    fields: List[Dict[str, str]] = []
    pending_comment: Optional[str] = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            pending_comment = None
            continue
        if stripped.startswith("#"):
            pending_comment = stripped.lstrip("#").strip()
            continue
        if "=" not in line:
            pending_comment = None
            continue
        key, default = line.split("=", 1)
        fields.append(
            {
                "key": key.strip(),
                "default": default.strip(),
                "comment": pending_comment or "",
            }
        )
        pending_comment = None
    return lines, fields


def _find_repo_env(start: Path) -> Optional[Path]:
    for parent in [start, *start.parents]:
        if (parent / ".env").exists() and (parent / "nuvion_app").is_dir():
            return parent / ".env"
    return None


def _default_system_paths() -> List[Path]:
    if sys.platform == "darwin":
        paths: List[Path] = []
        opt_homebrew = Path("/opt/homebrew/etc/nuv-agent/agent.env")
        usr_local = Path("/usr/local/etc/nuv-agent/agent.env")
        if opt_homebrew.exists() or Path("/opt/homebrew").exists():
            paths.append(opt_homebrew)
            paths.append(usr_local)
        else:
            paths.append(usr_local)
            paths.append(opt_homebrew)
        paths.append(Path("/etc/nuv-agent/agent.env"))
        return paths
    return [Path("/etc/nuv-agent/agent.env")]


def resolve_config_path(explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    env_path = os.getenv("NUV_AGENT_CONFIG")
    if env_path:
        return Path(env_path).expanduser()

    repo_env = _find_repo_env(Path.cwd())
    if repo_env:
        return repo_env

    candidates = _default_system_paths()
    for path in candidates:
        try:
            if path.exists():
                return path
        except PermissionError:
            continue
    return candidates[0]


def load_env(path: Optional[str] = None) -> Path:
    global _LOADED
    global _LOADED_PATH
    config_path = resolve_config_path(path)
    if _LOADED and _LOADED_PATH == config_path:
        return config_path

    override = _LOADED and _LOADED_PATH is not None and _LOADED_PATH != config_path
    os.environ["NUV_AGENT_CONFIG"] = str(config_path)
    load_dotenv(config_path, override=override)

    _LOADED = True
    _LOADED_PATH = config_path
    return config_path


def read_env(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    values = dotenv_values(path)
    return {key: value for key, value in values.items() if value is not None}


def render_env(lines: List[str], values: Dict[str, str]) -> str:
    rendered: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            rendered.append(line)
            continue
        key = line.split("=", 1)[0].strip()
        value = values.get(key, "")
        rendered.append(f"{key}={value}")
    return "\n".join(rendered) + "\n"


def write_env(path: Path, lines: List[str], values: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = render_env(lines, values)
    path.write_text(content)


def _normalize_base_url(url: str) -> str:
    return url.rstrip("/")


def _request_json(
    url: str,
    method: str = "POST",
    payload: Optional[Dict[str, object]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 10,
) -> Dict[str, object] | None:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if headers:
        for key, value in headers.items():
            req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8")
        except Exception:
            pass
        return {"error": f"{exc.code} {exc.reason}", "details": detail}
    except Exception as exc:
        return {"error": str(exc)}


def _extract_data(response: Optional[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if not response:
        return None
    if "data" in response and isinstance(response["data"], dict):
        return response["data"]  # type: ignore[return-value]
    return response


def _extract_list(response: Optional[Dict[str, object]] | List[object]) -> Optional[List[object]]:
    if not response:
        return None
    if isinstance(response, list):
        return response
    data = response.get("data")
    if isinstance(data, list):
        return data
    return None


def _login_user(server_base_url: str, username: str, password: str) -> Optional[str]:
    url = f"{_normalize_base_url(server_base_url)}/auth/login"
    payload = {"username": username, "password": password}
    response = _request_json(url, payload=payload)
    data = _extract_data(response)
    if not data:
        return None
    token = data.get("accessToken") or data.get("token")
    return token if isinstance(token, str) else None


def _provision_device(
    server_base_url: str,
    username: str,
    password: str,
    space_id: str,
    device_name: str,
) -> Optional[Dict[str, object]]:
    token = _login_user(server_base_url, username, password)
    if not token:
        return None
    payload: Dict[str, object] = {
        "spaceId": space_id,
        "deviceName": device_name,
        "deviceType": DEVICE_TYPE,
        "model": platform.machine(),
        "os": platform.system(),
    }
    url = f"{_normalize_base_url(server_base_url)}{PROVISION_ENDPOINT}"
    response = _request_json(url, payload=payload, headers={"Authorization": f"Bearer {token}"})
    return _extract_data(response)


def _fetch_spaces(
    server_base_url: str,
    username: str,
    password: str,
) -> Optional[List[object]]:
    token = _login_user(server_base_url, username, password)
    if not token:
        return None
    url = f"{_normalize_base_url(server_base_url)}/spaces/me"
    response = _request_json(url, method="GET", headers={"Authorization": f"Bearer {token}"})
    return _extract_list(response)


def _init_pairing(server_base_url: str, device_name: str) -> Optional[Dict[str, object]]:
    payload: Dict[str, object] = {
        "deviceName": device_name,
        "deviceType": DEVICE_TYPE,
        "model": platform.machine(),
        "os": platform.system(),
    }
    base_url = _normalize_base_url(server_base_url)
    url = f"{base_url}{PAIRING_INIT_ENDPOINT}"
    response = _request_json(url, payload=payload)
    return _extract_data(response)


def _wait_for_pairing(
    server_base_url: str,
    pairing_id: str,
    pairing_secret: Optional[str],
) -> Optional[Dict[str, object]]:
    base_url = _normalize_base_url(server_base_url)
    deadline = time.time() + PAIRING_TIMEOUT_SEC
    headers = {}
    if pairing_secret:
        headers["X-Pairing-Secret"] = pairing_secret
    status_url = f"{base_url}{PAIRING_STATUS_ENDPOINT.format(pairing_id=pairing_id)}"
    while time.time() < deadline:
        response = _request_json(status_url, method="GET", headers=headers)
        data = _extract_data(response)
        if data:
            status = str(data.get("status") or data.get("state") or "").upper()
            if status in {"ISSUED", "READY", "APPROVED", "ACTIVE"}:
                return data
            if status in {"EXPIRED", "REJECTED"}:
                return None
        time.sleep(PAIRING_POLL_INTERVAL_SEC)
    return None


def _print_qr(url: str) -> None:
    print("Pairing URL:", url)
    try:
        import qrcode  # type: ignore

        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
    except Exception:
        print("Install 'qrcode' to render QR in terminal.")


def _merge_defaults(fields: List[Dict[str, str]], existing: Dict[str, str]) -> Dict[str, str]:
    merged: Dict[str, str] = dict(existing)
    for field in fields:
        key = field["key"]
        if key not in merged:
            merged[key] = field["default"]
    return merged


def _validate_required(values: Dict[str, str]) -> List[str]:
    missing = []
    for key in effective_required_keys(values):
        if _is_placeholder(values.get(key)):
            missing.append(key)
    return missing


def prompt_cli(fields: List[Dict[str, str]], existing: Dict[str, str], advanced: bool) -> Dict[str, str]:
    values = _merge_defaults(fields, existing)
    values = _prompt_camera_setup(fields, values)
    values = _prompt_tracking_motor_setup(fields, values)
    required_keys = effective_required_keys(values)
    handled_keys = {
        "NUVION_VIDEO_SOURCE",
        "NUVION_VIDEO_ROTATION",
        "NUVION_VIDEO_FLIP_HORIZONTAL",
        "NUVION_VIDEO_FLIP_VERTICAL",
        "NUVION_FACE_TRACKING_ENABLED",
        "NUVION_FACE_TRACKING_SHOW_BBOX",
        "NUVION_MOTOR_ENABLED",
        "NUVION_MOTOR_BACKEND",
        "NUVION_MOTOR_UART_PORT",
        "NUVION_MOTOR_UART_BAUD",
        "NUVION_MOTOR_UART_TIMEOUT_SEC",
        "NUVION_MOTOR_PAN_INVERT",
        "NUVION_MOTOR_TILT_INVERT",
    }
    for field in fields:
        key = field["key"]
        if key in handled_keys:
            continue
        default = values.get(key, "")
        required = key in required_keys
        basic = _is_basic_setup_field(key)
        if not advanced and not required and not basic:
            continue

        label = field["comment"] or key
        prompt = f"{label} ({key})"
        if default:
            prompt += f" [{default}]"
        prompt += ": "

        while True:
            if _is_secret_key(key):
                import getpass

                entered = getpass.getpass(prompt)
            else:
                entered = input(prompt)

            if not entered:
                entered = default
            values[key] = entered

            if required and _is_placeholder(values.get(key)):
                print("A value is required.")
                continue
            break

    return values


def _has_display() -> bool:
    if sys.platform == "darwin":
        return True
    return bool(os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY"))


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _field_group(key: str) -> str:
    if key.startswith("NUVION_TRITON_"):
        return "triton"
    if key.startswith("NUVION_ZERO_SHOT_"):
        return "siglip"
    return "general"


def _collect_env_overrides(fields: List[Dict[str, str]], values: Dict[str, str]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for field in fields:
        key = field["key"]
        env_value = os.getenv(key)
        if env_value is None:
            continue
        file_value = values.get(key, "")
        if env_value != file_value:
            overrides[key] = env_value
    return overrides


def _parse_triton_health_url(triton_url: str) -> str:
    candidate = (triton_url or "localhost:8000").strip()
    if "://" not in candidate:
        candidate = f"http://{candidate}"
    parsed = urllib.parse.urlparse(candidate)
    scheme = parsed.scheme or "http"
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if scheme == "https" else 8000)
    return f"{scheme}://{host}:{port}/v2/health/ready"


def _check_server_login(values: Dict[str, str]) -> Dict[str, str]:
    base_url = (values.get("NUVION_SERVER_BASE_URL") or "").strip()
    username = (values.get("NUVION_DEVICE_USERNAME") or "").strip()
    password = values.get("NUVION_DEVICE_PASSWORD") or ""
    if not base_url or not username or not password or _is_placeholder(password):
        return {
            "name": "Server login",
            "status": "warn",
            "detail": "NUVION_SERVER_BASE_URL / device credentials are required.",
        }
    token = _login_user(base_url, username, password)
    if token:
        return {
            "name": "Server login",
            "status": "pass",
            "detail": "Device credentials can obtain auth token.",
        }
    return {
        "name": "Server login",
        "status": "fail",
        "detail": "Failed to login with NUVION_DEVICE_USERNAME/NUVION_DEVICE_PASSWORD.",
    }


def _check_triton_health(values: Dict[str, str]) -> Dict[str, str]:
    backend = (values.get("NUVION_ZSAD_BACKEND") or "triton").strip().lower()
    if backend not in {"triton"}:
        return {
            "name": "Triton health",
            "status": "skip",
            "detail": f"Skipped because backend={backend}.",
        }
    health_url = _parse_triton_health_url(values.get("NUVION_TRITON_URL") or "localhost:8000")
    req = urllib.request.Request(health_url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=3) as response:
            if 200 <= response.getcode() < 300:
                return {
                    "name": "Triton health",
                    "status": "pass",
                    "detail": f"Ready endpoint reachable: {health_url}",
                }
    except Exception as exc:
        return {
            "name": "Triton health",
            "status": "fail",
            "detail": f"Health check failed: {exc}",
        }
    return {
        "name": "Triton health",
        "status": "fail",
        "detail": f"Unexpected status from {health_url}",
    }


def _check_camera_source(values: Dict[str, str]) -> Dict[str, str]:
    source = (values.get("NUVION_VIDEO_SOURCE") or "").strip()
    gst_source_override = (values.get("NUVION_GST_SOURCE") or "").strip()
    if gst_source_override:
        return {
            "name": "Camera source",
            "status": "pass",
            "detail": "Custom GStreamer source override configured.",
        }
    if not source:
        return {
            "name": "Camera source",
            "status": "warn",
            "detail": "NUVION_VIDEO_SOURCE is empty.",
        }
    if source == "auto":
        return {
            "name": "Camera source",
            "status": "pass",
            "detail": "Automatic camera detection is enabled.",
        }
    if sys.platform == "darwin":
        if source.startswith("avf"):
            return {
                "name": "Camera source",
                "status": "pass",
                "detail": f"macOS AVFoundation source configured: {source}",
            }
        if source.startswith("/dev/"):
            return {
                "name": "Camera source",
                "status": "warn",
                "detail": f"macOS usually expects avf/avf:<index>, current={source}",
            }
        return {
            "name": "Camera source",
            "status": "warn",
            "detail": f"Unrecognized macOS source format: {source}",
        }
    if source == "rpi":
        return {
            "name": "Camera source",
            "status": "pass",
            "detail": "Raspberry Pi camera source selected.",
        }
    if source in {"jetson", "argus", "csi"}:
        return {
            "name": "Camera source",
            "status": "pass",
            "detail": "Jetson CSI camera source selected.",
        }
    if source.startswith("/dev/"):
        if Path(source).exists():
            return {
                "name": "Camera source",
                "status": "pass",
                "detail": f"Device path exists: {source}",
            }
        return {
            "name": "Camera source",
            "status": "fail",
            "detail": f"Device path not found: {source}",
        }
    return {
        "name": "Camera source",
        "status": "warn",
        "detail": f"Custom source configured: {source}",
    }


def _check_demo_video_source(values: Dict[str, str]) -> Dict[str, str]:
    try:
        detail = validate_mvtec_demo_settings(
            base_url=values.get("NUVION_DEMO_MVTEC_BASE_URL"),
            categories=values.get("NUVION_DEMO_MVTEC_CATEGORIES"),
            cache_dir=values.get("NUVION_DEMO_MVTEC_CACHE_DIR"),
        )
    except ValueError as exc:
        return {
            "name": "Demo dataset source",
            "status": "fail",
            "detail": str(exc),
        }
    return {"name": "Demo dataset source", "status": "pass", "detail": detail}


def _check_tracking_overlay(values: Dict[str, str]) -> Dict[str, str]:
    if not _is_truthy(values.get("NUVION_FACE_TRACKING_ENABLED", "false")):
        return {"name": "Tracking overlay", "status": "skip", "detail": "Face tracking is disabled."}
    if _is_truthy(values.get("NUVION_FACE_TRACKING_SHOW_BBOX", "true")):
        return {"name": "Tracking overlay", "status": "pass", "detail": "Face tracking and bbox overlay are enabled."}
    return {"name": "Tracking overlay", "status": "warn", "detail": "Face tracking is enabled, but bbox overlay is hidden."}


def _check_motor_backend(values: Dict[str, str]) -> Dict[str, str]:
    from nuvion_app.inference.motor import MotorController

    if not _is_truthy(values.get("NUVION_MOTOR_ENABLED", "false")):
        return {"name": "Motor backend", "status": "skip", "detail": "Motor control is disabled."}

    controller = MotorController(_build_motor_config_from_values(values))
    try:
        if controller.available:
            return {
                "name": "Motor backend",
                "status": "pass",
                "detail": f"Motor backend ready: {controller.config.backend}",
            }
        return {
            "name": "Motor backend",
            "status": "warn",
            "detail": controller.reason or "Motor backend unavailable. Tracking will stay overlay-only.",
        }
    finally:
        controller.close()


def _run_preflight(values: Dict[str, str]) -> Dict[str, object]:
    demo_mode = _is_truthy(values.get("NUVION_DEMO_MODE", "false"))
    source_check = _check_demo_video_source(values) if demo_mode else _check_camera_source(values)
    checks = [
        _check_server_login(values),
        _check_triton_health(values),
        source_check,
        _check_tracking_overlay(values),
        _check_motor_backend(values),
    ]
    has_fail = any(check["status"] == "fail" for check in checks)
    return {"ok": not has_fail, "checks": checks}


def _render_form(
    fields: List[Dict[str, str]],
    values: Dict[str, str],
    missing: List[str],
    device_name: str,
    env_overrides: Optional[Dict[str, str]] = None,
    video_source_options: Optional[List[Dict[str, str]]] = None,
) -> str:
    env_overrides = env_overrides or {}
    video_source_options = video_source_options or []
    required_keys = effective_required_keys(values)
    backend_value = (values.get("NUVION_ZSAD_BACKEND") or "triton").strip().lower() or "triton"
    if backend_value not in {"triton", "siglip", "mps", "none"}:
        backend_value = "triton"
    siglip_device_value = (values.get("NUVION_ZERO_SHOT_DEVICE") or "auto").strip().lower() or "auto"
    if siglip_device_value not in {"auto", "mps", "cuda", "cpu"}:
        siglip_device_value = "auto"

    basic_rows: List[str] = []
    advanced_sections: Dict[str, List[str]] = {section: [] for section in ADVANCED_SECTION_ORDER}
    hidden_inputs: List[str] = []
    for field in fields:
        key = field["key"]
        comment = field["comment"] or key
        value = values.get(key, field["default"])
        if key in {"NUVION_ZSAD_BACKEND", "NUVION_ZERO_SHOT_DEVICE"}:
            hidden_inputs.append(
                '<input type="hidden" name="{key}" value="{value}">'.format(
                    key=html.escape(key),
                    value=html.escape(value or ""),
                )
            )
            continue

        group = _field_group(key)
        is_secret = _is_secret_key(key)
        input_type = "password" if is_secret else "text"
        placeholder = ""
        note_parts: List[str] = []
        if is_secret and values.get(key):
            value = ""
            note_parts.append("Leave blank to keep current value.")
        extra_note = _field_note(key)
        if extra_note:
            note_parts.append(extra_note)
        required_attr = "required" if (key in required_keys and _is_placeholder(values.get(key))) else ""
        if key in missing:
            placeholder = " required"
        note_html = ""
        if note_parts:
            note_html = "<div class=\"note\">{note}</div>".format(note=html.escape(" ".join(note_parts)))
        is_basic = _is_basic_setup_field(key)
        target_rows = basic_rows if is_basic else advanced_sections.setdefault(_field_section(key), [])
        if key == "NUVION_VIDEO_SOURCE" and video_source_options:
            options = list(video_source_options)
            if value and all(option["value"] != value for option in options):
                options.append(
                    {
                        "value": value,
                        "label": f"Current: {value}",
                        "detail": "Custom source",
                    }
                )

            option_rows = "\n".join(
                '<option value="{value}" {selected}>{label}</option>'.format(
                    value=html.escape(option["value"]),
                    selected="selected" if option["value"] == value else "",
                    label=html.escape(
                        option["label"] + (f" ({option['detail']})" if option.get("detail") else "")
                    ),
                )
                for option in options
            )
            target_rows.append(
                """
                <div class="field field-row group-{group}" data-group="{group}">
                  <label>{label}<span class="key">{key}</span></label>
                  <select name="{key}">
                    {options}
                  </select>
                  <div class="note">Use detected camera names instead of typing /dev/video0 or avf:2 manually.</div>
                </div>
                """.format(
                    group=html.escape(group),
                    label=html.escape(comment),
                    key=html.escape(key),
                    options=option_rows,
                )
            )
            continue
        if _is_motor_backend_field(key):
            backend_options = ("auto", "uart", "pwm", "none")
            target_rows.append(
                """
                <div class="field field-row group-{group}" data-group="{group}">
                  <label>{label}<span class="key">{key}</span></label>
                  <select name="{key}">
                    {options}
                  </select>
                  {note}
                </div>
                """.format(
                    group=html.escape(group),
                    label=html.escape(comment),
                    key=html.escape(key),
                    options="\n".join(
                        '<option value="{value}" {selected}>{value}</option>'.format(
                            value=option,
                            selected="selected" if option == (value or "auto") else "",
                        )
                        for option in backend_options
                    ),
                    note=note_html,
                )
            )
            continue
        if _is_rotation_field(key):
            rotation_options = ("0", "90", "180", "270")
            target_rows.append(
                """
                <div class="field field-row group-{group}" data-group="{group}">
                  <label>{label}<span class="key">{key}</span></label>
                  <select name="{key}">
                    {options}
                  </select>
                  {note}
                </div>
                """.format(
                    group=html.escape(group),
                    label=html.escape(comment),
                    key=html.escape(key),
                    options="\n".join(
                        '<option value="{value}" {selected}>{label}</option>'.format(
                            value=option,
                            selected="selected" if option == (value or "0") else "",
                            label=f"{option}°",
                        )
                        for option in rotation_options
                    ),
                    note=note_html,
                )
            )
            continue
        if _is_boolean_like(field["default"], value):
            normalized = (value or field["default"] or "false").strip().lower()
            if normalized not in {"true", "false"}:
                normalized = "false"
            target_rows.append(
                """
                <div class="field field-row group-{group}" data-group="{group}">
                  <label>{label}<span class="key">{key}</span></label>
                  <select name="{key}">
                    <option value="false" {false_selected}>Off</option>
                    <option value="true" {true_selected}>On</option>
                  </select>
                  {note}
                </div>
                """.format(
                    group=html.escape(group),
                    label=html.escape(comment),
                    key=html.escape(key),
                    false_selected="selected" if normalized == "false" else "",
                    true_selected="selected" if normalized == "true" else "",
                    note=note_html,
                )
            )
            continue

        target_rows.append(
            """
            <div class="field field-row group-{group}" data-group="{group}">
              <label>{label}<span class="key">{key}</span></label>
              <input type="{input_type}" name="{key}" value="{value}" {required} placeholder="{placeholder}">
              {note}
            </div>
            """.format(
                group=html.escape(group),
                label=html.escape(comment),
                key=html.escape(key),
                input_type=input_type,
                value=html.escape(value or ""),
                required=required_attr,
                placeholder=html.escape(placeholder.strip()),
                note=note_html,
            )
        )

    basic_block = "\n".join(basic_rows)
    advanced_count = sum(len(rows) for rows in advanced_sections.values())
    advanced_blocks: List[str] = []
    for section in ADVANCED_SECTION_ORDER:
        rows = advanced_sections.get(section, [])
        if not rows:
            continue
        title, description = ADVANCED_SECTION_META[section]
        advanced_blocks.append(
            """
            <section class="advanced-section" data-section="{section}">
              <div class="section-heading">
                <h3>{title}</h3>
                <p>{description}</p>
              </div>
              {rows}
            </section>
            """.format(
                section=html.escape(section),
                title=html.escape(title),
                description=html.escape(description),
                rows="\n".join(rows),
            )
        )

    advanced_block = """
        <details class="advanced-card">
          <summary>Advanced Options <span class="summary-meta">{count} fields</span></summary>
          <p class="muted">
            These settings are for debugging, local runtime tuning, or environment-specific overrides.
            Most devices do not need changes here after initial setup.
          </p>
          {sections}
        </details>
    """.format(count=advanced_count, sections="\n".join(advanced_blocks))

    error_block = ""
    if missing:
        error_items = " ".join(html.escape(key) for key in missing)
        error_block = f"<div class=\"error\">Missing required values: {error_items}</div>"

    override_block = ""
    if env_overrides:
        override_rows: List[str] = []
        for key in sorted(env_overrides.keys()):
            env_value = "***" if _is_secret_key(key) else env_overrides[key]
            override_rows.append(
                "<li><code>{key}</code> = <code>{value}</code></li>".format(
                    key=html.escape(key),
                    value=html.escape(env_value),
                )
            )
        override_block = """
          <div class="card warning">
            <h2>Environment Override Detected</h2>
            <p class="muted">
              These shell environment variables differ from the file values and will take precedence at runtime.
            </p>
            <ul class="override-list">
              {rows}
            </ul>
          </div>
        """.format(rows="\n".join(override_rows))

    inference_block = """
          <div class="card">
            <h2>Inference Mode</h2>
            <p class="muted">Choose backend first, then tune backend-specific options below.</p>
            <div class="grid">
              <div class="field">
                <label>Backend</label>
                <select id="inference-backend">
                  <option value="triton" {triton_selected}>Triton (server/runtime)</option>
                  <option value="siglip" {siglip_selected}>SigLIP (local)</option>
                  <option value="mps" {mps_selected}>SigLIP + MPS (macOS)</option>
                  <option value="none" {none_selected}>None (streaming only)</option>
                </select>
              </div>
              <div class="field">
                <label>SigLIP Device</label>
                <select id="siglip-device">
                  <option value="auto" {dev_auto_selected}>auto</option>
                  <option value="mps" {dev_mps_selected}>mps</option>
                  <option value="cuda" {dev_cuda_selected}>cuda</option>
                  <option value="cpu" {dev_cpu_selected}>cpu</option>
                </select>
              </div>
            </div>
            <div class="actions left">
              <button type="button" id="preflight-btn" onclick="runPreflight()">Run Preflight Check</button>
            </div>
            <div id="preflight-status" class="status"></div>
          </div>
    """.format(
        triton_selected="selected" if backend_value == "triton" else "",
        siglip_selected="selected" if backend_value == "siglip" else "",
        mps_selected="selected" if backend_value == "mps" else "",
        none_selected="selected" if backend_value == "none" else "",
        dev_auto_selected="selected" if siglip_device_value == "auto" else "",
        dev_mps_selected="selected" if siglip_device_value == "mps" else "",
        dev_cuda_selected="selected" if siglip_device_value == "cuda" else "",
        dev_cpu_selected="selected" if siglip_device_value == "cpu" else "",
    )

    provision_block = """
          <div class="card">
            <h2>Auto Provision (recommended)</h2>
            <p class="muted">
              Login with a space owner/admin account to create a device credential.
              Your account is not stored on the device.
            </p>
            <div class="grid">
              <div class="field">
                <label>Account username</label>
                <input type="text" id="prov-username" autocomplete="username">
              </div>
              <div class="field">
                <label>Account password</label>
                <input type="password" id="prov-password" autocomplete="current-password">
              </div>
              <div class="field">
                <label>Space</label>
                <select id="prov-space-select" disabled>
                  <option value="">Login to load spaces</option>
                </select>
              </div>
              <div class="field">
                <label>Device name</label>
                <input type="text" id="prov-device" value="{device_name}">
              </div>
            </div>
            <div class="actions">
              <button type="button" id="prov-login" onclick="loadSpaces()">Login & Load Spaces</button>
              <button type="button" id="prov-create" onclick="provisionDevice()" disabled>Create Device</button>
            </div>
            <div id="provision-status" class="status"></div>
          </div>
    """.format(device_name=html.escape(device_name))

    motor_test_block = """
          <div class="card">
            <h2>Motor Test</h2>
            <p class="muted">Send a single movement command using the current form values before saving.</p>
            <div class="actions left">
              <button type="button" onclick="sendMotorCommand('LEFT')">Left</button>
              <button type="button" onclick="sendMotorCommand('RIGHT')">Right</button>
              <button type="button" onclick="sendMotorCommand('UP')">Up</button>
              <button type="button" onclick="sendMotorCommand('DOWN')">Down</button>
              <button type="button" onclick="sendMotorCommand('CENTER')">Center</button>
            </div>
            <div id="motor-test-status" class="status"></div>
          </div>
    """

    return """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Nuvion Agent Setup</title>
        <style>
          :root {{
            color-scheme: light;
            --bg: #f6f4f0;
            --card: #ffffff;
            --ink: #1a1a1a;
            --muted: #606060;
            --accent: #2f6b4f;
            --border: #e3ded7;
          }}
          body {{
            margin: 0;
            font-family: "Avenir Next", "Helvetica Neue", Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--ink);
          }}
          .wrap {{
            max-width: 820px;
            margin: 40px auto 80px;
            padding: 0 20px;
          }}
          header h1 {{
            font-size: 28px;
            margin-bottom: 6px;
          }}
          header p {{
            color: var(--muted);
            margin-top: 0;
          }}
          .card {{
            background: var(--card);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 12px 30px rgba(0,0,0,0.08);
            border: 1px solid var(--border);
          }}
          .card + .card {{
            margin-top: 20px;
          }}
          .card h2 {{
            margin-top: 0;
            margin-bottom: 8px;
          }}
          .field {{
            display: flex;
            flex-direction: column;
            margin-bottom: 18px;
          }}
          .warning {{
            border-color: #f2c979;
            background: #fff8ea;
          }}
          .override-list {{
            margin: 0;
            padding-left: 18px;
            font-size: 13px;
          }}
          label {{
            font-weight: 600;
            margin-bottom: 6px;
            display: flex;
            justify-content: space-between;
            gap: 12px;
          }}
          .key {{
            font-size: 12px;
            color: var(--muted);
            font-weight: 400;
          }}
          input {{
            padding: 10px 12px;
            border: 1px solid var(--border);
            border-radius: 8px;
            font-size: 14px;
          }}
          select {{
            padding: 10px 12px;
            border: 1px solid var(--border);
            border-radius: 8px;
            font-size: 14px;
            background: white;
          }}
          code {{
            font-family: "SFMono-Regular", Menlo, Consolas, monospace;
          }}
          .note {{
            font-size: 12px;
            color: var(--muted);
            margin-top: 6px;
          }}
          .actions {{
            display: flex;
            justify-content: flex-end;
            margin-top: 18px;
            gap: 10px;
          }}
          .actions.left {{
            justify-content: flex-start;
          }}
          .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 12px;
          }}
          .muted {{
            color: var(--muted);
            margin-top: 4px;
            margin-bottom: 16px;
          }}
          .status {{
            margin-top: 12px;
            font-size: 13px;
            color: var(--muted);
          }}
          .summary-meta {{
            color: var(--muted);
            font-size: 13px;
            font-weight: 500;
            margin-left: 8px;
          }}
          .quick-start {{
            background: linear-gradient(180deg, #fffdf8 0%, #ffffff 100%);
          }}
          .advanced-card {{
            margin-top: 20px;
            border-top: 1px solid var(--border);
            padding-top: 16px;
          }}
          .advanced-card summary {{
            cursor: pointer;
            font-weight: 700;
            list-style: none;
            display: flex;
            align-items: center;
            justify-content: space-between;
          }}
          .advanced-card summary::-webkit-details-marker {{
            display: none;
          }}
          .advanced-section + .advanced-section {{
            margin-top: 24px;
            padding-top: 20px;
            border-top: 1px solid var(--border);
          }}
          .section-heading h3 {{
            margin: 0 0 6px;
            font-size: 16px;
          }}
          .section-heading p {{
            margin: 0 0 16px;
            color: var(--muted);
            font-size: 13px;
          }}
          .checks {{
            margin: 0;
            padding-left: 18px;
          }}
          .checks li {{
            margin-bottom: 6px;
          }}
          .check-pass {{
            color: #0c7a34;
          }}
          .check-fail {{
            color: #8a1f1f;
          }}
          .check-warn {{
            color: #8b5a00;
          }}
          .check-skip {{
            color: #555;
          }}
          button {{
            background: var(--accent);
            color: white;
            border: none;
            padding: 12px 18px;
            border-radius: 10px;
            font-weight: 600;
            cursor: pointer;
          }}
          .error {{
            background: #ffe3e3;
            color: #8a1f1f;
            padding: 12px 14px;
            border-radius: 8px;
            margin-bottom: 18px;
          }}
        </style>
      </head>
      <body>
        <div class="wrap">
          <header>
            <h1>Nuvion Agent Setup</h1>
            <p>Start with the essentials, then open advanced options only if you need local tuning.</p>
          </header>
          {override_block}
          <form id="config-form" method="post" action="/save">
            {hidden_inputs}
            <div class="card quick-start">
              <h2>Quick Start</h2>
              <p class="muted">
                Most devices only need the server address, device credential, camera source, and demo mode.
              </p>
              {error_block}
              {basic_rows}
            </div>
          {provision_block}
          {inference_block}
          {motor_test_block}
          <div class="card">
              {advanced_block}
              {rows}
              <div class="actions">
                <button type="submit">Save</button>
              </div>
            </div>
          </form>
        </div>
        <script>
          async function loadSpaces() {{
            const statusEl = document.getElementById("provision-status");
            const loginBtn = document.getElementById("prov-login");
            const createBtn = document.getElementById("prov-create");
            const spaceSelect = document.getElementById("prov-space-select");
            const serverBaseUrl = document.querySelector('input[name="NUVION_SERVER_BASE_URL"]').value.trim();
            const username = document.getElementById("prov-username").value.trim();
            const password = document.getElementById("prov-password").value;
            if (!serverBaseUrl || !username || !password) {{
              statusEl.textContent = "Server URL, username, and password are required.";
              return;
            }}
            loginBtn.disabled = true;
            createBtn.disabled = true;
            statusEl.textContent = "Loading spaces...";
            spaceSelect.innerHTML = "<option value=''>Loading...</option>";
            spaceSelect.disabled = true;
            try {{
              const resp = await fetch("/api/spaces", {{
                method: "POST",
                headers: {{ "Content-Type": "application/json" }},
                body: JSON.stringify({{ serverBaseUrl, username, password }})
              }});
              const data = await resp.json();
              if (!resp.ok || data.error) {{
                statusEl.textContent = data.error || "Failed to load spaces.";
                return;
              }}
              const spaces = (data.spaces || data || []);
              spaceSelect.innerHTML = "";
              if (!spaces.length) {{
                spaceSelect.innerHTML = "<option value=''>No spaces found</option>";
                statusEl.textContent = "No spaces found for this account.";
                return;
              }}
              spaceSelect.innerHTML = "<option value=''>Select a space</option>";
              spaces.forEach((space) => {{
                const option = document.createElement("option");
                option.value = space.id;
                option.textContent = `${{space.name || "Space"}} (#${{space.id}})`;
                spaceSelect.appendChild(option);
              }});
              spaceSelect.disabled = false;
              createBtn.disabled = false;
              statusEl.textContent = "Spaces loaded. Select a space to provision.";
            }} catch (err) {{
              statusEl.textContent = "Failed to load spaces: " + err;
            }} finally {{
              loginBtn.disabled = false;
            }}
          }}

          async function provisionDevice() {{
            const statusEl = document.getElementById("provision-status");
            const spaceSelect = document.getElementById("prov-space-select");
            const spaceId = spaceSelect.value;
            if (!spaceId) {{
              statusEl.textContent = "Please select a space.";
              return;
            }}
            statusEl.textContent = "Provisioning device credentials...";
            const payload = {{
              serverBaseUrl: document.querySelector('input[name="NUVION_SERVER_BASE_URL"]').value.trim(),
              username: document.getElementById("prov-username").value.trim(),
              password: document.getElementById("prov-password").value,
              spaceId: spaceId,
              deviceName: document.getElementById("prov-device").value.trim()
            }};
            try {{
              const resp = await fetch("/api/provision", {{
                method: "POST",
                headers: {{ "Content-Type": "application/json" }},
                body: JSON.stringify(payload)
              }});
              const data = await resp.json();
              if (!resp.ok || data.error) {{
                statusEl.textContent = data.error || "Provisioning failed.";
                return;
              }}
              const deviceUsername = data.deviceUsername || data.username || "";
              const devicePassword = data.devicePassword || data.password || data.deviceSecret || "";
              if (deviceUsername) {{
                document.querySelector('input[name="NUVION_DEVICE_USERNAME"]').value = deviceUsername;
              }}
              if (devicePassword) {{
                document.querySelector('input[name="NUVION_DEVICE_PASSWORD"]').value = devicePassword;
              }}
              statusEl.textContent = "Device credentials created. Review and click Save.";
            }} catch (err) {{
              statusEl.textContent = "Provisioning error: " + err;
            }}
          }}

          function applyInferenceMode() {{
            const backendSelect = document.getElementById("inference-backend");
            const deviceSelect = document.getElementById("siglip-device");
            if (!backendSelect || !deviceSelect) {{
              return;
            }}
            const backend = (backendSelect.value || "triton").toLowerCase();
            const siglipMode = backend === "siglip" || backend === "mps";
            const tritonMode = backend === "triton";

            document.querySelectorAll('.field-row[data-group="siglip"]').forEach((el) => {{
              el.style.display = siglipMode ? "" : "none";
            }});
            document.querySelectorAll('.field-row[data-group="triton"]').forEach((el) => {{
              el.style.display = tritonMode ? "" : "none";
            }});

            const backendInput = document.querySelector('input[name="NUVION_ZSAD_BACKEND"]');
            if (backendInput) {{
              backendInput.value = backend;
            }}

            if (backend === "mps") {{
              deviceSelect.value = "mps";
              deviceSelect.disabled = true;
            }} else if (siglipMode) {{
              deviceSelect.disabled = false;
            }} else {{
              deviceSelect.disabled = true;
            }}

            const deviceInput = document.querySelector('input[name="NUVION_ZERO_SHOT_DEVICE"]');
            if (deviceInput) {{
              deviceInput.value = deviceSelect.value || "auto";
            }}
          }}

          async function runPreflight() {{
            applyInferenceMode();
            const statusEl = document.getElementById("preflight-status");
            const btn = document.getElementById("preflight-btn");
            const form = document.getElementById("config-form");
            if (!statusEl || !btn || !form) {{
              return;
            }}

            const values = {{}};
            const data = new FormData(form);
            data.forEach((value, key) => {{
              values[key] = String(value);
            }});

            btn.disabled = true;
            statusEl.textContent = "Running preflight checks...";
            try {{
              const resp = await fetch("/api/preflight", {{
                method: "POST",
                headers: {{ "Content-Type": "application/json" }},
                body: JSON.stringify({{ values }})
              }});
              const result = await resp.json();
              if (!resp.ok || result.error) {{
                statusEl.textContent = result.error || "Preflight failed.";
                return;
              }}
              const checks = result.checks || [];
              if (!checks.length) {{
                statusEl.textContent = "No check results.";
                return;
              }}
              const lines = checks.map((check) => {{
                const st = check.status || "warn";
                return `<li class="check-${{st}}"><strong>${{check.name}}:</strong> ${{check.detail}}</li>`;
              }}).join("");
              statusEl.innerHTML = `<ul class="checks">${{lines}}</ul>`;
            }} catch (err) {{
              statusEl.textContent = "Preflight error: " + err;
            }} finally {{
              btn.disabled = false;
            }}
          }}

          async function sendMotorCommand(command) {{
            const statusEl = document.getElementById("motor-test-status");
            const form = document.getElementById("config-form");
            if (!statusEl || !form) {{
              return;
            }}
            const values = {{}};
            const data = new FormData(form);
            data.forEach((value, key) => {{
              values[key] = String(value);
            }});
            statusEl.textContent = `Sending ${{command}}...`;
            try {{
              const resp = await fetch("/api/motor-test", {{
                method: "POST",
                headers: {{ "Content-Type": "application/json" }},
                body: JSON.stringify({{ values, command }})
              }});
              const result = await resp.json();
              if (!resp.ok || result.error) {{
                statusEl.textContent = result.error || "Motor test failed.";
                return;
              }}
              statusEl.textContent = result.detail || `Sent ${{command}}`;
            }} catch (err) {{
              statusEl.textContent = "Motor test error: " + err;
            }}
          }}

          const backendSelect = document.getElementById("inference-backend");
          const deviceSelect = document.getElementById("siglip-device");
          if (backendSelect) {{
            backendSelect.addEventListener("change", applyInferenceMode);
          }}
          if (deviceSelect) {{
            deviceSelect.addEventListener("change", applyInferenceMode);
          }}
          applyInferenceMode();
        </script>
      </body>
    </html>
    """.format(
        error_block=error_block,
        rows="",
        basic_rows=basic_block,
        hidden_inputs="\n".join(hidden_inputs),
        override_block=override_block,
        inference_block=inference_block,
        provision_block=provision_block,
        motor_test_block=motor_test_block,
        advanced_block=advanced_block,
    )


def run_web_setup(
    config_path: Path,
    host: str,
    port: int,
    open_browser: bool,
) -> None:
    lines, fields = load_template()
    existing = _merge_defaults(fields, read_env(config_path))
    device_name = socket.gethostname()
    video_source_options = discover_video_source_options()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

        def _send_html(self, status: HTTPStatus, body: str) -> None:
            payload = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(self, status: HTTPStatus, body: Dict[str, object]) -> None:
            payload = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html"):
                missing = _validate_required(existing)
                body = _render_form(
                    fields,
                    existing,
                    missing,
                    device_name,
                    env_overrides=_collect_env_overrides(fields, existing),
                    video_source_options=video_source_options,
                )
                self._send_html(HTTPStatus.OK, body)
                return
            if self.path == "/health":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"ok")
                return
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/api/spaces":
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8")
                try:
                    payload = json.loads(body) if body else {}
                except json.JSONDecodeError:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON"})
                    return

                server_base_url = str(payload.get("serverBaseUrl") or existing.get("NUVION_SERVER_BASE_URL") or "").strip()
                username = str(payload.get("username") or "").strip()
                password = str(payload.get("password") or "")
                if not server_base_url or not username or not password:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Missing credentials or server base URL."})
                    return
                spaces = _fetch_spaces(server_base_url, username, password)
                if spaces is None:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Failed to load spaces."})
                    return
                self._send_json(HTTPStatus.OK, {"spaces": spaces})
                return

            if self.path == "/api/provision":
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8")
                try:
                    payload = json.loads(body) if body else {}
                except json.JSONDecodeError:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON"})
                    return

                server_base_url = str(payload.get("serverBaseUrl") or existing.get("NUVION_SERVER_BASE_URL") or "").strip()
                username = str(payload.get("username") or "").strip()
                password = str(payload.get("password") or "")
                space_id = str(payload.get("spaceId") or "").strip()
                device_name_local = str(payload.get("deviceName") or device_name).strip()

                if not server_base_url or not username or not password or not space_id:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Missing required provisioning fields."})
                    return

                data = _provision_device(server_base_url, username, password, space_id, device_name_local)
                if not data:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Provisioning failed."})
                    return
                self._send_json(HTTPStatus.OK, data)
                return

            if self.path == "/api/preflight":
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8")
                try:
                    payload = json.loads(body) if body else {}
                except json.JSONDecodeError:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON"})
                    return

                incoming_values = payload.get("values")
                if not isinstance(incoming_values, dict):
                    incoming_values = {}

                values = dict(existing)
                for field in fields:
                    key = field["key"]
                    raw = incoming_values.get(key)
                    if raw is None:
                        continue
                    posted = str(raw).strip()
                    if not posted and _is_secret_key(key) and existing.get(key):
                        values[key] = existing[key]
                    else:
                        values[key] = posted

                self._send_json(HTTPStatus.OK, _run_preflight(values))
                return

            if self.path == "/api/motor-test":
                from nuvion_app.inference.motor import MotorCommand, MotorController

                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8")
                try:
                    payload = json.loads(body) if body else {}
                except json.JSONDecodeError:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON"})
                    return

                incoming_values = payload.get("values")
                command_name = str(payload.get("command") or "").strip().upper()
                if not isinstance(incoming_values, dict):
                    incoming_values = {}

                values = dict(existing)
                for field in fields:
                    key = field["key"]
                    raw = incoming_values.get(key)
                    if raw is None:
                        continue
                    posted = str(raw).strip()
                    if not posted and _is_secret_key(key) and existing.get(key):
                        values[key] = existing[key]
                    else:
                        values[key] = posted

                if command_name not in MotorCommand.__members__:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"Unsupported command: {command_name}"})
                    return

                controller = MotorController(_build_motor_config_from_values(values))
                try:
                    if not controller.available:
                        self._send_json(
                            HTTPStatus.OK,
                            {"detail": controller.reason or "Motor backend unavailable. Tracking will stay overlay-only."},
                        )
                        return
                    sent = controller.send(MotorCommand[command_name])
                    detail = f"Sent {command_name}" if sent else f"Skipped {command_name} due to command throttling."
                    self._send_json(HTTPStatus.OK, {"detail": detail})
                finally:
                    controller.close()
                return

            if self.path != "/save":
                self.send_response(HTTPStatus.NOT_FOUND)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            parsed = urllib.parse.parse_qs(body)
            values = dict(existing)

            for field in fields:
                key = field["key"]
                posted = parsed.get(key, [""])[0].strip()
                if not posted and _is_secret_key(key) and existing.get(key):
                    values[key] = existing[key]
                else:
                    values[key] = posted

            missing = _validate_required(values)
            if missing:
                body = _render_form(
                    fields,
                    values,
                    missing,
                    device_name,
                    env_overrides=_collect_env_overrides(fields, values),
                    video_source_options=video_source_options,
                )
                self._send_html(HTTPStatus.BAD_REQUEST, body)
                return

            write_env(config_path, lines, values)
            success = """
            <!doctype html>
            <html lang="en">
              <head>
                <meta charset="utf-8" />
                <title>Saved</title>
              </head>
              <body>
                <h2>Saved</h2>
                <p>Settings saved. Restart the service to apply.</p>
              </body>
            </html>
            """
            self._send_html(HTTPStatus.OK, success)
            threading.Thread(target=self.server.shutdown, daemon=True).start()

    server = ThreadingHTTPServer((host, port), Handler)
    address = f"http://{host}:{server.server_address[1]}"
    print(f"Setup page: {address}")
    if open_browser:
        browser_target = address
        if host == "0.0.0.0":
            browser_target = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            webbrowser.open(browser_target)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Setup server stopped.")
    finally:
        server.server_close()


def run_qr_setup(config_path: Path, advanced: bool) -> None:
    lines, fields = load_template()
    existing = _merge_defaults(fields, read_env(config_path))
    server_base_url = str(existing.get("NUVION_SERVER_BASE_URL") or "").strip()
    if not server_base_url:
        raise RuntimeError("NUVION_SERVER_BASE_URL is required for QR setup.")
    device_name = socket.gethostname()

    pairing = _init_pairing(server_base_url, device_name)
    if not pairing:
        raise RuntimeError("Failed to initiate pairing. Check server URL and network.")

    pairing_url = str(
        pairing.get("pairingUrl")
        or pairing.get("url")
        or pairing.get("pairingURL")
        or ""
    )
    pairing_code = str(pairing.get("pairingCode") or pairing.get("code") or "").strip()
    pairing_id = str(pairing.get("pairingId") or pairing.get("id") or "").strip()
    pairing_secret = str(pairing.get("pairingSecret") or pairing.get("secret") or "").strip() or None

    if pairing_code:
        print("Pairing code:", pairing_code)
    if pairing_url:
        _print_qr(pairing_url)

    if not pairing_id:
        raise RuntimeError("Pairing response missing pairingId.")

    print("Waiting for pairing approval...")
    result = _wait_for_pairing(server_base_url, pairing_id, pairing_secret)
    if not result:
        raise RuntimeError("Pairing not approved or expired.")

    values = dict(existing)
    device_username = (
        result.get("deviceUsername")
        or result.get("username")
        or result.get("deviceId")
    )
    device_password = (
        result.get("devicePassword")
        or result.get("password")
        or result.get("deviceSecret")
        or result.get("secret")
    )
    if device_username:
        values["NUVION_DEVICE_USERNAME"] = str(device_username)
    if device_password:
        values["NUVION_DEVICE_PASSWORD"] = str(device_password)

    values = _prompt_camera_setup(fields, values)
    values = _prompt_tracking_motor_setup(fields, values)

    missing = _validate_required(values)
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(f"Missing required values after pairing: {missing_str}")

    write_env(config_path, lines, values)
    print(f"Saved: {config_path}")


def setup_config(
    config_path: Optional[str] = None,
    use_web: Optional[bool] = None,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
    advanced: bool = False,
    qr: bool = False,
) -> Path:
    path = resolve_config_path(config_path)

    if qr:
        use_web = False

    if use_web is None:
        use_web = _has_display()

    if use_web:
        run_web_setup(path, host=host, port=port, open_browser=open_browser)
    else:
        if not qr and not _has_display():
            qr = True
        if qr:
            run_qr_setup(path, advanced=advanced)
        else:
            lines, fields = load_template()
            existing = read_env(path)
            values = prompt_cli(fields, existing, advanced=advanced)
            missing = _validate_required(values)
            if missing:
                missing_str = ", ".join(missing)
                raise RuntimeError(f"Missing required values: {missing_str}")
            write_env(path, lines, values)
            print(f"Saved: {path}")

    try:
        os.environ["NUV_AGENT_CONFIG"] = str(path)
        for key, value in read_env(path).items():
            os.environ[key] = value

        from nuvion_app.runtime.config_guard import ensure_runtime_config
        from nuvion_app.runtime.bootstrap import ensure_ready

        ensure_runtime_config(config_path=path, stage="setup", apply_fixes=True)
        ready = ensure_ready(stage="setup")
        if ready:
            print("Runtime bootstrap: ready")
        else:
            print("Runtime bootstrap: degraded (backend switched to none for this session)")
    except Exception as exc:
        print(f"Runtime bootstrap skipped: {exc}")

    return path
