from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from nuvion_app.model_store import (
    DEFAULT_MODEL_GCS_POINTER_URI,
    DEFAULT_MODEL_POINTER,
    DEFAULT_MODEL_PRESIGN_TTL_SECONDS,
    DEFAULT_MODEL_PROFILE,
    DEFAULT_MODEL_SERVER_BASE_URL,
    DEFAULT_MODEL_SOURCE,
    _DEFAULT_LOCAL_PATHS,
    _PROFILE_KEYS,
    ensure_default_face_tracking_model,
    merge_required_keys,
    pull_model_from_gcs,
    pull_model_from_server,
    resolve_default_model_dir,
)
from nuvion_app.runtime.errors import BootstrapError
from nuvion_app.runtime.inference_mode import face_tracking_uses_triton, normalize_backend

log = logging.getLogger(__name__)


_VALID_PROFILES = {"runtime", "light", "full"}
_FACE_TRACKING_REQUIRED_KEYS = ["face_onnx"]


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _emit_progress(message: str) -> None:
    sys.stderr.write(f"[BOOTSTRAP] {message}\n")
    sys.stderr.flush()


def _is_darwin() -> bool:
    return os.uname().sysname.lower() == "darwin"


def _is_raspberry_pi_linux() -> bool:
    try:
        if os.uname().sysname.lower() != "linux":
            return False
    except Exception:
        return False

    probe_paths = (
        Path("/proc/device-tree/model"),
        Path("/sys/firmware/devicetree/base/model"),
        Path("/proc/device-tree/compatible"),
        Path("/sys/firmware/devicetree/base/compatible"),
    )
    for probe_path in probe_paths:
        try:
            content = probe_path.read_text(encoding="utf-8", errors="ignore").replace("\x00", " ").lower()
        except OSError:
            continue
        if "raspberry pi" in content or "raspberrypi" in content:
            return True
    return False


def _should_use_full_triton_profile() -> bool:
    return _is_darwin() or _is_raspberry_pi_linux()


def resolve_effective_profile() -> str:
    default_profile = (os.getenv("NUVION_MODEL_PROFILE", DEFAULT_MODEL_PROFILE) or DEFAULT_MODEL_PROFILE).strip().lower()
    if _should_use_full_triton_profile():
        if _is_darwin():
            profile_env_name = "NUVION_TRITON_MAC_PROFILE"
        else:
            profile_env_name = "NUVION_TRITON_RPI_PROFILE"
        profile = (os.getenv(profile_env_name, "full") or "full").strip().lower()
    else:
        # Jetson/Linux Triton path should default to the minimal runtime bundle
        # (text_features + plan + triton_config + manifest) unless explicitly overridden.
        profile = (os.getenv("NUVION_TRITON_JETSON_PROFILE", "runtime") or "runtime").strip().lower()

    if profile not in _VALID_PROFILES:
        return DEFAULT_MODEL_PROFILE
    return profile


def resolve_model_dir(profile: str) -> Path:
    explicit = (os.getenv("NUVION_MODEL_LOCAL_DIR") or "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()

    source = (os.getenv("NUVION_MODEL_SOURCE", DEFAULT_MODEL_SOURCE) or DEFAULT_MODEL_SOURCE).strip().lower()
    if source == "server":
        pointer = (os.getenv("NUVION_MODEL_POINTER", DEFAULT_MODEL_POINTER) or DEFAULT_MODEL_POINTER).strip()
        identifier = f"server:{pointer}:{profile}"
        return resolve_default_model_dir(identifier)

    gcs_pointer_uri = (os.getenv("NUVION_MODEL_GCS_POINTER_URI", DEFAULT_MODEL_GCS_POINTER_URI) or DEFAULT_MODEL_GCS_POINTER_URI).strip()
    return resolve_default_model_dir(gcs_pointer_uri)


def _required_model_keys(profile: str) -> list[str]:
    keys: list[str] = []
    if normalize_backend(os.getenv("NUVION_ZSAD_BACKEND", "triton"), default="triton") == "triton":
        keys = merge_required_keys(keys, _PROFILE_KEYS[profile])
    if face_tracking_uses_triton():
        keys = merge_required_keys(keys, _FACE_TRACKING_REQUIRED_KEYS)
    return keys


def _ensure_optional_face_tracking_assets(model_dir: Path) -> None:
    if not face_tracking_uses_triton():
        return
    face_model_path = (model_dir / _DEFAULT_LOCAL_PATHS["face_onnx"]).resolve()
    if face_model_path.exists():
        return
    ensure_default_face_tracking_model(model_dir)


def _missing_required_files(model_dir: Path, profile: str) -> list[str]:
    missing: list[str] = []
    required_keys = _required_model_keys(profile)
    for key in required_keys:
        rel_path = _DEFAULT_LOCAL_PATHS.get(key)
        if not rel_path:
            continue
        file_path = (model_dir / rel_path).resolve()
        if not file_path.exists():
            missing.append(rel_path)
    return missing


def _pull_model(profile: str, model_dir: Path) -> None:
    anomaly_uses_triton = normalize_backend(os.getenv("NUVION_ZSAD_BACKEND", "triton"), default="triton") == "triton"
    tracking_uses_triton = face_tracking_uses_triton()
    optional_tracking_keys = _FACE_TRACKING_REQUIRED_KEYS if tracking_uses_triton else []

    if not anomaly_uses_triton and tracking_uses_triton:
        ensure_default_face_tracking_model(model_dir)
        return

    source = (os.getenv("NUVION_MODEL_SOURCE", DEFAULT_MODEL_SOURCE) or DEFAULT_MODEL_SOURCE).strip().lower()
    if source == "server":
        pull_model_from_server(
            server_base_url=(os.getenv("NUVION_MODEL_SERVER_BASE_URL", os.getenv("NUVION_SERVER_BASE_URL", DEFAULT_MODEL_SERVER_BASE_URL)) or DEFAULT_MODEL_SERVER_BASE_URL).strip(),
            pointer=(os.getenv("NUVION_MODEL_POINTER", DEFAULT_MODEL_POINTER) or DEFAULT_MODEL_POINTER).strip(),
            profile=profile,
            local_dir=str(model_dir),
            ttl_seconds=int(os.getenv("NUVION_MODEL_PRESIGN_TTL_SECONDS", str(DEFAULT_MODEL_PRESIGN_TTL_SECONDS))),
            access_token=(os.getenv("NUVION_MODEL_SERVER_ACCESS_TOKEN") or "").strip() or None,
            username=(os.getenv("NUVION_DEVICE_USERNAME") or "").strip() or None,
            password=(os.getenv("NUVION_DEVICE_PASSWORD") or "").strip() or None,
            optional_keys=optional_tracking_keys,
        )
    else:
        pull_model_from_gcs(
            pointer_uri=(os.getenv("NUVION_MODEL_GCS_POINTER_URI", DEFAULT_MODEL_GCS_POINTER_URI) or DEFAULT_MODEL_GCS_POINTER_URI).strip(),
            local_dir=str(model_dir),
            profile=profile,
            optional_keys=optional_tracking_keys,
        )

    if tracking_uses_triton:
        _ensure_optional_face_tracking_assets(model_dir)


def ensure_model_ready(stage: str) -> Path:
    backend = normalize_backend(os.getenv("NUVION_ZSAD_BACKEND", "triton"), default="triton")
    if backend != "triton" and not face_tracking_uses_triton():
        return resolve_model_dir(resolve_effective_profile())

    if stage == "setup" and not _truthy(os.getenv("NUVION_MODEL_AUTO_PULL_ON_SETUP"), default=True):
        return resolve_model_dir(resolve_effective_profile())
    if stage == "run" and not _truthy(os.getenv("NUVION_MODEL_AUTO_PULL_ON_RUN"), default=True):
        return resolve_model_dir(resolve_effective_profile())

    profile = resolve_effective_profile()
    model_dir = resolve_model_dir(profile)
    missing_before = _missing_required_files(model_dir, profile)
    if not missing_before:
        _emit_progress(f"모델 파일 점검 완료: 누락 없음 ({model_dir})")
        return model_dir

    try:
        log.info(
            "[BOOTSTRAP] Missing model artifacts detected (%s). Pulling profile=%s source=%s",
            ", ".join(missing_before),
            os.getenv("NUVION_MODEL_PROFILE", DEFAULT_MODEL_PROFILE),
            os.getenv("NUVION_MODEL_SOURCE", DEFAULT_MODEL_SOURCE),
        )
        _emit_progress(
            f"모델 파일 누락 감지({len(missing_before)}개). "
            f"자동 다운로드 시작: source={os.getenv('NUVION_MODEL_SOURCE', DEFAULT_MODEL_SOURCE)} "
            f"profile={profile}"
        )
        _pull_model(profile=profile, model_dir=model_dir)
        _emit_progress("모델 다운로드 완료. 무결성 점검 중...")
    except Exception as exc:
        raise BootstrapError("model_pull_failed", f"Failed to pull model artifacts: {exc}") from exc

    missing_after = _missing_required_files(model_dir, profile)
    if missing_after:
        raise BootstrapError(
            "model_pull_failed",
            f"Model pull finished but required files are still missing: {', '.join(missing_after)}",
        )
    _emit_progress(f"모델 파일 점검 완료: 준비됨 ({model_dir})")

    return model_dir
