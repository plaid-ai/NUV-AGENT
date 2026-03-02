from __future__ import annotations

import logging
import os
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
    pull_model_from_gcs,
    pull_model_from_server,
    resolve_default_model_dir,
)
from nuvion_app.runtime.errors import BootstrapError

log = logging.getLogger(__name__)


_VALID_PROFILES = {"runtime", "light", "full"}


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_darwin() -> bool:
    return os.uname().sysname.lower() == "darwin"


def resolve_effective_profile() -> str:
    default_profile = (os.getenv("NUVION_MODEL_PROFILE", DEFAULT_MODEL_PROFILE) or DEFAULT_MODEL_PROFILE).strip().lower()
    if _is_darwin():
        profile = (os.getenv("NUVION_TRITON_MAC_PROFILE", "full") or "full").strip().lower()
    else:
        profile = (os.getenv("NUVION_TRITON_JETSON_PROFILE", default_profile) or default_profile).strip().lower()

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


def _missing_required_files(model_dir: Path, profile: str) -> list[str]:
    missing: list[str] = []
    for key in _PROFILE_KEYS[profile]:
        rel_path = _DEFAULT_LOCAL_PATHS.get(key)
        if not rel_path:
            continue
        file_path = (model_dir / rel_path).resolve()
        if not file_path.exists():
            missing.append(rel_path)
    return missing


def _pull_model(profile: str, model_dir: Path) -> None:
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
        )
        return

    pull_model_from_gcs(
        pointer_uri=(os.getenv("NUVION_MODEL_GCS_POINTER_URI", DEFAULT_MODEL_GCS_POINTER_URI) or DEFAULT_MODEL_GCS_POINTER_URI).strip(),
        local_dir=str(model_dir),
        profile=profile,
    )


def ensure_model_ready(stage: str) -> Path:
    backend = (os.getenv("NUVION_ZSAD_BACKEND", "triton") or "triton").strip().lower()
    if backend != "triton":
        return resolve_model_dir(resolve_effective_profile())

    if stage == "setup" and not _truthy(os.getenv("NUVION_MODEL_AUTO_PULL_ON_SETUP"), default=True):
        return resolve_model_dir(resolve_effective_profile())
    if stage == "run" and not _truthy(os.getenv("NUVION_MODEL_AUTO_PULL_ON_RUN"), default=True):
        return resolve_model_dir(resolve_effective_profile())

    profile = resolve_effective_profile()
    model_dir = resolve_model_dir(profile)
    missing_before = _missing_required_files(model_dir, profile)
    if not missing_before:
        return model_dir

    try:
        log.info(
            "[BOOTSTRAP] Missing model artifacts detected (%s). Pulling profile=%s source=%s",
            ", ".join(missing_before),
            os.getenv("NUVION_MODEL_PROFILE", DEFAULT_MODEL_PROFILE),
            os.getenv("NUVION_MODEL_SOURCE", DEFAULT_MODEL_SOURCE),
        )
        _pull_model(profile=profile, model_dir=model_dir)
    except Exception as exc:
        raise BootstrapError("model_pull_failed", f"Failed to pull model artifacts: {exc}") from exc

    missing_after = _missing_required_files(model_dir, profile)
    if missing_after:
        raise BootstrapError(
            "model_pull_failed",
            f"Model pull finished but required files are still missing: {', '.join(missing_after)}",
        )

    return model_dir
