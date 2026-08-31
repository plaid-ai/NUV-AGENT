from __future__ import annotations

import json
import os
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path

from nuvion_app import build_info

DEFAULT_CONFIG_SCHEMA = "10"
DEFAULT_MODEL_POINTER = "anomalyclip/prod"
DEFAULT_MODEL_PROFILE = "runtime"


def _package_version() -> str:
    if build_info.AGENT_VERSION and build_info.AGENT_VERSION != "unknown":
        return build_info.AGENT_VERSION
    try:
        return metadata.version("nuv-agent")
    except metadata.PackageNotFoundError:
        return "unknown"


def _default_model_dir(environ: Mapping[str, str], pointer: str) -> Path:
    explicit = str(environ.get("NUVION_MODEL_LOCAL_DIR") or "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    root = Path(str(environ.get("NUVION_MODEL_DIR") or "~/.cache/nuvion/models")).expanduser()
    profile = str(environ.get("NUVION_MODEL_PROFILE") or DEFAULT_MODEL_PROFILE).strip().lower()
    identifier = f"server:{pointer}:{profile}".replace("/", "__").replace(":", "_")
    return (root / identifier).resolve()


def _read_resolved_model_version(model_dir: Path) -> str | None:
    metadata_path = model_dir / "metadata" / "server_presign_response.json"
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        payload = payload["data"]
    if not isinstance(payload, dict):
        return None
    value = str(payload.get("resolvedVersion") or "").strip()
    return value or None


def build_runtime_telemetry(
    *,
    environ: Mapping[str, str] | None = None,
    model_dir: str | Path | None = None,
    agent_version: str | None = None,
    component_sha: str | None = None,
) -> dict[str, str]:
    values = os.environ if environ is None else environ
    pointer = str(values.get("NUVION_MODEL_POINTER") or DEFAULT_MODEL_POINTER).strip()
    resolved_model_dir = Path(model_dir).expanduser().resolve() if model_dir else _default_model_dir(values, pointer)
    model_version = str(values.get("NUVION_MODEL_VERSION") or "").strip()
    if not model_version:
        model_version = _read_resolved_model_version(resolved_model_dir) or "unknown"

    resolved_agent_version = str(
        agent_version
        or values.get("NUVION_AGENT_VERSION")
        or _package_version()
    ).strip() or "unknown"
    resolved_component_sha = str(
        component_sha
        or values.get("NUVION_COMPONENT_SHA")
        or build_info.COMPONENT_SHA
        or "unknown"
    ).strip() or "unknown"

    return {
        "agentVersion": resolved_agent_version,
        "componentSha": resolved_component_sha,
        "configSchema": str(values.get("NUVION_CONFIG_SCHEMA_VERSION") or DEFAULT_CONFIG_SCHEMA).strip(),
        "modelPointer": pointer,
        "modelVersion": model_version,
    }
