from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from typing import Any

from nuvion_app import build_info
from nuvion_app.runtime.platform_identity import (
    PlatformIdentity,
    resolve_platform_identity,
)
from nuvion_app.runtime.release_bom import (
    ReleaseBomValidationError,
    VerifiedReleaseBom,
    load_release_bom,
)

DEFAULT_CONFIG_SCHEMA = "11"
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
    root = Path(
        str(environ.get("NUVION_MODEL_DIR") or "~/.cache/nuvion/models")
    ).expanduser()
    profile = (
        str(environ.get("NUVION_MODEL_PROFILE") or DEFAULT_MODEL_PROFILE)
        .strip()
        .lower()
    )
    identifier = f"server:{pointer}:{profile}".replace("/", "__").replace(":", "_")
    return (root / identifier).resolve()


def _read_server_model_identity(model_dir: Path) -> tuple[str | None, str | None]:
    metadata_path = model_dir / "metadata" / "server_presign_response.json"
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None, None
    if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        payload = payload["data"]
    if not isinstance(payload, dict):
        return None, None
    version = str(payload.get("resolvedVersion") or "").strip() or None
    digest = (
        str(
            payload.get("modelDigest")
            or payload.get("bundleDigest")
            or payload.get("manifestDigest")
            or ""
        ).strip()
        or None
    )
    return version, digest


def _artifact_manifest_digest(model_dir: Path) -> str | None:
    metadata_path = model_dir / "metadata" / "downloaded_from_server.json"
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, list):
        return None
    entries: list[tuple[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            return None
        key = str(item.get("key") or "").strip()
        digest = str(item.get("sha256") or "").strip().lower()
        if (
            not key
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            return None
        entries.append((key, digest))
    if not entries:
        return None
    canonical = "".join(f"{key}:{digest}\n" for key, digest in sorted(entries)).encode(
        "utf-8"
    )
    return hashlib.sha256(canonical).hexdigest()


def _build_info_value(name: str) -> str:
    return str(getattr(build_info, name, "") or "").strip()


def _load_configured_release_bom(
    values: Mapping[str, str],
) -> tuple[VerifiedReleaseBom | None, str, str | None]:
    raw_path = str(values.get("NUVION_RELEASE_BOM_PATH") or "").strip()
    if not raw_path:
        return None, "UNCONFIGURED", None
    expected_digest = (
        str(values.get("NUVION_EXPECTED_BOM_DIGEST") or "").strip() or None
    )
    try:
        bom = load_release_bom(raw_path, expected_bom_digest=expected_digest)
    except ReleaseBomValidationError as exc:
        return None, "INVALID", str(exc)[:200]
    return bom, "VERIFIED" if expected_digest else "SELF_CONSISTENT", None


def build_runtime_telemetry(
    *,
    environ: Mapping[str, str] | None = None,
    model_dir: str | Path | None = None,
    agent_version: str | None = None,
    component_sha: str | None = None,
    platform_identity: PlatformIdentity | None = None,
) -> dict[str, Any]:
    values = os.environ if environ is None else environ
    pointer = str(values.get("NUVION_MODEL_POINTER") or DEFAULT_MODEL_POINTER).strip()
    resolved_model_dir = (
        Path(model_dir).expanduser().resolve()
        if model_dir
        else _default_model_dir(values, pointer)
    )
    metadata_model_version, metadata_model_digest = _read_server_model_identity(
        resolved_model_dir
    )
    model_version = str(values.get("NUVION_MODEL_VERSION") or "").strip()
    if not model_version:
        model_version = metadata_model_version or "unknown"
    model_digest = str(values.get("NUVION_MODEL_DIGEST") or "").strip()
    if not model_digest:
        model_digest = (
            metadata_model_digest
            or _artifact_manifest_digest(resolved_model_dir)
            or "unknown"
        )

    resolved_agent_version = (
        str(
            agent_version or values.get("NUVION_AGENT_VERSION") or _package_version()
        ).strip()
        or "unknown"
    )
    resolved_component_sha = (
        str(
            component_sha
            or values.get("NUVION_COMPONENT_SHA")
            or build_info.COMPONENT_SHA
            or "unknown"
        ).strip()
        or "unknown"
    )

    identity = platform_identity or resolve_platform_identity(environ=values)
    release_bom, bom_status, bom_error = _load_configured_release_bom(values)
    if release_bom is not None:
        runtime_matches = (
            release_bom.agent_version == resolved_agent_version
            and release_bom.component_sha == resolved_component_sha
            and release_bom.config_schema
            == str(
                values.get("NUVION_CONFIG_SCHEMA_VERSION") or DEFAULT_CONFIG_SCHEMA
            ).strip()
        )
        platform_matches = identity.platform_profile in release_bom.platform_profiles
        if not runtime_matches:
            bom_status = "RUNTIME_MISMATCH"
        elif not platform_matches:
            bom_status = "PLATFORM_MISMATCH"

    bom_telemetry = release_bom.to_telemetry() if release_bom is not None else {}
    result: dict[str, Any] = {
        "agentVersion": resolved_agent_version,
        "componentSha": resolved_component_sha,
        "configSchema": str(
            values.get("NUVION_CONFIG_SCHEMA_VERSION") or DEFAULT_CONFIG_SCHEMA
        ).strip(),
        "modelPointer": pointer,
        "modelVersion": model_version,
        "modelDigest": model_digest,
        "updaterVersion": str(
            bom_telemetry.get("updaterVersion")
            or values.get("NUVION_UPDATER_VERSION")
            or _build_info_value("UPDATER_VERSION")
            or "unknown"
        ).strip(),
        "bomId": str(
            bom_telemetry.get("bomId")
            or values.get("NUVION_BOM_ID")
            or _build_info_value("BOM_ID")
            or "unknown"
        ).strip(),
        "bomDigest": str(
            bom_telemetry.get("bomDigest")
            or values.get("NUVION_BOM_DIGEST")
            or _build_info_value("BOM_DIGEST")
            or "unknown"
        ).strip(),
        "artifactDigest": str(
            bom_telemetry.get("artifactDigest")
            or values.get("NUVION_ARTIFACT_DIGEST")
            or _build_info_value("ARTIFACT_DIGEST")
            or "unknown"
        ).strip(),
        "bomVerificationStatus": bom_status,
    }
    if bom_error:
        result["bomVerificationError"] = bom_error
    result.update(identity.to_telemetry())
    # Keep the established flat fields during rollout while giving BE/FE one
    # evolvable object for platform, component and BOM details.
    result["runtimeTelemetry"] = dict(result)
    return result
