from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
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

DEFAULT_CONFIG_SCHEMA = "12"
DEFAULT_MODEL_POINTER = "anomalyclip/prod"
DEFAULT_MODEL_PROFILE = "runtime"
FUNCTIONAL_HEALTH_VALUES = frozenset(
    {"FUNCTIONAL_HEALTHY", "FUNCTIONAL_UNHEALTHY"}
)
UPDATE_PHASE_VALUES = frozenset(
    {
        "IDLE",
        "STAGED",
        "INSTALLING",
        "VERIFYING",
        "SUCCEEDED",
        "ROLLED_BACK",
        "FAILED",
    }
)
_RUNTIME_PUBLIC_STATE_KEYS = frozenset(
    {
        "functionalHealth",
        "updatePhase",
        "updateEvidence",
        "targetVersion",
        "agentVersion",
        "bomDigest",
        "artifactDigest",
        "componentSha",
        "configSchema",
        "releaseSequence",
        "bomVerificationStatus",
    }
)


def merge_runtime_public_state(
    telemetry: Mapping[str, Any],
    public_state: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Merge a fail-closed updater/health adapter into runtime telemetry."""

    merged = dict(telemetry)
    if public_state is None:
        return merged
    state = dict(public_state)
    unknown = set(state) - _RUNTIME_PUBLIC_STATE_KEYS
    if unknown:
        raise ValueError(
            "runtime public state contains unsupported fields: "
            + ",".join(sorted(unknown))
        )
    functional_health = state.get("functionalHealth")
    if (
        functional_health is not None
        and functional_health not in FUNCTIONAL_HEALTH_VALUES
    ):
        raise ValueError("functionalHealth is not canonical")
    update_phase = state.get("updatePhase")
    if update_phase is not None and update_phase not in UPDATE_PHASE_VALUES:
        raise ValueError("updatePhase is not canonical")
    evidence = state.get("updateEvidence")
    if evidence is not None and not isinstance(evidence, Mapping):
        raise ValueError("updateEvidence must be an object")
    if update_phase == "ROLLED_BACK" and (not isinstance(evidence, Mapping) or not evidence):
        raise ValueError("ROLLED_BACK requires persistent updateEvidence")
    merged.update(state)
    return merged


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


def _read_server_model_identity(
    model_dir: Path,
) -> tuple[str | None, str | None, str | None]:
    metadata_path = model_dir / "metadata" / "server_presign_response.json"
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None, None, None
    if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        payload = payload["data"]
    if not isinstance(payload, dict):
        return None, None, None
    pointer = str(payload.get("pointer") or "").strip() or None
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
    return pointer, version, digest


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _verified_artifact_digests(model_dir: Path) -> tuple[str | None, str | None]:
    """Return (manifest-file digest, aggregate digest) after hashing every artifact."""

    metadata_path = model_dir / "metadata" / "downloaded_from_server.json"
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None, None
    if not isinstance(payload, list):
        return None, None
    entries: list[tuple[str, str]] = []
    seen_keys: set[str] = set()
    manifest_digest: str | None = None
    root = model_dir.resolve()
    for item in payload:
        if not isinstance(item, dict):
            return None, None
        key = str(item.get("key") or "").strip()
        digest = str(item.get("sha256") or "").strip().lower()
        raw_destination = str(item.get("dst") or "").strip()
        if (
            not key
            or not raw_destination
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            return None, None
        if key in seen_keys:
            return None, None
        seen_keys.add(key)
        destination = Path(raw_destination).expanduser()
        try:
            if stat.S_ISLNK(destination.lstat().st_mode):
                return None, None
            resolved_destination = destination.resolve(strict=True)
            resolved_destination.relative_to(root)
            actual_digest = _sha256_file(resolved_destination)
        except (OSError, ValueError):
            return None, None
        if actual_digest != digest:
            return None, None
        entries.append((key, digest))
        if key == "manifest":
            manifest_digest = "sha256:" + actual_digest
    if not entries:
        return None, None
    canonical = "".join(f"{key}:{digest}\n" for key, digest in sorted(entries)).encode(
        "utf-8"
    )
    return manifest_digest, "sha256:" + hashlib.sha256(canonical).hexdigest()


def _artifact_manifest_digest(model_dir: Path) -> str | None:
    """Compatibility helper returning only verified, content-derived evidence."""

    manifest_digest, aggregate_digest = _verified_artifact_digests(model_dir)
    return manifest_digest or aggregate_digest


def verify_model_artifact_identity(
    model_dir: str | Path,
    *,
    expected_pointer: str,
    expected_digest: str,
) -> dict[str, str] | None:
    """Verify a resolver pointer and signed digest against actual local bytes.

    The expected environment value is deliberately not consulted.  A successful
    result requires persisted resolver metadata plus a digest derived by hashing
    every downloaded artifact.  The signed digest may name either the downloaded
    manifest file or the canonical aggregate manifest.
    """

    root = Path(model_dir).expanduser().resolve()
    observed_pointer, _version, _declared_digest = _read_server_model_identity(root)
    manifest_digest, aggregate_digest = _verified_artifact_digests(root)
    normalized_pointer = str(expected_pointer or "").strip()
    normalized_digest = str(expected_digest or "").strip().lower()
    candidates = {value for value in (manifest_digest, aggregate_digest) if value}
    if observed_pointer != normalized_pointer or normalized_digest not in candidates:
        return None
    return {"pointer": observed_pointer, "digest": normalized_digest}


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
    effect_capabilities: AbstractSet[str] = frozenset(),
    public_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    values = os.environ if environ is None else environ
    pointer = str(values.get("NUVION_MODEL_POINTER") or DEFAULT_MODEL_POINTER).strip()
    resolved_model_dir = (
        Path(model_dir).expanduser().resolve()
        if model_dir
        else _default_model_dir(values, pointer)
    )
    metadata_model_pointer, metadata_model_version, metadata_model_digest = _read_server_model_identity(
        resolved_model_dir
    )
    model_version = str(values.get("NUVION_MODEL_VERSION") or "").strip()
    if not model_version:
        model_version = metadata_model_version or "unknown"
    expected_model_digest = str(values.get("NUVION_MODEL_DIGEST") or "").strip()
    observed_artifact_digest = _artifact_manifest_digest(resolved_model_dir)
    # Expected config is not actual evidence.  Only byte-verified artifact
    # metadata is published as modelDigest.  The resolver-declared value remains
    # useful context but cannot override the observed digest.
    model_digest = observed_artifact_digest or "unknown"

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
    raw_functional_health = str(
        values.get("NUVION_FUNCTIONAL_HEALTH") or "FUNCTIONAL_UNHEALTHY"
    ).strip().upper()
    functional_health = (
        raw_functional_health
        if raw_functional_health
        in {"FUNCTIONAL_HEALTHY", "FUNCTIONAL_UNHEALTHY"}
        else "FUNCTIONAL_UNHEALTHY"
    )
    result: dict[str, Any] = {
        "agentVersion": resolved_agent_version,
        "componentSha": resolved_component_sha,
        "configSchema": str(
            values.get("NUVION_CONFIG_SCHEMA_VERSION") or DEFAULT_CONFIG_SCHEMA
        ).strip(),
        "modelPointer": pointer,
        "modelObservedPointer": (
            metadata_model_pointer if observed_artifact_digest else "unknown"
        ),
        "modelVersion": model_version,
        "modelDigest": model_digest,
        # An environment value, build constant, or BOM requirement proves
        # neither that the privileged updater is installed nor that it is
        # alive. Fresh peer-authenticated STATUS telemetry is merged by the
        # heartbeat provider; the static snapshot must fail closed.
        "updaterVersion": "unknown",
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
        "functionalHealth": functional_health,
    }
    if expected_model_digest:
        result["modelExpectedDigest"] = expected_model_digest
        result["modelDigestMatchesExpected"] = model_digest == expected_model_digest
    if metadata_model_digest:
        result["modelResolverDigest"] = metadata_model_digest
    if metadata_model_pointer:
        result["modelResolverPointer"] = metadata_model_pointer
    release_sequence = bom_telemetry.get("releaseSequence")
    if isinstance(release_sequence, int) and not isinstance(release_sequence, bool):
        result["releaseSequence"] = release_sequence
    if bom_error:
        result["bomVerificationError"] = bom_error
    result.update(identity.to_telemetry())
    result["capabilities"] = sorted(
        set(identity.capabilities) | {str(value) for value in effect_capabilities}
    )
    result = merge_runtime_public_state(result, public_state)
    # Keep the established flat fields during rollout while giving BE/FE one
    # evolvable object for platform, component and BOM details.
    result["runtimeTelemetry"] = dict(result)
    return result
