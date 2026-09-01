from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

RELEASE_BOM_SCHEMA_VERSION = 1
MAX_RELEASE_BOM_BYTES = 1024 * 1024

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMPONENT_SHA = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$")
_CONFIG_SCHEMA = re.compile(r"^[1-9][0-9]*$")
_BOM_ID = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._+-]{0,254}$")
_ARTIFACT_BASENAME = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._+-]{0,254}$")
_VALID_PLATFORM_PROFILES = frozenset(
    {"rpi5_deepx_dx_m1", "ventuno_q", "jetson_orin_nx", "macos_dev"}
)
_VALID_ARTIFACT_KINDS = frozenset({"python-sdist", "deb", "model", "config"})


class ReleaseBomValidationError(ValueError):
    pass


@dataclass(frozen=True)
class VerifiedReleaseBom:
    bom_id: str
    bom_digest: str
    agent_version: str
    component_sha: str
    config_schema: str
    updater_version: str
    platform_profiles: tuple[str, ...]
    artifact_name: str
    artifact_kind: str
    artifact_sha256: str
    artifact_size_bytes: int
    built_at: str

    def to_telemetry(self) -> dict[str, str]:
        return {
            "bomId": self.bom_id,
            "bomDigest": f"sha256:{self.bom_digest}",
            "artifactDigest": f"sha256:{self.artifact_sha256}",
            "updaterVersion": self.updater_version,
        }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseBomValidationError(f"duplicate JSON member: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ReleaseBomValidationError(f"non-standard JSON constant: {value}")


def _canonical_json(payload: dict[str, Any]) -> bytes:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ReleaseBomValidationError("BOM must be canonical JSON data") from exc


def compute_bom_digest(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("bomDigest", None)
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def _required_string(
    payload: dict[str, Any], key: str, *, max_length: int = 255
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value or value != value.strip():
        raise ReleaseBomValidationError(f"{key} must be a trimmed non-empty string")
    if len(value) > max_length:
        raise ReleaseBomValidationError(f"{key} exceeds {max_length} characters")
    return value


def _parse_built_at(value: str) -> None:
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ReleaseBomValidationError("builtAt must be an RFC3339 date-time") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ReleaseBomValidationError("builtAt must include a timezone")


def verify_release_bom(
    payload: dict[str, Any],
    *,
    expected_bom_digest: str | None = None,
) -> VerifiedReleaseBom:
    if not isinstance(payload, dict):
        raise ReleaseBomValidationError("release BOM root must be an object")
    expected_keys = {
        "schemaVersion",
        "bomId",
        "bomDigest",
        "agentVersion",
        "componentSha",
        "configSchema",
        "updaterVersion",
        "platformProfiles",
        "artifact",
        "builtAt",
    }
    if set(payload) != expected_keys:
        raise ReleaseBomValidationError("BOM fields do not match release-bom-v1")
    schema_version = payload.get("schemaVersion")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != RELEASE_BOM_SCHEMA_VERSION
    ):
        raise ReleaseBomValidationError("unsupported BOM schemaVersion")

    bom_id = _required_string(payload, "bomId")
    if not _BOM_ID.fullmatch(bom_id):
        raise ReleaseBomValidationError("bomId contains unsafe characters")
    bom_digest = _required_string(payload, "bomDigest", max_length=71)
    if not bom_digest.startswith("sha256:") or not _SHA256.fullmatch(bom_digest[7:]):
        raise ReleaseBomValidationError("bomDigest must be sha256:<64 lowercase hex>")
    computed_digest = compute_bom_digest(payload)
    if bom_digest[7:] != computed_digest:
        raise ReleaseBomValidationError(
            "bomDigest does not match canonical BOM content"
        )
    if expected_bom_digest is not None:
        normalized_expected = expected_bom_digest.removeprefix("sha256:").lower()
        if (
            not _SHA256.fullmatch(normalized_expected)
            or normalized_expected != computed_digest
        ):
            raise ReleaseBomValidationError(
                "BOM does not match the trusted expected digest"
            )

    agent_version = _required_string(payload, "agentVersion", max_length=100)
    if not _SEMVER.fullmatch(agent_version):
        raise ReleaseBomValidationError("agentVersion must be semantic version text")
    component_sha = _required_string(payload, "componentSha", max_length=64)
    if not _COMPONENT_SHA.fullmatch(component_sha):
        raise ReleaseBomValidationError(
            "componentSha must be a full lowercase commit/tree SHA"
        )
    config_schema = _required_string(payload, "configSchema", max_length=20)
    if not _CONFIG_SCHEMA.fullmatch(config_schema):
        raise ReleaseBomValidationError(
            "configSchema must be a positive integer string"
        )
    updater_version = _required_string(payload, "updaterVersion", max_length=100)

    raw_profiles = payload.get("platformProfiles")
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ReleaseBomValidationError("platformProfiles must be a non-empty array")
    if any(not isinstance(profile, str) for profile in raw_profiles):
        raise ReleaseBomValidationError("platformProfiles entries must be strings")
    profiles = tuple(raw_profiles)
    if list(profiles) != sorted(set(profiles)):
        raise ReleaseBomValidationError("platformProfiles must be unique and sorted")
    if not set(profiles).issubset(_VALID_PLATFORM_PROFILES):
        raise ReleaseBomValidationError(
            "platformProfiles contains an unsupported profile"
        )

    artifact = payload.get("artifact")
    if not isinstance(artifact, dict) or set(artifact) != {
        "name",
        "kind",
        "sha256",
        "sizeBytes",
    }:
        raise ReleaseBomValidationError("artifact fields do not match release-bom-v1")
    artifact_name = _required_string(artifact, "name")
    if (
        Path(artifact_name).name != artifact_name
        or artifact_name in {".", ".."}
        or not _ARTIFACT_BASENAME.fullmatch(artifact_name)
    ):
        raise ReleaseBomValidationError("artifact.name must be a safe basename")
    artifact_kind = _required_string(artifact, "kind", max_length=50)
    if artifact_kind not in _VALID_ARTIFACT_KINDS:
        raise ReleaseBomValidationError("artifact.kind is unsupported")
    artifact_sha256 = _required_string(artifact, "sha256", max_length=64)
    if not _SHA256.fullmatch(artifact_sha256):
        raise ReleaseBomValidationError("artifact.sha256 must be 64 lowercase hex")
    artifact_size = artifact.get("sizeBytes")
    if (
        isinstance(artifact_size, bool)
        or not isinstance(artifact_size, int)
        or artifact_size < 1
    ):
        raise ReleaseBomValidationError("artifact.sizeBytes must be a positive integer")

    built_at = _required_string(payload, "builtAt", max_length=50)
    _parse_built_at(built_at)
    return VerifiedReleaseBom(
        bom_id=bom_id,
        bom_digest=computed_digest,
        agent_version=agent_version,
        component_sha=component_sha,
        config_schema=config_schema,
        updater_version=updater_version,
        platform_profiles=profiles,
        artifact_name=artifact_name,
        artifact_kind=artifact_kind,
        artifact_sha256=artifact_sha256,
        artifact_size_bytes=artifact_size,
        built_at=built_at,
    )


def load_release_bom(
    path: str | Path,
    *,
    expected_bom_digest: str | None = None,
) -> VerifiedReleaseBom:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        initial_path_metadata = candidate.lstat()
    except OSError as exc:
        raise ReleaseBomValidationError(f"cannot stat release BOM: {exc}") from exc
    if stat.S_ISLNK(initial_path_metadata.st_mode):
        raise ReleaseBomValidationError("release BOM path must not be a symbolic link")
    if not stat.S_ISREG(initial_path_metadata.st_mode):
        raise ReleaseBomValidationError("release BOM path must be a regular file")
    if initial_path_metadata.st_size > MAX_RELEASE_BOM_BYTES:
        raise ReleaseBomValidationError("release BOM exceeds size limit")

    bom_path = candidate.resolve(strict=True)
    open_flags = os.O_RDONLY
    open_flags |= getattr(os, "O_CLOEXEC", 0)
    open_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(bom_path, open_flags)
        with os.fdopen(descriptor, "rb", closefd=True) as bom_file:
            before = os.fstat(bom_file.fileno())
            raw = bom_file.read(MAX_RELEASE_BOM_BYTES + 1)
            after = os.fstat(bom_file.fileno())
        final_path_metadata = bom_path.stat(follow_symlinks=False)
        final_candidate_metadata = candidate.lstat()
    except OSError as exc:
        raise ReleaseBomValidationError(f"cannot read release BOM: {exc}") from exc
    expected_identity = _artifact_identity(before)
    if (
        expected_identity != _artifact_identity(initial_path_metadata)
        or expected_identity != _artifact_identity(after)
        or expected_identity != _artifact_identity(final_path_metadata)
        or expected_identity != _artifact_identity(final_candidate_metadata)
    ):
        raise ReleaseBomValidationError("release BOM changed while being read")
    if len(raw) > MAX_RELEASE_BOM_BYTES:
        raise ReleaseBomValidationError("release BOM exceeds size limit")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise ReleaseBomValidationError("release BOM is not strict UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ReleaseBomValidationError("release BOM root must be an object")
    return verify_release_bom(payload, expected_bom_digest=expected_bom_digest)


def _artifact_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _hash_release_artifact(artifact_path: str | Path) -> tuple[Path, str, int]:
    candidate = Path(artifact_path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        initial_path_metadata = candidate.lstat()
    except OSError as exc:
        raise ReleaseBomValidationError(f"cannot stat artifact: {exc}") from exc
    if stat.S_ISLNK(initial_path_metadata.st_mode):
        raise ReleaseBomValidationError("artifact path must not be a symbolic link")
    if not stat.S_ISREG(initial_path_metadata.st_mode):
        raise ReleaseBomValidationError("artifact path must be a regular file")

    path = candidate.resolve(strict=True)
    open_flags = os.O_RDONLY
    open_flags |= getattr(os, "O_CLOEXEC", 0)
    open_flags |= getattr(os, "O_NOFOLLOW", 0)
    digest = hashlib.sha256()
    bytes_read = 0
    try:
        descriptor = os.open(path, open_flags)
        with os.fdopen(descriptor, "rb", closefd=True) as artifact_file:
            before = os.fstat(artifact_file.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ReleaseBomValidationError("artifact path must be a regular file")
            if before.st_size < 1:
                raise ReleaseBomValidationError("artifact must not be empty")
            while chunk := artifact_file.read(1024 * 1024):
                bytes_read += len(chunk)
                digest.update(chunk)
            after = os.fstat(artifact_file.fileno())
        final_path_metadata = path.stat(follow_symlinks=False)
        final_candidate_metadata = candidate.lstat()
    except OSError as exc:
        raise ReleaseBomValidationError(f"cannot hash artifact: {exc}") from exc

    expected_identity = _artifact_identity(before)
    if (
        bytes_read != before.st_size
        or expected_identity != _artifact_identity(initial_path_metadata)
        or expected_identity != _artifact_identity(after)
        or expected_identity != _artifact_identity(final_path_metadata)
        or expected_identity != _artifact_identity(final_candidate_metadata)
    ):
        raise ReleaseBomValidationError("artifact changed while the BOM was generated")
    return path, digest.hexdigest(), before.st_size


def verify_release_artifact(
    bom: VerifiedReleaseBom,
    artifact_path: str | Path,
) -> None:
    path, digest, size = _hash_release_artifact(artifact_path)
    if path.name != bom.artifact_name:
        raise ReleaseBomValidationError(
            "artifact basename does not match the release BOM"
        )
    if digest != bom.artifact_sha256 or size != bom.artifact_size_bytes:
        raise ReleaseBomValidationError(
            "artifact digest or size does not match the release BOM"
        )


def build_release_bom_payload(
    *,
    bom_id: str,
    agent_version: str,
    component_sha: str,
    config_schema: str,
    updater_version: str,
    platform_profiles: list[str],
    artifact_path: str | Path,
    artifact_kind: str,
    built_at: str,
) -> dict[str, Any]:
    path, artifact_digest, artifact_size = _hash_release_artifact(artifact_path)
    payload: dict[str, Any] = {
        "schemaVersion": RELEASE_BOM_SCHEMA_VERSION,
        "bomId": bom_id,
        "agentVersion": agent_version,
        "componentSha": component_sha,
        "configSchema": config_schema,
        "updaterVersion": updater_version,
        "platformProfiles": sorted(set(platform_profiles)),
        "artifact": {
            "name": path.name,
            "kind": artifact_kind,
            "sha256": artifact_digest,
            "sizeBytes": artifact_size,
        },
        "builtAt": built_at,
    }
    payload["bomDigest"] = f"sha256:{compute_bom_digest(payload)}"
    verify_release_bom(payload)
    return payload


def canonical_release_bom_json(payload: dict[str, Any]) -> str:
    verify_release_bom(payload)
    return (
        json.dumps(
            payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
        )
        + "\n"
    )
