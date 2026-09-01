from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from base64 import b64decode, b64encode
from binascii import Error as BinasciiError
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

RELEASE_BOM_SCHEMA_VERSION = 1
RELEASE_BOM_V2_SCHEMA_VERSION = 2
RELEASE_BOM_SIGNATURE_SCHEMA_VERSION = 1
MAX_RELEASE_BOM_BYTES = 1024 * 1024
MAX_RELEASE_SIGNATURE_BYTES = 16 * 1024
MAX_RELEASE_SEQUENCE = 9_223_372_036_854_775_807

_RELEASE_SIGNATURE_DOMAIN = b"NUVION-RELEASE-BOM-V2\x00"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMPONENT_SHA = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$")
_STRICT_SEMVER = re.compile(
    r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
    r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)
_CONFIG_SCHEMA = re.compile(r"^[1-9][0-9]*$")
_BOM_ID = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._+-]{0,254}$")
_ARTIFACT_BASENAME = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._+-]{0,254}$")
_KEY_ID = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._:@+-]{0,254}$")
_HARDWARE_REVISION = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._+-]{0,99}$")
_VALID_PLATFORM_PROFILES = frozenset(
    {
        "rpi5_deepx_dx_m1",
        "ventuno_q",
        "jetson_orin_nx",
        "iq9075_dev",
        "macos_dev",
    }
)
_VALID_V1_ARTIFACT_KINDS = frozenset(
    {"python-sdist", "deb", "model", "config"}
)
_VALID_V2_ARTIFACT_KINDS = _VALID_V1_ARTIFACT_KINDS | {"agent-bundle"}
_PRODUCT_PROFILE = {
    "NUVION": "rpi5_deepx_dx_m1",
    "NUVION_PRO": "ventuno_q",
    "NUVION_ULTRA": "jetson_orin_nx",
    "IQ9075_DEV": "iq9075_dev",
    "MACOS_DEV": "macos_dev",
}
_VALID_ARCHITECTURES = frozenset({"aarch64", "arm64", "x86_64", "amd64"})


class ReleaseBomValidationError(ValueError):
    pass


@dataclass(frozen=True, order=True)
class ReleaseTarget:
    product_model: str
    platform_profile: str
    hardware_revision: str
    architecture: str

    def to_payload(self) -> dict[str, str]:
        return {
            "productModel": self.product_model,
            "platformProfile": self.platform_profile,
            "hardwareRevision": self.hardware_revision,
            "architecture": self.architecture,
        }


@dataclass(frozen=True)
class VerifiedReleaseBom:
    schema_version: int
    bom_id: str
    bom_digest: str
    agent_version: str
    component_sha: str
    config_schema: str
    updater_version: str | None
    release_sequence: int | None
    min_updater_version: str | None
    targets: tuple[ReleaseTarget, ...]
    publisher_key_id: str | None
    platform_profiles: tuple[str, ...]
    artifact_name: str
    artifact_kind: str
    artifact_sha256: str
    artifact_size_bytes: int
    built_at: str

    def to_telemetry(self) -> dict[str, str | int]:
        telemetry: dict[str, str | int] = {
            "bomId": self.bom_id,
            "bomDigest": f"sha256:{self.bom_digest}",
            "artifactDigest": f"sha256:{self.artifact_sha256}",
        }
        # v1 recorded the updater version in the BOM. In v2 the BOM records a
        # minimum compatibility constraint instead, so runtime telemetry must
        # continue to source the actually installed updater version elsewhere.
        if self.updater_version is not None:
            telemetry["updaterVersion"] = self.updater_version
        if self.release_sequence is not None:
            telemetry["releaseSequence"] = self.release_sequence
        return telemetry


class ReleaseKeyring:
    """Pinned Ed25519 release-publisher public keys indexed by key id.

    Values follow the Fleet command keyring convention: a 32-byte raw Ed25519
    public key or DER SubjectPublicKeyInfo. Private key material is rejected.
    """

    def __init__(self, keys: Mapping[str, bytes]) -> None:
        parsed: dict[str, Ed25519PublicKey] = {}
        for raw_key_id, material in keys.items():
            key_id = str(raw_key_id or "").strip()
            if not _KEY_ID.fullmatch(key_id):
                raise ValueError("release key id contains unsafe characters")
            if not isinstance(material, bytes) or not material:
                raise ValueError(
                    f"release public key material for keyId={key_id} must be bytes"
                )
            parsed[key_id] = self._parse_public_key(material, key_id)
        if not parsed:
            raise ValueError("release keyring must contain at least one public key")
        self._keys = parsed

    @staticmethod
    def _parse_public_key(material: bytes, key_id: str) -> Ed25519PublicKey:
        if len(material) == 32:
            try:
                return Ed25519PublicKey.from_public_bytes(material)
            except ValueError as exc:
                raise ValueError(
                    f"invalid raw Ed25519 release public key for keyId={key_id}"
                ) from exc
        try:
            public_key = serialization.load_der_public_key(material)
        except ValueError as exc:
            raise ValueError(
                f"invalid DER release public key for keyId={key_id}"
            ) from exc
        if not isinstance(public_key, Ed25519PublicKey):
            raise TypeError(f"release public key for keyId={key_id} is not Ed25519")
        return public_key

    def get(self, key_id: str) -> Ed25519PublicKey | None:
        return self._keys.get(key_id)


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


def _verify_release_bom_v1(
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
    if artifact_kind not in _VALID_V1_ARTIFACT_KINDS:
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
        schema_version=RELEASE_BOM_SCHEMA_VERSION,
        bom_id=bom_id,
        bom_digest=computed_digest,
        agent_version=agent_version,
        component_sha=component_sha,
        config_schema=config_schema,
        updater_version=updater_version,
        release_sequence=None,
        min_updater_version=None,
        targets=(),
        publisher_key_id=None,
        platform_profiles=profiles,
        artifact_name=artifact_name,
        artifact_kind=artifact_kind,
        artifact_sha256=artifact_sha256,
        artifact_size_bytes=artifact_size,
        built_at=built_at,
    )


def _strict_semver_parts(value: str, field: str) -> tuple[int, int, int, str | None]:
    match = _STRICT_SEMVER.fullmatch(value)
    if match is None:
        raise ReleaseBomValidationError(f"{field} must be semantic version text")
    prerelease = match.group(4)
    if prerelease is not None:
        for identifier in prerelease.split("."):
            if (
                identifier.isdigit()
                and len(identifier) > 1
                and identifier.startswith("0")
            ):
                raise ReleaseBomValidationError(
                    f"{field} contains a numeric prerelease identifier "
                    "with a leading zero"
                )
    return int(match.group(1)), int(match.group(2)), int(match.group(3)), prerelease


def _parse_release_target(raw_target: Any) -> ReleaseTarget:
    expected_keys = {
        "productModel",
        "platformProfile",
        "hardwareRevision",
        "architecture",
    }
    if not isinstance(raw_target, dict) or set(raw_target) != expected_keys:
        raise ReleaseBomValidationError(
            "target fields do not match release-bom-v2"
        )
    product_model = _required_string(raw_target, "productModel", max_length=50)
    platform_profile = _required_string(
        raw_target, "platformProfile", max_length=100
    )
    hardware_revision = _required_string(
        raw_target, "hardwareRevision", max_length=100
    )
    architecture = _required_string(raw_target, "architecture", max_length=20)

    expected_profile = _PRODUCT_PROFILE.get(product_model)
    if expected_profile is None:
        raise ReleaseBomValidationError("target.productModel is unsupported")
    if platform_profile != expected_profile:
        raise ReleaseBomValidationError(
            "target platformProfile does not match productModel"
        )
    if (
        not _HARDWARE_REVISION.fullmatch(hardware_revision)
        or hardware_revision.lower() in {"all", "any", "unknown"}
    ):
        raise ReleaseBomValidationError(
            "target.hardwareRevision must be an exact safe revision"
        )
    if architecture not in _VALID_ARCHITECTURES:
        raise ReleaseBomValidationError("target.architecture is unsupported")
    return ReleaseTarget(
        product_model=product_model,
        platform_profile=platform_profile,
        hardware_revision=hardware_revision,
        architecture=architecture,
    )


def _verify_release_bom_v2(
    payload: dict[str, Any],
    *,
    expected_bom_digest: str | None = None,
) -> VerifiedReleaseBom:
    expected_keys = {
        "schemaVersion",
        "bomId",
        "bomDigest",
        "releaseSequence",
        "agentVersion",
        "componentSha",
        "configSchema",
        "minUpdaterVersion",
        "targets",
        "artifact",
        "builtAt",
    }
    if set(payload) != expected_keys:
        raise ReleaseBomValidationError("BOM fields do not match release-bom-v2")

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

    release_sequence = payload.get("releaseSequence")
    if (
        isinstance(release_sequence, bool)
        or not isinstance(release_sequence, int)
        or release_sequence < 1
        or release_sequence > MAX_RELEASE_SEQUENCE
    ):
        raise ReleaseBomValidationError(
            "releaseSequence must be a positive signed 64-bit integer"
        )

    agent_version = _required_string(payload, "agentVersion", max_length=100)
    _strict_semver_parts(agent_version, "agentVersion")
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
    min_updater_version = _required_string(
        payload, "minUpdaterVersion", max_length=100
    )
    _strict_semver_parts(min_updater_version, "minUpdaterVersion")

    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ReleaseBomValidationError("targets must be a non-empty array")
    targets = tuple(_parse_release_target(target) for target in raw_targets)
    if list(targets) != sorted(set(targets)):
        raise ReleaseBomValidationError("targets must be unique and sorted")

    artifact = payload.get("artifact")
    if not isinstance(artifact, dict) or set(artifact) != {
        "name",
        "kind",
        "sha256",
        "sizeBytes",
    }:
        raise ReleaseBomValidationError("artifact fields do not match release-bom-v2")
    artifact_name = _required_string(artifact, "name")
    if (
        Path(artifact_name).name != artifact_name
        or artifact_name in {".", ".."}
        or not _ARTIFACT_BASENAME.fullmatch(artifact_name)
    ):
        raise ReleaseBomValidationError("artifact.name must be a safe basename")
    artifact_kind = _required_string(artifact, "kind", max_length=50)
    if artifact_kind not in _VALID_V2_ARTIFACT_KINDS:
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
        schema_version=RELEASE_BOM_V2_SCHEMA_VERSION,
        bom_id=bom_id,
        bom_digest=computed_digest,
        agent_version=agent_version,
        component_sha=component_sha,
        config_schema=config_schema,
        updater_version=None,
        release_sequence=release_sequence,
        min_updater_version=min_updater_version,
        targets=targets,
        publisher_key_id=None,
        platform_profiles=tuple(
            sorted({target.platform_profile for target in targets})
        ),
        artifact_name=artifact_name,
        artifact_kind=artifact_kind,
        artifact_sha256=artifact_sha256,
        artifact_size_bytes=artifact_size,
        built_at=built_at,
    )


def verify_release_bom(
    payload: dict[str, Any],
    *,
    expected_bom_digest: str | None = None,
) -> VerifiedReleaseBom:
    """Verify BOM structure and content digest without authenticating its publisher.

    OTA activation code must use :func:`verify_signed_release_bom` or
    :func:`load_signed_release_bom` for schema v2 releases.
    """

    if not isinstance(payload, dict):
        raise ReleaseBomValidationError("release BOM root must be an object")
    schema_version = payload.get("schemaVersion")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ReleaseBomValidationError("unsupported BOM schemaVersion")
    if schema_version == RELEASE_BOM_SCHEMA_VERSION:
        return _verify_release_bom_v1(
            payload, expected_bom_digest=expected_bom_digest
        )
    if schema_version == RELEASE_BOM_V2_SCHEMA_VERSION:
        return _verify_release_bom_v2(
            payload, expected_bom_digest=expected_bom_digest
        )
    raise ReleaseBomValidationError("unsupported BOM schemaVersion")


def _load_strict_json_object(
    path: str | Path,
    *,
    max_bytes: int,
    label: str,
) -> dict[str, Any]:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        initial_path_metadata = candidate.lstat()
    except OSError as exc:
        raise ReleaseBomValidationError(f"cannot stat {label}: {exc}") from exc
    if stat.S_ISLNK(initial_path_metadata.st_mode):
        raise ReleaseBomValidationError(f"{label} path must not be a symbolic link")
    if not stat.S_ISREG(initial_path_metadata.st_mode):
        raise ReleaseBomValidationError(f"{label} path must be a regular file")
    if initial_path_metadata.st_size > max_bytes:
        raise ReleaseBomValidationError(f"{label} exceeds size limit")

    try:
        resolved_path = candidate.resolve(strict=True)
    except OSError as exc:
        raise ReleaseBomValidationError(f"cannot resolve {label}: {exc}") from exc
    open_flags = os.O_RDONLY
    open_flags |= getattr(os, "O_CLOEXEC", 0)
    open_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved_path, open_flags)
        with os.fdopen(descriptor, "rb", closefd=True) as bom_file:
            before = os.fstat(bom_file.fileno())
            raw = bom_file.read(max_bytes + 1)
            after = os.fstat(bom_file.fileno())
        final_path_metadata = resolved_path.stat(follow_symlinks=False)
        final_candidate_metadata = candidate.lstat()
    except OSError as exc:
        raise ReleaseBomValidationError(f"cannot read {label}: {exc}") from exc
    expected_identity = _artifact_identity(before)
    if (
        expected_identity != _artifact_identity(initial_path_metadata)
        or expected_identity != _artifact_identity(after)
        or expected_identity != _artifact_identity(final_path_metadata)
        or expected_identity != _artifact_identity(final_candidate_metadata)
    ):
        raise ReleaseBomValidationError(f"{label} changed while being read")
    if len(raw) > max_bytes:
        raise ReleaseBomValidationError(f"{label} exceeds size limit")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise ReleaseBomValidationError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ReleaseBomValidationError(f"{label} root must be an object")
    return payload


def load_release_bom(
    path: str | Path,
    *,
    expected_bom_digest: str | None = None,
) -> VerifiedReleaseBom:
    """Load a v1/v2 BOM and verify structure plus its content digest.

    This preserves the v1 telemetry loader contract. OTA activation code must
    use :func:`load_signed_release_bom` for publisher authentication.
    """

    payload = _load_strict_json_object(
        path, max_bytes=MAX_RELEASE_BOM_BYTES, label="release BOM"
    )
    return verify_release_bom(payload, expected_bom_digest=expected_bom_digest)


def _release_signature_message(payload: dict[str, Any]) -> bytes:
    return _RELEASE_SIGNATURE_DOMAIN + _canonical_json(payload)


def _parse_release_signature(signature_payload: dict[str, Any]) -> tuple[str, bytes]:
    expected_keys = {"schemaVersion", "keyId", "algorithm", "signature"}
    if set(signature_payload) != expected_keys:
        raise ReleaseBomValidationError(
            "signature fields do not match release-signature-v1"
        )
    schema_version = signature_payload.get("schemaVersion")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != RELEASE_BOM_SIGNATURE_SCHEMA_VERSION
    ):
        raise ReleaseBomValidationError("unsupported release signature schemaVersion")
    key_id = _required_string(signature_payload, "keyId")
    if not _KEY_ID.fullmatch(key_id):
        raise ReleaseBomValidationError("signature.keyId contains unsafe characters")
    algorithm = _required_string(signature_payload, "algorithm", max_length=20)
    if algorithm != "Ed25519":
        raise ReleaseBomValidationError("signature.algorithm must be Ed25519")
    encoded_signature = _required_string(
        signature_payload, "signature", max_length=128
    )
    try:
        signature = b64decode(encoded_signature, validate=True)
    except (BinasciiError, ValueError) as exc:
        raise ReleaseBomValidationError(
            "signature.signature must be canonical base64"
        ) from exc
    if (
        len(signature) != 64
        or b64encode(signature).decode("ascii") != encoded_signature
    ):
        raise ReleaseBomValidationError(
            "signature.signature must be a canonical base64 Ed25519 signature"
        )
    return key_id, signature


def build_release_bom_signature(
    payload: dict[str, Any],
    *,
    key_id: str,
    private_key: Ed25519PrivateKey,
) -> dict[str, Any]:
    """Create a detached publisher signature for an already-digested v2 BOM."""

    verified = verify_release_bom(payload)
    if verified.schema_version != RELEASE_BOM_V2_SCHEMA_VERSION:
        raise ReleaseBomValidationError("only release-bom-v2 can be signed")
    normalized_key_id = str(key_id or "").strip()
    if not _KEY_ID.fullmatch(normalized_key_id):
        raise ReleaseBomValidationError("signature.keyId contains unsafe characters")
    if not isinstance(private_key, Ed25519PrivateKey):
        raise TypeError("release signing private key must be Ed25519")
    signature = private_key.sign(_release_signature_message(payload))
    signature_payload: dict[str, Any] = {
        "schemaVersion": RELEASE_BOM_SIGNATURE_SCHEMA_VERSION,
        "keyId": normalized_key_id,
        "algorithm": "Ed25519",
        "signature": b64encode(signature).decode("ascii"),
    }
    _parse_release_signature(signature_payload)
    return signature_payload


def verify_signed_release_bom(
    payload: dict[str, Any],
    signature_payload: dict[str, Any],
    *,
    release_keyring: ReleaseKeyring,
    expected_bom_digest: str | None = None,
) -> VerifiedReleaseBom:
    """Verify v2 structure, digest and detached publisher signature."""

    verified = verify_release_bom(
        payload, expected_bom_digest=expected_bom_digest
    )
    if verified.schema_version != RELEASE_BOM_V2_SCHEMA_VERSION:
        raise ReleaseBomValidationError("signed release BOM must use schemaVersion 2")
    if not isinstance(signature_payload, dict):
        raise ReleaseBomValidationError("release signature root must be an object")
    key_id, signature = _parse_release_signature(signature_payload)
    public_key = release_keyring.get(key_id)
    if public_key is None:
        raise ReleaseBomValidationError("release signature keyId is not trusted")
    try:
        public_key.verify(signature, _release_signature_message(payload))
    except InvalidSignature as exc:
        raise ReleaseBomValidationError(
            "release BOM publisher signature verification failed"
        ) from exc
    return replace(verified, publisher_key_id=key_id)


def load_signed_release_bom(
    bom_path: str | Path,
    signature_path: str | Path,
    *,
    release_keyring: ReleaseKeyring,
    expected_bom_digest: str | None = None,
) -> VerifiedReleaseBom:
    """Race-safe load and publisher authentication for a v2 release BOM."""

    payload = _load_strict_json_object(
        bom_path, max_bytes=MAX_RELEASE_BOM_BYTES, label="release BOM"
    )
    signature_payload = _load_strict_json_object(
        signature_path,
        max_bytes=MAX_RELEASE_SIGNATURE_BYTES,
        label="release signature",
    )
    return verify_signed_release_bom(
        payload,
        signature_payload,
        release_keyring=release_keyring,
        expected_bom_digest=expected_bom_digest,
    )


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


def build_release_bom_v2_payload(
    *,
    bom_id: str,
    release_sequence: int,
    agent_version: str,
    component_sha: str,
    config_schema: str,
    min_updater_version: str,
    targets: list[ReleaseTarget] | tuple[ReleaseTarget, ...],
    artifact_path: str | Path,
    artifact_kind: str,
    built_at: str,
) -> dict[str, Any]:
    """Build a content-addressed v2 BOM ready for detached signing."""

    if any(not isinstance(target, ReleaseTarget) for target in targets):
        raise TypeError("targets must contain ReleaseTarget values")
    path, artifact_digest, artifact_size = _hash_release_artifact(artifact_path)
    payload: dict[str, Any] = {
        "schemaVersion": RELEASE_BOM_V2_SCHEMA_VERSION,
        "bomId": bom_id,
        "releaseSequence": release_sequence,
        "agentVersion": agent_version,
        "componentSha": component_sha,
        "configSchema": config_schema,
        "minUpdaterVersion": min_updater_version,
        "targets": [target.to_payload() for target in sorted(set(targets))],
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


def canonical_release_bom_signature_json(signature_payload: dict[str, Any]) -> str:
    if not isinstance(signature_payload, dict):
        raise ReleaseBomValidationError("release signature root must be an object")
    _parse_release_signature(signature_payload)
    return (
        json.dumps(
            signature_payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _compare_semver(left: str, right: str) -> int:
    left_major, left_minor, left_patch, left_prerelease = _strict_semver_parts(
        left, "updaterVersion"
    )
    right_major, right_minor, right_patch, right_prerelease = _strict_semver_parts(
        right, "minUpdaterVersion"
    )
    left_core = (left_major, left_minor, left_patch)
    right_core = (right_major, right_minor, right_patch)
    if left_core != right_core:
        return -1 if left_core < right_core else 1
    if left_prerelease is None:
        return 0 if right_prerelease is None else 1
    if right_prerelease is None:
        return -1

    left_identifiers = left_prerelease.split(".")
    right_identifiers = right_prerelease.split(".")
    for left_identifier, right_identifier in zip(
        left_identifiers, right_identifiers
    ):
        if left_identifier == right_identifier:
            continue
        left_is_numeric = left_identifier.isdigit()
        right_is_numeric = right_identifier.isdigit()
        if left_is_numeric and right_is_numeric:
            return -1 if int(left_identifier) < int(right_identifier) else 1
        if left_is_numeric != right_is_numeric:
            return -1 if left_is_numeric else 1
        return -1 if left_identifier < right_identifier else 1
    if len(left_identifiers) == len(right_identifiers):
        return 0
    return -1 if len(left_identifiers) < len(right_identifiers) else 1


def assert_minimum_updater_version(
    bom: VerifiedReleaseBom,
    *,
    current_updater_version: str,
) -> None:
    if (
        bom.schema_version != RELEASE_BOM_V2_SCHEMA_VERSION
        or bom.min_updater_version is None
    ):
        raise ReleaseBomValidationError(
            "minimum updater compatibility requires release-bom-v2"
        )
    if _compare_semver(current_updater_version, bom.min_updater_version) < 0:
        raise ReleaseBomValidationError(
            "installed updater version is below minUpdaterVersion"
        )


def assert_release_compatible(
    bom: VerifiedReleaseBom,
    *,
    product_model: str,
    platform_profile: str,
    hardware_revision: str,
    architecture: str,
    current_updater_version: str,
) -> ReleaseTarget:
    """Require an exact v2 target match and a compatible updater version."""

    if bom.schema_version != RELEASE_BOM_V2_SCHEMA_VERSION:
        raise ReleaseBomValidationError(
            "exact target compatibility requires release-bom-v2"
        )
    current_target = _parse_release_target(
        {
            "productModel": product_model,
            "platformProfile": platform_profile,
            "hardwareRevision": hardware_revision,
            "architecture": architecture,
        }
    )
    if current_target not in bom.targets:
        raise ReleaseBomValidationError(
            "release BOM has no exact target for this device"
        )
    assert_minimum_updater_version(
        bom, current_updater_version=current_updater_version
    )
    return current_target


def assert_release_sequence_allowed(
    bom: VerifiedReleaseBom,
    *,
    current_release_sequence: int,
    current_bom_digest: str | None = None,
) -> None:
    """Reject downgrade and same-sequence equivocation while allowing replay."""

    if (
        bom.schema_version != RELEASE_BOM_V2_SCHEMA_VERSION
        or bom.release_sequence is None
    ):
        raise ReleaseBomValidationError(
            "anti-downgrade verification requires release-bom-v2"
        )
    if (
        isinstance(current_release_sequence, bool)
        or not isinstance(current_release_sequence, int)
        or current_release_sequence < 0
        or current_release_sequence > MAX_RELEASE_SEQUENCE
    ):
        raise ReleaseBomValidationError(
            "current release sequence must be a non-negative signed 64-bit integer"
        )
    if bom.release_sequence < current_release_sequence:
        raise ReleaseBomValidationError("releaseSequence downgrade is not allowed")
    if bom.release_sequence != current_release_sequence:
        return
    if current_bom_digest is None:
        raise ReleaseBomValidationError(
            "current BOM digest is required when releaseSequence is unchanged"
        )
    normalized_digest = current_bom_digest.removeprefix("sha256:").lower()
    if not _SHA256.fullmatch(normalized_digest):
        raise ReleaseBomValidationError("current BOM digest is invalid")
    if normalized_digest != bom.bom_digest:
        raise ReleaseBomValidationError(
            "different BOM cannot reuse the current releaseSequence"
        )
