from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nuvion_app.inference.fleet_command import (
    Ed25519Keyring,
    FleetCommandVerifier,
)
from nuvion_app.runtime.release_bom import ReleaseKeyring
from nuvion_updater.errors import UpdaterSecurityError
from nuvion_updater.secure_io import read_fixed_regular_file

MAX_TRUST_FILE_BYTES = 64 * 1024


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise UpdaterSecurityError(
                "INVALID_TRUST_FILE", f"duplicate JSON member: {key}"
            )
        result[key] = value
    return result


def _load_json(path: str | Path, *, require_root_owner: bool) -> dict[str, Any]:
    candidate = Path(path)
    raw = read_fixed_regular_file(
        candidate.parent,
        candidate.name,
        max_bytes=MAX_TRUST_FILE_BYTES,
        require_root_owner=require_root_owner,
        require_private=False,
    )
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UpdaterSecurityError(
            "INVALID_TRUST_FILE", "trust file is not strict UTF-8 JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise UpdaterSecurityError(
            "INVALID_TRUST_FILE", "trust file root must be an object"
        )
    return payload


@dataclass(frozen=True)
class DeviceBinding:
    trust_domain: str
    device_id: str
    space_id: int
    product_model: str
    platform_profile: str
    hardware_revision: str
    architecture: str
    docker_required: bool


def load_device_binding(
    path: str | Path,
    *,
    require_root_owner: bool = True,
) -> DeviceBinding:
    payload = _load_json(path, require_root_owner=require_root_owner)
    expected = {
        "schemaVersion",
        "trustDomain",
        "deviceId",
        "spaceId",
        "productModel",
        "platformProfile",
        "hardwareRevision",
        "architecture",
        "dockerRequired",
    }
    if set(payload) != expected or payload.get("schemaVersion") != 1:
        raise UpdaterSecurityError(
            "INVALID_DEVICE_BINDING", "device binding fields do not match schema v1"
        )

    def required_text(key: str) -> str:
        value = payload.get(key)
        if (
            not isinstance(value, str)
            or not value
            or value != value.strip()
            or len(value) > 255
        ):
            raise UpdaterSecurityError(
                "INVALID_DEVICE_BINDING", f"{key} must be trimmed text"
            )
        return value

    space_id = payload.get("spaceId")
    if isinstance(space_id, bool) or not isinstance(space_id, int) or space_id < 1:
        raise UpdaterSecurityError(
            "INVALID_DEVICE_BINDING", "spaceId must be a positive integer"
        )
    docker_required = payload.get("dockerRequired")
    if not isinstance(docker_required, bool):
        raise UpdaterSecurityError(
            "INVALID_DEVICE_BINDING", "dockerRequired must be boolean"
        )
    return DeviceBinding(
        trust_domain=required_text("trustDomain"),
        device_id=required_text("deviceId"),
        space_id=space_id,
        product_model=required_text("productModel"),
        platform_profile=required_text("platformProfile"),
        hardware_revision=required_text("hardwareRevision"),
        architecture=required_text("architecture"),
        docker_required=docker_required,
    )


def load_release_keyring(
    path: str | Path,
    *,
    expected_trust_domain: str,
    require_root_owner: bool = True,
) -> ReleaseKeyring:
    payload = _load_json(path, require_root_owner=require_root_owner)
    if set(payload) != {"schemaVersion", "trustDomain", "keys"}:
        raise UpdaterSecurityError(
            "INVALID_RELEASE_KEYRING", "release keyring fields do not match schema v1"
        )
    if payload.get("schemaVersion") != 1:
        raise UpdaterSecurityError(
            "INVALID_RELEASE_KEYRING", "unsupported release keyring schemaVersion"
        )
    if payload.get("trustDomain") != expected_trust_domain:
        raise UpdaterSecurityError(
            "INVALID_RELEASE_KEYRING", "release keyring trust domain mismatch"
        )
    raw_keys = payload.get("keys")
    if not isinstance(raw_keys, dict) or not raw_keys or len(raw_keys) > 32:
        raise UpdaterSecurityError(
            "INVALID_RELEASE_KEYRING", "release keyring must contain 1..32 keys"
        )
    decoded: dict[str, bytes] = {}
    for kid, encoded in raw_keys.items():
        if not isinstance(kid, str) or not kid or len(kid) > 128:
            raise UpdaterSecurityError(
                "INVALID_RELEASE_KEYRING", "release key id is invalid"
            )
        if not isinstance(encoded, str) or not encoded:
            raise UpdaterSecurityError(
                "INVALID_RELEASE_KEYRING", f"release key {kid} must be base64"
            )
        try:
            material = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise UpdaterSecurityError(
                "INVALID_RELEASE_KEYRING", f"release key {kid} is invalid base64"
            ) from exc
        if base64.b64encode(material).decode("ascii") != encoded:
            raise UpdaterSecurityError(
                "INVALID_RELEASE_KEYRING", f"release key {kid} is not canonical base64"
            )
        decoded[kid] = material
    try:
        return ReleaseKeyring(decoded)
    except (TypeError, ValueError) as exc:
        raise UpdaterSecurityError("INVALID_RELEASE_KEYRING", str(exc)) from exc


def load_health_attestation_keyring(
    path: str | Path,
    *,
    expected_trust_domain: str,
    require_root_owner: bool = True,
) -> Ed25519Keyring:
    payload = _load_json(path, require_root_owner=require_root_owner)
    if set(payload) != {"schemaVersion", "trustDomain", "purpose", "keys"}:
        raise UpdaterSecurityError(
            "INVALID_HEALTH_ATTESTATION_KEYRING",
            "health attestation keyring fields do not match schema v1",
        )
    if payload.get("schemaVersion") != 1:
        raise UpdaterSecurityError(
            "INVALID_HEALTH_ATTESTATION_KEYRING",
            "unsupported health attestation keyring schemaVersion",
        )
    if payload.get("trustDomain") != expected_trust_domain:
        raise UpdaterSecurityError(
            "INVALID_HEALTH_ATTESTATION_KEYRING",
            "health attestation keyring trust domain mismatch",
        )
    if payload.get("purpose") != "agent-update-health-attestation":
        raise UpdaterSecurityError(
            "INVALID_HEALTH_ATTESTATION_KEYRING",
            "health attestation keyring purpose mismatch",
        )
    raw_keys = payload.get("keys")
    if not isinstance(raw_keys, dict) or not raw_keys or len(raw_keys) > 32:
        raise UpdaterSecurityError(
            "INVALID_HEALTH_ATTESTATION_KEYRING",
            "health attestation keyring must contain 1..32 keys",
        )
    decoded: dict[str, bytes] = {}
    for kid, encoded in raw_keys.items():
        if not isinstance(kid, str) or not kid or len(kid) > 128:
            raise UpdaterSecurityError(
                "INVALID_HEALTH_ATTESTATION_KEYRING",
                "health attestation key id is invalid",
            )
        if not isinstance(encoded, str) or not encoded:
            raise UpdaterSecurityError(
                "INVALID_HEALTH_ATTESTATION_KEYRING",
                f"health attestation key {kid} must be base64",
            )
        try:
            material = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise UpdaterSecurityError(
                "INVALID_HEALTH_ATTESTATION_KEYRING",
                f"health attestation key {kid} is invalid base64",
            ) from exc
        if base64.b64encode(material).decode("ascii") != encoded:
            raise UpdaterSecurityError(
                "INVALID_HEALTH_ATTESTATION_KEYRING",
                f"health attestation key {kid} is not canonical base64",
            )
        decoded[kid] = material
    try:
        return Ed25519Keyring(decoded)
    except (TypeError, ValueError) as exc:
        raise UpdaterSecurityError(
            "INVALID_HEALTH_ATTESTATION_KEYRING", str(exc)
        ) from exc


def build_root_command_verifier(
    *,
    binding: DeviceBinding,
    command_keyring_path: str | Path,
    require_root_owner: bool = True,
) -> FleetCommandVerifier:
    payload = _load_json(
        command_keyring_path, require_root_owner=require_root_owner
    )
    if set(payload) != {"schemaVersion", "trustDomain", "keys"}:
        raise UpdaterSecurityError(
            "INVALID_COMMAND_KEYRING", "command keyring fields do not match schema v1"
        )
    if payload.get("schemaVersion") != 1:
        raise UpdaterSecurityError(
            "INVALID_COMMAND_KEYRING", "unsupported command keyring schemaVersion"
        )
    if payload.get("trustDomain") != binding.trust_domain:
        raise UpdaterSecurityError(
            "INVALID_COMMAND_KEYRING", "command keyring trust domain mismatch"
        )
    raw_keys = payload.get("keys")
    if not isinstance(raw_keys, dict) or not raw_keys or len(raw_keys) > 32:
        raise UpdaterSecurityError(
            "INVALID_COMMAND_KEYRING", "command keyring must contain 1..32 keys"
        )
    decoded: dict[str, bytes] = {}
    for kid, encoded in raw_keys.items():
        if not isinstance(kid, str) or not kid or len(kid) > 128:
            raise UpdaterSecurityError(
                "INVALID_COMMAND_KEYRING", "command key id is invalid"
            )
        if not isinstance(encoded, str) or not encoded:
            raise UpdaterSecurityError(
                "INVALID_COMMAND_KEYRING", f"command key {kid} must be base64"
            )
        try:
            material = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise UpdaterSecurityError(
                "INVALID_COMMAND_KEYRING", f"command key {kid} is invalid base64"
            ) from exc
        if base64.b64encode(material).decode("ascii") != encoded:
            raise UpdaterSecurityError(
                "INVALID_COMMAND_KEYRING", f"command key {kid} is not canonical base64"
            )
        decoded[kid] = material
    try:
        keyring = Ed25519Keyring(decoded)
    except (TypeError, ValueError) as exc:
        raise UpdaterSecurityError("INVALID_COMMAND_KEYRING", str(exc)) from exc
    return FleetCommandVerifier(
        keyring=keyring,
        expected_device_id=binding.device_id,
        expected_space_id=binding.space_id,
        capabilities={"command.agent.update"},
    )
