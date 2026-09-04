from __future__ import annotations

import re
import uuid
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

MAX_ATTESTATION_JWS_CHARS = 32 * 1024
MAX_ATTESTATION_TTL = timedelta(seconds=60)
ALLOWED_CLOCK_SKEW = timedelta(seconds=5)

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMPONENT_SHA = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_CHALLENGE = re.compile(r"[A-Za-z0-9_-]{43}\Z")
_SLOT = re.compile(r"releases/[0-9a-f]{64}\Z")
_KEY_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}\Z")
_IDENTITY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_DEVICE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}\Z")
_JWS_SEGMENT = re.compile(r"[A-Za-z0-9_-]+\Z")
_RFC3339_UTC = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,6})?Z\Z"
)
_GATE_FIELDS = frozenset(
    {
        "schemaVersion",
        "gateId",
        "challenge",
        "commandId",
        "commandExpiresAt",
        "bomDigest",
        "componentSha",
        "releaseSequence",
        "candidateSlot",
        "agentPid",
        "agentStartTicks",
        "bootId",
        "expiresAt",
    }
)
_RESPONSE_FIELDS = frozenset({"keyId", "issuedAt", "expiresAt", "compactJws"})
_RESPONSE_ENVELOPE_FIELDS = frozenset({"message", "data"})


class UpdateHealthAttestationError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _canonical_uuid(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise UpdateHealthAttestationError(
            "HEALTH_ATTESTATION_INVALID",
            f"{field} must be a canonical UUID",
        )
    try:
        normalized = str(uuid.UUID(value))
    except (AttributeError, ValueError) as exc:
        raise UpdateHealthAttestationError(
            "HEALTH_ATTESTATION_INVALID",
            f"{field} must be a canonical UUID",
        ) from exc
    if normalized != value:
        raise UpdateHealthAttestationError(
            "HEALTH_ATTESTATION_INVALID",
            f"{field} must be a canonical UUID",
        )
    return normalized


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise UpdateHealthAttestationError(
            "HEALTH_ATTESTATION_INVALID",
            f"{field} must be a positive integer",
        )
    return value


def _timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not _RFC3339_UTC.fullmatch(value):
        raise UpdateHealthAttestationError(
            "HEALTH_ATTESTATION_INVALID",
            f"{field} must be an RFC3339 UTC timestamp",
        )
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
            timezone.utc
        )
    except ValueError as exc:
        raise UpdateHealthAttestationError(
            "HEALTH_ATTESTATION_INVALID",
            f"{field} is not a valid timestamp",
        ) from exc


def _identity_value(identity: Mapping[str, Any], field: str) -> str:
    value = identity.get(field)
    if not isinstance(value, str) or not _IDENTITY.fullmatch(value):
        raise UpdateHealthAttestationError(
            "HEALTH_ATTESTATION_IDENTITY_INVALID",
            f"{field} is unavailable or invalid",
        )
    return value


def build_health_attestation_request(
    *,
    device_id: str,
    command_id: str,
    expected_bom_digest: str,
    expected_component_sha: str,
    expected_release_sequence: int,
    gate: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> dict[str, object]:
    """Bind the BE request to the root-authenticated gate and platform identity."""

    if set(gate) != _GATE_FIELDS or gate.get("schemaVersion") != 1:
        raise UpdateHealthAttestationError(
            "COMMIT_GATE_INVALID",
            "root commit gate schema is invalid",
        )
    normalized_command = _canonical_uuid(command_id, "commandId")
    if gate.get("commandId") != normalized_command:
        raise UpdateHealthAttestationError(
            "COMMIT_GATE_MISMATCH",
            "root commit gate commandId does not match desired command",
        )
    _timestamp(gate.get("commandExpiresAt"), "commandExpiresAt")
    gate_id = _canonical_uuid(gate.get("gateId"), "gateId")
    _canonical_uuid(gate.get("bootId"), "bootId")
    challenge = gate.get("challenge")
    if not isinstance(challenge, str) or not _CHALLENGE.fullmatch(challenge):
        raise UpdateHealthAttestationError(
            "COMMIT_GATE_INVALID",
            "root commit gate challenge is invalid",
        )
    if not _DIGEST.fullmatch(expected_bom_digest) or gate.get(
        "bomDigest"
    ) != expected_bom_digest:
        raise UpdateHealthAttestationError(
            "COMMIT_GATE_MISMATCH",
            "root commit gate BOM digest does not match verified release",
        )
    if not _COMPONENT_SHA.fullmatch(expected_component_sha) or gate.get(
        "componentSha"
    ) != expected_component_sha:
        raise UpdateHealthAttestationError(
            "COMMIT_GATE_MISMATCH",
            "root commit gate component SHA does not match verified release",
        )
    release_sequence = _positive_int(expected_release_sequence, "releaseSequence")
    if gate.get("releaseSequence") != release_sequence:
        raise UpdateHealthAttestationError(
            "COMMIT_GATE_MISMATCH",
            "root commit gate release sequence does not match verified release",
        )
    candidate_slot = gate.get("candidateSlot")
    if not isinstance(candidate_slot, str) or not _SLOT.fullmatch(candidate_slot):
        raise UpdateHealthAttestationError(
            "COMMIT_GATE_INVALID",
            "root commit gate candidate slot is invalid",
        )
    if candidate_slot != f"releases/{expected_bom_digest[7:]}":
        raise UpdateHealthAttestationError(
            "COMMIT_GATE_MISMATCH",
            "root commit gate candidate slot does not match the BOM digest",
        )
    _positive_int(gate.get("agentPid"), "agentPid")
    _positive_int(gate.get("agentStartTicks"), "agentStartTicks")
    _timestamp(gate.get("expiresAt"), "expiresAt")

    if not isinstance(device_id, str) or not _DEVICE_ID.fullmatch(device_id):
        raise UpdateHealthAttestationError(
            "HEALTH_ATTESTATION_IDENTITY_INVALID",
            "deviceId is invalid",
        )
    return {
        "deviceId": device_id,
        "commandId": normalized_command,
        "gateId": gate_id,
        "challenge": challenge,
        "bomDigest": expected_bom_digest,
        "componentSha": expected_component_sha,
        "releaseSequence": release_sequence,
        "productModel": _identity_value(identity, "productModel"),
        "platformProfile": _identity_value(identity, "platformProfile"),
        "hardwareRevision": _identity_value(identity, "hardwareRevision"),
        "architecture": _identity_value(identity, "architecture"),
    }


def parse_health_attestation_response(
    response: Mapping[str, Any] | object,
    *,
    now: datetime | None = None,
) -> dict[str, str]:
    """Strictly parse the public response without trusting its signed claims."""

    if not isinstance(response, Mapping):
        raise UpdateHealthAttestationError(
            "HEALTH_ATTESTATION_UNAVAILABLE",
            "BE health attestation response is unavailable",
        )
    if set(response) != _RESPONSE_ENVELOPE_FIELDS or not isinstance(
        response.get("message"), str
    ):
        raise UpdateHealthAttestationError(
            "HEALTH_ATTESTATION_INVALID",
            "BE health attestation response envelope is invalid",
        )
    raw = response.get("data")
    if not isinstance(raw, Mapping) or set(raw) != _RESPONSE_FIELDS:
        raise UpdateHealthAttestationError(
            "HEALTH_ATTESTATION_INVALID",
            "BE health attestation response schema is invalid",
        )
    key_id = raw.get("keyId")
    compact_jws = raw.get("compactJws")
    if not isinstance(key_id, str) or not _KEY_ID.fullmatch(key_id):
        raise UpdateHealthAttestationError(
            "HEALTH_ATTESTATION_INVALID",
            "BE health attestation keyId is invalid",
        )
    if (
        not isinstance(compact_jws, str)
        or not compact_jws
        or len(compact_jws) > MAX_ATTESTATION_JWS_CHARS
    ):
        raise UpdateHealthAttestationError(
            "HEALTH_ATTESTATION_INVALID",
            "BE compact health attestation JWS is invalid",
        )
    segments = compact_jws.split(".")
    if len(segments) != 3 or any(
        not _JWS_SEGMENT.fullmatch(segment) for segment in segments
    ):
        raise UpdateHealthAttestationError(
            "HEALTH_ATTESTATION_INVALID",
            "BE compact health attestation JWS is not canonical base64url",
        )
    issued_at = _timestamp(raw.get("issuedAt"), "issuedAt")
    expires_at = _timestamp(raw.get("expiresAt"), "expiresAt")
    if expires_at <= issued_at or expires_at - issued_at > MAX_ATTESTATION_TTL:
        raise UpdateHealthAttestationError(
            "HEALTH_ATTESTATION_INVALID",
            "BE health attestation TTL is invalid",
        )
    observed_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if issued_at > observed_at + ALLOWED_CLOCK_SKEW:
        raise UpdateHealthAttestationError(
            "HEALTH_ATTESTATION_NOT_YET_VALID",
            "BE health attestation was issued in the future",
        )
    if expires_at < observed_at - ALLOWED_CLOCK_SKEW:
        raise UpdateHealthAttestationError(
            "HEALTH_ATTESTATION_EXPIRED",
            "BE health attestation is expired",
        )
    return {
        "keyId": key_id,
        "issuedAt": str(raw["issuedAt"]),
        "expiresAt": str(raw["expiresAt"]),
        "compactJws": compact_jws,
    }


def request_health_attestation(
    request: Mapping[str, object],
    *,
    transport: Callable[[dict[str, object]], Mapping[str, Any] | None],
) -> dict[str, str]:
    try:
        response = transport(dict(request))
    except Exception as exc:  # noqa: BLE001 - transport failure is retryable until watchdog.
        raise UpdateHealthAttestationError(
            "HEALTH_ATTESTATION_UNAVAILABLE",
            "BE health attestation request failed",
        ) from exc
    return parse_health_attestation_response(response)
