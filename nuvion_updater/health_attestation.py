from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from cryptography.exceptions import InvalidSignature

from nuvion_app.inference.fleet_command import Ed25519Keyring
from nuvion_updater.errors import UpdaterSecurityError

HEALTH_ATTESTATION_SCHEMA_VERSION = 1
HEALTH_ATTESTATION_AUDIENCE = "nuvion-updater"
HEALTH_ATTESTATION_PURPOSE = "agent-update-commit"
HEALTH_ATTESTATION_JWS_TYPE = "nuvion-update-health+jws"
MAX_HEALTH_ATTESTATION_CHARS = 32 * 1024
MAX_HEALTH_ATTESTATION_TTL = timedelta(seconds=60)
DEFAULT_ALLOWED_CLOCK_SKEW = timedelta(seconds=5)

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMPONENT_SHA = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_CHALLENGE = re.compile(r"[A-Za-z0-9_-]{43}\Z")
_SLOT = re.compile(r"releases/[0-9a-f]{64}\Z")
_RFC3339_UTC = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,6})?Z\Z"
)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise UpdaterSecurityError(
                "INVALID_HEALTH_ATTESTATION", f"duplicate JSON member: {key}"
            )
        result[key] = value
    return result


def _decode_base64url(value: str) -> bytes:
    if not isinstance(value, str) or not value or "=" in value:
        raise UpdaterSecurityError(
            "INVALID_HEALTH_ATTESTATION", "JWS segment is not canonical base64url"
        )
    try:
        decoded = base64.b64decode(
            value + ("=" * (-len(value) % 4)),
            altchars=b"-_",
            validate=True,
        )
    except (binascii.Error, ValueError) as exc:
        raise UpdaterSecurityError(
            "INVALID_HEALTH_ATTESTATION", "JWS segment is invalid base64url"
        ) from exc
    if base64.urlsafe_b64encode(decoded).decode("ascii").rstrip("=") != value:
        raise UpdaterSecurityError(
            "INVALID_HEALTH_ATTESTATION", "JWS segment is not canonical base64url"
        )
    return decoded


def _decode_object(value: bytes) -> dict[str, Any]:
    try:
        decoded = json.loads(
            value.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda raw: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant: {raw}")
            ),
        )
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        if isinstance(exc, UpdaterSecurityError):
            raise
        raise UpdaterSecurityError(
            "INVALID_HEALTH_ATTESTATION", "JWS JSON is invalid"
        ) from exc
    if not isinstance(decoded, dict):
        raise UpdaterSecurityError(
            "INVALID_HEALTH_ATTESTATION", "JWS JSON must be an object"
        )
    return decoded


def _canonical_uuid(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise UpdaterSecurityError(
            "HEALTH_ATTESTATION_MISMATCH", f"{field} must be a canonical UUID"
        )
    try:
        normalized = str(uuid.UUID(value))
    except (AttributeError, ValueError) as exc:
        raise UpdaterSecurityError(
            "HEALTH_ATTESTATION_MISMATCH", f"{field} must be a canonical UUID"
        ) from exc
    if normalized != value:
        raise UpdaterSecurityError(
            "HEALTH_ATTESTATION_MISMATCH", f"{field} must be a canonical UUID"
        )
    return normalized


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise UpdaterSecurityError(
            "HEALTH_ATTESTATION_MISMATCH", f"{field} must be a positive integer"
        )
    return value


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not _RFC3339_UTC.fullmatch(value):
        raise UpdaterSecurityError(
            "INVALID_HEALTH_ATTESTATION", f"{field} must be an RFC3339 UTC timestamp"
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise UpdaterSecurityError(
            "INVALID_HEALTH_ATTESTATION", f"{field} is not a valid timestamp"
        ) from exc
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class CommitProcessIdentity:
    pid: int
    start_ticks: int
    boot_id: str
    active_slot: str


@dataclass(frozen=True)
class ExpectedHealthAttestation:
    gate_id: str
    challenge: str
    trust_domain: str
    device_id: str
    command_id: str
    bom_digest: str
    component_sha: str
    release_sequence: int
    product_model: str
    platform_profile: str
    hardware_revision: str
    architecture: str


@dataclass(frozen=True)
class VerifiedHealthAttestation:
    attestation_id: str
    compact_jws_sha256: str
    issued_at: datetime
    expires_at: datetime
    key_id: str


class HealthAttestationVerifier:
    """Verify the BE-issued, challenge-bound post-update health proof."""

    def __init__(
        self,
        *,
        keyring: Ed25519Keyring,
        clock: Any | None = None,
        max_ttl: timedelta = MAX_HEALTH_ATTESTATION_TTL,
        allowed_clock_skew: timedelta = DEFAULT_ALLOWED_CLOCK_SKEW,
    ) -> None:
        if max_ttl <= timedelta(0) or max_ttl > MAX_HEALTH_ATTESTATION_TTL:
            raise ValueError("health attestation max_ttl must be in (0, 60s]")
        if allowed_clock_skew < timedelta(0) or allowed_clock_skew > timedelta(
            seconds=30
        ):
            raise ValueError("health attestation clock skew must be in [0, 30s]")
        self.keyring = keyring
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.max_ttl = max_ttl
        self.allowed_clock_skew = allowed_clock_skew

    def verify(
        self,
        compact_jws: str,
        *,
        expected: ExpectedHealthAttestation,
    ) -> VerifiedHealthAttestation:
        if (
            not isinstance(compact_jws, str)
            or not compact_jws
            or len(compact_jws) > MAX_HEALTH_ATTESTATION_CHARS
        ):
            raise UpdaterSecurityError(
                "INVALID_HEALTH_ATTESTATION", "health attestation JWS is invalid"
            )
        segments = compact_jws.split(".")
        if len(segments) != 3 or any(not segment for segment in segments):
            raise UpdaterSecurityError(
                "INVALID_HEALTH_ATTESTATION", "compact JWS must contain three segments"
            )
        protected_segment, claims_segment, signature_segment = segments
        protected = _decode_object(_decode_base64url(protected_segment))
        if set(protected) != {"alg", "kid", "typ"}:
            raise UpdaterSecurityError(
                "INVALID_HEALTH_ATTESTATION", "protected header fields are invalid"
            )
        if protected.get("alg") != "EdDSA":
            raise UpdaterSecurityError(
                "INVALID_HEALTH_ATTESTATION", "only EdDSA is accepted"
            )
        if protected.get("typ") != HEALTH_ATTESTATION_JWS_TYPE:
            raise UpdaterSecurityError(
                "INVALID_HEALTH_ATTESTATION_DOMAIN", "JWS type is not health-attestation"
            )
        key_id = protected.get("kid")
        if not isinstance(key_id, str) or not key_id or len(key_id) > 128:
            raise UpdaterSecurityError(
                "INVALID_HEALTH_ATTESTATION", "protected header kid is invalid"
            )
        public_key = self.keyring.get(key_id)
        if public_key is None:
            raise UpdaterSecurityError(
                "UNKNOWN_HEALTH_ATTESTATION_KEY", "health attestation key is not trusted"
            )
        signature = _decode_base64url(signature_segment)
        if len(signature) != 64:
            raise UpdaterSecurityError(
                "INVALID_HEALTH_ATTESTATION", "Ed25519 signature must be 64 bytes"
            )
        try:
            public_key.verify(
                signature,
                f"{protected_segment}.{claims_segment}".encode("ascii"),
            )
        except (InvalidSignature, ValueError) as exc:
            raise UpdaterSecurityError(
                "INVALID_HEALTH_ATTESTATION_SIGNATURE",
                "health attestation signature verification failed",
            ) from exc

        claims = _decode_object(_decode_base64url(claims_segment))
        required = {
            "schemaVersion",
            "jti",
            "aud",
            "purpose",
            "trustDomain",
            "gateId",
            "challenge",
            "deviceId",
            "commandId",
            "bomDigest",
            "componentSha",
            "releaseSequence",
            "productModel",
            "platformProfile",
            "hardwareRevision",
            "architecture",
            "health",
            "issuedAt",
            "expiresAt",
        }
        if set(claims) != required:
            raise UpdaterSecurityError(
                "INVALID_HEALTH_ATTESTATION", "health attestation claims are incomplete"
            )
        if claims.get("schemaVersion") != HEALTH_ATTESTATION_SCHEMA_VERSION:
            raise UpdaterSecurityError(
                "INVALID_HEALTH_ATTESTATION", "unsupported health attestation schema"
            )
        if claims.get("aud") != HEALTH_ATTESTATION_AUDIENCE:
            raise UpdaterSecurityError(
                "INVALID_HEALTH_ATTESTATION_DOMAIN", "attestation audience is invalid"
            )
        if claims.get("purpose") != HEALTH_ATTESTATION_PURPOSE:
            raise UpdaterSecurityError(
                "INVALID_HEALTH_ATTESTATION_DOMAIN", "attestation purpose is invalid"
            )
        if claims.get("health") != "HEALTHY":
            raise UpdaterSecurityError(
                "HEALTH_ATTESTATION_UNHEALTHY", "issuer did not attest healthy runtime"
            )

        attestation_id = _canonical_uuid(claims.get("jti"), "jti")
        _canonical_uuid(claims.get("gateId"), "gateId")
        _canonical_uuid(claims.get("commandId"), "commandId")
        if claims.get("challenge") != expected.challenge or not _CHALLENGE.fullmatch(
            str(claims.get("challenge") or "")
        ):
            raise UpdaterSecurityError(
                "HEALTH_ATTESTATION_CHALLENGE_MISMATCH", "challenge does not match gate"
            )
        exact = {
            "gateId": expected.gate_id,
            "trustDomain": expected.trust_domain,
            "deviceId": expected.device_id,
            "commandId": expected.command_id,
            "bomDigest": expected.bom_digest,
            "componentSha": expected.component_sha,
            "releaseSequence": expected.release_sequence,
            "productModel": expected.product_model,
            "platformProfile": expected.platform_profile,
            "hardwareRevision": expected.hardware_revision,
            "architecture": expected.architecture,
        }
        for field, value in exact.items():
            if claims.get(field) != value:
                raise UpdaterSecurityError(
                    "HEALTH_ATTESTATION_MISMATCH", f"{field} does not match commit gate"
                )
        if not _DIGEST.fullmatch(str(claims.get("bomDigest") or "")):
            raise UpdaterSecurityError(
                "INVALID_HEALTH_ATTESTATION", "bomDigest is invalid"
            )
        if not _COMPONENT_SHA.fullmatch(str(claims.get("componentSha") or "")):
            raise UpdaterSecurityError(
                "INVALID_HEALTH_ATTESTATION", "componentSha is invalid"
            )
        _positive_int(claims.get("releaseSequence"), "releaseSequence")

        issued_at = _timestamp(claims.get("issuedAt"), "issuedAt")
        expires_at = _timestamp(claims.get("expiresAt"), "expiresAt")
        if expires_at <= issued_at or expires_at - issued_at > self.max_ttl:
            raise UpdaterSecurityError(
                "INVALID_HEALTH_ATTESTATION_TTL", "attestation TTL must be in (0, 60s]"
            )
        now = self.clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise RuntimeError("health attestation clock must be timezone-aware")
        now = now.astimezone(timezone.utc)
        if issued_at > now + self.allowed_clock_skew:
            raise UpdaterSecurityError(
                "HEALTH_ATTESTATION_NOT_YET_VALID", "attestation was issued in the future"
            )
        if now >= expires_at + self.allowed_clock_skew:
            raise UpdaterSecurityError(
                "HEALTH_ATTESTATION_EXPIRED", "health attestation has expired"
            )
        return VerifiedHealthAttestation(
            attestation_id=attestation_id,
            compact_jws_sha256=hashlib.sha256(compact_jws.encode("ascii")).hexdigest(),
            issued_at=issued_at,
            expires_at=expires_at,
            key_id=key_id,
        )
