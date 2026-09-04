from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import re
import uuid
from collections.abc import Callable, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from nuvion_app.runtime.settings_overlay import (
    SettingsOverlayError,
    validate_model_pointer,
)

JWS_ALGORITHM = "EdDSA"
JWS_TYPE = "nuvion-command+jws"
MAX_COMMAND_PAYLOAD_BYTES = 64 * 1024
MAX_COMPACT_JWS_CHARS = 192 * 1024
DEFAULT_MAX_COMMAND_TTL = timedelta(hours=24)
DEFAULT_AUTHORIZATION_CONTEXTS = frozenset({"SPACE_ADMIN"})

COMMAND_CAPABILITY_BY_TYPE = {
    "CONFIG_APPLY": "command.config.apply",
    "STREAM_POLICY": "command.stream.policy",
    "AGENT_UPDATE": "command.agent.update",
}

AUTHENTICATED_REJECTION_CODES = frozenset(
    {
        "EXPIRED",
        "UNSUPPORTED_SCHEMA",
        "UNSUPPORTED_COMMAND",
        "MISSING_CAPABILITY",
        "INVALID_PAYLOAD_SCHEMA",
        "UNSUPPORTED_AUTHORIZATION_CONTEXT",
        "INVALID_TIME_WINDOW",
    }
)

_BASE64URL_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_SEMVER_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$")


class CommandValidationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class VerifiedFleetCommand:
    command_id: str
    device_id: str
    space_id: int
    command_type: str
    schema_version: int
    issued_at: str
    expires_at: str
    sequence: int
    payload_base64: str
    payload_hash: str
    payload: dict[str, Any]
    actor: str
    authorization_context: str
    key_id: str
    required_capability: str | None
    compact_jws: str


@dataclass(frozen=True)
class FleetCommandEvaluation:
    """An integrity-authenticated command and its execution-policy verdict."""

    command: VerifiedFleetCommand
    rejection_code: str | None = None
    rejection_message: str | None = None

    @property
    def executable(self) -> bool:
        return self.rejection_code is None


@dataclass(frozen=True)
class AuthenticatedCommandRejection:
    """Proof that a fully authenticated command has a terminal policy rejection."""

    command: VerifiedFleetCommand
    code: str
    message: str


@dataclass(frozen=True)
class _AuthenticatedFleetCommand:
    command: VerifiedFleetCommand
    issued_datetime: datetime
    expires_datetime: datetime


def _decode_base64url(segment: str, *, code: str = "INVALID_JWS") -> bytes:
    if (
        not isinstance(segment, str)
        or not segment
        or not _BASE64URL_PATTERN.fullmatch(segment)
    ):
        raise CommandValidationError(code, "value is not canonical unpadded base64url")
    padding = "=" * (-len(segment) % 4)
    try:
        decoded = base64.b64decode(segment + padding, altchars=b"-_", validate=True)
    except (binascii.Error, ValueError) as exc:
        raise CommandValidationError(code, "invalid base64url value") from exc
    canonical = base64.urlsafe_b64encode(decoded).decode("ascii").rstrip("=")
    if canonical != segment:
        raise CommandValidationError(code, "non-canonical base64url value")
    return decoded


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON member: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _decode_json_object(raw: bytes, *, code: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
        payload = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CommandValidationError(
            code, "value is not a valid UTF-8 JSON object"
        ) from exc
    if not isinstance(payload, dict):
        raise CommandValidationError(code, "value must be a JSON object")
    return payload


class Ed25519Keyring:
    """Pinned public verification keys indexed by JWS ``kid``.

    Values must be either a 32-byte Ed25519 raw public key, DER SubjectPublicKeyInfo,
    or a DER X.509 certificate containing an Ed25519 public key. Private keys are
    intentionally unsupported.
    """

    def __init__(self, keys: Mapping[str, bytes]) -> None:
        parsed: dict[str, Ed25519PublicKey] = {}
        for raw_kid, material in keys.items():
            kid = str(raw_kid or "").strip()
            if not kid:
                raise ValueError("keyring kid must be non-empty")
            if not isinstance(material, bytes) or not material:
                raise ValueError(f"public key material for kid={kid} must be bytes")
            parsed[kid] = self._parse_public_key(material, kid)
        if not parsed:
            raise ValueError("keyring must contain at least one public key")
        self._keys = parsed

    @staticmethod
    def _parse_public_key(material: bytes, kid: str) -> Ed25519PublicKey:
        if len(material) == 32:
            try:
                return Ed25519PublicKey.from_public_bytes(material)
            except ValueError as exc:
                raise ValueError(
                    f"invalid raw Ed25519 public key for kid={kid}"
                ) from exc

        public_key: object | None = None
        try:
            public_key = serialization.load_der_public_key(material)
        except ValueError:
            try:
                public_key = x509.load_der_x509_certificate(material).public_key()
            except ValueError as exc:
                raise ValueError(
                    f"invalid DER public key or certificate for kid={kid}"
                ) from exc
        if not isinstance(public_key, Ed25519PublicKey):
            raise TypeError(f"public key for kid={kid} is not Ed25519")
        return public_key

    def get(self, kid: str) -> Ed25519PublicKey | None:
        return self._keys.get(kid)

    def public_key_der(self, kid: str) -> bytes:
        public_key = self._keys[kid]
        return public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )


def _required_string(claims: Mapping[str, Any], key: str) -> str:
    value = claims.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CommandValidationError(
            "INVALID_CLAIMS", f"{key} must be a non-empty string"
        )
    return value


def _required_positive_int(claims: Mapping[str, Any], key: str) -> int:
    value = claims.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CommandValidationError(
            "INVALID_CLAIMS", f"{key} must be a positive integer"
        )
    return value


def _parse_rfc3339(value: str, key: str) -> datetime:
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise CommandValidationError(
            "INVALID_CLAIMS", f"{key} must be RFC3339 date-time"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CommandValidationError("INVALID_CLAIMS", f"{key} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _positive_payload_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CommandValidationError(
            "INVALID_PAYLOAD_SCHEMA",
            f"{key} must be a positive integer",
        )
    return value


def _bounded_payload_int(
    payload: Mapping[str, Any], key: str, *, minimum: int, maximum: int
) -> int:
    value = payload.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        raise CommandValidationError(
            "INVALID_PAYLOAD_SCHEMA",
            f"{key} must be an integer in [{minimum}, {maximum}]",
        )
    return value


def _bounded_payload_number(
    payload: Mapping[str, Any], key: str, *, minimum: float, maximum: float
) -> float:
    value = payload.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
        or float(value) > maximum
    ):
        raise CommandValidationError(
            "INVALID_PAYLOAD_SCHEMA",
            f"{key} must be a finite number in [{minimum}, {maximum}]",
        )
    return float(value)


def _require_exact_payload_keys(
    payload: Mapping[str, Any], *, required: set[str], optional: set[str] | None = None
) -> None:
    optional_keys = optional or set()
    actual = set(payload)
    missing = sorted(required - actual)
    unknown = sorted(actual - required - optional_keys)
    if missing or unknown:
        detail: list[str] = []
        if missing:
            detail.append(f"missing={','.join(missing)}")
        if unknown:
            detail.append(f"unknown={','.join(unknown)}")
        raise CommandValidationError(
            "INVALID_PAYLOAD_SCHEMA",
            "command payload keys are invalid (" + "; ".join(detail) + ")",
        )


def _payload_object(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise CommandValidationError(
            "INVALID_PAYLOAD_SCHEMA", f"{key} must be an object"
        )
    return value


def _validate_command_payload(command_type: str, payload: Mapping[str, Any]) -> None:
    """Validate stable v1 discriminator fields before an effect is journaled."""

    if command_type == "CONFIG_APPLY":
        _require_exact_payload_keys(
            payload,
            required={"configVersion", "activation"},
            optional={"model", "labels", "clip", "video"},
        )
        _bounded_payload_int(
            payload,
            "configVersion",
            minimum=1,
            maximum=2**63 - 1,
        )
        if payload.get("activation") not in {"IMMEDIATE", "RESTART"}:
            raise CommandValidationError(
                "INVALID_PAYLOAD_SCHEMA",
                "CONFIG_APPLY activation must be IMMEDIATE or RESTART",
            )
        if not any(key in payload for key in ("model", "labels", "clip", "video")):
            raise CommandValidationError(
                "INVALID_PAYLOAD_SCHEMA",
                "CONFIG_APPLY requires at least one model/labels/clip/video section",
            )
        if "model" in payload:
            model = _payload_object(payload, "model")
            _require_exact_payload_keys(
                model,
                required={"pointer", "digest"},
            )
            try:
                validate_model_pointer(model.get("pointer"))
            except SettingsOverlayError as exc:
                raise CommandValidationError(
                    "INVALID_PAYLOAD_SCHEMA",
                    f"model.pointer is invalid: {exc}",
                ) from exc
            digest = model.get("digest")
            if not isinstance(digest, str) or not _DIGEST_PATTERN.fullmatch(digest):
                raise CommandValidationError(
                    "INVALID_PAYLOAD_SCHEMA",
                    "model.digest must be sha256:<64 lowercase hex>",
                )
        if "labels" in payload:
            labels = _payload_object(payload, "labels")
            _require_exact_payload_keys(
                labels,
                required=set(),
                optional={"inspection", "anomaly"},
            )
            if not labels:
                raise CommandValidationError(
                    "INVALID_PAYLOAD_SCHEMA",
                    "labels requires inspection or anomaly",
                )
            for key in labels:
                values = labels.get(key)
                if not isinstance(values, list) or not 1 <= len(values) <= 100:
                    raise CommandValidationError(
                        "INVALID_PAYLOAD_SCHEMA",
                        f"labels.{key} must be an array of 1..100 labels",
                    )
                canonical: set[str] = set()
                for value in values:
                    if (
                        not isinstance(value, str)
                        or not value
                        or value != value.strip()
                        or len(value) > 100
                    ):
                        raise CommandValidationError(
                            "INVALID_PAYLOAD_SCHEMA",
                            f"labels.{key} entries must be trimmed non-empty strings "
                            "up to 100 characters",
                        )
                    duplicate_key = value.lower()
                    if duplicate_key in canonical:
                        raise CommandValidationError(
                            "INVALID_PAYLOAD_SCHEMA",
                            f"labels.{key} entries must be unique",
                        )
                    canonical.add(duplicate_key)
        if "clip" in payload:
            clip = _payload_object(payload, "clip")
            _require_exact_payload_keys(
                clip,
                required={"enabled", "preSeconds", "postSeconds"},
            )
            if not isinstance(clip.get("enabled"), bool):
                raise CommandValidationError(
                    "INVALID_PAYLOAD_SCHEMA", "clip.enabled must be boolean"
                )
            _bounded_payload_int(clip, "preSeconds", minimum=0, maximum=60)
            _bounded_payload_int(clip, "postSeconds", minimum=0, maximum=300)
        if "video" in payload:
            video = _payload_object(payload, "video")
            _require_exact_payload_keys(
                video,
                required={"width", "height", "fps", "bitrateKbps"},
            )
            _bounded_payload_int(video, "width", minimum=160, maximum=7680)
            _bounded_payload_int(video, "height", minimum=120, maximum=4320)
            _bounded_payload_int(video, "fps", minimum=1, maximum=120)
            _bounded_payload_int(
                video, "bitrateKbps", minimum=100, maximum=20_000
            )
        return

    if command_type == "STREAM_POLICY":
        _positive_payload_int(payload, "policyVersion")
        mode = payload.get("mode")
        if mode not in {"ADAPTIVE", "FIXED", "DISABLED"}:
            raise CommandValidationError(
                "INVALID_PAYLOAD_SCHEMA",
                "STREAM_POLICY mode must be ADAPTIVE, FIXED or DISABLED",
            )
        common = {"policyVersion", "mode"}
        if mode == "DISABLED":
            _require_exact_payload_keys(payload, required=common)
            return
        if mode == "FIXED":
            _require_exact_payload_keys(
                payload,
                required=common | {"targetBitrateKbps"},
            )
            _bounded_payload_int(
                payload,
                "targetBitrateKbps",
                minimum=100,
                maximum=20_000,
            )
            return

        required = common | {
            "minBitrateKbps",
            "maxBitrateKbps",
            "initialBitrateKbps",
        }
        optional = {
            "increaseStepKbps",
            "decreaseFactor",
            "congestionSamples",
            "recoverySamples",
            "cooldownSeconds",
        }
        _require_exact_payload_keys(payload, required=required, optional=optional)
        minimum = _bounded_payload_int(
            payload, "minBitrateKbps", minimum=100, maximum=20_000
        )
        initial = _bounded_payload_int(
            payload, "initialBitrateKbps", minimum=100, maximum=20_000
        )
        maximum = _bounded_payload_int(
            payload, "maxBitrateKbps", minimum=100, maximum=20_000
        )
        if not minimum <= initial <= maximum:
            raise CommandValidationError(
                "INVALID_PAYLOAD_SCHEMA",
                "STREAM_POLICY bitrate bounds must satisfy min <= initial <= max",
            )
        if "increaseStepKbps" in payload:
            _bounded_payload_int(
                payload, "increaseStepKbps", minimum=1, maximum=10_000
            )
        if "decreaseFactor" in payload:
            _bounded_payload_number(
                payload, "decreaseFactor", minimum=0.5, maximum=0.95
            )
        if "congestionSamples" in payload:
            _bounded_payload_int(
                payload, "congestionSamples", minimum=1, maximum=20
            )
        if "recoverySamples" in payload:
            _bounded_payload_int(
                payload, "recoverySamples", minimum=1, maximum=20
            )
        if "cooldownSeconds" in payload:
            _bounded_payload_int(
                payload, "cooldownSeconds", minimum=1, maximum=300
            )
        return

    if command_type == "AGENT_UPDATE":
        if set(payload) != {"targetVersion", "bomDigest"}:
            raise CommandValidationError(
                "INVALID_PAYLOAD_SCHEMA",
                "AGENT_UPDATE payload must contain exactly targetVersion and bomDigest",
            )
        target_version = payload.get("targetVersion")
        if not isinstance(target_version, str) or not _SEMVER_PATTERN.fullmatch(
            target_version
        ):
            raise CommandValidationError(
                "INVALID_PAYLOAD_SCHEMA",
                "AGENT_UPDATE targetVersion must be semantic version text",
            )
        bom_digest = payload.get("bomDigest")
        if not isinstance(bom_digest, str) or not _DIGEST_PATTERN.fullmatch(bom_digest):
            raise CommandValidationError(
                "INVALID_PAYLOAD_SCHEMA",
                "AGENT_UPDATE bomDigest must be sha256:<64 lowercase hex>",
            )
        return

    raise CommandValidationError("UNSUPPORTED_COMMAND", "command type is not supported")


class FleetCommandVerifier:
    def __init__(
        self,
        *,
        keyring: Ed25519Keyring,
        expected_device_id: str,
        expected_space_id: int,
        capabilities: AbstractSet[str],
        capability_provider: Callable[[], AbstractSet[str]] | None = None,
        unready_command_admission: Callable[[VerifiedFleetCommand], bool] | None = None,
        supported_schema_versions: AbstractSet[int] = frozenset({1}),
        allowed_authorization_contexts: AbstractSet[
            str
        ] = DEFAULT_AUTHORIZATION_CONTEXTS,
        max_command_ttl: timedelta = DEFAULT_MAX_COMMAND_TTL,
        clock: Callable[[], datetime] | None = None,
        allowed_clock_skew: timedelta = timedelta(seconds=30),
    ) -> None:
        if not isinstance(expected_device_id, str) or not expected_device_id.strip():
            raise ValueError("expected_device_id must be non-empty")
        if (
            isinstance(expected_space_id, bool)
            or not isinstance(expected_space_id, int)
            or expected_space_id < 1
        ):
            raise ValueError("expected_space_id must be a positive integer")
        if allowed_clock_skew < timedelta(0):
            raise ValueError("allowed_clock_skew must not be negative")
        if isinstance(allowed_authorization_contexts, (str, bytes)):
            raise TypeError("allowed_authorization_contexts must be a set of strings")
        normalized_contexts = frozenset(allowed_authorization_contexts)
        if not normalized_contexts or any(
            not isinstance(value, str) or not value or value != value.strip()
            for value in normalized_contexts
        ):
            raise ValueError(
                "allowed_authorization_contexts must be canonical non-empty strings"
            )
        if not isinstance(max_command_ttl, timedelta) or max_command_ttl <= timedelta(
            0
        ):
            raise ValueError("max_command_ttl must be positive")
        self.keyring = keyring
        self.expected_device_id = expected_device_id
        self.expected_space_id = expected_space_id
        self._capabilities = frozenset(capabilities)
        if capability_provider is not None and not callable(capability_provider):
            raise TypeError("capability_provider must be callable")
        self._capability_provider = capability_provider
        if unready_command_admission is not None and not callable(
            unready_command_admission
        ):
            raise TypeError("unready_command_admission must be callable")
        self._unready_command_admission = unready_command_admission
        self.supported_schema_versions = frozenset(supported_schema_versions)
        self.allowed_authorization_contexts = normalized_contexts
        self.max_command_ttl = max_command_ttl
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.allowed_clock_skew = allowed_clock_skew

    @property
    def capabilities(self) -> frozenset[str]:
        if self._capability_provider is None:
            return self._capabilities
        try:
            return frozenset(self._capability_provider())
        except Exception:  # noqa: BLE001 - command admission must fail closed.
            return frozenset()

    def verify(self, compact_jws: str) -> VerifiedFleetCommand:
        evaluation = self.evaluate(compact_jws)
        if not evaluation.executable:
            raise CommandValidationError(
                str(evaluation.rejection_code),
                str(evaluation.rejection_message),
            )
        return evaluation.command

    def evaluate(self, compact_jws: str) -> FleetCommandEvaluation:
        """Authenticate first, then return only terminal execution-policy verdicts.

        Authentication, device/space binding, required claim parsing, time-window
        structure and payload integrity are a hard boundary. Failures in that stage
        are raised and can never be converted into a durable rejection. Only the
        explicit policy codes in :data:`AUTHENTICATED_REJECTION_CODES` are returned
        to the runtime as commands that may safely advance the sequence cursor.
        """

        authenticated = self._authenticate_integrity(compact_jws)
        policy_error = self._execution_policy_error(authenticated)
        if policy_error is None:
            return FleetCommandEvaluation(command=authenticated.command)
        if policy_error.code not in AUTHENTICATED_REJECTION_CODES:
            raise policy_error
        return FleetCommandEvaluation(
            command=authenticated.command,
            rejection_code=policy_error.code,
            rejection_message=str(policy_error),
        )

    def verify_for_rejection(
        self,
        compact_jws: str,
    ) -> AuthenticatedCommandRejection:
        """Return proof only for an authenticated terminal policy rejection."""

        evaluation = self.evaluate(compact_jws)
        if evaluation.executable:
            raise CommandValidationError(
                "NOT_REJECTABLE",
                "authenticated command is executable",
            )
        return AuthenticatedCommandRejection(
            command=evaluation.command,
            code=str(evaluation.rejection_code),
            message=str(evaluation.rejection_message),
        )

    def verify_expired_for_rejection(self, compact_jws: str) -> VerifiedFleetCommand:
        """Return a fully authenticated expired command only for durable rejection."""

        try:
            rejection = self.verify_for_rejection(compact_jws)
        except CommandValidationError as exc:
            if exc.code != "NOT_REJECTABLE":
                raise
            raise CommandValidationError(
                "NOT_EXPIRED",
                "command is not expired",
            ) from exc
        if rejection.code != "EXPIRED":
            raise CommandValidationError("NOT_EXPIRED", "command is not expired")
        return rejection.command

    def _authenticate_integrity(
        self,
        compact_jws: str,
    ) -> _AuthenticatedFleetCommand:
        if not isinstance(compact_jws, str):
            raise CommandValidationError("INVALID_JWS", "compact JWS must be a string")
        if len(compact_jws) > MAX_COMPACT_JWS_CHARS:
            raise CommandValidationError(
                "COMMAND_TOO_LARGE", "compact JWS exceeds the v1 size limit"
            )
        segments = compact_jws.split(".")
        if len(segments) != 3 or any(not segment for segment in segments):
            raise CommandValidationError(
                "INVALID_JWS", "compact JWS must contain exactly three segments"
            )
        protected_segment, claims_segment, signature_segment = segments

        protected = _decode_json_object(
            _decode_base64url(protected_segment),
            code="INVALID_JWS",
        )
        if set(protected) != {"alg", "kid", "typ"}:
            raise CommandValidationError(
                "UNSAFE_PROTECTED_HEADER",
                "protected header must contain exactly alg, kid and typ",
            )
        if protected.get("alg") != JWS_ALGORITHM:
            raise CommandValidationError(
                "UNSUPPORTED_ALGORITHM", "only EdDSA is accepted"
            )
        if protected.get("typ") != JWS_TYPE:
            raise CommandValidationError(
                "INVALID_JWS_TYPE", "unexpected protected header typ"
            )
        kid = protected.get("kid")
        if not isinstance(kid, str) or not kid:
            raise CommandValidationError(
                "INVALID_JWS", "protected header kid is required"
            )
        public_key = self.keyring.get(kid)
        if public_key is None:
            raise CommandValidationError(
                "UNKNOWN_KEY_ID", "protected header kid is not trusted"
            )

        signature = _decode_base64url(signature_segment)
        if len(signature) != 64:
            raise CommandValidationError(
                "INVALID_SIGNATURE", "Ed25519 signature must be 64 bytes"
            )
        signing_input = f"{protected_segment}.{claims_segment}".encode("ascii")
        try:
            public_key.verify(signature, signing_input)
        except (InvalidSignature, ValueError) as exc:
            raise CommandValidationError(
                "INVALID_SIGNATURE", "Ed25519 signature verification failed"
            ) from exc

        claims = _decode_json_object(
            _decode_base64url(claims_segment),
            code="INVALID_CLAIMS",
        )
        command_id = _required_string(claims, "commandId")
        try:
            normalized_command_id = str(uuid.UUID(command_id))
        except (ValueError, AttributeError) as exc:
            raise CommandValidationError(
                "INVALID_CLAIMS", "commandId must be a UUID"
            ) from exc
        if normalized_command_id != command_id:
            raise CommandValidationError(
                "INVALID_CLAIMS", "commandId must use canonical UUID form"
            )

        device_id = _required_string(claims, "deviceId")
        space_id = _required_positive_int(claims, "spaceId")
        raw_command_type = _required_string(claims, "type")
        command_type = raw_command_type.upper()
        if raw_command_type != command_type:
            raise CommandValidationError(
                "INVALID_CLAIMS", "type must use canonical uppercase form"
            )
        schema_version = _required_positive_int(claims, "schemaVersion")
        issued_at = _required_string(claims, "issuedAt")
        expires_at = _required_string(claims, "expiresAt")
        sequence = _required_positive_int(claims, "sequence")
        payload_base64 = _required_string(claims, "payloadBase64")
        payload_hash = _required_string(claims, "payloadHash")
        actor = _required_string(claims, "actor")
        authorization_context = _required_string(claims, "authorizationContext")

        if device_id != self.expected_device_id:
            raise CommandValidationError(
                "DEVICE_MISMATCH", "command is bound to another device"
            )
        if space_id != self.expected_space_id:
            raise CommandValidationError(
                "SPACE_MISMATCH", "command is bound to another space"
            )

        issued_datetime = _parse_rfc3339(issued_at, "issuedAt")
        expires_datetime = _parse_rfc3339(expires_at, "expiresAt")
        if expires_datetime <= issued_datetime:
            raise CommandValidationError(
                "INVALID_TIME_WINDOW", "expiresAt must be after issuedAt"
            )

        if not _SHA256_PATTERN.fullmatch(payload_hash):
            raise CommandValidationError(
                "INVALID_CLAIMS", "payloadHash must be lowercase SHA-256 hex"
            )
        payload_bytes = _decode_base64url(payload_base64, code="INVALID_PAYLOAD")
        if len(payload_bytes) > MAX_COMMAND_PAYLOAD_BYTES:
            raise CommandValidationError(
                "COMMAND_TOO_LARGE",
                "decoded command payload exceeds 64 KiB",
            )
        actual_hash = hashlib.sha256(payload_bytes).hexdigest()
        if actual_hash != payload_hash:
            raise CommandValidationError(
                "INVALID_PAYLOAD_HASH", "payload SHA-256 does not match"
            )
        payload = _decode_json_object(payload_bytes, code="INVALID_PAYLOAD")

        required_capability = COMMAND_CAPABILITY_BY_TYPE.get(command_type)
        return _AuthenticatedFleetCommand(
            command=VerifiedFleetCommand(
                command_id=normalized_command_id,
                device_id=device_id,
                space_id=space_id,
                command_type=command_type,
                schema_version=schema_version,
                issued_at=issued_at,
                expires_at=expires_at,
                sequence=sequence,
                payload_base64=payload_base64,
                payload_hash=payload_hash,
                payload=payload,
                actor=actor,
                authorization_context=authorization_context,
                key_id=kid,
                required_capability=required_capability,
                compact_jws=compact_jws,
            ),
            issued_datetime=issued_datetime,
            expires_datetime=expires_datetime,
        )

    def _execution_policy_error(
        self,
        authenticated: _AuthenticatedFleetCommand,
    ) -> CommandValidationError | None:
        command = authenticated.command
        now = self.clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise RuntimeError(
                "FleetCommandVerifier clock must return timezone-aware datetime"
            )
        now = now.astimezone(timezone.utc)
        if authenticated.issued_datetime > now + self.allowed_clock_skew:
            return CommandValidationError(
                "NOT_YET_VALID",
                "command issuedAt is in the future",
            )
        if (
            authenticated.expires_datetime - authenticated.issued_datetime
            > self.max_command_ttl
        ):
            return CommandValidationError(
                "INVALID_TIME_WINDOW",
                "command validity window exceeds the configured maximum TTL",
            )
        if now >= authenticated.expires_datetime + self.allowed_clock_skew:
            return CommandValidationError("EXPIRED", "command has expired")
        if command.authorization_context not in self.allowed_authorization_contexts:
            return CommandValidationError(
                "UNSUPPORTED_AUTHORIZATION_CONTEXT",
                "authorizationContext is not permitted for Fleet command execution",
            )
        if command.schema_version not in self.supported_schema_versions:
            return CommandValidationError(
                "UNSUPPORTED_SCHEMA",
                "schemaVersion is not supported",
            )
        if command.required_capability is None:
            return CommandValidationError(
                "UNSUPPORTED_COMMAND",
                "command type is not supported",
            )
        if command.required_capability not in self.capabilities:
            admitted = False
            if self._unready_command_admission is not None:
                try:
                    admitted = bool(self._unready_command_admission(command))
                except Exception:  # noqa: BLE001 - admission must fail closed.
                    admitted = False
            if not admitted:
                return CommandValidationError(
                    "MISSING_CAPABILITY",
                    "platform does not provide command capability",
                )
        try:
            _validate_command_payload(command.command_type, command.payload)
        except CommandValidationError as exc:
            if exc.code != "INVALID_PAYLOAD_SCHEMA":
                raise
            return exc
        return None
