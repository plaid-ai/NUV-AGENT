#!/usr/bin/env python3
"""Fail-closed IQ9075 release-catalog and staged-rollout control client."""

from __future__ import annotations

import argparse
import hashlib
import http.cookiejar
import json
import os
import re
import stat
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_API_ORIGIN = "https://api.nuvion-dev.plaidlabs.ai"
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_COOKIE_BYTES = 64 * 1024
DEVICE_ID_RE = re.compile(r"^sp-([1-9][0-9]*)-nuvion-[a-z0-9][a-z0-9-]{0,100}$")
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
COMPONENT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SEMVER_RE = re.compile(
    r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)
SLOT_RE = re.compile(
    r"^(?:releases/[0-9a-f]{64}|bootstrap/[0-9A-Za-z][0-9A-Za-z._+-]{0,99})$"
)
LEGACY_SLOTLESS_AGENT_VERSION = "0.1.120"
RELEASE_RESPONSE_FIELDS = {
    "releaseId",
    "spaceId",
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
    "publisherKeyId",
    "createdBy",
    "createdAt",
    "bom",
    "signature",
}
ROLLOUT_RESPONSE_FIELDS = {
    "rolloutId",
    "clientRequestId",
    "spaceId",
    "releaseId",
    "bomDigest",
    "agentVersion",
    "componentSha",
    "configSchema",
    "releaseSequence",
    "minUpdaterVersion",
    "artifact",
    "status",
    "policy",
    "targetCount",
    "healthReason",
    "haltReason",
    "nextEvaluationAt",
    "waves",
    "targets",
    "createdBy",
    "createdAt",
    "updatedAt",
}
TARGET_RESPONSE_FIELDS = {
    "deviceId",
    "cohortKey",
    "waveNumber",
    "eligibility",
    "eligibilityReason",
    "productModel",
    "platformProfile",
    "hardwareRevision",
    "architecture",
    "identitySnapshot",
    "status",
    "statusReason",
    "latestCommand",
    "desiredEvidence",
    "reportedEvidence",
    "rollbackEvidence",
    "commandIssuedAt",
    "succeededAt",
    "terminalAt",
}
COMMAND_RESPONSE_FIELDS = {
    "commandId",
    "deviceId",
    "spaceId",
    "type",
    "schemaVersion",
    "issuedAt",
    "expiresAt",
    "sequence",
    "payloadHash",
    "actor",
    "authorizationContext",
    "keyId",
    "status",
    "expiredAt",
}
ACTIVE_COMMAND_STATUSES = {"QUEUED", "RECEIVED", "IN_PROGRESS"}
TERMINAL_COMMAND_STATUS_BY_PURPOSE = {
    "rollback": "ROLLED_BACK",
    "commit": "SUCCEEDED",
}


class RolloutControlError(RuntimeError):
    """Stable operational failure that never includes credentials or bodies."""


def _reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RolloutControlError(f"duplicate JSON member: {key}")
        result[key] = value
    return result


def _strict_json(raw: bytes, *, label: str) -> Any:
    if len(raw) > MAX_JSON_BYTES:
        raise RolloutControlError(f"{label} exceeds the byte limit")
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate,
            parse_constant=lambda value: (_ for _ in ()).throw(
                RolloutControlError(f"invalid {label} constant: {value}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise RolloutControlError(f"{label} is not strict JSON") from exc


def _canonical(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                dict(value),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise RolloutControlError("evidence cannot be encoded canonically") from exc


def _file_bytes(path: Path, *, label: str, require_private: bool = False) -> bytes:
    if (
        not path.is_absolute()
        or path != path.resolve(strict=False)
        or path.is_symlink()
    ):
        raise RolloutControlError(f"{label} path is unsafe")
    try:
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_mode & 0o022
            or (require_private and stat.S_IMODE(before.st_mode) != 0o600)
            or before.st_size < 1
            or before.st_size > MAX_JSON_BYTES
        ):
            raise RolloutControlError(f"{label} metadata is unsafe")
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            opened = os.fstat(descriptor)
            raw = bytearray()
            while len(raw) <= MAX_JSON_BYTES:
                chunk = os.read(descriptor, min(65536, MAX_JSON_BYTES + 1 - len(raw)))
                if not chunk:
                    break
                raw.extend(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise RolloutControlError(f"{label} is unavailable") from exc

    def identity(item: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )

    if identity(before) != identity(opened) or identity(opened) != identity(after):
        raise RolloutControlError(f"{label} changed while being read")
    if not raw or len(raw) > MAX_JSON_BYTES:
        raise RolloutControlError(f"{label} byte length is invalid")
    return bytes(raw)


def _object_file(
    path: Path, *, label: str, canonical: bool = False
) -> tuple[dict[str, Any], bytes]:
    raw = _file_bytes(path, label=label)
    value = _strict_json(raw, label=label)
    if not isinstance(value, dict):
        raise RolloutControlError(f"{label} root must be an object")
    if canonical and raw != _canonical(value):
        raise RolloutControlError(f"{label} is not canonical JSON")
    return value, raw


def _private_output_parent(path: Path) -> Path:
    if not path.is_absolute() or path != path.resolve(strict=False):
        raise RolloutControlError("output path must be normalized and absolute")
    parent = path.parent
    try:
        metadata = parent.lstat()
        resolved = parent.resolve(strict=True)
        resolved_metadata = resolved.stat(follow_symlinks=False)
    except OSError as exc:
        raise RolloutControlError("output directory is unavailable") from exc

    def identity(item: os.stat_result) -> tuple[int, int, int]:
        return (item.st_dev, item.st_ino, item.st_mode)

    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_mode & 0o077
        or identity(metadata) != identity(resolved_metadata)
    ):
        raise RolloutControlError("output directory must be private and stable")
    return resolved


def _write_new(path: Path, value: Mapping[str, Any]) -> bytes:
    parent = _private_output_parent(path)
    target = parent / path.name
    if target.exists() or target.is_symlink():
        raise RolloutControlError("output evidence already exists")
    raw = _canonical(value)
    descriptor = os.open(
        target,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
    finally:
        os.close(descriptor)
    if _file_bytes(target, label="written evidence", require_private=True) != raw:
        raise RolloutControlError("written evidence read-back differs")
    return raw


def _preflight_output(path: Path, *, allow_existing: bool = False) -> Path:
    """Validate an evidence destination before any remote side effect."""

    parent = _private_output_parent(path)
    target = parent / path.name
    if target.is_symlink():
        raise RolloutControlError("output evidence path is unsafe")
    if target.exists():
        if not allow_existing:
            raise RolloutControlError("output evidence already exists")
        _file_bytes(target, label="existing output evidence", require_private=True)
    return target


def _uuid(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise RolloutControlError(f"{label} is invalid")
    try:
        canonical = str(uuid.UUID(value))
    except ValueError as exc:
        raise RolloutControlError(f"{label} is invalid") from exc
    if value != canonical:
        raise RolloutControlError(f"{label} is not canonical")
    return canonical


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RolloutControlError(f"{label} must be a positive integer")
    return value


def _instant(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RolloutControlError(f"{label} is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RolloutControlError(f"{label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RolloutControlError(f"{label} is not timezone-qualified")
    return parsed.astimezone(timezone.utc)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RolloutControlError("Fleet API redirects are not allowed")


class FleetApi:
    def __init__(
        self,
        origin: str,
        cookie_jar: Path,
        *,
        opener: Any | None = None,
    ) -> None:
        parsed = urllib.parse.urlsplit(origin)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "api.nuvion-dev.plaidlabs.ai"
            or parsed.port is not None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or origin.rstrip("/") != DEFAULT_API_ORIGIN
        ):
            raise RolloutControlError(
                "Fleet API origin must be the authoritative Nuvion dev API"
            )
        raw = _file_bytes(
            cookie_jar,
            label="Fleet API cookie jar",
            require_private=True,
        )
        if len(raw) > MAX_COOKIE_BYTES:
            raise RolloutControlError("Fleet API cookie jar exceeds the byte limit")
        jar = http.cookiejar.MozillaCookieJar(str(cookie_jar))
        try:
            jar.load(ignore_discard=True, ignore_expires=False)
        except (OSError, http.cookiejar.LoadError) as exc:
            raise RolloutControlError("Fleet API cookie jar is invalid") from exc
        cookies = list(jar)
        now = int(time.time())
        if not cookies or len(cookies) > 16:
            raise RolloutControlError("Fleet API cookie jar cardinality is invalid")
        if any(
            cookie.domain != parsed.hostname
            or cookie.path != "/"
            or not cookie.secure
            or (cookie.expires is not None and cookie.expires <= now)
            for cookie in cookies
        ):
            raise RolloutControlError(
                "Fleet API cookie jar contains an unbound or expired cookie"
            )
        self.origin = origin.rstrip("/")
        self._opener = opener or urllib.request.build_opener(
            _NoRedirect(), urllib.request.HTTPCookieProcessor(jar)
        )

    def request(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: Mapping[str, Any] | None = None,
    ) -> Any:
        if (
            method not in {"GET", "POST"}
            or not path.startswith("/")
            or "//" in path
            or ".." in path
            or "?" in path
            or "#" in path
            or (method == "GET" and payload is not None)
        ):
            raise RolloutControlError("Fleet API request shape is invalid")
        body = _canonical(payload) if payload is not None else None
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "User-Agent": "nuvion-iq9075-rollout-control/1",
        }
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.origin + path,
            data=body,
            method=method,
            headers=headers,
        )
        try:
            with self._opener.open(request, timeout=15) as response:
                if response.geturl() != self.origin + path or response.status != 200:
                    raise RolloutControlError(
                        "Fleet API response origin/status is invalid"
                    )
                encoding = response.headers.get("Content-Encoding")
                if encoding not in {None, "identity"}:
                    raise RolloutControlError("Fleet API response encoding is invalid")
                length = response.headers.get("Content-Length")
                if length is not None and int(length) > MAX_JSON_BYTES:
                    raise RolloutControlError(
                        "Fleet API response exceeds the byte limit"
                    )
                raw = response.read(MAX_JSON_BYTES + 1)
        except RolloutControlError:
            raise
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            OSError,
            ValueError,
        ) as exc:
            raise RolloutControlError("Fleet API request failed") from exc
        envelope = _strict_json(raw, label="Fleet API response")
        if (
            not isinstance(envelope, dict)
            or set(envelope) != {"message", "data"}
            or not isinstance(envelope.get("message"), str)
            or not envelope["message"].strip()
        ):
            raise RolloutControlError("Fleet API response envelope is invalid")
        return envelope["data"]


def _release_summary(
    data: Any,
    *,
    space_id: int,
    bom: Mapping[str, Any],
    signature: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(data, dict) or set(data) != RELEASE_RESPONSE_FIELDS:
        raise RolloutControlError("release registration response fields are invalid")
    if data.get("bom") != bom or data.get("signature") != signature:
        raise RolloutControlError("release registration bytes are not preserved")
    expected = {
        "schemaVersion": bom.get("schemaVersion"),
        "bomId": bom.get("bomId"),
        "bomDigest": bom.get("bomDigest"),
        "releaseSequence": bom.get("releaseSequence"),
        "agentVersion": bom.get("agentVersion"),
        "componentSha": bom.get("componentSha"),
        "configSchema": bom.get("configSchema"),
        "minUpdaterVersion": bom.get("minUpdaterVersion"),
        "targets": bom.get("targets"),
        "artifact": bom.get("artifact"),
        "builtAt": bom.get("builtAt"),
        "publisherKeyId": signature.get("keyId"),
    }
    if data.get("spaceId") != space_id or any(
        data.get(key) != value for key, value in expected.items()
    ):
        raise RolloutControlError("release registration identity differs from the BOM")
    release_id = _uuid(data.get("releaseId"), label="releaseId")
    if (
        bom.get("schemaVersion") != 2
        or not isinstance(bom.get("bomId"), str)
        or DIGEST_RE.fullmatch(str(bom.get("bomDigest") or "")) is None
        or SEMVER_RE.fullmatch(str(bom.get("agentVersion") or "")) is None
        or COMPONENT_RE.fullmatch(str(bom.get("componentSha") or "")) is None
        or not isinstance(bom.get("configSchema"), str)
        or _positive_int(bom.get("releaseSequence"), label="releaseSequence") < 1
        or not isinstance(bom.get("targets"), list)
        or not bom["targets"]
        or not isinstance(bom.get("artifact"), dict)
        or set(bom["artifact"]) != {"name", "kind", "sha256", "sizeBytes"}
        or SHA256_RE.fullmatch(str(bom["artifact"].get("sha256") or "")) is None
        or _positive_int(bom["artifact"].get("sizeBytes"), label="artifact size") < 1
    ):
        raise RolloutControlError("release BOM identity is invalid")
    if not isinstance(data.get("createdBy"), str) or not data["createdBy"].strip():
        raise RolloutControlError("release creator is unavailable")
    if not isinstance(data.get("createdAt"), str) or not data["createdAt"]:
        raise RolloutControlError("release creation time is unavailable")
    return {
        "releaseId": release_id,
        "bomDigest": bom["bomDigest"],
        "releaseSequence": bom["releaseSequence"],
        "agentVersion": bom["agentVersion"],
        "componentSha": bom["componentSha"],
        "configSchema": bom["configSchema"],
        "minUpdaterVersion": bom["minUpdaterVersion"],
        "targets": bom["targets"],
        "artifact": bom["artifact"],
        "publisherKeyId": signature["keyId"],
        "createdAt": data["createdAt"],
    }


def register_release(
    api: FleetApi,
    *,
    space_id: int,
    bom_path: Path,
    signature_path: Path,
    output: Path,
) -> dict[str, Any]:
    bom, bom_raw = _object_file(bom_path, label="candidate BOM", canonical=True)
    signature, signature_raw = _object_file(
        signature_path,
        label="candidate BOM signature",
        canonical=True,
    )
    _preflight_output(output, allow_existing=True)
    if output.exists():
        persisted, _persisted_raw = _load_release(
            output, space_id=space_id, origin=api.origin
        )
        if (
            persisted.get("bomFileSha256") != hashlib.sha256(bom_raw).hexdigest()
            or persisted.get("signatureFileSha256")
            != hashlib.sha256(signature_raw).hexdigest()
        ):
            raise RolloutControlError("persisted release evidence input hashes differ")
        live = api.request(
            f"/spaces/{space_id}/agent-releases/{persisted['release']['releaseId']}"
        )
        live_release = _release_summary(
            live, space_id=space_id, bom=bom, signature=signature
        )
        if live_release != persisted.get("release"):
            raise RolloutControlError("live release differs from persisted evidence")
        return persisted
    data = api.request(
        f"/spaces/{space_id}/agent-releases",
        method="POST",
        payload={"bom": bom, "signature": signature},
    )
    release = _release_summary(data, space_id=space_id, bom=bom, signature=signature)
    evidence = {
        "schemaVersion": 1,
        "kind": "nuvion-agent-release-registration",
        "apiOrigin": api.origin,
        "spaceId": space_id,
        "bomFileSha256": hashlib.sha256(bom_raw).hexdigest(),
        "signatureFileSha256": hashlib.sha256(signature_raw).hexdigest(),
        "release": release,
    }
    _write_new(output, evidence)
    return evidence


def _load_release(
    path: Path, *, space_id: int, origin: str
) -> tuple[dict[str, Any], bytes]:
    value, raw = _object_file(
        path, label="release registration evidence", canonical=True
    )
    if set(value) != {
        "schemaVersion",
        "kind",
        "apiOrigin",
        "spaceId",
        "bomFileSha256",
        "signatureFileSha256",
        "release",
    } or any(
        (
            value.get("schemaVersion") != 1,
            value.get("kind") != "nuvion-agent-release-registration",
            value.get("apiOrigin") != origin,
            value.get("spaceId") != space_id,
            not isinstance(value.get("release"), dict),
        )
    ):
        raise RolloutControlError("release registration evidence is invalid")
    release = value["release"]
    expected = {
        "releaseId",
        "bomDigest",
        "releaseSequence",
        "agentVersion",
        "componentSha",
        "configSchema",
        "minUpdaterVersion",
        "targets",
        "artifact",
        "publisherKeyId",
        "createdAt",
    }
    if set(release) != expected:
        raise RolloutControlError("release evidence fields are invalid")
    _uuid(release.get("releaseId"), label="release evidence releaseId")
    if (
        DIGEST_RE.fullmatch(str(release.get("bomDigest") or "")) is None
        or SEMVER_RE.fullmatch(str(release.get("agentVersion") or "")) is None
        or COMPONENT_RE.fullmatch(str(release.get("componentSha") or "")) is None
        or _positive_int(release.get("releaseSequence"), label="releaseSequence") < 1
        or not isinstance(release.get("targets"), list)
        or not release["targets"]
        or not isinstance(release.get("artifact"), dict)
        or SHA256_RE.fullmatch(str(value.get("bomFileSha256") or "")) is None
        or SHA256_RE.fullmatch(str(value.get("signatureFileSha256") or "")) is None
    ):
        raise RolloutControlError("release evidence identity is invalid")
    return value, raw


def _policy(
    pre_commit_soak_seconds: int,
    command_ttl_seconds: int,
    max_failure_percent: int,
) -> dict[str, int]:
    if (
        isinstance(pre_commit_soak_seconds, bool)
        or not isinstance(pre_commit_soak_seconds, int)
        or isinstance(command_ttl_seconds, bool)
        or not isinstance(command_ttl_seconds, int)
        or isinstance(max_failure_percent, bool)
        or not isinstance(max_failure_percent, int)
        or not 30 <= pre_commit_soak_seconds <= 60
        or not pre_commit_soak_seconds + 60 <= command_ttl_seconds <= 86400
        or not 0 <= max_failure_percent <= 100
    ):
        raise RolloutControlError("rollout policy is outside the BE contract")
    return {
        "preCommitSoakSeconds": pre_commit_soak_seconds,
        "commandTtlSeconds": command_ttl_seconds,
        "maxFailurePercent": max_failure_percent,
    }


def _validate_rollout(
    data: Any,
    *,
    space_id: int,
    device_id: str,
    release: Mapping[str, Any],
    policy: Mapping[str, Any],
    rollout_id: str | None = None,
    client_request_id: str | None = None,
    require_eligible: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(data, dict) or set(data) != ROLLOUT_RESPONSE_FIELDS:
        raise RolloutControlError("rollout response fields are invalid")
    actual_rollout_id = _uuid(data.get("rolloutId"), label="rolloutId")
    actual_client_request_id = _uuid(
        data.get("clientRequestId"), label="clientRequestId"
    )
    if rollout_id is not None and actual_rollout_id != rollout_id:
        raise RolloutControlError("rolloutId changed")
    if client_request_id is not None and actual_client_request_id != client_request_id:
        raise RolloutControlError("clientRequestId changed")
    if (
        data.get("spaceId") != space_id
        or data.get("releaseId") != release.get("releaseId")
        or data.get("bomDigest") != release.get("bomDigest")
        or data.get("agentVersion") != release.get("agentVersion")
        or data.get("componentSha") != release.get("componentSha")
        or data.get("configSchema") != release.get("configSchema")
        or data.get("releaseSequence") != release.get("releaseSequence")
        or data.get("minUpdaterVersion") != release.get("minUpdaterVersion")
        or data.get("artifact") != release.get("artifact")
        or data.get("policy") != policy
        or data.get("targetCount") != 1
        or not isinstance(data.get("waves"), list)
        or not isinstance(data.get("targets"), list)
        or len(data["targets"]) != 1
        or not isinstance(data.get("createdBy"), str)
        or not data["createdBy"].strip()
        or not isinstance(data.get("createdAt"), str)
        or not isinstance(data.get("updatedAt"), str)
        or data.get("status")
        not in {
            "DRAFT",
            "RUNNING",
            "PAUSED",
            "PAUSED_HEALTH_UNKNOWN",
            "HALTED",
            "SUCCEEDED",
        }
    ):
        raise RolloutControlError("rollout release/policy identity is invalid")
    target = data["targets"][0]
    if not isinstance(target, dict) or set(target) != TARGET_RESPONSE_FIELDS:
        raise RolloutControlError("rollout target response is invalid")
    identity_tuple = {
        "productModel": target.get("productModel"),
        "platformProfile": target.get("platformProfile"),
        "hardwareRevision": target.get("hardwareRevision"),
        "architecture": target.get("architecture"),
    }
    identity_snapshot = target.get("identitySnapshot")
    identity_is_exact = identity_tuple in release.get("targets", []) and isinstance(
        identity_snapshot, dict
    )
    identity_is_transiently_unknown = (
        target.get("eligibility") == "IDENTITY_UNKNOWN"
        and data.get("status") in {"DRAFT", "PAUSED_HEALTH_UNKNOWN"}
        and (
            identity_is_exact
            or (
                all(value is None for value in identity_tuple.values())
                and identity_snapshot is None
            )
        )
    )
    identity_shape_is_valid = (
        all(value is None for value in identity_tuple.values())
        and (identity_snapshot is None or isinstance(identity_snapshot, dict))
    ) or (
        all(isinstance(value, str) and value for value in identity_tuple.values())
        and isinstance(identity_snapshot, dict)
    )
    expected_cohort_key = (
        "UNKNOWN|UNKNOWN|UNKNOWN"
        if all(value is None for value in identity_tuple.values())
        else "|".join(
            str(identity_tuple[field])
            for field in ("productModel", "platformProfile", "hardwareRevision")
        )
    )
    if (
        target.get("deviceId") != device_id
        or target.get("eligibility")
        not in {"ELIGIBLE", "IDENTITY_UNKNOWN", "COMPATIBILITY_MISMATCH"}
        or target.get("status")
        not in {
            "PENDING",
            "COMMAND_ISSUED",
            "SUCCEEDED",
            "FAILED",
            "REJECTED",
            "ROLLED_BACK",
            "EXPIRED",
            "HEALTH_UNKNOWN",
        }
        or not identity_shape_is_valid
        or (
            require_eligible
            and not (
                (target.get("eligibility") == "ELIGIBLE" and identity_is_exact)
                or identity_is_transiently_unknown
            )
        )
        or target.get("cohortKey") != expected_cohort_key
        or target.get("waveNumber") != 0
    ):
        raise RolloutControlError("rollout target identity is not eligible/exact")
    created_at = _instant(data.get("createdAt"), label="rollout createdAt")
    updated_at = _instant(data.get("updatedAt"), label="rollout updatedAt")
    if updated_at < created_at:
        raise RolloutControlError("rollout timestamps are reordered")
    return data, target


def _command(
    target: Mapping[str, Any],
    *,
    space_id: int,
    device_id: str,
    release: Mapping[str, Any],
    expected_status: str,
) -> dict[str, Any]:
    command = target.get("latestCommand")
    if not isinstance(command, dict) or set(command) != COMMAND_RESPONSE_FIELDS:
        raise RolloutControlError("rollout command response is invalid")
    command_id = _uuid(command.get("commandId"), label="commandId")
    sequence = _positive_int(command.get("sequence"), label="command sequence")
    command_issued_at = _instant(
        target.get("commandIssuedAt"), label="rollout target commandIssuedAt"
    )
    signed_issued_at = _instant(
        command.get("issuedAt"), label="rollout command issuedAt"
    )
    desired = target.get("desiredEvidence")
    if (
        command.get("deviceId") != device_id
        or command.get("spaceId") != space_id
        or command.get("type") != "AGENT_UPDATE"
        or command.get("schemaVersion") != 1
        or command.get("status") != expected_status
        or not isinstance(command.get("issuedAt"), str)
        or not isinstance(command.get("expiresAt"), str)
        or not isinstance(command.get("payloadHash"), str)
        or not isinstance(command.get("actor"), str)
        or not command["actor"].strip()
        or command.get("authorizationContext") != "SPACE_ADMIN"
        or not isinstance(command.get("keyId"), str)
        or desired
        != {
            "targetVersion": release.get("agentVersion"),
            "bomDigest": release.get("bomDigest"),
        }
        or command_issued_at > signed_issued_at
    ):
        raise RolloutControlError("rollout command identity/evidence is invalid")
    summary = {
        "commandId": command_id,
        "sequence": sequence,
        "type": "AGENT_UPDATE",
        "status": expected_status,
        "issuedAt": command["issuedAt"],
        "expiresAt": command["expiresAt"],
        "payloadHash": command["payloadHash"],
        "actor": command["actor"],
        "authorizationContext": command["authorizationContext"],
        "keyId": command["keyId"],
    }
    return _validate_command_summary(
        summary,
        expected_statuses={expected_status},
        label="rollout command",
    )


def _validate_command_summary(
    command: Any,
    *,
    expected_statuses: set[str],
    label: str,
) -> dict[str, Any]:
    expected_fields = {
        "commandId",
        "sequence",
        "type",
        "status",
        "issuedAt",
        "expiresAt",
        "payloadHash",
        "actor",
        "authorizationContext",
        "keyId",
    }
    if not isinstance(command, dict) or set(command) != expected_fields:
        raise RolloutControlError(f"{label} fields are invalid")
    _uuid(command.get("commandId"), label=f"{label} commandId")
    _positive_int(command.get("sequence"), label=f"{label} sequence")
    issued = _instant(command.get("issuedAt"), label=f"{label} issuedAt")
    expires = _instant(command.get("expiresAt"), label=f"{label} expiresAt")
    if (
        command.get("type") != "AGENT_UPDATE"
        or command.get("status") not in expected_statuses
        or expires <= issued
        or SHA256_RE.fullmatch(str(command.get("payloadHash") or "")) is None
        or not isinstance(command.get("actor"), str)
        or not command["actor"].strip()
        or command.get("authorizationContext") != "SPACE_ADMIN"
        or not isinstance(command.get("keyId"), str)
        or not command["keyId"].strip()
    ):
        raise RolloutControlError(f"{label} identity is invalid")
    return command


def _validate_terminal_reported_evidence(
    evidence: Any,
    *,
    purpose: str,
    release: Mapping[str, Any],
    command: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(evidence, dict):
        raise RolloutControlError("terminal reported evidence is unavailable")
    expected_artifact = "sha256:" + str(release.get("artifact", {}).get("sha256") or "")
    common = {
        "targetVersion": release.get("agentVersion"),
        "bomDigest": release.get("bomDigest"),
        "artifactDigest": expected_artifact,
        "componentSha": release.get("componentSha"),
        "configSchema": release.get("configSchema"),
        "releaseSequence": release.get("releaseSequence"),
        "publisherKeyId": release.get("publisherKeyId"),
        "bomVerificationStatus": "VERIFIED",
        "candidateSlot": "/opt/nuv-agent/releases/"
        + str(release.get("bomDigest") or "")[7:],
    }
    if any(evidence.get(key) != value for key, value in common.items()):
        raise RolloutControlError("terminal reported release evidence is not exact")
    previous_version = evidence.get("previousVersion")
    if (
        not isinstance(previous_version, str)
        or SEMVER_RE.fullmatch(previous_version) is None
    ):
        raise RolloutControlError("terminal previousVersion is invalid")
    if purpose == "rollback":
        previous_slot = evidence.get("previousSlot")
        expected = {
            "commandId": command.get("commandId"),
            "phase": "ROLLED_BACK",
            "updatePhase": "ROLLED_BACK",
            "errorCode": "ROLLED_BACK",
            "health": "LKG_RESTORED",
            "functionalHealth": "FUNCTIONAL_UNHEALTHY",
            "rollbackVersion": previous_version,
            "rollbackSlot": previous_slot,
            "slot": previous_slot,
        }
        if (
            not isinstance(previous_slot, str)
            or SLOT_RE.fullmatch(previous_slot) is None
            or any(evidence.get(key) != value for key, value in expected.items())
        ):
            raise RolloutControlError("rollback terminal evidence is not strong/exact")
    else:
        expected_slot = "releases/" + str(release.get("bomDigest") or "")[7:]
        expected = {
            "commandId": command.get("commandId"),
            "phase": "COMMITTED",
            "agentVersion": release.get("agentVersion"),
            "health": "FUNCTIONAL_HEALTHY",
            "functionalHealth": "FUNCTIONAL_HEALTHY",
            "updatePhase": "COMMITTED",
            "slot": expected_slot,
        }
        if any(evidence.get(key) != value for key, value in expected.items()):
            raise RolloutControlError("commit terminal evidence is not strong/exact")
    return evidence


def _rollback_baseline_slot(
    identity: Mapping[str, Any], evidence: Mapping[str, Any]
) -> str:
    """Resolve the root-reported physical rollback baseline without top-level fallback."""
    agent_update = identity.get("agentUpdate")
    outer_updater_version = identity.get("updaterVersion")
    if (
        not isinstance(agent_update, dict)
        or agent_update.get("authenticatedHelper") is not True
        or agent_update.get("capabilityAvailable") is not True
        or not isinstance(outer_updater_version, str)
        or SEMVER_RE.fullmatch(outer_updater_version) is None
        or agent_update.get("updaterVersion") != outer_updater_version
    ):
        raise RolloutControlError("captured updater identity is not authenticated")
    active_slot = agent_update.get("activeSlot")
    if active_slot is not None:
        if not isinstance(active_slot, str) or SLOT_RE.fullmatch(active_slot) is None:
            raise RolloutControlError("captured rollback baseline slot is invalid")
        return active_slot

    # v0.1.120 is the single signed first-update baseline that predates activeSlot
    # forwarding.  Do not guess its physical slot before command execution: the
    # root updater captures previousSlot atomically.  At terminal validation,
    # accept only the canonical release slot that matches the captured verified
    # BOM; a stale BOM or bootstrap slot remains a hard failure.
    bom_digest = identity.get("bomDigest")
    previous_slot = evidence.get("previousSlot")
    if (
        identity.get("agentVersion") == LEGACY_SLOTLESS_AGENT_VERSION
        and identity.get("bomVerificationStatus") == "VERIFIED"
        and isinstance(bom_digest, str)
        and DIGEST_RE.fullmatch(bom_digest) is not None
        and previous_slot == "releases/" + bom_digest[7:]
    ):
        return str(previous_slot)
    raise RolloutControlError("captured rollback baseline slot is invalid")


def _active_matching_rollouts(
    values: Any,
    *,
    space_id: int,
    release: Mapping[str, Any],
    device_id: str,
    policy: Mapping[str, Any],
    client_request_id: str,
) -> list[str]:
    if not isinstance(values, list) or any(
        not isinstance(item, dict) for item in values
    ):
        raise RolloutControlError("rollout inventory is invalid")
    matches: list[str] = []
    for item in values:
        if (
            set(item) != ROLLOUT_RESPONSE_FIELDS
            or item.get("spaceId") != space_id
            or item.get("status")
            not in {
                "DRAFT",
                "RUNNING",
                "PAUSED",
                "PAUSED_HEALTH_UNKNOWN",
                "HALTED",
                "SUCCEEDED",
            }
            or not isinstance(item.get("targets"), list)
            or any(
                not isinstance(target, dict) or set(target) != TARGET_RESPONSE_FIELDS
                for target in item.get("targets", [])
            )
        ):
            raise RolloutControlError("rollout inventory entry is invalid")
        inventory_rollout_id = _uuid(item.get("rolloutId"), label="inventory rolloutId")
        targets = item.get("targets")
        targets_device = any(target.get("deviceId") == device_id for target in targets)
        if not targets_device:
            continue
        if item.get("status") in {"HALTED", "SUCCEEDED"}:
            continue
        if item.get("releaseId") != release.get("releaseId"):
            raise RolloutControlError(
                "conflicting active rollout exists for this device"
            )
        if item.get("policy") != policy:
            raise RolloutControlError(
                "conflicting active rollout exists for this release/device"
            )
        if item.get("clientRequestId") != client_request_id:
            raise RolloutControlError(
                "conflicting active rollout exists for this client request"
            )
        _validate_rollout(
            item,
            space_id=space_id,
            device_id=device_id,
            release=release,
            policy=policy,
            rollout_id=inventory_rollout_id,
            client_request_id=client_request_id,
            require_eligible=False,
        )
        matches.append(inventory_rollout_id)
    if len(matches) != len(set(matches)):
        raise RolloutControlError("rollout inventory contains duplicate entries")
    return sorted(matches)


def _load_issuance(
    path: Path,
    *,
    space_id: int,
    device_id: str,
    origin: str,
) -> tuple[dict[str, Any], bytes]:
    value, raw = _object_file(path, label="rollout issuance evidence", canonical=True)
    if set(value) != {
        "schemaVersion",
        "kind",
        "apiOrigin",
        "purpose",
        "spaceId",
        "deviceId",
        "releaseEvidenceSha256",
        "createdEvidenceSha256",
        "release",
        "rolloutId",
        "clientRequestId",
        "policy",
        "command",
        "createdAt",
        "commandIssuedAt",
    } or any(
        (
            value.get("schemaVersion") != 1,
            value.get("kind") != "nuvion-agent-rollout-issuance",
            value.get("apiOrigin") != origin,
            value.get("purpose") not in {"rollback", "commit"},
            value.get("spaceId") != space_id,
            value.get("deviceId") != device_id,
            not isinstance(value.get("release"), dict),
            not isinstance(value.get("policy"), dict),
            not isinstance(value.get("command"), dict),
            _uuid(value.get("clientRequestId"), label="issuance clientRequestId")
            != value.get("clientRequestId"),
        )
    ):
        raise RolloutControlError("rollout issuance evidence is invalid")
    _uuid(value.get("rolloutId"), label="issuance rolloutId")
    release = value["release"]
    command = value["command"]
    policy = value["policy"]
    if (
        set(release)
        != {
            "releaseId",
            "bomDigest",
            "releaseSequence",
            "agentVersion",
            "componentSha",
            "configSchema",
            "minUpdaterVersion",
            "targets",
            "artifact",
            "publisherKeyId",
            "createdAt",
        }
        or set(command)
        != {
            "commandId",
            "sequence",
            "type",
            "status",
            "issuedAt",
            "expiresAt",
            "payloadHash",
            "actor",
            "authorizationContext",
            "keyId",
        }
        or set(policy)
        != {
            "preCommitSoakSeconds",
            "commandTtlSeconds",
            "maxFailurePercent",
        }
        or SHA256_RE.fullmatch(str(value.get("releaseEvidenceSha256") or "")) is None
        or SHA256_RE.fullmatch(str(value.get("createdEvidenceSha256") or "")) is None
        or not isinstance(value.get("createdAt"), str)
        or not isinstance(value.get("commandIssuedAt"), str)
    ):
        raise RolloutControlError("rollout issuance identity is invalid")
    _uuid(release.get("releaseId"), label="issuance releaseId")
    expected_terminal_status = TERMINAL_COMMAND_STATUS_BY_PURPOSE[value["purpose"]]
    _validate_command_summary(
        command,
        expected_statuses=ACTIVE_COMMAND_STATUSES | {expected_terminal_status},
        label="issuance command",
    )
    created_at = _instant(value.get("createdAt"), label="issuance createdAt")
    command_issued_at = _instant(
        value.get("commandIssuedAt"), label="issuance commandIssuedAt"
    )
    issued_at = _instant(command.get("issuedAt"), label="issuance command issuedAt")
    if not (created_at <= command_issued_at <= issued_at):
        raise RolloutControlError("issuance timestamps are reordered")
    _policy(
        policy.get("preCommitSoakSeconds"),
        policy.get("commandTtlSeconds"),
        policy.get("maxFailurePercent"),
    )
    return value, raw


def _load_terminal(
    path: Path,
    *,
    issuance: Mapping[str, Any],
    issuance_raw: bytes,
) -> tuple[dict[str, Any], bytes]:
    value, raw = _object_file(path, label="terminal rollout evidence", canonical=True)
    if set(value) != {
        "schemaVersion",
        "kind",
        "apiOrigin",
        "purpose",
        "spaceId",
        "deviceId",
        "issuanceEvidenceSha256",
        "release",
        "rolloutId",
        "clientRequestId",
        "rolloutStatus",
        "targetStatus",
        "command",
        "reportedEvidence",
        "createdAt",
        "updatedAt",
        "terminalAt",
    } or any(
        (
            value.get("schemaVersion") != 1,
            value.get("kind") != "nuvion-agent-rollout-terminal",
            value.get("apiOrigin") != issuance.get("apiOrigin"),
            value.get("purpose") != issuance.get("purpose"),
            value.get("spaceId") != issuance.get("spaceId"),
            value.get("deviceId") != issuance.get("deviceId"),
            value.get("issuanceEvidenceSha256")
            != hashlib.sha256(issuance_raw).hexdigest(),
            value.get("release") != issuance.get("release"),
            value.get("rolloutId") != issuance.get("rolloutId"),
            value.get("clientRequestId") != issuance.get("clientRequestId"),
            not isinstance(value.get("reportedEvidence"), dict),
            not isinstance(value.get("terminalAt"), str),
        )
    ):
        raise RolloutControlError("terminal rollout evidence is invalid")
    expected_rollout = "HALTED" if issuance["purpose"] == "rollback" else "SUCCEEDED"
    expected_target = (
        "ROLLED_BACK" if issuance["purpose"] == "rollback" else "SUCCEEDED"
    )
    terminal_command = value.get("command")
    issuance_command = issuance.get("command")
    expected_command_status = TERMINAL_COMMAND_STATUS_BY_PURPOSE[issuance["purpose"]]
    if (
        value.get("rolloutStatus") != expected_rollout
        or value.get("targetStatus") != expected_target
        or not isinstance(terminal_command, dict)
        or not isinstance(issuance_command, dict)
        or any(
            terminal_command.get(field) != issuance_command.get(field)
            for field in (
                "commandId",
                "sequence",
                "type",
                "issuedAt",
                "expiresAt",
                "payloadHash",
                "actor",
                "authorizationContext",
                "keyId",
            )
        )
    ):
        raise RolloutControlError("terminal rollout result is not exact")
    _validate_command_summary(
        terminal_command,
        expected_statuses={expected_command_status},
        label="terminal command",
    )
    _validate_terminal_reported_evidence(
        value.get("reportedEvidence"),
        purpose=issuance["purpose"],
        release=issuance["release"],
        command=terminal_command,
    )
    created_at = _instant(value.get("createdAt"), label="terminal createdAt")
    updated_at = _instant(value.get("updatedAt"), label="terminal updatedAt")
    terminal_at = _instant(value.get("terminalAt"), label="terminal terminalAt")
    issued_at = _instant(terminal_command.get("issuedAt"), label="terminal issuedAt")
    issuance_command_issued_at = _instant(
        issuance.get("commandIssuedAt"), label="terminal issuance commandIssuedAt"
    )
    if value.get("createdAt") != issuance.get("createdAt") or not (
        created_at
        <= issuance_command_issued_at
        <= issued_at
        <= terminal_at
        <= updated_at
    ):
        raise RolloutControlError("terminal rollout timestamps are reordered")
    return value, raw


def _created_evidence(
    *,
    api: FleetApi,
    purpose: str,
    space_id: int,
    device_id: str,
    release: Mapping[str, Any],
    release_raw: bytes,
    policy: Mapping[str, Any],
    client_request_id: str,
    projection: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "kind": "nuvion-agent-rollout-created",
        "apiOrigin": api.origin,
        "purpose": purpose,
        "spaceId": space_id,
        "deviceId": device_id,
        "releaseEvidenceSha256": hashlib.sha256(release_raw).hexdigest(),
        "release": dict(release),
        "rolloutId": projection["rolloutId"],
        "clientRequestId": client_request_id,
        "policy": dict(policy),
        "createdAt": projection["createdAt"],
    }


def _load_created(
    path: Path,
    *,
    api: FleetApi,
    purpose: str,
    space_id: int,
    device_id: str,
    release: Mapping[str, Any],
    release_raw: bytes,
    policy: Mapping[str, Any],
    client_request_id: str | None = None,
) -> tuple[dict[str, Any], bytes]:
    value, raw = _object_file(path, label="rollout creation evidence", canonical=True)
    if set(value) != {
        "schemaVersion",
        "kind",
        "apiOrigin",
        "purpose",
        "spaceId",
        "deviceId",
        "releaseEvidenceSha256",
        "release",
        "rolloutId",
        "clientRequestId",
        "policy",
        "createdAt",
    } or any(
        (
            value.get("schemaVersion") != 1,
            value.get("kind") != "nuvion-agent-rollout-created",
            value.get("apiOrigin") != api.origin,
            value.get("purpose") != purpose,
            value.get("spaceId") != space_id,
            value.get("deviceId") != device_id,
            value.get("releaseEvidenceSha256")
            != hashlib.sha256(release_raw).hexdigest(),
            value.get("release") != release,
            value.get("policy") != policy,
            client_request_id is not None
            and value.get("clientRequestId") != client_request_id,
        )
    ):
        raise RolloutControlError("rollout creation evidence is invalid")
    _uuid(value.get("rolloutId"), label="created rolloutId")
    _uuid(value.get("clientRequestId"), label="created clientRequestId")
    _instant(value.get("createdAt"), label="created rollout createdAt")
    return value, raw


def _validated_predecessor(
    api: FleetApi,
    *,
    space_id: int,
    device_id: str,
    release: Mapping[str, Any],
    release_raw: bytes,
    issuance_path: Path,
    terminal_path: Path,
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    previous, previous_raw = _load_issuance(
        issuance_path,
        space_id=space_id,
        device_id=device_id,
        origin=api.origin,
    )
    if (
        previous.get("purpose") != "rollback"
        or previous.get("release") != release
        or previous.get("releaseEvidenceSha256")
        != hashlib.sha256(release_raw).hexdigest()
    ):
        raise RolloutControlError("commit predecessor is not bound to this release")
    previous_terminal, _previous_terminal_raw = _load_terminal(
        terminal_path,
        issuance=previous,
        issuance_raw=previous_raw,
    )
    projection = api.request(
        f"/spaces/{space_id}/agent-rollouts/{previous['rolloutId']}"
    )
    projection, target = _validate_rollout(
        projection,
        space_id=space_id,
        device_id=device_id,
        release=release,
        policy=previous["policy"],
        rollout_id=previous["rolloutId"],
        client_request_id=previous["clientRequestId"],
    )
    live_terminal = _terminal_state(projection, target, issuance=previous)
    if live_terminal is None:
        raise RolloutControlError("rollback rollout is not terminal before commit")
    live_terminal["issuanceEvidenceSha256"] = hashlib.sha256(previous_raw).hexdigest()
    if live_terminal != previous_terminal:
        raise RolloutControlError("rollback terminal evidence differs from live state")

    journal = api.request(f"/spaces/{space_id}/devices/{device_id}/commands")
    if not isinstance(journal, list) or not journal or not isinstance(journal[0], dict):
        raise RolloutControlError("Fleet command journal is unavailable")
    latest = journal[0]
    latest_summary = {
        field: latest.get(field)
        for field in (
            "commandId",
            "sequence",
            "type",
            "status",
            "issuedAt",
            "expiresAt",
            "payloadHash",
            "actor",
            "authorizationContext",
            "keyId",
        )
    }
    expected_latest_summary = {**previous["command"], "status": "ROLLED_BACK"}
    if (
        set(latest) != COMMAND_RESPONSE_FIELDS
        or latest.get("deviceId") != device_id
        or latest.get("spaceId") != space_id
        or latest.get("schemaVersion") != 1
        or latest_summary != expected_latest_summary
    ):
        raise RolloutControlError(
            "rollback command is no longer the latest device sequence"
        )
    _validate_command_summary(
        latest_summary,
        expected_statuses={"ROLLED_BACK"},
        label="latest rollback command",
    )
    return previous, previous_raw, previous_terminal


def _abort_rollout(
    api: FleetApi,
    *,
    space_id: int,
    device_id: str,
    release: Mapping[str, Any],
    policy: Mapping[str, Any],
    rollout_id: str,
    client_request_id: str,
    reason: str,
    expected_created_at: str | None = None,
) -> dict[str, Any]:
    projection = api.request(f"/spaces/{space_id}/agent-rollouts/{rollout_id}")
    projection, target = _validate_rollout(
        projection,
        space_id=space_id,
        device_id=device_id,
        release=release,
        policy=policy,
        rollout_id=rollout_id,
        client_request_id=client_request_id,
        require_eligible=False,
    )
    if (
        expected_created_at is not None
        and projection.get("createdAt") != expected_created_at
    ):
        raise RolloutControlError(
            "live rollout creation time differs from creation evidence"
        )
    initial_latest = target.get("latestCommand")
    initial_command_status = (
        initial_latest.get("status") if isinstance(initial_latest, dict) else None
    )
    action = "already-terminal"
    if projection.get("status") not in {"HALTED", "SUCCEEDED"}:
        projection = api.request(
            f"/spaces/{space_id}/agent-rollouts/{rollout_id}/halt",
            method="POST",
        )
        projection, target = _validate_rollout(
            projection,
            space_id=space_id,
            device_id=device_id,
            release=release,
            policy=policy,
            rollout_id=rollout_id,
            client_request_id=client_request_id,
            require_eligible=False,
        )
        action = "halted"
    if projection.get("status") not in {"HALTED", "SUCCEEDED"}:
        raise RolloutControlError("rollout abort did not reach a terminal state")
    latest = target.get("latestCommand")
    latest_status = latest.get("status") if isinstance(latest, dict) else None
    if isinstance(latest, dict):
        if latest_status not in {
            "QUEUED",
            "RECEIVED",
            "IN_PROGRESS",
            "SUCCEEDED",
            "FAILED",
            "REJECTED",
            "EXPIRED",
            "ROLLED_BACK",
        }:
            raise RolloutControlError("halted rollout command status is invalid")
        _command(
            target,
            space_id=space_id,
            device_id=device_id,
            release=release,
            expected_status=latest_status,
        )
    if latest_status in ACTIVE_COMMAND_STATUSES:
        raise RolloutControlError("halted rollout still has an in-flight Agent command")
    if initial_command_status == "QUEUED" and (
        latest_status != "EXPIRED" or target.get("status") != "EXPIRED"
    ):
        raise RolloutControlError("halt did not expire the queued Agent command")
    return {
        "schemaVersion": 1,
        "kind": "nuvion-agent-rollout-abort",
        "apiOrigin": api.origin,
        "spaceId": space_id,
        "deviceId": device_id,
        "releaseId": release["releaseId"],
        "rolloutId": rollout_id,
        "clientRequestId": client_request_id,
        "action": action,
        "reason": reason,
        "rolloutStatus": projection["status"],
        "targetStatus": target.get("status"),
        "commandId": latest.get("commandId") if isinstance(latest, dict) else None,
        "commandStatus": latest.get("status") if isinstance(latest, dict) else None,
        "updatedAt": projection["updatedAt"],
    }


def issue_rollout(
    api: FleetApi,
    *,
    space_id: int,
    device_id: str,
    release_evidence_path: Path,
    purpose: str,
    client_request_id: str,
    pre_commit_soak_seconds: int,
    command_ttl_seconds: int,
    max_failure_percent: int,
    previous_issuance_path: Path | None,
    previous_terminal_path: Path | None,
    created_evidence_path: Path,
    output: Path,
    adopt_rollout_id: str | None = None,
    failure_evidence_path: Path | None = None,
    wait_seconds: int = 180,
    poll_seconds: float = 5.0,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    client_request_id = _uuid(client_request_id, label="clientRequestId")
    release_evidence, release_raw = _load_release(
        release_evidence_path, space_id=space_id, origin=api.origin
    )
    release = release_evidence["release"]
    policy = _policy(pre_commit_soak_seconds, command_ttl_seconds, max_failure_percent)
    if purpose not in {"rollback", "commit"}:
        raise RolloutControlError("rollout purpose is invalid")
    if not 30 <= wait_seconds <= 1800 or not 0.5 <= poll_seconds <= 30.0:
        raise RolloutControlError("issuance wait bounds are invalid")

    # All caller-controlled files and predecessor claims are validated before
    # create/start can allocate a rollout or a signed command.
    _preflight_output(created_evidence_path, allow_existing=True)
    _preflight_output(output, allow_existing=True)
    if failure_evidence_path is not None:
        _preflight_output(failure_evidence_path)
    evidence_paths = [created_evidence_path, output]
    if failure_evidence_path is not None:
        evidence_paths.append(failure_evidence_path)
    if len(set(evidence_paths)) != len(evidence_paths):
        raise RolloutControlError("rollout evidence paths must be distinct")
    predecessor: tuple[dict[str, Any], bytes, dict[str, Any]] | None = None
    if purpose == "commit":
        if previous_issuance_path is None or previous_terminal_path is None:
            raise RolloutControlError(
                "commit issuance requires rollback issuance and terminal evidence"
            )
        predecessor = _validated_predecessor(
            api,
            space_id=space_id,
            device_id=device_id,
            release=release,
            release_raw=release_raw,
            issuance_path=previous_issuance_path,
            terminal_path=previous_terminal_path,
        )
    elif previous_issuance_path is not None or previous_terminal_path is not None:
        raise RolloutControlError("rollback issuance cannot reference a predecessor")

    if adopt_rollout_id is not None:
        adopt_rollout_id = _uuid(adopt_rollout_id, label="adopt rolloutId")

    deadline = monotonic() + wait_seconds

    def read_rollout(current_rollout_id: str) -> Any:
        while True:
            try:
                return api.request(
                    f"/spaces/{space_id}/agent-rollouts/{current_rollout_id}"
                )
            except RolloutControlError as exc:
                if monotonic() >= deadline:
                    raise RolloutControlError(
                        "timed out after Fleet rollout status read failures"
                    ) from exc
                sleeper(poll_seconds)

    if output.exists():
        if not created_evidence_path.exists():
            raise RolloutControlError(
                "issuance evidence exists without creation evidence"
            )
        created_record, created_raw = _load_created(
            created_evidence_path,
            api=api,
            purpose=purpose,
            space_id=space_id,
            device_id=device_id,
            release=release,
            release_raw=release_raw,
            policy=policy,
            client_request_id=client_request_id,
        )
        persisted, _persisted_raw = _load_issuance(
            output,
            space_id=space_id,
            device_id=device_id,
            origin=api.origin,
        )
        if (
            persisted.get("purpose") != purpose
            or persisted.get("release") != release
            or persisted.get("policy") != policy
            or persisted.get("rolloutId") != created_record.get("rolloutId")
            or persisted.get("clientRequestId") != client_request_id
            or persisted.get("releaseEvidenceSha256")
            != hashlib.sha256(release_raw).hexdigest()
            or persisted.get("createdEvidenceSha256")
            != hashlib.sha256(created_raw).hexdigest()
            or (
                adopt_rollout_id is not None
                and adopt_rollout_id != persisted["rolloutId"]
            )
        ):
            raise RolloutControlError("persisted issuance evidence is not exact")
        live = read_rollout(persisted["rolloutId"])
        live, live_target = _validate_rollout(
            live,
            space_id=space_id,
            device_id=device_id,
            release=release,
            policy=policy,
            rollout_id=persisted["rolloutId"],
            client_request_id=client_request_id,
        )
        live_command = live_target.get("latestCommand")
        allowed_live_statuses = ACTIVE_COMMAND_STATUSES | {
            TERMINAL_COMMAND_STATUS_BY_PURPOSE[purpose]
        }
        if (
            not isinstance(live_command, dict)
            or live_command.get("status") not in allowed_live_statuses
        ):
            raise RolloutControlError("live rollout differs from persisted issuance")
        live_summary = _command(
            live_target,
            space_id=space_id,
            device_id=device_id,
            release=release,
            expected_status=live_command["status"],
        )
        if any(
            live_summary.get(field) != persisted["command"].get(field)
            for field in (
                "commandId",
                "sequence",
                "type",
                "issuedAt",
                "expiresAt",
                "payloadHash",
                "actor",
                "authorizationContext",
                "keyId",
            )
        ):
            raise RolloutControlError("live rollout differs from persisted issuance")
        return persisted

    created_raw: bytes | None = None
    rollout_id: str | None = None
    try:
        if created_evidence_path.exists() or created_evidence_path.is_symlink():
            if created_evidence_path.is_symlink():
                raise RolloutControlError("rollout creation evidence path is unsafe")
            created_record, created_raw = _load_created(
                created_evidence_path,
                api=api,
                purpose=purpose,
                space_id=space_id,
                device_id=device_id,
                release=release,
                release_raw=release_raw,
                policy=policy,
                client_request_id=client_request_id,
            )
            rollout_id = created_record["rolloutId"]
            if adopt_rollout_id is not None and adopt_rollout_id != rollout_id:
                raise RolloutControlError(
                    "adopt rolloutId differs from creation evidence"
                )
            projection = read_rollout(rollout_id)
            projection, _target = _validate_rollout(
                projection,
                space_id=space_id,
                device_id=device_id,
                release=release,
                policy=policy,
                rollout_id=rollout_id,
                client_request_id=client_request_id,
                require_eligible=False,
            )
            if projection.get("createdAt") != created_record.get("createdAt"):
                raise RolloutControlError(
                    "live rollout creation time differs from creation evidence"
                )
        else:
            inventory = api.request(f"/spaces/{space_id}/agent-rollouts")
            active = _active_matching_rollouts(
                inventory,
                space_id=space_id,
                release=release,
                device_id=device_id,
                policy=policy,
                client_request_id=client_request_id,
            )
            if adopt_rollout_id is not None:
                if active != [adopt_rollout_id]:
                    raise RolloutControlError(
                        "adopt rolloutId is not the unique exact active rollout"
                    )
                rollout_id = adopt_rollout_id
                projection = read_rollout(rollout_id)
                projection, _target = _validate_rollout(
                    projection,
                    space_id=space_id,
                    device_id=device_id,
                    release=release,
                    policy=policy,
                    rollout_id=rollout_id,
                    client_request_id=client_request_id,
                    require_eligible=False,
                )
            elif active:
                rollout_id = active[0]
                projection = read_rollout(rollout_id)
                projection, _target = _validate_rollout(
                    projection,
                    space_id=space_id,
                    device_id=device_id,
                    release=release,
                    policy=policy,
                    rollout_id=rollout_id,
                    client_request_id=client_request_id,
                    require_eligible=False,
                )
            else:
                try:
                    projection = api.request(
                        f"/spaces/{space_id}/agent-rollouts",
                        method="POST",
                        payload={
                            "clientRequestId": client_request_id,
                            "releaseId": release["releaseId"],
                            "deviceIds": [device_id],
                            "policy": policy,
                        },
                    )
                except Exception as create_error:
                    # The BE uniqueness key makes this response-loss lookup
                    # attributable to this exact request rather than merely a
                    # matching release/device/policy tuple.
                    try:
                        recovered_inventory = api.request(
                            f"/spaces/{space_id}/agent-rollouts"
                        )
                        recovered = _active_matching_rollouts(
                            recovered_inventory,
                            space_id=space_id,
                            release=release,
                            device_id=device_id,
                            policy=policy,
                            client_request_id=client_request_id,
                        )
                    except Exception:  # noqa: BLE001 - preserve the first failure.
                        raise create_error
                    if len(recovered) != 1:
                        raise create_error  # noqa: TRY201
                    rollout_id = recovered[0]
                    projection = read_rollout(rollout_id)
                if rollout_id is None:
                    if not isinstance(projection, dict):
                        raise RolloutControlError(
                            "new rollout response is not an object"
                        )
                    rollout_id = _uuid(
                        projection.get("rolloutId"), label="created rolloutId"
                    )
                projection, target = _validate_rollout(
                    projection,
                    space_id=space_id,
                    device_id=device_id,
                    release=release,
                    policy=policy,
                    rollout_id=rollout_id,
                    client_request_id=client_request_id,
                    require_eligible=False,
                )
                if (
                    projection.get("status") != "DRAFT"
                    or target.get("status") != "PENDING"
                    or target.get("latestCommand") is not None
                ):
                    raise RolloutControlError(
                        "new rollout did not enter the exact DRAFT state"
                    )
            created_raw = _write_new(
                created_evidence_path,
                _created_evidence(
                    api=api,
                    purpose=purpose,
                    space_id=space_id,
                    device_id=device_id,
                    release=release,
                    release_raw=release_raw,
                    policy=policy,
                    client_request_id=client_request_id,
                    projection=projection,
                ),
            )

        if rollout_id is None or created_raw is None:
            raise RolloutControlError("rollout creation evidence was not persisted")

        resume_attempted = False
        expected_terminal_status = TERMINAL_COMMAND_STATUS_BY_PURPOSE[purpose]
        allowed_command_statuses = ACTIVE_COMMAND_STATUSES | {expected_terminal_status}
        while True:
            projection, target = _validate_rollout(
                projection,
                space_id=space_id,
                device_id=device_id,
                release=release,
                policy=policy,
                rollout_id=rollout_id,
                client_request_id=client_request_id,
                require_eligible=False,
            )
            live_command = target.get("latestCommand")
            live_command_status = (
                live_command.get("status") if isinstance(live_command, dict) else None
            )
            active_issuance = (
                live_command_status in ACTIVE_COMMAND_STATUSES
                and target.get("status") == "COMMAND_ISSUED"
                and projection.get("status") in {"RUNNING", "PAUSED_HEALTH_UNKNOWN"}
            )
            recovered_terminal = live_command_status == expected_terminal_status and (
                (
                    purpose == "rollback"
                    and projection.get("status") == "HALTED"
                    and target.get("status") == "ROLLED_BACK"
                )
                or (
                    purpose == "commit"
                    and projection.get("status") == "SUCCEEDED"
                    and target.get("status") == "SUCCEEDED"
                )
                or (
                    purpose == "commit"
                    and projection.get("status") == "PAUSED_HEALTH_UNKNOWN"
                    and target.get("status") == "SUCCEEDED"
                )
            )
            command_ready = (
                isinstance(live_command, dict)
                and live_command_status in allowed_command_statuses
                and (active_issuance or recovered_terminal)
            )
            if command_ready:
                break

            status = projection.get("status")
            should_sleep = True
            if status == "DRAFT":
                try:
                    projection = api.request(
                        f"/spaces/{space_id}/agent-rollouts/{rollout_id}/start",
                        method="POST",
                    )
                    should_sleep = False
                except Exception:  # noqa: BLE001 - reconcile ambiguous POST.
                    # GET disambiguates a lost successful response from an
                    # uncommitted request. A still-DRAFT rollout is retried
                    # within the same bounded issuance deadline.
                    projection = read_rollout(rollout_id)
            elif status == "PAUSED_HEALTH_UNKNOWN":
                if not resume_attempted:
                    resume_attempted = True
                    try:
                        projection = api.request(
                            f"/spaces/{space_id}/agent-rollouts/{rollout_id}/resume",
                            method="POST",
                        )
                        should_sleep = False
                    except Exception:  # noqa: BLE001 - reconcile ambiguous POST.
                        projection = read_rollout(rollout_id)
                else:
                    projection = read_rollout(rollout_id)
            elif status == "RUNNING":
                projection = read_rollout(rollout_id)
            else:
                raise RolloutControlError(
                    "rollout cannot reach an exact command issuance state"
                )
            if not should_sleep:
                continue
            if monotonic() >= deadline:
                raise RolloutControlError(
                    "timed out waiting for exact rollout command issuance"
                )
            sleeper(poll_seconds)

        command = _command(
            target,
            space_id=space_id,
            device_id=device_id,
            release=release,
            expected_status=live_command["status"],
        )
        if predecessor is not None:
            previous = predecessor[0]
            if (
                previous["command"]["commandId"] == command["commandId"]
                or previous["command"]["sequence"] + 1 != command["sequence"]
            ):
                raise RolloutControlError(
                    "commit command is not adjacent to rollback command"
                )
        evidence = {
            "schemaVersion": 1,
            "kind": "nuvion-agent-rollout-issuance",
            "apiOrigin": api.origin,
            "purpose": purpose,
            "spaceId": space_id,
            "deviceId": device_id,
            "releaseEvidenceSha256": hashlib.sha256(release_raw).hexdigest(),
            "createdEvidenceSha256": hashlib.sha256(created_raw).hexdigest(),
            "release": release,
            "rolloutId": rollout_id,
            "clientRequestId": client_request_id,
            "policy": policy,
            "command": command,
            "createdAt": projection["createdAt"],
            "commandIssuedAt": target["commandIssuedAt"],
        }
        _write_new(output, evidence)
        return evidence
    except Exception:
        if rollout_id is not None:
            try:
                abort_created_at: str | None = None
                if created_evidence_path.exists():
                    abort_created, _abort_created_raw = _load_created(
                        created_evidence_path,
                        api=api,
                        purpose=purpose,
                        space_id=space_id,
                        device_id=device_id,
                        release=release,
                        release_raw=release_raw,
                        policy=policy,
                        client_request_id=client_request_id,
                    )
                    abort_created_at = abort_created["createdAt"]
                aborted = _abort_rollout(
                    api,
                    space_id=space_id,
                    device_id=device_id,
                    release=release,
                    policy=policy,
                    rollout_id=rollout_id,
                    client_request_id=client_request_id,
                    reason="issuance-failed",
                    expected_created_at=abort_created_at,
                )
                if failure_evidence_path is not None:
                    _write_new(failure_evidence_path, aborted)
            except Exception:  # noqa: BLE001, S110 - primary failure wins.
                pass
        raise


def halt_rollout(
    api: FleetApi,
    *,
    space_id: int,
    device_id: str,
    release_evidence_path: Path,
    created_evidence_path: Path,
    purpose: str,
    pre_commit_soak_seconds: int,
    command_ttl_seconds: int,
    max_failure_percent: int,
    output: Path,
) -> dict[str, Any]:
    _preflight_output(output, allow_existing=True)
    if output in {release_evidence_path, created_evidence_path}:
        raise RolloutControlError("rollout halt evidence paths must be distinct")
    if purpose not in {"rollback", "commit"}:
        raise RolloutControlError("rollout purpose is invalid")
    release_evidence, release_raw = _load_release(
        release_evidence_path, space_id=space_id, origin=api.origin
    )
    release = release_evidence["release"]
    policy = _policy(pre_commit_soak_seconds, command_ttl_seconds, max_failure_percent)
    created, _created_raw = _load_created(
        created_evidence_path,
        api=api,
        purpose=purpose,
        space_id=space_id,
        device_id=device_id,
        release=release,
        release_raw=release_raw,
        policy=policy,
    )
    persisted: dict[str, Any] | None = None
    if output.exists():
        persisted, _raw = _object_file(
            output, label="rollout abort evidence", canonical=True
        )
        expected_fields = {
            "schemaVersion",
            "kind",
            "apiOrigin",
            "spaceId",
            "deviceId",
            "releaseId",
            "rolloutId",
            "clientRequestId",
            "action",
            "reason",
            "rolloutStatus",
            "targetStatus",
            "commandId",
            "commandStatus",
            "updatedAt",
        }
        if (
            set(persisted) != expected_fields
            or persisted.get("schemaVersion") != 1
            or persisted.get("kind") != "nuvion-agent-rollout-abort"
            or persisted.get("apiOrigin") != api.origin
            or persisted.get("rolloutId") != created["rolloutId"]
            or persisted.get("clientRequestId") != created["clientRequestId"]
            or persisted.get("releaseId") != release["releaseId"]
            or persisted.get("spaceId") != space_id
            or persisted.get("deviceId") != device_id
            or persisted.get("action") not in {"halted", "already-terminal"}
            or not isinstance(persisted.get("reason"), str)
            or not persisted["reason"]
            or persisted.get("rolloutStatus") not in {"HALTED", "SUCCEEDED"}
            or persisted.get("targetStatus")
            not in {
                "PENDING",
                "COMMAND_ISSUED",
                "SUCCEEDED",
                "FAILED",
                "REJECTED",
                "ROLLED_BACK",
                "EXPIRED",
                "HEALTH_UNKNOWN",
            }
            or (persisted.get("commandId") is None)
            != (persisted.get("commandStatus") is None)
            or (
                persisted.get("commandStatus") is not None
                and persisted.get("commandStatus")
                not in {"SUCCEEDED", "FAILED", "REJECTED", "EXPIRED", "ROLLED_BACK"}
            )
            or (
                persisted.get("commandId") is not None
                and _uuid(persisted.get("commandId"), label="abort commandId")
                != persisted.get("commandId")
            )
        ):
            raise RolloutControlError("persisted rollout abort evidence is invalid")
        _instant(persisted.get("updatedAt"), label="abort updatedAt")
    result = _abort_rollout(
        api,
        space_id=space_id,
        device_id=device_id,
        release=release,
        policy=policy,
        rollout_id=created["rolloutId"],
        client_request_id=created["clientRequestId"],
        reason=(
            persisted["reason"] if persisted is not None else "runbook-failure-cleanup"
        ),
        expected_created_at=created["createdAt"],
    )
    if persisted is not None:
        if any(
            persisted.get(field) != result.get(field)
            for field in (
                "schemaVersion",
                "kind",
                "apiOrigin",
                "spaceId",
                "deviceId",
                "releaseId",
                "rolloutId",
                "reason",
                "rolloutStatus",
                "targetStatus",
                "commandId",
                "commandStatus",
                "updatedAt",
            )
        ):
            raise RolloutControlError(
                "live rollout differs from persisted abort evidence"
            )
        return persisted
    _write_new(output, result)
    return result


def _terminal_state(
    projection: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    issuance: Mapping[str, Any],
) -> dict[str, Any] | None:
    release = issuance["release"]
    purpose = issuance["purpose"]
    expected_command = issuance["command"]
    command_status = "ROLLED_BACK" if purpose == "rollback" else "SUCCEEDED"
    latest = target.get("latestCommand")
    command = (
        _command(
            target,
            space_id=issuance["spaceId"],
            device_id=issuance["deviceId"],
            release=release,
            expected_status=command_status,
        )
        if isinstance(latest, dict) and latest.get("status") == command_status
        else None
    )
    if command is None:
        if projection.get("status") in {"HALTED", "SUCCEEDED"} or target.get(
            "status"
        ) in {"FAILED", "REJECTED", "ROLLED_BACK", "EXPIRED", "SUCCEEDED"}:
            raise RolloutControlError("rollout reached an unexpected terminal state")
        return None
    if (
        purpose == "rollback"
        and command_status == "ROLLED_BACK"
        and target.get("status") == "COMMAND_ISSUED"
        and projection.get("status") == "RUNNING"
    ):
        # The command ACK and rollout projection are committed by separate BE
        # transactions. Keep polling during that bounded, expected interval.
        return None
    if (
        command["commandId"] != expected_command["commandId"]
        or command["sequence"] != expected_command["sequence"]
        or any(
            command.get(field) != expected_command.get(field)
            for field in (
                "type",
                "issuedAt",
                "expiresAt",
                "payloadHash",
                "actor",
                "authorizationContext",
                "keyId",
            )
        )
    ):
        raise RolloutControlError("terminal rollout command identity changed")
    if (
        purpose == "commit"
        and command_status == "SUCCEEDED"
        and target.get("status") in {"COMMAND_ISSUED", "SUCCEEDED"}
        and projection.get("status") in {"RUNNING", "PAUSED_HEALTH_UNKNOWN"}
    ):
        # PAUSED_HEALTH_UNKNOWN is an explicitly recoverable coordinator state.
        # Keep polling until a fresh exact heartbeat makes the rollout
        # SUCCEEDED or the bounded deadline expires.
        return None
    identity = target.get("identitySnapshot")
    if not isinstance(identity, dict):
        raise RolloutControlError("captured rollout identity is unavailable")
    expected_slot = "releases/" + str(release["bomDigest"])[7:]
    expected_artifact = "sha256:" + str(release["artifact"].get("sha256") or "")
    common = {
        "targetVersion": release["agentVersion"],
        "bomDigest": release["bomDigest"],
        "artifactDigest": expected_artifact,
        "componentSha": release["componentSha"],
        "configSchema": release["configSchema"],
        "releaseSequence": release["releaseSequence"],
        "publisherKeyId": release["publisherKeyId"],
        "bomVerificationStatus": "VERIFIED",
        "candidateSlot": "/opt/nuv-agent/releases/" + release["bomDigest"][7:],
    }
    if purpose == "rollback":
        evidence = target.get("rollbackEvidence")
        previous_version = identity.get("agentVersion")
        previous_slot = _rollback_baseline_slot(identity, evidence)
        expected = {
            **common,
            "commandId": command["commandId"],
            "phase": "ROLLED_BACK",
            "updatePhase": "ROLLED_BACK",
            "errorCode": "ROLLED_BACK",
            "health": "LKG_RESTORED",
            "functionalHealth": "FUNCTIONAL_UNHEALTHY",
            "previousVersion": previous_version,
            "rollbackVersion": previous_version,
            "previousSlot": previous_slot,
            "rollbackSlot": previous_slot,
            "slot": previous_slot,
        }
        if (
            projection.get("status") != "HALTED"
            or target.get("status") != "ROLLED_BACK"
            or not isinstance(evidence, dict)
            or not isinstance(previous_version, str)
            or any(evidence.get(key) != value for key, value in expected.items())
        ):
            raise RolloutControlError("rollback terminal evidence is not strong/exact")
    else:
        evidence = target.get("reportedEvidence")
        previous_version = identity.get("agentVersion")
        expected = {
            **common,
            "commandId": command["commandId"],
            "phase": "COMMITTED",
            "agentVersion": release["agentVersion"],
            "health": "FUNCTIONAL_HEALTHY",
            "functionalHealth": "FUNCTIONAL_HEALTHY",
            "updatePhase": "COMMITTED",
            "slot": expected_slot,
            "previousVersion": previous_version,
        }
        if (
            projection.get("status") != "SUCCEEDED"
            or target.get("status") != "SUCCEEDED"
            or not isinstance(evidence, dict)
            or not isinstance(previous_version, str)
            or any(evidence.get(key) != value for key, value in expected.items())
        ):
            raise RolloutControlError("commit terminal evidence is not strong/exact")
    terminal_at = target.get("terminalAt") or target.get("succeededAt")
    created_at = _instant(projection.get("createdAt"), label="rollout createdAt")
    updated_at = _instant(projection.get("updatedAt"), label="rollout updatedAt")
    terminal_instant = _instant(terminal_at, label="rollout terminalAt")
    issued_at = _instant(command.get("issuedAt"), label="rollout issuedAt")
    if not (created_at <= issued_at <= terminal_instant <= updated_at):
        raise RolloutControlError("terminal rollout timestamps are reordered")
    return {
        "schemaVersion": 1,
        "kind": "nuvion-agent-rollout-terminal",
        "apiOrigin": issuance["apiOrigin"],
        "purpose": purpose,
        "spaceId": issuance["spaceId"],
        "deviceId": issuance["deviceId"],
        "issuanceEvidenceSha256": None,
        "release": release,
        "rolloutId": issuance["rolloutId"],
        "clientRequestId": issuance["clientRequestId"],
        "rolloutStatus": projection["status"],
        "targetStatus": target["status"],
        "command": command,
        "reportedEvidence": evidence,
        "createdAt": projection["createdAt"],
        "updatedAt": projection["updatedAt"],
        "terminalAt": terminal_at,
    }


def wait_terminal(
    api: FleetApi,
    *,
    space_id: int,
    device_id: str,
    issuance_path: Path,
    release_evidence_path: Path,
    output: Path,
    wait_seconds: int,
    poll_seconds: float,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if not 30 <= wait_seconds <= 1800 or not 0.5 <= poll_seconds <= 30.0:
        raise RolloutControlError("terminal wait bounds are invalid")
    _preflight_output(output, allow_existing=True)
    issuance, issuance_raw = _load_issuance(
        issuance_path,
        space_id=space_id,
        device_id=device_id,
        origin=api.origin,
    )
    release_evidence, release_raw = _load_release(
        release_evidence_path,
        space_id=space_id,
        origin=api.origin,
    )
    if (
        issuance.get("release") != release_evidence.get("release")
        or issuance.get("releaseEvidenceSha256")
        != hashlib.sha256(release_raw).hexdigest()
    ):
        raise RolloutControlError("issuance is not bound to release evidence")
    deadline = monotonic() + wait_seconds
    while True:
        try:
            projection = api.request(
                f"/spaces/{space_id}/agent-rollouts/{issuance['rolloutId']}"
            )
        except RolloutControlError as exc:
            if monotonic() >= deadline:
                raise RolloutControlError(
                    "timed out after Fleet rollout status read failures"
                ) from exc
            sleeper(poll_seconds)
            continue
        projection, target = _validate_rollout(
            projection,
            space_id=space_id,
            device_id=device_id,
            release=issuance["release"],
            policy=issuance["policy"],
            rollout_id=issuance["rolloutId"],
            client_request_id=issuance["clientRequestId"],
        )
        terminal = _terminal_state(projection, target, issuance=issuance)
        if terminal is not None:
            terminal["issuanceEvidenceSha256"] = hashlib.sha256(
                issuance_raw
            ).hexdigest()
            if output.exists() or output.is_symlink():
                if output.is_symlink():
                    raise RolloutControlError("terminal evidence path is unsafe")
                persisted, _persisted_raw = _load_terminal(
                    output,
                    issuance=issuance,
                    issuance_raw=issuance_raw,
                )
                if persisted != terminal:
                    raise RolloutControlError(
                        "live terminal rollout differs from persisted evidence"
                    )
            else:
                _write_new(output, terminal)
            return terminal
        if projection.get("status") == "PAUSED":
            raise RolloutControlError(
                "rollout was operator-paused before terminal evidence"
            )
        if monotonic() >= deadline:
            raise RolloutControlError("timed out waiting for terminal rollout evidence")
        sleeper(poll_seconds)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-origin", default=DEFAULT_API_ORIGIN)
    parser.add_argument("--api-cookie-jar", required=True, type=Path)
    parser.add_argument("--space-id", required=True, type=int)
    parser.add_argument("--device-id", required=True)
    commands = parser.add_subparsers(dest="command", required=True)

    register = commands.add_parser("register-release")
    register.add_argument("--bom", required=True, type=Path)
    register.add_argument("--signature", required=True, type=Path)
    register.add_argument("--output", required=True, type=Path)

    issue = commands.add_parser("issue")
    issue.add_argument("--release-evidence", required=True, type=Path)
    issue.add_argument("--purpose", choices=("rollback", "commit"), required=True)
    issue.add_argument("--client-request-id", required=True)
    issue.add_argument("--pre-commit-soak-seconds", type=int, default=30)
    issue.add_argument("--command-ttl-seconds", type=int, default=1800)
    issue.add_argument("--max-failure-percent", type=int, default=0)
    issue.add_argument("--previous-issuance", type=Path)
    issue.add_argument("--previous-terminal", type=Path)
    issue.add_argument("--adopt-rollout-id")
    issue.add_argument("--wait-seconds", type=int, default=180)
    issue.add_argument("--poll-seconds", type=float, default=5.0)
    issue.add_argument("--created-evidence", required=True, type=Path)
    issue.add_argument("--failure-evidence", type=Path)
    issue.add_argument("--output", required=True, type=Path)

    halt = commands.add_parser("halt")
    halt.add_argument("--release-evidence", required=True, type=Path)
    halt.add_argument("--created-evidence", required=True, type=Path)
    halt.add_argument("--purpose", choices=("rollback", "commit"), required=True)
    halt.add_argument("--pre-commit-soak-seconds", type=int, default=30)
    halt.add_argument("--command-ttl-seconds", type=int, default=1800)
    halt.add_argument("--max-failure-percent", type=int, default=0)
    halt.add_argument("--output", required=True, type=Path)

    terminal = commands.add_parser("wait-terminal")
    terminal.add_argument("--issuance-evidence", required=True, type=Path)
    terminal.add_argument("--release-evidence", required=True, type=Path)
    terminal.add_argument("--wait-seconds", type=int, default=600)
    terminal.add_argument("--poll-seconds", type=float, default=5.0)
    terminal.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    if not sys.flags.isolated:
        parser.error("this control client requires Python isolated mode (-I)")
    arguments = parser.parse_args(argv)
    try:
        if (
            isinstance(arguments.space_id, bool)
            or arguments.space_id < 1
            or DEVICE_ID_RE.fullmatch(arguments.device_id) is None
            or int(DEVICE_ID_RE.fullmatch(arguments.device_id).group(1))
            != arguments.space_id
        ):
            raise RolloutControlError("deviceId/spaceId binding is invalid")
        api = FleetApi(arguments.api_origin, arguments.api_cookie_jar)
        if arguments.command == "register-release":
            result = register_release(
                api,
                space_id=arguments.space_id,
                bom_path=arguments.bom,
                signature_path=arguments.signature,
                output=arguments.output,
            )
        elif arguments.command == "issue":
            result = issue_rollout(
                api,
                space_id=arguments.space_id,
                device_id=arguments.device_id,
                release_evidence_path=arguments.release_evidence,
                purpose=arguments.purpose,
                client_request_id=arguments.client_request_id,
                pre_commit_soak_seconds=arguments.pre_commit_soak_seconds,
                command_ttl_seconds=arguments.command_ttl_seconds,
                max_failure_percent=arguments.max_failure_percent,
                previous_issuance_path=arguments.previous_issuance,
                previous_terminal_path=arguments.previous_terminal,
                created_evidence_path=arguments.created_evidence,
                adopt_rollout_id=arguments.adopt_rollout_id,
                failure_evidence_path=arguments.failure_evidence,
                wait_seconds=arguments.wait_seconds,
                poll_seconds=arguments.poll_seconds,
                output=arguments.output,
            )
        elif arguments.command == "halt":
            result = halt_rollout(
                api,
                space_id=arguments.space_id,
                device_id=arguments.device_id,
                release_evidence_path=arguments.release_evidence,
                created_evidence_path=arguments.created_evidence,
                purpose=arguments.purpose,
                pre_commit_soak_seconds=arguments.pre_commit_soak_seconds,
                command_ttl_seconds=arguments.command_ttl_seconds,
                max_failure_percent=arguments.max_failure_percent,
                output=arguments.output,
            )
        else:
            result = wait_terminal(
                api,
                space_id=arguments.space_id,
                device_id=arguments.device_id,
                issuance_path=arguments.issuance_evidence,
                release_evidence_path=arguments.release_evidence,
                output=arguments.output,
                wait_seconds=arguments.wait_seconds,
                poll_seconds=arguments.poll_seconds,
            )
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    except (RolloutControlError, OSError, ValueError) as exc:
        print(f"run-iq9075-agent-rollout-control: {exc}", file=sys.stderr)
        return 1
    except Exception:  # noqa: BLE001 - never expose cookies/response bodies.
        print(
            "run-iq9075-agent-rollout-control: unexpected internal failure",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
