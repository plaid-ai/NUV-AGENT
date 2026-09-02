from __future__ import annotations

import json
import re
import socket
import stat
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

DEFAULT_UPDATER_SOCKET = Path("/run/nuvion-updater/control.sock")
MAX_RESPONSE_BYTES = 256 * 1024
_UPDATER_VERSION = re.compile(
    r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)
_ROOT_UPDATE_PHASE_TO_PUBLIC = {
    "AUTHORIZED": "STAGED",
    "DOWNLOADING": "STAGED",
    "STAGING": "STAGED",
    "VERIFIED": "STAGED",
    "ACTIVATING": "INSTALLING",
    "BOOT_HEALTHY": "VERIFYING",
    "FUNCTIONAL_HEALTHY": "VERIFYING",
    "ROLLING_BACK": "VERIFYING",
    "COMMITTED": "SUCCEEDED",
    "ROLLED_BACK": "ROLLED_BACK",
    "ROLLBACK_FAILED": "FAILED",
    "FAILED": "FAILED",
}
_UPDATE_EVIDENCE_KEYS = frozenset(
    {
        "commandId",
        "sequence",
        "targetVersion",
        "bomDigest",
        "phase",
        "updatePhase",
        "updatedAt",
        "candidateSlot",
        "previousSlot",
        "previousVersion",
        "releaseSequence",
        "artifactDigest",
        "componentSha",
        "configSchema",
        "bomVerificationStatus",
        "publisherKeyId",
        "healthDeadline",
        "errorCode",
        "message",
        "slot",
        "rollbackSlot",
        "rollbackVersion",
        "health",
        "functionalHealth",
    }
)


class UpdaterClientError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def authenticated_updater_version(status: Mapping[str, Any] | object) -> str:
    """Return live helper version evidence, otherwise the fail-closed sentinel."""

    if not isinstance(status, Mapping) or status.get("authenticatedHelper") is not True:
        return "unknown"
    value = status.get("updaterVersion")
    if not isinstance(value, str):
        return "unknown"
    normalized = value.strip()
    return normalized if _UPDATER_VERSION.fullmatch(normalized) else "unknown"


class UpdaterClient:
    """Bounded typed client; command payloads never control socket/path/URL."""

    def __init__(
        self,
        socket_path: str | Path = DEFAULT_UPDATER_SOCKET,
        *,
        timeout_seconds: float = 10.0,
        require_root_owner: bool = True,
        expected_peer_uid: int = 0,
    ) -> None:
        self.socket_path = Path(socket_path)
        self.timeout_seconds = timeout_seconds
        self.require_root_owner = require_root_owner
        self.expected_peer_uid = expected_peer_uid

    def capability_available(self) -> bool:
        return self.capability_status()["capabilityAvailable"] is True

    def capability_status(self) -> dict[str, Any]:
        try:
            status = self.status()
        except OSError:
            return {
                "capabilityAvailable": False,
                "authenticatedHelper": False,
                "reason": "UPDATER_UNAVAILABLE",
            }
        except UpdaterClientError as exc:
            return {
                "capabilityAvailable": False,
                "authenticatedHelper": False,
                "reason": exc.code,
            }
        available = status.get("capabilityAvailable") is True
        raw_updater_version = status.get("updaterVersion")
        updater_version = (
            raw_updater_version.strip()
            if isinstance(raw_updater_version, str)
            and _UPDATER_VERSION.fullmatch(raw_updater_version.strip())
            else None
        )
        if updater_version is None:
            available = False
        raw_update = status.get("update")
        update: dict[str, Any] | None = None
        if raw_update is not None:
            if not isinstance(raw_update, Mapping):
                available = False
            else:
                update = {
                    key: value
                    for key, value in raw_update.items()
                    if key in _UPDATE_EVIDENCE_KEYS
                    and (
                        value is None
                        or isinstance(value, (str, int, bool, float))
                    )
                }
        result: dict[str, Any] = {
            "capabilityAvailable": available,
            # This bit is constructed only after _request() has authenticated
            # the connected process with kernel peer credentials. It is not
            # copied from the helper-controlled response payload.
            "authenticatedHelper": True,
            "reason": str(
                "INVALID_UPDATER_VERSION"
                if updater_version is None
                else status.get("capabilityReason")
                or ("READY" if available else "UPDATER_NOT_READY")
            )[:100],
            "updaterVersion": updater_version or "unknown",
        }
        if raw_update is not None and update is None:
            result["reason"] = "INVALID_UPDATE_STATUS"
        elif update is not None:
            result["update"] = update
        return result

    def authorize_and_stage(self, compact_jws: str) -> dict[str, Any]:
        return self._request(
            {
                "schemaVersion": 1,
                "operation": "AUTHORIZE_AND_STAGE",
                "compactCommandJws": compact_jws,
            }
        )

    def status(self, command_id: str | None = None) -> dict[str, Any]:
        request: dict[str, Any] = {"schemaVersion": 1, "operation": "STATUS"}
        if command_id is not None:
            request["commandId"] = command_id
        return self._request(request)

    def activate(self, command_id: str) -> dict[str, Any]:
        return self._operation("ACTIVATE", command_id)

    def report_boot_health(
        self, command_id: str, *, healthy: bool, detail: str | None = None
    ) -> dict[str, Any]:
        return self._health("REPORT_BOOT_HEALTH", command_id, healthy, detail)

    def report_functional_health(
        self, command_id: str, *, healthy: bool, detail: str | None = None
    ) -> dict[str, Any]:
        return self._health("REPORT_FUNCTIONAL_HEALTH", command_id, healthy, detail)

    def begin_commit_gate(self, command_id: str) -> dict[str, Any]:
        return self._operation("BEGIN_COMMIT_GATE", command_id)

    def commit(
        self,
        command_id: str,
        *,
        gate_id: str,
        health_attestation_jws: str,
    ) -> dict[str, Any]:
        return self._request(
            {
                "schemaVersion": 1,
                "operation": "COMMIT",
                "commandId": command_id,
                "gateId": gate_id,
                "healthAttestationJws": health_attestation_jws,
            }
        )

    def rollback(self, command_id: str, *, reason: str | None = None) -> dict[str, Any]:
        request: dict[str, Any] = {
            "schemaVersion": 1,
            "operation": "ROLLBACK",
            "commandId": command_id,
        }
        if reason is not None:
            request["reason"] = reason
        return self._request(request)

    def _operation(self, operation: str, command_id: str) -> dict[str, Any]:
        return self._request(
            {
                "schemaVersion": 1,
                "operation": operation,
                "commandId": command_id,
            }
        )

    def _health(
        self,
        operation: str,
        command_id: str,
        healthy: bool,
        detail: str | None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "schemaVersion": 1,
            "operation": operation,
            "commandId": command_id,
            "healthy": healthy,
        }
        if detail is not None:
            request["detail"] = detail
        return self._request(request)

    def _request(self, request: dict[str, Any]) -> dict[str, Any]:
        self._validate_socket()
        payload = (
            json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        if len(payload) > 256 * 1024:
            raise UpdaterClientError("REQUEST_TOO_LARGE", "updater request exceeds limit")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(self.timeout_seconds)
            connection.connect(str(self.socket_path))
            self._validate_connected_peer(connection)
            connection.sendall(payload)
            connection.shutdown(socket.SHUT_WR)
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = connection.recv(min(64 * 1024, MAX_RESPONSE_BYTES + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > MAX_RESPONSE_BYTES:
                    raise UpdaterClientError(
                        "RESPONSE_TOO_LARGE", "updater response exceeds limit"
                    )
        try:
            response = json.loads(b"".join(chunks).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise UpdaterClientError(
                "INVALID_RESPONSE", "updater response is not UTF-8 JSON"
            ) from exc
        if not isinstance(response, dict) or response.get("ok") not in {True, False}:
            raise UpdaterClientError("INVALID_RESPONSE", "updater response is invalid")
        if response["ok"] is False:
            error = response.get("error")
            if not isinstance(error, dict):
                raise UpdaterClientError("INVALID_RESPONSE", "updater error is invalid")
            raise UpdaterClientError(
                str(error.get("code") or "UPDATER_ERROR")[:100],
                str(error.get("message") or "updater request failed")[:500],
            )
        result = response.get("result")
        if not isinstance(result, dict):
            raise UpdaterClientError("INVALID_RESPONSE", "updater result is invalid")
        return result

    def _validate_socket(self) -> None:
        try:
            metadata = self.socket_path.lstat()
        except OSError as exc:
            raise UpdaterClientError("UPDATER_UNAVAILABLE", "updater socket is unavailable") from exc
        if not stat.S_ISSOCK(metadata.st_mode):
            raise UpdaterClientError(
                "UNSAFE_UPDATER_SOCKET", "updater endpoint must be a Unix socket"
            )
        if self.require_root_owner and metadata.st_uid != 0:
            raise UpdaterClientError(
                "UNSAFE_UPDATER_SOCKET", "updater socket must be root-owned"
            )
        if metadata.st_mode & 0o002:
            raise UpdaterClientError(
                "UNSAFE_UPDATER_SOCKET", "updater socket must not be world-writable"
            )

    def _validate_connected_peer(self, connection: socket.socket) -> None:
        """Authenticate the connected process, closing the lstat/connect race."""

        uid: int | None = None
        if hasattr(socket, "SO_PEERCRED"):
            raw = connection.getsockopt(
                socket.SOL_SOCKET,
                socket.SO_PEERCRED,
                struct.calcsize("3i"),
            )
            _pid, uid, _gid = struct.unpack("3i", raw)
        else:
            getpeereid = getattr(connection, "getpeereid", None)
            if callable(getpeereid):
                uid, _gid = getpeereid()
            elif hasattr(socket, "LOCAL_PEERCRED"):
                raw = connection.getsockopt(0, socket.LOCAL_PEERCRED, 128)
                if len(raw) >= 12:
                    _version, uid, _group_count = struct.unpack_from("=III", raw, 0)
        if uid is None:
            raise UpdaterClientError(
                "PEER_CREDENTIALS_UNAVAILABLE",
                "kernel updater peer credentials are unavailable",
            )
        if uid != self.expected_peer_uid:
            raise UpdaterClientError(
                "UNSAFE_UPDATER_PEER",
                "connected updater process is not the trusted root peer",
            )


def build_updater_capability_telemetry(
    client: UpdaterClient | None = None,
) -> dict[str, Any]:
    """Fresh, fail-closed updater capability evidence for each heartbeat."""

    status = (client or UpdaterClient()).capability_status()
    updater_version = authenticated_updater_version(status)
    update = status.get("update")
    capability_evidence = dict(status)
    capability_evidence.pop("update", None)
    result: dict[str, Any] = {
        "agentUpdate": capability_evidence,
        # Static build metadata and BOM requirements are not evidence that the
        # root helper is installed or alive. Only a peer-authenticated STATUS
        # response may populate this rollout-eligibility field.
        "updaterVersion": updater_version,
    }
    if isinstance(update, Mapping):
        root_phase = str(update.get("phase") or update.get("updatePhase") or "")
        public_phase = _ROOT_UPDATE_PHASE_TO_PUBLIC.get(root_phase)
        if public_phase is None:
            result["updatePhase"] = "FAILED"
            result["updateEvidence"] = {
                **dict(update),
                "telemetryError": "UNKNOWN_ROOT_UPDATE_PHASE",
            }
        else:
            result["updatePhase"] = public_phase
            result["updateEvidence"] = dict(update)
    else:
        result["updatePhase"] = "IDLE"
    return result
