from __future__ import annotations

import json
import os
import socket
import struct
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from nuvion_updater.controller import UpdaterController
from nuvion_updater.errors import UpdaterError, UpdaterSecurityError

PROTOCOL_SCHEMA_VERSION = 1
MAX_REQUEST_BYTES = 256 * 1024
MAX_RESPONSE_BYTES = 256 * 1024
MAX_DETAIL_CHARS = 500
DEFAULT_REQUEST_DEADLINE_SECONDS = 10.0


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise UpdaterSecurityError(
                "INVALID_REQUEST", f"duplicate JSON member: {key}"
            )
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise UpdaterSecurityError(
        "INVALID_REQUEST", f"non-standard JSON constant: {value}"
    )


@dataclass(frozen=True)
class PeerCredentials:
    pid: int | None
    uid: int
    gid: int


def get_peer_credentials(connection: socket.socket) -> PeerCredentials:
    """Use the kernel-authenticated local peer identity (SO_PEERCRED on Linux)."""

    if hasattr(socket, "SO_PEERCRED"):
        raw = connection.getsockopt(
            socket.SOL_SOCKET,
            socket.SO_PEERCRED,
            struct.calcsize("3i"),
        )
        pid, uid, gid = struct.unpack("3i", raw)
        return PeerCredentials(pid=pid, uid=uid, gid=gid)
    getpeereid = getattr(connection, "getpeereid", None)
    if callable(getpeereid):
        uid, gid = getpeereid()
        return PeerCredentials(pid=None, uid=uid, gid=gid)
    if hasattr(socket, "LOCAL_PEERCRED"):
        # Darwin exposes struct xucred through SOL_LOCAL (numeric level 0),
        # but Python does not export SOL_LOCAL. uid/ngroups/groups[0] are fixed
        # width in the public xucred ABI.
        raw = connection.getsockopt(0, socket.LOCAL_PEERCRED, 128)
        if len(raw) >= 16:
            _version, uid, group_count = struct.unpack_from("=III", raw, 0)
            if group_count > 0:
                gid = struct.unpack_from("=I", raw, 12)[0]
                return PeerCredentials(pid=None, uid=uid, gid=gid)
    raise UpdaterSecurityError(
        "PEER_CREDENTIALS_UNAVAILABLE",
        "kernel peer credential verification is unavailable",
    )


class UpdaterProtocol:
    def __init__(self, controller: UpdaterController) -> None:
        self.controller = controller

    def dispatch(
        self,
        request: dict[str, Any],
        *,
        peer: PeerCredentials | None = None,
    ) -> dict[str, object]:
        if not isinstance(request, dict):
            raise UpdaterSecurityError("INVALID_REQUEST", "request must be an object")
        if request.get("schemaVersion") != PROTOCOL_SCHEMA_VERSION:
            raise UpdaterSecurityError(
                "INVALID_REQUEST", "unsupported protocol schemaVersion"
            )
        operation = request.get("operation")
        if not isinstance(operation, str):
            raise UpdaterSecurityError("INVALID_REQUEST", "operation is required")

        if operation == "AUTHORIZE_AND_STAGE":
            self._exact_fields(request, {"schemaVersion", "operation", "compactCommandJws"})
            compact_jws = request.get("compactCommandJws")
            if (
                not isinstance(compact_jws, str)
                or not compact_jws
                or len(compact_jws) > 192 * 1024
            ):
                raise UpdaterSecurityError(
                    "INVALID_REQUEST", "compactCommandJws is invalid"
                )
            return self.controller.authorize_and_stage(compact_jws).public_dict()

        if operation == "STATUS":
            allowed = {"schemaVersion", "operation"}
            if "commandId" in request:
                allowed.add("commandId")
            self._exact_fields(request, allowed)
            command_id = self._optional_command_id(request.get("commandId"))
            return self.controller.status(command_id)

        if operation == "BEGIN_COMMIT_GATE":
            self._exact_fields(
                request, {"schemaVersion", "operation", "commandId"}
            )
            command_id = self._command_id(request.get("commandId"))
            peer_pid = self._peer_pid(peer)
            return self.controller.begin_commit_gate(
                command_id, peer_pid=peer_pid
            ).public_dict()

        if operation == "COMMIT":
            self._exact_fields(
                request,
                {
                    "schemaVersion",
                    "operation",
                    "commandId",
                    "gateId",
                    "healthAttestationJws",
                },
            )
            command_id = self._command_id(request.get("commandId"))
            gate_id = self._command_id(request.get("gateId"))
            compact_jws = request.get("healthAttestationJws")
            if (
                not isinstance(compact_jws, str)
                or not compact_jws
                or len(compact_jws) > 32 * 1024
            ):
                raise UpdaterSecurityError(
                    "INVALID_REQUEST", "healthAttestationJws is invalid"
                )
            return self.controller.commit(
                command_id,
                gate_id=gate_id,
                health_attestation_jws=compact_jws,
                peer_pid=self._peer_pid(peer),
            ).public_dict()

        if operation in {"ACTIVATE", "ROLLBACK"}:
            allowed = {"schemaVersion", "operation", "commandId"}
            if operation == "ROLLBACK" and "reason" in request:
                allowed.add("reason")
            self._exact_fields(request, allowed)
            command_id = self._command_id(request.get("commandId"))
            if operation == "ACTIVATE":
                return self.controller.activate(command_id).public_dict()
            reason = self._detail(request.get("reason")) or "OPERATOR_REQUEST"
            return self.controller.rollback(command_id, reason=reason).public_dict()

        if operation in {"REPORT_BOOT_HEALTH", "REPORT_FUNCTIONAL_HEALTH"}:
            allowed = {"schemaVersion", "operation", "commandId", "healthy"}
            if "detail" in request:
                allowed.add("detail")
            self._exact_fields(request, allowed)
            command_id = self._command_id(request.get("commandId"))
            healthy = request.get("healthy")
            if not isinstance(healthy, bool):
                raise UpdaterSecurityError(
                    "INVALID_REQUEST", "healthy must be boolean"
                )
            detail = self._detail(request.get("detail"))
            if operation == "REPORT_BOOT_HEALTH":
                state = self.controller.report_boot_health(
                    command_id, healthy=healthy, detail=detail
                )
            else:
                state = self.controller.report_functional_health(
                    command_id, healthy=healthy, detail=detail
                )
            return state.public_dict()

        raise UpdaterSecurityError("UNSUPPORTED_OPERATION", "operation is not allowed")

    @staticmethod
    def _exact_fields(request: dict[str, Any], expected: set[str]) -> None:
        if set(request) != expected:
            raise UpdaterSecurityError(
                "INVALID_REQUEST", "request contains missing or unknown fields"
            )

    @staticmethod
    def _command_id(value: Any) -> str:
        if not isinstance(value, str):
            raise UpdaterSecurityError("INVALID_REQUEST", "commandId is required")
        try:
            normalized = str(uuid.UUID(value))
        except (ValueError, AttributeError) as exc:
            raise UpdaterSecurityError(
                "INVALID_REQUEST", "commandId must be a canonical UUID"
            ) from exc
        if normalized != value:
            raise UpdaterSecurityError(
                "INVALID_REQUEST", "commandId must be a canonical UUID"
            )
        return normalized

    @classmethod
    def _optional_command_id(cls, value: Any) -> str | None:
        return None if value is None else cls._command_id(value)

    @staticmethod
    def _detail(value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or value != value.strip() or not value:
            raise UpdaterSecurityError(
                "INVALID_REQUEST", "detail/reason must be trimmed text"
            )
        if len(value) > MAX_DETAIL_CHARS:
            raise UpdaterSecurityError(
                "INVALID_REQUEST", "detail/reason exceeds size limit"
            )
        return value

    @staticmethod
    def _peer_pid(peer: PeerCredentials | None) -> int:
        if peer is None or peer.pid is None or peer.pid < 1:
            raise UpdaterSecurityError(
                "PEER_PID_UNAVAILABLE",
                "Linux SO_PEERCRED pid is required for commit gating",
            )
        return peer.pid


class UpdaterUnixServer:
    def __init__(
        self,
        *,
        listener: socket.socket,
        protocol: UpdaterProtocol,
        allowed_uids: set[int],
        watchdog: Callable[[], object] | None = None,
        watchdog_interval_seconds: float = 1.0,
        request_deadline_seconds: float = DEFAULT_REQUEST_DEADLINE_SECONDS,
        monotonic_clock: Callable[[], float] | None = None,
    ) -> None:
        if not allowed_uids:
            raise ValueError("allowed_uids must not be empty")
        self.listener = listener
        self.protocol = protocol
        self.allowed_uids = frozenset(allowed_uids)
        if watchdog_interval_seconds <= 0 or watchdog_interval_seconds > 60:
            raise ValueError("watchdog interval must be in (0, 60] seconds")
        if request_deadline_seconds <= 0 or request_deadline_seconds > 60:
            raise ValueError("request deadline must be in (0, 60] seconds")
        self.watchdog = watchdog
        self.watchdog_interval_seconds = float(watchdog_interval_seconds)
        self.request_deadline_seconds = float(request_deadline_seconds)
        self.monotonic_clock = monotonic_clock or time.monotonic

    def serve_forever(self) -> None:
        next_watchdog = self.monotonic_clock() + self.watchdog_interval_seconds
        while True:
            now = self.monotonic_clock()
            if now >= next_watchdog:
                if self.watchdog is not None:
                    self.watchdog()
                next_watchdog = now + self.watchdog_interval_seconds
            self.listener.settimeout(max(0.05, next_watchdog - now))
            try:
                connection, _ = self.listener.accept()
            except TimeoutError:
                continue
            with connection:
                self.handle_connection(connection)

    def handle_connection(self, connection: socket.socket) -> None:
        try:
            peer = get_peer_credentials(connection)
            if peer.uid not in self.allowed_uids:
                raise UpdaterSecurityError(
                    "UNAUTHORIZED_PEER", "local peer is not authorized"
                )
            request = self._read_request(
                connection,
                deadline=self.monotonic_clock() + self.request_deadline_seconds,
            )
            # A candidate Agent may request a safety rollback only by reporting
            # a failed boot/functional health gate. The unauthenticated
            # operator rollback verb is root-only so compromised Agent code
            # cannot arbitrarily move the device to an older release.
            if request.get("operation") == "ROLLBACK" and peer.uid != 0:
                raise UpdaterSecurityError(
                    "OPERATOR_AUTH_REQUIRED",
                    "explicit rollback requires the root peer",
                )
            result = self.protocol.dispatch(request, peer=peer)
            response: dict[str, object] = {"ok": True, "result": result}
        except UpdaterError as exc:
            response = {
                "ok": False,
                "error": {"code": exc.code, "message": str(exc)[:500]},
            }
        except Exception:  # noqa: BLE001 - protocol never exposes traceback.
            response = {
                "ok": False,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "internal updater failure",
                },
            }
        payload = (
            json.dumps(response, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        if len(payload) > MAX_RESPONSE_BYTES:
            payload = b'{"error":{"code":"RESPONSE_TOO_LARGE","message":"response exceeds limit"},"ok":false}\n'
        try:
            connection.sendall(payload)
        except OSError:
            # Activation and IQ functional probes intentionally restart the
            # calling Agent. Its socket may disappear after the state change
            # was durably committed; recovery must happen on the next request,
            # not by crashing the privileged daemon.
            return

    def _read_request(
        self,
        connection: socket.socket,
        *,
        deadline: float,
    ) -> dict[str, Any]:
        chunks: list[bytes] = []
        total = 0
        while True:
            remaining = deadline - self.monotonic_clock()
            if remaining <= 0:
                raise UpdaterSecurityError(
                    "REQUEST_TIMEOUT", "request exceeded its absolute deadline"
                )
            connection.settimeout(remaining)
            try:
                chunk = connection.recv(
                    min(64 * 1024, MAX_REQUEST_BYTES + 1 - total)
                )
            except TimeoutError as exc:
                raise UpdaterSecurityError(
                    "REQUEST_TIMEOUT", "request exceeded its absolute deadline"
                ) from exc
            if not chunk:
                break
            newline = chunk.find(b"\n")
            if newline >= 0:
                chunks.append(chunk[:newline])
                total += newline
                if chunk[newline + 1 :]:
                    raise UpdaterSecurityError(
                        "INVALID_REQUEST", "only one request is allowed per connection"
                    )
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_REQUEST_BYTES:
                raise UpdaterSecurityError("REQUEST_TOO_LARGE", "request exceeds limit")
        raw = b"".join(chunks)
        if not raw:
            raise UpdaterSecurityError("INVALID_REQUEST", "empty request")
        try:
            request = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise UpdaterSecurityError(
                "INVALID_REQUEST", "request is not UTF-8 JSON"
            ) from exc
        if not isinstance(request, dict):
            raise UpdaterSecurityError("INVALID_REQUEST", "request must be an object")
        return request


def systemd_listener() -> socket.socket:
    listen_pid = int(os.environ.get("LISTEN_PID", "0") or "0")
    listen_fds = int(os.environ.get("LISTEN_FDS", "0") or "0")
    if listen_pid != os.getpid() or listen_fds != 1:
        raise UpdaterSecurityError(
            "SOCKET_ACTIVATION_REQUIRED", "exactly one systemd socket is required"
        )
    listener = socket.fromfd(3, socket.AF_UNIX, socket.SOCK_STREAM)
    listener.setblocking(True)
    return listener
