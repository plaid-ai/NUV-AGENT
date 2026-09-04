#!/usr/bin/env python3
"""Camera-independent IQ9075 CONFIG_APPLY/adaptive-streaming Fleet E2E."""

from __future__ import annotations

import argparse
import hashlib
import http.cookiejar
import importlib.util
import json
import os
import re
import shlex
import stat
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol


SCHEMA_VERSION = 1
KIND = "nuvion-iq9075-config-stream-e2e-evidence"
DEFAULT_API_BASE_URL = "https://api.nuvion-dev.plaidlabs.ai"
MAX_HTTP_BYTES = 2 * 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
COMPONENT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
DEVICE_ID_RE = re.compile(r"^sp-([1-9][0-9]*)-nuvion-[a-z0-9][a-z0-9-]{0,100}$")


def _load_fleet_runner() -> Any:
    path = Path(__file__).with_name("run-iq9075-fleet-e2e.py")
    spec = importlib.util.spec_from_file_location("nuvion_iq9075_fleet_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Fleet runner module is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FLEET = _load_fleet_runner()


class ConfigStreamError(RuntimeError):
    """Stable error that must not include credentials or response bodies."""


def canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def settings_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class IssuedCommand:
    command_id: str
    sequence: int
    command_type: str


class BoardPort(Protocol):
    def prepare(self, *, run_id: str, manifest_sha256: str) -> dict[str, Any]: ...

    def set_link(self, *, run_id: str, quality: str) -> dict[str, Any]: ...

    def inspect(self, *, run_id: str, command_id: str) -> dict[str, Any]: ...

    def restore(self, *, run_id: str) -> dict[str, Any]: ...

    def recover_after_reboot(self, *, run_id: str) -> dict[str, Any]: ...


class ApiPort(Protocol):
    def issue(
        self,
        *,
        space_id: int,
        device_id: str,
        command_type: str,
        payload: Mapping[str, Any],
        desired_state: Mapping[str, Any],
    ) -> IssuedCommand: ...

    def projection(self, *, space_id: int, device_id: str) -> dict[str, Any]: ...

    def commands(self, *, space_id: int, device_id: str) -> list[dict[str, Any]]: ...


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        raise ConfigStreamError("Fleet API redirects are not allowed")


class FleetApi:
    def __init__(
        self,
        base_url: str,
        access_token: bytearray | None = None,
        *,
        cookie_jar: str | Path | None = None,
        opener: Any | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        parsed = urllib.parse.urlsplit(base_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
            or base_url.rstrip("/") != DEFAULT_API_BASE_URL
        ):
            raise ConfigStreamError(
                "Fleet API origin must be the authoritative Nuvion dev API"
            )
        if (access_token is None) == (cookie_jar is None):
            raise ConfigStreamError("exactly one Fleet API credential source is required")
        token: str | None = None
        if access_token is not None:
            try:
                token = bytes(access_token).decode("ascii")
            except UnicodeDecodeError as exc:
                raise ConfigStreamError("Fleet API token is not ASCII") from exc
            if not token or token != token.strip() or any(ord(ch) < 0x21 for ch in token):
                raise ConfigStreamError("Fleet API token is invalid")
        self.base_url = base_url.rstrip("/")
        self._token = token
        if cookie_jar is not None:
            jar_candidate = Path(cookie_jar).expanduser()
            if jar_candidate.is_symlink():
                raise ConfigStreamError("Fleet API cookie jar is unsafe")
            jar_path = jar_candidate.resolve()
            metadata = jar_path.lstat()
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_size > 64 * 1024
            ):
                raise ConfigStreamError("Fleet API cookie jar is unsafe")
            jar = http.cookiejar.MozillaCookieJar(str(jar_path))
            try:
                jar.load(ignore_discard=True, ignore_expires=False)
            except (OSError, http.cookiejar.LoadError) as exc:
                raise ConfigStreamError("Fleet API cookie jar is invalid") from exc
            if not any(
                cookie.domain.lstrip(".") == parsed.hostname
                and cookie.path == "/"
                and cookie.secure
                for cookie in jar
            ):
                raise ConfigStreamError("Fleet API cookie jar has no origin-bound secure cookie")
            if any(cookie.domain.lstrip(".") != parsed.hostname for cookie in jar):
                raise ConfigStreamError("Fleet API cookie jar contains another origin")
            self._opener = opener or urllib.request.build_opener(
                _NoRedirect(), urllib.request.HTTPCookieProcessor(jar)
            )
        else:
            self._opener = opener or urllib.request.build_opener(_NoRedirect())
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    @staticmethod
    def _strict_document(raw: bytes) -> dict[str, Any]:
        if len(raw) > MAX_HTTP_BYTES:
            raise ConfigStreamError("Fleet API response exceeds size limit")

        def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate member")
                result[key] = value
            return result

        try:
            value = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=unique,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(value)
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ConfigStreamError("Fleet API returned invalid JSON") from exc
        if not isinstance(value, dict) or "data" not in value:
            raise ConfigStreamError("Fleet API envelope is invalid")
        return value

    def _request(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: Mapping[str, Any] | None = None,
    ) -> Any:
        if not path.startswith("/") or "//" in path or ".." in path:
            raise ConfigStreamError("Fleet API path is invalid")
        body = canonical_json(payload) if payload is not None else None
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self._token is not None:
            headers["Authorization"] = "Bearer " + self._token
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            method=method,
            headers=headers,
        )
        try:
            with self._opener.open(request, timeout=15) as response:
                length = response.headers.get("Content-Length")
                if length is not None and int(length) > MAX_HTTP_BYTES:
                    raise ConfigStreamError("Fleet API response exceeds size limit")
                raw = response.read(MAX_HTTP_BYTES + 1)
        except ConfigStreamError:
            raise
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as exc:
            raise ConfigStreamError("Fleet API request failed") from exc
        return self._strict_document(raw)["data"]

    def issue(
        self,
        *,
        space_id: int,
        device_id: str,
        command_type: str,
        payload: Mapping[str, Any],
        desired_state: Mapping[str, Any],
    ) -> IssuedCommand:
        expires_at = (self._clock() + timedelta(seconds=120)).isoformat(
            timespec="milliseconds"
        ).replace("+00:00", "Z")
        data = self._request(
            f"/spaces/{space_id}/devices/{urllib.parse.quote(device_id, safe='')}/commands",
            method="POST",
            payload={
                "type": command_type,
                "schemaVersion": 1,
                "expiresAt": expires_at,
                "payload": dict(payload),
                "desiredState": dict(desired_state),
            },
        )
        if not isinstance(data, Mapping):
            raise ConfigStreamError("Fleet command issue response is invalid")
        try:
            command_id = str(uuid.UUID(str(data.get("commandId") or "")))
        except ValueError as exc:
            raise ConfigStreamError("Fleet commandId is invalid") from exc
        sequence = data.get("sequence")
        if (
            command_id != data.get("commandId")
            or type(sequence) is not int
            or sequence < 1
            or data.get("type") != command_type
            or data.get("status") != "QUEUED"
        ):
            raise ConfigStreamError("Fleet command issue identity is invalid")
        return IssuedCommand(command_id, sequence, command_type)

    def projection(self, *, space_id: int, device_id: str) -> dict[str, Any]:
        data = self._request(
            f"/spaces/{space_id}/devices/{urllib.parse.quote(device_id, safe='')}/fleet-runtime"
        )
        if not isinstance(data, dict):
            raise ConfigStreamError("Fleet projection response is invalid")
        return data

    def commands(self, *, space_id: int, device_id: str) -> list[dict[str, Any]]:
        data = self._request(
            f"/spaces/{space_id}/devices/{urllib.parse.quote(device_id, safe='')}/commands?limit=100"
        )
        if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
            raise ConfigStreamError("Fleet command journal response is invalid")
        return data


BOARD_PROGRAM = r'''
import base64
import fcntl
import hashlib
import json
import os
import pwd
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

MAX_FILE = 64 * 1024 * 1024
MAX_MODEL_FILE = 8 * 1024 * 1024 * 1024
MAX_PROCESS_ENV = 256 * 1024
DEADMAN_SECONDS = 30 * 60
DEADMAN_RECOVERY_SECONDS = 5 * 60
FLEET_PROTOCOL = "iq9075-fleet-e2e-v2"
RUN_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z")
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
COMPONENT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
COMMAND_ID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z")
CONFIG = Path("/etc/nuv-agent/agent.env")
COMMAND_DB = Path("/var/lib/nuv-agent/commands.sqlite3")
SETTINGS = Path("/var/lib/nuv-agent/settings")
ACTIVE_RUN = Path("/var/lib/nuvion-fleet-e2e/active-run.json")
CONFIG_LEASE = Path("/var/lib/nuvion-fleet-e2e/config-stream-active.json")
GLOBAL_LOCK = Path("/run/lock/nuvion-fleet-e2e.lock")
INSTALL_ROOT = Path("/opt/nuv-agent")
FIXED = (
    CONFIG,
    COMMAND_DB,
    Path(str(COMMAND_DB) + "-journal"),
    Path(str(COMMAND_DB) + "-wal"),
    Path(str(COMMAND_DB) + "-shm"),
    SETTINGS / "active.env",
    SETTINGS / "candidate.env",
    SETTINGS / "lkg.env",
    SETTINGS / "restart-marker.json",
)
UPDATES = {
    "NUVION_FLEET_COMMAND_ENABLED": "true",
    "NUVION_GST_SOURCE": "videotestsrc is-live=true pattern=smpte",
    "NUVION_FLEET_COMMAND_POLL_INTERVAL_SEC": "1",
    "NUVION_FLEET_EFFECT_RECONCILE_INTERVAL_SEC": "0.25",
    "NUVION_FLEET_OBSERVATION_REPLAY_INTERVAL_SEC": "0.5",
    "NUVION_CONNECTIVITY_INTERVAL_SEC": "1",
    "NUVION_CONNECTIVITY_MIN_SEND_INTERVAL_SEC": "1",
    "NUVION_WIFI_INTERFACE": "fleet0",
}

class Failure(RuntimeError):
    pass

def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")

def sha(payload):
    return hashlib.sha256(payload).hexdigest()

def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

def read_regular(path, maximum=MAX_FILE):
    meta = path.lstat()
    if stat.S_ISLNK(meta.st_mode) or not stat.S_ISREG(meta.st_mode) or meta.st_size > maximum:
        raise Failure("unsafe Fleet E2E file")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != (meta.st_dev, meta.st_ino):
            raise Failure("Fleet E2E file changed while opening")
        chunks = []
        total = 0
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise Failure("Fleet E2E file exceeds bound")
            chunks.append(chunk)
        after = os.fstat(fd)
        if (after.st_dev, after.st_ino, after.st_size) != (opened.st_dev, opened.st_ino, opened.st_size):
            raise Failure("Fleet E2E file changed while reading")
        return b"".join(chunks), opened
    finally:
        os.close(fd)

def sha_regular(path, maximum):
    meta = path.lstat()
    if stat.S_ISLNK(meta.st_mode) or not stat.S_ISREG(meta.st_mode) or meta.st_size > maximum:
        raise Failure("unsafe model artifact")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    total = 0
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (meta.st_dev, meta.st_ino, meta.st_size):
            raise Failure("model artifact changed while opening")
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise Failure("model artifact exceeds bound")
            digest.update(chunk)
        after = os.fstat(fd)
        if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns):
            raise Failure("model artifact changed while hashing")
        return digest.hexdigest()
    finally:
        os.close(fd)

def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def atomic(path, payload, mode=0o600, uid=0, gid=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        meta = path.lstat()
        if stat.S_ISLNK(meta.st_mode) or not stat.S_ISREG(meta.st_mode):
            raise Failure("unsafe Fleet E2E destination")
    temp = path.parent / ("." + path.name + ".tmp-" + uuid.uuid4().hex)
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        view = memoryview(payload)
        while view:
            count = os.write(fd, view)
            if count <= 0:
                raise Failure("Fleet E2E write made no progress")
            view = view[count:]
        os.fchown(fd, uid, gid)
        os.fchmod(fd, mode)
        os.fsync(fd)
        os.replace(temp, path)
        fsync_dir(path.parent)
    finally:
        os.close(fd)
        try:
            temp.unlink()
        except FileNotFoundError:
            pass

def load_json(path):
    raw, _ = read_regular(path, 2 * 1024 * 1024)
    value = json.loads(raw.decode())
    if not isinstance(value, dict):
        raise Failure("Fleet E2E state is invalid")
    return value

def run_id(value):
    normalized = str(uuid.UUID(value))
    if normalized != value or RUN_RE.fullmatch(value) is None:
        raise Failure("runId is invalid")
    return value

def work_paths(value):
    rid = run_id(value)
    run_root = Path("/var/lib/nuvion-fleet-e2e/runs") / rid
    work = run_root / "config-stream-e2e"
    runtime = Path("/run/nuvion-config-stream-e2e") / rid
    dropin = Path("/run/systemd/system/nuv-agent.service.d") / ("90-nuvion-config-stream-" + rid + ".conf")
    return run_root, work, runtime, dropin

def directory_snapshot(path):
    if not path.exists() and not path.is_symlink():
        return {"exists": False, "mode": None, "uid": None, "gid": None}
    meta = path.lstat()
    if stat.S_ISLNK(meta.st_mode) or not stat.S_ISDIR(meta.st_mode) or meta.st_uid != 0 or meta.st_mode & 0o022:
        raise Failure("config-stream parent ownership or mode is unsafe")
    return {"exists": True, "mode": stat.S_IMODE(meta.st_mode), "uid": meta.st_uid, "gid": meta.st_gid}

def ensure_directory_from_snapshot(path, snapshot):
    if set(snapshot) != {"exists", "mode", "uid", "gid"} or type(snapshot.get("exists")) is not bool:
        raise Failure("config-stream parent snapshot is invalid")
    if snapshot["exists"]:
        if directory_snapshot(path) != snapshot:
            raise Failure("config-stream parent metadata changed")
        return
    if path.exists() or path.is_symlink():
        raise Failure("config-stream parent appeared before creation")
    parent_meta = path.parent.lstat()
    if stat.S_ISLNK(parent_meta.st_mode) or not stat.S_ISDIR(parent_meta.st_mode) or parent_meta.st_uid != 0 or parent_meta.st_mode & 0o022:
        raise Failure("config-stream parent ancestor is unsafe")
    path.mkdir(mode=0o755)
    os.chown(path, 0, 0)
    os.chmod(path, 0o755)
    fsync_dir(path.parent)

def restore_parent(path, snapshot):
    if snapshot.get("exists") is True:
        if directory_snapshot(path) != snapshot:
            raise Failure("config-stream parent metadata was not preserved")
        return
    if snapshot.get("exists") is not False:
        raise Failure("config-stream parent snapshot is invalid")
    if not path.exists() and not path.is_symlink():
        return
    meta = path.lstat()
    if stat.S_ISLNK(meta.st_mode) or not stat.S_ISDIR(meta.st_mode) or meta.st_uid != 0:
        raise Failure("config-stream parent restore path is unsafe")
    try:
        path.rmdir()
    except OSError as exc:
        raise Failure("config-stream parent is not empty after cleanup") from exc
    fsync_dir(path.parent)

def validate_active_run(rid):
    payload, meta = read_regular(ACTIVE_RUN, 4096)
    if meta.st_uid != 0 or stat.S_IMODE(meta.st_mode) != 0o600:
        raise Failure("Fleet E2E active-run lease metadata is unsafe")
    lease = json.loads(payload.decode("utf-8"))
    if lease != {"schemaVersion": 1, "protocolVersion": FLEET_PROTOCOL, "runId": rid}:
        raise Failure("Fleet E2E active-run lease belongs to another run")

@contextmanager
def operation_lock(rid):
    validate_active_run(rid)
    lock_parent = GLOBAL_LOCK.parent.lstat()
    if stat.S_ISLNK(lock_parent.st_mode) or not stat.S_ISDIR(lock_parent.st_mode) or lock_parent.st_uid != 0 or lock_parent.st_mode & 0o022:
        raise Failure("Fleet E2E lock directory is unsafe")
    descriptor = os.open(GLOBAL_LOCK, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_uid != 0 or stat.S_IMODE(opened.st_mode) != 0o600:
            raise Failure("Fleet E2E global lock endpoint is unsafe")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Failure("another Fleet E2E operation owns the global lock") from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

def claim_config_lease(rid):
    parent = CONFIG_LEASE.parent.lstat()
    if stat.S_ISLNK(parent.st_mode) or not stat.S_ISDIR(parent.st_mode) or parent.st_uid != 0 or parent.st_mode & 0o022:
        raise Failure("config-stream lease directory is unsafe")
    payload = canonical({"schemaVersion": 1, "runId": rid})
    descriptor = os.open(CONFIG_LEASE, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        os.fchown(descriptor, 0, 0)
        os.fchmod(descriptor, 0o600)
        view = memoryview(payload)
        while view:
            count = os.write(descriptor, view)
            if count <= 0:
                raise Failure("config-stream lease write made no progress")
            view = view[count:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    fsync_dir(CONFIG_LEASE.parent)

def validate_config_lease(rid):
    payload, meta = read_regular(CONFIG_LEASE, 4096)
    if meta.st_uid != 0 or stat.S_IMODE(meta.st_mode) != 0o600 or json.loads(payload.decode("utf-8")) != {"schemaVersion": 1, "runId": rid}:
        raise Failure("config-stream exclusive lease is unsafe or belongs to another run")

def release_config_lease(rid):
    if not CONFIG_LEASE.exists() and not CONFIG_LEASE.is_symlink():
        return
    validate_config_lease(rid)
    CONFIG_LEASE.unlink()
    fsync_dir(CONFIG_LEASE.parent)

def deadman_unit(rid):
    return "nuvion-config-stream-deadman-" + rid.replace("-", "") + ".service"

def deadman_timer(rid):
    return deadman_unit(rid).removesuffix(".service") + ".timer"

def program_source():
    arguments = list(getattr(sys, "orig_argv", ()))
    try:
        index = arguments.index("-c")
        source = arguments[index + 1]
    except (ValueError, IndexError) as exc:
        raise Failure("config-stream recovery source is unavailable") from exc
    if not isinstance(source, str) or not source.startswith("\nimport base64\n") or len(source.encode("utf-8")) > 512 * 1024:
        raise Failure("config-stream recovery source is invalid")
    return source

def arm_deadman(rid):
    result = subprocess.run(
        [
            "/usr/bin/systemd-run",
            "--unit=" + deadman_unit(rid),
            "--collect",
            "--on-active=" + str(DEADMAN_SECONDS) + "s",
            "--timer-property=AccuracySec=1s",
            "--property=Type=oneshot",
            "--property=LimitCORE=0",
            "--property=StandardOutput=null",
            "--property=StandardError=journal",
            "--property=RuntimeMaxSec=" + str(DEADMAN_RECOVERY_SECONDS) + "s",
            "--property=TimeoutStartSec=" + str(DEADMAN_RECOVERY_SECONDS) + "s",
            "--property=TimeoutStopSec=" + str(DEADMAN_RECOVERY_SECONDS) + "s",
            "--property=Restart=on-failure",
            "--property=RestartSec=15s",
            "--property=StartLimitIntervalSec=0",
            "/usr/bin/python3",
            "-I",
            "-c",
            program_source(),
            "deadman-restore",
            rid,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=30,
        check=False,
    )
    if result.returncode != 0 or systemctl("is-active", "--quiet", deadman_timer(rid), check=False).returncode != 0:
        raise Failure("config-stream recovery deadman did not arm")

def disarm_deadman(rid):
    for target in (deadman_timer(rid), deadman_unit(rid)):
        systemctl("stop", target, check=False)
        if systemctl("is-active", "--quiet", target, check=False).returncode == 0:
            raise Failure("config-stream recovery deadman remains active")
        systemctl("reset-failed", target, check=False)

def systemctl(*args, check=True):
    result = subprocess.run(["/usr/bin/systemctl", *args], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=60, check=False, text=True)
    if check and result.returncode != 0:
        raise Failure("systemd operation failed")
    return result

def service_active():
    return systemctl("is-active", "--quiet", "nuv-agent.service", check=False).returncode == 0

def service_pid():
    result = systemctl("show", "--property=MainPID", "--value", "nuv-agent.service")
    value = result.stdout.strip()
    if not value.isdigit() or int(value) < 1:
        raise Failure("Agent service PID is unavailable")
    return int(value)

def runtime_identity(run_root, manifest_sha):
    manifest_path = run_root / "trust-transaction/immutable-manifest.json"
    manifest_raw, manifest_meta = read_regular(manifest_path, 64 * 1024)
    if manifest_meta.st_uid != 0 or stat.S_IMODE(manifest_meta.st_mode) != 0o600 or sha(manifest_raw) != manifest_sha:
        raise Failure("immutable Fleet manifest metadata or digest is invalid")
    manifest = json.loads(manifest_raw.decode("utf-8"))
    scenario = manifest.get("scenario") if isinstance(manifest, dict) else None
    release = scenario.get("release") if isinstance(scenario, dict) else None
    bom = scenario.get("expectedBomDigest") if isinstance(scenario, dict) else None
    if not isinstance(scenario, dict) or scenario.get("type") != "commit" or not isinstance(release, dict) or not isinstance(bom, str) or DIGEST_RE.fullmatch(bom) is None:
        raise Failure("config-stream requires an immutable commit manifest")
    if set(release) != {"agentVersion", "releaseSequence", "artifactDigest", "componentSha", "configSchema", "publisherKeyId"}:
        raise Failure("commit manifest release identity fields are invalid")
    expected_slot = "releases/" + bom.removeprefix("sha256:")
    expected_release = {"schemaVersion": 2, "bomDigest": bom, **release}
    current = INSTALL_ROOT / "current"
    current_meta = current.lstat()
    if not stat.S_ISLNK(current_meta.st_mode) or current_meta.st_uid != 0 or os.readlink(current) != expected_slot:
        raise Failure("active release slot differs from the commit manifest")
    slot = INSTALL_ROOT / expected_slot
    slot_meta = slot.lstat()
    if stat.S_ISLNK(slot_meta.st_mode) or not stat.S_ISDIR(slot_meta.st_mode) or slot_meta.st_uid != 0 or slot_meta.st_mode & 0o022:
        raise Failure("active release slot metadata is unsafe")
    marker_path = slot / ".nuvion/release.json"
    marker_raw, marker_meta = read_regular(marker_path, 64 * 1024)
    marker = json.loads(marker_raw.decode("utf-8"))
    expected_marker_raw = canonical(expected_release)
    if marker_meta.st_uid != 0 or marker_meta.st_mode & 0o022 or marker_raw != expected_marker_raw or marker != expected_release:
        raise Failure("active release marker differs from the commit manifest")
    build_info_raw, build_info_meta = read_regular(slot / "venv/lib/python3.12/site-packages/nuvion_app/build_info.py", 64 * 1024)
    expected_build_info_raw = (
        '"""Generated release identity. Do not edit in release artifacts."""\n\n'
        'AGENT_VERSION = "' + str(release.get("agentVersion")) + '"\n'
        'COMPONENT_SHA = "' + str(release.get("componentSha")) + '"\n'
    ).encode("utf-8")
    if build_info_meta.st_uid != 0 or build_info_meta.st_mode & 0o022 or build_info_raw != expected_build_info_raw:
        raise Failure("active Agent build info differs from the commit manifest")
    pid = service_pid()
    environment_raw, _ = read_regular(Path("/proc") / str(pid) / "environ", MAX_PROCESS_ENV)
    selected = {}
    for entry in environment_raw.split(b"\0"):
        if b"=" not in entry:
            continue
        key, value = entry.split(b"=", 1)
        if key in {b"NUVION_ACTIVE_SLOT", b"NUVION_EXPECTED_BOM_DIGEST"}:
            selected[key.decode("ascii")] = value.decode("ascii")
    if selected != {"NUVION_ACTIVE_SLOT": expected_slot, "NUVION_EXPECTED_BOM_DIGEST": bom}:
        raise Failure("running Agent process is not bound to the active release")
    current_after = current.lstat()
    if (current_after.st_dev, current_after.st_ino) != (current_meta.st_dev, current_meta.st_ino) or os.readlink(current) != expected_slot:
        raise Failure("active release slot changed during identity inspection")
    return {
        "activeSlot": expected_slot,
        "processActiveSlot": selected["NUVION_ACTIVE_SLOT"],
        "processExpectedBomDigest": selected["NUVION_EXPECTED_BOM_DIGEST"],
        "servicePid": pid,
        "releaseMarkerSha256": sha(marker_raw),
        "buildInfoSha256": sha(build_info_raw),
        "release": expected_release,
    }

def parse_env(raw):
    result = {}
    for line in raw.decode("utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        result[key] = value
    return result

def render_updates(raw):
    lines = raw.decode("utf-8").splitlines()
    seen = set()
    rendered = []
    for line in lines:
        stripped = line.lstrip()
        key = stripped.split("=", 1)[0].strip() if stripped and not stripped.startswith("#") and "=" in stripped else ""
        if key in UPDATES:
            if key not in seen:
                rendered.append(key + "=" + UPDATES[key])
                seen.add(key)
        else:
            rendered.append(line)
    for key, value in UPDATES.items():
        if key not in seen:
            rendered.append(key + "=" + value)
    return ("\n".join(rendered) + "\n").encode()

def decode_labels(value, fallback):
    if value:
        padding = "=" * ((4 - len(value) % 4) % 4)
        labels = json.loads(base64.b64decode(value + padding, altchars=b"-_", validate=True).decode())
    else:
        labels = [item.strip() for item in fallback.split(",") if item.strip()]
    if not isinstance(labels, list) or any(not isinstance(item, str) or not item or item != item.strip() for item in labels):
        raise Failure("active label baseline is unavailable")
    return labels

def resolved_model(values):
    user_home = Path(pwd.getpwnam("nuvion").pw_dir)
    pointer = values.get("NUVION_MODEL_POINTER", "anomalyclip/prod")
    explicit = values.get("NUVION_MODEL_LOCAL_DIR", "")
    if explicit:
        model_dir = Path(explicit.replace("~", str(user_home), 1)) if explicit.startswith("~") else Path(explicit)
    else:
        root = values.get("NUVION_MODEL_DIR", "~/.cache/nuvion/models")
        root_path = Path(root.replace("~", str(user_home), 1)) if root.startswith("~") else Path(root)
        profile = values.get("NUVION_MODEL_PROFILE", "runtime").lower()
        identifier = ("server:" + pointer + ":" + profile).replace("/", "__").replace(":", "_")
        model_dir = root_path / identifier
    root = model_dir.resolve(strict=True)
    presign = json.loads((root / "metadata/server_presign_response.json").read_text(encoding="utf-8"))
    if isinstance(presign, dict) and isinstance(presign.get("data"), dict):
        presign = presign["data"]
    if not isinstance(presign, dict) or presign.get("pointer") != pointer:
        raise Failure("active model pointer cannot be proven")
    downloaded = json.loads((root / "metadata/downloaded_from_server.json").read_text(encoding="utf-8"))
    if not isinstance(downloaded, list) or not downloaded:
        raise Failure("active model manifest cannot be proven")
    entries = []
    manifest_digest = None
    for item in downloaded:
        if not isinstance(item, dict):
            raise Failure("active model manifest is invalid")
        key = str(item.get("key") or "")
        expected = str(item.get("sha256") or "").lower()
        destination = Path(str(item.get("dst") or ""))
        if not key or SHA_RE.fullmatch(expected) is None or not destination.is_absolute() or destination.is_symlink():
            raise Failure("active model artifact identity is invalid")
        resolved = destination.resolve(strict=True)
        resolved.relative_to(root)
        actual = sha_regular(resolved, MAX_MODEL_FILE)
        if actual != expected:
            raise Failure("active model artifact digest mismatch")
        entries.append((key, actual))
        if key == "manifest":
            manifest_digest = "sha256:" + actual
    aggregate = "sha256:" + sha("".join(key + ":" + digest + "\n" for key, digest in sorted(entries)).encode())
    configured = values.get("NUVION_MODEL_DIGEST", "")
    digest = configured if configured in {manifest_digest, aggregate} else (manifest_digest or aggregate)
    return {"pointer": pointer, "digest": digest}

def baseline(verify_artifact=True, prior_model=None):
    config_raw, _ = read_regular(CONFIG, 2 * 1024 * 1024)
    values = parse_env(config_raw)
    active = SETTINGS / "active.env"
    if active.exists():
        overlay_raw, _ = read_regular(active, 64 * 1024)
        values.update(parse_env(overlay_raw))
    def integer(name, default, minimum, maximum):
        raw = values.get(name, str(default))
        if not raw.isdigit() or not minimum <= int(raw) <= maximum:
            raise Failure("active setting baseline is invalid")
        return int(raw)
    clip_enabled = values.get("NUVION_CLIP_ENABLED", "true").lower()
    if clip_enabled not in {"true", "false"}:
        raise Failure("active clip baseline is invalid")
    pointer = values.get("NUVION_MODEL_POINTER", "anomalyclip/prod") or "anomalyclip/prod"
    configured_digest = values.get("NUVION_MODEL_DIGEST", "") or None
    model = {
        "pointer": pointer,
        "configuredDigest": configured_digest,
        "artifactDigest": None,
        "artifactVerified": False,
        "runtimeEnabled": values.get("NUVION_ZERO_SHOT_ENABLED", "true").lower() in {"1", "true", "yes", "on"},
        "runtimeBackend": (values.get("NUVION_ZSAD_BACKEND", "triton") or "triton").lower(),
    }
    if verify_artifact and model["runtimeEnabled"] and model["runtimeBackend"] in {"siglip", "mps"}:
        try:
            verified = resolved_model(values)
        except (Failure, OSError, ValueError, KeyError, json.JSONDecodeError):
            verified = None
        if isinstance(verified, dict):
            model["artifactDigest"] = verified["digest"]
            model["artifactVerified"] = True
    elif isinstance(prior_model, dict):
        model["artifactDigest"] = prior_model.get("artifactDigest")
        model["artifactVerified"] = prior_model.get("artifactVerified") is True
    return {
        "model": model,
        "labels": {
            "inspection": decode_labels(values.get("NUVION_ZERO_SHOT_LABELS_B64", ""), values.get("NUVION_ZERO_SHOT_LABELS", "normal,defect")),
            "anomaly": decode_labels(values.get("NUVION_ZERO_SHOT_ANOMALY_LABELS_B64", ""), values.get("NUVION_ZERO_SHOT_ANOMALY_LABELS", "defect")),
        },
        "clip": {
            "enabled": clip_enabled == "true",
            "preSeconds": integer("NUVION_CLIP_PRE_SEC", 5, 0, 60),
            "postSeconds": integer("NUVION_CLIP_POST_SEC", 5, 0, 300),
        },
        "video": {
            "width": integer("NUVION_VIDEO_WIDTH", 640, 160, 7680),
            "height": integer("NUVION_VIDEO_HEIGHT", 480, 120, 4320),
            "fps": integer("NUVION_VIDEO_FPS", 30, 1, 120),
            "bitrateKbps": integer("NUVION_VIDEO_BITRATE_KBPS", 1000, 100, 20000),
        },
    }

def db_counts():
    connection = sqlite3.connect("file:" + str(COMMAND_DB) + "?mode=ro", uri=True, timeout=5)
    try:
        inbox = int(connection.execute("SELECT COUNT(*) FROM command_inbox WHERE status IN ('RECEIVED','IN_PROGRESS')").fetchone()[0])
        observations = int(connection.execute("SELECT COUNT(*) FROM fleet_command_observation WHERE acknowledged_at IS NULL").fetchone()[0])
        reserved = int(connection.execute("SELECT COALESCE(SUM(slots),0) FROM fleet_command_observation_reservation").fetchone()[0])
        dlq = int(connection.execute("SELECT COUNT(*) FROM fleet_command_observation_dlq").fetchone()[0])
        return {"inboxPendingRows": inbox, "observationPendingRows": observations, "observationReservedRows": reserved, "observationDlqRows": dlq}
    finally:
        connection.close()

def save_snapshots(work):
    before = work / "before"
    before.mkdir(parents=True, exist_ok=False)
    os.chown(before, 0, 0)
    os.chmod(before, 0o700)
    fsync_dir(work)
    records = []
    for index, path in enumerate(FIXED):
        if not path.exists() and not path.is_symlink():
            records.append({"path": str(path), "exists": False, "sha256": None, "mode": None, "uid": None, "gid": None, "snapshot": None})
            continue
        payload, meta = read_regular(path)
        target = before / str(index)
        atomic(target, payload, 0o600, 0, 0)
        records.append({"path": str(path), "exists": True, "sha256": sha(payload), "mode": stat.S_IMODE(meta.st_mode), "uid": meta.st_uid, "gid": meta.st_gid, "snapshot": str(index)})
    return records

def restore_snapshots(work, records):
    before = work / "before"
    for record in records:
        path = Path(record["path"])
        if path not in FIXED:
            raise Failure("snapshot path is outside allowlist")
        if record["exists"] is False:
            if path.exists() or path.is_symlink():
                meta = path.lstat()
                if stat.S_ISLNK(meta.st_mode) or not stat.S_ISREG(meta.st_mode):
                    raise Failure("restore target is unsafe")
                path.unlink()
                fsync_dir(path.parent)
            continue
        payload, _ = read_regular(before / record["snapshot"])
        if sha(payload) != record["sha256"]:
            raise Failure("snapshot digest mismatch")
        atomic(path, payload, int(record["mode"]), int(record["uid"]), int(record["gid"]))

def verify_restored(records):
    for record in records:
        path = Path(record["path"])
        if record["exists"] is False:
            if path.exists() or path.is_symlink():
                return False
        else:
            payload, meta = read_regular(path)
            if sha(payload) != record["sha256"] or stat.S_IMODE(meta.st_mode) != record["mode"] or meta.st_uid != record["uid"] or meta.st_gid != record["gid"]:
                return False
    return True

def prepare(rid, manifest_sha):
    if SHA_RE.fullmatch(manifest_sha) is None:
        raise Failure("manifest digest is invalid")
    run_root, work, runtime, dropin = work_paths(rid)
    run_state = load_json(run_root / "run.json")
    transaction = run_state.get("trustTransaction")
    if not isinstance(transaction, dict) or transaction.get("phase") != "APPLIED" or transaction.get("liveVerified") is not True or transaction.get("manifestSha256") != manifest_sha:
        raise Failure("Fleet trust transaction is not live and release-bound")
    validate_active_run(rid)
    if work.exists() or work.is_symlink() or runtime.exists() or runtime.is_symlink() or dropin.exists() or dropin.is_symlink() or CONFIG_LEASE.exists() or CONFIG_LEASE.is_symlink():
        raise Failure("config-stream workspace already exists")
    parent_snapshots = {
        str(runtime.parent): directory_snapshot(runtime.parent),
        str(dropin.parent): directory_snapshot(dropin.parent),
    }
    was_active = service_active()
    if not was_active:
        raise Failure("Agent service must be active before config-stream E2E")
    initial_identity = runtime_identity(run_root, manifest_sha)
    work.mkdir(mode=0o700)
    os.chown(work, 0, 0)
    os.chmod(work, 0o700)
    dropin_payload = ("[Service]\nEnvironment=PATH=" + str(runtime / "bin") + ":/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n").encode()
    state = {
        "schemaVersion": 1,
        "runId": rid,
        "manifestSha256": manifest_sha,
        "phase": "ARMING",
        "serviceActiveBefore": was_active,
        "initialRuntimeIdentity": initial_identity,
        "configBeforeSha256": None,
        "configTestSha256": None,
        "dropinSha256": sha(dropin_payload),
        "parentSnapshots": parent_snapshots,
        "deadman": {"unit": deadman_unit(rid), "armed": True, "lifecycle": "ARMING"},
    }
    atomic(work / "state.json", canonical(state))
    try:
        arm_deadman(rid)
        claim_config_lease(rid)
        state["deadman"]["lifecycle"] = "ARMED"
        state["phase"] = "ARMED"
        atomic(work / "state.json", canonical(state))
        systemctl("stop", "nuv-agent.service")
        counts = db_counts()
        if any(counts.values()):
            raise Failure("pre-existing command work is not drained")
        active_baseline = baseline()
        config_raw, config_meta = read_regular(CONFIG, 2 * 1024 * 1024)
        test_config = render_updates(config_raw)
        records = save_snapshots(work)
        if records[0].get("sha256") != sha(config_raw):
            raise Failure("Agent config changed while snapshotting")
        state.update({"phase": "PREPARED", "snapshots": records, "baseline": active_baseline, "configBeforeSha256": sha(config_raw), "configTestSha256": sha(test_config)})
        atomic(work / "state.json", canonical(state))
        atomic(CONFIG, test_config, stat.S_IMODE(config_meta.st_mode), config_meta.st_uid, config_meta.st_gid)
        ensure_directory_from_snapshot(runtime.parent, parent_snapshots[str(runtime.parent)])
        ensure_directory_from_snapshot(dropin.parent, parent_snapshots[str(dropin.parent)])
        runtime.mkdir(mode=0o755)
        os.chown(runtime, 0, 0)
        os.chmod(runtime, 0o755)
        (runtime / "bin").mkdir(mode=0o755)
        os.chown(runtime / "bin", 0, 0)
        os.chmod(runtime / "bin", 0o755)
        atomic(runtime / "quality", b"GOOD\n", 0o644, 0, 0)
        iw = ("#!/bin/sh\nset -eu\nmode=$(/bin/cat " + shlex_quote(str(runtime / "quality")) + ")\nif [ \"${1:-}\" = dev ] && [ \"$#\" -eq 1 ]; then /bin/printf 'phy#0\\n\\tInterface fleet0\\n'; exit 0; fi\nif [ \"$mode\" = POOR ]; then signal=-90; rate='0.1 MBit/s'; else signal=-50; rate='100.0 MBit/s'; fi\n/bin/printf 'Connected to 00:11:22:33:44:55 (on fleet0)\\n\\tsignal: %s dBm\\n\\ttx bitrate: %s\\n\\trx bitrate: %s\\n' \"$signal\" \"$rate\" \"$rate\"\n").encode()
        ping = ("#!/bin/sh\nset -eu\nmode=$(/bin/cat " + shlex_quote(str(runtime / "quality")) + ")\nif [ \"$mode\" = POOR ]; then /bin/printf '3 packets transmitted, 2 received, 20%% packet loss\\nrtt min/avg/max/mdev = 300.000/300.000/300.000/0.000 ms\\n'; else /bin/printf '3 packets transmitted, 3 received, 0%% packet loss\\nrtt min/avg/max/mdev = 20.000/20.000/20.000/0.000 ms\\n'; fi\n").encode()
        atomic(runtime / "bin/iw", iw, 0o755, 0, 0)
        atomic(runtime / "bin/ping", ping, 0o755, 0, 0)
        atomic(dropin, dropin_payload, 0o644, 0, 0)
        state["phase"] = "ACTIVE"
        atomic(work / "state.json", canonical(state))
        systemctl("daemon-reload")
        systemctl("start", "nuv-agent.service")
        if not service_active():
            raise Failure("synthetic Agent service did not become active")
        state["testServicePid"] = service_pid()
        active_identity = runtime_identity(run_root, manifest_sha)
        if active_identity["release"] != initial_identity["release"] or active_identity["activeSlot"] != initial_identity["activeSlot"]:
            raise Failure("Agent release identity changed during preparation")
        state["activeRuntimeIdentity"] = active_identity
        atomic(work / "state.json", canonical(state))
    except BaseException:
        if (work / "state.json").exists():
            restore(rid, internal=True)
        raise
    return {"schemaVersion": 1, "runId": rid, "prepared": True, "syntheticSource": "videotestsrc", "connectivityShim": "scoped-iw-ping", "baseline": active_baseline, "configBeforeSha256": state["configBeforeSha256"], "configTestSha256": state["configTestSha256"], "runtimeIdentity": active_identity, "exclusiveLease": True, "deadmanArmed": True, "queue": counts}

def shlex_quote(value):
    return "'" + value.replace("'", "'\\''") + "'"

def set_link(rid, quality):
    _, work, runtime, _ = work_paths(rid)
    validate_config_lease(rid)
    state = load_json(work / "state.json")
    if state.get("phase") != "ACTIVE" or quality not in {"GOOD", "POOR"}:
        raise Failure("config-stream link transition is invalid")
    target = runtime / "quality"
    current, meta = read_regular(target, 16)
    del current
    atomic(target, (quality + "\n").encode(), stat.S_IMODE(meta.st_mode), meta.st_uid, meta.st_gid)
    return {"schemaVersion": 1, "runId": rid, "quality": quality, "changed": True}

def inspect(rid, command_id):
    if COMMAND_ID_RE.fullmatch(command_id) is None or str(uuid.UUID(command_id)) != command_id:
        raise Failure("commandId is invalid")
    _, work, _, _ = work_paths(rid)
    validate_config_lease(rid)
    state = load_json(work / "state.json")
    if state.get("phase") != "ACTIVE":
        raise Failure("config-stream workspace is not active")
    current_settings = baseline(False, state.get("baseline", {}).get("model"))
    settings_sha = sha(canonical(current_settings))
    connection = sqlite3.connect("file:" + str(COMMAND_DB) + "?mode=ro", uri=True, timeout=5)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute("SELECT command_id, sequence, command_type, status, reported_state_json FROM command_inbox WHERE command_id = ?", (command_id,)).fetchone()
        transitions = [str(item[0]) for item in connection.execute("SELECT status FROM command_ack_transitions WHERE command_id = ? ORDER BY rowid", (command_id,)).fetchall()]
        observation = connection.execute("SELECT revision, reported_state_json, acknowledged_at FROM fleet_command_observation WHERE command_id = ? ORDER BY revision DESC LIMIT 1", (command_id,)).fetchone()
        counts = db_counts()
        if row is None:
            command = None
        else:
            command = {"commandId": str(row["command_id"]), "sequence": int(row["sequence"]), "type": str(row["command_type"]), "status": str(row["status"]), "ackStatuses": transitions, "reportedState": json.loads(row["reported_state_json"]) if row["reported_state_json"] else None}
        observed = None if observation is None else {"revision": int(observation["revision"]), "reportedState": json.loads(observation["reported_state_json"]), "acked": observation["acknowledged_at"] is not None}
        return {"schemaVersion": 1, "runId": rid, "command": command, "observation": observed, "queue": counts, "settings": current_settings, "settingsSha256": settings_sha, "serviceActive": service_active()}
    finally:
        connection.close()

def validate_snapshot_records(records):
    if not isinstance(records, list) or len(records) != len(FIXED):
        raise Failure("config-stream snapshot set is invalid")
    for index, (record, expected_path) in enumerate(zip(records, FIXED)):
        if not isinstance(record, dict) or Path(str(record.get("path") or "")) != expected_path or type(record.get("exists")) is not bool:
            raise Failure("config-stream snapshot identity is invalid")
        if record["exists"]:
            if record.get("snapshot") != str(index) or SHA_RE.fullmatch(str(record.get("sha256") or "")) is None or type(record.get("mode")) is not int or not 0 <= record["mode"] <= 0o7777 or type(record.get("uid")) is not int or record["uid"] < 0 or type(record.get("gid")) is not int or record["gid"] < 0:
                raise Failure("config-stream snapshot metadata is invalid")
        elif any(record.get(key) is not None for key in ("snapshot", "sha256", "mode", "uid", "gid")):
            raise Failure("config-stream missing snapshot metadata is invalid")
    return records

def parent_snapshots(state, runtime, dropin):
    snapshots = state.get("parentSnapshots")
    expected = {str(runtime.parent), str(dropin.parent)}
    if not isinstance(snapshots, dict) or set(snapshots) != expected:
        raise Failure("config-stream parent snapshot set is invalid")
    for value in snapshots.values():
        if not isinstance(value, dict) or set(value) != {"exists", "mode", "uid", "gid"} or type(value.get("exists")) is not bool:
            raise Failure("config-stream parent snapshot is invalid")
    return snapshots

def remove_runtime_artifacts(state, runtime, dropin):
    if dropin.exists() or dropin.is_symlink():
        payload, meta = read_regular(dropin, 64 * 1024)
        if sha(payload) != state.get("dropinSha256") or meta.st_uid != 0 or meta.st_gid != 0 or stat.S_IMODE(meta.st_mode) != 0o644:
            raise Failure("config-stream drop-in changed")
        dropin.unlink()
        fsync_dir(dropin.parent)
    if runtime.exists() or runtime.is_symlink():
        meta = runtime.lstat()
        if stat.S_ISLNK(meta.st_mode) or not stat.S_ISDIR(meta.st_mode) or meta.st_uid != 0 or stat.S_IMODE(meta.st_mode) != 0o755:
            raise Failure("config-stream runtime path is unsafe")
        shutil.rmtree(runtime)
        fsync_dir(runtime.parent)

def purge_snapshots(work):
    before = work / "before"
    if not before.exists() and not before.is_symlink():
        return
    meta = before.lstat()
    if stat.S_ISLNK(meta.st_mode) or not stat.S_ISDIR(meta.st_mode) or meta.st_uid != 0 or stat.S_IMODE(meta.st_mode) != 0o700:
        raise Failure("config-stream snapshot directory is unsafe")
    shutil.rmtree(before)
    fsync_dir(work)

def complete_deadman_cleanup(rid, state, work, *, deadman):
    if deadman:
        state["deadman"] = {"unit": deadman_unit(rid), "armed": False, "lifecycle": "RECOVERED"}
    else:
        disarm_deadman(rid)
        state["deadman"] = {"unit": deadman_unit(rid), "armed": False, "lifecycle": "DISARMED"}
    atomic(work / "state.json", canonical(state))

def restoration_response(state, rid, work, restored_settings, identity, *, idempotent, no_mutation):
    if not isinstance(state.get("restoredCompletedAt"), str):
        state["restoredCompletedAt"] = utc_now()
        atomic(work / "state.json", canonical(state))
    cached = state.get("restorationResponse")
    if isinstance(cached, dict):
        if cached.get("schemaVersion") != 1 or cached.get("runId") != rid or not isinstance(cached.get("completedAt"), str) or cached.get("completedAt") != state.get("restoredCompletedAt") or cached.get("restored") is not True or cached.get("exactRestoration") is not True or cached.get("noMutation") is not no_mutation or cached.get("exclusiveLeaseReleased") is not True or cached.get("deadmanDisarmed") is not True:
            raise Failure("sealed config-stream restoration response is invalid")
        cached_settings = cached.get("settings")
        if isinstance(cached_settings, dict) and sha(canonical(cached_settings)) != cached.get("settingsSha256"):
            raise Failure("sealed config-stream settings evidence is invalid")
        response = dict(cached)
        response["idempotent"] = idempotent
        return response
    response = {
        "schemaVersion": 1,
        "runId": rid,
        "completedAt": state.get("restoredCompletedAt"),
        "restored": True,
        "idempotent": idempotent,
        "noMutation": no_mutation,
        "exactRestoration": True,
        "runtimeRestarted": state.get("runtimeRestarted", not no_mutation),
        "configSha256": state.get("configBeforeSha256"),
        "settings": restored_settings,
        "settingsSha256": sha(canonical(restored_settings)) if isinstance(restored_settings, dict) else None,
        "encoderStartupBitrateKbps": restored_settings["video"]["bitrateKbps"] if isinstance(restored_settings, dict) else None,
        "runtimeIdentity": identity,
        "exclusiveLeaseReleased": not CONFIG_LEASE.exists() and not CONFIG_LEASE.is_symlink(),
        "deadmanDisarmed": state.get("deadman", {}).get("armed") is False,
    }
    state["restorationResponse"] = dict(response)
    atomic(work / "state.json", canonical(state))
    return response

def restore(rid, internal=False, deadman=False):
    run_root, work, runtime, dropin = work_paths(rid)
    if not work.exists() and not work.is_symlink():
        if CONFIG_LEASE.exists() or CONFIG_LEASE.is_symlink():
            raise Failure("config-stream lease exists without a recovery journal")
        if not deadman:
            disarm_deadman(rid)
        return {"schemaVersion": 1, "runId": rid, "completedAt": utc_now(), "restored": True, "idempotent": True, "noMutation": True, "exactRestoration": True, "runtimeRestarted": False, "configSha256": None, "settings": None, "settingsSha256": None, "encoderStartupBitrateKbps": None, "runtimeIdentity": None, "exclusiveLeaseReleased": True, "deadmanDisarmed": True}
    work_meta = work.lstat()
    state_meta = (work / "state.json").lstat()
    if stat.S_ISLNK(work_meta.st_mode) or not stat.S_ISDIR(work_meta.st_mode) or work_meta.st_uid != 0 or stat.S_IMODE(work_meta.st_mode) != 0o700 or stat.S_ISLNK(state_meta.st_mode) or not stat.S_ISREG(state_meta.st_mode) or state_meta.st_uid != 0 or stat.S_IMODE(state_meta.st_mode) != 0o600:
        raise Failure("config-stream recovery journal metadata is unsafe")
    state = load_json(work / "state.json")
    if state.get("runId") != rid or state.get("schemaVersion") != 1 or SHA_RE.fullmatch(str(state.get("manifestSha256") or "")) is None:
        raise Failure("config-stream state identity is invalid")
    parents = parent_snapshots(state, runtime, dropin)
    phase = state.get("phase")
    if phase not in {"ARMING", "ARMED", "PREPARED", "ACTIVE", "RESTORING", "RESTORED"}:
        raise Failure("config-stream state phase is invalid")
    no_mutation = phase in {"ARMING", "ARMED"} or (phase == "RESTORED" and state.get("noMutation") is True)
    if no_mutation:
        if runtime.exists() or runtime.is_symlink() or dropin.exists() or dropin.is_symlink():
            raise Failure("config-stream mutation exists without snapshots")
        if state.get("serviceActiveBefore") is True and not service_active():
            systemctl("start", "nuv-agent.service")
        if not service_active():
            raise Failure("Agent service did not recover from pre-mutation state")
        identity = runtime_identity(run_root, state["manifestSha256"])
        state.update({"phase": "RESTORED", "restoredExact": True, "restoredServicePid": identity["servicePid"], "runtimeRestarted": False, "noMutation": True})
        atomic(work / "state.json", canonical(state))
        purge_snapshots(work)
        release_config_lease(rid)
        complete_deadman_cleanup(rid, state, work, deadman=deadman)
        return restoration_response(state, rid, work, None, identity, idempotent=phase == "RESTORED", no_mutation=True)

    records = validate_snapshot_records(state.get("snapshots"))
    if phase != "RESTORED":
        state["phase"] = "RESTORING"
        atomic(work / "state.json", canonical(state))
        systemctl("stop", "nuv-agent.service")
        restore_snapshots(work, records)
        remove_runtime_artifacts(state, runtime, dropin)
        systemctl("daemon-reload")
        if not verify_restored(records):
            raise Failure("config-stream byte restoration failed")
        if state.get("serviceActiveBefore") is True:
            systemctl("start", "nuv-agent.service")
        if not service_active():
            raise Failure("Agent service did not recover")
        restored_settings = baseline(False, state.get("baseline", {}).get("model"))
        if restored_settings != state.get("baseline") or not verify_restored(records):
            raise Failure("Agent restart did not retain the exact settings baseline")
        identity = runtime_identity(run_root, state["manifestSha256"])
        if identity["release"] != state.get("initialRuntimeIdentity", {}).get("release") or identity["activeSlot"] != state.get("initialRuntimeIdentity", {}).get("activeSlot"):
            raise Failure("restored Agent release identity changed")
        restored_pid = identity["servicePid"]
        if isinstance(state.get("testServicePid"), int) and restored_pid == state["testServicePid"]:
            raise Failure("Agent runtime was not restarted from restored config")
        restore_parent(runtime.parent, parents[str(runtime.parent)])
        restore_parent(dropin.parent, parents[str(dropin.parent)])
        if not verify_restored(records):
            raise Failure("config-stream restoration changed after restart")
        state.update({
            "phase": "RESTORED",
            "restoredExact": True,
            "restoredServicePid": restored_pid,
            "restoredSettings": restored_settings,
            "restoredSettingsSha256": sha(canonical(restored_settings)),
            "restoredRuntimeIdentity": identity,
            "runtimeRestarted": True,
            "noMutation": False,
        })
        # The durable RESTORED journal must precede deletion of the only
        # rollback bytes. A retry can therefore return full evidence after a
        # lost SSH response without depending on already-purged snapshots.
        atomic(work / "state.json", canonical(state))
    else:
        restored_settings = baseline(False, state.get("baseline", {}).get("model"))
        if state.get("restoredExact") is not True or restored_settings != state.get("baseline") or not verify_restored(records):
            raise Failure("idempotent config-stream restoration is not exact")
        if runtime.exists() or runtime.is_symlink() or dropin.exists() or dropin.is_symlink():
            raise Failure("idempotent config-stream artifacts remain")
        restore_parent(runtime.parent, parents[str(runtime.parent)])
        restore_parent(dropin.parent, parents[str(dropin.parent)])
        identity = runtime_identity(run_root, state["manifestSha256"])
        if identity["release"] != state.get("restoredRuntimeIdentity", {}).get("release"):
            raise Failure("idempotent Agent release identity changed")
    purge_snapshots(work)
    release_config_lease(rid)
    complete_deadman_cleanup(rid, state, work, deadman=deadman)
    return restoration_response(state, rid, work, restored_settings, identity, idempotent=phase == "RESTORED", no_mutation=False)

def reboot_restore(rid):
    run_root, work, runtime, dropin = work_paths(rid)
    del run_root
    if not work.exists() and not work.is_symlink():
        raise Failure("config-stream reboot recovery journal is unavailable")
    work_meta = work.lstat()
    state_meta = (work / "state.json").lstat()
    if stat.S_ISLNK(work_meta.st_mode) or not stat.S_ISDIR(work_meta.st_mode) or work_meta.st_uid != 0 or stat.S_IMODE(work_meta.st_mode) != 0o700 or stat.S_ISLNK(state_meta.st_mode) or not stat.S_ISREG(state_meta.st_mode) or state_meta.st_uid != 0 or stat.S_IMODE(state_meta.st_mode) != 0o600:
        raise Failure("config-stream reboot recovery journal metadata is unsafe")
    state = load_json(work / "state.json")
    if state.get("runId") != rid or state.get("schemaVersion") != 1 or SHA_RE.fullmatch(str(state.get("manifestSha256") or "")) is None:
        raise Failure("config-stream reboot recovery identity is invalid")
    lease_present = CONFIG_LEASE.exists() or CONFIG_LEASE.is_symlink()
    if lease_present:
        validate_config_lease(rid)
    cached_response = state.get("restorationResponse")
    if isinstance(cached_response, dict):
        if lease_present or state.get("rebootRecovery") is not True:
            raise Failure("completed config-stream restoration retained unsafe state")
        return restoration_response(
            state,
            rid,
            work,
            state.get("restoredSettings"),
            None,
            idempotent=True,
            no_mutation=state.get("noMutation") is True,
        )
    parents = parent_snapshots(state, runtime, dropin)
    phase = state.get("phase")
    if phase not in {"ARMING", "ARMED", "PREPARED", "ACTIVE", "RESTORING", "RESTORED"}:
        raise Failure("config-stream reboot recovery phase is invalid")
    if not lease_present and phase != "RESTORED":
        raise Failure("config-stream reboot recovery lease is absent")
    no_mutation = phase in {"ARMING", "ARMED"} or (phase == "RESTORED" and state.get("noMutation") is True)
    systemctl("stop", "nuv-agent.service", check=False)
    if no_mutation:
        if runtime.exists() or runtime.is_symlink() or dropin.exists() or dropin.is_symlink():
            raise Failure("config-stream mutation exists without snapshots")
        restored_settings = None
    else:
        records = validate_snapshot_records(state.get("snapshots"))
        if phase != "RESTORED":
            state["phase"] = "RESTORING"
            atomic(work / "state.json", canonical(state))
            restore_snapshots(work, records)
            remove_runtime_artifacts(state, runtime, dropin)
            systemctl("daemon-reload")
        elif runtime.exists() or runtime.is_symlink() or dropin.exists() or dropin.is_symlink():
            raise Failure("restored config-stream artifacts remain after reboot")
        if not verify_restored(records):
            raise Failure("config-stream reboot byte restoration failed")
        restored_settings = baseline(False, state.get("baseline", {}).get("model"))
        if restored_settings != state.get("baseline"):
            raise Failure("config-stream reboot settings restoration failed")
        restore_parent(runtime.parent, parents[str(runtime.parent)])
        restore_parent(dropin.parent, parents[str(dropin.parent)])
        if not verify_restored(records):
            raise Failure("config-stream reboot restoration changed")
    state.update({
        "phase": "RESTORED",
        "restoredExact": True,
        "restoredServicePid": None,
        "restoredSettings": restored_settings,
        "restoredSettingsSha256": sha(canonical(restored_settings)) if isinstance(restored_settings, dict) else None,
        "restoredRuntimeIdentity": None,
        "runtimeRestarted": False,
        "rebootRecovery": True,
        "noMutation": no_mutation,
    })
    atomic(work / "state.json", canonical(state))
    purge_snapshots(work)
    release_config_lease(rid)
    complete_deadman_cleanup(rid, state, work, deadman=False)
    return restoration_response(state, rid, work, restored_settings, None, idempotent=False, no_mutation=no_mutation)

def main():
    if os.geteuid() != 0:
        raise Failure("root privileges are required")
    if len(sys.argv) < 3:
        raise Failure("typed action and runId are required")
    action = sys.argv[1]
    rid = run_id(sys.argv[2])
    with operation_lock(rid):
        if action == "prepare" and len(sys.argv) == 4:
            result = prepare(rid, sys.argv[3])
        elif action == "set-link" and len(sys.argv) == 4:
            result = set_link(rid, sys.argv[3])
        elif action == "inspect" and len(sys.argv) == 4:
            result = inspect(rid, sys.argv[3])
        elif action == "restore" and len(sys.argv) == 3:
            result = restore(rid)
        elif action == "deadman-restore" and len(sys.argv) == 3:
            result = restore(rid, internal=True, deadman=True)
        elif action == "reboot-restore" and len(sys.argv) == 3:
            result = reboot_restore(rid)
        else:
            raise Failure("action is outside the typed allowlist")
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))

try:
    main()
except (Failure, OSError, ValueError, KeyError, sqlite3.Error, subprocess.SubprocessError):
    print("iq9075-config-stream-board: operation failed", file=sys.stderr)
    raise SystemExit(1)
'''


class RemoteBoard:
    ACTIONS = frozenset(
        {"prepare", "set-link", "inspect", "restore", "reboot-restore"}
    )

    def __init__(self, transport: Any) -> None:
        self.transport = transport

    def _invoke(
        self, action: str, run_id: str, arguments: Sequence[str] = ()
    ) -> dict[str, Any]:
        if action not in self.ACTIONS:
            raise ConfigStreamError("board action is outside the typed allowlist")
        if any(any(ch in item for ch in "\x00\r\n") for item in arguments):
            raise ConfigStreamError("board argument contains a control character")
        remote = [
            "/usr/bin/python3",
            "-I",
            "-c",
            BOARD_PROGRAM,
            action,
            run_id,
            *arguments,
        ]
        if self.transport.sudo_password is None:
            remote = ["/usr/bin/sudo", "-n", "--", *remote]
            input_bytes = None
        else:
            remote = ["/usr/bin/sudo", "-S", "-p", "", "--", *remote]
            input_bytes = bytes(self.transport.sudo_password) + b"\n"
        result = self.transport._run_with_auth(
            [
                "/usr/bin/ssh",
                *self.transport.base_options,
                "-p",
                str(self.transport.port),
                f"{self.transport.user}@{self.transport.host}",
                shlex.join(remote),
            ],
            timeout=360
            if action in {"prepare", "restore", "reboot-restore"}
            else 120,
            input_bytes=input_bytes,
        )
        return self.transport._parse_result(result, operation="config-stream-" + action)

    def prepare(self, *, run_id: str, manifest_sha256: str) -> dict[str, Any]:
        return self._invoke("prepare", run_id, (manifest_sha256,))

    def set_link(self, *, run_id: str, quality: str) -> dict[str, Any]:
        return self._invoke("set-link", run_id, (quality,))

    def inspect(self, *, run_id: str, command_id: str) -> dict[str, Any]:
        return self._invoke("inspect", run_id, (command_id,))

    def restore(self, *, run_id: str) -> dict[str, Any]:
        return self._invoke("restore", run_id)

    def recover_after_reboot(self, *, run_id: str) -> dict[str, Any]:
        return self._invoke("reboot-restore", run_id)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigStreamError(f"{label} is invalid")
    return value


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ConfigStreamError(f"{label} is invalid")
    return value


def _desired_is_reported(
    desired: Mapping[str, Any], reported: Mapping[str, Any]
) -> bool:
    return all(reported.get(key) == value for key, value in desired.items())


def validate_baseline(value: Any) -> dict[str, Any]:
    baseline = _mapping(value, "board settings baseline")
    if set(baseline) != {"model", "labels", "clip", "video"}:
        raise ConfigStreamError("board settings baseline fields are invalid")
    model = _mapping(baseline["model"], "model baseline")
    labels = _mapping(baseline["labels"], "labels baseline")
    clip = _mapping(baseline["clip"], "clip baseline")
    video = _mapping(baseline["video"], "video baseline")
    if (
        set(model)
        != {
            "pointer",
            "configuredDigest",
            "artifactDigest",
            "artifactVerified",
            "runtimeEnabled",
            "runtimeBackend",
        }
        or not isinstance(model.get("pointer"), str)
        or not model["pointer"]
        or (
            model.get("configuredDigest") is not None
            and (
                not isinstance(model.get("configuredDigest"), str)
                or re.fullmatch(
                    r"sha256:[0-9a-f]{64}", model["configuredDigest"]
                )
                is None
            )
        )
        or (
            model.get("artifactDigest") is not None
            and (
                not isinstance(model.get("artifactDigest"), str)
                or re.fullmatch(
                    r"sha256:[0-9a-f]{64}", model["artifactDigest"]
                )
                is None
            )
        )
        or type(model.get("artifactVerified")) is not bool
        or type(model.get("runtimeEnabled")) is not bool
        or not isinstance(model.get("runtimeBackend"), str)
        or not model["runtimeBackend"]
        or (model["artifactVerified"] is True and model["artifactDigest"] is None)
        or set(labels) != {"inspection", "anomaly"}
        or any(
            not isinstance(items, list)
            or any(not isinstance(item, str) or not item for item in items)
            for items in labels.values()
        )
        or set(clip) != {"enabled", "preSeconds", "postSeconds"}
        or type(clip.get("enabled")) is not bool
        or type(clip.get("preSeconds")) is not int
        or type(clip.get("postSeconds")) is not int
        or set(video) != {"width", "height", "fps", "bitrateKbps"}
        or any(type(video.get(key)) is not int for key in video)
    ):
        raise ConfigStreamError("board settings baseline values are invalid")
    return baseline


def validate_queue_drained(value: Any) -> dict[str, int]:
    queue = _mapping(value, "board command queue")
    expected = {
        "inboxPendingRows",
        "observationPendingRows",
        "observationReservedRows",
        "observationDlqRows",
    }
    if set(queue) != expected or any(
        type(queue.get(key)) is not int or queue[key] != 0 for key in expected
    ):
        raise ConfigStreamError("board command inbox/outbox is not drained")
    return queue


def twin_domain(projection: Mapping[str, Any], domain: str) -> dict[str, Any]:
    # The deployed dev contract exposes the authoritative current command as
    # a single twin. Prefer it when a rollout also includes advisory domains.
    twin = projection.get("twin")
    if isinstance(twin, dict):
        return twin
    twins = projection.get("twins")
    if isinstance(twins, Mapping) and isinstance(twins.get(domain), dict):
        return dict(twins[domain])
    raise ConfigStreamError("Fleet twin domain projection is unavailable")


def projection_shape(projection: Mapping[str, Any], domain: str) -> str:
    if isinstance(projection.get("twin"), dict):
        return "single"
    twins = projection.get("twins")
    if isinstance(twins, Mapping) and isinstance(twins.get(domain), dict):
        return "domained"
    raise ConfigStreamError("Fleet twin projection shape is unavailable")


def _utc_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ConfigStreamError(f"{label} is invalid")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise ConfigStreamError(f"{label} is invalid") from exc
    if parsed.tzinfo != timezone.utc:
        raise ConfigStreamError(f"{label} is invalid")
    return parsed


def validate_runtime_identity(
    value: Any, *, scenario: Mapping[str, Any]
) -> dict[str, Any]:
    identity = _mapping(value, "board runtime release identity")
    release = _mapping(scenario.get("release"), "Fleet manifest release")
    expected_bom = scenario.get("expectedBomDigest")
    if not isinstance(expected_bom, str) or DIGEST_RE.fullmatch(expected_bom) is None:
        raise ConfigStreamError("Fleet manifest BOM digest is invalid")
    expected_slot = "releases/" + expected_bom.removeprefix("sha256:")
    marker = {
        "schemaVersion": 2,
        "bomDigest": expected_bom,
        **dict(release),
    }
    expected_marker_sha256 = hashlib.sha256(canonical_json(marker)).hexdigest()
    expected_build_info_sha256 = hashlib.sha256(
        (
            '"""Generated release identity. Do not edit in release artifacts."""\n\n'
            f'AGENT_VERSION = "{release.get("agentVersion")}"\n'
            f'COMPONENT_SHA = "{release.get("componentSha")}"\n'
        ).encode("utf-8")
    ).hexdigest()
    if (
        set(release)
        != {
            "agentVersion",
            "releaseSequence",
            "artifactDigest",
            "componentSha",
            "configSchema",
            "publisherKeyId",
        }
        or set(identity)
        != {
            "activeSlot",
            "processActiveSlot",
            "processExpectedBomDigest",
            "servicePid",
            "releaseMarkerSha256",
            "buildInfoSha256",
            "release",
        }
        or identity.get("activeSlot") != expected_slot
        or identity.get("processActiveSlot") != expected_slot
        or identity.get("processExpectedBomDigest") != expected_bom
        or type(identity.get("servicePid")) is not int
        or identity["servicePid"] < 2
        or identity.get("releaseMarkerSha256") != expected_marker_sha256
        or identity.get("buildInfoSha256") != expected_build_info_sha256
        or identity.get("release") != marker
        or not isinstance(release.get("agentVersion"), str)
        or type(release.get("releaseSequence")) is not int
        or release["releaseSequence"] < 1
        or not isinstance(release.get("artifactDigest"), str)
        or DIGEST_RE.fullmatch(release["artifactDigest"]) is None
        or not isinstance(release.get("componentSha"), str)
        or COMPONENT_RE.fullmatch(release["componentSha"]) is None
        or not isinstance(release.get("configSchema"), str)
        or re.fullmatch(r"[1-9][0-9]*", release["configSchema"]) is None
        or not isinstance(release.get("publisherKeyId"), str)
        or not release["publisherKeyId"]
    ):
        raise ConfigStreamError("board runtime release identity is invalid")
    return dict(identity)


class ConfigStreamOrchestrator:
    def __init__(
        self,
        *,
        api: ApiPort,
        board: BoardPort,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        wall_clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.api = api
        self.board = board
        self.monotonic = monotonic
        self.sleeper = sleeper
        self.wall_clock = wall_clock or (lambda: datetime.now(timezone.utc))

    def _version_seed(self) -> int:
        return max(1, int(self.wall_clock().timestamp() * 1000))

    def _restore_board(self, *, run_id: str) -> dict[str, Any]:
        failure: BaseException | None = None
        for attempt in range(2):
            try:
                return self.board.restore(run_id=run_id)
            except BaseException as exc:
                failure = exc
                if attempt == 0:
                    self.sleeper(1.0)
        raise ConfigStreamError("board restoration RPC failed after retry") from failure

    def _journal_status(
        self, issued: IssuedCommand, *, space_id: int, device_id: str
    ) -> str | None:
        for item in self.api.commands(space_id=space_id, device_id=device_id):
            if item.get("commandId") == issued.command_id:
                if item.get("sequence") != issued.sequence or item.get("type") != issued.command_type:
                    raise ConfigStreamError("Fleet command journal identity changed")
                status = item.get("status")
                return str(status) if status is not None else None
        return None

    def _wait_command(
        self,
        issued: IssuedCommand,
        *,
        space_id: int,
        device_id: str,
        domain: str,
        desired: Mapping[str, Any],
        run_id: str,
        deadline: float,
    ) -> dict[str, Any]:
        while self.monotonic() < deadline:
            status = self._journal_status(
                issued, space_id=space_id, device_id=device_id
            )
            if status in {"FAILED", "REJECTED", "ROLLED_BACK", "EXPIRED"}:
                raise ConfigStreamError("Fleet command reached a non-success terminal state")
            board = self.board.inspect(run_id=run_id, command_id=issued.command_id)
            command = board.get("command") if isinstance(board, Mapping) else None
            if isinstance(command, Mapping):
                local_status = command.get("status")
                if local_status in {"FAILED", "REJECTED", "ROLLED_BACK"}:
                    raise ConfigStreamError("board command effect did not apply")
            projection = self.api.projection(space_id=space_id, device_id=device_id)
            twin = twin_domain(projection, domain)
            shape = projection_shape(projection, domain)
            reported = twin.get("reportedState")
            complete = (
                status == "SUCCEEDED"
                and isinstance(command, Mapping)
                and command.get("commandId") == issued.command_id
                and command.get("sequence") == issued.sequence
                and command.get("type") == issued.command_type
                and command.get("status") == "SUCCEEDED"
                and command.get("ackStatuses")
                == ["RECEIVED", "IN_PROGRESS", "SUCCEEDED"]
                and twin.get("convergenceStatus") == "CONVERGED"
                and twin.get("desiredCommandId") == issued.command_id
                and twin.get("reportedCommandId") == issued.command_id
                and twin.get("desiredSequence") == issued.sequence
                and twin.get("reportedSequence") == issued.sequence
                and isinstance(reported, Mapping)
                and _desired_is_reported(desired, reported)
                and board.get("serviceActive") is True
            )
            if complete:
                queue = board.get("queue")
                try:
                    validate_queue_drained(queue)
                except ConfigStreamError:
                    pass
                else:
                    observation = board.get("observation")
                    if (
                        isinstance(observation, Mapping)
                        and type(observation.get("revision")) is int
                        and observation["revision"] >= 1
                        and observation.get("acked") is True
                        and twin.get("reportedRevision")
                        == observation.get("revision")
                        and reported == observation.get("reportedState")
                    ):
                        board_settings = validate_baseline(board.get("settings"))
                        board_settings_sha = str(
                            board.get("settingsSha256") or ""
                        )
                        if (
                            not SHA256_RE.fullmatch(board_settings_sha)
                            or hashlib.sha256(canonical_json(board_settings)).hexdigest()
                            != board_settings_sha
                        ):
                            raise ConfigStreamError(
                                "board settings readback digest is invalid"
                            )
                        return {
                            "commandId": issued.command_id,
                            "sequence": issued.sequence,
                            "type": issued.command_type,
                            "lifecycleAckStatuses": list(command["ackStatuses"]),
                            "effectPhase": "APPLIED",
                            "reportedState": dict(reported),
                            "reportedRevision": int(twin.get("reportedRevision") or 0),
                            "localObservationRevision": int(observation["revision"]),
                            "boardSettings": board_settings,
                            "boardSettingsSha256": board_settings_sha,
                            "projectionShape": shape,
                            "queue": dict(queue),
                        }
            self.sleeper(1.0)
        raise ConfigStreamError("timed out waiting for command ACK and twin convergence")

    def _wait_adaptation(
        self,
        issued: IssuedCommand,
        *,
        space_id: int,
        device_id: str,
        run_id: str,
        after_revision: int,
        bitrate_test: Callable[[int], bool],
        deadline: float,
    ) -> dict[str, Any]:
        while self.monotonic() < deadline:
            status = self._journal_status(
                issued, space_id=space_id, device_id=device_id
            )
            if status in {"FAILED", "REJECTED", "ROLLED_BACK", "EXPIRED"}:
                raise ConfigStreamError(
                    "adaptive Fleet command reached a non-success terminal state"
                )
            if status != "SUCCEEDED":
                self.sleeper(1.0)
                continue
            board = self.board.inspect(run_id=run_id, command_id=issued.command_id)
            observation = board.get("observation") if isinstance(board, Mapping) else None
            projection = self.api.projection(space_id=space_id, device_id=device_id)
            twin = twin_domain(projection, "streaming")
            shape = projection_shape(projection, "streaming")
            reported = twin.get("reportedState")
            if (
                isinstance(observation, Mapping)
                and type(observation.get("revision")) is int
                and observation["revision"] > after_revision
                and observation.get("acked") is True
                and isinstance(reported, Mapping)
                and twin.get("desiredCommandId") == issued.command_id
                and twin.get("reportedCommandId") == issued.command_id
                and twin.get("desiredSequence") == issued.sequence
                and twin.get("reportedSequence") == issued.sequence
                and twin.get("convergenceStatus") == "CONVERGED"
                and twin.get("reportedRevision") == observation["revision"]
                and reported == observation.get("reportedState")
                and reported.get("encoder") == "x264enc"
                and reported.get("health") == "STREAM_CONTINUOUS"
                and type(reported.get("appliedBitrateKbps")) is int
                and bitrate_test(reported["appliedBitrateKbps"])
            ):
                try:
                    queue = validate_queue_drained(board.get("queue"))
                except ConfigStreamError:
                    pass
                else:
                    return {
                        "commandId": issued.command_id,
                        "sequence": issued.sequence,
                        "policyRevision": int(observation["revision"]),
                        "appliedBitrateKbps": int(reported["appliedBitrateKbps"]),
                        "lastAdjustmentReason": str(
                            reported.get("lastAdjustmentReason") or ""
                        ),
                        "health": str(reported.get("health") or ""),
                        "encoder": str(reported.get("encoder") or ""),
                        "projectionShape": shape,
                        "queue": queue,
                    }
            self.sleeper(1.0)
        raise ConfigStreamError("timed out waiting for adaptive streaming observation")

    def run(
        self,
        *,
        run_id: str,
        manifest: Mapping[str, Any],
        manifest_sha256: str,
        ota_evidence_sha256: str,
        wait_seconds: int,
    ) -> dict[str, Any]:
        identity = _mapping(manifest.get("identity"), "Fleet manifest identity")
        scenario = _mapping(manifest.get("scenario"), "Fleet manifest scenario")
        release = _mapping(scenario.get("release"), "Fleet manifest release")
        if scenario.get("type") != "commit":
            raise ConfigStreamError("config-stream E2E requires a commit manifest")
        try:
            expected_release_command_id = str(
                uuid.UUID(str(scenario.get("expectedCommandId") or ""))
            )
        except ValueError as exc:
            raise ConfigStreamError("commit manifest command identity is invalid") from exc
        if expected_release_command_id != scenario.get("expectedCommandId"):
            raise ConfigStreamError("commit manifest command identity is invalid")
        space_id = _positive_int(identity.get("spaceId"), "spaceId")
        device_id = identity.get("deviceId")
        if (
            not isinstance(device_id, str)
            or DEVICE_ID_RE.fullmatch(device_id) is None
            or int(DEVICE_ID_RE.fullmatch(device_id).group(1)) != space_id
        ):
            raise ConfigStreamError("Fleet manifest device identity is invalid")
        deadline = self.monotonic() + max(30, min(int(wait_seconds), 900))
        cleanup_required = False
        prep: dict[str, Any] | None = None
        baseline: dict[str, Any] | None = None
        runtime_identity: dict[str, Any] | None = None
        primary: BaseException | None = None
        result: dict[str, Any] | None = None
        try:
            # The board may durably mutate state and then lose its SSH reply.
            # Arm cleanup before dispatch so that ambiguous prepare outcomes
            # always reconcile through the board-owned journal.
            cleanup_required = True
            prep = self.board.prepare(
                run_id=run_id, manifest_sha256=manifest_sha256
            )
            if (
                prep.get("schemaVersion") != 1
                or prep.get("runId") != run_id
                or prep.get("prepared") is not True
                or prep.get("syntheticSource") != "videotestsrc"
                or prep.get("connectivityShim") != "scoped-iw-ping"
                or not SHA256_RE.fullmatch(str(prep.get("configBeforeSha256") or ""))
                or not SHA256_RE.fullmatch(str(prep.get("configTestSha256") or ""))
                or prep.get("exclusiveLease") is not True
                or prep.get("deadmanArmed") is not True
            ):
                raise ConfigStreamError("board preparation evidence is invalid")
            runtime_identity = validate_runtime_identity(
                prep.get("runtimeIdentity"), scenario=scenario
            )
            validate_queue_drained(prep.get("queue"))
            baseline = validate_baseline(prep.get("baseline"))
            terminal_statuses = {
                "SUCCEEDED",
                "FAILED",
                "REJECTED",
                "ROLLED_BACK",
                "EXPIRED",
            }
            journal_by_sequence: dict[int, dict[str, Any]] = {}
            seen_command_ids: set[str] = set()
            release_command: dict[str, Any] | None = None
            for item in self.api.commands(space_id=space_id, device_id=device_id):
                try:
                    command_id = str(uuid.UUID(str(item.get("commandId") or "")))
                except ValueError as exc:
                    raise ConfigStreamError("Fleet journal command identity is invalid") from exc
                sequence = _positive_int(item.get("sequence"), "Fleet journal sequence")
                command_type = item.get("type")
                status = item.get("status")
                if (
                    command_id != item.get("commandId")
                    or command_id in seen_command_ids
                    or sequence in journal_by_sequence
                    or not isinstance(command_type, str)
                    or not command_type
                    or not isinstance(status, str)
                ):
                    raise ConfigStreamError("Fleet command journal identity is ambiguous")
                seen_command_ids.add(command_id)
                journal_by_sequence[sequence] = {
                    "commandId": command_id,
                    "sequence": sequence,
                    "type": command_type,
                    "status": status,
                    "issuedAt": item.get("issuedAt"),
                    "expiresAt": item.get("expiresAt"),
                }
                if command_id == expected_release_command_id:
                    if release_command is not None:
                        raise ConfigStreamError("commit Fleet command is duplicated")
                    if (
                        command_type != "AGENT_UPDATE"
                        or status != "SUCCEEDED"
                    ):
                        raise ConfigStreamError("commit Fleet command is not successful")
                    release_command = {
                        "commandId": expected_release_command_id,
                        "sequence": sequence,
                        "type": "AGENT_UPDATE",
                        "status": "SUCCEEDED",
                        "issuedAt": item.get("issuedAt"),
                    }
            if release_command is None:
                raise ConfigStreamError("commit Fleet command is absent from the journal")
            if any(
                sequence > release_command["sequence"]
                for sequence in journal_by_sequence
            ):
                raise ConfigStreamError("commit Fleet command is not the journal head")
            prior_rollback_raw = journal_by_sequence.get(
                release_command["sequence"] - 1
            )
            if (
                prior_rollback_raw is None
                or prior_rollback_raw["type"] != "AGENT_UPDATE"
                or prior_rollback_raw["status"] != "ROLLED_BACK"
            ):
                raise ConfigStreamError(
                    "prior rollback command is not adjacent to the commit command"
                )
            prior_rollback_command = {
                key: prior_rollback_raw[key]
                for key in ("commandId", "sequence", "type", "status", "issuedAt")
            }
            release_issued_at = _utc_timestamp(
                release_command.get("issuedAt"), "commit command issuedAt"
            )
            rollback_issued_at = _utc_timestamp(
                prior_rollback_command.get("issuedAt"),
                "prior rollback command issuedAt",
            )
            if (
                rollback_issued_at > release_issued_at
                or release_issued_at > self.wall_clock()
            ):
                raise ConfigStreamError("release command issue ordering is invalid")
            historical = [
                item
                for sequence, item in journal_by_sequence.items()
                if sequence < release_command["sequence"]
            ]
            if any(item["status"] not in terminal_statuses for item in historical):
                raise ConfigStreamError("pre-commit Fleet command is not terminal")
            expired_predecessors = []
            for item in sorted(historical, key=lambda value: value["sequence"]):
                if item["status"] != "EXPIRED":
                    continue
                expires_at = item.get("expiresAt")
                if _utc_timestamp(
                    expires_at, "expired predecessor expiresAt"
                ) > self.wall_clock():
                    raise ConfigStreamError("expired predecessor deadline is in the future")
                expired_predecessors.append(
                    {
                        "commandId": item["commandId"],
                        "sequence": item["sequence"],
                        "type": item["type"],
                        "status": "EXPIRED",
                        "expiresAt": expires_at,
                    }
                )
            if not expired_predecessors:
                raise ConfigStreamError("expired predecessor evidence is unavailable")
            version = self._version_seed()
            baseline_bitrate = int(baseline["video"]["bitrateKbps"])
            changed_bitrate = (
                baseline_bitrate + 100
                if baseline_bitrate <= 19_900
                else baseline_bitrate - 100
            )
            config_payload = {
                "configVersion": version,
                "activation": "IMMEDIATE",
                "clip": dict(baseline["clip"]),
                "video": {**baseline["video"], "bitrateKbps": changed_bitrate},
            }
            issued_config = self.api.issue(
                space_id=space_id,
                device_id=device_id,
                command_type="CONFIG_APPLY",
                payload=config_payload,
                desired_state=config_payload,
            )
            if issued_config.sequence != release_command["sequence"] + 1:
                raise ConfigStreamError("fresh Fleet command does not follow the commit")
            config = self._wait_command(
                issued_config,
                space_id=space_id,
                device_id=device_id,
                domain="settings",
                desired=config_payload,
                run_id=run_id,
                deadline=deadline,
            )
            if (
                config["reportedState"].get("settingsDigest")
                != settings_digest(config_payload)
                or config["reportedState"].get("health")
                != "FUNCTIONAL_HEALTHY"
                or config["reportedState"].get("clip") != baseline["clip"]
                or config["reportedState"].get("video", {}).get("bitrateKbps")
                != changed_bitrate
                or config["boardSettings"].get("model") != baseline["model"]
                or config["boardSettings"].get("labels") != baseline["labels"]
                or config["boardSettings"].get("clip") != baseline["clip"]
                or config["boardSettings"].get("video", {}).get("bitrateKbps")
                != changed_bitrate
            ):
                raise ConfigStreamError("CONFIG_APPLY reported state is incomplete")

            restore_payload = {
                **config_payload,
                "configVersion": version + 1,
                "video": dict(baseline["video"]),
            }
            issued_restore = self.api.issue(
                space_id=space_id,
                device_id=device_id,
                command_type="CONFIG_APPLY",
                payload=restore_payload,
                desired_state=restore_payload,
            )
            if issued_restore.sequence != issued_config.sequence + 1:
                raise ConfigStreamError("fresh Fleet command sequence is not contiguous")
            config_restore = self._wait_command(
                issued_restore,
                space_id=space_id,
                device_id=device_id,
                domain="settings",
                desired=restore_payload,
                run_id=run_id,
                deadline=deadline,
            )
            if (
                config_restore["reportedState"].get("video") != baseline["video"]
                or config_restore["reportedState"].get("health")
                != "FUNCTIONAL_HEALTHY"
                or config_restore["boardSettings"] != baseline
            ):
                raise ConfigStreamError("CONFIG_APPLY restoration did not converge")

            self.board.set_link(run_id=run_id, quality="GOOD")
            initial = max(400, min(2000, baseline_bitrate))
            adaptive_payload = {
                "policyVersion": version + 2,
                "mode": "ADAPTIVE",
                "minBitrateKbps": max(100, initial // 4),
                "maxBitrateKbps": min(20_000, max(initial + 800, initial * 2)),
                "initialBitrateKbps": initial,
                "decreaseFactor": 0.5,
                "increaseStepKbps": 200,
                "congestionSamples": 2,
                "recoverySamples": 2,
                "cooldownSeconds": 1,
            }
            issued_adaptive = self.api.issue(
                space_id=space_id,
                device_id=device_id,
                command_type="STREAM_POLICY",
                payload=adaptive_payload,
                desired_state=adaptive_payload,
            )
            if issued_adaptive.sequence != issued_restore.sequence + 1:
                raise ConfigStreamError("fresh Fleet command sequence is not contiguous")
            adaptive = self._wait_command(
                issued_adaptive,
                space_id=space_id,
                device_id=device_id,
                domain="streaming",
                desired=adaptive_payload,
                run_id=run_id,
                deadline=deadline,
            )
            initial_state = {
                "commandId": issued_adaptive.command_id,
                "sequence": issued_adaptive.sequence,
                "policyRevision": adaptive["reportedRevision"],
                "appliedBitrateKbps": adaptive["reportedState"].get(
                    "appliedBitrateKbps"
                ),
                "health": adaptive["reportedState"].get("health"),
                "encoder": adaptive["reportedState"].get("encoder"),
                "lastAdjustmentReason": adaptive["reportedState"].get(
                    "lastAdjustmentReason"
                ),
                "projectionShape": adaptive["projectionShape"],
                "queue": adaptive["queue"],
            }
            initial_applied = initial_state["appliedBitrateKbps"]
            if (
                type(initial_applied) is not int
                or not adaptive_payload["minBitrateKbps"]
                <= initial_applied
                <= adaptive_payload["maxBitrateKbps"]
                or adaptive["reportedState"].get("mode") != "ADAPTIVE"
                or adaptive["reportedState"].get("encoder") != "x264enc"
                or adaptive["reportedState"].get("health") != "STREAM_CONTINUOUS"
                or adaptive["reportedState"].get("lastAdjustmentReason")
                != "policy_activated"
            ):
                raise ConfigStreamError("adaptive initial bitrate readback is invalid")

            self.board.set_link(run_id=run_id, quality="POOR")
            poor = self._wait_adaptation(
                issued_adaptive,
                space_id=space_id,
                device_id=device_id,
                run_id=run_id,
                after_revision=max(
                    adaptive["localObservationRevision"],
                    adaptive["reportedRevision"],
                ),
                bitrate_test=lambda bitrate: bitrate < initial_applied,
                deadline=deadline,
            )
            if "connectivity_poor" not in poor["lastAdjustmentReason"].split(","):
                raise ConfigStreamError("adaptive poor-link reason is invalid")
            self.board.set_link(run_id=run_id, quality="GOOD")
            recovered = self._wait_adaptation(
                issued_adaptive,
                space_id=space_id,
                device_id=device_id,
                run_id=run_id,
                after_revision=poor["policyRevision"],
                bitrate_test=lambda bitrate: bitrate > poor["appliedBitrateKbps"],
                deadline=deadline,
            )
            if recovered["lastAdjustmentReason"] != "healthy_recovery":
                raise ConfigStreamError("adaptive recovery reason is invalid")
            disabled_payload = {"policyVersion": version + 3, "mode": "DISABLED"}
            issued_disabled = self.api.issue(
                space_id=space_id,
                device_id=device_id,
                command_type="STREAM_POLICY",
                payload=disabled_payload,
                desired_state=disabled_payload,
            )
            if issued_disabled.sequence != issued_adaptive.sequence + 1:
                raise ConfigStreamError("fresh Fleet command sequence is not contiguous")
            disabled = self._wait_command(
                issued_disabled,
                space_id=space_id,
                device_id=device_id,
                domain="streaming",
                desired=disabled_payload,
                run_id=run_id,
                deadline=deadline,
            )
            if (
                disabled["reportedState"].get("mode") != "DISABLED"
                or disabled["reportedState"].get("encoder") != "x264enc"
                or disabled["reportedState"].get("health") != "STREAM_CONTINUOUS"
                or disabled["reportedState"].get("lastAdjustmentReason")
                != "policy_disabled"
            ):
                raise ConfigStreamError("stream policy did not disable")
            projection_shapes = {
                config["projectionShape"],
                config_restore["projectionShape"],
                adaptive["projectionShape"],
                poor["projectionShape"],
                recovered["projectionShape"],
                disabled["projectionShape"],
            }
            if len(projection_shapes) != 1:
                raise ConfigStreamError("Fleet twin projection shape changed during run")
            result = {
                "schemaVersion": SCHEMA_VERSION,
                "kind": KIND,
                "runId": run_id,
                "generatedAt": self.wall_clock()
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z"),
                "source": {
                    "manifestSha256": manifest_sha256,
                    "otaEvidenceSha256": ota_evidence_sha256,
                    "apiOrigin": DEFAULT_API_BASE_URL,
                    "agentVersion": release.get("agentVersion"),
                    "componentSha": release.get("componentSha"),
                    "bomDigest": scenario.get("expectedBomDigest"),
                    "configSchema": release.get("configSchema"),
                    "releaseSequence": release.get("releaseSequence"),
                    "artifactDigest": release.get("artifactDigest"),
                    "publisherKeyId": release.get("publisherKeyId"),
                    "runtimeIdentity": runtime_identity,
                },
                "identity": dict(identity),
                "releaseCommand": release_command,
                "priorRollbackCommand": prior_rollback_command,
                "expiredPredecessors": expired_predecessors,
                "projectionShape": next(iter(projection_shapes)),
                "config": {
                    "baseline": baseline,
                    "changedBitrateKbps": changed_bitrate,
                    "fieldCoverage": {
                        "model": "PRESERVED_WITHOUT_ACTIVATION",
                        "labels": "PRESERVED_WITHOUT_ACTIVATION",
                        "clipPolicy": "SAME_VALUE_RECONCILED",
                        "video": "CHANGED_AND_RESTORED",
                    },
                    "apply": config,
                    "restore": config_restore,
                },
                "stream": {
                    "adaptiveCommand": {
                        "commandId": issued_adaptive.command_id,
                        "sequence": issued_adaptive.sequence,
                        "lifecycleAckStatuses": adaptive[
                            "lifecycleAckStatuses"
                        ],
                        "effectPhase": "APPLIED",
                    },
                    "initialGood": initial_state,
                    "poor": poor,
                    "recoveredGood": recovered,
                    "disabled": disabled,
                },
                "boardPreparation": {
                    "syntheticSource": prep["syntheticSource"],
                    "connectivityShim": prep["connectivityShim"],
                    "configBeforeSha256": prep["configBeforeSha256"],
                    "configTestSha256": prep["configTestSha256"],
                },
                "gates": {
                    "releaseBound": True,
                    "cameraIndependent": True,
                    "modelConfigurationPreservedWithoutActivation": True,
                    "labelConfigurationPreservedWithoutActivation": True,
                    "clipPolicyReconciled": True,
                    "videoChangedAndRestored": True,
                    "ackReceivedToApplied": True,
                    "twinsConverged": True,
                    "adaptiveClosedLoop": True,
                    "commandQueuesDrained": True,
                    "encoderStartupBaselineRestored": False,
                    "exactBoardRestoration": False,
                },
                "modelQualification": (
                    {
                        "status": "ARTIFACT_IDENTITY_VERIFIED",
                        "artifactDigest": baseline["model"]["artifactDigest"],
                    }
                    if baseline["model"]["artifactVerified"] is True
                    else (
                        {
                        "status": "NOT_APPLICABLE_BACKEND_DISABLED",
                        "artifactDigest": None,
                        }
                        if baseline["model"]["runtimeEnabled"] is False
                        or baseline["model"]["runtimeBackend"] == "none"
                        else {
                            "status": "NOT_VERIFIED",
                            "artifactDigest": None,
                        }
                    )
                ),
            }
        except BaseException as exc:
            primary = exc
        finally:
            if cleanup_required:
                try:
                    restored = self._restore_board(run_id=run_id)
                    no_mutation = restored.get("noMutation") is True
                    valid_restoration = (
                        restored.get("schemaVersion") != 1
                        or restored.get("runId") != run_id
                        or type(restored.get("idempotent")) is not bool
                        or not isinstance(restored.get("completedAt"), str)
                        or restored.get("restored") is not True
                        or restored.get("exactRestoration") is not True
                        or restored.get("exclusiveLeaseReleased") is not True
                        or restored.get("deadmanDisarmed") is not True
                    )
                    if prep is not None:
                        valid_restoration = valid_restoration or (
                            no_mutation
                            or restored.get("runtimeRestarted") is not True
                            or restored.get("configSha256")
                            != prep.get("configBeforeSha256")
                            or (
                                result is not None
                                and _utc_timestamp(
                                    restored.get("completedAt"),
                                    "config-stream cleanup completedAt",
                                )
                                < _utc_timestamp(
                                    result.get("generatedAt"),
                                    "config-stream generatedAt",
                                )
                            )
                        )
                    elif not no_mutation:
                        valid_restoration = valid_restoration or (
                            restored.get("runtimeRestarted") is not True
                        )
                    if baseline is not None and not no_mutation:
                        restored_settings = _mapping(
                            restored.get("settings"),
                            "restored settings readback",
                        )
                        valid_restoration = valid_restoration or (
                            restored_settings != baseline
                            or restored.get("encoderStartupBitrateKbps")
                            != baseline["video"]["bitrateKbps"]
                            or hashlib.sha256(
                                canonical_json(restored_settings)
                            ).hexdigest()
                            != restored.get("settingsSha256")
                        )
                    if not no_mutation:
                        restored_identity = validate_runtime_identity(
                            restored.get("runtimeIdentity"), scenario=scenario
                        )
                        if runtime_identity is not None:
                            valid_restoration = valid_restoration or (
                                restored_identity["release"]
                                != runtime_identity["release"]
                                or restored_identity["activeSlot"]
                                != runtime_identity["activeSlot"]
                            )
                    if valid_restoration:
                        raise ConfigStreamError("board restoration evidence is invalid")
                    if result is not None:
                        result["cleanup"] = dict(restored)
                        result["gates"]["exactBoardRestoration"] = True
                        result["gates"]["encoderStartupBaselineRestored"] = True
                except BaseException as exc:
                    if primary is None:
                        primary = exc
                    else:
                        primary = ConfigStreamError(
                            "config-stream run and exact board restoration both failed"
                        )
        if primary is not None:
            if isinstance(primary, ConfigStreamError):
                raise primary
            raise ConfigStreamError("config-stream E2E failed") from primary
        if result is None or not all(result["gates"].values()):
            raise ConfigStreamError("config-stream evidence is incomplete")
        FLEET.assert_no_secret_material(result)
        return result


def validate_manifest_binding(
    output_dir: Path, *, run_id: str
) -> tuple[dict[str, Any], str, str]:
    manifest_path = output_dir / "immutable-manifest.json"
    ota_evidence_path = output_dir / "evidence.json"
    manifest_bytes = FLEET.read_regular(manifest_path, FLEET.MAX_INPUT_BYTES)
    evidence_bytes = FLEET.read_regular(ota_evidence_path, FLEET.MAX_OUTPUT_BYTES)
    manifest = FLEET.validate_manifest(
        FLEET.strict_json(manifest_bytes, label="immutable Fleet manifest")
    )
    FLEET.validate_final_evidence(
        FLEET.strict_json(evidence_bytes, label="immutable Fleet evidence"), manifest
    )
    if manifest.get("runId") != run_id:
        raise ConfigStreamError("Fleet manifest runId differs from requested run")
    return (
        manifest,
        hashlib.sha256(manifest_bytes).hexdigest(),
        hashlib.sha256(evidence_bytes).hexdigest(),
    )


def validate_reboot_recovery(value: Any, *, run_id: str) -> dict[str, Any]:
    recovery = _mapping(value, "config-stream reboot recovery")
    expected = {
        "schemaVersion",
        "runId",
        "completedAt",
        "restored",
        "idempotent",
        "noMutation",
        "exactRestoration",
        "runtimeRestarted",
        "configSha256",
        "settings",
        "settingsSha256",
        "encoderStartupBitrateKbps",
        "runtimeIdentity",
        "exclusiveLeaseReleased",
        "deadmanDisarmed",
    }
    if (
        set(recovery) != expected
        or recovery.get("schemaVersion") != 1
        or recovery.get("runId") != run_id
        or recovery.get("restored") is not True
        or type(recovery.get("idempotent")) is not bool
        or type(recovery.get("noMutation")) is not bool
        or recovery.get("exactRestoration") is not True
        or recovery.get("runtimeRestarted") is not False
        or recovery.get("runtimeIdentity") is not None
        or recovery.get("exclusiveLeaseReleased") is not True
        or recovery.get("deadmanDisarmed") is not True
    ):
        raise ConfigStreamError("config-stream reboot recovery is incomplete")
    _utc_timestamp(recovery.get("completedAt"), "reboot recovery completedAt")
    if recovery["noMutation"] is True:
        if any(
            recovery.get(name) is not None
            for name in (
                "configSha256",
                "settings",
                "settingsSha256",
                "encoderStartupBitrateKbps",
            )
        ):
            raise ConfigStreamError("no-mutation reboot recovery proof is invalid")
    else:
        settings = validate_baseline(recovery.get("settings"))
        settings_sha = recovery.get("settingsSha256")
        if (
            not SHA256_RE.fullmatch(str(recovery.get("configSha256") or ""))
            or not SHA256_RE.fullmatch(str(settings_sha or ""))
            or hashlib.sha256(canonical_json(settings)).hexdigest() != settings_sha
            or recovery.get("encoderStartupBitrateKbps")
            != settings["video"]["bitrateKbps"]
        ):
            raise ConfigStreamError("mutated reboot recovery proof is invalid")
    return dict(recovery)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run camera-independent IQ9075 CONFIG_APPLY/adaptive-streaming E2E"
    )
    parser.add_argument("--host", default=FLEET.DEFAULT_HOST)
    parser.add_argument("--user", default=FLEET.DEFAULT_USER)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--known-hosts", required=True)
    parser.add_argument("--host-key-sha256", default=FLEET.DEFAULT_FINGERPRINT)
    parser.add_argument("--ssh-password-fd", type=int)
    parser.add_argument("--sudo-password-fd", type=int)
    credentials = parser.add_mutually_exclusive_group()
    credentials.add_argument("--api-access-token-fd", type=int)
    credentials.add_argument("--api-cookie-jar")
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wait-seconds", type=int, default=420)
    parser.add_argument("--recover-after-reboot", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    ssh_password: bytearray | None = None
    sudo_password: bytearray | None = None
    api_token: bytearray | None = None
    try:
        FLEET.disable_core_dumps()
        arguments = build_parser().parse_args(argv)
        if arguments.api_base_url.rstrip("/") != DEFAULT_API_BASE_URL:
            raise ConfigStreamError(
                "qualification evidence requires the authoritative Nuvion dev API"
            )
        has_api_credential = (
            arguments.api_access_token_fd is not None
            or arguments.api_cookie_jar is not None
        )
        if arguments.recover_after_reboot and has_api_credential:
            raise ConfigStreamError(
                "reboot recovery does not accept Fleet API credentials"
            )
        if not arguments.recover_after_reboot and not has_api_credential:
            raise ConfigStreamError(
                "exactly one Fleet API credential source is required"
            )
        run_id = FLEET.canonical_run_id(arguments.run_id)
        output_dir = FLEET.prepare_output_dir(arguments.output_dir)
        if output_dir.name != run_id:
            raise ConfigStreamError("run output directory basename must equal runId")
        if arguments.ssh_password_fd is not None:
            ssh_password = FLEET.read_secret_fd(arguments.ssh_password_fd)
        if arguments.sudo_password_fd is not None:
            if (
                arguments.sudo_password_fd == arguments.ssh_password_fd
                and ssh_password is not None
            ):
                sudo_password = bytearray(ssh_password)
            else:
                sudo_password = FLEET.read_secret_fd(arguments.sudo_password_fd)
        if arguments.api_access_token_fd is not None:
            if arguments.api_access_token_fd in {
                arguments.ssh_password_fd,
                arguments.sudo_password_fd,
            }:
                raise ConfigStreamError("API token FD must be distinct from password FDs")
            api_token = FLEET.read_secret_fd(arguments.api_access_token_fd)
        manifest, manifest_sha, ota_evidence_sha = validate_manifest_binding(
            output_dir, run_id=run_id
        )
        pinned = output_dir / "known_hosts"
        fingerprint = FLEET.create_pinned_known_hosts(
            arguments.known_hosts,
            pinned,
            host=arguments.host,
            port=arguments.port,
            expected=arguments.host_key_sha256,
        )
        transport = FLEET.OpenSshTransport(
            host=arguments.host,
            user=arguments.user,
            port=arguments.port,
            pinned_known_hosts=pinned,
            expected_fingerprint=fingerprint,
            ssh_password=ssh_password,
            sudo_password=sudo_password,
            askpass_program=Path(__file__).with_name("run-iq9075-fleet-e2e.py"),
        )
        board = RemoteBoard(transport)
        if arguments.recover_after_reboot:
            recovery = validate_reboot_recovery(
                board.recover_after_reboot(run_id=run_id), run_id=run_id
            )
            recovery_document = {
                "schemaVersion": 1,
                "kind": "nuvion-iq9075-config-stream-reboot-recovery",
                "runId": run_id,
                "generatedAt": recovery["completedAt"],
                "recovery": recovery,
            }
            FLEET.assert_no_secret_material(recovery_document)
            recovery_path = output_dir / "config-stream-reboot-recovery.json"
            FLEET.atomic_json(recovery_path, recovery_document, immutable=True)
            persisted = FLEET.read_regular(
                recovery_path, FLEET.MAX_OUTPUT_BYTES
            )
            if persisted != canonical_json(recovery_document):
                raise ConfigStreamError(
                    "persisted reboot recovery evidence bytes differ"
                )
            print(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "runId": run_id,
                        "recovered": True,
                        "evidenceSha256": hashlib.sha256(persisted).hexdigest(),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0
        orchestrator = ConfigStreamOrchestrator(
            api=FleetApi(
                arguments.api_base_url,
                api_token,
                cookie_jar=arguments.api_cookie_jar,
            ),
            board=board,
        )
        evidence = orchestrator.run(
            run_id=run_id,
            manifest=manifest,
            manifest_sha256=manifest_sha,
            ota_evidence_sha256=ota_evidence_sha,
            wait_seconds=arguments.wait_seconds,
        )
        evidence_path = output_dir / "config-stream-evidence.json"
        FLEET.atomic_json(evidence_path, evidence, immutable=True)
        persisted = FLEET.read_regular(evidence_path, FLEET.MAX_OUTPUT_BYTES)
        if persisted != canonical_json(evidence):
            raise ConfigStreamError("persisted config-stream evidence bytes differ")
        print(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "runId": run_id,
                    "complete": True,
                    "evidenceSha256": hashlib.sha256(persisted).hexdigest(),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    except (ConfigStreamError, FLEET.RunnerError, OSError, ValueError) as exc:
        print(f"run-iq9075-config-stream-e2e: {exc}", file=sys.stderr)
        return 1
    except Exception:  # noqa: BLE001 - credentials must never enter a traceback.
        print(
            "run-iq9075-config-stream-e2e: unexpected internal failure",
            file=sys.stderr,
        )
        return 1
    finally:
        FLEET.zero_secret(ssh_password)
        FLEET.zero_secret(sudo_password)
        FLEET.zero_secret(api_token)


if __name__ == "__main__":
    raise SystemExit(main())
