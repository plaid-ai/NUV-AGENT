#!/usr/bin/env python3
"""Root-owned, crash-reconciling IQ9075 Fleet E2E board primitives."""

from __future__ import annotations

import argparse
import base64
import binascii
import errno
import fcntl
import gzip
import hashlib
import io
import json
import os
import re
import resource
import selectors
import shutil
import signal
import sqlite3
import stat
import subprocess
import sys
import tarfile
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any

PROTOCOL_VERSION = "iq9075-fleet-e2e-v2"
REQUIRED_UPDATER_VERSION = "0.2.0"
TRUST_DOMAIN = "iq9075-dev"
USB_ROOT_HUB = "2-1"
USB_TOPOLOGY_RE = re.compile(
    rf"^{re.escape(USB_ROOT_HUB)}(?:\.[1-9][0-9]*)+$"
)
OAK_VENDOR = "03e7"
OAK_PRODUCT = "f63b"
OAK_MIN_SPEED_MBPS = 5000.0
DEADMAN_SECONDS = 120
MAX_FAULT_HOLD_SECONDS = 60
MIN_FREE_BYTES = 2 * 1024 * 1024 * 1024
MAX_TRUST_BYTES = 64 * 1024
MAX_STATE_BYTES = 2 * 1024 * 1024
MAX_BACKUP_BYTES = 2 * 1024 * 1024 * 1024
MAX_BACKUP_ENTRIES = 200_000
MAX_CANDIDATE_BUNDLE_BYTES = 4 * 1024 * 1024 * 1024
MAX_CANDIDATE_UNPACKED_BYTES = 8 * 1024 * 1024 * 1024
MAX_CANDIDATE_ENTRIES = 100_000
MAX_CANDIDATE_TAR_METADATA_BYTES = 1024 * 1024
MAX_CANDIDATE_TAR_METADATA_TOTAL_BYTES = 16 * 1024 * 1024
# Includes tar headers/padding in addition to the sum of regular-file payloads.
# This is a hard bound on bytes returned by a gzip decoder before ``tarfile``
# is allowed to parse attacker-controlled extended headers.
MAX_CANDIDATE_TAR_STREAM_BYTES = MAX_CANDIDATE_UNPACKED_BYTES + 512 * 1024 * 1024
MAX_COMMAND_STDOUT_BYTES = 256 * 1024
MAX_COMMAND_STDERR_BYTES = 16 * 1024
COMMAND_IO_CHUNK_BYTES = 64 * 1024
MAX_MOUNTINFO_BYTES = 2 * 1024 * 1024
SYSTEM_PYTHON = "/usr/bin/python3"
CANDIDATE_SITE_PACKAGES_RELATIVE = Path("venv/lib/python3.12/site-packages")
CANDIDATE_PYTHON_WRAPPER = b"""#!/bin/sh
set -eu
slot_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
export PYTHONNOUSERSITE=1
export PYTHONPATH="$slot_root/venv/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
exec "${NUVION_SYSTEM_PYTHON:-/usr/bin/python3}" "$@"
"""
CANDIDATE_DEADMAN_SECONDS = 900
# Upper bound: controller fence/lock 50s + transient termination 180s +
# candidate baseline validation/restore 360s + trust transaction restore 360s +
# final production attestation 150s = 1100s; retain a 100s scheduler margin.
CANDIDATE_DEADMAN_RECOVERY_SECONDS = 1200
# The host intentionally crosses two independent CLI invocations between
# Fleet evidence and candidate-soak evidence.  Keep that complete workflow
# bounded while leaving enough room for the maximum Fleet wait, candidate
# execution, and recovery retries.
TRANSACTION_DEADMAN_SECONDS = 4 * 60 * 60
TRANSACTION_DEADMAN_RECOVERY_SECONDS = 30 * 60
CANDIDATE_REQUIRED_SOAK_SECONDS = 120
CANDIDATE_UID_SCAN_INTERVAL_SECONDS = 0.05
TOOL_DESTINATION = Path("/usr/local/libexec/nuvion/iq9075-board-e2e.py")
CANDIDATE_HARNESS = Path("/usr/lib/nuvion-updater/test-iq9075.sh")
UPDATER_STATE_DB = Path("/var/lib/nuvion-updater/updater.sqlite3")
BOOT_RECONCILE_UNIT = "nuvion-fleet-e2e-reconcile.service"


def disable_core_dumps() -> None:
    """Fail closed before this root process can read trust material."""

    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (OSError, ValueError) as exc:
        raise HarnessError("cannot disable core dumps") from exc

AGENT_CONFIG = "/etc/nuv-agent/agent.env"
AGENT_COMMAND_KEYRING = "/etc/nuv-agent/fleet-command-keyring.json"
UPDATER_COMMAND_KEYRING = "/etc/nuvion-updater/command-keyring.json"
RELEASE_KEYRING = "/etc/nuvion-updater/release-keyring.json"
HEALTH_KEYRING = "/etc/nuvion-updater/health-attestation-keyring.json"
DEVICE_BINDING = "/etc/nuvion-updater/device-binding.json"
FIXED_DESTINATIONS = {
    "agentCommand": AGENT_COMMAND_KEYRING,
    "updaterCommand": UPDATER_COMMAND_KEYRING,
    "release": RELEASE_KEYRING,
    "health": HEALTH_KEYRING,
    "binding": DEVICE_BINDING,
}
TRANSACTION_FILES = frozenset({AGENT_CONFIG, *FIXED_DESTINATIONS.values()})
TRANSACTION_DIRECTORIES = frozenset({"/etc/nuv-agent", "/etc/nuvion-updater"})
if len({AGENT_CONFIG, *FIXED_DESTINATIONS.values()}) != 6:
    raise RuntimeError("Fleet E2E destinations must be distinct")

UNITS = (
    "nuv-agent.service",
    "nuv-agent-updater.service",
    "nuv-agent-updater.socket",
)
RESTART_ORDER = (
    "nuv-agent-updater.socket",
    "nuv-agent-updater.service",
    "nuv-agent.service",
)
STOP_ORDER = (
    "nuv-agent.service",
    "nuv-agent-updater.socket",
    "nuv-agent-updater.service",
)
BACKUP_PATHS = (
    "/etc/nuv-agent",
    "/etc/nuvion-updater",
    "/var/lib/nuv-agent",
    "/var/lib/nuvion-updater",
    "/opt/nuv-agent/current",
    "/opt/nuv-agent/previous",
)
RUN_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DIGEST_RE = re.compile(r"^sha256:([0-9a-f]{64})$")
COMPONENT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
SEMVER_RE = re.compile(
    r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)
KEY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
DEVICE_ID_RE = re.compile(r"^sp-([1-9][0-9]*)-nuvion-[a-z0-9][a-z0-9-]{0,100}$")
PHASES = {"PREPARED", "APPLYING", "APPLIED", "RESTORING", "RESTORED"}
INPUT_ROLES = ("command", "release", "health", "binding", "manifest")
CANDIDATE_INPUT_ROLES = ("candidate-bundle", "candidate-bom", "oak-harness")
CANDIDATE_PERSISTENT_PATHS = (
    "/etc/nuv-agent",
    "/etc/nuvion-updater",
    "/var/lib/nuv-agent",
    "/var/lib/nuvion-updater",
)
CANDIDATE_RESOURCE_PROPERTIES = (
    "MemoryHigh=805306368",
    "MemoryMax=1073741824",
    "MemorySwapMax=0",
    "TasksMax=256",
    "CPUQuota=300%",
    "CPUQuotaPeriodSec=1s",
    "LimitCORE=0",
    "LimitFSIZE=8388608",
)
CANDIDATE_TMPFS_LIMITS = {
    "/tmp": {"bytes": 256 * 1024 * 1024, "inodes": 8192},
    "/var/tmp": {"bytes": 64 * 1024 * 1024, "inodes": 4096},
    "/dev/shm": {"bytes": 256 * 1024 * 1024, "inodes": 8192},
}
CANDIDATE_INACCESSIBLE_PATHS = ("/run/user",)
CANDIDATE_SANDBOX_PROPERTIES = (
    "ProtectSystem=strict",
    "ProtectHome=yes",
    *(
        "TemporaryFileSystem="
        f"{path}:rw,nosuid,nodev,size={limits['bytes']},"
        f"nr_inodes={limits['inodes']},mode=1777"
        for path, limits in CANDIDATE_TMPFS_LIMITS.items()
    ),
    *(f"InaccessiblePaths={path}" for path in CANDIDATE_INACCESSIBLE_PATHS),
    "NoNewPrivileges=yes",
    "RestrictSUIDSGID=yes",
    "RestrictNamespaces=yes",
    "ProtectKernelTunables=yes",
    "ProtectKernelModules=yes",
    "ProtectControlGroups=yes",
    "ProtectProc=invisible",
    "ProcSubset=pid",
    "LockPersonality=yes",
    "RestrictRealtime=yes",
    "StandardOutput=null",
    "StandardError=null",
)
CANDIDATE_SYSTEMD_EXPECTED_PROPERTIES = {
    "CPUQuotaPerSecUSec": "3s",
    "CPUQuotaPeriodUSec": "1s",
    "KillMode": "control-group",
    "LimitCORE": "0",
    "LimitFSIZE": "8388608",
    "LockPersonality": "yes",
    "MemoryHigh": "805306368",
    "MemoryMax": "1073741824",
    "MemorySwapMax": "0",
    "NoNewPrivileges": "yes",
    "ProtectControlGroups": "yes",
    "ProtectHome": "yes",
    "ProtectKernelModules": "yes",
    "ProtectKernelTunables": "yes",
    "ProtectProc": "invisible",
    "ProtectSystem": "strict",
    "ProcSubset": "pid",
    "RemainAfterExit": "yes",
    "RestrictNamespaces": "yes",
    "RestrictRealtime": "yes",
    "RestrictSUIDSGID": "yes",
    "RuntimeMaxUSec": "12min",
    "SendSIGKILL": "yes",
    "StandardError": "null",
    "StandardOutput": "null",
    "TasksMax": "256",
    "TimeoutStopUSec": "30s",
    "Type": "exec",
}
CANDIDATE_PHASES = (
    "PREPARED",
    "STAGING",
    "STAGED",
    "QUIESCING",
    "QUIESCED",
    "RUNNING",
    "CAPTURED",
    "RESTORING",
    "RESTORED",
    "COMPLETE",
)
ALLOWED_UPDATE_FIELDS = frozenset(
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
        "commandExpiresAt",
        "healthDeadline",
        "errorCode",
        "slot",
        "rollbackSlot",
        "rollbackVersion",
        "health",
        "functionalHealth",
    }
)
RELEASE_FIELDS = frozenset(
    {
        "schemaVersion",
        "bomDigest",
        "agentVersion",
        "releaseSequence",
        "artifactDigest",
        "componentSha",
        "configSchema",
        "publisherKeyId",
    }
)
SECRET_KEY_RE = re.compile(
    r"(?:password|passwd|secret|private|credential|compactjws|access.?token|"
    r"refresh.?token|authorization|raw.?config|sqlite|sdp|ice.?candidate)",
    re.IGNORECASE,
)
SECRET_VALUE_RES = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\b(?:authorization|password|passwd|token)\s*[:=]"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\b[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"(?im)^v=0\r?$.*^m=(?:audio|video)\s"),
    re.compile(r"(?i)\bcandidate:[0-9]+\s"),
)
SAFE_SECRET_LIKE_FIELDS = frozenset({"offerSdpHadPinnedProfile"})

UPDATER_PROBE = r"""
import json
from nuvion_app.runtime.updater_client import UpdaterClient
status = UpdaterClient().capability_status()
safe = {
    "capabilityAvailable": status.get("capabilityAvailable") is True,
    "authenticatedHelper": status.get("authenticatedHelper") is True,
    "reason": str(status.get("reason") or "")[:100],
    "updaterVersion": str(status.get("updaterVersion") or "unknown")[:100],
}
update = status.get("update")
if isinstance(update, dict):
    safe["update"] = update
print(json.dumps(safe, sort_keys=True, separators=(",", ":")))
raise SystemExit(0 if safe["capabilityAvailable"] and safe["authenticatedHelper"] else 3)
""".strip()


class HarnessError(RuntimeError):
    """Stable error text that never contains configuration or command payloads."""


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class CommandRunner:
    def run(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        input_bytes: bytes | None = None,
    ) -> CommandResult:
        if isinstance(argv, (str, bytes)) or not argv:
            raise TypeError("argv must be a non-empty sequence")
        command = list(argv)
        process = subprocess.Popen(  # noqa: S603 - fixed argv at every call site.
            command,
            stdin=subprocess.PIPE if input_bytes is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout = bytearray()
        stderr = bytearray()
        deadline = time.monotonic() + timeout
        selector = selectors.DefaultSelector()

        def register_reader(stream: Any, target: bytearray, maximum: int) -> None:
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, (target, maximum))

        assert process.stdout is not None
        assert process.stderr is not None
        register_reader(process.stdout, stdout, MAX_COMMAND_STDOUT_BYTES)
        register_reader(process.stderr, stderr, MAX_COMMAND_STDERR_BYTES)
        input_offset = 0
        if process.stdin is not None:
            if input_bytes:
                os.set_blocking(process.stdin.fileno(), False)
                selector.register(process.stdin, selectors.EVENT_WRITE, None)
            else:
                process.stdin.close()

        try:
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                events = selector.select(remaining)
                if not events:
                    raise subprocess.TimeoutExpired(command, timeout)
                for key, _mask in events:
                    stream = key.fileobj
                    if key.data is None:
                        assert input_bytes is not None
                        try:
                            written = os.write(
                                stream.fileno(),
                                input_bytes[input_offset : input_offset + COMMAND_IO_CHUNK_BYTES],
                            )
                        except BrokenPipeError:
                            written = len(input_bytes) - input_offset
                        input_offset += written
                        if input_offset >= len(input_bytes):
                            selector.unregister(stream)
                            stream.close()
                        continue
                    target, maximum = key.data
                    try:
                        chunk = os.read(stream.fileno(), COMMAND_IO_CHUNK_BYTES)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(stream)
                        stream.close()
                        continue
                    retained = maximum - len(target)
                    if retained > 0:
                        target.extend(chunk[:retained])
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.wait()
            raise HarnessError("bounded command timed out") from exc
        finally:
            for key in list(selector.get_map().values()):
                stream = key.fileobj
                selector.unregister(stream)
                stream.close()
            selector.close()
            if process.poll() is None:
                process.kill()
                process.wait()
        return CommandResult(
            returncode,
            bytes(stdout).decode("utf-8", errors="replace"),
            bytes(stderr).decode("utf-8", errors="replace"),
        )


@dataclass(frozen=True)
class BoardPaths:
    root: Path
    config: Path
    install_root: Path
    state_root: Path
    recovery_root: Path
    lock_root: Path
    global_fleet_lock: Path
    global_usb_lock: Path
    active_run: Path
    config_stream_active: Path
    package_maintenance: Path
    candidate_root: Path
    updater_state_db: Path
    candidate_harness: Path
    proc_root: Path
    boot_id: Path
    cgroup_root: Path
    usb_devices: Path
    usb_unbind: Path
    usb_bind: Path
    os_release: Path
    device_model: Path

    @classmethod
    def from_root(cls, root: str | Path = "/") -> BoardPaths:
        root_path = Path(root)

        def p(absolute: str) -> Path:
            return root_path / absolute.lstrip("/")

        return cls(
            root=root_path,
            config=p(AGENT_CONFIG),
            install_root=p("/opt/nuv-agent"),
            state_root=p("/var/lib/nuvion-fleet-e2e/runs"),
            recovery_root=p("/var/lib/nuvion-fleet-e2e/recovery"),
            lock_root=p("/run/lock"),
            global_fleet_lock=p("/run/lock/nuvion-fleet-e2e.lock"),
            global_usb_lock=p("/run/lock/nuvion-oak-e2e.lock"),
            active_run=p("/var/lib/nuvion-fleet-e2e/active-run.json"),
            config_stream_active=p(
                "/var/lib/nuvion-fleet-e2e/config-stream-active.json"
            ),
            package_maintenance=p(
                "/var/lib/nuvion-fleet-e2e/package-maintenance.json"
            ),
            candidate_root=p("/opt/nuv-agent/candidates"),
            updater_state_db=p(str(UPDATER_STATE_DB)),
            candidate_harness=p(str(CANDIDATE_HARNESS)),
            proc_root=p("/proc"),
            boot_id=p("/proc/sys/kernel/random/boot_id"),
            cgroup_root=p("/sys/fs/cgroup"),
            usb_devices=p("/sys/bus/usb/devices"),
            usb_unbind=p("/sys/bus/usb/drivers/usb/unbind"),
            usb_bind=p("/sys/bus/usb/drivers/usb/bind"),
            # Ubuntu exposes /etc/os-release as a symlink. Read the canonical
            # vendor file so the strict regular-file reader remains fail closed.
            os_release=p("/usr/lib/os-release"),
            device_model=p("/proc/device-tree/model"),
        )

    def rooted(self, absolute: str) -> Path:
        value = PurePosixPath(absolute)
        if not value.is_absolute() or ".." in value.parts:
            raise HarnessError("path must be absolute and normalized")
        return self.root / str(value).lstrip("/")


@dataclass(frozen=True)
class FileSnapshot:
    exists: bool
    payload: bytes
    mode: int = 0
    uid: int = 0
    gid: int = 0


@dataclass(frozen=True)
class StagedInput:
    role: str
    path: Path
    payload: bytes
    sha256: str
    device: int
    inode: int


@dataclass(frozen=True)
class CandidateInput:
    role: str
    path: Path
    sha256: str
    size: int
    device: int
    inode: int


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def canonical_run_id(value: str) -> str:
    try:
        normalized = str(uuid.UUID(value))
    except (ValueError, AttributeError) as exc:
        raise HarnessError("runId must be a canonical UUIDv4") from exc
    if normalized != value or not RUN_ID_RE.fullmatch(normalized):
        raise HarnessError("runId must be a canonical UUIDv4")
    return normalized


def canonical_oak_port(value: object) -> str:
    port = str(value or "").strip()
    if USB_TOPOLOGY_RE.fullmatch(port) is None:
        raise HarnessError("OAK USB topology is outside the USB1 downstream hub")
    return port


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def strict_json(payload: bytes, *, label: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise HarnessError(f"{label} contains duplicate JSON members")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise HarnessError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise HarnessError(f"{label} root must be an object")
    return value


def read_regular(
    path: Path,
    *,
    maximum: int,
    kernel_virtual_size: bool = False,
) -> tuple[bytes, os.stat_result]:
    try:
        before = path.lstat()
    except OSError as exc:
        raise HarnessError(
            f"required regular file is unavailable: {path.name}"
        ) from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise HarnessError(f"unsafe non-regular file: {path.name}")
    # sysfs attributes commonly report PAGE_SIZE rather than their readable
    # payload length. The descriptor read below remains bounded to maximum + 1.
    if not kernel_virtual_size and before.st_size > maximum:
        raise HarnessError(f"file exceeds size limit: {path.name}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise HarnessError(f"file changed while opening: {path.name}")
        result = bytearray()
        while len(result) <= maximum:
            chunk = os.read(descriptor, min(64 * 1024, maximum + 1 - len(result)))
            if not chunk:
                break
            result.extend(chunk)
        if len(result) > maximum:
            raise HarnessError(f"file exceeds size limit: {path.name}")
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino, after.st_size) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
        ):
            raise HarnessError(f"file changed while reading: {path.name}")
        return bytes(result), opened
    finally:
        os.close(descriptor)


def sha256_regular(path: Path, *, maximum: int) -> tuple[str, os.stat_result]:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise HarnessError(f"unsafe non-regular file: {path.name}")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    digest = hashlib.sha256()
    total = 0
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise HarnessError(f"file changed while opening: {path.name}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise HarnessError(f"file exceeds size limit: {path.name}")
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino, after.st_size) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
        ):
            raise HarnessError(f"file changed while hashing: {path.name}")
        return digest.hexdigest(), opened
    finally:
        os.close(descriptor)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def ensure_directory(path: Path, *, mode: int, uid: int, gid: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise HarnessError(f"unsafe directory: {path.name}")
    if (metadata.st_uid, metadata.st_gid) != (uid, gid):
        os.chown(path, uid, gid)
    if stat.S_IMODE(metadata.st_mode) != mode:
        os.chmod(path, mode)
    fsync_directory(path.parent)


def atomic_write(
    path: Path,
    payload: bytes,
    *,
    mode: int,
    uid: int,
    gid: int,
) -> None:
    if path.exists() or path.is_symlink():
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise HarnessError(f"refusing to replace unsafe file: {path.name}")
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    descriptor = os.open(
        temporary,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise HarnessError("atomic write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        if (opened.st_uid, opened.st_gid) != (uid, gid):
            os.fchown(descriptor, uid, gid)
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def snapshot(path: Path, *, maximum: int = MAX_STATE_BYTES) -> FileSnapshot:
    if not path.exists() and not path.is_symlink():
        return FileSnapshot(False, b"")
    payload, metadata = read_regular(path, maximum=maximum)
    return FileSnapshot(
        True,
        payload,
        stat.S_IMODE(metadata.st_mode),
        metadata.st_uid,
        metadata.st_gid,
    )


def restore_snapshot(path: Path, value: FileSnapshot) -> None:
    if value.exists:
        atomic_write(
            path,
            value.payload,
            mode=value.mode,
            uid=value.uid,
            gid=value.gid,
        )
        return
    if path.exists() or path.is_symlink():
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise HarnessError(f"refusing to remove unsafe file: {path.name}")
        path.unlink()
        fsync_directory(path.parent)


def assert_no_secret_material(value: object) -> None:
    def visit(item: object) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if str(key) in SAFE_SECRET_LIKE_FIELDS and not isinstance(
                    child, bool
                ):
                    raise HarnessError("output contains an invalid safe evidence field")
                if (
                    str(key) not in SAFE_SECRET_LIKE_FIELDS
                    and SECRET_KEY_RE.search(str(key))
                ):
                    raise HarnessError("output contains a forbidden field")
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
        elif isinstance(item, str) and any(
            pattern.search(item) for pattern in SECRET_VALUE_RES
        ):
            raise HarnessError("output contains secret or session material")

    visit(value)


def validate_ed25519_keyring(payload: bytes, *, role: str) -> dict[str, Any]:
    value = strict_json(payload, label=f"{role} keyring")
    expected = {"schemaVersion", "trustDomain", "keys"}
    if role == "health":
        expected.add("purpose")
    if (
        set(value) != expected
        or type(value.get("schemaVersion")) is not int
        or value.get("schemaVersion") != 1
    ):
        raise HarnessError(f"{role} keyring fields do not match schema v1")
    if value.get("trustDomain") != TRUST_DOMAIN:
        raise HarnessError(f"{role} keyring trust domain mismatch")
    if role == "health" and value.get("purpose") != "agent-update-health-attestation":
        raise HarnessError("health keyring purpose mismatch")
    keys = value.get("keys")
    if not isinstance(keys, dict) or not 1 <= len(keys) <= 32:
        raise HarnessError(f"{role} keyring must contain 1..32 keys")
    for key_id, encoded in keys.items():
        if not isinstance(key_id, str) or not KEY_ID_RE.fullmatch(key_id):
            raise HarnessError(f"{role} key id is invalid")
        if not isinstance(encoded, str):
            raise HarnessError(f"{role} public key is invalid")
        try:
            material = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HarnessError(f"{role} public key is not canonical base64") from exc
        if base64.b64encode(material).decode("ascii") != encoded:
            raise HarnessError(f"{role} public key is not canonical base64")
        if role != "release" and len(material) != 32:
            raise HarnessError(
                f"{role} key must be a canonical 32-byte Ed25519 public key"
            )
        if role == "release":
            try:
                from cryptography.hazmat.primitives import serialization
                from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                    Ed25519PublicKey,
                )

                if len(material) == 32:
                    public_key = Ed25519PublicKey.from_public_bytes(material)
                else:
                    public_key = serialization.load_der_public_key(material)
                if not isinstance(public_key, Ed25519PublicKey):
                    raise ValueError("release public key algorithm is not Ed25519")
                canonical_der = public_key.public_bytes(
                    encoding=serialization.Encoding.DER,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                )
                if len(material) != 32 and canonical_der != material:
                    raise ValueError("release public key DER is not canonical SPKI")
            except (ImportError, TypeError, ValueError) as exc:
                raise HarnessError(
                    "release key must be a raw 32-byte or canonical DER SPKI "
                    "Ed25519 public key"
                ) from exc
    if b"PRIVATE KEY" in payload or b'"seed"' in payload.lower():
        raise HarnessError("private-like key material is forbidden")
    return value


def validate_binding(payload: bytes, identity: Mapping[str, object]) -> dict[str, Any]:
    value = strict_json(payload, label="device binding")
    expected_fields = {
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
    if (
        set(value) != expected_fields
        or type(value.get("schemaVersion")) is not int
        or value.get("schemaVersion") != 1
    ):
        raise HarnessError("device binding fields do not match schema v1")
    expected_binding = {"schemaVersion": 1, "trustDomain": TRUST_DOMAIN, **identity}
    if value != expected_binding:
        raise HarnessError(
            "device binding does not match the immutable IQ9075 identity"
        )
    device_id = value.get("deviceId")
    space_id = value.get("spaceId")
    match = DEVICE_ID_RE.fullmatch(str(device_id))
    if (
        match is None
        or isinstance(space_id, bool)
        or not isinstance(space_id, int)
        or int(match.group(1)) != space_id
    ):
        raise HarnessError("device binding deviceId/spaceId pair is invalid")
    if (
        value.get("productModel") != "IQ9075_DEV"
        or value.get("platformProfile") != "iq9075_dev"
        or value.get("hardwareRevision") != "QCS9075-EVK"
        or value.get("architecture") != "aarch64"
        or value.get("dockerRequired") is not False
    ):
        raise HarnessError("device binding hardware identity is invalid")
    return value


def validate_manifest(value: Mapping[str, Any], *, run_id: str) -> None:
    if set(value) != {
        "schemaVersion",
        "protocolVersion",
        "runId",
        "toolSha256",
        "inputs",
        "destinations",
        "identity",
        "scenario",
    }:
        raise HarnessError("immutable manifest fields are invalid")
    if (
        type(value.get("schemaVersion")) is not int
        or value.get("schemaVersion") != 1
        or value.get("protocolVersion") != PROTOCOL_VERSION
        or value.get("runId") != run_id
        or not isinstance(value.get("toolSha256"), str)
        or not SHA256_RE.fullmatch(value["toolSha256"])
    ):
        raise HarnessError("immutable manifest identity is invalid")
    inputs = value.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
        "commandSha256",
        "releaseSha256",
        "healthSha256",
        "bindingSha256",
    }:
        raise HarnessError("immutable manifest input digests are invalid")
    if not all(
        isinstance(item, str) and SHA256_RE.fullmatch(item)
        for item in inputs.values()
    ):
        raise HarnessError("immutable manifest contains an invalid input digest")
    if value.get("destinations") != FIXED_DESTINATIONS:
        raise HarnessError("immutable manifest destinations are not fixed")
    identity = value.get("identity")
    if not isinstance(identity, dict) or set(identity) != {
        "deviceId",
        "spaceId",
        "productModel",
        "platformProfile",
        "hardwareRevision",
        "architecture",
        "dockerRequired",
    }:
        raise HarnessError("immutable manifest identity tuple is invalid")
    scenario = value.get("scenario")
    if not isinstance(scenario, dict) or set(scenario) != {
        "type",
        "expectedCommandId",
        "expectedBomDigest",
        "expectedCandidateSlot",
        "expectedPreviousSlot",
        "expectedPreviousVersion",
        "holdSeconds",
        "release",
    }:
        raise HarnessError("immutable manifest scenario is invalid")
    scenario_type = scenario.get("type")
    hold = scenario.get("holdSeconds")
    if scenario_type not in {"commit", "oak-fault-rollback"}:
        raise HarnessError("immutable manifest scenario type is invalid")
    if isinstance(hold, bool) or not isinstance(hold, int):
        raise HarnessError("immutable manifest holdSeconds is invalid")
    if (scenario_type == "commit" and hold != 0) or (
        scenario_type == "oak-fault-rollback"
        and not 0 <= hold <= MAX_FAULT_HOLD_SECONDS
    ):
        raise HarnessError("immutable manifest fault hold is invalid")
    expected_command_id = scenario.get("expectedCommandId")
    if not isinstance(expected_command_id, str):
        raise HarnessError("expected commandId is invalid")
    try:
        command_id = str(uuid.UUID(expected_command_id))
    except ValueError as exc:
        raise HarnessError("expected commandId is invalid") from exc
    if command_id != scenario.get("expectedCommandId"):
        raise HarnessError("expected commandId is not canonical")
    expected_bom_digest = scenario.get("expectedBomDigest")
    match = (
        DIGEST_RE.fullmatch(expected_bom_digest)
        if isinstance(expected_bom_digest, str)
        else None
    )
    if match is None:
        raise HarnessError("expected BOM digest is invalid")
    digest = match.group(1)
    if scenario.get("expectedCandidateSlot") != f"/opt/nuv-agent/releases/{digest}":
        raise HarnessError("expected candidate slot is not BOM-addressed")
    previous = scenario.get("expectedPreviousSlot")
    if not isinstance(previous, str) or not re.fullmatch(
        r"(?:releases/[0-9a-f]{64}|bootstrap/[0-9A-Za-z.+-]{1,64})", previous
    ):
        raise HarnessError("expected previous slot is invalid")
    if (
        not isinstance(scenario.get("expectedPreviousVersion"), str)
        or not SEMVER_RE.fullmatch(scenario["expectedPreviousVersion"])
    ):
        raise HarnessError("expected previous Agent version is invalid")
    release = scenario.get("release")
    if not isinstance(release, dict) or set(release) != {
        "agentVersion",
        "releaseSequence",
        "artifactDigest",
        "componentSha",
        "configSchema",
        "publisherKeyId",
    }:
        raise HarnessError("expected release identity is invalid")
    if (
        not isinstance(release.get("agentVersion"), str)
        or not SEMVER_RE.fullmatch(release["agentVersion"])
    ):
        raise HarnessError("expected Agent version is invalid")
    sequence = release.get("releaseSequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
        raise HarnessError("expected release sequence is invalid")
    if (
        not isinstance(release.get("artifactDigest"), str)
        or DIGEST_RE.fullmatch(release["artifactDigest"]) is None
    ):
        raise HarnessError("expected artifact digest is invalid")
    if (
        not isinstance(release.get("componentSha"), str)
        or COMPONENT_RE.fullmatch(release["componentSha"]) is None
    ):
        raise HarnessError("expected component SHA is invalid")
    if (
        not isinstance(release.get("configSchema"), str)
        or re.fullmatch(r"[1-9][0-9]*", release["configSchema"]) is None
    ):
        raise HarnessError("expected config schema is invalid")
    if (
        not isinstance(release.get("publisherKeyId"), str)
        or KEY_ID_RE.fullmatch(release["publisherKeyId"]) is None
    ):
        raise HarnessError("expected publisher key id is invalid")


def validate_candidate_bom(
    payload: bytes,
    *,
    manifest: Mapping[str, Any],
    bundle_sha256: str,
    bundle_size: int,
) -> dict[str, Any]:
    """Validate an unsigned copy against the already authenticated rollback release.

    The publisher trust anchor is the signed OTA lifecycle and its immutable release
    marker.  The candidate copy may therefore be used only after that exact release
    has reached ROLLED_BACK; it can never activate a production slot.
    """

    value = strict_json(payload, label="candidate BOM")
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
    if set(value) != expected_keys or type(value.get("schemaVersion")) is not int:
        raise HarnessError("candidate BOM fields are invalid")
    if value["schemaVersion"] != 2:
        raise HarnessError("candidate BOM must use schemaVersion 2")
    unsigned = dict(value)
    unsigned.pop("bomDigest")
    canonical = json.dumps(
        unsigned,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    computed = "sha256:" + sha256_bytes(canonical)
    scenario = manifest.get("scenario")
    if not isinstance(scenario, Mapping) or scenario.get("type") != "oak-fault-rollback":
        raise HarnessError("candidate soak requires an OAK rollback manifest")
    release = scenario.get("release")
    if not isinstance(release, Mapping):
        raise HarnessError("candidate soak release identity is invalid")
    if value.get("bomDigest") != computed or computed != scenario.get(
        "expectedBomDigest"
    ):
        raise HarnessError("candidate BOM digest differs from signed rollback")
    if (
        value.get("agentVersion") != release.get("agentVersion")
        or value.get("releaseSequence") != release.get("releaseSequence")
        or value.get("componentSha") != release.get("componentSha")
        or value.get("configSchema") != release.get("configSchema")
    ):
        raise HarnessError("candidate BOM release identity differs from rollback")
    if (
        not isinstance(value.get("bomId"), str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}", value["bomId"])
        is None
        or not isinstance(value.get("agentVersion"), str)
        or SEMVER_RE.fullmatch(value["agentVersion"]) is None
        or not isinstance(value.get("componentSha"), str)
        or COMPONENT_RE.fullmatch(value["componentSha"]) is None
        or not isinstance(value.get("configSchema"), str)
        or re.fullmatch(r"[1-9][0-9]*", value["configSchema"]) is None
        or not isinstance(value.get("releaseSequence"), int)
        or isinstance(value.get("releaseSequence"), bool)
        or not 1 <= value["releaseSequence"] <= 2**63 - 1
        or not isinstance(value.get("builtAt"), str)
        or len(value["builtAt"]) > 50
    ):
        raise HarnessError("candidate BOM identity value is invalid")
    minimum = value.get("minUpdaterVersion")
    if not isinstance(minimum, str) or SEMVER_RE.fullmatch(minimum) is None:
        raise HarnessError("candidate BOM minimum updater version is invalid")
    minimum_core = tuple(int(part) for part in minimum.split("-", 1)[0].split("."))
    required_core = tuple(
        int(part) for part in REQUIRED_UPDATER_VERSION.split("-", 1)[0].split(".")
    )
    if minimum_core > required_core:
        raise HarnessError("candidate BOM requires a newer updater")
    targets = value.get("targets")
    required_target = {
        "productModel": "IQ9075_DEV",
        "platformProfile": "iq9075_dev",
        "hardwareRevision": "QCS9075-EVK",
        "architecture": "aarch64",
    }
    if (
        not isinstance(targets, list)
        or required_target not in targets
        or any(not isinstance(target, dict) or set(target) != set(required_target) for target in targets)
        or targets != sorted(
            targets,
            key=lambda target: (
                target["productModel"],
                target["platformProfile"],
                target["hardwareRevision"],
                target["architecture"],
            ),
        )
        or len({json.dumps(target, sort_keys=True) for target in targets}) != len(targets)
    ):
        raise HarnessError("candidate BOM does not target IQ9075 exactly")
    artifact = value.get("artifact")
    if (
        not isinstance(artifact, dict)
        or set(artifact) != {"name", "kind", "sha256", "sizeBytes"}
        or not isinstance(artifact.get("name"), str)
        or Path(artifact["name"]).name != artifact["name"]
        or artifact["name"] in {"", ".", ".."}
        or artifact.get("kind") != "agent-bundle"
        or artifact.get("sha256") != bundle_sha256
        or artifact.get("sizeBytes") != bundle_size
        or release.get("artifactDigest") != f"sha256:{bundle_sha256}"
    ):
        raise HarnessError("candidate bundle identity differs from BOM")
    return value


class BoardHarness:
    def __init__(
        self,
        *,
        paths: BoardPaths | None = None,
        runner: CommandRunner | None = None,
        root_uid: int = 0,
        root_gid: int | None = None,
        nuvion_gid: int | None = None,
        nuvion_uid: int | None = None,
        tool_path: str | Path | None = None,
        enforce_installed_tool: bool = True,
        clock: Callable[[], str] = utc_now,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        disk_usage: Callable[[str | Path], Any] = shutil.disk_usage,
        usb_write_hook: Callable[[str, str], None] | None = None,
        controller_identity: Callable[[], Mapping[str, object]] | None = None,
    ) -> None:
        self.paths = paths or BoardPaths.from_root()
        self.runner = runner or CommandRunner()
        self.root_uid = root_uid
        self.root_gid = root_uid if root_gid is None else root_gid
        self.nuvion_gid = self._nuvion_gid() if nuvion_gid is None else nuvion_gid
        self.nuvion_uid = self._nuvion_uid() if nuvion_uid is None else nuvion_uid
        self.tool_path = Path(tool_path or __file__).resolve()
        self.enforce_installed_tool = enforce_installed_tool
        self.clock = clock
        self.monotonic = monotonic
        self.sleeper = sleeper
        self.disk_usage = disk_usage
        self.usb_write_hook = usb_write_hook
        self._portable_controller_start = time.monotonic_ns()
        self._portable_boot_id = str(uuid.uuid4())
        self.controller_identity = (
            controller_identity or self._local_controller_identity
        )
        self._package_maintenance_authorized = False

    @staticmethod
    def _read_linux_process_start(pid: int) -> int | None:
        try:
            raw = Path(f"/proc/{pid}/stat").read_text(
                encoding="ascii", errors="strict"
            ).strip()
        except (FileNotFoundError, PermissionError, OSError, UnicodeError):
            return None
        close = raw.rfind(")")
        fields = raw[close + 1 :].strip().split() if close > 0 else []
        if len(fields) <= 19 or not fields[19].isdigit():
            return None
        return int(fields[19])

    def _local_controller_identity(self) -> Mapping[str, object]:
        pid = os.getpid()
        start_ticks = self._read_linux_process_start(pid)
        linux_proc = Path("/proc").is_dir()
        if linux_proc and start_ticks is None:
            raise HarnessError("cannot bind candidate controller process identity")
        try:
            boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
                encoding="ascii", errors="strict"
            ).strip()
            boot_id = str(uuid.UUID(boot_id))
        except (FileNotFoundError, PermissionError, OSError, UnicodeError, ValueError):
            if linux_proc:
                raise HarnessError("cannot bind candidate controller boot identity")
            start_ticks = start_ticks or self._portable_controller_start
            boot_id = self._portable_boot_id
        if start_ticks is None:
            start_ticks = self._portable_controller_start
        return {"pid": pid, "startTicks": start_ticks, "bootId": boot_id}

    def _validate_controller_identity(
        self, value: object
    ) -> dict[str, object]:
        if (
            not isinstance(value, Mapping)
            or set(value) != {"pid", "startTicks", "bootId"}
            or type(value.get("pid")) is not int
            or value["pid"] < 2
            or type(value.get("startTicks")) is not int
            or value["startTicks"] < 1
            or not isinstance(value.get("bootId"), str)
        ):
            raise HarnessError("candidate controller identity is invalid")
        try:
            boot_id = str(uuid.UUID(value["bootId"]))
        except ValueError as exc:
            raise HarnessError("candidate controller boot identity is invalid") from exc
        if boot_id != value["bootId"]:
            raise HarnessError("candidate controller boot identity is invalid")
        return dict(value)

    def _controller_identity_is_live(self, identity: Mapping[str, object]) -> bool:
        checked = self._validate_controller_identity(identity)
        pid = int(checked["pid"])
        start_ticks = self._read_linux_process_start(pid)
        try:
            boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
                encoding="ascii", errors="strict"
            ).strip()
        except (FileNotFoundError, PermissionError, OSError, UnicodeError):
            current = self._validate_controller_identity(self.controller_identity())
            return checked == current and pid == os.getpid()
        return start_ticks == checked["startTicks"] and boot_id == checked["bootId"]

    def _fence_candidate_controller(self, identity: Mapping[str, object]) -> None:
        checked = self._validate_controller_identity(identity)
        pid = int(checked["pid"])
        if not self._controller_identity_is_live(checked):
            return
        if pid == os.getpid():
            raise HarnessError("candidate deadman cannot fence its own process")
        for signum, timeout in ((signal.SIGTERM, 10.0), (signal.SIGKILL, 10.0)):
            try:
                os.kill(pid, signum)
            except ProcessLookupError:
                return
            deadline = self.monotonic() + timeout
            while self.monotonic() < deadline:
                if not self._controller_identity_is_live(checked):
                    return
                self.sleeper(0.1)
        if self._controller_identity_is_live(checked):
            raise HarnessError("candidate controller did not terminate at the fence")

    @staticmethod
    def _nuvion_gid() -> int:
        import grp

        try:
            return grp.getgrnam("nuvion").gr_gid
        except KeyError as exc:
            raise HarnessError("required nuvion group is unavailable") from exc

    @staticmethod
    def _nuvion_uid() -> int:
        import pwd

        try:
            return pwd.getpwnam("nuvion").pw_uid
        except KeyError as exc:
            raise HarnessError("required nuvion user is unavailable") from exc

    def _prepare_state_root(self) -> None:
        base = self.paths.state_root.parent
        ensure_directory(base, mode=0o700, uid=self.root_uid, gid=self.root_gid)
        ensure_directory(
            self.paths.state_root, mode=0o700, uid=self.root_uid, gid=self.root_gid
        )

    def _prepare_run(self, run_id: str) -> Path:
        canonical_run_id(run_id)
        self._prepare_state_root()
        run_dir = self.paths.state_root / run_id
        ensure_directory(run_dir, mode=0o700, uid=self.root_uid, gid=self.root_gid)
        return run_dir

    def _state_path(self, run_id: str) -> Path:
        return self.paths.state_root / run_id / "run.json"

    def _load_state(self, run_id: str) -> dict[str, Any]:
        self._prepare_run(run_id)
        path = self._state_path(run_id)
        if not path.exists():
            state: dict[str, Any] = {
                "schemaVersion": 2,
                "protocolVersion": PROTOCOL_VERSION,
                "runId": run_id,
                "createdAt": self.clock(),
            }
            self._save_state(run_id, state)
            return state
        payload, metadata = read_regular(path, maximum=MAX_STATE_BYTES)
        if metadata.st_uid != self.root_uid or stat.S_IMODE(metadata.st_mode) != 0o600:
            raise HarnessError("run journal ownership or mode is unsafe")
        state = strict_json(payload, label="run journal")
        if (
            type(state.get("schemaVersion")) is not int
            or state.get("schemaVersion") != 2
            or state.get("protocolVersion") != PROTOCOL_VERSION
            or state.get("runId") != run_id
        ):
            raise HarnessError("run journal identity mismatch")
        return state

    def _save_state(self, run_id: str, state: Mapping[str, Any]) -> None:
        payload = (
            json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        if len(payload) > MAX_STATE_BYTES:
            raise HarnessError("run journal exceeds size limit")
        atomic_write(
            self._state_path(run_id),
            payload,
            mode=0o600,
            uid=self.root_uid,
            gid=self.root_gid,
        )

    @contextmanager
    def _lock_file(self, path: Path) -> Iterator[None]:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            path,
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise HarnessError("lock endpoint is unsafe")
            if (opened.st_uid, opened.st_gid) != (self.root_uid, self.root_gid):
                os.fchown(descriptor, self.root_uid, self.root_gid)
            os.fchmod(descriptor, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise HarnessError("another Fleet E2E operation owns the lock") from exc
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    @contextmanager
    def _transaction_writer_binding(self, run_id: str) -> Iterator[None]:
        path = self._state_path(run_id)
        if not path.exists() and not path.is_symlink():
            yield
            return
        state = self._read_existing_run_state(run_id)
        raw_guard = state.get("transactionGuard")
        if not isinstance(raw_guard, Mapping):
            yield
            return
        guard = self._validate_transaction_guard(run_id, raw_guard)
        if guard.get("armed") is not True:
            yield
            return
        epoch = str(uuid.uuid4())
        active_writer = {
            "writerEpoch": epoch,
            "controller": self._validate_controller_identity(
                self.controller_identity()
            ),
            "boundAt": self.clock(),
        }
        guard["activeWriter"] = active_writer
        state["transactionGuard"] = guard
        self._save_state(run_id, state)
        try:
            yield
        finally:
            try:
                latest = self._read_existing_run_state(run_id)
                latest_guard_raw = latest.get("transactionGuard")
                if isinstance(latest_guard_raw, Mapping):
                    latest_guard = self._validate_transaction_guard(
                        run_id, latest_guard_raw
                    )
                    current = latest_guard.get("activeWriter")
                    if (
                        isinstance(current, Mapping)
                        and current.get("writerEpoch") == epoch
                    ):
                        latest_guard.pop("activeWriter", None)
                        latest["transactionGuard"] = latest_guard
                        self._save_state(run_id, latest)
            except BaseException:
                # Preserve the durable binding for the recovery callback. The
                # primary operation failure must not erase fencing evidence.
                pass

    @contextmanager
    def _run_operation_lock(
        self,
        run_id: str,
        *,
        claim: bool = False,
        allow_unclaimed: bool = False,
    ) -> Iterator[None]:
        # The package maintainer and every Fleet operation serialize on the
        # global lock.  Recheck the durable barrier only after acquiring that
        # lock so an operation that began just before preinst cannot resume
        # after the maintainer has quiesced the protected writers.
        if (
            self._package_maintenance_is_active()
            and not self._package_maintenance_authorized
        ):
            raise HarnessError("package maintenance blocks Fleet E2E operations")
        self._prepare_state_root()
        run_path = self.paths.state_root / run_id
        run_preexisted = run_path.exists() or run_path.is_symlink()
        run_dir = self._prepare_run(run_id)
        if claim and not run_preexisted:
            # A canonical non-secret journal must precede the active lease.
            # Power loss can then be reconciled instead of leaving a lease
            # whose recovery identity has no durable journal.
            self._load_state(run_id)
            self._claim_active_run(run_id)
        with self._lock_file(run_dir / "operation.lock"):
            if claim and run_preexisted:
                self._claim_active_run(run_id)
            elif not claim:
                self._assert_active_run(run_id, allow_unclaimed=allow_unclaimed)
            with self._transaction_writer_binding(run_id):
                yield

    @contextmanager
    def _run_lock(
        self,
        run_id: str,
        *,
        usb: bool = False,
        claim: bool = False,
        allow_unclaimed: bool = False,
    ) -> Iterator[None]:
        # This precheck avoids creating the global lock endpoint during a
        # known maintenance window.  The authoritative race-free check is in
        # _run_operation_lock, after this lock has been acquired.
        if (
            self._package_maintenance_is_active()
            and not self._package_maintenance_authorized
        ):
            raise HarnessError("package maintenance blocks Fleet E2E operations")
        if claim and (
            self.paths.active_run.exists() or self.paths.active_run.is_symlink()
        ):
            # Reject a different owner before creating any directory that
            # could later be mistaken for crash-corrupt recovery state.
            self._assert_active_run(run_id, allow_unclaimed=True)
        self.paths.lock_root.mkdir(parents=True, exist_ok=True)
        with self._lock_file(self.paths.global_fleet_lock):
            if usb:
                with self._lock_file(self.paths.global_usb_lock):
                    with self._run_operation_lock(
                        run_id,
                        claim=claim,
                        allow_unclaimed=allow_unclaimed,
                    ):
                        yield
            else:
                with self._run_operation_lock(
                    run_id,
                    claim=claim,
                    allow_unclaimed=allow_unclaimed,
                ):
                    yield

    @contextmanager
    def _candidate_writer_lock(
        self, run_id: str, *, timeout: float = 0
    ) -> Iterator[None]:
        """Serialize the normal executor and the root-owned recovery writer."""

        path = self._prepare_run(run_id) / "candidate-writer.lock"
        descriptor = os.open(
            path,
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise HarnessError("candidate writer lock endpoint is unsafe")
            if (opened.st_uid, opened.st_gid) != (self.root_uid, self.root_gid):
                os.fchown(descriptor, self.root_uid, self.root_gid)
            os.fchmod(descriptor, 0o600)
            deadline = self.monotonic() + timeout
            while True:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError as exc:
                    if timeout <= 0 or self.monotonic() >= deadline:
                        raise HarnessError(
                            "candidate recovery writer did not reach a fenced boundary"
                        ) from exc
                    self.sleeper(0.1)
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
    def _assert_active_run(self, run_id: str, *, allow_unclaimed: bool) -> None:
        path = self.paths.active_run
        if not path.exists() and not path.is_symlink():
            if allow_unclaimed:
                return
            raise HarnessError("Fleet E2E run does not own the board lease")
        payload, metadata = read_regular(path, maximum=4096)
        if metadata.st_uid != self.root_uid or stat.S_IMODE(metadata.st_mode) != 0o600:
            raise HarnessError("Fleet E2E board lease ownership or mode is unsafe")
        lease = strict_json(payload, label="Fleet E2E board lease")
        if set(lease) != {"schemaVersion", "protocolVersion", "runId"} or (
            type(lease.get("schemaVersion")) is not int
            or lease.get("schemaVersion") != 1
            or lease.get("protocolVersion") != PROTOCOL_VERSION
            or lease.get("runId") != run_id
        ):
            raise HarnessError("another Fleet E2E run owns the board lease")

    def _claim_active_run(self, run_id: str) -> None:
        if (
            self.paths.package_maintenance.exists()
            or self.paths.package_maintenance.is_symlink()
        ):
            raise HarnessError("package maintenance blocks Fleet E2E claims")
        self._assert_active_run(run_id, allow_unclaimed=True)
        if self._existing_config_stream_run_id() is not None:
            raise HarnessError(
                "unfinished config-stream recovery blocks a new Fleet E2E claim"
            )
        if self.paths.active_run.exists():
            return
        # Lease absence is not sufficient ownership proof: a crash can occur
        # after unlink and before a guard's durable DISARMED record. Refuse a
        # new claimant until boot/explicit cleanup reconciles every old owner.
        states = self._scan_existing_runs()
        for historical_run, state in states.items():
            self._validate_existing_recovery_state(historical_run, state)
        if any(self._state_needs_boot_recovery(state) for state in states.values()):
            raise HarnessError(
                "unfinished Fleet recovery state requires reconciliation"
            )
        payload = (
            json.dumps(
                {
                    "schemaVersion": 1,
                    "protocolVersion": PROTOCOL_VERSION,
                    "runId": run_id,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        atomic_write(
            self.paths.active_run,
            payload,
            mode=0o600,
            uid=self.root_uid,
            gid=self.root_gid,
        )

    def _release_active_run(self, run_id: str) -> None:
        self._assert_active_run(run_id, allow_unclaimed=False)
        metadata = self.paths.active_run.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise HarnessError("Fleet E2E board lease endpoint is unsafe")
        self.paths.active_run.unlink()
        fsync_directory(self.paths.active_run.parent)

    def identity(self) -> dict[str, object]:
        expected = self.paths.rooted(str(TOOL_DESTINATION))
        if self.enforce_installed_tool and self.tool_path != expected:
            raise HarnessError("board tool is not running from the packaged fixed path")
        digest, metadata = sha256_regular(self.tool_path, maximum=2 * 1024 * 1024)
        if metadata.st_uid != self.root_uid or stat.S_IMODE(metadata.st_mode) != 0o755:
            raise HarnessError("board tool ownership or mode is invalid")
        return {
            "schemaVersion": 1,
            "protocolVersion": PROTOCOL_VERSION,
            "toolPath": str(TOOL_DESTINATION),
            "toolSha256": digest,
            "rootOwned": True,
            "mode": "0755",
        }

    def _systemctl(self, action: str, unit: str, *, timeout: float = 30) -> None:
        if unit not in UNITS or action not in {
            "start",
            "stop",
            "restart",
            "enable",
            "disable",
            "reset-failed",
        }:
            raise HarnessError("systemd operation is outside the fixed allowlist")
        result = self.runner.run(["/usr/bin/systemctl", action, unit], timeout=timeout)
        if result.returncode != 0:
            raise HarnessError("required systemd operation failed")

    def _unit_status(self, unit: str) -> dict[str, object]:
        if unit not in UNITS:
            raise HarnessError("unit is outside the fixed allowlist")
        active_state = self._unit_active_state(unit)
        enabled = self.runner.run(
            ["/usr/bin/systemctl", "is-enabled", unit], timeout=10
        )
        unit_file_state = enabled.stdout.strip()
        if unit_file_state not in {
            "enabled",
            "disabled",
            "static",
            "indirect",
            "masked",
            "generated",
            "transient",
            "alias",
        }:
            raise HarnessError("unit file state is unavailable or unsupported")
        pid = 0
        if unit.endswith(".service"):
            shown = self.runner.run(
                [
                    "/usr/bin/systemctl",
                    "show",
                    "--property=MainPID",
                    "--value",
                    unit,
                ],
                timeout=10,
            )
            try:
                pid = int(shown.stdout.strip()) if shown.returncode == 0 else 0
            except ValueError:
                pid = 0
        return {
            "active": active_state == "active",
            "enabled": enabled.returncode == 0
            and enabled.stdout.strip() in {"enabled", "static", "indirect"},
            "unitFileState": unit_file_state,
            "mainPid": pid,
        }

    def _unit_active_state(self, unit: str) -> str:
        if unit not in UNITS:
            raise HarnessError("unit is outside the fixed allowlist")
        result = self.runner.run(
            ["/usr/bin/systemctl", "is-active", unit], timeout=10
        )
        value = result.stdout.strip()
        if value not in {
            "active",
            "activating",
            "reloading",
            "deactivating",
            "inactive",
            "failed",
            "maintenance",
            "unknown",
        }:
            raise HarnessError("unit active state is unavailable or unsupported")
        return value

    def _unit_snapshot(self) -> dict[str, dict[str, object]]:
        return {unit: self._unit_status(unit) for unit in UNITS}

    def _agent_process_identity(self, expected_slot: str) -> dict[str, object]:
        status = self._unit_status("nuv-agent.service")
        pid = status.get("mainPid")
        if (
            status.get("active") is not True
            or isinstance(pid, bool)
            or not isinstance(pid, int)
            or pid < 2
        ):
            raise HarnessError("baseline Agent process is unavailable")
        stat_payload, _ = read_regular(
            self.paths.proc_root / str(pid) / "stat",
            maximum=16 * 1024,
            kernel_virtual_size=True,
        )
        raw_stat = stat_payload.decode("ascii", errors="strict").strip()
        close = raw_stat.rfind(")")
        fields = raw_stat[close + 1 :].strip().split() if close > 0 else []
        if len(fields) <= 19 or not fields[19].isdigit():
            raise HarnessError("baseline Agent process start time is invalid")
        environ_payload, _ = read_regular(
            self.paths.proc_root / str(pid) / "environ",
            maximum=1024 * 1024,
            kernel_virtual_size=True,
        )
        active_values = [
            item.split(b"=", 1)[1].decode("ascii", errors="strict")
            for item in environ_payload.split(b"\0")
            if item.startswith(b"NUVION_ACTIVE_SLOT=")
        ]
        if active_values != [expected_slot]:
            raise HarnessError("baseline Agent process slot does not match current")
        boot_payload, _ = read_regular(
            self.paths.boot_id,
            maximum=128,
            kernel_virtual_size=True,
        )
        boot_id = boot_payload.decode("ascii", errors="strict").strip()
        try:
            canonical_boot_id = str(uuid.UUID(boot_id))
        except ValueError as exc:
            raise HarnessError("board boot identity is invalid") from exc
        if canonical_boot_id != boot_id:
            raise HarnessError("board boot identity is not canonical")
        confirmed_status = self._unit_status("nuv-agent.service")
        confirmed_stat_payload, _ = read_regular(
            self.paths.proc_root / str(pid) / "stat",
            maximum=16 * 1024,
            kernel_virtual_size=True,
        )
        confirmed_raw_stat = confirmed_stat_payload.decode(
            "ascii", errors="strict"
        ).strip()
        confirmed_close = confirmed_raw_stat.rfind(")")
        confirmed_fields = (
            confirmed_raw_stat[confirmed_close + 1 :].strip().split()
            if confirmed_close > 0
            else []
        )
        if (
            confirmed_status.get("active") is not True
            or confirmed_status.get("mainPid") != pid
            or len(confirmed_fields) <= 19
            or confirmed_fields[19] != fields[19]
        ):
            raise HarnessError("baseline Agent process changed during inspection")
        return {
            "pid": pid,
            "startTicks": int(fields[19]),
            "bootId": boot_id,
            "activeSlot": active_values[0],
        }

    def _anti_replay_snapshot(self) -> dict[str, object]:
        metadata = self.paths.updater_state_db.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise HarnessError("updater anti-replay journal endpoint is unsafe")
        if metadata.st_uid != self.root_uid or metadata.st_mode & 0o022:
            raise HarnessError("updater anti-replay journal metadata is unsafe")
        try:
            connection = sqlite3.connect(
                f"file:{self.paths.updater_state_db}?mode=ro",
                uri=True,
                timeout=5.0,
            )
            try:
                connection.execute("PRAGMA query_only = ON")
                schema_version = int(
                    connection.execute("PRAGMA user_version").fetchone()[0]
                )
                maximum_command_sequence = int(
                    connection.execute(
                        "SELECT COALESCE(MAX(sequence), 0) FROM updater_command"
                    ).fetchone()[0]
                )
                meta = {
                    str(key): str(value)
                    for key, value in connection.execute(
                        "SELECT meta_key, meta_value FROM updater_meta "
                        "WHERE meta_key IN "
                        "('currentReleaseSequence','currentBomDigest') "
                        "ORDER BY meta_key"
                    ).fetchall()
                }
                terminal = connection.execute(
                    "SELECT command_id, sequence, phase, bom_digest, "
                    "release_sequence, health_deadline FROM updater_command "
                    "ORDER BY sequence DESC LIMIT 1"
                ).fetchone()
                semantic_tables: dict[str, object] = {}
                for table, order_by in (
                    ("updater_command", "sequence"),
                    ("updater_transition", "id"),
                    ("updater_meta", "meta_key"),
                    ("updater_commit_gate", "command_id"),
                ):
                    columns = [
                        str(row[1])
                        for row in connection.execute(
                            f"PRAGMA table_info({table})"
                        ).fetchall()
                    ]
                    if not columns:
                        raise HarnessError(
                            "updater anti-replay journal schema is incomplete"
                        )
                    rows = [
                        list(row)
                        for row in connection.execute(
                            f"SELECT * FROM {table} ORDER BY {order_by}"
                        ).fetchall()
                    ]
                    semantic_tables[table] = {"columns": columns, "rows": rows}
                semantic_sha256 = sha256_bytes(
                    json.dumps(
                        semantic_tables,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                )
            finally:
                connection.close()
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            raise HarnessError("updater anti-replay journal cannot be inspected") from exc
        latest = None
        if terminal is not None:
            latest = {
                "commandId": str(terminal[0]),
                "sequence": int(terminal[1]),
                "phase": str(terminal[2]),
                "bomDigest": str(terminal[3]),
                "releaseSequence": (
                    int(terminal[4]) if terminal[4] is not None else None
                ),
                "healthDeadline": (
                    str(terminal[5]) if terminal[5] is not None else None
                ),
            }
        return {
            "schemaVersion": schema_version,
            "semanticSha256": semantic_sha256,
            "maximumCommandSequence": maximum_command_sequence,
            "currentReleaseSequence": meta.get("currentReleaseSequence"),
            "currentBomDigest": meta.get("currentBomDigest"),
            "latest": latest,
        }

    @staticmethod
    def _validate_candidate_anti_replay(
        snapshot: Mapping[str, object], terminal: Mapping[str, object]
    ) -> None:
        sequence = terminal.get("sequence")
        if (
            set(snapshot)
            != {
                "schemaVersion",
                "semanticSha256",
                "maximumCommandSequence",
                "currentReleaseSequence",
                "currentBomDigest",
                "latest",
            }
            or snapshot.get("schemaVersion") != 4
            or not isinstance(snapshot.get("semanticSha256"), str)
            or SHA256_RE.fullmatch(str(snapshot["semanticSha256"])) is None
            or isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 1
            or snapshot.get("maximumCommandSequence") != sequence
            or snapshot.get("latest") != dict(terminal)
        ):
            raise HarnessError(
                "candidate soak anti-replay boundary differs from rollback"
            )

    def _restore_units(self, expected: Mapping[str, object]) -> None:
        # Fleet E2E never changes persistent enablement.  The baseline is
        # recorded only so cleanup can prove that no operation (or external
        # actor) changed it while development trust was installed.
        for unit in RESTART_ORDER:
            raw = expected.get(unit)
            if not isinstance(raw, Mapping):
                raise HarnessError("saved unit state is invalid")
            unit_file_state = raw.get("unitFileState")
            if unit_file_state not in {
                "enabled",
                "disabled",
                "static",
                "indirect",
                "masked",
                "generated",
                "transient",
                "alias",
            }:
                raise HarnessError("saved unit file state is invalid")
            live = self._unit_status(unit)
            if (
                live["enabled"] is not (raw.get("enabled") is True)
                or live["unitFileState"] != unit_file_state
            ):
                raise HarnessError("unit enablement changed during Fleet E2E")
            self._systemctl("start" if raw.get("active") is True else "stop", unit)
        for unit, raw in expected.items():
            if unit not in UNITS or not isinstance(raw, Mapping):
                raise HarnessError("saved unit state is invalid")
            live = self._unit_status(unit)
            if (
                live["active"] is not (raw.get("active") is True)
                or live["enabled"] is not (raw.get("enabled") is True)
                or live["unitFileState"] != raw.get("unitFileState")
            ):
                raise HarnessError("unit state restoration did not converge")

    def _restore_unit_enablement_for_boot(
        self, expected: Mapping[str, object]
    ) -> None:
        """Restore persistent unit policy without starting a protected writer."""

        for unit in STOP_ORDER:
            raw = expected.get(unit)
            if not isinstance(raw, Mapping):
                raise HarnessError("saved unit state is invalid")
            unit_file_state = raw.get("unitFileState")
            if unit_file_state not in {
                "enabled",
                "disabled",
                "static",
                "indirect",
                "masked",
                "generated",
                "transient",
                "alias",
            }:
                raise HarnessError("saved unit file state is invalid")
        for unit, raw in expected.items():
            if unit not in UNITS or not isinstance(raw, Mapping):
                raise HarnessError("saved unit state is invalid")
            live = self._unit_status(unit)
            if (
                live["active"] is True
                or live["enabled"] is not (raw.get("enabled") is True)
                or live["unitFileState"] != raw.get("unitFileState")
            ):
                raise HarnessError("boot unit policy restoration did not converge")

    def _stop_writers(self) -> None:
        failures: list[BaseException] = []
        for unit in STOP_ORDER:
            try:
                self._systemctl("stop", unit)
            except BaseException as exc:
                failures.append(exc)
        for unit in UNITS:
            try:
                if self._unit_active_state(unit) in {
                    "active",
                    "activating",
                    "reloading",
                    "deactivating",
                }:
                    failures.append(
                        HarnessError("writer unit did not stop for consistent backup")
                    )
            except BaseException as exc:
                failures.append(exc)
        if failures:
            raise HarnessError("protected writers did not stop for consistent backup")

    def _stop_writers_for_boot(self) -> None:
        """Avoid conflicting with start jobs queued behind the boot gate."""

        failures: list[BaseException] = []
        for unit in STOP_ORDER:
            try:
                state = self._unit_active_state(unit)
            except BaseException as exc:
                failures.append(exc)
                continue
            if state in {"active", "activating", "reloading", "deactivating"}:
                try:
                    self._systemctl("stop", unit)
                except BaseException as exc:
                    failures.append(exc)
        for unit in UNITS:
            try:
                if self._unit_active_state(unit) in {
                    "active",
                    "activating",
                    "reloading",
                    "deactivating",
                }:
                    failures.append(
                        HarnessError("writer unit remained active at the boot gate")
                    )
            except BaseException as exc:
                failures.append(exc)
        if failures:
            raise HarnessError("protected writers did not quiesce at the boot gate")

    def _probe_updater(self) -> dict[str, object]:
        python = self.paths.install_root / "current/venv/bin/python"
        result = self.runner.run(
            [
                "/usr/sbin/runuser",
                "-u",
                "nuvion",
                "--",
                str(python),
                "-s",
                "-c",
                UPDATER_PROBE,
            ],
            timeout=20,
        )
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            raise HarnessError("live updater capability probe returned no result")
        raw = strict_json(lines[-1].encode("utf-8"), label="updater capability")
        update = raw.get("update")
        safe_update: dict[str, object] | None = None
        if isinstance(update, Mapping):
            safe_update = {
                str(key): value
                for key, value in update.items()
                if str(key) in ALLOWED_UPDATE_FIELDS
                and (value is None or isinstance(value, (str, int, bool, float)))
            }
        safe: dict[str, object] = {
            "capabilityAvailable": raw.get("capabilityAvailable") is True,
            "authenticatedHelper": raw.get("authenticatedHelper") is True,
            "reason": str(raw.get("reason") or "")[:100],
            "updaterVersion": str(raw.get("updaterVersion") or "unknown"),
        }
        if safe_update is not None:
            safe["update"] = safe_update
        if (
            result.returncode != 0
            or safe["capabilityAvailable"] is not True
            or safe["authenticatedHelper"] is not True
            or safe["updaterVersion"] != REQUIRED_UPDATER_VERSION
        ):
            raise HarnessError("authenticated updater 0.2.0 capability is unavailable")
        return safe

    def _slot_link(self, name: str, *, required: bool) -> str | None:
        if name not in {"current", "previous"}:
            raise HarnessError("slot link name is invalid")
        path = self.paths.install_root / name
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            if required:
                raise HarnessError(f"required {name} slot is missing")
            return None
        if not stat.S_ISLNK(metadata.st_mode):
            raise HarnessError(f"{name} slot must be a symlink")
        if metadata.st_uid != self.root_uid:
            raise HarnessError(f"{name} slot must be root-owned")
        target = os.readlink(path)
        if not re.fullmatch(
            r"(?:releases/[0-9a-f]{64}|bootstrap/[0-9A-Za-z.+-]{1,64})", target
        ):
            raise HarnessError(f"{name} slot target is invalid")
        return target

    def _release_marker(self, target: str) -> dict[str, object]:
        if not re.fullmatch(r"releases/[0-9a-f]{64}", target):
            raise HarnessError("release marker requires a content-addressed slot")
        digest = target.split("/", 1)[1]
        path = self.paths.install_root / target / ".nuvion/release.json"
        payload, metadata = read_regular(path, maximum=64 * 1024)
        if metadata.st_uid != self.root_uid or stat.S_IMODE(metadata.st_mode) & 0o022:
            raise HarnessError("release marker ownership or mode is unsafe")
        value = strict_json(payload, label="release marker")
        if (
            set(value) != RELEASE_FIELDS
            or type(value.get("schemaVersion")) is not int
            or value.get("schemaVersion") != 2
        ):
            raise HarnessError("release marker fields are invalid")
        if value.get("bomDigest") != f"sha256:{digest}":
            raise HarnessError("release marker does not match its slot digest")
        if (
            not isinstance(value.get("agentVersion"), str)
            or not SEMVER_RE.fullmatch(value["agentVersion"])
        ):
            raise HarnessError("release marker Agent version is invalid")
        sequence = value.get("releaseSequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
            raise HarnessError("release marker sequence is invalid")
        if (
            not isinstance(value.get("artifactDigest"), str)
            or DIGEST_RE.fullmatch(value["artifactDigest"]) is None
        ):
            raise HarnessError("release marker artifact digest is invalid")
        if (
            not isinstance(value.get("componentSha"), str)
            or COMPONENT_RE.fullmatch(value["componentSha"]) is None
        ):
            raise HarnessError("release marker component SHA is invalid")
        if (
            not isinstance(value.get("configSchema"), str)
            or re.fullmatch(r"[1-9][0-9]*", value["configSchema"]) is None
        ):
            raise HarnessError("release marker config schema is invalid")
        if (
            not isinstance(value.get("publisherKeyId"), str)
            or KEY_ID_RE.fullmatch(value["publisherKeyId"]) is None
        ):
            raise HarnessError("release marker publisher id is invalid")
        return {key: value[key] for key in sorted(RELEASE_FIELDS)}

    def _slot_version(self, target: str) -> str:
        if target.startswith("bootstrap/"):
            version = target.split("/", 1)[1]
            if not SEMVER_RE.fullmatch(version):
                raise HarnessError("bootstrap slot version is invalid")
            return version
        return str(self._release_marker(target)["agentVersion"])

    def _usb_device_path(self, port: str) -> Path:
        return self.paths.usb_devices / canonical_oak_port(port)

    def _oak_usb_devices(self) -> list[Path]:
        devices_root = self.paths.usb_devices
        metadata = devices_root.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise HarnessError("USB sysfs devices root is unsafe")
        sys_root = (self.paths.root / "sys").resolve(strict=True)
        matches: list[Path] = []
        for device in sorted(devices_root.iterdir(), key=lambda item: item.name):
            if USB_TOPOLOGY_RE.fullmatch(device.name) is None:
                continue
            metadata = device.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                try:
                    resolved = device.resolve(strict=True)
                except OSError as exc:
                    raise HarnessError(
                        "OAK USB topology endpoint is unavailable"
                    ) from exc
                if not resolved.is_dir() or not resolved.is_relative_to(sys_root):
                    raise HarnessError("OAK USB topology endpoint is unsafe")
            elif not stat.S_ISDIR(metadata.st_mode):
                raise HarnessError("OAK USB topology endpoint is unsafe")
            vendor_path = device / "idVendor"
            product_path = device / "idProduct"
            if not vendor_path.exists() or not product_path.exists():
                continue
            vendor_payload, _ = read_regular(
                vendor_path,
                maximum=128,
                kernel_virtual_size=True,
            )
            product_payload, _ = read_regular(
                product_path,
                maximum=128,
                kernel_virtual_size=True,
            )
            vendor = vendor_payload.decode("ascii", errors="strict").strip().lower()
            product = product_payload.decode("ascii", errors="strict").strip().lower()
            if (vendor, product) == (OAK_VENDOR, OAK_PRODUCT):
                matches.append(device)
        if len(matches) != 1:
            raise HarnessError(
                "USB1 downstream must contain exactly one OAK-D Lite"
            )
        return matches

    def verify_oak(
        self,
        *,
        require_bound: bool = True,
        expected_port: str | None = None,
    ) -> dict[str, object]:
        expected = canonical_oak_port(expected_port) if expected_port else None
        device = self._oak_usb_devices()[0]
        if expected is not None and device.name != expected:
            raise HarnessError("OAK USB topology changed from the journaled port")
        metadata = device.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            resolved = device.resolve(strict=True)
            sys_root = (self.paths.root / "sys").resolve(strict=True)
            if not resolved.is_dir() or not resolved.is_relative_to(sys_root):
                raise HarnessError("OAK USB topology endpoint is unsafe")
        elif not stat.S_ISDIR(metadata.st_mode):
            raise HarnessError("OAK USB topology endpoint is unavailable")

        def text(name: str) -> str:
            payload, _ = read_regular(
                device / name,
                maximum=128,
                kernel_virtual_size=True,
            )
            return payload.decode("ascii", errors="strict").strip().lower()

        vendor = text("idVendor")
        product = text("idProduct")
        if (vendor, product) != (OAK_VENDOR, OAK_PRODUCT):
            raise HarnessError("USB topology is not the expected OAK-D Lite")
        try:
            speed = float(text("speed"))
        except ValueError as exc:
            raise HarnessError("OAK USB speed is invalid") from exc
        if speed < OAK_MIN_SPEED_MBPS:
            raise HarnessError("OAK-D Lite must negotiate at 5Gbps")
        mxid = text("serial")
        if re.fullmatch(r"[a-z0-9._:-]{1,128}", mxid) is None:
            raise HarnessError("OAK-D Lite sysfs serial/MXID is invalid")
        driver = device / "driver"
        bound = driver.is_symlink()
        if require_bound:
            expected_driver = (self.paths.root / "sys/bus/usb/drivers/usb").resolve(
                strict=True
            )
            if not bound or driver.resolve(strict=True) != expected_driver:
                raise HarnessError("OAK-D Lite is not bound to the exact USB driver")
        return {
            "port": device.name,
            "vendorId": vendor,
            "productId": product,
            "speedMbps": int(speed),
            "mxidSha256": hashlib.sha256(mxid.encode("utf-8")).hexdigest(),
            "attached": True,
            "bound": bound,
        }

    def _foundation(self) -> dict[str, object]:
        os_payload, _ = read_regular(self.paths.os_release, maximum=16 * 1024)
        os_values: dict[str, str] = {}
        for line in os_payload.decode("utf-8", errors="strict").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                os_values[key] = value.strip().strip('"')
        if os_values.get("ID") != "ubuntu" or os_values.get("VERSION_ID") != "24.04":
            raise HarnessError("IQ9075 E2E requires Ubuntu 24.04")
        model_payload, _ = read_regular(self.paths.device_model, maximum=4096)
        model = model_payload.decode("utf-8", errors="replace").strip("\x00\n ")
        if re.search(r"(?:qcs9075|iq[- ]?9075)", model, re.IGNORECASE) is None:
            raise HarnessError("board model is not IQ9075/QCS9075")
        architecture = self.runner.run(
            ["/usr/bin/dpkg", "--print-architecture"], timeout=10
        )
        if architecture.returncode != 0 or architecture.stdout.strip() != "arm64":
            raise HarnessError("IQ9075 E2E requires arm64")
        usage = self.disk_usage(self.paths.state_root.parent)
        free_bytes = int(getattr(usage, "free", 0))
        if free_bytes < MIN_FREE_BYTES:
            raise HarnessError("IQ9075 E2E requires at least 2 GiB free space")
        current = self._slot_link("current", required=True)
        if current is None:
            raise HarnessError("current slot is required")
        previous = self._slot_link("previous", required=False)
        current_marker = (
            self._release_marker(current) if current.startswith("releases/") else None
        )
        return {
            "board": {
                "profile": "IQ9075_DEV",
                "model": model[:120],
                "architecture": "arm64",
                "os": "ubuntu-24.04",
            },
            "freeBytes": free_bytes,
            "oak": self.verify_oak(),
            "slots": {
                "current": current,
                "previous": previous,
                "currentVersion": self._slot_version(current),
                "currentRelease": current_marker,
            },
        }

    def preflight(self, run_id: str) -> dict[str, object]:
        with self._run_lock(run_id, claim=True):
            state = self._load_state(run_id)
            foundation = self._foundation()
            transaction = state.get("trustTransaction")
            if transaction is None:
                state["foundation"] = {
                    "verified": True,
                    "verifiedAt": self.clock(),
                    "currentSlot": foundation["slots"]["current"],
                    "currentVersion": foundation["slots"]["currentVersion"],
                    "previousSlot": foundation["slots"]["previous"],
                }
                self._save_state(run_id, state)
            elif not isinstance(transaction, Mapping):
                raise HarnessError("trust transaction journal is invalid")
            elif (
                not isinstance(state.get("foundation"), Mapping)
                or state["foundation"].get("verified") is not True
            ):
                raise HarnessError("transaction exists without a verified baseline")
            recorded = state["foundation"]
            return {
                "schemaVersion": 1,
                "runId": run_id,
                "foundation": foundation,
                "recordedBaseline": {
                    "slot": recorded["currentSlot"],
                    "version": recorded["currentVersion"],
                },
                "transactionPresent": transaction is not None,
                "verified": True,
            }

    class _HashingReader:
        def __init__(self, source: io.BufferedReader) -> None:
            self.source = source
            self.digest = hashlib.sha256()

        def read(self, size: int = -1) -> bytes:
            chunk = self.source.read(size)
            self.digest.update(chunk)
            return chunk

    def _archive_entry(
        self,
        archive: tarfile.TarFile,
        path: Path,
        arcname: str,
        budget: dict[str, int],
        integrity: dict[str, dict[str, object]],
    ) -> None:
        metadata = path.lstat()
        budget["entries"] += 1
        if budget["entries"] > MAX_BACKUP_ENTRIES:
            raise HarnessError("recovery archive entry limit exceeded")
        info = tarfile.TarInfo(arcname)
        info.mode = stat.S_IMODE(metadata.st_mode)
        info.uid = metadata.st_uid
        info.gid = metadata.st_gid
        info.uname = ""
        info.gname = ""
        info.mtime = int(metadata.st_mtime)
        if stat.S_ISLNK(metadata.st_mode):
            target = os.readlink(path)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            archive.addfile(info)
            integrity[arcname] = {"type": "symlink", "target": target}
            return
        if stat.S_ISDIR(metadata.st_mode):
            info.type = tarfile.DIRTYPE
            archive.addfile(info)
            integrity[arcname] = {"type": "directory"}
            with os.scandir(path) as entries:
                for entry in sorted(entries, key=lambda item: item.name):
                    self._archive_entry(
                        archive,
                        Path(entry.path),
                        f"{arcname.rstrip('/')}/{entry.name}",
                        budget,
                        integrity,
                    )
            return
        if not stat.S_ISREG(metadata.st_mode):
            raise HarnessError(f"unsupported recovery entry type: {path.name}")
        budget["bytes"] += metadata.st_size
        if budget["bytes"] > MAX_BACKUP_BYTES:
            raise HarnessError("recovery archive byte limit exceeded")
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino, opened.st_size) != (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_size,
            ):
                raise HarnessError("recovery entry changed while opening")
            info.type = tarfile.REGTYPE
            info.size = opened.st_size
            with os.fdopen(os.dup(descriptor), "rb") as source:
                hashing = self._HashingReader(source)
                archive.addfile(info, hashing)
                digest = hashing.digest.hexdigest()
            after = os.fstat(descriptor)
            if (after.st_dev, after.st_ino, after.st_size) != (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
            ):
                raise HarnessError("recovery entry changed while reading")
            integrity[arcname] = {
                "type": "regular",
                "size": opened.st_size,
                "sha256": digest,
            }
        finally:
            os.close(descriptor)

    def _write_archive(self, run_id: str, archive_path: Path) -> str:
        if archive_path.exists() or archive_path.is_symlink():
            metadata = archive_path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise HarnessError("partial recovery archive is unsafe")
            archive_path.unlink()
            fsync_directory(archive_path.parent)
        descriptor = os.open(
            archive_path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        integrity: dict[str, dict[str, object]] = {}
        missing: list[str] = []
        try:
            opened = os.fstat(descriptor)
            if (opened.st_uid, opened.st_gid) != (self.root_uid, self.root_gid):
                os.fchown(descriptor, self.root_uid, self.root_gid)
            os.fchmod(descriptor, 0o600)
            with os.fdopen(os.dup(descriptor), "wb") as stream:
                with tarfile.open(
                    fileobj=stream, mode="w", format=tarfile.PAX_FORMAT
                ) as archive:
                    archive.dereference = False
                    budget = {"entries": 0, "bytes": 0}
                    for absolute in BACKUP_PATHS:
                        target = self.paths.rooted(absolute)
                        if not target.exists() and not target.is_symlink():
                            missing.append(absolute)
                            continue
                        self._archive_entry(
                            archive,
                            target,
                            "rootfs/" + absolute.lstrip("/"),
                            budget,
                            integrity,
                        )
                    manifest = {
                        "schemaVersion": 1,
                        "runId": run_id,
                        "entries": integrity,
                        "missing": sorted(missing),
                    }
                    payload = (
                        json.dumps(manifest, sort_keys=True, separators=(",", ":"))
                        + "\n"
                    ).encode("utf-8")
                    info = tarfile.TarInfo("recovery-integrity.json")
                    info.mode = 0o600
                    info.uid = self.root_uid
                    info.gid = self.root_gid
                    info.size = len(payload)
                    info.mtime = int(time.time())
                    archive.addfile(info, io.BytesIO(payload))
                stream.flush()
                os.fsync(stream.fileno())
            os.fsync(descriptor)
        except BaseException:
            os.close(descriptor)
            try:
                archive_path.unlink()
            except FileNotFoundError:
                pass
            raise
        else:
            os.close(descriptor)
        fsync_directory(archive_path.parent)
        digest, _ = sha256_regular(
            archive_path, maximum=MAX_BACKUP_BYTES + 256 * 1024 * 1024
        )
        self._verify_archive(run_id, archive_path, digest)
        return digest

    def _verify_archive(self, run_id: str, archive_path: Path, digest: str) -> None:
        actual, metadata = sha256_regular(
            archive_path, maximum=MAX_BACKUP_BYTES + 256 * 1024 * 1024
        )
        if (
            actual != digest
            or metadata.st_uid != self.root_uid
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise HarnessError("emergency recovery archive integrity failed")
        with tarfile.open(archive_path, "r") as archive:
            members = archive.getmembers()
            control_members = [
                item for item in members if item.name == "recovery-integrity.json"
            ]
            if len(control_members) != 1:
                raise HarnessError("recovery archive integrity manifest is missing")
            control = archive.extractfile(control_members[0])
            if control is None:
                raise HarnessError("recovery archive integrity manifest is unreadable")
            manifest = strict_json(
                control.read(MAX_STATE_BYTES), label="recovery integrity"
            )
            if (
                set(manifest) != {"schemaVersion", "runId", "entries", "missing"}
                or type(manifest.get("schemaVersion")) is not int
                or manifest.get("schemaVersion") != 1
                or manifest.get("runId") != run_id
                or not isinstance(manifest.get("entries"), dict)
                or not isinstance(manifest.get("missing"), list)
            ):
                raise HarnessError("recovery integrity manifest is invalid")
            entries = manifest["entries"]
            by_name = {item.name: item for item in members}
            if set(by_name) != {*entries, "recovery-integrity.json"}:
                raise HarnessError("recovery archive member set is invalid")
            for name, expected in entries.items():
                if not isinstance(expected, Mapping):
                    raise HarnessError("recovery integrity entry is invalid")
                member = by_name[name]
                entry_type = expected.get("type")
                if entry_type == "regular":
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise HarnessError("recovery regular member is unreadable")
                    member_digest = hashlib.sha256()
                    total = 0
                    while True:
                        chunk = stream.read(1024 * 1024)
                        if not chunk:
                            break
                        total += len(chunk)
                        member_digest.update(chunk)
                    if total != expected.get(
                        "size"
                    ) or member_digest.hexdigest() != expected.get("sha256"):
                        raise HarnessError("recovery regular member digest mismatch")
                elif entry_type == "directory" and not member.isdir():
                    raise HarnessError("recovery directory member mismatch")
                elif entry_type == "symlink" and (
                    not member.issym() or member.linkname != expected.get("target")
                ):
                    raise HarnessError("recovery symlink member mismatch")
                elif entry_type not in {"regular", "directory", "symlink"}:
                    raise HarnessError("recovery member type is invalid")

    def backup(self, run_id: str) -> dict[str, object]:
        with self._run_lock(run_id):
            state = self._load_state(run_id)
            if state.get("foundation", {}).get("verified") is not True:
                raise HarnessError("foundation preflight must complete before backup")
            # The recovery archive contains production secrets.  Arm the
            # transaction-wide owner before the first archive/snapshot byte is
            # written, not merely before development trust is applied.
            self._ensure_transaction_guard(run_id, state)
            ensure_directory(
                self.paths.recovery_root,
                mode=0o700,
                uid=self.root_uid,
                gid=self.root_gid,
            )
            archive_path = self.paths.recovery_root / f"iq9075-{run_id}.tar"
            existing = state.get("backup")
            resumed = existing is not None
            if existing is None:
                units_before = self._unit_snapshot()
                backup_state: dict[str, object] = {
                    "phase": "PREPARED",
                    "complete": False,
                    "unitsBefore": units_before,
                    "preparedAt": self.clock(),
                }
                state["backup"] = backup_state
                self._save_state(run_id, state)
            elif not isinstance(existing, dict):
                raise HarnessError("backup journal is invalid")
            else:
                backup_state = existing
                units_before = backup_state.get("unitsBefore")
                if (
                    backup_state.get("phase") not in PHASES
                    or not isinstance(backup_state.get("complete"), bool)
                    or not isinstance(units_before, Mapping)
                    or set(units_before) != set(UNITS)
                    or (
                        "archivePath" in backup_state
                        and backup_state.get("archivePath") != str(archive_path)
                    )
                ):
                    raise HarnessError("backup journal fields are invalid")
            if backup_state.get("complete") is True:
                if (
                    backup_state.get("phase") != "RESTORED"
                    or not SHA256_RE.fullmatch(str(backup_state.get("sha256") or ""))
                    or backup_state.get("archivePath") != str(archive_path)
                    or backup_state.get("verified") is not True
                ):
                    raise HarnessError("completed backup journal is invalid")
                self._verify_archive(
                    run_id, archive_path, str(backup_state.get("sha256"))
                )
                return {
                    "schemaVersion": 1,
                    "runId": run_id,
                    "phase": "RESTORED",
                    "emergencyArchiveVerified": True,
                    "sha256": backup_state["sha256"],
                    "idempotent": True,
                }
            try:
                if backup_state["phase"] in {"PREPARED", "APPLYING"}:
                    backup_state["phase"] = "APPLYING"
                    backup_state["applyingAt"] = self.clock()
                    state["backup"] = backup_state
                    self._save_state(run_id, state)
                    self._stop_writers()
                    digest = self._write_archive(run_id, archive_path)
                    backup_state.update(
                        {
                            "phase": "APPLIED",
                            "archivePath": str(archive_path),
                            "sha256": digest,
                            "verified": True,
                            "appliedAt": self.clock(),
                        }
                    )
                    state["backup"] = backup_state
                    self._save_state(run_id, state)
                elif backup_state["phase"] in {"APPLIED", "RESTORING"}:
                    digest = str(backup_state.get("sha256") or "")
                    if not SHA256_RE.fullmatch(digest):
                        raise HarnessError("backup archive digest is invalid")
                    self._verify_archive(run_id, archive_path, digest)
                else:
                    raise HarnessError("incomplete backup phase is invalid")
                if backup_state["phase"] == "APPLIED":
                    backup_state["phase"] = "RESTORING"
                    backup_state["restoringAt"] = self.clock()
                    state["backup"] = backup_state
                    self._save_state(run_id, state)
                self._restore_units(units_before)
                backup_state.update(
                    {
                        "phase": "RESTORED",
                        "complete": True,
                        "restoredAt": self.clock(),
                    }
                )
                state["backup"] = backup_state
                self._save_state(run_id, state)
            except BaseException:
                try:
                    self._restore_units(units_before)
                except Exception:  # noqa: BLE001 - any restore failure stops Agent.
                    self._stop_agent_fail_closed()
                raise
            return {
                "schemaVersion": 1,
                "runId": run_id,
                "phase": "RESTORED",
                "emergencyArchiveVerified": True,
                "sha256": digest,
                "idempotent": resumed,
            }

    @staticmethod
    def _render_config(payload: bytes) -> bytes:
        try:
            lines = payload.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise HarnessError("Agent config is not UTF-8") from exc
        updates = {
            "NUVION_FLEET_COMMAND_KEYRING_PATH": AGENT_COMMAND_KEYRING,
            "NUVION_FLEET_COMMAND_ENABLED": "true",
        }
        seen: set[str] = set()
        result: list[str] = []
        for line in lines:
            stripped = line.lstrip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                key = stripped.split("=", 1)[0].strip()
                if key in updates:
                    if key not in seen:
                        result.append(f"{key}={updates[key]}")
                        seen.add(key)
                    continue
            result.append(line)
        for key, value in updates.items():
            if key not in seen:
                result.append(f"{key}={value}")
        return ("\n".join(result) + "\n").encode("utf-8")

    @staticmethod
    def _configured_device_identity(payload: bytes) -> tuple[str, int]:
        try:
            lines = payload.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise HarnessError("Agent config is not UTF-8") from exc
        values: dict[str, str] = {}
        wanted = {
            "NUVION_DEVICE_ID",
            "NUVION_DEVICE_USERNAME",
            "NUVION_SPACE_ID",
        }
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            if key not in wanted:
                continue
            if key in values:
                raise HarnessError("Agent config contains duplicate device identity")
            values[key] = value.strip()
        device_id = values.get("NUVION_DEVICE_ID") or values.get(
            "NUVION_DEVICE_USERNAME"
        )
        username = values.get("NUVION_DEVICE_USERNAME")
        if (
            device_id is None
            or (username is not None and username != device_id)
            or DEVICE_ID_RE.fullmatch(device_id) is None
        ):
            raise HarnessError("Agent config device identity is missing or invalid")
        try:
            space_id = int(values.get("NUVION_SPACE_ID") or "")
        except ValueError as exc:
            raise HarnessError("Agent config space identity is invalid") from exc
        match = DEVICE_ID_RE.fullmatch(device_id)
        assert match is not None
        if space_id < 1 or int(match.group(1)) != space_id:
            raise HarnessError("Agent config device/space identity does not match")
        return device_id, space_id

    def _staging_path(self, run_id: str, role: str) -> Path:
        if role not in INPUT_ROLES:
            raise HarnessError("staging role is invalid")
        return Path(f"/tmp/nuvion-fleet-e2e-{run_id}-{role}.json")

    def _read_staged(
        self, run_id: str, role: str, path: str | Path, expected_sha256: str
    ) -> StagedInput:
        expected_path = self._staging_path(run_id, role)
        candidate = Path(path)
        if candidate != expected_path or not SHA256_RE.fullmatch(expected_sha256):
            raise HarnessError("staging path or digest is invalid")
        payload, metadata = read_regular(candidate, maximum=MAX_TRUST_BYTES)
        digest = sha256_bytes(payload)
        if digest != expected_sha256:
            raise HarnessError("staged input digest mismatch")
        return StagedInput(
            role,
            candidate,
            payload,
            digest,
            metadata.st_dev,
            metadata.st_ino,
        )

    @staticmethod
    def _cleanup_staged(inputs: Sequence[StagedInput]) -> None:
        for item in inputs:
            try:
                metadata = item.path.lstat()
                if stat.S_ISREG(metadata.st_mode) and (
                    metadata.st_dev,
                    metadata.st_ino,
                ) == (item.device, item.inode):
                    digest, _ = sha256_regular(item.path, maximum=MAX_TRUST_BYTES)
                    if digest == item.sha256:
                        item.path.unlink()
                        fsync_directory(item.path.parent)
            except FileNotFoundError:
                continue

    def discard_staging(
        self,
        run_id: str,
        *,
        command_sha256: str,
        release_sha256: str,
        health_sha256: str,
        binding_sha256: str,
        manifest_sha256: str,
    ) -> dict[str, object]:
        with self._run_lock(run_id):
            removed: list[str] = []
            for role, digest in (
                ("command", command_sha256),
                ("release", release_sha256),
                ("health", health_sha256),
                ("binding", binding_sha256),
                ("manifest", manifest_sha256),
            ):
                if not SHA256_RE.fullmatch(digest):
                    raise HarnessError("staging cleanup digest is invalid")
                path = self._staging_path(run_id, role)
                if not path.exists() and not path.is_symlink():
                    continue
                actual, metadata = sha256_regular(path, maximum=MAX_TRUST_BYTES)
                if actual != digest:
                    raise HarnessError("staging cleanup digest mismatch")
                if not stat.S_ISREG(metadata.st_mode):
                    raise HarnessError("staging cleanup target is unsafe")
                path.unlink()
                fsync_directory(path.parent)
                removed.append(role)
            return {
                "schemaVersion": 1,
                "runId": run_id,
                "removed": sorted(removed),
                "complete": True,
            }

    def _candidate_staging_path(self, run_id: str, role: str) -> Path:
        if role == "candidate-bundle":
            suffix = "candidate-bundle.tar.gz"
        elif role == "candidate-bom":
            suffix = "candidate-bom.json"
        elif role == "oak-harness":
            suffix = "oak-harness.sh"
        else:
            raise HarnessError("candidate staging role is invalid")
        return Path(f"/tmp/nuvion-fleet-e2e-{run_id}-{suffix}")

    def _read_candidate_input(
        self,
        run_id: str,
        role: str,
        path: str | Path,
        expected_sha256: str,
    ) -> CandidateInput:
        expected_path = self._candidate_staging_path(run_id, role)
        candidate = Path(path)
        if candidate != expected_path or SHA256_RE.fullmatch(expected_sha256) is None:
            raise HarnessError("candidate staging path or digest is invalid")
        maximum = (
            MAX_CANDIDATE_BUNDLE_BYTES
            if role == "candidate-bundle"
            else MAX_TRUST_BYTES
            if role == "candidate-bom"
            else 2 * 1024 * 1024
        )
        digest, metadata = sha256_regular(candidate, maximum=maximum)
        if digest != expected_sha256:
            raise HarnessError("candidate staged input digest mismatch")
        if metadata.st_mode & 0o022 or metadata.st_nlink != 1:
            raise HarnessError("candidate staged input metadata is unsafe")
        return CandidateInput(
            role=role,
            path=candidate,
            sha256=digest,
            size=metadata.st_size,
            device=metadata.st_dev,
            inode=metadata.st_ino,
        )

    @staticmethod
    def _cleanup_candidate_inputs(inputs: Sequence[CandidateInput]) -> None:
        for item in inputs:
            maximum = (
                MAX_CANDIDATE_BUNDLE_BYTES
                if item.role == "candidate-bundle"
                else MAX_TRUST_BYTES
                if item.role == "candidate-bom"
                else 2 * 1024 * 1024
            )
            try:
                digest, metadata = sha256_regular(item.path, maximum=maximum)
                if (
                    (metadata.st_dev, metadata.st_ino) == (item.device, item.inode)
                    and digest == item.sha256
                ):
                    item.path.unlink()
                    fsync_directory(item.path.parent)
            except FileNotFoundError:
                continue

    def discard_candidate_staging(
        self,
        run_id: str,
        *,
        bundle_sha256: str,
        bom_sha256: str,
        harness_sha256: str,
    ) -> dict[str, object]:
        with self._run_lock(run_id):
            removed: list[str] = []
            for role, digest in (
                ("candidate-bundle", bundle_sha256),
                ("candidate-bom", bom_sha256),
                ("oak-harness", harness_sha256),
            ):
                if SHA256_RE.fullmatch(digest) is None:
                    raise HarnessError("candidate staging cleanup digest is invalid")
                path = self._candidate_staging_path(run_id, role)
                if not path.exists() and not path.is_symlink():
                    continue
                item = self._read_candidate_input(run_id, role, path, digest)
                self._cleanup_candidate_inputs((item,))
                removed.append(role)
            return {
                "schemaVersion": 1,
                "runId": run_id,
                "removed": sorted(removed),
                "complete": True,
            }

    def _transaction_dir(self, run_id: str) -> Path:
        return self.paths.state_root / run_id / "trust-transaction"

    def _validate_transaction_shape(self, transaction: Mapping[str, Any]) -> None:
        allowed = {
            "phase",
            "manifestSha256",
            "files",
            "directories",
            "unitsBefore",
            "preparedAt",
            "applyingAt",
            "appliedAt",
            "appliedPids",
            "liveVerified",
            "restoringAt",
            "restoredAt",
            "runtimeRestored",
        }
        required = {
            "phase",
            "manifestSha256",
            "files",
            "directories",
            "unitsBefore",
            "preparedAt",
        }
        if not required.issubset(transaction) or not set(transaction).issubset(allowed):
            raise HarnessError("transaction journal fields are invalid")
        if transaction.get("phase") not in PHASES or not SHA256_RE.fullmatch(
            str(transaction.get("manifestSha256") or "")
        ):
            raise HarnessError("transaction identity is invalid")
        if "runtimeRestored" in transaction and not isinstance(
            transaction.get("runtimeRestored"), bool
        ):
            raise HarnessError("transaction runtime restoration state is invalid")
        files = transaction.get("files")
        directories = transaction.get("directories")
        units = transaction.get("unitsBefore")
        if not isinstance(files, Mapping) or set(files) != TRANSACTION_FILES:
            raise HarnessError("transaction file destinations are invalid")
        if (
            not isinstance(directories, Mapping)
            or set(directories) != TRANSACTION_DIRECTORIES
        ):
            raise HarnessError("transaction directory destinations are invalid")
        if not isinstance(units, Mapping) or set(units) != set(UNITS):
            raise HarnessError("transaction unit snapshot is invalid")
        expected_after = {
            AGENT_CONFIG: (0o640, self.root_uid, self.nuvion_gid),
            AGENT_COMMAND_KEYRING: (0o640, self.root_uid, self.nuvion_gid),
            UPDATER_COMMAND_KEYRING: (0o600, self.root_uid, self.root_gid),
            RELEASE_KEYRING: (0o600, self.root_uid, self.root_gid),
            HEALTH_KEYRING: (0o600, self.root_uid, self.root_gid),
            DEVICE_BINDING: (0o600, self.root_uid, self.root_gid),
        }
        for absolute, raw in files.items():
            if not isinstance(raw, Mapping) or set(raw) != {"before", "after"}:
                raise HarnessError("transaction file snapshot is invalid")
            for side in ("before", "after"):
                snapshot_raw = raw.get(side)
                if not isinstance(snapshot_raw, Mapping) or set(snapshot_raw) != {
                    "exists",
                    "payload",
                    "sha256",
                    "mode",
                    "uid",
                    "gid",
                }:
                    raise HarnessError("transaction file snapshot fields are invalid")
            after = raw["after"]
            if (
                after.get("exists") is not True
                or not isinstance(after.get("payload"), str)
                or not SHA256_RE.fullmatch(str(after.get("sha256") or ""))
                or (
                    after.get("mode"),
                    after.get("uid"),
                    after.get("gid"),
                )
                != expected_after[str(absolute)]
            ):
                raise HarnessError("transaction intended file metadata is invalid")
        for raw in directories.values():
            if not isinstance(raw, Mapping) or set(raw) != {"before", "after"}:
                raise HarnessError("transaction directory snapshot is invalid")
            for side in ("before", "after"):
                metadata = raw.get(side)
                if not isinstance(metadata, Mapping) or set(metadata) != {
                    "mode",
                    "uid",
                    "gid",
                }:
                    raise HarnessError(
                        "transaction directory snapshot fields are invalid"
                    )
                if not all(
                    isinstance(metadata.get(key), int)
                    and not isinstance(metadata.get(key), bool)
                    for key in ("mode", "uid", "gid")
                ):
                    raise HarnessError("transaction directory metadata is invalid")
        directory_after = {
            "/etc/nuv-agent": (0o750, self.root_uid, self.nuvion_gid),
            "/etc/nuvion-updater": (0o700, self.root_uid, self.root_gid),
        }
        for absolute, raw in directories.items():
            after = raw["after"]
            if (after.get("mode"), after.get("uid"), after.get("gid")) != (
                directory_after[str(absolute)]
            ):
                raise HarnessError("transaction intended directory metadata is invalid")
        if "appliedPids" in transaction:
            pids = transaction["appliedPids"]
            expected_services = {unit for unit in UNITS if unit.endswith(".service")}
            if (
                not isinstance(pids, Mapping)
                or set(pids) != expected_services
                or not all(
                    isinstance(pid, int) and not isinstance(pid, bool) and pid > 0
                    for pid in pids.values()
                )
            ):
                raise HarnessError("transaction applied PID proof is invalid")

    def _store_snapshot(
        self, directory: Path, name: str, value: FileSnapshot
    ) -> dict[str, object]:
        payload_name: str | None = None
        if value.exists:
            payload_name = name
            atomic_write(
                directory / name,
                value.payload,
                mode=0o600,
                uid=self.root_uid,
                gid=self.root_gid,
            )
        return {
            "exists": value.exists,
            "payload": payload_name,
            "sha256": sha256_bytes(value.payload) if value.exists else None,
            "mode": value.mode,
            "uid": value.uid,
            "gid": value.gid,
        }

    def _load_snapshot(
        self, directory: Path, raw: object, *, maximum: int
    ) -> FileSnapshot:
        if not isinstance(raw, Mapping) or raw.get("exists") not in {True, False}:
            raise HarnessError("transaction snapshot metadata is invalid")
        if raw["exists"] is False:
            return FileSnapshot(False, b"")
        name = raw.get("payload")
        if not isinstance(name, str) or Path(name).name != name:
            raise HarnessError("transaction snapshot payload name is invalid")
        payload, _ = read_regular(directory / name, maximum=maximum)
        if sha256_bytes(payload) != raw.get("sha256"):
            raise HarnessError("transaction snapshot digest mismatch")
        mode, uid, gid = raw.get("mode"), raw.get("uid"), raw.get("gid")
        if not all(isinstance(item, int) for item in (mode, uid, gid)):
            raise HarnessError("transaction snapshot ownership is invalid")
        return FileSnapshot(True, payload, int(mode), int(uid), int(gid))

    def _prepare_transaction(
        self,
        run_id: str,
        state: dict[str, Any],
        staged: Mapping[str, StagedInput],
        manifest: dict[str, Any],
    ) -> dict[str, Any]:
        transaction_dir = self._transaction_dir(run_id)
        if transaction_dir.exists():
            if state.get("trustTransaction") is None:
                metadata = transaction_dir.lstat()
                if (
                    stat.S_ISLNK(metadata.st_mode)
                    or not stat.S_ISDIR(metadata.st_mode)
                    or metadata.st_uid != self.root_uid
                    or stat.S_IMODE(metadata.st_mode) != 0o700
                ):
                    raise HarnessError("orphan transaction directory is unsafe")
                shutil.rmtree(transaction_dir)
                fsync_directory(transaction_dir.parent)
            else:
                raise HarnessError("transaction directory already exists")
        ensure_directory(
            transaction_dir, mode=0o700, uid=self.root_uid, gid=self.root_gid
        )
        manifest_payload = staged["manifest"].payload
        atomic_write(
            transaction_dir / "immutable-manifest.json",
            manifest_payload,
            mode=0o600,
            uid=self.root_uid,
            gid=self.root_gid,
        )
        config_before = snapshot(self.paths.config, maximum=MAX_STATE_BYTES)
        if not config_before.exists:
            raise HarnessError("Agent config is missing")
        desired = {
            AGENT_CONFIG: (
                self._render_config(config_before.payload),
                0o640,
                self.root_uid,
                self.nuvion_gid,
            ),
            AGENT_COMMAND_KEYRING: (
                staged["command"].payload,
                0o640,
                self.root_uid,
                self.nuvion_gid,
            ),
            UPDATER_COMMAND_KEYRING: (
                staged["command"].payload,
                0o600,
                self.root_uid,
                self.root_gid,
            ),
            RELEASE_KEYRING: (
                staged["release"].payload,
                0o600,
                self.root_uid,
                self.root_gid,
            ),
            HEALTH_KEYRING: (
                staged["health"].payload,
                0o600,
                self.root_uid,
                self.root_gid,
            ),
            DEVICE_BINDING: (
                staged["binding"].payload,
                0o600,
                self.root_uid,
                self.root_gid,
            ),
        }
        files: dict[str, object] = {}
        for index, (absolute, (payload, mode, uid, gid)) in enumerate(desired.items()):
            target = self.paths.rooted(absolute)
            before = snapshot(target, maximum=MAX_STATE_BYTES)
            before_meta = self._store_snapshot(
                transaction_dir, f"file-{index}.before", before
            )
            after_name = f"file-{index}.after"
            atomic_write(
                transaction_dir / after_name,
                payload,
                mode=0o600,
                uid=self.root_uid,
                gid=self.root_gid,
            )
            files[absolute] = {
                "before": before_meta,
                "after": {
                    "exists": True,
                    "payload": after_name,
                    "sha256": sha256_bytes(payload),
                    "mode": mode,
                    "uid": uid,
                    "gid": gid,
                },
            }
        directories: dict[str, object] = {}
        for absolute, after in {
            "/etc/nuv-agent": {
                "mode": 0o750,
                "uid": self.root_uid,
                "gid": self.nuvion_gid,
            },
            "/etc/nuvion-updater": {
                "mode": 0o700,
                "uid": self.root_uid,
                "gid": self.root_gid,
            },
        }.items():
            target = self.paths.rooted(absolute)
            metadata = target.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise HarnessError("trust directory is unsafe")
            directories[absolute] = {
                "before": {
                    "mode": stat.S_IMODE(metadata.st_mode),
                    "uid": metadata.st_uid,
                    "gid": metadata.st_gid,
                },
                "after": after,
            }
        transaction: dict[str, Any] = {
            "phase": "PREPARED",
            "manifestSha256": staged["manifest"].sha256,
            "files": files,
            "directories": directories,
            "unitsBefore": self._unit_snapshot(),
            "preparedAt": self.clock(),
        }
        state["trustTransaction"] = transaction
        self._save_state(run_id, state)
        return transaction

    def _transaction_manifest(
        self, run_id: str, transaction: Mapping[str, Any]
    ) -> dict[str, Any]:
        self._validate_transaction_shape(transaction)
        directory = self._transaction_dir(run_id)
        metadata = directory.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != self.root_uid
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise HarnessError("transaction directory ownership or mode is unsafe")
        payload, _ = read_regular(
            directory / "immutable-manifest.json",
            maximum=MAX_TRUST_BYTES,
        )
        if sha256_bytes(payload) != transaction.get("manifestSha256"):
            raise HarnessError("immutable transaction manifest digest mismatch")
        value = strict_json(payload, label="immutable transaction manifest")
        validate_manifest(value, run_id=run_id)
        return value

    def _file_matches(self, path: Path, expected: Mapping[str, object]) -> bool:
        if expected.get("exists") is False:
            return not path.exists() and not path.is_symlink()
        try:
            digest, metadata = sha256_regular(path, maximum=MAX_STATE_BYTES)
        except (HarnessError, OSError):
            return False
        return (
            digest == expected.get("sha256")
            and stat.S_IMODE(metadata.st_mode) == expected.get("mode")
            and metadata.st_uid == expected.get("uid")
            and metadata.st_gid == expected.get("gid")
        )

    def _directory_matches(self, path: Path, expected: Mapping[str, object]) -> bool:
        try:
            metadata = path.lstat()
        except OSError:
            return False
        return (
            stat.S_ISDIR(metadata.st_mode)
            and not stat.S_ISLNK(metadata.st_mode)
            and stat.S_IMODE(metadata.st_mode) == expected.get("mode")
            and metadata.st_uid == expected.get("uid")
            and metadata.st_gid == expected.get("gid")
        )

    def _transaction_matches(self, run_id: str, transaction: Mapping[str, Any]) -> bool:
        try:
            self._validate_transaction_shape(transaction)
        except HarnessError:
            return False
        directory = self._transaction_dir(run_id)
        files = transaction.get("files")
        directories = transaction.get("directories")
        if not isinstance(files, Mapping) or not isinstance(directories, Mapping):
            return False
        for absolute, raw in files.items():
            if not isinstance(raw, Mapping) or not isinstance(
                raw.get("after"), Mapping
            ):
                return False
            expected = raw["after"]
            try:
                self._load_snapshot(directory, expected, maximum=MAX_STATE_BYTES)
            except HarnessError:
                return False
            if not self._file_matches(self.paths.rooted(str(absolute)), expected):
                return False
        for absolute, raw in directories.items():
            if not isinstance(raw, Mapping) or not isinstance(
                raw.get("after"), Mapping
            ):
                return False
            if not self._directory_matches(
                self.paths.rooted(str(absolute)), raw["after"]
            ):
                return False
        return True

    def _apply_files(self, run_id: str, transaction: Mapping[str, Any]) -> None:
        self._validate_transaction_shape(transaction)
        directory = self._transaction_dir(run_id)
        directories = transaction.get("directories")
        files = transaction.get("files")
        if not isinstance(directories, Mapping) or not isinstance(files, Mapping):
            raise HarnessError("transaction metadata is invalid")
        for absolute, raw in directories.items():
            if not isinstance(raw, Mapping) or not isinstance(
                raw.get("after"), Mapping
            ):
                raise HarnessError("transaction directory metadata is invalid")
            expected = raw["after"]
            ensure_directory(
                self.paths.rooted(str(absolute)),
                mode=int(expected["mode"]),
                uid=int(expected["uid"]),
                gid=int(expected["gid"]),
            )
        for absolute, raw in files.items():
            if not isinstance(raw, Mapping) or not isinstance(
                raw.get("after"), Mapping
            ):
                raise HarnessError("transaction file metadata is invalid")
            expected = raw["after"]
            value = self._load_snapshot(directory, expected, maximum=MAX_STATE_BYTES)
            if not self._file_matches(self.paths.rooted(str(absolute)), expected):
                restore_snapshot(self.paths.rooted(str(absolute)), value)
        if not self._transaction_matches(run_id, transaction):
            raise HarnessError("trust files did not converge to exact intended digests")

    def _restart_runtime(self) -> dict[str, int]:
        before = {
            unit: int(self._unit_status(unit)["mainPid"])
            for unit in UNITS
            if unit.endswith(".service")
        }
        for unit in RESTART_ORDER:
            self._systemctl("restart", unit)
            if self._unit_status(unit)["active"] is not True:
                raise HarnessError("required runtime unit did not become active")
        after = {
            unit: int(self._unit_status(unit)["mainPid"])
            for unit in UNITS
            if unit.endswith(".service")
        }
        for unit, pid in after.items():
            if pid <= 0 or (before[unit] > 0 and pid == before[unit]):
                raise HarnessError("service PID did not change after exact restart")
        self._probe_updater()
        return after

    def _stop_agent_fail_closed(self) -> None:
        self._systemctl("stop", "nuv-agent.service")
        if self._unit_status("nuv-agent.service")["active"] is True:
            raise HarnessError("Agent could not be stopped at the safety boundary")

    def _apply_transaction(
        self, run_id: str, state: dict[str, Any], transaction: dict[str, Any]
    ) -> dict[str, object]:
        self._validate_transaction_shape(transaction)
        self._require_transaction_guard(run_id, state)
        phase = transaction.get("phase")
        if phase not in PHASES or phase in {"RESTORING", "RESTORED"}:
            raise HarnessError("trust transaction is not applyable")
        try:
            if phase == "APPLIED" and self._transaction_matches(run_id, transaction):
                services = {unit: self._unit_status(unit) for unit in UNITS}
                if all(
                    item["active"] is True
                    and (not unit.endswith(".service") or int(item["mainPid"]) > 0)
                    for unit, item in services.items()
                ) and transaction.get("liveVerified") is True:
                    updater = self._probe_updater()
                    return {"updater": updater, "idempotent": True}
            # No old process may observe a partially replaced config/keyring
            # set.  Quiesce every writer before APPLYING is durable.
            self._stop_writers()
            if phase != "APPLIED" or not self._transaction_matches(
                run_id, transaction
            ):
                transaction["phase"] = "APPLYING"
                transaction["applyingAt"] = self.clock()
                state["trustTransaction"] = transaction
                self._save_state(run_id, state)
                self._apply_files(run_id, transaction)
                # APPLIED means every byte and metadata field has already
                # converged.  Persist that fact while writers are still down;
                # a crash here can safely retry runtime activation without
                # exposing a mixed trust set.
                transaction.update(
                    {
                        "phase": "APPLIED",
                        "appliedAt": self.clock(),
                        "liveVerified": False,
                    }
                )
                transaction.pop("appliedPids", None)
                state["trustTransaction"] = transaction
                self._save_state(run_id, state)
            pids = self._restart_runtime()
            transaction.update(
                {
                    "appliedPids": pids,
                    "liveVerified": True,
                }
            )
            state["trustTransaction"] = transaction
            self._save_state(run_id, state)
            return {"updater": self._probe_updater(), "idempotent": False}
        except BaseException:
            try:
                self._stop_writers()
            except BaseException:
                pass
            raise

    def enable_fleet(
        self,
        run_id: str,
        *,
        command_keyring: str | Path,
        command_sha256: str,
        release_keyring: str | Path,
        release_sha256: str,
        health_keyring: str | Path,
        health_sha256: str,
        device_binding: str | Path,
        binding_sha256: str,
        manifest_path: str | Path,
        manifest_sha256: str,
    ) -> dict[str, object]:
        staged: list[StagedInput] = []
        with self._run_lock(run_id):
            try:
                for role, path, digest in (
                    ("command", command_keyring, command_sha256),
                    ("release", release_keyring, release_sha256),
                    ("health", health_keyring, health_sha256),
                    ("binding", device_binding, binding_sha256),
                    ("manifest", manifest_path, manifest_sha256),
                ):
                    staged.append(self._read_staged(run_id, role, path, digest))
                by_role = {item.role: item for item in staged}
                manifest = strict_json(
                    by_role["manifest"].payload, label="immutable manifest"
                )
                validate_manifest(manifest, run_id=run_id)
                identity = self.identity()
                if manifest["toolSha256"] != identity["toolSha256"]:
                    raise HarnessError("immutable manifest tool digest mismatch")
                expected_inputs = {
                    "commandSha256": by_role["command"].sha256,
                    "releaseSha256": by_role["release"].sha256,
                    "healthSha256": by_role["health"].sha256,
                    "bindingSha256": by_role["binding"].sha256,
                }
                if manifest["inputs"] != expected_inputs:
                    raise HarnessError("immutable manifest input digest mismatch")
                validate_ed25519_keyring(by_role["command"].payload, role="command")
                validate_ed25519_keyring(by_role["release"].payload, role="release")
                validate_ed25519_keyring(by_role["health"].payload, role="health")
                binding = validate_binding(
                    by_role["binding"].payload, manifest["identity"]
                )
                state = self._load_state(run_id)
                foundation = state.get("foundation")
                if (
                    not isinstance(foundation, Mapping)
                    or foundation.get("verified") is not True
                ):
                    raise HarnessError("foundation preflight is incomplete")
                live_foundation = self._foundation()
                if manifest["scenario"]["expectedPreviousSlot"] != foundation.get(
                    "currentSlot"
                ) or manifest["scenario"]["expectedPreviousVersion"] != foundation.get(
                    "currentVersion"
                ):
                    raise HarnessError(
                        "immutable manifest does not match the recorded baseline slot"
                    )
                backup = state.get("backup")
                if (
                    not isinstance(backup, Mapping)
                    or backup.get("complete") is not True
                ):
                    raise HarnessError("consistent emergency backup is incomplete")
                self._verify_archive(
                    run_id,
                    Path(str(backup.get("archivePath") or "")),
                    str(backup.get("sha256") or ""),
                )
                transaction = state.get("trustTransaction")
                if transaction is None:
                    if (
                        foundation.get("currentSlot")
                        != live_foundation["slots"]["current"]
                        or foundation.get("currentVersion")
                        != live_foundation["slots"]["currentVersion"]
                    ):
                        raise HarnessError(
                            "immutable manifest does not match the live baseline slot"
                        )
                    config_payload, config_metadata = read_regular(
                        self.paths.config, maximum=MAX_STATE_BYTES
                    )
                    if (
                        config_metadata.st_uid != self.root_uid
                        or config_metadata.st_gid != self.nuvion_gid
                        or stat.S_IMODE(config_metadata.st_mode) != 0o660
                    ):
                        raise HarnessError(
                            "Agent config ownership or mode is unsafe before trust"
                        )
                    configured_device, configured_space = (
                        self._configured_device_identity(config_payload)
                    )
                    if (
                        configured_device != binding["deviceId"]
                        or configured_space != binding["spaceId"]
                    ):
                        raise HarnessError(
                            "device binding does not match the provisioned Agent identity"
                        )
                    transaction = self._prepare_transaction(
                        run_id, state, by_role, manifest
                    )
                elif not isinstance(transaction, dict):
                    raise HarnessError("trust transaction journal is invalid")
                else:
                    persisted = self._transaction_manifest(run_id, transaction)
                    if persisted != manifest or transaction.get("phase") == "RESTORED":
                        raise HarnessError("immutable transaction manifest mismatch")
                    files = transaction["files"]
                    intended_config = self._load_snapshot(
                        self._transaction_dir(run_id),
                        files[AGENT_CONFIG]["after"],
                        maximum=MAX_STATE_BYTES,
                    )
                    configured_device, configured_space = (
                        self._configured_device_identity(intended_config.payload)
                    )
                    if (
                        configured_device != binding["deviceId"]
                        or configured_space != binding["spaceId"]
                    ):
                        raise HarnessError(
                            "immutable transaction device identity mismatch"
                        )
                # This root-owned guard spans the deliberate host CLI gap
                # between Fleet evidence and candidate-soak cleanup.  It must
                # be durably bound and externally active before APPLYING.
                self._ensure_transaction_guard(run_id, state)
                result = self._apply_transaction(run_id, state, transaction)
                return {
                    "schemaVersion": 1,
                    "runId": run_id,
                    "phase": "APPLIED",
                    "trustProvisioned": True,
                    **result,
                }
            finally:
                self._cleanup_staged(staged)

    def _candidate_slot_path(self, run_id: str, bom_digest: str) -> Path:
        match = DIGEST_RE.fullmatch(bom_digest)
        if match is None:
            raise HarnessError("candidate BOM digest is invalid")
        expected = self.paths.candidate_root / f"{run_id}-{match.group(1)}"
        if expected.parent != self.paths.candidate_root:
            raise HarnessError("candidate slot escaped the fixed root")
        return expected

    def _candidate_incoming_path(self, slot: Path) -> Path:
        if (
            slot.parent != self.paths.candidate_root
            or re.fullmatch(
                rf"{RUN_ID_RE.pattern[:-1]}-[0-9a-f]{{64}}$", slot.name
            )
            is None
        ):
            raise HarnessError("candidate incoming slot identity is invalid")
        incoming = self.paths.candidate_root / f".{slot.name}.incoming"
        if incoming.parent != self.paths.candidate_root:
            raise HarnessError("candidate incoming slot escaped the fixed root")
        return incoming

    def _remove_candidate_incoming(self, incoming: Path) -> bool:
        """Remove only the exact, root-owned partial candidate tree."""

        if incoming.parent != self.paths.candidate_root:
            raise HarnessError("candidate incoming cleanup path is unsafe")
        if (
            not self.paths.candidate_root.exists()
            and not self.paths.candidate_root.is_symlink()
        ):
            return False
        root_metadata = self.paths.candidate_root.lstat()
        if (
            stat.S_ISLNK(root_metadata.st_mode)
            or not stat.S_ISDIR(root_metadata.st_mode)
            or (root_metadata.st_uid, root_metadata.st_gid)
            != (self.root_uid, self.root_gid)
            or stat.S_IMODE(root_metadata.st_mode) != 0o755
        ):
            raise HarnessError("candidate incoming cleanup root is unsafe")
        if not incoming.exists() and not incoming.is_symlink():
            return False
        metadata = incoming.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or (metadata.st_uid, metadata.st_gid)
            != (self.root_uid, self.root_gid)
            or stat.S_IMODE(metadata.st_mode) not in {0o700, 0o755}
        ):
            raise HarnessError("candidate incoming slot is unsafe")
        entries = 0
        total = 0
        for directory, subdirs, files in os.walk(incoming, followlinks=False):
            directory_path = Path(directory)
            directory_metadata = directory_path.lstat()
            if (
                stat.S_ISLNK(directory_metadata.st_mode)
                or not stat.S_ISDIR(directory_metadata.st_mode)
                or (directory_metadata.st_uid, directory_metadata.st_gid)
                != (self.root_uid, self.root_gid)
                or directory_metadata.st_mode & 0o022
            ):
                raise HarnessError("candidate incoming tree metadata is unsafe")
            for name in (*subdirs, *files):
                entries += 1
                if entries > MAX_CANDIDATE_ENTRIES + 4:
                    raise HarnessError("candidate incoming tree entry count is invalid")
                path = directory_path / name
                item = path.lstat()
                if (
                    stat.S_ISLNK(item.st_mode)
                    or not (stat.S_ISDIR(item.st_mode) or stat.S_ISREG(item.st_mode))
                    or (item.st_uid, item.st_gid) != (self.root_uid, self.root_gid)
                    or item.st_mode & 0o022
                ):
                    raise HarnessError("candidate incoming tree metadata is unsafe")
                if stat.S_ISREG(item.st_mode):
                    total += item.st_size
                    if total > MAX_CANDIDATE_UNPACKED_BYTES + 3 * MAX_TRUST_BYTES:
                        raise HarnessError("candidate incoming tree size is invalid")
        shutil.rmtree(incoming)
        fsync_directory(incoming.parent)
        return True

    def _recover_candidate_incoming(
        self, run_id: str, soak: Mapping[str, Any]
    ) -> bool:
        raw_slot = soak.get("candidateSlot")
        raw_incoming = soak.get("candidateIncomingPath")
        if not isinstance(raw_slot, str) or not isinstance(raw_incoming, str):
            raise HarnessError("candidate STAGING recovery journal is incomplete")
        slot = Path(raw_slot)
        if re.fullmatch(
            rf"{re.escape(run_id)}-[0-9a-f]{{64}}", slot.name
        ) is None:
            raise HarnessError("candidate STAGING recovery slot is invalid")
        expected = self._candidate_incoming_path(slot)
        if raw_incoming != str(expected):
            raise HarnessError("candidate STAGING recovery path mismatch")
        return self._remove_candidate_incoming(expected)

    def _candidate_harness_execution_path(
        self, run_id: str, harness_sha256: str
    ) -> Path:
        canonical_run_id(run_id)
        if SHA256_RE.fullmatch(harness_sha256) is None:
            raise HarnessError("candidate harness digest is invalid")
        return (
            self.paths.state_root
            / run_id
            / f"candidate-oak-harness-{harness_sha256}.sh"
        )

    def _pin_candidate_harness(self, run_id: str, harness_sha256: str) -> Path:
        """Copy reviewed harness bytes into a private, non-writable run path."""

        source_payload, source_metadata = read_regular(
            self.paths.candidate_harness, maximum=2 * 1024 * 1024
        )
        if (
            sha256_bytes(source_payload) != harness_sha256
            or (source_metadata.st_uid, source_metadata.st_gid)
            != (self.root_uid, self.root_gid)
            or source_metadata.st_nlink != 1
            or stat.S_IMODE(source_metadata.st_mode) != 0o755
        ):
            raise HarnessError("installed OAK harness differs from reviewed source")
        path = self._candidate_harness_execution_path(run_id, harness_sha256)
        self._prepare_run(run_id)
        if not path.exists() and not path.is_symlink():
            atomic_write(
                path,
                source_payload,
                mode=0o500,
                uid=self.root_uid,
                gid=self.root_gid,
            )
        payload, metadata = read_regular(path, maximum=2 * 1024 * 1024)
        if (
            payload != source_payload
            or sha256_bytes(payload) != harness_sha256
            or (metadata.st_uid, metadata.st_gid)
            != (self.root_uid, self.root_gid)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o500
        ):
            raise HarnessError("pinned OAK harness metadata or bytes are invalid")
        return path

    def _verify_pinned_candidate_harness(
        self, run_id: str, soak: Mapping[str, Any], harness_sha256: str
    ) -> Path:
        expected = self._candidate_harness_execution_path(run_id, harness_sha256)
        execution = soak.get("harnessExecution")
        if not isinstance(execution, Mapping) or dict(execution) != {
            "path": str(expected),
            "sha256": harness_sha256,
        }:
            raise HarnessError("candidate harness execution journal is invalid")
        payload, metadata = read_regular(expected, maximum=2 * 1024 * 1024)
        if (
            sha256_bytes(payload) != harness_sha256
            or (metadata.st_uid, metadata.st_gid)
            != (self.root_uid, self.root_gid)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o500
        ):
            raise HarnessError("pinned OAK harness changed before execution")
        return expected

    def _remove_pinned_candidate_harness(
        self, run_id: str, soak: Mapping[str, Any]
    ) -> bool:
        execution = soak.get("harnessExecution")
        if execution is None:
            return False
        if not isinstance(execution, Mapping):
            raise HarnessError("candidate harness cleanup journal is invalid")
        digest = execution.get("sha256")
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            raise HarnessError("candidate harness cleanup digest is invalid")
        expected = self._candidate_harness_execution_path(run_id, digest)
        if execution.get("path") != str(expected):
            raise HarnessError("candidate harness cleanup path mismatch")
        if not expected.exists() and not expected.is_symlink():
            return False
        payload, metadata = read_regular(expected, maximum=2 * 1024 * 1024)
        if (
            sha256_bytes(payload) != digest
            or (metadata.st_uid, metadata.st_gid)
            != (self.root_uid, self.root_gid)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o500
        ):
            raise HarnessError("candidate harness cleanup target is unsafe")
        expected.unlink()
        fsync_directory(expected.parent)
        return True

    @staticmethod
    def _digest_frame(digest: Any, payload: bytes) -> None:
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    def _bounded_tree_snapshot(
        self,
        root: Path,
        *,
        label: str,
        totals: dict[str, int],
        allow_missing: bool,
    ) -> dict[str, object]:
        """Hash a stable tree without retaining any file names or payloads."""

        digest = hashlib.sha256()
        root_entries = 0
        root_bytes = 0
        if not root.exists() and not root.is_symlink():
            if not allow_missing:
                raise HarnessError(f"{label} is missing")
            self._digest_frame(digest, b"missing")
            return {
                "exists": False,
                "entries": 0,
                "bytes": 0,
                "sha256": digest.hexdigest(),
            }

        def identity(item: os.stat_result) -> tuple[int, ...]:
            return (
                item.st_dev,
                item.st_ino,
                item.st_mode,
                item.st_nlink,
                item.st_uid,
                item.st_gid,
                item.st_size,
                item.st_mtime_ns,
                item.st_ctime_ns,
            )

        def visit(path: Path, relative: Path) -> None:
            nonlocal root_entries, root_bytes
            before = path.lstat()
            totals["entries"] += 1
            root_entries += 1
            if totals["entries"] > MAX_BACKUP_ENTRIES:
                raise HarnessError(f"{label} entry limit exceeded")
            self._digest_frame(digest, os.fsencode(str(relative)))
            for number in identity(before):
                self._digest_frame(digest, str(number).encode("ascii"))
            if stat.S_ISLNK(before.st_mode):
                self._digest_frame(digest, b"symlink")
                target = os.readlink(path)
                self._digest_frame(digest, os.fsencode(target))
                if identity(before) != identity(path.lstat()):
                    raise HarnessError(f"{label} changed while hashing")
                return
            if stat.S_ISDIR(before.st_mode):
                self._digest_frame(digest, b"directory")
                with os.scandir(path) as entries:
                    children = sorted(entries, key=lambda item: item.name)
                for entry in children:
                    visit(Path(entry.path), relative / entry.name)
                if identity(before) != identity(path.lstat()):
                    raise HarnessError(f"{label} changed while hashing")
                return
            if not stat.S_ISREG(before.st_mode):
                raise HarnessError(f"{label} contains unsupported file type")
            if totals["bytes"] + before.st_size > MAX_BACKUP_BYTES:
                raise HarnessError(f"{label} byte limit exceeded")
            file_digest, opened = sha256_regular(path, maximum=MAX_BACKUP_BYTES)
            after = path.lstat()
            if identity(before) != identity(opened) or identity(opened) != identity(
                after
            ):
                raise HarnessError(f"{label} changed while hashing")
            totals["bytes"] += opened.st_size
            root_bytes += opened.st_size
            self._digest_frame(digest, b"regular")
            self._digest_frame(digest, file_digest.encode("ascii"))

        visit(root, Path("."))
        return {
            "exists": True,
            "entries": root_entries,
            "bytes": root_bytes,
            "sha256": digest.hexdigest(),
        }

    def _candidate_persistent_state_snapshot(self) -> dict[str, Any]:
        """Digest persistent Agent/updater state without retaining secret payloads."""

        totals = {"entries": 0, "bytes": 0}
        roots: dict[str, dict[str, object]] = {}
        for absolute in CANDIDATE_PERSISTENT_PATHS:
            roots[absolute] = self._bounded_tree_snapshot(
                self.paths.rooted(absolute),
                label="candidate persistent state",
                totals=totals,
                allow_missing=True,
            )

        canonical = (
            json.dumps(roots, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        return {
            "schemaVersion": 1,
            "roots": roots,
            "sha256": sha256_bytes(canonical),
            "entries": totals["entries"],
            "bytes": totals["bytes"],
        }

    def _release_tree_snapshot(
        self, slots: Mapping[str, object]
    ) -> dict[str, object]:
        """Bind slot links to stable recursive release content and metadata."""

        if set(slots) != {"current", "previous"}:
            raise HarnessError("release tree slot set is invalid")
        totals = {"entries": 0, "bytes": 0}
        trees: dict[str, dict[str, object]] = {}
        for role in ("current", "previous"):
            target = slots.get(role)
            if not isinstance(target, str) or re.fullmatch(
                r"(?:releases/[0-9a-f]{64}|bootstrap/[0-9A-Za-z.+-]{1,64})",
                target,
            ) is None:
                raise HarnessError("release tree slot target is invalid")
            root = self.paths.install_root / target
            metadata = root.lstat()
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != self.root_uid
                or metadata.st_mode & 0o022
            ):
                raise HarnessError("release tree root metadata is unsafe")
            trees[role] = {
                "target": target,
                **self._bounded_tree_snapshot(
                    root,
                    label="release tree",
                    totals=totals,
                    allow_missing=False,
                ),
            }
        canonical = (
            json.dumps(trees, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        return {
            "schemaVersion": 1,
            "slots": trees,
            "sha256": sha256_bytes(canonical),
            "entries": totals["entries"],
            "bytes": totals["bytes"],
        }

    def _candidate_control_marker(
        self,
        *,
        run_id: str,
        slot: Path,
        bom: Mapping[str, Any],
        bundle_sha256: str,
        harness_sha256: str,
    ) -> dict[str, object]:
        return {
            "schemaVersion": 1,
            "protocolVersion": PROTOCOL_VERSION,
            "runId": run_id,
            "slotKind": "candidate",
            "candidateSlot": str(slot),
            "bomDigest": bom["bomDigest"],
            "artifactDigest": f"sha256:{bundle_sha256}",
            "agentVersion": bom["agentVersion"],
            "componentSha": bom["componentSha"],
            "harnessSha256": harness_sha256,
        }

    def _verify_candidate_slot(
        self,
        slot: Path,
        *,
        control: Mapping[str, object],
        release_marker: Mapping[str, object],
        bom: Mapping[str, Any],
    ) -> str:
        try:
            metadata = slot.lstat()
        except OSError as exc:
            raise HarnessError("candidate slot is unavailable") from exc
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != self.root_uid
            or metadata.st_mode & 0o022
            or slot.parent != self.paths.candidate_root
        ):
            raise HarnessError("candidate slot metadata is unsafe")
        marker_path = slot / ".nuvion/candidate-soak.json"
        marker_payload, marker_metadata = read_regular(
            marker_path, maximum=MAX_TRUST_BYTES
        )
        if marker_metadata.st_uid != self.root_uid or marker_metadata.st_mode & 0o022:
            raise HarnessError("candidate control marker metadata is unsafe")
        if strict_json(marker_payload, label="candidate control marker") != dict(control):
            raise HarnessError("candidate control marker collision")
        release_payload, release_metadata = read_regular(
            slot / ".nuvion/release.json", maximum=MAX_TRUST_BYTES
        )
        if release_metadata.st_uid != self.root_uid or release_metadata.st_mode & 0o022:
            raise HarnessError("candidate release marker metadata is unsafe")
        if strict_json(release_payload, label="candidate release marker") != dict(
            release_marker
        ):
            raise HarnessError("candidate release marker collision")
        bom_payload, bom_metadata = read_regular(
            slot / ".nuvion/release-bom.json", maximum=MAX_TRUST_BYTES
        )
        if (
            bom_metadata.st_uid != self.root_uid
            or bom_metadata.st_mode & 0o022
            or strict_json(bom_payload, label="candidate staged BOM") != dict(bom)
        ):
            raise HarnessError("candidate staged BOM collision")
        entrypoint = slot / "bin/nuv-agent"
        python = slot / "venv/bin/python"
        for required in (entrypoint, python):
            required_metadata = required.lstat()
            if (
                stat.S_ISLNK(required_metadata.st_mode)
                or not stat.S_ISREG(required_metadata.st_mode)
                or required_metadata.st_uid != self.root_uid
                or required_metadata.st_mode & 0o022
                or not required_metadata.st_mode & 0o111
            ):
                raise HarnessError("candidate executable metadata is unsafe")
        wrapper_payload, wrapper_metadata = read_regular(python, maximum=4096)
        if (
            wrapper_payload != CANDIDATE_PYTHON_WRAPPER
            or wrapper_metadata.st_uid != self.root_uid
            or stat.S_IMODE(wrapper_metadata.st_mode) != 0o755
        ):
            raise HarnessError("candidate Python wrapper differs from canonical bytes")
        site_packages = slot / CANDIDATE_SITE_PACKAGES_RELATIVE
        site_metadata = site_packages.lstat()
        if (
            stat.S_ISLNK(site_metadata.st_mode)
            or not stat.S_ISDIR(site_metadata.st_mode)
            or site_metadata.st_uid != self.root_uid
            or site_metadata.st_mode & 0o022
        ):
            raise HarnessError("candidate site-packages metadata is unsafe")
        return sha256_bytes(marker_payload)

    @staticmethod
    def _tar_octal_size(field: bytes) -> int:
        """Parse the portable tar size field without allocating extension data."""

        if len(field) != 12 or field[:1] and field[0] & 0x80:
            # GNU/base-256 values are unnecessary for our bounded release
            # format and accepting them would duplicate subtle tarfile rules.
            raise HarnessError("candidate tar uses an unsupported size encoding")
        raw = field.rstrip(b"\0 ").lstrip(b" ")
        if not raw:
            return 0
        if re.fullmatch(rb"[0-7]+", raw) is None:
            raise HarnessError("candidate tar size field is invalid")
        return int(raw, 8)

    def _prescan_candidate_archive(self, bundle_file: io.BufferedReader) -> None:
        """Bound gzip output and tar extension payloads before ``tarfile``.

        ``tarfile`` materializes PAX and GNU long-name payloads while creating
        ``TarInfo``.  Inspecting the raw 512-byte headers first lets us reject a
        declared metadata bomb before any such allocation occurs.
        """

        bundle_file.seek(0)
        magic = bundle_file.read(2)
        bundle_file.seek(0)
        source: Any
        gzip_source: gzip.GzipFile | None = None
        if magic == b"\x1f\x8b":
            gzip_source = gzip.GzipFile(fileobj=bundle_file, mode="rb")
            source = gzip_source
        else:
            source = bundle_file
        decoded = 0

        def read_bounded(size: int) -> bytes:
            nonlocal decoded
            if size < 0 or decoded + size > MAX_CANDIDATE_TAR_STREAM_BYTES:
                raise HarnessError("candidate tar stream exceeds hard byte limit")
            try:
                payload = source.read(size)
            except (EOFError, OSError, gzip.BadGzipFile) as exc:
                raise HarnessError("candidate bundle compression is invalid") from exc
            decoded += len(payload)
            if decoded > MAX_CANDIDATE_TAR_STREAM_BYTES:
                raise HarnessError("candidate tar stream exceeds hard byte limit")
            return payload

        try:
            headers = 0
            regular_bytes = 0
            metadata_bytes = 0
            extension_pending = False
            zero_blocks = 0
            while True:
                header = read_bounded(512)
                if not header:
                    break
                if len(header) != 512:
                    raise HarnessError("candidate tar header is truncated")
                if header == b"\0" * 512:
                    zero_blocks += 1
                    if zero_blocks >= 2:
                        # Drain bounded decoder output so concatenated gzip data
                        # and trailing decompression bombs cannot evade the cap.
                        while True:
                            capacity = MAX_CANDIDATE_TAR_STREAM_BYTES - decoded
                            if capacity == 0:
                                try:
                                    probe = source.read(1)
                                except (EOFError, OSError, gzip.BadGzipFile) as exc:
                                    raise HarnessError(
                                        "candidate bundle compression is invalid"
                                    ) from exc
                                if probe:
                                    raise HarnessError(
                                        "candidate tar stream exceeds hard byte limit"
                                    )
                                break
                            trailer = read_bounded(
                                min(COMMAND_IO_CHUNK_BYTES, capacity)
                            )
                            if not trailer:
                                break
                            if any(trailer):
                                raise HarnessError(
                                    "candidate tar has non-zero trailing data"
                                )
                        break
                    continue
                if zero_blocks:
                    raise HarnessError("candidate tar has data after an end marker")
                headers += 1
                if headers > MAX_CANDIDATE_ENTRIES * 2 + 16:
                    raise HarnessError("candidate tar raw header count is invalid")
                size = self._tar_octal_size(header[124:136])
                type_flag = header[156:157] or b"0"
                if type_flag == b"L":
                    if extension_pending:
                        raise HarnessError(
                            "candidate tar extension chain is unsupported"
                        )
                    if size > MAX_CANDIDATE_TAR_METADATA_BYTES:
                        raise HarnessError(
                            "candidate tar extended metadata exceeds hard limit"
                        )
                    metadata_bytes += size
                    if metadata_bytes > MAX_CANDIDATE_TAR_METADATA_TOTAL_BYTES:
                        raise HarnessError(
                            "candidate tar aggregate metadata exceeds hard limit"
                        )
                    extension_pending = True
                elif type_flag in {b"0", b"\0"}:
                    regular_bytes += size
                    if regular_bytes > MAX_CANDIDATE_UNPACKED_BYTES:
                        raise HarnessError("candidate bundle exceeds unpacked limit")
                    extension_pending = False
                elif type_flag == b"5":
                    if size != 0:
                        raise HarnessError("candidate tar directory payload is invalid")
                    extension_pending = False
                else:
                    # The native release builder emits only regular files,
                    # directories and GNU long-name records. PAX/global/link,
                    # sparse and device types are unnecessary and may allocate
                    # metadata before TarInfo is yielded.
                    raise HarnessError("candidate tar member type is unsupported")
                remaining = ((size + 511) // 512) * 512
                while remaining:
                    chunk_size = min(COMMAND_IO_CHUNK_BYTES, remaining)
                    chunk = read_bounded(chunk_size)
                    if len(chunk) != chunk_size:
                        raise HarnessError("candidate tar member is truncated")
                    remaining -= chunk_size
            if headers == 0:
                raise HarnessError("candidate bundle entry count is invalid")
            if zero_blocks != 2:
                raise HarnessError("candidate tar lacks canonical end markers")
            if extension_pending:
                raise HarnessError("candidate tar extension has no target member")
        finally:
            if gzip_source is not None:
                gzip_source.close()
            bundle_file.seek(0)

    def _stage_candidate_bundle(
        self,
        *,
        run_id: str,
        bundle: CandidateInput,
        bom: Mapping[str, Any],
        harness_sha256: str,
        release_marker: Mapping[str, object],
    ) -> tuple[Path, str]:
        ensure_directory(
            self.paths.candidate_root,
            mode=0o755,
            uid=self.root_uid,
            gid=self.root_gid,
        )
        slot = self._candidate_slot_path(run_id, str(bom["bomDigest"]))
        control = self._candidate_control_marker(
            run_id=run_id,
            slot=slot,
            bom=bom,
            bundle_sha256=bundle.sha256,
            harness_sha256=harness_sha256,
        )
        if slot.exists() or slot.is_symlink():
            return slot, self._verify_candidate_slot(
                slot, control=control, release_marker=release_marker, bom=bom
            )
        incoming = self._candidate_incoming_path(slot)
        if incoming.exists() or incoming.is_symlink():
            raise HarnessError(
                "candidate incoming slot requires journal recovery before staging"
            )
        incoming.mkdir(mode=0o700)
        os.chown(incoming, self.root_uid, self.root_gid)
        os.chmod(incoming, 0o700)
        fsync_directory(incoming.parent)
        try:
            bundle_descriptor = os.open(
                bundle.path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                opened_bundle = os.fstat(bundle_descriptor)
                if (
                    opened_bundle.st_dev,
                    opened_bundle.st_ino,
                    opened_bundle.st_size,
                ) != (bundle.device, bundle.inode, bundle.size):
                    raise HarnessError("candidate bundle changed before extraction")
                opened_identity = (
                    opened_bundle.st_dev,
                    opened_bundle.st_ino,
                    opened_bundle.st_size,
                    opened_bundle.st_mtime_ns,
                    opened_bundle.st_ctime_ns,
                )
                opened_digest = hashlib.sha256()
                opened_total = 0
                while True:
                    chunk = os.read(bundle_descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    opened_total += len(chunk)
                    if opened_total > MAX_CANDIDATE_BUNDLE_BYTES:
                        raise HarnessError("candidate bundle exceeds staged size limit")
                    opened_digest.update(chunk)
                hashed_bundle = os.fstat(bundle_descriptor)
                if (
                    opened_digest.hexdigest() != bundle.sha256
                    or opened_total != bundle.size
                    or (
                        hashed_bundle.st_dev,
                        hashed_bundle.st_ino,
                        hashed_bundle.st_size,
                        hashed_bundle.st_mtime_ns,
                        hashed_bundle.st_ctime_ns,
                    )
                    != opened_identity
                ):
                    raise HarnessError("candidate bundle changed before extraction")
                os.lseek(bundle_descriptor, 0, os.SEEK_SET)
            except BaseException:
                os.close(bundle_descriptor)
                raise
            bundle_file = os.fdopen(bundle_descriptor, "rb", closefd=True)

            def consume_archive(*, extract: bool) -> tuple[int, int, str]:
                bundle_file.seek(0)
                try:
                    archive = tarfile.open(  # noqa: SIM115
                        fileobj=bundle_file, mode="r|*"
                    )
                except (tarfile.TarError, OSError) as exc:
                    raise HarnessError(
                        "candidate bundle is not a readable tar archive"
                    ) from exc
                count = 0
                total = 0
                seen: set[bytes] = set()
                manifest_digest = hashlib.sha256()
                reserved_metadata = False
                with archive:
                    for member in archive:
                        count += 1
                        if count > MAX_CANDIDATE_ENTRIES:
                            archive.members.clear()
                            raise HarnessError(
                                "candidate bundle entry count is invalid"
                            )
                        relative = PurePosixPath(member.name)
                        normalized = relative.as_posix()
                        try:
                            name_bytes = normalized.encode("utf-8", errors="strict")
                        except UnicodeError as exc:
                            raise HarnessError(
                                "candidate bundle contains an unsafe member"
                            ) from exc
                        name_digest = hashlib.sha256(name_bytes).digest()
                        if (
                            len(name_bytes) > 1024
                            or len(relative.parts) > 64
                            or relative.is_absolute()
                            or not relative.parts
                            or "\\" in member.name
                            or any(part in {"", ".", ".."} for part in relative.parts)
                            or name_digest in seen
                            or member.uid != 0
                            or member.gid != 0
                            or not (member.isdir() or member.isreg())
                            or (member.isreg() and member.size < 0)
                        ):
                            raise HarnessError(
                                "candidate bundle contains an unsafe member"
                            )
                        seen.add(name_digest)
                        reserved_metadata = reserved_metadata or normalized == ".nuvion" or normalized.startswith(
                            ".nuvion/"
                        )
                        if member.isreg():
                            total += member.size
                            if total > MAX_CANDIDATE_UNPACKED_BYTES:
                                raise HarnessError(
                                    "candidate bundle exceeds unpacked limit"
                                )
                        manifest_digest.update(
                            json.dumps(
                                [
                                    normalized,
                                    "dir" if member.isdir() else "file",
                                    member.size,
                                    member.mode & 0o7777,
                                    member.uid,
                                    member.gid,
                                ],
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ).encode("utf-8")
                            + b"\n"
                        )
                        if extract:
                            output = incoming.joinpath(*relative.parts)
                            if member.isdir():
                                output.mkdir(parents=True, exist_ok=True, mode=0o755)
                                os.chmod(output, 0o755)
                                os.chown(output, self.root_uid, self.root_gid)
                            else:
                                output.parent.mkdir(
                                    parents=True, exist_ok=True, mode=0o755
                                )
                                source = archive.extractfile(member)
                                if source is None:
                                    raise HarnessError(
                                        "candidate bundle member cannot be read"
                                    )
                                mode = 0o755 if member.mode & 0o111 else 0o644
                                descriptor = os.open(
                                    output,
                                    os.O_WRONLY
                                    | os.O_CREAT
                                    | os.O_EXCL
                                    | getattr(os, "O_CLOEXEC", 0)
                                    | getattr(os, "O_NOFOLLOW", 0),
                                    mode,
                                )
                                try:
                                    with (
                                        source,
                                        os.fdopen(
                                            descriptor, "wb", closefd=False
                                        ) as target,
                                    ):
                                        shutil.copyfileobj(
                                            source, target, length=1024 * 1024
                                        )
                                        if target.tell() != member.size:
                                            raise HarnessError(
                                                "candidate bundle member size changed during extraction"
                                            )
                                        target.flush()
                                        os.fsync(target.fileno())
                                    os.fchmod(descriptor, mode)
                                    os.fchown(
                                        descriptor, self.root_uid, self.root_gid
                                    )
                                finally:
                                    os.close(descriptor)
                        # TarFile keeps every yielded TarInfo in ``members`` even
                        # in some streaming implementations. Discard it after the
                        # current payload is consumed so an attacker cannot turn
                        # the entry-count guard into a metadata OOM.
                        archive.members.clear()
                if count == 0:
                    raise HarnessError("candidate bundle entry count is invalid")
                if reserved_metadata:
                    raise HarnessError("candidate bundle reserves .nuvion metadata")
                return count, total, manifest_digest.hexdigest()

            with bundle_file:
                self._prescan_candidate_archive(bundle_file)
                validated_archive = consume_archive(extract=False)
                free = int(self.disk_usage(self.paths.candidate_root).free)
                if free < validated_archive[1] + MIN_FREE_BYTES:
                    raise HarnessError("candidate bundle lacks install disk reserve")
                extracted_archive = consume_archive(extract=True)
                if extracted_archive != validated_archive:
                    raise HarnessError("candidate archive changed between passes")
                after_bundle = os.fstat(bundle_descriptor)
                if (
                    after_bundle.st_dev,
                    after_bundle.st_ino,
                    after_bundle.st_size,
                    after_bundle.st_mtime_ns,
                    after_bundle.st_ctime_ns,
                ) != opened_identity:
                    raise HarnessError("candidate bundle changed during extraction")
            metadata_root = incoming / ".nuvion"
            metadata_root.mkdir(mode=0o755)
            os.chown(metadata_root, self.root_uid, self.root_gid)
            release_payload = (
                json.dumps(
                    release_marker,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            control_payload = (
                json.dumps(
                    control, sort_keys=True, separators=(",", ":"), allow_nan=False
                )
                + "\n"
            ).encode("utf-8")
            bom_payload = (
                json.dumps(
                    bom,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            atomic_write(
                metadata_root / "release.json",
                release_payload,
                mode=0o644,
                uid=self.root_uid,
                gid=self.root_gid,
            )
            atomic_write(
                metadata_root / "candidate-soak.json",
                control_payload,
                mode=0o644,
                uid=self.root_uid,
                gid=self.root_gid,
            )
            atomic_write(
                metadata_root / "release-bom.json",
                bom_payload,
                mode=0o644,
                uid=self.root_uid,
                gid=self.root_gid,
            )
            for directory, subdirs, files in os.walk(incoming):
                for name in subdirs:
                    os.chmod(Path(directory) / name, 0o755)
                for name in files:
                    path = Path(directory) / name
                    mode = path.stat(follow_symlinks=False).st_mode
                    os.chmod(path, 0o755 if mode & 0o111 else 0o644)
            os.chmod(incoming, 0o755)
            os.rename(incoming, slot)
            fsync_directory(self.paths.candidate_root)
        except BaseException:
            self._remove_candidate_incoming(incoming)
            raise
        return slot, self._verify_candidate_slot(
            slot, control=control, release_marker=release_marker, bom=bom
        )

    @staticmethod
    def _slots_equal(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
        return left.get("current") == right.get("current") and left.get(
            "previous"
        ) == right.get("previous")

    def _slot_snapshot(self) -> dict[str, object]:
        return {
            "current": self._slot_link("current", required=True),
            "previous": self._slot_link("previous", required=False),
        }

    def _remove_candidate_slot(self, slot: Path) -> None:
        expected_parent = self.paths.candidate_root
        metadata = slot.lstat()
        if (
            slot.parent != expected_parent
            or stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != self.root_uid
            or metadata.st_mode & 0o022
        ):
            raise HarnessError("candidate cleanup target is unsafe")
        shutil.rmtree(slot)
        fsync_directory(expected_parent)

    def _load_candidate_soak_evidence(
        self, run_id: str, *, expected_sha256: str | None = None
    ) -> dict[str, Any]:
        path = self.paths.state_root / run_id / "candidate-soak-evidence.json"
        payload, metadata = read_regular(path, maximum=MAX_STATE_BYTES)
        if metadata.st_uid != self.root_uid or stat.S_IMODE(metadata.st_mode) != 0o600:
            raise HarnessError("candidate soak evidence metadata is unsafe")
        value = strict_json(payload, label="candidate soak evidence")
        if (
            (expected_sha256 is not None and sha256_bytes(payload) != expected_sha256)
            or (
                expected_sha256 is not None
                and SHA256_RE.fullmatch(expected_sha256) is None
            )
            or type(value.get("schemaVersion")) is not int
            or value.get("schemaVersion") != 1
            or value.get("kind") != "nuvion-iq9075-candidate-soak-evidence"
            or value.get("runId") != run_id
            or value.get("complete") is not True
        ):
            raise HarnessError("candidate soak evidence identity is invalid")
        assert_no_secret_material(value)
        return value

    @staticmethod
    def _candidate_unit(run_id: str) -> str:
        canonical_run_id(run_id)
        return f"nuvion-candidate-soak-{run_id.replace('-', '')}.service"

    def _candidate_unit_cgroup(self, unit: str) -> Path | None:
        if not re.fullmatch(r"nuvion-candidate-soak-[0-9a-f]{32}\.service", unit):
            raise HarnessError("candidate execution unit is invalid")
        shown = self.runner.run(
            [
                "/usr/bin/systemctl",
                "show",
                "--property=ControlGroup",
                "--value",
                unit,
            ],
            timeout=10,
        )
        raw = shown.stdout.strip()
        if shown.returncode != 0 or raw == "":
            active = self.runner.run(
                ["/usr/bin/systemctl", "is-active", unit], timeout=10
            )
            if active.returncode == 0 and active.stdout.strip() == "active":
                raise HarnessError("active candidate unit has no cgroup identity")
            return None
        relative = Path(raw.removeprefix("/"))
        if (
            not raw.startswith("/")
            or relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
            or len(raw) > 512
        ):
            raise HarnessError("candidate execution cgroup is invalid")
        return self.paths.cgroup_root.joinpath(*relative.parts)

    @staticmethod
    def _candidate_cgroup_empty(path: Path | None) -> bool:
        if path is None or (not path.exists() and not path.is_symlink()):
            # A cgroup can be removed only after its last process exits or moves.
            return True
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise HarnessError("candidate execution cgroup endpoint is unsafe")
        payload, _ = read_regular(
            path / "cgroup.events", maximum=64 * 1024, kernel_virtual_size=True
        )
        values: dict[str, int] = {}
        try:
            for line in payload.decode("ascii").splitlines():
                key, raw = line.split()
                if (
                    re.fullmatch(r"[a-z][a-z0-9_]{0,63}", key) is None
                    or key in values
                    or not raw.isdigit()
                ):
                    raise ValueError
                values[key] = int(raw)
        except (UnicodeError, ValueError) as exc:
            raise HarnessError("candidate cgroup.events is invalid") from exc
        if values.get("populated") not in {0, 1}:
            raise HarnessError("candidate cgroup.events lacks recursive population")
        return values["populated"] == 0

    def _nuvion_process_cgroups(self) -> dict[int, str]:
        processes: dict[int, str] = {}
        try:
            entries = list(self.paths.proc_root.iterdir())
        except OSError as exc:
            raise HarnessError("cannot enumerate nuvion processes") from exc
        for entry in entries:
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid < 2:
                continue
            try:
                status_payload, _ = read_regular(
                    entry / "status",
                    maximum=64 * 1024,
                    kernel_virtual_size=True,
                )
            except FileNotFoundError:
                continue
            try:
                uid_lines = [
                    line for line in status_payload.decode("ascii").splitlines()
                    if line.startswith("Uid:")
                ]
                if len(uid_lines) != 1:
                    raise ValueError
                uid_fields = uid_lines[0].split()[1:]
                if len(uid_fields) != 4 or any(not value.isdigit() for value in uid_fields):
                    raise ValueError
                uids = [int(value) for value in uid_fields]
            except (UnicodeError, ValueError) as exc:
                raise HarnessError("nuvion process credentials are invalid") from exc
            if self.nuvion_uid not in uids:
                continue
            if uids != [self.nuvion_uid] * 4:
                raise HarnessError("nuvion process has ambiguous credentials")
            try:
                cgroup_payload, _ = read_regular(
                    entry / "cgroup",
                    maximum=64 * 1024,
                    kernel_virtual_size=True,
                )
            except FileNotFoundError:
                continue
            try:
                cgroup_lines = cgroup_payload.decode("ascii").splitlines()
            except UnicodeError as exc:
                raise HarnessError("nuvion process cgroup is invalid") from exc
            if (
                len(cgroup_lines) != 1
                or not cgroup_lines[0].startswith("0::/")
                or len(cgroup_lines[0]) > 512
            ):
                raise HarnessError("nuvion process cgroup is invalid")
            group = cgroup_lines[0][3:]
            relative = PurePosixPath(group)
            if (
                not relative.is_absolute()
                or any(part in {"", ".", ".."} for part in relative.parts[1:])
            ):
                raise HarnessError("nuvion process cgroup is invalid")
            processes[pid] = group
        return processes

    def _candidate_uid_isolation_proof(
        self,
        *,
        expected_control_group: str | None,
        require_process: bool,
        timeout: float,
    ) -> dict[str, object]:
        deadline = self.monotonic() + timeout
        previous: dict[int, str] | None = None
        stable = 0
        while self.monotonic() < deadline:
            current = self._nuvion_process_cgroups()
            if expected_control_group is None:
                if current:
                    raise HarnessError("pre-existing nuvion process can escape candidate isolation")
            elif any(group != expected_control_group for group in current.values()):
                raise HarnessError("nuvion process escaped the candidate cgroup")
            if require_process and not current:
                previous = None
                stable = 0
                self.sleeper(0.05)
                continue
            if current == previous:
                stable += 1
            else:
                previous = current
                stable = 1
            if stable >= 2:
                return {
                    "schemaVersion": 1,
                    "uid": self.nuvion_uid,
                    "pids": sorted(current),
                    "controlGroup": expected_control_group,
                    "stableScans": stable,
                }
            self.sleeper(0.02)
        raise HarnessError("nuvion process isolation did not stabilize")

    @staticmethod
    def _systemctl_properties(result: CommandResult, names: set[str]) -> dict[str, str]:
        if result.returncode != 0:
            raise HarnessError("candidate systemd property inspection failed")
        values: dict[str, str] = {}
        for line in result.stdout.splitlines():
            key, separator, value = line.partition("=")
            if separator != "=" or key not in names or key in values:
                raise HarnessError("candidate systemd property output is invalid")
            values[key] = value
        if set(values) != names:
            raise HarnessError("candidate systemd property set is incomplete")
        return values

    def _show_candidate_properties(
        self, unit: str, names: set[str]
    ) -> dict[str, str]:
        if not re.fullmatch(r"nuvion-candidate-soak-[0-9a-f]{32}\.service", unit):
            raise HarnessError("candidate execution unit is invalid")
        result = self.runner.run(
            [
                "/usr/bin/systemctl",
                "show",
                *(f"--property={name}" for name in sorted(names)),
                unit,
            ],
            timeout=10,
        )
        return self._systemctl_properties(result, names)

    @staticmethod
    def _mountinfo_path(raw: str) -> str:
        try:
            decoded = re.sub(
                r"\\([0-7]{3})",
                lambda match: chr(int(match.group(1), 8)),
                raw,
            )
        except ValueError as exc:
            raise HarnessError("candidate mount path escaping is invalid") from exc
        value = PurePosixPath(decoded)
        if (
            not value.is_absolute()
            or ".." in value.parts
            or "\x00" in decoded
            or len(decoded) > 4096
        ):
            raise HarnessError("candidate mount path is invalid")
        return str(value)

    @staticmethod
    def _mount_limit(options: set[str], name: str) -> int:
        matches = [value.split("=", 1)[1] for value in options if value.startswith(name + "=")]
        if len(matches) != 1:
            raise HarnessError("candidate tmpfs limit is missing")
        match = re.fullmatch(r"([1-9][0-9]*)([kKmMgG]?)", matches[0])
        if match is None:
            raise HarnessError("candidate tmpfs limit is invalid")
        multiplier = {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3}[
            match.group(2).lower()
        ]
        return int(match.group(1)) * multiplier

    def _candidate_mount_sandbox_proof(
        self, pid: int, *, writable_path: str
    ) -> dict[str, object]:
        payload, _ = read_regular(
            self.paths.proc_root / str(pid) / "mountinfo",
            maximum=MAX_MOUNTINFO_BYTES,
            kernel_virtual_size=True,
        )
        mounts: dict[str, dict[str, object]] = {}
        try:
            for line in payload.decode("utf-8", errors="strict").splitlines():
                fields = line.split()
                separator = fields.index("-")
                if separator < 6 or len(fields) < separator + 4:
                    raise ValueError
                mount_id = int(fields[0])
                if mount_id < 1:
                    raise ValueError
                mount_point = self._mountinfo_path(fields[4])
                record = {
                    "mountId": mount_id,
                    "mountPoint": mount_point,
                    "mountOptions": set(fields[5].split(",")),
                    "fsType": fields[separator + 1],
                    "superOptions": set(fields[separator + 3].split(",")),
                }
                previous = mounts.get(mount_point)
                if previous is None or int(previous["mountId"]) < mount_id:
                    mounts[mount_point] = record
        except (UnicodeError, ValueError) as exc:
            raise HarnessError("candidate mount namespace evidence is invalid") from exc

        def effective_mount(target: str) -> Mapping[str, object] | None:
            candidates = [
                record
                for mount_point, record in mounts.items()
                if target == mount_point
                or target.startswith(mount_point.rstrip("/") + "/")
            ]
            if not candidates:
                return None
            return max(
                candidates,
                key=lambda item: (
                    len(PurePosixPath(str(item["mountPoint"])).parts),
                    int(item["mountId"]),
                ),
            )

        temporary: dict[str, dict[str, object]] = {}
        total_bytes = 0
        total_inodes = 0
        for path, limits in CANDIDATE_TMPFS_LIMITS.items():
            record = mounts.get(path)
            if (
                not isinstance(record, Mapping)
                or record.get("fsType") != "tmpfs"
                or "rw" not in record["mountOptions"]
                or not {"nosuid", "nodev"}.issubset(
                    record["mountOptions"] | record["superOptions"]
                )
            ):
                raise HarnessError("candidate private tmpfs mount is missing")
            size_bytes = self._mount_limit(record["superOptions"], "size")
            inode_limit = self._mount_limit(record["superOptions"], "nr_inodes")
            if size_bytes > limits["bytes"] or inode_limit > limits["inodes"]:
                raise HarnessError("candidate private tmpfs exceeds its hard limit")
            total_bytes += size_bytes
            total_inodes += inode_limit
            temporary[path] = {
                "mountId": record["mountId"],
                "fsType": "tmpfs",
                "sizeBytes": size_bytes,
                "inodeLimit": inode_limit,
                "readOnly": False,
            }

        read_only: dict[str, dict[str, object]] = {}
        for path in CANDIDATE_PERSISTENT_PATHS:
            record = effective_mount(path)
            if not isinstance(record, Mapping) or "ro" not in record["mountOptions"]:
                raise HarnessError("candidate persistent path is not read-only")
            read_only[path] = {
                "mountId": record["mountId"],
                "mountPoint": record["mountPoint"],
                "readOnly": True,
            }

        writable = mounts.get(writable_path)
        if not isinstance(writable, Mapping) or "rw" not in writable["mountOptions"]:
            raise HarnessError("candidate evidence path is not writable")

        inaccessible: dict[str, dict[str, object]] = {}
        for path in CANDIDATE_INACCESSIBLE_PATHS:
            record = mounts.get(path)
            visible = self.paths.proc_root / str(pid) / "root" / path.lstrip("/")
            metadata = visible.lstat()
            if (
                not isinstance(record, Mapping)
                or "ro" not in record["mountOptions"]
                or stat.S_IMODE(metadata.st_mode) != 0
                or not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != self.root_uid
            ):
                raise HarnessError("candidate inaccessible path isolation is missing")
            inaccessible[path] = {
                "mountId": record["mountId"],
                "mountPoint": path,
                "mode": "0000",
                "readOnly": True,
            }

        maximum_bytes = sum(item["bytes"] for item in CANDIDATE_TMPFS_LIMITS.values())
        maximum_inodes = sum(item["inodes"] for item in CANDIDATE_TMPFS_LIMITS.values())
        if total_bytes > maximum_bytes or total_inodes > maximum_inodes:
            raise HarnessError("candidate aggregate tmpfs limit is invalid")
        return {
            "temporaryFilesystems": temporary,
            "readOnlyPaths": read_only,
            "readWritePath": {
                "mountId": writable["mountId"],
                "mountPoint": writable_path,
                "readOnly": False,
            },
            "inaccessiblePaths": inaccessible,
            "totalTmpfsBytes": total_bytes,
            "totalTmpfsInodes": total_inodes,
        }

    def _candidate_execution_proof(
        self,
        unit: str,
        *,
        writable_path: str,
        uid_before: Mapping[str, object],
    ) -> dict[str, object]:
        lifecycle = {"ActiveState", "ControlGroup", "MainPID", "SubState"}
        names = lifecycle | set(CANDIDATE_SYSTEMD_EXPECTED_PROPERTIES)
        shown = self._show_candidate_properties(unit, names)
        try:
            pid = int(shown["MainPID"])
        except ValueError as exc:
            raise HarnessError("candidate MainPID is invalid") from exc
        control_group = shown["ControlGroup"]
        expected_group = "/system.slice/" + unit
        if (
            pid < 2
            or control_group != expected_group
            or shown["ActiveState"] != "active"
            or shown["SubState"] != "running"
            or {key: shown[key] for key in CANDIDATE_SYSTEMD_EXPECTED_PROPERTIES}
            != CANDIDATE_SYSTEMD_EXPECTED_PROPERTIES
        ):
            raise HarnessError("candidate systemd runtime proof is invalid")
        cgroup = self.paths.cgroup_root / control_group.lstrip("/")
        if self._candidate_cgroup_empty(cgroup):
            raise HarnessError("candidate cgroup was not populated during execution")
        cgroup_payload, _ = read_regular(
            self.paths.proc_root / str(pid) / "cgroup",
            maximum=64 * 1024,
            kernel_virtual_size=True,
        )
        try:
            cgroup_lines = cgroup_payload.decode("ascii").splitlines()
        except UnicodeError as exc:
            raise HarnessError("candidate PID cgroup membership is invalid") from exc
        if cgroup_lines != [f"0::{control_group}"]:
            raise HarnessError("candidate PID escaped its transient cgroup")
        mount_sandbox = self._candidate_mount_sandbox_proof(
            pid, writable_path=writable_path
        )
        uid_during = self._candidate_uid_isolation_proof(
            expected_control_group=control_group,
            require_process=True,
            timeout=30,
        )
        if (
            uid_before
            != {
                "schemaVersion": 1,
                "uid": self.nuvion_uid,
                "pids": [],
                "controlGroup": None,
                "stableScans": 2,
            }
        ):
            raise HarnessError("pre-candidate nuvion isolation proof is invalid")
        confirmed = self._show_candidate_properties(unit, lifecycle)
        if confirmed != {key: shown[key] for key in lifecycle}:
            raise HarnessError("candidate process changed during proof capture")
        return {
            "schemaVersion": 1,
            "unit": unit,
            "mainPid": pid,
            "controlGroup": control_group,
            "pidControlGroup": control_group,
            "recursivePopulated": True,
            "uidIsolation": {"before": dict(uid_before), "during": uid_during},
            "systemdProperties": {
                key: shown[key] for key in sorted(CANDIDATE_SYSTEMD_EXPECTED_PROPERTIES)
            },
            "mountSandbox": mount_sandbox,
        }

    def _wait_candidate_unit(self, unit: str, *, timeout: float) -> CommandResult:
        names = {
            "ActiveState",
            "ExecMainCode",
            "ExecMainStatus",
            "Result",
            "SubState",
        }
        deadline = self.monotonic() + timeout
        while self.monotonic() < deadline:
            shown = self._show_candidate_properties(unit, names)
            active_state = shown["ActiveState"]
            sub_state = shown["SubState"]
            if active_state == "active" and sub_state == "running":
                self.sleeper(0.25)
                continue
            if (active_state, sub_state) not in {
                ("active", "exited"),
                ("failed", "failed"),
                ("inactive", "dead"),
            }:
                raise HarnessError("candidate unit entered an unexpected state")
            try:
                main_code = int(shown["ExecMainCode"])
                main_status = int(shown["ExecMainStatus"])
            except ValueError as exc:
                raise HarnessError("candidate unit exit status is invalid") from exc
            passed = (
                shown["Result"] == "success"
                and main_code == 1
                and main_status == 0
            )
            return CommandResult(0 if passed else max(main_status, 1), "", "")
        raise HarnessError("candidate unit completion timed out")

    def _monitor_candidate_unit(
        self,
        unit: str,
        *,
        expected_control_group: str,
        timeout: float,
    ) -> tuple[CommandResult, dict[str, object]]:
        """Trusted outer timing/status collector with continuous UID sampling."""

        if expected_control_group != "/system.slice/" + unit:
            raise HarnessError("candidate collector cgroup binding is invalid")
        names = {
            "ActiveState",
            "ExecMainCode",
            "ExecMainStatus",
            "Result",
            "SubState",
        }
        started = self.monotonic()
        deadline = started + timeout
        samples = 0
        observed: set[int] = set()
        escape: dict[str, object] | None = None
        terminal: dict[str, str] | None = None
        while self.monotonic() < deadline:
            processes = self._nuvion_process_cgroups()
            samples += 1
            observed.update(processes)
            escaped = {
                pid: group
                for pid, group in processes.items()
                if group != expected_control_group
            }
            if escaped:
                escape = {
                    "pids": sorted(escaped),
                    "controlGroups": sorted(set(escaped.values())),
                }
                result = CommandResult(1, "", "")
                break
            shown = self._show_candidate_properties(unit, names)
            active_state = shown["ActiveState"]
            sub_state = shown["SubState"]
            if active_state == "active" and sub_state == "running":
                self.sleeper(CANDIDATE_UID_SCAN_INTERVAL_SECONDS)
                continue
            if (active_state, sub_state) not in {
                ("active", "exited"),
                ("failed", "failed"),
                ("inactive", "dead"),
            }:
                raise HarnessError("candidate unit entered an unexpected state")
            try:
                main_code = int(shown["ExecMainCode"])
                main_status = int(shown["ExecMainStatus"])
            except ValueError as exc:
                raise HarnessError("candidate unit exit status is invalid") from exc
            passed = (
                shown["Result"] == "success"
                and main_code == 1
                and main_status == 0
            )
            result = CommandResult(0 if passed else max(main_status, 1), "", "")
            terminal = dict(shown)
            break
        else:
            raise HarnessError("candidate unit completion timed out")
        elapsed = self.monotonic() - started
        duration_satisfied = elapsed >= CANDIDATE_REQUIRED_SOAK_SECONDS
        proof = {
            "schemaVersion": 1,
            "unit": unit,
            "controlGroup": expected_control_group,
            "requiredSeconds": CANDIDATE_REQUIRED_SOAK_SECONDS,
            "elapsedSeconds": round(elapsed, 6),
            "scanIntervalSeconds": CANDIDATE_UID_SCAN_INTERVAL_SECONDS,
            "sampleCount": samples,
            "observedPids": sorted(observed),
            "escapeDetected": escape,
            "allSamplesWithinCgroup": escape is None,
            "durationSatisfied": duration_satisfied,
            "terminalStatus": terminal,
        }
        if not duration_satisfied:
            result = CommandResult(1, "", "")
        return result, proof

    @staticmethod
    def _candidate_security_gates(
        execution: object,
        termination: object,
        *,
        unit: str,
        writable_path: str,
    ) -> dict[str, bool]:
        if not isinstance(execution, Mapping):
            return {
                "candidateBound": False,
                "resourceLimitsApplied": False,
                "boundedOutput": False,
                "persistentStateReadOnly": False,
                "cgroupTerminated": False,
            }
        properties = execution.get("systemdProperties")
        mounts = execution.get("mountSandbox")
        temporary = mounts.get("temporaryFilesystems") if isinstance(mounts, Mapping) else None
        expected_bytes = sum(item["bytes"] for item in CANDIDATE_TMPFS_LIMITS.values())
        expected_inodes = sum(item["inodes"] for item in CANDIDATE_TMPFS_LIMITS.values())
        resource_limits = bool(
            isinstance(properties, Mapping)
            and dict(properties) == CANDIDATE_SYSTEMD_EXPECTED_PROPERTIES
            and isinstance(temporary, Mapping)
            and set(temporary) == set(CANDIDATE_TMPFS_LIMITS)
            and all(
                isinstance(temporary.get(path), Mapping)
                and temporary[path].get("fsType") == "tmpfs"
                and temporary[path].get("readOnly") is False
                and type(temporary[path].get("sizeBytes")) is int
                and 0 < temporary[path]["sizeBytes"] <= limits["bytes"]
                and type(temporary[path].get("inodeLimit")) is int
                and 0 < temporary[path]["inodeLimit"] <= limits["inodes"]
                for path, limits in CANDIDATE_TMPFS_LIMITS.items()
            )
            and mounts.get("totalTmpfsBytes")
            == sum(temporary[path]["sizeBytes"] for path in CANDIDATE_TMPFS_LIMITS)
            and mounts.get("totalTmpfsBytes") <= expected_bytes
            and mounts.get("totalTmpfsInodes")
            == sum(temporary[path]["inodeLimit"] for path in CANDIDATE_TMPFS_LIMITS)
            and mounts.get("totalTmpfsInodes") <= expected_inodes
        )
        read_only = mounts.get("readOnlyPaths") if isinstance(mounts, Mapping) else None
        inaccessible = (
            mounts.get("inaccessiblePaths") if isinstance(mounts, Mapping) else None
        )
        write_proof = mounts.get("readWritePath") if isinstance(mounts, Mapping) else None

        def mount_covers(target: str, mount_point: object) -> bool:
            return bool(
                isinstance(mount_point, str)
                and mount_point.startswith("/")
                and (
                    target == mount_point
                    or target.startswith(mount_point.rstrip("/") + "/")
                )
            )

        persistent_read_only = bool(
            isinstance(read_only, Mapping)
            and set(read_only) == set(CANDIDATE_PERSISTENT_PATHS)
            and all(
                isinstance(read_only.get(path), Mapping)
                and mount_covers(path, read_only[path].get("mountPoint"))
                and read_only[path].get("readOnly") is True
                for path in CANDIDATE_PERSISTENT_PATHS
            )
            and isinstance(write_proof, Mapping)
            and write_proof.get("mountPoint") == writable_path
            and write_proof.get("readOnly") is False
        )
        output_bounded = bool(
            resource_limits
            and properties.get("LimitCORE") == "0"
            and properties.get("LimitFSIZE") == "8388608"
            and properties.get("StandardOutput") == "null"
            and properties.get("StandardError") == "null"
            and isinstance(inaccessible, Mapping)
            and set(inaccessible) == set(CANDIDATE_INACCESSIBLE_PATHS)
            and all(
                isinstance(inaccessible.get(path), Mapping)
                and inaccessible[path].get("mountPoint") == path
                and inaccessible[path].get("mode") == "0000"
                and inaccessible[path].get("readOnly") is True
                for path in CANDIDATE_INACCESSIBLE_PATHS
            )
        )
        expected_group = "/system.slice/" + unit
        uid_isolation = execution.get("uidIsolation")
        uid_before = (
            uid_isolation.get("before") if isinstance(uid_isolation, Mapping) else None
        )
        uid_during = (
            uid_isolation.get("during") if isinstance(uid_isolation, Mapping) else None
        )
        uid_bound = bool(
            isinstance(uid_before, Mapping)
            and isinstance(uid_during, Mapping)
            and uid_before.get("schemaVersion") == 1
            and type(uid_before.get("uid")) is int
            and uid_before["uid"] >= 1
            and uid_before.get("pids") == []
            and uid_before.get("controlGroup") is None
            and uid_before.get("stableScans") == 2
            and uid_during.get("schemaVersion") == 1
            and uid_during.get("uid") == uid_before.get("uid")
            and isinstance(uid_during.get("pids"), list)
            and bool(uid_during["pids"])
            and all(type(pid) is int and pid >= 2 for pid in uid_during["pids"])
            and len(set(uid_during["pids"])) == len(uid_during["pids"])
            and uid_during.get("controlGroup") == expected_group
            and uid_during.get("stableScans") == 2
        )
        candidate_bound = bool(
            execution.get("unit") == unit
            and execution.get("controlGroup") == expected_group
            and execution.get("pidControlGroup") == expected_group
            and execution.get("recursivePopulated") is True
            and type(execution.get("mainPid")) is int
            and execution["mainPid"] >= 2
            and uid_bound
        )
        cgroup_terminated = bool(
            isinstance(termination, Mapping)
            and termination.get("unit") == unit
            and termination.get("controlGroup") == expected_group
            and termination.get("recursivePopulated") is False
            and termination.get("stopSucceeded") is True
            and termination.get("loadState") == "not-found"
            and termination.get("cgroupRemoved") is True
        )
        return {
            "candidateBound": candidate_bound,
            "resourceLimitsApplied": resource_limits,
            "boundedOutput": output_bounded,
            "persistentStateReadOnly": persistent_read_only,
            "cgroupTerminated": cgroup_terminated,
        }

    def _terminate_candidate_unit(
        self, unit: str, *, expected_control_group: str | None = None
    ) -> dict[str, object]:
        """Kill the complete transient cgroup and prove it empty before restore."""

        cgroup = self._candidate_unit_cgroup(unit)
        if expected_control_group is not None:
            expected_path = self.paths.cgroup_root / expected_control_group.lstrip("/")
            if cgroup is not None and cgroup != expected_path:
                raise HarnessError("candidate execution cgroup identity changed")
            cgroup = cgroup or expected_path
        control_group = (
            "/" + str(cgroup.relative_to(self.paths.cgroup_root))
            if cgroup is not None
            else None
        )
        initial_present = bool(
            cgroup is not None and (cgroup.exists() or cgroup.is_symlink())
        )
        initial_populated = not self._candidate_cgroup_empty(cgroup)
        signals: list[str] = []
        if initial_populated:
            for signal_name in ("SIGTERM", "SIGKILL"):
                killed = self.runner.run(
                    [
                        "/usr/bin/systemctl",
                        "kill",
                        "--kill-whom=all",
                        f"--signal={signal_name}",
                        unit,
                    ],
                    timeout=20,
                )
                if killed.returncode != 0:
                    raise HarnessError("candidate systemd kill failed")
                signals.append(signal_name)
                stopped = self.runner.run(
                    ["/usr/bin/systemctl", "stop", unit], timeout=45
                )
                if stopped.returncode != 0:
                    raise HarnessError("candidate systemd stop failed")
                if self._candidate_cgroup_empty(cgroup):
                    break
            else:
                raise HarnessError("candidate execution cgroup is not empty")
        elif cgroup is not None and initial_present:
            stopped = self.runner.run(
                ["/usr/bin/systemctl", "stop", unit], timeout=45
            )
            if stopped.returncode != 0:
                raise HarnessError("candidate systemd stop failed")

        current_cgroup = self._candidate_unit_cgroup(unit)
        if current_cgroup is not None and cgroup is not None and current_cgroup != cgroup:
            raise HarnessError("candidate execution cgroup identity changed")
        final_cgroup = current_cgroup or cgroup
        if not self._candidate_cgroup_empty(final_cgroup):
            raise HarnessError("candidate execution cgroup is not empty")
        active = self.runner.run(
            ["/usr/bin/systemctl", "is-active", unit], timeout=10
        )
        if active.returncode == 0 or active.stdout.strip() not in {
            "inactive",
            "failed",
            "unknown",
        }:
            raise HarnessError("candidate unit remained active after termination")
        lifecycle_names = {"ActiveState", "ControlGroup", "LoadState"}
        lifecycle = self._show_candidate_properties(unit, lifecycle_names)
        if lifecycle["LoadState"] not in {"loaded", "not-found"}:
            raise HarnessError("candidate unit load state is invalid")
        reset_performed = lifecycle["LoadState"] == "loaded"
        if reset_performed:
            reset = self.runner.run(
                ["/usr/bin/systemctl", "reset-failed", unit], timeout=10
            )
            if reset.returncode != 0:
                raise HarnessError("candidate systemd reset failed")
        deadline = self.monotonic() + 10
        while self.monotonic() < deadline:
            lifecycle = self._show_candidate_properties(unit, lifecycle_names)
            cgroup_removed = final_cgroup is None or (
                not final_cgroup.exists() and not final_cgroup.is_symlink()
            )
            if lifecycle == {
                "ActiveState": "inactive",
                "ControlGroup": "",
                "LoadState": "not-found",
            } and cgroup_removed:
                break
            if lifecycle["LoadState"] not in {"loaded", "not-found"}:
                raise HarnessError("candidate unit unload proof is invalid")
            self.sleeper(0.05)
        else:
            raise HarnessError("candidate transient unit did not unload completely")
        return {
            "schemaVersion": 1,
            "unit": unit,
            "controlGroup": control_group,
            "initialPresent": initial_present,
            "initialPopulated": initial_populated,
            "killSignals": signals,
            "stopSucceeded": True,
            "resetPerformed": reset_performed,
            "recursivePopulated": False,
            "loadState": lifecycle["LoadState"],
            "activeState": lifecycle["ActiveState"],
            "cgroupRemoved": True,
        }

    def candidate_soak(
        self,
        run_id: str,
        *,
        candidate_bundle: str | Path,
        bundle_sha256: str,
        candidate_bom: str | Path,
        bom_sha256: str,
        oak_harness: str | Path,
        harness_sha256: str,
    ) -> dict[str, object]:
        """Run a candidate outside signed slots and restore the rolled-back LKG."""

        with (
            self._run_lock(run_id, usb=True),
            self._candidate_writer_lock(run_id),
        ):
            state = self._load_state(run_id)
            # Candidate work is subordinate to the transaction-wide guard;
            # an OAK/candidate deadman may add protection but never replace it.
            self._require_transaction_guard(run_id, state)
            existing = state.get("candidateSoak")
            if existing is not None and not isinstance(existing, Mapping):
                raise HarnessError("candidate soak journal is invalid")
            requested_digests = {
                "bundleSha256": bundle_sha256,
                "bomSha256": bom_sha256,
                "harnessSha256": harness_sha256,
            }
            if isinstance(existing, Mapping) and existing.get("phase") == "COMPLETE":
                if existing.get("inputDigests") != requested_digests:
                    raise HarnessError("candidate soak retry input identity mismatch")
                completed_evidence = self._load_candidate_soak_evidence(
                    run_id,
                    expected_sha256=str(existing.get("evidenceSha256") or ""),
                )
                outcome = completed_evidence.get("outcome")
                if isinstance(outcome, Mapping) and outcome.get("status") == "passed":
                    candidate_identity = completed_evidence.get("candidate")
                    if not isinstance(candidate_identity, Mapping):
                        raise HarnessError("completed candidate identity is invalid")
                    completed_slot = self._candidate_slot_path(
                        run_id, str(candidate_identity.get("bomDigest") or "")
                    )
                    if (
                        candidate_identity.get("slot") != str(completed_slot)
                        or existing.get("candidateSlot") != str(completed_slot)
                    ):
                        raise HarnessError("completed candidate slot identity is invalid")
                    if completed_slot.exists() or completed_slot.is_symlink():
                        self._remove_candidate_slot(completed_slot)
                    completed_inputs: list[CandidateInput] = []
                    for role, path, digest in (
                        ("candidate-bundle", candidate_bundle, bundle_sha256),
                        ("candidate-bom", candidate_bom, bom_sha256),
                        ("oak-harness", oak_harness, harness_sha256),
                    ):
                        staged_path = Path(path)
                        if staged_path.exists() or staged_path.is_symlink():
                            completed_inputs.append(
                                self._read_candidate_input(
                                    run_id, role, staged_path, digest
                                )
                            )
                    self._cleanup_candidate_inputs(completed_inputs)
                self._remove_pinned_candidate_harness(run_id, existing)
                return completed_evidence
            inputs = (
                self._read_candidate_input(
                    run_id, "candidate-bundle", candidate_bundle, bundle_sha256
                ),
                self._read_candidate_input(
                    run_id, "candidate-bom", candidate_bom, bom_sha256
                ),
                self._read_candidate_input(
                    run_id, "oak-harness", oak_harness, harness_sha256
                ),
            )
            by_role = {item.role: item for item in inputs}
            transaction = state.get("trustTransaction")
            if not isinstance(transaction, dict):
                raise HarnessError("candidate soak requires the Fleet trust transaction")
            manifest = self._transaction_manifest(run_id, transaction)
            scenario = manifest["scenario"]
            fleet_evidence = state.get("fleetEvidence")
            phase = existing.get("phase") if isinstance(existing, Mapping) else None
            if existing is not None and phase not in CANDIDATE_PHASES:
                raise HarnessError("candidate soak journal phase is invalid")
            if phase is None or phase in {"PREPARED", "STAGING", "STAGED"}:
                if (
                    not isinstance(fleet_evidence, Mapping)
                    or fleet_evidence.get("complete") is not True
                    or fleet_evidence.get("scenario") != "oak-fault-rollback"
                    or transaction.get("phase") != "APPLIED"
                    or transaction.get("liveVerified") is not True
                    or not self._transaction_matches(run_id, transaction)
                ):
                    raise HarnessError(
                        "candidate soak requires completed signed rollback evidence"
                    )
                slots = self._slot_snapshot()
                expected_candidate_release = (
                    "releases/" + str(scenario["expectedBomDigest"])[7:]
                )
                if (
                    scenario.get("type") != "oak-fault-rollback"
                    or slots.get("current") != scenario.get("expectedPreviousSlot")
                    or slots.get("previous") != expected_candidate_release
                ):
                    raise HarnessError("candidate soak rollback slots are not terminal")
                updater = self._probe_updater()
                update = updater.get("update")
                current = str(slots["current"])
                current_marker = (
                    self._release_marker(current)
                    if current.startswith("releases/")
                    else {}
                )
                previous_marker = self._release_marker(expected_candidate_release)
                if (
                    not isinstance(update, Mapping)
                    or "healthDeadline" in update
                    or not self._scenario_gate(
                        manifest,
                        updater,
                        current,
                        str(slots["previous"]),
                        self._slot_version(current),
                        current_marker,
                        previous_marker,
                        state,
                    )
                ):
                    raise HarnessError("candidate soak updater rollback is not terminal")
                update_sequence = update.get("sequence")
                if (
                    isinstance(update_sequence, bool)
                    or not isinstance(update_sequence, int)
                    or update_sequence < 1
                ):
                    raise HarnessError(
                        "candidate soak rollback command sequence is invalid"
                    )
                rollback_terminal = {
                    "commandId": update.get("commandId"),
                    "sequence": update_sequence,
                    "phase": update.get("phase"),
                    "bomDigest": update.get("bomDigest"),
                    "releaseSequence": update.get("releaseSequence"),
                    "healthDeadline": update.get("healthDeadline"),
                }
                units_before = self._unit_snapshot()
                if not all(
                    item.get("active") is True
                    for item in units_before.values()
                ):
                    raise HarnessError("candidate soak requires healthy Fleet services")
                oak_before = self.verify_oak()
                baseline_runtime = self._agent_process_identity(current)
                bom_payload, bom_metadata = read_regular(
                    by_role["candidate-bom"].path, maximum=MAX_TRUST_BYTES
                )
                if (
                    (bom_metadata.st_dev, bom_metadata.st_ino)
                    != (
                        by_role["candidate-bom"].device,
                        by_role["candidate-bom"].inode,
                    )
                    or sha256_bytes(bom_payload) != bom_sha256
                ):
                    raise HarnessError("candidate BOM changed during validation")
                bom = validate_candidate_bom(
                    bom_payload,
                    manifest=manifest,
                    bundle_sha256=bundle_sha256,
                    bundle_size=by_role["candidate-bundle"].size,
                )
                release_marker = {
                    "schemaVersion": 2,
                    "bomDigest": bom["bomDigest"],
                    **dict(scenario["release"]),
                }
                if previous_marker != release_marker:
                    raise HarnessError("candidate BOM differs from authenticated slot")
                authenticated_bom_payload, authenticated_bom_metadata = read_regular(
                    self.paths.install_root
                    / expected_candidate_release
                    / ".nuvion/release-bom.json",
                    maximum=MAX_TRUST_BYTES,
                )
                if (
                    authenticated_bom_metadata.st_uid != self.root_uid
                    or authenticated_bom_metadata.st_mode & 0o022
                    or strict_json(
                        authenticated_bom_payload, label="authenticated release BOM"
                    )
                    != bom
                ):
                    raise HarnessError(
                        "candidate BOM differs from authenticated release bytes"
                    )
                candidate_slot = self._candidate_slot_path(
                    run_id, str(bom["bomDigest"])
                )
                candidate_incoming = self._candidate_incoming_path(candidate_slot)
                pinned_harness = self._candidate_harness_execution_path(
                    run_id, harness_sha256
                )
                soak: dict[str, Any] = {
                    "phase": "PREPARED",
                    "startedAt": self.clock(),
                    "inputDigests": requested_digests,
                    "candidateSlot": str(candidate_slot),
                    "candidateIncomingPath": str(candidate_incoming),
                    "harnessExecution": {
                        "path": str(pinned_harness),
                        "sha256": harness_sha256,
                    },
                    "baselineSlots": slots,
                    "releaseTreesBefore": self._release_tree_snapshot(slots),
                    "baselineRuntime": baseline_runtime,
                    "oakBefore": oak_before,
                    "unitsBefore": units_before,
                    "rollbackTerminal": rollback_terminal,
                    "fleetEvidenceSha256": fleet_evidence.get("sha256"),
                }
            else:
                if not isinstance(existing, dict):
                    raise HarnessError("candidate soak journal is invalid")
                soak = existing
                if soak.get("inputDigests") != requested_digests:
                    raise HarnessError("candidate soak retry input identity mismatch")
                if soak.get("phase") not in CANDIDATE_PHASES:
                    raise HarnessError("candidate soak journal phase is invalid")
                bom_payload, bom_metadata = read_regular(
                    by_role["candidate-bom"].path, maximum=MAX_TRUST_BYTES
                )
                if (
                    (bom_metadata.st_dev, bom_metadata.st_ino)
                    != (
                        by_role["candidate-bom"].device,
                        by_role["candidate-bom"].inode,
                    )
                    or sha256_bytes(bom_payload) != bom_sha256
                ):
                    raise HarnessError("candidate BOM changed during validation")
                bom = validate_candidate_bom(
                    bom_payload,
                    manifest=manifest,
                    bundle_sha256=bundle_sha256,
                    bundle_size=by_role["candidate-bundle"].size,
                )
                release_marker = {
                    "schemaVersion": 2,
                    "bomDigest": bom["bomDigest"],
                    **dict(scenario["release"]),
                }
                candidate_slot = self._candidate_slot_path(
                    run_id, str(bom["bomDigest"])
                )

            expected_incoming = self._candidate_incoming_path(candidate_slot)
            expected_harness = self._candidate_harness_execution_path(
                run_id, harness_sha256
            )
            if (
                soak.get("candidateSlot") != str(candidate_slot)
                or soak.get("candidateIncomingPath") != str(expected_incoming)
                or soak.get("harnessExecution")
                != {"path": str(expected_harness), "sha256": harness_sha256}
            ):
                raise HarnessError("candidate soak execution journal binding is invalid")
            # Arm and verify root-owned recovery before PREPARED/STAGING becomes
            # durable or any large candidate bytes are parsed/extracted.
            self._ensure_candidate_deadman(run_id, state, soak)
            phase = str(soak["phase"])
            if phase == "STAGING":
                soak["candidateIncomingRecovered"] = self._recover_candidate_incoming(
                    run_id, soak
                )
                state["candidateSoak"] = soak
                self._save_state(run_id, state)
            pinned_harness = self._pin_candidate_harness(run_id, harness_sha256)

            rollback_terminal = soak.get("rollbackTerminal")
            if not isinstance(rollback_terminal, Mapping):
                raise HarnessError("candidate soak rollback boundary is missing")

            if CANDIDATE_PHASES.index(phase) <= CANDIDATE_PHASES.index("STAGING"):
                soak["phase"] = "STAGING"
                state["candidateSoak"] = soak
                self._save_state(run_id, state)
                candidate_slot, control_sha256 = self._stage_candidate_bundle(
                    run_id=run_id,
                    bundle=by_role["candidate-bundle"],
                    bom=bom,
                    harness_sha256=harness_sha256,
                    release_marker=release_marker,
                )
                soak.update(
                    {
                        "phase": "STAGED",
                        "controlMarkerSha256": control_sha256,
                        "stagedAt": self.clock(),
                    }
                )
                state["candidateSoak"] = soak
                self._save_state(run_id, state)
            else:
                control = self._candidate_control_marker(
                    run_id=run_id,
                    slot=candidate_slot,
                    bom=bom,
                    bundle_sha256=bundle_sha256,
                    harness_sha256=harness_sha256,
                )
                control_sha256 = self._verify_candidate_slot(
                    candidate_slot,
                    control=control,
                    release_marker=release_marker,
                    bom=bom,
                )
                if soak.get("controlMarkerSha256") != control_sha256:
                    raise HarnessError("candidate soak marker journal mismatch")

            raw_path = self.paths.state_root / run_id / "candidate-soak-raw.json"
            operation_failure: str | None = None
            raw_evidence: dict[str, Any] | None = None
            quiesced = CANDIDATE_PHASES.index(str(soak["phase"])) >= CANDIDATE_PHASES.index(
                "QUIESCING"
            )
            try:
                if not quiesced:
                    soak["phase"] = "QUIESCING"
                    state["candidateSoak"] = soak
                    self._save_state(run_id, state)
                    # From this point any partial stop must take the restoration
                    # path; a failed systemctl call cannot leave a mixed writer set.
                    quiesced = True
                    self._stop_writers()
                    anti_replay = self._anti_replay_snapshot()
                    self._validate_candidate_anti_replay(
                        anti_replay, rollback_terminal
                    )
                    persistent_state = self._candidate_persistent_state_snapshot()
                    current_slots = self._slot_snapshot()
                    if current_slots != soak["baselineSlots"]:
                        raise HarnessError("signed slot pointers changed while quiescing")
                    release_trees = self._release_tree_snapshot(current_slots)
                    if release_trees != soak.get("releaseTreesBefore"):
                        raise HarnessError("signed release trees changed while quiescing")
                    soak.update(
                        {
                            "phase": "QUIESCED",
                            "antiReplay": anti_replay,
                            "persistentStateBefore": persistent_state,
                            "releaseTreesBefore": release_trees,
                            "quiescedAt": self.clock(),
                        }
                    )
                    state["candidateSoak"] = soak
                    self._save_state(run_id, state)
                elif any(self._unit_status(unit)["active"] is True for unit in UNITS):
                    self._stop_writers()
                if quiesced and not isinstance(soak.get("antiReplay"), Mapping):
                    anti_replay = self._anti_replay_snapshot()
                    self._validate_candidate_anti_replay(
                        anti_replay, rollback_terminal
                    )
                    persistent_state = self._candidate_persistent_state_snapshot()
                    current_slots = self._slot_snapshot()
                    if current_slots != soak["baselineSlots"]:
                        raise HarnessError("signed slot pointers changed while resuming")
                    release_trees = self._release_tree_snapshot(current_slots)
                    if release_trees != soak.get("releaseTreesBefore"):
                        raise HarnessError("signed release trees changed while resuming")
                    soak.update(
                        {
                            "phase": "QUIESCED",
                            "antiReplay": anti_replay,
                            "persistentStateBefore": persistent_state,
                            "releaseTreesBefore": release_trees,
                            "quiescedAt": self.clock(),
                        }
                    )
                    state["candidateSoak"] = soak
                    self._save_state(run_id, state)

                if not isinstance(soak.get("persistentStateBefore"), Mapping):
                    raise HarnessError(
                        "candidate persistent state baseline is missing"
                    )
                if not isinstance(soak.get("releaseTreesBefore"), Mapping):
                    raise HarnessError("candidate release tree baseline is missing")
                if soak["phase"] == "RUNNING" and not raw_path.exists():
                    operation_failure = "INTERRUPTED_BEFORE_CAPTURE"
                elif CANDIDATE_PHASES.index(str(soak["phase"])) <= CANDIDATE_PHASES.index(
                    "QUIESCED"
                ):
                    execution_unit = self._candidate_unit(run_id)
                    pinned_harness = self._verify_pinned_candidate_harness(
                        run_id, soak, harness_sha256
                    )
                    uid_before = self._candidate_uid_isolation_proof(
                        expected_control_group=None,
                        require_process=False,
                        timeout=2,
                    )
                    soak["uidIsolationBefore"] = uid_before
                    soak["phase"] = "RUNNING"
                    soak["runningAt"] = self.clock()
                    soak["executionUnit"] = execution_unit
                    soak["executionCgroupEmpty"] = False
                    soak["executedHarnessSha256"] = harness_sha256
                    state["candidateSoak"] = soak
                    self._save_state(run_id, state)
                    try:
                        result = self.runner.run(
                            [
                                "/usr/bin/systemd-run",
                                f"--unit={execution_unit}",
                                "--quiet",
                                "--property=Type=exec",
                                "--property=RemainAfterExit=yes",
                                "--property=KillMode=control-group",
                                "--property=SendSIGKILL=yes",
                                "--property=TimeoutStopSec=30s",
                                "--property=RuntimeMaxSec=720s",
                                *(
                                    f"--property={value}"
                                    for value in CANDIDATE_RESOURCE_PROPERTIES
                                ),
                                *(
                                    f"--property={value}"
                                    for value in CANDIDATE_SANDBOX_PROPERTIES
                                ),
                                *(
                                    f"--property=ReadOnlyPaths={value}"
                                    for value in CANDIDATE_PERSISTENT_PATHS
                                ),
                                "--property=ReadWritePaths="
                                + str(self.paths.state_root / run_id),
                                "/usr/bin/env",
                                "-u",
                                "PYTHONPATH",
                                f"NUVION_AGENT_PYTHON={SYSTEM_PYTHON}",
                                "NUVION_AGENT_SITE_PACKAGES="
                                + str(candidate_slot / CANDIDATE_SITE_PACKAGES_RELATIVE),
                                f"NUVION_SYSTEM_PYTHON={SYSTEM_PYTHON}",
                                "NUVION_IQ9075_OAK_SOAK_SECONDS=120",
                                "NUVION_IQ9075_OAK_MAX_RSS_SLOPE_MIB_PER_MIN=2",
                                "NUVION_IQ9075_OAK_MAX_RSS_RANGE_MIB=32",
                                str(pinned_harness),
                                "--camera",
                                "oak",
                                "--evidence-output",
                                str(raw_path),
                                "--expected-version",
                                str(bom["agentVersion"]),
                                "--expected-component-sha",
                                str(bom["componentSha"]),
                                "--expected-bom-digest",
                                str(bom["bomDigest"]),
                                "--run-id",
                                run_id,
                                "--expected-slot-kind",
                                "candidate",
                                "--expected-slot-path",
                                str(candidate_slot),
                                "--expected-control-marker-sha256",
                                control_sha256,
                            ],
                            timeout=30,
                        )
                        if result.returncode != 0:
                            raise HarnessError("candidate transient unit failed to start")
                        execution_proof = self._candidate_execution_proof(
                            execution_unit,
                            writable_path=str(self.paths.state_root / run_id),
                            uid_before=uid_before,
                        )
                        soak["executionProof"] = execution_proof
                        state["candidateSoak"] = soak
                        self._save_state(run_id, state)
                        result, collector_proof = self._monitor_candidate_unit(
                            execution_unit,
                            expected_control_group=str(
                                execution_proof["controlGroup"]
                            ),
                            timeout=730,
                        )
                        soak["collectorProof"] = collector_proof
                        state["candidateSoak"] = soak
                        self._save_state(run_id, state)
                    finally:
                        expected_group = None
                        proof = soak.get("executionProof")
                        if isinstance(proof, Mapping) and isinstance(
                            proof.get("controlGroup"), str
                        ):
                            expected_group = proof["controlGroup"]
                        termination_proof = self._terminate_candidate_unit(
                            execution_unit,
                            expected_control_group=expected_group,
                        )
                        soak["terminationProof"] = termination_proof
                        soak["executionCgroupEmpty"] = (
                            termination_proof.get("recursivePopulated") is False
                        )
                        soak["executionStoppedAt"] = self.clock()
                        uid_after = self._candidate_uid_isolation_proof(
                            expected_control_group=None,
                            require_process=False,
                            timeout=2,
                        )
                        collector = soak.get("collectorProof")
                        if isinstance(collector, dict):
                            collector["afterTermination"] = uid_after
                            soak["collectorProof"] = collector
                        state["candidateSoak"] = soak
                        self._save_state(run_id, state)
                    if result.returncode != 0:
                        operation_failure = "OAK_HARNESS_FAILED"
                if raw_path.exists():
                    raw_payload, raw_metadata = read_regular(
                        raw_path, maximum=1024 * 1024
                    )
                    if (
                        raw_metadata.st_uid != self.root_uid
                        or stat.S_IMODE(raw_metadata.st_mode) != 0o600
                    ):
                        raise HarnessError("candidate raw evidence metadata is unsafe")
                    raw_evidence = strict_json(
                        raw_payload, label="candidate raw evidence"
                    )
                    runtime_identity = raw_evidence.get("runtimeIdentity")
                    if (
                        raw_evidence.get("schemaVersion") != 3
                        or raw_evidence.get("kind")
                        != "nuvion-iq9075-oak-soak-result"
                        or raw_evidence.get("runId") != run_id
                        or raw_evidence.get("slotKind") != "candidate"
                        or not isinstance(runtime_identity, Mapping)
                        or runtime_identity.get("pythonPath")
                        != SYSTEM_PYTHON
                        or runtime_identity.get("sitePackagesPath")
                        != str(candidate_slot / CANDIDATE_SITE_PACKAGES_RELATIVE)
                        or runtime_identity.get("buildInfoPath")
                        != str(
                            candidate_slot
                            / CANDIDATE_SITE_PACKAGES_RELATIVE
                            / "nuvion_app/build_info.py"
                        )
                        or runtime_identity.get("candidateSlot") != str(candidate_slot)
                        or runtime_identity.get("controlMarkerSha256")
                        != control_sha256
                    ):
                        raise HarnessError("candidate raw evidence binding is invalid")
                    assert_no_secret_material(raw_evidence)
                    soak["rawEvidenceSha256"] = sha256_bytes(raw_payload)
                    outcome = raw_evidence.get("outcome")
                    if (
                        not isinstance(outcome, Mapping)
                        or outcome.get("status") != "passed"
                    ):
                        operation_failure = operation_failure or "OAK_HARNESS_FAILED"
                else:
                    operation_failure = operation_failure or "OAK_EVIDENCE_MISSING"
                soak["phase"] = "CAPTURED"
                soak["capturedAt"] = self.clock()
                state["candidateSoak"] = soak
                self._save_state(run_id, state)
            except (HarnessError, OSError, tarfile.TarError):
                operation_failure = operation_failure or "CANDIDATE_SOAK_FAILED"
            finally:
                if quiesced:
                    try:
                        soak["phase"] = "RESTORING"
                        state["candidateSoak"] = soak
                        self._save_state(run_id, state)
                        if self._slot_snapshot() != soak["baselineSlots"]:
                            raise HarnessError("signed slot pointers changed during soak")
                        if self._anti_replay_snapshot() != soak["antiReplay"]:
                            raise HarnessError("updater anti-replay journal changed during soak")
                        oak_during = self.verify_oak()
                        if (
                            oak_during.get("port") != soak["oakBefore"].get("port")
                            or oak_during.get("mxidSha256")
                            != soak["oakBefore"].get("mxidSha256")
                        ):
                            raise HarnessError("OAK identity changed during candidate soak")
                        execution_unit = soak.get("executionUnit")
                        if soak.get("runningAt") is not None:
                            expected_unit = self._candidate_unit(run_id)
                            if execution_unit is None:
                                execution_unit = expected_unit
                            if (
                                not isinstance(execution_unit, str)
                                or execution_unit != expected_unit
                            ):
                                raise HarnessError("candidate execution unit is invalid")
                            termination = soak.get("terminationProof")
                            if not isinstance(termination, Mapping):
                                proof = soak.get("executionProof")
                                expected_group = (
                                    proof.get("controlGroup")
                                    if isinstance(proof, Mapping)
                                    and isinstance(proof.get("controlGroup"), str)
                                    else None
                                )
                                termination = self._terminate_candidate_unit(
                                    execution_unit,
                                    expected_control_group=expected_group,
                                )
                                soak["terminationProof"] = termination
                            soak["executionCgroupEmpty"] = (
                                termination.get("recursivePopulated") is False
                            )
                            state["candidateSoak"] = soak
                            self._save_state(run_id, state)
                        if soak.get("runningAt") is not None and (
                            soak.get("executionCgroupEmpty") is not True
                        ):
                            raise HarnessError("candidate execution cgroup is not empty")
                        termination = soak.get("terminationProof")
                        if soak.get("runningAt") is not None and (
                            not isinstance(termination, Mapping)
                            or termination.get("unit") != self._candidate_unit(run_id)
                            or termination.get("controlGroup")
                            != "/system.slice/" + self._candidate_unit(run_id)
                            or termination.get("recursivePopulated") is not False
                            or termination.get("stopSucceeded") is not True
                            or termination.get("loadState") != "not-found"
                            or termination.get("activeState") != "inactive"
                            or termination.get("cgroupRemoved") is not True
                        ):
                            raise HarnessError(
                                "candidate recursive cgroup termination proof is invalid"
                            )
                        persistent_state_after = (
                            self._candidate_persistent_state_snapshot()
                        )
                        if persistent_state_after != soak.get(
                            "persistentStateBefore"
                        ):
                            raise HarnessError(
                                "candidate modified protected persistent state"
                            )
                        soak["persistentStateAfter"] = persistent_state_after
                        release_trees_after = self._release_tree_snapshot(
                            self._slot_snapshot()
                        )
                        if release_trees_after != soak.get("releaseTreesBefore"):
                            raise HarnessError(
                                "candidate modified signed release tree state"
                            )
                        soak["releaseTreesAfter"] = release_trees_after
                        self._restore_units(soak["unitsBefore"])
                        restored_runtime = self._agent_process_identity(
                            str(soak["baselineSlots"]["current"])
                        )
                        baseline_runtime = soak["baselineRuntime"]
                        if (
                            (
                                restored_runtime["bootId"],
                                restored_runtime["pid"],
                                restored_runtime["startTicks"],
                            )
                            == (
                                baseline_runtime["bootId"],
                                baseline_runtime["pid"],
                                baseline_runtime["startTicks"],
                            )
                            or self._slot_snapshot() != soak["baselineSlots"]
                            or self._anti_replay_snapshot() != soak["antiReplay"]
                        ):
                            raise HarnessError("fresh baseline runtime proof did not converge")
                        oak_after = self.verify_oak()
                        if (
                            oak_after.get("port") != soak["oakBefore"].get("port")
                            or oak_after.get("mxidSha256")
                            != soak["oakBefore"].get("mxidSha256")
                        ):
                            raise HarnessError("restored baseline OAK identity changed")
                        updater_after = self._probe_updater()
                        update_after = updater_after.get("update")
                        if (
                            not isinstance(update_after, Mapping)
                            or update_after.get("phase") != "ROLLED_BACK"
                            or "healthDeadline" in update_after
                        ):
                            raise HarnessError("restored updater lifecycle changed")
                        # Candidate isolation is not a completion boundary.  The
                        # temporary Fleet trust transaction must be rolled back
                        # to the pre-run production bytes before evidence can be
                        # emitted or the recovery deadman can be disarmed.
                        self._restore_transaction(run_id, state, transaction)
                        production_restoration = (
                            self._production_restoration_attestation(
                                run_id, state, transaction
                            )
                        )
                        restored_runtime = self._agent_process_identity(
                            str(soak["baselineSlots"]["current"])
                        )
                        soak.update(
                            {
                                "phase": "RESTORED",
                                "restoredAt": self.clock(),
                                "restoredRuntime": restored_runtime,
                                "oakAfter": oak_after,
                                "productionRestoration": production_restoration,
                            }
                        )
                        state["candidateSoak"] = soak
                        self._save_state(run_id, state)
                    except BaseException as exc:
                        try:
                            self._stop_writers()
                        except BaseException:
                            pass
                        raise HarnessError(
                            "candidate soak restoration failed; runtime remains fail-closed"
                        ) from exc

            self._remove_pinned_candidate_harness(run_id, soak)
            soak["harnessExecutionRemoved"] = True
            if soak.get("executedHarnessSha256") != harness_sha256:
                raise HarnessError("candidate soak lacks exact harness execution proof")
            post_slots = self._slot_snapshot()
            post_anti_replay = self._anti_replay_snapshot()
            execution_unit = self._candidate_unit(run_id)
            security_gates = self._candidate_security_gates(
                soak.get("executionProof"),
                soak.get("terminationProof"),
                unit=execution_unit,
                writable_path=str(self.paths.state_root / run_id),
            )
            collector_proof = soak.get("collectorProof")
            trusted_duration = bool(
                isinstance(collector_proof, Mapping)
                and collector_proof.get("durationSatisfied") is True
                and type(collector_proof.get("elapsedSeconds")) in {int, float}
                and collector_proof["elapsedSeconds"]
                >= CANDIDATE_REQUIRED_SOAK_SECONDS
            )
            continuous_uid_isolation = bool(
                isinstance(collector_proof, Mapping)
                and collector_proof.get("allSamplesWithinCgroup") is True
                and collector_proof.get("escapeDetected") is None
                and isinstance(collector_proof.get("afterTermination"), Mapping)
                and collector_proof["afterTermination"].get("pids") == []
            )
            release_trees_unchanged = soak.get("releaseTreesBefore") == soak.get(
                "releaseTreesAfter"
            )
            slots_unchanged = (
                post_slots == soak.get("baselineSlots") and release_trees_unchanged
            )
            anti_replay_unchanged = post_anti_replay == soak.get("antiReplay")
            oak_identity_unchanged = bool(
                isinstance(soak.get("oakAfter"), Mapping)
                and soak["oakAfter"].get("port") == soak["oakBefore"].get("port")
                and soak["oakAfter"].get("mxidSha256")
                == soak["oakBefore"].get("mxidSha256")
            )
            baseline_runtime = soak.get("baselineRuntime")
            restored_runtime = soak.get("restoredRuntime")
            fresh_baseline = bool(
                isinstance(baseline_runtime, Mapping)
                and isinstance(restored_runtime, Mapping)
                and (
                    restored_runtime.get("bootId"),
                    restored_runtime.get("pid"),
                    restored_runtime.get("startTicks"),
                )
                != (
                    baseline_runtime.get("bootId"),
                    baseline_runtime.get("pid"),
                    baseline_runtime.get("startTicks"),
                )
            )
            persistent_unchanged = soak.get("persistentStateBefore") == soak.get(
                "persistentStateAfter"
            )
            production_restoration = soak.get("productionRestoration")
            production_trust_restored = bool(
                isinstance(production_restoration, Mapping)
                and production_restoration.get("transactionPhase") == "RESTORED"
                and production_restoration.get("manifestSha256")
                == transaction.get("manifestSha256")
                and isinstance(production_restoration.get("sha256"), str)
                and SHA256_RE.fullmatch(production_restoration["sha256"]) is not None
                and transaction.get("phase") == "RESTORED"
            )
            signed_rollback_terminal = soak.get("rollbackTerminal") == soak.get(
                "antiReplay", {}
            ).get("latest")
            harness_bytes_pinned = (
                soak.get("executedHarnessSha256") == harness_sha256
            )
            harness_copy_removed = soak.get("harnessExecutionRemoved") is True
            safety_passed = all(security_gates.values()) and all(
                (
                    signed_rollback_terminal,
                    slots_unchanged,
                    anti_replay_unchanged,
                    oak_identity_unchanged,
                    fresh_baseline,
                    persistent_unchanged,
                    production_trust_restored,
                    trusted_duration,
                    continuous_uid_isolation,
                    harness_bytes_pinned,
                    harness_copy_removed,
                )
            )
            passed = (
                operation_failure is None
                and raw_evidence is not None
                and safety_passed
            )
            if not passed and operation_failure is None:
                operation_failure = "CANDIDATE_SECURITY_PROOF_FAILED"
            evidence = {
                "schemaVersion": 1,
                "kind": "nuvion-iq9075-candidate-soak-evidence",
                "protocolVersion": PROTOCOL_VERSION,
                "runId": run_id,
                "startedAt": soak["startedAt"],
                "completedAt": self.clock(),
                "complete": True,
                "outcome": {
                    "status": "passed" if passed else "failed",
                    "errorCode": operation_failure,
                },
                "candidate": {
                    "slotKind": "candidate",
                    "slot": str(candidate_slot),
                    "bomDigest": bom["bomDigest"],
                    "bundleSha256": bundle_sha256,
                    "bomSha256": bom_sha256,
                    "harnessSha256": soak["executedHarnessSha256"],
                    "controlMarkerSha256": control_sha256,
                },
                "fleetEvidenceSha256": soak["fleetEvidenceSha256"],
                "rawEvidenceSha256": soak.get("rawEvidenceSha256"),
                "rawEvidence": raw_evidence,
                "executionProof": soak.get("executionProof"),
                "collectorProof": collector_proof,
                "terminationProof": soak.get("terminationProof"),
                "productionRestoration": production_restoration,
                "pre": {
                    "slots": soak["baselineSlots"],
                    "antiReplay": soak["antiReplay"],
                    "oak": soak["oakBefore"],
                    "runtime": soak["baselineRuntime"],
                    "persistentState": soak["persistentStateBefore"],
                    "releaseTrees": soak["releaseTreesBefore"],
                },
                "post": {
                    "restoredAt": soak["restoredAt"],
                    "slots": post_slots,
                    "antiReplay": post_anti_replay,
                    "oak": soak["oakAfter"],
                    "runtime": soak["restoredRuntime"],
                    "persistentState": soak["persistentStateAfter"],
                    "releaseTrees": soak["releaseTreesAfter"],
                },
                "gates": {
                    "signedRollbackTerminal": signed_rollback_terminal,
                    "candidateBound": security_gates["candidateBound"],
                    "rawEvidencePreserved": raw_evidence is not None,
                    "slotsUnchanged": slots_unchanged,
                    "releaseTreesUnchanged": release_trees_unchanged,
                    "antiReplayUnchanged": anti_replay_unchanged,
                    "oakIdentityUnchanged": oak_identity_unchanged,
                    "freshBaselineProcess": fresh_baseline,
                    "harnessBytesPinned": harness_bytes_pinned,
                    "harnessCopyRemoved": harness_copy_removed,
                    "resourceLimitsApplied": security_gates[
                        "resourceLimitsApplied"
                    ],
                    "boundedOutput": security_gates["boundedOutput"],
                    "persistentStateReadOnly": security_gates[
                        "persistentStateReadOnly"
                    ],
                    "persistentStateUnchanged": persistent_unchanged,
                    "productionTrustRestored": production_trust_restored,
                    "trustedSoakDuration": trusted_duration,
                    "continuousUidIsolation": continuous_uid_isolation,
                    "cgroupTerminated": security_gates["cgroupTerminated"],
                    "harnessPassed": passed,
                },
            }
            assert_no_secret_material(evidence)
            evidence_payload = (
                json.dumps(
                    evidence,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            atomic_write(
                self.paths.state_root / run_id / "candidate-soak-evidence.json",
                evidence_payload,
                mode=0o600,
                uid=self.root_uid,
                gid=self.root_gid,
            )
            soak.update(
                {
                    "phase": "COMPLETE",
                    "complete": True,
                    "passed": passed,
                    "evidenceSha256": sha256_bytes(evidence_payload),
                    "completedAt": evidence["completedAt"],
                }
            )
            state["candidateSoak"] = soak
            self._save_state(run_id, state)
            if passed:
                self._remove_candidate_slot(candidate_slot)
                self._cleanup_candidate_inputs(inputs)
            # Keep recovery armed across the SSH response boundary. Only the
            # transaction-wide cleanup path may disarm it after secret purge,
            # durable cleanup journal, lease release, and root attestation.
            return evidence

    def _deadman_unit(self, run_id: str) -> str:
        return f"nuvion-oak-deadman-{run_id.replace('-', '')}.service"

    def _transaction_deadman_unit(self, run_id: str) -> str:
        canonical_run_id(run_id)
        return f"nuvion-fleet-transaction-{run_id.replace('-', '')}.service"

    def _transaction_deadman_timer(self, run_id: str) -> str:
        return self._transaction_deadman_unit(run_id).removesuffix(
            ".service"
        ) + ".timer"

    def _validate_transaction_guard(
        self, run_id: str, value: object
    ) -> dict[str, Any]:
        allowed = {
            "unit",
            "armed",
            "lifecycle",
            "writerEpoch",
            "controller",
            "deadlineSeconds",
            "deadlineAt",
            "armingAt",
            "armedAt",
            "disarmedAt",
            "stopped",
            "recoveredAt",
            "recoveredBy",
            "recoveryUnitExitPending",
            "activeWriter",
        }
        if (
            not isinstance(value, Mapping)
            or not {
                "unit",
                "armed",
                "lifecycle",
                "writerEpoch",
                "controller",
                "deadlineSeconds",
                "deadlineAt",
            }.issubset(value)
            or not set(value).issubset(allowed)
            or value.get("unit") != self._transaction_deadman_unit(run_id)
            or not isinstance(value.get("armed"), bool)
            or value.get("lifecycle") not in {"ARMING", "ARMED", "DISARMED"}
            or (value.get("armed") is False and value.get("lifecycle") != "DISARMED")
            or (value.get("armed") is True and value.get("lifecycle") == "DISARMED")
            or not isinstance(value.get("writerEpoch"), str)
            or RUN_ID_RE.fullmatch(str(value.get("writerEpoch"))) is None
            or value.get("deadlineSeconds") != TRANSACTION_DEADMAN_SECONDS
            or ("stopped" in value and not isinstance(value.get("stopped"), bool))
        ):
            raise HarnessError("transaction recovery guard journal is invalid")
        guard = dict(value)
        active_writer = value.get("activeWriter")
        if active_writer is not None:
            if (
                not isinstance(active_writer, Mapping)
                or set(active_writer) != {"writerEpoch", "controller", "boundAt"}
                or not isinstance(active_writer.get("writerEpoch"), str)
                or RUN_ID_RE.fullmatch(str(active_writer.get("writerEpoch"))) is None
                or not isinstance(active_writer.get("boundAt"), str)
            ):
                raise HarnessError("transaction active writer journal is invalid")
            guard["activeWriter"] = {
                **dict(active_writer),
                "controller": self._validate_controller_identity(
                    active_writer.get("controller")
                ),
            }
        deadline_at = value.get("deadlineAt")
        if not isinstance(deadline_at, str):
            raise HarnessError("transaction recovery guard deadline is invalid")
        try:
            parsed_deadline = datetime.fromisoformat(
                deadline_at.removesuffix("Z") + "+00:00"
            )
        except ValueError as exc:
            raise HarnessError("transaction recovery guard deadline is invalid") from exc
        if (
            not deadline_at.endswith("Z")
            or parsed_deadline.tzinfo != timezone.utc
        ):
            raise HarnessError("transaction recovery guard deadline is invalid")
        guard["controller"] = self._validate_controller_identity(
            value.get("controller")
        )
        return guard

    def _start_transaction_deadman(
        self,
        run_id: str,
        *,
        epoch: str,
        controller: Mapping[str, object],
    ) -> str:
        unit = self._transaction_deadman_unit(run_id)
        if RUN_ID_RE.fullmatch(epoch) is None:
            raise HarnessError("transaction guard writer epoch is invalid")
        checked = self._validate_controller_identity(controller)
        tool = str(self.tool_path)
        if any(character.isspace() for character in tool):
            raise HarnessError("board tool path is unsafe for transient unit")
        result = self.runner.run(
            [
                "/usr/bin/systemd-run",
                f"--unit={unit}",
                "--collect",
                f"--on-active={TRANSACTION_DEADMAN_SECONDS}s",
                "--timer-property=AccuracySec=1s",
                "--property=Type=oneshot",
                "--property=LimitCORE=0",
                f"--property=RuntimeMaxSec={TRANSACTION_DEADMAN_RECOVERY_SECONDS}s",
                f"--property=TimeoutStartSec={TRANSACTION_DEADMAN_RECOVERY_SECONDS}s",
                f"--property=TimeoutStopSec={TRANSACTION_DEADMAN_RECOVERY_SECONDS}s",
                "--property=Restart=on-failure",
                "--property=RestartSec=15s",
                "--property=StartLimitIntervalSec=0",
                SYSTEM_PYTHON,
                "-I",
                tool,
                "cleanup",
                "--run-id",
                run_id,
                "--transaction-deadman-only",
                "--transaction-deadman-epoch",
                epoch,
                "--transaction-controller-pid",
                str(checked["pid"]),
                "--transaction-controller-start-ticks",
                str(checked["startTicks"]),
                "--transaction-controller-boot-id",
                str(checked["bootId"]),
            ],
            timeout=20,
        )
        if result.returncode != 0:
            raise HarnessError("cannot arm the transaction recovery guard")
        active = self.runner.run(
            [
                "/usr/bin/systemctl",
                "is-active",
                self._transaction_deadman_timer(run_id),
            ],
            timeout=10,
        )
        if active.returncode != 0 or active.stdout.strip() != "active":
            raise HarnessError("transaction recovery guard timer is not active")
        return unit

    def _stop_transaction_deadman(self, unit: str) -> None:
        if not re.fullmatch(
            r"nuvion-fleet-transaction-[0-9a-f]{32}\.service", unit
        ):
            raise HarnessError("transaction recovery guard unit is invalid")
        timer = unit.removesuffix(".service") + ".timer"
        for target in (timer, unit):
            stopped = self.runner.run(
                ["/usr/bin/systemctl", "stop", target], timeout=30
            )
            if stopped.returncode not in {0, 5}:
                raise HarnessError("transaction recovery guard did not stop")
            status = self.runner.run(
                ["/usr/bin/systemctl", "is-active", target], timeout=10
            )
            if status.returncode == 0 or status.stdout.strip() == "active":
                raise HarnessError("transaction recovery guard remains active")
            reset = self.runner.run(
                ["/usr/bin/systemctl", "reset-failed", target], timeout=10
            )
            if reset.returncode not in {0, 1, 5}:
                raise HarnessError("transaction recovery guard reset failed")

    def _ensure_transaction_guard(
        self, run_id: str, state: dict[str, Any]
    ) -> None:
        existing = state.get("transactionGuard")
        if existing is not None:
            guard = self._validate_transaction_guard(run_id, existing)
            if guard["armed"] is not True:
                raise HarnessError("restored transaction guard cannot be rearmed")
            timer = self.runner.run(
                [
                    "/usr/bin/systemctl",
                    "is-active",
                    self._transaction_deadman_timer(run_id),
                ],
                timeout=10,
            )
            if timer.returncode != 0 or timer.stdout.strip() != "active":
                raise HarnessError(
                    "armed transaction guard is unavailable; recovery is required"
                )
            return
        transaction = state.get("trustTransaction")
        if transaction is not None and (
            not isinstance(transaction, Mapping)
            or transaction.get("phase") != "PREPARED"
        ):
            raise HarnessError("transaction guard must precede trust application")
        if transaction is None and state.get("foundation", {}).get("verified") is not True:
            raise HarnessError("transaction guard requires verified foundation")
        epoch = str(uuid.uuid4())
        controller = self._validate_controller_identity(self.controller_identity())
        arming_at = self.clock()
        try:
            deadline_at = (
                datetime.fromisoformat(arming_at.removesuffix("Z") + "+00:00")
                + timedelta(seconds=TRANSACTION_DEADMAN_SECONDS)
            ).isoformat(timespec="seconds").replace("+00:00", "Z")
        except ValueError as exc:
            raise HarnessError("transaction guard clock is invalid") from exc
        guard: dict[str, Any] = {
            "unit": self._transaction_deadman_unit(run_id),
            "armed": True,
            "lifecycle": "ARMING",
            "writerEpoch": epoch,
            "controller": controller,
            "deadlineSeconds": TRANSACTION_DEADMAN_SECONDS,
            "deadlineAt": deadline_at,
            "armingAt": arming_at,
        }
        # Arm the external owner first.  Its invocation-bound arguments can
        # reconstruct this exact guard if this process dies before the journal
        # bind; no secret snapshot/archive exists before this call returns.
        self._start_transaction_deadman(
            run_id, epoch=epoch, controller=controller
        )
        guard.update({"lifecycle": "ARMED", "armedAt": self.clock()})
        state["transactionGuard"] = guard
        self._save_state(run_id, state)

    def _require_transaction_guard(
        self, run_id: str, state: Mapping[str, Any]
    ) -> dict[str, Any]:
        guard = self._validate_transaction_guard(
            run_id, state.get("transactionGuard")
        )
        if guard.get("armed") is not True or guard.get("lifecycle") != "ARMED":
            raise HarnessError("transaction recovery guard is not armed")
        timer = self.runner.run(
            [
                "/usr/bin/systemctl",
                "is-active",
                self._transaction_deadman_timer(run_id),
            ],
            timeout=10,
        )
        if timer.returncode != 0 or timer.stdout.strip() != "active":
            raise HarnessError("transaction recovery guard timer is not active")
        timer_unit = self._transaction_deadman_timer(run_id)
        resolved = self.runner.run(
            [
                "/usr/bin/busctl",
                "--system",
                "--json=short",
                "call",
                "org.freedesktop.systemd1",
                "/org/freedesktop/systemd1",
                "org.freedesktop.systemd1.Manager",
                "GetUnit",
                "s",
                timer_unit,
            ],
            timeout=10,
        )
        if resolved.returncode != 0:
            raise HarnessError("transaction recovery timer identity is unavailable")
        resolved_value = strict_json(
            resolved.stdout.encode("utf-8"), label="systemd timer identity"
        )
        resolved_data = resolved_value.get("data")
        if (
            isinstance(resolved_data, list)
            and len(resolved_data) == 1
            and isinstance(resolved_data[0], str)
        ):
            object_path = resolved_data[0]
        elif isinstance(resolved_data, str):
            object_path = resolved_data
        else:
            raise HarnessError("transaction recovery timer identity is invalid")
        escaped_unit = "".join(
            character
            if character.isascii() and character.isalnum()
            else f"_{ord(character):02x}"
            for character in timer_unit
        )
        expected_object_path = "/org/freedesktop/systemd1/unit/" + escaped_unit
        if (
            set(resolved_value) != {"type", "data"}
            or resolved_value.get("type") != "o"
            or object_path != expected_object_path
        ):
            raise HarnessError("transaction recovery timer identity is invalid")
        shown = self.runner.run(
            [
                "/usr/bin/busctl",
                "--system",
                "--json=short",
                "get-property",
                "org.freedesktop.systemd1",
                object_path,
                "org.freedesktop.systemd1.Timer",
                "NextElapseUSecMonotonic",
            ],
            timeout=10,
        )
        if shown.returncode != 0:
            raise HarnessError("transaction recovery timer deadline is unavailable")
        shown_value = strict_json(
            shown.stdout.encode("utf-8"), label="systemd timer deadline"
        )
        next_monotonic_usec = shown_value.get("data")
        try:
            deadline_valid = (
                set(shown_value) == {"type", "data"}
                and shown_value.get("type") == "t"
                and type(next_monotonic_usec) is int
                and next_monotonic_usec > 0
            )
        except TypeError:
            deadline_valid = False
        if not deadline_valid:
            raise HarnessError("transaction recovery timer deadline is unavailable")
        required_remaining = (
            CANDIDATE_DEADMAN_SECONDS
            + CANDIDATE_DEADMAN_RECOVERY_SECONDS
            + 120
        )
        remaining = int(next_monotonic_usec) / 1_000_000 - self.monotonic()
        if remaining < required_remaining:
            raise HarnessError("transaction recovery guard deadline is too near")
        return guard

    def _disarm_transaction_guard(
        self,
        run_id: str,
        state: dict[str, Any],
        *,
        recovery_unit: bool = False,
    ) -> None:
        guard = self._validate_transaction_guard(
            run_id, state.get("transactionGuard")
        )
        transaction = state.get("trustTransaction")
        cleanup = state.get("cleanup")
        proof = self._cleanup_attestation(run_id, state)
        transaction_restored = transaction is None or (
            isinstance(transaction, Mapping)
            and transaction.get("phase") == "RESTORED"
        )
        if (
            not transaction_restored
            or not isinstance(cleanup, Mapping)
            or cleanup.get("complete") is not True
        ):
            raise HarnessError("transaction guard cannot be disarmed before cleanup")
        self._require_complete_cleanup_attestation(
            proof, require_transaction=isinstance(transaction, Mapping)
        )
        if guard.get("armed") is True:
            guard.update(
                {
                    "armed": False,
                    "lifecycle": "DISARMED",
                    "disarmedAt": self.clock(),
                }
            )
            if recovery_unit:
                guard["recoveryUnitExitPending"] = True
            state["transactionGuard"] = guard
            self._save_state(run_id, state)
        if recovery_unit:
            return
        if guard.get("stopped") is not True:
            self._stop_transaction_deadman(str(guard["unit"]))
            guard["stopped"] = True
            guard.pop("recoveryUnitExitPending", None)
            state["transactionGuard"] = guard
            self._save_state(run_id, state)

    def _candidate_deadman_unit(self, run_id: str) -> str:
        canonical_run_id(run_id)
        return f"nuvion-candidate-deadman-{run_id.replace('-', '')}.service"

    def _candidate_deadman_timer(self, run_id: str) -> str:
        return self._candidate_deadman_unit(run_id).removesuffix(".service") + ".timer"

    def _start_candidate_deadman(
        self,
        run_id: str,
        *,
        epoch: str,
        controller: Mapping[str, object],
    ) -> str:
        unit = self._candidate_deadman_unit(run_id)
        if RUN_ID_RE.fullmatch(epoch) is None:
            raise HarnessError("candidate deadman writer epoch is invalid")
        controller = self._validate_controller_identity(controller)
        tool = str(self.tool_path)
        if any(character.isspace() for character in tool):
            raise HarnessError("board tool path is unsafe for transient unit")
        result = self.runner.run(
            [
                "/usr/bin/systemd-run",
                f"--unit={unit}",
                "--collect",
                f"--on-active={CANDIDATE_DEADMAN_SECONDS}s",
                "--timer-property=AccuracySec=1s",
                "--property=Type=oneshot",
                "--property=LimitCORE=0",
                f"--property=RuntimeMaxSec={CANDIDATE_DEADMAN_RECOVERY_SECONDS}s",
                f"--property=TimeoutStartSec={CANDIDATE_DEADMAN_RECOVERY_SECONDS}s",
                f"--property=TimeoutStopSec={CANDIDATE_DEADMAN_RECOVERY_SECONDS}s",
                "--property=Restart=on-failure",
                "--property=RestartSec=15s",
                "--property=StartLimitIntervalSec=0",
                SYSTEM_PYTHON,
                "-I",
                tool,
                "cleanup",
                "--run-id",
                run_id,
                "--candidate-deadman-only",
                "--candidate-deadman-epoch",
                epoch,
                "--candidate-controller-pid",
                str(controller["pid"]),
                "--candidate-controller-start-ticks",
                str(controller["startTicks"]),
                "--candidate-controller-boot-id",
                str(controller["bootId"]),
            ],
            timeout=20,
        )
        if result.returncode != 0:
            raise HarnessError("cannot arm the candidate recovery deadman")
        active = self.runner.run(
            ["/usr/bin/systemctl", "is-active", self._candidate_deadman_timer(run_id)],
            timeout=10,
        )
        if active.returncode != 0 or active.stdout.strip() != "active":
            raise HarnessError("candidate recovery deadman timer is not active")
        return unit

    def _stop_candidate_deadman(self, unit: str) -> None:
        if not re.fullmatch(
            r"nuvion-candidate-deadman-[0-9a-f]{32}\.service", unit
        ):
            raise HarnessError("candidate deadman unit is invalid")
        timer = unit.removesuffix(".service") + ".timer"
        for target in (timer, unit):
            stopped = self.runner.run(
                ["/usr/bin/systemctl", "stop", target], timeout=30
            )
            if stopped.returncode not in {0, 5}:
                raise HarnessError("candidate deadman unit did not stop")
            status = self.runner.run(
                ["/usr/bin/systemctl", "is-active", target], timeout=10
            )
            if status.returncode == 0 or status.stdout.strip() == "active":
                raise HarnessError("candidate deadman unit remains active")
            reset = self.runner.run(
                ["/usr/bin/systemctl", "reset-failed", target], timeout=10
            )
            if reset.returncode not in {0, 1, 5}:
                raise HarnessError("candidate deadman reset failed")
            deadline = self.monotonic() + 10
            while self.monotonic() < deadline:
                shown = self.runner.run(
                    [
                        "/usr/bin/systemctl",
                        "show",
                        "--property=ActiveState",
                        "--property=LoadState",
                        target,
                    ],
                    timeout=10,
                )
                properties = dict(
                    line.split("=", 1)
                    for line in shown.stdout.splitlines()
                    if "=" in line
                )
                if shown.returncode == 0 and properties == {
                    "ActiveState": "inactive",
                    "LoadState": "not-found",
                }:
                    break
                self.sleeper(0.05)
            else:
                raise HarnessError("candidate deadman unit did not unload")

    def _ensure_candidate_deadman(
        self, run_id: str, state: dict[str, Any], soak: dict[str, Any]
    ) -> None:
        expected_unit = self._candidate_deadman_unit(run_id)
        deadman = soak.get("deadman")
        if deadman is not None and (
            not isinstance(deadman, dict)
            or deadman.get("unit") != expected_unit
            or not isinstance(deadman.get("armed"), bool)
            or deadman.get("lifecycle")
            not in {"DISARMED", "ARMING", "ARMED"}
        ):
            raise HarnessError("candidate deadman journal binding is invalid")
        if isinstance(deadman, dict) and deadman["armed"]:
            self._validate_controller_identity(deadman.get("controller"))
            if not isinstance(deadman.get("writerEpoch"), str) or RUN_ID_RE.fullmatch(
                deadman["writerEpoch"]
            ) is None:
                raise HarnessError("candidate deadman writer epoch is invalid")
            active = self.runner.run(
                [
                    "/usr/bin/systemctl",
                    "is-active",
                    self._candidate_deadman_timer(run_id),
                ],
                timeout=10,
            )
            if active.returncode != 0 or active.stdout.strip() != "active":
                raise HarnessError("armed candidate deadman is not active")
            # Never rebind a live timer to a new controller/epoch: the old
            # callback could already be executing. Recovery must converge the
            # old epoch before a later run can be started.
            raise HarnessError("armed candidate soak requires recovery before resume")
        if soak.get("phase") in {"RESTORED", "COMPLETE"}:
            return
        epoch = str(uuid.uuid4())
        controller = self._validate_controller_identity(
            self.controller_identity()
        )
        try:
            # External guard first: if systemd accepted the timer but this
            # process dies before the journal bind, invocation-bound arguments
            # let the callback conservatively restore the APPLIED transaction.
            self._start_candidate_deadman(
                run_id, epoch=epoch, controller=controller
            )
        except BaseException:
            transaction = state.get("trustTransaction")
            if isinstance(transaction, dict):
                self._restore_transaction(run_id, state, transaction)
                soak["phase"] = "RESTORED"
                soak["productionRestoration"] = (
                    self._production_restoration_attestation(
                        run_id, state, transaction
                    )
                )
                soak["deadman"] = {
                    "unit": expected_unit,
                    "armed": False,
                    "lifecycle": "DISARMED",
                    "writerEpoch": epoch,
                    "controller": controller,
                    "armFailedAt": self.clock(),
                }
                state["candidateSoak"] = soak
                self._save_state(run_id, state)
            raise
        deadman = {
            "unit": expected_unit,
            "armed": True,
            "lifecycle": "ARMED",
            "writerEpoch": epoch,
            "controller": controller,
            "armedAt": self.clock(),
        }
        soak["deadman"] = deadman
        state["candidateSoak"] = soak
        self._save_state(run_id, state)

    def _disarm_candidate_deadman(
        self, run_id: str, state: dict[str, Any], soak: dict[str, Any]
    ) -> None:
        deadman = soak.get("deadman")
        expected_unit = self._candidate_deadman_unit(run_id)
        if (
            not isinstance(deadman, dict)
            or deadman.get("unit") != expected_unit
            or not isinstance(deadman.get("armed"), bool)
            or (
                deadman.get("armed") is False
                and deadman.get("lifecycle") != "DISARMED"
            )
            or soak.get("phase") not in {"RESTORED", "COMPLETE"}
            or not isinstance(soak.get("productionRestoration"), Mapping)
            or not isinstance(state.get("trustTransaction"), Mapping)
            or state["trustTransaction"].get("phase") != "RESTORED"
            or not isinstance(state.get("cleanup"), Mapping)
            or state["cleanup"].get("complete") is not True
        ):
            raise HarnessError("candidate deadman cannot be disarmed before restore")
        self._require_complete_cleanup_attestation(
            self._cleanup_attestation(run_id, state),
            require_transaction=True,
        )
        # Persist the safe restored boundary first. A timer callback racing
        # systemctl stop must observe lifecycle=DISARMED before doing any work.
        if deadman.get("armed") is True:
            deadman.update(
                {
                    "armed": False,
                    "lifecycle": "DISARMED",
                    "disarmedAt": self.clock(),
                }
            )
            soak["deadman"] = deadman
            state["candidateSoak"] = soak
            self._save_state(run_id, state)
        if deadman.get("stopped") is True:
            return
        self._stop_candidate_deadman(expected_unit)
        deadman["stopped"] = True
        soak["deadman"] = deadman
        state["candidateSoak"] = soak
        self._save_state(run_id, state)

    def _start_deadman(self, run_id: str) -> str:
        unit = self._deadman_unit(run_id)
        tool = str(self.tool_path)
        if any(character.isspace() for character in tool):
            raise HarnessError("board tool path is unsafe for transient unit")
        command = f"/usr/bin/python3 -I {tool} cleanup --run-id {run_id} --deadman-only"
        result = self.runner.run(
            [
                "/usr/bin/systemd-run",
                f"--unit={unit}",
                "--collect",
                "--property=Type=oneshot",
                "--property=LimitCORE=0",
                "--property=RuntimeMaxSec=180",
                "--property=TimeoutStopSec=45",
                f"--property=ExecStopPost={command}",
                "/usr/bin/sleep",
                str(DEADMAN_SECONDS),
            ],
            timeout=20,
        )
        if result.returncode != 0:
            raise HarnessError("cannot arm the OAK recovery deadman")
        if (
            self.runner.run(
                ["/usr/bin/systemctl", "is-active", unit], timeout=10
            ).stdout.strip()
            != "active"
        ):
            raise HarnessError("OAK recovery deadman is not active")
        return unit

    def _stop_deadman(self, unit: str) -> None:
        if not re.fullmatch(r"nuvion-oak-deadman-[0-9a-f]{32}\.service", unit):
            raise HarnessError("deadman unit is invalid")
        result = self.runner.run(["/usr/bin/systemctl", "stop", unit], timeout=20)
        if result.returncode not in {0, 5}:
            raise HarnessError("deadman unit did not stop")
        status = self.runner.run(["/usr/bin/systemctl", "is-active", unit], timeout=10)
        if status.returncode == 0 or status.stdout.strip() == "active":
            raise HarnessError("deadman unit remains active")
        reset = self.runner.run(
            ["/usr/bin/systemctl", "reset-failed", unit], timeout=10
        )
        if reset.returncode not in {0, 1, 5}:
            raise HarnessError("deadman reset failed")

    def _write_usb(self, action: str, port: str) -> None:
        if action not in {"bind", "unbind"}:
            raise HarnessError("USB action is invalid")
        safe_port = canonical_oak_port(port)
        path = self.paths.usb_bind if action == "bind" else self.paths.usb_unbind
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise HarnessError("USB control endpoint is unsafe")
        descriptor = os.open(
            path,
            os.O_WRONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            payload = safe_port.encode("ascii")
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise HarnessError("USB control write was incomplete")
                view = view[written:]
        finally:
            os.close(descriptor)
        if self.usb_write_hook is not None:
            self.usb_write_hook(action, safe_port)

    def _poll_detached(self, port: str, timeout: float = 15) -> None:
        safe_port = canonical_oak_port(port)
        device = self._usb_device_path(safe_port)
        deadline = self.monotonic() + timeout
        while self.monotonic() < deadline:
            if not device.exists() and not device.is_symlink():
                return
            try:
                self.verify_oak(require_bound=False, expected_port=safe_port)
                driver = device / "driver"
                if not driver.exists() and not driver.is_symlink():
                    return
                metadata = driver.lstat()
                if not stat.S_ISLNK(metadata.st_mode):
                    raise HarnessError("OAK USB driver endpoint became unsafe")
            except (FileNotFoundError, HarnessError):
                # The kernel may remove the device node entirely while an
                # unbind/reset is in flight. Absence is detached; an identity
                # mismatch while the node exists is not.
                if not device.exists() and not device.is_symlink():
                    return
                raise
            self.sleeper(0.25)
        raise HarnessError("OAK USB device did not detach after unbind")

    def _recover_oak(self, port: str, timeout: float = 30) -> None:
        safe_port = canonical_oak_port(port)
        # Never write a journal-provided topology into the root usb bind
        # endpoint until the still-present unbound device proves exact OAK
        # identity, speed, USB1 ancestry, and global uniqueness.
        self.verify_oak(require_bound=False, expected_port=safe_port)
        deadline = self.monotonic() + timeout
        last_error: BaseException | None = None
        while self.monotonic() < deadline:
            try:
                self._write_usb("bind", safe_port)
            except OSError as exc:
                if exc.errno not in {errno.EBUSY, errno.EEXIST, errno.ENODEV}:
                    last_error = exc
            try:
                self.verify_oak(require_bound=True, expected_port=safe_port)
                return
            except (HarnessError, OSError) as exc:
                last_error = exc
            self.sleeper(0.5)
        raise HarnessError("OAK USB device did not recover before the deadline") from last_error

    def arm_oak_fault(self, run_id: str) -> dict[str, object]:
        with self._run_lock(run_id, usb=True):
            state = self._load_state(run_id)
            transaction = state.get("trustTransaction")
            if (
                not isinstance(transaction, dict)
                or transaction.get("phase") != "APPLIED"
            ):
                raise HarnessError("trust transaction must be APPLIED before OAK fault")
            manifest = self._transaction_manifest(run_id, transaction)
            scenario = manifest["scenario"]
            if scenario["type"] != "oak-fault-rollback":
                raise HarnessError("immutable scenario is not OAK fault rollback")
            hold = int(scenario["holdSeconds"])
            updater = self._probe_updater()
            update = updater.get("update")
            if not isinstance(update, Mapping):
                raise HarnessError("updater has no candidate update to fault")
            if (
                update.get("commandId") != scenario["expectedCommandId"]
                or update.get("bomDigest") != scenario["expectedBomDigest"]
                or update.get("candidateSlot") != scenario["expectedCandidateSlot"]
                or update.get("phase")
                not in {
                    "ACTIVATING",
                    "BOOT_HEALTHY",
                    "FUNCTIONAL_HEALTHY",
                    "COMMIT_GATE",
                }
            ):
                raise HarnessError("live updater candidate identity is not faultable")
            digest = str(scenario["expectedBomDigest"])[7:]
            if self._slot_link("current", required=True) != f"releases/{digest}":
                raise HarnessError("active slot is not the exact candidate")
            candidate_service = self._unit_status("nuv-agent.service")
            candidate_pid = candidate_service.get("mainPid")
            if (
                candidate_service.get("active") is not True
                or isinstance(candidate_pid, bool)
                or not isinstance(candidate_pid, int)
                or candidate_pid < 2
            ):
                raise HarnessError("candidate Agent PID is unavailable before OAK fault")
            oak = self.verify_oak()
            port = canonical_oak_port(oak["port"])
            fault = {
                "armed": True,
                "unit": self._deadman_unit(run_id),
                "port": port,
                "commandId": scenario["expectedCommandId"],
                "bomDigest": scenario["expectedBomDigest"],
                "candidateSlot": scenario["expectedCandidateSlot"],
                "candidatePid": candidate_pid,
                "armedAt": self.clock(),
            }
            state["oakFault"] = fault
            self._save_state(run_id, state)
            unit = self._start_deadman(run_id)

            def interrupted(_signum: int, _frame: object) -> None:
                raise InterruptedError("OAK fault interrupted")

            previous = {
                signum: signal.getsignal(signum)
                for signum in (signal.SIGINT, signal.SIGTERM)
            }
            for signum in previous:
                signal.signal(signum, interrupted)
            recovered = False
            try:
                self._write_usb("unbind", port)
                self._poll_detached(port)
                if hold:
                    self.sleeper(float(hold))
                self._recover_oak(port)
                latest = self._load_state(run_id)
                latest_fault = latest.get("oakFault")
                if not isinstance(latest_fault, dict):
                    raise HarnessError("OAK fault journal disappeared")
                latest_fault.update(
                    {"armed": False, "recovered": True, "recoveredAt": self.clock()}
                )
                latest["oakFault"] = latest_fault
                self._save_state(run_id, latest)
                try:
                    self._stop_deadman(unit)
                except BaseException:
                    # Never persist a disarmed claim when the bounded recovery
                    # unit could not be retired cleanly.
                    failed = self._load_state(run_id)
                    failed_fault = failed.get("oakFault")
                    if isinstance(failed_fault, dict):
                        failed_fault.update(
                            {
                                "armed": True,
                                "recovered": False,
                                "recoveryError": "DEADMAN_STOP_FAILED",
                            }
                        )
                        failed["oakFault"] = failed_fault
                        self._save_state(run_id, failed)
                    raise
                recovered = True
            finally:
                for signum, handler in previous.items():
                    signal.signal(signum, handler)
                if not recovered:
                    # Keep armed=true and the deadman active. Never claim a
                    # recovered camera before exact driver/identity/speed proof.
                    pass
            return {
                "schemaVersion": 1,
                "runId": run_id,
                "fault": "oak-usb-disconnect",
                "port": port,
                "holdSeconds": hold,
                "deadmanSeconds": DEADMAN_SECONDS,
                "recovered": True,
            }

    def _restore_transaction(
        self,
        run_id: str,
        state: dict[str, Any],
        transaction: dict[str, Any],
        *,
        start_runtime: bool = True,
    ) -> None:
        self._validate_transaction_shape(transaction)
        if transaction.get("phase") == "RESTORED":
            units_before = transaction.get("unitsBefore")
            if not isinstance(units_before, Mapping):
                raise HarnessError("transaction unit snapshot is invalid")
            if transaction.get("runtimeRestored") is False:
                if start_runtime:
                    self._restore_units(units_before)
                    transaction["runtimeRestored"] = True
                    state["trustTransaction"] = transaction
                    self._save_state(run_id, state)
                else:
                    self._restore_unit_enablement_for_boot(units_before)
            return
        transaction["phase"] = "RESTORING"
        transaction["restoringAt"] = self.clock()
        state["trustTransaction"] = transaction
        self._save_state(run_id, state)

        if start_runtime:
            self._stop_writers()
        else:
            self._stop_writers_for_boot()
        if any(
            self._unit_status(unit)["active"] is True
            for unit in ("nuv-agent-updater.service", "nuv-agent-updater.socket")
        ):
            raise HarnessError("updater writers did not stop before trust restoration")
        directory = self._transaction_dir(run_id)
        files = transaction.get("files")
        directories = transaction.get("directories")
        if not isinstance(files, Mapping) or not isinstance(directories, Mapping):
            raise HarnessError("transaction rollback metadata is invalid")
        for absolute, raw in files.items():
            if not isinstance(raw, Mapping):
                raise HarnessError("transaction rollback file is invalid")
            before = self._load_snapshot(
                directory, raw.get("before"), maximum=MAX_STATE_BYTES
            )
            restore_snapshot(self.paths.rooted(str(absolute)), before)
        for absolute, raw in directories.items():
            if not isinstance(raw, Mapping) or not isinstance(
                raw.get("before"), Mapping
            ):
                raise HarnessError("transaction rollback directory is invalid")
            before = raw["before"]
            ensure_directory(
                self.paths.rooted(str(absolute)),
                mode=int(before["mode"]),
                uid=int(before["uid"]),
                gid=int(before["gid"]),
            )
        for absolute, raw in files.items():
            if not isinstance(raw, Mapping) or not isinstance(
                raw.get("before"), Mapping
            ):
                raise HarnessError("transaction rollback file is invalid")
            if not self._file_matches(self.paths.rooted(str(absolute)), raw["before"]):
                raise HarnessError("transaction rollback file digest did not converge")
        for absolute, raw in directories.items():
            if not isinstance(raw, Mapping) or not isinstance(
                raw.get("before"), Mapping
            ):
                raise HarnessError("transaction rollback directory is invalid")
            if not self._directory_matches(
                self.paths.rooted(str(absolute)), raw["before"]
            ):
                raise HarnessError("transaction rollback directory did not converge")
        units_before = transaction.get("unitsBefore")
        if not isinstance(units_before, Mapping):
            raise HarnessError("transaction unit snapshot is invalid")
        transaction.update(
            {
                "phase": "RESTORED",
                "liveVerified": False,
                "runtimeRestored": False,
                "restoredAt": self.clock(),
            }
        )
        state["trustTransaction"] = transaction
        self._save_state(run_id, state)
        if start_runtime:
            self._restore_units(units_before)
            transaction["runtimeRestored"] = True
            state["trustTransaction"] = transaction
            self._save_state(run_id, state)
        else:
            self._restore_unit_enablement_for_boot(units_before)

    def _production_restoration_attestation(
        self,
        run_id: str,
        state: Mapping[str, Any],
        transaction: Mapping[str, Any],
        *,
        boot_recovery: bool = False,
    ) -> dict[str, object]:
        """Attest the exact pre-run trust baseline without exposing payloads."""

        self._assert_active_run(run_id, allow_unclaimed=False)
        self._validate_transaction_shape(transaction)
        if transaction.get("phase") != "RESTORED" or state.get(
            "trustTransaction"
        ) != transaction:
            raise HarnessError("production trust transaction is not restored")
        if not boot_recovery and transaction.get("runtimeRestored") is False:
            raise HarnessError("production runtime baseline is not restored")
        files_raw = transaction.get("files")
        directories_raw = transaction.get("directories")
        units_raw = transaction.get("unitsBefore")
        if not all(
            isinstance(item, Mapping)
            for item in (files_raw, directories_raw, units_raw)
        ):
            raise HarnessError("production restoration metadata is invalid")
        files: dict[str, dict[str, object]] = {}
        for absolute in sorted(TRANSACTION_FILES):
            raw = files_raw.get(absolute)  # type: ignore[union-attr]
            before = raw.get("before") if isinstance(raw, Mapping) else None
            if not isinstance(before, Mapping):
                raise HarnessError("production restoration file baseline is invalid")
            current = snapshot(self.paths.rooted(absolute), maximum=MAX_STATE_BYTES)
            actual = {
                "exists": current.exists,
                "sha256": sha256_bytes(current.payload) if current.exists else None,
                "mode": current.mode,
                "uid": current.uid,
                "gid": current.gid,
            }
            expected = {
                "exists": before.get("exists"),
                "sha256": before.get("sha256"),
                "mode": before.get("mode"),
                "uid": before.get("uid"),
                "gid": before.get("gid"),
            }
            if actual != expected:
                raise HarnessError("production trust file baseline did not converge")
            files[absolute] = actual
        directories: dict[str, dict[str, int]] = {}
        for absolute in sorted(TRANSACTION_DIRECTORIES):
            raw = directories_raw.get(absolute)  # type: ignore[union-attr]
            before = raw.get("before") if isinstance(raw, Mapping) else None
            metadata = self.paths.rooted(absolute).lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(
                metadata.st_mode
            ):
                raise HarnessError("production trust directory endpoint is unsafe")
            actual_directory = {
                "mode": stat.S_IMODE(metadata.st_mode),
                "uid": metadata.st_uid,
                "gid": metadata.st_gid,
            }
            if not isinstance(before, Mapping) or actual_directory != {
                "mode": before.get("mode"),
                "uid": before.get("uid"),
                "gid": before.get("gid"),
            }:
                raise HarnessError("production trust directory baseline did not converge")
            directories[absolute] = actual_directory
        units: dict[str, dict[str, object]] = {}
        for unit in UNITS:
            expected_unit = units_raw.get(unit)  # type: ignore[union-attr]
            current_unit = self._unit_status(unit)
            comparable = {
                key: current_unit.get(key)
                for key in ("active", "enabled", "unitFileState")
            }
            expected_comparable = {
                key: expected_unit.get(key)
                for key in ("active", "enabled", "unitFileState")
            } if isinstance(expected_unit, Mapping) else None
            valid_boot_unit = bool(
                boot_recovery
                and isinstance(expected_unit, Mapping)
                and comparable["active"] is False
                and comparable["enabled"] is expected_comparable["enabled"]
                and comparable["unitFileState"] == expected_comparable["unitFileState"]
            )
            if not isinstance(expected_unit, Mapping) or (
                comparable != expected_comparable and not valid_boot_unit
            ):
                raise HarnessError("production unit baseline did not converge")
            units[unit] = comparable
        attestation: dict[str, object] = {
            "schemaVersion": 1,
            "transactionPhase": "RESTORED",
            "manifestSha256": transaction["manifestSha256"],
            "files": files,
            "directories": directories,
            "units": units,
        }
        if boot_recovery:
            attestation["bootRecovery"] = True
        attestation["sha256"] = sha256_bytes(
            (json.dumps(attestation, sort_keys=True, separators=(",", ":")) + "\n").encode()
        )
        return attestation

    def _cleanup_known_staging(
        self, run_id: str, manifest: Mapping[str, Any] | None
    ) -> None:
        expected: dict[str, str] = {}
        if isinstance(manifest, Mapping) and isinstance(
            manifest.get("inputs"), Mapping
        ):
            inputs = manifest["inputs"]
            expected = {
                "command": str(inputs.get("commandSha256") or ""),
                "release": str(inputs.get("releaseSha256") or ""),
                "health": str(inputs.get("healthSha256") or ""),
                "binding": str(inputs.get("bindingSha256") or ""),
            }
        for role, digest in expected.items():
            path = self._staging_path(run_id, role)
            if path.exists() and SHA256_RE.fullmatch(digest):
                actual, _ = sha256_regular(path, maximum=MAX_TRUST_BYTES)
                if actual == digest:
                    path.unlink()
                    fsync_directory(path.parent)

    def _cleanup_candidate_soak(
        self,
        run_id: str,
        state: dict[str, Any],
        *,
        from_deadman: bool = False,
        boot_recovery: bool = False,
    ) -> bool:
        soak = state.get("candidateSoak")
        if not isinstance(soak, dict):
            return False
        stop_writers = (
            self._stop_writers_for_boot if boot_recovery else self._stop_writers
        )
        phase = soak.get("phase")
        if phase not in CANDIDATE_PHASES:
            raise HarnessError("candidate soak cleanup journal is invalid")
        recovered = False
        if "candidateIncomingPath" in soak:
            incoming_recovered = self._recover_candidate_incoming(run_id, soak)
            recovered = incoming_recovered or recovered
            if incoming_recovered:
                soak["candidateIncomingRecovered"] = True
                state["candidateSoak"] = soak
        else:
            raw_slot_for_incoming = soak.get("candidateSlot")
            if isinstance(raw_slot_for_incoming, str):
                slot_for_incoming = Path(raw_slot_for_incoming)
                if slot_for_incoming.parent == self.paths.candidate_root:
                    unjournaled = self._candidate_incoming_path(slot_for_incoming)
                    if unjournaled.exists() or unjournaled.is_symlink():
                        raise HarnessError(
                            "candidate STAGING recovery journal is incomplete"
                        )
        if CANDIDATE_PHASES.index(str(phase)) >= CANDIDATE_PHASES.index(
            "QUIESCING"
        ) and phase not in {"RESTORED", "COMPLETE"}:
            slots = soak.get("baselineSlots")
            units = soak.get("unitsBefore")
            release_trees_before = soak.get("releaseTreesBefore")
            if (
                not isinstance(slots, Mapping)
                or not isinstance(units, Mapping)
                or not isinstance(release_trees_before, Mapping)
            ):
                stop_writers()
                raise HarnessError("candidate soak recovery metadata is incomplete")
            candidate_execution_started = any(
                key in soak for key in ("runningAt", "capturedAt", "rawEvidenceSha256")
            )
            try:
                stop_writers()
                if candidate_execution_started:
                    execution_unit = soak.get("executionUnit")
                    expected_unit = self._candidate_unit(run_id)
                    if execution_unit is None:
                        execution_unit = expected_unit
                    if (
                        not isinstance(execution_unit, str)
                        or execution_unit != expected_unit
                    ):
                        raise HarnessError("candidate execution unit is invalid")
                    execution_proof = soak.get("executionProof")
                    expected_group = (
                        execution_proof.get("controlGroup")
                        if isinstance(execution_proof, Mapping)
                        and isinstance(execution_proof.get("controlGroup"), str)
                        else "/system.slice/" + execution_unit
                    )
                    termination = self._terminate_candidate_unit(
                        execution_unit,
                        expected_control_group=expected_group,
                    )
                    soak["executionUnit"] = execution_unit
                    soak["terminationProof"] = termination
                    soak["executionCgroupEmpty"] = (
                        termination.get("recursivePopulated") is False
                    )
                    state["candidateSoak"] = soak
                    self._save_state(run_id, state)
                if self._slot_snapshot() != dict(slots):
                    raise HarnessError(
                        "candidate soak recovery found changed signed slots"
                    )
                release_trees_after = self._release_tree_snapshot(slots)
                if release_trees_after != release_trees_before:
                    raise HarnessError(
                        "candidate soak recovery found changed signed release trees"
                    )
                soak["releaseTreesAfter"] = release_trees_after
                anti_replay = soak.get("antiReplay")
                rollback_terminal = soak.get("rollbackTerminal")
                if candidate_execution_started and not isinstance(
                    anti_replay, Mapping
                ):
                    raise HarnessError(
                        "candidate soak recovery anti-replay proof is missing"
                    )
                if candidate_execution_started and not isinstance(
                    rollback_terminal, Mapping
                ):
                    raise HarnessError(
                        "candidate soak recovery rollback boundary is missing"
                    )
                if isinstance(anti_replay, Mapping) and isinstance(
                    rollback_terminal, Mapping
                ):
                    self._validate_candidate_anti_replay(
                        anti_replay, rollback_terminal
                    )
                if isinstance(anti_replay, Mapping) and (
                    self._anti_replay_snapshot() != dict(anti_replay)
                ):
                    raise HarnessError(
                        "candidate soak recovery found changed anti-replay state"
                    )
                oak_before = soak.get("oakBefore")
                oak_current = self.verify_oak()
                if (
                    not isinstance(oak_before, Mapping)
                    or oak_current.get("port") != oak_before.get("port")
                    or oak_current.get("mxidSha256")
                    != oak_before.get("mxidSha256")
                ):
                    raise HarnessError("candidate soak recovery found changed OAK")
                if candidate_execution_started:
                    persistent_before = soak.get("persistentStateBefore")
                    if not isinstance(persistent_before, Mapping):
                        raise HarnessError(
                            "candidate soak recovery persistent baseline is missing"
                        )
                    persistent_after = self._candidate_persistent_state_snapshot()
                    if persistent_after != persistent_before:
                        raise HarnessError(
                            "candidate soak recovery found changed persistent state"
                        )
                    soak["persistentStateAfter"] = persistent_after
                    termination = soak.get("terminationProof")
                    expected_unit = self._candidate_unit(run_id)
                    if (
                        not isinstance(termination, Mapping)
                        or termination.get("unit") != expected_unit
                        or termination.get("controlGroup")
                        != "/system.slice/" + expected_unit
                        or termination.get("recursivePopulated") is not False
                        or termination.get("stopSucceeded") is not True
                        or termination.get("loadState") != "not-found"
                        or termination.get("activeState") != "inactive"
                        or termination.get("cgroupRemoved") is not True
                    ):
                        raise HarnessError(
                            "candidate recovery cgroup termination proof is invalid"
                        )
                if boot_recovery:
                    self._restore_unit_enablement_for_boot(units)
                    restored = {
                        "bootRecovery": True,
                        "activeSlot": str(slots.get("current") or ""),
                    }
                else:
                    self._restore_units(units)
                    restored = self._agent_process_identity(
                        str(slots.get("current") or "")
                    )
                baseline = soak.get("baselineRuntime")
                oak_after = self.verify_oak()
                updater_after = None if boot_recovery else self._probe_updater()
                update_after = (
                    updater_after.get("update")
                    if isinstance(updater_after, Mapping)
                    else None
                )
                if (
                    not isinstance(baseline, Mapping)
                    or (not boot_recovery and (
                        restored.get("bootId"),
                        restored.get("pid"),
                        restored.get("startTicks"),
                    )
                    == (
                        baseline.get("bootId"),
                        baseline.get("pid"),
                        baseline.get("startTicks"),
                    ))
                    or self._slot_snapshot() != dict(slots)
                    or (
                        isinstance(anti_replay, Mapping)
                        and self._anti_replay_snapshot() != dict(anti_replay)
                    )
                    or oak_after.get("port") != oak_before.get("port")
                    or oak_after.get("mxidSha256") != oak_before.get("mxidSha256")
                    or (
                        not boot_recovery
                        and (
                            not isinstance(update_after, Mapping)
                            or update_after.get("phase") != "ROLLED_BACK"
                            or "healthDeadline" in update_after
                        )
                    )
                ):
                    raise HarnessError(
                        "candidate soak recovery lacks fresh baseline proof"
                    )
            except BaseException as exc:
                try:
                    stop_writers()
                except BaseException:
                    pass
                raise HarnessError(
                    "candidate soak recovery failed; runtime remains fail-closed"
                ) from exc
            soak.update(
                {
                    "phase": "RESTORED",
                    "restoredAt": self.clock(),
                    "restoredRuntime": restored,
                    "cleanupRecovered": True,
                    "passed": False,
                    "failureCode": "INTERRUPTED_CANDIDATE_SOAK",
                }
            )
            state["candidateSoak"] = soak
            self._save_state(run_id, state)
            recovered = True
        deadman = soak.get("deadman")
        if (
            CANDIDATE_PHASES.index(str(soak.get("phase")))
            <= CANDIDATE_PHASES.index("STAGED")
            and isinstance(deadman, Mapping)
            and deadman.get("armed") is True
        ):
            soak.update(
                {
                    "phase": "RESTORED",
                    "restoredAt": self.clock(),
                    "cleanupRecovered": True,
                    "passed": False,
                    "failureCode": "INTERRUPTED_CANDIDATE_SOAK",
                }
            )
            state["candidateSoak"] = soak
            self._save_state(run_id, state)
            recovered = True
        raw_slot = soak.get("candidateSlot")
        if isinstance(raw_slot, str):
            slot = Path(raw_slot)
            if slot.exists() or slot.is_symlink():
                self._remove_candidate_slot(slot)
        digests = soak.get("inputDigests")
        if isinstance(digests, Mapping):
            for role, key, maximum in (
                ("candidate-bundle", "bundleSha256", MAX_CANDIDATE_BUNDLE_BYTES),
                ("candidate-bom", "bomSha256", MAX_TRUST_BYTES),
                ("oak-harness", "harnessSha256", 2 * 1024 * 1024),
            ):
                digest = digests.get(key)
                path = self._candidate_staging_path(run_id, role)
                if (
                    isinstance(digest, str)
                    and SHA256_RE.fullmatch(digest)
                    and path.exists()
                ):
                    actual, _ = sha256_regular(path, maximum=maximum)
                    if actual != digest:
                        raise HarnessError("candidate cleanup staging digest mismatch")
                    path.unlink()
                    fsync_directory(path.parent)
        if self._remove_pinned_candidate_harness(run_id, soak):
            recovered = True
            soak["harnessExecutionRemoved"] = True
            state["candidateSoak"] = soak
        return recovered

    def _candidate_deadman_cleanup(
        self,
        run_id: str,
        *,
        invocation_epoch: str | None = None,
        invocation_controller: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        canonical_run_id(run_id)
        self._assert_active_run(run_id, allow_unclaimed=False)
        state = self._read_existing_run_state(run_id)
        soak = state.get("candidateSoak")
        deadman = soak.get("deadman") if isinstance(soak, Mapping) else None
        supplied = invocation_epoch is not None or invocation_controller is not None
        if supplied:
            if (
                not isinstance(invocation_epoch, str)
                or RUN_ID_RE.fullmatch(invocation_epoch) is None
                or invocation_controller is None
            ):
                raise HarnessError("candidate deadman invocation identity is invalid")
            supplied_controller = self._validate_controller_identity(
                invocation_controller
            )
        else:
            supplied_controller = None
        if isinstance(deadman, Mapping) and deadman.get("armed") is not True:
            transaction = state.get("trustTransaction")
            cleanup = state.get("cleanup")
            if (
                deadman.get("recoveredBy") == "deadman"
                and isinstance(transaction, Mapping)
                and transaction.get("phase") == "RESTORED"
                and isinstance(soak, Mapping)
                and isinstance(soak.get("productionRestoration"), Mapping)
                and isinstance(cleanup, Mapping)
                and cleanup.get("complete") is True
                and (self.paths.active_run.exists() or self.paths.active_run.is_symlink())
            ):
                self._release_active_run(run_id)
            return {
                "schemaVersion": 1,
                "runId": run_id,
                "candidateDeadmanOnly": True,
                "recovered": False,
                "complete": True,
            }
        if isinstance(deadman, Mapping):
            if deadman.get("lifecycle") not in {"ARMING", "ARMED"}:
                raise HarnessError("candidate deadman lifecycle is invalid")
            epoch = deadman.get("writerEpoch")
            controller = self._validate_controller_identity(
                deadman.get("controller")
            )
            if supplied and (
                invocation_epoch != epoch or supplied_controller != controller
            ):
                raise HarnessError("candidate deadman invocation/journal mismatch")
        else:
            if not supplied:
                raise HarnessError("candidate deadman journal is missing")
            epoch = invocation_epoch
            controller = supplied_controller
        if not isinstance(epoch, str) or RUN_ID_RE.fullmatch(epoch) is None:
            raise HarnessError("candidate deadman writer epoch is invalid")
        if controller is None:
            raise HarnessError("candidate deadman controller identity is missing")
        # At the bounded deadline an alive/hung controller is no longer a
        # legitimate writer. Exact boot-id/PID/start-ticks matching prevents
        # PID reuse from fencing an unrelated process.
        self._fence_candidate_controller(controller)
        with self._candidate_writer_lock(run_id, timeout=30):
            latest = self._load_state(run_id)
            latest_soak = latest.get("candidateSoak")
            latest_deadman = (
                latest_soak.get("deadman")
                if isinstance(latest_soak, Mapping)
                else None
            )
            if not isinstance(latest_deadman, Mapping):
                if isinstance(latest_soak, dict):
                    latest_soak["deadman"] = {
                        "unit": self._candidate_deadman_unit(run_id),
                        "armed": True,
                        "lifecycle": "ARMED",
                        "writerEpoch": epoch,
                        "controller": controller,
                        "recoveredOrphanTimer": True,
                    }
                    latest["candidateSoak"] = latest_soak
                    self._save_state(run_id, latest)
                else:
                    return self._candidate_orphan_deadman_cleanup_fenced(
                        run_id, epoch=epoch
                    )
            elif latest_deadman.get("writerEpoch") != epoch:
                raise HarnessError("candidate deadman writer epoch changed at fence")
            return self._candidate_deadman_cleanup_fenced(run_id, epoch=epoch)

    def _candidate_orphan_deadman_cleanup_fenced(
        self, run_id: str, *, epoch: str
    ) -> dict[str, object]:
        state = self._load_state(run_id)
        transaction = state.get("trustTransaction")
        if not isinstance(transaction, dict):
            raise HarnessError("orphan candidate deadman transaction is missing")
        self._restore_transaction(run_id, state, transaction)
        state = self._load_state(run_id)
        transaction = state.get("trustTransaction")
        if not isinstance(transaction, dict):
            raise HarnessError("orphan candidate restoration journal is invalid")
        attestation = self._production_restoration_attestation(
            run_id, state, transaction
        )
        state["candidateRecovery"] = {
            "schemaVersion": 1,
            "writerEpoch": epoch,
            "productionRestoration": attestation,
            "recoveredAt": self.clock(),
            "reason": "ORPHAN_DEADMAN_TIMER",
        }
        state["cleanup"] = {
            "complete": True,
            "completedAt": self.clock(),
            "recoveredBy": "candidate-deadman",
        }
        self._purge_run_sensitive_material(run_id)
        self._save_state(run_id, state)
        self._require_complete_cleanup_attestation(
            self._cleanup_attestation(run_id, state),
            require_transaction=True,
            require_lease_absent=False,
        )
        if self.paths.active_run.exists() or self.paths.active_run.is_symlink():
            self._release_active_run(run_id)
        self._require_complete_cleanup_attestation(
            self._cleanup_attestation(run_id, state), require_transaction=True
        )
        if isinstance(state.get("transactionGuard"), Mapping):
            self._disarm_transaction_guard(run_id, state)
        return {
            "schemaVersion": 1,
            "runId": run_id,
            "candidateDeadmanOnly": True,
            "recovered": True,
            "complete": True,
        }

    def _candidate_deadman_cleanup_fenced(
        self, run_id: str, *, epoch: str
    ) -> dict[str, object]:
        # Deliberately avoid the operation/USB locks: those descriptors vanish
        # on controller death, while the writer fence above serializes recovery.
        self._assert_active_run(run_id, allow_unclaimed=False)
        state = self._load_state(run_id)
        soak = state.get("candidateSoak")
        if not isinstance(soak, dict):
            raise HarnessError("candidate deadman journal is missing")
        deadman = soak.get("deadman")
        expected_unit = self._candidate_deadman_unit(run_id)
        if (
            not isinstance(deadman, dict)
            or deadman.get("unit") != expected_unit
            or deadman.get("armed") is not True
            or deadman.get("lifecycle") not in {"ARMING", "ARMED"}
            or deadman.get("writerEpoch") != epoch
        ):
            raise HarnessError("candidate deadman ownership mismatch")
        recovered = self._cleanup_candidate_soak(
            run_id, state, from_deadman=True
        )
        latest = self._load_state(run_id)
        latest_soak = latest.get("candidateSoak")
        latest_transaction = latest.get("trustTransaction")
        if not isinstance(latest_transaction, dict):
            raise HarnessError("candidate deadman trust transaction is missing")
        self._restore_transaction(run_id, latest, latest_transaction)
        latest = self._load_state(run_id)
        latest_soak = latest.get("candidateSoak")
        latest_transaction = latest.get("trustTransaction")
        if not isinstance(latest_soak, dict) or not isinstance(
            latest_transaction, dict
        ):
            raise HarnessError("candidate deadman restoration journal is invalid")
        latest_soak["productionRestoration"] = (
            self._production_restoration_attestation(
                run_id, latest, latest_transaction
            )
        )
        latest["candidateSoak"] = latest_soak
        self._save_state(run_id, latest)
        latest_deadman = (
            latest_soak.get("deadman") if isinstance(latest_soak, dict) else None
        )
        if (
            not isinstance(latest_soak, dict)
            or latest_soak.get("phase") not in {"RESTORED", "COMPLETE"}
            or latest_transaction.get("phase") != "RESTORED"
            or not isinstance(latest_soak.get("productionRestoration"), Mapping)
            or not isinstance(latest_deadman, dict)
            or latest_deadman.get("unit") != expected_unit
            or latest_deadman.get("armed") is not True
        ):
            raise HarnessError("candidate deadman recovery did not converge")
        # Remove secret-bearing rollback payloads while recovery ownership is
        # still armed. A failure is retried by systemd and cannot strand an
        # apparently disarmed lease.
        self._purge_run_sensitive_material(run_id)
        latest["cleanup"] = {
            "complete": True,
            "completedAt": self.clock(),
            "recoveredBy": "candidate-deadman",
        }
        self._save_state(run_id, latest)
        self._require_complete_cleanup_attestation(
            self._cleanup_attestation(run_id, latest),
            require_transaction=True,
            require_lease_absent=False,
        )
        if self.paths.active_run.exists() or self.paths.active_run.is_symlink():
            self._release_active_run(run_id)
        self._require_complete_cleanup_attestation(
            self._cleanup_attestation(run_id, latest), require_transaction=True
        )
        latest_deadman.update(
            {
                "armed": False,
                "lifecycle": "DISARMED",
                "recoveryUnitExitPending": True,
                "recoveredAt": self.clock(),
                "recoveredBy": "deadman",
            }
        )
        latest_soak["deadman"] = latest_deadman
        latest["candidateSoak"] = latest_soak
        self._save_state(run_id, latest)
        if isinstance(latest.get("transactionGuard"), Mapping):
            self._disarm_transaction_guard(run_id, latest)
        return {
            "schemaVersion": 1,
            "runId": run_id,
            "candidateDeadmanOnly": True,
            "recovered": recovered,
            "complete": True,
        }

    def _deadman_cleanup(self, run_id: str) -> dict[str, object]:
        # The emergency recovery path intentionally does not wait on the
        # operation/USB flock held by a wedged fault injector. It is root-only,
        # bound to the persistent active-run lease and performs only the
        # idempotent exact-port rebind plus journal convergence.
        canonical_run_id(run_id)
        self._assert_active_run(run_id, allow_unclaimed=False)
        state = self._load_state(run_id)
        fault = state.get("oakFault")
        if not isinstance(fault, dict):
            return {
                "schemaVersion": 1,
                "runId": run_id,
                "deadmanOnly": True,
                "recovered": False,
                "complete": True,
            }
        port = canonical_oak_port(fault.get("port"))
        if fault.get("unit") != self._deadman_unit(run_id):
            raise HarnessError("OAK deadman ownership mismatch")
        if fault.get("armed") is not True:
            self.verify_oak(require_bound=True, expected_port=port)
            return {
                "schemaVersion": 1,
                "runId": run_id,
                "deadmanOnly": True,
                "recovered": False,
                "complete": True,
            }
        self._recover_oak(port)
        latest = self._load_state(run_id)
        latest_fault = latest.get("oakFault")
        if (
            not isinstance(latest_fault, dict)
            or latest_fault.get("port") != port
        ):
            raise HarnessError("OAK fault journal changed during deadman recovery")
        latest_fault.update(
            {
                "armed": False,
                "recovered": True,
                "recoveredAt": self.clock(),
                "recoveredBy": "deadman",
            }
        )
        latest["oakFault"] = latest_fault
        self._save_state(run_id, latest)
        return {
            "schemaVersion": 1,
            "runId": run_id,
            "deadmanOnly": True,
            "recovered": True,
            "complete": True,
        }

    def _purge_run_sensitive_material(self, run_id: str) -> None:
        transaction_dir = self._transaction_dir(run_id)
        if transaction_dir.exists() or transaction_dir.is_symlink():
            metadata = transaction_dir.lstat()
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != self.root_uid
                or stat.S_IMODE(metadata.st_mode) != 0o700
            ):
                raise HarnessError("transaction scrub target is unsafe")
            shutil.rmtree(transaction_dir)
            fsync_directory(transaction_dir.parent)
        archive = self.paths.recovery_root / f"iq9075-{run_id}.tar"
        if archive.exists() or archive.is_symlink():
            metadata = archive.lstat()
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != self.root_uid
                or stat.S_IMODE(metadata.st_mode) != 0o600
            ):
                raise HarnessError("recovery archive scrub target is unsafe")
            archive.unlink()
            fsync_directory(archive.parent)

    @staticmethod
    def _path_absent(path: Path) -> bool:
        return not path.exists() and not path.is_symlink()

    def _cleanup_attestation(
        self, run_id: str, state: Mapping[str, Any]
    ) -> dict[str, object]:
        """Attest only post-cleanup state; never retain secret payload bytes."""

        transaction = state.get("trustTransaction")
        transaction_phase = (
            transaction.get("phase") if isinstance(transaction, Mapping) else None
        )
        soak = state.get("candidateSoak")
        candidate_paths: list[Path] = []
        if isinstance(soak, Mapping):
            raw_slot = soak.get("candidateSlot")
            raw_incoming = soak.get("candidateIncomingPath")
            harness = soak.get("harnessExecution")
            if raw_slot is not None:
                if not isinstance(raw_slot, str):
                    raise HarnessError("cleanup candidate slot journal is invalid")
                slot = Path(raw_slot)
                if slot.parent != self.paths.candidate_root:
                    raise HarnessError("cleanup candidate slot escaped fixed root")
                candidate_paths.append(slot)
                if raw_incoming is not None:
                    if Path(str(raw_incoming)) != self._candidate_incoming_path(slot):
                        raise HarnessError("cleanup candidate incoming path mismatch")
                    candidate_paths.append(Path(str(raw_incoming)))
            elif raw_incoming is not None:
                raise HarnessError("cleanup candidate incoming journal is unbound")
            if harness is not None:
                if not isinstance(harness, Mapping) or not isinstance(
                    harness.get("sha256"), str
                ):
                    raise HarnessError("cleanup candidate harness journal is invalid")
                harness_path = self._candidate_harness_execution_path(
                    run_id, str(harness["sha256"])
                )
                if harness.get("path") != str(harness_path):
                    raise HarnessError("cleanup candidate harness path mismatch")
                candidate_paths.append(harness_path)
        candidate_staging = [
            self._candidate_staging_path(run_id, role)
            for role in CANDIDATE_INPUT_ROLES
        ]
        trust_staging = [
            self._staging_path(run_id, role) for role in INPUT_ROLES
        ]
        cleanup = state.get("cleanup")
        candidate_named_absent = True
        if self.paths.candidate_root.is_dir():
            candidate_named_absent = not any(
                entry.name.startswith(run_id + "-")
                or entry.name.startswith("." + run_id + "-")
                for entry in self.paths.candidate_root.iterdir()
            )
        run_dir = self.paths.state_root / run_id
        if run_dir.is_dir() and any(
            entry.name.startswith("candidate-oak-harness-")
            for entry in run_dir.iterdir()
        ):
            candidate_named_absent = False
        return {
            "schemaVersion": 1,
            "transactionPhase": transaction_phase,
            "cleanupJournalComplete": bool(
                isinstance(cleanup, Mapping) and cleanup.get("complete") is True
            ),
            "activeRunLeaseAbsent": self._path_absent(self.paths.active_run),
            "transactionSnapshotsAbsent": self._path_absent(
                self._transaction_dir(run_id)
            ),
            "recoveryArchiveAbsent": self._path_absent(
                self.paths.recovery_root / f"iq9075-{run_id}.tar"
            ),
            "candidateArtifactsAbsent": all(
                self._path_absent(path) for path in candidate_paths
            )
            and candidate_named_absent,
            "candidateStagingAbsent": all(
                self._path_absent(path) for path in candidate_staging
            ),
            "trustStagingAbsent": all(
                self._path_absent(path) for path in trust_staging
            ),
        }

    @staticmethod
    def _require_complete_cleanup_attestation(
        proof: Mapping[str, object], *, require_transaction: bool,
        require_lease_absent: bool = True,
    ) -> None:
        accepted_phases = {"RESTORED"} if require_transaction else {None, "RESTORED"}
        if proof.get("transactionPhase") not in accepted_phases:
            raise HarnessError("cleanup trust transaction is not restored")
        required = (
            "cleanupJournalComplete",
            "transactionSnapshotsAbsent",
            "recoveryArchiveAbsent",
            "candidateArtifactsAbsent",
            "candidateStagingAbsent",
            "trustStagingAbsent",
        ) + (("activeRunLeaseAbsent",) if require_lease_absent else ())
        if any(
            proof.get(key) is not True
            for key in required
        ):
            raise HarnessError("cleanup attestation did not converge")

    def _read_existing_run_state(self, run_id: str) -> dict[str, Any]:
        canonical_run_id(run_id)
        path = self._state_path(run_id)
        payload, metadata = read_regular(path, maximum=MAX_STATE_BYTES)
        if metadata.st_uid != self.root_uid or stat.S_IMODE(metadata.st_mode) != 0o600:
            raise HarnessError("run journal ownership or mode is unsafe")
        state = strict_json(payload, label="run journal")
        if (
            type(state.get("schemaVersion")) is not int
            or state.get("schemaVersion") != 2
            or state.get("protocolVersion") != PROTOCOL_VERSION
            or state.get("runId") != run_id
        ):
            raise HarnessError("run journal identity mismatch")
        return state

    def _existing_active_run_id(self) -> str | None:
        path = self.paths.active_run
        if not path.exists() and not path.is_symlink():
            return None
        payload, metadata = read_regular(path, maximum=4096)
        if metadata.st_uid != self.root_uid or stat.S_IMODE(metadata.st_mode) != 0o600:
            raise HarnessError("Fleet E2E board lease ownership or mode is unsafe")
        lease = strict_json(payload, label="Fleet E2E board lease")
        if (
            set(lease) != {"schemaVersion", "protocolVersion", "runId"}
            or type(lease.get("schemaVersion")) is not int
            or lease.get("schemaVersion") != 1
            or lease.get("protocolVersion") != PROTOCOL_VERSION
            or not isinstance(lease.get("runId"), str)
        ):
            raise HarnessError("Fleet E2E board lease is corrupt")
        return canonical_run_id(lease["runId"])

    def _existing_config_stream_run_id(self) -> str | None:
        path = self.paths.config_stream_active
        if not path.exists() and not path.is_symlink():
            return None
        payload, metadata = read_regular(path, maximum=4096)
        if metadata.st_uid != self.root_uid or stat.S_IMODE(metadata.st_mode) != 0o600:
            raise HarnessError(
                "config-stream lease ownership or mode is unsafe"
            )
        lease = strict_json(payload, label="config-stream board lease")
        if (
            set(lease) != {"schemaVersion", "runId"}
            or type(lease.get("schemaVersion")) is not int
            or lease.get("schemaVersion") != 1
            or not isinstance(lease.get("runId"), str)
        ):
            raise HarnessError("config-stream board lease is corrupt")
        return canonical_run_id(lease["runId"])

    def _package_maintenance_is_active(self) -> bool:
        path = self.paths.package_maintenance
        if not path.exists() and not path.is_symlink():
            return False
        payload, metadata = read_regular(path, maximum=4096)
        value = strict_json(payload, label="package maintenance marker")
        if (
            metadata.st_uid != self.root_uid
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or value
            != {
                "schemaVersion": 1,
                "kind": "nuvion-package-maintenance",
                "active": True,
            }
        ):
            raise HarnessError("package maintenance marker is unsafe")
        return True

    def _scan_existing_runs(
        self, *, allow_missing_run: str | None = None
    ) -> dict[str, dict[str, Any]]:
        root = self.paths.state_root
        if not root.exists() and not root.is_symlink():
            return {}
        metadata = root.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != self.root_uid
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise HarnessError("Fleet E2E run root is unsafe")
        states: dict[str, dict[str, Any]] = {}
        for entry in root.iterdir():
            entry_metadata = entry.lstat()
            if (
                stat.S_ISLNK(entry_metadata.st_mode)
                or not stat.S_ISDIR(entry_metadata.st_mode)
                or entry_metadata.st_uid != self.root_uid
                or stat.S_IMODE(entry_metadata.st_mode) != 0o700
            ):
                raise HarnessError("Fleet E2E run directory is unsafe")
            run_id = canonical_run_id(entry.name)
            if (
                allow_missing_run == run_id
                and not self._state_path(run_id).exists()
                and not self._state_path(run_id).is_symlink()
            ):
                continue
            states[run_id] = self._read_existing_run_state(run_id)
        return states

    @staticmethod
    def _state_needs_boot_recovery(state: Mapping[str, Any]) -> bool:
        transaction = state.get("trustTransaction")
        if transaction is not None:
            if not isinstance(transaction, Mapping):
                raise HarnessError("trust transaction journal is corrupt")
            if transaction.get("phase") != "RESTORED":
                return True
        guard = state.get("transactionGuard")
        if guard is not None:
            if not isinstance(guard, Mapping):
                raise HarnessError("transaction recovery guard journal is corrupt")
            if guard.get("armed") is True:
                return True
        backup = state.get("backup")
        if backup is not None and (
            not isinstance(backup, Mapping) or backup.get("complete") is not True
        ):
            return True
        recovery_state_present = any(
            key in state
            for key in (
                "backup",
                "trustTransaction",
                "transactionGuard",
                "oakFault",
                "candidateSoak",
            )
        )
        cleanup = state.get("cleanup")
        if recovery_state_present and (
            not isinstance(cleanup, Mapping)
            or cleanup.get("complete") is not True
        ):
            return True
        oak = state.get("oakFault")
        if oak is not None:
            if not isinstance(oak, Mapping):
                raise HarnessError("OAK recovery journal is corrupt")
            if oak.get("armed") is True:
                return True
        soak = state.get("candidateSoak")
        if soak is not None:
            if not isinstance(soak, Mapping):
                raise HarnessError("candidate recovery journal is corrupt")
            deadman = soak.get("deadman")
            if deadman is not None and not isinstance(deadman, Mapping):
                raise HarnessError("candidate deadman journal is corrupt")
            if isinstance(deadman, Mapping) and deadman.get("armed") is True:
                return True
        return False

    def _validate_existing_recovery_state(
        self, run_id: str, state: Mapping[str, Any]
    ) -> None:
        transaction = state.get("trustTransaction")
        if transaction is not None:
            if not isinstance(transaction, Mapping):
                raise HarnessError("trust transaction journal is corrupt")
            self._validate_transaction_shape(transaction)
        guard = state.get("transactionGuard")
        if guard is not None:
            self._validate_transaction_guard(run_id, guard)
        cleanup = state.get("cleanup")
        if cleanup is not None and (
            not isinstance(cleanup, Mapping)
            or not isinstance(cleanup.get("complete"), bool)
        ):
            raise HarnessError("cleanup journal is corrupt")
        if isinstance(cleanup, Mapping) and cleanup.get("complete") is True:
            self._require_complete_cleanup_attestation(
                self._cleanup_attestation(run_id, state),
                require_transaction=isinstance(transaction, Mapping),
                require_lease_absent=not (
                    self.paths.active_run.exists()
                    or self.paths.active_run.is_symlink()
                ),
            )

    def boot_reconcile(
        self, *, package_maintenance: bool = False
    ) -> dict[str, object]:
        """Offline, fail-closed recovery gate for protected runtime units."""

        # This must precede every state read: corrupt state is a reason to keep
        # all trust-consuming endpoints down, never a reason to leave them up.
        self._stop_writers_for_boot()
        maintenance_active = self._package_maintenance_is_active()
        if maintenance_active:
            if not package_maintenance:
                raise HarnessError("incomplete package maintenance blocks runtime")
            self._package_maintenance_authorized = True
        active_run = self._existing_active_run_id()
        config_stream_run = self._existing_config_stream_run_id()
        if config_stream_run is not None:
            if active_run != config_stream_run:
                raise HarnessError(
                    "config-stream recovery lease is detached from its Fleet run"
                )
            raise HarnessError(
                "active config-stream transaction requires explicit recovery"
            )
        states = self._scan_existing_runs()
        for run_id, state in states.items():
            self._validate_existing_recovery_state(run_id, state)
        unfinished = {
            run_id
            for run_id, state in states.items()
            if self._state_needs_boot_recovery(state)
        }
        if active_run is None:
            if unfinished:
                if len(unfinished) != 1:
                    raise HarnessError(
                        "unfinished Fleet transaction has no active-run lease"
                    )
                orphan_id = next(iter(unfinished))
                orphan = states[orphan_id]
                cleanup = orphan.get("cleanup")
                transaction = orphan.get("trustTransaction")
                guard = orphan.get("transactionGuard")
                if (
                    not isinstance(cleanup, Mapping)
                    or cleanup.get("complete") is not True
                    or (
                        transaction is not None
                        and (
                            not isinstance(transaction, Mapping)
                            or transaction.get("phase") != "RESTORED"
                        )
                    )
                    or not isinstance(guard, Mapping)
                    or guard.get("armed") is not True
                ):
                    raise HarnessError(
                        "unfinished Fleet transaction has no active-run lease"
                    )
                self._require_complete_cleanup_attestation(
                    self._cleanup_attestation(orphan_id, orphan),
                    require_transaction=isinstance(transaction, Mapping),
                )
                candidate_soak = orphan.get("candidateSoak")
                candidate_deadman = (
                    candidate_soak.get("deadman")
                    if isinstance(candidate_soak, dict)
                    else None
                )
                if (
                    isinstance(candidate_soak, dict)
                    and isinstance(candidate_deadman, Mapping)
                    and (
                        candidate_deadman.get("armed") is True
                        or candidate_deadman.get("stopped") is not True
                    )
                ):
                    self._disarm_candidate_deadman(
                        orphan_id, orphan, candidate_soak
                    )
                    orphan = self._read_existing_run_state(orphan_id)
                self._disarm_transaction_guard(orphan_id, orphan)
                return {
                    "schemaVersion": 1,
                    "kind": "nuvion-iq9075-boot-reconciliation",
                    "runId": orphan_id,
                    "recovered": True,
                    "complete": True,
                }
            return {
                "schemaVersion": 1,
                "kind": "nuvion-iq9075-boot-reconciliation",
                "recovered": False,
                "complete": True,
            }
        if active_run not in states:
            raise HarnessError("active-run lease journal is missing")
        if unfinished - {active_run}:
            raise HarnessError("multiple unfinished Fleet transactions are unsafe")
        result = self.cleanup(active_run, _boot_recovery=True)
        if result.get("complete") is not True:
            raise HarnessError("boot Fleet recovery did not complete")
        if self._existing_active_run_id() is not None:
            raise HarnessError("boot Fleet recovery did not release the lease")
        final = self._read_existing_run_state(active_run)
        transaction = final.get("trustTransaction")
        if isinstance(transaction, Mapping) and transaction.get("phase") != "RESTORED":
            raise HarnessError("boot Fleet trust restoration did not converge")
        if any(self._unit_status(unit)["active"] is True for unit in UNITS):
            raise HarnessError("boot recovery started a protected runtime unit")
        return {
            "schemaVersion": 1,
            "kind": "nuvion-iq9075-boot-reconciliation",
            "runId": active_run,
            "recovered": True,
            "complete": True,
        }

    def resume_boot_gate(self) -> dict[str, object]:
        """Finish offline parent recovery and clear a failed systemd boot gate."""

        result = self.boot_reconcile()
        if result.get("complete") is not True:
            raise HarnessError("Fleet boot reconciliation did not complete")
        for action in ("reset-failed", "start"):
            completed = self.runner.run(
                ["/usr/bin/systemctl", action, BOOT_RECONCILE_UNIT],
                timeout=180,
            )
            if completed.returncode != 0:
                raise HarnessError("Fleet boot reconciliation gate did not resume")
        active = self.runner.run(
            ["/usr/bin/systemctl", "is-active", "--quiet", BOOT_RECONCILE_UNIT],
            timeout=30,
        )
        if active.returncode != 0:
            raise HarnessError("Fleet boot reconciliation gate is not active")
        if self._existing_active_run_id() is not None:
            raise HarnessError("Fleet boot reconciliation retained the board lease")
        if self._existing_config_stream_run_id() is not None:
            raise HarnessError("Fleet boot reconciliation retained config-stream state")
        return {
            "schemaVersion": 1,
            "kind": "nuvion-iq9075-boot-gate-resumption",
            "complete": True,
            "gateActive": True,
            "protectedUnitsStopped": all(
                self._unit_status(unit)["active"] is False for unit in UNITS
            ),
        }

    def _transaction_deadman_cleanup(
        self,
        run_id: str,
        *,
        invocation_epoch: str,
        invocation_controller: Mapping[str, object],
    ) -> dict[str, object]:
        canonical_run_id(run_id)
        if (
            not isinstance(invocation_epoch, str)
            or RUN_ID_RE.fullmatch(invocation_epoch) is None
        ):
            raise HarnessError("transaction guard invocation epoch is invalid")
        supplied_controller = self._validate_controller_identity(
            invocation_controller
        )
        try:
            active_run = self._existing_active_run_id()
        except BaseException:
            # A corrupt lease may name the currently trust-consuming run.  No
            # endpoint may remain live while recovery ownership is ambiguous.
            try:
                self._stop_writers()
            except BaseException:
                pass
            raise
        if active_run is not None and active_run != run_id:
            return {
                "schemaVersion": 1,
                "runId": run_id,
                "transactionDeadmanOnly": True,
                "recovered": False,
                "complete": True,
                "stale": True,
            }
        lease_present = active_run == run_id
        if lease_present:
            self._stop_writers()
        try:
            state = self._read_existing_run_state(run_id)
        except BaseException:
            if active_run is None:
                try:
                    self._stop_writers()
                except BaseException:
                    pass
            raise
        raw_guard = state.get("transactionGuard")
        if raw_guard is None:
            if state.get("foundation", {}).get("verified") is not True:
                raise HarnessError("orphan transaction guard has no verified run")
            guard = {
                "unit": self._transaction_deadman_unit(run_id),
                "armed": True,
                "lifecycle": "ARMED",
                "writerEpoch": invocation_epoch,
                "controller": supplied_controller,
                "deadlineSeconds": TRANSACTION_DEADMAN_SECONDS,
                "deadlineAt": self.clock(),
                "armingAt": self.clock(),
                "armedAt": self.clock(),
            }
            state["transactionGuard"] = guard
            self._save_state(run_id, state)
        else:
            guard = self._validate_transaction_guard(run_id, raw_guard)
        if (
            guard.get("writerEpoch") != invocation_epoch
            or guard.get("controller") != supplied_controller
        ):
            raise HarnessError("transaction guard invocation/journal mismatch")
        if guard.get("armed") is not True:
            cleanup = state.get("cleanup")
            transaction = state.get("trustTransaction")
            if (
                lease_present
                or not isinstance(cleanup, Mapping)
                or cleanup.get("complete") is not True
                or (
                    transaction is not None
                    and (
                        not isinstance(transaction, Mapping)
                        or transaction.get("phase") != "RESTORED"
                    )
                )
            ):
                raise HarnessError("disarmed transaction guard is not fully restored")
            return {
                "schemaVersion": 1,
                "runId": run_id,
                "transactionDeadmanOnly": True,
                "recovered": False,
                "complete": True,
            }
        active_writer = guard.get("activeWriter")
        controller_to_fence = (
            active_writer.get("controller")
            if isinstance(active_writer, Mapping)
            else supplied_controller
        )
        if not lease_present:
            cleanup = state.get("cleanup")
            transaction = state.get("trustTransaction")
            if (
                not isinstance(cleanup, Mapping)
                or cleanup.get("complete") is not True
                or (
                    transaction is not None
                    and (
                        not isinstance(transaction, Mapping)
                        or transaction.get("phase") != "RESTORED"
                    )
                )
            ):
                self._stop_writers()
                self._fence_candidate_controller(
                    self._validate_controller_identity(controller_to_fence)
                )
                self._stop_writers()
                raise HarnessError("armed transaction guard lost its active-run lease")
            self._require_complete_cleanup_attestation(
                self._cleanup_attestation(run_id, state),
                require_transaction=isinstance(transaction, Mapping),
            )
            self._disarm_transaction_guard(
                run_id, state, recovery_unit=True
            )
            return {
                "schemaVersion": 1,
                "runId": run_id,
                "transactionDeadmanOnly": True,
                "recovered": False,
                "complete": True,
            }
        self._fence_candidate_controller(
            self._validate_controller_identity(controller_to_fence)
        )
        self._stop_writers()
        self._assert_active_run(run_id, allow_unclaimed=False)
        result = self.cleanup(run_id, _transaction_recovery=True)
        return {
            "schemaVersion": 1,
            "runId": run_id,
            "transactionDeadmanOnly": True,
            "recovered": bool(result.get("recovered")),
            "complete": result.get("complete") is True,
        }

    def cleanup(
        self,
        run_id: str,
        *,
        deadman_only: bool = False,
        candidate_deadman_only: bool = False,
        transaction_deadman_only: bool = False,
        candidate_deadman_epoch: str | None = None,
        candidate_controller: Mapping[str, object] | None = None,
        transaction_deadman_epoch: str | None = None,
        transaction_controller: Mapping[str, object] | None = None,
        _transaction_recovery: bool = False,
        _boot_recovery: bool = False,
    ) -> dict[str, object]:
        selected_deadmen = sum(
            bool(item)
            for item in (
                deadman_only,
                candidate_deadman_only,
                transaction_deadman_only,
            )
        )
        if selected_deadmen > 1:
            raise HarnessError("cleanup deadman mode is ambiguous")
        config_stream_run = self._existing_config_stream_run_id()
        if config_stream_run is not None:
            if config_stream_run != run_id:
                raise HarnessError(
                    "config-stream recovery lease belongs to another Fleet run"
                )
            self._stop_writers()
            raise HarnessError(
                "active config-stream transaction requires explicit recovery"
            )
        if transaction_deadman_only:
            if transaction_deadman_epoch is None or transaction_controller is None:
                raise HarnessError("transaction guard invocation identity is incomplete")
            return self._transaction_deadman_cleanup(
                run_id,
                invocation_epoch=transaction_deadman_epoch,
                invocation_controller=transaction_controller,
            )
        if candidate_deadman_only:
            return self._candidate_deadman_cleanup(
                run_id,
                invocation_epoch=candidate_deadman_epoch,
                invocation_controller=candidate_controller,
            )
        if deadman_only:
            return self._deadman_cleanup(run_id)
        with (
            self._run_lock(run_id, usb=True, allow_unclaimed=True),
            self._candidate_writer_lock(run_id),
        ):
            state = self._load_state(run_id)
            if state.get("cleanup", {}).get("complete") is True:
                fault = state.get("oakFault")
                transaction = state.get("trustTransaction")
                if (
                    not _boot_recovery
                    and isinstance(transaction, dict)
                    and transaction.get("phase") == "RESTORED"
                    and transaction.get("runtimeRestored") is False
                ):
                    self._restore_transaction(
                        run_id, state, transaction, start_runtime=True
                    )
                    state = self._load_state(run_id)
                    transaction = state.get("trustTransaction")
                if (
                    isinstance(fault, Mapping)
                    and fault.get("armed") is True
                ) or (
                    isinstance(transaction, Mapping)
                    and (
                        transaction.get("phase") != "RESTORED"
                        or (
                            not _boot_recovery
                            and transaction.get("runtimeRestored") is False
                        )
                    )
                ):
                    raise HarnessError("completed cleanup journal is not restored")
                self._purge_run_sensitive_material(run_id)
                before_release = self._cleanup_attestation(run_id, state)
                self._require_complete_cleanup_attestation(
                    before_release,
                    require_transaction=isinstance(transaction, Mapping),
                    require_lease_absent=False,
                )
                if self.paths.active_run.exists() or self.paths.active_run.is_symlink():
                    self._release_active_run(run_id)
                proof = self._cleanup_attestation(run_id, state)
                self._require_complete_cleanup_attestation(
                    proof, require_transaction=isinstance(transaction, Mapping)
                )
                candidate_soak = state.get("candidateSoak")
                candidate_deadman = (
                    candidate_soak.get("deadman")
                    if isinstance(candidate_soak, dict)
                    else None
                )
                if (
                    isinstance(candidate_soak, dict)
                    and isinstance(candidate_deadman, Mapping)
                    and (
                        candidate_deadman.get("armed") is True
                        or candidate_deadman.get("stopped") is not True
                    )
                ):
                    self._disarm_candidate_deadman(
                        run_id, state, candidate_soak
                    )
                state = self._load_state(run_id)
                if isinstance(state.get("transactionGuard"), Mapping):
                    self._disarm_transaction_guard(
                        run_id,
                        state,
                        recovery_unit=_transaction_recovery,
                    )
                return {
                    "schemaVersion": 1,
                    "kind": "nuvion-iq9075-cleanup-evidence",
                    "runId": run_id,
                    "complete": True,
                    "recovered": False,
                    "phase": transaction.get("phase")
                    if isinstance(transaction, Mapping)
                    else None,
                    "idempotent": True,
                    "proof": proof,
                }
            candidate_recovered = self._cleanup_candidate_soak(
                run_id, state, boot_recovery=_boot_recovery
            )
            fault = state.get("oakFault")
            recovered = candidate_recovered
            if isinstance(fault, dict) and fault.get("armed") is True:
                port = canonical_oak_port(fault.get("port"))
                if fault.get("unit") != self._deadman_unit(run_id):
                    raise HarnessError("OAK cleanup ownership mismatch")
                self._recover_oak(port)
                recovered = True
                fault.update(
                    {"armed": False, "recovered": True, "recoveredAt": self.clock()}
                )
                state["oakFault"] = fault
                self._save_state(run_id, state)
                self._stop_deadman(str(fault["unit"]))
            transaction = state.get("trustTransaction")
            manifest: dict[str, Any] | None = None
            if isinstance(transaction, dict) and transaction.get("phase") != "RESTORED":
                manifest = self._transaction_manifest(run_id, transaction)
            self._cleanup_known_staging(run_id, manifest)
            if isinstance(transaction, dict):
                if transaction.get("phase") != "RESTORED" or (
                    not _boot_recovery
                    and transaction.get("runtimeRestored") is False
                ):
                    try:
                        self._restore_transaction(
                            run_id,
                            state,
                            transaction,
                            start_runtime=not _boot_recovery,
                        )
                    except Exception:  # noqa: BLE001 - incomplete cleanup stays fail-closed.
                        try:
                            if _boot_recovery:
                                self._stop_writers_for_boot()
                            else:
                                self._stop_writers()
                        except BaseException:
                            pass
                        return {
                            "schemaVersion": 1,
                            "runId": run_id,
                            "complete": False,
                            "recovered": recovered,
                            "phase": "RESTORING",
                        }
                candidate_soak = state.get("candidateSoak")
                if isinstance(candidate_soak, dict):
                    candidate_soak["productionRestoration"] = (
                        self._production_restoration_attestation(
                            run_id,
                            state,
                            transaction,
                            boot_recovery=_boot_recovery,
                        )
                    )
                    state["candidateSoak"] = candidate_soak
                    self._save_state(run_id, state)
            complete = not (
                isinstance(state.get("oakFault"), dict)
                and state["oakFault"].get("armed") is True
            ) and (
                not isinstance(transaction, dict)
                or (
                    transaction.get("phase") == "RESTORED"
                    and (
                        _boot_recovery
                        or transaction.get("runtimeRestored") is not False
                    )
                )
            )
            if complete:
                self._purge_run_sensitive_material(run_id)
                backup_state = state.get("backup")
                if isinstance(backup_state, dict) and backup_state.get("complete") is not True:
                    backup_state.update(
                        {
                            "phase": "RESTORED",
                            "complete": True,
                            "recoveryAbandoned": True,
                            "restoredAt": self.clock(),
                        }
                    )
                    state["backup"] = backup_state
                state["cleanup"] = {"complete": True, "completedAt": self.clock()}
                self._save_state(run_id, state)
                before_release = self._cleanup_attestation(run_id, state)
                self._require_complete_cleanup_attestation(
                    before_release,
                    require_transaction=isinstance(transaction, Mapping),
                    require_lease_absent=False,
                )
                if self.paths.active_run.exists() or self.paths.active_run.is_symlink():
                    self._release_active_run(run_id)
                proof = self._cleanup_attestation(run_id, state)
                self._require_complete_cleanup_attestation(
                    proof, require_transaction=isinstance(transaction, Mapping)
                )
                candidate_soak = state.get("candidateSoak")
                candidate_deadman = (
                    candidate_soak.get("deadman")
                    if isinstance(candidate_soak, dict)
                    else None
                )
                if (
                    isinstance(candidate_soak, dict)
                    and isinstance(candidate_deadman, Mapping)
                    and (
                        candidate_deadman.get("armed") is True
                        or candidate_deadman.get("stopped") is not True
                    )
                ):
                    self._disarm_candidate_deadman(
                        run_id, state, candidate_soak
                    )
                state = self._load_state(run_id)
                if isinstance(state.get("transactionGuard"), Mapping):
                    self._disarm_transaction_guard(
                        run_id,
                        state,
                        recovery_unit=_transaction_recovery,
                    )
            else:
                proof = None
            return {
                "schemaVersion": 1,
                "kind": "nuvion-iq9075-cleanup-evidence",
                "runId": run_id,
                "complete": complete,
                "recovered": recovered,
                "phase": transaction.get("phase")
                if isinstance(transaction, dict)
                else None,
                "proof": proof,
            }

    def _scenario_gate(
        self,
        manifest: Mapping[str, Any],
        updater: Mapping[str, object],
        current: str,
        previous: str | None,
        current_version: str,
        marker: Mapping[str, object],
        previous_marker: Mapping[str, object],
        state: Mapping[str, Any],
    ) -> bool:
        scenario = manifest["scenario"]
        update = updater.get("update")
        if not isinstance(update, Mapping):
            return False
        common = (
            update.get("commandId") == scenario["expectedCommandId"]
            and update.get("bomDigest") == scenario["expectedBomDigest"]
            and update.get("targetVersion") == scenario["release"]["agentVersion"]
            and update.get("candidateSlot") == scenario["expectedCandidateSlot"]
            and update.get("previousSlot") == scenario["expectedPreviousSlot"]
            and update.get("previousVersion") == scenario["expectedPreviousVersion"]
            and update.get("releaseSequence") == scenario["release"]["releaseSequence"]
            and update.get("artifactDigest") == scenario["release"]["artifactDigest"]
            and update.get("componentSha") == scenario["release"]["componentSha"]
            and update.get("configSchema") == scenario["release"]["configSchema"]
            and update.get("publisherKeyId") == scenario["release"]["publisherKeyId"]
            and update.get("bomVerificationStatus") == "VERIFIED"
        )
        expected_marker = {
            "schemaVersion": 2,
            "bomDigest": scenario["expectedBomDigest"],
            **scenario["release"],
        }
        if scenario["type"] == "commit":
            return (
                common
                and update.get("phase") == "COMMITTED"
                and update.get("slot")
                == "releases/" + scenario["expectedBomDigest"][7:]
                and update.get("health") == "FUNCTIONAL_HEALTHY"
                and update.get("functionalHealth") == "FUNCTIONAL_HEALTHY"
                and current == "releases/" + scenario["expectedBomDigest"][7:]
                and previous == scenario["expectedPreviousSlot"]
                and current_version == scenario["release"]["agentVersion"]
                and marker == expected_marker
            )
        fault = state.get("oakFault")
        return (
            common
            and update.get("phase") == "ROLLED_BACK"
            and current == scenario["expectedPreviousSlot"]
            and previous == "releases/" + scenario["expectedBomDigest"][7:]
            and previous_marker == expected_marker
            and current_version == scenario["expectedPreviousVersion"]
            and update.get("slot") == scenario["expectedPreviousSlot"]
            and update.get("rollbackSlot") == scenario["expectedPreviousSlot"]
            and update.get("rollbackVersion") == scenario["expectedPreviousVersion"]
            and update.get("errorCode") == "ROLLED_BACK"
            and update.get("health") == "LKG_RESTORED"
            and update.get("functionalHealth") == "FUNCTIONAL_UNHEALTHY"
            and isinstance(fault, Mapping)
            and fault.get("armed") is False
            and fault.get("recovered") is True
        )

    def evidence(
        self, run_id: str, *, output: str | Path | None = None
    ) -> dict[str, object]:
        with self._run_lock(run_id):
            state = self._load_state(run_id)
            transaction = state.get("trustTransaction")
            if not isinstance(transaction, dict):
                raise HarnessError("trust transaction is missing")
            manifest = self._transaction_manifest(run_id, transaction)
            foundation_live = False
            try:
                self._foundation()
                foundation_live = True
            except (HarnessError, OSError):
                pass
            backup_verified = False
            backup = state.get("backup")
            if isinstance(backup, dict) and backup.get("complete") is True:
                try:
                    self._verify_archive(
                        run_id,
                        Path(str(backup.get("archivePath") or "")),
                        str(backup.get("sha256") or ""),
                    )
                    backup_verified = True
                except (HarnessError, OSError):
                    pass
            trust_live = (
                transaction.get("phase") == "APPLIED"
                and transaction.get("liveVerified") is True
                and self._transaction_matches(run_id, transaction)
            )
            services = {unit: self._unit_status(unit) for unit in UNITS}
            service_gate = all(
                item["active"] is True
                for item in services.values()
            )
            updater: dict[str, object]
            updater_gate = False
            try:
                updater = self._probe_updater()
                updater_gate = updater["updaterVersion"] == REQUIRED_UPDATER_VERSION
            except HarnessError:
                updater = {
                    "capabilityAvailable": False,
                    "authenticatedHelper": False,
                    "reason": "UNAVAILABLE",
                    "updaterVersion": "unknown",
                }
            oak_gate = False
            try:
                oak = self.verify_oak()
                oak_gate = True
            except (HarnessError, OSError):
                oak = {
                    "port": None,
                    "vendorId": OAK_VENDOR,
                    "productId": OAK_PRODUCT,
                    "speedMbps": None,
                    "mxidSha256": None,
                    "attached": False,
                    "bound": False,
                }
            current = self._slot_link("current", required=True)
            previous = self._slot_link("previous", required=False)
            marker: dict[str, object] = {}
            previous_marker: dict[str, object] = {}
            current_version = ""
            try:
                if current is not None:
                    current_version = self._slot_version(current)
                    if current.startswith("releases/"):
                        marker = self._release_marker(current)
                if previous is not None and previous.startswith("releases/"):
                    previous_marker = self._release_marker(previous)
            except HarnessError:
                pass
            scenario_gate = self._scenario_gate(
                manifest,
                updater,
                str(current),
                previous,
                current_version,
                marker,
                previous_marker,
                state,
            )
            runtime_pids: dict[str, int] | None = None
            if manifest["scenario"]["type"] == "oak-fault-rollback":
                fault = state.get("oakFault")
                candidate_pid = (
                    fault.get("candidatePid") if isinstance(fault, Mapping) else None
                )
                restored_pid = services["nuv-agent.service"].get("mainPid")
                pid_gate = (
                    isinstance(candidate_pid, int)
                    and not isinstance(candidate_pid, bool)
                    and candidate_pid >= 2
                    and isinstance(restored_pid, int)
                    and not isinstance(restored_pid, bool)
                    and restored_pid >= 2
                    and restored_pid != candidate_pid
                )
                scenario_gate = scenario_gate and pid_gate
                if pid_gate:
                    runtime_pids = {
                        "candidate": candidate_pid,
                        "restored": restored_pid,
                    }
            anti_replay: dict[str, object] | None = None
            if scenario_gate and isinstance(updater.get("update"), Mapping):
                try:
                    snapshot = self._anti_replay_snapshot()
                    update = updater["update"]
                    terminal = {
                        "commandId": update.get("commandId"),
                        "sequence": update.get("sequence"),
                        "phase": update.get("phase"),
                        "bomDigest": update.get("bomDigest"),
                        "releaseSequence": update.get("releaseSequence"),
                        "healthDeadline": None,
                    }
                    self._validate_candidate_anti_replay(snapshot, terminal)
                    current_release_sequence = (
                        str(marker.get("releaseSequence")) if marker else None
                    )
                    current_bom_digest = marker.get("bomDigest") if marker else None
                    if (
                        snapshot.get("currentReleaseSequence")
                        != current_release_sequence
                        or snapshot.get("currentBomDigest") != current_bom_digest
                    ):
                        raise HarnessError(
                            "updater anti-replay journal differs from current slot"
                        )
                    anti_replay = snapshot
                except (HarnessError, OSError):
                    # Schema v1 remains readable for historical physical evidence.
                    # New Fleet Runtime release evidence requires the v2 snapshot.
                    anti_replay = None
            gates = {
                "foundation": state.get("foundation", {}).get("verified") is True
                and foundation_live,
                "backup": backup_verified,
                "trust": trust_live,
                "updater2": updater_gate,
                "oak": oak_gate,
                "services": service_gate,
                "scenario": scenario_gate,
            }
            complete = all(gates.values())
            result: dict[str, object] = {
                "schemaVersion": 2 if anti_replay is not None else 1,
                "protocolVersion": PROTOCOL_VERSION,
                "runId": run_id,
                "generatedAt": self.clock(),
                "scenario": manifest["scenario"]["type"],
                "complete": complete,
                "gates": gates,
                "oak": oak,
                "services": services,
                "runtimePids": runtime_pids,
                "slots": {
                    "current": current,
                    "previous": previous,
                    "currentVersion": current_version or None,
                    "release": marker or None,
                    "previousRelease": previous_marker or None,
                },
                "updater": updater,
            }
            if anti_replay is not None:
                result["antiReplay"] = anti_replay
            # Scan the freshly generated envelope before any complete evidence
            # bytes become durable. A persisted retry is scanned again below.
            assert_no_secret_material(result)
            if complete:
                canonical_evidence_path = (
                    self.paths.state_root / run_id / "evidence.json"
                )
                canonical_evidence = (
                    json.dumps(
                        result,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8")
                existing_evidence = state.get("fleetEvidence")
                if isinstance(existing_evidence, Mapping):
                    persisted_payload, persisted_metadata = read_regular(
                        canonical_evidence_path, maximum=MAX_STATE_BYTES
                    )
                    persisted = strict_json(
                        persisted_payload, label="persisted Fleet evidence"
                    )
                    if (
                        set(existing_evidence)
                        != {"complete", "scenario", "sha256", "generatedAt"}
                        or existing_evidence.get("complete") is not True
                        or existing_evidence.get("scenario")
                        != manifest["scenario"]["type"]
                        or not isinstance(existing_evidence.get("sha256"), str)
                        or SHA256_RE.fullmatch(existing_evidence["sha256"]) is None
                        or sha256_bytes(persisted_payload)
                        != existing_evidence["sha256"]
                        or persisted_metadata.st_uid != self.root_uid
                        or stat.S_IMODE(persisted_metadata.st_mode) != 0o600
                        or persisted.get("runId") != run_id
                        or persisted.get("scenario")
                        != manifest["scenario"]["type"]
                        or persisted.get("complete") is not True
                        or persisted.get("generatedAt")
                        != existing_evidence.get("generatedAt")
                    ):
                        raise HarnessError("persisted Fleet evidence changed")
                    # The live gates above were freshly recomputed. Return the
                    # first immutable complete envelope so board and host retain
                    # one stable digest across reconnects and process retries.
                    result = persisted
                elif existing_evidence is not None:
                    raise HarnessError("persisted Fleet evidence journal is invalid")
                else:
                    atomic_write(
                        canonical_evidence_path,
                        canonical_evidence,
                        mode=0o600,
                        uid=self.root_uid,
                        gid=self.root_gid,
                    )
                    state["fleetEvidence"] = {
                        "complete": True,
                        "scenario": manifest["scenario"]["type"],
                        "sha256": sha256_bytes(canonical_evidence),
                        "generatedAt": result["generatedAt"],
                    }
                    self._save_state(run_id, state)
            assert_no_secret_material(result)
            if output is not None:
                if not complete and isinstance(state.get("fleetEvidence"), Mapping):
                    raise HarnessError(
                        "incomplete live evidence cannot replace immutable evidence"
                    )
                path = Path(output)
                expected = self.paths.state_root / run_id / "evidence.json"
                if path != expected:
                    raise HarnessError(
                        "evidence output must be the run-owned fixed path"
                    )
                payload = (
                    json.dumps(
                        result, sort_keys=True, separators=(",", ":"), allow_nan=False
                    )
                    + "\n"
                ).encode("utf-8")
                try:
                    atomic_write(
                        path,
                        payload,
                        mode=0o600,
                        uid=self.root_uid,
                        gid=self.root_gid,
                    )
                    verify, _ = read_regular(path, maximum=MAX_STATE_BYTES)
                    assert_no_secret_material(strict_json(verify, label="evidence"))
                except BaseException:
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
                    raise
            return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IQ9075 Fleet E2E board primitives")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("identity")
    commands.add_parser("resume-boot-gate")
    boot_reconcile = commands.add_parser("boot-reconcile")
    boot_reconcile.add_argument(
        "--package-maintenance", action="store_true", help=argparse.SUPPRESS
    )

    def run_command(name: str) -> argparse.ArgumentParser:
        item = commands.add_parser(name)
        item.add_argument("--run-id", required=True)
        return item

    run_command("preflight")
    run_command("backup")
    enable = run_command("enable-fleet")
    for role in ("command", "release", "health"):
        enable.add_argument(f"--{role}-keyring", required=True)
        enable.add_argument(f"--{role}-sha256", required=True)
    enable.add_argument("--device-binding", required=True)
    enable.add_argument("--binding-sha256", required=True)
    enable.add_argument("--manifest", required=True)
    enable.add_argument("--manifest-sha256", required=True)
    discard = run_command("discard-staging")
    for role in ("command", "release", "health", "binding", "manifest"):
        discard.add_argument(f"--{role}-sha256", required=True)
    soak = run_command("candidate-soak")
    soak.add_argument("--candidate-bundle", required=True)
    soak.add_argument("--bundle-sha256", required=True)
    soak.add_argument("--candidate-bom", required=True)
    soak.add_argument("--bom-sha256", required=True)
    soak.add_argument("--oak-harness", required=True)
    soak.add_argument("--harness-sha256", required=True)
    discard_candidate = run_command("discard-candidate-staging")
    discard_candidate.add_argument("--bundle-sha256", required=True)
    discard_candidate.add_argument("--bom-sha256", required=True)
    discard_candidate.add_argument("--harness-sha256", required=True)
    run_command("arm-oak-fault")
    evidence = run_command("evidence")
    evidence.add_argument("--output")
    cleanup = run_command("cleanup")
    cleanup.add_argument("--deadman-only", action="store_true", help=argparse.SUPPRESS)
    cleanup.add_argument(
        "--candidate-deadman-only", action="store_true", help=argparse.SUPPRESS
    )
    cleanup.add_argument("--candidate-deadman-epoch", help=argparse.SUPPRESS)
    cleanup.add_argument("--candidate-controller-pid", type=int, help=argparse.SUPPRESS)
    cleanup.add_argument(
        "--candidate-controller-start-ticks", type=int, help=argparse.SUPPRESS
    )
    cleanup.add_argument("--candidate-controller-boot-id", help=argparse.SUPPRESS)
    cleanup.add_argument(
        "--transaction-deadman-only", action="store_true", help=argparse.SUPPRESS
    )
    cleanup.add_argument("--transaction-deadman-epoch", help=argparse.SUPPRESS)
    cleanup.add_argument(
        "--transaction-controller-pid", type=int, help=argparse.SUPPRESS
    )
    cleanup.add_argument(
        "--transaction-controller-start-ticks", type=int, help=argparse.SUPPRESS
    )
    cleanup.add_argument(
        "--transaction-controller-boot-id", help=argparse.SUPPRESS
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        disable_core_dumps()
    except HarnessError as exc:
        print(f"iq9075-board-e2e: {exc}", file=sys.stderr)
        return 1
    if os.geteuid() != 0:
        print("iq9075-board-e2e: root privileges are required", file=sys.stderr)
        return 1
    arguments = build_parser().parse_args(argv)
    try:
        harness = BoardHarness()
        if arguments.command == "identity":
            result = harness.identity()
        elif arguments.command == "resume-boot-gate":
            result = harness.resume_boot_gate()
        elif arguments.command == "boot-reconcile":
            result = harness.boot_reconcile(
                package_maintenance=arguments.package_maintenance
            )
        else:
            run_id = canonical_run_id(arguments.run_id)
            if arguments.command == "preflight":
                result = harness.preflight(run_id)
            elif arguments.command == "backup":
                result = harness.backup(run_id)
            elif arguments.command == "enable-fleet":
                result = harness.enable_fleet(
                    run_id,
                    command_keyring=arguments.command_keyring,
                    command_sha256=arguments.command_sha256,
                    release_keyring=arguments.release_keyring,
                    release_sha256=arguments.release_sha256,
                    health_keyring=arguments.health_keyring,
                    health_sha256=arguments.health_sha256,
                    device_binding=arguments.device_binding,
                    binding_sha256=arguments.binding_sha256,
                    manifest_path=arguments.manifest,
                    manifest_sha256=arguments.manifest_sha256,
                )
            elif arguments.command == "arm-oak-fault":
                result = harness.arm_oak_fault(run_id)
            elif arguments.command == "discard-staging":
                result = harness.discard_staging(
                    run_id,
                    command_sha256=arguments.command_sha256,
                    release_sha256=arguments.release_sha256,
                    health_sha256=arguments.health_sha256,
                    binding_sha256=arguments.binding_sha256,
                    manifest_sha256=arguments.manifest_sha256,
                )
            elif arguments.command == "evidence":
                result = harness.evidence(run_id, output=arguments.output)
            elif arguments.command == "candidate-soak":
                result = harness.candidate_soak(
                    run_id,
                    candidate_bundle=arguments.candidate_bundle,
                    bundle_sha256=arguments.bundle_sha256,
                    candidate_bom=arguments.candidate_bom,
                    bom_sha256=arguments.bom_sha256,
                    oak_harness=arguments.oak_harness,
                    harness_sha256=arguments.harness_sha256,
                )
            elif arguments.command == "discard-candidate-staging":
                result = harness.discard_candidate_staging(
                    run_id,
                    bundle_sha256=arguments.bundle_sha256,
                    bom_sha256=arguments.bom_sha256,
                    harness_sha256=arguments.harness_sha256,
                )
            elif arguments.command == "cleanup":
                controller_values = (
                    arguments.candidate_controller_pid,
                    arguments.candidate_controller_start_ticks,
                    arguments.candidate_controller_boot_id,
                )
                if any(value is not None for value in controller_values) and not all(
                    value is not None for value in controller_values
                ):
                    raise HarnessError("candidate deadman invocation identity is incomplete")
                candidate_controller = (
                    {
                        "pid": arguments.candidate_controller_pid,
                        "startTicks": arguments.candidate_controller_start_ticks,
                        "bootId": arguments.candidate_controller_boot_id,
                    }
                    if all(value is not None for value in controller_values)
                    else None
                )
                transaction_controller_values = (
                    arguments.transaction_controller_pid,
                    arguments.transaction_controller_start_ticks,
                    arguments.transaction_controller_boot_id,
                )
                if any(
                    value is not None for value in transaction_controller_values
                ) and not all(
                    value is not None for value in transaction_controller_values
                ):
                    raise HarnessError(
                        "transaction guard invocation identity is incomplete"
                    )
                transaction_controller = (
                    {
                        "pid": arguments.transaction_controller_pid,
                        "startTicks": arguments.transaction_controller_start_ticks,
                        "bootId": arguments.transaction_controller_boot_id,
                    }
                    if all(
                        value is not None for value in transaction_controller_values
                    )
                    else None
                )
                result = harness.cleanup(
                    run_id,
                    deadman_only=arguments.deadman_only,
                    candidate_deadman_only=arguments.candidate_deadman_only,
                    transaction_deadman_only=arguments.transaction_deadman_only,
                    candidate_deadman_epoch=arguments.candidate_deadman_epoch,
                    candidate_controller=candidate_controller,
                    transaction_deadman_epoch=arguments.transaction_deadman_epoch,
                    transaction_controller=transaction_controller,
                )
            else:  # pragma: no cover
                raise HarnessError("unsupported command")
        assert_no_secret_material(result)
        print(
            json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
        )
        return 0
    except (HarnessError, OSError, ValueError) as exc:
        print(f"iq9075-board-e2e: {exc}", file=sys.stderr)
        return 1
    except Exception:  # noqa: BLE001 - never expose a root-tool traceback remotely.
        print("iq9075-board-e2e: unexpected internal failure", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
