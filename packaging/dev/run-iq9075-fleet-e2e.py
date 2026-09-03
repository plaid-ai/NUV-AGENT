#!/usr/bin/env python3
"""Strict-host IQ9075 Fleet E2E runner with immutable run manifests."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import resource
import shlex
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

PROTOCOL_VERSION = "iq9075-fleet-e2e-v2"
DEFAULT_HOST = "iq9075"
DEFAULT_USER = "plaid"
DEFAULT_FINGERPRINT = "SHA256:qOaWjGEiWC+Jr5JMEbemUyFc3PkZkh3fD/7Yqx3Mx1Y"
REMOTE_TOOL = "/usr/local/libexec/nuvion/iq9075-board-e2e.py"
REMOTE_BOOTSTRAP_INSTALLER = "/tmp/nuvion-fleet-e2e-{run_id}-bootstrap-installer.sh"
REMOTE_BOOTSTRAP_DEB = "/tmp/nuvion-fleet-e2e-{run_id}-bootstrap.deb"
REQUIRED_UPDATER_VERSION = "0.2.0"
MAX_SECRET_BYTES = 4096
MAX_INPUT_BYTES = 64 * 1024
MAX_OUTPUT_BYTES = 2 * 1024 * 1024
MAX_BOOTSTRAP_INSTALLER_BYTES = 2 * 1024 * 1024
MAX_BOOTSTRAP_DEB_BYTES = 4 * 1024 * 1024 * 1024
MAX_CANDIDATE_BUNDLE_BYTES = 4 * 1024 * 1024 * 1024
RUN_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
FINGERPRINT_RE = re.compile(r"^SHA256:[A-Za-z0-9+/]{43}$")
HOST_RE = re.compile(
    r"^(?:[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?|[0-9a-fA-F:]+)$"
)
USER_RE = re.compile(r"^[a-z_][a-z0-9_-]{0,31}$")
DEVICE_ID_RE = re.compile(r"^sp-([1-9][0-9]*)-nuvion-[a-z0-9][a-z0-9-]{0,100}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
COMPONENT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
KEY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
SEMVER_RE = re.compile(
    r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)
BOOTSTRAP_FAILURE_RE = re.compile(
    r"^out-of-band updater bootstrap failed: "
    r"primary=[A-Za-z0-9 .:_-]{1,160}; "
    r"cleanup=(?:none|[A-Za-z0-9._:-]+(?:,[A-Za-z0-9._:-]+)*)$"
)
BOARD_COMMANDS = frozenset(
    {
        "identity",
        "resume-boot-gate",
        "preflight",
        "backup",
        "enable-fleet",
        "discard-staging",
        "arm-oak-fault",
        "evidence",
        "cleanup",
        "candidate-soak",
        "discard-candidate-staging",
    }
)
CANDIDATE_PERSISTENT_PATHS = (
    "/etc/nuv-agent",
    "/etc/nuvion-updater",
    "/var/lib/nuv-agent",
    "/var/lib/nuvion-updater",
)
CANDIDATE_TMPFS_LIMITS = {
    "/tmp": {"bytes": 256 * 1024 * 1024, "inodes": 8192},
    "/var/tmp": {"bytes": 64 * 1024 * 1024, "inodes": 4096},
    "/dev/shm": {"bytes": 256 * 1024 * 1024, "inodes": 8192},
}
CANDIDATE_INACCESSIBLE_PATHS = ("/run/user",)
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
FIXED_DESTINATIONS = {
    "agentCommand": "/etc/nuv-agent/fleet-command-keyring.json",
    "updaterCommand": "/etc/nuvion-updater/command-keyring.json",
    "release": "/etc/nuvion-updater/release-keyring.json",
    "health": "/etc/nuvion-updater/health-attestation-keyring.json",
    "binding": "/etc/nuvion-updater/device-binding.json",
}
PRODUCTION_TRANSACTION_FILES = frozenset(
    {"/etc/nuv-agent/agent.env", *FIXED_DESTINATIONS.values()}
)
PRODUCTION_TRANSACTION_DIRECTORIES = frozenset(
    {"/etc/nuv-agent", "/etc/nuvion-updater"}
)
PRODUCTION_UNITS = frozenset(
    {
        "nuv-agent.service",
        "nuv-agent-updater.service",
        "nuv-agent-updater.socket",
    }
)
CANDIDATE_REQUIRED_SOAK_SECONDS = 120
CANDIDATE_UID_SCAN_INTERVAL_SECONDS = 0.05
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
CLEANUP_IDENTITY_FIELDS = (
    "deviceId",
    "spaceId",
    "productModel",
    "platformProfile",
    "hardwareRevision",
    "architecture",
    "dockerRequired",
)

BOOTSTRAP_REMOTE_PROGRAM = r"""
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

run_id, installer_source, deb_source, installer_sha, deb_sha, version, board_user = sys.argv[1:]
uuid4 = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z")
sha256 = re.compile(r"[0-9a-f]{64}\Z")
semver = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?\Z")
user_re = re.compile(r"[a-z_][a-z0-9_-]{0,31}\Z")
expected_installer = f"/tmp/nuvion-fleet-e2e-{run_id}-bootstrap-installer.sh"
expected_deb = f"/tmp/nuvion-fleet-e2e-{run_id}-bootstrap.deb"
if not uuid4.fullmatch(run_id) or not sha256.fullmatch(installer_sha) or not sha256.fullmatch(deb_sha):
    raise SystemExit("invalid bootstrap identity")
if not semver.fullmatch(version) or not user_re.fullmatch(board_user):
    raise SystemExit("invalid bootstrap package identity")
if installer_source != expected_installer or deb_source != expected_deb:
    raise SystemExit("bootstrap paths are not fixed")

bootstrap_root = Path("/var/lib/nuvion-fleet-e2e/bootstrap")
private = bootstrap_root / run_id
private_installer = private / "install-iq9075.sh"
private_deb = private / "nuv-agent.deb"

def safe_unlink(path):
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return
    if stat.S_ISDIR(metadata.st_mode):
        raise SystemExit("bootstrap staging target is a directory")
    os.unlink(path)

def safe_rmtree(path):
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise SystemExit("private bootstrap staging is unsafe")
    if metadata.st_uid != 0 or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise SystemExit("private bootstrap staging ownership is unsafe")
    shutil.rmtree(path)

def ensure_root_directory(path, mode):
    path.mkdir(parents=True, exist_ok=True)
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise SystemExit("bootstrap root is unsafe")
    if metadata.st_uid != 0:
        raise SystemExit("bootstrap root is not root-owned")
    os.chmod(path, mode)

def copy_verified(source, destination, expected, maximum):
    before = os.lstat(source)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise SystemExit("bootstrap input is not a regular file")
    if before.st_nlink != 1 or before.st_size > maximum or stat.S_IMODE(before.st_mode) & 0o022:
        raise SystemExit("bootstrap input metadata is unsafe")
    source_fd = os.open(source, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    destination_fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    digest = hashlib.sha256()
    total = 0
    try:
        opened = os.fstat(source_fd)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (before.st_dev, before.st_ino, before.st_size):
            raise SystemExit("bootstrap input changed while opening")
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise SystemExit("bootstrap input exceeds size limit")
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                if written <= 0:
                    raise SystemExit("bootstrap copy made no progress")
                view = view[written:]
        after = os.fstat(source_fd)
        if (after.st_dev, after.st_ino, after.st_size) != (opened.st_dev, opened.st_ino, opened.st_size):
            raise SystemExit("bootstrap input changed while reading")
        if digest.hexdigest() != expected:
            raise SystemExit("bootstrap input digest mismatch")
        os.fchmod(destination_fd, 0o600)
        os.fsync(destination_fd)
    finally:
        os.close(source_fd)
        os.close(destination_fd)

def query_package_version():
    result = subprocess.run(["/usr/bin/dpkg-query", "-W", "-f=${Version}", "nuv-agent"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=20, check=False, text=True)
    return result.stdout.strip()[:128] if result.returncode == 0 else None

def control(field):
    result = subprocess.run(["/usr/bin/dpkg-deb", "-f", str(private_deb), field], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=20, check=False, text=True)
    if result.returncode != 0:
        raise SystemExit("cannot inspect bootstrap package")
    return result.stdout.strip()

RUNTIME_UNITS = (
    "nuv-agent.service",
    "nuv-agent-updater.socket",
    "nuv-agent-updater.service",
)

def quiesce_runtime():
    errors = []
    for unit in RUNTIME_UNITS:
        for action in ("stop", "disable"):
            try:
                result = subprocess.run(
                    ["/usr/bin/systemctl", action, unit],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                    check=False,
                )
                if result.returncode != 0:
                    errors.append(f"{unit}:{action}:rc{result.returncode}")
            except BaseException as exc:
                errors.append(f"{unit}:{action}:{type(exc).__name__}")
    for unit in RUNTIME_UNITS:
        for action, expected in (("is-active", "inactive"), ("is-enabled", "disabled")):
            try:
                result = subprocess.run(
                    ["/usr/bin/systemctl", action, unit],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                    check=False,
                    text=True,
                )
                if result.stdout.strip() != expected:
                    errors.append(f"{unit}:{action}:not-{expected}")
            except BaseException as exc:
                errors.append(f"{unit}:{action}:{type(exc).__name__}")
    return errors

def stable_failure(exc):
    if isinstance(exc, SystemExit) and isinstance(exc.code, str) and re.fullmatch(r"[A-Za-z0-9 .:_-]{1,160}", exc.code):
        return exc.code
    return type(exc).__name__

def bootstrap_failure_message(primary, cleanup):
    primary_text = primary or "none"
    cleanup_text = ",".join(cleanup) if cleanup else "none"
    return f"out-of-band updater bootstrap failed: primary={primary_text}; cleanup={cleanup_text}"

primary_failure = None
cleanup_failures = []
evidence = None
try:
    ensure_root_directory(bootstrap_root.parent, 0o700)
    ensure_root_directory(bootstrap_root, 0o700)
    safe_rmtree(private)
    private.mkdir(mode=0o700)
    previous_version = query_package_version()
    copy_verified(installer_source, private_installer, installer_sha, 2 * 1024 * 1024)
    copy_verified(deb_source, private_deb, deb_sha, 4 * 1024 * 1024 * 1024)
    if control("Package") != "nuv-agent" or control("Architecture") != "arm64" or control("Version") != version:
        raise SystemExit("bootstrap package metadata mismatch")
    environment = {
        "HOME": "/root",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "SUDO_USER": board_user,
    }
    installed = subprocess.run(
        ["/bin/bash", str(private_installer), str(private_deb), "--expected-version", version, "--expected-sha256", deb_sha, "--camera", "oak"],
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=1800,
        check=False,
    )
    if installed.returncode != 0:
        raise SystemExit("out-of-band updater bootstrap failed")
    installed_version = query_package_version()
    if installed_version != version:
        raise SystemExit("installed package version mismatch")
    sys.path.insert(0, "/usr/lib/nuvion-updater")
    from nuvion_updater.version import UPDATER_VERSION
    if UPDATER_VERSION != "0.2.0":
        raise SystemExit("bootstrap did not install updater 0.2.0")
    tool = Path("/usr/local/libexec/nuvion/iq9075-board-e2e.py")
    tool_metadata = tool.lstat()
    if not stat.S_ISREG(tool_metadata.st_mode) or stat.S_ISLNK(tool_metadata.st_mode) or tool_metadata.st_uid != 0 or stat.S_IMODE(tool_metadata.st_mode) != 0o755:
        raise SystemExit("packaged board tool identity is unsafe")
    tool_sha = hashlib.sha256(tool.read_bytes()).hexdigest()
    current = Path("/opt/nuv-agent/current")
    current_metadata = current.lstat()
    if not stat.S_ISLNK(current_metadata.st_mode):
        raise SystemExit("bootstrap current slot is not a symlink")
    current_slot = os.readlink(current)
    if re.fullmatch(r"(?:bootstrap/[0-9A-Za-z.+-]{1,64}|releases/[0-9a-f]{64})", current_slot) is None:
        raise SystemExit("bootstrap current slot is invalid")
    evidence = {
        "schemaVersion": 1,
        "protocolVersion": "iq9075-fleet-e2e-v2",
        "runId": run_id,
        "outOfBandBootstrap": True,
        "otaEvidence": False,
        "previousPackageVersion": previous_version,
        "installedPackageVersion": installed_version,
        "packageSha256": deb_sha,
        "installerSha256": installer_sha,
        "updaterCodeVersion": UPDATER_VERSION,
        "boardToolSha256": tool_sha,
        "currentSlot": current_slot,
    }
except BaseException as exc:
    primary_failure = stable_failure(exc)
finally:
    cleanup_failures.extend(quiesce_runtime())
    for label, cleanup in (
        ("installer-staging", lambda: safe_unlink(installer_source)),
        ("package-staging", lambda: safe_unlink(deb_source)),
        ("private-staging", lambda: safe_rmtree(private)),
    ):
        try:
            cleanup()
        except BaseException as exc:
            cleanup_failures.append(f"{label}:{type(exc).__name__}")

if primary_failure is not None or cleanup_failures:
    raise SystemExit(bootstrap_failure_message(primary_failure, cleanup_failures))
if not isinstance(evidence, dict):
    raise SystemExit("out-of-band updater bootstrap produced no evidence")
evidence["servicesInactive"] = True
print(json.dumps(evidence, sort_keys=True, separators=(",", ":")))
""".strip()

BOOTSTRAP_CLEANUP_PROGRAM = r"""
import json
import os
import re
import shutil
import stat
import sys
from pathlib import Path

run_id, installer, package = sys.argv[1:]
if re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", run_id) is None:
    raise SystemExit("invalid bootstrap cleanup identity")
if installer != f"/tmp/nuvion-fleet-e2e-{run_id}-bootstrap-installer.sh" or package != f"/tmp/nuvion-fleet-e2e-{run_id}-bootstrap.deb":
    raise SystemExit("bootstrap cleanup paths are not fixed")
for path in (installer, package):
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        continue
    if stat.S_ISDIR(metadata.st_mode):
        raise SystemExit("bootstrap cleanup target is a directory")
    os.unlink(path)
private = Path("/var/lib/nuvion-fleet-e2e/bootstrap") / run_id
try:
    metadata = private.lstat()
except FileNotFoundError:
    pass
else:
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != 0 or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise SystemExit("private bootstrap cleanup target is unsafe")
    shutil.rmtree(private)
print(json.dumps({"schemaVersion": 1, "runId": run_id, "complete": True}, sort_keys=True, separators=(",", ":")))
""".strip()


class RunnerError(RuntimeError):
    """Stable host-side failure without remote output or credential material."""


def disable_core_dumps() -> None:
    """Fail closed before credentials can enter this process."""

    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (OSError, ValueError) as exc:
        raise RunnerError("cannot disable core dumps") from exc


def strict_json(payload: bytes | str, *, label: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RunnerError(f"{label} contains duplicate JSON members")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    try:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        value = json.loads(
            text,
            object_pairs_hook=unique,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RunnerError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise RunnerError(f"{label} root is not an object")
    return value


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
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
        raise RunnerError("evidence is not canonical JSON data") from exc


def strict_canonical_json(payload: bytes, *, label: str) -> dict[str, Any]:
    value = strict_json(payload, label=label)
    if payload != canonical_json_bytes(value):
        raise RunnerError(f"{label} is not canonical compact JSON")
    return value


def canonical_utc(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
        r"(?:\.[0-9]{3})?Z",
        value,
    ) is None:
        raise RunnerError(f"{label} is not canonical UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise RunnerError(f"{label} is not canonical UTC") from exc
    if parsed.tzinfo != timezone.utc:
        raise RunnerError(f"{label} is not canonical UTC")
    return parsed


def _askpass_entrypoint() -> bool:
    endpoint = os.environ.get("NUVION_E2E_ASKPASS_SOCKET")
    if not endpoint:
        return False
    # A failure propagates to the dedicated bottom-level handler and produces
    # a nonzero askpass exit without reading or printing the password.
    disable_core_dumps()
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(2.0)
            connection.connect(endpoint)
            payload = bytearray()
            while len(payload) <= MAX_SECRET_BYTES:
                chunk = connection.recv(MAX_SECRET_BYTES + 1 - len(payload))
                if not chunk:
                    break
                payload.extend(chunk)
        if not payload or len(payload) > MAX_SECRET_BYTES:
            return True
        sys.stdout.buffer.write(bytes(payload).rstrip(b"\r\n") + b"\n")
        sys.stdout.buffer.flush()
        for index in range(len(payload)):
            payload[index] = 0
    except OSError:
        return True
    return True


def utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def canonical_run_id(value: str) -> str:
    try:
        normalized = str(uuid.UUID(value))
    except (ValueError, AttributeError) as exc:
        raise RunnerError("runId must be a canonical UUIDv4") from exc
    if normalized != value or not RUN_ID_RE.fullmatch(normalized):
        raise RunnerError("runId must be a canonical UUIDv4")
    return normalized


def read_secret_fd(descriptor: int) -> bytearray:
    if descriptor < 0:
        raise RunnerError("credential FD must be non-negative")
    result = bytearray()
    while len(result) <= MAX_SECRET_BYTES:
        chunk = os.read(descriptor, min(1024, MAX_SECRET_BYTES + 1 - len(result)))
        if not chunk:
            break
        result.extend(chunk)
    result = bytearray(result.rstrip(b"\r\n"))
    if not result or len(result) > MAX_SECRET_BYTES or b"\x00" in result:
        raise RunnerError("credential FD is empty or invalid")
    return result


def zero_secret(value: bytearray | None) -> None:
    if value is not None:
        for index in range(len(value)):
            value[index] = 0


def read_regular(path: Path, maximum: int) -> bytes:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RunnerError(f"unsafe regular file: {path.name}")
    if metadata.st_size > maximum:
        raise RunnerError(f"file exceeds size limit: {path.name}")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise RunnerError(f"file changed while opening: {path.name}")
        result = bytearray()
        while len(result) <= maximum:
            chunk = os.read(descriptor, min(64 * 1024, maximum + 1 - len(result)))
            if not chunk:
                break
            result.extend(chunk)
        after = os.fstat(descriptor)
        if len(result) > maximum or (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ) != (opened.st_dev, opened.st_ino, opened.st_size):
            raise RunnerError(f"file changed or exceeded limit: {path.name}")
        return bytes(result)
    finally:
        os.close(descriptor)


def sha256_regular(path: Path, maximum: int) -> str:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RunnerError(f"unsafe regular file: {path.name}")
    if metadata.st_size > maximum:
        raise RunnerError(f"file exceeds size limit: {path.name}")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    digest = hashlib.sha256()
    total = 0
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
        ):
            raise RunnerError(f"file changed while opening: {path.name}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise RunnerError(f"file exceeds size limit: {path.name}")
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino, after.st_size) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
        ):
            raise RunnerError(f"file changed while hashing: {path.name}")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def assert_no_secret_material(value: object) -> None:
    def visit(item: object) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if str(key) in SAFE_SECRET_LIKE_FIELDS and not isinstance(
                    child, bool
                ):
                    raise RunnerError("output contains an invalid safe evidence field")
                if (
                    str(key) not in SAFE_SECRET_LIKE_FIELDS
                    and SECRET_KEY_RE.search(str(key))
                ):
                    raise RunnerError("output contains a forbidden field")
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
        elif isinstance(item, str) and any(
            pattern.search(item) for pattern in SECRET_VALUE_RES
        ):
            raise RunnerError("output contains secret or session material")

    visit(value)


def atomic_json(
    path: Path, value: Mapping[str, Any], *, immutable: bool = False
) -> None:
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    parent = path.parent.lstat()
    if stat.S_ISLNK(parent.st_mode) or not stat.S_ISDIR(parent.st_mode):
        raise RunnerError("output directory is unsafe")
    if path.exists() or path.is_symlink():
        existing = read_regular(path, MAX_OUTPUT_BYTES)
        if immutable:
            if existing != payload:
                raise RunnerError(
                    "immutable run artifact already exists with other bytes"
                )
            return
        metadata = path.lstat()
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            raise RunnerError("existing run artifact mode is unsafe")
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
                raise RunnerError("atomic JSON write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o600)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def prepare_output_dir(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise RunnerError("run output directory must not be a symlink")
    output = candidate.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = output.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RunnerError("run output directory is unsafe")
    if metadata.st_uid != os.getuid():
        raise RunnerError("run output directory is not owned by the caller")
    os.chmod(output, 0o700)
    return output


def known_host_entry(
    known_hosts: str | Path, *, host: str, port: int, expected: str
) -> tuple[str, str]:
    if not HOST_RE.fullmatch(host) or not 1 <= port <= 65535:
        raise RunnerError("SSH host or port is invalid")
    if not FINGERPRINT_RE.fullmatch(expected):
        raise RunnerError("expected SSH fingerprint is invalid")
    payload = read_regular(Path(known_hosts), 1024 * 1024)
    token = host if port == 22 else f"[{host}]:{port}"
    matches: list[tuple[str, str]] = []
    for raw_line in payload.decode("utf-8", errors="strict").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if fields[0].startswith("@") or len(fields) < 3:
            continue
        patterns, _key_type, encoded = fields[:3]
        host_patterns = patterns.split(",")
        if token not in host_patterns:
            continue
        if any(
            item.startswith("|") or "*" in item or "?" in item for item in host_patterns
        ):
            raise RunnerError("hashed or wildcard board host entries are not accepted")
        try:
            key_blob = base64.b64decode(encoded, validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise RunnerError("known_hosts key is invalid base64") from exc
        fingerprint = "SHA256:" + base64.b64encode(
            hashlib.sha256(key_blob).digest()
        ).decode("ascii").rstrip("=")
        if fingerprint == expected:
            matches.append((line, fingerprint))
    if len(matches) != 1:
        raise RunnerError("known_hosts does not contain exactly one pinned board key")
    return matches[0]


def create_pinned_known_hosts(
    source: str | Path,
    destination: Path,
    *,
    host: str,
    port: int,
    expected: str,
) -> str:
    line, fingerprint = known_host_entry(
        source, host=host, port=port, expected=expected
    )
    if destination.exists():
        metadata = destination.lstat()
        if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o600:
            raise RunnerError("run-pinned known_hosts ownership or mode is unsafe")
        payload = read_regular(destination, 16 * 1024)
        if payload != (line + "\n").encode("utf-8"):
            raise RunnerError("run-pinned known_hosts bytes changed")
        return fingerprint
    descriptor = os.open(
        destination,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.write(descriptor, (line + "\n").encode("utf-8"))
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o600)
    finally:
        os.close(descriptor)
    return fingerprint


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: bytes = b""
    stderr: bytes = b""


class LocalProcessRunner:
    def run(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        input_bytes: bytes | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ProcessResult:
        if isinstance(argv, (str, bytes)) or not argv:
            raise TypeError("argv must be a non-empty sequence")
        process = subprocess.Popen(
            list(argv),
            stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(env) if env is not None else None,
            start_new_session=True,
        )
        if process.stdout is None or process.stderr is None:
            raise RunnerError("bounded subprocess pipes are unavailable")
        stdout = bytearray()
        stderr = bytearray()
        output_limit = threading.Event()
        drain_errors: list[BaseException] = []
        kill_lock = threading.Lock()

        def kill_process_group() -> None:
            with kill_lock:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except OSError:
                    try:
                        process.kill()
                    except OSError:
                        pass

        def drain(stream: Any, destination: bytearray, maximum: int) -> None:
            try:
                total = 0
                while True:
                    chunk = stream.read(64 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    remaining = maximum - len(destination)
                    if remaining > 0:
                        destination.extend(chunk[:remaining])
                    if total > maximum and not output_limit.is_set():
                        output_limit.set()
                        kill_process_group()
            except BaseException as exc:
                drain_errors.append(exc)
                kill_process_group()
            finally:
                stream.close()

        readers = [
            threading.Thread(
                target=drain,
                args=(process.stdout, stdout, MAX_OUTPUT_BYTES),
                daemon=True,
            ),
            threading.Thread(
                target=drain,
                args=(process.stderr, stderr, 32 * 1024),
                daemon=True,
            ),
        ]
        for reader in readers:
            reader.start()
        if process.stdin is not None:
            try:
                process.stdin.write(input_bytes or b"")
                process.stdin.flush()
            except BrokenPipeError:
                pass
            finally:
                process.stdin.close()
        timed_out = False
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            kill_process_group()
            returncode = process.wait(timeout=10)
        for reader in readers:
            reader.join(timeout=10)
        if any(reader.is_alive() for reader in readers):
            kill_process_group()
            for stream in (process.stdout, process.stderr):
                try:
                    stream.close()
                except OSError:
                    pass
            for reader in readers:
                reader.join(timeout=1)
        if timed_out:
            raise RunnerError("bounded SSH operation timed out")
        if output_limit.is_set():
            raise RunnerError("bounded SSH operation exceeded output limit")
        if drain_errors or any(reader.is_alive() for reader in readers):
            raise RunnerError("bounded SSH output drain failed")
        return ProcessResult(
            returncode,
            bytes(stdout),
            bytes(stderr),
        )


class _OneShotAskpass:
    def __init__(self, secret: bytearray, directory: str | Path) -> None:
        self.secret = secret
        self.path = Path(directory) / "askpass.sock"
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.settimeout(10.0)
        self.server.bind(str(self.path))
        os.chmod(self.path, 0o600)
        self.server.listen(1)
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._serve, daemon=True)

    def _serve(self) -> None:
        try:
            connection, _ = self.server.accept()
            with connection:
                connection.sendall(bytes(self.secret))
        except BaseException as exc:  # noqa: BLE001 - surfaced to the caller.
            self.error = exc
        finally:
            self.server.close()
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass

    def start(self) -> None:
        self.thread.start()

    def finish(self) -> None:
        self.thread.join(timeout=12)
        if self.thread.is_alive():
            self.server.close()
            raise RunnerError("SSH askpass did not consume its one allowed prompt")
        if self.error is not None:
            raise RunnerError(
                "SSH askpass failed before authentication"
            ) from self.error


class OpenSshTransport:
    def __init__(
        self,
        *,
        host: str,
        user: str,
        port: int,
        pinned_known_hosts: str | Path,
        expected_fingerprint: str,
        ssh_password: bytearray | None = None,
        sudo_password: bytearray | None = None,
        process_runner: LocalProcessRunner | None = None,
        askpass_program: str | Path | None = None,
    ) -> None:
        if not HOST_RE.fullmatch(host) or not USER_RE.fullmatch(user):
            raise RunnerError("SSH host or username is invalid")
        line, fingerprint = known_host_entry(
            pinned_known_hosts, host=host, port=port, expected=expected_fingerprint
        )
        del line
        self.host = host
        self.user = user
        self.port = port
        self.known_hosts = Path(pinned_known_hosts)
        self.fingerprint = fingerprint
        self.ssh_password = ssh_password
        self.sudo_password = sudo_password
        self.process_runner = process_runner or LocalProcessRunner()
        self.askpass_program = Path(askpass_program or __file__).resolve()
        self.base_options = [
            "-F",
            "/dev/null",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            f"UserKnownHostsFile={self.known_hosts}",
            "-o",
            "GlobalKnownHostsFile=/dev/null",
            "-o",
            # Keep the run-pinned hostname entry immutable. OpenSSH otherwise
            # appends the resolved Tailscale IP to UserKnownHostsFile.
            "CheckHostIP=no",
            "-o",
            "UpdateHostKeys=no",
            "-o",
            "NumberOfPasswordPrompts=1",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ServerAliveCountMax=2",
            "-o",
            "ControlMaster=no",
            "-o",
            "ControlPath=none",
            "-o",
            "ControlPersist=no",
            "-o",
            "ProxyCommand=none",
            "-o",
            "ProxyJump=none",
            "-o",
            "ForwardAgent=no",
            "-o",
            "ClearAllForwardings=yes",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "PermitLocalCommand=no",
            "-o",
            "RequestTTY=no",
            "-o",
            "LogLevel=ERROR",
        ]
        if ssh_password is None:
            self.base_options.extend(["-o", "BatchMode=yes"])
        else:
            self.base_options.extend(
                [
                    "-o",
                    "BatchMode=no",
                    "-o",
                    "PubkeyAuthentication=no",
                    "-o",
                    "HostbasedAuthentication=no",
                    "-o",
                    "GSSAPIAuthentication=no",
                    "-o",
                    "KbdInteractiveAuthentication=no",
                    "-o",
                    "ChallengeResponseAuthentication=no",
                    "-o",
                    "PasswordAuthentication=yes",
                    "-o",
                    "PreferredAuthentications=password",
                ]
            )

    @staticmethod
    def _parse_result(result: ProcessResult, *, operation: str) -> dict[str, Any]:
        if result.returncode != 0:
            if operation == "out-of-band-updater-bootstrap":
                stderr = result.stderr.decode("utf-8", errors="replace")
                for line in reversed(stderr.splitlines()):
                    detail = line.strip()
                    if BOOTSTRAP_FAILURE_RE.fullmatch(detail):
                        raise RunnerError(detail)
            raise RunnerError(f"remote operation failed: {operation}")
        lines = [
            line
            for line in result.stdout.decode("utf-8", errors="strict").splitlines()
            if line
        ]
        if not lines:
            raise RunnerError(f"remote operation returned no JSON: {operation}")
        payload = strict_json(lines[-1], label=f"remote operation {operation}")
        assert_no_secret_material(payload)
        return payload

    def _run_with_auth(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        input_bytes: bytes | None = None,
    ) -> ProcessResult:
        if self.ssh_password is None:
            return self.process_runner.run(
                argv, timeout=timeout, input_bytes=input_bytes
            )
        metadata = self.askpass_program.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise RunnerError("SSH askpass program is unsafe")
        with tempfile.TemporaryDirectory(prefix="nuvion-e2e-askpass-") as directory:
            os.chmod(directory, 0o700)
            broker = _OneShotAskpass(self.ssh_password, directory)
            broker.start()
            environment = os.environ.copy()
            environment.update(
                {
                    "DISPLAY": "nuvion-e2e:0",
                    "SSH_ASKPASS": str(self.askpass_program),
                    "SSH_ASKPASS_REQUIRE": "force",
                    "NUVION_E2E_ASKPASS_SOCKET": str(broker.path),
                }
            )
            try:
                result = self.process_runner.run(
                    argv,
                    timeout=timeout,
                    input_bytes=input_bytes,
                    env=environment,
                )
            finally:
                broker.finish()
            return result

    def invoke_board(
        self,
        command: str,
        arguments: Sequence[str] = (),
        *,
        timeout: float = 90,
    ) -> dict[str, Any]:
        if command not in BOARD_COMMANDS:
            raise RunnerError("board primitive is outside the typed allowlist")
        if any(
            "\x00" in value or "\n" in value or "\r" in value for value in arguments
        ):
            raise RunnerError("board primitive argument contains a control character")
        remote = ["/usr/bin/python3", "-I", REMOTE_TOOL, command, *arguments]
        if self.sudo_password is None:
            remote = ["/usr/bin/sudo", "-n", "--", *remote]
            input_bytes = None
        else:
            remote = ["/usr/bin/sudo", "-S", "-p", "", "--", *remote]
            input_bytes = bytes(self.sudo_password) + b"\n"
        result = self._run_with_auth(
            [
                "/usr/bin/ssh",
                *self.base_options,
                "-p",
                str(self.port),
                f"{self.user}@{self.host}",
                shlex.join(remote),
            ],
            timeout=timeout,
            input_bytes=input_bytes,
        )
        return self._parse_result(result, operation=command)

    def _invoke_root_python(
        self,
        program: str,
        arguments: Sequence[str],
        *,
        operation: str,
        timeout: float,
    ) -> dict[str, Any]:
        if any(
            "\x00" in value or "\n" in value or "\r" in value for value in arguments
        ):
            raise RunnerError("root operation argument contains a control character")
        remote = ["/usr/bin/python3", "-I", "-c", program, *arguments]
        if self.sudo_password is None:
            remote = ["/usr/bin/sudo", "-n", "--", *remote]
            input_bytes = None
        else:
            remote = ["/usr/bin/sudo", "-S", "-p", "", "--", *remote]
            input_bytes = bytes(self.sudo_password) + b"\n"
        result = self._run_with_auth(
            [
                "/usr/bin/ssh",
                *self.base_options,
                "-p",
                str(self.port),
                f"{self.user}@{self.host}",
                shlex.join(remote),
            ],
            timeout=timeout,
            input_bytes=input_bytes,
        )
        return self._parse_result(result, operation=operation)

    def copy_input(self, source: Path, *, run_id: str, role: str) -> str:
        canonical_run_id(run_id)
        if role not in {"command", "release", "health", "binding", "manifest"}:
            raise RunnerError("staging role is invalid")
        destination = f"/tmp/nuvion-fleet-e2e-{run_id}-{role}.json"
        result = self._run_with_auth(
            [
                "/usr/bin/scp",
                *self.base_options,
                "-P",
                str(self.port),
                "--",
                str(source),
                f"{self.user}@{self.host}:{destination}",
            ],
            timeout=30,
        )
        if result.returncode != 0:
            raise RunnerError(f"staging upload failed: {role}")
        return destination

    def copy_bootstrap_artifact(self, source: Path, *, run_id: str, role: str) -> str:
        canonical_run_id(run_id)
        if role == "installer":
            destination = REMOTE_BOOTSTRAP_INSTALLER.format(run_id=run_id)
        elif role == "package":
            destination = REMOTE_BOOTSTRAP_DEB.format(run_id=run_id)
        else:
            raise RunnerError("bootstrap staging role is invalid")
        result = self._run_with_auth(
            [
                "/usr/bin/scp",
                *self.base_options,
                "-P",
                str(self.port),
                "--",
                str(source),
                f"{self.user}@{self.host}:{destination}",
            ],
            timeout=180,
        )
        if result.returncode != 0:
            raise RunnerError(f"bootstrap staging upload failed: {role}")
        return destination

    def copy_candidate_input(self, source: Path, *, run_id: str, role: str) -> str:
        canonical_run_id(run_id)
        suffixes = {
            "candidate-bundle": "candidate-bundle.tar.gz",
            "candidate-bom": "candidate-bom.json",
            "oak-harness": "oak-harness.sh",
        }
        if role not in suffixes:
            raise RunnerError("candidate staging role is invalid")
        destination = f"/tmp/nuvion-fleet-e2e-{run_id}-{suffixes[role]}"
        result = self._run_with_auth(
            [
                "/usr/bin/scp",
                *self.base_options,
                "-P",
                str(self.port),
                "--",
                str(source),
                f"{self.user}@{self.host}:{destination}",
            ],
            timeout=300 if role == "candidate-bundle" else 30,
        )
        if result.returncode != 0:
            raise RunnerError(f"candidate staging upload failed: {role}")
        return destination

    def bootstrap_updater(
        self,
        *,
        run_id: str,
        installer_path: str,
        package_path: str,
        installer_sha256: str,
        package_sha256: str,
        expected_version: str,
    ) -> dict[str, Any]:
        canonical_run_id(run_id)
        if installer_path != REMOTE_BOOTSTRAP_INSTALLER.format(
            run_id=run_id
        ) or package_path != REMOTE_BOOTSTRAP_DEB.format(run_id=run_id):
            raise RunnerError("bootstrap staging paths are not fixed")
        if not SHA256_RE.fullmatch(installer_sha256) or not SHA256_RE.fullmatch(
            package_sha256
        ):
            raise RunnerError("bootstrap digest is invalid")
        if not SEMVER_RE.fullmatch(expected_version):
            raise RunnerError("bootstrap version is invalid")
        return self._invoke_root_python(
            BOOTSTRAP_REMOTE_PROGRAM,
            [
                run_id,
                installer_path,
                package_path,
                installer_sha256,
                package_sha256,
                expected_version,
                self.user,
            ],
            operation="out-of-band-updater-bootstrap",
            timeout=1900,
        )

    def discard_bootstrap_staging(self, *, run_id: str) -> dict[str, Any]:
        canonical_run_id(run_id)
        return self._invoke_root_python(
            BOOTSTRAP_CLEANUP_PROGRAM,
            [
                run_id,
                REMOTE_BOOTSTRAP_INSTALLER.format(run_id=run_id),
                REMOTE_BOOTSTRAP_DEB.format(run_id=run_id),
            ],
            operation="bootstrap-staging-cleanup",
            timeout=60,
        )


class HostJournal:
    def __init__(self, path: Path, *, run_id: str, host: str, fingerprint: str) -> None:
        self.path = path
        self.run_id = run_id
        if path.exists():
            value = strict_json(
                read_regular(path, MAX_OUTPUT_BYTES), label="host journal"
            )
            if (
                set(value)
                != {
                    "schemaVersion",
                    "runId",
                    "host",
                    "hostKeyFingerprint",
                    "createdAt",
                    "steps",
                }
                or not isinstance(value.get("steps"), dict)
                or (
                    value.get("schemaVersion"),
                    value.get("runId"),
                    value.get("host"),
                    value.get("hostKeyFingerprint"),
                )
                != (1, run_id, host, fingerprint)
            ):
                raise RunnerError("host journal identity mismatch")
            for step, raw in value["steps"].items():
                if (
                    not isinstance(step, str)
                    or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", step) is None
                    or not isinstance(raw, dict)
                    or set(raw) != {"status", "attempt", "updatedAt"}
                    or raw.get("status") not in {"RUNNING", "COMPLETE", "FAILED"}
                    or isinstance(raw.get("attempt"), bool)
                    or not isinstance(raw.get("attempt"), int)
                    or raw.get("attempt") < 0
                    or not isinstance(raw.get("updatedAt"), str)
                ):
                    raise RunnerError("host journal step is invalid")
            self.state = value
        else:
            self.state = {
                "schemaVersion": 1,
                "runId": run_id,
                "host": host,
                "hostKeyFingerprint": fingerprint,
                "createdAt": utc_now(),
                "steps": {},
            }
            atomic_json(path, self.state)

    def mark(self, step: str, status: str) -> None:
        if re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", step) is None or status not in {
            "RUNNING",
            "COMPLETE",
            "FAILED",
        }:
            raise RunnerError("journal status is invalid")
        current = self.state["steps"].get(step, {})
        attempt = int(current.get("attempt", 0)) + (1 if status == "RUNNING" else 0)
        self.state["steps"][step] = {
            "status": status,
            "attempt": attempt,
            "updatedAt": utc_now(),
        }
        atomic_json(self.path, self.state)


def validate_paths_distinct(output_dir: Path, inputs: Sequence[Path]) -> None:
    output_paths = {
        (output_dir / name).resolve(strict=False)
        for name in (
            "journal.json",
            "immutable-manifest.json",
            "evidence.json",
            "bootstrap-evidence.json",
            "candidate-soak-raw.json",
            "candidate-soak-evidence.json",
            "cleanup-evidence.json",
            "known_hosts",
        )
    }
    resolved_inputs = {path.resolve() for path in inputs}
    if len(resolved_inputs) != len(inputs):
        raise RunnerError("trust input paths must be distinct")
    identities: set[tuple[int, int]] = set()
    for path in inputs:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise RunnerError("run input must be a regular non-symlink file")
        identity = (metadata.st_dev, metadata.st_ino)
        if identity in identities:
            raise RunnerError("run input files must have distinct inodes")
        identities.add(identity)
    if output_paths & resolved_inputs:
        raise RunnerError("trust inputs and run outputs must be distinct")


def build_manifest(
    *,
    run_id: str,
    tool_sha256: str,
    input_digests: Mapping[str, str],
    identity: Mapping[str, object],
    scenario_type: str,
    expected_command_id: str,
    expected_bom_digest: str,
    expected_candidate_slot: str,
    expected_previous_slot: str,
    expected_previous_version: str,
    hold_seconds: int,
    release: Mapping[str, object],
) -> dict[str, object]:
    canonical_run_id(run_id)
    try:
        command_id = str(uuid.UUID(expected_command_id))
    except ValueError as exc:
        raise RunnerError("expected commandId is invalid") from exc
    if command_id != expected_command_id:
        raise RunnerError("expected commandId is not canonical")
    if not SHA256_RE.fullmatch(tool_sha256) or set(input_digests) != {
        "commandSha256",
        "releaseSha256",
        "healthSha256",
        "bindingSha256",
    }:
        raise RunnerError("tool/input digest set is invalid")
    if not all(SHA256_RE.fullmatch(item) for item in input_digests.values()):
        raise RunnerError("input digest is invalid")
    if scenario_type not in {"commit", "oak-fault-rollback"}:
        raise RunnerError("scenario type is invalid")
    if (
        type(hold_seconds) is not int
        or (scenario_type == "commit" and hold_seconds != 0)
        or (scenario_type == "oak-fault-rollback" and not 0 <= hold_seconds <= 60)
    ):
        raise RunnerError("scenario hold is invalid")
    if not DIGEST_RE.fullmatch(expected_bom_digest):
        raise RunnerError("expected BOM digest is invalid")
    if expected_candidate_slot != f"/opt/nuv-agent/releases/{expected_bom_digest[7:]}":
        raise RunnerError("expected candidate slot is not BOM-addressed")
    if (
        re.fullmatch(
            r"(?:releases/[0-9a-f]{64}|bootstrap/[0-9A-Za-z.+-]{1,64})",
            expected_previous_slot,
        )
        is None
    ):
        raise RunnerError("expected previous slot is invalid")
    if not SEMVER_RE.fullmatch(expected_previous_version):
        raise RunnerError("expected previous Agent version is invalid")
    expected_identity = {
        "productModel": "IQ9075_DEV",
        "platformProfile": "iq9075_dev",
        "hardwareRevision": "QCS9075-EVK",
        "architecture": "aarch64",
        "dockerRequired": False,
    }
    device_match = DEVICE_ID_RE.fullmatch(str(identity.get("deviceId") or ""))
    if (
        set(identity) != {"deviceId", "spaceId", *expected_identity}
        or isinstance(identity.get("spaceId"), bool)
        or not isinstance(identity.get("spaceId"), int)
        or device_match is None
        or int(device_match.group(1)) != identity.get("spaceId")
        or identity.get("dockerRequired") is not False
        or any(identity.get(key) != value for key, value in expected_identity.items())
    ):
        raise RunnerError("IQ9075 identity tuple is invalid")
    if set(release) != {
        "agentVersion",
        "releaseSequence",
        "artifactDigest",
        "componentSha",
        "configSchema",
        "publisherKeyId",
    }:
        raise RunnerError("release identity fields are invalid")
    if (
        not isinstance(release["agentVersion"], str)
        or not SEMVER_RE.fullmatch(release["agentVersion"])
        or isinstance(release["releaseSequence"], bool)
        or not isinstance(release["releaseSequence"], int)
        or int(release["releaseSequence"]) < 1
        or not isinstance(release["artifactDigest"], str)
        or not DIGEST_RE.fullmatch(release["artifactDigest"])
        or not isinstance(release["componentSha"], str)
        or not COMPONENT_RE.fullmatch(release["componentSha"])
        or not isinstance(release["configSchema"], str)
        or re.fullmatch(r"[1-9][0-9]*", release["configSchema"]) is None
        or not isinstance(release["publisherKeyId"], str)
        or KEY_ID_RE.fullmatch(release["publisherKeyId"]) is None
    ):
        raise RunnerError("release identity value is invalid")
    return {
        "schemaVersion": 1,
        "protocolVersion": PROTOCOL_VERSION,
        "runId": run_id,
        "toolSha256": tool_sha256,
        "inputs": dict(input_digests),
        "destinations": FIXED_DESTINATIONS,
        "identity": dict(identity),
        "scenario": {
            "type": scenario_type,
            "expectedCommandId": expected_command_id,
            "expectedBomDigest": expected_bom_digest,
            "expectedCandidateSlot": expected_candidate_slot,
            "expectedPreviousSlot": expected_previous_slot,
            "expectedPreviousVersion": expected_previous_version,
            "holdSeconds": hold_seconds,
            "release": dict(release),
        },
    }


def validate_manifest(manifest: Mapping[str, Any]) -> dict[str, object]:
    """Rebuild and exact-compare an externally supplied immutable manifest."""

    if set(manifest) != {
        "schemaVersion",
        "protocolVersion",
        "runId",
        "toolSha256",
        "inputs",
        "destinations",
        "identity",
        "scenario",
    }:
        raise RunnerError("immutable manifest root fields are invalid")
    scenario = manifest.get("scenario")
    if not isinstance(scenario, Mapping) or set(scenario) != {
        "type",
        "expectedCommandId",
        "expectedBomDigest",
        "expectedCandidateSlot",
        "expectedPreviousSlot",
        "expectedPreviousVersion",
        "holdSeconds",
        "release",
    }:
        raise RunnerError("immutable manifest scenario fields are invalid")
    identity = manifest.get("identity")
    inputs = manifest.get("inputs")
    release = scenario.get("release")
    if (
        type(manifest.get("schemaVersion")) is not int
        or manifest.get("schemaVersion") != 1
        or manifest.get("protocolVersion") != PROTOCOL_VERSION
        or manifest.get("destinations") != FIXED_DESTINATIONS
        or not isinstance(identity, Mapping)
        or not isinstance(inputs, Mapping)
        or not isinstance(release, Mapping)
    ):
        raise RunnerError("immutable manifest envelope is invalid")
    rebuilt = build_manifest(
        run_id=str(manifest.get("runId") or ""),
        tool_sha256=str(manifest.get("toolSha256") or ""),
        input_digests=inputs,
        identity=identity,
        scenario_type=str(scenario.get("type") or ""),
        expected_command_id=str(scenario.get("expectedCommandId") or ""),
        expected_bom_digest=str(scenario.get("expectedBomDigest") or ""),
        expected_candidate_slot=str(scenario.get("expectedCandidateSlot") or ""),
        expected_previous_slot=str(scenario.get("expectedPreviousSlot") or ""),
        expected_previous_version=str(
            scenario.get("expectedPreviousVersion") or ""
        ),
        hold_seconds=scenario.get("holdSeconds"),
        release=release,
    )
    if dict(manifest) != rebuilt:
        raise RunnerError("immutable manifest is not canonical")
    return rebuilt


def validate_candidate_inputs(
    *,
    bom_payload: bytes,
    bundle_sha256: str,
    bundle_size: int,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    bom = strict_json(bom_payload, label="candidate BOM")
    expected_fields = {
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
    scenario = manifest.get("scenario")
    release = scenario.get("release") if isinstance(scenario, Mapping) else None
    artifact = bom.get("artifact")
    unsigned = dict(bom)
    unsigned.pop("bomDigest", None)
    canonical_digest = "sha256:" + hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    required_target = {
        "productModel": "IQ9075_DEV",
        "platformProfile": "iq9075_dev",
        "hardwareRevision": "QCS9075-EVK",
        "architecture": "aarch64",
    }
    if (
        set(bom) != expected_fields
        or type(bom.get("schemaVersion")) is not int
        or bom.get("schemaVersion") != 2
        or not isinstance(scenario, Mapping)
        or scenario.get("type") != "oak-fault-rollback"
        or not isinstance(release, Mapping)
        or bom.get("bomDigest") != canonical_digest
        or bom.get("bomDigest") != scenario.get("expectedBomDigest")
        or bom.get("agentVersion") != release.get("agentVersion")
        or bom.get("releaseSequence") != release.get("releaseSequence")
        or bom.get("componentSha") != release.get("componentSha")
        or bom.get("configSchema") != release.get("configSchema")
        or not isinstance(bom.get("targets"), list)
        or required_target not in bom["targets"]
        or not isinstance(artifact, Mapping)
        or set(artifact) != {"name", "kind", "sha256", "sizeBytes"}
        or artifact.get("kind") != "agent-bundle"
        or artifact.get("sha256") != bundle_sha256
        or artifact.get("sizeBytes") != bundle_size
        or release.get("artifactDigest") != f"sha256:{bundle_sha256}"
    ):
        raise RunnerError("candidate bundle/BOM differs from signed rollback")
    return bom


def validate_candidate_soak_evidence(
    evidence: Mapping[str, Any],
    *,
    run_id: str,
    manifest: Mapping[str, Any],
    bundle_sha256: str,
    bom_sha256: str,
    harness_sha256: str,
    fleet_evidence_sha256: str,
    raw_evidence_sha256: str | None = None,
    cleanup_evidence_sha256: str | None = None,
    require_cleanup_evidence: bool = False,
    manifest_raw: bytes | None = None,
    fleet_evidence_raw: bytes | None = None,
) -> None:
    def timestamp(value: object, *, label: str) -> datetime:
        if not isinstance(value, str) or re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
            r"(?:\.[0-9]{3})?Z",
            value,
        ) is None:
            raise RunnerError(f"candidate soak {label} is not canonical UTC")
        try:
            parsed = datetime.fromisoformat(value[:-1] + "+00:00")
        except ValueError as exc:
            raise RunnerError(
                f"candidate soak {label} is not canonical UTC"
            ) from exc
        if parsed.tzinfo != timezone.utc:
            raise RunnerError(f"candidate soak {label} is not canonical UTC")
        return parsed

    expected_root = {
        "schemaVersion",
        "kind",
        "protocolVersion",
        "runId",
        "startedAt",
        "completedAt",
        "complete",
        "outcome",
        "candidate",
        "fleetEvidenceSha256",
        "rawEvidenceSha256",
        "rawEvidence",
        "executionProof",
        "collectorProof",
        "terminationProof",
        "productionRestoration",
        "pre",
        "post",
        "gates",
    }
    if require_cleanup_evidence:
        expected_root.update({"cleanupEvidenceSha256", "cleanupEvidence"})
    scenario = manifest["scenario"]
    expected_slot = (
        f"/opt/nuv-agent/candidates/{run_id}-{scenario['expectedBomDigest'][7:]}"
    )
    candidate = evidence.get("candidate")
    outcome = evidence.get("outcome")
    gates = evidence.get("gates")
    pre = evidence.get("pre")
    post = evidence.get("post")
    expected_slots = {
        "current": scenario["expectedPreviousSlot"],
        "previous": "releases/" + scenario["expectedBomDigest"][7:],
    }
    if (
        set(evidence) != expected_root
        or evidence.get("schemaVersion") != 1
        or evidence.get("kind") != "nuvion-iq9075-candidate-soak-evidence"
        or evidence.get("protocolVersion") != PROTOCOL_VERSION
        or evidence.get("runId") != run_id
        or evidence.get("complete") is not True
        or not isinstance(candidate, Mapping)
        or set(candidate)
        != {
            "slotKind",
            "slot",
            "bomDigest",
            "bundleSha256",
            "bomSha256",
            "harnessSha256",
            "controlMarkerSha256",
        }
        or candidate.get("slotKind") != "candidate"
        or candidate.get("slot") != expected_slot
        or candidate.get("bomDigest") != scenario["expectedBomDigest"]
        or candidate.get("bundleSha256") != bundle_sha256
        or candidate.get("bomSha256") != bom_sha256
        or candidate.get("harnessSha256") != harness_sha256
        or not isinstance(candidate.get("controlMarkerSha256"), str)
        or SHA256_RE.fullmatch(candidate["controlMarkerSha256"]) is None
        or evidence.get("fleetEvidenceSha256") != fleet_evidence_sha256
        or not isinstance(outcome, Mapping)
        or set(outcome) != {"status", "errorCode"}
        or outcome.get("status") not in {"passed", "failed"}
        or not isinstance(gates, Mapping)
        or set(gates)
        != {
            "signedRollbackTerminal",
            "candidateBound",
            "rawEvidencePreserved",
            "slotsUnchanged",
            "releaseTreesUnchanged",
            "antiReplayUnchanged",
            "oakIdentityUnchanged",
            "freshBaselineProcess",
            "harnessBytesPinned",
            "harnessCopyRemoved",
            "resourceLimitsApplied",
            "boundedOutput",
            "persistentStateReadOnly",
            "persistentStateUnchanged",
            "productionTrustRestored",
            "trustedSoakDuration",
            "continuousUidIsolation",
            "cgroupTerminated",
            "harnessPassed",
        }
        or not isinstance(pre, Mapping)
        or not isinstance(post, Mapping)
        or set(pre)
        != {
            "slots",
            "antiReplay",
            "oak",
            "runtime",
            "persistentState",
            "releaseTrees",
        }
        or set(post)
        != {
            "restoredAt",
            "slots",
            "antiReplay",
            "oak",
            "runtime",
            "persistentState",
            "releaseTrees",
        }
    ):
        raise RunnerError("candidate soak evidence schema is invalid")
    if require_cleanup_evidence:
        cleanup_evidence = evidence.get("cleanupEvidence")
        if not isinstance(cleanup_evidence, Mapping):
            raise RunnerError("candidate soak cleanup evidence is missing")
        if cleanup_evidence.get("schemaVersion") == 2:
            if manifest_raw is None or fleet_evidence_raw is None:
                raise RunnerError("candidate soak cleanup binding inputs are missing")
            validate_bound_cleanup_evidence(
                cleanup_evidence,
                run_id=run_id,
                manifest_raw=manifest_raw,
                fleet_evidence_raw=fleet_evidence_raw,
            )
        else:
            # Historical candidate-soak evidence used the normalized board
            # receipt directly. It remains readable but is not accepted by the
            # camera-independent Fleet Runtime release gate.
            validate_cleanup_result(cleanup_evidence, run_id=run_id)
        cleanup_proof = cleanup_evidence.get("proof")
        cleanup_payload = canonical_json_bytes(cleanup_evidence)
        actual_cleanup_sha256 = hashlib.sha256(cleanup_payload).hexdigest()
        if (
            cleanup_evidence.get("phase") != "RESTORED"
            or not isinstance(cleanup_proof, Mapping)
            or cleanup_proof.get("transactionPhase") != "RESTORED"
            or evidence.get("cleanupEvidenceSha256") != actual_cleanup_sha256
            or (
                cleanup_evidence_sha256 is not None
                and cleanup_evidence_sha256 != actual_cleanup_sha256
            )
        ):
            raise RunnerError("candidate soak cleanup evidence is not bound")
    passed = outcome["status"] == "passed"
    raw = evidence.get("rawEvidence")
    pre_runtime = pre["runtime"]
    post_runtime = post["runtime"]
    pre_oak = pre["oak"]
    post_oak = post["oak"]

    def valid_oak(value: object) -> bool:
        return bool(
            isinstance(value, Mapping)
            and set(value)
            == {
                "port",
                "vendorId",
                "productId",
                "speedMbps",
                "mxidSha256",
                "attached",
                "bound",
            }
            and isinstance(value.get("port"), str)
            and re.fullmatch(r"2-1(?:\.[1-9][0-9]*)+", value["port"])
            is not None
            and value.get("vendorId") == "03e7"
            and value.get("productId") == "f63b"
            and not isinstance(value.get("speedMbps"), bool)
            and isinstance(value.get("speedMbps"), int)
            and value["speedMbps"] >= 5000
            and isinstance(value.get("mxidSha256"), str)
            and SHA256_RE.fullmatch(value["mxidSha256"]) is not None
            and value.get("attached") is True
            and value.get("bound") is True
        )

    def valid_persistent_state(value: object) -> bool:
        if not isinstance(value, Mapping) or set(value) != {
            "schemaVersion",
            "roots",
            "sha256",
            "entries",
            "bytes",
        }:
            return False
        roots = value.get("roots")
        if (
            value.get("schemaVersion") != 1
            or isinstance(value.get("schemaVersion"), bool)
            or not isinstance(roots, Mapping)
            or set(roots) != set(CANDIDATE_PERSISTENT_PATHS)
            or not isinstance(value.get("sha256"), str)
            or SHA256_RE.fullmatch(value["sha256"]) is None
        ):
            return False
        total_entries = 0
        total_bytes = 0
        for root in roots.values():
            if (
                not isinstance(root, Mapping)
                or set(root) != {"exists", "entries", "bytes", "sha256"}
                or not isinstance(root.get("exists"), bool)
                or isinstance(root.get("entries"), bool)
                or not isinstance(root.get("entries"), int)
                or root["entries"] < 0
                or isinstance(root.get("bytes"), bool)
                or not isinstance(root.get("bytes"), int)
                or root["bytes"] < 0
                or not isinstance(root.get("sha256"), str)
                or SHA256_RE.fullmatch(root["sha256"]) is None
                or (root["exists"] is False and (root["entries"] or root["bytes"]))
            ):
                return False
            total_entries += root["entries"]
            total_bytes += root["bytes"]
        serialized = (
            json.dumps(
                dict(roots),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        return bool(
            value.get("entries") == total_entries
            and value.get("bytes") == total_bytes
            and hashlib.sha256(serialized).hexdigest() == value["sha256"]
        )

    def valid_release_trees(value: object) -> bool:
        if not isinstance(value, Mapping) or set(value) != {
            "schemaVersion",
            "slots",
            "sha256",
            "entries",
            "bytes",
        }:
            return False
        trees = value.get("slots")
        if (
            value.get("schemaVersion") != 1
            or not isinstance(trees, Mapping)
            or set(trees) != {"current", "previous"}
            or not isinstance(value.get("sha256"), str)
            or SHA256_RE.fullmatch(value["sha256"]) is None
        ):
            return False
        total_entries = 0
        total_bytes = 0
        for role, target in expected_slots.items():
            tree = trees.get(role)
            if (
                not isinstance(tree, Mapping)
                or set(tree)
                != {"target", "exists", "entries", "bytes", "sha256"}
                or tree.get("target") != target
                or tree.get("exists") is not True
                or isinstance(tree.get("entries"), bool)
                or not isinstance(tree.get("entries"), int)
                or tree["entries"] < 1
                or isinstance(tree.get("bytes"), bool)
                or not isinstance(tree.get("bytes"), int)
                or tree["bytes"] < 0
                or not isinstance(tree.get("sha256"), str)
                or SHA256_RE.fullmatch(tree["sha256"]) is None
            ):
                return False
            total_entries += tree["entries"]
            total_bytes += tree["bytes"]
        serialized = (
            json.dumps(
                dict(trees),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        return bool(
            value.get("entries") == total_entries
            and value.get("bytes") == total_bytes
            and hashlib.sha256(serialized).hexdigest() == value["sha256"]
        )

    def valid_execution_proof(value: object) -> bool:
        if not isinstance(value, Mapping) or set(value) != {
            "schemaVersion",
            "unit",
            "mainPid",
            "controlGroup",
            "pidControlGroup",
            "recursivePopulated",
            "uidIsolation",
            "systemdProperties",
            "mountSandbox",
        }:
            return False
        unit = f"nuvion-candidate-soak-{run_id.replace('-', '')}.service"
        control_group = "/system.slice/" + unit
        properties = value.get("systemdProperties")
        mounts = value.get("mountSandbox")
        uid_isolation = value.get("uidIsolation")
        if (
            value.get("schemaVersion") != 1
            or value.get("unit") != unit
            or value.get("controlGroup") != control_group
            or value.get("pidControlGroup") != control_group
            or value.get("recursivePopulated") is not True
            or type(value.get("mainPid")) is not int
            or value["mainPid"] < 2
            or not isinstance(properties, Mapping)
            or dict(properties) != CANDIDATE_SYSTEMD_EXPECTED_PROPERTIES
            or not isinstance(mounts, Mapping)
            or not isinstance(uid_isolation, Mapping)
            or set(uid_isolation) != {"before", "during"}
            or set(mounts)
            != {
                "temporaryFilesystems",
                "readOnlyPaths",
                "readWritePath",
                "inaccessiblePaths",
                "totalTmpfsBytes",
                "totalTmpfsInodes",
            }
        ):
            return False
        uid_before = uid_isolation.get("before")
        uid_during = uid_isolation.get("during")
        if (
            not isinstance(uid_before, Mapping)
            or not isinstance(uid_during, Mapping)
            or set(uid_before)
            != {"schemaVersion", "uid", "pids", "controlGroup", "stableScans"}
            or set(uid_during)
            != {"schemaVersion", "uid", "pids", "controlGroup", "stableScans"}
            or uid_before.get("schemaVersion") != 1
            or type(uid_before.get("uid")) is not int
            or uid_before["uid"] < 1
            or uid_before.get("pids") != []
            or uid_before.get("controlGroup") is not None
            or uid_before.get("stableScans") != 2
            or uid_during.get("schemaVersion") != 1
            or uid_during.get("uid") != uid_before.get("uid")
            or not isinstance(uid_during.get("pids"), list)
            or not uid_during["pids"]
            or any(type(pid) is not int or pid < 2 for pid in uid_during["pids"])
            or len(set(uid_during["pids"])) != len(uid_during["pids"])
            or uid_during.get("controlGroup") != control_group
            or uid_during.get("stableScans") != 2
        ):
            return False
        temporary = mounts.get("temporaryFilesystems")
        read_only = mounts.get("readOnlyPaths")
        inaccessible = mounts.get("inaccessiblePaths")
        writable = mounts.get("readWritePath")
        if (
            not isinstance(temporary, Mapping)
            or set(temporary) != set(CANDIDATE_TMPFS_LIMITS)
            or not isinstance(read_only, Mapping)
            or set(read_only) != set(CANDIDATE_PERSISTENT_PATHS)
            or not isinstance(inaccessible, Mapping)
            or set(inaccessible) != set(CANDIDATE_INACCESSIBLE_PATHS)
            or not isinstance(writable, Mapping)
            or set(writable) != {"mountId", "mountPoint", "readOnly"}
            or writable.get("mountPoint")
            != f"/var/lib/nuvion-fleet-e2e/runs/{run_id}"
            or writable.get("readOnly") is not False
        ):
            return False
        mount_id_points: dict[int, str] = {}

        def valid_mount_id(raw: object, mount_point: str) -> bool:
            if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
                return False
            previous = mount_id_points.get(raw)
            if previous is not None and previous != mount_point:
                return False
            mount_id_points[raw] = mount_point
            return True

        if not valid_mount_id(writable.get("mountId"), str(writable["mountPoint"])):
            return False
        total_bytes = 0
        total_inodes = 0
        for path, limits in CANDIDATE_TMPFS_LIMITS.items():
            mount = temporary.get(path)
            if (
                not isinstance(mount, Mapping)
                or set(mount)
                != {"mountId", "fsType", "sizeBytes", "inodeLimit", "readOnly"}
                or not valid_mount_id(mount.get("mountId"), path)
                or mount.get("fsType") != "tmpfs"
                or mount.get("readOnly") is not False
                or isinstance(mount.get("sizeBytes"), bool)
                or not isinstance(mount.get("sizeBytes"), int)
                or not 0 < mount["sizeBytes"] <= limits["bytes"]
                or isinstance(mount.get("inodeLimit"), bool)
                or not isinstance(mount.get("inodeLimit"), int)
                or not 0 < mount["inodeLimit"] <= limits["inodes"]
            ):
                return False
            total_bytes += mount["sizeBytes"]
            total_inodes += mount["inodeLimit"]
        for path in CANDIDATE_PERSISTENT_PATHS:
            mount = read_only.get(path)
            mount_point = mount.get("mountPoint") if isinstance(mount, Mapping) else None
            normalized_mount = (
                isinstance(mount_point, str)
                and PurePosixPath(mount_point).is_absolute()
                and ".." not in PurePosixPath(mount_point).parts
                and str(PurePosixPath(mount_point)) == mount_point
            )
            if (
                not isinstance(mount, Mapping)
                or set(mount) != {"mountId", "mountPoint", "readOnly"}
                or not normalized_mount
                or not valid_mount_id(mount.get("mountId"), str(mount_point))
                or not (
                    path == mount_point
                    or path.startswith(str(mount_point).rstrip("/") + "/")
                )
                or mount.get("readOnly") is not True
            ):
                return False
        for path in CANDIDATE_INACCESSIBLE_PATHS:
            mount = inaccessible.get(path)
            if (
                not isinstance(mount, Mapping)
                or set(mount) != {"mountId", "mountPoint", "mode", "readOnly"}
                or not valid_mount_id(mount.get("mountId"), path)
                or mount.get("mountPoint") != path
                or mount.get("mode") != "0000"
                or mount.get("readOnly") is not True
            ):
                return False
        return bool(
            mounts.get("totalTmpfsBytes") == total_bytes
            and mounts.get("totalTmpfsInodes") == total_inodes
            and total_bytes
            <= sum(item["bytes"] for item in CANDIDATE_TMPFS_LIMITS.values())
            and total_inodes
            <= sum(item["inodes"] for item in CANDIDATE_TMPFS_LIMITS.values())
        )

    def valid_termination_proof(value: object) -> bool:
        if not isinstance(value, Mapping) or set(value) != {
            "schemaVersion",
            "unit",
            "controlGroup",
            "initialPresent",
            "initialPopulated",
            "killSignals",
            "stopSucceeded",
            "resetPerformed",
            "recursivePopulated",
            "loadState",
            "activeState",
            "cgroupRemoved",
        }:
            return False
        unit = f"nuvion-candidate-soak-{run_id.replace('-', '')}.service"
        signals = value.get("killSignals")
        return bool(
            value.get("schemaVersion") == 1
            and value.get("unit") == unit
            and value.get("controlGroup") == "/system.slice/" + unit
            and isinstance(value.get("initialPresent"), bool)
            and isinstance(value.get("initialPopulated"), bool)
            and isinstance(signals, list)
            and signals in ([], ["SIGTERM"], ["SIGTERM", "SIGKILL"])
            and value.get("stopSucceeded") is True
            and isinstance(value.get("resetPerformed"), bool)
            and value.get("recursivePopulated") is False
            and value.get("loadState") == "not-found"
            and value.get("activeState") == "inactive"
            and value.get("cgroupRemoved") is True
            and (value.get("initialPopulated") is False or bool(signals))
        )

    def valid_collector_proof(value: object) -> bool:
        if not isinstance(value, Mapping) or set(value) != {
            "schemaVersion",
            "unit",
            "controlGroup",
            "requiredSeconds",
            "elapsedSeconds",
            "scanIntervalSeconds",
            "sampleCount",
            "observedPids",
            "escapeDetected",
            "allSamplesWithinCgroup",
            "durationSatisfied",
            "terminalStatus",
            "afterTermination",
        }:
            return False
        unit = f"nuvion-candidate-soak-{run_id.replace('-', '')}.service"
        terminal = value.get("terminalStatus")
        after = value.get("afterTermination")
        pids = value.get("observedPids")
        return bool(
            value.get("schemaVersion") == 1
            and value.get("unit") == unit
            and value.get("controlGroup") == "/system.slice/" + unit
            and value.get("requiredSeconds") == CANDIDATE_REQUIRED_SOAK_SECONDS
            and type(value.get("elapsedSeconds")) in {int, float}
            and value["elapsedSeconds"] >= CANDIDATE_REQUIRED_SOAK_SECONDS
            and value.get("scanIntervalSeconds")
            == CANDIDATE_UID_SCAN_INTERVAL_SECONDS
            and type(value.get("sampleCount")) is int
            and value["sampleCount"] >= 1
            and isinstance(pids, list)
            and all(type(pid) is int and pid >= 2 for pid in pids)
            and len(pids) == len(set(pids))
            and value.get("escapeDetected") is None
            and value.get("allSamplesWithinCgroup") is True
            and value.get("durationSatisfied") is True
            and isinstance(terminal, Mapping)
            and set(terminal)
            == {
                "ActiveState",
                "ExecMainCode",
                "ExecMainStatus",
                "Result",
                "SubState",
            }
            and terminal.get("Result") == "success"
            and terminal.get("ExecMainCode") == "1"
            and terminal.get("ExecMainStatus") == "0"
            and isinstance(after, Mapping)
            and set(after)
            == {"schemaVersion", "uid", "pids", "controlGroup", "stableScans"}
            and after.get("schemaVersion") == 1
            and type(after.get("uid")) is int
            and after["uid"] >= 1
            and after.get("pids") == []
            and after.get("controlGroup") is None
            and after.get("stableScans") == 2
        )

    def valid_production_restoration(value: object) -> bool:
        if not isinstance(value, Mapping) or set(value) != {
            "schemaVersion",
            "transactionPhase",
            "manifestSha256",
            "files",
            "directories",
            "units",
            "sha256",
        }:
            return False
        files = value.get("files")
        directories = value.get("directories")
        units = value.get("units")
        expected_manifest_sha = hashlib.sha256(
            (
                json.dumps(
                    manifest,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode()
        ).hexdigest()
        if (
            value.get("schemaVersion") != 1
            or value.get("transactionPhase") != "RESTORED"
            or value.get("manifestSha256") != expected_manifest_sha
            or not isinstance(files, Mapping)
            or set(files) != PRODUCTION_TRANSACTION_FILES
            or not isinstance(directories, Mapping)
            or set(directories) != PRODUCTION_TRANSACTION_DIRECTORIES
            or not isinstance(units, Mapping)
            or set(units) != PRODUCTION_UNITS
            or not isinstance(value.get("sha256"), str)
            or SHA256_RE.fullmatch(value["sha256"]) is None
        ):
            return False
        for item in files.values():
            if not isinstance(item, Mapping) or set(item) != {
                "exists",
                "sha256",
                "mode",
                "uid",
                "gid",
            }:
                return False
            exists = item.get("exists")
            if exists is True:
                if (
                    not isinstance(item.get("sha256"), str)
                    or SHA256_RE.fullmatch(item["sha256"]) is None
                    or any(type(item.get(key)) is not int for key in ("mode", "uid", "gid"))
                ):
                    return False
            elif exists is False:
                if any(item.get(key) is not None for key in ("sha256", "mode", "uid", "gid")):
                    return False
            else:
                return False
        if any(
            not isinstance(item, Mapping)
            or set(item) != {"mode", "uid", "gid"}
            or any(type(item.get(key)) is not int for key in ("mode", "uid", "gid"))
            for item in directories.values()
        ):
            return False
        if any(
            not isinstance(item, Mapping)
            or set(item) != {"active", "enabled", "unitFileState"}
            or type(item.get("active")) is not bool
            or type(item.get("enabled")) is not bool
            or not isinstance(item.get("unitFileState"), str)
            for item in units.values()
        ):
            return False
        core = {key: value[key] for key in value if key != "sha256"}
        serialized = (
            json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode()
        return hashlib.sha256(serialized).hexdigest() == value["sha256"]

    anti_replay = pre["antiReplay"]
    latest_replay = (
        anti_replay.get("latest") if isinstance(anti_replay, Mapping) else None
    )
    if (
        not isinstance(anti_replay, Mapping)
        or set(anti_replay)
        != {
            "schemaVersion",
            "semanticSha256",
            "maximumCommandSequence",
            "currentReleaseSequence",
            "currentBomDigest",
            "latest",
        }
        or anti_replay.get("schemaVersion") != 4
        or isinstance(anti_replay.get("schemaVersion"), bool)
        or not isinstance(anti_replay.get("semanticSha256"), str)
        or SHA256_RE.fullmatch(anti_replay["semanticSha256"]) is None
        or isinstance(anti_replay.get("maximumCommandSequence"), bool)
        or not isinstance(anti_replay.get("maximumCommandSequence"), int)
        or anti_replay["maximumCommandSequence"] < 1
        or not isinstance(latest_replay, Mapping)
        or set(latest_replay)
        != {
            "commandId",
            "sequence",
            "phase",
            "bomDigest",
            "releaseSequence",
            "healthDeadline",
        }
        or latest_replay.get("commandId") != scenario["expectedCommandId"]
        or isinstance(latest_replay.get("sequence"), bool)
        or not isinstance(latest_replay.get("sequence"), int)
        or latest_replay.get("sequence")
        != anti_replay.get("maximumCommandSequence")
        or latest_replay.get("phase") != "ROLLED_BACK"
        or latest_replay.get("bomDigest") != scenario["expectedBomDigest"]
        or isinstance(latest_replay.get("releaseSequence"), bool)
        or not isinstance(latest_replay.get("releaseSequence"), int)
        or latest_replay.get("releaseSequence")
        != scenario["release"]["releaseSequence"]
        or latest_replay.get("healthDeadline") is not None
        or (
            anti_replay.get("currentReleaseSequence") is not None
            and (
                not isinstance(anti_replay.get("currentReleaseSequence"), str)
                or re.fullmatch(
                    r"(?:0|[1-9][0-9]*)",
                    anti_replay["currentReleaseSequence"],
                )
                is None
            )
        )
        or (
            anti_replay.get("currentBomDigest") is not None
            and (
                not isinstance(anti_replay.get("currentBomDigest"), str)
                or DIGEST_RE.fullmatch(anti_replay["currentBomDigest"]) is None
            )
        )
    ):
        raise RunnerError("candidate soak anti-replay proof is invalid")
    if (
        any(
            gates.get(key) is not True
            for key in (
                "signedRollbackTerminal",
                "candidateBound",
                "slotsUnchanged",
                "releaseTreesUnchanged",
                "antiReplayUnchanged",
                "oakIdentityUnchanged",
                "freshBaselineProcess",
                "harnessBytesPinned",
                "harnessCopyRemoved",
                "resourceLimitsApplied",
                "boundedOutput",
                "persistentStateReadOnly",
                "persistentStateUnchanged",
                "productionTrustRestored",
                "trustedSoakDuration",
                "continuousUidIsolation",
                "cgroupTerminated",
            )
        )
        or gates.get("harnessPassed") is not passed
        or gates.get("rawEvidencePreserved") is not (raw is not None)
        or (passed and outcome.get("errorCode") is not None)
        or (
            not passed
            and (
                not isinstance(outcome.get("errorCode"), str)
                or not outcome["errorCode"]
            )
        )
        or pre["slots"] != expected_slots
        or pre["slots"] != post["slots"]
        or not valid_release_trees(pre.get("releaseTrees"))
        or pre.get("releaseTrees") != post.get("releaseTrees")
        or pre["antiReplay"] != post["antiReplay"]
        or not valid_persistent_state(pre.get("persistentState"))
        or pre.get("persistentState") != post.get("persistentState")
        or not valid_execution_proof(evidence.get("executionProof"))
        or not valid_collector_proof(evidence.get("collectorProof"))
        or not valid_termination_proof(evidence.get("terminationProof"))
        or not valid_production_restoration(
            evidence.get("productionRestoration")
        )
        or not valid_oak(pre_oak)
        or not valid_oak(post_oak)
        or pre_oak.get("port") != post_oak.get("port")
        or pre_oak.get("mxidSha256") != post_oak.get("mxidSha256")
        or not isinstance(pre_runtime, Mapping)
        or not isinstance(post_runtime, Mapping)
        or set(pre_runtime) != {"pid", "startTicks", "bootId", "activeSlot"}
        or set(post_runtime) != {"pid", "startTicks", "bootId", "activeSlot"}
        or any(
            isinstance(runtime.get(field), bool)
            or not isinstance(runtime.get(field), int)
            or runtime[field] < 1
            for runtime in (pre_runtime, post_runtime)
            for field in ("pid", "startTicks")
        )
        or any(runtime["pid"] < 2 for runtime in (pre_runtime, post_runtime))
        or pre_runtime.get("activeSlot") != expected_slots["current"]
        or post_runtime.get("activeSlot") != expected_slots["current"]
        or pre_runtime.get("bootId") != post_runtime.get("bootId")
        or (
            pre_runtime.get("pid"),
            pre_runtime.get("startTicks"),
        )
        == (
            post_runtime.get("pid"),
            post_runtime.get("startTicks"),
        )
    ):
        raise RunnerError("candidate soak restoration proof is invalid")
    try:
        canonical_boot_id = str(uuid.UUID(str(pre_runtime.get("bootId"))))
    except ValueError as exc:
        raise RunnerError("candidate soak boot identity is invalid") from exc
    if canonical_boot_id != pre_runtime.get("bootId"):
        raise RunnerError("candidate soak boot identity is invalid")
    candidate_started = timestamp(evidence.get("startedAt"), label="start time")
    restored_at = timestamp(post.get("restoredAt"), label="restore time")
    completed_at = timestamp(evidence.get("completedAt"), label="completion time")
    if not candidate_started <= restored_at <= completed_at:
        raise RunnerError("candidate soak lifecycle ordering is invalid")
    raw_digest = evidence.get("rawEvidenceSha256")
    if raw_evidence_sha256 is not None and raw_digest != raw_evidence_sha256:
        raise RunnerError("candidate soak supplied raw bytes digest mismatch")
    if raw is None:
        if passed or raw_digest is not None:
            raise RunnerError("candidate soak raw evidence is missing")
    else:
        if not isinstance(raw, Mapping):
            raise RunnerError("candidate soak raw evidence is invalid")
        serialized = (
            json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        if hashlib.sha256(serialized).hexdigest() != raw_digest:
            raise RunnerError("candidate soak raw evidence digest mismatch")
        runtime_identity = raw.get("runtimeIdentity")
        raw_outcome = raw.get("outcome")
        if (
            raw.get("schemaVersion") != 3
            or raw.get("kind") != "nuvion-iq9075-oak-soak-result"
            or raw.get("runId") != run_id
            or raw.get("slotKind") != "candidate"
            or not isinstance(runtime_identity, Mapping)
            or set(runtime_identity)
            != {
                "agentVersion",
                "componentSha",
                "bomDigest",
                "pythonPath",
                "sitePackagesPath",
                "buildInfoPath",
                "releaseMarkerSha256",
                "candidateSlot",
                "controlMarkerSha256",
            }
            or runtime_identity.get("agentVersion")
            != scenario["release"]["agentVersion"]
            or runtime_identity.get("componentSha")
            != scenario["release"]["componentSha"]
            or runtime_identity.get("bomDigest")
            != scenario["expectedBomDigest"]
            or runtime_identity.get("pythonPath")
            != "/usr/bin/python3"
            or runtime_identity.get("sitePackagesPath")
            != expected_slot + "/venv/lib/python3.12/site-packages"
            or runtime_identity.get("buildInfoPath")
            != expected_slot
            + "/venv/lib/python3.12/site-packages/nuvion_app/build_info.py"
            or not isinstance(runtime_identity.get("releaseMarkerSha256"), str)
            or SHA256_RE.fullmatch(runtime_identity["releaseMarkerSha256"]) is None
            or runtime_identity.get("candidateSlot") != expected_slot
            or runtime_identity.get("controlMarkerSha256")
            != candidate["controlMarkerSha256"]
            or not isinstance(raw_outcome, Mapping)
            or raw_outcome.get("status") != ("passed" if passed else "failed")
        ):
            raise RunnerError("candidate soak raw evidence is not bound to the run")
        raw_started = timestamp(raw.get("startedAt"), label="raw start time")
        if not candidate_started <= raw_started <= restored_at:
            raise RunnerError("candidate soak lifecycle ordering is invalid")
    assert_no_secret_material(evidence)


def validate_release_marker(
    marker: object,
    *,
    expected_bom_digest: str,
    expected_version: str,
    expected_release: Mapping[str, Any] | None = None,
) -> None:
    fields = {
        "schemaVersion",
        "bomDigest",
        "agentVersion",
        "releaseSequence",
        "artifactDigest",
        "componentSha",
        "configSchema",
        "publisherKeyId",
    }
    if (
        not isinstance(marker, Mapping)
        or set(marker) != fields
        or type(marker.get("schemaVersion")) is not int
        or marker.get("schemaVersion") != 2
        or marker.get("bomDigest") != expected_bom_digest
        or marker.get("agentVersion") != expected_version
        or isinstance(marker.get("releaseSequence"), bool)
        or not isinstance(marker.get("releaseSequence"), int)
        or marker["releaseSequence"] < 1
        or not isinstance(marker.get("artifactDigest"), str)
        or DIGEST_RE.fullmatch(marker["artifactDigest"]) is None
        or not isinstance(marker.get("componentSha"), str)
        or COMPONENT_RE.fullmatch(marker["componentSha"]) is None
        or not isinstance(marker.get("configSchema"), str)
        or re.fullmatch(r"[1-9][0-9]*", marker["configSchema"]) is None
        or not isinstance(marker.get("publisherKeyId"), str)
        or KEY_ID_RE.fullmatch(marker["publisherKeyId"]) is None
    ):
        raise RunnerError("release marker identity is invalid")
    if expected_release is not None and dict(marker) != {
        "schemaVersion": 2,
        "bomDigest": expected_bom_digest,
        **dict(expected_release),
    }:
        raise RunnerError("release marker differs from the expected release")


def validate_anti_replay_evidence(
    snapshot: object,
    *,
    update: Mapping[str, Any],
    slots: Mapping[str, Any],
) -> None:
    """Bind the live updater journal snapshot to its terminal public state."""

    latest = snapshot.get("latest") if isinstance(snapshot, Mapping) else None
    current_marker = slots.get("release")
    current_release_sequence = (
        str(current_marker.get("releaseSequence"))
        if isinstance(current_marker, Mapping)
        else None
    )
    current_bom_digest = (
        current_marker.get("bomDigest")
        if isinstance(current_marker, Mapping)
        else None
    )
    expected_latest = {
        "commandId": update.get("commandId"),
        "sequence": update.get("sequence"),
        "phase": update.get("phase"),
        "bomDigest": update.get("bomDigest"),
        "releaseSequence": update.get("releaseSequence"),
        "healthDeadline": None,
    }
    if (
        not isinstance(snapshot, Mapping)
        or set(snapshot)
        != {
            "schemaVersion",
            "semanticSha256",
            "maximumCommandSequence",
            "currentReleaseSequence",
            "currentBomDigest",
            "latest",
        }
        or type(snapshot.get("schemaVersion")) is not int
        or snapshot.get("schemaVersion") != 4
        or not isinstance(snapshot.get("semanticSha256"), str)
        or SHA256_RE.fullmatch(snapshot["semanticSha256"]) is None
        or isinstance(snapshot.get("maximumCommandSequence"), bool)
        or not isinstance(snapshot.get("maximumCommandSequence"), int)
        or snapshot.get("maximumCommandSequence") != update.get("sequence")
        or not isinstance(latest, Mapping)
        or dict(latest) != expected_latest
        or snapshot.get("currentReleaseSequence") != current_release_sequence
        or snapshot.get("currentBomDigest") != current_bom_digest
    ):
        raise RunnerError(
            "final updater anti-replay snapshot differs from terminal state"
        )


def validate_final_evidence(
    evidence: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    validate_manifest(manifest)
    evidence_fields = {
        "schemaVersion",
        "protocolVersion",
        "runId",
        "generatedAt",
        "scenario",
        "complete",
        "gates",
        "oak",
        "services",
        "runtimePids",
        "slots",
        "updater",
    }
    schema_version = evidence.get("schemaVersion")
    if schema_version == 2:
        evidence_fields.add("antiReplay")
    if set(evidence) != evidence_fields:
        raise RunnerError("final evidence root fields are invalid")
    generated_at = evidence.get("generatedAt")
    if not isinstance(generated_at, str) or re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
        r"(?:\.[0-9]{3})?Z",
        generated_at,
    ) is None:
        raise RunnerError("final evidence generatedAt is invalid")
    try:
        parsed_generated_at = datetime.fromisoformat(
            generated_at[:-1] + "+00:00"
        )
    except ValueError as exc:
        raise RunnerError("final evidence generatedAt is invalid") from exc
    if (
        type(schema_version) is not int
        or schema_version not in {1, 2}
        or evidence.get("protocolVersion") != PROTOCOL_VERSION
        or evidence.get("runId") != manifest.get("runId")
        or evidence.get("scenario") != manifest["scenario"]["type"]
        or evidence.get("complete") is not True
        or parsed_generated_at.tzinfo != timezone.utc
    ):
        raise RunnerError("final evidence is not complete for the immutable run")
    gates = evidence.get("gates")
    if (
        not isinstance(gates, dict)
        or set(gates)
        != {
            "foundation",
            "backup",
            "trust",
            "updater2",
            "oak",
            "services",
            "scenario",
        }
        or any(value is not True for value in gates.values())
    ):
        raise RunnerError("final evidence contains a false or missing gate")
    oak = evidence.get("oak")
    if (
        not isinstance(oak, dict)
        or set(oak)
        != {
            "port",
            "vendorId",
            "productId",
            "speedMbps",
            "mxidSha256",
            "attached",
            "bound",
        }
        or not isinstance(oak.get("port"), str)
        or re.fullmatch(r"2-1(?:\.[1-9][0-9]*)+", oak["port"]) is None
        or oak.get("vendorId") != "03e7"
        or oak.get("productId") != "f63b"
        or isinstance(oak.get("speedMbps"), bool)
        or not isinstance(oak.get("speedMbps"), int)
        or oak["speedMbps"] < 5000
        or not isinstance(oak.get("mxidSha256"), str)
        or SHA256_RE.fullmatch(oak["mxidSha256"]) is None
        or oak.get("attached") is not True
        or oak.get("bound") is not True
    ):
        raise RunnerError("final evidence lacks exact USB1 runtime OAK proof")
    services = evidence.get("services")
    expected_units = {
        "nuv-agent.service",
        "nuv-agent-updater.service",
        "nuv-agent-updater.socket",
    }
    if not isinstance(services, dict) or set(services) != expected_units:
        raise RunnerError("final evidence service set is invalid")
    for unit, status in services.items():
        if (
            not isinstance(status, dict)
            or set(status) != {"active", "enabled", "unitFileState", "mainPid"}
            or status.get("active") is not True
            or type(status.get("enabled")) is not bool
            or status.get("unitFileState")
            not in {
                "enabled",
                "disabled",
                "static",
                "indirect",
                "masked",
                "generated",
                "transient",
                "alias",
            }
            or status.get("enabled")
            is not (status.get("unitFileState") in {"enabled", "static", "indirect"})
            or isinstance(status.get("mainPid"), bool)
            or not isinstance(status.get("mainPid"), int)
            or (unit.endswith(".service") and status["mainPid"] < 2)
            or (unit.endswith(".socket") and status["mainPid"] != 0)
        ):
            raise RunnerError("final evidence service state is invalid")
    updater = evidence.get("updater")
    if (
        not isinstance(updater, dict)
        or set(updater)
        != {
            "capabilityAvailable",
            "authenticatedHelper",
            "reason",
            "updaterVersion",
            "update",
        }
        or updater.get("capabilityAvailable") is not True
        or updater.get("authenticatedHelper") is not True
        or updater.get("reason") != "READY"
        or updater.get("updaterVersion") != "0.2.0"
    ):
        raise RunnerError("final evidence is missing updater 0.2.0 proof")
    update = updater.get("update")
    scenario = manifest["scenario"]
    release = scenario["release"]
    expected_candidate_relative = "releases/" + scenario["expectedBomDigest"][7:]
    allowed_update_fields = {
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
        "message",
        "slot",
        "rollbackSlot",
        "rollbackVersion",
        "health",
        "functionalHealth",
    }
    required_update_fields = {
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
        "slot",
        "health",
        "functionalHealth",
    }
    if (
        not isinstance(update, dict)
        or not required_update_fields.issubset(update)
        or not set(update).issubset(allowed_update_fields)
        or any(
        (
            update.get("commandId") != scenario["expectedCommandId"],
            update.get("bomDigest") != scenario["expectedBomDigest"],
            update.get("targetVersion") != release["agentVersion"],
            update.get("candidateSlot") != scenario["expectedCandidateSlot"],
            update.get("previousSlot") != scenario["expectedPreviousSlot"],
            update.get("previousVersion") != scenario["expectedPreviousVersion"],
            update.get("releaseSequence") != release["releaseSequence"],
            update.get("artifactDigest") != release["artifactDigest"],
            update.get("componentSha") != release["componentSha"],
            update.get("configSchema") != release["configSchema"],
            update.get("publisherKeyId") != release["publisherKeyId"],
            update.get("bomVerificationStatus") != "VERIFIED",
        )
        )
    ):
        raise RunnerError("final updater identity does not match the manifest")
    if (
        isinstance(update.get("sequence"), bool)
        or not isinstance(update.get("sequence"), int)
        or update["sequence"] < 1
        or isinstance(update.get("releaseSequence"), bool)
        or not isinstance(update.get("releaseSequence"), int)
        or update.get("updatePhase") != update.get("phase")
    ):
        raise RunnerError("final updater lifecycle fields are invalid")
    update_times: dict[str, datetime] = {}
    for field in ("updatedAt", "commandExpiresAt", "healthDeadline"):
        if field not in update:
            continue
        value = update[field]
        if not isinstance(value, str) or re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
            r"(?:\.[0-9]{3})?Z",
            value,
        ) is None:
            raise RunnerError(f"final updater {field} is invalid")
        try:
            parsed = datetime.fromisoformat(value[:-1] + "+00:00")
        except ValueError as exc:
            raise RunnerError(f"final updater {field} is invalid") from exc
        if parsed.tzinfo != timezone.utc:
            raise RunnerError(f"final updater {field} is invalid")
        update_times[field] = parsed
    for field in ("errorCode", "message", "rollbackSlot", "rollbackVersion"):
        if field in update and (
            not isinstance(update[field], str) or not update[field]
        ):
            raise RunnerError(f"final updater {field} is invalid")
    if parsed_generated_at < update_times["updatedAt"]:
        raise RunnerError("final evidence predates its updater state")
    if "healthDeadline" in update:
        raise RunnerError("terminal evidence contains a stale health deadline")
    slots = evidence.get("slots")
    if not isinstance(slots, dict) or set(slots) != {
        "current",
        "previous",
        "currentVersion",
        "release",
        "previousRelease",
    }:
        raise RunnerError("final evidence is missing live slot proof")
    if scenario["type"] == "commit":
        forbidden_commit_fields = {
            "errorCode",
            "message",
            "rollbackSlot",
            "rollbackVersion",
        }
        if evidence.get("runtimePids") is not None:
            raise RunnerError("commit evidence must not contain rollback PIDs")
        if set(update) & forbidden_commit_fields:
            raise RunnerError("commit evidence contains contradictory updater fields")
        if (
            update_times["updatedAt"] >= update_times["commandExpiresAt"]
        ):
            raise RunnerError("commit evidence timestamp ordering is invalid")
        expected_marker = {
            "schemaVersion": 2,
            "bomDigest": scenario["expectedBomDigest"],
            **release,
        }
        validate_release_marker(
            slots.get("release"),
            expected_bom_digest=scenario["expectedBomDigest"],
            expected_version=release["agentVersion"],
            expected_release=release,
        )
        previous_marker = slots.get("previousRelease")
        if scenario["expectedPreviousSlot"].startswith("releases/"):
            validate_release_marker(
                previous_marker,
                expected_bom_digest=(
                    "sha256:" + scenario["expectedPreviousSlot"][9:]
                ),
                expected_version=scenario["expectedPreviousVersion"],
            )
        elif previous_marker is not None:
            raise RunnerError("bootstrap previous slot cannot have a release marker")
        if any(
            (
                update.get("phase") != "COMMITTED",
                update.get("slot") != expected_candidate_relative,
                update.get("health") != "FUNCTIONAL_HEALTHY",
                update.get("functionalHealth") != "FUNCTIONAL_HEALTHY",
                slots.get("current") != expected_candidate_relative,
                slots.get("previous") != scenario["expectedPreviousSlot"],
                slots.get("currentVersion") != release["agentVersion"],
                slots.get("release") != expected_marker,
            )
        ):
            raise RunnerError("commit scenario did not reach COMMITTED")
    else:
        runtime_pids = evidence.get("runtimePids")
        services = evidence.get("services")
        agent_service = (
            services.get("nuv-agent.service")
            if isinstance(services, Mapping)
            else None
        )
        if (
            not isinstance(runtime_pids, Mapping)
            or set(runtime_pids) != {"candidate", "restored"}
            or isinstance(runtime_pids.get("candidate"), bool)
            or not isinstance(runtime_pids.get("candidate"), int)
            or runtime_pids["candidate"] < 2
            or isinstance(runtime_pids.get("restored"), bool)
            or not isinstance(runtime_pids.get("restored"), int)
            or runtime_pids["restored"] < 2
            or runtime_pids["candidate"] == runtime_pids["restored"]
            or not isinstance(agent_service, Mapping)
            or agent_service.get("mainPid") != runtime_pids["restored"]
        ):
            raise RunnerError("rollback evidence lacks exact candidate/restored PID proof")
        expected_previous = scenario["expectedPreviousSlot"]
        marker = slots.get("release")
        if expected_previous.startswith("releases/"):
            validate_release_marker(
                marker,
                expected_bom_digest="sha256:" + expected_previous[9:],
                expected_version=scenario["expectedPreviousVersion"],
            )
        elif marker is not None:
            raise RunnerError("bootstrap rollback slot cannot have a release marker")
        validate_release_marker(
            slots.get("previousRelease"),
            expected_bom_digest=scenario["expectedBomDigest"],
            expected_version=release["agentVersion"],
            expected_release=release,
        )
        if any(
            (
                update.get("phase") != "ROLLED_BACK",
                update.get("slot") != expected_previous,
                update.get("rollbackSlot") != expected_previous,
                update.get("rollbackVersion") != scenario["expectedPreviousVersion"],
                update.get("errorCode") != "ROLLED_BACK",
                update.get("health") != "LKG_RESTORED",
                update.get("functionalHealth") != "FUNCTIONAL_UNHEALTHY",
                slots.get("current") != expected_previous,
                slots.get("previous") != expected_candidate_relative,
                slots.get("currentVersion") != scenario["expectedPreviousVersion"],
                slots.get("previousRelease")
                != {
                    "schemaVersion": 2,
                    "bomDigest": scenario["expectedBomDigest"],
                    **release,
                },
            )
        ):
            raise RunnerError("rollback scenario lacks exact rollback/error proof")
    if schema_version == 2:
        validate_anti_replay_evidence(
            evidence.get("antiReplay"), update=update, slots=slots
        )
    assert_no_secret_material(evidence)


def validate_cleanup_result(result: Mapping[str, Any], *, run_id: str) -> None:
    allowed = {
        "schemaVersion",
        "kind",
        "runId",
        "complete",
        "recovered",
        "phase",
        "idempotent",
        "proof",
    }
    if not set(result).issubset(allowed) or not {
        "schemaVersion",
        "runId",
        "complete",
        "recovered",
        "phase",
        "kind",
        "proof",
    }.issubset(result):
        raise RunnerError("board cleanup did not reach the exact restored state")
    proof = result.get("proof")
    expected_proof_fields = {
        "schemaVersion",
        "transactionPhase",
        "cleanupJournalComplete",
        "activeRunLeaseAbsent",
        "transactionSnapshotsAbsent",
        "recoveryArchiveAbsent",
        "candidateArtifactsAbsent",
        "candidateStagingAbsent",
        "trustStagingAbsent",
    }
    if (
        type(result.get("schemaVersion")) is not int
        or result.get("schemaVersion") != 1
        or result.get("kind") != "nuvion-iq9075-cleanup-evidence"
        or result.get("runId") != run_id
        or result.get("complete") is not True
        or not isinstance(result.get("recovered"), bool)
        or result.get("phase") not in {None, "RESTORED"}
        or (
            "idempotent" in result
            and not isinstance(result.get("idempotent"), bool)
        )
        or not isinstance(proof, Mapping)
        or set(proof) != expected_proof_fields
        or type(proof.get("schemaVersion")) is not int
        or proof.get("schemaVersion") != 1
        or proof.get("transactionPhase") != result.get("phase")
        or any(
            proof.get(key) is not True
            for key in expected_proof_fields
            - {"schemaVersion", "transactionPhase"}
        )
    ):
        raise RunnerError("board cleanup did not reach the exact restored state")


def canonical_cleanup_evidence(
    result: Mapping[str, Any], *, run_id: str
) -> dict[str, Any]:
    validate_cleanup_result(result, run_id=run_id)
    return {
        "schemaVersion": 1,
        "kind": "nuvion-iq9075-cleanup-evidence",
        "runId": run_id,
        "complete": True,
        "recovered": bool(result["recovered"]),
        "phase": result["phase"],
        "proof": dict(result["proof"]),
    }


def validate_bound_cleanup_evidence(
    evidence: Mapping[str, Any],
    *,
    run_id: str,
    manifest_raw: bytes,
    fleet_evidence_raw: bytes,
) -> dict[str, Any]:
    """Validate the host wrapper binding cleanup to one exact rollback run."""

    canonical_run_id(run_id)
    manifest = strict_canonical_json(
        manifest_raw, label="immutable manifest"
    )
    fleet_evidence = strict_canonical_json(
        fleet_evidence_raw, label="Fleet evidence"
    )
    validated_manifest = validate_manifest(manifest)
    validate_final_evidence(fleet_evidence, validated_manifest)
    expected_fields = {
        "schemaVersion",
        "kind",
        "protocolVersion",
        "runId",
        "complete",
        "recovered",
        "phase",
        "proof",
        "completedAt",
        "manifestSha256",
        "fleetEvidenceSha256",
        "identity",
    }
    identity = evidence.get("identity")
    expected_identity = {
        field: validated_manifest["identity"][field]
        for field in CLEANUP_IDENTITY_FIELDS
    }
    if (
        set(evidence) != expected_fields
        or type(evidence.get("schemaVersion")) is not int
        or evidence.get("schemaVersion") != 2
        or evidence.get("kind") != "nuvion-iq9075-cleanup-evidence"
        or evidence.get("protocolVersion") != PROTOCOL_VERSION
        or evidence.get("runId") != run_id
        or evidence.get("manifestSha256")
        != hashlib.sha256(manifest_raw).hexdigest()
        or evidence.get("fleetEvidenceSha256")
        != hashlib.sha256(fleet_evidence_raw).hexdigest()
        or not isinstance(identity, Mapping)
        or dict(identity) != expected_identity
    ):
        raise RunnerError("cleanup evidence is not bound to the exact Fleet run")
    remote_result = {
        "schemaVersion": 1,
        "kind": evidence["kind"],
        "runId": evidence["runId"],
        "complete": evidence.get("complete"),
        "recovered": evidence.get("recovered"),
        "phase": evidence.get("phase"),
        "proof": evidence.get("proof"),
    }
    validate_cleanup_result(remote_result, run_id=run_id)
    completed_at = canonical_utc(
        evidence.get("completedAt"), label="cleanup completedAt"
    )
    generated_at = canonical_utc(
        fleet_evidence.get("generatedAt"), label="Fleet evidence generatedAt"
    )
    if generated_at > completed_at:
        raise RunnerError("cleanup evidence predates Fleet rollback evidence")
    assert_no_secret_material(evidence)
    return dict(evidence)


def build_bound_cleanup_evidence(
    result: Mapping[str, Any],
    *,
    run_id: str,
    manifest_raw: bytes,
    fleet_evidence_raw: bytes,
    completed_at: str,
) -> dict[str, Any]:
    """Wrap the unchanged board cleanup RPC in exact local run bindings."""

    remote = canonical_cleanup_evidence(result, run_id=run_id)
    manifest = strict_canonical_json(
        manifest_raw, label="immutable manifest"
    )
    fleet_evidence = strict_canonical_json(
        fleet_evidence_raw, label="Fleet evidence"
    )
    validated_manifest = validate_manifest(manifest)
    validate_final_evidence(fleet_evidence, validated_manifest)
    evidence = {
        "schemaVersion": 2,
        "kind": remote["kind"],
        "protocolVersion": PROTOCOL_VERSION,
        "runId": run_id,
        "complete": remote["complete"],
        "recovered": remote["recovered"],
        "phase": remote["phase"],
        "proof": remote["proof"],
        "completedAt": completed_at,
        "manifestSha256": hashlib.sha256(manifest_raw).hexdigest(),
        "fleetEvidenceSha256": hashlib.sha256(fleet_evidence_raw).hexdigest(),
        "identity": {
            field: validated_manifest["identity"][field]
            for field in CLEANUP_IDENTITY_FIELDS
        },
    }
    return validate_bound_cleanup_evidence(
        evidence,
        run_id=run_id,
        manifest_raw=manifest_raw,
        fleet_evidence_raw=fleet_evidence_raw,
    )


class FleetRunner:
    def __init__(
        self,
        *,
        transport: OpenSshTransport,
        journal: HostJournal,
        output_dir: Path,
        run_id: str,
        monotonic: Any = time.monotonic,
        sleeper: Any = time.sleep,
    ) -> None:
        self.transport = transport
        self.journal = journal
        self.output_dir = output_dir
        self.run_id = run_id
        self.monotonic = monotonic
        self.sleeper = sleeper

    def _call(
        self,
        step: str,
        command: str,
        arguments: Sequence[str] = (),
        *,
        timeout: float = 90,
        validate: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        # Never skip a board call from a stale host journal. Every primitive
        # performs a live, idempotent reconcile and returns current proof.
        journal_failure: BaseException | None = None
        try:
            self.journal.mark(step, "RUNNING")
        except BaseException as exc:
            # The local journal is evidence, not the authority that may block
            # a root cleanup RPC from reaching the board.
            journal_failure = exc
        try:
            result = self.transport.invoke_board(command, arguments, timeout=timeout)
            if validate is not None:
                validate(result)
        except BaseException:
            try:
                self.journal.mark(step, "FAILED")
            except BaseException:
                # Board failure remains the primary safety signal. A corrupt
                # host journal must not replace it or skip outer recovery.
                pass
            raise
        try:
            self.journal.mark(step, "COMPLETE")
        except BaseException as exc:
            if journal_failure is None:
                journal_failure = exc
        if journal_failure is not None:
            raise RunnerError("host journal update failed after board call") from journal_failure
        return result

    def bootstrap(
        self,
        *,
        installer: Path,
        package: Path,
        local_tool: Path,
        expected_version: str,
        expected_package_sha256: str,
    ) -> dict[str, object]:
        if not SEMVER_RE.fullmatch(expected_version) or not SHA256_RE.fullmatch(
            expected_package_sha256
        ):
            raise RunnerError("bootstrap version or package digest is invalid")
        installer_sha = sha256_regular(installer, MAX_BOOTSTRAP_INSTALLER_BYTES)
        package_sha = sha256_regular(package, MAX_BOOTSTRAP_DEB_BYTES)
        local_tool_sha = sha256_regular(local_tool, MAX_OUTPUT_BYTES)
        if package_sha != expected_package_sha256:
            raise RunnerError("local bootstrap package digest mismatch")
        self.journal.mark("bootstrap-staging", "RUNNING")
        try:
            installer_remote = self.transport.copy_bootstrap_artifact(
                installer, run_id=self.run_id, role="installer"
            )
            package_remote = self.transport.copy_bootstrap_artifact(
                package, run_id=self.run_id, role="package"
            )
            self.journal.mark("bootstrap-staging", "COMPLETE")
            self.journal.mark("out-of-band-bootstrap", "RUNNING")
            remote = self.transport.bootstrap_updater(
                run_id=self.run_id,
                installer_path=installer_remote,
                package_path=package_remote,
                installer_sha256=installer_sha,
                package_sha256=package_sha,
                expected_version=expected_version,
            )
            self.journal.mark("out-of-band-bootstrap", "COMPLETE")
        except BaseException:
            if (
                self.journal.state["steps"].get("bootstrap-staging", {}).get("status")
                == "RUNNING"
            ):
                self.journal.mark("bootstrap-staging", "FAILED")
            if (
                self.journal.state["steps"]
                .get("out-of-band-bootstrap", {})
                .get("status")
                == "RUNNING"
            ):
                self.journal.mark("out-of-band-bootstrap", "FAILED")
            raise
        finally:
            self.journal.mark("bootstrap-staging-cleanup", "RUNNING")
            try:
                self.transport.discard_bootstrap_staging(run_id=self.run_id)
            except BaseException:
                self.journal.mark("bootstrap-staging-cleanup", "FAILED")
                raise
            else:
                self.journal.mark("bootstrap-staging-cleanup", "COMPLETE")
        expected_fields = {
            "schemaVersion",
            "protocolVersion",
            "runId",
            "outOfBandBootstrap",
            "otaEvidence",
            "previousPackageVersion",
            "installedPackageVersion",
            "packageSha256",
            "installerSha256",
            "updaterCodeVersion",
            "boardToolSha256",
            "currentSlot",
            "servicesInactive",
        }
        if set(remote) != expected_fields or any(
            (
                type(remote.get("schemaVersion")) is not int,
                remote.get("schemaVersion") != 1,
                remote.get("protocolVersion") != PROTOCOL_VERSION,
                remote.get("runId") != self.run_id,
                remote.get("outOfBandBootstrap") is not True,
                remote.get("otaEvidence") is not False,
                remote.get("installedPackageVersion") != expected_version,
                remote.get("packageSha256") != package_sha,
                remote.get("installerSha256") != installer_sha,
                remote.get("updaterCodeVersion") != REQUIRED_UPDATER_VERSION,
                remote.get("boardToolSha256") != local_tool_sha,
                remote.get("servicesInactive") is not True,
                re.fullmatch(
                    r"(?:bootstrap/[0-9A-Za-z.+-]{1,64}|releases/[0-9a-f]{64})",
                    str(remote.get("currentSlot") or ""),
                )
                is None,
            )
        ):
            raise RunnerError("out-of-band bootstrap evidence is invalid")
        previous_version = remote.get("previousPackageVersion")
        if previous_version is not None and (
            not isinstance(previous_version, str)
            or not previous_version
            or len(previous_version) > 128
        ):
            raise RunnerError("previous bootstrap package version is invalid")
        identity = self._call("bootstrap-tool-identity", "identity")
        if any(
            (
                identity.get("protocolVersion") != PROTOCOL_VERSION,
                identity.get("toolPath") != REMOTE_TOOL,
                identity.get("toolSha256") != local_tool_sha,
                identity.get("rootOwned") is not True,
                identity.get("mode") != "0755",
            )
        ):
            raise RunnerError(
                "bootstrap board tool identity does not match the package"
            )
        evidence = {
            **remote,
            "boardToolIdentityVerified": True,
        }
        assert_no_secret_material(evidence)
        evidence_path = self.output_dir / "bootstrap-evidence.json"
        if evidence_path.exists() or evidence_path.is_symlink():
            persisted = strict_json(
                read_regular(evidence_path, MAX_OUTPUT_BYTES),
                label="bootstrap evidence",
            )
        else:
            atomic_json(evidence_path, evidence, immutable=True)
            persisted = strict_json(
                read_regular(evidence_path, MAX_OUTPUT_BYTES),
                label="bootstrap evidence",
            )
        if set(persisted) != {*expected_fields, "boardToolIdentityVerified"} or any(
            (
                type(persisted.get("schemaVersion")) is not int,
                persisted.get("schemaVersion") != 1,
                persisted.get("protocolVersion") != PROTOCOL_VERSION,
                persisted.get("runId") != self.run_id,
                persisted.get("outOfBandBootstrap") is not True,
                persisted.get("otaEvidence") is not False,
                persisted.get("installedPackageVersion") != expected_version,
                persisted.get("packageSha256") != package_sha,
                persisted.get("installerSha256") != installer_sha,
                persisted.get("updaterCodeVersion") != REQUIRED_UPDATER_VERSION,
                persisted.get("boardToolSha256") != local_tool_sha,
                persisted.get("boardToolIdentityVerified") is not True,
                persisted.get("servicesInactive") is not True,
                re.fullmatch(
                    r"(?:bootstrap/[0-9A-Za-z.+-]{1,64}|releases/[0-9a-f]{64})",
                    str(persisted.get("currentSlot") or ""),
                )
                is None,
            )
        ):
            raise RunnerError("persisted bootstrap evidence is invalid")
        persisted_previous = persisted.get("previousPackageVersion")
        if persisted_previous is not None and (
            not isinstance(persisted_previous, str)
            or not persisted_previous
            or len(persisted_previous) > 128
        ):
            raise RunnerError("persisted previous package version is invalid")
        assert_no_secret_material(persisted)
        return {
            "schemaVersion": 1,
            "runId": self.run_id,
            "bootstrapComplete": True,
            "outOfBandBootstrap": True,
            "otaEvidence": False,
            "bootstrapEvidenceSha256": hashlib.sha256(
                read_regular(evidence_path, MAX_OUTPUT_BYTES)
            ).hexdigest(),
        }

    def run(
        self,
        *,
        local_tool: Path,
        command_keyring: Path,
        release_keyring: Path,
        health_keyring: Path,
        device_binding: Path,
        manifest_arguments: Mapping[str, Any],
        wait_seconds: int,
        poll_seconds: float,
    ) -> dict[str, object]:
        self._run_cleanup_required = False
        try:
            return self._run_once(
                local_tool=local_tool,
                command_keyring=command_keyring,
                release_keyring=release_keyring,
                health_keyring=health_keyring,
                device_binding=device_binding,
                manifest_arguments=manifest_arguments,
                wait_seconds=wait_seconds,
                poll_seconds=poll_seconds,
            )
        except BaseException as primary:
            if not self._run_cleanup_required:
                raise
            try:
                self.cleanup()
            except BaseException as cleanup_failure:
                try:
                    self.journal.mark("run-recovery-pending", "FAILED")
                except BaseException:
                    pass
                raise RunnerError(
                    "Fleet run failed after preflight and full cleanup did not converge "
                    f"(primary={type(primary).__name__}, "
                    f"cleanup={type(cleanup_failure).__name__})"
                ) from primary
            raise
        finally:
            self._run_cleanup_required = False

    def _run_once(
        self,
        *,
        local_tool: Path,
        command_keyring: Path,
        release_keyring: Path,
        health_keyring: Path,
        device_binding: Path,
        manifest_arguments: Mapping[str, Any],
        wait_seconds: int,
        poll_seconds: float,
    ) -> dict[str, object]:
        if (
            isinstance(wait_seconds, bool)
            or not isinstance(wait_seconds, int)
            or not 30 <= wait_seconds <= 7200
            or isinstance(poll_seconds, bool)
            or not isinstance(poll_seconds, (int, float))
            or not 0.5 <= float(poll_seconds) <= 30
        ):
            raise RunnerError("E2E wait/poll bounds are invalid")
        identity = self._call("identity", "identity")
        local_tool_sha = hashlib.sha256(
            read_regular(local_tool, MAX_OUTPUT_BYTES)
        ).hexdigest()
        if (
            identity.get("protocolVersion") != PROTOCOL_VERSION
            or identity.get("toolSha256") != local_tool_sha
            or identity.get("toolPath") != REMOTE_TOOL
            or identity.get("rootOwned") is not True
            or identity.get("mode") != "0755"
        ):
            raise RunnerError(
                "packaged board tool identity does not match local source"
            )
        sources = {
            "command": command_keyring,
            "release": release_keyring,
            "health": health_keyring,
            "binding": device_binding,
        }
        payloads = {
            role: read_regular(path, MAX_INPUT_BYTES) for role, path in sources.items()
        }
        digests = {
            role: hashlib.sha256(payload).hexdigest()
            for role, payload in payloads.items()
        }
        manifest_path = self.output_dir / "immutable-manifest.json"
        manifest_exists = manifest_path.exists() or manifest_path.is_symlink()
        persisted_manifest = (
            strict_json(
                read_regular(manifest_path, MAX_INPUT_BYTES),
                label="immutable manifest",
            )
            if manifest_exists
            else None
        )
        if persisted_manifest is not None:
            validate_manifest(persisted_manifest)
        # Board preflight claims the persistent run lease before performing
        # hardware/foundation checks. Arm host cleanup before dispatch so a
        # lost response or any later local error cannot strand that lease.
        self._run_cleanup_required = True
        foundation = self._call("foundation", "preflight", ["--run-id", self.run_id])
        live_foundation = foundation.get("foundation")
        live_slots = (
            live_foundation.get("slots")
            if isinstance(live_foundation, Mapping)
            else None
        )
        recorded_baseline = foundation.get("recordedBaseline")
        if (
            foundation.get("verified") is not True
            or not isinstance(live_slots, Mapping)
            or not isinstance(live_slots.get("current"), str)
            or not isinstance(live_slots.get("currentVersion"), str)
            or not isinstance(recorded_baseline, Mapping)
            or not isinstance(recorded_baseline.get("slot"), str)
            or not isinstance(recorded_baseline.get("version"), str)
            or not isinstance(foundation.get("transactionPresent"), bool)
        ):
            raise RunnerError("physical foundation response is invalid")
        resolved_manifest_arguments = dict(manifest_arguments)
        if persisted_manifest is None:
            if (
                live_slots.get("current")
                != manifest_arguments.get("expected_previous_slot")
                or recorded_baseline.get("slot") != live_slots.get("current")
                or recorded_baseline.get("version") != live_slots.get("currentVersion")
                or foundation.get("transactionPresent") is not False
            ):
                raise RunnerError(
                    "live baseline slot does not match the requested immutable run"
                )
            resolved_manifest_arguments["expected_previous_version"] = live_slots[
                "currentVersion"
            ]
        else:
            scenario = persisted_manifest.get("scenario")
            if not isinstance(scenario, Mapping) or not isinstance(
                scenario.get("expectedPreviousVersion"), str
            ):
                raise RunnerError("persisted immutable manifest baseline is invalid")
            resolved_manifest_arguments["expected_previous_version"] = scenario[
                "expectedPreviousVersion"
            ]
        expected_manifest = build_manifest(
            run_id=self.run_id,
            tool_sha256=local_tool_sha,
            input_digests={
                "commandSha256": digests["command"],
                "releaseSha256": digests["release"],
                "healthSha256": digests["health"],
                "bindingSha256": digests["binding"],
            },
            **resolved_manifest_arguments,
        )
        if persisted_manifest is None:
            manifest = expected_manifest
            atomic_json(manifest_path, manifest, immutable=True)
        else:
            manifest = persisted_manifest
            scenario = manifest["scenario"]
            if manifest != expected_manifest or (
                recorded_baseline.get("slot") != scenario["expectedPreviousSlot"]
                or recorded_baseline.get("version")
                != scenario["expectedPreviousVersion"]
                or (
                    foundation.get("transactionPresent") is False
                    and (
                        live_slots.get("current") != scenario["expectedPreviousSlot"]
                        or live_slots.get("currentVersion")
                        != scenario["expectedPreviousVersion"]
                    )
                )
            ):
                raise RunnerError(
                    "persisted immutable manifest does not match live run ownership"
                )
        manifest_payload = read_regular(manifest_path, MAX_INPUT_BYTES)
        manifest_sha = hashlib.sha256(manifest_payload).hexdigest()
        self._call("backup", "backup", ["--run-id", self.run_id], timeout=1800)
        remote_paths: dict[str, str] = {}
        discard_arguments = [
            "--run-id",
            self.run_id,
            "--command-sha256",
            digests["command"],
            "--release-sha256",
            digests["release"],
            "--health-sha256",
            digests["health"],
            "--binding-sha256",
            digests["binding"],
            "--manifest-sha256",
            manifest_sha,
        ]
        try:
            for role, source in {**sources, "manifest": manifest_path}.items():
                remote_paths[role] = self.transport.copy_input(
                    source, run_id=self.run_id, role=role
                )
            self._call(
                "trust",
                "enable-fleet",
                [
                    "--run-id",
                    self.run_id,
                    "--command-keyring",
                    remote_paths["command"],
                    "--command-sha256",
                    digests["command"],
                    "--release-keyring",
                    remote_paths["release"],
                    "--release-sha256",
                    digests["release"],
                    "--health-keyring",
                    remote_paths["health"],
                    "--health-sha256",
                    digests["health"],
                    "--device-binding",
                    remote_paths["binding"],
                    "--binding-sha256",
                    digests["binding"],
                    "--manifest",
                    remote_paths["manifest"],
                    "--manifest-sha256",
                    manifest_sha,
                ],
                timeout=180,
            )
        except BaseException:
            try:
                self._call(
                    "discard-staging", "discard-staging", discard_arguments
                )
            except BaseException:
                pass
            raise
        self._call("discard-staging", "discard-staging", discard_arguments)
        scenario = manifest["scenario"]
        deadline = self.monotonic() + wait_seconds
        if scenario["type"] == "oak-fault-rollback":
            expected_candidate = "releases/" + scenario["expectedBomDigest"][7:]
            rollback_already_complete = False
            while True:
                readiness = self._call(
                    "fault-readiness", "evidence", ["--run-id", self.run_id]
                )
                if readiness.get("complete") is True:
                    validate_final_evidence(readiness, manifest)
                    rollback_already_complete = True
                    break
                update = (
                    readiness.get("updater", {}).get("update")
                    if isinstance(readiness.get("updater"), Mapping)
                    else None
                )
                slots = readiness.get("slots")
                gates = readiness.get("gates")
                faultable = (
                    isinstance(update, Mapping)
                    and isinstance(slots, Mapping)
                    and isinstance(gates, Mapping)
                    and all(
                        gates.get(name) is True
                        for name in (
                            "foundation",
                            "backup",
                            "trust",
                            "updater2",
                            "oak",
                            "services",
                        )
                    )
                    and update.get("commandId") == scenario["expectedCommandId"]
                    and update.get("bomDigest") == scenario["expectedBomDigest"]
                    and update.get("candidateSlot") == scenario["expectedCandidateSlot"]
                    and update.get("phase")
                    in {
                        "ACTIVATING",
                        "BOOT_HEALTHY",
                        "FUNCTIONAL_HEALTHY",
                        "COMMIT_GATE",
                    }
                    and slots.get("current") == expected_candidate
                )
                if faultable:
                    break
                if self.monotonic() >= deadline:
                    raise RunnerError(
                        "timed out waiting for the exact OTA candidate fault window"
                    )
                self.sleeper(float(poll_seconds))
            if not rollback_already_complete:
                self._call(
                    "oak-fault",
                    "arm-oak-fault",
                    ["--run-id", self.run_id],
                    timeout=int(scenario["holdSeconds"]) + 90,
                )
        while True:
            evidence = self._call("evidence", "evidence", ["--run-id", self.run_id])
            if evidence.get("complete") is True:
                validate_final_evidence(evidence, manifest)
                break
            if self.monotonic() >= deadline:
                raise RunnerError("timed out waiting for exact final OTA evidence")
            self.sleeper(float(poll_seconds))
        evidence_path = self.output_dir / "evidence.json"
        created_evidence = False
        try:
            if evidence_path.exists() or evidence_path.is_symlink():
                persisted_evidence = strict_json(
                    read_regular(evidence_path, MAX_OUTPUT_BYTES),
                    label="final evidence",
                )
            else:
                atomic_json(evidence_path, evidence, immutable=True)
                created_evidence = True
                persisted_evidence = strict_json(
                    read_regular(evidence_path, MAX_OUTPUT_BYTES),
                    label="final evidence",
                )
            # A resumed run still obtains fresh live proof above. Preserve an
            # already-written immutable historical artifact when it proves the
            # exact same manifest, even if generatedAt/PIDs have advanced.
            validate_final_evidence(persisted_evidence, manifest)
        except BaseException:
            if created_evidence:
                try:
                    evidence_path.unlink()
                except FileNotFoundError:
                    pass
            raise
        return {
            "schemaVersion": 1,
            "runId": self.run_id,
            "complete": True,
            "scenario": scenario["type"],
            "evidenceSha256": hashlib.sha256(
                read_regular(evidence_path, MAX_OUTPUT_BYTES)
            ).hexdigest(),
        }

    def candidate_soak(
        self,
        *,
        local_tool: Path,
        local_harness: Path,
        candidate_bundle: Path,
        candidate_bom: Path,
    ) -> dict[str, object]:
        candidate_inputs = (
            local_tool,
            local_harness,
            candidate_bundle,
            candidate_bom,
        )
        if any(not path.is_absolute() or path != path.resolve() for path in candidate_inputs):
            raise RunnerError("candidate soak inputs must be normalized absolute paths")
        manifest_path = self.output_dir / "immutable-manifest.json"
        fleet_evidence_path = self.output_dir / "evidence.json"
        manifest_raw = read_regular(manifest_path, MAX_INPUT_BYTES)
        manifest = validate_manifest(
            strict_canonical_json(
                manifest_raw,
                label="immutable manifest",
            )
        )
        fleet_evidence_raw = read_regular(fleet_evidence_path, MAX_OUTPUT_BYTES)
        fleet_evidence = strict_json(fleet_evidence_raw, label="Fleet evidence")
        validate_final_evidence(fleet_evidence, manifest)
        if manifest["scenario"]["type"] != "oak-fault-rollback":
            raise RunnerError("candidate soak requires a completed rollback run")
        identity = self._call("identity", "identity")
        tool_sha256 = sha256_regular(local_tool, MAX_OUTPUT_BYTES)
        if (
            identity.get("protocolVersion") != PROTOCOL_VERSION
            or identity.get("toolSha256") != tool_sha256
            or identity.get("toolPath") != REMOTE_TOOL
            or identity.get("rootOwned") is not True
            or identity.get("mode") != "0755"
        ):
            raise RunnerError("packaged board tool identity does not match local source")
        bundle_sha256 = sha256_regular(
            candidate_bundle, MAX_CANDIDATE_BUNDLE_BYTES
        )
        bundle_size = candidate_bundle.lstat().st_size
        bom_payload = read_regular(candidate_bom, MAX_INPUT_BYTES)
        bom_sha256 = hashlib.sha256(bom_payload).hexdigest()
        harness_sha256 = sha256_regular(local_harness, MAX_OUTPUT_BYTES)
        validate_candidate_inputs(
            bom_payload=bom_payload,
            bundle_sha256=bundle_sha256,
            bundle_size=bundle_size,
            manifest=manifest,
        )
        digests = {
            "bundle": bundle_sha256,
            "bom": bom_sha256,
            "harness": harness_sha256,
        }
        sources = {
            "candidate-bundle": candidate_bundle,
            "candidate-bom": candidate_bom,
            "oak-harness": local_harness,
        }
        remote: dict[str, str] = {}
        result: dict[str, Any] | None = None
        primary: BaseException | None = None
        try:
            self.journal.mark("candidate-staging", "RUNNING")
            for role, source in sources.items():
                remote[role] = self.transport.copy_candidate_input(
                    source, run_id=self.run_id, role=role
                )
            self.journal.mark("candidate-staging", "COMPLETE")
            result = self._call(
                "candidate-soak",
                "candidate-soak",
                [
                    "--run-id",
                    self.run_id,
                    "--candidate-bundle",
                    remote["candidate-bundle"],
                    "--bundle-sha256",
                    bundle_sha256,
                    "--candidate-bom",
                    remote["candidate-bom"],
                    "--bom-sha256",
                    bom_sha256,
                    "--oak-harness",
                    remote["oak-harness"],
                    "--harness-sha256",
                    harness_sha256,
                ],
                timeout=900,
            )
        except BaseException as exc:
            primary = exc
            try:
                self.journal.mark("candidate-staging", "FAILED")
            except BaseException:
                pass
        staging_cleanup_failure: BaseException | None = None
        try:
            self._call(
                "candidate-staging-cleanup",
                "discard-candidate-staging",
                [
                    "--run-id",
                    self.run_id,
                    "--bundle-sha256",
                    bundle_sha256,
                    "--bom-sha256",
                    bom_sha256,
                    "--harness-sha256",
                    harness_sha256,
                ],
            )
        except BaseException as exc:
            staging_cleanup_failure = exc
        cleanup_evidence: dict[str, Any] | None = None
        full_cleanup_failure: BaseException | None = None
        try:
            # The board deadman remains armed across the candidate RPC. Always
            # converge the full transaction even when the RPC response was
            # lost, malformed, or reported a candidate failure.
            cleanup_evidence = self.cleanup()
        except BaseException as exc:
            full_cleanup_failure = exc
        if primary is not None:
            raise RunnerError(
                "candidate soak failed; staging cleanup="
                + ("failed" if staging_cleanup_failure is not None else "complete")
                + "; full cleanup="
                + ("failed" if full_cleanup_failure is not None else "complete")
            ) from primary
        if staging_cleanup_failure is not None:
            raise RunnerError(
                "candidate soak staging cleanup failed; full cleanup="
                + ("failed" if full_cleanup_failure is not None else "complete")
            ) from staging_cleanup_failure
        if full_cleanup_failure is not None:
            raise RunnerError("candidate soak full cleanup failed") from full_cleanup_failure
        if cleanup_evidence is None:
            raise RunnerError("candidate soak cleanup returned no evidence")
        if result is None:
            raise RunnerError("candidate soak returned no evidence")
        fleet_evidence_sha256 = hashlib.sha256(fleet_evidence_raw).hexdigest()
        raw = result.get("rawEvidence")
        persisted_raw_sha256: str | None = None
        if isinstance(raw, Mapping):
            raw_path = self.output_dir / "candidate-soak-raw.json"
            if raw_path.exists() or raw_path.is_symlink():
                raw_bytes = read_regular(raw_path, MAX_OUTPUT_BYTES)
                persisted_raw = strict_json(
                    raw_bytes,
                    label="candidate soak raw evidence",
                )
                if persisted_raw != dict(raw):
                    raise RunnerError("persisted candidate raw evidence differs")
            else:
                atomic_json(raw_path, raw, immutable=True)
                raw_bytes = read_regular(raw_path, MAX_OUTPUT_BYTES)
            persisted_raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
            if persisted_raw_sha256 != result.get("rawEvidenceSha256"):
                raise RunnerError("persisted candidate raw evidence bytes differ")
        validate_candidate_soak_evidence(
            result,
            run_id=self.run_id,
            manifest=manifest,
            bundle_sha256=bundle_sha256,
            bom_sha256=bom_sha256,
            harness_sha256=harness_sha256,
            fleet_evidence_sha256=fleet_evidence_sha256,
            raw_evidence_sha256=persisted_raw_sha256,
            require_cleanup_evidence=False,
        )
        cleanup_path = self.output_dir / "cleanup-evidence.json"
        cleanup_bytes = read_regular(cleanup_path, MAX_OUTPUT_BYTES)
        if strict_json(cleanup_bytes, label="cleanup evidence") != cleanup_evidence:
            raise RunnerError("persisted cleanup evidence differs")
        cleanup_sha256 = hashlib.sha256(cleanup_bytes).hexdigest()
        result = {
            **result,
            "cleanupEvidenceSha256": cleanup_sha256,
            "cleanupEvidence": cleanup_evidence,
        }
        validate_candidate_soak_evidence(
            result,
            run_id=self.run_id,
            manifest=manifest,
            bundle_sha256=bundle_sha256,
            bom_sha256=bom_sha256,
            harness_sha256=harness_sha256,
            fleet_evidence_sha256=fleet_evidence_sha256,
            raw_evidence_sha256=persisted_raw_sha256,
            cleanup_evidence_sha256=cleanup_sha256,
            require_cleanup_evidence=True,
            manifest_raw=manifest_raw,
            fleet_evidence_raw=fleet_evidence_raw,
        )
        evidence_path = self.output_dir / "candidate-soak-evidence.json"
        if evidence_path.exists() or evidence_path.is_symlink():
            persisted = strict_json(
                read_regular(evidence_path, MAX_OUTPUT_BYTES),
                label="candidate soak evidence",
            )
            if persisted != result:
                raise RunnerError("persisted candidate soak evidence differs")
        else:
            atomic_json(evidence_path, result, immutable=True)
        return {
            "schemaVersion": 1,
            "runId": self.run_id,
            "complete": True,
            "passed": result["outcome"]["status"] == "passed",
            "candidateSoakEvidenceSha256": hashlib.sha256(
                read_regular(evidence_path, MAX_OUTPUT_BYTES)
            ).hexdigest(),
            "rawEvidenceSha256": result.get("rawEvidenceSha256"),
        }

    def cleanup(self) -> dict[str, Any]:
        result = self._call(
            "cleanup",
            "cleanup",
            ["--run-id", self.run_id],
            timeout=180,
            validate=lambda result: validate_cleanup_result(
                result,
                run_id=self.run_id,
            ),
        )
        manifest_path = self.output_dir / "immutable-manifest.json"
        fleet_evidence_path = self.output_dir / "evidence.json"
        evidence_path = self.output_dir / "cleanup-evidence.json"
        manifest_present = manifest_path.exists() or manifest_path.is_symlink()
        fleet_evidence_present = (
            fleet_evidence_path.exists() or fleet_evidence_path.is_symlink()
        )
        persisted_present = evidence_path.exists() or evidence_path.is_symlink()
        if persisted_present:
            persisted_bytes = read_regular(evidence_path, MAX_OUTPUT_BYTES)
            persisted = strict_canonical_json(
                persisted_bytes, label="cleanup evidence"
            )
            if persisted.get("schemaVersion") == 2:
                if not (manifest_present and fleet_evidence_present):
                    raise RunnerError("bound cleanup inputs are unavailable")
                validate_bound_cleanup_evidence(
                    persisted,
                    run_id=self.run_id,
                    manifest_raw=read_regular(manifest_path, MAX_INPUT_BYTES),
                    fleet_evidence_raw=read_regular(
                        fleet_evidence_path, MAX_OUTPUT_BYTES
                    ),
                )
            else:
                validate_cleanup_result(persisted, run_id=self.run_id)
                if persisted != canonical_cleanup_evidence(
                    persisted, run_id=self.run_id
                ):
                    raise RunnerError("persisted cleanup evidence is not canonical")
            return persisted
        evidence: dict[str, Any]
        if manifest_present and fleet_evidence_present:
            try:
                manifest_raw = read_regular(manifest_path, MAX_INPUT_BYTES)
                fleet_evidence_raw = read_regular(
                    fleet_evidence_path, MAX_OUTPUT_BYTES
                )
                evidence = build_bound_cleanup_evidence(
                    result,
                    run_id=self.run_id,
                    manifest_raw=manifest_raw,
                    fleet_evidence_raw=fleet_evidence_raw,
                    completed_at=utc_now(),
                )
            except (RunnerError, KeyError, TypeError):
                # Cleanup itself must remain available for crash recovery. An
                # invalid/incomplete local chain receives only a schema-v1
                # operational receipt, which release validation rejects.
                evidence = canonical_cleanup_evidence(result, run_id=self.run_id)
        else:
            # Failure cleanup can legitimately happen before a manifest or
            # terminal result exists. Such a receipt is operational evidence,
            # but schema v1 can never satisfy the release readiness validator.
            evidence = canonical_cleanup_evidence(result, run_id=self.run_id)
        atomic_json(evidence_path, evidence, immutable=True)
        persisted_bytes = read_regular(evidence_path, MAX_OUTPUT_BYTES)
        if persisted_bytes != canonical_json_bytes(evidence):
            raise RunnerError("cleanup evidence read-back digest differs")
        return evidence

    def resume_boot_gate(self) -> dict[str, Any]:
        result = self.transport.invoke_board(
            "resume-boot-gate", timeout=600
        )
        expected = {
            "schemaVersion": 1,
            "kind": "nuvion-iq9075-boot-gate-resumption",
            "complete": True,
            "gateActive": True,
            "protectedUnitsStopped": True,
        }
        if result != expected:
            raise RunnerError("board boot reconciliation gate did not resume")
        output = self.output_dir / "boot-gate-recovery.json"
        atomic_json(output, result, immutable=True)
        if read_regular(output, MAX_OUTPUT_BYTES) != canonical_json_bytes(result):
            raise RunnerError("boot gate recovery evidence read-back differs")
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the IQ9075 Fleet E2E harness")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--known-hosts", required=True)
    parser.add_argument("--host-key-sha256", default=DEFAULT_FINGERPRINT)
    parser.add_argument("--ssh-password-fd", type=int)
    parser.add_argument("--sudo-password-fd", type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True)
    subcommands = parser.add_subparsers(dest="command", required=True)
    run = subcommands.add_parser("run")
    run.add_argument(
        "--local-board-tool",
        default=str(Path(__file__).with_name("iq9075-board-e2e.py")),
    )
    run.add_argument("--command-keyring", required=True)
    run.add_argument("--release-keyring", required=True)
    run.add_argument("--health-keyring", required=True)
    run.add_argument("--device-binding", required=True)
    run.add_argument(
        "--scenario", choices=("commit", "oak-fault-rollback"), required=True
    )
    run.add_argument("--expected-command-id", required=True)
    run.add_argument("--expected-bom-digest", required=True)
    run.add_argument("--expected-candidate-slot", required=True)
    run.add_argument("--expected-previous-slot", required=True)
    run.add_argument("--hold-seconds", type=int, default=0)
    run.add_argument("--device-id", required=True)
    run.add_argument("--space-id", type=int, required=True)
    run.add_argument("--agent-version", required=True)
    run.add_argument("--release-sequence", type=int, required=True)
    run.add_argument("--artifact-digest", required=True)
    run.add_argument("--component-sha", required=True)
    run.add_argument("--config-schema", required=True)
    run.add_argument("--publisher-key-id", required=True)
    run.add_argument("--wait-seconds", type=int, default=900)
    run.add_argument("--poll-seconds", type=float, default=2.0)
    candidate = subcommands.add_parser("candidate-soak")
    candidate.add_argument(
        "--local-board-tool",
        default=str(Path(__file__).with_name("iq9075-board-e2e.py")),
    )
    candidate.add_argument(
        "--local-oak-harness",
        default=str(Path(__file__).with_name("test-iq9075.sh")),
    )
    candidate.add_argument("--candidate-bundle", required=True)
    candidate.add_argument("--candidate-bom", required=True)
    bootstrap = subcommands.add_parser("bootstrap-updater")
    bootstrap.add_argument(
        "--local-board-tool",
        default=str(Path(__file__).with_name("iq9075-board-e2e.py")),
    )
    bootstrap.add_argument(
        "--installer",
        default=str(Path(__file__).with_name("install-iq9075.sh")),
    )
    bootstrap.add_argument("--package", required=True)
    bootstrap.add_argument("--expected-version", required=True)
    bootstrap.add_argument("--expected-sha256", required=True)
    subcommands.add_parser("cleanup")
    subcommands.add_parser("resume-boot-gate")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        disable_core_dumps()
    except RunnerError as exc:
        print(f"run-iq9075-fleet-e2e: {exc}", file=sys.stderr)
        return 1
    arguments = build_parser().parse_args(argv)
    ssh_password: bytearray | None = None
    sudo_password: bytearray | None = None
    try:
        run_id = canonical_run_id(arguments.run_id)
        output_dir = prepare_output_dir(arguments.output_dir)
        if output_dir.name != run_id:
            raise RunnerError("run output directory basename must equal runId")
        if arguments.ssh_password_fd is not None:
            ssh_password = read_secret_fd(arguments.ssh_password_fd)
        if arguments.sudo_password_fd is not None:
            if (
                arguments.sudo_password_fd == arguments.ssh_password_fd
                and ssh_password is not None
            ):
                sudo_password = bytearray(ssh_password)
            else:
                sudo_password = read_secret_fd(arguments.sudo_password_fd)
        pinned = output_dir / "known_hosts"
        fingerprint = create_pinned_known_hosts(
            arguments.known_hosts,
            pinned,
            host=arguments.host,
            port=arguments.port,
            expected=arguments.host_key_sha256,
        )
        transport = OpenSshTransport(
            host=arguments.host,
            user=arguments.user,
            port=arguments.port,
            pinned_known_hosts=pinned,
            expected_fingerprint=fingerprint,
            ssh_password=ssh_password,
            sudo_password=sudo_password,
        )
        journal = HostJournal(
            output_dir / "journal.json",
            run_id=run_id,
            host=arguments.host,
            fingerprint=fingerprint,
        )
        runner = FleetRunner(
            transport=transport,
            journal=journal,
            output_dir=output_dir,
            run_id=run_id,
        )
        if arguments.command == "resume-boot-gate":
            result = runner.resume_boot_gate()
        elif arguments.command == "cleanup":
            result = runner.cleanup()
        elif arguments.command == "bootstrap-updater":
            inputs = [
                Path(arguments.local_board_tool),
                Path(arguments.installer),
                Path(arguments.package),
            ]
            validate_paths_distinct(output_dir, inputs)
            result = runner.bootstrap(
                local_tool=inputs[0],
                installer=inputs[1],
                package=inputs[2],
                expected_version=arguments.expected_version,
                expected_package_sha256=arguments.expected_sha256,
            )
        elif arguments.command == "candidate-soak":
            inputs = [
                Path(arguments.local_board_tool),
                Path(arguments.local_oak_harness),
                Path(arguments.candidate_bundle),
                Path(arguments.candidate_bom),
            ]
            validate_paths_distinct(output_dir, inputs)
            result = runner.candidate_soak(
                local_tool=inputs[0],
                local_harness=inputs[1],
                candidate_bundle=inputs[2],
                candidate_bom=inputs[3],
            )
        else:
            inputs = [
                Path(arguments.local_board_tool),
                Path(arguments.command_keyring),
                Path(arguments.release_keyring),
                Path(arguments.health_keyring),
                Path(arguments.device_binding),
            ]
            validate_paths_distinct(output_dir, inputs)
            result = runner.run(
                local_tool=inputs[0],
                command_keyring=inputs[1],
                release_keyring=inputs[2],
                health_keyring=inputs[3],
                device_binding=inputs[4],
                wait_seconds=arguments.wait_seconds,
                poll_seconds=arguments.poll_seconds,
                manifest_arguments={
                    "identity": {
                        "deviceId": arguments.device_id,
                        "spaceId": arguments.space_id,
                        "productModel": "IQ9075_DEV",
                        "platformProfile": "iq9075_dev",
                        "hardwareRevision": "QCS9075-EVK",
                        "architecture": "aarch64",
                        "dockerRequired": False,
                    },
                    "scenario_type": arguments.scenario,
                    "expected_command_id": arguments.expected_command_id,
                    "expected_bom_digest": arguments.expected_bom_digest,
                    "expected_candidate_slot": arguments.expected_candidate_slot,
                    "expected_previous_slot": arguments.expected_previous_slot,
                    "hold_seconds": arguments.hold_seconds,
                    "release": {
                        "agentVersion": arguments.agent_version,
                        "releaseSequence": arguments.release_sequence,
                        "artifactDigest": arguments.artifact_digest,
                        "componentSha": arguments.component_sha,
                        "configSchema": arguments.config_schema,
                        "publisherKeyId": arguments.publisher_key_id,
                    },
                },
            )
        assert_no_secret_material(result)
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        if (
            arguments.command == "candidate-soak"
            and result.get("passed") is not True
        ):
            return 1
        return 0
    except (RunnerError, OSError, ValueError, UnicodeError) as exc:
        print(f"run-iq9075-fleet-e2e: {exc}", file=sys.stderr)
        return 1
    except Exception:  # noqa: BLE001 - credentials must never enter a traceback.
        print("run-iq9075-fleet-e2e: unexpected internal failure", file=sys.stderr)
        return 1
    finally:
        zero_secret(ssh_password)
        zero_secret(sudo_password)


if __name__ == "__main__":
    try:
        if _askpass_entrypoint():
            raise SystemExit(0)
    except RunnerError:
        raise SystemExit(1)
    raise SystemExit(main())
