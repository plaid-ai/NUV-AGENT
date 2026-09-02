#!/usr/bin/env python3
"""Root-owned, crash-reconciling IQ9075 Fleet E2E board primitives."""

from __future__ import annotations

import argparse
import base64
import binascii
import errno
import fcntl
import hashlib
import io
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
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
TOOL_DESTINATION = Path("/usr/local/libexec/nuvion/iq9075-board-e2e.py")

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
    "nuv-agent-updater.service",
    "nuv-agent-updater.socket",
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
        try:
            completed = subprocess.run(
                list(argv),
                input=input_bytes,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise HarnessError("bounded command timed out") from exc
        return CommandResult(
            completed.returncode,
            completed.stdout.decode("utf-8", errors="replace")[: 256 * 1024],
            completed.stderr.decode("utf-8", errors="replace")[: 16 * 1024],
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
            usb_devices=p("/sys/bus/usb/devices"),
            usb_unbind=p("/sys/bus/usb/drivers/usb/unbind"),
            usb_bind=p("/sys/bus/usb/drivers/usb/bind"),
            os_release=p("/etc/os-release"),
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


def read_regular(path: Path, *, maximum: int) -> tuple[bytes, os.stat_result]:
    try:
        before = path.lstat()
    except OSError as exc:
        raise HarnessError(
            f"required regular file is unavailable: {path.name}"
        ) from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise HarnessError(f"unsafe non-regular file: {path.name}")
    if before.st_size > maximum:
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
                if SECRET_KEY_RE.search(str(key)):
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


class BoardHarness:
    def __init__(
        self,
        *,
        paths: BoardPaths | None = None,
        runner: CommandRunner | None = None,
        root_uid: int = 0,
        root_gid: int | None = None,
        nuvion_gid: int | None = None,
        tool_path: str | Path | None = None,
        enforce_installed_tool: bool = True,
        clock: Callable[[], str] = utc_now,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        disk_usage: Callable[[str | Path], Any] = shutil.disk_usage,
        usb_write_hook: Callable[[str, str], None] | None = None,
    ) -> None:
        self.paths = paths or BoardPaths.from_root()
        self.runner = runner or CommandRunner()
        self.root_uid = root_uid
        self.root_gid = root_uid if root_gid is None else root_gid
        self.nuvion_gid = self._nuvion_gid() if nuvion_gid is None else nuvion_gid
        self.tool_path = Path(tool_path or __file__).resolve()
        self.enforce_installed_tool = enforce_installed_tool
        self.clock = clock
        self.monotonic = monotonic
        self.sleeper = sleeper
        self.disk_usage = disk_usage
        self.usb_write_hook = usb_write_hook

    @staticmethod
    def _nuvion_gid() -> int:
        import grp

        try:
            return grp.getgrnam("nuvion").gr_gid
        except KeyError as exc:
            raise HarnessError("required nuvion group is unavailable") from exc

    def _prepare_run(self, run_id: str) -> Path:
        canonical_run_id(run_id)
        base = self.paths.state_root.parent
        ensure_directory(base, mode=0o700, uid=self.root_uid, gid=self.root_gid)
        ensure_directory(
            self.paths.state_root, mode=0o700, uid=self.root_uid, gid=self.root_gid
        )
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
    def _run_lock(
        self,
        run_id: str,
        *,
        usb: bool = False,
        claim: bool = False,
        allow_unclaimed: bool = False,
    ) -> Iterator[None]:
        run_dir = self._prepare_run(run_id)
        self.paths.lock_root.mkdir(parents=True, exist_ok=True)
        if usb:
            with (
                self._lock_file(self.paths.global_fleet_lock),
                self._lock_file(self.paths.global_usb_lock),
                self._lock_file(run_dir / "operation.lock"),
            ):
                if claim:
                    self._claim_active_run(run_id)
                else:
                    self._assert_active_run(run_id, allow_unclaimed=allow_unclaimed)
                yield
        else:
            with (
                self._lock_file(self.paths.global_fleet_lock),
                self._lock_file(run_dir / "operation.lock"),
            ):
                if claim:
                    self._claim_active_run(run_id)
                else:
                    self._assert_active_run(run_id, allow_unclaimed=allow_unclaimed)
                yield

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
        self._assert_active_run(run_id, allow_unclaimed=True)
        if self.paths.active_run.exists():
            return
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
        active = self.runner.run(["/usr/bin/systemctl", "is-active", unit], timeout=10)
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
            "active": active.returncode == 0 and active.stdout.strip() == "active",
            "enabled": enabled.returncode == 0
            and enabled.stdout.strip() in {"enabled", "static", "indirect"},
            "unitFileState": unit_file_state,
            "mainPid": pid,
        }

    def _unit_snapshot(self) -> dict[str, dict[str, object]]:
        return {unit: self._unit_status(unit) for unit in UNITS}

    def _restore_units(self, expected: Mapping[str, object]) -> None:
        for unit in RESTART_ORDER:
            raw = expected.get(unit)
            if not isinstance(raw, Mapping):
                raise HarnessError("saved unit state is invalid")
            unit_file_state = raw.get("unitFileState")
            if unit_file_state == "enabled":
                self._systemctl("enable", unit)
            elif unit_file_state == "disabled":
                self._systemctl("disable", unit)
            elif unit_file_state not in {
                "static",
                "indirect",
                "masked",
                "generated",
                "transient",
                "alias",
            }:
                raise HarnessError("saved unit file state is invalid")
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

    def _stop_writers(self) -> None:
        for unit in STOP_ORDER:
            self._systemctl("stop", unit)
        for unit in UNITS:
            if self._unit_status(unit)["active"] is True:
                raise HarnessError("writer unit did not stop for consistent backup")

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
            vendor_payload, _ = read_regular(vendor_path, maximum=128)
            product_payload, _ = read_regular(product_path, maximum=128)
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
            payload, _ = read_regular(device / name, maximum=128)
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
            self._systemctl("enable", unit)
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
        phase = transaction.get("phase")
        if phase not in PHASES or phase in {"RESTORING", "RESTORED"}:
            raise HarnessError("trust transaction is not applyable")
        try:
            if phase == "APPLIED" and self._transaction_matches(run_id, transaction):
                services = {unit: self._unit_status(unit) for unit in UNITS}
                if all(
                    item["active"] is True
                    and item["enabled"] is True
                    and (not unit.endswith(".service") or int(item["mainPid"]) > 0)
                    for unit, item in services.items()
                ):
                    updater = self._probe_updater()
                    return {"updater": updater, "idempotent": True}
            transaction["phase"] = "APPLYING"
            transaction["applyingAt"] = self.clock()
            state["trustTransaction"] = transaction
            self._save_state(run_id, state)
            self._apply_files(run_id, transaction)
            pids = self._restart_runtime()
            transaction.update(
                {
                    "phase": "APPLIED",
                    "appliedAt": self.clock(),
                    "appliedPids": pids,
                    "liveVerified": True,
                }
            )
            state["trustTransaction"] = transaction
            self._save_state(run_id, state)
            return {"updater": self._probe_updater(), "idempotent": False}
        except BaseException:
            self._stop_agent_fail_closed()
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

    def _deadman_unit(self, run_id: str) -> str:
        return f"nuvion-oak-deadman-{run_id.replace('-', '')}.service"

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
        if result.returncode != 0:
            raise HarnessError("deadman unit did not stop")
        status = self.runner.run(["/usr/bin/systemctl", "is-active", unit], timeout=10)
        if status.stdout.strip() == "active":
            raise HarnessError("deadman unit remains active")
        self.runner.run(["/usr/bin/systemctl", "reset-failed", unit], timeout=10)

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
        self, run_id: str, state: dict[str, Any], transaction: dict[str, Any]
    ) -> None:
        self._validate_transaction_shape(transaction)
        if transaction.get("phase") == "RESTORED":
            return
        transaction["phase"] = "RESTORING"
        transaction["restoringAt"] = self.clock()
        state["trustTransaction"] = transaction
        self._save_state(run_id, state)
        self._stop_agent_fail_closed()
        self._systemctl("stop", "nuv-agent-updater.service")
        self._systemctl("stop", "nuv-agent-updater.socket")
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
        self._restore_units(units_before)
        transaction.update(
            {"phase": "RESTORED", "liveVerified": False, "restoredAt": self.clock()}
        )
        state["trustTransaction"] = transaction
        self._save_state(run_id, state)

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

    def cleanup(self, run_id: str, *, deadman_only: bool = False) -> dict[str, object]:
        if deadman_only:
            return self._deadman_cleanup(run_id)
        with self._run_lock(run_id, usb=True, allow_unclaimed=True):
            state = self._load_state(run_id)
            if state.get("cleanup", {}).get("complete") is True:
                fault = state.get("oakFault")
                transaction = state.get("trustTransaction")
                if (
                    isinstance(fault, Mapping)
                    and fault.get("armed") is True
                ) or (
                    isinstance(transaction, Mapping)
                    and transaction.get("phase") != "RESTORED"
                ):
                    raise HarnessError("completed cleanup journal is not restored")
                self._purge_run_sensitive_material(run_id)
                if self.paths.active_run.exists() or self.paths.active_run.is_symlink():
                    self._release_active_run(run_id)
                return {
                    "schemaVersion": 1,
                    "runId": run_id,
                    "complete": True,
                    "recovered": False,
                    "phase": "RESTORED",
                    "idempotent": True,
                }
            fault = state.get("oakFault")
            recovered = False
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
                if transaction.get("phase") != "RESTORED":
                    try:
                        self._restore_transaction(run_id, state, transaction)
                    except Exception:  # noqa: BLE001 - incomplete cleanup stays fail-closed.
                        self._stop_agent_fail_closed()
                        return {
                            "schemaVersion": 1,
                            "runId": run_id,
                            "complete": False,
                            "recovered": recovered,
                            "phase": "RESTORING",
                        }
            complete = not (
                isinstance(state.get("oakFault"), dict)
                and state["oakFault"].get("armed") is True
            ) and (
                not isinstance(transaction, dict)
                or transaction.get("phase") == "RESTORED"
            )
            if complete:
                self._purge_run_sensitive_material(run_id)
                if self.paths.active_run.exists() or self.paths.active_run.is_symlink():
                    self._release_active_run(run_id)
                state["cleanup"] = {"complete": True, "completedAt": self.clock()}
                self._save_state(run_id, state)
            return {
                "schemaVersion": 1,
                "runId": run_id,
                "complete": complete,
                "recovered": recovered,
                "phase": transaction.get("phase")
                if isinstance(transaction, dict)
                else None,
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
                item["active"] is True and item["enabled"] is True
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
                "schemaVersion": 1,
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
            assert_no_secret_material(result)
            if output is not None:
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
    run_command("arm-oak-fault")
    evidence = run_command("evidence")
    evidence.add_argument("--output")
    cleanup = run_command("cleanup")
    cleanup.add_argument("--deadman-only", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if os.geteuid() != 0:
        print("iq9075-board-e2e: root privileges are required", file=sys.stderr)
        return 1
    arguments = build_parser().parse_args(argv)
    try:
        harness = BoardHarness()
        if arguments.command == "identity":
            result = harness.identity()
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
            elif arguments.command == "cleanup":
                result = harness.cleanup(run_id, deadman_only=arguments.deadman_only)
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
