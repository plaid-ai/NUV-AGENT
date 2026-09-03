from __future__ import annotations

import ast
import base64
import copy
import gzip
import hashlib
import importlib.util
import io
import json
import os
import signal
import socket
import sqlite3
import sys
import tarfile
import tempfile
import time
import unittest
import uuid
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BOARD = load_module(
    "iq9075_board_e2e_v2",
    ROOT / "packaging/dev/iq9075-board-e2e.py",
)
HOST = load_module(
    "run_iq9075_fleet_e2e_v2",
    ROOT / "packaging/dev/run-iq9075-fleet-e2e.py",
)


def persistent_state_evidence(paths: Sequence[str]) -> dict[str, object]:
    roots = {
        path: {
            "exists": False,
            "entries": 0,
            "bytes": 0,
            "sha256": hashlib.sha256(path.encode("utf-8")).hexdigest(),
        }
        for path in paths
    }
    serialized = (
        json.dumps(roots, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    return {
        "schemaVersion": 1,
        "roots": roots,
        "sha256": hashlib.sha256(serialized).hexdigest(),
        "entries": 0,
        "bytes": 0,
    }


def release_tree_evidence(slots: Mapping[str, object]) -> dict[str, object]:
    trees = {
        role: {
            "target": target,
            "exists": True,
            "entries": 1,
            "bytes": 0,
            "sha256": hashlib.sha256(f"{role}:{target}".encode()).hexdigest(),
        }
        for role, target in slots.items()
    }
    serialized = (
        json.dumps(trees, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    return {
        "schemaVersion": 1,
        "slots": trees,
        "sha256": hashlib.sha256(serialized).hexdigest(),
        "entries": len(trees),
        "bytes": 0,
    }


def candidate_execution_proof(
    run_id: str, *, writable_path: str | None = None
) -> dict[str, object]:
    unit = f"nuvion-candidate-soak-{run_id.replace('-', '')}.service"
    control_group = "/system.slice/" + unit
    temporary = {
        path: {
            "mountId": 101 + index,
            "fsType": "tmpfs",
            "sizeBytes": limits["bytes"],
            "inodeLimit": limits["inodes"],
            "readOnly": False,
        }
        for index, (path, limits) in enumerate(BOARD.CANDIDATE_TMPFS_LIMITS.items())
    }
    read_only = {
        path: {
            "mountId": 111 + index,
            "mountPoint": path,
            "readOnly": True,
        }
        for index, path in enumerate(BOARD.CANDIDATE_PERSISTENT_PATHS)
    }
    inaccessible = {
        path: {
            "mountId": 121 + index,
            "mountPoint": path,
            "mode": "0000",
            "readOnly": True,
        }
        for index, path in enumerate(BOARD.CANDIDATE_INACCESSIBLE_PATHS)
    }
    return {
        "schemaVersion": 1,
        "unit": unit,
        "mainPid": 9001,
        "controlGroup": control_group,
        "pidControlGroup": control_group,
        "recursivePopulated": True,
        "uidIsolation": {
            "before": {
                "schemaVersion": 1,
                "uid": 4242,
                "pids": [],
                "controlGroup": None,
                "stableScans": 2,
            },
            "during": {
                "schemaVersion": 1,
                "uid": 4242,
                "pids": [9001],
                "controlGroup": control_group,
                "stableScans": 2,
            },
        },
        "systemdProperties": dict(BOARD.CANDIDATE_SYSTEMD_EXPECTED_PROPERTIES),
        "mountSandbox": {
            "temporaryFilesystems": temporary,
            "readOnlyPaths": read_only,
            "readWritePath": {
                "mountId": 131,
                "mountPoint": writable_path
                or f"/var/lib/nuvion-fleet-e2e/runs/{run_id}",
                "readOnly": False,
            },
            "inaccessiblePaths": inaccessible,
            "totalTmpfsBytes": sum(
                item["sizeBytes"] for item in temporary.values()
            ),
            "totalTmpfsInodes": sum(
                item["inodeLimit"] for item in temporary.values()
            ),
        },
    }


def candidate_termination_proof(run_id: str) -> dict[str, object]:
    unit = f"nuvion-candidate-soak-{run_id.replace('-', '')}.service"
    return {
        "schemaVersion": 1,
        "unit": unit,
        "controlGroup": "/system.slice/" + unit,
        "initialPresent": True,
        "initialPopulated": False,
        "killSignals": [],
        "stopSucceeded": True,
        "resetPerformed": True,
        "recursivePopulated": False,
        "loadState": "not-found",
        "activeState": "inactive",
        "cgroupRemoved": True,
    }


def candidate_collector_proof(run_id: str) -> dict[str, object]:
    unit = f"nuvion-candidate-soak-{run_id.replace('-', '')}.service"
    return {
        "schemaVersion": 1,
        "unit": unit,
        "controlGroup": "/system.slice/" + unit,
        "requiredSeconds": BOARD.CANDIDATE_REQUIRED_SOAK_SECONDS,
        "elapsedSeconds": float(BOARD.CANDIDATE_REQUIRED_SOAK_SECONDS),
        "scanIntervalSeconds": BOARD.CANDIDATE_UID_SCAN_INTERVAL_SECONDS,
        "sampleCount": 2,
        "observedPids": [9001],
        "escapeDetected": None,
        "allSamplesWithinCgroup": True,
        "durationSatisfied": True,
        "terminalStatus": {
            "ActiveState": "active",
            "ExecMainCode": "1",
            "ExecMainStatus": "0",
            "Result": "success",
            "SubState": "exited",
        },
        "afterTermination": candidate_uid_before(),
    }


def cleanup_proof(phase: str | None = "RESTORED") -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "transactionPhase": phase,
        "cleanupJournalComplete": True,
        "activeRunLeaseAbsent": True,
        "transactionSnapshotsAbsent": True,
        "recoveryArchiveAbsent": True,
        "candidateArtifactsAbsent": True,
        "candidateStagingAbsent": True,
        "trustStagingAbsent": True,
    }


def production_restoration_evidence(
    manifest: Mapping[str, object],
) -> dict[str, object]:
    files = {
        path: {
            "exists": False,
            "sha256": None,
            "mode": None,
            "uid": None,
            "gid": None,
        }
        for path in HOST.PRODUCTION_TRANSACTION_FILES
    }
    directories = {
        path: {"mode": 0o700, "uid": 0, "gid": 0}
        for path in HOST.PRODUCTION_TRANSACTION_DIRECTORIES
    }
    units = {
        unit: {
            "active": True,
            "enabled": True,
            "unitFileState": "enabled",
        }
        for unit in HOST.PRODUCTION_UNITS
    }
    value: dict[str, object] = {
        "schemaVersion": 1,
        "transactionPhase": "RESTORED",
        "manifestSha256": hashlib.sha256(
            (
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode()
        ).hexdigest(),
        "files": files,
        "directories": directories,
        "units": units,
    }
    value["sha256"] = hashlib.sha256(
        (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    return value


def candidate_uid_before() -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "uid": 4242,
        "pids": [],
        "controlGroup": None,
        "stableScans": 2,
    }


def seed_candidate_kernel_proof(
    fixture: "HarnessFixture", *, pid: int = 9001
) -> tuple[str, Path]:
    unit = fixture.harness._candidate_unit(fixture.run_id)
    control_group = "/system.slice/" + unit
    cgroup = fixture.paths.cgroup_root / control_group.lstrip("/")
    cgroup.mkdir(parents=True, exist_ok=True)
    (cgroup / "cgroup.events").write_text(
        "populated 1\nfrozen 0\n", encoding="ascii"
    )
    proc = fixture.paths.proc_root / str(pid)
    proc.mkdir(parents=True, exist_ok=True)
    (proc / "cgroup").write_text(f"0::{control_group}\n", encoding="ascii")
    (proc / "status").write_text(
        "Name:\tpython3\nUid:\t4242\t4242\t4242\t4242\n", encoding="ascii"
    )
    writable = str(fixture.paths.state_root / fixture.run_id)
    mount_lines = [
        "1 0 0:1 / / ro - ext4 /dev/root ro",
    ]
    mount_id = 100
    for path, limits in BOARD.CANDIDATE_TMPFS_LIMITS.items():
        mount_id += 1
        mount_lines.append(
            f"{mount_id} 1 0:{mount_id} / {path} rw,nosuid,nodev - "
            f"tmpfs tmpfs rw,size={limits['bytes']},nr_inodes={limits['inodes']}"
        )
    for path in BOARD.CANDIDATE_PERSISTENT_PATHS:
        mount_id += 1
        mount_lines.append(
            f"{mount_id} 1 0:{mount_id} / {path} ro - ext4 /dev/root rw"
        )
    mount_id += 1
    mount_lines.append(
        f"{mount_id} 1 0:{mount_id} / {writable} rw - ext4 /dev/root rw"
    )
    for path in BOARD.CANDIDATE_INACCESSIBLE_PATHS:
        mount_id += 1
        mount_lines.append(
            f"{mount_id} 1 0:{mount_id} / {path} ro - tmpfs tmpfs "
            "ro,size=4096,nr_inodes=1"
        )
        visible = proc / "root" / path.lstrip("/")
        visible.mkdir(parents=True, exist_ok=True)
        visible.chmod(0)
    (proc / "mountinfo").write_text("\n".join(mount_lines) + "\n", encoding="utf-8")
    return control_group, cgroup


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        self.value += 0.01
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


class FakeBoardRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.pid_sequence = 100
        self.units: dict[str, dict[str, object]] = {
            "nuv-agent.service": {"active": True, "enabled": True, "pid": 100},
            "nuv-agent-updater.service": {
                "active": False,
                "enabled": False,
                "pid": 0,
            },
            "nuv-agent-updater.socket": {
                "active": False,
                "enabled": False,
                "pid": 0,
            },
        }
        self.deadmen: dict[str, bool] = {}
        self.updater: dict[str, object] = {
            "capabilityAvailable": True,
            "authenticatedHelper": True,
            "reason": "READY",
            "updaterVersion": "0.2.0",
        }

    def _status(self, unit: str) -> dict[str, object]:
        if unit in self.units:
            return self.units[unit]
        return {"active": self.deadmen.get(unit, False), "enabled": False, "pid": 0}

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        input_bytes: bytes | None = None,
    ):
        del timeout, input_bytes
        call = tuple(argv)
        self.calls.append(call)
        if call[:2] == ("/usr/bin/dpkg", "--print-architecture"):
            return BOARD.CommandResult(0, "arm64\n", "")
        if call and call[0] == "/usr/sbin/runuser":
            return BOARD.CommandResult(0, json.dumps(self.updater) + "\n", "")
        if call and call[0] == "/usr/bin/systemd-run":
            unit = next(
                value.split("=", 1)[1] for value in call if value.startswith("--unit=")
            )
            if any(value.startswith("--on-active=") for value in call):
                self.deadmen[unit] = False
                self.deadmen[unit.removesuffix(".service") + ".timer"] = True
            else:
                self.deadmen[unit] = not unit.startswith("nuvion-candidate-soak-")
            return BOARD.CommandResult(0, "", "")
        if call[:4] == (
            "/usr/bin/busctl",
            "--system",
            "--json=short",
            "call",
        ):
            unit = call[-1]
            escaped = "".join(
                character
                if character.isascii() and character.isalnum()
                else f"_{ord(character):02x}"
                for character in unit
            )
            return BOARD.CommandResult(
                0,
                json.dumps(
                    {
                        "type": "o",
                        "data": [
                            "/org/freedesktop/systemd1/unit/" + escaped
                        ],
                    },
                    separators=(",", ":"),
                )
                + "\n",
                "",
            )
        if call[:4] == (
            "/usr/bin/busctl",
            "--system",
            "--json=short",
            "get-property",
        ):
            return BOARD.CommandResult(
                0,
                '{"type":"t","data":999999999999}\n',
                "",
            )
        if len(call) >= 3 and call[0] == "/usr/bin/systemctl":
            action, unit = call[1], call[-1]
            status = self._status(unit)
            if action == "is-active":
                return BOARD.CommandResult(
                    0 if status["active"] else 3,
                    "active\n" if status["active"] else "inactive\n",
                    "",
                )
            if action == "is-enabled":
                return BOARD.CommandResult(
                    0 if status["enabled"] else 1,
                    "enabled\n" if status["enabled"] else "disabled\n",
                    "",
                )
            if action == "show":
                if "--property=LoadState" in call:
                    names = {
                        item.split("=", 1)[1]
                        for item in call
                        if item.startswith("--property=")
                    }
                    values = {
                        "ActiveState": "inactive",
                        "ControlGroup": "",
                        "LoadState": "not-found",
                    }
                    return BOARD.CommandResult(
                        0,
                        "".join(
                            f"{name}={values[name]}\n" for name in sorted(names)
                        ),
                        "",
                    )
                if "--property=ControlGroup" in call:
                    return BOARD.CommandResult(1, "", "unit not loaded")
                return BOARD.CommandResult(0, f"{status['pid']}\n", "")
            if action == "kill":
                return BOARD.CommandResult(0, "", "")
            if action in {"enable", "disable"}:
                status["enabled"] = action == "enable"
                return BOARD.CommandResult(0, "", "")
            if action in {"start", "restart"}:
                status["active"] = True
                if unit.endswith(".service") and (
                    action == "restart" or int(status["pid"]) == 0
                ):
                    self.pid_sequence += 1
                    status["pid"] = self.pid_sequence
                return BOARD.CommandResult(0, "", "")
            if action == "stop":
                status["active"] = False
                status["pid"] = 0
                if unit in self.deadmen:
                    self.deadmen[unit] = False
                return BOARD.CommandResult(0, "", "")
            if action == "reset-failed":
                return BOARD.CommandResult(0, "", "")
        return BOARD.CommandResult(0, "", "")


class HarnessFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.current_digest = "a" * 64
        self.previous_digest = "d" * 64
        self.run_id = str(uuid.uuid4())
        self.command_id = str(uuid.uuid4())
        self.paths = BOARD.BoardPaths.from_root(root)
        self.runner = FakeBoardRunner()
        self.clock = FakeClock()
        self._seed()
        self.harness = BOARD.BoardHarness(
            paths=self.paths,
            runner=self.runner,
            root_uid=os.getuid(),
            root_gid=os.getgid(),
            nuvion_gid=os.getgid(),
            nuvion_uid=4242,
            tool_path=ROOT / "packaging/dev/iq9075-board-e2e.py",
            enforce_installed_tool=False,
            monotonic=self.clock.monotonic,
            sleeper=self.clock.sleep,
            disk_usage=lambda _path: SimpleNamespace(free=8 * 1024**3),
            usb_write_hook=self._usb_hook,
        )

    def _write(self, absolute: str, payload: str, mode: int = 0o600) -> Path:
        path = self.root / absolute.lstrip("/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
        path.chmod(mode)
        return path

    def _marker(self, digest: str, version: str, sequence: int) -> dict[str, object]:
        return {
            "schemaVersion": 2,
            "bomDigest": f"sha256:{digest}",
            "agentVersion": version,
            "releaseSequence": sequence,
            "artifactDigest": "sha256:"
            + ("b" if digest == self.current_digest else "e") * 64,
            "componentSha": ("c" if digest == self.current_digest else "f") * 40,
            "configSchema": "12",
            "publisherKeyId": "release-iq9075-dev",
        }

    def _seed_release(self, digest: str, version: str, sequence: int) -> None:
        release = self.root / f"opt/nuv-agent/releases/{digest}/.nuvion"
        release.mkdir(parents=True)
        (release / "release.json").write_text(
            json.dumps(self._marker(digest, version, sequence)), encoding="utf-8"
        )

    def _seed(self) -> None:
        self._write("/usr/lib/os-release", 'ID=ubuntu\nVERSION_ID="24.04"\n', 0o644)
        etc_os_release = self.root / "etc/os-release"
        etc_os_release.parent.mkdir(parents=True, exist_ok=True)
        etc_os_release.symlink_to("../usr/lib/os-release")
        self._write("/proc/device-tree/model", "Thundercomm IQ-9075 QCS9075\0", 0o444)
        self._write("/sys/bus/usb/devices/2-1/idVendor", "1d6b\n", 0o444)
        self._write("/sys/bus/usb/devices/2-1/idProduct", "0003\n", 0o444)
        self._write("/sys/bus/usb/devices/2-1/speed", "10000\n", 0o444)
        self._write("/sys/bus/usb/devices/2-1.1/idVendor", "03e7\n", 0o444)
        self._write("/sys/bus/usb/devices/2-1.1/idProduct", "f63b\n", 0o444)
        self._write("/sys/bus/usb/devices/2-1.1/serial", "oak-iq9075-test\n", 0o444)
        self._write("/sys/bus/usb/devices/2-1.1/speed", "5000\n", 0o444)
        self._write("/sys/bus/usb/drivers/usb/unbind", "", 0o600)
        self._write("/sys/bus/usb/drivers/usb/bind", "", 0o600)
        (self.root / "sys/bus/usb/devices/2-1/driver").symlink_to("../../drivers/usb")
        (self.root / "sys/bus/usb/devices/2-1.1/driver").symlink_to("../../drivers/usb")
        self._write(
            "/etc/nuv-agent/agent.env",
            "NUVION_CONFIG_SCHEMA_VERSION=12\n"
            "NUVION_DEVICE_ID=sp-3-nuvion-iq9075\n"
            "NUVION_DEVICE_USERNAME=sp-3-nuvion-iq9075\n"
            "NUVION_SPACE_ID=3\n"
            "NUVION_FLEET_COMMAND_KEYRING_PATH=\n"
            "NUVION_FLEET_COMMAND_ENABLED=false\n"
            "NUVION_DEVICE_PASSWORD=must-stay-private\n",
            0o660,
        )
        self._write(
            "/etc/nuvion-updater/updater.env",
            "NUVION_RELEASE_BASE_URL=https://example\n",
        )
        self._write("/var/lib/nuv-agent/events.sqlite3", "event-state")
        self._write("/var/lib/nuvion-updater/updater.sqlite3", "updater-state")
        self._seed_release(self.current_digest, "0.1.121", 2)
        self._seed_release(self.previous_digest, "0.1.120", 1)
        install = self.root / "opt/nuv-agent"
        (install / "bootstrap/0.1.119").mkdir(parents=True)
        (install / "current").symlink_to(f"releases/{self.previous_digest}")
        (install / "previous").symlink_to("bootstrap/0.1.119")
        self.paths.lock_root.mkdir(parents=True, exist_ok=True)

    def _usb_hook(self, action: str, port: str) -> None:
        driver = self.paths.usb_devices / port / "driver"
        if action == "unbind":
            try:
                driver.unlink()
            except FileNotFoundError:
                pass
        elif not driver.exists():
            driver.symlink_to("../../drivers/usb")

    @staticmethod
    def _keyring(role: str) -> bytes:
        payload: dict[str, object] = {
            "schemaVersion": 1,
            "trustDomain": "iq9075-dev",
            "keys": {
                f"{role}-dev": base64.b64encode(bytes([len(role)]) * 32).decode("ascii")
            },
        }
        if role == "health":
            payload["purpose"] = "agent-update-health-attestation"
        return (
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()

    @staticmethod
    def _binding() -> bytes:
        return (
            json.dumps(
                {
                    "schemaVersion": 1,
                    "trustDomain": "iq9075-dev",
                    "deviceId": "sp-3-nuvion-iq9075",
                    "spaceId": 3,
                    "productModel": "IQ9075_DEV",
                    "platformProfile": "iq9075_dev",
                    "hardwareRevision": "QCS9075-EVK",
                    "architecture": "aarch64",
                    "dockerRequired": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()

    def payloads(self, scenario: str = "commit") -> dict[str, bytes]:
        command = self._keyring("command")
        release = self._keyring("release")
        health = self._keyring("health")
        binding = self._binding()
        inputs = {
            "commandSha256": hashlib.sha256(command).hexdigest(),
            "releaseSha256": hashlib.sha256(release).hexdigest(),
            "healthSha256": hashlib.sha256(health).hexdigest(),
            "bindingSha256": hashlib.sha256(binding).hexdigest(),
        }
        manifest = {
            "schemaVersion": 1,
            "protocolVersion": BOARD.PROTOCOL_VERSION,
            "runId": self.run_id,
            "toolSha256": self.harness.identity()["toolSha256"],
            "inputs": inputs,
            "destinations": BOARD.FIXED_DESTINATIONS,
            "identity": {
                "deviceId": "sp-3-nuvion-iq9075",
                "spaceId": 3,
                "productModel": "IQ9075_DEV",
                "platformProfile": "iq9075_dev",
                "hardwareRevision": "QCS9075-EVK",
                "architecture": "aarch64",
                "dockerRequired": False,
            },
            "scenario": {
                "type": scenario,
                "expectedCommandId": self.command_id,
                "expectedBomDigest": f"sha256:{self.current_digest}",
                "expectedCandidateSlot": f"/opt/nuv-agent/releases/{self.current_digest}",
                "expectedPreviousSlot": f"releases/{self.previous_digest}",
                "expectedPreviousVersion": "0.1.120",
                "holdSeconds": 0 if scenario == "commit" else 10,
                "release": {
                    key: value
                    for key, value in self._marker(
                        self.current_digest, "0.1.121", 2
                    ).items()
                    if key not in {"schemaVersion", "bomDigest"}
                },
            },
        }
        return {
            "command": command,
            "release": release,
            "health": health,
            "binding": binding,
            "manifest": (
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode(),
        }

    def stage(self, payloads: Mapping[str, bytes]) -> dict[str, tuple[Path, str]]:
        result: dict[str, tuple[Path, str]] = {}
        for role, payload in payloads.items():
            path = Path(f"/tmp/nuvion-fleet-e2e-{self.run_id}-{role}.json")
            path.write_bytes(payload)
            path.chmod(0o600)
            result[role] = (path, hashlib.sha256(payload).hexdigest())
        return result

    def enable(self, payloads: Mapping[str, bytes]) -> dict[str, object]:
        staged = self.stage(payloads)
        return self.harness.enable_fleet(
            self.run_id,
            command_keyring=staged["command"][0],
            command_sha256=staged["command"][1],
            release_keyring=staged["release"][0],
            release_sha256=staged["release"][1],
            health_keyring=staged["health"][0],
            health_sha256=staged["health"][1],
            device_binding=staged["binding"][0],
            binding_sha256=staged["binding"][1],
            manifest_path=staged["manifest"][0],
            manifest_sha256=staged["manifest"][1],
        )

    def foundation_backup(self) -> None:
        self.harness.preflight(self.run_id)
        self.harness.backup(self.run_id)

    def activate_candidate(self) -> None:
        current = self.root / "opt/nuv-agent/current"
        previous = self.root / "opt/nuv-agent/previous"
        current.unlink()
        current.symlink_to(f"releases/{self.current_digest}")
        previous.unlink()
        previous.symlink_to(f"releases/{self.previous_digest}")

    def update_state(self, phase: str) -> dict[str, object]:
        state: dict[str, object] = {
            "commandId": self.command_id,
            "sequence": 2,
            "targetVersion": "0.1.121",
            "bomDigest": f"sha256:{self.current_digest}",
            "phase": phase,
            "updatePhase": phase,
            "updatedAt": "2026-09-02T10:02:00Z",
            "commandExpiresAt": "2026-09-02T23:00:00Z",
            "candidateSlot": f"/opt/nuv-agent/releases/{self.current_digest}",
            "previousSlot": f"releases/{self.previous_digest}",
            "previousVersion": "0.1.120",
            "releaseSequence": 2,
            "artifactDigest": "sha256:" + "b" * 64,
            "componentSha": "c" * 40,
            "configSchema": "12",
            "bomVerificationStatus": "VERIFIED",
            "publisherKeyId": "release-iq9075-dev",
        }
        if phase == "COMMITTED":
            state.update(
                {
                    "slot": f"releases/{self.current_digest}",
                    "health": "FUNCTIONAL_HEALTHY",
                    "functionalHealth": "FUNCTIONAL_HEALTHY",
                }
            )
        elif phase == "ROLLED_BACK":
            state.update(
                {
                    "slot": f"releases/{self.previous_digest}",
                    "rollbackSlot": f"releases/{self.previous_digest}",
                    "rollbackVersion": "0.1.120",
                    "errorCode": "ROLLED_BACK",
                    "health": "LKG_RESTORED",
                    "functionalHealth": "FUNCTIONAL_UNHEALTHY",
                }
            )
        return state

    def provision(self, scenario: str = "commit") -> dict[str, bytes]:
        payloads = self.payloads(scenario)
        self.foundation_backup()
        self.enable(payloads)
        return payloads

    def close(self) -> None:
        for role in BOARD.INPUT_ROLES:
            path = Path(f"/tmp/nuvion-fleet-e2e-{self.run_id}-{role}.json")
            try:
                path.unlink()
            except FileNotFoundError:
                pass


class Iq9075FleetBoardHarnessTest(unittest.TestCase):
    def test_command_runner_drains_but_retains_only_bounded_output(self) -> None:
        script = (
            "import sys; "
            f"sys.stdout.buffer.write(b'x' * {BOARD.MAX_COMMAND_STDOUT_BYTES + 65536}); "
            f"sys.stderr.buffer.write(b'y' * {BOARD.MAX_COMMAND_STDERR_BYTES + 65536})"
        )
        result = BOARD.CommandRunner().run(
            [sys.executable, "-I", "-c", script], timeout=10
        )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "x" * BOARD.MAX_COMMAND_STDOUT_BYTES)
        self.assertEqual(result.stderr, "y" * BOARD.MAX_COMMAND_STDERR_BYTES)

    def test_candidate_execution_proof_binds_systemd_cgroup_and_mount_namespace(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                control_group, _cgroup = seed_candidate_kernel_proof(fixture)
                unit = fixture.harness._candidate_unit(fixture.run_id)
                original_run = fixture.runner.run

                def run_with_properties(argv, *, timeout, input_bytes=None):
                    call = tuple(argv)
                    if call[:2] == ("/usr/bin/systemctl", "show") and not any(
                        item == "--value" for item in call
                    ):
                        names = {
                            item.split("=", 1)[1]
                            for item in call
                            if item.startswith("--property=")
                        }
                        values = {
                            **BOARD.CANDIDATE_SYSTEMD_EXPECTED_PROPERTIES,
                            "ActiveState": "active",
                            "SubState": "running",
                            "ControlGroup": control_group,
                            "MainPID": "9001",
                        }
                        return BOARD.CommandResult(
                            0,
                            "".join(f"{name}={values[name]}\n" for name in sorted(names)),
                            "",
                        )
                    return original_run(
                        argv, timeout=timeout, input_bytes=input_bytes
                    )

                fixture.runner.run = run_with_properties
                proof = fixture.harness._candidate_execution_proof(
                    unit,
                    writable_path=str(fixture.paths.state_root / fixture.run_id),
                    uid_before=candidate_uid_before(),
                )
                self.assertEqual(proof["controlGroup"], control_group)
                self.assertTrue(proof["recursivePopulated"])
                self.assertEqual(
                    set(proof["mountSandbox"]["temporaryFilesystems"]),
                    set(BOARD.CANDIDATE_TMPFS_LIMITS),
                )
                self.assertEqual(
                    set(proof["mountSandbox"]["inaccessiblePaths"]),
                    set(BOARD.CANDIDATE_INACCESSIBLE_PATHS),
                )

                mountinfo = fixture.paths.proc_root / "9001/mountinfo"
                original_mountinfo = mountinfo.read_text(encoding="utf-8")
                oversized = original_mountinfo.replace(
                    "size=268435456,nr_inodes=8192",
                    "size=536870912,nr_inodes=8192",
                    1,
                )
                mountinfo.write_text(oversized, encoding="utf-8")
                with self.assertRaisesRegex(BOARD.HarnessError, "hard limit"):
                    fixture.harness._candidate_execution_proof(
                        unit,
                        writable_path=str(fixture.paths.state_root / fixture.run_id),
                        uid_before=candidate_uid_before(),
                    )
                ancestor_only = "\n".join(
                    line
                    for line in original_mountinfo.splitlines()
                    if line.split()[4] not in BOARD.CANDIDATE_PERSISTENT_PATHS
                ) + "\n"
                mountinfo.write_text(ancestor_only, encoding="utf-8")
                ancestor_proof = fixture.harness._candidate_execution_proof(
                    unit,
                    writable_path=str(fixture.paths.state_root / fixture.run_id),
                    uid_before=candidate_uid_before(),
                )
                self.assertEqual(
                    {
                        item["mountPoint"]
                        for item in ancestor_proof["mountSandbox"][
                            "readOnlyPaths"
                        ].values()
                    },
                    {"/"},
                )
                self.assertTrue(
                    fixture.harness._candidate_security_gates(
                        ancestor_proof,
                        candidate_termination_proof(fixture.run_id),
                        unit=unit,
                        writable_path=str(
                            fixture.paths.state_root / fixture.run_id
                        ),
                    )["persistentStateReadOnly"]
                )
                mountinfo.write_text(
                    ancestor_only
                    + "200 1 0:200 / /etc rw - ext4 /dev/root rw\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(BOARD.HarnessError, "not read-only"):
                    fixture.harness._candidate_execution_proof(
                        unit,
                        writable_path=str(fixture.paths.state_root / fixture.run_id),
                        uid_before=candidate_uid_before(),
                    )
                mountinfo.write_text(ancestor_only, encoding="utf-8")
                (fixture.paths.proc_root / "9001/root/run/user").chmod(0o755)
                with self.assertRaisesRegex(BOARD.HarnessError, "inaccessible"):
                    fixture.harness._candidate_execution_proof(
                        unit,
                        writable_path=str(fixture.paths.state_root / fixture.run_id),
                        uid_before=candidate_uid_before(),
                    )
            finally:
                fixture.close()

    def test_candidate_execution_proof_rejects_property_or_pid_cgroup_drift(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                control_group, _cgroup = seed_candidate_kernel_proof(fixture)
                unit = fixture.harness._candidate_unit(fixture.run_id)
                original_run = fixture.runner.run
                property_drift = True

                def run_with_drift(argv, *, timeout, input_bytes=None):
                    call = tuple(argv)
                    if call[:2] == ("/usr/bin/systemctl", "show") and not any(
                        item == "--value" for item in call
                    ):
                        names = {
                            item.split("=", 1)[1]
                            for item in call
                            if item.startswith("--property=")
                        }
                        values = {
                            **BOARD.CANDIDATE_SYSTEMD_EXPECTED_PROPERTIES,
                            "ActiveState": "active",
                            "SubState": "running",
                            "ControlGroup": control_group,
                            "MainPID": "9001",
                        }
                        if property_drift:
                            values["MemoryMax"] = "max"
                        return BOARD.CommandResult(
                            0,
                            "".join(f"{name}={values[name]}\n" for name in sorted(names)),
                            "",
                        )
                    return original_run(
                        argv, timeout=timeout, input_bytes=input_bytes
                    )

                fixture.runner.run = run_with_drift
                with self.assertRaisesRegex(BOARD.HarnessError, "runtime proof"):
                    fixture.harness._candidate_execution_proof(
                        unit,
                        writable_path=str(fixture.paths.state_root / fixture.run_id),
                        uid_before=candidate_uid_before(),
                    )
                property_drift = False
                (fixture.paths.proc_root / "9001/cgroup").write_text(
                    "0::/user.slice/user-1000.slice\n", encoding="ascii"
                )
                with self.assertRaisesRegex(BOARD.HarnessError, "escaped"):
                    fixture.harness._candidate_execution_proof(
                        unit,
                        writable_path=str(fixture.paths.state_root / fixture.run_id),
                        uid_before=candidate_uid_before(),
                    )
            finally:
                fixture.close()

    def test_candidate_uid_isolation_rejects_host_process_and_scan_race(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                control_group, _cgroup = seed_candidate_kernel_proof(fixture)
                proof = fixture.harness._candidate_uid_isolation_proof(
                    expected_control_group=control_group,
                    require_process=True,
                    timeout=1,
                )
                self.assertEqual(proof["pids"], [9001])

                outsider = fixture.paths.proc_root / "9002"
                outsider.mkdir()
                (outsider / "status").write_text(
                    "Name:\tpython3\nUid:\t4242\t4242\t4242\t4242\n",
                    encoding="ascii",
                )
                (outsider / "cgroup").write_text(
                    "0::/user.slice/user-4242.slice/session.scope\n",
                    encoding="ascii",
                )
                with self.assertRaisesRegex(BOARD.HarnessError, "escaped"):
                    fixture.harness._candidate_uid_isolation_proof(
                        expected_control_group=control_group,
                        require_process=True,
                        timeout=1,
                    )
                with self.assertRaisesRegex(BOARD.HarnessError, "pre-existing"):
                    fixture.harness._candidate_uid_isolation_proof(
                        expected_control_group=None,
                        require_process=False,
                        timeout=1,
                    )

                alternating = [
                    {9001: control_group},
                    {9001: control_group, 9003: control_group},
                ]
                calls = 0

                def racing_snapshot():
                    nonlocal calls
                    value = alternating[calls % 2]
                    calls += 1
                    return value

                fixture.harness._nuvion_process_cgroups = racing_snapshot  # type: ignore[method-assign]
                with self.assertRaisesRegex(BOARD.HarnessError, "did not stabilize"):
                    fixture.harness._candidate_uid_isolation_proof(
                        expected_control_group=control_group,
                        require_process=True,
                        timeout=0.2,
                    )
            finally:
                fixture.close()

    def test_default_os_release_uses_canonical_file_with_ubuntu_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                self.assertEqual(
                    fixture.paths.os_release,
                    fixture.root / "usr/lib/os-release",
                )
                self.assertTrue((fixture.root / "etc/os-release").is_symlink())
                self.assertTrue(fixture.harness.preflight(fixture.run_id)["verified"])
            finally:
                fixture.close()

    def test_fixed_destinations_and_strict_public_key_schemas(self) -> None:
        self.assertNotIn("--keyring-path", BOARD.build_parser().format_help())
        self.assertEqual(
            BOARD.FIXED_DESTINATIONS,
            {
                "agentCommand": "/etc/nuv-agent/fleet-command-keyring.json",
                "updaterCommand": "/etc/nuvion-updater/command-keyring.json",
                "release": "/etc/nuvion-updater/release-keyring.json",
                "health": "/etc/nuvion-updater/health-attestation-keyring.json",
                "binding": "/etc/nuvion-updater/device-binding.json",
            },
        )
        command = json.loads(HarnessFixture._keyring("command"))
        command["purpose"] = "unexpected"
        with self.assertRaisesRegex(BOARD.HarnessError, "fields do not match"):
            BOARD.validate_ed25519_keyring(json.dumps(command).encode(), role="command")
        command = json.loads(HarnessFixture._keyring("command"))
        command["keys"]["command-dev"] = base64.b64encode(b"short").decode()
        with self.assertRaisesRegex(BOARD.HarnessError, "canonical 32-byte"):
            BOARD.validate_ed25519_keyring(json.dumps(command).encode(), role="command")
        with self.assertRaisesRegex(BOARD.HarnessError, "strict UTF-8 JSON"):
            BOARD.strict_json(b'{"value":NaN}', label="review repro")

    def test_anti_replay_snapshot_reads_semantic_sqlite_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                fixture.paths.updater_state_db.unlink()
                connection = sqlite3.connect(fixture.paths.updater_state_db)
                try:
                    connection.executescript(
                        """
                        PRAGMA user_version = 4;
                        CREATE TABLE updater_command (
                            command_id TEXT PRIMARY KEY,
                            sequence INTEGER NOT NULL UNIQUE,
                            phase TEXT NOT NULL,
                            bom_digest TEXT NOT NULL,
                            release_sequence INTEGER,
                            health_deadline TEXT
                        );
                        CREATE TABLE updater_meta (
                            meta_key TEXT PRIMARY KEY,
                            meta_value TEXT NOT NULL
                        );
                        CREATE TABLE updater_transition (
                            id INTEGER PRIMARY KEY,
                            command_id TEXT NOT NULL,
                            to_phase TEXT NOT NULL
                        );
                        CREATE TABLE updater_commit_gate (
                            command_id TEXT PRIMARY KEY,
                            gate_id TEXT NOT NULL
                        );
                        """
                    )
                    connection.execute(
                        "INSERT INTO updater_command VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            fixture.command_id,
                            2,
                            "ROLLED_BACK",
                            f"sha256:{fixture.current_digest}",
                            2,
                            None,
                        ),
                    )
                    connection.executemany(
                        "INSERT INTO updater_meta VALUES (?, ?)",
                        (
                            ("currentReleaseSequence", "1"),
                            (
                                "currentBomDigest",
                                f"sha256:{fixture.previous_digest}",
                            ),
                        ),
                    )
                    connection.commit()
                finally:
                    connection.close()
                fixture.paths.updater_state_db.chmod(0o600)

                before = fixture.harness._anti_replay_snapshot()
                self.assertEqual(before["schemaVersion"], 4)
                self.assertEqual(before["maximumCommandSequence"], 2)
                self.assertRegex(before["semanticSha256"], r"^[0-9a-f]{64}$")
                self.assertEqual(before["currentReleaseSequence"], "1")
                self.assertEqual(
                    before["currentBomDigest"],
                    f"sha256:{fixture.previous_digest}",
                )
                self.assertEqual(before["latest"]["commandId"], fixture.command_id)
                self.assertIsNone(before["latest"]["healthDeadline"])

                payloads = fixture.provision("oak-fault-rollback")
                fixture.activate_candidate()
                fixture.runner.updater["update"] = fixture.update_state(
                    "FUNCTIONAL_HEALTHY"
                )
                fixture.harness.arm_oak_fault(fixture.run_id)
                current = fixture.root / "opt/nuv-agent/current"
                previous = fixture.root / "opt/nuv-agent/previous"
                current.unlink()
                current.symlink_to(f"releases/{fixture.previous_digest}")
                previous.unlink()
                previous.symlink_to(f"releases/{fixture.current_digest}")
                fixture.runner.run(
                    ["/usr/bin/systemctl", "restart", "nuv-agent.service"],
                    timeout=30,
                )
                fixture.runner.updater["update"] = fixture.update_state(
                    "ROLLED_BACK"
                )
                fleet_evidence = fixture.harness.evidence(fixture.run_id)
                self.assertEqual(fleet_evidence["schemaVersion"], 2)
                self.assertEqual(fleet_evidence["antiReplay"], before)
                HOST.validate_final_evidence(
                    fleet_evidence, json.loads(payloads["manifest"])
                )

                connection = sqlite3.connect(fixture.paths.updater_state_db)
                try:
                    connection.execute(
                        "UPDATE updater_meta SET meta_value = ? "
                        "WHERE meta_key = 'currentReleaseSequence'",
                        ("2",),
                    )
                    connection.commit()
                finally:
                    connection.close()
                after = fixture.harness._anti_replay_snapshot()
                self.assertNotEqual(after, before)
                self.assertEqual(after["currentReleaseSequence"], "2")
            finally:
                fixture.close()

    def test_release_keyring_accepts_policy_spki_and_rejects_private_or_wrong_algorithm(
        self,
    ) -> None:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ec, ed25519

        policy_keyring = (
            ROOT
            / "packaging/release/trusted-release-keyrings/iq9075-dev.json"
        ).read_bytes()
        validated = BOARD.validate_ed25519_keyring(
            policy_keyring,
            role="release",
        )
        self.assertEqual(validated["trustDomain"], "iq9075-dev")

        private_der = ed25519.Ed25519PrivateKey.generate().private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        wrong_algorithm = ec.generate_private_key(ec.SECP256R1()).public_key().public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

        def release_keyring(material: bytes) -> bytes:
            return json.dumps(
                {
                    "schemaVersion": 1,
                    "trustDomain": "iq9075-dev",
                    "keys": {
                        "release-dev": base64.b64encode(material).decode("ascii")
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()

        for material in (private_der, wrong_algorithm):
            with self.subTest(material_length=len(material)):
                with self.assertRaisesRegex(
                    BOARD.HarnessError,
                    "raw 32-byte or canonical DER SPKI Ed25519",
                ):
                    BOARD.validate_ed25519_keyring(
                        release_keyring(material),
                        role="release",
                    )

        with self.assertRaisesRegex(BOARD.HarnessError, "canonical 32-byte"):
            BOARD.validate_ed25519_keyring(
                release_keyring(
                    ed25519.Ed25519PrivateKey.generate()
                    .public_key()
                    .public_bytes(
                        encoding=serialization.Encoding.DER,
                        format=serialization.PublicFormat.SubjectPublicKeyInfo,
                    )
                ),
                role="command",
            )

    def test_foundation_does_not_require_unprovisioned_updater(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                result = fixture.harness.preflight(fixture.run_id)
                self.assertTrue(result["verified"])
                self.assertFalse(
                    any(call[0] == "/usr/sbin/runuser" for call in fixture.runner.calls)
                )
            finally:
                fixture.close()

    def test_oak_identity_resolves_unique_usb1_hub_downstream_topology(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                oak = fixture.harness.verify_oak()

                self.assertEqual(oak["port"], "2-1.1")
                self.assertEqual(oak["vendorId"], "03e7")
                self.assertEqual(oak["productId"], "f63b")
                self.assertNotEqual(oak["port"], BOARD.USB_ROOT_HUB)
            finally:
                fixture.close()

    def test_oak_sysfs_virtual_size_is_bounded_by_read_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            attributes = {
                fixture.paths.usb_devices / "2-1.1" / name
                for name in ("idVendor", "idProduct", "speed", "serial")
            }
            original_lstat = Path.lstat
            original_fstat = os.fstat
            identities = {
                (metadata.st_dev, metadata.st_ino)
                for metadata in (original_lstat(path) for path in attributes)
            }

            def virtual_size(metadata: os.stat_result) -> os.stat_result:
                fields = list(metadata)
                fields[6] = 4096
                return os.stat_result(fields)

            def fake_lstat(path: Path) -> os.stat_result:
                metadata = original_lstat(path)
                return virtual_size(metadata) if path in attributes else metadata

            def fake_fstat(descriptor: int) -> os.stat_result:
                metadata = original_fstat(descriptor)
                if (metadata.st_dev, metadata.st_ino) in identities:
                    return virtual_size(metadata)
                return metadata

            try:
                with (
                    mock.patch.object(Path, "lstat", new=fake_lstat),
                    mock.patch.object(BOARD.os, "fstat", new=fake_fstat),
                ):
                    with self.assertRaisesRegex(
                        BOARD.HarnessError,
                        "file exceeds size limit: idVendor",
                    ):
                        BOARD.read_regular(
                            fixture.paths.usb_devices / "2-1.1/idVendor",
                            maximum=128,
                        )
                    self.assertEqual(fixture.harness.verify_oak()["vendorId"], "03e7")
                    (fixture.paths.usb_devices / "2-1.1/serial").chmod(0o600)
                    fixture._write(
                        "/sys/bus/usb/devices/2-1.1/serial",
                        "x" * 129,
                        0o444,
                    )
                    with self.assertRaisesRegex(
                        BOARD.HarnessError,
                        "file exceeds size limit: serial",
                    ):
                        fixture.harness.verify_oak()
            finally:
                fixture.close()

    def test_oak_identity_rejects_ambiguous_usb1_downstream_devices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                fixture._write(
                    "/sys/bus/usb/devices/2-1.2/idVendor",
                    "03e7\n",
                    0o444,
                )
                fixture._write(
                    "/sys/bus/usb/devices/2-1.2/idProduct",
                    "f63b\n",
                    0o444,
                )
                fixture._write(
                    "/sys/bus/usb/devices/2-1.2/speed",
                    "5000\n",
                    0o444,
                )
                (fixture.paths.usb_devices / "2-1.2/driver").symlink_to(
                    "../../drivers/usb"
                )

                with self.assertRaisesRegex(
                    BOARD.HarnessError,
                    "exactly one OAK-D Lite",
                ):
                    fixture.harness.verify_oak()
            finally:
                fixture.close()

    def test_persistent_board_lease_blocks_concurrent_run_until_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            second_run = str(uuid.uuid4())
            try:
                fixture.harness.preflight(fixture.run_id)
                with self.assertRaisesRegex(
                    BOARD.HarnessError, "another Fleet E2E run"
                ):
                    fixture.harness.preflight(second_run)
                self.assertTrue(fixture.harness.cleanup(fixture.run_id)["complete"])
                self.assertTrue(fixture.harness.preflight(second_run)["verified"])
                self.assertTrue(fixture.harness.cleanup(second_run)["complete"])
            finally:
                fixture.close()

    def test_consistent_backup_stops_writers_verifies_archive_and_restores_units(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                before = json.loads(json.dumps(fixture.runner.units))
                fixture.harness.preflight(fixture.run_id)
                result = fixture.harness.backup(fixture.run_id)
                self.assertTrue(result["emergencyArchiveVerified"])
                self.assertNotIn("recoveryComplete", result)
                self.assertEqual(
                    {
                        unit: {
                            "active": value["active"],
                            "enabled": value["enabled"],
                        }
                        for unit, value in fixture.runner.units.items()
                    },
                    {
                        unit: {
                            "active": value["active"],
                            "enabled": value["enabled"],
                        }
                        for unit, value in before.items()
                    },
                )
                calls = fixture.runner.calls
                first_stop = next(
                    index
                    for index, call in enumerate(calls)
                    if call[:2] == ("/usr/bin/systemctl", "stop")
                )
                self.assertTrue(first_stop >= 0)
                state = fixture.harness._load_state(fixture.run_id)
                archive = Path(state["backup"]["archivePath"])
                with tarfile.open(archive, "r") as opened:
                    self.assertIn("recovery-integrity.json", opened.getnames())
            finally:
                fixture.close()

    def test_backup_resume_keeps_original_unit_snapshot_after_applied_save_crash(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                fixture.harness.preflight(fixture.run_id)
                before = json.loads(json.dumps(fixture.runner.units))
                original_save = fixture.harness._save_state
                failed = False

                def crash_before_applied_save(
                    run_id: str, state: Mapping[str, Any]
                ) -> None:
                    nonlocal failed
                    backup = state.get("backup")
                    if (
                        not failed
                        and isinstance(backup, Mapping)
                        and backup.get("phase") == "APPLIED"
                    ):
                        failed = True
                        raise OSError(
                            "simulated power loss before APPLIED backup fsync"
                        )
                    original_save(run_id, state)

                fixture.harness._save_state = crash_before_applied_save
                with self.assertRaisesRegex(OSError, "simulated power loss"):
                    fixture.harness.backup(fixture.run_id)
                fixture.harness._save_state = original_save
                persisted = fixture.harness._load_state(fixture.run_id)
                self.assertEqual(persisted["backup"]["phase"], "APPLYING")
                before_state = {
                    unit: (value["active"], value["enabled"])
                    for unit, value in before.items()
                }
                self.assertEqual(
                    {
                        unit: (value["active"], value["enabled"])
                        for unit, value in fixture.runner.units.items()
                    },
                    before_state,
                )
                resumed = fixture.harness.backup(fixture.run_id)
                self.assertTrue(resumed["idempotent"])
                self.assertEqual(
                    {
                        unit: (value["active"], value["enabled"])
                        for unit, value in fixture.runner.units.items()
                    },
                    before_state,
                )
                self.assertEqual(
                    fixture.harness._load_state(fixture.run_id)["backup"]["phase"],
                    "RESTORED",
                )
            finally:
                fixture.close()

    def test_transaction_installs_exact_files_modes_and_same_command_digest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                payloads = fixture.provision()
                state = fixture.harness._load_state(fixture.run_id)
                self.assertEqual(state["trustTransaction"]["phase"], "APPLIED")
                agent_command = fixture.paths.rooted(BOARD.AGENT_COMMAND_KEYRING)
                updater_command = fixture.paths.rooted(BOARD.UPDATER_COMMAND_KEYRING)
                self.assertEqual(agent_command.read_bytes(), payloads["command"])
                self.assertEqual(updater_command.read_bytes(), payloads["command"])
                self.assertEqual(agent_command.stat().st_mode & 0o777, 0o640)
                for absolute in (
                    BOARD.UPDATER_COMMAND_KEYRING,
                    BOARD.RELEASE_KEYRING,
                    BOARD.HEALTH_KEYRING,
                    BOARD.DEVICE_BINDING,
                ):
                    self.assertEqual(
                        fixture.paths.rooted(absolute).stat().st_mode & 0o777, 0o600
                    )
                config = fixture.paths.config.read_text(encoding="utf-8")
                self.assertEqual(fixture.paths.config.stat().st_mode & 0o777, 0o640)
                self.assertIn(
                    f"NUVION_FLEET_COMMAND_KEYRING_PATH={BOARD.AGENT_COMMAND_KEYRING}",
                    config,
                )
                self.assertIn("NUVION_FLEET_COMMAND_ENABLED=true", config)
                self.assertEqual(fixture.runner.updater["updaterVersion"], "0.2.0")
            finally:
                fixture.close()

    def test_manifest_must_match_live_pretrust_baseline_slot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                payloads = fixture.payloads()
                fixture.foundation_backup()
                current = fixture.root / "opt/nuv-agent/current"
                current.unlink()
                current.symlink_to("bootstrap/0.1.119")
                with self.assertRaisesRegex(BOARD.HarnessError, "live baseline slot"):
                    fixture.enable(payloads)
                self.assertFalse(
                    fixture.paths.rooted(BOARD.AGENT_COMMAND_KEYRING).exists()
                )
            finally:
                fixture.close()

    def test_binding_must_match_provisioned_agent_device_and_space(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                payloads = fixture.payloads()
                changed = fixture.paths.config.read_text(encoding="utf-8")
                changed = changed.replace("sp-3-nuvion", "sp-4-nuvion").replace(
                    "NUVION_SPACE_ID=3", "NUVION_SPACE_ID=4"
                )
                fixture.paths.config.write_text(changed, encoding="utf-8")
                fixture.foundation_backup()
                with self.assertRaisesRegex(
                    BOARD.HarnessError, "provisioned Agent identity"
                ):
                    fixture.enable(payloads)
                self.assertFalse(fixture.paths.rooted(BOARD.DEVICE_BINDING).exists())
            finally:
                fixture.close()

    def test_applied_state_reconciles_agent_env_overwrite_and_missing_file(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                payloads = fixture.provision()
                fixture.paths.config.write_text(
                    "NUVION_FLEET_COMMAND_ENABLED=false\n", encoding="utf-8"
                )
                fixture.paths.rooted(BOARD.RELEASE_KEYRING).unlink()
                result = fixture.enable(payloads)
                self.assertEqual(result["phase"], "APPLIED")
                self.assertIn(
                    "NUVION_FLEET_COMMAND_ENABLED=true",
                    fixture.paths.config.read_text(encoding="utf-8"),
                )
                self.assertEqual(
                    fixture.paths.rooted(BOARD.RELEASE_KEYRING).read_bytes(),
                    payloads["release"],
                )
            finally:
                fixture.close()

    def test_restart_then_state_save_crash_resumes_from_applying(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                payloads = fixture.payloads()
                fixture.foundation_backup()
                original_save = fixture.harness._save_state
                failed = False

                def crash_on_applied(run_id: str, state: Mapping[str, Any]) -> None:
                    nonlocal failed
                    transaction = state.get("trustTransaction")
                    if (
                        not failed
                        and isinstance(transaction, Mapping)
                        and transaction.get("phase") == "APPLIED"
                    ):
                        failed = True
                        raise OSError("simulated power loss before APPLIED fsync")
                    original_save(run_id, state)

                fixture.harness._save_state = crash_on_applied
                with self.assertRaisesRegex(OSError, "simulated power loss"):
                    fixture.enable(payloads)
                fixture.harness._save_state = original_save
                persisted = fixture.harness._load_state(fixture.run_id)
                self.assertEqual(persisted["trustTransaction"]["phase"], "APPLYING")
                self.assertFalse(fixture.runner.units["nuv-agent.service"]["active"])
                fixture.enable(payloads)
                resumed = fixture.harness._load_state(fixture.run_id)
                self.assertEqual(resumed["trustTransaction"]["phase"], "APPLIED")
                self.assertTrue(fixture.runner.units["nuv-agent.service"]["active"])
            finally:
                fixture.close()

    def test_applied_bytes_are_durable_before_runtime_restart_and_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                payloads = fixture.payloads()
                fixture.foundation_backup()
                original_restart = fixture.harness._restart_runtime

                def crash_before_runtime() -> dict[str, int]:
                    persisted = fixture.harness._read_existing_run_state(
                        fixture.run_id
                    )["trustTransaction"]
                    self.assertEqual(persisted["phase"], "APPLIED")
                    self.assertFalse(persisted["liveVerified"])
                    self.assertNotIn("appliedPids", persisted)
                    self.assertTrue(
                        all(
                            unit["active"] is False
                            for unit in fixture.runner.units.values()
                        )
                    )
                    raise OSError("simulated crash before runtime restart")

                fixture.harness._restart_runtime = crash_before_runtime  # type: ignore[method-assign]
                with self.assertRaisesRegex(OSError, "before runtime restart"):
                    fixture.enable(payloads)
                fixture.harness._restart_runtime = original_restart  # type: ignore[method-assign]
                resumed = fixture.enable(payloads)
                self.assertEqual(resumed["phase"], "APPLIED")
                final = fixture.harness._read_existing_run_state(fixture.run_id)[
                    "trustTransaction"
                ]
                self.assertTrue(final["liveVerified"])
                self.assertIn("appliedPids", final)
            finally:
                fixture.close()

    def test_failed_rebind_stays_armed_until_deadman_cleanup_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                fixture.provision("oak-fault-rollback")
                fixture.activate_candidate()
                fixture.runner.updater["update"] = fixture.update_state(
                    "FUNCTIONAL_HEALTHY"
                )

                def never_rebind(action: str, port: str) -> None:
                    driver = fixture.paths.usb_devices / port / "driver"
                    if action == "unbind" and driver.exists():
                        driver.unlink()

                fixture.harness.usb_write_hook = never_rebind
                with self.assertRaisesRegex(BOARD.HarnessError, "did not recover"):
                    fixture.harness.arm_oak_fault(fixture.run_id)
                state = fixture.harness._load_state(fixture.run_id)
                self.assertTrue(state["oakFault"]["armed"])
                self.assertEqual(state["oakFault"]["port"], "2-1.1")
                unit = fixture.harness._deadman_unit(fixture.run_id)
                self.assertTrue(fixture.runner.deadmen[unit])
                deadman_call = next(
                    call
                    for call in fixture.runner.calls
                    if call
                    and call[0] == "/usr/bin/systemd-run"
                    and any(
                        item.startswith("--unit=nuvion-oak-deadman-")
                        for item in call
                    )
                )
                self.assertIn("--property=RuntimeMaxSec=180", deadman_call)
                self.assertIn("--property=TimeoutStopSec=45", deadman_call)
                self.assertIn("--property=LimitCORE=0", deadman_call)
                fixture.harness.usb_write_hook = fixture._usb_hook
                recovered = fixture.harness.cleanup(fixture.run_id, deadman_only=True)
                self.assertTrue(recovered["complete"])
                self.assertTrue(recovered["recovered"])
            finally:
                fixture.close()

    def test_false_evidence_never_completes_then_commit_proof_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                payloads = fixture.provision("commit")
                manifest = json.loads(payloads["manifest"])
                fixture.runner.updater["update"] = fixture.update_state(
                    "FUNCTIONAL_HEALTHY"
                )
                incomplete = fixture.harness.evidence(fixture.run_id)
                self.assertFalse(incomplete["complete"])
                with self.assertRaisesRegex(HOST.RunnerError, "not complete"):
                    HOST.validate_final_evidence(incomplete, manifest)
                fixture.runner.updater["update"] = fixture.update_state("COMMITTED")
                fixture.activate_candidate()
                complete = fixture.harness.evidence(fixture.run_id)
                self.assertTrue(complete["complete"])
                HOST.validate_final_evidence(complete, manifest)
                serialized = json.dumps(complete)
                self.assertNotIn("must-stay-private", serialized)
                self.assertNotIn("archivePath", serialized)
            finally:
                fixture.close()

    def test_rollback_evidence_requires_exact_slot_version_and_error_tuple(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                fixture.provision("oak-fault-rollback")
                fixture.activate_candidate()
                fixture.runner.updater["update"] = fixture.update_state(
                    "FUNCTIONAL_HEALTHY"
                )
                fixture.harness.arm_oak_fault(fixture.run_id)
                current = fixture.root / "opt/nuv-agent/current"
                previous = fixture.root / "opt/nuv-agent/previous"
                current.unlink()
                current.symlink_to(f"releases/{fixture.previous_digest}")
                previous.unlink()
                previous.symlink_to(f"releases/{fixture.current_digest}")
                fixture.runner.run(
                    ["/usr/bin/systemctl", "restart", "nuv-agent.service"],
                    timeout=30,
                )
                fixture.runner.updater["update"] = fixture.update_state("ROLLED_BACK")
                exact = fixture.harness.evidence(fixture.run_id)
                self.assertTrue(exact["complete"])
                retried = fixture.harness.evidence(fixture.run_id)
                self.assertEqual(retried, exact)
                persisted = (
                    fixture.paths.state_root / fixture.run_id / "evidence.json"
                ).read_bytes()
                self.assertEqual(
                    fixture.harness._load_state(fixture.run_id)["fleetEvidence"][
                        "sha256"
                    ],
                    hashlib.sha256(persisted).hexdigest(),
                )
                fixture.runner.updater["update"]["rollbackVersion"] = "0.1.119"
                wrong = fixture.harness.evidence(fixture.run_id)
                self.assertFalse(wrong["complete"])
            finally:
                fixture.close()

    def test_cleanup_restores_exact_before_digests_and_is_repeatable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                config_before = fixture.paths.config.read_bytes()
                fixture.provision()
                first = fixture.harness.cleanup(fixture.run_id)
                second = fixture.harness.cleanup(fixture.run_id)
                self.assertTrue(first["complete"])
                self.assertTrue(second["complete"])
                for result in (first, second):
                    self.assertEqual(
                        result["kind"], "nuvion-iq9075-cleanup-evidence"
                    )
                    self.assertEqual(result["proof"]["transactionPhase"], "RESTORED")
                    self.assertTrue(
                        all(
                            value is True
                            for key, value in result["proof"].items()
                            if key not in {"schemaVersion", "transactionPhase"}
                        )
                    )
                self.assertEqual(fixture.paths.config.read_bytes(), config_before)
                for absolute in BOARD.FIXED_DESTINATIONS.values():
                    self.assertFalse(fixture.paths.rooted(absolute).exists())
                state = fixture.harness._load_state(fixture.run_id)
                self.assertEqual(state["trustTransaction"]["phase"], "RESTORED")
                self.assertFalse(
                    fixture.harness._transaction_dir(fixture.run_id).exists()
                )
                self.assertFalse(
                    (
                        fixture.paths.recovery_root / f"iq9075-{fixture.run_id}.tar"
                    ).exists()
                )
            finally:
                fixture.close()

    def test_completed_cleanup_releases_lease_left_by_final_crash_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                fixture.provision()
                self.assertTrue(fixture.harness.cleanup(fixture.run_id)["complete"])
                self.assertFalse(fixture.paths.active_run.exists())

                # Recreate the only externally visible state of a crash between
                # saving cleanup.complete and releasing the persistent lease.
                fixture.harness._claim_active_run(fixture.run_id)
                self.assertTrue(fixture.paths.active_run.exists())

                resumed = fixture.harness.cleanup(fixture.run_id)

                self.assertTrue(resumed["complete"])
                self.assertTrue(resumed["idempotent"])
                self.assertFalse(fixture.paths.active_run.exists())
            finally:
                fixture.close()

    def test_invalid_private_like_staging_is_removed_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                payloads = fixture.payloads()
                payloads["command"] = b'{"privateKey":"forbidden"}\n'
                staged = fixture.stage(payloads)
                fixture.foundation_backup()
                with self.assertRaises(BOARD.HarnessError):
                    fixture.harness.enable_fleet(
                        fixture.run_id,
                        command_keyring=staged["command"][0],
                        command_sha256=staged["command"][1],
                        release_keyring=staged["release"][0],
                        release_sha256=staged["release"][1],
                        health_keyring=staged["health"][0],
                        health_sha256=staged["health"][1],
                        device_binding=staged["binding"][0],
                        binding_sha256=staged["binding"][1],
                        manifest_path=staged["manifest"][0],
                        manifest_sha256=staged["manifest"][1],
                    )
                self.assertTrue(all(not path.exists() for path, _ in staged.values()))
            finally:
                fixture.close()

    def test_candidate_bundle_stage_is_fixed_idempotent_and_slot_neutral(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            bundle_path = Path(directory) / "candidate.tar.gz"
            try:
                with tarfile.open(bundle_path, "w:gz") as archive:
                    for name in (
                        "bin",
                        "venv",
                        "venv/bin",
                        "venv/lib",
                        "venv/lib/python3.12",
                        "venv/lib/python3.12/site-packages",
                    ):
                        member = tarfile.TarInfo(name)
                        member.type = tarfile.DIRTYPE
                        member.mode = 0o755
                        member.uid = member.gid = 0
                        archive.addfile(member)
                    for name, payload in (
                        ("bin/nuv-agent", b"#!/bin/sh\nexit 0\n"),
                        ("venv/bin/python", BOARD.CANDIDATE_PYTHON_WRAPPER),
                    ):
                        member = tarfile.TarInfo(name)
                        member.mode = 0o755
                        member.uid = member.gid = 0
                        member.size = len(payload)
                        archive.addfile(member, io.BytesIO(payload))
                bundle_sha = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
                bom: dict[str, object] = {
                    "schemaVersion": 2,
                    "bomId": "nuv-agent-0.1.121-iq9075",
                    "bomDigest": "",
                    "releaseSequence": 2,
                    "agentVersion": "0.1.121",
                    "componentSha": "c" * 40,
                    "configSchema": "12",
                    "minUpdaterVersion": "0.2.0",
                    "targets": [
                        {
                            "productModel": "IQ9075_DEV",
                            "platformProfile": "iq9075_dev",
                            "hardwareRevision": "QCS9075-EVK",
                            "architecture": "aarch64",
                        }
                    ],
                    "artifact": {
                        "name": "nuv-agent_0.1.121_iq9075.agent-bundle.tar.gz",
                        "kind": "agent-bundle",
                        "sha256": bundle_sha,
                        "sizeBytes": bundle_path.stat().st_size,
                    },
                    "builtAt": "2026-09-03T00:00:00Z",
                }
                unsigned = dict(bom)
                unsigned.pop("bomDigest")
                bom["bomDigest"] = "sha256:" + hashlib.sha256(
                    json.dumps(
                        unsigned,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                release = {
                    "agentVersion": "0.1.121",
                    "releaseSequence": 2,
                    "artifactDigest": f"sha256:{bundle_sha}",
                    "componentSha": "c" * 40,
                    "configSchema": "12",
                    "publisherKeyId": "release-iq9075-dev",
                }
                manifest = {
                    "scenario": {
                        "type": "oak-fault-rollback",
                        "expectedBomDigest": bom["bomDigest"],
                        "release": release,
                    }
                }
                bom_payload = json.dumps(bom, separators=(",", ":")).encode()
                BOARD.validate_candidate_bom(
                    bom_payload,
                    manifest=manifest,
                    bundle_sha256=bundle_sha,
                    bundle_size=bundle_path.stat().st_size,
                )
                metadata = bundle_path.stat()
                candidate_input = BOARD.CandidateInput(
                    role="candidate-bundle",
                    path=bundle_path,
                    sha256=bundle_sha,
                    size=metadata.st_size,
                    device=metadata.st_dev,
                    inode=metadata.st_ino,
                )
                release_marker = {
                    "schemaVersion": 2,
                    "bomDigest": bom["bomDigest"],
                    **release,
                }
                original_bundle = bundle_path.read_bytes()
                tampered_bundle = bytearray(original_bundle)
                tampered_bundle[-1] ^= 0x01
                bundle_path.write_bytes(tampered_bundle)
                with self.assertRaisesRegex(
                    BOARD.HarnessError, "changed before extraction"
                ):
                    fixture.harness._stage_candidate_bundle(
                        run_id=fixture.run_id,
                        bundle=candidate_input,
                        bom=bom,
                        harness_sha256="f" * 64,
                        release_marker=release_marker,
                    )
                bundle_path.write_bytes(original_bundle)
                expected_slot = fixture.harness._candidate_slot_path(
                    fixture.run_id, str(bom["bomDigest"])
                )
                with (
                    mock.patch.object(BOARD, "MAX_CANDIDATE_ENTRIES", 2),
                    mock.patch.object(
                        BOARD.tarfile.TarFile,
                        "getmembers",
                        side_effect=AssertionError("unbounded metadata API used"),
                    ),
                    self.assertRaisesRegex(
                        BOARD.HarnessError, "entry count is invalid"
                    ),
                ):
                    fixture.harness._stage_candidate_bundle(
                        run_id=fixture.run_id,
                        bundle=candidate_input,
                        bom=bom,
                        harness_sha256="f" * 64,
                        release_marker=release_marker,
                    )
                self.assertFalse(
                    fixture.harness._candidate_incoming_path(expected_slot).exists()
                )
                slots_before = fixture.harness._slot_snapshot()
                with mock.patch.object(
                    BOARD.tarfile.TarFile,
                    "getmembers",
                    side_effect=AssertionError("unbounded metadata API used"),
                ):
                    first, marker_sha = fixture.harness._stage_candidate_bundle(
                        run_id=fixture.run_id,
                        bundle=candidate_input,
                        bom=bom,
                        harness_sha256="f" * 64,
                        release_marker=release_marker,
                    )
                second, second_sha = fixture.harness._stage_candidate_bundle(
                    run_id=fixture.run_id,
                    bundle=candidate_input,
                    bom=bom,
                    harness_sha256="f" * 64,
                    release_marker=release_marker,
                )
                self.assertEqual(first, second)
                self.assertEqual(marker_sha, second_sha)
                self.assertEqual(first.parent, fixture.paths.candidate_root)
                self.assertEqual(fixture.harness._slot_snapshot(), slots_before)
                self.assertTrue((first / "bin/nuv-agent").is_file())
                self.assertTrue((first / "venv/bin/python").is_file())
                candidate_wrapper = first / "venv/bin/python"
                candidate_wrapper.write_bytes(
                    b"#!/bin/sh\nprintf '%s\\n' '{\"outcome\":{\"status\":\"passed\"}}'\n"
                )
                candidate_wrapper.chmod(0o755)
                with self.assertRaisesRegex(
                    BOARD.HarnessError, "differs from canonical bytes"
                ):
                    fixture.harness._verify_candidate_slot(
                        first,
                        control=fixture.harness._candidate_control_marker(
                            run_id=fixture.run_id,
                            slot=first,
                            bom=bom,
                            bundle_sha256=bundle_sha,
                            harness_sha256="f" * 64,
                        ),
                        release_marker=release_marker,
                        bom=bom,
                    )
            finally:
                fixture.close()

    def test_candidate_bom_rejects_bundle_digest_or_target_drift(self) -> None:
        release = {
            "agentVersion": "0.1.121",
            "releaseSequence": 2,
            "artifactDigest": "sha256:" + "a" * 64,
            "componentSha": "b" * 40,
            "configSchema": "12",
            "publisherKeyId": "release-test",
        }
        bom: dict[str, object] = {
            "schemaVersion": 2,
            "bomId": "candidate",
            "bomDigest": "",
            "releaseSequence": 2,
            "agentVersion": "0.1.121",
            "componentSha": "b" * 40,
            "configSchema": "12",
            "minUpdaterVersion": "0.2.0",
            "targets": [
                {
                    "productModel": "IQ9075_DEV",
                    "platformProfile": "iq9075_dev",
                    "hardwareRevision": "QCS9075-EVK",
                    "architecture": "aarch64",
                }
            ],
            "artifact": {
                "name": "candidate.tar.gz",
                "kind": "agent-bundle",
                "sha256": "a" * 64,
                "sizeBytes": 100,
            },
            "builtAt": "2026-09-03T00:00:00Z",
        }
        unsigned = dict(bom)
        unsigned.pop("bomDigest")
        bom["bomDigest"] = "sha256:" + hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        manifest = {
            "scenario": {
                "type": "oak-fault-rollback",
                "expectedBomDigest": bom["bomDigest"],
                "release": release,
            }
        }
        with self.assertRaisesRegex(BOARD.HarnessError, "bundle identity"):
            BOARD.validate_candidate_bom(
                json.dumps(bom).encode(),
                manifest=manifest,
                bundle_sha256="c" * 64,
                bundle_size=100,
            )

    def test_candidate_soak_restores_fresh_baseline_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            bundle_path = fixture.harness._candidate_staging_path(
                fixture.run_id, "candidate-bundle"
            )
            bom_path = fixture.harness._candidate_staging_path(
                fixture.run_id, "candidate-bom"
            )
            harness_path = fixture.harness._candidate_staging_path(
                fixture.run_id, "oak-harness"
            )
            try:
                with tarfile.open(bundle_path, "w:gz") as archive:
                    for name in (
                        "bin",
                        "venv",
                        "venv/bin",
                        "venv/lib",
                        "venv/lib/python3.12",
                        "venv/lib/python3.12/site-packages",
                    ):
                        member = tarfile.TarInfo(name)
                        member.type = tarfile.DIRTYPE
                        member.mode = 0o755
                        member.uid = member.gid = 0
                        archive.addfile(member)
                    for name in ("bin/nuv-agent", "venv/bin/python"):
                        payload = (
                            BOARD.CANDIDATE_PYTHON_WRAPPER
                            if name == "venv/bin/python"
                            else b"#!/bin/sh\nexit 0\n"
                        )
                        member = tarfile.TarInfo(name)
                        member.mode = 0o755
                        member.uid = member.gid = 0
                        member.size = len(payload)
                        archive.addfile(member, io.BytesIO(payload))
                bundle_sha = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
                bom: dict[str, object] = {
                    "schemaVersion": 2,
                    "bomId": "candidate-soak-test",
                    "bomDigest": "",
                    "releaseSequence": 2,
                    "agentVersion": "0.1.121",
                    "componentSha": "c" * 40,
                    "configSchema": "12",
                    "minUpdaterVersion": "0.2.0",
                    "targets": [
                        {
                            "productModel": "IQ9075_DEV",
                            "platformProfile": "iq9075_dev",
                            "hardwareRevision": "QCS9075-EVK",
                            "architecture": "aarch64",
                        }
                    ],
                    "artifact": {
                        "name": "candidate.tar.gz",
                        "kind": "agent-bundle",
                        "sha256": bundle_sha,
                        "sizeBytes": bundle_path.stat().st_size,
                    },
                    "builtAt": "2026-09-03T00:00:00Z",
                }
                unsigned = dict(bom)
                unsigned.pop("bomDigest")
                candidate_digest = hashlib.sha256(
                    json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                bom["bomDigest"] = f"sha256:{candidate_digest}"
                fixture.current_digest = candidate_digest

                def marker(_digest: str, version: str, sequence: int):
                    return {
                        "schemaVersion": 2,
                        "bomDigest": f"sha256:{candidate_digest}",
                        "agentVersion": version,
                        "releaseSequence": sequence,
                        "artifactDigest": f"sha256:{bundle_sha}",
                        "componentSha": "c" * 40,
                        "configSchema": "12",
                        "publisherKeyId": "release-iq9075-dev",
                    }

                fixture._marker = marker  # type: ignore[method-assign]
                fixture._seed_release(candidate_digest, "0.1.121", 2)
                (
                    fixture.root
                    / f"opt/nuv-agent/releases/{candidate_digest}/.nuvion/release-bom.json"
                ).write_text(json.dumps(bom), encoding="utf-8")
                fixture.provision("oak-fault-rollback")
                previous = fixture.root / "opt/nuv-agent/previous"
                previous.unlink()
                previous.symlink_to(f"releases/{candidate_digest}")
                rolled_back = fixture.update_state("ROLLED_BACK")
                rolled_back["artifactDigest"] = f"sha256:{bundle_sha}"
                fixture.runner.updater["update"] = rolled_back
                state = fixture.harness._load_state(fixture.run_id)
                state["oakFault"] = {
                    "armed": False,
                    "recovered": True,
                    "candidatePid": 99,
                }
                fixture.harness._save_state(fixture.run_id, state)
                self.assertTrue(fixture.harness.evidence(fixture.run_id)["complete"])

                bom_path.write_text(json.dumps(bom), encoding="utf-8")
                installed_harness = fixture.paths.candidate_harness
                installed_harness.parent.mkdir(parents=True, exist_ok=True)
                harness_bytes = b"#!/bin/sh\nexit 0\n"
                installed_harness.write_bytes(harness_bytes)
                installed_harness.chmod(0o755)
                harness_path.write_bytes(harness_bytes)
                harness_path.chmod(0o600)
                for path in (bundle_path, bom_path):
                    path.chmod(0o600)
                harness_sha = hashlib.sha256(harness_bytes).hexdigest()
                bom_sha = hashlib.sha256(bom_path.read_bytes()).hexdigest()
                pinned_harness_path = (
                    fixture.harness._candidate_harness_execution_path(
                        fixture.run_id, harness_sha
                    )
                )

                fixture.harness._agent_process_identity = (  # type: ignore[method-assign]
                    lambda expected: {
                        "pid": fixture.runner.units["nuv-agent.service"]["pid"],
                        "startTicks": int(
                            fixture.runner.units["nuv-agent.service"]["pid"]
                        )
                        * 10,
                        "bootId": "11111111-1111-4111-8111-111111111111",
                        "activeSlot": expected,
                    }
                )
                fixture.harness._anti_replay_snapshot = (  # type: ignore[method-assign]
                    lambda: {
                        "schemaVersion": 4,
                        "semanticSha256": "0" * 64,
                        "maximumCommandSequence": 2,
                        "currentReleaseSequence": "1",
                        "currentBomDigest": "sha256:" + fixture.previous_digest,
                        "latest": {
                            "commandId": fixture.command_id,
                            "sequence": 2,
                            "phase": "ROLLED_BACK",
                            "bomDigest": f"sha256:{candidate_digest}",
                            "releaseSequence": 2,
                            "healthDeadline": None,
                        },
                    }
                )
                original_run = fixture.runner.run
                execution_proof = candidate_execution_proof(
                    fixture.run_id,
                    writable_path=str(fixture.paths.state_root / fixture.run_id),
                )
                termination_proof = candidate_termination_proof(fixture.run_id)
                fixture.harness._candidate_execution_proof = (  # type: ignore[method-assign]
                    lambda unit, *, writable_path, uid_before: execution_proof
                )
                fixture.harness._monitor_candidate_unit = (  # type: ignore[method-assign]
                    lambda unit, *, expected_control_group, timeout: (
                        BOARD.CommandResult(0, "", ""),
                        candidate_collector_proof(fixture.run_id),
                    )
                )
                fixture.harness._terminate_candidate_unit = (  # type: ignore[method-assign]
                    lambda unit, *, expected_control_group=None: termination_proof
                )

                def run_with_raw(argv, *, timeout, input_bytes=None):
                    if (
                        argv
                        and argv[0] == "/usr/bin/systemd-run"
                        and any(
                            item.startswith("--unit=nuvion-candidate-soak-")
                            for item in argv
                        )
                    ):
                        self.assertEqual(timeout, 30)
                        fixture.runner.calls.append(tuple(argv))
                        installed_harness.write_bytes(b"#!/bin/sh\nexit 91\n")
                        installed_harness.chmod(0o755)
                        self.assertIn(str(pinned_harness_path), argv)
                        self.assertNotIn(str(installed_harness), argv)
                        self.assertEqual(pinned_harness_path.read_bytes(), harness_bytes)
                        self.assertEqual(
                            pinned_harness_path.stat().st_mode & 0o777, 0o500
                        )
                        output = Path(argv[argv.index("--evidence-output") + 1])
                        slot = str(
                            argv[argv.index("--expected-slot-path") + 1]
                        )
                        control_sha = argv[
                            argv.index("--expected-control-marker-sha256") + 1
                        ]
                        raw = {
                            "schemaVersion": 3,
                            "kind": "nuvion-iq9075-oak-soak-result",
                            "runId": fixture.run_id,
                            "slotKind": "candidate",
                            "outcome": {
                                "status": "passed",
                                "error": None,
                                "cleanupErrors": [],
                            },
                            "runtimeIdentity": {
                                "pythonPath": "/usr/bin/python3",
                                "sitePackagesPath": slot
                                + "/venv/lib/python3.12/site-packages",
                                "buildInfoPath": slot
                                + "/venv/lib/python3.12/site-packages/nuvion_app/build_info.py",
                                "candidateSlot": slot,
                                "controlMarkerSha256": control_sha,
                            },
                        }
                        output.write_text(
                            json.dumps(raw, sort_keys=True, separators=(",", ":"))
                            + "\n",
                            encoding="utf-8",
                        )
                        output.chmod(0o600)
                        return BOARD.CommandResult(0, "", "")
                    return original_run(
                        argv, timeout=timeout, input_bytes=input_bytes
                    )

                fixture.runner.run = run_with_raw
                slots_before = fixture.harness._slot_snapshot()
                result = fixture.harness.candidate_soak(
                    fixture.run_id,
                    candidate_bundle=bundle_path,
                    bundle_sha256=bundle_sha,
                    candidate_bom=bom_path,
                    bom_sha256=bom_sha,
                    oak_harness=harness_path,
                    harness_sha256=harness_sha,
                )
                self.assertEqual(result["outcome"]["status"], "passed")
                self.assertEqual(
                    result["candidate"]["harnessSha256"], harness_sha
                )
                self.assertFalse(pinned_harness_path.exists())
                self.assertEqual(result["pre"]["slots"], slots_before)
                self.assertEqual(result["post"]["slots"], slots_before)
                self.assertEqual(
                    result["pre"]["persistentState"],
                    result["post"]["persistentState"],
                )
                self.assertEqual(
                    result["pre"]["releaseTrees"],
                    result["post"]["releaseTrees"],
                )
                self.assertEqual(result["executionProof"], execution_proof)
                self.assertEqual(result["terminationProof"], termination_proof)
                self.assertEqual(
                    result["productionRestoration"]["transactionPhase"],
                    "RESTORED",
                )
                self.assertEqual(
                    fixture.harness._load_state(fixture.run_id)[
                        "trustTransaction"
                    ]["phase"],
                    "RESTORED",
                )
                for gate in (
                    "resourceLimitsApplied",
                    "boundedOutput",
                    "persistentStateReadOnly",
                    "persistentStateUnchanged",
                    "releaseTreesUnchanged",
                    "cgroupTerminated",
                    "productionTrustRestored",
                    "trustedSoakDuration",
                    "continuousUidIsolation",
                ):
                    self.assertTrue(result["gates"][gate])
                self.assertNotEqual(
                    result["pre"]["runtime"]["pid"],
                    result["post"]["runtime"]["pid"],
                )
                harness_call = next(
                    call
                    for call in fixture.runner.calls
                    if call
                    and call[0] == "/usr/bin/systemd-run"
                    and any(
                        item.startswith("--unit=nuvion-candidate-soak-")
                        for item in call
                    )
                )
                self.assertIn("--quiet", harness_call)
                self.assertNotIn("--wait", harness_call)
                self.assertNotIn("--pipe", harness_call)
                self.assertNotIn("--collect", harness_call)
                self.assertIn("--property=RemainAfterExit=yes", harness_call)
                self.assertIn("--property=KillMode=control-group", harness_call)
                self.assertIn("--property=SendSIGKILL=yes", harness_call)
                self.assertIn("--property=RuntimeMaxSec=720s", harness_call)
                for resource in BOARD.CANDIDATE_RESOURCE_PROPERTIES:
                    self.assertIn(f"--property={resource}", harness_call)
                for sandbox in BOARD.CANDIDATE_SANDBOX_PROPERTIES:
                    self.assertIn(f"--property={sandbox}", harness_call)
                for path in BOARD.CANDIDATE_PERSISTENT_PATHS:
                    self.assertIn(
                        f"--property=ReadOnlyPaths={path}", harness_call
                    )
                self.assertIn(
                    "--property=ReadWritePaths="
                    + str(fixture.paths.state_root / fixture.run_id),
                    harness_call,
                )
                self.assertIn("NUVION_SYSTEM_PYTHON=/usr/bin/python3", harness_call)
                self.assertIn("NUVION_AGENT_PYTHON=/usr/bin/python3", harness_call)
                self.assertIn(
                    "NUVION_AGENT_SITE_PACKAGES="
                    + str(
                        fixture.harness._candidate_slot_path(
                            fixture.run_id, str(bom["bomDigest"])
                        )
                        / BOARD.CANDIDATE_SITE_PACKAGES_RELATIVE
                    ),
                    harness_call,
                )
                deadman_call = next(
                    call
                    for call in fixture.runner.calls
                    if call
                    and call[0] == "/usr/bin/systemd-run"
                    and any(
                        item.startswith("--unit=nuvion-candidate-deadman-")
                        for item in call
                    )
                )
                self.assertIn(
                    f"--on-active={BOARD.CANDIDATE_DEADMAN_SECONDS}s",
                    deadman_call,
                )
                self.assertIn("--collect", deadman_call)
                self.assertIn("--property=LimitCORE=0", deadman_call)
                self.assertIn(
                    "--property=RuntimeMaxSec="
                    f"{BOARD.CANDIDATE_DEADMAN_RECOVERY_SECONDS}s",
                    deadman_call,
                )
                self.assertIn(
                    "--property=TimeoutStartSec="
                    f"{BOARD.CANDIDATE_DEADMAN_RECOVERY_SECONDS}s",
                    deadman_call,
                )
                self.assertIn(
                    "--property=StartLimitIntervalSec=0", deadman_call
                )
                self.assertFalse(
                    any(
                        item.startswith("--property=StartLimitBurst=")
                        for item in deadman_call
                    )
                )
                self.assertFalse(
                    any(item.startswith("--property=ExecStopPost=") for item in deadman_call)
                )
                self.assertTrue(
                    any("--candidate-deadman-only" in item for item in deadman_call)
                )
                persisted_soak = fixture.harness._load_state(fixture.run_id)[
                    "candidateSoak"
                ]
                self.assertTrue(persisted_soak["deadman"]["armed"])
                self.assertIsNot(persisted_soak["deadman"].get("stopped"), True)
                self.assertTrue(fixture.paths.active_run.exists())
                self.assertTrue(
                    fixture.harness._transaction_dir(fixture.run_id).exists()
                )
                original_purge = fixture.harness._purge_run_sensitive_material

                def fail_before_purge(_run_id: str) -> None:
                    raise BOARD.HarnessError("simulated purge boundary crash")

                fixture.harness._purge_run_sensitive_material = fail_before_purge  # type: ignore[method-assign]
                with self.assertRaisesRegex(
                    BOARD.HarnessError, "purge boundary crash"
                ):
                    fixture.harness.cleanup(fixture.run_id)
                crash_state = fixture.harness._load_state(fixture.run_id)
                self.assertTrue(crash_state["candidateSoak"]["deadman"]["armed"])
                self.assertTrue(fixture.paths.active_run.exists())
                self.assertTrue(
                    fixture.runner.deadmen[
                        fixture.harness._candidate_deadman_timer(fixture.run_id)
                    ]
                )
                fixture.harness._purge_run_sensitive_material = original_purge  # type: ignore[method-assign]
                cleanup_result = fixture.harness.cleanup(fixture.run_id)
                self.assertTrue(cleanup_result["complete"])
                self.assertTrue(
                    all(
                        value is True
                        for key, value in cleanup_result["proof"].items()
                        if key not in {"schemaVersion", "transactionPhase"}
                    )
                )
                persisted_soak = fixture.harness._load_state(fixture.run_id)[
                    "candidateSoak"
                ]
                self.assertFalse(persisted_soak["deadman"]["armed"])
                self.assertTrue(persisted_soak["deadman"]["stopped"])
                self.assertFalse(fixture.paths.active_run.exists())
                self.assertFalse(
                    fixture.harness._transaction_dir(fixture.run_id).exists()
                )
                self.assertIn("NUVION_IQ9075_OAK_SOAK_SECONDS=120", harness_call)
                self.assertIn(
                    "NUVION_IQ9075_OAK_MAX_RSS_SLOPE_MIB_PER_MIN=2",
                    harness_call,
                )
                self.assertIn(
                    "NUVION_IQ9075_OAK_MAX_RSS_RANGE_MIB=32", harness_call
                )
                self.assertFalse(
                    fixture.harness._candidate_slot_path(
                        fixture.run_id, str(bom["bomDigest"])
                    ).exists()
                )
                # Simulate SIGKILL after durable DISARMED but before the timer
                # stop/unload proof was journaled. Cleanup retry must reconcile
                # the external unit instead of silently skipping it.
                crash_state = fixture.harness._load_state(fixture.run_id)
                crash_deadman = crash_state["candidateSoak"]["deadman"]
                crash_deadman.pop("stopped")
                timer = fixture.harness._candidate_deadman_timer(fixture.run_id)
                fixture.runner.deadmen[timer] = True
                fixture.harness._save_state(fixture.run_id, crash_state)
                resumed = fixture.harness.cleanup(fixture.run_id)
                self.assertTrue(resumed["complete"])
                self.assertTrue(resumed["idempotent"])
                reconciled = fixture.harness._load_state(fixture.run_id)[
                    "candidateSoak"
                ]["deadman"]
                self.assertTrue(reconciled["stopped"])
            finally:
                fixture.close()
                for path in (bundle_path, bom_path, harness_path):
                    path.unlink(missing_ok=True)

    def test_staging_crash_cleanup_removes_only_journal_bound_incoming_tree(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                digest = "a" * 64
                slot = fixture.harness._candidate_slot_path(
                    fixture.run_id, f"sha256:{digest}"
                )
                incoming = fixture.harness._candidate_incoming_path(slot)
                incoming.mkdir(parents=True, mode=0o700)
                partial = incoming / "venv/bin/python"
                partial.parent.mkdir(parents=True, mode=0o755)
                partial.write_bytes(b"partial candidate")
                partial.chmod(0o644)
                state = fixture.harness._load_state(fixture.run_id)
                state["candidateSoak"] = {
                    "phase": "STAGING",
                    "candidateSlot": str(slot),
                    "candidateIncomingPath": str(incoming),
                    "inputDigests": {},
                }
                fixture.harness._save_state(fixture.run_id, state)

                restarted = BOARD.BoardHarness(
                    paths=fixture.paths,
                    runner=fixture.runner,
                    root_uid=os.getuid(),
                    root_gid=os.getgid(),
                    nuvion_gid=os.getgid(),
                    nuvion_uid=4242,
                    tool_path=ROOT / "packaging/dev/iq9075-board-e2e.py",
                    enforce_installed_tool=False,
                    monotonic=fixture.clock.monotonic,
                    sleeper=fixture.clock.sleep,
                    disk_usage=lambda _path: SimpleNamespace(free=8 * 1024**3),
                    usb_write_hook=fixture._usb_hook,
                )
                result = restarted.cleanup(fixture.run_id)
                self.assertTrue(result["complete"])
                self.assertTrue(result["recovered"])
                self.assertFalse(incoming.exists())
                recovered_state = restarted._load_state(fixture.run_id)
                self.assertTrue(
                    recovered_state["candidateSoak"][
                        "candidateIncomingRecovered"
                    ]
                )
            finally:
                fixture.close()

    def test_candidate_persistent_snapshot_detects_state_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                before = fixture.harness._candidate_persistent_state_snapshot()
                state_file = fixture.root / "var/lib/nuv-agent/events.sqlite3"
                state_file.write_bytes(b"candidate mutation")
                after = fixture.harness._candidate_persistent_state_snapshot()
                self.assertNotEqual(before, after)
                self.assertNotEqual(before["sha256"], after["sha256"])
                serialized = json.dumps(before, sort_keys=True)
                self.assertNotIn("must-stay-private", serialized)
                self.assertNotIn("event-state", serialized)
            finally:
                fixture.close()

    def test_staging_crash_cleanup_rejects_unjournaled_or_unsafe_incoming(
        self,
    ) -> None:
        for unsafe_kind in ("missing-journal", "path-mismatch", "symlink"):
            with (
                self.subTest(unsafe_kind=unsafe_kind),
                tempfile.TemporaryDirectory() as directory,
            ):
                fixture = HarnessFixture(Path(directory))
                try:
                    slot = fixture.harness._candidate_slot_path(
                        fixture.run_id, "sha256:" + "a" * 64
                    )
                    incoming = fixture.harness._candidate_incoming_path(slot)
                    incoming.parent.mkdir(parents=True, exist_ok=True)
                    state = fixture.harness._load_state(fixture.run_id)
                    journal_path = str(incoming)
                    if unsafe_kind in {"missing-journal", "path-mismatch"}:
                        incoming.mkdir(mode=0o700)
                        if unsafe_kind == "path-mismatch":
                            journal_path = str(
                                incoming.parent / ".different.incoming"
                            )
                    else:
                        victim = fixture.root / "must-not-delete"
                        victim.mkdir()
                        incoming.symlink_to(victim, target_is_directory=True)
                    candidate_soak = {
                        "phase": "STAGING",
                        "candidateSlot": str(slot),
                        "inputDigests": {},
                    }
                    if unsafe_kind != "missing-journal":
                        candidate_soak["candidateIncomingPath"] = journal_path
                    state["candidateSoak"] = candidate_soak
                    fixture.harness._save_state(fixture.run_id, state)
                    with self.assertRaises(BOARD.HarnessError):
                        fixture.harness.cleanup(fixture.run_id)
                    self.assertTrue(incoming.exists() or incoming.is_symlink())
                    if unsafe_kind == "symlink":
                        self.assertTrue(victim.exists())
                finally:
                    fixture.close()

    def test_candidate_soak_cleanup_recovers_interrupted_running_phase(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                fixture.harness.preflight(fixture.run_id)
                slots = fixture.harness._slot_snapshot()
                units = fixture.harness._unit_snapshot()
                baseline_pid = int(
                    fixture.runner.units["nuv-agent.service"]["pid"]
                )
                fixture.paths.candidate_root.mkdir(parents=True, mode=0o755)
                candidate_slot = (
                    fixture.paths.candidate_root
                    / f"{fixture.run_id}-{'a' * 64}"
                )
                candidate_slot.mkdir(mode=0o755)
                anti_replay = {
                    "schemaVersion": 4,
                    "semanticSha256": "0" * 64,
                    "maximumCommandSequence": 2,
                    "currentReleaseSequence": "1",
                    "currentBomDigest": "sha256:" + fixture.previous_digest,
                    "latest": {
                        "commandId": fixture.command_id,
                        "sequence": 2,
                        "phase": "ROLLED_BACK",
                        "bomDigest": "sha256:" + fixture.current_digest,
                        "releaseSequence": 2,
                        "healthDeadline": None,
                    },
                }
                persistent_state = (
                    fixture.harness._candidate_persistent_state_snapshot()
                )
                state = fixture.harness._load_state(fixture.run_id)
                state["candidateSoak"] = {
                    "phase": "RUNNING",
                    "runningAt": "2026-09-03T00:01:00Z",
                    "candidateSlot": str(candidate_slot),
                    "baselineSlots": slots,
                    "releaseTreesBefore": fixture.harness._release_tree_snapshot(
                        slots
                    ),
                    "unitsBefore": units,
                    "baselineRuntime": {
                        "pid": baseline_pid,
                        "startTicks": baseline_pid * 10,
                        "bootId": "11111111-1111-4111-8111-111111111111",
                        "activeSlot": slots["current"],
                    },
                    "antiReplay": anti_replay,
                    "persistentStateBefore": persistent_state,
                    "rollbackTerminal": anti_replay["latest"],
                    "oakBefore": fixture.harness.verify_oak(),
                    "inputDigests": {},
                }
                fixture.harness._save_state(fixture.run_id, state)
                fixture.harness._anti_replay_snapshot = (  # type: ignore[method-assign]
                    lambda: anti_replay
                )
                fixture.runner.updater["update"] = {"phase": "ROLLED_BACK"}
                fixture.harness._agent_process_identity = (  # type: ignore[method-assign]
                    lambda expected: {
                        # Linux may reuse both PID and start ticks after reboot.
                        # The changed boot identity still proves a fresh process.
                        "pid": baseline_pid,
                        "startTicks": baseline_pid * 10,
                        "bootId": "22222222-2222-4222-8222-222222222222",
                        "activeSlot": expected,
                    }
                )
                result = fixture.harness.cleanup(fixture.run_id)
                self.assertTrue(result["complete"])
                self.assertTrue(result["recovered"])
                self.assertFalse(candidate_slot.exists())
                self.assertTrue(
                    fixture.runner.units["nuv-agent.service"]["active"]
                )
                recovered_state = fixture.harness._load_state(fixture.run_id)
                self.assertFalse(recovered_state["candidateSoak"]["passed"])
                self.assertEqual(
                    recovered_state["candidateSoak"]["failureCode"],
                    "INTERRUPTED_CANDIDATE_SOAK",
                )
                self.assertFalse(fixture.paths.active_run.exists())
            finally:
                fixture.close()

    def test_candidate_deadman_recovers_running_phase_without_operation_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                fixture.provision("oak-fault-rollback")
                slots = fixture.harness._slot_snapshot()
                units = fixture.harness._unit_snapshot()
                baseline_pid = int(
                    fixture.runner.units["nuv-agent.service"]["pid"]
                )
                candidate_slot = (
                    fixture.paths.candidate_root
                    / f"{fixture.run_id}-{'a' * 64}"
                )
                candidate_slot.mkdir(parents=True, mode=0o755)
                anti_replay = {
                    "schemaVersion": 4,
                    "semanticSha256": "0" * 64,
                    "maximumCommandSequence": 2,
                    "currentReleaseSequence": "1",
                    "currentBomDigest": "sha256:" + fixture.previous_digest,
                    "latest": {
                        "commandId": fixture.command_id,
                        "sequence": 2,
                        "phase": "ROLLED_BACK",
                        "bomDigest": "sha256:" + fixture.current_digest,
                        "releaseSequence": 2,
                        "healthDeadline": None,
                    },
                }
                deadman_unit = fixture.harness._candidate_deadman_unit(
                    fixture.run_id
                )
                fixture.runner.deadmen[deadman_unit] = True
                state = fixture.harness._load_state(fixture.run_id)
                state["candidateSoak"] = {
                    "phase": "RUNNING",
                    "runningAt": "2026-09-03T00:01:00Z",
                    "candidateSlot": str(candidate_slot),
                    "baselineSlots": slots,
                    "releaseTreesBefore": fixture.harness._release_tree_snapshot(
                        slots
                    ),
                    "unitsBefore": units,
                    "baselineRuntime": {
                        "pid": baseline_pid,
                        "startTicks": baseline_pid * 10,
                        "bootId": "11111111-1111-4111-8111-111111111111",
                        "activeSlot": slots["current"],
                    },
                    "antiReplay": anti_replay,
                    "persistentStateBefore": (
                        fixture.harness._candidate_persistent_state_snapshot()
                    ),
                    "rollbackTerminal": anti_replay["latest"],
                    "oakBefore": fixture.harness.verify_oak(),
                    "inputDigests": {},
                    "deadman": {
                        "unit": deadman_unit,
                        "armed": True,
                        "lifecycle": "ARMED",
                        "writerEpoch": "33333333-3333-4333-8333-333333333333",
                        "controller": {
                            "pid": 999999,
                            "startTicks": 12345,
                            "bootId": "11111111-1111-4111-8111-111111111111",
                        },
                        "armedAt": "2026-09-03T00:00:59Z",
                    },
                }
                fixture.harness._save_state(fixture.run_id, state)
                fixture.harness._anti_replay_snapshot = (  # type: ignore[method-assign]
                    lambda: anti_replay
                )
                fixture.runner.updater["update"] = {"phase": "ROLLED_BACK"}
                fixture.harness._agent_process_identity = (  # type: ignore[method-assign]
                    lambda expected: {
                        "pid": baseline_pid + 1,
                        "startTicks": (baseline_pid + 1) * 10,
                        "bootId": "22222222-2222-4222-8222-222222222222",
                        "activeSlot": expected,
                    }
                )

                recovered = fixture.harness.cleanup(
                    fixture.run_id, candidate_deadman_only=True
                )

                self.assertTrue(recovered["complete"])
                self.assertTrue(recovered["recovered"])
                self.assertFalse(fixture.paths.active_run.exists())
                self.assertTrue(
                    fixture.runner.units["nuv-agent.service"]["active"]
                )
                self.assertFalse(candidate_slot.exists())
                final = fixture.harness._load_state(fixture.run_id)["candidateSoak"]
                self.assertEqual(final["phase"], "RESTORED")
                self.assertFalse(final["deadman"]["armed"])
                self.assertEqual(final["deadman"]["recoveredBy"], "deadman")
                self.assertEqual(
                    fixture.harness._load_state(fixture.run_id)[
                        "trustTransaction"
                    ]["phase"],
                    "RESTORED",
                )
                self.assertEqual(
                    final["productionRestoration"]["transactionPhase"],
                    "RESTORED",
                )
                self.assertTrue(final["terminationProof"]["cgroupRemoved"])
                self.assertEqual(
                    final["terminationProof"]["loadState"], "not-found"
                )
            finally:
                fixture.close()

    def test_candidate_deadman_fences_before_writer_lock_and_preserves_epoch_on_resume(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                fixture.harness.preflight(fixture.run_id)
                unit = fixture.harness._candidate_deadman_unit(fixture.run_id)
                timer = fixture.harness._candidate_deadman_timer(fixture.run_id)
                fixture.runner.deadmen[timer] = True
                epoch = "33333333-3333-4333-8333-333333333333"
                controller = {
                    "pid": 999999,
                    "startTicks": 12345,
                    "bootId": "11111111-1111-4111-8111-111111111111",
                }
                state = fixture.harness._load_state(fixture.run_id)
                soak = {
                    "phase": "RUNNING",
                    "deadman": {
                        "unit": unit,
                        "armed": True,
                        "lifecycle": "ARMED",
                        "writerEpoch": epoch,
                        "controller": controller,
                    },
                }
                state["candidateSoak"] = soak
                fixture.harness._save_state(fixture.run_id, state)
                with self.assertRaisesRegex(BOARD.HarnessError, "requires recovery"):
                    fixture.harness._ensure_candidate_deadman(
                        fixture.run_id, state, soak
                    )
                unchanged = fixture.harness._load_state(fixture.run_id)[
                    "candidateSoak"
                ]["deadman"]
                self.assertEqual(unchanged["writerEpoch"], epoch)
                self.assertEqual(unchanged["controller"], controller)

                with (
                    mock.patch.object(
                        fixture.harness,
                        "_controller_identity_is_live",
                        side_effect=[True, False],
                    ),
                    mock.patch.object(BOARD.os, "kill") as kill,
                ):
                    fixture.harness._fence_candidate_controller(controller)
                kill.assert_called_once_with(controller["pid"], signal.SIGTERM)

                order: list[str] = []

                @contextmanager
                def writer_lock(_run_id: str, *, timeout: float = 0):
                    self.assertEqual(timeout, 30)
                    order.append("lock-enter")
                    try:
                        yield
                    finally:
                        order.append("lock-exit")

                fixture.harness._fence_candidate_controller = (  # type: ignore[method-assign]
                    lambda identity: order.append("fence")
                )
                fixture.harness._candidate_writer_lock = writer_lock  # type: ignore[method-assign]
                fixture.harness._candidate_deadman_cleanup_fenced = (  # type: ignore[method-assign]
                    lambda _run_id, *, epoch: (
                        order.append("cleanup")
                        or {
                            "schemaVersion": 1,
                            "runId": fixture.run_id,
                            "candidateDeadmanOnly": True,
                            "recovered": True,
                            "complete": True,
                        }
                    )
                )
                fixture.harness._candidate_deadman_cleanup(fixture.run_id)
                self.assertEqual(
                    order, ["fence", "lock-enter", "cleanup", "lock-exit"]
                )
            finally:
                fixture.close()

    def test_candidate_orphan_timer_restores_applied_trust_before_prepare_journal(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                fixture.provision("oak-fault-rollback")
                state = fixture.harness._load_state(fixture.run_id)
                self.assertNotIn("candidateSoak", state)
                self.assertEqual(state["trustTransaction"]["phase"], "APPLIED")
                recovered = fixture.harness.cleanup(
                    fixture.run_id,
                    candidate_deadman_only=True,
                    candidate_deadman_epoch=(
                        "33333333-3333-4333-8333-333333333333"
                    ),
                    candidate_controller={
                        "pid": 999999,
                        "startTicks": 12345,
                        "bootId": "11111111-1111-4111-8111-111111111111",
                    },
                )
                self.assertTrue(recovered["complete"])
                final = fixture.harness._load_state(fixture.run_id)
                self.assertEqual(final["trustTransaction"]["phase"], "RESTORED")
                self.assertEqual(
                    final["candidateRecovery"]["productionRestoration"][
                        "transactionPhase"
                    ],
                    "RESTORED",
                )
                self.assertFalse(fixture.paths.active_run.exists())
                self.assertFalse(
                    fixture.harness._transaction_dir(fixture.run_id).exists()
                )
            finally:
                fixture.close()

    def test_candidate_deadman_recovers_staging_tree_and_production_trust(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                fixture.provision("oak-fault-rollback")
                slot = fixture.harness._candidate_slot_path(
                    fixture.run_id, "sha256:" + "a" * 64
                )
                incoming = fixture.harness._candidate_incoming_path(slot)
                incoming.mkdir(parents=True, mode=0o700)
                partial = incoming / "partial"
                partial.write_bytes(b"partial")
                partial.chmod(0o600)
                unit = fixture.harness._candidate_deadman_unit(fixture.run_id)
                state = fixture.harness._load_state(fixture.run_id)
                state["candidateSoak"] = {
                    "phase": "STAGING",
                    "candidateSlot": str(slot),
                    "candidateIncomingPath": str(incoming),
                    "inputDigests": {},
                    "deadman": {
                        "unit": unit,
                        "armed": True,
                        "lifecycle": "ARMED",
                        "writerEpoch": "33333333-3333-4333-8333-333333333333",
                        "controller": {
                            "pid": 999999,
                            "startTicks": 12345,
                            "bootId": "11111111-1111-4111-8111-111111111111",
                        },
                    },
                }
                fixture.harness._save_state(fixture.run_id, state)
                recovered = fixture.harness.cleanup(
                    fixture.run_id, candidate_deadman_only=True
                )
                self.assertTrue(recovered["complete"])
                self.assertFalse(incoming.exists())
                final = fixture.harness._load_state(fixture.run_id)
                self.assertEqual(final["candidateSoak"]["phase"], "RESTORED")
                self.assertEqual(final["trustTransaction"]["phase"], "RESTORED")
                self.assertFalse(fixture.paths.active_run.exists())
            finally:
                fixture.close()

    def test_candidate_soak_cleanup_fails_closed_on_release_tree_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                fixture.harness.preflight(fixture.run_id)
                slots = fixture.harness._slot_snapshot()
                units = fixture.harness._unit_snapshot()
                baseline_trees = fixture.harness._release_tree_snapshot(slots)
                candidate_slot = (
                    fixture.paths.candidate_root
                    / f"{fixture.run_id}-{'a' * 64}"
                )
                candidate_slot.mkdir(parents=True, mode=0o755)
                state = fixture.harness._load_state(fixture.run_id)
                state["candidateSoak"] = {
                    "phase": "QUIESCING",
                    "candidateSlot": str(candidate_slot),
                    "baselineSlots": slots,
                    "releaseTreesBefore": baseline_trees,
                    "unitsBefore": units,
                    "baselineRuntime": {
                        "pid": 100,
                        "startTicks": 1000,
                        "bootId": "11111111-1111-4111-8111-111111111111",
                        "activeSlot": slots["current"],
                    },
                    "oakBefore": fixture.harness.verify_oak(),
                    "inputDigests": {},
                }
                fixture.harness._save_state(fixture.run_id, state)
                changed = (
                    fixture.paths.install_root
                    / str(slots["current"])
                    / ".nuvion/release.json"
                )
                changed.write_text("mutated\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    BOARD.HarnessError, "runtime remains fail-closed"
                ):
                    fixture.harness.cleanup(fixture.run_id)
                self.assertTrue(candidate_slot.exists())
                self.assertTrue(
                    all(not item["active"] for item in fixture.runner.units.values())
                )
            finally:
                fixture.close()

    def test_candidate_unit_kills_entire_cgroup_before_restore(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                unit = fixture.harness._candidate_unit(fixture.run_id)
                cgroup = fixture.paths.cgroup_root / "system.slice" / unit
                cgroup.mkdir(parents=True)
                events = cgroup / "cgroup.events"
                # A setsid/daemonized descendant remains a member of the service
                # subtree even when the root cgroup.procs file itself is empty.
                (cgroup / "cgroup.procs").write_text("", encoding="ascii")
                events.write_text("populated 1\nfrozen 0\n", encoding="ascii")
                original_run = fixture.runner.run

                def run_with_cgroup(argv, *, timeout, input_bytes=None):
                    call = tuple(argv)
                    if (
                        call[:3]
                        == (
                            "/usr/bin/systemctl",
                            "show",
                            "--property=ControlGroup",
                        )
                        and call[-1] == unit
                    ):
                        fixture.runner.calls.append(call)
                        return BOARD.CommandResult(
                            0, f"/system.slice/{unit}\n", ""
                        )
                    if (
                        call[:2] == ("/usr/bin/systemctl", "kill")
                        and "--signal=SIGKILL" in call
                        and call[-1] == unit
                    ):
                        events.write_text("populated 0\nfrozen 0\n", encoding="ascii")
                    result = original_run(
                        argv, timeout=timeout, input_bytes=input_bytes
                    )
                    if (
                        call[:2] == ("/usr/bin/systemctl", "stop")
                        and events.exists()
                        and "populated 0" in events.read_text(encoding="ascii")
                    ):
                        events.unlink()
                        cgroup.joinpath("cgroup.procs").unlink(missing_ok=True)
                        cgroup.rmdir()
                    return result

                fixture.runner.run = run_with_cgroup
                proof = fixture.harness._terminate_candidate_unit(unit)
                self.assertFalse(events.exists())
                self.assertFalse(proof["resetPerformed"])
                self.assertTrue(
                    any(
                        call[:2] == ("/usr/bin/systemctl", "kill")
                        and "--kill-whom=all" in call
                        and "--signal=SIGKILL" in call
                        for call in fixture.runner.calls
                    )
                )
            finally:
                fixture.close()

    def test_candidate_unit_rejects_nonempty_cgroup_after_sigkill(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                unit = fixture.harness._candidate_unit(fixture.run_id)
                cgroup = fixture.paths.cgroup_root / "system.slice" / unit
                cgroup.mkdir(parents=True)
                (cgroup / "cgroup.events").write_text(
                    "populated 1\nfrozen 0\n", encoding="ascii"
                )
                original_run = fixture.runner.run

                def run_with_stuck_cgroup(argv, *, timeout, input_bytes=None):
                    call = tuple(argv)
                    if (
                        call[:3]
                        == (
                            "/usr/bin/systemctl",
                            "show",
                            "--property=ControlGroup",
                        )
                        and call[-1] == unit
                    ):
                        return BOARD.CommandResult(
                            0, f"/system.slice/{unit}\n", ""
                        )
                    return original_run(
                        argv, timeout=timeout, input_bytes=input_bytes
                    )

                fixture.runner.run = run_with_stuck_cgroup
                with self.assertRaisesRegex(
                    BOARD.HarnessError, "cgroup is not empty"
                ):
                    fixture.harness._terminate_candidate_unit(unit)
            finally:
                fixture.close()

    def test_candidate_unit_fails_closed_on_systemd_kill_stop_or_reset_failure(
        self,
    ) -> None:
        for failed_action, populated in (
            ("kill", True),
            ("stop", False),
            ("reset-failed", False),
        ):
            with self.subTest(action=failed_action), tempfile.TemporaryDirectory() as directory:
                fixture = HarnessFixture(Path(directory))
                try:
                    unit = fixture.harness._candidate_unit(fixture.run_id)
                    cgroup = fixture.paths.cgroup_root / "system.slice" / unit
                    cgroup.mkdir(parents=True)
                    events = cgroup / "cgroup.events"
                    events.write_text(
                        f"populated {1 if populated else 0}\nfrozen 0\n",
                        encoding="ascii",
                    )
                    original_run = fixture.runner.run

                    def run_with_failure(argv, *, timeout, input_bytes=None):
                        call = tuple(argv)
                        if (
                            call[:3]
                            == (
                                "/usr/bin/systemctl",
                                "show",
                                "--property=ControlGroup",
                            )
                            and call[-1] == unit
                        ):
                            return BOARD.CommandResult(
                                0, f"/system.slice/{unit}\n", ""
                            )
                        if (
                            call[:2] == ("/usr/bin/systemctl", "show")
                            and "--property=LoadState" in call
                        ):
                            names = {
                                item.split("=", 1)[1]
                                for item in call
                                if item.startswith("--property=")
                            }
                            values = {
                                "ActiveState": "inactive",
                                "ControlGroup": f"/system.slice/{unit}",
                                "LoadState": "loaded",
                            }
                            return BOARD.CommandResult(
                                0,
                                "".join(
                                    f"{name}={values[name]}\n"
                                    for name in sorted(names)
                                ),
                                "",
                            )
                        if call[:2] == ("/usr/bin/systemctl", failed_action):
                            return BOARD.CommandResult(1, "", "failed")
                        return original_run(
                            argv, timeout=timeout, input_bytes=input_bytes
                        )

                    fixture.runner.run = run_with_failure
                    with self.assertRaisesRegex(
                        BOARD.HarnessError,
                        f"systemd {failed_action.split('-')[0]}",
                    ):
                        fixture.harness._terminate_candidate_unit(unit)
                finally:
                    fixture.close()

    def test_candidate_unit_accepts_verified_unloaded_boundary_after_reboot(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                unit = fixture.harness._candidate_unit(fixture.run_id)
                proof = fixture.harness._terminate_candidate_unit(
                    unit,
                    expected_control_group="/system.slice/" + unit,
                )
                self.assertFalse(proof["initialPresent"])
                self.assertFalse(proof["recursivePopulated"])
                self.assertTrue(proof["cgroupRemoved"])
                self.assertFalse(proof["resetPerformed"])
            finally:
                fixture.close()

    def test_candidate_unit_does_not_reset_successfully_unloaded_transient(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                unit = fixture.harness._candidate_unit(fixture.run_id)
                cgroup = fixture.paths.cgroup_root / "system.slice" / unit
                cgroup.mkdir(parents=True)
                (cgroup / "cgroup.events").write_text(
                    "populated 0\nfrozen 0\n", encoding="ascii"
                )
                original_run = fixture.runner.run
                loaded = True

                def run_with_unload(argv, *, timeout, input_bytes=None):
                    nonlocal loaded
                    call = tuple(argv)
                    if (
                        call[:3]
                        == (
                            "/usr/bin/systemctl",
                            "show",
                            "--property=ControlGroup",
                        )
                        and call[-1] == unit
                    ):
                        return BOARD.CommandResult(
                            0 if loaded else 1,
                            f"/system.slice/{unit}\n" if loaded else "",
                            "",
                        )
                    if call[:2] == ("/usr/bin/systemctl", "stop"):
                        loaded = False
                        cgroup.joinpath("cgroup.events").unlink()
                        cgroup.rmdir()
                    return original_run(
                        argv, timeout=timeout, input_bytes=input_bytes
                    )

                fixture.runner.run = run_with_unload
                proof = fixture.harness._terminate_candidate_unit(unit)
                self.assertFalse(proof["resetPerformed"])
                self.assertTrue(proof["cgroupRemoved"])
                self.assertFalse(
                    any(
                        call[:2] == ("/usr/bin/systemctl", "reset-failed")
                        for call in fixture.runner.calls
                    )
                )
            finally:
                fixture.close()

    def test_candidate_failed_transient_resets_then_proves_unloaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                unit = fixture.harness._candidate_unit(fixture.run_id)
                cgroup = fixture.paths.cgroup_root / "system.slice" / unit
                cgroup.mkdir(parents=True)
                (cgroup / "cgroup.events").write_text(
                    "populated 0\nfrozen 0\n", encoding="ascii"
                )
                original_run = fixture.runner.run
                loaded = True

                def run_failed_then_unloaded(argv, *, timeout, input_bytes=None):
                    nonlocal loaded
                    call = tuple(argv)
                    if (
                        call[:3]
                        == (
                            "/usr/bin/systemctl",
                            "show",
                            "--property=ControlGroup",
                        )
                        and call[-1] == unit
                    ):
                        return BOARD.CommandResult(
                            0 if loaded else 1,
                            f"/system.slice/{unit}\n" if loaded else "",
                            "",
                        )
                    if (
                        call[:2] == ("/usr/bin/systemctl", "show")
                        and "--property=LoadState" in call
                    ):
                        names = {
                            item.split("=", 1)[1]
                            for item in call
                            if item.startswith("--property=")
                        }
                        values = {
                            "ActiveState": "failed" if loaded else "inactive",
                            "ControlGroup": f"/system.slice/{unit}" if loaded else "",
                            "LoadState": "loaded" if loaded else "not-found",
                        }
                        return BOARD.CommandResult(
                            0,
                            "".join(
                                f"{name}={values[name]}\n" for name in sorted(names)
                            ),
                            "",
                        )
                    result = original_run(
                        argv, timeout=timeout, input_bytes=input_bytes
                    )
                    if call[:2] == ("/usr/bin/systemctl", "reset-failed"):
                        loaded = False
                        cgroup.joinpath("cgroup.events").unlink()
                        cgroup.rmdir()
                    return result

                fixture.runner.run = run_failed_then_unloaded
                proof = fixture.harness._terminate_candidate_unit(unit)
                self.assertTrue(proof["resetPerformed"])
                self.assertEqual(proof["loadState"], "not-found")
                self.assertEqual(proof["activeState"], "inactive")
                self.assertTrue(proof["cgroupRemoved"])
            finally:
                fixture.close()

    def test_candidate_async_unit_wait_requires_terminal_exec_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                states = [
                    {
                        "ActiveState": "active",
                        "SubState": "running",
                        "Result": "success",
                        "ExecMainCode": "0",
                        "ExecMainStatus": "0",
                    },
                    {
                        "ActiveState": "active",
                        "SubState": "exited",
                        "Result": "success",
                        "ExecMainCode": "1",
                        "ExecMainStatus": "0",
                    },
                ]
                fixture.harness._show_candidate_properties = (  # type: ignore[method-assign]
                    lambda unit, names: states.pop(0)
                )
                result = fixture.harness._wait_candidate_unit(
                    fixture.harness._candidate_unit(fixture.run_id), timeout=5
                )
                self.assertEqual(result.returncode, 0)
                self.assertFalse(states)
                fixture.harness._show_candidate_properties = (  # type: ignore[method-assign]
                    lambda unit, names: {
                        "ActiveState": "failed",
                        "SubState": "failed",
                        "Result": "exit-code",
                        "ExecMainCode": "1",
                        "ExecMainStatus": "23",
                    }
                )
                failed = fixture.harness._wait_candidate_unit(
                    fixture.harness._candidate_unit(fixture.run_id), timeout=5
                )
                self.assertEqual(failed.returncode, 23)
            finally:
                fixture.close()

    def test_candidate_outer_collector_rejects_short_or_transient_uid_escape(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                unit = fixture.harness._candidate_unit(fixture.run_id)
                group = "/system.slice/" + unit
                fixture.harness._show_candidate_properties = (  # type: ignore[method-assign]
                    lambda _unit, _names: {
                        "ActiveState": "active",
                        "SubState": "exited",
                        "Result": "success",
                        "ExecMainCode": "1",
                        "ExecMainStatus": "0",
                    }
                )
                fixture.harness._nuvion_process_cgroups = (  # type: ignore[method-assign]
                    lambda: {9001: group}
                )
                result, proof = fixture.harness._monitor_candidate_unit(
                    unit, expected_control_group=group, timeout=5
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(proof["durationSatisfied"])

                scans = iter(({9001: group}, {9001: "/user.slice/escape.scope"}))
                fixture.harness._nuvion_process_cgroups = (  # type: ignore[method-assign]
                    lambda: next(scans)
                )
                fixture.harness._show_candidate_properties = (  # type: ignore[method-assign]
                    lambda _unit, _names: {
                        "ActiveState": "active",
                        "SubState": "running",
                        "Result": "success",
                        "ExecMainCode": "0",
                        "ExecMainStatus": "0",
                    }
                )
                result, proof = fixture.harness._monitor_candidate_unit(
                    unit, expected_control_group=group, timeout=5
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(proof["allSamplesWithinCgroup"])
                self.assertEqual(proof["escapeDetected"]["pids"], [9001])
            finally:
                fixture.close()

    def test_candidate_tar_prescan_bounds_extensions_and_exact_stream_limit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                header = bytearray(512)
                header[0:5] = b"entry"
                header[124:136] = b"00000000000\0"
                header[156:157] = b"x"
                with self.assertRaisesRegex(BOARD.HarnessError, "member type"):
                    fixture.harness._prescan_candidate_archive(
                        io.BytesIO(bytes(header))
                    )

                long_header = bytearray(header)
                long_header[124:136] = b"00000000021\0"  # 17 bytes.
                long_header[156:157] = b"L"
                with mock.patch.object(BOARD, "MAX_CANDIDATE_TAR_METADATA_BYTES", 16):
                    with self.assertRaisesRegex(
                        BOARD.HarnessError, "extended metadata"
                    ):
                        fixture.harness._prescan_candidate_archive(
                            io.BytesIO(bytes(long_header))
                        )

                sparse = bytearray(header)
                sparse[156:157] = b"S"
                with self.assertRaisesRegex(BOARD.HarnessError, "member type"):
                    fixture.harness._prescan_candidate_archive(
                        io.BytesIO(bytes(sparse))
                    )

                long_payload = b"longname\0" + b"\0" * (512 - 9)
                file_header = bytearray(header)
                file_header[156:157] = b"0"
                long_nine = bytearray(long_header)
                long_nine[124:136] = b"00000000011\0"
                chained = (
                    bytes(long_nine)
                    + long_payload
                    + bytes(long_nine)
                    + long_payload
                )
                with self.assertRaisesRegex(BOARD.HarnessError, "extension chain"):
                    fixture.harness._prescan_candidate_archive(io.BytesIO(chained))
                aggregate = (
                    bytes(long_nine)
                    + long_payload
                    + bytes(file_header)
                    + bytes(long_nine)
                )
                with mock.patch.object(
                    BOARD, "MAX_CANDIDATE_TAR_METADATA_TOTAL_BYTES", 16
                ):
                    with self.assertRaisesRegex(
                        BOARD.HarnessError, "aggregate metadata"
                    ):
                        fixture.harness._prescan_candidate_archive(
                            io.BytesIO(aggregate)
                        )

                directory_info = tarfile.TarInfo("payload")
                directory_info.type = tarfile.DIRTYPE
                directory_info.uid = directory_info.gid = 0
                exact = directory_info.tobuf() + b"\0" * 1024
                compressed = gzip.compress(exact + b"x")
                with mock.patch.object(
                    BOARD, "MAX_CANDIDATE_TAR_STREAM_BYTES", len(exact)
                ):
                    with self.assertRaisesRegex(
                        BOARD.HarnessError, "stream exceeds hard byte limit"
                    ):
                        fixture.harness._prescan_candidate_archive(
                            io.BytesIO(compressed)
                        )
                with self.assertRaisesRegex(
                    BOARD.HarnessError, "non-zero trailing data"
                ):
                    fixture.harness._prescan_candidate_archive(
                        io.BytesIO(gzip.compress(exact) + gzip.compress(b"NOT-A-TAR"))
                    )
                with self.assertRaisesRegex(
                    BOARD.HarnessError, "canonical end markers"
                ):
                    fixture.harness._prescan_candidate_archive(
                        io.BytesIO(directory_info.tobuf())
                    )
            finally:
                fixture.close()

    def test_candidate_soak_cleanup_fails_closed_on_anti_replay_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                fixture.harness.preflight(fixture.run_id)
                slots = fixture.harness._slot_snapshot()
                units = fixture.harness._unit_snapshot()
                baseline_pid = int(
                    fixture.runner.units["nuv-agent.service"]["pid"]
                )
                fixture.paths.candidate_root.mkdir(parents=True, mode=0o755)
                candidate_slot = (
                    fixture.paths.candidate_root
                    / f"{fixture.run_id}-{'a' * 64}"
                )
                candidate_slot.mkdir(mode=0o755)
                anti_replay = {
                    "schemaVersion": 4,
                    "semanticSha256": "0" * 64,
                    "maximumCommandSequence": 2,
                    "currentReleaseSequence": "1",
                    "currentBomDigest": "sha256:" + fixture.previous_digest,
                    "latest": None,
                }
                state = fixture.harness._load_state(fixture.run_id)
                state["candidateSoak"] = {
                    "phase": "RUNNING",
                    "runningAt": "2026-09-03T00:01:00Z",
                    "candidateSlot": str(candidate_slot),
                    "baselineSlots": slots,
                    "releaseTreesBefore": fixture.harness._release_tree_snapshot(
                        slots
                    ),
                    "unitsBefore": units,
                    "baselineRuntime": {
                        "pid": baseline_pid,
                        "startTicks": baseline_pid * 10,
                        "bootId": "11111111-1111-4111-8111-111111111111",
                        "activeSlot": slots["current"],
                    },
                    "antiReplay": anti_replay,
                    "rollbackTerminal": anti_replay["latest"],
                    "oakBefore": fixture.harness.verify_oak(),
                    "inputDigests": {},
                }
                fixture.harness._save_state(fixture.run_id, state)
                fixture.harness._anti_replay_snapshot = (  # type: ignore[method-assign]
                    lambda: {**anti_replay, "maximumCommandSequence": 3}
                )

                with self.assertRaisesRegex(
                    BOARD.HarnessError, "runtime remains fail-closed"
                ):
                    fixture.harness.cleanup(fixture.run_id)
                self.assertTrue(
                    all(
                        item["active"] is False
                        for item in fixture.runner.units.values()
                    )
                )
                self.assertTrue(candidate_slot.exists())
            finally:
                fixture.close()

    def test_transaction_guard_spans_backup_apply_and_run_candidate_gap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            controller = {
                "pid": 999999,
                "startTicks": 424242,
                "bootId": "11111111-1111-4111-8111-111111111111",
            }
            fixture.harness.controller_identity = lambda: controller
            try:
                fixture.harness.preflight(fixture.run_id)
                original_archive = fixture.harness._write_archive

                def guarded_archive(run_id: str, path: Path) -> str:
                    state = fixture.harness._load_state(run_id)
                    self.assertEqual(state["transactionGuard"]["lifecycle"], "ARMED")
                    self.assertTrue(
                        fixture.runner.deadmen[
                            fixture.harness._transaction_deadman_timer(run_id)
                        ]
                    )
                    return original_archive(run_id, path)

                fixture.harness._write_archive = guarded_archive  # type: ignore[method-assign]
                fixture.harness.backup(fixture.run_id)
                payloads = fixture.payloads()
                original_apply = fixture.harness._apply_files

                def guarded_apply(run_id: str, transaction: Mapping[str, Any]) -> None:
                    self.assertTrue(
                        all(
                            unit["active"] is False
                            for unit in fixture.runner.units.values()
                        )
                    )
                    self.assertTrue(
                        fixture.harness._load_state(run_id)["transactionGuard"][
                            "armed"
                        ]
                    )
                    original_apply(run_id, transaction)

                fixture.harness._apply_files = guarded_apply  # type: ignore[method-assign]
                fixture.enable(payloads)
                state = fixture.harness._load_state(fixture.run_id)
                guard = state["transactionGuard"]
                self.assertEqual(state["trustTransaction"]["phase"], "APPLIED")
                self.assertTrue(guard["armed"])
                transaction_call = next(
                    call
                    for call in fixture.runner.calls
                    if call
                    and call[0] == "/usr/bin/systemd-run"
                    and any(
                        item.startswith("--unit=nuvion-fleet-transaction-")
                        for item in call
                    )
                )
                self.assertIn(
                    "--property=TimeoutStartSec="
                    f"{BOARD.TRANSACTION_DEADMAN_RECOVERY_SECONDS}s",
                    transaction_call,
                )
                self.assertIn(
                    "--property=StartLimitIntervalSec=0", transaction_call
                )
                self.assertFalse(
                    any(
                        item.startswith("--property=StartLimitBurst=")
                        for item in transaction_call
                    )
                )
                recovered = fixture.harness.cleanup(
                    fixture.run_id,
                    transaction_deadman_only=True,
                    transaction_deadman_epoch=guard["writerEpoch"],
                    transaction_controller=controller,
                )
                self.assertTrue(recovered["complete"])
                final = fixture.harness._load_state(fixture.run_id)
                self.assertEqual(final["trustTransaction"]["phase"], "RESTORED")
                self.assertFalse(final["transactionGuard"]["armed"])
                self.assertFalse(fixture.paths.active_run.exists())
                self.assertFalse(fixture.harness._transaction_dir(fixture.run_id).exists())
            finally:
                fixture.close()

    def test_transaction_guard_recovers_backup_gap_and_orphan_timer_bind(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            controller = {
                "pid": 999998,
                "startTicks": 515151,
                "bootId": "22222222-2222-4222-8222-222222222222",
            }
            fixture.harness.controller_identity = lambda: controller
            try:
                fixture.harness.preflight(fixture.run_id)
                fixture.harness.backup(fixture.run_id)
                state = fixture.harness._load_state(fixture.run_id)
                guard = state["transactionGuard"]
                state["backup"]["phase"] = "APPLYING"
                state["backup"]["complete"] = False
                fixture.harness._save_state(fixture.run_id, state)
                result = fixture.harness.cleanup(
                    fixture.run_id,
                    transaction_deadman_only=True,
                    transaction_deadman_epoch=guard["writerEpoch"],
                    transaction_controller=controller,
                )
                self.assertTrue(result["complete"])
                final = fixture.harness._load_state(fixture.run_id)
                self.assertTrue(final["backup"]["recoveryAbandoned"])
                self.assertFalse(
                    (fixture.paths.recovery_root / f"iq9075-{fixture.run_id}.tar").exists()
                )

                second = str(uuid.uuid4())
                fixture.harness.preflight(second)
                epoch = str(uuid.uuid4())
                fixture.harness._start_transaction_deadman(
                    second, epoch=epoch, controller=controller
                )
                orphan = fixture.harness.cleanup(
                    second,
                    transaction_deadman_only=True,
                    transaction_deadman_epoch=epoch,
                    transaction_controller=controller,
                )
                self.assertTrue(orphan["complete"])
                self.assertFalse(fixture.paths.active_run.exists())
            finally:
                fixture.close()

    def test_transaction_guard_rejects_stale_epoch_without_restoring(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            controller = {
                "pid": 999997,
                "startTicks": 616161,
                "bootId": "33333333-3333-4333-8333-333333333333",
            }
            fixture.harness.controller_identity = lambda: controller
            try:
                fixture.provision()
                with self.assertRaisesRegex(
                    BOARD.HarnessError, "invocation/journal mismatch"
                ):
                    fixture.harness.cleanup(
                        fixture.run_id,
                        transaction_deadman_only=True,
                        transaction_deadman_epoch=str(uuid.uuid4()),
                        transaction_controller=controller,
                    )
                state = fixture.harness._load_state(fixture.run_id)
                self.assertEqual(state["trustTransaction"]["phase"], "APPLIED")
                self.assertTrue(state["transactionGuard"]["armed"])
            finally:
                fixture.close()

    def test_boot_reconciles_lost_timer_without_starting_protected_units(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                fixture.provision()
                timer = fixture.harness._transaction_deadman_timer(fixture.run_id)
                fixture.runner.deadmen[timer] = False
                for unit in fixture.runner.units.values():
                    unit["active"] = False
                    unit["pid"] = 0
                result = fixture.harness.boot_reconcile()
                self.assertTrue(result["complete"])
                state = fixture.harness._load_state(fixture.run_id)
                self.assertEqual(state["trustTransaction"]["phase"], "RESTORED")
                self.assertFalse(state["transactionGuard"]["armed"])
                self.assertFalse(fixture.paths.active_run.exists())
                self.assertTrue(
                    all(
                        unit["active"] is False
                        for unit in fixture.runner.units.values()
                    )
                )
            finally:
                fixture.close()

    def test_boot_candidate_recovery_does_not_cancel_queued_runtime_starts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                fixture.harness.preflight(fixture.run_id)
                slots = fixture.harness._slot_snapshot()
                units = fixture.harness._unit_snapshot()
                state = fixture.harness._load_state(fixture.run_id)
                state["candidateSoak"] = {
                    "phase": "QUIESCING",
                    "candidateSlot": str(
                        fixture.paths.candidate_root
                        / f"{fixture.run_id}-{'a' * 64}"
                    ),
                    "baselineSlots": slots,
                    "releaseTreesBefore": fixture.harness._release_tree_snapshot(
                        slots
                    ),
                    "unitsBefore": units,
                    "baselineRuntime": {
                        "pid": 100,
                        "startTicks": 1000,
                        "bootId": "11111111-1111-4111-8111-111111111111",
                        "activeSlot": slots["current"],
                    },
                    "oakBefore": fixture.harness.verify_oak(),
                    "inputDigests": {},
                }
                fixture.harness._save_state(fixture.run_id, state)
                for unit in fixture.runner.units.values():
                    unit["active"] = False
                    unit["pid"] = 0
                fixture.runner.calls.clear()

                result = fixture.harness.boot_reconcile()
                self.assertTrue(result["complete"])
                protected_stops = [
                    call
                    for call in fixture.runner.calls
                    if call[:2] == ("/usr/bin/systemctl", "stop")
                    and len(call) > 2
                    and call[2] in BOARD.UNITS
                ]
                self.assertEqual(protected_stops, [])
                self.assertTrue(
                    all(
                        unit["active"] is False
                        for unit in fixture.runner.units.values()
                    )
                )
            finally:
                fixture.close()

    def test_normal_cleanup_resumes_runtime_after_completed_boot_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                fixture.provision()
                for unit in fixture.runner.units.values():
                    unit["active"] = False
                    unit["pid"] = 0
                boot = fixture.harness.boot_reconcile()
                self.assertTrue(boot["complete"])
                boot_state = fixture.harness._read_existing_run_state(
                    fixture.run_id
                )
                self.assertFalse(
                    boot_state["trustTransaction"]["runtimeRestored"]
                )

                normal = fixture.harness.cleanup(fixture.run_id)
                self.assertTrue(normal["complete"])
                self.assertTrue(normal["idempotent"])
                final = fixture.harness._read_existing_run_state(fixture.run_id)
                self.assertTrue(final["trustTransaction"]["runtimeRestored"])
                self.assertTrue(fixture.runner.units["nuv-agent.service"]["active"])
            finally:
                fixture.close()

    def test_candidate_requires_monotonic_transaction_deadline_margin(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                fixture.harness.preflight(fixture.run_id)
                fixture.harness.backup(fixture.run_id)
                state = fixture.harness._load_state(fixture.run_id)
                original_run = fixture.runner.run

                def near_deadline(argv, *, timeout, input_bytes=None):
                    call = tuple(argv)
                    if (
                        call[:4]
                        == (
                            "/usr/bin/busctl",
                            "--system",
                            "--json=short",
                            "get-property",
                        )
                        and call[-1] == "NextElapseUSecMonotonic"
                    ):
                        required = (
                            BOARD.CANDIDATE_DEADMAN_SECONDS
                            + BOARD.CANDIDATE_DEADMAN_RECOVERY_SECONDS
                            + 120
                        )
                        next_usec = int(
                            (fixture.clock.value + required - 1) * 1_000_000
                        )
                        return BOARD.CommandResult(
                            0,
                            json.dumps(
                                {"type": "t", "data": next_usec},
                                separators=(",", ":"),
                            )
                            + "\n",
                            "",
                        )
                    return original_run(
                        argv, timeout=timeout, input_bytes=input_bytes
                    )

                fixture.runner.run = near_deadline
                # Wall-clock movement must not affect the systemd monotonic
                # deadline decision.
                fixture.harness.clock = lambda: "2099-01-01T00:00:00Z"
                with self.assertRaisesRegex(BOARD.HarnessError, "deadline is too near"):
                    fixture.harness._require_transaction_guard(
                        fixture.run_id, state
                    )
            finally:
                fixture.close()

    def test_transaction_deadline_requires_typed_busctl_uint64(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                fixture.harness.preflight(fixture.run_id)
                fixture.harness.backup(fixture.run_id)
                state = fixture.harness._load_state(fixture.run_id)
                original_run = fixture.runner.run

                def human_readable_deadline(argv, *, timeout, input_bytes=None):
                    call = tuple(argv)
                    if (
                        call[:4]
                        == (
                            "/usr/bin/busctl",
                            "--system",
                            "--json=short",
                            "get-property",
                        )
                        and call[-1] == "NextElapseUSecMonotonic"
                    ):
                        return BOARD.CommandResult(
                            0,
                            '{"type":"t","data":"1d 15min 7s"}\n',
                            "",
                        )
                    return original_run(
                        argv, timeout=timeout, input_bytes=input_bytes
                    )

                fixture.runner.run = human_readable_deadline
                with self.assertRaisesRegex(
                    BOARD.HarnessError,
                    "timer deadline is unavailable",
                ):
                    fixture.harness._require_transaction_guard(
                        fixture.run_id, state
                    )
                self.assertTrue(
                    any(
                        call[:4]
                        == (
                            "/usr/bin/busctl",
                            "--system",
                            "--json=short",
                            "call",
                        )
                        and call[-2:] == ("s", fixture.harness._transaction_deadman_timer(fixture.run_id))
                        for call in fixture.runner.calls
                    )
                )
            finally:
                fixture.close()

    def test_transaction_callback_stops_writers_before_corrupt_journal_parse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            controller = {
                "pid": 999996,
                "startTicks": 717171,
                "bootId": "44444444-4444-4444-8444-444444444444",
            }
            fixture.harness.controller_identity = lambda: controller
            try:
                fixture.provision()
                state = fixture.harness._load_state(fixture.run_id)
                guard = state["transactionGuard"]
                fixture.harness._state_path(fixture.run_id).write_text(
                    "{truncated\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(BOARD.HarnessError, "strict UTF-8 JSON"):
                    fixture.harness.cleanup(
                        fixture.run_id,
                        transaction_deadman_only=True,
                        transaction_deadman_epoch=guard["writerEpoch"],
                        transaction_controller=controller,
                    )
                self.assertTrue(
                    all(
                        unit["active"] is False
                        for unit in fixture.runner.units.values()
                    )
                )
            finally:
                fixture.close()

    def test_boot_fails_closed_on_corrupt_lease_after_stopping_writers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                fixture.provision()
                fixture.paths.active_run.write_text("{broken\n", encoding="utf-8")
                with self.assertRaisesRegex(BOARD.HarnessError, "strict UTF-8 JSON"):
                    fixture.harness.boot_reconcile()
                self.assertTrue(
                    all(
                        unit["active"] is False
                        for unit in fixture.runner.units.values()
                    )
                )
            finally:
                fixture.close()

    def test_boot_fails_closed_on_interrupted_package_maintenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                fixture.paths.package_maintenance.parent.mkdir(
                    parents=True, exist_ok=True
                )
                fixture.paths.package_maintenance.write_text(
                    json.dumps(
                        {
                            "schemaVersion": 1,
                            "kind": "nuvion-package-maintenance",
                            "active": True,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n",
                    encoding="utf-8",
                )
                fixture.paths.package_maintenance.chmod(0o600)
                with self.assertRaisesRegex(
                    BOARD.HarnessError, "package maintenance blocks runtime"
                ):
                    fixture.harness.boot_reconcile()
                self.assertTrue(
                    all(
                        unit["active"] is False
                        for unit in fixture.runner.units.values()
                    )
                )
            finally:
                fixture.close()

    def test_claim_rejects_existing_missing_or_malformed_recovery_journal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                fixture.paths.state_root.mkdir(parents=True, mode=0o700)
                orphan_id = str(uuid.uuid4())
                orphan = fixture.paths.state_root / orphan_id
                orphan.mkdir(mode=0o700)
                (orphan / "trust-transaction").mkdir(mode=0o700)
                with self.assertRaisesRegex(BOARD.HarnessError, "run.json"):
                    fixture.harness.preflight(orphan_id)
                self.assertFalse(fixture.paths.active_run.exists())
                self.assertFalse((orphan / "run.json").exists())
            finally:
                fixture.close()

        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                fixture.provision()
                fixture.harness.cleanup(fixture.run_id)
                state = fixture.harness._load_state(fixture.run_id)
                state["transactionGuard"]["deadlineSeconds"] = 1
                fixture.harness._save_state(fixture.run_id, state)
                with self.assertRaisesRegex(
                    BOARD.HarnessError, "guard journal is invalid"
                ):
                    fixture.harness.preflight(str(uuid.uuid4()))
                self.assertFalse(fixture.paths.active_run.exists())
            finally:
                fixture.close()

    def test_rejected_claim_leaves_retryable_minimal_run_journal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                fixture.provision()
                original_disarm = fixture.harness._disarm_transaction_guard

                def crash_before_disarm(*_args, **_kwargs) -> None:
                    raise OSError("simulated disarm crash")

                fixture.harness._disarm_transaction_guard = crash_before_disarm  # type: ignore[method-assign]
                with self.assertRaisesRegex(OSError, "disarm crash"):
                    fixture.harness.cleanup(fixture.run_id)
                new_run_id = str(uuid.uuid4())
                with self.assertRaisesRegex(
                    BOARD.HarnessError,
                    "unfinished Fleet recovery state",
                ):
                    fixture.harness.preflight(new_run_id)
                minimal = fixture.harness._read_existing_run_state(new_run_id)
                self.assertEqual(
                    set(minimal),
                    {"schemaVersion", "protocolVersion", "runId", "createdAt"},
                )

                fixture.harness._disarm_transaction_guard = original_disarm  # type: ignore[method-assign]
                fixture.harness.boot_reconcile()
                result = fixture.harness.preflight(new_run_id)
                self.assertEqual(result["runId"], new_run_id)
                self.assertTrue((fixture.paths.state_root / new_run_id / "run.json").is_file())
            finally:
                fixture.close()

    def test_package_marker_is_rechecked_after_global_lock_acquisition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            original_lock = fixture.harness._lock_file

            @contextmanager
            def racing_lock(path: Path):
                with original_lock(path):
                    if path == fixture.paths.global_fleet_lock:
                        fixture.paths.package_maintenance.parent.mkdir(
                            parents=True, exist_ok=True
                        )
                        fixture.paths.package_maintenance.write_text(
                            json.dumps(
                                {
                                    "schemaVersion": 1,
                                    "kind": "nuvion-package-maintenance",
                                    "active": True,
                                },
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                            + "\n",
                            encoding="utf-8",
                        )
                        fixture.paths.package_maintenance.chmod(0o600)
                    yield

            fixture.harness._lock_file = racing_lock  # type: ignore[method-assign]
            try:
                run_id = str(uuid.uuid4())
                with self.assertRaisesRegex(
                    BOARD.HarnessError,
                    "package maintenance blocks Fleet E2E operations",
                ):
                    fixture.harness.preflight(run_id)
                self.assertFalse(fixture.paths.active_run.exists())
                self.assertFalse((fixture.paths.state_root / run_id).exists())
            finally:
                fixture.close()

    def test_claim_journal_precedes_lease_and_lease_precedes_operation_lock(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            original_claim = fixture.harness._claim_active_run

            def crash_before_lease(run_id: str) -> None:
                minimal = fixture.harness._read_existing_run_state(run_id)
                self.assertEqual(
                    set(minimal),
                    {"schemaVersion", "protocolVersion", "runId", "createdAt"},
                )
                self.assertFalse(fixture.paths.active_run.exists())
                raise OSError("simulated crash before lease")

            fixture.harness._claim_active_run = crash_before_lease  # type: ignore[method-assign]
            try:
                run_id = str(uuid.uuid4())
                with self.assertRaisesRegex(OSError, "before lease"):
                    fixture.harness.preflight(run_id)
                fixture.harness._claim_active_run = original_claim  # type: ignore[method-assign]
                self.assertEqual(
                    fixture.harness.preflight(run_id)["runId"], run_id
                )
            finally:
                fixture.close()

        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            original_lock = fixture.harness._lock_file

            @contextmanager
            def crash_before_operation_lock(path: Path):
                if path.name == "operation.lock":
                    self.assertTrue(fixture.paths.active_run.is_file())
                    self.assertTrue(
                        fixture.harness._state_path(fixture.run_id).is_file()
                    )
                    raise OSError("simulated crash before operation lock")
                with original_lock(path):
                    yield

            fixture.harness._lock_file = crash_before_operation_lock  # type: ignore[method-assign]
            try:
                with self.assertRaisesRegex(OSError, "before operation lock"):
                    fixture.harness.preflight(fixture.run_id)
                fixture.harness._lock_file = original_lock  # type: ignore[method-assign]
                recovered = fixture.harness.boot_reconcile()
                self.assertTrue(recovered["complete"])
                self.assertFalse(fixture.paths.active_run.exists())
                self.assertEqual(
                    fixture.harness.preflight(fixture.run_id)["runId"],
                    fixture.run_id,
                )
            finally:
                fixture.close()

    def test_transaction_callback_fences_latest_bound_writer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            creator = {
                "pid": 999995,
                "startTicks": 818181,
                "bootId": "55555555-5555-4555-8555-555555555555",
            }
            latest = {
                "pid": 999994,
                "startTicks": 919191,
                "bootId": "66666666-6666-4666-8666-666666666666",
            }
            fixture.harness.controller_identity = lambda: creator
            try:
                fixture.provision()
                state = fixture.harness._load_state(fixture.run_id)
                guard = state["transactionGuard"]
                guard["activeWriter"] = {
                    "writerEpoch": str(uuid.uuid4()),
                    "controller": latest,
                    "boundAt": "2026-09-03T00:00:00Z",
                }
                state["transactionGuard"] = guard
                fixture.harness._save_state(fixture.run_id, state)
                fenced: list[Mapping[str, object]] = []
                original_assert = fixture.harness._assert_active_run
                checked_second_stop = False

                def fence_and_race(identity: Mapping[str, object]) -> None:
                    fenced.append(dict(identity))
                    for unit in fixture.runner.units.values():
                        unit["active"] = True

                def assert_stopped_after_fence(
                    run_id: str, *, allow_unclaimed: bool
                ) -> None:
                    nonlocal checked_second_stop
                    if not checked_second_stop:
                        self.assertTrue(
                            all(
                                unit["active"] is False
                                for unit in fixture.runner.units.values()
                            )
                        )
                        checked_second_stop = True
                    original_assert(run_id, allow_unclaimed=allow_unclaimed)

                fixture.harness._fence_candidate_controller = fence_and_race  # type: ignore[method-assign]
                fixture.harness._assert_active_run = assert_stopped_after_fence  # type: ignore[method-assign]
                result = fixture.harness.cleanup(
                    fixture.run_id,
                    transaction_deadman_only=True,
                    transaction_deadman_epoch=guard["writerEpoch"],
                    transaction_controller=creator,
                )
                self.assertTrue(result["complete"])
                self.assertTrue(checked_second_stop)
                self.assertEqual(fenced, [latest])
            finally:
                fixture.close()

    def test_transaction_callback_second_stop_covers_lost_lease_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            creator = {
                "pid": 999993,
                "startTicks": 717171,
                "bootId": "77777777-7777-4777-8777-777777777777",
            }
            fixture.harness.controller_identity = lambda: creator
            try:
                fixture.provision()
                state = fixture.harness._read_existing_run_state(fixture.run_id)
                guard = state["transactionGuard"]
                fixture.harness._release_active_run(fixture.run_id)

                def fence_and_race(_identity: Mapping[str, object]) -> None:
                    for unit in fixture.runner.units.values():
                        unit["active"] = True

                fixture.harness._fence_candidate_controller = fence_and_race  # type: ignore[method-assign]
                with self.assertRaisesRegex(
                    BOARD.HarnessError,
                    "lost its active-run lease",
                ):
                    fixture.harness.cleanup(
                        fixture.run_id,
                        transaction_deadman_only=True,
                        transaction_deadman_epoch=guard["writerEpoch"],
                        transaction_controller=creator,
                    )
                self.assertTrue(
                    all(
                        unit["active"] is False
                        for unit in fixture.runner.units.values()
                    )
                )
            finally:
                fixture.close()

    def test_boot_finalizes_crash_after_lease_release_before_guard_disarm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                fixture.provision()
                original_disarm = fixture.harness._disarm_transaction_guard

                def crash_before_disarm(*_args, **_kwargs) -> None:
                    raise OSError("simulated final disarm crash")

                fixture.harness._disarm_transaction_guard = crash_before_disarm  # type: ignore[method-assign]
                with self.assertRaisesRegex(OSError, "final disarm crash"):
                    fixture.harness.cleanup(fixture.run_id)
                self.assertFalse(fixture.paths.active_run.exists())
                state = fixture.harness._load_state(fixture.run_id)
                self.assertTrue(state["cleanup"]["complete"])
                self.assertTrue(state["transactionGuard"]["armed"])
                fixture.harness._disarm_transaction_guard = original_disarm  # type: ignore[method-assign]
                result = fixture.harness.boot_reconcile()
                self.assertTrue(result["complete"])
                self.assertFalse(
                    fixture.harness._load_state(fixture.run_id)["transactionGuard"][
                        "armed"
                    ]
                )
            finally:
                fixture.close()

    def test_boot_finalizes_candidate_and_transaction_guards_after_lost_lease(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                fixture.provision()
                state = fixture.harness._load_state(fixture.run_id)
                transaction = state["trustTransaction"]
                fixture.harness._restore_transaction(
                    fixture.run_id, state, transaction
                )
                state = fixture.harness._load_state(fixture.run_id)
                candidate_unit = fixture.harness._candidate_deadman_unit(
                    fixture.run_id
                )
                state["candidateSoak"] = {
                    "phase": "RESTORED",
                    "productionRestoration": {},
                    "deadman": {
                        "unit": candidate_unit,
                        "armed": True,
                        "lifecycle": "ARMED",
                        "writerEpoch": str(uuid.uuid4()),
                        "controller": fixture.harness.controller_identity(),
                    },
                }
                fixture.harness._purge_run_sensitive_material(fixture.run_id)
                state["cleanup"] = {
                    "complete": True,
                    "completedAt": "2026-09-03T00:00:00Z",
                }
                fixture.harness._save_state(fixture.run_id, state)
                fixture.harness._release_active_run(fixture.run_id)
                fixture.runner.deadmen[
                    fixture.harness._candidate_deadman_timer(fixture.run_id)
                ] = False

                result = fixture.harness.boot_reconcile()
                self.assertTrue(result["complete"])
                final = fixture.harness._read_existing_run_state(fixture.run_id)
                self.assertFalse(final["candidateSoak"]["deadman"]["armed"])
                self.assertTrue(final["candidateSoak"]["deadman"]["stopped"])
                self.assertFalse(final["transactionGuard"]["armed"])
                next_run = str(uuid.uuid4())
                self.assertEqual(
                    fixture.harness.preflight(next_run)["runId"], next_run
                )
            finally:
                fixture.close()

    def test_writer_stop_attempts_all_units_before_aggregate_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            original_run = fixture.runner.run
            first = BOARD.STOP_ORDER[0]

            def fail_first_stop(argv, *, timeout, input_bytes=None):
                call = tuple(argv)
                if call == ("/usr/bin/systemctl", "stop", first):
                    fixture.runner.calls.append(call)
                    return BOARD.CommandResult(1, "", "failed")
                return original_run(argv, timeout=timeout, input_bytes=input_bytes)

            fixture.runner.run = fail_first_stop
            try:
                for unit in fixture.runner.units.values():
                    unit["active"] = True
                with self.assertRaisesRegex(
                    BOARD.HarnessError, "protected writers did not stop"
                ):
                    fixture.harness._stop_writers()
                stopped = {
                    call[-1]
                    for call in fixture.runner.calls
                    if call[:2] == ("/usr/bin/systemctl", "stop")
                }
                self.assertEqual(stopped, set(BOARD.STOP_ORDER))
                self.assertTrue(fixture.runner.units[first]["active"])
                self.assertTrue(
                    all(
                        not fixture.runner.units[unit]["active"]
                        for unit in BOARD.STOP_ORDER[1:]
                    )
                )
            finally:
                fixture.close()

        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            original_run = fixture.runner.run
            first = BOARD.STOP_ORDER[0]

            def fail_first_boot_stop(argv, *, timeout, input_bytes=None):
                call = tuple(argv)
                if call == ("/usr/bin/systemctl", "stop", first):
                    fixture.runner.calls.append(call)
                    return BOARD.CommandResult(1, "", "failed")
                return original_run(argv, timeout=timeout, input_bytes=input_bytes)

            fixture.runner.run = fail_first_boot_stop
            try:
                for unit in fixture.runner.units.values():
                    unit["active"] = True
                with self.assertRaisesRegex(
                    BOARD.HarnessError, "did not quiesce at the boot gate"
                ):
                    fixture.harness._stop_writers_for_boot()
                stopped = {
                    call[-1]
                    for call in fixture.runner.calls
                    if call[:2] == ("/usr/bin/systemctl", "stop")
                }
                self.assertEqual(stopped, set(BOARD.STOP_ORDER))
                self.assertTrue(fixture.runner.units[first]["active"])
                self.assertTrue(
                    all(
                        not fixture.runner.units[unit]["active"]
                        for unit in BOARD.STOP_ORDER[1:]
                    )
                )
            finally:
                fixture.close()

    def test_cleanup_resumes_runtime_after_durable_trust_restore_crash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                fixture.provision()
                original_restore_units = fixture.harness._restore_units

                def crash_before_runtime(_expected: Mapping[str, object]) -> None:
                    state = fixture.harness._load_state(fixture.run_id)
                    self.assertEqual(state["trustTransaction"]["phase"], "RESTORED")
                    self.assertFalse(state["trustTransaction"]["runtimeRestored"])
                    raise BOARD.HarnessError("simulated runtime restore crash")

                fixture.harness._restore_units = crash_before_runtime  # type: ignore[method-assign]
                interrupted = fixture.harness.cleanup(fixture.run_id)
                self.assertFalse(interrupted["complete"])
                self.assertTrue(fixture.paths.active_run.exists())
                fixture.harness._restore_units = original_restore_units  # type: ignore[method-assign]
                resumed = fixture.harness.cleanup(fixture.run_id)
                self.assertTrue(resumed["complete"])
                final = fixture.harness._load_state(fixture.run_id)
                self.assertTrue(final["trustTransaction"]["runtimeRestored"])
                self.assertFalse(fixture.paths.active_run.exists())
            finally:
                fixture.close()

    def test_core_limit_is_fail_closed_for_board_host_and_askpass(self) -> None:
        with mock.patch.object(BOARD.resource, "setrlimit") as board_limit:
            BOARD.disable_core_dumps()
        board_limit.assert_called_once_with(BOARD.resource.RLIMIT_CORE, (0, 0))
        with mock.patch.object(HOST.resource, "setrlimit") as host_limit:
            HOST.disable_core_dumps()
        host_limit.assert_called_once_with(HOST.resource.RLIMIT_CORE, (0, 0))
        with mock.patch.dict(
            os.environ,
            {"NUVION_E2E_ASKPASS_SOCKET": "/tmp/never-contact-askpass.sock"},
            clear=False,
        ), mock.patch.object(
            HOST, "disable_core_dumps", side_effect=HOST.RunnerError("denied")
        ), mock.patch.object(HOST.socket, "socket") as socket_factory:
            with self.assertRaisesRegex(HOST.RunnerError, "denied"):
                HOST._askpass_entrypoint()
            socket_factory.assert_not_called()


class FakeLocalRunner:
    def __init__(self, payload: Mapping[str, object] | None = None) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.payload = dict(payload or {"schemaVersion": 1})

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        input_bytes: bytes | None = None,
        env: Mapping[str, str] | None = None,
    ):
        del timeout, input_bytes, env
        self.calls.append(tuple(argv))
        return HOST.ProcessResult(0, (json.dumps(self.payload) + "\n").encode(), b"")


class FakeTransport:
    def __init__(self) -> None:
        self.calls = 0

    def invoke_board(
        self, command: str, arguments: Sequence[str] = (), *, timeout: float = 90
    ) -> dict[str, Any]:
        del command, arguments, timeout
        self.calls += 1
        return {"schemaVersion": 1}


class FakeBootstrapTransport:
    def __init__(self, *, local_tool_sha: str, current_slot: str) -> None:
        self.local_tool_sha = local_tool_sha
        self.current_slot = current_slot
        self.cleanup_calls = 0
        self.bootstrap_arguments: dict[str, object] | None = None

    def copy_bootstrap_artifact(self, source: Path, *, run_id: str, role: str) -> str:
        del source
        if role == "installer":
            return HOST.REMOTE_BOOTSTRAP_INSTALLER.format(run_id=run_id)
        return HOST.REMOTE_BOOTSTRAP_DEB.format(run_id=run_id)

    def bootstrap_updater(self, **arguments: object) -> dict[str, object]:
        self.bootstrap_arguments = dict(arguments)
        return {
            "schemaVersion": 1,
            "protocolVersion": HOST.PROTOCOL_VERSION,
            "runId": arguments["run_id"],
            "outOfBandBootstrap": True,
            "otaEvidence": False,
            "previousPackageVersion": "0.1.115",
            "installedPackageVersion": arguments["expected_version"],
            "packageSha256": arguments["package_sha256"],
            "installerSha256": arguments["installer_sha256"],
            "updaterCodeVersion": "0.2.0",
            "boardToolSha256": self.local_tool_sha,
            "currentSlot": self.current_slot,
            "servicesInactive": True,
        }

    def discard_bootstrap_staging(self, *, run_id: str) -> dict[str, object]:
        self.cleanup_calls += 1
        return {"schemaVersion": 1, "runId": run_id, "complete": True}

    def invoke_board(
        self, command: str, arguments: Sequence[str] = (), *, timeout: float = 90
    ) -> dict[str, object]:
        del arguments, timeout
        if command == "identity":
            return {
                "schemaVersion": 1,
                "protocolVersion": HOST.PROTOCOL_VERSION,
                "toolPath": HOST.REMOTE_TOOL,
                "toolSha256": self.local_tool_sha,
                "rootOwned": True,
                "mode": "0755",
            }
        if command == "preflight":
            return {
                "schemaVersion": 1,
                "verified": True,
                "foundation": {
                    "slots": {
                        "current": self.current_slot,
                        "currentVersion": "0.1.121",
                    }
                },
            }
        raise AssertionError(command)


class PollingRunTransport:
    def __init__(self, *, tool_sha: str, baseline_digest: str) -> None:
        self.tool_sha = tool_sha
        self.baseline_digest = baseline_digest
        self.manifest: dict[str, Any] | None = None
        self.evidence_calls = 0
        self.transaction_present = False

    def copy_input(self, source: Path, *, run_id: str, role: str) -> str:
        if role == "manifest":
            self.manifest = json.loads(source.read_text(encoding="utf-8"))
        return f"/tmp/nuvion-fleet-e2e-{run_id}-{role}.json"

    def invoke_board(
        self, command: str, arguments: Sequence[str] = (), *, timeout: float = 90
    ) -> dict[str, Any]:
        del arguments, timeout
        if command == "identity":
            return {
                "schemaVersion": 1,
                "protocolVersion": HOST.PROTOCOL_VERSION,
                "toolPath": HOST.REMOTE_TOOL,
                "toolSha256": self.tool_sha,
                "rootOwned": True,
                "mode": "0755",
            }
        if command == "preflight":
            current = (
                "releases/" + self.manifest["scenario"]["expectedBomDigest"][7:]
                if self.transaction_present and self.manifest is not None
                else f"releases/{self.baseline_digest}"
            )
            return {
                "schemaVersion": 1,
                "verified": True,
                "foundation": {
                    "slots": {
                        "current": current,
                        "currentVersion": "0.1.121"
                        if self.transaction_present
                        else "0.1.120",
                    }
                },
                "recordedBaseline": {
                    "slot": f"releases/{self.baseline_digest}",
                    "version": "0.1.120",
                },
                "transactionPresent": self.transaction_present,
            }
        if command == "enable-fleet":
            self.transaction_present = True
            return {"schemaVersion": 1, "complete": True}
        if command in {"backup", "discard-staging"}:
            return {"schemaVersion": 1, "complete": True}
        if command == "evidence":
            self.evidence_calls += 1
            if self.evidence_calls == 1:
                return {
                    "schemaVersion": 1,
                    "protocolVersion": HOST.PROTOCOL_VERSION,
                    "runId": self.manifest["runId"],
                    "scenario": "commit",
                    "complete": False,
                    "gates": {
                        "foundation": True,
                        "backup": True,
                        "trust": True,
                        "updater2": True,
                        "oak": True,
                        "services": True,
                        "scenario": False,
                    },
                    "updater": {"updaterVersion": "0.2.0", "update": None},
                }
            assert self.manifest is not None
            scenario = self.manifest["scenario"]
            release = scenario["release"]
            candidate = "releases/" + scenario["expectedBomDigest"][7:]
            return {
                "schemaVersion": 1,
                "protocolVersion": HOST.PROTOCOL_VERSION,
                "runId": self.manifest["runId"],
                "generatedAt": f"2026-09-02T00:00:{self.evidence_calls:02d}Z",
                "scenario": "commit",
                "complete": True,
                "gates": {
                    "foundation": True,
                    "backup": True,
                    "trust": True,
                    "updater2": True,
                    "oak": True,
                    "services": True,
                    "scenario": True,
                },
                "oak": {
                    "port": "2-1.1",
                    "vendorId": "03e7",
                    "productId": "f63b",
                    "speedMbps": 5000,
                    "mxidSha256": "a" * 64,
                    "attached": True,
                    "bound": True,
                },
                "services": {
                    "nuv-agent.service": {
                        "active": True,
                        "enabled": True,
                        "unitFileState": "enabled",
                        "mainPid": 101,
                    },
                    "nuv-agent-updater.service": {
                        "active": True,
                        "enabled": True,
                        "unitFileState": "enabled",
                        "mainPid": 102,
                    },
                    "nuv-agent-updater.socket": {
                        "active": True,
                        "enabled": True,
                        "unitFileState": "enabled",
                        "mainPid": 0,
                    },
                },
                "runtimePids": None,
                "slots": {
                    "current": candidate,
                    "previous": scenario["expectedPreviousSlot"],
                    "currentVersion": release["agentVersion"],
                    "release": {
                        "schemaVersion": 2,
                        "bomDigest": scenario["expectedBomDigest"],
                        **release,
                    },
                    "previousRelease": {
                        "schemaVersion": 2,
                        "bomDigest": "sha256:" + self.baseline_digest,
                        "agentVersion": "0.1.120",
                        "releaseSequence": 1,
                        "artifactDigest": "sha256:" + "e" * 64,
                        "componentSha": "f" * 40,
                        "configSchema": "12",
                        "publisherKeyId": "release-iq9075-dev",
                    },
                },
                "updater": {
                    "capabilityAvailable": True,
                    "authenticatedHelper": True,
                    "reason": "READY",
                    "updaterVersion": "0.2.0",
                    "update": {
                        "commandId": scenario["expectedCommandId"],
                        "sequence": 2,
                        "targetVersion": release["agentVersion"],
                        "bomDigest": scenario["expectedBomDigest"],
                        "phase": "COMMITTED",
                        "updatePhase": "COMMITTED",
                        "updatedAt": "2026-09-02T00:00:02Z",
                        "commandExpiresAt": "2026-09-02T01:00:00Z",
                        "candidateSlot": scenario["expectedCandidateSlot"],
                        "previousSlot": scenario["expectedPreviousSlot"],
                        "previousVersion": scenario["expectedPreviousVersion"],
                        "releaseSequence": release["releaseSequence"],
                        "artifactDigest": release["artifactDigest"],
                        "componentSha": release["componentSha"],
                        "configSchema": release["configSchema"],
                        "bomVerificationStatus": "VERIFIED",
                        "publisherKeyId": release["publisherKeyId"],
                        "slot": candidate,
                        "health": "FUNCTIONAL_HEALTHY",
                        "functionalHealth": "FUNCTIONAL_HEALTHY",
                    },
                },
            }
        raise AssertionError(command)


class Iq9075FleetHostHarnessTest(unittest.TestCase):
    def test_candidate_cli_returns_nonzero_but_prints_failed_evidence(self) -> None:
        run_id = str(uuid.uuid4())
        arguments = SimpleNamespace(
            run_id=run_id,
            output_dir=f"/tmp/{run_id}",
            ssh_password_fd=None,
            sudo_password_fd=None,
            known_hosts="/tmp/known_hosts",
            host="iq9075",
            user="plaid",
            port=22,
            host_key_sha256=HOST.DEFAULT_FINGERPRINT,
            command="candidate-soak",
            local_board_tool="/tmp/board.py",
            local_oak_harness="/tmp/oak.sh",
            candidate_bundle="/tmp/candidate.tar.gz",
            candidate_bom="/tmp/candidate-bom.json",
        )
        parser = mock.Mock()
        parser.parse_args.return_value = arguments
        fleet = mock.Mock()
        failed = {
            "schemaVersion": 1,
            "runId": run_id,
            "complete": True,
            "passed": False,
            "candidateSoakEvidenceSha256": "a" * 64,
            "rawEvidenceSha256": "b" * 64,
        }
        fleet.candidate_soak.return_value = failed
        output = io.StringIO()
        with (
            mock.patch.object(HOST, "disable_core_dumps"),
            mock.patch.object(HOST, "build_parser", return_value=parser),
            mock.patch.object(
                HOST,
                "prepare_output_dir",
                return_value=Path(arguments.output_dir),
            ),
            mock.patch.object(
                HOST,
                "create_pinned_known_hosts",
                return_value=HOST.DEFAULT_FINGERPRINT,
            ),
            mock.patch.object(HOST, "OpenSshTransport", return_value=mock.Mock()),
            mock.patch.object(HOST, "HostJournal", return_value=mock.Mock()),
            mock.patch.object(HOST, "FleetRunner", return_value=fleet),
            mock.patch.object(HOST, "validate_paths_distinct"),
            mock.patch.object(sys, "stdout", output),
        ):
            status = HOST.main([])
        self.assertEqual(status, 1)
        self.assertEqual(json.loads(output.getvalue()), failed)
        fleet.candidate_soak.assert_called_once()

    def test_local_process_capture_and_cleanup_output_collision_are_bounded(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            pid_path = output_dir / "emitter-pids"
            program = (
                "import os,subprocess,sys,time;"
                "child=subprocess.Popen([sys.executable,'-c',"
                "'import time;time.sleep(60)']);"
                f"open({str(pid_path)!r},'w').write(f'{{os.getpid()}} {{child.pid}}');"
                f"os.write(1,b'o'*({HOST.MAX_OUTPUT_BYTES}+65536));"
                "time.sleep(60)"
            )
            started = time.monotonic()
            with self.assertRaisesRegex(
                HOST.RunnerError, "exceeded output limit"
            ):
                HOST.LocalProcessRunner().run(
                    [sys.executable, "-c", program], timeout=30
                )
            self.assertLess(time.monotonic() - started, 5)
            pids = [int(value) for value in pid_path.read_text().split()]
            deadline = time.monotonic() + 3
            live: list[int] = []
            while time.monotonic() < deadline:
                live = []
                for pid in pids:
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        continue
                    live.append(pid)
                if not live:
                    break
                time.sleep(0.05)
            self.assertEqual(live, [])

            cleanup_path = output_dir / "cleanup-evidence.json"
            cleanup_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(
                HOST.RunnerError, "trust inputs and run outputs"
            ):
                HOST.validate_paths_distinct(output_dir, [cleanup_path])

    def test_preflight_response_loss_runs_full_cleanup_and_persists_receipt(
        self,
    ) -> None:
        class LostPreflightTransport:
            def __init__(self, *, run_id: str, tool_sha256: str) -> None:
                self.run_id = run_id
                self.tool_sha256 = tool_sha256
                self.commands: list[str] = []
                self.preflight_claimed = False

            def invoke_board(
                self,
                command: str,
                arguments: Sequence[str] = (),
                *,
                timeout: float = 90,
            ) -> dict[str, Any]:
                del arguments, timeout
                self.commands.append(command)
                if command == "identity":
                    return {
                        "schemaVersion": 1,
                        "protocolVersion": HOST.PROTOCOL_VERSION,
                        "toolPath": HOST.REMOTE_TOOL,
                        "toolSha256": self.tool_sha256,
                        "rootOwned": True,
                        "mode": "0755",
                    }
                if command == "preflight":
                    self.preflight_claimed = True
                    raise HOST.RunnerError("simulated lost preflight response")
                if command == "cleanup":
                    self.preflight_claimed = False
                    return {
                        "schemaVersion": 1,
                        "kind": "nuvion-iq9075-cleanup-evidence",
                        "runId": self.run_id,
                        "complete": True,
                        "recovered": True,
                        "phase": None,
                        "proof": cleanup_proof(None),
                    }
                raise AssertionError(command)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_id = str(uuid.uuid4())
            output = root / run_id
            output.mkdir()
            tool = ROOT / "packaging/dev/iq9075-board-e2e.py"
            transport = LostPreflightTransport(
                run_id=run_id,
                tool_sha256=hashlib.sha256(tool.read_bytes()).hexdigest(),
            )
            runner = HOST.FleetRunner(
                transport=transport,
                journal=HOST.HostJournal(
                    output / "journal.json",
                    run_id=run_id,
                    host="iq9075",
                    fingerprint=HOST.DEFAULT_FINGERPRINT,
                ),
                output_dir=output,
                run_id=run_id,
            )
            inputs: list[Path] = []
            for name in ("command", "release", "health", "binding"):
                path = root / f"{name}.json"
                path.write_text("{}\n", encoding="utf-8")
                inputs.append(path)
            with self.assertRaisesRegex(
                HOST.RunnerError, "simulated lost preflight response"
            ):
                runner.run(
                    local_tool=tool,
                    command_keyring=inputs[0],
                    release_keyring=inputs[1],
                    health_keyring=inputs[2],
                    device_binding=inputs[3],
                    manifest_arguments={},
                    wait_seconds=30,
                    poll_seconds=1,
                )
            self.assertEqual(
                transport.commands, ["identity", "preflight", "cleanup"]
            )
            self.assertFalse(transport.preflight_claimed)
            receipt = HOST.strict_json(
                (output / "cleanup-evidence.json").read_bytes(),
                label="cleanup receipt",
            )
            self.assertTrue(receipt["complete"])
            self.assertTrue(receipt["proof"]["activeRunLeaseAbsent"])

    def test_board_call_dispatch_is_not_blocked_by_host_journal_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_id = str(uuid.uuid4())
            transport = FakeTransport()
            journal = HOST.HostJournal(
                root / "journal.json",
                run_id=run_id,
                host="iq9075",
                fingerprint=HOST.DEFAULT_FINGERPRINT,
            )
            runner = HOST.FleetRunner(
                transport=transport,
                journal=journal,
                output_dir=root,
                run_id=run_id,
            )
            journal.mark = mock.Mock(  # type: ignore[method-assign]
                side_effect=OSError("journal unavailable")
            )
            with self.assertRaisesRegex(HOST.RunnerError, "journal update failed"):
                runner._call("cleanup", "cleanup", ["--run-id", run_id])
            self.assertEqual(transport.calls, 1)

    def test_run_failure_after_backup_always_attempts_full_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_id = str(uuid.uuid4())
            runner = HOST.FleetRunner(
                transport=FakeTransport(),
                journal=HOST.HostJournal(
                    root / "journal.json",
                    run_id=run_id,
                    host="iq9075",
                    fingerprint=HOST.DEFAULT_FINGERPRINT,
                ),
                output_dir=root,
                run_id=run_id,
            )
            primary = HOST.RunnerError("post-backup evidence failure")

            def fail_after_backup(**_kwargs):
                runner._run_cleanup_required = True
                raise primary

            runner._run_once = fail_after_backup  # type: ignore[method-assign]
            cleanup = mock.Mock(return_value={"complete": True})
            runner.cleanup = cleanup  # type: ignore[method-assign]
            with self.assertRaisesRegex(
                HOST.RunnerError, "post-backup evidence failure"
            ) as raised:
                runner.run(
                    local_tool=root / "tool",
                    command_keyring=root / "command",
                    release_keyring=root / "release",
                    health_keyring=root / "health",
                    device_binding=root / "binding",
                    manifest_arguments={},
                    wait_seconds=30,
                    poll_seconds=1,
                )
            self.assertIs(raised.exception, primary)
            cleanup.assert_called_once_with()

            runner._run_once = fail_after_backup  # type: ignore[method-assign]
            runner.cleanup = mock.Mock(  # type: ignore[method-assign]
                side_effect=HOST.RunnerError("cleanup failed")
            )
            with self.assertRaisesRegex(
                HOST.RunnerError, "full cleanup did not converge"
            ):
                runner.run(
                    local_tool=root / "tool",
                    command_keyring=root / "command",
                    release_keyring=root / "release",
                    health_keyring=root / "health",
                    device_binding=root / "binding",
                    manifest_arguments={},
                    wait_seconds=30,
                    poll_seconds=1,
                )
            self.assertEqual(
                runner.journal.state["steps"]["run-recovery-pending"]["status"],
                "FAILED",
            )

    def test_candidate_rpc_failure_still_runs_full_cleanup_and_persists_receipt(
        self,
    ) -> None:
        class FailedCandidateTransport:
            def __init__(self, *, run_id: str, tool_sha256: str) -> None:
                self.run_id = run_id
                self.tool_sha256 = tool_sha256
                self.commands: list[str] = []

            def copy_candidate_input(
                self, source: Path, *, run_id: str, role: str
            ) -> str:
                del source
                return f"/tmp/nuvion-fleet-e2e-{run_id}-{role}"

            def invoke_board(
                self,
                command: str,
                arguments: Sequence[str] = (),
                *,
                timeout: float = 90,
            ) -> dict[str, Any]:
                del arguments, timeout
                self.commands.append(command)
                if command == "identity":
                    return {
                        "schemaVersion": 1,
                        "protocolVersion": HOST.PROTOCOL_VERSION,
                        "toolPath": HOST.REMOTE_TOOL,
                        "toolSha256": self.tool_sha256,
                        "rootOwned": True,
                        "mode": "0755",
                    }
                if command == "candidate-soak":
                    raise HOST.RunnerError("simulated lost SSH response")
                if command == "discard-candidate-staging":
                    return {"schemaVersion": 1, "complete": True}
                if command == "cleanup":
                    return {
                        "schemaVersion": 1,
                        "kind": "nuvion-iq9075-cleanup-evidence",
                        "runId": self.run_id,
                        "complete": True,
                        "recovered": True,
                        "phase": "RESTORED",
                        "proof": cleanup_proof(),
                    }
                raise AssertionError(command)

        with tempfile.TemporaryDirectory() as directory:
            run_id = str(uuid.uuid4())
            output = Path(directory) / run_id
            output.mkdir()
            for name in ("immutable-manifest.json", "evidence.json"):
                (output / name).write_text("{}\n", encoding="utf-8")
            tool = Path(directory) / "board.py"
            harness = Path(directory) / "oak.sh"
            bundle = Path(directory) / "bundle.tar.gz"
            bom = Path(directory) / "bom.json"
            for path, payload in (
                (tool, b"tool"),
                (harness, b"harness"),
                (bundle, b"bundle"),
                (bom, b"{}\n"),
            ):
                path.write_bytes(payload)
            transport = FailedCandidateTransport(
                run_id=run_id,
                tool_sha256=hashlib.sha256(tool.resolve().read_bytes()).hexdigest(),
            )
            runner = HOST.FleetRunner(
                transport=transport,
                journal=HOST.HostJournal(
                    output / "journal.json",
                    run_id=run_id,
                    host="iq9075",
                    fingerprint=HOST.DEFAULT_FINGERPRINT,
                ),
                output_dir=output,
                run_id=run_id,
            )
            manifest = {"scenario": {"type": "oak-fault-rollback"}}
            with (
                mock.patch.object(HOST, "validate_manifest", return_value=manifest),
                mock.patch.object(HOST, "validate_final_evidence"),
                mock.patch.object(HOST, "validate_candidate_inputs"),
                self.assertRaisesRegex(HOST.RunnerError, "full cleanup=complete"),
            ):
                runner.candidate_soak(
                    local_tool=tool.resolve(),
                    local_harness=harness.resolve(),
                    candidate_bundle=bundle.resolve(),
                    candidate_bom=bom.resolve(),
                )

            self.assertLess(
                transport.commands.index("candidate-soak"),
                transport.commands.index("cleanup"),
            )
            receipt = HOST.strict_json(
                (output / "cleanup-evidence.json").read_bytes(),
                label="cleanup receipt",
            )
            self.assertTrue(receipt["complete"])
            self.assertTrue(receipt["proof"]["activeRunLeaseAbsent"])
            self.assertFalse((output / "candidate-soak-evidence.json").exists())

    def test_candidate_soak_validator_binds_raw_and_fresh_restore(self) -> None:
        run_id = str(uuid.uuid4())
        digest = "e" * 64
        manifest = HOST.build_manifest(
            run_id=run_id,
            tool_sha256="f" * 64,
            input_digests={
                "commandSha256": "1" * 64,
                "releaseSha256": "2" * 64,
                "healthSha256": "3" * 64,
                "bindingSha256": "4" * 64,
            },
            identity={
                "deviceId": "sp-3-nuvion-iq9075",
                "spaceId": 3,
                "productModel": "IQ9075_DEV",
                "platformProfile": "iq9075_dev",
                "hardwareRevision": "QCS9075-EVK",
                "architecture": "aarch64",
                "dockerRequired": False,
            },
            scenario_type="oak-fault-rollback",
            expected_command_id=str(uuid.uuid4()),
            expected_bom_digest=f"sha256:{digest}",
            expected_candidate_slot=f"/opt/nuv-agent/releases/{digest}",
            expected_previous_slot="releases/" + "d" * 64,
            expected_previous_version="0.1.120",
            hold_seconds=10,
            release={
                "agentVersion": "0.1.121",
                "releaseSequence": 2,
                "artifactDigest": "sha256:" + "a" * 64,
                "componentSha": "b" * 40,
                "configSchema": "12",
                "publisherKeyId": "release-test",
            },
        )
        candidate_slot = f"/opt/nuv-agent/candidates/{run_id}-{digest}"
        raw = {
            "schemaVersion": 3,
            "kind": "nuvion-iq9075-oak-soak-result",
            "runId": run_id,
            "slotKind": "candidate",
            "startedAt": "2026-09-03T00:01:00Z",
            "outcome": {"status": "passed"},
            "runtimeIdentity": {
                "agentVersion": "0.1.121",
                "componentSha": "b" * 40,
                "bomDigest": "sha256:" + digest,
                "candidateSlot": candidate_slot,
                "pythonPath": "/usr/bin/python3",
                "sitePackagesPath": candidate_slot
                + "/venv/lib/python3.12/site-packages",
                "buildInfoPath": candidate_slot
                + "/venv/lib/python3.12/site-packages/nuvion_app/build_info.py",
                "releaseMarkerSha256": "6" * 64,
                "controlMarkerSha256": "9" * 64,
            },
        }
        raw_sha = hashlib.sha256(
            (
                json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode()
        ).hexdigest()
        slots = {"current": "releases/" + "d" * 64, "previous": "releases/" + digest}
        anti = {
            "schemaVersion": 4,
            "semanticSha256": "8" * 64,
            "maximumCommandSequence": 2,
            "currentReleaseSequence": "1",
            "currentBomDigest": "sha256:" + "d" * 64,
            "latest": {
                "commandId": manifest["scenario"]["expectedCommandId"],
                "sequence": 2,
                "phase": "ROLLED_BACK",
                "bomDigest": "sha256:" + digest,
                "releaseSequence": 2,
                "healthDeadline": None,
            },
        }
        oak = {
            "port": "2-1.1",
            "vendorId": "03e7",
            "productId": "f63b",
            "speedMbps": 5000,
            "mxidSha256": "7" * 64,
            "attached": True,
            "bound": True,
        }
        before_runtime = {
            "pid": 101,
            "startTicks": 1000,
            "bootId": "11111111-1111-4111-8111-111111111111",
            "activeSlot": slots["current"],
        }
        after_runtime = {**before_runtime, "pid": 202, "startTicks": 2000}
        persistent_state = persistent_state_evidence(
            HOST.CANDIDATE_PERSISTENT_PATHS
        )
        release_trees = release_tree_evidence(slots)
        evidence = {
            "schemaVersion": 1,
            "kind": "nuvion-iq9075-candidate-soak-evidence",
            "protocolVersion": HOST.PROTOCOL_VERSION,
            "runId": run_id,
            "startedAt": "2026-09-03T00:00:00Z",
            "completedAt": "2026-09-03T00:03:00Z",
            "complete": True,
            "outcome": {"status": "passed", "errorCode": None},
            "candidate": {
                "slotKind": "candidate",
                "slot": candidate_slot,
                "bomDigest": f"sha256:{digest}",
                "bundleSha256": "a" * 64,
                "bomSha256": "c" * 64,
                "harnessSha256": "6" * 64,
                "controlMarkerSha256": "9" * 64,
            },
            "fleetEvidenceSha256": "5" * 64,
            "rawEvidenceSha256": raw_sha,
            "rawEvidence": raw,
            "executionProof": candidate_execution_proof(run_id),
            "collectorProof": candidate_collector_proof(run_id),
            "terminationProof": candidate_termination_proof(run_id),
            "productionRestoration": production_restoration_evidence(manifest),
            "pre": {
                "slots": slots,
                "antiReplay": anti,
                "oak": oak,
                "runtime": before_runtime,
                "persistentState": copy.deepcopy(persistent_state),
                "releaseTrees": copy.deepcopy(release_trees),
            },
            "post": {
                "restoredAt": "2026-09-03T00:02:00Z",
                "slots": slots,
                "antiReplay": anti,
                "oak": oak,
                "runtime": after_runtime,
                "persistentState": persistent_state,
                "releaseTrees": release_trees,
            },
            "gates": {
                "signedRollbackTerminal": True,
                "candidateBound": True,
                "rawEvidencePreserved": True,
                "slotsUnchanged": True,
                "releaseTreesUnchanged": True,
                "antiReplayUnchanged": True,
                "oakIdentityUnchanged": True,
                "freshBaselineProcess": True,
                "harnessBytesPinned": True,
                "harnessCopyRemoved": True,
                "resourceLimitsApplied": True,
                "boundedOutput": True,
                "persistentStateReadOnly": True,
                "persistentStateUnchanged": True,
                "productionTrustRestored": True,
                "trustedSoakDuration": True,
                "continuousUidIsolation": True,
                "cgroupTerminated": True,
                "harnessPassed": True,
            },
        }
        HOST.validate_candidate_soak_evidence(
            evidence,
            run_id=run_id,
            manifest=manifest,
            bundle_sha256="a" * 64,
            bom_sha256="c" * 64,
            harness_sha256="6" * 64,
            fleet_evidence_sha256="5" * 64,
            raw_evidence_sha256=raw_sha,
        )
        with self.assertRaisesRegex(HOST.RunnerError, "raw bytes digest"):
            HOST.validate_candidate_soak_evidence(
                evidence,
                run_id=run_id,
                manifest=manifest,
                bundle_sha256="a" * 64,
                bom_sha256="c" * 64,
                harness_sha256="6" * 64,
                fleet_evidence_sha256="5" * 64,
                raw_evidence_sha256="0" * 64,
            )
        unrestored = copy.deepcopy(evidence)
        restoration = unrestored["productionRestoration"]
        restoration["transactionPhase"] = "APPLIED"
        restoration["sha256"] = hashlib.sha256(
            (
                json.dumps(
                    {key: value for key, value in restoration.items() if key != "sha256"},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode()
        ).hexdigest()
        with self.assertRaisesRegex(HOST.RunnerError, "restoration proof"):
            HOST.validate_candidate_soak_evidence(
                unrestored,
                run_id=run_id,
                manifest=manifest,
                bundle_sha256="a" * 64,
                bom_sha256="c" * 64,
                harness_sha256="6" * 64,
                fleet_evidence_sha256="5" * 64,
            )
        ancestor_mount = copy.deepcopy(evidence)
        for item in ancestor_mount["executionProof"]["mountSandbox"][
            "readOnlyPaths"
        ].values():
            item["mountId"] = 141
            item["mountPoint"] = "/"
        HOST.validate_candidate_soak_evidence(
            ancestor_mount,
            run_id=run_id,
            manifest=manifest,
            bundle_sha256="a" * 64,
            bom_sha256="c" * 64,
            harness_sha256="6" * 64,
            fleet_evidence_sha256="5" * 64,
        )
        wrong_ancestor = copy.deepcopy(ancestor_mount)
        wrong_ancestor["executionProof"]["mountSandbox"]["readOnlyPaths"][
            "/etc/nuv-agent"
        ]["mountPoint"] = "/var"
        with self.assertRaisesRegex(HOST.RunnerError, "restoration proof"):
            HOST.validate_candidate_soak_evidence(
                wrong_ancestor,
                run_id=run_id,
                manifest=manifest,
                bundle_sha256="a" * 64,
                bom_sha256="c" * 64,
                harness_sha256="6" * 64,
                fleet_evidence_sha256="5" * 64,
            )
        failed_without_raw = copy.deepcopy(evidence)
        failed_without_raw["outcome"] = {
            "status": "failed",
            "errorCode": "OAK_EVIDENCE_MISSING",
        }
        failed_without_raw["rawEvidence"] = None
        failed_without_raw["rawEvidenceSha256"] = None
        failed_without_raw["gates"]["rawEvidencePreserved"] = False
        failed_without_raw["gates"]["harnessPassed"] = False
        HOST.validate_candidate_soak_evidence(
            failed_without_raw,
            run_id=run_id,
            manifest=manifest,
            bundle_sha256="a" * 64,
            bom_sha256="c" * 64,
            harness_sha256="6" * 64,
            fleet_evidence_sha256="5" * 64,
        )
        tampered = copy.deepcopy(evidence)
        tampered["post"]["antiReplay"] = {"sha256": "0" * 64, "sizeBytes": 4096}
        with self.assertRaisesRegex(HOST.RunnerError, "restoration proof"):
            HOST.validate_candidate_soak_evidence(
                tampered,
                run_id=run_id,
                manifest=manifest,
                bundle_sha256="a" * 64,
                bom_sha256="c" * 64,
                harness_sha256="6" * 64,
                fleet_evidence_sha256="5" * 64,
            )
        persistent_drift = copy.deepcopy(evidence)
        persistent_drift["post"]["persistentState"]["roots"][
            "/var/lib/nuv-agent"
        ]["sha256"] = "f" * 64
        roots = persistent_drift["post"]["persistentState"]["roots"]
        serialized = (
            json.dumps(roots, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        persistent_drift["post"]["persistentState"]["sha256"] = (
            hashlib.sha256(serialized).hexdigest()
        )
        with self.assertRaisesRegex(HOST.RunnerError, "restoration proof"):
            HOST.validate_candidate_soak_evidence(
                persistent_drift,
                run_id=run_id,
                manifest=manifest,
                bundle_sha256="a" * 64,
                bom_sha256="c" * 64,
                harness_sha256="6" * 64,
                fleet_evidence_sha256="5" * 64,
            )
        resource_drift = copy.deepcopy(evidence)
        resource_drift["executionProof"]["systemdProperties"]["MemoryMax"] = "max"
        with self.assertRaisesRegex(HOST.RunnerError, "restoration proof"):
            HOST.validate_candidate_soak_evidence(
                resource_drift,
                run_id=run_id,
                manifest=manifest,
                bundle_sha256="a" * 64,
                bom_sha256="c" * 64,
                harness_sha256="6" * 64,
                fleet_evidence_sha256="5" * 64,
            )
        core_dump_enabled = copy.deepcopy(evidence)
        core_dump_enabled["executionProof"]["systemdProperties"]["LimitCORE"] = (
            "infinity"
        )
        with self.assertRaisesRegex(HOST.RunnerError, "restoration proof"):
            HOST.validate_candidate_soak_evidence(
                core_dump_enabled,
                run_id=run_id,
                manifest=manifest,
                bundle_sha256="a" * 64,
                bom_sha256="c" * 64,
                harness_sha256="6" * 64,
                fleet_evidence_sha256="5" * 64,
            )
        for label, mutate in (
            (
                "retained-cgroup",
                lambda item: item["terminationProof"].update(
                    cgroupRemoved=False
                ),
            ),
            (
                "loaded-unit",
                lambda item: item["terminationProof"].update(
                    loadState="loaded"
                ),
            ),
            (
                "pre-existing-same-uid-process",
                lambda item: item["executionProof"]["uidIsolation"][
                    "before"
                ].update(pids=[777]),
            ),
        ):
            with self.subTest(label=label):
                invalid_boundary = copy.deepcopy(evidence)
                mutate(invalid_boundary)
                with self.assertRaisesRegex(HOST.RunnerError, "restoration proof"):
                    HOST.validate_candidate_soak_evidence(
                        invalid_boundary,
                        run_id=run_id,
                        manifest=manifest,
                        bundle_sha256="a" * 64,
                        bom_sha256="c" * 64,
                        harness_sha256="6" * 64,
                        fleet_evidence_sha256="5" * 64,
                    )
        release_drift = copy.deepcopy(evidence)
        release_drift["post"]["releaseTrees"]["slots"]["current"][
            "sha256"
        ] = "0" * 64
        trees = release_drift["post"]["releaseTrees"]["slots"]
        serialized = (
            json.dumps(trees, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        release_drift["post"]["releaseTrees"]["sha256"] = hashlib.sha256(
            serialized
        ).hexdigest()
        with self.assertRaisesRegex(HOST.RunnerError, "restoration proof"):
            HOST.validate_candidate_soak_evidence(
                release_drift,
                run_id=run_id,
                manifest=manifest,
                bundle_sha256="a" * 64,
                bom_sha256="c" * 64,
                harness_sha256="6" * 64,
                fleet_evidence_sha256="5" * 64,
            )

    def test_host_cleanup_rejects_incomplete_board_restore(self) -> None:
        class IncompleteCleanupTransport(FakeTransport):
            def invoke_board(
                self,
                command: str,
                arguments: Sequence[str] = (),
                *,
                timeout: float = 90,
            ) -> dict[str, Any]:
                del arguments, timeout
                self.calls += 1
                self.assert_cleanup_command = command
                return {
                    "schemaVersion": 1,
                    "runId": run_id,
                    "complete": False,
                    "recovered": False,
                    "phase": "RESTORING",
                }

        with tempfile.TemporaryDirectory() as directory:
            run_id = str(uuid.uuid4())
            output = Path(directory) / run_id
            output.mkdir()
            transport = IncompleteCleanupTransport()
            runner = HOST.FleetRunner(
                transport=transport,
                journal=HOST.HostJournal(
                    output / "journal.json",
                    run_id=run_id,
                    host="iq9075",
                    fingerprint=HOST.DEFAULT_FINGERPRINT,
                ),
                output_dir=output,
                run_id=run_id,
            )

            with self.assertRaisesRegex(HOST.RunnerError, "exact restored state"):
                runner.cleanup()

            self.assertEqual(transport.calls, 1)
            self.assertEqual(transport.assert_cleanup_command, "cleanup")
            self.assertEqual(
                runner.journal.state["steps"]["cleanup"]["status"],
                "FAILED",
            )

    def test_host_cleanup_accepts_only_exact_success_contract(self) -> None:
        run_id = str(uuid.uuid4())
        exact = {
            "schemaVersion": 1,
            "kind": "nuvion-iq9075-cleanup-evidence",
            "runId": run_id,
            "complete": True,
            "recovered": False,
            "phase": "RESTORED",
            "idempotent": True,
            "proof": cleanup_proof(),
        }

        HOST.validate_cleanup_result(exact, run_id=run_id)
        with self.assertRaises(HOST.RunnerError):
            HOST.validate_cleanup_result({**exact, "unexpected": True}, run_id=run_id)
        with self.assertRaises(HOST.RunnerError):
            HOST.validate_cleanup_result(exact, run_id=str(uuid.uuid4()))
        with self.assertRaises(HOST.RunnerError):
            HOST.validate_cleanup_result(
                {**exact, "schemaVersion": True}, run_id=run_id
            )

    def _known_hosts(self, root: Path) -> tuple[Path, str]:
        key = b"deterministic-iq9075-host-public-key"
        encoded = base64.b64encode(key).decode()
        fingerprint = "SHA256:" + base64.b64encode(hashlib.sha256(key).digest()).decode(
            "ascii"
        ).rstrip("=")
        path = root / "source-known-hosts"
        path.write_text(f"iq9075 ssh-ed25519 {encoded}\n", encoding="utf-8")
        path.chmod(0o600)
        return path, fingerprint

    def test_password_mode_disables_config_proxy_control_and_all_other_auth(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, fingerprint = self._known_hosts(root)
            pinned = root / "known_hosts"
            HOST.create_pinned_known_hosts(
                source,
                pinned,
                host="iq9075",
                port=22,
                expected=fingerprint,
            )
            transport = HOST.OpenSshTransport(
                host="iq9075",
                user="plaid",
                port=22,
                pinned_known_hosts=pinned,
                expected_fingerprint=fingerprint,
                ssh_password=bytearray(b"password"),
                process_runner=FakeLocalRunner(),
            )
            options = " ".join(transport.base_options)
            for expected in (
                "-F /dev/null",
                "StrictHostKeyChecking=yes",
                "CheckHostIP=no",
                "UpdateHostKeys=no",
                "ControlMaster=no",
                "ControlPath=none",
                "ProxyCommand=none",
                "ProxyJump=none",
                "ForwardAgent=no",
                "ClearAllForwardings=yes",
                "PubkeyAuthentication=no",
                "HostbasedAuthentication=no",
                "GSSAPIAuthentication=no",
                "KbdInteractiveAuthentication=no",
                "ChallengeResponseAuthentication=no",
                "PreferredAuthentications=password",
            ):
                self.assertIn(expected, options)
            self.assertNotIn("CheckHostIP=yes", options)
            self.assertNotIn("accept-new", options)
            with self.assertRaisesRegex(HOST.RunnerError, "duplicate JSON"):
                HOST.strict_json('{"a":1,"a":2}', label="review repro")

    def test_candidate_upload_uses_only_fixed_run_owned_destinations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            known_hosts, fingerprint = self._known_hosts(root)
            pinned = root / "known_hosts"
            HOST.create_pinned_known_hosts(
                known_hosts,
                pinned,
                host="iq9075",
                port=22,
                expected=fingerprint,
            )
            local = FakeLocalRunner()
            transport = HOST.OpenSshTransport(
                host="iq9075",
                user="plaid",
                port=22,
                pinned_known_hosts=pinned,
                expected_fingerprint=fingerprint,
                process_runner=local,
            )
            run_id = str(uuid.uuid4())
            source = root / "candidate-input"
            source.write_bytes(b"candidate")
            expected = {
                "candidate-bundle": "candidate-bundle.tar.gz",
                "candidate-bom": "candidate-bom.json",
                "oak-harness": "oak-harness.sh",
            }
            for role, suffix in expected.items():
                destination = transport.copy_candidate_input(
                    source, run_id=run_id, role=role
                )
                self.assertEqual(
                    destination, f"/tmp/nuvion-fleet-e2e-{run_id}-{suffix}"
                )
                self.assertEqual(local.calls[-1][0], "/usr/bin/scp")
                self.assertEqual(local.calls[-1][-2], str(source))
                self.assertEqual(
                    local.calls[-1][-1], f"plaid@iq9075:{destination}"
                )
            with self.assertRaisesRegex(HOST.RunnerError, "role is invalid"):
                transport.copy_candidate_input(
                    source, run_id=run_id, role="arbitrary"
                )

    def test_askpass_serves_exactly_one_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            broker = HOST._OneShotAskpass(bytearray(b"one-shot"), directory)
            broker.start()
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.connect(str(broker.path))
                self.assertEqual(client.recv(64), b"one-shot")
            broker.finish()
            with (
                socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as second,
                self.assertRaises(OSError),
            ):
                second.connect(str(broker.path))

    def test_stale_host_journal_never_skips_board_reconcile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            journal = HOST.HostJournal(
                root / "journal.json",
                run_id=str(uuid.uuid4()),
                host="iq9075",
                fingerprint=HOST.DEFAULT_FINGERPRINT,
            )
            transport = FakeTransport()
            runner = HOST.FleetRunner(
                transport=transport,
                journal=journal,
                output_dir=root,
                run_id=journal.run_id,
            )
            runner._call("preflight", "preflight")
            runner._call("preflight", "preflight")
            self.assertEqual(transport.calls, 2)
            self.assertEqual(journal.state["steps"]["preflight"]["attempt"], 2)

    def test_run_derives_baseline_version_and_polls_until_exact_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_id = str(uuid.uuid4())
            output = root / run_id
            output.mkdir()
            tool = ROOT / "packaging/dev/iq9075-board-e2e.py"
            tool_sha = hashlib.sha256(tool.read_bytes()).hexdigest()
            baseline = "d" * 64
            candidate = "a" * 64
            transport = PollingRunTransport(tool_sha=tool_sha, baseline_digest=baseline)
            clock = FakeClock()
            runner = HOST.FleetRunner(
                transport=transport,
                journal=HOST.HostJournal(
                    output / "journal.json",
                    run_id=run_id,
                    host="iq9075",
                    fingerprint=HOST.DEFAULT_FINGERPRINT,
                ),
                output_dir=output,
                run_id=run_id,
                monotonic=clock.monotonic,
                sleeper=clock.sleep,
            )
            inputs: list[Path] = []
            for name in ("command", "release", "health", "binding"):
                path = root / f"{name}.json"
                path.write_text(f'{{"role":"{name}"}}\n', encoding="utf-8")
                path.chmod(0o600)
                inputs.append(path)
            manifest_arguments = {
                "identity": {
                    "deviceId": "sp-3-nuvion-iq9075",
                    "spaceId": 3,
                    "productModel": "IQ9075_DEV",
                    "platformProfile": "iq9075_dev",
                    "hardwareRevision": "QCS9075-EVK",
                    "architecture": "aarch64",
                    "dockerRequired": False,
                },
                "scenario_type": "commit",
                "expected_command_id": str(uuid.uuid4()),
                "expected_bom_digest": f"sha256:{candidate}",
                "expected_candidate_slot": f"/opt/nuv-agent/releases/{candidate}",
                "expected_previous_slot": f"releases/{baseline}",
                "hold_seconds": 0,
                "release": {
                    "agentVersion": "0.1.121",
                    "releaseSequence": 2,
                    "artifactDigest": "sha256:" + "b" * 64,
                    "componentSha": "c" * 40,
                    "configSchema": "12",
                    "publisherKeyId": "release-iq9075-dev",
                },
            }
            result = runner.run(
                local_tool=tool,
                command_keyring=inputs[0],
                release_keyring=inputs[1],
                health_keyring=inputs[2],
                device_binding=inputs[3],
                wait_seconds=30,
                poll_seconds=0.5,
                manifest_arguments=manifest_arguments,
            )
            self.assertTrue(result["complete"])
            self.assertGreaterEqual(transport.evidence_calls, 2)
            self.assertEqual(
                transport.manifest["scenario"]["expectedPreviousVersion"],
                "0.1.120",
            )
            evidence_path = output / "evidence.json"
            evidence_before = evidence_path.read_bytes()
            resumed = runner.run(
                local_tool=tool,
                command_keyring=inputs[0],
                release_keyring=inputs[1],
                health_keyring=inputs[2],
                device_binding=inputs[3],
                wait_seconds=30,
                poll_seconds=0.5,
                manifest_arguments=manifest_arguments,
            )
            self.assertTrue(resumed["complete"])
            self.assertEqual(evidence_path.read_bytes(), evidence_before)

    def test_final_validator_rejects_any_false_gate(self) -> None:
        run_id = str(uuid.uuid4())
        manifest = HOST.build_manifest(
            run_id=run_id,
            tool_sha256="f" * 64,
            input_digests={
                "commandSha256": "1" * 64,
                "releaseSha256": "2" * 64,
                "healthSha256": "3" * 64,
                "bindingSha256": "4" * 64,
            },
            identity={
                "deviceId": "sp-3-nuvion-iq9075",
                "spaceId": 3,
                "productModel": "IQ9075_DEV",
                "platformProfile": "iq9075_dev",
                "hardwareRevision": "QCS9075-EVK",
                "architecture": "aarch64",
                "dockerRequired": False,
            },
            scenario_type="commit",
            expected_command_id=str(uuid.uuid4()),
            expected_bom_digest="sha256:" + "a" * 64,
            expected_candidate_slot="/opt/nuv-agent/releases/" + "a" * 64,
            expected_previous_slot="releases/" + "b" * 64,
            expected_previous_version="0.1.120",
            hold_seconds=0,
            release={
                "agentVersion": "0.1.121",
                "releaseSequence": 2,
                "artifactDigest": "sha256:" + "c" * 64,
                "componentSha": "d" * 40,
                "configSchema": "12",
                "publisherKeyId": "release-test",
            },
        )
        evidence = {
            "schemaVersion": 1,
            "protocolVersion": HOST.PROTOCOL_VERSION,
            "runId": run_id,
            "generatedAt": "2026-09-02T00:00:00Z",
            "scenario": "commit",
            "complete": True,
            "gates": {
                "foundation": True,
                "backup": True,
                "trust": True,
                "updater2": True,
                "oak": True,
                "services": True,
                "scenario": False,
            },
            "oak": {},
            "services": {},
            "runtimePids": None,
            "slots": {},
            "updater": {"updaterVersion": "0.2.0", "update": {}},
        }
        with self.assertRaisesRegex(HOST.RunnerError, "false or missing gate"):
            HOST.validate_final_evidence(evidence, manifest)

    def test_commit_evidence_rejects_contradictory_fields_and_expired_order(self) -> None:
        baseline = "b" * 64
        transport = PollingRunTransport(tool_sha="f" * 64, baseline_digest=baseline)
        manifest = HOST.build_manifest(
            run_id=str(uuid.uuid4()),
            tool_sha256="f" * 64,
            input_digests={
                "commandSha256": "1" * 64,
                "releaseSha256": "2" * 64,
                "healthSha256": "3" * 64,
                "bindingSha256": "4" * 64,
            },
            identity={
                "deviceId": "sp-3-nuvion-iq9075",
                "spaceId": 3,
                "productModel": "IQ9075_DEV",
                "platformProfile": "iq9075_dev",
                "hardwareRevision": "QCS9075-EVK",
                "architecture": "aarch64",
                "dockerRequired": False,
            },
            scenario_type="commit",
            expected_command_id=str(uuid.uuid4()),
            expected_bom_digest="sha256:" + "a" * 64,
            expected_candidate_slot="/opt/nuv-agent/releases/" + "a" * 64,
            expected_previous_slot="releases/" + baseline,
            expected_previous_version="0.1.120",
            hold_seconds=0,
            release={
                "agentVersion": "0.1.121",
                "releaseSequence": 2,
                "artifactDigest": "sha256:" + "c" * 64,
                "componentSha": "d" * 40,
                "configSchema": "12",
                "publisherKeyId": "release-iq9075-dev",
            },
        )
        transport.manifest = manifest
        transport.evidence_calls = 1
        evidence = transport.invoke_board("evidence")
        HOST.validate_final_evidence(evidence, manifest)
        same_second = copy.deepcopy(evidence)
        same_second["updater"]["update"]["updatedAt"] = (
            "2026-09-02T00:00:02.900Z"
        )
        same_second["generatedAt"] = "2026-09-02T00:00:02.901Z"
        HOST.validate_final_evidence(same_second, manifest)
        generated_too_early = copy.deepcopy(same_second)
        generated_too_early["generatedAt"] = "2026-09-02T00:00:02.899Z"
        with self.assertRaisesRegex(HOST.RunnerError, "predates"):
            HOST.validate_final_evidence(generated_too_early, manifest)

        mutations = {
            "errorCode": lambda item: item["updater"]["update"].__setitem__(
                "errorCode", "CONTRADICTORY"
            ),
            "message": lambda item: item["updater"]["update"].__setitem__(
                "message", "contradictory"
            ),
            "rollbackSlot": lambda item: item["updater"]["update"].__setitem__(
                "rollbackSlot", "releases/" + baseline
            ),
            "healthDeadline": lambda item: item["updater"]["update"].__setitem__(
                "healthDeadline", "2026-09-02T00:30:00Z"
            ),
            "expired": lambda item: item["updater"]["update"].__setitem__(
                "updatedAt", "2026-09-02T02:00:00Z"
            ),
            "generated-before-update": lambda item: item.__setitem__(
                "generatedAt", "2026-09-01T23:59:59Z"
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                candidate = copy.deepcopy(evidence)
                mutate(candidate)
                with self.assertRaises(HOST.RunnerError):
                    HOST.validate_final_evidence(candidate, manifest)

    def test_out_of_band_bootstrap_is_separate_non_ota_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_id = str(uuid.uuid4())
            output = root / run_id
            output.mkdir()
            tool = ROOT / "packaging/dev/iq9075-board-e2e.py"
            installer = ROOT / "packaging/dev/install-iq9075.sh"
            package = root / "nuv-agent_0.1.121_arm64.deb"
            package.write_bytes(b"deterministic-test-deb")
            package.chmod(0o600)
            package_sha = hashlib.sha256(package.read_bytes()).hexdigest()
            tool_sha = hashlib.sha256(tool.read_bytes()).hexdigest()
            transport = FakeBootstrapTransport(
                local_tool_sha=tool_sha,
                current_slot="bootstrap/0.1.121",
            )
            journal = HOST.HostJournal(
                output / "journal.json",
                run_id=run_id,
                host="iq9075",
                fingerprint=HOST.DEFAULT_FINGERPRINT,
            )
            runner = HOST.FleetRunner(
                transport=transport,
                journal=journal,
                output_dir=output,
                run_id=run_id,
            )
            result = runner.bootstrap(
                installer=installer,
                package=package,
                local_tool=tool,
                expected_version="0.1.121",
                expected_package_sha256=package_sha,
            )
            self.assertTrue(result["bootstrapComplete"])
            self.assertFalse(result["otaEvidence"])
            self.assertEqual(transport.cleanup_calls, 1)
            evidence = json.loads(
                (output / "bootstrap-evidence.json").read_text(encoding="utf-8")
            )
            self.assertTrue(evidence["outOfBandBootstrap"])
            self.assertFalse(evidence["otaEvidence"])
            self.assertEqual(evidence["updaterCodeVersion"], "0.2.0")
            self.assertTrue(evidence["boardToolIdentityVerified"])
            self.assertNotIn("foundationVerified", evidence)

    def test_failed_out_of_band_bootstrap_still_cleans_fixed_staging(self) -> None:
        class FailingBootstrapTransport(FakeBootstrapTransport):
            def bootstrap_updater(self, **arguments: object) -> dict[str, object]:
                del arguments
                raise HOST.RunnerError("simulated bootstrap failure")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_id = str(uuid.uuid4())
            output = root / run_id
            output.mkdir()
            tool = ROOT / "packaging/dev/iq9075-board-e2e.py"
            installer = ROOT / "packaging/dev/install-iq9075.sh"
            package = root / "nuv-agent_0.1.121_arm64.deb"
            package.write_bytes(b"deterministic-test-deb")
            package.chmod(0o600)
            transport = FailingBootstrapTransport(
                local_tool_sha=hashlib.sha256(tool.read_bytes()).hexdigest(),
                current_slot="bootstrap/0.1.121",
            )
            runner = HOST.FleetRunner(
                transport=transport,
                journal=HOST.HostJournal(
                    output / "journal.json",
                    run_id=run_id,
                    host="iq9075",
                    fingerprint=HOST.DEFAULT_FINGERPRINT,
                ),
                output_dir=output,
                run_id=run_id,
            )
            with self.assertRaisesRegex(HOST.RunnerError, "simulated"):
                runner.bootstrap(
                    installer=installer,
                    package=package,
                    local_tool=tool,
                    expected_version="0.1.121",
                    expected_package_sha256=hashlib.sha256(
                        package.read_bytes()
                    ).hexdigest(),
                )
            self.assertEqual(transport.cleanup_calls, 1)
            self.assertFalse((output / "bootstrap-evidence.json").exists())

    def test_bootstrap_program_is_typed_private_and_fail_closed(self) -> None:
        compile(HOST.BOOTSTRAP_REMOTE_PROGRAM, "<bootstrap>", "exec")
        compile(HOST.BOOTSTRAP_CLEANUP_PROGRAM, "<cleanup>", "exec")
        for required in (
            "/var/lib/nuvion-fleet-e2e/bootstrap",
            "O_NOFOLLOW",
            'control("Package")',
            'control("Architecture")',
            'UPDATER_VERSION != "0.2.0"',
            '"otaEvidence": False',
            'evidence["servicesInactive"] = True',
        ):
            self.assertIn(required, HOST.BOOTSTRAP_REMOTE_PROGRAM)
        self.assertNotIn("shell=True", HOST.BOOTSTRAP_REMOTE_PROGRAM)
        self.assertIn(
            "cleanup_failures.extend(quiesce_runtime())",
            HOST.BOOTSTRAP_REMOTE_PROGRAM,
        )
        help_text = HOST.build_parser().format_help()
        self.assertIn("bootstrap-updater", help_text)

    def test_bootstrap_quiesce_attempts_every_unit_and_preserves_both_errors(
        self,
    ) -> None:
        syntax = ast.parse(HOST.BOOTSTRAP_REMOTE_PROGRAM)
        functions = {
            node.name: node
            for node in syntax.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"quiesce_runtime", "bootstrap_failure_message"}
        }
        self.assertEqual(
            set(functions),
            {"quiesce_runtime", "bootstrap_failure_message"},
        )

        class FakeSubprocess:
            DEVNULL = object()
            PIPE = object()

            def __init__(self) -> None:
                self.calls: list[tuple[str, str]] = []

            def run(self, argv: Sequence[str], **_kwargs: object) -> object:
                action, unit = argv[1], argv[2]
                self.calls.append((action, unit))
                if action in {"stop", "disable"}:
                    return SimpleNamespace(
                        returncode=(
                            1
                            if (action, unit)
                            == ("stop", "nuv-agent.service")
                            else 0
                        ),
                        stdout="",
                    )
                return SimpleNamespace(
                    returncode=3 if action == "is-active" else 1,
                    stdout="inactive\n" if action == "is-active" else "disabled\n",
                )

        fake = FakeSubprocess()
        namespace: dict[str, object] = {
            "subprocess": fake,
            "RUNTIME_UNITS": (
                "nuv-agent.service",
                "nuv-agent-updater.socket",
                "nuv-agent-updater.service",
            ),
        }
        extracted = ast.Module(body=list(functions.values()), type_ignores=[])
        ast.fix_missing_locations(extracted)
        exec(compile(extracted, "<bootstrap-functions>", "exec"), namespace)

        errors = namespace["quiesce_runtime"]()
        message = namespace["bootstrap_failure_message"](
            "INSTALL_FAILED",
            errors,
        )

        for unit in namespace["RUNTIME_UNITS"]:
            self.assertIn(("stop", unit), fake.calls)
            self.assertIn(("disable", unit), fake.calls)
            self.assertIn(("is-active", unit), fake.calls)
            self.assertIn(("is-enabled", unit), fake.calls)
        self.assertIn("nuv-agent.service:stop:rc1", errors)
        self.assertIn("primary=INSTALL_FAILED", message)
        self.assertIn("cleanup=nuv-agent.service:stop:rc1", message)

        safe_detail = (
            "out-of-band updater bootstrap failed: primary=INSTALL_FAILED; "
            "cleanup=nuv-agent.service:stop:rc1"
        )
        with self.assertRaisesRegex(
            HOST.RunnerError,
            "primary=INSTALL_FAILED.*cleanup=nuv-agent.service:stop:rc1",
        ):
            HOST.OpenSshTransport._parse_result(
                HOST.ProcessResult(1, b"", (safe_detail + "\n").encode()),
                operation="out-of-band-updater-bootstrap",
            )
        with self.assertRaisesRegex(HOST.RunnerError, "remote operation failed"):
            HOST.OpenSshTransport._parse_result(
                HOST.ProcessResult(1, b"", b"password=must-not-surface\n"),
                operation="out-of-band-updater-bootstrap",
            )

    def test_deb_installs_root_owned_board_tool_at_fixed_path(self) -> None:
        build = (ROOT / "packaging/deb/build-deb.sh").read_text(encoding="utf-8")
        self.assertIn("$PKG_DIR/usr/local/libexec/nuvion", build)
        self.assertIn('"$ROOT_DIR/packaging/dev/iq9075-board-e2e.py"', build)
        self.assertIn('"$PKG_DIR/usr/local/libexec/nuvion/iq9075-board-e2e.py"', build)
        self.assertIn("install -m 0755", build)


if __name__ == "__main__":
    unittest.main()
