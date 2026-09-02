from __future__ import annotations

import base64
import ast
import hashlib
import importlib.util
import json
import os
import socket
import sys
import tarfile
import tempfile
import unittest
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
            self.deadmen[unit] = True
            return BOARD.CommandResult(0, "", "")
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
                return BOARD.CommandResult(0, f"{status['pid']}\n", "")
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
        self._write("/etc/os-release", 'ID=ubuntu\nVERSION_ID="24.04"\n', 0o644)
        self._write("/proc/device-tree/model", "Thundercomm IQ-9075 QCS9075\0", 0o444)
        self._write("/sys/bus/usb/devices/2-1/idVendor", "03e7\n", 0o444)
        self._write("/sys/bus/usb/devices/2-1/idProduct", "f63b\n", 0o444)
        self._write("/sys/bus/usb/devices/2-1/speed", "5000\n", 0o444)
        self._write("/sys/bus/usb/drivers/usb/unbind", "", 0o600)
        self._write("/sys/bus/usb/drivers/usb/bind", "", 0o600)
        (self.root / "sys/bus/usb/devices/2-1/driver").symlink_to("../../drivers/usb")
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
        (install / "current").symlink_to(f"releases/{self.previous_digest}")
        (install / "previous").symlink_to("bootstrap/0.1.119")
        self.paths.lock_root.mkdir(parents=True, exist_ok=True)

    def _usb_hook(self, action: str) -> None:
        driver = self.paths.usb_device / "driver"
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
            "targetVersion": "0.1.121",
            "bomDigest": f"sha256:{self.current_digest}",
            "phase": phase,
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

    def test_failed_rebind_stays_armed_until_deadman_cleanup_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = HarnessFixture(Path(directory))
            try:
                fixture.provision("oak-fault-rollback")
                fixture.activate_candidate()
                fixture.runner.updater["update"] = fixture.update_state(
                    "FUNCTIONAL_HEALTHY"
                )

                def never_rebind(action: str) -> None:
                    driver = fixture.paths.usb_device / "driver"
                    if action == "unbind" and driver.exists():
                        driver.unlink()

                fixture.harness.usb_write_hook = never_rebind
                with self.assertRaisesRegex(BOARD.HarnessError, "did not recover"):
                    fixture.harness.arm_oak_fault(fixture.run_id)
                state = fixture.harness._load_state(fixture.run_id)
                self.assertTrue(state["oakFault"]["armed"])
                unit = fixture.harness._deadman_unit(fixture.run_id)
                self.assertTrue(fixture.runner.deadmen[unit])
                deadman_call = next(
                    call
                    for call in fixture.runner.calls
                    if call and call[0] == "/usr/bin/systemd-run"
                )
                self.assertIn("--property=RuntimeMaxSec=180", deadman_call)
                self.assertIn("--property=TimeoutStopSec=45", deadman_call)
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
                fixture.runner.updater["update"] = fixture.update_state("ROLLED_BACK")
                exact = fixture.harness.evidence(fixture.run_id)
                self.assertTrue(exact["complete"])
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
                "slots": {
                    "current": candidate,
                    "previous": scenario["expectedPreviousSlot"],
                    "currentVersion": release["agentVersion"],
                    "release": {
                        "schemaVersion": 2,
                        "bomDigest": scenario["expectedBomDigest"],
                        **release,
                    },
                },
                "updater": {
                    "updaterVersion": "0.2.0",
                    "update": {
                        "commandId": scenario["expectedCommandId"],
                        "targetVersion": release["agentVersion"],
                        "bomDigest": scenario["expectedBomDigest"],
                        "phase": "COMMITTED",
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

    def test_host_cleanup_accepts_only_exact_success_contract(self) -> None:
        run_id = str(uuid.uuid4())
        exact = {
            "schemaVersion": 1,
            "runId": run_id,
            "complete": True,
            "recovered": False,
            "phase": "RESTORED",
            "idempotent": True,
        }

        HOST.validate_cleanup_result(exact, run_id=run_id)
        with self.assertRaises(HOST.RunnerError):
            HOST.validate_cleanup_result({**exact, "unexpected": True}, run_id=run_id)
        with self.assertRaises(HOST.RunnerError):
            HOST.validate_cleanup_result(exact, run_id=str(uuid.uuid4()))

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
            self.assertNotIn("accept-new", options)
            with self.assertRaisesRegex(HOST.RunnerError, "duplicate JSON"):
                HOST.strict_json('{"a":1,"a":2}', label="review repro")

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
        manifest = {
            "runId": run_id,
            "scenario": {
                "type": "commit",
                "expectedCommandId": str(uuid.uuid4()),
                "expectedBomDigest": "sha256:" + "a" * 64,
            },
        }
        evidence = {
            "schemaVersion": 1,
            "protocolVersion": HOST.PROTOCOL_VERSION,
            "runId": run_id,
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
            "updater": {"updaterVersion": "0.2.0", "update": {}},
        }
        with self.assertRaisesRegex(HOST.RunnerError, "false or missing gate"):
            HOST.validate_final_evidence(evidence, manifest)

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
