from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any

from nuvion_app.runtime.release_bom import (
    ReleaseBomValidationError,
    load_release_bom,
)
from nuvion_updater.secure_io import read_fixed_regular_file
from nuvion_updater.slots import SLOT_MARKER, ReleaseSlotManager
from nuvion_updater.store import UpdateState
from nuvion_updater.trust import DeviceBinding

AGENT_SERVICE = "nuv-agent.service"
SYSTEMCTL = "/usr/bin/systemctl"
CURRENT_AGENT = "/usr/bin/nuv-agent"
IQ9075_PROBE = "/usr/lib/nuvion-updater/test-iq9075.sh"
BASH = "/usr/bin/bash"
RUNUSER = "/usr/sbin/runuser"
PROC_ROOT = Path("/proc")

SYSTEMCTL_TIMEOUT_SECONDS = 30
DOCTOR_TIMEOUT_SECONDS = 120
IQ9075_PROBE_TIMEOUT_SECONDS = 300
MAX_ENVIRON_BYTES = 1024 * 1024
MAX_MARKER_BYTES = 64 * 1024

_COMMAND_ENV = {
    "LANG": "C",
    "LC_ALL": "C",
    # The updater runs as root. Never resolve probe utilities from locally
    # managed prefixes that may be writable outside the OS package boundary.
    "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
    "PYTHONNOUSERSITE": "1",
    "NUV_AGENT_CONFIG": "/etc/nuv-agent/agent.env",
}


class SystemdRuntime:
    """Fail-closed production callbacks for the privileged release updater."""

    def __init__(
        self,
        *,
        slots: ReleaseSlotManager,
        binding: DeviceBinding,
        require_root_owner: bool = True,
    ) -> None:
        self.slots = slots
        self.binding = binding
        self.require_root_owner = require_root_owner

    def restart_agent(self, expected_slot: str) -> None:
        if self.slots.current_slot() != expected_slot:
            self._stop_after_failure()
            raise RuntimeError("SYSTEMD_RESTART_SLOT_MISMATCH")
        self._reset_agent_start_limit()
        result = self._run(
            (SYSTEMCTL, "restart", AGENT_SERVICE),
            timeout=SYSTEMCTL_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            self._stop_after_failure()
            raise RuntimeError("SYSTEMD_RESTART_FAILED")

    def safe_stop(self) -> None:
        try:
            result = self._run(
                (SYSTEMCTL, "stop", AGENT_SERVICE),
                timeout=SYSTEMCTL_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError("SYSTEMD_SAFE_STOP_FAILED") from exc
        if result.returncode != 0:
            raise RuntimeError("SYSTEMD_SAFE_STOP_FAILED")

    def boot_health_check(self, state: UpdateState) -> tuple[bool, str]:
        try:
            expected_slot = self._validate_staged_state(state)
            first_pid = self._main_pid()
            active_slot = self._active_slot_from_environ(first_pid)
            second_pid = self._main_pid()
            if second_pid != first_pid:
                raise RuntimeError("BOOT_MAIN_PID_CHANGED")
            if active_slot != expected_slot:
                raise RuntimeError("BOOT_ACTIVE_SLOT_ENV_MISMATCH")
        except (
            OSError,
            TypeError,
            ValueError,
            RuntimeError,
            ReleaseBomValidationError,
        ) as exc:
            return self._failed_check(self._reason(exc, "BOOT_HEALTH_FAILED"))
        return True, "BOOT_HEALTHY"

    def rollback_boot_health_check(self, expected_slot: str) -> tuple[bool, str]:
        try:
            if self.slots.current_slot() != expected_slot:
                raise RuntimeError("ROLLBACK_ACTIVE_SLOT_MISMATCH")
            first_pid = self._main_pid()
            active_slot = self._active_slot_from_environ(first_pid)
            if self._main_pid() != first_pid:
                raise RuntimeError("ROLLBACK_MAIN_PID_CHANGED")
            if active_slot != expected_slot:
                raise RuntimeError("ROLLBACK_ACTIVE_SLOT_ENV_MISMATCH")
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            return self._failed_check(self._reason(exc, "ROLLBACK_BOOT_FAILED"))
        return True, "ROLLBACK_BOOT_HEALTHY"

    def functional_health_check(self, state: UpdateState) -> tuple[bool, str]:
        try:
            if state.candidate_slot is None or not self.slots.is_active(
                state.candidate_slot
            ):
                raise RuntimeError("FUNCTIONAL_ACTIVE_SLOT_MISMATCH")
            if self.binding.docker_required:
                raise RuntimeError("FUNCTIONAL_DOCKER_PROFILE_UNSUPPORTED")
            if self.binding.platform_profile == "iq9075_dev":
                healthy, detail = self._iq9075_functional_check()
                if not healthy:
                    return healthy, detail
                # The probe intentionally stops the candidate to obtain
                # exclusive OAK access. Prove the restarted service is still
                # the exact staged slot before accepting functional health.
                boot_healthy, boot_detail = self.boot_health_check(state)
                if not boot_healthy:
                    return False, boot_detail
                return True, "FUNCTIONAL_HEALTHY"
            result = self._run(
                (
                    RUNUSER,
                    "-u",
                    "nuvion",
                    "--",
                    CURRENT_AGENT,
                    "doctor",
                    "--hardware",
                ),
                timeout=DOCTOR_TIMEOUT_SECONDS,
            )
            if result.returncode != 0:
                raise RuntimeError("FUNCTIONAL_HARDWARE_DOCTOR_FAILED")
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            return self._failed_check(self._reason(exc, "FUNCTIONAL_HEALTH_FAILED"))
        return True, "FUNCTIONAL_HEALTHY"

    def _iq9075_functional_check(self) -> tuple[bool, str]:
        try:
            self._validate_fixed_probe()
            stop_result = self._run(
                (SYSTEMCTL, "stop", AGENT_SERVICE),
                timeout=SYSTEMCTL_TIMEOUT_SECONDS,
            )
            if stop_result.returncode != 0:
                raise RuntimeError("FUNCTIONAL_SERVICE_STOP_FAILED")
            probe_result = self._run(
                (BASH, IQ9075_PROBE, "--camera", "oak"),
                timeout=IQ9075_PROBE_TIMEOUT_SECONDS,
            )
            if probe_result.returncode != 0:
                raise RuntimeError("FUNCTIONAL_IQ9075_PROBE_FAILED")
            self._reset_agent_start_limit()
            restart_result = self._run(
                (SYSTEMCTL, "restart", AGENT_SERVICE),
                timeout=SYSTEMCTL_TIMEOUT_SECONDS,
            )
            if restart_result.returncode != 0:
                raise RuntimeError("FUNCTIONAL_SERVICE_RESTART_FAILED")
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            return self._failed_check(
                self._reason(exc, "FUNCTIONAL_IQ9075_PROBE_FAILED")
            )
        return True, "FUNCTIONAL_HEALTHY"

    def _reset_agent_start_limit(self) -> None:
        result = self._run(
            (SYSTEMCTL, "reset-failed", AGENT_SERVICE),
            timeout=SYSTEMCTL_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            self._stop_after_failure()
            raise RuntimeError("SYSTEMD_RESET_FAILED")

    def _validate_staged_state(self, state: UpdateState) -> str:
        if state.candidate_slot is None:
            raise RuntimeError("BOOT_CANDIDATE_SLOT_MISSING")
        if state.release_sequence is None:
            raise RuntimeError("BOOT_RELEASE_SEQUENCE_MISSING")
        if (
            state.artifact_digest is None
            or state.component_sha is None
            or state.config_schema is None
            or state.publisher_key_id is None
            or state.bom_verification_status != "VERIFIED"
        ):
            raise RuntimeError("BOOT_STAGED_IDENTITY_MISSING")
        if not self.slots.is_active(state.candidate_slot):
            raise RuntimeError("BOOT_ACTIVE_SLOT_MISMATCH")
        current_slot = self.slots.current_slot()
        if current_slot is None:
            raise RuntimeError("BOOT_CURRENT_SLOT_MISSING")

        slot = Path(state.candidate_slot).resolve(strict=True)
        metadata_root = slot / ".nuvion"
        marker = self._read_marker(metadata_root / Path(SLOT_MARKER).name)
        bom = load_release_bom(
            metadata_root / "release-bom.json",
            expected_bom_digest=state.bom_digest,
        )
        if bom.schema_version != 2:
            raise RuntimeError("BOOT_BOM_SCHEMA_MISMATCH")
        expected_marker: dict[str, object] = {
            "schemaVersion": 2,
            "bomDigest": state.bom_digest,
            "agentVersion": state.target_version,
            "releaseSequence": state.release_sequence,
            "artifactDigest": state.artifact_digest,
            "componentSha": state.component_sha,
            "configSchema": state.config_schema,
            "publisherKeyId": state.publisher_key_id,
        }
        if marker != expected_marker:
            raise RuntimeError("BOOT_RELEASE_MARKER_MISMATCH")
        if (
            bom.agent_version != state.target_version
            or bom.release_sequence != state.release_sequence
            or f"sha256:{bom.artifact_sha256}" != state.artifact_digest
            or bom.component_sha != state.component_sha
            or bom.config_schema != state.config_schema
        ):
            raise RuntimeError("BOOT_STAGED_STATE_MISMATCH")
        return current_slot

    def _main_pid(self) -> int:
        result = self._run(
            (SYSTEMCTL, "show", "--property=MainPID", "--value", AGENT_SERVICE),
            timeout=SYSTEMCTL_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            raise RuntimeError("BOOT_MAIN_PID_UNAVAILABLE")
        raw_pid = result.stdout.strip()
        if not raw_pid.isascii() or not raw_pid.isdecimal() or raw_pid == "0":
            raise RuntimeError("BOOT_MAIN_PID_UNAVAILABLE")
        pid = int(raw_pid, 10)
        if pid < 1 or pid > 2_147_483_647:
            raise RuntimeError("BOOT_MAIN_PID_UNAVAILABLE")
        return pid

    @staticmethod
    def _active_slot_from_environ(pid: int) -> str:
        environ_path = PROC_ROOT / str(pid) / "environ"
        descriptor = os.open(
            environ_path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(
                    descriptor,
                    min(64 * 1024, MAX_ENVIRON_BYTES + 1 - total),
                )
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > MAX_ENVIRON_BYTES:
                    raise RuntimeError("BOOT_PROCESS_ENV_TOO_LARGE")
        finally:
            os.close(descriptor)
        prefix = b"NUVION_ACTIVE_SLOT="
        matches = [entry[len(prefix) :] for entry in b"".join(chunks).split(b"\0") if entry.startswith(prefix)]
        if len(matches) != 1:
            raise RuntimeError("BOOT_ACTIVE_SLOT_ENV_MISMATCH")
        try:
            value = matches[0].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError("BOOT_ACTIVE_SLOT_ENV_MISMATCH") from exc
        if not value or value != value.strip():
            raise RuntimeError("BOOT_ACTIVE_SLOT_ENV_MISMATCH")
        return value

    def _validate_fixed_probe(self) -> None:
        probe = Path(IQ9075_PROBE)
        parent_metadata = probe.parent.lstat()
        metadata = probe.lstat()
        if (
            stat.S_ISLNK(parent_metadata.st_mode)
            or not stat.S_ISDIR(parent_metadata.st_mode)
            or parent_metadata.st_mode & 0o022
        ):
            raise RuntimeError("FUNCTIONAL_UNSAFE_IQ9075_PROBE")
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_mode & 0o022
        ):
            raise RuntimeError("FUNCTIONAL_UNSAFE_IQ9075_PROBE")
        if self.require_root_owner and (
            parent_metadata.st_uid != 0 or metadata.st_uid != 0
        ):
            raise RuntimeError("FUNCTIONAL_UNSAFE_IQ9075_PROBE")

    def _read_marker(self, path: Path) -> dict[str, Any]:
        raw = read_fixed_regular_file(
            path.parent,
            path.name,
            max_bytes=MAX_MARKER_BYTES,
            require_root_owner=self.require_root_owner,
        )

        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate marker field")
                result[key] = value
            return result

        try:
            payload = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object)
        except (UnicodeDecodeError, ValueError, RecursionError) as exc:
            raise RuntimeError("BOOT_RELEASE_MARKER_INVALID") from exc
        if not isinstance(payload, dict):
            raise TypeError("BOOT_RELEASE_MARKER_INVALID")
        return payload

    def _failed_check(self, reason: str) -> tuple[bool, str]:
        try:
            self.safe_stop()
        except RuntimeError:
            return False, f"{reason};SYSTEMD_SAFE_STOP_FAILED"
        return False, reason

    def _stop_after_failure(self) -> None:
        try:
            self.safe_stop()
        except RuntimeError:
            pass

    @staticmethod
    def _reason(exc: BaseException, fallback: str) -> str:
        if isinstance(exc, RuntimeError):
            value = str(exc).strip()
            if value and len(value) <= 100 and value.replace("_", "").isalnum():
                return value
        return fallback

    @staticmethod
    def _run(
        argv: tuple[str, ...],
        *,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            argv,
            check=False,
            shell=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="/",
            env=_COMMAND_ENV,
        )
