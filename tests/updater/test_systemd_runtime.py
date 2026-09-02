from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nuvion_app.runtime.release_bom import (
    ReleaseTarget,
    build_release_bom_v2_payload,
    canonical_release_bom_json,
)
from nuvion_updater.health_attestation import CommitProcessIdentity
from nuvion_updater.slots import ReleaseSlotManager
from nuvion_updater.store import UpdatePhase, UpdateState
from nuvion_updater.systemd_runtime import SystemdRuntime
from nuvion_updater.trust import DeviceBinding


class SystemdRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.install_root = self.root / "opt" / "nuv-agent"
        self.slots = ReleaseSlotManager(
            self.install_root,
            require_root_owner=False,
        )
        artifact = self.root / "nuv-agent.bundle.tar"
        artifact.write_bytes(b"verified agent bundle")
        self.bom_payload = build_release_bom_v2_payload(
            bom_id="nuv-agent-0.1.116-arm64",
            release_sequence=42,
            agent_version="0.1.116",
            component_sha="a" * 40,
            config_schema="12",
            min_updater_version="0.1.0",
            targets=[
                ReleaseTarget(
                    product_model="NUVION",
                    platform_profile="rpi5_deepx_dx_m1",
                    hardware_revision="REV_A",
                    architecture="arm64",
                )
            ],
            artifact_path=artifact,
            artifact_kind="agent-bundle",
            built_at="2026-09-01T10:00:00Z",
        )
        digest = str(self.bom_payload["bomDigest"])[7:]
        self.slot = self.slots.releases_root / digest
        metadata_root = self.slot / ".nuvion"
        metadata_root.mkdir(parents=True)
        (self.slot / "bin").mkdir()
        entrypoint = self.slot / "bin" / "nuv-agent"
        entrypoint.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        entrypoint.chmod(0o755)
        (metadata_root / "release-bom.json").write_text(
            canonical_release_bom_json(self.bom_payload),
            encoding="utf-8",
        )
        self.marker = {
            "schemaVersion": 2,
            "bomDigest": self.bom_payload["bomDigest"],
            "agentVersion": self.bom_payload["agentVersion"],
            "releaseSequence": self.bom_payload["releaseSequence"],
            "artifactDigest": "sha256:" + self.bom_payload["artifact"]["sha256"],
            "componentSha": self.bom_payload["componentSha"],
            "configSchema": self.bom_payload["configSchema"],
            "publisherKeyId": "release-test",
        }
        self.marker_path = metadata_root / "release.json"
        self._write_marker(self.marker)
        (self.install_root / "current").symlink_to(f"releases/{digest}")
        self.state = UpdateState(
            command_id="00000000-0000-4000-8000-000000000001",
            sequence=5,
            compact_jws="a.b.c",
            compact_jws_sha256="b" * 64,
            target_version="0.1.116",
            bom_digest=str(self.bom_payload["bomDigest"]),
            command_expires_at="2026-09-01T10:10:00.000Z",
            phase=UpdatePhase.ACTIVATING,
            candidate_slot=str(self.slot),
            previous_slot="releases/" + "0" * 64,
            previous_version="0.1.115",
            release_sequence=42,
            artifact_digest="sha256:" + self.bom_payload["artifact"]["sha256"],
            component_sha=str(self.bom_payload["componentSha"]),
            config_schema=str(self.bom_payload["configSchema"]),
            bom_verification_status="VERIFIED",
            publisher_key_id="release-test",
            health_deadline="2026-09-01T10:02:00.000Z",
            error_code=None,
            message=None,
            created_at="2026-09-01T10:00:00.000Z",
            updated_at="2026-09-01T10:00:00.000Z",
        )
        self.binding = self._binding()
        self.runtime = SystemdRuntime(
            slots=self.slots,
            binding=self.binding,
            require_root_owner=False,
        )

    def _binding(
        self,
        *,
        profile: str = "rpi5_deepx_dx_m1",
        docker_required: bool = False,
    ) -> DeviceBinding:
        return DeviceBinding(
            trust_domain="production",
            device_id="sp-3-device-1",
            space_id=3,
            product_model="NUVION",
            platform_profile=profile,
            hardware_revision="REV_A",
            architecture="arm64",
            docker_required=docker_required,
        )

    def _write_marker(self, marker: dict[str, object]) -> None:
        self.marker_path.write_text(
            json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _completed(
        argv: tuple[str, ...],
        returncode: int = 0,
        stdout: str = "",
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, returncode, stdout, "")

    def test_restart_and_safe_stop_use_only_fixed_argv_without_shell(self) -> None:
        with mock.patch(
            "nuvion_updater.systemd_runtime.subprocess.run",
            return_value=self._completed(()),
        ) as run:
            current_slot = self.slots.current_slot()
            assert current_slot is not None
            self.runtime.restart_agent(current_slot)
            self.assertIsNone(self.runtime.safe_stop())

        self.assertEqual(
            run.call_args_list[0].args[0],
            ("/usr/bin/systemctl", "reset-failed", "nuv-agent.service"),
        )
        self.assertEqual(
            run.call_args_list[1].args[0],
            ("/usr/bin/systemctl", "restart", "nuv-agent.service"),
        )
        self.assertEqual(
            run.call_args_list[2].args[0],
            ("/usr/bin/systemctl", "stop", "nuv-agent.service"),
        )
        for call in run.call_args_list:
            self.assertIs(call.kwargs["shell"], False)
            self.assertEqual(call.kwargs["cwd"], "/")
            self.assertNotIn("command", call.kwargs["env"])
            self.assertEqual(
                call.kwargs["env"]["PATH"],
                "/usr/sbin:/usr/bin:/sbin:/bin",
            )

    def test_restart_slot_mismatch_fails_closed_without_restart(self) -> None:
        with mock.patch(
            "nuvion_updater.systemd_runtime.subprocess.run",
            return_value=self._completed(()),
        ) as run, self.assertRaisesRegex(
            RuntimeError,
            "SYSTEMD_RESTART_SLOT_MISMATCH",
        ):
            self.runtime.restart_agent("releases/" + "f" * 64)
        self.assertEqual(
            [call.args[0] for call in run.call_args_list],
            [("/usr/bin/systemctl", "stop", "nuv-agent.service")],
        )

    def test_start_limit_reset_failure_safe_stops_before_restart(self) -> None:
        failed = self._completed((), returncode=1)
        stopped = self._completed(())
        with mock.patch(
            "nuvion_updater.systemd_runtime.subprocess.run",
            side_effect=[failed, stopped],
        ) as run, self.assertRaisesRegex(RuntimeError, "SYSTEMD_RESET_FAILED"):
            current_slot = self.slots.current_slot()
            assert current_slot is not None
            self.runtime.restart_agent(current_slot)
        self.assertEqual(
            [call.args[0] for call in run.call_args_list],
            [
                ("/usr/bin/systemctl", "reset-failed", "nuv-agent.service"),
                ("/usr/bin/systemctl", "stop", "nuv-agent.service"),
            ],
        )

    def test_boot_check_matches_marker_current_slot_and_stable_main_pid_env(
        self,
    ) -> None:
        proc_root = self.root / "proc"
        process = proc_root / "412"
        process.mkdir(parents=True)
        expected_slot = self.slots.current_slot()
        assert expected_slot is not None
        (process / "environ").write_bytes(
            b"PATH=/usr/bin\0NUVION_ACTIVE_SLOT="
            + expected_slot.encode("utf-8")
            + b"\0"
        )
        completed = self._completed((), stdout="412\n")
        with (
            mock.patch("nuvion_updater.systemd_runtime.PROC_ROOT", proc_root),
            mock.patch(
                "nuvion_updater.systemd_runtime.subprocess.run",
                side_effect=[completed, completed],
            ) as run,
        ):
            self.assertEqual(
                self.runtime.boot_health_check(self.state),
                (True, "BOOT_HEALTHY"),
            )

        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            self.assertEqual(
                call.args[0],
                (
                    "/usr/bin/systemctl",
                    "show",
                    "--property=MainPID",
                    "--value",
                    "nuv-agent.service",
                ),
            )

    def test_boot_marker_mismatch_is_rejected_and_safe_stopped(self) -> None:
        for field, replacement in (
            ("agentVersion", "9.9.9"),
            ("bomDigest", "sha256:" + "f" * 64),
            ("artifactDigest", "sha256:" + "e" * 64),
            ("componentSha", "d" * 40),
            ("releaseSequence", 43),
        ):
            with self.subTest(field=field):
                tampered = dict(self.marker)
                tampered[field] = replacement
                self._write_marker(tampered)
                with mock.patch(
                    "nuvion_updater.systemd_runtime.subprocess.run",
                    return_value=self._completed(()),
                ) as run:
                    healthy, detail = self.runtime.boot_health_check(self.state)
                self.assertFalse(healthy)
                self.assertEqual(detail, "BOOT_RELEASE_MARKER_MISMATCH")
                self.assertEqual(
                    run.call_args.args[0],
                    ("/usr/bin/systemctl", "stop", "nuv-agent.service"),
                )
        self._write_marker(self.marker)

    def test_boot_env_must_have_one_exact_current_slot_and_stable_pid(self) -> None:
        proc_root = self.root / "proc"
        process = proc_root / "412"
        process.mkdir(parents=True)
        (process / "environ").write_bytes(
            b"NUVION_ACTIVE_SLOT=releases/not-the-current-slot\0"
        )
        show = self._completed((), stdout="412\n")
        stop = self._completed(())
        with (
            mock.patch("nuvion_updater.systemd_runtime.PROC_ROOT", proc_root),
            mock.patch(
                "nuvion_updater.systemd_runtime.subprocess.run",
                side_effect=[show, show, stop],
            ) as run,
        ):
            self.assertEqual(
                self.runtime.boot_health_check(self.state),
                (False, "BOOT_ACTIVE_SLOT_ENV_MISMATCH"),
            )
        self.assertEqual(
            run.call_args_list[-1].args[0],
            ("/usr/bin/systemctl", "stop", "nuv-agent.service"),
        )

    def test_commit_gate_binds_peer_main_pid_start_ticks_boot_and_slot(self) -> None:
        proc_root = self.root / "proc"
        process = proc_root / "412"
        process.mkdir(parents=True)
        expected_slot = self.slots.current_slot()
        assert expected_slot is not None
        (process / "environ").write_bytes(
            b"NUVION_ACTIVE_SLOT=" + expected_slot.encode("utf-8") + b"\0"
        )
        stat_suffix = ["S", *(["1"] * 18), "987654", "0", "0"]
        (process / "stat").write_text(
            f"412 (nuv-agent worker) {' '.join(stat_suffix)}\n",
            encoding="ascii",
        )
        boot_id = "00000000-0000-4000-8000-000000000123"
        boot_path = proc_root / "sys" / "kernel" / "random"
        boot_path.mkdir(parents=True)
        (boot_path / "boot_id").write_text(boot_id + "\n", encoding="ascii")
        completed = self._completed((), stdout="412\n")
        with (
            mock.patch("nuvion_updater.systemd_runtime.PROC_ROOT", proc_root),
            mock.patch(
                "nuvion_updater.systemd_runtime.subprocess.run",
                side_effect=[completed, completed],
            ),
        ):
            identity = self.runtime.commit_process_identity(self.state, 412)
        self.assertEqual(
            identity,
            CommitProcessIdentity(
                pid=412,
                start_ticks=987654,
                boot_id=boot_id,
                active_slot=expected_slot,
            ),
        )

    def test_commit_gate_rejects_non_main_peer_and_pid_reuse(self) -> None:
        completed = self._completed((), stdout="412\n")
        with mock.patch(
            "nuvion_updater.systemd_runtime.subprocess.run",
            return_value=completed,
        ), self.assertRaisesRegex(RuntimeError, "COMMIT_PEER_MAIN_PID_MISMATCH"):
            self.runtime.commit_process_identity(self.state, 999)

        expected_slot = self.slots.current_slot()
        assert expected_slot is not None
        with (
            mock.patch.object(self.runtime, "_main_pid", return_value=412),
            mock.patch.object(
                self.runtime, "_process_start_ticks", side_effect=[100, 101]
            ),
            mock.patch.object(
                self.runtime,
                "_active_slot_from_environ",
                return_value=expected_slot,
            ),
            mock.patch.object(
                self.runtime,
                "_boot_id",
                return_value="00000000-0000-4000-8000-000000000123",
            ),
            self.assertRaisesRegex(RuntimeError, "COMMIT_PROCESS_INSTANCE_CHANGED"),
        ):
            self.runtime.commit_process_identity(self.state, 412)

    def test_rollback_boot_check_requires_exact_running_restored_slot(self) -> None:
        proc_root = self.root / "proc"
        process = proc_root / "512"
        process.mkdir(parents=True)
        expected_slot = self.slots.current_slot()
        assert expected_slot is not None
        (process / "environ").write_bytes(
            b"NUVION_ACTIVE_SLOT=" + expected_slot.encode("utf-8") + b"\0"
        )
        show = self._completed((), stdout="512\n")
        with (
            mock.patch("nuvion_updater.systemd_runtime.PROC_ROOT", proc_root),
            mock.patch(
                "nuvion_updater.systemd_runtime.subprocess.run",
                side_effect=[show, show],
            ),
        ):
            self.assertEqual(
                self.runtime.rollback_boot_health_check(expected_slot),
                (True, "ROLLBACK_BOOT_HEALTHY"),
            )

    def test_non_docker_functional_check_runs_fixed_current_agent_doctor(self) -> None:
        with mock.patch(
            "nuvion_updater.systemd_runtime.subprocess.run",
            return_value=self._completed(()),
        ) as run:
            self.assertEqual(
                self.runtime.functional_health_check(self.state),
                (True, "FUNCTIONAL_HEALTHY"),
            )
        self.assertEqual(
            run.call_args.args[0],
            (
                "/usr/sbin/runuser",
                "-u",
                "nuvion",
                "--",
                "/usr/bin/nuv-agent",
                "doctor",
                "--hardware",
            ),
        )

    def test_failed_hardware_doctor_safe_stops_service(self) -> None:
        failed = self._completed((), returncode=2)
        stopped = self._completed(())
        with mock.patch(
            "nuvion_updater.systemd_runtime.subprocess.run",
            side_effect=[failed, stopped],
        ) as run:
            self.assertEqual(
                self.runtime.functional_health_check(self.state),
                (False, "FUNCTIONAL_HARDWARE_DOCTOR_FAILED"),
            )
        self.assertEqual(
            run.call_args_list[-1].args[0],
            ("/usr/bin/systemctl", "stop", "nuv-agent.service"),
        )

    def test_iq9075_stops_runs_root_fixed_oak_probe_then_restarts(self) -> None:
        probe_dir = self.root / "usr" / "lib" / "nuvion-updater"
        probe_dir.mkdir(parents=True, mode=0o755)
        probe = probe_dir / "test-iq9075.sh"
        probe.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        probe.chmod(0o755)
        runtime = SystemdRuntime(
            slots=self.slots,
            binding=self._binding(profile="iq9075_dev"),
            require_root_owner=False,
        )
        with (
            mock.patch("nuvion_updater.systemd_runtime.IQ9075_PROBE", str(probe)),
            mock.patch(
                "nuvion_updater.systemd_runtime.subprocess.run",
                return_value=self._completed(()),
            ) as run,
            mock.patch.object(
                runtime,
                "boot_health_check",
                return_value=(True, "BOOT_HEALTHY"),
            ) as boot_check,
        ):
            self.assertEqual(
                runtime.functional_health_check(self.state),
                (True, "FUNCTIONAL_HEALTHY"),
            )
        self.assertEqual(
            [call.args[0] for call in run.call_args_list],
            [
                ("/usr/bin/systemctl", "stop", "nuv-agent.service"),
                ("/usr/bin/bash", str(probe), "--camera", "oak"),
                ("/usr/bin/systemctl", "reset-failed", "nuv-agent.service"),
                ("/usr/bin/systemctl", "restart", "nuv-agent.service"),
            ],
        )
        boot_check.assert_called_once_with(self.state)

    def test_iq9075_probe_failure_remains_safe_stopped(self) -> None:
        probe_dir = self.root / "usr" / "lib" / "nuvion-updater"
        probe_dir.mkdir(parents=True, mode=0o755)
        probe = probe_dir / "test-iq9075.sh"
        probe.write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
        probe.chmod(0o755)
        runtime = SystemdRuntime(
            slots=self.slots,
            binding=self._binding(profile="iq9075_dev"),
            require_root_owner=False,
        )
        ok = self._completed(())
        failed = self._completed((), returncode=1)
        with (
            mock.patch("nuvion_updater.systemd_runtime.IQ9075_PROBE", str(probe)),
            mock.patch(
                "nuvion_updater.systemd_runtime.subprocess.run",
                side_effect=[ok, failed, ok],
            ) as run,
        ):
            self.assertEqual(
                runtime.functional_health_check(self.state),
                (False, "FUNCTIONAL_IQ9075_PROBE_FAILED"),
            )
        self.assertEqual(
            [call.args[0] for call in run.call_args_list],
            [
                ("/usr/bin/systemctl", "stop", "nuv-agent.service"),
                ("/usr/bin/bash", str(probe), "--camera", "oak"),
                ("/usr/bin/systemctl", "stop", "nuv-agent.service"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
