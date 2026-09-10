from __future__ import annotations

import base64
import hashlib
import json
import tempfile
import unittest
import uuid
from pathlib import Path

from nuvion_app.inference.device_mode import (
    DeviceModeReconciler,
    DeviceModeStore,
    rollback_failed_mode_startup,
)
from nuvion_app.inference.fleet_command import VerifiedFleetCommand
from nuvion_app.inference.settings_reconciler import AtomicSettingsStore
from nuvion_app.runtime.settings_overlay import parse_settings_overlay


def _command(mode: str = "DEMO", revision: int = 3) -> VerifiedFleetCommand:
    payload: dict[str, object] = {"modeRevision": revision, "mode": mode}
    if mode == "DEMO":
        payload.update(
            {
                "profileId": "metal-nut-showcase-v1",
                "profileDigest": "sha256:" + "a" * 64,
            }
        )
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return VerifiedFleetCommand(
        command_id=str(uuid.uuid4()),
        device_id="device-1",
        space_id=1,
        command_type="DEVICE_MODE_SET",
        schema_version=1,
        issued_at="2026-09-10T00:00:00Z",
        expires_at="2026-09-10T01:00:00Z",
        sequence=revision,
        payload_base64=base64.urlsafe_b64encode(encoded).decode().rstrip("="),
        payload_hash=hashlib.sha256(encoded).hexdigest(),
        payload=payload,
        actor="admin@example.com",
        authorization_context="PLATFORM_ADMIN",
        key_id="test",
        required_capability="command.device.mode.set",
        compact_jws=f"header.{revision}.signature",
    )


class _Runtime:
    def __init__(self) -> None:
        self.state: dict[str, object] = {
            "effectiveMode": "PRODUCTION",
            "inputSource": "CAMERA",
            "modeRevision": 2,
            "profileId": None,
            "profileDigest": None,
            "sessionId": None,
            "frameReady": True,
            "inferenceReady": True,
            "health": "FUNCTIONAL_HEALTHY",
        }
        self.preflight_error: Exception | None = None
        self.pending = False

    def preflight(self, _desired) -> None:
        if self.preflight_error:
            raise self.preflight_error

    def snapshot(self) -> dict[str, object]:
        return dict(self.state)

    def startup_pending(self) -> bool:
        return self.pending


class DeviceModeReconcilerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        root = Path(self.tempdir.name)
        self.base = root / "agent.env"
        self.base.write_text("NUVION_DEVICE_PASSWORD=secret\n")
        self.store = DeviceModeStore(AtomicSettingsStore(self.base, root / "settings"))

    def test_stages_mode_without_copying_base_secrets_and_commits_after_restart(self) -> None:
        command = _command()
        runtime = _Runtime()
        first = DeviceModeReconciler(
            store=self.store, runtime=runtime, process_instance_id="process-a"
        ).reconcile(command)

        self.assertEqual(first.checkpoint["markerPhase"], "ACTIVATED")
        overlay = parse_settings_overlay(self.store.active_path.read_text())
        self.assertEqual(overlay["NUVION_DEMO_MODE"], "true")
        self.assertNotIn("NUVION_DEVICE_PASSWORD", overlay)
        session_id = overlay["NUVION_DEMO_SESSION_ID"]

        runtime.state.update(
            {
                "effectiveMode": "DEMO",
                "inputSource": "DATASET",
                "modeRevision": 3,
                "profileId": command.payload["profileId"],
                "profileDigest": command.payload["profileDigest"],
                "sessionId": session_id,
            }
        )
        second = DeviceModeReconciler(
            store=self.store, runtime=runtime, process_instance_id="process-b"
        ).reconcile(command)

        self.assertEqual(second.status, "SUCCEEDED")
        self.assertEqual(second.reported_state["transitionState"], "ACTIVE")
        self.assertEqual(second.reported_state["sessionId"], session_id)
        self.assertEqual(self.store.marker()["phase"], "COMMITTED")

    def test_preflight_failure_does_not_change_active_overlay(self) -> None:
        self.store.active_path.write_text("NUVION_DEMO_MODE=false\n")
        runtime = _Runtime()
        runtime.preflight_error = ValueError("profile unavailable")

        outcome = DeviceModeReconciler(
            store=self.store, runtime=runtime, process_instance_id="process-a"
        ).reconcile(_command())

        self.assertEqual(outcome.status, "FAILED")
        self.assertEqual(outcome.code, "DEVICE_MODE_PREFLIGHT_FAILED")
        self.assertEqual(self.store.active_path.read_text(), "NUVION_DEMO_MODE=false\n")

    def test_unhealthy_new_mode_rolls_back_then_requires_one_recovery_restart(self) -> None:
        self.store.active_path.write_text("NUVION_DEMO_MODE=false\n")
        command = _command()
        runtime = _Runtime()
        DeviceModeReconciler(
            store=self.store, runtime=runtime, process_instance_id="process-a"
        ).reconcile(command)

        runtime.state.update({"health": "UNHEALTHY", "frameReady": False})
        outcome = DeviceModeReconciler(
            store=self.store,
            runtime=runtime,
            process_instance_id="process-b",
            startup_clock=lambda: 1000.0,
        ).reconcile(command)

        self.assertEqual(outcome.checkpoint["markerPhase"], "ROLLBACK_STAGED")
        self.assertEqual(self.store.active_path.read_text(), "NUVION_DEMO_MODE=false\n")

    def test_failed_startup_guard_restores_predecessor_before_crash_loop(self) -> None:
        self.store.active_path.write_text("NUVION_DEMO_MODE=false\n")
        command = _command()
        runtime = _Runtime()
        DeviceModeReconciler(
            store=self.store, runtime=runtime, process_instance_id="process-a"
        ).reconcile(command)

        restored = rollback_failed_mode_startup(
            self.store.settings_store,
            process_instance_id="failed-process",
        )

        self.assertTrue(restored)
        self.assertEqual(self.store.active_path.read_text(), "NUVION_DEMO_MODE=false\n")
        self.assertEqual(self.store.marker()["phase"], "ROLLBACK_STAGED")


if __name__ == "__main__":
    unittest.main()
