from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from nuvion_app.inference.command_inbox import (
    COMMAND_STATUS_FAILED,
    COMMAND_STATUS_ROLLED_BACK,
    CommandEffectOutcome,
)
from nuvion_app.inference.demo_mvtec import (
    MANAGED_DEMO_PROFILE_DIGEST,
    MANAGED_DEMO_PROFILE_ID,
)
from nuvion_app.inference.effect_reconciler import ReconcileDeferred
from nuvion_app.inference.fleet_command import VerifiedFleetCommand
from nuvion_app.inference.settings_reconciler import AtomicSettingsStore
from nuvion_app.runtime.settings_overlay import (
    parse_settings_overlay,
    serialize_settings_overlay,
)

DEVICE_MODE_COMMAND_TYPE = "DEVICE_MODE_SET"
DEVICE_MODE_CAPABILITY = "command.device.mode.set"
_MODE_SETTING_KEYS = frozenset(
    {
        "NUVION_DEMO_MODE",
        "NUVION_DEMO_PROFILE_ID",
        "NUVION_DEMO_PROFILE_DIGEST",
        "NUVION_DEMO_MODE_REVISION",
        "NUVION_DEMO_SESSION_ID",
    }
)


class DeviceModeRuntime(Protocol):
    def preflight(self, desired: Mapping[str, Any]) -> None: ...

    def snapshot(self) -> dict[str, Any]: ...

    def startup_pending(self) -> bool: ...


def device_mode_env_updates(
    payload: Mapping[str, Any], *, session_id: str | None
) -> dict[str, str]:
    mode = str(payload["mode"])
    updates = {
        "NUVION_DEMO_MODE": "true" if mode == "DEMO" else "false",
        "NUVION_DEMO_MODE_REVISION": str(int(payload["modeRevision"])),
    }
    if mode == "DEMO":
        if session_id is None:
            raise ValueError("DEMO activation requires a session id")
        updates.update(
            {
                "NUVION_DEMO_PROFILE_ID": str(payload["profileId"]),
                "NUVION_DEMO_PROFILE_DIGEST": str(payload["profileDigest"]),
                "NUVION_DEMO_SESSION_ID": session_id,
            }
        )
    return updates


class DeviceModeStore:
    """Atomically switches mode fields in the shared dynamic settings overlay."""

    def __init__(self, settings_store: AtomicSettingsStore) -> None:
        self.settings_store = settings_store
        self.state_dir = settings_store.state_dir
        self.active_path = settings_store.active_path
        self.candidate_path = self.state_dir / "device-mode-candidate.env"
        self.lkg_path = self.state_dir / "device-mode-lkg.env"
        self.marker_path = self.state_dir / "device-mode-restart-marker.json"

    def _read(self, path: Path, *, missing: bytes | None = None) -> bytes:
        return self.settings_store._read_bytes(path, missing=missing)

    def _write(self, path: Path, content: bytes) -> None:
        self.settings_store._atomic_write(path, content)

    def marker(self) -> dict[str, Any] | None:
        try:
            raw = self._read(self.marker_path)
        except FileNotFoundError:
            return None
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("device mode restart marker is corrupt") from exc
        if not isinstance(value, dict):
            raise ValueError("device mode restart marker must be an object")
        return value

    def _write_marker(self, marker: Mapping[str, Any]) -> dict[str, Any]:
        normalized = dict(marker)
        self._write(
            self.marker_path,
            json.dumps(normalized, separators=(",", ":"), sort_keys=True).encode(),
        )
        return normalized

    def stage_and_activate(
        self,
        *,
        command: VerifiedFleetCommand,
        process_instance_id: str,
        fence_check: Callable[[], None],
    ) -> dict[str, Any]:
        existing = self.marker()
        if existing and existing.get("commandId") == command.command_id:
            return existing
        current = self._read(self.active_path, missing=b"")
        values = parse_settings_overlay(current.decode()) if current else {}
        for key in _MODE_SETTING_KEYS:
            values.pop(key, None)
        session_id = str(uuid.uuid4()) if command.payload["mode"] == "DEMO" else None
        values.update(device_mode_env_updates(command.payload, session_id=session_id))
        candidate = serialize_settings_overlay(values)
        marker = {
            "schemaVersion": 1,
            "commandId": command.command_id,
            "mode": command.payload["mode"],
            "modeRevision": command.payload["modeRevision"],
            "profileId": command.payload.get("profileId"),
            "profileDigest": command.payload.get("profileDigest"),
            "sessionId": session_id,
            "phase": "PREPARED",
            "stagedProcessInstanceId": process_instance_id,
            "candidateSha256": self.settings_store._sha256(candidate),
            "lkgSha256": self.settings_store._sha256(current),
        }
        fence_check()
        self._write(self.lkg_path, current)
        fence_check()
        self._write(self.candidate_path, candidate)
        fence_check()
        self._write_marker(marker)
        fence_check()
        os.replace(self.candidate_path, self.active_path)
        self.settings_store._fsync_directory(self.state_dir)
        marker["phase"] = "ACTIVATED"
        return self._write_marker(marker)

    def readback_matches(self, payload: Mapping[str, Any], session_id: str | None) -> bool:
        active = self._read(self.active_path, missing=b"")
        values = parse_settings_overlay(active.decode()) if active else {}
        expected = device_mode_env_updates(payload, session_id=session_id)
        if any(values.get(key) != value for key, value in expected.items()):
            return False
        if payload["mode"] == "PRODUCTION":
            return all(key not in values for key in _MODE_SETTING_KEYS - set(expected))
        return True

    def rollback(self, *, process_instance_id: str) -> dict[str, Any]:
        lkg = self._read(self.lkg_path)
        self._write(self.active_path, lkg)
        marker = dict(self.marker() or {})
        marker.update(
            {
                "phase": "ROLLBACK_STAGED",
                "rollbackProcessInstanceId": process_instance_id,
            }
        )
        return self._write_marker(marker)

    def lkg_is_active(self) -> bool:
        try:
            return self.settings_store._sha256(
                self._read(self.active_path, missing=b"")
            ) == self.settings_store._sha256(self._read(self.lkg_path))
        except OSError:
            return False

    def commit(self, *, phase: str = "COMMITTED") -> None:
        marker = dict(self.marker() or {})
        marker["phase"] = phase
        self._write_marker(marker)


class DeviceModeReconciler:
    command_type = DEVICE_MODE_COMMAND_TYPE
    capability = DEVICE_MODE_CAPABILITY

    def __init__(
        self,
        *,
        store: DeviceModeStore,
        runtime: DeviceModeRuntime,
        process_instance_id: str | None = None,
        startup_clock: Callable[[], float] | None = None,
    ) -> None:
        import time

        self.store = store
        self.runtime = runtime
        self.process_instance_id = str(process_instance_id or uuid.uuid4())
        self._effect_fence: Callable[[], None] = lambda: None
        self._clock = startup_clock or time.monotonic
        self._started_at = self._clock()

    def ready(self) -> bool:
        return True

    def set_effect_fence(self, fence_check: Callable[[], None]) -> None:
        self._effect_fence = fence_check

    def reconcile(
        self, command: VerifiedFleetCommand
    ) -> CommandEffectOutcome | ReconcileDeferred:
        try:
            marker = self.store.marker()
        except ValueError as exc:
            return CommandEffectOutcome(
                status=COMMAND_STATUS_FAILED,
                code="DEVICE_MODE_MARKER_INVALID",
                message=str(exc),
                reported_state=self._reported(command, None, "FAILED", "NOT_APPLIED"),
            )

        if marker and marker.get("commandId") == command.command_id:
            phase = marker.get("phase")
            if phase == "ROLLBACK_STAGED":
                if marker.get("rollbackProcessInstanceId") == self.process_instance_id:
                    return self._deferred(command, marker, "ROLLBACK_RESTART_REQUIRED")
                snapshot = self.runtime.snapshot()
                if self.store.lkg_is_active() and snapshot.get("health") == "FUNCTIONAL_HEALTHY":
                    self._effect_fence()
                    self.store.commit(phase="ROLLED_BACK")
                    return CommandEffectOutcome(
                        status=COMMAND_STATUS_ROLLED_BACK,
                        code="DEVICE_MODE_HEALTH_ROLLBACK",
                        message="previous device mode restored after failed transition",
                        reported_state=self._reported(
                            command, snapshot, "ROLLED_BACK", "LKG_RESTORED"
                        ),
                    )
                return CommandEffectOutcome(
                    status=COMMAND_STATUS_FAILED,
                    code="DEVICE_MODE_ROLLBACK_FAILED",
                    message="previous device mode did not recover functional health",
                    reported_state=self._reported(
                        command, snapshot, "FAILED", "LKG_UNHEALTHY"
                    ),
                )
            if phase == "ACTIVATED" and marker.get("stagedProcessInstanceId") != self.process_instance_id:
                snapshot = self.runtime.snapshot()
                if self._matches(command, marker, snapshot):
                    self._effect_fence()
                    self.store.commit()
                    return CommandEffectOutcome.succeeded(
                        self._reported(command, snapshot, "ACTIVE", "FUNCTIONAL_HEALTHY")
                    )
                if self._startup_pending():
                    return self._deferred(
                        command,
                        marker,
                        "STARTUP_PENDING",
                        retry_without_restart=True,
                    )
                self._effect_fence()
                rollback = self.store.rollback(
                    process_instance_id=self.process_instance_id
                )
                return self._deferred(command, rollback, "ROLLBACK_RESTART_REQUIRED")

        try:
            self._effect_fence()
            self.runtime.preflight(command.payload)
        except (OSError, RuntimeError, ValueError) as exc:
            return CommandEffectOutcome(
                status=COMMAND_STATUS_FAILED,
                code="DEVICE_MODE_PREFLIGHT_FAILED",
                message=str(exc)[:1000],
                reported_state=self._reported(
                    command, self.runtime.snapshot(), "FAILED", "NOT_APPLIED"
                ),
            )
        self._effect_fence()
        marker = self.store.stage_and_activate(
            command=command,
            process_instance_id=self.process_instance_id,
            fence_check=self._effect_fence,
        )
        return self._deferred(command, marker, "RESTART_REQUIRED")

    def _startup_pending(self) -> bool:
        try:
            return (
                0 <= self._clock() - self._started_at < 600
                and self.runtime.startup_pending() is True
            )
        except Exception:
            return False

    def _matches(
        self,
        command: VerifiedFleetCommand,
        marker: Mapping[str, Any],
        snapshot: Mapping[str, Any],
    ) -> bool:
        if not self.store.readback_matches(command.payload, marker.get("sessionId")):
            return False
        expected_source = "DATASET" if command.payload["mode"] == "DEMO" else "CAMERA"
        if (
            snapshot.get("effectiveMode") != command.payload["mode"]
            or snapshot.get("inputSource") != expected_source
            or snapshot.get("modeRevision") != command.payload["modeRevision"]
            or snapshot.get("health") != "FUNCTIONAL_HEALTHY"
            or snapshot.get("frameReady") is not True
            or snapshot.get("inferenceReady") is not True
        ):
            return False
        if command.payload["mode"] == "DEMO":
            return (
                snapshot.get("profileId") == command.payload["profileId"]
                and snapshot.get("profileDigest") == command.payload["profileDigest"]
                and snapshot.get("sessionId") == marker.get("sessionId")
            )
        return snapshot.get("sessionId") is None

    def _deferred(
        self,
        command: VerifiedFleetCommand,
        marker: Mapping[str, Any],
        health: str,
        *,
        retry_without_restart: bool = False,
    ) -> ReconcileDeferred:
        return ReconcileDeferred(
            reported_state=self._reported(
                command, self.runtime.snapshot(), "TRANSITIONING", health
            ),
            checkpoint={
                "modeRevision": command.payload["modeRevision"],
                "modeMarker": str(self.store.marker_path),
                "markerPhase": marker.get("phase"),
                **(
                    {"nextAction": "RETRY_EFFECT", "restartRequired": False}
                    if retry_without_restart
                    else {}
                ),
            },
        )

    @staticmethod
    def _reported(
        command: VerifiedFleetCommand,
        snapshot: Mapping[str, Any] | None,
        transition_state: str,
        health: str,
    ) -> dict[str, Any]:
        current = dict(snapshot or {})
        return {
            **command.payload,
            "effectiveMode": current.get("effectiveMode"),
            "transitionState": transition_state,
            "inputSource": current.get("inputSource"),
            "profileId": command.payload.get("profileId"),
            "profileDigest": command.payload.get("profileDigest"),
            "sessionId": current.get("sessionId"),
            "health": health,
        }


def validate_managed_demo_profile(payload: Mapping[str, Any]) -> None:
    if payload.get("mode") != "DEMO":
        return
    if (
        payload.get("profileId") != MANAGED_DEMO_PROFILE_ID
        or payload.get("profileDigest") != MANAGED_DEMO_PROFILE_DIGEST
    ):
        raise ValueError("device does not support the requested demo profile identity")


def rollback_failed_mode_startup(
    settings_store: AtomicSettingsStore,
    *,
    process_instance_id: str,
) -> bool:
    """Restore the predecessor before a failed managed mode can crash-loop."""

    store = DeviceModeStore(settings_store)
    marker = store.marker()
    if (
        marker is None
        or marker.get("phase") != "ACTIVATED"
        or not store.lkg_path.exists()
    ):
        return False
    store.rollback(process_instance_id=process_instance_id)
    return True
