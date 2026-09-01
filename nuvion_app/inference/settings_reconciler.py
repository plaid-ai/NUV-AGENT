from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import tempfile
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from nuvion_app.inference.command_inbox import (
    COMMAND_STATUS_FAILED,
    COMMAND_STATUS_ROLLED_BACK,
    CommandEffectOutcome,
)
from nuvion_app.inference.effect_reconciler import ReconcileDeferred
from nuvion_app.inference.fleet_command import VerifiedFleetCommand
from nuvion_app.runtime.settings_overlay import (
    parse_settings_overlay,
    serialize_settings_overlay,
)
from nuvion_app.runtime.telemetry import DEFAULT_CONFIG_SCHEMA

CONFIG_APPLY_COMMAND_TYPE = "CONFIG_APPLY"
CONFIG_APPLY_CAPABILITY = "command.config.apply"


class UnsupportedSettingsEffect(RuntimeError):
    pass


class SettingsRuntimeAdapter(Protocol):
    def snapshot(self) -> dict[str, Any]: ...

    def apply_immediate(self, desired: Mapping[str, Any]) -> dict[str, Any]: ...

    def restore(self, snapshot: Mapping[str, Any]) -> None: ...

    def functional_health(self) -> bool: ...

    def verify_model(self, desired: Mapping[str, Any]) -> dict[str, str]: ...

    def verify_labels(self, desired: Mapping[str, Any]) -> dict[str, Any]: ...


def canonical_settings_digest(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        dict(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def config_env_updates(payload: Mapping[str, Any]) -> dict[str, str]:
    updates: dict[str, str] = {}
    model = payload.get("model")
    if isinstance(model, dict):
        updates["NUVION_MODEL_POINTER"] = str(model["pointer"])
        updates["NUVION_MODEL_DIGEST"] = str(model["digest"])
    labels = payload.get("labels")
    if isinstance(labels, dict):
        if "inspection" in labels:
            updates["NUVION_ZERO_SHOT_LABELS_B64"] = _encode_label_array(
                labels["inspection"]
            )
        if "anomaly" in labels:
            updates["NUVION_ZERO_SHOT_ANOMALY_LABELS_B64"] = _encode_label_array(
                labels["anomaly"]
            )
    clip = payload.get("clip")
    if isinstance(clip, dict):
        updates["NUVION_CLIP_ENABLED"] = (
            "true" if bool(clip["enabled"]) else "false"
        )
        updates["NUVION_CLIP_PRE_SEC"] = str(int(clip["preSeconds"]))
        updates["NUVION_CLIP_POST_SEC"] = str(int(clip["postSeconds"]))
    video = payload.get("video")
    if isinstance(video, dict):
        updates["NUVION_VIDEO_WIDTH"] = str(int(video["width"]))
        updates["NUVION_VIDEO_HEIGHT"] = str(int(video["height"]))
        updates["NUVION_VIDEO_FPS"] = str(int(video["fps"]))
        updates["NUVION_VIDEO_BITRATE_KBPS"] = str(int(video["bitrateKbps"]))
    return updates


def _encode_label_array(labels: Any) -> str:
    encoded = json.dumps(
        list(labels),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return base64.urlsafe_b64encode(encoded).decode("ascii").rstrip("=")


class AtomicSettingsStore:
    """Crash-consistent typed overlay activation; the base secret env is read-only."""

    def __init__(
        self,
        base_config_path: str | Path,
        state_dir: str | Path,
        *,
        fault_hook: Callable[[str], None] | None = None,
    ) -> None:
        self.base_config_path = Path(
            os.path.abspath(Path(base_config_path).expanduser())
        )
        self.state_dir = Path(os.path.abspath(Path(state_dir).expanduser()))
        self.active_path = self.state_dir / "active.env"
        self.candidate_path = self.state_dir / "candidate.env"
        self.lkg_path = self.state_dir / "lkg.env"
        self.marker_path = self.state_dir / "restart-marker.json"
        self._fault_hook = fault_hook or (lambda _boundary: None)
        self._reject_symlink(self.base_config_path)
        self._reject_symlink(self.state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.state_dir, 0o700)
        except OSError:
            pass

    @staticmethod
    def _reject_symlink(path: Path) -> None:
        try:
            if stat.S_ISLNK(path.lstat().st_mode):
                raise ValueError(f"settings path must not be a symlink: {path}")
        except FileNotFoundError:
            return

    @staticmethod
    def _sha256(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def _atomic_write(path: Path, content: bytes, *, mode: int = 0o600) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            dir=str(path.parent),
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, mode & 0o777)
            offset = 0
            while offset < len(content):
                offset += os.write(descriptor, content[offset:])
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            os.replace(temporary, path)
            AtomicSettingsStore._fsync_directory(path.parent)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _read_bytes(self, path: Path, *, missing: bytes | None = None) -> bytes:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
        except FileNotFoundError:
            if missing is not None:
                return missing
            raise
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"settings file must be regular: {path}")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(descriptor, 8192)
                if not chunk:
                    break
                total += len(chunk)
                if total > 64 * 1024:
                    raise ValueError(f"settings file exceeds 64 KiB: {path}")
                chunks.append(chunk)
            after = os.fstat(descriptor)
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ValueError(f"settings file changed while reading: {path}")
            return b"".join(chunks)
        finally:
            os.close(descriptor)

    def _write_marker(self, marker: Mapping[str, Any]) -> dict[str, Any]:
        normalized = dict(marker)
        self._atomic_write(
            self.marker_path,
            json.dumps(
                normalized,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8"),
        )
        return normalized

    def marker(self) -> dict[str, Any] | None:
        try:
            raw = self._read_bytes(self.marker_path)
        except FileNotFoundError:
            return None
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("settings restart marker is unreadable or corrupt") from exc
        if not isinstance(value, dict):
            raise ValueError(  # noqa: TRY004 - persisted corruption is one value error.
                "settings restart marker must be a JSON object"
            )
        return value

    def _switch_candidate_to_active(self, *, expected_digest: str | None = None) -> None:
        self._reject_symlink(self.candidate_path)
        self._reject_symlink(self.active_path)
        if expected_digest is not None:
            candidate = self._read_bytes(self.candidate_path)
            if self._sha256(candidate) != expected_digest:
                raise ValueError("settings candidate changed before active switch")
        os.replace(self.candidate_path, self.active_path)
        self._fsync_directory(self.state_dir)

    def recover_prepared(self) -> dict[str, Any] | None:
        marker = self.marker()
        if marker is None or marker.get("phase") != "PREPARED":
            return marker
        candidate_digest = str(marker.get("candidateSha256") or "")
        lkg_digest = str(marker.get("lkgSha256") or "")
        active = self._read_bytes(self.active_path, missing=b"")
        candidate = self._read_bytes(self.candidate_path, missing=b"")
        lkg = self._read_bytes(self.lkg_path)
        if self._sha256(lkg) != lkg_digest:
            raise ValueError("settings LKG digest does not match PREPARED marker")
        if self._sha256(active) == candidate_digest:
            marker["phase"] = "ACTIVATED"
            return self._write_marker(marker)
        if (
            self._sha256(active) == lkg_digest
            and candidate
            and self._sha256(candidate) == candidate_digest
        ):
            self._switch_candidate_to_active(expected_digest=candidate_digest)
            marker["phase"] = "ACTIVATED"
            return self._write_marker(marker)
        self._atomic_write(self.active_path, lkg)
        marker.update(
            {
                "phase": "ROLLBACK_STAGED",
                "restoredSha256": lkg_digest,
                "recoveryReason": "PREPARED_HASH_MISMATCH",
                "rollbackBootAttempts": 0,
            }
        )
        return self._write_marker(marker)

    def _restore_uncommitted_predecessor(
        self,
        marker: Mapping[str, Any] | None,
    ) -> None:
        if marker is None or marker.get("phase") not in {
            "PREPARED",
            "ACTIVATED",
            "ROLLBACK_STAGED",
        }:
            return
        recovered = self.recover_prepared() if marker.get("phase") == "PREPARED" else marker
        if recovered is not None and recovered.get("phase") == "ACTIVATED":
            lkg = self._read_bytes(self.lkg_path)
            self._atomic_write(self.active_path, lkg)

    def stage_and_activate(
        self,
        *,
        command: VerifiedFleetCommand,
        process_instance_id: str,
        settings_digest: str,
        fence_check: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        ensure_fence = fence_check or (lambda: None)
        existing = self.marker()
        if (
            existing is not None
            and existing.get("commandId") == command.command_id
            and existing.get("settingsDigest") == settings_digest
        ):
            return self.recover_prepared() or existing
        self._restore_uncommitted_predecessor(existing)
        current = self._read_bytes(self.active_path, missing=b"")
        values = parse_settings_overlay(current.decode("utf-8")) if current else {}
        values.update(config_env_updates(command.payload))
        candidate = serialize_settings_overlay(values)
        marker = {
            "schemaVersion": 2,
            "commandId": command.command_id,
            "configVersion": command.payload["configVersion"],
            "activation": command.payload["activation"],
            "settingsDigest": settings_digest,
            "phase": "PREPARED",
            "stagedProcessInstanceId": process_instance_id,
            "candidateSha256": self._sha256(candidate),
            "lkgSha256": self._sha256(current),
            "bootAttempts": 0,
            "rollbackBootAttempts": 0,
        }
        ensure_fence()
        self._atomic_write(self.lkg_path, current)
        self._fault_hook("LKG_FSYNCED")
        ensure_fence()
        self._atomic_write(self.candidate_path, candidate)
        self._fault_hook("CANDIDATE_FSYNCED")
        ensure_fence()
        self._write_marker(marker)
        self._fault_hook("PREPARED_FSYNCED")
        ensure_fence()
        self._switch_candidate_to_active(expected_digest=marker["candidateSha256"])
        self._fault_hook("ACTIVE_SWITCHED")
        marker["phase"] = "ACTIVATED"
        activated = self._write_marker(marker)
        self._fault_hook("ACTIVATED_FSYNCED")
        return activated

    def readback_matches(self, payload: Mapping[str, Any]) -> bool:
        active = self._read_bytes(self.active_path, missing=b"")
        values = parse_settings_overlay(active.decode("utf-8")) if active else {}
        return all(
            values.get(key) == value
            for key, value in config_env_updates(payload).items()
        )

    def lkg_is_active(self) -> bool:
        try:
            active = self._read_bytes(self.active_path, missing=b"")
            lkg = self._read_bytes(self.lkg_path)
            return self._sha256(active) == self._sha256(lkg)
        except OSError:
            return False

    def rollback(
        self,
        *,
        process_instance_id: str,
        restart_required: bool = True,
        command: VerifiedFleetCommand | None = None,
        settings_digest: str | None = None,
    ) -> dict[str, Any]:
        lkg = self._read_bytes(self.lkg_path)
        self._atomic_write(self.active_path, lkg)
        try:
            marker = dict(self.marker() or {})
        except ValueError:
            marker = {}
        if command is not None:
            marker.update(
                {
                    "commandId": command.command_id,
                    "configVersion": command.payload["configVersion"],
                    "activation": command.payload["activation"],
                    "settingsDigest": settings_digest,
                }
            )
        marker.update(
            {
                "phase": (
                    "ROLLBACK_STAGED" if restart_required else "ROLLED_BACK"
                ),
                "rollbackProcessInstanceId": process_instance_id,
                "restoredSha256": self._sha256(lkg),
                "rollbackBootAttempts": 0,
            }
        )
        return self._write_marker(marker)

    def commit(self) -> None:
        marker = dict(self.marker() or {})
        marker["phase"] = "COMMITTED"
        self._write_marker(marker)

    def update_marker(self, updates: Mapping[str, Any]) -> dict[str, Any]:
        marker = dict(self.marker() or {})
        marker.update(dict(updates))
        return self._write_marker(marker)


class SettingsReconciler:
    command_type = CONFIG_APPLY_COMMAND_TYPE
    capability = CONFIG_APPLY_CAPABILITY

    def __init__(
        self,
        *,
        store: AtomicSettingsStore,
        runtime: SettingsRuntimeAdapter,
        process_instance_id: str | None = None,
        config_schema: str = DEFAULT_CONFIG_SCHEMA,
    ) -> None:
        self.store = store
        self.runtime = runtime
        self.process_instance_id = str(process_instance_id or uuid.uuid4())
        self.config_schema = str(config_schema)
        self._effect_fence: Callable[[], None] = lambda: None
        if not self.config_schema.isdigit() or int(self.config_schema) < 1:
            raise ValueError("config_schema must be a positive integer string")

    def set_effect_fence(self, fence_check: Callable[[], None]) -> None:
        if not callable(fence_check):
            raise TypeError("settings effect fence must be callable")
        self._effect_fence = fence_check
        bind_runtime_fence = getattr(self.runtime, "set_effect_fence", None)
        if callable(bind_runtime_fence):
            bind_runtime_fence(fence_check)

    def _ensure_fence(self) -> None:
        self._effect_fence()

    def reconcile(
        self, command: VerifiedFleetCommand
    ) -> CommandEffectOutcome | ReconcileDeferred:
        digest = canonical_settings_digest(command.payload)
        try:
            marker = self.store.marker()
        except ValueError as exc:
            if self.store.lkg_path.exists():
                self._ensure_fence()
                rollback_marker = self.store.rollback(
                    process_instance_id=self.process_instance_id,
                    command=command,
                    settings_digest=digest,
                )
                return self._deferred(
                    command,
                    digest,
                    health="ROLLBACK_RESTART_REQUIRED",
                    marker=rollback_marker,
                )
            return CommandEffectOutcome(
                status=COMMAND_STATUS_FAILED,
                code="SETTINGS_MARKER_INVALID",
                message=str(exc)[:1000],
                reported_state=self._reported(
                    command,
                    digest,
                    health="NOT_APPLIED",
                ),
            )
        if marker is not None and marker.get("commandId") == command.command_id:
            phase = marker.get("phase")
            if phase == "ROLLBACK_STAGED":
                if marker.get("rollbackProcessInstanceId") == self.process_instance_id:
                    return self._deferred(
                        command,
                        digest,
                        health="ROLLBACK_RESTART_REQUIRED",
                        marker=marker,
                    )
                if self.store.lkg_is_active() and self._functional_health():
                    self._ensure_fence()
                    self.store.update_marker(
                        {
                            "phase": "ROLLED_BACK",
                            "rollbackCompletedByProcessInstanceId": self.process_instance_id,
                        }
                    )
                    return CommandEffectOutcome(
                        status=COMMAND_STATUS_ROLLED_BACK,
                        code="FUNCTIONAL_HEALTH_ROLLBACK",
                        message="last-known-good settings restored after restart",
                        reported_state=self._reported(
                            command,
                            digest,
                            health="LKG_RESTORED",
                        ),
                    )
                return CommandEffectOutcome(
                    status=COMMAND_STATUS_FAILED,
                    code="ROLLBACK_HEALTH_FAILED",
                    message="last-known-good settings did not recover functional health",
                    reported_state=self._reported(
                        command,
                        digest,
                        health="LKG_UNHEALTHY",
                    ),
                )
            if (
                command.payload["activation"] == "RESTART"
                and marker.get("stagedProcessInstanceId") != self.process_instance_id
            ):
                runtime_matches = False
                try:
                    readback = self.runtime.snapshot()
                    self._verify_runtime_readback(command.payload, readback)
                    runtime_matches = self._functional_health()
                except Exception:  # noqa: BLE001 - any failed proof triggers LKG.
                    runtime_matches = False
                if (
                    self.store.readback_matches(command.payload)
                    and runtime_matches
                ):
                    self._ensure_fence()
                    self.store.commit()
                    return CommandEffectOutcome.succeeded(
                        self._reported(command, digest, health="FUNCTIONAL_HEALTHY")
                    )
                self._ensure_fence()
                rollback_marker = self.store.rollback(
                    process_instance_id=self.process_instance_id,
                    command=command,
                    settings_digest=digest,
                )
                return self._deferred(
                    command,
                    digest,
                    health="ROLLBACK_RESTART_REQUIRED",
                    marker=rollback_marker,
                )

        runtime_snapshot = self.runtime.snapshot()
        try:
            self._ensure_fence()
            marker = self.store.stage_and_activate(
                command=command,
                process_instance_id=self.process_instance_id,
                settings_digest=digest,
                fence_check=self._effect_fence,
            )
            if command.payload["activation"] == "RESTART":
                return self._deferred(
                    command,
                    digest,
                    health="RESTART_REQUIRED",
                    marker=marker,
                )

            self._ensure_fence()
            readback = self.runtime.apply_immediate(command.payload)
            self._ensure_fence()
            self._verify_runtime_readback(command.payload, readback)
            if not self.store.readback_matches(command.payload):
                raise RuntimeError("activated config file readback mismatch")
            if not self.runtime.functional_health():
                raise RuntimeError("functional health gate failed")
            self._ensure_fence()
            self.store.commit()
            return CommandEffectOutcome.succeeded(
                self._reported(command, digest, health="FUNCTIONAL_HEALTHY")
            )
        except UnsupportedSettingsEffect as exc:
            self._ensure_fence()
            self.runtime.restore(runtime_snapshot)
            self.store.rollback(
                process_instance_id=self.process_instance_id,
                restart_required=False,
                command=command,
                settings_digest=digest,
            )
            return CommandEffectOutcome(
                status=COMMAND_STATUS_FAILED,
                code="UNSUPPORTED_ACTUAL_EFFECT",
                message=str(exc)[:1000],
                reported_state=self._reported(
                    command,
                    digest,
                    health="NOT_APPLIED",
                ),
            )
        except Exception as exc:  # noqa: BLE001 - rollback is the safety boundary.
            self._ensure_fence()
            self.runtime.restore(runtime_snapshot)
            self.store.rollback(
                process_instance_id=self.process_instance_id,
                restart_required=False,
                command=command,
                settings_digest=digest,
            )
            return CommandEffectOutcome(
                status=COMMAND_STATUS_ROLLED_BACK,
                code="FUNCTIONAL_HEALTH_ROLLBACK",
                message=f"{type(exc).__name__}: {exc}"[:1000],
                reported_state=self._reported(
                    command,
                    digest,
                    health="LKG_RESTORED",
                ),
            )

    def _deferred(
        self,
        command: VerifiedFleetCommand,
        digest: str,
        *,
        health: str,
        marker: Mapping[str, Any],
    ) -> ReconcileDeferred:
        return ReconcileDeferred(
            reported_state=self._reported(command, digest, health=health),
            checkpoint={
                "settingsDigest": digest,
                "restartMarker": str(self.store.marker_path),
                "markerPhase": marker.get("phase"),
            },
        )

    def _reported(
        self,
        command: VerifiedFleetCommand,
        digest: str,
        *,
        health: str,
    ) -> dict[str, Any]:
        return {
            **command.payload,
            "configSchema": self.config_schema,
            "settingsDigest": digest,
            "health": health,
        }

    def _functional_health(self) -> bool:
        try:
            return self.runtime.functional_health() is True
        except Exception:  # noqa: BLE001 - health evidence is fail-closed.
            return False

    def _verify_runtime_readback(
        self,
        desired: Mapping[str, Any],
        readback: Mapping[str, Any],
    ) -> None:
        if "model" in desired:
            verifier = getattr(self.runtime, "verify_model", None)
            if not callable(verifier):
                raise UnsupportedSettingsEffect(
                    "runtime has no authenticated model artifact verifier"
                )
            verified_model = verifier(desired["model"])
            if verified_model != desired["model"]:
                raise RuntimeError("runtime model artifact identity mismatch")
        if "labels" in desired:
            verifier = getattr(self.runtime, "verify_labels", None)
            if not callable(verifier):
                raise UnsupportedSettingsEffect(
                    "runtime has no labels readback verifier"
                )
            if verifier(desired["labels"]) != desired["labels"]:
                raise RuntimeError("runtime labels readback mismatch")
        for section in ("clip", "video"):
            if section in desired and readback.get(section) != desired[section]:
                raise RuntimeError(f"runtime readback mismatch for {section}")
