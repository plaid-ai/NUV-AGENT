from __future__ import annotations

import base64
import secrets
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nuvion_app.inference.fleet_command import (
    CommandValidationError,
    FleetCommandVerifier,
    VerifiedFleetCommand,
)
from nuvion_app.runtime.release_bom import (
    ReleaseBomValidationError,
    ReleaseKeyring,
    assert_release_compatible,
    assert_release_sequence_allowed,
    load_signed_release_bom,
    verify_release_artifact,
)
from nuvion_updater.errors import UpdaterError, UpdaterSecurityError
from nuvion_updater.health_attestation import (
    CommitProcessIdentity,
    ExpectedHealthAttestation,
    HealthAttestationVerifier,
)
from nuvion_updater.repository import ContentAddressedReleaseRepository
from nuvion_updater.slots import ReleaseSlotManager
from nuvion_updater.store import CommitGate, UpdatePhase, UpdaterStore, UpdateState
from nuvion_updater.trust import DeviceBinding
from nuvion_updater.util import parse_digest


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UpdaterController:
    """Idempotent state machine behind the privileged updater protocol."""

    def __init__(
        self,
        *,
        store: UpdaterStore,
        slots: ReleaseSlotManager,
        repository: ContentAddressedReleaseRepository,
        command_verifier: FleetCommandVerifier,
        release_keyring: ReleaseKeyring,
        binding: DeviceBinding,
        updater_version: str,
        privileged_runtime_ready: Callable[[], bool] | None = None,
        activation_callback: Callable[[str], None] | None = None,
        boot_health_check: Callable[[UpdateState], tuple[bool, str]] | None = None,
        functional_health_check: Callable[[UpdateState], tuple[bool, str]] | None = None,
        commit_process_check: (
            Callable[[UpdateState, int], CommitProcessIdentity] | None
        ) = None,
        health_attestation_verifier: HealthAttestationVerifier | None = None,
        rollback_boot_health_check: Callable[[str], tuple[bool, str]] | None = None,
        safe_stop_callback: Callable[[], None] | None = None,
        activation_timeout_seconds: int = 300,
        boot_timeout_seconds: int = 120,
        functional_timeout_seconds: int = 300,
        commit_timeout_seconds: int = 120,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if any(
            timeout < 1
            for timeout in (
                activation_timeout_seconds,
                boot_timeout_seconds,
                functional_timeout_seconds,
                commit_timeout_seconds,
            )
        ):
            raise ValueError("update phase timeouts must be positive")
        self.store = store
        self.slots = slots
        self.repository = repository
        self.command_verifier = command_verifier
        self.release_keyring = release_keyring
        self.binding = binding
        self.updater_version = updater_version
        self.privileged_runtime_ready = privileged_runtime_ready or (lambda: False)
        self.activation_callback = activation_callback
        self.boot_health_check = boot_health_check
        self.functional_health_check = functional_health_check
        self.commit_process_check = commit_process_check
        self.health_attestation_verifier = health_attestation_verifier
        self.rollback_boot_health_check = rollback_boot_health_check
        self.safe_stop_callback = safe_stop_callback
        self.activation_timeout_seconds = activation_timeout_seconds
        self.boot_timeout_seconds = boot_timeout_seconds
        self.functional_timeout_seconds = functional_timeout_seconds
        self.commit_timeout_seconds = commit_timeout_seconds
        self.clock = clock or _utcnow

    @property
    def capability_available(self) -> bool:
        runtime_ready = (
            not self.binding.docker_required or bool(self.privileged_runtime_ready())
        )
        return (
            runtime_ready
            and self.activation_callback is not None
            and self.boot_health_check is not None
            and self.functional_health_check is not None
            and self.commit_process_check is not None
            and self.health_attestation_verifier is not None
            and self.rollback_boot_health_check is not None
            and self.safe_stop_callback is not None
        )

    @property
    def capability_reason(self) -> str:
        if self.binding.docker_required and not self.privileged_runtime_ready():
            return "PRIVILEGED_RUNTIME_UNAVAILABLE"
        if not self.capability_available:
            return "HEALTH_ADAPTER_UNAVAILABLE"
        return "READY"

    def authorize_and_stage(self, compact_jws: str) -> UpdateState:
        if not self.capability_available:
            raise UpdaterError(
                "PRIVILEGED_RUNTIME_UNAVAILABLE",
                "this profile requires a separate privileged runtime helper",
            )
        try:
            command = self.command_verifier.verify(compact_jws)
        except CommandValidationError as exc:
            raise UpdaterSecurityError(exc.code, str(exc)) from exc
        self._validate_update_command(command)
        state, duplicate = self.store.authorize(
            command_id=command.command_id,
            sequence=command.sequence,
            compact_jws=command.compact_jws,
            target_version=str(command.payload["targetVersion"]),
            bom_digest=str(command.payload["bomDigest"]),
            command_expires_at=command.expires_at,
        )
        if duplicate and state.phase not in {
            UpdatePhase.AUTHORIZED,
            UpdatePhase.DOWNLOADING,
            UpdatePhase.STAGING,
        }:
            return self._enforce_command_expiry(state)
        return self._stage_authorized(state)

    def status(self, command_id: str | None = None) -> dict[str, object]:
        state = self.store.get(command_id) if command_id else self.store.current()
        if state is not None:
            state = self._enforce_command_expiry(state)
            state = self._enforce_health_deadline(state)
        return {
            "capabilityAvailable": self.capability_available,
            "capabilityReason": self.capability_reason,
            "updaterVersion": self.updater_version,
            "activeSlot": self.slots.current_slot(),
            "previousSlot": self.slots.previous_slot(),
            "update": state.public_dict() if state is not None else None,
        }

    def activate(self, command_id: str) -> UpdateState:
        state = self._require_state(command_id)
        if state.phase in {
            UpdatePhase.ACTIVATING,
            UpdatePhase.BOOT_HEALTHY,
            UpdatePhase.FUNCTIONAL_HEALTHY,
            UpdatePhase.COMMITTED,
        }:
            return state
        if state.phase != UpdatePhase.VERIFIED or state.candidate_slot is None:
            raise UpdaterError(
                "INVALID_PHASE", "only a VERIFIED release can be activated"
            )
        deadline = self._deadline(self.boot_timeout_seconds)
        state = self.store.transition(
            command_id,
            UpdatePhase.ACTIVATING,
            allowed_from={UpdatePhase.VERIFIED},
            health_deadline=deadline,
        )
        try:
            active, previous = self.slots.activate(Path(state.candidate_slot))
            previous_version = self.slots.slot_version(previous)
        except (OSError, UpdaterError) as exc:
            self._handle_activation_failure(state, exc)
            raise
        state = self.store.checkpoint(
            command_id,
            required_phase=UpdatePhase.ACTIVATING,
            previous_slot=previous,
            previous_version=previous_version,
            health_deadline=deadline,
        )
        try:
            assert self.activation_callback is not None
            self.activation_callback(active)
        except Exception as exc:
            self._handle_activation_failure(state, exc)
            raise
        return state

    def report_boot_health(
        self, command_id: str, *, healthy: bool, detail: str | None = None
    ) -> UpdateState:
        state = self._require_state(command_id)
        if state.phase in {
            UpdatePhase.BOOT_HEALTHY,
            UpdatePhase.FUNCTIONAL_HEALTHY,
            UpdatePhase.COMMITTED,
        } and healthy:
            return state
        if state.phase != UpdatePhase.ACTIVATING:
            raise UpdaterError("INVALID_PHASE", "boot health is not expected now")
        if not healthy:
            return self.rollback(command_id, reason=detail or "BOOT_HEALTH_FAILED")
        if state.candidate_slot is None or not self.slots.is_active(state.candidate_slot):
            return self.rollback(command_id, reason="ACTIVE_SLOT_MISMATCH")
        assert self.boot_health_check is not None
        try:
            locally_healthy, local_detail = self.boot_health_check(state)
        except Exception as exc:  # noqa: BLE001 - any missing proof rolls back.
            locally_healthy = False
            local_detail = f"{type(exc).__name__}: {exc}"
        if not locally_healthy:
            return self.rollback(
                command_id, reason=local_detail or "ROOT_BOOT_HEALTH_FAILED"
            )
        # systemd/PID verification may itself consume the remaining boot
        # window. Never promote evidence collected after the durable deadline.
        state = self._enforce_health_deadline(state)
        if state.phase != UpdatePhase.ACTIVATING:
            return state
        return self.store.transition(
            command_id,
            UpdatePhase.BOOT_HEALTHY,
            allowed_from={UpdatePhase.ACTIVATING},
            health_deadline=self._deadline(self.functional_timeout_seconds),
        )

    def report_functional_health(
        self, command_id: str, *, healthy: bool, detail: str | None = None
    ) -> UpdateState:
        state = self._require_state(command_id)
        if state.phase in {UpdatePhase.FUNCTIONAL_HEALTHY, UpdatePhase.COMMITTED} and healthy:
            return state
        if state.phase != UpdatePhase.BOOT_HEALTHY:
            raise UpdaterError("INVALID_PHASE", "functional health is not expected now")
        if not healthy:
            return self.rollback(command_id, reason=detail or "FUNCTIONAL_HEALTH_FAILED")
        if state.candidate_slot is None or not self.slots.is_active(state.candidate_slot):
            return self.rollback(command_id, reason="ACTIVE_SLOT_MISMATCH")
        assert self.functional_health_check is not None
        try:
            locally_healthy, local_detail = self.functional_health_check(state)
        except Exception as exc:  # noqa: BLE001 - any missing proof rolls back.
            locally_healthy = False
            local_detail = f"{type(exc).__name__}: {exc}"
        if not locally_healthy:
            return self.rollback(
                command_id,
                reason=local_detail or "ROOT_FUNCTIONAL_HEALTH_FAILED",
            )
        # A physical probe may consume most of the functional-health window.
        # Re-check the persisted deadline after it returns; the accept loop's
        # watchdog cannot run concurrently with this single privileged request.
        state = self._enforce_health_deadline(state)
        if state.phase != UpdatePhase.BOOT_HEALTHY:
            return state
        return self.store.transition(
            command_id,
            UpdatePhase.FUNCTIONAL_HEALTHY,
            allowed_from={UpdatePhase.BOOT_HEALTHY},
            health_deadline=self._deadline(self.commit_timeout_seconds),
        )

    def begin_commit_gate(self, command_id: str, *, peer_pid: int) -> CommitGate:
        state = self._require_state(command_id)
        if state.phase != UpdatePhase.FUNCTIONAL_HEALTHY:
            raise UpdaterError(
                "INVALID_PHASE", "FUNCTIONAL_HEALTHY is required before commit gate"
            )
        if (
            state.candidate_slot is None
            or state.component_sha is None
            or state.release_sequence is None
            or state.health_deadline is None
            or state.command_expires_at is None
        ):
            raise UpdaterError(
                "INVALID_JOURNAL", "commit gate release identity is incomplete"
            )
        process = self._capture_commit_process(state, peer_pid)
        return self.store.begin_commit_gate(
            command_id=command_id,
            gate_id=str(uuid.uuid4()),
            challenge=base64.urlsafe_b64encode(secrets.token_bytes(32))
            .decode("ascii")
            .rstrip("="),
            process=process,
            bom_digest=state.bom_digest,
            component_sha=state.component_sha,
            release_sequence=state.release_sequence,
            health_deadline=state.health_deadline,
            command_expires_at=state.command_expires_at,
        )

    def commit(
        self,
        command_id: str,
        *,
        gate_id: str,
        health_attestation_jws: str,
        peer_pid: int,
    ) -> UpdateState:
        state = self._require_state(command_id)
        if state.phase not in {UpdatePhase.FUNCTIONAL_HEALTHY, UpdatePhase.COMMITTED}:
            raise UpdaterError(
                "INVALID_PHASE", "FUNCTIONAL_HEALTHY is required before commit"
            )
        gate = self.store.commit_gate(command_id)
        if gate is None:
            raise UpdaterSecurityError(
                "COMMIT_GATE_REQUIRED", "BEGIN_COMMIT_GATE is required before commit"
            )
        if gate.gate_id != gate_id:
            raise UpdaterSecurityError(
                "COMMIT_GATE_MISMATCH", "gateId does not match the durable commit gate"
            )
        process = self._capture_commit_process(state, peer_pid)
        if process != gate.process:
            raise UpdaterSecurityError(
                "COMMIT_PROCESS_MISMATCH",
                "live Agent process does not match the durable commit gate",
            )
        assert self.health_attestation_verifier is not None
        verified = self.health_attestation_verifier.verify(
            health_attestation_jws,
            expected=self._expected_health_attestation(gate),
        )
        # Signature verification and local process inspection must not extend
        # the persisted absolute deadline. Timeout follows the existing
        # watchdog rollback path.
        state = self._require_state(command_id)
        if state.phase not in {UpdatePhase.FUNCTIONAL_HEALTHY, UpdatePhase.COMMITTED}:
            return state
        second_process = self._capture_commit_process(state, peer_pid)
        if second_process != gate.process:
            raise UpdaterSecurityError(
                "COMMIT_PROCESS_MISMATCH",
                "Agent process changed while validating the health attestation",
            )
        try:
            return self.store.commit_command_attested(
                command_id,
                gate_id=gate.gate_id,
                process=second_process,
                attestation_id=verified.attestation_id,
                attestation_jws_sha256=verified.compact_jws_sha256,
                attestation_expires_at=verified.expires_at,
            )
        except UpdaterError as exc:
            # The store rechecks every signed/durable absolute deadline inside
            # the same IMMEDIATE transaction that would consume the proof. A
            # slow process check or verifier must therefore roll back, never
            # commit evidence that expired between the outer checks and the
            # atomic journal mutation.
            if exc.code in {
                "COMMAND_EXPIRED",
                "COMMIT_TIMEOUT",
                "HEALTH_ATTESTATION_EXPIRED",
            }:
                self.rollback(command_id, reason=exc.code)
            raise

    def rollback(self, command_id: str, *, reason: str = "OPERATOR_REQUEST") -> UpdateState:
        # Do not call _require_state here: deadline enforcement itself enters
        # rollback and would recurse before ROLLING_BACK is durably recorded.
        state = self.store.get(command_id)
        if state is None:
            raise UpdaterError("UPDATE_NOT_FOUND", "update command is not journaled")
        if state.phase == UpdatePhase.ROLLED_BACK:
            return state
        allowed = {
            UpdatePhase.ACTIVATING,
            UpdatePhase.BOOT_HEALTHY,
            UpdatePhase.FUNCTIONAL_HEALTHY,
            UpdatePhase.ROLLING_BACK,
        }
        if state.phase not in allowed:
            raise UpdaterError("INVALID_PHASE", "release cannot be rolled back now")
        if state.previous_slot is None:
            restore_target = self._recover_restore_target(state)
            state = self.store.checkpoint(
                command_id,
                required_phase=state.phase,
                previous_slot=restore_target,
                previous_version=self.slots.slot_version(restore_target),
            )
        if state.phase != UpdatePhase.ROLLING_BACK:
            state = self.store.transition(
                command_id,
                UpdatePhase.ROLLING_BACK,
                allowed_from=allowed - {UpdatePhase.ROLLING_BACK},
                error_code="HEALTH_GATE_FAILED",
                message=self._safe_detail(reason),
            )
        try:
            assert state.previous_slot is not None
            restored, _ = self.slots.restore(state.previous_slot)
            assert self.activation_callback is not None
            self.activation_callback(restored)
            assert self.rollback_boot_health_check is not None
            healthy, detail = self.rollback_boot_health_check(restored)
            if not healthy:
                raise UpdaterError(
                    "ROLLBACK_HEALTH_FAILED",
                    detail or "restored release did not become healthy",
                )
        except (OSError, UpdaterError) as exc:
            return self._rollback_failed(command_id, exc)
        except Exception as exc:  # noqa: BLE001 - rollback must safe-stop.
            return self._rollback_failed(command_id, exc)
        return self.store.transition(
            command_id,
            UpdatePhase.ROLLED_BACK,
            allowed_from={UpdatePhase.ROLLING_BACK},
            previous_slot=restored,
            error_code="ROLLED_BACK",
            message=self._safe_detail(reason),
        )

    def recover(self) -> UpdateState | None:
        state = self.store.current()
        if state is None:
            return state
        if state.terminal:
            self.repository.cleanup_release(state.bom_digest)
            return state
        state = self._enforce_command_expiry(state)
        if state.terminal:
            self.repository.cleanup_release(state.bom_digest)
            return state
        if state.phase in {
            UpdatePhase.AUTHORIZED,
            UpdatePhase.DOWNLOADING,
            UpdatePhase.STAGING,
        }:
            return self._stage_authorized(state)
        if state.phase == UpdatePhase.ROLLING_BACK:
            return self.rollback(state.command_id, reason=state.message or "RECOVERY")
        if state.phase == UpdatePhase.VERIFIED:
            # The slot is already immutable and complete. A crash between the
            # durable VERIFIED transition and cache cleanup must not consume a
            # bounded download slot forever.
            self.repository.cleanup_release(state.bom_digest)
            return self._enforce_health_deadline(state)
        if state.phase in {
            UpdatePhase.ACTIVATING,
            UpdatePhase.BOOT_HEALTHY,
            UpdatePhase.FUNCTIONAL_HEALTHY,
        }:
            if state.candidate_slot is None:
                return self.rollback(state.command_id, reason="RECOVERY_SLOT_MISSING")
            if not self.slots.is_active(state.candidate_slot):
                return self.rollback(state.command_id, reason="RECOVERY_SLOT_MISMATCH")
            if state.previous_slot is None:
                previous = self.slots.previous_slot()
                if previous is None:
                    return self.rollback(
                        state.command_id, reason="RECOVERY_PREVIOUS_SLOT_MISSING"
                    )
                state = self.store.checkpoint(
                    state.command_id,
                    required_phase=state.phase,
                    previous_slot=previous,
                    previous_version=self.slots.slot_version(previous),
                )
            return self._enforce_health_deadline(state)
        return state

    def _stage_authorized(self, state: UpdateState) -> UpdateState:
        try:
            state = self._require_live_staging_command(state)
            if state.phase == UpdatePhase.AUTHORIZED:
                state = self.store.transition(
                    state.command_id,
                    UpdatePhase.DOWNLOADING,
                    allowed_from={UpdatePhase.AUTHORIZED},
                )
            files = self.repository.fetch_manifest(state.bom_digest)
            bom = load_signed_release_bom(
                files.bom_path,
                files.signature_path,
                release_keyring=self.release_keyring,
                expected_bom_digest=state.bom_digest,
            )
            if bom.schema_version != 2:
                raise UpdaterSecurityError(
                    "UNSIGNED_RELEASE", "AGENT_UPDATE requires signed release-bom-v2"
                )
            if bom.agent_version != state.target_version:
                raise UpdaterSecurityError(
                    "RELEASE_VERSION_MISMATCH",
                    "signed command targetVersion does not match release BOM",
                )
            assert_release_compatible(
                bom,
                product_model=self.binding.product_model,
                platform_profile=self.binding.platform_profile,
                hardware_revision=self.binding.hardware_revision,
                architecture=self.binding.architecture,
                current_updater_version=self.updater_version,
            )
            assert_release_sequence_allowed(
                bom,
                current_release_sequence=self.store.current_release_sequence(),
                current_bom_digest=self.store.current_bom_digest(),
            )
            state = self.store.transition(
                state.command_id,
                UpdatePhase.STAGING,
                allowed_from={UpdatePhase.DOWNLOADING, UpdatePhase.STAGING},
            )
            fetched = self.repository.fetch_artifact(
                files,
                bom_digest=state.bom_digest,
                artifact_name=bom.artifact_name,
                artifact_size=bom.artifact_size_bytes,
            )
            if fetched.artifact_path is None:
                raise UpdaterError("DOWNLOAD_FAILED", "artifact was not downloaded")
            verify_release_artifact(bom, fetched.artifact_path)
            state = self._require_live_staging_command(state)
            slot = self.slots.stage_bundle(
                bom=bom,
                bom_path=fetched.bom_path,
                signature_path=fetched.signature_path,
                artifact_path=fetched.artifact_path,
            )
            state = self._require_live_staging_command(state)
            verified_state = self.store.transition(
                state.command_id,
                UpdatePhase.VERIFIED,
                allowed_from={UpdatePhase.STAGING},
                candidate_slot=str(slot),
                release_sequence=bom.release_sequence,
                artifact_digest=f"sha256:{bom.artifact_sha256}",
                component_sha=bom.component_sha,
                config_schema=bom.config_schema,
                bom_verification_status="VERIFIED",
                publisher_key_id=bom.publisher_key_id,
                health_deadline=self._deadline(self.activation_timeout_seconds),
            )
            self.repository.cleanup_release(state.bom_digest)
            return verified_state
        except Exception as exc:
            latest = self.store.get(state.command_id)
            if latest is not None and latest.phase in {
                UpdatePhase.AUTHORIZED,
                UpdatePhase.DOWNLOADING,
                UpdatePhase.STAGING,
            }:
                code = getattr(exc, "code", "RELEASE_VERIFICATION_FAILED")
                try:
                    self.store.transition(
                        state.command_id,
                        UpdatePhase.FAILED,
                        allowed_from={
                            UpdatePhase.AUTHORIZED,
                            UpdatePhase.DOWNLOADING,
                            UpdatePhase.STAGING,
                        },
                        error_code=str(code)[:100],
                        message=f"{type(exc).__name__}: {str(exc)[:300]}",
                    )
                except UpdaterError:
                    pass
            try:
                self.repository.cleanup_release(state.bom_digest)
            except (OSError, UpdaterError):
                # Preserve the authenticated failure as the primary outcome.
                # The bounded cache quota still prevents unbounded growth and
                # requires operator repair if a root-owned cache is corrupted.
                pass
            if isinstance(exc, UpdaterError):
                raise
            if isinstance(exc, ReleaseBomValidationError):
                raise UpdaterSecurityError(
                    "RELEASE_VERIFICATION_FAILED", str(exc)
                ) from exc
            raise

    @staticmethod
    def _validate_update_command(command: VerifiedFleetCommand) -> None:
        if command.command_type != "AGENT_UPDATE":
            raise UpdaterSecurityError(
                "UNSUPPORTED_COMMAND", "root updater accepts only AGENT_UPDATE"
            )
        if set(command.payload) != {"targetVersion", "bomDigest"}:
            raise UpdaterSecurityError(
                "INVALID_PAYLOAD_SCHEMA", "AGENT_UPDATE payload has unknown fields"
            )
        parse_digest(str(command.payload.get("bomDigest") or ""))

    def _require_state(self, command_id: str) -> UpdateState:
        state = self.store.get(command_id)
        if state is None:
            raise UpdaterError("UPDATE_NOT_FOUND", "update command is not journaled")
        state = self._enforce_command_expiry(state)
        return self._enforce_health_deadline(state)

    def _require_live_staging_command(self, state: UpdateState) -> UpdateState:
        state = self._enforce_command_expiry(state)
        if state.phase == UpdatePhase.FAILED and state.error_code in {
            "COMMAND_EXPIRED",
            "COMMAND_EXPIRY_MISSING",
        }:
            raise UpdaterSecurityError(
                state.error_code,
                state.message or "accepted command is no longer live",
            )
        return state

    def _enforce_command_expiry(self, state: UpdateState) -> UpdateState:
        if state.terminal or state.phase == UpdatePhase.ROLLING_BACK:
            return state
        expiry = self._parse_command_expiry(state.command_expires_at)
        now = self.clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise RuntimeError("updater clock must be timezone-aware")
        now = now.astimezone(timezone.utc)
        code = "COMMAND_EXPIRY_MISSING" if expiry is None else "COMMAND_EXPIRED"
        if expiry is not None and now < expiry:
            return state
        if state.phase in {
            UpdatePhase.AUTHORIZED,
            UpdatePhase.DOWNLOADING,
            UpdatePhase.STAGING,
            UpdatePhase.VERIFIED,
        }:
            return self.store.transition(
                state.command_id,
                UpdatePhase.FAILED,
                allowed_from={state.phase},
                error_code=code,
                message=(
                    "accepted command has no durable expiry"
                    if expiry is None
                    else "accepted command expired before activation"
                ),
            )
        return self.rollback(state.command_id, reason=code)

    @staticmethod
    def _parse_command_expiry(value: str | None) -> datetime | None:
        if value is None:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        return parsed.astimezone(timezone.utc)

    def _enforce_health_deadline(self, state: UpdateState) -> UpdateState:
        if state.phase not in {
            UpdatePhase.VERIFIED,
            UpdatePhase.ACTIVATING,
            UpdatePhase.BOOT_HEALTHY,
            UpdatePhase.FUNCTIONAL_HEALTHY,
        }:
            return state
        if state.health_deadline is None:
            if state.phase == UpdatePhase.VERIFIED:
                return self.store.transition(
                    state.command_id,
                    UpdatePhase.FAILED,
                    allowed_from={UpdatePhase.VERIFIED},
                    error_code="ACTIVATION_DEADLINE_MISSING",
                    message="verified release has no activation lease",
                )
            return self.rollback(state.command_id, reason="HEALTH_DEADLINE_MISSING")
        deadline = datetime.fromisoformat(state.health_deadline.replace("Z", "+00:00"))
        now = self.clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise RuntimeError("updater clock must be timezone-aware")
        if now >= deadline:
            if state.phase == UpdatePhase.VERIFIED:
                return self.store.transition(
                    state.command_id,
                    UpdatePhase.FAILED,
                    allowed_from={UpdatePhase.VERIFIED},
                    error_code="ACTIVATION_TIMEOUT",
                    message="verified release was not activated before its lease expired",
                )
            if state.phase == UpdatePhase.FUNCTIONAL_HEALTHY:
                return self.rollback(state.command_id, reason="COMMIT_TIMEOUT")
            return self.rollback(state.command_id, reason="HEALTH_TIMEOUT")
        return state

    def watchdog_tick(self) -> UpdateState | None:
        """Enforce persisted health deadlines even when no Agent connects."""

        state = self.store.current()
        if state is None or state.terminal:
            return state
        if state.phase == UpdatePhase.ROLLING_BACK:
            return self.rollback(state.command_id, reason=state.message or "WATCHDOG")
        state = self._enforce_command_expiry(state)
        return self._enforce_health_deadline(state)

    def _recover_restore_target(self, state: UpdateState) -> str:
        current = self.slots.current_slot()
        if current is None:
            raise UpdaterError(
                "ROLLBACK_UNAVAILABLE", "no current slot exists for rollback"
            )
        if state.candidate_slot is None:
            return current
        candidate_target = self.slots.relative_target(state.candidate_slot)
        if current != candidate_target:
            return current
        previous = self.slots.previous_slot()
        if previous is None:
            raise UpdaterError(
                "ROLLBACK_UNAVAILABLE", "no previous slot exists for rollback"
            )
        return previous

    def _capture_commit_process(
        self,
        state: UpdateState,
        peer_pid: int,
    ) -> CommitProcessIdentity:
        if self.commit_process_check is None:
            raise UpdaterSecurityError(
                "COMMIT_PROCESS_UNAVAILABLE", "commit process verifier is unavailable"
            )
        try:
            process = self.commit_process_check(state, peer_pid)
        except Exception as exc:
            raise UpdaterSecurityError(
                "COMMIT_PROCESS_UNVERIFIED",
                f"candidate process identity could not be verified: {type(exc).__name__}",
            ) from exc
        if not isinstance(process, CommitProcessIdentity) or process.pid != peer_pid:
            raise UpdaterSecurityError(
                "COMMIT_PROCESS_UNVERIFIED", "candidate process identity is invalid"
            )
        return process

    def _expected_health_attestation(
        self,
        gate: CommitGate,
    ) -> ExpectedHealthAttestation:
        command_expires_at = self._parse_command_expiry(gate.command_expires_at)
        if command_expires_at is None:
            raise UpdaterSecurityError(
                "COMMIT_GATE_BINDING_MISMATCH",
                "commit gate command expiry is invalid",
            )
        return ExpectedHealthAttestation(
            gate_id=gate.gate_id,
            challenge=gate.challenge,
            trust_domain=self.binding.trust_domain,
            device_id=self.binding.device_id,
            command_id=gate.command_id,
            command_expires_at=command_expires_at,
            bom_digest=gate.bom_digest,
            component_sha=gate.component_sha,
            release_sequence=gate.release_sequence,
            product_model=self.binding.product_model,
            platform_profile=self.binding.platform_profile,
            hardware_revision=self.binding.hardware_revision,
            architecture=self.binding.architecture,
        )

    def _rollback_failed(self, command_id: str, exc: Exception) -> UpdateState:
        stop_detail = ""
        try:
            assert self.safe_stop_callback is not None
            self.safe_stop_callback()
        except Exception as stop_exc:  # noqa: BLE001 - preserve both failures.
            stop_detail = (
                f"; safeStop={type(stop_exc).__name__}: {str(stop_exc)[:120]}"
            )
        return self.store.transition(
            command_id,
            UpdatePhase.ROLLBACK_FAILED,
            allowed_from={UpdatePhase.ROLLING_BACK},
            error_code="ROLLBACK_FAILED",
            message=(
                f"{type(exc).__name__}: {str(exc)[:300]}{stop_detail}"
            )[:500],
        )

    def _deadline(self, seconds: int) -> str:
        now = self.clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise RuntimeError("updater clock must be timezone-aware")
        return (
            (now.astimezone(timezone.utc) + timedelta(seconds=seconds))
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )

    def _handle_activation_failure(self, state: UpdateState, exc: Exception) -> None:
        rollback_failure: str | None = None
        try:
            if state.candidate_slot is not None and self.slots.is_active(
                state.candidate_slot
            ):
                self.rollback(state.command_id, reason="ACTIVATION_FAILED")
                return
        except (OSError, UpdaterError) as rollback_exc:
            rollback_failure = (
                f"; rollback={type(rollback_exc).__name__}: {str(rollback_exc)[:150]}"
            )
        latest = self.store.get(state.command_id)
        if latest is not None and latest.phase == UpdatePhase.ACTIVATING:
            self.store.transition(
                state.command_id,
                UpdatePhase.FAILED,
                allowed_from={UpdatePhase.ACTIVATING},
                error_code="ACTIVATION_FAILED",
                message=(
                    f"{type(exc).__name__}: {str(exc)[:300]}"
                    f"{rollback_failure or ''}"
                )[:500],
            )

    @staticmethod
    def _safe_detail(value: str) -> str:
        text = str(value or "").strip()
        return text[:300] if text else "unspecified"
