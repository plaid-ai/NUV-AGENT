from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

from nuvion_app.inference.command_inbox import (
    COMMAND_STATUS_FAILED,
    COMMAND_STATUS_ROLLED_BACK,
    CommandEffectOutcome,
)
from nuvion_app.inference.effect_reconciler import ReconcileDeferred
from nuvion_app.inference.fleet_command import VerifiedFleetCommand
from nuvion_app.runtime.update_health_attestation import (
    UpdateHealthAttestationError,
)
from nuvion_app.runtime.updater_client import UpdaterClient, UpdaterClientError


_TRANSIENT_UPDATER_CODES = frozenset(
    {
        "UPDATER_UNAVAILABLE",
        "INVALID_RESPONSE",
        "DOWNLOAD_FAILED",
        "INTERNAL_ERROR",
    }
)
_FULL_SHA = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMPACT_JWS = re.compile(
    r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\Z"
)

HealthAttestationProvider = Callable[
    [VerifiedFleetCommand, Mapping[str, Any], Mapping[str, Any]],
    Mapping[str, Any],
]


class AgentUpdateReconciler:
    """Effect worker for the root updater; never called inside Agent SQLite."""

    command_type = "AGENT_UPDATE"
    capability = "command.agent.update"

    def __init__(
        self,
        client: UpdaterClient,
        *,
        readiness_provider: Callable[[], Mapping[str, Any]] | None = None,
        commit_readiness_provider: Callable[[], Mapping[str, Any]] | None = None,
        health_attestation_provider: HealthAttestationProvider | None = None,
    ) -> None:
        self.client = client
        self.readiness_provider = readiness_provider
        self.commit_readiness_provider = commit_readiness_provider
        self.health_attestation_provider = health_attestation_provider

    @property
    def ready(self) -> bool:
        status = (
            self.readiness_provider()
            if self.readiness_provider is not None
            else self.client.capability_status()
        )
        return bool(
            self.health_attestation_provider is not None
            and status.get("capabilityAvailable") is True
            and status.get("authenticatedHelper") is True
        )

    def reconcile(
        self, command: VerifiedFleetCommand
    ) -> CommandEffectOutcome | ReconcileDeferred:
        update: dict[str, Any] | None = None
        activation_attempted = False
        try:
            status = self.client.status(command.command_id)
            candidate = status.get("update")
            if (
                isinstance(candidate, dict)
                and candidate.get("commandId") == command.command_id
            ):
                update = dict(candidate)
                if "slot" not in update and isinstance(status.get("activeSlot"), str):
                    update["slot"] = status["activeSlot"]
            else:
                update = self.client.authorize_and_stage(command.compact_jws)

            phase = str(update.get("phase") or update.get("updatePhase") or "")
            if phase == "VERIFIED":
                activation_attempted = True
                update = self.client.activate(command.command_id)
                return self._deferred(command, update, restart_expected=True)
            if phase == "ACTIVATING":
                update = self.client.report_boot_health(
                    command.command_id,
                    healthy=True,
                    detail="Agent resumed from the activated immutable slot",
                )
                phase = str(update.get("phase") or update.get("updatePhase") or "")
            if phase == "BOOT_HEALTHY":
                update = self.client.report_functional_health(
                    command.command_id,
                    healthy=True,
                    detail="Agent requested the root-owned local functional probe",
                )
                return self._deferred(command, update, restart_expected=True)
            if phase == "FUNCTIONAL_HEALTHY":
                commit_readiness = self._commit_readiness()
                if commit_readiness.get("ready") is not True:
                    return self._deferred(
                        command,
                        update,
                        restart_expected=False,
                        detail=str(
                            commit_readiness.get("reason")
                            or "RUNTIME_READINESS_UNAVAILABLE"
                        ),
                    )
                if self.health_attestation_provider is None:
                    return self._deferred(
                        command,
                        update,
                        restart_expected=False,
                        detail="HEALTH_ATTESTATION_UNAVAILABLE",
                    )
                gate = self.client.begin_commit_gate(command.command_id)
                attestation = self._health_attestation(command, update, gate)
                update = self.client.commit(
                    command.command_id,
                    gate_id=str(gate["gateId"]),
                    health_attestation_jws=str(attestation["compactJws"]),
                )
        except UpdateHealthAttestationError as exc:
            return self._deferred(
                command,
                update,
                restart_expected=False,
                detail=exc.code,
            )
        except UpdaterClientError as exc:
            if exc.code in _TRANSIENT_UPDATER_CODES:
                return self._deferred(
                    command,
                    update,
                    restart_expected=(
                        activation_attempted and exc.code == "INVALID_RESPONSE"
                    ),
                    detail=exc.code,
                )
            return CommandEffectOutcome(
                status=COMMAND_STATUS_FAILED,
                code=exc.code[:100],
                message=str(exc)[:1000],
                reported_state=self._reported(command, update),
            )
        except OSError as exc:
            return self._deferred(
                command,
                update,
                restart_expected=False,
                detail=type(exc).__name__,
            )

        assert update is not None
        phase = str(update.get("phase") or update.get("updatePhase") or "")
        if phase == "COMMITTED":
            try:
                evidence = self._committed_evidence(command, update)
            except ValueError as exc:
                return CommandEffectOutcome(
                    status=COMMAND_STATUS_FAILED,
                    code="UPDATE_EVIDENCE_INCOMPLETE",
                    message=str(exc)[:1000],
                    reported_state=self._reported(command, update),
                )
            return CommandEffectOutcome.succeeded(evidence)
        if phase == "ROLLED_BACK":
            return CommandEffectOutcome(
                status=COMMAND_STATUS_ROLLED_BACK,
                code=str(update.get("errorCode") or "ROLLED_BACK")[:100],
                message=str(update.get("message") or "release rolled back")[:1000],
                reported_state=self._reported(command, update),
            )
        if phase in {"FAILED", "ROLLBACK_FAILED"}:
            return CommandEffectOutcome(
                status=COMMAND_STATUS_FAILED,
                code=str(update.get("errorCode") or phase)[:100],
                message=str(update.get("message") or "release update failed")[:1000],
                reported_state=self._reported(command, update),
            )
        return self._deferred(command, update, restart_expected=False)

    def _health_attestation(
        self,
        command: VerifiedFleetCommand,
        update: Mapping[str, Any],
        gate: Mapping[str, Any] | object,
    ) -> Mapping[str, Any]:
        if self.health_attestation_provider is None or not isinstance(gate, Mapping):
            raise UpdateHealthAttestationError(
                "HEALTH_ATTESTATION_UNAVAILABLE",
                "health attestation provider or root commit gate is unavailable",
            )
        try:
            attestation = self.health_attestation_provider(command, update, gate)
        except UpdateHealthAttestationError:
            raise
        except Exception as exc:  # noqa: BLE001 - watchdog remains the rollback boundary.
            raise UpdateHealthAttestationError(
                "HEALTH_ATTESTATION_UNAVAILABLE",
                "health attestation provider failed",
            ) from exc
        if not isinstance(attestation, Mapping):
            raise UpdateHealthAttestationError(
                "HEALTH_ATTESTATION_INVALID",
                "health attestation response is invalid",
            )
        compact_jws = attestation.get("compactJws")
        if (
            not isinstance(compact_jws, str)
            or len(compact_jws) > 32 * 1024
            or not _COMPACT_JWS.fullmatch(compact_jws)
        ):
            raise UpdateHealthAttestationError(
                "HEALTH_ATTESTATION_INVALID",
                "health attestation compact JWS is invalid",
            )
        return attestation

    def _commit_readiness(self) -> Mapping[str, Any]:
        if self.commit_readiness_provider is None:
            return {
                "ready": False,
                "reason": "RUNTIME_READINESS_UNAVAILABLE",
            }
        try:
            readiness = self.commit_readiness_provider()
        except Exception:  # noqa: BLE001 - malformed runtime evidence fails closed.
            return {
                "ready": False,
                "reason": "RUNTIME_READINESS_UNAVAILABLE",
            }
        if not isinstance(readiness, Mapping):
            return {
                "ready": False,
                "reason": "RUNTIME_READINESS_UNAVAILABLE",
            }
        reason = readiness.get("reason")
        if (
            readiness.get("ready") is not True
            or not isinstance(reason, str)
            or not reason
            or len(reason) > 100
            or not reason.replace("_", "").isalnum()
        ):
            return {
                "ready": False,
                "reason": (
                    reason
                    if isinstance(reason, str)
                    and 0 < len(reason) <= 100
                    and reason.replace("_", "").isalnum()
                    else "RUNTIME_READINESS_UNAVAILABLE"
                ),
            }
        return readiness

    def _deferred(
        self,
        command: VerifiedFleetCommand,
        update: dict[str, Any] | None,
        *,
        restart_expected: bool,
        detail: str | None = None,
    ) -> ReconcileDeferred:
        phase = str(
            (update or {}).get("phase")
            or (update or {}).get("updatePhase")
            or "UNKNOWN"
        )
        checkpoint: dict[str, Any] = {
            "updaterPhase": phase,
            "restartExpected": restart_expected,
            "restartRequired": restart_expected,
            "nextAction": "RESTART_AGENT" if restart_expected else "RETRY_EFFECT",
        }
        if detail:
            checkpoint["detail"] = detail[:100]
        return ReconcileDeferred(
            reported_state=self._reported(command, update),
            checkpoint=checkpoint,
        )

    @staticmethod
    def _reported(
        command: VerifiedFleetCommand,
        update: dict[str, Any] | None,
    ) -> dict[str, Any]:
        evidence = dict(command.payload)
        if update is None:
            evidence.update(
                {
                    "updatePhase": "UNKNOWN",
                    "functionalHealth": "FUNCTIONAL_UNHEALTHY",
                }
            )
            return evidence
        for key in (
            "artifactDigest",
            "componentSha",
            "configSchema",
            "releaseSequence",
            "bomVerificationStatus",
            "health",
            "functionalHealth",
            "slot",
            "candidateSlot",
            "previousSlot",
            "previousVersion",
            "rollbackSlot",
            "rollbackVersion",
            "publisherKeyId",
            "errorCode",
            "message",
        ):
            if key in update:
                evidence[key] = update[key]
        phase = str(update.get("phase") or update.get("updatePhase") or "UNKNOWN")
        evidence["updatePhase"] = phase
        if phase == "ROLLED_BACK" and isinstance(update.get("previousVersion"), str):
            evidence["agentVersion"] = update["previousVersion"]
        else:
            evidence["agentVersion"] = str(
                update.get("targetVersion") or command.payload["targetVersion"]
            )
        return evidence

    @classmethod
    def _committed_evidence(
        cls,
        command: VerifiedFleetCommand,
        update: dict[str, Any],
    ) -> dict[str, Any]:
        reported = cls._reported(command, update)
        if update.get("targetVersion") != command.payload["targetVersion"]:
            raise ValueError("helper targetVersion does not match desired targetVersion")
        if update.get("bomDigest") != command.payload["bomDigest"]:
            raise ValueError("helper bomDigest does not match desired bomDigest")
        if reported.get("agentVersion") != command.payload["targetVersion"]:
            raise ValueError(
                "committed agentVersion does not match desired targetVersion"
            )
        if reported.get("bomDigest") != command.payload["bomDigest"]:
            raise ValueError("committed bomDigest does not match desired bomDigest")
        artifact_digest = reported.get("artifactDigest")
        if not isinstance(artifact_digest, str) or not _DIGEST.fullmatch(
            artifact_digest
        ):
            raise ValueError("committed artifactDigest is missing or invalid")
        component_sha = reported.get("componentSha")
        if not isinstance(component_sha, str) or not _FULL_SHA.fullmatch(component_sha):
            raise ValueError("committed full componentSha is missing or invalid")
        config_schema = reported.get("configSchema")
        if (
            not isinstance(config_schema, str)
            or not config_schema.isdigit()
            or int(config_schema) < 1
        ):
            raise ValueError("committed configSchema is missing or invalid")
        release_sequence = reported.get("releaseSequence")
        if (
            isinstance(release_sequence, bool)
            or not isinstance(release_sequence, int)
            or release_sequence < 1
        ):
            raise ValueError("committed releaseSequence is missing or invalid")
        if reported.get("bomVerificationStatus") != "VERIFIED":
            raise ValueError(
                "publisher-authenticated BOM verification evidence is missing"
            )
        if reported.get("health") != "FUNCTIONAL_HEALTHY":
            raise ValueError("functional health evidence is missing")
        if reported.get("functionalHealth") != "FUNCTIONAL_HEALTHY":
            raise ValueError("canonical functionalHealth evidence is missing")
        expected_slot = f"releases/{str(command.payload['bomDigest'])[7:]}"
        if reported.get("slot") != expected_slot:
            raise ValueError("committed slot evidence is missing")
        if (
            not isinstance(reported.get("previousVersion"), str)
            or not reported["previousVersion"]
        ):
            raise ValueError("committed previousVersion evidence is missing")
        return reported


def configure_agent_update_reconciler(
    registry: Any,
    *,
    client: UpdaterClient | None = None,
    commit_readiness_provider: Callable[[], Mapping[str, Any]] | None = None,
    health_attestation_provider: HealthAttestationProvider | None = None,
) -> dict[str, Any]:
    """Install the durable handler while gating admission on live readiness.

    The registry keeps the reconciler even while the helper is temporarily down,
    so an already accepted desired state can retry instead of becoming a permanent
    HANDLER_NOT_REGISTERED failure. ``ReconcilerRegistry.capabilities`` consults
    ``ready`` and therefore hides command.agent.update from new admission and
    heartbeat telemetry until the authenticated helper reports READY again.
    """

    updater_client = client or UpdaterClient()
    status = updater_client.capability_status()
    registry.register(
        AgentUpdateReconciler(
            updater_client,
            commit_readiness_provider=commit_readiness_provider,
            health_attestation_provider=health_attestation_provider,
        )
    )
    return status
