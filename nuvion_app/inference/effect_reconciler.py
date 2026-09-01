from __future__ import annotations

import fcntl
import os
import stat
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from nuvion_app.inference.command_inbox import (
    COMMAND_STATUS_FAILED,
    COMMAND_STATUS_IN_PROGRESS,
    CommandAck,
    CommandEffectOutcome,
    DurableCommandInbox,
)
from nuvion_app.inference.fleet_command import (
    COMMAND_CAPABILITY_BY_TYPE,
    VerifiedFleetCommand,
)
from nuvion_app.inference.reconcile_store import (
    DurableReconcileStore,
    EffectFence,
    EffectFenceStale,
)


class EffectReconciler(Protocol):
    command_type: str
    capability: str

    def reconcile(self, command: VerifiedFleetCommand) -> CommandEffectOutcome: ...


@dataclass(frozen=True)
class ObservedStateUpdate:
    command_type: str
    command_id: str
    reported_state: dict[str, Any]


@dataclass(frozen=True)
class ReconcileDeferred:
    reported_state: dict[str, Any]
    checkpoint: dict[str, Any]


@dataclass(frozen=True)
class CoordinatorRunResult:
    processed: int
    terminal_acks: tuple[CommandAck, ...]


class _ActuatorFileLocks:
    """Non-blocking cross-process serialization for physical actuator domains."""

    def __init__(self, root: Path, domains: tuple[str, ...]) -> None:
        self.root = root
        self.domains = domains
        self._descriptors: list[int] = []

    def acquire(self) -> bool:
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.root, 0o700)
        except OSError:
            pass
        try:
            for domain in self.domains:
                path = self.root / f"{domain}.lock"
                flags = os.O_RDWR | os.O_CREAT
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                descriptor = os.open(path, flags, 0o600)
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    os.close(descriptor)
                    raise OSError(f"actuator lock is not a regular file: {path}")
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    os.close(descriptor)
                    self.release()
                    return False
                self._descriptors.append(descriptor)
        except Exception:
            self.release()
            raise
        return True

    def release(self) -> None:
        for descriptor in reversed(self._descriptors):
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        self._descriptors.clear()

class ReconcilerRegistry:
    """Thread-safe registry and the single source for executable capabilities."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._reconcilers: dict[str, EffectReconciler] = {}

    def register(self, reconciler: EffectReconciler) -> None:
        command_type = str(reconciler.command_type or "").strip().upper()
        capability = str(reconciler.capability or "").strip()
        expected = COMMAND_CAPABILITY_BY_TYPE.get(command_type)
        if expected is None:
            raise ValueError(f"unsupported reconciler command type: {command_type}")
        if capability != expected:
            raise ValueError(
                f"reconciler capability mismatch for {command_type}: {capability}"
            )
        with self._lock:
            self._reconcilers[command_type] = reconciler

    def unregister(self, command_type: str) -> None:
        with self._lock:
            self._reconcilers.pop(str(command_type or "").strip().upper(), None)

    def get(self, command_type: str) -> EffectReconciler | None:
        with self._lock:
            return self._reconcilers.get(
                str(command_type or "").strip().upper()
            )

    @property
    def command_types(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._reconcilers)

    @property
    def capabilities(self) -> frozenset[str]:
        with self._lock:
            reconcilers = tuple(self._reconcilers.values())
        capabilities: set[str] = set()
        for reconciler in reconcilers:
            readiness = getattr(reconciler, "ready", True)
            try:
                ready = bool(readiness() if callable(readiness) else readiness)
            except Exception:  # noqa: BLE001 - admission must fail closed.
                ready = False
            if ready:
                capabilities.add(reconciler.capability)
        return frozenset(capabilities)

    def observe_connectivity(
        self, sample: Mapping[str, Any]
    ) -> tuple[ObservedStateUpdate, ...]:
        return self._observe("observe_connectivity", sample)

    def observe_stream_metrics(
        self, sample: Mapping[str, Any]
    ) -> tuple[ObservedStateUpdate, ...]:
        return self._observe("observe_stream_metrics", sample)

    def observation_committed(self, update: ObservedStateUpdate) -> None:
        self._notify_observation("observation_committed", update)

    def observation_failed(self, update: ObservedStateUpdate) -> None:
        self._notify_observation("observation_failed", update)

    def _notify_observation(
        self,
        method_name: str,
        update: ObservedStateUpdate,
    ) -> None:
        reconciler = self.get(update.command_type)
        notifier = getattr(reconciler, method_name, None)
        if callable(notifier):
            notifier(update)

    def _observe(
        self,
        method_name: str,
        sample: Mapping[str, Any],
    ) -> tuple[ObservedStateUpdate, ...]:
        updates: list[ObservedStateUpdate] = []
        with self._lock:
            reconcilers = tuple(self._reconcilers.values())
        for reconciler in reconcilers:
            observer = getattr(reconciler, method_name, None)
            if not callable(observer):
                continue
            update = observer(sample)
            if update is None:
                continue
            if not isinstance(update, ObservedStateUpdate):
                raise TypeError("connectivity observer returned an invalid update")
            updates.append(update)
        return tuple(updates)


class FleetEffectCoordinator:
    """Bounded durable effect worker with a persisted lease/checkpoint boundary."""

    def __init__(
        self,
        *,
        inbox: DurableCommandInbox,
        store: DurableReconcileStore,
        registry: ReconcilerRegistry,
        owner: str | None = None,
        lease_seconds: float = 30.0,
        max_jobs_per_run: int = 8,
        process_instance_id: str | None = None,
        restart_requester: Callable[[], bool] | None = None,
    ) -> None:
        self.inbox = inbox
        self.store = store
        self.registry = registry
        self.owner = str(owner or f"agent-{uuid.uuid4()}")
        self.lease_seconds = max(1.0, min(float(lease_seconds), 3600.0))
        self.max_jobs_per_run = max(1, min(int(max_jobs_per_run), 100))
        self.process_instance_id = str(process_instance_id or uuid.uuid4())
        self.restart_requester = restart_requester
        self._run_lock = threading.Lock()
        self._observation_lock = threading.Lock()
        self._restored_command_ids: set[str] = set()
        self._restart_resume_checked = False

    def run_once(self) -> CoordinatorRunResult:
        """Run a bounded batch; external effects execute with no SQLite transaction."""

        if not self._run_lock.acquire(blocking=False):
            return CoordinatorRunResult(processed=0, terminal_acks=())
        processed = 0
        acks: list[CommandAck] = []
        try:
            if not self._restart_resume_checked:
                self.store.requeue_waiting_restart(
                    process_instance_id=self.process_instance_id
                )
                self._restart_resume_checked = True
            for _ in range(self.max_jobs_per_run):
                job = self.store.claim_next(
                    owner=self.owner,
                    lease_seconds=self.lease_seconds,
                )
                if job is None:
                    break
                record = self.inbox.get(job.command_id)
                if record is None or record.terminal:
                    continue
                command = self.inbox.rehydrate(record)
                reconciler = self.registry.get(command.command_type)
                restart_payload = (
                    command.command_type == "CONFIG_APPLY"
                    and command.payload.get("activation") == "RESTART"
                )
                encoder_conflict = (
                    command.command_type == "CONFIG_APPLY"
                    and isinstance(command.payload.get("video"), dict)
                    and self.store.encoder_owned_by_stream_policy()
                )
                fence: EffectFence | None = None
                actuator_locks: _ActuatorFileLocks | None = None
                if encoder_conflict:
                    outcome = CommandEffectOutcome(
                        status=COMMAND_STATUS_FAILED,
                        code="ENCODER_OWNED_BY_STREAM_POLICY",
                        message=(
                            "CONFIG_APPLY video bitrate cannot replace an active "
                            "STREAM_POLICY; disable streaming policy first"
                        ),
                        reported_state={"health": "NOT_APPLIED"},
                    )
                elif restart_payload and self.restart_requester is None:
                    outcome = CommandEffectOutcome(
                        status=COMMAND_STATUS_FAILED,
                        code="RESTART_UNSUPPORTED",
                        message=(
                            "whole-process supervisor restart is not configured; "
                            "settings were not staged"
                        ),
                        reported_state={"health": "NOT_APPLIED"},
                    )
                elif reconciler is None:
                    outcome = CommandEffectOutcome(
                        status=COMMAND_STATUS_FAILED,
                        code="HANDLER_NOT_REGISTERED",
                        message=f"No reconciler registered for {command.command_type}",
                    )
                else:
                    try:
                        _applying, fence = self.store.begin_effect(
                            command,
                            owner=self.owner,
                            lease_seconds=self.lease_seconds,
                        )
                        actuator_locks = _ActuatorFileLocks(
                            self.inbox.path.parent
                            / f".{self.inbox.path.name}.actuators",
                            fence.domains,
                        )
                        if not actuator_locks.acquire():
                            self.store.release_effect_for_retry(
                                command.command_id,
                                owner=self.owner,
                                code="ACTUATOR_BUSY",
                            )
                            break
                        fence.assert_current()
                        bind_fence = getattr(reconciler, "set_effect_fence", None)
                        if callable(bind_fence):
                            bind_fence(fence.assert_current)
                        # Intentionally outside every SQLite transaction.
                        outcome = reconciler.reconcile(command)
                        fence.assert_current()
                        if not isinstance(
                            outcome,
                            (CommandEffectOutcome, ReconcileDeferred),
                        ):
                            raise TypeError(
                                "effect reconciler returned an invalid outcome"
                            )
                        if (
                            isinstance(outcome, CommandEffectOutcome)
                            and outcome.status == COMMAND_STATUS_IN_PROGRESS
                        ):
                            raise ValueError(
                                "effect reconciler must return a terminal outcome"
                            )
                    except EffectFenceStale:
                        # A newer desired/lease owns reconciliation. Its worker
                        # converges the actuator; this stale worker emits no ACK.
                        continue
                    except Exception as exc:  # noqa: BLE001 - durable terminal failure.
                        outcome = CommandEffectOutcome(
                            status=COMMAND_STATUS_FAILED,
                            code="RECONCILE_ERROR",
                            message=f"{type(exc).__name__}: {exc}"[:1000],
                        )
                    finally:
                        if actuator_locks is not None:
                            actuator_locks.release()

                if isinstance(outcome, ReconcileDeferred):
                    checkpoint = {
                        **outcome.checkpoint,
                        "processInstanceId": self.process_instance_id,
                    }
                    next_action = str(
                        checkpoint.get("nextAction") or "RESTART_AGENT"
                    )
                    restart_required = checkpoint.get(
                        "restartRequired", next_action == "RESTART_AGENT"
                    )
                    if next_action == "RETRY_EFFECT" and restart_required is False:
                        retry_delay = min(60.0, float(2 ** min(job.attempts, 6)))
                        self.store.defer_for_retry(
                            command,
                            owner=self.owner,
                            reported_state=outcome.reported_state,
                            checkpoint=checkpoint,
                            retry_after_seconds=retry_delay,
                        )
                        processed += 1
                        # Do not reclaim the same deferred update in this run.
                        break
                    if next_action == "RESTART_AGENT" and restart_required is True:
                        self.store.defer_for_restart(
                            command,
                            owner=self.owner,
                            reported_state=outcome.reported_state,
                            checkpoint=checkpoint,
                        )
                        processed += 1
                        restart_accepted = False
                        try:
                            restart_accepted = bool(
                                self.restart_requester
                                and self.restart_requester()
                            )
                        except Exception:  # noqa: BLE001 - requester is injectable.
                            restart_accepted = False
                        if not restart_accepted:
                            self.store.retry_restart_request(command.command_id)
                        break
                    outcome = CommandEffectOutcome(
                        status=COMMAND_STATUS_FAILED,
                        code="INVALID_DEFERRED_ACTION",
                        message="reconciler returned an inconsistent deferred action",
                        reported_state=outcome.reported_state,
                    )

                ack = self.store.finish(
                    command,
                    owner=self.owner,
                    outcome=outcome,
                )
                processed += 1
                if ack is not None:
                    acks.append(ack)
                    if ack.status == "SUCCEEDED":
                        self._restored_command_ids.add(command.command_id)
                    else:
                        # A failed replacement may have partially touched the
                        # device. Re-apply the last durable applied state next run.
                        self._restored_command_ids.clear()
            if processed == 0:
                self._restore_applied_once()
        finally:
            self._run_lock.release()
        return CoordinatorRunResult(
            processed=processed,
            terminal_acks=tuple(acks),
        )

    def _restore_applied_once(self) -> None:
        """Re-arm long-lived controllers from durable applied state after restart."""

        for command_type in sorted(self.registry.command_types):
            reconciler = self.registry.get(command_type)
            restore = getattr(reconciler, "restore_applied", None)
            if not callable(restore):
                continue
            applied = self.store.applied_state(command_type)
            if applied is None or applied.command_id in self._restored_command_ids:
                continue
            record = self.inbox.get(applied.command_id)
            if record is None or not record.terminal:
                continue
            command = self.inbox.rehydrate(record)
            fence = self.store.fence_applied_state(command, owner=self.owner)
            actuator_locks = _ActuatorFileLocks(
                self.inbox.path.parent / f".{self.inbox.path.name}.actuators",
                fence.domains,
            )
            if not actuator_locks.acquire():
                continue
            try:
                fence.assert_current()
                bind_fence = getattr(reconciler, "set_effect_fence", None)
                if callable(bind_fence):
                    bind_fence(fence.assert_current)
                # No SQLite transaction is open while the controller is restored.
                reported_state = restore(command, applied.reported_state)
                fence.assert_current()
            finally:
                actuator_locks.release()
            if not isinstance(reported_state, Mapping):
                raise TypeError("reconciler restore_applied must return reported state")
            self.store.update_applied_reported_state(
                command_type=command_type,
                command_id=command.command_id,
                reported_state=reported_state,
            )
            self._restored_command_ids.add(command.command_id)

    def observe_connectivity(self, sample: Mapping[str, Any]) -> int:
        """Feed auxiliary link samples and persist the latest applied state."""

        with self._observation_lock:
            return self._persist_observations(
                self.registry.observe_connectivity(sample)
            )

    def observe_stream_metrics(self, sample: Mapping[str, Any]) -> int:
        with self._observation_lock:
            return self._persist_observations(
                self.registry.observe_stream_metrics(sample)
            )

    def _persist_observations(
        self,
        observations: tuple[ObservedStateUpdate, ...],
    ) -> int:
        updated = 0
        for observation in observations:
            try:
                persisted = self.store.update_applied_reported_state(
                    command_type=observation.command_type,
                    command_id=observation.command_id,
                    reported_state=observation.reported_state,
                )
            except Exception:
                self.registry.observation_failed(observation)
                raise
            if persisted:
                self.registry.observation_committed(observation)
                updated += 1
            else:
                self.registry.observation_failed(observation)
        return updated
