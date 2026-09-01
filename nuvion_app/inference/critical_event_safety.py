from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from nuvion_app.inference.durable_events import (
    DurableEvent,
    DurableEventCapacityError,
)


class CriticalEventBackpressureError(DurableEventCapacityError):
    """A critical observation could not cross the durable storage boundary."""

    def __init__(self, event_id: str, reason: str) -> None:
        self.event_id = event_id
        self.reason = reason
        super().__init__(
            f"critical eventId={event_id} retained by safety gate; operator stop required: {reason}"
        )


@dataclass(frozen=True)
class PendingCriticalEvent:
    event_type: str
    destination: str
    payload_json: str
    event_id: str
    occurred_at: str

    @classmethod
    def create(
        cls,
        *,
        event_type: str,
        destination: str,
        payload: Mapping[str, Any],
        event_id: str,
        occurred_at: str,
    ) -> PendingCriticalEvent:
        return cls(
            event_type=str(event_type),
            destination=str(destination),
            payload_json=json.dumps(
                dict(payload),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
            event_id=str(event_id),
            occurred_at=str(occurred_at),
        )

    @property
    def payload(self) -> dict[str, Any]:
        value = json.loads(self.payload_json)
        if not isinstance(value, dict):
            raise TypeError("critical event payload must be an object")
        return value


PersistCritical = Callable[[PendingCriticalEvent], DurableEvent]
RetainCritical = Callable[[PendingCriticalEvent, str], None]
ClearRetainedCritical = Callable[[str], bool]


class CriticalEventSafetyGate:
    """Own one unsaved critical observation and stop new inspection work.

    The slot is deliberately bounded to one exact canonical payload. A capacity
    failure is retried a bounded number of times on the caller stack. If it still
    cannot be persisted, the payload remains in this slot and the runtime enters
    an explicit operator-stop state. A reconnect worker may persist the retained
    payload later, but processing remains stopped until an operator restarts or
    explicitly clears the gate.
    """

    def __init__(
        self,
        *,
        max_attempts: int = 3,
        retry_delay_seconds: float = 0.05,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.max_attempts = max(1, int(max_attempts))
        self.retry_delay_seconds = max(0.0, float(retry_delay_seconds))
        self._sleep = sleep
        self._lock = threading.RLock()
        self._operation_lock = threading.Lock()
        self._pending: PendingCriticalEvent | None = None
        self._operator_stop = False
        self._stop_reason = ""
        self._last_error = ""
        self._attempt_count = 0
        self._retained_event_persisted = False
        self._protocol_stop = False
        self._durable_safety_retained = False

    def persist(
        self,
        event: PendingCriticalEvent,
        persist_once: PersistCritical,
        retain_failed: RetainCritical | None = None,
        clear_retained: ClearRetainedCritical | None = None,
    ) -> DurableEvent:
        with self._operation_lock:
            with self._lock:
                if self._pending is not None and self._pending != event:
                    raise CriticalEventBackpressureError(
                        event.event_id,
                        "another unsaved critical event already owns the bounded safety slot",
                    )
                if self._operator_stop and self._pending is None:
                    raise CriticalEventBackpressureError(
                        event.event_id,
                        self._stop_reason or "runtime is in operator-stop state",
                    )

            last_error: DurableEventCapacityError | None = None
            for attempt in range(1, self.max_attempts + 1):
                try:
                    result = persist_once(event)
                    if result.dead_lettered or result.dropped:
                        raise DurableEventCapacityError(
                            "critical event was not retained in pending outbox"
                        )
                    with self._lock:
                        self._attempt_count += 1
                        if self._pending == event:
                            if clear_retained is not None:
                                try:
                                    clear_retained(event.event_id)
                                except Exception as clear_error:  # noqa: BLE001
                                    self._last_error = (
                                        f"normal outbox persisted but safety slot clear failed: {clear_error}"
                                    )[:500]
                                    return result
                            self._pending = None
                            self._retained_event_persisted = True
                            self._durable_safety_retained = False
                    return result
                except DurableEventCapacityError as exc:
                    last_error = exc
                    with self._lock:
                        self._attempt_count += 1
                    if attempt < self.max_attempts and self.retry_delay_seconds > 0:
                        self._sleep(self.retry_delay_seconds)

            assert last_error is not None
            stop_reason = str(last_error)
            durable_retained = False
            if retain_failed is not None:
                try:
                    retain_failed(event, stop_reason)
                    durable_retained = True
                except Exception as retain_error:  # noqa: BLE001 - must fail-stop uniformly.
                    stop_reason = f"{stop_reason}; durable safety retention failed: {retain_error}"
            self._enter_operator_stop(
                event,
                stop_reason,
                durable_retained=durable_retained,
            )
            raise CriticalEventBackpressureError(event.event_id, stop_reason)

    def retry_retained(
        self,
        persist_once: PersistCritical,
        clear_retained: ClearRetainedCritical | None = None,
    ) -> bool:
        """Try the exact retained payload once without clearing operator stop."""
        with self._operation_lock:
            with self._lock:
                event = self._pending
            if event is None:
                return False
            try:
                result = persist_once(event)
                if result.dead_lettered or result.dropped:
                    raise DurableEventCapacityError(
                        "critical retained event was not accepted into pending outbox"
                    )
            except DurableEventCapacityError as exc:
                with self._lock:
                    self._attempt_count += 1
                    self._last_error = str(exc)[:500]
                return False

            with self._lock:
                if self._pending == event:
                    if clear_retained is not None:
                        try:
                            clear_retained(event.event_id)
                        except Exception as clear_error:  # noqa: BLE001
                            self._last_error = (
                                f"normal outbox persisted but safety slot clear failed: {clear_error}"
                            )[:500]
                            return False
                    self._pending = None
                    self._retained_event_persisted = True
                    self._durable_safety_retained = False
                    self._last_error = ""
            return True

    def enter_protocol_stop(self, reason: str) -> None:
        """Stop replay/inspection when a permanent error has no usable event identity."""
        with self._lock:
            self._operator_stop = True
            self._protocol_stop = True
            self._stop_reason = str(reason or "uncorrelated permanent protocol error")[
                :500
            ]
            self._last_error = self._stop_reason
            self._durable_safety_retained = False

    def restore_retained(self, event: PendingCriticalEvent, reason: str = "") -> None:
        """Restore the crash-safe SQLite slot during startup."""
        with self._lock:
            if self._pending is not None and self._pending != event:
                raise CriticalEventBackpressureError(
                    event.event_id,
                    "in-memory safety slot conflicts with durable safety slot",
                )
            self._pending = event
            self._operator_stop = True
            self._protocol_stop = False
            self._stop_reason = "CRITICAL_OUTBOX_BACKPRESSURE"
            self._last_error = str(reason)[:500]
            self._retained_event_persisted = False
            self._durable_safety_retained = True

    def clear_operator_stop(self) -> bool:
        """Explicit operator action; never clears while an event remains unsaved."""
        with self._lock:
            if self._pending is not None:
                return False
            self._operator_stop = False
            self._stop_reason = ""
            self._last_error = ""
            self._retained_event_persisted = False
            self._protocol_stop = False
            self._durable_safety_retained = False
            return True

    def is_stopped(self) -> bool:
        with self._lock:
            return self._operator_stop

    def pending_event(self) -> PendingCriticalEvent | None:
        with self._lock:
            return self._pending

    def replay_allowed(self) -> bool:
        with self._lock:
            return not self._protocol_stop

    def health_overlay(self) -> dict[str, Any]:
        with self._lock:
            return {
                "unsavedCriticalEvents": 1 if self._pending is not None else 0,
                "safetyStop": self._operator_stop,
                "safetyStopReason": self._stop_reason or None,
                "lastCapacityError": self._last_error or None,
                "criticalPersistAttempts": self._attempt_count,
                "retainedEventPersisted": self._retained_event_persisted,
                "protocolStop": self._protocol_stop,
                "durableSafetyRetained": self._durable_safety_retained,
            }

    def _enter_operator_stop(
        self,
        event: PendingCriticalEvent,
        reason: str,
        *,
        durable_retained: bool = False,
    ) -> None:
        with self._lock:
            if self._pending is not None and self._pending != event:
                raise CriticalEventBackpressureError(
                    event.event_id,
                    "another unsaved critical event already owns the bounded safety slot",
                )
            self._pending = event
            self._operator_stop = True
            self._protocol_stop = False
            self._stop_reason = "CRITICAL_OUTBOX_BACKPRESSURE"
            self._last_error = str(reason)[:500]
            self._retained_event_persisted = False
            self._durable_safety_retained = durable_retained
