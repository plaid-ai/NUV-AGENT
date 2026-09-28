from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any
from datetime import datetime, timezone

from nuvion_app.inference.command_inbox import (
    COMMAND_STATUS_FAILED,
    CommandEffectOutcome,
)
from nuvion_app.inference.fleet_command import VerifiedFleetCommand
from nuvion_app.inference.motor import MotorCommand, MotorController

CAMERA_POSITION_COMMAND_TYPE = "CAMERA_POSITION_SET"
CAMERA_POSITION_CAPABILITY = "command.camera.position.set"

_DIRECTION_COMMANDS = {
    "LEFT": MotorCommand.LEFT,
    "RIGHT": MotorCommand.RIGHT,
    "UP": MotorCommand.UP,
    "DOWN": MotorCommand.DOWN,
}


class CameraPositionReconciler:
    """Acknowledged NUV1 steps, status, hold and persistent operator limits."""

    command_type = CAMERA_POSITION_COMMAND_TYPE
    capability = CAMERA_POSITION_CAPABILITY

    def __init__(self, controller: MotorController, *, stop_realtime: Callable[[], None] | None = None) -> None:
        self.controller = controller
        self.stop_realtime = stop_realtime

    @property
    def ready(self) -> bool:
        return self.controller.available and self.controller.protocol == "nuv1"

    def reconcile(self, command: VerifiedFleetCommand) -> CommandEffectOutcome:
        direction = str(command.payload.get("direction") or "")
        motor_command = _DIRECTION_COMMANDS.get(direction)
        if motor_command is None and direction not in {"STATUS", "STOP", "LIMITS"}:
            return CommandEffectOutcome(
                status=COMMAND_STATUS_FAILED,
                code="CAMERA_POSITION_INVALID_DIRECTION",
                message="unsupported camera control action",
            )
        try:
            if direction in {"STOP", "LIMITS"} and self.stop_realtime is not None:
                self.stop_realtime()
            if motor_command is not None:
                expiry = datetime.fromisoformat(command.expires_at.replace("Z", "+00:00"))
                if expiry <= datetime.now(timezone.utc):
                    raise ValueError("camera movement command has expired")
                response = self.controller.move_position(motor_command, expires_at=expiry.timestamp())
            else:
                response = self.controller.position_action(direction, command.payload.get("limits"))
            state = self._reported_state(direction, response)
            state["controlVersion"] = 2
            state["limits"] = dict(self.controller.position_limits)
            state["observedAt"] = datetime.now(timezone.utc).isoformat()
            return CommandEffectOutcome.succeeded(state)
        except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
            return CommandEffectOutcome(
                status=COMMAND_STATUS_FAILED,
                code="CAMERA_POSITION_APPLY_FAILED",
                message=str(exc)[:1000],
                reported_state={"direction": direction, "health": "CONTROL_FAILED"},
            )

    @staticmethod
    def _reported_state(
        direction: str, response: Mapping[str, Any]
    ) -> dict[str, Any]:
        positions: dict[str, int] = {}
        motors = response.get("motors")
        if isinstance(motors, list):
            for motor in motors:
                if not isinstance(motor, dict):
                    continue
                identifier = motor.get("id")
                position = motor.get("position")
                if (
                    isinstance(identifier, int)
                    and not isinstance(identifier, bool)
                    and isinstance(position, int)
                    and not isinstance(position, bool)
                    and identifier in {1, 2}
                    and motor.get("present") is True
                    and 0 <= position <= 4095
                ):
                    positions["pan" if identifier == 1 else "tilt"] = position
        if set(positions) != {"pan", "tilt"}:
            raise RuntimeError("both motor positions must be available")
        return {
            "direction": direction,
            "health": "FUNCTIONAL_HEALTHY",
            "protocol": "NUV1",
            "positions": positions,
        }
