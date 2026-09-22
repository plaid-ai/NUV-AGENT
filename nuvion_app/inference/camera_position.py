from __future__ import annotations

from collections.abc import Mapping
from typing import Any

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
    """Executes one bounded, acknowledged NUV1 pan/tilt step per Fleet command."""

    command_type = CAMERA_POSITION_COMMAND_TYPE
    capability = CAMERA_POSITION_CAPABILITY

    def __init__(self, controller: MotorController) -> None:
        self.controller = controller

    @property
    def ready(self) -> bool:
        return self.controller.available and self.controller.protocol == "nuv1"

    def reconcile(self, command: VerifiedFleetCommand) -> CommandEffectOutcome:
        direction = str(command.payload.get("direction") or "")
        motor_command = _DIRECTION_COMMANDS.get(direction)
        if motor_command is None:
            return CommandEffectOutcome(
                status=COMMAND_STATUS_FAILED,
                code="CAMERA_POSITION_INVALID_DIRECTION",
                message="direction must be LEFT, RIGHT, UP or DOWN",
            )
        try:
            response = self.controller.move_position(motor_command)
            return CommandEffectOutcome.succeeded(
                self._reported_state(direction, response)
            )
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
                ):
                    positions["pan" if identifier == 1 else "tilt"] = position
        return {
            "direction": direction,
            "health": "FUNCTIONAL_HEALTHY",
            "protocol": "NUV1",
            "positions": positions,
        }
