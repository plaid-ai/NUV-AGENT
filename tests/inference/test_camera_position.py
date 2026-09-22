from __future__ import annotations

import unittest

from nuvion_app.inference.camera_position import CameraPositionReconciler
from nuvion_app.inference.fleet_command import VerifiedFleetCommand
from nuvion_app.inference.motor import BaseMotorBackend, MotorConfig, MotorController


class FakeNuv1Backend(BaseMotorBackend):
    def __init__(self) -> None:
        super().__init__()
        self.last_response = {
            "protocol": "NUV1",
            "motors": [
                {"id": 1, "position": 2011},
                {"id": 2, "position": 2048},
            ],
        }

    @property
    def protocol(self) -> str:
        return "nuv1"

    def send_command(self, _command) -> None:
        return None


def command(direction: str) -> VerifiedFleetCommand:
    return VerifiedFleetCommand(
        command_id="00000000-0000-0000-0000-000000000001",
        device_id="ultra-1",
        space_id=1,
        command_type="CAMERA_POSITION_SET",
        schema_version=1,
        issued_at="2026-09-22T00:00:00Z",
        expires_at="2026-09-22T00:00:30Z",
        sequence=1,
        payload_base64="e30",
        payload_hash="0" * 64,
        payload={"direction": direction},
        actor="owner@example.com",
        authorization_context="SPACE_ADMIN",
        key_id="test",
        required_capability="command.camera.position.set",
        compact_jws="a.b.c",
    )


class CameraPositionReconcilerTest(unittest.TestCase):
    def test_reports_acknowledged_pan_and_tilt_positions(self) -> None:
        controller = MotorController(
            MotorConfig(enabled=True, backend="nuv1", command_interval_sec=0),
            backend=FakeNuv1Backend(),
        )
        result = CameraPositionReconciler(controller).reconcile(command("RIGHT"))
        self.assertEqual(result.status, "SUCCEEDED")
        self.assertEqual(result.reported_state["positions"], {"pan": 2011, "tilt": 2048})
        self.assertEqual(result.reported_state["protocol"], "NUV1")


if __name__ == "__main__":
    unittest.main()
