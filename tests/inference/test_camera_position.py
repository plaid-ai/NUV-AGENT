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
                {"id": 1, "position": 2011, "present": True},
                {"id": 2, "position": 2048, "present": True},
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
        expires_at="2099-09-22T00:00:30Z",
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



class CameraActionSafetyTest(unittest.TestCase):
    def test_continuous_encoder_position_is_measured_but_not_movement_ready(self):
        for position in (4395, -1, -(2**31), 2**31 - 1):
            response = FakeNuv1Backend().last_response
            response['motors'][1]['position'] = position
            state = CameraPositionReconciler._reported_state('STATUS', response)
            self.assertEqual(state['positions']['tilt'], position)
            self.assertTrue(state['positionReferenceRequired'])

    def test_malformed_encoder_position_is_rejected(self):
        for position in (True, 1.5, None, 2**31, -(2**31)-1):
            response = FakeNuv1Backend().last_response
            response['motors'][1]['position'] = position
            with self.assertRaises(RuntimeError):
                CameraPositionReconciler._reported_state('STATUS', response)

    def test_durable_stop_also_cancels_realtime_intent(self):
        from unittest.mock import Mock
        controller = Mock()
        controller.position_limits = dict(panMin=0, panMax=4095, tiltMin=0, tiltMax=4095)
        controller.position_action.return_value = FakeNuv1Backend().last_response
        stop = Mock()
        effect = CameraPositionReconciler(controller, stop_realtime=stop)
        effect.reconcile(command('STATUS'))
        stop.assert_not_called()
        effect.reconcile(command('STOP'))
        stop.assert_called_once()

    def test_expired_jog_is_not_executed(self):
        from dataclasses import replace
        from unittest.mock import Mock
        controller = Mock()
        result = CameraPositionReconciler(controller).reconcile(replace(command('RIGHT'), expires_at='2020-01-01T00:00:00Z'))
        self.assertEqual(result.status, 'FAILED')
        controller.move_position.assert_not_called()

    def test_missing_motor_does_not_report_healthy(self):
        backend = FakeNuv1Backend()
        backend.last_response['motors'][1]['present'] = False
        controller = MotorController(MotorConfig(enabled=True, backend='nuv1'), backend=backend)
        self.assertEqual(CameraPositionReconciler(controller).reconcile(command('RIGHT')).status, 'FAILED')

    def test_status_stop_and_limits_are_not_jogs(self):
        from dataclasses import replace
        from unittest.mock import Mock
        controller = Mock()
        controller.position_limits = dict(panMin=0, panMax=4095, tiltMin=0, tiltMax=4095)
        controller.position_action.return_value = FakeNuv1Backend().last_response
        for action in ('STATUS', 'STOP', 'LIMITS'):
            result = CameraPositionReconciler(controller).reconcile(replace(command(action), payload={'direction': action, **({'limits': controller.position_limits} if action == 'LIMITS' else {})}))
            self.assertEqual(result.status, 'SUCCEEDED')
            self.assertEqual(result.reported_state['limits'], controller.position_limits)
            self.assertEqual(result.reported_state['controlVersion'], 2)
        controller.move_position.assert_not_called()


if __name__ == "__main__":
    unittest.main()
