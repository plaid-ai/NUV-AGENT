from __future__ import annotations

import unittest
from unittest import mock

from nuvion_app.inference import motor as motor_module


class FakeBackend(motor_module.BaseMotorBackend):
    def __init__(self) -> None:
        super().__init__()
        self.commands: list[motor_module.MotorCommand] = []

    def send_command(self, command: motor_module.MotorCommand) -> None:
        self.commands.append(command)


class MotorTest(unittest.TestCase):
    def test_motor_controller_throttles_commands_but_allows_repeat_after_interval(self) -> None:
        backend = FakeBackend()
        config = motor_module.MotorConfig(enabled=True, command_interval_sec=0.5)
        controller = motor_module.MotorController(config, backend=backend)

        with mock.patch("nuvion_app.inference.motor.time.time", side_effect=[10.0, 10.1, 10.7, 10.8, 11.4]):
            self.assertTrue(controller.send(motor_module.MotorCommand.LEFT))
            self.assertFalse(controller.send(motor_module.MotorCommand.LEFT))
            self.assertTrue(controller.send(motor_module.MotorCommand.LEFT))
            self.assertFalse(controller.send(motor_module.MotorCommand.RIGHT))
            self.assertTrue(controller.send(motor_module.MotorCommand.RIGHT))

        self.assertEqual(
            backend.commands,
            [
                motor_module.MotorCommand.LEFT,
                motor_module.MotorCommand.LEFT,
                motor_module.MotorCommand.RIGHT,
            ],
        )

    def test_pan_and_tilt_invert_are_applied(self) -> None:
        backend = FakeBackend()
        config = motor_module.MotorConfig(enabled=True, pan_invert=True, tilt_invert=True, command_interval_sec=0.0)
        controller = motor_module.MotorController(config, backend=backend)

        controller.send_pan(motor_module.MotorCommand.LEFT)
        controller.send_tilt(motor_module.MotorCommand.UP)

        self.assertEqual(
            backend.commands,
            [motor_module.MotorCommand.RIGHT, motor_module.MotorCommand.DOWN],
        )

    def test_pan_and_tilt_have_independent_rate_limits(self) -> None:
        backend = FakeBackend()
        config = motor_module.MotorConfig(enabled=True, command_interval_sec=0.5)
        controller = motor_module.MotorController(config, backend=backend)

        with mock.patch("nuvion_app.inference.motor.time.time", side_effect=[10.0, 10.0, 10.1, 10.1, 10.7, 10.7]):
            self.assertTrue(controller.send_pan(motor_module.MotorCommand.LEFT))
            self.assertTrue(controller.send_tilt(motor_module.MotorCommand.UP))
            self.assertFalse(controller.send_pan(motor_module.MotorCommand.LEFT))
            self.assertFalse(controller.send_tilt(motor_module.MotorCommand.UP))
            self.assertTrue(controller.send_pan(motor_module.MotorCommand.LEFT))
            self.assertTrue(controller.send_tilt(motor_module.MotorCommand.UP))

        self.assertEqual(
            backend.commands,
            [
                motor_module.MotorCommand.LEFT,
                motor_module.MotorCommand.UP,
                motor_module.MotorCommand.LEFT,
                motor_module.MotorCommand.UP,
            ],
        )

    def test_build_motor_backend_returns_noop_when_serial_unavailable(self) -> None:
        config = motor_module.MotorConfig(enabled=True, backend="uart")
        with mock.patch.object(motor_module, "serial", None):
            backend = motor_module.build_motor_backend(config)
        self.assertFalse(backend.available)

    def test_run_motor_test_sends_expected_commands(self) -> None:
        backend = FakeBackend()
        controller = motor_module.MotorController(
            motor_module.MotorConfig(enabled=True, command_interval_sec=0.0),
            backend=backend,
        )
        keys = iter(
            [
                motor_module.MotorTestKey.LEFT.value,
                motor_module.MotorTestKey.CENTER.value,
                motor_module.MotorTestKey.QUIT.value,
            ]
        )
        outputs: list[str] = []
        motor_module.run_motor_test(
            controller,
            key_reader=lambda: next(keys),
            printer=outputs.append,
        )

        self.assertIn("send: L", outputs)
        self.assertIn("send: C", outputs)
        self.assertEqual(
            backend.commands,
            [motor_module.MotorCommand.LEFT, motor_module.MotorCommand.CENTER],
        )


if __name__ == "__main__":
    unittest.main()
