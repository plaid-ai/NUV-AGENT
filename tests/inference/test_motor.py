from __future__ import annotations

import unittest
import json
from unittest import mock

from nuvion_app.inference import motor as motor_module


class FakeBackend(motor_module.BaseMotorBackend):
    def __init__(self) -> None:
        super().__init__()
        self.commands: list[motor_module.MotorCommand] = []

    def send_command(self, command: motor_module.MotorCommand) -> None:
        self.commands.append(command)


class MotorTest(unittest.TestCase):
    def test_nuv1_backend_arms_then_sends_one_acknowledged_jog(self) -> None:
        class FakeSerial:
            def __init__(self) -> None:
                self.sent: list[bytes] = []
                self.responses: list[bytes] = []

            def reset_input_buffer(self) -> None:
                self.responses.clear()

            def write(self, data: bytes) -> int:
                self.sent.append(data)
                sequence = int(data.split()[1])
                command = data.decode().strip().split(" ", 2)[2]
                response = {
                    "protocol": "NUV1",
                    "seq": sequence,
                    "ok": True,
                    "armed": command != "STATUS",
                    "motors": [
                        {"id": 1, "position": 2000},
                        {"id": 2, "position": 2100},
                    ],
                }
                self.responses.append(json.dumps(response).encode())
                return len(data)

            def flush(self) -> None:
                return None

            def readline(self, _limit: int) -> bytes:
                return self.responses.pop(0) if self.responses else b""

            def close(self) -> None:
                return None

        fake = FakeSerial()
        serial_module = mock.Mock()
        serial_module.Serial.return_value = fake
        with mock.patch.object(motor_module, "serial", serial_module):
            backend = motor_module.Nuv1UartMotorBackend("/dev/test", 115200, 0.2)
        controller = motor_module.MotorController(
            motor_module.MotorConfig(enabled=True, backend="nuv1"), backend=backend
        )

        result = controller.move_position(motor_module.MotorCommand.LEFT)

        self.assertEqual(
            fake.sent,
            [b"NUV1 1 STATUS\n", b"NUV1 2 ARM\n", b"NUV1 3 JOG 1 -1\n"],
        )
        self.assertEqual(result["motors"][0]["position"], 2000)

    def test_nuv1_position_move_respects_mount_inversion(self) -> None:
        backend = mock.Mock(spec=motor_module.BaseMotorBackend)
        backend.available = True
        backend.protocol = "nuv1"
        backend.last_response = {"protocol": "NUV1", "motors": [{"id": 1, "position": 2000}]}
        controller = motor_module.MotorController(
            motor_module.MotorConfig(enabled=True, backend="nuv1", pan_invert=True),
            backend=backend,
        )

        controller.move_position(motor_module.MotorCommand.LEFT)

        backend.send_command.assert_called_once_with(motor_module.MotorCommand.RIGHT)

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
