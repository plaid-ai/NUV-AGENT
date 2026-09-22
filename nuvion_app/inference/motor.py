from __future__ import annotations

import json
import logging
import os
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping

log = logging.getLogger(__name__)

try:
    import serial
except Exception as exc:  # pragma: no cover - import availability depends on runtime image
    serial = None
    _SERIAL_IMPORT_ERROR = exc
else:
    _SERIAL_IMPORT_ERROR = None


class MotorCommand(str, Enum):
    LEFT = "L"
    RIGHT = "R"
    UP = "U"
    DOWN = "D"
    CENTER = "C"

    @property
    def payload(self) -> bytes:
        return self.value.encode("ascii")


class MotorTestKey(str, Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP = "UP"
    DOWN = "DOWN"
    CENTER = "CENTER"
    QUIT = "QUIT"
    ESC = "ESC"


@dataclass(frozen=True)
class MotorConfig:
    enabled: bool = False
    backend: str = "auto"
    uart_port: str = "/dev/ttyTHS1"
    uart_baud: int = 115200
    uart_timeout_sec: float = 1.0
    pan_invert: bool = False
    tilt_invert: bool = False
    command_interval_sec: float = 0.05


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        parsed = float(raw)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def motor_config_from_env() -> MotorConfig:
    return MotorConfig(
        enabled=_truthy(os.getenv("NUVION_MOTOR_ENABLED", "false")),
        backend=(os.getenv("NUVION_MOTOR_BACKEND", "auto") or "auto").strip().lower() or "auto",
        uart_port=(os.getenv("NUVION_MOTOR_UART_PORT", "/dev/ttyTHS1") or "/dev/ttyTHS1").strip(),
        uart_baud=_env_int("NUVION_MOTOR_UART_BAUD", 115200),
        uart_timeout_sec=_env_float("NUVION_MOTOR_UART_TIMEOUT_SEC", 1.0),
        pan_invert=_truthy(os.getenv("NUVION_MOTOR_PAN_INVERT", "false")),
        tilt_invert=_truthy(os.getenv("NUVION_MOTOR_TILT_INVERT", "false")),
        command_interval_sec=_env_float("NUVION_MOTOR_COMMAND_INTERVAL_SEC", 0.05),
    )


class BaseMotorBackend:
    def __init__(self) -> None:
        self.available = True
        self.reason = ""

    def send_command(self, command: MotorCommand) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return None

    @property
    def protocol(self) -> str:
        return "legacy"


class NoOpMotorBackend(BaseMotorBackend):
    def __init__(self, reason: str = "") -> None:
        super().__init__()
        self.available = False
        self.reason = reason

    def send_command(self, command: MotorCommand) -> None:
        return None


class UartMotorBackend(BaseMotorBackend):
    def __init__(self, port: str, baud: int, timeout_sec: float) -> None:
        super().__init__()
        if serial is None:
            raise RuntimeError(f"pyserial unavailable: {_SERIAL_IMPORT_ERROR}")
        self.port = port
        self.baud = baud
        self.timeout_sec = timeout_sec
        self._serial = serial.Serial(self.port, self.baud, timeout=self.timeout_sec)

    def send_command(self, command: MotorCommand) -> None:
        self._serial.write(command.payload)
        self._serial.flush()

    def close(self) -> None:
        try:
            self._serial.close()
        except Exception:
            pass


class Nuv1UartMotorBackend(BaseMotorBackend):
    """Acknowledged OpenRB NUV1 bridge used by the Nuvion Ultra pan/tilt head."""

    _COMMANDS = {
        MotorCommand.LEFT: "JOG 1 -1",
        MotorCommand.RIGHT: "JOG 1 1",
        MotorCommand.UP: "JOG 2 1",
        MotorCommand.DOWN: "JOG 2 -1",
    }

    def __init__(self, port: str, baud: int, timeout_sec: float) -> None:
        super().__init__()
        if serial is None:
            raise RuntimeError(f"pyserial unavailable: {_SERIAL_IMPORT_ERROR}")
        self.port = port
        self.baud = baud
        self.timeout_sec = timeout_sec
        self._serial = serial.Serial(
            self.port,
            self.baud,
            timeout=min(self.timeout_sec, 0.1),
            write_timeout=min(self.timeout_sec, 0.5),
            exclusive=True,
        )
        self._sequence = 0
        self.last_response: dict[str, Any] | None = None

    @property
    def protocol(self) -> str:
        return "nuv1"

    def _request(self, command: str) -> dict[str, Any]:
        self._sequence += 1
        sequence = self._sequence
        self._serial.reset_input_buffer()
        self._serial.write(f"NUV1 {sequence} {command}\n".encode("ascii"))
        self._serial.flush()
        deadline = time.monotonic() + max(0.2, self.timeout_sec)
        while time.monotonic() < deadline:
            raw = self._serial.readline(2048)
            if not raw:
                continue
            try:
                response = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(response, dict):
                continue
            if response.get("protocol") != "NUV1" or response.get("seq") != sequence:
                continue
            self.last_response = dict(response)
            if response.get("ok") is not True:
                reason = str(response.get("error") or "OpenRB command rejected")
                raise RuntimeError(reason)
            return dict(response)
        raise TimeoutError("OpenRB NUV1 response timed out; command was not retried")

    def send_command(self, command: MotorCommand) -> None:
        encoded = self._COMMANDS.get(command)
        if encoded is None:
            raise ValueError(f"NUV1 does not support {command.name}")
        status = self._request("STATUS")
        if status.get("armed") is not True:
            self._request("ARM")
        self._request(encoded)

    def close(self) -> None:
        try:
            self._serial.close()
        except Exception:
            pass


class PwmMotorBackend(NoOpMotorBackend):
    def __init__(self) -> None:
        super().__init__("PWM backend is not implemented yet.")


def build_motor_backend(config: MotorConfig) -> BaseMotorBackend:
    backend = config.backend
    if not config.enabled:
        return NoOpMotorBackend("Motor support is disabled.")

    if backend == "none":
        return NoOpMotorBackend("Motor backend is disabled.")

    if backend == "nuv1":
        try:
            return Nuv1UartMotorBackend(
                config.uart_port, config.uart_baud, config.uart_timeout_sec
            )
        except Exception as exc:
            log.warning("[MOTOR] NUV1 UART backend unavailable: %s", exc)
            return NoOpMotorBackend(str(exc))

    if backend in {"auto", "uart"}:
        try:
            return UartMotorBackend(config.uart_port, config.uart_baud, config.uart_timeout_sec)
        except Exception as exc:
            log.warning("[MOTOR] UART backend unavailable: %s", exc)
            return NoOpMotorBackend(str(exc))

    if backend == "pwm":
        log.warning("[MOTOR] PWM backend requested but not implemented. Falling back to NoOp.")
        return PwmMotorBackend()

    log.warning("[MOTOR] Unknown backend '%s'. Falling back to NoOp.", backend)
    return NoOpMotorBackend(f"Unsupported motor backend: {backend}")


class MotorController:
    def __init__(self, config: MotorConfig, backend: BaseMotorBackend | None = None) -> None:
        self.config = config
        self.backend = backend or build_motor_backend(config)
        self._lock = threading.Lock()
        self._last_sent_at: dict[str, float] = {}

    @property
    def available(self) -> bool:
        return self.backend.available

    @property
    def reason(self) -> str:
        return getattr(self.backend, "reason", "")

    @property
    def protocol(self) -> str:
        return self.backend.protocol

    def send(self, command: MotorCommand, *, force: bool = False, lane: str = "generic") -> bool:
        if not self.config.enabled:
            return False
        if not self.backend.available:
            return False

        now = time.time()
        with self._lock:
            last_sent_at = self._last_sent_at.get(lane, 0.0)
            if not force and now - last_sent_at < self.config.command_interval_sec:
                return False

            self.backend.send_command(command)
            self._last_sent_at[lane] = now
            return True

    def send_pan(self, command: MotorCommand | None) -> bool:
        if command is None:
            return False
        mapped = command
        if self.config.pan_invert:
            if command == MotorCommand.LEFT:
                mapped = MotorCommand.RIGHT
            elif command == MotorCommand.RIGHT:
                mapped = MotorCommand.LEFT
        return self.send(mapped, lane="pan")

    def send_tilt(self, command: MotorCommand | None) -> bool:
        if command is None:
            return False
        mapped = command
        if self.config.tilt_invert:
            if command == MotorCommand.UP:
                mapped = MotorCommand.DOWN
            elif command == MotorCommand.DOWN:
                mapped = MotorCommand.UP
        return self.send(mapped, lane="tilt")

    def center(self) -> bool:
        return self.send(MotorCommand.CENTER)

    def move_position(self, command: MotorCommand) -> Mapping[str, Any]:
        if self.protocol != "nuv1":
            raise RuntimeError("camera position control requires the NUV1 motor backend")
        mapped = command
        if self.config.pan_invert and command in {MotorCommand.LEFT, MotorCommand.RIGHT}:
            mapped = MotorCommand.RIGHT if command == MotorCommand.LEFT else MotorCommand.LEFT
        if self.config.tilt_invert and command in {MotorCommand.UP, MotorCommand.DOWN}:
            mapped = MotorCommand.DOWN if command == MotorCommand.UP else MotorCommand.UP
        if not self.send(mapped, force=True, lane="camera_position"):
            raise RuntimeError(self.reason or "motor command was not sent")
        response = getattr(self.backend, "last_response", None)
        if not isinstance(response, dict):
            raise RuntimeError("OpenRB did not provide acknowledged motor telemetry")
        return dict(response)

    def close(self) -> None:
        self.backend.close()


def read_motor_test_key() -> str:
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch1 = sys.stdin.read(1)
        if ch1 == "\x1b":
            ch2 = sys.stdin.read(1)
            ch3 = sys.stdin.read(1)
            if ch2 == "[":
                if ch3 == "A":
                    return MotorTestKey.UP.value
                if ch3 == "B":
                    return MotorTestKey.DOWN.value
                if ch3 == "C":
                    return MotorTestKey.RIGHT.value
                if ch3 == "D":
                    return MotorTestKey.LEFT.value
            return MotorTestKey.ESC.value
        if ch1 == " ":
            return MotorTestKey.CENTER.value
        if ch1.lower() == "q":
            return MotorTestKey.QUIT.value
        return ch1
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def run_motor_test(
    controller: MotorController,
    *,
    key_reader: Callable[[], str] = read_motor_test_key,
    printer: Callable[[str], None] = print,
) -> None:
    try:
        printer("Motor test")
        printer("LEFT/RIGHT : motor1")
        printer("UP/DOWN    : motor2")
        printer("SPACE      : center")
        printer("q          : quit")

        if not controller.available:
            printer(f"Motor backend unavailable: {controller.reason or 'unknown reason'}")
            return
        if key_reader is read_motor_test_key and not sys.stdin.isatty():
            printer("Motor test requires an interactive terminal.")
            return

        while True:
            key = key_reader()
            if key == MotorTestKey.LEFT.value:
                controller.send(MotorCommand.LEFT, force=True)
                printer("send: L")
            elif key == MotorTestKey.RIGHT.value:
                controller.send(MotorCommand.RIGHT, force=True)
                printer("send: R")
            elif key == MotorTestKey.UP.value:
                controller.send(MotorCommand.UP, force=True)
                printer("send: U")
            elif key == MotorTestKey.DOWN.value:
                controller.send(MotorCommand.DOWN, force=True)
                printer("send: D")
            elif key == MotorTestKey.CENTER.value:
                controller.send(MotorCommand.CENTER, force=True)
                printer("send: C")
            elif key == MotorTestKey.QUIT.value:
                break
    finally:
        controller.close()
