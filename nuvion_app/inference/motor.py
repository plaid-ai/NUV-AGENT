from __future__ import annotations

import logging
import os
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from enum import Enum
from typing import Callable

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
    command_interval_sec: float = 0.1


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
        command_interval_sec=_env_float("NUVION_MOTOR_COMMAND_INTERVAL_SEC", 0.1),
    )


class BaseMotorBackend:
    def __init__(self) -> None:
        self.available = True
        self.reason = ""

    def send_command(self, command: MotorCommand) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return None


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


class PwmMotorBackend(NoOpMotorBackend):
    def __init__(self) -> None:
        super().__init__("PWM backend is not implemented yet.")


def build_motor_backend(config: MotorConfig) -> BaseMotorBackend:
    backend = config.backend
    if not config.enabled:
        return NoOpMotorBackend("Motor support is disabled.")

    if backend == "none":
        return NoOpMotorBackend("Motor backend is disabled.")

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
        self._last_sent_at = 0.0

    @property
    def available(self) -> bool:
        return self.backend.available

    @property
    def reason(self) -> str:
        return getattr(self.backend, "reason", "")

    def send(self, command: MotorCommand, *, force: bool = False) -> bool:
        if not self.config.enabled:
            return False
        if not self.backend.available:
            return False

        now = time.time()
        with self._lock:
            if not force and now - self._last_sent_at < self.config.command_interval_sec:
                return False

            self.backend.send_command(command)
            self._last_sent_at = now
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
        return self.send(mapped)

    def send_tilt(self, command: MotorCommand | None) -> bool:
        if command is None:
            return False
        mapped = command
        if self.config.tilt_invert:
            if command == MotorCommand.UP:
                mapped = MotorCommand.DOWN
            elif command == MotorCommand.DOWN:
                mapped = MotorCommand.UP
        return self.send(mapped)

    def center(self) -> bool:
        return self.send(MotorCommand.CENTER)

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
