from __future__ import annotations

import json
import logging
from pathlib import Path
import tempfile
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


class MotorLimitReached(RuntimeError):
    """A bounded tracking step was blocked by operator limits."""


def validate_motor_limits(value: Mapping[str, Any]) -> dict[str, int]:
    keys = {"panMin", "panMax", "tiltMin", "tiltMax"}
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError("limits require panMin, panMax, tiltMin and tiltMax")
    if any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 4095 for v in value.values()):
        raise ValueError("motor limits must be encoder integers from 0 to 4095")
    if value["panMin"] >= value["panMax"] or value["tiltMin"] >= value["tiltMax"]:
        raise ValueError("motor minimum must be less than maximum")
    return dict(value)


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
        from nuvion_app.runtime.settings_overlay import resolve_settings_state_dir
        self.limits_path = Path(resolve_settings_state_dir(os.environ)) / "camera-limits.json"
        self.limits = {"panMin": 0, "panMax": 4095, "tiltMin": 0, "tiltMax": 4095}
        if self.limits_path.exists():
            self.limits = validate_motor_limits(json.loads(self.limits_path.read_text()))

        self._serial = serial.Serial(
            self.port,
            self.baud,
            timeout=min(self.timeout_sec, 0.1),
            write_timeout=min(self.timeout_sec, 0.5),
            exclusive=True,
        )
        self._sequence = 0
        self.last_response: dict[str, Any] | None = None
        self._run_command: MotorCommand | None = None

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

    def send_command(self, command: MotorCommand, *, expires_at: float | None = None,
                     max_lead: int | None = None, status: dict | None = None,
                     is_active: Callable[[], bool] = lambda: True) -> dict[str, Any]:
        encoded = self._COMMANDS.get(command)
        if encoded is None:
            raise ValueError(f"NUV1 does not support {command.name}")
        provided_status = status is not None
        status = status if provided_status else self._request("STATUS")
        if status.get("armed") is not True:
            if not is_active():
                raise ValueError("camera movement input cancelled")
            self._request("ARM")
            provided_status = False
        # STATUS may precede ARM: always check the goal that the firmware will use.
        if not provided_status:
            status = self._request("STATUS")
        axis = "pan" if command in {MotorCommand.LEFT, MotorCommand.RIGHT} else "tilt"
        identifier = 1 if axis == "pan" else 2
        step = -11 if command in {MotorCommand.LEFT, MotorCommand.DOWN} else 11
        motor = next((m for m in status.get("motors", []) if isinstance(m, dict) and m.get("id") == identifier), None)
        if motor is None or motor.get("present") is not True:
            raise RuntimeError("motor position unavailable")
        goal = motor.get("goal")
        position = motor.get("position")
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (goal, position)):
            raise RuntimeError("motor position unavailable")
        low, high = self.limits[axis + "Min"], self.limits[axis + "Max"]
        if not low <= position <= high or not low <= goal + step <= high:
            raise MotorLimitReached("configured camera movement limit reached")
        if not is_active() or (expires_at is not None and time.time() >= expires_at):
            raise ValueError("camera movement command has expired")
        # Do not build a queue of goals ahead of the physical servo. A held key
        # streams intent; slow servos simply catch up before another small step.
        if max_lead is not None and abs(goal + step - position) > max_lead:
            return status
        return self._request(encoded)

    def read_position(self) -> dict[str, Any]:
        return self._request("STATUS")

    def realtime_step(self, command: MotorCommand, *, expires_at: float,
                      is_active: Callable[[], bool] = lambda: True) -> dict[str, Any]:
        """Use smooth-v3's hardware deadman; never rewrite a running goal.

        RUN targets the firmware's position limits. Keep bounded JOG for custom
        software ranges until firmware can accept those bounds atomically.
        """
        def check_input():
            if time.time() >= expires_at or not is_active():
                raise ValueError("camera movement input expired or cancelled")

        check_input()
        active = getattr(self, "_run_command", None)
        if active is not None and active != command:
            self.stop_position()
        if getattr(self, "_run_command", None) == command:
            state = self.last_response or {}
            identifier = 1 if command in {MotorCommand.LEFT, MotorCommand.RIGHT} else 2
            if state.get("moving_id") != identifier or state.get("armed") is not True:
                raise RuntimeError("continuous camera movement stopped; release and press again")
            check_input()
            response = self._request("KEEP")
            if response.get("moving_id") != identifier or response.get("armed") is not True:
                raise RuntimeError("continuous camera movement stopped; release and press again")
            return response

        state = self.read_position()
        axis = "pan" if command in {MotorCommand.LEFT, MotorCommand.RIGHT} else "tilt"
        identifier = 1 if axis == "pan" else 2
        motor = next((m for m in state.get("motors", []) if m.get("id") == identifier), {})
        continuous = (state.get("firmware") == "NUV1-smooth-v3"
                      and motor.get("mode") == 3
                      and self.limits[axis + "Min"] == 0 and self.limits[axis + "Max"] == 4095)
        if not continuous:
            check_input()
            return self.send_command(command, expires_at=expires_at, max_lead=22,
                                     status=state, is_active=is_active)
        if state.get("armed") is not True:
            check_input()
            self._request("ARM")
            # ARM replies contain the pre-enable torque snapshot in smooth-v3.
            # Refresh once after arming, not on every KEEP.
            state = self.read_position()
        motor = next((m for m in state.get("motors", []) if m.get("id") == identifier), {})
        if (motor.get("present") is not True or motor.get("torque") != 1
                or motor.get("hardware_error") != 0 or state.get("moving_id") != 0):
            raise RuntimeError("motor is not ready for continuous camera movement")
        check_input()
        direction = -1 if command in {MotorCommand.LEFT, MotorCommand.DOWN} else 1
        # 4 * 0.229 rpm ~= 5.5 deg/s, firmware KEEP deadman remains 400ms.
        response = self._request(f"RUN {identifier} {direction} 4")
        if response.get("moving_id") != identifier:
            raise RuntimeError("continuous camera movement did not start")
        self._run_command = command
        return response

    def stop_position(self) -> dict[str, Any]:
        try:
            return self._request("STOP")
        finally:
            self._run_command = None

    def set_limits(self, limits: Mapping[str, Any]) -> dict[str, Any]:
        validated = validate_motor_limits(limits)
        status = self.stop_position()
        status = self._request("STATUS")
        for identifier, axis in ((1, "pan"), (2, "tilt")):
            motor = next((m for m in status.get("motors", []) if isinstance(m, dict) and m.get("id") == identifier), {})
            position = motor.get("position")
            if (motor.get("present") is not True or isinstance(position, bool)
                    or not isinstance(position, int)
                    or not validated[axis + "Min"] <= position <= validated[axis + "Max"]):
                raise ValueError("movement limits must contain the current motor positions")
        self.limits_path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=self.limits_path.parent, prefix=".camera-limits-")
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(validated, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.limits_path)
            self.limits = validated
        finally:
            Path(name).unlink(missing_ok=True)
        return status

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
            if now < getattr(self, "manual_control_until", 0.0):
                return False
            last_sent_at = self._last_sent_at.get(lane, 0.0)
            if not force and now - last_sent_at < self.config.command_interval_sec:
                return False

            try:
                self.backend.send_command(command)
            except MotorLimitReached:
                return False
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

    def move_position(self, command: MotorCommand, *, expires_at: float | None = None) -> Mapping[str, Any]:
        if self.protocol != "nuv1":
            raise RuntimeError("camera position control requires the NUV1 motor backend")
        mapped = command
        if self.config.pan_invert and command in {MotorCommand.LEFT, MotorCommand.RIGHT}:
            mapped = MotorCommand.RIGHT if command == MotorCommand.LEFT else MotorCommand.LEFT
        if self.config.tilt_invert and command in {MotorCommand.UP, MotorCommand.DOWN}:
            mapped = MotorCommand.DOWN if command == MotorCommand.UP else MotorCommand.UP
        if not self.config.enabled or not self.available:
            raise RuntimeError(self.reason or "motor unavailable")
        with self._lock:
            if time.time() < getattr(self, "manual_control_until", 0.0):
                raise RuntimeError("manual WebSocket camera control is active")
            if expires_at is not None and time.time() >= expires_at:
                raise ValueError("camera movement command has expired")
            if isinstance(self.backend, Nuv1UartMotorBackend):
                self.backend.send_command(mapped, expires_at=expires_at)
            else:
                self.backend.send_command(mapped)
            reader = getattr(self.backend, "read_position", None)
            response = getattr(self.backend, "last_response", None)
            if callable(reader):
                deadline = time.monotonic() + 1.0
                while True:
                    response = reader()
                    motors = response.get("motors", [])
                    if len(motors) == 2 and all(
                        isinstance(m, dict) and m.get("present") is True
                        and isinstance(m.get("position"), int) and isinstance(m.get("goal"), int)
                        and abs(m["position"] - m["goal"]) <= 2 for m in motors
                    ):
                        break
                    if time.monotonic() >= deadline:
                        self.backend.stop_position()
                        raise TimeoutError("motor did not reach the requested position")
                    time.sleep(0.025)
            if not isinstance(response, dict):
                raise RuntimeError("OpenRB did not provide acknowledged motor telemetry")
            return dict(response)

    def realtime_step(self, command: MotorCommand, *, expires_at: float,
                      is_active: Callable[[], bool] = lambda: True) -> Mapping[str, Any]:
        """Renew continuous movement, with a bounded-jog compatibility path."""
        if not self.config.enabled or not self.available or self.protocol != "nuv1":
            raise RuntimeError("camera position control requires the NUV1 motor backend")
        mapped = command
        if self.config.pan_invert and command in {MotorCommand.LEFT, MotorCommand.RIGHT}:
            mapped = MotorCommand.RIGHT if command == MotorCommand.LEFT else MotorCommand.LEFT
        if self.config.tilt_invert and command in {MotorCommand.UP, MotorCommand.DOWN}:
            mapped = MotorCommand.DOWN if command == MotorCommand.UP else MotorCommand.UP
        with self._lock:
            if time.time() >= expires_at or not is_active():
                raise ValueError("camera movement input expired or cancelled")
            return self.backend.realtime_step(mapped, expires_at=expires_at, is_active=is_active)

    def position_action(self, action: str, limits: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        if not self.config.enabled or not self.available or self.protocol != "nuv1":
            raise RuntimeError("camera position control requires the NUV1 motor backend")
        with self._lock:
            if action == "STATUS":
                return self.backend.read_position()
            if action == "STOP":
                return self.backend.stop_position()
            if action == "LIMITS":
                return self.backend.set_limits(limits or {})
            raise ValueError("unknown camera action")

    @property
    def position_limits(self) -> Mapping[str, int]:
        return dict(getattr(self.backend, "limits", {}))

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
