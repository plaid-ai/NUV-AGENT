from __future__ import annotations

import fcntl
import logging
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

CAMERA_PROFILE_AUTO = "auto"
CAMERA_PROFILE_GENERIC = "generic"
CAMERA_PROFILE_B0272 = "arducam_b0272"
CAMERA_PROFILE_B0273 = "arducam_b0273"
VALID_CAMERA_PROFILES = frozenset(
    {
        CAMERA_PROFILE_AUTO,
        CAMERA_PROFILE_GENERIC,
        CAMERA_PROFILE_B0272,
        CAMERA_PROFILE_B0273,
    }
)

FOCUS_MODE_STARTUP_LOCK = "startup-lock"
FOCUS_MODE_CONTINUOUS = "continuous"
FOCUS_MODE_MANUAL = "manual"
FOCUS_MODE_OFF = "off"
VALID_FOCUS_MODES = frozenset(
    {
        FOCUS_MODE_STARTUP_LOCK,
        FOCUS_MODE_CONTINUOUS,
        FOCUS_MODE_MANUAL,
        FOCUS_MODE_OFF,
    }
)

PLATFORM_RASPBERRY_PI = "raspberry_pi"
PLATFORM_JETSON = "jetson"
PLATFORM_GENERIC = "generic"

FOCUS_STATE_DISABLED = "DISABLED"
FOCUS_STATE_STARTUP_PENDING = "STARTUP_PENDING"
FOCUS_STATE_SCANNING = "SCANNING"
FOCUS_STATE_CONTINUOUS = "CONTINUOUS"
FOCUS_STATE_LOCKED = "LOCKED"
FOCUS_STATE_LOCKED_UNVERIFIED = "LOCKED_UNVERIFIED"
FOCUS_STATE_MANUAL = "MANUAL"
FOCUS_STATE_UNSUPPORTED = "UNSUPPORTED"
FOCUS_STATE_ERROR = "ERROR"

CAMERA_SOURCE_ELEMENT_NAME = "nuvion_camera_source"
EXPECTED_SENSOR_BY_PROFILE = {
    CAMERA_PROFILE_B0272: "imx477",
    CAMERA_PROFILE_B0273: "imx477",
}
PLATFORM_BY_PROFILE = {
    CAMERA_PROFILE_B0272: PLATFORM_RASPBERRY_PI,
    CAMERA_PROFILE_B0273: PLATFORM_JETSON,
}

_GST_AF_MODE_PROPERTIES = ("af-mode", "auto-focus-mode")
_GST_AF_STATE_PROPERTIES = ("af-state",)
_GST_LENS_POSITION_PROPERTIES = ("lens-position",)
_I2C_SLAVE = 0x0703
_ARDUCAM_FOCUS_I2C_ADDRESS = 0x0C
_JETSON_FOCUS_MIN = 0
_JETSON_FOCUS_MAX = 1000
_JETSON_FOCUS_STEP = 50


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def normalize_camera_profile(value: str | None) -> str:
    normalized = (value or CAMERA_PROFILE_AUTO).strip().lower()
    return normalized if normalized in VALID_CAMERA_PROFILES else CAMERA_PROFILE_AUTO


def normalize_focus_mode(value: str | None) -> str:
    normalized = (value or FOCUS_MODE_STARTUP_LOCK).strip().lower().replace("_", "-")
    return normalized if normalized in VALID_FOCUS_MODES else FOCUS_MODE_STARTUP_LOCK


def detect_camera_platform(
    *,
    model_path: str | Path = "/proc/device-tree/model",
    jetson_release_path: str | Path = "/etc/nv_tegra_release",
) -> str:
    try:
        model = (
            Path(model_path)
            .read_text(encoding="utf-8", errors="ignore")
            .strip()
            .lower()
        )
    except OSError:
        model = ""
    if "jetson" in model or "nvidia" in model or Path(jetson_release_path).exists():
        return PLATFORM_JETSON
    if "raspberry pi" in model:
        return PLATFORM_RASPBERRY_PI
    return PLATFORM_GENERIC


def resolve_camera_profile(
    raw_profile: str | None,
    *,
    video_source: str | None,
    platform_kind: str,
) -> str:
    profile = normalize_camera_profile(raw_profile)
    if profile != CAMERA_PROFILE_AUTO:
        return profile

    source = (video_source or "auto").strip().lower()
    if source in {"rpi", "libcamera"}:
        return CAMERA_PROFILE_B0272
    if source in {"jetson", "argus", "csi"}:
        return CAMERA_PROFILE_B0273
    if source == "auto" and platform_kind == PLATFORM_RASPBERRY_PI:
        return CAMERA_PROFILE_B0272
    if source == "auto" and platform_kind == PLATFORM_JETSON:
        return CAMERA_PROFILE_B0273
    return CAMERA_PROFILE_GENERIC


def _optional_float(value: str | None) -> float | None:
    normalized = (value or "").strip()
    if not normalized:
        return None
    try:
        parsed = float(normalized)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _positive_float(value: str | None, default: float) -> float:
    try:
        parsed = float((value or "").strip())
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _optional_nonnegative_int(value: str | None) -> int | None:
    normalized = (value or "").strip()
    if not normalized:
        return None
    try:
        parsed = int(normalized)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _positive_int(value: str | None, default: int) -> int:
    try:
        parsed = int((value or "").strip())
    except ValueError:
        return default
    return parsed if parsed > 0 else default


@dataclass(frozen=True)
class CameraControlConfig:
    profile: str
    platform_kind: str
    focus_mode: str
    focus_required: bool
    focus_settle_seconds: float
    focus_step_frames: int
    manual_lens_position: float | None
    i2c_bus: int | None

    @property
    def expected_sensor(self) -> str | None:
        return EXPECTED_SENSOR_BY_PROFILE.get(self.profile)

    @property
    def source_backend(self) -> str:
        if self.profile == CAMERA_PROFILE_B0272:
            return "libcamera"
        if self.profile == CAMERA_PROFILE_B0273:
            return "nvargus+i2c"
        return "generic"


def camera_control_config_from_env(
    video_source: str | None,
    *,
    environ: Mapping[str, str] | None = None,
    platform_kind: str | None = None,
) -> CameraControlConfig:
    values = os.environ if environ is None else environ
    resolved_platform = platform_kind or detect_camera_platform()
    profile = resolve_camera_profile(
        values.get("NUVION_CAMERA_PROFILE"),
        video_source=video_source,
        platform_kind=resolved_platform,
    )
    return CameraControlConfig(
        profile=profile,
        platform_kind=resolved_platform,
        focus_mode=normalize_focus_mode(values.get("NUVION_CAMERA_FOCUS_MODE")),
        focus_required=_truthy(values.get("NUVION_CAMERA_FOCUS_REQUIRED", "false")),
        focus_settle_seconds=_positive_float(
            values.get("NUVION_CAMERA_FOCUS_SETTLE_SEC"), 3.0
        ),
        focus_step_frames=_positive_int(
            values.get("NUVION_CAMERA_FOCUS_STEP_FRAMES"), 2
        ),
        manual_lens_position=_optional_float(
            values.get("NUVION_CAMERA_MANUAL_LENS_POSITION")
        ),
        i2c_bus=_optional_nonnegative_int(values.get("NUVION_CAMERA_I2C_BUS")),
    )


@dataclass(frozen=True)
class CameraHardwareProbe:
    sensor_model: str
    evidence: tuple[str, ...]


def probe_camera_hardware(
    *,
    sys_class_root: str | Path = "/sys/class/video4linux",
    i2c_root: str | Path = "/sys/bus/i2c/devices",
) -> CameraHardwareProbe:
    evidence: list[str] = []
    paths: list[Path] = []
    for root in (Path(sys_class_root), Path(i2c_root)):
        try:
            paths.extend(sorted(root.glob("*/name")))
        except OSError:
            continue
    for path in paths:
        try:
            value = path.read_text(encoding="utf-8", errors="ignore").strip()
        except OSError:
            continue
        if value:
            evidence.append(value[:160])

    combined = " ".join(evidence).lower()
    sensor_model = "imx477" if "imx477" in combined else "unknown"
    return CameraHardwareProbe(sensor_model=sensor_model, evidence=tuple(evidence[:16]))


def validate_camera_contract(
    config: CameraControlConfig,
    probe: CameraHardwareProbe,
) -> tuple[str, str]:
    expected_platform = PLATFORM_BY_PROFILE.get(config.profile)
    if expected_platform is not None and config.platform_kind != expected_platform:
        return (
            "fail",
            f"{config.profile} requires {expected_platform}, observed={config.platform_kind}",
        )
    expected_sensor = config.expected_sensor
    if expected_sensor and probe.sensor_model not in {expected_sensor, "unknown"}:
        return (
            "fail",
            f"{config.profile} requires {expected_sensor}, observed={probe.sensor_model}",
        )
    if expected_sensor and probe.sensor_model == "unknown":
        status = "fail" if config.focus_required else "warn"
        return status, f"{expected_sensor} sensor identity is not observable"
    if (
        config.profile == CAMERA_PROFILE_B0273
        and config.focus_mode != FOCUS_MODE_OFF
        and config.i2c_bus is None
    ):
        status = "fail" if config.focus_required else "warn"
        return status, "B0273 lens actuator I2C bus is not configured"
    if (
        config.profile == CAMERA_PROFILE_B0273
        and config.focus_mode == FOCUS_MODE_CONTINUOUS
    ):
        status = "fail" if config.focus_required else "warn"
        return status, "B0273 continuous focus is unavailable"
    if config.profile == CAMERA_PROFILE_GENERIC:
        return "pass", "generic camera profile"
    return "pass", f"{config.profile} hardware identity verified"


def _write_arducam_focus(i2c_bus: int, position: int) -> None:
    bounded = max(_JETSON_FOCUS_MIN, min(_JETSON_FOCUS_MAX, int(position)))
    encoded = (bounded << 4) & 0x3FF0
    device = f"/dev/i2c-{i2c_bus}"
    descriptor = os.open(device, os.O_RDWR)
    try:
        fcntl.ioctl(descriptor, _I2C_SLAVE, _ARDUCAM_FOCUS_I2C_ADDRESS)
        os.write(descriptor, bytes(((encoded >> 8) & 0x3F, encoded & 0xF0)))
    finally:
        os.close(descriptor)


def _sharpness_score(frame: np.ndarray) -> float:
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[0] < 3 or image.shape[1] < 3:
        raise ValueError("camera frame is too small for focus scoring")
    # Downsample before calculating the Laplacian variance. This preserves the
    # relative focus peak while keeping the appsink callback inexpensive.
    stride = max(1, min(image.shape[0], image.shape[1]) // 240)
    sampled = image[::stride, ::stride, :3].astype(np.float32, copy=False)
    gray = sampled.mean(axis=2)
    center = gray[1:-1, 1:-1]
    laplacian = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - (4.0 * center)
    )
    return float(np.var(laplacian))


class CameraController:
    """Apply platform camera focus policy and publish honest runtime state."""

    def __init__(
        self,
        config: CameraControlConfig,
        *,
        hardware_probe: CameraHardwareProbe | None = None,
        lens_writer: Callable[[int, int], None] = _write_arducam_focus,
        on_failure: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.hardware_probe = hardware_probe or probe_camera_hardware()
        self._lens_writer = lens_writer
        self._on_failure = on_failure
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._started = False
        self._worker: threading.Thread | None = None
        self._gst_source: Any | None = None
        self._gst_mode_property: str | None = None
        self._gst_state_property: str | None = None
        self._gst_lens_property: str | None = None
        self._focus_state = FOCUS_STATE_DISABLED
        self._lens_position: float | None = None
        self._focus_score: float | None = None
        self._focus_positions = tuple(
            range(_JETSON_FOCUS_MIN, _JETSON_FOCUS_MAX + 1, _JETSON_FOCUS_STEP)
        )
        self._focus_position_index = 0
        self._focus_frames_until_sample = 0
        self._focus_scores: list[tuple[int, float]] = []
        self._focus_started_monotonic: float | None = None
        self._error: str | None = None
        self._control_backend = "none"

    def configure(self, gst_source: Any | None) -> None:
        status, detail = validate_camera_contract(self.config, self.hardware_probe)
        if status == "fail":
            self._fail(detail, fatal=True)

        if (
            self.config.profile == CAMERA_PROFILE_GENERIC
            or self.config.focus_mode == FOCUS_MODE_OFF
        ):
            self._set_state(FOCUS_STATE_DISABLED)
            return
        if self.config.profile == CAMERA_PROFILE_B0272:
            self._configure_libcamera(gst_source)
            return
        if self.config.profile == CAMERA_PROFILE_B0273:
            self._configure_jetson_i2c()
            return
        self._unsupported(f"unsupported camera profile: {self.config.profile}")

    def start(self) -> None:
        with self._lock:
            if self._focus_state != FOCUS_STATE_STARTUP_PENDING or self._started:
                return
            self._started = True
            if self.config.profile == CAMERA_PROFILE_B0273:
                self._focus_state = FOCUS_STATE_SCANNING
                self._focus_position_index = 0
                self._focus_frames_until_sample = self.config.focus_step_frames
                self._focus_scores.clear()
                self._focus_started_monotonic = time.monotonic()
                initial_position = self._focus_positions[0]
            else:
                initial_position = None
        if initial_position is not None:
            try:
                self._write_jetson_lens(initial_position)
            except (OSError, RuntimeError, ValueError) as exc:
                self._fail(
                    f"unable to start Jetson focus scan: {exc}",
                    fatal=self.config.focus_required,
                )
            return
        with self._lock:
            self._worker = threading.Thread(
                target=self._complete_libcamera_startup_focus,
                name="nuvion-camera-focus",
                daemon=True,
            )
            self._worker.start()

    def observe_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            if (
                self.config.profile != CAMERA_PROFILE_B0273
                or self._focus_state != FOCUS_STATE_SCANNING
            ):
                return
            if self._focus_frames_until_sample > 0:
                self._focus_frames_until_sample -= 1
                return
            position = self._focus_positions[self._focus_position_index]

        try:
            score = _sharpness_score(frame)
            self._record_jetson_focus_score(position, score)
        except (OSError, RuntimeError, ValueError) as exc:
            self._fail(f"Jetson autofocus scan failed: {exc}")

    def close(self) -> None:
        self._stop.set()
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=min(1.0, self.config.focus_settle_seconds + 0.1))

    def capabilities(self) -> frozenset[str]:
        with self._lock:
            if self._focus_state in {
                FOCUS_STATE_UNSUPPORTED,
                FOCUS_STATE_ERROR,
                FOCUS_STATE_DISABLED,
            }:
                return frozenset()
            capabilities = {"camera.imx477", "camera.focus.manual"}
            if self.config.focus_mode in {
                FOCUS_MODE_STARTUP_LOCK,
                FOCUS_MODE_CONTINUOUS,
            }:
                capabilities.add("camera.autofocus")
            if self.config.focus_mode == FOCUS_MODE_STARTUP_LOCK:
                capabilities.add("camera.autofocus.startup_lock")
            return frozenset(capabilities)

    def blocks_runtime_health(self) -> bool:
        with self._lock:
            return self.config.focus_required and self._focus_state in {
                FOCUS_STATE_UNSUPPORTED,
                FOCUS_STATE_ERROR,
                FOCUS_STATE_LOCKED_UNVERIFIED,
            }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            result: dict[str, Any] = {
                "profile": self.config.profile,
                "platform": self.config.platform_kind,
                "sourceBackend": self.config.source_backend,
                "controlBackend": self._control_backend,
                "sensorModel": self.hardware_probe.sensor_model,
                "expectedSensor": self.config.expected_sensor,
                "sensorVerified": (
                    self.config.expected_sensor is None
                    or self.hardware_probe.sensor_model == self.config.expected_sensor
                ),
                "focusMode": self.config.focus_mode,
                "focusRequired": self.config.focus_required,
                "focusState": self._focus_state,
            }
            if self._lens_position is not None:
                result["lensPosition"] = self._lens_position
            if self._focus_score is not None:
                result["focusScore"] = round(self._focus_score, 4)
            if self.config.i2c_bus is not None:
                result["controlDevice"] = f"/dev/i2c-{self.config.i2c_bus}"
            if self._error:
                result["error"] = self._error
            return result

    def _configure_libcamera(self, gst_source: Any | None) -> None:
        if gst_source is None:
            self._unsupported("named libcamerasrc element is unavailable")
            return
        self._gst_source = gst_source
        self._gst_mode_property = self._first_gst_property(
            gst_source, _GST_AF_MODE_PROPERTIES
        )
        self._gst_state_property = self._first_gst_property(
            gst_source, _GST_AF_STATE_PROPERTIES
        )
        self._gst_lens_property = self._first_gst_property(
            gst_source, _GST_LENS_POSITION_PROPERTIES
        )
        if self._gst_mode_property is None:
            self._unsupported("libcamerasrc autofocus control is unavailable")
            return

        self._control_backend = "libcamera"
        if (
            self.config.focus_mode == FOCUS_MODE_STARTUP_LOCK
            and self._gst_mode_property != "af-mode"
        ):
            self._unsupported(
                "libcamerasrc runtime startup focus lock control is unavailable"
            )
            return
        if self.config.focus_mode in {FOCUS_MODE_STARTUP_LOCK, FOCUS_MODE_CONTINUOUS}:
            self._set_gst_property(self._gst_mode_property, 2)
            self._set_state(
                FOCUS_STATE_STARTUP_PENDING
                if self.config.focus_mode == FOCUS_MODE_STARTUP_LOCK
                else FOCUS_STATE_CONTINUOUS
            )
            return

        self._set_gst_property(self._gst_mode_property, 0)
        if self.config.focus_mode == FOCUS_MODE_MANUAL:
            if self.config.manual_lens_position is None:
                self._fail(
                    "manual focus requires NUVION_CAMERA_MANUAL_LENS_POSITION",
                    fatal=self.config.focus_required,
                )
                return
            if self._gst_lens_property is None:
                self._unsupported("libcamerasrc manual lens control is unavailable")
                return
            self._set_gst_property(
                self._gst_lens_property, self.config.manual_lens_position
            )
            self._lens_position = self.config.manual_lens_position
            self._set_state(FOCUS_STATE_MANUAL)

    def _configure_jetson_i2c(self) -> None:
        if self.config.i2c_bus is None:
            self._unsupported("B0273 requires NUVION_CAMERA_I2C_BUS")
            return
        device = Path(f"/dev/i2c-{self.config.i2c_bus}")
        if self._lens_writer is _write_arducam_focus and not os.access(
            device, os.R_OK | os.W_OK
        ):
            self._unsupported(f"Jetson lens I2C device is unavailable: {device}")
            return
        self._control_backend = "i2c"

        if self.config.focus_mode == FOCUS_MODE_STARTUP_LOCK:
            self._set_state(FOCUS_STATE_STARTUP_PENDING)
            return
        if self.config.focus_mode == FOCUS_MODE_CONTINUOUS:
            self._unsupported(
                "B0273 continuous focus is unavailable; use startup-lock or manual"
            )
            return

        if self.config.focus_mode == FOCUS_MODE_MANUAL:
            if self.config.manual_lens_position is None:
                self._fail(
                    "manual focus requires NUVION_CAMERA_MANUAL_LENS_POSITION",
                    fatal=self.config.focus_required,
                )
                return
            position = round(self.config.manual_lens_position)
            if position < _JETSON_FOCUS_MIN or position > _JETSON_FOCUS_MAX:
                self._fail(
                    "B0273 manual lens position must be between 0 and 1000",
                    fatal=self.config.focus_required,
                )
                return
            try:
                self._write_jetson_lens(position)
            except (OSError, RuntimeError, ValueError) as exc:
                self._fail(
                    f"unable to set B0273 manual focus: {exc}",
                    fatal=self.config.focus_required,
                )
                return
            self._lens_position = float(position)
            self._set_state(FOCUS_STATE_MANUAL)

    def _complete_libcamera_startup_focus(self) -> None:
        self._set_state(FOCUS_STATE_SCANNING)
        if self._stop.wait(self.config.focus_settle_seconds):
            return
        try:
            verified = self._lock_libcamera_focus()
            if not verified and self.config.focus_required:
                self._fail("libcamera startup focus result could not be verified")
                return
            self._set_state(
                FOCUS_STATE_LOCKED if verified else FOCUS_STATE_LOCKED_UNVERIFIED
            )
        except (OSError, RuntimeError, ValueError) as exc:
            self._fail(f"startup focus lock failed: {exc}")

    def _lock_libcamera_focus(self) -> bool:
        if self._gst_source is None or self._gst_mode_property is None:
            raise RuntimeError("libcamera source is unavailable")
        verified = False
        if self._gst_state_property is not None:
            try:
                state = int(self._gst_source.get_property(self._gst_state_property))
                verified = state == 2
                if state == 3:
                    raise RuntimeError("libcamera autofocus reported failure")
            except (TypeError, ValueError):
                verified = False
        if self._gst_lens_property is not None:
            try:
                position = float(self._gst_source.get_property(self._gst_lens_property))
                if position >= 0:
                    self._lens_position = position
            except (TypeError, ValueError):
                pass
        self._set_gst_property(self._gst_mode_property, 0)
        if self._gst_lens_property is not None and self._lens_position is not None:
            self._set_gst_property(self._gst_lens_property, self._lens_position)
        return verified

    def _record_jetson_focus_score(self, position: int, score: float) -> None:
        with self._lock:
            if self._focus_state != FOCUS_STATE_SCANNING:
                return
            self._focus_scores.append((position, score))
            elapsed = time.monotonic() - (self._focus_started_monotonic or 0.0)
            scan_finished = (
                self._focus_position_index + 1 >= len(self._focus_positions)
                or elapsed >= self.config.focus_settle_seconds
            )
            if scan_finished:
                best_position, best_score = max(
                    self._focus_scores, key=lambda sample: sample[1]
                )
                self._write_jetson_lens(best_position)
                self._lens_position = float(best_position)
                self._focus_score = best_score
                self._focus_state = FOCUS_STATE_LOCKED
                return

            self._focus_position_index += 1
            next_position = self._focus_positions[self._focus_position_index]
            self._write_jetson_lens(next_position)
            self._focus_frames_until_sample = self.config.focus_step_frames

    def _write_jetson_lens(self, position: int) -> None:
        if self.config.i2c_bus is None:
            raise RuntimeError("Jetson focus I2C bus is unavailable")
        self._lens_writer(self.config.i2c_bus, position)

    @staticmethod
    def _first_gst_property(source: Any, names: Sequence[str]) -> str | None:
        for name in names:
            try:
                if source.find_property(name) is not None:
                    return name
            except (AttributeError, TypeError):
                return None
        return None

    def _set_gst_property(self, name: str, value: float) -> None:
        if self._gst_source is None:
            raise RuntimeError("GStreamer camera source is unavailable")
        try:
            self._gst_source.set_property(name, value)
        except Exception as exc:  # PyGObject uses runtime-specific exception types.
            raise RuntimeError(f"unable to set {name}: {exc}") from exc

    def _unsupported(self, message: str) -> None:
        if self.config.focus_required:
            self._fail(message, fatal=True)
            return
        with self._lock:
            self._focus_state = FOCUS_STATE_UNSUPPORTED
            self._error = message

    def _fail(self, message: str, *, fatal: bool = False) -> None:
        with self._lock:
            self._focus_state = FOCUS_STATE_ERROR
            self._error = message[:500]
        if self.config.focus_required and self._on_failure is not None:
            try:
                self._on_failure(message)
            except Exception as exc:  # noqa: BLE001 - preserve the camera failure.
                log.error("camera failure callback failed: %s", exc)
        if fatal:
            raise RuntimeError(message)

    def _set_state(self, state: str) -> None:
        with self._lock:
            self._focus_state = state


def build_camera_controller(
    video_source: str | None,
    *,
    environ: Mapping[str, str] | None = None,
    platform_kind: str | None = None,
    on_failure: Callable[[str], None] | None = None,
) -> CameraController:
    return CameraController(
        camera_control_config_from_env(
            video_source,
            environ=environ,
            platform_kind=platform_kind,
        ),
        on_failure=on_failure,
    )
