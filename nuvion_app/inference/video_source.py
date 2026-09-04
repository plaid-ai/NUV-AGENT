from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from nuvion_app.inference.demo_mvtec import MvtecDemoSource
from nuvion_app.inference.demo_mvtec import prepare_mvtec_demo_source


DEPTHAI_SOURCE_ALIASES = frozenset({"depthai", "oak", "oak-d", "oak-d-lite"})
DEPTHAI_APPSRC_NAME = "oak_depthai_source"
DEPTHAI_APPSRC_MAX_BUFFERS = 2


def is_depthai_video_source(raw: str | None) -> bool:
    source = (raw or "").strip().lower()
    prefix = source.split(":", 1)[0]
    return prefix in DEPTHAI_SOURCE_ALIASES


def should_use_depthai_source(
    video_source: str | None,
    *,
    gst_source_override: str | None = None,
    demo_mode: bool = False,
) -> bool:
    """Return whether the effective pipeline needs the DepthAI frame bridge.

    A custom GStreamer source and demo mode both replace the configured camera
    source. Centralizing that precedence keeps construction, bridge setup, and
    doctor probes consistent.
    """
    if gst_source_override and gst_source_override.strip():
        return False
    if demo_mode:
        return False
    return is_depthai_video_source(video_source)


def depthai_device_id_from_source(raw: str | None) -> str | None:
    source = (raw or "").strip()
    prefix, separator, value = source.partition(":")
    if prefix.lower() not in DEPTHAI_SOURCE_ALIASES or not separator:
        return None
    normalized = value.strip()
    return normalized or None


def resolve_depthai_device_id(
    video_source: str | None,
    configured_device_id: str | None,
) -> str | None:
    inline_device_id = depthai_device_id_from_source(video_source)
    explicit_device_id = (configured_device_id or "").strip() or None
    if (
        inline_device_id is not None
        and explicit_device_id is not None
        and inline_device_id != explicit_device_id
    ):
        raise ValueError(
            "DepthAI device ID conflict between NUVION_VIDEO_SOURCE and "
            "NUVION_DEPTHAI_DEVICE_ID"
        )
    return explicit_device_id or inline_device_id


def is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def normalize_camera_preference(raw: str | None) -> str:
    value = (raw or "").strip().lower()
    if value in {"auto", "csi", "usb"}:
        return value
    return "auto"


def normalize_camera_wb_mode(raw: str | None) -> str:
    value = (raw or "").strip().lower()
    valid_modes = {
        "auto",
        "off",
        "incandescent",
        "fluorescent",
        "warm-fluorescent",
        "daylight",
        "cloudy-daylight",
        "twilight",
        "shade",
        "manual",
    }
    if value in valid_modes:
        return value
    return "auto"


def _normalize_rotation(raw: str | None) -> int:
    value = (raw or "").strip().lower()
    mapping = {
        "0": 0,
        "90": 90,
        "180": 180,
        "270": 270,
        "-90": 270,
        "-180": 180,
        "-270": 90,
    }
    return mapping.get(value, 0)


def _build_video_transform_chain() -> str:
    methods: list[str] = []
    if is_truthy(os.getenv("NUVION_VIDEO_FLIP_HORIZONTAL", "false")):
        methods.append("horizontal-flip")
    if is_truthy(os.getenv("NUVION_VIDEO_FLIP_VERTICAL", "false")):
        methods.append("vertical-flip")

    rotation = _normalize_rotation(os.getenv("NUVION_VIDEO_ROTATION", "0"))
    if rotation == 90:
        methods.append("clockwise")
    elif rotation == 180:
        methods.append("rotate-180")
    elif rotation == 270:
        methods.append("counterclockwise")

    return " ! ".join(f"videoflip method={method}" for method in methods)


def _append_video_transforms(pipeline: str) -> str:
    transforms = _build_video_transform_chain()
    if not transforms:
        return pipeline
    return f"{pipeline} ! videoconvert ! {transforms} ! video/x-raw,format=RGB"


@dataclass(frozen=True)
class LinuxVideoDeviceInfo:
    path: str
    name: str = ""
    driver: str = ""


def _video_device_sort_key(name: str) -> tuple[int, str]:
    suffix = name.removeprefix("video")
    if suffix.isdigit():
        return int(suffix), name
    return 10_000, name


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _read_linux_video_driver(sys_device_dir: Path) -> str:
    uevent = _read_text(sys_device_dir / "device" / "uevent")
    for line in uevent.splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "DRIVER":
            return value.strip()

    driver_link = sys_device_dir / "device" / "driver"
    try:
        return driver_link.resolve(strict=True).name
    except OSError:
        return ""


def _linux_video_devices(
    *,
    dev_root: str = "/dev",
    sys_class_root: str = "/sys/class/video4linux",
) -> list[LinuxVideoDeviceInfo]:
    devices: list[LinuxVideoDeviceInfo] = []
    sys_class_dir = Path(sys_class_root)
    for node in sorted(Path(dev_root).glob("video*"), key=lambda path: _video_device_sort_key(path.name)):
        sys_device_dir = sys_class_dir / node.name
        name = _read_text(sys_device_dir / "name")
        driver = _read_linux_video_driver(sys_device_dir)
        devices.append(LinuxVideoDeviceInfo(path=str(node), name=name, driver=driver))
    return devices


def _find_linux_video_device(path: str) -> LinuxVideoDeviceInfo | None:
    for device in _linux_video_devices():
        if device.path == path:
            return device
    return None


@lru_cache(maxsize=None)
def _gst_element_available(element_name: str) -> bool:
    gst_inspect = shutil.which("gst-inspect-1.0")
    if not gst_inspect:
        return False
    result = subprocess.run(
        [gst_inspect, element_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


@lru_cache(maxsize=1)
def _is_jetson_platform() -> bool:
    model_text = _read_text(Path("/proc/device-tree/model")).lower()
    if "jetson" in model_text or "nvidia" in model_text:
        return True
    return Path("/etc/nv_tegra_release").exists()


def _is_probable_jetson_csi_device(device: LinuxVideoDeviceInfo | None) -> bool:
    if device is None:
        return False
    name = device.name.lower()
    return any(marker in name for marker in ("vi-output", "tegra", "camrtc", "nvargus"))


def _is_probable_linux_camera_device(device: LinuxVideoDeviceInfo | None) -> bool:
    if device is None:
        return False

    metadata = f"{device.name} {device.driver}".strip().lower()
    if not metadata:
        return False

    codec_markers = (
        "codec",
        "decoder",
        "encoder",
        "m2m",
        "msm_vidc",
        "iris_vpu",
        "vicodec",
    )
    if any(marker in metadata for marker in codec_markers):
        return False

    if _is_probable_jetson_csi_device(device):
        return True

    camera_markers = (
        "uvcvideo",
        "uvc",
        "usb camera",
        "webcam",
        "camera",
        "capture",
    )
    return any(marker in metadata for marker in camera_markers)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _camera_preference() -> str:
    return normalize_camera_preference(os.getenv("NUVION_CAMERA_PREFERENCE", "auto"))


def _build_camera_balance_chain() -> str:
    brightness = _clamp(_env_float("NUVION_CAMERA_BRIGHTNESS", 0.0), -1.0, 1.0)
    contrast = _clamp(_env_float("NUVION_CAMERA_CONTRAST", 1.0), 0.0, 2.0)
    saturation = _clamp(_env_float("NUVION_CAMERA_SATURATION", 1.0), 0.0, 2.0)
    if brightness == 0.0 and contrast == 1.0 and saturation == 1.0:
        return ""
    return (
        "videobalance "
        f"brightness={brightness:.3f} "
        f"contrast={contrast:.3f} "
        f"saturation={saturation:.3f}"
    )


def _finalize_camera_pipeline(base_pipeline: str) -> str:
    balance = _build_camera_balance_chain()
    if balance:
        base_pipeline = f"{base_pipeline} ! {balance}"
    pipeline = f"{base_pipeline} ! video/x-raw,format=RGB"
    return _append_video_transforms(pipeline)


def _build_depthai_appsrc_pipeline(width: int, height: int, fps: int) -> str:
    max_bytes = DEPTHAI_APPSRC_MAX_BUFFERS * width * height * 3
    appsrc = (
        f"appsrc name={DEPTHAI_APPSRC_NAME} "
        "is-live=true format=time do-timestamp=true block=false emit-signals=false "
        f"max-buffers={DEPTHAI_APPSRC_MAX_BUFFERS} max-bytes={max_bytes} "
        "max-time=0 leaky-type=downstream "
        f"caps=video/x-raw,format=RGB,width={width},height={height},framerate={fps}/1"
    )
    return _finalize_camera_pipeline(f"{appsrc} ! videoconvert")


def _build_jetson_camera_source_prefix() -> str:
    sensor_id = _env_int("NUVION_JETSON_SENSOR_ID", 0)
    properties = [f"sensor-id={sensor_id}"]

    if not is_truthy(os.getenv("NUVION_CAMERA_AUTO_EXPOSURE", "true")):
        properties.append("aelock=true")

    exposure_compensation = _clamp(_env_float("NUVION_CAMERA_EXPOSURE_COMPENSATION", 0.0), -2.0, 2.0)
    if exposure_compensation != 0.0:
        properties.append(f"exposurecompensation={exposure_compensation:.3f}")

    if not is_truthy(os.getenv("NUVION_CAMERA_AUTO_WHITE_BALANCE", "true")):
        properties.append("awblock=true")

    white_balance_mode = normalize_camera_wb_mode(os.getenv("NUVION_CAMERA_WB_MODE", "auto"))
    if white_balance_mode != "auto":
        properties.append(f"wbmode={white_balance_mode}")

    return f"nvarguscamerasrc {' '.join(properties)}"


def _pick_linux_v4l2_device(
    devices: list[LinuxVideoDeviceInfo],
    *,
    preference: str,
) -> LinuxVideoDeviceInfo | None:
    camera_devices = [device for device in devices if _is_probable_linux_camera_device(device)]
    usb_devices = [device for device in camera_devices if not _is_probable_jetson_csi_device(device)]
    if preference == "usb" and usb_devices:
        return usb_devices[0]
    if preference == "usb":
        return None
    return camera_devices[0] if camera_devices else None


def _build_standard_camera_pipeline(source: str, width: int, height: int, fps: int) -> str:
    base_pipeline = (
        f"{source} ! "
        f"video/x-raw,width={width},height={height},framerate={fps}/1 ! "
        "videoconvert"
    )
    return _finalize_camera_pipeline(base_pipeline)


def _build_jetson_argus_pipeline(width: int, height: int, fps: int) -> str:
    capture_width = max(width, _env_int("NUVION_JETSON_CAPTURE_WIDTH", 1920))
    capture_height = max(height, _env_int("NUVION_JETSON_CAPTURE_HEIGHT", 1080))
    capture_fps = max(fps, _env_int("NUVION_JETSON_CAPTURE_FPS", 30))
    base_pipeline = (
        f"{_build_jetson_camera_source_prefix()} ! "
        f"video/x-raw(memory:NVMM),width={capture_width},height={capture_height},framerate={capture_fps}/1,format=NV12 ! "
        "nvvidconv ! "
        f"video/x-raw,width={width},height={height},format=BGRx ! "
        "videoconvert"
    )
    return _finalize_camera_pipeline(base_pipeline)


def _build_linux_camera_pipeline(video_source: str, width: int, height: int, fps: int) -> str:
    resolved_source = video_source.strip() if video_source else "auto"
    lowered_source = resolved_source.lower()
    linux_devices = _linux_video_devices()
    preference = _camera_preference()
    default_video_device = _pick_linux_v4l2_device(linux_devices, preference=preference)

    if lowered_source in {"jetson", "argus", "csi"}:
        if _gst_element_available("nvarguscamerasrc"):
            return _build_jetson_argus_pipeline(width, height, fps)
        return _build_standard_camera_pipeline("autovideosrc", width, height, fps)

    if lowered_source in {"rpi", "libcamera"}:
        return _build_standard_camera_pipeline("libcamerasrc", width, height, fps)

    if resolved_source.startswith("/dev/"):
        if _is_jetson_platform() and _gst_element_available("nvarguscamerasrc"):
            if resolved_source.startswith("/dev/video") and _is_probable_jetson_csi_device(_find_linux_video_device(resolved_source)):
                return _build_jetson_argus_pipeline(width, height, fps)
        return _build_standard_camera_pipeline(f"v4l2src device={resolved_source}", width, height, fps)

    if lowered_source == "auto":
        if preference != "usb" and _is_jetson_platform() and _gst_element_available("nvarguscamerasrc"):
            if any(_is_probable_jetson_csi_device(device) for device in linux_devices):
                return _build_jetson_argus_pipeline(width, height, fps)

        if default_video_device is not None:
            return _build_standard_camera_pipeline(f"v4l2src device={default_video_device.path}", width, height, fps)

        if preference == "usb" and _is_jetson_platform() and _gst_element_available("nvarguscamerasrc"):
            if any(_is_probable_jetson_csi_device(device) for device in linux_devices):
                return _build_jetson_argus_pipeline(width, height, fps)

        if preference != "usb" and _gst_element_available("libcamerasrc"):
            return _build_standard_camera_pipeline("libcamerasrc", width, height, fps)

        if preference == "usb" and _gst_element_available("libcamerasrc"):
            return _build_standard_camera_pipeline("libcamerasrc", width, height, fps)

    return _build_standard_camera_pipeline("autovideosrc", width, height, fps)


def build_video_source_pipeline(
    video_source: str,
    width: int,
    height: int,
    fps: int,
    *,
    gst_source_override: str | None = None,
    demo_mode: bool = False,
    platform_name: str | None = None,
    demo_source: MvtecDemoSource | None = None,
) -> str:
    if gst_source_override and gst_source_override.strip():
        return _append_video_transforms(gst_source_override.strip())

    current_platform = platform_name or sys.platform

    if demo_mode:
        mvtec_source = demo_source or prepare_mvtec_demo_source(
            base_url=os.getenv("NUVION_DEMO_MVTEC_BASE_URL"),
            categories=os.getenv("NUVION_DEMO_MVTEC_CATEGORIES"),
            cache_dir=os.getenv("NUVION_DEMO_MVTEC_CACHE_DIR"),
            image_duration_sec=float(os.getenv("NUVION_DEMO_IMAGE_DURATION_SEC", "1.0")),
        )
        pipeline = (
            f'multifilesrc location="{mvtec_source.stage_pattern}" '
            'index=0 loop=true '
            f'caps="{mvtec_source.slideshow_caps}" ! '
            f"{mvtec_source.decoder} ! "
            "videoconvert ! "
            "videoscale ! "
            "videorate ! "
            f"video/x-raw,width={width},height={height},framerate={fps}/1 ! "
            "videoconvert ! "
            "video/x-raw,format=RGB"
        )
        return _append_video_transforms(pipeline)

    if should_use_depthai_source(
        video_source,
        gst_source_override=gst_source_override,
        demo_mode=demo_mode,
    ):
        return _build_depthai_appsrc_pipeline(width, height, fps)

    resolved_source = video_source
    if not resolved_source or resolved_source == "auto":
        resolved_source = "avf" if current_platform == "darwin" else "auto"

    if current_platform.startswith("linux"):
        return _build_linux_camera_pipeline(resolved_source, width, height, fps)

    if resolved_source.startswith("/dev/video"):
        if current_platform == "darwin":
            source = "avfvideosrc"
        else:
            source = f"v4l2src device={resolved_source}"
    elif resolved_source.lower() in {"rpi", "libcamera"}:
        source = "libcamerasrc"
    elif resolved_source.lower().startswith(("avf", "avfoundation", "mac")):
        device_index = None
        if ":" in resolved_source:
            _, maybe_index = resolved_source.split(":", 1)
            if maybe_index.isdigit():
                device_index = int(maybe_index)
        source = f"avfvideosrc device-index={device_index}" if device_index is not None else "avfvideosrc"
    else:
        source = "autovideosrc"

    return _build_standard_camera_pipeline(source, width, height, fps)
