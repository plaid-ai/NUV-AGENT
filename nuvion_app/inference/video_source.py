from __future__ import annotations

import os
import sys
from pathlib import Path


DEFAULT_DEMO_VIDEO_FILENAME = "exhibition-demo.webm"
DEFAULT_DEMO_VIDEO_PATHS = (
    Path("/var/lib/nuv-agent/demo") / DEFAULT_DEMO_VIDEO_FILENAME,
    Path("/opt/homebrew/var/nuv-agent/demo") / DEFAULT_DEMO_VIDEO_FILENAME,
    Path("/usr/local/var/nuv-agent/demo") / DEFAULT_DEMO_VIDEO_FILENAME,
)


def is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _validate_path(path: Path) -> Path:
    if not path.exists():
        raise ValueError(f"Demo video path does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Demo video path is not a file: {path}")
    if not os.access(path, os.R_OK):
        raise ValueError(f"Demo video path is not readable: {path}")
    return path.resolve()


def _fallback_paths_from_env() -> tuple[Path, ...]:
    raw = (os.getenv("NUVION_DEMO_VIDEO_FALLBACK_PATHS", "") or "").strip()
    if not raw:
        return ()
    paths: list[Path] = []
    for item in raw.split(","):
        value = item.strip()
        if not value:
            continue
        paths.append(Path(value).expanduser())
    return tuple(paths)


def resolve_demo_video_path(demo_video_path: str | None) -> Path:
    raw = (demo_video_path or "").strip()
    if raw:
        return _validate_path(Path(raw).expanduser())

    for path in (*_fallback_paths_from_env(), *DEFAULT_DEMO_VIDEO_PATHS):
        if not path.exists():
            continue
        try:
            return _validate_path(path)
        except ValueError:
            continue

    example_path = DEFAULT_DEMO_VIDEO_PATHS[0] if DEFAULT_DEMO_VIDEO_PATHS else Path("/var/lib/nuv-agent/demo/exhibition-demo.webm")
    raise ValueError(
        "NUVION_DEMO_VIDEO_PATH is required when NUVION_DEMO_MODE=true. "
        f"No fallback demo video found (expected e.g. {example_path})."
    )


def build_video_source_pipeline(
    video_source: str,
    width: int,
    height: int,
    fps: int,
    *,
    gst_source_override: str | None = None,
    demo_mode: bool = False,
    demo_video_path: str | None = None,
    platform_name: str | None = None,
    output_format: str | None = "RGB",
) -> str:
    if gst_source_override and gst_source_override.strip():
        return gst_source_override.strip()

    current_platform = platform_name or sys.platform

    if demo_mode:
        demo_path = resolve_demo_video_path(demo_video_path)
        uri = demo_path.as_uri()
        pipeline = (
            f'uridecodebin uri="{uri}" ! '
            "queue ! "
            "videoconvert ! "
            "videoscale ! "
            "videorate ! "
            f"video/x-raw,width={width},height={height},framerate={fps}/1 ! "
        )
        if output_format:
            pipeline = f"{pipeline} videoconvert ! video/x-raw,format={output_format}"
        return pipeline

    resolved_source = video_source
    if not resolved_source or resolved_source == "auto":
        resolved_source = "avf" if current_platform == "darwin" else "/dev/video0"

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

    pipeline = (
        f"{source} ! "
        f"video/x-raw,width={width},height={height},framerate={fps}/1 ! "
    )
    if output_format:
        pipeline = f"{pipeline} videoconvert ! video/x-raw,format={output_format}"
    return pipeline
