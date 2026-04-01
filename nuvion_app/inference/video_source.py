from __future__ import annotations

import os
import sys

from nuvion_app.inference.demo_mvtec import MvtecDemoSource
from nuvion_app.inference.demo_mvtec import prepare_mvtec_demo_source


def is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
        return gst_source_override.strip()

    current_platform = platform_name or sys.platform

    if demo_mode:
        mvtec_source = demo_source or prepare_mvtec_demo_source(
            base_url=os.getenv("NUVION_DEMO_MVTEC_BASE_URL"),
            categories=os.getenv("NUVION_DEMO_MVTEC_CATEGORIES"),
            cache_dir=os.getenv("NUVION_DEMO_MVTEC_CACHE_DIR"),
            image_duration_sec=float(os.getenv("NUVION_DEMO_IMAGE_DURATION_SEC", "1.0")),
        )
        return (
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

    return (
        f"{source} ! "
        f"video/x-raw,width={width},height={height},framerate={fps}/1 ! "
        "videoconvert ! "
        "video/x-raw,format=RGB"
    )
