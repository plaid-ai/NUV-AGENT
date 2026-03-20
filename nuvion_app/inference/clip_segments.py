from __future__ import annotations

import glob
import os
import subprocess
import time


def is_stable_mp4(path: str, ffprobe_path: str) -> bool:
    result = subprocess.run(
        [
            ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def list_stable_segments(
    segment_dir: str,
    *,
    settle_sec: float,
    ffprobe_path: str | None = None,
    probe_func: callable | None = None,
    now_ts: float | None = None,
) -> list[str]:
    now_value = time.time() if now_ts is None else now_ts
    stable_before = now_value - max(0.0, settle_sec)

    pattern = os.path.join(segment_dir, "segment_*.mp4")
    segments = glob.glob(pattern)
    segments.sort(key=os.path.getmtime)

    stable_segments = [segment for segment in segments if os.path.getmtime(segment) <= stable_before]
    if not ffprobe_path and probe_func is None:
        return stable_segments

    probe = probe_func or (lambda path: is_stable_mp4(path, ffprobe_path or "ffprobe"))
    return [segment for segment in stable_segments if probe(segment)]


def collect_stable_segments(
    segment_dir: str,
    *,
    settle_sec: float,
    ffprobe_path: str | None = None,
    probe_func: callable | None = None,
    before: float | None = None,
    after: float | None = None,
    count: int = 5,
    now_ts: float | None = None,
) -> list[str]:
    segments = list_stable_segments(
        segment_dir,
        settle_sec=settle_sec,
        ffprobe_path=ffprobe_path,
        probe_func=probe_func,
        now_ts=now_ts,
    )

    if before is not None:
        segments = [segment for segment in segments if os.path.getmtime(segment) <= before]
        return segments[-count:]
    if after is not None:
        segments = [segment for segment in segments if os.path.getmtime(segment) >= after]
        return segments[:count]
    return segments[-count:]
