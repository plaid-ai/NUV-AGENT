from __future__ import annotations

import glob
import os
import time


def list_stable_segments(
    segment_dir: str,
    *,
    settle_sec: float,
    now_ts: float | None = None,
) -> list[str]:
    now_value = time.time() if now_ts is None else now_ts
    stable_before = now_value - max(0.0, settle_sec)

    pattern = os.path.join(segment_dir, "segment_*.mp4")
    segments = glob.glob(pattern)
    segments.sort(key=os.path.getmtime)

    return [segment for segment in segments if os.path.getmtime(segment) <= stable_before]


def collect_stable_segments(
    segment_dir: str,
    *,
    settle_sec: float,
    before: float | None = None,
    after: float | None = None,
    count: int = 5,
    now_ts: float | None = None,
) -> list[str]:
    segments = list_stable_segments(segment_dir, settle_sec=settle_sec, now_ts=now_ts)

    if before is not None:
        segments = [segment for segment in segments if os.path.getmtime(segment) <= before]
        return segments[-count:]
    if after is not None:
        segments = [segment for segment in segments if os.path.getmtime(segment) >= after]
        return segments[:count]
    return segments[-count:]
