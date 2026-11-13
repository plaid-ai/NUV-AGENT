"""Track model regions against the actual frames of a recorded event clip."""
from __future__ import annotations

import json
import logging
import subprocess
from bisect import bisect_left, bisect_right

import numpy as np

log = logging.getLogger(__name__)
TRACK_WINDOW_MS = 1500
MAX_FRAMES = 180


def _frame_times_ms(ffprobe: str, path: str) -> list[int]:
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_frames",
         "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", path],
        capture_output=True, check=True, timeout=8,
    )
    times = [round(float(frame["best_effort_timestamp_time"]) * 1000)
             for frame in json.loads(result.stdout)["frames"]]
    if not times or times[0] < 0 or any(a >= b for a, b in zip(times, times[1:])):
        return []
    return times


def _follow(cv, frames, times, anchor, box, step, output):
    height, width = frames[anchor].shape
    x, y, w, h = box
    # Pick features inside the region and its immediate surroundings, preferring
    # the brighter part over the darker, textured conveyor. Never reacquire.
    left, top = max(0, int(x - w * .45)), max(0, int(y - h * .45))
    right, bottom = min(width, int(x + w * 1.45)), min(height, int(y + h * 1.45))
    reference = frames[anchor]
    patch = reference[top:bottom, left:right]
    if patch.size < 25:
        return
    low = float(np.percentile(patch, 50))
    high = float(np.percentile(patch, 94))
    if high - low < 12:
        return
    mask = np.zeros_like(reference)
    mask[top:bottom, left:right] = (patch >= low + .36 * (high - low)) * 255
    points = cv.goodFeaturesToTrack(reference, maxCorners=48, qualityLevel=.018,
                                    minDistance=4, mask=mask)
    if points is None or len(points) < 5:
        return
    original = points.copy()
    corners = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]], dtype=np.float32)
    previous = anchor
    for current in range(anchor + step, len(frames) if step > 0 else -1, step):
        if abs(times[current] - times[previous]) > 120:
            break
        new, ok, error = cv.calcOpticalFlowPyrLK(
            frames[previous], frames[current], points, None, winSize=(21, 21), maxLevel=3,
        )
        if new is None:
            break
        reverse, back_ok, _ = cv.calcOpticalFlowPyrLK(
            frames[current], frames[previous], new, None, winSize=(21, 21), maxLevel=3,
        )
        if reverse is None:
            break
        dx = new[:, 0, :]
        keep = ((ok[:, 0] != 0) & (back_ok[:, 0] != 0) & (error[:, 0] < 22)
                & (np.linalg.norm(points[:, 0] - reverse[:, 0], axis=1) < 1.25)
                & (dx[:, 0] >= 2) & (dx[:, 0] < width - 2)
                & (dx[:, 1] >= 2) & (dx[:, 1] < height - 2))
        if keep.sum() < max(4, len(points) * .48):
            break
        new, original = new[keep], original[keep]
        affine, inliers = cv.estimateAffinePartial2D(original, new, ransacReprojThreshold=2)
        if affine is None:
            break
        keep = inliers[:, 0].astype(bool)
        scale = np.linalg.norm(affine[:, 0])
        if keep.sum() < max(4, len(new) * .6) or not .75 <= scale <= 1.3:
            break
        new, original = new[keep], original[keep]
        # Do not leave the region on the belt after its features disappear.
        origin_i = np.rint(original[:, 0]).astype(int)
        target_i = np.rint(new[:, 0]).astype(int)
        origin_values = reference[origin_i[:, 1], origin_i[:, 0]].astype(float)
        target_values = frames[current][target_i[:, 1], target_i[:, 0]].astype(float)
        brightness_delta = target_values - origin_values
        shift = np.median(brightness_delta)
        # The same steel surface can brighten as it moves under ambient light.
        # Permit a shared brightness shift while still rejecting a changed pattern.
        if abs(shift) > 60 or np.median(np.abs(brightness_delta - shift)) > 35:
            break
        transformed = corners @ affine[:, :2].T + affine[:, 2]
        a, b = transformed.min(axis=0), transformed.max(axis=0)
        center = (a + b) / 2
        if not (0 <= center[0] < width and 0 <= center[1] < height):
            break
        lo = np.maximum(a, 0)
        hi = np.minimum(b, (width, height))
        if np.prod(hi - lo) < .5 * np.prod(b - a):
            break
        output[current].append(_normalized(lo, hi, width, height))
        points, previous = new, current


def _normalized(lo, hi, width, height):
    return [round(float(value), 6)
            for value in (lo[0] / width, lo[1] / height,
                          (hi[0] - lo[0]) / width, (hi[1] - lo[1]) / height)]


def build_clip_track(path: str, *, ffmpeg: str, ffprobe: str, event_ms: int,
                     frame_rgb: np.ndarray, regions: tuple) -> dict | None:
    """Return presentation timestamps and normalized boxes for confident frames."""
    import cv2 as cv

    height, width = frame_rgb.shape[:2]
    ratio = min(1, 640 / max(width, height))
    width, height = round(width * ratio), round(height * ratio)
    timestamps = _frame_times_ms(ffprobe, path)
    first = bisect_left(timestamps, event_ms - TRACK_WINDOW_MS)
    end = bisect_right(timestamps, event_ms + TRACK_WINDOW_MS)
    if not 0 < end - first <= MAX_FRAMES:
        return None
    result = subprocess.run(
        [ffmpeg, "-v", "error", "-nostdin", "-threads", "2", "-i", path,
         "-filter_threads", "1", "-map", "0:v:0", "-vf",
         f"select=between(n\\,{first}\\,{end-1}),scale={width}:{height},format=gray",
         "-fps_mode", "passthrough", "-frames:v", str(end - first), "-an", "-threads", "1",
         "-f", "rawvideo", "pipe:1"],
        capture_output=True, check=True, timeout=12,
    )
    if len(result.stdout) != (end - first) * height * width:
        return None
    frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, height, width)
    times = timestamps[first:end]
    reference = cv.resize(cv.cvtColor(frame_rgb, cv.COLOR_RGB2GRAY), (width, height),
                          interpolation=cv.INTER_AREA).astype(np.int16)
    # The mux boundary gives a search hint. Match the image near the inferred
    # frame, since concat timing or a dropped recording frame can move the hint.
    candidates = [i for i, time in enumerate(times) if abs(time - event_ms) <= 220]
    if not candidates:
        return None
    areas = []
    for x, y, w, h in regions[:5]:
        left, top = max(0, int((x - w * .25) * width)), max(0, int((y - h * .25) * height))
        right = min(width, int((x + w * 1.25) * width))
        bottom = min(height, int((y + h * 1.25) * height))
        if right - left >= 5 and bottom - top >= 5:
            areas.append(np.s_[top:bottom, left:right])
    if not areas:
        return None

    def frame_error(index):
        return max(float(np.abs(frames[index][area] - reference[area]).mean()) for area in areas)

    anchor = min(candidates, key=lambda i: (frame_error(i), abs(times[i] - event_ms)))
    if frame_error(anchor) > 12:
        return None
    output = [[] for _ in times]
    for x, y, w, h in regions[:5]:
        x1, y1, x2, y2 = np.clip([x, y, x + w, y + h], 0, 1)
        if x2 <= x1 or y2 <= y1:
            continue
        output[anchor].append(_normalized((x1 * width, y1 * height),
                                          (x2 * width, y2 * height), width, height))
        for step in (-1, 1):
            _follow(cv, frames, times, anchor,
                    (x1 * width, y1 * height, (x2 - x1) * width, (y2 - y1) * height),
                    step, output)
    track_frames = [
        {"timeMs": time, "boxes": boxes} for time, boxes in zip(times, output)
        if boxes and abs(time - times[anchor]) <= TRACK_WINDOW_MS
    ]
    return {"eventTimeMs": times[anchor], "frames": track_frames} if track_frames else None
