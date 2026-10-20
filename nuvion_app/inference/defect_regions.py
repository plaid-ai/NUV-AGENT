"""Display-only localization from model pixel scores, never dataset masks.

Regions are relative hotspots, not calibrated defect segmentation. The image
classifier still makes the defect decision. Coordinates are normalized to the
original frame because AnomalyVFM resizes the entire RGB image without cropping.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
import threading
import time

import numpy as np


def extract_defect_regions(anomaly_map, *, is_defect: bool):
    """Return up to five connected hotspots as normalized (x, y, w, h)."""
    if not is_defect or anomaly_map is None:
        return ()
    values = np.asarray(anomaly_map)
    if (values.ndim != 2 or min(values.shape) < 5 or max(values.shape) > 512
            or not np.isfinite(values).all()):
        return ()
    # Ignore the two-pixel zero-padding rim when choosing the baseline.
    baseline = float(np.median(values[2:-2, 2:-2]))
    peak = float(values.max())
    if peak - baseline < 1e-4:
        return ()
    mask = values >= baseline + 0.6 * (peak - baseline)
    height, width = mask.shape
    seen = np.zeros_like(mask)
    regions = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        seen[y, x] = True
        stack, pixels = [(int(y), int(x))], []
        while stack:
            cy, cx = stack.pop()
            pixels.append((cy, cx))
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = cy + dy, cx + dx
                if (0 <= ny < height and 0 <= nx < width
                        and mask[ny, nx] and not seen[ny, nx]):
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if len(pixels) < 4:
            continue
        ys, xs = zip(*pixels)
        x0, y0 = max(0, min(xs) - 1), max(0, min(ys) - 1)
        x1, y1 = min(width, max(xs) + 2), min(height, max(ys) + 2)
        # Diffuse scores cannot honestly localize a particular defect.
        if (x1 - x0) * (y1 - y0) > height * width * 0.8:
            continue
        strength = max(float(values[py, px]) for py, px in pixels)
        regions.append((strength, len(pixels), (x0 / width, y0 / height,
                                               (x1 - x0) / width, (y1 - y0) / height)))
    regions.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return tuple(item[2] for item in regions[:5])


def sample_key(sample):
    if sample is None:
        return None
    return sample.sample_id, sample.sample_index, sample.loop_index


@dataclass(frozen=True)
class RegionSnapshot:
    key: tuple | None = None
    regions: tuple = ()
    at: float = 0.0
    status: str = ""
    score: float | None = None


class DefectRegionState:
    """Hand off immutable inference results; never draw on a different slide."""
    def __init__(self):
        self._lock = threading.Lock()
        self._snapshots = OrderedDict()

    def clear(self):
        with self._lock:
            self._snapshots.clear()

    def update(self, sample, regions, *, now=None, status="", score=None):
        key = sample_key(sample)
        if key is None:
            return
        with self._lock:
            self._snapshots[key] = RegionSnapshot(
                key, tuple(regions), time.monotonic() if now is None else now,
                status, score,
            )
            self._snapshots.move_to_end(key)
            # Inference and the encoded preview have independent queues. Keep a
            # bounded history so a newer result cannot erase a visible slide.
            while len(self._snapshots) > 8:
                self._snapshots.popitem(last=False)

    def snapshot(self, sample, *, now=None):
        with self._lock:
            result = self._snapshots.get(sample_key(sample))
        if result is None:
            return None
        age = (time.monotonic() if now is None else now) - result.at
        return result if 0 <= age <= 3 else None

    def visible(self, sample, *, now=None):
        result = self.snapshot(sample, now=now)
        return result.regions if result else ()


def draw_demo_result(context, sample, result, width, height, *, failed=False):
    """Draw labels and regions atomically on the outgoing slide, never old text."""
    status = result.status if result else ("UNAVAILABLE" if failed else "ANALYZING")
    ground_truth = getattr(sample, "ground_truth", "unknown").upper()
    score = f"{result.score:.2f}" if result and result.score is not None else "--"
    comparison = ""
    color = (1, 1, 1)
    if result:
        color = (1, .15, .1) if status == "DEFECT" else (.35, 1, .35)
        comparison = "MATCH" if ground_truth == status else "MISMATCH"
        if ground_truth not in {"NORMAL", "DEFECT"}:
            comparison = "GT UNAVAILABLE"
        if status == "DEFECT":
            if not result.regions:
                comparison = "LOCATION UNAVAILABLE"
    scale = max(.5, width / 768)
    context.save()
    context.set_source_rgba(0, 0, 0, .78)
    context.rectangle(0, 0, width, 100 * scale)
    context.fill()
    context.set_font_size(20 * scale)
    for x, title, value, rgb in (
        (20, "AI RESULT", status, color),
        (width / scale / 3 + 10, "ANOMALY SCORE", score, (1, 1, 1)),
        (width / scale * 2 / 3, "DATASET GT", ground_truth, (1, 1, 1)),
    ):
        context.set_source_rgb(*rgb)
        context.move_to(x * scale, 25 * scale)
        context.show_text(title)
        context.move_to(x * scale, 52 * scale)
        context.show_text(value)
    context.set_font_size(14 * scale)
    context.set_source_rgb(1, 1, 1)
    context.move_to(20 * scale, 80 * scale)
    context.show_text(comparison)
    context.restore()
    if result and status == "DEFECT":
        draw_defect_regions(context, result.regions, width, height)


def draw_defect_regions(context, regions, width, height):
    for x, y, w, h in regions:
        left, top = x * width, y * height
        context.save()
        context.rectangle(left, top, w * width, h * height)
        context.set_source_rgb(0, 0, 0)
        context.set_line_width(6)
        context.stroke_preserve()
        context.set_source_rgb(1, 0.15, 0.1)
        context.set_line_width(3)
        context.stroke()
        context.set_font_size(16)
        context.move_to(left + 3, max(17, top - 5))
        context.show_text("AI REGION")
        context.restore()
