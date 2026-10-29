"""Display-only localization from model pixel scores, never dataset masks.

Regions are relative hotspots, not calibrated defect segmentation. The image
classifier still makes the defect decision. Coordinates are normalized to the
original frame because AnomalyVFM resizes the entire RGB image without cropping.
"""
from __future__ import annotations

from dataclasses import dataclass
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


class DefectRegionState:
    """Hand off immutable inference results; never draw on a different slide."""
    def __init__(self):
        self._lock = threading.Lock()
        self._snapshot = RegionSnapshot()

    def clear(self):
        with self._lock:
            self._snapshot = RegionSnapshot()

    def update(self, sample, regions, *, now=None):
        with self._lock:
            self._snapshot = RegionSnapshot(sample_key(sample), tuple(regions),
                                           time.monotonic() if now is None else now)

    def visible(self, sample, *, now=None, max_age=3):
        with self._lock:
            result = self._snapshot
        age = (time.monotonic() if now is None else now) - result.at
        if result.key is None or result.key != sample_key(sample) or not 0 <= age <= max_age:
            return ()
        return result.regions


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
