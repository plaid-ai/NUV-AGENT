from __future__ import annotations

import io

import numpy as np
from PIL import Image


def encode_snapshot_jpeg(frame_rgb: np.ndarray, quality: int = 90) -> bytes:
    normalized_quality = max(1, min(95, int(quality)))
    image = Image.fromarray(np.asarray(frame_rgb, dtype=np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=normalized_quality)
    return buffer.getvalue()
