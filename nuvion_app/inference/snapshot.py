from __future__ import annotations

from io import BytesIO
import struct
import threading
import zlib

import numpy as np


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class LatestFrameBuffer:
    def __init__(self) -> None:
        self._frame = None
        self._lock = threading.Lock()

    def remember(self, frame_rgb: np.ndarray) -> None:
        with self._lock:
            self._frame = np.ascontiguousarray(frame_rgb)

    def copy(self) -> np.ndarray | None:
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", checksum)


def encode_png(frame_rgb: np.ndarray) -> bytes:
    array = np.asarray(frame_rgb, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("frame_rgb must be an HxWx3 uint8 array")

    array = np.ascontiguousarray(array)
    height, width, _ = array.shape
    scanlines = b"".join(b"\x00" + array[row].tobytes() for row in range(height))
    compressed = zlib.compress(scanlines, level=6)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"".join(
        [
            PNG_SIGNATURE,
            _png_chunk(b"IHDR", header),
            _png_chunk(b"IDAT", compressed),
            _png_chunk(b"IEND", b""),
        ]
    )


def _encode_with_pillow(frame_rgb: np.ndarray, fmt: str, **save_kwargs) -> bytes | None:
    try:
        from PIL import Image
    except ImportError:
        return None

    image = Image.fromarray(np.asarray(frame_rgb, dtype=np.uint8), mode="RGB")
    buffer = BytesIO()
    image.save(buffer, format=fmt, **save_kwargs)
    return buffer.getvalue()


def encode_snapshot(frame_rgb: np.ndarray, preferred_content_type: str = "image/jpeg") -> tuple[bytes, str]:
    normalized = (preferred_content_type or "").strip().lower()

    if normalized in {"image/jpeg", "image/jpg"}:
        encoded = _encode_with_pillow(frame_rgb, "JPEG", quality=90, optimize=True)
        if encoded is not None:
            return encoded, "image/jpeg"

    if normalized == "image/webp":
        encoded = _encode_with_pillow(frame_rgb, "WEBP", quality=90, method=6)
        if encoded is not None:
            return encoded, "image/webp"

    if normalized == "image/png":
        return encode_png(frame_rgb), "image/png"

    return encode_png(frame_rgb), "image/png"


def capture_and_upload_snapshot(
    frame_rgb: np.ndarray | None,
    request_upload_url,
    upload_bytes_to_url,
    preferred_content_type: str = "image/jpeg",
) -> str | None:
    if frame_rgb is None:
        return None

    snapshot_bytes, content_type = encode_snapshot(frame_rgb, preferred_content_type=preferred_content_type)
    upload_meta = request_upload_url("SNAPSHOT", content_type)
    if not upload_meta:
        return None

    object_name = upload_meta.get("objectName")
    upload_url = upload_meta.get("uploadUrl")
    if not object_name or not upload_url:
        return None

    if not upload_bytes_to_url(upload_url, snapshot_bytes, content_type):
        return None
    return object_name
