"""Low-memory safetensors access for macOS unified-memory inference."""

from __future__ import annotations

import os
import stat
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def open_safetensors_for_sequential_load(checkpoint: Path) -> Iterator[object]:
    """Open a checkpoint without retaining its pages in the macOS file cache."""

    from safetensors import safe_open

    if sys.platform != "darwin":
        with safe_open(checkpoint, framework="pt", device="cpu") as weights:
            yield weights
        return

    import fcntl

    no_cache = getattr(fcntl, "F_NOCACHE", None)
    if no_cache is None:
        raise RuntimeError("macOS F_NOCACHE is required for MPS checkpoint loading")

    resolved = checkpoint.resolve(strict=True)
    metadata = resolved.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
        raise RuntimeError("safetensors checkpoint must resolve to a regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(resolved, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != metadata.st_dev
            or opened.st_ino != metadata.st_ino
            or opened.st_size != metadata.st_size
        ):
            raise RuntimeError("safetensors checkpoint identity changed while opening")
        fcntl.fcntl(descriptor, no_cache, 1)
        with safe_open(
            f"/dev/fd/{descriptor}", framework="pt", device="cpu"
        ) as weights:
            yield weights
    finally:
        os.close(descriptor)
