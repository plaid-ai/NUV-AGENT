from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path


class StableFileError(OSError):
    """Raised when a security-sensitive file cannot be read atomically."""


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _open_stable_regular(
    path: Path,
    *,
    minimum: int,
    maximum: int,
) -> tuple[int, os.stat_result]:
    if minimum < 0 or maximum < minimum:
        raise ValueError("invalid stable-file size bounds")
    candidate = Path(path)
    try:
        initial = candidate.lstat()
    except OSError as exc:
        raise StableFileError(f"cannot stat stable file: {candidate}") from exc
    if stat.S_ISLNK(initial.st_mode) or not stat.S_ISREG(initial.st_mode):
        raise StableFileError(f"stable file is not a regular file: {candidate}")
    if not minimum <= initial.st_size <= maximum:
        raise StableFileError(f"stable file size is invalid: {candidate}")
    if not hasattr(os, "O_NOFOLLOW"):
        raise StableFileError("O_NOFOLLOW is required for stable file reads")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(candidate, flags)
        opened = os.fstat(descriptor)
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise StableFileError(f"cannot open stable file: {candidate}") from exc
    if (
        not stat.S_ISREG(opened.st_mode)
        or _identity(initial) != _identity(opened)
    ):
        os.close(descriptor)
        raise StableFileError(f"stable file changed while opening: {candidate}")
    return descriptor, opened


def _verify_stable_close(
    path: Path,
    descriptor: int,
    opened: os.stat_result,
) -> None:
    try:
        after = os.fstat(descriptor)
        final_path = Path(path).lstat()
    except OSError as exc:
        raise StableFileError(f"cannot restat stable file: {path}") from exc
    if _identity(opened) != _identity(after) or _identity(opened) != _identity(
        final_path
    ):
        raise StableFileError(f"stable file changed while reading: {path}")


def read_stable_regular_file(
    path: Path,
    *,
    maximum: int,
    minimum: int = 1,
) -> bytes:
    descriptor, opened = _open_stable_regular(
        path,
        minimum=minimum,
        maximum=maximum,
    )
    chunks: list[bytes] = []
    total = 0
    try:
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                raise StableFileError(f"stable file exceeds size limit: {path}")
        _verify_stable_close(path, descriptor, opened)
    except OSError as exc:
        if isinstance(exc, StableFileError):
            raise
        raise StableFileError(f"cannot read stable file: {path}") from exc
    finally:
        os.close(descriptor)
    if total != opened.st_size or total < minimum:
        raise StableFileError(f"stable file length changed while reading: {path}")
    return b"".join(chunks)


def digest_stable_regular_file(
    path: Path,
    *,
    maximum: int,
    minimum: int = 1,
) -> tuple[str, int]:
    descriptor, opened = _open_stable_regular(
        path,
        minimum=minimum,
        maximum=maximum,
    )
    digest = hashlib.sha256()
    total = 0
    try:
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise StableFileError(f"stable file exceeds size limit: {path}")
            digest.update(chunk)
        _verify_stable_close(path, descriptor, opened)
    except OSError as exc:
        if isinstance(exc, StableFileError):
            raise
        raise StableFileError(f"cannot hash stable file: {path}") from exc
    finally:
        os.close(descriptor)
    if total != opened.st_size or total < minimum:
        raise StableFileError(f"stable file length changed while hashing: {path}")
    return digest.hexdigest(), total
