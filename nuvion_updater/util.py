from __future__ import annotations

import os
import re
import stat
from pathlib import Path

from nuvion_updater.errors import UpdaterSecurityError

_SAFE_BASENAME = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._+-]{0,254}$")


def require_safe_basename(value: str, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not _SAFE_BASENAME.fullmatch(value)
        or Path(value).name != value
        or value in {".", ".."}
    ):
        raise UpdaterSecurityError("UNSAFE_PATH", f"{field} must be a safe basename")
    return value


def parse_digest(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or not re.fullmatch(r"[0-9a-f]{64}", value[7:])
    ):
        raise UpdaterSecurityError(
            "INVALID_BOM_DIGEST", "bomDigest must be sha256:<64 lowercase hex>"
        )
    return value[7:]


def ensure_directory(
    path: str | Path,
    *,
    mode: int,
    require_root_owner: bool,
) -> Path:
    candidate = Path(path)
    candidate.mkdir(parents=True, exist_ok=True, mode=mode)
    try:
        metadata = candidate.lstat()
    except OSError as exc:
        raise UpdaterSecurityError("UNSAFE_DIRECTORY", f"cannot inspect {candidate}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise UpdaterSecurityError(
            "UNSAFE_DIRECTORY", f"{candidate} must be a real directory"
        )
    if metadata.st_mode & 0o022:
        raise UpdaterSecurityError(
            "UNSAFE_DIRECTORY", f"{candidate} must not be group/other writable"
        )
    if require_root_owner and metadata.st_uid != 0:
        raise UpdaterSecurityError(
            "UNSAFE_DIRECTORY", f"{candidate} must be owned by root"
        )
    try:
        os.chmod(candidate, mode)
    except OSError as exc:
        raise UpdaterSecurityError("UNSAFE_DIRECTORY", f"cannot secure {candidate}: {exc}") from exc
    return candidate.resolve(strict=True)


def fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
