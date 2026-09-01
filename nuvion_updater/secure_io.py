from __future__ import annotations

import os
import stat
from pathlib import Path

from nuvion_updater.errors import UpdaterSecurityError
from nuvion_updater.util import require_safe_basename


def read_fixed_regular_file(
    directory: str | Path,
    basename: str,
    *,
    max_bytes: int,
    require_root_owner: bool,
    require_private: bool = False,
) -> bytes:
    """Read one fixed-directory basename without following either symlink."""

    name = require_safe_basename(basename, field="filename")
    root = Path(directory)
    try:
        root_metadata = root.lstat()
    except OSError as exc:
        raise UpdaterSecurityError("UNSAFE_DIRECTORY", "cannot inspect fixed directory") from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise UpdaterSecurityError("UNSAFE_DIRECTORY", "fixed directory must be real")
    if root_metadata.st_mode & 0o022:
        raise UpdaterSecurityError(
            "UNSAFE_DIRECTORY", "fixed directory must not be group/other writable"
        )
    if require_root_owner and root_metadata.st_uid != 0:
        raise UpdaterSecurityError("UNSAFE_DIRECTORY", "fixed directory must be root-owned")

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_descriptor = os.open(root, directory_flags)
    try:
        file_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(name, file_flags, dir_fd=directory_descriptor)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise UpdaterSecurityError("UNSAFE_FILE", "input must be a regular file")
            if before.st_size > max_bytes:
                raise UpdaterSecurityError("INPUT_TOO_LARGE", "input exceeds size limit")
            if require_root_owner and before.st_uid != 0:
                raise UpdaterSecurityError("UNSAFE_FILE", "input must be root-owned")
            forbidden = 0o077 if require_private else 0o022
            if before.st_mode & forbidden:
                raise UpdaterSecurityError("UNSAFE_FILE", "input permissions are too broad")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(descriptor, min(64 * 1024, max_bytes + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > max_bytes:
                    raise UpdaterSecurityError(
                        "INPUT_TOO_LARGE", "input exceeds size limit"
                    )
            after = os.fstat(descriptor)
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise UpdaterSecurityError("INPUT_CHANGED", "input changed while read")
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    finally:
        os.close(directory_descriptor)
