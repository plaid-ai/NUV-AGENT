from __future__ import annotations

import argparse
import gzip
import io
import os
import stat
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


def _validate_member(member: tarfile.TarInfo) -> None:
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe sdist member path: {member.name}")
    if member.ischr() or member.isblk() or member.isfifo():
        raise ValueError(f"unsupported special sdist member: {member.name}")
    if member.issym() or member.islnk():
        target = PurePosixPath(member.linkname)
        if target.is_absolute() or ".." in target.parts:
            raise ValueError(f"unsafe sdist link target: {member.linkname}")


def _normalized_mode(member: tarfile.TarInfo) -> int:
    if member.isdir():
        return 0o755
    if member.issym() or member.islnk():
        return 0o777
    return 0o755 if member.mode & stat.S_IXUSR else 0o644


def normalize_sdist(path: Path, source_date_epoch: int) -> None:
    if source_date_epoch < 0:
        raise ValueError("source date epoch must not be negative")
    path = path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"sdist is not a regular file: {path}")

    entries: list[tuple[tarfile.TarInfo, bytes | None]] = []
    with tarfile.open(path, mode="r:gz") as source:
        for member in source.getmembers():
            _validate_member(member)
            data: bytes | None = None
            if member.isfile():
                extracted = source.extractfile(member)
                if extracted is None:
                    raise ValueError(f"cannot read sdist member: {member.name}")
                data = extracted.read()
                if len(data) != member.size:
                    raise ValueError(f"sdist member changed while reading: {member.name}")
            entries.append((member, data))
    if not entries:
        raise ValueError("sdist is empty")
    roots = {PurePosixPath(member.name).parts[0] for member, _ in entries}
    if len(roots) != 1:
        raise ValueError("sdist must contain exactly one top-level directory")

    temporary_path: Path | None = None
    tar_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tar", delete=False
        ) as tar_handle:
            tar_path = Path(tar_handle.name)
        with tarfile.open(tar_path, mode="w", format=tarfile.PAX_FORMAT) as target:
            for member, data in sorted(entries, key=lambda entry: entry[0].name):
                normalized = tarfile.TarInfo(member.name)
                normalized.type = member.type
                normalized.linkname = member.linkname
                normalized.size = len(data) if data is not None else 0
                normalized.mode = _normalized_mode(member)
                normalized.uid = 0
                normalized.gid = 0
                normalized.uname = ""
                normalized.gname = ""
                normalized.mtime = source_date_epoch
                normalized.pax_headers = {}
                if data is None:
                    target.addfile(normalized)
                else:
                    target.addfile(normalized, io.BytesIO(data))

        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as compressed:
            temporary_path = Path(compressed.name)
            with tar_path.open("rb") as raw_tar:
                with gzip.GzipFile(
                    filename="",
                    mode="wb",
                    compresslevel=9,
                    fileobj=compressed,
                    mtime=source_date_epoch,
                ) as output:
                    for chunk in iter(lambda: raw_tar.read(1024 * 1024), b""):
                        output.write(chunk)
            compressed.flush()
            os.fsync(compressed.fileno())
        os.chmod(temporary_path, 0o644)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        if tar_path is not None and tar_path.exists():
            tar_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize a Python sdist to deterministic tar and gzip metadata"
    )
    parser.add_argument("sdist")
    parser.add_argument("--source-date-epoch", required=True, type=int)
    args = parser.parse_args()
    try:
        normalize_sdist(Path(args.sdist), args.source_date_epoch)
    except (OSError, tarfile.TarError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
