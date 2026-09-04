#!/usr/bin/env python3
"""Fail closed on Homebrew channel downgrade or same-version byte drift."""

from __future__ import annotations

import argparse
import json
import re
import stat
from pathlib import Path


SEMVER = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
VERSION_LINE = re.compile(
    r'^\s*version\s+"((?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*))"\s*$',
    re.MULTILINE,
)
MAX_FORMULA_BYTES = 2 * 1024 * 1024


class HomebrewPromotionError(RuntimeError):
    pass


def _formula(path: Path, *, label: str) -> tuple[bytes, str]:
    try:
        metadata = path.lstat()
        raw = path.read_bytes()
        final = path.lstat()
    except OSError as exc:
        raise HomebrewPromotionError(f"cannot read {label} formula") from exc
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size < 1
        or metadata.st_size > MAX_FORMULA_BYTES
        or identity(metadata) != identity(final)
        or len(raw) != metadata.st_size
    ):
        raise HomebrewPromotionError(f"{label} formula identity is unsafe")
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise HomebrewPromotionError(f"{label} formula is not UTF-8") from exc
    versions = VERSION_LINE.findall(text)
    if len(versions) != 1:
        raise HomebrewPromotionError(f"{label} formula has no unique exact version")
    return raw, versions[0]


def verify_promotion(
    current_path: Path, candidate_path: Path, *, requested_version: str
) -> dict[str, str | int]:
    match = SEMVER.fullmatch(requested_version)
    if match is None:
        raise HomebrewPromotionError("requested Homebrew version is invalid")
    current, current_version = _formula(current_path, label="current")
    candidate, candidate_version = _formula(candidate_path, label="candidate")
    if candidate_version != requested_version:
        raise HomebrewPromotionError("candidate formula version differs from request")
    numeric = lambda value: tuple(int(part) for part in value.split("."))
    if numeric(current_version) > numeric(requested_version):
        raise HomebrewPromotionError("refusing to move Homebrew channel backwards")
    if current_version == requested_version:
        if current != candidate:
            raise HomebrewPromotionError(
                "same Homebrew version already has different bytes"
            )
        status = "NOOP"
    else:
        status = "UPDATE"
    return {
        "schemaVersion": 1,
        "currentVersion": current_version,
        "requestedVersion": requested_version,
        "status": status,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--version", required=True)
    arguments = parser.parse_args()
    try:
        result = verify_promotion(
            arguments.current.resolve(),
            arguments.candidate.resolve(),
            requested_version=arguments.version,
        )
    except HomebrewPromotionError as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
