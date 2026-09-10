#!/usr/bin/env python3
"""Resolve a prior immutable source-plan publisher for a partial release retry."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
from pathlib import Path
from typing import Any


TAG = re.compile(r"^v[0-9]+\.[0-9]+\.[0-9]+$")
SHA = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")
MAX_REGISTRY_BYTES = 256 * 1024


class RecoveryError(RuntimeError):
    pass


def _strict_json(path: Path) -> dict[str, Any]:
    candidate = path.resolve()
    try:
        metadata = candidate.lstat()
        raw = candidate.read_bytes()
    except OSError as exc:
        raise RecoveryError(f"cannot read source-plan recovery registry: {candidate}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RecoveryError("source-plan recovery registry must be a regular file")
    if not raw or len(raw) > MAX_REGISTRY_BYTES:
        raise RecoveryError("source-plan recovery registry size is invalid")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RecoveryError(f"duplicate recovery registry member: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate,
            parse_constant=lambda value: (_ for _ in ()).throw(
                RecoveryError(f"invalid JSON constant: {value}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise RecoveryError("source-plan recovery registry is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RecoveryError("source-plan recovery registry root must be an object")
    return payload


def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )


def resolve(
    *,
    registry_path: Path,
    repository: Path,
    tag: str,
    current_publisher_sha: str,
) -> dict[str, str]:
    if not TAG.fullmatch(tag) or not SHA.fullmatch(current_publisher_sha):
        raise RecoveryError("release tag or current publisher SHA is invalid")
    registry = _strict_json(registry_path)
    if set(registry) != {"schemaVersion", "recoveries"} or registry.get("schemaVersion") != 1:
        raise RecoveryError("source-plan recovery registry schema is invalid")
    recoveries = registry.get("recoveries")
    if not isinstance(recoveries, dict):
        raise RecoveryError("source-plan recoveries must be an object")

    expected_fields = {
        "sourcePublisherSha",
        "sourcePlanSha256",
        "evidenceRunId",
        "evidenceAssetName",
    }
    for recovery_tag, entry in recoveries.items():
        if not isinstance(recovery_tag, str) or not TAG.fullmatch(recovery_tag):
            raise RecoveryError("source-plan recovery tag is invalid")
        if not isinstance(entry, dict) or set(entry) != expected_fields:
            raise RecoveryError(f"source-plan recovery fields are invalid: {recovery_tag}")
        if (
            not isinstance(entry.get("sourcePublisherSha"), str)
            or not SHA.fullmatch(entry["sourcePublisherSha"])
            or not isinstance(entry.get("sourcePlanSha256"), str)
            or not SHA256.fullmatch(entry["sourcePlanSha256"])
            or isinstance(entry.get("evidenceRunId"), bool)
            or not isinstance(entry.get("evidenceRunId"), int)
            or entry["evidenceRunId"] < 1
            or not isinstance(entry.get("evidenceAssetName"), str)
            or not SAFE_NAME.fullmatch(entry["evidenceAssetName"])
        ):
            raise RecoveryError(f"source-plan recovery identity is invalid: {recovery_tag}")

    entry = recoveries.get(tag)
    if entry is None:
        return {
            "publisher_sha": current_publisher_sha,
            "expected_sha256": "",
            "evidence_run_id": "",
            "evidence_asset_name": "",
        }

    source_publisher_sha = entry["sourcePublisherSha"]
    for commit in (source_publisher_sha, current_publisher_sha):
        result = _git(repository, "cat-file", "-e", f"{commit}^{{commit}}")
        if result.returncode != 0:
            raise RecoveryError("source-plan publisher commit is unavailable")
    ancestry = _git(
        repository,
        "merge-base",
        "--is-ancestor",
        source_publisher_sha,
        current_publisher_sha,
    )
    if ancestry.returncode != 0:
        raise RecoveryError("source-plan publisher is outside the current trusted lineage")
    return {
        "publisher_sha": source_publisher_sha,
        "expected_sha256": entry["sourcePlanSha256"],
        "evidence_run_id": str(entry["evidenceRunId"]),
        "evidence_asset_name": entry["evidenceAssetName"],
    }


def _append_outputs(path: Path, outputs: dict[str, str]) -> None:
    for key, value in outputs.items():
        if "\n" in value or "\r" in value:
            raise RecoveryError("source-plan recovery output contains a newline")
    with path.open("a", encoding="utf-8") as destination:
        for key, value in outputs.items():
            destination.write(f"{key}={value}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--current-publisher-sha", required=True)
    parser.add_argument("--github-output", type=Path)
    arguments = parser.parse_args()
    try:
        result = resolve(
            registry_path=arguments.registry,
            repository=arguments.repository.resolve(),
            tag=arguments.tag,
            current_publisher_sha=arguments.current_publisher_sha,
        )
        if arguments.github_output is not None:
            _append_outputs(arguments.github_output, result)
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    except (RecoveryError, OSError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
