#!/usr/bin/env python3
"""Fail closed when a release has unresolved audited dependency blockers."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
BLOCKER_ID = re.compile(r"^[A-Z0-9][A-Z0-9._-]{0,127}$")
LEGACY_VERSION = "0.1.120"


class ReadinessError(RuntimeError):
    pass


def _load(path: Path) -> dict[str, Any]:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ReadinessError(f"duplicate release readiness member: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReadinessError("release readiness document is invalid") from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schemaVersion", "releases"}
        or payload.get("schemaVersion") != 1
        or not isinstance(payload.get("releases"), dict)
    ):
        raise ReadinessError("release readiness document does not match schema v1")
    return payload


def verify_readiness(path: Path, *, version: str, allow_legacy: bool) -> None:
    if not SEMVER.fullmatch(version):
        raise ReadinessError("release readiness version must be exact SemVer")
    if allow_legacy:
        if version != LEGACY_VERSION:
            raise ReadinessError("release readiness legacy exception is restricted to v0.1.120")
        return
    payload = _load(path)
    releases = payload["releases"]
    for configured_version in releases:
        if not isinstance(configured_version, str) or not SEMVER.fullmatch(configured_version):
            raise ReadinessError("release readiness contains an invalid version key")
    release = releases.get(version)
    if not isinstance(release, dict) or set(release) != {"status", "blockers"}:
        raise ReadinessError(f"release {version} has no reviewed readiness decision")
    status = release.get("status")
    blockers = release.get("blockers")
    if status not in {"READY", "BLOCKED"} or not isinstance(blockers, list):
        raise ReadinessError("release readiness decision is invalid")
    blocker_ids: list[str] = []
    for blocker in blockers:
        if not isinstance(blocker, dict):
            raise ReadinessError("release readiness blocker is invalid")
        identifier = blocker.get("id")
        if not isinstance(identifier, str) or not BLOCKER_ID.fullmatch(identifier):
            raise ReadinessError("release readiness blocker id is invalid")
        blocker_ids.append(identifier)
    if len(set(blocker_ids)) != len(blocker_ids):
        raise ReadinessError("release readiness blocker ids are duplicated")
    if status == "READY" and blocker_ids:
        raise ReadinessError("READY release must have no blockers")
    if status == "BLOCKED" and not blocker_ids:
        raise ReadinessError("BLOCKED release must identify at least one blocker")
    if status != "READY":
        raise ReadinessError(
            f"release {version} is blocked by: {', '.join(sorted(blocker_ids))}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--allow-legacy", action="store_true")
    arguments = parser.parse_args()
    try:
        verify_readiness(
            arguments.policy,
            version=arguments.version,
            allow_legacy=arguments.allow_legacy,
        )
    except ReadinessError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
