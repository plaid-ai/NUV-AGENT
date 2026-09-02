#!/usr/bin/env python3
"""Stage assets in a draft and atomically publish an immutable GitHub release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
TAG = re.compile(r"^v[0-9]+\.[0-9]+\.[0-9]+$")
SHA = re.compile(r"^[0-9a-f]{40}$")
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")
SHA256_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
MAX_API_BYTES = 2 * 1024 * 1024
LEGACY_TAG = "v0.1.120"
LEGACY_SHA = "b354026f73d63a82ad4c64923f46dc400a73efcb"


class GitHubReleaseError(RuntimeError):
    pass


class GitHubApiError(GitHubReleaseError):
    def __init__(self, message: str, *, status: int) -> None:
        super().__init__(message)
        self.status = status


def _strict_json(raw: bytes, *, label: str) -> Any:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise GitHubReleaseError(f"duplicate {label} member: {key}")
            result[key] = value
        return result

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate,
            parse_constant=lambda value: (_ for _ in ()).throw(
                GitHubReleaseError(f"invalid JSON constant: {value}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise GitHubReleaseError(f"invalid {label} response") from exc


class GitHubApi:
    def __init__(self, repository: str, token: str) -> None:
        if not REPOSITORY.fullmatch(repository):
            raise GitHubReleaseError("GitHub repository identity is invalid")
        if not token or "\n" in token or "\r" in token:
            raise GitHubReleaseError("GitHub token is unavailable")
        self.repository = repository
        self.token = token

    def request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> Any:
        body = (
            json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
            if payload is not None
            else None
        )
        request = urllib.request.Request(
            f"https://api.github.com{path}",
            data=body,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2026-03-10",
                "User-Agent": "nuv-immutable-release-publisher/1",
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read(MAX_API_BYTES + 1)
        except urllib.error.HTTPError as exc:
            raise GitHubApiError(
                f"GitHub release API rejected {method} {path} with HTTP {exc.code}",
                status=exc.code,
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise GitHubReleaseError(f"GitHub release API failed for {path}: {exc}") from exc
        if len(raw) > MAX_API_BYTES:
            raise GitHubReleaseError("GitHub release API response exceeds size limit")
        return _strict_json(raw, label="GitHub release API")

    def release(self, tag: str) -> dict[str, Any] | None:
        encoded = urllib.parse.quote(tag, safe="")
        try:
            payload = self.request(
                "GET", f"/repos/{self.repository}/releases/tags/{encoded}"
            )
        except GitHubApiError as exc:
            if exc.status == 404:
                return None
            raise
        if not isinstance(payload, dict):
            raise GitHubReleaseError("GitHub release response is invalid")
        return payload


def _artifact(path: Path) -> dict[str, Any]:
    candidate = path if path.is_absolute() else Path.cwd() / path
    try:
        before = candidate.lstat()
    except OSError as exc:
        raise GitHubReleaseError(f"cannot stat release asset: {candidate}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode) or before.st_size < 1:
        raise GitHubReleaseError(f"release asset must be a nonempty regular file: {candidate}")
    if not SAFE_NAME.fullmatch(candidate.name):
        raise GitHubReleaseError(f"release asset name is unsafe: {candidate.name}")
    digest = hashlib.sha256()
    size = 0
    try:
        with candidate.open("rb") as source:
            opened = os.fstat(source.fileno())
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
            after = os.fstat(source.fileno())
        final = candidate.lstat()
    except OSError as exc:
        raise GitHubReleaseError(f"cannot hash release asset: {candidate}") from exc
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if not (identity(before) == identity(opened) == identity(after) == identity(final)):
        raise GitHubReleaseError(f"release asset changed while hashing: {candidate}")
    return {
        "path": candidate,
        "name": candidate.name,
        "digest": f"sha256:{digest.hexdigest()}",
        "size": size,
    }


def _validate_release(release: dict[str, Any], *, tag: str) -> None:
    if (
        release.get("tag_name") != tag
        or isinstance(release.get("id"), bool)
        or not isinstance(release.get("id"), int)
        or release["id"] < 1
        or not isinstance(release.get("draft"), bool)
        or not isinstance(release.get("immutable"), bool)
        or not isinstance(release.get("assets"), list)
    ):
        raise GitHubReleaseError("GitHub release identity or lifecycle state is invalid")
    if release["draft"] and release["immutable"]:
        raise GitHubReleaseError("draft release cannot already be immutable")


def _remote_assets(release: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for asset in release["assets"]:
        if not isinstance(asset, dict) or not isinstance(asset.get("name"), str):
            raise GitHubReleaseError("GitHub release asset metadata is invalid")
        name = asset["name"]
        if name in result:
            raise GitHubReleaseError(f"GitHub release has duplicate asset name: {name}")
        result[name] = asset
    return result


def _verify_asset(remote: dict[str, Any], local: dict[str, Any]) -> None:
    if (
        remote.get("state") != "uploaded"
        or remote.get("name") != local["name"]
        or isinstance(remote.get("size"), bool)
        or remote.get("size") != local["size"]
        or not isinstance(remote.get("digest"), str)
        or not SHA256_DIGEST.fullmatch(remote["digest"])
        or remote["digest"] != local["digest"]
    ):
        raise GitHubReleaseError(
            f"existing GitHub release asset bytes differ: {local['name']}"
        )


def _create_draft(
    api: GitHubApi, *, tag: str, component_sha: str
) -> dict[str, Any]:
    try:
        payload = api.request(
            "POST",
            f"/repos/{api.repository}/releases",
            {
                "tag_name": tag,
                "target_commitish": component_sha,
                "name": tag,
                "draft": True,
                "prerelease": False,
                "generate_release_notes": False,
            },
        )
    except GitHubApiError as exc:
        if exc.status != 422:
            raise
        raced = api.release(tag)
        if raced is None:
            raise GitHubReleaseError("release creation raced without a visible release") from exc
        return raced
    if not isinstance(payload, dict):
        raise GitHubReleaseError("created GitHub draft response is invalid")
    return payload


def _upload_asset(
    api: GitHubApi, *, tag: str, local: dict[str, Any]
) -> None:
    environment = {**os.environ, "GH_TOKEN": api.token, "GITHUB_TOKEN": api.token}
    uploaded = subprocess.run(
        [
            "gh",
            "release",
            "upload",
            tag,
            "--repo",
            api.repository,
            "--",
            str(local["path"]),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=600,
        check=False,
        env=environment,
    )
    if uploaded.returncode != 0:
        refreshed = api.release(tag)
        if refreshed is None:
            raise GitHubReleaseError("GitHub release disappeared during asset upload")
        remote = _remote_assets(refreshed).get(local["name"])
        if remote is None:
            raise GitHubReleaseError("GitHub release asset upload failed")
        _verify_asset(remote, local)


def publish_release(
    *,
    api: GitHubApi,
    tag: str,
    component_sha: str,
    phase: str,
    asset_paths: list[Path],
    allow_legacy_mutable: bool,
) -> dict[str, Any]:
    if not TAG.fullmatch(tag) or not SHA.fullmatch(component_sha):
        raise GitHubReleaseError("release tag or component SHA is invalid")
    if phase not in {"stage", "finalize"}:
        raise GitHubReleaseError("release publication phase is invalid")
    if allow_legacy_mutable and (tag != LEGACY_TAG or component_sha != LEGACY_SHA):
        raise GitHubReleaseError("mutable legacy exception is restricted to exact v0.1.120")
    local_assets = [_artifact(path) for path in asset_paths]
    local_by_name = {asset["name"]: asset for asset in local_assets}
    if len(local_by_name) != len(local_assets):
        raise GitHubReleaseError("local release asset names must be unique")

    release = api.release(tag)
    if release is None:
        release = _create_draft(api, tag=tag, component_sha=component_sha)
    _validate_release(release, tag=tag)
    if not release["draft"] and not release["immutable"] and not allow_legacy_mutable:
        raise GitHubReleaseError("existing published release is mutable")

    remote_by_name = _remote_assets(release)
    for name, local in local_by_name.items():
        remote = remote_by_name.get(name)
        if remote is None:
            if release["immutable"]:
                raise GitHubReleaseError(f"immutable GitHub release is missing asset: {name}")
            _upload_asset(api, tag=tag, local=local)
        else:
            _verify_asset(remote, local)

    release = api.release(tag)
    if release is None:
        raise GitHubReleaseError("GitHub release disappeared after staging")
    _validate_release(release, tag=tag)
    remote_by_name = _remote_assets(release)
    for name, local in local_by_name.items():
        if name not in remote_by_name:
            raise GitHubReleaseError(f"GitHub release is missing staged asset: {name}")
        _verify_asset(remote_by_name[name], local)
    if (
        phase == "finalize"
        and not allow_legacy_mutable
        and set(remote_by_name) != set(local_by_name)
    ):
        raise GitHubReleaseError("draft GitHub release has an unexpected asset set")

    if phase == "finalize" and release["draft"]:
        release = api.request(
            "PATCH",
            f"/repos/{api.repository}/releases/{release['id']}",
            {"draft": False},
        )
        if not isinstance(release, dict):
            raise GitHubReleaseError("published GitHub release response is invalid")
        _validate_release(release, tag=tag)

    if phase == "finalize":
        if release["draft"]:
            raise GitHubReleaseError("GitHub release remained a draft after finalization")
        if not release["immutable"] and not allow_legacy_mutable:
            raise GitHubReleaseError("published GitHub release is not immutable")
        if not allow_legacy_mutable and set(_remote_assets(release)) != set(local_by_name):
            raise GitHubReleaseError("immutable GitHub release has an unexpected asset set")

    return {
        "schemaVersion": 1,
        "tag": tag,
        "phase": phase,
        "draft": release["draft"],
        "immutable": release["immutable"],
        "legacyMutable": allow_legacy_mutable,
        "assets": {name: local_by_name[name]["digest"] for name in sorted(local_by_name)},
        "status": "VERIFIED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--component-sha", required=True)
    parser.add_argument("--phase", choices=("stage", "finalize"), required=True)
    parser.add_argument("--asset", action="append", type=Path, required=True)
    parser.add_argument("--token-env", default="GITHUB_TOKEN")
    parser.add_argument("--allow-legacy-mutable", action="store_true")
    arguments = parser.parse_args()
    try:
        result = publish_release(
            api=GitHubApi(
                arguments.repository, os.environ.get(arguments.token_env, "")
            ),
            tag=arguments.tag,
            component_sha=arguments.component_sha,
            phase=arguments.phase,
            asset_paths=arguments.asset,
            allow_legacy_mutable=arguments.allow_legacy_mutable,
        )
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    except (GitHubReleaseError, OSError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
