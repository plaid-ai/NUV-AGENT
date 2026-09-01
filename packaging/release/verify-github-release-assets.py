from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


_SHA256_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class ReleaseAssetVerificationError(RuntimeError):
    pass


def sha256_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def load_release(
    *, api_url: str, repository: str, tag: str, token: str
) -> dict[str, Any] | None:
    encoded_tag = urllib.parse.quote(tag, safe="")
    url = f"{api_url.rstrip('/')}/repos/{repository}/releases/tags/{encoded_tag}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "nuv-agent-release-gate",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise ReleaseAssetVerificationError(
            f"GitHub release lookup failed with HTTP {exc.code}"
        ) from exc
    except urllib.error.URLError as exc:
        raise ReleaseAssetVerificationError(
            f"GitHub release lookup failed: {exc.reason}"
        ) from exc
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseAssetVerificationError(
            "GitHub release lookup returned invalid JSON"
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("assets"), list):
        raise ReleaseAssetVerificationError(
            "GitHub release response does not contain an asset list"
        )
    return payload


def verify_existing_assets(
    release: dict[str, Any] | None, asset_paths: list[Path]
) -> None:
    if release is None:
        print("Release does not exist yet; all assets are new.")
        return
    raw_assets = release.get("assets")
    if not isinstance(raw_assets, list):
        raise ReleaseAssetVerificationError("release assets must be a list")

    for path in asset_paths:
        if not path.is_file() or path.is_symlink():
            raise ReleaseAssetVerificationError(
                f"local release asset is not a regular file: {path}"
            )
        matches = [
            asset
            for asset in raw_assets
            if isinstance(asset, dict) and asset.get("name") == path.name
        ]
        if len(matches) > 1:
            raise ReleaseAssetVerificationError(
                f"GitHub release contains duplicate asset names: {path.name}"
            )
        if not matches:
            print(f"Asset is new: {path.name}")
            continue

        remote_digest = matches[0].get("digest")
        if not isinstance(remote_digest, str) or not _SHA256_DIGEST.fullmatch(
            remote_digest
        ):
            raise ReleaseAssetVerificationError(
                f"existing GitHub asset has no trustworthy SHA-256 digest: {path.name}"
            )
        local_digest = sha256_digest(path)
        if local_digest != remote_digest:
            raise ReleaseAssetVerificationError(
                "refusing to reuse an existing GitHub asset name with different bytes: "
                f"{path.name} local={local_digest} remote={remote_digest}"
            )
        print(f"Existing asset digest matches: {path.name} {local_digest}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail closed when a GitHub release asset name already has other bytes"
    )
    parser.add_argument("--repository", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--asset", action="append", required=True)
    parser.add_argument(
        "--api-url", default=os.environ.get("GITHUB_API_URL", "https://api.github.com")
    )
    args = parser.parse_args()
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        parser.error("GITHUB_TOKEN is required")
    asset_paths = [Path(value).expanduser().resolve() for value in args.asset]
    try:
        release = load_release(
            api_url=args.api_url,
            repository=args.repository,
            tag=args.tag,
            token=token,
        )
        verify_existing_assets(release, asset_paths)
    except (OSError, ReleaseAssetVerificationError) as exc:
        print(f"release asset verification failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
