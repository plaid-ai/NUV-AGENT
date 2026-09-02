#!/usr/bin/env python3
"""Download one independently authenticated previous APT package for rollback."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import os
import re
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Mapping


SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
FINGERPRINT = re.compile(r"^[0-9A-F]{40}$")
PACKAGE_PATH = re.compile(
    r"^pool/main/n/nuv-agent/nuv-agent_([0-9]+\.[0-9]+\.[0-9]+)_arm64\.deb$"
)
MAX_METADATA_BYTES = 16 * 1024 * 1024
MAX_DEB_BYTES = 4 * 1024 * 1024 * 1024


class RollbackPreparationError(RuntimeError):
    pass


def _semver_tuple(value: str) -> tuple[int, int, int]:
    if not SEMVER.fullmatch(value):
        raise RollbackPreparationError(f"invalid Agent package version: {value}")
    return tuple(int(part) for part in value.split("."))  # type: ignore[return-value]


def parse_release_sha256(payload: str, relative_path: str) -> tuple[str, int]:
    in_sha256 = False
    matches: list[tuple[str, int]] = []
    for raw_line in payload.splitlines():
        if raw_line == "SHA256:":
            in_sha256 = True
            continue
        if in_sha256 and raw_line and not raw_line.startswith(" "):
            break
        if not in_sha256 or not raw_line.strip():
            continue
        fields = raw_line.split()
        if len(fields) != 3:
            raise RollbackPreparationError("APT Release SHA256 entry is invalid")
        digest, size_text, path = fields
        if path != relative_path:
            continue
        if not SHA256.fullmatch(digest) or not size_text.isdigit():
            raise RollbackPreparationError("APT Release checksum is invalid")
        matches.append((digest, int(size_text)))
    if len(matches) != 1:
        raise RollbackPreparationError(
            f"APT Release must contain exactly one checksum for {relative_path}"
        )
    return matches[0]


def parse_packages(payload: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    current: dict[str, str] = {}
    last_key: str | None = None
    for line in [*payload.splitlines(), ""]:
        if not line:
            if current:
                records.append(current)
                current = {}
                last_key = None
            continue
        if line.startswith((" ", "\t")):
            if last_key is None:
                raise RollbackPreparationError("APT Packages continuation has no field")
            current[last_key] += "\n" + line
            continue
        if ":" not in line:
            raise RollbackPreparationError("APT Packages field is invalid")
        key, value = line.split(":", 1)
        if not re.fullmatch(r"[A-Za-z0-9-]+", key) or key in current:
            raise RollbackPreparationError("APT Packages contains an invalid duplicate field")
        current[key] = value.lstrip(" ")
        last_key = key
    return records


def select_rollback_record(
    records: list[dict[str, str]], *, current_version: str
) -> dict[str, str] | None:
    current_tuple = _semver_tuple(current_version)
    candidates: list[tuple[tuple[int, int, int], dict[str, str]]] = []
    for record in records:
        if record.get("Package") != "nuv-agent" or record.get("Architecture") != "arm64":
            continue
        version = record.get("Version", "")
        try:
            version_tuple = _semver_tuple(version)
        except RollbackPreparationError:
            continue
        if version_tuple >= current_tuple:
            continue
        filename = record.get("Filename", "")
        match = PACKAGE_PATH.fullmatch(filename)
        digest = record.get("SHA256", "")
        size = record.get("Size", "")
        if (
            match is None
            or match.group(1) != version
            or not SHA256.fullmatch(digest)
            or not size.isdigit()
            or int(size) < 1
            or int(size) > MAX_DEB_BYTES
        ):
            raise RollbackPreparationError("previous APT package identity is invalid")
        candidates.append((version_tuple, record))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def _download_bytes(url: str, *, max_bytes: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "nuv-apt-rollback/1"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            if urllib.parse.urlsplit(response.geturl()).scheme != "https":
                raise RollbackPreparationError("APT download redirected away from HTTPS")
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) > max_bytes:
                raise RollbackPreparationError("APT metadata exceeds size limit")
            payload = response.read(max_bytes + 1)
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise RollbackPreparationError(f"cannot download APT metadata: {exc}") from exc
    if len(payload) > max_bytes:
        raise RollbackPreparationError("APT metadata exceeds size limit")
    return payload


def _gpg_fingerprints(key_path: Path, home: Path) -> set[str]:
    imported = subprocess.run(
        ["gpg", "--batch", "--homedir", str(home), "--import", str(key_path)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    if imported.returncode != 0:
        raise RollbackPreparationError("cannot import APT public key")
    listed = subprocess.run(
        ["gpg", "--batch", "--homedir", str(home), "--with-colons", "--list-keys"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )
    if listed.returncode != 0:
        raise RollbackPreparationError("cannot inspect APT public key")
    fingerprints: set[str] = set()
    expect_primary = False
    for line in listed.stdout.splitlines():
        fields = line.split(":")
        if fields[0] == "pub":
            expect_primary = True
        elif fields[0] == "sub":
            expect_primary = False
        elif fields[0] == "fpr" and expect_primary and len(fields) > 9:
            fingerprints.add(fields[9].upper())
            expect_primary = False
    return fingerprints


def _verified_release_text(base_url: str, expected_fingerprint: str) -> str:
    with tempfile.TemporaryDirectory(prefix="nuv-apt-verify-") as temporary:
        root = Path(temporary)
        home = root / "gnupg"
        home.mkdir(mode=0o700)
        key = root / "public.gpg"
        inrelease = root / "InRelease"
        verified = root / "Release"
        key.write_bytes(_download_bytes(f"{base_url}/public.gpg", max_bytes=MAX_METADATA_BYTES))
        inrelease.write_bytes(
            _download_bytes(f"{base_url}/dists/stable/InRelease", max_bytes=MAX_METADATA_BYTES)
        )
        if _gpg_fingerprints(key, home) != {expected_fingerprint}:
            raise RollbackPreparationError("APT public key fingerprint does not match policy")
        result = subprocess.run(
            [
                "gpg",
                "--batch",
                "--homedir",
                str(home),
                "--output",
                str(verified),
                "--decrypt",
                str(inrelease),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            raise RollbackPreparationError("APT InRelease signature verification failed")
        try:
            return verified.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise RollbackPreparationError("verified APT Release is invalid UTF-8") from exc


def _download_deb(url: str, destination: Path, *, expected_sha: str, expected_size: int) -> None:
    if destination.exists() or destination.is_symlink():
        raise RollbackPreparationError("rollback package destination already exists")
    request = urllib.request.Request(url, headers={"User-Agent": "nuv-apt-rollback/1"})
    digest = hashlib.sha256()
    size = 0
    try:
        with urllib.request.urlopen(request, timeout=120) as response, destination.open("xb") as output:
            if urllib.parse.urlsplit(response.geturl()).scheme != "https":
                raise RollbackPreparationError("APT package redirected away from HTTPS")
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_DEB_BYTES or size > expected_size:
                    raise RollbackPreparationError("APT package exceeds authenticated size")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
    except RollbackPreparationError:
        if destination.exists() and destination.is_file() and not destination.is_symlink():
            destination.unlink()
        raise
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        if destination.exists() and destination.is_file() and not destination.is_symlink():
            destination.unlink()
        raise RollbackPreparationError(f"cannot download rollback DEB: {exc}") from exc
    if size != expected_size or digest.hexdigest() != expected_sha:
        destination.unlink()
        raise RollbackPreparationError("APT package digest or size mismatch")


def _gunzip_bounded(payload: bytes) -> bytes:
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as compressed:
            result = compressed.read(MAX_METADATA_BYTES + 1)
    except (OSError, EOFError) as exc:
        raise RollbackPreparationError("APT Packages.gz is invalid") from exc
    if len(result) > MAX_METADATA_BYTES:
        raise RollbackPreparationError("APT Packages content exceeds size limit")
    return result


def prepare_rollback(
    *,
    base_url: str,
    expected_fingerprint: str,
    current_version: str,
    output_dir: Path,
    allow_none: bool,
) -> dict[str, str]:
    parsed = urllib.parse.urlsplit(base_url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise RollbackPreparationError("APT base URL must be a fixed HTTPS origin")
    base_url = base_url.rstrip("/")
    expected_fingerprint = expected_fingerprint.upper()
    if not FINGERPRINT.fullmatch(expected_fingerprint):
        raise RollbackPreparationError("APT fingerprint is invalid")
    _semver_tuple(current_version)
    release = _verified_release_text(base_url, expected_fingerprint)
    packages_path = "main/binary-arm64/Packages.gz"
    packages_sha, packages_size = parse_release_sha256(release, packages_path)
    packages_gz = _download_bytes(
        f"{base_url}/dists/stable/{packages_path}", max_bytes=MAX_METADATA_BYTES
    )
    if len(packages_gz) != packages_size or hashlib.sha256(packages_gz).hexdigest() != packages_sha:
        raise RollbackPreparationError("APT Packages.gz does not match signed Release")
    packages_bytes = _gunzip_bounded(packages_gz)
    try:
        records = parse_packages(packages_bytes.decode("utf-8"))
    except UnicodeError as exc:
        raise RollbackPreparationError("APT Packages is invalid UTF-8") from exc
    selected = select_rollback_record(records, current_version=current_version)
    if selected is None:
        if not allow_none:
            raise RollbackPreparationError("signed APT index has no previous rollback package")
        return {"path": "", "version": "none", "sha256": "none"}
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / Path(selected["Filename"]).name
    _download_deb(
        f"{base_url}/{selected['Filename']}",
        destination,
        expected_sha=selected["SHA256"],
        expected_size=int(selected["Size"]),
    )
    inspected = subprocess.run(
        [
            "dpkg-deb",
            "-f",
            str(destination),
            "Package",
            "Version",
            "Architecture",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )
    if inspected.returncode != 0 or inspected.stdout.splitlines() != [
        "Package: nuv-agent",
        f"Version: {selected['Version']}",
        "Architecture: arm64",
    ]:
        destination.unlink()
        raise RollbackPreparationError("rollback DEB control identity is invalid")
    return {
        "path": str(destination.resolve()),
        "version": selected["Version"],
        "sha256": selected["SHA256"],
    }


def _write_github_output(path: Path, values: Mapping[str, str]) -> None:
    with path.open("a", encoding="utf-8") as output:
        for key, value in values.items():
            if "\n" in value or "\r" in value:
                raise RollbackPreparationError(f"invalid GitHub output: {key}")
            output.write(f"{key}={value}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--expected-fingerprint", required=True)
    parser.add_argument("--current-version", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-none", action="store_true")
    parser.add_argument("--github-output", type=Path)
    arguments = parser.parse_args()
    try:
        result = prepare_rollback(
            base_url=arguments.base_url,
            expected_fingerprint=arguments.expected_fingerprint,
            current_version=arguments.current_version,
            output_dir=arguments.output_dir.resolve(),
            allow_none=arguments.allow_none,
        )
        if arguments.github_output is not None:
            _write_github_output(arguments.github_output, result)
        print(result["path"])
    except RollbackPreparationError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
