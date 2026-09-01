from __future__ import annotations

import os
import posixpath
import re
import shutil
import ssl
import stat
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

from nuvion_updater.errors import UpdaterError, UpdaterSecurityError
from nuvion_updater.secure_io import read_fixed_regular_file
from nuvion_updater.util import (
    ensure_directory,
    fsync_directory,
    parse_digest,
    require_safe_basename,
)

MAX_BOM_BYTES = 1024 * 1024
MAX_SIGNATURE_BYTES = 64 * 1024
DEFAULT_MAX_ARTIFACT_BYTES = 8 * 1024 * 1024 * 1024
DEFAULT_MAX_RELEASE_DOWNLOADS = 16


@dataclass(frozen=True)
class FetchedReleaseFiles:
    directory: Path
    bom_path: Path
    signature_path: Path
    artifact_path: Path | None = None


class ContentAddressedReleaseRepository:
    """Fetches only digest-derived immutable release paths from one pinned origin."""

    def __init__(
        self,
        *,
        base_url: str,
        download_root: str | Path,
        require_root_owner: bool = True,
        allow_file_url: bool = False,
        timeout_seconds: float = 30.0,
        max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
        disk_reserve_bytes: int = 256 * 1024 * 1024,
        max_release_downloads: int = DEFAULT_MAX_RELEASE_DOWNLOADS,
    ) -> None:
        parsed = urllib.parse.urlsplit(str(base_url or ""))
        if parsed.scheme not in ({"https", "file"} if allow_file_url else {"https"}):
            raise UpdaterSecurityError(
                "UNSAFE_RELEASE_ORIGIN", "release base URL must use HTTPS"
            )
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise UpdaterSecurityError(
                "UNSAFE_RELEASE_ORIGIN", "release base URL contains unsafe components"
            )
        if parsed.scheme == "https" and (not parsed.hostname or parsed.port not in {None, 443}):
            raise UpdaterSecurityError(
                "UNSAFE_RELEASE_ORIGIN", "release origin must use the default HTTPS port"
            )
        if parsed.scheme == "file" and parsed.netloc not in {"", "localhost"}:
            raise UpdaterSecurityError(
                "UNSAFE_RELEASE_ORIGIN", "file release origin must be local"
            )
        self.base_url = base_url.rstrip("/") + "/"
        self._base = urllib.parse.urlsplit(self.base_url)
        self.download_root = ensure_directory(
            download_root,
            mode=0o700,
            require_root_owner=require_root_owner,
        )
        self.require_root_owner = require_root_owner
        self.timeout_seconds = timeout_seconds
        self.max_artifact_bytes = max_artifact_bytes
        self.disk_reserve_bytes = disk_reserve_bytes
        if max_release_downloads < 2 or max_release_downloads > 10_000:
            raise ValueError("max_release_downloads must be in [2, 10000]")
        self.max_release_downloads = max_release_downloads
        self._ssl_context = ssl.create_default_context()

    def fetch_manifest(self, bom_digest: str) -> FetchedReleaseFiles:
        digest = parse_digest(bom_digest)
        directory = self._release_download_directory(digest)
        bom_path = directory / "release-bom.json"
        signature_path = directory / "release-bom.json.sig"
        self._download(
            self._relative_release_path(digest, "release-bom.json"),
            bom_path,
            max_bytes=MAX_BOM_BYTES,
        )
        self._download(
            self._relative_release_path(digest, "release-bom.json.sig"),
            signature_path,
            max_bytes=MAX_SIGNATURE_BYTES,
        )
        return FetchedReleaseFiles(
            directory=directory,
            bom_path=bom_path,
            signature_path=signature_path,
        )

    def fetch_artifact(
        self,
        files: FetchedReleaseFiles,
        *,
        bom_digest: str,
        artifact_name: str,
        artifact_size: int,
    ) -> FetchedReleaseFiles:
        digest = parse_digest(bom_digest)
        name = require_safe_basename(artifact_name, field="artifact.name")
        if files.directory != self._release_download_directory(digest):
            raise UpdaterSecurityError(
                "UNSAFE_DOWNLOAD_DIRECTORY", "manifest is outside the digest download directory"
            )
        if (
            isinstance(artifact_size, bool)
            or not isinstance(artifact_size, int)
            or artifact_size < 1
            or artifact_size > self.max_artifact_bytes
        ):
            raise UpdaterSecurityError(
                "ARTIFACT_TOO_LARGE", "artifact size exceeds updater policy"
            )
        self.ensure_disk_capacity(artifact_size)
        artifact_path = files.directory / name
        self._download(
            self._relative_release_path(digest, name),
            artifact_path,
            max_bytes=artifact_size,
            exact_bytes=artifact_size,
        )
        return FetchedReleaseFiles(
            directory=files.directory,
            bom_path=files.bom_path,
            signature_path=files.signature_path,
            artifact_path=artifact_path,
        )

    def ensure_disk_capacity(self, artifact_size: int) -> None:
        free = shutil.disk_usage(self.download_root).free
        required = artifact_size * 2 + self.disk_reserve_bytes
        if free < required:
            raise UpdaterError(
                "INSUFFICIENT_DISK",
                f"insufficient disk for staging: required={required}, free={free}",
            )

    def cleanup_release(self, bom_digest: str) -> None:
        """Remove a completed/failed digest cache without following links."""

        digest = parse_digest(bom_digest)
        path = self.download_root / digest
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_mode & 0o022
            or (self.require_root_owner and metadata.st_uid != 0)
            or path.resolve(strict=True).parent != self.download_root
        ):
            raise UpdaterSecurityError(
                "UNSAFE_DOWNLOAD_DIRECTORY",
                "release download cache cannot be removed safely",
            )
        shutil.rmtree(path)
        fsync_directory(self.download_root)

    def _release_download_directory(self, digest: str) -> Path:
        path = self.download_root / digest
        if not path.exists():
            self._assert_download_capacity()
        path.mkdir(mode=0o700, exist_ok=True)
        metadata = path.lstat()
        if path.is_symlink() or not path.is_dir() or metadata.st_mode & 0o022:
            raise UpdaterSecurityError(
                "UNSAFE_DOWNLOAD_DIRECTORY", "digest download directory is unsafe"
            )
        if self.require_root_owner and metadata.st_uid != 0:
            raise UpdaterSecurityError(
                "UNSAFE_DOWNLOAD_DIRECTORY", "digest download directory must be root-owned"
            )
        os.chmod(path, 0o700)
        resolved = path.resolve(strict=True)
        self._purge_stale_parts(resolved)
        return resolved

    def _assert_download_capacity(self) -> None:
        count = 0
        for candidate in self.download_root.iterdir():
            if not re.fullmatch(r"[0-9a-f]{64}", candidate.name):
                continue
            metadata = candidate.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise UpdaterSecurityError(
                    "UNSAFE_DOWNLOAD_DIRECTORY",
                    "download capacity contains an unsafe digest entry",
                )
            count += 1
        if count >= self.max_release_downloads:
            raise UpdaterError(
                "DOWNLOAD_CAPACITY_EXHAUSTED",
                f"release download limit reached ({self.max_release_downloads}); "
                "operator GC is required",
            )

    @staticmethod
    def _purge_stale_parts(directory: Path) -> None:
        pattern = re.compile(r"^\.[0-9A-Za-z][0-9A-Za-z._+-]{0,254}\.[0-9a-f]{32}\.part$")
        for candidate in directory.iterdir():
            if not pattern.fullmatch(candidate.name):
                continue
            try:
                metadata = candidate.lstat()
            except OSError:
                continue
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise UpdaterSecurityError(
                    "UNSAFE_PARTIAL_DOWNLOAD",
                    "stale partial download is not a regular file",
                )
            if metadata.st_mode & 0o077:
                raise UpdaterSecurityError(
                    "UNSAFE_PARTIAL_DOWNLOAD",
                    "stale partial download permissions are unsafe",
                )
            candidate.unlink()
        fsync_directory(directory)

    @staticmethod
    def _relative_release_path(digest: str, name: str) -> str:
        safe_name = require_safe_basename(name, field="release filename")
        return f"releases/by-bom-sha256/{digest}/{safe_name}"

    def _download(
        self,
        relative_path: str,
        destination: Path,
        *,
        max_bytes: int,
        exact_bytes: int | None = None,
    ) -> None:
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.part"
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(temporary, flags, 0o600)
        total = 0
        try:
            url = urllib.parse.urljoin(self.base_url, relative_path)
            self._validate_final_url(url)
            request = urllib.request.Request(url, headers={"Accept": "application/octet-stream"})
            try:
                response = urllib.request.urlopen(
                    request,
                    timeout=self.timeout_seconds,
                    context=self._ssl_context if self._base.scheme == "https" else None,
                )
            except (urllib.error.URLError, OSError) as exc:
                raise UpdaterError("DOWNLOAD_FAILED", "release download failed") from exc
            with response:
                final_url = response.geturl()
                self._validate_final_url(final_url)
                content_length = response.headers.get("Content-Length")
                if content_length is not None:
                    try:
                        declared_size = int(content_length)
                    except ValueError as exc:
                        raise UpdaterSecurityError(
                            "INVALID_DOWNLOAD", "invalid Content-Length"
                        ) from exc
                    if declared_size < 0 or declared_size > max_bytes:
                        raise UpdaterSecurityError(
                            "DOWNLOAD_TOO_LARGE", "download exceeds declared size limit"
                        )
                    if exact_bytes is not None and declared_size != exact_bytes:
                        raise UpdaterSecurityError(
                            "ARTIFACT_SIZE_MISMATCH", "artifact Content-Length does not match BOM"
                        )
                with os.fdopen(descriptor, "wb", closefd=False) as output:
                    while True:
                        chunk = response.read(min(1024 * 1024, max_bytes + 1 - total))
                        if not chunk:
                            break
                        output.write(chunk)
                        total += len(chunk)
                        if total > max_bytes:
                            raise UpdaterSecurityError(
                                "DOWNLOAD_TOO_LARGE", "download exceeds size limit"
                            )
                    output.flush()
                    os.fsync(output.fileno())
            if exact_bytes is not None and total != exact_bytes:
                raise UpdaterSecurityError(
                    "ARTIFACT_SIZE_MISMATCH", "artifact byte count does not match BOM"
                )
            os.fchmod(descriptor, 0o600)
            os.replace(temporary, destination)
            fsync_directory(destination.parent)
        finally:
            os.close(descriptor)
            if temporary.exists():
                temporary.unlink()

    def _validate_final_url(self, url: str) -> None:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != self._base.scheme:
            raise UpdaterSecurityError(
                "UNSAFE_RELEASE_REDIRECT", "release redirect changed scheme"
            )
        if parsed.scheme == "https":
            if parsed.hostname != self._base.hostname or parsed.port not in {None, 443}:
                raise UpdaterSecurityError(
                    "UNSAFE_RELEASE_REDIRECT", "release redirect changed origin"
                )
        elif parsed.netloc not in {"", "localhost"}:
            raise UpdaterSecurityError(
                "UNSAFE_RELEASE_REDIRECT", "file release origin changed"
            )
        base_path = posixpath.normpath(urllib.parse.unquote(self._base.path or "/"))
        candidate_path = posixpath.normpath(urllib.parse.unquote(parsed.path or "/"))
        if candidate_path != base_path and not candidate_path.startswith(
            base_path.rstrip("/") + "/"
        ):
            raise UpdaterSecurityError(
                "UNSAFE_RELEASE_REDIRECT", "release redirect escaped base path"
            )


def read_ingested_request(
    request_directory: str | Path,
    request_name: str,
    *,
    require_root_owner: bool = True,
    max_bytes: int = 256 * 1024,
) -> bytes:
    """Optional spool adapter; socket requests do not accept filesystem paths."""

    return read_fixed_regular_file(
        request_directory,
        request_name,
        max_bytes=max_bytes,
        require_root_owner=require_root_owner,
        require_private=True,
    )
