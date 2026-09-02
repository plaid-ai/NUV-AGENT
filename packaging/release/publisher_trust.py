"""SHA-256 identity for the complete trusted publisher tree and workflow."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import subprocess
from pathlib import Path
from typing import Any


COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
WORKFLOW_RELATIVE_PATH = Path(".github/workflows/release-publish.yml")


class PublisherTrustError(RuntimeError):
    pass


def _git(root: Path, *arguments: str, text: bool = True) -> subprocess.CompletedProcess[Any]:
    environment = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_OPTIONAL_LOCKS": "0",
    }
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
        timeout=30,
        check=False,
        env=environment,
    )
    if result.returncode != 0:
        raise PublisherTrustError("trusted publisher git identity is unavailable")
    return result


def _regular_file_hashes(
    path: Path, *, git_object_format: str | None = None
) -> tuple[bytes, str | None]:
    try:
        before = path.lstat()
    except OSError as exc:
        raise PublisherTrustError(f"trusted publisher file is unavailable: {path}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise PublisherTrustError(f"trusted publisher file is not regular: {path}")
    digest = hashlib.sha256()
    git_digest = hashlib.new(git_object_format) if git_object_format else None
    if git_digest is not None:
        git_digest.update(f"blob {before.st_size}\0".encode("ascii"))
    try:
        with path.open("rb") as source:
            opened = os.fstat(source.fileno())
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                if git_digest is not None:
                    git_digest.update(chunk)
            after = os.fstat(source.fileno())
        final = path.lstat()
    except OSError as exc:
        raise PublisherTrustError(f"cannot hash trusted publisher file: {path}") from exc
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if not (identity(before) == identity(opened) == identity(after) == identity(final)):
        raise PublisherTrustError(f"trusted publisher file changed while hashing: {path}")
    return digest.digest(), git_digest.hexdigest() if git_digest is not None else None


def _dirty_checkout(root: Path) -> bool:
    # Ignored files are execution surfaces too (for example a forged Python
    # bytecode cache imported before a source helper). Trusted checkouts are
    # deliberately fresh, so fail closed on tracked, untracked, or ignored
    # entries rather than trusting .gitignore as a security boundary.
    return bool(
        _git(
            root,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--ignored=matching",
        ).stdout
    )


def publisher_surface(root: Path, *, expected_sha: str) -> dict[str, str]:
    root = root.resolve()
    if not COMMIT_SHA.fullmatch(expected_sha):
        raise PublisherTrustError("trusted publisher SHA is invalid")
    if _git(root, "rev-parse", "HEAD").stdout.strip() != expected_sha:
        raise PublisherTrustError("trusted publisher checkout SHA differs from attestation")
    if _dirty_checkout(root):
        raise PublisherTrustError("trusted publisher checkout is not clean")

    object_format = _git(root, "rev-parse", "--show-object-format").stdout.strip()
    if object_format not in {"sha1", "sha256"}:
        raise PublisherTrustError("trusted publisher Git object format is unsupported")

    head_raw = _git(root, "ls-tree", "-r", "--full-tree", "-z", "HEAD", text=False).stdout
    if not isinstance(head_raw, bytes) or not head_raw:
        raise PublisherTrustError("trusted publisher HEAD tree is empty")
    head: dict[str, tuple[str, str]] = {}
    for raw_record in head_raw.split(b"\0"):
        if not raw_record:
            continue
        try:
            metadata, raw_path = raw_record.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii").split(" ")
            path_text = raw_path.decode("utf-8")
        except (UnicodeError, ValueError) as exc:
            raise PublisherTrustError("trusted publisher HEAD record is invalid") from exc
        if (
            mode not in {"100644", "100755"}
            or object_type != "blob"
            or not re.fullmatch(r"[0-9a-f]{40,64}", object_id)
            or not path_text
            or path_text in head
        ):
            raise PublisherTrustError("trusted publisher HEAD contains an unsafe entry")
        head[path_text] = (mode, object_id)

    index = _git(root, "ls-files", "--stage", "-z", text=False).stdout
    if not isinstance(index, bytes) or not index:
        raise PublisherTrustError("trusted publisher index is empty")
    tree = hashlib.sha256()
    seen: set[str] = set()
    for raw_record in index.split(b"\0"):
        if not raw_record:
            continue
        try:
            metadata, raw_path = raw_record.split(b"\t", 1)
            mode, object_id, stage = metadata.decode("ascii").split(" ")
            path_text = raw_path.decode("utf-8")
        except (UnicodeError, ValueError) as exc:
            raise PublisherTrustError("trusted publisher index record is invalid") from exc
        if (
            mode not in {"100644", "100755"}
            or not re.fullmatch(r"[0-9a-f]{40,64}", object_id)
            or stage != "0"
            or not path_text
            or path_text in seen
        ):
            raise PublisherTrustError("trusted publisher index contains an unsafe entry")
        if head.get(path_text) != (mode, object_id):
            raise PublisherTrustError("trusted publisher index differs from HEAD")
        seen.add(path_text)
        path = root / path_text
        file_mode = path.lstat().st_mode
        expected_permissions = 0o755 if mode == "100755" else 0o644
        if stat.S_IMODE(file_mode) != expected_permissions:
            raise PublisherTrustError(f"trusted publisher mode mismatch: {path_text}")
        content_digest, worktree_object_id = _regular_file_hashes(
            path, git_object_format=object_format
        )
        if worktree_object_id != object_id:
            raise PublisherTrustError(
                f"trusted publisher bytes differ from HEAD: {path_text}"
            )
        tree.update(mode.encode("ascii"))
        tree.update(b"\0")
        tree.update(raw_path)
        tree.update(b"\0")
        tree.update(content_digest)

    if seen != set(head):
        raise PublisherTrustError("trusted publisher index file set differs from HEAD")

    workflow_path = root / WORKFLOW_RELATIVE_PATH
    workflow_digest = _regular_file_hashes(workflow_path)[0].hex()
    if _git(root, "rev-parse", "HEAD").stdout.strip() != expected_sha:
        raise PublisherTrustError("trusted publisher SHA changed during verification")
    if _dirty_checkout(root):
        raise PublisherTrustError("trusted publisher checkout changed during verification")
    return {
        "publisherTreeSha256": tree.hexdigest(),
        "workflowSha256": workflow_digest,
    }


def verify_executing_workflow(
    publisher_root: Path,
    executing_workflow: Path,
    *,
    expected_workflow_sha256: str,
) -> None:
    executing_digest = _regular_file_hashes(executing_workflow.resolve())[0].hex()
    publisher_digest = _regular_file_hashes(
        publisher_root.resolve() / WORKFLOW_RELATIVE_PATH
    )[0].hex()
    if (
        executing_digest != expected_workflow_sha256
        or publisher_digest != expected_workflow_sha256
    ):
        raise PublisherTrustError(
            "executing default-branch workflow differs from trusted publisher bytes"
        )
