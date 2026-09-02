#!/usr/bin/env python3
"""Create or verify a signed, release-bound face artifact manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any


REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
SEMVER_TAG = re.compile(r"^v[0-9]+\.[0-9]+\.[0-9]+$")
COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
FINGERPRINT = re.compile(r"^[0-9A-F]{40}$")
MODEL_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
MODEL_VERSION = re.compile(r"^v[0-9]{4,}$")
ARTIFACT_NAMES = (
    "face_detector.config.pbtxt",
    "face_detector.onnx",
    "face_detector.plan",
)
MAX_DOCUMENT_BYTES = 256 * 1024
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024 * 1024


class FaceManifestError(RuntimeError):
    pass


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _regular_file(path: Path, *, label: str, maximum: int) -> tuple[bytes, int]:
    try:
        before = path.lstat()
    except OSError as exc:
        raise FaceManifestError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise FaceManifestError(f"{label} must be a regular non-symlink file")
    if before.st_size < 1 or before.st_size > maximum:
        raise FaceManifestError(f"{label} size is invalid")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            opened = os.fstat(source.fileno())
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
            after = os.fstat(source.fileno())
        final = path.lstat()
    except OSError as exc:
        raise FaceManifestError(f"cannot read {label}") from exc
    if not (
        _identity(before)
        == _identity(opened)
        == _identity(after)
        == _identity(final)
    ):
        raise FaceManifestError(f"{label} changed while hashing")
    return digest.digest(), before.st_size


def _read_regular_file(path: Path, *, label: str, maximum: int) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise FaceManifestError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise FaceManifestError(f"{label} must be a regular non-symlink file")
    if before.st_size < 1 or before.st_size > maximum:
        raise FaceManifestError(f"{label} size is invalid")
    try:
        with path.open("rb") as source:
            opened = os.fstat(source.fileno())
            raw = source.read(maximum + 1)
            after = os.fstat(source.fileno())
        final = path.lstat()
    except OSError as exc:
        raise FaceManifestError(f"cannot read {label}") from exc
    if len(raw) != before.st_size or not (
        _identity(before)
        == _identity(opened)
        == _identity(after)
        == _identity(final)
    ):
        raise FaceManifestError(f"{label} changed while loading")
    return raw


def _strict_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular_file(path, label=label, maximum=MAX_DOCUMENT_BYTES)

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FaceManifestError(f"duplicate {label} member: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicate)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise FaceManifestError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise FaceManifestError(f"{label} root must be an object")
    return value, raw


def _validate_identity(
    *,
    repository: str,
    release_tag: str,
    component_sha: str,
    model_name: str,
    model_version: str,
    channel_pointer: str,
) -> None:
    if not REPOSITORY.fullmatch(repository):
        raise FaceManifestError("repository identity is invalid")
    if not SEMVER_TAG.fullmatch(release_tag):
        raise FaceManifestError("release tag is invalid")
    if not COMMIT_SHA.fullmatch(component_sha):
        raise FaceManifestError("release component SHA is invalid")
    if not MODEL_NAME.fullmatch(model_name):
        raise FaceManifestError("model name is invalid")
    if not MODEL_VERSION.fullmatch(model_version):
        raise FaceManifestError("model version is invalid")
    expected_prefix = f"gs://nuv-model/pointers/{model_name}/"
    if channel_pointer and (
        not channel_pointer.startswith(expected_prefix)
        or not channel_pointer.endswith(".json")
        or any(value in channel_pointer for value in ("..", "\\", "\r", "\n"))
    ):
        raise FaceManifestError("channel pointer is outside the signed model scope")


def _artifact_metadata(artifact_directory: Path) -> dict[str, dict[str, Any]]:
    root = artifact_directory.absolute()
    try:
        root_metadata = root.lstat()
    except OSError as exc:
        raise FaceManifestError("artifact directory is unavailable") from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise FaceManifestError("artifact directory is invalid")
    result: dict[str, dict[str, Any]] = {}
    for name in ARTIFACT_NAMES:
        digest, size = _regular_file(
            root / name,
            label=f"face artifact {name}",
            maximum=MAX_ARTIFACT_BYTES,
        )
        result[name] = {"sha256": digest.hex(), "sizeBytes": size}
    return result


def build_manifest(
    *,
    repository: str,
    release_tag: str,
    component_sha: str,
    model_name: str,
    model_version: str,
    channel_pointer: str,
    artifact_directory: Path,
) -> dict[str, Any]:
    _validate_identity(
        repository=repository,
        release_tag=release_tag,
        component_sha=component_sha,
        model_name=model_name,
        model_version=model_version,
        channel_pointer=channel_pointer,
    )
    return {
        "schemaVersion": 1,
        "kind": "nuvion-face-artifact-release",
        "repository": repository,
        "releaseTag": release_tag,
        "componentSha": component_sha,
        "modelName": model_name,
        "modelVersion": model_version,
        "channelPointer": channel_pointer,
        "artifacts": _artifact_metadata(artifact_directory),
    }


def _primary_fingerprints(gpg_home: Path) -> set[str]:
    listed = subprocess.run(
        ["gpg", "--batch", "--homedir", str(gpg_home), "--with-colons", "--list-keys"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )
    if listed.returncode != 0:
        raise FaceManifestError("cannot inspect face manifest keyring")
    result: set[str] = set()
    primary = False
    for line in listed.stdout.splitlines():
        fields = line.split(":")
        record = fields[0] if fields else ""
        if record == "pub":
            primary = True
        elif record == "sub":
            primary = False
        elif record == "fpr" and primary and len(fields) > 9:
            fingerprint = fields[9].upper()
            if FINGERPRINT.fullmatch(fingerprint):
                result.add(fingerprint)
            primary = False
    return result


def _verify_signature(
    *,
    manifest_raw: bytes,
    signature_path: Path,
    signer_directory: Path,
    allowed_fingerprints: set[str],
) -> str:
    signature_raw = _read_regular_file(
        signature_path,
        label="face manifest signature",
        maximum=MAX_DOCUMENT_BYTES,
    )
    public_keys = sorted(signer_directory.glob("*.asc"))
    if not public_keys:
        raise FaceManifestError("face manifest signer directory is empty")
    with tempfile.TemporaryDirectory(prefix="nuv-face-manifest-") as temporary:
        gpg_home = Path(temporary)
        gpg_home.chmod(0o700)
        verified_manifest = gpg_home / "manifest.json"
        verified_signature = gpg_home / "manifest.json.asc"
        verified_manifest.write_bytes(manifest_raw)
        verified_signature.write_bytes(signature_raw)
        imported = subprocess.run(
            ["gpg", "--batch", "--homedir", str(gpg_home), "--import", *map(str, public_keys)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )
        if imported.returncode != 0 or _primary_fingerprints(gpg_home) != allowed_fingerprints:
            raise FaceManifestError("face manifest signer files differ from policy")
        verified = subprocess.run(
            [
                "gpg",
                "--batch",
                "--status-fd=1",
                "--homedir",
                str(gpg_home),
                "--verify",
                str(verified_signature),
                str(verified_manifest),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )
    if verified.returncode != 0:
        raise FaceManifestError("face manifest signature verification failed")
    observed: set[str] = set()
    for line in verified.stdout.splitlines():
        if not line.startswith("[GNUPG:] VALIDSIG "):
            continue
        for token in line.split("VALIDSIG", 1)[1].split():
            normalized = token.upper()
            if FINGERPRINT.fullmatch(normalized):
                observed.add(normalized)
    accepted = observed & allowed_fingerprints
    if len(accepted) != 1:
        raise FaceManifestError("face manifest signer is not allowlisted")
    return next(iter(accepted))


def verify_manifest(
    *,
    manifest_path: Path,
    signature_path: Path,
    policy_path: Path,
    signer_directory: Path,
    repository: str,
    release_tag: str,
    component_sha: str,
    model_name: str,
    model_version: str,
    channel_pointer: str,
    artifact_directory: Path,
) -> dict[str, Any]:
    _validate_identity(
        repository=repository,
        release_tag=release_tag,
        component_sha=component_sha,
        model_name=model_name,
        model_version=model_version,
        channel_pointer=channel_pointer,
    )
    policy, _ = _strict_json(policy_path, label="release security policy")
    configured = policy.get("trustedTagSignerFingerprints")
    if (
        not isinstance(configured, list)
        or not configured
        or len(set(configured)) != len(configured)
        or not all(isinstance(value, str) and FINGERPRINT.fullmatch(value) for value in configured)
    ):
        raise FaceManifestError("face manifest signer policy is invalid")
    manifest, manifest_raw = _strict_json(
        manifest_path, label="face artifact manifest"
    )
    canonical_manifest = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    if manifest_raw != canonical_manifest:
        raise FaceManifestError("face artifact manifest is not canonical JSON")
    actual_artifacts = _artifact_metadata(artifact_directory)
    expected = {
        "schemaVersion": 1,
        "kind": "nuvion-face-artifact-release",
        "repository": repository,
        "releaseTag": release_tag,
        "componentSha": component_sha,
        "modelName": model_name,
        "modelVersion": model_version,
        "channelPointer": channel_pointer,
        "artifacts": actual_artifacts,
    }
    if manifest != expected:
        raise FaceManifestError("face artifact manifest differs from exact release inputs or bytes")
    signer = _verify_signature(
        manifest_raw=manifest_raw,
        signature_path=signature_path,
        signer_directory=signer_directory,
        allowed_fingerprints=set(configured),
    )
    return {
        "schemaVersion": 1,
        "status": "VERIFIED",
        "signerFingerprint": signer,
        "manifestSha256": hashlib.sha256(manifest_raw).hexdigest(),
        "componentSha": component_sha,
    }


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("create", "verify"):
        target = subparsers.add_parser(command)
        target.add_argument("--repository", required=True)
        target.add_argument("--release-tag", required=True)
        target.add_argument("--component-sha", required=True)
        target.add_argument("--model-name", required=True)
        target.add_argument("--model-version", required=True)
        target.add_argument("--channel-pointer", default="")
        target.add_argument("--artifact-directory", type=Path, required=True)
    create = subparsers.choices["create"]
    create.add_argument("--output", type=Path, required=True)
    verify = subparsers.choices["verify"]
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--signature", type=Path, required=True)
    verify.add_argument("--policy", type=Path, required=True)
    verify.add_argument("--signer-directory", type=Path, required=True)
    arguments = parser.parse_args()
    common = {
        "repository": arguments.repository,
        "release_tag": arguments.release_tag,
        "component_sha": arguments.component_sha,
        "model_name": arguments.model_name,
        "model_version": arguments.model_version,
        "channel_pointer": arguments.channel_pointer,
        "artifact_directory": arguments.artifact_directory.absolute(),
    }
    try:
        if arguments.command == "create":
            manifest = build_manifest(**common)
            _write_new(
                arguments.output.absolute(),
                (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"),
            )
            result: dict[str, Any] = {"status": "CREATED", **manifest}
        else:
            result = verify_manifest(
                manifest_path=arguments.manifest.absolute(),
                signature_path=arguments.signature.absolute(),
                policy_path=arguments.policy.absolute(),
                signer_directory=arguments.signer_directory.absolute(),
                **common,
            )
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    except (FaceManifestError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
