#!/usr/bin/env python3
"""Create and verify exactly three content-addressed IQ9075 candidate objects."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import signal
import ssl
import stat
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator


BUCKET = "apt.plaidai.io"
PREFIX_ROOT = "releases/by-bom-sha256/"
GCS_HOST = "storage.googleapis.com"
MAX_MANIFEST_BYTES = 64 * 1024
MAX_METADATA_BYTES = 64 * 1024
MAX_TOKEN_BYTES = 16 * 1024
MAX_ARTIFACT_BYTES = 8 * 1024 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 30.0
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")
BROAD_CREDENTIAL_ENVIRONMENT = (
    "GOOGLE_APPLICATION_CREDENTIALS",
    "CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE",
    "GOOGLE_GHA_CREDS_PATH",
    "GOOGLE_CREDENTIALS",
    "GCP_SA_KEY",
)
SAFE_HTTP_STATUSES = frozenset({200, 201, 400, 401, 403, 404, 408, 409, 412, 413, 429, 500, 502, 503, 504})
SAFE_STAGES = frozenset({
    "environment", "token", "manifest", "inputs", "validation", "client",
    "insert", "metadata", "readback", "verification", "cleanup", "interrupted", "unknown",
})
SAFE_OBJECT_FIELDS = frozenset({"bom", "signature", "artifact"})
SAFE_DENIED_PERMISSIONS = frozenset({
    "storage.objects.create", "storage.objects.get", "storage.objects.delete", "storage.objects.getIamPolicy",
})
SAFE_ERROR_CODES = {
    "broad Google credential environment is forbidden": "BROAD_CREDENTIAL_ENVIRONMENT",
    "verified candidate input is closed": "INPUT_CLOSED",
    "verified candidate input is unavailable": "INPUT_UNAVAILABLE",
    "verified candidate input could not be read": "INPUT_READ_FAILED",
    "candidate input changed after validation": "INPUT_CHANGED",
    "candidate input changed during validation": "INPUT_CHANGED",
    "candidate input changed during upload": "INPUT_CHANGED",
    "candidate input exceeds its read boundary": "INPUT_BOUNDARY_EXCEEDED",
    "candidate input path must be absolute": "INPUT_PATH_INVALID",
    "candidate input is unavailable": "INPUT_UNAVAILABLE",
    "candidate input must be a regular non-symlink file": "INPUT_TYPE_INVALID",
    "candidate input has the wrong owner": "INPUT_OWNER_INVALID",
    "candidate input has an invalid size": "INPUT_SIZE_INVALID",
    "candidate input could not be read": "INPUT_READ_FAILED",
    "downscoped token must have mode 0600": "TOKEN_MODE_INVALID",
    "downscoped token filename is invalid": "TOKEN_FILENAME_INVALID",
    "downscoped token is invalid": "TOKEN_INVALID",
    "downscoped token could not be removed": "TOKEN_REMOVAL_FAILED",
    "downscoped token removal could not be verified": "TOKEN_REMOVAL_UNVERIFIED",
    "candidate evidence manifest is invalid": "MANIFEST_INVALID",
    "candidate evidence manifest is not canonical": "MANIFEST_NOT_CANONICAL",
    "candidate evidence manifest fields are invalid": "MANIFEST_FIELDS_INVALID",
    "candidate evidence manifest identity is invalid": "MANIFEST_IDENTITY_INVALID",
    "candidate evidence manifest integer is invalid": "MANIFEST_INTEGER_INVALID",
    "candidate component SHA is invalid": "COMPONENT_SHA_INVALID",
    "candidate version is invalid": "VERSION_INVALID",
    "candidate keyring digest is invalid": "KEYRING_DIGEST_INVALID",
    "candidate object descriptor is invalid": "OBJECT_DESCRIPTOR_INVALID",
    "candidate bootstrap DEB descriptor is invalid": "BOOTSTRAP_DESCRIPTOR_INVALID",
    "candidate content-addressed path is invalid": "CONTENT_ADDRESS_INVALID",
    "candidate input filename differs from its manifest": "INPUT_FILENAME_MISMATCH",
    "candidate input digest differs from its manifest": "INPUT_DIGEST_MISMATCH",
    "candidate release BOM is invalid": "BOM_INVALID",
    "candidate release BOM digest is invalid": "BOM_DIGEST_INVALID",
    "candidate release BOM prefix differs from its manifest": "BOM_PREFIX_MISMATCH",
    "candidate inputs must be distinct files": "INPUTS_NOT_DISTINCT",
    "candidate object set escaped its exact prefix": "OBJECT_PREFIX_MISMATCH",
    "Cloud Storage response exceeded its boundary": "GCS_RESPONSE_TOO_LARGE",
    "Cloud Storage insert failed": "GCS_INSERT_FAILED",
    "Cloud Storage insert metadata is invalid": "GCS_INSERT_METADATA_INVALID",
    "Cloud Storage insert returned an unexpected status": "GCS_INSERT_STATUS_INVALID",
    "Cloud Storage metadata lookup failed": "GCS_METADATA_FAILED",
    "Cloud Storage metadata is invalid": "GCS_METADATA_INVALID",
    "Cloud Storage generation-pinned read failed": "GCS_READBACK_FAILED",
    "Cloud Storage object exceeded expected size": "GCS_READBACK_TOO_LARGE",
    "Cloud Storage object identity differs": "GCS_OBJECT_IDENTITY_MISMATCH",
    "Cloud Storage generation is invalid": "GCS_GENERATION_INVALID",
    "Cloud Storage object size differs": "GCS_OBJECT_SIZE_MISMATCH",
    "remote bytes differ from the exact candidate input": "GCS_BYTES_MISMATCH",
    "publication interrupted": "PUBLICATION_INTERRUPTED",
    "unexpected internal failure": "INTERNAL_FAILURE",
}


class PublishError(RuntimeError):
    """A fail-closed candidate publication or verification error."""

    def __init__(
        self, message: str, *, stage: str = "unknown", http_status: int | None = None,
        object_field: str | None = None, denied_permission: str | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.http_status = http_status
        self.object_field = object_field
        self.denied_permission = denied_permission


@dataclass
class _PublicationProgress:
    stage: str = "environment"
    object_field: str | None = None


def _safe_failure(error: PublishError) -> dict[str, object]:
    """Emit only local enums; never exception text, requests, responses or tokens."""
    message = error.args[0] if error.args and type(error.args[0]) is str else ""
    return {
        "kind": "nuvion-iq9075-candidate-gcs-stage-failure",
        "code": SAFE_ERROR_CODES.get(message, "PUBLICATION_BOUNDARY_FAILED"),
        "stage": error.stage if type(error.stage) is str and error.stage in SAFE_STAGES else "unknown",
        "httpStatus": error.http_status if type(error.http_status) is int and error.http_status in SAFE_HTTP_STATUSES else None,
        "object": error.object_field if type(error.object_field) is str and error.object_field in SAFE_OBJECT_FIELDS else None,
        "deniedPermission": error.denied_permission if type(error.denied_permission) is str and error.denied_permission in SAFE_DENIED_PERMISSIONS else None,
    }


def _safe_denied_permission(body: bytes) -> str | None:
    if type(body) is not bytes or len(body) > MAX_METADATA_BYTES:
        return None
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        return None
    error = value.get("error") if isinstance(value, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    if not isinstance(message, str):
        return None
    matches = [permission for permission in SAFE_DENIED_PERMISSIONS
               if re.search(r"(?<![\w.])" + re.escape(permission) + r"(?![\w.])", message)]
    return matches[0] if len(matches) == 1 else None


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _stat_identity(
    details: os.stat_result,
) -> tuple[int, int, int, int, int, int, int, int]:
    return (
        details.st_dev,
        details.st_ino,
        details.st_size,
        details.st_mode,
        details.st_uid,
        details.st_gid,
        details.st_mtime_ns,
        details.st_ctime_ns,
    )


@dataclass
class VerifiedInput:
    """One open descriptor binding validation, BOM parsing, and upload bytes."""

    path: Path
    descriptor: int
    details: os.stat_result
    sha256: str

    @property
    def size(self) -> int:
        return self.details.st_size

    def assert_unchanged(self) -> None:
        if self.descriptor < 0:
            raise PublishError("verified candidate input is closed")
        try:
            current = os.fstat(self.descriptor)
        except OSError:
            raise PublishError("verified candidate input is unavailable") from None
        if _stat_identity(current) != _stat_identity(self.details):
            raise PublishError("candidate input changed after validation")

    def iter_chunks(self) -> Iterator[bytes]:
        self.assert_unchanged()
        offset = 0
        while offset < self.size:
            try:
                chunk = os.pread(
                    self.descriptor, min(CHUNK_BYTES, self.size - offset), offset
                )
            except OSError:
                raise PublishError("verified candidate input could not be read") from None
            if not chunk:
                raise PublishError("candidate input changed after validation")
            offset += len(chunk)
            yield chunk
        self.assert_unchanged()

    def read_bytes(self, *, maximum_bytes: int = MAX_ARTIFACT_BYTES) -> bytes:
        if self.size > maximum_bytes:
            raise PublishError("candidate input exceeds its read boundary")
        payload = b"".join(self.iter_chunks())
        if len(payload) != self.size:
            raise PublishError("candidate input changed after validation")
        return payload

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


def _open_verified_input(
    path: Path,
    *,
    maximum_bytes: int,
    require_mode_0600: bool = False,
) -> VerifiedInput:
    if not path.is_absolute():
        raise PublishError("candidate input path must be absolute")
    try:
        details = path.lstat()
    except OSError:
        raise PublishError("candidate input is unavailable") from None
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise PublishError("candidate input must be a regular non-symlink file")
    if hasattr(os, "getuid") and details.st_uid != os.getuid():
        raise PublishError("candidate input has the wrong owner")
    if require_mode_0600 and stat.S_IMODE(details.st_mode) != 0o600:
        raise PublishError("downscoped token must have mode 0600")
    if details.st_size <= 0 or details.st_size > maximum_bytes:
        raise PublishError("candidate input has an invalid size")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    keep_open = False
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if _stat_identity(opened) != _stat_identity(details):
            raise PublishError("candidate input changed during validation")
        digest = hashlib.sha256()
        offset = 0
        while offset < opened.st_size:
            chunk = os.pread(
                descriptor, min(CHUNK_BYTES, opened.st_size - offset), offset
            )
            if not chunk:
                raise PublishError("candidate input changed during validation")
            offset += len(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        if _stat_identity(after) != _stat_identity(opened):
            raise PublishError("candidate input changed during validation")
        keep_open = True
        return VerifiedInput(path, descriptor, opened, digest.hexdigest())
    except PublishError:
        raise
    except OSError:
        raise PublishError("candidate input could not be read") from None
    finally:
        if descriptor >= 0 and not keep_open:
            os.close(descriptor)


def _read_manifest(source: VerifiedInput) -> dict[str, object]:
    try:
        raw = source.read_bytes(maximum_bytes=MAX_MANIFEST_BYTES)
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise PublishError("candidate evidence manifest is invalid") from None
    if not isinstance(payload, dict) or raw != _canonical_json(payload):
        raise PublishError("candidate evidence manifest is not canonical")
    expected_keys = {
        "schemaVersion",
        "kind",
        "workflowRunId",
        "workflowRunAttempt",
        "componentSha",
        "agentVersion",
        "releaseSequence",
        "artifact",
        "bootstrapDeb",
        "bom",
        "signature",
        "releaseKeyringSha256",
        "contentAddressedPath",
    }
    if set(payload) != expected_keys:
        raise PublishError("candidate evidence manifest fields are invalid")
    if payload.get("schemaVersion") != 1 or payload.get("kind") != (
        "nuvion-iq9075-signed-evidence-candidate"
    ):
        raise PublishError("candidate evidence manifest identity is invalid")
    for integer_field in ("workflowRunId", "workflowRunAttempt", "releaseSequence"):
        value = payload.get(integer_field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise PublishError("candidate evidence manifest integer is invalid")
    component_sha = payload.get("componentSha")
    if not isinstance(component_sha, str) or not re.fullmatch(r"[0-9a-f]{40}", component_sha):
        raise PublishError("candidate component SHA is invalid")
    version = payload.get("agentVersion")
    if not isinstance(version, str) or VERSION_PATTERN.fullmatch(version) is None:
        raise PublishError("candidate version is invalid")
    keyring_sha = payload.get("releaseKeyringSha256")
    if not isinstance(keyring_sha, str) or SHA256_PATTERN.fullmatch(keyring_sha) is None:
        raise PublishError("candidate keyring digest is invalid")

    expected_names = {
        "artifact": f"nuv-agent_{version}_iq9075-aarch64.agent-bundle.tar.gz",
        "bom": f"nuv-agent_{version}_iq9075-aarch64.release-bom.json",
        "signature": f"nuv-agent_{version}_iq9075-aarch64.release-bom.json.sig",
    }
    for field, expected_name in expected_names.items():
        value = payload.get(field)
        if not isinstance(value, dict) or set(value) != {"name", "sha256"}:
            raise PublishError("candidate object descriptor is invalid")
        name = value.get("name")
        digest = value.get("sha256")
        if (
            name != expected_name
            or not isinstance(name, str)
            or NAME_PATTERN.fullmatch(name) is None
            or not isinstance(digest, str)
            or SHA256_PATTERN.fullmatch(digest) is None
        ):
            raise PublishError("candidate object descriptor is invalid")

    bootstrap_deb = payload.get("bootstrapDeb")
    if not isinstance(bootstrap_deb, dict) or set(bootstrap_deb) != {
        "name",
        "sha256",
        "sizeBytes",
    }:
        raise PublishError("candidate bootstrap DEB descriptor is invalid")
    if (
        bootstrap_deb.get("name") != f"nuv-agent_{version}_arm64.deb"
        or not isinstance(bootstrap_deb.get("sha256"), str)
        or SHA256_PATTERN.fullmatch(bootstrap_deb["sha256"]) is None
        or type(bootstrap_deb.get("sizeBytes")) is not int
        or not 0 < bootstrap_deb["sizeBytes"] <= MAX_ARTIFACT_BYTES
    ):
        raise PublishError("candidate bootstrap DEB descriptor is invalid")

    content_path = payload.get("contentAddressedPath")
    if not isinstance(content_path, str) or re.fullmatch(
        r"releases/by-bom-sha256/[0-9a-f]{64}", content_path
    ) is None:
        raise PublishError("candidate content-addressed path is invalid")
    return payload


def _load_token(path: Path) -> str:
    if path.name != "cab-token":
        raise PublishError("downscoped token filename is invalid")
    source = _open_verified_input(
        path, maximum_bytes=MAX_TOKEN_BYTES, require_mode_0600=True
    )
    try:
        raw = source.read_bytes(maximum_bytes=MAX_TOKEN_BYTES)
        token = raw.decode("ascii").strip()
    except UnicodeDecodeError:
        raise PublishError("downscoped token is invalid") from None
    finally:
        source.close()
    if raw != token.encode("ascii") + b"\n" or not 16 <= len(token) <= MAX_TOKEN_BYTES:
        raise PublishError("downscoped token is invalid")
    if any(character.isspace() or ord(character) < 0x21 for character in token):
        raise PublishError("downscoped token is invalid")
    return token


def _destroy_token(path: Path) -> None:
    if not path.is_absolute() or path.name != "cab-token":
        return
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        raise PublishError("downscoped token could not be removed") from None
    if path.exists() or path.is_symlink():
        raise PublishError("downscoped token removal could not be verified")


def _bounded_response(response: http.client.HTTPResponse) -> bytes:
    body = response.read(MAX_METADATA_BYTES + 1)
    if len(body) > MAX_METADATA_BYTES:
        raise PublishError("Cloud Storage response exceeded its boundary")
    return body


class GoogleStorageJsonClient:
    """Minimal JSON API client: object insert, exact metadata GET, exact media GET."""

    def __init__(self, bearer_token: str) -> None:
        self._authorization = "Bearer " + bearer_token
        self._ssl_context = ssl.create_default_context()

    def _connection(self) -> http.client.HTTPSConnection:
        return http.client.HTTPSConnection(
            GCS_HOST,
            443,
            timeout=REQUEST_TIMEOUT_SECONDS,
            context=self._ssl_context,
        )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": self._authorization,
            "Accept-Encoding": "identity",
            "User-Agent": "nuvion-iq9075-candidate-publisher/1",
        }

    def close(self) -> None:
        self._authorization = ""

    @staticmethod
    def _object_segment(object_name: str) -> str:
        return urllib.parse.quote(object_name, safe="")

    def insert(
        self, object_name: str, source: VerifiedInput
    ) -> tuple[int, dict[str, object]]:
        source.assert_unchanged()
        encoded_name = urllib.parse.quote(object_name, safe="")
        target = (
            "/upload/storage/v1/b/apt.plaidai.io/o?uploadType=media&name="
            + encoded_name
            + "&ifGenerationMatch=0"
        )
        connection = self._connection()
        response_status = None
        try:
            connection.putrequest("POST", target, skip_accept_encoding=True)
            for key, value in self._headers().items():
                connection.putheader(key, value)
            connection.putheader("Content-Type", "application/octet-stream")
            connection.putheader("Content-Length", str(source.size))
            connection.endheaders()
            sent_digest = hashlib.sha256()
            sent_size = 0
            for chunk in source.iter_chunks():
                connection.send(chunk)
                sent_digest.update(chunk)
                sent_size += len(chunk)
            if sent_size != source.size or sent_digest.hexdigest() != source.sha256:
                raise PublishError("candidate input changed during upload")
            response = connection.getresponse()
            response_status = response.status
            body = _bounded_response(response)
            if response.status == 412:
                return 412, {}
            if response.status not in (200, 201):
                raise PublishError("Cloud Storage insert failed", denied_permission=(
                    _safe_denied_permission(body) if response.status == 403 else None
                ))
            try:
                metadata = json.loads(body)
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise PublishError("Cloud Storage insert metadata is invalid") from None
            if not isinstance(metadata, dict):
                raise PublishError("Cloud Storage insert metadata is invalid")
            return response.status, metadata
        except PublishError as exc:
            exc.stage = "insert"
            exc.http_status = response_status
            raise
        except (OSError, http.client.HTTPException, ssl.SSLError):
            raise PublishError("Cloud Storage insert failed", stage="insert", http_status=response_status) from None
        finally:
            connection.close()

    def metadata(self, object_name: str) -> dict[str, object]:
        target = (
            "/storage/v1/b/apt.plaidai.io/o/" + self._object_segment(object_name)
        )
        connection = self._connection()
        response_status = None
        try:
            connection.request("GET", target, headers=self._headers())
            response = connection.getresponse()
            response_status = response.status
            body = _bounded_response(response)
            if response.status != 200:
                raise PublishError("Cloud Storage metadata lookup failed", denied_permission=(
                    _safe_denied_permission(body) if response.status == 403 else None
                ))
            try:
                metadata = json.loads(body)
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise PublishError("Cloud Storage metadata is invalid") from None
            if not isinstance(metadata, dict):
                raise PublishError("Cloud Storage metadata is invalid")
            return metadata
        except PublishError as exc:
            exc.stage = "metadata"
            exc.http_status = response_status
            raise
        except (OSError, http.client.HTTPException, ssl.SSLError):
            raise PublishError("Cloud Storage metadata lookup failed", stage="metadata", http_status=response_status) from None
        finally:
            connection.close()

    def digest(
        self, object_name: str, generation: str, *, maximum_bytes: int
    ) -> tuple[str, int]:
        if re.fullmatch(r"[1-9][0-9]*", generation) is None:
            raise PublishError("Cloud Storage generation is invalid")
        query = urllib.parse.urlencode({"alt": "media", "generation": generation})
        target = (
            "/storage/v1/b/apt.plaidai.io/o/"
            + self._object_segment(object_name)
            + "?"
            + query
        )
        connection = self._connection()
        digest = hashlib.sha256()
        total = 0
        response_status = None
        try:
            connection.request("GET", target, headers=self._headers())
            response = connection.getresponse()
            response_status = response.status
            if response.status != 200:
                body = _bounded_response(response)
                raise PublishError("Cloud Storage generation-pinned read failed", denied_permission=(
                    _safe_denied_permission(body) if response.status == 403 else None
                ))
            while True:
                chunk = response.read(min(CHUNK_BYTES, maximum_bytes - total + 1))
                if not chunk:
                    break
                total += len(chunk)
                if total > maximum_bytes:
                    raise PublishError("Cloud Storage object exceeded expected size")
                digest.update(chunk)
            return digest.hexdigest(), total
        except PublishError as exc:
            exc.stage = "readback"
            exc.http_status = response_status
            raise
        except (OSError, http.client.HTTPException, ssl.SSLError):
            raise PublishError("Cloud Storage generation-pinned read failed", stage="readback", http_status=response_status) from None
        finally:
            connection.close()


def _validated_metadata(
    metadata: dict[str, object], *, object_name: str, expected_size: int
) -> str:
    if metadata.get("bucket") != BUCKET or metadata.get("name") != object_name:
        raise PublishError("Cloud Storage object identity differs")
    generation = metadata.get("generation")
    size = metadata.get("size")
    if not isinstance(generation, str) or re.fullmatch(r"[1-9][0-9]*", generation) is None:
        raise PublishError("Cloud Storage generation is invalid")
    if not isinstance(size, str) or not size.isdigit() or int(size) != expected_size:
        raise PublishError("Cloud Storage object size differs")
    return generation


def _validate_local_descriptor(
    manifest: dict[str, object], field: str, source: VerifiedInput
) -> tuple[str, int]:
    descriptor = manifest[field]
    if not isinstance(descriptor, dict):
        raise PublishError("candidate object descriptor is invalid")
    if source.path.name != descriptor["name"]:
        raise PublishError("candidate input filename differs from its manifest")
    if source.sha256 != descriptor["sha256"]:
        raise PublishError("candidate input digest differs from its manifest")
    return source.sha256, source.size


def _validate_bom_prefix(bom_source: VerifiedInput, manifest: dict[str, object]) -> str:
    try:
        raw = bom_source.read_bytes(maximum_bytes=MAX_MANIFEST_BYTES)
        bom = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise PublishError("candidate release BOM is invalid") from None
    digest_value = bom.get("bomDigest") if isinstance(bom, dict) else None
    if not isinstance(digest_value, str) or re.fullmatch(
        r"sha256:[0-9a-f]{64}", digest_value
    ) is None:
        raise PublishError("candidate release BOM digest is invalid")
    digest = digest_value.removeprefix("sha256:")
    expected_path = PREFIX_ROOT + digest
    if manifest.get("contentAddressedPath") != expected_path:
        raise PublishError("candidate release BOM prefix differs from its manifest")
    return expected_path + "/"


def _publish_with_token(
    *,
    token_path: Path,
    manifest_path: Path,
    artifact_path: Path,
    bom_path: Path,
    signature_path: Path,
    client_factory: Callable[[str], object],
    progress: _PublicationProgress | None = None,
) -> dict[str, object]:
    progress = progress if progress is not None else _PublicationProgress()
    for variable in BROAD_CREDENTIAL_ENVIRONMENT:
        if os.environ.get(variable):
            raise PublishError("broad Google credential environment is forbidden")
    progress.stage = "token"
    token = _load_token(token_path)
    _destroy_token(token_path)
    progress.stage = "manifest"
    manifest_source = _open_verified_input(
        manifest_path, maximum_bytes=MAX_MANIFEST_BYTES
    )
    try:
        manifest = _read_manifest(manifest_source)
    finally:
        manifest_source.close()

    local: dict[str, VerifiedInput] = {}
    client: object | None = None
    try:
        progress.stage = "inputs"
        for field, path in (
            ("artifact", artifact_path),
            ("bom", bom_path),
            ("signature", signature_path),
        ):
            progress.object_field = field
            local[field] = _open_verified_input(
                path, maximum_bytes=MAX_ARTIFACT_BYTES
            )
        progress.stage = "validation"
        progress.object_field = None
        inode_ids = {(item.details.st_dev, item.details.st_ino) for item in local.values()}
        if len(inode_ids) != len(local):
            raise PublishError("candidate inputs must be distinct files")
        identities = {}
        for field, source in local.items():
            progress.object_field = field
            identities[field] = _validate_local_descriptor(manifest, field, source)
        progress.object_field = "bom"
        prefix = _validate_bom_prefix(local["bom"], manifest)
        progress.object_field = None
        remote = {
            "artifact": prefix + local["artifact"].path.name,
            "bom": prefix + "release-bom.json",
            "signature": prefix + "release-bom.json.sig",
        }
        if len(set(remote.values())) != 3 or any(
            not name.startswith(prefix) or not name.startswith(PREFIX_ROOT)
            for name in remote.values()
        ):
            raise PublishError("candidate object set escaped its exact prefix")

        progress.stage = "client"
        client = client_factory(token)
        token = ""
        published: list[dict[str, object]] = []
        for field in ("bom", "signature", "artifact"):
            progress.stage = "insert"
            progress.object_field = field
            object_name = remote[field]
            expected_digest, expected_size = identities[field]
            status, metadata = client.insert(object_name, local[field])
            created = status in (200, 201)
            if status == 412:
                progress.stage = "metadata"
                metadata = client.metadata(object_name)
            elif not created:
                raise PublishError("Cloud Storage insert returned an unexpected status")
            try:
                progress.stage = "verification"
                generation = _validated_metadata(
                    metadata, object_name=object_name, expected_size=expected_size
                )
            except PublishError:
                if status == 412:
                    raise PublishError(
                        "remote bytes differ from the exact candidate input"
                    ) from None
                raise
            progress.stage = "readback"
            remote_digest, remote_size = client.digest(
                object_name, generation, maximum_bytes=expected_size
            )
            progress.stage = "verification"
            if remote_size != expected_size or remote_digest != expected_digest:
                raise PublishError("remote bytes differ from the exact candidate input")
            published.append(
                {
                    "name": object_name,
                    "sha256": expected_digest,
                    "sizeBytes": expected_size,
                    "generation": generation,
                    "created": created,
                }
            )
        return {
            "schemaVersion": 1,
            "kind": "nuvion-iq9075-candidate-gcs-stage",
            "bucket": BUCKET,
            "prefix": prefix,
            "objects": published,
        }
    finally:
        token = ""
        try:
            if client is not None:
                close_client = getattr(client, "close", None)
                if callable(close_client):
                    close_client()
        finally:
            for source in local.values():
                source.close()


def publish(
    *,
    token_path: Path,
    manifest_path: Path,
    artifact_path: Path,
    bom_path: Path,
    signature_path: Path,
    client_factory: Callable[[str], object] = GoogleStorageJsonClient,
) -> dict[str, object]:
    token_path = Path(token_path)
    progress = _PublicationProgress()
    try:
        return _publish_with_token(
            token_path=token_path,
            manifest_path=Path(manifest_path),
            artifact_path=Path(artifact_path),
            bom_path=Path(bom_path),
            signature_path=Path(signature_path),
            client_factory=client_factory,
            progress=progress,
        )
    except PublishError as exc:
        if exc.stage == "unknown":
            exc.stage = progress.stage
        if exc.object_field is None:
            exc.object_field = progress.object_field
        raise
    except Exception:
        raise PublishError("unexpected internal failure", stage=progress.stage,
                           object_field=progress.object_field) from None
    finally:
        try:
            _destroy_token(token_path)
        except PublishError as exc:
            exc.stage = "cleanup"
            raise


def _install_cleanup_signal_handlers() -> None:
    def interrupt(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, interrupt)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--bom", type=Path, required=True)
    parser.add_argument("--signature", type=Path, required=True)
    arguments = parser.parse_args(argv)
    _install_cleanup_signal_handlers()
    try:
        result = publish(
            token_path=arguments.token_file,
            manifest_path=arguments.manifest,
            artifact_path=arguments.artifact,
            bom_path=arguments.bom,
            signature_path=arguments.signature,
        )
    except (PublishError, KeyboardInterrupt) as exc:
        error = exc if isinstance(exc, PublishError) else PublishError(
            "publication interrupted", stage="interrupted"
        )
        print("candidate Cloud Storage publication failed closed", file=sys.stderr)
        print(json.dumps(_safe_failure(error), sort_keys=True, separators=(",", ":")), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
