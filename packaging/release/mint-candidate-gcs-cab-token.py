#!/usr/bin/env python3
"""Mint one short-lived, prefix-bound token for IQ9075 candidate staging.

This is the only process in the candidate stage that may see the source Google
credential.  It removes that credential before contacting STS and never emits
either the source or downscoped token on stdout/stderr.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import signal
import ssl
import stat
import subprocess
import sys
import tempfile
import urllib.parse
from pathlib import Path
from typing import Callable


STS_HOST = "sts.googleapis.com"
STS_PATH = "/v1/token"
ACCESS_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:access_token"
TOKEN_EXCHANGE_GRANT = "urn:ietf:params:oauth:grant-type:token-exchange"
MAX_CREDENTIAL_BYTES = 128 * 1024
MAX_POLICY_BYTES = 16 * 1024
MAX_STS_RESPONSE_BYTES = 64 * 1024
MAX_TOKEN_BYTES = 16 * 1024
MAX_TOKEN_LIFETIME_SECONDS = 3600
REQUEST_TIMEOUT_SECONDS = 30.0

EXPECTED_POLICY = {
    "accessBoundary": {
        "accessBoundaryRules": [
            {
                "availabilityCondition": {
                    "expression": (
                        "resource.type == 'storage.googleapis.com/Object' && "
                        "resource.name.startsWith('projects/_/buckets/"
                        "apt.plaidai.io/objects/releases/by-bom-sha256/')"
                    ),
                    "title": "iq9075-candidate-content-addressed-v1",
                },
                "availablePermissions": [
                    "inRole:roles/storage.objectCreator",
                    "inRole:roles/storage.objectViewer",
                ],
                "availableResource": (
                    "//storage.googleapis.com/projects/_/buckets/apt.plaidai.io"
                ),
            }
        ]
    }
}


class CabError(RuntimeError):
    """A fail-closed candidate credential boundary violation."""


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _validate_owned_regular_file(
    path: Path, *, maximum_bytes: int, require_mode_0600: bool
) -> os.stat_result:
    if not path.is_absolute():
        raise CabError("security-sensitive path must be absolute")
    try:
        details = path.lstat()
    except OSError as exc:
        raise CabError("security-sensitive input is unavailable") from None
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise CabError("security-sensitive input must be a regular non-symlink file")
    if hasattr(os, "getuid") and details.st_uid != os.getuid():
        raise CabError("security-sensitive input has the wrong owner")
    mode = stat.S_IMODE(details.st_mode)
    if require_mode_0600 and mode != 0o600:
        raise CabError("security-sensitive input must have mode 0600")
    if details.st_size <= 0 or details.st_size > maximum_bytes:
        raise CabError("security-sensitive input has an invalid size")
    return details


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


def _read_bound_file(
    path: Path, *, expected: os.stat_result, maximum_bytes: int
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if _stat_identity(opened) != _stat_identity(expected):
            raise CabError("security-sensitive input changed while opening")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(8192, maximum_bytes + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > maximum_bytes:
                raise CabError("security-sensitive input exceeds its size boundary")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if _stat_identity(after) != _stat_identity(opened) or total != opened.st_size:
            raise CabError("security-sensitive input changed while reading")
        return b"".join(chunks)
    except CabError:
        raise
    except OSError:
        raise CabError("security-sensitive input could not be read") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_exact_policy(path: Path) -> tuple[dict[str, object], bytes]:
    details = _validate_owned_regular_file(
        path, maximum_bytes=MAX_POLICY_BYTES, require_mode_0600=False
    )
    try:
        raw = _read_bound_file(
            path, expected=details, maximum_bytes=MAX_POLICY_BYTES
        )
        parsed = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise CabError("candidate access-boundary policy is invalid") from None
    expected = _canonical_json(EXPECTED_POLICY)
    if raw != expected or parsed != EXPECTED_POLICY:
        raise CabError("candidate access-boundary policy differs from the pinned policy")
    return parsed, raw


def _validate_source_credential_target(path: Path) -> os.stat_result:
    if not path.is_absolute():
        raise CabError("source credential path must be absolute")
    if not path.name.startswith("gha-creds") or path.suffix != ".json":
        raise CabError("source credential is not a GitHub Actions credential file")
    return _validate_owned_regular_file(
        path, maximum_bytes=MAX_CREDENTIAL_BYTES, require_mode_0600=True
    )


def _validate_source_credential_bytes(raw: bytes) -> None:
    try:
        credential = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise CabError("source credential JSON is invalid") from None
    if not isinstance(credential, dict) or credential.get("type") != "service_account":
        raise CabError("source credential must be a service account credential")
    for field in ("client_email", "private_key", "private_key_id", "token_uri"):
        if not isinstance(credential.get(field), str) or not credential[field]:
            raise CabError("source credential is missing a required field")
    if credential["token_uri"] != "https://oauth2.googleapis.com/token":
        raise CabError("source credential token endpoint is not pinned")


def _remove_required(path: Path, *, expected: os.stat_result | None = None) -> None:
    if expected is not None:
        try:
            current = path.lstat()
        except FileNotFoundError:
            raise CabError(
                "source credential disappeared before controlled removal"
            ) from None
        except OSError:
            raise CabError("source credential could not be inspected") from None
        if _stat_identity(current) != _stat_identity(expected):
            raise CabError("source credential changed before controlled removal")
    try:
        path.unlink()
    except FileNotFoundError:
        raise CabError("source credential disappeared before controlled removal") from None
    except OSError:
        raise CabError("source credential could not be removed") from None
    if path.exists() or path.is_symlink():
        raise CabError("source credential removal could not be verified")


def _validate_token(value: str, *, label: str) -> str:
    if not isinstance(value, str):
        raise CabError(f"{label} token is invalid")
    encoded = value.encode("ascii", errors="strict")
    if not (16 <= len(encoded) <= MAX_TOKEN_BYTES):
        raise CabError(f"{label} token is invalid")
    if any(character.isspace() or ord(character) < 0x21 for character in value):
        raise CabError(f"{label} token is invalid")
    return value


def _child_environment(credential_path: Path, cloud_sdk_config: Path) -> dict[str, str]:
    environment = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(cloud_sdk_config),
        "LC_ALL": "C",
        "CLOUDSDK_CORE_DISABLE_PROMPTS": "1",
        "CLOUDSDK_CONFIG": str(cloud_sdk_config),
        "GOOGLE_APPLICATION_CREDENTIALS": str(credential_path),
        "CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE": str(credential_path),
    }
    return environment


def _write_exclusive_bytes(path: Path, payload: bytes) -> os.stat_result:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(path, flags, 0o600)
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or stat.S_IMODE(details.st_mode) != 0o600:
            raise CabError("private credential copy could not be protected")
        return details
    except FileExistsError:
        raise CabError("private credential copy already exists") from None
    except CabError:
        raise
    except OSError:
        raise CabError("private credential copy could not be written") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _write_exclusive_secret(path: Path, token: str) -> None:
    if not path.is_absolute():
        raise CabError("downscoped token path must be absolute")
    try:
        parent = path.parent.lstat()
    except OSError:
        raise CabError("downscoped token directory is unavailable") from None
    if not stat.S_ISDIR(parent.st_mode) or stat.S_ISLNK(parent.st_mode):
        raise CabError("downscoped token parent must be a non-symlink directory")
    if hasattr(os, "getuid") and parent.st_uid != os.getuid():
        raise CabError("downscoped token directory has the wrong owner")
    if stat.S_IMODE(parent.st_mode) & 0o022:
        raise CabError("downscoped token directory is writable by another user")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(token.encode("ascii") + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        if stat.S_IMODE(path.lstat().st_mode) != 0o600:
            raise CabError("downscoped token file mode is not 0600")
    except FileExistsError:
        raise CabError("downscoped token output already exists") from None
    except CabError:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    except OSError:
        try:
            path.unlink()
        except OSError:
            pass
        raise CabError("downscoped token could not be written") from None


def _direct_sts_exchange(
    request_body: bytes, *, timeout: float = REQUEST_TIMEOUT_SECONDS
) -> tuple[int, bytes]:
    """POST only to the pinned STS origin; no redirect or ambient proxy support."""

    connection = http.client.HTTPSConnection(
        STS_HOST,
        443,
        timeout=timeout,
        context=ssl.create_default_context(),
    )
    try:
        connection.request(
            "POST",
            STS_PATH,
            body=request_body,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "identity",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "nuvion-iq9075-candidate-cab/1",
            },
        )
        response = connection.getresponse()
        raw = response.read(MAX_STS_RESPONSE_BYTES + 1)
        return response.status, raw
    except (OSError, http.client.HTTPException, ssl.SSLError):
        raise CabError("STS exchange failed") from None
    finally:
        connection.close()


def mint(
    *,
    credential_path: Path,
    policy_path: Path,
    output_path: Path,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    sts_exchange: Callable[..., tuple[int, bytes]] = _direct_sts_exchange,
) -> dict[str, object]:
    """Exchange a source ADC token for one CAB token and destroy source ADC."""

    credential_path = Path(credential_path)
    policy_path = Path(policy_path)
    output_path = Path(output_path)
    credential_details: os.stat_result | None = None
    credential_removed = False
    credential_bytes = bytearray()
    source_token = ""
    downscoped_token = ""
    try:
        try:
            credential_details = _validate_source_credential_target(credential_path)
            credential_bytes.extend(
                _read_bound_file(
                    credential_path,
                    expected=credential_details,
                    maximum_bytes=MAX_CREDENTIAL_BYTES,
                )
            )
            _validate_source_credential_bytes(bytes(credential_bytes))
            _remove_required(credential_path, expected=credential_details)
            credential_removed = True

            policy, policy_raw = _read_exact_policy(policy_path)
            if output_path.exists() or output_path.is_symlink():
                raise CabError("downscoped token output already exists")

            with tempfile.TemporaryDirectory(
                prefix="nuvion-candidate-gcloud-"
            ) as raw_config:
                cloud_sdk_config = Path(raw_config)
                private_credential = cloud_sdk_config / "source-service-account.json"
                private_details = _write_exclusive_bytes(
                    private_credential, bytes(credential_bytes)
                )
                try:
                    try:
                        completed = command_runner(
                            [
                                "gcloud",
                                "--quiet",
                                "auth",
                                "application-default",
                                "print-access-token",
                            ],
                            check=False,
                            text=True,
                            timeout=REQUEST_TIMEOUT_SECONDS,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL,
                            env=_child_environment(
                                private_credential, cloud_sdk_config
                            ),
                        )
                    except (OSError, subprocess.SubprocessError):
                        raise CabError("source access token mint failed") from None
                finally:
                    _remove_required(private_credential, expected=private_details)
                if completed.returncode != 0:
                    raise CabError("source access token mint failed")
                try:
                    source_token = _validate_token(
                        completed.stdout.strip(), label="source"
                    )
                except (AttributeError, UnicodeEncodeError):
                    raise CabError("source token is invalid") from None
        finally:
            for index in range(len(credential_bytes)):
                credential_bytes[index] = 0
            credential_bytes.clear()
            if credential_details is not None and not credential_removed:
                _remove_required(credential_path, expected=credential_details)
                credential_removed = True

        request_body = urllib.parse.urlencode(
            {
                "grant_type": TOKEN_EXCHANGE_GRANT,
                "requested_token_type": ACCESS_TOKEN_TYPE,
                "subject_token_type": ACCESS_TOKEN_TYPE,
                "subject_token": source_token,
                "options": json.dumps(policy, sort_keys=True, separators=(",", ":")),
            }
        ).encode("ascii")
        try:
            status_code, raw_response = sts_exchange(
                request_body, timeout=REQUEST_TIMEOUT_SECONDS
            )
        except CabError:
            raise
        except (OSError, http.client.HTTPException, ssl.SSLError):
            raise CabError("STS exchange failed") from None
        finally:
            source_token = ""
        if (
            status_code != 200
            or not isinstance(raw_response, bytes)
            or len(raw_response) > MAX_STS_RESPONSE_BYTES
        ):
            raise CabError("STS exchange failed")
        try:
            payload = json.loads(raw_response)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise CabError("STS exchange returned an invalid response") from None
        if not isinstance(payload, dict):
            raise CabError("STS exchange returned an invalid response")
        if payload.get("issued_token_type") != ACCESS_TOKEN_TYPE:
            raise CabError("STS exchange returned an unexpected token type")
        if payload.get("token_type") != "Bearer":
            raise CabError("STS exchange returned an unexpected bearer type")
        expires_in = payload.get("expires_in")
        if (
            not isinstance(expires_in, int)
            or isinstance(expires_in, bool)
            or not 1 <= expires_in <= MAX_TOKEN_LIFETIME_SECONDS
        ):
            raise CabError("STS exchange returned an invalid lifetime")
        try:
            downscoped_token = _validate_token(
                payload.get("access_token"), label="downscoped"
            )
        except (UnicodeEncodeError, AttributeError):
            raise CabError("downscoped token is invalid") from None
        _write_exclusive_secret(output_path, downscoped_token)
        downscoped_token = ""
        return {
            "schemaVersion": 1,
            "kind": "nuvion-iq9075-candidate-cab-token-mint",
            "expiresIn": expires_in,
            "policySha256": hashlib.sha256(policy_raw).hexdigest(),
            "credentialRemoved": True,
        }
    finally:
        source_token = ""
        downscoped_token = ""


def _install_cleanup_signal_handlers() -> None:
    def interrupt(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, interrupt)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credential", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-token", type=Path, required=True)
    arguments = parser.parse_args(argv)
    _install_cleanup_signal_handlers()
    try:
        result = mint(
            credential_path=arguments.credential,
            policy_path=arguments.policy,
            output_path=arguments.output_token,
        )
    except (CabError, KeyboardInterrupt):
        print("candidate CAB token mint failed closed", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
