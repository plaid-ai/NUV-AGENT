#!/usr/bin/env python3
"""Verify a short-lived, platform-admin-signed GitHub settings audit."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from publisher_trust import (
    PublisherTrustError,
    publisher_surface,
    verify_additional_executing_workflow,
    verify_executing_workflow,
)


FINGERPRINT = re.compile(r"^[0-9A-F]{40}$")
REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
MAX_DOCUMENT_BYTES = 256 * 1024


class AttestationError(RuntimeError):
    pass


def _strict_object(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise AttestationError(f"{label} must be a regular non-symlink file")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise AttestationError(f"cannot read {label}") from exc
    if not raw or len(raw) > MAX_DOCUMENT_BYTES:
        raise AttestationError(f"{label} size is invalid")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AttestationError(f"duplicate {label} member: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            raw.decode("utf-8"), object_pairs_hook=reject_duplicate
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise AttestationError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise AttestationError(f"{label} root must be an object")
    return payload, raw


def _timestamp(value: Any, *, label: str) -> dt.datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise AttestationError(f"settings attestation {label} is invalid")
    try:
        parsed = dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise AttestationError(f"settings attestation {label} is invalid") from exc
    if parsed.tzinfo != dt.timezone.utc or parsed.microsecond != 0:
        raise AttestationError(f"settings attestation {label} must be whole-second UTC")
    return parsed


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
        raise AttestationError("cannot inspect settings attestation keyring")
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
    attestation_path: Path,
    signature_path: Path,
    signer_directory: Path,
    allowed_fingerprints: set[str],
) -> str:
    if signature_path.is_symlink() or not signature_path.is_file():
        raise AttestationError("settings attestation signature is unavailable")
    try:
        signature_size = signature_path.stat().st_size
    except OSError as exc:
        raise AttestationError("cannot stat settings attestation signature") from exc
    if signature_size < 1 or signature_size > MAX_DOCUMENT_BYTES:
        raise AttestationError("settings attestation signature size is invalid")
    public_keys = sorted(signer_directory.glob("*.asc"))
    if not public_keys:
        raise AttestationError("settings attestation signer directory is empty")
    with tempfile.TemporaryDirectory(prefix="nuv-settings-attestation-") as temporary:
        gpg_home = Path(temporary)
        gpg_home.chmod(0o700)
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
            raise AttestationError("settings attestation signer files differ from policy")
        environment = {**os.environ, "GNUPGHOME": str(gpg_home)}
        verified = subprocess.run(
            [
                "gpg",
                "--batch",
                "--status-fd=1",
                "--homedir",
                str(gpg_home),
                "--verify",
                str(signature_path),
                str(attestation_path),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
            env=environment,
        )
    if verified.returncode != 0:
        raise AttestationError("settings attestation signature verification failed")
    observed: set[str] = set()
    for line in verified.stdout.splitlines():
        if not line.startswith("[GNUPG:] VALIDSIG "):
            continue
        # VALIDSIG starts with the signing-key fingerprint and, for a signing
        # subkey, also carries the primary fingerprint near the end. Policy is
        # intentionally anchored to primary keys, so inspect every fingerprint
        # field rather than accidentally rejecting legitimate subkey signatures.
        for token in line.split("VALIDSIG", 1)[1].split():
            normalized = token.upper()
            if FINGERPRINT.fullmatch(normalized):
                observed.add(normalized)
    accepted = observed & allowed_fingerprints
    if len(accepted) != 1:
        raise AttestationError("settings attestation signer is not allowlisted")
    return next(iter(accepted))


def verify_attestation(
    *,
    attestation_path: Path,
    signature_path: Path,
    policy_path: Path,
    signer_directory: Path,
    repository: str,
    trusted_publisher_sha: str,
    publisher_root: Path,
    executing_workflow: Path,
    additional_executing_workflows: tuple[tuple[str, Path], ...] = (),
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    if not REPOSITORY.fullmatch(repository):
        raise AttestationError("settings attestation repository is invalid")
    if re.fullmatch(r"[0-9a-f]{40}", trusted_publisher_sha) is None:
        raise AttestationError("trusted publisher SHA is invalid")
    policy, policy_raw = _strict_object(policy_path, label="release security policy")
    fingerprints = policy.get("trustedTagSignerFingerprints")
    if (
        policy.get("schemaVersion") != 1
        or not isinstance(fingerprints, list)
        or not fingerprints
        or len(set(fingerprints)) != len(fingerprints)
        or not all(isinstance(value, str) and FINGERPRINT.fullmatch(value) for value in fingerprints)
    ):
        raise AttestationError("release security policy signer allowlist is invalid")
    attestation, _ = _strict_object(
        attestation_path, label="release settings attestation"
    )
    if set(attestation) != {
        "schemaVersion",
        "kind",
        "repository",
        "trustedPublisherSha",
        "publisherTreeSha256",
        "workflowSha256",
        "policySha256",
        "verifiedAt",
        "expiresAt",
        "settings",
    }:
        raise AttestationError("settings attestation fields do not match schema v1")
    settings = attestation.get("settings")
    if (
        attestation.get("schemaVersion") != 1
        or attestation.get("kind") != "nuvion-release-settings-attestation"
        or attestation.get("repository") != repository
        or attestation.get("trustedPublisherSha") != trusted_publisher_sha
        or not isinstance(attestation.get("publisherTreeSha256"), str)
        or not SHA256.fullmatch(attestation["publisherTreeSha256"])
        or not isinstance(attestation.get("workflowSha256"), str)
        or not SHA256.fullmatch(attestation["workflowSha256"])
        or not isinstance(attestation.get("policySha256"), str)
        or not SHA256.fullmatch(attestation["policySha256"])
        or attestation["policySha256"] != hashlib.sha256(policy_raw).hexdigest()
        or settings
        != {
            "defaultBranch": policy.get("defaultBranch"),
            "governance": policy.get("governance"),
            "secretScopesChecked": True,
            "status": "VERIFIED",
        }
    ):
        raise AttestationError("settings attestation identity does not match policy")
    verified_at = _timestamp(attestation.get("verifiedAt"), label="verifiedAt")
    expires_at = _timestamp(attestation.get("expiresAt"), label="expiresAt")
    current = now or dt.datetime.now(dt.timezone.utc)
    if current.tzinfo != dt.timezone.utc:
        raise AttestationError("settings attestation verifier clock must be UTC")
    if verified_at > current + dt.timedelta(minutes=5):
        raise AttestationError("settings attestation is from the future")
    if expires_at <= current:
        raise AttestationError("settings attestation has expired")
    if expires_at <= verified_at or expires_at - verified_at > dt.timedelta(hours=24):
        raise AttestationError("settings attestation validity window is invalid")
    signer = _verify_signature(
        attestation_path=attestation_path,
        signature_path=signature_path,
        signer_directory=signer_directory,
        allowed_fingerprints=set(fingerprints),
    )
    surface = publisher_surface(
        publisher_root,
        expected_sha=trusted_publisher_sha,
    )
    if surface["publisherTreeSha256"] != attestation["publisherTreeSha256"]:
        raise AttestationError("trusted publisher tree differs from signed attestation")
    verify_executing_workflow(
        publisher_root,
        executing_workflow,
        expected_workflow_sha256=attestation["workflowSha256"],
    )
    if surface["workflowSha256"] != attestation["workflowSha256"]:
        raise AttestationError("trusted publisher workflow digest differs from attestation")
    for publisher_relative_path, additional_workflow in additional_executing_workflows:
        verify_additional_executing_workflow(
            publisher_root,
            additional_workflow,
            publisher_relative_path=publisher_relative_path,
        )
    if additional_executing_workflows:
        final_surface = publisher_surface(
            publisher_root,
            expected_sha=trusted_publisher_sha,
        )
        if final_surface != surface:
            raise AttestationError(
                "trusted publisher changed during additional workflow verification"
            )
    return {
        "schemaVersion": 1,
        "repository": repository,
        "trustedPublisherSha": trusted_publisher_sha,
        "publisherTreeSha256": attestation["publisherTreeSha256"],
        "workflowSha256": attestation["workflowSha256"],
        "policySha256": attestation["policySha256"],
        "signerFingerprint": signer,
        "expiresAt": attestation["expiresAt"],
        "status": "VERIFIED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--signature", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--signer-directory", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--trusted-publisher-sha", required=True)
    parser.add_argument("--publisher-root", type=Path, required=True)
    parser.add_argument("--executing-workflow", type=Path, required=True)
    parser.add_argument(
        "--trusted-additional-workflow",
        action="append",
        nargs=2,
        default=[],
        metavar=("PUBLISHER_RELATIVE_PATH", "EXECUTING_PATH"),
    )
    arguments = parser.parse_args()
    try:
        result = verify_attestation(
            attestation_path=arguments.attestation,
            signature_path=arguments.signature,
            policy_path=arguments.policy,
            signer_directory=arguments.signer_directory,
            repository=arguments.repository,
            trusted_publisher_sha=arguments.trusted_publisher_sha,
            publisher_root=arguments.publisher_root,
            executing_workflow=arguments.executing_workflow,
            additional_executing_workflows=tuple(
                (relative, Path(path))
                for relative, path in arguments.trusted_additional_workflow
            ),
        )
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    except (
        AttestationError,
        OSError,
        PublisherTrustError,
        subprocess.SubprocessError,
    ) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
