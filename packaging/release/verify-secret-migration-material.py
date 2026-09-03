#!/usr/bin/env python3
"""Validate release-secret material without printing or persisting it.

Sensitive values are read only from named environment variables.  Every
subprocess output that could contain credential material is kept out of logs.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
from pathlib import Path
import subprocess
import tempfile


class MaterialError(RuntimeError):
    """Secret material does not match the committed release trust policy."""


_SAFE_SUBPROCESS_ENVIRONMENT = (
    "CLOUDSDK_PYTHON",
    "CLOUDSDK_PYTHON_ARGS",
    "CLOUDSDK_PYTHON_SITEPACKAGES",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "PATH",
    "TMPDIR",
)


def _subprocess_environment(**overrides: str) -> dict[str, str]:
    """Build a capability-minimal child environment without source secrets."""

    environment = {
        name: os.environ[name]
        for name in _SAFE_SUBPROCESS_ENVIRONMENT
        if name in os.environ
    }
    environment.update(overrides)
    return environment


def _secret(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise MaterialError("required secret material is unavailable")
    return value


def _load_policy(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise MaterialError("release security policy is unreadable") from error
    if not isinstance(value, dict):
        raise MaterialError("release security policy is invalid")
    return value


def _gcp_identity(value: str, expected_project: str) -> None:
    try:
        document = json.loads(value)
    except ValueError as error:
        raise MaterialError("GCP credential JSON is invalid") from error
    if not isinstance(document, dict):
        raise MaterialError("GCP credential JSON is invalid")
    email = document.get("client_email")
    key_id = document.get("private_key_id")
    private_key = document.get("private_key")
    if (
        document.get("type") != "service_account"
        or document.get("project_id") != expected_project
        or not isinstance(email, str)
        or not email.endswith(".iam.gserviceaccount.com")
        or not isinstance(key_id, str)
        or len(key_id) < 8
        or not isinstance(private_key, str)
        or "BEGIN PRIVATE KEY" not in private_key
    ):
        raise MaterialError("GCP credential identity is invalid")


def verify_gcp_auth(key_environment: str, project_environment: str) -> None:
    credential = _secret(key_environment)
    project = _secret(project_environment)
    _gcp_identity(credential, project)
    with tempfile.TemporaryDirectory(prefix="nuv-secret-migration-gcp-") as directory:
        root = Path(directory)
        credential_path = root / "credential.json"
        credential_path.write_text(credential, encoding="utf-8")
        credential_path.chmod(0o600)
        environment = _subprocess_environment(
            CLOUDSDK_CONFIG=str(root / "gcloud"),
            CLOUDSDK_CORE_DISABLE_PROMPTS="1",
        )
        activated = subprocess.run(
            [
                "gcloud",
                "--quiet",
                "auth",
                "activate-service-account",
                f"--key-file={credential_path}",
                f"--project={project}",
            ],
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if activated.returncode != 0:
            raise MaterialError("GCP service-account authentication failed")
        token = subprocess.run(
            ["gcloud", "--quiet", "auth", "print-access-token"],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
        )
    if token.returncode != 0 or not token.stdout.strip():
        raise MaterialError("GCP service-account token exchange failed")


def verify_apt_gpg(
    private_key_environment: str,
    passphrase_environment: str,
    policy_path: Path,
) -> None:
    policy = _load_policy(policy_path)
    apt = policy.get("apt")
    fingerprint = apt.get("gpgFingerprint") if isinstance(apt, dict) else None
    if not isinstance(fingerprint, str) or len(fingerprint) != 40:
        raise MaterialError("APT signing policy is invalid")

    with tempfile.TemporaryDirectory(prefix="nuv-secret-migration-gpg-") as directory:
        root = Path(directory)
        root.chmod(0o700)
        key_path = root / "private.asc"
        passphrase_path = root / "passphrase"
        signature_path = root / "probe.sig"
        key_path.write_text(_secret(private_key_environment), encoding="utf-8")
        passphrase_path.write_text(_secret(passphrase_environment), encoding="utf-8")
        key_path.chmod(0o600)
        passphrase_path.chmod(0o600)
        environment = _subprocess_environment(GNUPGHOME=str(root))
        imported = subprocess.run(
            ["gpg", "--batch", "--import", str(key_path)],
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        listed = subprocess.run(
            ["gpg", "--batch", "--with-colons", "--list-secret-keys"],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
        )
        fingerprints = {
            fields[9]
            for line in listed.stdout.splitlines()
            if line.startswith("fpr:")
            for fields in [line.split(":")]
            if len(fields) > 9
        }
        if imported.returncode != 0 or listed.returncode != 0 or fingerprint not in fingerprints:
            raise MaterialError("APT private key differs from policy")
        signed = subprocess.run(
            [
                "gpg",
                "--batch",
                "--yes",
                "--pinentry-mode",
                "loopback",
                "--passphrase-file",
                str(passphrase_path),
                "--local-user",
                fingerprint,
                "--output",
                str(signature_path),
                "--detach-sign",
            ],
            env=environment,
            input=b"nuvion-release-secret-migration-probe\n",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if signed.returncode != 0 or not signature_path.is_file():
            raise MaterialError("APT private key/passphrase verification failed")


def _openssl_private_key(material: str) -> tuple[bytes, str]:
    encoded = material.encode("utf-8")
    if b"-----BEGIN" in encoded:
        return encoded, "PEM"
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        decoded = encoded
    if len(decoded) == 32:
        return bytes.fromhex("302e020100300506032b657004220420") + decoded, "DER"
    return decoded, "DER"


def verify_iq_signing_key(private_key_environment: str, policy_path: Path) -> None:
    policy = _load_policy(policy_path)
    iq9075 = policy.get("iq9075")
    if not isinstance(iq9075, dict):
        raise MaterialError("IQ9075 release policy is invalid")
    key_id = iq9075.get("publisherKeyId")
    keyring_name = iq9075.get("publicKeyringFile")
    if not isinstance(key_id, str) or not isinstance(keyring_name, str):
        raise MaterialError("IQ9075 signing policy is invalid")
    try:
        keyring = json.loads((policy_path.parent / keyring_name).read_text(encoding="utf-8"))
        expected_public = base64.b64decode(keyring["keys"][key_id], validate=True)
    except (OSError, ValueError, KeyError, TypeError, binascii.Error) as error:
        raise MaterialError("IQ9075 public keyring is invalid") from error
    if len(expected_public) != 32:
        raise MaterialError("IQ9075 public key is invalid")

    private_material, private_format = _openssl_private_key(
        _secret(private_key_environment)
    )
    with tempfile.TemporaryDirectory(prefix="nuv-secret-migration-iq-") as directory:
        private_path = Path(directory) / "private.key"
        private_path.write_bytes(private_material)
        private_path.chmod(0o600)
        command = ["openssl", "pkey"]
        if private_format == "DER":
            command.extend(["-inform", "DER"])
        command.extend(["-in", str(private_path), "-pubout", "-outform", "DER"])
        public_key = subprocess.run(
            command,
            env=_subprocess_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    if public_key.returncode != 0 or public_key.stdout[-32:] != expected_public:
        raise MaterialError("IQ9075 private key differs from protected keyring")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    gcp = commands.add_parser("gcp-auth")
    gcp.add_argument("--key-env", required=True)
    gcp.add_argument("--project-env", required=True)

    apt = commands.add_parser("apt-gpg")
    apt.add_argument("--private-key-env", required=True)
    apt.add_argument("--passphrase-env", required=True)
    apt.add_argument("--policy", type=Path, required=True)

    iq = commands.add_parser("iq-key")
    iq.add_argument("--private-key-env", required=True)
    iq.add_argument("--policy", type=Path, required=True)

    arguments = parser.parse_args()
    try:
        if arguments.command == "gcp-auth":
            verify_gcp_auth(arguments.key_env, arguments.project_env)
        elif arguments.command == "apt-gpg":
            verify_apt_gpg(
                arguments.private_key_env,
                arguments.passphrase_env,
                arguments.policy.resolve(),
            )
        elif arguments.command == "iq-key":
            verify_iq_signing_key(arguments.private_key_env, arguments.policy.resolve())
        else:  # pragma: no cover
            raise MaterialError("unknown verification command")
    except MaterialError as error:
        print(f"verification failed: {error}", file=os.sys.stderr)
        return 1
    print("secret material verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
