"""Cloud KMS Ed25519 adapter; private key material never leaves KMS.

This module is release tooling only. Device verification has no Google dependency.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from datetime import datetime, timedelta, timezone

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

VERSION = re.compile(
    r"projects/[a-z0-9-]+/locations/[a-z0-9-]+/keyRings/[A-Za-z0-9_-]+/"
    r"cryptoKeys/[A-Za-z0-9_-]+/cryptoKeyVersions/[1-9][0-9]*"
)


def kms_client():
    from google.cloud import kms_v1

    account = os.environ.get("NUVION_KMS_GCLOUD_ACCOUNT")
    if not account:
        return kms_v1.KeyManagementServiceClient(transport="rest")
    if os.environ.get("GITHUB_ACTIONS") == "true":
        raise ValueError("CI must use Workload Identity Federation, not a gcloud user")
    from google.oauth2.credentials import Credentials

    def refresh(request, scopes):
        result = subprocess.run(
            ["gcloud", "auth", "print-access-token", f"--account={account}", "--quiet"],
            capture_output=True, text=True, timeout=60, check=False,
        )
        if result.returncode:
            raise ValueError("gcloud authentication failed; run gcloud auth login for the selected account")
        token = result.stdout.strip()
        if not token or any(c.isspace() for c in token):
            raise ValueError("gcloud returned an invalid access token")
        return token, datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=30)

    credentials = Credentials(None, refresh_handler=refresh, scopes=["https://www.googleapis.com/auth/cloud-platform"])
    return kms_v1.KeyManagementServiceClient(credentials=credentials, transport="rest")


class CloudKmsEd25519Key(Ed25519PrivateKey):
    """Use an exact version and a separately trusted SHA-256 of public SPKI DER."""

    def __init__(self, version: str, public_key_sha256: str, *, client=None):
        if not VERSION.fullmatch(version):
            raise ValueError("KMS signing requires an exact cryptoKeyVersions/N resource")
        if not re.fullmatch(r"[0-9a-f]{64}", public_key_sha256):
            raise ValueError("KMS public key SHA-256 must be 64 lowercase hex characters")
        # Lazy imports keep the offline/device and existing local signing paths intact.
        from google.cloud import kms_v1
        import google_crc32c

        self._crc32c = google_crc32c.value
        self._version = version
        self._client = client or kms_client()
        response = self._client.get_public_key(request={"name": version}, timeout=30, retry=None)
        if response.name != version:
            raise ValueError("KMS returned a different public key version")
        if response.algorithm != kms_v1.CryptoKeyVersion.CryptoKeyVersionAlgorithm.EC_SIGN_ED25519:
            raise ValueError("KMS key algorithm must be EC_SIGN_ED25519")
        pem = response.pem.encode("ascii")
        if response.pem_crc32c != self._crc32c(pem):
            raise ValueError("KMS public key CRC32C mismatch")
        key = serialization.load_pem_public_key(pem)
        if not isinstance(key, Ed25519PublicKey):
            raise ValueError("KMS public key must be Ed25519")
        der = key.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
        if hashlib.sha256(der).hexdigest() != public_key_sha256:
            raise ValueError("KMS public key does not match the trusted SHA-256")
        self._public_key = key

    def public_key(self):
        return self._public_key

    def sign(self, data: bytes) -> bytes:
        if not isinstance(data, bytes) or len(data) > 65536:
            raise ValueError("KMS signing input must be bytes, at most 64 KiB")
        # PureEdDSA takes raw data. Do not replace data with a KMS digest field.
        response = self._client.asymmetric_sign(
            request={"name": self._version, "data": data, "data_crc32c": self._crc32c(data)},
            timeout=30,
            retry=None,
        )
        if response.name != self._version:
            raise ValueError("KMS signed with a different key version")
        if not response.verified_data_crc32c:
            raise ValueError("KMS did not verify the signing input CRC32C")
        signature = bytes(response.signature)
        if len(signature) != 64 or response.signature_crc32c != self._crc32c(signature):
            raise ValueError("KMS signature length or CRC32C mismatch")
        try:
            self._public_key.verify(signature, data)
        except InvalidSignature as exc:
            raise ValueError("KMS signature failed local public-key verification") from exc
        return signature

    def private_bytes(self, *args, **kwargs):
        raise TypeError("Cloud KMS private key material is not exportable")

    def private_bytes_raw(self):
        raise TypeError("Cloud KMS private key material is not exportable")

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self
