#!/usr/bin/env python3
"""Verify a GitHub-hosted standalone publisher OIDC token without network installs."""

from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


SHA = re.compile(r"^[0-9a-f]{40}$")
RUN_ID = re.compile(r"^[1-9][0-9]*$")
TAG = "candidate-publisher-v1"
TAG_REF = f"refs/tags/{TAG}"
WORKFLOW_REF = (
    "plaid-ai/NUV-AGENT/.github/workflows/"
    f"iq9075-candidate-trusted-publish.yml@{TAG_REF}"
)
ISSUER = "https://token.actions.githubusercontent.com"
JWKS_URI = f"{ISSUER}/.well-known/jwks"
MAX_INPUT_BYTES = 1024 * 1024


class OidcVerificationError(RuntimeError):
    pass


def _strict_json(path: Path, *, label: str) -> Any:
    raw = path.read_bytes()
    if not raw or len(raw) > MAX_INPUT_BYTES:
        raise OidcVerificationError(f"{label} size is invalid")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise OidcVerificationError(f"duplicate {label} member: {key}")
            result[key] = value
        return result

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=no_duplicates)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise OidcVerificationError(f"invalid {label}") from exc


def _base64url(value: str, *, label: str) -> bytes:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise OidcVerificationError(f"invalid {label}")
    try:
        return base64.b64decode(
            value + "=" * (-len(value) % 4), altchars=b"-_", validate=True
        )
    except ValueError as exc:
        raise OidcVerificationError(f"invalid {label}") from exc


def _der_length(length: int) -> bytes:
    if length < 0:
        raise OidcVerificationError("negative DER length")
    if length < 128:
        return bytes([length])
    encoded = length.to_bytes((length.bit_length() + 7) // 8, "big")
    return bytes([0x80 | len(encoded)]) + encoded


def _der(tag: int, payload: bytes) -> bytes:
    return bytes([tag]) + _der_length(len(payload)) + payload


def _der_integer(raw: bytes) -> bytes:
    normalized = raw.lstrip(b"\0") or b"\0"
    if normalized[0] & 0x80:
        normalized = b"\0" + normalized
    return _der(0x02, normalized)


def _rsa_spki(modulus: bytes, exponent: bytes) -> bytes:
    if len(modulus) < 256 or int.from_bytes(exponent, "big") != 65537:
        raise OidcVerificationError("GitHub OIDC RSA key parameters are invalid")
    rsa_key = _der(0x30, _der_integer(modulus) + _der_integer(exponent))
    rsa_oid = bytes.fromhex("06092a864886f70d010101")
    algorithm = _der(0x30, rsa_oid + bytes.fromhex("0500"))
    return _der(0x30, algorithm + _der(0x03, b"\0" + rsa_key))


def _pem(spki: bytes) -> bytes:
    encoded = base64.b64encode(spki)
    lines = [encoded[index : index + 64] for index in range(0, len(encoded), 64)]
    return b"-----BEGIN PUBLIC KEY-----\n" + b"\n".join(lines) + b"\n-----END PUBLIC KEY-----\n"


def verify(
    *,
    token_response_path: Path,
    configuration_path: Path,
    jwks_path: Path,
    publisher_sha: str,
    environment: str,
    audience: str,
    run_id: str,
    run_attempt: str,
) -> dict[str, str]:
    if not SHA.fullmatch(publisher_sha):
        raise OidcVerificationError("publisher SHA is invalid")
    if environment not in {"iq9075-candidate-sign", "iq9075-candidate-stage"}:
        raise OidcVerificationError("publisher environment is invalid")
    if audience != "nuvion-iq9075-candidate-trusted-publisher":
        raise OidcVerificationError("OIDC audience is invalid")
    if not RUN_ID.fullmatch(run_id) or not RUN_ID.fullmatch(run_attempt):
        raise OidcVerificationError("workflow run identity is invalid")

    configuration = _strict_json(configuration_path, label="OIDC configuration")
    if not isinstance(configuration, dict) or (
        configuration.get("issuer") != ISSUER
        or configuration.get("jwks_uri") != JWKS_URI
    ):
        raise OidcVerificationError("GitHub OIDC discovery identity is invalid")

    response = _strict_json(token_response_path, label="OIDC token response")
    token = response.get("value") if isinstance(response, dict) else None
    if not isinstance(token, str) or len(token) > 64 * 1024:
        raise OidcVerificationError("GitHub OIDC token is invalid")
    pieces = token.split(".")
    if len(pieces) != 3:
        raise OidcVerificationError("GitHub OIDC token is not a JWT")
    encoded_header, encoded_payload, encoded_signature = pieces
    try:
        header = json.loads(_base64url(encoded_header, label="JWT header").decode("utf-8"))
        claims = json.loads(_base64url(encoded_payload, label="JWT claims").decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise OidcVerificationError("GitHub OIDC JWT JSON is invalid") from exc
    if not isinstance(header, dict) or not isinstance(claims, dict):
        raise OidcVerificationError("GitHub OIDC JWT shape is invalid")
    kid = header.get("kid")
    if header.get("alg") != "RS256" or not isinstance(kid, str) or not kid:
        raise OidcVerificationError("GitHub OIDC JWT algorithm or key id is invalid")

    jwks = _strict_json(jwks_path, label="OIDC JWKS")
    keys = jwks.get("keys") if isinstance(jwks, dict) else None
    matches = [
        key
        for key in keys
        if isinstance(key, dict)
        and key.get("kid") == kid
        and key.get("kty") == "RSA"
        and key.get("alg") in (None, "RS256")
        and key.get("use") in (None, "sig")
        and isinstance(key.get("n"), str)
        and isinstance(key.get("e"), str)
    ] if isinstance(keys, list) else []
    if len(matches) != 1:
        raise OidcVerificationError("GitHub OIDC signing key is missing or ambiguous")
    key = matches[0]
    public_key = _pem(
        _rsa_spki(
            _base64url(key["n"], label="RSA modulus"),
            _base64url(key["e"], label="RSA exponent"),
        )
    )
    signature = _base64url(encoded_signature, label="JWT signature")
    openssl = shutil.which("openssl")
    if openssl is None:
        raise OidcVerificationError("openssl is unavailable")
    with tempfile.TemporaryDirectory(prefix="nuvion-oidc-") as raw_directory:
        directory = Path(raw_directory)
        key_path = directory / "public.pem"
        input_path = directory / "signed-input"
        signature_path = directory / "signature"
        key_path.write_bytes(public_key)
        input_path.write_bytes(f"{encoded_header}.{encoded_payload}".encode("ascii"))
        signature_path.write_bytes(signature)
        completed = subprocess.run(
            [openssl, "dgst", "-sha256", "-verify", str(key_path), "-signature", str(signature_path), str(input_path)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    if completed.returncode != 0:
        raise OidcVerificationError("GitHub OIDC JWT signature is invalid")

    expected = {
        "iss": ISSUER,
        "aud": audience,
        "repository": "plaid-ai/NUV-AGENT",
        "repository_id": "1149331364",
        "repository_owner": "plaid-ai",
        "repository_owner_id": "199492120",
        "repository_visibility": "public",
        "sub": f"repo:plaid-ai/NUV-AGENT:environment:{environment}",
        "ref": TAG_REF,
        "ref_type": "tag",
        "event_name": "workflow_dispatch",
        "runner_environment": "github-hosted",
        "sha": publisher_sha,
        "workflow_ref": WORKFLOW_REF,
        "workflow_sha": publisher_sha,
        "run_id": run_id,
        "run_attempt": run_attempt,
        "environment": environment,
    }
    for name, expected_value in expected.items():
        if claims.get(name) != expected_value:
            raise OidcVerificationError(f"GitHub OIDC claim {name} is not protected")
    if "job_workflow_ref" in claims or "job_workflow_sha" in claims:
        raise OidcVerificationError("reusable-workflow identity appeared in standalone token")
    now = int(time.time())
    for name in ("nbf", "iat", "exp"):
        if isinstance(claims.get(name), bool) or not isinstance(claims.get(name), int):
            raise OidcVerificationError(f"GitHub OIDC time claim {name} is invalid")
    if claims["nbf"] > now + 30 or claims["iat"] > now + 30 or claims["exp"] <= now:
        raise OidcVerificationError("GitHub OIDC token time window is invalid")
    if not 0 < claims["exp"] - claims["iat"] <= 600:
        raise OidcVerificationError("GitHub OIDC token lifetime is invalid")
    return {"publisherSha": publisher_sha, "workflowRef": WORKFLOW_REF}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-response", type=Path, required=True)
    parser.add_argument("--openid-configuration", type=Path, required=True)
    parser.add_argument("--jwks", type=Path, required=True)
    parser.add_argument("--publisher-sha", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--audience", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-attempt", required=True)
    arguments = parser.parse_args()
    try:
        result = verify(
            token_response_path=arguments.token_response.resolve(),
            configuration_path=arguments.openid_configuration.resolve(),
            jwks_path=arguments.jwks.resolve(),
            publisher_sha=arguments.publisher_sha,
            environment=arguments.environment,
            audience=arguments.audience,
            run_id=arguments.run_id,
            run_attempt=arguments.run_attempt,
        )
    except (OSError, OidcVerificationError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
