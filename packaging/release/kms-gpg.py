#!/usr/bin/env python3
"""Sign release evidence and Git tags through Cloud KMS, without a GPG password."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys

from cloud_kms_signer import CloudKmsEd25519Key
from kms_openpgp import create_public_certificate, detached_signature, load_signing_certificate


def load_config(path: Path):
    config = json.loads(path.read_text())
    if set(config) != {"keyVersion", "publicKeySha256", "publicCertificate", "fingerprint"}:
        raise ValueError("KMS GPG configuration has unexpected or missing fields")
    if not re.fullmatch(r"[0-9A-F]{40}", config["fingerprint"]):
        raise ValueError("Expected a full OpenPGP fingerprint")
    key = CloudKmsEd25519Key(config["keyVersion"], config["publicKeySha256"])
    certificate = path.parent / config["publicCertificate"]
    return load_signing_certificate(key, certificate.read_text(), fingerprint=config["fingerprint"])


def write_new(path: str, text: str):
    # Evidence and public certificates are immutable; never replace an old output.
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Git invokes gpg.program with --status-fd=2 -bsau <full fingerprint>.
    if argv[:2] == ["--status-fd=2", "-bsau"]:
        if len(argv) != 3 or not os.environ.get("NUVION_GPG_KMS_CONFIG"):
            raise ValueError("Git signing requires NUVION_GPG_KMS_CONFIG and a full fingerprint")
        key = load_config(Path(os.environ["NUVION_GPG_KMS_CONFIG"]).resolve())
        if argv[2] != str(key.fingerprint):
            raise ValueError("Git requested a different signing fingerprint")
        data = sys.stdin.buffer.read(16 * 1024 * 1024 + 1)
        if len(data) > 16 * 1024 * 1024:
            raise ValueError("Git signing input exceeds 16 MiB")
        signature = detached_signature(key, data)
        sys.stdout.write(signature)
        print(f"[GNUPG:] SIG_CREATED D 22 8 00 {int(datetime.now().timestamp())} {key.fingerprint}", file=sys.stderr)
        return 0

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export-public")
    export.add_argument("--key-version", required=True)
    export.add_argument("--public-key-sha256", required=True)
    export.add_argument("--name", required=True)
    export.add_argument("--email", required=True)
    export.add_argument("--created", required=True)
    export.add_argument("--output", required=True)
    sign = commands.add_parser("sign")
    sign.add_argument("--config", type=Path, required=True)
    sign.add_argument("--input", type=Path, required=True)
    sign.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "export-public":
        key = CloudKmsEd25519Key(args.key_version, args.public_key_sha256)
        result = create_public_certificate(
            key, name=args.name, email=args.email,
            created=datetime.fromisoformat(args.created.replace("Z", "+00:00")),
        )
    else:
        key = load_config(args.config.resolve())
        result = detached_signature(key, args.input.read_bytes())
    write_new(args.output, result)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError) as exc:
        print(f"kms-gpg: {exc}", file=sys.stderr)
        raise SystemExit(1)
