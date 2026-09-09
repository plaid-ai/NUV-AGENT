#!/usr/bin/env python3
"""Real KMS signing proof. Creates only offline smoke artifacts, never a release."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cloud_kms_signer import CloudKmsEd25519Key
from kms_openpgp import detached_signature, load_signing_certificate
from nuvion_app.runtime.release_bom import (
    ReleaseKeyring, ReleaseTarget, build_release_bom_v2_payload, build_release_bom_signature,
    canonical_release_bom_json, canonical_release_bom_signature_json, verify_signed_release_bom,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("dev", "prod"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    config_root = ROOT / "packaging/release/cloud-kms" / args.stage
    approval = json.loads((config_root / "approval.json").read_text())
    ota = json.loads((config_root / "ota.json").read_text())
    approval_key = CloudKmsEd25519Key(approval["keyVersion"], approval["publicKeySha256"])
    ota_key = CloudKmsEd25519Key(ota["keyVersion"], ota["publicKeySha256"])
    pgp = load_signing_certificate(approval_key, (config_root / "approval.asc").read_text(),
                                   fingerprint=approval["fingerprint"])
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    evidence = json.dumps({"purpose": "KMS migration smoke test; not OTA acceptance",
                           "stage": args.stage, "time": now, "runId": os.environ.get("GITHUB_RUN_ID")}, sort_keys=True).encode()
    (root / "evidence.json").write_bytes(evidence)
    (root / "evidence.json.asc").write_text(detached_signature(pgp, evidence))
    artifact = root / "offline-smoke-artifact.txt"
    artifact.write_text("Cloud KMS signing compatibility test only\n")
    payload = build_release_bom_v2_payload(
        bom_id="cloud-kms-smoke-only", release_sequence=1, agent_version="0.0.0",
        component_sha="0" * 40, config_schema="12", min_updater_version="0.2.0",
        targets=[ReleaseTarget(product_model="IQ9075_DEV", platform_profile="iq9075_dev",
                               hardware_revision="OFFLINE-KMS-SMOKE", architecture="aarch64")],
        artifact_path=artifact, artifact_kind="agent-bundle", built_at=now,
    )
    signature = build_release_bom_signature(payload, key_id=ota["keyId"], private_key=ota_key)
    ring = ReleaseKeyring({ota["keyId"]: ota_key.public_key().public_bytes_raw()})
    verify_signed_release_bom(payload, signature, release_keyring=ring)
    (root / "release-bom.json").write_text(canonical_release_bom_json(payload))
    (root / "release-bom.json.sig").write_text(canonical_release_bom_signature_json(signature))
    with tempfile.TemporaryDirectory(prefix="kms-verify-") as tmp:
        home = Path(tmp) / "gnupg"
        home.mkdir(mode=0o700)
        env = {**os.environ, "GNUPGHOME": str(home), "GIT_CONFIG_GLOBAL": os.devnull,
               "GIT_CONFIG_SYSTEM": os.devnull, "GIT_AUTHOR_NAME": "KMS smoke",
               "GIT_AUTHOR_EMAIL": "platform-admins@plaid.ai.kr", "GIT_COMMITTER_NAME": "KMS smoke",
               "GIT_COMMITTER_EMAIL": "platform-admins@plaid.ai.kr",
               "NUVION_GPG_KMS_CONFIG": str(config_root / "approval.json"),
               "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]}
        def run(*command, data=None):
            result = subprocess.run(command, input=data, capture_output=True, env=env, timeout=30)
            if result.returncode:
                raise RuntimeError(result.stderr.decode(errors="replace"))
            return result
        run("gpg", "--batch", "--import", str(config_root / "approval.asc"))
        verified = run("gpg", "--batch", "--status-fd=1", "--verify", str(root / "evidence.json.asc"), str(root / "evidence.json"))
        (root / "gpg-verification.txt").write_bytes(verified.stdout + verified.stderr)
        repo = Path(tmp) / "git"
        run("git", "init", "--quiet", str(repo))
        tree = run("git", "-C", str(repo), "mktree", data=b"").stdout.decode().strip()
        commit = run("git", "-C", str(repo), "commit-tree", tree, data=b"KMS smoke\n").stdout.decode().strip()
        run("git", "-C", str(repo), "-c", "gpg.format=openpgp", "-c",
            f"gpg.program={ROOT}/packaging/release/kms-gpg.py", "tag", "-s", "-u",
            approval["fingerprint"], "-m", "Cloud KMS smoke only", "kms-smoke", commit)
        (root / "signed-tag.txt").write_bytes(run("git", "-C", str(repo), "cat-file", "tag", "kms-smoke").stdout)
        verified = run("git", "-C", str(repo), "verify-tag", "--raw", "kms-smoke")
        (root / "git-verification.txt").write_bytes(verified.stdout + verified.stderr)
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.iterdir()) if p.is_file()}
    summary = {"stage": args.stage, "time": now, "status": "PASS", "otaAcceptance": False,
               "approval": approval, "ota": ota, "files": files}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"stage": args.stage, "status": "PASS", "proof": str(root)}))


if __name__ == "__main__":
    main()
