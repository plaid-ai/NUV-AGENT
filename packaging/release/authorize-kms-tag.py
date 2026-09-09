#!/usr/bin/env python3
"""Authorize a protected-main approval tag, optionally sign and push it once."""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def run(command, *, env=None):
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise ValueError(f"Command {command[0]} {command[1]} failed")
    return result.stdout.strip()


def network_environment(env):
    authorization = base64.b64encode(f"x-access-token:{env['GH_TOKEN']}".encode()).decode()
    return {**env, "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
            "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {authorization}"}


def authorize():
    policy = json.loads((ROOT / "packaging/release/release-security-policy.json").read_text())
    env = os.environ
    if env.get("GITHUB_REPOSITORY") != "plaid-ai/NUV-AGENT" or env.get("GITHUB_REF") != "refs/heads/main":
        raise ValueError("Company signing requires the exact protected-main workflow")
    if env.get("GITHUB_EVENT_NAME") != "workflow_dispatch" or env.get("GITHUB_RUN_ATTEMPT") != "1":
        raise ValueError("Only a new explicit workflow dispatch can approve a release")
    if int(env.get("ACTOR_ID", "0")) not in [u["id"] for u in policy["releaseAdminUsers"]]:
        raise ValueError("Only a release administrator can request an approval tag")
    sha, tag, message = env["TARGET_SHA"], env["TAG_NAME"], env["TAG_MESSAGE"]
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("An exact commit SHA is required")
    if not re.fullmatch(r"(?:v[0-9]+\.[0-9]+\.[0-9]+|candidate-publisher-v[1-9][0-9]*)", tag):
        raise ValueError("Invalid release approval tag name")
    if not message.strip() or len(message.encode()) > 8192 or "\x00" in message or "-----BEGIN PGP SIGNATURE-----" in message:
        raise ValueError("Invalid approval message")
    if tag.startswith("candidate-"):
        if tag != policy["candidatePublisher"]["tag"]:
            raise ValueError("Candidate publisher tag must match the reviewed policy")
    elif tag[1:] != tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]:
        raise ValueError("Release tag must match the reviewed Agent version")
    main = json.loads(run(["gh", "api", "repos/plaid-ai/NUV-AGENT/git/ref/heads/main"]))["object"]["sha"]
    if sha != main or sha != run(["git", "rev-parse", "HEAD"]):
        raise ValueError("Target and workflow must both be the current protected-main commit")
    # Resolve only an already-validated ref; no untrusted shell interpolation.
    if run(["git", "ls-remote", "--tags", "origin", f"refs/tags/{tag}"], env=network_environment(env)):
        raise ValueError("Approval tag already exists; tags are immutable")
    return sha, tag, message


def sign_and_push():
    sha, tag, message = authorize()
    config_path = ROOT / "packaging/release/cloud-kms/prod/approval.json"
    config = json.loads(config_path.read_text())
    policy = json.loads((ROOT / "packaging/release/release-security-policy.json").read_text())
    if config["fingerprint"] not in policy["trustedTagSignerFingerprints"]:
        raise ValueError("KMS production approval key has not been admitted to release policy")
    with tempfile.TemporaryDirectory(prefix="kms-tag-") as tmp:
        home = Path(tmp) / "gnupg"
        home.mkdir(mode=0o700)
        env = {**os.environ, "NUVION_GPG_KMS_CONFIG": str(config_path), "GNUPGHOME": str(home),
               "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull,
               "GIT_COMMITTER_NAME": "Plaid NUV Release Approval", "GIT_COMMITTER_EMAIL": "platform-admins@plaid.ai.kr"}
        message_path = Path(tmp) / "message.txt"
        message_path.write_text(message + "\n")
        run(["git", "-c", "gpg.format=openpgp", "-c", f"gpg.program={ROOT}/packaging/release/kms-gpg.py",
             "tag", "-s", "-u", config["fingerprint"], "-F", str(message_path), tag, sha], env=env)
        run(["gpg", "--batch", "--import", str(config_path.parent / "approval.asc")], env=env)
        run(["git", "verify-tag", "--raw", tag], env=env)
        authorize()  # Fail if protected main advanced while KMS was signing.
        push_env = network_environment(env)
        run(["git", "push", "origin", f"refs/tags/{tag}:refs/tags/{tag}"], env=push_env)
        print(f"Created and verified immutable KMS approval tag {tag} at {sha}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sign-and-push", action="store_true")
    args = parser.parse_args()
    sign_and_push() if args.sign_and_push else authorize()
