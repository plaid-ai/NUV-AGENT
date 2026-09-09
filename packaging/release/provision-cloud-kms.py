#!/usr/bin/env python3
"""Plan/apply NUV company signing keys and tightly scoped GitHub federation.

Adds resources and per-key bindings only; never replaces project IAM policy,
rotates/destroys keys, or exports service-account/private-key credentials.
"""

from __future__ import annotations

import argparse
import json
import subprocess

PROJECT = "plaid-451114"
NUMBER = "472626352998"
LOCATION = "asia-northeast3"
REPOSITORY = "plaid-ai/NUV-AGENT"
REPOSITORY_ID = "1149331364"
OWNER_ID = "199492120"


def run(args, account, *, read=False):
    command = ["gcloud", *args, f"--project={PROJECT}", f"--account={account}", "--quiet", "--format=json"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if result.returncode:
        # Authentication/permission errors must not be interpreted as absent resources.
        if read and ("NOT_FOUND" in result.stderr or "does not exist" in result.stderr):
            return None
        raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout) if result.stdout.strip() else {}


def plan():
    return {
        "project": PROJECT, "projectNumber": NUMBER, "location": LOCATION,
        "algorithm": "EC_SIGN_ED25519", "protectionLevel": "SOFTWARE",
        "environments": {
            stage: {
                "keyRing": f"nuv-agent-{stage}", "keys": ["approval", "ota"],
                "serviceAccount": f"nuv-agent-sign-{stage}@{PROJECT}.iam.gserviceaccount.com",
                "workloadIdentityPool": f"nuv-agent-sign-{stage}",
                "githubEnvironment": f"nuv-signing-{stage}",
            } for stage in ("dev", "prod")
        },
        "trust": {"repositoryId": REPOSITORY_ID, "ownerId": OWNER_ID, "ref": "refs/heads/main"},
        "privateKeyExport": False,
    }


def provision(account):
    run(["services", "enable", "cloudkms.googleapis.com", "iam.googleapis.com",
         "iamcredentials.googleapis.com", "sts.googleapis.com"], account)
    for stage, config in plan()["environments"].items():
        ring, sa, pool = config["keyRing"], config["serviceAccount"], config["workloadIdentityPool"]
        location = [f"--location={LOCATION}"]
        if run(["kms", "keyrings", "describe", ring, *location], account, read=True) is None:
            run(["kms", "keyrings", "create", ring, *location], account)
        if run(["iam", "service-accounts", "describe", sa], account, read=True) is None:
            run(["iam", "service-accounts", "create", f"nuv-agent-sign-{stage}",
                 f"--display-name=NUV Agent {stage} signing only"], account)
        for purpose in config["keys"]:
            flags = [f"--keyring={ring}", *location]
            existing = run(["kms", "keys", "describe", purpose, *flags], account, read=True)
            if existing is None:
                run(["kms", "keys", "create", purpose, *flags,
                               "--purpose=asymmetric-signing", "--default-algorithm=ec-sign-ed25519",
                               "--protection-level=software"], account)
                existing = run(["kms", "keys", "describe", purpose, *flags], account)
            if existing.get("purpose") != "ASYMMETRIC_SIGN" or existing.get("versionTemplate") != {
                "algorithm": "EC_SIGN_ED25519", "protectionLevel": "SOFTWARE"
            }:
                raise ValueError(f"Existing key {ring}/{purpose} differs from the plan")
            for role in ("roles/cloudkms.signer", "roles/cloudkms.publicKeyViewer"):
                run(["kms", "keys", "add-iam-policy-binding", purpose, *flags,
                     f"--member=serviceAccount:{sa}", f"--role={role}"], account)
        if run(["iam", "workload-identity-pools", "describe", pool, "--location=global"], account, read=True) is None:
            run(["iam", "workload-identity-pools", "create", pool, "--location=global",
                 f"--display-name=NUV signing {stage}"], account)
        environment = config["githubEnvironment"]
        allowed_workflows = [
            f"{REPOSITORY}/.github/workflows/kms-signing-smoke.yml@refs/heads/main",
            f"{REPOSITORY}/.github/workflows/kms-approve-release.yml@refs/heads/main",
        ]
        condition = (
            f"assertion.repository_id == '{REPOSITORY_ID}' && "
            f"assertion.repository_owner_id == '{OWNER_ID}' && "
            "assertion.ref == 'refs/heads/main' && assertion.event_name == 'workflow_dispatch' && "
            f"assertion.sub == 'repo:{REPOSITORY}:environment:{environment}' && "
            f"assertion.workflow_ref in {json.dumps(allowed_workflows)}"
        )
        mapping = "google.subject=assertion.sub,attribute.repository_id=assertion.repository_id"
        flags = [f"--workload-identity-pool={pool}", "--location=global"]
        provider = run(["iam", "workload-identity-pools", "providers", "describe", "github", *flags], account, read=True)
        if provider is None:
            run(["iam", "workload-identity-pools", "providers", "create-oidc", "github", *flags,
                 "--issuer-uri=https://token.actions.githubusercontent.com",
                 f"--attribute-mapping={mapping}", f"--attribute-condition={condition}"], account)
        elif provider.get("attributeCondition") != condition or provider.get("oidc", {}).get("issuerUri") != "https://token.actions.githubusercontent.com":
            raise ValueError(f"Existing {pool}/github provider differs from the plan")
        principal = f"principal://iam.googleapis.com/projects/{NUMBER}/locations/global/workloadIdentityPools/{pool}/subject/repo:{REPOSITORY}:environment:{environment}"
        run(["iam", "service-accounts", "add-iam-policy-binding", sa,
             f"--member={principal}", "--role=roles/iam.workloadIdentityUser"], account)
        print(f"Provisioned {stage}: {ring}, {sa}, {pool}/github")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--account", default="swiftsjh@plaid.ai.kr")
    args = parser.parse_args()
    if args.apply:
        provision(args.account)
    else:
        print(json.dumps(plan(), indent=2))
