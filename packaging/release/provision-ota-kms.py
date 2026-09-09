#!/usr/bin/env python3
"""Add exact-workflow OTA federation; defaults to a reviewable plan."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re

spec = importlib.util.spec_from_file_location("provision_kms", Path(__file__).with_name("provision-cloud-kms.py"))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def plan(publisher_sha: str):
    if not re.fullmatch(r"[0-9a-f]{40}", publisher_sha):
        raise ValueError("An exact approved workflow commit is required")
    tag = "candidate-publisher-v16"
    providers = {}
    for name, environment, ref, workflow in (
        ("candidate-v16", "iq9075-candidate-sign", f"refs/tags/{tag}", "iq9075-candidate-trusted-publish.yml"),
        ("release-main-v16", "iq9075-release", "refs/heads/main", "release-publish.yml"),
    ):
        subject = f"repo:{base.REPOSITORY}:environment:{environment}"
        condition = (
            f"assertion.repository_id == '{base.REPOSITORY_ID}' && "
            f"assertion.repository_owner_id == '{base.OWNER_ID}' && "
            f"assertion.ref == '{ref}' && assertion.event_name == 'workflow_dispatch' && "
            f"assertion.sub == '{subject}' && "
            f"assertion.workflow_ref == '{base.REPOSITORY}/.github/workflows/{workflow}@{ref}' && "
            f"assertion.workflow_sha == '{publisher_sha}' && "
            "assertion.actor_id in ['57535980', '89565530']"
        )
        providers[name] = {"subject": subject, "condition": condition}
    return providers


def apply(publisher_sha, account):
    pool = "nuv-agent-sign-dev"
    flags = [f"--workload-identity-pool={pool}", "--location=global"]
    mapping = {"google.subject": "assertion.sub", "attribute.repository_id": "assertion.repository_id"}
    issuer = "https://token.actions.githubusercontent.com"
    for name, config in plan(publisher_sha).items():
        existing = base.run(["iam", "workload-identity-pools", "providers", "describe", name, *flags], account, read=True)
        if existing is None:
            base.run(["iam", "workload-identity-pools", "providers", "create-oidc", name, *flags,
                      f"--issuer-uri={issuer}",
                      "--attribute-mapping=" + ",".join(f"{k}={v}" for k, v in mapping.items()),
                      f"--attribute-condition={config['condition']}"], account)
        elif (existing.get("attributeCondition") != config["condition"]
              or existing.get("attributeMapping") != mapping
              or existing.get("oidc", {}).get("issuerUri") != issuer
              or existing.get("disabled", False)):
            raise ValueError(f"Existing {name} differs; refuse an implicit trust change")
        principal = f"principal://iam.googleapis.com/projects/{base.NUMBER}/locations/global/workloadIdentityPools/{pool}/subject/{config['subject']}"
        base.run(["iam", "service-accounts", "add-iam-policy-binding",
                  f"nuv-agent-sign-dev@{base.PROJECT}.iam.gserviceaccount.com",
                  f"--member={principal}", "--role=roles/iam.workloadIdentityUser"], account)
        print(f"Provisioned exact workflow: {name} at {publisher_sha}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publisher-sha", required=True)
    parser.add_argument("--account", default="swiftsjh@plaid.ai.kr")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.apply:
        apply(args.publisher_sha, args.account)
    else:
        print(json.dumps(plan(args.publisher_sha), indent=2))
