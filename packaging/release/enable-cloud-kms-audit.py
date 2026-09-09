#!/usr/bin/env python3
"""Enable KMS read/sign audit events, preserving IAM bindings with etag CAS."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--account", default="swiftsjh@plaid.ai.kr")
    args = parser.parse_args()
    flags = ["--account=" + args.account, "--quiet", "--format=json"]
    def gcloud(*command):
        result = subprocess.run(["gcloud", *command, *flags], capture_output=True, text=True, check=True, timeout=60)
        return json.loads(result.stdout)
    if args.apply:
        gcloud("services", "enable", "logging.googleapis.com", "--project=plaid-451114")
    policy = gcloud("projects", "get-iam-policy", "plaid-451114")
    before = json.dumps(policy, sort_keys=True)
    if not policy.get("etag"):
        raise ValueError("IAM etag is required to preserve concurrent changes")
    configs = policy.setdefault("auditConfigs", [])
    existing = next((item for item in configs if item["service"] == "cloudkms.googleapis.com"), None)
    if existing is None:
        existing = {"service": "cloudkms.googleapis.com", "auditLogConfigs": []}
        configs.append(existing)
    for log_type in ("ADMIN_READ", "DATA_READ"):
        setting = next((item for item in existing["auditLogConfigs"] if item["logType"] == log_type), None)
        if setting is None:
            existing["auditLogConfigs"].append({"logType": log_type})
        elif setting.get("exemptedMembers"):
            raise ValueError("Existing audit exemptions need explicit review")
    if args.apply and json.dumps(policy, sort_keys=True) != before:
        with tempfile.TemporaryDirectory(prefix="kms-audit-policy-") as tmp:
            path = Path(tmp) / "policy.json"
            path.write_text(json.dumps(policy))
            path.chmod(0o600)
            applied = gcloud("projects", "set-iam-policy", "plaid-451114", str(path))
            if applied["bindings"] != policy["bindings"]:
                raise ValueError("IAM bindings changed unexpectedly")
    print(json.dumps({"applied": args.apply, "audit": existing}, indent=2))


if __name__ == "__main__":
    main()
