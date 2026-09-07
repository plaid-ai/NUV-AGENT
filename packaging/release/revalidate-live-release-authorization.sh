#!/usr/bin/env bash
set -euo pipefail

for variable in \
  GITHUB_REPOSITORY GITHUB_TOKEN RELEASE_AUTH_VERSION \
  RELEASE_AUTH_COMPONENT_SHA RELEASE_AUTH_GATE_RUN_ID \
  RELEASE_AUTH_GATE_CHECK_ID RELEASE_AUTH_GATE_CHECK_SUITE_ID \
  RELEASE_AUTH_GATE_WORKFLOW_SHA256; do
  [ -n "${!variable:-}" ] || {
    echo "Missing live release authorization input: $variable" >&2
    exit 2
  }
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
publisher_root="$(cd "$script_dir/../.." && pwd -P)"
workflow="$publisher_root/.github/workflows/agent-release-gate.yml"

env -u PYTHONPATH PYTHONNOUSERSITE=1 python3 -I \
  "$script_dir/verify-agent-release-gate.py" \
  --repository "$GITHUB_REPOSITORY" \
  --component-sha "$RELEASE_AUTH_COMPONENT_SHA" \
  --policy "$script_dir/release-security-policy.json" \
  --candidate-workflow "$workflow" \
  --trusted-workflow "$workflow" \
  --expected-run-id "$RELEASE_AUTH_GATE_RUN_ID" \
  --expected-check-id "$RELEASE_AUTH_GATE_CHECK_ID" \
  --expected-check-suite-id "$RELEASE_AUTH_GATE_CHECK_SUITE_ID" \
  --expected-workflow-sha256 "$RELEASE_AUTH_GATE_WORKFLOW_SHA256"

env -u PYTHONPATH PYTHONNOUSERSITE=1 python3 -I \
  "$script_dir/verify-release-readiness.py" \
  --policy "$script_dir/release-readiness.json" \
  --version "$RELEASE_AUTH_VERSION" \
  --component-sha "$RELEASE_AUTH_COMPONENT_SHA" \
  --gate-run-id "$RELEASE_AUTH_GATE_RUN_ID" \
  --gate-check-id "$RELEASE_AUTH_GATE_CHECK_ID" \
  --gate-check-suite-id "$RELEASE_AUTH_GATE_CHECK_SUITE_ID" \
  --gate-workflow-sha256 "$RELEASE_AUTH_GATE_WORKFLOW_SHA256" \
  --security-policy "$script_dir/release-security-policy.json" \
  --signer-directory "$script_dir/trusted-tag-signers" \
  --candidate-harness "$publisher_root/packaging/dev/test-iq9075.sh" \
  --candidate-fleet-runner "$publisher_root/packaging/dev/run-iq9075-fleet-e2e.py" \
  --candidate-config-stream-runner "$publisher_root/packaging/dev/run-iq9075-config-stream-e2e.py" \
  --candidate-board-tool "$publisher_root/packaging/dev/iq9075-board-e2e.py" \
  --candidate-installer "$publisher_root/packaging/dev/install-iq9075.sh" \
  --candidate-rollout-control "$publisher_root/packaging/dev/run-iq9075-agent-rollout-control.py"
