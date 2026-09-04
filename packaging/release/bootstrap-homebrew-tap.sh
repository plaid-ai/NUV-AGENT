#!/usr/bin/env bash
set -euo pipefail

ORG=${ORG:-plaid-ai}
REPO=${REPO:-homebrew-NUV-AGENT-HOMEBREW}
FORMULA_SRC=${FORMULA_SRC:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../homebrew" && pwd)/nuv-agent.rb"}
VISIBILITY=${VISIBILITY:-public}

if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI not found. Install GitHub CLI and authenticate." >&2
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "gh not authenticated. Run: gh auth login" >&2
  exit 1
fi

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

if gh repo view "$ORG/$REPO" >/dev/null 2>&1; then
  echo "Repo exists: $ORG/$REPO"
else
  gh repo create "$ORG/$REPO" --"$VISIBILITY" --confirm
fi

git clone "https://github.com/$ORG/$REPO.git" "$TMP_DIR/tap"
mkdir -p "$TMP_DIR/tap/Formula"
cp "$FORMULA_SRC" "$TMP_DIR/tap/Formula/nuv-agent.rb"

cd "$TMP_DIR/tap"
git add Formula/nuv-agent.rb
if ! git diff --cached --quiet; then
  git config user.email "release-bot@plaidai.io"
  git config user.name "release-bot"
  git commit -m "Add nuv-agent formula"
  git push origin HEAD
else
  echo "No changes to commit."
fi

echo "Tap repo ready: https://github.com/$ORG/$REPO"
