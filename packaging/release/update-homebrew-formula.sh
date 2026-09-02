#!/usr/bin/env bash
set -euo pipefail

FORMULA_PATH=${FORMULA_PATH:-""}
URL=${URL:-""}
SHA256=${SHA256:-""}
VERSION=${VERSION:-""}

usage() {
  echo "Usage: FORMULA_PATH=... URL=... SHA256=... VERSION=... $0" >&2
}

if [ -z "$FORMULA_PATH" ] || [ -z "$URL" ] || [ -z "$SHA256" ] || [ -z "$VERSION" ]; then
  usage
  exit 1
fi

if [ ! -f "$FORMULA_PATH" ]; then
  echo "Formula not found: $FORMULA_PATH" >&2
  exit 1
fi

FORMULA_PATH="$FORMULA_PATH" URL="$URL" SHA256="$SHA256" VERSION="$VERSION" \
python3 - <<'PY'
import os
from pathlib import Path
import re

path = Path(os.environ["FORMULA_PATH"])
text = path.read_text(encoding="utf-8")
url = os.environ["URL"]
sha = os.environ["SHA256"]
version = os.environ["VERSION"]
if (
    not re.fullmatch(r"[0-9a-f]{64}", sha)
    or not re.fullmatch(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)", version)
    or not (url.startswith("https://") or url.startswith("file:///"))
    or any(character in url for character in ('"', "\\", "\n", "\r"))
):
    raise SystemExit("unsafe Homebrew formula identity")

parts = text.split("\n  resource ", 1)
head = parts[0]
tail = f"\n  resource {parts[1]}" if len(parts) > 1 else ""

head = head.replace("__URL__", url)
head = head.replace("__SHA256__", sha)
head, url_count = re.subn(r'url\s+"[^"]+"', f'url "{url}"', head, count=1)
head, sha_count = re.subn(r'sha256\s+"[^"]+"', f'sha256 "{sha}"', head, count=1)
head, version_count = re.subn(r'version\s+"[^"]+"', f'version "{version}"', head, count=1)
if (url_count, sha_count, version_count) != (1, 1, 1):
    raise SystemExit("Homebrew formula identity fields are missing")

text = head + tail
path.write_text(text, encoding="utf-8")
print(f"Updated: {path}")
PY
