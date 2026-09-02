#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 VERSION /path/to/output.agent-bundle.tar.gz" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
VERSION="$1"
OUTPUT="$2"
SOURCE_EPOCH="${SOURCE_DATE_EPOCH:-}"
EXPECTED_COMPONENT_SHA="${COMPONENT_SHA:-}"
readonly BUILDER_IMAGE="python@sha256:9bb659dc6d5218917236f3711e866a5634bb4c2f208de9d4533aa4863f57c1d3"

if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "VERSION must be an exact semantic version" >&2
  exit 2
fi
if [[ ! "$SOURCE_EPOCH" =~ ^[0-9]+$ ]]; then
  echo "SOURCE_DATE_EPOCH must be a non-negative integer" >&2
  exit 2
fi
if [[ ! "$EXPECTED_COMPONENT_SHA" =~ ^[0-9a-f]{40}$|^[0-9a-f]{64}$ ]]; then
  echo "COMPONENT_SHA must be the exact stamped full source revision" >&2
  exit 2
fi

# The outer invocation is a hermetic launcher. Pinning the platform-specific
# image manifest fixes CPython, pip, libc and the build userspace across reruns;
# the inner invocation never runs on the mutable GitHub runner filesystem.
if [ "${NUVION_BUNDLE_INNER:-0}" != "1" ]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required for the digest-pinned arm64 bundle build" >&2
    exit 1
  fi
  case "$(uname -m)" in
    arm64|aarch64) ;;
    *)
      echo "agent-bundle build requires a native arm64 Docker host" >&2
      exit 1
      ;;
  esac
  mkdir -p "$(dirname "$OUTPUT")"
  output_directory="$(cd "$(dirname "$OUTPUT")" && pwd -P)"
  output_name="$(basename "$OUTPUT")"
  docker run --rm \
    --platform linux/arm64 \
    --user "$(id -u):$(id -g)" \
    --read-only \
    --tmpfs /tmp:rw,exec,nosuid,nodev,size=2g \
    --mount "type=bind,src=$ROOT_DIR,dst=/src,readonly" \
    --mount "type=bind,src=$output_directory,dst=/out" \
    --env NUVION_BUNDLE_INNER=1 \
    --env "SOURCE_DATE_EPOCH=$SOURCE_EPOCH" \
    --env "COMPONENT_SHA=$EXPECTED_COMPONENT_SHA" \
    "$BUILDER_IMAGE" \
    /src/packaging/release/build-agent-bundle.sh \
      "$VERSION" "/out/$output_name"
  exit $?
fi

mkdir -p "$(dirname "$OUTPUT")"
OUTPUT="$(cd "$(dirname "$OUTPUT")" && pwd)/$(basename "$OUTPUT")"
bundle_root="$(mktemp -d)"
cleanup_bundle() {
  rm -rf -- "$bundle_root"
}
trap cleanup_bundle EXIT

slot_root="$bundle_root/slot"
source_root="$bundle_root/source"
site_packages="$slot_root/venv/lib/python3.12/site-packages"
mkdir -p "$slot_root/bin" "$slot_root/share" "$slot_root/venv/bin" "$site_packages"
mkdir -p "$source_root"
install -m 0644 "$ROOT_DIR/pyproject.toml" "$source_root/pyproject.toml"
install -m 0644 "$ROOT_DIR/README.md" "$source_root/README.md"
cp -a "$ROOT_DIR/nuvion_app" "$source_root/"
cp -a "$ROOT_DIR/nuvion_updater" "$source_root/"
find "$source_root" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
find "$source_root" -depth -type d -name __pycache__ -exec rm -rf -- {} +
PYTHONDONTWRITEBYTECODE=1 python3 -m pip install \
  --disable-pip-version-check \
  --no-cache-dir \
  --no-compile \
  --ignore-installed \
  --target "$site_packages" \
  --only-binary=:all: \
  --require-hashes \
  --requirement "$ROOT_DIR/packaging/release/requirements-agent-bundle-arm64.txt"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$site_packages" python3 -m pip install \
  --disable-pip-version-check \
  --no-cache-dir \
  --no-compile \
  --ignore-installed \
  --target "$site_packages" \
  --no-deps \
  --no-build-isolation \
  "$source_root"
PYTHONDONTWRITEBYTECODE=1 python3 -m pip install \
  --disable-pip-version-check \
  --no-cache-dir \
  --no-compile \
  --ignore-installed \
  --target "$site_packages" \
  --no-deps \
  --only-binary=:all: \
  --require-hashes \
  --requirement "$ROOT_DIR/packaging/deb/requirements-depthai-arm64.txt"
cat > "$slot_root/venv/bin/python" <<'PYTHON_WRAPPER'
#!/bin/sh
set -eu
slot_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
export PYTHONNOUSERSITE=1
export PYTHONPATH="$slot_root/venv/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
exec "${NUVION_SYSTEM_PYTHON:-/usr/bin/python3}" "$@"
PYTHON_WRAPPER
install -m 0755 \
  "$ROOT_DIR/packaging/systemd/nuv-agent-slot-entrypoint" \
  "$slot_root/bin/nuv-agent"
PYTHONPATH="$site_packages" python3 - "$site_packages" \
  > "$slot_root/share/python-freeze.txt" <<'PY'
from importlib.metadata import distributions
import sys

packages = {
    (distribution.metadata.get("Name") or "unknown", distribution.version)
    for distribution in distributions(path=[sys.argv[1]])
}
for name, version in sorted(packages, key=lambda item: item[0].lower()):
    print(f"{name}=={version}")
PY

# Remove build-host paths from PEP 610 metadata and wheel bookkeeping.
# Runtime enters only through the canonical system-Python wrapper.
find "$slot_root/venv" -type f -name direct_url.json -exec unlink {} \;
find "$slot_root/venv" -type f \( -name '*.pyc' -o -name '*.pyo' -o -name RECORD \) -exec unlink {} \;
find "$slot_root/venv" -depth -type d -name __pycache__ -empty -exec rmdir {} \;
if find "$slot_root" -type l -print -quit | grep -q .; then
  echo "agent-bundle must not contain symbolic links" >&2
  find "$slot_root" -type l -print | sed "s#^$slot_root/##" | head -20 >&2
  exit 1
fi
find "$slot_root" -type d -exec chmod 0755 {} +
find "$slot_root" -type f -exec chmod 0644 {} +
chmod 0755 "$slot_root/bin/nuv-agent" "$slot_root/venv/bin/python"

# Validate the post-pruning runtime from inside the slot. Clearing PYTHONPATH
# and changing cwd prevent the checkout from shadowing the installed package.
(
  cd "$slot_root"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='' NUVION_SYSTEM_PYTHON=/usr/local/bin/python3 \
    "$slot_root/venv/bin/python" -s - \
      "$VERSION" "$EXPECTED_COMPONENT_SHA" "$slot_root" <<'PY'
from importlib.metadata import version
from pathlib import Path
import sys

import nuvion_app
from nuvion_app import build_info

expected_version, expected_sha, raw_slot = sys.argv[1:]
slot = Path(raw_slot).resolve(strict=True)
package_path = Path(nuvion_app.__file__).resolve(strict=True)
if not package_path.is_relative_to(slot / "venv"):
    raise SystemExit("agent-bundle imported nuvion_app outside the immutable slot")
if version("nuv-agent") != expected_version:
    raise SystemExit("agent-bundle distribution version does not match the release")
if build_info.AGENT_VERSION != expected_version:
    raise SystemExit("agent-bundle build_info version does not match the release")
if build_info.COMPONENT_SHA != expected_sha:
    raise SystemExit("agent-bundle component SHA does not match the stamped release")
PY
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='' NUVION_SYSTEM_PYTHON=/usr/local/bin/python3 \
    "$slot_root/bin/nuv-agent" --help >/dev/null
)
find "$slot_root" -exec touch -h -d "@$SOURCE_EPOCH" {} +

file_list="$bundle_root/files.list"
(cd "$slot_root" && find . -mindepth 1 -printf '%P\0' | LC_ALL=C sort -z > "$file_list")
raw_tar="$bundle_root/agent-bundle.tar"
tar \
  --create \
  --file "$raw_tar" \
  --directory "$slot_root" \
  --null \
  --no-recursion \
  --hard-dereference \
  --numeric-owner \
  --owner=0 \
  --group=0 \
  --mtime="@$SOURCE_EPOCH" \
  --files-from "$file_list"
gzip -n -9 "$raw_tar"
candidate="$raw_tar.gz"

if [ -e "$OUTPUT" ]; then
  if ! cmp -s "$candidate" "$OUTPUT"; then
    echo "Refusing to overwrite an existing agent-bundle with different bytes" >&2
    exit 1
  fi
else
  install -m 0644 "$candidate" "$OUTPUT"
fi

echo "$OUTPUT"
