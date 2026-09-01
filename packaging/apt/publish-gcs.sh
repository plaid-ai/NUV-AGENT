#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  echo "Usage: $0 /path/to/nuv-agent_*.deb [/path/to/release-bom.json]" >&2
  exit 1
fi

DEB_PATH="$1"
DEB_PATH="$(realpath "$DEB_PATH")"
if [ ! -f "$DEB_PATH" ]; then
  echo "Deb not found: $DEB_PATH" >&2
  exit 1
fi
BOM_PATH="${2:-}"
if [ -n "$BOM_PATH" ]; then
  BOM_PATH="$(realpath "$BOM_PATH")"
  if [ ! -f "$BOM_PATH" ]; then
    echo "Release BOM not found: $BOM_PATH" >&2
    exit 1
  fi
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
APTLY_CONFIG="$ROOT_DIR/aptly.conf"
REPO_NAME=${REPO_NAME:-nuv-agent}
DIST=${DIST:-stable}
COMPONENT=${COMPONENT:-main}
ARCH=${ARCH:-arm64}
BUCKET=${BUCKET:-apt.plaidai.io}
CACHE_CONTROL=${CACHE_CONTROL:-"no-cache, max-age=0"}
PUBLIC_DIR="$ROOT_DIR/.aptly/public"
PUBLIC_KEY_PATH="$PUBLIC_DIR/public.gpg"
INSTALL_SCRIPT_SRC="$ROOT_DIR/install-apt.sh"
INSTALL_SCRIPT_DST="$PUBLIC_DIR/install-apt.sh"
PROJECT_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"

aptly -config="$APTLY_CONFIG" repo create -distribution="$DIST" -component="$COMPONENT" "$REPO_NAME" || true
aptly -config="$APTLY_CONFIG" repo add "$REPO_NAME" "$DEB_PATH"

if aptly -config="$APTLY_CONFIG" publish list | grep -q "^$DIST"; then
  aptly -config="$APTLY_CONFIG" publish update -distribution="$DIST" "$REPO_NAME"
else
  aptly -config="$APTLY_CONFIG" publish repo -distribution="$DIST" -architectures="$ARCH" -component="$COMPONENT" "$REPO_NAME"
fi

if ! command -v gpg >/dev/null 2>&1; then
  echo "gpg not found. Install gpg to export the public key." >&2
  exit 1
fi

mkdir -p "$PUBLIC_DIR"
if [ ! -s "$PUBLIC_KEY_PATH" ]; then
  echo "Exporting public GPG key to $PUBLIC_KEY_PATH"
  if [ -n "${GPG_KEY_ID:-}" ]; then
    gpg --armor --export "$GPG_KEY_ID" > "$PUBLIC_KEY_PATH"
  else
    gpg --armor --export > "$PUBLIC_KEY_PATH"
  fi
fi

if [ ! -s "$PUBLIC_KEY_PATH" ]; then
  echo "Failed to export public key. Set GPG_KEY_ID and retry." >&2
  exit 1
fi

if [ -f "$INSTALL_SCRIPT_SRC" ]; then
  cp "$INSTALL_SCRIPT_SRC" "$INSTALL_SCRIPT_DST"
  chmod 0644 "$INSTALL_SCRIPT_DST"
fi

PUBLISHED_BOM_PATHS=()
if [ -n "$BOM_PATH" ]; then
  if [[ ! "${VERSION:-}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "VERSION must be an exact semantic version when publishing a BOM" >&2
    exit 1
  fi
  BOM_METADATA=$(PYTHONPATH="$PROJECT_ROOT" python3 - "$BOM_PATH" "$DEB_PATH" <<'PY'
from pathlib import Path
import sys

from nuvion_app.runtime.release_bom import load_release_bom, verify_release_artifact

bom = load_release_bom(Path(sys.argv[1]))
verify_release_artifact(bom, Path(sys.argv[2]))
print(f"{bom.bom_digest}\t{bom.agent_version}")
PY
  )
  IFS=$'\t' read -r BOM_DIGEST BOM_VERSION <<< "$BOM_METADATA"
  if [ "$BOM_VERSION" != "$VERSION" ]; then
    echo "Release BOM version does not match VERSION" >&2
    exit 1
  fi

  VERSION_BOM_DIR="$PUBLIC_DIR/releases/$VERSION"
  CONTENT_BOM_DIR="$PUBLIC_DIR/releases/by-bom-sha256/$BOM_DIGEST"
  mkdir -p "$VERSION_BOM_DIR" "$CONTENT_BOM_DIR"
  VERSION_BOM_PATH="$VERSION_BOM_DIR/$(basename "$BOM_PATH")"
  CONTENT_BOM_PATH="$CONTENT_BOM_DIR/$(basename "$BOM_PATH")"
  for destination in "$VERSION_BOM_PATH" "$CONTENT_BOM_PATH"; do
    if [ -e "$destination" ]; then
      if ! cmp -s "$BOM_PATH" "$destination"; then
        echo "Refusing to overwrite an existing release BOM: $destination" >&2
        exit 1
      fi
    else
      install -m 0644 "$BOM_PATH" "$destination"
    fi
  done
  PUBLISHED_BOM_PATHS+=("$VERSION_BOM_PATH" "$CONTENT_BOM_PATH")
fi

RELEASE_FILE="$PUBLIC_DIR/dists/$DIST/Release"
if [ ! -f "$RELEASE_FILE" ]; then
  echo "No published repo found (missing $RELEASE_FILE)" >&2
  exit 1
fi

echo "Syncing to gs://$BUCKET"
# Requires: gcloud auth login, gsutil configured

for published_bom in "${PUBLISHED_BOM_PATHS[@]}"; do
  relative_path="${published_bom#"$PUBLIC_DIR/"}"
  remote_path="gs://$BUCKET/$relative_path"
  if gsutil -q stat "$remote_path"; then
    if ! gsutil cat "$remote_path" | cmp -s - "$published_bom"; then
      echo "Refusing to overwrite an existing remote BOM: $remote_path" >&2
      exit 1
    fi
  fi
done

gsutil -m -h "Cache-Control:$CACHE_CONTROL" rsync -r "$PUBLIC_DIR" "gs://$BUCKET"
gsutil -m setmeta -h "Cache-Control:$CACHE_CONTROL" "gs://$BUCKET/**" >/dev/null

for published_bom in "${PUBLISHED_BOM_PATHS[@]}"; do
  relative_path="${published_bom#"$PUBLIC_DIR/"}"
  if ! gsutil cat "gs://$BUCKET/$relative_path" | cmp -s - "$published_bom"; then
    echo "Published BOM verification failed: gs://$BUCKET/$relative_path" >&2
    exit 1
  fi
  echo "Verified BOM: gs://$BUCKET/$relative_path"
done

echo "Published: https://$BUCKET"
