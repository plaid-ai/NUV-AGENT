#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 4 ]; then
  echo "Usage: $0 /path/to/nuv-agent_*.deb [/path/to/release-bom.json] [/path/to/release-bom.json.sig] [/path/to/exact-bom-artifact]" >&2
  exit 1
fi

DEB_PATH="$1"
DEB_PATH="$(realpath "$DEB_PATH")"
if [ ! -f "$DEB_PATH" ]; then
  echo "Deb not found: $DEB_PATH" >&2
  exit 1
fi
ROLLBACK_DEB_PATH="${APT_PREVIOUS_DEB_PATH:-}"
if [ -n "$ROLLBACK_DEB_PATH" ]; then
  ROLLBACK_DEB_PATH="$(realpath "$ROLLBACK_DEB_PATH")"
  if [ ! -f "$ROLLBACK_DEB_PATH" ] || [ "$ROLLBACK_DEB_PATH" = "$DEB_PATH" ]; then
    echo "Previous rollback Deb is missing or aliases the current Deb" >&2
    exit 1
  fi
fi
BOM_PATH="${2:-}"
SIGNATURE_PATH="${3:-}"
BOM_ARTIFACT_PATH="${4:-$DEB_PATH}"
if [ -n "$BOM_PATH" ]; then
  BOM_PATH="$(realpath "$BOM_PATH")"
  if [ ! -f "$BOM_PATH" ]; then
    echo "Release BOM not found: $BOM_PATH" >&2
    exit 1
  fi
  if [ -z "$SIGNATURE_PATH" ] && [ -f "$BOM_PATH.sig" ]; then
    SIGNATURE_PATH="$BOM_PATH.sig"
  fi
  if [ -n "$SIGNATURE_PATH" ]; then
    SIGNATURE_PATH="$(realpath "$SIGNATURE_PATH")"
    if [ ! -f "$SIGNATURE_PATH" ]; then
      echo "Release BOM signature not found: $SIGNATURE_PATH" >&2
      exit 1
    fi
  fi
  BOM_ARTIFACT_PATH="$(realpath "$BOM_ARTIFACT_PATH")"
  if [ ! -f "$BOM_ARTIFACT_PATH" ]; then
    echo "Release BOM artifact not found: $BOM_ARTIFACT_PATH" >&2
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
SKIP_APT_PUBLISH="${SKIP_APT_PUBLISH:-false}"
case "$SKIP_APT_PUBLISH" in
  true|false) ;;
  *) echo "SKIP_APT_PUBLISH must be true or false" >&2; exit 2 ;;
esac

mkdir -p "$PUBLIC_DIR"
if [ "$SKIP_APT_PUBLISH" = false ]; then
  aptly -config="$APTLY_CONFIG" repo create -distribution="$DIST" -component="$COMPONENT" "$REPO_NAME" || true
  if [ -n "$ROLLBACK_DEB_PATH" ]; then
    # The publisher database is ephemeral. Re-add the independently verified
    # previous package so the signed Packages index always supports one-step
    # rollback instead of merely retaining an unindexed pool object.
    aptly -config="$APTLY_CONFIG" repo add "$REPO_NAME" "$ROLLBACK_DEB_PATH"
  fi
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
fi

PUBLISHED_RELEASE_PATHS=()
if [ -n "$BOM_PATH" ]; then
  if [[ ! "${VERSION:-}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "VERSION must be an exact semantic version when publishing a BOM" >&2
    exit 1
  fi
  BOM_METADATA=$(PYTHONPATH="$PROJECT_ROOT" python3 - "$BOM_PATH" "$BOM_ARTIFACT_PATH" <<'PY'
from pathlib import Path
import sys

from nuvion_app.runtime.release_bom import load_release_bom, verify_release_artifact

bom = load_release_bom(Path(sys.argv[1]))
verify_release_artifact(bom, Path(sys.argv[2]))
print(f"{bom.bom_digest}\t{bom.agent_version}\t{bom.schema_version}")
PY
  )
  IFS=$'\t' read -r BOM_DIGEST BOM_VERSION BOM_SCHEMA <<< "$BOM_METADATA"
  if [ "$BOM_VERSION" != "$VERSION" ]; then
    echo "Release BOM version does not match VERSION" >&2
    exit 1
  fi

  VERSION_BOM_DIR="$PUBLIC_DIR/releases/$VERSION"
  CONTENT_BOM_DIR="$PUBLIC_DIR/releases/by-bom-sha256/$BOM_DIGEST"
  mkdir -p "$VERSION_BOM_DIR" "$CONTENT_BOM_DIR"
  if [ "$BOM_SCHEMA" = "2" ] && [ -z "$SIGNATURE_PATH" ]; then
    echo "release-bom-v2 requires a detached signature sidecar" >&2
    exit 1
  fi
  if [ "$BOM_SCHEMA" = "2" ]; then
    if [ -z "${RELEASE_KEYRING_PATH:-}" ]; then
      echo "RELEASE_KEYRING_PATH is required to publish release-bom-v2" >&2
      exit 1
    fi
    if [ -z "${RELEASE_TRUST_DOMAIN:-}" ]; then
      echo "RELEASE_TRUST_DOMAIN is required to publish release-bom-v2" >&2
      exit 1
    fi
    release_keyring_path="$(realpath "$RELEASE_KEYRING_PATH")"
    PYTHONPATH="$PROJECT_ROOT" python3 - \
      "$BOM_PATH" \
      "$SIGNATURE_PATH" \
      "$release_keyring_path" \
      "$RELEASE_TRUST_DOMAIN" <<'PY'
from pathlib import Path
import sys

from nuvion_app.runtime.release_bom import load_signed_release_bom
from nuvion_updater.trust import load_release_keyring

keyring = load_release_keyring(
    Path(sys.argv[3]),
    expected_trust_domain=sys.argv[4],
    require_root_owner=False,
)
load_signed_release_bom(
    Path(sys.argv[1]),
    Path(sys.argv[2]),
    release_keyring=keyring,
)
PY
  fi

  install_immutable_release_file() {
    local source="$1"
    local destination="$2"
    if [ -e "$destination" ]; then
      if ! cmp -s "$source" "$destination"; then
        echo "Refusing to overwrite immutable release bytes: $destination" >&2
        exit 1
      fi
    else
      install -m 0644 "$source" "$destination"
    fi
    PUBLISHED_RELEASE_PATHS+=("$destination")
  }

  for release_dir in "$VERSION_BOM_DIR" "$CONTENT_BOM_DIR"; do
    install_immutable_release_file "$BOM_PATH" "$release_dir/release-bom.json"
    install_immutable_release_file \
      "$BOM_ARTIFACT_PATH" \
      "$release_dir/$(basename "$BOM_ARTIFACT_PATH")"
    if [ -n "$SIGNATURE_PATH" ]; then
      install_immutable_release_file \
        "$SIGNATURE_PATH" \
        "$release_dir/release-bom.json.sig"
    fi
  done
fi

RELEASE_FILE="$PUBLIC_DIR/dists/$DIST/Release"
if [ "$SKIP_APT_PUBLISH" = false ] && [ ! -f "$RELEASE_FILE" ]; then
  echo "No published repo found (missing $RELEASE_FILE)" >&2
  exit 1
fi
if [ "$SKIP_APT_PUBLISH" = true ] && [ "${#PUBLISHED_RELEASE_PATHS[@]}" -eq 0 ]; then
  echo "SKIP_APT_PUBLISH requires an exact release BOM and artifact" >&2
  exit 1
fi
if [ "$SKIP_APT_PUBLISH" = false ]; then
  # APT indices are mutable, but pool artifacts are immutable under the Debian
  # (package, version, architecture) identity. Publish them through the same
  # create-only + byte-compare path as OTA artifacts.
  while IFS= read -r -d '' pool_artifact; do
    PUBLISHED_RELEASE_PATHS+=("$pool_artifact")
  done < <(find "$PUBLIC_DIR/pool" -type f -print0 | LC_ALL=C sort -z)
fi

echo "Syncing to gs://$BUCKET"
# Requires: gcloud auth login, gsutil configured

for published_release in "${PUBLISHED_RELEASE_PATHS[@]}"; do
  relative_path="${published_release#"$PUBLIC_DIR/"}"
  remote_path="gs://$BUCKET/$relative_path"
  # generation-match=0 is the Cloud Storage atomic create-only CAS. A 412 from
  # an existing/concurrent writer is idempotent only when its bytes are exact.
  if ! gcloud storage cp \
    --if-generation-match=0 \
    --cache-control="$CACHE_CONTROL" \
    "$published_release" "$remote_path"; then
    if ! gcloud storage cat "$remote_path" | cmp -s - "$published_release"; then
      echo "Refusing to overwrite existing immutable release bytes: $remote_path" >&2
      exit 1
    fi
  fi
done

if [ "$SKIP_APT_PUBLISH" = false ]; then
  # Immutable release and pool objects are created explicitly above. Only APT
  # indices/key/install metadata may pass through mutable rsync semantics.
  gsutil -m -h "Cache-Control:$CACHE_CONTROL" rsync -r \
    -x '^(releases/|pool/)' "$PUBLIC_DIR" "gs://$BUCKET"
fi

for published_release in "${PUBLISHED_RELEASE_PATHS[@]}"; do
  relative_path="${published_release#"$PUBLIC_DIR/"}"
  if ! gcloud storage cat "gs://$BUCKET/$relative_path" | cmp -s - "$published_release"; then
    echo "Published release verification failed: gs://$BUCKET/$relative_path" >&2
    exit 1
  fi
  echo "Verified immutable release: gs://$BUCKET/$relative_path"
done

echo "Published: https://$BUCKET"
