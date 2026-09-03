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
RUNTIME_ROOT="${APT_RUNTIME_ROOT:-$ROOT_DIR}"
if [ -L "$RUNTIME_ROOT" ]; then
  echo "APT_RUNTIME_ROOT must not be a symbolic link" >&2
  exit 1
fi
mkdir -p "$RUNTIME_ROOT"
RUNTIME_ROOT="$(realpath "$RUNTIME_ROOT")"
if [ ! -d "$RUNTIME_ROOT" ]; then
  echo "APT_RUNTIME_ROOT must be a regular directory" >&2
  exit 1
fi
cd "$RUNTIME_ROOT"
APTLY_CONFIG="$ROOT_DIR/aptly.conf"
REPO_NAME=${REPO_NAME:-nuv-agent}
DIST=${DIST:-stable}
COMPONENT=${COMPONENT:-main}
ARCH=${ARCH:-arm64}
BUCKET=${BUCKET:-apt.plaidai.io}
CACHE_CONTROL=${CACHE_CONTROL:-"no-cache, max-age=0"}
PUBLIC_DIR="${APT_PUBLIC_DIR:-$RUNTIME_ROOT/.aptly/public}"
mkdir -p "$PUBLIC_DIR"
PUBLIC_DIR="$(realpath "$PUBLIC_DIR")"
PUBLIC_KEY_PATH="$PUBLIC_DIR/public.gpg"
INSTALL_SCRIPT_SRC="$ROOT_DIR/install-apt.sh"
INSTALL_SCRIPT_DST="$PUBLIC_DIR/install-apt.sh"
PROJECT_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"
SKIP_APT_PUBLISH="${SKIP_APT_PUBLISH:-false}"
case "$SKIP_APT_PUBLISH" in
  true|false) ;;
  *) echo "SKIP_APT_PUBLISH must be true or false" >&2; exit 2 ;;
esac
OTA_CONTENT_ONLY="${OTA_CONTENT_ONLY:-false}"
case "$OTA_CONTENT_ONLY" in
  true|false) ;;
  *) echo "OTA_CONTENT_ONLY must be true or false" >&2; exit 2 ;;
esac
if [ "$OTA_CONTENT_ONLY" = true ] && [ "$SKIP_APT_PUBLISH" != true ]; then
  echo "OTA_CONTENT_ONLY requires SKIP_APT_PUBLISH=true" >&2
  exit 2
fi
APTLY_PASSPHRASE_FILE="${APTLY_PASSPHRASE_FILE:-}"

if [ "$SKIP_APT_PUBLISH" = false ]; then
  if [ -z "$APTLY_PASSPHRASE_FILE" ] || [ -L "$APTLY_PASSPHRASE_FILE" ] \
    || [ ! -f "$APTLY_PASSPHRASE_FILE" ] || [ ! -s "$APTLY_PASSPHRASE_FILE" ]; then
    echo "APTLY_PASSPHRASE_FILE must be a non-empty regular file" >&2
    exit 1
  fi
  APTLY_PASSPHRASE_FILE="$(realpath "$APTLY_PASSPHRASE_FILE")"
  if file_mode="$(stat -c '%a' "$APTLY_PASSPHRASE_FILE" 2>/dev/null)"; then
    :
  else
    file_mode="$(stat -f '%Lp' "$APTLY_PASSPHRASE_FILE")"
  fi
  if file_owner="$(stat -c '%u' "$APTLY_PASSPHRASE_FILE" 2>/dev/null)"; then
    :
  else
    file_owner="$(stat -f '%u' "$APTLY_PASSPHRASE_FILE")"
  fi
  if [ "$file_mode" != "600" ] || [ "$file_owner" != "$(id -u)" ]; then
    echo "APTLY_PASSPHRASE_FILE must be mode 0600 and owned by the publisher" >&2
    exit 1
  fi
fi

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
    aptly -config="$APTLY_CONFIG" publish update \
      -batch -passphrase-file="$APTLY_PASSPHRASE_FILE" "$DIST"
  else
    aptly -config="$APTLY_CONFIG" publish repo \
      -batch -passphrase-file="$APTLY_PASSPHRASE_FILE" -acquire-by-hash \
      -distribution="$DIST" -architectures="$ARCH" \
      -component="$COMPONENT" "$REPO_NAME"
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
VERSION_PAYLOAD_PATHS=()
CONTENT_RELEASE_PATHS=()
POOL_ARTIFACT_PATHS=()
APT_BY_HASH_PATHS=()
APT_MUTABLE_METADATA_PATHS=()
APT_DISCOVERY_PATH=""
VERSION_DISCOVERY_PATH=""
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

  CONTENT_BOM_DIR="$PUBLIC_DIR/releases/by-bom-sha256/$BOM_DIGEST"
  mkdir -p "$CONTENT_BOM_DIR"
  if [ "$OTA_CONTENT_ONLY" = false ]; then
    VERSION_BOM_DIR="$PUBLIC_DIR/releases/$VERSION"
    mkdir -p "$VERSION_BOM_DIR"
  fi
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

  if [ "$OTA_CONTENT_ONLY" = false ]; then
    version_artifact="$VERSION_BOM_DIR/$(basename "$BOM_ARTIFACT_PATH")"
    install_immutable_release_file "$BOM_ARTIFACT_PATH" "$version_artifact"
    VERSION_PAYLOAD_PATHS+=("$version_artifact")
    if [ -n "$SIGNATURE_PATH" ]; then
      version_signature="$VERSION_BOM_DIR/release-bom.json.sig"
      install_immutable_release_file "$SIGNATURE_PATH" "$version_signature"
      VERSION_PAYLOAD_PATHS+=("$version_signature")
    fi
  fi

  content_artifact="$CONTENT_BOM_DIR/$(basename "$BOM_ARTIFACT_PATH")"
  install_immutable_release_file "$BOM_ARTIFACT_PATH" "$content_artifact"
  CONTENT_RELEASE_PATHS+=("$content_artifact")
  if [ -n "$SIGNATURE_PATH" ]; then
    content_signature="$CONTENT_BOM_DIR/release-bom.json.sig"
    install_immutable_release_file "$SIGNATURE_PATH" "$content_signature"
    CONTENT_RELEASE_PATHS+=("$content_signature")
  fi
  content_bom="$CONTENT_BOM_DIR/release-bom.json"
  install_immutable_release_file "$BOM_PATH" "$content_bom"
  CONTENT_RELEASE_PATHS+=("$content_bom")

  if [ "$OTA_CONTENT_ONLY" = false ]; then
    VERSION_DISCOVERY_PATH="$VERSION_BOM_DIR/release-bom.json"
    install_immutable_release_file "$BOM_PATH" "$VERSION_DISCOVERY_PATH"
  fi
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
    POOL_ARTIFACT_PATHS+=("$pool_artifact")
    PUBLISHED_RELEASE_PATHS+=("$pool_artifact")
  done < <(find "$PUBLIC_DIR/pool" -type f -print0 | LC_ALL=C sort -z)
  if ! grep -qx 'Acquire-By-Hash: yes' "$RELEASE_FILE"; then
    echo "APT Release must enable Acquire-By-Hash before publication" >&2
    exit 1
  fi
  while IFS= read -r -d '' apt_by_hash; do
    APT_BY_HASH_PATHS+=("$apt_by_hash")
    PUBLISHED_RELEASE_PATHS+=("$apt_by_hash")
  done < <(
    find "$PUBLIC_DIR/dists/$DIST" -type f -path '*/by-hash/*' \
      -print0 | LC_ALL=C sort -z
  )
  if [ "${#APT_BY_HASH_PATHS[@]}" -eq 0 ]; then
    echo "APT Acquire-By-Hash metadata is missing" >&2
    exit 1
  fi
  APT_DISCOVERY_PATH="$PUBLIC_DIR/dists/$DIST/InRelease"
  if [ ! -s "$APT_DISCOVERY_PATH" ]; then
    echo "APT signed InRelease discovery marker is missing" >&2
    exit 1
  fi
  while IFS= read -r -d '' apt_metadata; do
    if [ "$apt_metadata" != "$APT_DISCOVERY_PATH" ]; then
      APT_MUTABLE_METADATA_PATHS+=("$apt_metadata")
    fi
  done < <(
    find "$PUBLIC_DIR" -type f \
      ! -path "$PUBLIC_DIR/releases/*" \
      ! -path "$PUBLIC_DIR/pool/*" \
      ! -path '*/by-hash/*' \
      -print0 | LC_ALL=C sort -z
  )
fi

echo "Syncing to gs://$BUCKET"
# Requires an authenticated gcloud storage client.

upload_immutable_release() {
  local published_release="$1"
  local relative_path="${published_release#"$PUBLIC_DIR/"}"
  local remote_path="gs://$BUCKET/$relative_path"
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
}

# The version-scoped release-bom.json is the discovery/eligibility marker. All
# artifact and detached-signature bytes, including their content-addressed
# copies, must be durable first. A failed stage is safely rerunnable because
# every preceding object is create-only and byte-compared.
if [ "${#VERSION_PAYLOAD_PATHS[@]}" -gt 0 ]; then
  for published_release in "${VERSION_PAYLOAD_PATHS[@]}"; do
    upload_immutable_release "$published_release"
  done
fi
if [ "${#CONTENT_RELEASE_PATHS[@]}" -gt 0 ]; then
  for published_release in "${CONTENT_RELEASE_PATHS[@]}"; do
    upload_immutable_release "$published_release"
  done
fi
if [ "${#POOL_ARTIFACT_PATHS[@]}" -gt 0 ]; then
  for published_release in "${POOL_ARTIFACT_PATHS[@]}"; do
    upload_immutable_release "$published_release"
  done
fi
if [ "${#APT_BY_HASH_PATHS[@]}" -gt 0 ]; then
  for published_release in "${APT_BY_HASH_PATHS[@]}"; do
    upload_immutable_release "$published_release"
  done
fi

if [ -n "$VERSION_DISCOVERY_PATH" ]; then
  upload_immutable_release "$VERSION_DISCOVERY_PATH"
fi

if [ "$SKIP_APT_PUBLISH" = false ]; then
  upload_mutable_metadata() {
    local local_path="$1"
    local relative_path="${local_path#"$PUBLIC_DIR/"}"
    gcloud storage cp \
      --cache-control="$CACHE_CONTROL" \
      "$local_path" "gs://$BUCKET/$relative_path"
  }

  # APT's by-hash objects are already durable. Publish ordinary metadata next
  # and the signed InRelease discovery pointer last. Readers that still hold
  # the previous InRelease continue to resolve its immutable by-hash objects;
  # a failed mutable stage is therefore safely rerunnable without deleting the
  # previous package/index set.
  for apt_metadata in "${APT_MUTABLE_METADATA_PATHS[@]}"; do
    upload_mutable_metadata "$apt_metadata"
  done
  upload_mutable_metadata "$APT_DISCOVERY_PATH"
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
