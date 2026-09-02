#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 /path/to/file.json bucket object-name" >&2
  exit 2
fi

source_argument="$1"
bucket="$2"
object_name="$3"

[[ "$bucket" =~ ^[a-z0-9][a-z0-9.-]{1,221}[a-z0-9]$ ]] \
  || { echo "invalid GCS bucket" >&2; exit 2; }
if [[ ! "$object_name" =~ ^releases/reservations/iq9075/[1-9][0-9]*\.json$ ]] \
  && [[ ! "$object_name" =~ ^releases/promotions/iq9075/[0-9]+\.[0-9]+\.[0-9]+\.json$ ]]; then
  echo "GCS object is outside the fixed release reservation/promotion prefixes" >&2
  exit 2
fi
[[ "$source_argument" = /* ]] || source_argument="$PWD/$source_argument"
source_path="$(realpath "$source_argument")"
[ -f "$source_path" ] && [ ! -L "$source_argument" ] \
  || { echo "immutable source must be a regular non-symlink file" >&2; exit 2; }
[ -s "$source_path" ] || { echo "immutable source must not be empty" >&2; exit 2; }

remote_path="gs://$bucket/$object_name"
# Cloud Storage's generation-match=0 is the actual atomic create-only CAS. A
# concurrent or idempotent writer receives a precondition failure, after which
# only byte-identical remote content is accepted.
if ! gcloud storage cp \
  --if-generation-match=0 \
  --cache-control="no-cache, max-age=0" \
  "$source_path" "$remote_path"; then
  if ! gcloud storage cat "$remote_path" | cmp -s - "$source_path"; then
    echo "refusing to replace an existing immutable object: $remote_path" >&2
    exit 1
  fi
  echo "Immutable object already matches after create-only CAS: $remote_path"
fi

if ! gcloud storage cat "$remote_path" | cmp -s - "$source_path"; then
  echo "immutable object verification failed: $remote_path" >&2
  exit 1
fi
echo "Verified immutable object: $remote_path"
