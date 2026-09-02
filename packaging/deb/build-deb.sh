#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PKG_NAME="nuv-agent"
VERSION="${VERSION:-0.1.121}"
ARCH="${ARCH:-arm64}"
BUILD_ROOT="${BUILD_ROOT:-}"
SOURCE_EPOCH="${SOURCE_DATE_EPOCH:-}"
EXPECTED_COMPONENT_SHA="${COMPONENT_SHA:-}"
readonly DEB_BUILDER_IMAGE="ubuntu@sha256:95fa486768020359141f1318720f43e7982ef926c792891d984aef9aaf05e7ea"

if [ "$ARCH" != "arm64" ]; then
  echo "The immutable bootstrap package is currently released only for arm64" >&2
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

if [ "${NUVION_DEB_INNER:-0}" != "1" ]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required for the digest-pinned arm64 deb build" >&2
    exit 1
  fi
  case "$(uname -m)" in
    arm64|aarch64) ;;
    *) echo "deb build requires a native arm64 Docker host" >&2; exit 1 ;;
  esac
  bundle_path="${BOOTSTRAP_BUNDLE_PATH:-}"
  if [ -z "$bundle_path" ]; then
    echo "BOOTSTRAP_BUNDLE_PATH is required; build it in the keyless bundle job" >&2
    exit 1
  fi
  bundle_path="$(realpath "$bundle_path")"
  case "$bundle_path" in
    "$ROOT_DIR"/*) ;;
    *) echo "BOOTSTRAP_BUNDLE_PATH must be inside the release checkout" >&2; exit 1 ;;
  esac
  if [ -L "$bundle_path" ] || [ ! -f "$bundle_path" ]; then
    echo "BOOTSTRAP_BUNDLE_PATH must be a regular non-symlink file" >&2
    exit 1
  fi
  output_path="${OUTPUT_DEB:-${ROOT_DIR}/dist/${PKG_NAME}_${VERSION}_${ARCH}.deb}"
  mkdir -p "$(dirname "$output_path")"
  output_directory="$(cd "$(dirname "$output_path")" && pwd -P)"
  output_name="$(basename "$output_path")"
  bundle_relative="${bundle_path#"$ROOT_DIR"/}"
  docker run --rm \
    --platform linux/arm64 \
    --user "$(id -u):$(id -g)" \
    --network none \
    --read-only \
    --tmpfs /tmp:rw,noexec,nosuid,nodev,size=4g \
    --mount "type=bind,src=$ROOT_DIR,dst=/src,readonly" \
    --mount "type=bind,src=$output_directory,dst=/out" \
    --env NUVION_DEB_INNER=1 \
    --env "VERSION=$VERSION" \
    --env ARCH=arm64 \
    --env "SOURCE_DATE_EPOCH=$SOURCE_EPOCH" \
    --env "COMPONENT_SHA=$EXPECTED_COMPONENT_SHA" \
    --env "BOOTSTRAP_BUNDLE_PATH=/src/$bundle_relative" \
    --env BUILD_ROOT=/tmp/deb-build \
    --env "OUTPUT_DEB=/out/$output_name" \
    "$DEB_BUILDER_IMAGE" \
    /src/packaging/deb/build-deb.sh
  exit $?
fi

if [ "$(dpkg --print-architecture)" != "arm64" ]; then
  echo "Pinned deb builder architecture is not arm64" >&2
  exit 1
fi
BUILD_ROOT="${BUILD_ROOT:-$(mktemp -d)}"
PKG_DIR="$BUILD_ROOT/${PKG_NAME}_${VERSION}_${ARCH}"

mkdir -p "$PKG_DIR/DEBIAN" \
         "$PKG_DIR/opt/nuv-agent" \
         "$PKG_DIR/usr/bin" \
         "$PKG_DIR/usr/local/libexec/nuvion" \
         "$PKG_DIR/usr/lib/udev/rules.d" \
         "$PKG_DIR/lib/systemd/system" \
         "$PKG_DIR/etc/nuv-agent" \
         "$PKG_DIR/etc/nuvion-updater" \
         "$PKG_DIR/opt/nuv-agent/share" \
         "$PKG_DIR/usr/lib/nuvion-updater"

cat > "$PKG_DIR/DEBIAN/control" <<CONTROL
Package: ${PKG_NAME}
Version: ${VERSION}
Section: utils
Priority: optional
Architecture: ${ARCH}
Maintainer: Nuvion <ops@nuvion.ai>
Depends: python3 (>= 3.12), python3 (<< 3.13), python3-cryptography, python3-gi, curl | wget, ffmpeg, psmisc, util-linux, udev, libusb-1.0-0, v4l-utils, gstreamer1.0-tools, gstreamer1.0-nice, gstreamer1.0-plugins-base, gstreamer1.0-plugins-good, gstreamer1.0-plugins-bad, gstreamer1.0-plugins-ugly, gstreamer1.0-libav, gir1.2-gstreamer-1.0, gir1.2-gst-plugins-base-1.0, gir1.2-gst-plugins-bad-1.0
Description: Nuvion on-device agent
CONTROL

cp "$ROOT_DIR/packaging/deb/postinst" "$PKG_DIR/DEBIAN/postinst"
cp "$ROOT_DIR/packaging/deb/prerm" "$PKG_DIR/DEBIAN/prerm"
cp "$ROOT_DIR/packaging/deb/postrm" "$PKG_DIR/DEBIAN/postrm"
cp "$ROOT_DIR/packaging/deb/conffiles" "$PKG_DIR/DEBIAN/conffiles"
chmod 0755 \
  "$PKG_DIR/DEBIAN/postinst" \
  "$PKG_DIR/DEBIAN/prerm" \
  "$PKG_DIR/DEBIAN/postrm"
chmod 0644 "$PKG_DIR/DEBIAN/conffiles"

# The root updater is installed outside the swappable Agent slot so an Agent
# release can never replace its own verifier or rollback implementation.
mkdir -p "$PKG_DIR/usr/lib/nuvion-updater/nuvion_app"
cp -a "$ROOT_DIR/nuvion_updater" "$PKG_DIR/usr/lib/nuvion-updater/"
cp -a "$ROOT_DIR/nuvion_app/." "$PKG_DIR/usr/lib/nuvion-updater/nuvion_app/"
find "$PKG_DIR/usr/lib/nuvion-updater" -type f \
  \( -name '*.pyc' -o -name '*.pyo' \) -delete
find "$PKG_DIR/usr/lib/nuvion-updater" -depth -type d \
  -name __pycache__ -exec rm -rf -- {} +
install -m 0755 \
  "$ROOT_DIR/packaging/dev/test-iq9075.sh" \
  "$PKG_DIR/usr/lib/nuvion-updater/test-iq9075.sh"
install -m 0755 \
  "$ROOT_DIR/packaging/dev/iq9075-board-e2e.py" \
  "$PKG_DIR/usr/local/libexec/nuvion/iq9075-board-e2e.py"

cp "$ROOT_DIR/nuvion_app/config_template.env" "$PKG_DIR/opt/nuv-agent/share/agent.env.example"
bundle_path="${BOOTSTRAP_BUNDLE_PATH:-}"
if [ -L "$bundle_path" ] || [ ! -f "$bundle_path" ]; then
  echo "BOOTSTRAP_BUNDLE_PATH must be a regular non-symlink file" >&2
  exit 1
fi
bundle_digest="$(sha256sum "$bundle_path" | awk '{print $1}')"
if [[ ! "$bundle_digest" =~ ^[0-9a-f]{64}$ ]]; then
  echo "Cannot derive bootstrap bundle SHA-256" >&2
  exit 1
fi
install -m 0644 \
  "$bundle_path" \
  "$PKG_DIR/opt/nuv-agent/share/bootstrap-agent-bundle.tar.gz"
printf '%s  %s\n' \
  "$bundle_digest" "bootstrap-agent-bundle.tar.gz" \
  > "$PKG_DIR/opt/nuv-agent/share/bootstrap-agent-bundle.sha256"
chmod 0644 "$PKG_DIR/opt/nuv-agent/share/bootstrap-agent-bundle.sha256"
install -m 0644 \
  "$ROOT_DIR/packaging/release/requirements-agent-bundle-arm64.txt" \
  "$PKG_DIR/opt/nuv-agent/share/requirements-agent-bundle-arm64.txt"
install -m 0644 \
  "$ROOT_DIR/packaging/deb/requirements-depthai-arm64.txt" \
  "$PKG_DIR/opt/nuv-agent/share/requirements-depthai-arm64.txt"
install -m 0755 \
  "$ROOT_DIR/packaging/systemd/nuv-agent-slot-entrypoint" \
  "$PKG_DIR/opt/nuv-agent/share/nuv-agent-slot-entrypoint"
install -m 0644 \
  "$ROOT_DIR/packaging/udev/80-movidius.rules" \
  "$PKG_DIR/usr/lib/udev/rules.d/80-movidius.rules"
cp "$ROOT_DIR/packaging/systemd/nuv-agent.service" "$PKG_DIR/lib/systemd/system/nuv-agent.service"
cp "$ROOT_DIR/packaging/systemd/nuv-agent-updater.service" "$PKG_DIR/lib/systemd/system/nuv-agent-updater.service"
cp "$ROOT_DIR/packaging/systemd/nuv-agent-updater.socket" "$PKG_DIR/lib/systemd/system/nuv-agent-updater.socket"
install -m 0644 \
  "$ROOT_DIR/packaging/systemd/updater.env.example" \
  "$PKG_DIR/etc/nuvion-updater/updater.env"

install -m 0755 \
  "$ROOT_DIR/packaging/systemd/nuv-agent-current" \
  "$PKG_DIR/usr/bin/nuv-agent"

chmod 0644 \
  "$PKG_DIR/lib/systemd/system/nuv-agent.service" \
  "$PKG_DIR/lib/systemd/system/nuv-agent-updater.service" \
  "$PKG_DIR/lib/systemd/system/nuv-agent-updater.socket"

OUTPUT_DEB="${OUTPUT_DEB:-${ROOT_DIR}/dist/${PKG_NAME}_${VERSION}_${ARCH}.deb}"
mkdir -p "$(dirname "$OUTPUT_DEB")"
find "$PKG_DIR" -exec touch -h -d "@$SOURCE_EPOCH" {} +

if command -v dpkg-deb >/dev/null 2>&1; then
  # GitHub-hosted builders are unprivileged. Normalize every archive member to
  # root ownership so runner UID/GID can never become a target-device trust
  # principal for updater code or systemd units.
  dpkg-deb --root-owner-group --build "$PKG_DIR" "$OUTPUT_DEB"
  echo "Built: $OUTPUT_DEB"
else
  echo "dpkg-deb not found. Install dpkg-dev." >&2
  exit 1
fi
