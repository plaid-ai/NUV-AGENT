#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PKG_NAME="nuv-agent"
VERSION="${VERSION:-0.1.91}"
ARCH="${ARCH:-$(dpkg --print-architecture)}"
BUILD_ROOT="${BUILD_ROOT:-$(mktemp -d)}"

PKG_DIR="$BUILD_ROOT/${PKG_NAME}_${VERSION}_${ARCH}"

mkdir -p "$PKG_DIR/DEBIAN" \
         "$PKG_DIR/opt/nuv-agent" \
         "$PKG_DIR/usr/bin" \
         "$PKG_DIR/lib/systemd/system" \
         "$PKG_DIR/etc/nuv-agent" \
         "$PKG_DIR/opt/nuv-agent/share"

cat > "$PKG_DIR/DEBIAN/control" <<CONTROL
Package: ${PKG_NAME}
Version: ${VERSION}
Section: utils
Priority: optional
Architecture: ${ARCH}
Maintainer: Nuvion <ops@nuvion.ai>
Depends: python3 (>= 3.10), python3-venv, python3-pip, python3-gi, curl | wget, ffmpeg, gstreamer1.0-tools, gstreamer1.0-plugins-base, gstreamer1.0-plugins-good, gstreamer1.0-plugins-bad, gstreamer1.0-plugins-ugly, gstreamer1.0-libav, gir1.2-gstreamer-1.0, gir1.2-gst-plugins-base-1.0
Description: Nuvion on-device agent
CONTROL

cp "$ROOT_DIR/packaging/deb/postinst" "$PKG_DIR/DEBIAN/postinst"
cp "$ROOT_DIR/packaging/deb/prerm" "$PKG_DIR/DEBIAN/prerm"
chmod 0755 "$PKG_DIR/DEBIAN/postinst" "$PKG_DIR/DEBIAN/prerm"

SRC_DIR="$PKG_DIR/opt/nuv-agent/src"
mkdir -p "$SRC_DIR"
rsync -a \
  --exclude ".git" \
  --exclude "dist" \
  --exclude ".venv*" \
  --exclude "__pycache__" \
  --exclude ".pytest_cache" \
  "$ROOT_DIR/" \
  "$SRC_DIR/"

cp "$ROOT_DIR/nuvion_app/config_template.env" "$PKG_DIR/opt/nuv-agent/share/agent.env.example"
cp "$ROOT_DIR/packaging/systemd/nuv-agent.service" "$PKG_DIR/lib/systemd/system/nuv-agent.service"

cat > "$PKG_DIR/usr/bin/nuv-agent" <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1

prepend_path() {
  local key="$1"
  local value="$2"
  local current="${!key-}"
  if [ -z "$value" ]; then
    return 0
  fi
  case ":$current:" in
    *":$value:"*) ;;
    *)
      if [ -n "$current" ]; then
        export "$key=$value:$current"
      else
        export "$key=$value"
      fi
      ;;
  esac
}

for prefix in /opt/homebrew /usr/local; do
  [ -d "$prefix/lib" ] && prepend_path DYLD_FALLBACK_LIBRARY_PATH "$prefix/lib"
  [ -d "$prefix/lib/girepository-1.0" ] && prepend_path GI_TYPELIB_PATH "$prefix/lib/girepository-1.0"
  [ -d "$prefix/lib/gstreamer-1.0" ] && prepend_path GST_PLUGIN_PATH "$prefix/lib/gstreamer-1.0"
  [ -d "$prefix/opt/libnice-gstreamer/libexec/gstreamer-1.0" ] && prepend_path GST_PLUGIN_PATH "$prefix/opt/libnice-gstreamer/libexec/gstreamer-1.0"
  if [ -z "${GST_PLUGIN_SCANNER:-}" ] && [ -x "$prefix/opt/gstreamer/libexec/gstreamer-1.0/gst-plugin-scanner" ]; then
    export GST_PLUGIN_SCANNER="$prefix/opt/gstreamer/libexec/gstreamer-1.0/gst-plugin-scanner"
  fi
done

exec /opt/nuv-agent/venv/bin/python -s -m nuvion_app.cli "$@"
SCRIPT
chmod 0755 "$PKG_DIR/usr/bin/nuv-agent"

chmod 0644 "$PKG_DIR/lib/systemd/system/nuv-agent.service"

OUTPUT_DEB="${OUTPUT_DEB:-${ROOT_DIR}/dist/${PKG_NAME}_${VERSION}_${ARCH}.deb}"
mkdir -p "$(dirname "$OUTPUT_DEB")"

if command -v dpkg-deb >/dev/null 2>&1; then
  dpkg-deb --build "$PKG_DIR" "$OUTPUT_DEB"
  echo "Built: $OUTPUT_DEB"
else
  echo "dpkg-deb not found. Install dpkg-dev." >&2
  exit 1
fi
