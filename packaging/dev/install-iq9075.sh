#!/usr/bin/env bash
set -euo pipefail

readonly MIN_AVAILABLE_KIB=$((4 * 1024 * 1024))
readonly IDENTITY_PATH="/etc/nuv-agent/device-identity.json"
readonly CONFIG_PATH="/etc/nuv-agent/agent.env"

usage() {
  echo "Usage: $0 /path/to/nuv-agent_<version>_arm64.deb [--camera oak|uvc]" >&2
}

die() {
  echo "[iq9075-install] ERROR: $*" >&2
  exit 1
}

read_device_tree() {
  local path
  for path in /sys/firmware/devicetree/base/model /sys/firmware/devicetree/base/compatible; do
    if [ -r "$path" ]; then
      tr '\0' '\n' < "$path"
    fi
  done
}

if [ "$#" -lt 1 ]; then
  usage
  exit 2
fi

deb_argument="$1"
shift
camera_mode="oak"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --camera)
      [ "$#" -ge 2 ] || { usage; exit 2; }
      camera_mode="$2"
      shift 2
      ;;
    --camera=*)
      camera_mode="${1#*=}"
      shift
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done
case "$camera_mode" in
  oak|uvc) ;;
  *) die "--camera must be oak or uvc" ;;
esac

command -v dpkg-deb >/dev/null 2>&1 || die "dpkg-deb is required"
command -v apt-get >/dev/null 2>&1 || die "apt-get is required"
command -v python3 >/dev/null 2>&1 || die "python3 is required"
command -v sudo >/dev/null 2>&1 || die "sudo is required"

deb_path="$(realpath "$deb_argument")"
[ -f "$deb_path" ] || die "package not found: $deb_path"
[ "$(dpkg --print-architecture)" = "arm64" ] || die "only Ubuntu arm64 is supported"

# shellcheck disable=SC1091
. /etc/os-release
[ "${ID:-}" = "ubuntu" ] || die "only Ubuntu is supported (found ${ID:-unknown})"

device_tree="$(read_device_tree | tr '[:upper:]' '[:lower:]')"
case "$device_tree" in
  *qcs9075*|*iq-9075*|*"iq 9075"*) ;;
  *) die "device tree is not an IQ-9075/QCS9075 board" ;;
esac

[ "$(dpkg-deb -f "$deb_path" Package)" = "nuv-agent" ] || die "package name must be nuv-agent"
[ "$(dpkg-deb -f "$deb_path" Architecture)" = "arm64" ] || die "package architecture must be arm64"

available_kib="$(df -Pk / | awk 'NR == 2 {print $4}')"
case "$available_kib" in
  ''|*[!0-9]*) die "could not determine root filesystem capacity" ;;
esac
[ "$available_kib" -ge "$MIN_AVAILABLE_KIB" ] || die "at least 4 GiB free space is required"

sudo -v
if sudo test -e "$IDENTITY_PATH"; then
  sudo python3 - "$IDENTITY_PATH" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected = {
    "productModel": "IQ9075_DEV",
    "hardwareRevision": "QCS9075-EVK",
    "platformProfile": "iq9075_dev",
}
try:
    actual = json.loads(path.read_text(encoding="utf-8"))
except Exception as exc:
    raise SystemExit(f"existing identity is unreadable: {exc}")
if actual != expected:
    raise SystemExit(f"existing identity conflicts with IQ-9075 development identity: {actual!r}")
PY
fi

sudo systemctl mask nuv-agent.service >/dev/null 2>&1 || true
cleanup_mask() {
  sudo systemctl unmask nuv-agent.service >/dev/null 2>&1 || true
  sudo systemctl disable --now nuv-agent.service >/dev/null 2>&1 || true
}
trap cleanup_mask EXIT

echo "[iq9075-install] installing $(basename "$deb_path") with base dependencies and autostart disabled"
sudo env \
  NUVION_INSTALL_PROFILE=base \
  NUVION_INSTALL_AUTOSTART=false \
  DEBIAN_FRONTEND=noninteractive \
  apt-get install -y --reinstall --no-install-recommends "$deb_path"

if ! sudo test -e "$IDENTITY_PATH"; then
  identity_tmp="$(mktemp)"
  trap 'rm -f "$identity_tmp"; cleanup_mask' EXIT
  printf '%s\n' \
    '{' \
    '  "productModel": "IQ9075_DEV",' \
    '  "hardwareRevision": "QCS9075-EVK",' \
    '  "platformProfile": "iq9075_dev"' \
    '}' > "$identity_tmp"
  sudo install -o root -g nuvion -m 0640 "$identity_tmp" "$IDENTITY_PATH"
  rm -f "$identity_tmp"
  trap cleanup_mask EXIT
fi

sudo python3 - "$CONFIG_PATH" "$camera_mode" <<'PY'
import os
import pathlib
import tempfile
import sys

path = pathlib.Path(sys.argv[1])
camera_mode = sys.argv[2]
updates = {
    "NUVION_VIDEO_SOURCE": "oak" if camera_mode == "oak" else "auto",
    "NUVION_GST_SOURCE": "",
    "NUVION_DEMO_MODE": "false",
    "NUVION_CAMERA_PREFERENCE": "usb",
    "NUVION_DEPTHAI_DEVICE_ID": "",
    "NUVION_DEPTHAI_STARTUP_TIMEOUT_SEC": "15",
    "NUVION_DEPTHAI_READ_TIMEOUT_SEC": "2",
    "NUVION_DEPTHAI_MAX_CONSECUTIVE_TIMEOUTS": "3",
    "NUVION_ZERO_SHOT_ENABLED": "false",
    "NUVION_ZSAD_BACKEND": "none",
    "NUVION_RUNTIME_BOOTSTRAP_ENABLED": "false",
    "NUVION_HOMEBREW_AUTOINSTALL": "false",
    "NUVION_DOCKER_AUTOINSTALL": "false",
    "NUVION_DOCKER_AUTOSTART": "false",
    "NUVION_TRITON_AUTOSTART": "false",
    "NUVION_MODEL_AUTO_PULL_ON_SETUP": "false",
    "NUVION_MODEL_AUTO_PULL_ON_RUN": "false",
    "NUVION_FLEET_COMMAND_ENABLED": "false",
    "NUVION_FACE_TRACKING_ENABLED": "false",
    "NUVION_MOTOR_ENABLED": "false",
}
preserve_if_present = {
    "NUVION_DEPTHAI_DEVICE_ID",
    "NUVION_DEPTHAI_STARTUP_TIMEOUT_SEC",
    "NUVION_DEPTHAI_READ_TIMEOUT_SEC",
    "NUVION_DEPTHAI_MAX_CONSECUTIVE_TIMEOUTS",
}

lines = path.read_text(encoding="utf-8").splitlines()
seen = set()
rendered = []
for line in lines:
    stripped = line.lstrip()
    if stripped and not stripped.startswith("#") and "=" in stripped:
        key = stripped.split("=", 1)[0].strip()
        if key in updates:
            if key not in seen:
                rendered.append(
                    line if key in preserve_if_present else f"{key}={updates[key]}"
                )
                seen.add(key)
            continue
    rendered.append(line)
for key, value in updates.items():
    if key not in seen:
        rendered.append(f"{key}={value}")

fd, temporary = tempfile.mkstemp(prefix="agent.env.", dir=str(path.parent), text=True)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write("\n".join(rendered) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o660)
    os.chown(temporary, 0, path.stat().st_gid)
    os.replace(temporary, path)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
PY

sudo chown root:nuvion "$CONFIG_PATH"
sudo chmod 0660 "$CONFIG_PATH"
sudo chown root:nuvion "$IDENTITY_PATH"
sudo chmod 0640 "$IDENTITY_PATH"

cleanup_mask
trap - EXIT
sudo apt-get clean

echo "[iq9075-install] complete"
echo "[iq9075-install] camera mode: $camera_mode"
echo "[iq9075-install] service is disabled and inactive"
echo "[iq9075-install] run packaging/dev/test-iq9075.sh before adding device credentials or enabling the service"
