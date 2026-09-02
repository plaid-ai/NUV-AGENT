#!/usr/bin/env bash
set -euo pipefail

readonly CONFIG_PATH="/etc/nuv-agent/agent.env"

usage() {
  echo "Usage: sudo $0 <device-credentials.json> [--synthetic-camera] [--consume]" >&2
}

die() {
  echo "[iq9075-provision] ERROR: $*" >&2
  exit 1
}

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
  usage
  exit 2
fi
if [ "$(id -u)" -ne 0 ]; then
  die "run this command with sudo"
fi

credentials_path="$1"
shift
synthetic_camera=false
consume=false
for option in "$@"; do
  case "$option" in
    --synthetic-camera) synthetic_camera=true ;;
    --consume) consume=true ;;
    *) usage; exit 2 ;;
  esac
done

[ ! -L "$credentials_path" ] || die "credential file must not be a symlink"
[ -f "$credentials_path" ] || die "credential file not found"
[ -f "$CONFIG_PATH" ] || die "nuv-agent config not found"

python3 - "$credentials_path" "$CONFIG_PATH" "$synthetic_camera" "$consume" <<'PY'
import json
import os
import pathlib
import re
import stat
import sys
import tempfile

credential_path = pathlib.Path(sys.argv[1])
config_path = pathlib.Path(sys.argv[2])
synthetic_camera = sys.argv[3] == "true"
consume = sys.argv[4] == "true"

metadata = credential_path.lstat()
if not stat.S_ISREG(metadata.st_mode):
    raise SystemExit("credential input must be a regular file")
if metadata.st_mode & 0o077:
    raise SystemExit("credential input must not be accessible by group or other")
if metadata.st_size > 64 * 1024:
    raise SystemExit("credential input exceeds 64 KiB")

flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
descriptor = os.open(credential_path, flags)
try:
    opened = os.fstat(descriptor)
    if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
        raise SystemExit("credential input changed while opening")
    raw = bytearray()
    while len(raw) <= 64 * 1024:
        chunk = os.read(descriptor, min(8192, 64 * 1024 + 1 - len(raw)))
        if not chunk:
            break
        raw.extend(chunk)
finally:
    os.close(descriptor)
if len(raw) > 64 * 1024:
    raise SystemExit("credential input exceeds 64 KiB")
payload = json.loads(bytes(raw).decode("utf-8"))
space_id = payload.get("spaceId")
username = str(payload.get("deviceUsername") or "").strip()
password = str(payload.get("devicePassword") or "")
if isinstance(space_id, bool) or not isinstance(space_id, int) or space_id < 1:
    raise SystemExit("spaceId must be a positive integer")
match = re.fullmatch(r"sp-([1-9][0-9]*)-nuvion-[a-z0-9]+", username)
if match is None or int(match.group(1)) != space_id:
    raise SystemExit("deviceUsername does not match spaceId")
if len(password.encode("utf-8")) < 12:
    raise SystemExit("devicePassword is missing or too short")

updates = {
    "NUVION_SERVER_BASE_URL": "https://api.nuvion-dev.plaidlabs.ai",
    "NUVION_MODEL_SERVER_BASE_URL": "https://api.nuvion-dev.plaidlabs.ai",
    "NUVION_DEVICE_USERNAME": username,
    "NUVION_DEVICE_PASSWORD": password,
    "NUVION_SPACE_ID": str(space_id),
    "NUVION_DEVICE_ID": username,
    "NUVION_FLEET_COMMAND_ENABLED": "false",
    "NUVION_GST_SOURCE": "",
    "NUVION_DEMO_MODE": "false",
}
if synthetic_camera:
    updates["NUVION_GST_SOURCE"] = (
        "videotestsrc is-live=true pattern=smpte ! "
        "video/x-raw,width=640,height=480,framerate=30/1 ! "
        "videoconvert ! video/x-raw,format=RGB"
    )

lines = config_path.read_text(encoding="utf-8").splitlines()
seen = set()
rendered = []
for line in lines:
    stripped = line.lstrip()
    if stripped and not stripped.startswith("#") and "=" in stripped:
        key = stripped.split("=", 1)[0].strip()
        if key in updates:
            if key not in seen:
                rendered.append(f"{key}={updates[key]}")
                seen.add(key)
            continue
    rendered.append(line)
for key, value in updates.items():
    if key not in seen:
        rendered.append(f"{key}={value}")

fd, temporary = tempfile.mkstemp(prefix="agent.env.", dir=str(config_path.parent))
try:
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write("\n".join(rendered) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o660)
    os.chown(temporary, 0, config_path.stat().st_gid)
    os.replace(temporary, config_path)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)

if consume:
    current = credential_path.lstat()
    if (
        not stat.S_ISREG(current.st_mode)
        or (current.st_dev, current.st_ino) != (metadata.st_dev, metadata.st_ino)
    ):
        raise SystemExit("credential input changed before secure consume")
    credential_path.unlink()
PY

chown root:nuvion "$CONFIG_PATH"
chmod 0660 "$CONFIG_PATH"

echo "[iq9075-provision] device credentials installed without displaying secrets"
if [ "$synthetic_camera" = true ]; then
  echo "[iq9075-provision] synthetic camera source enabled for control-plane E2E"
fi
echo "[iq9075-provision] Fleet command issuance remains disabled"
