#!/usr/bin/env bash
set -euo pipefail

die() {
  echo "[iq9075-oak-ready] ERROR: $*" >&2
  exit 1
}

[ "$#" -eq 0 ] || die "this fixed readiness probe accepts no arguments"
[ "$(id -u)" -eq 0 ] || die "the readiness probe must run as root"
for command in chown id install mktemp runuser timeout; do
  command -v "$command" >/dev/null 2>&1 || die "$command is required"
done
[ -x /usr/bin/python3 ] || die "/usr/bin/python3 is required"

agent_python=/opt/nuv-agent/current/venv/bin/python
set +e
validated_python="$(/usr/bin/python3 -I - "$agent_python" <<'PY'
import os
import stat
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_absolute() or os.path.normpath(str(path)) != str(path):
    raise SystemExit(2)
metadata = path.stat()
parent = path.parent.resolve(strict=True)
install_root = Path("/opt/nuv-agent").resolve(strict=True)
if (
    path.name != "python"
    or not stat.S_ISREG(metadata.st_mode)
    or metadata.st_uid != 0
    or metadata.st_mode & 0o022
    or not os.access(path, os.X_OK)
    or not parent.is_relative_to(install_root)
):
    raise SystemExit(2)
print(path)
PY
)"
validation_status=$?
set -e
[ "$validation_status" -eq 0 ] && [ "$validated_python" = "$agent_python" ] || \
  die "active slot Python is not a fixed root-owned executable"

runtime_dir="$(mktemp -d /tmp/nuvion-iq9075-oak-ready.XXXXXX)"
case "$runtime_dir" in
  /tmp/nuvion-iq9075-oak-ready.*) ;;
  *) die "unsafe readiness runtime directory" ;;
esac
cleanup() {
  case "$runtime_dir" in
    /tmp/nuvion-iq9075-oak-ready.*)
      [ ! -L "$runtime_dir" ] && rm -rf -- "$runtime_dir"
      ;;
  esac
}
trap cleanup EXIT
chown nuvion:nuvion "$runtime_dir"
for directory in home cache config runtime; do
  install -d -m 0700 -o nuvion -g nuvion "$runtime_dir/$directory"
done

timeout 30s runuser -u nuvion -- \
  /usr/bin/env \
    "HOME=$runtime_dir/home" \
    "XDG_CACHE_HOME=$runtime_dir/cache" \
    "XDG_CONFIG_HOME=$runtime_dir/config" \
    "XDG_RUNTIME_DIR=$runtime_dir/runtime" \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    "$agent_python" -s - <<'PY'
from importlib.metadata import version
import time

import depthai


if version("depthai") != "2.32.0.0":
    raise SystemExit("unexpected DepthAI runtime")

pipeline = depthai.Pipeline()
camera = pipeline.create(depthai.node.ColorCamera)
camera.setPreviewSize(640, 480)
camera.setInterleaved(False)
camera.setFps(30.0)
camera.setColorOrder(depthai.ColorCameraProperties.ColorOrder.RGB)
if hasattr(camera, "setBoardSocket") and hasattr(depthai.CameraBoardSocket, "RGB"):
    camera.setBoardSocket(depthai.CameraBoardSocket.RGB)
output = pipeline.create(depthai.node.XLinkOut)
output.setStreamName("nuvion_oak_readiness")
camera.preview.link(output.input)

with depthai.Device(pipeline) as device:
    queue = device.getOutputQueue(
        name="nuvion_oak_readiness",
        maxSize=1,
        blocking=False,
    )
    deadline = time.monotonic() + 15.0
    packet = None
    while packet is None and time.monotonic() < deadline:
        packet = queue.tryGet()
        if packet is None:
            time.sleep(0.01)
    if packet is None:
        raise SystemExit("OAK readiness frame timeout")
    frame = packet.getCvFrame()
    if getattr(frame, "shape", None) != (480, 640, 3):
        raise SystemExit(f"unexpected OAK frame shape: {getattr(frame, 'shape', None)}")
    del frame
    del packet

print("[iq9075-oak-ready] OAK RGB frame readiness: PASS")
PY
