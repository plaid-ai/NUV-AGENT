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

timeout --signal=TERM --kill-after=5s 60s runuser -u nuvion -- \
  /usr/bin/env \
    "HOME=$runtime_dir/home" \
    "XDG_CACHE_HOME=$runtime_dir/cache" \
    "XDG_CONFIG_HOME=$runtime_dir/config" \
    "XDG_RUNTIME_DIR=$runtime_dir/runtime" \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    "$agent_python" -s - /etc/nuv-agent/agent.env <<'PY'
import gc
from importlib.metadata import version
from pathlib import Path
import re
import sys
import time

import depthai


if version("depthai") != "2.32.0.0":
    raise SystemExit("unexpected DepthAI runtime")


def configured_mxid(config_path):
    try:
        raw = Path(config_path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise SystemExit("cannot read fixed Agent configuration") from error
    matches = []
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != "NUVION_DEPTHAI_DEVICE_ID":
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        matches.append(value)
    if len(matches) > 1:
        raise SystemExit("duplicate NUVION_DEPTHAI_DEVICE_ID")
    value = matches[0] if matches else ""
    if value and not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", value):
        raise SystemExit("invalid NUVION_DEPTHAI_DEVICE_ID")
    return value or None


def select_device(devices, required_mxid):
    observed = {}
    for device_info in devices:
        mxid = str(device_info.getMxId()).strip()
        if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", mxid):
            raise RuntimeError("OAK returned an invalid MXID")
        if mxid in observed:
            raise RuntimeError("OAK returned duplicate MXIDs")
        observed[mxid] = device_info
    if len(observed) != 1:
        return None
    selected_mxid, selected = next(iter(observed.items()))
    if required_mxid is not None and required_mxid != selected_mxid:
        return None
    return selected


def physical_oak_paths():
    root = Path("/sys/bus/usb/devices")
    matches = []
    for candidate in root.iterdir():
        try:
            vendor = (candidate / "idVendor").read_text(encoding="utf-8").strip().lower()
            product = (candidate / "idProduct").read_text(encoding="utf-8").strip().lower()
        except OSError:
            continue
        if vendor == "03e7" and product in {"2485", "f63b"}:
            matches.append(candidate.name)
    if len(matches) != 1:
        return []
    if re.fullmatch(r"[12]-1(?:\.[1-9][0-9]*)+", matches[0]) is None:
        raise RuntimeError("the only OAK is outside the IQ9075 USB1 dual hub")
    return matches

def make_pipeline():
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
    return pipeline


# Stopping the production Agent closes XLink and re-enumerates OAK-D Lite from
# application PID 2485 to bootloader PID f63b (or the reverse). A one-shot open
# during that physical USB transition produced a false rollback on IQ9075.
# Require two consecutive enumerations and retry the complete Device lifecycle
# within one strict wall-clock deadline.
deadline = time.monotonic() + 45.0
required_mxid = configured_mxid(sys.argv[1])
stable_mxid = None
stable_polls = 0
last_failure = "no OAK device enumerated"
ready = False
while time.monotonic() < deadline and not ready:
    try:
        available = depthai.Device.getAllAvailableDevices()
    except Exception as error:  # Native XLink can fail while USB re-enumerates.
        available = []
        last_failure = type(error).__name__
    try:
        selected = select_device(available, required_mxid)
        physical_paths = physical_oak_paths()
    except Exception as error:
        selected = None
        physical_paths = []
        last_failure = type(error).__name__
    if selected is None or len(physical_paths) != 1:
        stable_mxid = None
        stable_polls = 0
        if len(available) > 1:
            last_failure = "IQ9075 requires exactly one attached OAK"
        elif available and required_mxid is not None:
            last_failure = "configured OAK MXID not enumerated"
        time.sleep(0.25)
        continue
    selected_mxid = str(selected.getMxId()).strip()
    if selected_mxid == stable_mxid:
        stable_polls += 1
    else:
        stable_mxid = selected_mxid
        stable_polls = 1
    if stable_polls < 2:
        time.sleep(0.25)
        continue
    try:
        pipeline = make_pipeline()
        with depthai.Device(pipeline, selected) as device:
            queue = device.getOutputQueue(
                name="nuvion_oak_readiness",
                maxSize=1,
                blocking=False,
            )
            frame_deadline = min(deadline, time.monotonic() + 10.0)
            packet = None
            while packet is None and time.monotonic() < frame_deadline:
                packet = queue.tryGet()
                if packet is None:
                    time.sleep(0.01)
            if packet is None:
                raise RuntimeError("OAK readiness frame timeout")
            frame = packet.getCvFrame()
            if getattr(frame, "shape", None) != (480, 640, 3):
                raise RuntimeError(
                    f"unexpected OAK frame shape: {getattr(frame, 'shape', None)}"
                )
            del frame
            del packet
            del queue
        ready = True
    except Exception as error:  # Retry only inside the bounded physical probe.
        last_failure = type(error).__name__
        stable_mxid = None
        stable_polls = 0
        time.sleep(0.5)
    finally:
        if "pipeline" in locals():
            del pipeline
        gc.collect()

if not ready:
    raise SystemExit(f"OAK readiness failed after bounded USB settle: {last_failure}")

print("[iq9075-oak-ready] OAK RGB frame readiness: PASS")
PY
