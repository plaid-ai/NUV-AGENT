#!/usr/bin/env bash
set -euo pipefail

allow_no_camera=false
camera_mode="oak"
evidence_output=""
expected_version=""
expected_component_sha=""
expected_bom_digest=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --allow-no-camera)
      allow_no_camera=true
      shift
      ;;
    --camera)
      [ "$#" -ge 2 ] || {
        echo "Usage: $0 [--camera oak|uvc] [--allow-no-camera]" >&2
        exit 2
      }
      camera_mode="$2"
      shift 2
      ;;
    --camera=*)
      camera_mode="${1#*=}"
      shift
      ;;
    --evidence-output)
      [ "$#" -ge 2 ] || {
        echo "Usage: $0 [--camera oak|uvc] [--allow-no-camera] [--evidence-output PATH]" >&2
        exit 2
      }
      evidence_output="$2"
      shift 2
      ;;
    --evidence-output=*)
      evidence_output="${1#*=}"
      shift
      ;;
    --expected-version)
      [ "$#" -ge 2 ] || exit 2
      expected_version="$2"
      shift 2
      ;;
    --expected-component-sha)
      [ "$#" -ge 2 ] || exit 2
      expected_component_sha="$2"
      shift 2
      ;;
    --expected-bom-digest)
      [ "$#" -ge 2 ] || exit 2
      expected_bom_digest="$2"
      shift 2
      ;;
    *)
      echo "Usage: $0 [--camera oak|uvc] [--allow-no-camera] [--evidence-output PATH]" >&2
      exit 2
      ;;
  esac
done
case "$camera_mode" in
  oak|uvc) ;;
  *)
    echo "[iq9075-e2e] ERROR: --camera must be oak or uvc" >&2
    exit 2
    ;;
esac
die() {
  echo "[iq9075-e2e] ERROR: $*" >&2
  exit 1
}

[ -z "$evidence_output" ] || [ "$camera_mode" = "oak" ] || \
  die "--evidence-output requires --camera oak"
if [ -n "$evidence_output" ]; then
  [[ "$expected_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || \
    die "--expected-version is required for evidence mode"
  [[ "$expected_component_sha" =~ ^[0-9a-f]{40}$ ]] || \
    die "--expected-component-sha is required for evidence mode"
  [[ "$expected_bom_digest" =~ ^sha256:[0-9a-f]{64}$ ]] || \
    die "--expected-bom-digest is required for evidence mode"
  [ -z "${PYTHONPATH:-}" ] || die "PYTHONPATH is forbidden in evidence mode"
fi

for command in gst-inspect-1.0 gst-launch-1.0 v4l2-ctl timeout python3 readlink \
  mktemp install id; do
  command -v "$command" >/dev/null 2>&1 || die "$command is required"
done
[ -x /usr/bin/python3 ] || die "/usr/bin/python3 is required"

echo "[iq9075-e2e] checking GStreamer runtime"
for element in v4l2src videoconvert jpegdec x264enc h264parse rtph264pay webrtcbin nicesrc; do
  gst-inspect-1.0 "$element" >/dev/null 2>&1 || die "missing GStreamer element: $element"
done

/usr/bin/python3 -I <<'PY'
import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstWebRTC", "1.0")
gi.require_version("GstSdp", "1.0")
from gi.repository import Gst  # noqa: E402

Gst.init(None)
pipeline = Gst.parse_launch(
    "videotestsrc num-buffers=1 ! "
    "video/x-raw,width=640,height=480,framerate=30/1 ! videoconvert ! "
    "x264enc tune=zerolatency speed-preset=ultrafast ! h264parse ! "
    "rtph264pay config-interval=-1 pt=96 ! "
    "application/x-rtp,media=video,encoding-name=H264,payload=96 ! webrtcbin name=pc"
)
pipeline.set_state(Gst.State.NULL)
print("[iq9075-e2e] Gst/GstWebRTC/GstSdp imports and WebRTC pipeline parse: PASS")
PY

echo "[iq9075-e2e] running bounded synthetic H264/RTP pipeline"
timeout 20s gst-launch-1.0 -q \
  videotestsrc num-buffers=30 ! \
  video/x-raw,width=640,height=480,framerate=30/1 ! \
  videoconvert ! x264enc tune=zerolatency speed-preset=ultrafast ! \
  h264parse ! rtph264pay pt=96 ! fakesink sync=false

default_agent_python=/opt/nuv-agent/current/venv/bin/python
requested_agent_python="${NUVION_AGENT_PYTHON:-$default_agent_python}"
if [ -n "$evidence_output" ]; then
  expected_agent_python="/opt/nuv-agent/releases/${expected_bom_digest#sha256:}/venv/bin/python"
  [ "$requested_agent_python" = "$expected_agent_python" ] || \
    die "evidence mode requires the exact BOM-addressed candidate Python"
fi
set +e
agent_python="$(/usr/bin/python3 -I - "$requested_agent_python" <<'PY'
import os
import stat
import sys
from pathlib import Path

raw = sys.argv[1]
path = Path(raw)
normalized = os.path.normpath(raw)
install_root = Path("/opt/nuv-agent").resolve(strict=True)
if (
    not path.is_absolute()
    or raw != normalized
    or not raw.startswith("/opt/nuv-agent/")
    or path.name != "python"
):
    raise SystemExit(2)
metadata = path.stat()
parent = path.parent.resolve(strict=True)
if (
    not stat.S_ISREG(metadata.st_mode)
    or metadata.st_uid != 0
    or metadata.st_mode & 0o022
    or not os.access(path, os.X_OK)
    or not parent.is_relative_to(install_root)
):
    raise SystemExit(2)
print(path)
PY
)"
python_status=$?
set -e
[ "$python_status" -eq 0 ] && [ -n "$agent_python" ] || \
  die "NUVION_AGENT_PYTHON must be a normalized root-owned executable /opt/nuv-agent/.../bin/python"
if [ "$agent_python" != "$default_agent_python" ] || [ -n "${PYTHONPATH:-}" ]; then
  echo "[iq9075-e2e] candidate Python/source override: pre-release hardware evidence only"
fi

probe_user="$(id -un)"
if [ "$(id -u)" -eq 0 ]; then
  probe_user=nuvion
fi
probe_group="$(id -gn "$probe_user")"
probe_runtime_dir="$(mktemp -d /tmp/nuvion-iq9075-e2e.XXXXXX)"
case "$probe_runtime_dir" in
  /tmp/nuvion-iq9075-e2e.*) ;;
  *) die "unsafe probe runtime directory" ;;
esac
cleanup_probe_runtime() {
  case "$probe_runtime_dir" in
    /tmp/nuvion-iq9075-e2e.*)
      [ ! -L "$probe_runtime_dir" ] && rm -rf -- "$probe_runtime_dir"
      ;;
  esac
}
trap cleanup_probe_runtime EXIT
if [ "$(id -u)" -eq 0 ]; then
  chown "$probe_user:$probe_group" "$probe_runtime_dir"
  for directory in home cache config runtime; do
    install -d -m 0700 -o "$probe_user" -g "$probe_group" \
      "$probe_runtime_dir/$directory"
  done
else
  chmod 0700 "$probe_runtime_dir"
  for directory in home cache config runtime; do
    install -d -m 0700 "$probe_runtime_dir/$directory"
  done
fi

validate_release_identity() {
  (
    cd "$probe_runtime_dir"
    PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
      "$agent_python" -I - \
      "$expected_version" "$expected_component_sha" "$expected_bom_digest" <<'PY'
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

version, component_sha, bom_digest = sys.argv[1:]
slot = Path("/opt/nuv-agent/releases") / bom_digest.removeprefix("sha256:")
marker = slot / ".nuvion/release.json"
before = marker.lstat()
if (
    stat.S_ISLNK(before.st_mode)
    or not stat.S_ISREG(before.st_mode)
    or before.st_uid != 0
    or before.st_mode & 0o022
):
    raise SystemExit("candidate release marker metadata is unsafe")
raw = marker.read_bytes()
after = marker.lstat()
identity = lambda item: (
    item.st_dev,
    item.st_ino,
    item.st_mode,
    item.st_size,
    item.st_mtime_ns,
    item.st_ctime_ns,
)
if identity(before) != identity(after):
    raise SystemExit("candidate release marker changed while reading")
try:
    value = json.loads(raw)
except (UnicodeError, json.JSONDecodeError) as error:
    raise SystemExit("candidate release marker is invalid") from error
if (
    not isinstance(value, dict)
    or value.get("agentVersion") != version
    or value.get("componentSha") != component_sha
    or value.get("bomDigest") != bom_digest
):
    raise SystemExit("candidate release marker identity mismatch")
from nuvion_app import build_info

module_path = Path(build_info.__file__).resolve(strict=True)
if (
    not module_path.is_relative_to(slot.resolve(strict=True))
    or build_info.AGENT_VERSION != version
    or build_info.COMPONENT_SHA != component_sha
):
    raise SystemExit("installed candidate build identity mismatch")
print(hashlib.sha256(raw).hexdigest())
PY
  )
}

release_marker_sha=""
if [ -n "$evidence_output" ]; then
  release_marker_sha="$(validate_release_identity)" || \
    die "candidate release identity validation failed"
  [[ "$release_marker_sha" =~ ^[0-9a-f]{64}$ ]] || \
    die "candidate release marker digest is invalid"
fi
probe_environment=(
  "HOME=$probe_runtime_dir/home"
  "XDG_CACHE_HOME=$probe_runtime_dir/cache"
  "XDG_CONFIG_HOME=$probe_runtime_dir/config"
  "XDG_RUNTIME_DIR=$probe_runtime_dir/runtime"
)

webrtc_python=(
  /usr/bin/env
  -C "$probe_runtime_dir"
  "${probe_environment[@]}"
  "PYTHONPATH=${PYTHONPATH:-}"
  "PYTHONNOUSERSITE=1"
  "PYTHONDONTWRITEBYTECODE=1"
  "NUVION_IQ9075_EXPECTED_BOM_DIGEST=$expected_bom_digest"
  "G_DEBUG=fatal-criticals"
  "$agent_python"
)
if [ -n "$evidence_output" ]; then
  webrtc_python+=("-I")
fi
if [ "$(id -u)" -eq 0 ]; then
  command -v runuser >/dev/null 2>&1 || die "runuser is required for non-root Agent probes"
  webrtc_python=(
    runuser -u nuvion --
    "${webrtc_python[@]}"
  )
fi

echo "[iq9075-e2e] checking disposable WebRTC branches across signaling resets"
G_DEBUG=fatal-criticals timeout 20s "${webrtc_python[@]}" <<'PY'
import gi
import os
from pathlib import Path

gi.require_version("Gst", "1.0")
gi.require_version("GstWebRTC", "1.0")
gi.require_version("GstSdp", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

from nuvion_app.inference.webrtc_uplink import WebRTCUplinkController
from nuvion_app import build_info as installed_build_info
import nuvion_app.inference.webrtc_uplink as installed_webrtc_module

Gst.init(None)
expected_bom_digest = os.environ.get("NUVION_IQ9075_EXPECTED_BOM_DIGEST", "")
if expected_bom_digest:
    expected_slot = (
        Path("/opt/nuv-agent/releases") / expected_bom_digest.removeprefix("sha256:")
    ).resolve(strict=True)
    for module in (installed_build_info, installed_webrtc_module):
        if not Path(module.__file__).resolve(strict=True).is_relative_to(expected_slot):
            raise RuntimeError("WebRTC evidence probe imported outside candidate slot")
pipeline = Gst.parse_launch(
    "videotestsrc is-live=true ! "
    "video/x-raw,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! "
    "x264enc tune=zerolatency speed-preset=ultrafast bitrate=800 key-int-max=30 ! "
    "rtph264pay config-interval=1 pt=96 ! "
    "application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000 ! "
    "tee name=webrtc_uplink_tee allow-not-linked=true"
)
sent = []
controller = WebRTCUplinkController(
    send_message=lambda destination, payload, remember, _token: (
        sent.append((destination, payload["sessionId"], remember)) or True
    )
)
if not controller.attach_pipeline(pipeline):
    raise RuntimeError("WebRTC RTP tee attachment failed")
if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
    raise RuntimeError("WebRTC reset test pipeline failed to enter PLAYING")

loop = GLib.MainLoop()
branch_names = []
session_ids = ["session-1", "session-2", "session-3"]
session_index = 0
sequence_complete = False


def start(session_id):
    controller.start(
        {
            "broadcastId": "iq9075-release-test",
            "sessionId": session_id,
            "iceServers": [],
        }
    )
    return False


def advance_after_offer():
    global sequence_complete, session_index
    offers = [item for item in sent if item[0].endswith("/offer")]
    branch = controller._branch
    if (
        len(offers) <= session_index
        or branch is None
        or not branch.offer_enqueued
    ):
        return True
    branch_names.append(branch.webrtcbin.get_name())
    if session_index == len(session_ids) - 1:
        sequence_complete = True
        loop.quit()
        return False
    controller.on_signaling_reset()
    session_index += 1
    start(session_ids[session_index])
    return True


def sequence_timeout():
    loop.quit()
    return False


advance_source = GLib.timeout_add(10, advance_after_offer)
deadline_source = GLib.timeout_add(10_000, sequence_timeout)
start(session_ids[0])
loop.run()
if sequence_complete:
    GLib.source_remove(deadline_source)
else:
    GLib.source_remove(advance_source)
    raise RuntimeError(
        "WebRTC reset/re-offer sequence timed out: "
        f"branches={branch_names} sent={sent}"
    )

health = controller.runtime_health_snapshot()
# Force one partial teardown after request-pad releases, then retry the same
# branch. G_DEBUG=fatal-criticals makes any repeated release_request_pad fatal.
controller._pipeline = None
first_teardown = controller._teardown_branch_on_main_loop()
controller._pipeline = pipeline
second_teardown = controller._teardown_branch_on_main_loop()
if first_teardown or not second_teardown:
    raise RuntimeError(
        "WebRTC partial teardown did not recover idempotently: "
        f"first={first_teardown} second={second_teardown}"
    )
controller.stop(send_signal=False)
context = GLib.MainContext.default()
while context.pending():
    context.iteration(False)
pipeline.set_state(Gst.State.NULL)

issues = []
bus = pipeline.get_bus()
while True:
    message = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.WARNING)
    if message is None:
        break
    if message.type == Gst.MessageType.ERROR:
        error, debug = message.parse_error()
    else:
        error, debug = message.parse_warning()
    issues.append((message.type.value_nick, str(error), debug))

expected_branches = [
    "webrtc_uplink_session_1",
    "webrtc_uplink_session_3",
    "webrtc_uplink_session_5",
]
offers = [item for item in sent if item[0].endswith("/offer")]
if branch_names != expected_branches:
    raise RuntimeError(f"WebRTC branches were reused or skipped: {branch_names}")
offer_session_ids = [item[1] for item in offers]
if health.get("sessionId") != "session-3" or offer_session_ids != session_ids:
    raise RuntimeError(f"WebRTC re-offer evidence is incomplete: health={health} offers={offers}")
if issues:
    raise RuntimeError(f"WebRTC reset emitted GStreamer bus failures: {issues}")
print("[iq9075-e2e] disposable WebRTC reset/re-offer: PASS")
PY

if [ "$camera_mode" = "oak" ]; then
  oak_python=(
    /usr/bin/env
    -C "$probe_runtime_dir"
    "${probe_environment[@]}"
    "PYTHONPATH=${PYTHONPATH:-}"
    "PYTHONNOUSERSITE=1"
    "PYTHONDONTWRITEBYTECODE=1"
    "NUVION_IQ9075_OAK_SOAK_SECONDS=${NUVION_IQ9075_OAK_SOAK_SECONDS:-120}"
    "NUVION_IQ9075_OAK_MAX_RSS_SLOPE_MIB_PER_MIN=${NUVION_IQ9075_OAK_MAX_RSS_SLOPE_MIB_PER_MIN:-2}"
    "NUVION_IQ9075_OAK_MAX_RSS_RANGE_MIB=${NUVION_IQ9075_OAK_MAX_RSS_RANGE_MIB:-32}"
    "NUVION_IQ9075_OAK_EVIDENCE_OUTPUT=$probe_runtime_dir/oak-soak-result.json"
    "NUVION_IQ9075_EXPECTED_VERSION=$expected_version"
    "NUVION_IQ9075_EXPECTED_COMPONENT_SHA=$expected_component_sha"
    "NUVION_IQ9075_EXPECTED_BOM_DIGEST=$expected_bom_digest"
    "NUVION_IQ9075_RELEASE_MARKER_SHA256=$release_marker_sha"
    "NUVION_AGENT_PYTHON=$agent_python"
    "$agent_python"
  )
  if [ -n "$evidence_output" ]; then
    oak_python+=("-I")
  fi
  if [ "$(id -u)" -eq 0 ]; then
    command -v runuser >/dev/null 2>&1 || die "runuser is required for the non-root OAK access check"
    oak_python=(
      runuser -u nuvion --
      "${oak_python[@]}"
    )
  elif [ "$(id -un)" != "nuvion" ]; then
    die "run the OAK test with sudo so capture can be verified as the nuvion service user"
  fi

  echo "[iq9075-e2e] checking OAK-D capture as non-root user nuvion"
  set +e
  timeout 720s "${oak_python[@]}" <<'PY'
from importlib.metadata import version
from pathlib import Path
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
import platform
import re
import threading
import time
import weakref

import depthai
import numpy as np
from dotenv import dotenv_values

from nuvion_app.inference.depthai_gst import DepthAIGStreamerBridge
from nuvion_app.inference.depthai_source import DepthAIConfig, DepthAIFrameSource
from nuvion_app.inference.pipeline import build_bounded_live_queue, build_uplink_pipeline
from nuvion_app.inference.video_source import DEPTHAI_APPSRC_NAME
from nuvion_app.inference.video_source import build_video_source_pipeline
from nuvion_app.inference.video_source import resolve_depthai_device_id
from nuvion_app.inference.webrtc_signaling import (
    WEBRTC_UPLINK_OFFER_DEST,
    WEBRTC_UPLINK_STOP_DEST,
)
from nuvion_app.inference.webrtc_uplink import WebRTCUplinkController
from nuvion_app import build_info as installed_build_info
import nuvion_app.inference.depthai_gst as installed_depthai_gst_module
import nuvion_app.inference.depthai_source as installed_depthai_source_module
import nuvion_app.inference.pipeline as installed_pipeline_module
import nuvion_app.inference.webrtc_uplink as installed_webrtc_module

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstSdp", "1.0")
gi.require_version("GstWebRTC", "1.0")
from gi.repository import GLib, Gst, GstSdp, GstWebRTC  # noqa: E402,F401

WIDTH = 640
HEIGHT = 480
FPS = 30
SEGMENT_SECONDS = 4
MAX_SEGMENTS = 30
RSS_WARMUP_SECONDS = 20.0
RSS_SAMPLE_SECONDS = 5.0
MIN_RAW_FPS = 27.0
MAX_APPSRC_BYTES = 2 * WIDTH * HEIGHT * 3
started_at = (
    datetime.now(timezone.utc)
    .replace(microsecond=0)
    .isoformat()
    .replace("+00:00", "Z")
)

expected_bom_digest = os.environ.get("NUVION_IQ9075_EXPECTED_BOM_DIGEST", "")
if expected_bom_digest:
    expected_slot = (
        Path("/opt/nuv-agent/releases") / expected_bom_digest.removeprefix("sha256:")
    ).resolve(strict=True)
    expected_version = os.environ["NUVION_IQ9075_EXPECTED_VERSION"]
    expected_component_sha = os.environ["NUVION_IQ9075_EXPECTED_COMPONENT_SHA"]
    if (
        installed_build_info.AGENT_VERSION != expected_version
        or installed_build_info.COMPONENT_SHA != expected_component_sha
    ):
        raise SystemExit("OAK evidence process loaded a different build identity")
    for module in (
        installed_build_info,
        installed_depthai_gst_module,
        installed_depthai_source_module,
        installed_pipeline_module,
        installed_webrtc_module,
    ):
        if not Path(module.__file__).resolve(strict=True).is_relative_to(expected_slot):
            raise SystemExit("OAK evidence process imported outside candidate slot")

expected_version = "2.32.0.0"
installed_version = version("depthai")
module_version = str(getattr(depthai, "__version__", "")).strip()
if installed_version != expected_version or module_version != expected_version:
    raise SystemExit(
        "depthai version mismatch: "
        f"expected {expected_version}, distribution={installed_version}, "
        f"module={module_version or '<missing>'}"
    )

values = dotenv_values(Path("/etc/nuv-agent/agent.env"))
configured_device_id = str(values.get("NUVION_DEVICE_ID") or "").strip()
try:
    configured_space_id = int(str(values.get("NUVION_SPACE_ID") or ""))
except ValueError as error:
    raise SystemExit("NUVION_SPACE_ID is invalid") from error
device_match = re.fullmatch(
    r"sp-([1-9][0-9]*)-nuvion-[a-z0-9][a-z0-9-]{0,100}",
    configured_device_id,
)
if device_match is None or int(device_match.group(1)) != configured_space_id:
    raise SystemExit("Agent device/space identity is invalid")
video_source = str(values.get("NUVION_VIDEO_SOURCE") or "oak").strip() or "oak"
gst_source_override = str(values.get("NUVION_GST_SOURCE") or "").strip()
demo_mode = str(values.get("NUVION_DEMO_MODE") or "false").strip().lower()
if gst_source_override:
    raise SystemExit("NUVION_GST_SOURCE must be empty for physical OAK E2E")
if demo_mode in {"1", "true", "yes", "on"}:
    raise SystemExit("NUVION_DEMO_MODE must be false for physical OAK E2E")
device_id = resolve_depthai_device_id(
    video_source,
    str(values.get("NUVION_DEPTHAI_DEVICE_ID") or ""),
)

usb_devices_root = Path("/sys/bus/usb/devices")
usb_topology_re = re.compile(r"^[12]-1(?:\.[1-9][0-9]*)+$")
sys_root = Path("/sys").resolve(strict=True)
oak_products = {"2485": "bootloader", "f63b": "runtime"}


def read_sysfs(device: Path, name: str) -> str:
    return (device / name).read_text(encoding="utf-8").strip().lower()


def discover_oak_usb_paths(products: set[str]) -> list[Path]:
    matches = []
    for candidate in sorted(usb_devices_root.iterdir(), key=lambda item: item.name):
        try:
            resolved = candidate.resolve(strict=True)
            vendor = read_sysfs(candidate, "idVendor")
            product = read_sysfs(candidate, "idProduct")
        except OSError:
            # DepthAI changes from USB2 bootloader to USB3 runtime identity.
            # A disappearing sysfs entry during that bounded transition is not
            # itself evidence of a second or unsafe device.
            continue
        if not resolved.is_dir() or not resolved.is_relative_to(sys_root):
            raise SystemExit("USB1 downstream topology escaped sysfs")
        if vendor == "03e7" and product in products:
            matches.append(candidate)
    return matches


def require_runtime_oak(timeout_seconds: float) -> Path:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        runtime_paths = discover_oak_usb_paths({"f63b"})
        if len(runtime_paths) > 1:
            raise SystemExit(
                "USB1 downstream must contain exactly one runtime OAK-D Lite"
            )
        if len(runtime_paths) == 1:
            runtime_path = runtime_paths[0]
            if not runtime_path.name.startswith("2-1."):
                raise SystemExit("runtime OAK-D Lite must enumerate on USB3")
            try:
                speed_mbps = float(read_sysfs(runtime_path, "speed"))
                driver_path = (runtime_path / "driver").resolve(strict=True)
                expected_driver = Path("/sys/bus/usb/drivers/usb").resolve(
                    strict=True
                )
            except (OSError, ValueError) as exc:
                raise SystemExit(
                    f"cannot validate runtime OAK USB identity: {exc}"
                ) from exc
            if driver_path != expected_driver or speed_mbps < 5000.0:
                raise SystemExit(
                    f"OAK-D Lite at {runtime_path.name} must use the exact USB "
                    f"driver at 5Gbps; observed driver={driver_path.name} "
                    f"speed={speed_mbps:g}Mbps"
                )
            return runtime_path
        time.sleep(0.1)
    raise SystemExit("OAK-D Lite did not enter USB3 runtime identity in time")


oak_usb_paths = discover_oak_usb_paths(set(oak_products))

if not oak_usb_paths:
    print("[iq9075-e2e] no OAK-D device detected below USB1 dual hub", flush=True)
    raise SystemExit(3)
if len(oak_usb_paths) != 1:
    raise SystemExit("USB1 downstream must contain exactly one OAK-D Lite")
initial_usb_path = oak_usb_paths[0]
if usb_topology_re.fullmatch(initial_usb_path.name) is None:
    raise SystemExit("the only attached OAK-D Lite is outside the USB1 dual hub")
try:
    initial_product = read_sysfs(initial_usb_path, "idProduct")
    initial_speed_mbps = float(read_sysfs(initial_usb_path, "speed"))
except OSError as exc:
    raise SystemExit(f"cannot read initial OAK USB identity: {exc}") from exc
except ValueError as exc:
    raise SystemExit("initial OAK USB speed is invalid") from exc
initial_mode = oak_products.get(initial_product)
if initial_mode is None:
    raise SystemExit("initial OAK USB product identity is invalid")
if initial_mode == "bootloader" and (
    not initial_usb_path.name.startswith("1-1.") or initial_speed_mbps < 480.0
):
    raise SystemExit(
        "OAK bootloader must enumerate on the USB2 side of the USB1 dual hub"
    )
if initial_mode == "runtime":
    require_runtime_oak(2.0)

available_devices = depthai.Device.getAllAvailableDevices()
available_ids = {
    str(device.getMxId()).strip()
    for device in available_devices
    if str(device.getMxId()).strip()
}
if not available_ids:
    raise SystemExit(
        "OAK USB device is present but DepthAI could not enumerate it; "
        "check udev permissions and native runtime"
    )
if len(available_ids) != 1:
    raise SystemExit("IQ9075 physical release test requires exactly one OAK MXID")
selected_mxid = next(iter(available_ids))
if device_id is not None and device_id != selected_mxid:
    raise SystemExit("configured OAK MXID is not attached")

startup_timeout = float(
    values.get("NUVION_DEPTHAI_STARTUP_TIMEOUT_SEC") or "15"
)
read_timeout = float(values.get("NUVION_DEPTHAI_READ_TIMEOUT_SEC") or "2")
soak_seconds = int(os.getenv("NUVION_IQ9075_OAK_SOAK_SECONDS", "120"))
max_rss_slope_mib_per_min = float(
    os.getenv("NUVION_IQ9075_OAK_MAX_RSS_SLOPE_MIB_PER_MIN", "2")
)
max_rss_range_mib = float(
    os.getenv("NUVION_IQ9075_OAK_MAX_RSS_RANGE_MIB", "32")
)
if soak_seconds < 120 or soak_seconds > 600:
    raise SystemExit("NUVION_IQ9075_OAK_SOAK_SECONDS must be in [120, 600]")
if (
    not math.isfinite(max_rss_slope_mib_per_min)
    or max_rss_slope_mib_per_min < 0.1
    or max_rss_slope_mib_per_min > 10.0
):
    raise SystemExit(
        "NUVION_IQ9075_OAK_MAX_RSS_SLOPE_MIB_PER_MIN must be in [0.1, 10]"
    )
if (
    not math.isfinite(max_rss_range_mib)
    or max_rss_range_mib < 8.0
    or max_rss_range_mib > 128.0
):
    raise SystemExit("NUVION_IQ9075_OAK_MAX_RSS_RANGE_MIB must be in [8, 128]")

config = DepthAIConfig(
    width=WIDTH,
    height=HEIGHT,
    fps=FPS,
    device_id=device_id,
    queue_size=1,
    startup_timeout=startup_timeout,
    read_timeout=read_timeout,
)

Gst.init(None)
runtime_root = Path(os.environ["XDG_RUNTIME_DIR"]).resolve(strict=True)
segment_dir = runtime_root / "oak-soak-segments"
segment_dir.mkdir(mode=0o700)
segment_dir = segment_dir.resolve(strict=True)
if segment_dir.parent != runtime_root or segment_dir.is_symlink():
    raise SystemExit("OAK clip evidence directory escaped the private runtime root")

source_pipeline = build_video_source_pipeline(
    video_source,
    WIDTH,
    HEIGHT,
    FPS,
    platform_name="linux",
)
raw_live_queue = build_bounded_live_queue(
    max_buffers=2,
    element_name="physical_raw_queue",
)
overlay_live_queue = build_bounded_live_queue(
    max_buffers=2,
    element_name="physical_overlay_queue",
)
uplink_pipeline = build_uplink_pipeline(
    rtp_ssrc=9075,
    clip_enabled=True,
    clip_segment_sec=SEGMENT_SECONDS,
    clip_max_segments=MAX_SEGMENTS,
    clip_segments_dir=str(segment_dir),
    video_bitrate_kbps=1000,
)
pipeline_description = (
    f"{source_pipeline} ! tee name=t "
    f"t. ! {raw_live_queue} ! "
    "appsink name=zsad_sink emit-signals=true max-buffers=1 drop=true sync=false "
    f"t. ! {overlay_live_queue} ! "
    "videoconvert ! textoverlay name=zsad_overlay "
    'font-desc="Sans 24" halignment=left valignment=top '
    'shaded-background=true text="" ! '
    f"{uplink_pipeline}"
)
pipeline = Gst.parse_launch(pipeline_description)
appsrc = pipeline.get_by_name(DEPTHAI_APPSRC_NAME)
raw_sink = pipeline.get_by_name("zsad_sink")
uplink_tee = pipeline.get_by_name("webrtc_uplink_tee")
clip_sink = pipeline.get_by_name("clip_sink")
if appsrc is None or raw_sink is None or uplink_tee is None or clip_sink is None:
    raise SystemExit("production OAK pipeline is missing a required element")

source = DepthAIFrameSource(config)
bridge = DepthAIGStreamerBridge(
    frame_source=source,
    appsrc=appsrc,
    gst=Gst,
    width=WIDTH,
    height=HEIGHT,
    read_timeout=read_timeout,
    max_consecutive_timeouts=3,
)
bus = pipeline.get_bus()
context = GLib.MainContext.default()
raw_lock = threading.Lock()
raw_samples = 0
raw_last_pts = None
raw_error = None
sent_lock = threading.Lock()
sent_messages = []
gst_errors = []
gst_warnings = []
fragment_opened = 0
max_appsrc_buffers = 0
max_appsrc_bytes = 0
queue_high_watermarks = {}


def read_proc_status_kib(field: str) -> int:
    prefix = f"{field}:"
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            parts = line.split()
            if len(parts) == 3 and parts[2] == "kB" and parts[1].isdigit():
                return int(parts[1])
    raise RuntimeError(f"/proc/self/status is missing exact {field} kB evidence")


def uint_property(element, name: str) -> int | None:
    if element.find_property(name) is None:
        return None
    return int(element.get_property(name))


def iterator_values(iterator):
    values = []
    while True:
        result, value = iterator.next()
        if result == Gst.IteratorResult.OK:
            values.append(value)
        elif result == Gst.IteratorResult.RESYNC:
            iterator.resync()
            values.clear()
        elif result == Gst.IteratorResult.DONE:
            return values
        else:
            raise RuntimeError("GStreamer iterator failed")


def on_raw_sample(appsink):
    global raw_error, raw_last_pts, raw_samples
    sample = appsink.emit("pull-sample")
    if sample is None:
        return Gst.FlowReturn.OK
    try:
        caps = sample.get_caps()
        structure = caps.get_structure(0) if caps is not None else None
        if structure is None or structure.get_name() != "video/x-raw":
            raise RuntimeError(f"unexpected raw caps: {caps}")
        if (
            int(structure.get_value("width")) != WIDTH
            or int(structure.get_value("height")) != HEIGHT
        ):
            raise RuntimeError(f"unexpected raw dimensions: {caps}")
        buffer = sample.get_buffer()
        if buffer is None or buffer.pts == Gst.CLOCK_TIME_NONE:
            raise RuntimeError("raw sample is missing a PTS")
        success, mapinfo = buffer.map(Gst.MapFlags.READ)
        if not success:
            raise RuntimeError("raw sample mapping failed")
        try:
            frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape(
                HEIGHT,
                WIDTH,
                3,
            ).copy()
            if frame.shape != (HEIGHT, WIDTH, 3):
                raise RuntimeError("raw RGB frame shape changed")
        finally:
            buffer.unmap(mapinfo)
        with raw_lock:
            if raw_last_pts is not None and buffer.pts < raw_last_pts:
                raise RuntimeError("raw sample PTS regressed")
            raw_last_pts = buffer.pts
            raw_samples += 1
    except Exception as exc:
        with raw_lock:
            if raw_error is None:
                raw_error = f"{type(exc).__name__}: {exc}"[:1000]
        return Gst.FlowReturn.ERROR
    return Gst.FlowReturn.OK


def raw_snapshot():
    with raw_lock:
        return raw_samples, raw_error


def send_message(destination, payload, remember, signaling_token):
    with sent_lock:
        sent_messages.append(
            (destination, dict(payload), bool(remember), signaling_token)
        )
    return True


def offer_snapshot():
    with sent_lock:
        return [
            item for item in sent_messages if item[0] == WEBRTC_UPLINK_OFFER_DEST
        ]


def drain_bus() -> None:
    global fragment_opened
    mask = (
        Gst.MessageType.ERROR
        | Gst.MessageType.WARNING
        | Gst.MessageType.EOS
        | Gst.MessageType.ELEMENT
    )
    while True:
        message = bus.timed_pop_filtered(0, mask)
        if message is None:
            break
        source_name = message.src.get_name() if message.src is not None else "unknown"
        if message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            gst_errors.append(f"{source_name}: {error}; debug={debug}"[:2000])
        elif message.type == Gst.MessageType.WARNING:
            warning, debug = message.parse_warning()
            gst_warnings.append(f"{source_name}: {warning}; debug={debug}"[:2000])
        elif message.type == Gst.MessageType.EOS:
            gst_errors.append(f"unexpected EOS from {source_name}")
        elif message.type == Gst.MessageType.ELEMENT:
            structure = message.get_structure()
            if (
                structure is not None
                and structure.get_name() == "splitmuxsink-fragment-opened"
            ):
                fragment_opened += 1
    if gst_errors:
        raise RuntimeError(f"GStreamer bus error: {gst_errors[-1]}")


def check_bounded_levels(static_queues) -> None:
    global max_appsrc_buffers, max_appsrc_bytes
    level_buffers = uint_property(appsrc, "current-level-buffers")
    level_bytes = uint_property(appsrc, "current-level-bytes")
    if level_buffers is not None:
        max_appsrc_buffers = max(max_appsrc_buffers, level_buffers)
        if level_buffers > 2:
            raise RuntimeError(f"appsrc buffer bound exceeded: {level_buffers} > 2")
    if level_bytes is not None:
        max_appsrc_bytes = max(max_appsrc_bytes, level_bytes)
        if level_bytes > MAX_APPSRC_BYTES:
            raise RuntimeError(
                f"appsrc byte bound exceeded: {level_bytes} > {MAX_APPSRC_BYTES}"
            )
    for queue_element in static_queues:
        name = queue_element.get_name()
        if uint_property(queue_element, "max-size-buffers") != 2:
            raise RuntimeError(f"live queue {name} is not configured for two buffers")
        current = uint_property(queue_element, "current-level-buffers") or 0
        queue_high_watermarks[name] = max(queue_high_watermarks.get(name, 0), current)
        if current > 2:
            raise RuntimeError(f"live queue {name} buffer bound exceeded: {current} > 2")


def service_runtime(static_queues) -> None:
    while context.pending():
        context.iteration(False)
    drain_bus()
    if bridge.failure is not None:
        raise RuntimeError(f"DepthAI bridge failed: {bridge.failure}")
    _, error = raw_snapshot()
    if error is not None:
        raise RuntimeError(f"raw appsink failed: {error}")
    check_bounded_levels(static_queues)


def wait_until(predicate, timeout_seconds: float, label: str, static_queues) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        service_runtime(static_queues)
        if predicate():
            return
        time.sleep(0.01)
    service_runtime(static_queues)
    if not predicate():
        raise RuntimeError(f"timed out waiting for {label}")


def rss_slope_mib_per_min(samples) -> float:
    if len(samples) < 18:
        raise RuntimeError(f"insufficient post-warmup RSS samples: {len(samples)} < 18")
    origin = samples[0][0]
    xs = [(timestamp - origin) / 60.0 for timestamp, _rss in samples]
    ys = [rss_kib / 1024.0 for _timestamp, rss_kib in samples]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((value - mean_x) ** 2 for value in xs)
    if denominator <= 0:
        raise RuntimeError("invalid RSS sample timestamps")
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator


def request_pad_count(element) -> int:
    return len(iterator_values(element.iterate_src_pads()))


raw_sink.connect("new-sample", on_raw_sample)
static_queues = [
    element
    for element in iterator_values(pipeline.iterate_elements())
    if element.get_factory() is not None
    and element.get_factory().get_name() == "queue"
]
if len(static_queues) != 4:
    raise SystemExit(
        f"production OAK pipeline must have four bounded live queues, got {len(static_queues)}"
    )
if uint_property(appsrc, "max-buffers") != 2:
    raise SystemExit("DepthAI appsrc max-buffers must equal 2")
if uint_property(appsrc, "max-bytes") != MAX_APPSRC_BYTES:
    raise SystemExit("DepthAI appsrc max-bytes does not match two RGB frames")

controller = WebRTCUplinkController(
    send_message=send_message,
    offer_answer_timeout_sec=3.0,
)
if not controller.attach_pipeline(pipeline):
    raise SystemExit("WebRTC RTP tee attachment failed")

state_result = pipeline.set_state(Gst.State.PLAYING)
if state_result == Gst.StateChangeReturn.FAILURE:
    pipeline.set_state(Gst.State.NULL)
    raise SystemExit("OAK GStreamer pipeline failed to enter PLAYING")

runtime_usb_path = initial_usb_path
webrtc_evidence = {}
soak_start_samples = 0
soak_started = time.monotonic()
soak_elapsed = 0.0
soak_raw_samples = 0
raw_fps = 0.0
rss_samples = []
rss_slope = None
rss_range = None
fragments_at_soak_start = 0
segments = []
fragment_delta = 0
newest_segment_age = None
bridge_stats = bridge.stats_snapshot()
run_failure = None
cleanup_errors = []
try:
    bridge.start()
    runtime_usb_path = require_runtime_oak(startup_timeout)
    wait_until(
        lambda: raw_snapshot()[0] >= 30,
        startup_timeout + 10.0,
        "30 production raw frames",
        static_queues,
    )

    controller.start(
        {
            "broadcastId": "iq9075-physical-soak",
            "sessionId": "unanswered-offer",
            "iceServers": [],
            "forceRelay": False,
        }
    )
    wait_until(
        lambda: len(offer_snapshot()) == 1,
        10.0,
        "one local SDP offer",
        static_queues,
    )
    offer_destination, offer_payload, offer_remember, offer_token = offer_snapshot()[0]
    offer_sdp = offer_payload.get("sdp")
    if (
        offer_destination != WEBRTC_UPLINK_OFFER_DEST
        or not offer_remember
        or offer_payload.get("sessionId") != "unanswered-offer"
        or not isinstance(offer_sdp, str)
        or "v=0" not in offer_sdp
        or "profile-level-id=42e01f" not in offer_sdp
    ):
        raise RuntimeError("physical WebRTC offer evidence is incomplete")
    rejected_branch = controller._branch
    if rejected_branch is None:
        raise RuntimeError("physical WebRTC offer has no owned disposable branch")
    old_queue = rejected_branch.queue
    old_webrtc = rejected_branch.webrtcbin
    old_queue_ref = weakref.ref(old_queue)
    old_webrtc_ref = weakref.ref(old_webrtc)
    wait_until(
        lambda: controller._branch is None,
        10.0,
        "unanswered WebRTC watchdog branch teardown",
        static_queues,
    )
    with sent_lock:
        terminal_stops = [
            item
            for item in sent_messages
            if item[0] == WEBRTC_UPLINK_STOP_DEST
            and item[1].get("sessionId") == "unanswered-offer"
            and getattr(item[3], "terminal", False)
        ]
    if len(terminal_stops) != 1 or getattr(offer_token, "terminal", True):
        raise RuntimeError("unanswered offer watchdog did not emit exact terminal STOP")
    if old_queue.get_parent() is not None or old_webrtc.get_parent() is not None:
        raise RuntimeError("rejected WebRTC elements still have a pipeline parent")
    if old_queue.get_state(0)[1] != Gst.State.NULL:
        raise RuntimeError("rejected WebRTC queue did not reach NULL")
    if old_webrtc.get_state(0)[1] != Gst.State.NULL:
        raise RuntimeError("rejected webrtcbin did not reach NULL")
    if request_pad_count(uplink_tee) != 0:
        raise RuntimeError("rejected WebRTC tee request pad was not released")
    if controller.runtime_health_snapshot()["hasPipeline"]:
        raise RuntimeError("rejected WebRTC runtime still reports an active pipeline")
    del rejected_branch, old_queue, old_webrtc
    gc.collect()
    wait_until(
        lambda: old_queue_ref() is None and old_webrtc_ref() is None,
        5.0,
        "rejected WebRTC branch object finalization",
        static_queues,
    )
    webrtc_evidence = {
        "offerCount": len(offer_snapshot()),
        "terminalStopCount": len(terminal_stops),
        "offerSdpHadPinnedProfile": "profile-level-id=42e01f" in offer_sdp,
        "branchParentDetached": True,
        "queueParentDetached": True,
        "webrtcParentDetached": True,
        "teeRequestPadCount": request_pad_count(uplink_tee),
        "queueState": "NULL",
        "webrtcState": "NULL",
        "branchObjectsFinalized": True,
        "hasPipeline": False,
    }

    warmup_deadline = time.monotonic() + RSS_WARMUP_SECONDS
    while time.monotonic() < warmup_deadline:
        service_runtime(static_queues)
        time.sleep(0.01)

    soak_start_samples = raw_snapshot()[0]
    soak_started = time.monotonic()
    soak_deadline = soak_started + soak_seconds
    next_rss_sample = soak_started
    rss_samples = []
    fragments_at_soak_start = fragment_opened
    while time.monotonic() < soak_deadline:
        service_runtime(static_queues)
        now = time.monotonic()
        if now >= next_rss_sample:
            rss_samples.append((now, read_proc_status_kib("RssAnon")))
            next_rss_sample += RSS_SAMPLE_SECONDS
        time.sleep(0.01)
    service_runtime(static_queues)
    if not rss_samples or soak_deadline - rss_samples[-1][0] >= 2.5:
        rss_samples.append((time.monotonic(), read_proc_status_kib("RssAnon")))

    soak_elapsed = time.monotonic() - soak_started
    soak_raw_samples = raw_snapshot()[0] - soak_start_samples
    raw_fps = soak_raw_samples / soak_elapsed
    minimum_raw_samples = int(0.9 * FPS * soak_seconds)
    if raw_fps < MIN_RAW_FPS or soak_raw_samples < minimum_raw_samples:
        raise RuntimeError(
            "production raw branch throughput fell below bound: "
            f"fps={raw_fps:.2f} samples={soak_raw_samples} "
            f"minimumSamples={minimum_raw_samples}"
        )
    rss_slope = rss_slope_mib_per_min(rss_samples)
    rss_values_mib = [value / 1024.0 for _timestamp, value in rss_samples]
    rss_range = max(rss_values_mib) - min(rss_values_mib)
    if rss_slope > max_rss_slope_mib_per_min:
        raise RuntimeError(
            "post-rejection anonymous RSS slope exceeded bound: "
            f"slope={rss_slope:.3f}MiB/min "
            f"limit={max_rss_slope_mib_per_min:.3f}MiB/min"
        )
    if rss_range > max_rss_range_mib:
        raise RuntimeError(
            "post-rejection anonymous RSS range exceeded bound: "
            f"range={rss_range:.3f}MiB limit={max_rss_range_mib:.3f}MiB"
        )

    segments = sorted(segment_dir.glob("segment_*.mp4"))
    if not segments or len(segments) > MAX_SEGMENTS:
        raise RuntimeError(
            f"splitmux segment retention is invalid: count={len(segments)}"
        )
    newest_segment_age = min(time.time() - item.stat().st_mtime for item in segments)
    if newest_segment_age > 2 * SEGMENT_SECONDS + 5:
        raise RuntimeError(
            f"splitmux newest segment is stale: age={newest_segment_age:.1f}s"
        )
    fragment_delta = fragment_opened - fragments_at_soak_start
    minimum_fragments = math.floor(soak_seconds / SEGMENT_SECONDS) - 3
    if fragment_delta < minimum_fragments:
        raise RuntimeError(
            "splitmux fragment progress fell below bound: "
            f"opened={fragment_delta} minimum={minimum_fragments}"
        )
    if request_pad_count(uplink_tee) != 0 or controller._branch is not None:
        raise RuntimeError("WebRTC branch resources reappeared during steady state")
    if gst_errors:
        raise RuntimeError(f"GStreamer errors were observed: {gst_errors}")
    bridge_stats = bridge.stats_snapshot()
except BaseException as exc:  # Persist bounded diagnostics before failing the gate.
    run_failure = f"{type(exc).__name__}: {exc}"[:2000]
finally:
    try:
        bridge_stats = bridge.stats_snapshot()
    except Exception as exc:
        cleanup_errors.append(f"bridge stats: {type(exc).__name__}: {exc}"[:1000])
    try:
        segments = sorted(segment_dir.glob("segment_*.mp4"))
        fragment_delta = fragment_opened - fragments_at_soak_start
        if segments:
            newest_segment_age = min(
                time.time() - item.stat().st_mtime for item in segments
            )
    except Exception as exc:
        cleanup_errors.append(f"splitmux snapshot: {type(exc).__name__}: {exc}"[:1000])
    try:
        controller.stop(send_signal=False)
        teardown_deadline = time.monotonic() + 5.0
        while context.pending() and time.monotonic() < teardown_deadline:
            context.iteration(False)
    except Exception as exc:
        cleanup_errors.append(f"WebRTC teardown: {type(exc).__name__}: {exc}"[:1000])
    try:
        bridge.close()
    except Exception as exc:
        cleanup_errors.append(f"DepthAI teardown: {type(exc).__name__}: {exc}"[:1000])
    try:
        pipeline.set_state(Gst.State.NULL)
        pipeline.get_state(5 * Gst.SECOND)
    except Exception as exc:
        cleanup_errors.append(f"GStreamer teardown: {type(exc).__name__}: {exc}"[:1000])

evidence_path = Path(os.environ["NUVION_IQ9075_OAK_EVIDENCE_OUTPUT"])
if evidence_path.parent.resolve(strict=True) != runtime_root or evidence_path.name != "oak-soak-result.json":
    raise SystemExit("OAK evidence output escaped the private runtime root")
evidence = {
    "schemaVersion": 2,
    "kind": "nuvion-iq9075-oak-soak-result",
    "startedAt": started_at,
    "outcome": {
        "status": "failed" if run_failure is not None or cleanup_errors else "passed",
        "error": run_failure,
        "cleanupErrors": cleanup_errors,
    },
    "board": {
        "productModel": "IQ9075_DEV",
        "platformProfile": "iq9075_dev",
        "hardwareRevision": "QCS9075-EVK",
        "architecture": "aarch64",
        "kernel": platform.release(),
        "depthaiVersion": installed_version,
        "gstreamerVersion": Gst.version_string(),
    },
    "oakMxidSha256": hashlib.sha256(
        selected_mxid.lower().encode("utf-8")
    ).hexdigest(),
    "deviceIdentity": {
        "deviceId": configured_device_id,
        "spaceId": configured_space_id,
    },
    "runtimeIdentity": {
        "agentVersion": os.environ["NUVION_IQ9075_EXPECTED_VERSION"],
        "componentSha": os.environ["NUVION_IQ9075_EXPECTED_COMPONENT_SHA"],
        "bomDigest": os.environ["NUVION_IQ9075_EXPECTED_BOM_DIGEST"],
        "pythonPath": str(Path(os.environ.get("NUVION_AGENT_PYTHON", "/opt/nuv-agent/current/venv/bin/python"))),
        "releaseMarkerSha256": os.environ["NUVION_IQ9075_RELEASE_MARKER_SHA256"],
    },
    "soak": {
        "durationSeconds": round(soak_elapsed, 6),
        "targetFps": float(FPS),
        "rawSamples": soak_raw_samples,
        "rssAnonSamples": [
            {
                "elapsedSec": round(timestamp - soak_started, 6),
                "rssAnonKiB": rss_kib,
            }
            for timestamp, rss_kib in rss_samples
        ],
        "rssAnonSlopeMiBPerMin": (
            round(rss_slope, 6) if rss_slope is not None else None
        ),
        "rssAnonRangeMiB": round(rss_range, 6) if rss_range is not None else None,
        "gstreamerErrors": list(gst_errors),
        "gstreamerWarnings": list(gst_warnings),
        "maxAppsrcBuffers": max_appsrc_buffers,
        "maxAppsrcBytes": max_appsrc_bytes,
        "queueHighWatermarks": dict(sorted(queue_high_watermarks.items())),
    },
    "webrtc": webrtc_evidence,
    "splitmux": {
        "segmentSeconds": float(SEGMENT_SECONDS),
        "retentionLimit": MAX_SEGMENTS,
        "segmentsAtEnd": len(segments),
        "fragmentsOpenedDuringSoak": fragment_delta,
        "newestSegmentAgeSeconds": (
            round(newest_segment_age, 6)
            if newest_segment_age is not None
            else None
        ),
    },
}
serialized = (json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
temporary_evidence_path = evidence_path.with_name(
    f".{evidence_path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp"
)
try:
    descriptor = os.open(
        temporary_evidence_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as output:
            output.write(serialized)
            output.flush()
            os.fsync(output.fileno())
    finally:
        os.close(descriptor)
    os.link(
        temporary_evidence_path,
        evidence_path,
        follow_symlinks=False,
    )
    directory_descriptor = os.open(runtime_root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
finally:
    temporary_evidence_path.unlink(missing_ok=True)

if run_failure is not None or cleanup_errors:
    detail = run_failure or cleanup_errors[0]
    print(
        "[iq9075-e2e] preserved failure evidence: "
        f"{serialized.decode('utf-8').strip()}",
        flush=True,
    )
    print(
        "[iq9075-e2e] OAK production offer/watchdog/teardown RSS soak: FAIL "
        f"({detail}; evidence={evidence_path})",
        flush=True,
    )
    raise SystemExit(detail)

print(
    "[iq9075-e2e] OAK production offer/watchdog/teardown RSS soak: PASS "
    f"(depthai={installed_version}, usb={runtime_usb_path.name}, "
    f"samples={soak_raw_samples}, fps={raw_fps:.2f}, duration={soak_elapsed:.1f}s, "
    f"rssAnonSlope={rss_slope:.3f}MiB/min, rssAnonRange={rss_range:.3f}MiB, "
    f"fragments={fragment_delta}, maxAppsrcBuffers={max_appsrc_buffers}, "
    f"maxAppsrcBytes={max_appsrc_bytes}, queues={queue_high_watermarks}, "
    f"gstWarnings={len(gst_warnings)}, bridge={bridge_stats})"
)
PY
  oak_status=$?
  set -e
  if [ -n "$evidence_output" ] && [ -f "$probe_runtime_dir/oak-soak-result.json" ]; then
    post_release_marker_sha="$(validate_release_identity)" || \
      die "candidate release identity changed after OAK soak"
    [ "$post_release_marker_sha" = "$release_marker_sha" ] || \
      die "candidate release marker changed during OAK soak"
    /usr/bin/python3 -I - \
      "$probe_runtime_dir/oak-soak-result.json" "$evidence_output" "$(id -u)" <<'PY'
import json
import os
import stat
import sys
import time
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
expected_uid = int(sys.argv[3])
if (
    not destination.is_absolute()
    or os.path.normpath(str(destination)) != str(destination)
    or destination.name in {"", ".", ".."}
    or destination.suffix != ".json"
    or destination.exists()
    or destination.is_symlink()
):
    raise SystemExit("unsafe IQ9075 evidence destination")
parent = destination.parent.resolve(strict=True)
metadata = parent.stat()
if (
    not stat.S_ISDIR(metadata.st_mode)
    or metadata.st_uid != expected_uid
    or metadata.st_mode & 0o022
):
    raise SystemExit("IQ9075 evidence destination directory is not private")
source_metadata = source.stat(follow_symlinks=False)
if not stat.S_ISREG(source_metadata.st_mode) or source_metadata.st_nlink != 1:
    raise SystemExit("IQ9075 evidence source must be one regular private file")
raw = source.read_bytes()
if not raw or len(raw) > 1024 * 1024:
    raise SystemExit("IQ9075 evidence payload size is invalid")


def reject_duplicate(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON member: {key}")
        result[key] = value
    return result


try:
    payload = json.loads(
        raw,
        object_pairs_hook=reject_duplicate,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"invalid JSON constant: {value}")
        ),
    )
except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
    raise SystemExit(f"IQ9075 evidence source is not strict JSON: {exc}") from exc

root_keys = {
    "schemaVersion", "kind", "startedAt", "outcome", "board",
    "oakMxidSha256", "deviceIdentity", "runtimeIdentity", "soak",
    "webrtc", "splitmux",
}
outcome_keys = {"status", "error", "cleanupErrors"}
board_keys = {
    "productModel", "platformProfile", "hardwareRevision", "architecture",
    "kernel", "depthaiVersion", "gstreamerVersion",
}
runtime_keys = {
    "agentVersion", "componentSha", "bomDigest", "pythonPath",
    "releaseMarkerSha256",
}
soak_keys = {
    "durationSeconds", "targetFps", "rawSamples", "rssAnonSamples",
    "rssAnonSlopeMiBPerMin", "rssAnonRangeMiB", "gstreamerErrors",
    "gstreamerWarnings", "maxAppsrcBuffers", "maxAppsrcBytes",
    "queueHighWatermarks",
}
webrtc_keys = {
    "offerCount", "terminalStopCount", "offerSdpHadPinnedProfile",
    "branchParentDetached", "queueParentDetached", "webrtcParentDetached",
    "teeRequestPadCount", "queueState", "webrtcState",
    "branchObjectsFinalized", "hasPipeline",
}
splitmux_keys = {
    "segmentSeconds", "retentionLimit", "segmentsAtEnd",
    "fragmentsOpenedDuringSoak", "newestSegmentAgeSeconds",
}
if (
    not isinstance(payload, dict)
    or set(payload) != root_keys
    or type(payload.get("schemaVersion")) is not int
    or payload.get("schemaVersion") != 2
    or payload.get("kind") != "nuvion-iq9075-oak-soak-result"
):
    raise SystemExit("IQ9075 evidence source root schema is invalid")

outcome = payload.get("outcome")
cleanup_errors = outcome.get("cleanupErrors") if isinstance(outcome, dict) else None
if (
    not isinstance(outcome, dict)
    or set(outcome) != outcome_keys
    or outcome.get("status") not in {"passed", "failed"}
    or not isinstance(cleanup_errors, list)
    or len(cleanup_errors) > 16
    or any(
        not isinstance(item, str) or not item or len(item) > 1000
        for item in cleanup_errors
    )
):
    raise SystemExit("IQ9075 evidence outcome schema is invalid")
if outcome["status"] == "passed":
    if outcome.get("error") is not None or cleanup_errors:
        raise SystemExit("passed IQ9075 evidence contains failure details")
elif (
    outcome.get("error") is not None
    and (
        not isinstance(outcome.get("error"), str)
        or not outcome["error"]
        or len(outcome["error"]) > 2000
    )
) or (outcome.get("error") is None and not cleanup_errors):
    raise SystemExit("failed IQ9075 evidence lacks bounded failure details")

board = payload.get("board")
device_identity = payload.get("deviceIdentity")
runtime_identity = payload.get("runtimeIdentity")
soak = payload.get("soak")
webrtc = payload.get("webrtc")
splitmux = payload.get("splitmux")
if (
    not isinstance(board, dict)
    or set(board) != board_keys
    or not isinstance(device_identity, dict)
    or set(device_identity) != {"deviceId", "spaceId"}
    or not isinstance(runtime_identity, dict)
    or set(runtime_identity) != runtime_keys
    or not isinstance(soak, dict)
    or set(soak) != soak_keys
    or not isinstance(webrtc, dict)
    or (
        set(webrtc) != webrtc_keys
        and not (outcome["status"] == "failed" and not webrtc)
    )
    or not isinstance(splitmux, dict)
    or set(splitmux) != splitmux_keys
):
    raise SystemExit("IQ9075 evidence nested object fields are invalid")

rss_samples = soak.get("rssAnonSamples")
errors = soak.get("gstreamerErrors")
warnings = soak.get("gstreamerWarnings")
queue_levels = soak.get("queueHighWatermarks")
if (
    not isinstance(rss_samples, list)
    or len(rss_samples) > 256
    or any(
        not isinstance(item, dict)
        or set(item) != {"elapsedSec", "rssAnonKiB"}
        for item in rss_samples
    )
    or not isinstance(errors, list)
    or not isinstance(warnings, list)
    or not isinstance(queue_levels, dict)
):
    raise SystemExit("IQ9075 evidence collection fields are invalid")

temporary = destination.with_name(
    f".{destination.name}.{os.getpid()}.{time.monotonic_ns()}.tmp"
)
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
try:
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
    finally:
        os.close(descriptor)
    os.link(temporary, destination, follow_symlinks=False)
    directory_descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
finally:
    temporary.unlink(missing_ok=True)
PY
    echo "[iq9075-e2e] canonical OAK evidence: $evidence_output"
  fi
  if [ "$oak_status" -ne 0 ]; then
    if [ "$allow_no_camera" = true ] && [ "$oak_status" -eq 3 ]; then
      echo "[iq9075-e2e] OAK-D camera: SKIP (no device; --allow-no-camera)"
      exit 0
    fi
    die "OAK-D RGB/appsrc/H264/RTP test failed (status=$oak_status); check runtime, USB permissions, configured MXID, camera health, and preserved evidence"
  fi
  echo "[iq9075-e2e] local OAK-D/GStreamer/WebRTC path: PASS"
  exit 0
fi

camera_path=""
camera_format=""
if [ -d /dev/v4l/by-id ]; then
  while IFS= read -r candidate; do
    [ -L "$candidate" ] || continue
    resolved="$(readlink -f "$candidate")"
    [ -c "$resolved" ] || continue
    driver="$(basename "$(readlink -f "/sys/class/video4linux/$(basename "$resolved")/device/driver" 2>/dev/null || true)")"
    [ "$driver" = "uvcvideo" ] || continue
    formats="$(v4l2-ctl --device "$candidate" --list-formats-ext 2>/dev/null || true)"
    case "$formats" in
      *"'MJPG'"*) camera_path="$candidate"; camera_format="mjpeg"; break ;;
      *"'YUYV'"*|*"'NV12'"*|*"'RGB3'"*) camera_path="$candidate"; camera_format="raw"; break ;;
    esac
  done < <(find /dev/v4l/by-id -maxdepth 1 -type l -print | LC_ALL=C sort)
fi

if [ -z "$camera_path" ]; then
  if [ "$allow_no_camera" = true ]; then
    echo "[iq9075-e2e] USB UVC capture camera: SKIP (--allow-no-camera)"
    exit 0
  fi
  die "no stable /dev/v4l/by-id USB UVC capture camera was found"
fi

if command -v fuser >/dev/null 2>&1 && fuser "$camera_path" >/dev/null 2>&1; then
  die "camera is already in use: $camera_path"
fi

echo "[iq9075-e2e] camera=$camera_path format=$camera_format"
v4l2-ctl --device "$camera_path" --all

echo "[iq9075-e2e] running bounded camera capture"
if [ "$camera_format" = "mjpeg" ]; then
  timeout 20s gst-launch-1.0 -q \
    v4l2src device="$camera_path" num-buffers=30 ! image/jpeg ! \
    jpegdec ! videoconvert ! fakesink sync=false
else
  timeout 20s gst-launch-1.0 -q \
    v4l2src device="$camera_path" num-buffers=30 ! video/x-raw ! \
    videoconvert ! fakesink sync=false
fi

echo "[iq9075-e2e] local USB camera/GStreamer/WebRTC construction: PASS"
