#!/usr/bin/env bash
set -euo pipefail

allow_no_camera=false
camera_mode="oak"
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
    *)
      echo "Usage: $0 [--camera oak|uvc] [--allow-no-camera]" >&2
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
probe_environment=(
  "HOME=$probe_runtime_dir/home"
  "XDG_CACHE_HOME=$probe_runtime_dir/cache"
  "XDG_CONFIG_HOME=$probe_runtime_dir/config"
  "XDG_RUNTIME_DIR=$probe_runtime_dir/runtime"
)

webrtc_python=(
  /usr/bin/env
  "${probe_environment[@]}"
  "PYTHONPATH=${PYTHONPATH:-}"
  "G_DEBUG=fatal-criticals"
  "$agent_python"
)
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

gi.require_version("Gst", "1.0")
gi.require_version("GstWebRTC", "1.0")
gi.require_version("GstSdp", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

from nuvion_app.inference.webrtc_uplink import WebRTCUplinkController

Gst.init(None)
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


def start(session_id):
    controller.start(
        {
            "broadcastId": "iq9075-release-test",
            "sessionId": session_id,
            "iceServers": [],
        }
    )
    return False


def reset():
    if controller._branch is not None:
        branch_names.append(controller._branch.webrtcbin.get_name())
    controller.on_signaling_reset()
    return False


GLib.timeout_add(300, reset)
GLib.timeout_add(450, start, "session-2")
GLib.timeout_add(750, reset)
GLib.timeout_add(900, start, "session-3")
GLib.timeout_add(1300, lambda: (loop.quit(), False)[1])
start("session-1")
loop.run()

health = controller.runtime_health_snapshot()
if controller._branch is not None:
    branch_names.append(controller._branch.webrtcbin.get_name())
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
if health.get("sessionId") != "session-3" or len(offers) != 3:
    raise RuntimeError(f"WebRTC re-offer evidence is incomplete: health={health} offers={offers}")
if issues:
    raise RuntimeError(f"WebRTC reset emitted GStreamer bus failures: {issues}")
print("[iq9075-e2e] disposable WebRTC reset/re-offer: PASS")
PY

if [ "$camera_mode" = "oak" ]; then
  oak_python=(
    /usr/bin/env
    "${probe_environment[@]}"
    "PYTHONPATH=${PYTHONPATH:-}"
    "NUVION_IQ9075_OAK_SOAK_SECONDS=${NUVION_IQ9075_OAK_SOAK_SECONDS:-120}"
    "NUVION_IQ9075_OAK_MAX_RSS_GROWTH_MIB=${NUVION_IQ9075_OAK_MAX_RSS_GROWTH_MIB:-96}"
    "$agent_python"
  )
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
  timeout 660s "${oak_python[@]}" <<'PY'
from importlib.metadata import version
from pathlib import Path
import os
import re
import time

import depthai
from dotenv import dotenv_values

from nuvion_app.inference.depthai_gst import DepthAIGStreamerBridge
from nuvion_app.inference.depthai_source import DepthAIConfig, DepthAIFrameSource
from nuvion_app.inference.video_source import DEPTHAI_APPSRC_NAME
from nuvion_app.inference.video_source import build_video_source_pipeline
from nuvion_app.inference.video_source import resolve_depthai_device_id

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

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
usb_topology_re = re.compile(r"^2-1(?:\.[1-9][0-9]*)+$")
sys_root = Path("/sys").resolve(strict=True)


def read_sysfs(device: Path, name: str) -> str:
    try:
        return (device / name).read_text(encoding="utf-8").strip().lower()
    except OSError as exc:
        raise SystemExit(f"cannot read OAK sysfs {name}: {exc}") from exc


oak_usb_paths = []
for candidate in sorted(usb_devices_root.iterdir(), key=lambda item: item.name):
    if usb_topology_re.fullmatch(candidate.name) is None:
        continue
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise SystemExit(f"cannot resolve USB1 downstream topology: {exc}") from exc
    if not resolved.is_dir() or not resolved.is_relative_to(sys_root):
        raise SystemExit("USB1 downstream topology escaped sysfs")
    vendor_path = candidate / "idVendor"
    product_path = candidate / "idProduct"
    if not vendor_path.is_file() or not product_path.is_file():
        continue
    if (
        read_sysfs(candidate, "idVendor"),
        read_sysfs(candidate, "idProduct"),
    ) == ("03e7", "f63b"):
        oak_usb_paths.append(candidate)

if not oak_usb_paths:
    print("[iq9075-e2e] no OAK-D device detected below USB1 hub 2-1", flush=True)
    raise SystemExit(3)
if len(oak_usb_paths) != 1:
    raise SystemExit("USB1 downstream must contain exactly one 03e7:f63b OAK-D Lite")
expected_usb_path = oak_usb_paths[0]
vendor = read_sysfs(expected_usb_path, "idVendor")
product = read_sysfs(expected_usb_path, "idProduct")
speed = read_sysfs(expected_usb_path, "speed")
try:
    driver_path = (expected_usb_path / "driver").resolve(strict=True)
    expected_driver = Path("/sys/bus/usb/drivers/usb").resolve(strict=True)
except OSError as exc:
    raise SystemExit(f"cannot resolve OAK USB driver: {exc}") from exc
if (vendor, product) != ("03e7", "f63b") or driver_path != expected_driver:
    raise SystemExit(
        f"exact OAK USB identity mismatch at {expected_usb_path.name}: "
        f"vendor={vendor}, product={product}, driver={driver_path.name}"
    )
try:
    speed_mbps = float(speed)
except ValueError as exc:
    raise SystemExit(
        f"invalid OAK USB speed at {expected_usb_path.name}: {speed!r}"
    ) from exc
if speed_mbps < 5000.0:
    raise SystemExit(
        f"OAK-D Lite at {expected_usb_path.name} must negotiate 5Gbps; "
        f"observed {speed_mbps:g}Mbps"
    )

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
if device_id is not None and device_id not in available_ids:
    raise SystemExit("configured OAK MXID is not attached")

startup_timeout = float(
    values.get("NUVION_DEPTHAI_STARTUP_TIMEOUT_SEC") or "15"
)
read_timeout = float(values.get("NUVION_DEPTHAI_READ_TIMEOUT_SEC") or "2")
soak_seconds = int(os.getenv("NUVION_IQ9075_OAK_SOAK_SECONDS", "120"))
max_rss_growth_mib = int(
    os.getenv("NUVION_IQ9075_OAK_MAX_RSS_GROWTH_MIB", "96")
)
if soak_seconds < 60 or soak_seconds > 600:
    raise SystemExit("NUVION_IQ9075_OAK_SOAK_SECONDS must be in [60, 600]")
if max_rss_growth_mib < 16 or max_rss_growth_mib > 512:
    raise SystemExit("NUVION_IQ9075_OAK_MAX_RSS_GROWTH_MIB must be in [16, 512]")

config = DepthAIConfig(
    width=640,
    height=480,
    fps=30,
    device_id=device_id,
    queue_size=1,
    startup_timeout=startup_timeout,
    read_timeout=read_timeout,
)

Gst.init(None)
source_pipeline = build_video_source_pipeline(video_source, 640, 480, 30, platform_name="linux")
pipeline = Gst.parse_launch(
    f"{source_pipeline} ! "
    "videoconvert ! video/x-raw,format=I420 ! "
    "x264enc tune=zerolatency speed-preset=ultrafast key-int-max=30 bframes=0 ! "
    "h264parse ! rtph264pay config-interval=-1 pt=96 ! "
    "appsink name=rtp_sink max-buffers=2 drop=true sync=false"
)
appsrc = pipeline.get_by_name(DEPTHAI_APPSRC_NAME)
rtp_sink = pipeline.get_by_name("rtp_sink")
if appsrc is None or rtp_sink is None:
    raise SystemExit("DepthAI appsrc or RTP appsink is missing")

source = DepthAIFrameSource(config)
bridge = DepthAIGStreamerBridge(
    frame_source=source,
    appsrc=appsrc,
    gst=Gst,
    width=640,
    height=480,
    read_timeout=read_timeout,
    max_consecutive_timeouts=3,
)
bus = pipeline.get_bus()
state_result = pipeline.set_state(Gst.State.PLAYING)
if state_result == Gst.StateChangeReturn.FAILURE:
    pipeline.set_state(Gst.State.NULL)
    raise SystemExit("OAK GStreamer pipeline failed to enter PLAYING")

last_pts = None
sample_count = 0


def read_proc_status_kib(field: str) -> int:
    prefix = f"{field}:"
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            parts = line.split()
            if len(parts) == 3 and parts[2] == "kB" and parts[1].isdigit():
                return int(parts[1])
    raise RuntimeError(f"/proc/self/status is missing exact {field} kB evidence")


def appsrc_level(name: str) -> int | None:
    if appsrc.find_property(name) is None:
        return None
    return int(appsrc.get_property(name))


def pull_and_validate_rtp(label: str) -> None:
    global last_pts, sample_count
    sample = rtp_sink.emit("try-pull-sample", 2 * Gst.SECOND)
    if sample is None:
        raise RuntimeError(f"timed out waiting for RTP sample {label}")
    caps = sample.get_caps()
    structure = caps.get_structure(0) if caps is not None else None
    if structure is None or structure.get_name() != "application/x-rtp":
        raise RuntimeError(f"unexpected RTP caps: {caps}")
    if structure.get_string("encoding-name") != "H264":
        raise RuntimeError(f"unexpected RTP encoding caps: {caps}")
    buffer = sample.get_buffer()
    if buffer is None or buffer.pts == Gst.CLOCK_TIME_NONE:
        raise RuntimeError("RTP sample is missing a PTS")
    if last_pts is not None and buffer.pts < last_pts:
        raise RuntimeError("RTP sample PTS regressed")
    last_pts = buffer.pts
    sample_count += 1

    level_buffers = appsrc_level("current-level-buffers")
    level_bytes = appsrc_level("current-level-bytes")
    if level_buffers is not None and level_buffers > 2:
        raise RuntimeError(f"appsrc buffer bound exceeded: {level_buffers} > 2")
    if level_bytes is not None and level_bytes > 2 * 640 * 480 * 3:
        raise RuntimeError(
            f"appsrc byte bound exceeded: {level_bytes} > {2 * 640 * 480 * 3}"
        )


try:
    bridge.start()
    for index in range(30):
        message = bus.pop_filtered(Gst.MessageType.ERROR)
        if message is not None:
            error, debug = message.parse_error()
            raise RuntimeError(f"GStreamer bus error: {error}; debug={debug}")
        pull_and_validate_rtp(f"{index + 1}/30")

    soak_started = time.monotonic()
    warmup_seconds = min(30.0, max(10.0, soak_seconds / 4.0))
    baseline_at = soak_started + warmup_seconds
    soak_deadline = soak_started + soak_seconds
    baseline_rss_anon_kib = None
    max_rss_anon_kib = 0
    while time.monotonic() < soak_deadline:
        pull_and_validate_rtp(f"soak-{sample_count + 1}")
        now = time.monotonic()
        rss_anon_kib = read_proc_status_kib("RssAnon")
        if now >= baseline_at and baseline_rss_anon_kib is None:
            baseline_rss_anon_kib = rss_anon_kib
        if baseline_rss_anon_kib is not None:
            max_rss_anon_kib = max(max_rss_anon_kib, rss_anon_kib)

    if baseline_rss_anon_kib is None:
        raise RuntimeError("OAK soak did not reach its RSS baseline boundary")
    rss_growth_kib = max(0, max_rss_anon_kib - baseline_rss_anon_kib)
    if rss_growth_kib > max_rss_growth_mib * 1024:
        raise RuntimeError(
            "OAK DepthAI/appsrc anonymous RSS growth exceeded bound: "
            f"growth={rss_growth_kib / 1024:.1f}MiB "
            f"limit={max_rss_growth_mib}MiB duration={soak_seconds}s"
        )
    if bridge.failure is not None:
        raise RuntimeError(f"DepthAI bridge failed: {bridge.failure}")
    message = bus.pop_filtered(Gst.MessageType.ERROR)
    if message is not None:
        error, debug = message.parse_error()
        raise RuntimeError(f"GStreamer bus error: {error}; debug={debug}")
finally:
    bridge.close()
    pipeline.set_state(Gst.State.NULL)

print(
    "[iq9075-e2e] OAK-D Lite RGB/appsrc/H264/RTP bounded soak: PASS "
    f"(depthai={installed_version}, samples={sample_count}, "
    f"duration={soak_seconds}s, rssAnonGrowth={rss_growth_kib / 1024:.1f}MiB, "
    f"bridge={bridge.stats_snapshot()})"
)
PY
  oak_status=$?
  set -e
  if [ "$oak_status" -ne 0 ]; then
    if [ "$allow_no_camera" = true ] && [ "$oak_status" -eq 3 ]; then
      echo "[iq9075-e2e] OAK-D camera: SKIP (no device; --allow-no-camera)"
      exit 0
    fi
    die "OAK-D RGB/appsrc/H264/RTP test failed (status=$oak_status); check runtime, USB permissions, configured MXID, and camera health"
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
