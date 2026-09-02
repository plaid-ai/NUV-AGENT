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

for command in gst-inspect-1.0 gst-launch-1.0 v4l2-ctl timeout python3 readlink; do
  command -v "$command" >/dev/null 2>&1 || die "$command is required"
done

echo "[iq9075-e2e] checking GStreamer runtime"
for element in v4l2src videoconvert jpegdec x264enc h264parse rtph264pay webrtcbin nicesrc; do
  gst-inspect-1.0 "$element" >/dev/null 2>&1 || die "missing GStreamer element: $element"
done

python3 <<'PY'
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

agent_python=/opt/nuv-agent/current/venv/bin/python
[ -x "$agent_python" ] || die "nuv-agent current slot venv is missing"

echo "[iq9075-e2e] checking disposable WebRTC branches across signaling resets"
G_DEBUG=fatal-criticals timeout 20s "$agent_python" <<'PY'
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
    send_message=lambda destination, payload, remember: (
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
  oak_python=("$agent_python")
  if [ "$(id -u)" -eq 0 ]; then
    command -v runuser >/dev/null 2>&1 || die "runuser is required for the non-root OAK access check"
    oak_python=(runuser -u nuvion -- /opt/nuv-agent/current/venv/bin/python)
  elif [ "$(id -un)" != "nuvion" ]; then
    die "run the OAK test with sudo so capture can be verified as the nuvion service user"
  fi

  echo "[iq9075-e2e] checking OAK-D capture as non-root user nuvion"
  set +e
  timeout 45s "${oak_python[@]}" <<'PY'
from importlib.metadata import version
from pathlib import Path

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

expected_usb_path = Path("/sys/bus/usb/devices/2-1")
if not expected_usb_path.is_dir():
    print("[iq9075-e2e] no OAK-D device detected at exact USB path 2-1", flush=True)
    raise SystemExit(3)

def read_sysfs(name: str) -> str:
    try:
        return (expected_usb_path / name).read_text(encoding="utf-8").strip().lower()
    except OSError as exc:
        raise SystemExit(f"cannot read OAK sysfs {name}: {exc}") from exc

vendor = read_sysfs("idVendor")
product = read_sysfs("idProduct")
speed = read_sysfs("speed")
try:
    driver = (expected_usb_path / "driver").resolve(strict=True).name
except OSError as exc:
    raise SystemExit(f"cannot resolve OAK USB driver: {exc}") from exc
if (vendor, product, driver) != ("03e7", "f63b", "usb"):
    raise SystemExit(
        "exact OAK USB identity mismatch at 2-1: "
        f"vendor={vendor}, product={product}, driver={driver}"
    )
try:
    speed_mbps = float(speed)
except ValueError as exc:
    raise SystemExit(f"invalid OAK USB speed at 2-1: {speed!r}") from exc
if speed_mbps < 5000.0:
    raise SystemExit(
        f"OAK-D Lite at 2-1 must negotiate 5Gbps; observed {speed_mbps:g}Mbps"
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
    "appsink name=rtp_sink max-buffers=30 drop=false sync=false"
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
try:
    bridge.start()
    for index in range(30):
        message = bus.pop_filtered(Gst.MessageType.ERROR)
        if message is not None:
            error, debug = message.parse_error()
            raise RuntimeError(f"GStreamer bus error: {error}; debug={debug}")
        sample = rtp_sink.emit("try-pull-sample", 2 * Gst.SECOND)
        if sample is None:
            raise RuntimeError(f"timed out waiting for RTP sample {index + 1}/30")
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
    "[iq9075-e2e] OAK-D Lite RGB/appsrc/H264/RTP: PASS "
    f"(depthai={installed_version}, samples=30)"
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
