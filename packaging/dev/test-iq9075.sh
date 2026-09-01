#!/usr/bin/env bash
set -euo pipefail

allow_no_camera=false
if [ "${1:-}" = "--allow-no-camera" ]; then
  allow_no_camera=true
  shift
fi
if [ "$#" -ne 0 ]; then
  echo "Usage: $0 [--allow-no-camera]" >&2
  exit 2
fi

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
