#!/usr/bin/env bash
# IQ9075 development-only runtime. This is NOT a signed OTA release.
set -euo pipefail

readonly demo_root=/opt/nuvion-demo/20260910-visualad
readonly baseline=/opt/nuv-agent/releases/26a7f1674bdd4a24bfe26fa37c681798244990408fe7d858ca76957a88bdb9f1
readonly demo_site="$demo_root/python/lib/python3.12/site-packages"
readonly baseline_site="$baseline/venv/lib/python3.12/site-packages"

[ -f "$demo_root/src/nuvion_app/runtime/visualad.py" ]
[ -d "$demo_site/torch" ]
[ -d "$baseline_site/depthai" ] || [ -f "$baseline_site/depthai.cpython-312-aarch64-linux-gnu.so" ]
[ -f "$demo_root/weights/open_clip_model.safetensors" ]

export PYTHONPATH="$demo_root/src:$demo_site:$baseline_site:/usr/lib/python3/dist-packages"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export NUV_AGENT_CONFIG=/etc/nuv-agent/agent.env
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
# CPU is an explicitly requested reference mode only; production target is HTP.
# Keep video available while the HTP artifact is being prepared.
export NUVION_ZSAD_BACKEND=visualad
export NUVION_ZERO_SHOT_ENABLED="${NUVION_VISUALAD_CPU_REFERENCE_ENABLED:-false}"
export NUVION_VISUALAD_SOURCE="$demo_root/external/VisualAD"
export NUVION_VISUALAD_BACKBONE="$demo_root/weights/open_clip_model.safetensors"
export NUVION_VISUALAD_CHECKPOINT="$demo_root/weights/visualad_train_on_visa_CLIP.pth"
export NUVION_VISUALAD_CPU_THREADS=2
export NUVION_ZERO_SHOT_SAMPLE_SEC=2
export NUVION_VISUALAD_THRESHOLD="${NUVION_VISUALAD_THRESHOLD:-0.0}"
export NUVION_DEMO_MODE=false NUVION_VIDEO_SOURCE=oak
export NUVION_FACE_TRACKING_ENABLED=false
# Preserve the released settings/OTA journals. A demo cannot attest a signed
# release or accept a Fleet update while executing a different source tree.
export NUVION_FLEET_COMMAND_ENABLED=false
export NUVION_SETTINGS_STATE_DIR=/var/lib/nuv-agent/visualad-demo-settings
export NUVION_MODEL_POINTER=visualad/iq9075-demo
export NUVION_MODEL_VERSION=visualad-visa-97eb5f88a44f
export NUVION_MODEL_LOCAL_DIR="$demo_root/weights"
export NUVION_AGENT_VERSION=0.1.121+iq9075.visualad.htp-prep1
export NUVION_COMPONENT_SHA=unknown
export NUVION_RELEASE_BOM_PATH= NUVION_EXPECTED_BOM_DIGEST=
export NUVION_BOM_ID=development-visualad-not-a-release
export NUVION_BOM_DIGEST= NUVION_ARTIFACT_DIGEST= NUVION_ACTIVE_SLOT=

if [ "${1:-}" = "--import-check" ]; then
  exec /usr/bin/python3 -s -c 'from nuvion_app.inference import pipeline; import depthai, gi, torch, scipy, safetensors; print("DEMO_RUNTIME_IMPORT_OK", torch.__version__, scipy.__version__)'
fi
exec /usr/bin/python3 -s -m nuvion_app.inference.main
