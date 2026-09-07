#!/usr/bin/env bash
# Operator-staged development runtime; not a signed Fleet OTA release.
# User-approved experimental deployment; numerical parity remains FAILED.
# Requires an external deployment pin. No CPU inference fallback.
set -euo pipefail

scoped_directory() {
  local requested="$1" allowed_root="$2" label="$3" canonical
  if [[ ! "$requested" =~ ^/[A-Za-z0-9._/-]+$ || "$requested" != "$allowed_root"/* ]]; then
    printf '%s must be an absolute directory inside its deployment scope.\n' "$label" >&2
    return 1
  fi
  if [[ ! -d "$requested" || -L "$requested" ]]; then
    printf '%s must be an existing non-symlink directory.\n' "$label" >&2
    return 1
  fi
  canonical=$(cd -- "$requested" && pwd -P) || return 1
  if [[ "$canonical" != "$requested" ]]; then
    printf '%s must be canonical, without symlink components or traversal.\n' "$label" >&2
    return 1
  fi
  printf '%s\n' "$canonical"
}

[ "$(hostname)" = iq9075 ]
readonly fleet_enabled="${NUVION_VISUALAD_HTP_FLEET_ENABLED-false}"
case "$fleet_enabled" in
  true|false) ;;
  *) echo 'VisualAD Fleet opt-in must be exactly true or false.' >&2; exit 1 ;;
esac
# Explicit empty overrides are errors. Deployment-specific state variables avoid
# accidentally inheriting the base unit's unrelated Fleet settings directory.
demo_root=$(scoped_directory "${NUVION_VISUALAD_DEPLOYMENT_ROOT-/opt/nuvion-demo/20260910-visualad-htp}" /opt/nuvion-demo deployment)
htp_state_dir=$(scoped_directory "${NUVION_VISUALAD_DEPLOYMENT_HTP_STATE_DIR-/var/lib/nuv-agent/visualad-htp}" /var/lib/nuv-agent HTP-state)
settings_state_dir=$(scoped_directory "${NUVION_VISUALAD_DEPLOYMENT_SETTINGS_DIR-/var/lib/nuv-agent/visualad-htp-settings}" /var/lib/nuv-agent settings)
readonly demo_root htp_state_dir settings_state_dir
readonly baseline=/opt/nuv-agent/releases/26a7f1674bdd4a24bfe26fa37c681798244990408fe7d858ca76957a88bdb9f1
readonly demo_site="$demo_root/python/lib/python3.12/site-packages"
readonly baseline_site="$baseline/venv/lib/python3.12/site-packages"
readonly qnn_bundle="$demo_site/onnxruntime_qnn"

[ -f "$demo_root/src/nuvion_app/runtime/visualad_htp.py" ]
[ -f "$qnn_bundle/libQnnHtp.so" ]
if [ "$fleet_enabled" = true ]; then
  [ -n "${NUVION_VISUALAD_HTP_FLEET_STORE:-}" ]
  [ -n "${NUVION_MODEL_POINTER:-}" ]
  [[ "${NUVION_MODEL_DIGEST:-}" =~ ^sha256:[0-9a-f]{64}$ ]] || {
    echo 'VisualAD Fleet requires a canonical bootstrap wrapper digest.' >&2
    exit 1
  }
  [ -n "${NUVION_COMMAND_INBOX_PATH:-}" ]
else
  [ -f "$demo_root/model/manifest.json" ]
  [[ "${NUVION_VISUALAD_HTP_MANIFEST_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || {
    echo 'An external deployment manifest SHA256 pin is required.' >&2
    exit 1
  }
fi

export PYTHONPATH="$demo_root/src:$demo_site:$baseline_site:/usr/lib/python3/dist-packages"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONSAFEPATH=1
export NUV_AGENT_CONFIG=/etc/nuv-agent/agent.env
# Replace, never append the board's older QAIRT library search paths.
export LD_LIBRARY_PATH="$qnn_bundle" ADSP_LIBRARY_PATH="$qnn_bundle"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export NUVION_ZSAD_BACKEND=visualad_htp NUVION_ZERO_SHOT_ENABLED=true
export NUVION_VISUALAD_HTP_MANIFEST="$demo_root/model/manifest.json"
export NUVION_VISUALAD_HTP_STATE_DIR="$htp_state_dir"
export NUVION_VISUALAD_THRESHOLD="${NUVION_VISUALAD_THRESHOLD:-0.0}"
export NUVION_VISUALAD_EXPERIMENTAL=true
export NUVION_VISUALAD_VALIDATION_STATUS=PARITY_FAILED
export NUVION_VISUALAD_THRESHOLD_STATUS=UNCALIBRATED
# No artificial interval: single NPU worker consumes the newest available frame.
export NUVION_ZERO_SHOT_SAMPLE_SEC=0
export NUVION_DEMO_MODE=false NUVION_VIDEO_SOURCE=oak
export NUVION_FACE_TRACKING_ENABLED=false
export NUVION_FLEET_COMMAND_ENABLED="$fleet_enabled"
export NUVION_SETTINGS_STATE_DIR="$settings_state_dir"
if [ "$fleet_enabled" = false ]; then
  export NUVION_MODEL_POINTER=visualad/iq9075-htp-demo
  export NUVION_MODEL_VERSION=visualad-visa-97eb5f88a44f-htp
  export NUVION_MODEL_LOCAL_DIR="$demo_root/model"
fi
export NUVION_AGENT_VERSION=0.1.121+iq9075.visualad.htp.exp3
export NUVION_COMPONENT_SHA=unknown
export NUVION_RELEASE_BOM_PATH="" NUVION_EXPECTED_BOM_DIGEST=""
export NUVION_BOM_ID=development-visualad-htp-not-a-release
export NUVION_BOM_DIGEST="" NUVION_ARTIFACT_DIGEST="" NUVION_ACTIVE_SLOT=""

exec /usr/bin/python3 -s -c '
import os, runpy, sys
from pathlib import Path
bundle = sys.argv[3]
assert os.environ["LD_LIBRARY_PATH"] == bundle
assert os.environ["ADSP_LIBRARY_PATH"] == bundle
if sys.argv[5] == "true" and sys.argv[4] != "--import-check":
    from nuvion_app.runtime.visualad_fleet_start import prepare_fleet_startup
    prepare_fleet_startup()
from nuvion_app.inference import pipeline
from nuvion_app.runtime import visualad_htp
import depthai, gi, cv2, numpy, onnx, onnxruntime, onnxruntime_qnn, scipy
demo_root = Path(sys.argv[1]).resolve(strict=True)
baseline_site = Path(sys.argv[2]).resolve(strict=True)
demo_site = demo_root / "python/lib/python3.12/site-packages"
for module, expected in (
    (pipeline, demo_root / "src"), (visualad_htp, demo_root / "src"),
    (numpy, demo_site), (cv2, demo_site), (onnx, demo_site),
    (onnxruntime, demo_site), (onnxruntime_qnn, demo_site), (scipy, demo_site),
    (depthai, baseline_site), (gi, Path("/usr/lib/python3/dist-packages")),
):
    assert Path(module.__file__).resolve(strict=True).is_relative_to(expected), (
        "Unexpected module origin: " + module.__name__
    )
assert "torch" not in sys.modules, "HTP startup must not import PyTorch"
assert onnxruntime.__version__ == "1.26.0"
assert onnxruntime_qnn.__version__ == "2.5.0"
# OpenCV 5 wheel prepends its own lib64 when imported. Load the pinned wheel
# first, reject other mutations, then restore the process-scoped QNN-only path.
# All libraries are still checked against the QNN bundle by the HTP adapter.
cv2_library_prefix = str(demo_site / "cv2/../../lib64")
assert os.environ["LD_LIBRARY_PATH"] in (bundle, cv2_library_prefix + ":" + bundle)
assert os.environ["ADSP_LIBRARY_PATH"] == bundle
os.environ["LD_LIBRARY_PATH"] = bundle
print("HTP_RUNTIME_IMPORT_OK", onnxruntime.__version__, onnxruntime_qnn.__version__, cv2.__version__)
if sys.argv[4] != "--import-check":
    sys.argv = ["nuvion_app.inference.main"]
    runpy.run_module("nuvion_app.inference.main", run_name="__main__")
' "$demo_root" "$baseline_site" "$qnn_bundle" "${1:-}" "$fleet_enabled"
