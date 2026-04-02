from __future__ import annotations

import atexit
import logging
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from nuvion_app.runtime.docker_manager import (
    container_exists,
    container_running,
    ensure_docker_ready,
    parse_triton_host_port,
    remove_container,
    run_triton_container,
    start_container,
    stop_container,
)
from nuvion_app.runtime.errors import BootstrapError
from nuvion_app.runtime.inference_mode import face_tracking_uses_triton, normalize_backend

log = logging.getLogger(__name__)
_managed_triton_container: str | None = None
_atexit_registered = False


_FALLBACK_CONFIG = """name: \"image_encoder\"
platform: \"onnxruntime_onnx\"
max_batch_size: 0
instance_group [
  {
    kind: KIND_CPU
    count: 2
  }
]
"""


def _ultraface_box_count(width: int, height: int) -> int:
    min_boxes = ((10.0, 16.0, 24.0), (32.0, 48.0), (64.0, 96.0), (128.0, 192.0, 256.0))
    strides = (8.0, 16.0, 32.0, 64.0)
    total = 0
    for min_sizes, stride in zip(min_boxes, strides):
        feature_map_w = int((width + stride - 1) // stride)
        feature_map_h = int((height + stride - 1) // stride)
        total += feature_map_w * feature_map_h * len(min_sizes)
    return total


def _default_face_tracking_config(platform: str) -> str:
    model_name = (os.getenv("NUVION_FACE_TRACKING_MODEL", "face_detector") or "face_detector").strip() or "face_detector"
    input_name = (os.getenv("NUVION_FACE_TRACKING_INPUT_NAME", "input") or "input").strip() or "input"
    boxes_output = (os.getenv("NUVION_FACE_TRACKING_BOXES_OUTPUT", "boxes") or "boxes").strip() or "boxes"
    scores_output = (os.getenv("NUVION_FACE_TRACKING_SCORES_OUTPUT", "scores") or "scores").strip() or "scores"
    num_output = (os.getenv("NUVION_FACE_TRACKING_NUM_DETECTIONS_OUTPUT", "") or "").strip()
    model_kind = (os.getenv("NUVION_FACE_TRACKING_MODEL_KIND", "ultraface_rfb_640") or "ultraface_rfb_640").strip().lower()
    width = max(int(os.getenv("NUVION_FACE_TRACKING_INPUT_WIDTH", "640") or "640"), 1)
    height = max(int(os.getenv("NUVION_FACE_TRACKING_INPUT_HEIGHT", "480") or "480"), 1)
    instance_kind = "KIND_CPU" if platform == "onnxruntime_onnx" else "KIND_GPU"
    use_fixed_ultraface_shapes = model_kind.startswith("ultraface")

    if use_fixed_ultraface_shapes:
        input_block = f"""  {{
    name: "{input_name}"
    data_type: TYPE_FP32
    dims: [ 1, 3, {height}, {width} ]
  }}"""
        box_count = _ultraface_box_count(width, height)
        output_blocks = [
            f"""  {{
    name: "{scores_output}"
    data_type: TYPE_FP32
    dims: [ 1, {box_count}, 2 ]
  }}""",
            f"""  {{
    name: "{boxes_output}"
    data_type: TYPE_FP32
    dims: [ 1, {box_count}, 4 ]
  }}""",
        ]
    else:
        input_block = f"""  {{
    name: "{input_name}"
    data_type: TYPE_FP32
    dims: [ 3, {height}, {width} ]
    format: FORMAT_NCHW
  }}"""
        output_blocks = [
            f"""  {{
    name: "{boxes_output}"
    data_type: TYPE_FP32
    dims: [ -1, 4 ]
  }}""",
            f"""  {{
    name: "{scores_output}"
    data_type: TYPE_FP32
    dims: [ -1 ]
  }}""",
        ]

    if num_output and not use_fixed_ultraface_shapes:
        output_blocks.append(
            f"""  {{
    name: "{num_output}"
    data_type: TYPE_INT32
    dims: [ 1 ]
  }}"""
        )
    outputs = ",\n".join(output_blocks)
    return f"""name: "{model_name}"
platform: "{platform}"
max_batch_size: 0
input [
{input_block}
]
output [
{outputs}
]
instance_group [
  {{
    kind: {instance_kind}
    count: 1
  }}
]
"""


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _emit_progress(message: str) -> None:
    sys.stderr.write(f"[BOOTSTRAP] {message}\n")
    sys.stderr.flush()


def _is_darwin() -> bool:
    return os.uname().sysname.lower() == "darwin"


def _is_raspberry_pi_linux() -> bool:
    try:
        if os.uname().sysname.lower() != "linux":
            return False
    except Exception:
        return False

    probe_paths = (
        Path("/proc/device-tree/model"),
        Path("/sys/firmware/devicetree/base/model"),
        Path("/proc/device-tree/compatible"),
        Path("/sys/firmware/devicetree/base/compatible"),
    )
    for probe_path in probe_paths:
        try:
            content = probe_path.read_text(encoding="utf-8", errors="ignore").replace("\x00", " ").lower()
        except OSError:
            continue
        if "raspberry pi" in content or "raspberrypi" in content:
            return True
    return False


def _is_jetson_linux() -> bool:
    try:
        return os.uname().sysname.lower() == "linux" and Path("/etc/nv_tegra_release").exists()
    except Exception:
        return False


def _should_use_onnx_repository() -> bool:
    return _is_darwin() or _is_raspberry_pi_linux()


def _should_autostop() -> bool:
    return _truthy(os.getenv("NUVION_TRITON_AUTOSTOP_ON_EXIT"), default=True)


def _register_managed_triton_container(container_name: str) -> None:
    global _managed_triton_container
    global _atexit_registered
    if not _should_autostop():
        return
    _managed_triton_container = container_name
    if not _atexit_registered:
        atexit.register(cleanup_managed_triton, "process_exit")
        _atexit_registered = True


def cleanup_managed_triton(reason: str = "agent_exit") -> None:
    global _managed_triton_container
    container_name = _managed_triton_container
    if not container_name:
        return
    _managed_triton_container = None

    if not _should_autostop():
        return

    try:
        if not container_exists(container_name):
            return
        if container_running(container_name):
            _emit_progress(f"Triton 컨테이너 자동 종료: {container_name} (reason={reason})")
            stop_container(container_name)
            log.info("[BOOTSTRAP] Stopped managed Triton container '%s' (reason=%s)", container_name, reason)
    except Exception as exc:
        log.warning("[BOOTSTRAP] Failed to stop managed Triton container '%s': %s", container_name, exc)


def _health_ready(host: str, port: int, timeout_sec: int) -> bool:
    url = f"http://{host}:{port}/v2/health/ready"
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if 200 <= response.getcode() < 300:
                    return True
        except urllib.error.URLError:
            pass
        except Exception:
            pass
        time.sleep(1)
    return False


def _model_ready(host: str, port: int, model_name: str, timeout_sec: int) -> bool:
    url = f"http://{host}:{port}/v2/models/{model_name}/ready"
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if 200 <= response.getcode() < 300:
                    return True
        except urllib.error.HTTPError:
            return False
        except urllib.error.URLError:
            pass
        except Exception:
            pass
        time.sleep(1)
    return False


def _required_models() -> list[str]:
    models: list[str] = []
    if normalize_backend(os.getenv("NUVION_ZSAD_BACKEND", "triton"), default="triton") == "triton":
        models.append("image_encoder")
    if face_tracking_uses_triton():
        model_name = (os.getenv("NUVION_FACE_TRACKING_MODEL", "face_detector") or "face_detector").strip() or "face_detector"
        if model_name not in models:
            models.append(model_name)
    return models


def _required_models_ready(host: str, port: int, timeout_sec: int) -> bool:
    for model_name in _required_models():
        if not _model_ready(host, port, model_name, timeout_sec=timeout_sec):
            return False
    return True


def _copy_if_needed(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return
    shutil.copy2(src, dst)


def _write_face_detector_config_if_missing(target_config: Path, platform: str, config_src: Path | None = None) -> None:
    target_config.parent.mkdir(parents=True, exist_ok=True)
    if (
        config_src is not None
        and config_src.exists()
        and config_src.resolve() != target_config.resolve()
    ):
        _copy_if_needed(config_src, target_config)
        return
    target_config.write_text(_default_face_tracking_config(platform))


def _build_face_detector_trt_plan(onnx_src: Path, plan_dst: Path) -> bool:
    trtexec = shutil.which("trtexec")
    if not trtexec:
        log.warning("[TRACK] trtexec not found. Falling back to ONNX face detector runtime.")
        return False

    plan_dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        trtexec,
        f"--onnx={onnx_src}",
        f"--saveEngine={plan_dst}",
        "--skipInference",
    ]
    if _truthy(os.getenv("NUVION_FACE_TRACKING_TRT_FP16"), default=True):
        cmd.append("--fp16")

    extra_args = (os.getenv("NUVION_FACE_TRACKING_TRTEXEC_ARGS") or "").strip()
    if extra_args:
        cmd.extend(extra_args.split())

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        log.info("[TRACK] Built TensorRT face detector plan: %s", plan_dst)
        return True
    except Exception as exc:
        log.warning("[TRACK] Failed to build TensorRT face detector plan: %s", exc)
        return False


def _ensure_face_detector_repository(model_dir: Path, repository_root: Path, *, prefer_tensorrt: bool) -> None:
    if not face_tracking_uses_triton():
        return

    config_src = model_dir / "triton" / "model_repository" / "face_detector" / "config.pbtxt"
    model_repo = repository_root / "face_detector" / "1"
    model_repo.mkdir(parents=True, exist_ok=True)
    onnx_src = model_dir / "onnx" / "face_detector.onnx"
    plan_path = model_repo / "model.plan"
    packaged_plan = model_dir / "triton" / "model_repository" / "face_detector" / "1" / "model.plan"

    if prefer_tensorrt:
        if packaged_plan.exists():
            _copy_if_needed(packaged_plan, plan_path)
            log.info("[TRACK] Using packaged TensorRT face detector plan: %s", packaged_plan)
        elif not plan_path.exists() and onnx_src.exists() and not _build_face_detector_trt_plan(onnx_src, plan_path):
            _copy_if_needed(onnx_src, model_repo / "model.onnx")
            _write_face_detector_config_if_missing(
                model_repo.parent / "config.pbtxt",
                "onnxruntime_onnx",
                None,
            )
            log.warning("[TRACK] Falling back to ONNXRuntime face detector on Jetson because TensorRT plan build failed.")
            return
        elif not plan_path.exists():
            raise BootstrapError(
                "triton_health_failed",
                f"Missing face tracking TensorRT artifacts: {packaged_plan} (and no ONNX fallback at {onnx_src})",
            )

        _write_face_detector_config_if_missing(
            model_repo.parent / "config.pbtxt",
            "tensorrt_plan",
            config_src if config_src.exists() else None,
        )
        return

    if not onnx_src.exists():
        raise BootstrapError(
            "triton_health_failed",
            f"Missing face tracking ONNX model: {onnx_src}",
        )
    _copy_if_needed(onnx_src, model_repo / "model.onnx")
    _write_face_detector_config_if_missing(
        model_repo.parent / "config.pbtxt",
        "onnxruntime_onnx",
        None,
    )


def _ensure_cpu_onnx_repository(model_dir: Path, repository_root: Path) -> Path:
    model_repo = repository_root / "image_encoder" / "1"
    model_repo.mkdir(parents=True, exist_ok=True)

    onnx_src = model_dir / "onnx" / "image_encoder_simplified.onnx"
    if not onnx_src.exists():
        raise BootstrapError(
            "triton_health_failed",
            f"Missing ONNX model for CPU fallback: {onnx_src}",
        )

    target_onnx = model_repo / "model.onnx"
    if not target_onnx.exists() or target_onnx.stat().st_size != onnx_src.stat().st_size:
        shutil.copy2(onnx_src, target_onnx)

    target_config_dir = repository_root / "image_encoder"
    target_config_dir.mkdir(parents=True, exist_ok=True)
    target_config = target_config_dir / "config.pbtxt"
    # CPU-only fallback uses ONNXRuntime config to avoid TensorRT(GPU-only) bootstrap failure.
    target_config.write_text(_FALLBACK_CONFIG)
    _ensure_face_detector_repository(model_dir=model_dir, repository_root=repository_root, prefer_tensorrt=False)

    return repository_root


def resolve_repository_for_runtime(model_dir: Path) -> Path:
    default_repo = model_dir / "triton" / "model_repository"
    if not _should_use_onnx_repository():
        if not default_repo.exists():
            if normalize_backend(os.getenv("NUVION_ZSAD_BACKEND", "triton"), default="triton") == "triton":
                raise BootstrapError("triton_health_failed", f"Triton model repository is missing: {default_repo}")
            default_repo.mkdir(parents=True, exist_ok=True)
        _ensure_face_detector_repository(
            model_dir=model_dir,
            repository_root=default_repo,
            prefer_tensorrt=_is_jetson_linux(),
        )
        return default_repo

    fallback = model_dir / "triton" / "model_repository_onnx"
    return _ensure_cpu_onnx_repository(model_dir=model_dir, repository_root=fallback)


def ensure_triton_ready(stage: str, model_dir: Path) -> None:
    backend = normalize_backend(os.getenv("NUVION_ZSAD_BACKEND", "triton"), default="triton")
    if backend != "triton" and not face_tracking_uses_triton():
        return

    if not _truthy(os.getenv("NUVION_TRITON_AUTOSTART"), default=True):
        return

    triton_url = os.getenv("NUVION_TRITON_URL", "localhost:8000")
    host, port = parse_triton_host_port(triton_url)

    local_only = _truthy(os.getenv("NUVION_TRITON_AUTOSTART_ONLY_LOCAL"), default=True)
    if local_only and host not in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}:
        return

    repository = resolve_repository_for_runtime(model_dir).resolve()

    if _health_ready(host, port, timeout_sec=3):
        if _required_models_ready(host, port, timeout_sec=3):
            _emit_progress(f"Triton 이미 준비됨: {host}:{port}")
            return
        _emit_progress("Triton은 응답하지만 필요한 모델이 없어 컨테이너를 다시 올립니다.")
        log.warning("[TRACK] Triton is healthy but required models are missing. Reloading container.")

    _emit_progress("Triton이 준비되지 않아 Docker/Triton 자동 복구를 시작합니다.")
    ensure_docker_ready(triton_url)

    container_name = os.getenv("NUVION_TRITON_CONTAINER_NAME", "triton-nuv").strip() or "triton-nuv"
    image = os.getenv("NUVION_TRITON_IMAGE", "nvcr.io/nvidia/tritonserver:24.10-py3").strip() or "nvcr.io/nvidia/tritonserver:24.10-py3"

    if container_exists(container_name):
        _emit_progress(f"기존 Triton 컨테이너 확인: {container_name}")
        if container_running(container_name):
            if _health_ready(host, port, timeout_sec=5) and _required_models_ready(host, port, timeout_sec=5):
                _emit_progress("기존 Triton 컨테이너 재사용 성공")
                return
            remove_container(container_name)
        else:
            start_container(container_name)
            if _health_ready(host, port, timeout_sec=10) and _required_models_ready(host, port, timeout_sec=10):
                _emit_progress("중지된 Triton 컨테이너 재기동 성공")
                _register_managed_triton_container(container_name)
                return
            remove_container(container_name)

    _emit_progress(f"Triton 컨테이너 생성 중: {container_name}")
    run_triton_container(
        name=container_name,
        image=image,
        model_repository=str(repository),
        host_port=port,
    )

    timeout_sec = int(os.getenv("NUVION_TRITON_BOOT_TIMEOUT_SEC", "40"))
    if not _health_ready(host, port, timeout_sec=timeout_sec):
        raise BootstrapError(
            "triton_health_failed",
            f"Triton health check failed at http://{host}:{port}/v2/health/ready",
        )
    if not _required_models_ready(host, port, timeout_sec=10):
        missing = ", ".join(_required_models()) or "unknown"
        raise BootstrapError(
            "triton_health_failed",
            f"Triton started but required models are not ready: {missing}",
        )
    _register_managed_triton_container(container_name)

    log.info("[BOOTSTRAP] Triton is ready (stage=%s, url=%s)", stage, triton_url)
    _emit_progress(f"Triton 준비 완료: {host}:{port}")
