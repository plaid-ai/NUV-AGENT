from __future__ import annotations

import atexit
import logging
import os
import shutil
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
from nuvion_app.runtime.inference_mode import normalize_backend

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

    return repository_root


def resolve_repository_for_runtime(model_dir: Path) -> Path:
    default_repo = model_dir / "triton" / "model_repository"
    if not _should_use_onnx_repository():
        if not default_repo.exists():
            raise BootstrapError("triton_health_failed", f"Triton model repository is missing: {default_repo}")
        return default_repo

    fallback = model_dir / "triton" / "model_repository_onnx"
    return _ensure_cpu_onnx_repository(model_dir=model_dir, repository_root=fallback)


def ensure_triton_ready(stage: str, model_dir: Path) -> None:
    backend = normalize_backend(os.getenv("NUVION_ZSAD_BACKEND", "triton"), default="triton")
    if backend != "triton":
        return

    if not _truthy(os.getenv("NUVION_TRITON_AUTOSTART"), default=True):
        return

    triton_url = os.getenv("NUVION_TRITON_URL", "localhost:8000")
    host, port = parse_triton_host_port(triton_url)

    local_only = _truthy(os.getenv("NUVION_TRITON_AUTOSTART_ONLY_LOCAL"), default=True)
    if local_only and host not in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}:
        return

    if _health_ready(host, port, timeout_sec=3):
        _emit_progress(f"Triton 이미 준비됨: {host}:{port}")
        return

    _emit_progress("Triton이 준비되지 않아 Docker/Triton 자동 복구를 시작합니다.")
    ensure_docker_ready(triton_url)

    repository = resolve_repository_for_runtime(model_dir).resolve()
    container_name = os.getenv("NUVION_TRITON_CONTAINER_NAME", "triton-nuv").strip() or "triton-nuv"
    image = os.getenv("NUVION_TRITON_IMAGE", "nvcr.io/nvidia/tritonserver:24.10-py3").strip() or "nvcr.io/nvidia/tritonserver:24.10-py3"

    if container_exists(container_name):
        _emit_progress(f"기존 Triton 컨테이너 확인: {container_name}")
        if container_running(container_name):
            if _health_ready(host, port, timeout_sec=5):
                _emit_progress("기존 Triton 컨테이너 재사용 성공")
                return
            remove_container(container_name)
        else:
            start_container(container_name)
            if _health_ready(host, port, timeout_sec=10):
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
    _register_managed_triton_container(container_name)

    log.info("[BOOTSTRAP] Triton is ready (stage=%s, url=%s)", stage, triton_url)
    _emit_progress(f"Triton 준비 완료: {host}:{port}")
