from __future__ import annotations

import logging
import os
import shutil
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
)
from nuvion_app.runtime.errors import BootstrapError

log = logging.getLogger(__name__)


_FALLBACK_CONFIG = """name: \"image_encoder\"
platform: \"onnxruntime_onnx\"
max_batch_size: 4
input [
  {
    name: \"images\"
    data_type: TYPE_FP32
    dims: [3, 336, 336]
  }
]
output [
  {
    name: \"patch_features\"
    data_type: TYPE_FP32
    dims: [4, 577, 768]
  },
  {
    name: \"image_features\"
    data_type: TYPE_FP32
    dims: [768]
  }
]
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


def _ensure_macos_onnx_repository(model_dir: Path, repository_root: Path) -> Path:
    model_repo = repository_root / "image_encoder" / "1"
    model_repo.mkdir(parents=True, exist_ok=True)

    onnx_src = model_dir / "onnx" / "image_encoder_simplified.onnx"
    if not onnx_src.exists():
        raise BootstrapError(
            "triton_health_failed",
            f"Missing ONNX model for macOS fallback: {onnx_src}",
        )

    target_onnx = model_repo / "model.onnx"
    if not target_onnx.exists() or target_onnx.stat().st_size != onnx_src.stat().st_size:
        shutil.copy2(onnx_src, target_onnx)

    source_config = model_dir / "triton" / "model_repository" / "image_encoder" / "config.pbtxt"
    target_config_dir = repository_root / "image_encoder"
    target_config_dir.mkdir(parents=True, exist_ok=True)
    target_config = target_config_dir / "config.pbtxt"

    if source_config.exists():
        shutil.copy2(source_config, target_config)
    else:
        target_config.write_text(_FALLBACK_CONFIG)

    return repository_root


def resolve_repository_for_runtime(model_dir: Path) -> Path:
    default_repo = model_dir / "triton" / "model_repository"
    if os.uname().sysname.lower() != "darwin":
        if not default_repo.exists():
            raise BootstrapError("triton_health_failed", f"Triton model repository is missing: {default_repo}")
        return default_repo

    model_onnx = default_repo / "image_encoder" / "1" / "model.onnx"
    if model_onnx.exists():
        return default_repo

    fallback = model_dir / "triton" / "model_repository_onnx"
    return _ensure_macos_onnx_repository(model_dir=model_dir, repository_root=fallback)


def ensure_triton_ready(stage: str, model_dir: Path) -> None:
    backend = os.getenv("NUVION_ZSAD_BACKEND", "triton").strip().lower()
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
        return

    ensure_docker_ready(triton_url)

    repository = resolve_repository_for_runtime(model_dir).resolve()
    container_name = os.getenv("NUVION_TRITON_CONTAINER_NAME", "triton-nuv").strip() or "triton-nuv"
    image = os.getenv("NUVION_TRITON_IMAGE", "nvcr.io/nvidia/tritonserver:24.10-py3").strip() or "nvcr.io/nvidia/tritonserver:24.10-py3"

    if container_exists(container_name):
        if container_running(container_name):
            if _health_ready(host, port, timeout_sec=5):
                return
            remove_container(container_name)
        else:
            start_container(container_name)
            if _health_ready(host, port, timeout_sec=10):
                return
            remove_container(container_name)

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

    log.info("[BOOTSTRAP] Triton is ready (stage=%s, url=%s)", stage, triton_url)
