from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Optional

DEFAULT_MODEL_SOURCE = "hf"
DEFAULT_MODEL_REPO_ID = "plaidlabs/nuvion-v1"
DEFAULT_MODEL_PROFILE = "runtime"
DEFAULT_MODEL_GCS_POINTER_URI = "gs://nuv-model/pointers/anomalyclip/prod.json"

_HF_PROFILE_ALLOW_PATTERNS: dict[str, list[str] | None] = {
    "full": None,
    "runtime": [
        "README.md",
        "metadata/**",
        "onnx/text_features.npy",
        "triton/model_repository/**",
    ],
    "light": [
        "README.md",
        "metadata/**",
        "onnx/text_features.npy",
    ],
}


_GCS_PROFILE_KEYS: dict[str, list[str]] = {
    "runtime": ["text_features", "plan", "triton_config", "manifest"],
    "light": ["text_features", "manifest"],
    "full": [
        "onnx",
        "text_features",
        "pytorch",
        "plan",
        "triton_config",
        "onnx_eval",
        "triton_eval",
        "manifest",
    ],
}

_DEFAULT_LOCAL_PATHS = {
    "text_features": "onnx/text_features.npy",
    "onnx": "onnx/image_encoder_simplified.onnx",
    "pytorch": "pytorch/epoch_15.pth",
    "plan": "triton/model_repository/image_encoder/1/model.plan",
    "triton_config": "triton/model_repository/image_encoder/config.pbtxt",
    "onnx_eval": "metadata/onnx_eval_results.txt",
    "triton_eval": "metadata/triton_eval_results.txt",
    "manifest": "metadata/gcs_manifest.json",
}


def resolve_default_model_dir(identifier: str) -> Path:
    root = Path(os.getenv("NUVION_MODEL_DIR", "~/.cache/nuvion/models")).expanduser()
    safe_id = identifier.replace("/", "__").replace(":", "_")
    return (root / safe_id).resolve()


def _resolve_local_dir(identifier: str, local_dir: Optional[str]) -> Path:
    if local_dir:
        return Path(local_dir).expanduser().resolve()
    return resolve_default_model_dir(identifier)


def _ensure_profile(profile: str) -> None:
    valid = set(_HF_PROFILE_ALLOW_PATTERNS.keys())
    if profile not in valid:
        allowed = ", ".join(sorted(valid))
        raise ValueError(f"Unsupported profile '{profile}'. Expected one of: {allowed}")


def pull_model_snapshot(
    repo_id: str,
    revision: Optional[str] = None,
    local_dir: Optional[str] = None,
    token: Optional[str] = None,
    profile: str = DEFAULT_MODEL_PROFILE,
) -> Path:
    _ensure_profile(profile)

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required for 'pull-model'. Install with: pip install huggingface_hub"
        ) from exc

    target_dir = _resolve_local_dir(identifier=repo_id, local_dir=local_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    kwargs: dict[str, object] = {
        "repo_id": repo_id,
        "repo_type": "model",
        "local_dir": str(target_dir),
        "allow_patterns": _HF_PROFILE_ALLOW_PATTERNS[profile],
    }
    if revision:
        kwargs["revision"] = revision
    if token:
        kwargs["token"] = token

    snapshot_path = snapshot_download(**kwargs)
    return Path(snapshot_path).resolve()


def _run_command(cmd: list[str], capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=capture_output,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Command not found: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() if exc.stderr else str(exc)
        raise RuntimeError(detail) from exc


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    trimmed = uri[5:]
    bucket, _, object_path = trimmed.partition("/")
    if not bucket:
        raise ValueError(f"Invalid gs:// URI (missing bucket): {uri}")
    return bucket, object_path


def _gcs_uri(bucket: str, object_path: str) -> str:
    object_path = object_path.lstrip("/")
    return f"gs://{bucket}/{object_path}"


def _gcs_cat_json(uri: str) -> dict:
    result = _run_command(["gcloud", "storage", "cat", uri], capture_output=True)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in pointer: {uri}") from exc


def _copy_gcs_object(src_uri: str, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    _run_command(["gcloud", "storage", "cp", src_uri, str(dst_path)])


def pull_model_from_gcs(
    pointer_uri: str,
    local_dir: Optional[str] = None,
    profile: str = DEFAULT_MODEL_PROFILE,
) -> tuple[Path, dict]:
    _ensure_profile(profile)

    bucket, _ = _parse_gs_uri(pointer_uri)
    pointer = _gcs_cat_json(pointer_uri)

    artifacts = pointer.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("Pointer JSON must include 'artifacts' object")

    runtime_layout = pointer.get("runtime_layout")
    local_paths = {}
    if isinstance(runtime_layout, dict):
        lp = runtime_layout.get("local_paths")
        if isinstance(lp, dict):
            local_paths = {str(k): str(v) for k, v in lp.items()}

    target_dir = _resolve_local_dir(identifier=pointer_uri, local_dir=local_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    required_keys = _GCS_PROFILE_KEYS[profile]
    missing = [key for key in required_keys if key not in artifacts]
    if missing:
        raise RuntimeError(f"Pointer is missing required artifacts for profile '{profile}': {', '.join(missing)}")

    downloaded: list[dict[str, str]] = []
    for key in required_keys:
        src_obj = str(artifacts[key])
        src_uri = _gcs_uri(bucket, src_obj)
        local_rel = local_paths.get(key, _DEFAULT_LOCAL_PATHS.get(key, f"extras/{key}"))
        dst = (target_dir / local_rel).resolve()
        _copy_gcs_object(src_uri=src_uri, dst_path=dst)
        downloaded.append({"key": key, "src": src_uri, "dst": str(dst)})

    metadata_dir = (target_dir / "metadata").resolve()
    metadata_dir.mkdir(parents=True, exist_ok=True)
    pointer_out = metadata_dir / "gcs_pointer.json"
    pointer_out.write_text(json.dumps(pointer, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    downloaded_index = metadata_dir / "downloaded_from_gcs.json"
    downloaded_index.write_text(json.dumps(downloaded, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return target_dir, pointer


def anomalyclip_text_features_path(model_dir: Path) -> Path:
    return model_dir / "onnx" / "text_features.npy"


def anomalyclip_triton_repository_path(model_dir: Path) -> Path:
    return model_dir / "triton" / "model_repository"
