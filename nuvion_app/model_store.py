from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

DEFAULT_MODEL_REPO_ID = "plaidlabs/nuvion-v1"
DEFAULT_MODEL_PROFILE = "runtime"

_PROFILE_ALLOW_PATTERNS: dict[str, list[str] | None] = {
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


def resolve_default_model_dir(repo_id: str) -> Path:
    root = Path(os.getenv("NUVION_MODEL_DIR", "~/.cache/nuvion/models")).expanduser()
    safe_repo = repo_id.replace("/", "__")
    return (root / safe_repo).resolve()


def _resolve_local_dir(repo_id: str, local_dir: Optional[str]) -> Path:
    if local_dir:
        return Path(local_dir).expanduser().resolve()
    return resolve_default_model_dir(repo_id)


def pull_model_snapshot(
    repo_id: str,
    revision: Optional[str] = None,
    local_dir: Optional[str] = None,
    token: Optional[str] = None,
    profile: str = DEFAULT_MODEL_PROFILE,
) -> Path:
    if profile not in _PROFILE_ALLOW_PATTERNS:
        valid = ", ".join(sorted(_PROFILE_ALLOW_PATTERNS.keys()))
        raise ValueError(f"Unsupported profile '{profile}'. Expected one of: {valid}")

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required for 'pull-model'. Install with: pip install huggingface_hub"
        ) from exc

    target_dir = _resolve_local_dir(repo_id=repo_id, local_dir=local_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    kwargs: dict[str, object] = {
        "repo_id": repo_id,
        "repo_type": "model",
        "local_dir": str(target_dir),
        "allow_patterns": _PROFILE_ALLOW_PATTERNS[profile],
    }
    if revision:
        kwargs["revision"] = revision
    if token:
        kwargs["token"] = token

    snapshot_path = snapshot_download(**kwargs)
    return Path(snapshot_path).resolve()


def anomalyclip_text_features_path(model_dir: Path) -> Path:
    return model_dir / "onnx" / "text_features.npy"


def anomalyclip_triton_repository_path(model_dir: Path) -> Path:
    return model_dir / "triton" / "model_repository"
