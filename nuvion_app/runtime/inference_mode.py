from __future__ import annotations

import os

VALID_BACKENDS = {"triton", "siglip", "none"}
BACKEND_ALIASES = {"mps": "siglip"}
VALID_SIGLIP_DEVICES = {"auto", "mps", "cuda", "cpu"}
VALID_FACE_TRACKING_BACKENDS = {"auto", "triton", "opencv"}


def normalize_backend(value: str | None, default: str = "triton") -> str:
    candidate = (value or default or "triton").strip().lower()
    if not candidate:
        candidate = default
    candidate = BACKEND_ALIASES.get(candidate, candidate)
    if candidate not in VALID_BACKENDS:
        return default
    return candidate


def normalize_siglip_device(value: str | None, default: str = "auto") -> str:
    candidate = (value or default or "auto").strip().lower()
    if candidate not in VALID_SIGLIP_DEVICES:
        return default
    return candidate


def normalize_face_tracking_backend(value: str | None, default: str = "auto") -> str:
    candidate = (value or default or "auto").strip().lower()
    if candidate not in VALID_FACE_TRACKING_BACKENDS:
        return default
    return candidate


def face_tracking_uses_triton(*, enabled: bool | None = None, backend: str | None = None) -> bool:
    if enabled is None:
        enabled = str(os.getenv("NUVION_FACE_TRACKING_ENABLED", "false")).strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
    if not enabled:
        return False
    resolved_backend = normalize_face_tracking_backend(
        backend if backend is not None else os.getenv("NUVION_FACE_TRACKING_BACKEND", "auto"),
        default="auto",
    )
    return resolved_backend in {"auto", "triton"}


def apply_inference_runtime_defaults() -> None:
    raw_backend = (os.getenv("NUVION_ZSAD_BACKEND", "triton") or "triton").strip().lower()
    backend = normalize_backend(raw_backend, default="triton")
    os.environ["NUVION_ZSAD_BACKEND"] = backend
    os.environ["NUVION_FACE_TRACKING_BACKEND"] = normalize_face_tracking_backend(
        os.getenv("NUVION_FACE_TRACKING_BACKEND", "auto"),
        default="auto",
    )

    if raw_backend == "mps":
        os.environ.setdefault("NUVION_ZERO_SHOT_DEVICE", "mps")

    if backend == "siglip":
        device = normalize_siglip_device(os.getenv("NUVION_ZERO_SHOT_DEVICE", "auto"), default="auto")
        os.environ["NUVION_ZERO_SHOT_DEVICE"] = device
