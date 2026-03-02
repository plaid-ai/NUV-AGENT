from __future__ import annotations

import os

VALID_BACKENDS = {"triton", "siglip", "none"}
BACKEND_ALIASES = {"mps": "siglip"}
VALID_SIGLIP_DEVICES = {"auto", "mps", "cuda", "cpu"}


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


def apply_inference_runtime_defaults() -> None:
    raw_backend = (os.getenv("NUVION_ZSAD_BACKEND", "triton") or "triton").strip().lower()
    backend = normalize_backend(raw_backend, default="triton")
    os.environ["NUVION_ZSAD_BACKEND"] = backend

    if raw_backend == "mps":
        os.environ.setdefault("NUVION_ZERO_SHOT_DEVICE", "mps")

    if backend == "siglip":
        device = normalize_siglip_device(os.getenv("NUVION_ZERO_SHOT_DEVICE", "auto"), default="auto")
        os.environ["NUVION_ZERO_SHOT_DEVICE"] = device
