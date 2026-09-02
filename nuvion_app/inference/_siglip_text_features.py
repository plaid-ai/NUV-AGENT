"""Isolated SigLIP text-feature worker used by the low-memory MPS runtime."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_MAX_REQUEST_BYTES = 1024 * 1024
_MAX_LABELS = 256
_MAX_LABEL_LENGTH = 512
_MAX_FEATURE_WIDTH = 8192


def _pooled_feature_tensor(output):
    pooled = getattr(output, "pooler_output", None)
    if pooled is not None:
        output = pooled
    if not callable(getattr(output, "norm", None)):
        raise TypeError("model feature output has no tensor pooler output")
    return output


def _read_request() -> tuple[str, list[str]]:
    raw = sys.stdin.buffer.read(_MAX_REQUEST_BYTES + 1)
    if not raw or len(raw) > _MAX_REQUEST_BYTES:
        raise ValueError("text-feature request size is invalid")
    payload = json.loads(raw)
    if not isinstance(payload, dict) or set(payload) != {
        "schemaVersion",
        "modelName",
        "labels",
    }:
        raise ValueError("text-feature request schema is invalid")
    if type(payload["schemaVersion"]) is not int or payload["schemaVersion"] != 1:
        raise ValueError("text-feature request version is invalid")
    model_name = payload["modelName"]
    labels = payload["labels"]
    if not isinstance(model_name, str) or not model_name or len(model_name) > 4096:
        raise ValueError("text-feature model name is invalid")
    if not isinstance(labels, list) or not 1 <= len(labels) <= _MAX_LABELS:
        raise ValueError("text-feature labels are invalid")
    if any(
        not isinstance(label, str)
        or not label
        or len(label) > _MAX_LABEL_LENGTH
        for label in labels
    ):
        raise ValueError("text-feature label is invalid")
    return model_name, labels


def main() -> int:
    model_name, labels = _read_request()

    import torch
    import transformers
    from safetensors import safe_open

    model_source = Path(model_name)
    checkpoint = model_source / "model.safetensors"
    if not checkpoint.is_file() or (
        model_source / "model.safetensors.index.json"
    ).exists():
        raise ValueError("text-feature worker requires unsharded model.safetensors")

    config = transformers.AutoConfig.from_pretrained(model_name)
    model_type = getattr(config, "model_type", None)
    if model_type == "siglip":
        text_cls = getattr(transformers, "SiglipTextModel", None)
    elif model_type == "siglip2":
        text_cls = getattr(transformers, "Siglip2TextModel", None)
    else:
        text_cls = None
    if text_cls is None or not hasattr(config, "text_config"):
        raise ValueError(f"unsupported text-feature model type: {model_type!r}")

    with torch.device("meta"):
        model = text_cls(config.text_config)
    model = model.to(dtype=torch.float16)
    expected = model.state_dict(keep_vars=True)
    with safe_open(checkpoint, framework="pt", device="cpu") as weights:
        checkpoint_keys = set(weights.keys())
        text_keys = {
            key for key in checkpoint_keys if key.startswith("text_model.")
        }
        if set(expected) == text_keys:
            checkpoint_key_by_model_key = {key: key for key in expected}
        elif {f"text_model.{key}" for key in expected} == text_keys:
            checkpoint_key_by_model_key = {
                key: f"text_model.{key}" for key in expected
            }
        else:
            raise ValueError("checkpoint text tensor set does not match model config")
        for model_key, target in expected.items():
            checkpoint_key = checkpoint_key_by_model_key[model_key]
            if tuple(weights.get_slice(checkpoint_key).get_shape()) != tuple(
                target.shape
            ):
                raise ValueError(f"checkpoint tensor shape mismatch: {checkpoint_key}")

        model.to_empty(device="cpu")
        buffers = dict(model.named_buffers())
        position_keys = [
            key for key in buffers if key.endswith("embeddings.position_ids")
        ]
        if len(position_keys) != 1 or len(buffers) != 1:
            raise ValueError("SigLIP text buffer set is unsupported")
        position_ids = buffers[position_keys[0]]
        if position_ids.ndim != 2 or position_ids.shape[0] != 1:
            raise ValueError("SigLIP text position-id buffer shape is invalid")
        position_ids.copy_(
            torch.arange(position_ids.shape[1]).reshape(position_ids.shape)
        )
        expected = model.state_dict(keep_vars=True)
        with torch.no_grad():
            for model_key in sorted(expected):
                checkpoint_key = checkpoint_key_by_model_key[model_key]
                expected[model_key].copy_(weights.get_tensor(checkpoint_key))
    model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    texts = [f"This is a photo of {label}." for label in labels]
    inputs = tokenizer(
        texts,
        padding="max_length",
        max_length=64,
        return_tensors="pt",
    )
    inputs = {
        key: value.to(dtype=torch.float16) if value.is_floating_point() else value
        for key, value in inputs.items()
    }
    with torch.inference_mode():
        features = _pooled_feature_tensor(model(**inputs))
    rows = features.detach().float().cpu().tolist()
    if (
        len(rows) != len(labels)
        or not rows
        or not isinstance(rows[0], list)
        or not 1 <= len(rows[0]) <= _MAX_FEATURE_WIDTH
    ):
        raise ValueError("text-feature output shape is invalid")
    width = len(rows[0])
    if any(
        not isinstance(row, list)
        or len(row) != width
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in row
        )
        for row in rows
    ):
        raise ValueError("text-feature output value is invalid")

    json.dump(
        {
            "schemaVersion": 1,
            "labels": labels,
            "features": rows,
        },
        sys.stdout,
        sort_keys=True,
        separators=(",", ":"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
