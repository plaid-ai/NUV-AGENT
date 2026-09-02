"""Isolated SigLIP text-feature worker used by the low-memory MPS runtime."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import stat
import sys
from pathlib import Path

try:
    from nuvion_app.inference._safetensors_io import (
        open_safetensors_for_sequential_load,
    )
except ModuleNotFoundError:
    # `python -I /path/to/worker.py` intentionally excludes the source root.
    # Load only this validated sibling for source-tree diagnostics; installed
    # releases resolve the normal package import above.
    helper_path = Path(__file__).resolve().with_name("_safetensors_io.py")
    helper_metadata = helper_path.lstat()
    if not stat.S_ISREG(helper_metadata.st_mode) or stat.S_ISLNK(
        helper_metadata.st_mode
    ):
        raise RuntimeError("safetensors helper must be a regular package file")
    helper_spec = importlib.util.spec_from_file_location(
        "_nuvion_safetensors_io", helper_path
    )
    if helper_spec is None or helper_spec.loader is None:
        raise RuntimeError("unable to load the safetensors helper")
    helper_module = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper_module)
    open_safetensors_for_sequential_load = (
        helper_module.open_safetensors_for_sequential_load
    )

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

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    texts = [f"This is a photo of {label}." for label in labels]
    inputs = tokenizer(
        texts,
        padding="max_length",
        max_length=64,
        return_tensors="pt",
    )
    input_ids = inputs.get("input_ids")
    if (
        not isinstance(input_ids, torch.Tensor)
        or input_ids.ndim != 2
        or input_ids.numel() == 0
        or input_ids.is_floating_point()
    ):
        raise ValueError("tokenizer input IDs are invalid")
    token_ids = sorted({int(value) for value in input_ids.reshape(-1).tolist()})
    original_vocab_size = getattr(config.text_config, "vocab_size", None)
    if (
        type(original_vocab_size) is not int
        or original_vocab_size <= 0
        or not token_ids
        or token_ids[0] < 0
        or token_ids[-1] >= original_vocab_size
    ):
        raise ValueError("tokenizer input IDs exceed the text checkpoint vocabulary")
    token_index = {token_id: index for index, token_id in enumerate(token_ids)}
    compact_input_ids = input_ids.clone()
    for token_id, index in token_index.items():
        compact_input_ids[input_ids == token_id] = index
    inputs["input_ids"] = compact_input_ids

    text_config = copy.deepcopy(config.text_config)
    text_config.vocab_size = len(token_ids)
    for attribute in ("bos_token_id", "eos_token_id", "pad_token_id"):
        original_id = getattr(text_config, attribute, None)
        setattr(text_config, attribute, token_index.get(original_id))

    with torch.device("meta"):
        model = text_cls(text_config)
    model = model.to(dtype=torch.float16)
    expected = model.state_dict(keep_vars=True)
    embedding_keys = [
        key
        for key in expected
        if key.endswith("embeddings.token_embedding.weight")
    ]
    if len(embedding_keys) != 1:
        raise ValueError("SigLIP text embedding layout is unsupported")
    embedding_key = embedding_keys[0]
    with open_safetensors_for_sequential_load(checkpoint) as weights:
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
            checkpoint_shape = tuple(weights.get_slice(checkpoint_key).get_shape())
            if model_key == embedding_key:
                expected_shape = (original_vocab_size, target.shape[1])
            else:
                expected_shape = tuple(target.shape)
            if checkpoint_shape != expected_shape:
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
                if model_key == embedding_key:
                    source = torch.cat(
                        [
                            weights.get_slice(checkpoint_key)[
                                token_id : token_id + 1
                            ]
                            for token_id in token_ids
                        ],
                        dim=0,
                    )
                else:
                    source = weights.get_tensor(checkpoint_key)
                expected[model_key].copy_(source)
                del source
    model.eval()

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
