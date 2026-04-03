#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from nuvion_app.runtime.triton_manager import (
    _build_face_detector_trt_plan,
    _default_face_tracking_config,
    _prepare_face_detector_onnx_for_runtime,
)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _artifact_metadata(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sizeBytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _set_env(key: str, value: str) -> None:
    os.environ[key] = value


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Jetson face tracking TensorRT artifacts from face_detector.onnx")
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Model bundle root containing onnx/face_detector.onnx",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write face_detector.onnx, face_detector.plan, face_detector.config.pbtxt",
    )
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument("--input-height", type=int, default=480)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--opt-batch-size", type=int, default=2)
    parser.add_argument("--workspace-gib", type=float, default=1.0)
    parser.add_argument(
        "--fp16",
        choices=("true", "false"),
        default="true",
        help="Enable FP16 during TensorRT plan build when supported",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _set_env("NUVION_FACE_TRACKING_INPUT_WIDTH", str(max(args.input_width, 1)))
    _set_env("NUVION_FACE_TRACKING_INPUT_HEIGHT", str(max(args.input_height, 1)))
    _set_env("NUVION_FACE_TRACKING_BATCH_SIZE", str(max(args.batch_size, 1)))
    _set_env("NUVION_FACE_TRACKING_OPT_BATCH_SIZE", str(max(min(args.opt_batch_size, args.batch_size), 1)))
    _set_env("NUVION_FACE_TRACKING_TRT_WORKSPACE_GIB", str(max(args.workspace_gib, 0.25)))
    _set_env("NUVION_FACE_TRACKING_TRT_FP16", args.fp16)

    onnx_src = model_dir / "onnx" / "face_detector.onnx"
    prepared_onnx = output_dir / "face_detector.onnx"
    plan_path = output_dir / "face_detector.plan"
    config_path = output_dir / "face_detector.config.pbtxt"
    metadata_path = output_dir / "face_detector.metadata.json"

    prepared_onnx = _prepare_face_detector_onnx_for_runtime(onnx_src, prepared_onnx)
    if not _build_face_detector_trt_plan(prepared_onnx, plan_path):
        raise SystemExit("failed to build TensorRT face_detector plan")

    config_path.write_text(_default_face_tracking_config("tensorrt_plan"), encoding="utf-8")
    metadata = {
        "inputWidth": int(os.environ["NUVION_FACE_TRACKING_INPUT_WIDTH"]),
        "inputHeight": int(os.environ["NUVION_FACE_TRACKING_INPUT_HEIGHT"]),
        "batchSize": int(os.environ["NUVION_FACE_TRACKING_BATCH_SIZE"]),
        "optBatchSize": int(os.environ["NUVION_FACE_TRACKING_OPT_BATCH_SIZE"]),
        "workspaceGiB": float(os.environ["NUVION_FACE_TRACKING_TRT_WORKSPACE_GIB"]),
        "fp16": os.environ["NUVION_FACE_TRACKING_TRT_FP16"].strip().lower() in {"1", "true", "yes", "on"},
        "artifacts": {
            "face_onnx": _artifact_metadata(prepared_onnx),
            "face_plan": _artifact_metadata(plan_path),
            "face_triton_config": _artifact_metadata(config_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Prepared ONNX: {prepared_onnx}")
    print(f"TensorRT plan: {plan_path}")
    print(f"Triton config: {config_path}")
    print(f"Metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
