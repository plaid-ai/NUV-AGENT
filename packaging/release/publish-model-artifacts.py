#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def _run(cmd: list[str], *, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, text=True, capture_output=capture_output)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _gs_to_object_name(uri: str, bucket: str) -> str:
    prefix = f"gs://{bucket}/"
    if not uri.startswith(prefix):
        raise ValueError(f"{uri} is not in bucket {bucket}")
    return uri[len(prefix) :]


def _gcs_cat_json(uri: str) -> dict[str, Any]:
    result = _run(["gcloud", "storage", "cat", uri], capture_output=True)
    return json.loads(result.stdout)


def _gcs_cp(src: str, dst: str) -> None:
    _run(["gcloud", "storage", "cp", src, dst])


def _artifact_entry(local_path: Path, gcs_uri: str, bucket: str) -> dict[str, Any]:
    return {
        "path": _gs_to_object_name(gcs_uri, bucket),
        "sha256": _sha256_file(local_path),
        "sizeBytes": local_path.stat().st_size,
    }


def _ordered_unique(items: list[str], extras: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for item in items + extras:
        normalized = str(item).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        merged.append(normalized)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload model artifacts to GCS and patch pointer manifests")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--version", required=True, help="Resolved model version, e.g. v0001")
    parser.add_argument("--version-pointer", required=True, help="Version pointer gs:// URI")
    parser.add_argument(
        "--channel-pointer",
        action="append",
        default=[],
        help="Optional channel pointer gs:// URI to overwrite from the updated version pointer",
    )
    parser.add_argument("--face-onnx", required=True)
    parser.add_argument("--face-plan", required=True)
    parser.add_argument("--face-triton-config", required=True)
    args = parser.parse_args()

    bucket = args.bucket.strip()
    model_name = args.model_name.strip()
    version = args.version.strip()
    version_root = f"gs://{bucket}/nuvion/{model_name}/{version}"

    face_onnx_path = Path(args.face_onnx).expanduser().resolve()
    face_plan_path = Path(args.face_plan).expanduser().resolve()
    face_config_path = Path(args.face_triton_config).expanduser().resolve()

    artifact_targets = {
        "face_onnx": (face_onnx_path, f"{version_root}/source/face_detector.onnx"),
        "face_plan": (face_plan_path, f"{version_root}/triton/model_repository/face_detector/1/model.plan"),
        "face_triton_config": (face_config_path, f"{version_root}/triton/model_repository/face_detector/config.pbtxt"),
    }

    for key, (local_path, gcs_uri) in artifact_targets.items():
        print(f"Uploading {key}: {local_path} -> {gcs_uri}")
        _gcs_cp(str(local_path), gcs_uri)

    pointer = _gcs_cat_json(args.version_pointer)
    artifacts = pointer.setdefault("artifacts", {})
    profiles = pointer.setdefault("profiles", {})

    for key, (local_path, gcs_uri) in artifact_targets.items():
        artifacts[key] = _artifact_entry(local_path, gcs_uri, bucket)

    profiles["runtime"] = _ordered_unique(
        list(profiles.get("runtime") or []),
        ["face_onnx", "face_plan", "face_triton_config"],
    )
    profiles["full"] = _ordered_unique(
        list(profiles.get("full") or []),
        ["face_onnx", "face_plan", "face_triton_config"],
    )
    profiles["light"] = _ordered_unique(
        list(profiles.get("light") or []),
        ["face_onnx"],
    )

    with tempfile.TemporaryDirectory() as tmp:
        pointer_path = Path(tmp) / "pointer.json"
        pointer_path.write_text(json.dumps(pointer, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        print(f"Updating version pointer: {args.version_pointer}")
        _gcs_cp(str(pointer_path), args.version_pointer)

        for channel_pointer in args.channel_pointer:
            channel_pointer = channel_pointer.strip()
            if not channel_pointer:
                continue
            print(f"Promoting updated pointer to channel: {channel_pointer}")
            _gcs_cp(str(pointer_path), channel_pointer)

    print("Updated artifacts:")
    for key, artifact in artifacts.items():
        if key.startswith("face_"):
            print(f"  {key}: {artifact['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
