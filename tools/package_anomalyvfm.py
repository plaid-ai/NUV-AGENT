#!/usr/bin/env python3
"""Create a deterministic Fleet package and GCS pointer; never upload or activate it."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nuvion_app.runtime.anomalyvfm import BACKEND, FILES, INPUT, OUTPUTS, POINTER


def package(directory: Path):
    expected = {
        "context": "6859496719e1d5a2864dd9619a1e859a52f2d0c43920798da861b9b9f9746630",
        "context_binary": "7b2cbe2882742360eb68490f83d1719679ae5f43c4b779bbb7987b1d2755f5c2",
    }
    artifacts = {}
    for key, filename in FILES.items():
        path = directory / filename
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != expected[key]:
            raise ValueError(f"Artifact differs from evaluated v16 package: {key}")
        artifacts[key] = {"path": filename, "sha256": digest, "sizeBytes": path.stat().st_size}
    manifest = {"schemaVersion": 1, "backend": BACKEND, "pointer": POINTER,
        "platformProfile": "ventuno_q", "ortVersion": "1.30.0", "qnnVersion": "2.6.0",
        "input": INPUT, "outputs": OUTPUTS, "postprocess": "sigmoid-zero-pad-avg5-v1",
        "thresholdStatus": "UNCALIBRATED", "artifacts": artifacts}
    raw = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    digest = hashlib.sha256(raw).hexdigest()
    (directory / "manifest.json").write_bytes(raw)
    prefix = f"nuvion/anomalyvfm/ventuno-q/v16/{digest}"
    pointer = {"schemaVersion": "2", "modelName": "AnomalyVFM RADIO MinMax W8A16 + V-range",
        "resolvedVersion": "v16-20261001", "artifacts": {
            **{k: {**v, "path": f"{prefix}/{v['path']}"} for k, v in artifacts.items()},
            "manifest": {"path": f"{prefix}/manifest.json", "sha256": digest, "sizeBytes": len(raw)}},
        "profiles": {"qnn-context": ["manifest", "context", "context_binary"]}}
    (directory / "pointer.json").write_text(json.dumps(pointer, indent=2) + "\n")
    print(json.dumps({"pointer": POINTER, "digest": "sha256:" + digest, "prefix": prefix}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    package(parser.parse_args().directory.resolve())
