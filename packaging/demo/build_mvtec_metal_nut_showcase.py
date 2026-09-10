#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import tempfile
import urllib.request
from pathlib import Path


SOURCE_REPOSITORY = "MSherbinii/mvtec-ad-metal-nut"
SOURCE_REVISION = "8db7371c35a94709cbb3e69ac69f80d2716e2f8a"
SOURCE_ROOT = (
    f"https://huggingface.co/datasets/{SOURCE_REPOSITORY}/resolve/{SOURCE_REVISION}"
)
NORMAL_SAMPLES = tuple(f"metal_nut/test/good/{index:03d}.png" for index in range(20))
DEFECT_SAMPLES = tuple(
    f"metal_nut/test/{defect}/000.png"
    for defect in ("bent", "color", "flip", "scratch")
)
LICENSE_SOURCE = "metal_nut/license.txt"
MAX_DOWNLOAD_BYTES = 4 * 1024 * 1024
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _download(source_path: str) -> bytes:
    request = urllib.request.Request(
        f"{SOURCE_ROOT}/{source_path}", headers={"User-Agent": "nuvion-demo-builder/1"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > MAX_DOWNLOAD_BYTES:
            raise ValueError(f"source file is too large: {source_path}")
        content = response.read(MAX_DOWNLOAD_BYTES + 1)
    if not content or len(content) > MAX_DOWNLOAD_BYTES:
        raise ValueError(f"source file has invalid size: {source_path}")
    return content


def _install(root: Path, relative_path: str, content: bytes) -> dict[str, object]:
    target = root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return {
        "path": relative_path,
        "sha256": hashlib.sha256(content).hexdigest(),
        "sizeBytes": len(content),
    }


def _archive(source_root: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, mode="w:xz", format=tarfile.PAX_FORMAT) as archive:
        paths = [source_root, *sorted(source_root.rglob("*"))]
        for path in paths:
            relative = path.relative_to(source_root.parent).as_posix()
            info = archive.gettarinfo(str(path), arcname=relative)
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            info.mode = 0o755 if path.is_dir() else 0o644
            if path.is_file():
                with path.open("rb") as source:
                    archive.addfile(info, source)
            else:
                archive.addfile(info)


def build(output: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="nuvion-mvtec-showcase-") as temporary:
        staging = Path(temporary)
        category_root = staging / "metal_nut"
        files: list[dict[str, object]] = []

        for source_path in NORMAL_SAMPLES:
            content = _download(source_path)
            if not content.startswith(PNG_SIGNATURE):
                raise ValueError(f"source is not a PNG: {source_path}")
            destination = "metal_nut/test/good/" + Path(source_path).name
            metadata = _install(staging, destination, content)
            metadata["source"] = source_path
            files.append(metadata)

        for index, source_path in enumerate(DEFECT_SAMPLES):
            content = _download(source_path)
            if not content.startswith(PNG_SIGNATURE):
                raise ValueError(f"source is not a PNG: {source_path}")
            defect = Path(source_path).parts[-2]
            destination = f"metal_nut/test/defect/{index:03d}-{defect}.png"
            metadata = _install(staging, destination, content)
            metadata["source"] = source_path
            files.append(metadata)

        license_metadata = _install(
            staging, "metal_nut/LICENSE.txt", _download(LICENSE_SOURCE)
        )
        license_metadata["source"] = LICENSE_SOURCE
        manifest = {
            "schemaVersion": 1,
            "profileId": "metal-nut-showcase-v1",
            "sourceRepository": SOURCE_REPOSITORY,
            "sourceRevision": SOURCE_REVISION,
            "normalImages": 20,
            "defectImages": 4,
            "files": files,
            "license": license_metadata,
        }
        _install(
            staging,
            "metal_nut/showcase-manifest.json",
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
        )
        _archive(category_root, output)

    archive_bytes = output.read_bytes()
    return {
        "path": str(output.resolve()),
        "sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "sizeBytes": len(archive_bytes),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(build(arguments.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
