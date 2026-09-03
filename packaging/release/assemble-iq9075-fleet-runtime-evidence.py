#!/usr/bin/env python3
"""Assemble signed-summary inputs for IQ9075 Fleet Runtime release evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from nuvion_app.runtime.stable_file import (
    StableFileError,
    digest_stable_regular_file,
    read_stable_regular_file,
)

SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
SHA = re.compile(r"^[0-9a-f]{40}$")
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")
MAX_JSON_BYTES = 1024 * 1024
MAX_ARTIFACT_BYTES = 4 * 1024 * 1024 * 1024


class AssemblyError(RuntimeError):
    pass


def _reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AssemblyError(f"duplicate JSON member: {key}")
        result[key] = value
    return result


def _regular_bytes(path: Path, *, maximum: int = MAX_JSON_BYTES) -> bytes:
    try:
        return read_stable_regular_file(path, maximum=maximum)
    except StableFileError as exc:
        raise AssemblyError(f"cannot read input: {path}") from exc


def _regular_digest(path: Path, *, maximum: int) -> tuple[str, int]:
    try:
        return digest_stable_regular_file(path, maximum=maximum)
    except StableFileError as exc:
        raise AssemblyError(f"cannot read input: {path}") from exc


def _object(
    path: Path, *, label: str, require_canonical: bool = False
) -> tuple[dict[str, Any], bytes]:
    raw = _regular_bytes(path)
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate,
            parse_constant=lambda item: (_ for _ in ()).throw(
                AssemblyError(f"invalid {label} constant: {item}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise AssemblyError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise AssemblyError(f"{label} root must be an object")
    if require_canonical and raw != _canonical(value):
        raise AssemblyError(
            f"{label} is not canonical sort_keys compact JSON with one newline"
        )
    return value, raw


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(payload: dict[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise AssemblyError("evidence is not canonical JSON data") from exc


def _private_output_directory(directory: Path) -> Path:
    try:
        original = directory.lstat()
        if stat.S_ISLNK(original.st_mode) or not stat.S_ISDIR(original.st_mode):
            raise AssemblyError("output directory must not be a symbolic link")
        resolved = directory.resolve(strict=True)
        resolved_metadata = resolved.stat(follow_symlinks=False)
        final = directory.lstat()
    except OSError as exc:
        raise AssemblyError("output directory is unavailable") from exc
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if (
        identity(original) != identity(resolved_metadata)
        or identity(original) != identity(final)
        or original.st_mode & 0o022
    ):
        raise AssemblyError("output directory must be a stable private directory")
    return resolved


def _safe_output(directory: Path, name: str) -> Path:
    if SAFE_NAME.fullmatch(name) is None:
        raise AssemblyError("output filename is unsafe")
    root = _private_output_directory(directory)
    path = root / name
    if path.exists() or path.is_symlink():
        raise AssemblyError(f"output already exists: {name}")
    return path


def _write_new(path: Path, raw: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
    finally:
        os.close(descriptor)


def _copy_input(source: Path, destination: Path, raw: bytes) -> None:
    if source.resolve() == destination.resolve(strict=False):
        raise AssemblyError("input and output evidence paths must be distinct")
    _write_new(destination, raw)


def _load_readiness_validator():
    path = Path(__file__).with_name("verify-release-readiness.py")
    specification = importlib.util.spec_from_file_location(
        "_nuvion_fleet_runtime_readiness_validator", path
    )
    if specification is None or specification.loader is None:
        raise AssemblyError("cannot load readiness validator")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _assemble_into(
    *,
    fleet_manifest_path: Path,
    fleet_evidence_path: Path,
    cleanup_evidence_path: Path,
    artifact_path: Path,
    bom_path: Path,
    candidate_fleet_runner: Path,
    candidate_board_tool: Path,
    security_policy_path: Path,
    output_directory: Path,
    version: str,
    component_sha: str,
) -> dict[str, str]:
    if SEMVER.fullmatch(version) is None or SHA.fullmatch(component_sha) is None:
        raise AssemblyError("release version or component SHA is invalid")
    fleet_manifest, fleet_manifest_raw = _object(
        fleet_manifest_path,
        label="Fleet Runtime manifest",
        require_canonical=True,
    )
    fleet_evidence, fleet_evidence_raw = _object(
        fleet_evidence_path,
        label="Fleet Runtime evidence",
        require_canonical=True,
    )
    cleanup_evidence, cleanup_evidence_raw = _object(
        cleanup_evidence_path,
        label="Fleet Runtime cleanup evidence",
        require_canonical=True,
    )
    _bom, bom_raw = _object(
        bom_path, label="release BOM", require_canonical=True
    )
    security, _security_raw = _object(
        security_policy_path, label="release security policy"
    )
    artifact_sha256, artifact_size = _regular_digest(
        artifact_path, maximum=MAX_ARTIFACT_BYTES
    )

    publisher_fleet_runner = (
        Path(__file__).resolve().parents[1] / "dev/run-iq9075-fleet-e2e.py"
    )
    publisher_board_tool = (
        Path(__file__).resolve().parents[1] / "dev/iq9075-board-e2e.py"
    )
    fleet_runner_sha256 = _digest(_regular_bytes(publisher_fleet_runner))
    board_tool_sha256 = _digest(_regular_bytes(publisher_board_tool))

    output_directory = _private_output_directory(output_directory)
    names = {
        "fleet_manifest": f"iq9075-v{version}-fleet-manifest.json",
        "fleet_evidence": f"iq9075-v{version}-fleet-evidence.json",
        "cleanup_evidence": f"iq9075-v{version}-cleanup-evidence.json",
        "bom": f"nuv-agent_{version}_iq9075-aarch64.release-bom.json",
        "summary": f"iq9075-v{version}-fleet-runtime-evidence.json",
    }
    paths = {key: _safe_output(output_directory, name) for key, name in names.items()}
    for source, key, raw in (
        (fleet_manifest_path, "fleet_manifest", fleet_manifest_raw),
        (fleet_evidence_path, "fleet_evidence", fleet_evidence_raw),
        (cleanup_evidence_path, "cleanup_evidence", cleanup_evidence_raw),
        (bom_path, "bom", bom_raw),
    ):
        _copy_input(source, paths[key], raw)

    readiness = _load_readiness_validator()
    try:
        runtime_gate = readiness._fleet_runtime_gate(
            fleet_evidence, cleanup_evidence, fleet_manifest
        )
    except Exception as exc:
        raise AssemblyError("Fleet Runtime result cannot be summarized") from exc
    summary = {
        "schemaVersion": 2,
        "kind": "nuvion-iq9075-fleet-runtime-release-evidence",
        "agentVersion": version,
        "componentSha": component_sha,
        "fleetRunnerSha256": fleet_runner_sha256,
        "boardToolSha256": board_tool_sha256,
        "fleetManifest": {
            "file": names["fleet_manifest"],
            "sha256": _digest(fleet_manifest_raw),
        },
        "fleetEvidence": {
            "file": names["fleet_evidence"],
            "sha256": _digest(fleet_evidence_raw),
        },
        "cleanupEvidence": {
            "file": names["cleanup_evidence"],
            "sha256": _digest(cleanup_evidence_raw),
        },
        "testedArtifact": {
            "name": artifact_path.name,
            "sha256": artifact_sha256,
            "sizeBytes": artifact_size,
        },
        "testedBom": {"file": names["bom"], "sha256": _digest(bom_raw)},
        "runtimeGate": runtime_gate,
    }
    summary_raw = _canonical(summary)
    _write_new(paths["summary"], summary_raw)
    try:
        readiness._validate_fleet_runtime_documents(
            policy_path=output_directory / "release-readiness.json",
            version=version,
            component_sha=component_sha,
            summary=summary,
            security=security,
            candidate_fleet_runner=candidate_fleet_runner,
            candidate_board_tool=candidate_board_tool,
        )
    except Exception as exc:
        raise AssemblyError(
            "assembled IQ9075 Fleet Runtime evidence failed validation"
        ) from exc
    return {
        "summary": str(paths["summary"]),
        "summarySha256": _digest(summary_raw),
        "fleetManifest": str(paths["fleet_manifest"]),
        "fleetEvidence": str(paths["fleet_evidence"]),
        "cleanupEvidence": str(paths["cleanup_evidence"]),
        "artifactSha256": artifact_sha256,
        "bomSha256": _digest(bom_raw),
    }


def assemble(
    *,
    fleet_manifest_path: Path,
    fleet_evidence_path: Path,
    cleanup_evidence_path: Path,
    artifact_path: Path,
    bom_path: Path,
    candidate_fleet_runner: Path,
    candidate_board_tool: Path,
    security_policy_path: Path,
    output_directory: Path,
    version: str,
    component_sha: str,
) -> dict[str, str]:
    final_root = _private_output_directory(output_directory)
    with tempfile.TemporaryDirectory(
        prefix=".iq9075-fleet-runtime-evidence-", dir=final_root
    ) as raw_staging:
        staging = Path(raw_staging)
        staged = _assemble_into(
            fleet_manifest_path=fleet_manifest_path,
            fleet_evidence_path=fleet_evidence_path,
            cleanup_evidence_path=cleanup_evidence_path,
            artifact_path=artifact_path,
            bom_path=bom_path,
            candidate_fleet_runner=candidate_fleet_runner,
            candidate_board_tool=candidate_board_tool,
            security_policy_path=security_policy_path,
            output_directory=staging,
            version=version,
            component_sha=component_sha,
        )
        staged_files = sorted(path for path in staging.iterdir() if path.is_file())
        if len(staged_files) != 5:
            raise AssemblyError("staged Fleet Runtime evidence file set is incomplete")
        final_paths = [_safe_output(final_root, path.name) for path in staged_files]
        published: list[Path] = []
        try:
            for source, destination in zip(staged_files, final_paths, strict=True):
                os.link(source, destination, follow_symlinks=False)
                published.append(destination)
            directory_fd = os.open(final_root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except Exception:
            for path in reversed(published):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            raise
        result = dict(staged)
        for key in (
            "summary",
            "fleetManifest",
            "fleetEvidence",
            "cleanupEvidence",
        ):
            result[key] = str(final_root / Path(result[key]).name)
        return result


def main() -> int:
    if not sys.flags.isolated:
        print(
            "assemble-iq9075-fleet-runtime-evidence.py requires Python isolated "
            "mode (-I)",
            file=sys.stderr,
        )
        return 2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fleet-manifest", required=True, type=Path)
    parser.add_argument("--fleet-evidence", required=True, type=Path)
    parser.add_argument("--cleanup-evidence", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--bom", required=True, type=Path)
    parser.add_argument("--candidate-fleet-runner", required=True, type=Path)
    parser.add_argument("--candidate-board-tool", required=True, type=Path)
    parser.add_argument("--security-policy", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--component-sha", required=True)
    arguments = parser.parse_args()
    try:
        result = assemble(
            fleet_manifest_path=arguments.fleet_manifest,
            fleet_evidence_path=arguments.fleet_evidence,
            cleanup_evidence_path=arguments.cleanup_evidence,
            artifact_path=arguments.artifact,
            bom_path=arguments.bom,
            candidate_fleet_runner=arguments.candidate_fleet_runner,
            candidate_board_tool=arguments.candidate_board_tool,
            security_policy_path=arguments.security_policy,
            output_directory=arguments.output_directory,
            version=arguments.version,
            component_sha=arguments.component_sha,
        )
    except AssemblyError as exc:
        print(
            f"IQ9075 Fleet Runtime evidence assembly failed: {exc}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
