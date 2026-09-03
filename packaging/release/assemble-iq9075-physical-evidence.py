#!/usr/bin/env python3
"""Assemble machine-verifiable IQ9075 release evidence from immutable raw runs."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import re
import stat
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from nuvion_app.runtime.release_bom import (
    ReleaseBomValidationError,
    verify_release_bom,
)
from nuvion_app.runtime.stable_file import (
    StableFileError,
    digest_stable_regular_file,
    read_stable_regular_file,
)

SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
SHA = re.compile(r"^[0-9a-f]{40}$")
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")
MAX_JSON_BYTES = 1024 * 1024
CANDIDATE_SOAK_REQUIRED_VERSIONS = frozenset({"0.1.121"})


class AssemblyError(RuntimeError):
    pass


def _reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AssemblyError(f"duplicate JSON member: {key}")
        result[key] = value
    return result


def _timestamp(value: object, *, label: str) -> dt.datetime:
    if not isinstance(value, str) or re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
        r"(?:\.[0-9]{3})?Z",
        value,
    ) is None:
        raise AssemblyError(f"{label} is not canonical UTC")
    try:
        parsed = dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise AssemblyError(f"{label} is not canonical UTC") from exc
    if parsed.tzinfo != dt.timezone.utc:
        raise AssemblyError(f"{label} is not canonical UTC")
    return parsed


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


def _object(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _regular_bytes(path)
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate,
            parse_constant=lambda item: (_ for _ in ()).throw(
                AssemblyError(f"invalid {label} constant: {item}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AssemblyError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise AssemblyError(f"{label} root must be an object")
    return value, raw


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


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


def _copy_input(source: Path, destination: Path, raw: bytes) -> None:
    if source.resolve() == destination.resolve(strict=False):
        raise AssemblyError("input and output evidence paths must be distinct")
    _write_new(destination, raw)


def _load_readiness_validator():
    path = Path(__file__).with_name("verify-release-readiness.py")
    specification = importlib.util.spec_from_file_location(
        "_nuvion_physical_readiness_validator", path
    )
    if specification is None or specification.loader is None:
        raise AssemblyError("cannot load readiness validator")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _module_from_verified_source(
    *, module_name: str, source: bytes, display_path: Path
) -> types.ModuleType:
    """Execute the already-verified source bytes without reopening their path."""

    module = types.ModuleType(module_name)
    module.__file__ = str(display_path)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        code = compile(source, str(display_path), "exec", dont_inherit=True)
        exec(code, module.__dict__)  # noqa: S102 - bytes are trusted and exact-compared.
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _assemble_into(
    *,
    soak_result_path: Path,
    fleet_manifest_path: Path,
    fleet_evidence_path: Path,
    artifact_path: Path,
    bom_path: Path,
    candidate_harness: Path,
    candidate_fleet_runner: Path,
    candidate_board_tool: Path,
    security_policy_path: Path,
    output_directory: Path,
    version: str,
    component_sha: str,
    candidate_soak_evidence_path: Path | None = None,
) -> dict[str, str]:
    if SEMVER.fullmatch(version) is None or SHA.fullmatch(component_sha) is None:
        raise AssemblyError("release version or component SHA is invalid")
    soak, soak_raw = _object(soak_result_path, label="OAK soak result")
    fleet_manifest, fleet_manifest_raw = _object(
        fleet_manifest_path, label="Fleet E2E manifest"
    )
    fleet_evidence, fleet_evidence_raw = _object(
        fleet_evidence_path, label="Fleet E2E evidence"
    )
    bom, bom_raw = _object(bom_path, label="release BOM")
    security, _security_raw = _object(
        security_policy_path, label="release security policy"
    )
    candidate_soak: dict[str, Any] | None = None
    candidate_soak_raw: bytes | None = None
    cleanup_evidence_raw: bytes | None = None
    if candidate_soak_evidence_path is not None:
        candidate_soak, candidate_soak_raw = _object(
            candidate_soak_evidence_path, label="candidate soak evidence"
        )
    if version in CANDIDATE_SOAK_REQUIRED_VERSIONS and candidate_soak is None:
        raise AssemblyError(
            f"v{version} requires schema-v3 candidate soak evidence"
        )
    try:
        verified_bom = verify_release_bom(bom)
    except ReleaseBomValidationError as exc:
        raise AssemblyError("release BOM is invalid") from exc

    artifact_sha, artifact_size = _regular_digest(
        artifact_path, maximum=4 * 1024 * 1024 * 1024
    )
    run_id = fleet_manifest.get("runId")
    fleet_scenario = fleet_manifest.get("scenario")
    fleet_release = (
        fleet_scenario.get("release") if isinstance(fleet_scenario, dict) else None
    )
    runtime_pids = fleet_evidence.get("runtimePids")
    if (
        verified_bom.schema_version != 2
        or verified_bom.agent_version != version
        or verified_bom.component_sha != component_sha
        or verified_bom.artifact_kind != "agent-bundle"
        or verified_bom.artifact_name != artifact_path.name
        or verified_bom.artifact_sha256 != artifact_sha
        or verified_bom.artifact_size_bytes != artifact_size
        or not isinstance(run_id, str)
        or not isinstance(fleet_scenario, dict)
        or not isinstance(fleet_release, dict)
        or not isinstance(runtime_pids, dict)
        or fleet_scenario.get("type") != "oak-fault-rollback"
        or fleet_release.get("agentVersion") != version
        or fleet_release.get("componentSha") != component_sha
    ):
        raise AssemblyError("artifact, BOM, and Fleet evidence identities differ")

    expected_soak_keys = {
        "schemaVersion",
        "kind",
        "startedAt",
        "outcome",
        "board",
        "oakMxidSha256",
        "deviceIdentity",
        "runtimeIdentity",
        "soak",
        "webrtc",
        "splitmux",
    }
    soak_schema = soak.get("schemaVersion")
    if soak_schema == 3:
        expected_soak_keys.update({"runId", "slotKind"})
    if (
        set(soak) != expected_soak_keys
        or type(soak.get("schemaVersion")) is not int
        or soak_schema not in {2, 3}
        or soak.get("kind") != "nuvion-iq9075-oak-soak-result"
        or (version in CANDIDATE_SOAK_REQUIRED_VERSIONS and soak_schema != 3)
        or (soak_schema == 3 and candidate_soak is None)
        or (soak_schema == 2 and candidate_soak is not None)
    ):
        raise AssemblyError("OAK soak result schema is invalid")
    outcome = soak.get("outcome")
    if (
        not isinstance(outcome, dict)
        or set(outcome) != {"status", "error", "cleanupErrors"}
        or outcome.get("status") != "passed"
        or outcome.get("error") is not None
        or outcome.get("cleanupErrors") != []
    ):
        raise AssemblyError("failed OAK soak result cannot be promoted")

    output_directory = _private_output_directory(output_directory)
    names = {
        "oak": f"iq9075-v{version}-oak-soak-result.json",
        "fleet_manifest": f"iq9075-v{version}-fleet-manifest.json",
        "fleet_evidence": f"iq9075-v{version}-fleet-evidence.json",
        "bom": f"nuv-agent_{version}_iq9075-aarch64.release-bom.json",
        "manifest": f"iq9075-v{version}-harness-manifest.json",
        "result": f"iq9075-v{version}-harness-result.json",
        "summary": f"iq9075-v{version}-physical-evidence.json",
    }
    if candidate_soak is not None:
        names["candidate_soak"] = (
            f"iq9075-v{version}-candidate-soak-evidence.json"
        )
        names["cleanup_evidence"] = f"iq9075-v{version}-cleanup-evidence.json"
    paths = {key: _safe_output(output_directory, name) for key, name in names.items()}
    for source, key, raw in (
        (soak_result_path, "oak", soak_raw),
        (fleet_manifest_path, "fleet_manifest", fleet_manifest_raw),
        (fleet_evidence_path, "fleet_evidence", fleet_evidence_raw),
        (bom_path, "bom", bom_raw),
    ):
        _copy_input(source, paths[key], raw)
    if candidate_soak_evidence_path is not None and candidate_soak_raw is not None:
        _copy_input(
            candidate_soak_evidence_path,
            paths["candidate_soak"],
            candidate_soak_raw,
        )

    harness_sha = _digest(_regular_bytes(candidate_harness))
    candidate_fleet_runner_raw = _regular_bytes(candidate_fleet_runner)
    publisher_fleet_runner = (
        Path(__file__).resolve().parents[1] / "dev/run-iq9075-fleet-e2e.py"
    )
    publisher_fleet_runner_raw = _regular_bytes(publisher_fleet_runner)
    if candidate_fleet_runner_raw != publisher_fleet_runner_raw:
        raise AssemblyError(
            "candidate Fleet runner bytes differ from trusted publisher runner"
        )
    fleet_runner_sha = _digest(publisher_fleet_runner_raw)
    board_tool_sha = _digest(_regular_bytes(candidate_board_tool))
    if fleet_manifest.get("toolSha256") != board_tool_sha:
        raise AssemblyError("Fleet manifest does not bind the candidate board tool")
    if candidate_soak is not None:
        module_name = "_nuvion_candidate_soak_validator"
        try:
            validator = _module_from_verified_source(
                module_name=module_name,
                source=publisher_fleet_runner_raw,
                display_path=publisher_fleet_runner,
            )
            validator.validate_candidate_soak_evidence(
                candidate_soak,
                run_id=str(run_id),
                manifest=fleet_manifest,
                bundle_sha256=artifact_sha,
                bom_sha256=_digest(bom_raw),
                harness_sha256=harness_sha,
                fleet_evidence_sha256=_digest(fleet_evidence_raw),
                raw_evidence_sha256=_digest(soak_raw),
                require_cleanup_evidence=True,
            )
        except Exception as exc:
            raise AssemblyError("candidate soak evidence is invalid") from exc
        finally:
            sys.modules.pop(module_name, None)
        if candidate_soak.get("rawEvidence") != soak:
            raise AssemblyError("candidate soak wrapper differs from raw OAK evidence")
        if candidate_soak.get("rawEvidenceSha256") != _digest(soak_raw):
            raise AssemblyError("candidate soak wrapper raw bytes digest differs")
        cleanup_evidence = candidate_soak.get("cleanupEvidence")
        if not isinstance(cleanup_evidence, dict):
            raise AssemblyError("candidate cleanup evidence is missing")
        cleanup_evidence_raw = _canonical(cleanup_evidence)
        if candidate_soak.get("cleanupEvidenceSha256") != _digest(
            cleanup_evidence_raw
        ):
            raise AssemblyError("candidate cleanup evidence bytes differ")
        candidate_post = candidate_soak.get("post")
        if not isinstance(candidate_post, dict):
            raise AssemblyError("candidate soak restoration proof is invalid")
        fleet_at = _timestamp(
            fleet_evidence.get("generatedAt"), label="Fleet rollback time"
        )
        candidate_at = _timestamp(
            candidate_soak.get("startedAt"), label="candidate operation start time"
        )
        soak_at = _timestamp(soak.get("startedAt"), label="candidate soak start time")
        restored_at = _timestamp(
            candidate_post.get("restoredAt"), label="candidate restoration time"
        )
        completed_at = _timestamp(
            candidate_soak.get("completedAt"),
            label="candidate operation completion time",
        )
        if not (
            fleet_at <= candidate_at <= soak_at <= restored_at <= completed_at
            and completed_at - fleet_at <= dt.timedelta(hours=24)
        ):
            raise AssemblyError("candidate evidence lifecycle ordering is invalid")
    iq_policy = security.get("iq9075")
    baseline = (
        iq_policy.get("legacyPromotedBaseline")
        if isinstance(iq_policy, dict)
        else None
    )
    if not isinstance(baseline, dict):
        raise AssemblyError("IQ9075 rollback policy is missing")

    manifest = {
        "schemaVersion": 1,
        "kind": "nuvion-iq9075-physical-manifest",
        "runId": run_id,
        "agentVersion": version,
        "componentSha": component_sha,
        "harnessSha256": harness_sha,
        "fleetRunnerSha256": fleet_runner_sha,
        "oakSoak": {"file": names["oak"], "sha256": _digest(soak_raw)},
        "fleetManifest": {
            "file": names["fleet_manifest"],
            "sha256": _digest(fleet_manifest_raw),
        },
        "fleetEvidence": {
            "file": names["fleet_evidence"],
            "sha256": _digest(fleet_evidence_raw),
        },
        "testedArtifact": {
            "name": artifact_path.name,
            "sha256": artifact_sha,
            "sizeBytes": artifact_size,
        },
        "testedBom": {"file": names["bom"], "sha256": _digest(bom_raw)},
        "board": soak.get("board"),
        "oakMxidSha256": soak.get("oakMxidSha256"),
        "startedAt": soak.get("startedAt"),
        "expectedRollback": {
            "agentVersion": baseline.get("agentVersion"),
            "slot": "releases/" + str(baseline.get("bomDigest", ""))[7:],
        },
    }
    if candidate_soak_raw is not None:
        manifest["candidateSoak"] = {
            "file": names["candidate_soak"],
            "sha256": _digest(candidate_soak_raw),
        }
        if cleanup_evidence_raw is None:
            raise AssemblyError("candidate cleanup evidence bytes are missing")
        _write_new(paths["cleanup_evidence"], cleanup_evidence_raw)
        manifest["cleanupEvidence"] = {
            "file": names["cleanup_evidence"],
            "sha256": _digest(cleanup_evidence_raw),
        }
    manifest_raw = _canonical(manifest)
    _write_new(paths["manifest"], manifest_raw)

    result = {
        "schemaVersion": 3 if candidate_soak is not None else 2,
        "kind": "nuvion-iq9075-physical-result",
        "runId": run_id,
        "agentVersion": version,
        "componentSha": component_sha,
        "manifestSha256": _digest(manifest_raw),
        "artifactSha256": artifact_sha,
        "bomSha256": _digest(bom_raw),
        "exitCode": 0,
        "outcome": soak.get("outcome"),
        "soak": soak.get("soak"),
        "webrtc": soak.get("webrtc"),
        "splitmux": soak.get("splitmux"),
        "rollback": {
            "expectedSlot": fleet_scenario.get("expectedPreviousSlot"),
            "candidateSlot": "releases/" + verified_bom.bom_digest,
            "restoredSlot": fleet_scenario.get("expectedPreviousSlot"),
            "candidatePid": runtime_pids.get("candidate"),
            "restoredPid": runtime_pids.get("restored"),
            "oakProbeExitCode": 0,
            "oakReady": True,
        },
    }
    if candidate_soak is not None:
        result["candidateRestore"] = candidate_soak.get("post")
        assert cleanup_evidence_raw is not None
        result["cleanupEvidenceSha256"] = _digest(cleanup_evidence_raw)
    result_raw = _canonical(result)
    _write_new(paths["result"], result_raw)

    soak_metrics = result["soak"]
    duration = float(soak_metrics["durationSeconds"])
    rss_samples = [
        (float(item["elapsedSec"]) / 60.0, float(item["rssAnonKiB"]) / 1024.0)
        for item in soak_metrics["rssAnonSamples"]
    ]
    mean_x = sum(item[0] for item in rss_samples) / len(rss_samples)
    mean_y = sum(item[1] for item in rss_samples) / len(rss_samples)
    denominator = sum((item[0] - mean_x) ** 2 for item in rss_samples)
    if not math.isfinite(duration) or duration <= 0 or denominator <= 0:
        raise AssemblyError("OAK soak metrics cannot be summarized")
    slope = sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in rss_samples
    ) / denominator
    rss_values = [item[1] for item in rss_samples]
    summary = {
        "schemaVersion": 2,
        "kind": "nuvion-iq9075-physical-release-evidence",
        "agentVersion": version,
        "componentSha": component_sha,
        "harnessManifest": {
            "file": names["manifest"],
            "sha256": _digest(manifest_raw),
        },
        "harnessResult": {
            "file": names["result"],
            "sha256": _digest(result_raw),
        },
        "physicalGate": {
            "oakSoakSeconds": round(duration, 6),
            "rawFps": round(float(soak_metrics["rawSamples"]) / duration, 6),
            "rssSlopeMiBPerMinute": round(slope, 6),
            "rssRangeMiB": round(max(rss_values) - min(rss_values), 6),
            "gstreamerErrors": len(soak_metrics["gstreamerErrors"]),
            "webrtcBranchDisposed": True,
            "splitmuxRotated": True,
            "rollbackOakReady": True,
        },
    }
    if cleanup_evidence_raw is not None:
        summary["cleanupEvidence"] = {
            "file": names["cleanup_evidence"],
            "sha256": _digest(cleanup_evidence_raw),
        }
    summary_raw = _canonical(summary)
    _write_new(paths["summary"], summary_raw)

    readiness = _load_readiness_validator()
    try:
        readiness._validate_physical_documents(
            policy_path=output_directory / "release-readiness.json",
            version=version,
            component_sha=component_sha,
            summary=summary,
            security=security,
            candidate_harness=candidate_harness,
        )
    except Exception as exc:
        raise AssemblyError("assembled IQ9075 evidence failed validation") from exc
    return {
        "summary": str(paths["summary"]),
        "summarySha256": _digest(summary_raw),
        "manifest": str(paths["manifest"]),
        "result": str(paths["result"]),
        "artifactSha256": artifact_sha,
        "bomSha256": _digest(bom_raw),
    }


def assemble(
    *,
    soak_result_path: Path,
    fleet_manifest_path: Path,
    fleet_evidence_path: Path,
    artifact_path: Path,
    bom_path: Path,
    candidate_harness: Path,
    candidate_fleet_runner: Path,
    candidate_board_tool: Path,
    security_policy_path: Path,
    output_directory: Path,
    version: str,
    component_sha: str,
    candidate_soak_evidence_path: Path | None = None,
) -> dict[str, str]:
    final_root = _private_output_directory(output_directory)
    with tempfile.TemporaryDirectory(
        prefix=".iq9075-physical-evidence-", dir=final_root
    ) as raw_staging:
        staging = Path(raw_staging)
        staged = _assemble_into(
            soak_result_path=soak_result_path,
            fleet_manifest_path=fleet_manifest_path,
            fleet_evidence_path=fleet_evidence_path,
            artifact_path=artifact_path,
            bom_path=bom_path,
            candidate_harness=candidate_harness,
            candidate_fleet_runner=candidate_fleet_runner,
            candidate_board_tool=candidate_board_tool,
            security_policy_path=security_policy_path,
            output_directory=staging,
            version=version,
            component_sha=component_sha,
            candidate_soak_evidence_path=candidate_soak_evidence_path,
        )
        staged_files = sorted(path for path in staging.iterdir() if path.is_file())
        if len(staged_files) != (9 if candidate_soak_evidence_path else 7):
            raise AssemblyError("staged IQ9075 evidence file set is incomplete")
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
        for key in ("summary", "manifest", "result"):
            result[key] = str(final_root / Path(result[key]).name)
        return result


def main() -> int:
    if not sys.flags.isolated:
        print(
            "assemble-iq9075-physical-evidence.py requires Python isolated mode (-I)",
            file=sys.stderr,
        )
        return 2
    parser = argparse.ArgumentParser()
    parser.add_argument("--soak-result", required=True, type=Path)
    parser.add_argument("--fleet-manifest", required=True, type=Path)
    parser.add_argument("--fleet-evidence", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--bom", required=True, type=Path)
    parser.add_argument("--candidate-harness", required=True, type=Path)
    parser.add_argument("--candidate-fleet-runner", required=True, type=Path)
    parser.add_argument("--candidate-board-tool", required=True, type=Path)
    parser.add_argument("--security-policy", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--component-sha", required=True)
    parser.add_argument("--candidate-soak-evidence", type=Path)
    arguments = parser.parse_args()
    try:
        result = assemble(
            soak_result_path=arguments.soak_result,
            fleet_manifest_path=arguments.fleet_manifest,
            fleet_evidence_path=arguments.fleet_evidence,
            artifact_path=arguments.artifact,
            bom_path=arguments.bom,
            candidate_harness=arguments.candidate_harness,
            candidate_fleet_runner=arguments.candidate_fleet_runner,
            candidate_board_tool=arguments.candidate_board_tool,
            security_policy_path=arguments.security_policy,
            output_directory=arguments.output_directory,
            version=arguments.version,
            component_sha=arguments.component_sha,
            candidate_soak_evidence_path=arguments.candidate_soak_evidence,
        )
    except AssemblyError as exc:
        print(f"IQ9075 physical evidence assembly failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
