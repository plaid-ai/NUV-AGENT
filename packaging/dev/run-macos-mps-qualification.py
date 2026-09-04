#!/usr/bin/env python3
"""Run the optional, local Apple Silicon Formula/MPS qualification.

This runner deliberately does not install or mutate a Homebrew Formula.  It
qualifies an already installed, source-built ``nuvion/release-gate/nuv-agent``
Formula only when its stamped build identity exactly matches the clean checkout.
The public SigLIP snapshot is fetched anonymously into a private temporary
directory, then both the CPU oracle and the real MPS proof run offline from the
installed Formula's isolated Python.

Stdout contains exactly one canonical JSON result on success or failure.  Human
diagnostics go to stderr.  ``--help`` is safe on every operating system.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 can still display --help safely.
    tomllib = None  # type: ignore[assignment]


SCHEMA_VERSION = 1
QUALIFICATION = "macos-mps"
DEFAULT_FORMULA = "nuvion/release-gate/nuv-agent"
MODEL_REPO = "google/siglip2-base-patch16-224"
MODEL_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
LABELS = ("normal scene", "anomalous scene")
REPEAT_COUNT = 16
PARITY_RTOL = 5e-2
PARITY_ATOL = 1e-5
PHYSICAL_MEMORY_FLOOR_BYTES = 12 * 1024**3
RECOMMENDED_MEMORY_FLOOR_BYTES = 8 * 1024**3
ALLOCATED_MEMORY_LIMIT_BYTES = 512 * 1024**2
DRIVER_MEMORY_LIMIT_BYTES = 1024 * 1024**2
GROWTH_LIMIT_BYTES = 16 * 1024**2
MAX_JSON_BYTES = 1024 * 1024
MAX_RECEIPT_BYTES = 4 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 3600
MIN_TIMEOUT_SECONDS = 60
MAX_TIMEOUT_SECONDS = 7200

EXPECTED_RUNTIME_VERSIONS = {
    "Pillow": "12.1.0",
    "hf-xet": "1.6.0",
    "huggingface-hub": "1.29.0",
    "numpy": "2.4.2",
    "protobuf": "7.36.1",
    "safetensors": "0.8.0",
    "sentencepiece": "0.2.2",
    "tokenizers": "0.23.1",
    "torch": "2.10.0",
    "transformers": "5.16.1",
}

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_FORMULA_RE = re.compile(
    r"^[a-z0-9][a-z0-9_.-]*/[a-z0-9][a-z0-9_.-]*/"
    r"[a-z0-9][a-z0-9@+_.-]*$"
)


class QualificationError(RuntimeError):
    """Expected, fail-closed qualification error with a stable code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise QualificationError(code, message)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value: object) -> bool:
    return type(value) is int


def _exact_mapping(
    value: object, expected_keys: set[str], *, code: str, name: str
) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), code, f"{name} must be an object")
    assert isinstance(value, Mapping)
    keys = set(value)
    _require(keys == expected_keys, code, f"{name} schema keys are invalid")
    return value


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def parse_json_document(raw: str, *, code: str, name: str) -> Any:
    _require(bool(raw) and len(raw.encode("utf-8")) <= MAX_JSON_BYTES, code, f"{name} size is invalid")
    try:
        return json.loads(
            raw,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {token}")
            ),
        )
    except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
        raise QualificationError(code, f"{name} is not strict JSON") from exc


def validate_component_sha(value: object, *, code: str = "identity-invalid") -> str:
    _require(isinstance(value, str) and bool(_SHA_RE.fullmatch(value)), code, "component SHA must be 40 lowercase hex characters")
    assert isinstance(value, str)
    return value


def validate_agent_version(value: object, *, code: str = "identity-invalid") -> str:
    _require(isinstance(value, str) and bool(_VERSION_RE.fullmatch(value)), code, "agent version must be strict SemVer core")
    assert isinstance(value, str)
    return value


def _validate_scores(value: object, *, code: str, name: str) -> list[float]:
    _require(isinstance(value, list) and len(value) == len(LABELS), code, f"{name} shape is invalid")
    assert isinstance(value, list)
    _require(
        all(_is_number(score) and math.isfinite(float(score)) and 0.0 <= float(score) <= 1.0 for score in value),
        code,
        f"{name} values are invalid",
    )
    return [float(score) for score in value]


def validate_reference(
    value: object, *, component_sha: str, agent_version: str
) -> dict[str, Any]:
    code = "cpu-reference-invalid"
    document = _exact_mapping(
        value,
        {
            "schemaVersion",
            "componentSha",
            "agentVersion",
            "modelRevision",
            "labels",
            "scores",
        },
        code=code,
        name="CPU reference",
    )
    _require(_is_int(document["schemaVersion"]) and document["schemaVersion"] == SCHEMA_VERSION, code, "CPU reference schema version is invalid")
    _require(document["componentSha"] == component_sha, code, "CPU reference component SHA differs")
    _require(document["agentVersion"] == agent_version, code, "CPU reference agent version differs")
    _require(document["modelRevision"] == MODEL_REVISION, code, "CPU reference model revision differs")
    _require(document["labels"] == list(LABELS), code, "CPU reference labels differ")
    scores = _validate_scores(document["scores"], code=code, name="CPU reference scores")
    return {
        "schemaVersion": SCHEMA_VERSION,
        "componentSha": component_sha,
        "agentVersion": agent_version,
        "modelRevision": MODEL_REVISION,
        "labels": list(LABELS),
        "scores": scores,
    }


def validate_install_receipt(
    value: object, *, formula: str, agent_version: str
) -> dict[str, Any]:
    """Validate the Homebrew receipt facts required for a source-built Formula."""

    code = "formula-receipt-invalid"
    _require(isinstance(value, Mapping), code, "Formula install receipt must be an object")
    assert isinstance(value, Mapping)
    for key in ("built_as_bottle", "poured_from_bottle", "loaded_from_api", "source", "arch", "built_on"):
        _require(key in value, code, f"Formula install receipt is missing {key}")
    _require(value["built_as_bottle"] is False, code, "Formula was built as a bottle, not from candidate source")
    _require(value["poured_from_bottle"] is False, code, "Formula was poured from a bottle, not built from source")
    _require(value["loaded_from_api"] is False, code, "Formula was loaded from the Homebrew API instead of the candidate tap")
    _require(value["arch"] == "arm64", code, "Formula receipt architecture is not arm64")

    built_on = value["built_on"]
    _require(isinstance(built_on, Mapping) and built_on.get("os") == "Macintosh", code, "Formula receipt does not record a macOS build")
    source = value["source"]
    _require(isinstance(source, Mapping), code, "Formula receipt source is invalid")
    assert isinstance(source, Mapping)
    expected_tap = "/".join(formula.split("/")[:2])
    _require(source.get("tap") == expected_tap, code, "Formula receipt tap differs from the requested candidate tap")
    _require(source.get("spec") == "stable", code, "Formula receipt is not the stable source spec")
    versions = source.get("versions")
    _require(isinstance(versions, Mapping) and versions.get("stable") == agent_version, code, "Formula receipt version differs from the checkout")
    return {
        "arch": "arm64",
        "sourceBuilt": True,
        "tap": expected_tap,
        "version": agent_version,
    }


def validate_formula_identity(
    value: object,
    *,
    component_sha: str,
    agent_version: str,
    formula_prefix: Path,
) -> dict[str, Any]:
    code = "formula-identity-invalid"
    document = _exact_mapping(
        value,
        {
            "schemaVersion",
            "componentSha",
            "agentVersion",
            "packagePath",
            "pythonVersion",
            "runtimeVersions",
            "sysPrefix",
            "mpsAvailable",
            "physicalMemoryBytes",
            "recommendedMaxMemoryBytes",
        },
        code=code,
        name="Formula identity",
    )
    _require(_is_int(document["schemaVersion"]) and document["schemaVersion"] == SCHEMA_VERSION, code, "Formula identity schema version is invalid")
    _require(document["componentSha"] == component_sha, code, "installed Formula component SHA differs from checkout")
    _require(document["agentVersion"] == agent_version, code, "installed Formula agent version differs from checkout")
    _require(document["pythonVersion"] == "3.14", code, "Formula Python is not 3.14")
    _require(document["runtimeVersions"] == EXPECTED_RUNTIME_VERSIONS, code, "Formula inference tuple differs from the locked tuple")
    _require(document["mpsAvailable"] is True, code, "real MPS is unavailable to the Formula runtime")

    libexec = (formula_prefix / "libexec").resolve(strict=True)
    try:
        package_path = Path(str(document["packagePath"])).resolve(strict=True)
        sys_prefix = Path(str(document["sysPrefix"])).resolve(strict=True)
    except OSError as exc:
        raise QualificationError(code, "Formula identity paths cannot be resolved") from exc
    _require(package_path.is_relative_to(libexec), code, "nuvion_app imported outside Formula libexec")
    _require(sys_prefix == libexec, code, "Formula Python sys.prefix is not Formula libexec")

    physical = document["physicalMemoryBytes"]
    recommended = document["recommendedMaxMemoryBytes"]
    _require(_is_int(physical) and physical >= PHYSICAL_MEMORY_FLOOR_BYTES, code, "physical memory is below the 12 GiB qualification floor")
    _require(_is_int(recommended) and recommended >= RECOMMENDED_MEMORY_FLOOR_BYTES, code, "MPS recommended memory is below the 8 GiB qualification floor")
    return dict(document)


def validate_mps_proof(
    value: object, *, component_sha: str, agent_version: str
) -> dict[str, Any]:
    code = "mps-proof-invalid"
    document = _exact_mapping(
        value,
        {
            "schemaVersion",
            "componentSha",
            "agentVersion",
            "modelRevision",
            "labels",
            "cpuScores",
            "mpsScores",
            "offline",
            "mpsAvailable",
            "device",
            "dtype",
            "visionModelClass",
            "visionParameterDevices",
            "visionParameterDtypes",
            "persistentStatePacked",
            "firstInference",
            "repeatCount",
            "physicalMemoryBytes",
            "recommendedMaxMemoryBytes",
            "stableAllocatedBytes",
            "finalAllocatedBytes",
            "driverAllocatedBytes",
        },
        code=code,
        name="MPS proof",
    )
    _require(_is_int(document["schemaVersion"]) and document["schemaVersion"] == SCHEMA_VERSION, code, "MPS proof schema version is invalid")
    _require(document["componentSha"] == component_sha, code, "MPS proof component SHA differs")
    _require(document["agentVersion"] == agent_version, code, "MPS proof agent version differs")
    _require(document["modelRevision"] == MODEL_REVISION, code, "MPS proof model revision differs")
    _require(document["labels"] == list(LABELS), code, "MPS proof labels differ")
    cpu_scores = _validate_scores(document["cpuScores"], code=code, name="MPS proof CPU scores")
    mps_scores = _validate_scores(document["mpsScores"], code=code, name="MPS proof MPS scores")
    _require(
        all(math.isclose(mps, cpu, rel_tol=PARITY_RTOL, abs_tol=PARITY_ATOL) for cpu, mps in zip(cpu_scores, mps_scores)),
        code,
        "CPU and MPS scores are outside the parity tolerance",
    )
    _require(document["offline"] is True, code, "MPS inference was not offline")
    _require(document["mpsAvailable"] is True and document["device"] == "mps", code, "proof did not execute on real MPS")
    _require(document["dtype"] == "float16", code, "MPS inference dtype is not float16")
    _require(document["visionModelClass"] == "SiglipVisionModel", code, "MPS proof did not use the isolated SigLIP vision tower")
    _require(document["visionParameterDevices"] == ["mps"], code, "vision parameters are not exclusively on MPS")
    _require(document["visionParameterDtypes"] == ["float16"], code, "vision parameters are not exclusively float16")
    _require(document["persistentStatePacked"] is True, code, "MPS scoring state is not packed persistently")
    _require(document["firstInference"] is True, code, "first MPS inference did not complete")
    _require(_is_int(document["repeatCount"]) and document["repeatCount"] == REPEAT_COUNT, code, "MPS repeat count is not exactly 16")

    physical = document["physicalMemoryBytes"]
    recommended = document["recommendedMaxMemoryBytes"]
    stable = document["stableAllocatedBytes"]
    final = document["finalAllocatedBytes"]
    driver = document["driverAllocatedBytes"]
    _require(_is_int(physical) and physical >= PHYSICAL_MEMORY_FLOOR_BYTES, code, "physical memory floor was not met")
    _require(_is_int(recommended) and recommended >= RECOMMENDED_MEMORY_FLOOR_BYTES, code, "MPS recommended memory floor was not met")
    _require(_is_int(stable) and stable >= 0, code, "stable MPS allocation is invalid")
    _require(_is_int(final) and final >= 0, code, "final MPS allocation is invalid")
    _require(_is_int(driver) and driver >= 0, code, "MPS driver allocation is invalid")
    _require(final <= stable + GROWTH_LIMIT_BYTES, code, "MPS allocation grew beyond 16 MiB")
    _require(final < ALLOCATED_MEMORY_LIMIT_BYTES, code, "active MPS allocation reached 512 MiB")
    _require(driver < DRIVER_MEMORY_LIMIT_BYTES, code, "MPS driver allocation reached 1 GiB")
    return dict(document)


def validate_result(value: object) -> dict[str, Any]:
    code = "result-schema-invalid"
    _require(isinstance(value, Mapping), code, "qualification result must be an object")
    assert isinstance(value, Mapping)
    status = value.get("status")
    if status == "failed":
        document = _exact_mapping(
            value,
            {"schemaVersion", "qualification", "status", "error"},
            code=code,
            name="failed qualification result",
        )
        error = _exact_mapping(document["error"], {"code", "message"}, code=code, name="qualification error")
        _require(isinstance(error["code"], str) and bool(error["code"]), code, "qualification error code is invalid")
        _require(isinstance(error["message"], str) and bool(error["message"]), code, "qualification error message is invalid")
    elif status == "passed":
        document = _exact_mapping(
            value,
            {"schemaVersion", "qualification", "status", "candidate", "formula", "model", "parity", "mps", "memory"},
            code=code,
            name="passed qualification result",
        )
        candidate = _exact_mapping(document["candidate"], {"componentSha", "agentVersion", "cleanCheckout"}, code=code, name="candidate result")
        validate_component_sha(candidate["componentSha"], code=code)
        validate_agent_version(candidate["agentVersion"], code=code)
        _require(candidate["cleanCheckout"] is True, code, "candidate checkout was not clean")
        formula = _exact_mapping(document["formula"], {"name", "prefix", "pythonVersion", "runtimeVersions", "sourceBuilt", "isolated"}, code=code, name="Formula result")
        _require(isinstance(formula["name"], str) and bool(_FORMULA_RE.fullmatch(formula["name"])), code, "Formula result name is invalid")
        _require(isinstance(formula["prefix"], str) and Path(formula["prefix"]).is_absolute(), code, "Formula result prefix is invalid")
        _require(formula["pythonVersion"] == "3.14", code, "Formula result Python differs")
        _require(formula["runtimeVersions"] == EXPECTED_RUNTIME_VERSIONS, code, "Formula result tuple differs")
        _require(formula["sourceBuilt"] is True and formula["isolated"] is True, code, "Formula result isolation facts are invalid")
        model = _exact_mapping(document["model"], {"repoId", "revision", "offline", "anonymousDownload"}, code=code, name="model result")
        _require(model == {"repoId": MODEL_REPO, "revision": MODEL_REVISION, "offline": True, "anonymousDownload": True}, code, "model result differs from the reviewed snapshot policy")
        parity = _exact_mapping(document["parity"], {"labels", "cpuScores", "mpsScores", "rtol", "atol"}, code=code, name="parity result")
        _require(parity["labels"] == list(LABELS), code, "parity labels differ")
        cpu_scores = _validate_scores(parity["cpuScores"], code=code, name="result CPU scores")
        mps_scores = _validate_scores(parity["mpsScores"], code=code, name="result MPS scores")
        _require(parity["rtol"] == PARITY_RTOL and parity["atol"] == PARITY_ATOL, code, "parity tolerances differ")
        _require(all(math.isclose(mps, cpu, rel_tol=PARITY_RTOL, abs_tol=PARITY_ATOL) for cpu, mps in zip(cpu_scores, mps_scores)), code, "result parity failed")
        mps = _exact_mapping(document["mps"], {"available", "device", "dtype", "visionModelClass", "visionParameterDevices", "visionParameterDtypes", "persistentStatePacked", "firstInference", "repeatCount"}, code=code, name="MPS result")
        _require(mps == {"available": True, "device": "mps", "dtype": "float16", "visionModelClass": "SiglipVisionModel", "visionParameterDevices": ["mps"], "visionParameterDtypes": ["float16"], "persistentStatePacked": True, "firstInference": True, "repeatCount": REPEAT_COUNT}, code, "MPS result facts differ")
        memory = _exact_mapping(document["memory"], {"physicalBytes", "physicalFloorBytes", "recommendedMaxBytes", "recommendedFloorBytes", "stableAllocatedBytes", "finalAllocatedBytes", "growthBytes", "growthLimitBytes", "allocatedLimitBytes", "driverAllocatedBytes", "driverLimitBytes"}, code=code, name="memory result")
        _require(memory["physicalFloorBytes"] == PHYSICAL_MEMORY_FLOOR_BYTES and memory["recommendedFloorBytes"] == RECOMMENDED_MEMORY_FLOOR_BYTES, code, "memory floors differ")
        _require(memory["growthLimitBytes"] == GROWTH_LIMIT_BYTES and memory["allocatedLimitBytes"] == ALLOCATED_MEMORY_LIMIT_BYTES and memory["driverLimitBytes"] == DRIVER_MEMORY_LIMIT_BYTES, code, "memory limits differ")
        for key in ("physicalBytes", "recommendedMaxBytes", "stableAllocatedBytes", "finalAllocatedBytes", "growthBytes", "driverAllocatedBytes"):
            _require(_is_int(memory[key]), code, f"memory result {key} is invalid")
        _require(memory["physicalBytes"] >= PHYSICAL_MEMORY_FLOOR_BYTES, code, "result physical floor failed")
        _require(memory["recommendedMaxBytes"] >= RECOMMENDED_MEMORY_FLOOR_BYTES, code, "result recommended floor failed")
        _require(memory["growthBytes"] == memory["finalAllocatedBytes"] - memory["stableAllocatedBytes"], code, "result growth is inconsistent")
        _require(memory["finalAllocatedBytes"] <= memory["stableAllocatedBytes"] + GROWTH_LIMIT_BYTES, code, "result growth bound failed")
        _require(memory["finalAllocatedBytes"] < ALLOCATED_MEMORY_LIMIT_BYTES and memory["driverAllocatedBytes"] < DRIVER_MEMORY_LIMIT_BYTES, code, "result allocation bound failed")
    else:
        raise QualificationError(code, "qualification result status is invalid")
    _require(_is_int(document["schemaVersion"]) and document["schemaVersion"] == SCHEMA_VERSION, code, "qualification result schema version is invalid")
    _require(document["qualification"] == QUALIFICATION, code, "qualification result kind is invalid")
    return dict(document)


def build_failure_result(error: QualificationError) -> dict[str, Any]:
    message = " ".join(str(error).split())[:1000] or "qualification failed"
    result = {
        "schemaVersion": SCHEMA_VERSION,
        "qualification": QUALIFICATION,
        "status": "failed",
        "error": {"code": error.code, "message": message},
    }
    return validate_result(result)


def build_success_result(
    *,
    component_sha: str,
    agent_version: str,
    formula: str,
    formula_prefix: Path,
    proof: Mapping[str, Any],
) -> dict[str, Any]:
    stable = int(proof["stableAllocatedBytes"])
    final = int(proof["finalAllocatedBytes"])
    result = {
        "schemaVersion": SCHEMA_VERSION,
        "qualification": QUALIFICATION,
        "status": "passed",
        "candidate": {
            "componentSha": component_sha,
            "agentVersion": agent_version,
            "cleanCheckout": True,
        },
        "formula": {
            "name": formula,
            "prefix": str(formula_prefix),
            "pythonVersion": "3.14",
            "runtimeVersions": dict(EXPECTED_RUNTIME_VERSIONS),
            "sourceBuilt": True,
            "isolated": True,
        },
        "model": {
            "repoId": MODEL_REPO,
            "revision": MODEL_REVISION,
            "offline": True,
            "anonymousDownload": True,
        },
        "parity": {
            "labels": list(LABELS),
            "cpuScores": list(proof["cpuScores"]),
            "mpsScores": list(proof["mpsScores"]),
            "rtol": PARITY_RTOL,
            "atol": PARITY_ATOL,
        },
        "mps": {
            "available": True,
            "device": "mps",
            "dtype": "float16",
            "visionModelClass": "SiglipVisionModel",
            "visionParameterDevices": ["mps"],
            "visionParameterDtypes": ["float16"],
            "persistentStatePacked": True,
            "firstInference": True,
            "repeatCount": REPEAT_COUNT,
        },
        "memory": {
            "physicalBytes": proof["physicalMemoryBytes"],
            "physicalFloorBytes": PHYSICAL_MEMORY_FLOOR_BYTES,
            "recommendedMaxBytes": proof["recommendedMaxMemoryBytes"],
            "recommendedFloorBytes": RECOMMENDED_MEMORY_FLOOR_BYTES,
            "stableAllocatedBytes": stable,
            "finalAllocatedBytes": final,
            "growthBytes": final - stable,
            "growthLimitBytes": GROWTH_LIMIT_BYTES,
            "allocatedLimitBytes": ALLOCATED_MEMORY_LIMIT_BYTES,
            "driverAllocatedBytes": proof["driverAllocatedBytes"],
            "driverLimitBytes": DRIVER_MEMORY_LIMIT_BYTES,
        },
    }
    return validate_result(result)


def _safe_environment(temp_root: Path, *, hf_home: Path | None = None, offline: bool = False) -> dict[str, str]:
    """Return a minimal environment that cannot forward caller credentials."""

    home = temp_root / "home"
    tmp = temp_root / "tmp"
    home.mkdir(mode=0o700, exist_ok=True)
    tmp.mkdir(mode=0o700, exist_ok=True)
    environment = {
        "HOME": str(home),
        "TMPDIR": str(tmp),
        "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOMEBREW_NO_ANALYTICS": "1",
        "HOMEBREW_NO_AUTO_UPDATE": "1",
        "HOMEBREW_NO_ENV_HINTS": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
    }
    if hf_home is not None:
        environment["HF_HOME"] = str(hf_home)
    if offline:
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
    return environment


def _run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout_seconds: int,
    code: str,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise QualificationError(code, f"command did not complete: {command[0]}") from exc
    if completed.returncode != 0:
        details = " ".join(completed.stderr[-2000:].split())
        suffix = f": {details}" if details else ""
        raise QualificationError(code, f"command failed with exit {completed.returncode}{suffix}")
    _require(len(completed.stdout.encode("utf-8")) <= MAX_JSON_BYTES, code, "command stdout exceeded the bound")
    return completed


def _run_json_worker(
    formula_python: Path,
    source: str,
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout_seconds: int,
    code: str,
    name: str,
) -> Any:
    completed = _run_command(
        [str(formula_python), "-I", "-c", source],
        cwd=cwd,
        environment=environment,
        timeout_seconds=timeout_seconds,
        code=code,
    )
    return parse_json_document(completed.stdout.strip(), code=code, name=name)


def _read_regular_json(path: Path, *, code: str, maximum_bytes: int) -> Any:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise QualificationError(code, f"required JSON file is unavailable: {path.name}") from exc
    _require(stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode), code, f"{path.name} is not a regular non-symlink file")
    _require(0 < metadata.st_size <= maximum_bytes, code, f"{path.name} size is invalid")
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise QualificationError(code, f"cannot read {path.name}") from exc
    return parse_json_document(raw, code=code, name=path.name)


def _write_private_json(path: Path, value: object) -> None:
    raw = (canonical_json(value) + "\n").encode("utf-8")
    _require(len(raw) <= MAX_JSON_BYTES, "cpu-reference-invalid", "CPU reference exceeds the size bound")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
    except OSError as exc:
        raise QualificationError("cpu-reference-invalid", "cannot create private CPU reference") from exc


def _candidate_identity(repo_root: Path, *, temp_root: Path, timeout_seconds: int) -> tuple[str, str]:
    git = shutil.which("git", path=_safe_environment(temp_root)["PATH"])
    _require(git is not None, "candidate-invalid", "git is required to bind the installed Formula to the checkout")
    environment = _safe_environment(temp_root)
    top = _run_command([git, "-C", str(repo_root), "rev-parse", "--show-toplevel"], cwd=temp_root, environment=environment, timeout_seconds=60, code="candidate-invalid").stdout.strip()
    _require(Path(top).resolve(strict=True) == repo_root, "candidate-invalid", "runner is not inside the expected repository root")
    status_output = _run_command([git, "-C", str(repo_root), "status", "--porcelain=v1", "--untracked-files=all"], cwd=temp_root, environment=environment, timeout_seconds=60, code="candidate-invalid").stdout
    _require(status_output == "", "candidate-dirty", "checkout must be clean before qualifying its stamped Formula")
    component_sha = _run_command([git, "-C", str(repo_root), "rev-parse", "HEAD"], cwd=temp_root, environment=environment, timeout_seconds=60, code="candidate-invalid").stdout.strip()
    validate_component_sha(component_sha)

    pyproject_path = repo_root / "pyproject.toml"
    try:
        metadata = pyproject_path.lstat()
        _require(stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode), "candidate-invalid", "pyproject.toml must be a regular non-symlink file")
        raw_project = pyproject_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise QualificationError("candidate-invalid", "cannot read candidate version from pyproject.toml") from exc
    _require(tomllib is not None, "python-unsupported", "qualification execution requires Python 3.11 or newer; --help remains available")
    try:
        project = tomllib.loads(raw_project)["project"]
        agent_version = project["version"]
    except (KeyError, ValueError) as exc:
        raise QualificationError("candidate-invalid", "cannot read candidate version from pyproject.toml") from exc
    validate_agent_version(agent_version)
    return component_sha, agent_version


def _resolve_formula(
    formula: str,
    *,
    agent_version: str,
    temp_root: Path,
    timeout_seconds: int,
) -> tuple[Path, Path, dict[str, Any]]:
    _require(bool(_FORMULA_RE.fullmatch(formula)), "formula-invalid", "--formula must be a fully qualified owner/tap/formula name")
    environment = _safe_environment(temp_root)
    brew = shutil.which("brew", path=environment["PATH"])
    _require(brew is not None, "formula-missing", "Homebrew is required; install the exact candidate Formula from source first")
    prefix_raw = _run_command([brew, "--prefix", formula], cwd=temp_root, environment=environment, timeout_seconds=60, code="formula-missing").stdout.strip()
    cellar_raw = _run_command([brew, "--cellar"], cwd=temp_root, environment=environment, timeout_seconds=60, code="formula-invalid").stdout.strip()
    _require(Path(prefix_raw).is_absolute() and Path(cellar_raw).is_absolute(), "formula-invalid", "Homebrew returned a non-absolute path")
    try:
        formula_prefix = Path(prefix_raw).resolve(strict=True)
        cellar = Path(cellar_raw).resolve(strict=True)
    except OSError as exc:
        raise QualificationError("formula-missing", "installed Formula prefix cannot be resolved") from exc
    _require(formula_prefix.is_relative_to(cellar), "formula-invalid", "Formula prefix is outside the Homebrew Cellar")
    receipt_path = formula_prefix / "INSTALL_RECEIPT.json"
    receipt = _read_regular_json(receipt_path, code="formula-receipt-invalid", maximum_bytes=MAX_RECEIPT_BYTES)
    receipt_facts = validate_install_receipt(receipt, formula=formula, agent_version=agent_version)
    formula_python = formula_prefix / "libexec/bin/python"
    _require(formula_python.exists() and os.access(formula_python, os.X_OK), "formula-missing", "installed Formula libexec Python is missing or not executable")
    return formula_prefix, formula_python, receipt_facts


FORMULA_IDENTITY_WORKER = r'''
import json
import os
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import nuvion_app
import torch
from nuvion_app import build_info

expected_versions = json.loads(os.environ["EXPECTED_RUNTIME_VERSIONS"])
formula_prefix = Path(os.environ["FORMULA_PREFIX"]).resolve(strict=True)
libexec = (formula_prefix / "libexec").resolve(strict=True)
package_path = Path(nuvion_app.__file__).resolve(strict=True)
assert platform.system() == "Darwin"
assert platform.machine() == "arm64"
assert sys.version_info[:2] == (3, 14)
assert Path(sys.prefix).resolve(strict=True) == libexec
assert package_path.is_relative_to(libexec)
assert build_info.COMPONENT_SHA == os.environ["CANDIDATE_SHA"]
assert build_info.AGENT_VERSION == os.environ["CANDIDATE_VERSION"]
assert {name: version(name) for name in expected_versions} == expected_versions
assert torch.backends.mps.is_available()
physical_memory = int(subprocess.check_output(
    ["/usr/sbin/sysctl", "-n", "hw.memsize"], text=True
).strip())
recommended_memory = int(torch.mps.recommended_max_memory())
assert physical_memory >= int(os.environ["PHYSICAL_MEMORY_FLOOR_BYTES"])
assert recommended_memory >= int(os.environ["RECOMMENDED_MEMORY_FLOOR_BYTES"])
print(json.dumps({
    "schemaVersion": 1,
    "componentSha": build_info.COMPONENT_SHA,
    "agentVersion": build_info.AGENT_VERSION,
    "packagePath": str(package_path),
    "pythonVersion": f"{sys.version_info.major}.{sys.version_info.minor}",
    "runtimeVersions": expected_versions,
    "sysPrefix": str(Path(sys.prefix).resolve(strict=True)),
    "mpsAvailable": True,
    "physicalMemoryBytes": physical_memory,
    "recommendedMaxMemoryBytes": recommended_memory,
}, sort_keys=True, separators=(",", ":"), allow_nan=False))
'''


SNAPSHOT_WORKER = r'''
import json
import os
from pathlib import Path

import huggingface_hub
from huggingface_hub import snapshot_download

formula_prefix = Path(os.environ["FORMULA_PREFIX"]).resolve(strict=True)
libexec = (formula_prefix / "libexec").resolve(strict=True)
hub_path = Path(huggingface_hub.__file__).resolve(strict=True)
assert hub_path.is_relative_to(libexec)
revision = os.environ["MODEL_REVISION"]
path = Path(snapshot_download(
    repo_id=os.environ["MODEL_REPO"],
    revision=revision,
    token=False,
)).resolve(strict=True)
hf_home = Path(os.environ["HF_HOME"]).resolve(strict=True)
assert path.is_relative_to(hf_home)
assert path.is_dir() and not path.is_symlink()
assert path.name == revision
for required in ("config.json", "model.safetensors"):
    target = path / required
    assert target.is_file()
print(json.dumps({
    "schemaVersion": 1,
    "repoId": os.environ["MODEL_REPO"],
    "revision": revision,
    "path": str(path),
}, sort_keys=True, separators=(",", ":"), allow_nan=False))
'''


CPU_REFERENCE_WORKER = r'''
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoProcessor
import nuvion_app
from nuvion_app import build_info

assert os.environ["HF_HUB_OFFLINE"] == "1"
assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
formula_prefix = Path(os.environ["FORMULA_PREFIX"]).resolve(strict=True)
assert Path(nuvion_app.__file__).resolve(strict=True).is_relative_to(formula_prefix / "libexec")
assert build_info.COMPONENT_SHA == os.environ["CANDIDATE_SHA"]
assert build_info.AGENT_VERSION == os.environ["CANDIDATE_VERSION"]
revision = os.environ["MODEL_REVISION"]
path = Path(os.environ["MODEL_PATH"]).resolve(strict=True)
assert path.name == revision
assert Path(snapshot_download(
    repo_id=os.environ["MODEL_REPO"],
    revision=revision,
    local_files_only=True,
    token=False,
)).resolve(strict=True) == path
labels = ["normal scene", "anomalous scene"]
texts = [f"This is a photo of {label}." for label in labels]
frame = np.zeros((224, 224, 3), dtype=np.uint8)
reference_model = AutoModel.from_pretrained(
    str(path), dtype=torch.float16, low_cpu_mem_usage=True
).eval()
reference_processor = AutoProcessor.from_pretrained(str(path))
reference_inputs = reference_processor(
    text=texts,
    images=frame,
    padding="max_length",
    max_length=64,
    return_tensors="pt",
)
reference_inputs = {
    key: value.to(dtype=torch.float16) if value.is_floating_point() else value
    for key, value in reference_inputs.items()
}
with torch.inference_mode():
    reference_outputs = reference_model(**reference_inputs)
reference_scores = torch.sigmoid(
    reference_outputs.logits_per_image
).squeeze(0).float()
scores = [float(score) for score in reference_scores]
assert len(scores) == len(labels)
assert all(math.isfinite(score) and 0.0 <= score <= 1.0 for score in scores)
print(json.dumps({
    "schemaVersion": 1,
    "componentSha": build_info.COMPONENT_SHA,
    "agentVersion": build_info.AGENT_VERSION,
    "modelRevision": revision,
    "labels": labels,
    "scores": scores,
}, sort_keys=True, separators=(",", ":"), allow_nan=False))
'''


MPS_PROOF_WORKER = r'''
import json
import math
import os
from pathlib import Path
import stat
import subprocess

import numpy as np
import torch
from huggingface_hub import snapshot_download
import nuvion_app
from nuvion_app import build_info
from nuvion_app.inference.zero_shot import ZeroShotAnomalyDetector

assert os.environ["HF_HUB_OFFLINE"] == "1"
assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
assert os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.0"
formula_prefix = Path(os.environ["FORMULA_PREFIX"]).resolve(strict=True)
assert Path(nuvion_app.__file__).resolve(strict=True).is_relative_to(formula_prefix / "libexec")
assert build_info.COMPONENT_SHA == os.environ["CANDIDATE_SHA"]
assert build_info.AGENT_VERSION == os.environ["CANDIDATE_VERSION"]
assert torch.backends.mps.is_available()
physical_memory = int(subprocess.check_output(
    ["/usr/sbin/sysctl", "-n", "hw.memsize"], text=True
).strip())
recommended_memory = int(torch.mps.recommended_max_memory())
assert physical_memory >= int(os.environ["PHYSICAL_MEMORY_FLOOR_BYTES"])
assert recommended_memory >= int(os.environ["RECOMMENDED_MEMORY_FLOOR_BYTES"])

revision = os.environ["MODEL_REVISION"]
path = Path(os.environ["MODEL_PATH"]).resolve(strict=True)
assert path.name == revision
assert Path(snapshot_download(
    repo_id=os.environ["MODEL_REPO"],
    revision=revision,
    local_files_only=True,
    token=False,
)).resolve(strict=True) == path

reference_path = Path(os.environ["REFERENCE_PATH"])
reference_stat = reference_path.lstat()
assert stat.S_ISREG(reference_stat.st_mode) and not stat.S_ISLNK(reference_stat.st_mode)
assert stat.S_IMODE(reference_stat.st_mode) == 0o600
assert 0 < reference_stat.st_size <= 1024 * 1024
reference = json.loads(reference_path.read_text(encoding="utf-8"))
assert set(reference) == {
    "schemaVersion", "componentSha", "agentVersion", "modelRevision", "labels", "scores"
}
assert type(reference["schemaVersion"]) is int and reference["schemaVersion"] == 1
assert reference["componentSha"] == build_info.COMPONENT_SHA
assert reference["agentVersion"] == build_info.AGENT_VERSION
assert reference["modelRevision"] == revision
labels = ["normal scene", "anomalous scene"]
assert reference["labels"] == labels
scores = reference["scores"]
assert isinstance(scores, list) and len(scores) == len(labels)
assert all(
    isinstance(score, (int, float))
    and not isinstance(score, bool)
    and math.isfinite(score)
    and 0.0 <= score <= 1.0
    for score in scores
)
reference_scores = torch.tensor(scores, dtype=torch.float32)
frame = np.zeros((224, 224, 3), dtype=np.uint8)

detector = ZeroShotAnomalyDetector(
    enabled=True,
    model_name=str(path),
    labels=labels,
    anomaly_labels=["anomalous scene"],
    threshold=0.5,
    device_preference="mps",
)
assert detector.ready and detector._device == "mps"
assert detector.loaded_model_source() == str(path)
assert detector._inference_dtype == torch.float16
assert detector._model.__class__.__name__ == "SiglipVisionModel"
assert not hasattr(detector._model, "text_model")
vision_root = getattr(detector._model, "vision_model", detector._model)
vision_parameters = tuple(vision_root.parameters())
assert vision_parameters
vision_parameter_devices = sorted({parameter.device.type for parameter in vision_parameters})
vision_parameter_dtypes = sorted({str(parameter.dtype).removeprefix("torch.") for parameter in vision_parameters})
assert vision_parameter_devices == ["mps"]
assert vision_parameter_dtypes == ["float16"]
assert detector._mps_text_features.device.type == "mps"
assert detector._mps_text_features.dtype == torch.float16
assert detector._mps_text_features._base is detector._mps_persistent_state
assert detector._mps_logit_scale._base is detector._mps_persistent_state
assert detector._mps_logit_bias._base is detector._mps_persistent_state
assert detector._mps_persistent_state.numel() == detector._mps_text_features.numel() + 2
for scalar in (detector._mps_logit_scale, detector._mps_logit_bias):
    assert scalar.device.type == "mps" and scalar.dtype == torch.float16

torch.mps.synchronize()
subprocess.run(["/bin/sync"], check=True)
first = detector.classify(frame)
assert first is not None and len(first["scores"]) == len(labels)
observed = {label: score for label, score in zip(first["labels"], first["scores"])}
observed_scores = torch.tensor([observed[label] for label in labels], dtype=torch.float32)
assert torch.allclose(observed_scores, reference_scores, rtol=5e-2, atol=1e-5)
torch.mps.synchronize()
stable_bytes = int(torch.mps.current_allocated_memory())
for _ in range(16):
    repeated = detector.classify(frame)
    assert repeated is not None and len(repeated["scores"]) == len(labels)
torch.mps.synchronize()
final_bytes = int(torch.mps.current_allocated_memory())
driver_bytes = int(torch.mps.driver_allocated_memory())
assert final_bytes <= stable_bytes + 16 * 1024**2
assert final_bytes < 512 * 1024**2
assert driver_bytes < 1024 * 1024**2

print(json.dumps({
    "schemaVersion": 1,
    "componentSha": build_info.COMPONENT_SHA,
    "agentVersion": build_info.AGENT_VERSION,
    "modelRevision": revision,
    "labels": labels,
    "cpuScores": [float(score) for score in reference_scores],
    "mpsScores": [float(score) for score in observed_scores],
    "offline": True,
    "mpsAvailable": True,
    "device": "mps",
    "dtype": "float16",
    "visionModelClass": detector._model.__class__.__name__,
    "visionParameterDevices": vision_parameter_devices,
    "visionParameterDtypes": vision_parameter_dtypes,
    "persistentStatePacked": True,
    "firstInference": True,
    "repeatCount": 16,
    "physicalMemoryBytes": physical_memory,
    "recommendedMaxMemoryBytes": recommended_memory,
    "stableAllocatedBytes": stable_bytes,
    "finalAllocatedBytes": final_bytes,
    "driverAllocatedBytes": driver_bytes,
}, sort_keys=True, separators=(",", ":"), allow_nan=False))
'''


def _worker_environment(
    temp_root: Path,
    *,
    hf_home: Path,
    formula_prefix: Path,
    component_sha: str,
    agent_version: str,
    offline: bool,
) -> dict[str, str]:
    environment = _safe_environment(temp_root, hf_home=hf_home, offline=offline)
    environment.update(
        {
            "FORMULA_PREFIX": str(formula_prefix),
            "CANDIDATE_SHA": component_sha,
            "CANDIDATE_VERSION": agent_version,
            "MODEL_REPO": MODEL_REPO,
            "MODEL_REVISION": MODEL_REVISION,
            "EXPECTED_RUNTIME_VERSIONS": canonical_json(EXPECTED_RUNTIME_VERSIONS),
            "PHYSICAL_MEMORY_FLOOR_BYTES": str(PHYSICAL_MEMORY_FLOOR_BYTES),
            "RECOMMENDED_MEMORY_FLOOR_BYTES": str(RECOMMENDED_MEMORY_FLOOR_BYTES),
        }
    )
    return environment


def run_qualification(*, formula: str, timeout_seconds: int) -> dict[str, Any]:
    _require(MIN_TIMEOUT_SECONDS <= timeout_seconds <= MAX_TIMEOUT_SECONDS, "argument-invalid", f"--timeout-seconds must be between {MIN_TIMEOUT_SECONDS} and {MAX_TIMEOUT_SECONDS}")
    _require(platform.system() == "Darwin", "platform-unsupported", "qualification requires macOS; --help is available on every platform")
    _require(platform.machine() == "arm64", "platform-unsupported", "qualification requires native Apple Silicon arm64")

    repo_root = Path(__file__).resolve(strict=True).parents[2]
    with tempfile.TemporaryDirectory(prefix="nuv-macos-mps-qualification-") as raw_temp:
        temp_root = Path(raw_temp).resolve(strict=True)
        temp_root.chmod(0o700)
        component_sha, agent_version = _candidate_identity(repo_root, temp_root=temp_root, timeout_seconds=timeout_seconds)
        formula_prefix, formula_python, receipt = _resolve_formula(
            formula,
            agent_version=agent_version,
            temp_root=temp_root,
            timeout_seconds=timeout_seconds,
        )
        _require(receipt["sourceBuilt"] is True, "formula-receipt-invalid", "Formula source-build proof is absent")

        hf_home = temp_root / "huggingface"
        hf_home.mkdir(mode=0o700)
        work_dir = temp_root / "work"
        work_dir.mkdir(mode=0o700)
        environment = _worker_environment(
            temp_root,
            hf_home=hf_home,
            formula_prefix=formula_prefix,
            component_sha=component_sha,
            agent_version=agent_version,
            offline=False,
        )

        _run_command(
            [str(formula_python), "-I", "-m", "pip", "check"],
            cwd=work_dir,
            environment=environment,
            timeout_seconds=min(timeout_seconds, 300),
            code="formula-dependencies-invalid",
        )
        identity_raw = _run_json_worker(
            formula_python,
            FORMULA_IDENTITY_WORKER,
            cwd=work_dir,
            environment=environment,
            timeout_seconds=min(timeout_seconds, 300),
            code="formula-identity-invalid",
            name="Formula identity worker output",
        )
        validate_formula_identity(
            identity_raw,
            component_sha=component_sha,
            agent_version=agent_version,
            formula_prefix=formula_prefix,
        )

        snapshot_raw = _run_json_worker(
            formula_python,
            SNAPSHOT_WORKER,
            cwd=work_dir,
            environment=environment,
            timeout_seconds=timeout_seconds,
            code="model-download-failed",
            name="model snapshot worker output",
        )
        snapshot = _exact_mapping(snapshot_raw, {"schemaVersion", "repoId", "revision", "path"}, code="model-snapshot-invalid", name="model snapshot")
        _require(_is_int(snapshot["schemaVersion"]) and snapshot["schemaVersion"] == SCHEMA_VERSION, "model-snapshot-invalid", "model snapshot schema version is invalid")
        _require(snapshot["repoId"] == MODEL_REPO and snapshot["revision"] == MODEL_REVISION, "model-snapshot-invalid", "resolved model identity differs")
        try:
            model_path = Path(str(snapshot["path"])).resolve(strict=True)
        except OSError as exc:
            raise QualificationError("model-snapshot-invalid", "resolved model path is unavailable") from exc
        _require(model_path.is_relative_to(hf_home) and model_path.name == MODEL_REVISION and model_path.is_dir(), "model-snapshot-invalid", "resolved model path escaped the private cache or differs in revision")

        offline_environment = _worker_environment(
            temp_root,
            hf_home=hf_home,
            formula_prefix=formula_prefix,
            component_sha=component_sha,
            agent_version=agent_version,
            offline=True,
        )
        offline_environment["MODEL_PATH"] = str(model_path)
        reference_raw = _run_json_worker(
            formula_python,
            CPU_REFERENCE_WORKER,
            cwd=work_dir,
            environment=offline_environment,
            timeout_seconds=timeout_seconds,
            code="cpu-reference-failed",
            name="CPU reference worker output",
        )
        reference = validate_reference(reference_raw, component_sha=component_sha, agent_version=agent_version)
        reference_path = work_dir / "siglip-reference-v1.json"
        _write_private_json(reference_path, reference)
        written_reference = _read_regular_json(reference_path, code="cpu-reference-invalid", maximum_bytes=MAX_JSON_BYTES)
        validate_reference(written_reference, component_sha=component_sha, agent_version=agent_version)
        _require(stat.S_IMODE(reference_path.lstat().st_mode) == 0o600, "cpu-reference-invalid", "CPU reference permissions are not 0600")

        offline_environment["REFERENCE_PATH"] = str(reference_path)
        offline_environment["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        sync = shutil.which("sync", path=offline_environment["PATH"])
        if sync is not None:
            _run_command([sync], cwd=work_dir, environment=offline_environment, timeout_seconds=60, code="memory-preflight-failed")
        proof_raw = _run_json_worker(
            formula_python,
            MPS_PROOF_WORKER,
            cwd=work_dir,
            environment=offline_environment,
            timeout_seconds=timeout_seconds,
            code="mps-proof-failed",
            name="MPS proof worker output",
        )
        proof = validate_mps_proof(proof_raw, component_sha=component_sha, agent_version=agent_version)
        return build_success_result(
            component_sha=component_sha,
            agent_version=agent_version,
            formula=formula,
            formula_prefix=formula_prefix,
            proof=proof,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Qualify an already installed, source-built Homebrew candidate on a "
            "local Apple Silicon Mac. The Formula's stamped SHA/version must "
            "exactly match a clean checkout; this command never installs or "
            "replaces a Formula and never uses GitHub runners."
        )
    )
    parser.add_argument(
        "--formula",
        default=DEFAULT_FORMULA,
        help=(
            "fully qualified installed candidate Formula "
            f"(default: {DEFAULT_FORMULA})"
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=(
            "per-model-stage timeout; must be between "
            f"{MIN_TIMEOUT_SECONDS} and {MAX_TIMEOUT_SECONDS} "
            f"(default: {DEFAULT_TIMEOUT_SECONDS})"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_qualification(
            formula=args.formula,
            timeout_seconds=args.timeout_seconds,
        )
    except QualificationError as exc:
        result = build_failure_result(exc)
        print(canonical_json(result))
        print(f"{exc.code}: {result['error']['message']}", file=sys.stderr)
        return 2
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:  # noqa: BLE001 - preserve one machine-readable result.
        failure = QualificationError(
            "internal-error",
            f"unexpected {exc.__class__.__name__}: {exc}",
        )
        result = build_failure_result(failure)
        print(canonical_json(result))
        print(f"{failure.code}: {result['error']['message']}", file=sys.stderr)
        return 1
    print(canonical_json(validate_result(result)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
