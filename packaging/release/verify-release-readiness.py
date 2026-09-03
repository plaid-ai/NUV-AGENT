#!/usr/bin/env python3
"""Fail closed when a release has unresolved audited dependency blockers."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import subprocess
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
from nuvion_app.runtime.stable_file import StableFileError, read_stable_regular_file

SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
BLOCKER_ID = re.compile(r"^[A-Z0-9][A-Z0-9._-]{0,127}$")
SHA = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
FINGERPRINT = re.compile(r"^[0-9A-F]{40}$")
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")
RUN_ID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
RELEASE_SLOT = re.compile(r"^releases/[0-9a-f]{64}$")
IDENTITY_TEXT = re.compile(r"^[\x20-\x7e]{1,255}$")
MAX_EVIDENCE_BYTES = 1024 * 1024
CANDIDATE_SOAK_REQUIRED_VERSIONS = frozenset({"0.1.121"})
MAX_APPSRC_BYTES = 640 * 480 * 3 * 2


class ReadinessError(RuntimeError):
    pass


def _load(path: Path) -> dict[str, Any]:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ReadinessError(f"duplicate release readiness member: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            read_stable_regular_file(
                path, maximum=MAX_EVIDENCE_BYTES
            ).decode("utf-8"),
            object_pairs_hook=reject_duplicate,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ReadinessError(f"invalid release readiness constant: {value}")
            ),
        )
    except (StableFileError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReadinessError("release readiness document is invalid") from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schemaVersion", "releases"}
        or type(payload.get("schemaVersion")) is not int
        or payload.get("schemaVersion") != 2
        or not isinstance(payload.get("releases"), dict)
    ):
        raise ReadinessError("release readiness document does not match schema v2")
    return payload


def _regular_bytes(path: Path) -> bytes:
    try:
        return read_stable_regular_file(path, maximum=MAX_EVIDENCE_BYTES)
    except StableFileError as exc:
        raise ReadinessError(f"release evidence is unavailable: {path.name}") from exc


def _primary_fingerprints(gpg_home: Path) -> set[str]:
    result = subprocess.run(
        ["gpg", "--batch", "--homedir", str(gpg_home), "--with-colons", "--list-keys"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise ReadinessError("cannot inspect physical-evidence signer keyring")
    fingerprints: set[str] = set()
    expect_primary = False
    for line in result.stdout.splitlines():
        fields = line.split(":")
        record = fields[0] if fields else ""
        if record == "pub":
            expect_primary = True
        elif record == "sub":
            expect_primary = False
        elif record == "fpr" and expect_primary and len(fields) > 9:
            value = fields[9].upper()
            if FINGERPRINT.fullmatch(value):
                fingerprints.add(value)
            expect_primary = False
    return fingerprints


def _verify_detached_signature(
    evidence_raw: bytes,
    signature_raw: bytes,
    *,
    signer_directory: Path,
    allowed_fingerprints: set[str],
) -> str:
    public_keys = sorted(signer_directory.glob("*.asc"))
    if not public_keys:
        raise ReadinessError("physical-evidence signer directory is empty")
    public_key_payloads = [_regular_bytes(path) for path in public_keys]
    with tempfile.TemporaryDirectory(prefix="nuv-physical-evidence-verify-") as raw_home:
        gpg_home = Path(raw_home)
        gpg_home.chmod(0o700)
        evidence_path = gpg_home / "physical-evidence.json"
        signature_path = gpg_home / "physical-evidence.json.asc"
        evidence_path.write_bytes(evidence_raw)
        signature_path.write_bytes(signature_raw)
        private_public_keys: list[Path] = []
        for index, payload in enumerate(public_key_payloads):
            target = gpg_home / f"trusted-signer-{index}.asc"
            target.write_bytes(payload)
            private_public_keys.append(target)
        imported = subprocess.run(
            [
                "gpg",
                "--batch",
                "--homedir",
                str(gpg_home),
                "--import",
                *map(str, private_public_keys),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if imported.returncode != 0:
            raise ReadinessError("cannot import physical-evidence signer keys")
        if _primary_fingerprints(gpg_home) != allowed_fingerprints:
            raise ReadinessError("physical-evidence signer files differ from policy")
        verified = subprocess.run(
            [
                "gpg",
                "--batch",
                "--status-fd=1",
                "--homedir",
                str(gpg_home),
                "--verify",
                str(signature_path),
                str(evidence_path),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    if verified.returncode != 0:
        raise ReadinessError("physical release evidence signature is invalid")
    observed = {
        token.upper()
        for line in verified.stdout.splitlines()
        if "[GNUPG:] VALIDSIG " in line
        for token in line.split("VALIDSIG", 1)[1].split()
        if FINGERPRINT.fullmatch(token.upper())
    }
    accepted = observed & allowed_fingerprints
    if len(accepted) != 1:
        raise ReadinessError("physical evidence signer is not allowlisted")
    return next(iter(accepted))


def _strict_object(raw: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ReadinessError(f"duplicate {label} member: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ReadinessError(f"invalid {label} constant: {value}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ReadinessError(f"{label} is invalid") from exc
    if not isinstance(payload, dict):
        raise ReadinessError(f"{label} must be an object")
    return payload


def _strict_canonical_object(raw: bytes, *, label: str) -> dict[str, Any]:
    payload = _strict_object(raw, label=label)
    try:
        canonical = (
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
        raise ReadinessError(f"{label} is not canonical JSON data") from exc
    if raw != canonical:
        raise ReadinessError(
            f"{label} is not canonical sort_keys compact JSON with one newline"
        )
    return payload


def _module_from_verified_source(
    *, module_name: str, source: bytes, display_path: Path
) -> types.ModuleType:
    """Execute the exact bytes already checked by the release trust boundary."""

    module = types.ModuleType(module_name)
    module.__file__ = str(display_path)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        code = compile(source, str(display_path), "exec", dont_inherit=True)
        exec(code, module.__dict__)  # noqa: S102 - exact trusted bytes only.
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _number(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ReadinessError(f"{label} must be a finite number")
    normalized = float(value)
    if minimum is not None and normalized < minimum:
        raise ReadinessError(f"{label} is below its safety bound")
    if maximum is not None and normalized > maximum:
        raise ReadinessError(f"{label} exceeds its safety bound")
    return normalized


def _integer(
    value: object,
    *,
    label: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ReadinessError(f"{label} must be a bounded integer")
    if maximum is not None and value > maximum:
        raise ReadinessError(f"{label} must be a bounded integer")
    return value


def _artifact_identity(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"name", "sha256", "sizeBytes"}:
        raise ReadinessError(f"{label} identity is invalid")
    name = value.get("name")
    digest = value.get("sha256")
    size = value.get("sizeBytes")
    if (
        not isinstance(name, str)
        or not SAFE_NAME.fullmatch(name)
        or not isinstance(digest, str)
        or not SHA256.fullmatch(digest)
    ):
        raise ReadinessError(f"{label} identity is invalid")
    _integer(size, label=f"{label} size", minimum=1)
    return value


def _timestamp(value: object, *, label: str) -> dt.datetime:
    if not isinstance(value, str) or re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
        r"(?:\.[0-9]{3})?Z",
        value,
    ) is None:
        raise ReadinessError(f"{label} is not canonical UTC")
    try:
        parsed = dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ReadinessError(f"{label} is not canonical UTC") from exc
    if parsed.tzinfo != dt.timezone.utc:
        raise ReadinessError(f"{label} is not canonical UTC")
    return parsed


def _evidence_reference(
    base: Path,
    value: object,
    *,
    label: str,
) -> tuple[bytes, str]:
    if not isinstance(value, dict) or set(value) != {"file", "sha256"}:
        raise ReadinessError(f"{label} reference is invalid")
    name = value.get("file")
    digest = value.get("sha256")
    if (
        not isinstance(name, str)
        or not SAFE_NAME.fullmatch(name)
        or not isinstance(digest, str)
        or not SHA256.fullmatch(digest)
    ):
        raise ReadinessError(f"{label} reference is invalid")
    raw = _regular_bytes(base / name)
    observed = hashlib.sha256(raw).hexdigest()
    if observed != digest:
        raise ReadinessError(f"{label} SHA-256 does not match signed summary")
    return raw, observed


def _fleet_runtime_gate(
    fleet_evidence: dict[str, Any],
    cleanup_evidence: dict[str, Any],
    fleet_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Return the exact security-relevant Fleet outcome bound by the summary."""

    updater = fleet_evidence.get("updater")
    update = updater.get("update") if isinstance(updater, dict) else None
    services = fleet_evidence.get("services")
    slots = fleet_evidence.get("slots")
    anti_replay = fleet_evidence.get("antiReplay")
    trust_inputs = fleet_manifest.get("inputs")
    if (
        not isinstance(updater, dict)
        or not isinstance(update, dict)
        or not isinstance(services, dict)
        or not isinstance(slots, dict)
        or not isinstance(anti_replay, dict)
        or not isinstance(trust_inputs, dict)
    ):
        raise ReadinessError("IQ9075 Fleet Runtime result is incomplete")
    return {
        "runId": fleet_manifest.get("runId"),
        "scenario": fleet_evidence.get("scenario"),
        # The detached summary signature transitively signs these immutable
        # command/release/health/device-binding keyring digests.
        "trustInputs": trust_inputs,
        "terminalPhase": update.get("phase"),
        "stagedActivation": {
            "candidateSlot": update.get("candidateSlot"),
            "previousSlot": update.get("previousSlot"),
            "previousVersion": update.get("previousVersion"),
        },
        "signedRelease": {
            "bomDigest": update.get("bomDigest"),
            "artifactDigest": update.get("artifactDigest"),
            "componentSha": update.get("componentSha"),
            "publisherKeyId": update.get("publisherKeyId"),
            "verificationStatus": update.get("bomVerificationStatus"),
        },
        "healthDecision": {
            "updaterPeerAuthenticated": updater.get("authenticatedHelper"),
            "health": update.get("health"),
            "functionalHealth": update.get("functionalHealth"),
        },
        "antiReplay": anti_replay,
        "runtimePids": fleet_evidence.get("runtimePids"),
        "services": services,
        "slots": slots,
        "cleanup": {
            "complete": cleanup_evidence.get("complete"),
            "recovered": cleanup_evidence.get("recovered"),
            "phase": cleanup_evidence.get("phase"),
            "completedAt": cleanup_evidence.get("completedAt"),
            "manifestSha256": cleanup_evidence.get("manifestSha256"),
            "fleetEvidenceSha256": cleanup_evidence.get(
                "fleetEvidenceSha256"
            ),
            "identity": cleanup_evidence.get("identity"),
            "proof": cleanup_evidence.get("proof"),
        },
    }


def _validate_fleet_runtime_documents(
    *,
    policy_path: Path,
    version: str,
    component_sha: str,
    summary: dict[str, Any],
    security: dict[str, Any],
    candidate_fleet_runner: Path,
    candidate_board_tool: Path,
) -> dict[str, str]:
    """Validate camera-independent OTA/rollback evidence for IQ9075."""

    expected_summary_fields = {
        "schemaVersion",
        "kind",
        "agentVersion",
        "componentSha",
        "fleetRunnerSha256",
        "boardToolSha256",
        "fleetManifest",
        "fleetEvidence",
        "cleanupEvidence",
        "testedArtifact",
        "testedBom",
        "runtimeGate",
    }
    if set(summary) != expected_summary_fields or (
        type(summary.get("schemaVersion")) is not int
        or summary.get("schemaVersion") != 2
        or summary.get("kind")
        != "nuvion-iq9075-fleet-runtime-release-evidence"
        or summary.get("agentVersion") != version
        or summary.get("componentSha") != component_sha
        or not isinstance(summary.get("fleetRunnerSha256"), str)
        or not SHA256.fullmatch(summary["fleetRunnerSha256"])
        or not isinstance(summary.get("boardToolSha256"), str)
        or not SHA256.fullmatch(summary["boardToolSha256"])
    ):
        raise ReadinessError(
            "IQ9075 Fleet Runtime evidence does not match schema v2"
        )

    manifest_raw, _manifest_sha256 = _evidence_reference(
        policy_path.parent,
        summary.get("fleetManifest"),
        label="IQ9075 Fleet Runtime manifest",
    )
    evidence_raw, _evidence_sha256 = _evidence_reference(
        policy_path.parent,
        summary.get("fleetEvidence"),
        label="IQ9075 Fleet Runtime result",
    )
    cleanup_raw, _cleanup_sha256 = _evidence_reference(
        policy_path.parent,
        summary.get("cleanupEvidence"),
        label="IQ9075 Fleet Runtime cleanup evidence",
    )
    bom_raw, tested_bom_sha256 = _evidence_reference(
        policy_path.parent,
        summary.get("testedBom"),
        label="IQ9075 Fleet Runtime tested BOM",
    )
    manifest = _strict_canonical_object(
        manifest_raw, label="IQ9075 Fleet Runtime manifest"
    )
    fleet_evidence = _strict_canonical_object(
        evidence_raw, label="IQ9075 Fleet Runtime result"
    )
    cleanup_evidence = _strict_canonical_object(
        cleanup_raw, label="IQ9075 Fleet Runtime cleanup evidence"
    )
    bom = _strict_canonical_object(
        bom_raw, label="IQ9075 Fleet Runtime tested BOM"
    )

    publisher_fleet_runner = (
        Path(__file__).resolve().parents[1] / "dev/run-iq9075-fleet-e2e.py"
    )
    publisher_board_tool = (
        Path(__file__).resolve().parents[1] / "dev/iq9075-board-e2e.py"
    )
    publisher_fleet_runner_raw = _regular_bytes(publisher_fleet_runner)
    candidate_fleet_runner_raw = _regular_bytes(candidate_fleet_runner)
    publisher_board_tool_raw = _regular_bytes(publisher_board_tool)
    candidate_board_tool_raw = _regular_bytes(candidate_board_tool)
    fleet_runner_sha256 = hashlib.sha256(publisher_fleet_runner_raw).hexdigest()
    board_tool_sha256 = hashlib.sha256(publisher_board_tool_raw).hexdigest()
    if (
        candidate_fleet_runner_raw != publisher_fleet_runner_raw
        or summary.get("fleetRunnerSha256") != fleet_runner_sha256
    ):
        raise ReadinessError(
            "IQ9075 Fleet Runtime runner differs from signed evidence"
        )
    if (
        candidate_board_tool_raw != publisher_board_tool_raw
        or summary.get("boardToolSha256") != board_tool_sha256
        or manifest.get("toolSha256") != board_tool_sha256
    ):
        raise ReadinessError(
            "IQ9075 board tool differs from Fleet Runtime manifest"
        )

    module_name = "_nuvion_trusted_iq9075_fleet_runtime_validator"
    fleet_validator = _module_from_verified_source(
        module_name=module_name,
        source=publisher_fleet_runner_raw,
        display_path=publisher_fleet_runner,
    )
    try:
        validated_manifest = fleet_validator.validate_manifest(manifest)
        fleet_validator.validate_final_evidence(fleet_evidence, validated_manifest)
        canonical_cleanup = fleet_validator.validate_bound_cleanup_evidence(
            cleanup_evidence,
            run_id=str(validated_manifest["runId"]),
            manifest_raw=manifest_raw,
            fleet_evidence_raw=evidence_raw,
        )
    except Exception as exc:
        raise ReadinessError("IQ9075 Fleet Runtime evidence is invalid") from exc
    finally:
        sys.modules.pop(module_name, None)
    if (
        fleet_evidence.get("schemaVersion") != 2
        or not isinstance(fleet_evidence.get("antiReplay"), dict)
    ):
        raise ReadinessError(
            "IQ9075 Fleet Runtime evidence lacks persisted anti-replay proof"
        )
    if (
        cleanup_evidence != canonical_cleanup
        or cleanup_evidence.get("schemaVersion") != 2
        or cleanup_evidence.get("phase") != "RESTORED"
        or cleanup_evidence.get("proof", {}).get("transactionPhase")
        != "RESTORED"
    ):
        raise ReadinessError(
            "IQ9075 Fleet Runtime cleanup did not restore the transaction"
        )

    scenario = manifest.get("scenario")
    updater = fleet_evidence.get("updater")
    update = updater.get("update") if isinstance(updater, dict) else None
    if (
        not isinstance(scenario, dict)
        or scenario.get("type") != "oak-fault-rollback"
        or fleet_evidence.get("scenario") != "oak-fault-rollback"
        or not isinstance(update, dict)
        or update.get("phase") != "ROLLED_BACK"
        or update.get("updatePhase") != "ROLLED_BACK"
        or update.get("errorCode") != "ROLLED_BACK"
        or update.get("health") != "LKG_RESTORED"
        or update.get("functionalHealth") != "FUNCTIONAL_UNHEALTHY"
        or "healthDeadline" in update
    ):
        raise ReadinessError(
            "IQ9075 Fleet Runtime evidence does not prove terminal rollback"
        )

    try:
        verified_bom = verify_release_bom(bom)
    except ReleaseBomValidationError as exc:
        raise ReadinessError("IQ9075 Fleet Runtime tested BOM is invalid") from exc
    tested_artifact = _artifact_identity(
        summary.get("testedArtifact"),
        label="IQ9075 Fleet Runtime tested artifact",
    )
    if tested_artifact["name"] != (
        f"nuv-agent_{version}_iq9075-aarch64.agent-bundle.tar.gz"
    ) or summary["testedBom"]["file"] != (
        f"nuv-agent_{version}_iq9075-aarch64.release-bom.json"
    ):
        raise ReadinessError("IQ9075 Fleet Runtime artifact names are invalid")

    iq_policy = security.get("iq9075")
    target_policy = iq_policy.get("target") if isinstance(iq_policy, dict) else None
    baseline = (
        iq_policy.get("legacyPromotedBaseline")
        if isinstance(iq_policy, dict)
        else None
    )
    if (
        not isinstance(iq_policy, dict)
        or not isinstance(target_policy, dict)
        or set(target_policy)
        != {"productModel", "platformProfile", "hardwareRevision", "architecture"}
        or not isinstance(baseline, dict)
        or set(baseline) != {"agentVersion", "releaseSequence", "bomDigest"}
        or not isinstance(baseline.get("agentVersion"), str)
        or not SEMVER.fullmatch(baseline["agentVersion"])
        or isinstance(baseline.get("releaseSequence"), bool)
        or not isinstance(baseline.get("releaseSequence"), int)
        or baseline["releaseSequence"] < 1
        or not isinstance(baseline.get("bomDigest"), str)
        or not baseline["bomDigest"].startswith("sha256:")
        or not SHA256.fullmatch(baseline["bomDigest"][7:])
        or not isinstance(iq_policy.get("publicKeyringSha256"), str)
        or not SHA256.fullmatch(iq_policy["publicKeyringSha256"])
        or not isinstance(iq_policy.get("publisherKeyId"), str)
    ):
        raise ReadinessError("IQ9075 Fleet Runtime security policy is invalid")

    release = scenario.get("release") if isinstance(scenario, dict) else None
    inputs = manifest.get("inputs")
    identity = manifest.get("identity")
    slots = fleet_evidence.get("slots")
    expected_bom_digest = "sha256:" + verified_bom.bom_digest
    expected_candidate_slot = "/opt/nuv-agent/releases/" + verified_bom.bom_digest
    expected_previous_slot = "releases/" + baseline["bomDigest"][7:]
    baseline_marker = None
    if isinstance(slots, dict) and isinstance(scenario, dict):
        baseline_marker = (
            slots.get("release")
            if scenario.get("type") == "oak-fault-rollback"
            else slots.get("previousRelease")
        )
    if (
        verified_bom.schema_version != 2
        or verified_bom.agent_version != version
        or verified_bom.component_sha != component_sha
        or verified_bom.release_sequence is None
        or verified_bom.release_sequence <= baseline["releaseSequence"]
        or verified_bom.min_updater_version != "0.2.0"
        or verified_bom.artifact_kind != "agent-bundle"
        or verified_bom.artifact_name != tested_artifact["name"]
        or verified_bom.artifact_sha256 != tested_artifact["sha256"]
        or verified_bom.artifact_size_bytes != tested_artifact["sizeBytes"]
        or len(verified_bom.targets) != 1
        or verified_bom.targets[0].to_payload() != target_policy
        or not isinstance(inputs, dict)
        or inputs.get("releaseSha256") != iq_policy["publicKeyringSha256"]
        or not isinstance(identity, dict)
        or {
            key: identity.get(key)
            for key in (
                "productModel",
                "platformProfile",
                "hardwareRevision",
                "architecture",
            )
        }
        != target_policy
        or not isinstance(baseline_marker, dict)
        or baseline_marker.get("agentVersion") != baseline["agentVersion"]
        or baseline_marker.get("releaseSequence") != baseline["releaseSequence"]
        or baseline_marker.get("bomDigest") != baseline["bomDigest"]
        or not isinstance(scenario, dict)
        or scenario.get("expectedBomDigest") != expected_bom_digest
        or scenario.get("expectedCandidateSlot") != expected_candidate_slot
        or scenario.get("expectedPreviousSlot") != expected_previous_slot
        or scenario.get("expectedPreviousVersion") != baseline["agentVersion"]
        or not isinstance(release, dict)
        or release
        != {
            "agentVersion": version,
            "releaseSequence": verified_bom.release_sequence,
            "artifactDigest": "sha256:" + verified_bom.artifact_sha256,
            "componentSha": component_sha,
            "configSchema": verified_bom.config_schema,
            "publisherKeyId": iq_policy["publisherKeyId"],
        }
    ):
        raise ReadinessError(
            "IQ9075 Fleet Runtime artifact, BOM, manifest, and policy differ"
        )

    expected_runtime_gate = _fleet_runtime_gate(
        fleet_evidence, cleanup_evidence, manifest
    )
    if summary.get("runtimeGate") != expected_runtime_gate:
        raise ReadinessError(
            "IQ9075 Fleet Runtime gate summary differs from raw evidence"
        )
    return {
        # Preserve the established workflow output contract while exposing names
        # that match the new evidence authority to future callers.
        "physical_artifact_sha256": verified_bom.artifact_sha256,
        "physical_bom_sha256": tested_bom_sha256,
        "runtime_artifact_sha256": verified_bom.artifact_sha256,
        "runtime_bom_sha256": tested_bom_sha256,
    }


def _validate_physical_documents(
    *,
    policy_path: Path,
    version: str,
    component_sha: str,
    summary: dict[str, Any],
    security: dict[str, Any],
    candidate_harness: Path,
) -> dict[str, str]:
    expected_summary_fields = {
        "schemaVersion",
        "kind",
        "agentVersion",
        "componentSha",
        "harnessManifest",
        "harnessResult",
        "physicalGate",
    }
    cleanup_summary_reference = summary.get("cleanupEvidence")
    if cleanup_summary_reference is not None:
        expected_summary_fields.add("cleanupEvidence")
    if set(summary) != expected_summary_fields or (
        type(summary.get("schemaVersion")) is not int
        or summary.get("schemaVersion") != 2
        or summary.get("kind") != "nuvion-iq9075-physical-release-evidence"
        or summary.get("agentVersion") != version
        or summary.get("componentSha") != component_sha
    ):
        raise ReadinessError("IQ9075 physical evidence does not match schema v2")

    manifest_raw, manifest_sha256 = _evidence_reference(
        policy_path.parent,
        summary.get("harnessManifest"),
        label="IQ9075 harness manifest",
    )
    result_raw, _result_sha256 = _evidence_reference(
        policy_path.parent,
        summary.get("harnessResult"),
        label="IQ9075 harness result",
    )
    manifest = _strict_object(manifest_raw, label="IQ9075 harness manifest")
    result = _strict_object(result_raw, label="IQ9075 harness result")

    expected_manifest_fields = {
        "schemaVersion",
        "kind",
        "runId",
        "agentVersion",
        "componentSha",
        "harnessSha256",
        "fleetRunnerSha256",
        "oakSoak",
        "fleetManifest",
        "fleetEvidence",
        "testedArtifact",
        "testedBom",
        "board",
        "oakMxidSha256",
        "startedAt",
        "expectedRollback",
    }
    candidate_soak_reference = manifest.get("candidateSoak")
    cleanup_evidence_reference = manifest.get("cleanupEvidence")
    if candidate_soak_reference is not None:
        expected_manifest_fields.update({"candidateSoak", "cleanupEvidence"})
    if set(manifest) != expected_manifest_fields:
        raise ReadinessError("IQ9075 harness manifest fields are invalid")
    run_id = manifest.get("runId")
    board = manifest.get("board")
    rollback_manifest = manifest.get("expectedRollback")
    iq_policy = security.get("iq9075")
    baseline = (
        iq_policy.get("legacyPromotedBaseline")
        if isinstance(iq_policy, dict)
        else None
    )
    target_policy = iq_policy.get("target") if isinstance(iq_policy, dict) else None
    if (
        type(manifest.get("schemaVersion")) is not int
        or manifest.get("schemaVersion") != 1
        or manifest.get("kind") != "nuvion-iq9075-physical-manifest"
        or not isinstance(run_id, str)
        or not RUN_ID.fullmatch(run_id)
        or manifest.get("agentVersion") != version
        or manifest.get("componentSha") != component_sha
        or not isinstance(manifest.get("harnessSha256"), str)
        or not SHA256.fullmatch(manifest["harnessSha256"])
        or not isinstance(manifest.get("fleetRunnerSha256"), str)
        or not SHA256.fullmatch(manifest["fleetRunnerSha256"])
        or not isinstance(manifest.get("oakMxidSha256"), str)
        or not SHA256.fullmatch(manifest["oakMxidSha256"])
        or not isinstance(board, dict)
        or set(board)
        != {
            "productModel",
            "platformProfile",
            "hardwareRevision",
            "architecture",
            "kernel",
            "depthaiVersion",
            "gstreamerVersion",
        }
        or not isinstance(target_policy, dict)
        or {
            key: board.get(key)
            for key in (
                "productModel",
                "platformProfile",
                "hardwareRevision",
                "architecture",
            )
        }
        != target_policy
        or board.get("depthaiVersion") != "2.32.0.0"
        or any(
            not isinstance(board.get(key), str)
            or not IDENTITY_TEXT.fullmatch(board[key])
            for key in ("kernel", "gstreamerVersion")
        )
        or not isinstance(rollback_manifest, dict)
        or set(rollback_manifest) != {"agentVersion", "slot"}
        or not isinstance(rollback_manifest.get("agentVersion"), str)
        or not SEMVER.fullmatch(rollback_manifest["agentVersion"])
        or not isinstance(baseline, dict)
        or set(baseline) != {"agentVersion", "releaseSequence", "bomDigest"}
        or baseline.get("agentVersion") != rollback_manifest["agentVersion"]
        or isinstance(baseline.get("releaseSequence"), bool)
        or not isinstance(baseline.get("releaseSequence"), int)
        or baseline["releaseSequence"] < 1
        or not isinstance(baseline.get("bomDigest"), str)
        or not baseline["bomDigest"].startswith("sha256:")
        or not SHA256.fullmatch(baseline["bomDigest"][7:])
        or not isinstance(rollback_manifest.get("slot"), str)
        or not RELEASE_SLOT.fullmatch(rollback_manifest["slot"])
        or rollback_manifest["slot"] != "releases/" + baseline["bomDigest"][7:]
    ):
        raise ReadinessError("IQ9075 harness manifest identity is invalid")
    soak_started_at = _timestamp(
        manifest.get("startedAt"), label="IQ9075 harness start time"
    )
    oak_soak_raw, _oak_soak_sha = _evidence_reference(
        policy_path.parent,
        manifest.get("oakSoak"),
        label="IQ9075 OAK soak result",
    )
    oak_soak = _strict_object(oak_soak_raw, label="IQ9075 OAK soak result")
    expected_oak_fields = {
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
    oak_schema = oak_soak.get("schemaVersion")
    if oak_schema == 3:
        expected_oak_fields.update({"runId", "slotKind"})
    if set(oak_soak) != expected_oak_fields:
        raise ReadinessError("IQ9075 OAK soak source fields are invalid")
    runtime_identity = oak_soak.get("runtimeIdentity")
    device_identity = oak_soak.get("deviceIdentity")
    outcome = oak_soak.get("outcome")
    if (
        type(oak_soak.get("schemaVersion")) is not int
        or oak_schema not in {2, 3}
        or (
            version in CANDIDATE_SOAK_REQUIRED_VERSIONS
            and (oak_schema != 3 or candidate_soak_reference is None)
        )
        or oak_soak.get("kind") != "nuvion-iq9075-oak-soak-result"
        or not isinstance(outcome, dict)
        or set(outcome) != {"status", "error", "cleanupErrors"}
        or outcome.get("status") != "passed"
        or outcome.get("error") is not None
        or outcome.get("cleanupErrors") != []
        or oak_soak.get("startedAt") != manifest.get("startedAt")
        or oak_soak.get("board") != board
        or oak_soak.get("oakMxidSha256") != manifest.get("oakMxidSha256")
        or not isinstance(device_identity, dict)
        or set(device_identity) != {"deviceId", "spaceId"}
        or not isinstance(device_identity.get("deviceId"), str)
        or isinstance(device_identity.get("spaceId"), bool)
        or not isinstance(device_identity.get("spaceId"), int)
        or not isinstance(runtime_identity, dict)
        or set(runtime_identity)
        != ({
            "agentVersion",
            "componentSha",
            "bomDigest",
            "pythonPath",
            "sitePackagesPath",
            "buildInfoPath",
            "releaseMarkerSha256",
        } | ({"candidateSlot", "controlMarkerSha256"} if oak_schema == 3 else set()))
        or runtime_identity.get("agentVersion") != version
        or runtime_identity.get("componentSha") != component_sha
        or not isinstance(runtime_identity.get("bomDigest"), str)
        or not isinstance(runtime_identity.get("pythonPath"), str)
        or not isinstance(runtime_identity.get("releaseMarkerSha256"), str)
        or not SHA256.fullmatch(runtime_identity["releaseMarkerSha256"])
        or (
            oak_schema == 3
            and (
                oak_soak.get("runId") != run_id
                or oak_soak.get("slotKind") != "candidate"
                or candidate_soak_reference is None
                or not isinstance(runtime_identity.get("controlMarkerSha256"), str)
                or not SHA256.fullmatch(runtime_identity["controlMarkerSha256"])
            )
        )
        or (oak_schema == 2 and candidate_soak_reference is not None)
    ):
        raise ReadinessError("IQ9075 OAK soak source identity is invalid")
    tested_artifact = _artifact_identity(
        manifest.get("testedArtifact"), label="IQ9075 tested artifact"
    )
    if tested_artifact["name"] != (
        f"nuv-agent_{version}_iq9075-aarch64.agent-bundle.tar.gz"
    ):
        raise ReadinessError("IQ9075 tested artifact name is invalid")
    tested_bom_raw, tested_bom_sha256 = _evidence_reference(
        policy_path.parent,
        manifest.get("testedBom"),
        label="IQ9075 tested BOM",
    )
    if manifest["testedBom"]["file"] != (
        f"nuv-agent_{version}_iq9075-aarch64.release-bom.json"
    ):
        raise ReadinessError("IQ9075 tested BOM name is invalid")
    try:
        verified_bom = verify_release_bom(
            _strict_object(tested_bom_raw, label="IQ9075 tested BOM")
        )
    except ReleaseBomValidationError as exc:
        raise ReadinessError("IQ9075 tested BOM is invalid") from exc
    expected_runtime_slot = (
        f"/opt/nuv-agent/candidates/{run_id}-{verified_bom.bom_digest}"
        if oak_schema == 3
        else "/opt/nuv-agent/releases/"
        + verified_bom.bom_digest
    )
    if (
        runtime_identity["bomDigest"] != "sha256:" + verified_bom.bom_digest
        or runtime_identity["pythonPath"] != "/usr/bin/python3"
        or runtime_identity["sitePackagesPath"]
        != expected_runtime_slot + "/venv/lib/python3.12/site-packages"
        or runtime_identity["buildInfoPath"]
        != expected_runtime_slot
        + "/venv/lib/python3.12/site-packages/nuvion_app/build_info.py"
        or (
            oak_schema == 3
            and runtime_identity.get("candidateSlot")
            != expected_runtime_slot
        )
    ):
        raise ReadinessError("IQ9075 OAK soak used a different installed release")
    publisher_harness = Path(__file__).resolve().parents[1] / "dev/test-iq9075.sh"
    publisher_harness_sha = hashlib.sha256(
        _regular_bytes(publisher_harness)
    ).hexdigest()
    candidate_harness_sha = hashlib.sha256(
        _regular_bytes(candidate_harness)
    ).hexdigest()
    if (
        publisher_harness_sha != manifest["harnessSha256"]
        or candidate_harness_sha != manifest["harnessSha256"]
    ):
        raise ReadinessError("IQ9075 harness bytes differ from signed manifest")
    publisher_fleet_runner = (
        Path(__file__).resolve().parents[1] / "dev/run-iq9075-fleet-e2e.py"
    )
    candidate_fleet_runner = candidate_harness.parent / "run-iq9075-fleet-e2e.py"
    publisher_fleet_runner_raw = _regular_bytes(publisher_fleet_runner)
    candidate_fleet_runner_raw = _regular_bytes(candidate_fleet_runner)
    publisher_fleet_sha = hashlib.sha256(publisher_fleet_runner_raw).hexdigest()
    candidate_fleet_sha = hashlib.sha256(candidate_fleet_runner_raw).hexdigest()
    if (
        publisher_fleet_sha != manifest["fleetRunnerSha256"]
        or candidate_fleet_sha != manifest["fleetRunnerSha256"]
    ):
        raise ReadinessError("IQ9075 Fleet E2E runner differs from signed manifest")
    fleet_manifest_raw, _fleet_manifest_sha = _evidence_reference(
        policy_path.parent,
        manifest.get("fleetManifest"),
        label="IQ9075 Fleet E2E manifest",
    )
    fleet_evidence_raw, _fleet_evidence_sha = _evidence_reference(
        policy_path.parent,
        manifest.get("fleetEvidence"),
        label="IQ9075 Fleet E2E result",
    )
    fleet_manifest = _strict_object(
        fleet_manifest_raw, label="IQ9075 Fleet E2E manifest"
    )
    fleet_evidence = _strict_object(
        fleet_evidence_raw, label="IQ9075 Fleet E2E result"
    )
    publisher_board_tool = (
        Path(__file__).resolve().parents[1] / "dev/iq9075-board-e2e.py"
    )
    candidate_board_tool = candidate_harness.parent / "iq9075-board-e2e.py"
    publisher_board_tool_sha = hashlib.sha256(
        _regular_bytes(publisher_board_tool)
    ).hexdigest()
    candidate_board_tool_sha = hashlib.sha256(
        _regular_bytes(candidate_board_tool)
    ).hexdigest()
    if (
        fleet_manifest.get("toolSha256") != publisher_board_tool_sha
        or fleet_manifest.get("toolSha256") != candidate_board_tool_sha
    ):
        raise ReadinessError(
            "IQ9075 board harness bytes differ from Fleet E2E manifest"
        )
    module_name = "_nuvion_trusted_iq9075_fleet_validator"
    fleet_validator = _module_from_verified_source(
        module_name=module_name,
        source=publisher_fleet_runner_raw,
        display_path=publisher_fleet_runner,
    )
    try:
        fleet_validator.validate_final_evidence(fleet_evidence, fleet_manifest)
        candidate_soak = None
        if oak_schema == 3:
            candidate_soak_raw, _candidate_soak_sha = _evidence_reference(
                policy_path.parent,
                candidate_soak_reference,
                label="IQ9075 candidate soak evidence",
            )
            candidate_soak = _strict_object(
                candidate_soak_raw, label="IQ9075 candidate soak evidence"
            )
            if (
                cleanup_evidence_reference is None
                or cleanup_summary_reference != cleanup_evidence_reference
            ):
                raise ReadinessError(
                    "IQ9075 cleanup evidence reference is missing or unbound"
                )
            cleanup_evidence_raw, cleanup_evidence_sha256 = _evidence_reference(
                policy_path.parent,
                cleanup_evidence_reference,
                label="IQ9075 cleanup evidence",
            )
            cleanup_evidence = _strict_object(
                cleanup_evidence_raw, label="IQ9075 cleanup evidence"
            )
            if (
                candidate_soak.get("cleanupEvidence") != cleanup_evidence
                or candidate_soak.get("cleanupEvidenceSha256")
                != cleanup_evidence_sha256
            ):
                raise ReadinessError(
                    "IQ9075 candidate wrapper cleanup bytes differ"
                )
            fleet_validator.validate_candidate_soak_evidence(
                candidate_soak,
                run_id=run_id,
                manifest=fleet_manifest,
                bundle_sha256=tested_artifact["sha256"],
                bom_sha256=tested_bom_sha256,
                harness_sha256=candidate_harness_sha,
                fleet_evidence_sha256=hashlib.sha256(fleet_evidence_raw).hexdigest(),
                raw_evidence_sha256=hashlib.sha256(oak_soak_raw).hexdigest(),
                cleanup_evidence_sha256=cleanup_evidence_sha256,
                require_cleanup_evidence=True,
                manifest_raw=fleet_manifest_raw,
                fleet_evidence_raw=fleet_evidence_raw,
            )
            if candidate_soak.get("rawEvidence") != oak_soak:
                raise ReadinessError(
                    "IQ9075 candidate soak wrapper differs from raw evidence"
                )
            if candidate_soak.get("rawEvidenceSha256") != hashlib.sha256(
                oak_soak_raw
            ).hexdigest():
                raise ReadinessError(
                    "IQ9075 candidate soak wrapper raw bytes digest differs"
                )
    except Exception as exc:
        raise ReadinessError("IQ9075 Fleet E2E evidence is invalid") from exc
    finally:
        sys.modules.pop(module_name, None)
    if (
        verified_bom.schema_version != 2
        or verified_bom.agent_version != version
        or verified_bom.component_sha != component_sha
        or verified_bom.release_sequence is None
        or verified_bom.release_sequence <= baseline["releaseSequence"]
        or verified_bom.artifact_kind != "agent-bundle"
        or verified_bom.artifact_name != tested_artifact["name"]
        or verified_bom.artifact_sha256 != tested_artifact["sha256"]
        or verified_bom.artifact_size_bytes != tested_artifact["sizeBytes"]
        or len(verified_bom.targets) != 1
        or {
            "productModel": verified_bom.targets[0].product_model,
            "platformProfile": verified_bom.targets[0].platform_profile,
            "hardwareRevision": verified_bom.targets[0].hardware_revision,
            "architecture": verified_bom.targets[0].architecture,
        }
        != target_policy
    ):
        raise ReadinessError("IQ9075 tested BOM does not match the release target")
    fleet_scenario = fleet_manifest.get("scenario")
    fleet_release = (
        fleet_scenario.get("release") if isinstance(fleet_scenario, dict) else None
    )
    fleet_inputs = fleet_manifest.get("inputs")
    fleet_identity = fleet_manifest.get("identity")
    fleet_runtime_pids = fleet_evidence.get("runtimePids")
    fleet_oak = fleet_evidence.get("oak")
    fleet_generated_at = _timestamp(
        fleet_evidence.get("generatedAt"), label="IQ9075 Fleet evidence time"
    )
    if oak_schema == 3:
        if not isinstance(candidate_soak, dict) or not isinstance(
            candidate_soak.get("post"), dict
        ):
            raise ReadinessError("IQ9075 candidate restoration proof is invalid")
        candidate_started_at = _timestamp(
            candidate_soak.get("startedAt"),
            label="IQ9075 candidate operation start time",
        )
        candidate_restored_at = _timestamp(
            candidate_soak["post"].get("restoredAt"),
            label="IQ9075 candidate restoration time",
        )
        candidate_completed_at = _timestamp(
            candidate_soak.get("completedAt"),
            label="IQ9075 candidate operation completion time",
        )
        lifecycle_order_valid = (
            fleet_generated_at
            <= candidate_started_at
            <= soak_started_at
            <= candidate_restored_at
            <= candidate_completed_at
            and candidate_completed_at - fleet_generated_at
            <= dt.timedelta(hours=24)
        )
    else:
        lifecycle_order_valid = (
            fleet_generated_at >= soak_started_at
            and fleet_generated_at - soak_started_at <= dt.timedelta(hours=24)
        )
    if (
        fleet_manifest.get("runId") != run_id
        or not isinstance(fleet_inputs, dict)
        or set(fleet_inputs)
        != {
            "commandSha256",
            "releaseSha256",
            "healthSha256",
            "bindingSha256",
        }
        or any(
            not isinstance(value, str) or not SHA256.fullmatch(value)
            for value in fleet_inputs.values()
        )
        or fleet_inputs.get("releaseSha256")
        != iq_policy.get("publicKeyringSha256")
        or not isinstance(fleet_identity, dict)
        or fleet_identity.get("deviceId") != device_identity["deviceId"]
        or fleet_identity.get("spaceId") != device_identity["spaceId"]
        or not isinstance(fleet_oak, dict)
        or fleet_oak.get("mxidSha256") != manifest.get("oakMxidSha256")
        or fleet_oak.get("attached") is not True
        or fleet_oak.get("bound") is not True
        or not lifecycle_order_valid
        or not isinstance(fleet_scenario, dict)
        or fleet_scenario.get("type") != "oak-fault-rollback"
        or fleet_scenario.get("expectedBomDigest")
        != "sha256:" + verified_bom.bom_digest
        or fleet_scenario.get("expectedCandidateSlot")
        != "/opt/nuv-agent/releases/" + verified_bom.bom_digest
        or fleet_scenario.get("expectedPreviousSlot") != rollback_manifest["slot"]
        or fleet_scenario.get("expectedPreviousVersion")
        != rollback_manifest["agentVersion"]
        or not isinstance(fleet_release, dict)
        or fleet_release.get("agentVersion") != version
        or fleet_release.get("releaseSequence") != verified_bom.release_sequence
        or fleet_release.get("artifactDigest")
        != "sha256:" + verified_bom.artifact_sha256
        or fleet_release.get("componentSha") != component_sha
        or fleet_release.get("configSchema") != verified_bom.config_schema
        or fleet_release.get("publisherKeyId")
        != iq_policy.get("publisherKeyId")
        or not isinstance(fleet_runtime_pids, dict)
        or set(fleet_runtime_pids) != {"candidate", "restored"}
    ):
        raise ReadinessError("IQ9075 Fleet E2E evidence differs from physical run")

    expected_result_fields = {
        "schemaVersion",
        "kind",
        "runId",
        "agentVersion",
        "componentSha",
        "manifestSha256",
        "artifactSha256",
        "bomSha256",
        "exitCode",
        "outcome",
        "soak",
        "webrtc",
        "splitmux",
        "rollback",
    }
    if oak_schema == 3:
        expected_result_fields.update(
            {"candidateRestore", "cleanupEvidenceSha256"}
        )
    if set(result) != expected_result_fields or (
        type(result.get("schemaVersion")) is not int
        or result.get("schemaVersion") != (3 if oak_schema == 3 else 2)
        or result.get("kind") != "nuvion-iq9075-physical-result"
        or result.get("runId") != run_id
        or result.get("agentVersion") != version
        or result.get("componentSha") != component_sha
        or result.get("manifestSha256") != manifest_sha256
        or result.get("artifactSha256") != tested_artifact["sha256"]
        or result.get("bomSha256") != tested_bom_sha256
        or result.get("exitCode") != 0
        or isinstance(result.get("exitCode"), bool)
        or result.get("outcome") != outcome
        or (
            oak_schema == 3
            and (
                not isinstance(candidate_soak, dict)
                or result.get("candidateRestore") != candidate_soak.get("post")
                or result.get("cleanupEvidenceSha256")
                != candidate_soak.get("cleanupEvidenceSha256")
            )
        )
    ):
        raise ReadinessError("IQ9075 harness result identity is invalid")

    soak = result.get("soak")
    if not isinstance(soak, dict) or set(soak) != {
        "durationSeconds",
        "targetFps",
        "rawSamples",
        "rssAnonSamples",
        "rssAnonSlopeMiBPerMin",
        "rssAnonRangeMiB",
        "gstreamerErrors",
        "gstreamerWarnings",
        "maxAppsrcBuffers",
        "maxAppsrcBytes",
        "queueHighWatermarks",
    }:
        raise ReadinessError("IQ9075 soak result fields are invalid")
    if (
        soak != oak_soak.get("soak")
        or result.get("webrtc") != oak_soak.get("webrtc")
        or result.get("splitmux") != oak_soak.get("splitmux")
    ):
        raise ReadinessError("IQ9075 physical result differs from OAK soak source")
    duration = _number(
        soak.get("durationSeconds"),
        label="IQ9075 soak duration",
        minimum=120.0,
        maximum=600.0,
    )
    target_fps = _number(
        soak.get("targetFps"), label="IQ9075 target FPS", minimum=30.0, maximum=30.0
    )
    raw_samples = _integer(
        soak.get("rawSamples"), label="IQ9075 raw sample count", minimum=1
    )
    raw_fps = raw_samples / duration
    if raw_fps < 27.0 or raw_samples < math.floor(0.9 * target_fps * duration):
        raise ReadinessError("IQ9075 raw frame throughput is below the release bound")
    _integer(
        soak.get("maxAppsrcBuffers"),
        label="IQ9075 appsrc buffer high-watermark",
        maximum=2,
    )
    _integer(
        soak.get("maxAppsrcBytes"),
        label="IQ9075 appsrc byte high-watermark",
        maximum=MAX_APPSRC_BYTES,
    )
    errors = soak.get("gstreamerErrors")
    if not isinstance(errors, list) or errors:
        raise ReadinessError("IQ9075 GStreamer errors were observed")
    warnings = soak.get("gstreamerWarnings")
    if (
        not isinstance(warnings, list)
        or len(warnings) > 256
        or any(
            not isinstance(item, str) or not item or len(item) > 2000
            for item in warnings
        )
    ):
        raise ReadinessError("IQ9075 GStreamer warnings are invalid")
    queue_levels = soak.get("queueHighWatermarks")
    expected_queues = {
        "physical_raw_queue",
        "physical_overlay_queue",
        "uplink_live_queue",
        "clip_live_queue",
    }
    if (
        not isinstance(queue_levels, dict)
        or set(queue_levels) != expected_queues
        or any(
            not isinstance(name, str)
            or not SAFE_NAME.fullmatch(name)
            or _integer(value, label=f"IQ9075 queue {name}", maximum=2) > 2
            for name, value in queue_levels.items()
        )
    ):
        raise ReadinessError("IQ9075 queue high-watermarks are invalid")

    rss_raw = soak.get("rssAnonSamples")
    if not isinstance(rss_raw, list) or len(rss_raw) < 18 or len(rss_raw) > 256:
        raise ReadinessError("IQ9075 RSS sample count is invalid")
    rss_samples: list[tuple[float, float]] = []
    for sample in rss_raw:
        if not isinstance(sample, dict) or set(sample) != {"elapsedSec", "rssAnonKiB"}:
            raise ReadinessError("IQ9075 RSS sample fields are invalid")
        elapsed = _number(
            sample.get("elapsedSec"),
            label="IQ9075 RSS elapsed time",
            minimum=0.0,
            maximum=duration,
        )
        rss_kib = _integer(
            sample.get("rssAnonKiB"), label="IQ9075 anonymous RSS", minimum=1
        )
        if rss_samples and elapsed <= rss_samples[-1][0]:
            raise ReadinessError("IQ9075 RSS sample times are not strictly ordered")
        rss_samples.append((elapsed, rss_kib / 1024.0))
    if (
        rss_samples[0][0] > 5.0
        or duration - rss_samples[-1][0] > 5.0
        or any(
            right[0] - left[0] > 10.0
            for left, right in zip(rss_samples, rss_samples[1:])
        )
    ):
        raise ReadinessError("IQ9075 RSS sampling has an unbounded time gap")
    mean_x = sum(item[0] / 60.0 for item in rss_samples) / len(rss_samples)
    mean_y = sum(item[1] for item in rss_samples) / len(rss_samples)
    denominator = sum((item[0] / 60.0 - mean_x) ** 2 for item in rss_samples)
    if denominator <= 0:
        raise ReadinessError("IQ9075 RSS sample times have no range")
    rss_slope = sum(
        (elapsed / 60.0 - mean_x) * (rss_mib - mean_y)
        for elapsed, rss_mib in rss_samples
    ) / denominator
    rss_values = [value for _elapsed, value in rss_samples]
    rss_range = max(rss_values) - min(rss_values)
    reported_rss_slope = _number(
        soak.get("rssAnonSlopeMiBPerMin"),
        label="IQ9075 reported RSS slope",
    )
    reported_rss_range = _number(
        soak.get("rssAnonRangeMiB"),
        label="IQ9075 reported RSS range",
        minimum=0.0,
    )
    if (
        not math.isclose(reported_rss_slope, rss_slope, rel_tol=0.0, abs_tol=0.001)
        or not math.isclose(reported_rss_range, rss_range, rel_tol=0.0, abs_tol=0.001)
    ):
        raise ReadinessError("IQ9075 reported RSS metrics differ from samples")
    if abs(rss_slope) > 2.0 or rss_range > 32.0:
        raise ReadinessError("IQ9075 RSS growth exceeds the release bound")

    webrtc = result.get("webrtc")
    if not isinstance(webrtc, dict) or set(webrtc) != {
        "offerCount",
        "terminalStopCount",
        "offerSdpHadPinnedProfile",
        "branchParentDetached",
        "queueParentDetached",
        "webrtcParentDetached",
        "teeRequestPadCount",
        "queueState",
        "webrtcState",
        "branchObjectsFinalized",
        "hasPipeline",
    } or (
        webrtc.get("offerCount") != 1
        or isinstance(webrtc.get("offerCount"), bool)
        or webrtc.get("terminalStopCount") != 1
        or isinstance(webrtc.get("terminalStopCount"), bool)
        or webrtc.get("offerSdpHadPinnedProfile") is not True
        or webrtc.get("branchParentDetached") is not True
        or webrtc.get("queueParentDetached") is not True
        or webrtc.get("webrtcParentDetached") is not True
        or webrtc.get("teeRequestPadCount") != 0
        or isinstance(webrtc.get("teeRequestPadCount"), bool)
        or webrtc.get("queueState") != "NULL"
        or webrtc.get("webrtcState") != "NULL"
        or webrtc.get("branchObjectsFinalized") is not True
        or webrtc.get("hasPipeline") is not False
    ):
        raise ReadinessError("IQ9075 WebRTC teardown proof is invalid")

    splitmux = result.get("splitmux")
    if not isinstance(splitmux, dict) or set(splitmux) != {
        "segmentSeconds",
        "retentionLimit",
        "segmentsAtEnd",
        "fragmentsOpenedDuringSoak",
        "newestSegmentAgeSeconds",
    }:
        raise ReadinessError("IQ9075 splitmux evidence fields are invalid")
    segment_seconds = _number(
        splitmux.get("segmentSeconds"),
        label="IQ9075 segment duration",
        minimum=4.0,
        maximum=4.0,
    )
    retention = _integer(
        splitmux.get("retentionLimit"),
        label="IQ9075 segment retention",
        minimum=30,
        maximum=30,
    )
    segments = _integer(
        splitmux.get("segmentsAtEnd"),
        label="IQ9075 retained segments",
        minimum=1,
        maximum=retention,
    )
    fragments = _integer(
        splitmux.get("fragmentsOpenedDuringSoak"),
        label="IQ9075 fragment progress",
        minimum=0,
    )
    newest_age = _number(
        splitmux.get("newestSegmentAgeSeconds"),
        label="IQ9075 newest segment age",
        minimum=0.0,
        maximum=2 * segment_seconds + 5.0,
    )
    if fragments < max(1, math.floor(duration / segment_seconds) - 3):
        raise ReadinessError("IQ9075 splitmux fragment progress is below bound")
    splitmux_rotated = segments > 0 and fragments > 0 and newest_age <= 2 * segment_seconds + 5

    rollback = result.get("rollback")
    if not isinstance(rollback, dict) or set(rollback) != {
        "expectedSlot",
        "candidateSlot",
        "restoredSlot",
        "candidatePid",
        "restoredPid",
        "oakProbeExitCode",
        "oakReady",
    }:
        raise ReadinessError("IQ9075 rollback evidence fields are invalid")
    candidate_pid = _integer(
        rollback.get("candidatePid"), label="IQ9075 candidate PID", minimum=2
    )
    restored_pid = _integer(
        rollback.get("restoredPid"), label="IQ9075 restored PID", minimum=2
    )
    if (
        rollback.get("expectedSlot") != rollback_manifest["slot"]
        or rollback.get("restoredSlot") != rollback_manifest["slot"]
        or not isinstance(rollback.get("candidateSlot"), str)
        or rollback["candidateSlot"]
        != "releases/" + verified_bom.bom_digest
        or candidate_pid != fleet_runtime_pids["candidate"]
        or restored_pid != fleet_runtime_pids["restored"]
        or candidate_pid == restored_pid
        or rollback.get("oakProbeExitCode") != 0
        or isinstance(rollback.get("oakProbeExitCode"), bool)
        or rollback.get("oakReady") is not True
    ):
        raise ReadinessError("IQ9075 rollback proof is invalid")

    expected_gate = {
        "oakSoakSeconds": round(duration, 6),
        "rawFps": round(raw_fps, 6),
        "rssSlopeMiBPerMinute": round(rss_slope, 6),
        "rssRangeMiB": round(rss_range, 6),
        "gstreamerErrors": len(errors),
        "webrtcBranchDisposed": True,
        "splitmuxRotated": splitmux_rotated,
        "rollbackOakReady": True,
    }
    gate = summary.get("physicalGate")
    if not isinstance(gate, dict):
        raise ReadinessError("IQ9075 physical gate summary is invalid")
    for key in (
        "oakSoakSeconds",
        "rawFps",
        "rssSlopeMiBPerMinute",
        "rssRangeMiB",
    ):
        _number(gate.get(key), label=f"IQ9075 summary {key}")
    if gate != expected_gate:
        raise ReadinessError("IQ9075 physical gate summary differs from raw result")
    if expected_gate["splitmuxRotated"] is not True:
        raise ReadinessError("IQ9075 splitmux did not rotate")
    return {
        "physical_artifact_sha256": verified_bom.artifact_sha256,
        "physical_bom_sha256": tested_bom_sha256,
    }


def _validate_ready_evidence(
    *,
    policy_path: Path,
    version: str,
    decision: object,
    component_sha: str | None,
    gate_evidence: dict[str, object] | None,
    security_policy: Path | None,
    signer_directory: Path | None,
    candidate_harness: Path | None,
    candidate_fleet_runner: Path | None,
    candidate_board_tool: Path | None,
) -> dict[str, str]:
    common_fields = {"componentSha", "agentReleaseGate"}
    runtime_fields = common_fields | {"iq9075FleetRuntime"}
    legacy_fields = common_fields | {"iq9075Physical"}
    if (
        not isinstance(decision, dict)
        or frozenset(decision)
        not in {frozenset(runtime_fields), frozenset(legacy_fields)}
        or component_sha is None
        or not SHA.fullmatch(component_sha)
        or decision.get("componentSha") != component_sha
        or gate_evidence is None
    ):
        raise ReadinessError("READY release lacks exact component evidence")
    expected_gate_keys = {
        "componentSha",
        "workflow",
        "workflowSha256",
        "workflowRunId",
        "checkRunId",
        "checkSuiteId",
        "context",
        "integrationId",
    }
    recorded_gate = decision.get("agentReleaseGate")
    if (
        not isinstance(recorded_gate, dict)
        or set(recorded_gate) != expected_gate_keys
        or recorded_gate != gate_evidence
    ):
        raise ReadinessError("READY release gate evidence does not match live GitHub proof")

    evidence_key = (
        "iq9075FleetRuntime"
        if "iq9075FleetRuntime" in decision
        else "iq9075Physical"
    )
    evidence_kind = (
        "Fleet Runtime" if evidence_key == "iq9075FleetRuntime" else "physical"
    )
    signed_evidence = decision.get(evidence_key)
    if not isinstance(signed_evidence, dict) or set(signed_evidence) != {
        "evidenceFile",
        "evidenceSha256",
        "signatureFile",
        "signatureSha256",
        "signerFingerprint",
    }:
        raise ReadinessError(
            f"READY release {evidence_kind} evidence identity is invalid"
        )
    evidence_name = signed_evidence.get("evidenceFile")
    signature_name = signed_evidence.get("signatureFile")
    evidence_sha256 = signed_evidence.get("evidenceSha256")
    signature_sha256 = signed_evidence.get("signatureSha256")
    signer_fingerprint = signed_evidence.get("signerFingerprint")
    if (
        not isinstance(evidence_name, str)
        or not SAFE_NAME.fullmatch(evidence_name)
        or not isinstance(signature_name, str)
        or not SAFE_NAME.fullmatch(signature_name)
        or not isinstance(evidence_sha256, str)
        or not SHA256.fullmatch(evidence_sha256)
        or not isinstance(signature_sha256, str)
        or not SHA256.fullmatch(signature_sha256)
        or not isinstance(signer_fingerprint, str)
        or not FINGERPRINT.fullmatch(signer_fingerprint)
        or security_policy is None
        or signer_directory is None
        or (
            evidence_key == "iq9075FleetRuntime"
            and (candidate_fleet_runner is None or candidate_board_tool is None)
        )
        or (evidence_key == "iq9075Physical" and candidate_harness is None)
    ):
        raise ReadinessError(
            f"READY release {evidence_kind} evidence identity is invalid"
        )
    evidence_path = policy_path.parent / evidence_name
    signature_path = policy_path.parent / signature_name
    evidence_raw = _regular_bytes(evidence_path)
    signature_raw = _regular_bytes(signature_path)
    if hashlib.sha256(evidence_raw).hexdigest() != evidence_sha256:
        raise ReadinessError(
            f"{evidence_kind} evidence SHA-256 does not match readiness"
        )
    if hashlib.sha256(signature_raw).hexdigest() != signature_sha256:
        raise ReadinessError(
            f"{evidence_kind} evidence signature SHA-256 does not match readiness"
        )

    security = _strict_object(
        _regular_bytes(security_policy),
        label="release security policy",
    )
    fingerprints = security.get("trustedTagSignerFingerprints")
    if (
        not isinstance(fingerprints, list)
        or not fingerprints
        or any(
            not isinstance(value, str) or not FINGERPRINT.fullmatch(value)
            for value in fingerprints
        )
        or len(set(fingerprints)) != len(fingerprints)
    ):
        raise ReadinessError(f"{evidence_kind} evidence signer policy is invalid")
    verified_fingerprint = _verify_detached_signature(
        evidence_raw,
        signature_raw,
        signer_directory=signer_directory,
        allowed_fingerprints=set(fingerprints),
    )
    if verified_fingerprint != signer_fingerprint:
        raise ReadinessError(
            f"{evidence_kind} evidence signer differs from readiness"
        )

    document = _strict_object(
        evidence_raw, label=f"IQ9075 {evidence_kind} evidence"
    )
    if evidence_key == "iq9075FleetRuntime":
        assert candidate_fleet_runner is not None
        assert candidate_board_tool is not None
        return _validate_fleet_runtime_documents(
            policy_path=policy_path,
            version=version,
            component_sha=component_sha,
            summary=document,
            security=security,
            candidate_fleet_runner=candidate_fleet_runner,
            candidate_board_tool=candidate_board_tool,
        )
    assert candidate_harness is not None
    return _validate_physical_documents(
        policy_path=policy_path,
        version=version,
        component_sha=component_sha,
        summary=document,
        security=security,
        candidate_harness=candidate_harness,
    )


def verify_readiness(
    path: Path,
    *,
    version: str,
    component_sha: str | None = None,
    gate_evidence: dict[str, object] | None = None,
    security_policy: Path | None = None,
    signer_directory: Path | None = None,
    candidate_harness: Path | None = None,
    candidate_fleet_runner: Path | None = None,
    candidate_board_tool: Path | None = None,
) -> dict[str, str]:
    if not SEMVER.fullmatch(version):
        raise ReadinessError("release readiness version must be exact SemVer")
    payload = _load(path)
    releases = payload["releases"]
    for configured_version in releases:
        if not isinstance(configured_version, str) or not SEMVER.fullmatch(configured_version):
            raise ReadinessError("release readiness contains an invalid version key")
    release = releases.get(version)
    if not isinstance(release, dict) or set(release) != {
        "status",
        "blockers",
        "evidence",
    }:
        raise ReadinessError(f"release {version} has no reviewed readiness decision")
    status = release.get("status")
    blockers = release.get("blockers")
    if status not in {"READY", "BLOCKED"} or not isinstance(blockers, list):
        raise ReadinessError("release readiness decision is invalid")
    blocker_ids: list[str] = []
    for blocker in blockers:
        if not isinstance(blocker, dict):
            raise ReadinessError("release readiness blocker is invalid")
        identifier = blocker.get("id")
        if not isinstance(identifier, str) or not BLOCKER_ID.fullmatch(identifier):
            raise ReadinessError("release readiness blocker id is invalid")
        blocker_ids.append(identifier)
    if len(set(blocker_ids)) != len(blocker_ids):
        raise ReadinessError("release readiness blocker ids are duplicated")
    if status == "READY" and blocker_ids:
        raise ReadinessError("READY release must have no blockers")
    if status == "BLOCKED" and not blocker_ids:
        raise ReadinessError("BLOCKED release must identify at least one blocker")
    if status == "BLOCKED" and release.get("evidence") is not None:
        raise ReadinessError("BLOCKED release cannot carry promotable evidence")
    if status != "READY":
        raise ReadinessError(
            f"release {version} is blocked by: {', '.join(sorted(blocker_ids))}"
        )
    return _validate_ready_evidence(
        policy_path=path,
        version=version,
        decision=release.get("evidence"),
        component_sha=component_sha,
        gate_evidence=gate_evidence,
        security_policy=security_policy,
        signer_directory=signer_directory,
        candidate_harness=candidate_harness,
        candidate_fleet_runner=candidate_fleet_runner,
        candidate_board_tool=candidate_board_tool,
    )


def main() -> int:
    if not sys.flags.isolated:
        print(
            "verify-release-readiness.py requires Python isolated mode (-I)",
            file=sys.stderr,
        )
        return 2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--component-sha")
    parser.add_argument("--gate-run-id", type=int)
    parser.add_argument("--gate-check-id", type=int)
    parser.add_argument("--gate-check-suite-id", type=int)
    parser.add_argument("--gate-workflow-sha256")
    parser.add_argument("--security-policy", type=Path)
    parser.add_argument("--signer-directory", type=Path)
    parser.add_argument("--candidate-harness", type=Path)
    parser.add_argument("--candidate-fleet-runner", type=Path)
    parser.add_argument("--candidate-board-tool", type=Path)
    parser.add_argument("--github-output", type=Path)
    arguments = parser.parse_args()
    try:
        result = verify_readiness(
            arguments.policy,
            version=arguments.version,
            component_sha=arguments.component_sha,
            gate_evidence={
                "componentSha": arguments.component_sha,
                "workflow": ".github/workflows/agent-release-gate.yml",
                "workflowSha256": arguments.gate_workflow_sha256,
                "workflowRunId": arguments.gate_run_id,
                "checkRunId": arguments.gate_check_id,
                "checkSuiteId": arguments.gate_check_suite_id,
                "context": "agent-release-gate",
                "integrationId": 15368,
            },
            security_policy=arguments.security_policy,
            signer_directory=arguments.signer_directory,
            candidate_harness=arguments.candidate_harness,
            candidate_fleet_runner=arguments.candidate_fleet_runner,
            candidate_board_tool=arguments.candidate_board_tool,
        )
        if arguments.github_output is not None:
            with arguments.github_output.open("a", encoding="utf-8") as output:
                for key, value in result.items():
                    output.write(f"{key}={value}\n")
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    except ReadinessError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
