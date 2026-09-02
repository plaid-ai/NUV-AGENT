#!/usr/bin/env python3
"""Fail closed when a release has unresolved audited dependency blockers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any

SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
BLOCKER_ID = re.compile(r"^[A-Z0-9][A-Z0-9._-]{0,127}$")
SHA = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
FINGERPRINT = re.compile(r"^[0-9A-F]{40}$")
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")
MAX_EVIDENCE_BYTES = 1024 * 1024


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
            path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReadinessError("release readiness document is invalid") from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schemaVersion", "releases"}
        or payload.get("schemaVersion") != 2
        or not isinstance(payload.get("releases"), dict)
    ):
        raise ReadinessError("release readiness document does not match schema v2")
    return payload


def _regular_bytes(path: Path) -> bytes:
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise ReadinessError(f"release evidence is not a regular file: {path.name}")
        if before.st_size < 1 or before.st_size > MAX_EVIDENCE_BYTES:
            raise ReadinessError(f"release evidence size is invalid: {path.name}")
        raw = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise ReadinessError(f"release evidence is unavailable: {path.name}") from exc
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(before) != identity(after) or len(raw) != before.st_size:
        raise ReadinessError(f"release evidence changed while reading: {path.name}")
    return raw


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
    evidence_path: Path,
    signature_path: Path,
    *,
    signer_directory: Path,
    allowed_fingerprints: set[str],
) -> str:
    public_keys = sorted(signer_directory.glob("*.asc"))
    if not public_keys:
        raise ReadinessError("physical-evidence signer directory is empty")
    with tempfile.TemporaryDirectory(prefix="nuv-physical-evidence-verify-") as raw_home:
        gpg_home = Path(raw_home)
        gpg_home.chmod(0o700)
        imported = subprocess.run(
            ["gpg", "--batch", "--homedir", str(gpg_home), "--import", *map(str, public_keys)],
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
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicate)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ReadinessError(f"{label} is invalid") from exc
    if not isinstance(payload, dict):
        raise ReadinessError(f"{label} must be an object")
    return payload


def _validate_ready_evidence(
    *,
    policy_path: Path,
    version: str,
    decision: object,
    component_sha: str | None,
    gate_evidence: dict[str, object] | None,
    security_policy: Path | None,
    signer_directory: Path | None,
) -> None:
    if (
        not isinstance(decision, dict)
        or set(decision) != {"componentSha", "agentReleaseGate", "iq9075Physical"}
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

    physical = decision.get("iq9075Physical")
    if not isinstance(physical, dict) or set(physical) != {
        "evidenceFile",
        "evidenceSha256",
        "signatureFile",
        "signatureSha256",
        "signerFingerprint",
    }:
        raise ReadinessError("READY release physical evidence identity is invalid")
    evidence_name = physical.get("evidenceFile")
    signature_name = physical.get("signatureFile")
    evidence_sha256 = physical.get("evidenceSha256")
    signature_sha256 = physical.get("signatureSha256")
    signer_fingerprint = physical.get("signerFingerprint")
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
    ):
        raise ReadinessError("READY release physical evidence identity is invalid")
    evidence_path = policy_path.parent / evidence_name
    signature_path = policy_path.parent / signature_name
    evidence_raw = _regular_bytes(evidence_path)
    signature_raw = _regular_bytes(signature_path)
    if hashlib.sha256(evidence_raw).hexdigest() != evidence_sha256:
        raise ReadinessError("physical evidence SHA-256 does not match readiness")
    if hashlib.sha256(signature_raw).hexdigest() != signature_sha256:
        raise ReadinessError("physical evidence signature SHA-256 does not match readiness")

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
        raise ReadinessError("physical evidence signer policy is invalid")
    verified_fingerprint = _verify_detached_signature(
        evidence_path,
        signature_path,
        signer_directory=signer_directory,
        allowed_fingerprints=set(fingerprints),
    )
    if verified_fingerprint != signer_fingerprint:
        raise ReadinessError("physical evidence signer differs from readiness")

    document = _strict_object(evidence_raw, label="IQ9075 physical evidence")
    if set(document) != {
        "schemaVersion",
        "kind",
        "agentVersion",
        "componentSha",
        "harnessManifestSha256",
        "harnessEvidenceSha256",
        "physicalGate",
    }:
        raise ReadinessError("IQ9075 physical evidence fields are invalid")
    gate = document.get("physicalGate")
    if (
        document.get("schemaVersion") != 1
        or document.get("kind") != "nuvion-iq9075-physical-release-evidence"
        or document.get("agentVersion") != version
        or document.get("componentSha") != component_sha
        or not isinstance(document.get("harnessManifestSha256"), str)
        or not SHA256.fullmatch(document["harnessManifestSha256"])
        or not isinstance(document.get("harnessEvidenceSha256"), str)
        or not SHA256.fullmatch(document["harnessEvidenceSha256"])
        or not isinstance(gate, dict)
        or set(gate)
        != {
            "oakSoakSeconds",
            "rawFps",
            "rssSlopeMiBPerMinute",
            "rssRangeMiB",
            "gstreamerErrors",
            "webrtcBranchDisposed",
            "splitmuxRotated",
            "rollbackOakReady",
        }
    ):
        raise ReadinessError("IQ9075 physical evidence does not match release")
    numeric = {
        key: gate.get(key)
        for key in (
            "oakSoakSeconds",
            "rawFps",
            "rssSlopeMiBPerMinute",
            "rssRangeMiB",
        )
    }
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in numeric.values()
    ):
        raise ReadinessError("IQ9075 physical gate metrics are invalid")
    if (
        float(numeric["oakSoakSeconds"]) < 120.0
        or float(numeric["rawFps"]) < 27.0
        or abs(float(numeric["rssSlopeMiBPerMinute"])) > 2.0
        or not 0.0 <= float(numeric["rssRangeMiB"]) <= 32.0
        or gate.get("gstreamerErrors") != 0
        or gate.get("webrtcBranchDisposed") is not True
        or gate.get("splitmuxRotated") is not True
        or gate.get("rollbackOakReady") is not True
    ):
        raise ReadinessError("IQ9075 physical release gate did not pass")


def verify_readiness(
    path: Path,
    *,
    version: str,
    component_sha: str | None = None,
    gate_evidence: dict[str, object] | None = None,
    security_policy: Path | None = None,
    signer_directory: Path | None = None,
) -> None:
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
    _validate_ready_evidence(
        policy_path=path,
        version=version,
        decision=release.get("evidence"),
        component_sha=component_sha,
        gate_evidence=gate_evidence,
        security_policy=security_policy,
        signer_directory=signer_directory,
    )


def main() -> int:
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
    arguments = parser.parse_args()
    try:
        verify_readiness(
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
        )
    except ReadinessError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
