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
import uuid
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
FLEET_RUNTIME_REQUIRED_FROM = (0, 1, 121)
IQ9075_QUALIFICATION_API_ORIGIN = "https://api.nuvion-dev.plaidlabs.ai"
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


def _validated_config_stream_gate(
    *,
    config_stream_evidence: dict[str, Any],
    fleet_manifest: dict[str, Any],
    fleet_manifest_raw: bytes,
    fleet_evidence: dict[str, Any],
    fleet_evidence_raw: bytes,
    cleanup_evidence: dict[str, Any],
    rollback_manifest: dict[str, Any],
    rollback_evidence: dict[str, Any],
    candidate_config_stream_runner: Path,
) -> tuple[dict[str, Any], str]:
    publisher_runner = (
        Path(__file__).resolve().parents[1]
        / "dev/run-iq9075-config-stream-e2e.py"
    )
    publisher_raw = _regular_bytes(publisher_runner)
    candidate_raw = _regular_bytes(candidate_config_stream_runner)
    runner_sha256 = hashlib.sha256(publisher_raw).hexdigest()
    if candidate_raw != publisher_raw:
        raise ReadinessError(
            "IQ9075 config-stream runner differs from trusted publisher"
        )
    try:
        def exact(value: object, fields: set[str], label: str) -> dict[str, Any]:
            if not isinstance(value, dict) or set(value) != fields:
                raise ReadinessError(f"IQ9075 config-stream {label} is invalid")
            return value

        def canonical_config(value: dict[str, Any]) -> bytes:
            try:
                return (
                    json.dumps(
                        value,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8")
            except (TypeError, ValueError, RecursionError) as exc:
                raise ReadinessError(
                    "IQ9075 config-stream value is not canonical JSON"
                ) from exc

        def config_digest(value: dict[str, Any]) -> str:
            encoded = canonical_config(value)[:-1]
            return "sha256:" + hashlib.sha256(encoded).hexdigest()

        def baseline(value: object) -> dict[str, Any]:
            item = exact(
                value,
                {"model", "labels", "clip", "video"},
                "settings baseline",
            )
            model = exact(
                item.get("model"),
                {
                    "pointer",
                    "configuredDigest",
                    "artifactDigest",
                    "artifactVerified",
                    "runtimeEnabled",
                    "runtimeBackend",
                },
                "model baseline",
            )
            labels = exact(
                item.get("labels"),
                {"inspection", "anomaly"},
                "label baseline",
            )
            clip = exact(
                item.get("clip"),
                {"enabled", "preSeconds", "postSeconds"},
                "clip baseline",
            )
            video = exact(
                item.get("video"),
                {"width", "height", "fps", "bitrateKbps"},
                "video baseline",
            )
            if (
                not isinstance(model.get("pointer"), str)
                or not model["pointer"]
                or any(
                    digest is not None
                    and (
                        not isinstance(digest, str)
                        or re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
                        is None
                    )
                    for digest in (
                        model.get("configuredDigest"),
                        model.get("artifactDigest"),
                    )
                )
                or type(model.get("artifactVerified")) is not bool
                or type(model.get("runtimeEnabled")) is not bool
                or not isinstance(model.get("runtimeBackend"), str)
                or not model["runtimeBackend"]
                or (
                    model["artifactVerified"] is True
                    and model["artifactDigest"] is None
                )
                or any(
                    not isinstance(values, list)
                    or any(
                        not isinstance(entry, str) or not entry
                        for entry in values
                    )
                    for values in labels.values()
                )
                or type(clip.get("enabled")) is not bool
                or type(clip.get("preSeconds")) is not int
                or type(clip.get("postSeconds")) is not int
                or any(type(video.get(name)) is not int for name in video)
            ):
                raise ReadinessError(
                    "IQ9075 config-stream settings baseline is invalid"
                )
            return item

        def queue_drained(value: object) -> dict[str, Any]:
            fields = {
                "inboxPendingRows",
                "observationPendingRows",
                "observationReservedRows",
                "observationDlqRows",
            }
            queue = exact(value, fields, "command queue")
            if any(type(queue.get(name)) is not int or queue[name] != 0 for name in fields):
                raise ReadinessError(
                    "IQ9075 config-stream command queue is not drained"
                )
            return queue

        def runtime_identity(
            value: object,
            *,
            release: dict[str, Any],
            bom_digest: object,
        ) -> dict[str, Any]:
            item = exact(
                value,
                {
                    "activeSlot",
                    "processActiveSlot",
                    "processExpectedBomDigest",
                    "servicePid",
                    "releaseMarkerSha256",
                    "buildInfoSha256",
                    "release",
                },
                "runtime release identity",
            )
            if not isinstance(bom_digest, str) or re.fullmatch(
                r"sha256:[0-9a-f]{64}", bom_digest
            ) is None:
                raise ReadinessError(
                    "IQ9075 config-stream runtime BOM identity is invalid"
                )
            slot = "releases/" + bom_digest[7:]
            marker = {"schemaVersion": 2, "bomDigest": bom_digest, **release}
            expected_marker_sha256 = hashlib.sha256(
                canonical_config(marker)
            ).hexdigest()
            expected_build_info = (
                '"""Generated release identity. Do not edit in release artifacts."""\n\n'
                f'AGENT_VERSION = "{release.get("agentVersion")}"\n'
                f'COMPONENT_SHA = "{release.get("componentSha")}"\n'
            ).encode("utf-8")
            expected_build_info_sha256 = hashlib.sha256(
                expected_build_info
            ).hexdigest()
            if (
                item.get("activeSlot") != slot
                or item.get("processActiveSlot") != slot
                or item.get("processExpectedBomDigest") != bom_digest
                or type(item.get("servicePid")) is not int
                or item["servicePid"] < 2
                or item.get("releaseMarkerSha256")
                != expected_marker_sha256
                or item.get("buildInfoSha256")
                != expected_build_info_sha256
                or item.get("release") != marker
            ):
                raise ReadinessError(
                    "IQ9075 config-stream runtime release identity is invalid"
                )
            return item

        def command(value: object, command_type: str) -> dict[str, Any]:
            item = exact(
                value,
                {
                    "commandId",
                    "sequence",
                    "type",
                    "lifecycleAckStatuses",
                    "effectPhase",
                    "reportedState",
                    "reportedRevision",
                    "localObservationRevision",
                    "boardSettings",
                    "boardSettingsSha256",
                    "projectionShape",
                    "queue",
                },
                f"{command_type} command",
            )
            try:
                command_id = str(uuid.UUID(str(item.get("commandId") or "")))
            except ValueError as exc:
                raise ReadinessError(
                    "IQ9075 config-stream command identity is invalid"
                ) from exc
            board_settings = baseline(item.get("boardSettings"))
            settings_sha = item.get("boardSettingsSha256")
            if (
                command_id != item.get("commandId")
                or item.get("type") != command_type
                or item.get("lifecycleAckStatuses")
                != ["RECEIVED", "IN_PROGRESS", "SUCCEEDED"]
                or item.get("effectPhase") != "APPLIED"
                or type(item.get("sequence")) is not int
                or item["sequence"] < 1
                or type(item.get("reportedRevision")) is not int
                or item["reportedRevision"] < 1
                or type(item.get("localObservationRevision")) is not int
                or item["localObservationRevision"] < 1
                or item["reportedRevision"]
                != item["localObservationRevision"]
                or item.get("projectionShape")
                != document.get("projectionShape")
                or not isinstance(item.get("reportedState"), dict)
                or not isinstance(settings_sha, str)
                or SHA256.fullmatch(settings_sha) is None
                or hashlib.sha256(
                    canonical_config(board_settings)
                ).hexdigest()
                != settings_sha
            ):
                raise ReadinessError(
                    "IQ9075 config-stream command proof is invalid"
                )
            queue_drained(item.get("queue"))
            return item

        document = exact(
            config_stream_evidence,
            {
                "schemaVersion",
                "kind",
                "runId",
                "generatedAt",
                "source",
                "identity",
                "releaseCommand",
                "priorRollbackCommand",
                "expiredPredecessors",
                "projectionShape",
                "config",
                "stream",
                "boardPreparation",
                "gates",
                "modelQualification",
                "cleanup",
            },
            "evidence root",
        )
        identity = fleet_manifest.get("identity")
        scenario = fleet_manifest.get("scenario")
        release = scenario.get("release") if isinstance(scenario, dict) else None
        source = exact(
            document.get("source"),
            {
                "manifestSha256",
                "otaEvidenceSha256",
                "apiOrigin",
                "agentVersion",
                "componentSha",
                "bomDigest",
                "configSchema",
                "releaseSequence",
                "artifactDigest",
                "publisherKeyId",
                "runtimeIdentity",
            },
            "source",
        )
        expected_source = {
            "manifestSha256": hashlib.sha256(fleet_manifest_raw).hexdigest(),
            "otaEvidenceSha256": hashlib.sha256(fleet_evidence_raw).hexdigest(),
            "apiOrigin": IQ9075_QUALIFICATION_API_ORIGIN,
            "agentVersion": release.get("agentVersion")
            if isinstance(release, dict)
            else None,
            "componentSha": release.get("componentSha")
            if isinstance(release, dict)
            else None,
            "bomDigest": scenario.get("expectedBomDigest")
            if isinstance(scenario, dict)
            else None,
            "configSchema": release.get("configSchema")
            if isinstance(release, dict)
            else None,
            "releaseSequence": release.get("releaseSequence")
            if isinstance(release, dict)
            else None,
            "artifactDigest": release.get("artifactDigest")
            if isinstance(release, dict)
            else None,
            "publisherKeyId": release.get("publisherKeyId")
            if isinstance(release, dict)
            else None,
            "runtimeIdentity": source.get("runtimeIdentity"),
        }
        if (
            type(document.get("schemaVersion")) is not int
            or document.get("schemaVersion") != 1
            or document.get("kind")
            != "nuvion-iq9075-config-stream-e2e-evidence"
            or document.get("runId") != fleet_manifest.get("runId")
            or not isinstance(identity, dict)
            or document.get("identity") != identity
            or source != expected_source
        ):
            raise ReadinessError(
                "IQ9075 config-stream source binding is invalid"
            )
        if (
            not isinstance(release, dict)
            or not isinstance(scenario, dict)
            or set(release)
            != {
                "agentVersion",
                "releaseSequence",
                "artifactDigest",
                "componentSha",
                "configSchema",
                "publisherKeyId",
            }
            or document.get("projectionShape") not in {"single", "domained"}
        ):
            raise ReadinessError(
                "IQ9075 config-stream release projection is invalid"
            )
        initial_runtime_identity = runtime_identity(
            source.get("runtimeIdentity"),
            release=release,
            bom_digest=scenario.get("expectedBomDigest"),
        )

        fleet_generated = _timestamp(
            fleet_evidence.get("generatedAt"), label="Fleet result generation"
        )
        config_generated = _timestamp(
            document.get("generatedAt"), label="config-stream result generation"
        )
        fleet_cleanup_completed = _timestamp(
            cleanup_evidence.get("completedAt"), label="Fleet cleanup completion"
        )
        if not fleet_generated <= config_generated <= fleet_cleanup_completed:
            raise ReadinessError(
                "IQ9075 config-stream and cleanup ordering is invalid"
            )

        def release_journal_command(
            value: object,
            *,
            status: str,
            label: str,
        ) -> dict[str, Any]:
            item = exact(
                value,
                {"commandId", "sequence", "type", "status", "issuedAt"},
                label,
            )
            try:
                command_id = str(uuid.UUID(str(item.get("commandId") or "")))
            except ValueError as exc:
                raise ReadinessError(
                    f"IQ9075 config-stream {label} identity is invalid"
                ) from exc
            if (
                command_id != item.get("commandId")
                or type(item.get("sequence")) is not int
                or item["sequence"] < 1
                or item.get("type") != "AGENT_UPDATE"
                or item.get("status") != status
            ):
                raise ReadinessError(
                    f"IQ9075 config-stream {label} proof is invalid"
                )
            return item

        release_command = release_journal_command(
            document.get("releaseCommand"),
            status="SUCCEEDED",
            label="committed release command",
        )
        prior_rollback_command = release_journal_command(
            document.get("priorRollbackCommand"),
            status="ROLLED_BACK",
            label="prior rollback command",
        )
        commit_scenario = fleet_manifest.get("scenario")
        commit_updater = fleet_evidence.get("updater")
        commit_update = (
            commit_updater.get("update")
            if isinstance(commit_updater, dict)
            else None
        )
        rollback_scenario = rollback_manifest.get("scenario")
        rollback_updater = rollback_evidence.get("updater")
        rollback_update = (
            rollback_updater.get("update")
            if isinstance(rollback_updater, dict)
            else None
        )
        release_issued = _timestamp(
            release_command.get("issuedAt"),
            label="IQ9075 committed release command issue",
        )
        prior_rollback_issued = _timestamp(
            prior_rollback_command.get("issuedAt"),
            label="IQ9075 rollback release command issue",
        )
        rollback_updated = _timestamp(
            rollback_update.get("updatedAt")
            if isinstance(rollback_update, dict)
            else None,
            label="IQ9075 rollback update completion",
        )
        rollback_generated = _timestamp(
            rollback_evidence.get("generatedAt"),
            label="IQ9075 rollback result generation",
        )
        commit_updated = _timestamp(
            commit_update.get("updatedAt")
            if isinstance(commit_update, dict)
            else None,
            label="IQ9075 commit update completion",
        )
        if (
            not isinstance(commit_scenario, dict)
            or not isinstance(commit_update, dict)
            or not isinstance(rollback_scenario, dict)
            or not isinstance(rollback_update, dict)
            or release_command["commandId"]
            != commit_scenario.get("expectedCommandId")
            or release_command["commandId"] != commit_update.get("commandId")
            or release_command["sequence"] != commit_update.get("sequence")
            or prior_rollback_command["commandId"]
            != rollback_scenario.get("expectedCommandId")
            or prior_rollback_command["commandId"]
            != rollback_update.get("commandId")
            or prior_rollback_command["sequence"]
            != rollback_update.get("sequence")
            or prior_rollback_command["sequence"] + 1
            != release_command["sequence"]
            or prior_rollback_issued > rollback_updated
            or rollback_updated > rollback_generated
            or release_issued > commit_updated
            or commit_updated > fleet_generated
            or release_issued > fleet_generated
        ):
            raise ReadinessError(
                "IQ9075 config-stream release command chain is invalid"
            )

        preparation = exact(
            document.get("boardPreparation"),
            {
                "syntheticSource",
                "connectivityShim",
                "configBeforeSha256",
                "configTestSha256",
            },
            "board preparation",
        )
        if (
            preparation.get("syntheticSource") != "videotestsrc"
            or preparation.get("connectivityShim") != "scoped-iw-ping"
            or not isinstance(preparation.get("configBeforeSha256"), str)
            or SHA256.fullmatch(preparation["configBeforeSha256"]) is None
            or not isinstance(preparation.get("configTestSha256"), str)
            or SHA256.fullmatch(preparation["configTestSha256"]) is None
            or preparation["configBeforeSha256"]
            == preparation["configTestSha256"]
        ):
            raise ReadinessError(
                "IQ9075 config-stream synthetic preparation proof is invalid"
            )

        config = exact(
            document.get("config"),
            {
                "baseline",
                "changedBitrateKbps",
                "fieldCoverage",
                "apply",
                "restore",
            },
            "CONFIG_APPLY evidence",
        )
        if config.get("fieldCoverage") != {
            "model": "PRESERVED_WITHOUT_ACTIVATION",
            "labels": "PRESERVED_WITHOUT_ACTIVATION",
            "clipPolicy": "SAME_VALUE_RECONCILED",
            "video": "CHANGED_AND_RESTORED",
        }:
            raise ReadinessError(
                "IQ9075 CONFIG_APPLY field coverage claim is invalid"
            )
        baseline_settings = baseline(config.get("baseline"))
        changed_bitrate = config.get("changedBitrateKbps")
        if (
            type(changed_bitrate) is not int
            or changed_bitrate < 1
            or changed_bitrate == baseline_settings["video"]["bitrateKbps"]
        ):
            raise ReadinessError(
                "IQ9075 CONFIG_APPLY video delta is invalid"
            )
        apply = command(config.get("apply"), "CONFIG_APPLY")
        restore = command(config.get("restore"), "CONFIG_APPLY")
        apply_state = apply["reportedState"]
        restore_state = restore["reportedState"]
        apply_payload = {
            "configVersion": apply_state.get("configVersion"),
            "activation": "IMMEDIATE",
            "clip": baseline_settings["clip"],
            "video": {
                **baseline_settings["video"],
                "bitrateKbps": changed_bitrate,
            },
        }
        restore_payload = {
            "configVersion": restore_state.get("configVersion"),
            "activation": "IMMEDIATE",
            "clip": baseline_settings["clip"],
            "video": baseline_settings["video"],
        }
        if (
            type(apply_state.get("configVersion")) is not int
            or apply_state["configVersion"] < 1
            or restore_state.get("configVersion")
            != apply_state["configVersion"] + 1
            or any(apply_state.get(key) != value for key, value in apply_payload.items())
            or any(
                restore_state.get(key) != value
                for key, value in restore_payload.items()
            )
            or apply_state.get("settingsDigest")
            != config_digest(apply_payload)
            or restore_state.get("settingsDigest")
            != config_digest(restore_payload)
            or apply_state.get("health") != "FUNCTIONAL_HEALTHY"
            or restore_state.get("health") != "FUNCTIONAL_HEALTHY"
            or apply_state.get("configSchema") != release.get("configSchema")
            or restore_state.get("configSchema") != release.get("configSchema")
            or apply["boardSettings"]
            != {
                **baseline_settings,
                "video": {
                    **baseline_settings["video"],
                    "bitrateKbps": changed_bitrate,
                },
            }
            or restore["boardSettings"] != baseline_settings
        ):
            raise ReadinessError(
                "IQ9075 CONFIG_APPLY preservation or convergence proof is invalid"
            )

        stream = exact(
            document.get("stream"),
            {
                "adaptiveCommand",
                "initialGood",
                "poor",
                "recoveredGood",
                "disabled",
            },
            "adaptive streaming evidence",
        )
        adaptive = exact(
            stream.get("adaptiveCommand"),
            {
                "commandId",
                "sequence",
                "lifecycleAckStatuses",
                "effectPhase",
            },
            "adaptive command",
        )
        try:
            adaptive_id = str(uuid.UUID(str(adaptive.get("commandId") or "")))
        except ValueError as exc:
            raise ReadinessError(
                "IQ9075 adaptive command identity is invalid"
            ) from exc
        initial = exact(
            stream.get("initialGood"),
            {
                "commandId",
                "sequence",
                "policyRevision",
                "appliedBitrateKbps",
                "health",
                "encoder",
                "lastAdjustmentReason",
                "projectionShape",
                "queue",
            },
            "initial GOOD observation",
        )

        def adaptation(value: object, label: str) -> dict[str, Any]:
            item = exact(
                value,
                {
                    "commandId",
                    "sequence",
                    "policyRevision",
                    "appliedBitrateKbps",
                    "lastAdjustmentReason",
                    "health",
                    "encoder",
                    "projectionShape",
                    "queue",
                },
                label,
            )
            if (
                type(item.get("policyRevision")) is not int
                or item["policyRevision"] < 1
                or type(item.get("appliedBitrateKbps")) is not int
                or item["appliedBitrateKbps"] < 1
                or not isinstance(item.get("lastAdjustmentReason"), str)
                or not item["lastAdjustmentReason"]
                or item.get("health") != "STREAM_CONTINUOUS"
                or item.get("encoder") != "x264enc"
                or item.get("projectionShape")
                != document.get("projectionShape")
                or item.get("commandId") != adaptive.get("commandId")
                or item.get("sequence") != adaptive.get("sequence")
            ):
                raise ReadinessError(
                    f"IQ9075 config-stream {label} is invalid"
                )
            queue_drained(item.get("queue"))
            return item

        poor = adaptation(stream.get("poor"), "POOR observation")
        recovered = adaptation(
            stream.get("recoveredGood"), "recovered GOOD observation"
        )
        disabled = command(stream.get("disabled"), "STREAM_POLICY")
        poor_reason_tokens = poor["lastAdjustmentReason"].split(",")
        if (
            adaptive_id != adaptive.get("commandId")
            or type(adaptive.get("sequence")) is not int
            or adaptive["sequence"] < 1
            or adaptive.get("lifecycleAckStatuses")
            != ["RECEIVED", "IN_PROGRESS", "SUCCEEDED"]
            or adaptive.get("effectPhase") != "APPLIED"
            or type(initial.get("policyRevision")) is not int
            or initial["policyRevision"] < 1
            or type(initial.get("appliedBitrateKbps")) is not int
            or initial["appliedBitrateKbps"] < 1
            or initial.get("health") != "STREAM_CONTINUOUS"
            or initial.get("encoder") != "x264enc"
            or initial.get("lastAdjustmentReason") != "policy_activated"
            or initial.get("commandId") != adaptive.get("commandId")
            or initial.get("sequence") != adaptive.get("sequence")
            or initial.get("projectionShape")
            != document.get("projectionShape")
            or not (
                initial["policyRevision"]
                < poor["policyRevision"]
                < recovered["policyRevision"]
            )
            or not (
                poor["appliedBitrateKbps"] < initial["appliedBitrateKbps"]
                and recovered["appliedBitrateKbps"]
                > poor["appliedBitrateKbps"]
            )
            or any(not token or token != token.strip() for token in poor_reason_tokens)
            or "connectivity_poor" not in poor_reason_tokens
            or recovered["lastAdjustmentReason"] != "healthy_recovery"
            or disabled["reportedState"].get("mode") != "DISABLED"
            or disabled["reportedState"].get("encoder") != "x264enc"
            or disabled["reportedState"].get("lastAdjustmentReason")
            != "policy_disabled"
            or disabled["reportedState"].get("health")
            != "STREAM_CONTINUOUS"
            or disabled["boardSettings"] != baseline_settings
        ):
            raise ReadinessError(
                "IQ9075 adaptive streaming convergence proof is invalid"
            )
        queue_drained(initial.get("queue"))

        anti_replay = fleet_evidence.get("antiReplay")
        maximum_sequence = (
            anti_replay.get("maximumCommandSequence")
            if isinstance(anti_replay, dict)
            else None
        )
        sequences = [
            apply["sequence"],
            restore["sequence"],
            adaptive["sequence"],
            disabled["sequence"],
        ]
        command_ids = [
            apply["commandId"],
            restore["commandId"],
            adaptive["commandId"],
            disabled["commandId"],
        ]
        if (
            type(maximum_sequence) is not int
            or maximum_sequence < 1
            or sequences
            != list(
                range(
                    release_command["sequence"] + 1,
                    release_command["sequence"] + 5,
                )
            )
            or len(set(command_ids)) != len(command_ids)
            or maximum_sequence != release_command["sequence"]
        ):
            raise ReadinessError(
                "IQ9075 config-stream command sequences are stale or reordered"
            )

        predecessors = document.get("expiredPredecessors")
        predecessor_sequences: list[int] = []
        if not isinstance(predecessors, list) or not predecessors:
            raise ReadinessError(
                "IQ9075 expired predecessor journal is invalid"
            )
        for predecessor in predecessors:
            item = exact(
                predecessor,
                {"commandId", "sequence", "type", "status", "expiresAt"},
                "expired predecessor",
            )
            try:
                predecessor_id = str(
                    uuid.UUID(str(item.get("commandId") or ""))
                )
            except ValueError as exc:
                raise ReadinessError(
                    "IQ9075 expired predecessor identity is invalid"
                ) from exc
            if (
                predecessor_id != item.get("commandId")
                or type(item.get("sequence")) is not int
                or item["sequence"] < 1
                or item.get("type")
                not in {"AGENT_UPDATE", "CONFIG_APPLY", "STREAM_POLICY"}
                or item.get("status") != "EXPIRED"
                or _timestamp(
                    item.get("expiresAt"),
                    label="IQ9075 expired predecessor expiry",
                )
                > config_generated
            ):
                raise ReadinessError(
                    "IQ9075 expired predecessor proof is invalid"
                )
            predecessor_sequences.append(item["sequence"])
        if (
            predecessor_sequences != sorted(set(predecessor_sequences))
            or any(
                sequence >= prior_rollback_command["sequence"]
                for sequence in predecessor_sequences
            )
            or len(
                {
                    release_command["commandId"],
                    prior_rollback_command["commandId"],
                    *(
                        str(item.get("commandId"))
                        for item in predecessors
                        if isinstance(item, dict)
                    ),
                }
            )
            != len(predecessors) + 2
        ):
            raise ReadinessError(
                "IQ9075 expired predecessor ordering is invalid"
            )

        expected_gates = {
            "releaseBound",
            "cameraIndependent",
            "modelConfigurationPreservedWithoutActivation",
            "labelConfigurationPreservedWithoutActivation",
            "clipPolicyReconciled",
            "videoChangedAndRestored",
            "ackReceivedToApplied",
            "twinsConverged",
            "adaptiveClosedLoop",
            "commandQueuesDrained",
            "encoderStartupBaselineRestored",
            "exactBoardRestoration",
        }
        gates = exact(document.get("gates"), expected_gates, "gate set")
        if any(gates.get(name) is not True for name in expected_gates):
            raise ReadinessError("IQ9075 config-stream gates are incomplete")

        cleanup = exact(
            document.get("cleanup"),
            {
                "schemaVersion",
                "runId",
                "completedAt",
                "restored",
                "idempotent",
                "noMutation",
                "exactRestoration",
                "runtimeRestarted",
                "configSha256",
                "settings",
                "settingsSha256",
                "encoderStartupBitrateKbps",
                "runtimeIdentity",
                "exclusiveLeaseReleased",
                "deadmanDisarmed",
            },
            "exact restoration",
        )
        cleanup_settings = baseline(cleanup.get("settings"))
        cleanup_completed = _timestamp(
            cleanup.get("completedAt"),
            label="IQ9075 config-stream cleanup completion",
        )
        cleanup_settings_sha = cleanup.get("settingsSha256")
        restored_runtime_identity = runtime_identity(
            cleanup.get("runtimeIdentity"),
            release=release,
            bom_digest=scenario.get("expectedBomDigest"),
        )
        initial_identity_without_pid = {
            key: value
            for key, value in initial_runtime_identity.items()
            if key != "servicePid"
        }
        restored_identity_without_pid = {
            key: value
            for key, value in restored_runtime_identity.items()
            if key != "servicePid"
        }
        if (
            type(cleanup.get("schemaVersion")) is not int
            or cleanup.get("schemaVersion") != 1
            or cleanup.get("runId") != document["runId"]
            or cleanup.get("restored") is not True
            or type(cleanup.get("idempotent")) is not bool
            or cleanup.get("noMutation") is not False
            or cleanup.get("exactRestoration") is not True
            or cleanup.get("runtimeRestarted") is not True
            or cleanup.get("exclusiveLeaseReleased") is not True
            or cleanup.get("deadmanDisarmed") is not True
            or cleanup.get("configSha256")
            != preparation["configBeforeSha256"]
            or cleanup_settings != baseline_settings
            or not isinstance(cleanup_settings_sha, str)
            or hashlib.sha256(
                canonical_config(cleanup_settings)
            ).hexdigest()
            != cleanup_settings_sha
            or cleanup.get("encoderStartupBitrateKbps")
            != baseline_settings["video"]["bitrateKbps"]
            or restored_identity_without_pid != initial_identity_without_pid
            or restored_runtime_identity["servicePid"]
            == initial_runtime_identity["servicePid"]
            or not config_generated
            <= cleanup_completed
            <= fleet_cleanup_completed
        ):
            raise ReadinessError(
                "IQ9075 config-stream exact restoration proof is invalid"
            )

        model = baseline_settings["model"]
        expected_model_qualification = (
            {
                "status": "ARTIFACT_IDENTITY_VERIFIED",
                "artifactDigest": model["artifactDigest"],
            }
            if model["artifactVerified"] is True
            else (
                {
                    "status": "NOT_APPLICABLE_BACKEND_DISABLED",
                    "artifactDigest": None,
                }
                if model["runtimeEnabled"] is False
                or model["runtimeBackend"] == "none"
                else {"status": "NOT_VERIFIED", "artifactDigest": None}
            )
        )
        if document.get("modelQualification") != expected_model_qualification:
            raise ReadinessError(
                "IQ9075 model qualification claim is invalid"
            )

        gate = {
            "runId": document["runId"],
            "generatedAt": document["generatedAt"],
            "identity": identity,
            "source": source,
            "releaseCommand": release_command,
            "priorRollbackCommand": prior_rollback_command,
            "projectionShape": document["projectionShape"],
            "configSequences": [apply["sequence"], restore["sequence"]],
            "streamSequences": [adaptive["sequence"], disabled["sequence"]],
            "adaptiveRevisions": [
                initial["policyRevision"],
                poor["policyRevision"],
                recovered["policyRevision"],
            ],
            "gates": gates,
            "modelQualification": expected_model_qualification,
        }
    except Exception as exc:
        raise ReadinessError("IQ9075 config-stream evidence is invalid") from exc
    if not isinstance(gate, dict):
        raise ReadinessError("IQ9075 config-stream gate is invalid")
    return gate, runner_sha256


def _validate_fleet_runtime_documents(
    *,
    policy_path: Path,
    version: str,
    component_sha: str,
    summary: dict[str, Any],
    security: dict[str, Any],
    candidate_fleet_runner: Path,
    candidate_config_stream_runner: Path,
    candidate_board_tool: Path,
) -> dict[str, str]:
    """Validate two-run rollback and committed-component Fleet evidence."""

    expected_summary_fields = {
        "schemaVersion",
        "kind",
        "agentVersion",
        "componentSha",
        "fleetRunnerSha256",
        "configStreamRunnerSha256",
        "boardToolSha256",
        "rollbackManifest",
        "rollbackEvidence",
        "rollbackCleanupEvidence",
        "commitManifest",
        "commitEvidence",
        "configStreamEvidence",
        "commitCleanupEvidence",
        "testedArtifact",
        "testedBom",
        "runtimeGate",
    }
    if set(summary) != expected_summary_fields or (
        type(summary.get("schemaVersion")) is not int
        or summary.get("schemaVersion") != 3
        or summary.get("kind")
        != "nuvion-iq9075-fleet-runtime-release-evidence"
        or summary.get("agentVersion") != version
        or summary.get("componentSha") != component_sha
        or not isinstance(summary.get("fleetRunnerSha256"), str)
        or not SHA256.fullmatch(summary["fleetRunnerSha256"])
        or not isinstance(summary.get("configStreamRunnerSha256"), str)
        or not SHA256.fullmatch(summary["configStreamRunnerSha256"])
        or not isinstance(summary.get("boardToolSha256"), str)
        or not SHA256.fullmatch(summary["boardToolSha256"])
    ):
        raise ReadinessError(
            "IQ9075 Fleet Runtime evidence does not match two-run schema v3"
        )

    rollback_manifest_raw, _rollback_manifest_sha256 = _evidence_reference(
        policy_path.parent,
        summary.get("rollbackManifest"),
        label="IQ9075 rollback manifest",
    )
    rollback_evidence_raw, _rollback_evidence_sha256 = _evidence_reference(
        policy_path.parent,
        summary.get("rollbackEvidence"),
        label="IQ9075 rollback result",
    )
    rollback_cleanup_raw, _rollback_cleanup_sha256 = _evidence_reference(
        policy_path.parent,
        summary.get("rollbackCleanupEvidence"),
        label="IQ9075 rollback cleanup evidence",
    )
    commit_manifest_raw, _commit_manifest_sha256 = _evidence_reference(
        policy_path.parent,
        summary.get("commitManifest"),
        label="IQ9075 commit manifest",
    )
    commit_evidence_raw, _commit_evidence_sha256 = _evidence_reference(
        policy_path.parent,
        summary.get("commitEvidence"),
        label="IQ9075 commit result",
    )
    config_stream_raw, _config_stream_sha256 = _evidence_reference(
        policy_path.parent,
        summary.get("configStreamEvidence"),
        label="IQ9075 config-stream evidence",
    )
    commit_cleanup_raw, _commit_cleanup_sha256 = _evidence_reference(
        policy_path.parent,
        summary.get("commitCleanupEvidence"),
        label="IQ9075 commit cleanup evidence",
    )
    bom_raw, tested_bom_sha256 = _evidence_reference(
        policy_path.parent,
        summary.get("testedBom"),
        label="IQ9075 Fleet Runtime tested BOM",
    )

    rollback_manifest = _strict_canonical_object(
        rollback_manifest_raw, label="IQ9075 rollback manifest"
    )
    rollback_evidence = _strict_canonical_object(
        rollback_evidence_raw, label="IQ9075 rollback result"
    )
    rollback_cleanup = _strict_canonical_object(
        rollback_cleanup_raw, label="IQ9075 rollback cleanup evidence"
    )
    commit_manifest = _strict_canonical_object(
        commit_manifest_raw, label="IQ9075 commit manifest"
    )
    commit_evidence = _strict_canonical_object(
        commit_evidence_raw, label="IQ9075 commit result"
    )
    config_stream_evidence = _strict_canonical_object(
        config_stream_raw, label="IQ9075 config-stream evidence"
    )
    commit_cleanup = _strict_canonical_object(
        commit_cleanup_raw, label="IQ9075 commit cleanup evidence"
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
        or rollback_manifest.get("toolSha256") != board_tool_sha256
        or commit_manifest.get("toolSha256") != board_tool_sha256
    ):
        raise ReadinessError(
            "IQ9075 board tool differs from either Fleet Runtime manifest"
        )

    module_name = "_nuvion_trusted_iq9075_fleet_runtime_validator"
    fleet_validator = _module_from_verified_source(
        module_name=module_name,
        source=publisher_fleet_runner_raw,
        display_path=publisher_fleet_runner,
    )
    try:
        validated_rollback_manifest = fleet_validator.validate_manifest(
            rollback_manifest
        )
        validated_commit_manifest = fleet_validator.validate_manifest(
            commit_manifest
        )
        fleet_validator.validate_final_evidence(
            rollback_evidence, validated_rollback_manifest
        )
        fleet_validator.validate_final_evidence(
            commit_evidence, validated_commit_manifest
        )
        canonical_rollback_cleanup = (
            fleet_validator.validate_bound_cleanup_evidence(
                rollback_cleanup,
                run_id=str(validated_rollback_manifest["runId"]),
                manifest_raw=rollback_manifest_raw,
                fleet_evidence_raw=rollback_evidence_raw,
            )
        )
        canonical_commit_cleanup = fleet_validator.validate_bound_cleanup_evidence(
            commit_cleanup,
            run_id=str(validated_commit_manifest["runId"]),
            manifest_raw=commit_manifest_raw,
            fleet_evidence_raw=commit_evidence_raw,
        )
    except Exception as exc:
        raise ReadinessError(
            "IQ9075 two-run Fleet Runtime evidence is invalid"
        ) from exc
    finally:
        sys.modules.pop(module_name, None)

    for label, evidence, cleanup, canonical_cleanup in (
        (
            "rollback",
            rollback_evidence,
            rollback_cleanup,
            canonical_rollback_cleanup,
        ),
        ("commit", commit_evidence, commit_cleanup, canonical_commit_cleanup),
    ):
        if (
            evidence.get("schemaVersion") != 2
            or not isinstance(evidence.get("antiReplay"), dict)
        ):
            raise ReadinessError(
                f"IQ9075 {label} evidence lacks persisted anti-replay proof"
            )
        if (
            cleanup != canonical_cleanup
            or cleanup.get("schemaVersion") != 2
            or cleanup.get("phase") != "RESTORED"
            or cleanup.get("proof", {}).get("transactionPhase") != "RESTORED"
        ):
            raise ReadinessError(
                f"IQ9075 {label} cleanup did not restore the transaction"
            )

    rollback_scenario = rollback_manifest.get("scenario")
    commit_scenario = commit_manifest.get("scenario")
    rollback_updater = rollback_evidence.get("updater")
    commit_updater = commit_evidence.get("updater")
    rollback_update = (
        rollback_updater.get("update")
        if isinstance(rollback_updater, dict)
        else None
    )
    commit_update = (
        commit_updater.get("update") if isinstance(commit_updater, dict) else None
    )
    if (
        not isinstance(rollback_scenario, dict)
        or rollback_scenario.get("type") != "oak-fault-rollback"
        or rollback_evidence.get("scenario") != "oak-fault-rollback"
        or not isinstance(rollback_update, dict)
        or rollback_update.get("phase") != "ROLLED_BACK"
        or rollback_update.get("updatePhase") != "ROLLED_BACK"
        or rollback_update.get("errorCode") != "ROLLED_BACK"
        or rollback_update.get("health") != "LKG_RESTORED"
        or rollback_update.get("functionalHealth") != "FUNCTIONAL_UNHEALTHY"
        or "healthDeadline" in rollback_update
    ):
        raise ReadinessError(
            "IQ9075 rollback run does not prove terminal rollback"
        )
    if (
        not isinstance(commit_scenario, dict)
        or commit_scenario.get("type") != "commit"
        or commit_evidence.get("scenario") != "commit"
        or not isinstance(commit_update, dict)
        or commit_update.get("phase") != "COMMITTED"
        or commit_update.get("updatePhase") != "COMMITTED"
        or commit_update.get("health") != "FUNCTIONAL_HEALTHY"
        or commit_update.get("functionalHealth") != "FUNCTIONAL_HEALTHY"
        or "errorCode" in commit_update
        or "rollbackSlot" in commit_update
        or "rollbackVersion" in commit_update
        or "healthDeadline" in commit_update
    ):
        raise ReadinessError(
            "IQ9075 commit run does not prove candidate A is active"
        )

    rollback_identity = rollback_manifest.get("identity")
    commit_identity = commit_manifest.get("identity")
    rollback_inputs = rollback_manifest.get("inputs")
    commit_inputs = commit_manifest.get("inputs")
    rollback_sequence = rollback_update.get("sequence")
    commit_sequence = commit_update.get("sequence")
    if (
        rollback_manifest.get("runId") == commit_manifest.get("runId")
        or rollback_identity != commit_identity
        or rollback_inputs != commit_inputs
        or rollback_scenario.get("release") != commit_scenario.get("release")
        or any(
            rollback_scenario.get(name) != commit_scenario.get(name)
            for name in (
                "expectedBomDigest",
                "expectedCandidateSlot",
                "expectedPreviousSlot",
                "expectedPreviousVersion",
            )
        )
        or rollback_scenario.get("expectedCommandId")
        == commit_scenario.get("expectedCommandId")
        or type(rollback_sequence) is not int
        or type(commit_sequence) is not int
        or commit_sequence <= rollback_sequence
    ):
        raise ReadinessError(
            "IQ9075 rollback and commit runs do not share one advancing release"
        )

    rollback_cleanup_completed = _timestamp(
        rollback_cleanup.get("completedAt"),
        label="IQ9075 rollback cleanup completion",
    )
    commit_generated = _timestamp(
        commit_evidence.get("generatedAt"),
        label="IQ9075 commit result generation",
    )
    if rollback_cleanup_completed > commit_generated:
        raise ReadinessError(
            "IQ9075 commit run precedes rollback cleanup"
        )

    config_stream_gate, config_stream_runner_sha256 = (
        _validated_config_stream_gate(
            config_stream_evidence=config_stream_evidence,
            fleet_manifest=commit_manifest,
            fleet_manifest_raw=commit_manifest_raw,
            fleet_evidence=commit_evidence,
            fleet_evidence_raw=commit_evidence_raw,
            cleanup_evidence=commit_cleanup,
            rollback_manifest=rollback_manifest,
            rollback_evidence=rollback_evidence,
            candidate_config_stream_runner=candidate_config_stream_runner,
        )
    )
    committed_release_issued = _timestamp(
        config_stream_gate["releaseCommand"].get("issuedAt"),
        label="IQ9075 committed release command issue",
    )
    if rollback_cleanup_completed > committed_release_issued:
        raise ReadinessError(
            "IQ9075 committed release command precedes rollback cleanup"
        )
    if (
        summary.get("configStreamRunnerSha256")
        != config_stream_runner_sha256
    ):
        raise ReadinessError(
            "IQ9075 config-stream runner differs from signed evidence"
        )

    try:
        verified_bom = verify_release_bom(bom)
    except ReleaseBomValidationError as exc:
        raise ReadinessError(
            "IQ9075 Fleet Runtime tested BOM is invalid"
        ) from exc
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
    baseline_policy = (
        iq_policy.get("legacyPromotedBaseline")
        if isinstance(iq_policy, dict)
        else None
    )
    if (
        not isinstance(iq_policy, dict)
        or not isinstance(target_policy, dict)
        or set(target_policy)
        != {"productModel", "platformProfile", "hardwareRevision", "architecture"}
        or not isinstance(baseline_policy, dict)
        or set(baseline_policy) != {"agentVersion", "releaseSequence", "bomDigest"}
        or not isinstance(baseline_policy.get("agentVersion"), str)
        or not SEMVER.fullmatch(baseline_policy["agentVersion"])
        or type(baseline_policy.get("releaseSequence")) is not int
        or baseline_policy["releaseSequence"] < 1
        or not isinstance(baseline_policy.get("bomDigest"), str)
        or not baseline_policy["bomDigest"].startswith("sha256:")
        or not SHA256.fullmatch(baseline_policy["bomDigest"][7:])
        or not isinstance(iq_policy.get("publicKeyringSha256"), str)
        or not SHA256.fullmatch(iq_policy["publicKeyringSha256"])
        or not isinstance(iq_policy.get("publisherKeyId"), str)
    ):
        raise ReadinessError("IQ9075 Fleet Runtime security policy is invalid")

    expected_bom_digest = "sha256:" + verified_bom.bom_digest
    expected_candidate_slot = "/opt/nuv-agent/releases/" + verified_bom.bom_digest
    expected_relative_candidate_slot = "releases/" + verified_bom.bom_digest
    expected_previous_slot = "releases/" + baseline_policy["bomDigest"][7:]
    expected_release = {
        "agentVersion": version,
        "releaseSequence": verified_bom.release_sequence,
        "artifactDigest": "sha256:" + verified_bom.artifact_sha256,
        "componentSha": component_sha,
        "configSchema": verified_bom.config_schema,
        "publisherKeyId": iq_policy["publisherKeyId"],
    }
    if (
        verified_bom.schema_version != 2
        or verified_bom.agent_version != version
        or verified_bom.component_sha != component_sha
        or verified_bom.release_sequence is None
        or verified_bom.release_sequence <= baseline_policy["releaseSequence"]
        or verified_bom.min_updater_version != "0.2.0"
        or verified_bom.artifact_kind != "agent-bundle"
        or verified_bom.artifact_name != tested_artifact["name"]
        or verified_bom.artifact_sha256 != tested_artifact["sha256"]
        or verified_bom.artifact_size_bytes != tested_artifact["sizeBytes"]
        or len(verified_bom.targets) != 1
        or verified_bom.targets[0].to_payload() != target_policy
    ):
        raise ReadinessError(
            "IQ9075 Fleet Runtime artifact, BOM, and policy differ"
        )

    for (
        label,
        manifest,
        evidence,
        scenario,
        baseline_marker_name,
        candidate_marker_name,
    ) in (
        (
            "rollback",
            rollback_manifest,
            rollback_evidence,
            rollback_scenario,
            "release",
            "previousRelease",
        ),
        (
            "commit",
            commit_manifest,
            commit_evidence,
            commit_scenario,
            "previousRelease",
            "release",
        ),
    ):
        identity = manifest.get("identity")
        inputs = manifest.get("inputs")
        slots = evidence.get("slots")
        baseline_marker = (
            slots.get(baseline_marker_name) if isinstance(slots, dict) else None
        )
        candidate_marker = (
            slots.get(candidate_marker_name) if isinstance(slots, dict) else None
        )
        if (
            not isinstance(inputs, dict)
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
            or scenario.get("expectedBomDigest") != expected_bom_digest
            or scenario.get("expectedCandidateSlot") != expected_candidate_slot
            or scenario.get("expectedPreviousSlot") != expected_previous_slot
            or scenario.get("expectedPreviousVersion")
            != baseline_policy["agentVersion"]
            or scenario.get("release") != expected_release
            or not isinstance(baseline_marker, dict)
            or baseline_marker.get("agentVersion")
            != baseline_policy["agentVersion"]
            or baseline_marker.get("releaseSequence")
            != baseline_policy["releaseSequence"]
            or baseline_marker.get("bomDigest") != baseline_policy["bomDigest"]
            or not isinstance(candidate_marker, dict)
            or candidate_marker
            != {
                "schemaVersion": 2,
                "bomDigest": expected_bom_digest,
                **expected_release,
            }
            or not isinstance(slots, dict)
            or (
                label == "rollback"
                and (
                    slots.get("current") != expected_previous_slot
                    or slots.get("previous") != expected_relative_candidate_slot
                    or slots.get("currentVersion")
                    != baseline_policy["agentVersion"]
                )
            )
            or (
                label == "commit"
                and (
                    slots.get("current") != expected_relative_candidate_slot
                    or slots.get("previous") != expected_previous_slot
                    or slots.get("currentVersion") != version
                )
            )
        ):
            raise ReadinessError(
                f"IQ9075 {label} run differs from artifact/BOM/release identity"
            )

    if (
        rollback_update.get("bomDigest") != expected_bom_digest
        or commit_update.get("bomDigest") != expected_bom_digest
        or rollback_update.get("artifactDigest")
        != expected_release["artifactDigest"]
        or commit_update.get("artifactDigest") != expected_release["artifactDigest"]
        or rollback_update.get("componentSha") != component_sha
        or commit_update.get("componentSha") != component_sha
        or rollback_update.get("releaseSequence")
        != verified_bom.release_sequence
        or commit_update.get("releaseSequence") != verified_bom.release_sequence
        or rollback_update.get("configSchema") != verified_bom.config_schema
        or commit_update.get("configSchema") != verified_bom.config_schema
        or rollback_update.get("publisherKeyId") != iq_policy["publisherKeyId"]
        or commit_update.get("publisherKeyId") != iq_policy["publisherKeyId"]
        or commit_update.get("slot") != expected_relative_candidate_slot
    ):
        raise ReadinessError(
            "IQ9075 rollback/commit updater release identity differs"
        )

    expected_runtime_gate = {
        "rollback": _fleet_runtime_gate(
            rollback_evidence, rollback_cleanup, rollback_manifest
        ),
        "commit": _fleet_runtime_gate(
            commit_evidence, commit_cleanup, commit_manifest
        ),
        "configStream": config_stream_gate,
    }
    if summary.get("runtimeGate") != expected_runtime_gate:
        raise ReadinessError(
            "IQ9075 Fleet Runtime gate summary differs from raw evidence"
        )
    return {
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
    candidate_config_stream_runner: Path | None,
    candidate_board_tool: Path | None,
) -> dict[str, str]:
    common_fields = {"componentSha", "agentReleaseGate"}
    runtime_fields = common_fields | {"iq9075FleetRuntime"}
    legacy_fields = common_fields | {"iq9075Physical"}
    try:
        version_tuple = tuple(int(part) for part in version.split("."))
    except ValueError as exc:
        raise ReadinessError("release version is invalid") from exc
    if (
        not isinstance(decision, dict)
        or frozenset(decision)
        not in {frozenset(runtime_fields), frozenset(legacy_fields)}
        or component_sha is None
        or not SHA.fullmatch(component_sha)
        or decision.get("componentSha") != component_sha
        or gate_evidence is None
        or (
            version_tuple >= FLEET_RUNTIME_REQUIRED_FROM
            and frozenset(decision) != frozenset(runtime_fields)
        )
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
            and (
                candidate_fleet_runner is None
                or candidate_config_stream_runner is None
                or candidate_board_tool is None
            )
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
        assert candidate_config_stream_runner is not None
        assert candidate_board_tool is not None
        return _validate_fleet_runtime_documents(
            policy_path=policy_path,
            version=version,
            component_sha=component_sha,
            summary=document,
            security=security,
            candidate_fleet_runner=candidate_fleet_runner,
            candidate_config_stream_runner=candidate_config_stream_runner,
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
    candidate_config_stream_runner: Path | None = None,
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
        candidate_config_stream_runner=candidate_config_stream_runner,
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
    parser.add_argument("--candidate-config-stream-runner", type=Path)
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
            candidate_config_stream_runner=(
                arguments.candidate_config_stream_runner
            ),
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
