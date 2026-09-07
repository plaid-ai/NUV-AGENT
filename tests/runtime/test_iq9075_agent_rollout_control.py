from __future__ import annotations

import hashlib
import http.cookiejar
import importlib.util
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "packaging/dev/run-iq9075-agent-rollout-control.py"
SPEC = importlib.util.spec_from_file_location("iq9075_agent_rollout_control", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

SPACE_ID = 33
DEVICE_ID = "sp-33-nuvion-iq9075"
RELEASE_ID = "10000000-0000-4000-8000-000000000001"
ROLLBACK_ROLLOUT_ID = "20000000-0000-4000-8000-000000000001"
COMMIT_ROLLOUT_ID = "20000000-0000-4000-8000-000000000002"
ROLLBACK_CLIENT_REQUEST_ID = "40000000-0000-4000-8000-000000000001"
COMMIT_CLIENT_REQUEST_ID = "40000000-0000-4000-8000-000000000002"
ROLLBACK_COMMAND_ID = "30000000-0000-4000-8000-000000000011"
COMMIT_COMMAND_ID = "30000000-0000-4000-8000-000000000012"
BASELINE_SLOT = "bootstrap/0.1.120"


def _canonical(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _bom() -> dict:
    return {
        "schemaVersion": 2,
        "bomId": "nuv-agent-0.1.121-iq9075-aarch64",
        "bomDigest": "sha256:" + "a" * 64,
        "releaseSequence": 2,
        "agentVersion": "0.1.121",
        "componentSha": "c" * 40,
        "configSchema": "12",
        "minUpdaterVersion": "0.2.0",
        "targets": [
            {
                "productModel": "IQ9075_DEV",
                "platformProfile": "iq9075_dev",
                "hardwareRevision": "iq9075-dev",
                "architecture": "aarch64",
            }
        ],
        "artifact": {
            "name": "nuv-agent_0.1.121_iq9075-aarch64.tar.gz",
            "kind": "agent-bundle",
            "sha256": "d" * 64,
            "sizeBytes": 1234,
        },
        "builtAt": "2026-09-04T00:00:00Z",
    }


def _signature() -> dict:
    return {
        "schemaVersion": 1,
        "keyId": "release-iq9075-dev-2026-09-01",
        "algorithm": "Ed25519",
        "signature": "A" * 86 + "==",
    }


def _release_response() -> dict:
    bom = _bom()
    signature = _signature()
    return {
        "releaseId": RELEASE_ID,
        "spaceId": SPACE_ID,
        "schemaVersion": bom["schemaVersion"],
        "bomId": bom["bomId"],
        "bomDigest": bom["bomDigest"],
        "releaseSequence": bom["releaseSequence"],
        "agentVersion": bom["agentVersion"],
        "componentSha": bom["componentSha"],
        "configSchema": bom["configSchema"],
        "minUpdaterVersion": bom["minUpdaterVersion"],
        "targets": bom["targets"],
        "artifact": bom["artifact"],
        "builtAt": bom["builtAt"],
        "publisherKeyId": signature["keyId"],
        "createdBy": "operator@plaid.ai.kr",
        "createdAt": "2026-09-04T00:00:01Z",
        "bom": bom,
        "signature": signature,
    }


def _command(command_id: str, sequence: int, status: str) -> dict:
    return {
        "commandId": command_id,
        "deviceId": DEVICE_ID,
        "spaceId": SPACE_ID,
        "type": "AGENT_UPDATE",
        "schemaVersion": 1,
        "issuedAt": "2026-09-04T00:00:02Z",
        "expiresAt": "2026-09-04T00:30:02Z",
        "sequence": sequence,
        "payloadHash": "e" * 64,
        "actor": "operator@plaid.ai.kr",
        "authorizationContext": "SPACE_ADMIN",
        "keyId": "fleet-command-dev",
        "status": status,
        "expiredAt": None,
    }


def _reported(command_id: str, *, rollback: bool) -> dict:
    bom = _bom()
    common = {
        "targetVersion": bom["agentVersion"],
        "bomDigest": bom["bomDigest"],
        "artifactDigest": "sha256:" + bom["artifact"]["sha256"],
        "componentSha": bom["componentSha"],
        "configSchema": bom["configSchema"],
        "releaseSequence": bom["releaseSequence"],
        "publisherKeyId": _signature()["keyId"],
        "bomVerificationStatus": "VERIFIED",
        "candidateSlot": "/opt/nuv-agent/releases/" + bom["bomDigest"][7:],
        "previousVersion": "0.1.120",
    }
    if rollback:
        return {
            **common,
            "commandId": command_id,
            "phase": "ROLLED_BACK",
            "updatePhase": "ROLLED_BACK",
            "errorCode": "ROLLED_BACK",
            "health": "LKG_RESTORED",
            "functionalHealth": "FUNCTIONAL_UNHEALTHY",
            "rollbackVersion": "0.1.120",
            "previousSlot": BASELINE_SLOT,
            "rollbackSlot": BASELINE_SLOT,
            "slot": BASELINE_SLOT,
        }
    return {
        **common,
        "commandId": command_id,
        "phase": "COMMITTED",
        "agentVersion": bom["agentVersion"],
        "health": "FUNCTIONAL_HEALTHY",
        "functionalHealth": "FUNCTIONAL_HEALTHY",
        "updatePhase": "COMMITTED",
        "slot": "releases/" + bom["bomDigest"][7:],
    }


def _rollout(
    rollout_id: str,
    *,
    status: str,
    target_status: str,
    command_id: str | None,
    sequence: int,
    command_status: str | None,
    rollback: bool = False,
) -> dict:
    bom = _bom()
    terminal = target_status in {"ROLLED_BACK", "SUCCEEDED"}
    latest = (
        _command(command_id, sequence, command_status)
        if command_id is not None and command_status is not None
        else None
    )
    desired = (
        {"targetVersion": bom["agentVersion"], "bomDigest": bom["bomDigest"]}
        if latest is not None
        else None
    )
    report = _reported(command_id, rollback=rollback) if terminal else None
    target = {
        "deviceId": DEVICE_ID,
        "cohortKey": "IQ9075_DEV|iq9075_dev|iq9075-dev",
        "waveNumber": 0,
        "eligibility": "ELIGIBLE",
        "eligibilityReason": None,
        "productModel": "IQ9075_DEV",
        "platformProfile": "iq9075_dev",
        "hardwareRevision": "iq9075-dev",
        "architecture": "aarch64",
        "identitySnapshot": {
            "agentVersion": "0.1.120",
            "updaterVersion": "0.2.0",
            "agentUpdate": {
                "authenticatedHelper": True,
                "capabilityAvailable": True,
                "updaterVersion": "0.2.0",
                "activeSlot": BASELINE_SLOT,
            },
        },
        "status": target_status,
        "statusReason": "expected rollback" if rollback and terminal else None,
        "latestCommand": latest,
        "desiredEvidence": desired,
        "reportedEvidence": report if terminal and not rollback else None,
        "rollbackEvidence": report if terminal and rollback else None,
        "commandIssuedAt": "2026-09-04T00:00:02Z" if latest else None,
        "succeededAt": "2026-09-04T00:00:10Z" if target_status == "SUCCEEDED" else None,
        "terminalAt": "2026-09-04T00:00:10Z" if terminal else None,
    }
    return {
        "rolloutId": rollout_id,
        "clientRequestId": (
            ROLLBACK_CLIENT_REQUEST_ID
            if rollout_id == ROLLBACK_ROLLOUT_ID
            else COMMIT_CLIENT_REQUEST_ID
        ),
        "spaceId": SPACE_ID,
        "releaseId": RELEASE_ID,
        "bomDigest": bom["bomDigest"],
        "agentVersion": bom["agentVersion"],
        "componentSha": bom["componentSha"],
        "configSchema": bom["configSchema"],
        "releaseSequence": bom["releaseSequence"],
        "minUpdaterVersion": bom["minUpdaterVersion"],
        "artifact": bom["artifact"],
        "status": status,
        "policy": {
            "preCommitSoakSeconds": 30,
            "commandTtlSeconds": 1800,
            "maxFailurePercent": 0,
        },
        "targetCount": 1,
        "healthReason": None,
        "haltReason": "canary terminal failure" if rollback and terminal else None,
        "nextEvaluationAt": None,
        "waves": [],
        "targets": [target],
        "createdBy": "operator@plaid.ai.kr",
        "createdAt": "2026-09-04T00:00:01Z",
        "updatedAt": "2026-09-04T00:00:10Z" if terminal else "2026-09-04T00:00:02Z",
    }


class _Api:
    origin = MODULE.DEFAULT_API_ORIGIN

    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str, dict | None]] = []

    def request(self, path: str, *, method: str = "GET", payload=None):
        self.calls.append((method, path, payload))
        if not self.responses:
            raise AssertionError("unexpected API call")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class _HttpResponse:
    status = 200

    def __init__(self, url: str, body: bytes) -> None:
        self._url = url
        self._body = body
        self.headers = {"Content-Length": str(len(body))}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def geturl(self) -> str:
        return self._url

    def read(self, maximum: int) -> bytes:
        return self._body[:maximum]


class _Opener:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.requests = []

    def open(self, request, timeout: int):
        self.requests.append((request, timeout))
        return _HttpResponse(request.full_url, self.body)


class RolloutControlTest(unittest.TestCase):
    def _inputs(self, root: Path) -> tuple[Path, Path, Path]:
        root.chmod(0o700)
        bom_path = root / "release-bom.json"
        signature_path = root / "release-bom.json.sig"
        release_path = root / "release-evidence.json"
        bom_path.write_bytes(_canonical(_bom()))
        signature_path.write_bytes(_canonical(_signature()))
        bom_path.chmod(0o600)
        signature_path.chmod(0o600)
        return bom_path, signature_path, release_path

    def _register(self, root: Path) -> Path:
        bom_path, signature_path, release_path = self._inputs(root)
        api = _Api([_release_response()])
        result = MODULE.register_release(
            api,
            space_id=SPACE_ID,
            bom_path=bom_path,
            signature_path=signature_path,
            output=release_path,
        )
        self.assertEqual(result["release"]["releaseId"], RELEASE_ID)
        self.assertEqual(
            api.calls,
            [
                (
                    "POST",
                    f"/spaces/{SPACE_ID}/agent-releases",
                    {"bom": _bom(), "signature": _signature()},
                )
            ],
        )
        self.assertEqual(release_path.read_bytes(), _canonical(result))
        return release_path

    def _issue_rollback(self, root: Path, release_path: Path) -> tuple[Path, Path]:
        created_path = root / "rollback-created.json"
        issuance_path = root / "rollback-issued.json"
        draft = _rollout(
            ROLLBACK_ROLLOUT_ID,
            status="DRAFT",
            target_status="PENDING",
            command_id=None,
            sequence=11,
            command_status=None,
        )
        started = _rollout(
            ROLLBACK_ROLLOUT_ID,
            status="RUNNING",
            target_status="COMMAND_ISSUED",
            command_id=ROLLBACK_COMMAND_ID,
            sequence=11,
            command_status="QUEUED",
        )
        api = _Api([[], draft, started])
        result = MODULE.issue_rollout(
            api,
            space_id=SPACE_ID,
            device_id=DEVICE_ID,
            release_evidence_path=release_path,
            purpose="rollback",
            client_request_id=ROLLBACK_CLIENT_REQUEST_ID,
            pre_commit_soak_seconds=30,
            command_ttl_seconds=1800,
            max_failure_percent=0,
            previous_issuance_path=None,
            previous_terminal_path=None,
            created_evidence_path=created_path,
            output=issuance_path,
        )
        self.assertEqual(result["command"]["commandId"], ROLLBACK_COMMAND_ID)
        self.assertTrue(created_path.is_file())
        self.assertFalse(
            any(
                "/devices/" in path or path.endswith("/commands")
                for _, path, _ in api.calls
            )
        )
        return issuance_path, created_path

    def _write_created(
        self,
        root: Path,
        release_path: Path,
        *,
        projection: dict,
        purpose: str = "rollback",
        name: str = "created.json",
    ) -> Path:
        release_evidence, release_raw = MODULE._load_release(
            release_path,
            space_id=SPACE_ID,
            origin=MODULE.DEFAULT_API_ORIGIN,
        )
        path = root / name
        value = MODULE._created_evidence(
            api=_Api([]),
            purpose=purpose,
            space_id=SPACE_ID,
            device_id=DEVICE_ID,
            release=release_evidence["release"],
            release_raw=release_raw,
            policy={
                "preCommitSoakSeconds": 30,
                "commandTtlSeconds": 1800,
                "maxFailurePercent": 0,
            },
            client_request_id=(
                ROLLBACK_CLIENT_REQUEST_ID
                if purpose == "rollback"
                else COMMIT_CLIENT_REQUEST_ID
            ),
            projection=projection,
        )
        path.write_bytes(_canonical(value))
        path.chmod(0o600)
        return path

    def test_release_then_two_verified_rollouts_are_exact_and_adjacent(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root).resolve()
            release_path = self._register(root)
            rollback_issuance, _created = self._issue_rollback(root, release_path)
            rollback_terminal = root / "rollback-terminal.json"
            rollback_api = _Api(
                [
                    _rollout(
                        ROLLBACK_ROLLOUT_ID,
                        status="HALTED",
                        target_status="ROLLED_BACK",
                        command_id=ROLLBACK_COMMAND_ID,
                        sequence=11,
                        command_status="ROLLED_BACK",
                        rollback=True,
                    )
                ]
            )
            terminal = MODULE.wait_terminal(
                rollback_api,
                space_id=SPACE_ID,
                device_id=DEVICE_ID,
                issuance_path=rollback_issuance,
                release_evidence_path=release_path,
                output=rollback_terminal,
                wait_seconds=30,
                poll_seconds=0.5,
            )
            self.assertEqual(terminal["targetStatus"], "ROLLED_BACK")

            commit_created = root / "commit-created.json"
            commit_issuance = root / "commit-issued.json"
            draft = _rollout(
                COMMIT_ROLLOUT_ID,
                status="DRAFT",
                target_status="PENDING",
                command_id=None,
                sequence=12,
                command_status=None,
            )
            started = _rollout(
                COMMIT_ROLLOUT_ID,
                status="RUNNING",
                target_status="COMMAND_ISSUED",
                command_id=COMMIT_COMMAND_ID,
                sequence=12,
                command_status="QUEUED",
            )
            rollback_projection = _rollout(
                ROLLBACK_ROLLOUT_ID,
                status="HALTED",
                target_status="ROLLED_BACK",
                command_id=ROLLBACK_COMMAND_ID,
                sequence=11,
                command_status="ROLLED_BACK",
                rollback=True,
            )
            commit_api = _Api(
                [
                    rollback_projection,
                    [_command(ROLLBACK_COMMAND_ID, 11, "ROLLED_BACK")],
                    [],
                    draft,
                    started,
                ]
            )
            issued = MODULE.issue_rollout(
                commit_api,
                space_id=SPACE_ID,
                device_id=DEVICE_ID,
                release_evidence_path=release_path,
                purpose="commit",
                client_request_id=COMMIT_CLIENT_REQUEST_ID,
                pre_commit_soak_seconds=30,
                command_ttl_seconds=1800,
                max_failure_percent=0,
                previous_issuance_path=rollback_issuance,
                previous_terminal_path=rollback_terminal,
                created_evidence_path=commit_created,
                output=commit_issuance,
            )
            self.assertEqual(issued["command"]["sequence"], 12)

            commit_terminal = root / "commit-terminal.json"
            committed_projection = _rollout(
                COMMIT_ROLLOUT_ID,
                status="SUCCEEDED",
                target_status="SUCCEEDED",
                command_id=COMMIT_COMMAND_ID,
                sequence=12,
                command_status="SUCCEEDED",
            )
            committed = MODULE.wait_terminal(
                _Api([committed_projection]),
                space_id=SPACE_ID,
                device_id=DEVICE_ID,
                issuance_path=commit_issuance,
                release_evidence_path=release_path,
                output=commit_terminal,
                wait_seconds=30,
                poll_seconds=0.5,
            )
            self.assertEqual(committed["rolloutStatus"], "SUCCEEDED")

            # Existing immutable evidence is revalidated against live state,
            # not silently accepted from disk.
            replay = MODULE.wait_terminal(
                _Api([committed_projection]),
                space_id=SPACE_ID,
                device_id=DEVICE_ID,
                issuance_path=commit_issuance,
                release_evidence_path=release_path,
                output=commit_terminal,
                wait_seconds=30,
                poll_seconds=0.5,
            )
            self.assertEqual(replay, committed)

    def test_ambiguous_start_failure_retries_then_aborts_without_second_create(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root).resolve()
            release_path = self._register(root)
            created_path = root / "created.json"
            issuance_path = root / "issued.json"
            draft = _rollout(
                ROLLBACK_ROLLOUT_ID,
                status="DRAFT",
                target_status="PENDING",
                command_id=None,
                sequence=11,
                command_status=None,
            )
            halted = _rollout(
                ROLLBACK_ROLLOUT_ID,
                status="HALTED",
                target_status="PENDING",
                command_id=None,
                sequence=11,
                command_status=None,
            )
            failure_path = root / "abort.json"
            failure_api = _Api(
                [
                    [],
                    draft,
                    MODULE.RolloutControlError("injected start failure"),
                    draft,
                    halted,
                    halted,
                ]
            )
            with self.assertRaisesRegex(
                MODULE.RolloutControlError, "cannot reach an exact command"
            ):
                MODULE.issue_rollout(
                    failure_api,
                    space_id=SPACE_ID,
                    device_id=DEVICE_ID,
                    release_evidence_path=release_path,
                    purpose="rollback",
                    client_request_id=ROLLBACK_CLIENT_REQUEST_ID,
                    pre_commit_soak_seconds=30,
                    command_ttl_seconds=1800,
                    max_failure_percent=0,
                    previous_issuance_path=None,
                    previous_terminal_path=None,
                    created_evidence_path=created_path,
                    failure_evidence_path=failure_path,
                    output=issuance_path,
                    monotonic=lambda: 0.0,
                    sleeper=lambda _seconds: None,
                )
            self.assertTrue(created_path.is_file())
            self.assertTrue(failure_path.is_file())
            self.assertEqual(
                json.loads(failure_path.read_text())["rolloutStatus"], "HALTED"
            )
            self.assertEqual(
                sum(
                    path.endswith("/agent-rollouts") and method == "POST"
                    for method, path, _ in failure_api.calls
                ),
                1,
            )

    def test_active_duplicate_and_nonadjacent_commit_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root).resolve()
            release_path = self._register(root)
            active = _rollout(
                ROLLBACK_ROLLOUT_ID,
                status="RUNNING",
                target_status="COMMAND_ISSUED",
                command_id=ROLLBACK_COMMAND_ID,
                sequence=11,
                command_status="QUEUED",
            )
            resume_api = _Api([[active], active])
            resumed = MODULE.issue_rollout(
                resume_api,
                space_id=SPACE_ID,
                device_id=DEVICE_ID,
                release_evidence_path=release_path,
                purpose="rollback",
                client_request_id=ROLLBACK_CLIENT_REQUEST_ID,
                pre_commit_soak_seconds=30,
                command_ttl_seconds=1800,
                max_failure_percent=0,
                previous_issuance_path=None,
                previous_terminal_path=None,
                created_evidence_path=root / "duplicate-created.json",
                output=root / "duplicate-issued.json",
            )
            self.assertEqual(resumed["rolloutId"], ROLLBACK_ROLLOUT_ID)
            self.assertFalse(
                any(
                    method == "POST" and path.endswith("/agent-rollouts")
                    for method, path, _payload in resume_api.calls
                )
            )

            rollback_issuance, _created = self._issue_rollback(root, release_path)
            rollback_terminal = root / "rollback-terminal.json"
            MODULE.wait_terminal(
                _Api(
                    [
                        _rollout(
                            ROLLBACK_ROLLOUT_ID,
                            status="HALTED",
                            target_status="ROLLED_BACK",
                            command_id=ROLLBACK_COMMAND_ID,
                            sequence=11,
                            command_status="ROLLED_BACK",
                            rollback=True,
                        )
                    ]
                ),
                space_id=SPACE_ID,
                device_id=DEVICE_ID,
                issuance_path=rollback_issuance,
                release_evidence_path=release_path,
                output=rollback_terminal,
                wait_seconds=30,
                poll_seconds=0.5,
            )
            draft = _rollout(
                COMMIT_ROLLOUT_ID,
                status="DRAFT",
                target_status="PENDING",
                command_id=None,
                sequence=13,
                command_status=None,
            )
            nonadjacent = _rollout(
                COMMIT_ROLLOUT_ID,
                status="RUNNING",
                target_status="COMMAND_ISSUED",
                command_id=COMMIT_COMMAND_ID,
                sequence=13,
                command_status="QUEUED",
            )
            rollback_projection = _rollout(
                ROLLBACK_ROLLOUT_ID,
                status="HALTED",
                target_status="ROLLED_BACK",
                command_id=ROLLBACK_COMMAND_ID,
                sequence=11,
                command_status="ROLLED_BACK",
                rollback=True,
            )
            halted_nonadjacent = _rollout(
                COMMIT_ROLLOUT_ID,
                status="HALTED",
                target_status="EXPIRED",
                command_id=COMMIT_COMMAND_ID,
                sequence=13,
                command_status="EXPIRED",
            )
            nonadjacent_api = _Api(
                [
                    rollback_projection,
                    [_command(ROLLBACK_COMMAND_ID, 11, "ROLLED_BACK")],
                    [],
                    draft,
                    nonadjacent,
                    nonadjacent,
                    halted_nonadjacent,
                ]
            )
            failure_evidence = root / "nonadjacent-abort.json"
            with self.assertRaisesRegex(MODULE.RolloutControlError, "not adjacent"):
                MODULE.issue_rollout(
                    nonadjacent_api,
                    space_id=SPACE_ID,
                    device_id=DEVICE_ID,
                    release_evidence_path=release_path,
                    purpose="commit",
                    client_request_id=COMMIT_CLIENT_REQUEST_ID,
                    pre_commit_soak_seconds=30,
                    command_ttl_seconds=1800,
                    max_failure_percent=0,
                    previous_issuance_path=rollback_issuance,
                    previous_terminal_path=rollback_terminal,
                    created_evidence_path=root / "nonadjacent-created.json",
                    failure_evidence_path=failure_evidence,
                    output=root / "nonadjacent-issued.json",
                )
            self.assertTrue(failure_evidence.is_file())
            self.assertEqual(
                json.loads(failure_evidence.read_text())["rolloutStatus"], "HALTED"
            )

    def test_inventory_schema_and_other_release_conflicts_fail_before_create(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root).resolve()
            release_path = self._register(root)
            foreign = _rollout(
                "20000000-0000-4000-8000-000000000099",
                status="DRAFT",
                target_status="PENDING",
                command_id=None,
                sequence=99,
                command_status=None,
            )
            foreign["releaseId"] = "10000000-0000-4000-8000-000000000099"
            request_collision = _rollout(
                "20000000-0000-4000-8000-000000000098",
                status="DRAFT",
                target_status="PENDING",
                command_id=None,
                sequence=98,
                command_status=None,
            )
            active = _rollout(
                ROLLBACK_ROLLOUT_ID,
                status="RUNNING",
                target_status="COMMAND_ISSUED",
                command_id=ROLLBACK_COMMAND_ID,
                sequence=11,
                command_status="QUEUED",
            )
            cases = (
                ("malformed", [{"releaseId": RELEASE_ID}], "inventory entry"),
                ("foreign", [foreign], "conflicting active rollout"),
                (
                    "request-collision",
                    [request_collision],
                    "conflicting active rollout exists for this client request",
                ),
                ("duplicate", [active, active], "duplicate entries"),
            )
            for label, inventory, expected in cases:
                with self.subTest(label=label):
                    api = _Api([inventory])
                    with self.assertRaisesRegex(MODULE.RolloutControlError, expected):
                        MODULE.issue_rollout(
                            api,
                            space_id=SPACE_ID,
                            device_id=DEVICE_ID,
                            release_evidence_path=release_path,
                            purpose="rollback",
                            client_request_id=ROLLBACK_CLIENT_REQUEST_ID,
                            pre_commit_soak_seconds=30,
                            command_ttl_seconds=1800,
                            max_failure_percent=0,
                            previous_issuance_path=None,
                            previous_terminal_path=None,
                            created_evidence_path=root / f"{label}-created.json",
                            output=root / f"{label}-issued.json",
                        )
                    self.assertFalse(
                        any(method == "POST" for method, _path, _payload in api.calls)
                    )

    def test_new_issuance_rejects_invalid_command_tuple_and_halts(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root).resolve()
            release_path = self._register(root)
            draft = _rollout(
                ROLLBACK_ROLLOUT_ID,
                status="DRAFT",
                target_status="PENDING",
                command_id=None,
                sequence=11,
                command_status=None,
            )
            malformed = _rollout(
                ROLLBACK_ROLLOUT_ID,
                status="RUNNING",
                target_status="COMMAND_ISSUED",
                command_id=ROLLBACK_COMMAND_ID,
                sequence=11,
                command_status="QUEUED",
            )
            malformed["targets"][0]["latestCommand"]["payloadHash"] = "not-a-hash"
            live = _rollout(
                ROLLBACK_ROLLOUT_ID,
                status="RUNNING",
                target_status="COMMAND_ISSUED",
                command_id=ROLLBACK_COMMAND_ID,
                sequence=11,
                command_status="QUEUED",
            )
            halted = _rollout(
                ROLLBACK_ROLLOUT_ID,
                status="HALTED",
                target_status="EXPIRED",
                command_id=ROLLBACK_COMMAND_ID,
                sequence=11,
                command_status="EXPIRED",
            )
            issuance_path = root / "invalid-command-issued.json"
            abort_path = root / "invalid-command-abort.json"
            api = _Api([[], draft, malformed, live, halted])
            with self.assertRaisesRegex(
                MODULE.RolloutControlError, "rollout command identity"
            ):
                MODULE.issue_rollout(
                    api,
                    space_id=SPACE_ID,
                    device_id=DEVICE_ID,
                    release_evidence_path=release_path,
                    purpose="rollback",
                    client_request_id=ROLLBACK_CLIENT_REQUEST_ID,
                    pre_commit_soak_seconds=30,
                    command_ttl_seconds=1800,
                    max_failure_percent=0,
                    previous_issuance_path=None,
                    previous_terminal_path=None,
                    created_evidence_path=root / "invalid-command-created.json",
                    failure_evidence_path=abort_path,
                    output=issuance_path,
                )
            self.assertFalse(issuance_path.exists())
            self.assertEqual(
                json.loads(abort_path.read_text())["commandStatus"], "EXPIRED"
            )

    def test_signed_command_principal_and_single_target_cohort_are_exact(self) -> None:
        release = _release_response()
        policy = {
            "preCommitSoakSeconds": 30,
            "commandTtlSeconds": 1800,
            "maxFailurePercent": 0,
        }
        for field, value in (
            ("actor", ""),
            ("authorizationContext", "DEVICE_OPERATOR"),
        ):
            with self.subTest(field=field):
                projection = _rollout(
                    ROLLBACK_ROLLOUT_ID,
                    status="RUNNING",
                    target_status="COMMAND_ISSUED",
                    command_id=ROLLBACK_COMMAND_ID,
                    sequence=11,
                    command_status="QUEUED",
                )
                projection["targets"][0]["latestCommand"][field] = value
                _validated, target = MODULE._validate_rollout(
                    projection,
                    space_id=SPACE_ID,
                    device_id=DEVICE_ID,
                    release=release,
                    policy=policy,
                )
                with self.assertRaisesRegex(
                    MODULE.RolloutControlError, "command identity"
                ):
                    MODULE._command(
                        target,
                        space_id=SPACE_ID,
                        device_id=DEVICE_ID,
                        release=release,
                        expected_status="QUEUED",
                    )

        for field, value in (
            ("cohortKey", "IQ9075_DEV/iq9075_dev/iq9075-dev"),
            ("waveNumber", 1),
        ):
            with self.subTest(field=field):
                projection = _rollout(
                    ROLLBACK_ROLLOUT_ID,
                    status="DRAFT",
                    target_status="PENDING",
                    command_id=None,
                    sequence=11,
                    command_status=None,
                )
                projection["targets"][0][field] = value
                with self.assertRaisesRegex(
                    MODULE.RolloutControlError, "target identity"
                ):
                    MODULE._validate_rollout(
                        projection,
                        space_id=SPACE_ID,
                        device_id=DEVICE_ID,
                        release=release,
                        policy=policy,
                    )

    def test_lost_start_response_reconciles_exact_live_command(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root).resolve()
            release_path = self._register(root)
            draft = _rollout(
                ROLLBACK_ROLLOUT_ID,
                status="DRAFT",
                target_status="PENDING",
                command_id=None,
                sequence=11,
                command_status=None,
            )
            started = _rollout(
                ROLLBACK_ROLLOUT_ID,
                status="RUNNING",
                target_status="COMMAND_ISSUED",
                command_id=ROLLBACK_COMMAND_ID,
                sequence=11,
                command_status="QUEUED",
            )
            api = _Api(
                [
                    [],
                    draft,
                    MODULE.RolloutControlError("lost start response"),
                    MODULE.RolloutControlError("lost status response"),
                    started,
                ]
            )
            issued = MODULE.issue_rollout(
                api,
                space_id=SPACE_ID,
                device_id=DEVICE_ID,
                release_evidence_path=release_path,
                purpose="rollback",
                client_request_id=ROLLBACK_CLIENT_REQUEST_ID,
                pre_commit_soak_seconds=30,
                command_ttl_seconds=1800,
                max_failure_percent=0,
                previous_issuance_path=None,
                previous_terminal_path=None,
                created_evidence_path=root / "created.json",
                failure_evidence_path=root / "abort.json",
                output=root / "issued.json",
                monotonic=lambda: 0.0,
                sleeper=lambda _seconds: None,
            )
            self.assertEqual(issued["command"]["commandId"], ROLLBACK_COMMAND_ID)
            self.assertFalse((root / "abort.json").exists())
            self.assertEqual(
                sum(
                    method == "POST" and path.endswith("/start")
                    for method, path, _payload in api.calls
                ),
                1,
            )

    def test_lost_create_response_recovers_only_the_idempotent_request(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root).resolve()
            release_path = self._register(root)
            draft = _rollout(
                ROLLBACK_ROLLOUT_ID,
                status="DRAFT",
                target_status="PENDING",
                command_id=None,
                sequence=11,
                command_status=None,
            )
            started = _rollout(
                ROLLBACK_ROLLOUT_ID,
                status="RUNNING",
                target_status="COMMAND_ISSUED",
                command_id=ROLLBACK_COMMAND_ID,
                sequence=11,
                command_status="QUEUED",
            )
            api = _Api(
                [
                    [],
                    MODULE.RolloutControlError("lost create response"),
                    [draft],
                    draft,
                    started,
                ]
            )
            issued = MODULE.issue_rollout(
                api,
                space_id=SPACE_ID,
                device_id=DEVICE_ID,
                release_evidence_path=release_path,
                purpose="rollback",
                client_request_id=ROLLBACK_CLIENT_REQUEST_ID,
                pre_commit_soak_seconds=30,
                command_ttl_seconds=1800,
                max_failure_percent=0,
                previous_issuance_path=None,
                previous_terminal_path=None,
                created_evidence_path=root / "created.json",
                output=root / "issued.json",
            )
            self.assertEqual(issued["rolloutId"], ROLLBACK_ROLLOUT_ID)
            self.assertEqual(issued["clientRequestId"], ROLLBACK_CLIENT_REQUEST_ID)
            self.assertTrue((root / "created.json").is_file())
            self.assertEqual(
                sum(
                    method == "POST" and path.endswith("/agent-rollouts")
                    for method, path, _payload in api.calls
                ),
                1,
            )

    def test_health_unknown_before_issuance_is_bounded_and_recoverable(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root).resolve()
            release_path = self._register(root)
            draft = _rollout(
                ROLLBACK_ROLLOUT_ID,
                status="DRAFT",
                target_status="PENDING",
                command_id=None,
                sequence=11,
                command_status=None,
            )
            paused = _rollout(
                ROLLBACK_ROLLOUT_ID,
                status="PAUSED_HEALTH_UNKNOWN",
                target_status="HEALTH_UNKNOWN",
                command_id=None,
                sequence=11,
                command_status=None,
            )
            started = _rollout(
                ROLLBACK_ROLLOUT_ID,
                status="RUNNING",
                target_status="COMMAND_ISSUED",
                command_id=ROLLBACK_COMMAND_ID,
                sequence=11,
                command_status="RECEIVED",
            )
            api = _Api([[], draft, paused, paused, started])
            issued = MODULE.issue_rollout(
                api,
                space_id=SPACE_ID,
                device_id=DEVICE_ID,
                release_evidence_path=release_path,
                purpose="rollback",
                client_request_id=ROLLBACK_CLIENT_REQUEST_ID,
                pre_commit_soak_seconds=30,
                command_ttl_seconds=1800,
                max_failure_percent=0,
                previous_issuance_path=None,
                previous_terminal_path=None,
                created_evidence_path=root / "created.json",
                output=root / "issued.json",
                monotonic=lambda: 0.0,
                sleeper=lambda _seconds: None,
            )
            self.assertEqual(issued["command"]["status"], "RECEIVED")
            self.assertEqual(
                sum(
                    method == "POST" and path.endswith("/resume")
                    for method, path, _payload in api.calls
                ),
                1,
            )

    def test_health_unknown_issuance_timeout_halts_and_preserves_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root).resolve()
            release_path = self._register(root)
            draft = _rollout(
                ROLLBACK_ROLLOUT_ID,
                status="DRAFT",
                target_status="PENDING",
                command_id=None,
                sequence=11,
                command_status=None,
            )
            paused = _rollout(
                ROLLBACK_ROLLOUT_ID,
                status="PAUSED_HEALTH_UNKNOWN",
                target_status="HEALTH_UNKNOWN",
                command_id=None,
                sequence=11,
                command_status=None,
            )
            halted = json.loads(json.dumps(paused))
            halted["status"] = "HALTED"
            halted["updatedAt"] = "2026-09-04T00:00:10Z"
            clock = iter((0.0, 31.0))
            created_path = root / "created.json"
            abort_path = root / "abort.json"
            with self.assertRaisesRegex(
                MODULE.RolloutControlError, "timed out waiting"
            ):
                MODULE.issue_rollout(
                    _Api(
                        [
                            [],
                            draft,
                            paused,
                            paused,
                            paused,
                            paused,
                            halted,
                        ]
                    ),
                    space_id=SPACE_ID,
                    device_id=DEVICE_ID,
                    release_evidence_path=release_path,
                    purpose="rollback",
                    client_request_id=ROLLBACK_CLIENT_REQUEST_ID,
                    pre_commit_soak_seconds=30,
                    command_ttl_seconds=1800,
                    max_failure_percent=0,
                    previous_issuance_path=None,
                    previous_terminal_path=None,
                    created_evidence_path=created_path,
                    failure_evidence_path=abort_path,
                    output=root / "issued.json",
                    wait_seconds=30,
                    poll_seconds=0.5,
                    monotonic=lambda: next(clock),
                    sleeper=lambda _seconds: None,
                )
            self.assertTrue(created_path.is_file())
            self.assertEqual(
                json.loads(abort_path.read_text())["rolloutStatus"], "HALTED"
            )

    def test_incompatible_draft_is_receipted_then_halted(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root).resolve()
            release_path = self._register(root)
            draft = _rollout(
                ROLLBACK_ROLLOUT_ID,
                status="DRAFT",
                target_status="PENDING",
                command_id=None,
                sequence=11,
                command_status=None,
            )
            draft["targets"][0]["eligibility"] = "COMPATIBILITY_MISMATCH"
            draft["targets"][0]["eligibilityReason"] = "updater too old"
            halted = json.loads(json.dumps(draft))
            halted["status"] = "HALTED"
            halted["haltReason"] = "signed target compatibility mismatch"
            halted["updatedAt"] = "2026-09-04T00:00:10Z"
            created_path = root / "created.json"
            abort_path = root / "abort.json"
            with self.assertRaisesRegex(
                MODULE.RolloutControlError, "cannot reach an exact command"
            ):
                MODULE.issue_rollout(
                    _Api([[], draft, halted, halted]),
                    space_id=SPACE_ID,
                    device_id=DEVICE_ID,
                    release_evidence_path=release_path,
                    purpose="rollback",
                    client_request_id=ROLLBACK_CLIENT_REQUEST_ID,
                    pre_commit_soak_seconds=30,
                    command_ttl_seconds=1800,
                    max_failure_percent=0,
                    previous_issuance_path=None,
                    previous_terminal_path=None,
                    created_evidence_path=created_path,
                    failure_evidence_path=abort_path,
                    output=root / "issued.json",
                )
            self.assertTrue(created_path.is_file())
            self.assertEqual(
                json.loads(abort_path.read_text())["rolloutStatus"], "HALTED"
            )

    def test_terminal_projection_recovers_lost_issuance_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root).resolve()
            release_path = self._register(root)
            issuance_path, created_path = self._issue_rollback(root, release_path)
            issuance_path.unlink()
            terminal = _rollout(
                ROLLBACK_ROLLOUT_ID,
                status="HALTED",
                target_status="ROLLED_BACK",
                command_id=ROLLBACK_COMMAND_ID,
                sequence=11,
                command_status="ROLLED_BACK",
                rollback=True,
            )
            recovered = MODULE.issue_rollout(
                _Api([terminal]),
                space_id=SPACE_ID,
                device_id=DEVICE_ID,
                release_evidence_path=release_path,
                purpose="rollback",
                client_request_id=ROLLBACK_CLIENT_REQUEST_ID,
                pre_commit_soak_seconds=30,
                command_ttl_seconds=1800,
                max_failure_percent=0,
                previous_issuance_path=None,
                previous_terminal_path=None,
                created_evidence_path=created_path,
                output=issuance_path,
            )
            self.assertEqual(recovered["command"]["status"], "ROLLED_BACK")
            result = MODULE.wait_terminal(
                _Api([terminal]),
                space_id=SPACE_ID,
                device_id=DEVICE_ID,
                issuance_path=issuance_path,
                release_evidence_path=release_path,
                output=root / "terminal.json",
                wait_seconds=30,
                poll_seconds=0.5,
            )
            self.assertEqual(result["targetStatus"], "ROLLED_BACK")

    def test_terminal_projection_rejects_release_binding_drift(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root).resolve()
            release_path = self._register(root)
            issuance_path, _created_path = self._issue_rollback(root, release_path)
            terminal = _rollout(
                ROLLBACK_ROLLOUT_ID,
                status="HALTED",
                target_status="ROLLED_BACK",
                command_id=ROLLBACK_COMMAND_ID,
                sequence=11,
                command_status="ROLLED_BACK",
                rollback=True,
            )
            terminal["targets"][0]["rollbackEvidence"]["publisherKeyId"] = (
                "untrusted-key"
            )
            output = root / "terminal.json"
            with self.assertRaisesRegex(
                MODULE.RolloutControlError, "rollback terminal evidence"
            ):
                MODULE.wait_terminal(
                    _Api([terminal]),
                    space_id=SPACE_ID,
                    device_id=DEVICE_ID,
                    issuance_path=issuance_path,
                    release_evidence_path=release_path,
                    output=output,
                    wait_seconds=30,
                    poll_seconds=0.5,
                )
            self.assertFalse(output.exists())

    def test_rollback_terminal_wait_tolerates_ack_projection_delay(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root).resolve()
            release_path = self._register(root)
            issuance_path, _created_path = self._issue_rollback(root, release_path)
            acknowledged = _rollout(
                ROLLBACK_ROLLOUT_ID,
                status="RUNNING",
                target_status="COMMAND_ISSUED",
                command_id=ROLLBACK_COMMAND_ID,
                sequence=11,
                command_status="ROLLED_BACK",
            )
            terminal = _rollout(
                ROLLBACK_ROLLOUT_ID,
                status="HALTED",
                target_status="ROLLED_BACK",
                command_id=ROLLBACK_COMMAND_ID,
                sequence=11,
                command_status="ROLLED_BACK",
                rollback=True,
            )
            result = MODULE.wait_terminal(
                _Api([acknowledged, terminal]),
                space_id=SPACE_ID,
                device_id=DEVICE_ID,
                issuance_path=issuance_path,
                release_evidence_path=release_path,
                output=root / "rollback-terminal.json",
                wait_seconds=30,
                poll_seconds=0.5,
                monotonic=lambda: 0.0,
                sleeper=lambda _seconds: None,
            )
            self.assertEqual(result["targetStatus"], "ROLLED_BACK")

    def test_terminal_wait_retries_transient_status_read_failure(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root).resolve()
            release_path = self._register(root)
            issuance_path, _created_path = self._issue_rollback(root, release_path)
            terminal = _rollout(
                ROLLBACK_ROLLOUT_ID,
                status="HALTED",
                target_status="ROLLED_BACK",
                command_id=ROLLBACK_COMMAND_ID,
                sequence=11,
                command_status="ROLLED_BACK",
                rollback=True,
            )
            result = MODULE.wait_terminal(
                _Api([MODULE.RolloutControlError("lost status response"), terminal]),
                space_id=SPACE_ID,
                device_id=DEVICE_ID,
                issuance_path=issuance_path,
                release_evidence_path=release_path,
                output=root / "rollback-terminal.json",
                wait_seconds=30,
                poll_seconds=0.5,
                monotonic=lambda: 0.0,
                sleeper=lambda _seconds: None,
            )
            self.assertEqual(result["rolloutStatus"], "HALTED")

    def test_recovered_terminal_issuance_uses_stable_command_marker(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root).resolve()
            release_path = self._register(root)
            draft = _rollout(
                ROLLBACK_ROLLOUT_ID,
                status="DRAFT",
                target_status="PENDING",
                command_id=None,
                sequence=11,
                command_status=None,
            )
            terminal = _rollout(
                ROLLBACK_ROLLOUT_ID,
                status="HALTED",
                target_status="ROLLED_BACK",
                command_id=ROLLBACK_COMMAND_ID,
                sequence=11,
                command_status="ROLLED_BACK",
                rollback=True,
            )
            terminal["updatedAt"] = "2026-09-04T00:00:20Z"
            issued = MODULE.issue_rollout(
                _Api(
                    [
                        [],
                        draft,
                        MODULE.RolloutControlError("lost start response"),
                        terminal,
                    ]
                ),
                space_id=SPACE_ID,
                device_id=DEVICE_ID,
                release_evidence_path=release_path,
                purpose="rollback",
                client_request_id=ROLLBACK_CLIENT_REQUEST_ID,
                pre_commit_soak_seconds=30,
                command_ttl_seconds=1800,
                max_failure_percent=0,
                previous_issuance_path=None,
                previous_terminal_path=None,
                created_evidence_path=root / "created.json",
                output=root / "issued.json",
                monotonic=lambda: 0.0,
                sleeper=lambda _seconds: None,
            )
            self.assertEqual(issued["commandIssuedAt"], "2026-09-04T00:00:02Z")
            self.assertNotIn("updatedAt", issued)

    def test_issuance_replay_rejects_any_live_command_tuple_drift(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root).resolve()
            release_path = self._register(root)
            issuance_path, created_path = self._issue_rollback(root, release_path)
            drifted = _rollout(
                ROLLBACK_ROLLOUT_ID,
                status="RUNNING",
                target_status="COMMAND_ISSUED",
                command_id=ROLLBACK_COMMAND_ID,
                sequence=11,
                command_status="RECEIVED",
            )
            drifted["targets"][0]["latestCommand"]["payloadHash"] = "f" * 64
            with self.assertRaisesRegex(
                MODULE.RolloutControlError, "live rollout differs"
            ):
                MODULE.issue_rollout(
                    _Api([drifted]),
                    space_id=SPACE_ID,
                    device_id=DEVICE_ID,
                    release_evidence_path=release_path,
                    purpose="rollback",
                    client_request_id=ROLLBACK_CLIENT_REQUEST_ID,
                    pre_commit_soak_seconds=30,
                    command_ttl_seconds=1800,
                    max_failure_percent=0,
                    previous_issuance_path=None,
                    previous_terminal_path=None,
                    created_evidence_path=created_path,
                    output=issuance_path,
                )

    def test_commit_predecessor_rejects_journal_tuple_drift_before_create(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root).resolve()
            release_path = self._register(root)
            issuance_path, _created_path = self._issue_rollback(root, release_path)
            terminal_path = root / "rollback-terminal.json"
            rollback_projection = _rollout(
                ROLLBACK_ROLLOUT_ID,
                status="HALTED",
                target_status="ROLLED_BACK",
                command_id=ROLLBACK_COMMAND_ID,
                sequence=11,
                command_status="ROLLED_BACK",
                rollback=True,
            )
            MODULE.wait_terminal(
                _Api([rollback_projection]),
                space_id=SPACE_ID,
                device_id=DEVICE_ID,
                issuance_path=issuance_path,
                release_evidence_path=release_path,
                output=terminal_path,
                wait_seconds=30,
                poll_seconds=0.5,
            )
            journal = _command(ROLLBACK_COMMAND_ID, 11, "ROLLED_BACK")
            journal["payloadHash"] = "f" * 64
            api = _Api([rollback_projection, [journal]])
            with self.assertRaisesRegex(
                MODULE.RolloutControlError, "latest device sequence"
            ):
                MODULE.issue_rollout(
                    api,
                    space_id=SPACE_ID,
                    device_id=DEVICE_ID,
                    release_evidence_path=release_path,
                    purpose="commit",
                    client_request_id=COMMIT_CLIENT_REQUEST_ID,
                    pre_commit_soak_seconds=30,
                    command_ttl_seconds=1800,
                    max_failure_percent=0,
                    previous_issuance_path=issuance_path,
                    previous_terminal_path=terminal_path,
                    created_evidence_path=root / "commit-created.json",
                    output=root / "commit-issued.json",
                )
            self.assertFalse(
                any(method == "POST" for method, _path, _payload in api.calls)
            )

    def test_halt_receipt_is_strict_and_revalidated_against_live_state(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root).resolve()
            release_path = self._register(root)
            draft = _rollout(
                ROLLBACK_ROLLOUT_ID,
                status="DRAFT",
                target_status="PENDING",
                command_id=None,
                sequence=11,
                command_status=None,
            )
            halted = _rollout(
                ROLLBACK_ROLLOUT_ID,
                status="HALTED",
                target_status="PENDING",
                command_id=None,
                sequence=11,
                command_status=None,
            )
            created_path = self._write_created(root, release_path, projection=draft)
            abort_path = root / "abort.json"
            first = MODULE.halt_rollout(
                _Api([draft, halted]),
                space_id=SPACE_ID,
                device_id=DEVICE_ID,
                release_evidence_path=release_path,
                created_evidence_path=created_path,
                purpose="rollback",
                pre_commit_soak_seconds=30,
                command_ttl_seconds=1800,
                max_failure_percent=0,
                output=abort_path,
            )
            replay_api = _Api([halted])
            replay = MODULE.halt_rollout(
                replay_api,
                space_id=SPACE_ID,
                device_id=DEVICE_ID,
                release_evidence_path=release_path,
                created_evidence_path=created_path,
                purpose="rollback",
                pre_commit_soak_seconds=30,
                command_ttl_seconds=1800,
                max_failure_percent=0,
                output=abort_path,
            )
            self.assertEqual(replay, first)
            self.assertEqual(len(replay_api.calls), 1)

            abort_path.write_bytes(_canonical({"kind": "nuvion-agent-rollout-abort"}))
            abort_path.chmod(0o600)
            invalid_api = _Api([])
            with self.assertRaisesRegex(
                MODULE.RolloutControlError, "persisted rollout abort evidence"
            ):
                MODULE.halt_rollout(
                    invalid_api,
                    space_id=SPACE_ID,
                    device_id=DEVICE_ID,
                    release_evidence_path=release_path,
                    created_evidence_path=created_path,
                    purpose="rollback",
                    pre_commit_soak_seconds=30,
                    command_ttl_seconds=1800,
                    max_failure_percent=0,
                    output=abort_path,
                )
            self.assertEqual(invalid_api.calls, [])

    def test_evidence_hashes_bind_exact_files(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root).resolve()
            release_path = self._register(root)
            evidence = json.loads(release_path.read_text())
            self.assertEqual(
                evidence["bomFileSha256"],
                hashlib.sha256(_canonical(_bom())).hexdigest(),
            )
            self.assertEqual(
                evidence["signatureFileSha256"],
                hashlib.sha256(_canonical(_signature())).hexdigest(),
            )
            self.assertEqual(os.stat(release_path).st_mode & 0o777, 0o600)

    def test_release_registration_retry_revalidates_live_without_new_create(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root).resolve()
            release_path = self._register(root)
            bom_path = root / "release-bom.json"
            signature_path = root / "release-bom.json.sig"
            api = _Api([_release_response()])
            replay = MODULE.register_release(
                api,
                space_id=SPACE_ID,
                bom_path=bom_path,
                signature_path=signature_path,
                output=release_path,
            )
            self.assertEqual(replay["release"]["releaseId"], RELEASE_ID)
            self.assertEqual(
                api.calls,
                [
                    (
                        "GET",
                        f"/spaces/{SPACE_ID}/agent-releases/{RELEASE_ID}",
                        None,
                    )
                ],
            )

            signature_path.write_bytes(signature_path.read_bytes() + b" ")
            invalid_api = _Api([])
            with self.assertRaisesRegex(
                MODULE.RolloutControlError, "not canonical JSON"
            ):
                MODULE.register_release(
                    invalid_api,
                    space_id=SPACE_ID,
                    bom_path=bom_path,
                    signature_path=signature_path,
                    output=release_path,
                )
            self.assertEqual(invalid_api.calls, [])

    def test_exact_active_rollout_can_be_explicitly_adopted(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root).resolve()
            release_path = self._register(root)
            draft = _rollout(
                ROLLBACK_ROLLOUT_ID,
                status="DRAFT",
                target_status="PENDING",
                command_id=None,
                sequence=11,
                command_status=None,
            )
            started = _rollout(
                ROLLBACK_ROLLOUT_ID,
                status="RUNNING",
                target_status="COMMAND_ISSUED",
                command_id=ROLLBACK_COMMAND_ID,
                sequence=11,
                command_status="QUEUED",
            )
            api = _Api([[draft], draft, started])
            issued = MODULE.issue_rollout(
                api,
                space_id=SPACE_ID,
                device_id=DEVICE_ID,
                release_evidence_path=release_path,
                purpose="rollback",
                client_request_id=ROLLBACK_CLIENT_REQUEST_ID,
                pre_commit_soak_seconds=30,
                command_ttl_seconds=1800,
                max_failure_percent=0,
                previous_issuance_path=None,
                previous_terminal_path=None,
                created_evidence_path=root / "adopt-created.json",
                output=root / "adopt-issued.json",
                adopt_rollout_id=ROLLBACK_ROLLOUT_ID,
            )
            self.assertEqual(issued["rolloutId"], ROLLBACK_ROLLOUT_ID)
            self.assertFalse(
                any(
                    method == "POST" and path.endswith("/agent-rollouts")
                    for method, path, _payload in api.calls
                )
            )

    def test_local_preconditions_fail_before_any_remote_side_effect(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root).resolve()
            release_path = self._register(root)
            output = root / "already-issued.json"
            output.write_bytes(_canonical({"invalid": True}))
            output.chmod(0o600)
            api = _Api([])
            with self.assertRaises(MODULE.RolloutControlError):
                MODULE.issue_rollout(
                    api,
                    space_id=SPACE_ID,
                    device_id=DEVICE_ID,
                    release_evidence_path=release_path,
                    purpose="rollback",
                    client_request_id=ROLLBACK_CLIENT_REQUEST_ID,
                    pre_commit_soak_seconds=30,
                    command_ttl_seconds=1800,
                    max_failure_percent=0,
                    previous_issuance_path=None,
                    previous_terminal_path=None,
                    created_evidence_path=root / "created.json",
                    output=output,
                )
            self.assertEqual(api.calls, [])

    def test_recoverable_health_pause_is_polled_until_succeeded(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root).resolve()
            release_path = self._register(root)
            rollback_issuance, _created = self._issue_rollback(root, release_path)
            rollback_terminal = root / "rollback-terminal.json"
            rollback_projection = _rollout(
                ROLLBACK_ROLLOUT_ID,
                status="HALTED",
                target_status="ROLLED_BACK",
                command_id=ROLLBACK_COMMAND_ID,
                sequence=11,
                command_status="ROLLED_BACK",
                rollback=True,
            )
            MODULE.wait_terminal(
                _Api([rollback_projection]),
                space_id=SPACE_ID,
                device_id=DEVICE_ID,
                issuance_path=rollback_issuance,
                release_evidence_path=release_path,
                output=rollback_terminal,
                wait_seconds=30,
                poll_seconds=0.5,
            )
            draft = _rollout(
                COMMIT_ROLLOUT_ID,
                status="DRAFT",
                target_status="PENDING",
                command_id=None,
                sequence=12,
                command_status=None,
            )
            started = _rollout(
                COMMIT_ROLLOUT_ID,
                status="RUNNING",
                target_status="COMMAND_ISSUED",
                command_id=COMMIT_COMMAND_ID,
                sequence=12,
                command_status="QUEUED",
            )
            commit_issuance = root / "commit-issued.json"
            MODULE.issue_rollout(
                _Api(
                    [
                        rollback_projection,
                        [_command(ROLLBACK_COMMAND_ID, 11, "ROLLED_BACK")],
                        [],
                        draft,
                        started,
                    ]
                ),
                space_id=SPACE_ID,
                device_id=DEVICE_ID,
                release_evidence_path=release_path,
                purpose="commit",
                client_request_id=COMMIT_CLIENT_REQUEST_ID,
                pre_commit_soak_seconds=30,
                command_ttl_seconds=1800,
                max_failure_percent=0,
                previous_issuance_path=rollback_issuance,
                previous_terminal_path=rollback_terminal,
                created_evidence_path=root / "commit-created.json",
                output=commit_issuance,
            )
            paused = _rollout(
                COMMIT_ROLLOUT_ID,
                status="PAUSED_HEALTH_UNKNOWN",
                target_status="SUCCEEDED",
                command_id=COMMIT_COMMAND_ID,
                sequence=12,
                command_status="SUCCEEDED",
            )
            succeeded = _rollout(
                COMMIT_ROLLOUT_ID,
                status="SUCCEEDED",
                target_status="SUCCEEDED",
                command_id=COMMIT_COMMAND_ID,
                sequence=12,
                command_status="SUCCEEDED",
            )
            result = MODULE.wait_terminal(
                _Api([paused, succeeded]),
                space_id=SPACE_ID,
                device_id=DEVICE_ID,
                issuance_path=commit_issuance,
                release_evidence_path=release_path,
                output=root / "commit-terminal.json",
                wait_seconds=30,
                poll_seconds=0.5,
                monotonic=lambda: 0.0,
                sleeper=lambda _seconds: None,
            )
            self.assertEqual(result["rolloutStatus"], "SUCCEEDED")

    def test_terminal_loader_rejects_weakened_reported_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root).resolve()
            release_path = self._register(root)
            issuance, _created = self._issue_rollback(root, release_path)
            terminal_path = root / "terminal.json"
            MODULE.wait_terminal(
                _Api(
                    [
                        _rollout(
                            ROLLBACK_ROLLOUT_ID,
                            status="HALTED",
                            target_status="ROLLED_BACK",
                            command_id=ROLLBACK_COMMAND_ID,
                            sequence=11,
                            command_status="ROLLED_BACK",
                            rollback=True,
                        )
                    ]
                ),
                space_id=SPACE_ID,
                device_id=DEVICE_ID,
                issuance_path=issuance,
                release_evidence_path=release_path,
                output=terminal_path,
                wait_seconds=30,
                poll_seconds=0.5,
            )
            terminal = json.loads(terminal_path.read_text())
            original = json.loads(json.dumps(terminal))
            terminal["reportedEvidence"]["functionalHealth"] = "FUNCTIONAL_HEALTHY"
            terminal_path.write_bytes(_canonical(terminal))
            terminal_path.chmod(0o600)
            issuance_value = json.loads(issuance.read_text())
            with self.assertRaisesRegex(
                MODULE.RolloutControlError, "rollback terminal"
            ):
                MODULE._load_terminal(
                    terminal_path,
                    issuance=issuance_value,
                    issuance_raw=issuance.read_bytes(),
                )
            reordered = json.loads(json.dumps(original))
            reordered["createdAt"] = "2026-09-03T23:59:59Z"
            terminal_path.write_bytes(_canonical(reordered))
            with self.assertRaisesRegex(
                MODULE.RolloutControlError, "timestamps are reordered"
            ):
                MODULE._load_terminal(
                    terminal_path,
                    issuance=issuance_value,
                    issuance_raw=issuance.read_bytes(),
                )
            original["command"]["payloadHash"] = "f" * 64
            terminal_path.write_bytes(_canonical(original))
            with self.assertRaisesRegex(
                MODULE.RolloutControlError, "terminal rollout result"
            ):
                MODULE._load_terminal(
                    terminal_path,
                    issuance=issuance_value,
                    issuance_raw=issuance.read_bytes(),
                )

    def test_rollback_terminal_rejects_legacy_top_level_slot_spoof(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root).resolve()
            release_path = self._register(root)
            issuance, _created = self._issue_rollback(root, release_path)
            projection = _rollout(
                ROLLBACK_ROLLOUT_ID,
                status="HALTED",
                target_status="ROLLED_BACK",
                command_id=ROLLBACK_COMMAND_ID,
                sequence=11,
                command_status="ROLLED_BACK",
                rollback=True,
            )
            identity = projection["targets"][0]["identitySnapshot"]
            identity["slot"] = BASELINE_SLOT
            del identity["agentUpdate"]["activeSlot"]
            with self.assertRaisesRegex(
                MODULE.RolloutControlError, "rollback baseline slot"
            ):
                MODULE.wait_terminal(
                    _Api([projection]),
                    space_id=SPACE_ID,
                    device_id=DEVICE_ID,
                    issuance_path=issuance,
                    release_evidence_path=release_path,
                    output=root / "spoofed-terminal.json",
                    wait_seconds=30,
                    poll_seconds=0.5,
                )

    def test_rollback_terminal_accepts_only_exact_legacy_verified_release_slot(
        self,
    ) -> None:
        digest = "sha256:" + "b" * 64
        release_slot = "releases/" + digest[7:]
        identity = {
            "agentVersion": "0.1.120",
            "bomDigest": digest,
            "bomVerificationStatus": "VERIFIED",
            "updaterVersion": "0.2.0",
            "agentUpdate": {
                "authenticatedHelper": True,
                "capabilityAvailable": True,
                "updaterVersion": "0.2.0",
            },
        }
        self.assertEqual(
            MODULE._rollback_baseline_slot(identity, {"previousSlot": release_slot}),
            release_slot,
        )
        for rejected in ("bootstrap/0.1.120", "releases/" + "c" * 64):
            with (
                self.subTest(rejected=rejected),
                self.assertRaisesRegex(
                    MODULE.RolloutControlError, "rollback baseline slot"
                ),
            ):
                MODULE._rollback_baseline_slot(identity, {"previousSlot": rejected})

    def test_http_client_is_cookie_bound_to_the_fixed_https_origin(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root).resolve()
            root.chmod(0o700)
            cookie_path = root / "cookies"
            jar = http.cookiejar.MozillaCookieJar(str(cookie_path))
            jar.set_cookie(
                http.cookiejar.Cookie(
                    version=0,
                    name="accessToken",
                    value="opaque-test-cookie",
                    port=None,
                    port_specified=False,
                    domain="api.nuvion-dev.plaidlabs.ai",
                    domain_specified=True,
                    domain_initial_dot=False,
                    path="/",
                    path_specified=True,
                    secure=True,
                    expires=int(time.time()) + 3600,
                    discard=False,
                    comment=None,
                    comment_url=None,
                    rest={"HttpOnly": None},
                    rfc2109=False,
                )
            )
            jar.save(ignore_discard=True, ignore_expires=True)
            cookie_path.chmod(0o600)
            body = _canonical({"message": "ok", "data": []})
            opener = _Opener(body)
            api = MODULE.FleetApi(
                MODULE.DEFAULT_API_ORIGIN,
                cookie_path,
                opener=opener,
            )

            self.assertEqual(api.request(f"/spaces/{SPACE_ID}/agent-rollouts"), [])
            request, timeout = opener.requests[0]
            self.assertEqual(
                request.full_url,
                MODULE.DEFAULT_API_ORIGIN + f"/spaces/{SPACE_ID}/agent-rollouts",
            )
            self.assertEqual(request.method, "GET")
            self.assertEqual(timeout, 15)
            with self.assertRaisesRegex(
                MODULE.RolloutControlError, "authoritative Nuvion dev API"
            ):
                MODULE.FleetApi(
                    "https://api.example.invalid",
                    cookie_path,
                    opener=opener,
                )


if __name__ == "__main__":
    unittest.main()
