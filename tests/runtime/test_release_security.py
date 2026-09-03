from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

from nuvion_app.runtime.release_bom import (
    ReleaseTarget,
    build_release_bom_v2_payload,
    canonical_release_bom_json,
)
from nuvion_app.runtime import stable_file as STABLE_FILE

ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str, relative: str):
    specification = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


PUBLISHER_TRUST = load_script(
    "publisher_trust", "packaging/release/publisher_trust.py"
)
VERIFY_SOURCE = load_script(
    "verify_release_source", "packaging/release/verify-release-source.py"
)
PLAN_OTA = load_script("plan_iq9075_ota", "packaging/release/plan-iq9075-ota.py")
PREPARE_APT = load_script(
    "prepare_apt_rollback", "packaging/release/prepare-apt-rollback.py"
)
PROMOTION = load_script(
    "generate_release_promotion", "packaging/release/generate-release-promotion.py"
)
SETTINGS = load_script(
    "verify_github_release_settings",
    "packaging/release/verify-github-release-settings.py",
)
GITHUB_RELEASE = load_script(
    "publish_github_release", "packaging/release/publish-github-release.py"
)
HOMEBREW_PROMOTION = load_script(
    "verify_homebrew_promotion",
    "packaging/release/verify-homebrew-promotion.py",
)
READINESS = load_script(
    "verify_release_readiness", "packaging/release/verify-release-readiness.py"
)
RELEASE_GATE = load_script(
    "verify_agent_release_gate",
    "packaging/release/verify-agent-release-gate.py",
)
SETTINGS_ATTESTATION = load_script(
    "verify_release_settings_attestation",
    "packaging/release/verify-release-settings-attestation.py",
)
FACE_MANIFEST = load_script(
    "face_artifact_manifest",
    "packaging/release/face-artifact-manifest.py",
)
FLEET_E2E = load_script(
    "release_security_iq9075_fleet_e2e",
    "packaging/dev/run-iq9075-fleet-e2e.py",
)
PHYSICAL_EVIDENCE = load_script(
    "assemble_iq9075_physical_evidence",
    "packaging/release/assemble-iq9075-physical-evidence.py",
)
FLEET_RUNTIME_EVIDENCE = load_script(
    "assemble_iq9075_fleet_runtime_evidence",
    "packaging/release/assemble-iq9075-fleet-runtime-evidence.py",
)


def persistent_state_evidence() -> dict[str, object]:
    roots = {
        path: {
            "exists": False,
            "entries": 0,
            "bytes": 0,
            "sha256": hashlib.sha256(path.encode("utf-8")).hexdigest(),
        }
        for path in FLEET_E2E.CANDIDATE_PERSISTENT_PATHS
    }
    serialized = (
        json.dumps(roots, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    return {
        "schemaVersion": 1,
        "roots": roots,
        "sha256": hashlib.sha256(serialized).hexdigest(),
        "entries": 0,
        "bytes": 0,
    }


def release_tree_evidence(slots: dict[str, str]) -> dict[str, object]:
    trees = {
        role: {
            "target": target,
            "exists": True,
            "entries": 1,
            "bytes": 0,
            "sha256": hashlib.sha256(f"{role}:{target}".encode()).hexdigest(),
        }
        for role, target in slots.items()
    }
    serialized = (
        json.dumps(trees, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    return {
        "schemaVersion": 1,
        "slots": trees,
        "sha256": hashlib.sha256(serialized).hexdigest(),
        "entries": len(trees),
        "bytes": 0,
    }


def candidate_execution_proof(run_id: str) -> dict[str, object]:
    unit = f"nuvion-candidate-soak-{run_id.replace('-', '')}.service"
    control_group = "/system.slice/" + unit
    temporary = {
        path: {
            "mountId": 101 + index,
            "fsType": "tmpfs",
            "sizeBytes": limits["bytes"],
            "inodeLimit": limits["inodes"],
            "readOnly": False,
        }
        for index, (path, limits) in enumerate(FLEET_E2E.CANDIDATE_TMPFS_LIMITS.items())
    }
    return {
        "schemaVersion": 1,
        "unit": unit,
        "mainPid": 9001,
        "controlGroup": control_group,
        "pidControlGroup": control_group,
        "recursivePopulated": True,
        "uidIsolation": {
            "before": {
                "schemaVersion": 1,
                "uid": 4242,
                "pids": [],
                "controlGroup": None,
                "stableScans": 2,
            },
            "during": {
                "schemaVersion": 1,
                "uid": 4242,
                "pids": [9001],
                "controlGroup": control_group,
                "stableScans": 2,
            },
        },
        "systemdProperties": dict(
            FLEET_E2E.CANDIDATE_SYSTEMD_EXPECTED_PROPERTIES
        ),
        "mountSandbox": {
            "temporaryFilesystems": temporary,
            "readOnlyPaths": {
                path: {
                    "mountId": 111 + index,
                    "mountPoint": path,
                    "readOnly": True,
                }
                for index, path in enumerate(FLEET_E2E.CANDIDATE_PERSISTENT_PATHS)
            },
            "readWritePath": {
                "mountId": 131,
                "mountPoint": f"/var/lib/nuvion-fleet-e2e/runs/{run_id}",
                "readOnly": False,
            },
            "inaccessiblePaths": {
                path: {
                    "mountId": 121 + index,
                    "mountPoint": path,
                    "mode": "0000",
                    "readOnly": True,
                }
                for index, path in enumerate(FLEET_E2E.CANDIDATE_INACCESSIBLE_PATHS)
            },
            "totalTmpfsBytes": sum(
                item["sizeBytes"] for item in temporary.values()
            ),
            "totalTmpfsInodes": sum(
                item["inodeLimit"] for item in temporary.values()
            ),
        },
    }


def candidate_termination_proof(run_id: str) -> dict[str, object]:
    unit = f"nuvion-candidate-soak-{run_id.replace('-', '')}.service"
    return {
        "schemaVersion": 1,
        "unit": unit,
        "controlGroup": "/system.slice/" + unit,
        "initialPresent": True,
        "initialPopulated": False,
        "killSignals": [],
        "stopSucceeded": True,
        "resetPerformed": True,
        "recursivePopulated": False,
        "loadState": "not-found",
        "activeState": "inactive",
        "cgroupRemoved": True,
    }


def candidate_collector_proof(run_id: str) -> dict[str, object]:
    unit = f"nuvion-candidate-soak-{run_id.replace('-', '')}.service"
    return {
        "schemaVersion": 1,
        "unit": unit,
        "controlGroup": "/system.slice/" + unit,
        "requiredSeconds": FLEET_E2E.CANDIDATE_REQUIRED_SOAK_SECONDS,
        "elapsedSeconds": float(FLEET_E2E.CANDIDATE_REQUIRED_SOAK_SECONDS),
        "scanIntervalSeconds": FLEET_E2E.CANDIDATE_UID_SCAN_INTERVAL_SECONDS,
        "sampleCount": 2,
        "observedPids": [9001],
        "escapeDetected": None,
        "allSamplesWithinCgroup": True,
        "durationSatisfied": True,
        "terminalStatus": {
            "ActiveState": "active",
            "ExecMainCode": "1",
            "ExecMainStatus": "0",
            "Result": "success",
            "SubState": "exited",
        },
        "afterTermination": {
            "schemaVersion": 1,
            "uid": 4242,
            "pids": [],
            "controlGroup": None,
            "stableScans": 2,
        },
    }


def cleanup_evidence(run_id: str) -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "kind": "nuvion-iq9075-cleanup-evidence",
        "runId": run_id,
        "complete": True,
        "recovered": False,
        "phase": "RESTORED",
        "proof": {
            "schemaVersion": 1,
            "transactionPhase": "RESTORED",
            "cleanupJournalComplete": True,
            "activeRunLeaseAbsent": True,
            "transactionSnapshotsAbsent": True,
            "recoveryArchiveAbsent": True,
            "candidateArtifactsAbsent": True,
            "candidateStagingAbsent": True,
            "trustStagingAbsent": True,
        },
    }


def canonical_sha256(value: Mapping[str, object]) -> str:
    payload = (
        json.dumps(dict(value), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def production_restoration_evidence(
    manifest: dict[str, object],
) -> dict[str, object]:
    value: dict[str, object] = {
        "schemaVersion": 1,
        "transactionPhase": "RESTORED",
        "manifestSha256": hashlib.sha256(
            (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
        ).hexdigest(),
        "files": {
            path: {
                "exists": False,
                "sha256": None,
                "mode": None,
                "uid": None,
                "gid": None,
            }
            for path in FLEET_E2E.PRODUCTION_TRANSACTION_FILES
        },
        "directories": {
            path: {"mode": 0o700, "uid": 0, "gid": 0}
            for path in FLEET_E2E.PRODUCTION_TRANSACTION_DIRECTORIES
        },
        "units": {
            unit: {
                "active": True,
                "enabled": True,
                "unitFileState": "enabled",
            }
            for unit in FLEET_E2E.PRODUCTION_UNITS
        },
    }
    value["sha256"] = hashlib.sha256(
        (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    return value


class ReleaseSecurityWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.publish = (
            ROOT / ".github/workflows/release-publish.yml"
        ).read_text(encoding="utf-8")
        self.request = (
            ROOT / ".github/workflows/release-request.yml"
        ).read_text(encoding="utf-8")
        self.face = (
            ROOT / ".github/workflows/publish-face-artifacts.yml"
        ).read_text(encoding="utf-8")

    def test_fleet_validators_execute_verified_bytes_not_reopened_paths(
        self,
    ) -> None:
        helpers = (
            PHYSICAL_EVIDENCE._module_from_verified_source,
            READINESS._module_from_verified_source,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run-iq9075-fleet-e2e.py"
            path.write_text(
                "raise RuntimeError('reopened untrusted path')\n", encoding="utf-8"
            )
            for index, helper in enumerate(helpers):
                module_name = f"_verified_fleet_validator_regression_{index}"
                with self.subTest(helper=helper.__module__):
                    module = helper(
                        module_name=module_name,
                        source=b"VALIDATOR_ORIGIN = 'verified-bytes'\n",
                        display_path=path,
                    )
                    try:
                        self.assertEqual(module.VALIDATOR_ORIGIN, "verified-bytes")
                        self.assertFalse((path.parent / "__pycache__").exists())
                    finally:
                        sys.modules.pop(module_name, None)

    def _job(self, name: str) -> str:
        start = self.publish.index(f"  {name}:")
        following = re.search(r"^  [a-z0-9-]+:\s*$", self.publish[start + 1 :], re.MULTILINE)
        return self.publish[start:] if following is None else self.publish[
            start : start + 1 + following.start()
        ]

    def _steps(self, section: str) -> list[str]:
        return re.split(r"^      - name: ", section, flags=re.MULTILINE)[1:]

    def _assert_immediate_revalidation(
        self,
        section: str,
        credential_step_name: str,
        *,
        require_live_authorization: bool = True,
    ) -> None:
        steps = self._steps(section)
        indexes = [
            index
            for index, step in enumerate(steps)
            if step.startswith(credential_step_name + "\n")
        ]
        self.assertEqual(len(indexes), 1, credential_step_name)
        self.assertGreater(indexes[0], 0, credential_step_name)
        previous = steps[indexes[0] - 1]
        self.assertIn("verify-release-settings-attestation.py", previous)
        self.assertIn("--publisher-root publisher", previous)
        if require_live_authorization:
            self.assertIn(
                "revalidate-live-release-authorization.sh", previous
            )

    @staticmethod
    def _physical_fixture(
        root: Path,
        *,
        component_sha: str,
    ) -> tuple[dict[str, object], Path, Path]:
        run_id = "12345678-1234-4abc-8def-123456789abc"
        manifest_path = root / "iq9075-v0.1.121-harness-manifest.json"
        result_path = root / "iq9075-v0.1.121-harness-result.json"
        oak_soak_path = root / "iq9075-v0.1.121-oak-soak-result.json"
        artifact_path = (
            root / "nuv-agent_0.1.121_iq9075-aarch64.agent-bundle.tar.gz"
        )
        artifact_path.write_bytes(b"exact deterministic candidate bundle")
        artifact_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        bom_path = root / "nuv-agent_0.1.121_iq9075-aarch64.release-bom.json"
        bom = build_release_bom_v2_payload(
            bom_id="nuv-agent-0.1.121-iq9075-aarch64",
            release_sequence=2,
            agent_version="0.1.121",
            component_sha=component_sha,
            config_schema="12",
            min_updater_version="0.2.0",
            targets=[
                ReleaseTarget(
                    product_model="IQ9075_DEV",
                    platform_profile="iq9075_dev",
                    hardware_revision="QCS9075-EVK",
                    architecture="aarch64",
                )
            ],
            artifact_path=artifact_path,
            artifact_kind="agent-bundle",
            built_at="2026-09-03T09:00:00Z",
        )
        bom_path.write_text(canonical_release_bom_json(bom), encoding="utf-8")
        bom_file_sha256 = hashlib.sha256(bom_path.read_bytes()).hexdigest()
        harness = ROOT / "packaging/dev/test-iq9075.sh"
        fleet_runner = ROOT / "packaging/dev/run-iq9075-fleet-e2e.py"
        fleet_manifest_path = root / "iq9075-v0.1.121-fleet-manifest.json"
        fleet_evidence_path = root / "iq9075-v0.1.121-fleet-evidence.json"
        baseline_digest = (
            "26a7f1674bdd4a24bfe26fa37c681798244990408fe7d858ca76957a88bdb9f1"
        )
        fleet_release = {
            "agentVersion": "0.1.121",
            "releaseSequence": 2,
            "artifactDigest": "sha256:" + artifact_sha256,
            "componentSha": component_sha,
            "configSchema": "12",
            "publisherKeyId": "release-iq9075-dev-2026-09-01",
        }
        fleet_manifest = FLEET_E2E.build_manifest(
            run_id=run_id,
            tool_sha256=hashlib.sha256(
                (ROOT / "packaging/dev/iq9075-board-e2e.py").read_bytes()
            ).hexdigest(),
            input_digests={
                "commandSha256": "6" * 64,
                "releaseSha256": (
                    "2d72a28745e14014d5988ecf7970dc6f09c2f077be35105b3ad233cda0d0969a"
                ),
                "healthSha256": "8" * 64,
                "bindingSha256": "9" * 64,
            },
            identity={
                "deviceId": "sp-3-nuvion-physical",
                "spaceId": 3,
                "productModel": "IQ9075_DEV",
                "platformProfile": "iq9075_dev",
                "hardwareRevision": "QCS9075-EVK",
                "architecture": "aarch64",
                "dockerRequired": False,
            },
            scenario_type="oak-fault-rollback",
            expected_command_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            expected_bom_digest=bom["bomDigest"],
            expected_candidate_slot=(
                "/opt/nuv-agent/releases/" + bom["bomDigest"][7:]
            ),
            expected_previous_slot="releases/" + baseline_digest,
            expected_previous_version="0.1.120",
            hold_seconds=10,
            release=fleet_release,
        )
        fleet_manifest_path.write_text(
            json.dumps(fleet_manifest, sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        candidate_relative = "releases/" + bom["bomDigest"][7:]
        baseline_relative = "releases/" + baseline_digest
        services = {
            "nuv-agent.service": {
                "active": True,
                "enabled": True,
                "unitFileState": "enabled",
                "mainPid": 4200,
            },
            "nuv-agent-updater.service": {
                "active": True,
                "enabled": True,
                "unitFileState": "enabled",
                "mainPid": 4300,
            },
            "nuv-agent-updater.socket": {
                "active": True,
                "enabled": True,
                "unitFileState": "enabled",
                "mainPid": 0,
            },
        }
        fleet_evidence = {
            "schemaVersion": 1,
            "protocolVersion": FLEET_E2E.PROTOCOL_VERSION,
            "runId": run_id,
            "generatedAt": "2026-09-03T10:03:00Z",
            "scenario": "oak-fault-rollback",
            "complete": True,
            "gates": {
                "foundation": True,
                "backup": True,
                "trust": True,
                "updater2": True,
                "oak": True,
                "services": True,
                "scenario": True,
            },
            "oak": {
                "port": "2-1.1",
                "vendorId": "03e7",
                "productId": "f63b",
                "speedMbps": 5000,
                "mxidSha256": "3" * 64,
                "attached": True,
                "bound": True,
            },
            "services": services,
            "runtimePids": {"candidate": 4100, "restored": 4200},
            "slots": {
                "current": baseline_relative,
                "previous": candidate_relative,
                "currentVersion": "0.1.120",
                "release": {
                    "schemaVersion": 2,
                    "bomDigest": "sha256:" + baseline_digest,
                    "agentVersion": "0.1.120",
                    "releaseSequence": 1,
                    "artifactDigest": "sha256:" + "e" * 64,
                    "componentSha": "f" * 40,
                    "configSchema": "12",
                    "publisherKeyId": "release-iq9075-dev-2026-09-01",
                },
                "previousRelease": {
                    "schemaVersion": 2,
                    "bomDigest": bom["bomDigest"],
                    **fleet_release,
                },
            },
            "updater": {
                "capabilityAvailable": True,
                "authenticatedHelper": True,
                "reason": "READY",
                "updaterVersion": "0.2.0",
                "update": {
                    "commandId": fleet_manifest["scenario"]["expectedCommandId"],
                    "sequence": 2,
                    "targetVersion": "0.1.121",
                    "bomDigest": bom["bomDigest"],
                    "phase": "ROLLED_BACK",
                    "updatePhase": "ROLLED_BACK",
                    "updatedAt": "2026-09-03T10:02:00Z",
                    "commandExpiresAt": "2026-09-03T11:00:00Z",
                    "candidateSlot": fleet_manifest["scenario"][
                        "expectedCandidateSlot"
                    ],
                    "previousSlot": baseline_relative,
                    "previousVersion": "0.1.120",
                    "releaseSequence": 2,
                    "artifactDigest": "sha256:" + artifact_sha256,
                    "componentSha": component_sha,
                    "configSchema": "12",
                    "publisherKeyId": "release-iq9075-dev-2026-09-01",
                    "bomVerificationStatus": "VERIFIED",
                    "slot": baseline_relative,
                    "rollbackSlot": baseline_relative,
                    "rollbackVersion": "0.1.120",
                    "errorCode": "ROLLED_BACK",
                    "health": "LKG_RESTORED",
                    "functionalHealth": "FUNCTIONAL_UNHEALTHY",
                },
            },
        }
        fleet_evidence_path.write_text(
            json.dumps(fleet_evidence, sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schemaVersion": 1,
            "kind": "nuvion-iq9075-physical-manifest",
            "runId": run_id,
            "agentVersion": "0.1.121",
            "componentSha": component_sha,
            "harnessSha256": hashlib.sha256(harness.read_bytes()).hexdigest(),
            "fleetRunnerSha256": hashlib.sha256(
                fleet_runner.read_bytes()
            ).hexdigest(),
            "fleetManifest": {
                "file": fleet_manifest_path.name,
                "sha256": hashlib.sha256(
                    fleet_manifest_path.read_bytes()
                ).hexdigest(),
            },
            "fleetEvidence": {
                "file": fleet_evidence_path.name,
                "sha256": hashlib.sha256(
                    fleet_evidence_path.read_bytes()
                ).hexdigest(),
            },
            "testedArtifact": {
                "name": artifact_path.name,
                "sha256": artifact_sha256,
                "sizeBytes": artifact_path.stat().st_size,
            },
            "testedBom": {
                "file": bom_path.name,
                "sha256": bom_file_sha256,
            },
            "board": {
                "productModel": "IQ9075_DEV",
                "platformProfile": "iq9075_dev",
                "hardwareRevision": "QCS9075-EVK",
                "architecture": "aarch64",
                "kernel": "6.8.0-qcom",
                "depthaiVersion": "2.32.0.0",
                "gstreamerVersion": "1.24.2",
            },
            "oakMxidSha256": "3" * 64,
            "startedAt": "2026-09-03T10:00:00Z",
            "expectedRollback": {
                "agentVersion": "0.1.120",
                "slot": (
                    "releases/"
                    "26a7f1674bdd4a24bfe26fa37c681798244990408fe7d858ca76957a88bdb9f1"
                ),
            },
        }
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        rss_samples = [
            {"elapsedSec": float(second), "rssAnonKiB": 131072}
            for second in range(0, 121, 5)
        ]
        result = {
            "schemaVersion": 2,
            "kind": "nuvion-iq9075-physical-result",
            "runId": run_id,
            "agentVersion": "0.1.121",
            "componentSha": component_sha,
            "manifestSha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "artifactSha256": artifact_sha256,
            "bomSha256": bom_file_sha256,
            "exitCode": 0,
            "outcome": {
                "status": "passed",
                "error": None,
                "cleanupErrors": [],
            },
            "soak": {
                "durationSeconds": 120.0,
                "targetFps": 30.0,
                "rawSamples": 3600,
                "rssAnonSamples": rss_samples,
                "rssAnonSlopeMiBPerMin": 0.0,
                "rssAnonRangeMiB": 0.0,
                "gstreamerErrors": [],
                "gstreamerWarnings": [],
                "maxAppsrcBuffers": 2,
                "maxAppsrcBytes": 1843200,
                "queueHighWatermarks": {
                    "clip_live_queue": 2,
                    "physical_overlay_queue": 1,
                    "physical_raw_queue": 1,
                    "uplink_live_queue": 2,
                },
            },
            "webrtc": {
                "offerCount": 1,
                "terminalStopCount": 1,
                "offerSdpHadPinnedProfile": True,
                "branchParentDetached": True,
                "queueParentDetached": True,
                "webrtcParentDetached": True,
                "teeRequestPadCount": 0,
                "queueState": "NULL",
                "webrtcState": "NULL",
                "branchObjectsFinalized": True,
                "hasPipeline": False,
            },
            "splitmux": {
                "segmentSeconds": 4.0,
                "retentionLimit": 30,
                "segmentsAtEnd": 30,
                "fragmentsOpenedDuringSoak": 30,
                "newestSegmentAgeSeconds": 2.0,
            },
            "rollback": {
                "expectedSlot": (
                    "releases/"
                    "26a7f1674bdd4a24bfe26fa37c681798244990408fe7d858ca76957a88bdb9f1"
                ),
                "candidateSlot": "releases/" + bom["bomDigest"][7:],
                "restoredSlot": (
                    "releases/"
                    "26a7f1674bdd4a24bfe26fa37c681798244990408fe7d858ca76957a88bdb9f1"
                ),
                "candidatePid": 4100,
                "restoredPid": 4200,
                "oakProbeExitCode": 0,
                "oakReady": True,
            },
        }
        result_path.write_text(
            json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        oak_soak = {
            "schemaVersion": 2,
            "kind": "nuvion-iq9075-oak-soak-result",
            "startedAt": manifest["startedAt"],
            "outcome": result["outcome"],
            "board": manifest["board"],
            "oakMxidSha256": manifest["oakMxidSha256"],
            "deviceIdentity": {
                "deviceId": "sp-3-nuvion-physical",
                "spaceId": 3,
            },
            "runtimeIdentity": {
                "agentVersion": "0.1.121",
                "componentSha": component_sha,
                "bomDigest": bom["bomDigest"],
                "pythonPath": "/usr/bin/python3",
                "sitePackagesPath": (
                    "/opt/nuv-agent/releases/"
                    + bom["bomDigest"][7:]
                    + "/venv/lib/python3.12/site-packages"
                ),
                "buildInfoPath": (
                    "/opt/nuv-agent/releases/"
                    + bom["bomDigest"][7:]
                    + "/venv/lib/python3.12/site-packages/nuvion_app/build_info.py"
                ),
                "releaseMarkerSha256": "4" * 64,
            },
            "soak": result["soak"],
            "webrtc": result["webrtc"],
            "splitmux": result["splitmux"],
        }
        oak_soak_path.write_text(
            json.dumps(oak_soak, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        manifest["oakSoak"] = {
            "file": oak_soak_path.name,
            "sha256": hashlib.sha256(oak_soak_path.read_bytes()).hexdigest(),
        }
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        result["manifestSha256"] = hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest()
        result_path.write_text(
            json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        summary: dict[str, object] = {
            "schemaVersion": 2,
            "kind": "nuvion-iq9075-physical-release-evidence",
            "agentVersion": "0.1.121",
            "componentSha": component_sha,
            "harnessManifest": {
                "file": manifest_path.name,
                "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            },
            "harnessResult": {
                "file": result_path.name,
                "sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
            },
            "physicalGate": {
                "oakSoakSeconds": 120.0,
                "rawFps": 30.0,
                "rssSlopeMiBPerMinute": 0.0,
                "rssRangeMiB": 0.0,
                "gstreamerErrors": 0,
                "webrtcBranchDisposed": True,
                "splitmuxRotated": True,
                "rollbackOakReady": True,
            },
        }
        return summary, manifest_path, result_path

    @staticmethod
    def _fleet_runtime_fixture(
        root: Path, *, component_sha: str
    ) -> dict[str, Path]:
        source = root / "source"
        source.mkdir(mode=0o700)
        run_id = "12345678-1234-4abc-8def-123456789abc"
        artifact_path = (
            source / "nuv-agent_0.1.121_iq9075-aarch64.agent-bundle.tar.gz"
        )
        artifact_path.write_bytes(b"exact deterministic candidate bundle")
        artifact_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        bom_path = source / "nuv-agent_0.1.121_iq9075-aarch64.release-bom.json"
        bom = build_release_bom_v2_payload(
            bom_id="nuv-agent-0.1.121-iq9075-aarch64",
            release_sequence=2,
            agent_version="0.1.121",
            component_sha=component_sha,
            config_schema="12",
            min_updater_version="0.2.0",
            targets=[
                ReleaseTarget(
                    product_model="IQ9075_DEV",
                    platform_profile="iq9075_dev",
                    hardware_revision="QCS9075-EVK",
                    architecture="aarch64",
                )
            ],
            artifact_path=artifact_path,
            artifact_kind="agent-bundle",
            built_at="2026-09-03T09:00:00Z",
        )
        bom_path.write_text(canonical_release_bom_json(bom), encoding="utf-8")
        baseline_digest = (
            "26a7f1674bdd4a24bfe26fa37c681798244990408fe7d858ca76957a88bdb9f1"
        )
        baseline_slot = "releases/" + baseline_digest
        candidate_slot = "releases/" + bom["bomDigest"][7:]
        fleet_release = {
            "agentVersion": "0.1.121",
            "releaseSequence": 2,
            "artifactDigest": "sha256:" + artifact_sha256,
            "componentSha": component_sha,
            "configSchema": "12",
            "publisherKeyId": "release-iq9075-dev-2026-09-01",
        }
        fleet_manifest = FLEET_E2E.build_manifest(
            run_id=run_id,
            tool_sha256=hashlib.sha256(
                (ROOT / "packaging/dev/iq9075-board-e2e.py").read_bytes()
            ).hexdigest(),
            input_digests={
                "commandSha256": "6" * 64,
                "releaseSha256": (
                    "2d72a28745e14014d5988ecf7970dc6f09c2f077be35105b3ad233cda0d0969a"
                ),
                "healthSha256": "8" * 64,
                "bindingSha256": "9" * 64,
            },
            identity={
                "deviceId": "sp-3-nuvion-runtime",
                "spaceId": 3,
                "productModel": "IQ9075_DEV",
                "platformProfile": "iq9075_dev",
                "hardwareRevision": "QCS9075-EVK",
                "architecture": "aarch64",
                "dockerRequired": False,
            },
            scenario_type="oak-fault-rollback",
            expected_command_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            expected_bom_digest=bom["bomDigest"],
            expected_candidate_slot="/opt/nuv-agent/releases/" + bom["bomDigest"][7:],
            expected_previous_slot=baseline_slot,
            expected_previous_version="0.1.120",
            hold_seconds=10,
            release=fleet_release,
        )
        fleet_manifest_path = source / "immutable-manifest.json"
        fleet_manifest_path.write_text(
            json.dumps(fleet_manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        services = {
            "nuv-agent.service": {
                "active": True,
                "enabled": True,
                "unitFileState": "enabled",
                "mainPid": 4200,
            },
            "nuv-agent-updater.service": {
                "active": True,
                "enabled": True,
                "unitFileState": "enabled",
                "mainPid": 4300,
            },
            "nuv-agent-updater.socket": {
                "active": True,
                "enabled": True,
                "unitFileState": "enabled",
                "mainPid": 0,
            },
        }
        update = {
            "commandId": fleet_manifest["scenario"]["expectedCommandId"],
            "sequence": 2,
            "targetVersion": "0.1.121",
            "bomDigest": bom["bomDigest"],
            "phase": "ROLLED_BACK",
            "updatePhase": "ROLLED_BACK",
            "updatedAt": "2026-09-03T10:02:00Z",
            "commandExpiresAt": "2026-09-03T11:00:00Z",
            "candidateSlot": fleet_manifest["scenario"]["expectedCandidateSlot"],
            "previousSlot": baseline_slot,
            "previousVersion": "0.1.120",
            "releaseSequence": 2,
            "artifactDigest": "sha256:" + artifact_sha256,
            "componentSha": component_sha,
            "configSchema": "12",
            "publisherKeyId": "release-iq9075-dev-2026-09-01",
            "bomVerificationStatus": "VERIFIED",
            "slot": baseline_slot,
            "rollbackSlot": baseline_slot,
            "rollbackVersion": "0.1.120",
            "errorCode": "ROLLED_BACK",
            "health": "LKG_RESTORED",
            "functionalHealth": "FUNCTIONAL_UNHEALTHY",
        }
        fleet_evidence = {
            "schemaVersion": 2,
            "protocolVersion": FLEET_E2E.PROTOCOL_VERSION,
            "runId": run_id,
            "generatedAt": "2026-09-03T10:03:00Z",
            "scenario": "oak-fault-rollback",
            "complete": True,
            "gates": {
                "foundation": True,
                "backup": True,
                "trust": True,
                "updater2": True,
                "oak": True,
                "services": True,
                "scenario": True,
            },
            "oak": {
                "port": "2-1.1",
                "vendorId": "03e7",
                "productId": "f63b",
                "speedMbps": 5000,
                "mxidSha256": "3" * 64,
                "attached": True,
                "bound": True,
            },
            "services": services,
            "runtimePids": {"candidate": 4100, "restored": 4200},
            "slots": {
                "current": baseline_slot,
                "previous": candidate_slot,
                "currentVersion": "0.1.120",
                "release": {
                    "schemaVersion": 2,
                    "bomDigest": "sha256:" + baseline_digest,
                    "agentVersion": "0.1.120",
                    "releaseSequence": 1,
                    "artifactDigest": "sha256:" + "e" * 64,
                    "componentSha": "f" * 40,
                    "configSchema": "12",
                    "publisherKeyId": "release-iq9075-dev-2026-09-01",
                },
                "previousRelease": {
                    "schemaVersion": 2,
                    "bomDigest": bom["bomDigest"],
                    **fleet_release,
                },
            },
            "updater": {
                "capabilityAvailable": True,
                "authenticatedHelper": True,
                "reason": "READY",
                "updaterVersion": "0.2.0",
                "update": update,
            },
            "antiReplay": {
                "schemaVersion": 4,
                "semanticSha256": "0" * 64,
                "maximumCommandSequence": 2,
                "currentReleaseSequence": "1",
                "currentBomDigest": "sha256:" + baseline_digest,
                "latest": {
                    "commandId": update["commandId"],
                    "sequence": 2,
                    "phase": "ROLLED_BACK",
                    "bomDigest": bom["bomDigest"],
                    "releaseSequence": 2,
                    "healthDeadline": None,
                },
            },
        }
        fleet_evidence_path = source / "evidence.json"
        fleet_evidence_path.write_text(
            json.dumps(fleet_evidence, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        cleanup_path = source / "cleanup-evidence.json"
        cleanup_path.write_text(
            json.dumps(
                cleanup_evidence(fleet_manifest["runId"]),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "fleet_manifest": fleet_manifest_path,
            "fleet_evidence": fleet_evidence_path,
            "cleanup_evidence": cleanup_path,
            "artifact": artifact_path,
            "bom": bom_path,
        }

    @classmethod
    def _candidate_physical_fixture(
        cls, root: Path, *, component_sha: str
    ) -> tuple[dict[str, object], Path, Path]:
        summary, manifest_path, result_path = cls._physical_fixture(
            root, component_sha=component_sha
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        fleet_manifest_path = root / manifest["fleetManifest"]["file"]
        fleet_evidence_path = root / manifest["fleetEvidence"]["file"]
        fleet_manifest = json.loads(fleet_manifest_path.read_text(encoding="utf-8"))
        fleet_evidence = json.loads(fleet_evidence_path.read_text(encoding="utf-8"))
        soak_path = root / manifest["oakSoak"]["file"]
        soak = json.loads(soak_path.read_text(encoding="utf-8"))
        run_id = fleet_manifest["runId"]
        bom_digest = fleet_manifest["scenario"]["expectedBomDigest"]
        candidate_slot = f"/opt/nuv-agent/candidates/{run_id}-{bom_digest[7:]}"
        control_sha = "c" * 64
        soak.update(
            {"schemaVersion": 3, "runId": run_id, "slotKind": "candidate"}
        )
        soak["startedAt"] = "2026-09-03T10:04:00Z"
        soak["runtimeIdentity"].update(
            {
                "pythonPath": "/usr/bin/python3",
                "sitePackagesPath": candidate_slot
                + "/venv/lib/python3.12/site-packages",
                "buildInfoPath": candidate_slot
                + "/venv/lib/python3.12/site-packages/nuvion_app/build_info.py",
                "candidateSlot": candidate_slot,
                "controlMarkerSha256": control_sha,
            }
        )
        soak_path.write_text(
            json.dumps(soak, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        artifact_path = (
            root / "nuv-agent_0.1.121_iq9075-aarch64.agent-bundle.tar.gz"
        )
        bom_path = root / manifest["testedBom"]["file"]
        slots = {
            "current": fleet_evidence["slots"]["current"],
            "previous": fleet_evidence["slots"]["previous"],
        }
        anti_replay = {
            "schemaVersion": 4,
            "semanticSha256": "0" * 64,
            "maximumCommandSequence": 2,
            "currentReleaseSequence": "1",
            "currentBomDigest": fleet_evidence["slots"]["release"]["bomDigest"],
            "latest": {
                "commandId": fleet_manifest["scenario"]["expectedCommandId"],
                "sequence": 2,
                "phase": "ROLLED_BACK",
                "bomDigest": bom_digest,
                "releaseSequence": 2,
                "healthDeadline": None,
            },
        }
        before_runtime = {
            "pid": 4200,
            "startTicks": 42000,
            "bootId": "11111111-1111-4111-8111-111111111111",
            "activeSlot": slots["current"],
        }
        after_runtime = {**before_runtime, "pid": 4400, "startTicks": 44000}
        post = {
            "restoredAt": "2026-09-03T10:06:00Z",
            "slots": slots,
            "antiReplay": anti_replay,
            "oak": fleet_evidence["oak"],
            "runtime": after_runtime,
        }
        release_trees = release_tree_evidence(slots)
        cleanup = cleanup_evidence(run_id)
        candidate_evidence = {
            "schemaVersion": 1,
            "kind": "nuvion-iq9075-candidate-soak-evidence",
            "protocolVersion": FLEET_E2E.PROTOCOL_VERSION,
            "runId": run_id,
            "startedAt": "2026-09-03T10:03:30Z",
            "completedAt": "2026-09-03T10:06:30Z",
            "complete": True,
            "outcome": {"status": "passed", "errorCode": None},
            "candidate": {
                "slotKind": "candidate",
                "slot": candidate_slot,
                "bomDigest": bom_digest,
                "bundleSha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
                "bomSha256": hashlib.sha256(bom_path.read_bytes()).hexdigest(),
                "harnessSha256": hashlib.sha256(
                    (ROOT / "packaging/dev/test-iq9075.sh").read_bytes()
                ).hexdigest(),
                "controlMarkerSha256": control_sha,
            },
            "fleetEvidenceSha256": hashlib.sha256(
                fleet_evidence_path.read_bytes()
            ).hexdigest(),
            "rawEvidenceSha256": hashlib.sha256(soak_path.read_bytes()).hexdigest(),
            "rawEvidence": soak,
            "cleanupEvidenceSha256": canonical_sha256(cleanup),
            "cleanupEvidence": cleanup,
            "executionProof": candidate_execution_proof(run_id),
            "collectorProof": candidate_collector_proof(run_id),
            "terminationProof": candidate_termination_proof(run_id),
            "productionRestoration": production_restoration_evidence(
                fleet_manifest
            ),
            "pre": {
                "slots": slots,
                "antiReplay": anti_replay,
                "oak": fleet_evidence["oak"],
                "runtime": before_runtime,
                "persistentState": persistent_state_evidence(),
                "releaseTrees": copy.deepcopy(release_trees),
            },
            "post": {
                **post,
                "persistentState": persistent_state_evidence(),
                "releaseTrees": release_trees,
            },
            "gates": {
                "signedRollbackTerminal": True,
                "candidateBound": True,
                "rawEvidencePreserved": True,
                "slotsUnchanged": True,
                "releaseTreesUnchanged": True,
                "antiReplayUnchanged": True,
                "oakIdentityUnchanged": True,
                "freshBaselineProcess": True,
                "harnessBytesPinned": True,
                "harnessCopyRemoved": True,
                "resourceLimitsApplied": True,
                "boundedOutput": True,
                "persistentStateReadOnly": True,
                "persistentStateUnchanged": True,
                "productionTrustRestored": True,
                "trustedSoakDuration": True,
                "continuousUidIsolation": True,
                "cgroupTerminated": True,
                "harnessPassed": True,
            },
        }
        candidate_path = root / "iq9075-v0.1.121-candidate-soak-evidence.json"
        candidate_path.write_text(
            json.dumps(candidate_evidence, sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        cleanup_path = root / "iq9075-v0.1.121-cleanup-evidence.json"
        cleanup_path.write_text(
            json.dumps(cleanup, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        manifest["oakSoak"]["sha256"] = hashlib.sha256(soak_path.read_bytes()).hexdigest()
        manifest["candidateSoak"] = {
            "file": candidate_path.name,
            "sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        }
        manifest["cleanupEvidence"] = {
            "file": cleanup_path.name,
            "sha256": hashlib.sha256(cleanup_path.read_bytes()).hexdigest(),
        }
        manifest["startedAt"] = soak["startedAt"]
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["schemaVersion"] = 3
        result["candidateRestore"] = candidate_evidence["post"]
        result["cleanupEvidenceSha256"] = hashlib.sha256(
            cleanup_path.read_bytes()
        ).hexdigest()
        result["manifestSha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        result_path.write_text(
            json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        summary["harnessManifest"]["sha256"] = hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest()
        summary["harnessResult"]["sha256"] = hashlib.sha256(
            result_path.read_bytes()
        ).hexdigest()
        summary["cleanupEvidence"] = manifest["cleanupEvidence"]
        return summary, manifest_path, result_path

    def test_tag_push_is_secret_zero_and_default_branch_starts_publisher(self) -> None:
        self.assertIn("on:\n  push:\n    tags:", self.request)
        self.assertNotIn("${{ secrets.", self.request)
        self.assertNotIn("contents: write", self.request)
        self.assertNotIn("environment:", self.request)
        self.assertIn('workflows: ["release-request"]', self.publish)
        self.assertIn("group: release-publisher-global", self.publish)
        self.assertIn('PYTHONDONTWRITEBYTECODE: "1"', self.publish.split("jobs:", 1)[0])
        trigger = self.publish.split("jobs:", maxsplit=1)[0]
        self.assertNotIn("  push:\n", trigger)
        self.assertIn('publisher workflow_dispatch must run from main', self.publish)
        self.assertIn("github.event.workflow_run.head_sha", self.publish)
        self.assertIn("ref: ${{ github.workflow_sha }}", self.publish)
        self.assertIn('[ "$WORKFLOW_SHA" = "$DEFAULT_BRANCH_SHA" ]', self.publish)
        self.assertIn("path: settings-evidence", self.publish)
        self.assertIn('--trusted-publisher-sha "$TRUSTED_PUBLISHER_SHA"', self.publish)

    def test_every_credential_job_uses_environment_and_trusted_checkout(self) -> None:
        job_names = [
            "github-release-publish",
            "homebrew-publish",
            "apt-publish",
            "iq9075-ota-publish",
        ]
        expected_environment = {
            "github-release-publish": "homebrew-release",
            "homebrew-publish": "homebrew-release",
            "apt-publish": "apt-release",
            "iq9075-ota-publish": "iq9075-release",
        }
        credential_reference = {
            "github-release-publish": "${{ github.token }}",
            "homebrew-publish": "${{ secrets.",
            "apt-publish": "${{ secrets.",
            "iq9075-ota-publish": "${{ secrets.",
        }
        verifier_count = {
            "github-release-publish": 2,
            "homebrew-publish": 2,
            "apt-publish": 6,
            "iq9075-ota-publish": 7,
        }
        for name in job_names:
            section = self._job(name)
            with self.subTest(job=name):
                self.assertIn(
                    f"environment: {expected_environment[name]}", section
                )
                self.assertIn("Checkout trusted publisher only", section)
                self.assertIn(
                    "ref: ${{ needs.release-preflight.outputs.trusted_publisher_sha }}",
                    section,
                )
                self.assertIn("path: publisher", section)
                self.assertIn("path: settings-evidence", section)
                self.assertEqual(
                    section.count("verify-release-settings-attestation.py"),
                    verifier_count[name],
                )
                self.assertEqual(
                    section.count("--publisher-root publisher"), verifier_count[name]
                )
                self.assertEqual(
                    section.count(
                        "--executing-workflow settings-evidence/.github/workflows/release-publish.yml"
                    ),
                    verifier_count[name],
                )
                self.assertIn(credential_reference[name], section)
                self.assertIn("actions: read", section)
                self.assertIn("checks: read", section)
                if name == "github-release-publish":
                    self.assertIn("contents: write", section)
                    self.assertNotIn("GITHUB_RELEASE_TOKEN", section)
                else:
                    self.assertNotIn("contents: write", section)
                self.assertNotIn("ref: ${{ needs.release-preflight.outputs.release_tag }}", section)
                self.assertNotIn("build-agent-bundle.sh", section)

        credential_steps = {
            "github-release-publish": [
                "Finalize exact immutable GitHub release before live channels"
            ],
            "homebrew-publish": ["Update Homebrew tap with trusted publisher"],
            "apt-publish": [
                "Import APT signing key",
                "Authenticate APT-only GCP publisher",
                "Setup gcloud",
                "Publish exact deb set with trusted publisher",
                "Publish final distribution promotion after both live channels",
            ],
            "iq9075-ota-publish": [
                "Authenticate OTA-only GCP publisher",
                "Setup gcloud",
                "Atomically reserve exact release sequence",
                "Sign exact bundle BOM with trusted signer",
                "Publish verified exact bundle with trusted publisher",
                "Generate and atomically publish final OTA promotion",
            ],
        }
        for job, names in credential_steps.items():
            for name in names:
                with self.subTest(job=job, credential_step=name):
                    self._assert_immediate_revalidation(self._job(job), name)

        self.assertEqual(
            self.publish.count(
                "publisher/packaging/release/"
                "revalidate-live-release-authorization.sh"
            ),
            13,
        )

    def test_homebrew_token_never_enters_argv_or_git_config(self) -> None:
        homebrew = self._job("homebrew-publish")
        self.assertNotIn("x-access-token:${GH_TOKEN}@", homebrew)
        self.assertNotIn("git remote set-url", homebrew)
        self.assertIn(
            'git clone "https://github.com/${TAP_REPO}.git" tap', homebrew
        )
        self.assertIn(
            '[ "$(git remote get-url origin)" = '
            '"https://github.com/${TAP_REPO}.git" ]',
            homebrew,
        )
        self.assertIn('askpass="$(mktemp ', homebrew)
        self.assertIn("chmod 0700 \"$askpass\"", homebrew)
        self.assertIn("trap cleanup_askpass EXIT", homebrew)
        self.assertIn("GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=", homebrew)
        self.assertIn('*Password*) printf \'%s\\n\' "${GH_TOKEN:?}"', homebrew)

    def test_face_publisher_is_main_only_pinned_and_revalidates_each_credential(self) -> None:
        self.assertIn("environment: face-artifacts-release", self.face)
        self.assertIn("group: face-artifacts-global-publisher", self.face)
        self.assertIn('[ "$GITHUB_REF" = "refs/heads/main" ]', self.face)
        self.assertIn("persist-credentials: false", self.face)
        self.assertIn("--trusted-additional-workflow", self.face)
        self.assertIn(".github/workflows/publish-face-artifacts.yml", self.face)
        self.assertIn('"immutable": True', self.face)
        self.assertIn("verify-release-source.py", self.face)
        self.assertEqual(self.face.count("face-artifact-manifest.py verify"), 2)
        self.assertIn("face-artifact-manifest.json.asc", self.face)
        self.assertIn("immutable face release asset set is not exact", self.face)
        self.assertEqual(
            self.face.count("verify-release-settings-attestation.py"), 5
        )
        for name in (
            "Checkout signed face release source as data",
            "Download exact face artifacts from immutable GitHub release",
            "Authenticate face-artifact-only GCP publisher",
            "Setup gcloud",
            "Publish to model-scoped GCS paths and update pointers",
        ):
            with self.subTest(credential_step=name):
                self._assert_immediate_revalidation(
                    self.face,
                    name,
                    require_live_authorization=False,
                )
        self.assertNotIn('"${{ inputs.', self.face)

    def test_apt_signing_uses_owned_mode_0600_passphrase_file(self) -> None:
        apt_job = self._job("apt-publish")
        apt_script = (ROOT / "packaging/apt/publish-gcs.sh").read_text(
            encoding="utf-8"
        )
        self.assertEqual(apt_job.count("secrets.APT_GPG_PASSPHRASE"), 1)
        self.assertIn('chmod 0600 "$passphrase_file"', apt_job)
        self.assertIn('--passphrase-file "$passphrase_file"', apt_job)
        self.assertNotIn('--passphrase "$GPG_PASSPHRASE"', apt_job)
        self.assertIn("trap cleanup_import_passphrase EXIT", apt_job)
        self.assertIn("trap - EXIT", apt_job)
        self.assertIn("trap cleanup_passphrase EXIT", apt_job)
        self.assertIn("Remove APT passphrase file on every exit path", apt_job)
        self.assertIn('APT_RUNTIME_ROOT: ${{ github.workspace }}/apt-runtime', apt_job)
        self.assertIn('file_mode" != "600"', apt_script)
        self.assertIn('file_owner" != "$(id -u)"', apt_script)
        self.assertEqual(
            apt_script.count('-batch -passphrase-file="$APTLY_PASSPHRASE_FILE"'),
            2,
        )

    def test_apt_import_failure_removes_passphrase_before_environment_handoff(self) -> None:
        apt_job = self._job("apt-publish")
        import_step = next(
            step
            for step in self._steps(apt_job)
            if step.startswith("Import APT signing key\n")
        )
        raw_script = import_step.split("        run: |\n", 1)[1]
        script_lines: list[str] = []
        for line in raw_script.splitlines():
            if line.startswith("      - name: "):
                break
            if line.startswith("          "):
                script_lines.append(line[10:])
            elif line:
                self.fail(f"unexpected workflow script indentation: {line}")
        script = "\n".join(script_lines) + "\n"
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            fake_bin = root / "bin"
            runner_temp = root / "runner-temp"
            gpg_home = root / "gnupg"
            github_env = root / "github-env"
            fake_bin.mkdir()
            runner_temp.mkdir()
            fake_gpg = fake_bin / "gpg"
            fake_gpg.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "if [[ \" $* \" == *\" --sign \"* ]]; then exit 19; fi\n"
                "cat >/dev/null\n",
                encoding="utf-8",
            )
            fake_gpg.chmod(0o755)
            fake_gpgconf = fake_bin / "gpgconf"
            fake_gpgconf.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            fake_gpgconf.chmod(0o755)
            result = subprocess.run(
                ["bash"],
                input=script,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={
                    **os.environ,
                    "PATH": f"{fake_bin}:{os.environ['PATH']}",
                    "GNUPGHOME": str(gpg_home),
                    "RUNNER_TEMP": str(runner_temp),
                    "GITHUB_ENV": str(github_env),
                    "GPG_PRIVATE_KEY": "private-key-material",
                    "GPG_PASSPHRASE": "passphrase-material",
                    "GPG_KEY_ID": "TEST-KEY",
                },
                check=False,
            )
            self.assertEqual(result.returncode, 19, result.stderr)
            self.assertEqual(list(runner_temp.glob("nuv-aptly-passphrase.*")), [])
            self.assertFalse(github_env.exists())

    def test_github_is_immutable_before_rerunnable_live_channel_promotion(self) -> None:
        github = self._job("github-release-publish")
        homebrew = self._job("homebrew-publish")
        apt = self._job("apt-publish")
        self.assertIn("--phase finalize", github)
        self.assertNotIn("--phase stage", self.publish)
        self.assertIn("distribution-source-plan", github)
        self.assertIn("needs: [release-preflight, github-release-publish]", homebrew)
        self.assertIn("homebrewFormula", (
            ROOT / "packaging/release/generate-release-promotion.py"
        ).read_text(encoding="utf-8"))
        self.assertIn("github-release-publish, homebrew-publish", apt)
        self.assertIn('NAME="nuv_agent-${VERSION}-distribution-promotion.json"', apt)
        self.assertIn("releases/promotions/distribution/${VERSION}.json", apt)
        self.assertLess(
            self.publish.index("  github-release-publish:"),
            self.publish.index("  homebrew-publish:"),
        )
        self.assertLess(
            self.publish.index("  homebrew-publish:"), self.publish.index("  apt-publish:")
        )
        self.assertNotIn("softprops/action-gh-release", self.publish)
        self.assertIn(
            "tag_object_sha: ${{ steps.identity.outputs.tag_object_sha }}",
            self.publish,
        )
        self.assertIn('--tag-object-sha "$TAG_OBJECT_SHA"', self.publish)

    def test_v121_release_is_blocked_until_live_release_gates_succeed(self) -> None:
        preflight = self.publish.split("  release-preflight:", maxsplit=1)[1].split(
            "  release-build:", maxsplit=1
        )[0]
        self.assertIn("verify-release-readiness.py", preflight)
        self.assertIn("release-readiness.json", preflight)
        self.assertIn("verify-agent-release-gate.py", preflight)
        self.assertIn("--component-sha \"$COMPONENT_SHA\"", preflight)
        self.assertIn("checks: read", preflight)
        self.assertIn("--candidate-workflow release-source/", preflight)
        self.assertIn("--trusted-workflow publisher/", preflight)
        self.assertIn("--gate-workflow-sha256 \"$GATE_WORKFLOW_SHA256\"", preflight)
        self.assertIn("--signer-directory publisher/packaging/release/", preflight)
        self.assertLess(
            preflight.index("verify-agent-release-gate.py"),
            preflight.index("verify-release-readiness.py"),
        )
        self.assertLess(
            self.publish.index("verify-release-readiness.py"),
            self.publish.index("  release-build:"),
        )
        with self.assertRaises(READINESS.ReadinessError):
            READINESS.verify_readiness(
                ROOT / "packaging/release/release-readiness.json",
                version="0.1.121",
            )
        with tempfile.TemporaryDirectory() as raw_root:
            blocked = Path(raw_root) / "readiness.json"
            blocked.write_text(
                json.dumps(
                    {
                        "schemaVersion": 2,
                        "releases": {
                            "0.1.121": {
                                "status": "BLOCKED",
                                "blockers": [{"id": "TRANSFORMERS-REGRESSION"}],
                                "evidence": None,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(READINESS.ReadinessError):
                READINESS.verify_readiness(
                    blocked,
                    version="0.1.121",
                )
        with self.assertRaises(READINESS.ReadinessError):
            READINESS.verify_readiness(
                ROOT / "packaging/release/release-readiness.json",
                version="0.1.120",
            )

    def test_exact_sha_release_gate_binds_check_app_workflow_and_run(self) -> None:
        component_sha = "a" * 40
        repository = "plaid-ai/NUV-AGENT"
        check = {
            "id": 7002,
            "name": "agent-release-gate",
            "head_sha": component_sha,
            "status": "completed",
            "conclusion": "success",
            "details_url": (
                "https://github.com/plaid-ai/NUV-AGENT/actions/runs/8003/job/9004"
            ),
            "app": {"id": 15368, "slug": "github-actions"},
            "check_suite": {"id": 6001},
        }
        run = {
            "id": 8003,
            "check_suite_id": 6001,
            "head_sha": component_sha,
            "name": "agent-release-gate",
            "path": ".github/workflows/agent-release-gate.yml",
            "status": "completed",
            "conclusion": "success",
            "event": "workflow_dispatch",
            "head_branch": "main",
            "repository": {"full_name": repository},
        }

        evidence = RELEASE_GATE.verify_release_gate(
            repository=repository,
            component_sha=component_sha,
            required_context="agent-release-gate",
            required_integration_id=15368,
            workflow_sha256="b" * 64,
            check_runs=[check],
            workflow_run=lambda run_id: run if run_id == 8003 else {},
        )

        self.assertEqual(evidence["componentSha"], component_sha)
        self.assertEqual(evidence["workflowRunId"], 8003)
        self.assertEqual(evidence["checkRunId"], 7002)
        self.assertEqual(evidence["checkSuiteId"], 6001)

        for event, branch in (("pull_request", "main"), ("workflow_dispatch", "dev")):
            with self.subTest(event=event, branch=branch), self.assertRaisesRegex(
                RELEASE_GATE.ReleaseGateError,
                "exact release workflow run",
            ):
                RELEASE_GATE.verify_release_gate(
                    repository=repository,
                    component_sha=component_sha,
                    required_context="agent-release-gate",
                    required_integration_id=15368,
                    workflow_sha256="b" * 64,
                    check_runs=[check],
                    workflow_run=lambda _run_id, event=event, branch=branch: {
                        **run,
                        "event": event,
                        "head_branch": branch,
                    },
                )

    def test_live_gate_revalidation_requires_exact_preflight_evidence(self) -> None:
        evidence = {
            "workflowRunId": 101,
            "checkRunId": 102,
            "checkSuiteId": 103,
            "workflowSha256": "a" * 64,
        }
        RELEASE_GATE.verify_expected_evidence(
            evidence,
            run_id=101,
            check_id=102,
            check_suite_id=103,
            workflow_sha256="a" * 64,
        )
        mutations = (
            {**evidence, "workflowRunId": 999},
            {**evidence, "checkRunId": 999},
            {**evidence, "checkSuiteId": 999},
            {**evidence, "workflowSha256": "b" * 64},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(
                RELEASE_GATE.ReleaseGateError
            ):
                RELEASE_GATE.verify_expected_evidence(
                    mutation,
                    run_id=101,
                    check_id=102,
                    check_suite_id=103,
                    workflow_sha256="a" * 64,
                )

        helper = (
            ROOT
            / "packaging/release/revalidate-live-release-authorization.sh"
        )
        self.assertEqual(helper.stat().st_mode & 0o777, 0o755)
        script = helper.read_text(encoding="utf-8")
        self.assertIn("verify-agent-release-gate.py", script)
        self.assertIn("verify-release-readiness.py", script)
        for option in (
            "--expected-run-id",
            "--expected-check-id",
            "--expected-check-suite-id",
            "--expected-workflow-sha256",
        ):
            self.assertIn(option, script)

    def test_ready_decision_requires_signed_physical_and_live_gate_evidence(self) -> None:
        component_sha = "a" * 40
        fingerprint = "9A07D327F3ADF6F452A4BF0055E5CAF706571888"
        gate_evidence = {
            "componentSha": component_sha,
            "workflow": ".github/workflows/agent-release-gate.yml",
            "workflowSha256": "b" * 64,
            "workflowRunId": 101,
            "checkRunId": 102,
            "checkSuiteId": 103,
            "context": "agent-release-gate",
            "integrationId": 15368,
        }
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            physical_document, _manifest, _result = self._candidate_physical_fixture(
                root,
                component_sha=component_sha,
            )
            evidence = root / "iq9075-v0.1.121-physical-evidence.json"
            signature = root / "iq9075-v0.1.121-physical-evidence.json.asc"
            evidence.write_text(
                json.dumps(physical_document, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            signature.write_text("detached-signature\n", encoding="utf-8")
            readiness = root / "release-readiness.json"
            readiness.write_text(
                json.dumps(
                    {
                        "schemaVersion": 2,
                        "releases": {
                            "0.1.121": {
                                "status": "READY",
                                "blockers": [],
                                "evidence": {
                                    "componentSha": component_sha,
                                    "agentReleaseGate": gate_evidence,
                                    "iq9075Physical": {
                                        "evidenceFile": evidence.name,
                                        "evidenceSha256": hashlib.sha256(
                                            evidence.read_bytes()
                                        ).hexdigest(),
                                        "signatureFile": signature.name,
                                        "signatureSha256": hashlib.sha256(
                                            signature.read_bytes()
                                        ).hexdigest(),
                                        "signerFingerprint": fingerprint,
                                    },
                                },
                            }
                        },
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(
                READINESS,
                "_verify_detached_signature",
                return_value=fingerprint,
            ) as verify_signature:
                verified = READINESS.verify_readiness(
                    readiness,
                    version="0.1.121",
                    component_sha=component_sha,
                    gate_evidence=gate_evidence,
                    security_policy=(
                        ROOT / "packaging/release/release-security-policy.json"
                    ),
                    signer_directory=(
                        ROOT / "packaging/release/trusted-tag-signers"
                    ),
                    candidate_harness=ROOT / "packaging/dev/test-iq9075.sh",
                )
            self.assertEqual(
                verified["physical_artifact_sha256"],
                json.loads(_result.read_text(encoding="utf-8"))["artifactSha256"],
            )
            self.assertEqual(
                verified["physical_bom_sha256"],
                json.loads(_result.read_text(encoding="utf-8"))["bomSha256"],
            )
            verify_signature.assert_called_once()

            mismatched_gate = {**gate_evidence, "workflowRunId": 999}
            with self.assertRaisesRegex(
                READINESS.ReadinessError,
                "does not match live GitHub proof",
            ):
                READINESS.verify_readiness(
                    readiness,
                    version="0.1.121",
                    component_sha=component_sha,
                    gate_evidence=mismatched_gate,
                    security_policy=(
                        ROOT / "packaging/release/release-security-policy.json"
                    ),
                    signer_directory=(
                        ROOT / "packaging/release/trusted-tag-signers"
                    ),
                    candidate_harness=ROOT / "packaging/dev/test-iq9075.sh",
                )

    def test_fleet_runtime_assembler_needs_no_media_soak_evidence(self) -> None:
        component_sha = "a" * 40
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            inputs = self._fleet_runtime_fixture(
                root, component_sha=component_sha
            )
            output = root / "output"
            output.mkdir(mode=0o700)
            assembled = FLEET_RUNTIME_EVIDENCE.assemble(
                fleet_manifest_path=inputs["fleet_manifest"],
                fleet_evidence_path=inputs["fleet_evidence"],
                cleanup_evidence_path=inputs["cleanup_evidence"],
                artifact_path=inputs["artifact"],
                bom_path=inputs["bom"],
                candidate_fleet_runner=ROOT
                / "packaging/dev/run-iq9075-fleet-e2e.py",
                candidate_board_tool=ROOT / "packaging/dev/iq9075-board-e2e.py",
                security_policy_path=ROOT
                / "packaging/release/release-security-policy.json",
                output_directory=output,
                version="0.1.121",
                component_sha=component_sha,
            )
            self.assertEqual(len(list(output.iterdir())), 5)
            summary_path = Path(assembled["summary"])
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            gate = summary["runtimeGate"]
            self.assertEqual(gate["terminalPhase"], "ROLLED_BACK")
            self.assertEqual(gate["antiReplay"]["maximumCommandSequence"], 2)
            self.assertEqual(
                gate["antiReplay"]["currentReleaseSequence"], "1"
            )
            self.assertEqual(
                gate["healthDecision"]["health"], "LKG_RESTORED"
            )
            self.assertEqual(gate["cleanup"]["phase"], "RESTORED")
            serialized = json.dumps(summary, sort_keys=True)
            for forbidden in (
                "oakSoak",
                "candidateSoak",
                "durationSeconds",
                "gstreamer",
                "webrtc",
                "splitmux",
            ):
                self.assertNotIn(forbidden, serialized)
            security = json.loads(
                (ROOT / "packaging/release/release-security-policy.json").read_text(
                    encoding="utf-8"
                )
            )
            verified = READINESS._validate_fleet_runtime_documents(
                policy_path=output / "release-readiness.json",
                version="0.1.121",
                component_sha=component_sha,
                summary=summary,
                security=security,
                candidate_fleet_runner=ROOT
                / "packaging/dev/run-iq9075-fleet-e2e.py",
                candidate_board_tool=ROOT / "packaging/dev/iq9075-board-e2e.py",
            )
            self.assertEqual(
                verified["runtime_artifact_sha256"], assembled["artifactSha256"]
            )
            self.assertEqual(verified["runtime_bom_sha256"], assembled["bomSha256"])
            tampered_summary = copy.deepcopy(summary)
            tampered_summary["runtimeGate"]["antiReplay"][
                "maximumCommandSequence"
            ] = 999
            with self.assertRaisesRegex(
                READINESS.ReadinessError, "differs from raw evidence"
            ):
                READINESS._validate_fleet_runtime_documents(
                    policy_path=output / "release-readiness.json",
                    version="0.1.121",
                    component_sha=component_sha,
                    summary=tampered_summary,
                    security=security,
                    candidate_fleet_runner=ROOT
                    / "packaging/dev/run-iq9075-fleet-e2e.py",
                    candidate_board_tool=ROOT
                    / "packaging/dev/iq9075-board-e2e.py",
                )

    def test_fleet_runtime_assembler_fails_closed_on_runtime_or_cleanup_drift(
        self,
    ) -> None:
        component_sha = "a" * 40

        def mutate_cleanup(inputs: dict[str, Path]) -> None:
            cleanup = json.loads(inputs["cleanup_evidence"].read_text(encoding="utf-8"))
            cleanup["proof"]["trustStagingAbsent"] = False
            inputs["cleanup_evidence"].write_text(
                json.dumps(cleanup, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )

        def mutate_sequence(inputs: dict[str, Path]) -> None:
            evidence = json.loads(inputs["fleet_evidence"].read_text(encoding="utf-8"))
            evidence["updater"]["update"]["sequence"] = False
            inputs["fleet_evidence"].write_text(
                json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )

        def mutate_anti_replay_snapshot(inputs: dict[str, Path]) -> None:
            evidence = json.loads(inputs["fleet_evidence"].read_text(encoding="utf-8"))
            evidence["antiReplay"]["maximumCommandSequence"] = 999
            inputs["fleet_evidence"].write_text(
                json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )

        def mutate_service(inputs: dict[str, Path]) -> None:
            evidence = json.loads(inputs["fleet_evidence"].read_text(encoding="utf-8"))
            evidence["services"]["nuv-agent.service"]["active"] = False
            inputs["fleet_evidence"].write_text(
                json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )

        def mutate_slot(inputs: dict[str, Path]) -> None:
            evidence = json.loads(inputs["fleet_evidence"].read_text(encoding="utf-8"))
            evidence["slots"]["current"] = evidence["slots"]["previous"]
            inputs["fleet_evidence"].write_text(
                json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )

        def mutate_artifact(inputs: dict[str, Path]) -> None:
            inputs["artifact"].write_bytes(b"different candidate artifact")

        def mutate_to_valid_commit(inputs: dict[str, Path]) -> None:
            manifest = json.loads(inputs["fleet_manifest"].read_text(encoding="utf-8"))
            evidence = json.loads(inputs["fleet_evidence"].read_text(encoding="utf-8"))
            scenario = manifest["scenario"]
            scenario["type"] = "commit"
            scenario["holdSeconds"] = 0
            evidence["scenario"] = "commit"
            evidence["runtimePids"] = None
            evidence["slots"]["release"], evidence["slots"]["previousRelease"] = (
                evidence["slots"]["previousRelease"],
                evidence["slots"]["release"],
            )
            evidence["slots"].update(
                {
                    "current": "releases/" + scenario["expectedBomDigest"][7:],
                    "previous": scenario["expectedPreviousSlot"],
                    "currentVersion": scenario["release"]["agentVersion"],
                }
            )
            update = evidence["updater"]["update"]
            for field in ("errorCode", "rollbackSlot", "rollbackVersion"):
                update.pop(field)
            update.update(
                {
                    "phase": "COMMITTED",
                    "updatePhase": "COMMITTED",
                    "slot": "releases/" + scenario["expectedBomDigest"][7:],
                    "health": "FUNCTIONAL_HEALTHY",
                    "functionalHealth": "FUNCTIONAL_HEALTHY",
                }
            )
            evidence["antiReplay"].update(
                {
                    "currentReleaseSequence": "2",
                    "currentBomDigest": scenario["expectedBomDigest"],
                    "latest": {
                        "commandId": update["commandId"],
                        "sequence": update["sequence"],
                        "phase": "COMMITTED",
                        "bomDigest": update["bomDigest"],
                        "releaseSequence": update["releaseSequence"],
                        "healthDeadline": None,
                    },
                }
            )
            for path, value in (
                (inputs["fleet_manifest"], manifest),
                (inputs["fleet_evidence"], evidence),
            ):
                path.write_text(
                    json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="utf-8",
                )

        for label, mutation in (
            ("cleanup", mutate_cleanup),
            ("anti-replay", mutate_sequence),
            ("anti-replay-snapshot", mutate_anti_replay_snapshot),
            ("service", mutate_service),
            ("slot", mutate_slot),
            ("artifact", mutate_artifact),
            ("commit-does-not-prove-rollback", mutate_to_valid_commit),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as raw_root:
                root = Path(raw_root)
                inputs = self._fleet_runtime_fixture(
                    root, component_sha=component_sha
                )
                mutation(inputs)
                if label == "commit-does-not-prove-rollback":
                    FLEET_E2E.validate_final_evidence(
                        json.loads(
                            inputs["fleet_evidence"].read_text(encoding="utf-8")
                        ),
                        json.loads(
                            inputs["fleet_manifest"].read_text(encoding="utf-8")
                        ),
                    )
                output = root / "output"
                output.mkdir(mode=0o700)
                with self.assertRaises(
                    FLEET_RUNTIME_EVIDENCE.AssemblyError
                ) as caught:
                    FLEET_RUNTIME_EVIDENCE.assemble(
                        fleet_manifest_path=inputs["fleet_manifest"],
                        fleet_evidence_path=inputs["fleet_evidence"],
                        cleanup_evidence_path=inputs["cleanup_evidence"],
                        artifact_path=inputs["artifact"],
                        bom_path=inputs["bom"],
                        candidate_fleet_runner=ROOT
                        / "packaging/dev/run-iq9075-fleet-e2e.py",
                        candidate_board_tool=ROOT
                        / "packaging/dev/iq9075-board-e2e.py",
                        security_policy_path=ROOT
                        / "packaging/release/release-security-policy.json",
                        output_directory=output,
                        version="0.1.121",
                        component_sha=component_sha,
                    )
                if label == "commit-does-not-prove-rollback":
                    self.assertIn(
                        "does not prove terminal rollback",
                        str(caught.exception.__cause__),
                    )
                self.assertEqual(list(output.iterdir()), [])

    def test_ready_decision_accepts_signed_fleet_runtime_evidence(self) -> None:
        component_sha = "a" * 40
        fingerprint = "9A07D327F3ADF6F452A4BF0055E5CAF706571888"
        gate_evidence = {
            "componentSha": component_sha,
            "workflow": ".github/workflows/agent-release-gate.yml",
            "workflowSha256": "b" * 64,
            "workflowRunId": 101,
            "checkRunId": 102,
            "checkSuiteId": 103,
            "context": "agent-release-gate",
            "integrationId": 15368,
        }
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            inputs = self._fleet_runtime_fixture(
                root, component_sha=component_sha
            )
            output = root / "output"
            output.mkdir(mode=0o700)
            assembled = FLEET_RUNTIME_EVIDENCE.assemble(
                fleet_manifest_path=inputs["fleet_manifest"],
                fleet_evidence_path=inputs["fleet_evidence"],
                cleanup_evidence_path=inputs["cleanup_evidence"],
                artifact_path=inputs["artifact"],
                bom_path=inputs["bom"],
                candidate_fleet_runner=ROOT
                / "packaging/dev/run-iq9075-fleet-e2e.py",
                candidate_board_tool=ROOT / "packaging/dev/iq9075-board-e2e.py",
                security_policy_path=ROOT
                / "packaging/release/release-security-policy.json",
                output_directory=output,
                version="0.1.121",
                component_sha=component_sha,
            )
            evidence = Path(assembled["summary"])
            signature = output / f"{evidence.name}.asc"
            signature.write_text("detached-signature\n", encoding="utf-8")
            signed_identity = {
                "evidenceFile": evidence.name,
                "evidenceSha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
                "signatureFile": signature.name,
                "signatureSha256": hashlib.sha256(signature.read_bytes()).hexdigest(),
                "signerFingerprint": fingerprint,
            }
            readiness = output / "release-readiness.json"
            readiness.write_text(
                json.dumps(
                    {
                        "schemaVersion": 2,
                        "releases": {
                            "0.1.121": {
                                "status": "READY",
                                "blockers": [],
                                "evidence": {
                                    "componentSha": component_sha,
                                    "agentReleaseGate": gate_evidence,
                                    "iq9075FleetRuntime": signed_identity,
                                },
                            }
                        },
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(
                READINESS,
                "_verify_detached_signature",
                return_value=fingerprint,
            ):
                verified = READINESS.verify_readiness(
                    readiness,
                    version="0.1.121",
                    component_sha=component_sha,
                    gate_evidence=gate_evidence,
                    security_policy=ROOT
                    / "packaging/release/release-security-policy.json",
                    signer_directory=ROOT
                    / "packaging/release/trusted-tag-signers",
                    candidate_fleet_runner=ROOT
                    / "packaging/dev/run-iq9075-fleet-e2e.py",
                    candidate_board_tool=ROOT
                    / "packaging/dev/iq9075-board-e2e.py",
                )
            self.assertEqual(
                verified["runtime_artifact_sha256"], assembled["artifactSha256"]
            )

            readiness_payload = json.loads(readiness.read_text(encoding="utf-8"))
            readiness_payload["releases"]["0.1.121"]["evidence"][
                "iq9075Physical"
            ] = signed_identity
            readiness.write_text(
                json.dumps(readiness_payload, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                READINESS.ReadinessError, "lacks exact component evidence"
            ):
                READINESS.verify_readiness(
                    readiness,
                    version="0.1.121",
                    component_sha=component_sha,
                    gate_evidence=gate_evidence,
                    security_policy=ROOT
                    / "packaging/release/release-security-policy.json",
                    signer_directory=ROOT
                    / "packaging/release/trusted-tag-signers",
                    candidate_harness=ROOT / "packaging/dev/test-iq9075.sh",
                )

    def test_raw_iq9075_evidence_is_dereferenced_and_recomputed(self) -> None:
        security = json.loads(
            (ROOT / "packaging/release/release-security-policy.json").read_text(
                encoding="utf-8"
            )
        )

        def validate(root: Path, summary: dict[str, object]) -> None:
            READINESS._validate_physical_documents(
                policy_path=root / "release-readiness.json",
                version="0.1.121",
                component_sha="a" * 40,
                summary=summary,
                security=security,
                candidate_harness=ROOT / "packaging/dev/test-iq9075.sh",
            )

        mutations = {
            "failed-outcome": lambda result: result["outcome"].update(
                status="failed", error="RSS gate failed"
            ),
            "raw-frame-shortfall": lambda result: result["soak"].update(
                rawSamples=100
            ),
            "gstreamer-error": lambda result: result["soak"].update(
                gstreamerErrors=["pipeline-error"]
            ),
            "bool-exit-code": lambda result: result.update(exitCode=False),
            "too-few-rss-samples": lambda result: result["soak"].update(
                rssAnonSamples=result["soak"]["rssAnonSamples"][:3]
            ),
            "duplicate-rss-time": lambda result: result["soak"][
                "rssAnonSamples"
            ][1].update(elapsedSec=0.0),
            "large-rss-gap": lambda result: result["soak"].update(
                rssAnonSamples=[
                    {"elapsedSec": 0.0, "rssAnonKiB": 131072},
                    *[
                        {"elapsedSec": float(value), "rssAnonKiB": 131072}
                        for value in range(20, 121, 5)
                    ],
                ]
            ),
            "terminal-stop-missing": lambda result: result["webrtc"].update(
                terminalStopCount=0
            ),
            "branch-parent-left": lambda result: result["webrtc"].update(
                branchParentDetached=False
            ),
            "queue-set-empty": lambda result: result["soak"].update(
                queueHighWatermarks={}
            ),
            "queue-set-one": lambda result: result["soak"].update(
                queueHighWatermarks={"physical_raw_queue": 1}
            ),
            "queue-set-three": lambda result: result["soak"].update(
                queueHighWatermarks={
                    key: value
                    for key, value in result["soak"]["queueHighWatermarks"].items()
                    if key != "clip_live_queue"
                }
            ),
            "queue-set-five": lambda result: result["soak"][
                "queueHighWatermarks"
            ].update(unknown_queue=0),
            "queue-set-unknown": lambda result: result["soak"].update(
                queueHighWatermarks={
                    "physical_raw_queue": 1,
                    "physical_overlay_queue": 1,
                    "uplink_live_queue": 2,
                    "unknown_queue": 2,
                }
            ),
            "tee-pad-left": lambda result: result["webrtc"].update(
                teeRequestPadCount=1
            ),
            "wrong-splitmux-contract": lambda result: result["splitmux"].update(
                segmentSeconds=60.0,
                retentionLimit=1,
                segmentsAtEnd=1,
                fragmentsOpenedDuringSoak=0,
            ),
            "rollback-slot-mismatch": lambda result: result["rollback"].update(
                restoredSlot="releases/" + "f" * 64
            ),
            "rollback-pid-reuse": lambda result: result["rollback"].update(
                restoredPid=result["rollback"]["candidatePid"]
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as raw_root:
                root = Path(raw_root)
                summary, _manifest_path, result_path = self._physical_fixture(
                    root, component_sha="a" * 40
                )
                result = json.loads(result_path.read_text(encoding="utf-8"))
                mutate(result)
                result_path.write_text(
                    json.dumps(
                        result,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                summary["harnessResult"]["sha256"] = hashlib.sha256(
                    result_path.read_bytes()
                ).hexdigest()
                with self.assertRaises(READINESS.ReadinessError):
                    validate(root, summary)

        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            summary, manifest_path, result_path = self._candidate_physical_fixture(
                root, component_sha="a" * 40
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            oak_path = root / manifest["oakSoak"]["file"]
            oak = json.loads(oak_path.read_text(encoding="utf-8"))
            oak["runtimeIdentity"]["agentVersion"] = "0.1.120"
            oak_path.write_text(
                json.dumps(oak, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            manifest["oakSoak"]["sha256"] = hashlib.sha256(
                oak_path.read_bytes()
            ).hexdigest()
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["manifestSha256"] = hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest()
            result_path.write_text(
                json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            summary["harnessManifest"]["sha256"] = hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest()
            summary["harnessResult"]["sha256"] = hashlib.sha256(
                result_path.read_bytes()
            ).hexdigest()
            with self.assertRaisesRegex(
                READINESS.ReadinessError, "OAK soak source identity"
            ):
                validate(root, summary)

        for mutation in ("omitted", "byte-drift"):
            with self.subTest(cleanup_receipt=mutation), tempfile.TemporaryDirectory() as raw_root:
                root = Path(raw_root)
                summary, _manifest_path, _result_path = (
                    self._candidate_physical_fixture(
                        root, component_sha="a" * 40
                    )
                )
                if mutation == "omitted":
                    summary.pop("cleanupEvidence")
                else:
                    cleanup_path = root / summary["cleanupEvidence"]["file"]
                    cleanup_value = json.loads(
                        cleanup_path.read_text(encoding="utf-8")
                    )
                    cleanup_path.write_text(
                        json.dumps(cleanup_value, indent=2) + "\n",
                        encoding="utf-8",
                    )
                with self.assertRaises(READINESS.ReadinessError):
                    validate(root, summary)

        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            summary, _manifest_path, result_path = self._physical_fixture(
                root, component_sha="a" * 40
            )
            result_path.write_bytes(result_path.read_bytes() + b" ")
            with self.assertRaisesRegex(
                READINESS.ReadinessError, "does not match signed summary"
            ):
                validate(root, summary)

        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            summary, _manifest_path, result_path = self._physical_fixture(
                root, component_sha="a" * 40
            )
            raw = result_path.read_text(encoding="utf-8")
            result_path.write_text(
                raw.replace('"schemaVersion":2', '"schemaVersion":2,"schemaVersion":2'),
                encoding="utf-8",
            )
            summary["harnessResult"]["sha256"] = hashlib.sha256(
                result_path.read_bytes()
            ).hexdigest()
            with self.assertRaisesRegex(READINESS.ReadinessError, "duplicate"):
                validate(root, summary)

        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            summary, _manifest_path, result_path = self._physical_fixture(
                root, component_sha="a" * 40
            )
            raw = result_path.read_text(encoding="utf-8")
            result_path.write_text(
                raw.replace('"durationSeconds":120.0', '"durationSeconds":NaN'),
                encoding="utf-8",
            )
            summary["harnessResult"]["sha256"] = hashlib.sha256(
                result_path.read_bytes()
            ).hexdigest()
            with self.assertRaisesRegex(READINESS.ReadinessError, "constant"):
                validate(root, summary)

        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            summary, manifest_path, result_path = self._candidate_physical_fixture(
                root, component_sha="a" * 40
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            fleet_manifest_path = root / manifest["fleetManifest"]["file"]
            fleet_manifest = json.loads(
                fleet_manifest_path.read_text(encoding="utf-8")
            )
            fleet_manifest["toolSha256"] = "f" * 64
            fleet_manifest_path.write_text(
                json.dumps(fleet_manifest, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            manifest["fleetManifest"]["sha256"] = hashlib.sha256(
                fleet_manifest_path.read_bytes()
            ).hexdigest()
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["manifestSha256"] = hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest()
            result_path.write_text(
                json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            summary["harnessManifest"]["sha256"] = hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest()
            summary["harnessResult"]["sha256"] = hashlib.sha256(
                result_path.read_bytes()
            ).hexdigest()
            with self.assertRaisesRegex(
                READINESS.ReadinessError, "board harness bytes"
            ):
                validate(root, summary)

        def mutate_release_sha(manifest, _evidence) -> None:
            manifest["inputs"]["releaseSha256"] = "e" * 64

        def mutate_publisher_key(manifest, evidence) -> None:
            manifest["scenario"]["release"]["publisherKeyId"] = "attacker-key"
            evidence["updater"]["update"]["publisherKeyId"] = "attacker-key"
            evidence["slots"]["previousRelease"]["publisherKeyId"] = (
                "attacker-key"
            )

        def mutate_config_schema(manifest, evidence) -> None:
            manifest["scenario"]["release"]["configSchema"] = "99"
            evidence["updater"]["update"]["configSchema"] = "99"
            evidence["slots"]["previousRelease"]["configSchema"] = "99"

        def mutate_oak_state(_manifest, evidence) -> None:
            evidence["oak"]["attached"] = False

        def mutate_service_state(_manifest, evidence) -> None:
            evidence["services"]["nuv-agent.service"]["active"] = False

        def mutate_device_identity(manifest, _evidence) -> None:
            manifest["identity"]["deviceId"] = "sp-3-nuvion-other-board"

        def mutate_manifest_schema(manifest, _evidence) -> None:
            manifest["schemaVersion"] = 999

        def mutate_manifest_protocol(manifest, _evidence) -> None:
            manifest["protocolVersion"] = "evil"

        def mutate_manifest_destinations(manifest, _evidence) -> None:
            manifest["destinations"] = {}

        def mutate_manifest_product(manifest, _evidence) -> None:
            manifest["identity"]["productModel"] = "NUVION"

        def mutate_manifest_command_id(manifest, evidence) -> None:
            manifest["scenario"]["expectedCommandId"] = "not-a-uuid"
            evidence["updater"]["update"]["commandId"] = "not-a-uuid"

        def mutate_manifest_bool_schema(manifest, _evidence) -> None:
            manifest["schemaVersion"] = True

        def mutate_manifest_docker_bool(manifest, _evidence) -> None:
            manifest["identity"]["dockerRequired"] = 0

        def mutate_manifest_hold_float(manifest, _evidence) -> None:
            manifest["scenario"]["holdSeconds"] = 10.0

        def mutate_release_type(manifest, evidence) -> None:
            manifest["scenario"]["release"]["configSchema"] = 12
            evidence["updater"]["update"]["configSchema"] = 12
            evidence["slots"]["previousRelease"]["configSchema"] = 12

        def mutate_evidence_bool_schema(_manifest, evidence) -> None:
            evidence["schemaVersion"] = True

        def mutate_evidence_extra_root(_manifest, evidence) -> None:
            evidence["contradictory"] = True

        def mutate_updater_unavailable(_manifest, evidence) -> None:
            evidence["updater"].update(
                capabilityAvailable=False,
                authenticatedHelper=False,
                reason="UPDATER_UNAVAILABLE",
            )

        def mutate_update_extra(_manifest, evidence) -> None:
            evidence["updater"]["update"]["contradictory"] = True

        def mutate_update_message_type(_manifest, evidence) -> None:
            evidence["updater"]["update"]["message"] = {}

        def mutate_update_expiry_type(_manifest, evidence) -> None:
            evidence["updater"]["update"]["commandExpiresAt"] = False

        def mutate_release_marker_extra(_manifest, evidence) -> None:
            evidence["slots"]["release"]["contradictory"] = True

        def mutate_rollback_generated_before_update(_manifest, evidence) -> None:
            evidence["generatedAt"] = "2026-09-03T10:01:59Z"

        def mutate_rollback_update_after_generated(_manifest, evidence) -> None:
            evidence["updater"]["update"]["updatedAt"] = (
                "2026-09-03T10:03:01Z"
            )

        def mutate_rollback_health_deadline(_manifest, evidence) -> None:
            evidence["updater"]["update"]["healthDeadline"] = (
                "2026-09-03T10:30:00Z"
            )

        for label, mutate in {
            "release-keyring": mutate_release_sha,
            "publisher-key": mutate_publisher_key,
            "config-schema": mutate_config_schema,
            "oak-state": mutate_oak_state,
            "service-state": mutate_service_state,
            "device-identity": mutate_device_identity,
            "manifest-schema": mutate_manifest_schema,
            "manifest-protocol": mutate_manifest_protocol,
            "manifest-destinations": mutate_manifest_destinations,
            "manifest-product": mutate_manifest_product,
            "manifest-command-id": mutate_manifest_command_id,
            "manifest-bool-schema": mutate_manifest_bool_schema,
            "manifest-docker-bool": mutate_manifest_docker_bool,
            "manifest-hold-float": mutate_manifest_hold_float,
            "release-type": mutate_release_type,
            "evidence-bool-schema": mutate_evidence_bool_schema,
            "evidence-extra-root": mutate_evidence_extra_root,
            "updater-unavailable": mutate_updater_unavailable,
            "update-extra": mutate_update_extra,
            "update-message-type": mutate_update_message_type,
            "update-expiry-type": mutate_update_expiry_type,
            "release-marker-extra": mutate_release_marker_extra,
            "rollback-generated-before-update": mutate_rollback_generated_before_update,
            "rollback-updated-after-generated": mutate_rollback_update_after_generated,
            "rollback-health-deadline": mutate_rollback_health_deadline,
        }.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as raw_root:
                root = Path(raw_root)
                summary, manifest_path, result_path = self._physical_fixture(
                    root, component_sha="a" * 40
                )
                physical_manifest = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
                fleet_manifest_path = (
                    root / physical_manifest["fleetManifest"]["file"]
                )
                fleet_evidence_path = (
                    root / physical_manifest["fleetEvidence"]["file"]
                )
                fleet_manifest = json.loads(
                    fleet_manifest_path.read_text(encoding="utf-8")
                )
                fleet_evidence = json.loads(
                    fleet_evidence_path.read_text(encoding="utf-8")
                )
                mutate(fleet_manifest, fleet_evidence)
                for path, payload in (
                    (fleet_manifest_path, fleet_manifest),
                    (fleet_evidence_path, fleet_evidence),
                ):
                    path.write_text(
                        json.dumps(payload, sort_keys=True, separators=(",", ":"))
                        + "\n",
                        encoding="utf-8",
                    )
                physical_manifest["fleetManifest"]["sha256"] = hashlib.sha256(
                    fleet_manifest_path.read_bytes()
                ).hexdigest()
                physical_manifest["fleetEvidence"]["sha256"] = hashlib.sha256(
                    fleet_evidence_path.read_bytes()
                ).hexdigest()
                manifest_path.write_text(
                    json.dumps(
                        physical_manifest, sort_keys=True, separators=(",", ":")
                    )
                    + "\n",
                    encoding="utf-8",
                )
                result = json.loads(result_path.read_text(encoding="utf-8"))
                result["manifestSha256"] = hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest()
                result_path.write_text(
                    json.dumps(result, sort_keys=True, separators=(",", ":"))
                    + "\n",
                    encoding="utf-8",
                )
                summary["harnessManifest"]["sha256"] = hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest()
                summary["harnessResult"]["sha256"] = hashlib.sha256(
                    result_path.read_bytes()
                ).hexdigest()
                with self.assertRaises(READINESS.ReadinessError):
                    validate(root, summary)

    def test_raw_iq9075_evidence_rejects_non_regular_or_oversize_files(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            regular = root / "regular.json"
            regular.write_text("{}\n", encoding="utf-8")
            link = root / "link.json"
            link.symlink_to(regular)
            fifo = root / "fifo.json"
            os.mkfifo(fifo)
            oversize = root / "oversize.json"
            with oversize.open("wb") as output:
                output.truncate(READINESS.MAX_EVIDENCE_BYTES + 1)
            for path in (link, fifo, oversize):
                with self.subTest(path=path.name), self.assertRaises(
                    READINESS.ReadinessError
                ):
                    READINESS._regular_bytes(path)

    def test_evidence_readers_reject_swap_to_symlink_races(self) -> None:
        for label, reader, error in (
            ("readiness", READINESS._regular_bytes, READINESS.ReadinessError),
            ("assembler", PHYSICAL_EVIDENCE._regular_bytes, PHYSICAL_EVIDENCE.AssemblyError),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as raw_root:
                root = Path(raw_root)
                victim = root / "evidence.json"
                backup = root / "evidence.original"
                attacker = root / "attacker.json"
                victim.write_text('{"trusted":true}\n', encoding="utf-8")
                attacker.write_text('{"trusted":false}\n', encoding="utf-8")
                real_open = STABLE_FILE.os.open

                def swap_before_open(path, flags, *args):
                    self.assertEqual(Path(path), victim)
                    victim.rename(backup)
                    victim.symlink_to(attacker)
                    try:
                        return real_open(path, flags, *args)
                    finally:
                        victim.unlink()
                        backup.rename(victim)

                with mock.patch.object(
                    STABLE_FILE.os, "open", side_effect=swap_before_open
                ), self.assertRaises(error):
                    reader(victim)
                self.assertEqual(victim.read_text(encoding="utf-8"), '{"trusted":true}\n')

    def test_iq9075_legacy_physical_evidence_is_rejected_atomically(self) -> None:
        component_sha = "a" * 40

        def inputs(source: Path) -> dict[str, object]:
            _summary, manifest_path, _result_path = self._physical_fixture(
                source, component_sha=component_sha
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            return {
                "soak_result_path": source / manifest["oakSoak"]["file"],
                "fleet_manifest_path": source / manifest["fleetManifest"]["file"],
                "fleet_evidence_path": source / manifest["fleetEvidence"]["file"],
                "artifact_path": (
                    source
                    / "nuv-agent_0.1.121_iq9075-aarch64.agent-bundle.tar.gz"
                ),
                "bom_path": source / manifest["testedBom"]["file"],
                "candidate_harness": ROOT / "packaging/dev/test-iq9075.sh",
                "candidate_fleet_runner": (
                    ROOT / "packaging/dev/run-iq9075-fleet-e2e.py"
                ),
                "candidate_board_tool": ROOT / "packaging/dev/iq9075-board-e2e.py",
                "security_policy_path": (
                    ROOT / "packaging/release/release-security-policy.json"
                ),
                "version": "0.1.121",
                "component_sha": component_sha,
            }

        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            source = root / "source"
            output = root / "output"
            source.mkdir(mode=0o700)
            output.mkdir(mode=0o700)
            arguments = inputs(source)
            summary, _manifest_path, _result_path = self._physical_fixture(
                source, component_sha=component_sha
            )
            security = json.loads(
                arguments["security_policy_path"].read_text(encoding="utf-8")
            )
            with self.assertRaisesRegex(
                READINESS.ReadinessError, "OAK soak source identity"
            ):
                READINESS._validate_physical_documents(
                    policy_path=source / "release-readiness.json",
                    version="0.1.121",
                    component_sha=component_sha,
                    summary=summary,
                    security=security,
                    candidate_harness=arguments["candidate_harness"],
                )
            with self.assertRaisesRegex(
                PHYSICAL_EVIDENCE.AssemblyError,
                "requires schema-v3 candidate soak evidence",
            ):
                PHYSICAL_EVIDENCE.assemble(
                    **arguments, output_directory=output
                )
            self.assertEqual(list(output.iterdir()), [])

        def corrupt_artifact(arguments: dict[str, object]) -> None:
            arguments["artifact_path"].write_bytes(b"different artifact")

        def corrupt_bom(arguments: dict[str, object]) -> None:
            arguments["bom_path"].write_text("{}\n", encoding="utf-8")

        def corrupt_fleet_tool(arguments: dict[str, object]) -> None:
            path = arguments["fleet_manifest_path"]
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["toolSha256"] = "f" * 64
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

        def duplicate_soak_member(arguments: dict[str, object]) -> None:
            path = arguments["soak_result_path"]
            raw = path.read_text(encoding="utf-8")
            path.write_text(
                raw.replace('"schemaVersion":2', '"schemaVersion":2,"schemaVersion":2'),
                encoding="utf-8",
            )

        def nan_soak_metric(arguments: dict[str, object]) -> None:
            path = arguments["soak_result_path"]
            raw = path.read_text(encoding="utf-8")
            path.write_text(
                raw.replace('"durationSeconds":120.0', '"durationSeconds":NaN'),
                encoding="utf-8",
            )

        def validation_failure(arguments: dict[str, object]) -> None:
            path = arguments["soak_result_path"]
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["soak"]["rawSamples"] = 1
            path.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )

        def failed_soak(arguments: dict[str, object]) -> None:
            path = arguments["soak_result_path"]
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["outcome"] = {
                "status": "failed",
                "error": "RuntimeError: RSS gate failed",
                "cleanupErrors": [],
            }
            path.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )

        def bool_schema(arguments: dict[str, object]) -> None:
            path = arguments["soak_result_path"]
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["schemaVersion"] = True
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

        for label, corrupt in {
            "artifact": corrupt_artifact,
            "bom": corrupt_bom,
            "fleet-tool": corrupt_fleet_tool,
            "duplicate": duplicate_soak_member,
            "nan": nan_soak_metric,
            "validator": validation_failure,
            "failed-soak": failed_soak,
            "bool-schema": bool_schema,
        }.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as raw_root:
                root = Path(raw_root)
                source = root / "source"
                output = root / "output"
                source.mkdir(mode=0o700)
                output.mkdir(mode=0o700)
                arguments = inputs(source)
                corrupt(arguments)
                with self.assertRaises(PHYSICAL_EVIDENCE.AssemblyError) as caught:
                    PHYSICAL_EVIDENCE.assemble(
                        **arguments, output_directory=output
                    )
                self.assertEqual(list(output.iterdir()), [])

        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            source = root / "source"
            output = root / "output"
            source.mkdir(mode=0o700)
            output.mkdir(mode=0o700)
            arguments = inputs(source)
            actual = arguments["soak_result_path"]
            link = source / "soak-link.json"
            link.symlink_to(actual)
            arguments["soak_result_path"] = link
            with self.assertRaises(PHYSICAL_EVIDENCE.AssemblyError):
                PHYSICAL_EVIDENCE.assemble(**arguments, output_directory=output)
            self.assertEqual(list(output.iterdir()), [])

        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            source = root / "source"
            real_output = root / "real-output"
            output_link = root / "output"
            source.mkdir(mode=0o700)
            real_output.mkdir(mode=0o700)
            output_link.symlink_to(real_output, target_is_directory=True)
            arguments = inputs(source)
            with self.assertRaisesRegex(
                PHYSICAL_EVIDENCE.AssemblyError, "symbolic link"
            ):
                PHYSICAL_EVIDENCE.assemble(
                    **arguments, output_directory=output_link
                )
            self.assertEqual(list(real_output.iterdir()), [])

    def test_candidate_soak_chain_orders_rollback_before_soak_and_restore(self) -> None:
        component_sha = "a" * 40
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            source = root / "source"
            output = root / "output"
            source.mkdir(mode=0o700)
            output.mkdir(mode=0o700)
            self._physical_fixture(source, component_sha=component_sha)
            manifest_path = source / "iq9075-v0.1.121-harness-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            fleet_manifest_path = source / manifest["fleetManifest"]["file"]
            fleet_evidence_path = source / manifest["fleetEvidence"]["file"]
            fleet_manifest = json.loads(
                fleet_manifest_path.read_text(encoding="utf-8")
            )
            fleet_evidence = json.loads(
                fleet_evidence_path.read_text(encoding="utf-8")
            )
            soak_path = source / manifest["oakSoak"]["file"]
            soak = json.loads(soak_path.read_text(encoding="utf-8"))
            run_id = fleet_manifest["runId"]
            bom_digest = fleet_manifest["scenario"]["expectedBomDigest"]
            candidate_slot = (
                f"/opt/nuv-agent/candidates/{run_id}-{bom_digest[7:]}"
            )
            control_sha = "c" * 64
            soak.update(
                {
                    "schemaVersion": 3,
                    "runId": run_id,
                    "slotKind": "candidate",
                    "startedAt": "2026-09-03T10:04:00Z",
                }
            )
            soak["runtimeIdentity"].update(
                {
                    "pythonPath": "/usr/bin/python3",
                    "sitePackagesPath": candidate_slot
                    + "/venv/lib/python3.12/site-packages",
                    "buildInfoPath": candidate_slot
                    + "/venv/lib/python3.12/site-packages/nuvion_app/build_info.py",
                    "candidateSlot": candidate_slot,
                    "controlMarkerSha256": control_sha,
                }
            )
            soak_path.write_text(
                json.dumps(soak, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            raw_sha = hashlib.sha256(soak_path.read_bytes()).hexdigest()
            artifact_path = (
                source / "nuv-agent_0.1.121_iq9075-aarch64.agent-bundle.tar.gz"
            )
            bom_path = source / "nuv-agent_0.1.121_iq9075-aarch64.release-bom.json"
            bundle_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            bom_sha = hashlib.sha256(bom_path.read_bytes()).hexdigest()
            harness_sha = hashlib.sha256(
                (ROOT / "packaging/dev/test-iq9075.sh").read_bytes()
            ).hexdigest()
            slots = {
                "current": fleet_evidence["slots"]["current"],
                "previous": fleet_evidence["slots"]["previous"],
            }
            anti_replay = {
                "schemaVersion": 4,
                "semanticSha256": "0" * 64,
                "maximumCommandSequence": 2,
                "currentReleaseSequence": "1",
                "currentBomDigest": fleet_evidence["slots"]["release"][
                    "bomDigest"
                ],
                "latest": {
                    "commandId": fleet_manifest["scenario"]["expectedCommandId"],
                    "sequence": 2,
                    "phase": "ROLLED_BACK",
                    "bomDigest": bom_digest,
                    "releaseSequence": 2,
                    "healthDeadline": None,
                },
            }
            before_runtime = {
                "pid": 4200,
                "startTicks": 42000,
                "bootId": "11111111-1111-4111-8111-111111111111",
                "activeSlot": slots["current"],
            }
            after_runtime = {**before_runtime, "pid": 4400, "startTicks": 44000}
            cleanup = cleanup_evidence(run_id)
            candidate_evidence = {
                "schemaVersion": 1,
                "kind": "nuvion-iq9075-candidate-soak-evidence",
                "protocolVersion": FLEET_E2E.PROTOCOL_VERSION,
                "runId": run_id,
                "startedAt": "2026-09-03T10:03:30Z",
                "completedAt": "2026-09-03T10:06:30Z",
                "complete": True,
                "outcome": {"status": "passed", "errorCode": None},
                "candidate": {
                    "slotKind": "candidate",
                    "slot": candidate_slot,
                    "bomDigest": bom_digest,
                    "bundleSha256": bundle_sha,
                    "bomSha256": bom_sha,
                    "harnessSha256": harness_sha,
                    "controlMarkerSha256": control_sha,
                },
                "fleetEvidenceSha256": hashlib.sha256(
                    fleet_evidence_path.read_bytes()
                ).hexdigest(),
                "rawEvidenceSha256": raw_sha,
                "rawEvidence": soak,
                "cleanupEvidenceSha256": canonical_sha256(cleanup),
                "cleanupEvidence": cleanup,
                "executionProof": candidate_execution_proof(run_id),
                "collectorProof": candidate_collector_proof(run_id),
                "terminationProof": candidate_termination_proof(run_id),
                "productionRestoration": production_restoration_evidence(
                    fleet_manifest
                ),
                "pre": {
                    "slots": slots,
                        "antiReplay": anti_replay,
                        "oak": fleet_evidence["oak"],
                        "runtime": before_runtime,
                        "persistentState": persistent_state_evidence(),
                        "releaseTrees": release_tree_evidence(slots),
                    },
                    "post": {
                    "restoredAt": "2026-09-03T10:06:00Z",
                    "slots": slots,
                    "antiReplay": anti_replay,
                        "oak": fleet_evidence["oak"],
                        "runtime": after_runtime,
                        "persistentState": persistent_state_evidence(),
                        "releaseTrees": release_tree_evidence(slots),
                    },
                "gates": {
                    "signedRollbackTerminal": True,
                    "candidateBound": True,
                    "rawEvidencePreserved": True,
                    "slotsUnchanged": True,
                    "releaseTreesUnchanged": True,
                    "antiReplayUnchanged": True,
                    "oakIdentityUnchanged": True,
                    "freshBaselineProcess": True,
                    "harnessBytesPinned": True,
                    "harnessCopyRemoved": True,
                    "resourceLimitsApplied": True,
                    "boundedOutput": True,
                    "persistentStateReadOnly": True,
                    "persistentStateUnchanged": True,
                    "productionTrustRestored": True,
                    "trustedSoakDuration": True,
                    "continuousUidIsolation": True,
                    "cgroupTerminated": True,
                    "harnessPassed": True,
                },
            }
            candidate_path = source / "candidate-soak-evidence.json"
            candidate_path.write_text(
                json.dumps(
                    candidate_evidence, sort_keys=True, separators=(",", ":")
                )
                + "\n",
                encoding="utf-8",
            )
            result = PHYSICAL_EVIDENCE.assemble(
                soak_result_path=soak_path,
                fleet_manifest_path=fleet_manifest_path,
                fleet_evidence_path=fleet_evidence_path,
                artifact_path=artifact_path,
                bom_path=bom_path,
                candidate_harness=ROOT / "packaging/dev/test-iq9075.sh",
                candidate_fleet_runner=ROOT
                / "packaging/dev/run-iq9075-fleet-e2e.py",
                candidate_board_tool=ROOT / "packaging/dev/iq9075-board-e2e.py",
                security_policy_path=ROOT
                / "packaging/release/release-security-policy.json",
                output_directory=output,
                version="0.1.121",
                component_sha=component_sha,
                candidate_soak_evidence_path=candidate_path,
            )
            self.assertEqual(len(list(output.iterdir())), 9)
            assembled_result = json.loads(
                Path(result["result"]).read_text(encoding="utf-8")
            )
            self.assertEqual(assembled_result["schemaVersion"], 3)
            self.assertEqual(
                assembled_result["candidateRestore"], candidate_evidence["post"]
            )
            cleanup_mutations: list[tuple[str, dict[str, object]]] = []
            missing_cleanup = copy.deepcopy(candidate_evidence)
            missing_cleanup.pop("cleanupEvidence")
            missing_cleanup.pop("cleanupEvidenceSha256")
            cleanup_mutations.append(("omitted", missing_cleanup))
            wrong_digest = copy.deepcopy(candidate_evidence)
            wrong_digest["cleanupEvidenceSha256"] = "0" * 64
            cleanup_mutations.append(("digest", wrong_digest))
            for label, key in (
                ("stale-run", "runId"),
                ("lease-present", "activeRunLeaseAbsent"),
                ("secret-snapshot-present", "transactionSnapshotsAbsent"),
            ):
                mutation = copy.deepcopy(candidate_evidence)
                if key == "runId":
                    mutation["cleanupEvidence"]["runId"] = str(uuid.uuid4())
                else:
                    mutation["cleanupEvidence"]["proof"][key] = False
                mutation["cleanupEvidenceSha256"] = canonical_sha256(
                    mutation["cleanupEvidence"]
                )
                cleanup_mutations.append((label, mutation))
            for label, mutation in cleanup_mutations:
                with self.subTest(cleanup_evidence=label):
                    candidate_path.write_text(
                        json.dumps(
                            mutation, sort_keys=True, separators=(",", ":")
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    cleanup_rejected = root / f"cleanup-rejected-{label}"
                    cleanup_rejected.mkdir(mode=0o700)
                    with self.assertRaises(PHYSICAL_EVIDENCE.AssemblyError):
                        PHYSICAL_EVIDENCE.assemble(
                            soak_result_path=soak_path,
                            fleet_manifest_path=fleet_manifest_path,
                            fleet_evidence_path=fleet_evidence_path,
                            artifact_path=artifact_path,
                            bom_path=bom_path,
                            candidate_harness=ROOT / "packaging/dev/test-iq9075.sh",
                            candidate_fleet_runner=ROOT
                            / "packaging/dev/run-iq9075-fleet-e2e.py",
                            candidate_board_tool=ROOT
                            / "packaging/dev/iq9075-board-e2e.py",
                            security_policy_path=ROOT
                            / "packaging/release/release-security-policy.json",
                            output_directory=cleanup_rejected,
                            version="0.1.121",
                            component_sha=component_sha,
                            candidate_soak_evidence_path=candidate_path,
                        )
                    self.assertEqual(list(cleanup_rejected.iterdir()), [])
            candidate_path.write_text(
                json.dumps(
                    candidate_evidence, sort_keys=True, separators=(",", ":")
                )
                + "\n",
                encoding="utf-8",
            )
            # Object equality is insufficient: the signed wrapper binds the
            # exact raw evidence bytes, including canonical serialization.
            soak_path.write_text(json.dumps(soak, indent=2) + "\n", encoding="utf-8")
            byte_drift_output = root / "byte-drift-output"
            byte_drift_output.mkdir(mode=0o700)
            with self.assertRaises(PHYSICAL_EVIDENCE.AssemblyError):
                PHYSICAL_EVIDENCE.assemble(
                    soak_result_path=soak_path,
                    fleet_manifest_path=fleet_manifest_path,
                    fleet_evidence_path=fleet_evidence_path,
                    artifact_path=artifact_path,
                    bom_path=bom_path,
                    candidate_harness=ROOT / "packaging/dev/test-iq9075.sh",
                    candidate_fleet_runner=ROOT
                    / "packaging/dev/run-iq9075-fleet-e2e.py",
                    candidate_board_tool=ROOT / "packaging/dev/iq9075-board-e2e.py",
                    security_policy_path=ROOT
                    / "packaging/release/release-security-policy.json",
                    output_directory=byte_drift_output,
                    version="0.1.121",
                    component_sha=component_sha,
                    candidate_soak_evidence_path=candidate_path,
                )
            self.assertEqual(list(byte_drift_output.iterdir()), [])
            soak_path.write_text(
                json.dumps(soak, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            invalid = copy.deepcopy(candidate_evidence)
            invalid["post"]["restoredAt"] = "2026-09-03T10:03:59Z"
            candidate_path.write_text(
                json.dumps(invalid, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            rejected_output = root / "rejected-output"
            rejected_output.mkdir(mode=0o700)
            with self.assertRaises(PHYSICAL_EVIDENCE.AssemblyError):
                PHYSICAL_EVIDENCE.assemble(
                    soak_result_path=soak_path,
                    fleet_manifest_path=fleet_manifest_path,
                    fleet_evidence_path=fleet_evidence_path,
                    artifact_path=artifact_path,
                    bom_path=bom_path,
                    candidate_harness=ROOT / "packaging/dev/test-iq9075.sh",
                    candidate_fleet_runner=ROOT
                    / "packaging/dev/run-iq9075-fleet-e2e.py",
                    candidate_board_tool=ROOT
                    / "packaging/dev/iq9075-board-e2e.py",
                    security_policy_path=ROOT
                    / "packaging/release/release-security-policy.json",
                    output_directory=rejected_output,
                    version="0.1.121",
                    component_sha=component_sha,
                    candidate_soak_evidence_path=candidate_path,
                )
            self.assertEqual(list(rejected_output.iterdir()), [])

            candidate_path.write_text(
                json.dumps(
                    candidate_evidence, sort_keys=True, separators=(",", ":")
                )
                + "\n",
                encoding="utf-8",
            )
            execution_marker = root / "untrusted-runner-executed"
            untrusted_runner = root / "run-iq9075-fleet-e2e.py"
            untrusted_runner.write_text(
                f"from pathlib import Path\nPath({str(execution_marker)!r}).touch()\n",
                encoding="utf-8",
            )
            untrusted_output = root / "untrusted-output"
            untrusted_output.mkdir(mode=0o700)
            with self.assertRaisesRegex(
                PHYSICAL_EVIDENCE.AssemblyError,
                "differ.*trusted publisher runner",
            ):
                PHYSICAL_EVIDENCE.assemble(
                    soak_result_path=soak_path,
                    fleet_manifest_path=fleet_manifest_path,
                    fleet_evidence_path=fleet_evidence_path,
                    artifact_path=artifact_path,
                    bom_path=bom_path,
                    candidate_harness=ROOT / "packaging/dev/test-iq9075.sh",
                    candidate_fleet_runner=untrusted_runner,
                    candidate_board_tool=ROOT
                    / "packaging/dev/iq9075-board-e2e.py",
                    security_policy_path=ROOT
                    / "packaging/release/release-security-policy.json",
                    output_directory=untrusted_output,
                    version="0.1.121",
                    component_sha=component_sha,
                    candidate_soak_evidence_path=candidate_path,
                )
            self.assertFalse(execution_marker.exists())
            self.assertEqual(list(untrusted_output.iterdir()), [])

    def test_physical_evidence_assembler_isolated_mode_blocks_source_shadow(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            fake = root / "nuvion_app/runtime"
            fake.mkdir(parents=True)
            (fake.parent / "__init__.py").write_text(
                'raise RuntimeError("source shadow imported")\n', encoding="utf-8"
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(root)
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(
                        ROOT
                        / "packaging/release/assemble-iq9075-physical-evidence.py"
                    ),
                    "--help",
                ],
                cwd=root,
                env=environment,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            for script in (
                "assemble-iq9075-physical-evidence.py",
                "assemble-iq9075-fleet-runtime-evidence.py",
                "verify-release-readiness.py",
            ):
                with self.subTest(script=script):
                    nonisolated = subprocess.run(
                        [
                            sys.executable,
                            str(ROOT / "packaging/release" / script),
                            "--help",
                        ],
                        cwd=root,
                        env=environment,
                        stdin=subprocess.DEVNULL,
                        capture_output=True,
                        text=True,
                        timeout=30,
                        check=False,
                    )
                    self.assertEqual(nonisolated.returncode, 2)
                    self.assertIn("requires Python isolated mode", nonisolated.stderr)

    def test_iq9075_runbook_keeps_optional_candidate_soak_isolated(
        self,
    ) -> None:
        runbook = (
            ROOT / "packaging/release/v0.1.121-release-runbook.md"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "When a release explicitly claims OAK media stability, use the reviewed",
            runbook,
        )
        self.assertIn("Camera frame continuity is a separate media", runbook)
        self.assertIn(
            "qualification and is not part of this Fleet Runtime release decision",
            runbook,
        )
        rollback_call = runbook.index("--scenario oak-fault-rollback")
        soak_call = runbook.index(
            '--run-id "$run_id" --output-dir "$fleet_run_dir" candidate-soak'
        )
        cleanup_call = runbook.index(
            '--run-id "$run_id" --output-dir "$fleet_run_dir" cleanup',
            soak_call,
        )
        self.assertLess(rollback_call, soak_call)
        self.assertLess(soak_call, cleanup_call)
        self.assertIn('--candidate-bundle "$candidate_bundle"', runbook)
        self.assertIn('--candidate-bom "$candidate_bom"', runbook)
        self.assertIn(
            '--candidate-soak-evidence "$candidate_soak_evidence"', runbook
        )
        self.assertNotIn("sudo -n env -u PYTHONPATH NUVION_AGENT_PYTHON", runbook)
        self.assertNotIn('scp -F /dev/null -P "$board_port"', runbook)
        self.assertIn(
            '"$candidate_bom" "$release_sha" <<\'PY\'', runbook
        )
        self.assertNotIn("--device-id sp-3-nuvion-iq9075 --space-id 3", runbook)
        self.assertIn('--device-id "$device_id" --space-id "$space_id"', runbook)
        self.assertGreaterEqual(
            runbook.count("env -u PYTHONPATH PYTHONNOUSERSITE=1"), 4
        )
        self.assertIn('test ! -e "$evidence_target" -a ! -L "$evidence_target"', runbook)

    def test_latest_failed_gate_supersedes_older_success(self) -> None:
        component_sha = "a" * 40
        base = {
            "name": "agent-release-gate",
            "head_sha": component_sha,
            "status": "completed",
            "details_url": (
                "https://github.com/plaid-ai/NUV-AGENT/actions/runs/8003/job/9004"
            ),
            "app": {"id": 15368, "slug": "github-actions"},
            "check_suite": {"id": 6001},
        }
        with self.assertRaisesRegex(
            RELEASE_GATE.ReleaseGateError,
            "latest trusted release gate check did not succeed",
        ):
            RELEASE_GATE.verify_release_gate(
                repository="plaid-ai/NUV-AGENT",
                component_sha=component_sha,
                required_context="agent-release-gate",
                required_integration_id=15368,
                workflow_sha256="b" * 64,
                check_runs=[
                    {**base, "id": 7001, "conclusion": "success"},
                    {**base, "id": 7002, "conclusion": "failure"},
                ],
                workflow_run=lambda _run_id: {},
            )

    def test_gate_rejects_wrong_actions_app_or_workflow_path(self) -> None:
        component_sha = "a" * 40
        repository = "plaid-ai/NUV-AGENT"
        check = {
            "id": 7002,
            "name": "agent-release-gate",
            "head_sha": component_sha,
            "status": "completed",
            "conclusion": "success",
            "details_url": (
                "https://github.com/plaid-ai/NUV-AGENT/actions/runs/8003/job/9004"
            ),
            "app": {"id": 15368, "slug": "github-actions"},
            "check_suite": {"id": 6001},
        }
        run = {
            "id": 8003,
            "check_suite_id": 6001,
            "head_sha": component_sha,
            "name": "agent-release-gate",
            "path": ".github/workflows/not-the-release-gate.yml",
            "status": "completed",
            "conclusion": "success",
            "event": "pull_request",
            "repository": {"full_name": repository},
        }
        with self.assertRaisesRegex(
            RELEASE_GATE.ReleaseGateError,
            "exact release workflow run",
        ):
            RELEASE_GATE.verify_release_gate(
                repository=repository,
                component_sha=component_sha,
                required_context="agent-release-gate",
                required_integration_id=15368,
                workflow_sha256="b" * 64,
                check_runs=[check],
                workflow_run=lambda _run_id: run,
            )

        check["app"] = {"id": 999, "slug": "third-party"}
        with self.assertRaisesRegex(
            RELEASE_GATE.ReleaseGateError,
            "no trusted release gate check",
        ):
            RELEASE_GATE.verify_release_gate(
                repository=repository,
                component_sha=component_sha,
                required_context="agent-release-gate",
                required_integration_id=15368,
                workflow_sha256="b" * 64,
                check_runs=[check],
                workflow_run=lambda _run_id: {},
            )

    def test_release_gate_workflow_bytes_must_match_trusted_publisher(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            candidate = root / "candidate.yml"
            trusted = root / "trusted.yml"
            candidate.write_text("name: agent-release-gate\n", encoding="utf-8")
            trusted.write_bytes(candidate.read_bytes())
            digest = RELEASE_GATE.verify_workflow_identity(candidate, trusted)
            self.assertEqual(digest, hashlib.sha256(candidate.read_bytes()).hexdigest())

            candidate.write_text("name: weakened-gate\n", encoding="utf-8")
            with self.assertRaisesRegex(
                RELEASE_GATE.ReleaseGateError,
                "differs from trusted publisher bytes",
            ):
                RELEASE_GATE.verify_workflow_identity(candidate, trusted)

    def test_ota_sequence_failure_precedes_private_key_and_uses_global_cas(self) -> None:
        ota = self.publish.split("  iq9075-ota-publish:", maxsplit=1)[1]
        self.assertIn("group: iq9075-ota-global-publisher", ota)
        self.assertLess(
            ota.index("Independently verify latest sequence and version absence"),
            ota.index("IQ9075_RELEASE_SIGNING_PRIVATE_KEY"),
        )
        self.assertLess(
            ota.index("Atomically reserve exact release sequence"),
            ota.index("Sign exact bundle BOM with trusted signer"),
        )
        immutable = (
            ROOT / "packaging/release/publish-immutable-gcs-file.sh"
        ).read_text(encoding="utf-8")
        apt = (ROOT / "packaging/apt/publish-gcs.sh").read_text(encoding="utf-8")
        for source in (immutable, apt):
            self.assertIn("--if-generation-match=0", source)
            self.assertNotIn(" cp -n ", source)
            self.assertIn("gcloud storage cat", source)

    def test_ota_verifier_uses_only_policy_pinned_public_keyring(self) -> None:
        policy_path = ROOT / "packaging/release/release-security-policy.json"
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        keyring_path = policy_path.parent / policy["iq9075"]["publicKeyringFile"]
        self.assertEqual(
            hashlib.sha256(keyring_path.read_bytes()).hexdigest(),
            policy["iq9075"]["publicKeyringSha256"],
        )
        self.assertEqual(
            policy["iq9075"]["publicKeyringSha256"],
            "2d72a28745e14014d5988ecf7970dc6f09c2f077be35105b3ad233cda0d0969a",
        )
        self.assertEqual(
            policy["iq9075"]["publisherKeyId"],
            "release-iq9075-dev-2026-09-01",
        )
        public_map = json.loads(keyring_path.read_text(encoding="utf-8"))["keys"]
        self.assertEqual(
            hashlib.sha256(
                json.dumps(public_map, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest(),
            "fe087dd340fbec31604a8c7910bc95a5c1615c5157c526cae5b4e18090a774c7",
        )
        self.assertNotIn("IQ9075_RELEASE_PUBLIC_KEYRING_JSON", self.publish)
        self.assertNotIn("secrets.IQ9075_RELEASE_SIGNING_KEY_ID", self.publish)
        self.assertNotIn(
            "IQ9075_RELEASE_PUBLIC_KEYRING_JSON",
            policy["requiredEnvironments"]["iq9075-release"]["requiredSecrets"],
        )
        self.assertIn(
            "IQ9075_RELEASE_PUBLIC_KEYRING_JSON",
            policy["forbiddenRepositorySecrets"],
        )

    def test_all_external_actions_are_full_sha_pinned(self) -> None:
        for path in sorted((ROOT / ".github/workflows").glob("*.yml")):
            for line in path.read_text(encoding="utf-8").splitlines():
                match = re.search(r"\buses:\s*([^\s#]+)", line)
                if match is None or match.group(1).startswith("./"):
                    continue
                self.assertRegex(match.group(1), r"@[0-9a-f]{40}$")

    def test_required_main_context_exists_and_is_secret_zero(self) -> None:
        gate = (ROOT / ".github/workflows/agent-release-gate.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("name: agent-release-gate", gate)
        self.assertIn("  agent-release-gate:\n    name: agent-release-gate", gate)
        self.assertIn("runs-on: ubuntu-24.04-arm", gate)
        self.assertIn(
            "needs: [arm64-release-prerequisite]",
            gate,
        )
        self.assertIn("if: always()", gate)
        self.assertIn("needs.arm64-release-prerequisite.result", gate)
        self.assertNotIn("macos-cpu-reference", gate)
        self.assertNotIn("macos-arm64-release-prerequisite", gate)
        self.assertNotIn("self-hosted", gate)
        self.assertIn("requirements-agent-bundle-arm64.txt", gate)
        self.assertIn("packaging/release/run-isolated-tests.py", gate)
        self.assertIn("actionlint", gate)
        self.assertIn("shellcheck", gate)
        self.assertNotIn("${{ secrets.", gate)
        self.assertNotIn("contents: write", gate)


class ReleaseSourceVerificationTest(unittest.TestCase):
    def _git(self, repository: Path, *arguments: str, environment=None) -> str:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        return result.stdout.strip()

    def _repository(self, root: Path) -> tuple[Path, str]:
        repository = root / "repository"
        repository.mkdir()
        self._git(repository, "init", "-b", "main")
        self._git(repository, "config", "user.name", "Release Test")
        self._git(repository, "config", "user.email", "release@example.invalid")
        (repository / "README").write_text("release\n", encoding="utf-8")
        self._git(repository, "add", "README")
        self._git(repository, "commit", "-m", "release")
        return repository, self._git(repository, "rev-parse", "HEAD")

    def _policy(self, root: Path, *, fingerprint: str, legacy: dict[str, str]) -> Path:
        payload = json.loads(
            (ROOT / "packaging/release/release-security-policy.json").read_text(
                encoding="utf-8"
            )
        )
        payload["trustedTagSignerFingerprints"] = [fingerprint]
        payload["legacyUnsignedReruns"] = legacy
        path = root / "policy.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_unsigned_legacy_tag_and_nonempty_fallback_policy_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            repository, commit = self._repository(root)
            self._git(repository, "tag", "-a", "v1.2.3", "-m", "legacy")
            signers = root / "signers"
            signers.mkdir()
            for legacy, event_name in (
                ({"v1.2.3": commit}, "workflow_dispatch"),
                ({}, "workflow_dispatch"),
                ({}, "workflow_run"),
            ):
                policy = self._policy(
                    root,
                    fingerprint="A" * 40,
                    legacy=legacy,
                )
                with self.subTest(legacy=bool(legacy), event_name=event_name):
                    with self.assertRaises(VERIFY_SOURCE.VerificationError):
                        VERIFY_SOURCE.verify_release_source(
                            repository=repository,
                            tag="v1.2.3",
                            origin_main_ref="refs/heads/main",
                            trusted_publisher_sha=commit,
                            event_name=event_name,
                            policy_path=policy,
                            signer_directory=signers,
                        )

    @unittest.skipUnless(shutil.which("gpg"), "gpg is required")
    def test_signed_tag_requires_exact_allowlisted_primary_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            repository, commit = self._repository(root)
            gpg_home = root / "gpg"
            gpg_home.mkdir(mode=0o700)
            environment = {**os.environ, "GNUPGHOME": str(gpg_home)}
            subprocess.run(
                [
                    "gpg",
                    "--batch",
                    "--passphrase",
                    "",
                    "--quick-generate-key",
                    "Release Test <release@example.invalid>",
                    "ed25519",
                    "sign",
                    "1d",
                ],
                check=True,
                capture_output=True,
                env=environment,
            )
            listing = subprocess.check_output(
                ["gpg", "--batch", "--with-colons", "--list-keys"],
                text=True,
                env=environment,
            )
            fingerprint = next(
                line.split(":")[9]
                for line in listing.splitlines()
                if line.startswith("fpr:")
            )
            self._git(repository, "config", "user.signingkey", fingerprint)
            self._git(repository, "config", "gpg.program", "gpg")
            self._git(
                repository,
                "tag",
                "-s",
                "v1.2.3",
                "-m",
                "signed",
                environment=environment,
            )
            signers = root / "signers"
            signers.mkdir()
            public_key = subprocess.check_output(
                ["gpg", "--batch", "--armor", "--export", fingerprint],
                env=environment,
            )
            (signers / "test.asc").write_bytes(public_key)
            policy = self._policy(root, fingerprint=fingerprint, legacy={})
            verified = VERIFY_SOURCE.verify_release_source(
                repository=repository,
                tag="v1.2.3",
                origin_main_ref="refs/heads/main",
                trusted_publisher_sha=commit,
                event_name="workflow_run",
                policy_path=policy,
                signer_directory=signers,
            )
            self.assertEqual(verified["tag_signer_fingerprint"], fingerprint)
            self.assertEqual(
                verified["tag_object_sha"],
                self._git(repository, "rev-parse", "refs/tags/v1.2.3^{tag}"),
            )


class SequenceAndPromotionTest(unittest.TestCase):
    def test_new_sequence_must_be_latest_plus_one(self) -> None:
        from nuvion_app.runtime.release_bom import ReleaseTarget, VerifiedReleaseBom

        target = ReleaseTarget(
            product_model="IQ9075_DEV",
            platform_profile="iq9075_dev",
            hardware_revision="QCS9075-EVK",
            architecture="aarch64",
        )
        published = VerifiedReleaseBom(
            schema_version=2,
            bom_id="nuv-agent-0.1.120-iq9075-aarch64",
            bom_digest="26a7f1674bdd4a24bfe26fa37c681798244990408fe7d858ca76957a88bdb9f1",
            agent_version="0.1.120",
            component_sha="b354026f73d63a82ad4c64923f46dc400a73efcb",
            config_schema="12",
            updater_version=None,
            release_sequence=1,
            min_updater_version="0.1.0",
            targets=(target,),
            publisher_key_id="release-test",
            platform_profiles=(),
            artifact_name="nuv-agent_0.1.120_iq9075-aarch64.agent-bundle.tar.gz",
            artifact_kind="agent-bundle",
            artifact_sha256="1" * 64,
            artifact_size_bytes=10,
            built_at="2026-09-01T12:00:00+00:00",
        )
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            artifact = root / "nuv-agent_0.1.121_iq9075-aarch64.agent-bundle.tar.gz"
            artifact.write_bytes(b"new release")
            keyring = (
                ROOT
                / "packaging/release/trusted-release-keyrings/iq9075-dev.json"
            )
            with mock.patch.object(
                PLAN_OTA, "list_version_boms", return_value={"0.1.120": "1"}
            ), mock.patch.object(
                PLAN_OTA, "_load_remote_signed_bom", return_value=published
            ):
                reservation, output = PLAN_OTA.plan_sequence(
                    policy_path=ROOT / "packaging/release/release-security-policy.json",
                    keyring_path=keyring,
                    artifact_path=artifact,
                    version="0.1.121",
                    component_sha="a" * 40,
                    requested_sequence=2,
                    config_schema="12",
                    min_updater_version="0.2.0",
                    built_at="2026-09-02T00:00:00+00:00",
                )
                self.assertEqual(reservation["releaseSequence"], 2)
                self.assertEqual(output["latest_sequence"], "1")
                self.assertEqual(output["reservation_object"], "releases/reservations/iq9075/2.json")
                with self.assertRaises(PLAN_OTA.SequencePlanError):
                    PLAN_OTA.plan_sequence(
                        policy_path=ROOT / "packaging/release/release-security-policy.json",
                        keyring_path=keyring,
                        artifact_path=artifact,
                        version="0.1.121",
                        component_sha="a" * 40,
                        requested_sequence=3,
                        config_schema="12",
                        min_updater_version="0.2.0",
                        built_at="2026-09-02T00:00:00+00:00",
                    )

    def test_distribution_promotion_is_deterministic_and_binds_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            paths = {}
            for name in ("sdist", "bom", "formula", "bundle", "deb"):
                path = root / name
                path.write_bytes(name.encode())
                paths[name] = path
            arguments = argparse.Namespace(
                version="0.1.121",
                tag="v0.1.121",
                component_sha="a" * 40,
                trusted_publisher_sha="b" * 40,
                gate_run_id=101,
                gate_check_id=102,
                gate_check_suite_id=103,
                gate_workflow_sha256="d" * 64,
                security_policy=ROOT / "packaging/release/release-security-policy.json",
                sdist=paths["sdist"],
                sdist_bom=paths["bom"],
                formula=paths["formula"],
                bundle=paths["bundle"],
                deb=paths["deb"],
                source_plan=None,
                rollback_version="0.1.120",
                rollback_sha256="c" * 64,
            )
            source_plan = root / "source-plan.json"
            source_plan.write_text(
                json.dumps(
                    PROMOTION.build_distribution_plan(arguments),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            arguments.source_plan = source_plan
            first = PROMOTION.build_distribution(arguments)
            second = PROMOTION.build_distribution(arguments)
            self.assertEqual(first, second)
            self.assertEqual(first["status"], "PROMOTED")
            self.assertEqual(first["governance"]["pullRequestApprovals"], 1)
            self.assertEqual(first["governance"]["environmentReviewers"], 0)
            self.assertEqual(first["releaseGate"]["workflowRunId"], 101)
            self.assertEqual(first["releaseGate"]["checkRunId"], 102)
            self.assertEqual(first["releaseGate"]["workflowSha256"], "d" * 64)
            self.assertEqual(
                first["artifacts"]["homebrewFormula"]["name"], "formula"
            )
            self.assertRegex(first["sourcePlanDigest"], r"^sha256:[0-9a-f]{64}$")
            self.assertEqual(
                first["rollbackPackage"],
                {"agentVersion": "0.1.120", "sha256": "c" * 64},
            )
            altered = json.loads(source_plan.read_text(encoding="utf-8"))
            altered["channels"]["homebrew"] = "PUBLISHED"
            source_plan.write_text(
                json.dumps(altered, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(PROMOTION.PromotionError):
                PROMOTION.build_distribution(arguments)

    def test_ota_promotion_binds_distribution_bundle_to_signed_bom(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            artifacts = {}
            names = {
                "sdist": "nuv_agent-0.1.121.tar.gz",
                "sdist_bom": "nuv_agent-0.1.121-sdist.release-bom.json",
                "formula": "nuv-agent-0.1.121.rb",
                "bundle": "nuv-agent_0.1.121_iq9075-aarch64.agent-bundle.tar.gz",
                "deb": "nuv-agent_0.1.121_arm64.deb",
            }
            for label, name in names.items():
                path = root / name
                path.write_bytes(label.encode("ascii"))
                artifacts[label] = path
            distribution_arguments = argparse.Namespace(
                version="0.1.121",
                tag="v0.1.121",
                component_sha="a" * 40,
                trusted_publisher_sha="b" * 40,
                gate_run_id=101,
                gate_check_id=102,
                gate_check_suite_id=103,
                gate_workflow_sha256="d" * 64,
                security_policy=ROOT / "packaging/release/release-security-policy.json",
                sdist=artifacts["sdist"],
                sdist_bom=artifacts["sdist_bom"],
                formula=artifacts["formula"],
                bundle=artifacts["bundle"],
                deb=artifacts["deb"],
                source_plan=None,
                rollback_version="0.1.120",
                rollback_sha256="c" * 64,
            )
            source_plan = root / "source-plan.json"
            source_plan.write_text(
                json.dumps(
                    PROMOTION.build_distribution_plan(distribution_arguments),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            distribution_arguments.source_plan = source_plan
            manifest = PROMOTION.build_distribution(distribution_arguments)
            manifest_path = root / "nuv_agent-0.1.121-distribution-promotion.json"
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            bundle_identity = manifest["artifacts"]["iq9075Bundle"]
            bom = mock.Mock(
                release_sequence=2,
                min_updater_version="0.2.0",
                agent_version="0.1.121",
                component_sha="a" * 40,
                bom_digest="d" * 64,
                artifact_name=bundle_identity["name"],
                artifact_sha256=bundle_identity["sha256"],
                artifact_size_bytes=bundle_identity["sizeBytes"],
                publisher_key_id="release-test",
            )
            ota_arguments = argparse.Namespace(
                distribution_promotion=manifest_path,
                bom=root / "release-bom.json",
                signature=root / "release-bom.json.sig",
                keyring=root / "keyring.json",
                trust_domain="iq9075-dev",
            )
            with mock.patch.object(PROMOTION, "load_release_keyring"), mock.patch.object(
                PROMOTION, "load_signed_release_bom", return_value=bom
            ):
                result = PROMOTION.build_ota(ota_arguments)
                self.assertEqual(result["releaseSequence"], 2)
                manifest["artifacts"]["iq9075Bundle"]["sha256"] = "e" * 64
                manifest_path.write_text(
                    json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaises(PROMOTION.PromotionError):
                    PROMOTION.build_ota(ota_arguments)


class AptRollbackAndCasTest(unittest.TestCase):
    def test_selects_highest_authenticated_lower_version(self) -> None:
        packages = """
Package: nuv-agent
Version: 0.1.119
Architecture: arm64
Filename: pool/main/n/nuv-agent/nuv-agent_0.1.119_arm64.deb
Size: 10
SHA256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa

Package: nuv-agent
Version: 0.1.120
Architecture: arm64
Filename: pool/main/n/nuv-agent/nuv-agent_0.1.120_arm64.deb
Size: 11
SHA256: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
""".strip()
        selected = PREPARE_APT.select_rollback_record(
            PREPARE_APT.parse_packages(packages), current_version="0.1.121"
        )
        assert selected is not None
        self.assertEqual(selected["Version"], "0.1.120")
        release = (
            "Origin: NUV\nSHA256:\n "
            + "c" * 64
            + " 123 main/binary-arm64/Packages.gz\n"
        )
        self.assertEqual(
            PREPARE_APT.parse_release_sha256(
                release, "main/binary-arm64/Packages.gz"
            ),
            ("c" * 64, 123),
        )

    def test_apt_passphrase_file_rejects_weak_mode_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            deb = root / "nuv-agent_0.1.121_arm64.deb"
            deb.write_bytes(b"not reached")
            passphrase = root / "apt-passphrase"
            passphrase.write_text("secret", encoding="utf-8")
            passphrase.chmod(0o644)
            environment = {
                **os.environ,
                "APTLY_PASSPHRASE_FILE": str(passphrase),
                "APT_RUNTIME_ROOT": str(root / "runtime"),
            }
            command = [str(ROOT / "packaging/apt/publish-gcs.sh"), str(deb)]
            weak = subprocess.run(
                command, check=False, capture_output=True, text=True, env=environment
            )
            self.assertNotEqual(weak.returncode, 0)
            self.assertIn("mode 0600", weak.stderr)
            passphrase.chmod(0o600)
            symlink = root / "passphrase-link"
            symlink.symlink_to(passphrase)
            environment["APTLY_PASSPHRASE_FILE"] = str(symlink)
            linked = subprocess.run(
                command, check=False, capture_output=True, text=True, env=environment
            )
            self.assertNotEqual(linked.returncode, 0)
            self.assertIn("regular file", linked.stderr)

    def _fake_gcloud(self, root: Path) -> tuple[Path, Path, Path]:
        binary = root / "bin"
        binary.mkdir()
        remote = root / "remote"
        log = root / "log"
        script = binary / "gcloud"
        script.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
echo "$*" >> "$FAKE_GCLOUD_LOG"
[ "$1" = storage ]
case "$2" in
  cp)
    [ "${FAKE_GCLOUD_CP_RC:-0}" = 0 ] || exit "$FAKE_GCLOUD_CP_RC"
    args=("$@")
    cp "${args[${#args[@]}-2]}" "$FAKE_GCLOUD_REMOTE"
    ;;
  cat) cat "$FAKE_GCLOUD_REMOTE" ;;
  *) exit 2 ;;
esac
""",
            encoding="utf-8",
        )
        script.chmod(0o755)
        return binary, remote, log

    def test_generation_zero_cas_accepts_only_identical_concurrent_writer(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            binary, remote, log = self._fake_gcloud(root)
            source = root / "reservation.json"
            source.write_text('{"releaseSequence":2}\n', encoding="utf-8")
            environment = {
                **os.environ,
                "PATH": f"{binary}:{os.environ['PATH']}",
                "FAKE_GCLOUD_REMOTE": str(remote),
                "FAKE_GCLOUD_LOG": str(log),
                "FAKE_GCLOUD_CP_RC": "0",
            }
            command = [
                str(ROOT / "packaging/release/publish-immutable-gcs-file.sh"),
                str(source),
                "apt.plaidai.io",
                "releases/reservations/iq9075/2.json",
            ]
            subprocess.run(command, check=True, capture_output=True, env=environment)
            self.assertEqual(remote.read_bytes(), source.read_bytes())
            self.assertIn("--if-generation-match=0", log.read_text(encoding="utf-8"))
            environment["FAKE_GCLOUD_CP_RC"] = "1"
            subprocess.run(command, check=True, capture_output=True, env=environment)
            remote.write_text("different\n", encoding="utf-8")
            failed = subprocess.run(command, check=False, capture_output=True, env=environment)
            self.assertNotEqual(failed.returncode, 0)

    def test_ota_discovery_is_last_and_every_partial_stage_is_rerunnable(self) -> None:
        from base64 import b64encode
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            artifact = root / "nuv-agent_0.1.121_iq9075-aarch64.agent-bundle.tar.gz"
            artifact.write_bytes(b"exact ota bundle")
            private_key = Ed25519PrivateKey.generate()
            private_raw = private_key.private_bytes(
                serialization.Encoding.Raw,
                serialization.PrivateFormat.Raw,
                serialization.NoEncryption(),
            )
            public_der = private_key.public_key().public_bytes(
                serialization.Encoding.DER,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            keyring = root / "keyring.json"
            keyring.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "trustDomain": "test-ota",
                        "keys": {"test-release": b64encode(public_der).decode("ascii")},
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            bom = root / "release-bom.json"
            signature = root / "release-bom.json.sig"
            generation_environment = {
                **os.environ,
                "TEST_RELEASE_PRIVATE_KEY": b64encode(private_raw).decode("ascii"),
            }
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "packaging/release/generate-release-bom.py"),
                    "--schema-version", "2",
                    "--bom-id", "nuv-agent-0.1.121-iq9075-aarch64",
                    "--version", "0.1.121",
                    "--component-sha", "a" * 40,
                    "--config-schema", "12",
                    "--release-sequence", "2",
                    "--min-updater-version", "0.1.0",
                    "--target", "IQ9075_DEV:iq9075_dev:QCS9075-EVK:aarch64",
                    "--artifact", str(artifact),
                    "--artifact-kind", "agent-bundle",
                    "--built-at", "2026-09-02T00:00:00Z",
                    "--output", str(bom),
                    "--signature-output", str(signature),
                    "--signing-key-id", "test-release",
                    "--signing-private-key-env", "TEST_RELEASE_PRIVATE_KEY",
                ],
                check=True,
                capture_output=True,
                env=generation_environment,
            )

            for failed_stage in range(1, 7):
                with self.subTest(failed_stage=failed_stage):
                    stage = root / f"stage-{failed_stage}"
                    binary = stage / "bin"
                    remote = stage / "remote"
                    public = stage / "public"
                    log = stage / "gcloud.log"
                    counter = stage / "counter"
                    binary.mkdir(parents=True)
                    remote.mkdir()
                    fake = binary / "gcloud"
                    fake.write_text(
                        """#!/usr/bin/env bash
set -euo pipefail
echo "$*" >> "$FAKE_GCLOUD_LOG"
[ "$1" = storage ]
args=("$@")
remote_arg="${args[${#args[@]}-1]}"
relative="${remote_arg#gs://test-bucket/}"
target="$FAKE_GCLOUD_REMOTE/$relative"
case "$2" in
  cp)
    count=0
    [ ! -f "$FAKE_GCLOUD_COUNTER" ] || count=$(cat "$FAKE_GCLOUD_COUNTER")
    count=$((count + 1))
    echo "$count" > "$FAKE_GCLOUD_COUNTER"
    if [ "$count" = "$FAKE_FAIL_STAGE" ]; then
      exit 75
    fi
    [ ! -e "$target" ] || exit 1
    mkdir -p "$(dirname "$target")"
    cp "${args[${#args[@]}-2]}" "$target"
    ;;
  cat) cat "$target" ;;
  *) exit 2 ;;
esac
""",
                        encoding="utf-8",
                    )
                    fake.chmod(0o755)
                    environment = {
                        **os.environ,
                        "PATH": f"{binary}:{Path(sys.executable).parent}:{os.environ['PATH']}",
                        "VERSION": "0.1.121",
                        "BUCKET": "test-bucket",
                        "SKIP_APT_PUBLISH": "true",
                        "APT_PUBLIC_DIR": str(public),
                        "RELEASE_KEYRING_PATH": str(keyring),
                        "RELEASE_TRUST_DOMAIN": "test-ota",
                        "FAKE_GCLOUD_LOG": str(log),
                        "FAKE_GCLOUD_REMOTE": str(remote),
                        "FAKE_GCLOUD_COUNTER": str(counter),
                        "FAKE_FAIL_STAGE": str(failed_stage),
                    }
                    command = [
                        str(ROOT / "packaging/apt/publish-gcs.sh"),
                        str(artifact),
                        str(bom),
                        str(signature),
                        str(artifact),
                    ]
                    first = subprocess.run(
                        command, check=False, capture_output=True, text=True, env=environment
                    )
                    self.assertNotEqual(first.returncode, 0)
                    environment["FAKE_FAIL_STAGE"] = "0"
                    second = subprocess.run(
                        command, check=False, capture_output=True, text=True, env=environment
                    )
                    self.assertEqual(second.returncode, 0, second.stderr)
                    third = subprocess.run(
                        command, check=False, capture_output=True, text=True, env=environment
                    )
                    self.assertEqual(third.returncode, 0, third.stderr)
                    objects = sorted(path for path in remote.rglob("*") if path.is_file())
                    self.assertEqual(len(objects), 6)
                    cp_lines = [
                        line for line in log.read_text(encoding="utf-8").splitlines()
                        if line.startswith("storage cp ")
                    ]
                    self.assertTrue(cp_lines)
                    self.assertTrue(
                        cp_lines[-1].endswith(
                            "gs://test-bucket/releases/0.1.121/release-bom.json"
                        )
                    )


class SettingsPolicyTest(unittest.TestCase):
    def test_general_writers_require_hardened_review_with_single_admin_bypass(self) -> None:
        policy = json.loads(
            (ROOT / "packaging/release/release-security-policy.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            policy["governance"],
            {
                "pullRequestApprovals": 1,
                "dismissStaleReviewsOnPush": True,
                "requireCodeOwnerReview": True,
                "requireLastPushApproval": True,
                "requireExtraApprovalForUnattributedChanges": True,
                "requiredReviewThreadResolution": True,
                "allowedMergeMethods": ["merge", "squash", "rebase"],
                "environmentReviewers": 0,
                "requiredStatusContext": "agent-release-gate",
                "requiredStatusIntegrationId": 15368,
            },
        )
        self.assertEqual(
            set(policy["requiredEnvironments"]),
            {
                "homebrew-release",
                "apt-release",
                "iq9075-release",
                "face-artifacts-release",
            },
        )
        for environment in policy["requiredEnvironments"].values():
            self.assertFalse(environment["requireReviewers"])
            self.assertFalse(environment["preventSelfReview"])
            self.assertIsNone(environment["reviewerTeamId"])
            self.assertEqual(
                environment["deploymentBranchPolicy"],
                {"protectedBranches": False, "customBranchPolicies": True},
            )
            self.assertEqual(
                environment["deploymentBranchPolicies"],
                [{"name": "main", "type": "branch"}],
            )
            self.assertEqual(environment["protectionRuleTypes"], ["branch_policy"])
        codeowners = (ROOT / ".github/CODEOWNERS").read_text(encoding="utf-8")
        self.assertIn("/.github/workflows/** @plaid-ai/platform-admin", codeowners)
        self.assertIn("/packaging/** @plaid-ai/platform-admin", codeowners)
        self.assertIn("/nuvion_updater/** @plaid-ai/platform-admin", codeowners)
        self.assertNotIn("GITHUB_RELEASE_TOKEN", json.dumps(policy))
        self.assertEqual(
            policy["forbiddenRepositorySecrets"],
            [
                "APT_GPG_PASSPHRASE",
                "APT_GPG_PRIVATE_KEY",
                "GCP_PROJECT_ID",
                "GCP_SA_KEY",
                "HOMEBREW_TAP_TOKEN",
                "IQ9075_RELEASE_PUBLIC_KEYRING_JSON",
                "IQ9075_RELEASE_SIGNING_KEY_ID",
                "IQ9075_RELEASE_SIGNING_PRIVATE_KEY",
            ],
        )

    def test_settings_audit_accepts_no_environment_wait_and_rejects_reviewer_rule(self) -> None:
        branch_ruleset = {
            "id": 1,
            "name": "protected-main",
            "source": "plaid-ai/NUV-AGENT",
            "source_type": "Repository",
            "target": "branch",
            "enforcement": "active",
            "conditions": {
                "ref_name": {"include": ["refs/heads/main"], "exclude": []}
            },
            "bypass_actors": [
                {
                    "actor_id": 16128529,
                    "actor_type": "Team",
                    "bypass_mode": "pull_request",
                }
            ],
            "rules": [
                {"type": "deletion"},
                {"type": "non_fast_forward"},
                {
                    "type": "pull_request",
                    "parameters": {
                        "allowed_merge_methods": ["merge", "squash", "rebase"],
                        "dismiss_stale_reviews_on_push": True,
                        "dismissal_restriction": {
                            "allowed_actors": [],
                            "enabled": False,
                        },
                        "require_code_owner_review": True,
                        "require_extra_approval_for_unattributed_changes": True,
                        "require_last_push_approval": True,
                        "required_approving_review_count": 1,
                        "required_review_thread_resolution": True,
                        "required_reviewers": [],
                    },
                },
                {
                    "type": "required_status_checks",
                    "parameters": {
                        "strict_required_status_checks_policy": True,
                        "do_not_enforce_on_create": False,
                        "required_status_checks": [
                            {
                                "context": "agent-release-gate",
                                "integration_id": 15368,
                            }
                        ],
                    },
                },
            ],
        }
        tag_ruleset = {
            "id": 2,
            "name": "protected-release-tags",
            "source": "plaid-ai/NUV-AGENT",
            "source_type": "Repository",
            "target": "tag",
            "enforcement": "active",
            "conditions": {
                "ref_name": {"include": ["refs/tags/v*"], "exclude": []}
            },
            "bypass_actors": [
                {
                    "actor_id": 16128529,
                    "actor_type": "Team",
                    "bypass_mode": "always",
                }
            ],
            "rules": [
                {"type": "creation"},
                {"type": "deletion"},
                {"type": "non_fast_forward"},
            ],
        }
        responses: dict[str, object] = {
            "/repos/plaid-ai/NUV-AGENT": {
                "id": 1149331364,
                "default_branch": "main",
                "private": False,
                "owner": {"login": "plaid-ai", "type": "Organization"},
            },
            "/repos/plaid-ai/NUV-AGENT/immutable-releases": {"enabled": True},
            "/repos/plaid-ai/NUV-AGENT/branches/main": {"protected": True},
            "/repos/plaid-ai/NUV-AGENT/rulesets?includes_parents=true&per_page=100&page=1": [
                {"id": 1},
                {"id": 2},
            ],
            "/repos/plaid-ai/NUV-AGENT/rulesets/1": branch_ruleset,
            "/repos/plaid-ai/NUV-AGENT/rulesets/2": tag_ruleset,
            "/repos/plaid-ai/NUV-AGENT/actions/permissions/workflow": {
                "default_workflow_permissions": "read",
                "can_approve_pull_request_reviews": False,
            },
        }
        for name in (
            "homebrew-release",
            "apt-release",
            "iq9075-release",
            "face-artifacts-release",
        ):
            responses[f"/repos/plaid-ai/NUV-AGENT/environments/{name}"] = {
                "name": name,
                "deployment_branch_policy": {
                    "protected_branches": False,
                    "custom_branch_policies": True,
                },
                "protection_rules": [{"id": 1, "type": "branch_policy"}],
            }
            responses[
                f"/repos/plaid-ai/NUV-AGENT/environments/{name}/deployment-branch-policies?per_page=100&page=1"
            ] = {
                "total_count": 1,
                "branch_policies": [{"id": 1, "name": "main", "type": "branch"}],
            }

        policy = json.loads(
            (ROOT / "packaging/release/release-security-policy.json").read_text(
                encoding="utf-8"
            )
        )
        responses[
            "/repos/plaid-ai/NUV-AGENT/actions/variables/RELEASE_SECURITY_POLICY_VERSION"
        ] = {"value": "1"}
        responses[
            "/repos/plaid-ai/NUV-AGENT/actions/variables/RELEASE_TRUSTED_PUBLISHER_SHA"
        ] = {"value": "a" * 40}
        responses[
            "/repos/plaid-ai/NUV-AGENT/actions/secrets?per_page=100&page=1"
        ] = {"total_count": 0, "secrets": []}
        responses[
            "/orgs/plaid-ai/actions/secrets?per_page=100&page=1"
        ] = {"total_count": 0, "secrets": []}
        for name, requirements in policy["requiredEnvironments"].items():
            responses[
                f"/repos/plaid-ai/NUV-AGENT/environments/{name}/secrets?per_page=100&page=1"
            ] = {
                "total_count": len(requirements["requiredSecrets"]),
                "secrets": [
                    {"name": secret} for secret in requirements["requiredSecrets"]
                ],
            }

        fake_api = mock.Mock()
        fake_api.get.side_effect = lambda path: responses[path]
        fake_api.get_optional.return_value = SETTINGS.API_NOT_FOUND
        with mock.patch.object(SETTINGS, "GitHubApi", return_value=fake_api):
            result = SETTINGS.verify_settings(
                repository="plaid-ai/NUV-AGENT",
                token="metadata-only",
                policy_path=ROOT / "packaging/release/release-security-policy.json",
                trusted_publisher_sha="a" * 40,
                include_secret_scopes=False,
            )
            self.assertEqual(result["governance"]["pullRequestApprovals"], 1)
            result = SETTINGS.verify_settings(
                repository="plaid-ai/NUV-AGENT",
                token="admin-metadata-only",
                policy_path=ROOT / "packaging/release/release-security-policy.json",
                trusted_publisher_sha="a" * 40,
                include_secret_scopes=True,
            )
            self.assertTrue(result["secretScopesChecked"])
            responses[
                "/repos/plaid-ai/NUV-AGENT/environments/homebrew-release"
            ]["protection_rules"].append({"type": "required_reviewers"})
            with self.assertRaises(SETTINGS.SettingsError):
                SETTINGS.verify_settings(
                    repository="plaid-ai/NUV-AGENT",
                    token="metadata-only",
                    policy_path=ROOT / "packaging/release/release-security-policy.json",
                    trusted_publisher_sha="a" * 40,
                    include_secret_scopes=False,
                )
            responses[
                "/repos/plaid-ai/NUV-AGENT/environments/homebrew-release"
            ]["protection_rules"].pop()
            fake_api.get_optional.return_value = {
                "required_pull_request_reviews": None,
                "required_status_checks": None,
            }
            with self.assertRaises(SETTINGS.SettingsError):
                SETTINGS.verify_settings(
                    repository="plaid-ai/NUV-AGENT",
                    token="metadata-only",
                    policy_path=ROOT / "packaging/release/release-security-policy.json",
                    trusted_publisher_sha="a" * 40,
                    include_secret_scopes=False,
                )
            fake_api.get_optional.return_value = SETTINGS.API_NOT_FOUND
            branch_policy_path = (
                "/repos/plaid-ai/NUV-AGENT/environments/homebrew-release/"
                "deployment-branch-policies?per_page=100&page=1"
            )
            responses[branch_policy_path]["total_count"] = 2
            responses[branch_policy_path]["branch_policies"].append(
                {"id": 2, "name": "develop", "type": "branch"}
            )
            with self.assertRaises(SETTINGS.SettingsError):
                SETTINGS.verify_settings(
                    repository="plaid-ai/NUV-AGENT",
                    token="metadata-only",
                    policy_path=ROOT / "packaging/release/release-security-policy.json",
                    trusted_publisher_sha="a" * 40,
                    include_secret_scopes=False,
                )
            responses[branch_policy_path]["total_count"] = 1
            responses[branch_policy_path]["branch_policies"].pop()
            extra = json.loads(json.dumps(branch_ruleset))
            extra["id"] = 3
            extra["rules"][2]["parameters"]["required_approving_review_count"] = 2
            responses[
                "/repos/plaid-ai/NUV-AGENT/rulesets?includes_parents=true&per_page=100&page=1"
            ].append({"id": 3})
            responses["/repos/plaid-ai/NUV-AGENT/rulesets/3"] = extra
            with self.assertRaises(SETTINGS.SettingsError):
                SETTINGS.verify_settings(
                    repository="plaid-ai/NUV-AGENT",
                    token="metadata-only",
                    policy_path=ROOT / "packaging/release/release-security-policy.json",
                    trusted_publisher_sha="a" * 40,
                    include_secret_scopes=False,
                )

    def test_settings_api_pagination_and_org_secret_scope_are_fail_closed(self) -> None:
        fake_api = mock.Mock()
        list_page = [{"id": value} for value in range(1, 101)]
        fake_api.get.side_effect = lambda path: {
            "/rulesets?includes_parents=true&per_page=100&page=1": list_page,
            "/rulesets?includes_parents=true&per_page=100&page=2": [{"id": 101}],
        }[path]
        self.assertEqual(
            len(
                SETTINGS._paginated_list(
                    fake_api,
                    "/rulesets?includes_parents=true",
                    label="rulesets",
                )
            ),
            101,
        )

        fake_api.get.side_effect = lambda path: {
            "/branch-policies?per_page=100&page=1": {
                "total_count": 1,
                "branch_policies": [{"name": "main", "type": "branch"}],
            }
        }[path]
        self.assertEqual(
            SETTINGS._paginated_collection(
                fake_api,
                "/branch-policies",
                member="branch_policies",
                label="branch policies",
            ),
            [{"name": "main", "type": "branch"}],
        )
        secret_page = [{"name": f"SECRET_{value:03d}"} for value in range(100)]
        fake_api.get.side_effect = lambda path: {
            "/actions/secrets?per_page=100&page=1": {
                "total_count": 101,
                "secrets": secret_page,
            },
            "/actions/secrets?per_page=100&page=2": {
                "total_count": 101,
                "secrets": [{"name": "GCP_SA_KEY"}],
            },
        }[path]
        paginated_secrets = SETTINGS._paginated_collection(
            fake_api,
            "/actions/secrets",
            member="secrets",
            label="secrets",
        )
        self.assertIn("GCP_SA_KEY", SETTINGS._secret_names(paginated_secrets, label="secrets"))
        fake_api.get.side_effect = lambda path: {
            "/branch-policies?per_page=100&page=1": {
                "total_count": 2,
                "branch_policies": [{"name": "main", "type": "branch"}],
            }
        }[path]
        with self.assertRaises(SETTINGS.SettingsError):
            SETTINGS._paginated_collection(
                fake_api,
                "/branch-policies",
                member="branch_policies",
                label="branch policies",
            )

        fake_api.get.side_effect = lambda path: {
            "/orgs/plaid-ai/actions/secrets/GCP_SA_KEY/repositories?per_page=100&page=1": {
                "total_count": 1,
                "repositories": [
                    {"id": 1149331364, "full_name": "plaid-ai/NUV-AGENT"}
                ],
            }
        }[path]
        self.assertTrue(
            SETTINGS._organization_secret_applies(
                fake_api,
                repository="plaid-ai/NUV-AGENT",
                repository_id=1149331364,
                repository_private=False,
                organization="plaid-ai",
                secret={"name": "GCP_SA_KEY", "visibility": "selected"},
            )
        )
        self.assertFalse(
            SETTINGS._organization_secret_applies(
                fake_api,
                repository="plaid-ai/NUV-AGENT",
                repository_id=1149331364,
                repository_private=False,
                organization="plaid-ai",
                secret={"name": "GCP_SA_KEY", "visibility": "private"},
            )
        )

    def test_classic_protection_probe_accepts_only_http_404(self) -> None:
        api = SETTINGS.GitHubApi("plaid-ai/NUV-AGENT", "metadata-token")
        not_found = SETTINGS.urllib.error.HTTPError(
            "https://api.github.test/protection", 404, "not found", None, None
        )
        with mock.patch.object(
            SETTINGS.urllib.request, "urlopen", side_effect=not_found
        ):
            self.assertIs(
                api.get_optional("/repos/plaid-ai/NUV-AGENT/branches/main/protection"),
                SETTINGS.API_NOT_FOUND,
            )
        denied = SETTINGS.urllib.error.HTTPError(
            "https://api.github.test/protection", 403, "forbidden", None, None
        )
        with mock.patch.object(
            SETTINGS.urllib.request, "urlopen", side_effect=denied
        ), self.assertRaises(SETTINGS.SettingsError):
            api.get_optional("/repos/plaid-ai/NUV-AGENT/branches/main/protection")

    def test_publisher_surface_binds_every_tracked_helper_and_workflow_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            repository = root / "publisher"
            workflow = repository / ".github/workflows/release-publish.yml"
            face_workflow = repository / ".github/workflows/publish-face-artifacts.yml"
            helper = repository / "packaging/release/helper.sh"
            workflow.parent.mkdir(parents=True)
            helper.parent.mkdir(parents=True)
            workflow.write_text("name: trusted\n", encoding="utf-8")
            face_workflow.write_text("name: trusted-face\n", encoding="utf-8")
            helper.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            helper.chmod(0o755)
            subprocess.run(["git", "init", "-b", "main", repository], check=True, capture_output=True)
            subprocess.run(["git", "-C", repository, "config", "user.name", "Test"], check=True)
            subprocess.run(["git", "-C", repository, "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", repository, "add", "."], check=True)
            subprocess.run(["git", "-C", repository, "commit", "-m", "publisher"], check=True, capture_output=True)
            commit = subprocess.check_output(
                ["git", "-C", repository, "rev-parse", "HEAD"], text=True
            ).strip()
            surface = PUBLISHER_TRUST.publisher_surface(repository, expected_sha=commit)
            executing = root / "executing.yml"
            executing.write_bytes(workflow.read_bytes())
            PUBLISHER_TRUST.verify_executing_workflow(
                repository,
                executing,
                expected_workflow_sha256=surface["workflowSha256"],
            )
            executing_face = root / "executing-face.yml"
            executing_face.write_bytes(face_workflow.read_bytes())
            PUBLISHER_TRUST.verify_additional_executing_workflow(
                repository,
                executing_face,
                publisher_relative_path=".github/workflows/publish-face-artifacts.yml",
            )
            executing_face.write_text("name: attacker-face\n", encoding="utf-8")
            with self.assertRaises(PUBLISHER_TRUST.PublisherTrustError):
                PUBLISHER_TRUST.verify_additional_executing_workflow(
                    repository,
                    executing_face,
                    publisher_relative_path=".github/workflows/publish-face-artifacts.yml",
                )
            with self.assertRaises(PUBLISHER_TRUST.PublisherTrustError):
                PUBLISHER_TRUST.verify_additional_executing_workflow(
                    repository,
                    executing_face,
                    publisher_relative_path="../../attacker.yml",
                )
            subprocess.run(
                [
                    "git",
                    "-C",
                    repository,
                    "update-index",
                    "--assume-unchanged",
                    "packaging/release/helper.sh",
                ],
                check=True,
            )
            helper.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
            with self.assertRaises(PUBLISHER_TRUST.PublisherTrustError):
                PUBLISHER_TRUST.publisher_surface(repository, expected_sha=commit)
            subprocess.run(
                [
                    "git",
                    "-C",
                    repository,
                    "update-index",
                    "--no-assume-unchanged",
                    "packaging/release/helper.sh",
                ],
                check=True,
            )
            helper.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            executing.write_text("name: attacker\n", encoding="utf-8")
            with self.assertRaises(PUBLISHER_TRUST.PublisherTrustError):
                PUBLISHER_TRUST.verify_executing_workflow(
                    repository,
                    executing,
                    expected_workflow_sha256=surface["workflowSha256"],
                )

    def test_required_ruleset_matching_is_exact(self) -> None:
        rulesets = [
            {
                "target": "tag",
                "enforcement": "active",
                "conditions": {
                    "ref_name": {"include": ["refs/tags/v*"], "exclude": []}
                },
                "rules": [
                    {"type": "creation"},
                    {"type": "deletion"},
                    {"type": "non_fast_forward"},
                ],
            }
        ]
        self.assertTrue(
            SETTINGS._ruleset_covers(
                rulesets,
                target="tag",
                include="refs/tags/v*",
                required_rules={"creation", "deletion", "non_fast_forward"},
            )
        )
        rulesets[0]["enforcement"] = "evaluate"
        self.assertFalse(
            SETTINGS._ruleset_covers(
                rulesets,
                target="tag",
                include="refs/tags/v*",
                required_rules={"creation", "deletion", "non_fast_forward"},
            )
        )

    def test_main_ruleset_requires_real_agent_release_gate_context(self) -> None:
        rulesets = [
            {
                "name": "protected-main",
                "source": "plaid-ai/NUV-AGENT",
                "source_type": "Repository",
                "target": "branch",
                "enforcement": "active",
                "conditions": {
                    "ref_name": {"include": ["refs/heads/main"], "exclude": []}
                },
                "rules": [
                    {"type": "deletion"},
                    {"type": "non_fast_forward"},
                    {
                        "type": "pull_request",
                        "parameters": {
                            "allowed_merge_methods": ["merge", "squash", "rebase"],
                            "dismiss_stale_reviews_on_push": True,
                            "dismissal_restriction": {
                                "allowed_actors": [],
                                "enabled": False,
                            },
                            "require_code_owner_review": True,
                            "require_extra_approval_for_unattributed_changes": True,
                            "require_last_push_approval": True,
                            "required_approving_review_count": 1,
                            "required_review_thread_resolution": True,
                            "required_reviewers": [],
                        },
                    },
                    {
                        "type": "required_status_checks",
                        "parameters": {
                            "strict_required_status_checks_policy": True,
                            "do_not_enforce_on_create": False,
                            "required_status_checks": [
                                {
                                    "context": "agent-release-gate",
                                    "integration_id": 15368,
                                }
                            ]
                        },
                    },
                ],
            }
        ]
        arguments = {
            "target": "branch",
            "include": "refs/heads/main",
            "required_name": "protected-main",
            "required_source": "plaid-ai/NUV-AGENT",
            "required_rules": {
                "deletion",
                "non_fast_forward",
                "pull_request",
                "required_status_checks",
            },
            "required_status_context": "agent-release-gate",
            "required_status_integration_id": 15368,
            "required_pull_request_approvals": 1,
        }
        self.assertTrue(SETTINGS._ruleset_covers(rulesets, **arguments))
        mutations = [
            ("name", lambda value: value[0].update(name="almost-protected-main")),
            ("source", lambda value: value[0].update(source_type="Organization")),
            ("exclude", lambda value: value[0]["conditions"]["ref_name"].update(exclude=["refs/heads/main-hotfix"])),
            ("approval-count", lambda value: value[0]["rules"][2]["parameters"].update(required_approving_review_count=2)),
            ("dismiss-stale", lambda value: value[0]["rules"][2]["parameters"].update(dismiss_stale_reviews_on_push=False)),
            ("code-owner", lambda value: value[0]["rules"][2]["parameters"].update(require_code_owner_review=False)),
            ("last-push", lambda value: value[0]["rules"][2]["parameters"].update(require_last_push_approval=False)),
            ("unattributed", lambda value: value[0]["rules"][2]["parameters"].update(require_extra_approval_for_unattributed_changes=False)),
            ("merge-method", lambda value: value[0]["rules"][2]["parameters"]["allowed_merge_methods"].remove("rebase")),
            ("fixed-reviewer", lambda value: value[0]["rules"][2]["parameters"]["required_reviewers"].append({"type": "User", "id": 1})),
            ("resolve-threads", lambda value: value[0]["rules"][2]["parameters"].update(required_review_thread_resolution=False)),
            ("strict", lambda value: value[0]["rules"][3]["parameters"].update(strict_required_status_checks_policy=False)),
            ("create", lambda value: value[0]["rules"][3]["parameters"].update(do_not_enforce_on_create=True)),
            ("integration", lambda value: value[0]["rules"][3]["parameters"]["required_status_checks"][0].update(integration_id=1)),
            ("context", lambda value: value[0]["rules"][3]["parameters"]["required_status_checks"][0].update(context="nonexistent-check")),
            ("extra-check", lambda value: value[0]["rules"][3]["parameters"]["required_status_checks"].append({"context": "extra", "integration_id": 15368})),
        ]
        for label, mutate in mutations:
            candidate = json.loads(json.dumps(rulesets))
            mutate(candidate)
            with self.subTest(label=label):
                self.assertFalse(SETTINGS._ruleset_covers(candidate, **arguments))

        duplicate = json.loads(json.dumps(rulesets)) + json.loads(json.dumps(rulesets))
        self.assertFalse(SETTINGS._ruleset_covers(duplicate, **arguments))
        extra_rule = json.loads(json.dumps(rulesets))
        extra_rule[0]["rules"].append({"type": "required_signatures"})
        self.assertFalse(SETTINGS._ruleset_covers(extra_rule, **arguments))

    def test_short_lived_settings_attestation_binds_policy_and_expiry(self) -> None:
        settings_verifier = (
            ROOT / "packaging/release/verify-github-release-settings.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '"RELEASE_TRUSTED_PUBLISHER_SHA": trusted_publisher_sha',
            settings_verifier,
        )
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            policy = ROOT / "packaging/release/release-security-policy.json"
            now = dt.datetime(2026, 9, 2, 0, 0, tzinfo=dt.timezone.utc)
            attestation = {
                "schemaVersion": 1,
                "kind": "nuvion-release-settings-attestation",
                "repository": "plaid-ai/NUV-AGENT",
                "trustedPublisherSha": "a" * 40,
                "publisherTreeSha256": "b" * 64,
                "workflowSha256": "c" * 64,
                "policySha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
                "verifiedAt": "2026-09-02T00:00:00Z",
                "expiresAt": "2026-09-03T00:00:00Z",
                "settings": {
                    "defaultBranch": "main",
                    "governance": json.loads(policy.read_text(encoding="utf-8"))[
                        "governance"
                    ],
                    "secretScopesChecked": True,
                    "status": "VERIFIED",
                },
            }
            path = root / "attestation.json"
            path.write_text(
                json.dumps(attestation, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            signature = root / "attestation.json.asc"
            signature.write_text("test-signature\n", encoding="utf-8")
            with mock.patch.object(
                SETTINGS_ATTESTATION,
                "_verify_signature",
                return_value="9A07D327F3ADF6F452A4BF0055E5CAF706571888",
            ), mock.patch.object(
                SETTINGS_ATTESTATION,
                "publisher_surface",
                return_value={
                    "publisherTreeSha256": "b" * 64,
                    "workflowSha256": "c" * 64,
                },
            ), mock.patch.object(
                SETTINGS_ATTESTATION, "verify_executing_workflow"
            ):
                result = SETTINGS_ATTESTATION.verify_attestation(
                    attestation_path=path,
                    signature_path=signature,
                    policy_path=policy,
                    signer_directory=ROOT / "packaging/release/trusted-tag-signers",
                    repository="plaid-ai/NUV-AGENT",
                    trusted_publisher_sha="a" * 40,
                    publisher_root=ROOT,
                    executing_workflow=ROOT / ".github/workflows/release-publish.yml",
                    now=now,
                )
                self.assertEqual(result["status"], "VERIFIED")
                self.assertEqual(result["trustedPublisherSha"], "a" * 40)
                with self.assertRaises(SETTINGS_ATTESTATION.AttestationError):
                    SETTINGS_ATTESTATION.verify_attestation(
                        attestation_path=path,
                        signature_path=signature,
                        policy_path=policy,
                        signer_directory=ROOT / "packaging/release/trusted-tag-signers",
                        repository="plaid-ai/NUV-AGENT",
                        trusted_publisher_sha="b" * 40,
                        publisher_root=ROOT,
                        executing_workflow=ROOT / ".github/workflows/release-publish.yml",
                        now=now,
                    )
                with self.assertRaises(SETTINGS_ATTESTATION.AttestationError):
                    SETTINGS_ATTESTATION.verify_attestation(
                        attestation_path=path,
                        signature_path=signature,
                        policy_path=policy,
                        signer_directory=ROOT / "packaging/release/trusted-tag-signers",
                        repository="plaid-ai/NUV-AGENT",
                        trusted_publisher_sha="a" * 40,
                        publisher_root=ROOT,
                        executing_workflow=ROOT / ".github/workflows/release-publish.yml",
                        now=now + dt.timedelta(days=1),
                    )

                moments = iter(
                    (
                        now,
                        now + dt.timedelta(days=1),
                    )
                )
                with self.assertRaisesRegex(
                    SETTINGS_ATTESTATION.AttestationError,
                    "expired during verification",
                ):
                    SETTINGS_ATTESTATION.verify_attestation(
                        attestation_path=path,
                        signature_path=signature,
                        policy_path=policy,
                        signer_directory=(
                            ROOT / "packaging/release/trusted-tag-signers"
                        ),
                        repository="plaid-ai/NUV-AGENT",
                        trusted_publisher_sha="a" * 40,
                        publisher_root=ROOT,
                        executing_workflow=(
                            ROOT / ".github/workflows/release-publish.yml"
                        ),
                        clock=lambda: next(moments),
                    )

    @unittest.skipUnless(shutil.which("gpg"), "gpg is required")
    def test_settings_attestation_requires_allowlisted_gpg_signature(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            gpg_home = root / "gpg"
            gpg_home.mkdir(mode=0o700)
            environment = {**os.environ, "GNUPGHOME": str(gpg_home)}
            subprocess.run(
                [
                    "gpg",
                    "--batch",
                    "--passphrase",
                    "",
                    "--quick-generate-key",
                    "Settings Auditor <settings@example.invalid>",
                    "ed25519",
                    "cert",
                    "1d",
                ],
                check=True,
                capture_output=True,
                env=environment,
            )
            listing = subprocess.check_output(
                ["gpg", "--batch", "--with-colons", "--list-keys"],
                text=True,
                env=environment,
            )
            fingerprint = next(
                line.split(":")[9]
                for line in listing.splitlines()
                if line.startswith("fpr:")
            )
            subprocess.run(
                [
                    "gpg",
                    "--batch",
                    "--passphrase",
                    "",
                    "--quick-add-key",
                    fingerprint,
                    "ed25519",
                    "sign",
                    "1d",
                ],
                check=True,
                capture_output=True,
                env=environment,
            )
            policy_payload = json.loads(
                (ROOT / "packaging/release/release-security-policy.json").read_text(
                    encoding="utf-8"
                )
            )
            policy_payload["trustedTagSignerFingerprints"] = [fingerprint]
            policy = root / "policy.json"
            policy.write_text(
                json.dumps(policy_payload, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            attestation_payload = {
                "schemaVersion": 1,
                "kind": "nuvion-release-settings-attestation",
                "repository": "plaid-ai/NUV-AGENT",
                "trustedPublisherSha": "a" * 40,
                "publisherTreeSha256": "b" * 64,
                "workflowSha256": "c" * 64,
                "policySha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
                "verifiedAt": "2026-09-02T00:00:00Z",
                "expiresAt": "2026-09-02T12:00:00Z",
                "settings": {
                    "defaultBranch": "main",
                    "governance": policy_payload["governance"],
                    "secretScopesChecked": True,
                    "status": "VERIFIED",
                },
            }
            attestation = root / "attestation.json"
            attestation.write_text(
                json.dumps(attestation_payload, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            signature = root / "attestation.json.asc"
            subprocess.run(
                [
                    "gpg",
                    "--batch",
                    "--armor",
                    "--detach-sign",
                    "--local-user",
                    fingerprint,
                    "--output",
                    str(signature),
                    str(attestation),
                ],
                check=True,
                capture_output=True,
                env=environment,
            )
            signers = root / "signers"
            signers.mkdir()
            (signers / "auditor.asc").write_bytes(
                subprocess.check_output(
                    ["gpg", "--batch", "--armor", "--export", fingerprint],
                    env=environment,
                )
            )
            with mock.patch.object(
                SETTINGS_ATTESTATION,
                "publisher_surface",
                return_value={
                    "publisherTreeSha256": "b" * 64,
                    "workflowSha256": "c" * 64,
                },
            ), mock.patch.object(
                SETTINGS_ATTESTATION, "verify_executing_workflow"
            ):
                result = SETTINGS_ATTESTATION.verify_attestation(
                    attestation_path=attestation,
                    signature_path=signature,
                    policy_path=policy,
                    signer_directory=signers,
                    repository="plaid-ai/NUV-AGENT",
                    trusted_publisher_sha="a" * 40,
                    publisher_root=ROOT,
                    executing_workflow=ROOT / ".github/workflows/release-publish.yml",
                    now=dt.datetime(2026, 9, 2, 1, 0, tzinfo=dt.timezone.utc),
                )
            self.assertEqual(result["signerFingerprint"], fingerprint)

    def test_ruleset_bypass_is_exact_release_admin_team(self) -> None:
        rulesets = [
            {
                "target": "tag",
                "enforcement": "active",
                "conditions": {
                    "ref_name": {"include": ["refs/tags/v*"], "exclude": []}
                },
                "rules": [
                    {"type": "creation"},
                    {"type": "deletion"},
                    {"type": "non_fast_forward"},
                ],
                "bypass_actors": [
                    {
                        "actor_id": 16128529,
                        "actor_type": "Team",
                        "bypass_mode": "always",
                    }
                ],
            }
        ]
        arguments = {
            "target": "tag",
            "include": "refs/tags/v*",
            "required_rules": {"creation", "deletion", "non_fast_forward"},
            "required_bypass_team_id": 16128529,
            "required_bypass_mode": "always",
        }
        self.assertTrue(SETTINGS._ruleset_covers(rulesets, **arguments))
        rulesets[0]["bypass_actors"].append(
            {"actor_id": 1, "actor_type": "User", "bypass_mode": "always"}
        )
        self.assertFalse(SETTINGS._ruleset_covers(rulesets, **arguments))


class FaceArtifactManifestTest(unittest.TestCase):
    @staticmethod
    def _artifacts(root: Path) -> Path:
        artifacts = root / "artifacts"
        artifacts.mkdir()
        (artifacts / "face_detector.onnx").write_bytes(b"onnx-model")
        (artifacts / "face_detector.plan").write_bytes(b"tensorrt-plan")
        (artifacts / "face_detector.config.pbtxt").write_bytes(b"name: face\n")
        return artifacts

    @unittest.skipUnless(shutil.which("gpg"), "gpg is required")
    def test_signed_manifest_binds_release_commit_model_channel_and_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            artifacts = self._artifacts(root)
            common = {
                "repository": "plaid-ai/NUV-AGENT",
                "release_tag": "v0.1.121",
                "component_sha": "a" * 40,
                "model_name": "anomalyclip",
                "model_version": "v0002",
                "channel_pointer": "gs://nuv-model/pointers/anomalyclip/prod.json",
                "artifact_directory": artifacts,
            }
            manifest_payload = FACE_MANIFEST.build_manifest(**common)
            manifest = root / "face-artifact-manifest.json"
            manifest.write_text(
                json.dumps(manifest_payload, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )

            gpg_home = root / "gpg"
            gpg_home.mkdir(mode=0o700)
            environment = {**os.environ, "GNUPGHOME": str(gpg_home)}
            subprocess.run(
                [
                    "gpg",
                    "--batch",
                    "--passphrase",
                    "",
                    "--quick-generate-key",
                    "Face Release Signer <face@example.invalid>",
                    "ed25519",
                    "cert",
                    "1d",
                ],
                check=True,
                capture_output=True,
                env=environment,
            )
            listing = subprocess.check_output(
                ["gpg", "--batch", "--with-colons", "--list-keys"],
                text=True,
                env=environment,
            )
            fingerprint = next(
                line.split(":")[9]
                for line in listing.splitlines()
                if line.startswith("fpr:")
            )
            subprocess.run(
                [
                    "gpg",
                    "--batch",
                    "--passphrase",
                    "",
                    "--quick-add-key",
                    fingerprint,
                    "ed25519",
                    "sign",
                    "1d",
                ],
                check=True,
                capture_output=True,
                env=environment,
            )
            policy_payload = json.loads(
                (ROOT / "packaging/release/release-security-policy.json").read_text(
                    encoding="utf-8"
                )
            )
            policy_payload["trustedTagSignerFingerprints"] = [fingerprint]
            policy = root / "policy.json"
            policy.write_text(
                json.dumps(policy_payload, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            signature = root / "face-artifact-manifest.json.asc"
            subprocess.run(
                [
                    "gpg",
                    "--batch",
                    "--armor",
                    "--detach-sign",
                    "--local-user",
                    fingerprint,
                    "--output",
                    str(signature),
                    str(manifest),
                ],
                check=True,
                capture_output=True,
                env=environment,
            )
            signers = root / "signers"
            signers.mkdir()
            (signers / "face-release.asc").write_bytes(
                subprocess.check_output(
                    ["gpg", "--batch", "--armor", "--export", fingerprint],
                    env=environment,
                )
            )

            result = FACE_MANIFEST.verify_manifest(
                manifest_path=manifest,
                signature_path=signature,
                policy_path=policy,
                signer_directory=signers,
                **common,
            )
            self.assertEqual(result["status"], "VERIFIED")
            self.assertEqual(result["signerFingerprint"], fingerprint)

            untrusted_policy_payload = json.loads(json.dumps(policy_payload))
            untrusted_policy_payload["trustedTagSignerFingerprints"] = ["B" * 40]
            untrusted_policy = root / "untrusted-policy.json"
            untrusted_policy.write_text(
                json.dumps(
                    untrusted_policy_payload, sort_keys=True, separators=(",", ":")
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(FACE_MANIFEST.FaceManifestError):
                FACE_MANIFEST.verify_manifest(
                    manifest_path=manifest,
                    signature_path=signature,
                    policy_path=untrusted_policy,
                    signer_directory=signers,
                    **common,
                )

            original = (artifacts / "face_detector.onnx").read_bytes()
            (artifacts / "face_detector.onnx").write_bytes(b"evil-model")
            with self.assertRaises(FACE_MANIFEST.FaceManifestError):
                FACE_MANIFEST.verify_manifest(
                    manifest_path=manifest,
                    signature_path=signature,
                    policy_path=policy,
                    signer_directory=signers,
                    **common,
                )
            (artifacts / "face_detector.onnx").write_bytes(original)

            original_manifest = manifest.read_bytes()
            tampered_manifest = json.loads(original_manifest)
            tampered_manifest["artifacts"]["face_detector.onnx"]["sha256"] = "0" * 64
            manifest.write_text(
                json.dumps(tampered_manifest, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(FACE_MANIFEST.FaceManifestError):
                FACE_MANIFEST.verify_manifest(
                    manifest_path=manifest,
                    signature_path=signature,
                    policy_path=policy,
                    signer_directory=signers,
                    **common,
                )
            manifest.write_bytes(original_manifest)

            manifest.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")
            noncanonical_signature = root / "noncanonical.asc"
            subprocess.run(
                [
                    "gpg",
                    "--batch",
                    "--armor",
                    "--detach-sign",
                    "--local-user",
                    fingerprint,
                    "--output",
                    str(noncanonical_signature),
                    str(manifest),
                ],
                check=True,
                capture_output=True,
                env=environment,
            )
            with self.assertRaises(FACE_MANIFEST.FaceManifestError):
                FACE_MANIFEST.verify_manifest(
                    manifest_path=manifest,
                    signature_path=noncanonical_signature,
                    policy_path=policy,
                    signer_directory=signers,
                    **common,
                )
            manifest.write_bytes(original_manifest)

            for changed in (
                {**common, "release_tag": "v0.1.122"},
                {**common, "component_sha": "b" * 40},
                {**common, "model_version": "v0003"},
                {
                    **common,
                    "channel_pointer": "gs://nuv-model/pointers/anomalyclip/canary.json",
                },
            ):
                with self.subTest(changed=changed):
                    with self.assertRaises(FACE_MANIFEST.FaceManifestError):
                        FACE_MANIFEST.verify_manifest(
                            manifest_path=manifest,
                            signature_path=signature,
                            policy_path=policy,
                            signer_directory=signers,
                            **changed,
                        )

            signature.write_text("unsigned\n", encoding="utf-8")
            with self.assertRaises(FACE_MANIFEST.FaceManifestError):
                FACE_MANIFEST.verify_manifest(
                    manifest_path=manifest,
                    signature_path=signature,
                    policy_path=policy,
                    signer_directory=signers,
                    **common,
                )


class ImmutableGitHubReleaseTest(unittest.TestCase):
    class FakeApi:
        repository = "plaid-ai/NUV-AGENT"
        token = "test-token"

        def __init__(self, release: dict[str, object]) -> None:
            self.value = release

        def release(self, tag: str):
            return self.value

        def tag_reference(self, tag: str):
            return {
                "ref": f"refs/tags/{tag}",
                "object": {"type": "tag", "sha": "b" * 40},
            }

        def annotated_tag(self, object_sha: str):
            return {
                "sha": object_sha,
                "tag": "v0.1.121",
                "object": {"type": "commit", "sha": "a" * 40},
            }

        def request(self, method: str, path: str, payload=None):
            if path.endswith("/immutable-releases"):
                return {"enabled": True, "enforced_by_owner": False}
            if method == "PATCH":
                self.value["draft"] = False
                self.value["immutable"] = True
                return self.value
            raise AssertionError((method, path, payload))

    @staticmethod
    def _release(*, draft: bool, immutable: bool) -> dict[str, object]:
        return {
            "id": 123,
            "tag_name": "v0.1.121",
            "draft": draft,
            "immutable": immutable,
            "assets": [],
        }

    def test_finalize_uploads_all_assets_before_immutable_publication(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            paths = []
            for name in (
                "nuv_agent-0.1.121.tar.gz",
                "release-bom.json",
                "nuv-agent-0.1.121.rb",
                "source-plan.json",
            ):
                path = root / name
                path.write_bytes(name.encode("ascii"))
                paths.append(path)
            api = self.FakeApi(self._release(draft=True, immutable=False))
            api.value["assets"].append(
                {
                    "name": paths[0].name,
                    "size": paths[0].stat().st_size,
                    "digest": f"sha256:{hashlib.sha256(paths[0].read_bytes()).hexdigest()}",
                    "state": "uploaded",
                }
            )

            def upload(fake_api, *, tag, local):
                fake_api.value["assets"].append(
                    {
                        "name": local["name"],
                        "size": local["size"],
                        "digest": local["digest"],
                        "state": "uploaded",
                    }
                )

            with mock.patch.object(GITHUB_RELEASE, "_upload_asset", side_effect=upload):
                result = GITHUB_RELEASE.publish_release(
                    api=api,
                    tag="v0.1.121",
                    tag_object_sha="b" * 40,
                    component_sha="a" * 40,
                    phase="finalize",
                    asset_paths=paths,
                )
            self.assertFalse(result["draft"])
            self.assertTrue(result["immutable"])
            self.assertEqual(len(result["assets"]), 4)
            rerun = GITHUB_RELEASE.publish_release(
                api=api,
                tag="v0.1.121",
                tag_object_sha="b" * 40,
                component_sha="a" * 40,
                phase="finalize",
                asset_paths=paths,
            )
            self.assertTrue(rerun["immutable"])

    def test_mutable_published_release_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            asset = Path(raw_root) / "asset.tar.gz"
            asset.write_bytes(b"asset")
            api = self.FakeApi(self._release(draft=False, immutable=False))
            with self.assertRaises(GITHUB_RELEASE.GitHubReleaseError):
                GITHUB_RELEASE.publish_release(
                    api=api,
                    tag="v0.1.121",
                    tag_object_sha="b" * 40,
                    component_sha="a" * 40,
                    phase="stage",
                    asset_paths=[asset],
                )

    def test_live_tag_must_still_resolve_to_preflight_object_and_commit(self) -> None:
        api = self.FakeApi(self._release(draft=True, immutable=False))
        for label, reference, annotated in (
            (
                "retargeted-reference",
                {
                    "ref": "refs/tags/v0.1.121",
                    "object": {"type": "tag", "sha": "c" * 40},
                },
                api.annotated_tag("b" * 40),
            ),
            (
                "retargeted-commit",
                api.tag_reference("v0.1.121"),
                {
                    "sha": "b" * 40,
                    "tag": "v0.1.121",
                    "object": {"type": "commit", "sha": "c" * 40},
                },
            ),
        ):
            with (
                self.subTest(label=label),
                mock.patch.object(api, "tag_reference", return_value=reference),
                mock.patch.object(api, "annotated_tag", return_value=annotated),
                self.assertRaises(GITHUB_RELEASE.GitHubReleaseError),
            ):
                GITHUB_RELEASE.verify_live_tag(
                    api,
                    tag="v0.1.121",
                    tag_object_sha="b" * 40,
                    component_sha="a" * 40,
                )


class HomebrewPromotionTest(unittest.TestCase):
    def test_formula_updater_treats_identity_values_as_data(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            formula = root / "nuv-agent.rb"
            formula.write_text(
                'class NuvAgent < Formula\n  url "__URL__"\n  sha256 "__SHA256__"\n  version "0.1.120"\nend\n',
                encoding="utf-8",
            )
            environment = {
                **os.environ,
                "FORMULA_PATH": str(formula),
                "URL": "https://github.com/plaid-ai/NUV-agent/release.tar.gz",
                "SHA256": "a" * 64,
                "VERSION": "0.1.121",
            }
            subprocess.run(
                [str(ROOT / "packaging/release/update-homebrew-formula.sh")],
                check=True,
                capture_output=True,
                env=environment,
            )
            self.assertIn('version "0.1.121"', formula.read_text(encoding="utf-8"))
            environment["URL"] = 'https://example.invalid/"; system("touch pwn")'
            failed = subprocess.run(
                [str(ROOT / "packaging/release/update-homebrew-formula.sh")],
                check=False,
                capture_output=True,
                env=environment,
                cwd=root,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertFalse((root / "pwn").exists())

    def test_update_exact_rerun_drift_and_downgrade_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            current = root / "current.rb"
            candidate = root / "candidate.rb"
            current.write_text('class Nuv < Formula\n  version "0.1.120"\nend\n', encoding="utf-8")
            candidate.write_text('class Nuv < Formula\n  version "0.1.121"\nend\n', encoding="utf-8")
            result = HOMEBREW_PROMOTION.verify_promotion(
                current, candidate, requested_version="0.1.121"
            )
            self.assertEqual(result["status"], "UPDATE")
            current.write_bytes(candidate.read_bytes())
            result = HOMEBREW_PROMOTION.verify_promotion(
                current, candidate, requested_version="0.1.121"
            )
            self.assertEqual(result["status"], "NOOP")
            candidate.write_text(
                'class Nuv < Formula\n  version "0.1.121"\n  # drift\nend\n',
                encoding="utf-8",
            )
            with self.assertRaises(HOMEBREW_PROMOTION.HomebrewPromotionError):
                HOMEBREW_PROMOTION.verify_promotion(
                    current, candidate, requested_version="0.1.121"
                )
            candidate.write_text(
                'class Nuv < Formula\n  version "0.1.119"\nend\n',
                encoding="utf-8",
            )
            with self.assertRaises(HOMEBREW_PROMOTION.HomebrewPromotionError):
                HOMEBREW_PROMOTION.verify_promotion(
                    current, candidate, requested_version="0.1.119"
                )


if __name__ == "__main__":
    unittest.main()
