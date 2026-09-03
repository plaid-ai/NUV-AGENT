from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "packaging/dev/run-iq9075-config-stream-e2e.py"
SPEC = importlib.util.spec_from_file_location("iq9075_config_stream_e2e", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


RUN_ID = "12345678-1234-4123-8123-123456789abc"
DEVICE_ID = "sp-7-nuvion-iq9075"
EXPIRED_COMMAND_ID = "00000000-0000-4000-8000-000000000003"
ROLLBACK_COMMAND_ID = "00000000-0000-4000-8000-000000000004"
COMMIT_COMMAND_ID = "00000000-0000-4000-8000-000000000005"


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


class _Board:
    def __init__(self) -> None:
        self.commands: dict[str, dict] = {}
        self.quality = "GOOD"
        self.restore_calls = 0
        self.prepared = False
        self.settings = self.baseline()

    @staticmethod
    def baseline() -> dict:
        return {
            "model": {
                "pointer": "anomalyclip/prod",
                "configuredDigest": None,
                "artifactDigest": None,
                "artifactVerified": False,
                "runtimeEnabled": False,
                "runtimeBackend": "none",
            },
            "labels": {
                "inspection": ["normal", "defect"],
                "anomaly": ["defect"],
            },
            "clip": {"enabled": True, "preSeconds": 5, "postSeconds": 7},
            "video": {
                "width": 640,
                "height": 480,
                "fps": 30,
                "bitrateKbps": 1000,
            },
        }

    def prepare(self, *, run_id: str, manifest_sha256: str) -> dict:
        self.prepared = True
        return {
            "schemaVersion": 1,
            "runId": run_id,
            "prepared": True,
            "syntheticSource": "videotestsrc",
            "connectivityShim": "scoped-iw-ping",
            "baseline": self.baseline(),
            "configBeforeSha256": "b" * 64,
            "configTestSha256": "c" * 64,
            "runtimeIdentity": _runtime_identity(service_pid=101),
            "exclusiveLease": True,
            "deadmanArmed": True,
            "queue": self._queue(),
        }

    def register(self, command_id: str, sequence: int, command_type: str, payload: dict) -> None:
        if command_type == "CONFIG_APPLY":
            self.assert_config_scope(payload)
            self.settings = json.loads(json.dumps(self.settings))
            self.settings["clip"] = dict(payload["clip"])
            self.settings["video"] = dict(payload["video"])
            reported = {
                **payload,
                "configSchema": "12",
                "settingsDigest": MODULE.settings_digest(payload),
                "health": "FUNCTIONAL_HEALTHY",
            }
        elif payload["mode"] == "DISABLED":
            reported = {
                **payload,
                "encoder": "x264enc",
                "requestedBitrateKbps": 1200,
                "appliedBitrateKbps": 700,
                "lastAdjustmentReason": "policy_disabled",
                "health": "STREAM_CONTINUOUS",
            }
        else:
            reported = {
                **payload,
                "encoder": "x264enc",
                "requestedBitrateKbps": payload["initialBitrateKbps"],
                "appliedBitrateKbps": payload["initialBitrateKbps"],
                "lastAdjustmentReason": "policy_activated",
                "health": "STREAM_CONTINUOUS",
            }
        self.commands[command_id] = {
            "commandId": command_id,
            "sequence": sequence,
            "type": command_type,
            "payload": payload,
            "reported": reported,
            "revision": 1,
        }

    @staticmethod
    def assert_config_scope(payload: dict) -> None:
        if "model" in payload or "labels" in payload:
            raise AssertionError("disabled IQ backend must not activate model/labels")

    @staticmethod
    def _queue() -> dict:
        return {
            "inboxPendingRows": 0,
            "observationPendingRows": 0,
            "observationReservedRows": 0,
            "observationDlqRows": 0,
        }

    def set_link(self, *, run_id: str, quality: str) -> dict:
        self.quality = quality
        for command in self.commands.values():
            payload = command["payload"]
            if command["type"] != "STREAM_POLICY" or payload.get("mode") != "ADAPTIVE":
                continue
            if quality == "POOR":
                command["reported"]["appliedBitrateKbps"] = int(
                    payload["initialBitrateKbps"] * payload["decreaseFactor"]
                )
                command["reported"]["lastAdjustmentReason"] = (
                    "connectivity_poor,packet_loss_high,round_trip_time_high"
                )
                command["revision"] += 1
            elif command["reported"]["appliedBitrateKbps"] < payload["initialBitrateKbps"]:
                command["reported"]["appliedBitrateKbps"] += payload[
                    "increaseStepKbps"
                ]
                command["reported"]["lastAdjustmentReason"] = "healthy_recovery"
                command["revision"] += 1
        return {"schemaVersion": 1, "runId": run_id, "quality": quality, "changed": True}

    def inspect(self, *, run_id: str, command_id: str) -> dict:
        command = self.commands.get(command_id)
        if command is None:
            return {
                "schemaVersion": 1,
                "runId": run_id,
                "command": None,
                "observation": None,
                "queue": self._queue(),
                "serviceActive": True,
            }
        return {
            "schemaVersion": 1,
            "runId": run_id,
            "command": {
                "commandId": command_id,
                "sequence": command["sequence"],
                "type": command["type"],
                "status": "SUCCEEDED",
                "ackStatuses": ["RECEIVED", "IN_PROGRESS", "SUCCEEDED"],
                "reportedState": dict(command["reported"]),
            },
            "observation": {
                "revision": command["revision"],
                "reportedState": dict(command["reported"]),
                "acked": True,
            },
            "queue": self._queue(),
            "settings": json.loads(json.dumps(self.settings)),
            "settingsSha256": hashlib.sha256(
                MODULE.canonical_json(self.settings)
            ).hexdigest(),
            "serviceActive": True,
        }

    def restore(self, *, run_id: str) -> dict:
        self.restore_calls += 1
        self.settings = self.baseline()
        return {
            "schemaVersion": 1,
            "runId": run_id,
            "restored": True,
            "idempotent": False,
            "exactRestoration": True,
            "runtimeRestarted": True,
            "configSha256": "b" * 64,
            "settings": self.baseline(),
            "settingsSha256": hashlib.sha256(
                MODULE.canonical_json(self.baseline())
            ).hexdigest(),
            "encoderStartupBitrateKbps": 1000,
            "runtimeIdentity": _runtime_identity(service_pid=202),
            "exclusiveLeaseReleased": True,
            "deadmanDisarmed": True,
        }


class _Api:
    def __init__(
        self,
        board: _Board,
        fail_on_issue: int | None = None,
        *,
        single_twin: bool = True,
        include_expired_predecessor: bool = True,
    ) -> None:
        self.board = board
        self.next_sequence = 6
        self.issued: list[MODULE.IssuedCommand] = []
        self.fail_on_issue = fail_on_issue
        self.single_twin = single_twin
        self.include_expired_predecessor = include_expired_predecessor

    def issue(
        self,
        *,
        space_id: int,
        device_id: str,
        command_type: str,
        payload: dict,
        desired_state: dict,
    ) -> MODULE.IssuedCommand:
        if self.fail_on_issue == len(self.issued) + 1:
            raise MODULE.ConfigStreamError("injected issue failure")
        assert desired_state == payload
        command = MODULE.IssuedCommand(
            str(uuid.uuid4()), self.next_sequence, command_type
        )
        self.next_sequence += 1
        self.issued.append(command)
        self.board.register(command.command_id, command.sequence, command_type, payload)
        return command

    def commands(self, *, space_id: int, device_id: str) -> list[dict]:
        commands = [
            {
                "commandId": command.command_id,
                "sequence": command.sequence,
                "type": command.command_type,
                "status": "SUCCEEDED",
            }
            for command in reversed(self.issued)
        ]
        if self.include_expired_predecessor:
            commands.append(
                {
                    "commandId": EXPIRED_COMMAND_ID,
                    "sequence": 3,
                    "type": "STREAM_POLICY",
                    "status": "EXPIRED",
                    "expiresAt": "2026-09-02T00:00:00.000Z",
                }
            )
        commands.append(
            {
                "commandId": ROLLBACK_COMMAND_ID,
                "sequence": 4,
                "type": "AGENT_UPDATE",
                "status": "ROLLED_BACK",
                "expiresAt": "2026-09-03T00:01:00.000Z",
            }
        )
        commands.append(
            {
                "commandId": COMMIT_COMMAND_ID,
                "sequence": 5,
                "type": "AGENT_UPDATE",
                "status": "SUCCEEDED",
                "expiresAt": "2026-09-03T00:02:00.000Z",
            }
        )
        return commands

    def projection(self, *, space_id: int, device_id: str) -> dict:
        domains: dict[str, dict] = {}
        for command in self.issued:
            record = self.board.commands[command.command_id]
            domain = "settings" if command.command_type == "CONFIG_APPLY" else "streaming"
            domains[domain] = {
                "convergenceStatus": "CONVERGED",
                "desiredSequence": command.sequence,
                "desiredCommandId": command.command_id,
                "desiredState": dict(record["payload"]),
                "reportedSequence": command.sequence,
                "reportedCommandId": command.command_id,
                "reportedRevision": record["revision"],
                "reportedState": dict(record["reported"]),
            }
        unknown = {
            "convergenceStatus": "UNKNOWN",
            "desiredSequence": None,
            "desiredCommandId": None,
            "desiredState": None,
            "reportedSequence": None,
            "reportedCommandId": None,
            "reportedRevision": None,
            "reportedState": None,
        }
        if self.single_twin:
            if not self.issued:
                return {"twin": unknown}
            command = self.issued[-1]
            domain = "settings" if command.command_type == "CONFIG_APPLY" else "streaming"
            return {"twin": domains[domain]}
        return {
            "twins": {
                "settings": domains.get("settings", unknown),
                "streaming": domains.get("streaming", unknown),
                "agent": unknown,
            }
        }


def _manifest() -> dict:
    return {
        "schemaVersion": 1,
        "protocolVersion": "iq9075-fleet-e2e-v2",
        "runId": RUN_ID,
        "toolSha256": "1" * 64,
        "inputs": {},
        "destinations": {},
        "identity": {
            "deviceId": DEVICE_ID,
            "spaceId": 7,
            "productModel": "IQ9075_DEV",
            "platformProfile": "iq9075_dev",
            "hardwareRevision": "QCS9075-EVK",
            "architecture": "aarch64",
            "dockerRequired": False,
        },
        "scenario": {
            "type": "commit",
            "expectedCommandId": COMMIT_COMMAND_ID,
            "expectedBomDigest": "sha256:" + "d" * 64,
            "release": {
                "agentVersion": "0.1.121",
                "releaseSequence": 121,
                "artifactDigest": "sha256:" + "a" * 64,
                "componentSha": "e" * 40,
                "configSchema": "12",
                "publisherKeyId": "release-test",
            },
        },
    }


def _runtime_identity(*, service_pid: int) -> dict:
    scenario = _manifest()["scenario"]
    relative_slot = "releases/" + scenario["expectedBomDigest"].removeprefix("sha256:")
    return {
        "activeSlot": relative_slot,
        "processActiveSlot": relative_slot,
        "processExpectedBomDigest": scenario["expectedBomDigest"],
        "servicePid": service_pid,
        "releaseMarkerSha256": "f" * 64,
        "buildInfoSha256": "9" * 64,
        "release": {
            "schemaVersion": 2,
            "bomDigest": scenario["expectedBomDigest"],
            **scenario["release"],
        },
    }


class ConfigStreamOrchestratorTest(unittest.TestCase):
    def test_complete_flow_uses_fresh_sequences_and_restores_exactly(self) -> None:
        clock = _Clock()
        board = _Board()
        api = _Api(board)
        orchestrator = MODULE.ConfigStreamOrchestrator(
            api=api,
            board=board,
            monotonic=clock.monotonic,
            sleeper=clock.sleep,
            wall_clock=lambda: datetime(2026, 9, 3, tzinfo=timezone.utc),
        )

        evidence = orchestrator.run(
            run_id=RUN_ID,
            manifest=_manifest(),
            manifest_sha256="1" * 64,
            ota_evidence_sha256="2" * 64,
            wait_seconds=120,
        )

        self.assertTrue(all(evidence["gates"].values()))
        self.assertEqual(board.restore_calls, 1)
        self.assertEqual([item.sequence for item in api.issued], [6, 7, 8, 9])
        self.assertNotIn(3, [item.sequence for item in api.issued])
        self.assertEqual(
            evidence["releaseCommand"],
            {
                "commandId": COMMIT_COMMAND_ID,
                "sequence": 5,
                "type": "AGENT_UPDATE",
                "status": "SUCCEEDED",
            },
        )
        self.assertEqual(
            evidence["priorRollbackCommand"],
            {
                "commandId": ROLLBACK_COMMAND_ID,
                "sequence": 4,
                "type": "AGENT_UPDATE",
                "status": "ROLLED_BACK",
            },
        )
        self.assertEqual(
            evidence["expiredPredecessors"],
            [
                {
                    "commandId": EXPIRED_COMMAND_ID,
                    "sequence": 3,
                    "type": "STREAM_POLICY",
                    "status": "EXPIRED",
                    "expiresAt": "2026-09-02T00:00:00.000Z",
                }
            ],
        )
        self.assertEqual(evidence["projectionShape"], "single")
        self.assertEqual(
            evidence["source"]["runtimeIdentity"]["release"],
            _runtime_identity(service_pid=101)["release"],
        )
        for issued in api.issued[:2]:
            payload = board.commands[issued.command_id]["payload"]
            self.assertNotIn("model", payload)
            self.assertNotIn("labels", payload)
            self.assertEqual(payload["clip"], _Board.baseline()["clip"])
        self.assertEqual(
            evidence["config"]["fieldCoverage"],
            {
                "model": "PRESERVED_WITHOUT_ACTIVATION",
                "labels": "PRESERVED_WITHOUT_ACTIVATION",
                "clipPolicy": "SAME_VALUE_RECONCILED",
                "video": "CHANGED_AND_RESTORED",
            },
        )
        self.assertIn(
            "modelConfigurationPreservedWithoutActivation", evidence["gates"]
        )
        self.assertIn(
            "labelConfigurationPreservedWithoutActivation", evidence["gates"]
        )
        self.assertNotIn("modelConfigurationPreserved", evidence["gates"])
        self.assertNotIn("labelConfigurationPreserved", evidence["gates"])
        self.assertEqual(
            evidence["config"]["apply"]["lifecycleAckStatuses"],
            ["RECEIVED", "IN_PROGRESS", "SUCCEEDED"],
        )
        self.assertLess(
            evidence["stream"]["poor"]["appliedBitrateKbps"],
            evidence["stream"]["initialGood"]["appliedBitrateKbps"],
        )
        self.assertGreater(
            evidence["stream"]["recoveredGood"]["appliedBitrateKbps"],
            evidence["stream"]["poor"]["appliedBitrateKbps"],
        )
        self.assertGreater(
            evidence["stream"]["recoveredGood"]["policyRevision"],
            evidence["stream"]["poor"]["policyRevision"],
        )
        MODULE.FLEET.assert_no_secret_material(evidence)
        with tempfile.TemporaryDirectory() as raw_directory:
            evidence_path = Path(raw_directory) / "config-stream-evidence.json"
            MODULE.FLEET.atomic_json(evidence_path, evidence, immutable=True)
            persisted = MODULE.FLEET.read_regular(
                evidence_path, MODULE.FLEET.MAX_OUTPUT_BYTES
            )
            self.assertEqual(persisted, MODULE.canonical_json(evidence))

    def test_failure_after_prepare_still_restores_board(self) -> None:
        clock = _Clock()
        board = _Board()
        api = _Api(board, fail_on_issue=2)
        orchestrator = MODULE.ConfigStreamOrchestrator(
            api=api,
            board=board,
            monotonic=clock.monotonic,
            sleeper=clock.sleep,
            wall_clock=lambda: datetime(2026, 9, 3, tzinfo=timezone.utc),
        )

        with self.assertRaises(MODULE.ConfigStreamError):
            orchestrator.run(
                run_id=RUN_ID,
                manifest=_manifest(),
                manifest_sha256="1" * 64,
                ota_evidence_sha256="2" * 64,
                wait_seconds=120,
            )

        self.assertEqual(board.restore_calls, 1)

    def test_prepare_response_loss_still_attempts_exact_restore(self) -> None:
        class ResponseLostBoard(_Board):
            def prepare(self, *, run_id: str, manifest_sha256: str) -> dict:
                super().prepare(run_id=run_id, manifest_sha256=manifest_sha256)
                raise OSError("injected response loss")

        clock = _Clock()
        board = ResponseLostBoard()
        orchestrator = MODULE.ConfigStreamOrchestrator(
            api=_Api(board),
            board=board,
            monotonic=clock.monotonic,
            sleeper=clock.sleep,
            wall_clock=lambda: datetime(2026, 9, 3, tzinfo=timezone.utc),
        )

        with self.assertRaises(MODULE.ConfigStreamError):
            orchestrator.run(
                run_id=RUN_ID,
                manifest=_manifest(),
                manifest_sha256="1" * 64,
                ota_evidence_sha256="2" * 64,
                wait_seconds=120,
            )

        self.assertTrue(board.prepared)
        self.assertEqual(board.restore_calls, 1)

    def test_restore_response_loss_replays_idempotent_cleanup_evidence(self) -> None:
        class RestoreResponseLostBoard(_Board):
            def __init__(self) -> None:
                super().__init__()
                self.cached_restore: dict | None = None

            def restore(self, *, run_id: str) -> dict:
                if self.cached_restore is None:
                    self.cached_restore = super().restore(run_id=run_id)
                    raise OSError("injected restore response loss")
                self.restore_calls += 1
                return {**self.cached_restore, "idempotent": True}

        clock = _Clock()
        board = RestoreResponseLostBoard()
        orchestrator = MODULE.ConfigStreamOrchestrator(
            api=_Api(board),
            board=board,
            monotonic=clock.monotonic,
            sleeper=clock.sleep,
            wall_clock=lambda: datetime(2026, 9, 3, tzinfo=timezone.utc),
        )

        evidence = orchestrator.run(
            run_id=RUN_ID,
            manifest=_manifest(),
            manifest_sha256="1" * 64,
            ota_evidence_sha256="2" * 64,
            wait_seconds=120,
        )

        self.assertEqual(board.restore_calls, 2)
        self.assertTrue(evidence["cleanup"]["idempotent"])
        self.assertTrue(evidence["gates"]["exactBoardRestoration"])

    def test_non_commit_manifest_is_rejected_before_board_mutation(self) -> None:
        manifest = _manifest()
        manifest["scenario"]["type"] = "oak-fault-rollback"
        clock = _Clock()
        board = _Board()
        orchestrator = MODULE.ConfigStreamOrchestrator(
            api=_Api(board),
            board=board,
            monotonic=clock.monotonic,
            sleeper=clock.sleep,
            wall_clock=lambda: datetime(2026, 9, 3, tzinfo=timezone.utc),
        )

        with self.assertRaisesRegex(MODULE.ConfigStreamError, "commit manifest"):
            orchestrator.run(
                run_id=RUN_ID,
                manifest=manifest,
                manifest_sha256="1" * 64,
                ota_evidence_sha256="2" * 64,
                wait_seconds=120,
            )

        self.assertFalse(board.prepared)
        self.assertEqual(board.restore_calls, 0)

    def test_missing_exact_expired_predecessor_fails_and_restores(self) -> None:
        clock = _Clock()
        board = _Board()
        orchestrator = MODULE.ConfigStreamOrchestrator(
            api=_Api(board, include_expired_predecessor=False),
            board=board,
            monotonic=clock.monotonic,
            sleeper=clock.sleep,
            wall_clock=lambda: datetime(2026, 9, 3, tzinfo=timezone.utc),
        )

        with self.assertRaisesRegex(MODULE.ConfigStreamError, "expired predecessor"):
            orchestrator.run(
                run_id=RUN_ID,
                manifest=_manifest(),
                manifest_sha256="1" * 64,
                ota_evidence_sha256="2" * 64,
                wait_seconds=120,
            )

        self.assertEqual(board.restore_calls, 1)

    def test_nonterminal_precommit_command_fails_and_restores(self) -> None:
        class NonterminalRollbackApi(_Api):
            def commands(self, *, space_id: int, device_id: str) -> list[dict]:
                commands = super().commands(space_id=space_id, device_id=device_id)
                for command in commands:
                    if command["commandId"] == ROLLBACK_COMMAND_ID:
                        command["status"] = "IN_PROGRESS"
                return commands

        clock = _Clock()
        board = _Board()
        orchestrator = MODULE.ConfigStreamOrchestrator(
            api=NonterminalRollbackApi(board),
            board=board,
            monotonic=clock.monotonic,
            sleeper=clock.sleep,
            wall_clock=lambda: datetime(2026, 9, 3, tzinfo=timezone.utc),
        )

        with self.assertRaisesRegex(MODULE.ConfigStreamError, "prior rollback"):
            orchestrator.run(
                run_id=RUN_ID,
                manifest=_manifest(),
                manifest_sha256="1" * 64,
                ota_evidence_sha256="2" * 64,
                wait_seconds=120,
            )

        self.assertEqual(board.restore_calls, 1)

    def test_future_expired_deadline_fails_and_restores(self) -> None:
        class FutureExpiredApi(_Api):
            def commands(self, *, space_id: int, device_id: str) -> list[dict]:
                commands = super().commands(space_id=space_id, device_id=device_id)
                for command in commands:
                    if command["commandId"] == EXPIRED_COMMAND_ID:
                        command["expiresAt"] = "2026-09-04T00:00:00.000Z"
                return commands

        clock = _Clock()
        board = _Board()
        orchestrator = MODULE.ConfigStreamOrchestrator(
            api=FutureExpiredApi(board),
            board=board,
            monotonic=clock.monotonic,
            sleeper=clock.sleep,
            wall_clock=lambda: datetime(2026, 9, 3, tzinfo=timezone.utc),
        )

        with self.assertRaisesRegex(MODULE.ConfigStreamError, "future"):
            orchestrator.run(
                run_id=RUN_ID,
                manifest=_manifest(),
                manifest_sha256="1" * 64,
                ota_evidence_sha256="2" * 64,
                wait_seconds=120,
            )

        self.assertEqual(board.restore_calls, 1)

    def test_domained_projection_remains_supported_when_single_twin_is_absent(self) -> None:
        clock = _Clock()
        board = _Board()
        orchestrator = MODULE.ConfigStreamOrchestrator(
            api=_Api(board, single_twin=False),
            board=board,
            monotonic=clock.monotonic,
            sleeper=clock.sleep,
            wall_clock=lambda: datetime(2026, 9, 3, tzinfo=timezone.utc),
        )

        evidence = orchestrator.run(
            run_id=RUN_ID,
            manifest=_manifest(),
            manifest_sha256="1" * 64,
            ota_evidence_sha256="2" * 64,
            wait_seconds=120,
        )

        self.assertEqual(evidence["projectionShape"], "domained")
        self.assertEqual(board.restore_calls, 1)

    def test_authoritative_single_twin_wins_during_schema_rollout(self) -> None:
        single = {"desiredCommandId": "single"}
        domain = {"desiredCommandId": "domain"}
        projection = {"twin": single, "twins": {"settings": domain}}

        self.assertEqual(MODULE.twin_domain(projection, "settings"), single)
        self.assertEqual(MODULE.projection_shape(projection, "settings"), "single")

    def test_release_identity_mismatch_fails_closed_and_restores(self) -> None:
        class WrongReleaseBoard(_Board):
            def prepare(self, *, run_id: str, manifest_sha256: str) -> dict:
                result = super().prepare(
                    run_id=run_id, manifest_sha256=manifest_sha256
                )
                result["runtimeIdentity"]["release"]["componentSha"] = "0" * 40
                return result

        clock = _Clock()
        board = WrongReleaseBoard()
        orchestrator = MODULE.ConfigStreamOrchestrator(
            api=_Api(board),
            board=board,
            monotonic=clock.monotonic,
            sleeper=clock.sleep,
            wall_clock=lambda: datetime(2026, 9, 3, tzinfo=timezone.utc),
        )

        with self.assertRaisesRegex(MODULE.ConfigStreamError, "runtime release identity"):
            orchestrator.run(
                run_id=RUN_ID,
                manifest=_manifest(),
                manifest_sha256="1" * 64,
                ota_evidence_sha256="2" * 64,
                wait_seconds=120,
            )

        self.assertEqual(board.restore_calls, 1)

    def test_unhealthy_or_non_x264_stream_never_passes_adaptation_gate(self) -> None:
        class UnhealthyBoard(_Board):
            def inspect(self, *, run_id: str, command_id: str) -> dict:
                result = super().inspect(run_id=run_id, command_id=command_id)
                command = result.get("command")
                observation = result.get("observation")
                if isinstance(command, dict) and command.get("type") == "STREAM_POLICY":
                    command["reportedState"]["encoder"] = "openh264enc"
                    command["reportedState"]["health"] = "DEGRADED"
                    if isinstance(observation, dict):
                        observation["reportedState"] = dict(command["reportedState"])
                return result

        clock = _Clock()
        board = UnhealthyBoard()
        orchestrator = MODULE.ConfigStreamOrchestrator(
            api=_Api(board),
            board=board,
            monotonic=clock.monotonic,
            sleeper=clock.sleep,
            wall_clock=lambda: datetime(2026, 9, 3, tzinfo=timezone.utc),
        )

        with self.assertRaises(MODULE.ConfigStreamError):
            orchestrator.run(
                run_id=RUN_ID,
                manifest=_manifest(),
                manifest_sha256="1" * 64,
                ota_evidence_sha256="2" * 64,
                wait_seconds=30,
            )

        self.assertEqual(board.restore_calls, 1)

    def test_invalid_prepared_baseline_does_not_mask_cleanup(self) -> None:
        class InvalidBaselineBoard(_Board):
            def prepare(self, *, run_id: str, manifest_sha256: str) -> dict:
                result = super().prepare(
                    run_id=run_id, manifest_sha256=manifest_sha256
                )
                result["baseline"] = {"invalid": True}
                return result

        clock = _Clock()
        board = InvalidBaselineBoard()
        orchestrator = MODULE.ConfigStreamOrchestrator(
            api=_Api(board),
            board=board,
            monotonic=clock.monotonic,
            sleeper=clock.sleep,
            wall_clock=lambda: datetime(2026, 9, 3, tzinfo=timezone.utc),
        )

        with self.assertRaisesRegex(
            MODULE.ConfigStreamError, "board settings baseline fields"
        ):
            orchestrator.run(
                run_id=RUN_ID,
                manifest=_manifest(),
                manifest_sha256="1" * 64,
                ota_evidence_sha256="2" * 64,
                wait_seconds=120,
            )

        self.assertEqual(board.restore_calls, 1)

    def test_remote_program_uses_scoped_shims_without_link_mutation_or_camera(self) -> None:
        program = MODULE.BOARD_PROGRAM
        compile(program, "<iq9075-config-stream-board>", "exec")
        self.assertIn("/run/nuvion-config-stream-e2e", program)
        self.assertIn('runtime / "bin/iw"', program)
        self.assertIn('runtime / "bin/ping"', program)
        self.assertIn("videotestsrc is-live=true", program)
        self.assertNotIn("ip link", program)
        self.assertNotIn("/dev/video", program)
        self.assertNotIn("USB", program)
        self.assertIn("nuvion-config-stream-deadman-", program)
        self.assertIn("/run/lock/nuvion-fleet-e2e.lock", program)
        self.assertIn("/var/lib/nuvion-fleet-e2e/active-run.json", program)
        self.assertNotIn("os.chmod(runtime.parent", program)
        restore_program = program[
            program.index("def restore(") : program.index("\ndef main()")
        ]
        prepare_program = program[
            program.index("def prepare(") : program.index("\ndef shlex_quote")
        ]
        self.assertLess(
            prepare_program.index("arm_deadman(rid)"),
            prepare_program.index("claim_config_lease(rid)"),
        )
        self.assertLess(
            prepare_program.index("arm_deadman(rid)"),
            prepare_program.index('systemctl("stop", "nuv-agent.service")'),
        )
        self.assertLess(
            prepare_program.index('"dropinSha256": sha(dropin_payload)'),
            prepare_program.index("atomic(dropin, dropin_payload"),
        )
        restored_index = restore_program.index(
            'state.update({"phase": "RESTORED"'
        )
        self.assertLess(
            restored_index,
            restore_program.index("purge_snapshots(work)", restored_index),
        )

    def test_restored_targets_verify_after_snapshot_payloads_are_purged(self) -> None:
        definitions = MODULE.BOARD_PROGRAM.split("\ntry:\n    main()", 1)[0]
        namespace: dict = {"__name__": "iq9075_config_stream_board_test"}
        exec(  # noqa: S102 - execute repository-owned embedded board program only.
            compile(definitions, "<iq9075-config-stream-board-definitions>", "exec"),
            namespace,
        )
        with tempfile.TemporaryDirectory() as raw_directory:
            target = Path(raw_directory) / "commands.sqlite3"
            target.write_bytes(b"restored-database")
            metadata = target.lstat()
            record = {
                "path": str(target),
                "exists": True,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "mode": metadata.st_mode & 0o7777,
                "uid": metadata.st_uid,
                "gid": metadata.st_gid,
                "snapshot": "0",
            }
            namespace["FIXED"] = (target,)

            self.assertEqual(namespace["validate_snapshot_records"]([record]), [record])
            self.assertTrue(namespace["verify_restored"]([record]))
            self.assertFalse((Path(raw_directory) / "before").exists())

    def test_queue_gate_requires_inbox_outbox_and_reservations_to_be_zero(self) -> None:
        for field in (
            "inboxPendingRows",
            "observationPendingRows",
            "observationReservedRows",
            "observationDlqRows",
        ):
            value = _Board._queue()
            value[field] = 1
            with self.assertRaises(MODULE.ConfigStreamError):
                MODULE.validate_queue_drained(value)


if __name__ == "__main__":
    unittest.main()
