from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
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
            "queue": self._queue(),
        }

    def register(self, command_id: str, sequence: int, command_type: str, payload: dict) -> None:
        if command_type == "CONFIG_APPLY":
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
        }


class _Api:
    def __init__(self, board: _Board, fail_on_issue: int | None = None) -> None:
        self.board = board
        self.next_sequence = 4
        self.issued: list[MODULE.IssuedCommand] = []
        self.fail_on_issue = fail_on_issue

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
        return [
            {
                "commandId": command.command_id,
                "sequence": command.sequence,
                "type": command.command_type,
                "status": "SUCCEEDED",
            }
            for command in reversed(self.issued)
        ]

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
            "expectedBomDigest": "sha256:" + "d" * 64,
            "release": {
                "agentVersion": "0.1.121",
                "componentSha": "e" * 40,
                "configSchema": "12",
            },
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
        self.assertEqual([item.sequence for item in api.issued], [4, 5, 6, 7])
        self.assertNotIn(3, [item.sequence for item in api.issued])
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
        self.assertEqual(
            hashlib.sha256(MODULE.canonical_json(evidence)).hexdigest(),
            hashlib.sha256(MODULE.canonical_json(evidence)).hexdigest(),
        )

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
        self.assertIn("/run/nuvion-config-stream-e2e", program)
        self.assertIn('runtime / "bin/iw"', program)
        self.assertIn('runtime / "bin/ping"', program)
        self.assertIn("videotestsrc is-live=true", program)
        self.assertNotIn("ip link", program)
        self.assertNotIn("/dev/video", program)
        self.assertNotIn("USB", program)

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
