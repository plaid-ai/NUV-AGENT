from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


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


def _board_namespace():
    namespace = {"__name__": "rtp_board_test"}
    exec(compile(MODULE.BOARD_PROGRAM.split("\ntry:\n    main()", 1)[0], "<rtp-board>", "exec"), namespace)
    return namespace


def _network_condition(phase, run_id=RUN_ID, service_pid=101, stamp="2026-09-02T23:59:"):
    ns = _board_namespace()
    table, unit = ns["packet_names"](run_id)
    value = {
        "kind": "socket-scoped-rtp-drop", "runId": run_id, "phase": phase,
        "table": table, "timerUnit": unit + ".timer", "automaticRemovalSeconds": 60,
        "dropPercent": 35, "servicePid": service_pid, "processStartTicks": 10000,
        "uid": 997, "cgroup": "0::/system.slice/nuv-agent.service", "udpSourcePorts": [31000, 31001],
        "previousTablesSha256": "a" * 64,
        "ruleShapeSha256": ns["sha"](ns["canonical"](ns["packet_shape"](table, 997, [31000, 31001]))),
        "appliedAt": stamp + "10.000Z", "timerArmed": True,
        "counter": {"packets": 100, "bytes": 120000},
    }
    if phase == "ACTIVE":
        value["observedAt"] = stamp + "20.000Z"
    else:
        value.update(releasedAt=stamp + "25.000Z", exactNetworkRestoration=True, timerDisarmed=True, tableAbsent=True)
    return value


class NativeSyntheticSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import gi

            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
        except (ImportError, ValueError):
            if os.environ.get("NUV_REQUIRE_NATIVE_GST") == "1":
                raise
            raise unittest.SkipTest("native GStreamer is unavailable") from None
        cls.gst = Gst
        Gst.init(None)

    def _frames(self, source: str, count: int = 3) -> list[tuple]:
        gst = self.gst
        pipeline = gst.parse_launch(
            source + " ! tee name=t "
            "t. ! queue ! appsink name=frames sync=false "
            "t. ! queue ! videoconvert ! video/x-raw,format=I420 ! fakesink sync=false"
        )
        frames = []
        try:
            pipeline.set_state(gst.State.PLAYING)
            sink = pipeline.get_by_name("frames")
            for _ in range(count):
                sample = sink.emit("try-pull-sample", 5 * gst.SECOND)
                self.assertIsNotNone(sample, "synthetic source did not deliver a frame")
                caps = sample.get_caps().get_structure(0)
                buffer = sample.get_buffer()
                expected = caps.get_value("width") * caps.get_value("height") * 3
                frames.append((caps.get_value("format"), buffer.get_size(), expected, buffer.pts))
        finally:
            pipeline.set_state(gst.State.NULL)
        return frames

    def test_configured_source_supplies_packed_rgb_frames(self) -> None:
        updates = next(
            ast.literal_eval(node.value)
            for node in ast.parse(MODULE.BOARD_PROGRAM).body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "UPDATES" for target in node.targets)
        )
        frames = self._frames(updates["NUVION_GST_SOURCE"])
        for pixel_format, actual_bytes, expected_bytes, _pts in frames:
            self.assertEqual(pixel_format, "RGB")
            self.assertEqual(actual_bytes, expected_bytes)
        timestamps = [frame[3] for frame in frames]
        self.assertEqual(timestamps, sorted(set(timestamps)))

    def test_planar_frames_do_not_meet_frame_reader_contract(self) -> None:
        frames = self._frames("videotestsrc num-buffers=1 ! video/x-raw,format=I420", count=1)
        self.assertEqual(frames[0][0], "I420")
        self.assertNotEqual(frames[0][1], frames[0][2])


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
        self.fault_attempted = False
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
            "connectivityShim": "socket-scoped-rtp-drop",
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
        self.fault_attempted = self.fault_attempted or quality == "POOR"
        for command in self.commands.values():
            payload = command["payload"]
            if command["type"] != "STREAM_POLICY" or payload.get("mode") != "ADAPTIVE":
                continue
            if quality == "POOR":
                command["reported"]["appliedBitrateKbps"] = int(
                    payload["initialBitrateKbps"] * payload["decreaseFactor"]
                )
                command["reported"]["lastAdjustmentReason"] = (
                    "packet_loss_high,round_trip_time_high"
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
            "networkCondition": _network_condition("ACTIVE" if self.quality == "POOR" else "RELEASED", run_id) if self.fault_attempted else None,
        }

    def restore(self, *, run_id: str) -> dict:
        self.restore_calls += 1
        self.settings = self.baseline()
        return {
            "schemaVersion": 1,
            "runId": run_id,
            "completedAt": "2026-09-03T00:00:01.000Z",
            "restored": True,
            "idempotent": False,
            "noMutation": False,
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
            "networkCondition": _network_condition("RELEASED", run_id),
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
                "issuedAt": "2026-09-03T00:00:00.000Z",
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
                    "issuedAt": "2026-09-01T23:59:00.000Z",
                    "expiresAt": "2026-09-02T00:00:00.000Z",
                }
            )
        commands.append(
            {
                "commandId": ROLLBACK_COMMAND_ID,
                "sequence": 4,
                "type": "AGENT_UPDATE",
                "status": "ROLLED_BACK",
                "issuedAt": "2026-09-02T23:58:00.000Z",
                "expiresAt": "2026-09-03T00:01:00.000Z",
            }
        )
        commands.append(
            {
                "commandId": COMMIT_COMMAND_ID,
                "sequence": 5,
                "type": "AGENT_UPDATE",
                "status": "SUCCEEDED",
                "issuedAt": "2026-09-02T23:59:00.000Z",
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
    marker = {
        "schemaVersion": 2,
        "bomDigest": scenario["expectedBomDigest"],
        **scenario["release"],
    }
    build_info = (
        '"""Generated release identity. Do not edit in release artifacts."""\n\n'
        f'AGENT_VERSION = "{scenario["release"]["agentVersion"]}"\n'
        f'COMPONENT_SHA = "{scenario["release"]["componentSha"]}"\n'
    ).encode("utf-8")
    return {
        "activeSlot": relative_slot,
        "processActiveSlot": relative_slot,
        "processExpectedBomDigest": scenario["expectedBomDigest"],
        "servicePid": service_pid,
        "releaseMarkerSha256": hashlib.sha256(
            MODULE.canonical_json(marker)
        ).hexdigest(),
        "buildInfoSha256": hashlib.sha256(build_info).hexdigest(),
        "release": marker,
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
                "issuedAt": "2026-09-02T23:59:00.000Z",
            },
        )
        self.assertEqual(
            evidence["priorRollbackCommand"],
            {
                "commandId": ROLLBACK_COMMAND_ID,
                "sequence": 4,
                "type": "AGENT_UPDATE",
                "status": "ROLLED_BACK",
                "issuedAt": "2026-09-02T23:58:00.000Z",
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
        self.assertEqual(
            evidence["source"]["apiOrigin"],
            "https://api.nuvion-dev.plaidlabs.ai",
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
        for name in ("initialGood", "poor", "recoveredGood"):
            self.assertEqual(
                evidence["stream"][name]["commandId"],
                evidence["stream"]["adaptiveCommand"]["commandId"],
            )
            self.assertEqual(
                evidence["stream"][name]["sequence"],
                evidence["stream"]["adaptiveCommand"]["sequence"],
            )
        MODULE.FLEET.assert_no_secret_material(evidence)
        with tempfile.TemporaryDirectory() as raw_directory:
            evidence_path = Path(raw_directory) / "config-stream-evidence.json"
            MODULE.FLEET.atomic_json(evidence_path, evidence, immutable=True)
            persisted = MODULE.FLEET.read_regular(
                evidence_path, MODULE.FLEET.MAX_OUTPUT_BYTES
            )
            self.assertEqual(persisted, MODULE.canonical_json(evidence))

    def test_delayed_controller_observations_complete_the_entire_flow(self) -> None:
        class DelayedBoard(_Board):
            def register(self, command_id, sequence, command_type, payload):
                super().register(command_id, sequence, command_type, payload)
                if payload.get("mode") == "ADAPTIVE":
                    self.commands[command_id]["reported"]["lastAdjustmentReason"] = "awaiting_hysteresis:stable"
                    self.commands[command_id]["revision"] = 3

            def set_link(self, *, run_id, quality):
                result = super().set_link(run_id=run_id, quality=quality)
                for command in self.commands.values():
                    if command["payload"].get("mode") == "ADAPTIVE":
                        command["reported"]["lastAdjustmentReason"] = (
                            "cooldown:connectivity_poor,packet_loss_high" if quality == "POOR"
                            else "awaiting_hysteresis:stable"
                        )
                        command["revision"] += 2
                return result

        clock = _Clock(); board = DelayedBoard(); api = _Api(board)
        evidence = MODULE.ConfigStreamOrchestrator(
            api=api, board=board, monotonic=clock.monotonic, sleeper=clock.sleep,
            wall_clock=lambda: datetime(2026, 9, 3, tzinfo=timezone.utc),
        ).run(run_id=RUN_ID, manifest=_manifest(), manifest_sha256="1" * 64,
              ota_evidence_sha256="2" * 64, wait_seconds=120)
        self.assertTrue(all(evidence["gates"].values()))
        self.assertEqual(evidence["stream"]["initialGood"]["policyRevision"], 3)
        self.assertLess(evidence["stream"]["poor"]["appliedBitrateKbps"], evidence["stream"]["initialGood"]["appliedBitrateKbps"])
        self.assertGreater(evidence["stream"]["recoveredGood"]["appliedBitrateKbps"], evidence["stream"]["poor"]["appliedBitrateKbps"])
        self.assertEqual(board.restore_calls, 1)

    def test_sampled_reason_gates_match_real_controller_and_reject_stale_health(self) -> None:
        from nuvion_app.inference.stream_policy import AdaptiveBitrateController, StreamPolicy
        spec = importlib.util.spec_from_file_location("sampled_readiness", ROOT / "packaging/release/verify-release-readiness.py")
        verifier = importlib.util.module_from_spec(spec); spec.loader.exec_module(verifier)
        payload = {"policyVersion": 1, "mode": "ADAPTIVE", "minBitrateKbps": 250,
                   "maxBitrateKbps": 2000, "initialBitrateKbps": 1000,
                   "congestionSamples": 2, "recoverySamples": 2, "cooldownSeconds": 1,
                   "increaseStepKbps": 200, "decreaseFactor": 0.5}
        controller = AdaptiveBitrateController(StreamPolicy.from_payload(payload))
        healthy = {"outboundPacketLossPct": 0, "outboundRttMs": 20,
                   "outboundPacketsDelta": 10, "outboundBytesDelta": 1000,
                   "connectivityQuality": "GOOD"}
        decisions = [controller.observe(healthy, now_ms=i * 1000) for i in range(14)]
        self.assertIn("awaiting_hysteresis:stable", {d.reason for d in decisions})
        self.assertIn("healthy_recovery", {d.reason for d in decisions})
        self.assertIn("at_maximum", {d.reason for d in decisions})
        for check in (MODULE, verifier):
            for decision in decisions:
                self.assertTrue(check._good_stream_reason(decision.reason, decision.bitrate_kbps, 2000, startup=True))
            self.assertTrue(check._poor_stream_reason("cooldown:connectivity_poor,packet_loss_high", 500, 250))
            self.assertTrue(check._poor_stream_reason("at_minimum", 250, 250))
            self.assertFalse(check._poor_stream_reason("at_minimum", 500, 250))
            self.assertFalse(check._good_stream_reason("at_maximum", 1000, 2000))
            for reason in ["not_connectivity_poor", "connectivity_poorish", "connectivity_poor,", "cooldown: connectiv ity_poor", {}, None]:
                self.assertFalse(check._poor_stream_reason(reason, 500, 250))
            for reason in ["outbound_progress_idle", "outbound_progress_unproven", "primary_stats_stale", "connectivity_poor", {}, None]:
                self.assertFalse(check._good_stream_reason(reason, 1000, 2000, startup=True))

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

    def test_cancelled_command_does_not_replace_actual_expiry_proof(self) -> None:
        class CancelledApi(_Api):
            def commands(self, *, space_id: int, device_id: str) -> list[dict]:
                commands = super().commands(space_id=space_id, device_id=device_id)
                commands.append({
                    "commandId": "00000000-0000-4000-8000-000000000002",
                    "sequence": 2, "type": "AGENT_UPDATE", "status": "EXPIRED",
                    "issuedAt": "2026-09-02T23:58:00.000Z",
                    "expiresAt": "2026-09-04T00:00:00.000Z",
                })
                return commands
        clock = _Clock()
        board = _Board()
        orchestrator = MODULE.ConfigStreamOrchestrator(
            api=CancelledApi(board), board=board, monotonic=clock.monotonic,
            sleeper=clock.sleep,
            wall_clock=lambda: datetime(2026, 9, 3, tzinfo=timezone.utc),
        )
        evidence = orchestrator.run(
            run_id=RUN_ID, manifest=_manifest(), manifest_sha256="1" * 64,
            ota_evidence_sha256="2" * 64, wait_seconds=120,
        )
        self.assertEqual([c["commandId"] for c in evidence["expiredPredecessors"]], [EXPIRED_COMMAND_ID])
        self.assertTrue(all(evidence["gates"].values()))

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

        with self.assertRaisesRegex(MODULE.ConfigStreamError, "expired predecessor"):
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

    def test_qualification_api_rejects_alternate_https_origin(self) -> None:
        with self.assertRaisesRegex(
            MODULE.ConfigStreamError, "authoritative Nuvion dev API"
        ):
            MODULE.FleetApi(
                "https://relay.example.invalid", bytearray(b"opaque-token")
            )

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

    def test_remote_program_uses_scoped_rtp_fault_without_interface_or_camera_mutation(self) -> None:
        program = MODULE.BOARD_PROGRAM
        compile(program, "<iq9075-config-stream-board>", "exec")
        self.assertIn("/run/nuvion-config-stream-e2e", program)
        self.assertNotIn('runtime / "bin/iw"', program)
        self.assertNotIn('runtime / "bin/ping"', program)
        self.assertIn("udp sport", program)
        self.assertIn("--on-active=60s", program)
        self.assertIn("videotestsrc is-live=true", program)
        self.assertNotIn("ip link", program)
        self.assertNotIn("/dev/video", program)
        self.assertNotIn("USB", program)
        self.assertIn("nuvion-config-stream-deadman-", program)
        self.assertIn("/run/lock/nuvion-fleet-e2e.lock", program)
        self.assertIn("/var/lib/nuvion-fleet-e2e/active-run.json", program)
        self.assertIn("config-stream-active.json", program)
        self.assertIn("def reboot_restore(rid):", program)
        self.assertIn('action == "reboot-restore"', program)
        self.assertNotIn("os.chmod(runtime.parent", program)
        restore_program = program[
            program.index("def restore(") : program.index("\ndef main()")
        ]
        reboot_program = program[
            program.index("def reboot_restore(") : program.index("\ndef main()")
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
        reboot_restored = reboot_program.index('"phase": "RESTORED"')
        reboot_purge = reboot_program.index("purge_snapshots(work)", reboot_restored)
        reboot_release = reboot_program.index(
            "release_config_lease(rid)", reboot_purge
        )
        reboot_deadman = reboot_program.index(
            "complete_deadman_cleanup", reboot_release
        )
        reboot_response = reboot_program.index(
            "return restoration_response", reboot_deadman
        )
        self.assertLess(reboot_restored, reboot_purge)
        self.assertLess(reboot_purge, reboot_release)
        self.assertLess(reboot_release, reboot_deadman)
        self.assertLess(reboot_deadman, reboot_response)
        self.assertNotIn('systemctl("start", "nuv-agent.service")', reboot_program)
        self.assertNotIn("runtime_identity(", reboot_program)

    def test_reboot_recovery_validator_requires_offline_exact_restoration(self) -> None:
        recovery = _Board().restore(run_id=RUN_ID)
        recovery.update(
            {
                "runtimeRestarted": False,
                "runtimeIdentity": None,
            }
        )

        self.assertEqual(
            MODULE.validate_reboot_recovery(recovery, run_id=RUN_ID), recovery
        )
        for field, invalid in (
            ("runtimeRestarted", True),
            ("runtimeIdentity", _runtime_identity(service_pid=303)),
            ("idempotent", "false"),
            ("exclusiveLeaseReleased", False),
        ):
            with self.subTest(field=field):
                candidate = {**recovery, field: invalid}
                with self.assertRaises(MODULE.ConfigStreamError):
                    MODULE.validate_reboot_recovery(candidate, run_id=RUN_ID)

    def test_shutdown_observation_is_preserved_until_worker_acknowledges(self) -> None:
        definitions = MODULE.BOARD_PROGRAM.split("\ntry:\n    main()", 1)[0]
        namespace = {"__name__": "board_drain_test"}
        exec(compile(definitions, "<board-drain-test>", "exec"), namespace)
        empty = {"inboxPendingRows": 0, "observationPendingRows": 0,
                 "observationReservedRows": 0, "observationDlqRows": 0}
        clock = [0.0]
        polls = []
        def counts():
            polls.append(clock[0])
            return {**empty, "observationPendingRows": int(clock[0] < 0.4)}
        def sleep(seconds):
            clock[0] += seconds
        namespace["time"] = SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep)
        namespace["db_counts"] = counts
        self.assertEqual(namespace["wait_observation_drain"](1), empty)
        self.assertGreaterEqual(clock[0], 0.4)
        self.assertGreaterEqual(len(polls), 3)
        namespace["db_counts"] = lambda: {**empty, "observationPendingRows": 1}
        with self.assertRaisesRegex(namespace["Failure"], "deadline"):
            namespace["wait_observation_drain"](0.4)
        self.assertLess(clock[0], 1.1)
        for key in ("inboxPendingRows", "observationReservedRows", "observationDlqRows"):
            namespace["db_counts"] = lambda key=key: {**empty, key: 1}
            before = clock[0]
            with self.assertRaisesRegex(namespace["Failure"], "not drained"):
                namespace["wait_observation_drain"](1)
            self.assertEqual(clock[0], before)

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


class RtpFaultLifecycleTest(unittest.TestCase):
    def _fixture(self, work):
        ns = _board_namespace()
        state = {"testServicePid": 101}
        kernel = {"present": False, "timer": False, "counter": {"packets": 100, "bytes": 120000}, "lost": None}
        binding = {k: _network_condition("ACTIVE")[k] for k in ("servicePid", "processStartTicks", "uid", "cgroup", "udpSourcePorts")}
        ns["atomic"] = lambda path, payload: path.write_bytes(payload)
        ns["packet_binding"] = lambda rid: dict(binding)
        ns["packet_tables"] = lambda: []
        ns["packet_counter"] = lambda fault, **kwargs: dict(kernel["counter"]) if kernel["present"] else None
        def systemctl(*args, **kwargs):
            if args[0] == "stop": kernel["timer"] = False
            return SimpleNamespace(stdout="active" if kernel["timer"] else "inactive", returncode=0)
        ns["systemctl"] = systemctl
        def run(args, **kwargs):
            saved = json.loads((work / "state.json").read_bytes())
            if args[0] == "/usr/bin/systemd-run":
                self.assertEqual(saved["packetFault"]["phase"], "ARMING")
                kernel["timer"] = True
                if kernel["lost"] == "arm": raise TimeoutError("lost timer response")
            elif args[:3] == ["/usr/sbin/nft", "-f", "-"]:
                self.assertTrue(kernel["timer"])
                self.assertEqual(saved["packetFault"]["phase"], "ARMING")
                kernel["present"] = True
                if kernel["lost"] == "apply": raise TimeoutError("lost apply response")
            elif args[:2] == ["/usr/sbin/nft", "delete"]:
                self.assertEqual(saved["packetFault"]["phase"], "RELEASING")
                kernel["present"] = False
                if kernel["lost"] == "release": raise TimeoutError("lost removal response")
            return SimpleNamespace(returncode=0, stdout="")
        ns["subprocess"] = SimpleNamespace(run=run, DEVNULL=-3, PIPE=-1)
        return ns, state, kernel

    def test_timer_and_apply_response_loss_restore_from_durable_intent(self):
        for stage in ("arm", "apply"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as d:
                work = Path(d); ns, state, kernel = self._fixture(work); kernel["lost"] = stage
                with self.assertRaises(TimeoutError): ns["packet_apply"](RUN_ID, state, work)
                recovered = json.loads((work / "state.json").read_bytes())
                self.assertEqual(recovered["packetFault"]["phase"], "ARMING")
                kernel["lost"] = None
                result = ns["packet_release"](RUN_ID, recovered, work)
                self.assertTrue(result["exactNetworkRestoration"])
                self.assertFalse(kernel["present"] or kernel["timer"])
                self.assertEqual(ns["packet_release"](RUN_ID, recovered, work), result)

    def test_release_response_loss_is_reconciled_without_reapplying_fault(self):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d); ns, state, kernel = self._fixture(work)
            ns["packet_apply"](RUN_ID, state, work); kernel["lost"] = "release"
            with self.assertRaises(TimeoutError): ns["packet_release"](RUN_ID, state, work, require_active=True)
            recovered = json.loads((work / "state.json").read_bytes())
            self.assertEqual(recovered["packetFault"]["phase"], "RELEASING")
            kernel["lost"] = None
            result = ns["packet_release"](RUN_ID, recovered, work)
            self.assertEqual(result["counter"]["packets"], 100)
            self.assertTrue(result["tableAbsent"])
            with self.assertRaises(ns["Failure"]): ns["packet_apply"](RUN_ID, recovered, work)

    def test_expired_timer_and_reboot_absence_recover_but_cannot_qualify(self):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d); ns, state, kernel = self._fixture(work)
            ns["packet_apply"](RUN_ID, state, work)
            kernel.update(present=False, timer=False)
            with self.assertRaises(ns["Failure"]): ns["packet_release"](RUN_ID, state, work, require_active=True)
            self.assertTrue(ns["packet_release"](RUN_ID, state, work)["exactNetworkRestoration"])

    def test_changed_or_foreign_rule_is_not_deleted(self):
        ns = _board_namespace(); fault = _network_condition("ACTIVE")
        original = ns["packet_shape"](fault["table"], fault["uid"], fault["udpSourcePorts"])
        for index in (0, 1, 2):
            with self.subTest(index=index):
                changed = json.loads(json.dumps(original))
                changed[2]["rule"]["expr"].pop(index)
                ns["nft_json"] = lambda *args, **kwargs: {"nftables": changed}
                with self.assertRaises(ns["Failure"]): ns["packet_counter"](fault)

    def test_host_and_independent_verifier_reject_incomplete_or_broad_evidence(self):
        spec = importlib.util.spec_from_file_location("rtp_readiness", ROOT / "packaging/release/verify-release-readiness.py")
        verifier = importlib.util.module_from_spec(spec); spec.loader.exec_module(verifier)
        for check, error in ((MODULE, MODULE.ConfigStreamError), (verifier, verifier.ReadinessError)):
            for phase in ("ACTIVE", "RELEASED"):
                original = _network_condition(phase)
                self.assertEqual(check.validate_network_condition(original, run_id=RUN_ID, service_pid=101, phase=phase), original)
                mutations = [("runId", str(uuid.uuid4())), ("uid", 0), ("uid", True), ("servicePid", 202),
                             ("udpSourcePorts", []), ("udpSourcePorts", [22]), ("udpSourcePorts", [31000, 31000]),
                             ("ruleShapeSha256", "b" * 64), ("dropPercent", 100), ("timerArmed", False),
                             ("counter", {"packets": 0, "bytes": 0}), ("counter", {"packets": True, "bytes": 5}),
                             ("automaticRemovalSeconds", 600), ("cgroup", "0::/other.service")]
                if phase == "RELEASED": mutations += [("timerDisarmed", False), ("tableAbsent", False)]
                for field, value in mutations:
                    with self.subTest(check=check.__name__, phase=phase, field=field):
                        modified = {**original, field: value}
                        with self.assertRaises(error): check.validate_network_condition(modified, run_id=RUN_ID, service_pid=101, phase=phase)
            poor, recovered = _network_condition("ACTIVE"), _network_condition("RELEASED")
            check.validate_network_transition(poor, recovered)
            for field, value in (("processStartTicks", 99999), ("previousTablesSha256", "f" * 64), ("counter", {"packets": 99, "bytes": 120000})):
                with self.assertRaises(error): check.validate_network_transition(poor, {**recovered, field: value})


class NativeRtpSocketScopeTest(unittest.TestCase):
    def test_packet_loss_only_affects_selected_unprivileged_udp_socket(self):
        import shutil
        import subprocess
        if sys.platform != "linux" or any(shutil.which(x) is None for x in ("nft", "ip", "unshare")):
            if os.environ.get("NUV_REQUIRE_NATIVE_NFT") == "1": self.fail("native nft prerequisites are missing")
            self.skipTest("native nft network namespaces are unavailable")
        prefix = [] if os.geteuid() == 0 else ["sudo", "-n"]
        program = r"""
import json, os, socket, subprocess, threading, time
ns = {"__name__": "native_socket_scope"}
exec(DEFINITIONS, ns)
subprocess.run(['/usr/sbin/ip', 'link', 'set', 'lo', 'up'], check=True)
table = 'nuvion_rtp_12345678123441238123123456789abc'
ports = [31000, 31001]
assert ns['packet_tables']() == []
for scoped_ports in ([31000], ports):
    rules = ns['packet_rules'](table, 65534, scoped_ports)
    subprocess.run(['/usr/sbin/nft', '-f', '-'], input=rules, text=True, check=True)
    fault = {'table': table, 'uid': 65534, 'udpSourcePorts': scoped_ports,
             'ruleShapeSha256': ns['sha'](ns['canonical'](ns['packet_shape'](table, 65534, scoped_ports)))}
    assert ns['packet_counter'](fault) == {'packets': 0, 'bytes': 0}
    subprocess.run(['/usr/sbin/nft', 'delete', 'table', 'inet', table], check=True)

def traffic():
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    rx.bind(('127.0.0.1', 54000)); rx.settimeout(0.1)
    received = {'selected': 0, 'control': 0}; done = threading.Event()
    def receive():
        while not done.is_set():
            try: packet, addr = rx.recvfrom(2048)
            except socket.timeout: continue
            received[packet.decode()] += 1
    thread = threading.Thread(target=receive); thread.start()
    sender = "import os,socket,time; os.setgroups([]); os.setgid(65534); os.setuid(65534); a=socket.socket(2,2); b=socket.socket(2,2); a.bind(('127.0.0.1',31000)); b.bind(('127.0.0.1',31002));\nfor i in range(1000):\n try: a.sendto(b'selected',('127.0.0.1',54000))\n except PermissionError: pass\n b.sendto(b'control',('127.0.0.1',54000)); time.sleep(.001)"
    try:
        subprocess.run(['/usr/bin/python3', '-c', sender], check=True, timeout=10)
        time.sleep(.2)
    finally:
        done.set(); thread.join(2); rx.close()
    return received
try:
    subprocess.run(['/usr/sbin/nft', '-f', '-'], input=ns['packet_rules'](table,65534,ports), text=True, check=True)
    affected = traffic(); count = ns['packet_counter'](fault)
    assert affected['control'] == 1000, affected
    assert 450 < affected['selected'] < 850, affected
    assert count['packets'] == 1000 - affected['selected'], (count, affected)
finally:
    subprocess.run(['/usr/sbin/nft', 'delete', 'table', 'inet', table], check=True)
assert ns['packet_tables']() == []
assert traffic() == {'selected': 1000, 'control': 1000}
print(json.dumps({'socketScopeVerified':True,'affected':affected,'dropped':count,'exactRestoration':True}))
""".replace("DEFINITIONS", repr(MODULE.BOARD_PROGRAM.split("\ntry:\n    main()", 1)[0]))
        result = subprocess.run(prefix + ["unshare", "--net", "/usr/bin/python3", "-B", "-c", program], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        outcome = json.loads(result.stdout)
        self.assertTrue(outcome["socketScopeVerified"] and outcome["exactRestoration"])


if __name__ == "__main__":
    unittest.main()
