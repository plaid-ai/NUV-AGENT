from __future__ import annotations

import base64
import hashlib
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from nuvion_app.inference.agent_update import (
    AgentUpdateReconciler,
    configure_agent_update_reconciler,
)
from nuvion_app.inference.command_runtime import (
    build_fleet_command_runtime,
)
from nuvion_app.inference.effect_reconciler import ReconcilerRegistry
from nuvion_app.inference.fleet_command import Ed25519Keyring, VerifiedFleetCommand
from nuvion_app.runtime.updater_client import UpdaterClientError


def _command() -> VerifiedFleetCommand:
    payload = {"targetVersion": "0.1.116", "bomDigest": "sha256:" + "a" * 64}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    command_id = str(uuid.uuid4())
    return VerifiedFleetCommand(
        command_id=command_id,
        device_id="sp-3-device",
        space_id=3,
        command_type="AGENT_UPDATE",
        schema_version=1,
        issued_at="2026-09-01T10:00:00Z",
        expires_at="2026-09-01T10:10:00Z",
        sequence=1,
        payload_base64=base64.urlsafe_b64encode(raw).decode("ascii").rstrip("="),
        payload_hash=hashlib.sha256(raw).hexdigest(),
        payload=payload,
        actor="release-manager@example.test",
        authorization_context="SPACE_ADMIN",
        key_id="test",
        required_capability="command.agent.update",
        compact_jws="header.claims.signature",
    )


def _committed(command: VerifiedFleetCommand) -> dict[str, object]:
    return {
        "commandId": command.command_id,
        "phase": "COMMITTED",
        "targetVersion": command.payload["targetVersion"],
        "bomDigest": command.payload["bomDigest"],
        "artifactDigest": "sha256:" + "b" * 64,
        "componentSha": "c" * 40,
        "configSchema": "12",
        "releaseSequence": 42,
        "bomVerificationStatus": "VERIFIED",
        "health": "FUNCTIONAL_HEALTHY",
        "functionalHealth": "FUNCTIONAL_HEALTHY",
        "slot": "releases/" + "a" * 64,
        "previousVersion": "0.1.115",
    }


class _Client:
    def __init__(self, *, initial: dict[str, object] | None = None) -> None:
        self.initial = initial
        self.calls: list[str] = []

    def capability_available(self) -> bool:
        return True

    def status(self, _command_id: str | None = None) -> dict[str, object]:
        self.calls.append("STATUS")
        return {"update": self.initial}

    def authorize_and_stage(self, _compact: str) -> dict[str, object]:
        self.calls.append("AUTHORIZE_AND_STAGE")
        return {"phase": "VERIFIED"}

    def activate(self, command_id: str) -> dict[str, object]:
        self.calls.append("ACTIVATE")
        return {"commandId": command_id, "phase": "ACTIVATING"}

    def report_boot_health(
        self, command_id: str, *, healthy: bool, detail: str | None = None
    ) -> dict[str, object]:
        self.calls.append("REPORT_BOOT_HEALTH")
        return {"commandId": command_id, "phase": "BOOT_HEALTHY"}

    def report_functional_health(
        self, command_id: str, *, healthy: bool, detail: str | None = None
    ) -> dict[str, object]:
        self.calls.append("REPORT_FUNCTIONAL_HEALTH")
        return {"commandId": command_id, "phase": "FUNCTIONAL_HEALTHY"}

    def commit(self, command_id: str) -> dict[str, object]:
        self.calls.append("COMMIT")
        command = _command()
        object.__setattr__(command, "command_id", command_id)
        return _committed(command)


class AgentUpdateReconcilerTest(unittest.TestCase):
    def test_registry_keeps_handler_while_live_capability_fails_closed(self) -> None:
        registry = ReconcilerRegistry()
        ready_client = mock.Mock()
        ready_client.capability_status.return_value = {
            "capabilityAvailable": True,
            "authenticatedHelper": True,
            "reason": "READY",
        }
        status = configure_agent_update_reconciler(
            registry,
            client=ready_client,
        )
        self.assertTrue(status["capabilityAvailable"])
        registered = registry.get("AGENT_UPDATE")
        self.assertIsInstance(registered, AgentUpdateReconciler)
        self.assertIn("command.agent.update", registry.capabilities)

        unavailable = mock.Mock()
        unavailable.capability_status.return_value = {
            "capabilityAvailable": False,
            "authenticatedHelper": False,
            "reason": "UNSAFE_UPDATER_PEER",
        }
        configure_agent_update_reconciler(registry, client=unavailable)
        self.assertIsInstance(
            registry.get("AGENT_UPDATE"),
            AgentUpdateReconciler,
        )
        self.assertNotIn("command.agent.update", registry.capabilities)

    def test_is_effect_reconciler_and_activation_defers_for_restart(self) -> None:
        command = _command()
        client = _Client()
        outcome = AgentUpdateReconciler(client).reconcile(command)  # type: ignore[arg-type]
        self.assertEqual(AgentUpdateReconciler.command_type, "AGENT_UPDATE")
        self.assertEqual(AgentUpdateReconciler.capability, "command.agent.update")
        self.assertEqual(outcome.checkpoint["updaterPhase"], "ACTIVATING")
        self.assertTrue(outcome.checkpoint["restartExpected"])
        self.assertTrue(outcome.checkpoint["restartRequired"])
        self.assertEqual(outcome.checkpoint["nextAction"], "RESTART_AGENT")
        self.assertEqual(
            client.calls,
            ["STATUS", "AUTHORIZE_AND_STAGE", "ACTIVATE"],
        )

    def test_committed_evidence_is_full_desired_superset(self) -> None:
        command = _command()
        client = _Client(
            initial={"commandId": command.command_id, "phase": "FUNCTIONAL_HEALTHY"}
        )
        outcome = AgentUpdateReconciler(
            client,  # type: ignore[arg-type]
            commit_readiness_provider=lambda: {"ready": True, "reason": "READY"},
        ).reconcile(command)

        self.assertEqual(outcome.status, "SUCCEEDED")
        self.assertEqual(outcome.reported_state["updatePhase"], "COMMITTED")
        self.assertEqual(outcome.reported_state["targetVersion"], "0.1.116")
        self.assertEqual(outcome.reported_state["agentVersion"], "0.1.116")
        self.assertEqual(outcome.reported_state["releaseSequence"], 42)
        self.assertEqual(
            outcome.reported_state["functionalHealth"], "FUNCTIONAL_HEALTHY"
        )
        self.assertEqual(client.calls, ["STATUS", "COMMIT"])

    def test_functional_health_waits_for_runtime_readiness_without_restart(self) -> None:
        command = _command()
        client = _Client(
            initial={"commandId": command.command_id, "phase": "FUNCTIONAL_HEALTHY"}
        )
        outcome = AgentUpdateReconciler(
            client,  # type: ignore[arg-type]
            commit_readiness_provider=lambda: {
                "ready": False,
                "reason": "STOMP_SOAK_PENDING",
            },
        ).reconcile(command)

        self.assertEqual(client.calls, ["STATUS"])
        self.assertFalse(outcome.checkpoint["restartRequired"])
        self.assertEqual(outcome.checkpoint["nextAction"], "RETRY_EFFECT")
        self.assertEqual(outcome.checkpoint["detail"], "STOMP_SOAK_PENDING")

    def test_missing_or_invalid_runtime_readiness_fails_closed(self) -> None:
        command = _command()
        for provider in (
            None,
            lambda: {"ready": True, "reason": "unsafe reason"},
            lambda: (_ for _ in ()).throw(RuntimeError("unavailable")),
        ):
            with self.subTest(provider=provider):
                client = _Client(
                    initial={
                        "commandId": command.command_id,
                        "phase": "FUNCTIONAL_HEALTHY",
                    }
                )
                outcome = AgentUpdateReconciler(
                    client,  # type: ignore[arg-type]
                    commit_readiness_provider=provider,
                ).reconcile(command)
                self.assertEqual(client.calls, ["STATUS"])
                self.assertEqual(
                    outcome.checkpoint["detail"],
                    "RUNTIME_READINESS_UNAVAILABLE",
                )

    def test_committed_without_strong_evidence_fails_closed(self) -> None:
        command = _command()
        client = _Client(initial={"commandId": command.command_id, "phase": "COMMITTED"})
        outcome = AgentUpdateReconciler(client).reconcile(command)  # type: ignore[arg-type]
        self.assertEqual(outcome.status, "FAILED")
        self.assertEqual(outcome.code, "UPDATE_EVIDENCE_INCOMPLETE")

    def test_committed_helper_identity_must_match_desired_command(self) -> None:
        command = _command()
        evidence = _committed(command)
        evidence["bomDigest"] = "sha256:" + "f" * 64
        client = _Client(initial=evidence)

        outcome = AgentUpdateReconciler(client).reconcile(command)  # type: ignore[arg-type]

        self.assertEqual(outcome.status, "FAILED")
        self.assertEqual(outcome.code, "UPDATE_EVIDENCE_INCOMPLETE")

    def test_boot_and_functional_probe_defer_for_second_restart(self) -> None:
        command = _command()
        client = _Client(initial={"commandId": command.command_id, "phase": "ACTIVATING"})
        outcome = AgentUpdateReconciler(client).reconcile(command)  # type: ignore[arg-type]
        self.assertEqual(
            client.calls,
            ["STATUS", "REPORT_BOOT_HEALTH", "REPORT_FUNCTIONAL_HEALTH"],
        )
        self.assertTrue(outcome.checkpoint["restartExpected"])

    def test_activation_connection_loss_is_deferred_for_restart_resume(self) -> None:
        command = _command()
        client = _Client(initial={"commandId": command.command_id, "phase": "VERIFIED"})

        def interrupted(_command_id: str) -> dict[str, object]:
            raise UpdaterClientError(
                "INVALID_RESPONSE", "Agent service restarted before socket response"
            )

        client.activate = interrupted  # type: ignore[method-assign]
        outcome = AgentUpdateReconciler(client).reconcile(command)  # type: ignore[arg-type]
        self.assertTrue(outcome.checkpoint["restartExpected"])

    def test_transient_helper_failure_retries_without_process_restart(self) -> None:
        command = _command()
        client = _Client()

        def unavailable(_command_id: str | None = None) -> dict[str, object]:
            raise UpdaterClientError("UPDATER_UNAVAILABLE", "helper unavailable")

        client.status = unavailable  # type: ignore[method-assign]
        outcome = AgentUpdateReconciler(client).reconcile(command)  # type: ignore[arg-type]

        self.assertFalse(outcome.checkpoint["restartRequired"])
        self.assertEqual(outcome.checkpoint["nextAction"], "RETRY_EFFECT")

    def test_runtime_transaction_handler_only_checkpoints_desired_state(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            private_key = Ed25519PrivateKey.generate()
            public_key = private_key.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
            identity = SimpleNamespace(
                identity_status="DEV",
                platform_profile="macos_dev",
                capabilities=frozenset(
                    {"command.config.apply", "command.agent.update"}
                ),
            )
            readiness = {
                "capabilityAvailable": False,
                "authenticatedHelper": False,
            }
            registry = ReconcilerRegistry()
            registry.register(
                AgentUpdateReconciler(
                    _Client(),  # type: ignore[arg-type]
                    readiness_provider=lambda: dict(readiness),
                )
            )
            with mock.patch(
                "nuvion_app.inference.command_runtime.load_fleet_command_keyring",
                return_value=Ed25519Keyring({"test": public_key}),
            ):
                runtime = build_fleet_command_runtime(
                    base_url="https://api.example.test",
                    access_token_provider=lambda: "token",
                    ack_sender=lambda _destination, _payload: True,
                    device_id="sp-3-device",
                    space_id=3,
                    keyring_path=root / "unused.json",
                    inbox_path=root / "commands.sqlite3",
                    platform_identity=identity,
                    reconciler_registry=registry,
                )
            self.assertNotIn("command.agent.update", runtime.processor.verifier.capabilities)
            self.assertIn("AGENT_UPDATE", runtime.processor.handlers)

            readiness.update(
                {
                    "capabilityAvailable": True,
                    "authenticatedHelper": True,
                }
            )
            self.assertIn(
                "command.agent.update", runtime.processor.verifier.capabilities
            )
            self.assertIs(
                runtime.processor.handlers["AGENT_UPDATE"].__self__,
                runtime.reconcile_store,
            )


if __name__ == "__main__":
    unittest.main()
