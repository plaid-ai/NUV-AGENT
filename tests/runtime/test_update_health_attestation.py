from __future__ import annotations

import unittest
import uuid
from datetime import datetime, timezone

from nuvion_app.runtime.update_health_attestation import (
    UpdateHealthAttestationError,
    build_health_attestation_request,
    parse_health_attestation_response,
    request_health_attestation,
)
from nuvion_updater.store import CommitGate


class UpdateHealthAttestationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.command_id = str(uuid.uuid4())
        self.gate_id = str(uuid.uuid4())
        self.bom_digest = "sha256:" + "a" * 64
        self.component_sha = "b" * 40
        self.gate = {
            "schemaVersion": 1,
            "gateId": self.gate_id,
            "challenge": "A" * 43,
            "commandId": self.command_id,
            "commandExpiresAt": "2026-09-02T10:05:00Z",
            "bomDigest": self.bom_digest,
            "componentSha": self.component_sha,
            "releaseSequence": 2,
            "candidateSlot": "releases/" + "a" * 64,
            "agentPid": 412,
            "agentStartTicks": 123456,
            "bootId": str(uuid.uuid4()),
            "expiresAt": "2026-09-02T10:01:00Z",
        }
        self.identity = {
            "productModel": "IQ9075_DEV",
            "platformProfile": "iq9075_dev",
            "hardwareRevision": "QCS9075-EVK",
            "architecture": "aarch64",
        }

    def _request(self, **overrides: object) -> dict[str, object]:
        arguments: dict[str, object] = {
            "device_id": "sp-33-nuvion-c4be72ac",
            "command_id": self.command_id,
            "expected_bom_digest": self.bom_digest,
            "expected_component_sha": self.component_sha,
            "expected_release_sequence": 2,
            "gate": self.gate,
            "identity": self.identity,
        }
        arguments.update(overrides)
        return build_health_attestation_request(**arguments)  # type: ignore[arg-type]

    def test_request_is_exactly_bound_to_root_gate_and_platform_identity(self) -> None:
        request = self._request()

        self.assertEqual(
            set(request),
            {
                "deviceId",
                "commandId",
                "gateId",
                "challenge",
                "bomDigest",
                "componentSha",
                "releaseSequence",
                "productModel",
                "platformProfile",
                "hardwareRevision",
                "architecture",
            },
        )
        self.assertEqual(request["gateId"], self.gate_id)
        self.assertEqual(request["productModel"], "IQ9075_DEV")

    def test_real_updater_public_commit_gate_is_accepted(self) -> None:
        gate = CommitGate(
            command_id=self.command_id,
            command_expires_at="2026-09-02T10:05:00Z",
            gate_id=self.gate_id,
            challenge="A" * 43,
            peer_pid=412,
            agent_start_ticks=123456,
            boot_id=self.gate["bootId"],
            candidate_slot="releases/" + "a" * 64,
            bom_digest=self.bom_digest,
            component_sha=self.component_sha,
            release_sequence=2,
            health_deadline="2026-09-02T10:01:00Z",
            created_at="2026-09-02T10:00:00Z",
            attestation_id=None,
            attestation_jws_sha256=None,
            consumed_at=None,
        )

        request = self._request(gate=gate.public_dict())

        self.assertEqual(request["commandId"], self.command_id)
        self.assertEqual(request["gateId"], self.gate_id)

    def test_gate_unknown_field_and_release_identity_mismatch_fail_closed(self) -> None:
        unknown = {**self.gate, "callerHealth": "HEALTHY"}
        with self.assertRaisesRegex(UpdateHealthAttestationError, "schema"):
            self._request(gate=unknown)

        mismatched = {**self.gate, "componentSha": "c" * 40}
        with self.assertRaisesRegex(UpdateHealthAttestationError, "component SHA"):
            self._request(gate=mismatched)

        wrong_slot = {**self.gate, "candidateSlot": "releases/" + "d" * 64}
        with self.assertRaisesRegex(UpdateHealthAttestationError, "candidate slot"):
            self._request(gate=wrong_slot)

    def test_strict_api_envelope_returns_only_public_attestation_metadata(self) -> None:
        response = {
            "message": "issued",
            "data": {
                "keyId": "health-iq9075-dev-2026",
                "issuedAt": "2026-09-02T10:00:00Z",
                "expiresAt": "2026-09-02T10:00:30Z",
                "compactJws": "a.b.c",
            },
        }
        parsed = parse_health_attestation_response(
            response,
            now=datetime(2026, 9, 2, 10, 0, 10, tzinfo=timezone.utc),
        )

        self.assertEqual(set(parsed), {"keyId", "issuedAt", "expiresAt", "compactJws"})
        self.assertEqual(parsed["compactJws"], "a.b.c")

    def test_response_extra_fields_bad_jws_ttl_and_expiry_are_rejected(self) -> None:
        valid = {
            "keyId": "health-test",
            "issuedAt": "2026-09-02T10:00:00Z",
            "expiresAt": "2026-09-02T10:00:30Z",
            "compactJws": "a.b.c",
        }
        cases = (
            (
                {"message": "issued", "data": {**valid, "gateId": self.gate_id}},
                "schema",
            ),
            (
                {"message": "issued", "data": {**valid, "compactJws": "not-a-jws"}},
                "base64url",
            ),
            (
                {
                    "message": "issued",
                    "data": {**valid, "expiresAt": "2026-09-02T10:01:01Z"},
                },
                "TTL",
            ),
        )
        for response, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(UpdateHealthAttestationError, message):
                    parse_health_attestation_response(
                        response,
                        now=datetime(2026, 9, 2, 10, 0, 10, tzinfo=timezone.utc),
                    )

        with self.assertRaisesRegex(UpdateHealthAttestationError, "expired"):
            parse_health_attestation_response(
                {"message": "issued", "data": valid},
                now=datetime(2026, 9, 2, 10, 1, 0, tzinfo=timezone.utc),
            )

        for invalid_envelope in (
            valid,
            {"data": valid},
            {"message": "issued", "data": valid, "debug": True},
        ):
            with self.subTest(invalid_envelope=invalid_envelope):
                with self.assertRaisesRegex(
                    UpdateHealthAttestationError, "envelope"
                ):
                    parse_health_attestation_response(
                        invalid_envelope,
                        now=datetime(2026, 9, 2, 10, 0, 10, tzinfo=timezone.utc),
                    )

    def test_transport_failure_is_retryable_until_root_watchdog_rolls_back(self) -> None:
        with self.assertRaisesRegex(UpdateHealthAttestationError, "failed") as raised:
            request_health_attestation(
                self._request(),
                transport=lambda _request: (_ for _ in ()).throw(OSError("offline")),
            )

        self.assertEqual(raised.exception.code, "HEALTH_ATTESTATION_UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
