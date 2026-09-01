from __future__ import annotations

import base64
import copy
import hashlib
import json
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.x509.oid import NameOID

from nuvion_app.inference.fleet_command import (
    CommandValidationError,
    Ed25519Keyring,
    FleetCommandVerifier,
)

NOW = datetime(2026, 9, 1, 2, 5, tzinfo=timezone.utc)
DEVICE_ID = "sp-3-nuvion-test"
SPACE_ID = 3
KID = "test-only-key"


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


class FleetCommandVerifierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.private_key = Ed25519PrivateKey.generate()
        self.raw_public_key = self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        self.verifier = self._verifier(Ed25519Keyring({KID: self.raw_public_key}))

    def _verifier(
        self,
        keyring: Ed25519Keyring,
        *,
        device_id: str = DEVICE_ID,
        space_id: int = SPACE_ID,
        capabilities: frozenset[str] = frozenset({"command.config.apply"}),
    ) -> FleetCommandVerifier:
        return FleetCommandVerifier(
            keyring=keyring,
            expected_device_id=device_id,
            expected_space_id=space_id,
            capabilities=capabilities,
            clock=lambda: NOW,
            allowed_clock_skew=timedelta(seconds=30),
        )

    def _claims(self, **overrides: object) -> dict[str, object]:
        payload = _json_bytes(
            {
                "configVersion": 11,
                "activation": "IMMEDIATE",
                "clip": {
                    "enabled": True,
                    "preSeconds": 5,
                    "postSeconds": 5,
                },
            }
        )
        claims: dict[str, object] = {
            "commandId": str(uuid.uuid4()),
            "deviceId": DEVICE_ID,
            "spaceId": SPACE_ID,
            "type": "CONFIG_APPLY",
            "schemaVersion": 1,
            "issuedAt": "2026-09-01T02:00:00Z",
            "expiresAt": "2026-09-01T02:10:00Z",
            "sequence": 42,
            "payloadBase64": _b64url(payload),
            "payloadHash": hashlib.sha256(payload).hexdigest(),
            "actor": "operator@example.com",
            "authorizationContext": "SPACE_ADMIN",
        }
        claims.update(overrides)
        return claims

    def _sign(
        self,
        claims: dict[str, object] | None = None,
        *,
        header: dict[str, object] | None = None,
        private_key: Ed25519PrivateKey | None = None,
    ) -> str:
        protected = header or {"alg": "EdDSA", "kid": KID, "typ": "nuvion-command+jws"}
        protected_segment = _b64url(_json_bytes(protected))
        claims_segment = _b64url(_json_bytes(claims or self._claims()))
        signing_input = f"{protected_segment}.{claims_segment}".encode("ascii")
        signature = (private_key or self.private_key).sign(signing_input)
        return f"{protected_segment}.{claims_segment}.{_b64url(signature)}"

    def _claims_with_payload(
        self,
        payload: object,
        **overrides: object,
    ) -> dict[str, object]:
        encoded = _json_bytes(payload)
        return self._claims(
            payloadBase64=_b64url(encoded),
            payloadHash=hashlib.sha256(encoded).hexdigest(),
            **overrides,
        )

    def _assert_code(self, expected: str, compact_jws: str) -> None:
        with self.assertRaises(CommandValidationError) as raised:
            self.verifier.verify(compact_jws)
        self.assertEqual(raised.exception.code, expected)

    def test_verifies_signature_binding_payload_and_capability(self) -> None:
        claims = self._claims()

        command = self.verifier.verify(self._sign(claims))

        self.assertEqual(command.command_id, claims["commandId"])
        self.assertEqual(command.device_id, DEVICE_ID)
        self.assertEqual(command.space_id, SPACE_ID)
        self.assertEqual(command.command_type, "CONFIG_APPLY")
        self.assertEqual(
            command.payload,
            {
                "configVersion": 11,
                "activation": "IMMEDIATE",
                "clip": {
                    "enabled": True,
                    "preSeconds": 5,
                    "postSeconds": 5,
                },
            },
        )
        self.assertEqual(command.required_capability, "command.config.apply")
        self.assertEqual(command.key_id, KID)

    def test_accepts_der_subject_public_key_info(self) -> None:
        der = self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        verifier = self._verifier(Ed25519Keyring({KID: der}))

        self.assertEqual(verifier.verify(self._sign()).key_id, KID)

    def test_accepts_der_x509_certificate_with_ed25519_public_key(self) -> None:
        subject = issuer = x509.Name(
            [x509.NameAttribute(NameOID.COMMON_NAME, "test-only")]
        )
        certificate = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(self.private_key.public_key())
            .serial_number(1)
            .not_valid_before(NOW - timedelta(days=1))
            .not_valid_after(NOW + timedelta(days=1))
            .sign(self.private_key, algorithm=None)
        )
        certificate_der = certificate.public_bytes(serialization.Encoding.DER)
        verifier = self._verifier(Ed25519Keyring({KID: certificate_der}))

        self.assertEqual(verifier.verify(self._sign()).key_id, KID)

    def test_shared_java_python_contract_vector_verifies_with_raw_and_x509_keys(
        self,
    ) -> None:
        fixture_path = (
            Path(__file__).resolve().parents[3]
            / "architecture"
            / "contracts"
            / "fixtures"
            / "fleet-command-v1-ed25519.json"
        )
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        self.assertIs(fixture["testOnly"], True)
        key_id = fixture["keyId"]
        compact_jws = fixture["compactJws"]
        expected_claims = fixture["claims"]

        for field in ("publicKeyRawBase64", "publicKeyX509Base64"):
            with self.subTest(public_key_format=field):
                public_material = base64.b64decode(fixture[field], validate=True)
                verifier = FleetCommandVerifier(
                    keyring=Ed25519Keyring({key_id: public_material}),
                    expected_device_id=expected_claims["deviceId"],
                    expected_space_id=expected_claims["spaceId"],
                    capabilities={"command.config.apply"},
                    clock=lambda: NOW,
                    allowed_clock_skew=timedelta(seconds=30),
                )
                rejection = verifier.verify_for_rejection(compact_jws)
                self.assertEqual(
                    rejection.command.command_id,
                    expected_claims["commandId"],
                )
                self.assertEqual(rejection.code, "INVALID_PAYLOAD_SCHEMA")
                self.assertEqual(
                    rejection.command.payload_hash,
                    fixture["payloadHash"],
                )

    def test_rejects_algorithm_type_and_unknown_key_id(self) -> None:
        cases = (
            (
                "UNSUPPORTED_ALGORITHM",
                {"alg": "none", "kid": KID, "typ": "nuvion-command+jws"},
            ),
            (
                "INVALID_JWS_TYPE",
                {"alg": "EdDSA", "kid": KID, "typ": "JWT"},
            ),
            (
                "UNKNOWN_KEY_ID",
                {"alg": "EdDSA", "kid": "remote-key", "typ": "nuvion-command+jws"},
            ),
        )
        for expected_code, header in cases:
            with self.subTest(expected_code=expected_code):
                self._assert_code(expected_code, self._sign(header=header))

    def test_rejects_remote_key_and_encoding_override_headers(self) -> None:
        for field, value in (
            ("jku", "https://attacker.invalid/jwks.json"),
            ("x5u", "https://attacker.invalid/cert.der"),
            ("b64", False),
            ("crit", ["b64"]),
        ):
            with self.subTest(field=field):
                header: dict[str, object] = {
                    "alg": "EdDSA",
                    "kid": KID,
                    "typ": "nuvion-command+jws",
                    field: value,
                }
                self._assert_code("UNSAFE_PROTECTED_HEADER", self._sign(header=header))

    def test_checks_signature_before_untrusted_claims(self) -> None:
        compact = self._sign()
        protected_segment, _claims_segment, signature_segment = compact.split(".")
        invalid_claims_segment = _b64url(b"not-json")

        self._assert_code(
            "INVALID_SIGNATURE",
            f"{protected_segment}.{invalid_claims_segment}.{signature_segment}",
        )

    def test_rejects_invalid_claim_types_and_device_space_binding(self) -> None:
        cases = (
            ("INVALID_CLAIMS", {"sequence": True}),
            ("INVALID_CLAIMS", {"commandId": "not-a-uuid"}),
            ("INVALID_CLAIMS", {"type": "config_apply"}),
            ("DEVICE_MISMATCH", {"deviceId": "another-device"}),
            ("SPACE_MISMATCH", {"spaceId": 999}),
        )
        for expected_code, overrides in cases:
            with self.subTest(overrides=overrides):
                self._assert_code(expected_code, self._sign(self._claims(**overrides)))

    def test_rejects_not_yet_valid_expired_and_inverted_windows(self) -> None:
        cases = (
            (
                "NOT_YET_VALID",
                {
                    "issuedAt": "2026-09-01T02:06:00Z",
                    "expiresAt": "2026-09-01T02:10:00Z",
                },
            ),
            (
                "EXPIRED",
                {
                    "issuedAt": "2026-09-01T01:00:00Z",
                    "expiresAt": "2026-09-01T02:04:00Z",
                },
            ),
            (
                "INVALID_TIME_WINDOW",
                {
                    "issuedAt": "2026-09-01T02:00:00Z",
                    "expiresAt": "2026-09-01T02:00:00Z",
                },
            ),
        )
        for expected_code, overrides in cases:
            with self.subTest(expected_code=expected_code):
                self._assert_code(expected_code, self._sign(self._claims(**overrides)))

    def test_expired_command_can_only_be_reverified_for_durable_rejection(self) -> None:
        expired = self._sign(
            self._claims(
                issuedAt="2026-09-01T01:00:00Z",
                expiresAt="2026-09-01T02:04:00Z",
            )
        )

        verified = self.verifier.verify_expired_for_rejection(expired)

        self.assertEqual(verified.sequence, 42)
        with self.assertRaises(CommandValidationError) as raised:
            self.verifier.verify_expired_for_rejection(self._sign())
        self.assertEqual(raised.exception.code, "NOT_EXPIRED")

    def test_authenticated_execution_policy_errors_return_rejection_proof(self) -> None:
        no_capability_verifier = self._verifier(
            Ed25519Keyring({KID: self.raw_public_key}), capabilities=frozenset()
        )
        cases = (
            (
                self.verifier,
                self._sign(self._claims(schemaVersion=2)),
                "UNSUPPORTED_SCHEMA",
            ),
            (
                self.verifier,
                self._sign(self._claims(type="FACTORY_RESET")),
                "UNSUPPORTED_COMMAND",
            ),
            (
                no_capability_verifier,
                self._sign(),
                "MISSING_CAPABILITY",
            ),
            (
                self.verifier,
                self._sign(self._claims_with_payload({})),
                "INVALID_PAYLOAD_SCHEMA",
            ),
            (
                self.verifier,
                self._sign(
                    self._claims(
                        issuedAt="2026-09-01T01:00:00Z",
                        expiresAt="2026-09-01T02:04:00Z",
                    )
                ),
                "EXPIRED",
            ),
        )

        for verifier, compact_jws, expected_code in cases:
            with self.subTest(expected_code=expected_code):
                rejection = verifier.verify_for_rejection(compact_jws)
                self.assertEqual(rejection.code, expected_code)
                self.assertEqual(rejection.command.device_id, DEVICE_ID)
                self.assertEqual(rejection.command.space_id, SPACE_ID)

    def test_authorization_context_and_max_ttl_are_authenticated_policy(self) -> None:
        self.assertEqual(
            self.verifier.verify(self._sign()).authorization_context,
            "SPACE_ADMIN",
        )

        unsupported_context = self.verifier.verify_for_rejection(
            self._sign(self._claims(authorizationContext="VIEWER"))
        )
        self.assertEqual(
            unsupported_context.code,
            "UNSUPPORTED_AUTHORIZATION_CONTEXT",
        )
        self.assertEqual(unsupported_context.command.authorization_context, "VIEWER")

        oversized_window = self.verifier.verify_for_rejection(
            self._sign(
                self._claims(
                    issuedAt="2026-09-01T02:00:00Z",
                    expiresAt="2026-09-02T02:00:01Z",
                )
            )
        )
        self.assertEqual(oversized_window.code, "INVALID_TIME_WINDOW")

        with self.assertRaises(CommandValidationError) as future:
            self.verifier.verify_for_rejection(
                self._sign(
                    self._claims(
                        issuedAt="2026-09-01T02:06:00Z",
                        expiresAt="2026-09-02T02:06:01Z",
                    )
                )
            )
        self.assertEqual(future.exception.code, "NOT_YET_VALID")

    def test_authorization_policy_cannot_bypass_signature_or_identity(self) -> None:
        viewer_claims = self._claims(authorizationContext="VIEWER")
        attacker_key = Ed25519PrivateKey.generate()
        cases = (
            (
                "INVALID_SIGNATURE",
                self._sign(viewer_claims, private_key=attacker_key),
            ),
            (
                "DEVICE_MISMATCH",
                self._sign(
                    self._claims(
                        authorizationContext="VIEWER",
                        deviceId="another-device",
                    )
                ),
            ),
            (
                "SPACE_MISMATCH",
                self._sign(
                    self._claims(
                        authorizationContext="VIEWER",
                        spaceId=999,
                    )
                ),
            ),
        )

        for expected_code, compact_jws in cases:
            with self.subTest(expected_code=expected_code):
                with self.assertRaises(CommandValidationError) as raised:
                    self.verifier.verify_for_rejection(compact_jws)
                self.assertEqual(raised.exception.code, expected_code)

    def test_policy_error_cannot_disguise_authentication_or_integrity_failure(
        self,
    ) -> None:
        unsupported_claims = self._claims(schemaVersion=2)
        attacker_key = Ed25519PrivateKey.generate()
        array_payload = _json_bytes(["not", "an", "object"])
        cases = (
            (
                "UNKNOWN_KEY_ID",
                self._sign(
                    unsupported_claims,
                    header={
                        "alg": "EdDSA",
                        "kid": "attacker-key",
                        "typ": "nuvion-command+jws",
                    },
                ),
            ),
            (
                "INVALID_SIGNATURE",
                self._sign(unsupported_claims, private_key=attacker_key),
            ),
            (
                "DEVICE_MISMATCH",
                self._sign(self._claims(schemaVersion=2, deviceId="another-device")),
            ),
            (
                "SPACE_MISMATCH",
                self._sign(self._claims(schemaVersion=2, spaceId=999)),
            ),
            (
                "INVALID_CLAIMS",
                self._sign(self._claims(schemaVersion=2, sequence=True)),
            ),
            (
                "INVALID_PAYLOAD_HASH",
                self._sign(self._claims(schemaVersion=2, payloadHash="0" * 64)),
            ),
            (
                "INVALID_PAYLOAD",
                self._sign(
                    self._claims(
                        schemaVersion=2,
                        payloadBase64=_b64url(array_payload),
                        payloadHash=hashlib.sha256(array_payload).hexdigest(),
                    )
                ),
            ),
            (
                "NOT_YET_VALID",
                self._sign(
                    self._claims(
                        schemaVersion=2,
                        issuedAt="2026-09-01T02:06:00Z",
                        expiresAt="2026-09-01T02:10:00Z",
                    )
                ),
            ),
            (
                "INVALID_TIME_WINDOW",
                self._sign(
                    self._claims(
                        schemaVersion=2,
                        issuedAt="2026-09-01T02:00:00Z",
                        expiresAt="2026-09-01T02:00:00Z",
                    )
                ),
            ),
        )

        for expected_code, compact_jws in cases:
            with self.subTest(expected_code=expected_code):
                with self.assertRaises(CommandValidationError) as raised:
                    self.verifier.verify_for_rejection(compact_jws)
                self.assertEqual(raised.exception.code, expected_code)

    def test_rejects_payload_hash_schema_type_and_missing_capability(self) -> None:
        self._assert_code(
            "INVALID_PAYLOAD_HASH",
            self._sign(self._claims(payloadHash="0" * 64)),
        )
        self._assert_code(
            "UNSUPPORTED_SCHEMA",
            self._sign(self._claims(schemaVersion=2)),
        )
        self._assert_code(
            "UNSUPPORTED_COMMAND",
            self._sign(self._claims(type="FACTORY_RESET")),
        )
        no_capability_verifier = self._verifier(
            Ed25519Keyring({KID: self.raw_public_key}), capabilities=frozenset()
        )
        with self.assertRaises(CommandValidationError) as raised:
            no_capability_verifier.verify(self._sign())
        self.assertEqual(raised.exception.code, "MISSING_CAPABILITY")

    def test_rejects_noncanonical_base64_and_non_object_payload(self) -> None:
        protected_segment, claims_segment, signature_segment = self._sign().split(".")
        self._assert_code(
            "INVALID_JWS",
            f"{protected_segment}=.{claims_segment}.{signature_segment}",
        )

        array_payload = _json_bytes(["not", "an", "object"])
        self._assert_code(
            "INVALID_PAYLOAD",
            self._sign(
                self._claims(
                    payloadBase64=_b64url(array_payload),
                    payloadHash=hashlib.sha256(array_payload).hexdigest(),
                )
            ),
        )

    def test_validates_command_specific_v1_payload_discriminators(self) -> None:
        cases = (
            ({}, {}, "INVALID_PAYLOAD_SCHEMA"),
            (
                {
                    "policyVersion": 1,
                    "mode": "ADAPTIVE",
                    "minBitrateKbps": 256,
                    "maxBitrateKbps": 4000,
                    "initialBitrateKbps": 1000,
                    "congestionSamples": 3,
                    "recoverySamples": 5,
                    "cooldownSeconds": 5,
                },
                {"type": "STREAM_POLICY"},
                None,
            ),
            (
                {
                    "policyVersion": 1,
                    "mode": "FIXED",
                    "targetBitrateKbps": 1200,
                },
                {"type": "STREAM_POLICY"},
                None,
            ),
            (
                {"policyVersion": 1, "mode": "DISABLED"},
                {"type": "STREAM_POLICY"},
                None,
            ),
            (
                {
                    "policyVersion": 1,
                    "mode": "FIXED",
                    "targetBitrateKbps": 1200,
                    "unexpected": True,
                },
                {"type": "STREAM_POLICY"},
                "INVALID_PAYLOAD_SCHEMA",
            ),
            (
                {
                    "policyVersion": 1,
                    "mode": "ADAPTIVE",
                    "minBitrateKbps": 2000,
                    "maxBitrateKbps": 4000,
                    "initialBitrateKbps": 1000,
                    "congestionSamples": 3,
                    "recoverySamples": 5,
                    "cooldownSeconds": 5,
                },
                {"type": "STREAM_POLICY"},
                "INVALID_PAYLOAD_SCHEMA",
            ),
            (
                {"policyVersion": 1, "mode": "AUTO"},
                {"type": "STREAM_POLICY"},
                "INVALID_PAYLOAD_SCHEMA",
            ),
            (
                {"targetVersion": "0.1.113", "bomDigest": "sha256:" + "a" * 64},
                {"type": "AGENT_UPDATE"},
                None,
            ),
            (
                {"targetVersion": "latest", "bomDigest": "sha256:" + "a" * 64},
                {"type": "AGENT_UPDATE"},
                "INVALID_PAYLOAD_SCHEMA",
            ),
        )
        capabilities = frozenset(
            {
                "command.config.apply",
                "command.stream.policy",
                "command.agent.update",
            }
        )
        verifier = self._verifier(
            Ed25519Keyring({KID: self.raw_public_key}),
            capabilities=capabilities,
        )
        for payload, overrides, expected_code in cases:
            with self.subTest(payload=payload, overrides=overrides):
                compact = self._sign(self._claims_with_payload(payload, **overrides))
                if expected_code is None:
                    self.assertEqual(verifier.verify(compact).payload, payload)
                else:
                    with self.assertRaises(CommandValidationError) as raised:
                        verifier.verify(compact)
                    self.assertEqual(raised.exception.code, expected_code)

    def test_config_apply_strict_schema_boundaries_and_label_arrays(self) -> None:
        canonical = {
            "configVersion": 1,
            "activation": "RESTART",
            "model": {
                "pointer": "anomalyclip/prod-v2",
                "digest": "sha256:" + "a" * 64,
            },
            "labels": {
                "inspection": [
                    "normal",
                    "defect",
                    "x" * 100,
                    "Straße",
                    "STRASSE",
                ],
                "anomaly": ["defect"],
            },
            "clip": {
                "enabled": True,
                "preSeconds": 0,
                "postSeconds": 300,
            },
            "video": {
                "width": 160,
                "height": 120,
                "fps": 120,
                "bitrateKbps": 20_000,
            },
        }
        self.assertEqual(
            self.verifier.verify(
                self._sign(self._claims_with_payload(canonical))
            ).payload,
            canonical,
        )

        invalid_payloads = []
        for mutate in (
            lambda value: value.update(unexpected=True),
            lambda value: value.update(configVersion=True),
            lambda value: value.update(activation="LATER"),
            lambda value: value["model"].update(pointer=" untrimmed"),
            lambda value: value["model"].update(pointer="line\nbreak"),
            lambda value: value["model"].update(pointer="/absolute/model"),
            lambda value: value["model"].update(pointer="model//artifact"),
            lambda value: value["model"].update(pointer="model/./artifact"),
            lambda value: value["model"].update(pointer="model/../artifact"),
            lambda value: value["model"].update(pointer="model$PASSWORD"),
            lambda value: value["model"].update(pointer="model${PASSWORD}"),
            lambda value: value["model"].update(pointer="model=value"),
            lambda value: value["model"].update(pointer="x" * 256),
            lambda value: value["model"].update(digest="sha256:" + "A" * 64),
            lambda value: value["clip"].update(preSeconds=61),
            lambda value: value["clip"].update(postSeconds=301),
            lambda value: value["video"].update(width=159),
            lambda value: value["video"].update(height=4321),
            lambda value: value["video"].update(fps=0),
            lambda value: value["video"].update(bitrateKbps=20_001),
            lambda value: value["labels"].update(inspection=[]),
            lambda value: value["labels"].update(inspection=["normal"] * 101),
            lambda value: value["labels"].update(inspection=[" normal"]),
            lambda value: value["labels"].update(inspection=["x" * 101]),
            lambda value: value["labels"].update(inspection=["normal", "NORMAL"]),
            lambda value: value["labels"].update(anomaly="defect"),
        ):
            candidate = copy.deepcopy(canonical)
            mutate(candidate)
            invalid_payloads.append(candidate)
        invalid_payloads.extend(
            [
                {"configVersion": 1, "activation": "IMMEDIATE"},
                {
                    "configVersion": 1,
                    "activation": "RESTART",
                    "labels": {},
                },
            ]
        )

        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(CommandValidationError) as raised:
                    self.verifier.verify(
                        self._sign(self._claims_with_payload(payload))
                    )
                self.assertEqual(raised.exception.code, "INVALID_PAYLOAD_SCHEMA")

    def test_fleet_effect_v2_stream_policy_cross_contract_fixture(self) -> None:
        fixture_path = (
            Path(__file__).resolve().parents[1]
            / "fixtures"
            / "fleet-effect-v2-stream-policy.json"
        )
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        self.assertEqual(fixture["contract"], "fleet-effect-v2")
        verifier = self._verifier(
            Ed25519Keyring({KID: self.raw_public_key}),
            capabilities={"command.stream.policy"},
        )

        for payload in fixture["validPayloads"]:
            with self.subTest(mode=payload["mode"]):
                compact = self._sign(
                    self._claims_with_payload(payload, type="STREAM_POLICY")
                )
                self.assertEqual(verifier.verify(compact).payload, payload)

        legacy_adaptive = {
            "policyVersion": 11,
            "mode": "ADAPTIVE",
            "targetBitrateKbps": 1200,
            "minBitrateKbps": 300,
            "maxBitrateKbps": 3000,
            "hysteresisSamples": 3,
            "cooldownMs": 5000,
        }
        with self.assertRaises(CommandValidationError) as raised:
            verifier.verify(
                self._sign(
                    self._claims_with_payload(
                        legacy_adaptive,
                        type="STREAM_POLICY",
                    )
                )
            )
        self.assertEqual(raised.exception.code, "INVALID_PAYLOAD_SCHEMA")

    def test_fleet_effect_v2_config_apply_cross_contract_fixture(self) -> None:
        fixture_path = (
            Path(__file__).resolve().parents[1]
            / "fixtures"
            / "fleet-effect-v2-config-apply.json"
        )
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        self.assertEqual(fixture["contract"], "fleet-effect-v2")
        self.assertEqual(fixture["commandType"], "CONFIG_APPLY")

        for payload in fixture["validPayloads"]:
            with self.subTest(config_version=payload["configVersion"]):
                compact = self._sign(
                    self._claims_with_payload(payload, type="CONFIG_APPLY")
                )
                self.assertEqual(self.verifier.verify(compact).payload, payload)

    def test_rejects_payload_larger_than_v1_limit(self) -> None:
        payload = {"configVersion": 11, "padding": "x" * (64 * 1024)}
        self._assert_code(
            "COMMAND_TOO_LARGE",
            self._sign(self._claims_with_payload(payload)),
        )


if __name__ == "__main__":
    unittest.main()
