from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from nuvion_app.runtime.config_guard import CURRENT_CONFIG_SCHEMA_VERSION
from nuvion_app.runtime.platform_identity import (
    PlatformProbe,
    resolve_platform_identity,
)
from nuvion_app.runtime.release_bom import (
    ReleaseTarget,
    build_release_bom_payload,
    build_release_bom_v2_payload,
    canonical_release_bom_json,
)
from nuvion_app.runtime.telemetry import (
    DEFAULT_CONFIG_SCHEMA,
    build_runtime_telemetry,
    merge_runtime_public_state,
)


class RuntimeTelemetryTest(unittest.TestCase):
    def test_default_telemetry_schema_matches_runtime_config_guard(self) -> None:
        self.assertEqual(DEFAULT_CONFIG_SCHEMA, CURRENT_CONFIG_SCHEMA_VERSION)

    def test_reports_build_config_and_resolved_model_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            metadata_dir = model_dir / "metadata"
            metadata_dir.mkdir()
            manifest = metadata_dir / "gcs_manifest.json"
            manifest.write_bytes(b'{"model":"v0007"}\n')
            manifest_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
            (metadata_dir / "server_presign_response.json").write_text(
                json.dumps(
                    {
                        "pointer": "anomalyclip/prod",
                        "resolvedVersion": "v0007",
                        "modelDigest": f"sha256:{manifest_digest}",
                    }
                ),
                encoding="utf-8",
            )
            (metadata_dir / "downloaded_from_server.json").write_text(
                json.dumps(
                    [
                        {
                            "key": "manifest",
                            "dst": str(manifest),
                            "sha256": manifest_digest,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            identity = resolve_platform_identity(
                environ={},
                identity_path=model_dir / "missing-device-identity.json",
                probe=PlatformProbe(
                    system="Darwin",
                    os_version="15.6",
                    kernel_version="24.6.0",
                    architecture="arm64",
                    hardware_text="Apple MacBook Pro",
                    accelerator_runtime="MPS",
                    gstreamer_version="1.26.0",
                ),
            )

            telemetry = build_runtime_telemetry(
                environ={
                    "NUVION_CONFIG_SCHEMA_VERSION": "11",
                    "NUVION_MODEL_POINTER": "anomalyclip/prod",
                    "NUVION_UPDATER_VERSION": "0.4.2",
                    "NUVION_BOM_ID": "nuv-agent-0.1.113-macos-arm64",
                    "NUVION_BOM_DIGEST": "sha256:bom-113",
                    "NUVION_ARTIFACT_DIGEST": "sha256:artifact-113",
                },
                model_dir=model_dir,
                agent_version="0.1.113",
                component_sha="0123456789abcdef",
                platform_identity=identity,
                effect_capabilities={"command.stream.policy"},
            )

        self.assertEqual(telemetry["agentVersion"], "0.1.113")
        self.assertEqual(telemetry["componentSha"], "0123456789abcdef")
        self.assertEqual(telemetry["configSchema"], "11")
        self.assertEqual(telemetry["modelPointer"], "anomalyclip/prod")
        self.assertEqual(telemetry["modelVersion"], "v0007")
        self.assertEqual(telemetry["modelDigest"], f"sha256:{manifest_digest}")
        self.assertEqual(telemetry["updaterVersion"], "unknown")
        self.assertEqual(telemetry["bomId"], "nuv-agent-0.1.113-macos-arm64")
        self.assertEqual(telemetry["bomDigest"], "sha256:bom-113")
        self.assertEqual(telemetry["artifactDigest"], "sha256:artifact-113")
        self.assertEqual(telemetry["productModel"], "MACOS_DEV")
        self.assertEqual(telemetry["platformProfile"], "macos_dev")
        self.assertEqual(telemetry["identityStatus"], "DEV")
        self.assertEqual(
            telemetry["capabilities"],
            sorted(set(identity.capabilities) | {"command.stream.policy"}),
        )
        self.assertNotIn("command.config.apply", telemetry["capabilities"])
        self.assertNotIn("command.agent.update", telemetry["capabilities"])
        self.assertEqual(telemetry["bomVerificationStatus"], "UNCONFIGURED")
        self.assertEqual(telemetry["functionalHealth"], "FUNCTIONAL_UNHEALTHY")
        self.assertNotIn("runtimeTelemetry", telemetry["runtimeTelemetry"])
        self.assertEqual(telemetry["runtimeTelemetry"]["agentVersion"], "0.1.113")
        self.assertEqual(telemetry["runtimeTelemetry"]["platformProfile"], "macos_dev")

    def test_explicit_model_version_wins_and_corrupt_metadata_is_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            metadata_dir = model_dir / "metadata"
            metadata_dir.mkdir()
            (metadata_dir / "server_presign_response.json").write_text(
                "not-json", encoding="utf-8"
            )

            telemetry = build_runtime_telemetry(
                environ={"NUVION_MODEL_VERSION": "v0099"},
                model_dir=model_dir,
                agent_version="0.1.113",
                component_sha="unknown",
            )

        self.assertEqual(telemetry["modelVersion"], "v0099")

    def test_model_digest_falls_back_to_verified_artifact_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            metadata_dir = model_dir / "metadata"
            metadata_dir.mkdir()
            model = model_dir / "model.onnx"
            labels = model_dir / "labels.json"
            model.write_bytes(b"model-bytes")
            labels.write_bytes(b"labels-bytes")
            model_digest = hashlib.sha256(model.read_bytes()).hexdigest()
            labels_digest = hashlib.sha256(labels.read_bytes()).hexdigest()
            (metadata_dir / "downloaded_from_server.json").write_text(
                json.dumps(
                    [
                        {
                            "key": "model.onnx",
                            "dst": str(model),
                            "sha256": model_digest,
                        },
                        {
                            "key": "labels.json",
                            "dst": str(labels),
                            "sha256": labels_digest,
                        },
                    ]
                ),
                encoding="utf-8",
            )
            identity = resolve_platform_identity(
                environ={},
                identity_path=model_dir / "missing-device-identity.json",
                probe=PlatformProbe(
                    system="Darwin",
                    os_version="15.6",
                    kernel_version="24.6.0",
                    architecture="arm64",
                    hardware_text="Apple MacBook Pro",
                    accelerator_runtime="MPS",
                    gstreamer_version="1.26.0",
                ),
            )

            telemetry = build_runtime_telemetry(
                environ={},
                model_dir=model_dir,
                platform_identity=identity,
            )

        digest = telemetry["modelDigest"]
        self.assertTrue(digest.startswith("sha256:"))
        self.assertEqual(len(digest), 71)
        self.assertTrue(
            all(character in "0123456789abcdef" for character in digest[7:])
        )

    def test_expected_model_digest_cannot_override_observed_artifact_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            metadata_dir = model_dir / "metadata"
            metadata_dir.mkdir()
            manifest = metadata_dir / "gcs_manifest.json"
            manifest.write_bytes(b"actual-manifest")
            actual = hashlib.sha256(manifest.read_bytes()).hexdigest()
            (metadata_dir / "downloaded_from_server.json").write_text(
                json.dumps(
                    [
                        {
                            "key": "manifest",
                            "dst": str(manifest),
                            "sha256": actual,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            telemetry = build_runtime_telemetry(
                environ={"NUVION_MODEL_DIGEST": "sha256:" + "f" * 64},
                model_dir=model_dir,
            )

        self.assertEqual(telemetry["modelDigest"], "sha256:" + actual)
        self.assertEqual(telemetry["modelExpectedDigest"], "sha256:" + "f" * 64)
        self.assertIs(telemetry["modelDigestMatchesExpected"], False)

    def test_verified_bom_sidecar_is_authoritative_and_runtime_bound(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "nuv-agent_0.1.113_arm64.deb"
            artifact.write_bytes(b"agent artifact")
            payload = build_release_bom_payload(
                bom_id="nuv-agent-0.1.113-arm64",
                agent_version="0.1.113",
                component_sha="a" * 40,
                config_schema="11",
                updater_version="0.5.0",
                platform_profiles=["macos_dev"],
                artifact_path=artifact,
                artifact_kind="deb",
                built_at="2026-09-01T03:00:00Z",
            )
            bom_path = root / "release-bom.json"
            bom_path.write_text(canonical_release_bom_json(payload), encoding="utf-8")
            identity = resolve_platform_identity(
                environ={},
                identity_path=root / "missing-device-identity.json",
                probe=PlatformProbe(
                    system="Darwin",
                    os_version="15.6",
                    kernel_version="24.6.0",
                    architecture="arm64",
                    hardware_text="Apple MacBook Pro",
                    accelerator_runtime="MPS",
                    gstreamer_version="1.26.0",
                ),
            )

            telemetry = build_runtime_telemetry(
                environ={
                    "NUVION_RELEASE_BOM_PATH": str(bom_path),
                    "NUVION_EXPECTED_BOM_DIGEST": payload["bomDigest"],
                    "NUVION_CONFIG_SCHEMA_VERSION": "11",
                },
                model_dir=root,
                agent_version="0.1.113",
                component_sha="a" * 40,
                platform_identity=identity,
            )

        self.assertEqual(telemetry["bomVerificationStatus"], "VERIFIED")
        self.assertNotIn("releaseSequence", telemetry)
        self.assertEqual(telemetry["bomId"], "nuv-agent-0.1.113-arm64")
        self.assertEqual(telemetry["bomDigest"], payload["bomDigest"])
        self.assertEqual(telemetry["updaterVersion"], "unknown")
        self.assertEqual(
            telemetry["runtimeTelemetry"]["bomVerificationStatus"], "VERIFIED"
        )

    def test_bom_tampering_is_reported_without_trusting_its_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bom_path = root / "release-bom.json"
            bom_path.write_text('{"schemaVersion":1}', encoding="utf-8")

            telemetry = build_runtime_telemetry(
                environ={"NUVION_RELEASE_BOM_PATH": str(bom_path)},
                model_dir=root,
                agent_version="0.1.113",
                component_sha="a" * 40,
            )

        self.assertEqual(telemetry["bomVerificationStatus"], "INVALID")
        self.assertEqual(telemetry["bomId"], "unknown")
        self.assertIn("bomVerificationError", telemetry)

    def test_updater_public_state_hook_requires_canonical_rollback_evidence(self) -> None:
        public_state = {
            "functionalHealth": "FUNCTIONAL_HEALTHY",
            "updatePhase": "ROLLED_BACK",
            "targetVersion": "0.1.120",
            "updateEvidence": {
                "rolledBackToVersion": "0.1.119",
                "reason": "health gate failed",
            },
        }

        merged = merge_runtime_public_state({}, public_state)

        self.assertEqual(merged, public_state)
        with self.assertRaises(ValueError):
            merge_runtime_public_state(
                {},
                {
                    "functionalHealth": "FUNCTIONAL_HEALTHY",
                    "updatePhase": "ROLLED_BACK",
                },
            )
        with self.assertRaises(ValueError):
            merge_runtime_public_state(
                {},
                {"functionalHealth": "HEALTHY"},
            )

    def test_command_agent_update_is_not_advertised_without_registered_effect(self) -> None:
        telemetry = build_runtime_telemetry(
            environ={},
            effect_capabilities=frozenset(),
        )

        self.assertNotIn("command.agent.update", telemetry["capabilities"])

    def test_active_v2_slot_reports_integer_release_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "agent-bundle.tar.gz"
            artifact.write_bytes(b"bundle")
            payload = build_release_bom_v2_payload(
                bom_id="nuv-agent-0.1.116-iq9075-aarch64",
                release_sequence=116,
                agent_version="0.1.116",
                component_sha="a" * 40,
                config_schema="12",
                min_updater_version="0.1.0",
                targets=[
                    ReleaseTarget(
                        product_model="IQ9075_DEV",
                        platform_profile="iq9075_dev",
                        hardware_revision="QCS9075-EVK",
                        architecture="aarch64",
                    )
                ],
                artifact_path=artifact,
                artifact_kind="agent-bundle",
                built_at="2026-09-01T10:00:00Z",
            )
            bom_path = root / "release-bom.json"
            bom_path.write_text(canonical_release_bom_json(payload), encoding="utf-8")
            identity = resolve_platform_identity(
                environ={},
                identity_path=root / "device-identity.json",
                probe=PlatformProbe(
                    system="Linux",
                    os_version="24.04",
                    kernel_version="6.8.0",
                    architecture="aarch64",
                    hardware_text="QCS9075 IQ-9075",
                    accelerator_runtime="unknown",
                    gstreamer_version="1.24.2",
                ),
            )

            telemetry = build_runtime_telemetry(
                environ={
                    "NUVION_RELEASE_BOM_PATH": str(bom_path),
                    "NUVION_EXPECTED_BOM_DIGEST": str(payload["bomDigest"]),
                    "NUVION_CONFIG_SCHEMA_VERSION": "12",
                },
                model_dir=root,
                agent_version="0.1.116",
                component_sha="a" * 40,
                platform_identity=identity,
            )

        self.assertEqual(telemetry["releaseSequence"], 116)
        self.assertIsInstance(telemetry["releaseSequence"], int)
        self.assertEqual(telemetry["bomVerificationStatus"], "VERIFIED")
        self.assertEqual(telemetry["runtimeTelemetry"]["releaseSequence"], 116)


if __name__ == "__main__":
    unittest.main()
