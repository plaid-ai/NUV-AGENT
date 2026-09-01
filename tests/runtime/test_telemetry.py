from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from nuvion_app.runtime.platform_identity import (
    PlatformProbe,
    resolve_platform_identity,
)
from nuvion_app.runtime.config_guard import CURRENT_CONFIG_SCHEMA_VERSION
from nuvion_app.runtime.release_bom import (
    build_release_bom_payload,
    canonical_release_bom_json,
)
from nuvion_app.runtime.telemetry import DEFAULT_CONFIG_SCHEMA, build_runtime_telemetry


class RuntimeTelemetryTest(unittest.TestCase):
    def test_default_telemetry_schema_matches_runtime_config_guard(self) -> None:
        self.assertEqual(DEFAULT_CONFIG_SCHEMA, CURRENT_CONFIG_SCHEMA_VERSION)

    def test_reports_build_config_and_resolved_model_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            metadata_dir = model_dir / "metadata"
            metadata_dir.mkdir()
            (metadata_dir / "server_presign_response.json").write_text(
                json.dumps(
                    {
                        "pointer": "anomalyclip/prod",
                        "resolvedVersion": "v0007",
                        "modelDigest": "sha256:model-bundle-7",
                    }
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
            )

        self.assertEqual(telemetry["agentVersion"], "0.1.113")
        self.assertEqual(telemetry["componentSha"], "0123456789abcdef")
        self.assertEqual(telemetry["configSchema"], "11")
        self.assertEqual(telemetry["modelPointer"], "anomalyclip/prod")
        self.assertEqual(telemetry["modelVersion"], "v0007")
        self.assertEqual(telemetry["modelDigest"], "sha256:model-bundle-7")
        self.assertEqual(telemetry["updaterVersion"], "0.4.2")
        self.assertEqual(telemetry["bomId"], "nuv-agent-0.1.113-macos-arm64")
        self.assertEqual(telemetry["bomDigest"], "sha256:bom-113")
        self.assertEqual(telemetry["artifactDigest"], "sha256:artifact-113")
        self.assertEqual(telemetry["productModel"], "MACOS_DEV")
        self.assertEqual(telemetry["platformProfile"], "macos_dev")
        self.assertEqual(telemetry["identityStatus"], "DEV")
        self.assertEqual(telemetry["capabilities"], sorted(identity.capabilities))
        self.assertEqual(telemetry["bomVerificationStatus"], "UNCONFIGURED")
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
            (metadata_dir / "downloaded_from_server.json").write_text(
                json.dumps(
                    [
                        {"key": "model.onnx", "sha256": "a" * 64},
                        {"key": "labels.json", "sha256": "b" * 64},
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
        self.assertEqual(len(digest), 64)
        self.assertTrue(all(character in "0123456789abcdef" for character in digest))

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
        self.assertEqual(telemetry["bomId"], "nuv-agent-0.1.113-arm64")
        self.assertEqual(telemetry["bomDigest"], payload["bomDigest"])
        self.assertEqual(telemetry["updaterVersion"], "0.5.0")
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


if __name__ == "__main__":
    unittest.main()
