from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from nuvion_app.runtime.telemetry import build_runtime_telemetry


class RuntimeTelemetryTest(unittest.TestCase):
    def test_reports_build_config_and_resolved_model_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            metadata_dir = model_dir / "metadata"
            metadata_dir.mkdir()
            (metadata_dir / "server_presign_response.json").write_text(
                json.dumps({"pointer": "anomalyclip/prod", "resolvedVersion": "v0007"}),
                encoding="utf-8",
            )

            telemetry = build_runtime_telemetry(
                environ={
                    "NUVION_CONFIG_SCHEMA_VERSION": "10",
                    "NUVION_MODEL_POINTER": "anomalyclip/prod",
                },
                model_dir=model_dir,
                agent_version="0.1.113",
                component_sha="0123456789abcdef",
            )

        self.assertEqual(
            telemetry,
            {
                "agentVersion": "0.1.113",
                "componentSha": "0123456789abcdef",
                "configSchema": "10",
                "modelPointer": "anomalyclip/prod",
                "modelVersion": "v0007",
            },
        )

    def test_explicit_model_version_wins_and_corrupt_metadata_is_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            metadata_dir = model_dir / "metadata"
            metadata_dir.mkdir()
            (metadata_dir / "server_presign_response.json").write_text("not-json", encoding="utf-8")

            telemetry = build_runtime_telemetry(
                environ={"NUVION_MODEL_VERSION": "v0099"},
                model_dir=model_dir,
                agent_version="0.1.113",
                component_sha="unknown",
            )

        self.assertEqual(telemetry["modelVersion"], "v0099")


if __name__ == "__main__":
    unittest.main()
