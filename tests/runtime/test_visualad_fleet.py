from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nuvion_app.runtime import visualad_fleet as fleet
from nuvion_app.runtime import visualad_htp as htp
from nuvion_app.runtime.visualad import (
    BACKBONE_SHA256,
    CHECKPOINT_SHA256,
    LN_POST_POLICY,
    SOURCE_COMMIT,
)


class VisualADFleetTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.owner = mock.patch.object(fleet, "_MODEL_OWNER_UID", os.getuid())
        self.owner.start()
        self.addCleanup(self.owner.stop)
        self.store = self.root / "store"
        self.store.mkdir(mode=0o755)
        graph = b"test-only opaque graph; never executed"
        self.graph_sha = hashlib.sha256(graph).hexdigest()
        manifest = {
            "schemaVersion": 1,
            "sourceCommit": SOURCE_COMMIT,
            "checkpointSha256": CHECKPOINT_SHA256,
            "backboneSha256": BACKBONE_SHA256,
            "lnPostPolicy": LN_POST_POLICY,
            "opset": 20,
            "graph": "visualad.onnx",
            "input": {"name": "image", "shape": [1, 3, 518, 518], "dtype": "float32"},
            "output": {
                "name": "patch_maps",
                "shape": [1, 4, 37, 37],
                "dtype": "float32",
            },
            "artifacts": [
                {"path": "visualad.onnx", "size": len(graph), "sha256": self.graph_sha}
            ],
        }
        self.manifest = json.dumps(manifest).encode()
        self.manifest_sha = hashlib.sha256(self.manifest).hexdigest()
        self.wrapper = {
            "schemaVersion": 1,
            "backend": "visualad_htp",
            "pointer": fleet.VISUALAD_FLEET_POINTER,
            "manifest": {"path": "manifest.json", "sha256": self.manifest_sha},
        }
        self.directory, self.digest = self.write_bundle(self.wrapper, graph)

    def write_bundle(self, wrapper, graph=b"different test graph"):
        raw = json.dumps(wrapper, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(raw).hexdigest()
        target = self.store / digest
        target.mkdir(mode=0o755)
        for name, content in (
            ("fleet-model.json", raw),
            ("manifest.json", self.manifest),
            ("visualad.onnx", graph),
        ):
            path = target / name
            path.write_bytes(content)
            path.chmod(0o644)
        return target, "sha256:" + digest

    def select(self, **kwargs):
        return fleet.select_visualad_fleet_model(
            self.store, fleet.VISUALAD_FLEET_POINTER, self.digest, **kwargs
        )

    def detector(self):
        return fleet.FleetVisualADHTPAnomalyDetector(
            selection=self.select(), state_dir=str(self.root)
        )

    def simulate_success(self, detector):
        # Hardware is deliberately mocked; this proves the publication boundary,
        # not actual model inference, precision, or accuracy.
        def loaded(_detector):
            _detector._verified_graph_sha256 = self.graph_sha

        with mock.patch.object(htp.VisualADHTPAnomalyDetector, "_load", loaded):
            detector._load()
        detector.ready = True
        detector.loaded_manifest_sha256 = self.manifest_sha
        detector.graph_sha256 = self.graph_sha
        detector.execution_provider = "QNNExecutionProvider/HTP"
        detector.inference_count = 1
        with mock.patch.object(
            htp.VisualADHTPAnomalyDetector, "classify", return_value={"score": 0.0}
        ):
            detector.classify(object())

    def test_wrapper_digest_and_original_manifest_are_distinct_verified_identity(self):
        selection = self.select(verify_artifacts=True)
        self.assertEqual(selection.digest, self.digest)
        self.assertNotEqual(selection.digest, "sha256:" + self.manifest_sha)
        self.assertEqual(selection.manifest_sha256, self.manifest_sha)

    def test_wrong_pointer_digest_backend_and_schema_fail_closed(self):
        for pointer in ("siglip/test", "../escape", "visualad/other"):
            with self.subTest(pointer=pointer), self.assertRaises(ValueError):
                fleet.select_visualad_fleet_model(self.store, pointer, self.digest)
        for digest in ("", "sha256:../escape", self.manifest_sha, "sha256:" + "0" * 64):
            with self.subTest(digest=digest), self.assertRaises((ValueError, OSError)):
                fleet.select_visualad_fleet_model(
                    self.store, fleet.VISUALAD_FLEET_POINTER, digest
                )
        for changes in (
            {"backend": "siglip"},
            {"schemaVersion": True},
            {"pointer": "visualad/other"},
            {"extra": "unsupported"},
        ):
            wrapper = {**self.wrapper, **changes}
            _, digest = self.write_bundle(wrapper)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                fleet.select_visualad_fleet_model(
                    self.store, fleet.VISUALAD_FLEET_POINTER, digest
                )

    def test_tampered_graph_fails_full_apply_verification(self):
        (self.directory / "visualad.onnx").write_bytes(b"tampered")
        with self.assertRaises(ValueError):
            self.select(verify_artifacts=True)

    def test_mutable_owner_mode_and_symlink_store_are_rejected(self):
        with (
            mock.patch.object(fleet, "_MODEL_OWNER_UID", os.getuid() + 1),
            self.assertRaises(ValueError),
        ):
            self.select()
        self.directory.chmod(0o777)
        with self.assertRaises(ValueError):
            self.select()
        self.directory.chmod(0o755)
        alias = self.root / "alias"
        alias.symlink_to(self.store, target_is_directory=True)
        with self.assertRaises(ValueError):
            fleet.select_visualad_fleet_model(
                alias, fleet.VISUALAD_FLEET_POINTER, self.digest
            )

    def test_compile_without_result_never_publishes_actual_proof(self):
        detector = self.detector()
        self.assertIsNone(detector.loaded_model_proof())
        self.assertTrue(detector.startup_pending())
        detector._session = object()
        self.assertIsNone(detector.loaded_model_proof())

    def test_fresh_success_exact_identity_and_no_heartbeat_weight_hashing(self):
        detector = self.detector()
        self.simulate_success(detector)
        desired = {"pointer": fleet.VISUALAD_FLEET_POINTER, "digest": self.digest}
        with mock.patch.object(
            fleet,
            "verify_manifest",
            side_effect=AssertionError("heartbeat must not hash graph"),
        ):
            self.assertEqual(detector.loaded_model_proof(), desired)
        self.assertEqual(detector.verify_model(desired), desired)
        # Provisioning another immutable digest must not revoke this model.
        (self.store / "another-candidate").mkdir()
        self.assertEqual(detector.loaded_model_proof(), desired)
        for changes in (
            {"execution_provider": "CPUExecutionProvider"},
            {"ready": False},
            {"_fleet_result_at": 0.0},
            {"loaded_manifest_sha256": "0" * 64},
        ):
            with (
                self.subTest(changes=changes),
                mock.patch.multiple(detector, **changes),
            ):
                self.assertIsNone(detector.loaded_model_proof())

    def test_mutated_loaded_bytes_and_runtime_failure_withdraw_proof(self):
        detector = self.detector()
        self.simulate_success(detector)
        (self.directory / "visualad.onnx").write_bytes(b"tampered")
        self.assertIsNone(detector.loaded_model_proof())
        detector._fail(RuntimeError("test failure"))
        self.assertFalse(detector.startup_pending())
        self.assertIsNone(detector.loaded_model_proof())

    def test_environment_alone_cannot_enable_fleet_detector(self):
        from nuvion_app.runtime import visualad_fleet_start as startup

        with (
            mock.patch.object(startup, "_BOOT_CONTEXT", None),
            self.assertRaises(RuntimeError),
        ):
            fleet.build_visualad_fleet_detector({fleet.STORE_ENV: str(self.store)})

    def test_provision_is_deterministic_preserves_source_and_refuses_overwrite(self):
        before = {
            name: (self.directory / name).read_bytes()
            for name in ("manifest.json", "visualad.onnx")
        }
        first_store = self.root / "provisioned-a"
        second_store = self.root / "provisioned-b"
        first_store.mkdir()
        second_store.mkdir()
        first = fleet.provision_visualad_fleet_model(
            first_store, self.directory / "manifest.json", self.manifest_sha
        )
        second = fleet.provision_visualad_fleet_model(
            second_store, self.directory / "manifest.json", self.manifest_sha
        )
        self.assertEqual(first.digest, second.digest)
        self.assertEqual(first.manifest_sha256, self.manifest_sha)
        for name, expected in before.items():
            self.assertEqual((self.directory / name).read_bytes(), expected)
            self.assertEqual((first.directory / name).read_bytes(), expected)
            self.assertNotEqual(
                (first.directory / name).stat().st_ino,
                (self.directory / name).stat().st_ino,
            )
        with self.assertRaises(FileExistsError):
            fleet.provision_visualad_fleet_model(
                first_store, self.directory / "manifest.json", self.manifest_sha
            )


if __name__ == "__main__":
    unittest.main()
