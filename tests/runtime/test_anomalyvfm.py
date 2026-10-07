from __future__ import annotations

import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock
import numpy as np
from nuvion_app.runtime import anomalyvfm as model


class AnomalyVFMTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.files = {"context": b"wrapper", "context_binary": b"binary"}
        self.manifest = {"schemaVersion": 1, "backend": model.BACKEND, "pointer": model.POINTER,
            "platformProfile": "ventuno_q", "ortVersion": "1.30.0", "qnnVersion": "2.6.0",
            "input": model.INPUT, "outputs": model.OUTPUTS, "postprocess": "sigmoid-zero-pad-avg5-v1",
            "artifacts": {k: {"path": model.FILES[k], "sizeBytes": len(v),
                "sha256": hashlib.sha256(v).hexdigest()} for k, v in self.files.items()}}
        self.env = {"NUVION_ANOMALYVFM_STORE": str(self.root), "NUVION_MODEL_POINTER": model.POINTER,
            "NUVION_MODEL_SERVER_BASE_URL": "https://api.example.test", "NUVION_MODEL_SERVER_ACCESS_TOKEN": "test-only"}
        self.save()

    def save(self):
        self.raw = json.dumps(self.manifest).encode()
        self.env["NUVION_MODEL_DIGEST"] = "sha256:" + hashlib.sha256(self.raw).hexdigest()
        self.selection = model.selection_from_env(self.env)
        self.selection.directory.mkdir(exist_ok=True)
        self.selection.manifest_path.write_bytes(self.raw)
        for key, data in self.files.items():
            (self.selection.directory / model.FILES[key]).write_bytes(data)

    def test_manifest_pins_both_files_and_contract(self):
        model.read_manifest(self.selection)
        (self.selection.directory / model.FILES["context_binary"]).write_bytes(b"tamper")
        with self.assertRaises(ValueError): model.read_manifest(self.selection)
        for key, value in (("platformProfile", "iq9075"), ("ortVersion", "1.26.0"),
                           ("postprocess", "none"), ("input", {})):
            original = self.manifest[key]
            self.manifest[key] = value
            self.save()
            with self.subTest(key=key), self.assertRaises(ValueError): model.read_manifest(self.selection)
            self.manifest[key] = original

    def test_preprocessing_output_zero_padding_and_extreme_logits(self):
        score, heatmap = model.postprocess(np.zeros((1, 1), np.float32), np.zeros((1, 1, 96, 96), np.float32))
        self.assertEqual(score, .5)
        self.assertAlmostEqual(float(heatmap[48, 48]), .5)
        self.assertAlmostEqual(float(heatmap[0, 0]), .18)
        score, heatmap = model.postprocess(np.full((1, 1), 1000, np.float32), np.full((1, 1, 96, 96), -1000, np.float32))
        self.assertEqual(score, 1.)
        self.assertTrue(np.all(heatmap == 0))
        with self.assertRaises(ValueError): model.postprocess(np.full((1, 1), np.nan, np.float32), heatmap)

    def test_unsupported_platform_never_downloads_or_loads(self):
        with mock.patch.object(model.platform, "system", return_value="Darwin"), self.assertRaises(ValueError):
            model.ensure_package(self.selection, self.env)

    def test_download_or_session_load_is_not_live_proof_and_mutations_withdraw_it(self):
        detector = model.AnomalyVFMDetector(self.env)
        self.assertIsNone(detector.loaded_model_proof())
        detector.ready = True
        detector._result_at = time.monotonic()
        detector.execution_provider = "QNNExecutionProvider/HTP"
        detector._loaded_fingerprint = self.selection.fingerprint()
        expected = {"pointer": self.selection.pointer, "digest": self.selection.digest}
        self.assertEqual(detector.verify_model(expected), expected)
        with mock.patch.object(model.time, "monotonic", return_value=detector._result_at + 61):
            self.assertIsNone(detector.loaded_model_proof())
        (self.selection.directory / model.FILES["context"]).write_bytes(b"changed")
        with self.assertRaises(RuntimeError): detector.verify_model(expected)

    def test_symlinks_unpinned_or_cross_platform_selection_rejected(self):
        with self.assertRaises(ValueError): model.selection_from_env(self.env, digest="")
        with self.assertRaises(ValueError): model.selection_from_env(self.env, pointer="anomalyvfm/jetson")
        path = self.selection.directory / model.FILES["context"]
        path.unlink(); path.symlink_to(self.selection.manifest_path)
        with self.assertRaises(ValueError): model.read_manifest(self.selection)

    def test_failed_candidate_download_keeps_current_package(self):
        other = "sha256:" + "a" * 64
        detector = model.AnomalyVFMDetector(self.env)
        with mock.patch.object(model, "require_compatible_platform"), mock.patch(
                "nuvion_app.model_store._fetch_server_presign", return_value={"pointer": model.POINTER, "artifacts": []}):
            desired = {"pointer": model.POINTER, "digest": other}
            self.assertFalse(detector.preflight_model(desired))
            with self.assertRaises(ValueError): detector._preflight_future.result(timeout=2)
            with self.assertRaises(ValueError): detector.preflight_model(desired)
            self.assertIsNone(detector._preflight_future)
            with mock.patch.object(model, "ensure_package", return_value=self.selection):
                self.assertFalse(detector.preflight_model(desired))
                detector._preflight_future.result(timeout=2)
        model.read_manifest(self.selection)
        self.assertFalse((self.root / ("a" * 64)).exists())

    def test_download_rejects_non_storage_redirect_targets(self):
        for url in ("http://storage.googleapis.com/x", "https://evil.test/x", "https://user@storage.googleapis.com/x"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                model._download(url, self.root / "download", "a" * 64, 1)

    def test_factory_requires_boot_guard(self):
        with self.assertRaises(RuntimeError): model.build_anomalyvfm_detector(self.env)

    def test_demo_threshold_never_changes_production_threshold(self):
        env = {**self.env, "NUVION_ZERO_SHOT_THRESHOLD": "0.7", "NUVION_ANOMALYVFM_DEMO_THRESHOLD": "0.35"}
        for mode in ("false", "UNKNOWN", ""):
            detector = model.AnomalyVFMDetector({**env, "NUVION_DEMO_MODE": mode})
            self.assertEqual(detector.threshold, .7)
            self.assertFalse(detector.demo_threshold_applied)
        detector = model.AnomalyVFMDetector({**env, "NUVION_DEMO_MODE": "true"})
        self.assertEqual(detector.threshold, .35)
        self.assertTrue(detector.demo_threshold_applied)
        for invalid in ("nan", "inf", "-0.1", "1.1"):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                model.AnomalyVFMDetector({**env, "NUVION_DEMO_MODE": "true", "NUVION_ANOMALYVFM_DEMO_THRESHOLD": invalid})
