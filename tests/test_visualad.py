"""Standard-library-only safety and public API tests for VisualAD integration."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nuvion_app.runtime import visualad


def _detector(**overrides):
    return visualad.VisualADAnomalyDetector(
        **{
            "enabled": True,
            "repo_path": "/external/VisualAD",
            "backbone_path": "/models/open_clip_model.safetensors",
            "checkpoint_path": "/models/VisA.pth",
            **overrides,
        }
    )


class _Frame:
    shape = (480, 640, 3)
    dtype = "uint8"


class VisualADConfigTests(unittest.TestCase):
    def test_constructor_does_not_import_optional_dependencies(self):
        with mock.patch.object(visualad.importlib, "import_module") as imports:
            detector = _detector()
        imports.assert_not_called()
        self.assertTrue(detector.enabled)
        self.assertFalse(detector.ready)
        self.assertEqual(detector.load_state, "not_loaded")
        self.assertIsNone(detector.model_sha256)

    def test_disabled_can_omit_all_external_inputs(self):
        detector = _detector(
            enabled=False, repo_path=None, backbone_path=None, checkpoint_path=None
        )
        with mock.patch.object(detector, "_load_runtime") as loader:
            self.assertFalse(detector.warmup())
            self.assertEqual(detector.is_anomaly(_Frame()), (False, None))
        loader.assert_not_called()
        self.assertEqual(detector.load_state, "disabled")

    def test_configuration_rejects_bad_paths_raw_thresholds_and_thread_counts(self):
        cases = (
            {"repo_path": None},
            {"repo_path": "github:7HHHHH/VisualAD"},
            {"backbone_path": "/models/pickle.pt"},
            {"checkpoint_path": "relative.pth"},
            {"threshold": float("nan")},
            {"threshold": float("inf")},
            {"threshold": -8.01},
            {"threshold": 8.01},
            {"threshold": True},
            {"threshold": "0.0"},
            {"num_threads": 0},
            {"num_threads": 5},
            {"num_threads": True},
            {"num_threads": 2.5},
        )
        for case in cases:
            with self.subTest(case=case):
                detector = _detector(**case)
                self.assertTrue(detector.enabled)
                self.assertEqual(detector.load_state, "error")
                self.assertFalse(detector.ready)
                self.assertIsNotNone(detector.last_error)
                self.assertEqual(detector.is_anomaly(_Frame()), (False, None))

    def test_negative_raw_threshold_and_boundaries_are_valid(self):
        for threshold in (-8, -0.2, 0, 8):
            with self.subTest(threshold=threshold):
                detector = _detector(threshold=threshold)
                self.assertEqual(detector.load_state, "not_loaded")
                self.assertEqual(detector.threshold, float(threshold))


class VisualADLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.detector = _detector()
        self.core = mock.Mock()
        self.core.predict.return_value = (-0.3, "raw map")
        self.image = SimpleNamespace(
            new=mock.Mock(return_value="dummy"),
            fromarray=mock.Mock(return_value="image"),
        )
        self.loader = mock.patch.object(
            self.detector,
            "_load_runtime",
            return_value=(object(), object(), self.image, self.core),
        )
        self.prepare = mock.patch.object(
            self.detector, "_prepare_image", return_value="batch"
        )

    def test_negative_score_is_preserved_without_probability_conversion(self):
        with self.loader as loader, self.prepare:
            anomalous, result = self.detector.is_anomaly(_Frame())
            self.assertFalse(anomalous)
            self.assertEqual(result["label"], "normal")
            self.assertEqual(result["score"], -0.3)
            self.assertEqual(result["scores"], [-0.3])
            self.assertEqual(result["labels"], ["anomaly_score"])
            self.assertEqual(result["ln_post_policy"], "official_test_base")
            self.assertEqual(result["image_size"], 518)
            self.assertEqual(result["source_commit"], visualad.SOURCE_COMMIT)
            self.assertEqual(result["model_sha256"], visualad.CHECKPOINT_SHA256)
            self.assertEqual(result["backbone_sha256"], visualad.BACKBONE_SHA256)
            self.assertTrue(self.detector.ready)
            self.assertEqual(self.detector.inference_count, 1)
            self.assertEqual(self.core.predict.call_count, 2)
            self.assertTrue(self.detector.warmup())
            loader.assert_called_once()

    def test_exact_raw_threshold_is_anomalous(self):
        self.core.predict.return_value = (0.0, "raw map")
        with self.loader, self.prepare:
            self.assertTrue(self.detector.is_anomaly(_Frame())[0])

    def test_raw_score_greater_than_one_is_not_clamped(self):
        self.core.predict.return_value = (3.5, "raw map")
        with self.loader, self.prepare:
            anomalous, result = self.detector.is_anomaly(_Frame())
        self.assertTrue(anomalous)
        self.assertEqual(result["score"], 3.5)

    def test_failed_load_does_not_emit_normal_or_repeatedly_retry(self):
        with mock.patch.object(
            self.detector, "_load_runtime", side_effect=ValueError("bad digest")
        ) as loader:
            for _ in range(3):
                self.assertEqual(self.detector.is_anomaly(_Frame()), (False, None))
            loader.assert_called_once()
        self.assertFalse(self.detector.ready)
        self.assertTrue(self.detector.enabled)
        self.assertIsNone(self.detector.model_sha256)
        self.assertIsNone(self.detector.backbone_sha256)
        self.assertIn("bad digest", self.detector.last_error)

    def test_explicit_retry_after_failure_can_recover(self):
        with mock.patch.object(
            self.detector, "_load_runtime", side_effect=OSError("missing file")
        ):
            self.assertFalse(self.detector.warmup())
        with self.loader as loader, self.prepare:
            self.assertFalse(self.detector.warmup())
            loader.assert_not_called()
            self.assertTrue(self.detector.warmup(retry=True))
            self.assertIsNone(self.detector.last_error)

    def test_bad_warmup_score_cannot_attest_loaded_identity(self):
        for score in (float("nan"), float("inf"), -8.1, 8.1):
            with self.subTest(score=score), self.loader, self.prepare:
                self.core.predict.return_value = (score, "map")
                self.assertFalse(self.detector.warmup(retry=True))
                self.assertIsNone(self.detector.model_sha256)
                self.assertFalse(self.detector.ready)

    def test_inference_exception_clears_readiness_and_never_returns_normal(self):
        with self.loader, self.prepare:
            self.assertTrue(self.detector.warmup())
            self.core.predict.side_effect = RuntimeError("tensor failure")
            self.assertEqual(self.detector.is_anomaly(_Frame()), (False, None))
        self.assertFalse(self.detector.ready)
        self.assertIsNone(self.detector._core)
        self.assertIsNone(self.detector.last_anomaly_map)

    def test_runtime_path_changes_require_explicit_reload(self):
        with self.loader, self.prepare:
            self.assertTrue(self.detector.warmup())
            self.detector.repo_path = "/different/source"
            self.assertEqual(self.detector.is_anomaly(_Frame()), (False, None))
        self.assertIn("configuration changed", self.detector.last_error)

    def test_finite_raw_threshold_can_change_without_reloading(self):
        with self.loader as loader, self.prepare:
            self.assertFalse(self.detector.is_anomaly(_Frame())[0])
            self.detector.threshold = -0.5
            self.assertTrue(self.detector.is_anomaly(_Frame())[0])
            loader.assert_called_once()

    def test_bad_input_type_returns_none(self):
        with self.loader, self.prepare:
            frame = SimpleNamespace(shape=(480, 640, 4), dtype="uint8")
            self.assertEqual(self.detector.is_anomaly(frame), (False, None))
        self.assertIn("RGB uint8", self.detector.last_error)


class VisualADArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "test.pth"
        self.content = b"test fixture, not a real checkpoint"
        self.path.write_bytes(self.content)
        self.path.chmod(0o600)
        self.digest = hashlib.sha256(self.content).hexdigest()

    def test_verified_descriptor_contains_exact_artifact(self):
        with visualad._verified_artifact(
            self.path, len(self.content), self.digest
        ) as pinned:
            self.assertNotEqual(pinned, str(self.path))
            self.assertEqual(Path(pinned).read_bytes(), self.content)

    def test_size_hash_and_mutation_fail_closed(self):
        for size, digest in (
            (len(self.content) + 1, self.digest),
            (len(self.content), "0" * 64),
        ):
            with (
                self.subTest(size=size, digest=digest),
                self.assertRaises(ValueError),
                visualad._verified_artifact(self.path, size, digest),
            ):
                self.fail("invalid artifact reached parser")
        with (
            self.assertRaisesRegex(ValueError, "changed"),
            visualad._verified_artifact(self.path, len(self.content), self.digest),
        ):
            self.path.write_bytes(self.content + b"changed")

    def test_symlink_and_writable_inputs_are_rejected(self):
        symlink = Path(self.temp.name) / "link.pth"
        symlink.symlink_to(self.path)
        with (
            self.assertRaisesRegex(ValueError, "regular"),
            visualad._verified_artifact(symlink, len(self.content), self.digest),
        ):
            self.fail("symlink reached parser")
        self.path.chmod(0o666)
        with (
            self.assertRaisesRegex(ValueError, "writable"),
            visualad._verified_artifact(self.path, len(self.content), self.digest),
        ):
            self.fail("writable file reached parser")


class VisualADSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.source = b"ANSWER = 42\n"
        self.path = self.repo / "module.py"
        self.path.write_bytes(self.source)
        self.path.chmod(0o600)
        self.manifest = {
            "fixture": ("module.py", hashlib.sha256(self.source).hexdigest())
        }

    def test_only_explicit_verified_file_runs_without_package_initializers(self):
        (self.repo / "__init__.py").write_text(
            "raise AssertionError('must not execute package')\n"
        )
        with mock.patch.object(visualad, "SOURCE_FILES", self.manifest):
            modules = visualad._load_external_modules(self.repo)
        self.assertEqual(modules["fixture"].ANSWER, 42)

    def test_modified_source_is_rejected_before_exec(self):
        self.path.write_text("raise AssertionError('unreviewed code')\n")
        with (
            mock.patch.object(visualad, "SOURCE_FILES", self.manifest),
            mock.patch("builtins.exec") as execute,
            self.assertRaisesRegex(ValueError, "SHA256"),
        ):
            visualad._load_external_modules(self.repo)
        execute.assert_not_called()

    def test_loader_executes_verified_bytes_not_reopened_path(self):
        self.path.write_text("ANSWER = 'replacement'\n")
        with mock.patch.object(
            visualad,
            "_verified_source_bytes",
            return_value={"fixture": (self.path, self.source)},
        ):
            modules = visualad._load_external_modules(self.repo)
        self.assertEqual(modules["fixture"].ANSWER, 42)

    def test_all_files_are_validated_before_any_exec(self):
        manifest = {**self.manifest, "bad": ("bad.py", "0" * 64)}
        (self.repo / "bad.py").write_text("ANSWER = 0\n")
        with (
            mock.patch.object(visualad, "SOURCE_FILES", manifest),
            mock.patch("builtins.exec") as execute,
            self.assertRaisesRegex(ValueError, "SHA256"),
        ):
            visualad._load_external_modules(self.repo)
        execute.assert_not_called()


class VisualADLoaderTests(unittest.TestCase):
    def test_trained_weights_only_and_safetensors_loading_after_verification(self):
        detector = _detector()
        torch = mock.Mock(__version__="2.10.0")
        torch.nn.ModuleDict.return_value = {}
        checkpoint = {
            "layer_transforms": {f"layer_{layer}": {} for layer in visualad.FEATURES},
            "cross_attn": {},
        }
        torch.load.return_value = checkpoint
        external = {name: mock.Mock() for name in visualad.SOURCE_FILES}
        safetensors = mock.Mock()
        weights = mock.Mock()
        weights.keys.return_value = ["visual.tensor", "unused_text.tensor"]
        safetensors.safe_open.return_value.__enter__ = mock.Mock(return_value=weights)
        safetensors.safe_open.return_value.__exit__ = mock.Mock(return_value=False)
        modules = {
            "torch": torch,
            "numpy": mock.Mock(),
            "PIL.Image": mock.Mock(),
            "scipy.ndimage": mock.Mock(),
            "safetensors": safetensors,
        }
        events = []

        @contextmanager
        def verified(path, size, digest):
            events.append((str(path), size, digest))
            yield "/proc/self/fd/42"

        with (
            mock.patch.object(
                visualad, "_load_external_modules", return_value=external
            ),
            mock.patch.object(
                visualad.importlib, "import_module", side_effect=modules.__getitem__
            ),
            mock.patch.object(visualad, "_verified_artifact", verified),
            mock.patch.object(visualad, "_validate_checkpoint"),
            mock.patch.object(visualad, "_vision_state", return_value={}),
            mock.patch.object(visualad, "_strict_state") as strict,
            mock.patch.object(visualad, "_VisualADCore"),
        ):
            detector._load_runtime()
        self.assertEqual(
            events,
            [
                (
                    detector.checkpoint_path,
                    visualad.CHECKPOINT_SIZE,
                    visualad.CHECKPOINT_SHA256,
                ),
                (
                    detector.backbone_path,
                    visualad.BACKBONE_SIZE,
                    visualad.BACKBONE_SHA256,
                ),
            ],
        )
        torch.load.assert_called_once_with(
            "/proc/self/fd/42", map_location="cpu", weights_only=True
        )
        torch.jit.load.assert_not_called()
        safetensors.safe_open.assert_called_once_with(
            "/proc/self/fd/42", framework="pt", device="cpu"
        )
        weights.get_tensor.assert_called_once_with("visual.tensor")
        self.assertEqual(strict.call_count, 6)  # vision + four MLPs + SCA
        external["vision"].VisionTransformer.assert_called_once_with(
            input_resolution=336,
            patch_size=14,
            width=1024,
            layers=24,
            heads=16,
            output_dim=768,
        )
        torch.set_num_threads.assert_called_once_with(2)

    def test_old_pytorch_cannot_reach_pickle_parser(self):
        torch = mock.Mock(__version__="2.0.0")
        with (
            mock.patch.object(visualad, "_load_external_modules", return_value={}),
            mock.patch.object(visualad.importlib, "import_module", return_value=torch),
        ):
            detector = _detector()
            self.assertFalse(detector.warmup())
        torch.load.assert_not_called()
        self.assertIn("PyTorch >=2.6", detector.last_error)


if __name__ == "__main__":
    unittest.main()
