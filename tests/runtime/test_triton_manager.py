from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nuvion_app.runtime import triton_manager


class TritonManagerTest(unittest.TestCase):
    def tearDown(self) -> None:
        triton_manager._managed_triton_container = None

    def test_resolve_repository_uses_default_on_linux(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "triton" / "model_repository"
            repo.mkdir(parents=True, exist_ok=True)
            with mock.patch.object(triton_manager, "_should_use_onnx_repository", return_value=False):
                resolved = triton_manager.resolve_repository_for_runtime(Path(tmp))
                self.assertEqual(resolved, repo)

    def test_resolve_repository_builds_macos_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "triton" / "model_repository" / "image_encoder").mkdir(parents=True, exist_ok=True)
            (model_dir / "onnx").mkdir(parents=True, exist_ok=True)
            (model_dir / "onnx" / "image_encoder_simplified.onnx").write_bytes(b"onnx")

            with mock.patch.object(triton_manager, "_should_use_onnx_repository", return_value=True):
                resolved = triton_manager.resolve_repository_for_runtime(model_dir)

            self.assertTrue((resolved / "image_encoder" / "1" / "model.onnx").exists())
            self.assertTrue((resolved / "image_encoder" / "config.pbtxt").exists())

    def test_resolve_repository_macos_always_uses_onnx_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            default_repo = model_dir / "triton" / "model_repository" / "image_encoder"
            (default_repo / "1").mkdir(parents=True, exist_ok=True)
            (default_repo / "1" / "model.onnx").write_bytes(b"default-onnx")
            (default_repo / "config.pbtxt").write_text('name: "image_encoder"\nplatform: "tensorrt_plan"\n')
            (model_dir / "onnx").mkdir(parents=True, exist_ok=True)
            (model_dir / "onnx" / "image_encoder_simplified.onnx").write_bytes(b"fallback-onnx")

            with mock.patch.object(triton_manager, "_should_use_onnx_repository", return_value=True):
                resolved = triton_manager.resolve_repository_for_runtime(model_dir)

            self.assertEqual(resolved, model_dir / "triton" / "model_repository_onnx")
            config = (resolved / "image_encoder" / "config.pbtxt").read_text()
            self.assertIn('platform: "onnxruntime_onnx"', config)
            self.assertNotIn('name: "images"', config)

    def test_resolve_repository_uses_onnx_repo_on_raspberry_pi(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "triton" / "model_repository" / "image_encoder" / "1").mkdir(parents=True, exist_ok=True)
            (model_dir / "onnx").mkdir(parents=True, exist_ok=True)
            (model_dir / "onnx" / "image_encoder_simplified.onnx").write_bytes(b"fallback-onnx")

            with mock.patch.object(triton_manager, "_should_use_onnx_repository", return_value=True):
                resolved = triton_manager.resolve_repository_for_runtime(model_dir)

            self.assertEqual(resolved, model_dir / "triton" / "model_repository_onnx")
            self.assertTrue((resolved / "image_encoder" / "1" / "model.onnx").exists())

    def test_cleanup_managed_triton_stops_running_container(self) -> None:
        triton_manager._managed_triton_container = "triton-nuv"
        with mock.patch.object(triton_manager, "container_exists", return_value=True):
            with mock.patch.object(triton_manager, "container_running", return_value=True):
                with mock.patch.object(triton_manager, "stop_container") as stop_mock:
                    triton_manager.cleanup_managed_triton(reason="unit_test")
        stop_mock.assert_called_once_with("triton-nuv")
        self.assertIsNone(triton_manager._managed_triton_container)


if __name__ == "__main__":
    unittest.main()
