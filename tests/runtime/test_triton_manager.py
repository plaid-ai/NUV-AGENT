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

    def test_resolve_repository_builds_face_detector_onnx_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "onnx").mkdir(parents=True, exist_ok=True)
            (model_dir / "onnx" / "image_encoder_simplified.onnx").write_bytes(b"image")
            (model_dir / "onnx" / "face_detector.onnx").write_bytes(b"face")

            with mock.patch.object(triton_manager, "_should_use_onnx_repository", return_value=True):
                with mock.patch.object(triton_manager, "face_tracking_uses_triton", return_value=True):
                    resolved = triton_manager.resolve_repository_for_runtime(model_dir)

            self.assertTrue((resolved / "face_detector" / "1" / "model.onnx").exists())
            config = (resolved / "face_detector" / "config.pbtxt").read_text()
            self.assertIn('platform: "onnxruntime_onnx"', config)
            self.assertIn('name: "face_detector"', config)

    def test_resolve_repository_jetson_face_detector_falls_back_to_onnx_when_plan_build_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            default_repo = model_dir / "triton" / "model_repository"
            default_repo.mkdir(parents=True, exist_ok=True)
            (model_dir / "onnx").mkdir(parents=True, exist_ok=True)
            (model_dir / "onnx" / "face_detector.onnx").write_bytes(b"face")

            with mock.patch.object(triton_manager, "_should_use_onnx_repository", return_value=False):
                with mock.patch.object(triton_manager, "_is_jetson_linux", return_value=True):
                    with mock.patch.object(triton_manager, "face_tracking_uses_triton", return_value=True):
                        with mock.patch.object(triton_manager, "_build_face_detector_trt_plan", return_value=False):
                            resolved = triton_manager.resolve_repository_for_runtime(model_dir)

            self.assertEqual(resolved, default_repo)
            self.assertTrue((resolved / "face_detector" / "1" / "model.onnx").exists())
            config = (resolved / "face_detector" / "config.pbtxt").read_text()
            self.assertIn('platform: "onnxruntime_onnx"', config)

    def test_resolve_repository_jetson_uses_packaged_face_plan_without_onnx(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            default_repo = model_dir / "triton" / "model_repository"
            (default_repo / "image_encoder" / "1").mkdir(parents=True, exist_ok=True)
            packaged_plan = model_dir / "triton" / "model_repository" / "face_detector" / "1" / "model.plan"
            packaged_plan.parent.mkdir(parents=True, exist_ok=True)
            packaged_plan.write_bytes(b"trt-plan")

            with mock.patch.object(triton_manager, "_should_use_onnx_repository", return_value=False):
                with mock.patch.object(triton_manager, "_is_jetson_linux", return_value=True):
                    with mock.patch.object(triton_manager, "face_tracking_uses_triton", return_value=True):
                        resolved = triton_manager.resolve_repository_for_runtime(model_dir)

            self.assertEqual(resolved, default_repo)
            self.assertTrue((resolved / "face_detector" / "1" / "model.plan").exists())
            config = (resolved / "face_detector" / "config.pbtxt").read_text()
            self.assertIn('platform: "tensorrt_plan"', config)

    def test_ensure_triton_ready_reloads_when_face_model_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            repo = model_dir / "triton" / "model_repository"
            (repo / "image_encoder" / "1").mkdir(parents=True, exist_ok=True)
            (model_dir / "onnx").mkdir(parents=True, exist_ok=True)
            (model_dir / "onnx" / "face_detector.onnx").write_bytes(b"face")

            with mock.patch.dict(
                "os.environ",
                {
                    "NUVION_ZSAD_BACKEND": "triton",
                    "NUVION_FACE_TRACKING_ENABLED": "true",
                    "NUVION_FACE_TRACKING_BACKEND": "triton",
                },
                clear=False,
            ):
                with mock.patch.object(triton_manager, "_should_use_onnx_repository", return_value=False):
                    with mock.patch.object(triton_manager, "_is_jetson_linux", return_value=True):
                        with mock.patch.object(triton_manager, "ensure_docker_ready"):
                            with mock.patch.object(triton_manager, "container_exists", return_value=True):
                                with mock.patch.object(triton_manager, "container_running", return_value=True):
                                    with mock.patch.object(triton_manager, "_health_ready", side_effect=[True, True, True]):
                                        with mock.patch.object(triton_manager, "_required_models_ready", side_effect=[False, False, True]):
                                            with mock.patch.object(triton_manager, "remove_container") as remove_mock:
                                                with mock.patch.object(triton_manager, "run_triton_container") as run_mock:
                                                    with mock.patch.object(triton_manager, "_register_managed_triton_container"):
                                                        with mock.patch.object(triton_manager, "_build_face_detector_trt_plan", return_value=False):
                                                            triton_manager.ensure_triton_ready("run", model_dir)

            remove_mock.assert_called_once_with("triton-nuv")
            run_mock.assert_called_once()

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
