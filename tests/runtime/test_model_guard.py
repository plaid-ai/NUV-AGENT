from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nuvion_app.runtime import model_guard


class ModelGuardTest(unittest.TestCase):
    def test_resolve_effective_profile_darwin_override(self) -> None:
        with mock.patch.object(model_guard, "_should_use_full_triton_profile", return_value=True):
            with mock.patch.object(model_guard, "_is_darwin", return_value=True):
                with mock.patch.dict(os.environ, {"NUVION_TRITON_MAC_PROFILE": "full", "NUVION_MODEL_PROFILE": "runtime"}):
                    self.assertEqual(model_guard.resolve_effective_profile(), "full")

    def test_resolve_effective_profile_raspberry_pi_defaults_full(self) -> None:
        with mock.patch.object(model_guard, "_should_use_full_triton_profile", return_value=True):
            with mock.patch.object(model_guard, "_is_darwin", return_value=False):
                with mock.patch.dict(os.environ, {"NUVION_MODEL_PROFILE": "runtime", "NUVION_TRITON_RPI_PROFILE": ""}, clear=False):
                    self.assertEqual(model_guard.resolve_effective_profile(), "full")

    def test_resolve_effective_profile_linux_defaults_runtime(self) -> None:
        with mock.patch.object(model_guard, "_should_use_full_triton_profile", return_value=False):
            with mock.patch.dict(os.environ, {"NUVION_TRITON_MAC_PROFILE": "full", "NUVION_MODEL_PROFILE": "runtime"}):
                with mock.patch.dict(
                    os.environ,
                    {
                        "NUVION_MODEL_PROFILE": "full",
                        "NUVION_TRITON_JETSON_PROFILE": "",
                    },
                    clear=False,
                ):
                    self.assertEqual(model_guard.resolve_effective_profile(), "runtime")

    def test_resolve_model_dir_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"NUVION_MODEL_LOCAL_DIR": tmp}):
                self.assertEqual(model_guard.resolve_model_dir("runtime"), Path(tmp).resolve())

    def test_missing_required_files_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = model_guard._missing_required_files(Path(tmp), "runtime")
            self.assertTrue(any(path.endswith("text_features.npy") for path in missing))

    def test_missing_required_files_face_tracking_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                os.environ,
                {
                    "NUVION_ZSAD_BACKEND": "none",
                    "NUVION_FACE_TRACKING_ENABLED": "true",
                    "NUVION_FACE_TRACKING_BACKEND": "triton",
                },
                clear=False,
            ):
                missing = model_guard._missing_required_files(Path(tmp), "runtime")
        self.assertEqual(missing, ["onnx/face_detector.onnx"])

    def test_pull_model_face_tracking_only_uses_server_presign(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                os.environ,
                {
                    "NUVION_ZSAD_BACKEND": "none",
                    "NUVION_FACE_TRACKING_ENABLED": "true",
                    "NUVION_FACE_TRACKING_BACKEND": "triton",
                    "NUVION_DEVICE_USERNAME": "device",
                    "NUVION_DEVICE_PASSWORD": "secret",
                },
                clear=False,
            ):
                with mock.patch.object(model_guard, "_is_jetson_linux", return_value=False):
                    with mock.patch.object(model_guard, "pull_model_from_server") as pull_server:
                        model_guard._pull_model("runtime", Path(tmp))

        _, kwargs = pull_server.call_args
        self.assertIn("face_onnx", kwargs["extra_keys"])

    def test_pull_model_requests_jetson_face_plan_as_optional_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                os.environ,
                {
                    "NUVION_ZSAD_BACKEND": "triton",
                    "NUVION_FACE_TRACKING_ENABLED": "true",
                    "NUVION_FACE_TRACKING_BACKEND": "triton",
                    "NUVION_DEVICE_USERNAME": "device",
                    "NUVION_DEVICE_PASSWORD": "secret",
                },
                clear=False,
            ):
                with mock.patch.object(model_guard, "_is_jetson_linux", return_value=True):
                    with mock.patch.object(model_guard, "pull_model_from_server") as pull_server:
                        model_guard._pull_model("runtime", Path(tmp))

        _, kwargs = pull_server.call_args
        self.assertIn("face_onnx", kwargs["extra_keys"])
        self.assertIn("face_plan", kwargs["extra_keys"])
        self.assertIn("face_triton_config", kwargs["extra_keys"])

    def test_missing_required_files_face_tracking_jetson_requires_face_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                os.environ,
                {
                    "NUVION_ZSAD_BACKEND": "none",
                    "NUVION_FACE_TRACKING_ENABLED": "true",
                    "NUVION_FACE_TRACKING_BACKEND": "triton",
                },
                clear=False,
            ):
                with mock.patch.object(model_guard, "_is_jetson_linux", return_value=True):
                    missing = model_guard._missing_required_files(Path(tmp), "runtime")

        self.assertIn("onnx/face_detector.onnx", missing)
        self.assertIn("triton/model_repository/face_detector/1/model.plan", missing)
        self.assertIn("triton/model_repository/face_detector/config.pbtxt", missing)


if __name__ == "__main__":
    unittest.main()
