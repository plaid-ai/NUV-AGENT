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


if __name__ == "__main__":
    unittest.main()
