from __future__ import annotations

import os
import unittest
from unittest import mock

from nuvion_app.runtime import inference_mode


class InferenceModeTest(unittest.TestCase):
    def test_normalize_backend_supports_mps_alias(self) -> None:
        self.assertEqual(inference_mode.normalize_backend("mps"), "siglip")
        self.assertEqual(inference_mode.normalize_backend("triton"), "triton")

    def test_normalize_backend_fallbacks_to_default(self) -> None:
        self.assertEqual(inference_mode.normalize_backend("unknown", default="triton"), "triton")

    def test_apply_defaults_maps_mps_to_siglip_and_pins_device(self) -> None:
        with mock.patch.dict(os.environ, {"NUVION_ZSAD_BACKEND": "mps"}, clear=False):
            inference_mode.apply_inference_runtime_defaults()
            self.assertEqual(os.getenv("NUVION_ZSAD_BACKEND"), "siglip")
            self.assertEqual(os.getenv("NUVION_ZERO_SHOT_DEVICE"), "mps")

    def test_apply_defaults_sanitizes_siglip_device(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"NUVION_ZSAD_BACKEND": "siglip", "NUVION_ZERO_SHOT_DEVICE": "invalid"},
            clear=False,
        ):
            inference_mode.apply_inference_runtime_defaults()
            self.assertEqual(os.getenv("NUVION_ZERO_SHOT_DEVICE"), "auto")

    def test_normalize_face_tracking_backend_falls_back_to_auto(self) -> None:
        self.assertEqual(inference_mode.normalize_face_tracking_backend("triton"), "triton")
        self.assertEqual(inference_mode.normalize_face_tracking_backend("weird"), "auto")

    def test_face_tracking_uses_triton_for_auto_and_triton(self) -> None:
        self.assertTrue(inference_mode.face_tracking_uses_triton(enabled=True, backend="auto"))
        self.assertTrue(inference_mode.face_tracking_uses_triton(enabled=True, backend="triton"))
        self.assertFalse(inference_mode.face_tracking_uses_triton(enabled=True, backend="opencv"))
        self.assertFalse(inference_mode.face_tracking_uses_triton(enabled=False, backend="triton"))


if __name__ == "__main__":
    unittest.main()
