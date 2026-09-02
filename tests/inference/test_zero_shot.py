from __future__ import annotations

import os
import platform
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from nuvion_app.inference.zero_shot import ZeroShotAnomalyDetector


class _FakeMPS:
    def __init__(self, available: bool):
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeBackends:
    def __init__(self, mps_available: bool):
        self.mps = _FakeMPS(mps_available)


class _FakeCUDA:
    def __init__(self, available: bool):
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeTorch:
    def __init__(self, mps_available: bool, cuda_available: bool):
        self.backends = _FakeBackends(mps_available)
        self.cuda = _FakeCUDA(cuda_available)


class ZeroShotDeviceResolveTest(unittest.TestCase):
    def test_auto_prefers_mps(self) -> None:
        fake_torch = _FakeTorch(mps_available=True, cuda_available=True)
        device = ZeroShotAnomalyDetector._resolve_device(fake_torch, "auto")
        self.assertEqual(device, "mps")

    def test_auto_uses_cuda_when_mps_unavailable(self) -> None:
        fake_torch = _FakeTorch(mps_available=False, cuda_available=True)
        device = ZeroShotAnomalyDetector._resolve_device(fake_torch, "auto")
        self.assertEqual(device, "cuda")

    def test_explicit_mps_falls_back_to_cpu(self) -> None:
        fake_torch = _FakeTorch(mps_available=False, cuda_available=True)
        device = ZeroShotAnomalyDetector._resolve_device(fake_torch, "mps")
        self.assertEqual(device, "cpu")


@unittest.skipUnless(
    os.environ.get("NUVION_ZSAD_REGRESSION_MODEL"),
    "set NUVION_ZSAD_REGRESSION_MODEL to an authenticated local model snapshot",
)
class MacMpsZeroShotRegressionTest(unittest.TestCase):
    def test_offline_local_siglip_model_classifies_frame_on_mps(self) -> None:
        import torch

        self.assertEqual(platform.system(), "Darwin")
        self.assertEqual(platform.machine(), "arm64")
        model_path = Path(os.environ["NUVION_ZSAD_REGRESSION_MODEL"]).resolve()
        self.assertTrue(model_path.is_dir())
        with mock.patch.dict(
            os.environ,
            {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            clear=False,
        ):
            detector = ZeroShotAnomalyDetector(
                enabled=True,
                model_name=str(model_path),
                labels=["normal scene", "anomalous scene"],
                anomaly_labels=["anomalous scene"],
                threshold=0.5,
                device_preference="mps",
            )
            self.assertTrue(detector.ready)
            self.assertEqual(detector._device, "mps")
            self.assertEqual(detector._inference_dtype, torch.float16)
            self.assertEqual(next(detector._model.parameters()).dtype, torch.float16)
            self.assertEqual(detector.loaded_model_source(), str(model_path))
            result = detector.classify(np.zeros((224, 224, 3), dtype=np.uint8))
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(len(result["labels"]), 2)
        self.assertEqual(len(result["scores"]), 2)


if __name__ == "__main__":
    unittest.main()
