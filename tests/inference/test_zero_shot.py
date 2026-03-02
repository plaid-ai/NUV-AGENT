from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
