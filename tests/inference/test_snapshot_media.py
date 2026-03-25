from __future__ import annotations

import io
import importlib.util
import unittest

import numpy as np

PIL_AVAILABLE = importlib.util.find_spec("PIL") is not None
if PIL_AVAILABLE:
    from PIL import Image
    from nuvion_app.inference.snapshot_media import encode_snapshot_jpeg


@unittest.skipUnless(PIL_AVAILABLE, "Pillow is not installed in the local test environment")
class SnapshotMediaTest(unittest.TestCase):
    def test_encode_snapshot_jpeg_returns_valid_jpeg(self) -> None:
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        frame[:, :, 0] = 255

        encoded = encode_snapshot_jpeg(frame, quality=85)

        self.assertTrue(encoded.startswith(b"\xff\xd8"))
        image = Image.open(io.BytesIO(encoded))
        self.assertEqual(image.size, (4, 4))
        self.assertEqual(image.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
