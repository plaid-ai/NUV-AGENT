from __future__ import annotations

import sys
import types
import unittest

# triton_client imports cv2 at module import time. Unit tests for config parsing
# should not require OpenCV runtime.
if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.SimpleNamespace(resize=lambda image, size: image)

from nuvion_app.agent.triton_client import _infer_layout_and_size, _parse_model_config


class TritonClientConfigParseTest(unittest.TestCase):
    def test_parse_model_config_nested(self) -> None:
        raw = {"config": {"input": [{"name": "image"}]}}
        self.assertEqual(_parse_model_config(raw), {"input": [{"name": "image"}]})

    def test_parse_model_config_flat(self) -> None:
        raw = {"input": [{"name": "image"}]}
        self.assertEqual(_parse_model_config(raw), raw)

    def test_infer_nchw_from_format_none_channel_first(self) -> None:
        inferred = _infer_layout_and_size([3, 336, 336], "FORMAT_NONE")
        self.assertEqual(inferred, ("NCHW", 336, 336))

    def test_infer_nhwc_from_format_none_channel_last(self) -> None:
        inferred = _infer_layout_and_size([224, 224, 3], "FORMAT_NONE")
        self.assertEqual(inferred, ("NHWC", 224, 224))

    def test_infer_from_declared_format(self) -> None:
        inferred = _infer_layout_and_size([3, 336, 336], "FORMAT_NCHW")
        self.assertEqual(inferred, ("NCHW", 336, 336))


if __name__ == "__main__":
    unittest.main()
