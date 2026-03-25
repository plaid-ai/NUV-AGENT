from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nuvion_app.inference.demo_mvtec import build_slideshow_caps
from nuvion_app.inference.demo_mvtec import build_stage_dir
from nuvion_app.inference.demo_mvtec import parse_mvtec_categories
from nuvion_app.inference.demo_mvtec import validate_mvtec_demo_settings


class DemoMvtecTest(unittest.TestCase):
    def test_parse_categories_uses_default_when_empty(self) -> None:
        categories = parse_mvtec_categories("")
        self.assertIn("screw", categories)
        self.assertIn("capsule", categories)

    def test_validate_settings_accepts_http_base_url(self) -> None:
        detail = validate_mvtec_demo_settings(
            base_url="https://storage.googleapis.com/mvtec-dataset/mvtec-ad",
            categories="screw,metal_nut",
            cache_dir="~/.cache/nuvion/demo/mvtec",
        )
        self.assertIn("screw,metal_nut", detail)

    def test_build_stage_dir_creates_sequential_links(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            image_dir.mkdir()
            image_a = image_dir / "010.png"
            image_b = image_dir / "200.png"
            image_a.write_bytes(b"a")
            image_b.write_bytes(b"b")

            stage_dir = build_stage_dir(root, "screw", [image_a, image_b], ".png")
            self.assertTrue((stage_dir / "00000.png").exists())
            self.assertTrue((stage_dir / "00001.png").exists())

    def test_build_slideshow_caps_uses_fractional_rate(self) -> None:
        caps = build_slideshow_caps(".png", 2.0)
        self.assertEqual(caps, "image/png,framerate=1/2")


if __name__ == "__main__":
    unittest.main()
