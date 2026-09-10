from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nuvion_app.inference.demo_mvtec import build_slideshow_caps
from nuvion_app.inference.demo_mvtec import build_managed_demo_playlist
from nuvion_app.inference.demo_mvtec import build_stage_dir
from nuvion_app.inference.demo_mvtec import collect_mvtec_demo_images
from nuvion_app.inference.demo_mvtec import ensure_mvtec_category_cached
from nuvion_app.inference.demo_mvtec import infer_mvtec_ground_truth_label
from nuvion_app.inference.demo_mvtec import MANAGED_DEMO_ARCHIVE_SHA256
from nuvion_app.inference.demo_mvtec import MANAGED_DEMO_BASE_URL
from nuvion_app.inference.demo_mvtec import MANAGED_DEMO_PROFILE_DIGEST
from nuvion_app.inference.demo_mvtec import MANAGED_DEMO_PROFILE_ID
from nuvion_app.inference.demo_mvtec import parse_mvtec_categories
from nuvion_app.inference.demo_mvtec import prepare_mvtec_demo_source
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

    def test_collect_mvtec_demo_images_prefers_test_mix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "metal_nut" / "test"
            (root / "good").mkdir(parents=True)
            (root / "scratch").mkdir(parents=True)
            (root / "good" / "000.png").write_bytes(b"good")
            (root / "scratch" / "001.png").write_bytes(b"defect")

            image_paths = collect_mvtec_demo_images(Path(tmp), "metal_nut")

            self.assertEqual(
                [path.relative_to(Path(tmp)).as_posix() for path in image_paths],
                [
                    "metal_nut/test/good/000.png",
                    "metal_nut/test/scratch/001.png",
                ],
            )

    def test_infer_mvtec_ground_truth_label_uses_good_as_normal(self) -> None:
        self.assertEqual(
            infer_mvtec_ground_truth_label(Path("/tmp/capsule/train/good/000.png")),
            "normal",
        )
        self.assertEqual(
            infer_mvtec_ground_truth_label(Path("/tmp/capsule/test/good/001.png")),
            "normal",
        )
        self.assertEqual(
            infer_mvtec_ground_truth_label(Path("/tmp/capsule/test/scratch/002.png")),
            "defect",
        )

    def test_managed_playlist_is_fixed_and_interleaves_four_defects(self) -> None:
        normal = [Path(f"/tmp/metal_nut/test/good/{index:03d}.png") for index in range(25)]
        defect = [Path(f"/tmp/metal_nut/test/scratch/{index:03d}.png") for index in range(7)]

        playlist = build_managed_demo_playlist(normal + defect)

        self.assertEqual(len(playlist), 24)
        self.assertEqual(
            [index for index, path in enumerate(playlist) if "good" not in path.parts],
            [5, 11, 17, 23],
        )
        self.assertEqual(playlist[0], normal[0])
        self.assertEqual(playlist[-1], defect[3])

    def test_managed_profile_pins_archive_url_and_writable_state_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            extracted = root / "extracted"
            for index in range(20):
                image = extracted / "metal_nut" / "test" / "good" / f"{index:03d}.png"
                image.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(f"normal-{index}".encode())
            for index in range(4):
                image = extracted / "metal_nut" / "test" / "defect" / f"{index:03d}.png"
                image.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(f"defect-{index}".encode())

            with (
                mock.patch.dict(
                    "os.environ", {"NUVION_SETTINGS_STATE_DIR": str(root / "settings")}
                ),
                mock.patch(
                    "nuvion_app.inference.demo_mvtec.ensure_mvtec_category_cached",
                    return_value=extracted,
                ) as ensure_cached,
            ):
                source = prepare_mvtec_demo_source(
                    base_url="https://invalid.example",
                    cache_dir="/unwritable/legacy-cache",
                    profile_id=MANAGED_DEMO_PROFILE_ID,
                    profile_digest=MANAGED_DEMO_PROFILE_DIGEST,
                )

            ensure_cached.assert_called_once_with(
                MANAGED_DEMO_BASE_URL,
                root
                / "demo"
                / "mvtec"
                / f"managed-{MANAGED_DEMO_ARCHIVE_SHA256[:16]}",
                "metal_nut",
                expected_archive_sha256=MANAGED_DEMO_ARCHIVE_SHA256,
            )
            self.assertEqual(source.image_count, 24)
            self.assertEqual(source.ground_truth_labels.count("normal"), 20)
            self.assertEqual(source.ground_truth_labels.count("defect"), 4)

    def test_managed_archive_digest_mismatch_removes_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)

            def write_wrong_archive(_url: str, target: Path) -> None:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"not-the-pinned-archive")

            with mock.patch(
                "nuvion_app.inference.demo_mvtec.download_to_path",
                side_effect=write_wrong_archive,
            ):
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    ensure_mvtec_category_cached(
                        MANAGED_DEMO_BASE_URL,
                        cache,
                        "metal_nut",
                        expected_archive_sha256=MANAGED_DEMO_ARCHIVE_SHA256,
                    )

            self.assertFalse((cache / "archives" / "metal_nut.tar.xz").exists())


if __name__ == "__main__":
    unittest.main()
