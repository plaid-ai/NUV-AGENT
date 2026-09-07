"""Optional CPU tensor tests; external author sources are never vendored here."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nuvion_app.runtime import visualad

try:
    import numpy as np
    import torch
    from PIL import Image
    from scipy.ndimage import gaussian_filter
except ImportError:
    torch = None


@unittest.skipIf(
    torch is None, "optional numerical tests require torch/numpy/Pillow/scipy"
)
class VisualADTensorTests(unittest.TestCase):
    @staticmethod
    def _checkpoint():
        return {
            "anomaly_token": torch.ones(1024),
            "normal_token": -torch.ones(1024),
            "ln_post_weight": torch.full((1024,), 7.0),
            "ln_post_bias": torch.full((1024,), -2.0),
            "backbone": visualad.BACKBONE,
            "image_size": 518,
            "features_list": [6, 12, 18, 24],
            "epoch": 1,
            "proj": None,
            "token_insert_layer": 0,
            "transform_type": "mlp",
            "use_1024_dim": True,
            "transform_config": dict(visualad._TRANSFORM_CONFIG),
            "cross_attn_config": dict(visualad._CROSS_ATTENTION_CONFIG),
            "layer_transforms": {f"layer_{layer}": {} for layer in visualad.FEATURES},
            "cross_attn": {},
        }

    def test_actual_pinned_checkpoint_schema_and_finite_tensors(self):
        checkpoint = self._checkpoint()
        visualad._validate_checkpoint(torch, checkpoint)
        invalid = (
            {**checkpoint, "backbone": "ViT-B/16"},
            {**checkpoint, "image_size": 336},
            {**checkpoint, "features_list": [6, 12]},
            {**checkpoint, "epoch": True},
            {**checkpoint, "extra": "field"},
            {**checkpoint, "anomaly_token": torch.zeros(768)},
            {**checkpoint, "normal_token": torch.ones(1024).double()},
            {**checkpoint, "ln_post_weight": torch.full((1024,), float("nan"))},
        )
        for case in invalid:
            with self.subTest(keys=list(case)), self.assertRaises(ValueError):
                visualad._validate_checkpoint(torch, case)

    def test_visual_mapping_restores_tokens_but_deliberately_preserves_base_ln(self):
        expected = {
            "ln_post.weight": torch.ones(1024),
            "ln_post.bias": torch.zeros(1024),
            "anomaly_token": torch.zeros(1024),
            "normal_token": torch.zeros(1024),
            "anomaly_pos": torch.zeros((1, 1024)),
            "normal_pos": torch.zeros((1, 1024)),
            "positional_embedding_frozen": torch.zeros((577, 1024)),
        }
        vision = SimpleNamespace(state_dict=lambda: expected)
        position = torch.ones((577, 1024))
        base = {
            "visual.ln_post.weight": torch.ones(1024),
            "visual.ln_post.bias": torch.zeros(1024),
            "visual.positional_embedding": position,
            "visual.proj": torch.zeros((1024, 768)),
        }
        checkpoint = self._checkpoint()
        mapped = visualad._vision_state(torch, vision, base, checkpoint)
        self.assertEqual(set(mapped), set(expected))
        torch.testing.assert_close(mapped["anomaly_token"], checkpoint["anomaly_token"])
        torch.testing.assert_close(mapped["anomaly_pos"], position[:1])
        torch.testing.assert_close(mapped["normal_pos"], position[:1])
        self.assertNotEqual(
            mapped["anomaly_pos"].data_ptr(), mapped["normal_pos"].data_ptr()
        )
        torch.testing.assert_close(
            mapped["ln_post.weight"], base["visual.ln_post.weight"]
        )
        self.assertFalse(
            torch.equal(mapped["ln_post.weight"], checkpoint["ln_post_weight"])
        )
        with self.assertRaisesRegex(ValueError, "keys mismatch"):
            visualad._vision_state(
                torch,
                vision,
                {**base, "visual.unreviewed": torch.tensor(0.0)},
                checkpoint,
            )

    def test_strict_state_rejects_partial_shape_dtype_and_nan_loads(self):
        module = torch.nn.Linear(3, 2)
        state = module.state_dict()
        visualad._strict_state(torch, module, state, "fixture")
        invalid = (
            {"weight": state["weight"]},
            {**state, "weight": torch.zeros(3, 2)},
            {**state, "bias": state["bias"].half()},
            {**state, "bias": torch.tensor([0.0, float("nan")])},
        )
        for value in invalid:
            with self.subTest(keys=list(value)), self.assertRaises(ValueError):
                visualad._strict_state(torch, module, value, "fixture")

    def test_preprocessing_is_square_resize_rgb_and_clip_normalization(self):
        detector = visualad.VisualADAnomalyDetector(False, None, None, None)
        detector._Image = Image
        image = Image.fromarray(np.full((32, 64, 3), 255, dtype=np.uint8))
        tensor = detector._prepare_image(image, torch, np)
        self.assertEqual(tuple(tensor.shape), (1, 3, 518, 518))
        expected = (
            torch.ones(3) - torch.tensor([0.48145466, 0.4578275, 0.40821073])
        ) / torch.tensor([0.26862954, 0.26130258, 0.27577711])
        torch.testing.assert_close(tensor[0, :, 0, 0], expected)
        self.assertEqual(tensor.dtype, torch.float32)


@unittest.skipUnless(
    torch is not None and os.environ.get("NUVION_VISUALAD_TEST_SOURCE"),
    "set NUVION_VISUALAD_TEST_SOURCE to a separately supplied pinned checkout",
)
class VisualADExternalSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = visualad._load_external_modules(
            Path(os.environ["NUVION_VISUALAD_TEST_SOURCE"])
        )

    def test_direct_visual_only_structure_has_expected_299_state_tensors(self):
        with torch.device("meta"):
            vision = self.modules["vision"].VisionTransformer(
                336, 14, 1024, 24, 16, 768
            )
        state = vision.state_dict()
        self.assertEqual(len(state), 299)
        self.assertEqual(tuple(state["positional_embedding_frozen"].shape), (577, 1024))
        self.assertNotIn("proj", state)
        self.assertFalse(any(key.startswith("token_embedding") for key in state))

    def test_source_top_one_percent_uses_ceil_and_keeps_raw_signed_score(self):
        values = torch.zeros((11, 11))
        values[0, 0] = 4
        values[0, 1] = 2
        # ceil(121*.01)=2, not floor=1.
        score = self.modules["scoring"].reduce_anomaly_map(
            values, mode="topk_mean", topk_ratio=0.01
        )
        self.assertEqual(float(score), 3.0)
        self.assertEqual(
            float(self.modules["scoring"].reduce_anomaly_map(-torch.ones((11, 11)))),
            -1.0,
        )

    def test_fused_pipeline_matches_original_map_sum_gaussian_and_topk(self):
        class Vision(torch.nn.Module):
            def forward(self, batch, layers):
                anomaly = torch.zeros((1, 1024))
                anomaly[:, 0] = 1
                patch = torch.zeros((1, 7, 1024))
                patch[:, :, 0] = 1
                patch[:, 3:, 1] = torch.tensor([1.0, 2.0, 3.0, 4.0])
                return {
                    "anomaly_features": anomaly,
                    "normal_features": -anomaly,
                    "patch_tokens": [patch.clone() for _ in layers],
                    "patch_start_idx": 3,
                }

        class Cross(torch.nn.Module):
            def forward(self, anomaly, normal, patches, layers):
                return [{"anomaly": anomaly, "normal": normal} for _ in layers]

        vision = Vision()
        transforms = torch.nn.ModuleDict(
            {f"layer_{layer}": torch.nn.Identity() for layer in visualad.FEATURES}
        )
        filtering = mock.Mock(side_effect=gaussian_filter)
        core = visualad._VisualADCore(
            torch, np, filtering, vision, transforms, Cross(), self.modules
        )
        image = torch.zeros((1, 3, 518, 518))
        score, actual_map = core.predict(image)
        output = vision(image, visualad.FEATURES)
        one_map = self.modules["anomaly_detection"].generate_anomaly_map_from_tokens(
            output["anomaly_features"],
            output["normal_features"],
            output["patch_tokens"][0][:, 3:, :],
            518,
        )
        expected_map = torch.from_numpy(
            gaussian_filter((4 * one_map)[0].numpy(), sigma=4)
        )
        expected_score = self.modules["scoring"].reduce_anomaly_map(
            expected_map, mode="topk_mean", topk_ratio=0.01
        )
        torch.testing.assert_close(actual_map, expected_map)
        self.assertEqual(score, float(expected_score))
        filtering.assert_called_once()
        self.assertEqual(filtering.call_args.kwargs, {"sigma": 4})
        self.assertGreater(score, 1.0)  # raw value, not normalized probability


if __name__ == "__main__":
    unittest.main()
