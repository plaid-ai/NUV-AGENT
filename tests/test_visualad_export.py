"""Exporter contract tests; tiny deterministic fixtures are not model evidence."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

SCRIPT = Path(__file__).parents[1] / "packaging/dev/export-visualad-onnx.py"
SPEC = importlib.util.spec_from_file_location("visualad_export", SCRIPT)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)

try:
    import numpy as np
    import onnx
    import torch
    from scipy.ndimage import gaussian_filter
except ImportError:
    torch = None


class ExportSafetyTests(unittest.TestCase):
    def test_output_directory_rejects_relative_or_existing_data_before_imports(self):
        common = [
            "--repo-path",
            "/source",
            "--backbone-path",
            "/base.safetensors",
            "--checkpoint-path",
            "/trained.pth",
            "--parity-image",
            "/image.png",
        ]
        with self.assertRaisesRegex(ValueError, "absolute"):
            exporter.main([*common, "--output-dir", "relative"])
        with TemporaryDirectory() as directory:
            path = Path(directory)
            existing = path / "keep.txt"
            existing.write_text("keep")
            with self.assertRaisesRegex(ValueError, "never overwritten"):
                exporter.main([*common, "--output-dir", directory])
            self.assertEqual(existing.read_text(), "keep")

    def test_manifest_hashes_every_declared_artifact_and_preserves_contract(self):
        runtime = SimpleNamespace(
            SOURCE_COMMIT="commit",
            CHECKPOINT_SHA256="trained",
            BACKBONE_SHA256="base",
            LN_POST_POLICY="official_test_base",
        )
        with TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "visualad.onnx").write_bytes(b"graph")
            (path / "visualad.weights").write_bytes(b"weights")
            manifest = exporter.make_manifest(runtime, path, {"visualad.weights"})
            self.assertEqual(manifest["schemaVersion"], 1)
            self.assertEqual(
                manifest["input"],
                {"name": "image", "shape": [1, 3, 518, 518], "dtype": "float32"},
            )
            self.assertEqual(
                manifest["output"],
                {"name": "patch_maps", "shape": [1, 4, 37, 37], "dtype": "float32"},
            )
            self.assertEqual(manifest["lnPostPolicy"], "official_test_base")
            self.assertEqual(manifest["opset"], 20)
            self.assertEqual(len(manifest["artifacts"]), 2)
            for artifact in manifest["artifacts"]:
                self.assertEqual(
                    artifact["sha256"], exporter.sha256_file(path / artifact["path"])
                )
                self.assertEqual(
                    artifact["size"], (path / artifact["path"]).stat().st_size
                )
            json.dumps(manifest, allow_nan=False)

    def test_actual_parity_image_is_required(self):
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            exporter.parse_args(
                [
                    "--repo-path",
                    "/source",
                    "--backbone-path",
                    "/base.safetensors",
                    "--checkpoint-path",
                    "/trained.pth",
                    "--output-dir",
                    "/output",
                ]
            )


@unittest.skipIf(torch is None, "optional torch/onnx/numpy/scipy are needed")
class ExportNumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    @staticmethod
    def core_fixture():
        # Share one deterministic fixture across 24 slots; never use this fixture
        # for actual export, deployment, or performance/accuracy claims.
        attention = SimpleNamespace(
            embed_dim=1024,
            num_heads=16,
            dropout=0,
            batch_first=False,
            _qkv_same_embed_dim=True,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            in_proj_weight=torch.nn.Parameter(torch.eye(1024).repeat(3, 1)),
            in_proj_bias=torch.nn.Parameter(torch.zeros(3072)),
            out_proj=torch.nn.Identity(),
        )
        source = SimpleNamespace(
            attn_mask=None,
            attn=attention,
            ln_1=torch.nn.Identity(),
            ln_2=torch.nn.Identity(),
            mlp=torch.nn.Identity(),
        )
        visual = SimpleNamespace(
            transformer=SimpleNamespace(resblocks=[source] * 24),
            positional_embedding_frozen=torch.arange(
                577 * 1024, dtype=torch.float32
            ).reshape(577, 1024)
            / (577 * 1024),
            anomaly_pos=torch.ones(1, 1024),
            normal_pos=-torch.ones(1, 1024),
            anomaly_token=torch.ones(1024),
            normal_token=-torch.ones(1024),
            class_embedding=torch.zeros(1024),
            conv1=torch.nn.Identity(),
            ln_pre=torch.nn.Identity(),
            ln_post=torch.nn.Identity(),
        )
        return SimpleNamespace(
            visual=visual,
            transforms=torch.nn.ModuleDict(),
            cross_attention=torch.nn.Identity(),
        )

    def test_explicit_mha_matches_scaled_dot_product_reference_and_shares_weights(self):
        core = self.core_fixture()
        original = core.visual.transformer.resblocks[0].attn
        model = exporter.build_export_model(torch, core)
        attention = model.blocks[0].attention
        self.assertIs(attention.in_proj_weight, original.in_proj_weight)
        self.assertIs(core.visual.transformer.resblocks[0].attn, original)
        x = torch.linspace(-0.1, 0.1, 1372 * 1024).reshape(1, 1372, 1024)
        with torch.inference_mode():
            actual = attention(x)
            q = x.reshape(1, 1372, 16, 64).transpose(1, 2)
            expected = torch.nn.functional.scaled_dot_product_attention(q, q, q)
            expected = expected.transpose(1, 2).reshape(1, 1372, 1024)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_attention_configuration_fail_closed(self):
        core = self.core_fixture()
        core.visual.transformer.resblocks[0].attn.num_heads = 8
        with self.assertRaisesRegex(ValueError, "self-attention"):
            exporter.build_export_model(torch, core)

    def test_position_precompute_matches_official_layout_and_size(self):
        core = self.core_fixture()
        model = exporter.build_export_model(torch, core)
        visual = core.visual
        patch_pos = (
            visual.positional_embedding_frozen[1:]
            .reshape(1, 24, 24, 1024)
            .permute(0, 3, 1, 2)
        )
        patch_pos = torch.nn.functional.interpolate(
            patch_pos, (37, 37), mode="bilinear"
        )
        patch_pos = patch_pos.reshape(1, 1024, 1369).transpose(1, 2)[0]
        expected = torch.cat(
            (
                visual.anomaly_pos,
                visual.normal_pos,
                visual.positional_embedding_frozen[:1],
                patch_pos,
            )
        )
        torch.testing.assert_close(model.position, expected, rtol=0, atol=0)
        self.assertEqual(tuple(model.tokens.shape), (1, 3, 1024))
        self.assertEqual(tuple(model.position.shape), (1372, 1024))

    def test_normalize_keeps_epsilon_outside_square_root(self):
        model = exporter.build_export_model(torch, self.core_fixture())
        value = torch.tensor([[0.0, 0.0], [1e-10, 0.0], [3.0, 4.0]])
        expected = torch.nn.functional.normalize(value, dim=-1, eps=1e-8)
        torch.testing.assert_close(model.normalize(value), expected, rtol=0, atol=0)

    def test_cpu_postprocess_preserves_four_resizes_sum_sigma_and_ceil_topk(self):
        maps = torch.linspace(-2, 2, 4 * 37 * 37).reshape(1, 4, 37, 37)
        actual_score, actual_map = exporter.postprocess(torch, gaussian_filter, maps)
        expected_layers = [
            torch.nn.functional.interpolate(
                maps[:, i : i + 1], (518, 518), mode="bilinear", align_corners=False
            ).squeeze(1)
            for i in range(4)
        ]
        expected = torch.from_numpy(
            gaussian_filter(torch.stack(expected_layers).sum(0)[0].numpy(), sigma=4)
        )
        expected_score = expected.flatten().topk(2684).values.mean().item()
        torch.testing.assert_close(actual_map, expected, rtol=0, atol=0)
        self.assertEqual(actual_score, expected_score)
        with self.assertRaisesRegex(ValueError, "shape"):
            exporter.postprocess(torch, gaussian_filter, maps[:, :1])

    @staticmethod
    def onnx_fixture():
        nodes = []
        previous = "constant"
        for index in range(4):
            output = "patch_maps" if index == 3 else f"gelu{index}"
            nodes.append(
                onnx.helper.make_node("Gelu", [previous], [output], approximate="none")
            )
            previous = output
        tensor = onnx.numpy_helper.from_array(
            np.zeros(exporter.OUTPUT_SHAPE, dtype=np.float32), "constant"
        )
        graph = onnx.helper.make_graph(
            nodes,
            "fixture",
            [
                onnx.helper.make_tensor_value_info(
                    "image", onnx.TensorProto.FLOAT, list(exporter.INPUT_SHAPE)
                )
            ],
            [
                onnx.helper.make_tensor_value_info(
                    "patch_maps", onnx.TensorProto.FLOAT, list(exporter.OUTPUT_SHAPE)
                )
            ],
            [tensor],
        )
        return onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 20)]
        )

    def test_graph_audit_requires_fixed_shape_and_exact_gelu(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "visualad.onnx"
            model = self.onnx_fixture()
            onnx.save(model, path)
            operators, external = exporter.audit_onnx(onnx, path)
            self.assertEqual(operators, {"Gelu": 4})
            self.assertEqual(external, set())
            model.graph.node[0].attribute[0].s = b"tanh"
            onnx.save(model, path)
            with self.assertRaisesRegex(ValueError, "approximation"):
                exporter.audit_onnx(onnx, path)

    def test_graph_audit_rejects_external_path_escape(self):
        model = self.onnx_fixture()
        tensor = model.graph.initializer[0]
        onnx.external_data_helper.set_external_data(tensor, location="../weights.bin")
        tensor.ClearField("raw_data")
        with (
            mock.patch.object(onnx.checker, "check_model"),
            mock.patch.object(onnx, "load", return_value=model),
            self.assertRaisesRegex(ValueError, "basename"),
        ):
            exporter.audit_onnx(onnx, "unused.onnx")

    def test_reference_tensor_is_pickle_free_finite_and_never_overwritten(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            value = np.full(exporter.OUTPUT_SHAPE, -0.2, dtype=np.float32)
            record = exporter.save_patch_reference(np, path, 0, value)
            actual = np.load(path / record["path"], allow_pickle=False)
            np.testing.assert_array_equal(actual, value)
            self.assertEqual(
                record["sha256"], exporter.sha256_file(path / record["path"])
            )
            with self.assertRaises(FileExistsError):
                exporter.save_patch_reference(np, path, 0, value)
            for bad in (value.astype(np.float64), value[:, :1], value * np.nan):
                with self.assertRaisesRegex(ValueError, "finite float32"):
                    exporter.save_patch_reference(np, path, 1, bad)


if __name__ == "__main__":
    unittest.main()
