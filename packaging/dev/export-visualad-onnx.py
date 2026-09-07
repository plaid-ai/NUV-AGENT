#!/usr/bin/env python3
"""Export the pinned VisualAD learned graph, without bundling author source.

Development-only CPU/CUDA reference execution is permitted here. The exported
graph's deployment contract is HTP-only; CPU fallback is not a deployment mode.
The external VisualAD source/checkpoint license remains unspecified.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import time
from pathlib import Path

INPUT_SHAPE = (1, 3, 518, 518)
OUTPUT_SHAPE = (1, 4, 37, 37)
OPSET = 20


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_runtime_module():
    # Loading this one local integration module avoids runtime/__init__.py,
    # which imports unrelated product bootstrap/Triton components.
    path = Path(__file__).resolve().parents[2] / "nuvion_app/runtime/visualad.py"
    spec = importlib.util.spec_from_file_location("nuv_visualad_export_runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_export_model(torch, core):
    """Use existing verified modules/parameters, replacing only MHA execution.

    No randomized model is constructed and no original module is mutated.
    Each wrapper owns references to the already strictly loaded parameters.
    """

    class Attention(torch.nn.Module):
        def __init__(self, source):
            super().__init__()
            if (
                source.embed_dim != 1024
                or source.num_heads != 16
                or source.dropout != 0
                or source.batch_first
                or not source._qkv_same_embed_dim
                or source.bias_k is not None
                or source.bias_v is not None
                or source.add_zero_attn
            ):
                raise ValueError("Unexpected VisualAD self-attention configuration")
            self.in_proj_weight = source.in_proj_weight
            self.in_proj_bias = source.in_proj_bias
            self.out_proj = source.out_proj

        def forward(self, x):
            # x: [1,1372,1024]. Explicit rank-3 batched MatMul avoids opaque
            # aten/native-MHA/SDPA operators and rank-5 QKV intermediates.
            projected = torch.nn.functional.linear(
                x, self.in_proj_weight, self.in_proj_bias
            )
            q, k, v = projected.split(1024, dim=-1)
            q = q.reshape(1372, 16, 64).transpose(0, 1)
            k = k.reshape(1372, 16, 64).transpose(0, 1)
            v = v.reshape(1372, 16, 64).transpose(0, 1)
            probabilities = torch.softmax(
                torch.bmm(q, k.transpose(1, 2)) * 0.125, dim=-1
            )
            attended = torch.bmm(probabilities, v)
            attended = attended.transpose(0, 1).reshape(1, 1372, 1024)
            return self.out_proj(attended)

    class Block(torch.nn.Module):
        def __init__(self, source):
            super().__init__()
            if source.attn_mask is not None:
                raise ValueError("VisualAD visual attention unexpectedly has a mask")
            self.attention = Attention(source.attn)
            self.ln_1 = source.ln_1
            self.ln_2 = source.ln_2
            self.mlp = source.mlp

        def forward(self, x):
            x = x + self.attention(self.ln_1(x))
            return x + self.mlp(self.ln_2(x))

    class ExportModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            visual = core.visual
            blocks = visual.transformer.resblocks
            if len(blocks) != 24 or tuple(visual.positional_embedding_frozen.shape) != (
                577,
                1024,
            ):
                raise ValueError("VisualAD export requires the pinned ViT-L/14 tower")
            self.conv = visual.conv1
            self.ln_pre = visual.ln_pre
            self.ln_post = visual.ln_post
            self.blocks = torch.nn.ModuleList(Block(block) for block in blocks)
            self.transforms = core.transforms
            self.cross_attention = core.cross_attention
            # Exact official 24x24 -> 37x37 position interpolation, performed
            # once on immutable verified weights, not per-frame CPU compute.
            with torch.no_grad():
                positions = visual.positional_embedding_frozen
                patches = (
                    positions[1:].reshape(24, 24, 1024).permute(2, 0, 1).unsqueeze(0)
                )
                patches = (
                    torch.nn.functional.interpolate(
                        patches, size=(37, 37), mode="bilinear", align_corners=False
                    )
                    .reshape(1024, 1369)
                    .transpose(0, 1)
                )
                position = torch.cat(
                    (visual.anomaly_pos, visual.normal_pos, positions[:1], patches),
                    dim=0,
                )
                tokens = torch.stack(
                    (visual.anomaly_token, visual.normal_token, visual.class_embedding)
                ).unsqueeze(0)
            self.register_buffer("position", position.detach())
            self.register_buffer("tokens", tokens.detach())

        @staticmethod
        def normalize(x):
            # Avoid ReduceL2 (not in older QNN EP support tables); keep the
            # original epsilon and clamp AFTER sqrt, not inside the sum.
            norm = (x * x).sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
            return x / norm

        def forward(self, image):
            x = self.conv(image).reshape(1, 1024, 1369).transpose(1, 2)
            x = self.ln_pre(torch.cat((self.tokens, x), dim=1) + self.position)
            features = []
            for index, block in enumerate(self.blocks, start=1):
                x = block(x)
                if index in (6, 12, 18, 24):
                    features.append(self.ln_post(x))
            anomaly = self.ln_post(x[:, 0, :])
            normal = self.ln_post(x[:, 1, :])
            adapted = self.cross_attention(
                anomaly,
                normal,
                [patch[:, 3:, :] for patch in features],
                [6, 12, 18, 24],
            )
            maps = []
            for layer, patch, tokens in zip(
                (6, 12, 18, 24), features, adapted, strict=True
            ):
                # Retain both normalizations from the reference: explicit
                # F.normalize(tokens), then cosine_similarity's own norm.
                a = self.normalize(self.normalize(tokens["anomaly"]))
                n = self.normalize(self.normalize(tokens["normal"]))
                transformed = self.transforms[f"layer_{layer}"](
                    patch.reshape(1372, 1024)
                ).reshape(1, 1372, 1024)[:, 3:, :]
                normalized = self.normalize(transformed)
                anomaly_sim = (normalized * a.unsqueeze(1)).sum(dim=-1)
                normal_sim = (normalized * n.unsqueeze(1)).sum(dim=-1)
                maps.append((anomaly_sim - normal_sim).reshape(1, 37, 37))
            return torch.stack(maps, dim=1)

    return ExportModel().eval().requires_grad_(False)


def postprocess(torch, gaussian_filter, patch_maps):
    if tuple(patch_maps.shape) != OUTPUT_SHAPE:
        raise ValueError("VisualAD patch map shape mismatch")
    # Preserve original per-layer resize BEFORE summation, including ordering.
    maps = [
        torch.nn.functional.interpolate(
            patch_maps[:, i : i + 1], (518, 518), mode="bilinear", align_corners=False
        ).squeeze(1)
        for i in range(4)
    ]
    fused = torch.stack(maps).sum(dim=0)
    filtered = torch.from_numpy(gaussian_filter(fused[0].cpu().numpy(), sigma=4))
    score = filtered.flatten().topk(math.ceil(518 * 518 * 0.01)).values.mean()
    return float(score), filtered


def audit_onnx(onnx, path):
    onnx.checker.check_model(str(path), full_check=True)
    model = onnx.load(str(path), load_external_data=False)
    if [(entry.domain, entry.version) for entry in model.opset_import] != [("", OPSET)]:
        raise ValueError("Exported model must use only ONNX opset 20")
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("Exported model IO count mismatch")
    for value, name, shape in (
        (model.graph.input[0], "image", INPUT_SHAPE),
        (model.graph.output[0], "patch_maps", OUTPUT_SHAPE),
    ):
        tensor = value.type.tensor_type
        if value.name != name or tensor.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError("Exported model IO name/dtype mismatch")
        if tuple(d.dim_value for d in tensor.shape.dim) != shape or any(
            d.dim_param for d in tensor.shape.dim
        ):
            raise ValueError("Exported model has dynamic or incorrect IO shape")
    operators = {}
    forbidden = {"Erf", "ReduceL2", "If", "Loop", "Scan", "GridSample", "Dropout"}
    for node in model.graph.node:
        if node.domain not in ("", "ai.onnx") or node.op_type in forbidden:
            raise ValueError(
                f"Unsupported export operator: {node.domain}:{node.op_type}"
            )
        if "attention" in node.op_type.lower():
            raise ValueError("Opaque attention operator survived decomposition")
        if node.op_type == "Gelu":
            for attribute in node.attribute:
                if attribute.name == "approximate" and attribute.s not in (
                    b"",
                    b"none",
                ):
                    raise ValueError("GELU approximation changed")
        operators[node.op_type] = operators.get(node.op_type, 0) + 1
    if operators.get("Gelu") != 4:
        raise ValueError("Expected four exact SAF GELU operators")
    locations = set()
    for tensor in model.graph.initializer:
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            info = {item.key: item.value for item in tensor.external_data}
            location = info.get("location", "")
            if Path(location).name != location or location in ("", ".", ".."):
                raise ValueError("External tensor data must use a local basename")
            locations.add(location)
    return operators, locations


def make_manifest(runtime, output_dir, external_locations):
    paths = ["visualad.onnx", *sorted(external_locations)]
    return {
        "schemaVersion": 1,
        "sourceCommit": runtime.SOURCE_COMMIT,
        "checkpointSha256": runtime.CHECKPOINT_SHA256,
        "backboneSha256": runtime.BACKBONE_SHA256,
        "lnPostPolicy": runtime.LN_POST_POLICY,
        "graph": "visualad.onnx",
        "input": {"name": "image", "shape": list(INPUT_SHAPE), "dtype": "float32"},
        "output": {
            "name": "patch_maps",
            "shape": list(OUTPUT_SHAPE),
            "dtype": "float32",
        },
        "opset": OPSET,
        "artifacts": [
            {
                "path": name,
                "sha256": sha256_file(output_dir / name),
                "size": (output_dir / name).stat().st_size,
            }
            for name in paths
        ],
    }


def save_patch_reference(numpy, output_dir, index, patch_maps):
    if (
        tuple(patch_maps.shape) != OUTPUT_SHAPE
        or str(patch_maps.dtype) != "float32"
        or not numpy.isfinite(patch_maps).all()
    ):
        raise ValueError("Parity reference must be finite float32 [1,4,37,37]")
    filename = f"parity-{index:03d}-patch-maps.npy"
    path = output_dir / filename
    with path.open("xb") as stream:
        numpy.save(stream, patch_maps, allow_pickle=False)
    return {
        "path": filename,
        "sha256": sha256_file(path),
        "shape": list(OUTPUT_SHAPE),
        "dtype": "float32",
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-path", required=True)
    parser.add_argument("--backbone-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parity-image", action="append", required=True)
    parser.add_argument("--num-threads", type=int, choices=range(1, 5), default=2)
    parser.add_argument(
        "--reference-device", choices=("cpu", "cuda:0", "cuda:1"), default="cpu"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        raise ValueError("output-dir must be absolute")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError(
            "output-dir must be new or empty; artifacts are never overwritten"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    # CLI is a standalone development script; do not import product media code.
    import onnx
    import onnxruntime as ort

    visualad = load_runtime_module()

    detector = visualad.VisualADAnomalyDetector(
        True,
        args.repo_path,
        args.backbone_path,
        args.checkpoint_path,
        num_threads=args.num_threads,
    )
    detector._validate_config()
    torch, numpy, image_module, core = detector._load_runtime()
    detector._Image = image_module
    # CUDA is only a faster FP32 development reference. TF32 would silently
    # change the numerical baseline used to validate ONNX CPU execution.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.reference_device)
    for module in (core.visual, core.transforms, core.cross_attention):
        module.to(device)
    model = build_export_model(torch, core)
    parity = []
    batches = []
    with torch.inference_mode():
        for filename in args.parity_image:
            path = Path(filename)
            if not path.is_absolute() or not path.is_file():
                raise ValueError("parity-image must be an existing absolute local file")
            with image_module.open(path) as image:
                if max(image.size) > 8192:
                    raise ValueError("parity-image dimensions exceed 8192")
                batch = detector._prepare_image(image, torch, numpy).to(device)
            started = time.monotonic()
            reference_score, reference_map = core.predict(batch)
            exported_maps = model(batch)
            score, full_map = postprocess(torch, core.gaussian_filter, exported_maps)
            torch.testing.assert_close(full_map, reference_map, rtol=1e-4, atol=1e-4)
            if not math.isfinite(score) or abs(score - reference_score) > 1e-4:
                raise ValueError(
                    "Decomposed attention/raw score failed CPU/CUDA reference parity"
                )
            batches.append(
                (
                    batch.cpu().numpy(),
                    exported_maps.cpu().numpy(),
                    reference_map.numpy(),
                    reference_score,
                )
            )
            parity.append(
                {
                    "imageSha256": sha256_file(path),
                    "referenceScore": reference_score,
                    "decomposedScore": score,
                    "decomposedMapMaxAbsError": float(
                        (full_map - reference_map).abs().max()
                    ),
                    "referenceSeconds": time.monotonic() - started,
                }
            )
            print(json.dumps({"stage": "torch_parity", **parity[-1]}), flush=True)
        # Legacy exporter supports native ONNX Gelu at opset20 without needing
        # onnxscript. There are no JIT archive loads or ATen fallback exports.
        graph_path = output_dir / "visualad.onnx"
        torch.onnx.export(
            model,
            (torch.from_numpy(batches[0][0]).to(device),),
            str(graph_path),
            input_names=["image"],
            output_names=["patch_maps"],
            opset_version=OPSET,
            dynamo=False,
            do_constant_folding=True,
            export_params=True,
            keep_initializers_as_inputs=False,
            dynamic_axes=None,
        )
    print(json.dumps({"stage": "exported", "graph": str(graph_path)}), flush=True)
    operators, locations = audit_onnx(onnx, graph_path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.num_threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(graph_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    for index, (record, (batch, expected, reference_map, reference_score)) in enumerate(
        zip(parity, batches, strict=True)
    ):
        started = time.monotonic()
        actual = session.run(["patch_maps"], {"image": batch})[0]
        numpy.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
        score, full_map = postprocess(
            torch, core.gaussian_filter, torch.from_numpy(actual)
        )
        numpy.testing.assert_allclose(
            full_map.numpy(), reference_map, rtol=1e-4, atol=1e-4
        )
        if abs(score - reference_score) > 1e-4:
            raise ValueError("ONNX raw score failed reference parity")
        record.update(
            {
                "onnxScore": score,
                "onnxMapMaxAbsError": float(
                    numpy.abs(full_map.numpy() - reference_map).max()
                ),
                "onnxSeconds": time.monotonic() - started,
                "onnxPatchMaps": save_patch_reference(numpy, output_dir, index, actual),
            }
        )
        print(json.dumps({"stage": "onnx_parity", **record}), flush=True)
    # Publish a manifest only after both graph audit and actual-weight parity.
    manifest = make_manifest(visualad, output_dir, locations)
    evidence = {
        "schemaVersion": 1,
        "status": "passed",
        "deploymentBackend": "HTP_ONLY_NOT_YET_VALIDATED",
        "exporterSha256": sha256_file(__file__),
        "runtimeSha256": sha256_file(visualad.__file__),
        "torchVersion": torch.__version__,
        "onnxVersion": onnx.__version__,
        "onnxruntimeVersion": ort.__version__,
        "referenceDevice": args.reference_device,
        "referencePurpose": "numerical_conversion_parity_not_ground_truth",
        "operators": operators,
        "parity": parity,
        "sourceFilesSha256": {
            path: digest for path, digest in visualad.SOURCE_FILES.values()
        },
        "postprocess": {
            "resize": "bilinear_half_pixel_518",
            "layerReduction": "sum",
            "gaussianSigma": 4,
            "topK": 2684,
            "scoreRange": [-8, 8],
        },
    }
    (output_dir / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "stage": "complete",
                "manifestSha256": sha256_file(output_dir / "manifest.json"),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    os.umask(0o077)
    raise SystemExit(main())
