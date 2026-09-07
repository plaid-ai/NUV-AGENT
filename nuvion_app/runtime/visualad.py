"""Offline CPU integration for an externally provided VisualAD checkout.

No VisualAD source is vendored or downloaded by this module. The author's code
and checkpoint license is unspecified at the pinned revision; this integration
does not confer redistribution/commercial rights. Source files must be supplied
separately and match the reviewed byte hashes before any of them is executed.

Reference: https://github.com/7HHHHH/VisualAD (97eb5f88a44f27c644ea7ed4ecac5e35ddef18bb).
Official-test compatibility deliberately retains BASE CLIP ln_post even though
the trained checkpoint contains different ln_post tensors. Those saved tensors
are validated but not applied, matching test.py rather than fixing it silently.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import logging
import math
import os
import stat
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger(__name__)

SOURCE_COMMIT = "97eb5f88a44f27c644ea7ed4ecac5e35ddef18bb"
SOURCE_FILES = {
    "vision": (
        "VisualAD_lib/VisualAD.py",
        "19906a5bf0aa35401deffc399f1db50fb6ca04fabf1661e51c272ffea89dbca3",
    ),
    "cross_attention": (
        "utils/spatial_cross_attention.py",
        "d67274a68294c5b55a1ad13f2eb4be283d2bb20832e506c840c1b433d9910582",
    ),
    "feature_transform": (
        "utils/feature_transform.py",
        "8f60b6b305d40eee5e890f3c0b7a44a51f9660da75dc1a901d38e5cf7ca35a1f",
    ),
    "scoring": (
        "utils/scoring.py",
        "11b35f555a83fffbcb23f265beaa529a9570358610a60ee3d2a1bbb84855c8ad",
    ),
    "anomaly_detection": (
        "utils/anomaly_detection.py",
        "ed9180c9aea08b04cd5166cb9dd4c4a3e7101f8f1d5b176fe1cfe66cfc765855",
    ),
}
BACKBONE = "ViT-L/14@336px"
BACKBONE_REPOSITORY = "timm/vit_large_patch14_clip_336.openai"
BACKBONE_REVISION = "81e38efc4637de5023b10e75a7f9bd1c6fa6b010"
BACKBONE_SHA256 = "fbc415c3d0d7b79faed8f5ccfb740c32b7c4f5ffe7283b851f89c6231c01a8e0"
BACKBONE_SIZE = 1_711_828_444
CHECKPOINT_SHA256 = "fed8ed5e0973e9adb53a2c91e066eb5ba3f9a42fbf80f115a26998629d1f0a5b"
CHECKPOINT_SIZE = 126_119_090
IMAGE_SIZE = 518
FEATURES = (6, 12, 18, 24)
FEATURE_DIM = 1024
LN_POST_POLICY = "official_test_base"
RAW_SCORE_LIMIT = 8.0
_TRANSFORM_CONFIG = {"dropout": 0.1, "mlp_hidden_ratio": 1.0}
_CROSS_ATTENTION_CONFIG = {
    "apply_to_layer24": True,
    "dropout": 0.1,
    "num_anchors": 4,
    "res_scale_init": 0.01,
}
_CHECKPOINT_KEYS = {
    "anomaly_token",
    "normal_token",
    "ln_post_weight",
    "ln_post_bias",
    "backbone",
    "cross_attn",
    "cross_attn_config",
    "epoch",
    "features_list",
    "image_size",
    "layer_transforms",
    "proj",
    "token_insert_layer",
    "transform_config",
    "transform_type",
    "use_1024_dim",
}


def _identity(metadata):
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


@contextmanager
def _opened_regular_file(path: Path):
    if not path.is_absolute():
        raise ValueError("VisualAD inputs must use absolute local paths")
    original = path.lstat()
    if not stat.S_ISREG(original.st_mode) or original.st_mode & 0o022:
        raise ValueError(
            "VisualAD input must be a regular, non-symlink, non-group/world-writable file"
        )
    flags = (
        os.O_RDONLY
        | os.O_NONBLOCK
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        identity = _identity(opened)
        if not stat.S_ISREG(opened.st_mode) or identity != _identity(original):
            raise ValueError("VisualAD input changed while opening")
        yield descriptor, opened
        if _identity(os.fstat(descriptor)) != identity:
            raise ValueError("VisualAD input changed while reading/loading")
    finally:
        os.close(descriptor)


@contextmanager
def _verified_artifact(path: Path, size: int, sha256: str):
    """Use one inode for digest validation and safe parsing; no path reopen race.

    Root/owner control of the model directory remains a trust assumption. This
    rejects normal in-place changes; it is not an OTA signature or protection
    against a compromised privileged file owner.
    """
    with _opened_regular_file(path) as (descriptor, metadata):
        if metadata.st_size != size:
            raise ValueError("VisualAD artifact size differs from the pinned release")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        if digest.hexdigest() != sha256:
            raise ValueError("VisualAD artifact SHA256 differs from the pinned release")
        if _identity(os.fstat(descriptor)) != _identity(metadata):
            raise ValueError("VisualAD artifact changed while hashing")
        os.lseek(descriptor, 0, os.SEEK_SET)
        if sys.platform.startswith("linux"):
            pinned_path = f"/proc/self/fd/{descriptor}"
        elif sys.platform == "darwin":
            pinned_path = f"/dev/fd/{descriptor}"
        else:
            raise RuntimeError(
                "VisualAD descriptor-pinned loading is unsupported on this OS"
            )
        yield pinned_path


def _verified_source_bytes(repo_path: Path) -> dict:
    if not repo_path.is_absolute() or not repo_path.is_dir():
        raise ValueError("VisualAD repo_path must be an existing absolute directory")
    root = repo_path.resolve(strict=True)
    sources = {}
    # Validate all five files BEFORE executing any external source.
    for name, (relative_path, expected_hash) in SOURCE_FILES.items():
        path = repo_path / relative_path
        path.resolve(strict=True).relative_to(root)
        with _opened_regular_file(path) as (descriptor, metadata):
            if not 1 <= metadata.st_size <= 64 * 1024:
                raise ValueError("VisualAD source size is invalid")
            chunks = []
            while chunk := os.read(descriptor, 8192):
                chunks.append(chunk)
                if sum(map(len, chunks)) > 64 * 1024:
                    raise ValueError("VisualAD source exceeds the size limit")
            source = b"".join(chunks)
            if hashlib.sha256(source).hexdigest() != expected_hash:
                raise ValueError(f"VisualAD source SHA256 mismatch: {relative_path}")
            sources[name] = (path, source)
    return sources


def _load_external_modules(repo_path: Path) -> dict:
    modules = {}
    for name, (path, source) in _verified_source_bytes(repo_path).items():
        module_name = f"_nuv_visualad_{SOURCE_COMMIT}_{name}"
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None:
            raise RuntimeError("Unable to construct VisualAD external module spec")
        module = importlib.util.module_from_spec(spec)
        # Compile the EXACT verified bytes; spec.loader.exec_module would read
        # the pathname a second time. Never execute package __init__ files.
        code = compile(source, str(path), "exec", dont_inherit=True)
        exec(code, module.__dict__)  # noqa: S102 - reviewed, hash-pinned source.
        modules[name] = module
    return modules


def _validate_tensor(torch, tensor, shape, name):
    if (
        not isinstance(tensor, torch.Tensor)
        or tuple(tensor.shape) != tuple(shape)
        or tensor.dtype != torch.float32
    ):
        raise ValueError(f"VisualAD tensor shape/dtype mismatch: {name}")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"VisualAD tensor is not finite: {name}")


def _strict_state(torch, module, state, name):
    if not isinstance(state, dict):
        raise TypeError(f"VisualAD state is not a tensor dictionary: {name}")
    expected = module.state_dict()
    if set(state) != set(expected):
        raise ValueError(f"VisualAD state keys mismatch: {name}")
    for key, target in expected.items():
        _validate_tensor(torch, state[key], target.shape, f"{name}.{key}")
    module.load_state_dict(state, strict=True)


def _validate_checkpoint(torch, checkpoint):
    if not isinstance(checkpoint, dict) or set(checkpoint) != _CHECKPOINT_KEYS:
        raise ValueError(
            "VisualAD trained checkpoint schema differs from the pinned VisA checkpoint"
        )
    expected_metadata = {
        "backbone": BACKBONE,
        "features_list": list(FEATURES),
        "image_size": IMAGE_SIZE,
        "epoch": 1,
        "proj": None,
        "token_insert_layer": 0,
        "transform_type": "mlp",
        "use_1024_dim": True,
        "transform_config": _TRANSFORM_CONFIG,
        "cross_attn_config": _CROSS_ATTENTION_CONFIG,
    }
    # Equality alone accepts True==1; metadata types are part of the contract.
    for name, expected in expected_metadata.items():
        actual = checkpoint[name]
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"VisualAD checkpoint metadata mismatch: {name}")
    if any(type(layer) is not int for layer in checkpoint["features_list"]):
        raise ValueError("VisualAD checkpoint layer IDs must be integers")
    for name in ("anomaly_token", "normal_token", "ln_post_weight", "ln_post_bias"):
        _validate_tensor(torch, checkpoint[name], (FEATURE_DIM,), name)
    if not isinstance(checkpoint["layer_transforms"], dict) or set(
        checkpoint["layer_transforms"]
    ) != {f"layer_{layer}" for layer in FEATURES}:
        raise ValueError("VisualAD checkpoint is missing a trained feature layer")


def _vision_state(torch, visual, base_state, checkpoint):
    """Explicitly map only CLIP's visual tower; no unused text tower allocated."""
    expected = visual.state_dict()
    custom = {
        "anomaly_token",
        "normal_token",
        "anomaly_pos",
        "normal_pos",
        "positional_embedding_frozen",
    }
    required = {"visual." + key for key in expected if key not in custom}
    required |= {"visual.positional_embedding", "visual.proj"}
    if set(base_state) != required:
        raise ValueError("VisualAD base visual tensor keys mismatch")
    state = {key: base_state["visual." + key] for key in expected if key not in custom}
    position = base_state["visual.positional_embedding"]
    _validate_tensor(torch, position, (577, FEATURE_DIM), "visual.positional_embedding")
    _validate_tensor(
        torch,
        base_state["visual.proj"],
        (FEATURE_DIM, 768),
        "visual.proj (explicitly unused)",
    )
    state.update(
        {
            "positional_embedding_frozen": position,
            "anomaly_pos": position[0:1].clone(),
            "normal_pos": position[0:1].clone(),
            "anomaly_token": checkpoint["anomaly_token"],
            "normal_token": checkpoint["normal_token"],
        }
    )
    # BASE ln_post deliberately remains: official test.py does not restore the
    # trained checkpoint's validated ln_post_weight/bias. Report this policy.
    return state


class _VisualADCore:
    def __init__(
        self,
        torch,
        numpy,
        gaussian_filter,
        visual,
        transforms,
        cross_attention,
        modules,
    ):
        self.torch = torch
        self.numpy = numpy
        self.gaussian_filter = gaussian_filter
        self.visual = visual.eval().requires_grad_(False)
        self.transforms = transforms.eval().requires_grad_(False)
        self.cross_attention = cross_attention.eval().requires_grad_(False)
        self.generate_map = modules[
            "anomaly_detection"
        ].generate_anomaly_map_from_tokens
        self.reduce_map = modules["scoring"].reduce_anomaly_map

    def predict(self, batch):
        torch = self.torch
        if tuple(batch.shape) != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError("VisualAD requires one 518x518 RGB image")
        with torch.inference_mode():
            output = self.visual(batch, list(FEATURES))
            patches = output["patch_tokens"]
            start = output["patch_start_idx"]
            if len(patches) != len(FEATURES) or start != 3:
                raise ValueError("VisualAD visual output contract mismatch")
            adapted = self.cross_attention(
                output["anomaly_features"],
                output["normal_features"],
                [patch[:, start:, :] for patch in patches],
                list(FEATURES),
            )
            if len(adapted) != len(FEATURES):
                raise ValueError("VisualAD cross-attention output contract mismatch")
            maps = []
            for layer, patch, tokens in zip(FEATURES, patches, adapted, strict=True):
                anomaly = torch.nn.functional.normalize(
                    tokens["anomaly"], dim=1, eps=1e-8
                )
                normal = torch.nn.functional.normalize(
                    tokens["normal"], dim=1, eps=1e-8
                )
                transformed = self.transforms[f"layer_{layer}"](
                    patch.reshape(-1, FEATURE_DIM)
                ).reshape(patch.shape)
                maps.append(
                    self.generate_map(
                        anomaly, normal, transformed[:, start:, :], IMAGE_SIZE
                    )
                )
            fused = torch.stack(maps).sum(dim=0).cpu()
            if tuple(fused.shape) != (1, IMAGE_SIZE, IMAGE_SIZE) or not bool(
                torch.isfinite(fused).all()
            ):
                raise ValueError("VisualAD fused anomaly map is invalid")
            filtered = self.gaussian_filter(fused[0].numpy(), sigma=4)
            anomaly_map = torch.from_numpy(filtered)
            if not bool(torch.isfinite(anomaly_map).all()) or bool(
                (anomaly_map.abs() > RAW_SCORE_LIMIT).any()
            ):
                raise ValueError("VisualAD filtered anomaly map is outside [-8, 8]")
            # The pinned official code uses CEIL for the top-1% pixel count.
            score = self.reduce_map(anomaly_map, mode="topk_mean", topk_ratio=0.01)
            return float(score), anomaly_map


class VisualADAnomalyDetector:
    """Lazy CPU detector. Missing/failed inference returns None, never NORMAL.

    score is the raw top-1% fused-map mean in [-8,8], NOT a probability.
    enabled means configured; ready means verified weights, trained adapters and
    one complete 518px inference probe succeeded. Automatic retries are bounded
    to one attempt; warmup(retry=True) explicitly retries a failed runtime.
    """

    def __init__(
        self,
        enabled,
        repo_path,
        backbone_path,
        checkpoint_path,
        threshold=0.0,
        num_threads=2,
    ):
        self.enabled = bool(enabled)
        self.repo_path = repo_path
        self.backbone_path = backbone_path
        self.checkpoint_path = checkpoint_path
        self.threshold = threshold
        self.num_threads = num_threads
        self.model_name = "VisualAD/ViT-L-14-336/VisA"
        self.source_commit = SOURCE_COMMIT
        self.ln_post_policy = LN_POST_POLICY
        self.model_sha256 = self.backbone_sha256 = None
        self.ready = False
        self.last_error = None
        self.load_state = "not_loaded" if self.enabled else "disabled"
        self.labels = ["anomaly_score"]
        self.anomaly_labels = {"defect"}
        self._device = self.device_preference = "cpu"
        self._core = self._torch = self._numpy = self._Image = None
        self._loaded_configuration = None
        self._load_attempted = False
        self._lock = threading.RLock()
        self.last_anomaly_map = None
        self.inference_count = 0
        if self.enabled:
            try:
                self._validate_config()
            except (TypeError, ValueError) as exc:
                self._fail(exc)
                self._load_attempted = True

    def _validate_config(self):
        for name in ("repo_path", "backbone_path", "checkpoint_path"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value.strip()
                or not Path(value).is_absolute()
            ):
                raise ValueError(
                    f"VisualAD {name} must be an explicit absolute local path"
                )
        if (
            Path(self.backbone_path).suffix != ".safetensors"
            or Path(self.checkpoint_path).suffix != ".pth"
        ):
            raise ValueError(
                "VisualAD requires a safetensors backbone and pinned .pth adapter"
            )
        if (
            isinstance(self.threshold, bool)
            or not isinstance(self.threshold, (int, float))
            or not math.isfinite(self.threshold)
            or not -8 <= self.threshold <= 8
        ):
            raise ValueError("VisualAD raw threshold must be finite in [-8, 8]")
        self.threshold = float(self.threshold)
        if type(self.num_threads) is not int or not 1 <= self.num_threads <= 4:
            raise ValueError("VisualAD num_threads must be an integer in [1, 4]")

    def _configuration(self):
        return (
            self.repo_path,
            self.backbone_path,
            self.checkpoint_path,
            self.num_threads,
        )

    def _fail(self, exc):
        self.ready = False
        self.load_state = "error"
        self.last_error = f"{type(exc).__name__}: {exc}"
        self.model_sha256 = self.backbone_sha256 = None
        self._core = self._torch = self._numpy = self._Image = None
        self._loaded_configuration = None
        self.last_anomaly_map = None
        log.warning("VisualAD unavailable: %s", self.last_error)

    def _load_runtime(self):
        modules = _load_external_modules(Path(self.repo_path))
        torch = importlib.import_module("torch")
        numpy = importlib.import_module("numpy")
        image_module = importlib.import_module("PIL.Image")
        gaussian_filter = importlib.import_module("scipy.ndimage").gaussian_filter
        safe_open = importlib.import_module("safetensors").safe_open
        version = tuple(
            int(part) for part in torch.__version__.split("+")[0].split(".")[:2]
        )
        if version < (2, 6):
            raise RuntimeError(
                "VisualAD safe checkpoint loading requires PyTorch >=2.6"
            )
        torch.set_num_threads(self.num_threads)
        with _verified_artifact(
            Path(self.checkpoint_path), CHECKPOINT_SIZE, CHECKPOINT_SHA256
        ) as path:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        _validate_checkpoint(torch, checkpoint)
        visual = modules["vision"].VisionTransformer(
            input_resolution=336,
            patch_size=14,
            width=FEATURE_DIM,
            layers=24,
            heads=16,
            output_dim=768,
        )
        with (
            _verified_artifact(
                Path(self.backbone_path), BACKBONE_SIZE, BACKBONE_SHA256
            ) as path,
            safe_open(path, framework="pt", device="cpu") as weights,
        ):
            # Text is deliberately not instantiated or read into CPU tensors.
            tensor_names = weights.keys()  # safe_open is not a dict/iterable.
            base = {
                key: weights.get_tensor(key)
                for key in tensor_names
                if key.startswith("visual.")
            }
            state = _vision_state(torch, visual, base, checkpoint)
            _strict_state(torch, visual, state, "visual")
            del state, base
        transforms = torch.nn.ModuleDict()
        for layer in FEATURES:
            name = f"layer_{layer}"
            transform = modules["feature_transform"].create_feature_transform(
                transform_type="mlp",
                input_dim=FEATURE_DIM,
                hidden_dim=FEATURE_DIM,
                output_dim=FEATURE_DIM,
                dropout=0.0,
            )
            _strict_state(torch, transform, checkpoint["layer_transforms"][name], name)
            transforms[name] = transform
        cross_attention = modules[
            "cross_attention"
        ].build_layer_adaptive_cross_attention(
            layers=list(FEATURES),
            embed_dim=FEATURE_DIM,
            num_anchors=4,
            dropout=0.1,
            res_scale_init=0.01,
        )
        _strict_state(
            torch, cross_attention, checkpoint["cross_attn"], "cross_attention"
        )
        del checkpoint
        log.warning(
            "VisualAD ln_post policy=%s: validated trained ln_post is intentionally not applied, matching official test.py",
            LN_POST_POLICY,
        )
        core = _VisualADCore(
            torch, numpy, gaussian_filter, visual, transforms, cross_attention, modules
        )
        return torch, numpy, image_module, core

    @staticmethod
    def _validate_score(score):
        if (
            isinstance(score, bool)
            or not isinstance(score, (float, int))
            or not math.isfinite(score)
            or not -RAW_SCORE_LIMIT <= score <= RAW_SCORE_LIMIT
        ):
            raise ValueError("VisualAD raw anomaly score must be finite in [-8, 8]")

    def _prepare_image(self, image, torch, numpy):
        # Official utils.transforms replaces the first resize with a square
        # bicubic resize; the following 518px center crop is therefore a no-op.
        image = image.convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE), resample=self._Image.Resampling.BICUBIC
        )
        array = numpy.array(image, dtype=numpy.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean = torch.tensor((0.48145466, 0.4578275, 0.40821073)).reshape(3, 1, 1)
        std = torch.tensor((0.26862954, 0.26130258, 0.27577711)).reshape(3, 1, 1)
        return ((tensor - mean) / std).unsqueeze(0)

    def warmup(self, *, retry=False):
        with self._lock:
            if not self.enabled:
                return False
            if self.ready:
                return True
            if self._load_attempted and not retry:
                return False
            self._load_attempted = True
            self.load_state = "loading"
            try:
                self._validate_config()
                torch, numpy, image_module, core = self._load_runtime()
                self._Image = image_module
                sample = self._prepare_image(
                    image_module.new("RGB", (IMAGE_SIZE, IMAGE_SIZE)), torch, numpy
                )
                score, _ = core.predict(sample)
                self._validate_score(score)
                self._torch, self._numpy, self._core = torch, numpy, core
                self._loaded_configuration = self._configuration()
                self.model_sha256 = CHECKPOINT_SHA256
                self.backbone_sha256 = BACKBONE_SHA256
                self.ready = True
                self.load_state = "ready"
                self.last_error = None
                log.info(
                    "VisualAD ready: checkpoint=%s backbone=%s source=%s raw_score=true",
                    self.model_sha256,
                    self.backbone_sha256,
                    SOURCE_COMMIT,
                )
                return True
            except Exception as exc:  # noqa: BLE001 - optional external runtime.
                self._fail(exc)
                return False

    def classify(self, frame_rgb):
        with self._lock:
            if not self.enabled or not self.warmup():
                return None
            try:
                self._validate_config()
                if self._configuration() != self._loaded_configuration:
                    raise ValueError(
                        "VisualAD runtime configuration changed; explicit reload required"
                    )
                shape = getattr(frame_rgb, "shape", ())
                if (
                    len(shape) != 3
                    or shape[2] != 3
                    or min(shape[:2]) <= 0
                    or max(shape[:2]) > 8192
                    or str(getattr(frame_rgb, "dtype", "")) != "uint8"
                ):
                    raise ValueError(
                        "VisualAD input must be HWC RGB uint8, dimensions <=8192"
                    )
                batch = self._prepare_image(
                    self._Image.fromarray(frame_rgb), self._torch, self._numpy
                )
                score, anomaly_map = self._core.predict(batch)
                self._validate_score(score)
                self.last_anomaly_map = anomaly_map
                self.inference_count += 1
                return {
                    "label": "defect" if score >= self.threshold else "normal",
                    "score": score,
                    "raw_score": score,
                    "scores": [score],
                    "labels": list(self.labels),
                    "backend": "visualad",
                    "model_sha256": self.model_sha256,
                    "backbone_sha256": self.backbone_sha256,
                    "source_commit": SOURCE_COMMIT,
                    "image_size": IMAGE_SIZE,
                    "ln_post_policy": LN_POST_POLICY,
                }
            except Exception as exc:  # noqa: BLE001 - optional external inference.
                self._fail(exc)
                return None

    def is_anomaly(self, frame_rgb):
        result = self.classify(frame_rgb)
        return result is not None and result["label"] == "defect", result
