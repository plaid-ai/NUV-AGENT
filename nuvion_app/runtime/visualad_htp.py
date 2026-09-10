"""VisualAD's learned graph on Qualcomm HTP, with no CPU/GPU model fallback.

CPU work is limited to camera-image preprocessing and the paper's non-learned
bilinear/Gaussian/top-1% postprocessing. PyTorch is never imported here.
The separately exported, embedded-weight ONNX artifact is hash-pinned.
External-data graphs are deliberately unsupported by this deployment contract.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from pathlib import Path

from nuvion_app.runtime.visualad import (
    BACKBONE_SHA256,
    CHECKPOINT_SHA256,
    IMAGE_SIZE,
    LN_POST_POLICY,
    SOURCE_COMMIT,
    _opened_regular_file,
    _verified_artifact,
)

log = logging.getLogger(__name__)
_registration_lock = threading.Lock()
_registered_library = None
_HASH = re.compile(r"^[0-9a-f]{64}$")
_INPUT = {"name": "image", "shape": [1, 3, 518, 518], "dtype": "float32"}
_OUTPUT = {"name": "patch_maps", "shape": [1, 4, 37, 37], "dtype": "float32"}
_CONTEXT_CACHE_SCHEMA_VERSION = 1
_CONTEXT_CACHE_MAX_BYTES = 2_000_000_000


def _sha256_regular_file(path: Path, *, max_bytes: int) -> tuple[int, str]:
    with _opened_regular_file(path) as (descriptor, metadata):
        if not 1 <= metadata.st_size <= max_bytes:
            raise ValueError(f"Invalid file size for digest: {path}")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
    return metadata.st_size, digest.hexdigest()


def verify_manifest(path: str, expected_sha256: str) -> tuple[dict, Path]:
    """The digest is supplied by deployment, never trusted from the same file."""
    if not isinstance(expected_sha256, str) or not _HASH.fullmatch(expected_sha256):
        raise ValueError("VisualAD HTP requires an externally pinned manifest SHA256")
    manifest_path = Path(path)
    with _opened_regular_file(manifest_path) as (fd, metadata):
        if not 1 <= metadata.st_size <= 65536:
            raise ValueError("VisualAD HTP manifest size is invalid")
        raw = os.read(fd, 65537)
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("VisualAD HTP manifest SHA256 mismatch")
    manifest = json.loads(raw)
    expected = {
        "schemaVersion": 1,
        "sourceCommit": SOURCE_COMMIT,
        "checkpointSha256": CHECKPOINT_SHA256,
        "backboneSha256": BACKBONE_SHA256,
        "lnPostPolicy": LN_POST_POLICY,
        "graph": "visualad.onnx",
        "input": _INPUT,
        "output": _OUTPUT,
        "opset": 20,
    }
    if not isinstance(manifest, dict):
        raise TypeError("VisualAD HTP manifest must be an object")
    for key, value in expected.items():
        if type(manifest.get(key)) is not type(value) or manifest[key] != value:
            raise ValueError(f"VisualAD HTP manifest contract mismatch: {key}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not 1 <= len(artifacts) <= 16:
        raise ValueError("VisualAD HTP artifacts must be a bounded list")
    seen = set()
    total = 0
    root = manifest_path.parent.resolve(strict=True)
    for artifact in artifacts:
        if not isinstance(artifact, dict) or set(artifact) != {
            "path",
            "sha256",
            "size",
        }:
            raise ValueError("Invalid VisualAD HTP artifact entry")
        name, sha, size = artifact["path"], artifact["sha256"], artifact["size"]
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name)
            or name in seen
            or not isinstance(sha, str)
            or not _HASH.fullmatch(sha)
            or type(size) is not int
            or not 1 <= size <= 2_000_000_000
        ):
            raise ValueError("Invalid/duplicate VisualAD HTP artifact identity")
        seen.add(name)
        total += size
        if total > 4_000_000_000:
            raise ValueError("VisualAD HTP artifact size limit exceeded")
        with _opened_regular_file(root / name) as (fd, metadata):
            if metadata.st_size != size:
                raise ValueError(f"VisualAD HTP artifact size mismatch: {name}")
            digest = hashlib.sha256()
            while chunk := os.read(fd, 1024 * 1024):
                digest.update(chunk)
            if digest.hexdigest() != sha:
                raise ValueError(f"VisualAD HTP artifact SHA256 mismatch: {name}")
    if manifest["graph"] not in seen:
        raise ValueError("VisualAD HTP graph is not in the verified artifact list")
    if seen != {manifest["graph"]}:
        raise ValueError("VisualAD HTP currently requires a single embedded graph")
    return manifest, root / manifest["graph"]


def _require_embedded_graph(pinned_graph: str) -> None:
    """Parse without resolving any external path, including nested tensors."""
    onnx = importlib.import_module("onnx")
    model = onnx.load(pinned_graph, load_external_data=False)

    def inspect(message):
        if isinstance(message, onnx.TensorProto) and (
            message.data_location == onnx.TensorProto.EXTERNAL
            or message.external_data
        ):
            raise ValueError("VisualAD HTP external-data tensors are not supported")
        for field, value in message.ListFields():
            if field.message_type is not None:
                for child in value if field.is_repeated else (value,):
                    inspect(child)

    inspect(model)


def postprocess_patch_maps(patch_maps):
    """Match torch bilinear align_corners=False, then official score reduction."""
    np = importlib.import_module("numpy")
    gaussian_filter = importlib.import_module("scipy.ndimage").gaussian_filter
    maps = np.asarray(patch_maps)
    if maps.shape != (1, 4, 37, 37) or maps.dtype != np.float32:
        raise ValueError("VisualAD HTP output must be FP32 [1,4,37,37]")
    if not np.isfinite(maps).all() or np.any(np.abs(maps) > 2.01):
        raise ValueError(
            "VisualAD HTP layer maps are not finite/bounded cosine differences"
        )
    coordinates = (
        np.arange(IMAGE_SIZE, dtype=np.float32) + np.float32(0.5)
    ) * np.float32(37 / IMAGE_SIZE) - np.float32(0.5)
    coordinates = np.clip(coordinates, 0, 36)
    lower = np.floor(coordinates).astype(np.intp)
    upper = np.minimum(lower + 1, 36)
    fraction = coordinates - lower.astype(np.float32)
    source = maps[0]
    rows = (
        source[:, lower, :] * (1 - fraction)[None, :, None]
        + source[:, upper, :] * fraction[None, :, None]
    )
    expanded = (
        rows[:, :, lower] * (1 - fraction)[None, None, :]
        + rows[:, :, upper] * fraction[None, None, :]
    )
    fused = expanded.sum(axis=0, dtype=np.float32)
    filtered = gaussian_filter(fused, sigma=4)
    if not np.isfinite(filtered).all() or np.any(np.abs(filtered) > 8):
        raise ValueError("VisualAD HTP filtered map is outside finite [-8,8]")
    flat = filtered.reshape(-1)
    k = math.ceil(flat.size * 0.01)
    score = float(np.partition(flat, flat.size - k)[-k:].mean(dtype=np.float32))
    if not math.isfinite(score) or not -8 <= score <= 8:
        raise ValueError("VisualAD HTP raw score is outside [-8,8]")
    return score, filtered


def _require_qnn_only_profile(profile_path: str) -> list[str]:
    events = json.loads(Path(profile_path).read_text())
    providers = sorted(
        {
            event.get("args", {}).get("provider")
            for event in events
            if event.get("args", {}).get("provider")
        }
    )
    if not providers or any(
        provider != "QNNExecutionProvider" for provider in providers
    ):
        raise RuntimeError(
            f"VisualAD graph did not execute exclusively on QNN: {providers}"
        )
    return providers


def _require_one_htp_bundle(bundle: Path) -> None:
    mappings = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        if "/" in line:
            path = line[line.index("/") :]
            if "libQnn" in path or "libonnxruntime_providers_qnn" in path:
                mappings.add(Path(path))
    if not any(path.name == "libQnnHtp.so" for path in mappings):
        raise RuntimeError("HTP backend is not loaded")
    if any(not path.is_relative_to(bundle) for path in mappings):
        raise RuntimeError("Mixed system/bundled QNN libraries are forbidden")


class VisualADHTPAnomalyDetector:
    def __init__(
        self, enabled, manifest_path, manifest_sha256, state_dir, threshold=0.0
    ):
        self.enabled = bool(enabled)
        self.manifest_path = manifest_path
        self.manifest_sha256 = manifest_sha256
        self.state_dir = state_dir
        self.threshold = threshold
        self.ready = False
        self.last_error = None
        self.model_name = "VisualAD/ViT-L-14-336/VisA/HTP"
        self.source_commit = SOURCE_COMMIT
        self.ln_post_policy = LN_POST_POLICY
        self.model_sha256 = self.backbone_sha256 = None
        self.graph_sha256 = self.loaded_manifest_sha256 = None
        self.execution_provider = None
        self.last_anomaly_map = None
        self.last_inference_seconds = None
        self.inference_count = 0
        self.context_cache_hit = False
        self.labels = ["anomaly_score"]
        self.anomaly_labels = {"defect"}
        self._session = None
        self._attempted = False
        self._profile_verified = False
        self._lock = threading.RLock()

    def _load(self):
        self._validate_threshold()
        manifest, graph = verify_manifest(self.manifest_path, self.manifest_sha256)
        self._loaded_identity = (self.manifest_path, self.manifest_sha256)
        artifact = manifest["artifacts"][0]
        # Keep the verified inode open through ONNX parsing and ORT session
        # creation. A pathname replacement must not change the loaded bytes.
        with _verified_artifact(graph, artifact["size"], artifact["sha256"]) as pinned:
            _require_embedded_graph(pinned)
            self._load_session(pinned, artifact["sha256"])
        self._verified_graph_sha256 = artifact["sha256"]

    def _validate_threshold(self):
        if (
            isinstance(self.threshold, bool)
            or not isinstance(self.threshold, (float, int))
            or not math.isfinite(self.threshold)
            or not -8 <= self.threshold <= 8
        ):
            raise ValueError("VisualAD raw threshold must be finite in [-8,8]")

    def _load_session(self, graph, graph_sha256):
        global _registered_library
        state = Path(self.state_dir)
        if not state.is_absolute() or not state.is_dir():
            raise ValueError(
                "VisualAD HTP requires an existing absolute writable state directory"
            )
        ort = importlib.import_module("onnxruntime")
        qnn = importlib.import_module("onnxruntime_qnn")
        if ort.__version__ != "1.26.0" or qnn.__version__ != "2.5.0":
            raise RuntimeError(
                "VisualAD HTP requires the tested ORT 1.26.0/QNN EP 2.5.0 combination"
            )
        library = str(Path(qnn.get_library_path()).resolve(strict=True))
        backend = Path(qnn.get_qnn_htp_path()).resolve(strict=True)
        bundle = str(backend.parent)
        if backend.name != "libQnnHtp.so" or any(
            os.getenv(key) != bundle for key in ("LD_LIBRARY_PATH", "ADSP_LIBRARY_PATH")
        ):
            raise RuntimeError(
                "VisualAD requires one command-scoped HTP SDK bundle, without system SDK mixing"
            )
        with _registration_lock:
            if _registered_library is None:
                ort.register_execution_provider_library("QNNExecutionProvider", library)
                _registered_library = library
            elif _registered_library != library:
                raise RuntimeError("A different QNN plugin is already registered")
        devices = [
            device
            for device in ort.get_ep_devices()
            if device.ep_name == "QNNExecutionProvider"
            and device.device.type == ort.OrtHardwareDeviceType.NPU
        ]
        if not devices:
            raise RuntimeError(
                "QNN did not enumerate an NPU; CPU/GPU fallback is forbidden"
            )
        _, plugin_sha256 = _sha256_regular_file(Path(library), max_bytes=512 * 1024 * 1024)
        _, backend_sha256 = _sha256_regular_file(backend, max_bytes=512 * 1024 * 1024)
        cache_identity = {
            "schemaVersion": _CONTEXT_CACHE_SCHEMA_VERSION,
            "graphSha256": graph_sha256,
            "ortVersion": ort.__version__,
            "qnnVersion": qnn.__version__,
            "pluginSha256": plugin_sha256,
            "backendSha256": backend_sha256,
            "providerProfile": "htp-fp16-ioquant0-balanced-v1",
        }
        identity_sha256 = hashlib.sha256(
            json.dumps(cache_identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        cache_path = state / f"visualad-htp-{identity_sha256[:24]}-ctx.onnx"
        metadata_path = cache_path.with_suffix(".json")
        provider_options = {
            "backend_path": str(backend),
            "enable_htp_fp16_precision": "1",
            "offload_graph_io_quantization": "0",
            "skip_qnn_version_check": "0",
            "htp_performance_mode": "balanced",
            # Detailed DSP tracing is performed by the bounded deployment
            # probe, not continuously in a long-running camera service.
            "profiling_level": "off",
        }

        def create_session(model_path: str, context_output: Path | None = None):
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            if context_output is not None:
                options.add_session_config_entry("ep.context_enable", "1")
                options.add_session_config_entry("ep.context_embed_mode", "1")
                options.add_session_config_entry(
                    "ep.context_file_path", str(context_output)
                )
            options.enable_profiling = True
            options.profile_file_prefix = str(state / "visualad-htp-ort")
            options.add_provider_for_devices([devices[0]], provider_options)
            return ort.InferenceSession(model_path, sess_options=options)

        def validate_session(candidate):
            candidate.disable_fallback()
            if "QNNExecutionProvider" not in candidate.get_providers():
                raise RuntimeError("VisualAD session lost QNN; refusing to execute")
            _require_one_htp_bundle(backend.parent)
            inputs, outputs = candidate.get_inputs(), candidate.get_outputs()
            for values, expected in ((inputs, _INPUT), (outputs, _OUTPUT)):
                if (
                    len(values) != 1
                    or values[0].name != expected["name"]
                    or values[0].shape != expected["shape"]
                    or values[0].type != "tensor(float)"
                ):
                    raise ValueError(
                        "VisualAD ONNX input/output differs from the pinned contract"
                    )
            return candidate

        session = None
        cache_record = self._read_context_cache(
            cache_path, metadata_path, cache_identity
        )
        if cache_record is not None:
            try:
                with _verified_artifact(
                    cache_path,
                    cache_record["cacheSize"],
                    cache_record["cacheSha256"],
                ) as pinned_cache:
                    session = validate_session(create_session(pinned_cache))
                self.context_cache_hit = True
                log.info("VisualAD HTP context cache hit: %s", cache_path)
            except Exception as exc:  # noqa: BLE001 - regenerate from signed source.
                log.warning("VisualAD HTP context cache rejected; regenerating: %s", exc)
                self._remove_context_cache(cache_path, metadata_path)

        if session is None:
            temporary_cache = cache_path.with_name(
                f".{cache_path.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp.onnx"
            )
            try:
                try:
                    session = validate_session(
                        create_session(str(graph), temporary_cache)
                    )
                except Exception as exc:  # noqa: BLE001 - keep QNN JIT available.
                    log.warning(
                        "VisualAD HTP context generation unavailable; using verified graph: %s",
                        exc,
                    )
                    session = validate_session(create_session(str(graph)))
                else:
                    try:
                        self._publish_context_cache(
                            temporary_cache,
                            cache_path,
                            metadata_path,
                            cache_identity,
                        )
                    except (OSError, TypeError, ValueError) as exc:
                        log.warning(
                            "VisualAD HTP context cache could not be persisted: %s",
                            exc,
                        )
                        self._remove_context_cache(cache_path, metadata_path)
            finally:
                temporary_cache.unlink(missing_ok=True)
            self.context_cache_hit = False
        self._session = session

    @staticmethod
    def _remove_context_cache(cache_path: Path, metadata_path: Path) -> None:
        cache_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)

    @staticmethod
    def _read_context_cache(
        cache_path: Path,
        metadata_path: Path,
        expected_identity: dict,
    ) -> dict | None:
        if not cache_path.exists() or not metadata_path.exists():
            return None
        try:
            with _opened_regular_file(metadata_path) as (descriptor, metadata):
                if not 1 <= metadata.st_size <= 16_384:
                    raise ValueError("context cache metadata size is invalid")
                record = json.loads(os.read(descriptor, 16_385))
            if not isinstance(record, dict):
                raise ValueError("context cache metadata must be an object")
            for key, value in expected_identity.items():
                if record.get(key) != value:
                    raise ValueError(f"context cache identity mismatch: {key}")
            size = record.get("cacheSize")
            digest = record.get("cacheSha256")
            if (
                type(size) is not int
                or not 1 <= size <= _CONTEXT_CACHE_MAX_BYTES
                or not isinstance(digest, str)
                or not _HASH.fullmatch(digest)
            ):
                raise ValueError("context cache digest contract is invalid")
            actual_size, actual_digest = _sha256_regular_file(
                cache_path, max_bytes=_CONTEXT_CACHE_MAX_BYTES
            )
            if actual_size != size or actual_digest != digest:
                raise ValueError("context cache bytes do not match metadata")
            return record
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            log.warning("VisualAD HTP context cache metadata rejected: %s", exc)
            VisualADHTPAnomalyDetector._remove_context_cache(
                cache_path, metadata_path
            )
            return None

    @staticmethod
    def _publish_context_cache(
        temporary_cache: Path,
        cache_path: Path,
        metadata_path: Path,
        identity: dict,
    ) -> None:
        if not temporary_cache.is_file():
            log.warning("VisualAD HTP did not produce a context cache")
            return
        temporary_cache.chmod(0o600)
        cache_size, cache_sha256 = _sha256_regular_file(
            temporary_cache, max_bytes=_CONTEXT_CACHE_MAX_BYTES
        )
        os.replace(temporary_cache, cache_path)
        metadata_tmp = metadata_path.with_name(
            f".{metadata_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            metadata_tmp.write_text(
                json.dumps(
                    {
                        **identity,
                        "cacheSize": cache_size,
                        "cacheSha256": cache_sha256,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            metadata_tmp.chmod(0o600)
            os.replace(metadata_tmp, metadata_path)
        finally:
            metadata_tmp.unlink(missing_ok=True)
        log.info(
            "VisualAD HTP context cache generated path=%s size=%d sha256=%s",
            cache_path,
            cache_size,
            cache_sha256,
        )

    def _fail(self, exc):
        self.ready = False
        self.last_error = f"{type(exc).__name__}: {exc}"
        self.model_sha256 = self.backbone_sha256 = None
        self.graph_sha256 = self.loaded_manifest_sha256 = None
        self.execution_provider = None
        self.last_inference_seconds = None
        self.last_anomaly_map = None
        self._session = None
        log.error("VisualAD HTP unavailable (no CPU fallback): %s", self.last_error)

    def close(self) -> None:
        """Release the native session after the inference worker has stopped."""
        with self._lock:
            self.ready = False
            self._session = None

    def prepare(self) -> bool:
        """Compile before selecting a camera frame; compilation is not health."""
        with self._lock:
            if not self.enabled or (self._attempted and self._session is None):
                return False
            if not self._attempted:
                self._attempted = True
                try:
                    self._load()
                except Exception as exc:  # noqa: BLE001 - no automatic backend fallback.
                    self._fail(exc)
                    return False
            return self._session is not None

    def classify(self, frame_rgb):
        with self._lock:
            if not self.prepare():
                return None
            try:
                self._validate_threshold()
                if (self.manifest_path, self.manifest_sha256) != self._loaded_identity:
                    raise RuntimeError(
                        "VisualAD HTP identity changed; explicit reload required"
                    )
                np = importlib.import_module("numpy")
                image_module = importlib.import_module("PIL.Image")
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
                image = (
                    image_module.fromarray(frame_rgb)
                    .convert("RGB")
                    .resize((518, 518), resample=image_module.Resampling.BICUBIC)
                )
                array = np.asarray(image, dtype=np.float32) / np.float32(255)
                mean = np.array((0.48145466, 0.4578275, 0.40821073), dtype=np.float32)
                std = np.array((0.26862954, 0.26130258, 0.27577711), dtype=np.float32)
                batch = np.ascontiguousarray(
                    ((array - mean) / std).transpose(2, 0, 1)[None]
                )
                started = time.perf_counter()
                maps = self._session.run(["patch_maps"], {"image": batch})[0]
                self.last_inference_seconds = time.perf_counter() - started
                if not self._profile_verified:
                    _require_qnn_only_profile(self._session.end_profiling())
                    self._profile_verified = True
                score, anomaly_map = postprocess_patch_maps(maps)
                self.ready = True
                self.last_error = None
                self.model_sha256 = CHECKPOINT_SHA256
                self.backbone_sha256 = BACKBONE_SHA256
                self.graph_sha256 = self._verified_graph_sha256
                self.loaded_manifest_sha256 = self._loaded_identity[1]
                self.execution_provider = "QNNExecutionProvider/HTP"
                self.last_anomaly_map = anomaly_map
                self.inference_count += 1
                return {
                    "label": "defect" if score >= self.threshold else "normal",
                    "score": score,
                    "raw_score": score,
                    "labels": self.labels,
                    "scores": [score],
                    "backend": "visualad_htp",
                    "execution_provider": self.execution_provider,
                    "model_sha256": self.model_sha256,
                    "backbone_sha256": self.backbone_sha256,
                    "graph_sha256": self.graph_sha256,
                    "manifest_sha256": self.loaded_manifest_sha256,
                    "source_commit": SOURCE_COMMIT,
                    "ln_post_policy": LN_POST_POLICY,
                    "image_size": IMAGE_SIZE,
                    "inference_seconds": self.last_inference_seconds,
                    "context_cache_hit": self.context_cache_hit,
                }
            except Exception as exc:  # noqa: BLE001 - fail closed at the hardware boundary.
                self._fail(exc)
                return None

    def is_anomaly(self, frame_rgb):
        result = self.classify(frame_rgb)
        return result is not None and result["label"] == "defect", result
