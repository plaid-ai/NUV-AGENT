"""Digest-pinned AnomalyVFM packages and the Ventuno Q QNN runtime.

The signed Fleet digest pins manifest.json, which pins both context files.
Downloads stage separately; activation/rollback belongs to SettingsReconciler.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import logging
import math
import os
import platform
import re
import errno
import stat
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass
from concurrent.futures import Future
from pathlib import Path
from urllib.parse import urlsplit

from nuvion_app.runtime.settings_overlay import SHA256_PATTERN
from nuvion_app.runtime.visualad_htp import VisualADHTPAnomalyDetector, _require_qnn_only_profile

BACKEND = "anomalyvfm_qnn"
CAPABILITY = "command.config.model.anomalyvfm_qnn.v1"
POINTER = "anomalyvfm/ventuno-q-v16"
FILES = {"context": "compiled_ctx.onnx", "context_binary": "compiled_ctx_qnn.bin"}
MAX_BYTES = {"manifest": 16384, "context": 1048576, "context_binary": 1024**3}
INPUT = {"name": "rgb", "shape": [1, 3, 768, 768], "dtype": "float32"}
OUTPUTS = [
    {"name": "image_logit", "shape": [1, 1], "dtype": "float32"},
    {"name": "pixel_logits", "shape": [1, 1, 96, 96], "dtype": "float32"},
]
_registration_lock = threading.Lock()
_registered_library = None


def require_compatible_platform(environ=None):
    from nuvion_app.runtime.platform_identity import resolve_platform_identity
    identity = resolve_platform_identity(environ=environ)
    if (platform.system() != "Linux" or platform.machine() not in {"aarch64", "arm64"}
            or identity.platform_profile != "ventuno_q"
            or identity.identity_status not in {"VERIFIED", "DEV"}):
        raise ValueError("This QNN context requires a verified Ventuno Q platform")


def _fingerprint(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.resolve(strict=True) != path:
        raise ValueError("Model artifact must be a regular non-symlink file")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _verify_file(path, digest, size):
    before = _fingerprint(path)
    if before[2] != size:
        raise ValueError("Model artifact size mismatch")
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    if hasher.hexdigest() != digest or before != _fingerprint(path):
        raise ValueError("Model artifact digest mismatch or concurrent modification")


@dataclass(frozen=True)
class ModelSelection:
    store: Path
    directory: Path
    pointer: str
    digest: str

    @property
    def manifest_path(self):
        return self.directory / "manifest.json"

    def fingerprint(self):
        return tuple(_fingerprint(self.directory / name)
                     for name in ("manifest.json", *FILES.values()))


def selection_from_env(environ, *, pointer=None, digest=None):
    pointer = pointer if pointer is not None else environ.get("NUVION_MODEL_POINTER", "")
    digest = digest if digest is not None else environ.get("NUVION_MODEL_DIGEST", "")
    if not re.fullmatch(r"anomalyvfm/ventuno-q-[A-Za-z0-9][A-Za-z0-9._-]{0,63}", pointer):
        raise ValueError("Unsupported AnomalyVFM model pointer")
    if not SHA256_PATTERN.fullmatch(digest):
        raise ValueError("AnomalyVFM requires an externally pinned manifest digest")
    root = Path(environ.get("NUVION_ANOMALYVFM_STORE", "/var/lib/nuv-agent/models/anomalyvfm"))
    if not root.is_absolute() or root.resolve() != root or root == Path("/"):
        raise ValueError("Model store must be an absolute path without symlinks")
    return ModelSelection(root, root / digest.removeprefix("sha256:"), pointer, digest)


def read_manifest(selection, *, verify_artifacts=True):
    path = selection.manifest_path
    before = _fingerprint(path)
    if not 0 < before[2] <= MAX_BYTES["manifest"]:
        raise ValueError("Invalid model manifest size")
    raw = path.read_bytes()
    if "sha256:" + hashlib.sha256(raw).hexdigest() != selection.digest:
        raise ValueError("Model manifest differs from signed Fleet digest")
    manifest = json.loads(raw)
    contract = {"schemaVersion": 1, "backend": BACKEND, "pointer": selection.pointer,
                "platformProfile": "ventuno_q", "ortVersion": "1.30.0",
                "qnnVersion": "2.6.0", "input": INPUT, "outputs": OUTPUTS,
                "postprocess": "sigmoid-zero-pad-avg5-v1"}
    if not isinstance(manifest, dict) or any(manifest.get(k) != v for k, v in contract.items()):
        raise ValueError("AnomalyVFM manifest runtime contract mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(FILES):
        raise ValueError("AnomalyVFM requires exactly two context artifacts")
    for key, name in FILES.items():
        item = artifacts[key]
        if (not isinstance(item, dict) or set(item) != {"path", "sha256", "sizeBytes"}
                or item["path"] != name or type(item["sizeBytes"]) is not int
                or not 0 < item["sizeBytes"] <= MAX_BYTES[key]
                or not isinstance(item["sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])):
            raise ValueError("Invalid AnomalyVFM artifact contract")
        if verify_artifacts:
            _verify_file(selection.directory / name, item["sha256"], item["sizeBytes"])
    if before != _fingerprint(path):
        raise ValueError("Model manifest changed during verification")
    return manifest


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError("Model download redirects are forbidden")


def _download(url, destination, expected_digest, size):
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname != "storage.googleapis.com"
            or parsed.port not in {None, 443} or parsed.username or parsed.password):
        raise ValueError("Model artifacts require a Google Storage HTTPS signed URL")
    opener = urllib.request.build_opener(_NoRedirect())
    hasher, count = hashlib.sha256(), 0
    deadline = time.monotonic() + 300
    try:
        with opener.open(url, timeout=30) as response, destination.open("xb") as output:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                if time.monotonic() > deadline:
                    raise TimeoutError("Model download deadline exceeded")
                count += len(chunk)
                if count > size:
                    raise ValueError("Model download exceeded declared size")
                output.write(chunk)
                hasher.update(chunk)
            output.flush()
            os.fsync(output.fileno())
        if count != size or hasher.hexdigest() != expected_digest:
            raise ValueError("Model download integrity mismatch")
        destination.chmod(0o444)
    except Exception:
        destination.unlink(missing_ok=True)
        # Never include signed URLs, device tokens or response bodies in reports.
        raise RuntimeError("Model artifact download failed integrity/transport validation") from None


def ensure_package(selection, environ=None):
    """Authenticate as the device, verify, then atomically publish a digest directory."""
    values = os.environ if environ is None else environ
    require_compatible_platform(values)
    if selection.directory.exists():
        read_manifest(selection)
        return selection
    from nuvion_app.model_store import _fetch_server_presign, _login_for_access_token
    base = values.get("NUVION_MODEL_SERVER_BASE_URL", values.get("NUVION_SERVER_BASE_URL", "")).rstrip("/")
    if urlsplit(base).scheme != "https":
        raise ValueError("Model server requires HTTPS")
    token = values.get("NUVION_MODEL_SERVER_ACCESS_TOKEN") or _login_for_access_token(
        base, values.get("NUVION_DEVICE_USERNAME", ""), values.get("NUVION_DEVICE_PASSWORD", ""))
    response = _fetch_server_presign(base_url=base, token=token, pointer=selection.pointer,
                                    profile="qnn-context", ttl_seconds=600)
    artifacts = response.get("artifacts", [])
    if response.get("pointer") != selection.pointer or len(artifacts) != 3:
        raise ValueError("Model resolver response mismatch")
    by_key = {item["key"]: item for item in artifacts}
    if set(by_key) != {"manifest", *FILES}:
        raise ValueError("Model resolver artifact set mismatch")
    for key, item in by_key.items():
        if (type(item.get("sizeBytes")) is not int or not 0 < item["sizeBytes"] <= MAX_BYTES[key]
                or not re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", ""))):
            raise ValueError("Model resolver integrity metadata invalid")
    if by_key["manifest"]["sha256"] != selection.digest.removeprefix("sha256:"):
        raise ValueError("Model pointer moved away from the signed requested digest")
    selection.store.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix=".download-", dir=selection.store) as temporary:
        staged = ModelSelection(selection.store, Path(temporary), selection.pointer, selection.digest)
        item = by_key["manifest"]
        _download(item["url"], staged.manifest_path, item["sha256"], item["sizeBytes"])
        manifest = read_manifest(staged, verify_artifacts=False)
        for key, name in FILES.items():
            item, expected = by_key[key], manifest["artifacts"][key]
            if any(item[k] != expected[k] for k in ("sha256", "sizeBytes")):
                raise ValueError("Resolver artifacts do not match pinned manifest")
            _download(item["url"], staged.directory / name, expected["sha256"], expected["sizeBytes"])
        read_manifest(staged)
        try:
            os.rename(staged.directory, selection.directory)
        except OSError as exc:
            if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                raise
            read_manifest(selection)
        fd = os.open(selection.store, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return selection


def postprocess(image_logit, pixel_logits):
    np = importlib.import_module("numpy")
    if (image_logit.shape != (1, 1) or pixel_logits.shape != (1, 1, 96, 96)
            or image_logit.dtype != np.float32 or pixel_logits.dtype != np.float32
            or not np.isfinite(image_logit).all() or not np.isfinite(pixel_logits).all()):
        raise ValueError("AnomalyVFM output contract mismatch")
    def sigmoid(x):
        # Stable for all finite logits, without overflow or changing saturation.
        exp = np.exp(-np.abs(x))
        return np.where(x >= 0, 1 / (1 + exp), exp / (1 + exp))
    score = float(sigmoid(image_logit)[0, 0])
    p = sigmoid(pixel_logits[0, 0])
    padded = np.pad(p, ((2, 2), (2, 2)))
    anomaly_map = np.zeros_like(p)
    for dy in range(5):
        for dx in range(5):
            anomaly_map += padded[dy:dy + 96, dx:dx + 96]
    return score, anomaly_map / np.float32(25)


class AnomalyVFMDetector(VisualADHTPAnomalyDetector):
    """Same pipeline interface, a distinct model/runtime and score contract."""
    def __init__(self, environ=None):
        self.environ = dict(os.environ if environ is None else environ)
        self.selection = selection_from_env(self.environ)
        super().__init__(True, str(self.selection.manifest_path),
                         self.selection.digest.removeprefix("sha256:"),
                         self.environ.get("NUVION_ANOMALYVFM_STATE_DIR", "/var/lib/nuv-agent/anomalyvfm"),
                         float(self.environ.get("NUVION_ZERO_SHOT_THRESHOLD", "0.7")))
        self.model_name = "AnomalyVFM/RADIO/MinMax-W8A16/V-range"
        self.source_commit = self.ln_post_policy = None
        self._loaded_fingerprint = self._result_at = None
        self._preflight_future = self._preflight_target = None

    def _fail(self, exc):
        self.ready = False
        self.last_error = f"{type(exc).__name__}: {exc}"
        self.model_sha256 = self.backbone_sha256 = None
        self.graph_sha256 = self.loaded_manifest_sha256 = None
        self.execution_provider = None
        self.last_inference_seconds = self.last_anomaly_map = None
        self._session = self._result_at = None
        logging.getLogger(__name__).error("AnomalyVFM NPU unavailable: %s", self.last_error)

    def _validate_threshold(self):
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("AnomalyVFM threshold must be finite in [0,1]")

    def _load(self):
        global _registered_library
        self._validate_threshold()
        ensure_package(self.selection, self.environ)
        manifest = read_manifest(self.selection)
        # Inspect the small wrapper before native QNN reads its external context.
        onnx = importlib.import_module("onnx")
        graph = onnx.load(str(self.selection.directory / FILES["context"]), load_external_data=False)
        if len(graph.graph.node) != 1:
            raise ValueError("Expected one QNN EPContext node")
        node = graph.graph.node[0]
        attributes = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        if (node.op_type != "EPContext" or node.domain != "com.microsoft"
                or attributes.get("ep_cache_context") != FILES["context_binary"].encode()
                or attributes.get("embed_mode") != 0
                or attributes.get("source") != b"QNNExecutionProvider"):
            raise ValueError("QNN context references an unsupported external artifact")
        before = self.selection.fingerprint()
        ort, qnn = importlib.import_module("onnxruntime"), importlib.import_module("onnxruntime_qnn")
        if ort.__version__ != manifest["ortVersion"] or qnn.__version__ != manifest["qnnVersion"]:
            raise RuntimeError("AnomalyVFM requires tested ORT 1.30.0 / QNN EP 2.6.0")
        library = qnn.get_library_path()
        with _registration_lock:
            if _registered_library is None:
                ort.register_execution_provider_library("QNNExecutionProvider", library)
                _registered_library = library
            elif _registered_library != library:
                raise RuntimeError("Mixed QNN provider libraries are forbidden")
        devices = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"
                   and d.device.type == ort.OrtHardwareDeviceType.NPU]
        if len(devices) != 1:
            raise RuntimeError("AnomalyVFM requires one QNN NPU, without CPU fallback")
        state = Path(self.state_dir)
        state.mkdir(parents=True, exist_ok=True, mode=0o700)
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        options.enable_profiling = True
        options.profile_file_prefix = str(state / "qnn-profile")
        options.add_provider_for_devices(devices, {
            "backend_path": str(Path(qnn.__file__).parent / "libQnnHtp.so"),
            "htp_performance_mode": "burst", "enable_htp_fp16_precision": "0",
            "offload_graph_io_quantization": "0", "skip_qnn_version_check": "0"})
        session = ort.InferenceSession(str(self.selection.directory / FILES["context"]), sess_options=options)
        session.disable_fallback()
        for actual, expected in ((session.get_inputs(), [INPUT]), (session.get_outputs(), OUTPUTS)):
            if len(actual) != len(expected) or any(
                    a.name != e["name"] or a.shape != e["shape"] or a.type != "tensor(float)"
                    for a, e in zip(actual, expected)):
                raise ValueError("Compiled QNN context input/output mismatch")
        if before != self.selection.fingerprint():
            raise ValueError("Model files changed while creating QNN session")
        self._loaded_fingerprint = before
        self._session = session
        self._verified_graph_sha256 = manifest["artifacts"]["context"]["sha256"]
        self._loaded_identity = (self.manifest_path, self.manifest_sha256)

    def startup_pending(self):
        return self.enabled and not self.ready and self.last_error is None

    def loaded_model_proof(self):
        if (not self.ready or self.last_error or self._result_at is None
                or not 0 <= time.monotonic() - self._result_at <= 60
                or self.execution_provider != "QNNExecutionProvider/HTP"):
            return None
        try:
            if self.selection.fingerprint() != self._loaded_fingerprint:
                return None
        except (OSError, ValueError):
            return None
        return {"pointer": self.selection.pointer, "digest": self.selection.digest}

    def preflight_model(self, desired):
        candidate = selection_from_env(self.environ, pointer=desired["pointer"], digest=desired["digest"])
        # Keep the 30-second settings effect lease and control-plane heartbeat free
        # while downloading a large candidate. No active setting is changed here.
        target = (candidate.pointer, candidate.digest)
        with self._lock:
            if self._preflight_future is not None:
                if not self._preflight_future.done():
                    return False
                if self._preflight_target == target:
                    completed = self._preflight_future
                    try:
                        completed.result()
                    except Exception:
                        # Fail this command, but permit a later signed command to retry
                        # a transient download failure without restarting the agent.
                        self._preflight_future = self._preflight_target = None
                        raise
                    read_manifest(candidate)
                    return True
            future = Future()
            self._preflight_future, self._preflight_target = future, target
            def download():
                try:
                    future.set_result(ensure_package(candidate, self.environ))
                except Exception as exc:
                    future.set_exception(exc)
            threading.Thread(target=download, name="model-preflight", daemon=True).start()
            return False

    def verify_model(self, desired):
        if self.loaded_model_proof() != dict(desired):
            raise RuntimeError("No matching fresh AnomalyVFM NPU inference proof")
        read_manifest(self.selection)
        if self.loaded_model_proof() != dict(desired):
            raise RuntimeError("AnomalyVFM files changed during verification")
        return dict(desired)

    def classify(self, frame_rgb):
        with self._lock:
            if not self.prepare():
                return None
            try:
                if self.selection.fingerprint() != self._loaded_fingerprint:
                    raise ValueError("Loaded AnomalyVFM package changed on disk")
                self._validate_threshold()
                np, image = importlib.import_module("numpy"), importlib.import_module("PIL.Image")
                if (frame_rgb.dtype != np.uint8 or frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3
                        or not 0 < min(frame_rgb.shape[:2]) <= max(frame_rgb.shape[:2]) <= 8192):
                    raise ValueError("Expected HWC RGB uint8 frame up to 8192 pixels")
                rgb = image.fromarray(frame_rgb).resize((768, 768), image.Resampling.BILINEAR)
                batch = np.ascontiguousarray(np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1)[None] / np.float32(255))
                started = time.perf_counter()
                outputs = self._session.run(["image_logit", "pixel_logits"], {"rgb": batch})
                self.last_inference_seconds = time.perf_counter() - started
                if not self._profile_verified:
                    _require_qnn_only_profile(self._session.end_profiling())
                    self._profile_verified = True
                score, self.last_anomaly_map = postprocess(*outputs)
                self.ready, self.last_error = True, None
                self.model_sha256 = self.selection.digest.removeprefix("sha256:")
                self.graph_sha256 = self._verified_graph_sha256
                self.loaded_manifest_sha256 = self.manifest_sha256
                self.execution_provider = "QNNExecutionProvider/HTP"
                self.inference_count += 1
                self._result_at = time.monotonic()
                return {"label": "defect" if score >= self.threshold else "normal", "score": score,
                        "raw_score": score, "backend": BACKEND, "labels": ["anomaly_score"], "scores": [score],
                        "model_sha256": self.model_sha256, "graph_sha256": self.graph_sha256,
                        "manifest_sha256": self.loaded_manifest_sha256, "image_size": 768,
                        "execution_provider": self.execution_provider, "inference_seconds": self.last_inference_seconds}
            except Exception as exc:
                self._result_at = None
                self._fail(exc)
                return None


def build_anomalyvfm_detector(environ):
    from nuvion_app.runtime.anomalyvfm_start import require_fleet_boot_guard
    require_fleet_boot_guard(environ)
    return AnomalyVFMDetector(environ)
