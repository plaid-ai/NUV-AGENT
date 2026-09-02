import gc
import json
import logging
import math
import os
import stat
import subprocess
import sys
from pathlib import Path

from nuvion_app.runtime.inference_mode import normalize_siglip_device

log = logging.getLogger(__name__)

_TEXT_WORKER_TIMEOUT_SEC = 300
_MAX_TEXT_WORKER_RESPONSE_BYTES = 64 * 1024 * 1024
_MAX_TEXT_FEATURE_WIDTH = 8192


class ZeroShotAnomalyDetector:
    @staticmethod
    def _format_exc(exc: Exception) -> str:
        return f"{exc.__class__.__name__}: {exc}"

    @staticmethod
    def _pooled_feature_tensor(output):
        """Accept both Transformers feature return contracts, fail closed otherwise."""

        pooled = getattr(output, "pooler_output", None)
        if pooled is not None:
            output = pooled
        if not callable(getattr(output, "norm", None)):
            raise TypeError("model feature output has no tensor pooler output")
        return output

    def _load_processor(self, transformers):
        attempts: list[str] = []

        AutoProcessor = getattr(transformers, "AutoProcessor", None)
        if AutoProcessor is not None:
            try:
                return AutoProcessor.from_pretrained(self.model_name)
            except Exception as exc:  # noqa: BLE001 - backend compatibility chain.
                attempts.append(f"AutoProcessor.from_pretrained failed ({self._format_exc(exc)})")

        for class_name in ("Siglip2Processor", "SiglipProcessor"):
            processor_cls = getattr(transformers, class_name, None)
            if processor_cls is None:
                continue
            try:
                return processor_cls.from_pretrained(self.model_name)
            except Exception as exc:  # noqa: BLE001 - backend compatibility chain.
                attempts.append(f"{class_name}.from_pretrained failed ({self._format_exc(exc)})")

        image_processor = None
        tokenizer = None
        processor_cls = getattr(transformers, "Siglip2Processor", None) or getattr(transformers, "SiglipProcessor", None)

        for class_name in ("SiglipImageProcessor", "AutoImageProcessor"):
            image_processor_cls = getattr(transformers, class_name, None)
            if image_processor_cls is None:
                continue
            try:
                image_processor = image_processor_cls.from_pretrained(self.model_name)
                break
            except Exception as exc:  # noqa: BLE001 - backend compatibility chain.
                attempts.append(f"{class_name}.from_pretrained failed ({self._format_exc(exc)})")

        for class_name in ("GemmaTokenizerFast", "GemmaTokenizer", "AutoTokenizer"):
            tokenizer_cls = getattr(transformers, class_name, None)
            if tokenizer_cls is None:
                continue
            try:
                tokenizer = tokenizer_cls.from_pretrained(self.model_name)
                break
            except Exception as exc:  # noqa: BLE001 - backend compatibility chain.
                attempts.append(f"{class_name}.from_pretrained failed ({self._format_exc(exc)})")

        if processor_cls is not None and image_processor is not None and tokenizer is not None:
            try:
                return processor_cls(image_processor=image_processor, tokenizer=tokenizer)
            except Exception as exc:  # noqa: BLE001 - backend compatibility chain.
                attempts.append(f"{processor_cls.__name__}(image_processor, tokenizer) failed ({self._format_exc(exc)})")

        details = "; ".join(attempts) if attempts else "processor class not found"
        raise RuntimeError(f"Unable to initialize processor for '{self.model_name}': {details}")

    def _load_image_processor(self, transformers, model_source: Path):
        attempts: list[str] = []
        for class_name in (
            "AutoImageProcessor",
            "SiglipImageProcessor",
            "Siglip2ImageProcessor",
        ):
            processor_cls = getattr(transformers, class_name, None)
            if processor_cls is None:
                continue
            try:
                return processor_cls.from_pretrained(str(model_source))
            except Exception as exc:  # noqa: BLE001 - backend compatibility chain.
                attempts.append(
                    f"{class_name}.from_pretrained failed ({self._format_exc(exc)})"
                )
        details = "; ".join(attempts) if attempts else "image processor class not found"
        raise RuntimeError(
            f"Unable to initialize image processor for '{self.model_name}': {details}"
        )

    def _resolve_mps_model_source(self) -> tuple[Path, str | None]:
        candidate = Path(self.model_name).expanduser()
        try:
            resolved = candidate.resolve(strict=True)
            if resolved.is_dir():
                metadata = candidate.lstat()
                evidence = (
                    str(resolved)
                    if stat.S_ISDIR(metadata.st_mode)
                    and not stat.S_ISLNK(metadata.st_mode)
                    else None
                )
                return resolved, evidence
        except OSError:
            pass

        from huggingface_hub import snapshot_download

        resolved = Path(snapshot_download(repo_id=self.model_name)).resolve(strict=True)
        if not resolved.is_dir():
            raise RuntimeError("resolved model snapshot is not a directory")
        return resolved, None

    @staticmethod
    def _validate_text_worker_payload(payload, labels: list[str]) -> list[list[float]]:
        if not isinstance(payload, dict) or set(payload) != {
            "schemaVersion",
            "labels",
            "features",
        }:
            raise ValueError("text-feature response schema is invalid")
        if type(payload["schemaVersion"]) is not int or payload["schemaVersion"] != 1:
            raise ValueError("text-feature response version is invalid")
        if payload["labels"] != labels:
            raise ValueError("text-feature response labels do not match")
        rows = payload["features"]
        if (
            not isinstance(rows, list)
            or len(rows) != len(labels)
            or not rows
            or not isinstance(rows[0], list)
            or not 1 <= len(rows[0]) <= _MAX_TEXT_FEATURE_WIDTH
        ):
            raise ValueError("text-feature response shape is invalid")
        width = len(rows[0])
        if any(
            not isinstance(row, list)
            or len(row) != width
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                for value in row
            )
            for row in rows
        ):
            raise ValueError("text-feature response value is invalid")
        return rows

    def _load_isolated_text_features(self, model_source: Path) -> list[list[float]]:
        worker_path = Path(__file__).resolve().with_name("_siglip_text_features.py")
        metadata = worker_path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError("text-feature worker must be a regular package file")
        request = json.dumps(
            {
                "schemaVersion": 1,
                "modelName": str(model_source),
                "labels": self.labels,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        environment = os.environ.copy()
        environment.pop("PYTHONHOME", None)
        environment.pop("PYTHONPATH", None)
        try:
            result = subprocess.run(
                [sys.executable, "-I", str(worker_path)],
                input=request,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=_TEXT_WORKER_TIMEOUT_SEC,
                env=environment,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("text-feature worker timed out") from exc
        if result.returncode != 0:
            details = result.stderr.decode("utf-8", errors="replace")[-1000:].strip()
            raise RuntimeError(
                f"text-feature worker failed with exit {result.returncode}: {details}"
            )
        if not result.stdout or len(result.stdout) > _MAX_TEXT_WORKER_RESPONSE_BYTES:
            raise RuntimeError("text-feature worker response size is invalid")
        try:
            payload = json.loads(result.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("text-feature worker response is not valid JSON") from exc
        return self._validate_text_worker_payload(payload, self.labels)

    def _load_mps_vision_model(self, torch, transformers, model_source: Path):
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise RuntimeError("safetensors is required for low-memory MPS loading") from exc

        checkpoint = model_source / "model.safetensors"
        if not checkpoint.is_file():
            raise RuntimeError("low-memory MPS loading requires model.safetensors")
        if (model_source / "model.safetensors.index.json").exists():
            raise RuntimeError("sharded safetensors are not supported by the MPS loader")

        AutoConfig = getattr(transformers, "AutoConfig", None)
        if AutoConfig is None:
            raise RuntimeError("transformers AutoConfig is unavailable")
        config = AutoConfig.from_pretrained(str(model_source))
        model_type = getattr(config, "model_type", None)
        if model_type == "siglip":
            vision_cls = getattr(transformers, "SiglipVisionModel", None)
        elif model_type == "siglip2":
            vision_cls = getattr(transformers, "Siglip2VisionModel", None)
        else:
            vision_cls = None
        if vision_cls is None or not hasattr(config, "vision_config"):
            raise RuntimeError(f"unsupported MPS vision model type: {model_type!r}")

        with torch.device("meta"):
            vision_model = vision_cls(config.vision_config)
        vision_model = vision_model.to(dtype=self._inference_dtype)
        expected = vision_model.state_dict(keep_vars=True)

        with safe_open(checkpoint, framework="pt", device="cpu") as weights:
            checkpoint_keys = set(weights.keys())
            vision_keys = {
                key for key in checkpoint_keys if key.startswith("vision_model.")
            }
            if set(expected) == vision_keys:
                checkpoint_key_by_model_key = {key: key for key in expected}
            elif {f"vision_model.{key}" for key in expected} == vision_keys:
                checkpoint_key_by_model_key = {
                    key: f"vision_model.{key}" for key in expected
                }
            else:
                raise RuntimeError("checkpoint vision tensor set does not match model config")
            if not any(key.startswith("text_model.") for key in checkpoint_keys):
                raise RuntimeError("checkpoint text tower is missing")
            if not {"logit_scale", "logit_bias"}.issubset(checkpoint_keys):
                raise RuntimeError("checkpoint SigLIP scoring scalars are missing")
            for model_key, target in expected.items():
                checkpoint_key = checkpoint_key_by_model_key[model_key]
                if tuple(weights.get_slice(checkpoint_key).get_shape()) != tuple(
                    target.shape
                ):
                    raise RuntimeError(
                        f"checkpoint tensor shape mismatch: {checkpoint_key}"
                    )

            vision_model.to_empty(device=self._device)
            buffers = dict(vision_model.named_buffers())
            if model_type == "siglip":
                position_keys = [
                    key
                    for key in buffers
                    if key.endswith("embeddings.position_ids")
                ]
                if len(position_keys) != 1 or len(buffers) != 1:
                    raise RuntimeError("SigLIP vision buffer set is unsupported")
                position_key = position_keys[0]
                position_ids = buffers[position_key]
                if position_ids.ndim != 2 or position_ids.shape[0] != 1:
                    raise RuntimeError("SigLIP position-id buffer shape is invalid")
                position_ids.copy_(
                    torch.arange(position_ids.shape[1], device=self._device).reshape(
                        position_ids.shape
                    )
                )
            elif buffers:
                raise RuntimeError("SigLIP2 vision buffers are unsupported")
            expected = vision_model.state_dict(keep_vars=True)
            with torch.no_grad():
                for model_key in sorted(expected):
                    checkpoint_key = checkpoint_key_by_model_key[model_key]
                    source = weights.get_tensor(checkpoint_key)
                    expected[model_key].copy_(source)
                    del source
                torch.mps.synchronize()

        torch.mps.empty_cache()
        return vision_model.eval()

    def _load_mps_scoring_scalars(self, torch, model_source: Path):
        from safetensors import safe_open

        checkpoint = model_source / "model.safetensors"
        with safe_open(checkpoint, framework="pt", device="cpu") as weights:
            checkpoint_keys = set(weights.keys())
            if not {"logit_scale", "logit_bias"}.issubset(checkpoint_keys):
                raise RuntimeError("checkpoint SigLIP scoring scalars are missing")
            logit_scale = weights.get_tensor("logit_scale").to(
                dtype=self._inference_dtype
            )
            logit_bias = weights.get_tensor("logit_bias").to(
                dtype=self._inference_dtype
            )
        if (
            logit_scale.numel() != 1
            or logit_bias.numel() != 1
            or not bool(torch.isfinite(logit_scale).all())
            or not bool(torch.isfinite(logit_bias).all())
        ):
            raise RuntimeError("checkpoint SigLIP scoring scalar is invalid")
        return logit_scale, logit_bias

    def __init__(
        self,
        enabled: bool,
        model_name: str,
        labels: list[str],
        anomaly_labels: list[str],
        threshold: float,
        device_preference: str = "auto",
    ):
        self.enabled = enabled
        self.ready = False
        self.labels = [label.strip() for label in labels if label and label.strip()]
        self.anomaly_labels = {label.strip().lower() for label in anomaly_labels if label and label.strip()}
        self.threshold = threshold
        self.model_name = model_name
        self.device_preference = normalize_siglip_device(device_preference, default="auto")
        self._model = None
        self._processor = None
        self._device = None
        self._inference_dtype = None
        self._mps_text_features = None
        self._mps_logit_scale = None
        self._mps_logit_bias = None
        self._loaded_model_source: str | None = None

        if not self.enabled:
            return
        if not self.labels:
            log.warning("Zero-shot enabled but labels are empty. Disabling.")
            self.enabled = False
            return

        try:
            import torch
            import transformers
            from PIL import Image
        except Exception as exc:  # noqa: BLE001 - optional dependencies vary.
            log.warning("Zero-shot dependencies not available: %s", exc)
            self.enabled = False
            return

        AutoModel = getattr(transformers, "AutoModel", None)
        if AutoModel is None:
            log.warning("Zero-shot dependencies not available: transformers AutoModel missing")
            self.enabled = False
            return

        device = self._resolve_device(torch, self.device_preference)
        self._device = device
        model = None

        try:
            if device == "mps":
                self._inference_dtype = torch.float16
                model_source, source_evidence = self._resolve_mps_model_source()
                text_rows = self._load_isolated_text_features(model_source)
                self._processor = self._load_image_processor(
                    transformers, model_source
                )
                logit_scale, logit_bias = self._load_mps_scoring_scalars(
                    torch, model_source
                )
                text_features = torch.tensor(
                    text_rows,
                    device=device,
                    dtype=self._inference_dtype,
                )
                norms = text_features.norm(dim=-1, keepdim=True)
                if not bool(torch.isfinite(norms).all()) or bool((norms <= 0).any()):
                    raise RuntimeError("text-feature norms are invalid")
                self._mps_text_features = text_features / norms
                self._mps_logit_scale = logit_scale.to(
                    device=device,
                    dtype=self._inference_dtype,
                ).exp()
                self._mps_logit_bias = logit_bias.to(
                    device=device,
                    dtype=self._inference_dtype,
                )
                del text_rows, text_features, norms, logit_scale, logit_bias
                gc.collect()
                torch.mps.empty_cache()
                model = self._load_mps_vision_model(
                    torch,
                    transformers,
                    model_source,
                )
                self._model = model
                self._loaded_model_source = source_evidence
            else:
                model = AutoModel.from_pretrained(self.model_name).to(device)
                self._model = model.eval()
                self._processor = self._load_processor(transformers)
            self._torch = torch
            self._Image = Image
            if device != "mps":
                candidate = Path(self.model_name).expanduser()
                try:
                    metadata = candidate.lstat()
                    if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(
                        metadata.st_mode
                    ):
                        self._loaded_model_source = str(candidate.resolve(strict=True))
                except OSError:
                    # A remote Hugging Face identifier has no authenticated local
                    # resolver identity and is intentionally not CONFIG evidence.
                    self._loaded_model_source = None
            self.ready = True
            log.info(
                "Zero-shot model loaded: %s (device=%s, preference=%s)",
                self.model_name,
                device,
                self.device_preference,
            )
        except Exception as exc:  # noqa: BLE001 - model loader is third-party.
            log.warning("Failed to load zero-shot model '%s': %s", self.model_name, exc)
            self._model = None
            model = None
            self._processor = None
            self._mps_text_features = None
            self._mps_logit_scale = None
            self._mps_logit_bias = None
            gc.collect()
            if device == "mps":
                try:
                    torch.mps.empty_cache()
                except Exception:  # noqa: BLE001 - preserve the original failure.
                    pass
            self.enabled = False

    def loaded_model_source(self) -> str | None:
        """Exact local directory handed to the successful model loader, if any."""

        if not self.enabled or not self.ready:
            return None
        return self._loaded_model_source

    @staticmethod
    def _resolve_device(torch_module, preference: str) -> str:
        mode = normalize_siglip_device(preference, default="auto")

        has_mps = bool(getattr(getattr(torch_module, "backends", None), "mps", None))
        mps_available = has_mps and bool(torch_module.backends.mps.is_available())
        cuda_available = bool(torch_module.cuda.is_available())

        if mode == "auto":
            if mps_available:
                return "mps"
            if cuda_available:
                return "cuda"
            return "cpu"

        if mode == "mps":
            if mps_available:
                return "mps"
            log.warning("Requested device=mps but MPS is unavailable. Falling back to cpu.")
            return "cpu"

        if mode == "cuda":
            if cuda_available:
                return "cuda"
            log.warning("Requested device=cuda but CUDA is unavailable. Falling back to cpu.")
            return "cpu"

        return "cpu"

    def classify(self, frame_rgb) -> dict | None:
        if not self.enabled or not self.ready:
            return None

        try:
            image = self._Image.fromarray(frame_rgb)
            if self._mps_text_features is not None:
                inputs = self._processor(images=image, return_tensors="pt")
            else:
                texts = [f"This is a photo of {label}." for label in self.labels]
                inputs = self._processor(
                    text=texts,
                    images=image,
                    padding="max_length",
                    max_length=64,
                    return_tensors="pt",
                )
            inputs = {
                key: (
                    value.to(device=self._device, dtype=self._inference_dtype)
                    if self._inference_dtype is not None
                    and value.is_floating_point()
                    else value.to(self._device)
                )
                for key, value in inputs.items()
            }
            with self._torch.no_grad():
                if self._mps_text_features is not None:
                    image_features = self._model(**inputs)
                    image_features = self._pooled_feature_tensor(image_features)
                    image_features = image_features / image_features.norm(
                        dim=-1, keepdim=True
                    )
                    logits = image_features @ self._mps_text_features.T
                    logits = (
                        logits * self._mps_logit_scale + self._mps_logit_bias
                    )
                else:
                    outputs = self._model(**inputs)
                    if (
                        hasattr(outputs, "logits_per_image")
                        and outputs.logits_per_image is not None
                    ):
                        logits = outputs.logits_per_image
                    else:
                        image_features = self._model.get_image_features(
                            **{
                                key: inputs[key]
                                for key in ("pixel_values",)
                                if key in inputs
                            }
                        )
                        text_features = self._model.get_text_features(
                            **{
                                key: inputs[key]
                                for key in ("input_ids", "attention_mask")
                                if key in inputs
                            }
                        )
                        image_features = self._pooled_feature_tensor(image_features)
                        text_features = self._pooled_feature_tensor(text_features)
                        image_features = image_features / image_features.norm(
                            dim=-1, keepdim=True
                        )
                        text_features = text_features / text_features.norm(
                            dim=-1, keepdim=True
                        )
                        logits = image_features @ text_features.T

            probs = self._torch.sigmoid(logits).squeeze(0).tolist()
        except Exception as exc:  # noqa: BLE001 - inference backend is third-party.
            log.warning("Zero-shot inference failed: %s", exc)
            return None

        if not probs:
            return None

        scored = list(zip(self.labels, probs))
        scored.sort(key=lambda item: item[1], reverse=True)
        labels = [item[0] for item in scored]
        scores = [float(item[1]) for item in scored]
        return {
            "label": labels[0],
            "score": scores[0],
            "labels": labels,
            "scores": scores,
        }

    def is_anomaly(self, frame_rgb):
        result = self.classify(frame_rgb)
        if not result:
            return False, None
        label = result["label"].lower()
        score = result["score"]
        if label in self.anomaly_labels and score >= self.threshold:
            return True, result
        return False, result
