import gc
import logging
import stat
from pathlib import Path

from nuvion_app.runtime.inference_mode import normalize_siglip_device

log = logging.getLogger(__name__)


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
        model = None

        try:
            if device == "mps":
                # Apple unified-memory runners can reject the transient FP32
                # allocation before even a small SigLIP2 inference. Load the
                # checkpoint directly as FP16 with the low-memory loader so a
                # complete FP32 CPU copy never overlaps the MPS allocation.
                self._inference_dtype = torch.float16
                model = AutoModel.from_pretrained(
                    self.model_name,
                    dtype=self._inference_dtype,
                    low_cpu_mem_usage=True,
                )
            else:
                model = AutoModel.from_pretrained(self.model_name).to(device)
            self._model = model.eval()
            self._processor = self._load_processor(transformers)
            self._torch = torch
            self._Image = Image
            self._device = device
            if device == "mps":
                self._partition_mps_model()
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

    def _partition_mps_model(self) -> None:
        """Cache static text features and keep only the vision tower on MPS."""

        texts = [f"This is a photo of {label}." for label in self.labels]
        text_inputs = self._processor(
            text=texts,
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        )
        with self._torch.inference_mode():
            text_features = self._model.get_text_features(**text_inputs)
        text_features = self._pooled_feature_tensor(text_features)

        self._mps_logit_scale = self._model.logit_scale.detach().to(
            device=self._device,
            dtype=self._inference_dtype,
        ).exp()
        self._mps_logit_bias = self._model.logit_bias.detach().to(
            device=self._device,
            dtype=self._inference_dtype,
        )

        # Labels never change after construction. Releasing the 538 MiB text
        # tower leaves only the 177 MiB vision tower resident on constrained
        # Apple unified-memory devices while preserving SigLIP scoring semantics.
        self._model.text_model = None
        gc.collect()
        self._model.vision_model = self._model.vision_model.to(self._device).eval()
        text_features = text_features.to(
            device=self._device,
            dtype=self._inference_dtype,
        )
        self._mps_text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        )

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
                    image_features = self._model.get_image_features(**inputs)
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
