import logging
from typing import List, Optional

from nuvion_app.runtime.inference_mode import normalize_siglip_device

log = logging.getLogger(__name__)


class ZeroShotAnomalyDetector:
    @staticmethod
    def _format_exc(exc: Exception) -> str:
        return f"{exc.__class__.__name__}: {exc}"

    def _load_processor(self, transformers):
        attempts: list[str] = []

        AutoProcessor = getattr(transformers, "AutoProcessor", None)
        if AutoProcessor is not None:
            try:
                return AutoProcessor.from_pretrained(self.model_name)
            except Exception as exc:
                attempts.append(f"AutoProcessor.from_pretrained failed ({self._format_exc(exc)})")

        for class_name in ("Siglip2Processor", "SiglipProcessor"):
            processor_cls = getattr(transformers, class_name, None)
            if processor_cls is None:
                continue
            try:
                return processor_cls.from_pretrained(self.model_name)
            except Exception as exc:
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
            except Exception as exc:
                attempts.append(f"{class_name}.from_pretrained failed ({self._format_exc(exc)})")

        for class_name in ("GemmaTokenizerFast", "GemmaTokenizer", "AutoTokenizer"):
            tokenizer_cls = getattr(transformers, class_name, None)
            if tokenizer_cls is None:
                continue
            try:
                tokenizer = tokenizer_cls.from_pretrained(self.model_name)
                break
            except Exception as exc:
                attempts.append(f"{class_name}.from_pretrained failed ({self._format_exc(exc)})")

        if processor_cls is not None and image_processor is not None and tokenizer is not None:
            try:
                return processor_cls(image_processor=image_processor, tokenizer=tokenizer)
            except Exception as exc:
                attempts.append(f"{processor_cls.__name__}(image_processor, tokenizer) failed ({self._format_exc(exc)})")

        details = "; ".join(attempts) if attempts else "processor class not found"
        raise RuntimeError(f"Unable to initialize processor for '{self.model_name}': {details}")

    def __init__(
        self,
        enabled: bool,
        model_name: str,
        labels: List[str],
        anomaly_labels: List[str],
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
        except Exception as exc:
            log.warning("Zero-shot dependencies not available: %s", exc)
            self.enabled = False
            return

        AutoModel = getattr(transformers, "AutoModel", None)
        if AutoModel is None:
            log.warning("Zero-shot dependencies not available: transformers AutoModel missing")
            self.enabled = False
            return

        device = self._resolve_device(torch, self.device_preference)

        try:
            self._model = AutoModel.from_pretrained(self.model_name).to(device).eval()
            self._processor = self._load_processor(transformers)
            self._torch = torch
            self._Image = Image
            self._device = device
            self.ready = True
            log.info(
                "Zero-shot model loaded: %s (device=%s, preference=%s)",
                self.model_name,
                device,
                self.device_preference,
            )
        except Exception as exc:
            log.warning("Failed to load zero-shot model '%s': %s", self.model_name, exc)
            self.enabled = False

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

    def classify(self, frame_rgb) -> Optional[dict]:
        if not self.enabled or not self.ready:
            return None

        try:
            image = self._Image.fromarray(frame_rgb)
            texts = [f"This is a photo of {label}." for label in self.labels]
            inputs = self._processor(
                text=texts,
                images=image,
                padding="max_length",
                max_length=64,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with self._torch.no_grad():
                outputs = self._model(**inputs)

            if hasattr(outputs, "logits_per_image") and outputs.logits_per_image is not None:
                logits = outputs.logits_per_image
            else:
                image_features = self._model.get_image_features(**{k: inputs[k] for k in ("pixel_values",) if k in inputs})
                text_features = self._model.get_text_features(**{k: inputs[k] for k in ("input_ids", "attention_mask") if k in inputs})
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                logits = image_features @ text_features.T

            probs = self._torch.sigmoid(logits).squeeze(0).tolist()
        except Exception as exc:
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
