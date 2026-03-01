from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from nuvion_app.config import load_env

try:
    import tritonclient.http as httpclient
except Exception as exc:  # pragma: no cover
    httpclient = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _truthy(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "y")


class TritonAnomalyClient:
    def __init__(self):
        load_env()
        if httpclient is None:
            raise ImportError(f"tritonclient is not available: {_IMPORT_ERROR}")

        self.url = os.getenv("NUVION_TRITON_URL", "localhost:8000")
        self.model_name = os.getenv("NUVION_TRITON_MODEL", "zsad")
        self.input_name = os.getenv("NUVION_TRITON_INPUT", "INPUT__0")
        self.output_name = os.getenv("NUVION_TRITON_OUTPUT", "OUTPUT__0")
        self.input_format = os.getenv("NUVION_TRITON_INPUT_FORMAT", "NHWC").upper()
        self.input_width = int(os.getenv("NUVION_TRITON_INPUT_WIDTH", "224"))
        self.input_height = int(os.getenv("NUVION_TRITON_INPUT_HEIGHT", "224"))
        self.input_dtype = os.getenv("NUVION_TRITON_INPUT_DTYPE", "FP32")
        self.scale = float(os.getenv("NUVION_TRITON_INPUT_SCALE", "255.0"))
        self.output_mode = os.getenv("NUVION_TRITON_OUTPUT_MODE", "score").lower()
        self.output_activation = os.getenv("NUVION_TRITON_OUTPUT_ACTIVATION", "sigmoid").lower()
        self.labels = [label.strip() for label in os.getenv("NUVION_TRITON_LABELS", "").split(",") if label.strip()]

        self.mode = os.getenv("NUVION_TRITON_MODE", "generic").lower()
        self.image_features_output = os.getenv("NUVION_TRITON_IMAGE_FEATURES_OUTPUT", "image_features")
        self.text_features_path = os.getenv("NUVION_TRITON_TEXT_FEATURES", "")
        self.text_temperature = max(float(os.getenv("NUVION_TRITON_TEXT_TEMPERATURE", "0.07")), 1e-6)
        self.anomaly_index = int(os.getenv("NUVION_TRITON_ANOMALY_INDEX", "1"))
        self.normalize_features = _truthy(os.getenv("NUVION_TRITON_NORMALIZE_FEATURES", "true"))

        self.client = httpclient.InferenceServerClient(url=self.url)

        self.text_features: np.ndarray | None = None
        if self.mode == "anomalyclip":
            self.text_features = self._load_text_features(self.text_features_path)
            class_count = int(self.text_features.shape[0])
            if not self.labels:
                self.labels = [f"class_{idx}" for idx in range(class_count)]
                if class_count == 2:
                    self.labels = ["normal", "defect"]
            elif len(self.labels) != class_count:
                raise ValueError(
                    "NUVION_TRITON_LABELS length must match text feature classes "
                    f"({len(self.labels)} != {class_count})"
                )
            if not (0 <= self.anomaly_index < class_count):
                raise ValueError(
                    f"NUVION_TRITON_ANOMALY_INDEX={self.anomaly_index} out of range [0, {class_count - 1}]"
                )

    def _load_text_features(self, path_str: str) -> np.ndarray:
        path = Path(path_str).expanduser() if path_str else None
        if not path:
            raise ValueError("NUVION_TRITON_TEXT_FEATURES is required when NUVION_TRITON_MODE=anomalyclip")
        if not path.exists():
            raise FileNotFoundError(f"Text feature file not found: {path}")

        features = np.asarray(np.load(path), dtype=np.float32)
        features = np.squeeze(features)
        if features.ndim == 1:
            features = np.expand_dims(features, axis=0)
        if features.ndim != 2:
            raise ValueError(f"Unsupported text feature shape: {features.shape}")

        # Some exports are transposed as [D, C] instead of [C, D].
        if features.shape[1] <= 16 and features.shape[0] > features.shape[1]:
            features = features.T

        if self.normalize_features:
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            features = features / np.clip(norms, 1e-12, None)

        return features

    def _preprocess(self, frame_rgb: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame_rgb, (self.input_width, self.input_height))
        arr = resized.astype(np.float32) / self.scale
        if self.input_format == "NCHW":
            arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, axis=0)
        return arr

    def _activate(self, scores: np.ndarray) -> np.ndarray:
        if self.output_activation == "softmax":
            shifted = scores - np.max(scores)
            exps = np.exp(shifted)
            denom = float(np.sum(exps))
            if denom <= 0:
                return exps
            return exps / denom
        if self.output_activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-scores))
        return scores

    def infer(self, frame_rgb: np.ndarray) -> np.ndarray:
        arr = self._preprocess(frame_rgb)
        input_tensor = httpclient.InferInput(self.input_name, arr.shape, self.input_dtype)
        input_tensor.set_data_from_numpy(arr)

        output_name = self.image_features_output if self.mode == "anomalyclip" else self.output_name
        output = httpclient.InferRequestedOutput(output_name)
        response = self.client.infer(
            model_name=self.model_name,
            inputs=[input_tensor],
            outputs=[output],
        )
        result = response.as_numpy(output_name)
        if result is None:
            raise RuntimeError(f"No output received from Triton for '{output_name}'")
        return result

    def _predict_anomalyclip(self, image_features: np.ndarray) -> dict:
        if self.text_features is None:
            raise RuntimeError("text features are not initialized")

        image_vec = np.asarray(image_features, dtype=np.float32).reshape(-1)
        if self.normalize_features:
            image_norm = np.linalg.norm(image_vec)
            image_vec = image_vec / max(float(image_norm), 1e-12)

        class_count, dim = self.text_features.shape
        if image_vec.shape[0] != dim:
            raise RuntimeError(
                "Feature dim mismatch between Triton image feature and text features "
                f"({image_vec.shape[0]} != {dim})"
            )

        logits = (self.text_features @ image_vec) / self.text_temperature
        shifted = logits - np.max(logits)
        exps = np.exp(shifted)
        probs = exps / np.clip(np.sum(exps), 1e-12, None)

        top_idx = int(np.argmax(probs))
        anomaly_idx = self.anomaly_index if 0 <= self.anomaly_index < class_count else min(1, class_count - 1)
        anomaly_score = float(probs[anomaly_idx])

        return {
            "label": self.labels[top_idx],
            "score": anomaly_score,
            "scores": probs.tolist(),
            "predicted_label": self.labels[top_idx],
            "predicted_score": float(probs[top_idx]),
            "anomaly_label": self.labels[anomaly_idx],
            "anomaly_index": anomaly_idx,
            "mode": "anomalyclip",
        }

    def predict(self, frame_rgb: np.ndarray) -> dict:
        result = self.infer(frame_rgb)

        if self.mode == "anomalyclip":
            return self._predict_anomalyclip(result)

        flat = result.reshape(-1)
        if self.output_mode == "score":
            return {"label": "ANOMALY", "score": float(flat[0]), "mode": "generic"}

        scores = self._activate(flat)
        scores_list = scores.tolist()
        if not self.labels or len(self.labels) != len(scores_list):
            top_idx = int(np.argmax(scores))
            return {
                "label": f"class_{top_idx}",
                "score": float(scores[top_idx]),
                "scores": scores_list,
                "mode": "generic",
            }

        top_idx = int(np.argmax(scores))
        return {
            "label": self.labels[top_idx],
            "score": float(scores[top_idx]),
            "scores": scores_list,
            "mode": "generic",
        }
