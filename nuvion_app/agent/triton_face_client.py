from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from functools import lru_cache

import numpy as np

from nuvion_app.config import load_env

try:
    import tritonclient.http as httpclient
except Exception as exc:  # pragma: no cover
    httpclient = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

try:
    import cv2
except Exception as exc:  # pragma: no cover
    cv2 = None
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None

log = logging.getLogger(__name__)


def _parse_model_config(raw: dict | None) -> dict:
    if not isinstance(raw, dict):
        return {}
    nested = raw.get("config")
    if isinstance(nested, dict):
        return nested
    return raw


def _infer_layout_and_size(dims: list[int], declared_format: str) -> tuple[str, int, int] | None:
    if len(dims) != 3:
        return None

    def _to_positive(value: int) -> int:
        return int(value) if int(value) > 0 else -1

    d0, d1, d2 = (_to_positive(dims[0]), _to_positive(dims[1]), _to_positive(dims[2]))

    declared = (declared_format or "").strip().upper()
    if declared in {"FORMAT_NCHW", "NCHW"}:
        return ("NCHW", d2, d1) if d1 > 0 and d2 > 0 else None
    if declared in {"FORMAT_NHWC", "NHWC"}:
        return ("NHWC", d1, d0) if d0 > 0 and d1 > 0 else None

    if d0 in {1, 3, 4} and d1 > 0 and d2 > 0:
        return ("NCHW", d2, d1)
    if d2 in {1, 3, 4} and d0 > 0 and d1 > 0:
        return ("NHWC", d1, d0)
    return None


@dataclass(frozen=True)
class FaceDetection:
    x: int
    y: int
    width: int
    height: int
    score: float


class TritonFaceClient:
    def __init__(self) -> None:
        load_env()
        if httpclient is None:
            raise ImportError(f"tritonclient is not available: {_IMPORT_ERROR}")

        self.url = os.getenv("NUVION_TRITON_URL", "localhost:8000")
        self.model_name = (os.getenv("NUVION_FACE_TRACKING_MODEL", "face_detector") or "face_detector").strip() or "face_detector"
        self.model_kind = (os.getenv("NUVION_FACE_TRACKING_MODEL_KIND", "ultraface_rfb_640") or "ultraface_rfb_640").strip().lower()
        self.input_name = (os.getenv("NUVION_FACE_TRACKING_INPUT_NAME", "input") or "input").strip() or "input"
        self.boxes_output = (os.getenv("NUVION_FACE_TRACKING_BOXES_OUTPUT", "boxes") or "boxes").strip() or "boxes"
        self.scores_output = (os.getenv("NUVION_FACE_TRACKING_SCORES_OUTPUT", "scores") or "scores").strip() or "scores"
        self.num_output = (os.getenv("NUVION_FACE_TRACKING_NUM_DETECTIONS_OUTPUT", "num_detections") or "num_detections").strip() or "num_detections"
        self.box_format = (os.getenv("NUVION_FACE_TRACKING_BOX_FORMAT", "xyxy_norm") or "xyxy_norm").strip().lower() or "xyxy_norm"
        self.input_format = (os.getenv("NUVION_FACE_TRACKING_INPUT_FORMAT", "NCHW") or "NCHW").strip().upper() or "NCHW"
        self.input_width = max(int(os.getenv("NUVION_FACE_TRACKING_INPUT_WIDTH", "640") or "640"), 1)
        self.input_height = max(int(os.getenv("NUVION_FACE_TRACKING_INPUT_HEIGHT", "480") or "480"), 1)
        self.input_dtype = (os.getenv("NUVION_FACE_TRACKING_INPUT_DTYPE", "FP32") or "FP32").strip().upper() or "FP32"
        self.input_scale = max(float(os.getenv("NUVION_FACE_TRACKING_INPUT_SCALE", "128.0") or "128.0"), 1e-6)
        self.input_mean = np.asarray(
            [float(part.strip()) for part in (os.getenv("NUVION_FACE_TRACKING_INPUT_MEAN", "127,127,127") or "127,127,127").split(",")[:3]],
            dtype=np.float32,
        )
        if self.input_mean.size != 3:
            self.input_mean = np.asarray([127.0, 127.0, 127.0], dtype=np.float32)
        self.threshold = float(os.getenv("NUVION_FACE_TRACKING_THRESHOLD", "0.7") or "0.7")
        self.max_detections = max(int(os.getenv("NUVION_FACE_TRACKING_MAX_DETECTIONS", "8") or "8"), 1)
        self.nms_iou = float(os.getenv("NUVION_FACE_TRACKING_NMS_IOU", "0.3") or "0.3")

        self.client = httpclient.InferenceServerClient(url=self.url)
        self._sync_output_names_from_metadata()
        self._sync_input_shape_from_config()

    def _sync_input_shape_from_config(self) -> None:
        try:
            raw_config = self.client.get_model_config(model_name=self.model_name)
        except Exception:
            return
        model_config = _parse_model_config(raw_config)
        inputs = model_config.get("input")
        if not isinstance(inputs, list) or not inputs:
            return

        selected = None
        for item in inputs:
            if isinstance(item, dict) and str(item.get("name", "")) == self.input_name:
                selected = item
                break
        if selected is None:
            selected = inputs[0] if isinstance(inputs[0], dict) else None
        if selected is None:
            return

        dims_raw = selected.get("dims")
        if not isinstance(dims_raw, list):
            return
        try:
            dims = [int(v) for v in dims_raw]
        except Exception:
            return

        inferred = _infer_layout_and_size(dims, str(selected.get("format", "")))
        if inferred is None:
            return
        self.input_format, self.input_width, self.input_height = inferred

    def _preprocess(self, frame_rgb: np.ndarray) -> np.ndarray:
        if cv2 is None:
            raise ImportError(f"opencv-python is required for Triton face tracking preprocessing: {_CV2_IMPORT_ERROR}")
        resized = cv2.resize(frame_rgb, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        tensor = (resized.astype(np.float32) - self.input_mean.reshape((1, 1, 3))) / self.input_scale
        if self.input_format == "NCHW":
            tensor = np.transpose(tensor, (2, 0, 1))
        return np.expand_dims(tensor, axis=0)

    @staticmethod
    def _reshape_boxes(raw: np.ndarray | None) -> np.ndarray:
        if raw is None:
            return np.empty((0, 4), dtype=np.float32)
        boxes = np.asarray(raw, dtype=np.float32)
        if boxes.ndim == 3:
            boxes = boxes[0]
        if boxes.ndim == 1:
            if boxes.size % 4 != 0:
                return np.empty((0, 4), dtype=np.float32)
            boxes = boxes.reshape((-1, 4))
        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            return np.empty((0, 4), dtype=np.float32)
        return boxes

    @staticmethod
    def _reshape_scores(raw: np.ndarray | None) -> np.ndarray:
        if raw is None:
            return np.empty((0,), dtype=np.float32)
        scores = np.asarray(raw, dtype=np.float32)
        if scores.ndim == 3 and scores.shape[-1] == 2:
            scores = scores[0, :, 1]
        elif scores.ndim >= 2:
            scores = scores.reshape((-1,))
        return scores

    @staticmethod
    def _resolve_num_detections(raw: np.ndarray | None, boxes: np.ndarray, scores: np.ndarray) -> int:
        if raw is None:
            return int(min(len(boxes), len(scores)))
        arr = np.asarray(raw)
        if arr.size == 0:
            return int(min(len(boxes), len(scores)))
        try:
            return int(arr.reshape((-1,))[0])
        except Exception:
            return int(min(len(boxes), len(scores)))

    def _decode_box(self, box: np.ndarray, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
        x1: float
        y1: float
        x2: float
        y2: float

        if self.box_format.startswith("xywh"):
            x1, y1, w, h = [float(v) for v in box]
            x2 = x1 + w
            y2 = y1 + h
        elif self.box_format.startswith("yxyx"):
            y1, x1, y2, x2 = [float(v) for v in box]
        else:
            x1, y1, x2, y2 = [float(v) for v in box]

        normalized = self.box_format.endswith("_norm")
        if not normalized and max(x1, y1, x2, y2) <= 1.5 and min(x1, y1, x2, y2) >= -0.1:
            normalized = True

        if normalized:
            x1 *= frame_width
            x2 *= frame_width
            y1 *= frame_height
            y2 *= frame_height
        else:
            x_scale = frame_width / float(self.input_width)
            y_scale = frame_height / float(self.input_height)
            x1 *= x_scale
            x2 *= x_scale
            y1 *= y_scale
            y2 *= y_scale

        left = max(int(round(min(x1, x2))), 0)
        top = max(int(round(min(y1, y2))), 0)
        right = min(int(round(max(x1, x2))), frame_width)
        bottom = min(int(round(max(y1, y2))), frame_height)
        return left, top, max(right - left, 0), max(bottom - top, 0)

    @staticmethod
    @lru_cache(maxsize=8)
    def _ultraface_priors(input_width: int, input_height: int) -> np.ndarray:
        min_boxes = ((10.0, 16.0, 24.0), (32.0, 48.0), (64.0, 96.0), (128.0, 192.0, 256.0))
        strides = (8.0, 16.0, 32.0, 64.0)
        priors: list[list[float]] = []
        for min_sizes, stride in zip(min_boxes, strides):
            feature_map_w = int(np.ceil(input_width / stride))
            feature_map_h = int(np.ceil(input_height / stride))
            for y in range(feature_map_h):
                for x in range(feature_map_w):
                    x_center = (x + 0.5) * stride / input_width
                    y_center = (y + 0.5) * stride / input_height
                    for min_box in min_sizes:
                        priors.append(
                            [
                                x_center,
                                y_center,
                                min_box / input_width,
                                min_box / input_height,
                            ]
                        )
        return np.asarray(priors, dtype=np.float32)

    def _decode_ultraface_boxes(self, raw_boxes: np.ndarray, frame_width: int, frame_height: int) -> np.ndarray:
        priors = self._ultraface_priors(self.input_width, self.input_height)
        boxes = raw_boxes[: len(priors)]
        center_variance = 0.1
        size_variance = 0.2
        decoded = np.concatenate(
            [
                priors[:, :2] + boxes[:, :2] * center_variance * priors[:, 2:],
                priors[:, 2:] * np.exp(boxes[:, 2:] * size_variance),
            ],
            axis=1,
        )
        decoded[:, :2] -= decoded[:, 2:] / 2
        decoded[:, 2:] += decoded[:, :2]
        decoded[:, 0::2] *= frame_width
        decoded[:, 1::2] *= frame_height
        return decoded

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float, limit: int) -> list[int]:
        if boxes.size == 0:
            return []
        order = np.argsort(scores)[::-1]
        keep: list[int] = []
        while order.size > 0 and len(keep) < limit:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
            widths = np.maximum(0.0, xx2 - xx1)
            heights = np.maximum(0.0, yy2 - yy1)
            inter = widths * heights
            area_i = max((boxes[i, 2] - boxes[i, 0]), 0.0) * max((boxes[i, 3] - boxes[i, 1]), 0.0)
            area_rest = np.maximum(boxes[rest, 2] - boxes[rest, 0], 0.0) * np.maximum(boxes[rest, 3] - boxes[rest, 1], 0.0)
            union = area_i + area_rest - inter
            iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
            order = rest[iou <= threshold]
        return keep

    def predict(self, frame_rgb: np.ndarray) -> list[FaceDetection]:
        frame_height, frame_width = frame_rgb.shape[:2]
        tensor = self._preprocess(frame_rgb)

        infer_input = httpclient.InferInput(self.input_name, list(tensor.shape), self.input_dtype)
        infer_input.set_data_from_numpy(tensor)
        outputs = [
            httpclient.InferRequestedOutput(self.boxes_output),
            httpclient.InferRequestedOutput(self.scores_output),
        ]
        if self.num_output:
            outputs.append(httpclient.InferRequestedOutput(self.num_output))

        response = self.client.infer(model_name=self.model_name, inputs=[infer_input], outputs=outputs)
        boxes = self._reshape_boxes(response.as_numpy(self.boxes_output))
        scores = self._reshape_scores(response.as_numpy(self.scores_output))

        if self.model_kind.startswith("ultraface"):
            decoded = self._decode_ultraface_boxes(boxes, frame_width, frame_height)
            scores = scores[: len(decoded)]
            mask = scores >= self.threshold
            filtered_boxes = decoded[mask]
            filtered_scores = scores[mask]
            keep = self._nms(filtered_boxes, filtered_scores, self.nms_iou, self.max_detections)
            detections: list[FaceDetection] = []
            for index in keep:
                x1, y1, x2, y2 = filtered_boxes[index]
                left = max(int(round(x1)), 0)
                top = max(int(round(y1)), 0)
                right = min(int(round(x2)), frame_width)
                bottom = min(int(round(y2)), frame_height)
                width = max(right - left, 0)
                height = max(bottom - top, 0)
                if width <= 0 or height <= 0:
                    continue
                detections.append(
                    FaceDetection(
                        x=left,
                        y=top,
                        width=width,
                        height=height,
                        score=float(filtered_scores[index]),
                    )
                )
            return detections

        count = self._resolve_num_detections(response.as_numpy(self.num_output), boxes, scores)
        count = max(0, min(count, len(boxes), len(scores)))

        detections: list[FaceDetection] = []
        order = np.argsort(scores[:count])[::-1]
        for index in order:
            score = float(scores[index])
            if score < self.threshold:
                continue
            x, y, width, height = self._decode_box(boxes[index], frame_width, frame_height)
            if width <= 0 or height <= 0:
                continue
            detections.append(FaceDetection(x=x, y=y, width=width, height=height, score=score))
            if len(detections) >= self.max_detections:
                break
        return detections
    def _sync_output_names_from_metadata(self) -> None:
        try:
            metadata = self.client.get_model_metadata(model_name=self.model_name)
        except Exception:
            return
        outputs = metadata.get("outputs", []) if isinstance(metadata, dict) else []
        names = [str(item.get("name", "")).strip() for item in outputs if isinstance(item, dict)]
        names = [name for name in names if name]
        if not names:
            return
        if self.boxes_output not in names:
            if "boxes" in names:
                self.boxes_output = "boxes"
            elif len(names) >= 2:
                self.boxes_output = names[-1]
        if self.scores_output not in names:
            if "scores" in names:
                self.scores_output = "scores"
            else:
                self.scores_output = names[0]
        if self.num_output not in names:
            self.num_output = ""
