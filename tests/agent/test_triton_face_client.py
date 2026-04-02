from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from nuvion_app.agent import triton_face_client as face_client_module
from nuvion_app.agent.triton_face_client import TritonFaceClient


class _FakeResponse:
    def __init__(self, outputs: dict[str, np.ndarray]) -> None:
        self._outputs = outputs

    def as_numpy(self, name: str):
        return self._outputs.get(name)


class _FakeClient:
    def __init__(self, outputs: dict[str, np.ndarray]) -> None:
        self._outputs = outputs

    def is_model_ready(self, **_kwargs):
        return True

    def get_model_metadata(self, **_kwargs):
        return {"outputs": []}

    def infer(self, **_kwargs):
        return _FakeResponse(self._outputs)


class _FakeInferInput:
    def __init__(self, *_args, **_kwargs) -> None:
        self.tensor = None

    def set_data_from_numpy(self, tensor):
        self.tensor = tensor


class _FakeRequestedOutput:
    def __init__(self, name: str) -> None:
        self.name = name


class TritonFaceClientTest(unittest.TestCase):
    def test_init_fails_when_triton_model_is_not_ready(self) -> None:
        fake_http = type(
            "FakeHttpClient",
            (),
            {
                "InferenceServerClient": lambda **_kwargs: type(
                    "Client",
                    (),
                    {
                        "is_model_ready": staticmethod(lambda **_kw: False),
                        "get_model_metadata": staticmethod(lambda **_kw: {"outputs": []}),
                    },
                )(),
            },
        )

        with mock.patch.object(face_client_module, "httpclient", fake_http):
            with self.assertRaises(RuntimeError):
                TritonFaceClient()

    def test_ultraface_scores_use_face_channel_and_decode_boxes(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        priors = TritonFaceClient._ultraface_priors(640, 480)
        count = len(priors)
        boxes = np.zeros((1, count, 4), dtype=np.float32)
        scores = np.zeros((1, count, 2), dtype=np.float32)
        scores[0, 0, 1] = 0.95

        client = TritonFaceClient.__new__(TritonFaceClient)
        client.model_name = "face_detector"
        client.model_kind = "ultraface_rfb_640"
        client.input_width = 640
        client.input_height = 480
        client.threshold = 0.7
        client.max_detections = 8
        client.nms_iou = 0.3
        client.input_name = "input"
        client.input_dtype = "FP32"
        client.input_format = "NCHW"
        client.input_scale = 128.0
        client.input_mean = np.asarray([127.0, 127.0, 127.0], dtype=np.float32)
        client.boxes_output = "boxes"
        client.scores_output = "scores"
        client.num_output = ""
        client._preprocess = lambda _frame: np.zeros((1, 3, 480, 640), dtype=np.float32)
        client.client = _FakeClient({"boxes": boxes, "scores": scores})

        with mock.patch.object(
            face_client_module,
            "httpclient",
            type(
                "FakeHttpClient",
                (),
                {
                    "InferInput": _FakeInferInput,
                    "InferRequestedOutput": _FakeRequestedOutput,
                },
            ),
        ):
            detections = client.predict(frame)

        self.assertEqual(len(detections), 1)
        self.assertGreater(detections[0].score, 0.9)
        self.assertGreater(detections[0].width, 0)
        self.assertGreater(detections[0].height, 0)


if __name__ == "__main__":
    unittest.main()
