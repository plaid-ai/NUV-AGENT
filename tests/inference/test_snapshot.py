from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from nuvion_app.inference.snapshot import LatestFrameBuffer
from nuvion_app.inference.snapshot import PNG_SIGNATURE
from nuvion_app.inference.snapshot import capture_and_upload_snapshot
from nuvion_app.inference.snapshot import encode_snapshot


class SnapshotTest(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((4, 5, 3), dtype=np.uint8)
        self.frame[..., 0] = 12
        self.frame[..., 1] = 34
        self.frame[..., 2] = 56

    def test_latest_frame_buffer_returns_copy(self) -> None:
        buffer = LatestFrameBuffer()
        buffer.remember(self.frame)

        copied = buffer.copy()
        self.assertIsNotNone(copied)
        self.assertIsNot(copied, self.frame)
        copied[0, 0, 0] = 255

        second_copy = buffer.copy()
        self.assertEqual(int(second_copy[0, 0, 0]), 12)

    def test_encode_snapshot_falls_back_to_png_when_jpeg_encoder_unavailable(self) -> None:
        with mock.patch("nuvion_app.inference.snapshot._encode_with_pillow", return_value=None):
            payload, content_type = encode_snapshot(self.frame, preferred_content_type="image/jpeg")

        self.assertEqual(content_type, "image/png")
        self.assertTrue(payload.startswith(PNG_SIGNATURE))

    def test_capture_and_upload_snapshot_returns_object_name_on_success(self) -> None:
        request_upload_url = mock.Mock(
            return_value={"objectName": "anomalies/1/snapshot.png", "uploadUrl": "https://example.com/upload"}
        )
        upload_bytes_to_url = mock.Mock(return_value=True)

        with mock.patch("nuvion_app.inference.snapshot._encode_with_pillow", return_value=None):
            object_name = capture_and_upload_snapshot(
                self.frame,
                request_upload_url=request_upload_url,
                upload_bytes_to_url=upload_bytes_to_url,
                preferred_content_type="image/jpeg",
            )

        self.assertEqual(object_name, "anomalies/1/snapshot.png")
        request_upload_url.assert_called_once_with("SNAPSHOT", "image/png")
        upload_url, payload, content_type = upload_bytes_to_url.call_args.args
        self.assertEqual(upload_url, "https://example.com/upload")
        self.assertEqual(content_type, "image/png")
        self.assertTrue(payload.startswith(PNG_SIGNATURE))

    def test_capture_and_upload_snapshot_returns_none_when_upload_fails(self) -> None:
        request_upload_url = mock.Mock(
            return_value={"objectName": "anomalies/1/snapshot.png", "uploadUrl": "https://example.com/upload"}
        )
        upload_bytes_to_url = mock.Mock(return_value=False)

        with mock.patch("nuvion_app.inference.snapshot._encode_with_pillow", return_value=None):
            object_name = capture_and_upload_snapshot(
                self.frame,
                request_upload_url=request_upload_url,
                upload_bytes_to_url=upload_bytes_to_url,
                preferred_content_type="image/jpeg",
            )

        self.assertIsNone(object_name)


if __name__ == "__main__":
    unittest.main()
