from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nuvion_app import model_store


class ModelStoreSelfHealTest(unittest.TestCase):
    def test_signed_url_error_detection(self) -> None:
        self.assertTrue(model_store._is_signed_url_refresh_error(RuntimeError("HTTP Error 400: Bad Request")))
        self.assertTrue(model_store._is_signed_url_refresh_error(RuntimeError("request has expired")))
        self.assertFalse(model_store._is_signed_url_refresh_error(RuntimeError("connection reset by peer")))

    def test_pull_model_from_server_refreshes_presign_and_recovers(self) -> None:
        bad_data = {
            "data": {
                "artifacts": [
                    {
                        "key": "text_features",
                        "url": "https://example.com/bad/text_features.npy",
                        "path": "nuvion/anomalyclip/v0001/source/text_features.npy",
                        "sha256": "a" * 64,
                        "sizeBytes": 8,
                        "expiresAt": "2026-01-01T00:00:00Z",
                    },
                    {
                        "key": "manifest",
                        "url": "https://example.com/bad/manifest.json",
                        "path": "nuvion/anomalyclip/v0001/source/gcs_manifest.json",
                        "sha256": "b" * 64,
                        "sizeBytes": 8,
                        "expiresAt": "2026-01-01T00:00:00Z",
                    },
                ]
            }
        }
        good_data = {
            "data": {
                "artifacts": [
                    {
                        "key": "text_features",
                        "url": "https://example.com/good/text_features.npy",
                        "path": "nuvion/anomalyclip/v0001/source/text_features.npy",
                        "sha256": "a" * 64,
                        "sizeBytes": 8,
                        "expiresAt": "2026-01-01T00:00:00Z",
                    },
                    {
                        "key": "manifest",
                        "url": "https://example.com/good/manifest.json",
                        "path": "nuvion/anomalyclip/v0001/source/gcs_manifest.json",
                        "sha256": "b" * 64,
                        "sizeBytes": 8,
                        "expiresAt": "2026-01-01T00:00:00Z",
                    },
                ]
            }
        }

        def fake_download(
            *,
            url: str,
            dst_path: Path,
            timeout: int = 120,
            max_retries: int = 3,
            progress_label: str | None = None,
        ) -> None:
            if "/bad/" in url:
                raise RuntimeError(f"Download failed after 3 attempts: {url} (HTTP Error 400: Bad Request)")
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            dst_path.write_bytes(b"12345678")

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(model_store, "_http_json", side_effect=[bad_data, good_data]) as http_mock:
                with mock.patch.object(model_store, "_download_http_file", side_effect=fake_download) as dl_mock:
                    with mock.patch.object(model_store, "_validate_download_integrity", return_value=None):
                        target_dir, data = model_store.pull_model_from_server(
                            server_base_url="https://api.nuvion-dev.plaidai.io",
                            pointer="anomalyclip/prod",
                            profile="light",
                            local_dir=tmp,
                            ttl_seconds=300,
                            access_token="test-token",
                        )
            self.assertEqual(http_mock.call_count, 2)
            self.assertGreaterEqual(dl_mock.call_count, 2)
            self.assertEqual(len(data["artifacts"]), 2)
            self.assertTrue((Path(target_dir) / "onnx" / "text_features.npy").exists())
            self.assertTrue((Path(target_dir) / "metadata" / "gcs_manifest.json").exists())

if __name__ == "__main__":
    unittest.main()
