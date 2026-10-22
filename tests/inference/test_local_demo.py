from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nuvion_app.inference.local_demo import LocalDemoServer
from nuvion_app.inference.local_demo import preview_branch


class LocalDemoServerTest(unittest.TestCase):
    def test_publish_and_snapshot_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server = LocalDemoServer(tmp)
            self.assertIsNone(server.snapshot()[0])
            server.publish(b"jpeg-bytes")
            jpeg, sequence, age = server.snapshot()
            self.assertEqual(jpeg, b"jpeg-bytes")
            self.assertEqual(sequence, 1)
            self.assertLess(age, 1.0)

    def test_snapshot_expires_after_three_seconds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server = LocalDemoServer(tmp)
            with mock.patch("time.monotonic", return_value=100.0):
                server.publish(b"jpeg-bytes")
                jpeg, _sequence, age = server.snapshot()
            self.assertEqual(jpeg, b"jpeg-bytes")  # published and read at same fake time

            # Simulate expiry by moving fake clock forward past the 3s window.
            with mock.patch("time.monotonic", return_value=1000.0):
                jpeg, _sequence, age = server.snapshot()
            self.assertIsNone(jpeg)
            self.assertGreaterEqual(age, 3.0)

    def test_rejects_oversized_jpeg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server = LocalDemoServer(tmp)
            server.publish(b"x" * (8 * 1024 * 1024 + 1))
            self.assertIsNone(server.snapshot()[0])


class LocalDemoHandlerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        assets = Path(self._tmp.name)
        (assets / "index.html").write_text("<html>demo</html>")
        (assets / "nested" / "frame.js").parent.mkdir(parents=True)
        (assets / "nested" / "frame.js").write_text("console.log('demo');")
        self.server = LocalDemoServer(assets, port=0)
        self.server.start()
        self.base = f"http://localhost:{self.server.port}"

    def tearDown(self) -> None:
        self.server.close()
        self._tmp.cleanup()

    def _get(self, path: str, host: str | None = None, origin: str | None = None):
        import urllib.request

        request = urllib.request.Request(f"{self.base}{path}")
        if host:
            request.add_unredirected_header("Host", host)
        if origin:
            request.add_unredirected_header("Origin", origin)
        try:
            response = urllib.request.urlopen(request, timeout=2)
            return response.status, response.read(), dict(response.headers)
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read(), dict(exc.headers)

    def test_serves_index_and_assets(self) -> None:
        status, body, headers = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn(b"demo", body)
        self.assertEqual(headers["Content-Type"], "text/html")

        status, body, headers = self._get("/nested/frame.js")
        self.assertEqual(status, 200)
        self.assertIn("javascript", headers["Content-Type"])

    def test_rejects_non_local_host_header(self) -> None:
        status, _body, _headers = self._get("/", host="evil.example:8080")
        self.assertEqual(status, 403)

    def test_rejects_frame_when_empty(self) -> None:
        status, _body, _headers = self._get("/api/frame.jpg")
        self.assertEqual(status, 503)

    def test_frame_served_after_publish(self) -> None:
        self.server.publish(b"frame-bytes")
        status, body, headers = self._get("/api/frame.jpg")
        self.assertEqual(status, 200)
        self.assertEqual(body, b"frame-bytes")
        self.assertEqual(headers["Content-Type"], "image/jpeg")
        self.assertEqual(headers["X-Frame-Sequence"], "1")


if __name__ == "__main__":
    unittest.main()
