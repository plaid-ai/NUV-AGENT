from __future__ import annotations

import hashlib
import http.client
import importlib.util
import io
import json
import socket
import ssl
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
PUBLISH_PATH = ROOT / "packaging/release/publish-iq9075-candidate-gcs.py"
SPEC = importlib.util.spec_from_file_location(
    "candidate_gcs_publish_transport_under_test", PUBLISH_PATH
)
assert SPEC is not None and SPEC.loader is not None
PUBLISH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PUBLISH
SPEC.loader.exec_module(PUBLISH)

SENTINEL = "sentinel-private-token-and-response-text"
OBJECT_NAME = "releases/by-bom-sha256/" + "a" * 64 + "/bundle name+%.tar.gz"
ENCODED_NAME = (
    "releases%2Fby-bom-sha256%2F" + "a" * 64 + "%2Fbundle%20name%2B%25.tar.gz"
)
OBJECT_PATH = "/storage/v1/b/apt.plaidai.io/o/" + ENCODED_NAME
FAILURE_MESSAGES = {
    "insert": "Cloud Storage insert failed",
    "metadata": "Cloud Storage metadata lookup failed",
    "readback": "Cloud Storage generation-pinned read failed",
}


class FakeResponse:
    def __init__(
        self, status: int, body: bytes, *, read_error: Exception | None = None
    ) -> None:
        self.status = status
        self.reason = SENTINEL
        self.body = io.BytesIO(body)
        self.read_error = read_error
        self.read_sizes: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if self.read_error is not None:
            raise self.read_error
        return self.body.read(size)

    def getheader(self, name: str, default: object = None) -> object:
        return default


class FakeConnection:
    def __init__(
        self,
        response: FakeResponse,
        *,
        fail_at: str | None = None,
        failure: Exception | None = None,
    ) -> None:
        self.response = response
        self.fail_at = fail_at
        self.failure = failure
        self.requests: list[tuple[str, str, dict[str, object]]] = []
        self.headers: dict[str, str] = {}
        self.chunks: list[bytes] = []
        self.headers_ended = False
        self.closed = False

    def _check_failure(self, operation: str) -> None:
        if operation == self.fail_at:
            assert self.failure is not None
            raise self.failure

    def putrequest(self, method: str, target: str, **kwargs: object) -> None:
        self._check_failure("putrequest")
        self.requests.append((method, target, kwargs))

    def putheader(self, name: str, value: str) -> None:
        self.headers[name] = value

    def endheaders(self) -> None:
        self.headers_ended = True

    def send(self, body: bytes) -> None:
        self._check_failure("send")
        self.chunks.append(bytes(body))

    def request(self, method: str, target: str, **kwargs: object) -> None:
        self._check_failure("request")
        self.requests.append((method, target, kwargs))

    def getresponse(self) -> FakeResponse:
        self._check_failure("getresponse")
        return self.response

    def close(self) -> None:
        self.closed = True


class CandidateGcsPublishTransportTest(unittest.TestCase):
    def setUp(self) -> None:
        network = mock.patch.object(
            socket, "create_connection", side_effect=AssertionError("network forbidden")
        )
        network.start()
        self.addCleanup(network.stop)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.payload = b"\x00candidate\xff-bundle-bytes\n"
        path = Path(temporary.name) / "candidate.bundle"
        path.write_bytes(self.payload)
        self.source = PUBLISH._open_verified_input(path, maximum_bytes=1024)
        self.addCleanup(self.source.close)
        self.client = PUBLISH.GoogleStorageJsonClient(SENTINEL)
        self.addCleanup(self.client.close)
        self.metadata = {
            "bucket": "apt.plaidai.io",
            "name": OBJECT_NAME,
            "generation": "123",
            "size": str(len(self.payload)),
        }

    def _invoke(self, stage: str, connection: FakeConnection):
        with mock.patch.object(self.client, "_connection", return_value=connection):
            if stage == "insert":
                return self.client.insert(OBJECT_NAME, self.source)
            if stage == "metadata":
                return self.client.metadata(OBJECT_NAME)
            return self.client.digest(
                OBJECT_NAME, "123", maximum_bytes=len(self.payload)
            )

    def _assert_headers(self, headers: dict[str, str]) -> None:
        self.assertEqual(headers["Authorization"], "Bearer " + SENTINEL)
        self.assertEqual(headers["Accept-Encoding"], "identity")

    def test_insert_streams_exact_bytes_with_encoded_name_and_create_precondition(self) -> None:
        for status in (200, 201):
            with self.subTest(status=status):
                response = FakeResponse(status, json.dumps(self.metadata).encode())
                connection = FakeConnection(response)
                with mock.patch.object(PUBLISH, "CHUNK_BYTES", 7):
                    self.assertEqual(
                        self._invoke("insert", connection), (status, self.metadata)
                    )
                self.assertEqual(
                    connection.requests,
                    [
                        (
                            "POST",
                            "/upload/storage/v1/b/apt.plaidai.io/o"
                            "?uploadType=media&name=" + ENCODED_NAME
                            + "&ifGenerationMatch=0",
                            {"skip_accept_encoding": True},
                        )
                    ],
                )
                self._assert_headers(connection.headers)
                self.assertEqual(connection.headers["Content-Type"], "application/octet-stream")
                self.assertEqual(connection.headers["Content-Length"], str(len(self.payload)))
                self.assertNotIn("Transfer-Encoding", connection.headers)
                self.assertTrue(connection.headers_ended)
                self.assertGreater(len(connection.chunks), 1)
                self.assertEqual(b"".join(connection.chunks), self.payload)
                self.assertEqual(response.read_sizes, [PUBLISH.MAX_METADATA_BYTES + 1])
                self.assertTrue(connection.closed)

    def test_insert_precondition_failure_does_not_parse_error_body_as_metadata(self) -> None:
        connection = FakeConnection(FakeResponse(412, SENTINEL.encode()))
        self.assertEqual(self._invoke("insert", connection), (412, {}))
        self.assertTrue(connection.closed)

    def test_http_failures_carry_only_static_message_status_and_operation(self) -> None:
        for stage in FAILURE_MESSAGES:
            for status in (403, 429):
                with self.subTest(stage=stage, status=status):
                    connection = FakeConnection(FakeResponse(status, SENTINEL.encode()))
                    with self.assertRaises(PUBLISH.PublishError) as raised:
                        self._invoke(stage, connection)
                    error = raised.exception
                    self.assertEqual(str(error), FAILURE_MESSAGES[stage])
                    self.assertEqual(error.http_status, status)
                    self.assertEqual(error.stage, stage)
                    self.assertNotIn(SENTINEL, str(error))
                    self.assertTrue(connection.closed)

    def test_insert_and_metadata_reject_malformed_or_non_object_json(self) -> None:
        for stage, message in (
            ("insert", "Cloud Storage insert metadata is invalid"),
            ("metadata", "Cloud Storage metadata is invalid"),
        ):
            for body in (b"{", b"\xff", b"[]", b"null", b"42"):
                with self.subTest(stage=stage, body=body):
                    connection = FakeConnection(FakeResponse(200, body))
                    with self.assertRaises(PUBLISH.PublishError) as raised:
                        self._invoke(stage, connection)
                    self.assertEqual(str(raised.exception), message)
                    self.assertTrue(connection.closed)

    def test_only_forbidden_responses_classify_allowlisted_denied_permission(self) -> None:
        for stage in FAILURE_MESSAGES:
            for status in (403, 429):
                with self.subTest(stage=stage, status=status):
                    body = json.dumps({"error": {"message": SENTINEL + " lacks storage.objects.get access"}}).encode()
                    connection = FakeConnection(FakeResponse(status, body))
                    with self.assertRaises(PUBLISH.PublishError) as raised:
                        self._invoke(stage, connection)
                    diagnostic = PUBLISH._safe_failure(raised.exception)
                    self.assertEqual(diagnostic["deniedPermission"], "storage.objects.get" if status == 403 else None)
                    self.assertNotIn(SENTINEL, json.dumps(diagnostic))
                    self.assertTrue(connection.closed)

    def test_metadata_and_error_response_reads_are_bounded(self) -> None:
        for stage, status in (("insert", 200), ("metadata", 200), ("readback", 403)):
            with self.subTest(stage=stage, status=status):
                response = FakeResponse(status, b"x" * (PUBLISH.MAX_METADATA_BYTES + 1))
                connection = FakeConnection(response)
                with self.assertRaisesRegex(
                    PUBLISH.PublishError, "^Cloud Storage response exceeded its boundary$"
                ):
                    self._invoke(stage, connection)
                self.assertEqual(response.read_sizes, [PUBLISH.MAX_METADATA_BYTES + 1])
                self.assertTrue(connection.closed)

    def test_transport_exceptions_never_echo_exception_text(self) -> None:
        for stage in FAILURE_MESSAGES:
            boundaries = ("send", "getresponse") if stage == "insert" else ("request", "getresponse")
            for boundary in boundaries:
                for error_type in (OSError, http.client.HTTPException, ssl.SSLError):
                    with self.subTest(stage=stage, boundary=boundary, error=error_type):
                        connection = FakeConnection(
                            FakeResponse(200, b"{}"),
                            fail_at=boundary,
                            failure=error_type(SENTINEL),
                        )
                        with self.assertRaises(PUBLISH.PublishError) as raised:
                            self._invoke(stage, connection)
                        self.assertEqual(str(raised.exception), FAILURE_MESSAGES[stage])
                        self.assertNotIn(SENTINEL, str(raised.exception))
                        self.assertTrue(raised.exception.__suppress_context__)
                        self.assertTrue(connection.closed)

    def test_response_read_exception_is_static_and_closes_connection(self) -> None:
        for stage in FAILURE_MESSAGES:
            with self.subTest(stage=stage):
                connection = FakeConnection(
                    FakeResponse(200, b"{}", read_error=OSError(SENTINEL))
                )
                with self.assertRaises(PUBLISH.PublishError) as raised:
                    self._invoke(stage, connection)
                self.assertEqual(str(raised.exception), FAILURE_MESSAGES[stage])
                self.assertTrue(raised.exception.__suppress_context__)
                self.assertTrue(connection.closed)

    def test_metadata_uses_exact_object_get(self) -> None:
        connection = FakeConnection(FakeResponse(200, json.dumps(self.metadata).encode()))
        self.assertEqual(self._invoke("metadata", connection), self.metadata)
        self.assertEqual(len(connection.requests), 1)
        method, target, kwargs = connection.requests[0]
        self.assertEqual((method, target), ("GET", OBJECT_PATH))
        self.assertEqual(set(kwargs), {"headers"})
        self._assert_headers(kwargs["headers"])
        self.assertTrue(connection.closed)

    def test_digest_get_is_generation_pinned_and_hashes_all_bytes(self) -> None:
        connection = FakeConnection(FakeResponse(200, self.payload))
        with mock.patch.object(PUBLISH, "CHUNK_BYTES", 7):
            self.assertEqual(
                self._invoke("readback", connection),
                (hashlib.sha256(self.payload).hexdigest(), len(self.payload)),
            )
        self.assertEqual(len(connection.requests), 1)
        method, target, kwargs = connection.requests[0]
        self.assertEqual((method, target), ("GET", OBJECT_PATH + "?alt=media&generation=123"))
        self.assertEqual(set(kwargs), {"headers"})
        self._assert_headers(kwargs["headers"])
        self.assertGreater(len(connection.response.read_sizes), 1)
        self.assertTrue(all(0 < size <= 7 for size in connection.response.read_sizes))
        self.assertTrue(connection.closed)

    def test_digest_rejects_a_single_byte_over_the_expected_size(self) -> None:
        connection = FakeConnection(FakeResponse(200, self.payload + b"x"))
        with self.assertRaisesRegex(
            PUBLISH.PublishError, "^Cloud Storage object exceeded expected size$"
        ):
            self._invoke("readback", connection)
        self.assertEqual(connection.response.read_sizes, [len(self.payload) + 1])
        self.assertTrue(connection.closed)

    def test_digest_reports_short_read_size_without_padding(self) -> None:
        payload = self.payload[:-1]
        connection = FakeConnection(FakeResponse(200, payload))
        self.assertEqual(
            self._invoke("readback", connection),
            (hashlib.sha256(payload).hexdigest(), len(payload)),
        )
        self.assertTrue(connection.closed)

    def test_invalid_generation_is_rejected_before_connecting(self) -> None:
        for generation in ("", "0", "01", "-1", "1.0", "123&alt=json"):
            with self.subTest(generation=generation):
                with mock.patch.object(self.client, "_connection") as connect:
                    with self.assertRaisesRegex(
                        PUBLISH.PublishError, "^Cloud Storage generation is invalid$"
                    ):
                        self.client.digest(OBJECT_NAME, generation, maximum_bytes=1024)
                    connect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
