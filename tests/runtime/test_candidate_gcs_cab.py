from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
import urllib.parse
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "packaging/release/iq9075-candidate-gcs-cab.json"
MINT_PATH = ROOT / "packaging/release/mint-candidate-gcs-cab-token.py"
PUBLISH_PATH = ROOT / "packaging/release/publish-iq9075-candidate-gcs.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FakeGcsClient:
    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, str]] = {}
        self.calls: list[tuple[str, str, str | None]] = []
        self._generation = 100

    def insert(self, object_name: str, source) -> tuple[int, dict[str, object]]:
        self.calls.append(("POST", object_name, "ifGenerationMatch=0"))
        if object_name in self.objects:
            return 412, {}
        self._generation += 1
        generation = str(self._generation)
        payload = source.read_bytes()
        self.objects[object_name] = (payload, generation)
        return 200, {
            "bucket": "apt.plaidai.io",
            "name": object_name,
            "generation": generation,
            "size": str(len(payload)),
        }

    def metadata(self, object_name: str) -> dict[str, object]:
        self.calls.append(("GET", object_name, "metadata"))
        payload, generation = self.objects[object_name]
        return {
            "bucket": "apt.plaidai.io",
            "name": object_name,
            "generation": generation,
            "size": str(len(payload)),
        }

    def digest(
        self, object_name: str, generation: str, *, maximum_bytes: int
    ) -> tuple[str, int]:
        self.calls.append(("GET", object_name, f"generation={generation}&alt=media"))
        payload, current_generation = self.objects[object_name]
        if current_generation != generation or len(payload) > maximum_bytes:
            raise RuntimeError("fake generation or bound mismatch")
        return hashlib.sha256(payload).hexdigest(), len(payload)


class CandidateGcsCabTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mint = load_module("mint_candidate_gcs_cab_token", MINT_PATH)
        cls.publish = load_module("publish_iq9075_candidate_gcs", PUBLISH_PATH)

    def test_policy_is_exact_canonical_prefix_boundary(self) -> None:
        raw = POLICY_PATH.read_bytes()
        policy = json.loads(raw)
        self.assertEqual(
            raw,
            (json.dumps(policy, sort_keys=True, separators=(",", ":")) + "\n").encode(),
        )
        rules = policy["accessBoundary"]["accessBoundaryRules"]
        self.assertEqual(len(rules), 1)
        self.assertEqual(
            rules[0]["availableResource"],
            "//storage.googleapis.com/projects/_/buckets/apt.plaidai.io",
        )
        self.assertEqual(
            rules[0]["availablePermissions"],
            [
                "inRole:roles/storage.objectCreator",
                "inRole:roles/storage.objectViewer",
            ],
        )
        expression = rules[0]["availabilityCondition"]["expression"]
        self.assertEqual(
            expression,
            "resource.name.startsWith('projects/_/buckets/apt.plaidai.io/objects/"
            "releases/by-bom-sha256/')",
        )

    def test_object_resource_prefix_keeps_bucket_folder_and_other_names_outside_boundary(self) -> None:
        policy = json.loads(POLICY_PATH.read_bytes())
        condition = policy["accessBoundary"]["accessBoundaryRules"][0]["availabilityCondition"]
        prefix = "projects/_/buckets/apt.plaidai.io/objects/releases/by-bom-sha256/"
        # The exact resource-name namespace identifies objects already. Bucket
        # and folder resource names cannot satisfy this unchanged object prefix.
        self.assertEqual(condition, {"expression": f"resource.name.startsWith('{prefix}')"})
        accepted = (
            prefix + "a" * 64 + "/release-bom.json",
            prefix + "a" * 64 + "/release-bom.json.sig",
            prefix + "a" * 64 + "/nuv-agent.agent-bundle.tar.gz",
        )
        rejected = (
            "projects/_/buckets/apt.plaidai.io",
            "projects/_/buckets/apt.plaidai.io/folders/releases/by-bom-sha256/a",
            "projects/_/buckets/apt.plaidai.io/managedFolders/releases/by-bom-sha256/a",
            "projects/_/buckets/another-bucket/objects/releases/by-bom-sha256/a",
            "projects/_/buckets/apt.plaidai.io.evil/objects/releases/by-bom-sha256/a",
            "projects/_/buckets/apt.plaidai.io/objects/releases/versions/0.1.121",
            "projects/_/buckets/apt.plaidai.io/objects/releases/by-bom-sha256-other/a",
            "projects/_/buckets/apt.plaidai.io/objects/releases/by-bom-sha256",
        )
        for name in accepted:
            with self.subTest(accepted=name):
                self.assertTrue(name.startswith(prefix))
        for name in rejected:
            with self.subTest(rejected=name):
                self.assertFalse(name.startswith(prefix))

    @staticmethod
    def _credential(path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "type": "service_account",
                    "project_id": "nuvion-project",
                    "private_key_id": "private-key-id",
                    "private_key": "-----BEGIN PRIVATE KEY-----\ntest\n-----END PRIVATE KEY-----\n",
                    "client_email": "candidate@nuvion-project.iam.gserviceaccount.com",
                    "token_uri": "https://oauth2.googleapis.com/token",
                }
            ),
            encoding="utf-8",
        )
        path.chmod(0o600)

    def test_mint_exchanges_in_memory_source_token_and_removes_broad_adc(self) -> None:
        source_token = "source-token-that-must-never-be-logged"
        downscoped_token = "downscoped-token-that-must-never-be-logged"
        calls: list[object] = []

        def command_runner(*args: object, **kwargs: object):
            self.assertFalse(credential.exists())
            calls.append((args, kwargs))
            return subprocess.CompletedProcess([], 0, stdout=source_token + "\n")

        def sts_exchange(request_body: bytes, *, timeout: float):
            self.assertLessEqual(timeout, 30)
            import urllib.parse

            form = urllib.parse.parse_qs(request_body.decode(), strict_parsing=True)
            self.assertEqual(form["subject_token"], [source_token])
            self.assertEqual(
                json.loads(urllib.parse.unquote(form["options"][0])),
                json.loads(POLICY_PATH.read_text()),
            )
            return (
                200,
                json.dumps(
                    {
                        "access_token": downscoped_token,
                        "issued_token_type": "urn:ietf:params:oauth:token-type:access_token",
                        "token_type": "Bearer",
                        "expires_in": 3600,
                    },
                    separators=(",", ":"),
                ).encode(),
            )

        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            credential = root / "gha-creds.json"
            output = root / "cab-token"
            self._credential(credential)
            with mock.patch.dict(
                os.environ,
                {"UNRELATED_SECRET": "must-not-reach-gcloud"},
                clear=False,
            ):
                result = self.mint.mint(
                    credential_path=credential,
                    policy_path=POLICY_PATH,
                    output_path=output,
                    command_runner=command_runner,
                    sts_exchange=sts_exchange,
                )
            self.assertFalse(credential.exists())
            self.assertEqual(output.read_text(encoding="utf-8").strip(), downscoped_token)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            self.assertEqual(result["expiresIn"], 3600)
            child_environment = calls[0][1]["env"]
            self.assertNotIn("UNRELATED_SECRET", child_environment)
            self.assertNotEqual(
                child_environment["GOOGLE_APPLICATION_CREDENTIALS"], str(credential)
            )
            self.assertEqual(
                Path(child_environment["GOOGLE_APPLICATION_CREDENTIALS"]).parent,
                Path(child_environment["CLOUDSDK_CONFIG"]),
            )
            self.assertEqual(child_environment["HOME"], child_environment["CLOUDSDK_CONFIG"])
            self.assertFalse(
                Path(child_environment["GOOGLE_APPLICATION_CREDENTIALS"]).exists()
            )
            self.assertNotIn(source_token, json.dumps(result))
            self.assertNotIn(downscoped_token, json.dumps(result))

    def test_sts_request_matches_official_sdk_golden_bytes(self) -> None:
        # Captured from google-auth 2.57.1 sts.Client.exchange_token with this
        # policy and a fake request transport. No live credential or STS call.
        def exchange(body: bytes, *, timeout: float):
            self.assertEqual(len(body), 892)
            self.assertEqual(
                hashlib.sha256(body).hexdigest(),
                "27a9c216129f3172a154ef0682a62dd24147b0e51786ff3244d8905e0eefe160",
            )
            form = urllib.parse.parse_qs(body.decode("ascii"), strict_parsing=True)
            self.assertTrue(form["options"][0].startswith("%7B%22accessBoundary%22"))
            self.assertEqual(
                json.loads(urllib.parse.unquote(form["options"][0])),
                json.loads(POLICY_PATH.read_text()),
            )
            return 200, json.dumps({
                "access_token": "fixture-downscoped-token",
                "issued_token_type": self.mint.ACCESS_TOKEN_TYPE,
                "token_type": "Bearer", "expires_in": 3600,
            }).encode()

        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            credential = root / "gha-creds.json"
            self._credential(credential)
            self.mint.mint(
                credential_path=credential, policy_path=POLICY_PATH,
                output_path=root / "cab-token", sts_exchange=exchange,
                command_runner=lambda *_args, **_kwargs: subprocess.CompletedProcess(
                    [], 0, stdout="sdk-fixture-subject-token\n"
                ),
            )

    def test_failure_metadata_is_fixed_and_never_echoes_untrusted_text(self) -> None:
        sentinel = "private-credential-source-token-response-description"
        errors = (
            self.mint.CabError(
                sentinel, stage=sentinel, http_status=987654,
                oauth_error=sentinel,
            ),
            RuntimeError(sentinel),
            KeyboardInterrupt(sentinel),
        )
        for error in errors:
            with self.subTest(kind=type(error).__name__):
                stdout, stderr = io.StringIO(), io.StringIO()
                with mock.patch.object(self.mint, "mint", side_effect=error), \
                     mock.patch.object(self.mint, "_install_cleanup_signal_handlers"), \
                     redirect_stdout(stdout), redirect_stderr(stderr):
                    status = self.mint.main([
                        "--credential", "/unused/gha-creds.json",
                        "--policy", "/unused/policy.json",
                        "--output-token", "/unused/cab-token",
                    ])
                self.assertEqual(status, 1)
                self.assertEqual(stdout.getvalue(), "")
                self.assertNotIn(sentinel, stderr.getvalue())
                self.assertNotIn("Traceback", stderr.getvalue())
                diagnostic = json.loads(stderr.getvalue())
                self.assertEqual(set(diagnostic), {"kind", "code", "stage", "httpStatus", "oauthError"})
                self.assertIn(diagnostic["stage"], self.mint.SAFE_STAGES)
                self.assertIsNone(diagnostic["httpStatus"])
                self.assertIsNone(diagnostic["oauthError"])

    def test_sts_failure_exposes_only_allowlisted_metadata_and_removes_adc(self) -> None:
        sentinel = "untrusted-secret-body-and-token"
        cases = (
            (400, {"error": "invalid_request", "error_description": sentinel}, "invalid_request"),
            (403, {"error": "access_denied", "access_token": sentinel}, "access_denied"),
            (401, {"error": sentinel}, None),
            (500, {"error": {"message": sentinel}}, None),
        )
        for status, body, expected_error in cases:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as raw_root:
                root = Path(raw_root)
                credential, output = root / "gha-creds.json", root / "cab-token"
                self._credential(credential)
                with self.assertRaises(self.mint.CabError) as raised:
                    self.mint.mint(
                        credential_path=credential, policy_path=POLICY_PATH,
                        output_path=output,
                        command_runner=lambda *_args, **_kwargs: subprocess.CompletedProcess(
                            [], 0, stdout="fixture-source-token\n", stderr=sentinel,
                        ),
                        sts_exchange=lambda *_args, **_kwargs: (status, json.dumps(body).encode()),
                    )
                diagnostic = self.mint._safe_failure(raised.exception)
                self.assertEqual(diagnostic["code"], "STS_EXCHANGE_FAILED")
                self.assertEqual(diagnostic["stage"], "sts_exchange")
                self.assertEqual(diagnostic["httpStatus"], status)
                self.assertEqual(diagnostic["oauthError"], expected_error)
                self.assertNotIn(sentinel, json.dumps(diagnostic))
                self.assertFalse(credential.exists())
                self.assertFalse(output.exists())

    def test_source_token_failure_is_classified_without_gcloud_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            credential, output = root / "gha-creds.json", root / "cab-token"
            self._credential(credential)
            with self.assertRaises(self.mint.CabError) as raised:
                self.mint.mint(
                    credential_path=credential, policy_path=POLICY_PATH,
                    output_path=output,
                    command_runner=lambda *_args, **_kwargs: subprocess.CompletedProcess(
                        [], 1, stdout="sensitive-source-token", stderr="sensitive-gcloud-error",
                    ),
                    sts_exchange=lambda *_args, **_kwargs: self.fail("STS must not run"),
                )
            diagnostic = self.mint._safe_failure(raised.exception)
            self.assertEqual(diagnostic["code"], "SOURCE_ACCESS_TOKEN_FAILED")
            self.assertEqual(diagnostic["stage"], "source_access_token")
            self.assertNotIn("sensitive", json.dumps(diagnostic))
            self.assertFalse(credential.exists())
            self.assertFalse(output.exists())

    def test_mint_failures_remove_broad_adc_and_never_write_token(self) -> None:
        failures = {
            "gcloud": (
                lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, stdout=""),
                lambda *_args, **_kwargs: (500, b"{}"),
                "source access token mint failed",
            ),
            "sts": (
                lambda *_args, **_kwargs: subprocess.CompletedProcess(
                    [], 0, stdout="source-token-value\n"
                ),
                lambda *_args, **_kwargs: (403, b"{}"),
                "STS exchange failed",
            ),
            "signal": (
                lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
                lambda *_args, **_kwargs: (500, b"{}"),
                None,
            ),
            "sts-signal": (
                lambda *_args, **_kwargs: subprocess.CompletedProcess(
                    [], 0, stdout="source-token-value\n"
                ),
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    KeyboardInterrupt()
                ),
                None,
            ),
        }
        for name, (runner, exchange, message) in failures.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as raw_root:
                root = Path(raw_root)
                credential = root / "gha-creds.json"
                output = root / "cab-token"
                self._credential(credential)
                context = (
                    self.assertRaisesRegex(self.mint.CabError, message)
                    if message is not None
                    else self.assertRaises(KeyboardInterrupt)
                )
                with context:
                    self.mint.mint(
                        credential_path=credential,
                        policy_path=POLICY_PATH,
                        output_path=output,
                        command_runner=runner,
                        sts_exchange=exchange,
                    )
                self.assertFalse(credential.exists())
                self.assertFalse(output.exists())

        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            credential = root / "gha-creds.json"
            output = root / "cab-token"
            policy = root / "policy.json"
            self._credential(credential)
            policy.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(self.mint.CabError, "pinned policy"):
                self.mint.mint(
                    credential_path=credential,
                    policy_path=policy,
                    output_path=output,
                    command_runner=lambda *_args, **_kwargs: None,
                    sts_exchange=lambda *_args, **_kwargs: (500, b"{}"),
                )
            self.assertFalse(credential.exists())
            self.assertFalse(output.exists())

    def test_sts_transport_is_direct_pinned_and_redirects_fail_closed(self) -> None:
        source = MINT_PATH.read_text(encoding="utf-8")
        self.assertIn('STS_HOST = "sts.googleapis.com"', source)
        self.assertIn('STS_PATH = "/v1/token"', source)
        self.assertIn("http.client.HTTPSConnection", source)
        self.assertNotIn("urllib.request.urlopen", source)
        self.assertNotIn("ProxyHandler", source)

        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            credential = root / "gha-creds.json"
            output = root / "cab-token"
            self._credential(credential)
            calls = 0

            def redirect(_body: bytes, *, timeout: float):
                nonlocal calls
                calls += 1
                return 302, b'{"Location":"https://evil.invalid/token"}'

            with self.assertRaisesRegex(self.mint.CabError, "STS exchange failed"):
                self.mint.mint(
                    credential_path=credential,
                    policy_path=POLICY_PATH,
                    output_path=output,
                    command_runner=lambda *_args, **_kwargs: subprocess.CompletedProcess(
                        [], 0, stdout="source-token-value\n"
                    ),
                    sts_exchange=redirect,
                )
            self.assertEqual(calls, 1)
            self.assertFalse(credential.exists())
            self.assertFalse(output.exists())

    @staticmethod
    def _candidate_fixture(root: Path) -> dict[str, Path]:
        artifact = root / "nuv-agent_0.1.121_iq9075-aarch64.agent-bundle.tar.gz"
        bom = root / "nuv-agent_0.1.121_iq9075-aarch64.release-bom.json"
        signature = root / (bom.name + ".sig")
        artifact.write_bytes(b"candidate bundle")
        bom_digest = "a" * 64
        bom.write_text(
            json.dumps({"bomDigest": "sha256:" + bom_digest}, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        signature.write_bytes(b"candidate signature")
        manifest = root / "candidate-evidence-manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "kind": "nuvion-iq9075-signed-evidence-candidate",
                    "workflowRunId": 123,
                    "workflowRunAttempt": 1,
                    "componentSha": "b" * 40,
                    "agentVersion": "0.1.121",
                    "releaseSequence": 2,
                    "artifact": {
                        "name": artifact.name,
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    },
                    "bootstrapDeb": {
                        "name": "nuv-agent_0.1.121_arm64.deb",
                        "sha256": "d" * 64,
                        "sizeBytes": 4096,
                    },
                    "bom": {
                        "name": bom.name,
                        "sha256": hashlib.sha256(bom.read_bytes()).hexdigest(),
                    },
                    "signature": {
                        "name": signature.name,
                        "sha256": hashlib.sha256(signature.read_bytes()).hexdigest(),
                    },
                    "releaseKeyringSha256": "c" * 64,
                    "contentAddressedPath": "releases/by-bom-sha256/" + bom_digest,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        token = root / "cab-token"
        token.write_text("downscoped-token-value\n", encoding="utf-8")
        token.chmod(0o600)
        return {
            "artifact": artifact,
            "bom": bom,
            "signature": signature,
            "manifest": manifest,
            "token": token,
        }

    def test_publisher_creates_exact_three_objects_and_idempotently_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            inputs = self._candidate_fixture(root)
            client = FakeGcsClient()
            first = self.publish.publish(
                token_path=inputs["token"],
                manifest_path=inputs["manifest"],
                artifact_path=inputs["artifact"],
                bom_path=inputs["bom"],
                signature_path=inputs["signature"],
                client_factory=lambda _token: client,
            )
            self.assertFalse(inputs["token"].exists())
            prefix = "releases/by-bom-sha256/" + "a" * 64 + "/"
            self.assertEqual(
                set(client.objects),
                {
                    prefix + inputs["artifact"].name,
                    prefix + "release-bom.json",
                    prefix + "release-bom.json.sig",
                },
            )
            self.assertTrue(all(item["created"] for item in first["objects"]))
            self.assertEqual({call[0] for call in client.calls}, {"POST", "GET"})
            self.assertTrue(
                all(call[1].startswith(prefix) for call in client.calls)
            )
            media_reads = [
                call
                for call in client.calls
                if call[0] == "GET" and "alt=media" in str(call[2])
            ]
            self.assertEqual(len(media_reads), 3)
            self.assertTrue(
                all(str(call[2]).startswith("generation=") for call in media_reads)
            )

            inputs["token"].write_text("fresh-downscoped-token-value\n", encoding="utf-8")
            inputs["token"].chmod(0o600)
            second = self.publish.publish(
                token_path=inputs["token"],
                manifest_path=inputs["manifest"],
                artifact_path=inputs["artifact"],
                bom_path=inputs["bom"],
                signature_path=inputs["signature"],
                client_factory=lambda _token: client,
            )
            self.assertTrue(all(not item["created"] for item in second["objects"]))
            second_media_reads = [
                call
                for call in client.calls
                if call[0] == "GET" and "alt=media" in str(call[2])
            ]
            self.assertEqual(len(second_media_reads), 6)

    def test_publisher_rejects_collision_and_broad_adc_environment(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            inputs = self._candidate_fixture(root)
            client = FakeGcsClient()
            prefix = "releases/by-bom-sha256/" + "a" * 64 + "/"
            expected_bom = inputs["bom"].read_bytes()
            client.objects[prefix + "release-bom.json"] = (
                bytes([expected_bom[0] ^ 1]) + expected_bom[1:],
                "99",
            )
            with self.assertRaisesRegex(self.publish.PublishError, "remote bytes differ"):
                self.publish.publish(
                    token_path=inputs["token"],
                    manifest_path=inputs["manifest"],
                    artifact_path=inputs["artifact"],
                    bom_path=inputs["bom"],
                    signature_path=inputs["signature"],
                    client_factory=lambda _token: client,
                )
            self.assertFalse(inputs["token"].exists())

            inputs = self._candidate_fixture(root)
            with mock.patch.dict(
                os.environ,
                {"GOOGLE_APPLICATION_CREDENTIALS": "/tmp/broad.json"},
                clear=False,
            ):
                with self.assertRaisesRegex(
                    self.publish.PublishError, "broad Google credential"
                ):
                    self.publish.publish(
                        token_path=inputs["token"],
                        manifest_path=inputs["manifest"],
                        artifact_path=inputs["artifact"],
                        bom_path=inputs["bom"],
                        signature_path=inputs["signature"],
                        client_factory=lambda _token: client,
                    )
            self.assertFalse(inputs["token"].exists())

    def test_publisher_destroys_token_for_every_local_or_remote_failure(self) -> None:
        class FailingClient(FakeGcsClient):
            def __init__(self, boundary: str) -> None:
                super().__init__()
                self.boundary = boundary

            def insert(self, object_name: str, source):
                if self.boundary == "insert":
                    raise RuntimeError("insert failure")
                return super().insert(object_name, source)

            def digest(self, object_name: str, generation: str, *, maximum_bytes: int):
                if self.boundary == "digest":
                    raise RuntimeError("digest failure")
                return super().digest(
                    object_name, generation, maximum_bytes=maximum_bytes
                )

        for boundary in ("factory", "insert", "digest"):
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as raw_root:
                inputs = self._candidate_fixture(Path(raw_root))

                def factory(_token: str, target: str = boundary):
                    if target == "factory":
                        raise RuntimeError("factory failure")
                    return FailingClient(target)

                with self.assertRaises(RuntimeError):
                    self.publish.publish(
                        token_path=inputs["token"],
                        manifest_path=inputs["manifest"],
                        artifact_path=inputs["artifact"],
                        bom_path=inputs["bom"],
                        signature_path=inputs["signature"],
                        client_factory=factory,
                    )
                self.assertFalse(inputs["token"].exists())

        with tempfile.TemporaryDirectory() as raw_root:
            inputs = self._candidate_fixture(Path(raw_root))
            inputs["manifest"].write_text("not-json\n", encoding="utf-8")
            with self.assertRaises(self.publish.PublishError):
                self.publish.publish(
                    token_path=inputs["token"],
                    manifest_path=inputs["manifest"],
                    artifact_path=inputs["artifact"],
                    bom_path=inputs["bom"],
                    signature_path=inputs["signature"],
                    client_factory=lambda _token: FakeGcsClient(),
                )
            self.assertFalse(inputs["token"].exists())

    def test_real_client_source_has_only_insert_and_exact_get_operations(self) -> None:
        source = PUBLISH_PATH.read_text(encoding="utf-8")
        self.assertIn("ifGenerationMatch=0", source)
        self.assertIn("generation", source)
        self.assertIn("alt", source)
        self.assertNotIn('"DELETE"', source)
        self.assertNotIn('"PATCH"', source)
        self.assertNotIn("/storage/v1/b/{bucket}/o?", source)
        self.assertIn("http.client.HTTPSConnection", source)
        self.assertNotIn("ProxyHandler", source)

    def test_publisher_diagnostics_only_emit_fixed_enums(self) -> None:
        sentinel = "SECRET-token-response-exception-path"
        for status in (sentinel, True, 999, [], {}):
            error = self.publish.PublishError(
                sentinel, stage=sentinel, http_status=status, object_field=sentinel,
                denied_permission=sentinel,
            )
            result = self.publish._safe_failure(error)
            self.assertEqual(result, {
                "kind": "nuvion-iq9075-candidate-gcs-stage-failure",
                "code": "PUBLICATION_BOUNDARY_FAILED", "stage": "unknown",
                "httpStatus": None, "object": None, "deniedPermission": None,
            })
            self.assertNotIn(sentinel, json.dumps(result))
        error = self.publish.PublishError(
            "Cloud Storage insert failed", stage="insert", http_status=403,
            object_field="bom",
        )
        self.assertEqual(self.publish._safe_failure(error)["code"], "GCS_INSERT_FAILED")
        self.assertEqual(self.publish._safe_failure(error)["httpStatus"], 403)
        self.assertEqual(self.publish._safe_failure(error)["object"], "bom")

    def test_publisher_main_diagnostics_never_print_exception_text(self) -> None:
        sentinel = "SECRET-raw-exception-body"
        for failure in (
            self.publish.PublishError(sentinel, stage="readback", http_status=403,
                                      object_field="signature"),
            KeyboardInterrupt(sentinel),
        ):
            with self.subTest(kind=type(failure).__name__):
                output, errors = io.StringIO(), io.StringIO()
                with mock.patch.object(self.publish, "publish", side_effect=failure), \
                     mock.patch.object(self.publish, "_install_cleanup_signal_handlers"), \
                     redirect_stdout(output), redirect_stderr(errors):
                    status = self.publish.main([
                        "--token-file", "/unused/cab-token", "--manifest", "/unused/manifest",
                        "--artifact", "/unused/artifact", "--bom", "/unused/bom",
                        "--signature", "/unused/signature",
                    ])
                self.assertEqual(status, 1)
                self.assertEqual(output.getvalue(), "")
                self.assertNotIn(sentinel, errors.getvalue())
                self.assertNotIn("Traceback", errors.getvalue())
                diagnostic = json.loads(errors.getvalue().splitlines()[-1])
                self.assertEqual(set(diagnostic), {"kind", "code", "stage", "httpStatus", "object", "deniedPermission"})

    def test_publisher_permission_diagnostic_never_echoes_message(self) -> None:
        for permission in self.publish.SAFE_DENIED_PERMISSIONS:
            body = json.dumps({"error": {"message": "SECRET-identity does not have " + permission + " access"}}).encode()
            self.assertEqual(self.publish._safe_denied_permission(body), permission)
        for body in (
            b"SECRET-response", b"[]", b'{"error":{"message":[]}}',
            b'{"error":{"message":"storage.objects.getIamPolicyExtra"}}',
            b'{"error":{"message":"prefix.storage.objects.get"}}',
            b'{"error":{"message":"storage.objects.get storage.objects.create"}}',
            b"X" * (self.publish.MAX_METADATA_BYTES + 1),
        ):
            self.assertIsNone(self.publish._safe_denied_permission(body))

    def test_publisher_progress_identifies_failed_object_and_never_exposes_raw_error(self) -> None:
        sentinel = "SECRET-transport-error-response"
        publisher = self.publish
        for boundary in ("insert", "metadata", "readback", "unexpected"):
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as raw_root:
                inputs = self._candidate_fixture(Path(raw_root))

                class FailingClient(FakeGcsClient):
                    def insert(self, object_name, source):
                        if object_name.endswith("release-bom.json.sig"):
                            if boundary == "insert":
                                raise publisher.PublishError("Cloud Storage insert failed", http_status=403)
                            if boundary == "metadata":
                                return 412, {}
                            if boundary == "unexpected":
                                raise RuntimeError(sentinel)
                        return super().insert(object_name, source)

                    def metadata(self, object_name):
                        raise publisher.PublishError("Cloud Storage metadata lookup failed", http_status=403)

                    def digest(self, object_name, generation, *, maximum_bytes):
                        if boundary == "readback" and object_name.endswith("release-bom.json.sig"):
                            raise publisher.PublishError("Cloud Storage generation-pinned read failed", http_status=403)
                        return super().digest(object_name, generation, maximum_bytes=maximum_bytes)

                with self.assertRaises(publisher.PublishError) as raised:
                    publisher.publish(
                        token_path=inputs["token"], manifest_path=inputs["manifest"],
                        artifact_path=inputs["artifact"], bom_path=inputs["bom"],
                        signature_path=inputs["signature"], client_factory=lambda _token: FailingClient(),
                    )
                result = publisher._safe_failure(raised.exception)
                self.assertEqual(result["object"], "signature")
                self.assertEqual(result["stage"], "insert" if boundary == "unexpected" else boundary)
                self.assertEqual(result["httpStatus"], None if boundary == "unexpected" else 403)
                self.assertNotIn(sentinel, json.dumps(result))
                self.assertFalse(inputs["token"].exists())

    def test_publisher_binds_validation_and_upload_to_one_open_descriptor(self) -> None:
        source = PUBLISH_PATH.read_text(encoding="utf-8")
        self.assertIn("class VerifiedInput", source)
        self.assertIn("os.pread", source)
        self.assertNotIn("source.read_bytes()", source)

        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            inputs = self._candidate_fixture(root)
            replacement = root / "replacement"
            replacement.write_bytes(b"X" * inputs["artifact"].stat().st_size)
            client = FakeGcsClient()

            def replace_path_after_open(_token: str):
                replacement.replace(inputs["artifact"])
                return client

            with self.assertRaisesRegex(
                self.publish.PublishError, "changed after validation"
            ):
                self.publish.publish(
                    token_path=inputs["token"],
                    manifest_path=inputs["manifest"],
                    artifact_path=inputs["artifact"],
                    bom_path=inputs["bom"],
                    signature_path=inputs["signature"],
                    client_factory=replace_path_after_open,
                )
            prefix = "releases/by-bom-sha256/" + "a" * 64 + "/"
            self.assertNotIn(prefix + inputs["artifact"].name, client.objects)
            self.assertFalse(inputs["token"].exists())


if __name__ == "__main__":
    unittest.main()
