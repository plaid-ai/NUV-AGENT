from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import importlib.util

import google_crc32c
from google.cloud import kms_v1
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packaging/release"))
from cloud_kms_signer import CloudKmsEd25519Key
from kms_openpgp import create_public_certificate, detached_signature, load_signing_certificate
from pgpy import PGPKey
from nuvion_app.runtime.release_bom import (
    ReleaseKeyring, build_release_bom_signature, verify_signed_release_bom,
)

VERSION = "projects/test-project/locations/global/keyRings/test/cryptoKeys/test/cryptoKeyVersions/1"


class FakeKms:
    def __init__(self):
        self.key = Ed25519PrivateKey.generate()
        public = self.key.public_key()
        pem = public.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        self.pin = hashlib.sha256(public.public_bytes(
            serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
        )).hexdigest()
        self.public_response = SimpleNamespace(
            name=VERSION, algorithm=kms_v1.CryptoKeyVersion.CryptoKeyVersionAlgorithm.EC_SIGN_ED25519,
            pem=pem.decode(), pem_crc32c=google_crc32c.value(pem),
        )
        self.sign_overrides = {}
        self.requests = []

    def get_public_key(self, *, request, timeout, retry):
        return self.public_response

    def asymmetric_sign(self, *, request, timeout, retry):
        self.requests.append(request)
        signature = self.key.sign(request["data"])
        return SimpleNamespace(**{
            **dict(name=VERSION, signature=signature, verified_data_crc32c=True,
                   signature_crc32c=google_crc32c.value(signature)),
            **self.sign_overrides,
        })

    def signer(self):
        return CloudKmsEd25519Key(VERSION, self.pin, client=self)


class CloudKmsSigningTest(unittest.TestCase):
    def test_pure_ed25519_raw_data_and_crc_are_sent(self):
        fake = FakeKms()
        data = b"NUVION\x00canonical UTF-8\r\n"
        signature = fake.signer().sign(data)
        fake.key.public_key().verify(signature, data)
        self.assertEqual(fake.requests, [{"name": VERSION, "data": data, "data_crc32c": google_crc32c.value(data)}])

    def test_rejects_wrong_version_algorithm_and_public_checksum(self):
        for field, value in (("name", VERSION + "0"), ("algorithm", 12), ("pem_crc32c", -1)):
            with self.subTest(field=field):
                fake = FakeKms()
                setattr(fake.public_response, field, value)
                with self.assertRaises(ValueError):
                    fake.signer()

    def test_requires_exact_version_and_independently_trusted_public_key(self):
        fake = FakeKms()
        for version, pin in ((VERSION.replace("/1", "/latest"), fake.pin), (VERSION, "0" * 64), (VERSION, "")):
            with self.subTest(version=version, pin=pin), self.assertRaises(ValueError):
                CloudKmsEd25519Key(version, pin, client=fake)

    def test_rejects_bad_signing_responses(self):
        for field, value in (("name", VERSION + "0"), ("verified_data_crc32c", False),
                             ("signature_crc32c", -1), ("signature", b"short")):
            with self.subTest(field=field):
                fake = FakeKms()
                fake.sign_overrides[field] = value
                with self.assertRaises(ValueError):
                    fake.signer().sign(b"message")

    def test_rejects_cryptographically_wrong_signature_even_with_valid_crc(self):
        fake = FakeKms()
        wrong = Ed25519PrivateKey.generate().sign(b"message")
        fake.sign_overrides = {"signature": wrong, "signature_crc32c": google_crc32c.value(wrong)}
        with self.assertRaisesRegex(ValueError, "local public-key"):
            fake.signer().sign(b"message")

    def test_private_key_cannot_be_exported(self):
        key = FakeKms().signer()
        for operation in (key.private_bytes, key.private_bytes_raw):
            with self.assertRaises(TypeError):
                operation()
        self.assertIs(copy.copy(key), key)
        self.assertIs(copy.deepcopy(key), key)

    def test_existing_bom_verifier_accepts_remote_signature_and_rejects_tampering(self):
        fixture = json.loads((ROOT / "tests/runtime/fixtures/release-bom-v2-ed25519.json").read_text())
        payload = fixture["bom"]
        fake = FakeKms()
        signature = build_release_bom_signature(payload, key_id="kms-dev", private_key=fake.signer())
        local = build_release_bom_signature(payload, key_id="kms-dev", private_key=fake.key)
        self.assertEqual(signature, local)  # Includes the existing domain separator and canonical bytes.
        ring = ReleaseKeyring({"kms-dev": fake.key.public_key().public_bytes_raw()})
        verify_signed_release_bom(payload, signature, release_keyring=ring)
        damaged = copy.deepcopy(payload)
        damaged["agentVersion"] = "99.99.99"
        with self.assertRaises(ValueError):
            verify_signed_release_bom(damaged, signature, release_keyring=ring)


@unittest.skipUnless(shutil.which("gpg") and shutil.which("git"), "GnuPG and Git required")
class KmsOpenPgpInteropTest(unittest.TestCase):
    def test_gnupg_and_git_accept_remote_cert_and_signature(self):
        fake = FakeKms()
        signer = fake.signer()
        cert = create_public_certificate(signer, name="NUV test signer", email="test@example.invalid",
                                         created=datetime(2026, 1, 1, tzinfo=timezone.utc))
        public, _ = PGPKey.from_blob(cert)
        key = load_signing_certificate(signer, cert, fingerprint=str(public.fingerprint))
        with self.assertRaises(TypeError):
            bytes(key)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "gnupg"
            home.mkdir(mode=0o700)
            (root / "public.asc").write_text(cert)
            def gpg(*args):
                return subprocess.run(["gpg", "--homedir", str(home), "--batch", *args], capture_output=True)
            self.assertEqual(gpg("--import", str(root / "public.asc")).returncode, 0)
            data = b"release evidence\r\nexact bytes\x00\xff"
            (root / "evidence").write_bytes(data)
            (root / "evidence.asc").write_text(detached_signature(key, data))
            verified = gpg("--verify", str(root / "evidence.asc"), str(root / "evidence"))
            self.assertEqual(verified.returncode, 0, verified.stderr)
            (root / "evidence").write_bytes(data + b"tampered")
            self.assertNotEqual(gpg("--verify", str(root / "evidence.asc"), str(root / "evidence")).returncode, 0)
            repo = root / "repo"
            subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
            def git(*args, **kwargs):
                return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, **kwargs)
            tree = git("mktree", input=b"").stdout.decode().strip()
            import os
            env = {**os.environ, "GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "test@example.invalid",
                   "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "test@example.invalid", "GNUPGHOME": str(home)}
            commit = git("commit-tree", tree, input=b"test\n", env=env).stdout.decode().strip()
            tag = (f"object {commit}\ntype commit\ntag kms-test\ntagger Test <test@example.invalid> 1788912000 +0000\n\nKMS test\n").encode()
            tag_sha = git("mktag", input=tag + detached_signature(key, tag).encode()).stdout.decode().strip()
            self.assertTrue(tag_sha)
            git("update-ref", "refs/tags/kms-test", tag_sha, check=True)
            result = git("verify-tag", "--raw", "kms-test", env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(str(public.fingerprint).encode(), result.stderr)

    def test_certificate_pin_and_public_key_must_match(self):
        fake = FakeKms()
        cert = create_public_certificate(fake.signer(), name="Test", email="test@example.invalid",
                                         created=datetime(2026, 1, 1, tzinfo=timezone.utc))
        public, _ = PGPKey.from_blob(cert)
        with self.assertRaises(ValueError):
            load_signing_certificate(fake.signer(), cert, fingerprint="A" * 40)
        with self.assertRaises(ValueError):
            load_signing_certificate(FakeKms().signer(), cert, fingerprint=str(public.fingerprint))


class ApprovalAuthorizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("kms_authorize", ROOT / "packaging/release/authorize-kms-tag.py")
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def env(self):
        return dict(GITHUB_REPOSITORY="plaid-ai/NUV-AGENT", GITHUB_REF="refs/heads/main",
                    GITHUB_EVENT_NAME="workflow_dispatch", GITHUB_RUN_ATTEMPT="1", ACTOR_ID="57535980",
                    TARGET_SHA="a" * 40, TAG_NAME="v0.1.121", TAG_MESSAGE="Reviewed evidence", GH_TOKEN="test-token")

    def test_valid_request_must_match_current_main(self):
        with patch.dict("os.environ", self.env(), clear=True), patch.object(
            self.module, "run", side_effect=[json.dumps({"object": {"sha": "a" * 40}}), "a" * 40, ""]
        ):
            self.assertEqual(self.module.authorize()[:2], ("a" * 40, "v0.1.121"))

    def test_rejects_unauthorized_events_identities_and_tag_values(self):
        for field, value in (("GITHUB_REF", "refs/heads/feature"), ("GITHUB_REPOSITORY", "other/repo"),
                             ("GITHUB_EVENT_NAME", "pull_request"), ("GITHUB_RUN_ATTEMPT", "2"),
                             ("ACTOR_ID", "123"), ("TARGET_SHA", "HEAD"), ("TAG_NAME", "--force"),
                             ("TAG_NAME", "v9.9.9"), ("TAG_MESSAGE", "")):
            with self.subTest(field=field), patch.dict("os.environ", {**self.env(), field: value}, clear=True):
                with self.assertRaises(ValueError):
                    self.module.authorize()

    def test_rejects_moved_main_and_existing_tag(self):
        responses = ([json.dumps({"object": {"sha": "b" * 40}})],
                     [json.dumps({"object": {"sha": "a" * 40}}), "b" * 40],
                     [json.dumps({"object": {"sha": "a" * 40}}), "a" * 40, "existing-ref"])
        for response in responses:
            with self.subTest(response=response), patch.dict("os.environ", self.env(), clear=True), patch.object(
                self.module, "run", side_effect=response
            ), self.assertRaises(ValueError):
                self.module.authorize()




class KmsCredentialsIsolationTest(unittest.TestCase):
    def test_explicit_kms_credentials_do_not_use_or_replace_publisher_adc(self):
        import os
        from cloud_kms_signer import kms_client
        credentials = object()
        environment = {"GOOGLE_APPLICATION_CREDENTIALS": "/publisher/gcs.json",
                       "NUVION_KMS_CREDENTIALS_FILE": "/oidc/kms.json", "GITHUB_ACTIONS": "true"}
        with patch.dict(os.environ, environment, clear=True), patch(
            "google.auth.load_credentials_from_file", return_value=(credentials, None)
        ) as load, patch.object(kms_v1, "KeyManagementServiceClient") as client:
            kms_client()
            load.assert_called_once_with("/oidc/kms.json", scopes=["https://www.googleapis.com/auth/cloud-platform"])
            client.assert_called_once_with(credentials=credentials, transport="rest")
            self.assertEqual(os.environ["GOOGLE_APPLICATION_CREDENTIALS"], "/publisher/gcs.json")

    def test_conflicting_local_and_ci_credentials_fail_closed(self):
        from cloud_kms_signer import kms_client
        with patch.dict("os.environ", {"NUVION_KMS_CREDENTIALS_FILE": "/oidc/kms.json",
                                     "NUVION_KMS_GCLOUD_ACCOUNT": "owner@example.com"}, clear=True):
            with self.assertRaisesRegex(ValueError, "either"):
                kms_client()


class OtaKmsFederationTest(unittest.TestCase):
    def test_each_provider_pins_workflow_sha_subject_ref_actor_and_repository(self):
        spec = importlib.util.spec_from_file_location("ota_provision_test", ROOT / "packaging/release/provision-ota-kms.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        plan = module.plan("a" * 40)
        self.assertEqual(set(plan), {"candidate-v17", "release-main-v17"})
        for name, config in plan.items():
            condition = config["condition"]
            for expected in ("assertion.workflow_sha == '" + "a" * 40 + "'",
                             "assertion.repository_id == '1149331364'",
                             "assertion.repository_owner_id == '199492120'",
                             "assertion.event_name == 'workflow_dispatch'",
                             "assertion.actor_id in ['57535980', '89565530']", config["subject"]):
                self.assertIn(expected, condition)
            self.assertIn("refs/tags/candidate-publisher-v17" if name == "candidate-v17" else "refs/heads/main", condition)
        with self.assertRaises(ValueError):
            module.plan("main")


if __name__ == "__main__":
    unittest.main()
