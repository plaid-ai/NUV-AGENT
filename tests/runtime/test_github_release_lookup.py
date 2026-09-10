from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "github_release_lookup", Path(__file__).resolve().parents[2] / "packaging/release/publish-github-release.py"
)
assert SPEC and SPEC.loader
PUBLISHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PUBLISHER)


class GitHubReleaseLookupTest(unittest.TestCase):
    def setUp(self):
        self.api = PUBLISHER.GitHubApi("plaid-ai/NUV-AGENT", "test-token")
        self.missing = PUBLISHER.GitHubApiError("not published", status=404)

    def test_published_release_does_not_require_draft_listing(self):
        release = {"id": 1, "tag_name": "v0.1.121", "draft": False}
        with mock.patch.object(self.api, "request", return_value=release) as request:
            self.assertEqual(self.api.release("v0.1.121"), release)
            request.assert_called_once()

    def test_draft_is_found_after_tag_endpoint_returns_404(self):
        draft = {"id": 385929886, "tag_name": "v0.1.121", "draft": True}
        with mock.patch.object(self.api, "request", side_effect=[self.missing, [draft]]) as request:
            self.assertEqual(self.api.release("v0.1.121"), draft)
            self.assertEqual(request.call_args.args, ("GET", "/repos/plaid-ai/NUV-AGENT/releases?per_page=100&page=1"))

    def test_draft_on_second_page_is_found(self):
        first = [{"id": index, "tag_name": f"v0.0.{index}"} for index in range(100)]
        draft = {"id": 101, "tag_name": "v0.1.121", "draft": True}
        with mock.patch.object(self.api, "request", side_effect=[self.missing, first, [draft]]):
            self.assertEqual(self.api.release("v0.1.121"), draft)

    def test_absent_release_is_none_only_after_complete_listing(self):
        with mock.patch.object(self.api, "request", side_effect=[self.missing, []]):
            self.assertIsNone(self.api.release("v0.1.121"))

    def test_duplicate_tag_candidates_fail_closed(self):
        drafts = [{"id": index, "tag_name": "v0.1.121", "draft": True} for index in (1, 2)]
        with mock.patch.object(self.api, "request", side_effect=[self.missing, drafts]):
            with self.assertRaises(PUBLISHER.GitHubReleaseError):
                self.api.release("v0.1.121")

    def test_non_404_error_is_not_treated_as_missing(self):
        with mock.patch.object(self.api, "request", side_effect=PUBLISHER.GitHubApiError("denied", status=403)) as request:
            with self.assertRaises(PUBLISHER.GitHubApiError):
                self.api.release("v0.1.121")
            request.assert_called_once()

    def test_invalid_listing_fails_closed(self):
        for response in ({}, [None], [{"tag_name": "v0.1.121"}]):
            with self.subTest(response=response):
                with mock.patch.object(self.api, "request", side_effect=[self.missing, response]):
                    with self.assertRaises(PUBLISHER.GitHubReleaseError):
                        self.api.release("v0.1.121")

    def test_listing_limit_does_not_claim_release_is_absent(self):
        page = [{"id": index, "tag_name": f"v0.0.{index}"} for index in range(100)]
        with mock.patch.object(self.api, "request", side_effect=[self.missing] + [page] * 20):
            with self.assertRaises(PUBLISHER.GitHubReleaseError):
                self.api.release("v0.1.121")


if __name__ == "__main__":
    unittest.main()
