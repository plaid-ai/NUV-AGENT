from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "packaging" / "release" / "verify-github-release-assets.py"
SPEC = importlib.util.spec_from_file_location("verify_github_release_assets", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReleaseAssetGateTest(unittest.TestCase):
    def _asset(self, root: Path, content: bytes = b"immutable") -> Path:
        path = root / "nuv-agent.tar.gz"
        path.write_bytes(content)
        return path

    def test_missing_release_or_asset_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            asset = self._asset(Path(raw_root))
            MODULE.verify_existing_assets(None, [asset])
            MODULE.verify_existing_assets({"assets": []}, [asset])

    def test_existing_asset_requires_matching_digest(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            asset = self._asset(Path(raw_root))
            digest = MODULE.sha256_digest(asset)
            MODULE.verify_existing_assets(
                {"assets": [{"name": asset.name, "digest": digest}]}, [asset]
            )
            with self.assertRaises(MODULE.ReleaseAssetVerificationError):
                MODULE.verify_existing_assets(
                    {"assets": [{"name": asset.name, "digest": "sha256:" + "0" * 64}]},
                    [asset],
                )

    def test_existing_asset_without_digest_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            asset = self._asset(Path(raw_root))
            with self.assertRaises(MODULE.ReleaseAssetVerificationError):
                MODULE.verify_existing_assets(
                    {"assets": [{"name": asset.name, "digest": None}]}, [asset]
                )

    def test_duplicate_remote_asset_names_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            asset = self._asset(Path(raw_root))
            digest = MODULE.sha256_digest(asset)
            with self.assertRaises(MODULE.ReleaseAssetVerificationError):
                MODULE.verify_existing_assets(
                    {
                        "assets": [
                            {"name": asset.name, "digest": digest},
                            {"name": asset.name, "digest": digest},
                        ]
                    },
                    [asset],
                )


if __name__ == "__main__":
    unittest.main()
