from __future__ import annotations

import gzip
import importlib.util
import io
import tarfile
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "packaging" / "release" / "normalize-sdist.py"
SPEC = importlib.util.spec_from_file_location("normalize_sdist", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReproducibleSdistTest(unittest.TestCase):
    def _write_archive(self, path: Path, timestamp: int) -> None:
        with path.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=timestamp) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|") as archive:
                    root = tarfile.TarInfo("nuv_agent-0.1.114")
                    root.type = tarfile.DIRTYPE
                    root.mode = 0o775
                    root.mtime = timestamp
                    archive.addfile(root)
                    payload = b"immutable\n"
                    member = tarfile.TarInfo("nuv_agent-0.1.114/README.md")
                    member.size = len(payload)
                    member.mode = 0o664
                    member.mtime = timestamp + 1
                    archive.addfile(member, io.BytesIO(payload))

    def test_normalization_produces_identical_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            first = root / "first.tar.gz"
            second = root / "second.tar.gz"
            self._write_archive(first, 100)
            self._write_archive(second, 200)

            MODULE.normalize_sdist(first, 42)
            MODULE.normalize_sdist(second, 42)

            self.assertEqual(first.read_bytes(), second.read_bytes())
            with tarfile.open(first, mode="r:gz") as archive:
                members = archive.getmembers()
                self.assertEqual({member.mtime for member in members}, {42})
                self.assertEqual(
                    archive.extractfile("nuv_agent-0.1.114/README.md").read(),
                    b"immutable\n",
                )

    def test_unsafe_member_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            archive_path = Path(raw_root) / "unsafe.tar.gz"
            with tarfile.open(archive_path, mode="w:gz") as archive:
                member = tarfile.TarInfo("../escape")
                member.size = 1
                archive.addfile(member, io.BytesIO(b"x"))
            with self.assertRaises(ValueError):
                MODULE.normalize_sdist(archive_path, 42)


if __name__ == "__main__":
    unittest.main()
