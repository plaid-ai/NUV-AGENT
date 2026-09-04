from __future__ import annotations

import importlib.util
import unittest
from importlib.metadata import PackageNotFoundError, version


HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


@unittest.skipUnless(HAS_TRANSFORMERS, "release ZSAD compatibility dependencies are optional")
class TransformersCompatibilityTest(unittest.TestCase):
    def test_verified_release_dependency_tuple_and_loader_symbols(self) -> None:
        expected = {
            "transformers": "5.16.1",
            "huggingface-hub": "1.29.0",
            "hf-xet": "1.6.0",
            "tokenizers": "0.23.1",
            "safetensors": "0.8.0",
        }
        try:
            observed = {package: version(package) for package in expected}
        except PackageNotFoundError as exc:
            self.fail(f"incomplete release ZSAD dependency tuple: {exc}")
        self.assertEqual(observed, expected)

        import transformers

        for name in (
            "AutoModel",
            "AutoProcessor",
            "Siglip2Processor",
            "SiglipProcessor",
            "SiglipImageProcessor",
            "AutoImageProcessor",
            "GemmaTokenizerFast",
            "GemmaTokenizer",
            "AutoTokenizer",
        ):
            with self.subTest(symbol=name):
                self.assertIsNotNone(getattr(transformers, name, None))


if __name__ == "__main__":
    unittest.main()
