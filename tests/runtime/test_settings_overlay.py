from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nuvion_app.runtime.settings_overlay import (
    SettingsOverlayError,
    apply_settings_overlay,
    load_settings_overlay,
    parse_settings_overlay,
    validate_model_pointer,
)


class SettingsOverlayTest(unittest.TestCase):
    def test_overlay_is_allowlisted_and_never_interpolates_base_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "active.env"
            path.write_text(
                "NUVION_MODEL_POINTER=${NUVION_DEVICE_PASSWORD}\n",
                encoding="utf-8",
            )
            environ = {"NUVION_DEVICE_PASSWORD": "reflected-secret"}

            with self.assertRaises(SettingsOverlayError):
                apply_settings_overlay(environ, path=path)

            self.assertEqual(environ["NUVION_DEVICE_PASSWORD"], "reflected-secret")
            self.assertNotIn("NUVION_MODEL_POINTER", environ)

    def test_unknown_secret_key_and_noncanonical_line_are_rejected(self) -> None:
        with self.assertRaises(SettingsOverlayError):
            parse_settings_overlay("NUVION_DEVICE_PASSWORD=overwrite\n")
        with self.assertRaises(SettingsOverlayError):
            parse_settings_overlay(" NUVION_CLIP_ENABLED=true\n")
        with self.assertRaises(SettingsOverlayError):
            parse_settings_overlay("NUVION_CLIP_ENABLED=true # comment\n")

    def test_safe_model_pointer_contract_matches_backend(self) -> None:
        self.assertEqual(
            validate_model_pointer("models/anomaly-v2.1/model_a"),
            "models/anomaly-v2.1/model_a",
        )
        invalid = (
            "/absolute/model",
            "models//model",
            "models/./model",
            "models/../model",
            "../model",
            "model$SECRET",
            "model${SECRET}",
            "model=value",
            "model\nvalue",
            "a" * 256,
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(SettingsOverlayError):
                validate_model_pointer(value)

    def test_overlay_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.env"
            target.write_text("NUVION_CLIP_ENABLED=true\n", encoding="utf-8")
            link = root / "active.env"
            link.symlink_to(target)

            with self.assertRaises(SettingsOverlayError):
                load_settings_overlay(link)


if __name__ == "__main__":
    unittest.main()
