from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from nuvion_app.runtime.errors import BootstrapError
from nuvion_app.runtime import platform_installer


class PlatformInstallerTest(unittest.TestCase):
    def test_detect_brew_path_from_env(self) -> None:
        with tempfile.NamedTemporaryFile() as tmp:
            with mock.patch.dict(os.environ, {"HOMEBREW_BIN": tmp.name}):
                self.assertEqual(platform_installer.detect_brew_path(), tmp.name)

    def test_ensure_homebrew_installed_respects_flag(self) -> None:
        with mock.patch.dict(os.environ, {"NUVION_HOMEBREW_AUTOINSTALL": "false"}):
            with mock.patch.object(platform_installer, "detect_brew_path", return_value=None):
                with self.assertRaises(BootstrapError) as ctx:
                    platform_installer.ensure_homebrew_installed()
        self.assertEqual(ctx.exception.code, "brew_install_failed")


if __name__ == "__main__":
    unittest.main()
