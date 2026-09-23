import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from nuvion_app.runtime.glib_compat import load_glib_unix


class GLibCompatibilityTest(unittest.TestCase):
    def test_modern_namespace_is_used(self):
        modern = SimpleNamespace(signal_source_new=Mock())
        with patch('nuvion_app.runtime.glib_compat.import_module', return_value=SimpleNamespace(GLibUnix=modern)):
            self.assertIs(load_glib_unix(Mock(), SimpleNamespace()), modern)

    def test_jetpack_legacy_source_is_used_when_namespace_missing(self):
        gi = Mock()
        gi.require_version.side_effect = ValueError('Namespace GLibUnix not available')
        source = Mock()
        legacy = SimpleNamespace(unix_signal_source_new=Mock(return_value=source))
        self.assertIs(load_glib_unix(gi, legacy).signal_source_new(15), source)
        legacy.unix_signal_source_new.assert_called_once_with(15)

    def test_legacy_source_is_used_when_repository_omits_requested_namespace(self):
        gi = Mock()
        source = Mock()
        legacy = SimpleNamespace(unix_signal_source_new=Mock(return_value=source))
        with patch(
            "nuvion_app.runtime.glib_compat.import_module",
            return_value=SimpleNamespace(),
        ):
            self.assertIs(load_glib_unix(gi, legacy).signal_source_new(15), source)
        legacy.unix_signal_source_new.assert_called_once_with(15)

    def test_missing_both_apis_fails_explicitly(self):
        gi = Mock()
        gi.require_version.side_effect = ValueError('missing')
        with self.assertRaises(AttributeError): load_glib_unix(gi, SimpleNamespace())
