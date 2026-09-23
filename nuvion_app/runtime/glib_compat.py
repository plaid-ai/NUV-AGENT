"""GLib Unix signal namespace compatibility for JetPack 6 and newer hosts."""
from importlib import import_module
from types import SimpleNamespace


def load_glib_unix(gi, glib):
    try:
        gi.require_version("GLibUnix", "2.0")
        return import_module("gi.repository").GLibUnix
    except (ValueError, ImportError, AttributeError):
        # GLib 2.72 keeps this function in GLib, before the namespace split.
        return SimpleNamespace(signal_source_new=glib.unix_signal_source_new)
