import json
import tempfile
import unittest
from pathlib import Path
from nuvion_app.inference.demo_profiles import DemoProfiles

class DemoProfilesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'video.mp4').write_bytes(b'video')
        (self.root / 'profiles.json').write_text(json.dumps([{'id':'video-1','name':'Video','file':'video.mp4'}]))

    def test_selection_persists_for_restart_without_claiming_active(self):
        p = DemoProfiles(self.root)
        result = p.select('video-1', lambda: True)
        self.assertEqual(result['activeId'], 'mvtec')
        self.assertEqual(result['pendingId'], 'video-1')
        self.assertEqual(DemoProfiles(self.root).active_id, 'video-1')
        with self.assertRaises(RuntimeError):p.select('mvtec', lambda: True)

    def test_failure_restores_old_selection(self):
        p = DemoProfiles(self.root)
        with self.assertRaises(RuntimeError):p.select('video-1', lambda: False)
        self.assertEqual(DemoProfiles(self.root).active_id, 'mvtec')
        self.assertIsNone(p.pending_id)

    def test_unknown_missing_and_escaped_media_rejected(self):
        p = DemoProfiles(self.root)
        with self.assertRaises(ValueError):p.select('../bad', lambda: True)
        (self.root / 'video.mp4').unlink()
        with self.assertRaises(ValueError):p.select('video-1', lambda: True)
        (self.root / 'profiles.json').write_text(json.dumps([{'id':'bad','name':'Bad','file':'../outside.mp4'}]))
        with self.assertRaises(ValueError):DemoProfiles(self.root)
