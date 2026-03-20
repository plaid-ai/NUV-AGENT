from __future__ import annotations

import os
import tempfile
import unittest

from nuvion_app.inference.clip_segments import collect_stable_segments
from nuvion_app.inference.clip_segments import list_stable_segments


class ClipSegmentsTest(unittest.TestCase):
    def _touch_segment(self, root: str, name: str, mtime: float) -> str:
        path = os.path.join(root, name)
        with open(path, "wb") as handle:
            handle.write(b"segment")
        os.utime(path, (mtime, mtime))
        return path

    def test_list_stable_segments_excludes_recent_segment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stable = self._touch_segment(tmp, "segment_00000.mp4", 95.0)
            self._touch_segment(tmp, "segment_00001.mp4", 99.5)

            result = list_stable_segments(tmp, settle_sec=1.0, now_ts=100.0)

            self.assertEqual(result, [stable])

    def test_list_stable_segments_returns_empty_when_only_unsettled_segment_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._touch_segment(tmp, "segment_00000.mp4", 99.8)

            result = list_stable_segments(tmp, settle_sec=1.0, now_ts=100.0)

            self.assertEqual(result, [])

    def test_collect_stable_segments_applies_before_and_after_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            seg0 = self._touch_segment(tmp, "segment_00000.mp4", 190.0)
            seg1 = self._touch_segment(tmp, "segment_00001.mp4", 193.0)
            seg2 = self._touch_segment(tmp, "segment_00002.mp4", 196.0)
            self._touch_segment(tmp, "segment_00003.mp4", 199.5)

            before_result = collect_stable_segments(
                tmp,
                settle_sec=1.0,
                before=194.0,
                count=2,
                now_ts=200.0,
            )
            after_result = collect_stable_segments(
                tmp,
                settle_sec=1.0,
                after=192.0,
                count=2,
                now_ts=200.0,
            )

            self.assertEqual(before_result, [seg0, seg1])
            self.assertEqual(after_result, [seg1, seg2])
