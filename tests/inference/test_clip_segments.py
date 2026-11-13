from __future__ import annotations

import os
import tempfile
import unittest

from nuvion_app.inference.clip_segments import (
    collect_stable_segments,
    list_stable_segments,
)


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

    def test_list_stable_segments_excludes_unprobeable_mp4(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            good = self._touch_segment(tmp, "segment_00000.mp4", 95.0)
            self._touch_segment(tmp, "segment_00001.mp4", 95.0)

            result = list_stable_segments(
                tmp,
                settle_sec=1.0,
                now_ts=100.0,
                probe_func=lambda path: path == good,
            )

            self.assertEqual(result, [good])


def test_event_offset_uses_inference_frame_and_actual_fragment_boundaries():
    from nuvion_app.inference.clip_segments import clip_event_offset_ms

    fragments = ((9_000_000_000, 10_100_000_000), (10_100_000_000, 11_250_000_000))
    assert clip_event_offset_ms(fragments, 10_400_000_000) == 1400
    assert clip_event_offset_ms(fragments, 10_100_000_000) == 1100
    assert clip_event_offset_ms(fragments, 11_250_000_000) is None
    assert clip_event_offset_ms((fragments[0], (10_200_000_000, 11_000_000_000)), 10_400_000_000) is None
    assert clip_event_offset_ms((None, fragments[1]), 10_400_000_000) is None


def test_capture_window_waits_for_closed_post_event_fragments_and_preserves_event():
    from nuvion_app.inference.clip_segments import select_event_fragments

    ns = 1_000_000_000
    bounds = {'early': (6*ns, 8*ns), 'before': (8*ns, 10*ns),
              'event': (10*ns, 12*ns), 'after': (12*ns, None)}
    assert select_event_fragments(bounds, 11*ns, 2*ns, 2*ns) is None
    bounds['after'] = (12*ns, 14*ns)
    assert [name for name, _ in select_event_fragments(bounds, 11*ns, 2*ns, 2*ns)] == ['before', 'event', 'after']
    del bounds['event']
    assert select_event_fragments(bounds, 11*ns, 2*ns, 2*ns) is None
