import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from nuvion_app.inference.defect_regions import (
    DefectRegionState, draw_defect_regions, extract_defect_regions,
)


class DefectRegionsTest(unittest.TestCase):
    def test_normal_flat_nonfinite_and_isolated_noise_have_no_boxes(self):
        for values in (None, np.ones((96, 96)), np.full((96, 96), np.nan),
                       np.ones((2, 2)), np.ones((1, 96, 96))):
            self.assertEqual(extract_defect_regions(values, is_defect=True), ())
        values = np.zeros((96, 96), np.float32)
        values[20, 20] = 1
        self.assertEqual(extract_defect_regions(values, is_defect=True), ())
        values[10:20, 30:40] = 1
        self.assertEqual(extract_defect_regions(values, is_defect=False), ())

    def test_two_regions_and_non_square_coordinate_mapping(self):
        values = np.zeros((48, 96), np.float32)
        values[10:15, 20:30] = .9
        values[30:35, 70:80] = .8
        boxes = extract_defect_regions(values, is_defect=True)
        self.assertEqual(len(boxes), 2)
        self.assertEqual(boxes[0], (19/96, 9/48, 12/96, 7/48))
        context = Mock()
        draw_defect_regions(context, boxes[:1], 1920, 1080)
        context.rectangle.assert_called_once_with(380., 202.5, 240., 157.5)
        context.show_text.assert_called_once_with("AI REGION")

    def test_edges_clamped_and_box_count_bounded(self):
        values = np.zeros((96, 96), np.float32)
        for y in range(0, 96, 15):
            values[y:y+3, 0:3] = .9
        boxes = extract_defect_regions(values, is_defect=True)
        self.assertEqual(len(boxes), 5)
        for x, y, w, h in boxes:
            self.assertTrue(0 <= x < x+w <= 1 and 0 <= y < y+h <= 1)

    def test_different_frame_loop_stale_or_missing_result_never_draws(self):
        state = DefectRegionState()
        sample = SimpleNamespace(sample_id="a", sample_index=0, loop_index=0)
        boxes = ((.1, .2, .3, .4),)
        state.update(sample, boxes, now=10)
        self.assertEqual(state.visible(sample, now=11), boxes)
        for other in (None, SimpleNamespace(sample_id="b", sample_index=1, loop_index=0),
                      SimpleNamespace(sample_id="a", sample_index=0, loop_index=1)):
            self.assertEqual(state.visible(other, now=11), ())
        self.assertEqual(state.visible(sample, now=14), ())
        self.assertEqual(state.visible(sample, now=9), ())
        state.update(sample, (), now=11)
        self.assertEqual(state.visible(sample, now=11), ())
        state.update(sample, boxes, now=11)
        state.clear()
        self.assertEqual(state.visible(sample, now=11), ())


if __name__ == "__main__":
    unittest.main()
