from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from nuvion_app.inference.face_tracking import FaceBox
from nuvion_app.inference.face_tracking import FaceTrackingController
from nuvion_app.inference.face_tracking import TrackingOverlayState
from nuvion_app.inference.face_tracking import TritonFaceDetector
from nuvion_app.inference.face_tracking import build_overlay_snapshot
from nuvion_app.inference.face_tracking import draw_tracking_overlay


class FakeDetector:
    def __init__(self, faces):
        self.faces = faces
        self.ready = True
        self.error = ""

    def detect(self, _frame):
        return list(self.faces)


class FakeContext:
    def __init__(self):
        self.rectangles = []
        self.line_widths = []
        self.colors = []
        self.arcs = []
        self.stroke_calls = 0
        self.fill_calls = 0

    def set_source_rgba(self, *args):
        self.colors.append(args)

    def set_line_width(self, width):
        self.line_widths.append(width)

    def rectangle(self, x, y, width, height):
        self.rectangles.append((x, y, width, height))

    def stroke(self):
        self.stroke_calls += 1

    def arc(self, x, y, radius, start, end):
        self.arcs.append((x, y, radius, start, end))

    def fill(self):
        self.fill_calls += 1


class FaceTrackingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_primary_face_is_selected_by_frame_center(self) -> None:
        controller = FaceTrackingController(
            detector=FakeDetector(
                [
                    FaceBox(10, 10, 50, 50),
                    FaceBox(300, 200, 80, 80),
                ]
            ),
            deadzone_pct=0.12,
            lost_timeout_sec=1.0,
        )

        decision = controller.process_frame(self.frame)

        self.assertEqual(decision.primary_face, FaceBox(300, 200, 80, 80))

    def test_face_inside_deadzone_is_centered(self) -> None:
        controller = FaceTrackingController(
            detector=FakeDetector([FaceBox(290, 210, 60, 60)]),
            deadzone_pct=0.2,
            lost_timeout_sec=1.0,
        )

        decision = controller.process_frame(self.frame)

        self.assertTrue(decision.centered)
        self.assertIsNone(decision.pan_command)
        self.assertIsNone(decision.tilt_command)

    def test_face_outside_deadzone_emits_motor_commands(self) -> None:
        controller = FaceTrackingController(
            detector=FakeDetector([FaceBox(20, 20, 60, 60)]),
            deadzone_pct=0.1,
            lost_timeout_sec=1.0,
        )

        decision = controller.process_frame(self.frame)

        self.assertEqual(decision.pan_command.value, "L")
        self.assertEqual(decision.tilt_command.value, "U")

    def test_lost_face_uses_stale_box_then_idles(self) -> None:
        controller = FaceTrackingController(
            detector=FakeDetector([FaceBox(300, 200, 80, 80)]),
            deadzone_pct=0.12,
            lost_timeout_sec=1.0,
        )

        with mock.patch("nuvion_app.inference.face_tracking.time.time", side_effect=[10.0, 10.5, 11.5]):
            controller.process_frame(self.frame)
            controller.detector.faces = []
            decision_lost = controller.process_frame(self.frame)
            decision_idle = controller.process_frame(self.frame)

        self.assertIsNotNone(decision_lost.stale_face)
        self.assertEqual(decision_lost.status_text, "TRACK face lost")
        self.assertIsNone(decision_idle.stale_face)
        self.assertEqual(decision_idle.status_text, "TRACK face idle")

    def test_draw_tracking_overlay_renders_faces_deadzone_and_center(self) -> None:
        snapshot = build_overlay_snapshot(
            type(
                "Decision",
                (),
                {
                    "status_text": "TRACK face active",
                    "faces": (FaceBox(10, 20, 30, 40),),
                    "primary_face": FaceBox(10, 20, 30, 40),
                    "stale_face": None,
                    "deadzone": (100, 100, 50, 60),
                },
            )(),
            enabled=True,
            show_bbox=True,
        )
        context = FakeContext()

        draw_tracking_overlay(context, snapshot)

        self.assertEqual(len(context.rectangles), 2)
        self.assertEqual(len(context.arcs), 1)
        self.assertGreaterEqual(context.stroke_calls, 2)
        self.assertEqual(context.fill_calls, 1)

    def test_triton_face_detector_recreates_client_on_thread_change(self) -> None:
        created_clients = []

        class FakeClient:
            def __init__(self) -> None:
                created_clients.append(self)

            def predict(self, _frame):
                return []

        with mock.patch("nuvion_app.inference.face_tracking.TritonFaceClient", FakeClient):
            detector = TritonFaceDetector()
            self.assertTrue(detector.ready)
            detector.detect(self.frame)
            first_client = detector._client
            detector._client_thread_id = -1
            detector.detect(self.frame)

        self.assertEqual(len(created_clients), 3)
        self.assertIsNot(first_client, detector._client)


if __name__ == "__main__":
    unittest.main()
