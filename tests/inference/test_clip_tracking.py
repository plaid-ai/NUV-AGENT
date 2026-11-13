import shutil
import subprocess
import unittest

import numpy as np

from nuvion_app.inference.clip_tracking import build_clip_track


class ClipTrackingTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path

        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV unavailable")
        self.ffmpeg, self.ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
        if not self.ffmpeg or not self.ffprobe:
            self.skipTest("ffmpeg/ffprobe unavailable")
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        self.tmp_path = Path(root.name)

    def test_track_matches_recorded_frame_moves_and_does_not_reacquire(self):
        ffmpeg, ffprobe = self.ffmpeg, self.ffprobe

        width, height, fps = 256, 144, 25
        rng = np.random.default_rng(37)
        belt = rng.integers(18, 47, (height, width), dtype=np.uint8)
        part = rng.integers(100, 230, (54, 46), dtype=np.uint8)
        frames = []
        for i in range(30):
            image = belt.copy()
            y = 85 - 6 * i if i < 24 else 38  # A different part appears after exit.
            bottom = min(height, y + len(part))
            if bottom > max(0, y):
                image[max(0, y):bottom, 100:146] = part[max(0, -y):bottom - y]
            frames.append(np.repeat(image[..., None], 3, axis=2))

        path = str(self.tmp_path / "clip.mp4")
        subprocess.run(
            [ffmpeg, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", path],
            input=np.asarray(frames).tobytes(), check=True,
        )
        track = build_clip_track(
            path, ffmpeg=ffmpeg, ffprobe=ffprobe, event_ms=360,  # Offset hint is one frame late.
            frame_rgb=frames[8], regions=((105 / width, 40 / height, 30 / width, 32 / height),),
        )
        assert track is not None
        assert track["eventTimeMs"] == 320
        by_time = {frame["timeMs"]: frame["boxes"][0] for frame in track["frames"]}
        self.assertAlmostEqual(by_time[440][1], 22 / height, delta=.025)
        assert max(by_time) < 24 * 40
        assert all(t not in by_time for t in range(960, 1200, 40))


    def test_does_not_attach_box_to_unrelated_footage(self):
        ffmpeg, ffprobe = self.ffmpeg, self.ffprobe
        path = str(self.tmp_path / "empty.mp4")
        subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=black:s=128x96:d=1",
                        "-c:v", "libx264", path], check=True)
        assert build_clip_track(
            path, ffmpeg=ffmpeg, ffprobe=ffprobe, event_ms=320,
            frame_rgb=np.full((96, 128, 3), 220, dtype=np.uint8),
            regions=((.2, .2, .3, .3),),
        ) is None
