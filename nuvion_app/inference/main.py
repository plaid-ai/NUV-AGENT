import os

from nuvion_app.config import load_env
from nuvion_app.runtime.bootstrap import ensure_ready


def main():
    load_env()
    ensure_ready(stage="run")
    from nuvion_app.inference.pipeline import GStreamerInferenceApp

    video_source = os.getenv("NUVION_VIDEO_SOURCE", "/dev/video0")
    app = GStreamerInferenceApp(video_source)
    app.run()


if __name__ == "__main__":
    main()
