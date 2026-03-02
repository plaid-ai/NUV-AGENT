import argparse
import os
import time

import cv2
from nuvion_app.config import load_env
from nuvion_app.inference.zero_shot import ZeroShotAnomalyDetector
from nuvion_app.runtime.inference_mode import normalize_backend, normalize_siglip_device


def parse_csv(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def build_detector():
    model_name = os.getenv("NUVION_ZERO_SHOT_MODEL", "google/siglip2-base-patch16-224")
    labels = parse_csv(os.getenv("NUVION_ZERO_SHOT_LABELS", "normal,defect"))
    anomaly_labels = parse_csv(os.getenv("NUVION_ZERO_SHOT_ANOMALY_LABELS", "defect,broken,crack,scratch"))
    threshold = float(os.getenv("NUVION_ZERO_SHOT_THRESHOLD", "0.7"))
    device_preference = normalize_siglip_device(os.getenv("NUVION_ZERO_SHOT_DEVICE", "auto"), default="auto")

    return ZeroShotAnomalyDetector(
        enabled=True,
        model_name=model_name,
        labels=labels,
        anomaly_labels=anomaly_labels,
        threshold=threshold,
        device_preference=device_preference,
    )

def try_open_camera(index: int):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None
    ret, _ = cap.read()
    if not ret:
        cap.release()
        return None
    return cap

def find_camera(max_index: int):
    for idx in range(max_index + 1):
        cap = try_open_camera(idx)
        if cap is not None:
            return idx, cap
    return None, None


def main():
    load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=os.getenv("NUVION_VIDEO_SOURCE", "auto"))
    parser.add_argument("--interval", type=float, default=float(os.getenv("NUVION_ZERO_SHOT_SAMPLE_SEC", "2")))
    show_default = os.getenv("NUVION_SHOW_PREVIEW", "true").lower() in ("1", "true", "yes")
    parser.add_argument("--show", action="store_true", default=show_default)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--backend", default=os.getenv("NUVION_ZSAD_BACKEND", "siglip"))
    args = parser.parse_args()

    if args.no_show:
        args.show = False

    source = args.source
    cap = None
    if isinstance(source, str) and source.lower() == "auto":
        max_index = int(os.getenv("NUVION_CAMERA_SCAN_MAX", "5"))
        idx, cap = find_camera(max_index)
        if cap is None:
            raise RuntimeError("Failed to auto-detect a camera device.")
        source = idx
    elif isinstance(source, str) and source.isdigit():
        source = int(source)

    if cap is None:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera source: {source}")

    detector = None
    triton_client = None
    raw_backend = args.backend.lower()
    backend = normalize_backend(raw_backend, default="siglip")
    if raw_backend == "mps":
        os.environ["NUVION_ZERO_SHOT_DEVICE"] = "mps"
    if backend == "siglip":
        detector = build_detector()
        if not detector.ready:
            raise RuntimeError("Zero-shot detector is not ready. Check dependencies/model download.")
    elif backend == "triton":
        from nuvion_app.agent.triton_client import TritonAnomalyClient

        triton_client = TritonAnomalyClient()
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")

    last_run = 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        if now - last_run >= args.interval:
            last_run = now
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if backend == "siglip":
                is_anomaly, result = detector.is_anomaly(frame_rgb)
                if result:
                    label = result["label"]
                    score = result["score"]
                    print(f"ZSAD(siglip): {label} ({score:.3f}) anomaly={is_anomaly}")
            else:
                result = triton_client.predict(frame_rgb)
                score = float(result.get("score", 0.0))
                threshold = float(os.getenv("NUVION_TRITON_THRESHOLD", os.getenv("NUVION_ZERO_SHOT_THRESHOLD", "0.7")))
                is_anomaly = score >= threshold
                label = result.get("label", "unknown")
                mode = result.get("mode", "generic")
                if "predicted_label" in result:
                    pred = result.get("predicted_label")
                    pred_score = float(result.get("predicted_score", 0.0))
                    print(
                        f"ZSAD(triton/{mode}): predicted={pred} ({pred_score:.3f}) "
                        f"anomaly_prob={score:.3f} anomaly={is_anomaly} anomaly_label={result.get('anomaly_label')}"
                    )
                else:
                    print(f"ZSAD(triton/{mode}): {label} score={score:.3f} anomaly={is_anomaly}")

        if args.show:
            cv2.imshow("ZSAD", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
