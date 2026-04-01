from __future__ import annotations

import logging
import os
import random
import shutil
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


log = logging.getLogger(__name__)

DEFAULT_MVTEC_BASE_URL = "https://storage.googleapis.com/mvtec-dataset/mvtec-ad"
DEFAULT_MVTEC_CATEGORIES = ("screw", "metal_nut", "cable", "capsule")
DEFAULT_MVTEC_CACHE_DIR = Path("~/.cache/nuvion/demo/mvtec").expanduser()
DEFAULT_DEMO_IMAGE_DURATION_SEC = 1.0


@dataclass(frozen=True)
class MvtecDemoSource:
    category: str
    image_count: int
    stage_pattern: str
    extension: str
    slideshow_caps: str
    decoder: str
    image_duration_sec: float
    ground_truth_labels: tuple[str, ...]


def parse_mvtec_categories(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return DEFAULT_MVTEC_CATEGORIES
    parsed = tuple(item.strip().lower() for item in raw.split(",") if item.strip())
    if not parsed:
        raise ValueError("NUVION_DEMO_MVTEC_CATEGORIES must contain at least one category.")
    return parsed


def validate_mvtec_demo_settings(
    *,
    base_url: str | None,
    categories: str | None,
    cache_dir: str | None,
) -> str:
    resolved_base_url = (base_url or DEFAULT_MVTEC_BASE_URL).strip()
    if not resolved_base_url.startswith(("http://", "https://")):
        raise ValueError("NUVION_DEMO_MVTEC_BASE_URL must start with http:// or https://")
    resolved_categories = parse_mvtec_categories(categories)
    resolved_cache_dir = Path(cache_dir or DEFAULT_MVTEC_CACHE_DIR).expanduser()
    return (
        f"MVTec slideshow ready: base={resolved_base_url}, "
        f"categories={','.join(resolved_categories)}, cache={resolved_cache_dir}"
    )


def prepare_mvtec_demo_source(
    *,
    base_url: str | None = None,
    categories: str | None = None,
    cache_dir: str | None = None,
    image_duration_sec: float | None = None,
    chooser: random.Random | None = None,
) -> MvtecDemoSource:
    resolved_base_url = (base_url or DEFAULT_MVTEC_BASE_URL).rstrip("/")
    resolved_categories = parse_mvtec_categories(categories)
    resolved_cache_dir = Path(cache_dir or DEFAULT_MVTEC_CACHE_DIR).expanduser()
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)

    rng = chooser or random.SystemRandom()
    category = rng.choice(resolved_categories)
    extracted_dir = ensure_mvtec_category_cached(resolved_base_url, resolved_cache_dir, category)
    image_paths = collect_mvtec_demo_images(extracted_dir, category)
    if not image_paths:
        raise ValueError(f"No demo images found for category '{category}' in {extracted_dir}")

    extension = image_paths[0].suffix.lower()
    if extension not in {".png", ".jpg", ".jpeg"}:
        raise ValueError(f"Unsupported MVTec demo image extension: {extension}")

    stage_dir = build_stage_dir(resolved_cache_dir, category, image_paths, extension)
    stage_pattern = str(stage_dir / f"%05d{extension}")
    resolved_image_duration_sec = image_duration_sec or DEFAULT_DEMO_IMAGE_DURATION_SEC
    slideshow_caps = build_slideshow_caps(extension, resolved_image_duration_sec)
    decoder = "pngdec" if extension == ".png" else "jpegdec"
    ground_truth_labels = tuple(infer_mvtec_ground_truth_label(path) for path in image_paths)
    log.info("[DEMO] selected mvtec category=%s images=%s", category, len(image_paths))
    return MvtecDemoSource(
        category=category,
        image_count=len(image_paths),
        stage_pattern=stage_pattern,
        extension=extension,
        slideshow_caps=slideshow_caps,
        decoder=decoder,
        image_duration_sec=resolved_image_duration_sec,
        ground_truth_labels=ground_truth_labels,
    )


def ensure_mvtec_category_cached(base_url: str, cache_dir: Path, category: str) -> Path:
    archives_dir = cache_dir / "archives"
    extracted_root = cache_dir / "extracted"
    archives_dir.mkdir(parents=True, exist_ok=True)
    extracted_root.mkdir(parents=True, exist_ok=True)

    archive_path = archives_dir / f"{category}.tar.xz"
    extracted_dir = extracted_root / category
    if extracted_dir.exists():
        return extracted_dir

    download_url = f"{base_url}/{category}.tar.xz"
    if not archive_path.exists():
        download_to_path(download_url, archive_path)

    tmp_extract_dir = extracted_root / f".{category}.tmp"
    if tmp_extract_dir.exists():
        shutil.rmtree(tmp_extract_dir)
    tmp_extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive_path, mode="r:xz") as tar:
            tar.extractall(tmp_extract_dir)
        tmp_extract_dir.rename(extracted_dir)
    finally:
        if tmp_extract_dir.exists():
            shutil.rmtree(tmp_extract_dir)
    return extracted_dir


def download_to_path(url: str, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target_path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=120) as response, tmp_path.open("wb") as dst:
            shutil.copyfileobj(response, dst)
        tmp_path.replace(target_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def collect_mvtec_demo_images(extracted_dir: Path, category: str) -> list[Path]:
    candidate_roots = [
        extracted_dir / category / "test",
        extracted_dir / "test",
        extracted_dir / category / "train" / "good",
        extracted_dir / "train" / "good",
    ]

    for root in candidate_roots:
        if not root.exists():
            continue
        if root.name == "test":
            image_paths = sorted(path for path in root.rglob("*") if path.is_file())
        else:
            image_paths = sorted(path for path in root.iterdir() if path.is_file())
        if image_paths:
            return image_paths

    for path in extracted_dir.rglob("test"):
        if not path.is_dir():
            continue
        image_paths = sorted(file_path for file_path in path.rglob("*") if file_path.is_file())
        if image_paths:
            return image_paths

    for path in extracted_dir.rglob("train/good"):
        if not path.is_dir():
            continue
        image_paths = sorted(file_path for file_path in path.iterdir() if file_path.is_file())
        if image_paths:
            return image_paths

    raise ValueError(f"Could not locate demo images for category '{category}' in {extracted_dir}")


def infer_mvtec_ground_truth_label(image_path: Path) -> str:
    normalized_parts = {part.lower() for part in image_path.parts}
    return "normal" if "good" in normalized_parts else "defect"


def build_stage_dir(cache_dir: Path, category: str, image_paths: list[Path], extension: str) -> Path:
    stage_dir = cache_dir / "slides" / category
    marker_path = stage_dir / ".ready"
    if marker_path.exists():
        recorded = marker_path.read_text(encoding="utf-8").strip()
        if recorded == str(len(image_paths)):
            return stage_dir

    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    for index, source_path in enumerate(image_paths):
        staged_path = stage_dir / f"{index:05d}{extension}"
        try:
            os.symlink(source_path, staged_path)
        except OSError:
            shutil.copy2(source_path, staged_path)
    marker_path.write_text(str(len(image_paths)), encoding="utf-8")
    return stage_dir


def build_slideshow_caps(extension: str, image_duration_sec: float) -> str:
    if image_duration_sec <= 0:
        raise ValueError("image_duration_sec must be greater than 0")
    frame_rate = Fraction(1 / image_duration_sec).limit_denominator(1000)
    if extension == ".png":
        return f"image/png,framerate={frame_rate.numerator}/{frame_rate.denominator}"
    return f"image/jpeg,framerate={frame_rate.numerator}/{frame_rate.denominator}"
