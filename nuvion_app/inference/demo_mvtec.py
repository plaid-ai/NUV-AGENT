from __future__ import annotations

import logging
import hashlib
import json
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
MANAGED_DEMO_PROFILE_ID = "metal-nut-showcase-v1"
MANAGED_DEMO_PROFILE = {
    "category": "metal_nut",
    "defectImages": 4,
    "imageDurationSeconds": 2,
    "normalImages": 20,
    "playlist": "deterministic-interleaved-v1",
    "profileId": MANAGED_DEMO_PROFILE_ID,
}
MANAGED_DEMO_PROFILE_DIGEST = "sha256:" + hashlib.sha256(
    json.dumps(
        MANAGED_DEMO_PROFILE,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()


@dataclass(frozen=True)
class MvtecDemoSource:
    profile_id: str
    profile_digest: str
    category: str
    image_count: int
    stage_pattern: str
    extension: str
    slideshow_caps: str
    decoder: str
    image_duration_sec: float
    ground_truth_labels: tuple[str, ...]
    sample_ids: tuple[str, ...]


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
    profile_id: str | None = None,
    profile_digest: str | None = None,
) -> MvtecDemoSource:
    resolved_base_url = (base_url or DEFAULT_MVTEC_BASE_URL).rstrip("/")
    managed_profile = profile_id == MANAGED_DEMO_PROFILE_ID
    if profile_id is not None and not managed_profile:
        raise ValueError(f"Unsupported managed demo profile: {profile_id}")
    if managed_profile and profile_digest != MANAGED_DEMO_PROFILE_DIGEST:
        raise ValueError("Managed demo profile digest does not match the built-in manifest")
    resolved_categories = (
        (str(MANAGED_DEMO_PROFILE["category"]),)
        if managed_profile
        else parse_mvtec_categories(categories)
    )
    resolved_cache_dir = Path(cache_dir or DEFAULT_MVTEC_CACHE_DIR).expanduser()
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)

    rng = chooser or random.SystemRandom()
    category = resolved_categories[0] if managed_profile else rng.choice(resolved_categories)
    extracted_dir = ensure_mvtec_category_cached(resolved_base_url, resolved_cache_dir, category)
    image_paths = collect_mvtec_demo_images(extracted_dir, category)
    if managed_profile:
        image_paths = build_managed_demo_playlist(image_paths)
    if not image_paths:
        raise ValueError(f"No demo images found for category '{category}' in {extracted_dir}")

    extension = image_paths[0].suffix.lower()
    if extension not in {".png", ".jpg", ".jpeg"}:
        raise ValueError(f"Unsupported MVTec demo image extension: {extension}")

    stage_dir = build_stage_dir(resolved_cache_dir, category, image_paths, extension)
    stage_pattern = str(stage_dir / f"%05d{extension}")
    resolved_image_duration_sec = (
        float(MANAGED_DEMO_PROFILE["imageDurationSeconds"])
        if managed_profile
        else image_duration_sec or DEFAULT_DEMO_IMAGE_DURATION_SEC
    )
    slideshow_caps = build_slideshow_caps(extension, resolved_image_duration_sec)
    decoder = "pngdec" if extension == ".png" else "jpegdec"
    ground_truth_labels = tuple(infer_mvtec_ground_truth_label(path) for path in image_paths)
    sample_ids = tuple(_sample_id(path, category) for path in image_paths)
    log.info("[DEMO] selected mvtec category=%s images=%s", category, len(image_paths))
    return MvtecDemoSource(
        profile_id=profile_id or "legacy-random-v1",
        profile_digest=profile_digest or "",
        category=category,
        image_count=len(image_paths),
        stage_pattern=stage_pattern,
        extension=extension,
        slideshow_caps=slideshow_caps,
        decoder=decoder,
        image_duration_sec=resolved_image_duration_sec,
        ground_truth_labels=ground_truth_labels,
        sample_ids=sample_ids,
    )


def build_managed_demo_playlist(image_paths: list[Path]) -> list[Path]:
    """Build the fixed 20 normal + 4 defect showcase sequence."""

    normal = [path for path in image_paths if infer_mvtec_ground_truth_label(path) == "normal"][:20]
    defect = [path for path in image_paths if infer_mvtec_ground_truth_label(path) == "defect"][:4]
    if len(normal) < 20 or len(defect) < 4:
        raise ValueError(
            "Managed demo profile requires at least 20 normal and 4 defect images"
        )
    playlist: list[Path] = []
    for index, image_path in enumerate(normal, start=1):
        playlist.append(image_path)
        if index % 5 == 0:
            playlist.append(defect[(index // 5) - 1])
    return playlist


def _sample_id(image_path: Path, category: str) -> str:
    parts = tuple(part.lower() for part in image_path.parts)
    category_index = max(
        (index for index, part in enumerate(parts) if part == category.lower()),
        default=max(0, len(parts) - 3),
    )
    return "/".join(image_path.parts[category_index:])


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
    digest = hashlib.sha256()
    for path in image_paths:
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\n")
    playlist_digest = digest.hexdigest()
    if marker_path.exists():
        recorded = marker_path.read_text(encoding="utf-8").strip()
        if recorded == playlist_digest:
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
    marker_path.write_text(playlist_digest, encoding="utf-8")
    return stage_dir


def build_slideshow_caps(extension: str, image_duration_sec: float) -> str:
    if image_duration_sec <= 0:
        raise ValueError("image_duration_sec must be greater than 0")
    frame_rate = Fraction(1 / image_duration_sec).limit_denominator(1000)
    if extension == ".png":
        return f"image/png,framerate={frame_rate.numerator}/{frame_rate.denominator}"
    return f"image/jpeg,framerate={frame_rate.numerator}/{frame_rate.denominator}"
