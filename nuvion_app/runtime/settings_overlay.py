from __future__ import annotations

import base64
import json
import os
import re
import stat
import sys
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

MODEL_POINTER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,254}$")
SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")

SETTING_KEYS = frozenset(
    {
        "NUVION_MODEL_POINTER",
        "NUVION_MODEL_DIGEST",
        "NUVION_ZERO_SHOT_LABELS_B64",
        "NUVION_ZERO_SHOT_ANOMALY_LABELS_B64",
        "NUVION_CLIP_ENABLED",
        "NUVION_CLIP_PRE_SEC",
        "NUVION_CLIP_POST_SEC",
        "NUVION_VIDEO_WIDTH",
        "NUVION_VIDEO_HEIGHT",
        "NUVION_VIDEO_FPS",
        "NUVION_VIDEO_BITRATE_KBPS",
    }
)

_INTEGER_BOUNDS = {
    "NUVION_CLIP_PRE_SEC": (0, 60),
    "NUVION_CLIP_POST_SEC": (0, 300),
    "NUVION_VIDEO_WIDTH": (160, 7680),
    "NUVION_VIDEO_HEIGHT": (120, 4320),
    "NUVION_VIDEO_FPS": (1, 120),
    "NUVION_VIDEO_BITRATE_KBPS": (100, 20_000),
}
_LABEL_KEYS = frozenset(
    {
        "NUVION_ZERO_SHOT_LABELS_B64",
        "NUVION_ZERO_SHOT_ANOMALY_LABELS_B64",
    }
)


class SettingsOverlayError(ValueError):
    pass


def validate_model_pointer(value: Any) -> str:
    if not isinstance(value, str) or not MODEL_POINTER_PATTERN.fullmatch(value):
        raise SettingsOverlayError("model pointer violates the safe pointer pattern")
    if value.startswith("/") or "//" in value:
        raise SettingsOverlayError("model pointer must be a normalized relative path")
    segments = value.split("/")
    if any(segment in {"", ".", ".."} for segment in segments):
        raise SettingsOverlayError("model pointer contains an unsafe path segment")
    return value


def _decode_label_array(value: str) -> list[str]:
    if not value or "=" in value:
        raise SettingsOverlayError("label arrays must use unpadded base64url")
    try:
        padding = "=" * ((4 - len(value) % 4) % 4)
        raw = base64.b64decode(
            value + padding,
            altchars=b"-_",
            validate=True,
        )
        labels = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SettingsOverlayError("label array is not canonical base64url JSON") from exc
    if (
        not isinstance(labels, list)
        or not 1 <= len(labels) <= 100
        or any(
            not isinstance(label, str)
            or not label
            or label != label.strip()
            or len(label) > 100
            for label in labels
        )
        or len({label.lower() for label in labels}) != len(labels)
    ):
        raise SettingsOverlayError("label array violates the canonical contract")
    canonical = base64.urlsafe_b64encode(
        json.dumps(
            labels,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).decode("ascii").rstrip("=")
    if canonical != value:
        raise SettingsOverlayError("label array must use canonical encoding")
    return labels


def validate_overlay_value(key: str, value: Any) -> str:
    if key not in SETTING_KEYS or not isinstance(value, str):
        raise SettingsOverlayError(f"unsupported dynamic setting: {key}")
    if any(character in value for character in "\r\n\x00"):
        raise SettingsOverlayError(f"dynamic setting contains a control byte: {key}")
    if key == "NUVION_MODEL_POINTER":
        return validate_model_pointer(value)
    if key == "NUVION_MODEL_DIGEST":
        if not SHA256_PATTERN.fullmatch(value):
            raise SettingsOverlayError("model digest must be canonical sha256")
        return value
    if key in _LABEL_KEYS:
        _decode_label_array(value)
        return value
    if key == "NUVION_CLIP_ENABLED":
        if value not in {"true", "false"}:
            raise SettingsOverlayError("clip enabled must be true or false")
        return value
    if key in _INTEGER_BOUNDS:
        if not value.isdigit():
            raise SettingsOverlayError(f"dynamic integer is invalid: {key}")
        number = int(value)
        minimum, maximum = _INTEGER_BOUNDS[key]
        if not minimum <= number <= maximum:
            raise SettingsOverlayError(f"dynamic integer is out of range: {key}")
        return str(number)
    raise SettingsOverlayError(f"unsupported dynamic setting: {key}")


def parse_settings_overlay(raw: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line or line.startswith("#"):
            continue
        if line != line.strip() or line.count("=") != 1:
            raise SettingsOverlayError(
                f"invalid settings overlay line {line_number}"
            )
        key, value = line.split("=", 1)
        if key in values:
            raise SettingsOverlayError(f"duplicate dynamic setting: {key}")
        values[key] = validate_overlay_value(key, value)
    return values


def serialize_settings_overlay(values: Mapping[str, str]) -> bytes:
    normalized = {
        key: validate_overlay_value(key, value)
        for key, value in values.items()
    }
    return (
        "".join(f"{key}={normalized[key]}\n" for key in sorted(normalized))
    ).encode("utf-8")


def resolve_settings_state_dir(
    environ: Mapping[str, str] | None = None,
) -> Path:
    values = os.environ if environ is None else environ
    explicit = str(values.get("NUVION_SETTINGS_STATE_DIR") or "").strip()
    if explicit:
        return Path(explicit).expanduser().absolute()
    inbox = str(values.get("NUVION_COMMAND_INBOX_PATH") or "").strip()
    if inbox:
        return (Path(inbox).expanduser().absolute().parent / "settings")
    if sys.platform.startswith("linux"):
        return Path("/var/lib/nuv-agent/settings")
    state_home = str(values.get("XDG_STATE_HOME") or "").strip()
    root = Path(state_home).expanduser() if state_home else Path.home() / ".local/state"
    return (root / "nuvion/settings").absolute()


def resolve_settings_overlay_path(
    environ: Mapping[str, str] | None = None,
) -> Path:
    values = os.environ if environ is None else environ
    explicit = str(values.get("NUVION_SETTINGS_OVERLAY_PATH") or "").strip()
    if explicit:
        return Path(explicit).expanduser().absolute()
    return resolve_settings_state_dir(values) / "active.env"


def load_settings_overlay(
    path: str | Path,
    *,
    max_bytes: int = 64 * 1024,
) -> dict[str, str]:
    overlay_path = Path(path).expanduser().absolute()
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(overlay_path, flags)
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise SettingsOverlayError("settings overlay cannot be opened safely") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SettingsOverlayError("settings overlay must be a regular file")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(8192, max_bytes + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise SettingsOverlayError("settings overlay exceeds 64 KiB")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise SettingsOverlayError("settings overlay changed while reading")
        raw_bytes = b"".join(chunks)
    finally:
        os.close(descriptor)
    try:
        raw = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SettingsOverlayError("settings overlay is not UTF-8") from exc
    return parse_settings_overlay(raw)


def apply_settings_overlay(
    environ: MutableMapping[str, str],
    *,
    path: str | Path | None = None,
) -> dict[str, str]:
    resolved = Path(path) if path is not None else resolve_settings_overlay_path(environ)
    values = load_settings_overlay(resolved)
    environ.update(values)
    return values
