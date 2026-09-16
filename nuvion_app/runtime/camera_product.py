from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

PRODUCT_BASE = "base"
PRODUCT_ULTRA = "ultra"
VALID_CAMERA_PRODUCTS = (PRODUCT_BASE, PRODUCT_ULTRA)


def camera_product_updates(
    *, product: str, i2c_bus: int | None = None
) -> dict[str, str]:
    normalized = product.strip().lower()
    if normalized == PRODUCT_BASE:
        return {
            "NUVION_VIDEO_SOURCE": "rpi",
            "NUVION_CAMERA_PREFERENCE": "csi",
            "NUVION_CAMERA_PROFILE": "arducam_b0272",
            "NUVION_CAMERA_FOCUS_MODE": "startup-lock",
            "NUVION_CAMERA_FOCUS_REQUIRED": "true",
            "NUVION_CAMERA_I2C_BUS": "",
        }
    if normalized == PRODUCT_ULTRA:
        if i2c_bus is None or isinstance(i2c_bus, bool) or i2c_bus < 0:
            raise ValueError(
                "NUVION Ultra requires an explicit non-negative camera I2C bus"
            )
        return {
            "NUVION_VIDEO_SOURCE": "jetson",
            "NUVION_CAMERA_PREFERENCE": "csi",
            "NUVION_CAMERA_PROFILE": "arducam_b0273",
            "NUVION_CAMERA_FOCUS_MODE": "startup-lock",
            "NUVION_CAMERA_FOCUS_REQUIRED": "true",
            "NUVION_CAMERA_I2C_BUS": str(i2c_bus),
        }
    raise ValueError(f"unsupported camera product: {product}")


def _render_env_updates(content: str, updates: Mapping[str, str]) -> str:
    pending = dict(updates)
    rendered: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in line:
            key = line.split("=", 1)[0].strip()
            if key in updates:
                rendered.append(f"{key}={updates[key]}")
                pending.pop(key, None)
                continue
        rendered.append(line)

    if pending and rendered and rendered[-1]:
        rendered.append("")
    rendered.extend(f"{key}={value}" for key, value in pending.items())
    return "\n".join(rendered) + "\n"


def apply_camera_product_preset(
    config_path: Path, *, product: str, i2c_bus: int | None = None
) -> dict[str, str]:
    updates = camera_product_updates(product=product, i2c_bus=i2c_bus)
    content = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(_render_env_updates(content, updates), encoding="utf-8")
    return updates


def _product_from_profile(profile: str) -> str:
    if profile == "arducam_b0272":
        return PRODUCT_BASE
    if profile == "arducam_b0273":
        return PRODUCT_ULTRA
    return "unknown"


def build_camera_qualification_report(
    *,
    config_path: Path,
    values: Mapping[str, str],
    config_ok: bool,
    hardware_checks: Sequence[Mapping[str, str]],
) -> dict[str, object]:
    checks = [
        {
            "name": str(check.get("name", "")),
            "status": str(check.get("status", "")),
            "detail": str(check.get("detail", "")),
        }
        for check in hardware_checks
    ]
    passed = (
        config_ok
        and bool(checks)
        and all(check["status"] not in {"fail", "warn", "skip"} for check in checks)
    )
    profile = (values.get("NUVION_CAMERA_PROFILE") or "auto").strip().lower()
    return {
        "schemaVersion": "nuvion.camera-qualification.v1",
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "configPath": str(config_path),
        "product": _product_from_profile(profile),
        "camera": {
            "videoSource": values.get("NUVION_VIDEO_SOURCE", ""),
            "profile": profile,
            "focusMode": values.get("NUVION_CAMERA_FOCUS_MODE", ""),
            "focusRequired": values.get("NUVION_CAMERA_FOCUS_REQUIRED", ""),
            "i2cBus": values.get("NUVION_CAMERA_I2C_BUS", ""),
        },
        "configOk": config_ok,
        "hardwareChecks": checks,
        "result": "PASS" if passed else "FAIL",
    }


def write_camera_qualification_report(
    output_path: Path,
    *,
    config_path: Path,
    values: Mapping[str, str],
    config_ok: bool,
    hardware_checks: Sequence[Mapping[str, str]],
) -> dict[str, object]:
    report = build_camera_qualification_report(
        config_path=config_path,
        values=values,
        config_ok=config_ok,
        hardware_checks=hardware_checks,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
