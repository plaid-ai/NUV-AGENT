from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from nuvion_app.config import effective_required_keys, load_template, read_env, write_env
from nuvion_app.inference.demo_mvtec import validate_mvtec_demo_settings
from nuvion_app.model_store import DEFAULT_MODEL_PROFILE, DEFAULT_MODEL_SOURCE
from nuvion_app.runtime.inference_mode import normalize_backend, normalize_siglip_device

CURRENT_CONFIG_SCHEMA_VERSION = "2"
_VALID_MODEL_SOURCES = {"server", "gcs"}
_VALID_MODEL_PROFILES = {"runtime", "light", "full"}
_VALID_TRITON_INPUT_FORMATS = {"NCHW", "NHWC"}
_SECRET_MARKERS = ("PASSWORD", "TOKEN", "SECRET")


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class ConfigIssue:
    key: str
    message: str


@dataclass
class ConfigGuardResult:
    config_path: Path
    changed: List[str] = field(default_factory=list)
    warnings: List[ConfigIssue] = field(default_factory=list)
    errors: List[ConfigIssue] = field(default_factory=list)
    values: Dict[str, str] = field(default_factory=dict)
    env_overrides: Dict[str, Tuple[str, str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def _is_placeholder(value: str | None) -> bool:
    if value is None:
        return True
    stripped = value.strip()
    if not stripped:
        return True
    if stripped == "***":
        return True
    if stripped.startswith("<") and stripped.endswith(">"):
        return True
    return False


def _mask_if_secret(key: str, value: str) -> str:
    if any(marker in key for marker in _SECRET_MARKERS):
        return "***"
    return value


def _merge_defaults(fields: List[Dict[str, str]], existing: Dict[str, str]) -> Dict[str, str]:
    merged = dict(existing)
    for field in fields:
        key = field["key"]
        if key not in merged:
            merged[key] = field["default"]
    return merged


def _normalize_int(value: str, default: int) -> int:
    try:
        parsed = int(str(value).strip())
    except Exception:
        return default
    if parsed <= 0:
        return default
    return parsed


def _apply_migrations(values: Dict[str, str]) -> List[str]:
    changed: List[str] = []

    def update(key: str, new_value: str, reason: str) -> None:
        old_value = values.get(key, "")
        if old_value == new_value:
            return
        values[key] = new_value
        changed.append(f"{key}: {reason} ({old_value!r} -> {new_value!r})")

    if values.get("NUVION_CONFIG_SCHEMA_VERSION", "").strip() != CURRENT_CONFIG_SCHEMA_VERSION:
        update("NUVION_CONFIG_SCHEMA_VERSION", CURRENT_CONFIG_SCHEMA_VERSION, "schema version update")

    raw_backend = (values.get("NUVION_ZSAD_BACKEND", "triton") or "triton").strip().lower()
    backend = normalize_backend(raw_backend, default="triton")
    if raw_backend != backend:
        update("NUVION_ZSAD_BACKEND", backend, "normalize backend value")

    if values.get("NUVION_TRITON_INPUT", "").strip() == "images":
        update("NUVION_TRITON_INPUT", "image", "legacy triton input alias")

    source = (values.get("NUVION_MODEL_SOURCE", DEFAULT_MODEL_SOURCE) or DEFAULT_MODEL_SOURCE).strip().lower()
    if source not in _VALID_MODEL_SOURCES:
        update("NUVION_MODEL_SOURCE", DEFAULT_MODEL_SOURCE, "invalid model source fallback")

    profile = (values.get("NUVION_MODEL_PROFILE", DEFAULT_MODEL_PROFILE) or DEFAULT_MODEL_PROFILE).strip().lower()
    if profile not in _VALID_MODEL_PROFILES:
        update("NUVION_MODEL_PROFILE", DEFAULT_MODEL_PROFILE, "invalid model profile fallback")

    jetson_profile = (values.get("NUVION_TRITON_JETSON_PROFILE", "") or "").strip().lower()
    if not jetson_profile:
        update("NUVION_TRITON_JETSON_PROFILE", "runtime", "set default Jetson profile")

    triton_input_format = (values.get("NUVION_TRITON_INPUT_FORMAT", "") or "").strip().upper()
    if triton_input_format not in _VALID_TRITON_INPUT_FORMATS:
        update("NUVION_TRITON_INPUT_FORMAT", "NCHW", "invalid triton input format fallback")

    width = _normalize_int(values.get("NUVION_TRITON_INPUT_WIDTH", ""), 336)
    if str(width) != str(values.get("NUVION_TRITON_INPUT_WIDTH", "")):
        update("NUVION_TRITON_INPUT_WIDTH", str(width), "normalize triton input width")

    height = _normalize_int(values.get("NUVION_TRITON_INPUT_HEIGHT", ""), 336)
    if str(height) != str(values.get("NUVION_TRITON_INPUT_HEIGHT", "")):
        update("NUVION_TRITON_INPUT_HEIGHT", str(height), "normalize triton input height")

    base_url = (values.get("NUVION_MODEL_SERVER_BASE_URL", "") or "").strip()
    if not base_url:
        fallback_base = (values.get("NUVION_SERVER_BASE_URL", "") or "").strip()
        if fallback_base:
            update("NUVION_MODEL_SERVER_BASE_URL", fallback_base, "fallback to NUVION_SERVER_BASE_URL")

    raw_siglip_device = (values.get("NUVION_ZERO_SHOT_DEVICE", "auto") or "auto").strip().lower()
    normalized_siglip = normalize_siglip_device(raw_siglip_device, default="auto")
    if raw_siglip_device != normalized_siglip:
        update("NUVION_ZERO_SHOT_DEVICE", normalized_siglip, "normalize zero-shot device")

    return changed


def _collect_effective_values(fields: List[Dict[str, str]], file_values: Dict[str, str]) -> tuple[Dict[str, str], Dict[str, Tuple[str, str]]]:
    effective = dict(file_values)
    overrides: Dict[str, Tuple[str, str]] = {}
    for field in fields:
        key = field["key"]
        env_value = os.getenv(key)
        if env_value is None:
            continue
        file_value = file_values.get(key, "")
        if env_value != file_value:
            overrides[key] = (file_value, env_value)
        effective[key] = env_value
    return effective, overrides


def _validate_values(values: Dict[str, str]) -> tuple[List[ConfigIssue], List[ConfigIssue]]:
    warnings: List[ConfigIssue] = []
    errors: List[ConfigIssue] = []

    for key in effective_required_keys(values):
        if _is_placeholder(values.get(key)):
            errors.append(ConfigIssue(key=key, message="필수 값이 비어있거나 placeholder입니다."))

    backend = normalize_backend(values.get("NUVION_ZSAD_BACKEND", "triton"), default="triton")
    source = (values.get("NUVION_MODEL_SOURCE", DEFAULT_MODEL_SOURCE) or DEFAULT_MODEL_SOURCE).strip().lower()

    if source not in _VALID_MODEL_SOURCES:
        errors.append(ConfigIssue(key="NUVION_MODEL_SOURCE", message="지원하지 않는 모델 source입니다. (server|gcs)"))

    if _is_truthy(values.get("NUVION_DEMO_MODE", "false")):
        try:
            validate_mvtec_demo_settings(
                base_url=values.get("NUVION_DEMO_MVTEC_BASE_URL"),
                categories=values.get("NUVION_DEMO_MVTEC_CATEGORIES"),
                cache_dir=values.get("NUVION_DEMO_MVTEC_CACHE_DIR"),
            )
        except ValueError as exc:
            errors.append(ConfigIssue(key="NUVION_DEMO_MVTEC_BASE_URL", message=str(exc)))

    if backend == "triton":
        triton_url = (values.get("NUVION_TRITON_URL", "") or "").strip()
        if not triton_url:
            errors.append(ConfigIssue(key="NUVION_TRITON_URL", message="Triton backend 사용 시 NUVION_TRITON_URL이 필요합니다."))

        triton_input = (values.get("NUVION_TRITON_INPUT", "") or "").strip()
        if not triton_input:
            errors.append(ConfigIssue(key="NUVION_TRITON_INPUT", message="Triton backend 사용 시 입력 텐서 이름이 필요합니다."))

        input_format = (values.get("NUVION_TRITON_INPUT_FORMAT", "") or "").strip().upper()
        if input_format not in _VALID_TRITON_INPUT_FORMATS:
            errors.append(ConfigIssue(key="NUVION_TRITON_INPUT_FORMAT", message="입력 포맷은 NCHW 또는 NHWC여야 합니다."))

        for key in ("NUVION_TRITON_INPUT_WIDTH", "NUVION_TRITON_INPUT_HEIGHT"):
            try:
                parsed = int(str(values.get(key, "")).strip())
                if parsed <= 0:
                    raise ValueError
            except Exception:
                errors.append(ConfigIssue(key=key, message="양수 정수여야 합니다."))

        if source == "server":
            pointer = (values.get("NUVION_MODEL_POINTER", "") or "").strip()
            if not pointer:
                errors.append(ConfigIssue(key="NUVION_MODEL_POINTER", message="server source 사용 시 pointer가 필요합니다."))
            server_url = (values.get("NUVION_MODEL_SERVER_BASE_URL", "") or "").strip()
            if not server_url:
                errors.append(ConfigIssue(key="NUVION_MODEL_SERVER_BASE_URL", message="server source 사용 시 base URL이 필요합니다."))

        if source == "gcs":
            pointer_uri = (values.get("NUVION_MODEL_GCS_POINTER_URI", "") or "").strip()
            if not pointer_uri.startswith("gs://"):
                errors.append(ConfigIssue(key="NUVION_MODEL_GCS_POINTER_URI", message="gcs source 사용 시 gs:// URI가 필요합니다."))

        mode = (values.get("NUVION_TRITON_MODE", "generic") or "generic").strip().lower()
        if mode == "anomalyclip":
            text_features = (values.get("NUVION_TRITON_TEXT_FEATURES", "") or "").strip()
            if not text_features:
                warnings.append(
                    ConfigIssue(
                        key="NUVION_TRITON_TEXT_FEATURES",
                        message="anomalyclip 모드에서는 text_features 경로가 필요합니다. (auto-pull로 채워질 수 있음)",
                    )
                )
            if values.get("NUVION_TRITON_INPUT", "").strip() == "images":
                warnings.append(
                    ConfigIssue(
                        key="NUVION_TRITON_INPUT",
                        message="legacy 입력 이름(images)이 감지되었습니다. image 사용을 권장합니다.",
                    )
                )

    return warnings, errors


def guard_config(config_path: Path, apply_fixes: bool = True) -> ConfigGuardResult:
    lines, fields = load_template()
    existing = read_env(config_path)
    merged = _merge_defaults(fields, existing)
    changed: List[str] = []

    if apply_fixes:
        changed = _apply_migrations(merged)
        if changed:
            write_env(config_path, lines, merged)

    effective_values, overrides = _collect_effective_values(fields, merged)
    warnings, errors = _validate_values(effective_values)

    return ConfigGuardResult(
        config_path=config_path,
        changed=changed,
        warnings=warnings,
        errors=errors,
        values=effective_values,
        env_overrides=overrides,
    )


def print_report(report: ConfigGuardResult) -> None:
    print(f"[DOCTOR] config: {report.config_path}")
    schema = report.values.get("NUVION_CONFIG_SCHEMA_VERSION", "<unset>")
    print(f"[DOCTOR] schema: {schema}")
    print(f"[DOCTOR] changed: {len(report.changed)}")
    if report.changed:
        for entry in report.changed:
            print(f"  - {entry}")

    print(f"[DOCTOR] env overrides: {len(report.env_overrides)}")
    for key, (file_value, env_value) in sorted(report.env_overrides.items()):
        print(
            f"  - {key}: file={_mask_if_secret(key, file_value)!r}, "
            f"env={_mask_if_secret(key, env_value)!r}"
        )

    print(f"[DOCTOR] warnings: {len(report.warnings)}")
    for issue in report.warnings:
        print(f"  - [{issue.key}] {issue.message}")

    print(f"[DOCTOR] errors: {len(report.errors)}")
    for issue in report.errors:
        print(f"  - [{issue.key}] {issue.message}")


def ensure_runtime_config(config_path: Path, stage: str, apply_fixes: bool = True) -> ConfigGuardResult:
    report = guard_config(config_path=config_path, apply_fixes=apply_fixes)
    if report.changed:
        print(f"[BOOTSTRAP] stage={stage} config migration applied: {len(report.changed)} change(s)")
    if report.env_overrides:
        print(f"[BOOTSTRAP] stage={stage} env override detected: {len(report.env_overrides)} key(s)")

    # Apply effective runtime values to current process.
    for key, value in report.values.items():
        os.environ[key] = value

    if report.errors:
        details = "; ".join(f"{issue.key}: {issue.message}" for issue in report.errors)
        raise RuntimeError(f"config preflight failed: {details}")
    return report
