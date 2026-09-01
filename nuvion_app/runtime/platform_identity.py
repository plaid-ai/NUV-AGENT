from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

NUVION = "NUVION"
NUVION_PRO = "NUVION_PRO"
NUVION_ULTRA = "NUVION_ULTRA"
IQ9075_DEV = "IQ9075_DEV"
MACOS_DEV = "MACOS_DEV"
UNKNOWN_PRODUCT = "UNKNOWN"

PROFILE_RPI5_DEEPX = "rpi5_deepx_dx_m1"
PROFILE_VENTUNO_Q = "ventuno_q"
PROFILE_JETSON_ORIN_NX = "jetson_orin_nx"
PROFILE_IQ9075_DEV = "iq9075_dev"
PROFILE_MACOS_DEV = "macos_dev"
PROFILE_UNKNOWN = "unknown"

IDENTITY_STATUS_VERIFIED = "VERIFIED"
IDENTITY_STATUS_DEV = "DEV"
IDENTITY_STATUS_MISMATCH = "MISMATCH"
IDENTITY_STATUS_UNPROVISIONED = "UNPROVISIONED"
IDENTITY_STATUS_UNVERIFIED = "UNVERIFIED"
IDENTITY_STATUS_INVALID = "INVALID"

DEFAULT_IDENTITY_PATH = Path("/etc/nuv-agent/device-identity.json")

PRODUCT_PROFILE = {
    NUVION: PROFILE_RPI5_DEEPX,
    NUVION_PRO: PROFILE_VENTUNO_Q,
    NUVION_ULTRA: PROFILE_JETSON_ORIN_NX,
    IQ9075_DEV: PROFILE_IQ9075_DEV,
    MACOS_DEV: PROFILE_MACOS_DEV,
}

_COMMON_COMMAND_CAPABILITIES = frozenset(
    {
        "fleet.command.v1",
        "command.config.apply",
        "command.stream.policy",
        "command.agent.update",
        "telemetry.runtime.v1",
        "video.gstreamer",
    }
)

PROFILE_CAPABILITIES = {
    PROFILE_RPI5_DEEPX: _COMMON_COMMAND_CAPABILITIES | {"accelerator.deepx"},
    PROFILE_VENTUNO_Q: _COMMON_COMMAND_CAPABILITIES | {"accelerator.ventuno_q"},
    PROFILE_JETSON_ORIN_NX: _COMMON_COMMAND_CAPABILITIES
    | {"accelerator.cuda", "accelerator.tensorrt"},
    PROFILE_IQ9075_DEV: _COMMON_COMMAND_CAPABILITIES
    | {"dev.hardware", "camera.usb", "camera.depthai"},
    PROFILE_MACOS_DEV: _COMMON_COMMAND_CAPABILITIES
    | {"accelerator.mps", "dev.simulation"},
}


@dataclass(frozen=True)
class PlatformProbe:
    system: str
    os_version: str
    kernel_version: str
    architecture: str
    hardware_text: str
    accelerator_runtime: str
    gstreamer_version: str


@dataclass(frozen=True)
class DeclaredPlatform:
    product_model: str
    hardware_revision: str
    platform_profile: str
    source: str


@dataclass(frozen=True)
class PlatformIdentity:
    product_model: str
    hardware_revision: str
    platform_profile: str
    observed_platform_profile: str
    identity_status: str
    identity_source: str
    capabilities: frozenset[str]
    os_name: str
    os_version: str
    kernel_version: str
    architecture: str
    accelerator: str
    accelerator_runtime: str
    gstreamer_version: str
    identity_error: str | None = None

    def to_telemetry(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "productModel": self.product_model,
            "hardwareRevision": self.hardware_revision,
            "platformProfile": self.platform_profile,
            "observedPlatformProfile": self.observed_platform_profile,
            "identityStatus": self.identity_status,
            "identitySource": self.identity_source,
            "capabilities": sorted(self.capabilities),
            "osName": self.os_name,
            "osVersion": self.os_version,
            "kernelVersion": self.kernel_version,
            "architecture": self.architecture,
            "accelerator": self.accelerator,
            "acceleratorRuntime": self.accelerator_runtime,
            "gstreamerVersion": self.gstreamer_version,
        }
        if self.identity_error:
            result["identityError"] = self.identity_error
        return result


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


def _run_version(command: str, *args: str) -> str:
    executable = shutil.which(command)
    if not executable:
        return "unknown"
    try:
        result = subprocess.run(
            [executable, *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    output = (result.stdout or result.stderr or "").strip()
    if not output:
        return "unknown"
    version_lines = [line for line in output.splitlines() if "version" in line.lower()]
    for line in [*version_lines, *output.splitlines()]:
        matches = re.findall(r"\d+(?:\.\d+){1,3}(?:[-+][A-Za-z0-9._-]+)?", line)
        if matches:
            # Tools such as `gst-launch-1.0` include a version-like suffix in
            # their executable name before the actual runtime version.
            return matches[-1]
    return output.splitlines()[0][:100]


def _run_hardware_probe(command: str, *args: str) -> str:
    executable = shutil.which(command)
    if not executable:
        return ""
    try:
        result = subprocess.run(
            [executable, *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return (result.stdout or result.stderr or "").strip()[:16_384]


def _os_version() -> str:
    try:
        value = platform.freedesktop_os_release().get("VERSION_ID", "")
    except (OSError, AttributeError):
        value = ""
    return str(value or platform.mac_ver()[0] or "unknown")


def collect_platform_probe(environ: Mapping[str, str] | None = None) -> PlatformProbe:
    values = os.environ if environ is None else environ
    system = platform.system() or "unknown"
    hardware_parts = [
        _read_text(Path("/proc/device-tree/model")),
        _read_text(Path("/proc/device-tree/compatible")),
        _read_text(Path("/sys/class/dmi/id/product_name")),
        _read_text(Path("/etc/nv_tegra_release")),
    ]
    if system.lower() == "linux":
        hardware_parts.extend(
            (
                _run_hardware_probe("lspci", "-nn"),
                _run_hardware_probe("lsusb"),
            )
        )
        if shutil.which("dxrt-cli"):
            hardware_parts.append("DEEPX DXRT runtime present")
    hardware_override = str(values.get("NUVION_OBSERVED_HARDWARE") or "").strip()
    if hardware_override:
        hardware_parts.append(hardware_override)
    hardware_text = "\n".join(part for part in hardware_parts if part)

    accelerator_runtime = str(
        values.get("NUVION_ACCELERATOR_RUNTIME_VERSION") or ""
    ).strip()
    if not accelerator_runtime:
        lowered = hardware_text.lower()
        if "jetson" in lowered or "tegra" in lowered or "orin" in lowered:
            accelerator_runtime = _read_text(Path("/etc/nv_tegra_release"))[:100]
        elif "deepx" in lowered or "dx-m1" in lowered:
            accelerator_runtime = _run_version("dxrt-cli", "--version")
        elif system.lower() == "darwin":
            accelerator_runtime = "MPS"
    if not accelerator_runtime:
        accelerator_runtime = "unknown"

    gstreamer_version = str(values.get("NUVION_GSTREAMER_VERSION") or "").strip()
    if not gstreamer_version:
        gstreamer_version = _run_version("gst-launch-1.0", "--version")

    return PlatformProbe(
        system=system,
        os_version=_os_version(),
        kernel_version=platform.release() or "unknown",
        architecture=platform.machine() or "unknown",
        hardware_text=hardware_text,
        accelerator_runtime=accelerator_runtime,
        gstreamer_version=gstreamer_version,
    )


def detect_platform_profile(probe: PlatformProbe) -> str:
    if probe.system.strip().lower() == "darwin":
        return PROFILE_MACOS_DEV
    evidence = probe.hardware_text.lower()
    if any(marker in evidence for marker in ("qcs9075", "iq-9075", "iq 9075")):
        return PROFILE_IQ9075_DEV
    if "ventuno q" in evidence or "ventuno_q" in evidence:
        return PROFILE_VENTUNO_Q
    if "orin nx" in evidence or "jetson orin" in evidence:
        return PROFILE_JETSON_ORIN_NX
    if "raspberry pi 5" in evidence or "raspberrypi,5" in evidence:
        return PROFILE_RPI5_DEEPX
    return PROFILE_UNKNOWN


def _accelerator_name(profile_name: str, hardware_text: str) -> str:
    evidence = hardware_text.lower()
    if profile_name == PROFILE_RPI5_DEEPX:
        return (
            "DEEPX DX-M1"
            if "deepx" in evidence or "dx-m1" in evidence
            else "DEEPX unconfirmed"
        )
    if profile_name == PROFILE_VENTUNO_Q:
        return "VENTUNO Q"
    if profile_name == PROFILE_JETSON_ORIN_NX:
        return "NVIDIA Jetson Orin NX"
    if profile_name == PROFILE_IQ9075_DEV:
        return "Qualcomm IQ-9075"
    if profile_name == PROFILE_MACOS_DEV:
        return "Apple MPS"
    return "unknown"


def _profile_hardware_confirmed(profile_name: str, probe: PlatformProbe) -> bool:
    evidence = probe.hardware_text.lower()
    if profile_name == PROFILE_RPI5_DEEPX:
        has_rpi5 = "raspberry pi 5" in evidence or "raspberrypi,5" in evidence
        has_deepx = "deepx" in evidence or "dx-m1" in evidence or "dxrt" in evidence
        return has_rpi5 and has_deepx
    if profile_name == PROFILE_VENTUNO_Q:
        return "ventuno q" in evidence or "ventuno_q" in evidence
    if profile_name == PROFILE_JETSON_ORIN_NX:
        return "orin nx" in evidence or "jetson orin" in evidence
    if profile_name == PROFILE_IQ9075_DEV:
        architecture = probe.architecture.strip().lower()
        return (
            probe.system.strip().lower() == "linux"
            and architecture in {"aarch64", "arm64"}
            and any(
                marker in evidence
                for marker in ("qcs9075", "iq-9075", "iq 9075")
            )
        )
    if profile_name == PROFILE_MACOS_DEV:
        return probe.system.strip().lower() == "darwin"
    return False


def _required_string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _load_declared_from_path(path: Path) -> DeclaredPlatform | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid identity file: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("identity file must contain a JSON object")  # noqa: TRY004
    return DeclaredPlatform(
        product_model=_required_string(payload, "productModel").upper(),
        hardware_revision=_required_string(payload, "hardwareRevision"),
        platform_profile=_required_string(payload, "platformProfile").lower(),
        source=str(path),
    )


def _validate_production_identity_file(metadata: os.stat_result) -> None:
    if metadata.st_uid != 0:
        raise ValueError("production identity file must be owned by root")
    if metadata.st_mode & 0o022:
        raise ValueError("production identity file must not be group/other writable")


def _load_declared_from_environment(
    values: Mapping[str, str],
) -> DeclaredPlatform | None:
    raw = {
        "productModel": str(values.get("NUVION_PRODUCT_MODEL") or "").strip(),
        "hardwareRevision": str(values.get("NUVION_HARDWARE_REVISION") or "").strip(),
        "platformProfile": str(values.get("NUVION_PLATFORM_PROFILE") or "").strip(),
    }
    if not any(raw.values()):
        return None
    if not all(raw.values()):
        raise ValueError(
            "NUVION product identity environment must provide all three fields"
        )
    return DeclaredPlatform(
        product_model=raw["productModel"].upper(),
        hardware_revision=raw["hardwareRevision"],
        platform_profile=raw["platformProfile"].lower(),
        source="environment",
    )


def resolve_platform_identity(
    *,
    environ: Mapping[str, str] | None = None,
    identity_path: str | Path | None = None,
    probe: PlatformProbe | None = None,
    require_secure_identity_file: bool = True,
    identity_file_stat: os.stat_result | None = None,
) -> PlatformIdentity:
    values = os.environ if environ is None else environ
    observed = probe or collect_platform_probe(values)
    observed_profile = detect_platform_profile(observed)
    configured_path = str(values.get("NUVION_DEVICE_IDENTITY_PATH") or "").strip()
    path = Path(identity_path or configured_path or DEFAULT_IDENTITY_PATH).expanduser()

    declaration_error: str | None = None
    declared: DeclaredPlatform | None = None
    try:
        declared = _load_declared_from_environment(values)
        if declared is None and path.is_file():
            if (
                require_secure_identity_file
                and observed.system.strip().lower() == "linux"
            ):
                _validate_production_identity_file(identity_file_stat or path.stat())
            declared = _load_declared_from_path(path)
    except (OSError, ValueError) as exc:
        declaration_error = str(exc)

    if declaration_error is not None:
        product_model = UNKNOWN_PRODUCT
        hardware_revision = "unknown"
        platform_profile = PROFILE_UNKNOWN
        status = IDENTITY_STATUS_INVALID
        source = str(path)
    elif declared is None and observed_profile == PROFILE_MACOS_DEV:
        product_model = MACOS_DEV
        hardware_revision = "DEV"
        platform_profile = PROFILE_MACOS_DEV
        status = IDENTITY_STATUS_DEV
        source = "auto-dev"
    elif declared is None:
        product_model = UNKNOWN_PRODUCT
        hardware_revision = "unknown"
        platform_profile = observed_profile
        status = IDENTITY_STATUS_UNPROVISIONED
        source = "none"
    else:
        product_model = declared.product_model
        hardware_revision = declared.hardware_revision
        platform_profile = declared.platform_profile
        source = declared.source
        expected_profile = PRODUCT_PROFILE.get(product_model)
        if product_model == IQ9075_DEV and declared.source == "environment":
            declaration_error = (
                "IQ9075_DEV identity must be declared by a secure identity file"
            )
            product_model = UNKNOWN_PRODUCT
            hardware_revision = "unknown"
            platform_profile = PROFILE_UNKNOWN
            status = IDENTITY_STATUS_INVALID
        elif expected_profile != platform_profile:
            status = IDENTITY_STATUS_MISMATCH
        elif observed_profile == PROFILE_UNKNOWN:
            status = IDENTITY_STATUS_UNVERIFIED
        elif observed_profile != platform_profile:
            status = IDENTITY_STATUS_MISMATCH
        elif not _profile_hardware_confirmed(platform_profile, observed):
            status = IDENTITY_STATUS_UNVERIFIED
        elif product_model in {MACOS_DEV, IQ9075_DEV}:
            status = IDENTITY_STATUS_DEV
        else:
            status = IDENTITY_STATUS_VERIFIED

    capabilities = (
        PROFILE_CAPABILITIES.get(platform_profile, frozenset())
        if status in {IDENTITY_STATUS_VERIFIED, IDENTITY_STATUS_DEV}
        else frozenset()
    )

    return PlatformIdentity(
        product_model=product_model,
        hardware_revision=hardware_revision,
        platform_profile=platform_profile,
        observed_platform_profile=observed_profile,
        identity_status=status,
        identity_source=source,
        capabilities=frozenset(capabilities),
        os_name=observed.system or "unknown",
        os_version=observed.os_version or "unknown",
        kernel_version=observed.kernel_version or "unknown",
        architecture=observed.architecture or "unknown",
        accelerator=_accelerator_name(observed_profile, observed.hardware_text),
        accelerator_runtime=observed.accelerator_runtime or "unknown",
        gstreamer_version=observed.gstreamer_version or "unknown",
        identity_error=declaration_error,
    )
