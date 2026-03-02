from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from nuvion_app.runtime.errors import BootstrapError

log = logging.getLogger(__name__)

_HOMEBREW_INSTALL_SCRIPT = "https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh"
_PATH_HINTS = [
    "/opt/homebrew/bin",
    "/opt/homebrew/sbin",
    "/usr/local/bin",
    "/usr/local/sbin",
    "/Applications/Docker.app/Contents/Resources/bin",
]


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _emit_progress(message: str) -> None:
    sys.stderr.write(f"[BOOTSTRAP] {message}\n")
    sys.stderr.flush()


def _augment_path(path_value: str | None) -> str:
    current_parts = [part for part in (path_value or "").split(os.pathsep) if part]
    seen = set(current_parts)
    for hint in _PATH_HINTS:
        if Path(hint).exists() and hint not in seen:
            current_parts.append(hint)
            seen.add(hint)
    return os.pathsep.join(current_parts)


def command_exists(command: str) -> bool:
    return shutil.which(command, path=_augment_path(os.environ.get("PATH", ""))) is not None


def detect_brew_path() -> str | None:
    for candidate in (
        os.environ.get("HOMEBREW_BIN", "").strip(),
        shutil.which("brew") or "",
        "/opt/homebrew/bin/brew",
        "/usr/local/bin/brew",
    ):
        if candidate and Path(candidate).exists():
            return candidate
    return None


def run_command(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    check: bool = True,
    capture_output: bool = False,
    as_root: bool = False,
) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    merged_env["PATH"] = _augment_path(merged_env.get("PATH", ""))
    if env:
        merged_env.update(env)
        merged_env["PATH"] = _augment_path(merged_env.get("PATH", ""))

    final_command = list(command)
    if as_root and os.geteuid() != 0:
        final_command = ["sudo", "-n", *final_command]

    return subprocess.run(
        final_command,
        check=check,
        text=True,
        capture_output=capture_output,
        env=merged_env,
    )


def ensure_homebrew_installed() -> str:
    brew_path = detect_brew_path()
    if brew_path:
        return brew_path

    if not _truthy(os.getenv("NUVION_HOMEBREW_AUTOINSTALL"), default=True):
        raise BootstrapError("brew_install_failed", "Homebrew is not installed.", retryable=False)

    log.info("[BOOTSTRAP] Homebrew missing. Installing Homebrew in non-interactive mode.")
    _emit_progress("Homebrew가 없어 자동 설치를 시작합니다.")
    try:
        run_command(
            [
                "/bin/bash",
                "-c",
                f"/bin/bash -c \"$(curl -fsSL {_HOMEBREW_INSTALL_SCRIPT})\"",
            ],
            env={"NONINTERACTIVE": "1"},
            check=True,
            capture_output=False,
        )
    except Exception as exc:
        raise BootstrapError("brew_install_failed", f"Failed to install Homebrew: {exc}") from exc

    brew_path = detect_brew_path()
    if not brew_path:
        raise BootstrapError("brew_install_failed", "Homebrew install finished but brew was not found.")

    return brew_path


def brew_install(packages: Iterable[str]) -> None:
    package_list = [pkg for pkg in packages if pkg]
    if not package_list:
        return

    brew_path = ensure_homebrew_installed()

    for package in package_list:
        check_result = run_command([brew_path, "list", package], check=False, capture_output=True)
        if check_result.returncode == 0:
            _emit_progress(f"brew 패키지 이미 설치됨: {package}")
            continue

        try:
            log.info("[BOOTSTRAP] Installing package via brew: %s", package)
            _emit_progress(f"brew install 진행 중: {package}")
            run_command([brew_path, "install", package], check=True)
        except Exception as exc:
            raise BootstrapError("docker_install_failed", f"brew install {package} failed: {exc}") from exc


def apt_install(packages: Iterable[str]) -> None:
    package_list = [pkg for pkg in packages if pkg]
    if not package_list:
        return

    try:
        _emit_progress("apt update 진행 중...")
        run_command(["apt-get", "update"], as_root=True)
        _emit_progress(f"apt install 진행 중: {' '.join(package_list)}")
        run_command(["apt-get", "install", "-y", *package_list], as_root=True)
    except Exception as exc:
        raise BootstrapError("docker_install_failed", f"apt install failed: {exc}") from exc


def ensure_nvidia_container_toolkit() -> None:
    if command_exists("nvidia-ctk"):
        return

    if not _truthy(os.getenv("NUVION_DOCKER_AUTOINSTALL"), default=True):
        return

    try:
        apt_install(["nvidia-container-toolkit"])
        if command_exists("nvidia-ctk"):
            run_command(["nvidia-ctk", "runtime", "configure", "--runtime=docker"], as_root=True, check=False)
            run_command(["systemctl", "restart", "docker"], as_root=True, check=False)
    except BootstrapError as exc:
        # Toolkit install is best-effort. Triton startup health-check will still decide final readiness.
        log.warning("[BOOTSTRAP] code=%s message=%s", exc.code, str(exc))
