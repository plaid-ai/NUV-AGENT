from __future__ import annotations

import logging
import os
from typing import Tuple
from urllib.parse import urlparse

from nuvion_app.runtime.errors import BootstrapError
from nuvion_app.runtime.platform_installer import (
    apt_install,
    brew_install,
    command_exists,
    ensure_homebrew_installed,
    ensure_nvidia_container_toolkit,
    run_command,
)

log = logging.getLogger(__name__)


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_triton_host_port(url: str) -> Tuple[str, int]:
    normalized = url.strip()
    if not normalized:
        return "localhost", 8000
    if "://" not in normalized:
        normalized = f"http://{normalized}"

    parsed = urlparse(normalized)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8000
    return host, port


def is_local_host(host: str) -> bool:
    return host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def docker_info_ok() -> bool:
    if not command_exists("docker"):
        return False
    result = run_command(["docker", "info"], check=False, capture_output=True)
    return result.returncode == 0


def _ensure_docker_cli_mac() -> None:
    ensure_homebrew_installed()
    if command_exists("docker"):
        return

    if not _truthy(os.getenv("NUVION_DOCKER_AUTOINSTALL"), default=True):
        raise BootstrapError("docker_install_failed", "Docker CLI is not installed.", retryable=False)

    brew_install(["docker", "colima"])


def _ensure_docker_cli_linux() -> None:
    if command_exists("docker"):
        return

    if not _truthy(os.getenv("NUVION_DOCKER_AUTOINSTALL"), default=True):
        raise BootstrapError("docker_install_failed", "Docker is not installed.", retryable=False)

    apt_install(["docker.io"])


def _start_docker_daemon_linux() -> None:
    # Best effort: systemctl first, then service.
    systemctl_result = run_command(["systemctl", "start", "docker"], as_root=True, check=False)
    if systemctl_result.returncode == 0:
        return
    run_command(["service", "docker", "start"], as_root=True, check=False)


def _ensure_colima_running() -> None:
    if not command_exists("colima"):
        if not _truthy(os.getenv("NUVION_DOCKER_AUTOINSTALL"), default=True):
            raise BootstrapError("docker_daemon_unavailable", "colima is not installed.", retryable=False)
        brew_install(["colima"])

    status = run_command(["colima", "status"], check=False, capture_output=True)
    if status.returncode == 0:
        return

    if not _truthy(os.getenv("NUVION_DOCKER_AUTOSTART"), default=True):
        raise BootstrapError("docker_daemon_unavailable", "Docker daemon is down and autostart is disabled.", retryable=False)

    log.info("[BOOTSTRAP] Docker daemon unavailable. Starting Colima fallback.")
    run_command(["colima", "start", "--cpu", "4", "--memory", "8", "--disk", "40"], check=True)


def ensure_docker_ready(triton_url: str) -> None:
    host, _ = parse_triton_host_port(triton_url)
    if not is_local_host(host):
        return

    if os.getenv("NUVION_DOCKER_AUTOSTART", "true").strip().lower() not in {"1", "true", "yes", "on"}:
        if docker_info_ok():
            return
        raise BootstrapError("docker_daemon_unavailable", "Docker daemon is unavailable and autostart is disabled.", retryable=False)

    if os.uname().sysname.lower() == "darwin":
        _ensure_docker_cli_mac()
    else:
        _ensure_docker_cli_linux()

    if docker_info_ok():
        return

    if os.uname().sysname.lower() == "darwin":
        _ensure_colima_running()
    else:
        _start_docker_daemon_linux()
        ensure_nvidia_container_toolkit()

    if not docker_info_ok():
        raise BootstrapError("docker_daemon_unavailable", "Docker daemon is not available.")


def container_exists(name: str) -> bool:
    result = run_command(["docker", "inspect", name], check=False, capture_output=True)
    return result.returncode == 0


def container_running(name: str) -> bool:
    result = run_command(["docker", "inspect", "-f", "{{.State.Running}}", name], check=False, capture_output=True)
    return result.returncode == 0 and result.stdout.strip().lower() == "true"


def start_container(name: str) -> None:
    result = run_command(["docker", "start", name], check=False, capture_output=True)
    if result.returncode != 0:
        raise BootstrapError("triton_health_failed", f"Failed to start existing container '{name}'.")


def remove_container(name: str) -> None:
    run_command(["docker", "rm", "-f", name], check=False)


def run_triton_container(name: str, image: str, model_repository: str, host_port: int) -> None:
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "-p",
        f"{host_port}:8000",
        "-v",
        f"{model_repository}:/models",
        image,
        "tritonserver",
        "--model-repository=/models",
    ]

    result = run_command(cmd, check=False, capture_output=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise BootstrapError("triton_health_failed", f"Failed to run Triton container: {stderr}")
