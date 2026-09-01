from __future__ import annotations

import argparse
import os
import pwd
import sys
from pathlib import Path

from nuvion_updater.controller import UpdaterController
from nuvion_updater.protocol import UpdaterProtocol, UpdaterUnixServer, systemd_listener
from nuvion_updater.repository import ContentAddressedReleaseRepository
from nuvion_updater.secure_io import read_fixed_regular_file
from nuvion_updater.slots import ReleaseSlotManager
from nuvion_updater.store import UpdaterStore
from nuvion_updater.systemd_runtime import SystemdRuntime
from nuvion_updater.trust import (
    DeviceBinding,
    build_root_command_verifier,
    load_device_binding,
    load_release_keyring,
)
from nuvion_updater.version import UPDATER_VERSION

DEFAULT_BINDING = Path("/etc/nuvion-updater/device-binding.json")
DEFAULT_COMMAND_KEYRING = Path("/etc/nuvion-updater/command-keyring.json")
DEFAULT_RELEASE_KEYRING = Path("/etc/nuvion-updater/release-keyring.json")
DEFAULT_STATE_DB = Path("/var/lib/nuvion-updater/updater.sqlite3")
DEFAULT_DOWNLOADS = Path("/var/lib/nuvion-updater/downloads")
DEFAULT_INSTALL_ROOT = Path("/opt/nuv-agent")


def validate_host_runtime(
    binding: DeviceBinding,
    *,
    python_version: tuple[int, int] | None = None,
    os_release_path: Path = Path("/usr/lib/os-release"),
    require_root_owner: bool = True,
) -> None:
    if getattr(binding, "platform_profile", None) != "iq9075_dev":
        return
    active_python = python_version or (sys.version_info.major, sys.version_info.minor)
    if active_python != (3, 12):
        raise SystemExit("IQ9075 updater requires CPython 3.12")
    raw = read_fixed_regular_file(
        os_release_path.parent,
        os_release_path.name,
        max_bytes=64 * 1024,
        require_root_owner=require_root_owner,
    )
    values: dict[str, str] = {}
    for line in raw.decode("utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    if values.get("ID") != "ubuntu" or values.get("VERSION_ID") != "24.04":
        raise SystemExit("IQ9075 updater requires Ubuntu 24.04")


def _runtime_helper_ready() -> bool:
    # A socket pathname is not proof of the audited Triton/Docker lifecycle
    # contract. Until that typed helper and peer-authenticated status protocol
    # ship, every dockerRequired product profile remains fail-closed.
    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Privileged NUVION Agent updater")
    parser.add_argument("--repository-base-url", required=True)
    parser.add_argument("--device-binding", type=Path, default=DEFAULT_BINDING)
    parser.add_argument("--command-keyring", type=Path, default=DEFAULT_COMMAND_KEYRING)
    parser.add_argument("--release-keyring", type=Path, default=DEFAULT_RELEASE_KEYRING)
    parser.add_argument("--state-db", type=Path, default=DEFAULT_STATE_DB)
    parser.add_argument("--downloads", type=Path, default=DEFAULT_DOWNLOADS)
    parser.add_argument("--install-root", type=Path, default=DEFAULT_INSTALL_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    if os.geteuid() != 0:
        raise SystemExit("nuvion-updater must run as root")
    args = build_parser().parse_args(argv)
    binding = load_device_binding(args.device_binding)
    validate_host_runtime(binding)
    command_verifier = build_root_command_verifier(
        binding=binding,
        command_keyring_path=args.command_keyring,
    )
    release_keyring = load_release_keyring(
        args.release_keyring,
        expected_trust_domain=binding.trust_domain,
    )
    slots = ReleaseSlotManager(args.install_root)
    systemd_runtime = SystemdRuntime(slots=slots, binding=binding)
    store = UpdaterStore(args.state_db)
    store.bind_device_identity(
        {
            "trustDomain": binding.trust_domain,
            "deviceId": binding.device_id,
            "spaceId": binding.space_id,
            "productModel": binding.product_model,
            "platformProfile": binding.platform_profile,
            "hardwareRevision": binding.hardware_revision,
            "architecture": binding.architecture,
            "dockerRequired": binding.docker_required,
        }
    )
    controller = UpdaterController(
        store=store,
        slots=slots,
        repository=ContentAddressedReleaseRepository(
            base_url=args.repository_base_url,
            download_root=args.downloads,
        ),
        command_verifier=command_verifier,
        release_keyring=release_keyring,
        binding=binding,
        updater_version=UPDATER_VERSION,
        privileged_runtime_ready=_runtime_helper_ready,
        activation_callback=systemd_runtime.restart_agent,
        boot_health_check=systemd_runtime.boot_health_check,
        functional_health_check=systemd_runtime.functional_health_check,
        rollback_boot_health_check=systemd_runtime.rollback_boot_health_check,
        safe_stop_callback=systemd_runtime.safe_stop,
    )
    controller.recover()
    try:
        agent_uid = pwd.getpwnam("nuvion").pw_uid
    except KeyError as exc:
        raise SystemExit("nuvion service user is missing") from exc
    listener = systemd_listener()
    server = UpdaterUnixServer(
        listener=listener,
        protocol=UpdaterProtocol(controller),
        allowed_uids={0, agent_uid},
        watchdog=controller.watchdog_tick,
        watchdog_interval_seconds=1.0,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
