from __future__ import annotations

import unittest
from unittest import mock

from nuvion_app.runtime import docker_manager


class _Uname:
    def __init__(self, sysname: str):
        self.sysname = sysname


class DockerManagerTest(unittest.TestCase):
    def test_parse_triton_host_port(self) -> None:
        host, port = docker_manager.parse_triton_host_port("127.0.0.1:8000")
        self.assertEqual(host, "127.0.0.1")
        self.assertEqual(port, 8000)

    def test_local_host(self) -> None:
        self.assertTrue(docker_manager.is_local_host("localhost"))
        self.assertFalse(docker_manager.is_local_host("api.nuvion-dev.plaidai.io"))

    def test_skip_remote_host(self) -> None:
        with mock.patch.object(docker_manager, "_ensure_docker_cli_mac") as mac_cli:
            with mock.patch.object(docker_manager, "_ensure_docker_cli_linux") as linux_cli:
                docker_manager.ensure_docker_ready("https://api.nuvion-dev.plaidai.io:8000")
                mac_cli.assert_not_called()
                linux_cli.assert_not_called()

    def test_macos_prefers_docker_desktop_before_colima(self) -> None:
        with mock.patch.object(docker_manager.os, "uname", return_value=_Uname("Darwin")):
            with mock.patch.object(docker_manager, "_ensure_docker_cli_mac"):
                with mock.patch.object(docker_manager, "docker_info_ok", return_value=False):
                    with mock.patch.object(docker_manager, "_start_docker_desktop_if_available", return_value=True) as desktop_mock:
                        with mock.patch.object(docker_manager, "_ensure_colima_running") as colima_mock:
                            docker_manager.ensure_docker_ready("127.0.0.1:8000")
        desktop_mock.assert_called_once()
        colima_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
