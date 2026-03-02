from __future__ import annotations

import unittest
from unittest import mock

from nuvion_app.runtime import docker_manager


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


if __name__ == "__main__":
    unittest.main()
