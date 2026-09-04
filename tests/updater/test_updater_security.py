from __future__ import annotations

import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nuvion_app.runtime.updater_client import (
    UpdaterClient,
    UpdaterClientError,
    build_updater_capability_telemetry,
)
from nuvion_updater.daemon import validate_host_runtime
from nuvion_updater.errors import UpdaterSecurityError
from nuvion_updater.protocol import (
    PeerCredentials,
    UpdaterUnixServer,
    get_peer_credentials,
)
from nuvion_updater.repository import (
    ContentAddressedReleaseRepository,
    read_ingested_request,
)
from nuvion_updater.store import UpdaterStore
from nuvion_updater.trust import DeviceBinding


class _ProtocolStub:
    def dispatch(self, request: dict[str, object]) -> dict[str, object]:
        return {"operation": request["operation"]}


class UpdaterSecurityTest(unittest.TestCase):
    def test_release_origin_is_https_and_cannot_escape_base_path(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            root.chmod(0o700)
            with self.assertRaises(UpdaterSecurityError):
                ContentAddressedReleaseRepository(
                    base_url="http://updates.example.test",
                    download_root=root / "http",
                    require_root_owner=False,
                )
            repository = ContentAddressedReleaseRepository(
                base_url="https://updates.example.test/nuvion",
                download_root=root / "https",
                require_root_owner=False,
            )
            with self.assertRaises(UpdaterSecurityError):
                repository._validate_final_url(
                    "https://updates.example.test/nuvion/%2e%2e/escape"
                )

    def test_iq9075_host_runtime_is_exactly_ubuntu_2404_python_312(self) -> None:
        binding = DeviceBinding(
            trust_domain="iq9075-dev",
            device_id="iq-device-1",
            space_id=3,
            product_model="IQ9075_DEV",
            platform_profile="iq9075_dev",
            hardware_revision="QCS9075-EVK",
            architecture="aarch64",
            docker_required=False,
        )
        with tempfile.TemporaryDirectory() as raw_root:
            os_release = Path(raw_root) / "os-release"
            os_release.write_text('ID=ubuntu\nVERSION_ID="24.04"\n', encoding="utf-8")
            os_release.parent.chmod(0o755)
            validate_host_runtime(
                binding,
                python_version=(3, 12),
                os_release_path=os_release,
                require_root_owner=False,
            )
            with self.assertRaisesRegex(SystemExit, "CPython 3.12"):
                validate_host_runtime(
                    binding,
                    python_version=(3, 11),
                    os_release_path=os_release,
                    require_root_owner=False,
                )
            os_release.write_text('ID=ubuntu\nVERSION_ID="22.04"\n', encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "Ubuntu 24.04"):
                validate_host_runtime(
                    binding,
                    python_version=(3, 12),
                    os_release_path=os_release,
                    require_root_owner=False,
                )

    def test_fixed_request_directory_rejects_traversal_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            root.chmod(0o700)
            request = root / "request.json"
            request.write_text('{"schemaVersion":1}', encoding="utf-8")
            request.chmod(0o600)
            self.assertEqual(
                read_ingested_request(root, "request.json", require_root_owner=False),
                b'{"schemaVersion":1}',
            )
            with self.assertRaises(UpdaterSecurityError):
                read_ingested_request(root, "../request.json", require_root_owner=False)
            link = root / "linked.json"
            link.symlink_to(request)
            with self.assertRaises((UpdaterSecurityError, OSError)):
                read_ingested_request(root, "linked.json", require_root_owner=False)

    def test_release_download_directory_count_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            root.chmod(0o700)
            repository = ContentAddressedReleaseRepository(
                base_url="https://updates.example.test/nuvion",
                download_root=root,
                require_root_owner=False,
                max_release_downloads=2,
            )
            (root / ("0" * 64)).mkdir(mode=0o700)
            (root / ("1" * 64)).mkdir(mode=0o700)
            with self.assertRaisesRegex(
                RuntimeError,
                "release download limit reached",
            ):
                repository.fetch_manifest("sha256:" + "2" * 64)

    def test_updater_journal_is_durably_bound_to_device_identity(self) -> None:
        identity: dict[str, object] = {
            "trustDomain": "iq9075-dev",
            "deviceId": "iq-device-1",
            "spaceId": 3,
            "productModel": "IQ9075_DEV",
            "platformProfile": "iq9075_dev",
            "hardwareRevision": "QCS9075-EVK",
            "architecture": "aarch64",
            "dockerRequired": False,
        }
        with tempfile.TemporaryDirectory() as raw_root:
            path = Path(raw_root) / "updater.sqlite3"
            store = UpdaterStore(path, require_root_owner=False)
            fingerprint = store.bind_device_identity(identity)
            self.assertEqual(store.bind_device_identity(identity), fingerprint)

            reopened = UpdaterStore(path, require_root_owner=False)
            other = {**identity, "deviceId": "iq-device-2"}
            with self.assertRaises(UpdaterSecurityError) as mismatch:
                reopened.bind_device_identity(other)
            self.assertEqual(mismatch.exception.code, "DEVICE_BINDING_MISMATCH")

    def test_kernel_peer_credentials_are_available_on_local_socket(self) -> None:
        left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        with left, right:
            credentials = get_peer_credentials(left)
        self.assertEqual(credentials.uid, os.getuid())
        self.assertEqual(credentials.gid, os.getgid())

    def test_server_rejects_unauthorized_peer_before_dispatch(self) -> None:
        listener = mock.Mock(spec=socket.socket)
        server = UpdaterUnixServer(
            listener=listener,
            protocol=_ProtocolStub(),  # type: ignore[arg-type]
            allowed_uids={12345},
        )
        server_side, client_side = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        with (
            server_side,
            client_side,
            mock.patch(
                "nuvion_updater.protocol.get_peer_credentials",
                return_value=PeerCredentials(pid=99, uid=54321, gid=54321),
            ),
        ):
            client_side.sendall(b'{"schemaVersion":1,"operation":"STATUS"}\n')
            client_side.shutdown(socket.SHUT_WR)
            server.handle_connection(server_side)
            response = json.loads(client_side.recv(4096).decode("utf-8"))
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["code"], "UNAUTHORIZED_PEER")

    def test_explicit_rollback_is_root_peer_only(self) -> None:
        listener = mock.Mock(spec=socket.socket)
        protocol = mock.Mock()
        server = UpdaterUnixServer(
            listener=listener,
            protocol=protocol,
            allowed_uids={501},
        )
        server_side, client_side = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        with (
            server_side,
            client_side,
            mock.patch(
                "nuvion_updater.protocol.get_peer_credentials",
                return_value=PeerCredentials(pid=99, uid=501, gid=501),
            ),
        ):
            client_side.sendall(
                b'{"commandId":"00000000-0000-4000-8000-000000000001",'
                b'"operation":"ROLLBACK","schemaVersion":1}\n'
            )
            client_side.shutdown(socket.SHUT_WR)
            server.handle_connection(server_side)
            response = json.loads(client_side.recv(4096).decode("utf-8"))
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["code"], "OPERATOR_AUTH_REQUIRED")
        protocol.dispatch.assert_not_called()

    def test_agent_client_authenticates_connected_peer_uid(self) -> None:
        left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        with left, right:
            UpdaterClient(expected_peer_uid=os.getuid())._validate_connected_peer(left)
            with self.assertRaises(UpdaterClientError) as raised:
                UpdaterClient(
                    expected_peer_uid=os.getuid() + 1
                )._validate_connected_peer(left)
        self.assertEqual(raised.exception.code, "UNSAFE_UPDATER_PEER")

    def test_response_disconnect_does_not_crash_privileged_server(self) -> None:
        listener = mock.Mock(spec=socket.socket)
        protocol = _ProtocolStub()
        connection = mock.Mock(spec=socket.socket)
        connection.recv.side_effect = [
            b'{"operation":"STATUS","schemaVersion":1}\n',
        ]
        connection.sendall.side_effect = BrokenPipeError
        server = UpdaterUnixServer(
            listener=listener,
            protocol=protocol,  # type: ignore[arg-type]
            allowed_uids={501},
        )
        with mock.patch(
            "nuvion_updater.protocol.get_peer_credentials",
            return_value=PeerCredentials(pid=99, uid=501, gid=501),
        ):
            self.assertIsNone(server.handle_connection(connection))
        connection.sendall.assert_called_once()

    def test_slowloris_request_is_bounded_by_absolute_deadline(self) -> None:
        listener = mock.Mock(spec=socket.socket)
        protocol = mock.Mock()
        connection = mock.Mock(spec=socket.socket)
        connection.recv.return_value = b"{"
        clock = mock.Mock(side_effect=[0.0, 9.0, 10.001])
        server = UpdaterUnixServer(
            listener=listener,
            protocol=protocol,
            allowed_uids={501},
            request_deadline_seconds=10.0,
            monotonic_clock=clock,
        )
        with mock.patch(
            "nuvion_updater.protocol.get_peer_credentials",
            return_value=PeerCredentials(pid=99, uid=501, gid=501),
        ):
            server.handle_connection(connection)

        response = json.loads(connection.sendall.call_args.args[0].decode("utf-8"))
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["code"], "REQUEST_TIMEOUT")
        protocol.dispatch.assert_not_called()
        self.assertEqual(connection.recv.call_count, 1)
        connection.settimeout.assert_called_once_with(1.0)

    def test_updater_version_requires_authenticated_live_helper_status(self) -> None:
        live_client = UpdaterClient(require_root_owner=False)
        with mock.patch.object(
            live_client,
            "status",
            return_value={
                "capabilityAvailable": True,
                "capabilityReason": "READY",
                "updaterVersion": "0.1.0",
            },
        ):
            live = build_updater_capability_telemetry(live_client)
        self.assertEqual(live["updaterVersion"], "0.1.0")
        self.assertTrue(live["agentUpdate"]["authenticatedHelper"])

        unavailable_client = UpdaterClient(require_root_owner=False)
        with mock.patch.object(
            unavailable_client,
            "status",
            side_effect=UpdaterClientError("UNSAFE_UPDATER_PEER", "peer is not root"),
        ):
            unavailable = build_updater_capability_telemetry(unavailable_client)
        self.assertEqual(unavailable["updaterVersion"], "unknown")
        self.assertFalse(unavailable["agentUpdate"]["authenticatedHelper"])

        malformed_client = UpdaterClient(require_root_owner=False)
        with mock.patch.object(
            malformed_client,
            "status",
            return_value={
                "capabilityAvailable": True,
                "capabilityReason": "READY",
                "updaterVersion": "latest",
            },
        ):
            malformed = build_updater_capability_telemetry(malformed_client)
        self.assertEqual(malformed["updaterVersion"], "unknown")
        self.assertFalse(malformed["agentUpdate"]["capabilityAvailable"])
        self.assertEqual(malformed["agentUpdate"]["reason"], "INVALID_UPDATER_VERSION")

    def test_root_update_phase_is_mapped_with_persistent_evidence(self) -> None:
        client = UpdaterClient(require_root_owner=False)
        update = {
            "commandId": "1a5ba2e1-3ee2-4ac7-8e57-79db2c373eaa",
            "phase": "ROLLED_BACK",
            "targetVersion": "0.1.116",
            "previousVersion": "0.1.115",
            "rollbackSlot": "bootstrap/0.1.115+dev1",
            "message": "functional health gate failed",
        }
        with mock.patch.object(
            client,
            "status",
            return_value={
                "capabilityAvailable": True,
                "capabilityReason": "READY",
                "updaterVersion": "0.1.0",
                "update": update,
            },
        ):
            telemetry = build_updater_capability_telemetry(client)

        self.assertEqual(telemetry["updatePhase"], "ROLLED_BACK")
        self.assertEqual(telemetry["updateEvidence"], update)
        self.assertNotIn("update", telemetry["agentUpdate"])

    def test_authenticated_slot_identity_is_forwarded_and_invalid_slot_fails_closed(
        self,
    ) -> None:
        client = UpdaterClient(require_root_owner=False)
        active_slot = "releases/" + "a" * 64
        previous_slot = "bootstrap/0.1.120"
        with mock.patch.object(
            client,
            "status",
            return_value={
                "capabilityAvailable": True,
                "capabilityReason": "READY",
                "updaterVersion": "0.2.0",
                "activeSlot": active_slot,
                "previousSlot": previous_slot,
                "update": None,
            },
        ):
            telemetry = build_updater_capability_telemetry(client)
        self.assertEqual(telemetry["agentUpdate"]["activeSlot"], active_slot)
        self.assertEqual(telemetry["agentUpdate"]["previousSlot"], previous_slot)
        self.assertTrue(telemetry["agentUpdate"]["capabilityAvailable"])

        with mock.patch.object(
            client,
            "status",
            return_value={
                "capabilityAvailable": True,
                "capabilityReason": "READY",
                "updaterVersion": "0.2.0",
                "activeSlot": "../../unsafe",
                "previousSlot": None,
                "update": None,
            },
        ):
            invalid = build_updater_capability_telemetry(client)
        self.assertFalse(invalid["agentUpdate"]["capabilityAvailable"])
        self.assertEqual(invalid["agentUpdate"]["reason"], "INVALID_SLOT_STATUS")
        self.assertIsNone(invalid["agentUpdate"]["activeSlot"])

    def test_server_monotonic_watchdog_runs_without_connections(self) -> None:
        listener = mock.Mock(spec=socket.socket)
        listener.accept.side_effect = KeyboardInterrupt
        watchdog = mock.Mock()
        clock = mock.Mock(side_effect=[0.0, 2.0])
        server = UpdaterUnixServer(
            listener=listener,
            protocol=_ProtocolStub(),  # type: ignore[arg-type]
            allowed_uids={os.getuid()},
            watchdog=watchdog,
            watchdog_interval_seconds=1.0,
            monotonic_clock=clock,
        )

        with self.assertRaises(KeyboardInterrupt):
            server.serve_forever()

        watchdog.assert_called_once_with()
        listener.settimeout.assert_called_once()


if __name__ == "__main__":
    unittest.main()
