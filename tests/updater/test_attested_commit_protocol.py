from __future__ import annotations

import base64
import json
import os
import socket
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from nuvion_app.runtime.updater_client import UpdaterClient
from nuvion_updater.errors import UpdaterSecurityError
from nuvion_updater.protocol import PeerCredentials, UpdaterProtocol, UpdaterUnixServer
from nuvion_updater.trust import load_health_attestation_keyring


class AttestedCommitProtocolTest(unittest.TestCase):
    def test_begin_and_commit_require_and_forward_kernel_peer_pid(self) -> None:
        controller = mock.Mock()
        controller.begin_commit_gate.return_value = SimpleNamespace(
            public_dict=lambda: {"gateId": "gate"}
        )
        controller.commit.return_value = SimpleNamespace(
            public_dict=lambda: {"phase": "COMMITTED"}
        )
        protocol = UpdaterProtocol(controller)
        command_id = "00000000-0000-4000-8000-000000000001"
        gate_id = "00000000-0000-4000-8000-000000000002"
        peer = PeerCredentials(pid=412, uid=1000, gid=1000)

        self.assertEqual(
            protocol.dispatch(
                {
                    "schemaVersion": 1,
                    "operation": "BEGIN_COMMIT_GATE",
                    "commandId": command_id,
                },
                peer=peer,
            ),
            {"gateId": "gate"},
        )
        controller.begin_commit_gate.assert_called_once_with(command_id, peer_pid=412)

        self.assertEqual(
            protocol.dispatch(
                {
                    "schemaVersion": 1,
                    "operation": "COMMIT",
                    "commandId": command_id,
                    "gateId": gate_id,
                    "healthAttestationJws": "a.b.c",
                },
                peer=peer,
            ),
            {"phase": "COMMITTED"},
        )
        controller.commit.assert_called_once_with(
            command_id,
            gate_id=gate_id,
            health_attestation_jws="a.b.c",
            peer_pid=412,
        )

        with self.assertRaises(UpdaterSecurityError) as unavailable:
            protocol.dispatch(
                {
                    "schemaVersion": 1,
                    "operation": "BEGIN_COMMIT_GATE",
                    "commandId": command_id,
                },
                peer=PeerCredentials(pid=None, uid=1000, gid=1000),
            )
        self.assertEqual(unavailable.exception.code, "PEER_PID_UNAVAILABLE")

    def test_commit_schema_rejects_missing_attestation_and_unknown_fields(self) -> None:
        protocol = UpdaterProtocol(mock.Mock())
        base = {
            "schemaVersion": 1,
            "operation": "COMMIT",
            "commandId": "00000000-0000-4000-8000-000000000001",
            "gateId": "00000000-0000-4000-8000-000000000002",
        }
        peer = PeerCredentials(pid=412, uid=1000, gid=1000)
        for request in (base, {**base, "healthAttestationJws": "a.b.c", "url": "x"}):
            with self.subTest(request=request), self.assertRaises(
                UpdaterSecurityError
            ) as raised:
                protocol.dispatch(request, peer=peer)
            self.assertEqual(raised.exception.code, "INVALID_REQUEST")

    def test_unix_server_passes_authenticated_peer_to_protocol(self) -> None:
        protocol = mock.Mock()
        protocol.dispatch.return_value = {"ok": "result"}
        server = UpdaterUnixServer(
            listener=mock.Mock(spec=socket.socket),
            protocol=protocol,
            allowed_uids={os.getuid()},
        )
        server_side, client_side = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        peer = PeerCredentials(pid=412, uid=os.getuid(), gid=os.getgid())
        with server_side, client_side, mock.patch(
            "nuvion_updater.protocol.get_peer_credentials", return_value=peer
        ):
            client_side.sendall(b'{"operation":"STATUS","schemaVersion":1}\n')
            client_side.shutdown(socket.SHUT_WR)
            server.handle_connection(server_side)
            response = json.loads(client_side.recv(4096))
        self.assertTrue(response["ok"])
        protocol.dispatch.assert_called_once_with(
            {"operation": "STATUS", "schemaVersion": 1}, peer=peer
        )

    def test_agent_client_emits_exact_begin_and_commit_schema(self) -> None:
        client = UpdaterClient(require_root_owner=False)
        command_id = str(uuid.uuid4())
        gate_id = str(uuid.uuid4())
        with mock.patch.object(client, "_request", return_value={}) as request:
            client.begin_commit_gate(command_id)
            client.commit(
                command_id,
                gate_id=gate_id,
                health_attestation_jws="header.claims.signature",
            )
        self.assertEqual(
            request.call_args_list,
            [
                mock.call(
                    {
                        "schemaVersion": 1,
                        "operation": "BEGIN_COMMIT_GATE",
                        "commandId": command_id,
                    }
                ),
                mock.call(
                    {
                        "schemaVersion": 1,
                        "operation": "COMMIT",
                        "commandId": command_id,
                        "gateId": gate_id,
                        "healthAttestationJws": "header.claims.signature",
                    }
                ),
            ],
        )

    def test_health_keyring_is_purpose_and_trust_domain_bound(self) -> None:
        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            path = root / "health-attestation-keyring.json"
            payload = {
                "schemaVersion": 1,
                "trustDomain": "production",
                "purpose": "agent-update-health-attestation",
                "keys": {
                    "health-2026": base64.b64encode(public_key).decode("ascii")
                },
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            path.chmod(0o600)
            keyring = load_health_attestation_keyring(
                path,
                expected_trust_domain="production",
                require_root_owner=False,
            )
            self.assertIsNotNone(keyring.get("health-2026"))

            for field, value in (
                ("trustDomain", "staging"),
                ("purpose", "fleet-command"),
            ):
                with self.subTest(field=field):
                    path.write_text(
                        json.dumps({**payload, field: value}), encoding="utf-8"
                    )
                    with self.assertRaises(UpdaterSecurityError):
                        load_health_attestation_keyring(
                            path,
                            expected_trust_domain="production",
                            require_root_owner=False,
                        )


if __name__ == "__main__":
    unittest.main()
