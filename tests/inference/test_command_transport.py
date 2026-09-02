from __future__ import annotations

import unittest
import uuid

from nuvion_app.inference.command_inbox import CommandAck, deterministic_ack_id
from nuvion_app.inference.command_transport import (
    COMMAND_ACK_DESTINATION,
    COMMAND_WAKE_DESTINATION,
    FleetCommandHttpClient,
    FleetCommandTransportError,
    build_command_ack_payload,
    parse_command_wakeup,
)
from nuvion_app.inference.signaling_contract import (
    FLEET_COMMAND_QUEUE_DEST,
    REQUIRED_AGENT_SUBSCRIPTIONS,
)


class FakeResponse:
    def __init__(self, *, status: int, payload: object = None, body: str = "") -> None:
        self.status = status
        self.payload = payload
        self.body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def json(self, *, content_type=None):
        return self.payload

    async def text(self) -> str:
        return self.body


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, object], dict[str, str]]] = []

    def get(
        self, url: str, *, params: dict[str, object], headers: dict[str, str]
    ) -> FakeResponse:
        self.calls.append((url, params, headers))
        return self.response


class CommandTransportTest(unittest.IsolatedAsyncioTestCase):
    def test_wakeup_and_ack_payload_match_contract_destinations(self) -> None:
        command_id = str(uuid.uuid4())
        wakeup = parse_command_wakeup(f'{{"commandId":"{command_id}","sequence":42}}')
        ack = CommandAck(
            ack_id=deterministic_ack_id(command_id, "SUCCEEDED"),
            command_id=command_id,
            sequence=42,
            status="SUCCEEDED",
            observed_at="2026-09-01T02:00:03Z",
            reported_state={"configVersion": 11},
        )

        payload = build_command_ack_payload(ack)

        self.assertEqual(COMMAND_WAKE_DESTINATION, "/user/queue/fleet.command")
        self.assertEqual(FLEET_COMMAND_QUEUE_DEST, COMMAND_WAKE_DESTINATION)
        self.assertIn(FLEET_COMMAND_QUEUE_DEST, REQUIRED_AGENT_SUBSCRIPTIONS)
        self.assertEqual(COMMAND_ACK_DESTINATION, "/app/device/command.ack")
        self.assertEqual(wakeup.command_id, command_id)
        self.assertEqual(wakeup.sequence, 42)
        self.assertEqual(
            set(payload),
            {
                "ackId",
                "commandId",
                "sequence",
                "status",
                "observedAt",
                "code",
                "message",
                "reportedState",
            },
        )
        self.assertEqual(
            payload["ackId"], deterministic_ack_id(command_id, "SUCCEEDED")
        )
        self.assertEqual(payload["reportedState"], {"configVersion": 11})

        lifecycle = build_command_ack_payload(
            CommandAck(
                ack_id=deterministic_ack_id(command_id, "RECEIVED"),
                command_id=command_id,
                sequence=42,
                status="RECEIVED",
                observed_at="2026-09-01T02:00:02Z",
                reported_state=None,
            )
        )
        self.assertEqual(lifecycle["reportedState"], {})
        self.assertIsInstance(lifecycle["reportedState"], dict)

    async def test_http_pull_uses_after_sequence_and_bearer_auth(self) -> None:
        first_id = str(uuid.uuid4())
        second_id = str(uuid.uuid4())
        session = FakeSession(
            FakeResponse(
                status=200,
                payload={
                    "data": [
                        {"commandId": first_id, "sequence": 42, "compactJws": "a.b.c"},
                        {"commandId": second_id, "sequence": 43, "compactJws": "d.e.f"},
                    ]
                },
            )
        )
        client = FleetCommandHttpClient(
            base_url="https://api.example.test/",
            access_token_provider=lambda: "device-access-token",
            session=session,
        )

        page = await client.pull_after(41, limit=2)

        self.assertEqual(
            [(item.command_id, item.sequence) for item in page.commands],
            [(first_id, 42), (second_id, 43)],
        )
        self.assertEqual(page.next_after_sequence, 43)
        self.assertTrue(page.has_more)
        self.assertEqual(
            session.calls,
            [
                (
                    "https://api.example.test/devices/me/commands",
                    {"afterSequence": 41, "limit": 2},
                    {"Authorization": "Bearer device-access-token"},
                )
            ],
        )

    async def test_http_pull_rejects_non_progressing_sequence_page(self) -> None:
        command_id = str(uuid.uuid4())
        session = FakeSession(
            FakeResponse(
                status=200,
                payload={
                    "data": [
                        {"commandId": command_id, "sequence": 42, "compactJws": "a.b.c"}
                    ]
                },
            )
        )
        client = FleetCommandHttpClient(
            base_url="https://api.example.test",
            access_token_provider=lambda: "device-access-token",
            session=session,
        )

        with self.assertRaises(FleetCommandTransportError) as raised:
            await client.pull_after(42)

        self.assertEqual(raised.exception.code, "NON_MONOTONIC_RESPONSE")

    def test_invalid_wakeup_and_ack_schema_are_rejected(self) -> None:
        with self.assertRaises(FleetCommandTransportError) as wakeup_error:
            parse_command_wakeup('{"commandId":"not-a-uuid","sequence":0}')
        self.assertEqual(wakeup_error.exception.code, "INVALID_WAKEUP")
        self.assertFalse(wakeup_error.exception.retryable)

        command_id = str(uuid.uuid4())
        invalid_ack = CommandAck(
            ack_id=deterministic_ack_id(command_id, "FAILED"),
            command_id=command_id,
            sequence=1,
            status="FAILED",
            observed_at="2026-09-01T02:00:03",
        )
        with self.assertRaises(ValueError):
            build_command_ack_payload(invalid_ack)

    async def test_server_failure_is_classified_retryable(self) -> None:
        session = FakeSession(FakeResponse(status=503, body="temporarily unavailable"))
        client = FleetCommandHttpClient(
            base_url="https://api.example.test",
            access_token_provider=lambda: "device-access-token",
            session=session,
        )

        with self.assertRaises(FleetCommandTransportError) as raised:
            await client.pull_after(0)

        self.assertEqual(raised.exception.code, "COMMAND_PULL_FAILED")
        self.assertEqual(raised.exception.status, 503)
        self.assertTrue(raised.exception.retryable)


if __name__ == "__main__":
    unittest.main()
