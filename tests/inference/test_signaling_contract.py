from __future__ import annotations

import unittest

from nuvion_app.inference.signaling_contract import (
    AGENT_COMMAND_QUEUE_DEST,
    AGENT_ERROR_QUEUE_DEST,
    EVENT_ACK_QUEUE_DEST,
    FLEET_COMMAND_QUEUE_DEST,
    REQUIRED_AGENT_SUBSCRIPTIONS,
)


class SignalingContractTest(unittest.TestCase):
    def test_agent_subscribes_to_command_error_and_event_ack_queues(self) -> None:
        self.assertEqual(AGENT_COMMAND_QUEUE_DEST, "/user/queue/command")
        self.assertEqual(AGENT_ERROR_QUEUE_DEST, "/user/queue/agent.error")
        self.assertEqual(EVENT_ACK_QUEUE_DEST, "/user/queue/event.ack")
        self.assertEqual(FLEET_COMMAND_QUEUE_DEST, "/user/queue/fleet.command")
        self.assertEqual(
            REQUIRED_AGENT_SUBSCRIPTIONS,
            (
                AGENT_COMMAND_QUEUE_DEST,
                AGENT_ERROR_QUEUE_DEST,
                EVENT_ACK_QUEUE_DEST,
                FLEET_COMMAND_QUEUE_DEST,
            ),
        )


if __name__ == "__main__":
    unittest.main()
