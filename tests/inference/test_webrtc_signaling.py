from __future__ import annotations

import unittest

from nuvion_app.inference.webrtc_signaling import (
    UPLINK_MODE_WEBRTC,
    negotiate_stomp_send_interval_ms,
    normalize_uplink_mode,
    parse_stomp_heartbeat_header,
    parse_ice_servers,
    to_gst_ice_server_config,
)


class WebRTCSignalingTest(unittest.TestCase):
    def test_normalize_uplink_mode_defaults_to_webrtc(self) -> None:
        self.assertEqual(normalize_uplink_mode(None), UPLINK_MODE_WEBRTC)
        self.assertEqual(normalize_uplink_mode("unknown"), UPLINK_MODE_WEBRTC)
        self.assertEqual(normalize_uplink_mode("RTP"), UPLINK_MODE_WEBRTC)

    def test_parse_ice_servers_accepts_json_string(self) -> None:
        raw = '[{"urls":["turn:turn.example.com:3478?transport=udp"],"username":"user","credential":"pass"}]'
        parsed = parse_ice_servers(raw)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["username"], "user")

    def test_to_gst_ice_server_config_converts_turn_and_stun(self) -> None:
        stun_server, turn_servers = to_gst_ice_server_config(
            [
                {
                    "urls": [
                        "stun:stunner.example.com:3478",
                        "turn:stunner.example.com:3478?transport=udp",
                    ],
                    "username": "1700000000:device-1",
                    "credential": "c2VjcmV0Og==",
                }
            ]
        )
        self.assertEqual(stun_server, "stun://stunner.example.com:3478")
        self.assertEqual(
            turn_servers,
            [
                "turn://1700000000%3Adevice-1:c2VjcmV0Og%3D%3D@stunner.example.com:3478?transport=udp",
            ],
        )

    def test_parse_stomp_heartbeat_header(self) -> None:
        self.assertEqual(parse_stomp_heartbeat_header("10000,10000"), (10000, 10000))
        self.assertEqual(parse_stomp_heartbeat_header("0,0"), (0, 0))
        self.assertEqual(parse_stomp_heartbeat_header(None), (0, 0))
        self.assertEqual(parse_stomp_heartbeat_header("bad"), (0, 0))

    def test_negotiate_stomp_send_interval_ms(self) -> None:
        self.assertEqual(negotiate_stomp_send_interval_ms(10000, "10000,10000"), 10000)
        self.assertEqual(negotiate_stomp_send_interval_ms(10000, "0,15000"), 15000)
        self.assertIsNone(negotiate_stomp_send_interval_ms(10000, "10000,0"))


if __name__ == "__main__":
    unittest.main()
