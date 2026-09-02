from __future__ import annotations

import unittest

from nuvion_app.inference.webrtc_signaling import (
    UPLINK_MODE_WEBRTC,
    enforce_h264_offer_parameters,
    h264_level_from_profile_level_id,
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
                "turn://1700000000%3Adevice-1:c2VjcmV0Og%3D%3D@stunner.example.com:3478",
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

    def test_h264_offer_is_canonicalized_to_ingest_contract(self) -> None:
        offer = (
            "v=0\r\n"
            "m=video 9 UDP/TLS/RTP/SAVPF 96\r\n"
            "a=rtpmap:96 H264/90000\r\n"
            "a=fmtp:96 packetization-mode=0;profile-level-id=42c01e;"
            "sprop-parameter-sets=Z0LAHtoCgPaE,aM4G4g==\r\n"
        )

        canonical = enforce_h264_offer_parameters(
            offer,
            profile_level_id="42E01F",
            packetization_mode="1",
            level_asymmetry_allowed="1",
        )

        self.assertIn(
            "a=fmtp:96 level-asymmetry-allowed=1;packetization-mode=1;"
            "profile-level-id=42e01f;sprop-parameter-sets=Z0LAHtoCgPaE,aM4G4g==",
            canonical,
        )
        self.assertNotIn("42c01e", canonical)
        self.assertTrue(canonical.endswith("\r\n"))
        self.assertEqual(h264_level_from_profile_level_id("42e01f"), "3.1")

    def test_h264_offer_inserts_missing_fmtp_and_rejects_ambiguity(self) -> None:
        offer = (
            "v=0\n"
            "m=video 9 UDP/TLS/RTP/SAVPF 96\n"
            "a=rtpmap:96 H264/90000\n"
        )
        canonical = enforce_h264_offer_parameters(
            offer,
            profile_level_id="42e01f",
            packetization_mode="1",
            level_asymmetry_allowed="1",
        )
        self.assertIn("a=fmtp:96 ", canonical)
        self.assertIn("profile-level-id=42e01f", canonical)

        invalid_offers = (
            offer.replace("H264/90000", "VP8/90000"),
            offer + "a=fmtp:96 profile-level-id=42c01e\n"
            "a=fmtp:96 profile-level-id=42e01f\n",
            offer + "a=fmtp:96 profile-level-id=42c01e;"
            "profile-level-id=42e01f\n",
        )
        for invalid in invalid_offers:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    enforce_h264_offer_parameters(
                        invalid,
                        profile_level_id="42e01f",
                        packetization_mode="1",
                        level_asymmetry_allowed="1",
                    )

        for invalid_profile in ("42c01", "42e01g", "42e000"):
            with self.subTest(invalid_profile=invalid_profile):
                with self.assertRaises(ValueError):
                    enforce_h264_offer_parameters(
                        offer,
                        profile_level_id=invalid_profile,
                        packetization_mode="1",
                        level_asymmetry_allowed="1",
                    )


if __name__ == "__main__":
    unittest.main()
