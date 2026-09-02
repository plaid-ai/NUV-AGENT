from __future__ import annotations

import importlib
import sys
import types
import unittest


class _FakeGLib:
    calls: list[tuple[object, tuple[object, ...]]] = []

    @classmethod
    def idle_add(cls, func: object, *args: object) -> int:
        cls.calls.append((func, args))
        return len(cls.calls)


class _FakePromise:
    @staticmethod
    def new_with_change_func(*_args: object, **_kwargs: object) -> object:
        return object()

    @staticmethod
    def new() -> object:
        return object()


class _FakeSessionDescription:
    @staticmethod
    def new(*_args: object, **_kwargs: object) -> object:
        return object()


class _ReplyPromise:
    def __init__(self, reply: dict[str, object]) -> None:
        self.reply = reply

    def get_reply(self) -> dict[str, object]:
        return self.reply


def _install_fake_gi() -> None:
    gi = types.ModuleType("gi")
    gi.require_version = lambda *_args, **_kwargs: None

    repository = types.ModuleType("gi.repository")
    repository.GLib = _FakeGLib
    repository.Gst = types.SimpleNamespace(
        Pipeline=object,
        Element=object,
        Promise=_FakePromise,
    )
    repository.GstSdp = types.SimpleNamespace(
        SDPMessage=types.SimpleNamespace(new=lambda: (0, object())),
        SDPResult=types.SimpleNamespace(OK=0),
        sdp_message_parse_buffer=lambda *_args, **_kwargs: 0,
    )
    repository.GstWebRTC = types.SimpleNamespace(
        WebRTCICETransportPolicy=types.SimpleNamespace(RELAY="relay", ALL="all"),
        WebRTCBundlePolicy=types.SimpleNamespace(MAX_BUNDLE="max-bundle"),
        WebRTCSDPType=types.SimpleNamespace(ANSWER="answer"),
        WebRTCSessionDescription=_FakeSessionDescription,
    )
    gi.repository = repository

    sys.modules["gi"] = gi
    sys.modules["gi.repository"] = repository


class WebRTCUplinkControllerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _install_fake_gi()
        sys.modules.pop("nuvion_app.inference.webrtc_uplink", None)
        cls.module = importlib.import_module("nuvion_app.inference.webrtc_uplink")

    def setUp(self) -> None:
        _FakeGLib.calls.clear()

    def test_start_ignores_duplicate_session(self) -> None:
        controller = self.module.WebRTCUplinkController(send_message=lambda *_args: True)

        payload = {
            "broadcastId": "device-1",
            "sessionId": "session-1",
            "forceRelay": True,
            "iceServers": [],
        }

        controller.start(payload)
        controller.start(payload)

        self.assertEqual(len(_FakeGLib.calls), 1)

    def test_on_ice_candidate_skips_empty_candidate(self) -> None:
        sent_messages: list[tuple[str, dict[str, object], bool]] = []

        def send_message(destination: str, payload: dict[str, object], remember: bool) -> bool:
            sent_messages.append((destination, payload, remember))
            return True

        controller = self.module.WebRTCUplinkController(send_message=send_message)
        controller._session = self.module.WebRTCUplinkSession(
            broadcast_id="device-1",
            session_id="session-1",
            force_relay=False,
            ice_servers=[],
        )

        controller._on_ice_candidate(None, 0, "")
        controller._on_ice_candidate(None, 0, "   ")

        self.assertEqual(sent_messages, [])

    def test_stats_accumulator_emits_interval_loss_rtt_and_feedback_deltas(self) -> None:
        accumulator = self.module.WebRTCStatsAccumulator()
        first = {
            "outbound": {
                "type": "outbound-rtp",
                "timestamp": 1_000_000,
                "packets-sent": 100,
                "bytes-sent": 100_000,
                "nack-count": 2,
                "pli-count": 1,
            },
            "remote": {
                "type": "remote-inbound-rtp",
                "packets-lost": 4,
                "round-trip-time": 0.08,
            },
        }
        second = {
            "outbound": {
                "type": "outbound-rtp",
                "timestamp": 2_000_000,
                "packets-sent": 200,
                "bytes-sent": 225_000,
                "nack-count": 4,
                "pli-count": 2,
            },
            "remote": {
                "type": "remote-inbound-rtp",
                "packets-lost": 9,
                "round-trip-time": 0.12,
            },
        }

        initial = accumulator.observe(first)
        sample = accumulator.observe(second)

        self.assertEqual(initial["outboundRttMs"], 80.0)
        self.assertAlmostEqual(sample["outboundRttMs"], 120.0)
        self.assertAlmostEqual(sample["outboundPacketLossPct"], 100 * 5 / 105)
        self.assertEqual(sample["nackDelta"], 2.0)
        self.assertEqual(sample["pliDelta"], 1.0)
        self.assertEqual(sample["outboundPacketsDelta"], 100.0)
        self.assertEqual(sample["outboundBytesDelta"], 125_000.0)
        self.assertAlmostEqual(sample["sendBitrateKbps"], 1000.0)

    def test_runtime_health_requires_current_connected_session_and_rtp_progress(self) -> None:
        controller = self.module.WebRTCUplinkController(
            send_message=lambda *_args: True
        )
        controller._webrtcbin = object()
        controller.start(
            {"broadcastId": "device-1", "sessionId": "session-1", "iceServers": []}
        )

        class _StateElement:
            def __init__(self, value: str) -> None:
                self.value = value

            def get_property(self, _name: str) -> object:
                return types.SimpleNamespace(value_nick=self.value)

        controller._on_connection_state_changed(_StateElement("connected"), None)
        controller._on_ice_connection_state_changed(_StateElement("completed"), None)
        token = (controller._stats_generation, "session-1")
        for index in range(3):
            controller._on_stats_created(
                _ReplyPromise(
                    {
                        "outbound": {
                            "type": "outbound-rtp",
                            "timestamp": (index + 1) * 1_000_000,
                            "packets-sent": (index + 1) * 100,
                            "bytes-sent": (index + 1) * 100_000,
                        },
                        "remote": {
                            "type": "remote-inbound-rtp",
                            "round-trip-time": 0.05,
                        },
                    }
                ),
                token,
            )

        health = controller.runtime_health_snapshot()
        self.assertTrue(health["hasPipeline"])
        self.assertEqual(health["sessionId"], "session-1")
        self.assertEqual(health["connectionState"], "connected")
        self.assertEqual(health["iceConnectionState"], "completed")
        self.assertEqual(health["outboundProgressSamples"], 2)
        self.assertGreater(health["lastOutboundProgressAt"], 0.0)

        controller.on_signaling_reset()
        reset = controller.runtime_health_snapshot()
        self.assertEqual(reset["connectionState"], "new")
        self.assertEqual(reset["outboundProgressSamples"], 0)

    def test_late_stats_callback_from_replaced_session_is_ignored(self) -> None:
        controller = self.module.WebRTCUplinkController(
            send_message=lambda *_args: True
        )
        controller.start(
            {"broadcastId": "device-1", "sessionId": "session-1", "iceServers": []}
        )
        old_token = (controller._stats_generation, "session-1")
        controller.start(
            {"broadcastId": "device-1", "sessionId": "session-2", "iceServers": []}
        )
        new_token = (controller._stats_generation, "session-2")
        stats = {
            "type": "remote-inbound-rtp",
            "round-trip-time": 0.05,
        }

        controller._on_stats_created(_ReplyPromise(stats), old_token)
        self.assertIsNone(controller.take_latest_outbound_stats())
        controller._on_stats_created(_ReplyPromise(stats), new_token)

        self.assertEqual(
            controller.take_latest_outbound_stats()["outboundRttMs"],
            50.0,
        )

    def test_stop_invalidates_inflight_stats_before_glib_callback(self) -> None:
        controller = self.module.WebRTCUplinkController(
            send_message=lambda *_args: True
        )
        controller.start(
            {"broadcastId": "device-1", "sessionId": "session-1", "iceServers": []}
        )
        token = (controller._stats_generation, "session-1")

        controller.stop(send_signal=False)
        controller._on_stats_created(
            _ReplyPromise(
                {
                    "type": "remote-inbound-rtp",
                    "round-trip-time": 0.05,
                }
            ),
            token,
        )

        self.assertIsNone(controller.take_latest_outbound_stats())


if __name__ == "__main__":
    unittest.main()
