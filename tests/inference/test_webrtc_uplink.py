from __future__ import annotations

import importlib
import sys
import types
import unittest
from typing import Any, ClassVar
from unittest import mock


class _FakeGLib:
    calls: ClassVar[list[tuple[Any, tuple[Any, ...]]]] = []

    @classmethod
    def idle_add(cls, func: Any, *args: Any) -> int:
        cls.calls.append((func, args))
        return len(cls.calls)

    @classmethod
    def drain(cls) -> None:
        while cls.calls:
            func, args = cls.calls.pop(0)
            func(*args)


class _FakePromise:
    @staticmethod
    def new_with_change_func(callback: Any, token: object, _notify: object) -> object:
        return types.SimpleNamespace(callback=callback, token=token)

    @staticmethod
    def new() -> object:
        return object()


class _FakeSessionDescription:
    @staticmethod
    def new(*_args: object, **_kwargs: object) -> object:
        return object()


class _ReplyPromise:
    def __init__(self, reply: Any) -> None:
        self.reply = reply

    def get_reply(self) -> Any:
        return self.reply


class _Offer:
    sdp = types.SimpleNamespace(
        as_text=lambda: (
            "v=0\r\n"
            "m=video 9 UDP/TLS/RTP/SAVPF 96\r\n"
            "a=rtpmap:96 H264/90000\r\n"
            "a=fmtp:96 packetization-mode=1;profile-level-id=42c01e\r\n"
        )
    )


class _OfferReply:
    def get_value(self, name: str) -> object | None:
        return _Offer() if name == "offer" else None


class _FakePad:
    def __init__(self, owner: _FakeElement, name: str) -> None:
        self.owner = owner
        self.name = name
        self.peer: _FakePad | None = None
        self.released = False

    def link(self, other: _FakePad) -> str:
        if self.peer is not None or other.peer is not None:
            return "was-linked"
        self.peer = other
        other.peer = self
        _FakeElementFactory.trace.append(
            ("link", self.owner.name, self.name, other.owner.name, other.name)
        )
        return "ok"

    def unlink(self, other: _FakePad) -> bool:
        if self.peer is not other or other.peer is not self:
            return False
        self.peer = None
        other.peer = None
        _FakeElementFactory.trace.append(
            ("unlink", self.owner.name, self.name, other.owner.name, other.name)
        )
        return True

    def get_peer(self) -> _FakePad | None:
        return self.peer


class _FakeElement:
    def __init__(self, factory: str, name: str) -> None:
        self.factory = factory
        self.name = name
        self.parent: _FakePipeline | None = None
        self.properties: dict[str, object] = {}
        self.emitted: list[tuple[str, tuple[object, ...]]] = []
        self.released_pads: list[_FakePad] = []
        self.state_changes: list[str] = []
        self.state_result = "success"
        self.sync_result = True
        self.handlers: dict[int, tuple[str, Any, tuple[object, ...]]] = {}
        self._next_handler = 1
        self._request_pad_count = 0
        self._static_pads: dict[str, _FakePad] = {}
        if factory == "queue":
            self._static_pads = {
                "sink": _FakePad(self, "sink"),
                "src": _FakePad(self, "src"),
            }

    def request_pad_simple(self, template_name: str) -> _FakePad:
        self._request_pad_count += 1
        return _FakePad(
            self,
            template_name.replace("%u", str(self._request_pad_count - 1)),
        )

    def release_request_pad(self, pad: _FakePad) -> None:
        pad.released = True
        self.released_pads.append(pad)
        _FakeElementFactory.trace.append(("release", self.name, pad.name))

    def get_static_pad(self, name: str) -> _FakePad | None:
        return self._static_pads.get(name)

    def set_property(self, name: str, value: object) -> None:
        self.properties[name] = value

    def get_property(self, name: str) -> object:
        return self.properties.get(name, types.SimpleNamespace(value_nick="new"))

    def connect(self, signal: str, callback: Any, *user_data: object) -> int:
        handler_id = self._next_handler
        self._next_handler += 1
        self.handlers[handler_id] = (signal, callback, user_data)
        return handler_id

    def disconnect(self, handler_id: int) -> None:
        del self.handlers[handler_id]

    def sync_state_with_parent(self) -> bool:
        _FakeElementFactory.trace.append(("sync", self.name))
        return self.sync_result

    def set_state(self, state: str) -> str:
        self.state_changes.append(state)
        _FakeElementFactory.trace.append(("state", self.name, state))
        return self.state_result

    def emit(self, signal: str, *args: object) -> None:
        self.emitted.append((signal, args))

    def get_parent(self) -> _FakePipeline | None:
        return self.parent


class _FakeElementFactory:
    created: ClassVar[list[_FakeElement]] = []
    trace: ClassVar[list[tuple[object, ...]]] = []

    @classmethod
    def make(cls, factory: str, name: str) -> _FakeElement:
        element = _FakeElement(factory, name)
        cls.created.append(element)
        cls.trace.append(("make", factory, name))
        return element


class _FakePipeline:
    def __init__(self) -> None:
        self.tee = _FakeElement("tee", "webrtc_uplink_tee")
        self.tee.parent = self
        self.elements: list[_FakeElement] = [self.tee]

    def get_by_name(self, name: str) -> _FakeElement | None:
        return next((item for item in self.elements if item.name == name), None)

    def add(self, element: _FakeElement) -> bool:
        if element.parent is not None:
            return False
        element.parent = self
        self.elements.append(element)
        _FakeElementFactory.trace.append(("add", element.name))
        return True

    def remove(self, element: _FakeElement) -> bool:
        if element.parent is not self:
            return False
        self.elements.remove(element)
        element.parent = None
        _FakeElementFactory.trace.append(("remove", element.name))
        return True


def _install_fake_gi() -> None:
    gi = types.ModuleType("gi")
    gi.require_version = lambda *_args, **_kwargs: None

    repository = types.ModuleType("gi.repository")
    repository.GLib = _FakeGLib
    repository.Gst = types.SimpleNamespace(
        Pipeline=object,
        Element=object,
        Pad=object,
        Promise=_FakePromise,
        ElementFactory=_FakeElementFactory,
        PadLinkReturn=types.SimpleNamespace(OK="ok"),
        State=types.SimpleNamespace(NULL="null"),
        StateChangeReturn=types.SimpleNamespace(FAILURE="failure"),
    )
    repository.GstSdp = types.SimpleNamespace(
        SDPMessage=types.SimpleNamespace(new=lambda: (0, object())),
        SDPResult=types.SimpleNamespace(OK=0),
        sdp_message_parse_buffer=lambda *_args, **_kwargs: 0,
    )
    repository.GstWebRTC = types.SimpleNamespace(
        WebRTCICETransportPolicy=types.SimpleNamespace(RELAY="relay", ALL="all"),
        WebRTCBundlePolicy=types.SimpleNamespace(MAX_BUNDLE="max-bundle"),
        WebRTCSDPType=types.SimpleNamespace(ANSWER="answer", OFFER="offer"),
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
        _FakeElementFactory.created.clear()
        _FakeElementFactory.trace.clear()

    def _controller(
        self,
        send_message: Any = None,
    ) -> tuple[Any, _FakePipeline]:
        controller = self.module.WebRTCUplinkController(
            send_message=send_message or (lambda *_args: True)
        )
        pipeline = _FakePipeline()
        self.assertTrue(controller.attach_pipeline(pipeline))
        return controller, pipeline

    @staticmethod
    def _start(controller: Any, session_id: str) -> None:
        controller.start(
            {
                "broadcastId": "device-1",
                "sessionId": session_id,
                "forceRelay": False,
                "iceServers": [],
            }
        )
        _FakeGLib.drain()

    def test_start_ignores_duplicate_session(self) -> None:
        controller, _pipeline = self._controller()
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
        sent_messages: list[tuple[str, dict[str, object], bool, object]] = []

        def send_message(
            destination: str,
            payload: dict[str, object],
            remember: bool,
            signaling_token: object,
        ) -> bool:
            sent_messages.append((destination, payload, remember, signaling_token))
            return True

        controller, _pipeline = self._controller(send_message)
        self._start(controller, "session-1")
        branch = controller._branch
        generation = controller._stats_generation

        controller._on_ice_candidate(
            branch.webrtcbin,
            0,
            "",
            generation,
            "session-1",
        )
        controller._on_ice_candidate(
            branch.webrtcbin,
            0,
            "   ",
            generation,
            "session-1",
        )

        self.assertEqual(sent_messages, [])

    def test_offer_transport_token_is_invalidated_by_signaling_reset(self) -> None:
        sent: list[tuple[str, object]] = []
        controller, _pipeline = self._controller(
            lambda destination, _payload, _remember, token: (
                sent.append((destination, token)) or True
            )
        )
        self._start(controller, "session-1")
        branch = controller._branch
        token = (controller._stats_generation, "session-1", branch.webrtcbin)

        controller._on_offer_created(_ReplyPromise(_OfferReply()), token)

        self.assertEqual(len(sent), 1)
        signaling_token = sent[0][1]
        self.assertTrue(controller.is_signaling_token_current(signaling_token))
        controller.on_signaling_reset()
        self.assertFalse(controller.is_signaling_token_current(signaling_token))

    def test_offer_sets_and_sends_the_same_canonical_h264_sdp(self) -> None:
        sent: list[dict[str, object]] = []
        controller, _pipeline = self._controller(
            lambda _destination, payload, _remember, _token: (
                sent.append(payload) or True
            )
        )
        self._start(controller, "session-1")
        branch = controller._branch
        token = (controller._stats_generation, "session-1", branch.webrtcbin)
        local_offer = object()

        with mock.patch.object(
            self.module,
            "_build_session_description",
            return_value=local_offer,
        ) as build_description:
            controller._on_offer_created(_ReplyPromise(_OfferReply()), token)

        self.assertEqual(len(sent), 1)
        canonical_sdp = sent[0]["sdp"]
        self.assertIn("profile-level-id=42e01f", canonical_sdp)
        self.assertNotIn("profile-level-id=42c01e", canonical_sdp)
        build_description.assert_called_once_with("offer", canonical_sdp)
        local_descriptions = [
            args
            for signal, args in branch.webrtcbin.emitted
            if signal == "set-local-description"
        ]
        self.assertEqual(len(local_descriptions), 1)
        self.assertIs(local_descriptions[0][0], local_offer)

    def test_repeated_stop_keeps_the_queued_terminal_token_current(self) -> None:
        sent: list[object] = []
        controller, _pipeline = self._controller(
            lambda _destination, _payload, _remember, token: (
                sent.append(token) or True
            )
        )
        self._start(controller, "session-1")

        controller.stop()
        self.assertEqual(len(sent), 1)
        stop_token = sent[0]
        generation = controller._stats_generation
        self.assertTrue(controller.is_signaling_token_current(stop_token))

        controller.stop()

        self.assertEqual(controller._stats_generation, generation)
        self.assertTrue(controller.is_signaling_token_current(stop_token))

    def test_stats_accumulator_emits_interval_loss_rtt_and_feedback_deltas(
        self,
    ) -> None:
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

    def test_runtime_health_requires_current_connected_session_and_rtp_progress(
        self,
    ) -> None:
        controller, _pipeline = self._controller()
        self._start(controller, "session-1")
        branch = controller._branch
        generation = controller._stats_generation
        branch.webrtcbin.properties["connection-state"] = types.SimpleNamespace(
            value_nick="connected"
        )
        branch.webrtcbin.properties["ice-connection-state"] = types.SimpleNamespace(
            value_nick="completed"
        )

        controller._on_connection_state_changed(
            branch.webrtcbin,
            None,
            generation,
            "session-1",
        )
        controller._on_ice_connection_state_changed(
            branch.webrtcbin,
            None,
            generation,
            "session-1",
        )
        token = (generation, "session-1", branch.webrtcbin)
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
        self.assertFalse(reset["hasPipeline"])
        self.assertIsNone(reset["sessionId"])
        self.assertEqual(reset["connectionState"], "new")
        self.assertEqual(reset["outboundProgressSamples"], 0)

    def test_late_stats_callback_from_replaced_session_is_ignored(self) -> None:
        controller, _pipeline = self._controller()
        self._start(controller, "session-1")
        old_branch = controller._branch
        old_token = (
            controller._stats_generation,
            "session-1",
            old_branch.webrtcbin,
        )
        self._start(controller, "session-2")
        new_branch = controller._branch
        new_token = (
            controller._stats_generation,
            "session-2",
            new_branch.webrtcbin,
        )
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
        controller, _pipeline = self._controller()
        self._start(controller, "session-1")
        branch = controller._branch
        token = (controller._stats_generation, "session-1", branch.webrtcbin)

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

    def test_reconnect_disposes_and_recreates_webrtc_branch(self) -> None:
        controller, pipeline = self._controller()
        self._start(controller, "session-1")
        first = controller._branch
        first_tee_pad = first.tee_src_pad
        first_webrtc_pad = first.webrtc_sink_pad

        controller.on_signaling_reset()
        _FakeGLib.drain()

        self.assertIsNone(controller._branch)
        self.assertIn("null", first.queue.state_changes)
        self.assertIn("null", first.webrtcbin.state_changes)
        self.assertTrue(first_tee_pad.released)
        self.assertTrue(first_webrtc_pad.released)
        self.assertNotIn(first.queue, pipeline.elements)
        self.assertNotIn(first.webrtcbin, pipeline.elements)

        _FakeElementFactory.trace.clear()
        self._start(controller, "session-2")
        second = controller._branch

        self.assertIsNot(second.webrtcbin, first.webrtcbin)
        self.assertIsNot(second.tee_src_pad, first.tee_src_pad)
        self.assertIs(second.tee_src_pad.get_peer(), second.queue_sink_pad)
        self.assertEqual(
            len(
                [
                    item
                    for item in _FakeElementFactory.created
                    if item.factory == "webrtcbin"
                ]
            ),
            2,
        )
        sync_webrtc = _FakeElementFactory.trace.index(("sync", second.webrtcbin.name))
        sync_queue = _FakeElementFactory.trace.index(("sync", second.queue.name))
        link_tee = _FakeElementFactory.trace.index(
            (
                "link",
                pipeline.tee.name,
                second.tee_src_pad.name,
                second.queue.name,
                "sink",
            )
        )
        self.assertLess(sync_webrtc, link_tee)
        self.assertLess(sync_queue, link_tee)

    def test_stale_generation_cannot_mutate_new_sdp_or_ice_state(self) -> None:
        sent: list[tuple[str, dict[str, object], bool]] = []
        controller, _pipeline = self._controller(
            lambda destination, payload, remember, _token: (
                sent.append((destination, payload, remember)) or True
            )
        )
        self._start(controller, "session-1")
        first = controller._branch
        first_generation = controller._stats_generation
        old_offer_token = (first_generation, "session-1", first.webrtcbin)

        controller.on_signaling_reset()
        _FakeGLib.drain()
        self._start(controller, "session-2")
        second = controller._branch
        sent.clear()

        controller._apply_answer_on_main_loop(
            first_generation,
            "session-1",
            "v=0\r\n",
        )
        controller._add_remote_candidate_on_main_loop(
            first_generation,
            "session-1",
            0,
            "candidate:old",
            "video",
        )
        controller._on_ice_candidate(
            first.webrtcbin,
            0,
            "candidate:late-local",
            first_generation,
            "session-1",
        )
        controller._on_offer_created(_ReplyPromise(_OfferReply()), old_offer_token)

        self.assertEqual(sent, [])
        self.assertFalse(
            any(
                signal in {"set-remote-description", "add-ice-candidate"}
                for signal, _args in second.webrtcbin.emitted
            )
        )

    def test_incomplete_teardown_refuses_a_replacement_offer(self) -> None:
        controller, _pipeline = self._controller()
        self._start(controller, "session-1")
        first = controller._branch
        first.queue.state_result = "failure"

        controller.on_signaling_reset()
        _FakeGLib.drain()
        self.assertTrue(controller._branch_cleanup_failed)
        released_once = (
            len(controller._uplink_tee.released_pads),
            len(first.webrtcbin.released_pads),
        )
        self.assertEqual(released_once, (1, 1))
        self.assertEqual(first.signal_handler_ids, [])

        create_count = len(
            [
                item
                for item in _FakeElementFactory.created
                if item.factory == "webrtcbin"
            ]
        )
        self._start(controller, "session-2")

        self.assertEqual(
            len(
                [
                    item
                    for item in _FakeElementFactory.created
                    if item.factory == "webrtcbin"
                ]
            ),
            create_count,
        )
        self.assertIsNone(controller._session)
        self.assertFalse(controller.runtime_health_snapshot()["hasPipeline"])

        controller.on_signaling_reset()
        _FakeGLib.drain()
        self.assertEqual(
            (
                len(controller._uplink_tee.released_pads),
                len(first.webrtcbin.released_pads),
            ),
            released_once,
        )
        self.assertEqual(first.signal_handler_ids, [])


if __name__ == "__main__":
    unittest.main()
