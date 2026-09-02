from __future__ import annotations

import gc
import importlib
import sys
import types
import unittest
import weakref
from typing import Any, ClassVar
from unittest import mock


class _FakeGLib:
    calls: ClassVar[list[tuple[Any, tuple[Any, ...]]]] = []
    timeouts: ClassVar[dict[int, tuple[Any, tuple[Any, ...]]]] = {}
    next_source_id: ClassVar[int] = 100

    @classmethod
    def idle_add(cls, func: Any, *args: Any) -> int:
        cls.calls.append((func, args))
        return len(cls.calls)

    @classmethod
    def drain(cls) -> None:
        while cls.calls:
            func, args = cls.calls.pop(0)
            func(*args)

    @classmethod
    def timeout_add(cls, _milliseconds: int, func: Any, *args: Any) -> int:
        source_id = cls.next_source_id
        cls.next_source_id += 1
        cls.timeouts[source_id] = (func, args)
        return source_id

    @classmethod
    def source_remove(cls, source_id: int) -> bool:
        return cls.timeouts.pop(source_id, None) is not None

    @classmethod
    def fire_timeout(cls, source_id: int) -> None:
        func, args = cls.timeouts.pop(source_id)
        func(*args)


class _ChangePromise:
    def __init__(self, callback: Any, token: object) -> None:
        self.callback = callback
        self.token = token
        self.result = "replied"
        self.reply = None

    def wait(self) -> str:
        return self.result

    def get_reply(self) -> object | None:
        return self.reply

    def complete(self, *, result: str = "replied", reply: object = None) -> None:
        self.result = result
        self.reply = reply
        self.callback(self, self.token)


class _FakePromise:
    @staticmethod
    def new_with_change_func(callback: Any, token: object, _notify: object) -> object:
        return _ChangePromise(callback, token)

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

    @staticmethod
    def wait() -> str:
        return "replied"


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


class _PromiseErrorReply:
    @staticmethod
    def has_field(name: str) -> bool:
        return name == "error"

    @staticmethod
    def get_value(name: str) -> object | None:
        return RuntimeError("rejected") if name == "error" else None


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
        self.auto_complete_descriptions = True
        self.description_promises: list[_ChangePromise] = []
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
        if signal in {"set-local-description", "set-remote-description"}:
            promise = args[-1]
            if isinstance(promise, _ChangePromise):
                self.description_promises.append(promise)
                if self.auto_complete_descriptions:
                    promise.complete()

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
        self.state_changes: list[str] = []

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

    def set_state(self, state: str) -> str:
        self.state_changes.append(state)
        return "success"


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
        PromiseResult=types.SimpleNamespace(REPLIED="replied"),
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
        _FakeGLib.timeouts.clear()
        _FakeGLib.next_source_id = 100
        _FakeElementFactory.created.clear()
        _FakeElementFactory.trace.clear()

    def _controller(
        self,
        send_message: Any = None,
        **kwargs: object,
    ) -> tuple[Any, _FakePipeline]:
        controller = self.module.WebRTCUplinkController(
            send_message=send_message or (lambda *_args: True),
            **kwargs,
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
        token = (controller._stats_generation, "session-1")

        controller._on_offer_created(_ReplyPromise(_OfferReply()), token)

        self.assertEqual(len(sent), 1)
        signaling_token = sent[0][1]
        self.assertTrue(controller.is_signaling_token_current(signaling_token))
        controller.on_signaling_reset()
        self.assertFalse(controller.is_signaling_token_current(signaling_token))

    def test_controller_resolves_only_exact_active_or_terminal_session_token(self) -> None:
        controller, _pipeline = self._controller()
        self._start(controller, "session-1")

        active = controller.signaling_token_for_session("session-1")
        self.assertIsNotNone(active)
        self.assertFalse(active.terminal)
        self.assertIsNone(controller.signaling_token_for_session("session-stale"))
        self.assertIsNone(
            controller.signaling_token_for_session("session-1", terminal=True)
        )

        controller.stop(send_signal=True)
        terminal = controller.signaling_token_for_session(
            "session-1",
            terminal=True,
        )
        self.assertIsNotNone(terminal)
        self.assertTrue(terminal.terminal)
        self.assertIsNone(controller.signaling_token_for_session("session-1"))

    def test_offer_sets_and_sends_the_same_canonical_h264_sdp(self) -> None:
        sent: list[dict[str, object]] = []
        controller, _pipeline = self._controller(
            lambda _destination, payload, _remember, _token: (
                sent.append(payload) or True
            )
        )
        self._start(controller, "session-1")
        branch = controller._branch
        token = (controller._stats_generation, "session-1")
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

    def test_offer_enqueue_failure_disposes_exact_branch_and_sends_stop(self) -> None:
        sent: list[tuple[str, object]] = []

        def send_message(
            destination: str,
            _payload: dict[str, object],
            _remember: bool,
            token: object,
        ) -> bool:
            sent.append((destination, token))
            return destination != self.module.WEBRTC_UPLINK_OFFER_DEST

        controller, pipeline = self._controller(send_message)
        self._start(controller, "session-1")
        branch = controller._branch
        token = (controller._stats_generation, "session-1")

        controller._on_offer_created(_ReplyPromise(_OfferReply()), token)
        _FakeGLib.drain()

        self.assertIsNone(controller._branch)
        self.assertIsNone(branch.queue.get_parent())
        self.assertIsNone(branch.webrtcbin.get_parent())
        self.assertNotIn(branch.queue, pipeline.elements)
        self.assertEqual(
            [destination for destination, _token in sent],
            [
                self.module.WEBRTC_UPLINK_OFFER_DEST,
                self.module.WEBRTC_UPLINK_STOP_DEST,
            ],
        )
        self.assertTrue(sent[-1][1].terminal)

    def test_offer_answer_timeout_disposes_branch_and_sends_terminal_stop(self) -> None:
        sent: list[tuple[str, object]] = []
        controller, _pipeline = self._controller(
            lambda destination, _payload, _remember, token: (
                sent.append((destination, token)) or True
            )
        )
        self._start(controller, "session-1")
        token = (controller._stats_generation, "session-1")
        controller._on_offer_created(_ReplyPromise(_OfferReply()), token)
        _FakeGLib.drain()
        self.assertEqual(len(_FakeGLib.timeouts), 1)

        _FakeGLib.fire_timeout(next(iter(_FakeGLib.timeouts)))
        _FakeGLib.drain()

        self.assertIsNone(controller._branch)
        self.assertEqual(sent[-1][0], self.module.WEBRTC_UPLINK_STOP_DEST)
        self.assertTrue(sent[-1][1].terminal)

    def test_retained_gst_promises_do_not_retain_disposed_branch(self) -> None:
        controller, _pipeline = self._controller()
        self._start(controller, "session-1")
        branch = controller._branch
        queue_ref = weakref.ref(branch.queue)
        webrtc_ref = weakref.ref(branch.webrtcbin)

        create_promise = next(
            args[1]
            for signal, args in branch.webrtcbin.emitted
            if signal == "create-offer"
        )
        create_promise.complete(reply=_OfferReply())
        _FakeGLib.drain()
        self.assertTrue(controller.request_outbound_stats())
        _FakeGLib.drain()
        stats_promise = next(
            args[1]
            for signal, args in branch.webrtcbin.emitted
            if signal == "get-stats"
        )
        retained_promises = [
            create_promise,
            *branch.webrtcbin.description_promises,
            stats_promise,
        ]

        # Model native GStreamer retaining replied/in-flight promises beyond
        # teardown. Their Python user-data must not strongly own webrtcbin.
        for promise in retained_promises:
            self.assertNotIn(branch.webrtcbin, promise.token)

        controller.on_signaling_reset()
        _FakeGLib.drain()
        self.assertIsNone(controller._branch)
        branch = None
        _FakeElementFactory.created.clear()
        gc.collect()

        self.assertIsNone(queue_ref())
        self.assertIsNone(webrtc_ref())
        self.assertGreaterEqual(len(retained_promises), 3)

    def test_valid_answer_requires_connection_before_watchdog_is_cancelled(self) -> None:
        controller, _pipeline = self._controller()
        self._start(controller, "session-1")
        branch = controller._branch
        token = (controller._stats_generation, "session-1")
        controller._on_offer_created(_ReplyPromise(_OfferReply()), token)
        _FakeGLib.drain()
        self.assertEqual(len(_FakeGLib.timeouts), 1)

        controller.apply_answer({"sessionId": "session-1", "sdp": "v=0\r\n"})
        _FakeGLib.drain()

        self.assertEqual(len(_FakeGLib.timeouts), 1)
        self.assertIs(controller._branch, branch)
        self.assertTrue(branch.answer_applied)
        branch.webrtcbin.properties["connection-state"] = types.SimpleNamespace(
            value_nick="connected"
        )
        branch.webrtcbin.properties["ice-connection-state"] = types.SimpleNamespace(
            value_nick="completed"
        )
        controller._on_connection_state_changed(
            branch.webrtcbin,
            None,
            controller._stats_generation,
            "session-1",
        )
        self.assertEqual(len(_FakeGLib.timeouts), 1)
        controller._on_ice_connection_state_changed(
            branch.webrtcbin,
            None,
            controller._stats_generation,
            "session-1",
        )
        self.assertEqual(_FakeGLib.timeouts, {})

    def test_answer_without_connection_is_disposed_by_establishment_timeout(self) -> None:
        sent: list[tuple[str, object]] = []
        controller, _pipeline = self._controller(
            lambda destination, _payload, _remember, token: (
                sent.append((destination, token)) or True
            )
        )
        self._start(controller, "session-1")
        offer_token = (controller._stats_generation, "session-1")
        controller._on_offer_created(_ReplyPromise(_OfferReply()), offer_token)
        _FakeGLib.drain()
        controller.apply_answer({"sessionId": "session-1", "sdp": "v=0\r\n"})
        _FakeGLib.drain()
        self.assertEqual(len(_FakeGLib.timeouts), 1)

        _FakeGLib.fire_timeout(next(iter(_FakeGLib.timeouts)))
        _FakeGLib.drain()

        self.assertIsNone(controller._branch)
        self.assertEqual(sent[-1][0], self.module.WEBRTC_UPLINK_STOP_DEST)
        self.assertTrue(sent[-1][1].terminal)

    def test_local_description_never_reply_times_out_and_late_reply_is_ignored(self) -> None:
        sent: list[str] = []
        controller, _pipeline = self._controller(
            lambda destination, *_args: sent.append(destination) or True
        )
        self._start(controller, "session-1")
        branch = controller._branch
        branch.webrtcbin.auto_complete_descriptions = False
        offer_token = (controller._stats_generation, "session-1")
        controller._on_offer_created(_ReplyPromise(_OfferReply()), offer_token)
        local_promise = branch.webrtcbin.description_promises[-1]
        self.assertNotIn(self.module.WEBRTC_UPLINK_OFFER_DEST, sent)

        _FakeGLib.fire_timeout(next(iter(_FakeGLib.timeouts)))
        _FakeGLib.drain()
        local_promise.complete()
        _FakeGLib.drain()

        self.assertIsNone(controller._branch)
        self.assertNotIn(self.module.WEBRTC_UPLINK_OFFER_DEST, sent)
        self.assertEqual(sent[-1], self.module.WEBRTC_UPLINK_STOP_DEST)

    def test_remote_description_error_fails_exact_session(self) -> None:
        controller, _pipeline = self._controller()
        self._start(controller, "session-1")
        branch = controller._branch
        offer_token = (controller._stats_generation, "session-1")
        controller._on_offer_created(_ReplyPromise(_OfferReply()), offer_token)
        _FakeGLib.drain()
        branch.webrtcbin.auto_complete_descriptions = False
        controller.apply_answer({"sessionId": "session-1", "sdp": "v=0\r\n"})
        _FakeGLib.drain()
        remote_promise = branch.webrtcbin.description_promises[-1]

        remote_promise.complete(reply=_PromiseErrorReply())
        _FakeGLib.drain()

        self.assertIsNone(controller._branch)
        self.assertFalse(branch.answer_applied)

    def test_remote_description_never_reply_keeps_answer_timeout_armed(self) -> None:
        controller, _pipeline = self._controller()
        self._start(controller, "session-1")
        branch = controller._branch
        offer_token = (controller._stats_generation, "session-1")
        controller._on_offer_created(_ReplyPromise(_OfferReply()), offer_token)
        _FakeGLib.drain()
        branch.webrtcbin.auto_complete_descriptions = False
        controller.apply_answer({"sessionId": "session-1", "sdp": "v=0\r\n"})
        _FakeGLib.drain()
        self.assertTrue(branch.answer_pending)
        self.assertEqual(len(_FakeGLib.timeouts), 1)

        _FakeGLib.fire_timeout(next(iter(_FakeGLib.timeouts)))
        _FakeGLib.drain()

        self.assertIsNone(controller._branch)

    def test_late_remote_description_error_cannot_kill_replacement(self) -> None:
        controller, _pipeline = self._controller()
        self._start(controller, "session-1")
        first = controller._branch
        offer_token = (controller._stats_generation, "session-1")
        controller._on_offer_created(_ReplyPromise(_OfferReply()), offer_token)
        _FakeGLib.drain()
        first.webrtcbin.auto_complete_descriptions = False
        controller.apply_answer({"sessionId": "session-1", "sdp": "v=0\r\n"})
        _FakeGLib.drain()
        late_promise = first.webrtcbin.description_promises[-1]
        controller.on_signaling_reset()
        _FakeGLib.drain()
        self._start(controller, "session-2")
        replacement = controller._branch

        late_promise.complete(reply=_PromiseErrorReply())
        _FakeGLib.drain()

        self.assertIs(controller._branch, replacement)
        self.assertEqual(controller._session.session_id, "session-2")

    def test_candidate_enqueue_false_disposes_exact_session(self) -> None:
        sent: list[str] = []

        def send_message(destination: str, *_args: object) -> bool:
            sent.append(destination)
            return destination != self.module.WEBRTC_UPLINK_ICE_CANDIDATE_DEST

        controller, _pipeline = self._controller(send_message)
        self._start(controller, "session-1")
        branch = controller._branch
        controller._on_ice_candidate(
            branch.webrtcbin,
            0,
            "candidate:1 1 UDP 1 127.0.0.1 9 typ host",
            controller._stats_generation,
            "session-1",
        )
        _FakeGLib.drain()

        self.assertIsNone(controller._branch)
        self.assertEqual(sent[-1], self.module.WEBRTC_UPLINK_STOP_DEST)

    def test_candidate_enqueue_exception_still_disposes_branch(self) -> None:
        calls = 0

        def send_message(destination: str, *_args: object) -> bool:
            nonlocal calls
            calls += 1
            if destination == self.module.WEBRTC_UPLINK_ICE_CANDIDATE_DEST:
                raise RuntimeError("queue unavailable")
            return True

        controller, _pipeline = self._controller(send_message)
        self._start(controller, "session-1")
        branch = controller._branch
        controller._on_ice_candidate(
            branch.webrtcbin,
            0,
            "candidate:1 1 UDP 1 127.0.0.1 9 typ host",
            controller._stats_generation,
            "session-1",
        )
        _FakeGLib.drain()

        self.assertIsNone(controller._branch)
        self.assertEqual(calls, 2)

    def test_stale_offer_rejection_cannot_dispose_replacement_session(self) -> None:
        controller, _pipeline = self._controller()
        self._start(controller, "session-1")
        stale = self.module.WebRTCSignalingToken(
            controller._stats_generation,
            "session-1",
        )
        controller.on_signaling_reset()
        _FakeGLib.drain()
        self._start(controller, "session-2")
        replacement = controller._branch

        self.assertFalse(
            controller.reject_signaling(stale, reason="late server rejection")
        )
        _FakeGLib.drain()

        self.assertIs(controller._branch, replacement)
        self.assertEqual(controller._session.session_id, "session-2")

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
        token = (generation, "session-1")
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
        old_token = (controller._stats_generation, "session-1")
        self._start(controller, "session-2")
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
        controller, _pipeline = self._controller()
        self._start(controller, "session-1")
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
        old_offer_token = (first_generation, "session-1")

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

    def test_reset_before_delayed_offer_drops_stale_and_sends_replacement(
        self,
    ) -> None:
        sent: list[tuple[str, str]] = []
        controller, _pipeline = self._controller(
            lambda destination, payload, _remember, _token: (
                sent.append((destination, payload["sessionId"])) or True
            )
        )
        self._start(controller, "session-1")
        first = controller._branch
        delayed_first = next(
            args[1]
            for signal, args in first.webrtcbin.emitted
            if signal == "create-offer"
        )

        controller.on_signaling_reset()
        _FakeGLib.drain()
        self._start(controller, "session-2")
        second = controller._branch
        second_offer = next(
            args[1]
            for signal, args in second.webrtcbin.emitted
            if signal == "create-offer"
        )

        delayed_first.complete(reply=_OfferReply())
        second_offer.complete(reply=_OfferReply())
        _FakeGLib.drain()

        offers = [
            session_id
            for destination, session_id in sent
            if destination == self.module.WEBRTC_UPLINK_OFFER_DEST
        ]
        self.assertEqual(offers, ["session-2"])
        self.assertIs(controller._branch, second)

    def test_incomplete_teardown_refuses_a_replacement_offer(self) -> None:
        fatal_reasons: list[str] = []
        controller, _pipeline = self._controller(
            on_fatal_cleanup=lambda reason: fatal_reasons.append(reason) or True
        )
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

        while _FakeGLib.timeouts:
            _FakeGLib.fire_timeout(next(iter(_FakeGLib.timeouts)))
            _FakeGLib.drain()

        self.assertEqual(len(fatal_reasons), 1)
        self.assertIn("4 attempts", fatal_reasons[0])

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
        self.assertEqual(len(fatal_reasons), 1)

    def test_cleanup_retry_releases_branch_before_fatal_escalation(self) -> None:
        fatal_reasons: list[str] = []
        controller, _pipeline = self._controller(
            on_fatal_cleanup=lambda reason: fatal_reasons.append(reason) or True
        )
        self._start(controller, "session-1")
        branch = controller._branch
        branch.queue.state_result = "failure"

        controller.on_signaling_reset()
        _FakeGLib.drain()
        self.assertTrue(controller._branch_cleanup_failed)
        self.assertEqual(len(_FakeGLib.timeouts), 1)

        branch.queue.state_result = "success"
        _FakeGLib.fire_timeout(next(iter(_FakeGLib.timeouts)))
        _FakeGLib.drain()

        self.assertIsNone(controller._branch)
        self.assertFalse(controller._branch_cleanup_failed)
        self.assertEqual(_FakeGLib.timeouts, {})
        self.assertEqual(fatal_reasons, [])


if __name__ == "__main__":
    unittest.main()
