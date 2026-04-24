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


if __name__ == "__main__":
    unittest.main()
