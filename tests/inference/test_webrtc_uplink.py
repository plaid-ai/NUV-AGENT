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


class _FakeSrc:
    def __init__(self, name: str, path: str) -> None:
        self._name = name
        self._path = path

    def get_name(self) -> str:
        return self._name

    def get_path_string(self) -> str:
        return self._path


class _FakeMessage:
    def __init__(self, src: object) -> None:
        self.src = src


class _FakeEventFactory:
    @staticmethod
    def new_flush_start() -> str:
        return "flush-start"

    @staticmethod
    def new_flush_stop(_reset_time: bool) -> str:
        return "flush-stop"


class _FakeEventTarget:
    def __init__(self) -> None:
        self.events: list[object] = []
        self.properties: dict[str, object] = {}

    def send_event(self, event: object) -> bool:
        self.events.append(event)
        return True

    def set_property(self, name: str, value: object) -> None:
        self.properties[name] = value

    def emit(self, signal_name: str, *_args: object) -> None:
        self.events.append(signal_name)


def _install_fake_gi() -> None:
    gi = types.ModuleType("gi")
    gi.require_version = lambda *_args, **_kwargs: None

    repository = types.ModuleType("gi.repository")
    repository.GLib = _FakeGLib
    repository.Gst = types.SimpleNamespace(
        Pipeline=object,
        Element=object,
        Promise=_FakePromise,
        Event=_FakeEventFactory,
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

    def test_start_ignores_new_session_while_existing_session_is_active(self) -> None:
        controller = self.module.WebRTCUplinkController(send_message=lambda *_args: True)

        controller.start(
            {
                "broadcastId": "device-1",
                "sessionId": "session-1",
                "forceRelay": True,
                "iceServers": [],
            }
        )
        controller.start(
            {
                "broadcastId": "device-1",
                "sessionId": "session-2",
                "forceRelay": True,
                "iceServers": [],
            }
        )

        self.assertEqual(len(_FakeGLib.calls), 1)

    def test_handle_gstreamer_error_recovers_uplink_internal_errors(self) -> None:
        controller = self.module.WebRTCUplinkController(send_message=lambda *_args: True)
        controller.start(
            {
                "broadcastId": "device-1",
                "sessionId": "session-1",
                "forceRelay": True,
                "iceServers": [],
            }
        )

        handled = controller.handle_gstreamer_error(
            _FakeMessage(
                _FakeSrc(
                    "nicesrc0",
                    "/GstPipeline:pipeline0/GstWebRTCBin:webrtc_uplink/TransportReceiveBin:transportreceivebin0/GstNiceSrc:nicesrc0",
                )
            ),
            "internal data stream error",
            "../subprojects/gstreamer/.../GstNiceSrc:nicesrc0",
        )

        self.assertTrue(handled)
        self.assertEqual(len(_FakeGLib.calls), 2)

    def test_handle_gstreamer_error_dedupes_repeated_stop_scheduling(self) -> None:
        controller = self.module.WebRTCUplinkController(send_message=lambda *_args: True)
        controller.start(
            {
                "broadcastId": "device-1",
                "sessionId": "session-1",
                "forceRelay": True,
                "iceServers": [],
            }
        )

        message = _FakeMessage(
            _FakeSrc(
                "nicesrc0",
                "/GstPipeline:pipeline0/GstWebRTCBin:webrtc_uplink/TransportReceiveBin:transportreceivebin0/GstNiceSrc:nicesrc0",
            )
        )
        controller.handle_gstreamer_error(
            message,
            "internal data stream error",
            "../subprojects/gstreamer/.../GstNiceSrc:nicesrc0",
        )
        controller.handle_gstreamer_error(
            message,
            "internal data stream error",
            "../subprojects/gstreamer/.../GstNiceSrc:nicesrc0",
        )

        self.assertEqual(len(_FakeGLib.calls), 2)

    def test_handle_gstreamer_error_does_not_swallow_non_uplink_errors(self) -> None:
        controller = self.module.WebRTCUplinkController(send_message=lambda *_args: True)

        handled = controller.handle_gstreamer_error(
            _FakeMessage(_FakeSrc("zsad_sink", "/GstPipeline:pipeline0/GstAppSink:zsad_sink")),
            "internal data stream error",
            "appsink failure",
        )

        self.assertFalse(handled)
        self.assertEqual(len(_FakeGLib.calls), 0)

    def test_handle_remote_state_stopped_schedules_local_stop(self) -> None:
        controller = self.module.WebRTCUplinkController(send_message=lambda *_args: True)
        controller.start(
            {
                "broadcastId": "device-1",
                "sessionId": "session-1",
                "forceRelay": True,
                "iceServers": [],
            }
        )

        controller.handle_remote_state(
            {
                "broadcastId": "device-1",
                "sessionId": "session-1",
                "state": "STOPPED",
                "reason": "viewer-stop",
            }
        )

        self.assertEqual(len(_FakeGLib.calls), 2)

    def test_stop_notifies_callback_with_reason(self) -> None:
        reasons: list[str] = []
        controller = self.module.WebRTCUplinkController(
            send_message=lambda *_args: True,
            on_session_stopped=reasons.append,
        )
        controller._webrtcbin = _FakeEventTarget()
        controller._webrtc_gate = _FakeEventTarget()
        controller._session = self.module.WebRTCUplinkSession(
            broadcast_id="device-1",
            session_id="session-1",
            force_relay=True,
            ice_servers=[],
        )

        controller.stop(send_signal=False, reason="broadcast-stop")
        stop_call = _FakeGLib.calls[-1]
        stop_call[0](*stop_call[1])

        self.assertEqual(reasons, ["broadcast-stop"])

    def test_stop_gates_webrtc_branch_and_clears_session(self) -> None:
        controller = self.module.WebRTCUplinkController(send_message=lambda *_args: True)
        controller._pipeline = _FakeEventTarget()
        controller._webrtcbin = _FakeEventTarget()
        controller._webrtc_gate = _FakeEventTarget()
        controller._session = self.module.WebRTCUplinkSession(
            broadcast_id="device-1",
            session_id="session-1",
            force_relay=True,
            ice_servers=[],
        )

        controller._stop_on_main_loop()

        self.assertEqual(controller._webrtcbin.events, ["close"])
        self.assertEqual(controller._webrtc_gate.properties, {"drop": True})
        self.assertIsNone(controller._session)


if __name__ == "__main__":
    unittest.main()
