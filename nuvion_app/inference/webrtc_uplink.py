from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstWebRTC", "1.0")
gi.require_version("GstSdp", "1.0")

from gi.repository import GLib, Gst, GstSdp, GstWebRTC

from nuvion_app.inference.webrtc_signaling import (
    WEBRTC_UPLINK_ICE_CANDIDATE,
    WEBRTC_UPLINK_ICE_CANDIDATE_DEST,
    WEBRTC_UPLINK_OFFER,
    WEBRTC_UPLINK_OFFER_DEST,
    WEBRTC_UPLINK_STOP,
    WEBRTC_UPLINK_STOP_DEST,
    build_uplink_payload,
    parse_ice_servers,
    to_gst_ice_server_config,
)

log = logging.getLogger(__name__)


@dataclass
class WebRTCUplinkSession:
    broadcast_id: str
    session_id: str
    force_relay: bool
    ice_servers: list[dict[str, Any]]


class WebRTCUplinkController:
    def __init__(
        self,
        *,
        send_message: Callable[[str, dict[str, Any], bool], bool],
        default_force_relay: bool = True,
    ) -> None:
        self._send_message = send_message
        self._default_force_relay = default_force_relay
        self._pipeline: Gst.Pipeline | None = None
        self._webrtcbin: Gst.Element | None = None
        self._webrtc_gate: Gst.Element | None = None
        self._session: WebRTCUplinkSession | None = None
        self._pending_session: WebRTCUplinkSession | None = None
        self._stop_sent = False
        self._stop_scheduled = False
        self._stop_reason = "unknown"
        self._teardown_error_suppressed = False

    def attach_pipeline(self, pipeline: Gst.Pipeline, element_name: str = "webrtc_uplink") -> bool:
        self._pipeline = pipeline
        self._webrtcbin = pipeline.get_by_name(element_name)
        self._webrtc_gate = pipeline.get_by_name("webrtc_gate")
        if not self._webrtcbin:
            log.warning("[WEBRTC-UPLINK] element '%s' not found.", element_name)
            return False

        self._webrtcbin.connect("on-ice-candidate", self._on_ice_candidate)
        self._webrtcbin.connect("notify::connection-state", self._on_connection_state_changed)
        self._webrtcbin.connect("notify::ice-connection-state", self._on_ice_connection_state_changed)
        return True

    def has_pipeline(self) -> bool:
        return self._webrtcbin is not None

    def handle_gstreamer_error(self, message: object, err: object, dbg: str | None) -> bool:
        if not self._is_uplink_error_message(message, dbg):
            return False

        if self._stop_scheduled or self._pending_session is not None:
            if not self._teardown_error_suppressed:
                log.info(
                    "[WEBRTC-UPLINK] suppressing repeated teardown GStreamer errors (reason=%s): %s, %s",
                    self._stop_reason,
                    err,
                    dbg,
                )
                self._teardown_error_suppressed = True
            return True

        log.warning(
            "[WEBRTC-UPLINK] handled internal GStreamer error without shutting down agent: %s, %s",
            err,
            dbg,
        )
        self.stop(
            send_signal=bool(self._session and not self._stop_sent),
            reason="gstreamer-error",
        )
        return True

    def start(self, payload: dict[str, Any]) -> None:
        session_id = str(payload.get("sessionId") or "").strip()
        broadcast_id = str(payload.get("broadcastId") or "").strip()
        if not session_id or not broadcast_id:
            log.warning("[WEBRTC-UPLINK] start payload missing sessionId or broadcastId: %s", payload)
            return

        ice_servers = parse_ice_servers(payload.get("iceServers"))
        force_relay = bool(payload.get("forceRelay", self._default_force_relay))
        next_session = WebRTCUplinkSession(
            broadcast_id=broadcast_id,
            session_id=session_id,
            force_relay=force_relay,
            ice_servers=ice_servers,
        )

        if self._session:
            if self._session.session_id == session_id:
                log.info("[WEBRTC-UPLINK] ignoring duplicate start for sessionId=%s", session_id)
                return
            log.info(
                "[WEBRTC-UPLINK] replacing active sessionId=%s with newer sessionId=%s",
                self._session.session_id,
                session_id,
            )
            self._pending_session = next_session
            self.stop(send_signal=False, reason="superseded-by-new-start")
            return

        if self._pending_session and self._pending_session.session_id == session_id:
            log.info("[WEBRTC-UPLINK] ignoring duplicate pending start for sessionId=%s", session_id)
            return

        self._activate_session(next_session)

    def apply_answer(self, payload: dict[str, Any]) -> None:
        if not self._session or not self._matches_session(payload):
            return
        sdp = str(payload.get("sdp") or "").strip()
        if not sdp:
            log.warning("[WEBRTC-UPLINK] answer payload missing sdp.")
            return
        GLib.idle_add(self._apply_answer_on_main_loop, sdp)

    def add_remote_ice_candidate(self, payload: dict[str, Any]) -> None:
        if not self._session or not self._matches_session(payload):
            return
        candidate_payload = payload.get("candidate")
        if isinstance(candidate_payload, dict):
            candidate = str(candidate_payload.get("candidate") or "").strip()
            sdp_mid = candidate_payload.get("sdpMid")
            sdp_mline_index = candidate_payload.get("sdpMLineIndex")
        else:
            candidate = str(payload.get("candidate") or "").strip()
            sdp_mid = payload.get("sdpMid")
            sdp_mline_index = payload.get("sdpMLineIndex")

        if not candidate:
            log.warning("[WEBRTC-UPLINK] remote ICE payload missing candidate: %s", payload)
            return

        try:
            mline_index = int(sdp_mline_index) if sdp_mline_index is not None else 0
        except (TypeError, ValueError):
            mline_index = 0
        GLib.idle_add(self._add_remote_candidate_on_main_loop, mline_index, candidate, sdp_mid)

    def handle_remote_state(self, payload: dict[str, Any]) -> None:
        if not self._session or not self._matches_session(payload):
            return
        state = str(payload.get("state") or payload.get("connectionState") or "").strip().lower()
        reason = str(payload.get("reason") or state or "remote-state").strip().lower()
        if state in {"failed", "closed", "stopped"}:
            log.warning(
                "[WEBRTC-UPLINK] remote state=%s reason=%s. stopping local session.",
                state,
                reason,
            )
            self.stop(send_signal=False, reason=reason)

    def stop(self, *, send_signal: bool = True, reason: str = "local-stop") -> None:
        if self._stop_scheduled:
            return
        self._stop_scheduled = True
        self._stop_reason = reason
        self._teardown_error_suppressed = False
        if send_signal and self._session and not self._stop_sent:
            self._send_stop_message()
        GLib.idle_add(self._stop_on_main_loop)

    def on_signaling_reset(self) -> None:
        self._stop_sent = False

    def _activate_session(self, session: WebRTCUplinkSession, *, reset_stop_flags: bool = True) -> None:
        self._session = session
        self._stop_sent = False
        if reset_stop_flags:
            self._stop_scheduled = False
            self._stop_reason = "running"
            self._teardown_error_suppressed = False
        GLib.idle_add(self._start_on_main_loop)

    def _matches_session(self, payload: dict[str, Any]) -> bool:
        session_id = str(payload.get("sessionId") or "").strip()
        return bool(self._session and session_id and session_id == self._session.session_id)

    def _is_uplink_error_message(self, message: object, dbg: str | None) -> bool:
        markers = ("webrtc_uplink", "transportreceivebin", "nicesrc", "gstwebrtcbin")
        haystacks: list[str] = []

        src = getattr(message, "src", None)
        if src is not None:
            get_name = getattr(src, "get_name", None)
            if callable(get_name):
                try:
                    value = get_name()
                    if value:
                        haystacks.append(str(value).lower())
                except Exception:
                    pass

            get_path_string = getattr(src, "get_path_string", None)
            if callable(get_path_string):
                try:
                    value = get_path_string()
                    if value:
                        haystacks.append(str(value).lower())
                except Exception:
                    pass

        if dbg:
            haystacks.append(str(dbg).lower())

        return any(marker in haystack for haystack in haystacks for marker in markers)

    def _start_on_main_loop(self) -> bool:
        if not self._webrtcbin or not self._session:
            return False

        if self._webrtc_gate:
            try:
                self._webrtc_gate.set_property("drop", False)
            except Exception:
                pass

        stun_server, turn_servers = to_gst_ice_server_config(self._session.ice_servers)
        self._webrtcbin.set_property("stun-server", stun_server or "")
        self._webrtcbin.set_property("turn-server", turn_servers[0] if turn_servers else "")
        policy = GstWebRTC.WebRTCICETransportPolicy.RELAY if self._session.force_relay else GstWebRTC.WebRTCICETransportPolicy.ALL
        self._webrtcbin.set_property("ice-transport-policy", policy)
        self._webrtcbin.set_property("bundle-policy", GstWebRTC.WebRTCBundlePolicy.MAX_BUNDLE)

        promise = Gst.Promise.new_with_change_func(self._on_offer_created, None, None)
        self._webrtcbin.emit("create-offer", None, promise)
        log.info(
            "[WEBRTC-UPLINK] creating offer. sessionId=%s relay=%s",
            self._session.session_id,
            self._session.force_relay,
        )
        return False

    def _stop_on_main_loop(self) -> bool:
        if self._webrtc_gate:
            try:
                self._webrtc_gate.set_property("drop", True)
            except Exception:
                pass
        if self._webrtcbin:
            try:
                # Prefer the native webrtcbin close action. Force-flushing the branch
                # and stop the upstream RTP flow with a valve before closing to reduce
                # noisy "data flow before segment event" warnings during teardown.
                self._webrtcbin.emit("close", Gst.Promise.new())
            except Exception:
                pass
        self._session = None
        if self._pending_session is not None:
            pending_session = self._pending_session
            self._pending_session = None
            self._session = pending_session
            GLib.idle_add(self._start_pending_session_on_main_loop)
        return False

    def _start_pending_session_on_main_loop(self) -> bool:
        if not self._session:
            return False
        self._stop_scheduled = False
        self._stop_reason = "running"
        self._teardown_error_suppressed = False
        return self._start_on_main_loop()

    def _apply_answer_on_main_loop(self, sdp_text: str) -> bool:
        if not self._webrtcbin:
            return False

        description = _build_session_description(GstWebRTC.WebRTCSDPType.ANSWER, sdp_text)
        if description is None:
            log.error("[WEBRTC-UPLINK] failed to parse SDP answer.")
            return False

        self._webrtcbin.emit("set-remote-description", description, Gst.Promise.new())
        log.info("[WEBRTC-UPLINK] applied SDP answer.")
        return False

    def _add_remote_candidate_on_main_loop(
        self,
        mline_index: int,
        candidate: str,
        _sdp_mid: str | None,
    ) -> bool:
        if not self._webrtcbin:
            return False
        self._webrtcbin.emit("add-ice-candidate", mline_index, candidate)
        log.debug("[WEBRTC-UPLINK] added remote ICE candidate mline=%s", mline_index)
        return False

    def _on_offer_created(self, promise: Gst.Promise, *_args: object) -> None:
        if not self._webrtcbin or not self._session:
            return

        reply = promise.get_reply()
        if reply is None:
            log.error("[WEBRTC-UPLINK] offer promise returned no reply.")
            return

        offer = reply.get_value("offer")
        if offer is None:
            log.error("[WEBRTC-UPLINK] offer promise missing offer value.")
            return

        self._webrtcbin.emit("set-local-description", offer, Gst.Promise.new())
        sdp_text = offer.sdp.as_text()
        payload = build_uplink_payload(
            WEBRTC_UPLINK_OFFER,
            self._session.broadcast_id,
            self._session.session_id,
            sdp=sdp_text,
        )
        self._send_message(WEBRTC_UPLINK_OFFER_DEST, payload, True)
        log.info("[WEBRTC-UPLINK] sent SDP offer. sessionId=%s", self._session.session_id)

    def _on_ice_candidate(self, _element: Gst.Element, mline_index: int, candidate: str) -> None:
        if not self._session:
            return
        payload = build_uplink_payload(
            WEBRTC_UPLINK_ICE_CANDIDATE,
            self._session.broadcast_id,
            self._session.session_id,
            candidate=candidate,
            sdpMLineIndex=int(mline_index),
            sdpMid="video",
        )
        self._send_message(WEBRTC_UPLINK_ICE_CANDIDATE_DEST, payload, False)

    def _on_connection_state_changed(self, element: Gst.Element, _pspec: object) -> None:
        state = element.get_property("connection-state")
        state_nick = getattr(state, "value_nick", str(state))
        log.info("[WEBRTC-UPLINK] connection-state=%s", state_nick)
        if state_nick in {"failed", "closed"}:
            self.stop(send_signal=not self._stop_sent, reason=f"connection-state-{state_nick}")

    def _on_ice_connection_state_changed(self, element: Gst.Element, _pspec: object) -> None:
        state = element.get_property("ice-connection-state")
        state_nick = getattr(state, "value_nick", str(state))
        log.info("[WEBRTC-UPLINK] ice-connection-state=%s", state_nick)
        if state_nick in {"failed", "closed", "disconnected"} and self._session:
            self.stop(send_signal=not self._stop_sent, reason=f"ice-state-{state_nick}")

    def _send_stop_message(self) -> None:
        if not self._session:
            return
        payload = build_uplink_payload(
            WEBRTC_UPLINK_STOP,
            self._session.broadcast_id,
            self._session.session_id,
        )
        self._stop_sent = self._send_message(WEBRTC_UPLINK_STOP_DEST, payload, False)


def _build_session_description(
    sdp_type: GstWebRTC.WebRTCSDPType,
    sdp_text: str,
) -> GstWebRTC.WebRTCSessionDescription | None:
    result, sdp_message = GstSdp.SDPMessage.new()
    if result != GstSdp.SDPResult.OK:
        return None
    parse_result = GstSdp.sdp_message_parse_buffer(bytes(sdp_text.encode("utf-8")), sdp_message)
    if parse_result != GstSdp.SDPResult.OK:
        return None
    return GstWebRTC.WebRTCSessionDescription.new(sdp_type, sdp_message)


def describe_payload(payload: dict[str, Any]) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False)
    except Exception:
        return str(payload)
