from __future__ import annotations

import json
import logging
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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


def _structure_to_mapping(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _structure_to_mapping(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_structure_to_mapping(item) for item in value]
    n_fields = getattr(value, "n_fields", None)
    nth_field_name = getattr(value, "nth_field_name", None)
    get_value = getattr(value, "get_value", None)
    if callable(n_fields) and callable(nth_field_name) and callable(get_value):
        mapped: dict[str, Any] = {}
        for index in range(int(n_fields())):
            name = str(nth_field_name(index))
            mapped[name] = _structure_to_mapping(get_value(name))
        return mapped
    return value


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _flatten_stats(value: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if isinstance(value, dict):
        result.append(value)
        for nested in value.values():
            result.extend(_flatten_stats(nested))
    elif isinstance(value, list):
        for nested in value:
            result.extend(_flatten_stats(nested))
    return result


class WebRTCStatsAccumulator:
    """Normalize cumulative webrtcbin stats into one interval feedback sample."""

    def __init__(self) -> None:
        self._previous: dict[str, float] | None = None

    def reset(self) -> None:
        self._previous = None

    def observe(self, raw_stats: Any) -> dict[str, Any] | None:
        mapped = _structure_to_mapping(raw_stats)
        structures = _flatten_stats(mapped)
        outbound: dict[str, Any] = {}
        remote: dict[str, Any] = {}
        for item in structures:
            stats_type = str(item.get("type") or item.get("statsType") or "").lower()
            if "remote-inbound-rtp" in stats_type:
                remote.update(item)
            elif "outbound-rtp" in stats_type:
                outbound.update(item)
        if not outbound and isinstance(mapped, dict):
            outbound = mapped

        def number(*names: str, source: dict[str, Any] | None = None) -> float | None:
            values = outbound if source is None else source
            for name in names:
                candidate = _finite_number(values.get(name))
                if candidate is not None:
                    return candidate
            return None

        current = {
            "timestamp": number("timestamp", "timestampUs", "timestamp-us"),
            "packetsSent": number("packetsSent", "packets-sent"),
            "packetsLost": number(
                "packetsLost", "packets-lost", source=remote or outbound
            ),
            "bytesSent": number("bytesSent", "bytes-sent"),
            "nackCount": number("nackCount", "nack-count"),
            "pliCount": number("pliCount", "pli-count"),
        }
        rtt = number(
            "roundTripTime",
            "round-trip-time",
            source=remote or outbound,
        )
        fraction_lost = number(
            "fractionLost",
            "fraction-lost",
            source=remote or outbound,
        )
        queue_pressure = number("queuePressurePct", "queue-pressure-pct")

        previous = self._previous
        self._previous = {
            key: value for key, value in current.items() if value is not None
        }
        sample: dict[str, Any] = {"source": "WEBRTC_OUTBOUND"}
        if rtt is not None:
            sample["outboundRttMs"] = rtt * 1000.0 if rtt < 10.0 else rtt
        if queue_pressure is not None:
            sample["queuePressurePct"] = queue_pressure
        if previous is not None:
            sent_delta = max(
                0.0,
                (current["packetsSent"] or 0.0) - previous.get("packetsSent", 0.0),
            )
            lost_delta = max(
                0.0,
                (current["packetsLost"] or 0.0) - previous.get("packetsLost", 0.0),
            )
            total_delta = sent_delta + lost_delta
            if total_delta > 0:
                sample["outboundPacketLossPct"] = 100.0 * lost_delta / total_delta
            elif fraction_lost is not None:
                sample["outboundPacketLossPct"] = (
                    fraction_lost * 100.0 if fraction_lost <= 1.0 else fraction_lost
                )
            for source_key, output_key in (
                ("nackCount", "nackDelta"),
                ("pliCount", "pliDelta"),
            ):
                value = current[source_key]
                if value is not None:
                    sample[output_key] = max(
                        0.0,
                        value - previous.get(source_key, value),
                    )
            timestamp = current["timestamp"]
            bytes_sent = current["bytesSent"]
            if timestamp is not None and bytes_sent is not None:
                elapsed = timestamp - previous.get("timestamp", timestamp)
                byte_delta = max(0.0, bytes_sent - previous.get("bytesSent", bytes_sent))
                # WebRTC/GStreamer timestamps are normally microseconds; accept
                # millisecond test fixtures as well.
                elapsed_seconds = elapsed / (1_000_000.0 if elapsed > 10_000 else 1000.0)
                if elapsed_seconds > 0:
                    sample["sendBitrateKbps"] = byte_delta * 8.0 / elapsed_seconds / 1000.0
        elif fraction_lost is not None:
            sample["outboundPacketLossPct"] = (
                fraction_lost * 100.0 if fraction_lost <= 1.0 else fraction_lost
            )

        return sample if len(sample) > 1 else None


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
        default_force_relay: bool = False,
    ) -> None:
        self._send_message = send_message
        self._default_force_relay = default_force_relay
        self._pipeline: Gst.Pipeline | None = None
        self._webrtcbin: Gst.Element | None = None
        self._session: WebRTCUplinkSession | None = None
        self._stop_sent = False
        self._stats_accumulator = WebRTCStatsAccumulator()
        self._stats_lock = threading.Lock()
        self._latest_outbound_stats: dict[str, Any] | None = None
        self._stats_generation = 0

    def attach_pipeline(self, pipeline: Gst.Pipeline, element_name: str = "webrtc_uplink") -> bool:
        self._pipeline = pipeline
        self._webrtcbin = pipeline.get_by_name(element_name)
        if not self._webrtcbin:
            log.warning("[WEBRTC-UPLINK] element '%s' not found.", element_name)
            return False

        self._webrtcbin.connect("on-ice-candidate", self._on_ice_candidate)
        self._webrtcbin.connect("notify::connection-state", self._on_connection_state_changed)
        self._webrtcbin.connect("notify::ice-connection-state", self._on_ice_connection_state_changed)
        return True

    def has_pipeline(self) -> bool:
        return self._webrtcbin is not None

    def start(self, payload: dict[str, Any]) -> None:
        session_id = str(payload.get("sessionId") or "").strip()
        broadcast_id = str(payload.get("broadcastId") or "").strip()
        if not session_id or not broadcast_id:
            log.warning("[WEBRTC-UPLINK] start payload missing sessionId or broadcastId: %s", payload)
            return

        if self._session and self._session.session_id == session_id:
            log.info("[WEBRTC-UPLINK] ignoring duplicate start for sessionId=%s", session_id)
            return

        ice_servers = parse_ice_servers(payload.get("iceServers"))
        force_relay = bool(payload.get("forceRelay", self._default_force_relay))
        self._session = WebRTCUplinkSession(
            broadcast_id=broadcast_id,
            session_id=session_id,
            force_relay=force_relay,
            ice_servers=ice_servers,
        )
        self._stop_sent = False
        with self._stats_lock:
            self._stats_generation += 1
            self._stats_accumulator.reset()
            self._latest_outbound_stats = None
        GLib.idle_add(self._start_on_main_loop)

    def request_outbound_stats(self) -> bool:
        if not self._webrtcbin or not self._session:
            return False
        with self._stats_lock:
            generation = self._stats_generation
            session_id = self._session.session_id
        return bool(
            GLib.idle_add(
                self._request_stats_on_main_loop,
                generation,
                session_id,
            )
        )

    def take_latest_outbound_stats(self) -> dict[str, Any] | None:
        with self._stats_lock:
            sample = self._latest_outbound_stats
            self._latest_outbound_stats = None
        return dict(sample) if sample is not None else None

    def _request_stats_on_main_loop(
        self,
        generation: int,
        session_id: str,
    ) -> bool:
        with self._stats_lock:
            current = self._session
            if (
                generation != self._stats_generation
                or current is None
                or current.session_id != session_id
            ):
                return False
        if not self._webrtcbin:
            return False
        try:
            promise = Gst.Promise.new_with_change_func(
                self._on_stats_created,
                (generation, session_id),
                None,
            )
            self._webrtcbin.emit("get-stats", None, promise)
        except Exception as exc:  # noqa: BLE001 - unsupported stats must not stop RTC.
            log.debug("[WEBRTC-UPLINK] get-stats unavailable: %s", exc)
        return False

    def _on_stats_created(
        self,
        promise: Gst.Promise,
        token: object = None,
        *_args: object,
    ) -> None:
        try:
            reply = promise.get_reply()
            if reply is None:
                return
            with self._stats_lock:
                if (
                    not isinstance(token, tuple)
                    or len(token) != 2
                    or token[0] != self._stats_generation
                    or self._session is None
                    or token[1] != self._session.session_id
                ):
                    return
                sample = self._stats_accumulator.observe(reply)
                if sample is None:
                    return
                self._latest_outbound_stats = sample
        except Exception as exc:  # noqa: BLE001 - malformed stats are one missed sample.
            log.debug("[WEBRTC-UPLINK] failed to normalize outbound stats: %s", exc)

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
        if state in {"failed", "closed", "stopped"}:
            log.warning("[WEBRTC-UPLINK] remote state=%s. stopping local session.", state)
            self.stop(send_signal=False)

    def stop(self, *, send_signal: bool = True) -> None:
        if send_signal and self._session and not self._stop_sent:
            self._send_stop_message()
        with self._stats_lock:
            session_id = self._session.session_id if self._session else None
            self._stats_generation += 1
            generation = self._stats_generation
            self._stats_accumulator.reset()
            self._latest_outbound_stats = None
        GLib.idle_add(self._stop_on_main_loop, generation, session_id)

    def on_signaling_reset(self) -> None:
        self._stop_sent = False

    def _matches_session(self, payload: dict[str, Any]) -> bool:
        session_id = str(payload.get("sessionId") or "").strip()
        return bool(self._session and session_id and session_id == self._session.session_id)

    def _start_on_main_loop(self) -> bool:
        if not self._webrtcbin or not self._session:
            return False

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

    def _stop_on_main_loop(
        self,
        generation: int,
        session_id: str | None,
    ) -> bool:
        with self._stats_lock:
            if generation != self._stats_generation:
                return False
            if (
                session_id is not None
                and self._session is not None
                and self._session.session_id != session_id
            ):
                return False
            self._session = None
            self._stats_accumulator.reset()
            self._latest_outbound_stats = None
        if self._pipeline:
            try:
                self._pipeline.send_event(Gst.Event.new_flush_start())
                self._pipeline.send_event(Gst.Event.new_flush_stop(False))
            except Exception as exc:  # noqa: BLE001 - teardown is best effort.
                log.debug("[WEBRTC-UPLINK] pipeline flush failed: %s", exc)
        return False

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
        candidate_text = str(candidate or "").strip()
        if not candidate_text:
            log.debug(
                "[WEBRTC-UPLINK] skip empty local ICE candidate. sessionId=%s mline=%s",
                self._session.session_id,
                mline_index,
            )
            return
        payload = build_uplink_payload(
            WEBRTC_UPLINK_ICE_CANDIDATE,
            self._session.broadcast_id,
            self._session.session_id,
            candidate=candidate_text,
            sdpMLineIndex=int(mline_index),
            sdpMid="video",
        )
        self._send_message(WEBRTC_UPLINK_ICE_CANDIDATE_DEST, payload, False)

    def _on_connection_state_changed(self, element: Gst.Element, _pspec: object) -> None:
        state = element.get_property("connection-state")
        state_nick = getattr(state, "value_nick", str(state))
        log.info("[WEBRTC-UPLINK] connection-state=%s", state_nick)
        if state_nick in {"failed", "closed"}:
            self.stop(send_signal=not self._stop_sent)

    def _on_ice_connection_state_changed(self, element: Gst.Element, _pspec: object) -> None:
        state = element.get_property("ice-connection-state")
        state_nick = getattr(state, "value_nick", str(state))
        log.info("[WEBRTC-UPLINK] ice-connection-state=%s", state_nick)
        if state_nick in {"failed", "closed", "disconnected"} and self._session:
            self.stop(send_signal=not self._stop_sent)

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
    except Exception:  # noqa: BLE001 - diagnostic fallback accepts any object.
        return str(payload)
