from __future__ import annotations

import ctypes
import ctypes.util
import json
import logging
import math
import threading
import time
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
    enforce_h264_offer_parameters,
    normalize_h264_profile_level_id,
    parse_ice_servers,
    to_gst_ice_server_config,
)

log = logging.getLogger(__name__)

_BRANCH_CLEANUP_MAX_ATTEMPTS = 4
_BRANCH_CLEANUP_RETRY_INTERVAL_MS = 100


def _disable_libnice_upnp(webrtcbin: Gst.Element) -> bool | None:
    """Disable libnice UPnP without creating unsafe transfer-none GI wrappers.

    Qualcomm's Ubuntu 24.04 GStreamer/libnice build continuously grows native
    heap while GUPnP discovery is active on a multi-interface edge host.  The
    control plane already supplies explicit STUN/TURN servers, so UPnP is not
    required for the managed uplink.

    PyGObject 3.48 incorrectly releases the transfer-none ``ice-agent``/``agent``
    wrappers on this image and can invalidate webrtcbin.  Use the GObject C API,
    which gives us explicit owned references that are released before returning.
    """

    if getattr(webrtcbin, "__gtype__", None) is None:
        # Unit-test doubles and non-GObject implementations cannot expose the
        # native NiceAgent.  Their behavior is covered at the controller seam.
        return None

    ice_agent = ctypes.c_void_p()
    nice_agent = ctypes.c_void_p()
    gobject: Any | None = None
    try:
        library_candidates = (
            None,
            ctypes.util.find_library("gobject-2.0"),
            "libgobject-2.0.so.0",
            "libgobject-2.0.dylib",
        )
        load_errors: list[str] = []
        for library_name in dict.fromkeys(library_candidates):
            if library_name is None and load_errors:
                continue
            try:
                candidate = ctypes.CDLL(library_name)
                candidate.g_object_get
                candidate.g_object_set
                candidate.g_object_unref
                gobject = candidate
                break
            except (AttributeError, OSError) as exc:
                load_errors.append(f"{library_name or 'process'}: {exc}")
        if gobject is None:
            raise OSError("; ".join(load_errors))
        gobject.g_object_get.restype = None
        gobject.g_object_set.restype = None
        gobject.g_object_unref.argtypes = (ctypes.c_void_p,)
        gobject.g_object_unref.restype = None

        gobject.g_object_get(
            ctypes.c_void_p(
                hash(webrtcbin) & ((1 << (ctypes.sizeof(ctypes.c_void_p) * 8)) - 1)
            ),
            ctypes.c_char_p(b"ice-agent"),
            ctypes.byref(ice_agent),
            ctypes.c_void_p(),
        )
        if not ice_agent.value:
            return False
        gobject.g_object_get(
            ice_agent,
            ctypes.c_char_p(b"agent"),
            ctypes.byref(nice_agent),
            ctypes.c_void_p(),
        )
        if not nice_agent.value:
            return False
        gobject.g_object_set(
            nice_agent,
            ctypes.c_char_p(b"upnp"),
            ctypes.c_int(0),
            ctypes.c_void_p(),
        )
        enabled = ctypes.c_int(1)
        gobject.g_object_get(
            nice_agent,
            ctypes.c_char_p(b"upnp"),
            ctypes.byref(enabled),
            ctypes.c_void_p(),
        )
        return enabled.value == 0
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        log.warning("[WEBRTC-UPLINK] could not disable libnice UPnP: %s", exc)
        return False
    finally:
        if gobject is not None and nice_agent.value:
            gobject.g_object_unref(nice_agent)
        if gobject is not None and ice_agent.value:
            gobject.g_object_unref(ice_agent)


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
            if current["packetsSent"] is not None:
                sample["outboundPacketsDelta"] = sent_delta
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
            byte_delta: float | None = None
            if bytes_sent is not None:
                byte_delta = max(
                    0.0,
                    bytes_sent - previous.get("bytesSent", bytes_sent),
                )
                sample["outboundBytesDelta"] = byte_delta
            if timestamp is not None and bytes_sent is not None:
                elapsed = timestamp - previous.get("timestamp", timestamp)
                # WebRTC/GStreamer timestamps are normally microseconds; accept
                # millisecond test fixtures as well.
                elapsed_seconds = elapsed / (
                    1_000_000.0 if elapsed > 10_000 else 1000.0
                )
                if elapsed_seconds > 0 and byte_delta is not None:
                    sample["sendBitrateKbps"] = (
                        byte_delta * 8.0 / elapsed_seconds / 1000.0
                    )
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


@dataclass(frozen=True)
class WebRTCSignalingToken:
    """Ownership token checked again by the transport immediately before send."""

    generation: int
    session_id: str
    terminal: bool = False


@dataclass
class _WebRTCBranch:
    """One disposable WebRTC media branch owned by exactly one generation."""

    generation: int
    session_id: str
    queue: Gst.Element
    webrtcbin: Gst.Element
    tee_src_pad: Gst.Pad
    queue_sink_pad: Gst.Pad
    queue_src_pad: Gst.Pad
    webrtc_sink_pad: Gst.Pad
    signal_handler_ids: list[int]
    offer_enqueued: bool = False
    answer_pending: bool = False
    answer_applied: bool = False
    tee_unlinked: bool = False
    tee_pad_released: bool = False
    webrtc_stopped: bool = False
    queue_stopped: bool = False
    webrtc_unlinked: bool = False
    webrtc_pad_released: bool = False
    queue_removed: bool = False
    webrtc_removed: bool = False


def _request_pad(element: Gst.Element, template_name: str) -> Gst.Pad | None:
    request_pad_simple = getattr(element, "request_pad_simple", None)
    if callable(request_pad_simple):
        return request_pad_simple(template_name)
    # Ubuntu 24.04 has request_pad_simple. Keep the fallback for older local
    # developer GStreamer builds without making it the production path.
    get_request_pad = getattr(element, "get_request_pad", None)
    return get_request_pad(template_name) if callable(get_request_pad) else None


def _unlink_exact(source_pad: Gst.Pad, expected_peer: Gst.Pad) -> bool:
    peer = source_pad.get_peer()
    if peer is None:
        return True
    if peer != expected_peer:
        return False
    return bool(source_pad.unlink(peer))


class WebRTCUplinkController:
    def __init__(
        self,
        *,
        send_message: Callable[
            [str, dict[str, Any], bool, WebRTCSignalingToken], bool
        ],
        default_force_relay: bool = False,
        h264_profile_level_id: str = "42e01f",
        h264_packetization_mode: str = "1",
        h264_level_asymmetry_allowed: str = "1",
        offer_answer_timeout_sec: float = 15.0,
        connection_timeout_sec: float = 20.0,
        on_fatal_cleanup: Callable[[str], bool] | None = None,
        enable_upnp: bool = False,
    ) -> None:
        self._send_message = send_message
        self._default_force_relay = default_force_relay
        self._h264_profile_level_id = normalize_h264_profile_level_id(
            h264_profile_level_id
        )
        self._h264_packetization_mode = str(h264_packetization_mode or "").strip()
        self._h264_level_asymmetry_allowed = str(
            h264_level_asymmetry_allowed or ""
        ).strip()
        if self._h264_packetization_mode not in {"0", "1"}:
            raise ValueError("H264 packetization-mode must be 0 or 1")
        if self._h264_level_asymmetry_allowed not in {"0", "1"}:
            raise ValueError("H264 level-asymmetry-allowed must be 0 or 1")
        self._offer_answer_timeout_sec = float(offer_answer_timeout_sec)
        if (
            not math.isfinite(self._offer_answer_timeout_sec)
            or self._offer_answer_timeout_sec < 1.0
            or self._offer_answer_timeout_sec > 120.0
        ):
            raise ValueError("WebRTC offer answer timeout must be in [1, 120] seconds")
        self._connection_timeout_sec = float(connection_timeout_sec)
        if (
            not math.isfinite(self._connection_timeout_sec)
            or self._connection_timeout_sec < 1.0
            or self._connection_timeout_sec > 120.0
        ):
            raise ValueError("WebRTC connection timeout must be in [1, 120] seconds")
        self._on_fatal_cleanup = on_fatal_cleanup
        self._enable_upnp = bool(enable_upnp)
        self._pipeline: Gst.Pipeline | None = None
        self._uplink_tee: Gst.Element | None = None
        self._branch: _WebRTCBranch | None = None
        self._branch_cleanup_failed = False
        self._session: WebRTCUplinkSession | None = None
        self._terminal_session_id: str | None = None
        self._stop_sent = False
        self._stats_accumulator = WebRTCStatsAccumulator()
        self._stats_lock = threading.RLock()
        self._latest_outbound_stats: dict[str, Any] | None = None
        self._stats_generation = 0
        self._connection_state = "new"
        self._ice_connection_state = "new"
        self._connected_since: float | None = None
        self._ice_connected_since: float | None = None
        self._outbound_progress_samples = 0
        self._first_outbound_progress_at: float | None = None
        self._last_outbound_progress_at: float | None = None
        self._offer_watchdog_source_id: int | None = None
        self._offer_watchdog_generation: int | None = None
        self._offer_watchdog_session_id: str | None = None
        self._connection_watchdog_source_id: int | None = None
        self._connection_watchdog_generation: int | None = None
        self._connection_watchdog_session_id: str | None = None
        self._cleanup_retry_source_id: int | None = None
        self._cleanup_retry_generation: int | None = None
        self._cleanup_attempts = 0
        self._cleanup_fatal_reported = False

    def attach_pipeline(
        self,
        pipeline: Gst.Pipeline,
        element_name: str = "webrtc_uplink_tee",
    ) -> bool:
        self._pipeline = pipeline
        self._uplink_tee = pipeline.get_by_name(element_name)
        if not self._uplink_tee:
            log.warning("[WEBRTC-UPLINK] element '%s' not found.", element_name)
            return False
        return True

    def has_pipeline(self) -> bool:
        with self._stats_lock:
            return (
                self._pipeline is not None
                and self._uplink_tee is not None
                and self._session is not None
                and self._branch is not None
                and self._branch.generation == self._stats_generation
                and not self._branch_cleanup_failed
            )

    def runtime_health_snapshot(self) -> dict[str, Any]:
        """Return current-session live RTP evidence without exposing SDP or ICE secrets."""

        with self._stats_lock:
            session_id = self._session.session_id if self._session else None
            return {
                "hasPipeline": (
                    self._pipeline is not None
                    and self._uplink_tee is not None
                    and self._session is not None
                    and self._branch is not None
                    and self._branch.generation == self._stats_generation
                    and not self._branch_cleanup_failed
                ),
                "sessionId": session_id,
                "generation": self._stats_generation,
                "connectionState": self._connection_state,
                "iceConnectionState": self._ice_connection_state,
                "connectedSince": self._connected_since,
                "iceConnectedSince": self._ice_connected_since,
                "outboundProgressSamples": self._outbound_progress_samples,
                "firstOutboundProgressAt": self._first_outbound_progress_at,
                "lastOutboundProgressAt": self._last_outbound_progress_at,
            }

    def _reset_outbound_progress_locked(self) -> None:
        self._outbound_progress_samples = 0
        self._first_outbound_progress_at = None
        self._last_outbound_progress_at = None

    def _reset_runtime_health_locked(self) -> None:
        self._connection_state = "new"
        self._ice_connection_state = "new"
        self._connected_since = None
        self._ice_connected_since = None
        self._reset_outbound_progress_locked()

    def start(self, payload: dict[str, Any]) -> None:
        session_id = str(payload.get("sessionId") or "").strip()
        broadcast_id = str(payload.get("broadcastId") or "").strip()
        if not session_id or not broadcast_id:
            log.warning(
                "[WEBRTC-UPLINK] start payload missing sessionId or broadcastId: %s",
                payload,
            )
            return

        ice_servers = parse_ice_servers(payload.get("iceServers"))
        force_relay = bool(payload.get("forceRelay", self._default_force_relay))
        session = WebRTCUplinkSession(
            broadcast_id=broadcast_id,
            session_id=session_id,
            force_relay=force_relay,
            ice_servers=ice_servers,
        )
        with self._stats_lock:
            if self._session and self._session.session_id == session_id:
                log.info(
                    "[WEBRTC-UPLINK] ignoring duplicate start for sessionId=%s",
                    session_id,
                )
                return
            watchdog_source_id = self._clear_offer_watchdog_locked()
            connection_source_id = self._clear_connection_watchdog_locked()
            self._stats_generation += 1
            generation = self._stats_generation
            self._session = session
            self._terminal_session_id = None
            self._stop_sent = False
            self._stats_accumulator.reset()
            self._latest_outbound_stats = None
            self._reset_runtime_health_locked()
        self._remove_glib_source(watchdog_source_id)
        self._remove_glib_source(connection_source_id)
        GLib.idle_add(self._start_on_main_loop, generation, session_id)

    def request_outbound_stats(self) -> bool:
        with self._stats_lock:
            if self._session is None or self._branch is None:
                return False
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
            branch = self._branch
            if (
                generation != self._stats_generation
                or current is None
                or current.session_id != session_id
                or branch is None
                or branch.generation != generation
                or self._branch_cleanup_failed
            ):
                return False
        try:
            promise = Gst.Promise.new_with_change_func(
                self._on_stats_created,
                (generation, session_id),
                None,
            )
            branch.webrtcbin.emit("get-stats", None, promise)
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
                    or self._branch is None
                    or self._branch_cleanup_failed
                ):
                    return
                sample = self._stats_accumulator.observe(reply)
                if sample is None:
                    return
                self._latest_outbound_stats = sample
                packet_delta = _finite_number(sample.get("outboundPacketsDelta")) or 0.0
                byte_delta = _finite_number(sample.get("outboundBytesDelta")) or 0.0
                if packet_delta > 0.0 or byte_delta > 0.0:
                    observed_at = time.monotonic()
                    self._outbound_progress_samples = min(
                        self._outbound_progress_samples + 1,
                        1_000_000,
                    )
                    if self._first_outbound_progress_at is None:
                        self._first_outbound_progress_at = observed_at
                    self._last_outbound_progress_at = observed_at
        except Exception as exc:  # noqa: BLE001 - malformed stats are one missed sample.
            log.debug("[WEBRTC-UPLINK] failed to normalize outbound stats: %s", exc)

    def apply_answer(self, payload: dict[str, Any]) -> None:
        sdp = str(payload.get("sdp") or "").strip()
        if not sdp:
            log.warning("[WEBRTC-UPLINK] answer payload missing sdp.")
            return
        with self._stats_lock:
            if not self._matches_session_locked(payload):
                return
            generation = self._stats_generation
            session_id = self._session.session_id
        GLib.idle_add(
            self._apply_answer_on_main_loop,
            generation,
            session_id,
            sdp,
        )

    def add_remote_ice_candidate(self, payload: dict[str, Any]) -> None:
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
            log.warning(
                "[WEBRTC-UPLINK] remote ICE payload missing candidate: %s", payload
            )
            return

        try:
            mline_index = int(sdp_mline_index) if sdp_mline_index is not None else 0
        except (TypeError, ValueError):
            mline_index = 0
        with self._stats_lock:
            if not self._matches_session_locked(payload):
                return
            generation = self._stats_generation
            session_id = self._session.session_id
        GLib.idle_add(
            self._add_remote_candidate_on_main_loop,
            generation,
            session_id,
            mline_index,
            candidate,
            sdp_mid,
        )

    def handle_remote_state(self, payload: dict[str, Any]) -> None:
        with self._stats_lock:
            if not self._matches_session_locked(payload):
                return
        state = (
            str(payload.get("state") or payload.get("connectionState") or "")
            .strip()
            .lower()
        )
        if state in {"failed", "closed", "stopped"}:
            log.warning(
                "[WEBRTC-UPLINK] remote state=%s. stopping local session.", state
            )
            self.stop(send_signal=False)

    def stop(self, *, send_signal: bool = True) -> None:
        with self._stats_lock:
            watchdog_source_id = self._clear_offer_watchdog_locked()
            connection_source_id = self._clear_connection_watchdog_locked()
            session = self._session
            branch_generation = self._branch.generation if self._branch else None
            if session is None:
                self._stats_accumulator.reset()
                self._latest_outbound_stats = None
                self._reset_runtime_health_locked()
                already_stopped = True
                should_send = False
                stop_generation = self._stats_generation
            else:
                already_stopped = False
                should_send = bool(send_signal and not self._stop_sent)
                self._stats_generation += 1
                stop_generation = self._stats_generation
                self._session = None
                self._terminal_session_id = (
                    session.session_id if should_send else None
                )
                self._stats_accumulator.reset()
                self._latest_outbound_stats = None
                self._reset_runtime_health_locked()
        self._remove_glib_source(watchdog_source_id)
        self._remove_glib_source(connection_source_id)
        if already_stopped:
            if branch_generation is not None:
                GLib.idle_add(self._stop_on_main_loop, branch_generation)
            return
        if branch_generation is not None:
            GLib.idle_add(self._stop_on_main_loop, branch_generation)
        if should_send and session is not None:
            try:
                sent = self._send_stop_message_for(
                    session,
                    WebRTCSignalingToken(
                        stop_generation,
                        session.session_id,
                        terminal=True,
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - teardown must still run.
                sent = False
                log.error("[WEBRTC-UPLINK] stop enqueue raised: %s", exc)
            with self._stats_lock:
                if self._stats_generation == stop_generation and self._session is None:
                    self._stop_sent = sent

    def on_signaling_reset(self) -> None:
        # A WebSocket reconnect is a hard session boundary. Reusing a
        # webrtcbin after remote signaling disappears leaves ICE/transceivers
        # and GStreamer sticky-event state behind, which can make the next RTP
        # session consume buffers before receiving a SEGMENT event.
        with self._stats_lock:
            watchdog_source_id = self._clear_offer_watchdog_locked()
            connection_source_id = self._clear_connection_watchdog_locked()
            branch_generation = self._branch.generation if self._branch else None
            self._stats_generation += 1
            self._session = None
            self._terminal_session_id = None
            self._stop_sent = False
            self._stats_accumulator.reset()
            self._latest_outbound_stats = None
            self._reset_runtime_health_locked()
        self._remove_glib_source(watchdog_source_id)
        self._remove_glib_source(connection_source_id)
        if branch_generation is not None:
            GLib.idle_add(self._stop_on_main_loop, branch_generation)

    def reject_signaling(
        self,
        token: WebRTCSignalingToken,
        *,
        reason: str,
    ) -> bool:
        """Fail only the session that owns a rejected outbound signaling frame."""

        if not isinstance(token, WebRTCSignalingToken) or token.terminal:
            return False
        return self._fail_current_signaling_session(
            token.generation,
            token.session_id,
            reason=reason,
        )

    def is_signaling_token_current(self, token: WebRTCSignalingToken) -> bool:
        if not isinstance(token, WebRTCSignalingToken):
            return False
        with self._stats_lock:
            if token.generation != self._stats_generation:
                return False
            if token.terminal:
                return bool(
                    self._session is None
                    and self._terminal_session_id == token.session_id
                )
            branch = self._branch
            return bool(
                self._session is not None
                and self._session.session_id == token.session_id
                and branch is not None
                and branch.generation == token.generation
                and branch.session_id == token.session_id
                and not self._branch_cleanup_failed
            )

    def signaling_token_for_session(
        self,
        session_id: str,
        *,
        terminal: bool = False,
    ) -> WebRTCSignalingToken | None:
        """Resolve an error envelope to the currently owned signaling generation."""

        normalized = str(session_id or "").strip()
        if not normalized:
            return None
        with self._stats_lock:
            if terminal:
                if self._session is None and self._terminal_session_id == normalized:
                    return WebRTCSignalingToken(
                        self._stats_generation,
                        normalized,
                        terminal=True,
                    )
                return None
            branch = self._branch
            if (
                self._session is not None
                and self._session.session_id == normalized
                and branch is not None
                and branch.generation == self._stats_generation
                and branch.session_id == normalized
                and not self._branch_cleanup_failed
            ):
                return WebRTCSignalingToken(self._stats_generation, normalized)
            return None

    def _matches_session_locked(self, payload: dict[str, Any]) -> bool:
        session_id = str(payload.get("sessionId") or "").strip()
        return bool(
            self._session and session_id and session_id == self._session.session_id
        )

    def _clear_offer_watchdog_locked(self) -> int | None:
        source_id = self._offer_watchdog_source_id
        self._offer_watchdog_source_id = None
        self._offer_watchdog_generation = None
        self._offer_watchdog_session_id = None
        return source_id

    def _clear_connection_watchdog_locked(self) -> int | None:
        source_id = self._connection_watchdog_source_id
        self._connection_watchdog_source_id = None
        self._connection_watchdog_generation = None
        self._connection_watchdog_session_id = None
        return source_id

    def _clear_cleanup_retry_locked(self) -> int | None:
        source_id = self._cleanup_retry_source_id
        self._cleanup_retry_source_id = None
        self._cleanup_retry_generation = None
        return source_id

    @staticmethod
    def _promise_failure_reason(promise: Gst.Promise) -> str | None:
        try:
            result = promise.wait()
        except Exception as exc:  # noqa: BLE001 - a broken promise is a failure.
            return f"promise wait failed: {exc}"
        if result != Gst.PromiseResult.REPLIED:
            name = str(getattr(result, "value_nick", result))
            return f"promise completed without reply: {name}"
        try:
            reply = promise.get_reply()
        except Exception as exc:  # noqa: BLE001 - a broken reply is a failure.
            return f"promise reply failed: {exc}"
        if reply is None:
            return None
        has_field = getattr(reply, "has_field", None)
        get_value = getattr(reply, "get_value", None)
        if callable(has_field) and callable(get_value) and has_field("error"):
            error = get_value("error")
            if error is not None:
                return f"promise error: {error}"
        return None

    @staticmethod
    def _remove_glib_source(source_id: int | None) -> None:
        if source_id is None:
            return
        try:
            GLib.source_remove(source_id)
        except Exception as exc:  # noqa: BLE001 - stale source IDs are harmless.
            log.debug("[WEBRTC-UPLINK] stale GLib source removal ignored: %s", exc)

    def _arm_offer_watchdog_on_main_loop(
        self,
        generation: int,
        session_id: str,
    ) -> bool:
        with self._stats_lock:
            branch = self._active_branch_for_token(generation, session_id)
            if branch is None or branch.answer_applied:
                return False
            previous_source_id = self._clear_offer_watchdog_locked()
        self._remove_glib_source(previous_source_id)
        source_id = GLib.timeout_add(
            max(1, int(self._offer_answer_timeout_sec * 1000)),
            self._on_offer_answer_timeout,
            generation,
            session_id,
        )
        with self._stats_lock:
            branch = self._active_branch_for_token(generation, session_id)
            if branch is None or branch.answer_applied:
                stale_source_id = source_id
            else:
                self._offer_watchdog_source_id = source_id
                self._offer_watchdog_generation = generation
                self._offer_watchdog_session_id = session_id
                stale_source_id = None
        self._remove_glib_source(stale_source_id)
        return False

    def _on_offer_answer_timeout(self, generation: int, session_id: str) -> bool:
        with self._stats_lock:
            if (
                self._offer_watchdog_generation != generation
                or self._offer_watchdog_session_id != session_id
            ):
                return False
            self._clear_offer_watchdog_locked()
        self._fail_current_signaling_session(
            generation,
            session_id,
            reason="WebRTC offer/answer signaling timeout",
        )
        return False

    def _arm_connection_watchdog_on_main_loop(
        self,
        generation: int,
        session_id: str,
    ) -> bool:
        with self._stats_lock:
            branch = self._active_branch_for_token(generation, session_id)
            if branch is None or not branch.answer_applied:
                return False
            if self._connection_state == "connected" and self._ice_connection_state in {
                "connected",
                "completed",
            }:
                return False
            previous_source_id = self._clear_connection_watchdog_locked()
        self._remove_glib_source(previous_source_id)
        source_id = GLib.timeout_add(
            max(1, int(self._connection_timeout_sec * 1000)),
            self._on_connection_timeout,
            generation,
            session_id,
        )
        with self._stats_lock:
            branch = self._active_branch_for_token(generation, session_id)
            if (
                branch is None
                or not branch.answer_applied
                or (
                    self._connection_state == "connected"
                    and self._ice_connection_state in {"connected", "completed"}
                )
            ):
                stale_source_id = source_id
            else:
                self._connection_watchdog_source_id = source_id
                self._connection_watchdog_generation = generation
                self._connection_watchdog_session_id = session_id
                stale_source_id = None
        self._remove_glib_source(stale_source_id)
        return False

    def _on_connection_timeout(self, generation: int, session_id: str) -> bool:
        with self._stats_lock:
            if (
                self._connection_watchdog_generation != generation
                or self._connection_watchdog_session_id != session_id
            ):
                return False
            self._clear_connection_watchdog_locked()
        self._fail_current_signaling_session(
            generation,
            session_id,
            reason="WebRTC connection establishment timeout",
        )
        return False

    def _fail_current_signaling_session(
        self,
        generation: int,
        session_id: str,
        *,
        reason: str,
    ) -> bool:
        with self._stats_lock:
            session = self._session
            branch = self._branch
            if (
                generation != self._stats_generation
                or session is None
                or session.session_id != session_id
            ):
                return False
            watchdog_source_id = self._clear_offer_watchdog_locked()
            connection_source_id = self._clear_connection_watchdog_locked()
            branch_generation = branch.generation if branch is not None else None
            self._stats_generation += 1
            stop_generation = self._stats_generation
            self._session = None
            self._terminal_session_id = session.session_id
            self._stop_sent = False
            self._stats_accumulator.reset()
            self._latest_outbound_stats = None
            self._reset_runtime_health_locked()
            self._connection_state = "failed"
            self._ice_connection_state = "failed"
            stop_token = WebRTCSignalingToken(
                stop_generation,
                session.session_id,
                terminal=True,
            )
        self._remove_glib_source(watchdog_source_id)
        self._remove_glib_source(connection_source_id)
        log.error(
            "[WEBRTC-UPLINK] terminating signaling sessionId=%s generation=%s: %s",
            session_id,
            generation,
            str(reason)[:500],
        )
        if branch_generation is not None:
            GLib.idle_add(self._stop_on_main_loop, branch_generation)
        try:
            sent = self._send_stop_message_for(session, stop_token)
        except Exception as exc:  # noqa: BLE001 - teardown must outlive transport failure.
            sent = False
            log.error("[WEBRTC-UPLINK] terminal stop enqueue raised: %s", exc)
        with self._stats_lock:
            if self._stats_generation == stop_generation and self._session is None:
                self._stop_sent = sent
        return True

    def _start_on_main_loop(self, generation: int, session_id: str) -> bool:
        with self._stats_lock:
            session = self._session
            if (
                generation != self._stats_generation
                or session is None
                or session.session_id != session_id
            ):
                return False
            if self._branch_cleanup_failed:
                log.error(
                    "[WEBRTC-UPLINK] refusing generation=%s after incomplete branch cleanup",
                    generation,
                )
                self._abort_current_session_on_main_loop(generation, session_id)
                return False

        # Every offer gets a physically new queue, request pads and webrtcbin.
        # The permanent tee is allow-not-linked, so removing the previous branch
        # cannot turn an expected signaling reset into a pipeline-wide NOT_LINKED
        # failure. Linking the fresh tee pad last causes GStreamer to replay its
        # sticky STREAM_START/CAPS/SEGMENT events before the first RTP buffer.
        if not self._teardown_branch_on_main_loop():
            log.error(
                "[WEBRTC-UPLINK] branch teardown failed; refusing generation=%s",
                generation,
            )
            self._abort_current_session_on_main_loop(generation, session_id)
            return False

        with self._stats_lock:
            if (
                generation != self._stats_generation
                or self._session is None
                or self._session.session_id != session_id
            ):
                return False
            session = self._session

        branch = self._create_branch_on_main_loop(generation, session)
        if branch is None:
            self._abort_current_session_on_main_loop(generation, session_id)
            return False

        with self._stats_lock:
            if (
                generation != self._stats_generation
                or self._session is None
                or self._session.session_id != session_id
                or self._branch is not branch
            ):
                self._teardown_branch_on_main_loop(generation)
                return False

        # Promise user-data must never own the element that owns the promise.
        # Some GStreamer/WebRTC code paths retain a replied promise until the
        # element is finalized. Putting ``webrtcbin`` in the token therefore
        # creates a native/Python reference cycle that survives parent removal
        # and keeps its internal media graph alive after a rejected offer.
        token = (generation, session_id)
        self._arm_offer_watchdog_on_main_loop(generation, session_id)
        promise = Gst.Promise.new_with_change_func(
            self._on_offer_created,
            token,
            None,
        )
        branch.webrtcbin.emit("create-offer", None, promise)
        log.info(
            "[WEBRTC-UPLINK] creating fresh offer. sessionId=%s generation=%s relay=%s",
            session_id,
            generation,
            session.force_relay,
        )
        return False

    def _create_branch_on_main_loop(
        self,
        generation: int,
        session: WebRTCUplinkSession,
    ) -> _WebRTCBranch | None:
        pipeline = self._pipeline
        uplink_tee = self._uplink_tee
        if pipeline is None or uplink_tee is None:
            log.error("[WEBRTC-UPLINK] RTP tee is not attached")
            return None

        queue_element = Gst.ElementFactory.make(
            "queue",
            f"webrtc_uplink_queue_{generation}",
        )
        webrtcbin = Gst.ElementFactory.make(
            "webrtcbin",
            f"webrtc_uplink_session_{generation}",
        )
        if queue_element is None or webrtcbin is None:
            log.error("[WEBRTC-UPLINK] failed to create disposable branch elements")
            return None

        queue_sink_pad = queue_element.get_static_pad("sink")
        queue_src_pad = queue_element.get_static_pad("src")
        tee_src_pad = _request_pad(uplink_tee, "src_%u")
        webrtc_sink_pad = _request_pad(webrtcbin, "sink_%u")
        if (
            queue_sink_pad is None
            or queue_src_pad is None
            or tee_src_pad is None
            or webrtc_sink_pad is None
        ):
            if tee_src_pad is not None:
                uplink_tee.release_request_pad(tee_src_pad)
            if webrtc_sink_pad is not None:
                webrtcbin.release_request_pad(webrtc_sink_pad)
            log.error("[WEBRTC-UPLINK] failed to allocate disposable branch pads")
            return None

        branch = _WebRTCBranch(
            generation=generation,
            session_id=session.session_id,
            queue=queue_element,
            webrtcbin=webrtcbin,
            tee_src_pad=tee_src_pad,
            queue_sink_pad=queue_sink_pad,
            queue_src_pad=queue_src_pad,
            webrtc_sink_pad=webrtc_sink_pad,
            signal_handler_ids=[],
        )
        with self._stats_lock:
            self._branch = branch
            self._branch_cleanup_failed = False

        try:
            pipeline.add(queue_element)
            if queue_element.get_parent() is not pipeline:
                raise RuntimeError("failed to add queue to pipeline")
            pipeline.add(webrtcbin)
            if webrtcbin.get_parent() is not pipeline:
                raise RuntimeError("failed to add webrtcbin to pipeline")

            if not self._enable_upnp:
                upnp_disabled = _disable_libnice_upnp(webrtcbin)
                if upnp_disabled:
                    log.info("[WEBRTC-UPLINK] libnice UPnP discovery disabled")
                elif upnp_disabled is False:
                    log.warning(
                        "[WEBRTC-UPLINK] libnice UPnP disable was unavailable; "
                        "cgroup memory limits remain the fallback guard"
                    )

            stun_server, turn_servers = to_gst_ice_server_config(session.ice_servers)
            webrtcbin.set_property("stun-server", stun_server or "")
            webrtcbin.set_property(
                "turn-server",
                turn_servers[0] if turn_servers else "",
            )
            policy = (
                GstWebRTC.WebRTCICETransportPolicy.RELAY
                if session.force_relay
                else GstWebRTC.WebRTCICETransportPolicy.ALL
            )
            webrtcbin.set_property("ice-transport-policy", policy)
            webrtcbin.set_property(
                "bundle-policy",
                GstWebRTC.WebRTCBundlePolicy.MAX_BUNDLE,
            )
            webrtcbin.set_property("latency", 0)
            queue_element.set_property("max-size-buffers", 60)
            queue_element.set_property("max-size-bytes", 0)
            queue_element.set_property("max-size-time", 0)
            queue_element.set_property("leaky", 2)

            if queue_src_pad.link(webrtc_sink_pad) != Gst.PadLinkReturn.OK:
                raise RuntimeError("failed to link queue to webrtcbin")

            for signal_name, callback in (
                ("on-ice-candidate", self._on_ice_candidate),
                ("notify::connection-state", self._on_connection_state_changed),
                (
                    "notify::ice-connection-state",
                    self._on_ice_connection_state_changed,
                ),
            ):
                branch.signal_handler_ids.append(
                    webrtcbin.connect(
                        signal_name,
                        callback,
                        generation,
                        session.session_id,
                    )
                )

            # Bring up downstream while it is still isolated, then attach the
            # tee request pad as the final operation. This guarantees that the
            # fresh branch receives sticky events before any media buffer.
            if not webrtcbin.sync_state_with_parent():
                raise RuntimeError("failed to sync webrtcbin state")
            if not queue_element.sync_state_with_parent():
                raise RuntimeError("failed to sync queue state")
            if tee_src_pad.link(queue_sink_pad) != Gst.PadLinkReturn.OK:
                raise RuntimeError("failed to link RTP tee to fresh queue")
        except Exception as exc:  # noqa: BLE001 - a partial branch must fail closed.
            log.error(
                "[WEBRTC-UPLINK] disposable branch creation failed: %s",
                exc,
            )
            self._teardown_branch_on_main_loop(generation)
            return None
        return branch

    def _teardown_branch_on_main_loop(
        self,
        expected_generation: int | None = None,
    ) -> bool:
        with self._stats_lock:
            branch = self._branch
            if branch is None:
                return not self._branch_cleanup_failed
            if (
                expected_generation is not None
                and branch.generation != expected_generation
            ):
                return True

        for handler_id in tuple(branch.signal_handler_ids):
            try:
                branch.webrtcbin.disconnect(handler_id)
                branch.signal_handler_ids.remove(handler_id)
            except Exception as exc:  # noqa: BLE001 - retry only this resource later.
                log.error("[WEBRTC-UPLINK] signal disconnect failed: %s", exc)

        if not branch.tee_unlinked:
            try:
                branch.tee_unlinked = _unlink_exact(
                    branch.tee_src_pad,
                    branch.queue_sink_pad,
                )
            except Exception as exc:  # noqa: BLE001 - retry only this resource later.
                log.error("[WEBRTC-UPLINK] tee unlink failed: %s", exc)

        if branch.tee_unlinked and not branch.tee_pad_released:
            try:
                if self._uplink_tee is not None:
                    self._uplink_tee.release_request_pad(branch.tee_src_pad)
                    branch.tee_pad_released = True
            except Exception as exc:  # noqa: BLE001 - retry only this resource later.
                log.error("[WEBRTC-UPLINK] tee request-pad release failed: %s", exc)

        # Stop downstream before the queue. Each completion flag is monotonic:
        # a later retry never repeats a successful request-pad release, handler
        # disconnect, unlink, or bin removal.
        for element, attribute in (
            (branch.webrtcbin, "webrtc_stopped"),
            (branch.queue, "queue_stopped"),
        ):
            if getattr(branch, attribute):
                continue
            try:
                result = element.set_state(Gst.State.NULL)
                if result != Gst.StateChangeReturn.FAILURE:
                    setattr(branch, attribute, True)
            except Exception as exc:  # noqa: BLE001 - retry only this resource later.
                log.error("[WEBRTC-UPLINK] element stop failed: %s", exc)

        if not branch.webrtc_unlinked:
            try:
                branch.webrtc_unlinked = _unlink_exact(
                    branch.queue_src_pad,
                    branch.webrtc_sink_pad,
                )
            except Exception as exc:  # noqa: BLE001 - retry only this resource later.
                log.error("[WEBRTC-UPLINK] webrtc unlink failed: %s", exc)

        if branch.webrtc_unlinked and not branch.webrtc_pad_released:
            try:
                branch.webrtcbin.release_request_pad(branch.webrtc_sink_pad)
                branch.webrtc_pad_released = True
            except Exception as exc:  # noqa: BLE001 - retry only this resource later.
                log.error("[WEBRTC-UPLINK] webrtc request-pad release failed: %s", exc)

        pipeline = self._pipeline
        for element, stopped, prerequisites, attribute in (
            (
                branch.queue,
                branch.queue_stopped,
                branch.tee_pad_released and branch.webrtc_unlinked,
                "queue_removed",
            ),
            (
                branch.webrtcbin,
                branch.webrtc_stopped,
                branch.webrtc_pad_released,
                "webrtc_removed",
            ),
        ):
            if getattr(branch, attribute) or not stopped or not prerequisites:
                continue
            try:
                parent = element.get_parent()
                if parent is pipeline and pipeline is not None:
                    pipeline.remove(element)
                    parent = element.get_parent()
                if parent is None:
                    setattr(branch, attribute, True)
            except Exception as exc:  # noqa: BLE001 - retry only this resource later.
                log.error("[WEBRTC-UPLINK] element removal failed: %s", exc)

        success = bool(
            not branch.signal_handler_ids
            and branch.tee_unlinked
            and branch.tee_pad_released
            and branch.webrtc_stopped
            and branch.queue_stopped
            and branch.webrtc_unlinked
            and branch.webrtc_pad_released
            and branch.queue_removed
            and branch.webrtc_removed
        )

        with self._stats_lock:
            if self._branch is branch:
                if success:
                    self._branch = None
                    self._branch_cleanup_failed = False
                    cleanup_source_id = self._clear_cleanup_retry_locked()
                    self._cleanup_attempts = 0
                    self._cleanup_fatal_reported = False
                else:
                    # Do not create another branch over incompletely detached
                    # request pads. Process restart is the only safe recovery.
                    self._branch_cleanup_failed = True
                    cleanup_source_id = None
            else:
                cleanup_source_id = None
        self._remove_glib_source(cleanup_source_id)
        if success:
            log.info(
                "[WEBRTC-UPLINK] disposed session branch generation=%s sessionId=%s",
                branch.generation,
                branch.session_id,
            )
        return success

    def _abort_current_session_on_main_loop(
        self,
        generation: int,
        session_id: str,
    ) -> None:
        self._fail_current_signaling_session(
            generation,
            session_id,
            reason="local WebRTC session setup failed",
        )

    def _stop_on_main_loop(self, branch_generation: int) -> bool:
        if self._teardown_branch_on_main_loop(branch_generation):
            return False
        self._schedule_cleanup_retry_or_escalate_on_main_loop(branch_generation)
        return False

    def _on_cleanup_retry(self, branch_generation: int) -> bool:
        with self._stats_lock:
            if self._cleanup_retry_generation != branch_generation:
                return False
            self._clear_cleanup_retry_locked()
        return self._stop_on_main_loop(branch_generation)

    def _schedule_cleanup_retry_or_escalate_on_main_loop(
        self,
        branch_generation: int,
    ) -> None:
        with self._stats_lock:
            branch = self._branch
            if branch is None or branch.generation != branch_generation:
                return
            if self._cleanup_retry_source_id is not None:
                return
            self._cleanup_attempts += 1
            if self._cleanup_attempts < _BRANCH_CLEANUP_MAX_ATTEMPTS:
                should_retry = True
                should_escalate = False
            elif not self._cleanup_fatal_reported:
                self._cleanup_fatal_reported = True
                should_retry = False
                should_escalate = True
            else:
                return

        if should_retry:
            source_id = GLib.timeout_add(
                _BRANCH_CLEANUP_RETRY_INTERVAL_MS,
                self._on_cleanup_retry,
                branch_generation,
            )
            with self._stats_lock:
                branch = self._branch
                if (
                    branch is not None
                    and branch.generation == branch_generation
                    and self._branch_cleanup_failed
                    and self._cleanup_retry_source_id is None
                ):
                    self._cleanup_retry_source_id = source_id
                    self._cleanup_retry_generation = branch_generation
                    stale_source_id = None
                else:
                    stale_source_id = source_id
            self._remove_glib_source(stale_source_id)
            return

        if not should_escalate:
            return
        reason = (
            "WebRTC branch cleanup remained incomplete after "
            f"{_BRANCH_CLEANUP_MAX_ATTEMPTS} attempts"
        )
        log.critical("[WEBRTC-UPLINK] %s; requesting process recovery", reason)
        handled = False
        if self._on_fatal_cleanup is not None:
            try:
                handled = bool(self._on_fatal_cleanup(reason))
            except Exception as exc:  # noqa: BLE001 - local release is mandatory.
                log.critical(
                    "[WEBRTC-UPLINK] fatal cleanup callback raised: %s",
                    exc,
                )
        if not handled and self._pipeline is not None:
            # Standalone/dev callers may not have a supervisor callback. Stop
            # the complete graph so the residual branch cannot keep consuming
            # physical camera buffers indefinitely.
            try:
                self._pipeline.set_state(Gst.State.NULL)
            except Exception as exc:  # noqa: BLE001 - process must be replaced.
                log.critical(
                    "[WEBRTC-UPLINK] failed to stop pipeline after cleanup failure: %s",
                    exc,
                )

    def _active_branch_for_token(
        self,
        generation: int,
        session_id: str,
        element: Gst.Element | None = None,
    ) -> _WebRTCBranch | None:
        with self._stats_lock:
            branch = self._branch
            if (
                generation != self._stats_generation
                or self._session is None
                or self._session.session_id != session_id
                or branch is None
                or branch.generation != generation
                or branch.session_id != session_id
                or self._branch_cleanup_failed
                or (element is not None and branch.webrtcbin is not element)
            ):
                return None
            return branch

    def _apply_answer_on_main_loop(
        self,
        generation: int,
        session_id: str,
        sdp_text: str,
    ) -> bool:
        branch = self._active_branch_for_token(generation, session_id)
        if branch is None:
            return False

        description = _build_session_description(
            GstWebRTC.WebRTCSDPType.ANSWER,
            sdp_text,
        )
        if description is None:
            log.error("[WEBRTC-UPLINK] failed to parse SDP answer.")
            self._fail_current_signaling_session(
                generation,
                session_id,
                reason="invalid SDP answer",
            )
            return False

        with self._stats_lock:
            if self._active_branch_for_token(generation, session_id) is not branch:
                return False
            if branch.answer_pending or branch.answer_applied:
                return False
            branch.answer_pending = True
        promise = Gst.Promise.new_with_change_func(
            self._on_remote_description_set,
            (generation, session_id),
            None,
        )
        try:
            branch.webrtcbin.emit(
                "set-remote-description",
                description,
                promise,
            )
        except Exception as exc:  # noqa: BLE001 - malformed remote state fails closed.
            self._fail_current_signaling_session(
                generation,
                session_id,
                reason=f"SDP answer application failed: {exc}",
            )
        return False

    def _on_remote_description_set(
        self,
        promise: Gst.Promise,
        token: object = None,
        *_args: object,
    ) -> None:
        if not isinstance(token, tuple) or len(token) != 2:
            return
        generation, session_id = token
        if not isinstance(generation, int) or not isinstance(session_id, str):
            return
        branch = self._active_branch_for_token(generation, session_id)
        if branch is None:
            return
        failure = self._promise_failure_reason(promise)
        if failure is not None:
            self._fail_current_signaling_session(
                generation,
                session_id,
                reason=f"remote SDP description failed: {failure}",
            )
            return
        with self._stats_lock:
            if self._active_branch_for_token(generation, session_id) is not branch:
                return
            if not branch.answer_pending:
                return
            branch.answer_pending = False
            branch.answer_applied = True
            watchdog_source_id = self._clear_offer_watchdog_locked()
        self._remove_glib_source(watchdog_source_id)
        GLib.idle_add(
            self._arm_connection_watchdog_on_main_loop,
            generation,
            session_id,
        )
        log.info(
            "[WEBRTC-UPLINK] applied SDP answer. sessionId=%s generation=%s",
            session_id,
            generation,
        )

    def _add_remote_candidate_on_main_loop(
        self,
        generation: int,
        session_id: str,
        mline_index: int,
        candidate: str,
        _sdp_mid: str | None,
    ) -> bool:
        branch = self._active_branch_for_token(generation, session_id)
        if branch is None:
            return False
        with self._stats_lock:
            if self._active_branch_for_token(generation, session_id) is not branch:
                return False
            branch.webrtcbin.emit("add-ice-candidate", mline_index, candidate)
        log.debug(
            "[WEBRTC-UPLINK] added remote ICE candidate sessionId=%s generation=%s mline=%s",
            session_id,
            generation,
            mline_index,
        )
        return False

    def _on_offer_created(
        self,
        promise: Gst.Promise,
        token: object = None,
        *_args: object,
    ) -> None:
        if not isinstance(token, tuple) or len(token) != 2:
            return
        generation, session_id = token
        if not isinstance(generation, int) or not isinstance(session_id, str):
            return
        branch = self._active_branch_for_token(generation, session_id)
        if branch is None:
            return

        failure = self._promise_failure_reason(promise)
        if failure is not None:
            self._fail_current_signaling_session(
                generation,
                session_id,
                reason=f"offer creation failed: {failure}",
            )
            return
        reply = promise.get_reply()
        if reply is None:
            log.error("[WEBRTC-UPLINK] offer promise returned no reply.")
            self._fail_current_signaling_session(
                generation,
                session_id,
                reason="offer promise returned no reply",
            )
            return

        offer = reply.get_value("offer")
        if offer is None:
            log.error("[WEBRTC-UPLINK] offer promise missing offer value.")
            self._fail_current_signaling_session(
                generation,
                session_id,
                reason="offer promise missing offer value",
            )
            return

        try:
            sdp_text = enforce_h264_offer_parameters(
                offer.sdp.as_text(),
                profile_level_id=self._h264_profile_level_id,
                packetization_mode=self._h264_packetization_mode,
                level_asymmetry_allowed=self._h264_level_asymmetry_allowed,
            )
        except ValueError as exc:
            log.error("[WEBRTC-UPLINK] refusing incompatible H264 offer: %s", exc)
            self._fail_current_signaling_session(
                generation,
                session_id,
                reason=f"incompatible H264 offer: {exc}",
            )
            return
        local_offer = _build_session_description(
            GstWebRTC.WebRTCSDPType.OFFER,
            sdp_text,
        )
        if local_offer is None:
            log.error("[WEBRTC-UPLINK] failed to parse canonical H264 SDP offer.")
            self._fail_current_signaling_session(
                generation,
                session_id,
                reason="canonical H264 SDP offer could not be parsed",
            )
            return
        with self._stats_lock:
            session = self._session
            if (
                session is None
                or session.session_id != session_id
                or generation != self._stats_generation
                or self._active_branch_for_token(
                    generation,
                    session_id,
                    branch.webrtcbin,
                )
                is not branch
            ):
                return
            payload = build_uplink_payload(
                WEBRTC_UPLINK_OFFER,
                session.broadcast_id,
                session_id,
                sdp=sdp_text,
            )
            signaling_token = WebRTCSignalingToken(generation, session_id)
        local_description_promise = Gst.Promise.new_with_change_func(
            self._on_local_description_set,
            (
                generation,
                session_id,
                payload,
                signaling_token,
            ),
            None,
        )
        try:
            branch.webrtcbin.emit(
                "set-local-description",
                local_offer,
                local_description_promise,
            )
        except Exception as exc:  # noqa: BLE001 - local negotiation must be bounded.
            self._fail_current_signaling_session(
                generation,
                session_id,
                reason=f"local SDP offer application failed: {exc}",
            )
        return

    def _on_local_description_set(
        self,
        promise: Gst.Promise,
        token: object = None,
        *_args: object,
    ) -> None:
        if not isinstance(token, tuple) or len(token) != 4:
            return
        generation, session_id, payload, signaling_token = token
        if (
            not isinstance(generation, int)
            or not isinstance(session_id, str)
            or not isinstance(payload, dict)
            or not isinstance(signaling_token, WebRTCSignalingToken)
        ):
            return
        branch = self._active_branch_for_token(generation, session_id)
        if branch is None:
            return
        failure = self._promise_failure_reason(promise)
        if failure is not None:
            self._fail_current_signaling_session(
                generation,
                session_id,
                reason=f"local SDP description failed: {failure}",
            )
            return
        if not self.is_signaling_token_current(signaling_token):
            return
        try:
            sent = bool(
                self._send_message(
                    WEBRTC_UPLINK_OFFER_DEST,
                    payload,
                    True,
                    signaling_token,
                )
            )
        except Exception as exc:  # noqa: BLE001 - transport callback must fail closed.
            sent = False
            log.error("[WEBRTC-UPLINK] offer enqueue raised: %s", exc)
        if not sent:
            self._fail_current_signaling_session(
                generation,
                session_id,
                reason="SDP offer was not accepted by the outbound queue",
            )
            return
        with self._stats_lock:
            current_branch = self._active_branch_for_token(generation, session_id)
            if current_branch is not branch:
                return
            branch.offer_enqueued = True
        GLib.idle_add(
            self._arm_offer_watchdog_on_main_loop,
            generation,
            session_id,
        )
        log.info(
            "[WEBRTC-UPLINK] sent SDP offer. sessionId=%s generation=%s",
            session_id,
            generation,
        )

    def _on_ice_candidate(
        self,
        element: Gst.Element,
        mline_index: int,
        candidate: str,
        generation: int,
        session_id: str,
    ) -> None:
        branch = self._active_branch_for_token(generation, session_id, element)
        if branch is None:
            return
        candidate_text = str(candidate or "").strip()
        if not candidate_text:
            log.debug(
                "[WEBRTC-UPLINK] skip empty local ICE candidate. sessionId=%s generation=%s mline=%s",
                session_id,
                generation,
                mline_index,
            )
            return
        with self._stats_lock:
            session = self._session
            if (
                session is None
                or session.session_id != session_id
                or generation != self._stats_generation
                or self._active_branch_for_token(
                    generation,
                    session_id,
                    element,
                )
                is not branch
            ):
                return
            payload = build_uplink_payload(
                WEBRTC_UPLINK_ICE_CANDIDATE,
                session.broadcast_id,
                session_id,
                candidate=candidate_text,
                sdpMLineIndex=int(mline_index),
                sdpMid="video",
            )
            signaling_token = WebRTCSignalingToken(generation, session_id)
        if not self.is_signaling_token_current(signaling_token):
            return
        try:
            sent = bool(
                self._send_message(
                    WEBRTC_UPLINK_ICE_CANDIDATE_DEST,
                    payload,
                    False,
                    signaling_token,
                )
            )
        except Exception as exc:  # noqa: BLE001 - a lost candidate fails negotiation.
            sent = False
            log.error("[WEBRTC-UPLINK] ICE candidate enqueue raised: %s", exc)
        if not sent:
            # ICE signaling is volatile and generation-scoped. Retrying after
            # queue backpressure can cross the negotiation boundary, so tear
            # down this exact generation and wait for a fresh start command.
            self._fail_current_signaling_session(
                generation,
                session_id,
                reason="local ICE candidate was not accepted by the outbound queue",
            )

    def _on_connection_state_changed(
        self,
        element: Gst.Element,
        _pspec: object,
        generation: int,
        session_id: str,
    ) -> None:
        if self._active_branch_for_token(generation, session_id, element) is None:
            return
        state = element.get_property("connection-state")
        state_nick = str(getattr(state, "value_nick", str(state))).lower()
        connection_source_id = None
        with self._stats_lock:
            if generation != self._stats_generation:
                return
            if state_nick == "connected":
                if self._connection_state != "connected":
                    self._connected_since = time.monotonic()
            else:
                self._connected_since = None
                self._reset_outbound_progress_locked()
            self._connection_state = state_nick
            if state_nick == "connected" and self._ice_connection_state in {
                "connected",
                "completed",
            }:
                connection_source_id = self._clear_connection_watchdog_locked()
            stop_sent = self._stop_sent
        self._remove_glib_source(connection_source_id)
        log.info(
            "[WEBRTC-UPLINK] connection-state=%s generation=%s",
            state_nick,
            generation,
        )
        if state_nick in {"failed", "closed"}:
            self.stop(send_signal=not stop_sent)

    def _on_ice_connection_state_changed(
        self,
        element: Gst.Element,
        _pspec: object,
        generation: int,
        session_id: str,
    ) -> None:
        if self._active_branch_for_token(generation, session_id, element) is None:
            return
        state = element.get_property("ice-connection-state")
        state_nick = str(getattr(state, "value_nick", str(state))).lower()
        connection_source_id = None
        with self._stats_lock:
            if generation != self._stats_generation:
                return
            if state_nick in {"connected", "completed"}:
                if self._ice_connection_state not in {"connected", "completed"}:
                    self._ice_connected_since = time.monotonic()
            else:
                self._ice_connected_since = None
                self._reset_outbound_progress_locked()
            self._ice_connection_state = state_nick
            if (
                state_nick in {"connected", "completed"}
                and self._connection_state == "connected"
            ):
                connection_source_id = self._clear_connection_watchdog_locked()
            stop_sent = self._stop_sent
        self._remove_glib_source(connection_source_id)
        log.info(
            "[WEBRTC-UPLINK] ice-connection-state=%s generation=%s",
            state_nick,
            generation,
        )
        if state_nick in {"failed", "closed", "disconnected"}:
            self.stop(send_signal=not stop_sent)

    def _send_stop_message_for(
        self,
        session: WebRTCUplinkSession,
        signaling_token: WebRTCSignalingToken,
    ) -> bool:
        if not self.is_signaling_token_current(signaling_token):
            return False
        payload = build_uplink_payload(
            WEBRTC_UPLINK_STOP,
            session.broadcast_id,
            session.session_id,
        )
        return self._send_message(
            WEBRTC_UPLINK_STOP_DEST,
            payload,
            False,
            signaling_token,
        )


def _build_session_description(
    sdp_type: GstWebRTC.WebRTCSDPType,
    sdp_text: str,
) -> GstWebRTC.WebRTCSessionDescription | None:
    result, sdp_message = GstSdp.SDPMessage.new()
    if result != GstSdp.SDPResult.OK:
        return None
    parse_result = GstSdp.sdp_message_parse_buffer(
        bytes(sdp_text.encode("utf-8")), sdp_message
    )
    if parse_result != GstSdp.SDPResult.OK:
        return None
    return GstWebRTC.WebRTCSessionDescription.new(sdp_type, sdp_message)


def describe_payload(payload: dict[str, Any]) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False)
    except Exception:  # noqa: BLE001 - diagnostic fallback accepts any object.
        return str(payload)
