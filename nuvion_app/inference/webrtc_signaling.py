from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import quote, urlparse


UPLINK_MODE_WEBRTC = "webrtc"
DEFAULT_UPLINK_MODE = UPLINK_MODE_WEBRTC

WEBRTC_UPLINK_START = "WEBRTC_UPLINK_START"
WEBRTC_UPLINK_ANSWER = "WEBRTC_UPLINK_ANSWER"
WEBRTC_UPLINK_ICE_CANDIDATE = "WEBRTC_UPLINK_ICE_CANDIDATE"
WEBRTC_UPLINK_STATE = "WEBRTC_UPLINK_STATE"
WEBRTC_UPLINK_OFFER = "WEBRTC_UPLINK_OFFER"
WEBRTC_UPLINK_STOP = "WEBRTC_UPLINK_STOP"

WEBRTC_UPLINK_OFFER_DEST = "/app/webrtc/uplink/offer"
WEBRTC_UPLINK_ICE_CANDIDATE_DEST = "/app/webrtc/uplink/ice-candidate"
WEBRTC_UPLINK_STOP_DEST = "/app/webrtc/uplink/stop"

_H264_PROFILE_LEVEL_ID = re.compile(r"^[0-9a-f]{6}$")
_H264_LEVEL_IDC = frozenset(
    {10, 11, 12, 13, 20, 21, 22, 30, 31, 32, 40, 41, 42, 50, 51, 52, 60, 61, 62}
)


def normalize_h264_profile_level_id(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _H264_PROFILE_LEVEL_ID.fullmatch(normalized):
        raise ValueError("H264 profile-level-id must be exactly six hexadecimal digits")
    if int(normalized[-2:], 16) not in _H264_LEVEL_IDC:
        raise ValueError("H264 profile-level-id has an unsupported level_idc")
    return normalized


def h264_level_from_profile_level_id(value: str) -> str:
    profile_level_id = normalize_h264_profile_level_id(value)
    level_idc = int(profile_level_id[-2:], 16)
    major, minor = divmod(level_idc, 10)
    return str(major) if minor == 0 else f"{major}.{minor}"


def enforce_h264_offer_parameters(
    sdp_text: str,
    *,
    profile_level_id: str,
    packetization_mode: str,
    level_asymmetry_allowed: str,
    payload_type: int = 96,
) -> str:
    """Bind the emitted H264 fmtp line to the configured ingest contract."""

    if not isinstance(sdp_text, str) or not sdp_text or len(sdp_text) > 1024 * 1024:
        raise ValueError("SDP offer size is invalid")
    if "\x00" in sdp_text:
        raise ValueError("SDP offer contains a NUL byte")
    if isinstance(payload_type, bool) or not isinstance(payload_type, int):
        raise ValueError("H264 payload type must be an integer")
    if not 0 <= payload_type <= 127:
        raise ValueError("H264 payload type is out of range")
    profile = normalize_h264_profile_level_id(profile_level_id)
    packetization = str(packetization_mode or "").strip()
    asymmetry = str(level_asymmetry_allowed or "").strip()
    if packetization not in {"0", "1"}:
        raise ValueError("H264 packetization-mode must be 0 or 1")
    if asymmetry not in {"0", "1"}:
        raise ValueError("H264 level-asymmetry-allowed must be 0 or 1")

    newline = "\r\n" if "\r\n" in sdp_text else "\n"
    terminated = sdp_text.endswith(("\r", "\n"))
    lines = sdp_text.splitlines()
    rtpmap_pattern = re.compile(
        rf"^a=rtpmap:{payload_type}\s+H264/90000(?:/\d+)?\s*$",
        re.IGNORECASE,
    )
    rtpmap_indexes = [
        index for index, line in enumerate(lines) if rtpmap_pattern.fullmatch(line)
    ]
    if len(rtpmap_indexes) != 1:
        raise ValueError("SDP offer must contain one configured H264 rtpmap line")

    fmtp_pattern = re.compile(rf"^a=fmtp:{payload_type}(?:\s+(.*))?$", re.IGNORECASE)
    fmtp_matches = [
        (index, match)
        for index, line in enumerate(lines)
        if (match := fmtp_pattern.fullmatch(line)) is not None
    ]
    if len(fmtp_matches) > 1:
        raise ValueError("SDP offer contains duplicate H264 fmtp lines")

    required = {
        "level-asymmetry-allowed": asymmetry,
        "packetization-mode": packetization,
        "profile-level-id": profile,
    }
    preserved: list[str] = []
    seen_required: set[str] = set()
    if fmtp_matches:
        fmtp_index, fmtp_match = fmtp_matches[0]
        raw_parameters = fmtp_match.group(1) or ""
        for raw_parameter in raw_parameters.split(";"):
            parameter = raw_parameter.strip()
            if not parameter:
                continue
            key = parameter.split("=", 1)[0].strip().lower()
            if key in required:
                if key in seen_required:
                    raise ValueError(f"SDP offer contains duplicate H264 parameter: {key}")
                seen_required.add(key)
            else:
                preserved.append(parameter)
    else:
        fmtp_index = rtpmap_indexes[0] + 1

    canonical = [f"{key}={value}" for key, value in required.items()]
    canonical.extend(preserved)
    fmtp_line = f"a=fmtp:{payload_type} {';'.join(canonical)}"
    if fmtp_matches:
        lines[fmtp_index] = fmtp_line
    else:
        lines.insert(fmtp_index, fmtp_line)
    rendered = newline.join(lines)
    return rendered + newline if terminated else rendered


def normalize_uplink_mode(value: str | None, default: str = DEFAULT_UPLINK_MODE) -> str:
    _ = value, default
    return UPLINK_MODE_WEBRTC


def parse_command_payload(body: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def parse_ice_servers(value: Any) -> list[dict[str, Any]]:
    if not value:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
    else:
        parsed = value
    if not isinstance(parsed, list):
        return []
    result: list[dict[str, Any]] = []
    for item in parsed:
        if isinstance(item, dict):
            result.append(item)
    return result


def _normalize_urls(raw_urls: Any) -> list[str]:
    if isinstance(raw_urls, str):
        return [raw_urls]
    if isinstance(raw_urls, list):
        return [str(item) for item in raw_urls if isinstance(item, str)]
    return []


def _uri_scheme_prefix(parsed_scheme: str) -> str:
    if parsed_scheme == "turns":
        return "turns://"
    return "turn://"


def _quote_turn_username(username: str) -> str:
    return quote(username, safe="")


def _quote_turn_password(password: str) -> str:
    return quote(password, safe="")


def _extract_host_port(raw_url: str, default_port: int) -> tuple[str | None, int]:
    parsed = urlparse(raw_url)
    host = parsed.hostname
    port = parsed.port
    if host:
        return host, port or default_port

    try:
        remainder = raw_url.split(":", 1)[1]
    except IndexError:
        return None, default_port

    remainder = remainder.lstrip("/")
    if "@" in remainder:
        remainder = remainder.rsplit("@", 1)[1]
    host_port = remainder.split("?", 1)[0]
    if ":" in host_port:
        host, port_str = host_port.rsplit(":", 1)
        try:
            return host, int(port_str)
        except ValueError:
            return host, default_port
    return host_port or None, default_port


def to_gst_ice_server_config(ice_servers: list[dict[str, Any]]) -> tuple[str | None, list[str]]:
    stun_server: str | None = None
    turn_servers: list[str] = []

    for server in ice_servers:
        username = str(server.get("username") or "")
        credential = str(server.get("credential") or "")
        for raw_url in _normalize_urls(server.get("urls")):
            parsed = urlparse(raw_url)
            if parsed.scheme == "stun":
                if not stun_server:
                    host, port = _extract_host_port(raw_url, 3478)
                    if host:
                        stun_server = f"stun://{host}:{port}"
                continue

            if parsed.scheme not in {"turn", "turns"}:
                continue

            host, port = _extract_host_port(raw_url, 3478)
            if not host:
                continue

            auth_prefix = ""
            if username and credential:
                auth_prefix = f"{_quote_turn_username(username)}:{_quote_turn_password(credential)}@"
            turn_servers.append(
                # webrtcbin expects turn(s)://username:password@host:port without browser-style
                # query parameters such as ?transport=udp. Passing the query through prevents relay
                # allocations on both macOS and Jetson runtimes.
                f"{_uri_scheme_prefix(parsed.scheme)}{auth_prefix}{host}:{port}"
            )

    return stun_server, turn_servers


def build_uplink_payload(
    message_type: str,
    broadcast_id: str,
    session_id: str,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": message_type,
        "broadcastId": broadcast_id,
        "sessionId": session_id,
    }
    payload.update(extra)
    return payload


def parse_stomp_heartbeat_header(value: Any) -> tuple[int, int]:
    if not isinstance(value, str):
        return 0, 0

    parts = [part.strip() for part in value.split(",", 1)]
    if len(parts) != 2:
        return 0, 0

    try:
        can_send = int(parts[0])
        wants_receive = int(parts[1])
    except ValueError:
        return 0, 0

    return max(0, can_send), max(0, wants_receive)


def negotiate_stomp_send_interval_ms(
    client_can_send_ms: int,
    server_heartbeat_header: Any,
) -> int | None:
    if client_can_send_ms <= 0:
        return None

    _server_can_send_ms, server_wants_receive_ms = parse_stomp_heartbeat_header(server_heartbeat_header)
    if server_wants_receive_ms <= 0:
        return None

    return max(client_can_send_ms, server_wants_receive_ms)
