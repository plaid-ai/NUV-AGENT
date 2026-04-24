from __future__ import annotations

import json
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

            query = f"?{parsed.query}" if parsed.query else ""
            auth_prefix = ""
            if username and credential:
                auth_prefix = f"{_quote_turn_username(username)}:{_quote_turn_password(credential)}@"
            turn_servers.append(
                f"{_uri_scheme_prefix(parsed.scheme)}{auth_prefix}{host}:{port}{query}"
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
