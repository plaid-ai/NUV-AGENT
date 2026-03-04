import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

AIRPORT_DEFAULT_PATH = "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport"

RSSI_AIRPORT_PATTERN = re.compile(r"agrCtlRSSI:\s*(-?\d+)")
RSSI_IW_PATTERN = re.compile(r"signal:\s*(-?\d+)\s*dBm", re.IGNORECASE)
PING_PACKET_LOSS_PATTERN = re.compile(r"(\d+(?:\.\d+)?)%\s*packet loss", re.IGNORECASE)
PING_RTT_PATTERN = re.compile(
    r"(?:round-trip|rtt)[^=]*=\s*([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+)\s*ms",
    re.IGNORECASE,
)


def run_command_output(cmd: list[str], timeout_sec: float = 3.0) -> str | None:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=max(1.0, timeout_sec))
    except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired, OSError):
        return None

    chunks = [chunk.strip() for chunk in (result.stdout, result.stderr) if chunk and chunk.strip()]
    if not chunks:
        return None
    return "\n".join(chunks)


def parse_airport_output_for_rssi(output: str | None) -> int | None:
    if not output:
        return None
    match = RSSI_AIRPORT_PATTERN.search(output)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def parse_iw_link_output_for_rssi(output: str | None) -> int | None:
    if not output:
        return None
    match = RSSI_IW_PATTERN.search(output)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def parse_ping_output(output: str | None) -> tuple[float | None, int | None]:
    if not output:
        return None, None

    packet_loss_pct = None
    rtt_avg_ms = None

    loss_match = PING_PACKET_LOSS_PATTERN.search(output)
    if loss_match:
        try:
            packet_loss_pct = float(loss_match.group(1))
        except ValueError:
            packet_loss_pct = None

    rtt_match = PING_RTT_PATTERN.search(output)
    if rtt_match:
        try:
            rtt_avg_ms = int(round(float(rtt_match.group(2))))
        except ValueError:
            rtt_avg_ms = None

    return packet_loss_pct, rtt_avg_ms


def detect_linux_wifi_interface(
    run_command_fn: Callable[[list[str], float], str | None] = run_command_output,
) -> str | None:
    output = run_command_fn(["iw", "dev"], 2.0)
    if not output:
        return None

    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("Interface "):
            return stripped.split(" ", 1)[1].strip() or None
    return None


def collect_rssi_dbm(
    wifi_interface: str | None = None,
    platform_name: str | None = None,
    run_command_fn: Callable[[list[str], float], str | None] = run_command_output,
) -> int | None:
    platform = platform_name or sys.platform

    if platform == "darwin":
        airport_path = os.getenv("NUVION_AIRPORT_PATH", AIRPORT_DEFAULT_PATH)
        output = run_command_fn([airport_path, "-I"], 2.0)
        if output is None and airport_path != "airport":
            output = run_command_fn(["airport", "-I"], 2.0)
        return parse_airport_output_for_rssi(output)

    if platform.startswith("linux"):
        iface = (wifi_interface or "").strip() or detect_linux_wifi_interface(run_command_fn)
        if not iface:
            return None
        output = run_command_fn(["iw", "dev", iface, "link"], 2.0)
        return parse_iw_link_output_for_rssi(output)

    return None


def collect_ping_metrics(
    host: str,
    platform_name: str | None = None,
    run_command_fn: Callable[[list[str], float], str | None] = run_command_output,
    ping_count: int = 3,
    ping_timeout_sec: int = 1,
) -> tuple[float | None, int | None]:
    target = (host or "").strip()
    if not target:
        return None, None

    platform = platform_name or sys.platform
    cmd = ["ping", "-c", str(max(1, ping_count)), target]
    if platform.startswith("linux"):
        cmd = ["ping", "-c", str(max(1, ping_count)), "-W", str(max(1, ping_timeout_sec)), target]

    timeout = max(3.0, float(ping_count) * float(ping_timeout_sec) + 2.0)
    output = run_command_fn(cmd, timeout)
    return parse_ping_output(output)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ConnectivityThresholds:
    poor_rssi_dbm: int = -80
    poor_packet_loss_pct: float = 8.0
    poor_rtt_ms: int = 250


class ConnectivityReporter:
    def __init__(
        self,
        target_host: str,
        wifi_interface: str | None = None,
        thresholds: ConnectivityThresholds | None = None,
        min_send_interval_sec: float = 30.0,
        rssi_collector: Callable[[], int | None] | None = None,
        ping_collector: Callable[[], tuple[float | None, int | None]] | None = None,
        clock: Callable[[], float] | None = None,
        measured_at_factory: Callable[[], str] | None = None,
    ):
        self.target_host = target_host
        self.wifi_interface = wifi_interface
        self.thresholds = thresholds or ConnectivityThresholds()
        self.min_send_interval_sec = max(1.0, float(min_send_interval_sec))

        self._clock = clock or time.time
        self._measured_at_factory = measured_at_factory or utc_now_iso
        self._rssi_collector = rssi_collector or (
            lambda: collect_rssi_dbm(wifi_interface=self.wifi_interface)
        )
        self._ping_collector = ping_collector or (
            lambda: collect_ping_metrics(self.target_host)
        )

        self._last_quality: str | None = None
        self._last_sent_at: float = 0.0

    def build_transition_payload(self) -> dict | None:
        rssi_dbm = self._rssi_collector()
        packet_loss_pct, rtt_ms = self._ping_collector()

        reasons: list[str] = []
        if rssi_dbm is not None and rssi_dbm <= self.thresholds.poor_rssi_dbm:
            reasons.append("wifi_rssi_low")
        if packet_loss_pct is not None and packet_loss_pct >= self.thresholds.poor_packet_loss_pct:
            reasons.append("packet_loss_high")
        if rtt_ms is not None and rtt_ms >= self.thresholds.poor_rtt_ms:
            reasons.append("rtt_high")

        if rssi_dbm is None and packet_loss_pct is None and rtt_ms is None:
            return None

        quality = "POOR" if reasons else "GOOD"
        reason = ",".join(reasons) if reasons else "healthy"
        now = self._clock()

        if self._last_quality is None:
            self._last_quality = quality
            if quality != "POOR":
                return None
            if now - self._last_sent_at < self.min_send_interval_sec:
                return None
            self._last_sent_at = now
            return self._build_payload(quality, reason, rssi_dbm, packet_loss_pct, rtt_ms)

        if quality == self._last_quality:
            return None

        if now - self._last_sent_at < self.min_send_interval_sec:
            return None

        self._last_quality = quality
        self._last_sent_at = now
        return self._build_payload(quality, reason, rssi_dbm, packet_loss_pct, rtt_ms)

    def _build_payload(
        self,
        quality: str,
        reason: str,
        rssi_dbm: int | None,
        packet_loss_pct: float | None,
        rtt_ms: int | None,
    ) -> dict:
        return {
            "quality": quality,
            "reason": reason,
            "rssiDbm": rssi_dbm,
            "packetLossPct": packet_loss_pct,
            "rttMs": rtt_ms,
            "uplinkKbps": None,
            "downlinkKbps": None,
            "measuredAt": self._measured_at_factory(),
        }
