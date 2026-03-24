from __future__ import annotations

import unittest

from nuvion_app.inference.connectivity import ConnectivityReporter
from nuvion_app.inference.connectivity import ConnectivityThresholds
from nuvion_app.inference.connectivity import collect_link_bitrate_kbps
from nuvion_app.inference.connectivity import collect_ping_metrics
from nuvion_app.inference.connectivity import collect_rssi_dbm
from nuvion_app.inference.connectivity import parse_airport_output_for_bitrate_kbps
from nuvion_app.inference.connectivity import parse_airport_output_for_rssi
from nuvion_app.inference.connectivity import parse_iw_link_output_for_bitrate_kbps
from nuvion_app.inference.connectivity import parse_iw_link_output_for_rssi
from nuvion_app.inference.connectivity import parse_ping_output


class ConnectivityParserTest(unittest.TestCase):
    def test_parse_airport_output_for_rssi(self) -> None:
        output = """
            agrCtlRSSI: -67
            agrCtlNoise: -91
        """
        self.assertEqual(parse_airport_output_for_rssi(output), -67)

    def test_parse_iw_link_output_for_rssi(self) -> None:
        output = """
            Connected to aa:bb:cc:dd:ee:ff (on wlan0)
            \tsignal: -78 dBm
            \ttx bitrate: 54.0 MBit/s
        """
        self.assertEqual(parse_iw_link_output_for_rssi(output), -78)

    def test_parse_airport_output_for_bitrate_kbps(self) -> None:
        output = """
            agrCtlRSSI: -67
            lastTxRate: 351
            maxRate: 780
        """
        self.assertEqual(parse_airport_output_for_bitrate_kbps(output), (351000, 780000))

    def test_parse_iw_link_output_for_bitrate_kbps(self) -> None:
        output = """
            Connected to aa:bb:cc:dd:ee:ff (on wlan0)
            \tsignal: -78 dBm
            \ttx bitrate: 54.0 MBit/s
            \trx bitrate: 18.5 MBit/s
        """
        self.assertEqual(parse_iw_link_output_for_bitrate_kbps(output), (54000, 18500))

    def test_parse_ping_output_linux_and_macos(self) -> None:
        linux_output = """
            3 packets transmitted, 3 received, 0% packet loss, time 2002ms
            rtt min/avg/max/mdev = 23.100/25.700/28.000/2.500 ms
        """
        self.assertEqual(parse_ping_output(linux_output), (0.0, 26))

        mac_output = """
            3 packets transmitted, 3 packets received, 0.0% packet loss
            round-trip min/avg/max/stddev = 19.436/21.018/23.971/1.890 ms
        """
        self.assertEqual(parse_ping_output(mac_output), (0.0, 21))

    def test_collect_rssi_dbm_for_darwin(self) -> None:
        def fake_run(cmd: list[str], timeout: float) -> str | None:
            _ = timeout
            if cmd[-1] == "-I":
                return "agrCtlRSSI: -64\nagrCtlNoise: -92"
            return None

        self.assertEqual(collect_rssi_dbm(platform_name="darwin", run_command_fn=fake_run), -64)

    def test_collect_rssi_dbm_for_linux(self) -> None:
        def fake_run(cmd: list[str], timeout: float) -> str | None:
            _ = timeout
            if cmd == ["iw", "dev"]:
                return "phy#0\n\tInterface wlan0\n"
            if cmd == ["iw", "dev", "wlan0", "link"]:
                return "Connected\n\tsignal: -72 dBm\n"
            return None

        self.assertEqual(collect_rssi_dbm(platform_name="linux", run_command_fn=fake_run), -72)

    def test_collect_ping_metrics(self) -> None:
        def fake_run(cmd: list[str], timeout: float) -> str | None:
            _ = cmd
            _ = timeout
            return "3 packets transmitted, 3 received, 0% packet loss\nrtt min/avg/max/mdev = 10.1/15.4/18.0/1.2 ms"

        self.assertEqual(collect_ping_metrics("api.example.com", platform_name="linux", run_command_fn=fake_run), (0.0, 15))

    def test_collect_link_bitrate_for_darwin(self) -> None:
        def fake_run(cmd: list[str], timeout: float) -> str | None:
            _ = timeout
            if cmd[-1] == "-I":
                return "lastTxRate: 433\nmaxRate: 866"
            return None

        self.assertEqual(collect_link_bitrate_kbps(platform_name="darwin", run_command_fn=fake_run), (433000, 866000))

    def test_collect_link_bitrate_for_linux(self) -> None:
        def fake_run(cmd: list[str], timeout: float) -> str | None:
            _ = timeout
            if cmd == ["iw", "dev"]:
                return "phy#0\n\tInterface wlan0\n"
            if cmd == ["iw", "dev", "wlan0", "link"]:
                return "Connected\n\ttx bitrate: 65.0 MBit/s\n\trx bitrate: 32.5 MBit/s\n"
            return None

        self.assertEqual(collect_link_bitrate_kbps(platform_name="linux", run_command_fn=fake_run), (65000, 32500))


class ConnectivityReporterTest(unittest.TestCase):
    def test_reporter_sends_only_transitions(self) -> None:
        samples = [
            {"rssi": -60, "loss": 0.0, "rtt": 40},
            {"rssi": -85, "loss": 0.0, "rtt": 40},
            {"rssi": -84, "loss": 0.0, "rtt": 40},
            {"rssi": -62, "loss": 0.0, "rtt": 40},
        ]
        idx = {"i": 0}

        def current_sample() -> dict:
            return samples[min(idx["i"], len(samples) - 1)]

        def rssi_collector() -> int | None:
            return current_sample()["rssi"]

        def ping_collector() -> tuple[float | None, int | None]:
            sample = current_sample()
            idx["i"] += 1
            return sample["loss"], sample["rtt"]

        def bitrate_collector() -> tuple[int | None, int | None]:
            return 12000, 18000

        ticks = iter([0.0, 2.0, 4.0, 6.0, 8.0])

        reporter = ConnectivityReporter(
            target_host="api.example.com",
            thresholds=ConnectivityThresholds(poor_rssi_dbm=-80, poor_packet_loss_pct=8.0, poor_rtt_ms=250),
            min_send_interval_sec=1.0,
            rssi_collector=rssi_collector,
            ping_collector=ping_collector,
            bitrate_collector=bitrate_collector,
            clock=lambda: next(ticks),
            measured_at_factory=lambda: "2026-03-04T07:42:10Z",
        )

        first = reporter.build_transition_payload()
        second = reporter.build_transition_payload()
        third = reporter.build_transition_payload()
        fourth = reporter.build_transition_payload()

        self.assertIsNone(first)  # Initial GOOD state is ignored.
        self.assertIsNotNone(second)
        self.assertEqual(second["quality"], "POOR")
        self.assertIn("wifi_rssi_low", second["reason"])
        self.assertEqual(second["uplinkKbps"], 12000)
        self.assertEqual(second["downlinkKbps"], 18000)
        self.assertIsNone(third)  # Repeated POOR state is ignored.
        self.assertIsNotNone(fourth)
        self.assertEqual(fourth["quality"], "GOOD")
        self.assertEqual(fourth["uplinkKbps"], 12000)
        self.assertEqual(fourth["downlinkKbps"], 18000)

    def test_reporter_builds_snapshot_without_transition(self) -> None:
        reporter = ConnectivityReporter(
            target_host="api.example.com",
            thresholds=ConnectivityThresholds(poor_rssi_dbm=-80, poor_packet_loss_pct=8.0, poor_rtt_ms=250),
            rssi_collector=lambda: -61,
            ping_collector=lambda: (0.0, 24),
            bitrate_collector=lambda: (433000, 866000),
            measured_at_factory=lambda: "2026-03-04T07:42:10Z",
        )

        snapshot = reporter.build_snapshot_payload()

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot["quality"], "GOOD")
        self.assertEqual(snapshot["rssiDbm"], -61)
        self.assertEqual(snapshot["rttMs"], 24)
        self.assertEqual(snapshot["uplinkKbps"], 433000)
        self.assertEqual(snapshot["downlinkKbps"], 866000)


if __name__ == "__main__":
    unittest.main()
