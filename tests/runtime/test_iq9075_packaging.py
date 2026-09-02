from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class Iq9075PackagingTest(unittest.TestCase):
    def test_rollback_oak_probe_is_version_neutral_and_bounded(self) -> None:
        probe = (ROOT / "packaging/dev/probe-iq9075-oak.sh").read_text(
            encoding="utf-8"
        )
        runtime = (ROOT / "nuvion_updater/systemd_runtime.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("/opt/nuv-agent/current/venv/bin/python", probe)
        self.assertIn('version("depthai") != "2.32.0.0"', probe)
        self.assertIn("queue.tryGet()", probe)
        self.assertIn(
            "timeout --signal=TERM --kill-after=5s 60s runuser -u nuvion",
            probe,
        )
        self.assertIn("depthai.Device.getAllAvailableDevices()", probe)
        self.assertIn("stable_polls < 2", probe)
        self.assertIn("NUVION_DEPTHAI_DEVICE_ID", probe)
        self.assertIn("depthai.Device(pipeline, selected)", probe)
        self.assertIn('Path("/sys/bus/usb/devices")', probe)
        self.assertIn("IQ9075 requires exactly one attached OAK", probe)
        self.assertIn("deadline = time.monotonic() + 45.0", probe)
        self.assertIn("time.monotonic() + 10.0", probe)
        self.assertIn('chown nuvion:nuvion "$runtime_dir"', probe)
        self.assertNotIn("nuvion_app", probe)
        self.assertNotIn("WebRTCUplinkController", probe)
        self.assertNotIn("build_uplink_pipeline", probe)
        self.assertIn(
            'IQ9075_PROBE = "/usr/lib/nuvion-updater/probe-iq9075-oak.sh"',
            runtime,
        )

    def test_rollback_oak_probe_selects_only_the_configured_stable_device(
        self,
    ) -> None:
        probe = (ROOT / "packaging/dev/probe-iq9075-oak.sh").read_text(
            encoding="utf-8"
        )
        embedded = probe.split("<<'PY'\n", 2)[2].rsplit("\nPY", 1)[0]
        tree = ast.parse(embedded)
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"configured_mxid", "select_device"}
        ]
        namespace: dict[str, object] = {"Path": Path, "re": __import__("re")}
        exec(  # noqa: S102 - execute only two AST-selected repository functions.
            compile(ast.Module(body=functions, type_ignores=[]), "<probe>", "exec"),
            namespace,
        )
        configured_mxid = namespace["configured_mxid"]
        select_device = namespace["select_device"]

        class DeviceInfo:
            def __init__(self, mxid: str) -> None:
                self.mxid = mxid

            def getMxId(self) -> str:
                return self.mxid

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "agent.env"
            config.write_text("NUVION_DEPTHAI_DEVICE_ID=oak-b\n", encoding="utf-8")
            self.assertEqual(configured_mxid(config), "oak-b")
            devices = [DeviceInfo("oak-a"), DeviceInfo("oak-b")]
            self.assertIsNone(select_device(devices, "oak-b"))
            self.assertIsNone(select_device(devices, "missing"))
            self.assertIsNone(select_device(devices, None))
            self.assertIs(select_device(devices[:1], None), devices[0])
            self.assertIsNone(select_device(devices[:1], "oak-b"))

            config.write_text(
                "NUVION_DEPTHAI_DEVICE_ID=oak-a\n"
                "NUVION_DEPTHAI_DEVICE_ID=oak-b\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SystemExit, "duplicate"):
                configured_mxid(config)

    def test_deb_declares_camera_and_webrtc_runtime_dependencies(self) -> None:
        build_script = (ROOT / "packaging/deb/build-deb.sh").read_text(
            encoding="utf-8"
        )
        for package in (
            "libusb-1.0-0",
            "udev",
            "v4l-utils",
            "gstreamer1.0-nice",
            "gir1.2-gst-plugins-bad-1.0",
        ):
            self.assertIn(package, build_script)

    def test_depthai_runtime_is_binary_only_version_and_hash_pinned(self) -> None:
        requirements = (
            ROOT / "packaging/deb/requirements-depthai-arm64.txt"
        ).read_text(encoding="utf-8")
        build_script = (ROOT / "packaging/deb/build-deb.sh").read_text(
            encoding="utf-8"
        )
        postinst = (ROOT / "packaging/deb/postinst").read_text(encoding="utf-8")
        bundle = (ROOT / "packaging/release/build-agent-bundle.sh").read_text(
            encoding="utf-8"
        )

        self.assertEqual(requirements.count("depthai==2.32.0.0"), 1)
        self.assertIn(
            "b3192ffff904482254def4cd2b9aac0c4d082a0787303bdc980768da4368331c",
            requirements,
        )
        self.assertIn("requirements-depthai-arm64.txt", build_script)
        for evidence in (
            'DEPTHAI_VERSION="2.32.0.0"',
            'dpkg --print-architecture)" != "arm64"',
            'version("depthai")',
            "import depthai",
            'getattr(depthai, "__version__"',
        ):
            self.assertIn(evidence, postinst)
        for evidence in ("--only-binary=:all:", "--require-hashes", "--no-deps"):
            self.assertIn(evidence, bundle)

    def test_oak_udev_rule_is_packaged_and_least_privilege(self) -> None:
        rule_path = ROOT / "packaging/udev/80-movidius.rules"
        rule = rule_path.read_text(encoding="utf-8")
        active_rule = "\n".join(
            line for line in rule.splitlines() if not line.lstrip().startswith("#")
        )
        build_script = (ROOT / "packaging/deb/build-deb.sh").read_text(
            encoding="utf-8"
        )
        postinst = (ROOT / "packaging/deb/postinst").read_text(encoding="utf-8")
        postrm = (ROOT / "packaging/deb/postrm").read_text(encoding="utf-8")

        self.assertIn('ENV{DEVTYPE}=="usb_device"', active_rule)
        self.assertIn('ATTR{idVendor}=="03e7"', active_rule)
        self.assertIn('MODE="0660"', active_rule)
        self.assertIn('GROUP="nuvion"', active_rule)
        self.assertNotIn('MODE="0666"', active_rule)
        self.assertIn("/usr/lib/udev/rules.d/80-movidius.rules", build_script)
        self.assertIn("udevadm control --reload-rules", postinst)
        self.assertIn("--attr-match=idVendor=03e7", postinst)
        self.assertIn("packaging/deb/postrm", build_script)
        self.assertIn("remove|purge", postrm)
        self.assertIn("udevadm control --reload-rules", postrm)
        self.assertIn("--attr-match=idVendor=03e7", postrm)

    def test_postinst_has_fail_closed_bounded_install_modes(self) -> None:
        postinst = (ROOT / "packaging/deb/postinst").read_text(encoding="utf-8")
        self.assertIn('NUVION_INSTALL_PROFILE:-base', postinst)
        self.assertIn('NUVION_INSTALL_AUTOSTART:-true', postinst)
        self.assertIn('full|runtime)', postinst)
        self.assertIn("requires its own hash-locked immutable bundle", postinst)
        self.assertIn('BOOTSTRAP_ROOT="$INSTALL_ROOT/bootstrap"', postinst)
        self.assertNotIn("venv --clear", postinst)
        self.assertNotIn("pip install", postinst)
        self.assertIn("Unsupported NUVION_INSTALL_PROFILE", postinst)
        self.assertIn("systemctl disable --now nuv-agent.service", postinst)

    def test_iq9075_installer_is_board_bound_and_leaves_service_disabled(self) -> None:
        installer = (ROOT / "packaging/dev/install-iq9075.sh").read_text(
            encoding="utf-8"
        )
        for evidence in ("qcs9075", "iq-9075", "dpkg --print-architecture"):
            self.assertIn(evidence, installer)
        self.assertIn('VERSION_ID:-}" = "24.04"', installer)
        self.assertIn('python_version" = "3.12"', installer)
        for safe_value in (
            '"NUVION_VIDEO_SOURCE": "oak" if camera_mode == "oak" else "auto"',
            '"NUVION_GST_SOURCE": ""',
            '"NUVION_DEMO_MODE": "false"',
            '"NUVION_ZSAD_BACKEND": "none"',
            '"NUVION_RUNTIME_BOOTSTRAP_ENABLED": "false"',
            '"NUVION_FLEET_COMMAND_ENABLED": "false"',
            '"NUVION_CAMERA_PREFERENCE": "usb"',
        ):
            self.assertIn(safe_value, installer)
        self.assertIn("systemctl disable --now nuv-agent.service", installer)
        self.assertIn("--no-install-recommends", installer)
        self.assertIn("--reinstall", installer)
        self.assertIn("apt-get clean", installer)
        self.assertIn('camera_mode="oak"', installer)
        self.assertIn("--camera must be oak or uvc", installer)
        self.assertIn("preserve_if_present", installer)
        self.assertIn("--expected-version", installer)
        self.assertIn("--expected-sha256", installer)
        self.assertIn('[[ "$expected_sha256" =~ ^[0-9a-f]{64}$ ]]', installer)
        self.assertIn('dpkg-deb -f "$deb_path" Version', installer)
        self.assertIn('sha256sum "$deb_path"', installer)
        self.assertIn('package version mismatch', installer)
        self.assertIn('package SHA-256 mismatch', installer)
        self.assertIn('package must not be a symlink', installer)

    def test_iq9075_camera_config_update_is_idempotent(self) -> None:
        installer = (ROOT / "packaging/dev/install-iq9075.sh").read_text(
            encoding="utf-8"
        )
        marker = 'sudo python3 - "$CONFIG_PATH" "$camera_mode" <<\'PY\'\n'
        updater = installer.split(marker, 1)[1].split("\nPY\n", 1)[0]
        updater = updater.replace(
            "os.chown(temporary, 0, path.stat().st_gid)",
            "os.chown(temporary, os.getuid(), path.stat().st_gid)",
        )

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "agent.env"
            config_path.write_text(
                "NUVION_CONFIG_SCHEMA_VERSION=12\n"
                "NUVION_VIDEO_SOURCE=auto\n"
                "NUVION_GST_SOURCE=videotestsrc pattern=smpte\n"
                "NUVION_DEMO_MODE=true\n"
                "NUVION_DEPTHAI_DEVICE_ID=existing-mxid\n"
                "NUVION_DEPTHAI_STARTUP_TIMEOUT_SEC=19\n",
                encoding="utf-8",
            )
            for _ in range(2):
                subprocess.run(
                    [sys.executable, "-", str(config_path), "oak"],
                    input=updater,
                    check=True,
                    capture_output=True,
                    text=True,
                )

            oak_lines = config_path.read_text(encoding="utf-8").splitlines()
            for key in (
                "NUVION_VIDEO_SOURCE",
                "NUVION_GST_SOURCE",
                "NUVION_DEMO_MODE",
                "NUVION_DEPTHAI_DEVICE_ID",
                "NUVION_DEPTHAI_STARTUP_TIMEOUT_SEC",
                "NUVION_DEPTHAI_READ_TIMEOUT_SEC",
                "NUVION_DEPTHAI_MAX_CONSECUTIVE_TIMEOUTS",
            ):
                self.assertEqual(
                    sum(line.startswith(f"{key}=") for line in oak_lines),
                    1,
                )
            self.assertIn("NUVION_VIDEO_SOURCE=oak", oak_lines)
            self.assertIn("NUVION_GST_SOURCE=", oak_lines)
            self.assertIn("NUVION_DEMO_MODE=false", oak_lines)
            self.assertIn("NUVION_DEPTHAI_DEVICE_ID=existing-mxid", oak_lines)
            self.assertIn("NUVION_DEPTHAI_STARTUP_TIMEOUT_SEC=19", oak_lines)

            subprocess.run(
                [sys.executable, "-", str(config_path), "uvc"],
                input=updater,
                check=True,
                capture_output=True,
                text=True,
            )
            uvc_lines = config_path.read_text(encoding="utf-8").splitlines()
            self.assertIn("NUVION_VIDEO_SOURCE=auto", uvc_lines)
            self.assertIn("NUVION_CAMERA_PREFERENCE=usb", uvc_lines)
            self.assertIn("NUVION_DEPTHAI_DEVICE_ID=existing-mxid", uvc_lines)

    def test_depthai_config_defaults_are_documented_without_replacing_uvc(self) -> None:
        template = (ROOT / "nuvion_app/config_template.env").read_text(
            encoding="utf-8"
        )
        self.assertIn("NUVION_CONFIG_SCHEMA_VERSION=12", template)
        self.assertIn("NUVION_VIDEO_SOURCE=auto", template)
        self.assertIn("NUVION_CAMERA_PREFERENCE=auto", template)
        for setting in (
            "NUVION_DEPTHAI_DEVICE_ID=",
            "NUVION_DEPTHAI_STARTUP_TIMEOUT_SEC=15",
            "NUVION_DEPTHAI_READ_TIMEOUT_SEC=2",
            "NUVION_DEPTHAI_MAX_CONSECUTIVE_TIMEOUTS=3",
        ):
            self.assertIn(setting, template)

    def test_hardware_e2e_covers_bounded_non_root_oak_and_stable_uvc(self) -> None:
        e2e = (ROOT / "packaging/dev/test-iq9075.sh").read_text(encoding="utf-8")
        self.assertIn('camera_mode="oak"', e2e)
        self.assertIn("runuser -u nuvion", e2e)
        self.assertIn("/opt/nuv-agent/current/venv/bin/python", e2e)
        self.assertIn("NUVION_AGENT_PYTHON", e2e)
        self.assertIn("normalized root-owned executable", e2e)
        self.assertIn("--evidence-output", e2e)
        self.assertIn(
            "evidence mode requires the exact BOM-addressed candidate Python", e2e
        )
        runbook = (
            ROOT / "packaging/release/v0.1.121-release-runbook.md"
        ).read_text(encoding="utf-8")
        self.assertIn("env -u PYTHONPATH", runbook)
        self.assertGreaterEqual(e2e.count('-C "$probe_runtime_dir"'), 2)
        self.assertGreaterEqual(e2e.count('+=("-I")'), 2)
        self.assertIn("OAK evidence process imported outside candidate slot", e2e)
        self.assertIn('"runtimeIdentity"', e2e)
        self.assertIn("physical release test requires exactly one OAK MXID", e2e)
        self.assertIn("pre-release hardware evidence only", e2e)
        self.assertIn("not path.is_absolute()", e2e)
        self.assertIn("raw != normalized", e2e)
        self.assertIn("metadata.st_uid != 0", e2e)
        self.assertIn("metadata.st_mode & 0o022", e2e)
        self.assertIn("parent.is_relative_to(install_root)", e2e)
        self.assertIn("/usr/bin/python3 -I", e2e)
        self.assertIn('runuser -u nuvion --', e2e)
        self.assertIn('"G_DEBUG=fatal-criticals"', e2e)
        self.assertIn("mktemp -d /tmp/nuvion-iq9075-e2e.XXXXXX", e2e)
        self.assertIn('"HOME=$probe_runtime_dir/home"', e2e)
        self.assertIn('"XDG_CACHE_HOME=$probe_runtime_dir/cache"', e2e)
        self.assertIn('"XDG_CONFIG_HOME=$probe_runtime_dir/config"', e2e)
        self.assertIn('"XDG_RUNTIME_DIR=$probe_runtime_dir/runtime"', e2e)
        self.assertIn('expected_version = "2.32.0.0"', e2e)
        self.assertIn("timeout 720s", e2e)
        self.assertIn("DepthAIFrameSource", e2e)
        self.assertIn("DepthAIGStreamerBridge", e2e)
        self.assertIn('NUVION_IQ9075_OAK_SOAK_SECONDS", "120"', e2e)
        self.assertIn('read_proc_status_kib("RssAnon")', e2e)
        self.assertIn("appsrc buffer bound exceeded", e2e)
        self.assertIn("appsrc byte bound exceeded", e2e)
        self.assertIn("post-rejection anonymous RSS slope exceeded bound", e2e)
        self.assertIn("post-rejection anonymous RSS range exceeded bound", e2e)
        self.assertIn("rss_slope_mib_per_min", e2e)
        self.assertIn("len(samples) < 18", e2e)
        self.assertIn("raw_fps < MIN_RAW_FPS", e2e)
        self.assertIn("minimum_raw_samples = int(0.9 * FPS * soak_seconds)", e2e)
        self.assertIn('structure.get_name() != "video/x-raw"', e2e)
        self.assertIn("raw sample is missing a PTS", e2e)
        self.assertIn("build_bounded_live_queue", e2e)
        self.assertIn("build_uplink_pipeline", e2e)
        self.assertIn("clip_enabled=True", e2e)
        self.assertIn("splitmux fragment progress fell below bound", e2e)
        self.assertIn("newest_segment_age > 2 * SEGMENT_SECONDS + 5", e2e)
        self.assertIn("len(segments) > MAX_SEGMENTS", e2e)
        self.assertIn("offer_answer_timeout_sec=3.0", e2e)
        self.assertIn('"sessionId": "unanswered-offer"', e2e)
        self.assertIn("unanswered WebRTC watchdog branch teardown", e2e)
        self.assertIn("unanswered offer watchdog did not emit exact terminal STOP", e2e)
        self.assertNotIn("controller.reject_signaling(", e2e)
        self.assertIn('"profile-level-id=42e01f"', e2e)
        self.assertIn("old_queue.get_parent() is not None", e2e)
        self.assertIn("old_webrtc.get_state(0)[1] != Gst.State.NULL", e2e)
        self.assertIn("request_pad_count(uplink_tee) != 0", e2e)
        self.assertIn("controller._branch is not None", e2e)
        self.assertIn("bridge.stats_snapshot()", e2e)
        self.assertIn("Gst.MessageType.ERROR", e2e)
        self.assertIn('oak_status" -eq 3', e2e)
        self.assertIn('NUVION_GST_SOURCE must be empty', e2e)
        self.assertIn('NUVION_DEMO_MODE must be false', e2e)
        self.assertIn('re.compile(r"^[12]-1(?:\\.[1-9][0-9]*)+$")', e2e)
        self.assertIn('oak_products = {"2485": "bootloader", "f63b": "runtime"}', e2e)
        self.assertIn("require_runtime_oak(startup_timeout)", e2e)
        self.assertIn("runtime OAK-D Lite must enumerate on USB3", e2e)
        self.assertIn("len(oak_usb_paths) != 1", e2e)
        self.assertIn('Path("/sys/bus/usb/drivers/usb")', e2e)
        self.assertNotIn('Path("/sys/bus/usb/devices/2-1")', e2e)
        self.assertNotIn("DevNum", e2e)
        self.assertIn("driver_path != expected_driver", e2e)
        self.assertIn("speed_mbps < 5000.0", e2e)
        self.assertIn("OAK USB device is present but DepthAI could not enumerate it", e2e)
        self.assertIn("no OAK-D device detected", e2e)
        self.assertIn("/dev/v4l/by-id", e2e)
        self.assertIn('driver" = "uvcvideo"', e2e)
        self.assertIn("timeout 20s gst-launch-1.0", e2e)
        self.assertIn("G_DEBUG=fatal-criticals", e2e)
        self.assertIn("WebRTCUplinkController", e2e)
        self.assertIn("controller.on_signaling_reset()", e2e)
        self.assertIn("partial teardown did not recover idempotently", e2e)
        self.assertIn('"webrtc_uplink_session_5"', e2e)
        self.assertIn("Gst.MessageType.ERROR | Gst.MessageType.WARNING", e2e)
        self.assertNotIn("NUVION_DEVICE_PASSWORD", e2e)
        self.assertNotIn("curl ", e2e)

    def test_packaging_shell_scripts_parse(self) -> None:
        scripts = sorted((ROOT / "packaging").rglob("*.sh"))
        self.assertTrue(scripts)
        for script in scripts:
            with self.subTest(script=script.relative_to(ROOT)):
                subprocess.run(
                    ["bash", "-n", str(script)],
                    check=True,
                    capture_output=True,
                    text=True,
                )

    def test_provisioner_validates_scope_and_never_prints_credentials(self) -> None:
        provisioner = (
            ROOT / "packaging/dev/provision-iq9075.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("deviceUsername does not match spaceId", provisioner)
        self.assertIn("credential input must not be accessible", provisioner)
        self.assertIn('"NUVION_FLEET_COMMAND_ENABLED": "false"', provisioner)
        self.assertIn('"NUVION_GST_SOURCE": ""', provisioner)
        self.assertIn('"NUVION_DEMO_MODE": "false"', provisioner)
        self.assertIn("--synthetic-camera", provisioner)
        self.assertIn("--consume", provisioner)
        self.assertIn("credential_path.lstat()", provisioner)
        self.assertIn('getattr(os, "O_NOFOLLOW", 0)', provisioner)
        self.assertIn("opened.st_dev, opened.st_ino", provisioner)
        self.assertIn("credential_path.unlink()", provisioner)
        self.assertNotIn('credentials_path="$(realpath', provisioner)
        self.assertNotIn('echo "$password"', provisioner)

    def test_provisioner_clears_stale_synthetic_source_in_physical_mode(self) -> None:
        provisioner = (
            ROOT / "packaging/dev/provision-iq9075.sh"
        ).read_text(encoding="utf-8")
        marker = (
            'python3 - "$credentials_path" "$CONFIG_PATH" '
            '"$synthetic_camera" "$consume" <<\'PY\'\n'
        )
        updater = provisioner.split(marker, 1)[1].split("\nPY\n", 1)[0]
        updater = updater.replace(
            "os.chown(temporary, 0, config_path.stat().st_gid)",
            "os.chown(temporary, os.getuid(), config_path.stat().st_gid)",
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            credentials_path = root / "credentials.json"
            config_path = root / "agent.env"
            credentials_path.write_text(
                json.dumps(
                    {
                        "spaceId": 33,
                        "deviceUsername": "sp-33-nuvion-test",
                        "devicePassword": "long-test-password",
                    }
                ),
                encoding="utf-8",
            )
            os.chmod(credentials_path, 0o600)
            config_path.write_text(
                "NUVION_GST_SOURCE=videotestsrc pattern=smpte\n"
                "NUVION_DEMO_MODE=true\n",
                encoding="utf-8",
            )

            for synthetic in ("true", "false"):
                subprocess.run(
                    [
                        sys.executable,
                        "-",
                        str(credentials_path),
                        str(config_path),
                        synthetic,
                        "false",
                    ],
                    input=updater,
                    check=True,
                    capture_output=True,
                    text=True,
                )

            values = dict(
                line.split("=", 1)
                for line in config_path.read_text(encoding="utf-8").splitlines()
                if "=" in line
            )
            self.assertEqual(values["NUVION_GST_SOURCE"], "")
            self.assertEqual(values["NUVION_DEMO_MODE"], "false")

            symlink_path = root / "credentials-link.json"
            symlink_path.symlink_to(credentials_path)
            rejected = subprocess.run(
                [
                    sys.executable,
                    "-",
                    str(symlink_path),
                    str(config_path),
                    "false",
                    "false",
                ],
                input=updater,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("regular file", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
