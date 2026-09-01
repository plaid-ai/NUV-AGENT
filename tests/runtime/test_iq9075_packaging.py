from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class Iq9075PackagingTest(unittest.TestCase):
    def test_deb_declares_camera_and_webrtc_runtime_dependencies(self) -> None:
        build_script = (ROOT / "packaging/deb/build-deb.sh").read_text(
            encoding="utf-8"
        )
        for package in (
            "v4l-utils",
            "gstreamer1.0-nice",
            "gir1.2-gst-plugins-bad-1.0",
        ):
            self.assertIn(package, build_script)

    def test_postinst_has_fail_closed_bounded_install_modes(self) -> None:
        postinst = (ROOT / "packaging/deb/postinst").read_text(encoding="utf-8")
        self.assertIn('NUVION_INSTALL_PROFILE:-full', postinst)
        self.assertIn('NUVION_INSTALL_AUTOSTART:-true', postinst)
        self.assertIn('package_spec="/opt/nuv-agent/src"', postinst)
        self.assertIn("Unsupported NUVION_INSTALL_PROFILE", postinst)
        self.assertIn("systemctl disable nuv-agent.service", postinst)

    def test_iq9075_installer_is_board_bound_and_leaves_service_disabled(self) -> None:
        installer = (ROOT / "packaging/dev/install-iq9075.sh").read_text(
            encoding="utf-8"
        )
        for evidence in ("qcs9075", "iq-9075", "dpkg --print-architecture"):
            self.assertIn(evidence, installer)
        for safe_value in (
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

    def test_hardware_e2e_is_local_bounded_and_requires_stable_uvc_path(self) -> None:
        e2e = (ROOT / "packaging/dev/test-iq9075.sh").read_text(encoding="utf-8")
        self.assertIn("/dev/v4l/by-id", e2e)
        self.assertIn('driver" = "uvcvideo"', e2e)
        self.assertIn("timeout 20s gst-launch-1.0", e2e)
        self.assertNotIn("NUVION_DEVICE_PASSWORD", e2e)
        self.assertNotIn("curl ", e2e)

    def test_provisioner_validates_scope_and_never_prints_credentials(self) -> None:
        provisioner = (
            ROOT / "packaging/dev/provision-iq9075.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("deviceUsername does not match spaceId", provisioner)
        self.assertIn("credential input must not be accessible", provisioner)
        self.assertIn('"NUVION_FLEET_COMMAND_ENABLED": "false"', provisioner)
        self.assertIn("--synthetic-camera", provisioner)
        self.assertIn("--consume", provisioner)
        self.assertNotIn('echo "$password"', provisioner)


if __name__ == "__main__":
    unittest.main()
