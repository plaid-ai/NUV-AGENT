from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nuvion_app.runtime.platform_identity import (
    IDENTITY_STATUS_DEV,
    IDENTITY_STATUS_MISMATCH,
    IDENTITY_STATUS_UNPROVISIONED,
    IDENTITY_STATUS_UNVERIFIED,
    IDENTITY_STATUS_VERIFIED,
    IQ9075_DEV,
    MACOS_DEV,
    NUVION,
    NUVION_PRO,
    NUVION_ULTRA,
    PROFILE_JETSON_ORIN_NX,
    PROFILE_IQ9075_DEV,
    PROFILE_MACOS_DEV,
    PROFILE_RPI5_DEEPX,
    PROFILE_VENTUNO_Q,
    PlatformProbe,
    _run_version,
    resolve_platform_identity,
)


class PlatformIdentityTest(unittest.TestCase):
    def test_version_probe_does_not_confuse_executable_suffix_with_runtime(
        self,
    ) -> None:
        completed = SimpleNamespace(
            stdout="gst-launch-1.0 version 1.28.2\nGStreamer 1.28.2\n",
            stderr="",
        )
        with (
            mock.patch(
                "nuvion_app.runtime.platform_identity.shutil.which",
                return_value="/opt/homebrew/bin/gst-launch-1.0",
            ),
            mock.patch(
                "nuvion_app.runtime.platform_identity.subprocess.run",
                return_value=completed,
            ),
        ):
            self.assertEqual(_run_version("gst-launch-1.0", "--version"), "1.28.2")

    def _resolve_declared(
        self,
        *,
        product_model: str,
        platform_profile: str,
        hardware_text: str,
        system: str = "Linux",
    ):
        with tempfile.TemporaryDirectory() as tmp:
            identity_path = Path(tmp) / "device-identity.json"
            identity_path.write_text(
                json.dumps(
                    {
                        "productModel": product_model,
                        "hardwareRevision": "REV-A",
                        "platformProfile": platform_profile,
                    }
                ),
                encoding="utf-8",
            )
            return resolve_platform_identity(
                environ={},
                identity_path=identity_path,
                identity_file_stat=SimpleNamespace(st_uid=0, st_mode=0o100644),
                probe=PlatformProbe(
                    system=system,
                    os_version="24.04",
                    kernel_version="6.8.0",
                    architecture="aarch64" if system == "Linux" else "arm64",
                    hardware_text=hardware_text,
                    accelerator_runtime="runtime-1",
                    gstreamer_version="1.24.2",
                ),
            )

    def test_canonical_product_profiles_are_verified_and_capable(self) -> None:
        cases = (
            (
                NUVION,
                PROFILE_RPI5_DEEPX,
                "Raspberry Pi 5 DEEPX DX-M1",
                "accelerator.deepx",
            ),
            (
                NUVION_PRO,
                PROFILE_VENTUNO_Q,
                "Arduino VENTUNO Q",
                "accelerator.ventuno_q",
            ),
            (
                NUVION_ULTRA,
                PROFILE_JETSON_ORIN_NX,
                "NVIDIA Jetson Orin NX",
                "accelerator.tensorrt",
            ),
            (
                IQ9075_DEV,
                PROFILE_IQ9075_DEV,
                "Qualcomm Technologies, Inc. Addons IQ 9075 EVK qcom,qcs9075",
                "dev.hardware",
            ),
            (MACOS_DEV, PROFILE_MACOS_DEV, "Apple Mac", "dev.simulation"),
        )

        for product, profile, hardware_text, expected_capability in cases:
            with self.subTest(product=product):
                identity = self._resolve_declared(
                    product_model=product,
                    platform_profile=profile,
                    hardware_text=hardware_text,
                    system="Darwin" if product == MACOS_DEV else "Linux",
                )
                self.assertEqual(
                    identity.identity_status,
                    IDENTITY_STATUS_DEV
                    if product in {MACOS_DEV, IQ9075_DEV}
                    else IDENTITY_STATUS_VERIFIED,
                )
                self.assertIn("command.config.apply", identity.capabilities)
                self.assertIn("command.agent.update", identity.capabilities)
                self.assertIn(expected_capability, identity.capabilities)
                if product == IQ9075_DEV:
                    self.assertIn("camera.usb", identity.capabilities)
                    self.assertIn("camera.depthai", identity.capabilities)
                    self.assertFalse(
                        any(
                            capability.startswith("accelerator.qnn")
                            for capability in identity.capabilities
                        )
                    )

    def test_unprovisioned_iq9075_is_detected_but_has_no_capabilities(self) -> None:
        identity = resolve_platform_identity(
            environ={},
            identity_path=Path("/definitely/missing/device-identity.json"),
            probe=PlatformProbe(
                system="Linux",
                os_version="24.04",
                kernel_version="6.8.0-1080-qcom",
                architecture="aarch64",
                hardware_text=(
                    "Qualcomm Technologies, Inc. Addons IQ 9075 EVK\n"
                    "qcom,qcs9075-addons-iq-9075-evk"
                ),
                accelerator_runtime="unknown",
                gstreamer_version="1.24.2",
            ),
        )

        self.assertEqual(identity.identity_status, IDENTITY_STATUS_UNPROVISIONED)
        self.assertEqual(identity.product_model, "UNKNOWN")
        self.assertEqual(identity.platform_profile, PROFILE_IQ9075_DEV)
        self.assertEqual(identity.observed_platform_profile, PROFILE_IQ9075_DEV)
        self.assertEqual(identity.capabilities, frozenset())
        self.assertEqual(identity.accelerator, "Qualcomm IQ-9075")

    def test_iq9075_identity_rejects_non_arm64_hardware(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity_path = Path(tmp) / "device-identity.json"
            identity_path.write_text(
                json.dumps(
                    {
                        "productModel": IQ9075_DEV,
                        "hardwareRevision": "QCS9075-EVK",
                        "platformProfile": PROFILE_IQ9075_DEV,
                    }
                ),
                encoding="utf-8",
            )
            identity = resolve_platform_identity(
                environ={},
                identity_path=identity_path,
                identity_file_stat=SimpleNamespace(st_uid=0, st_mode=0o100640),
                probe=PlatformProbe(
                    system="Linux",
                    os_version="24.04",
                    kernel_version="6.8.0",
                    architecture="x86_64",
                    hardware_text="qcom,qcs9075-addons-iq-9075-evk",
                    accelerator_runtime="unknown",
                    gstreamer_version="1.24.2",
                ),
            )

        self.assertEqual(identity.identity_status, IDENTITY_STATUS_UNVERIFIED)
        self.assertEqual(identity.capabilities, frozenset())

    def test_iq9075_environment_identity_cannot_bypass_secure_file(self) -> None:
        identity = resolve_platform_identity(
            environ={
                "NUVION_PRODUCT_MODEL": IQ9075_DEV,
                "NUVION_HARDWARE_REVISION": "QCS9075-EVK",
                "NUVION_PLATFORM_PROFILE": PROFILE_IQ9075_DEV,
            },
            identity_path=Path("/definitely/missing/device-identity.json"),
            probe=PlatformProbe(
                system="Linux",
                os_version="24.04",
                kernel_version="6.8.0-1080-qcom",
                architecture="aarch64",
                hardware_text="qcom,qcs9075-addons-iq-9075-evk",
                accelerator_runtime="unknown",
                gstreamer_version="1.24.2",
            ),
        )

        self.assertEqual(identity.identity_status, "INVALID")
        self.assertEqual(identity.product_model, "UNKNOWN")
        self.assertEqual(identity.capabilities, frozenset())
        self.assertIn("secure identity file", identity.identity_error or "")

    def test_declared_and_observed_profile_mismatch_fails_closed(self) -> None:
        identity = self._resolve_declared(
            product_model=NUVION_ULTRA,
            platform_profile=PROFILE_JETSON_ORIN_NX,
            hardware_text="Raspberry Pi 5 DEEPX DX-M1",
        )

        self.assertEqual(identity.identity_status, IDENTITY_STATUS_MISMATCH)
        self.assertEqual(identity.observed_platform_profile, PROFILE_RPI5_DEEPX)
        self.assertEqual(identity.capabilities, frozenset())

    def test_declared_rpi_without_observed_deepx_is_unverified(self) -> None:
        identity = self._resolve_declared(
            product_model=NUVION,
            platform_profile=PROFILE_RPI5_DEEPX,
            hardware_text="Raspberry Pi 5 Model B Rev 1.0",
        )

        self.assertEqual(identity.identity_status, IDENTITY_STATUS_UNVERIFIED)
        self.assertEqual(identity.capabilities, frozenset())
        self.assertEqual(identity.accelerator, "DEEPX unconfirmed")

    def test_unprovisioned_production_host_has_no_effect_capabilities(self) -> None:
        identity = resolve_platform_identity(
            environ={},
            identity_path=Path("/definitely/missing/device-identity.json"),
            probe=PlatformProbe(
                system="Linux",
                os_version="24.04",
                kernel_version="6.8.0",
                architecture="aarch64",
                hardware_text="Raspberry Pi 5 DEEPX DX-M1",
                accelerator_runtime="unknown",
                gstreamer_version="1.24.2",
            ),
        )

        self.assertEqual(identity.identity_status, IDENTITY_STATUS_UNPROVISIONED)
        self.assertEqual(identity.product_model, "UNKNOWN")
        self.assertEqual(identity.capabilities, frozenset())

    def test_unprovisioned_macos_is_an_explicit_dev_identity(self) -> None:
        identity = resolve_platform_identity(
            environ={},
            identity_path=Path("/definitely/missing/device-identity.json"),
            probe=PlatformProbe(
                system="Darwin",
                os_version="15.6",
                kernel_version="24.6.0",
                architecture="arm64",
                hardware_text="Apple MacBook Pro",
                accelerator_runtime="MPS",
                gstreamer_version="1.26.0",
            ),
        )

        self.assertEqual(identity.product_model, MACOS_DEV)
        self.assertEqual(identity.platform_profile, PROFILE_MACOS_DEV)
        self.assertEqual(identity.identity_status, IDENTITY_STATUS_DEV)
        self.assertIn("dev.simulation", identity.capabilities)

    def test_insecure_linux_identity_file_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity_path = Path(tmp) / "device-identity.json"
            identity_path.write_text(
                json.dumps(
                    {
                        "productModel": NUVION,
                        "hardwareRevision": "REV-A",
                        "platformProfile": PROFILE_RPI5_DEEPX,
                    }
                ),
                encoding="utf-8",
            )
            probe = PlatformProbe(
                system="Linux",
                os_version="24.04",
                kernel_version="6.8.0",
                architecture="aarch64",
                hardware_text="Raspberry Pi 5 DEEPX DX-M1",
                accelerator_runtime="runtime-1",
                gstreamer_version="1.24.2",
            )
            cases = (
                (SimpleNamespace(st_uid=1234, st_mode=0o100644), "root"),
                (SimpleNamespace(st_uid=0, st_mode=0o100664), "writable"),
            )
            for metadata, reason in cases:
                with self.subTest(reason=reason):
                    identity = resolve_platform_identity(
                        environ={},
                        identity_path=identity_path,
                        identity_file_stat=metadata,
                        probe=probe,
                    )
                    self.assertEqual(identity.identity_status, "INVALID")
                    self.assertEqual(identity.capabilities, frozenset())
                    self.assertIn(reason, identity.identity_error or "")

    def test_telemetry_separates_declared_and_observed_identity(self) -> None:
        identity = self._resolve_declared(
            product_model=NUVION,
            platform_profile=PROFILE_RPI5_DEEPX,
            hardware_text="Raspberry Pi 5 DEEPX DX-M1",
        )

        telemetry = identity.to_telemetry()

        self.assertEqual(telemetry["productModel"], NUVION)
        self.assertEqual(telemetry["platformProfile"], PROFILE_RPI5_DEEPX)
        self.assertEqual(telemetry["observedPlatformProfile"], PROFILE_RPI5_DEEPX)
        self.assertEqual(telemetry["identityStatus"], IDENTITY_STATUS_VERIFIED)
        self.assertEqual(telemetry["osName"], "Linux")
        self.assertEqual(telemetry["kernelVersion"], "6.8.0")
        self.assertEqual(telemetry["architecture"], "aarch64")
        self.assertEqual(telemetry["accelerator"], "DEEPX DX-M1")
        self.assertEqual(telemetry["acceleratorRuntime"], "runtime-1")
        self.assertEqual(telemetry["gstreamerVersion"], "1.24.2")


if __name__ == "__main__":
    unittest.main()
