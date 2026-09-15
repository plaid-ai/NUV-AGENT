from __future__ import annotations

import unittest

import numpy as np

from nuvion_app.inference.camera_control import (
    CAMERA_PROFILE_B0272,
    CAMERA_PROFILE_B0273,
    FOCUS_STATE_ERROR,
    FOCUS_STATE_LOCKED,
    FOCUS_STATE_MANUAL,
    FOCUS_STATE_UNSUPPORTED,
    PLATFORM_JETSON,
    PLATFORM_RASPBERRY_PI,
    CameraControlConfig,
    CameraController,
    CameraHardwareProbe,
    camera_control_config_from_env,
    resolve_camera_profile,
    validate_camera_contract,
)


class _FakeGstSource:
    def __init__(
        self, properties: set[str], values: dict[str, object] | None = None
    ) -> None:
        self.properties = properties
        self.values = dict(values or {})

    def find_property(self, name: str):
        return object() if name in self.properties else None

    def set_property(self, name: str, value: object) -> None:
        if name not in self.properties:
            raise ValueError(name)
        self.values[name] = value

    def get_property(self, name: str):
        return self.values[name]


def _config(
    profile: str,
    platform_kind: str,
    *,
    focus_mode: str = "startup-lock",
    required: bool = True,
    manual_position: float | None = None,
    focus_settle_seconds: float = 0.001,
    i2c_bus: int | None = None,
) -> CameraControlConfig:
    return CameraControlConfig(
        profile=profile,
        platform_kind=platform_kind,
        focus_mode=focus_mode,
        focus_required=required,
        focus_settle_seconds=focus_settle_seconds,
        focus_step_frames=2,
        manual_lens_position=manual_position,
        i2c_bus=i2c_bus,
    )


class CameraControlTest(unittest.TestCase):
    def test_auto_profile_resolves_only_supported_product_platforms(self) -> None:
        self.assertEqual(
            resolve_camera_profile(
                "auto", video_source="rpi", platform_kind=PLATFORM_RASPBERRY_PI
            ),
            CAMERA_PROFILE_B0272,
        )
        self.assertEqual(
            resolve_camera_profile(
                "auto", video_source="auto", platform_kind=PLATFORM_JETSON
            ),
            CAMERA_PROFILE_B0273,
        )
        self.assertEqual(
            camera_control_config_from_env(
                "jetson",
                environ={"NUVION_CAMERA_FOCUS_MODE": "startup_lock"},
                platform_kind=PLATFORM_JETSON,
            ).focus_mode,
            "startup-lock",
        )

    def test_b0272_starts_autofocus_then_locks_reported_position(self) -> None:
        source = _FakeGstSource(
            {"af-mode", "af-state", "lens-position"},
            {"af-state": 2, "lens-position": 4.25},
        )
        controller = CameraController(
            _config(CAMERA_PROFILE_B0272, PLATFORM_RASPBERRY_PI),
            hardware_probe=CameraHardwareProbe("imx477", ("imx477",)),
        )

        controller.configure(source)
        self.assertEqual(source.values["af-mode"], 2)
        controller.start()
        assert controller._worker is not None
        controller._worker.join(timeout=1)

        snapshot = controller.snapshot()
        self.assertEqual(snapshot["focusState"], FOCUS_STATE_LOCKED)
        self.assertFalse(controller.blocks_runtime_health())
        self.assertEqual(snapshot["lensPosition"], 4.25)
        self.assertEqual(source.values["af-mode"], 0)
        self.assertIn("camera.autofocus.startup_lock", controller.capabilities())

    def test_b0272_legacy_gstreamer_property_supports_continuous_only(self) -> None:
        source = _FakeGstSource({"auto-focus-mode"})
        controller = CameraController(
            _config(
                CAMERA_PROFILE_B0272,
                PLATFORM_RASPBERRY_PI,
                focus_mode="continuous",
            ),
            hardware_probe=CameraHardwareProbe("imx477", ("imx477",)),
        )

        controller.configure(source)

        self.assertEqual(source.values["auto-focus-mode"], 2)
        self.assertEqual(controller.snapshot()["focusState"], "CONTINUOUS")

    def test_b0272_required_startup_lock_rejects_legacy_property(self) -> None:
        source = _FakeGstSource({"auto-focus-mode"})
        controller = CameraController(
            _config(CAMERA_PROFILE_B0272, PLATFORM_RASPBERRY_PI),
            hardware_probe=CameraHardwareProbe("imx477", ("imx477",)),
        )

        with self.assertRaisesRegex(RuntimeError, "runtime startup focus lock"):
            controller.configure(source)

    def test_b0272_manual_focus_requires_and_applies_lens_position(self) -> None:
        source = _FakeGstSource({"af-mode", "lens-position"})
        controller = CameraController(
            _config(
                CAMERA_PROFILE_B0272,
                PLATFORM_RASPBERRY_PI,
                focus_mode="manual",
                manual_position=3.5,
            ),
            hardware_probe=CameraHardwareProbe("imx477", ("imx477",)),
        )

        controller.configure(source)

        self.assertEqual(source.values["af-mode"], 0)
        self.assertEqual(source.values["lens-position"], 3.5)
        self.assertEqual(controller.snapshot()["focusState"], FOCUS_STATE_MANUAL)

    def test_b0273_scores_live_frames_and_locks_best_i2c_position(self) -> None:
        writes: list[tuple[int, int]] = []
        controller = CameraController(
            _config(
                CAMERA_PROFILE_B0273,
                PLATFORM_JETSON,
                focus_settle_seconds=60.0,
                i2c_bus=9,
            ),
            hardware_probe=CameraHardwareProbe("imx477", ("vi-output, imx477",)),
            lens_writer=lambda bus, position: writes.append((bus, position)),
        )

        controller.configure(None)
        controller.start()

        grid = np.indices((32, 32)).sum(axis=0) % 2
        for _scan_step in range(100):
            if controller.snapshot()["focusState"] == FOCUS_STATE_LOCKED:
                break
            current_position = writes[-1][1]
            amplitude = max(10, 255 - abs(current_position - 400) // 2)
            frame = np.repeat(
                (grid * amplitude).astype(np.uint8)[:, :, None], 3, axis=2
            )
            for _ in range(3):
                controller.observe_frame(frame)

        snapshot = controller.snapshot()
        self.assertEqual(snapshot["focusState"], FOCUS_STATE_LOCKED)
        self.assertEqual(snapshot["lensPosition"], 400.0)
        self.assertEqual(snapshot["controlDevice"], "/dev/i2c-9")
        self.assertEqual(writes[0], (9, 0))
        self.assertEqual(writes[-1], (9, 400))

    def test_optional_focus_reports_unsupported_without_claiming_capability(
        self,
    ) -> None:
        source = _FakeGstSource(set())
        controller = CameraController(
            _config(
                CAMERA_PROFILE_B0272,
                PLATFORM_RASPBERRY_PI,
                required=False,
            ),
            hardware_probe=CameraHardwareProbe("imx477", ("imx477",)),
        )

        controller.configure(source)

        self.assertEqual(controller.snapshot()["focusState"], FOCUS_STATE_UNSUPPORTED)
        self.assertEqual(controller.capabilities(), frozenset())

    def test_required_focus_fails_when_control_is_missing(self) -> None:
        source = _FakeGstSource(set())
        controller = CameraController(
            _config(CAMERA_PROFILE_B0272, PLATFORM_RASPBERRY_PI),
            hardware_probe=CameraHardwareProbe("imx477", ("imx477",)),
        )

        with self.assertRaisesRegex(RuntimeError, "autofocus control"):
            controller.configure(source)
        self.assertEqual(controller.snapshot()["focusState"], FOCUS_STATE_ERROR)

    def test_required_b0273_focus_fails_without_explicit_i2c_bus(self) -> None:
        controller = CameraController(
            _config(CAMERA_PROFILE_B0273, PLATFORM_JETSON),
            hardware_probe=CameraHardwareProbe("imx477", ("imx477",)),
            lens_writer=lambda _bus, _position: None,
        )

        with self.assertRaisesRegex(RuntimeError, "I2C bus"):
            controller.configure(None)
        self.assertEqual(controller.snapshot()["focusState"], FOCUS_STATE_ERROR)

    def test_required_async_focus_failure_marks_runtime_unhealthy(self) -> None:
        failures: list[str] = []
        source = _FakeGstSource(
            {"af-mode", "af-state"},
            {"af-state": 3},
        )
        controller = CameraController(
            _config(CAMERA_PROFILE_B0272, PLATFORM_RASPBERRY_PI),
            hardware_probe=CameraHardwareProbe("imx477", ("imx477",)),
            on_failure=failures.append,
        )

        controller.configure(source)
        controller.start()
        assert controller._worker is not None
        controller._worker.join(timeout=1)

        self.assertEqual(controller.snapshot()["focusState"], FOCUS_STATE_ERROR)
        self.assertEqual(len(failures), 1)
        self.assertIn("autofocus reported failure", failures[0])
        self.assertTrue(controller.blocks_runtime_health())

    def test_required_b0272_focus_fails_when_lock_result_is_unverifiable(self) -> None:
        failures: list[str] = []
        source = _FakeGstSource({"af-mode"})
        controller = CameraController(
            _config(CAMERA_PROFILE_B0272, PLATFORM_RASPBERRY_PI),
            hardware_probe=CameraHardwareProbe("imx477", ("imx477",)),
            on_failure=failures.append,
        )

        controller.configure(source)
        controller.start()
        assert controller._worker is not None
        controller._worker.join(timeout=1)

        self.assertEqual(controller.snapshot()["focusState"], FOCUS_STATE_ERROR)
        self.assertIn("could not be verified", failures[0])
        self.assertTrue(controller.blocks_runtime_health())

    def test_profile_platform_mismatch_fails_contract(self) -> None:
        status, detail = validate_camera_contract(
            _config(CAMERA_PROFILE_B0273, PLATFORM_RASPBERRY_PI),
            CameraHardwareProbe("imx477", ("imx477",)),
        )

        self.assertEqual(status, "fail")
        self.assertIn("requires jetson", detail)


if __name__ == "__main__":
    unittest.main()
