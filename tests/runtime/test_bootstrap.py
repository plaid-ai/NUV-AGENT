from __future__ import annotations

import os
import unittest
from unittest import mock

from nuvion_app.runtime import bootstrap
from nuvion_app.runtime.errors import BootstrapError


class BootstrapTest(unittest.TestCase):
    def test_disable_runtime_bootstrap(self) -> None:
        with mock.patch.dict(os.environ, {"NUVION_RUNTIME_BOOTSTRAP_ENABLED": "false"}, clear=False):
            self.assertTrue(bootstrap.ensure_ready(stage="run"))

    def test_degrade_backend_on_failure(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "NUVION_RUNTIME_BOOTSTRAP_ENABLED": "true",
                "NUVION_ZSAD_BACKEND": "triton",
                "NUVION_BOOTSTRAP_MAX_RETRIES": "1",
            },
            clear=False,
        ):
            with mock.patch.object(bootstrap, "ensure_model_ready", side_effect=BootstrapError("model_pull_failed", "boom", retryable=False)):
                ok = bootstrap.ensure_ready(stage="run")
            self.assertEqual(os.getenv("NUVION_ZSAD_BACKEND"), "none")
        self.assertFalse(ok)

    def test_face_tracking_triton_still_bootstraps_when_zsad_is_none(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "NUVION_RUNTIME_BOOTSTRAP_ENABLED": "true",
                "NUVION_ZSAD_BACKEND": "none",
                "NUVION_FACE_TRACKING_ENABLED": "true",
                "NUVION_FACE_TRACKING_BACKEND": "triton",
                "NUVION_BOOTSTRAP_MAX_RETRIES": "1",
            },
            clear=False,
        ):
            with mock.patch.object(bootstrap, "ensure_model_ready", return_value=mock.Mock()):
                with mock.patch.object(bootstrap, "ensure_triton_ready") as ensure_triton_ready:
                    ok = bootstrap.ensure_ready(stage="run")
        self.assertTrue(ok)
        ensure_triton_ready.assert_called_once()


if __name__ == "__main__":
    unittest.main()
