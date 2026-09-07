from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from unittest import mock

from nuvion_app import config
from nuvion_app.inference.settings_reconciler import AtomicSettingsStore
from nuvion_app.runtime import visualad_fleet_start as startup
from nuvion_app.runtime.settings_overlay import apply_settings_overlay


class VisualADFleetStartupTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.path = self.root / "agent.env"
        self.path.write_text("NUVION_MODEL_POINTER=visualad/iq9075-htp-demo\n")
        self.values = {
            "NUV_AGENT_CONFIG": str(self.path),
            "NUVION_ZSAD_BACKEND": "visualad_htp",
            "NUVION_COMMAND_INBOX_PATH": str(self.root / "inbox.sqlite3"),
            "NUVION_SETTINGS_STATE_DIR": str(self.root / "settings"),
            "NUVION_VISUALAD_HTP_FLEET_STORE": str(self.root / "models"),
        }

    def test_module_entrypoint_shares_the_canonical_boot_context(self):
        # Stub external startup effects in the child interpreter only. The
        # actual -m entrypoint and canonical guard must share process state.
        (self.root / "sitecustomize.py").write_text(
            textwrap.dedent(
                """\
                import os
                import sys
                import types
                from pathlib import Path
                import nuvion_app

                calls = []
                config = types.ModuleType("nuvion_app.config")
                config._LOADED = False
                config.resolve_config_path = lambda: Path(os.environ["NUV_AGENT_CONFIG"])
                config.dotenv_values = lambda path: {}
                def load_env(path):
                    assert calls == ["guard"]
                    calls.append("load")
                    config._LOADED = True
                config.load_env = load_env
                sys.modules[config.__name__] = config
                nuvion_app.config = config

                guard = types.ModuleType("nuvion_app.runtime.settings_boot_guard")
                def run_settings_boot_guard(values, **kwargs):
                    assert not config._LOADED
                    calls.append("guard")
                    return "NO_PENDING_SETTINGS"
                guard.run_settings_boot_guard = run_settings_boot_guard
                sys.modules[guard.__name__] = guard

                inference = types.ModuleType("nuvion_app.inference")
                inference.__path__ = []
                sys.modules[inference.__name__] = inference
                nuvion_app.inference = inference
                main = types.ModuleType("nuvion_app.inference.main")
                def inference_main():
                    from nuvion_app.runtime.visualad_fleet_start import require_fleet_boot_guard
                    require_fleet_boot_guard(os.environ)
                    assert calls == ["guard", "load"]
                    assert "nuvion_app.inference.pipeline" not in sys.modules
                    print("CANONICAL_BOOT_CONTEXT_OK")
                main.main = inference_main
                sys.modules[main.__name__] = main
                """
            )
        )
        repository = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [sys.executable, "-m", "nuvion_app.runtime.visualad_fleet_start"],
            cwd=self.root,
            env={
                **self.values,
                "PYTHONPATH": os.pathsep.join((str(self.root), str(repository))),
                "PYTHONNOUSERSITE": "1",
            },
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "CANONICAL_BOOT_CONTEXT_OK")

    def test_guard_recovers_overlay_before_loading_constants(self):
        order = []
        state = self.root / "settings"
        state.mkdir()
        (state / "active.env").write_text(
            "NUVION_MODEL_DIGEST=sha256:" + "a" * 64 + "\n"
        )

        def guard(values, **_kwargs):
            self.assertNotIn("NUVION_MODEL_DIGEST", values)
            order.append("guard")
            (state / "active.env").write_text(
                "NUVION_MODEL_DIGEST=sha256:" + "b" * 64 + "\n"
            )
            return "LKG_RESTORED"

        def load(_path):
            order.append("load")
            apply_settings_overlay(os.environ)

        with (
            mock.patch.dict(os.environ, self.values, clear=True),
            mock.patch.object(config, "_LOADED", False),
            mock.patch.object(config, "load_env", side_effect=load),
            mock.patch.object(startup, "_BOOT_CONTEXT", None),
            mock.patch.dict(sys.modules),
            mock.patch(
                "nuvion_app.runtime.settings_boot_guard.run_settings_boot_guard",
                side_effect=guard,
            ),
        ):
            sys.modules.pop("nuvion_app.inference.pipeline", None)
            self.assertEqual(startup.prepare_fleet_startup(), "LKG_RESTORED")
            self.assertEqual(order, ["guard", "load"])
            self.assertEqual(os.environ["NUVION_MODEL_DIGEST"], "sha256:" + "b" * 64)
            startup.require_fleet_boot_guard(os.environ)
            changed = {
                **os.environ,
                "NUVION_COMMAND_INBOX_PATH": str(self.root / "different.sqlite3"),
            }
            with self.assertRaises(RuntimeError):
                startup.require_fleet_boot_guard(changed)
            with self.assertRaises(RuntimeError):
                startup.prepare_fleet_startup()

    def test_already_imported_pipeline_or_loaded_overlay_is_rejected(self):
        for loaded, imported in ((True, False), (False, True)):
            with (
                self.subTest(loaded=loaded, imported=imported),
                mock.patch.object(config, "_LOADED", loaded),
                mock.patch.dict(sys.modules),
                mock.patch.object(startup, "_BOOT_CONTEXT", None),
                self.assertRaises(RuntimeError),
            ):
                sys.modules.pop("nuvion_app.inference.pipeline", None)
                if imported:
                    sys.modules["nuvion_app.inference.pipeline"] = object()
                startup.prepare_fleet_startup()

    def test_missing_inbox_path_cannot_record_a_candidate_boot(self):
        values = {
            key: value
            for key, value in self.values.items()
            if key != "NUVION_COMMAND_INBOX_PATH"
        }
        with (
            mock.patch.dict(os.environ, values, clear=True),
            mock.patch.object(config, "_LOADED", False),
            mock.patch.object(startup, "_BOOT_CONTEXT", None),
            mock.patch.dict(sys.modules),
            mock.patch(
                "nuvion_app.runtime.settings_boot_guard.run_settings_boot_guard"
            ) as guard,
        ):
            sys.modules.pop("nuvion_app.inference.pipeline", None)
            with self.assertRaises(ValueError):
                startup.prepare_fleet_startup()
            guard.assert_not_called()

    def test_real_boot_guard_and_overlay_override_then_restore_bootstrap_lkg(self):
        store = AtomicSettingsStore(self.path, self.root / "settings")
        pointer = "visualad/iq9075-htp-demo"
        bootstrap_digest = "sha256:" + "b" * 64
        candidate_digest = "sha256:" + "c" * 64
        store.stage_and_activate(
            command=types.SimpleNamespace(
                command_id="test-only-command",
                payload={
                    "configVersion": 1,
                    "activation": "RESTART",
                    "model": {"pointer": pointer, "digest": candidate_digest},
                },
            ),
            process_instance_id="previous-process",
            settings_digest="sha256:" + "d" * 64,
        )
        values = {
            **self.values,
            "NUVION_MODEL_POINTER": pointer,
            "NUVION_MODEL_DIGEST": bootstrap_digest,
        }
        for expected_status, expected_digest in (
            ("CANDIDATE_BOOT_ATTEMPT", candidate_digest),
            ("LKG_RESTORED", bootstrap_digest),
        ):
            with (
                self.subTest(status=expected_status),
                mock.patch.dict(os.environ, values, clear=True),
                mock.patch.object(config, "_LOADED", False),
                mock.patch.object(config, "_LOADED_PATH", None),
                mock.patch.object(startup, "_BOOT_CONTEXT", None),
                mock.patch.dict(sys.modules),
            ):
                sys.modules.pop("nuvion_app.inference.pipeline", None)
                self.assertEqual(startup.prepare_fleet_startup(), expected_status)
                self.assertEqual(os.environ["NUVION_MODEL_POINTER"], pointer)
                self.assertEqual(os.environ["NUVION_MODEL_DIGEST"], expected_digest)
                startup.require_fleet_boot_guard(os.environ)
        self.assertEqual(store.marker()["phase"], "ROLLBACK_STAGED")


if __name__ == "__main__":
    unittest.main()
