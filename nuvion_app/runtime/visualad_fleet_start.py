"""Fleet-only entrypoint: recover pending settings before any pipeline import.

Do not use this entrypoint as an import smoke test: it records a real candidate
boot attempt. The existing non-Fleet experimental launcher remains unchanged.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from nuvion_app.runtime.settings_overlay import resolve_settings_state_dir

_BOOT_CONTEXT: tuple[str, ...] | None = None
_CONTEXT_KEYS = (
    "NUV_AGENT_CONFIG",
    "NUVION_COMMAND_INBOX_PATH",
    "NUVION_SETTINGS_STATE_DIR",
    "NUVION_VISUALAD_HTP_FLEET_STORE",
)


def _context(values) -> tuple[str, ...]:
    result = []
    for key in _CONTEXT_KEYS:
        raw = values.get(key, "")
        if not raw or not Path(raw).is_absolute():
            raise ValueError(f"VisualAD Fleet startup requires an absolute {key}")
        result.append(str(Path(raw).resolve()))
    if str(resolve_settings_state_dir(values).resolve()) != result[2]:
        raise ValueError("VisualAD Fleet settings state differs from the boot guard")
    return tuple(result)


def require_fleet_boot_guard(values) -> None:
    if _BOOT_CONTEXT is None or _BOOT_CONTEXT != _context(values):
        raise RuntimeError("VisualAD Fleet requires its settings boot guard entrypoint")


def prepare_fleet_startup() -> str:
    global _BOOT_CONTEXT
    from nuvion_app import config
    from nuvion_app.runtime.settings_boot_guard import run_settings_boot_guard

    if config._LOADED or "nuvion_app.inference.pipeline" in sys.modules:
        raise RuntimeError(
            "Settings boot guard must precede config overlay and pipeline import"
        )
    if _BOOT_CONTEXT is not None:
        raise RuntimeError(
            "VisualAD Fleet boot guard must run exactly once per process"
        )
    path = config.resolve_config_path()
    # Read only the base settings here. Applying active.env before recovery can
    # import constants from a candidate that the guard is about to roll back.
    values = {
        **{
            key: value
            for key, value in config.dotenv_values(path).items()
            if value is not None
        },
        **os.environ,
        "NUV_AGENT_CONFIG": str(path),
    }
    if values.get("NUVION_ZSAD_BACKEND") != "visualad_htp":
        raise ValueError("VisualAD Fleet entrypoint supports only visualad_htp")
    context = _context(values)
    result = run_settings_boot_guard(values, base_config_path=path)
    config.load_env(str(path))
    if _context(os.environ) != context:
        raise RuntimeError(
            "VisualAD Fleet startup paths changed after settings recovery"
        )
    _BOOT_CONTEXT = context
    return result


def main() -> None:
    prepare_fleet_startup()
    from nuvion_app.inference.main import main as inference_main

    inference_main()


if __name__ == "__main__":
    # -m runs this file as __main__; keep the guard state in the same canonical
    # module that the detector imports when it validates startup completion.
    from nuvion_app.runtime.visualad_fleet_start import main as canonical_main

    canonical_main()
