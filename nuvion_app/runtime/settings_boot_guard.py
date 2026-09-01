from __future__ import annotations

import os
import sqlite3
import uuid
from collections.abc import Mapping
from pathlib import Path

from nuvion_app.config import resolve_config_path
from nuvion_app.inference.settings_reconciler import AtomicSettingsStore
from nuvion_app.runtime.settings_overlay import resolve_settings_state_dir


class SettingsBootGuardError(RuntimeError):
    pass


def _durable_job_phase(
    values: Mapping[str, str],
    command_id: str,
) -> str | None:
    raw_path = str(values.get("NUVION_COMMAND_INBOX_PATH") or "").strip()
    if not raw_path:
        return None
    path = Path(raw_path).expanduser().absolute()
    if not path.is_file() or path.is_symlink():
        return None
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=1.0)
        connection.row_factory = sqlite3.Row
        try:
            row = connection.execute(
                "SELECT phase FROM fleet_reconcile_job WHERE command_id = ?",
                (command_id,),
            ).fetchone()
        finally:
            connection.close()
    except sqlite3.Error:
        return None
    return str(row["phase"]) if row is not None else None


def run_settings_boot_guard(
    environ: Mapping[str, str] | None = None,
    *,
    base_config_path: str | Path | None = None,
) -> str:
    values = os.environ if environ is None else environ
    state_dir = resolve_settings_state_dir(values)
    store = AtomicSettingsStore(
        base_config_path or resolve_config_path(),
        state_dir,
    )
    marker = store.recover_prepared()
    if marker is None:
        return "NO_PENDING_SETTINGS"
    phase = str(marker.get("phase") or "")
    if phase == "ACTIVATED":
        command_id = str(marker.get("commandId") or "")
        if command_id and _durable_job_phase(values, command_id) == "SUPERSEDED":
            store.rollback(
                process_instance_id=f"boot-guard-{uuid.uuid4()}",
                restart_required=False,
            )
            store.update_marker(
                {"recoveryReason": "SUPERSEDED_BEFORE_BOOT"}
            )
            return "SUPERSEDED_LKG_RESTORED"
        attempts = int(marker.get("bootAttempts") or 0)
        if attempts >= 1:
            store.rollback(
                process_instance_id=f"boot-guard-{uuid.uuid4()}",
                restart_required=True,
            )
            store.update_marker(
                {
                    "rollbackBootAttempts": 1,
                    "recoveryReason": "CANDIDATE_BOOT_DID_NOT_COMMIT",
                }
            )
            return "LKG_RESTORED"
        store.update_marker({"bootAttempts": attempts + 1})
        return "CANDIDATE_BOOT_ATTEMPT"
    if phase == "ROLLBACK_STAGED":
        rollback_attempts = int(marker.get("rollbackBootAttempts") or 0)
        if rollback_attempts >= 1:
            raise SettingsBootGuardError(
                "LKG boot did not reach terminal reconciliation; operator action required"
            )
        store.update_marker({"rollbackBootAttempts": rollback_attempts + 1})
        return "LKG_BOOT_ATTEMPT"
    return phase or "NO_PENDING_SETTINGS"


def main() -> None:
    run_settings_boot_guard()


if __name__ == "__main__":
    main()
