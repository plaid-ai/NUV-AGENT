from __future__ import annotations

import argparse
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from nuvion_app.runtime.release_bom import (
    build_release_bom_payload,
    canonical_release_bom_json,
    verify_release_artifact,
    verify_release_bom,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a content-addressed NUVION release BOM"
    )
    parser.add_argument("--bom-id", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--component-sha", required=True)
    parser.add_argument("--config-schema", required=True)
    parser.add_argument("--updater-version", required=True)
    parser.add_argument("--platform-profile", action="append", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--artifact-kind", required=True)
    parser.add_argument("--built-at")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    built_at = args.built_at or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    artifact = Path(args.artifact).expanduser().resolve()
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output
    if output == artifact or (output.exists() and output.samefile(artifact)):
        parser.error("--output must not overwrite --artifact")
    if output.is_symlink():
        parser.error("--output must not be a symbolic link")
    payload = build_release_bom_payload(
        bom_id=args.bom_id,
        agent_version=args.version,
        component_sha=args.component_sha.lower(),
        config_schema=args.config_schema,
        updater_version=args.updater_version,
        platform_profiles=args.platform_profile,
        artifact_path=artifact,
        artifact_kind=args.artifact_kind,
        built_at=built_at,
    )
    document = canonical_release_bom_json(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if not output.is_file():
            parser.error("--output must be a regular file")
        try:
            existing = output.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            parser.error(f"cannot read existing --output: {exc}")
        if existing != document:
            parser.error("refusing to overwrite an existing release BOM")
        verify_release_artifact(verify_release_bom(payload), artifact)
        print(output)
        return 0
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(document)
            handle.flush()
            os.fsync(handle.fileno())
        # Re-read the artifact after the sidecar has been fully staged. This
        # catches accidental replacement or mutation between hashing and
        # publication instead of emitting a stale BOM.
        verify_release_artifact(verify_release_bom(payload), artifact)
        os.chmod(temporary, 0o644)
        try:
            os.link(temporary, output, follow_symlinks=False)
        except FileExistsError as exc:
            raise RuntimeError("release BOM output appeared during generation") from exc
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
