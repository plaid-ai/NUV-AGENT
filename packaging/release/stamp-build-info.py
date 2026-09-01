from __future__ import annotations

import argparse
import re
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"
BUILD_INFO = ROOT / "nuvion_app" / "build_info.py"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stamp immutable agent release identity"
    )
    parser.add_argument("--sha", required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()

    sha = args.sha.strip().lower()
    version = args.version.strip()
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", sha):
        parser.error("--sha must be a full hexadecimal commit SHA")

    declared_version = str(
        tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["version"]
    )
    if version != declared_version:
        parser.error(
            f"version mismatch: release={version} pyproject={declared_version}"
        )

    BUILD_INFO.write_text(
        '"""Generated release identity. Do not edit in release artifacts."""\n\n'
        f'AGENT_VERSION = "{version}"\n'
        f'COMPONENT_SHA = "{sha}"\n',
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
