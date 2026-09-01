from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _test_module(path: Path) -> str:
    return ".".join(path.relative_to(ROOT).with_suffix("").parts)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run each unittest module in a fresh interpreter and config scope"
    )
    parser.add_argument("--pattern", default="test_*.py")
    args = parser.parse_args()

    modules = sorted((ROOT / "tests").rglob(args.pattern))
    if not modules:
        parser.error(f"no tests matched {args.pattern!r}")

    with tempfile.TemporaryDirectory(prefix="nuv-agent-release-tests-") as config_root:
        for index, test_path in enumerate(modules):
            module = _test_module(test_path)
            environment = os.environ.copy()
            environment["NUV_AGENT_CONFIG"] = str(
                Path(config_root) / f"{index:03d}-{test_path.stem}.env"
            )
            print(f"\n=== {module} ===", flush=True)
            result = subprocess.run(
                [sys.executable, "-m", "unittest", "-v", module],
                cwd=ROOT,
                env=environment,
                check=False,
            )
            if result.returncode != 0:
                return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
