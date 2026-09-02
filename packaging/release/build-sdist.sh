#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIST_DIR="$ROOT_DIR/dist"

python3 -m pip install --upgrade build >/dev/null
python3 -m build --sdist --outdir "$DIST_DIR" >/dev/null

TARBALL=$(python3 - "$DIST_DIR" <<'PY'
from pathlib import Path
import sys

dist = Path(sys.argv[1])
candidates = [*dist.glob("nuv_agent-*.tar.gz"), *dist.glob("nuv-agent-*.tar.gz")]
if candidates:
    print(max(candidates, key=lambda path: path.stat().st_mtime_ns))
PY
)
if [ -z "$TARBALL" ]; then
  echo "No sdist tarball found in $DIST_DIR" >&2
  exit 1
fi

SHA=$(python3 - <<PY
import hashlib
from pathlib import Path
p = Path("$TARBALL")
print(hashlib.sha256(p.read_bytes()).hexdigest())
PY
)

echo "TARBALL=$TARBALL"
echo "SHA256=$SHA"
