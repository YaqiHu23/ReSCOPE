#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -I -c 'import sys; assert sys.version_info[:2] == (3,9), "Set PYTHON_BIN to Python 3.9"'
command -v git >/dev/null
for METHOD in bleep istar; do
    ENV_PATH="$ROOT/.venv-$METHOD"
    if [[ ! -x "$ENV_PATH/bin/python" ]]; then
        "$PYTHON_BIN" -m venv "$ENV_PATH"
    fi
    "$ENV_PATH/bin/python" -I -m pip install --upgrade pip
    "$ENV_PATH/bin/python" -I -m pip install -r "$ROOT/requirements-$METHOD.txt"
    "$ENV_PATH/bin/python" -I -m pip check
done
"$PYTHON_BIN" "$ROOT/scripts/setup_downstream_repos.py"
echo 'Ready. Obtain official weights as described in docs/downstream.md, then edit examples/downstream.yaml.'
