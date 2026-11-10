#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$([[ -x "$ROOT/.venv-rescope/bin/python" ]] && echo "$ROOT/.venv-rescope/bin/python" || echo python3)}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python not found: $PYTHON_BIN. Install Python 3.9-3.12 or set PYTHON_BIN." >&2
    exit 2
fi
if [[ $# -eq 0 ]]; then
    set -- --help
fi
case "$1" in
    --help|-h|--version)
        exec "$PYTHON_BIN" -I -B "$ROOT/launch.py" "$@" ;;
esac
"$PYTHON_BIN" -I -B -c 'import sys; assert (3, 9) <= sys.version_info[:2] < (3, 13), "Use Python 3.9-3.12"'

check_dependencies() {
    "$1" -I -B -c 'from importlib.metadata import version
import torch, numpy, scipy.optimize, pandas, yaml
for name, low, high in [("torch", (2,4), (2,9)), ("numpy", (1,26), (2,0)), ("scipy", (1,11), (1,14)), ("pandas", (2,1), (2,4)), ("PyYAML", (6,0), (7,0))]:
    v = tuple(map(int, version(name).split("+")[0].split(".")[:2]))
    assert low <= v < high, name' >/dev/null 2>&1
}
if ! check_dependencies "$PYTHON_BIN"; then
    RESCOPE_ENV="${RESCOPE_VENV:-${ISCOUT_SCRATCH_ROOT:-$ROOT}/.venv-rescope}"
    if [[ ! -x "$RESCOPE_ENV/bin/python" ]]; then
        "$PYTHON_BIN" -I -B -m venv "$RESCOPE_ENV"
    fi
    PYTHON_BIN="$RESCOPE_ENV/bin/python"
    if ! check_dependencies "$PYTHON_BIN"; then
        echo "Installing ReSCOPE dependencies in $RESCOPE_ENV (internet required on first run)."
        "$PYTHON_BIN" -I -B -m pip install -r "$ROOT/requirements.txt"
    fi
    if ! check_dependencies "$PYTHON_BIN"; then
        echo "Dependency check failed. Use a compatible Python environment via PYTHON_BIN." >&2
        exit 2
    fi
fi
case "$1" in
    run|train|select|validate|demo|pipeline|downstream)
        exec "$PYTHON_BIN" -I -B -u "$ROOT/launch.py" "$@" ;;
    *)
        exec "$PYTHON_BIN" -I -B -u "$ROOT/launch.py" run "$@" ;;
esac
