#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -I -c 'import sys; assert sys.version_info[:2] == (3,9), "Use Python 3.9 for the two preprocessing environments (set PYTHON_BIN)."'
command -v git >/dev/null
for ENV_NAME in rescope cellvit; do
    ENV_PATH="$ROOT/.venv-$ENV_NAME"
    if [[ ! -x "$ENV_PATH/bin/python" ]]; then
        "$PYTHON_BIN" -m venv "$ENV_PATH"
    fi
    "$ENV_PATH/bin/python" -I -m pip install --upgrade pip
    if [[ "$ENV_NAME" == rescope ]]; then
        "$ENV_PATH/bin/python" -I -m pip install -r "$ROOT/requirements-preprocess.txt"
    else
        "$ENV_PATH/bin/python" -I -m pip install -r "$ROOT/requirements-cellvit.txt"
    fi
    "$ENV_PATH/bin/python" -I -m pip check
 done
CELLVIT_DIR="$ROOT/third_party/CellViT"
CELLVIT_REVISION=05097e18e3d194a65121042f631b5753069f5ee3
if [[ ! -d "$CELLVIT_DIR" ]]; then
    mkdir -p "$ROOT/third_party"
    git clone --no-checkout https://github.com/TIO-IKIM/CellViT.git "$CELLVIT_DIR"
    git -C "$CELLVIT_DIR" checkout --detach "$CELLVIT_REVISION"
fi
if [[ "$(git -C "$CELLVIT_DIR" rev-parse HEAD)" != "$CELLVIT_REVISION" ]]; then
    echo "Existing third_party/CellViT has a different revision; move it aside before setup." >&2
    exit 2
fi
mkdir -p "$ROOT/weights/uni2-h" "$ROOT/weights/cellvit"
echo 'Ready. Download the two weights described in README.md, then edit examples/pipeline.yaml.'
