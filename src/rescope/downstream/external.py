"""Locate separately downloaded upstream code; never bundle model weights."""
import os
from pathlib import Path


def repository(method):
    variable = f"RESCOPE_{method.upper()}_REPO"
    value = os.environ.get(variable)
    if not value:
        raise RuntimeError(f"Set {variable}, or use 'run.sh downstream --config ...'")
    path = Path(value).expanduser().resolve()
    entry = "models.py" if method == "bleep" else "impute.py"
    if not (path / entry).is_file():
        raise FileNotFoundError(f"Missing {path / entry}; run scripts/setup_downstream.sh")
    return path
