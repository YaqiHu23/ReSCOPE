"""Run the source checkout without importing files from the caller's directory."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from rescope.cli import main

raise SystemExit(main())
