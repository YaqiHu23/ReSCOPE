"""Run each ROI budget with a newly initialized model and its own checkpoint."""
from datetime import datetime, timezone
from pathlib import Path

from .config import load_config, write_json
from .workflow import execute


def run_budgets(config, *, budgets=(1, 2, 3), command="run", output_dir=None,
                device=None, epochs=None):
    if command not in {"run", "train", "validate"}:
        raise ValueError("Batch command must be run, train, or validate")
    budgets = tuple(budgets)
    if not budgets or len(set(budgets)) != len(budgets):
        raise ValueError("Provide distinct ROI budgets")
    configs = [load_config(config, num_rois=k, output_dir=output_dir,
                           device=device, epochs=epochs) for k in budgets]
    if command == "validate":
        return [execute(config, command="validate", num_rois=k, device=device)
                for k in budgets]
    parent = Path(configs[0]["runtime"]["output_dir"])
    parent.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "budgets": list(budgets), "command": command,
                "training": "independent model initialization for each budget",
                "started_utc": datetime.now(timezone.utc).isoformat(), "runs": {}}
    write_json(parent / "summary.json", manifest)
    try:
        for k in budgets:
            result = execute(config, command=command, num_rois=k, device=device,
                             epochs=epochs, output_dir=parent / f"roi_{k}")
            manifest["runs"][str(k)] = str(result)
            write_json(parent / "summary.json", manifest)
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(parent / "summary.json", manifest)
    return parent
