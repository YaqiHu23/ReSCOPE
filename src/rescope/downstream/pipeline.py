"""One-config entry point for the paper's ROI-supervised downstream workflows."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess

from ..config import resolve_path, write_json

STAGES = ("prepare", "bleep", "istar")
DEFAULTS = {
    "bleep": {"seed": 42, "max_epochs": 20, "batch_size": 128, "num_workers": 4,
              "lr": 0.001, "weight_decay": 0.001, "val_fraction": 0.2, "svd_components": 512,
              "patch_size": 224, "max_cells": 80000, "top_k": 50},
    "istar": {"seed": 20260730, "epochs": 50, "learning_rate": 0.0001,
              "n_states": 1, "n_jobs": 1, "gene_chunk_size": 256,
              "pixel_size_raw": 1.0, "pixel_size_target": 1.0, "radius_raw": None},
}


def load_downstream(path):
    import yaml
    path = Path(path).expanduser().resolve()
    cfg = yaml.safe_load(path.read_text())
    path_fields = {"histology_image", "single_cell_dir", "roi_results_dir", "gene_list", "work_dir",
                   "bleep_repo", "istar_repo", "bleep_python", "istar_python"}
    known = path_fields | {"device", "bleep", "istar"}
    if not isinstance(cfg, dict) or set(cfg) - known:
        raise ValueError("Downstream config must contain only documented fields in examples/downstream.yaml")
    cfg = copy.deepcopy(cfg)
    for key in path_fields:
        if not cfg.get(key):
            raise ValueError(f"Set {key} in the downstream config")
        cfg[key] = str(resolve_path(cfg[key], path.parent, preserve_symlinks=key.endswith("_python")))
    cfg.setdefault("device", "cuda")
    if cfg["device"] not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    for method, defaults in DEFAULTS.items():
        options = cfg.get(method, {})
        if not isinstance(options, dict) or set(options) - set(defaults):
            raise ValueError(f"Unknown {method} options")
        cfg[method] = {**defaults, **options}
        for key, value in cfg[method].items():
            if key == "radius_raw" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Invalid {method}.{key}")
            if isinstance(defaults[key], int) and not isinstance(value, int):
                raise ValueError(f"{method}.{key} must be an integer")
            if value < 0 or (value == 0 and key not in {"seed", "num_workers", "weight_decay"}):
                raise ValueError(f"Invalid {method}.{key}")
    if not 0 < cfg["bleep"]["val_fraction"] < 1:
        raise ValueError("bleep.val_fraction must be between zero and one")
    return cfg


def command(cfg, budget, method):
    root = Path(cfg["work_dir"]) / f"roi_{budget}"
    result = [cfg[f"{method}_python"], "-m", f"rescope.downstream.{method}",
              "--experiment-dir", str(root / "prepared"), "--histology-image", cfg["histology_image"],
              "--outdir", str(root / method), "--roi-names", f"roi_{budget}",
              "--device", cfg["device"], "--gene-list", cfg["gene_list"],
              "--min-total-counts", "1", "--min-genes-detected", "1"]
    for key, value in cfg[method].items():
        if value is not None:
            result += ["--" + key.replace("_", "-"), str(value)]
    if method == "bleep":
        result += ["--model", "resnet50", "--method", "weighted_average"]
    else:
        result += ["--radius-source", "fixed" if cfg[method]["radius_raw"] is not None else "median_area"]
    return result


def preflight(cfg, stage, budgets, methods):
    from .common import load_single_cell_inputs, read_genes
    from .prepare import read_rois
    if not Path(cfg["histology_image"]).is_file():
        raise FileNotFoundError(cfg["histology_image"])
    _, _, features, metadata = load_single_cell_inputs(cfg["single_cell_dir"])
    read_genes(cfg["gene_list"], features.gene_name)
    for budget in budgets:
        read_rois(Path(cfg["roi_results_dir"]) / f"roi_{budget}/visium_hd_region_eval_metadata.json", budget)
    wanted = methods if stage in {"all", "check"} else ((stage,) if stage in methods else ())
    for method in wanted:
        repo = Path(cfg[f"{method}_repo"])
        required = [Path(cfg[f"{method}_python"]), repo / ("models.py" if method == "bleep" else "impute.py")]
        if method == "istar":
            required += [repo / "checkpoints/vit256_small_dino.pth", repo / "checkpoints/vit4k_xs_dino.pth"]
            if cfg["istar"]["radius_raw"] is None and "polygon_area_fullres" not in metadata:
                raise ValueError("Set istar.radius_raw (full-resolution pixels), or supply polygon_area_fullres")
            required += [repo / "train.py"]
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            raise FileNotFoundError("Missing downstream code/weights; see docs/downstream.md:\n" + "\n".join(missing))
        if method == "istar" and 'monitor=\'train_loss\'' not in (repo / "train.py").read_text():
            raise ValueError("iSTAR needs the supplied training patch; run scripts/setup_downstream.sh")


def summarize(cfg, budgets, methods):
    import numpy as np
    import pandas as pd
    from .common import compute_metrics
    rows, sources, common_ids, gene_order = [], [], None, None
    for budget in budgets:
        for method in methods:
            directory = Path(cfg["work_dir"]) / f"roi_{budget}/{method}/roi_{budget}/{method}_eval"
            metrics = json.loads((directory / "metrics.json").read_text())
            rows.append({"num_rois": budget, "method": method, **metrics})
            source = directory / "predictions_and_truth.npz"
            with np.load(source, allow_pickle=False) as data:
                ids = data["query_cell_ids"].astype(str).tolist()
                genes = data["gene_names"].astype(str).tolist()
            if len(set(ids)) != len(ids):
                raise ValueError("Prediction output contains duplicate query IDs")
            if gene_order is None:
                gene_order = genes
            elif genes != gene_order:
                raise ValueError("All downstream runs must use the same gene panel and ordering")
            common_ids = set(ids) if common_ids is None else common_ids & set(ids)
            sources.append((budget, method, source))
    # Each budget has its own held-out region. Also score their common cells for comparison.
    common_ids = sorted(common_ids or [])
    common_rows = []
    for budget, method, source in sources:
        if not common_ids:
            break
        with np.load(source, allow_pickle=False) as data:
            lookup = {x: i for i, x in enumerate(data["query_cell_ids"].astype(str))}
            idx = [lookup[x] for x in common_ids]
            score, _ = compute_metrics(data["predicted"][idx], data["truth"][idx], gene_order)
        common_rows.append({"num_rois": budget, "method": method, **score})
    root = Path(cfg["work_dir"])
    pd.DataFrame(rows).to_csv(root / "summary.csv", index=False)
    pd.DataFrame(common_rows, columns=["num_rois", "method", "mse", "log1p_mse", "n_query_cells", "n_genes"]).to_csv(root / "common_query_summary.csv", index=False)
    write_json(root / "comparison.json", {"budgets": list(budgets), "methods": list(methods),
               "common_query_cells": len(common_ids), "common_gene_count": len(gene_order or []),
               "common_query_available": bool(common_ids)})


def run_downstream(config, stage="all", budgets=(1, 2, 3), methods=("bleep", "istar"), dry_run=False):
    if (stage not in {*STAGES, "all", "check", "summarize"} or not budgets
            or len(set(budgets)) != len(budgets)
            or any(isinstance(k, bool) or not isinstance(k, int) or k not in (1, 2, 3) for k in budgets)):
        raise ValueError("Invalid downstream stage or ROI budgets")
    if not methods or set(methods) - {"bleep", "istar"}:
        raise ValueError("Choose bleep and/or istar")
    if stage in {"bleep", "istar"} and stage not in methods:
        raise ValueError("The requested stage conflicts with --method")
    cfg = load_downstream(config)
    stages = ["prepare", *methods] if stage == "all" else [stage]
    if dry_run:
        for budget in budgets:
            for current in stages:
                print(shlex.join(command(cfg, budget, current)) if current in methods else f"{current}: roi_{budget}")
        return Path(cfg["work_dir"])
    preflight(cfg, stage, budgets, methods)
    if stage == "check":
        print("Input tables, ROI budgets, official code paths and required iSTAR weights are ready.")
        return Path(cfg["work_dir"])
    root = Path(cfg["work_dir"])
    root.mkdir(parents=True, exist_ok=True)
    frozen = root / "downstream_config.json"
    if frozen.exists() and json.loads(frozen.read_text()) != cfg:
        raise ValueError("work_dir belongs to a different configuration; use a new directory")
    write_json(frozen, cfg)
    if stage == "summarize":
        summarize(cfg, budgets, methods)
        return root
    env = os.environ.copy()
    env.update(PYTHONPATH=str(Path(__file__).resolve().parents[2]), PYTHONDONTWRITEBYTECODE="1",
               RESCOPE_BLEEP_REPO=cfg["bleep_repo"], RESCOPE_ISTAR_REPO=cfg["istar_repo"])
    for budget in budgets:
        directory = root / f"roi_{budget}"
        directory.mkdir(exist_ok=True)
        for current in stages:
            status = directory / f"stage_{current}.json"
            if status.exists():
                raise FileExistsError(f"Stage already attempted: {status}; use a fresh work_dir for a full rerun")
            report = {"stage": current, "num_rois": budget, "status": "running"}
            write_json(status, report)
            try:
                if current == "prepare":
                    from .prepare import prepare
                    report["split"] = prepare(cfg, budget)
                    report["gene_list_sha256"] = hashlib.sha256(Path(cfg["gene_list"]).read_bytes()).hexdigest()
                else:
                    report["command"] = command(cfg, budget, current)
                    with (directory / f"{current}.log").open("w") as log:
                        print(f"roi_{budget} {current}: {log.name}", flush=True)
                        subprocess.run(report["command"], check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
                report["status"] = "completed"
            except BaseException as exc:
                report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                write_json(status, report)
    if stage == "all":
        summarize(cfg, budgets, methods)
    return root
