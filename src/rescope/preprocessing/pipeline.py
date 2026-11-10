"""Explicit, sequential WSI-to-ROI stages; all paths come from one YAML file."""
from pathlib import Path
import copy
import json
import math
import os
import re
import shlex
import subprocess
import sys

from ..config import resolve_path, write_json

STAGES = ("embed", "patches", "segment", "cells", "config", "train")


def load_pipeline(path):
    import yaml
    path = Path(path).expanduser().resolve()
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError("Pipeline config must be a YAML mapping")
    known = {"sample", "wsi_path", "mpp", "tile_um", "work_dir", "uni_checkpoint", "cellvit_repo",
             "cellvit_checkpoint", "cellvit_python", "magnification", "embedding_batch_size",
             "embedding_workers", "cellvit_batch_size", "gpu", "reader", "device",
             "hd_square_002um_dir", "rescope"}
    if set(cfg) - known:
        raise ValueError(f"Unknown pipeline fields: {sorted(set(cfg)-known)}")
    cfg = copy.deepcopy(cfg)
    for key in ("sample", "wsi_path", "mpp", "uni_checkpoint", "cellvit_repo", "cellvit_checkpoint", "cellvit_python"):
        if cfg.get(key) is None:
            raise ValueError(f"Set {key} in the pipeline config")
    if not isinstance(cfg["sample"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", cfg["sample"]):
        raise ValueError("sample may contain only letters, numbers, underscore and hyphen")
    cfg.setdefault("tile_um", 8.0)
    cfg.setdefault("work_dir", str(Path(os.environ.get("ISCOUT_SCRATCH_ROOT", path.parent.parent)) / "runs" / cfg["sample"]))
    cfg.setdefault("magnification", 40)
    cfg.setdefault("embedding_batch_size", 64)
    cfg.setdefault("embedding_workers", 4)
    cfg.setdefault("cellvit_batch_size", 8)
    cfg.setdefault("gpu", 0)
    cfg.setdefault("reader", "auto")
    cfg.setdefault("device", "cuda")
    cfg.setdefault("rescope", {})
    if not isinstance(cfg["rescope"], dict) or set(cfg["rescope"]) - {"model", "train", "selection", "visium_hd_region", "runtime", "eval"}:
        raise ValueError("rescope contains unsupported sections")
    for key in ("wsi_path", "work_dir", "uni_checkpoint", "cellvit_repo", "cellvit_checkpoint", "cellvit_python", "hd_square_002um_dir"):
        if cfg.get(key) is not None:
            cfg[key] = str(resolve_path(cfg[key], path.parent))
    for key in ("mpp", "tile_um"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], (float, int)) or not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be a finite positive number")
    for key in ("embedding_batch_size", "cellvit_batch_size", "embedding_workers", "gpu"):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key in {"embedding_workers", "gpu"} else 1):
            raise ValueError(f"Invalid {key}")
    if cfg["magnification"] not in (20, 40) or cfg["device"] not in ("cpu", "cuda"):
        raise ValueError("Use magnification 20/40 and device cpu/cuda")
    if cfg["reader"] not in ("auto", "openslide", "tiff-zarr", "tiff-memmap", "tiff-memory"):
        raise ValueError("Unknown reader")
    if cfg["device"] == "cuda" and cfg["gpu"] != 0:
        raise ValueError("Set CUDA_VISIBLE_DEVICES to the requested GPU and use gpu: 0 for consistent device selection")
    return cfg


def paths(cfg):
    work = Path(cfg["work_dir"])
    return {"work": work, "embeddings": work / "embeddings", "patches": work / "patches",
            "cells_json": work / "patches" / Path(cfg["wsi_path"]).stem / "cell_detection/rescope/cells.json",
            "cells": work / "cells", "config": work / "rescope.yaml", "results": work / "results"}


def commands(cfg):
    p = paths(cfg)
    return {
        "embed": [sys.executable, "-m", "rescope.preprocessing.uni", "--wsi-path", cfg["wsi_path"],
                  "--sample", cfg["sample"], "--output-dir", str(p["embeddings"]),
                  "--weight-path", cfg["uni_checkpoint"], "--tile-um", str(cfg["tile_um"]),
                  "--mpp", str(cfg["mpp"]), "--batch-size", str(cfg["embedding_batch_size"]),
                  "--num-workers", str(cfg["embedding_workers"]), "--reader", cfg["reader"],
                  "--device", cfg["device"], "--skip-background"],
        "segment": [cfg["cellvit_python"], str(Path(__file__).with_name("cellvit.py")),
                    "--repo", cfg["cellvit_repo"], "--checkpoint", cfg["cellvit_checkpoint"],
                    "--wsi", cfg["wsi_path"], "--patch-root", str(p["patches"]),
                    "--magnification", str(cfg["magnification"]), "--gpu", str(cfg["gpu"]),
                    "--batch-size", str(cfg["cellvit_batch_size"])],
    }


def preflight(cfg, stage):
    # Raw inputs remain persistent. Generated features and patches are intermediates.
    raw_files = [cfg["wsi_path"]]
    required = list(raw_files)
    if stage in ("all", "check", "embed"):
        required += [cfg["uni_checkpoint"]]
    if stage in ("all", "check", "segment"):
        required += [cfg["cellvit_checkpoint"], cfg["cellvit_python"],
                     str(Path(cfg["cellvit_repo"]) / "cell_segmentation/inference/cell_detection.py")]
        raw_files += [cfg["cellvit_checkpoint"]]
    if cfg.get("hd_square_002um_dir") and stage in ("all", "check", "cells"):
        square = Path(cfg["hd_square_002um_dir"])
        required += [str(square / "filtered_feature_bc_matrix.h5"), str(square / "spatial/tissue_positions.parquet")]
    missing = [str(p) for p in required if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))
    if os.environ.get("ISCOUT_PROJECT_ROOT"):
        project = Path(os.environ["ISCOUT_PROJECT_ROOT"]).expanduser().resolve()
        if not Path(cfg["wsi_path"]).is_relative_to(project):
            raise ValueError("Raw WSI must be under ISCOUT_PROJECT_ROOT")
    from importlib.util import find_spec
    modules = {"embed": ("timm", "torchvision", "tifffile"), "patches": ("PIL", "tifffile"),
               "cells": ("ijson",), "config": ("numpy", "pandas"), "train": ("torch",)}
    wanted = STAGES if stage in ("all", "check") else (stage,)
    if cfg.get("hd_square_002um_dir") and "cells" in wanted:
        modules["cells"] += ("h5py", "pyarrow", "shapely")
    missing_modules = sorted({name for s in wanted for name in modules.get(s, ()) if find_spec(name) is None})
    if missing_modules:
        raise ImportError(f"Missing {missing_modules}; run scripts/setup_preprocessing.sh or install requirements-preprocess.txt")


def make_rescope_config(cfg):
    import numpy as np
    import pandas as pd
    import yaml
    p = paths(cfg)
    if p["config"].exists():
        raise FileExistsError(p["config"])
    meta = json.loads((p["embeddings"] / "manifest.json").read_text())
    if Path(meta["wsi_path"]).resolve() != Path(cfg["wsi_path"]):
        raise ValueError("Embedding manifest belongs to a different WSI")
    if not math.isclose(float(meta["mpp"]), cfg["mpp"]) or not math.isclose(float(meta["tile_um"]), cfg["tile_um"]):
        raise ValueError("Embedding geometry differs from pipeline config")
    matrix = np.load(p["embeddings"] / "embeddings.npy", mmap_mode="r", allow_pickle=False)
    if matrix.ndim != 2 or matrix.shape[1] != 1536 or matrix.shape[0] != meta["tiles_embedded"]:
        raise ValueError("Expected complete UNI2-h embeddings with 1536 features per tile")
    columns = pd.read_csv(p["cells"] / "cell_metadata.csv.gz", nrows=2)
    if not {"centroid_x", "centroid_y", "cell_type"}.issubset(columns) or len(columns) < 2:
        raise ValueError("Provide a valid cell table before preparing the ReSCOPE config")
    result = copy.deepcopy(cfg["rescope"])
    result["data"] = {"embedding_dir": "embeddings", "single_cell_dir": "cells", "feature_dim": 1536,
                      "tile_um": cfg["tile_um"], "mpp": cfg["mpp"], "tile_px": meta["tile_px"],
                      "grid_shape": [meta["num_tiles_y"], meta["num_tiles_x"]]}
    result.setdefault("train", {}).setdefault("device", cfg["device"])
    result.setdefault("model", {}).setdefault("base_roi_tiles", 164)
    result["model"].setdefault("base_stride_tiles", 7)
    result["model"].setdefault("scale_roi_to_region", False)
    p["config"].write_text(yaml.safe_dump(result, sort_keys=False))
    return p["config"]


def run_pipeline(config, stage="all", dry_run=False):
    if stage not in (*STAGES, "all", "check"):
        raise ValueError("Unknown pipeline stage")
    cfg = load_pipeline(config)
    p, cmds = paths(cfg), commands(cfg)
    selected = STAGES if stage == "all" else (stage,)
    if dry_run:
        for s in selected:
            print(shlex.join(cmds[s]) if s in cmds else f"{s}: {p['work']}")
        return p["work"]
    preflight(cfg, stage)
    if stage == "check":
        print("Paths and preprocessing dependencies are ready. CellViT inference needs CUDA.")
        return p["work"]
    p["work"].mkdir(parents=True, exist_ok=True)
    frozen = p["work"] / "pipeline_config.json"
    if frozen.exists() and json.loads(frozen.read_text()) != cfg:
        raise ValueError("This work_dir belongs to a different pipeline configuration; use a new directory")
    write_json(frozen, cfg)
    for s in selected:
        status = p["work"] / f"stage_{s}.json"
        if status.exists():
            raise FileExistsError(f"Stage already attempted: {s}. Inspect outputs and use a fresh work_dir for a full rerun")
        report = {"stage": s, "status": "running"}
        write_json(status, report)
        try:
            print(f"=== {s} ===", flush=True)
            if s in cmds:
                env = os.environ.copy()
                source = str(Path(__file__).resolve().parents[2])
                env["PYTHONPATH"] = source
                env["PYTHONDONTWRITEBYTECODE"] = "1"
                report["command"] = cmds[s]
                with (p["work"] / f"{s}.log").open("w") as log:
                    print(f"Progress log: {log.name}", flush=True)
                    subprocess.run(cmds[s], check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
            elif s == "patches":
                from .patches import create_patches
                create_patches(cfg["wsi_path"], p["patches"], cfg["mpp"], cfg["magnification"], cfg["reader"])
            elif s == "cells":
                from .cells import convert_cells
                from .image_io import open_reader
                slide = open_reader(cfg["wsi_path"], cfg["reader"])
                try:
                    width, height = slide.dimensions
                finally:
                    slide.close()
                convert_cells(p["cells_json"], p["cells"], width, height, cfg.get("hd_square_002um_dir"))
            elif s == "config":
                make_rescope_config(cfg)
            elif s == "train":
                from ..batch import run_budgets
                # The pipeline has checked the raw WSI's persistent location.
                # Its own generated inputs are temporary preprocessing products.
                project = os.environ.pop("ISCOUT_PROJECT_ROOT", None)
                try:
                    run_budgets(p["config"], output_dir=p["results"])
                finally:
                    if project is not None:
                        os.environ["ISCOUT_PROJECT_ROOT"] = project
            report["status"] = "completed"
        except BaseException as exc:
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            write_json(status, report)
    return p["work"]
