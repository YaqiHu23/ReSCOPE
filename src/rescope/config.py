from __future__ import annotations

import copy
import json
import math
import os
import tempfile
from datetime import datetime
from collections.abc import Mapping
from pathlib import Path
from uuid import uuid4

DEFAULTS = {
    "data": {"embedding_dir": None, "feature_dim": 1536, "tile_um": 8.0, "mpp": None,
             "single_cell_dir": None, "tile_px": None, "grid_shape": None},
    "visium_hd_region": {"source": "ratio", "top_ratio": 0.0, "bottom_ratio": 1.0,
                         "left_ratio": 0.0, "right_ratio": 1.0, "shrink_ratio": 1.0},
    "model": {"base_roi_tiles": 164, "base_stride_tiles": 7, "scale_roi_to_region": False,
              "min_roi_tiles": 1, "min_stride_tiles": 1, "roi_valid_ratio": 0.7,
              "tau_select": 2.0, "tau_match": 0.05, "recon_percent": 0.1, "max_samples": None,
              "normalize_select_logits": True, "select_logit_clip": 8.0,
              "select_entropy_weight": 0.001, "avoid_roi_overlap": True,
              "selection_mode": "combinatorial_thretopk", "selection_noise": "gumbel",
              "sample_in_eval": False, "thretopk_eps": 1e-6,
              "reconstruction_mode": "roi_prototypes", "prototype_grid_size": 8,
              "prototype_use_straight_through": True, "target_outside_selected_roi": True},
    "train": {"num_rois": None, "epochs": 30, "lr": 0.0001, "chunk_size": 64,
              "amp": True, "device": "auto", "seed": 42, "eval_seed": 1234},
    "eval": {"reconstruction_mode": "roi_prototypes", "sample_targets": False,
             "exact_roi_size_px": None},
    "selection": {"seed": 20260628, "knn_k": 6, "projection_dim": 32, "chunk_size": 8192,
                  "n_clusters": 24, "kmeans_iters": 25, "min_cells": 6000,
                  "min_valid_ratio": 0.70, "target_reference_cells": 40000.0,
                  "quota_reference_cells": 40000.0, "candidate_pool_size": 1000,
                  "solver": "exact", "time_limit": 1200.0, "mip_gap": 0.001,
                  "adaptive_weights": False, "max_reference_fraction": 0.30,
                  "min_reference_fraction": 0.045,
                  "reference_fraction_retry_multipliers": [1.0, 1.2, 1.5, 0.0],
                  "reference_size_penalty_weight": 0.0, "enforce_celltype_upper_bounds": True,
                  "celltype_upper_tolerance": 0.30, "min_celltype_fraction_for_upper": 0.05,
                  "cnn_weight": 0.95, "cell_feature_weight": 1.2, "uni_feature_weight": 1.0,
                  "cluster_weight": 0.65, "adjacency_weight": 0.95, "composition_weight": 0.95,
                  "coverage_weight": 0.80, "density_weight": 0.10, "boundary_weight": 0.25,
                  "uni_represent_weight": 0.90, "reference_budget_weight": 0.20,
                  "celltype_quota_weight": 0.20, "architecture_coverage_weight": 0.0,
                  "architecture_centroids": 24, "architecture_local_grid_size": 4,
                  "architecture_kmeans_sample_size": 50000, "architecture_kmeans_iters": 25},
    "runtime": {"threads": 4, "output_dir": None},
}


def positive_int(value, name="num_rois"):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return value


def resolve_path(value, base, *, preserve_symlinks=False):
    text = os.path.expandvars(str(value))
    if "$" in text:
        raise ValueError(f"Unset environment variable in path: {value}")
    path = Path(text).expanduser()
    path = base / path if not path.is_absolute() else path
    # A venv's python symlink must be invoked through the venv path so Python
    # finds pyvenv.cfg and uses that environment's installed dependencies.
    return Path(os.path.abspath(path)) if preserve_symlinks else path.resolve()


def load_config(config, *, num_rois=None, output_dir=None, device=None, epochs=None, base_dir=None):
    """Resolve a config file or mapping without mutating the caller's settings.

    Mapping paths use base_dir (default: cwd); file paths use the file's parent.
    Explicit output_dir overrides always resolve from the current directory.
    """
    if isinstance(config, Mapping):
        raw = config
        config_base = Path(base_dir).expanduser().resolve() if base_dir is not None else Path.cwd()
    else:
        import yaml
        if base_dir is not None:
            raise ValueError("base_dir is only supported with a config mapping; files use their own directory")
        path = Path(config).expanduser().resolve()
        raw = yaml.safe_load(path.read_text())
        config_base = path.parent
    if not isinstance(raw, Mapping):
        raise ValueError("Configuration must be a YAML/JSON mapping or a Python mapping")
    cfg = copy.deepcopy(DEFAULTS)
    for section, values in raw.items():
        if section not in cfg or not isinstance(values, Mapping):
            raise ValueError(f"Unknown or invalid config section: {section}")
        unknown = set(values) - set(cfg[section])
        # These optional region fields are supported by the original geometry reader.
        if section == "visium_hd_region":
            unknown -= {"stats_json", "region_size_txt", "bbox_key", "image_width", "image_height"}
        if section == "model":
            unknown -= {"roi_scale_mode"}
        if unknown:
            raise ValueError(f"Unknown config fields in {section}: {sorted(unknown)}")
        cfg[section].update(copy.deepcopy(dict(values)))
    for name, value in (("num_rois", num_rois), ("device", device), ("epochs", epochs)):
        if value is not None:
            cfg["train"][name] = value
    positive_int(cfg["train"]["num_rois"])
    if cfg["train"]["num_rois"] not in (1, 2, 3):
        raise ValueError("This release supports num_rois = 1, 2, or 3")
    for section, names in {"train": ["epochs", "chunk_size"], "data": ["feature_dim"],
                           "model": ["base_roi_tiles", "base_stride_tiles", "min_roi_tiles",
                                     "min_stride_tiles", "prototype_grid_size"],
                           "runtime": ["threads"],
                           "selection": ["knn_k", "projection_dim", "chunk_size", "n_clusters",
                                         "kmeans_iters", "candidate_pool_size"]}.items():
        for name in names:
            positive_int(cfg[section][name], f"{section}.{name}")
    for section, name in [("data", "tile_um"), ("data", "mpp"), ("train", "lr"),
                          ("model", "tau_select"), ("model", "tau_match"),
                          ("selection", "time_limit")]:
        value = cfg[section][name]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{section}.{name} must be a finite positive number")
    for section, name in [("model", "roi_valid_ratio"), ("model", "recon_percent"),
                          ("selection", "min_valid_ratio"), ("selection", "mip_gap"),
                          ("selection", "min_reference_fraction"), ("selection", "max_reference_fraction")]:
        value = cfg[section][name]
        if not isinstance(value, (float, int)) or isinstance(value, bool) or not 0 <= value <= 1:
            raise ValueError(f"{section}.{name} must be between 0 and 1")
    if cfg["data"]["tile_px"] is not None:
        positive_int(cfg["data"]["tile_px"], "data.tile_px")
    shape = cfg["data"]["grid_shape"]
    if shape is not None:
        if not isinstance(shape, list) or len(shape) != 2:
            raise ValueError("data.grid_shape must be [rows, columns]")
        for value in shape:
            positive_int(value, "data.grid_shape")
    if cfg["model"]["recon_percent"] == 0:
        raise ValueError("model.recon_percent must be > 0")
    if not cfg["model"]["avoid_roi_overlap"]:
        raise ValueError("This tool requires non-overlapping ROIs; model.avoid_roi_overlap must be true")
    if cfg["visium_hd_region"]["source"] not in {"ratio", "region_size_txt", "stats_json"}:
        raise ValueError("Unknown visium_hd_region.source")
    for section, name in [("data", "embedding_dir"), ("data", "single_cell_dir"),
                          ("visium_hd_region", "stats_json"), ("visium_hd_region", "region_size_txt")]:
        value = cfg[section].get(name)
        if value:
            cfg[section][name] = str(resolve_path(value, config_base))
    if not cfg["data"]["embedding_dir"]:
        raise ValueError("data.embedding_dir is required")
    if cfg["selection"]["solver"] != "exact":
        raise ValueError("This release uses selection.solver: exact")
    k = cfg["train"]["num_rois"]
    chosen_output = output_dir or cfg["runtime"]["output_dir"]
    if chosen_output:
        base = Path.cwd() if output_dir else config_base
        cfg["runtime"]["output_dir"] = str(resolve_path(chosen_output, base))
    else:
        scratch = Path(os.environ.get("ISCOUT_SCRATCH_ROOT", str(Path.cwd()))).expanduser().resolve()
        cfg["runtime"]["output_dir"] = str(scratch / "runs" / f"rescope_{k}roi_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}")
    return cfg


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)
