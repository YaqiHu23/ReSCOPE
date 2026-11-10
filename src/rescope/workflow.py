from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
from argparse import Namespace
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .config import load_config, write_json


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(cfg, *, needs_cells):
    import numpy as np
    import torch
    from .dataset import OnlyVisiumHDRegionDataset
    from .geometry import resolve_model_geometry, grid_tile_pixels

    torch.set_num_threads(cfg["runtime"]["threads"])
    device = cfg["train"]["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"}:
        raise ValueError("train.device must be auto, cpu, or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this Python environment; use --device cpu or a GPU environment")
    cfg["train"]["device"] = device
    cfg["train"]["amp"] = bool(cfg["train"]["amp"] and device == "cuda")
    if os.environ.get("ISCOUT_PROJECT_ROOT"):
        project = Path(os.environ["ISCOUT_PROJECT_ROOT"]).expanduser().resolve()
        for section, field in [("data", "embedding_dir"), ("data", "single_cell_dir"),
                               ("visium_hd_region", "stats_json"), ("visium_hd_region", "region_size_txt")]:
            value = cfg[section].get(field)
            if value and not Path(value).is_relative_to(project):
                raise ValueError(f"{section}.{field} must be under ISCOUT_PROJECT_ROOT={project}")
    dataset = OnlyVisiumHDRegionDataset(cfg["data"]["embedding_dir"], cfg["visium_hd_region"], cfg["data"].get("grid_shape"))
    if len(dataset) != 1 or not dataset.file_list[0].endswith(".npy"):
        raise ValueError("Use one merged *embedding*.npy array [tiles, features] per sample; sharded/.pt grids are not supported by the full selector")
    array = np.load(Path(dataset.path) / dataset.file_list[0], mmap_mode="r", allow_pickle=False)
    if array.ndim != 2 or array.shape != (len(dataset.rows), cfg["data"]["feature_dim"]):
        raise ValueError(f"Embedding shape {array.shape} must be ({len(dataset.rows)}, {cfg['data']['feature_dim']})")
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError("Embeddings must contain floating point features")
    for offset in range(0, len(array), 8192):
        if not np.isfinite(array[offset:offset + 8192]).all():
            raise ValueError("Embeddings contain NaN or infinity")
    if len(np.unique(np.column_stack([dataset.rows, dataset.cols]), axis=0)) != len(dataset.rows):
        raise ValueError("Embedding metadata contains duplicate tile coordinates")
    info = dataset.get_region_info()
    geom = resolve_model_geometry(cfg, info["full_grid_shape"], info["region_grid_shape"])
    if not cfg["model"]["scale_roi_to_region"] and geom["roi_tiles"] != cfg["model"]["base_roi_tiles"]:
        raise ValueError("Requested ROI size exceeds the configured tissue region; reduce model.base_roi_tiles")
    bounds = info["region_bounds"]
    valid = torch.zeros(tuple(info["region_grid_shape"]), dtype=torch.bool)
    inside = ((dataset.rows >= bounds["top"]) & (dataset.rows < bounds["bottom"]) &
              (dataset.cols >= bounds["left"]) & (dataset.cols < bounds["right"]))
    valid[dataset.rows[inside] - bounds["top"], dataset.cols[inside] - bounds["left"]] = True
    pooled = torch.nn.functional.avg_pool2d(valid.float()[None, None], geom["roi_tiles"], geom["stride_tiles"])
    candidates = int((pooled >= cfg["model"]["roi_valid_ratio"]).sum())
    k = cfg["train"]["num_rois"]
    if candidates < k:
        raise ValueError(f"Requested {k} ROIs, but only {candidates} candidates pass the valid-tile threshold")
    h, w = info["region_grid_shape"]
    if k * geom["roi_tiles"] ** 2 > h * w:
        raise ValueError(f"{k} non-overlapping ROIs cannot fit in region {h} x {w}; reduce count or ROI size")
    if cfg["model"]["target_outside_selected_roi"] and k * geom["roi_tiles"] ** 2 >= h * w:
        raise ValueError("ROIs would leave no outside tiles for reconstruction; reduce count or ROI size")
    if needs_cells:
        from .biology import load_cell_metadata, filter_to_grid
        if not cfg["data"]["single_cell_dir"]:
            raise ValueError("data.single_cell_dir is required for the CNN + biological-prior selector")
        metadata = load_cell_metadata(Path(cfg["data"]["single_cell_dir"]))
        coords, *_ = filter_to_grid(metadata, info["full_grid_shape"], grid_tile_pixels(cfg))
        if len(coords) < 2:
            raise ValueError("At least two cells must fall inside the embedding grid; check full-resolution coordinates and mpp")
        if cfg["selection"]["candidate_pool_size"] < k:
            raise ValueError("selection.candidate_pool_size must be >= num_rois")
    return dataset, info, geom, candidates


def train_model(cfg, dataset, info, outdir):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from .factory import build_model
    from .trainer import Trainer

    seed = cfg["train"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model, geom = build_model(cfg, info)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]))
    trainer = Trainer(model, optimizer, cfg)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    best = float("inf")
    checkpoint_path = outdir / "best_model.pt"
    with (outdir / "training_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "eval_loss", "num_rois"])
        writer.writeheader()
        for epoch in range(cfg["train"]["epochs"]):
            loss = trainer.train_epoch(loader, cfg["train"]["device"])
            eval_loss = trainer.eval_epoch(loader, cfg["train"]["device"], cfg["train"]["eval_seed"])
            if not math.isfinite(loss) or not math.isfinite(eval_loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}; no invalid checkpoint will be exported")
            writer.writerow({"epoch": epoch, "train_loss": loss, "eval_loss": eval_loss,
                             "num_rois": cfg["train"]["num_rois"]})
            handle.flush()
            print(f"Epoch {epoch + 1}/{cfg['train']['epochs']} | N={cfg['train']['num_rois']} | train={loss:.6f} | eval={eval_loss:.6f}", flush=True)
            if loss < best:
                best = loss
                state = {"epoch": epoch, "model_state_dict": model.state_dict(),
                         "optimizer_state_dict": optimizer.state_dict(), "loss": loss, "eval_loss": eval_loss,
                         "num_rois": cfg["train"]["num_rois"], "train_seed": seed,
                         "eval_seed": cfg["train"]["eval_seed"], "model_geometry": geom,
                         "region_info": info, "config": cfg, "package_version": __version__,
                         "checkpoint_criterion": "minimum training loss (original model behavior)"}
                temp = checkpoint_path.with_suffix(".partial")
                torch.save(state, temp)
                temp.replace(checkpoint_path)
    return checkpoint_path


def check_checkpoint(path, cfg):
    import torch
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or "model_state_dict" not in state:
        raise ValueError("Checkpoint must contain model_state_dict")
    trained = state.get("num_rois")
    requested = cfg["train"]["num_rois"]
    if isinstance(trained, bool) or not isinstance(trained, int) or trained != requested:
        raise ValueError(f"Checkpoint trained with num_rois={trained}, requested {requested}. Train a separate model with the requested N")
    return {"path": str(Path(path).resolve()), "sha256": sha256(path), "train_num_rois": trained,
            "selection_num_rois": requested}


def select_rois(cfg, dataset, info, geom, outdir, checkpoint, checkpoint_info):
    import numpy as np
    import torch
    from .architecture import build_architecture_coverage_features
    from .cnn import load_cnn_logits
    from .features import candidate_features
    from .priority import single_roi_priority, write_metadata
    from .solver import effective_domain_weights, solve_mip

    args = Namespace(**cfg["selection"], single_cell_dir=Path(cfg["data"]["single_cell_dir"]),
                     ckpt=Path(checkpoint), outdir=outdir, config=outdir / "resolved_config.json",
                     roi_prefix="rescope", selection_mode_name="mip_domain_generalized",
                     phase_name="cnn_biological_prior", search_report_key="domain_generalized_mip_search")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    logits, mask, cnn_geom = load_cnn_logits(cfg, dataset, info, Path(checkpoint))
    if cnn_geom != geom:
        raise RuntimeError("CNN and selector geometry do not match")
    feat = candidate_features(args=args, cfg=cfg, dataset=dataset, region_info=info, geom=geom,
                              cnn_candidate_mask=mask, return_projected_grid=args.architecture_coverage_weight > 0)
    feat.update(roi_tiles=geom["roi_tiles"], stride_tiles=geom["stride_tiles"])
    k = cfg["train"]["num_rois"]
    weights = effective_domain_weights(args, k)
    args.effective_architecture_coverage_weight = weights["architecture_coverage"]
    if args.effective_architecture_coverage_weight > 0:
        feat.update(build_architecture_coverage_features(
            projected_grid=feat.pop("_projected_grid"), valid_mask=feat.pop("_valid_mask"), feat=feat,
            roi_tiles=geom["roi_tiles"], stride_tiles=geom["stride_tiles"],
            n_centroids=args.architecture_centroids, local_grid_size=args.architecture_local_grid_size,
            kmeans_sample_size=args.architecture_kmeans_sample_size,
            kmeans_iters=args.architecture_kmeans_iters, seed=args.seed))
    priority = single_roi_priority(logits, feat, weights, args)
    selected, report = solve_mip(priority=priority, feat=feat, num_rois=k, args=args)
    if len(selected) != k or len(set(selected)) != k:
        raise RuntimeError(f"Solver returned {len(selected)} ROIs for requested {k}")
    radius = max(0, math.ceil(geom["roi_tiles"] / geom["stride_tiles"]) - 1)
    for i, a in enumerate(selected):
        for b in selected[i + 1:]:
            if abs(int(feat["rows"][a]) - int(feat["rows"][b])) <= radius and abs(int(feat["cols"][a]) - int(feat["cols"][b])) <= radius:
                raise RuntimeError("Solver returned overlapping ROIs")
    report.update(effective_weights=weights, selection_seed=args.seed, solver=args.solver)
    write_metadata(args=args, cfg=cfg, geom=geom, region_info=info, feat=feat,
                   weights=weights, selected_indices=selected, projection_report=report)
    metadata_path = outdir / "visium_hd_region_eval_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["checkpoint_role"] = "user-selected ROI scorer"
    metadata["checkpoint_provenance"] = checkpoint_info
    metadata["num_rois_requested"] = k
    metadata["num_rois_selected"] = len(selected)
    write_json(metadata_path, metadata)
    rois = metadata["model_rois"]
    fields = ["roi_name", "left_px", "top_px", "right_px", "bottom_px", "size_px"]
    with (outdir / "rois.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rois)
    features = []
    for roi in rois:
        x, y, r, b = (roi[key] for key in ["left_px", "top_px", "right_px", "bottom_px"])
        features.append({"type": "Feature", "properties": {"name": roi["roi_name"], "coordinate_space": "full_resolution_pixels"},
                         "geometry": {"type": "Polygon", "coordinates": [[[x, y], [r, y], [r, b], [x, b], [x, y]]]}})
    write_json(outdir / "rois.geojson", {"type": "FeatureCollection", "features": features})


def execute(config, *, command="run", num_rois=None, output_dir=None, device=None, epochs=None,
            checkpoint=None, base_dir=None, validate_for="run"):
    if command not in {"run", "train", "select", "validate"}:
        raise ValueError(f"Unknown command: {command}")
    if validate_for not in {"run", "train", "select"}:
        raise ValueError("validate_for must be run, train, or select")
    cfg = load_config(config, num_rois=num_rois, output_dir=output_dir, device=device, epochs=epochs,
                      base_dir=base_dir)
    needs_cells = (validate_for if command == "validate" else command) != "train"
    dataset, info, geom, candidates = prepare(cfg, needs_cells=needs_cells)
    if command == "validate":
        return {"status": "valid", "stage": validate_for, "num_rois": cfg["train"]["num_rois"], "device": cfg["train"]["device"],
                "candidate_count": candidates, "model_geometry": geom, "solver": cfg["selection"]["solver"],
                "note": "Candidate and area checks passed; final non-overlap and biological constraints are checked during selection"}
    checkpoint_info = None
    if command == "select":
        if checkpoint is None:
            raise ValueError("select requires an explicit checkpoint")
        checkpoint = Path(checkpoint).expanduser().resolve()
        checkpoint_info = check_checkpoint(checkpoint, cfg)
    outdir = Path(cfg["runtime"]["output_dir"])
    outdir.mkdir(parents=True, exist_ok=False)
    manifest = {"version": __version__, "status": "running", "command": command,
                "started_utc": datetime.now(timezone.utc).isoformat(), "num_rois": cfg["train"]["num_rois"],
                "source_config": None if isinstance(config, Mapping) else str(Path(config).expanduser().resolve()),
                "config_source": "mapping" if isinstance(config, Mapping) else "file",
                "model_geometry": geom}
    write_json(outdir / "resolved_config.json", cfg)
    write_json(outdir / "run_manifest.json", manifest)
    print(f"Output: {outdir}\nROI budget: {cfg['train']['num_rois']} | device: {cfg['train']['device']} | solver: {cfg['selection']['solver']}", flush=True)
    try:
        if command in {"run", "train"}:
            checkpoint = train_model(cfg, dataset, info, outdir)
            checkpoint_info = check_checkpoint(checkpoint, cfg)
        if command in {"run", "select"}:
            select_rois(cfg, dataset, info, geom, outdir, checkpoint, checkpoint_info)
        manifest.update(status="completed", checkpoint=checkpoint_info,
                        finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(outdir / "run_manifest.json", manifest)
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(outdir / "run_manifest.json", manifest)
        raise
    return outdir
