"""Use the union of selected ROIs as reference and its complement as query."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from ..config import write_json
from .common import load_single_cell_inputs, read_genes


def read_rois(path, budget):
    payload = json.loads(Path(path).read_text())
    rois = payload.get("model_rois")
    if not isinstance(rois, list) or len(rois) != budget:
        raise ValueError(f"Expected exactly {budget} model_rois in {path}")
    provenance = payload.get("checkpoint_provenance", {})
    if provenance.get("train_num_rois", budget) != budget:
        raise ValueError("Selected ROI budget does not match the training checkpoint")
    for roi in rois:
        bounds = np.array([roi[k] for k in ("left_px", "top_px", "right_px", "bottom_px")], dtype=float)
        if not np.isfinite(bounds).all() or min(bounds) < 0 or bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
            raise ValueError("ROI bounds must be finite nonnegative full-resolution x,y rectangles")
    return rois


def roi_mask(rois, xy):
    inside = np.zeros(len(xy), dtype=bool)
    for roi in rois:
        inside |= ((xy[:, 0] >= roi["left_px"]) & (xy[:, 0] < roi["right_px"]) &
                   (xy[:, 1] >= roi["top_px"]) & (xy[:, 1] < roi["bottom_px"]))
    return inside


def save_subset(directory, matrix, ids, features, metadata):
    directory.mkdir(parents=True)
    sparse.save_npz(directory / "cell_gene_counts.npz", matrix)
    pd.Series(ids).to_csv(directory / "cell_ids.tsv.gz", sep="\t", index=False, header=False)
    features.to_csv(directory / "features.tsv.gz", sep="\t", index=False, header=False)
    metadata.to_csv(directory / "cell_metadata.csv.gz", index=False)


def prepare(cfg, budget):
    roi_path = Path(cfg["roi_results_dir"]) / f"roi_{budget}/visium_hd_region_eval_metadata.json"
    rois = read_rois(roi_path, budget)
    matrix, ids, features, metadata = load_single_cell_inputs(cfg["single_cell_dir"])
    read_genes(cfg["gene_list"], features.gene_name)
    xy = metadata[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    inside = roi_mask(rois, xy)
    if inside.sum() < 3 or inside.all():
        raise ValueError("Each ROI budget needs at least three reference cells and one outside query cell")
    experiment = Path(cfg["work_dir"]) / f"roi_{budget}/prepared"
    if experiment.exists():
        raise FileExistsError(f"Prepared inputs already exist: {experiment}")
    name = f"roi_{budget}"
    root = experiment / name
    reference, query = root / "reference_single_cell", root / "query_single_cell"
    for mask, dest in ((inside, reference), (~inside, query)):
        indices = np.flatnonzero(mask)
        save_subset(dest, matrix[indices], [ids[i] for i in indices], features,
                    metadata.iloc[indices].reset_index(drop=True))
    query_metadata = metadata.loc[~inside, ["cell_id", "centroid_x", "centroid_y"]].copy()
    query_metadata["image_x"] = query_metadata.centroid_x
    query_metadata["image_y"] = query_metadata.centroid_y
    query_metadata.to_csv(root / "query_metadata.csv", index=False)
    manifest = {"roi_name": name, "components": rois, "source_roi_json": str(roi_path),
                "n_reference_cells": int(inside.sum()), "n_query_cells": int((~inside).sum()),
                "reference_single_cell_dir": str(reference), "query_single_cell_dir": str(query),
                "query_metadata_csv": str(root / "query_metadata.csv"),
                "reference_region_kind": "inside_roi_union", "query_region_kind": "outside_roi_union",
                "coordinate_source": "centroid_xy"}
    write_json(root / "roi_manifest.json", manifest)
    return manifest
