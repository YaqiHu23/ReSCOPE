"""Aligned count-table loading and evaluation shared by downstream adapters."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

CELL_FILES = ("cell_gene_counts.npz", "cell_ids.tsv.gz", "features.tsv.gz", "cell_metadata.csv.gz")


def load_single_cell_inputs(directory):
    directory = Path(directory)
    for name in CELL_FILES:
        if not (directory / name).is_file():
            raise FileNotFoundError(f"Missing {directory / name}; downstream prediction needs measured gene counts")
    matrix = sparse.load_npz(directory / CELL_FILES[0]).tocsr()
    cell_ids = pd.read_csv(directory / CELL_FILES[1], sep="\t", header=None, dtype=str,
                           keep_default_na=False)[0].tolist()
    features = pd.read_csv(directory / CELL_FILES[2], sep="\t", header=None, dtype=str,
                           names=["gene_id", "gene_name", "feature_type"], keep_default_na=False)
    metadata = pd.read_csv(directory / CELL_FILES[3], dtype={"cell_id": str})
    if matrix.shape != (len(cell_ids), len(features)) or min(matrix.shape) == 0:
        raise ValueError("Count matrix must match cell_ids (rows) and features (columns)")
    if len(set(cell_ids)) != len(cell_ids) or any(not x for x in cell_ids):
        raise ValueError("Cell IDs must be nonempty and unique")
    if not {"cell_id", "centroid_x", "centroid_y"}.issubset(metadata):
        raise ValueError("Cell metadata needs cell_id, centroid_x, centroid_y")
    if metadata.cell_id.duplicated().any() or set(metadata.cell_id) != set(cell_ids):
        raise ValueError("Metadata must contain exactly one row for every count-matrix cell")
    if features.gene_name.duplicated().any() or (features.gene_name == "").any():
        raise ValueError("Use unique, nonempty gene_name values in features.tsv.gz")
    if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
        raise ValueError("Gene counts must be finite and nonnegative")
    metadata = metadata.set_index("cell_id").loc[cell_ids].reset_index()
    xy = metadata[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    if not np.isfinite(xy).all():
        raise ValueError("Cell centroids must be finite full-resolution x,y pixels")
    return matrix, cell_ids, features, metadata


def choose_spatial_coordinates(metadata, image_width, image_height):
    # Use the public preprocessing contract; do not guess whether axes are swapped.
    metadata = metadata.copy()
    xy = metadata[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    keep = np.isfinite(xy).all(axis=1)
    keep &= (xy[:, 0] >= 0) & (xy[:, 0] < image_width)
    keep &= (xy[:, 1] >= 0) & (xy[:, 1] < image_height)
    metadata["image_x"], metadata["image_y"] = xy[:, 0], xy[:, 1]
    return metadata, keep, "centroid_xy"


def load_truth(directory, cell_ids, gene_names):
    matrix, all_ids, features, _ = load_single_cell_inputs(directory)
    rows = {name: i for i, name in enumerate(all_ids)}
    cols = {name: i for i, name in enumerate(features.gene_name)}
    return matrix[[rows[x] for x in cell_ids]][:, [cols[x] for x in gene_names]].toarray().astype(np.float32)


def read_genes(path, available=None):
    genes = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if len(genes) < 2 or len(set(genes)) != len(genes):
        raise ValueError("gene_list must contain at least two unique gene names, one per line")
    if any('/' in x or '\\' in x or '\x00' in x or x in {'.', '..'} for x in genes):
        raise ValueError("Gene names must also be safe filenames for iSTAR outputs")
    if available is not None and set(genes) - set(available):
        raise ValueError(f"Genes absent from features.tsv.gz: {sorted(set(genes) - set(available))[:10]}")
    return genes


def compute_metrics(predicted, truth, gene_names):
    if predicted.shape != truth.shape or predicted.ndim != 2 or min(predicted.shape) == 0:
        raise ValueError("Prediction and truth must be matching, nonempty cells-by-genes arrays")
    if predicted.shape[1] != len(gene_names):
        raise ValueError("Gene names do not match the prediction columns")
    if not np.isfinite(predicted).all() or not np.isfinite(truth).all():
        raise ValueError("Nonfinite predictions/truth cannot be scored")
    if (predicted < 0).any() or (truth < 0).any():
        raise ValueError("MSE/log1p MSE expect nonnegative counts")
    squared = (predicted - truth) ** 2
    log_squared = (np.log1p(predicted) - np.log1p(truth)) ** 2
    metrics = {"mse": float(squared.mean()), "log1p_mse": float(log_squared.mean()),
               "n_query_cells": len(truth), "n_genes": len(gene_names)}
    per_gene = pd.DataFrame({"gene_name": gene_names, "mse": squared.mean(axis=0),
                             "log1p_mse": log_squared.mean(axis=0)}).sort_values("mse", ascending=False)
    return metrics, per_gene
