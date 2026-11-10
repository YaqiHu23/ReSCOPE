"""Reconstruct raw reference expression and aggregate BLEEP neighbor predictions."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import load_single_cell_inputs


def load_reference_context(run_dir):
    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    cells = pd.read_csv(run_dir / "selected_cells.csv", dtype={"cell_id": str})
    genes = (run_dir / "selected_gene_names.txt").read_text().splitlines()
    return metadata, cells, genes


def rebuild_reference_matrix(run_metadata, selected_cells, selected_gene_names):
    matrix, ids, features, _ = load_single_cell_inputs(Path(run_metadata["single_cell_dir"]))
    rows = {name: i for i, name in enumerate(ids)}
    cols = {name: i for i, name in enumerate(features.gene_name)}
    return matrix[[rows[x] for x in selected_cells.cell_id]][:, [cols[x] for x in selected_gene_names]]


def aggregate_predictions(reference_expr, topk_indices, topk_scores, gene_indices, method):
    if method not in {"simple", "average", "weighted_average"}:
        raise ValueError("Unknown BLEEP neighbor aggregation")
    reference = reference_expr[:, gene_indices].toarray().astype(np.float32, copy=False)
    if method == "simple":
        return reference[topk_indices[:, 0]]
    result = np.empty((len(topk_indices), len(gene_indices)), dtype=np.float32)
    # Bound the temporary neighbors tensor instead of allocating query x k x genes.
    for start in range(0, len(result), 64):
        end = min(start + 64, len(result))
        values = reference[topk_indices[start:end]]
        if method == "average":
            result[start:end] = values.mean(axis=1)
        else:
            scores = topk_scores[start:end]
            weights = np.exp(scores - scores.max(axis=1, keepdims=True))
            weights /= weights.sum(axis=1, keepdims=True)
            result[start:end] = np.einsum("qk,qkg->qg", weights, values)
    return result
