from __future__ import annotations

import argparse

import json

import numpy as np

from .biology import absolute_celltype_quota, adjacency_boundary_score, cosine_similarity, distribution, js_divergence, rare_type_coverage

from .cnn import zscore

from .geometry import roi_index_to_global_tile_box, tile_box_to_image_box

def single_roi_priority(cnn_logits: np.ndarray, feat: dict, weights: dict[str, float], args: argparse.Namespace) -> np.ndarray:
    available = feat["available_mask"]
    cnn_z = zscore(np.where(available, cnn_logits, np.nan))
    density_z = zscore(np.where(available, np.log1p(feat["cell_counts"].sum(axis=1)), np.nan))
    boundary_z = zscore(
        np.where(
            available,
            np.asarray([adjacency_boundary_score(row, feat["pair_labels"]) for row in feat["edge_counts"]]),
            np.nan,
        )
    )
    cluster_weight_z = zscore(feat["cluster_weights"])
    cluster = np.zeros_like(cnn_z)
    valid = feat["cluster_labels"] >= 0
    cluster[valid] = cluster_weight_z[feat["cluster_labels"][valid]]

    full_cell_dist = distribution(feat["full_cell_counts"])
    full_edge_dist = distribution(feat["full_edge_counts"])
    target_reference = max(float(args.target_reference_cells), 1.0)
    quota_reference = max(float(args.quota_reference_cells), 1.0)
    priority = np.full_like(cnn_z, -np.inf, dtype=np.float64)

    for idx in np.flatnonzero(available):
        cells = feat["cell_counts"][idx]
        edges = feat["edge_counts"][idx]
        uni_mean = feat["uni_sums"][idx] / max(float(feat["uni_counts"][idx]), 1.0)
        priority[idx] = (
            weights["cnn"] * cnn_z[idx]
            + weights["cluster"] * cluster[idx]
            - weights["adjacency"] * js_divergence(distribution(edges), full_edge_dist)
            - weights["composition"] * js_divergence(distribution(cells), full_cell_dist)
            + weights["coverage"] * rare_type_coverage(cells, feat["full_cell_counts"])
            + weights["density"] * density_z[idx]
            + weights["boundary"] * boundary_z[idx]
            + weights["uni_represent"] * cosine_similarity(uni_mean, feat["full_uni_mean"])
            + weights["reference_budget"] * min(float(cells.sum()) / target_reference, 1.0)
            + weights["celltype_quota"] * absolute_celltype_quota(cells, feat["full_cell_counts"], quota_reference)
        )
    priority[available] = zscore(priority[available])
    return priority

def write_metadata(
    *,
    args: argparse.Namespace,
    cfg: dict,
    geom: dict,
    region_info: dict,
    feat: dict,
    weights: dict[str, float],
    selected_indices: list[int],
    projection_report: dict,
) -> None:
    roi_prefix = getattr(args, "roi_prefix", "exact_topm_bioprior")
    selection_mode_name = getattr(args, "selection_mode_name", "exact_topm_bioprior")
    phase_name = getattr(args, "phase_name", "scheme2_topm_exact_set_search_iter06_iter09_priors")
    report_key = getattr(args, "search_report_key", "exact_topm_search")
    selected_cell_counts = feat["cell_counts"][selected_indices].sum(axis=0)
    selected_edge_counts = feat["edge_counts"][selected_indices].sum(axis=0)
    selected_uni_sum = feat["uni_sums"][selected_indices].sum(axis=0)
    selected_uni_count = float(feat["uni_counts"][selected_indices].sum())
    selected_uni_mean = selected_uni_sum / max(selected_uni_count, 1.0)
    model_rois = []
    for idx, roi_index in enumerate(selected_indices, start=1):
        tile_box = roi_index_to_global_tile_box(
            roi_index=int(roi_index),
            nw=int(feat["nw"]),
            stride_tiles=int(geom["stride_tiles"]),
            roi_tiles=int(geom["roi_tiles"]),
            region_bounds=region_info["region_bounds"],
        )
        image_box = tile_box_to_image_box(tile_box, cfg["data"]["tile_um"], cfg["data"]["mpp"], cfg["data"].get("tile_px"))
        # Tile-to-image conversion can round a matched physical ROI down by one
        # pixel (for example, 3,887 px -> 3,886 px in colon_P3).  The tile box
        # still defines the searched location, while this optional export-only
        # value preserves the exact size used by the matched baselines.
        exact_size_px = cfg.get("eval", {}).get("exact_roi_size_px")
        if exact_size_px is not None:
            image_box["size_px"] = int(exact_size_px)
        image_box["right_px"] = float(image_box["left_px"] + image_box["size_px"])
        image_box["bottom_px"] = float(image_box["top_px"] + image_box["size_px"])
        image_box["roi_name"] = f"{roi_prefix}_roi_{idx}"
        model_rois.append(image_box)

    output = {
        "selection_mode": selection_mode_name,
        "base_selection_mode": cfg["model"].get("selection_mode", "combinatorial_thretopk"),
        "unified_model_phase": phase_name,
        "checkpoint": str(args.ckpt.resolve()),
        "checkpoint_role": "user-selected ROI scorer",
        "config": str(args.config.resolve()),
        "model_geometry": geom,
        "region_info": region_info,
        "roi_grid_shape": {"nh": int(feat["nh"]), "nw": int(feat["nw"])},
        "num_rois_requested": int(len(selected_indices)),
        "model_roi_indices": [int(idx) for idx in selected_indices],
        "model_rois": model_rois,
        "celltype_labels": feat["celltype_labels"],
        "adjacency_pair_labels": [[int(a), int(b)] for a, b in feat["pair_labels"]],
        "full_slide_celltype_counts": feat["full_cell_counts"].astype(int).tolist(),
        "selected_celltype_counts": selected_cell_counts.astype(int).tolist(),
        "full_slide_adjacency_edge_counts": feat["full_edge_counts"].astype(int).tolist(),
        "selected_adjacency_edge_counts": selected_edge_counts.astype(int).tolist(),
        "selected_vs_full_celltype_js": js_divergence(distribution(selected_cell_counts), distribution(feat["full_cell_counts"])),
        "selected_vs_full_adjacency_js": js_divergence(distribution(selected_edge_counts), distribution(feat["full_edge_counts"])),
        "selected_vs_full_adjacency_cosine": cosine_similarity(distribution(selected_edge_counts), distribution(feat["full_edge_counts"])),
        "selected_rare_type_coverage": rare_type_coverage(selected_cell_counts, feat["full_cell_counts"]),
        "selected_absolute_celltype_quota": absolute_celltype_quota(
            selected_cell_counts,
            feat["full_cell_counts"],
            float(args.quota_reference_cells),
        ),
        "selected_reference_cell_count": int(selected_cell_counts.sum()),
        "selected_reference_budget_fraction": float(
            min(selected_cell_counts.sum() / max(float(args.target_reference_cells), 1.0), 1.0)
        ),
        "selected_vs_full_uni_cosine": cosine_similarity(selected_uni_mean, feat["full_uni_mean"]),
        "selected_uni_valid_ratio": selected_uni_count / float(max(len(selected_indices) * int(geom["roi_tiles"]) * int(geom["roi_tiles"]), 1)),
        "coordinate_source": feat["metadata"].attrs.get("coord_source", "unknown"),
        "knn_k": int(args.knn_k),
        "n_clusters": int(feat["cluster_weights"].size),
        "cluster_weights": feat["cluster_weights"].tolist(),
        report_key: {
            "adaptive_weights": bool(args.adaptive_weights),
            "effective_weights": weights,
            "projection_report": projection_report,
        },
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    outpath = args.outdir / "visium_hd_region_eval_metadata.json"
    outpath.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved exact top-M bio-prior ROI metadata to {outpath}")
    print(f"Selected ROI indices: {selected_indices}")
    print(f"Projection report: {projection_report}")
