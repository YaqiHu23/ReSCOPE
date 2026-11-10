from __future__ import annotations

import argparse

import numpy as np

from .dataset import OnlyVisiumHDRegionDataset
from .geometry import grid_tile_pixels

from .biology import build_adjacency_integrals, build_celltype_integrals, build_feature_integrals, build_projected_region, filter_to_grid, load_cell_metadata, rect_count, rect_counts, rect_feature_sum

from .cnn import build_joint_cluster_labels

def candidate_features(
    *,
    args: argparse.Namespace,
    cfg: dict,
    dataset: OnlyVisiumHDRegionDataset,
    region_info: dict,
    geom: dict,
    cnn_candidate_mask: np.ndarray,
    return_projected_grid: bool = False,
) -> dict:
    region_h, region_w = region_info["region_grid_shape"]
    roi_tiles = int(geom["roi_tiles"])
    stride_tiles = int(geom["stride_tiles"])
    nh = ((int(region_h) - roi_tiles) // stride_tiles) + 1
    nw = ((int(region_w) - roi_tiles) // stride_tiles) + 1
    rows, cols = np.indices((nh, nw), dtype=np.int64)
    rows = rows.reshape(-1)
    cols = cols.reshape(-1)

    metadata = load_cell_metadata(args.single_cell_dir)
    tile_px = grid_tile_pixels(cfg)
    coords, tile_y, tile_x, cell_type = filter_to_grid(
        metadata=metadata,
        full_grid_shape=region_info["full_grid_shape"],
        tile_px=tile_px,
    )
    celltype_labels = sorted(int(label) for label in np.unique(cell_type))
    full_cell_counts = np.asarray([(cell_type == label).sum() for label in celltype_labels], dtype=np.float64)

    region_bounds = region_info["region_bounds"]
    region_mask = (
        (tile_y >= int(region_bounds["top"]))
        & (tile_y < int(region_bounds["bottom"]))
        & (tile_x >= int(region_bounds["left"]))
        & (tile_x < int(region_bounds["right"]))
    )
    cell_integrals = build_celltype_integrals(
        tile_y=tile_y[region_mask] - int(region_bounds["top"]),
        tile_x=tile_x[region_mask] - int(region_bounds["left"]),
        cell_type=cell_type[region_mask],
        labels=celltype_labels,
        grid_shape=region_info["region_grid_shape"],
    )
    pair_labels, edge_integrals, full_edge_counts = build_adjacency_integrals(
        coords=coords,
        cell_type=cell_type,
        labels=celltype_labels,
        region_bounds=region_info["region_bounds"],
        region_grid_shape=region_info["region_grid_shape"],
        tile_px=tile_px,
        knn_k=args.knn_k,
    )
    projected_grid, valid_mask = build_projected_region(
        dataset=dataset,
        cfg=cfg,
        region_info=region_info,
        projection_dim=int(args.projection_dim),
        chunk_size=int(args.chunk_size),
        seed=int(args.seed),
    )
    feature_integral, valid_integral = build_feature_integrals(projected_grid, valid_mask)
    full_uni_mean = projected_grid[:, valid_mask].mean(axis=1).astype(np.float64)

    cell_counts = []
    edge_counts = []
    uni_sums = []
    uni_counts = []
    uni_valid_ratios = []
    for row, col in zip(rows, cols):
        local_top = int(row) * stride_tiles
        local_left = int(col) * stride_tiles
        bottom = min(local_top + roi_tiles, int(region_h))
        right = min(local_left + roi_tiles, int(region_w))
        cell_counts.append(rect_counts(cell_integrals, local_top, local_left, bottom, right))
        edge_counts.append(rect_counts(edge_integrals, local_top, local_left, bottom, right))
        valid_count = rect_count(valid_integral, local_top, local_left, bottom, right)
        uni_counts.append(valid_count)
        uni_valid_ratios.append(valid_count / float(max(roi_tiles * roi_tiles, 1)))
        uni_sums.append(rect_feature_sum(feature_integral, local_top, local_left, bottom, right))

    cell_counts = np.vstack(cell_counts)
    edge_counts = np.vstack(edge_counts)
    uni_sums = np.vstack(uni_sums)
    uni_counts = np.asarray(uni_counts, dtype=np.float64)
    uni_valid_ratios = np.asarray(uni_valid_ratios, dtype=np.float64)
    available_mask = (
        cnn_candidate_mask
        & (cell_counts.sum(axis=1) >= int(args.min_cells))
        & (uni_valid_ratios >= float(args.min_valid_ratio))
    )
    cluster_labels, cluster_weights = build_joint_cluster_labels(
        cell_counts=cell_counts,
        uni_sums=uni_sums,
        uni_counts=uni_counts,
        valid_counts=uni_counts,
        available_seed_mask=available_mask,
        args=args,
    )
    available_mask &= cluster_labels >= 0

    result = {
        "nh": nh,
        "nw": nw,
        "rows": rows,
        "cols": cols,
        "metadata": metadata,
        "celltype_labels": celltype_labels,
        "pair_labels": pair_labels,
        "full_cell_counts": full_cell_counts,
        "full_edge_counts": full_edge_counts,
        "full_uni_mean": full_uni_mean,
        "cell_counts": cell_counts,
        "edge_counts": edge_counts,
        "uni_sums": uni_sums,
        "uni_counts": uni_counts,
        "uni_valid_ratios": uni_valid_ratios,
        "cluster_labels": cluster_labels,
        "cluster_weights": cluster_weights,
        "available_mask": available_mask,
    }
    if return_projected_grid:
        # The architecture-coverage selector consumes the same deterministic
        # projection.  Returning it on request avoids a second full embedding
        # pass while preserving the memory behavior of all existing callers.
        result["_projected_grid"] = projected_grid
        result["_valid_mask"] = valid_mask
    return result
