from __future__ import annotations

import math

from pathlib import Path

import numpy as np

import pandas as pd

from scipy.spatial import cKDTree


from .dataset import OnlyVisiumHDRegionDataset

def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))

def cosine_similarity(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    denom = float(np.linalg.norm(p) * np.linalg.norm(q))
    if denom <= eps:
        return 0.0
    return float(np.dot(p, q) / denom)

def l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-8) -> np.ndarray:
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(denom, eps)

def choose_spatial_coordinates(metadata: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, str]:
    candidates = []
    centroid_x = pd.to_numeric(metadata["centroid_x"], errors="coerce")
    centroid_y = pd.to_numeric(metadata["centroid_y"], errors="coerce")
    candidates.append(("centroid_xy", centroid_x, centroid_y))
    candidates.append(("centroid_yx_swapped", centroid_y, centroid_x))

    bbox_cols = ["bbox_xmin", "bbox_xmax", "bbox_ymin", "bbox_ymax"]
    if all(col in metadata.columns for col in bbox_cols):
        bbox_xmin = pd.to_numeric(metadata["bbox_xmin"], errors="coerce")
        bbox_xmax = pd.to_numeric(metadata["bbox_xmax"], errors="coerce")
        bbox_ymin = pd.to_numeric(metadata["bbox_ymin"], errors="coerce")
        bbox_ymax = pd.to_numeric(metadata["bbox_ymax"], errors="coerce")
        bbox_center_x = (bbox_xmin + bbox_xmax) / 2.0
        bbox_center_y = (bbox_ymin + bbox_ymax) / 2.0
        candidates.append(("bbox_center_xy", bbox_center_x, bbox_center_y))
        candidates.append(("bbox_center_yx_swapped", bbox_center_y, bbox_center_x))

    best_name = ""
    best_keep = None
    best_x = None
    best_y = None
    best_count = -1
    for name, x_coord, y_coord in candidates:
        keep = np.isfinite(x_coord.to_numpy()) & np.isfinite(y_coord.to_numpy())
        if int(keep.sum()) > best_count:
            best_name = name
            best_keep = keep
            best_x = x_coord
            best_y = y_coord
            best_count = int(keep.sum())

    if best_keep is None or best_x is None or best_y is None or best_count <= 0:
        raise RuntimeError("No valid spatial coordinates were found in cell metadata.")

    metadata = metadata.copy()
    metadata["image_x"] = best_x.to_numpy()
    metadata["image_y"] = best_y.to_numpy()
    return metadata, best_keep, best_name



def load_cell_metadata(single_cell_dir: Path) -> pd.DataFrame:
    metadata = pd.read_csv(single_cell_dir / "cell_metadata.csv.gz")
    required = {"centroid_x", "centroid_y", "cell_type"}
    missing = required.difference(metadata.columns)
    if missing:
        raise KeyError(f"{single_cell_dir}/cell_metadata.csv.gz is missing columns: {sorted(missing)}")
    metadata["cell_type_source"] = "annotation"
    metadata, keep, coord_source = choose_spatial_coordinates(metadata)
    metadata = metadata.loc[keep].reset_index(drop=True)
    labels = pd.to_numeric(metadata["cell_type"], errors="raise")
    if not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
        raise ValueError("cell_type must contain finite integer labels")
    metadata["cell_type"] = labels.astype(int)
    metadata.attrs["coord_source"] = coord_source
    return metadata

def filter_to_grid(
    metadata: pd.DataFrame,
    full_grid_shape: list[int],
    tile_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    full_h, full_w = int(full_grid_shape[0]), int(full_grid_shape[1])
    image_x = metadata["image_x"].to_numpy(dtype=np.float64)
    image_y = metadata["image_y"].to_numpy(dtype=np.float64)
    tile_y = np.floor(image_y / tile_px).astype(np.int64)
    tile_x = np.floor(image_x / tile_px).astype(np.int64)
    keep = (tile_y >= 0) & (tile_y < full_h) & (tile_x >= 0) & (tile_x < full_w)
    coords = np.column_stack([image_x[keep], image_y[keep]]).astype(np.float64)
    cell_type = metadata["cell_type"].to_numpy(dtype=np.int16)[keep]
    return coords, tile_y[keep], tile_x[keep], cell_type

def build_celltype_integrals(
    tile_y: np.ndarray,
    tile_x: np.ndarray,
    cell_type: np.ndarray,
    labels: list[int],
    grid_shape: list[int],
) -> np.ndarray:
    grid_h, grid_w = int(grid_shape[0]), int(grid_shape[1])
    integrals = []
    for label in labels:
        grid = np.zeros((grid_h + 1, grid_w + 1), dtype=np.int32)
        mask = cell_type == label
        np.add.at(grid, (tile_y[mask] + 1, tile_x[mask] + 1), 1)
        integral = grid.cumsum(axis=0, dtype=np.int32).cumsum(axis=1, dtype=np.int32)
        integrals.append(integral)
    return np.stack(integrals, axis=0)

def pair_key(a: int, b: int) -> tuple[int, int]:
    a = int(a)
    b = int(b)
    return (a, b) if a <= b else (b, a)

def build_adjacency_integrals(
    coords: np.ndarray,
    cell_type: np.ndarray,
    labels: list[int],
    region_bounds: dict,
    region_grid_shape: list[int],
    tile_px: float,
    knn_k: int,
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    region_h, region_w = int(region_grid_shape[0]), int(region_grid_shape[1])
    region_top = int(region_bounds["top"])
    region_left = int(region_bounds["left"])
    region_bottom = int(region_bounds["bottom"])
    region_right = int(region_bounds["right"])
    pair_labels = [(a, b) for i, a in enumerate(labels) for b in labels[i:]]
    pair_to_idx = {pair: idx for idx, pair in enumerate(pair_labels)}

    if coords.shape[0] < 2:
        raise RuntimeError("Need at least two cells to build adjacency features.")

    effective_k = min(int(knn_k), coords.shape[0] - 1)
    tree = cKDTree(coords)
    try:
        _, idx = tree.query(coords, k=effective_k + 1, workers=1)
    except TypeError:
        _, idx = tree.query(coords, k=effective_k + 1)
    idx = np.asarray(idx)
    if idx.ndim == 1:
        idx = idx.reshape(-1, 1)
    neighbor_idx = idx[:, 1:]

    src_idx = np.repeat(np.arange(coords.shape[0]), effective_k)
    dst_idx = neighbor_idx.reshape(-1)
    type_a = cell_type[src_idx].astype(np.int32)
    type_b = cell_type[dst_idx].astype(np.int32)
    low = np.minimum(type_a, type_b)
    high = np.maximum(type_a, type_b)
    midpoint_x = (coords[src_idx, 0] + coords[dst_idx, 0]) / 2.0
    midpoint_y = (coords[src_idx, 1] + coords[dst_idx, 1]) / 2.0
    pair_indices = np.asarray([pair_to_idx[pair_key(a, b)] for a, b in zip(low, high)], dtype=np.int16)
    full_edge_counts = np.bincount(pair_indices, minlength=len(pair_labels)).astype(np.float64)

    edge_tile_y = np.floor(midpoint_y / tile_px).astype(np.int64)
    edge_tile_x = np.floor(midpoint_x / tile_px).astype(np.int64)
    keep = (
        (edge_tile_y >= region_top)
        & (edge_tile_y < region_bottom)
        & (edge_tile_x >= region_left)
        & (edge_tile_x < region_right)
    )
    edge_tile_y = edge_tile_y[keep] - region_top
    edge_tile_x = edge_tile_x[keep] - region_left
    pair_indices = pair_indices[keep]

    integrals = []
    for pair_idx in range(len(pair_labels)):
        grid = np.zeros((region_h + 1, region_w + 1), dtype=np.int32)
        mask = pair_indices == pair_idx
        np.add.at(grid, (edge_tile_y[mask] + 1, edge_tile_x[mask] + 1), 1)
        integral = grid.cumsum(axis=0, dtype=np.int32).cumsum(axis=1, dtype=np.int32)
        integrals.append(integral)
    return pair_labels, np.stack(integrals, axis=0), full_edge_counts

def build_projected_region(
    dataset: OnlyVisiumHDRegionDataset,
    cfg: dict,
    region_info: dict,
    projection_dim: int,
    chunk_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    embedding_dir = Path(cfg["data"]["embedding_dir"])
    embedding_path = embedding_dir / dataset.file_list[0]
    embeddings = np.load(embedding_path, mmap_mode="r")
    rows = dataset.rows
    cols = dataset.cols
    feat_dim = int(embeddings.shape[1])
    rng = np.random.default_rng(seed)
    projection = rng.normal(0.0, 1.0 / math.sqrt(float(projection_dim)), size=(feat_dim, projection_dim)).astype(
        np.float32
    )
    bounds = region_info["region_bounds"]
    region_h, region_w = region_info["region_grid_shape"]
    inside = (rows >= bounds["top"]) & (rows < bounds["bottom"]) & (cols >= bounds["left"]) & (cols < bounds["right"])
    inside_idx = np.flatnonzero(inside)
    local_rows = (rows[inside_idx] - bounds["top"]).astype(np.int64)
    local_cols = (cols[inside_idx] - bounds["left"]).astype(np.int64)
    projected_grid = np.zeros((projection_dim, region_h, region_w), dtype=np.float32)
    valid_mask = np.zeros((region_h, region_w), dtype=bool)
    for start in range(0, inside_idx.size, chunk_size):
        idx = inside_idx[start : start + chunk_size]
        x = np.asarray(embeddings[idx], dtype=np.float32)
        x = l2_normalize(x, axis=1)
        projected = l2_normalize(x @ projection, axis=1)
        rr = local_rows[start : start + idx.size]
        cc = local_cols[start : start + idx.size]
        projected_grid[:, rr, cc] = projected.T
        valid_mask[rr, cc] = True
    return projected_grid, valid_mask

def build_feature_integrals(projected_grid: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    feature_integral = projected_grid.cumsum(axis=1, dtype=np.float32).cumsum(axis=2, dtype=np.float32)
    feature_integral = np.pad(feature_integral, ((0, 0), (1, 0), (1, 0)), mode="constant")
    valid_grid = valid_mask.astype(np.int32)
    valid_integral = valid_grid.cumsum(axis=0, dtype=np.int32).cumsum(axis=1, dtype=np.int32)
    valid_integral = np.pad(valid_integral, ((1, 0), (1, 0)), mode="constant")
    return feature_integral, valid_integral

def rect_counts(integrals: np.ndarray, top: int, left: int, bottom: int, right: int) -> np.ndarray:
    return (
        integrals[:, bottom, right]
        - integrals[:, top, right]
        - integrals[:, bottom, left]
        + integrals[:, top, left]
    ).astype(np.float64)

def rect_feature_sum(integrals: np.ndarray, top: int, left: int, bottom: int, right: int) -> np.ndarray:
    return (
        integrals[:, bottom, right]
        - integrals[:, top, right]
        - integrals[:, bottom, left]
        + integrals[:, top, left]
    ).astype(np.float64)

def rect_count(integral: np.ndarray, top: int, left: int, bottom: int, right: int) -> float:
    return float(integral[bottom, right] - integral[top, right] - integral[bottom, left] + integral[top, left])

def rare_type_coverage(union_counts: np.ndarray, full_counts: np.ndarray) -> float:
    expected = full_counts / max(float(full_counts.sum()), 1.0)
    expected_cells = expected * max(float(union_counts.sum()), 1.0)
    ratio = np.divide(
        union_counts,
        np.maximum(expected_cells, 1.0),
        out=np.zeros_like(union_counts, dtype=np.float64),
        where=expected_cells > 0,
    )
    return float(np.minimum(ratio, 1.0).mean())

def absolute_celltype_quota(
    union_counts: np.ndarray,
    full_counts: np.ndarray,
    quota_reference_cells: float,
) -> float:
    full_dist = distribution(full_counts)
    target = np.maximum(full_dist * max(float(quota_reference_cells), 1.0), 1.0)
    ratio = np.divide(
        union_counts,
        target,
        out=np.zeros_like(union_counts, dtype=np.float64),
        where=target > 0,
    )
    return float(np.minimum(ratio, 1.0).mean())

def adjacency_boundary_score(edge_counts: np.ndarray, pair_labels: list[tuple[int, int]]) -> float:
    total = float(np.sum(edge_counts))
    if total <= 0:
        return 0.0
    fractions = {pair: float(count) / total for pair, count in zip(pair_labels, edge_counts)}
    tumor_stroma = fractions.get(pair_key(1, 3), 0.0)
    immune_tumor = fractions.get(pair_key(1, 2), 0.0)
    epithelial_stroma = fractions.get(pair_key(5, 3), 0.0)
    stromal_immune = fractions.get(pair_key(3, 2), 0.0)
    return float(tumor_stroma + 0.75 * immune_tumor + 0.5 * epithelial_stroma + 0.25 * stromal_immune)

def distribution(values: np.ndarray) -> np.ndarray:
    total = float(np.sum(values))
    if total <= 0:
        return np.zeros_like(values, dtype=np.float64)
    return np.asarray(values, dtype=np.float64) / total
