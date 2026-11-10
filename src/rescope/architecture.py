from __future__ import annotations

import math

import numpy as np

from .biology import build_feature_integrals, rect_count, rect_feature_sum

EPS = 1e-8

def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, EPS)

def deterministic_spherical_kmeans(
    points: np.ndarray,
    n_clusters: int,
    n_iters: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic spherical k-means used only for morphology states."""
    points = normalize_rows(points)
    if points.shape[0] == 0:
        raise ValueError("Cannot cluster an empty morphology sample.")
    k = min(max(int(n_clusters), 1), int(points.shape[0]))
    rng = np.random.default_rng(int(seed))

    centers = np.empty((k, points.shape[1]), dtype=np.float64)
    first = int(np.argmax(np.linalg.norm(points - points.mean(axis=0), axis=1)))
    centers[0] = points[first]
    closest_distance = 1.0 - np.clip(points @ centers[0], -1.0, 1.0)
    for center_idx in range(1, k):
        probability = np.maximum(closest_distance, 0.0) ** 2
        total = float(probability.sum())
        if total <= EPS:
            chosen = int(rng.integers(0, points.shape[0]))
        else:
            chosen = int(rng.choice(points.shape[0], p=probability / total))
        centers[center_idx] = points[chosen]
        distance = 1.0 - np.clip(points @ centers[center_idx], -1.0, 1.0)
        closest_distance = np.minimum(closest_distance, distance)

    labels = np.zeros(points.shape[0], dtype=np.int32)
    for _ in range(max(int(n_iters), 1)):
        new_labels = np.argmax(points @ centers.T, axis=1).astype(np.int32)
        new_centers = centers.copy()
        for center_idx in range(k):
            members = points[new_labels == center_idx]
            if members.size:
                new_centers[center_idx] = normalize_rows(members.mean(axis=0, keepdims=True))[0]
        if np.array_equal(labels, new_labels) and np.allclose(centers, new_centers, atol=1e-7):
            labels = new_labels
            centers = new_centers
            break
        labels = new_labels
        centers = new_centers
    return centers.astype(np.float32), labels

def _local_roi_prototypes(
    *,
    feature_integral: np.ndarray,
    valid_integral: np.ndarray,
    top: int,
    left: int,
    roi_tiles: int,
    grid_size: int,
    region_shape: tuple[int, int],
) -> np.ndarray:
    region_h, region_w = region_shape
    bottom = min(int(top) + int(roi_tiles), int(region_h))
    right = min(int(left) + int(roi_tiles), int(region_w))
    row_edges = np.linspace(int(top), bottom, int(grid_size) + 1).round().astype(int)
    col_edges = np.linspace(int(left), right, int(grid_size) + 1).round().astype(int)
    prototypes: list[np.ndarray] = []
    for row_idx in range(int(grid_size)):
        for col_idx in range(int(grid_size)):
            row0, row1 = int(row_edges[row_idx]), int(row_edges[row_idx + 1])
            col0, col1 = int(col_edges[col_idx]), int(col_edges[col_idx + 1])
            if row1 <= row0 or col1 <= col0:
                continue
            count = rect_count(valid_integral, row0, col0, row1, col1)
            if count <= 0:
                continue
            mean = rect_feature_sum(feature_integral, row0, col0, row1, col1) / count
            norm = float(np.linalg.norm(mean))
            if np.isfinite(norm) and norm > EPS:
                prototypes.append(mean / norm)
    if not prototypes:
        return np.empty((0, feature_integral.shape[0]), dtype=np.float64)
    return np.vstack(prototypes)

def build_architecture_coverage_features(
    *,
    projected_grid: np.ndarray,
    valid_mask: np.ndarray,
    feat: dict,
    roi_tiles: int,
    stride_tiles: int,
    n_centroids: int,
    local_grid_size: int,
    kmeans_sample_size: int,
    kmeans_iters: int,
    seed: int,
) -> dict:
    """Return candidate-by-state coverage and the tissue state weights."""
    valid_points = projected_grid[:, valid_mask].T.astype(np.float64, copy=False)
    norms = np.linalg.norm(valid_points, axis=1)
    valid_points = valid_points[np.isfinite(norms) & (norms > EPS)]
    if valid_points.shape[0] == 0:
        raise RuntimeError("No valid UNI points are available for architecture coverage.")
    valid_points = normalize_rows(valid_points)

    rng = np.random.default_rng(int(seed))
    sample_n = min(int(kmeans_sample_size), int(valid_points.shape[0]))
    if sample_n < valid_points.shape[0]:
        sample_idx = np.sort(rng.choice(valid_points.shape[0], size=sample_n, replace=False))
        sample = valid_points[sample_idx]
    else:
        sample = valid_points
    centers, labels = deterministic_spherical_kmeans(
        sample,
        n_clusters=int(n_centroids),
        n_iters=int(kmeans_iters),
        seed=int(seed),
    )
    state_weights = np.bincount(labels, minlength=centers.shape[0]).astype(np.float64)
    state_weights /= max(float(state_weights.sum()), 1.0)

    feature_integral, valid_integral = build_feature_integrals(projected_grid, valid_mask)
    signatures = np.full((len(feat["rows"]), centers.shape[0]), -1.0, dtype=np.float32)
    region_shape = (int(projected_grid.shape[1]), int(projected_grid.shape[2]))
    for idx, (row, col) in enumerate(zip(feat["rows"], feat["cols"])):
        if not bool(feat["available_mask"][idx]):
            continue
        top = int(row) * int(stride_tiles)
        left = int(col) * int(stride_tiles)
        prototypes = _local_roi_prototypes(
            feature_integral=feature_integral,
            valid_integral=valid_integral,
            top=top,
            left=left,
            roi_tiles=int(roi_tiles),
            grid_size=int(local_grid_size),
            region_shape=region_shape,
        )
        if prototypes.shape[0] == 0:
            continue
        signatures[idx] = np.max(prototypes @ centers.T, axis=0).astype(np.float32)

    raw = signatures.astype(np.float64) @ state_weights
    available = np.asarray(feat["available_mask"], dtype=bool)
    finite = available & np.isfinite(raw)
    if not finite.any():
        raise RuntimeError("Architecture coverage produced no finite candidate scores.")
    mean = float(raw[finite].mean())
    std = float(raw[finite].std())
    if std <= EPS:
        std = 1.0
    z = (raw - mean) / std
    z[~finite] = -np.inf
    return {
        "architecture_signatures": signatures,
        "architecture_state_weights": state_weights,
        "architecture_state_centers": centers,
        "architecture_candidate_raw": raw,
        "architecture_candidate_z": z,
        "architecture_candidate_mean": mean,
        "architecture_candidate_std": std,
        "architecture_valid_tiles": int(valid_points.shape[0]),
        "architecture_kmeans_sample": int(sample.shape[0]),
        "architecture_local_prototypes": int(math.pow(int(local_grid_size), 2)),
    }
