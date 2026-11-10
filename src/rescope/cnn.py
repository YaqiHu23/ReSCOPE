from __future__ import annotations

import argparse

from pathlib import Path

import numpy as np

import torch

import torch.nn.functional as F

from .dataset import OnlyVisiumHDRegionDataset

from .factory import build_model

from .clustering import simple_kmeans

def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    out = np.zeros_like(values, dtype=np.float64)
    if finite.sum() < 2:
        return out
    mu = values[finite].mean()
    sigma = values[finite].std()
    if sigma <= 0:
        return out
    out[finite] = (values[finite] - mu) / sigma
    return out

def l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-8) -> np.ndarray:
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(denom, eps)

def load_cnn_logits(
    cfg: dict,
    dataset: OnlyVisiumHDRegionDataset,
    region_info: dict,
    ckpt: Path,
) -> tuple[np.ndarray, np.ndarray, dict]:
    sample = dataset[0]
    x = sample["x"].unsqueeze(0).to(cfg["train"]["device"])
    valid_mask = sample["valid_mask"].unsqueeze(0).to(cfg["train"]["device"])
    model, geom = build_model(cfg, region_info)
    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=True)
    state_dict = dict(checkpoint["model_state_dict"])
    model_state = model.state_dict()
    adapted: list[dict] = []
    for key, value in list(state_dict.items()):
        if key not in model_state or tuple(value.shape) == tuple(model_state[key].shape):
            continue
        target = model_state[key]
        if value.ndim == 4 and target.ndim == 4 and value.shape[:2] == target.shape[:2]:
            state_dict[key] = F.interpolate(
                value.float(),
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).to(dtype=target.dtype, device=target.device)
            adapted.append(
                {
                    "key": key,
                    "source_shape": list(value.shape),
                    "target_shape": list(target.shape),
                    "method": "bilinear_spatial_resize",
                }
            )
            continue
        raise RuntimeError(
            f"Checkpoint tensor {key} has incompatible shape {tuple(value.shape)} "
            f"for model shape {tuple(target.shape)}."
        )
    if adapted:
        print(f"[INFO] Adapted checkpoint tensors for current ROI geometry: {adapted}")
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        logits_map = model.scorer(x)
        candidate_mask = model.compute_roi_candidate_mask(valid_mask)
    logits = logits_map.flatten().detach().float().cpu().numpy()
    mask = candidate_mask.flatten().detach().cpu().numpy().astype(bool)
    return logits, mask, geom

def build_joint_cluster_labels(
    cell_counts: np.ndarray,
    uni_sums: np.ndarray,
    uni_counts: np.ndarray,
    valid_counts: np.ndarray,
    available_seed_mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    cell_totals = np.maximum(cell_counts.sum(axis=1, keepdims=True), 1.0)
    cell_features = (cell_counts / cell_totals).astype(np.float32)
    uni_features = uni_sums / np.maximum(uni_counts[:, None], 1.0)
    uni_features = l2_normalize(uni_features.astype(np.float32), axis=1)
    joint = np.concatenate(
        [
            float(args.cell_feature_weight) * cell_features,
            float(args.uni_feature_weight) * uni_features,
        ],
        axis=1,
    )
    available_idx = np.flatnonzero(available_seed_mask)
    if available_idx.size == 0:
        raise RuntimeError("No candidates available for joint CellViT/UNI clustering.")
    labels_available, _ = simple_kmeans(
        zscore(joint[available_idx]).astype(np.float32),
        n_clusters=int(args.n_clusters),
        n_iters=int(args.kmeans_iters),
        seed=int(args.seed) + 11,
    )
    labels = np.full(cell_counts.shape[0], -1, dtype=np.int64)
    labels[available_idx] = labels_available
    cluster_weights = np.zeros(int(labels_available.max()) + 1, dtype=np.float64)
    for cluster_id in range(cluster_weights.size):
        members = available_idx[labels_available == cluster_id]
        cluster_weights[cluster_id] = float(valid_counts[members].sum())
    return labels, cluster_weights
