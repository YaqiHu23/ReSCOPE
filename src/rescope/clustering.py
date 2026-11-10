from __future__ import annotations

import numpy as np

def simple_kmeans(x: np.ndarray, n_clusters: int, n_iters: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    n_clusters = min(int(n_clusters), n)
    centers = x[rng.choice(n, size=n_clusters, replace=False)].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(int(n_iters)):
        dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(dist, axis=1)
        for k in range(n_clusters):
            members = labels == k
            if members.any():
                centers[k] = x[members].mean(axis=0)
            else:
                centers[k] = x[rng.integers(0, n)]
    return labels, centers
