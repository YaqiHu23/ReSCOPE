from __future__ import annotations

import itertools

import numpy as np

from scipy.sparse import coo_matrix

def build_overlap_matrix(pool: np.ndarray, rows: np.ndarray, cols: np.ndarray, radius: int) -> coo_matrix | None:
    row_ids: list[int] = []
    col_ids: list[int] = []
    data: list[float] = []
    constraint_row = 0
    for a, b in itertools.combinations(range(int(pool.size)), 2):
        ia = int(pool[a])
        ib = int(pool[b])
        if abs(int(rows[ia]) - int(rows[ib])) <= radius and abs(int(cols[ia]) - int(cols[ib])) <= radius:
            row_ids.extend([constraint_row, constraint_row])
            col_ids.extend([a, b])
            data.extend([1.0, 1.0])
            constraint_row += 1
    if constraint_row == 0:
        return None
    return coo_matrix((data, (row_ids, col_ids)), shape=(constraint_row, int(pool.size)))
