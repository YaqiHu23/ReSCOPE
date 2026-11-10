"""Synthetic fixture for installation and workflow checks; not scientific data."""
from pathlib import Path
import csv
import gzip
import json


def create_demo(directory):
    import numpy as np
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    embeddings = directory / "embeddings"
    cells = directory / "cells"
    embeddings.mkdir()
    cells.mkdir()
    rng = np.random.default_rng(42)
    side, channels = 12, 8
    np.save(embeddings / "embeddings.npy", rng.normal(size=(side * side, channels)).astype(np.float32))
    with (embeddings / "meta.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", "col"])
        writer.writerows((row, col) for row in range(side) for col in range(side))
    with gzip.open(cells / "cell_metadata.csv.gz", "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cell_id", "centroid_x", "centroid_y", "cell_type"])
        writer.writerows((f"cell_{row}_{col}", col + 0.5, row + 0.5, [1, 2, 3, 5][(row + col) % 4])
                         for row in range(side) for col in range(side))
    config = {"data": {"embedding_dir": "embeddings", "single_cell_dir": "cells", "feature_dim": channels,
                       "tile_um": 1.0, "mpp": 1.0},
              "model": {"base_roi_tiles": 2, "base_stride_tiles": 2, "prototype_grid_size": 2},
              "train": {"epochs": 1, "device": "cpu", "amp": False, "num_rois": None},
              "runtime": {"threads": 1},
              "selection": {"min_cells": 1, "projection_dim": 4, "n_clusters": 4, "candidate_pool_size": 36,
                            "target_reference_cells": 20.0, "quota_reference_cells": 20.0,
                            "min_reference_fraction": 0.0, "enforce_celltype_upper_bounds": False}}
    path = directory / "config.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    return path
