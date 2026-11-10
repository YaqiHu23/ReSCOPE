import csv
import os
import re

import numpy as np
import torch

from .geometry import compute_region_bounds


TILE_COORD_PATTERN = re.compile(r"_(\d+)_(\d+)_")


class OnlyVisiumHDRegionDataset(torch.utils.data.Dataset):
    def __init__(self, path, region_cfg, grid_shape=None):
        self.path = path
        self.region_cfg = region_cfg
        self.file_list = self._collect_embedding_files()
        self.meta_path = self._resolve_meta_path()
        self.rows, self.cols = self._load_tile_coords()
        self.full_h = int(self.rows.max()) + 1
        self.full_w = int(self.cols.max()) + 1
        if grid_shape is not None:
            if grid_shape[0] < self.full_h or grid_shape[1] < self.full_w:
                raise ValueError("data.grid_shape does not contain all embedding coordinates")
            self.full_h, self.full_w = map(int, grid_shape)
        self.region_bounds = compute_region_bounds(self.full_h, self.full_w, region_cfg)

    def _collect_embedding_files(self):
        candidates = []
        for name in os.listdir(self.path):
            full_path = os.path.join(self.path, name)
            if not os.path.isfile(full_path):
                continue
            if not (name.endswith(".npy") or name.endswith(".pt")):
                continue
            candidates.append(name)

        candidates.sort()
        excluded_names = {"keep_mask.npy"}
        embedding_files = [
            name
            for name in candidates
            if name not in excluded_names and "embedding" in os.path.splitext(name)[0].lower()
        ]

        if embedding_files:
            return embedding_files
        if candidates:
            raise ValueError(
                f"No embedding files found in {self.path}. Saw tensor files: {candidates}. "
                "Expected filenames containing 'embedding'."
            )
        raise FileNotFoundError(f"No .npy or .pt files found in {self.path}")

    def _resolve_meta_path(self):
        for name in ("meta.csv", "filtered_meta.csv"):
            meta_path = os.path.join(self.path, name)
            if os.path.exists(meta_path):
                return meta_path
        raise FileNotFoundError(f"Expected meta.csv or filtered_meta.csv in {self.path}")

    def _load_tile_coords(self):
        rows = []
        cols = []
        with open(self.meta_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            fields = set(reader.fieldnames or [])
            explicit = {"row", "col"}.issubset(fields)
            if not explicit and "path" not in fields:
                raise ValueError(f"Expected row,col columns or a legacy path column in {self.meta_path}")
            for item in reader:
                if explicit:
                    row_idx, col_idx = int(item["row"]), int(item["col"])
                    if row_idx < 0 or col_idx < 0:
                        raise ValueError("Tile row/col must be non-negative integers")
                    rows.append(row_idx)
                    cols.append(col_idx)
                    continue
                tile_path = item["path"]
                match = TILE_COORD_PATTERN.findall(tile_path)
                if not match:
                    raise ValueError(f"Could not parse row/col from path: {tile_path}")
                row_idx, col_idx = match[-1]
                rows.append(int(row_idx))
                cols.append(int(col_idx))
        if not rows:
            raise ValueError(f"No tile coordinates in {self.meta_path}")
        return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)

    def __len__(self):
        return len(self.file_list)

    def get_region_info(self):
        return {
            "full_grid_shape": [self.full_h, self.full_w],
            "region_bounds": dict(self.region_bounds),
            "region_grid_shape": [self.region_bounds["height"], self.region_bounds["width"]],
            "region_source": self.region_cfg.get("source", "ratio"),
        }

    def _load_embedding(self, path):
        if path.endswith(".npy"):
            return np.load(path, mmap_mode="r")
        return torch.load(path, map_location="cpu", weights_only=True)

    def __getitem__(self, idx):
        name = self.file_list[idx]
        full_path = os.path.join(self.path, name)
        x = self._load_embedding(full_path)

        top = self.region_bounds["top"]
        bottom = self.region_bounds["bottom"]
        left = self.region_bounds["left"]
        right = self.region_bounds["right"]
        region_h = self.region_bounds["height"]
        region_w = self.region_bounds["width"]

        if isinstance(x, torch.Tensor) and x.ndim == 3:
            cropped = x[:, top:bottom, left:right].float()
            valid_mask = torch.ones((region_h, region_w), dtype=torch.bool)
            return {
                "x": cropped,
                "valid_mask": valid_mask,
                "region_bounds": dict(self.region_bounds),
                "full_grid_shape": torch.tensor([self.full_h, self.full_w], dtype=torch.long),
            }

        if isinstance(x, np.ndarray) and x.ndim == 3:
            cropped = torch.from_numpy(np.asarray(x[:, top:bottom, left:right])).float()
            valid_mask = torch.ones((region_h, region_w), dtype=torch.bool)
            return {
                "x": cropped,
                "valid_mask": valid_mask,
                "region_bounds": dict(self.region_bounds),
                "full_grid_shape": torch.tensor([self.full_h, self.full_w], dtype=torch.long),
            }

        if getattr(x, "ndim", None) != 2:
            raise ValueError(f"Embedding file {name} has unsupported shape {getattr(x, 'shape', None)}")

        if x.shape[0] != len(self.rows):
            raise ValueError(
                f"Embedding length ({x.shape[0]}) does not match metadata length ({len(self.rows)}) in {self.meta_path}"
            )

        inside = (
            (self.rows >= top)
            & (self.rows < bottom)
            & (self.cols >= left)
            & (self.cols < right)
        )
        if not np.any(inside):
            raise ValueError("No tiles found inside the configured Visium HD region")

        region_rows = self.rows[inside] - top
        region_cols = self.cols[inside] - left

        x_region = np.asarray(x[inside], dtype=np.float32)
        feat_dim = int(x_region.shape[1])

        grid = torch.zeros((feat_dim, region_h, region_w), dtype=torch.float32)
        valid_mask = torch.zeros((region_h, region_w), dtype=torch.bool)

        region_rows_t = torch.from_numpy(region_rows.astype(np.int64))
        region_cols_t = torch.from_numpy(region_cols.astype(np.int64))
        grid[:, region_rows_t, region_cols_t] = torch.from_numpy(x_region).t().contiguous()
        valid_mask[region_rows_t, region_cols_t] = True

        return {
            "x": grid,
            "valid_mask": valid_mask,
            "region_bounds": dict(self.region_bounds),
            "full_grid_shape": torch.tensor([self.full_h, self.full_w], dtype=torch.long),
        }
