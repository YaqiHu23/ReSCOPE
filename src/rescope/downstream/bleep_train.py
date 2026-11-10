#!/usr/bin/env python3

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


from .external import repository
BLEEP_DIR = repository("bleep")
sys.path.insert(0, str(BLEEP_DIR))
from models import CLIPModel
from utils import AvgMeter, get_lr
from .common import load_single_cell_inputs, choose_spatial_coordinates


Image.MAX_IMAGE_PIXELS = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run BLEEP on CellViT-derived single-cell Visium HD data."
    )
    parser.add_argument(
        "--single-cell-dir",
        type=Path,
        required=True,
        help="Directory with cell_gene_counts.npz, cell_ids.tsv.gz, features.tsv.gz, cell_metadata.csv.gz.",
    )
    parser.add_argument(
        "--histology-image",
        type=Path,
        required=True,
        help="Path to full-resolution histology image (tif).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output directory.",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        choices=["resnet50"],
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=224,
        help="Image patch size in pixels. BLEEP uses 224 by default.",
    )
    parser.add_argument(
        "--gene-mode",
        type=str,
        default="top",
        choices=["all", "top"],
        help="Use all variable genes or top variable genes before SVD.",
    )
    parser.add_argument("--n-top-genes", type=int, default=2000)
    parser.add_argument(
        "--gene-list",
        type=Path,
        default=None,
        help="Optional text file with one gene name per line. Overrides --gene-mode/--n-top-genes.",
    )
    parser.add_argument(
        "--svd-components",
        type=int,
        default=512,
        help="Expression embedding dimension used by BLEEP.",
    )
    parser.add_argument("--min-total-counts", type=float, default=1.0)
    parser.add_argument("--min-genes-detected", type=float, default=1.0)
    parser.add_argument(
        "--max-cells",
        type=int,
        default=80000,
        help="Optional cap on number of cells (highest counts kept).",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sparse_column_variance(matrix):
    mean = np.asarray(matrix.mean(axis=0)).ravel()
    mean_sq = np.asarray(matrix.power(2).mean(axis=0)).ravel()
    var = mean_sq - mean**2
    var[var < 0] = 0
    return var


def filter_cells(matrix, cell_ids, metadata, min_total_counts, min_genes_detected, max_cells):
    def numeric_metadata_series(column_name):
        if column_name not in metadata.columns:
            return pd.Series(np.nan, index=metadata.index, dtype="float64")
        values = metadata.loc[:, column_name]
        if isinstance(values, pd.DataFrame):
            values = values.iloc[:, 0]
        if np.isscalar(values):
            values = pd.Series(values, index=metadata.index)
        values = pd.Series(values, index=metadata.index)
        return pd.to_numeric(values, errors="coerce")

    total_counts = numeric_metadata_series("total_counts")
    n_genes_detected = numeric_metadata_series("n_genes_detected")

    if total_counts.isna().all():
        total_counts = pd.Series(np.asarray(matrix.sum(axis=1)).ravel(), index=metadata.index)
    else:
        total_counts = total_counts.fillna(0)

    if n_genes_detected.isna().all():
        n_genes_detected = pd.Series(matrix.getnnz(axis=1), index=metadata.index)
    else:
        n_genes_detected = n_genes_detected.fillna(0)

    keep_mask = (
        (total_counts.to_numpy() >= min_total_counts)
        & (n_genes_detected.to_numpy() >= min_genes_detected)
    )
    keep_indices = np.flatnonzero(keep_mask)
    if len(keep_indices) == 0:
        raise RuntimeError("No cells remain after filtering.")

    if max_cells is not None and len(keep_indices) > max_cells:
        ranked = np.argsort(total_counts.to_numpy()[keep_indices])[::-1]
        keep_indices = keep_indices[ranked[:max_cells]]
        keep_indices.sort()

    matrix = matrix[keep_indices]
    metadata = metadata.iloc[keep_indices].reset_index(drop=True)
    cell_ids = [cell_ids[i] for i in keep_indices]
    return matrix, cell_ids, metadata


def filter_cells_in_image_bounds(matrix, cell_ids, metadata, image_width, image_height):
    metadata, keep_mask, coord_source = choose_spatial_coordinates(
        metadata=metadata,
        image_width=image_width,
        image_height=image_height,
    )
    keep_indices = np.flatnonzero(keep_mask)
    if len(keep_indices) == 0:
        raise RuntimeError(
            "No cells have valid coordinates within the histology image bounds."
        )
    matrix = matrix[keep_indices]
    metadata = metadata.iloc[keep_indices].reset_index(drop=True)
    cell_ids = [cell_ids[i] for i in keep_indices]
    dropped = len(keep_mask) - len(keep_indices)
    return matrix, cell_ids, metadata, dropped, coord_source


def select_gene_indices(matrix, gene_names, gene_mode, n_top_genes):
    variance = sparse_column_variance(matrix)
    keep = variance > 0
    if gene_mode == "all":
        indices = np.flatnonzero(keep)
    else:
        order = np.argsort(variance)[::-1]
        order = order[keep[order]]
        indices = order[:n_top_genes]
    if len(indices) == 0:
        raise RuntimeError("No variable genes found.")
    names = [gene_names[i] for i in indices]
    return indices, names


def read_gene_list(gene_list_path):
    with open(gene_list_path, "r", encoding="utf-8") as handle:
        genes = [line.strip().split("\t")[0] for line in handle if line.strip()]
    if not genes:
        raise RuntimeError(f"Gene list is empty: {gene_list_path}")
    return genes


def select_gene_indices_from_list(gene_names, gene_list_path):
    requested = read_gene_list(gene_list_path)
    name_to_index = {}
    for idx, gene_name in enumerate(gene_names):
        if gene_name not in name_to_index:
            name_to_index[gene_name] = idx

    indices = []
    selected_names = []
    missing = []
    seen = set()
    for gene_name in requested:
        if gene_name in seen:
            continue
        seen.add(gene_name)
        idx = name_to_index.get(gene_name)
        if idx is None:
            missing.append(gene_name)
            continue
        indices.append(idx)
        selected_names.append(gene_name)

    if missing:
        preview = ", ".join(missing[:10])
        raise RuntimeError(
            f"{len(missing)} genes from {gene_list_path} were not found in features.tsv.gz: {preview}"
        )
    if not indices:
        raise RuntimeError(f"No genes from {gene_list_path} were found in features.tsv.gz.")

    return np.asarray(indices, dtype=np.int64), selected_names


def build_expression_embedding(matrix, n_components, seed):
    n_samples, n_features = matrix.shape
    max_components = max(1, min(n_components, n_features - 1, n_samples - 1))
    if max_components < 1:
        raise RuntimeError("Not enough samples/features to run TruncatedSVD.")
    svd = TruncatedSVD(n_components=max_components, random_state=seed)
    reduced = svd.fit_transform(matrix).astype(np.float32)
    return reduced, svd, max_components


class SingleCellPatchDataset(Dataset):
    def __init__(self, image_path, metadata, reduced_expression, patch_size, is_train):
        self.image_path = str(image_path)
        self.metadata = metadata.reset_index(drop=True)
        self.reduced_expression = reduced_expression
        self.patch_size = int(patch_size)
        self.half = self.patch_size // 2
        self.is_train = is_train
        self._img = None

    def _get_image(self):
        if self._img is None:
            self._img = Image.open(self.image_path)
        return self._img

    def _crop_with_padding(self, cx, cy):
        img = self._get_image()
        left, upper = int(round(cx)) - self.half, int(round(cy)) - self.half
        right, lower = left + self.patch_size, upper + self.patch_size

        pad_left = max(0, -left)
        pad_top = max(0, -upper)
        pad_right = max(0, right - img.width)
        pad_bottom = max(0, lower - img.height)

        crop_left = max(0, left)
        crop_top = max(0, upper)
        crop_right = min(img.width, right)
        crop_bottom = min(img.height, lower)

        # If a centroid is outside the image after preprocessing, return a blank patch.
        if crop_right <= crop_left or crop_bottom <= crop_top:
            return Image.new("RGB", (self.patch_size, self.patch_size), color=(0, 0, 0))

        patch = img.crop((crop_left, crop_top, crop_right, crop_bottom)).convert("RGB")
        if any(v > 0 for v in [pad_left, pad_top, pad_right, pad_bottom]):
            canvas = Image.new("RGB", (self.patch_size, self.patch_size), color=(0, 0, 0))
            canvas.paste(patch, (pad_left, pad_top))
            patch = canvas
        return patch

    def _transform(self, image):
        if self.is_train:
            if random.random() > 0.5:
                image = TF.hflip(image)
            if random.random() > 0.5:
                image = TF.vflip(image)
            angle = random.choice([0, 90, 180, 270])
            image = TF.rotate(image, angle)

        image = TF.to_tensor(image)
        image = TF.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        return image.float()

    def __getitem__(self, idx):
        x = float(self.metadata.loc[idx, "image_x"])
        y = float(self.metadata.loc[idx, "image_y"])
        patch = self._crop_with_padding(x, y)
        patch = self._transform(patch)
        return {
            "image": patch,
            "reduced_expression": torch.from_numpy(self.reduced_expression[idx]),
            "cell_id": self.metadata.loc[idx, "cell_id"],
            "spatial_coords": torch.tensor([x, y], dtype=torch.float32),
        }

    def __len__(self):
        return len(self.metadata)


def choose_model(name, spot_embedding):
    if name != "resnet50":
        raise ValueError("The paper adapter uses the BLEEP ResNet50 encoder")
    return CLIPModel(spot_embedding=spot_embedding)


def evaluate(model, loader, device):
    meter = AvgMeter("val_loss")
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader), desc="Validation", leave=False):
            batch = {
                "image": batch["image"].to(device, non_blocking=True),
                "reduced_expression": batch["reduced_expression"].to(device, non_blocking=True),
            }
            loss = model(batch)
            meter.update(loss.item(), batch["image"].size(0))
    return float(meter.avg)


def train_one_epoch(model, loader, optimizer, device):
    meter = AvgMeter("train_loss")
    model.train()
    pbar = tqdm(loader, total=len(loader), desc="Train", leave=False)
    for batch in pbar:
        batch = {
            "image": batch["image"].to(device, non_blocking=True),
            "reduced_expression": batch["reduced_expression"].to(device, non_blocking=True),
        }
        loss = model(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        meter.update(loss.item(), batch["image"].size(0))
        pbar.set_postfix(train_loss=f"{meter.avg:.4f}", lr=f"{get_lr(optimizer):.2e}")
    return float(meter.avg)


def save_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def main():
    args = parse_args()
    set_seed(args.seed)

    outdir = args.outdir.resolve()
    if outdir.exists():
        raise FileExistsError(f"Use a new output directory: {outdir}")
    outdir.mkdir(parents=True)

    print("Loading single-cell inputs...")
    matrix, cell_ids, features, metadata = load_single_cell_inputs(args.single_cell_dir.resolve())
    matrix, cell_ids, metadata = filter_cells(
        matrix=matrix,
        cell_ids=cell_ids,
        metadata=metadata,
        min_total_counts=args.min_total_counts,
        min_genes_detected=args.min_genes_detected,
        max_cells=args.max_cells,
    )
    with Image.open(args.histology_image.resolve()) as img:
        image_width, image_height = img.size
    matrix, cell_ids, metadata, dropped_outside, coord_source = filter_cells_in_image_bounds(
        matrix=matrix,
        cell_ids=cell_ids,
        metadata=metadata,
        image_width=image_width,
        image_height=image_height,
    )
    if dropped_outside > 0:
        print(
            f"Dropped {dropped_outside} cells with invalid/out-of-image centroids "
            f"(image size: {image_width}x{image_height})."
        )
    print(f"Using spatial coordinates from: {coord_source}")

    gene_names = features["gene_name"].astype(str).tolist()
    if args.gene_list is not None:
        gene_indices, selected_gene_names = select_gene_indices_from_list(
            gene_names=gene_names,
            gene_list_path=args.gene_list.resolve(),
        )
        print(f"Using {len(selected_gene_names)} genes from {args.gene_list.resolve()}.")
    else:
        gene_indices, selected_gene_names = select_gene_indices(
            matrix=matrix,
            gene_names=gene_names,
            gene_mode=args.gene_mode,
            n_top_genes=args.n_top_genes,
        )
    matrix = matrix[:, gene_indices]

    print("Running TruncatedSVD for expression embedding...")
    reduced_expression, svd, actual_components = build_expression_embedding(
        matrix=matrix, n_components=args.svd_components, seed=args.seed
    )
    np.save(outdir / "reduced_expression.npy", reduced_expression)
    np.save(outdir / "svd_components.npy", svd.components_)
    pd.Series(selected_gene_names).to_csv(outdir / "selected_gene_names.txt", index=False, header=False)
    metadata.to_csv(outdir / "selected_cells.csv", index=False)

    train_dataset = SingleCellPatchDataset(
        image_path=args.histology_image.resolve(),
        metadata=metadata[["cell_id", "image_x", "image_y"]].copy(),
        reduced_expression=reduced_expression,
        patch_size=args.patch_size,
        is_train=True,
    )
    val_dataset = SingleCellPatchDataset(
        image_path=args.histology_image.resolve(),
        metadata=metadata[["cell_id", "image_x", "image_y"]].copy(),
        reduced_expression=reduced_expression,
        patch_size=args.patch_size,
        is_train=False,
    )

    n_total = len(train_dataset)
    if n_total < 3 or not 0 < args.val_fraction < 1:
        raise ValueError("BLEEP needs at least three reference cells and 0 < val_fraction < 1")
    val_size = int(n_total * args.val_fraction)
    train_size = n_total - val_size
    if val_size < 1:
        val_size = 1
        train_size = n_total - 1

    generator = torch.Generator().manual_seed(args.seed)
    all_indices = torch.randperm(n_total, generator=generator).tolist()
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:]
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    if len(train_loader) == 0:
        raise RuntimeError(
            "No training batches were created. Reduce --batch-size or increase --max-cells."
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select device: cpu explicitly")
    device = torch.device(args.device)
    model = choose_model(args.model, spot_embedding=actual_components).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = []
    best_val = float("inf")
    best_epoch = -1

    print("Training BLEEP...")
    for epoch in range(args.max_epochs):
        print(f"Epoch {epoch + 1}/{args.max_epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        print(f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), outdir / "best.pt")

    pd.DataFrame(history).to_csv(outdir / "training_history.csv", index=False)
    save_json(
        outdir / "run_metadata.json",
        {
            "single_cell_dir": str(args.single_cell_dir.resolve()),
            "histology_image": str(args.histology_image.resolve()),
            "n_cells_used": int(len(metadata)),
            "dropped_outside_image": int(dropped_outside),
            "coordinate_source": coord_source,
            "n_genes_selected": int(matrix.shape[1]),
            "gene_selection_mode": "gene_list" if args.gene_list is not None else args.gene_mode,
            "gene_list": str(args.gene_list.resolve()) if args.gene_list is not None else None,
            "expression_embedding_dim": int(actual_components),
            "model": args.model,
            "device_requested": args.device,
            "device_used": str(device),
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val),
            "patch_size": args.patch_size,
            "seed": args.seed,
        },
    )
    print(f"Done. Best model saved to {outdir / 'best.pt'}")


if __name__ == "__main__":
    main()
