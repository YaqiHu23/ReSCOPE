#!/usr/bin/env python3

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import sparse


from .external import repository
from .common import load_single_cell_inputs
ISTAR_DIR = repository("istar")
sys.path.insert(0, str(ISTAR_DIR))

from impute import impute as istar_impute
from impute_by_basic import get_embeddings as load_istar_embeddings
from impute_by_basic import get_locs as load_istar_locs
from utils import read_string, save_pickle, save_tsv, write_lines, write_string


Image.MAX_IMAGE_PIXELS = None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run iSTAR directly from CellViT-derived single-cell Visium HD data."
        )
    )
    parser.add_argument(
        "--single-cell-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing cell_gene_counts.npz, cell_ids.tsv.gz, "
            "features.tsv.gz, and cell_metadata.csv.gz."
        ),
    )
    parser.add_argument(
        "--histology-image",
        type=Path,
        required=True,
        help="Path to the full histology image used for iSTAR.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Directory for all generated iSTAR inputs and outputs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device passed to iSTAR feature extraction and training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs per gene chunk.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate passed to the iSTAR optimizer.",
    )
    parser.add_argument(
        "--n-states",
        type=int,
        default=1,
        help="Number of independent iSTAR model states per chunk.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel jobs used by iSTAR across model states.",
    )
    parser.add_argument(
        "--gene-mode",
        type=str,
        default="top",
        choices=["all", "top"],
        help="Whether to impute all genes or only top variable genes.",
    )
    parser.add_argument(
        "--gene-list",
        type=Path,
        default=None,
        help="Optional text file with one gene name per line. Overrides --gene-mode/--n-top-genes.",
    )
    parser.add_argument(
        "--n-top-genes",
        type=int,
        default=2000,
        help="Number of genes when --gene-mode top.",
    )
    parser.add_argument(
        "--gene-chunk-size",
        type=int,
        default=256,
        help="Number of genes per iSTAR chunk when imputing many genes.",
    )
    parser.add_argument(
        "--pixel-size-raw",
        type=float,
        default=1.0,
        help=(
            "Pixel size written into iSTAR raw metadata. Keep 1.0 if your "
            "cell centroids are already in the histology image pixel space."
        ),
    )
    parser.add_argument(
        "--pixel-size-target",
        type=float,
        default=1.0,
        help="Target pixel size written to pixel-size.txt for iSTAR rescaling.",
    )
    parser.add_argument(
        "--max-scaled-image-dim",
        type=int,
        default=8192,
        help=(
            "If the raw histology image is larger than this, automatically "
            "increase pixel-size-target so iSTAR rescales the image down "
            "before saving JPEG intermediates."
        ),
    )
    parser.add_argument(
        "--radius-source",
        type=str,
        default="median_area",
        choices=["median_area", "fixed"],
        help="How to determine radius-raw for cell-level supervision.",
    )
    parser.add_argument(
        "--radius-raw",
        type=float,
        default=None,
        help="Explicit raw radius in image pixels. Required for --radius-source fixed.",
    )
    parser.add_argument(
        "--min-total-counts",
        type=float,
        default=1.0,
        help="Drop cells with total counts below this threshold.",
    )
    parser.add_argument(
        "--min-genes-detected",
        type=float,
        default=1.0,
        help="Drop cells with detected genes below this threshold.",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help=(
            "Optional cap on the number of cells used for supervision. "
            "Keeps cells with the largest total counts."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260730,
        help="Shared random seed for NumPy/Python/PyTorch iSTAR training.",
    )
    parser.add_argument(
        "--skip-feature-extraction",
        action="store_true",
        help="Reuse existing iSTAR histology embeddings if present.",
    )
    return parser.parse_args()


def prepare_output_dir(outdir):
    if outdir.exists():
        raise FileExistsError(f"Use a new output directory: {outdir}")
    outdir.mkdir(parents=True)


def save_histology_image(src, dst_prefix):
    dst_prefix.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(src)
    src_suffix = src.suffix.lower()
    image_size = image.size

    # For whole-slide TIFF/BigTIFF/SVS inputs, avoid re-encoding entirely.
    # PIL can overflow when writing huge TIFFs, and JPEG has a 65500-pixel
    # dimension limit. A symlink lets iSTAR open the original image through
    # the filename pattern it expects.
    if src_suffix in {".tif", ".tiff", ".btf", ".svs"}:
        dst = dst_prefix.with_suffix(".tiff")
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
        return dst, image_size

    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    if image.mode == "L":
        image = image.convert("RGB")

    dst = dst_prefix.with_suffix(".jpg")
    image.save(dst, quality=95)
    return dst, image_size


def run_command(cmd, cwd):
    print("$", " ".join(str(part) for part in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def write_scaled_spatial_inputs(prefix):
    pixel_size_raw = float(read_string(str(prefix / "pixel-size-raw.txt")))
    pixel_size = float(read_string(str(prefix / "pixel-size.txt")))
    scale = pixel_size_raw / pixel_size

    locs = pd.read_csv(prefix / "locs-raw.tsv", sep="\t", index_col=0)
    locs = locs * scale
    locs = locs.round().astype(int)
    save_tsv(locs, str(prefix / "locs.tsv"))

    radius = float(read_string(str(prefix / "radius-raw.txt")))
    radius = int(np.round(radius * scale))
    write_string(radius, str(prefix / "radius.txt"))


def run_istar_feature_pipeline(prefix, device, skip_feature_extraction, random_weights):
    if skip_feature_extraction and (prefix / "embeddings-hist.pickle").exists() and (prefix / "he.jpg").exists():
        print("Reusing cached iSTAR histology embeddings and preprocessed H&E image.")
        write_scaled_spatial_inputs(prefix)
        return

    prefix_str = str(prefix) + "/"
    run_command(
        [
            sys.executable,
            str(ISTAR_DIR / "rescale.py"),
            prefix_str,
            "--image",
            "--locs",
            "--radius",
        ],
        cwd=ISTAR_DIR,
    )
    run_command(
        [sys.executable, str(ISTAR_DIR / "preprocess.py"), prefix_str, "--image"],
        cwd=ISTAR_DIR,
    )
    if not skip_feature_extraction or not (prefix / "embeddings-hist.pickle").exists():
        cmd = [sys.executable, str(ISTAR_DIR / "extract_features.py"), prefix_str, f"--device={device}"]
        if random_weights:
            cmd.append("--random-weights")
        run_command(cmd, cwd=ISTAR_DIR)


def sparse_column_variance(matrix):
    mean = np.asarray(matrix.mean(axis=0)).ravel()
    mean_sq = np.asarray(matrix.power(2).mean(axis=0)).ravel()
    var = mean_sq - mean ** 2
    var[var < 0] = 0
    return var


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
        raise RuntimeError("No genes with non-zero variance were found after filtering cells.")

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


def chunk_indices(indices, chunk_size):
    for start in range(0, len(indices), chunk_size):
        yield start // chunk_size, indices[start : start + chunk_size]


def load_impute_inputs(prefix):
    prefix_str = str(prefix) + "/"
    embs = load_istar_embeddings(prefix_str)
    locs = load_istar_locs(prefix_str, target_shape=embs.shape[:2])
    radius = int(read_string(prefix_str + "radius.txt"))
    radius = max(1, int(round(radius / 16)))
    return embs, locs, radius


def infer_radius_raw(metadata, radius_source, radius_raw):
    if radius_source == "fixed":
        if radius_raw is None:
            raise ValueError("--radius-raw is required when --radius-source fixed.")
        return max(1, int(round(radius_raw)))

    if "polygon_area_fullres" in metadata.columns:
        area = pd.to_numeric(metadata["polygon_area_fullres"], errors="coerce")
        area_source = "polygon_area_fullres"
    elif "cell_area_um2" in metadata.columns:
        required = [
            "centroid_x",
            "centroid_y",
            "xenium_x_centroid_um",
            "xenium_y_centroid_um",
        ]
        missing = [column for column in required if column not in metadata.columns]
        if missing:
            raise ValueError(
                "cell_metadata.csv.gz is missing polygon_area_fullres and cannot "
                "convert cell_area_um2 because these coordinate columns are missing: "
                + ", ".join(missing)
            )

        coordinate_df = metadata[required].apply(pd.to_numeric, errors="coerce")
        coordinate_df = coordinate_df.replace([np.inf, -np.inf], np.nan).dropna()
        if len(coordinate_df) < 3:
            raise ValueError(
                "Need at least 3 cells with finite fullres and micron coordinates "
                "to convert cell_area_um2 to fullres pixels."
            )

        micron_xy = coordinate_df[["xenium_x_centroid_um", "xenium_y_centroid_um"]].to_numpy()
        fullres_xy = coordinate_df[["centroid_x", "centroid_y"]].to_numpy()
        micron_xy = micron_xy - micron_xy.mean(axis=0)
        fullres_xy = fullres_xy - fullres_xy.mean(axis=0)
        transform, *_ = np.linalg.lstsq(micron_xy, fullres_xy, rcond=None)
        area_scale = abs(float(np.linalg.det(transform)))
        if not np.isfinite(area_scale) or area_scale <= 0:
            raise ValueError(
                "Could not infer a positive fullres-pixel area scale from "
                "xenium_x/y_centroid_um and centroid_x/y."
            )

        area = pd.to_numeric(metadata["cell_area_um2"], errors="coerce") * area_scale
        area_source = f"cell_area_um2 scaled by {area_scale:.6g} fullres_px2_per_um2"
    else:
        raise ValueError(
            "cell_metadata.csv.gz is missing polygon_area_fullres and cell_area_um2, "
            "so radius cannot be estimated from cell area."
        )

    area = area[np.isfinite(area) & (area > 0)]
    if len(area) == 0:
        raise ValueError(f"No positive {area_source} values were found.")

    radius = np.sqrt(area.to_numpy() / np.pi)
    radius_raw = max(1, int(round(float(np.median(radius)))))
    print(f"Inferred radius_raw={radius_raw} from {area_source}.", flush=True)
    return radius_raw


def filter_cells(matrix, cell_ids, metadata, min_total_counts, min_genes_detected, max_cells):
    def numeric_metadata_series(column_name):
        if column_name not in metadata.columns:
            return pd.Series(np.nan, index=metadata.index, dtype="float64")
        return pd.to_numeric(metadata[column_name], errors="coerce")

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

    coord_keep = np.ones(len(metadata), dtype=bool)
    if "centroid_x" in metadata.columns and "centroid_y" in metadata.columns:
        centroid_x = pd.to_numeric(metadata["centroid_x"], errors="coerce")
        centroid_y = pd.to_numeric(metadata["centroid_y"], errors="coerce")
        coord_keep = np.isfinite(centroid_x.to_numpy()) & np.isfinite(centroid_y.to_numpy())

    keep_mask = (
        (total_counts.to_numpy() >= min_total_counts)
        & (n_genes_detected.to_numpy() >= min_genes_detected)
        & coord_keep
    )
    keep_indices = np.flatnonzero(keep_mask)
    if len(keep_indices) == 0:
        raise RuntimeError("No cells remain after applying the requested filters.")

    if max_cells is not None and len(keep_indices) > max_cells:
        ranked = np.argsort(total_counts.to_numpy()[keep_indices])[::-1]
        keep_indices = keep_indices[ranked[:max_cells]]
        keep_indices.sort()

    filtered_matrix = matrix[keep_indices]
    filtered_metadata = metadata.iloc[keep_indices].reset_index(drop=True)
    filtered_cell_ids = [cell_ids[i] for i in keep_indices]
    return filtered_matrix, filtered_cell_ids, filtered_metadata


def filter_cells_by_embedding_bounds(matrix, cell_ids, metadata, locs, emb_shape, radius):
    radius_ceil = int(np.ceil(radius))
    keep_mask = (
        (locs[:, 0] >= radius_ceil)
        & (locs[:, 1] >= radius_ceil)
        & (locs[:, 0] < emb_shape[0] - radius_ceil)
        & (locs[:, 1] < emb_shape[1] - radius_ceil)
    )
    keep_indices = np.flatnonzero(keep_mask)
    if len(keep_indices) == 0:
        raise RuntimeError(
            "No cells remain after removing coordinates that are too close "
            "to the embedding boundary for the requested radius."
        )

    filtered_matrix = matrix[keep_indices]
    filtered_metadata = metadata.iloc[keep_indices].reset_index(drop=True)
    filtered_cell_ids = [cell_ids[i] for i in keep_indices]
    filtered_locs = locs[keep_indices]
    return filtered_matrix, filtered_cell_ids, filtered_metadata, filtered_locs


def write_istar_inputs(
    outdir,
    histology_image,
    cell_ids,
    cell_metadata,
    pixel_size_raw,
    pixel_size_target,
    max_scaled_image_dim,
    radius_raw,
):
    for col in ["centroid_x", "centroid_y"]:
        if col not in cell_metadata.columns:
            raise ValueError(f"cell_metadata.csv.gz is missing required column: {col}")

    prefix = outdir / "istar_input"
    prefix.mkdir(parents=True, exist_ok=True)

    histology_copy, image_size = save_histology_image(histology_image, prefix / "he-raw")
    effective_pixel_size_target = float(pixel_size_target)
    max_dim = max(image_size)
    if max_scaled_image_dim is not None and max_dim > max_scaled_image_dim:
        min_target = pixel_size_raw * (max_dim / float(max_scaled_image_dim))
        effective_pixel_size_target = max(effective_pixel_size_target, min_target)

    x = pd.to_numeric(cell_metadata["centroid_x"], errors="coerce")
    y = pd.to_numeric(cell_metadata["centroid_y"], errors="coerce")
    if x.isna().any() or y.isna().any():
        raise ValueError("centroid_x/centroid_y contain missing or non-numeric values.")
    locs = pd.DataFrame(
        {
            "x": np.round(x.to_numpy()).astype(int),
            "y": np.round(y.to_numpy()).astype(int),
        },
        index=cell_ids,
    )
    if locs.isna().any(axis=None):
        raise ValueError("locs-raw.tsv still contains missing coordinates after conversion.")

    locs.index.name = "spot_id"
    save_tsv(locs, str(prefix / "locs-raw.tsv"))
    write_string(str(pixel_size_raw), str(prefix / "pixel-size-raw.txt"))
    write_string(str(effective_pixel_size_target), str(prefix / "pixel-size.txt"))
    write_string(str(radius_raw), str(prefix / "radius-raw.txt"))
    return prefix, locs, histology_copy, effective_pixel_size_target, image_size


def write_metadata(outdir, metadata):
    with open(outdir / "run_metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2)


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select device: cpu explicitly")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    prepare_output_dir(args.outdir)
    args.outdir = args.outdir.resolve()
    single_cell_dir = args.single_cell_dir.resolve()

    print("Loading single-cell inputs...")
    matrix, cell_ids, features, metadata = load_single_cell_inputs(single_cell_dir)

    print("Filtering cells...")
    n_cells_before_filter = matrix.shape[0]
    matrix, cell_ids, metadata = filter_cells(
        matrix=matrix,
        cell_ids=cell_ids,
        metadata=metadata,
        min_total_counts=args.min_total_counts,
        min_genes_detected=args.min_genes_detected,
        max_cells=args.max_cells,
    )
    if len(cell_ids) == 0:
        raise RuntimeError("No cells remain after filtering.")
    print(f"Using {len(cell_ids)} cells after filtering (from {n_cells_before_filter}).")

    radius_raw = infer_radius_raw(
        metadata=metadata,
        radius_source=args.radius_source,
        radius_raw=args.radius_raw,
    )
    prefix, locs, histology_copy, effective_pixel_size_target, image_size = write_istar_inputs(
        outdir=args.outdir,
        histology_image=args.histology_image,
        cell_ids=cell_ids,
        cell_metadata=metadata,
        pixel_size_raw=args.pixel_size_raw,
        pixel_size_target=args.pixel_size_target,
        max_scaled_image_dim=args.max_scaled_image_dim,
        radius_raw=radius_raw,
    )
    print(
        "Histology image size "
        f"{image_size[0]}x{image_size[1]}, "
        f"pixel-size-target={effective_pixel_size_target:.6g}"
    )

    gene_names = features["gene_name"].astype(str).tolist()
    gene_ids = features["gene_id"].astype(str).tolist()

    print("Preparing histology features with iSTAR...")
    run_istar_feature_pipeline(
        prefix=prefix,
        device=args.device,
        skip_feature_extraction=args.skip_feature_extraction,
        random_weights=False,
    )

    print("Loading iSTAR-ready embeddings and cell locations...")
    embs, locs_arr, radius = load_impute_inputs(prefix)
    n_cells_before_bounds = len(cell_ids)
    matrix, cell_ids, metadata, locs_arr = filter_cells_by_embedding_bounds(
        matrix=matrix,
        cell_ids=cell_ids,
        metadata=metadata,
        locs=locs_arr,
        emb_shape=embs.shape[:2],
        radius=radius,
    )
    print(
        f"Using {len(cell_ids)} cells after boundary filtering "
        f"(from {n_cells_before_bounds})."
    )

    filtered_cells_table = metadata.copy()
    filtered_cells_table.to_csv(args.outdir / "selected_cells.csv", index=False)

    if args.gene_list is not None:
        gene_indices, selected_gene_names = select_gene_indices_from_list(
            gene_names=gene_names,
            gene_list_path=args.gene_list.resolve(),
        )
        print(f"Using {len(selected_gene_names)} genes from {args.gene_list.resolve()}.")
    else:
        gene_indices, selected_gene_names = select_gene_indices(
            matrix,
            gene_names,
            gene_mode=args.gene_mode,
            n_top_genes=args.n_top_genes,
        )
    write_lines(selected_gene_names, str(args.outdir / "selected_gene_names.txt"))

    batch_size = max(1, min(128, len(cell_ids) // 16))

    chunk_root = args.outdir / "chunks"
    chunk_root.mkdir(parents=True, exist_ok=True)
    merged_root = args.outdir / "cnts-super-all"
    merged_root.mkdir(parents=True, exist_ok=True)

    print(f"Running gene imputation in chunks of {args.gene_chunk_size} genes...")
    for chunk_id, gene_idx_chunk in chunk_indices(gene_indices, args.gene_chunk_size):
        chunk_gene_names = [gene_names[i] for i in gene_idx_chunk]
        chunk_gene_ids = [gene_ids[i] for i in gene_idx_chunk]
        chunk_dir = chunk_root / f"chunk_{chunk_id:04d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        chunk_counts = matrix[:, gene_idx_chunk].toarray()
        cnts_df = pd.DataFrame(chunk_counts, index=cell_ids, columns=chunk_gene_names)
        cnts_df.index.name = "spot_id"
        save_tsv(cnts_df, str(chunk_dir / "cnts.tsv"))
        write_lines(chunk_gene_names, str(chunk_dir / "gene-names.txt"))

        print(
            f"Chunk {chunk_id:04d}: training iSTAR for "
            f"{len(chunk_gene_names)} genes."
        )
        istar_impute(
            embs=embs.copy(),
            cnts=cnts_df,
            locs=locs_arr,
            radius=radius,
            epochs=args.epochs,
            batch_size=batch_size,
            prefix=str(chunk_dir) + "/",
            n_states=args.n_states,
            load_saved=False,
            device=args.device,
            n_jobs=args.n_jobs,
            lr=args.learning_rate,
        )

        for gene_name, gene_id in zip(chunk_gene_names, chunk_gene_ids):
            src = chunk_dir / "cnts-super" / f"{gene_name}.pickle"
            dst = merged_root / f"{gene_name}.pickle"
            if src.exists():
                shutil.copy2(src, dst)
            else:
                raise FileNotFoundError(f"Expected iSTAR output not found: {src}")

        save_pickle(
            {
                "gene_names": chunk_gene_names,
                "gene_ids": chunk_gene_ids,
            },
            str(chunk_dir / "chunk_metadata.pickle"),
        )

    metadata_payload = {
        "single_cell_dir": str(single_cell_dir),
        "histology_image": str(args.histology_image.resolve()),
        "histology_copy": str(histology_copy),
        "n_cells_used": len(cell_ids),
        "n_genes_total": int(matrix.shape[1]),
        "n_genes_selected": len(selected_gene_names),
        "radius_raw": radius_raw,
        "pixel_size_raw": args.pixel_size_raw,
        "pixel_size_target": effective_pixel_size_target,
        "max_scaled_image_dim": args.max_scaled_image_dim,
        "filters": {
            "min_total_counts": args.min_total_counts,
            "min_genes_detected": args.min_genes_detected,
            "max_cells": args.max_cells,
        },
        "gene_mode": args.gene_mode,
        "gene_selection_mode": "gene_list" if args.gene_list is not None else args.gene_mode,
        "gene_list": str(args.gene_list.resolve()) if args.gene_list is not None else None,
        "n_top_genes": args.n_top_genes,
        "gene_chunk_size": args.gene_chunk_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "n_states": args.n_states,
        "n_jobs": args.n_jobs,
        "device": args.device,
        "random_weights": False,
        "seed": args.seed,
        "locs_preview": {
            "n_rows": int(len(locs)),
            "x_min": int(locs["x"].min()),
            "x_max": int(locs["x"].max()),
            "y_min": int(locs["y"].min()),
            "y_max": int(locs["y"].max()),
        },
    }
    write_metadata(args.outdir, metadata_payload)
    print(f"Done. Outputs written to {args.outdir}")


if __name__ == "__main__":
    main()
