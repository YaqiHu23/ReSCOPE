#!/usr/bin/env python3
import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import sparse


from .common import compute_metrics, load_truth, load_single_cell_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ROI-supervised iSTAR prediction and held-out-cell evaluation."
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--histology-image", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--roi-names", nargs="+", required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--n-states", type=int, default=1)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--gene-mode", type=str, default="top")
    parser.add_argument("--n-top-genes", type=int, default=2000)
    parser.add_argument("--gene-list", type=Path, default=None)
    parser.add_argument("--gene-chunk-size", type=int, default=256)
    parser.add_argument("--pixel-size-raw", type=float, default=1.0)
    parser.add_argument("--pixel-size-target", type=float, default=1.0)
    parser.add_argument("--radius-source", type=str, default="median_area")
    parser.add_argument("--radius-raw", type=float, default=None)
    parser.add_argument("--min-total-counts", type=float, default=1.0)
    parser.add_argument("--min-genes-detected", type=float, default=1.0)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--skip-feature-extraction", action="store_true")
    return parser.parse_args()


def run_command(cmd: list[str]) -> None:
    print("$", " ".join(str(part) for part in cmd))
    subprocess.run(cmd, check=True)


def compute_query_locs(prefix: Path, query_metadata: pd.DataFrame, target_shape: tuple[int, int]) -> np.ndarray:
    pixel_size_raw = float((prefix / "pixel-size-raw.txt").read_text().strip())
    pixel_size = float((prefix / "pixel-size.txt").read_text().strip())
    scale = pixel_size_raw / pixel_size

    with Image.open(prefix / "he.jpg") as img:
        current_shape = np.array([img.height, img.width], dtype=np.int64)

    target_shape_arr = np.array(target_shape, dtype=np.int64)
    rescale_factor = current_shape // target_shape_arr
    if np.any(rescale_factor <= 0):
        raise RuntimeError(
            f"Invalid iSTAR rescale_factor: current_shape={current_shape}, target_shape={target_shape_arr}"
        )

    x_scaled = np.round(pd.to_numeric(query_metadata["centroid_x"], errors="raise").to_numpy() * scale)
    y_scaled = np.round(pd.to_numeric(query_metadata["centroid_y"], errors="raise").to_numpy() * scale)
    locs = np.stack([y_scaled, x_scaled], axis=1).astype(np.float64)
    locs /= rescale_factor
    locs = np.round(locs).astype(np.int64)
    return locs


def main() -> None:
    args = parse_args()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for roi_name in args.roi_names:
        roi_dir = args.experiment_dir.resolve() / roi_name
        manifest_path = roi_dir / "roi_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing ROI manifest: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            roi_manifest = json.load(handle)

        reference_dir = Path(roi_manifest["reference_single_cell_dir"])
        query_dir = Path(roi_manifest["query_single_cell_dir"])

        roi_outdir = outdir / roi_name
        run_dir = roi_outdir / "istar_run"
        eval_dir = roi_outdir / "istar_eval"
        eval_dir.mkdir(parents=True, exist_ok=True)

        train_cmd = [
            sys.executable,
            "-m", "rescope.downstream.istar_train",
            "--single-cell-dir",
            str(reference_dir),
            "--histology-image",
            str(args.histology_image.resolve()),
            "--outdir",
            str(run_dir),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--epochs",
            str(args.epochs),
            "--learning-rate",
            str(args.learning_rate),
            "--n-states",
            str(args.n_states),
            "--n-jobs",
            str(args.n_jobs),
            "--gene-mode",
            args.gene_mode,
            "--n-top-genes",
            str(args.n_top_genes),
            "--gene-chunk-size",
            str(args.gene_chunk_size),
            "--pixel-size-raw",
            str(args.pixel_size_raw),
            "--pixel-size-target",
            str(args.pixel_size_target),
            "--radius-source",
            args.radius_source,
            "--min-total-counts",
            str(args.min_total_counts),
            "--min-genes-detected",
            str(args.min_genes_detected),
        ]
        if args.gene_list is not None:
            train_cmd.extend(["--gene-list", str(args.gene_list.resolve())])
        if args.radius_raw is not None:
            train_cmd.extend(["--radius-raw", str(args.radius_raw)])
        if args.max_cells is not None:
            train_cmd.extend(["--max-cells", str(args.max_cells)])
        if args.skip_feature_extraction:
            train_cmd.append("--skip-feature-extraction")
        run_command(train_cmd)

        selected_gene_names = (run_dir / "selected_gene_names.txt").read_text().splitlines()
        _, query_cell_ids, _, query_metadata = load_single_cell_inputs(query_dir)
        truth = load_truth(query_dir, query_cell_ids, selected_gene_names)

        first_gene_path = run_dir / "cnts-super-all" / f"{selected_gene_names[0]}.pickle"
        with first_gene_path.open("rb") as handle:
            first_gene_grid = pickle.load(handle)
        target_shape = tuple(first_gene_grid.shape)
        query_locs = compute_query_locs(run_dir / "istar_input", query_metadata, target_shape)

        in_bounds = (
            (query_locs[:, 0] >= 0)
            & (query_locs[:, 0] < target_shape[0])
            & (query_locs[:, 1] >= 0)
            & (query_locs[:, 1] < target_shape[1])
        )
        if not np.all(in_bounds):
            query_locs = query_locs[in_bounds]
            truth = truth[in_bounds]
            query_metadata = query_metadata.loc[in_bounds].reset_index(drop=True)
            query_cell_ids = [cell_id for keep, cell_id in zip(in_bounds, query_cell_ids) if keep]

        predicted = np.empty((len(query_cell_ids), len(selected_gene_names)), dtype=np.float32)
        for gene_idx, gene_name in enumerate(selected_gene_names):
            with (run_dir / "cnts-super-all" / f"{gene_name}.pickle").open("rb") as handle:
                gene_grid = pickle.load(handle)
            predicted[:, gene_idx] = gene_grid[query_locs[:, 0], query_locs[:, 1]]

        finite = np.isfinite(predicted).all(axis=1)
        n_nonfinite = int((~finite).sum())
        predicted, truth = predicted[finite], truth[finite]
        query_cell_ids = [cell_id for cell_id, keep in zip(query_cell_ids, finite) if keep]
        metrics, per_gene_df = compute_metrics(predicted=predicted, truth=truth, gene_names=selected_gene_names)
        metrics["n_query_nonfinite_excluded"] = n_nonfinite
        metrics["n_query_out_of_bounds_excluded"] = int((~in_bounds).sum())
        metrics.update(
            {
                "roi_name": roi_name,
                "method_name": "istar",
                "reference_single_cell_dir": str(reference_dir),
                "query_single_cell_dir": str(query_dir),
                "run_dir": str(run_dir),
                "target_shape": list(target_shape),
                "seed": int(args.seed),
            }
        )

        np.savez_compressed(
            eval_dir / "predictions_and_truth.npz",
            predicted=predicted,
            truth=truth,
            query_cell_ids=np.asarray(query_cell_ids),
            gene_names=np.asarray(selected_gene_names),
        )
        per_gene_df.to_csv(eval_dir / "per_gene_mse.csv", index=False)
        with (eval_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

        summary_rows.append(metrics)

    pd.DataFrame(summary_rows).to_csv(outdir / "istar_roi_mse_summary.csv", index=False)
    print(f"Finished ROI iSTAR experiments. Summary: {outdir / 'istar_roi_mse_summary.csv'}")


if __name__ == "__main__":
    main()
