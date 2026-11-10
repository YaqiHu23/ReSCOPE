#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


from .bleep_expression import aggregate_predictions, load_reference_context, rebuild_reference_matrix
from .common import compute_metrics, load_truth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ROI-supervised BLEEP prediction and held-out-cell evaluation."
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--histology-image", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--roi-names", nargs="+", required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--gene-mode", type=str, default="top")
    parser.add_argument("--n-top-genes", type=int, default=3000)
    parser.add_argument("--gene-list", type=Path, default=None)
    parser.add_argument("--svd-components", type=int, default=512)
    parser.add_argument("--min-total-counts", type=float, default=1.0)
    parser.add_argument("--min-genes-detected", type=float, default=1.0)
    parser.add_argument("--max-cells", type=int, default=80000)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--method", type=str, default="weighted_average")
    return parser.parse_args()


def run_command(cmd: list[str]) -> None:
    print("$", " ".join(str(part) for part in cmd))
    subprocess.run(cmd, check=True)


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
        query_metadata = Path(roi_manifest["query_metadata_csv"])

        roi_outdir = outdir / roi_name
        run_dir = roi_outdir / "bleep_run"
        pred_dir = roi_outdir / "bleep_prediction"
        eval_dir = roi_outdir / "bleep_eval"
        eval_dir.mkdir(parents=True, exist_ok=True)

        train_cmd = [
            sys.executable,
            "-m", "rescope.downstream.bleep_train",
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
            "--max-epochs",
            str(args.max_epochs),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--val-fraction",
            str(args.val_fraction),
            "--model",
            args.model,
            "--patch-size",
            str(args.patch_size),
            "--gene-mode",
            args.gene_mode,
            "--n-top-genes",
            str(args.n_top_genes),
            "--svd-components",
            str(args.svd_components),
            "--min-total-counts",
            str(args.min_total_counts),
            "--min-genes-detected",
            str(args.min_genes_detected),
            "--max-cells",
            str(args.max_cells),
        ]
        if args.gene_list is not None:
            train_cmd.extend(["--gene-list", str(args.gene_list.resolve())])
        run_command(train_cmd)

        selected_cells = pd.read_csv(run_dir / "selected_cells.csv")
        effective_top_k = min(args.top_k, max(1, len(selected_cells)))
        predict_cmd = [
            sys.executable,
            "-m", "rescope.downstream.bleep_predict",
            "--run-dir",
            str(run_dir),
            "--query-metadata",
            str(query_metadata),
            "--outdir",
            str(pred_dir),
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--top-k",
            str(effective_top_k),
            "--method",
            args.method,
            "--seed",
            str(args.seed),
        ]
        run_command(predict_cmd)

        run_metadata, selected_cells, selected_gene_names = load_reference_context(run_dir)
        reference_expr = rebuild_reference_matrix(run_metadata, selected_cells, selected_gene_names)
        topk_indices = np.load(pred_dir / "topk_indices.npy")
        topk_scores = np.load(pred_dir / "topk_scores.npy")
        query_pred_meta = pd.read_csv(pred_dir / "query_metadata.csv", dtype={"cell_id": str})
        query_cell_ids = query_pred_meta["cell_id"].astype(str).tolist()
        truth = load_truth(query_dir, query_cell_ids, selected_gene_names)
        predicted = aggregate_predictions(
            reference_expr=reference_expr,
            topk_indices=topk_indices,
            topk_scores=topk_scores,
            gene_indices=np.arange(len(selected_gene_names)),
            method=args.method,
        ).astype(np.float32, copy=False)

        metrics, per_gene_df = compute_metrics(predicted=predicted, truth=truth, gene_names=selected_gene_names)
        metrics.update(
            {
                "roi_name": roi_name,
                "method_name": "bleep",
                "aggregation_method": args.method,
                "top_k": int(effective_top_k),
                "seed": int(args.seed),
                "reference_single_cell_dir": str(reference_dir),
                "query_single_cell_dir": str(query_dir),
                "run_dir": str(run_dir),
                "prediction_dir": str(pred_dir),
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

    pd.DataFrame(summary_rows).to_csv(outdir / "bleep_roi_mse_summary.csv", index=False)
    print(f"Finished ROI BLEEP experiments. Summary: {outdir / 'bleep_roi_mse_summary.csv'}")


if __name__ == "__main__":
    main()
