#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


from .bleep_train import (  # noqa: E402
    SingleCellPatchDataset,
    choose_model,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run BLEEP nearest-neighbor prediction from a trained single-cell model."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training output directory containing best.pt and metadata files.",
    )
    parser.add_argument(
        "--query-metadata",
        type=Path,
        default=None,
        help="Optional CSV with columns cell_id,image_x,image_y for query patches. Defaults to selected_cells.csv.",
    )
    parser.add_argument(
        "--grid-query",
        action="store_true",
        help="Use a full-image grid as query instead of query metadata.",
    )
    parser.add_argument(
        "--grid-step",
        type=int,
        default=50,
        help="Pixel step size for full-image grid queries.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Directory for prediction outputs. Defaults to <run-dir>/predictions/self.",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--similarity-batch-size", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--method",
        type=str,
        default="weighted_average",
        choices=["simple", "average", "weighted_average"],
        help="Aggregation method used downstream for expression prediction.",
    )
    parser.add_argument(
        "--allow-self-match",
        action="store_true",
        help="Keep self matches when query set is the same as the reference set.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def save_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def load_model_weights(model, checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        new_key = new_key.replace("well", "spot")
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict, strict=True)
    return model


def build_query_loader(query_metadata, histology_image, embedding_dim, patch_size, batch_size, num_workers):
    dummy_expression = np.zeros((len(query_metadata), embedding_dim), dtype=np.float32)
    dataset = SingleCellPatchDataset(
        image_path=histology_image,
        metadata=query_metadata[["cell_id", "image_x", "image_y"]].copy(),
        reduced_expression=dummy_expression,
        patch_size=patch_size,
        is_train=False,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def compute_image_embeddings(model, loader, device):
    embeddings = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding query patches", total=len(loader)):
            image = batch["image"].to(device, non_blocking=True)
            image_features = model.image_encoder(image)
            image_embeddings = model.image_projection(image_features)
            embeddings.append(image_embeddings.detach().cpu())
    return torch.cat(embeddings, dim=0).numpy()


def compute_spot_embeddings(model, reduced_expression, device, batch_size):
    embeddings = []
    model.eval()
    with torch.no_grad():
        for start in tqdm(
            range(0, len(reduced_expression), batch_size),
            desc="Encoding reference expressions",
        ):
            batch = torch.from_numpy(reduced_expression[start : start + batch_size]).to(device)
            spot_embeddings = model.spot_projection(batch)
            embeddings.append(spot_embeddings.detach().cpu())
    return torch.cat(embeddings, dim=0).numpy()


def compute_topk_matches(
    query_embeddings,
    reference_embeddings,
    top_k,
    device,
    similarity_batch_size,
    exclude_self=False,
):
    query_tensor = F.normalize(torch.from_numpy(query_embeddings).to(device), p=2, dim=-1)
    reference_tensor = F.normalize(torch.from_numpy(reference_embeddings).to(device), p=2, dim=-1)

    n_query = query_tensor.shape[0]
    all_indices = np.empty((n_query, top_k), dtype=np.int64)
    all_scores = np.empty((n_query, top_k), dtype=np.float32)

    for start in tqdm(range(0, n_query, similarity_batch_size), desc="Searching nearest neighbors"):
        end = min(start + similarity_batch_size, n_query)
        similarity = query_tensor[start:end] @ reference_tensor.T
        if exclude_self and query_tensor.shape[0] == reference_tensor.shape[0]:
            rows = torch.arange(end - start, device=device)
            cols = torch.arange(start, end, device=device)
            similarity[rows, cols] = -1e9
        scores, indices = torch.topk(similarity, k=top_k, dim=1)
        all_indices[start:end] = indices.detach().cpu().numpy()
        all_scores[start:end] = scores.detach().cpu().numpy()

    return all_indices, all_scores


def prepare_query_metadata(query_metadata_path, selected_cells):
    if query_metadata_path is None:
        return selected_cells.copy(), "selected_cells.csv"

    query_metadata = pd.read_csv(query_metadata_path, dtype={"cell_id": str})
    required_cols = {"cell_id", "image_x", "image_y"}
    missing = required_cols - set(query_metadata.columns)
    if missing:
        raise ValueError(
            f"Query metadata is missing required columns: {sorted(missing)}"
        )
    return query_metadata.copy(), str(query_metadata_path)


def build_grid_query(histology_image, step):
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(histology_image) as img:
        width, height = img.size

    xs = np.arange(0, width, step, dtype=np.int32)
    ys = np.arange(0, height, step, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    grid_x = grid_x.ravel()
    grid_y = grid_y.ravel()

    cell_ids = [f"grid_x{int(x)}_y{int(y)}" for x, y in zip(grid_x, grid_y)]
    query_metadata = pd.DataFrame(
        {"cell_id": cell_ids, "image_x": grid_x, "image_y": grid_y}
    )
    return query_metadata, f"grid_step_{step}"


def main():
    args = parse_args()
    set_seed(args.seed)

    run_dir = args.run_dir.resolve()
    run_metadata = load_json(run_dir / "run_metadata.json")
    selected_cells = pd.read_csv(run_dir / "selected_cells.csv", dtype={"cell_id": str})
    reduced_expression = np.load(run_dir / "reduced_expression.npy")

    outdir = (
        args.outdir.resolve()
        if args.outdir is not None
        else (run_dir / "predictions" / "self").resolve()
    )
    outdir.mkdir(parents=True, exist_ok=True)

    histology_image = Path(run_metadata["histology_image"])
    patch_size = int(run_metadata["patch_size"])
    model_name = run_metadata["model"]
    expression_embedding_dim = int(run_metadata["expression_embedding_dim"])
    if args.grid_query:
        query_metadata, query_source = build_grid_query(
            histology_image=histology_image,
            step=args.grid_step,
        )
    else:
        query_metadata, query_source = prepare_query_metadata(args.query_metadata, selected_cells)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select device: cpu explicitly")
    device = torch.device(args.device)
    model = choose_model(model_name, spot_embedding=expression_embedding_dim).to(device)
    model = load_model_weights(model, run_dir / "best.pt", device)
    model.eval()

    query_loader = build_query_loader(
        query_metadata=query_metadata,
        histology_image=histology_image,
        embedding_dim=expression_embedding_dim,
        patch_size=patch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("Computing query image embeddings...")
    query_image_embeddings = compute_image_embeddings(model, query_loader, device)

    print("Computing reference spot embeddings...")
    reference_spot_embeddings = compute_spot_embeddings(
        model=model,
        reduced_expression=reduced_expression.astype(np.float32, copy=False),
        device=device,
        batch_size=max(args.batch_size, 512),
    )

    query_ids = query_metadata["cell_id"].astype(str).tolist()
    reference_ids = selected_cells["cell_id"].astype(str).tolist()
    exclude_self = bool((not args.allow_self_match) and query_ids == reference_ids)

    print("Finding nearest neighbors...")
    nn_indices, nn_scores = compute_topk_matches(
        query_embeddings=query_image_embeddings,
        reference_embeddings=reference_spot_embeddings,
        top_k=args.top_k,
        device=device,
        similarity_batch_size=args.similarity_batch_size,
        exclude_self=exclude_self,
    )

    np.save(outdir / "query_image_embeddings.npy", query_image_embeddings)
    np.save(outdir / "reference_spot_embeddings.npy", reference_spot_embeddings)
    np.save(outdir / "topk_indices.npy", nn_indices)
    np.save(outdir / "topk_scores.npy", nn_scores)
    query_metadata.to_csv(outdir / "query_metadata.csv", index=False)

    save_json(
        outdir / "prediction_metadata.json",
        {
            "run_dir": str(run_dir),
            "query_source": query_source,
            "histology_image": str(histology_image),
            "model": model_name,
            "device_requested": args.device,
            "device_used": str(device),
            "top_k": args.top_k,
            "method": args.method,
            "exclude_self_applied": exclude_self,
            "n_queries": int(len(query_metadata)),
            "n_reference_cells": int(len(selected_cells)),
        },
    )
    print(f"Prediction artifacts written to {outdir}")


if __name__ == "__main__":
    main()
