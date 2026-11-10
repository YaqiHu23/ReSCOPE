"""Run in the separate CellViT environment; use only official trusted checkpoints."""
from pathlib import Path
import argparse
import os
import subprocess
import sys

CELLVIT_REVISION = "05097e18e3d194a65121042f631b5753069f5ee3"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--wsi", type=Path, required=True)
    parser.add_argument("--patch-root", type=Path, required=True)
    parser.add_argument("--magnification", type=int, choices=(20, 40), default=40)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    repo = args.repo.resolve()
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if revision != CELLVIT_REVISION:
        raise ValueError("CellViT revision differs from the pinned release; run scripts/setup_preprocessing.sh")
    if not args.checkpoint.is_file() or not args.wsi.is_file():
        raise FileNotFoundError("Provide existing WSI and CellViT checkpoint files")
    if f"x{args.magnification}" not in args.checkpoint.name:
        raise ValueError("CellViT checkpoint filename must match the requested x20/x40 magnification")
    folder = args.patch_root.resolve() / args.wsi.stem
    output = folder / "cell_detection" / "rescope"
    if output.exists():
        raise FileExistsError(output)
    # The upstream checkpoint includes training metadata and uses torch.load
    # without weights_only. This variable applies only to this child process.
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    sys.path.insert(0, str(repo))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CellViT inference requires a CUDA GPU in the CellViT environment")
    from cell_segmentation.inference.cell_detection import CellSegmentationInference, check_wsi
    from datamodel.wsi_datamodel import WSI
    wsi = WSI(name=args.wsi.stem, patient=args.wsi.stem,
              slide_path=args.wsi.resolve(), patched_slide_path=folder)
    check_wsi(wsi, magnification=args.magnification)
    runner = CellSegmentationInference(model_path=args.checkpoint.resolve(), gpu=args.gpu)
    runner.process_wsi(wsi=wsi, subdir_name="rescope", batch_size=args.batch_size, geojson=False)
    if not (output / "cells.json").is_file():
        raise RuntimeError("CellViT finished without cells.json")
    print(output / "cells.json")


if __name__ == "__main__":
    main()
