#!/usr/bin/env python3
"""
Lazy H&E-to-UNI embedding preprocessing.

This script does not write H&E tile images. It treats the WSI as a virtual
tile grid, reads each patch on demand, runs UNI, and writes embeddings plus
metadata in the format consumed by the current training code.
"""

import argparse
import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    class SimpleTqdm:
        def __init__(self, iterable=None, total=None, desc=None, leave=True):
            self.iterable = iterable
            self.total = total
            self.desc = desc or "progress"
            self.count = 0

        def __iter__(self):
            for item in self.iterable:
                yield item

        def update(self, n=1):
            self.count += n
            if self.total:
                print(f"{self.desc}: {self.count}/{self.total}", flush=True)

        def set_postfix(self, values):
            if values:
                print(f"{self.desc}: {values}", flush=True)

        def close(self):
            pass

    def tqdm(iterable=None, total=None, desc=None, leave=True):
        return SimpleTqdm(iterable=iterable, total=total, desc=desc, leave=leave)

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm


from .image_io import SlideReader, open_reader, read_thumbnail, normalize_rgb_image, to_uint8, pad_patch

Image.MAX_IMAGE_PIXELS = None


def format_um(value: float) -> str:
    if float(value).is_integer():
        return str(int(round(value)))
    return str(value).replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read virtual H&E tiles directly from a WSI, run UNI, and save "
            "embeddings.npy/meta.csv without materializing tile image files."
        )
    )
    parser.add_argument("--wsi-path", required=True, help="Path to H&E WSI/image.")
    parser.add_argument("--sample", required=True, help="Sample name, e.g. colon_P2.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for embeddings.npy, meta.csv and manifest.json.",
    )
    parser.add_argument(
        "--weight-path",
        required=True,
        help="Path to the locally downloaded UNI2-h pytorch_model.bin.",
    )
    parser.add_argument("--tile-um", type=float, default=8.0, help="Tile size in microns.")
    parser.add_argument("--mpp", type=float, required=True, help="Microns per pixel.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")),
        help="PyTorch DataLoader workers. Each worker opens its own slide handle.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--reader",
        default="auto",
        choices=["auto", "openslide", "tiff-zarr", "tiff-memmap", "tiff-memory"],
        help=(
            "Slide reader backend. auto tries OpenSlide, tifffile+zarr, "
            "tifffile memmap, then a full in-memory tifffile fallback."
        ),
    )
    parser.add_argument(
        "--skip-background",
        action="store_true",
        help="Use a thumbnail white mask to skip mostly-background tiles before UNI.",
    )
    parser.add_argument(
        "--white-threshold",
        type=int,
        default=235,
        help="Background-like thumbnail pixel if all RGB channels are >= this value.",
    )
    parser.add_argument(
        "--max-white-fraction",
        type=float,
        default=0.8,
        help="Keep tile only if thumbnail-estimated white fraction is below this.",
    )
    parser.add_argument(
        "--thumbnail-max-side",
        type=int,
        default=4096,
        help="Max side for background-mask thumbnail.",
    )
    parser.add_argument(
        "--pad-value",
        type=int,
        default=0,
        help="Pad value for edge tiles. Default is 0, matching the research extraction settings.",
    )
    parser.add_argument(
        "--limit-tiles",
        type=int,
        default=None,
        help="Optional debug limit after background filtering.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Zero-based contiguous shard index after background filtering.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Number of contiguous shards. Shard boundaries align to batch size.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already exists.",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=500000,
        help="Progress interval while scoring background tiles.",
    )
    return parser.parse_args()
























def build_uni_model(weight_path: str, device: str) -> torch.nn.Module:
    timm_kwargs = {
        "model_name": "vit_giant_patch14_224",
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }

    model = timm.create_model(pretrained=False, **timm_kwargs)
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def make_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )




def integral_sum(integral: np.ndarray, y0: int, x0: int, y1: int, x1: int) -> int:
    return int(integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0])


def estimate_white_fractions(
    wsi_path: str,
    reader_kind: str,
    full_hw: Tuple[int, int],
    tile_px: int,
    num_tiles_y: int,
    num_tiles_x: int,
    white_threshold: int,
    max_side: int,
    report_every: int,
) -> np.ndarray:
    thumb, thumb_full_hw = read_thumbnail(wsi_path, reader_kind, max_side)
    if tuple(thumb_full_hw) != tuple(full_hw):
        print(
            f"[WARN] Thumbnail full H/W {thumb_full_hw} differs from reader H/W {full_hw}; "
            "using reader dimensions for coordinate scaling.",
            flush=True,
        )

    background = (thumb >= int(white_threshold)).all(axis=-1).astype(np.int64)
    integral = np.pad(background.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
    thumb_h, thumb_w = background.shape
    full_h, full_w = full_hw
    row_scale = thumb_h / float(full_h)
    col_scale = thumb_w / float(full_w)

    total = num_tiles_y * num_tiles_x
    fractions = np.empty(total, dtype=np.float32)
    pbar = tqdm(total=total, desc="background scoring", leave=True)

    for row in range(num_tiles_y):
        y0 = row * tile_px
        y1 = min(y0 + tile_px, full_h)
        ty0 = min(max(int(math.floor(y0 * row_scale)), 0), thumb_h - 1)
        ty1 = min(max(int(math.ceil(y1 * row_scale)), ty0 + 1), thumb_h)

        for col in range(num_tiles_x):
            x0 = col * tile_px
            x1 = min(x0 + tile_px, full_w)
            tx0 = min(max(int(math.floor(x0 * col_scale)), 0), thumb_w - 1)
            tx1 = min(max(int(math.ceil(x1 * col_scale)), tx0 + 1), thumb_w)

            area = max((ty1 - ty0) * (tx1 - tx0), 1)
            white = integral_sum(integral, ty0, tx0, ty1, tx1)
            fractions[row * num_tiles_x + col] = white / float(area)

        pbar.update(num_tiles_x)
        scored = (row + 1) * num_tiles_x
        if report_every > 0 and scored % report_every < num_tiles_x:
            pbar.set_postfix({"scored": scored})

    pbar.close()
    return fractions


class LazyWSITileDataset(Dataset):
    def __init__(
        self,
        wsi_path: str,
        sample: str,
        tile_um: float,
        mpp: float,
        reader_kind: str,
        width: int,
        height: int,
        indices: Optional[np.ndarray],
        white_fractions: Optional[np.ndarray],
        transform,
        pad_value: int,
    ):
        self.wsi_path = str(wsi_path)
        self.sample = sample
        self.tile_um = float(tile_um)
        self.tile_um_name = format_um(tile_um)
        self.mpp = float(mpp)
        self.tile_px = max(1, int(round(self.tile_um / self.mpp)))
        self.reader_kind = reader_kind
        self.width = int(width)
        self.height = int(height)
        self.num_tiles_x = int(math.ceil(self.width / float(self.tile_px)))
        self.num_tiles_y = int(math.ceil(self.height / float(self.tile_px)))
        self.total_tiles = self.num_tiles_x * self.num_tiles_y
        self.indices = indices
        self.white_fractions = white_fractions
        self.transform = transform
        self.pad_value = int(pad_value)
        self._reader = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_reader"] = None
        return state

    def __len__(self) -> int:
        if self.indices is None:
            return self.total_tiles
        return int(len(self.indices))

    def _get_reader(self) -> SlideReader:
        if self._reader is None:
            self._reader = open_reader(self.wsi_path, self.reader_kind)
        return self._reader

    def _linear_index(self, idx: int) -> int:
        if self.indices is None:
            return int(idx)
        return int(self.indices[idx])

    def _white_fraction(self, linear_idx: int) -> float:
        if self.white_fractions is None:
            return float("nan")
        return float(self.white_fractions[linear_idx])

    def _pseudo_path(self, row: int, col: int) -> str:
        name = f"{self.sample}_tile_{self.tile_um_name}um_{row}_{col}_he.tif"
        return os.path.join("virtual_tiles", f"tile_{self.tile_um_name}um", self.sample, name)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        linear_idx = self._linear_index(idx)
        row = linear_idx // self.num_tiles_x
        col = linear_idx % self.num_tiles_x
        x0 = col * self.tile_px
        y0 = row * self.tile_px
        read_w = min(self.tile_px, self.width - x0)
        read_h = min(self.tile_px, self.height - y0)

        patch = self._get_reader().read_patch(x0, y0, read_w, read_h)
        patch = to_uint8(normalize_rgb_image(patch))
        patch = pad_patch(patch, self.tile_px, self.tile_px, self.pad_value)

        image = Image.fromarray(patch, mode="RGB")
        x = self.transform(image)

        return {
            "image": x,
            "path": self._pseudo_path(row, col),
            "row": int(row),
            "col": int(col),
            "linear_index": int(linear_idx),
            "x0": int(x0),
            "y0": int(y0),
            "width_px": int(read_w),
            "height_px": int(read_h),
            "white_fraction": self._white_fraction(linear_idx),
        }


def write_json(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}. Use --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def tensor_to_list(value) -> List[object]:
    if torch.is_tensor(value):
        return value.cpu().tolist()
    return list(value)


def main() -> None:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if args.mpp <= 0 or args.tile_um <= 0 or args.batch_size < 1:
        raise ValueError("mpp, tile_um, and batch_size must be positive")
    if not Path(args.wsi_path).is_file() or not Path(args.weight_path).is_file():
        raise FileNotFoundError("Provide an existing WSI and UNI2-h checkpoint")

    tile_um_name = format_um(args.tile_um)
    tile_size_name = f"tile_{tile_um_name}um"
    out_dir = Path(args.output_dir).expanduser().resolve()
    ensure_output_dir(out_dir, args.overwrite)

    print("=== Lazy UNI embedding preprocessing ===", flush=True)
    print(f"WSI: {args.wsi_path}", flush=True)
    print(f"Sample: {args.sample}", flush=True)
    print(f"Output: {out_dir}", flush=True)
    print(f"tile_um={args.tile_um} mpp={args.mpp}", flush=True)

    probe_reader = open_reader(args.wsi_path, args.reader)
    width, height = probe_reader.dimensions
    reader_name = probe_reader.backend_name
    probe_reader.close()
    runtime_reader = reader_name

    tile_px = max(1, int(round(float(args.tile_um) / float(args.mpp))))
    num_tiles_x = int(math.ceil(width / float(tile_px)))
    num_tiles_y = int(math.ceil(height / float(tile_px)))
    total_tiles = num_tiles_x * num_tiles_y
    print(
        f"Reader: {reader_name}; image W/H={width}x{height}; tile_px={tile_px}; "
        f"grid={num_tiles_y}x{num_tiles_x}; total={total_tiles}",
        flush=True,
    )
    if runtime_reader == "tiff-memory" and args.num_workers > 0:
        print(
            "[WARN] tiff-memory reader loads the whole image. Setting num_workers=0 "
            "to avoid one full image copy per worker.",
            flush=True,
        )
        args.num_workers = 0

    white_fractions = None
    indices = None
    if args.skip_background:
        full_hw = (height, width)
        white_fractions = estimate_white_fractions(
            wsi_path=args.wsi_path,
            reader_kind=args.reader,
            full_hw=full_hw,
            tile_px=tile_px,
            num_tiles_y=num_tiles_y,
            num_tiles_x=num_tiles_x,
            white_threshold=args.white_threshold,
            max_side=args.thumbnail_max_side,
            report_every=args.report_every,
        )
        keep_mask = white_fractions < float(args.max_white_fraction)
        indices = np.flatnonzero(keep_mask).astype(np.int64)
        print(
            f"Background filter kept {len(indices)} / {total_tiles} tiles "
            f"({len(indices) / max(total_tiles, 1):.2%})",
            flush=True,
        )
        np.save(out_dir / "white_fractions.npy", white_fractions)
        np.save(out_dir / "keep_indices.npy", indices)

    if args.limit_tiles is not None:
        if indices is None:
            indices = np.arange(min(args.limit_tiles, total_tiles), dtype=np.int64)
        else:
            indices = indices[: args.limit_tiles]
        print(f"[DEBUG] limit_tiles active: embedding {len(indices)} tiles", flush=True)

    shard_start = 0
    tiles_after_filter_before_shard = total_tiles if indices is None else len(indices)
    shard_end = tiles_after_filter_before_shard
    if (args.shard_index is None) != (args.num_shards is None):
        raise ValueError("--shard-index and --num-shards must be supplied together")
    if args.num_shards is not None:
        if args.num_shards <= 0:
            raise ValueError("--num-shards must be positive")
        if args.shard_index < 0 or args.shard_index >= args.num_shards:
            raise ValueError("--shard-index must be in [0, num_shards)")
        if indices is None:
            indices = np.arange(total_tiles, dtype=np.int64)
        aligned_chunk = int(
            math.ceil(len(indices) / float(args.num_shards * args.batch_size))
            * args.batch_size
        )
        shard_start = min(args.shard_index * aligned_chunk, len(indices))
        shard_end = min(shard_start + aligned_chunk, len(indices))
        indices = indices[shard_start:shard_end]
        print(
            f"[INFO] shard {args.shard_index + 1}/{args.num_shards}: "
            f"filtered rows [{shard_start}, {shard_end}) = {len(indices)} tiles",
            flush=True,
        )

    transform = make_transform()
    dataset = LazyWSITileDataset(
        wsi_path=args.wsi_path,
        sample=args.sample,
        tile_um=args.tile_um,
        mpp=args.mpp,
        reader_kind=runtime_reader,
        width=width,
        height=height,
        indices=indices,
        white_fractions=white_fractions,
        transform=transform,
        pad_value=args.pad_value,
    )

    if len(dataset) == 0:
        raise RuntimeError("No tiles to embed after filtering.")

    dl = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=args.device == "cuda",
    )

    print("[INFO] Building UNI model", flush=True)
    model = build_uni_model(args.weight_path, args.device)
    print("[INFO] Model ready", flush=True)

    embeddings_path = out_dir / "embeddings.npy"
    meta_path = out_dir / "meta.csv"
    paths_path = out_dir / "paths.txt"
    manifest_path = out_dir / "manifest.json"

    fieldnames = [
        "path",
        "tile_size",
        "sample",
        "tile_um",
        "mpp",
        "tile_px",
        "row",
        "col",
        "linear_index",
        "x0",
        "y0",
        "width_px",
        "height_px",
        "white_fraction",
        "source_wsi",
    ]

    emb_memmap = None
    write_offset = 0

    with meta_path.open("w", newline="") as meta_f, paths_path.open("w") as paths_f:
        writer = csv.DictWriter(meta_f, fieldnames=fieldnames)
        writer.writeheader()

        with torch.no_grad():
            pbar = tqdm(dl, desc=f"{tile_size_name} embedding", leave=True)
            for batch in pbar:
                x = batch["image"].to(args.device, non_blocking=True)
                emb = model(x).detach().cpu().numpy().astype(np.float32, copy=False)

                if emb_memmap is None:
                    emb_dim = int(emb.shape[1])
                    emb_memmap = np.lib.format.open_memmap(
                        embeddings_path,
                        mode="w+",
                        dtype=np.float32,
                        shape=(len(dataset), emb_dim),
                    )

                batch_size = emb.shape[0]
                emb_memmap[write_offset : write_offset + batch_size] = emb

                paths = list(batch["path"])
                rows = tensor_to_list(batch["row"])
                cols = tensor_to_list(batch["col"])
                linear_indices = tensor_to_list(batch["linear_index"])
                x0s = tensor_to_list(batch["x0"])
                y0s = tensor_to_list(batch["y0"])
                widths = tensor_to_list(batch["width_px"])
                heights = tensor_to_list(batch["height_px"])
                white_vals = tensor_to_list(batch["white_fraction"])

                for i, path in enumerate(paths):
                    paths_f.write(path + "\n")
                    white_fraction = white_vals[i]
                    if isinstance(white_fraction, float) and math.isnan(white_fraction):
                        white_text = ""
                    else:
                        white_text = f"{float(white_fraction):.6f}"
                    writer.writerow(
                        {
                            "path": path,
                            "tile_size": tile_size_name,
                            "sample": args.sample,
                            "tile_um": args.tile_um,
                            "mpp": args.mpp,
                            "tile_px": tile_px,
                            "row": int(rows[i]),
                            "col": int(cols[i]),
                            "linear_index": int(linear_indices[i]),
                            "x0": int(x0s[i]),
                            "y0": int(y0s[i]),
                            "width_px": int(widths[i]),
                            "height_px": int(heights[i]),
                            "white_fraction": white_text,
                            "source_wsi": args.wsi_path,
                        }
                    )

                write_offset += batch_size
                if args.device == "cuda":
                    mem_gb = torch.cuda.memory_allocated() / 1e9
                else:
                    mem_gb = 0.0
                pbar.set_postfix({"written": write_offset, "mem_GB": f"{mem_gb:.2f}"})

    if emb_memmap is None:
        raise RuntimeError("Embedding loop finished without writing any batches.")
    emb_memmap.flush()

    manifest = {
        "wsi_path": args.wsi_path,
        "sample": args.sample,
        "tile_size": tile_size_name,
        "tile_um": args.tile_um,
        "mpp": args.mpp,
        "tile_px": tile_px,
        "image_width": width,
        "image_height": height,
        "num_tiles_x": num_tiles_x,
        "num_tiles_y": num_tiles_y,
        "total_tiles_before_filter": total_tiles,
        "tiles_embedded": len(dataset),
        "feature_dim": int(emb_memmap.shape[1]),
        "tiles_after_filter_before_shard": tiles_after_filter_before_shard,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "shard_start": shard_start,
        "shard_end": shard_end,
        "reader": reader_name,
        "skip_background": bool(args.skip_background),
        "white_threshold": args.white_threshold if args.skip_background else None,
        "max_white_fraction": args.max_white_fraction if args.skip_background else None,
        "weight_path": args.weight_path,
        "embedding_path": str(embeddings_path),
        "meta_path": str(meta_path),
        "paths_path": str(paths_path),
    }
    write_json(manifest_path, manifest)

    print(f"[DONE] embeddings: {embeddings_path} shape=({len(dataset)}, {emb_memmap.shape[1]})", flush=True)
    print(f"[DONE] meta: {meta_path}", flush=True)
    print(f"[DONE] paths: {paths_path}", flush=True)
    print(f"[DONE] manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
