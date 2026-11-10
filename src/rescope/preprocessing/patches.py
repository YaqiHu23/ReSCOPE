"""Create 1024-pixel CellViT patches with 64-pixel overlap, in full-slide coordinates."""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image
import yaml

from .image_io import open_reader


def create_patches(wsi_path, output_root, mpp, magnification=40, reader="auto"):
    if not math.isfinite(mpp) or mpp <= 0 or magnification not in (20, 40):
        raise ValueError("Use positive mpp and magnification 20 or 40")
    wsi_path, output_root = Path(wsi_path), Path(output_root)
    slide = open_reader(str(wsi_path), reader)
    try:
        width, height = slide.dimensions
        folder = output_root / wsi_path.stem
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "patches").mkdir()
        (folder / "metadata").mkdir()
        size, overlap, stride, margin = 1024, 64, 960, 32
        n_rows, n_cols = math.ceil(height / stride), math.ceil(width / stride)
        meta = {"orig_n_tiles_cols": n_cols, "orig_n_tiles_rows": n_rows,
                "base_magnification": magnification, "magnification": magnification,
                "downsampling": 1, "label_map": {"Background": 0},
                "patch_overlap": overlap, "patch_size": size, "base_mpp": mpp,
                "target_patch_mpp": mpp, "stain_normalization": False}
        index = []
        for row in range(n_rows):
            for col in range(n_cols):
                # CellViT uses row=y, col=x and flips the row/column offset
                # before adding it to local xy centroids and contours.
                left, top = col * stride - margin, row * stride - margin
                x0, y0 = max(0, left), max(0, top)
                x1, y1 = min(width, left + size), min(height, top + size)
                patch = np.full((size, size, 3), 255, dtype=np.uint8)
                if x1 > x0 and y1 > y0:
                    patch[y0-top:y1-top, x0-left:x1-left] = slide.read_patch(x0, y0, x1-x0, y1-y0)
                bg = float(np.all(patch >= 235, axis=2).mean())
                if 1 - bg < 0.01:
                    continue
                name = f"{wsi_path.stem}_{row}_{col}"
                Image.fromarray(patch).save(folder / "patches" / f"{name}.png")
                item = {"row": row, "col": col, "background_ratio": bg,
                        "intersected_labels": [], "label_ratio": {}}
                (folder / "metadata" / f"{name}.yaml").write_text(yaml.safe_dump(dict(item, wsi_metadata=meta)))
                index.append({f"{name}.png": dict(item, metadata_path=f"./metadata/{name}.yaml")})
        if not index:
            raise ValueError("No tissue patches remain after background filtering")
        meta["patch_distribution"] = {"Background": len(index)}
        (folder / "metadata.yaml").write_text(yaml.safe_dump(meta))
        (folder / "patch_metadata.json").write_text(json.dumps(index))
        (output_root / "processed.json").write_text(json.dumps({"processed_files": [wsi_path.stem]}))
        print(f"CellViT patches: {len(index)} -> {folder}", flush=True)
        return folder
    finally:
        slide.close()
