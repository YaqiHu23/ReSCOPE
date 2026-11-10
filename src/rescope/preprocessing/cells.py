"""Convert CellViT global xy nuclei into the ReSCOPE cell table."""
import csv
import gzip
import json
import math
from pathlib import Path


def convert_cells(cells_json, output_dir, image_width, image_height, hd_dir=None):
    cells_json, output_dir = Path(cells_json), Path(output_dir)
    if not cells_json.is_file():
        raise FileNotFoundError(cells_json)
    if hd_dir is not None:
        from .hd_cells import process_paths, resolve_square_002um_dir
        square = resolve_square_002um_dir(Path(hd_dir))
        for file in (square / "filtered_feature_bc_matrix.h5", square / "spatial/tissue_positions.parquet"):
            if not file.is_file():
                raise FileNotFoundError(file)
        output_dir.mkdir(parents=True, exist_ok=False)
        report = process_paths("sample", cells_json, square / "filtered_feature_bc_matrix.h5",
                               square / "spatial/tissue_positions.parquet", output_dir,
                               cell_coord_mode="xy")
        if report["n_cells_with_2um_barcodes"] < 2:
            raise ValueError("Fewer than two cells received Visium HD bins; check image alignment")
        return output_dir / "cell_metadata.csv.gz"
    import ijson
    output_dir.mkdir(parents=True, exist_ok=False)
    temp = output_dir / "cell_metadata.partial.csv.gz"
    count, outside = 0, 0
    try:
        with cells_json.open("rb") as source, gzip.open(temp, "wt", newline="") as out:
            writer = csv.writer(out)
            writer.writerow(["cell_id", "centroid_x", "centroid_y", "cell_type"])
            for index, cell in enumerate(ijson.items(source, "cells.item")):
                xy, kind = cell["centroid"], cell["type"]
                if len(xy) != 2:
                    raise ValueError(f"Invalid centroid at cell {index}")
                x, y = map(float, xy)
                value = float(kind)
                if not all(map(math.isfinite, (x, y, value))) or not value.is_integer() or int(value) not in range(6):
                    raise ValueError(f"Invalid coordinate or CellViT type at cell {index}")
                if not (0 <= x < image_width and 0 <= y < image_height):
                    outside += 1
                    continue
                writer.writerow([f"cell_{index:07d}", x, y, int(value)])
                count += 1
        if count < 2:
            raise ValueError("Fewer than two valid cells inside the image")
        output = output_dir / "cell_metadata.csv.gz"
        temp.replace(output)
        (output_dir / "summary.json").write_text(json.dumps({"cells": count, "outside_image": outside,
            "coordinate_order": "xy", "cell_population": "all annotated nuclei inside the image",
            "source_cells_json": str(cells_json.resolve())}, indent=2) + "\n")
        return output
    finally:
        if temp.exists():
            temp.unlink()
