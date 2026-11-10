"""Optional Visium HD 2-um bin aggregation, retaining cells assigned at least one bin."""
import argparse


import csv


import gzip


import json


from pathlib import Path


import h5py


import ijson


import numpy as np


import pyarrow.parquet as pq


from scipy import sparse


from shapely.geometry import Polygon, box


from shapely.prepared import prep


BARCODE_FMT = "s_002um_{:05d}_{:05d}-1"


def parse_barcode_chunk(barcode_chunk):
    rows = np.fromiter((int(x[8:13]) for x in barcode_chunk), dtype=np.int32, count=len(barcode_chunk))
    cols = np.fromiter((int(x[14:19]) for x in barcode_chunk), dtype=np.int32, count=len(barcode_chunk))
    return rows, cols


def make_coord_keys(rows, cols):
    return (rows.astype(np.uint64) << np.uint64(32)) | cols.astype(np.uint64)


def load_sorted_barcode_keys(barcode_ds, chunk_size=500000):
    n_barcodes = len(barcode_ds)
    keys = np.empty(n_barcodes, dtype=np.uint64)
    for start in range(0, n_barcodes, chunk_size):
        end = min(start + chunk_size, n_barcodes)
        barcode_chunk = barcode_ds[start:end]
        rows, cols = parse_barcode_chunk(barcode_chunk)
        keys[start:end] = make_coord_keys(rows, cols)
        print("  parsed barcode keys: {} / {}".format(end, n_barcodes), flush=True)
    order = np.argsort(keys)
    return keys[order], order.astype(np.int64)


def load_or_build_barcode_key_cache(barcode_ds, cache_path: Path):
    if cache_path.exists():
        with np.load(cache_path) as cached:
            print("  loaded barcode key cache: {}".format(cache_path), flush=True)
            return cached["sorted_keys"], cached["sorted_cols"]

    sorted_keys, sorted_cols = load_sorted_barcode_keys(barcode_ds)
    np.savez_compressed(cache_path, sorted_keys=sorted_keys, sorted_cols=sorted_cols)
    print("  wrote barcode key cache: {}".format(cache_path), flush=True)
    return sorted_keys, sorted_cols


def lookup_column_indices(sorted_keys, sorted_cols, query_keys):
    if len(query_keys) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=bool)
    pos = np.searchsorted(sorted_keys, query_keys)
    ok = np.zeros(len(query_keys), dtype=bool)
    in_bounds = pos < len(sorted_keys)
    ok[in_bounds] = sorted_keys[pos[in_bounds]] == query_keys[in_bounds]
    cols = np.full(len(query_keys), -1, dtype=np.int64)
    cols[ok] = sorted_cols[pos[ok]]
    return cols, ok


def load_affine_from_parquet(parquet_path: Path):
    want = {(0, 0), (0, 1), (1, 0)}
    hits = {}
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(
        batch_size=5000,
        columns=["array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"],
    ):
        for row in batch.to_pylist():
            key = (row["array_row"], row["array_col"])
            if key in want:
                hits[key] = np.array(
                    [float(row["pxl_col_in_fullres"]), float(row["pxl_row_in_fullres"])],
                    dtype=np.float64,
                )
        if len(hits) == len(want):
            break
    if len(hits) != len(want):
        raise RuntimeError("Failed to recover affine anchors from {}".format(parquet_path))

    p00 = hits[(0, 0)]
    p01 = hits[(0, 1)]
    p10 = hits[(1, 0)]
    v_row = p10 - p00
    v_col = p01 - p00
    affine = np.column_stack([v_row, v_col])
    inv_affine = np.linalg.inv(affine)
    return p00, affine, inv_affine


def polygon_area(points_xy):
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def iter_cells(cells_json_path: Path):
    with cells_json_path.open("rb") as handle:
        yielded = 0
        for index, cell in enumerate(ijson.items(handle, "cells.item")):
            yielded += 1
            yield index, cell

    if yielded > 0:
        return

    with cells_json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    cells = payload.get("cells", [])
    if not isinstance(cells, list):
        raise RuntimeError("Could not find a valid 'cells' list in {}".format(cells_json_path))
    for index, cell in enumerate(cells):
        yield index, cell


def normalize_xy_pair(point_like, coord_mode):
    point = np.asarray(point_like, dtype=np.float64)
    if point.shape != (2,):
        raise ValueError("Expected 2D point, got shape {}".format(point.shape))
    if coord_mode == "xy":
        return point
    if coord_mode == "row_col":
        return np.array([point[1], point[0]], dtype=np.float64)
    raise ValueError("Unsupported cell coordinate mode: {}".format(coord_mode))


def normalize_contour_xy(contour_like, coord_mode):
    contour = np.asarray(contour_like, dtype=np.float64)
    if contour.ndim != 2 or contour.shape[1] != 2:
        return contour
    if coord_mode == "xy":
        return contour
    if coord_mode == "row_col":
        return contour[:, ::-1].copy()
    raise ValueError("Unsupported cell coordinate mode: {}".format(coord_mode))


def normalize_bbox_xy(bbox_like, coord_mode):
    bbox = np.asarray(bbox_like, dtype=np.float64)
    if bbox.shape != (2, 2):
        raise ValueError("Expected bbox shape (2, 2), got {}".format(bbox.shape))
    bbox_xy = np.vstack(
        [
            normalize_xy_pair(bbox[0], coord_mode),
            normalize_xy_pair(bbox[1], coord_mode),
        ]
    )
    xmin = float(min(bbox_xy[0, 0], bbox_xy[1, 0]))
    ymin = float(min(bbox_xy[0, 1], bbox_xy[1, 1]))
    xmax = float(max(bbox_xy[0, 0], bbox_xy[1, 0]))
    ymax = float(max(bbox_xy[0, 1], bbox_xy[1, 1]))
    return xmin, ymin, xmax, ymax


def contour_xy_to_grid(contour_xy, origin_xy, inv_affine):
    rc = (contour_xy - origin_xy) @ inv_affine.T
    return np.column_stack([rc[:, 1], rc[:, 0]])


def build_cell_assignments(
    cells_json_path,
    sorted_keys,
    sorted_cols,
    origin_xy,
    affine,
    inv_affine,
    max_cells=None,
    cell_coord_mode="xy",
):
    n_barcodes = len(sorted_cols)
    assigned_cell = np.full(n_barcodes, -1, dtype=np.int32)
    assigned_dist2 = np.full(n_barcodes, np.inf, dtype=np.float32)
    cell_records = {}
    overlap_replaced = 0

    for cell_idx, cell in iter_cells(cells_json_path):
        if max_cells is not None and cell_idx >= max_cells:
            break

        contour = normalize_contour_xy(cell.get("contour", []), cell_coord_mode)
        if contour.ndim != 2 or contour.shape[0] < 3:
            continue
        contour_grid = contour_xy_to_grid(contour, origin_xy, inv_affine)
        polygon = Polygon(contour_grid)
        if polygon.is_empty:
            continue
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty:
            continue
        prepared_polygon = prep(polygon)

        min_col = int(np.floor(polygon.bounds[0] - 0.5))
        max_col = int(np.ceil(polygon.bounds[2] + 0.5))
        min_row = int(np.floor(polygon.bounds[1] - 0.5))
        max_row = int(np.ceil(polygon.bounds[3] + 0.5))
        if max_col < 0 or max_row < 0:
            continue
        if min_col < 0:
            min_col = 0
        if min_row < 0:
            min_row = 0

        cand_cols = np.arange(min_col, max_col + 1, dtype=np.int32)
        cand_rows = np.arange(min_row, max_row + 1, dtype=np.int32)
        if len(cand_cols) == 0 or len(cand_rows) == 0:
            continue

        grid_cols, grid_rows = np.meshgrid(cand_cols, cand_rows)
        candidate_points = np.column_stack([grid_cols.ravel(), grid_rows.ravel()])
        hit_mask = np.fromiter(
            (
                prepared_polygon.intersects(
                    box(col - 0.5, row - 0.5, col + 0.5, row + 0.5)
                )
                for col, row in candidate_points
            ),
            dtype=bool,
            count=len(candidate_points),
        )
        if not np.any(hit_mask):
            continue

        inside_points = candidate_points[hit_mask]
        query_keys = make_coord_keys(inside_points[:, 1], inside_points[:, 0])
        column_indices, ok = lookup_column_indices(sorted_keys, sorted_cols, query_keys)
        if not np.any(ok):
            continue

        column_indices = column_indices[ok]
        inside_points = inside_points[ok]
        spot_xy = origin_xy + inside_points[:, 1:2] * affine[:, 0] + inside_points[:, 0:1] * affine[:, 1]

        centroid = normalize_xy_pair(cell["centroid"], cell_coord_mode)
        dist2 = np.sum((spot_xy - centroid[None, :]) ** 2, axis=1).astype(np.float32)

        prev_dist = assigned_dist2[column_indices]
        better = dist2 < prev_dist
        overlap_replaced += int(np.count_nonzero((assigned_cell[column_indices] >= 0) & better))
        if np.any(better):
            update_cols = column_indices[better]
            assigned_cell[update_cols] = cell_idx
            assigned_dist2[update_cols] = dist2[better]

        bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = normalize_bbox_xy(cell["bbox"], "row_col")
        cell_records[cell_idx] = {
            "raw_cell_index": cell_idx,
            "cell_id": "cell_{:09d}".format(cell_idx),
            "centroid_x": float(centroid[0]),
            "centroid_y": float(centroid[1]),
            "bbox_xmin": bbox_xmin,
            "bbox_ymin": bbox_ymin,
            "bbox_xmax": bbox_xmax,
            "bbox_ymax": bbox_ymax,
            "polygon_area_fullres": float(polygon_area(contour)),
            "cell_type": int(cell.get("type", -1)),
            "type_prob": float(cell.get("type_prob", 0.0)),
            "cell_status": int(cell.get("cell_status", -1)),
            "edge_position": bool(cell.get("edge_position", False)),
        }

        if cell_idx % 5000 == 0:
            assigned = int(np.count_nonzero(assigned_cell >= 0))
            print("  processed cells: {} assigned_barcodes={}".format(cell_idx + 1, assigned), flush=True)

    return assigned_cell, assigned_dist2, cell_records, overlap_replaced


def aggregate_matrix_by_cell(h5_path: Path, assigned_cell, kept_old_to_new, n_output_cells, out_dir: Path):
    with h5py.File(h5_path, "r") as handle:
        matrix = handle["matrix"]
        indptr = matrix["indptr"]
        indices = matrix["indices"]
        data = matrix["data"]
        feature_ids = [x.decode("utf-8") for x in matrix["features"]["id"][:]]
        feature_names = [x.decode("utf-8") for x in matrix["features"]["name"][:]]
        feature_types = [x.decode("utf-8") for x in matrix["features"]["feature_type"][:]]
        n_genes = int(matrix["shape"][0])

        assigned_mask = assigned_cell >= 0
        assigned_cols = np.flatnonzero(assigned_mask)
        assigned_rows = kept_old_to_new[assigned_cell[assigned_cols]]
        order = np.argsort(assigned_rows, kind="mergesort")
        assigned_cols = assigned_cols[order]
        assigned_rows = assigned_rows[order]

        unique_rows, starts = np.unique(assigned_rows, return_index=True)
        starts = np.append(starts, len(assigned_rows))

        out_indptr = np.zeros(n_output_cells + 1, dtype=np.int64)
        all_indices = []
        all_data = []
        total_counts = np.zeros(n_output_cells, dtype=np.int64)

        for row_pos, cell_row in enumerate(unique_rows):
            lo = starts[row_pos]
            hi = starts[row_pos + 1]
            cols = assigned_cols[lo:hi]

            gene_parts = []
            value_parts = []
            for col in cols:
                p0 = int(indptr[col])
                p1 = int(indptr[col + 1])
                if p1 <= p0:
                    continue
                gene_parts.append(indices[p0:p1])
                value_parts.append(data[p0:p1])

            if gene_parts:
                genes = np.concatenate(gene_parts).astype(np.int32, copy=False)
                values = np.concatenate(value_parts).astype(np.int64, copy=False)
                order2 = np.argsort(genes, kind="mergesort")
                genes = genes[order2]
                values = values[order2]
                uniq_genes, uniq_start = np.unique(genes, return_index=True)
                summed = np.add.reduceat(values, uniq_start)
                all_indices.append(uniq_genes.astype(np.int32, copy=False))
                all_data.append(summed.astype(np.int64, copy=False))
                out_indptr[cell_row + 1] = out_indptr[cell_row] + len(uniq_genes)
                total_counts[cell_row] = int(np.sum(summed))
            else:
                out_indptr[cell_row + 1] = out_indptr[cell_row]

            if row_pos % 5000 == 0:
                print("  aggregated cells: {} / {}".format(row_pos + 1, len(unique_rows)), flush=True)

        for idx in range(1, len(out_indptr)):
            if out_indptr[idx] == 0 and out_indptr[idx - 1] != 0:
                out_indptr[idx] = out_indptr[idx - 1]

        matrix_indices = (
            np.concatenate(all_indices).astype(np.int32, copy=False)
            if all_indices
            else np.empty(0, dtype=np.int32)
        )
        matrix_data = (
            np.concatenate(all_data).astype(np.int64, copy=False)
            if all_data
            else np.empty(0, dtype=np.int64)
        )

        cell_gene_csr = sparse.csr_matrix(
            (matrix_data, matrix_indices, out_indptr),
            shape=(n_output_cells, n_genes),
            dtype=np.int64,
        )

        sparse.save_npz(out_dir / "cell_gene_counts.npz", cell_gene_csr, compressed=True)

        with gzip.open(out_dir / "features.tsv.gz", "wt", encoding="utf-8", newline="") as handle_out:
            writer = csv.writer(handle_out, delimiter="\t")
            for feature_id, feature_name, feature_type in zip(feature_ids, feature_names, feature_types):
                writer.writerow([feature_id, feature_name, feature_type])

        n_genes_detected = np.diff(cell_gene_csr.indptr).astype(np.int64, copy=False)
        return total_counts, n_genes_detected


def write_cell_ids(cell_ids, out_path: Path):
    with gzip.open(out_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        for cell_id in cell_ids:
            writer.writerow([cell_id])


def key_to_barcode(coord_key):
    row = int(coord_key >> np.uint64(32))
    col = int(coord_key & np.uint64(0xFFFFFFFF))
    return BARCODE_FMT.format(row, col)


def write_barcode_mapping_from_columns(
    out_path: Path,
    assigned_cell,
    kept_old_to_new,
    kept_cell_ids,
    sorted_keys,
    sorted_cols,
):
    assigned_cols = np.flatnonzero(assigned_cell >= 0)
    old_rows = assigned_cell[assigned_cols]
    new_rows = kept_old_to_new[old_rows]
    order = np.argsort(new_rows, kind="mergesort")
    assigned_cols = assigned_cols[order]
    old_rows = old_rows[order]
    new_rows = new_rows[order]

    inverse_order = np.argsort(sorted_cols)
    keys_by_original_col = sorted_keys[inverse_order]

    with gzip.open(out_path, "wt", encoding="utf-8", newline="") as handle_out:
        writer = csv.writer(handle_out)
        writer.writerow(["barcode_002um", "raw_cell_index", "cell_id"])

        chunk_size = 200000
        for start in range(0, len(assigned_cols), chunk_size):
            end = min(start + chunk_size, len(assigned_cols))
            cols = assigned_cols[start:end]
            barcode_keys = keys_by_original_col[cols]
            for barcode_key, old_row, new_row in zip(
                barcode_keys, old_rows[start:end], new_rows[start:end]
            ):
                writer.writerow(
                    [key_to_barcode(barcode_key), int(old_row), kept_cell_ids[int(new_row)]]
                )


def write_cell_metadata(out_path: Path, kept_records):
    if not kept_records:
        return
    fieldnames = list(kept_records[0].keys())
    with gzip.open(out_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_records)


def resolve_square_002um_dir(dataset_dir: Path) -> Path:
    dataset_dir = dataset_dir.resolve()
    if dataset_dir.is_file():
        raise RuntimeError("dataset_dir must be a directory, got file: {}".format(dataset_dir))
    if (dataset_dir / "filtered_feature_bc_matrix.h5").exists():
        return dataset_dir
    candidate = dataset_dir / "binned_outputs" / "square_002um"
    if candidate.exists():
        return candidate
    raise RuntimeError(
        "Could not locate square_002um under {}. Pass either the square_002um directory "
        "itself or the extracted dataset root that contains binned_outputs/square_002um.".format(
            dataset_dir
        )
    )


def infer_default_output_dir(square_dir: Path) -> Path:
    if square_dir.name == "square_002um" and square_dir.parent.name == "binned_outputs":
        return square_dir.parent.parent / "single_cell_from_002um"
    return square_dir / "single_cell_from_002um"


def is_completed_output_dir(out_dir: Path) -> bool:
    return out_dir.exists() and (out_dir / "summary.json").exists()


def process_paths(
    example_id,
    cells_json: Path,
    h5_path: Path,
    pos_path: Path,
    out_dir: Path,
    max_cells=None,
    cell_coord_mode="xy",
):
    if not cells_json.exists():
        raise RuntimeError("missing cells.json: {}".format(cells_json))
    if not h5_path.exists():
        raise RuntimeError("missing square_002um filtered_feature_bc_matrix.h5: {}".format(h5_path))
    if not pos_path.exists():
        raise RuntimeError("missing square_002um tissue_positions.parquet: {}".format(pos_path))

    out_dir.mkdir(parents=True, exist_ok=True)

    print("[{}] loading 2um affine anchors".format(example_id), flush=True)
    origin_xy, affine, inv_affine = load_affine_from_parquet(pos_path)

    print("[{}] loading 2um barcode keys".format(example_id), flush=True)
    cache_path = out_dir / "barcode_key_index_002um.npz"
    with h5py.File(h5_path, "r") as handle_h5:
        sorted_keys, sorted_cols = load_or_build_barcode_key_cache(
            handle_h5["matrix"]["barcodes"], cache_path
        )

    print("[{}] assigning 2um barcodes to cells".format(example_id), flush=True)
    assigned_cell, _, cell_records, overlap_replaced = build_cell_assignments(
        cells_json,
        sorted_keys,
        sorted_cols,
        origin_xy,
        affine,
        inv_affine,
        max_cells=max_cells,
        cell_coord_mode=cell_coord_mode,
    )

    max_raw_cell_index = max(cell_records) if cell_records else -1
    barcode_counts_all = np.bincount(
        assigned_cell[assigned_cell >= 0],
        minlength=max_raw_cell_index + 1,
    ).astype(np.int64)
    valid_record_mask = np.zeros(max_raw_cell_index + 1, dtype=bool)
    for raw_idx in cell_records:
        valid_record_mask[int(raw_idx)] = True
    keep_mask = (barcode_counts_all > 0) & valid_record_mask
    kept_old_indices = np.flatnonzero(keep_mask)
    kept_old_to_new = np.full(max_raw_cell_index + 1, -1, dtype=np.int32)
    kept_old_to_new[kept_old_indices] = np.arange(len(kept_old_indices), dtype=np.int32)
    kept_records = []
    kept_cell_ids = []
    for old_idx in kept_old_indices:
        record = dict(cell_records[int(old_idx)])
        record["n_2um_barcodes"] = int(barcode_counts_all[int(old_idx)])
        kept_records.append(record)
        kept_cell_ids.append(record["cell_id"])

    print(
        "[{}] aggregating gene counts for {} cells".format(
            example_id, len(kept_records)
        ),
        flush=True,
    )
    total_counts, n_genes_detected = aggregate_matrix_by_cell(
        h5_path, assigned_cell, kept_old_to_new, len(kept_records), out_dir
    )

    for idx, record in enumerate(kept_records):
        record["total_counts"] = int(total_counts[idx])
        record["n_genes_detected"] = int(n_genes_detected[idx])

    write_cell_ids(kept_cell_ids, out_dir / "cell_ids.tsv.gz")
    write_cell_metadata(out_dir / "cell_metadata.csv.gz", kept_records)
    write_barcode_mapping_from_columns(
        out_dir / "barcode_to_cell.csv.gz",
        assigned_cell,
        kept_old_to_new,
        kept_cell_ids,
        sorted_keys,
        sorted_cols,
    )

    summary = {
        "example_id": example_id,
        "source_cells_json": str(cells_json),
        "source_h5_002um": str(h5_path),
        "source_positions_parquet": str(pos_path),
        "output_dir": str(out_dir),
        "cell_coord_mode": cell_coord_mode,
        "n_cells_total_seen": len(cell_records),
        "n_cells_with_2um_barcodes": len(kept_records),
        "n_assigned_2um_barcodes": int(np.count_nonzero(assigned_cell >= 0)),
        "n_overlap_reassignments": int(overlap_replaced),
        "affine_origin_xy": origin_xy.tolist(),
        "affine_row_vector_xy": affine[:, 0].tolist(),
        "affine_col_vector_xy": affine[:, 1].tolist(),
        "storage_layout": {
            "matrix": "cell_gene_counts.npz (cells x genes, scipy CSR)",
            "cells": "cell_ids.tsv.gz",
            "features": "features.tsv.gz",
            "cell_metadata": "cell_metadata.csv.gz",
            "barcode_mapping": "barcode_to_cell.csv.gz",
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary
