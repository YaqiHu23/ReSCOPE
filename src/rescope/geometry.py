import math
import json


def clamp_int(value, low, high):
    return max(low, min(int(value), high))


def compute_region_bounds(full_h, full_w, region_cfg):
    source = str(region_cfg.get("source", "ratio")).lower()
    if source == "stats_json":
        return compute_region_bounds_from_stats(full_h, full_w, region_cfg)
    if source == "region_size_txt":
        return compute_region_bounds_from_region_size_txt(full_h, full_w, region_cfg)

    top_ratio = float(region_cfg.get("top_ratio", 0.0))
    bottom_ratio = float(region_cfg.get("bottom_ratio", 1.0))
    left_ratio = float(region_cfg.get("left_ratio", 0.0))
    right_ratio = float(region_cfg.get("right_ratio", 1.0))
    shrink_ratio = float(region_cfg.get("shrink_ratio", 1.0))

    top = clamp_int(round(full_h * top_ratio), 0, full_h)
    bottom = clamp_int(round(full_h * bottom_ratio), 0, full_h)
    left = clamp_int(round(full_w * left_ratio), 0, full_w)
    right = clamp_int(round(full_w * right_ratio), 0, full_w)

    if bottom <= top or right <= left:
        raise ValueError(
            f"Invalid region ratios: top={top_ratio}, bottom={bottom_ratio}, left={left_ratio}, right={right_ratio}"
        )

    if shrink_ratio <= 0:
        raise ValueError(f"shrink_ratio must be > 0, got {shrink_ratio}")

    if shrink_ratio != 1.0:
        center_h = 0.5 * (top + bottom)
        center_w = 0.5 * (left + right)
        half_h = max(1.0, 0.5 * (bottom - top) * shrink_ratio)
        half_w = max(1.0, 0.5 * (right - left) * shrink_ratio)
        top = clamp_int(math.floor(center_h - half_h), 0, full_h - 1)
        bottom = clamp_int(math.ceil(center_h + half_h), top + 1, full_h)
        left = clamp_int(math.floor(center_w - half_w), 0, full_w - 1)
        right = clamp_int(math.ceil(center_w + half_w), left + 1, full_w)

    return {
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "height": bottom - top,
        "width": right - left,
    }


def load_region_size_txt(region_txt_path):
    parsed = {}
    current_section = None

    with open(region_txt_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("[") and line.endswith("]"):
                current_section = line[1:-1]
                parsed[current_section] = {}
                continue

            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"Invalid line in region size txt: {raw_line.rstrip()}")

            key, value = parts
            if current_section is None:
                parsed[key] = value
            else:
                parsed[current_section][key] = float(value)

    return parsed


def compute_region_bounds_from_stats(full_h, full_w, region_cfg):
    stats_json = region_cfg.get("stats_json")
    if not stats_json:
        raise ValueError("visium_hd_region.stats_json must be set when source=stats_json")

    with open(stats_json, "r", encoding="utf-8") as f:
        stats = json.load(f)

    bbox_key = str(region_cfg.get("bbox_key", "bbox_robust_q0p5_q99p5"))
    if bbox_key not in stats:
        raise KeyError(f"{bbox_key} not found in {stats_json}")

    bbox = stats[bbox_key]
    image_w = float(stats["image_width"])
    image_h = float(stats["image_height"])
    if image_w <= 0 or image_h <= 0:
        raise ValueError(f"Invalid image size in stats: width={image_w}, height={image_h}")

    left_ratio = float(bbox["x_min"]) / image_w
    right_ratio = float(bbox["x_max"]) / image_w
    top_ratio = float(bbox["y_min"]) / image_h
    bottom_ratio = float(bbox["y_max"]) / image_h

    ratio_cfg = dict(region_cfg)
    ratio_cfg["source"] = "ratio"
    ratio_cfg["left_ratio"] = left_ratio
    ratio_cfg["right_ratio"] = right_ratio
    ratio_cfg["top_ratio"] = top_ratio
    ratio_cfg["bottom_ratio"] = bottom_ratio
    return compute_region_bounds(full_h, full_w, ratio_cfg)


def compute_region_bounds_from_region_size_txt(full_h, full_w, region_cfg):
    region_txt = region_cfg.get("region_size_txt")
    if not region_txt:
        raise ValueError("visium_hd_region.region_size_txt must be set when source=region_size_txt")

    stats = load_region_size_txt(region_txt)
    bbox_key = str(region_cfg.get("bbox_key", "bbox_full_minmax"))
    if bbox_key not in stats:
        raise KeyError(f"{bbox_key} not found in {region_txt}")

    bbox = stats[bbox_key]
    image_w = float(region_cfg.get("image_width"))
    image_h = float(region_cfg.get("image_height"))
    if image_w <= 0 or image_h <= 0:
        raise ValueError(
            "visium_hd_region.image_width and visium_hd_region.image_height must be set "
            "when source=region_size_txt"
        )

    left_ratio = float(bbox["x_min"]) / image_w
    right_ratio = float(bbox["x_max"]) / image_w
    top_ratio = float(bbox["y_min"]) / image_h
    bottom_ratio = float(bbox["y_max"]) / image_h

    ratio_cfg = dict(region_cfg)
    ratio_cfg["source"] = "ratio"
    ratio_cfg["left_ratio"] = left_ratio
    ratio_cfg["right_ratio"] = right_ratio
    ratio_cfg["top_ratio"] = top_ratio
    ratio_cfg["bottom_ratio"] = bottom_ratio
    return compute_region_bounds(full_h, full_w, ratio_cfg)


def resolve_model_geometry(cfg, full_grid_shape, region_grid_shape):
    full_h, full_w = full_grid_shape
    region_h, region_w = region_grid_shape

    model_cfg = cfg["model"]
    base_roi_tiles = int(model_cfg["base_roi_tiles"])
    base_stride_tiles = int(model_cfg["base_stride_tiles"])
    min_roi_tiles = int(model_cfg.get("min_roi_tiles", 1))
    min_stride_tiles = int(model_cfg.get("min_stride_tiles", 1))

    scale = 1.0
    if bool(model_cfg.get("scale_roi_to_region", True)):
        scale_h = float(region_h) / float(full_h)
        scale_w = float(region_w) / float(full_w)
        mode = str(model_cfg.get("roi_scale_mode", "min")).lower()
        if mode == "max":
            scale = max(scale_h, scale_w)
        elif mode == "mean":
            scale = 0.5 * (scale_h + scale_w)
        else:
            scale = min(scale_h, scale_w)

    roi_tiles = max(min_roi_tiles, int(round(base_roi_tiles * scale)))
    stride_tiles = max(min_stride_tiles, int(round(base_stride_tiles * scale)))

    roi_tiles = min(roi_tiles, region_h, region_w)
    stride_tiles = min(stride_tiles, roi_tiles)

    if roi_tiles < 1 or stride_tiles < 1:
        raise ValueError(
            f"Invalid derived ROI geometry: roi_tiles={roi_tiles}, stride_tiles={stride_tiles}, "
            f"region_shape={(region_h, region_w)}"
        )

    return {
        "scale_factor": scale,
        "roi_tiles": roi_tiles,
        "stride_tiles": stride_tiles,
        "base_roi_tiles": base_roi_tiles,
        "base_stride_tiles": base_stride_tiles,
        "full_grid_shape": [full_h, full_w],
        "region_grid_shape": [region_h, region_w],
    }


def roi_index_to_global_tile_box(roi_index, nw, stride_tiles, roi_tiles, region_bounds):
    row = int(roi_index) // int(nw)
    col = int(roi_index) % int(nw)

    local_top = row * int(stride_tiles)
    local_left = col * int(stride_tiles)

    global_top = int(region_bounds["top"]) + local_top
    global_left = int(region_bounds["left"]) + local_left

    return {
        "grid_row": row,
        "grid_col": col,
        "local_top_tile": local_top,
        "local_left_tile": local_left,
        "global_top_tile": global_top,
        "global_left_tile": global_left,
        "roi_tiles": int(roi_tiles),
    }


def tile_box_to_image_box(tile_box, tile_um, mpp, tile_px=None):
    tile_px = float(tile_px) if tile_px is not None else float(tile_um) / float(mpp)
    top_px = int(round(tile_box["global_top_tile"] * tile_px))
    left_px = int(round(tile_box["global_left_tile"] * tile_px))
    size_px = int(round(tile_box["roi_tiles"] * tile_px))
    return {
        **tile_box,
        "top_px": top_px,
        "left_px": left_px,
        "size_px": size_px,
    }


def grid_tile_pixels(cfg):
    value = cfg["data"].get("tile_px")
    return float(value) if value is not None else float(cfg["data"]["tile_um"]) / float(cfg["data"]["mpp"])
