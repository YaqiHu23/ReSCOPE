from __future__ import annotations

from .model import PartTileROIDictionary

from .geometry import resolve_model_geometry

def build_model(cfg: dict, region_info: dict) -> tuple[PartTileROIDictionary, dict]:
    geom = resolve_model_geometry(
        cfg=cfg,
        full_grid_shape=region_info["full_grid_shape"],
        region_grid_shape=region_info["region_grid_shape"],
    )
    model = PartTileROIDictionary(
        in_ch=cfg["data"]["feature_dim"],
        roi_tiles=geom["roi_tiles"],
        stride_tiles=geom["stride_tiles"],
        tau_select=cfg["model"]["tau_select"],
        tau_match=cfg["model"]["tau_match"],
        recon_percent=cfg["model"].get("recon_percent", 0.25),
        max_samples=cfg["model"].get("max_samples"),
        normalize_select_logits=cfg["model"].get("normalize_select_logits", True),
        select_logit_clip=cfg["model"].get("select_logit_clip", 8.0),
        select_entropy_weight=float(cfg["model"].get("select_entropy_weight", 1e-3)),
        roi_valid_ratio=float(cfg["model"].get("roi_valid_ratio", 0.7)),
        avoid_roi_overlap=cfg["model"].get("avoid_roi_overlap", True),
        selection_mode=cfg["model"].get("selection_mode", "combinatorial_thretopk"),
        selection_noise=cfg["model"].get("selection_noise", "gumbel"),
        sample_in_eval=cfg["model"].get("sample_in_eval", False),
        thretopk_eps=float(cfg["model"].get("thretopk_eps", 1e-6)),
        reconstruction_mode=cfg["model"].get("reconstruction_mode", "roi_prototypes"),
        prototype_grid_size=cfg["model"].get("prototype_grid_size", 8),
        prototype_use_straight_through=cfg["model"].get("prototype_use_straight_through", True),
        target_outside_selected_roi=cfg["model"].get("target_outside_selected_roi", True),
    ).to(cfg["train"]["device"])
    return model, geom
