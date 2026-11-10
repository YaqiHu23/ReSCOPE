from __future__ import annotations

import argparse

import itertools

import json

import math

import numpy as np

from scipy.optimize import Bounds, LinearConstraint, milp

from .overlap import build_overlap_matrix

from .cnn import zscore

def effective_domain_weights(args: argparse.Namespace, num_rois: int) -> dict[str, float]:
    weights = {
        "cnn": float(args.cnn_weight),
        "cluster": float(args.cluster_weight),
        "adjacency": float(args.adjacency_weight),
        "composition": float(args.composition_weight),
        "coverage": float(args.coverage_weight),
        "density": float(args.density_weight),
        "boundary": float(args.boundary_weight),
        "uni_represent": float(args.uni_represent_weight),
        "reference_budget": float(args.reference_budget_weight),
        "celltype_quota": float(args.celltype_quota_weight),
        "architecture_coverage": float(args.architecture_coverage_weight),
    }
    if not bool(args.adaptive_weights):
        return weights

    if int(num_rois) == 1:
        multipliers = {
            "cnn": 1.05,
            "cluster": 0.90,
            "adjacency": 0.80,
            "composition": 0.85,
            "coverage": 0.80,
            "density": 0.70,
            "boundary": 0.90,
            "uni_represent": 0.90,
            "reference_budget": 0.70,
            "celltype_quota": 0.70,
            "architecture_coverage": 1.00,
        }
    elif int(num_rois) == 2:
        multipliers = {
            "cnn": 0.95,
            "cluster": 0.85,
            "adjacency": 1.00,
            "composition": 1.10,
            "coverage": 1.05,
            "density": 0.70,
            "boundary": 0.90,
            "uni_represent": 1.05,
            "reference_budget": 0.80,
            "celltype_quota": 1.00,
            "architecture_coverage": 1.00,
        }
    else:
        multipliers = {
            "cnn": 0.90,
            "cluster": 0.90,
            "adjacency": 1.10,
            "composition": 1.20,
            "coverage": 1.15,
            "density": 0.65,
            "boundary": 0.95,
            "uni_represent": 1.15,
            "reference_budget": 0.75,
            "celltype_quota": 1.10,
            "architecture_coverage": 1.00,
        }
    return {name: weights[name] * multipliers[name] for name in weights}

def adjusted_priority(priority: np.ndarray, feat: dict, args: argparse.Namespace) -> np.ndarray:
    ref_counts = feat["cell_counts"].sum(axis=1).astype(np.float64)
    ref_size_z = zscore(np.log1p(ref_counts))
    out = priority.astype(np.float64).copy()
    out -= float(args.reference_size_penalty_weight) * ref_size_z
    out[~feat["available_mask"]] = -np.inf
    return out

def build_celltype_upper_constraints(
    *,
    pool: np.ndarray,
    feat: dict,
    ref_counts: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[LinearConstraint], list[dict]]:
    if not bool(args.enforce_celltype_upper_bounds):
        return [], []

    full_counts = feat["full_cell_counts"].astype(np.float64)
    total = float(full_counts.sum())
    if total <= 0:
        return [], []
    full_dist = full_counts / total
    constraints: list[LinearConstraint] = []
    reports: list[dict] = []
    roi_cell_counts = feat["cell_counts"][pool].astype(np.float64)
    for label_idx, frac in enumerate(full_dist):
        if float(frac) < float(args.min_celltype_fraction_for_upper):
            continue
        upper = min(0.98, float(frac) + float(args.celltype_upper_tolerance))
        coeff = roi_cell_counts[:, label_idx] - upper * ref_counts
        constraints.append(LinearConstraint(coeff.reshape(1, -1), lb=[-np.inf], ub=[0.0]))
        reports.append(
            {
                "celltype_label": int(feat["celltype_labels"][label_idx]),
                "full_fraction": float(frac),
                "upper_fraction": float(upper),
            }
        )
    return constraints, reports

def build_constraints(
    *,
    pool: np.ndarray,
    feat: dict,
    num_rois: int,
    args: argparse.Namespace,
    max_reference_fraction: float | None,
    use_celltype_upper: bool,
) -> tuple[list[LinearConstraint], dict]:
    constraints: list[LinearConstraint] = [
        LinearConstraint(np.ones((1, int(pool.size))), lb=[float(num_rois)], ub=[float(num_rois)])
    ]

    radius = max(0, int(math.ceil(float(feat["roi_tiles"]) / float(feat["stride_tiles"]))) - 1)
    overlap = build_overlap_matrix(pool, feat["rows"], feat["cols"], radius)
    overlap_constraints = 0
    if overlap is not None:
        overlap_constraints = int(overlap.shape[0])
        constraints.append(
            LinearConstraint(
                overlap,
                lb=np.full(overlap.shape[0], -np.inf, dtype=np.float64),
                ub=np.ones(overlap.shape[0], dtype=np.float64),
            )
        )

    ref_counts = feat["cell_counts"][pool].sum(axis=1).astype(np.float64)
    total_reference_cells = float(feat["full_cell_counts"].sum())
    min_ref = float(args.min_reference_fraction) * total_reference_cells
    max_ref = np.inf
    if max_reference_fraction is not None and float(max_reference_fraction) > 0:
        max_ref = float(max_reference_fraction) * total_reference_cells
    if np.isfinite(max_ref) and min_ref > max_ref:
        min_ref = 0.0
    constraints.append(LinearConstraint(ref_counts.reshape(1, -1), lb=[min_ref], ub=[max_ref]))

    celltype_report: list[dict] = []
    if use_celltype_upper:
        old_flag = bool(args.enforce_celltype_upper_bounds)
        args.enforce_celltype_upper_bounds = True
        celltype_constraints, celltype_report = build_celltype_upper_constraints(
            pool=pool,
            feat=feat,
            ref_counts=ref_counts,
            args=args,
        )
        args.enforce_celltype_upper_bounds = old_flag
        constraints.extend(celltype_constraints)

    return constraints, {
        "overlap_constraints": overlap_constraints,
        "reference_min_cells": float(min_ref),
        "reference_max_cells": None if not np.isfinite(max_ref) else float(max_ref),
        "reference_min_fraction": float(args.min_reference_fraction),
        "reference_max_fraction": None if max_reference_fraction is None else float(max_reference_fraction),
        "celltype_upper_constraints": celltype_report,
    }

def reference_fraction_attempts(args: argparse.Namespace) -> list[tuple[float | None, bool]]:
    attempts: list[tuple[float | None, bool]] = []
    base = float(args.max_reference_fraction)
    if base > 0:
        for multiplier in args.reference_fraction_retry_multipliers:
            if float(multiplier) <= 0:
                attempts.append((None, False))
            else:
                attempts.append((min(0.95, base * float(multiplier)), False))
        if bool(args.enforce_celltype_upper_bounds):
            attempts.insert(0, (base, True))
    else:
        attempts.append((None, bool(args.enforce_celltype_upper_bounds)))
        attempts.append((None, False))

    deduped: list[tuple[float | None, bool]] = []
    seen: set[tuple[str, bool]] = set()
    for max_fraction, use_celltype_upper in attempts:
        key = ("none" if max_fraction is None else f"{max_fraction:.6f}", bool(use_celltype_upper))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((max_fraction, bool(use_celltype_upper)))
    return deduped

def exact_constraint_report(
    *,
    feat: dict,
    args: argparse.Namespace,
    max_reference_fraction: float | None,
    use_celltype_upper: bool,
) -> dict:
    total_reference_cells = float(feat["full_cell_counts"].sum())
    min_ref = float(args.min_reference_fraction) * total_reference_cells
    max_ref = np.inf
    if max_reference_fraction is not None and float(max_reference_fraction) > 0:
        max_ref = float(max_reference_fraction) * total_reference_cells
    if np.isfinite(max_ref) and min_ref > max_ref:
        min_ref = 0.0

    celltype_report: list[dict] = []
    if use_celltype_upper:
        full_counts = feat["full_cell_counts"].astype(np.float64)
        full_dist = full_counts / max(float(full_counts.sum()), 1.0)
        for label_idx, frac in enumerate(full_dist):
            if float(frac) < float(args.min_celltype_fraction_for_upper):
                continue
            upper = min(0.98, float(frac) + float(args.celltype_upper_tolerance))
            celltype_report.append(
                {
                    "celltype_label": int(feat["celltype_labels"][label_idx]),
                    "full_fraction": float(frac),
                    "upper_fraction": float(upper),
                }
            )

    return {
        "reference_min_cells": float(min_ref),
        "reference_max_cells": None if not np.isfinite(max_ref) else float(max_ref),
        "reference_min_fraction": float(args.min_reference_fraction),
        "reference_max_fraction": None if max_reference_fraction is None else float(max_reference_fraction),
        "celltype_upper_constraints": celltype_report,
    }

def passes_celltype_upper(cells: np.ndarray, feat: dict, args: argparse.Namespace) -> bool:
    full_counts = feat["full_cell_counts"].astype(np.float64)
    full_total = float(full_counts.sum())
    selected_total = float(cells.sum())
    if full_total <= 0 or selected_total <= 0:
        return True
    full_dist = full_counts / full_total
    selected_dist = cells.astype(np.float64) / selected_total
    for idx, full_fraction in enumerate(full_dist):
        if float(full_fraction) < float(args.min_celltype_fraction_for_upper):
            continue
        upper = min(0.98, float(full_fraction) + float(args.celltype_upper_tolerance))
        if float(selected_dist[idx]) > upper + 1e-9:
            return False
    return True

def exact_combo_has_overlap(combo: tuple[int, ...], rows: np.ndarray, cols: np.ndarray, radius: int) -> bool:
    for a, b in itertools.combinations(combo, 2):
        if abs(int(rows[a]) - int(rows[b])) <= radius and abs(int(cols[a]) - int(cols[b])) <= radius:
            return True
    return False

def celltype_upper_mask(cells: np.ndarray, feat: dict, args: argparse.Namespace) -> np.ndarray:
    cells = np.asarray(cells, dtype=np.float64)
    if cells.ndim == 1:
        cells = cells.reshape(1, -1)
    full_counts = feat["full_cell_counts"].astype(np.float64)
    full_total = float(full_counts.sum())
    selected_total = cells.sum(axis=1)
    mask = np.ones(cells.shape[0], dtype=bool)
    valid_total = selected_total > 0
    if full_total <= 0:
        return mask
    full_dist = full_counts / full_total
    for idx, full_fraction in enumerate(full_dist):
        if float(full_fraction) < float(args.min_celltype_fraction_for_upper):
            continue
        upper = min(0.98, float(full_fraction) + float(args.celltype_upper_tolerance))
        mask &= (~valid_total) | (cells[:, idx] <= upper * selected_total + 1e-9)
    return mask

def solve_exact_vectorized(
    *,
    priority: np.ndarray,
    feat: dict,
    num_rois: int,
    args: argparse.Namespace,
) -> tuple[list[int], dict]:
    available_idx = np.flatnonzero(feat["available_mask"])
    objective_priority = adjusted_priority(priority, feat, args)
    architecture_weight = float(getattr(args, "effective_architecture_coverage_weight", 0.0))
    use_architecture = architecture_weight > 0.0
    if use_architecture:
        required = {
            "architecture_signatures",
            "architecture_state_weights",
            "architecture_candidate_z",
            "architecture_candidate_mean",
            "architecture_candidate_std",
        }
        missing = sorted(required.difference(feat))
        if missing:
            raise RuntimeError(f"Missing architecture coverage features: {missing}")
        ranking_priority = objective_priority + architecture_weight * feat["architecture_candidate_z"]
    else:
        ranking_priority = objective_priority
    order = available_idx[np.argsort(-ranking_priority[available_idx])]
    pool = order[: min(int(args.candidate_pool_size), int(order.size))]
    if pool.size < int(num_rois):
        raise RuntimeError(f"Candidate pool has only {pool.size} candidates for K={num_rois}")

    n = int(pool.size)
    pool_scores = objective_priority[pool].astype(np.float64)
    pool_ref_counts = feat["cell_counts"][pool].sum(axis=1).astype(np.float64)
    pool_cell_counts = feat["cell_counts"][pool].astype(np.float64)
    rows = feat["rows"][pool]
    cols = feat["cols"][pool]
    radius = max(0, int(math.ceil(float(feat["roi_tiles"]) / float(feat["stride_tiles"]))) - 1)
    full_reference_cells = float(feat["full_cell_counts"].sum())
    architecture_signatures: np.ndarray | None = None
    architecture_state_weights: np.ndarray | None = None
    architecture_mean = 0.0
    architecture_std = 1.0
    single_architecture_score: np.ndarray | None = None
    if use_architecture:
        architecture_signatures = feat["architecture_signatures"][pool].astype(np.float64)
        architecture_state_weights = feat["architecture_state_weights"].astype(np.float64)
        architecture_mean = float(feat["architecture_candidate_mean"])
        architecture_std = max(float(feat["architecture_candidate_std"]), 1e-8)
        single_raw = architecture_signatures @ architecture_state_weights
        single_architecture_score = pool_scores + architecture_weight * (
            (single_raw - architecture_mean) / architecture_std
        )

    pair_a: np.ndarray | None = None
    pair_b: np.ndarray | None = None
    pair_nonoverlap: np.ndarray | None = None
    pair_ref: np.ndarray | None = None
    pair_score: np.ndarray | None = None
    pair_cells: np.ndarray | None = None
    pair_architecture: np.ndarray | None = None
    if int(num_rois) >= 2:
        pair_a, pair_b = np.triu_indices(n, k=1)
        pair_nonoverlap = (
            (np.abs(rows[pair_a] - rows[pair_b]) > radius)
            | (np.abs(cols[pair_a] - cols[pair_b]) > radius)
        )
        pair_ref = pool_ref_counts[pair_a] + pool_ref_counts[pair_b]
        pair_score = pool_scores[pair_a] + pool_scores[pair_b]
        if use_architecture:
            assert architecture_signatures is not None and architecture_state_weights is not None
            pair_architecture = np.maximum(
                architecture_signatures[pair_a], architecture_signatures[pair_b]
            )
            pair_raw = pair_architecture @ architecture_state_weights
            pair_score = pair_score + 2.0 * architecture_weight * (
                (pair_raw - architecture_mean) / architecture_std
            )
        pair_cells = pool_cell_counts[pair_a] + pool_cell_counts[pair_b]

    attempt_reports: list[dict] = []
    best_local: tuple[int, ...] | None = None
    best_score = -np.inf
    best_constraints_report: dict | None = None

    for max_fraction, use_celltype_upper in reference_fraction_attempts(args):
        constraints_report = exact_constraint_report(
            feat=feat,
            args=args,
            max_reference_fraction=max_fraction,
            use_celltype_upper=use_celltype_upper,
        )
        min_ref = float(constraints_report["reference_min_cells"])
        max_ref_value = constraints_report["reference_max_cells"]
        max_ref = np.inf if max_ref_value is None else float(max_ref_value)
        searched = math.comb(n, int(num_rois))
        feasible = 0
        local_best: tuple[int, ...] | None = None
        local_best_score = -np.inf

        if int(num_rois) == 1:
            mask = (pool_ref_counts >= min_ref) & (pool_ref_counts <= max_ref)
            if use_celltype_upper:
                mask &= celltype_upper_mask(pool_cell_counts, feat, args)
            feasible = int(mask.sum())
            if feasible > 0:
                scores = single_architecture_score if use_architecture else pool_scores
                assert scores is not None
                local_idx = int(np.flatnonzero(mask)[np.argmax(scores[mask])])
                local_best = (local_idx,)
                local_best_score = float(scores[local_idx])
        elif int(num_rois) == 2:
            assert pair_a is not None and pair_b is not None and pair_nonoverlap is not None
            assert pair_ref is not None and pair_score is not None and pair_cells is not None
            mask = pair_nonoverlap & (pair_ref >= min_ref) & (pair_ref <= max_ref)
            if use_celltype_upper and mask.any():
                valid = np.flatnonzero(mask)
                upper_valid = celltype_upper_mask(pair_cells[valid], feat, args)
                tmp = np.zeros_like(mask)
                tmp[valid[upper_valid]] = True
                mask = tmp
            feasible = int(mask.sum())
            if feasible > 0:
                valid = np.flatnonzero(mask)
                best_pos = int(valid[np.argmax(pair_score[valid])])
                local_best = (int(pair_a[best_pos]), int(pair_b[best_pos]))
                local_best_score = float(pair_score[best_pos])
        elif int(num_rois) == 3:
            assert pair_a is not None and pair_b is not None and pair_nonoverlap is not None
            assert pair_ref is not None and pair_score is not None and pair_cells is not None
            for third in range(n):
                nonoverlap_third = (
                    ((np.abs(rows[pair_a] - rows[third]) > radius) | (np.abs(cols[pair_a] - cols[third]) > radius))
                    & ((np.abs(rows[pair_b] - rows[third]) > radius) | (np.abs(cols[pair_b] - cols[third]) > radius))
                )
                ref_total = pair_ref + pool_ref_counts[third]
                mask = (
                    (pair_b < third)
                    & pair_nonoverlap
                    & nonoverlap_third
                    & (ref_total >= min_ref)
                    & (ref_total <= max_ref)
                )
                if use_celltype_upper and mask.any():
                    valid = np.flatnonzero(mask)
                    cells = pair_cells[valid] + pool_cell_counts[third]
                    upper_valid = celltype_upper_mask(cells, feat, args)
                    tmp = np.zeros_like(mask)
                    tmp[valid[upper_valid]] = True
                    mask = tmp
                current_feasible = int(mask.sum())
                feasible += current_feasible
                if current_feasible <= 0:
                    continue
                valid = np.flatnonzero(mask)
                scores = pair_score[valid] + pool_scores[third]
                if use_architecture:
                    assert pair_architecture is not None
                    assert architecture_signatures is not None and architecture_state_weights is not None
                    triple_raw = (
                        np.maximum(pair_architecture[valid], architecture_signatures[third])
                        @ architecture_state_weights
                    )
                    pair_raw_valid = pair_architecture[valid] @ architecture_state_weights
                    # pair_score already contains the 2-ROI coverage term; replace
                    # it with the single 3-ROI facility-location term.
                    scores = (
                        scores
                        - 2.0 * architecture_weight * (
                            (pair_raw_valid - architecture_mean) / architecture_std
                        )
                        + 3.0 * architecture_weight * (
                            (triple_raw - architecture_mean) / architecture_std
                        )
                    )
                best_in_valid = int(valid[np.argmax(scores)])
                score = float(scores[np.argmax(scores)])
                if score > local_best_score:
                    local_best_score = score
                    local_best = (int(pair_a[best_in_valid]), int(pair_b[best_in_valid]), int(third))
        else:
            raise RuntimeError("Vectorized exact solver supports K=1, K=2, or K=3.")

        attempt_report = {
            "max_reference_fraction": None if max_fraction is None else float(max_fraction),
            "celltype_upper_bounds": bool(use_celltype_upper),
            "solver_status": 0 if local_best is not None else 2,
            "solver_message": "Vectorized exact constrained enumeration found a feasible optimum."
            if local_best is not None
            else "No feasible combination in candidate pool.",
            "solver_success": local_best is not None,
            "has_solution": local_best is not None,
            "searched_combinations": int(searched),
            "feasible_combinations": int(feasible),
        }
        attempt_reports.append(attempt_report)
        if local_best is not None:
            best_local = local_best
            best_score = local_best_score
            best_constraints_report = constraints_report
            break

    if best_local is None or best_constraints_report is None:
        raise RuntimeError(f"Vectorized exact domain-generalized search failed without solution: {json.dumps(attempt_reports, indent=2)}")

    selected = [int(pool[idx]) for idx in best_local]
    selected_ref = float(feat["cell_counts"][selected].sum())
    architecture_report = None
    if use_architecture:
        assert architecture_state_weights is not None
        selected_signature = np.max(feat["architecture_signatures"][selected], axis=0)
        selected_raw = float(selected_signature @ architecture_state_weights)
        individual_raw = (
            feat["architecture_signatures"][selected].astype(np.float64)
            @ architecture_state_weights
        )
        architecture_report = {
            "weight": architecture_weight,
            "selected_set_coverage": selected_raw,
            "selected_set_coverage_z": float((selected_raw - architecture_mean) / architecture_std),
            "selected_individual_coverage": individual_raw.tolist(),
            "complementarity_gain_over_best_single": float(selected_raw - individual_raw.max()),
            "n_states": int(architecture_state_weights.size),
            "state_weights": architecture_state_weights.tolist(),
            "candidate_coverage_mean": architecture_mean,
            "candidate_coverage_std": architecture_std,
        }
    return selected, {
        "candidate_pool_size": int(pool.size),
        "solver_backend": "vectorized_exact_constrained_enumeration",
        "domain_generalization": {
            "reference_fraction_attempts": attempt_reports,
            "used_constraints": best_constraints_report,
            "reference_size_penalty_weight": float(args.reference_size_penalty_weight),
            "selected_reference_cell_count": int(selected_ref),
            "selected_reference_fraction": float(selected_ref / max(full_reference_cells, 1.0)),
        },
        "solver_status": 0,
        "solver_message": "Vectorized exact constrained enumeration found a feasible optimum.",
        "solver_success": True,
        "solver_fun": float(-best_score),
        "mip_gap": 0.0,
        "selected_binary_values": [1.0 for _ in selected],
        "architecture_coverage": architecture_report,
    }

def solve_exact(
    *,
    priority: np.ndarray,
    feat: dict,
    num_rois: int,
    args: argparse.Namespace,
) -> tuple[list[int], dict]:
    if float(getattr(args, "effective_architecture_coverage_weight", 0.0)) > 0.0 and int(num_rois) > 3:
        raise RuntimeError("Architecture-coverage exact search currently supports K=1, K=2, or K=3.")
    if int(num_rois) in {1, 2, 3}:
        return solve_exact_vectorized(priority=priority, feat=feat, num_rois=num_rois, args=args)

    available_idx = np.flatnonzero(feat["available_mask"])
    objective_priority = adjusted_priority(priority, feat, args)
    order = available_idx[np.argsort(-objective_priority[available_idx])]
    pool = order[: min(int(args.candidate_pool_size), int(order.size))]
    if pool.size < int(num_rois):
        raise RuntimeError(f"Candidate pool has only {pool.size} candidates for K={num_rois}")

    pool_scores = objective_priority[pool].astype(np.float64)
    pool_ref_counts = feat["cell_counts"][pool].sum(axis=1).astype(np.float64)
    pool_cell_counts = feat["cell_counts"][pool].astype(np.float64)
    rows = feat["rows"][pool]
    cols = feat["cols"][pool]
    radius = max(0, int(math.ceil(float(feat["roi_tiles"]) / float(feat["stride_tiles"]))) - 1)
    full_reference_cells = float(feat["full_cell_counts"].sum())

    attempt_reports: list[dict] = []
    best_combo: tuple[int, ...] | None = None
    best_score = -np.inf
    best_constraints_report: dict | None = None

    for max_fraction, use_celltype_upper in reference_fraction_attempts(args):
        constraints_report = exact_constraint_report(
            feat=feat,
            args=args,
            max_reference_fraction=max_fraction,
            use_celltype_upper=use_celltype_upper,
        )
        min_ref = float(constraints_report["reference_min_cells"])
        max_ref_value = constraints_report["reference_max_cells"]
        max_ref = np.inf if max_ref_value is None else float(max_ref_value)
        searched = 0
        feasible = 0
        local_best_combo: tuple[int, ...] | None = None
        local_best_score = -np.inf

        for combo in itertools.combinations(range(int(pool.size)), int(num_rois)):
            searched += 1
            if exact_combo_has_overlap(combo, rows, cols, radius):
                continue
            ref_total = float(pool_ref_counts[list(combo)].sum())
            if ref_total < min_ref or ref_total > max_ref:
                continue
            cells = pool_cell_counts[list(combo)].sum(axis=0)
            if use_celltype_upper and not passes_celltype_upper(cells, feat, args):
                continue
            feasible += 1
            score = float(pool_scores[list(combo)].sum())
            if score > local_best_score:
                local_best_score = score
                local_best_combo = combo

        attempt_report = {
            "max_reference_fraction": None if max_fraction is None else float(max_fraction),
            "celltype_upper_bounds": bool(use_celltype_upper),
            "solver_status": 0 if local_best_combo is not None else 2,
            "solver_message": "Exact constrained enumeration found a feasible optimum."
            if local_best_combo is not None
            else "No feasible combination in candidate pool.",
            "solver_success": local_best_combo is not None,
            "has_solution": local_best_combo is not None,
            "searched_combinations": int(searched),
            "feasible_combinations": int(feasible),
        }
        attempt_reports.append(attempt_report)
        if local_best_combo is not None:
            best_combo = local_best_combo
            best_score = local_best_score
            best_constraints_report = constraints_report
            break

    if best_combo is None or best_constraints_report is None:
        raise RuntimeError(f"Exact domain-generalized search failed without solution: {json.dumps(attempt_reports, indent=2)}")

    selected = [int(pool[idx]) for idx in best_combo]
    selected_ref = float(feat["cell_counts"][selected].sum())
    return selected, {
        "candidate_pool_size": int(pool.size),
        "solver_backend": "exact_constrained_enumeration",
        "domain_generalization": {
            "reference_fraction_attempts": attempt_reports,
            "used_constraints": best_constraints_report,
            "reference_size_penalty_weight": float(args.reference_size_penalty_weight),
            "selected_reference_cell_count": int(selected_ref),
            "selected_reference_fraction": float(selected_ref / max(full_reference_cells, 1.0)),
        },
        "solver_status": 0,
        "solver_message": "Exact constrained enumeration found a feasible optimum.",
        "solver_success": True,
        "solver_fun": float(-best_score),
        "mip_gap": 0.0,
        "selected_binary_values": [1.0 for _ in selected],
    }

def solve_mip(
    *,
    priority: np.ndarray,
    feat: dict,
    num_rois: int,
    args: argparse.Namespace,
) -> tuple[list[int], dict]:
    if str(args.solver) == "exact":
        return solve_exact(priority=priority, feat=feat, num_rois=num_rois, args=args)
    if float(getattr(args, "effective_architecture_coverage_weight", 0.0)) > 0.0:
        raise RuntimeError(
            "Set-level architecture coverage uses the exact binary set solver; pass --solver exact."
        )

    available_idx = np.flatnonzero(feat["available_mask"])
    objective_priority = adjusted_priority(priority, feat, args)
    order = available_idx[np.argsort(-objective_priority[available_idx])]
    pool = order[: min(int(args.candidate_pool_size), int(order.size))]
    if pool.size < int(num_rois):
        raise RuntimeError(f"Candidate pool has only {pool.size} candidates for K={num_rois}")

    c = -objective_priority[pool].astype(np.float64)
    c = np.nan_to_num(c, nan=1e6, posinf=1e6, neginf=-1e6)
    attempt_reports: list[dict] = []
    best_result = None
    best_constraints_report: dict | None = None
    best_values = None

    for max_fraction, use_celltype_upper in reference_fraction_attempts(args):
        constraints, constraints_report = build_constraints(
            pool=pool,
            feat=feat,
            num_rois=num_rois,
            args=args,
            max_reference_fraction=max_fraction,
            use_celltype_upper=use_celltype_upper,
        )
        result = milp(
            c=c,
            integrality=np.ones(int(pool.size), dtype=np.int8),
            bounds=Bounds(np.zeros(int(pool.size)), np.ones(int(pool.size))),
            constraints=constraints,
            options={"time_limit": float(args.time_limit), "mip_rel_gap": float(args.mip_gap)},
        )
        attempt_report = {
            "max_reference_fraction": None if max_fraction is None else float(max_fraction),
            "celltype_upper_bounds": bool(use_celltype_upper),
            "solver_status": int(result.status),
            "solver_message": str(result.message),
            "solver_success": bool(result.success),
            "has_solution": result.x is not None,
        }
        attempt_reports.append(attempt_report)
        if result.x is not None:
            best_result = result
            best_constraints_report = constraints_report
            best_values = np.asarray(result.x, dtype=np.float64)
            break

    if best_values is None or best_result is None or best_constraints_report is None:
        raise RuntimeError(f"Domain-generalized MILP failed without solution: {json.dumps(attempt_reports, indent=2)}")

    chosen_local = np.flatnonzero(best_values >= 0.5)
    if chosen_local.size != int(num_rois):
        raise RuntimeError(f"MILP returned {chosen_local.size} integral choices for requested {num_rois}")
    if not np.allclose(best_values, np.round(best_values), atol=1e-5):
        raise RuntimeError("MILP returned a fractional solution; increase time_limit")
    integer_values = np.round(best_values)
    for constraint in constraints:
        actual = np.asarray(constraint.A @ integer_values).ravel()
        if np.any(actual < constraint.lb - 1e-5) or np.any(actual > constraint.ub + 1e-5):
            raise RuntimeError("MILP integer solution violates the active constraints")
    selected = [int(pool[idx]) for idx in chosen_local]

    selected_ref = float(feat["cell_counts"][selected].sum())
    full_ref = float(feat["full_cell_counts"].sum())
    return selected, {
        "candidate_pool_size": int(pool.size),
        "domain_generalization": {
            "reference_fraction_attempts": attempt_reports,
            "used_constraints": best_constraints_report,
            "reference_size_penalty_weight": float(args.reference_size_penalty_weight),
            "selected_reference_cell_count": int(selected_ref),
            "selected_reference_fraction": float(selected_ref / max(full_ref, 1.0)),
        },
        "solver_status": int(best_result.status),
        "solver_message": str(best_result.message),
        "solver_success": bool(best_result.success),
        "solver_fun": float(best_result.fun) if best_result.fun is not None else None,
        "mip_gap": float(getattr(best_result, "mip_gap", np.nan)) if hasattr(best_result, "mip_gap") else None,
        "selected_binary_values": best_values[chosen_local].tolist(),
    }
