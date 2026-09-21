from __future__ import annotations

"""Deterministic profiler for VF-11 C0 range engine v2."""

from collections import Counter
import time
from typing import Callable, Iterable

import numpy as np
import torch

from range_engine_c0_v2 import certify_cell_c0


def deterministic_cell_indices(
    depth: int,
    *,
    max_cells: int | None,
    seed: int,
):
    if depth < 0:
        raise ValueError("DEPTH_NEGATIVE")
    n = 1 << int(depth)
    total = n ** 3
    take = total if max_cells is None else min(total, int(max_cells))
    if take <= 0:
        raise ValueError("MAX_CELLS_NONPOSITIVE")
    if take == total:
        for code in range(total):
            yield code % n, (code // n) % n, code // (n * n)
        return

    step = 2654435761 % total
    if step % 2 == 0:
        step = (step + 1) % total
    start = int(seed) % total
    for j in range(take):
        code = (start + j * step) % total
        yield code % n, (code // n) % n, code // (n * n)


def cell_bounds(
    index_xyz,
    depth: int,
    domain_lo: float,
    domain_hi: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = 1 << int(depth)
    width = (float(domain_hi) - float(domain_lo)) / float(n)
    idx = np.asarray(index_xyz, dtype=np.float64)
    lo = float(domain_lo) + idx * width
    return lo, lo + width


def _quantiles(values: list[int | float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    a = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(a)),
        "p50": float(np.quantile(a, 0.50)),
        "p90": float(np.quantile(a, 0.90)),
        "p99": float(np.quantile(a, 0.99)),
        "max": float(np.max(a)),
    }


def profile_depth(
    field,
    planes: torch.Tensor,
    *,
    depth: int,
    max_cells: int | None,
    domain_lo: float,
    domain_hi: float,
    max_micro_depth: int,
    seed: int,
    log_every: int = 0,
    logger: Callable[[str], None] | None = None,
) -> dict:
    indices = list(
        deterministic_cell_indices(
            depth,
            max_cells=max_cells,
            seed=seed,
        )
    )
    states = Counter()
    proof_depths = Counter()
    eval_counts: list[int] = []
    regime_counts: list[int] = []
    unresolved_counts: list[int] = []
    positive_leaf_counts: list[int] = []
    negative_leaf_counts: list[int] = []

    start = time.perf_counter()
    for i, idx in enumerate(indices, 1):
        lo, hi = cell_bounds(idx, depth, domain_lo, domain_hi)
        cert = certify_cell_c0(
            field,
            planes,
            lo,
            hi,
            max_micro_depth=max_micro_depth,
        )
        states[cert.state] += 1
        proof_depths[f"{cert.state}@{cert.max_depth_reached}"] += 1
        eval_counts.append(cert.evaluated_box_count)
        regime_counts.append(cert.regime_box_count)
        unresolved_counts.append(cert.unresolved_leaf_count)
        positive_leaf_counts.append(cert.positive_leaf_count)
        negative_leaf_counts.append(cert.negative_leaf_count)

        if log_every and logger and i % int(log_every) == 0:
            logger(
                f"depth={depth} processed={i}/{len(indices)} "
                f"states={dict(states)} "
                f"mean_box_evals={np.mean(eval_counts):.2f}"
            )

    elapsed = time.perf_counter() - start
    visited = len(indices)
    total = (1 << depth) ** 3
    empty = states["PROVEN_EMPTY_POSITIVE"] + states["PROVEN_EMPTY_NEGATIVE"]

    return {
        "depth": int(depth),
        "grid_axis_cells": int(1 << depth),
        "full_grid_cell_count": int(total),
        "sampled_cell_count": int(visited),
        "sample_is_exhaustive": bool(visited == total),
        "domain_lo": float(domain_lo),
        "domain_hi": float(domain_hi),
        "max_micro_depth": int(max_micro_depth),
        "states": {
            "PROVEN_EMPTY": int(empty),
            "PROVEN_EMPTY_POSITIVE": int(states["PROVEN_EMPTY_POSITIVE"]),
            "PROVEN_EMPTY_NEGATIVE": int(states["PROVEN_EMPTY_NEGATIVE"]),
            "PROVEN_ZERO_EXISTS": int(states["PROVEN_ZERO_EXISTS"]),
            "UNKNOWN": int(states["UNKNOWN"]),
        },
        "fractions": {
            "PROVEN_EMPTY": float(empty / max(visited, 1)),
            "PROVEN_ZERO_EXISTS": float(states["PROVEN_ZERO_EXISTS"] / max(visited, 1)),
            "UNKNOWN": float(states["UNKNOWN"] / max(visited, 1)),
        },
        "proof_depth_histogram": dict(sorted(proof_depths.items())),
        "evaluated_box_count": _quantiles(eval_counts),
        "regime_box_count": _quantiles(regime_counts),
        "unresolved_leaf_count": _quantiles(unresolved_counts),
        "positive_leaf_count": _quantiles(positive_leaf_counts),
        "negative_leaf_count": _quantiles(negative_leaf_counts),
        "elapsed_seconds": float(elapsed),
        "parent_cells_per_second": float(visited / max(elapsed, 1e-12)),
        "range_box_evaluations_per_second": float(sum(eval_counts) / max(elapsed, 1e-12)),
    }


def profile_depths(
    field,
    planes: torch.Tensor,
    *,
    depths: Iterable[int],
    max_cells_per_depth: int | None,
    domain_lo: float,
    domain_hi: float,
    max_micro_depth: int,
    seed: int,
    log_every: int = 0,
    logger: Callable[[str], None] | None = print,
) -> dict:
    rows = []
    for depth in depths:
        if logger:
            logger(f"VF11_C0_PROFILE_START depth={int(depth)}")
        row = profile_depth(
            field,
            planes,
            depth=int(depth),
            max_cells=max_cells_per_depth,
            domain_lo=domain_lo,
            domain_hi=domain_hi,
            max_micro_depth=max_micro_depth,
            seed=seed + int(depth),
            log_every=log_every,
            logger=logger,
        )
        rows.append(row)
        if logger:
            logger(
                f"VF11_C0_PROFILE_DONE depth={row['depth']} "
                f"states={row['states']} "
                f"elapsed={row['elapsed_seconds']:.3f}s"
            )
    return {
        "schema": "RealSaS.VF11C0CertifiabilityProfile.v2",
        "status": "RESEARCH_MEASUREMENT_ONLY__NO_PRODUCT_AUTHORITY",
        "numerically_rigorous": False,
        "depths": rows,
    }
