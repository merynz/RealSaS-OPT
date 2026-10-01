from __future__ import annotations

"""Frozen-Knight C1 handoff profiler.

Consumes the completed C0 paired profile and evaluates directional regularity on
exactly the depth-8 NEAR_ZERO cells. The primary cohort is the C0 handoff set:
PROVEN_ZERO_EXISTS or UNKNOWN at micro=2. All 32 depth-8 NEAR_ZERO cells are
also retained as a stress cohort.

Research-only. Ordinary float64. No product or witness authority.
"""

from collections import Counter
import math
import time
from typing import Callable

import numpy as np
import torch

from directional_regular_c1_v1 import (
    prepare_field,
    prepare_planes,
    certify_directional_regular_c1_prepared,
)
from paired_profile_c0_v3 import cell_bounds_from_index


PRIMARY_C0_STATES = {"PROVEN_ZERO_EXISTS", "UNKNOWN"}


def wilson95(successes: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = successes / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / den
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)


def _direct_directional_samples(
    field,
    planes: torch.Tensor,
    lo: np.ndarray,
    hi: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    axes = [np.linspace(float(lo[i]), float(hi[i]), 3, dtype=np.float64) for i in range(3)]
    pts = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
    q = torch.tensor(pts.reshape(1, -1, 3), dtype=torch.float64, requires_grad=True)
    out = field(planes, q)["sdf"]
    grad = torch.autograd.grad(out.sum(), q, create_graph=False)[0]
    d = torch.tensor(direction, dtype=torch.float64).reshape(1, 1, 3)
    vals = (grad * d).sum(dim=-1).reshape(-1)
    return vals.detach().cpu().numpy()


def profile_c1_handoff(
    field,
    planes: torch.Tensor,
    c0_profile: dict,
    *,
    domain_lo: float,
    domain_hi: float,
    target_depth: int = 8,
    max_micro_depth: int = 2,
    logger: Callable[[str], None] | None = print,
) -> dict:
    records0 = [
        r for r in c0_profile["records"]
        if r["stratum"] == "NEAR_ZERO" and int(r["depth"]) == int(target_depth)
    ]
    if len(records0) != 32:
        raise ValueError(f"EXPECTED_32_NEAR_ZERO_D{target_depth}_RECORDS:{len(records0)}")

    p = prepare_field(field)
    pp = prepare_planes(planes)
    rows = []
    started = time.perf_counter()

    for ordinal, rec0 in enumerate(records0, 1):
        c0_state = rec0["by_micro"]["2"]["state"]
        idx = tuple(int(v) for v in rec0["cell_index_xyz"])
        lo, hi = cell_bounds_from_index(idx, target_depth, domain_lo, domain_hi)

        t0 = time.perf_counter()
        cert = certify_directional_regular_c1_prepared(
            field, p, pp, lo, hi, max_micro_depth=max_micro_depth
        )
        elapsed = time.perf_counter() - t0

        margin = None
        sampled_min = None
        sampled_max = None
        contradiction = False
        if cert.direction is not None:
            vals = _direct_directional_samples(
                field, pp, lo, hi, np.asarray(cert.direction, dtype=np.float64)
            )
            sampled_min = float(vals.min())
            sampled_max = float(vals.max())
            if cert.state == "PROVEN_DIRECTIONAL_REGULAR_POSITIVE":
                margin = float(cert.derivative_lower)
                contradiction = sampled_min <= 0.0
            elif cert.state == "PROVEN_DIRECTIONAL_REGULAR_NEGATIVE":
                margin = float(-cert.derivative_upper)
                contradiction = sampled_max >= 0.0

        row = {
            "anchor_id": int(rec0["anchor_id"]),
            "cell_index_xyz": list(idx),
            "depth": int(target_depth),
            "c0_state_m2": c0_state,
            "primary_handoff": bool(c0_state in PRIMARY_C0_STATES),
            "c1_state": cert.state,
            "c1_direction": None if cert.direction is None else list(cert.direction),
            "c1_derivative_lower": float(cert.derivative_lower),
            "c1_derivative_upper": float(cert.derivative_upper),
            "c1_positive_margin": margin,
            "c1_evaluated_box_count": int(cert.evaluated_box_count),
            "c1_candidate_count": int(len(cert.candidate_results)),
            "elapsed_seconds": float(elapsed),
            "sampled_directional_derivative_min": sampled_min,
            "sampled_directional_derivative_max": sampled_max,
            "empirical_contradiction": bool(contradiction),
            "candidate_results": [
                {
                    "direction": list(x.direction),
                    "state": x.state,
                    "lower_margin": float(x.lower_margin),
                    "upper_margin": float(x.upper_margin),
                    "evaluated_box_count": int(x.evaluated_box_count),
                    "positive_leaf_count": int(x.positive_leaf_count),
                    "negative_leaf_count": int(x.negative_leaf_count),
                    "unresolved_leaf_count": int(x.unresolved_leaf_count),
                    "max_depth_reached": int(x.max_depth_reached),
                    "coverage_complete": bool(x.coverage_complete),
                }
                for x in cert.candidate_results
            ],
        }
        rows.append(row)
        if logger:
            logger(
                f"C1_PROGRESS {ordinal}/{len(records0)} anchor={row['anchor_id']} "
                f"c0={c0_state} primary={row['primary_handoff']} "
                f"c1={cert.state} candidates={row['c1_candidate_count']} "
                f"boxes={row['c1_evaluated_box_count']} elapsed={elapsed:.3f}s"
            )

    def summarize(subset: list[dict]) -> dict:
        states = Counter(r["c1_state"] for r in subset)
        regular = sum(v for k, v in states.items() if k != "UNKNOWN")
        n = len(subset)
        margins = np.asarray(
            [r["c1_positive_margin"] for r in subset if r["c1_positive_margin"] is not None],
            dtype=np.float64,
        )
        boxes = np.asarray([r["c1_evaluated_box_count"] for r in subset], dtype=np.float64)
        secs = np.asarray([r["elapsed_seconds"] for r in subset], dtype=np.float64)
        lo95, hi95 = wilson95(regular, n)
        return {
            "count": int(n),
            "states": dict(sorted(states.items())),
            "regular_count": int(regular),
            "regular_fraction": float(regular / n) if n else 0.0,
            "regular_fraction_wilson95": [float(lo95), float(hi95)],
            "margin_min": None if margins.size == 0 else float(margins.min()),
            "margin_p10": None if margins.size == 0 else float(np.quantile(margins, 0.10)),
            "margin_median": None if margins.size == 0 else float(np.median(margins)),
            "box_eval_mean": float(boxes.mean()) if boxes.size else 0.0,
            "box_eval_p90": float(np.quantile(boxes, 0.90)) if boxes.size else 0.0,
            "seconds_mean": float(secs.mean()) if secs.size else 0.0,
            "empirical_contradiction_count": int(sum(r["empirical_contradiction"] for r in subset)),
        }

    primary = [r for r in rows if r["primary_handoff"]]
    zero = [r for r in rows if r["c0_state_m2"] == "PROVEN_ZERO_EXISTS"]
    unknown = [r for r in rows if r["c0_state_m2"] == "UNKNOWN"]

    return {
        "schema": "RealSaS.VF11C1KnightHandoffProfile.v1",
        "status": "RESEARCH_MEASUREMENT_ONLY__NO_PRODUCT_AUTHORITY",
        "numeric_rigor": "ORDINARY_FLOAT64__NOT_DIRECTED_ROUNDING",
        "target_depth": int(target_depth),
        "max_micro_depth": int(max_micro_depth),
        "c0_source_profile_sha256_expected": "16c2f43535a3c577c62494e464631ca9a79680e480cfeeac92e8a97b852b8ea5",
        "elapsed_seconds_total": float(time.perf_counter() - started),
        "stress_all_near_zero": summarize(rows),
        "primary_handoff_zero_or_unknown": summarize(primary),
        "primary_c0_zero_exists": summarize(zero),
        "primary_c0_unknown": summarize(unknown),
        "empirical_contradiction_count": int(sum(r["empirical_contradiction"] for r in rows)),
        "records": rows,
    }
