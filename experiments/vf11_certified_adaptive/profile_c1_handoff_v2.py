from __future__ import annotations

"""Frozen-Knight C1 V2 handoff profiler.

Runs the preregistered correlated tangent-affine C1 V2 engine on the exact
depth-8 NEAR_ZERO cohort inherited from C0. The primary cohort is unchanged:
C0 d8/m2 PROVEN_ZERO_EXISTS or UNKNOWN (19 cells).

Optional V1 profile input is used only for paired bound-width comparison on the
same anchor/candidate ordering. Direct autograd sampling is diagnostic only and
never mints a certificate.
"""

from collections import Counter
import math
import time
from typing import Callable

import numpy as np
import torch

from directional_regular_c1_v2 import (
    prepare_field,
    prepare_planes,
    certify_directional_regular_c1_v2_prepared,
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


def _center_gradient_direction(field, planes: torch.Tensor, lo: np.ndarray, hi: np.ndarray):
    center = torch.tensor(
        ((np.asarray(lo) + np.asarray(hi)) * 0.5).reshape(1, 1, 3),
        dtype=torch.float64,
        requires_grad=True,
    )
    y = field(planes, center)["sdf"].reshape(())
    grad = torch.autograd.grad(y, center, create_graph=False)[0].reshape(3)
    g = grad.detach().cpu().numpy().astype(np.float64)
    norm = float(np.linalg.norm(g))
    if not math.isfinite(norm) or norm <= 1e-12:
        return None
    return g / norm


def _direct_directional_samples(
    field,
    planes: torch.Tensor,
    lo: np.ndarray,
    hi: np.ndarray,
    direction: np.ndarray,
    *,
    per_axis: int = 5,
) -> np.ndarray:
    axes = [
        np.linspace(float(lo[i]), float(hi[i]), per_axis, dtype=np.float64)
        for i in range(3)
    ]
    pts = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
    q = torch.tensor(pts.reshape(1, -1, 3), dtype=torch.float64, requires_grad=True)
    out = field(planes, q)["sdf"]
    grad = torch.autograd.grad(out.sum(), q, create_graph=False)[0]
    d = torch.tensor(direction, dtype=torch.float64).reshape(1, 1, 3)
    vals = (grad * d).sum(dim=-1).reshape(-1)
    return vals.detach().cpu().numpy()


def _v1_by_anchor(prior_v1_profile: dict | None) -> dict[int, dict]:
    if prior_v1_profile is None:
        return {}
    return {int(r["anchor_id"]): r for r in prior_v1_profile.get("records", [])}


def profile_c1_handoff_v2(
    field,
    planes: torch.Tensor,
    c0_profile: dict,
    *,
    prior_v1_profile: dict | None,
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

    prior = _v1_by_anchor(prior_v1_profile)
    p = prepare_field(field)
    pp = prepare_planes(planes)
    rows = []
    started = time.perf_counter()

    for ordinal, rec0 in enumerate(records0, 1):
        anchor_id = int(rec0["anchor_id"])
        c0_state = rec0["by_micro"]["2"]["state"]
        idx = tuple(int(v) for v in rec0["cell_index_xyz"])
        lo, hi = cell_bounds_from_index(idx, target_depth, domain_lo, domain_hi)

        t0 = time.perf_counter()
        cert = certify_directional_regular_c1_v2_prepared(
            field, p, pp, lo, hi, max_micro_depth=max_micro_depth
        )
        elapsed = time.perf_counter() - t0

        cert_margin = None
        cert_sample_min = None
        cert_sample_max = None
        contradiction = False
        if cert.direction is not None:
            vals = _direct_directional_samples(
                field, pp, lo, hi, np.asarray(cert.direction, dtype=np.float64),
                per_axis=5,
            )
            cert_sample_min = float(vals.min())
            cert_sample_max = float(vals.max())
            if cert.state == "PROVEN_DIRECTIONAL_REGULAR_POSITIVE":
                cert_margin = float(cert.derivative_lower)
                contradiction = cert_sample_min <= 0.0
            elif cert.state == "PROVEN_DIRECTIONAL_REGULAR_NEGATIVE":
                cert_margin = float(-cert.derivative_upper)
                contradiction = cert_sample_max >= 0.0

        # Non-authoritative diagnostic on the proposed center-gradient direction.
        center_d = _center_gradient_direction(field, pp, lo, hi)
        diag_min = diag_max = None
        diag_sign_consistent = None
        if center_d is not None:
            vals = _direct_directional_samples(
                field, pp, lo, hi, center_d, per_axis=5
            )
            diag_min = float(vals.min())
            diag_max = float(vals.max())
            diag_sign_consistent = bool(diag_min > 0.0 or diag_max < 0.0)

        candidate_rows = []
        for x in cert.candidate_results:
            width = float(x.upper_margin - x.lower_margin)
            candidate_rows.append({
                "direction": list(x.direction),
                "state": x.state,
                "lower_margin": float(x.lower_margin),
                "upper_margin": float(x.upper_margin),
                "width": width,
                "evaluated_box_count": int(x.evaluated_box_count),
                "positive_leaf_count": int(x.positive_leaf_count),
                "negative_leaf_count": int(x.negative_leaf_count),
                "unresolved_leaf_count": int(x.unresolved_leaf_count),
                "max_depth_reached": int(x.max_depth_reached),
                "coverage_complete": bool(x.coverage_complete),
            })

        v1_row = prior.get(anchor_id)
        v1_candidate0_width = None
        v2_candidate0_width = None
        ratio = None
        if candidate_rows:
            v2_candidate0_width = float(candidate_rows[0]["width"])
        if v1_row and v1_row.get("candidate_results"):
            v1c0 = v1_row["candidate_results"][0]
            v1_candidate0_width = float(v1c0["upper_margin"] - v1c0["lower_margin"])
        if (
            v1_candidate0_width is not None
            and v2_candidate0_width is not None
            and v1_candidate0_width > 0.0
        ):
            ratio = float(v2_candidate0_width / v1_candidate0_width)

        row = {
            "anchor_id": anchor_id,
            "cell_index_xyz": list(idx),
            "depth": int(target_depth),
            "c0_state_m2": c0_state,
            "primary_handoff": bool(c0_state in PRIMARY_C0_STATES),
            "c1_v2_state": cert.state,
            "c1_v2_direction": None if cert.direction is None else list(cert.direction),
            "c1_v2_derivative_lower": float(cert.derivative_lower),
            "c1_v2_derivative_upper": float(cert.derivative_upper),
            "c1_v2_positive_margin": cert_margin,
            "c1_v2_evaluated_box_count": int(cert.evaluated_box_count),
            "c1_v2_candidate_count": int(len(cert.candidate_results)),
            "elapsed_seconds": float(elapsed),
            "certificate_sampled_derivative_min": cert_sample_min,
            "certificate_sampled_derivative_max": cert_sample_max,
            "empirical_contradiction": bool(contradiction),
            "diagnostic_center_direction": None if center_d is None else list(center_d),
            "diagnostic_center_direction_sample_min": diag_min,
            "diagnostic_center_direction_sample_max": diag_max,
            "diagnostic_center_direction_sign_consistent": diag_sign_consistent,
            "v1_candidate0_width": v1_candidate0_width,
            "v2_candidate0_width": v2_candidate0_width,
            "v2_to_v1_candidate0_width_ratio": ratio,
            "candidate_results": candidate_rows,
        }
        rows.append(row)

        if logger:
            logger(
                f"C1V2_PROGRESS {ordinal}/{len(records0)} anchor={anchor_id} "
                f"c0={c0_state} primary={row['primary_handoff']} "
                f"c1v2={cert.state} candidates={row['c1_v2_candidate_count']} "
                f"boxes={row['c1_v2_evaluated_box_count']} "
                f"ratio_v2_v1={ratio} elapsed={elapsed:.3f}s"
            )

    def summarize(subset: list[dict]) -> dict:
        states = Counter(r["c1_v2_state"] for r in subset)
        regular = sum(v for k, v in states.items() if k != "UNKNOWN")
        n = len(subset)
        margins = np.asarray(
            [r["c1_v2_positive_margin"] for r in subset if r["c1_v2_positive_margin"] is not None],
            dtype=np.float64,
        )
        boxes = np.asarray([r["c1_v2_evaluated_box_count"] for r in subset], dtype=np.float64)
        secs = np.asarray([r["elapsed_seconds"] for r in subset], dtype=np.float64)
        ratios = np.asarray(
            [r["v2_to_v1_candidate0_width_ratio"] for r in subset
             if r["v2_to_v1_candidate0_width_ratio"] is not None],
            dtype=np.float64,
        )
        diagnostic_consistent = sum(
            r["diagnostic_center_direction_sign_consistent"] is True for r in subset
        )
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
            "diagnostic_center_direction_sign_consistent_count": int(diagnostic_consistent),
            "diagnostic_center_direction_sign_consistent_fraction": float(diagnostic_consistent / n) if n else 0.0,
            "v2_to_v1_candidate0_width_ratio_min": None if ratios.size == 0 else float(ratios.min()),
            "v2_to_v1_candidate0_width_ratio_median": None if ratios.size == 0 else float(np.median(ratios)),
            "v2_to_v1_candidate0_width_ratio_p90": None if ratios.size == 0 else float(np.quantile(ratios, 0.90)),
        }

    primary = [r for r in rows if r["primary_handoff"]]
    zero = [r for r in rows if r["c0_state_m2"] == "PROVEN_ZERO_EXISTS"]
    unknown = [r for r in rows if r["c0_state_m2"] == "UNKNOWN"]

    return {
        "schema": "RealSaS.VF11C1V2KnightHandoffProfile.v1",
        "status": "RESEARCH_MEASUREMENT_ONLY__NO_PRODUCT_AUTHORITY",
        "numeric_rigor": "ORDINARY_FLOAT64__NOT_DIRECTED_ROUNDING",
        "target_depth": int(target_depth),
        "max_micro_depth": int(max_micro_depth),
        "elapsed_seconds_total": float(time.perf_counter() - started),
        "stress_all_near_zero": summarize(rows),
        "primary_handoff_zero_or_unknown": summarize(primary),
        "primary_c0_zero_exists": summarize(zero),
        "primary_c0_unknown": summarize(unknown),
        "empirical_contradiction_count": int(sum(r["empirical_contradiction"] for r in rows)),
        "records": rows,
    }
