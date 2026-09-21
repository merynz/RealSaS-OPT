from __future__ import annotations

"""Paired multi-depth profiler for VF-11 C0 V3.

One spatial anchor is followed through depths 5..8. One max-depth refinement
tree yields micro-depth 0/1/2 states without recomputing the same parent three
times.

Sampling strata:
  * VOLUME: deterministic quasi-uniform anchor cells.
  * NEAR_ZERO: deterministic candidate anchors ranked by direct |f| only for
    diagnostic sampling. This heuristic NEVER certifies absence.
"""

from collections import Counter, defaultdict
import math
import time
from typing import Callable, Iterable

import numpy as np
import torch

from range_engine_c0_v3 import (
    PreparedField,
    prepare_field,
    prepare_planes,
    certify_cell_c0_ladder_prepared,
)


def _cell_code_sequence(depth: int, count: int, seed: int):
    n = 1 << depth
    total = n ** 3
    if count > total:
        raise ValueError("COUNT_EXCEEDS_GRID")
    step = 2654435761 % total
    if step % 2 == 0:
        step = (step + 1) % total
    start = int(seed) % total
    seen = set()
    j = 0
    while len(seen) < count:
        code = (start + j * step) % total
        j += 1
        if code in seen:
            continue
        seen.add(code)
        yield int(code)


def _code_to_index(code: int, depth: int) -> tuple[int, int, int]:
    n = 1 << depth
    return int(code % n), int((code // n) % n), int(code // (n * n))


def cell_bounds_from_index(
    index_xyz,
    depth: int,
    domain_lo: float,
    domain_hi: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = 1 << depth
    width = (float(domain_hi) - float(domain_lo)) / float(n)
    idx = np.asarray(index_xyz, dtype=np.float64)
    lo = float(domain_lo) + idx * width
    return lo, lo + width


def point_to_cell_index(
    point: np.ndarray,
    depth: int,
    domain_lo: float,
    domain_hi: float,
) -> tuple[int, int, int]:
    point = np.asarray(point, dtype=np.float64)
    n = 1 << depth
    scaled = (point - float(domain_lo)) / (float(domain_hi) - float(domain_lo)) * n
    idx = np.floor(scaled).astype(np.int64)
    idx = np.clip(idx, 0, n - 1)
    return int(idx[0]), int(idx[1]), int(idx[2])


def cell_center(
    index_xyz,
    depth: int,
    domain_lo: float,
    domain_hi: float,
) -> np.ndarray:
    lo, hi = cell_bounds_from_index(index_xyz, depth, domain_lo, domain_hi)
    return (lo + hi) * 0.5


def direct_field_values(
    field,
    planes: torch.Tensor,
    points_np: np.ndarray,
    *,
    batch_size: int = 1024,
) -> np.ndarray:
    pts = np.asarray(points_np, dtype=np.float64)
    out = []
    with torch.no_grad():
        for st in range(0, len(pts), batch_size):
            q = torch.from_numpy(pts[st:st+batch_size]).double().reshape(1, -1, 3)
            out.append(field(planes, q)["sdf"].reshape(-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0,), dtype=np.float64)


def choose_paired_anchors(
    field,
    planes: torch.Tensor,
    *,
    domain_lo: float,
    domain_hi: float,
    anchor_depth: int = 8,
    distinct_depth: int = 5,
    volume_count: int = 32,
    near_zero_count: int = 32,
    near_zero_candidate_count: int = 4096,
    seed: int = 26092111,
) -> list[dict]:
    if anchor_depth < distinct_depth:
        raise ValueError("ANCHOR_DEPTH_LT_DISTINCT_DEPTH")

    anchors: list[dict] = []
    occupied = set()

    # Volume anchors: deterministic depth-8 cell centers, de-duplicated at depth 5.
    for code in _cell_code_sequence(anchor_depth, volume_count * 64, seed + 101):
        p = cell_center(_code_to_index(code, anchor_depth), anchor_depth, domain_lo, domain_hi)
        dkey = point_to_cell_index(p, distinct_depth, domain_lo, domain_hi)
        if dkey in occupied:
            continue
        occupied.add(dkey)
        anchors.append({
            "stratum": "VOLUME",
            "point": p.tolist(),
            "diagnostic_abs_sdf": None,
        })
        if sum(a["stratum"] == "VOLUME" for a in anchors) >= volume_count:
            break

    if sum(a["stratum"] == "VOLUME" for a in anchors) != volume_count:
        raise RuntimeError("VOLUME_ANCHOR_SELECTION_INCOMPLETE")

    # Near-zero anchors: direct field is a sampling heuristic only.
    candidate_codes = list(
        _cell_code_sequence(anchor_depth, near_zero_candidate_count, seed + 202)
    )
    candidate_points = np.asarray([
        cell_center(_code_to_index(code, anchor_depth), anchor_depth, domain_lo, domain_hi)
        for code in candidate_codes
    ])
    candidate_sdf = direct_field_values(field, planes, candidate_points)
    order = np.argsort(np.abs(candidate_sdf), kind="stable")

    picked = 0
    for j in order:
        p = candidate_points[int(j)]
        dkey = point_to_cell_index(p, distinct_depth, domain_lo, domain_hi)
        if dkey in occupied:
            continue
        occupied.add(dkey)
        anchors.append({
            "stratum": "NEAR_ZERO",
            "point": p.tolist(),
            "diagnostic_abs_sdf": float(abs(candidate_sdf[int(j)])),
            "diagnostic_signed_sdf": float(candidate_sdf[int(j)]),
        })
        picked += 1
        if picked >= near_zero_count:
            break

    if picked != near_zero_count:
        raise RuntimeError("NEAR_ZERO_ANCHOR_SELECTION_INCOMPLETE")

    for i, a in enumerate(anchors):
        a["anchor_id"] = int(i)
    return anchors


def _sanity_points(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    axes = [np.linspace(float(lo[i]), float(hi[i]), 3, dtype=np.float64) for i in range(3)]
    return np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T


def run_paired_profile(
    field,
    planes: torch.Tensor,
    *,
    anchors: list[dict],
    depths: Iterable[int],
    domain_lo: float,
    domain_hi: float,
    max_micro_depth: int = 2,
    logger: Callable[[str], None] | None = print,
    record_callback: Callable[[dict], None] | None = None,
) -> dict:
    prepared_field: PreparedField = prepare_field(field)
    prepared_planes = prepare_planes(planes)
    depths = [int(d) for d in depths]
    records: list[dict] = []
    started = time.perf_counter()

    total_parents = len(anchors) * len(depths)
    done = 0
    for anchor in anchors:
        point = np.asarray(anchor["point"], dtype=np.float64)
        for depth in depths:
            idx = point_to_cell_index(point, depth, domain_lo, domain_hi)
            lo, hi = cell_bounds_from_index(idx, depth, domain_lo, domain_hi)

            t0 = time.perf_counter()
            cert = certify_cell_c0_ladder_prepared(
                prepared_field,
                prepared_planes,
                lo,
                hi,
                max_micro_depth=max_micro_depth,
            )
            elapsed = time.perf_counter() - t0

            samples = direct_field_values(field, prepared_planes, _sanity_points(lo, hi))
            smin = float(samples.min())
            smax = float(samples.max())
            sampled_bracket = bool(smin <= 0.0 <= smax)

            by_micro = {}
            contradiction = False
            for micro, s in sorted(cert.by_micro_depth.items()):
                row = {
                    "state": s.state,
                    "sign": int(s.sign),
                    "lower": float(s.lower),
                    "upper": float(s.upper),
                    "unresolved_leaf_count": int(s.unresolved_leaf_count),
                    "positive_leaf_count": int(s.positive_leaf_count),
                    "negative_leaf_count": int(s.negative_leaf_count),
                }
                if s.state == "PROVEN_EMPTY_POSITIVE" and smin <= 0.0:
                    contradiction = True
                if s.state == "PROVEN_EMPTY_NEGATIVE" and smax >= 0.0:
                    contradiction = True
                by_micro[str(micro)] = row

            record = {
                "anchor_id": int(anchor["anchor_id"]),
                "stratum": anchor["stratum"],
                "anchor_point": anchor["point"],
                "anchor_diagnostic_abs_sdf": anchor.get("diagnostic_abs_sdf"),
                "depth": int(depth),
                "cell_index_xyz": [int(x) for x in idx],
                "actual_evaluated_box_count": int(cert.actual_evaluated_box_count),
                "regime_box_count": int(cert.regime_box_count),
                "elapsed_seconds": float(elapsed),
                "sample_min": smin,
                "sample_max": smax,
                "sampled_zero_bracket": sampled_bracket,
                "empirical_contradiction": bool(contradiction),
                "by_micro": by_micro,
            }
            records.append(record)
            if record_callback is not None:
                record_callback(record)

            done += 1
            if logger:
                m2 = by_micro[str(max_micro_depth)]["state"]
                logger(
                    f"PAIRED_PROGRESS {done}/{total_parents} "
                    f"anchor={anchor['anchor_id']} stratum={anchor['stratum']} depth={depth} "
                    f"m{max_micro_depth}={m2} boxes={cert.actual_evaluated_box_count} "
                    f"elapsed={elapsed:.3f}s"
                )

    elapsed_total = time.perf_counter() - started

    summary = {}
    for stratum in ("ALL", "VOLUME", "NEAR_ZERO"):
        summary[stratum] = {}
        source = records if stratum == "ALL" else [r for r in records if r["stratum"] == stratum]
        for depth in depths:
            summary[stratum][str(depth)] = {}
            rows = [r for r in source if r["depth"] == depth]
            for micro in range(max_micro_depth + 1):
                states = Counter(r["by_micro"][str(micro)]["state"] for r in rows)
                decisive = sum(v for k, v in states.items() if k != "UNKNOWN")
                box_counts = np.asarray([r["actual_evaluated_box_count"] for r in rows], dtype=np.float64)
                times = np.asarray([r["elapsed_seconds"] for r in rows], dtype=np.float64)
                summary[stratum][str(depth)][str(micro)] = {
                    "parent_count": int(len(rows)),
                    "states": dict(sorted(states.items())),
                    "fractions": {
                        k: float(v / max(len(rows), 1)) for k, v in sorted(states.items())
                    },
                    "decisive_fraction": float(decisive / max(len(rows), 1)),
                    "actual_box_eval_mean": float(box_counts.mean()) if len(box_counts) else 0.0,
                    "actual_box_eval_p90": float(np.quantile(box_counts, .90)) if len(box_counts) else 0.0,
                    "actual_parent_seconds_mean": float(times.mean()) if len(times) else 0.0,
                    "empirical_contradiction_count": int(sum(r["empirical_contradiction"] for r in rows)),
                    "sampled_zero_bracket_count": int(sum(r["sampled_zero_bracket"] for r in rows)),
                }

    return {
        "schema": "RealSaS.VF11C0PairedProfile.v3",
        "status": "RESEARCH_MEASUREMENT_ONLY__NO_PRODUCT_AUTHORITY",
        "numeric_rigor": "ORDINARY_FLOAT64__NOT_DIRECTED_ROUNDING",
        "depths": depths,
        "max_micro_depth": int(max_micro_depth),
        "anchor_count": int(len(anchors)),
        "parent_count": int(len(records)),
        "elapsed_seconds_total": float(elapsed_total),
        "anchors": anchors,
        "summary": summary,
        "records": records,
        "empirical_contradiction_count": int(sum(r["empirical_contradiction"] for r in records)),
    }
