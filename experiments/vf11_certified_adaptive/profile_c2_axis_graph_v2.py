from __future__ import annotations

"""Frozen-Knight profiler for C2 V2 axis-aligned graph patches."""

from collections import Counter
import time
from typing import Callable

import numpy as np
import torch

from axis_graph_c2_v2 import (
    C1_POS,
    C1_NEG,
    prepare_field,
    prepare_planes,
    certify_axis_graph_c2_v2_prepared,
)
from paired_profile_c0_v3 import cell_bounds_from_index


C1_REGULAR_STATES = {C1_POS, C1_NEG}


def _sample_box_points(lo: np.ndarray, hi: np.ndarray, per_axis: int = 5) -> np.ndarray:
    grids = [
        np.linspace(float(lo[i]), float(hi[i]), per_axis, dtype=np.float64)
        for i in range(3)
    ]
    return np.array(np.meshgrid(*grids, indexing="ij")).reshape(3, -1).T


def _direct_directional_samples(
    field,
    planes: torch.Tensor,
    pts: np.ndarray,
    axis: int,
) -> np.ndarray:
    q = torch.tensor(
        np.asarray(pts, dtype=np.float64).reshape(1, -1, 3),
        dtype=torch.float64,
        requires_grad=True,
    )
    out = field(planes, q)["sdf"]
    grad = torch.autograd.grad(out.sum(), q, create_graph=False)[0]
    return grad[..., axis].reshape(-1).detach().cpu().numpy()


def _sample_face(
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    axis: int,
    coordinate: float,
    per_axis: int = 5,
) -> np.ndarray:
    varying = [a for a in range(3) if a != axis]
    grids = [
        np.linspace(float(lo[a]), float(hi[a]), per_axis, dtype=np.float64)
        for a in varying
    ]
    pts = []
    for u in grids[0]:
        for v in grids[1]:
            p = np.empty(3, dtype=np.float64)
            p[axis] = coordinate
            p[varying[0]] = u
            p[varying[1]] = v
            pts.append(p)
    return np.asarray(pts, dtype=np.float64)


def _direct_values(field, planes: torch.Tensor, pts: np.ndarray) -> np.ndarray:
    q = torch.tensor(
        np.asarray(pts, dtype=np.float64).reshape(1, -1, 3),
        dtype=torch.float64,
    )
    with torch.no_grad():
        out = field(planes, q)["sdf"].reshape(-1)
    return out.detach().cpu().numpy()


def profile_c2_axis_graph_v2(
    field,
    planes: torch.Tensor,
    c1_fixed_profile: dict,
    *,
    domain_lo: float,
    domain_hi: float,
    c1_max_micro_depth: int = 2,
    face_micro_depth: int = 2,
    logger: Callable[[str], None] | None = print,
) -> dict:
    source = [
        r for r in c1_fixed_profile["records"]
        if bool(r["primary_handoff"])
        and r["c1_v2_state"] in C1_REGULAR_STATES
    ]
    if len(source) != 12:
        raise ValueError(f"EXPECTED_12_PRIMARY_C1_REGULAR_CELLS:{len(source)}")

    p = prepare_field(field)
    pp = prepare_planes(planes)
    rows = []
    started = time.perf_counter()

    for ordinal, src in enumerate(source, 1):
        anchor = int(src["anchor_id"])
        idx = tuple(int(v) for v in src["cell_index_xyz"])
        lo, hi = cell_bounds_from_index(
            idx, int(src["depth"]), domain_lo, domain_hi
        )

        t0 = time.perf_counter()
        cert = certify_axis_graph_c2_v2_prepared(
            p,
            pp,
            lo,
            hi,
            c1_max_micro_depth=c1_max_micro_depth,
            face_micro_depth=face_micro_depth,
        )
        elapsed = time.perf_counter() - t0

        empirical_contradictions = []
        attempt_rows = []

        for a in cert.attempts:
            deriv = a.derivative_result
            deriv_contradiction = False
            entry_contradiction = False
            exit_contradiction = False

            if deriv.state in C1_REGULAR_STATES:
                pts = _sample_box_points(lo, hi, per_axis=5)
                vals = _direct_directional_samples(
                    field, pp, pts, a.axis
                )
                if deriv.state == C1_POS:
                    deriv_contradiction = bool(np.any(vals <= 0.0))
                else:
                    deriv_contradiction = bool(np.any(vals >= 0.0))
                if deriv_contradiction:
                    empirical_contradictions.append({
                        "axis": a.axis,
                        "kind": "DERIVATIVE_SIGN",
                        "sample_min": float(vals.min()),
                        "sample_max": float(vals.max()),
                    })

            for face_name, fc in (
                ("ENTRY", a.entry_face),
                ("EXIT", a.exit_face),
            ):
                if fc is None or fc.state != "PROVEN_FACE_SIGN":
                    continue
                pts = _sample_face(
                    lo,
                    hi,
                    axis=fc.axis,
                    coordinate=fc.coordinate,
                    per_axis=5,
                )
                vals = _direct_values(field, pp, pts)
                contradiction = (
                    bool(np.any(vals <= 0.0))
                    if fc.desired_sign > 0
                    else bool(np.any(vals >= 0.0))
                )
                if face_name == "ENTRY":
                    entry_contradiction = contradiction
                else:
                    exit_contradiction = contradiction
                if contradiction:
                    empirical_contradictions.append({
                        "axis": a.axis,
                        "kind": f"{face_name}_FACE_SIGN",
                        "sample_min": float(vals.min()),
                        "sample_max": float(vals.max()),
                    })

            attempt_rows.append({
                "axis": int(a.axis),
                "axis_name": a.axis_name,
                "derivative_state": deriv.state,
                "derivative_margin": a.derivative_margin,
                "oriented_sign": a.oriented_sign,
                "entry_face_state": None if a.entry_face is None else a.entry_face.state,
                "entry_face_margin": None if a.entry_face is None else a.entry_face.margin,
                "entry_face_unresolved_leaf_count": (
                    None if a.entry_face is None else a.entry_face.unresolved_leaf_count
                ),
                "exit_face_state": None if a.exit_face is None else a.exit_face.state,
                "exit_face_margin": None if a.exit_face is None else a.exit_face.margin,
                "exit_face_unresolved_leaf_count": (
                    None if a.exit_face is None else a.exit_face.unresolved_leaf_count
                ),
                "graph_certified": bool(a.graph_certified),
                "bottleneck_margin": a.bottleneck_margin,
                "evaluated_box_count": int(a.evaluated_box_count),
                "derivative_empirical_contradiction": deriv_contradiction,
                "entry_empirical_contradiction": entry_contradiction,
                "exit_empirical_contradiction": exit_contradiction,
            })

        row = {
            "anchor_id": anchor,
            "cell_index_xyz": list(idx),
            "depth": int(src["depth"]),
            "c0_state_m2": src["c0_state_m2"],
            "source_c1_state": src["c1_v2_state"],
            "source_c1_direction": src["c1_v2_direction"],
            "source_c1_terminal_margin": src["c1_v2_certificate_margin"],
            "axis_graph_state": cert.state,
            "selected_axis": cert.selected_axis,
            "selected_axis_name": cert.selected_axis_name,
            "selected_oriented_sign": cert.selected_oriented_sign,
            "bottleneck_margin": cert.bottleneck_margin,
            "evaluated_box_count": int(cert.evaluated_box_count),
            "attempts": attempt_rows,
            "empirical_contradictions": empirical_contradictions,
            "empirical_contradiction": bool(empirical_contradictions),
            "elapsed_seconds": float(elapsed),
        }
        rows.append(row)

        if logger:
            logger(
                f"C2V2_AXIS_GRAPH {ordinal}/{len(source)} anchor={anchor} "
                f"state={cert.state} axis={cert.selected_axis_name} "
                f"margin={cert.bottleneck_margin} boxes={cert.evaluated_box_count} "
                f"elapsed={elapsed:.3f}s contradictions={len(empirical_contradictions)}"
            )

    states = Counter(r["axis_graph_state"] for r in rows)
    graph_rows = [r for r in rows if r["axis_graph_state"] == "PROVEN_AXIS_GRAPH_PATCH"]
    axis_counts = Counter(r["selected_axis_name"] for r in graph_rows)
    margins = np.asarray(
        [r["bottleneck_margin"] for r in graph_rows],
        dtype=np.float64,
    )
    boxes = np.asarray([r["evaluated_box_count"] for r in rows], dtype=np.float64)
    secs = np.asarray([r["elapsed_seconds"] for r in rows], dtype=np.float64)
    contradictions = sum(r["empirical_contradiction"] for r in rows)

    regular_axis_counts = Counter()
    for r in rows:
        n = sum(
            a["derivative_state"] in C1_REGULAR_STATES
            for a in r["attempts"]
        )
        regular_axis_counts[str(n)] += 1

    return {
        "schema": "RealSaS.VF11C2V2AxisGraphProfile.v1",
        "status": "RESEARCH_MEASUREMENT_ONLY__NO_PRODUCT_AUTHORITY",
        "numeric_rigor": "ORDINARY_FLOAT64__NOT_DIRECTED_ROUNDING",
        "primary_c1_regular_count": 12,
        "state_counts": dict(sorted(states.items())),
        "graph_patch_count": int(len(graph_rows)),
        "graph_patch_fraction": float(len(graph_rows) / len(rows)),
        "selected_axis_counts": dict(sorted(axis_counts.items())),
        "regular_axis_count_distribution": dict(sorted(regular_axis_counts.items())),
        "bottleneck_margin_min": None if margins.size == 0 else float(margins.min()),
        "bottleneck_margin_median": None if margins.size == 0 else float(np.median(margins)),
        "box_eval_mean": float(boxes.mean()),
        "box_eval_p90": float(np.quantile(boxes, 0.90)),
        "seconds_mean": float(secs.mean()),
        "elapsed_seconds_total": float(time.perf_counter() - started),
        "empirical_contradiction_count": int(contradictions),
        "records": rows,
    }
