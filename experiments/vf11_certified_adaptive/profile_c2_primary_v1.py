from __future__ import annotations

"""Frozen-Knight C2 oriented-crossing profiler.

Consumes the corrected C1 V2 margin-replay profile and evaluates only the
12 primary handoff cells already certified directionally regular.

Research-only. No full-domain traversal. No topology/single-component claim.
"""

from collections import Counter
import math
import time
from typing import Callable

import numpy as np
import torch

from oriented_crossing_c2_v1 import (
    C2CrossingCertificate,
    prepare_field,
    prepare_planes,
    certify_oriented_crossing_c2_prepared,
)
from paired_profile_c0_v3 import cell_bounds_from_index


C1_REGULAR_STATES = {
    "PROVEN_DIRECTIONAL_REGULAR_POSITIVE",
    "PROVEN_DIRECTIONAL_REGULAR_NEGATIVE",
}


def wilson95(successes: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = successes / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / den
    half = z * math.sqrt(
        p * (1.0 - p) / n + z * z / (4.0 * n * n)
    ) / den
    return max(0.0, center - half), min(1.0, center + half)


def _direct_values(field, planes: torch.Tensor, pts: np.ndarray) -> np.ndarray:
    q = torch.tensor(
        np.asarray(pts, dtype=np.float64).reshape(1, -1, 3),
        dtype=torch.float64,
    )
    with torch.no_grad():
        out = field(planes, q)["sdf"].reshape(-1)
    return out.detach().cpu().numpy()


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
            p[axis] = float(coordinate)
            p[varying[0]] = u
            p[varying[1]] = v
            pts.append(p)
    return np.asarray(pts, dtype=np.float64)


def _certificate_to_dict(cert: C2CrossingCertificate) -> dict:
    return {
        "state": cert.state,
        "oriented_direction": list(cert.oriented_direction),
        "center_entry": {
            "point": list(cert.center_entry.point),
            "lower": cert.center_entry.lower,
            "upper": cert.center_entry.upper,
        },
        "center_exit": {
            "point": list(cert.center_exit.point),
            "lower": cert.center_exit.lower,
            "upper": cert.center_exit.upper,
        },
        "center_chord_certified": cert.center_chord_certified,
        "center_entry_margin": cert.center_entry_margin,
        "center_exit_margin": cert.center_exit_margin,
        "full_oriented_crossing_certified": cert.full_oriented_crossing_certified,
        "c0_zero_exists": cert.c0_zero_exists,
        "evaluated_box_count": cert.evaluated_box_count,
        "face_certificates": [
            {
                "axis": x.axis,
                "coordinate": x.coordinate,
                "role": x.role,
                "desired_sign": x.desired_sign,
                "state": x.state,
                "margin": x.margin,
                "evaluated_box_count": x.evaluated_box_count,
                "certified_leaf_count": x.certified_leaf_count,
                "unresolved_leaf_count": x.unresolved_leaf_count,
            }
            for x in cert.face_certificates
        ],
    }


def profile_c2_primary(
    field,
    planes: torch.Tensor,
    c1_replay_profile: dict,
    *,
    domain_lo: float,
    domain_hi: float,
    face_micro_depth: int = 2,
    logger: Callable[[str], None] | None = print,
) -> dict:
    source = [
        r for r in c1_replay_profile["records"]
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
        cert = certify_oriented_crossing_c2_prepared(
            p,
            pp,
            lo,
            hi,
            c1_state=src["c1_v2_state"],
            c1_direction=np.asarray(src["c1_v2_direction"], dtype=np.float64),
            c0_state=src["c0_state_m2"],
            face_micro_depth=face_micro_depth,
        )
        elapsed = time.perf_counter() - t0

        # Empirical diagnostics only.
        endpoint_pts = np.asarray(
            [cert.center_entry.point, cert.center_exit.point],
            dtype=np.float64,
        )
        endpoint_vals = _direct_values(field, pp, endpoint_pts)
        center_empirical_contradiction = False
        if cert.center_chord_certified:
            center_empirical_contradiction = not (
                float(endpoint_vals[0]) < 0.0
                and float(endpoint_vals[1]) > 0.0
            )

        face_empirical_contradictions = []
        for fc in cert.face_certificates:
            if fc.state != "PROVEN_FACE_SIGN":
                continue
            pts = _sample_face(
                lo, hi,
                axis=fc.axis,
                coordinate=fc.coordinate,
                per_axis=5,
            )
            vals = _direct_values(field, pp, pts)
            if fc.desired_sign > 0:
                contradiction = bool(np.any(vals <= 0.0))
            else:
                contradiction = bool(np.any(vals >= 0.0))
            if contradiction:
                face_empirical_contradictions.append({
                    "axis": fc.axis,
                    "role": fc.role,
                    "desired_sign": fc.desired_sign,
                    "sample_min": float(vals.min()),
                    "sample_max": float(vals.max()),
                })

        contradiction = bool(
            center_empirical_contradiction
            or face_empirical_contradictions
        )

        row = {
            "anchor_id": anchor,
            "cell_index_xyz": list(idx),
            "depth": int(src["depth"]),
            "c0_state_m2": src["c0_state_m2"],
            "c1_state": src["c1_v2_state"],
            "c1_direction": list(src["c1_v2_direction"]),
            "c1_terminal_margin": src["c1_v2_certificate_margin"],
            "c2": _certificate_to_dict(cert),
            "existence_proven": cert.state != "REGULAR_ONLY",
            "line_uniqueness_proven": True,
            "full_chord_family_crossing_proven": (
                cert.state == "PROVEN_FULL_ORIENTED_CROSSING"
            ),
            "center_endpoint_direct_values": [
                float(endpoint_vals[0]), float(endpoint_vals[1])
            ],
            "center_empirical_contradiction": bool(center_empirical_contradiction),
            "face_empirical_contradictions": face_empirical_contradictions,
            "empirical_contradiction": contradiction,
            "elapsed_seconds": float(elapsed),
        }
        rows.append(row)

        if logger:
            logger(
                f"C2_PROGRESS {ordinal}/{len(source)} anchor={anchor} "
                f"c0={row['c0_state_m2']} c1={row['c1_state']} "
                f"c2={cert.state} boxes={cert.evaluated_box_count} "
                f"elapsed={elapsed:.3f}s contradiction={contradiction}"
            )

    states = Counter(r["c2"]["state"] for r in rows)
    existence = sum(r["existence_proven"] for r in rows)
    full = sum(r["full_chord_family_crossing_proven"] for r in rows)
    center = sum(r["c2"]["center_chord_certified"] for r in rows)
    contradictions = sum(r["empirical_contradiction"] for r in rows)

    existence_ci = wilson95(existence, len(rows))
    full_ci = wilson95(full, len(rows))

    center_margins = []
    face_margins = []
    boxes = []
    secs = []
    for r in rows:
        c = r["c2"]
        if c["center_chord_certified"]:
            center_margins.extend([
                float(c["center_entry_margin"]),
                float(c["center_exit_margin"]),
            ])
        for fc in c["face_certificates"]:
            if fc["margin"] is not None:
                face_margins.append(float(fc["margin"]))
        boxes.append(float(c["evaluated_box_count"]))
        secs.append(float(r["elapsed_seconds"]))

    return {
        "schema": "RealSaS.VF11C2PrimaryProfile.v1",
        "status": "RESEARCH_MEASUREMENT_ONLY__NO_PRODUCT_AUTHORITY",
        "numeric_rigor": "ORDINARY_FLOAT64__NOT_DIRECTED_ROUNDING",
        "face_micro_depth": int(face_micro_depth),
        "primary_c1_regular_count": 12,
        "state_counts": dict(sorted(states.items())),
        "existence_proven_count": int(existence),
        "existence_proven_fraction": float(existence / len(rows)),
        "existence_proven_wilson95": [float(x) for x in existence_ci],
        "center_chord_crossing_count": int(center),
        "full_oriented_crossing_count": int(full),
        "full_oriented_crossing_fraction": float(full / len(rows)),
        "full_oriented_crossing_wilson95": [float(x) for x in full_ci],
        "center_margin_min": (
            None if not center_margins else float(min(center_margins))
        ),
        "center_margin_median": (
            None if not center_margins else float(np.median(center_margins))
        ),
        "face_margin_min": (
            None if not face_margins else float(min(face_margins))
        ),
        "face_margin_median": (
            None if not face_margins else float(np.median(face_margins))
        ),
        "box_eval_mean": float(np.mean(boxes)),
        "box_eval_p90": float(np.quantile(boxes, 0.90)),
        "seconds_mean": float(np.mean(secs)),
        "elapsed_seconds_total": float(time.perf_counter() - started),
        "empirical_contradiction_count": int(contradictions),
        "records": rows,
    }
