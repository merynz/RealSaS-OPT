from __future__ import annotations

"""Research-only C2 V2 axis-aligned graph-patch certificate.

For each coordinate axis e_i:
1) certify strict whole-cell directional regularity with the corrected C1 V2 engine,
2) orient the axis so D_i f > 0,
3) certify f < 0 on the entry coordinate face and f > 0 on the opposite exit face.

If all three hold, every axis-parallel chord spanning the cell has exactly one
zero. The zero set in the cell is therefore a local graph over the connected
transverse coordinate rectangle.

This is a local cell statement only. It is not a global topology/component
claim and ordinary float64 remains research-only.
"""

from dataclasses import dataclass
import math

import numpy as np
import torch

from directional_regular_c1_v2 import (
    C1CandidateResult,
    _evaluate_candidate_v2,
    prepare_field,
    prepare_planes,
)
from oriented_crossing_c2_v1 import (
    FaceSignCertificate,
    certify_face_sign_prepared,
)


C1_POS = "PROVEN_DIRECTIONAL_REGULAR_POSITIVE"
C1_NEG = "PROVEN_DIRECTIONAL_REGULAR_NEGATIVE"
C1_REGULAR = {C1_POS, C1_NEG}


@dataclass(frozen=True)
class AxisGraphAttempt:
    axis: int
    axis_name: str
    derivative_result: C1CandidateResult
    oriented_sign: int | None
    derivative_margin: float | None
    entry_face: FaceSignCertificate | None
    exit_face: FaceSignCertificate | None
    graph_certified: bool
    bottleneck_margin: float | None
    evaluated_box_count: int


@dataclass(frozen=True)
class AxisGraphCertificate:
    state: str
    selected_axis: int | None
    selected_axis_name: str | None
    selected_oriented_sign: int | None
    bottleneck_margin: float | None
    attempts: tuple[AxisGraphAttempt, ...]
    evaluated_box_count: int
    numerically_rigorous: bool = False


def _face_box(
    lo: np.ndarray,
    hi: np.ndarray,
    axis: int,
    coordinate: float,
) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(lo, dtype=np.float64).copy()
    b = np.asarray(hi, dtype=np.float64).copy()
    a[axis] = b[axis] = float(coordinate)
    return a, b


def _axis_name(axis: int) -> str:
    return ("X", "Y", "Z")[axis]


def certify_axis_graph_c2_v2_prepared(
    p,
    planes: torch.Tensor,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    c1_max_micro_depth: int = 2,
    face_micro_depth: int = 2,
) -> AxisGraphCertificate:
    lo = np.asarray(lo, dtype=np.float64).reshape(3)
    hi = np.asarray(hi, dtype=np.float64).reshape(3)
    if bool(np.any(hi <= lo)):
        raise ValueError("INVALID_BOX")

    attempts: list[AxisGraphAttempt] = []
    total_evals = 0

    for axis in range(3):
        direction = np.zeros(3, dtype=np.float64)
        direction[axis] = 1.0

        deriv = _evaluate_candidate_v2(
            p,
            planes,
            lo,
            hi,
            direction,
            max_micro_depth=c1_max_micro_depth,
        )
        total_evals += int(deriv.evaluated_box_count)

        oriented_sign = None
        derivative_margin = None
        entry = None
        exit = None
        graph = False
        bottleneck = None

        if deriv.state in C1_REGULAR:
            oriented_sign = 1 if deriv.state == C1_POS else -1
            derivative_margin = float(deriv.certificate_margin)
            if not (math.isfinite(derivative_margin) and derivative_margin > 0.0):
                raise RuntimeError(
                    f"INVALID_AXIS_DERIVATIVE_MARGIN:axis={axis}:"
                    f"state={deriv.state}:margin={derivative_margin}"
                )

            entry_coord = float(lo[axis] if oriented_sign > 0 else hi[axis])
            exit_coord = float(hi[axis] if oriented_sign > 0 else lo[axis])

            elo, ehi = _face_box(lo, hi, axis, entry_coord)
            xlo, xhi = _face_box(lo, hi, axis, exit_coord)

            entry = certify_face_sign_prepared(
                p,
                planes,
                elo,
                ehi,
                fixed_axis=axis,
                role="AXIS_ENTRY",
                desired_sign=-1,
                max_micro_depth=face_micro_depth,
            )
            exit = certify_face_sign_prepared(
                p,
                planes,
                xlo,
                xhi,
                fixed_axis=axis,
                role="AXIS_EXIT",
                desired_sign=1,
                max_micro_depth=face_micro_depth,
            )
            total_evals += int(entry.evaluated_box_count + exit.evaluated_box_count)

            graph = bool(
                entry.state == "PROVEN_FACE_SIGN"
                and exit.state == "PROVEN_FACE_SIGN"
            )
            if graph:
                bottleneck = float(
                    min(
                        derivative_margin,
                        float(entry.margin),
                        float(exit.margin),
                    )
                )
                if not (math.isfinite(bottleneck) and bottleneck > 0.0):
                    raise RuntimeError(
                        f"INVALID_AXIS_GRAPH_MARGIN:axis={axis}:margin={bottleneck}"
                    )

        attempts.append(
            AxisGraphAttempt(
                axis=axis,
                axis_name=_axis_name(axis),
                derivative_result=deriv,
                oriented_sign=oriented_sign,
                derivative_margin=derivative_margin,
                entry_face=entry,
                exit_face=exit,
                graph_certified=graph,
                bottleneck_margin=bottleneck,
                evaluated_box_count=(
                    int(deriv.evaluated_box_count)
                    + (0 if entry is None else int(entry.evaluated_box_count))
                    + (0 if exit is None else int(exit.evaluated_box_count))
                ),
            )
        )

    successes = [a for a in attempts if a.graph_certified]
    if not successes:
        return AxisGraphCertificate(
            state="UNKNOWN_AXIS_GRAPH",
            selected_axis=None,
            selected_axis_name=None,
            selected_oriented_sign=None,
            bottleneck_margin=None,
            attempts=tuple(attempts),
            evaluated_box_count=total_evals,
        )

    # Frozen preregistered selection: maximize the minimum proof margin;
    # deterministic tie-break X, then Y, then Z.
    best = max(
        successes,
        key=lambda a: (float(a.bottleneck_margin), -int(a.axis)),
    )
    return AxisGraphCertificate(
        state="PROVEN_AXIS_GRAPH_PATCH",
        selected_axis=int(best.axis),
        selected_axis_name=str(best.axis_name),
        selected_oriented_sign=int(best.oriented_sign),
        bottleneck_margin=float(best.bottleneck_margin),
        attempts=tuple(attempts),
        evaluated_box_count=total_evals,
    )


def certify_axis_graph_c2_v2(
    field,
    planes: torch.Tensor,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    c1_max_micro_depth: int = 2,
    face_micro_depth: int = 2,
) -> AxisGraphCertificate:
    return certify_axis_graph_c2_v2_prepared(
        prepare_field(field),
        prepare_planes(planes),
        lo,
        hi,
        c1_max_micro_depth=c1_max_micro_depth,
        face_micro_depth=face_micro_depth,
    )
