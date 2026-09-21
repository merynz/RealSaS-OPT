from __future__ import annotations

"""Research-only C2 oriented crossing engine for VF-11.

C2 consumes an already-certified C1 directional-regularity cell. The C1
certificate is oriented so that D_d f > 0 throughout the cell. C2 then adds
value-sign evidence:

1) Center chord:
   If the chord through the box center parallel to d has conservatively
   negative entry and positive exit values, continuity + strict monotonicity
   proves exactly one zero on that chord.

2) Full oriented boundary crossing:
   For an axis-aligned convex box, every d-parallel chord enters through one of
   the inflow faces and exits through one of the outflow faces. If every inflow
   face is strictly negative and every outflow face strictly positive, then
   every nondegenerate d-parallel chord intersecting the box has exactly one
   zero.

3) C0 bridge:
   If C0 already proved zero existence, C1 regularity adds at-most-one-zero per
   d-parallel chord, even if C2 cannot prove a full boundary crossing.

This does NOT prove a single connected component unless the stronger full
oriented crossing condition is established, and even then the claim is only
the stated chord-family graph property, not global topology.

Ordinary IEEE-754 float64 only; research certificate, not shipping authority.
"""

from dataclasses import dataclass
import itertools
import math

import numpy as np
import torch

from range_engine_c0_v3 import (
    PreparedField,
    bound_single_regime_prepared,
    interpolation_knots,
    prepare_field,
    prepare_planes,
)


C1_POS = "PROVEN_DIRECTIONAL_REGULAR_POSITIVE"
C1_NEG = "PROVEN_DIRECTIONAL_REGULAR_NEGATIVE"


@dataclass(frozen=True)
class PointValueBound:
    lower: float
    upper: float
    point: tuple[float, float, float]


@dataclass(frozen=True)
class FaceSignCertificate:
    axis: int
    coordinate: float
    role: str
    desired_sign: int
    state: str
    margin: float | None
    evaluated_box_count: int
    certified_leaf_count: int
    unresolved_leaf_count: int


@dataclass(frozen=True)
class C2CrossingCertificate:
    state: str
    oriented_direction: tuple[float, float, float]
    center_entry: PointValueBound
    center_exit: PointValueBound
    center_chord_certified: bool
    center_entry_margin: float | None
    center_exit_margin: float | None
    full_oriented_crossing_certified: bool
    face_certificates: tuple[FaceSignCertificate, ...]
    c0_zero_exists: bool
    evaluated_box_count: int
    numerically_rigorous: bool = False


def orient_increasing_direction(
    c1_state: str,
    direction: np.ndarray,
) -> np.ndarray:
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(d))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("INVALID_C1_DIRECTION")
    d = d / norm
    if c1_state == C1_POS:
        return d
    if c1_state == C1_NEG:
        return -d
    raise ValueError(f"C1_STATE_NOT_REGULAR:{c1_state}")


def center_chord_endpoints(
    lo: np.ndarray,
    hi: np.ndarray,
    direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lo = np.asarray(lo, dtype=np.float64).reshape(3)
    hi = np.asarray(hi, dtype=np.float64).reshape(3)
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    if bool(np.any(hi <= lo)):
        raise ValueError("INVALID_BOX")
    center = (lo + hi) * 0.5
    half = (hi - lo) * 0.5
    candidates = [
        float(half[i] / abs(d[i]))
        for i in range(3)
        if float(d[i]) != 0.0
    ]
    if not candidates:
        raise ValueError("ZERO_DIRECTION")
    t = min(candidates)
    entry = center - t * d
    exit = center + t * d
    entry = np.minimum(np.maximum(entry, lo), hi)
    exit = np.minimum(np.maximum(exit, lo), hi)
    return entry, exit


def point_value_bound_prepared(
    p: PreparedField,
    planes: torch.Tensor,
    point: np.ndarray,
) -> PointValueBound:
    q = np.asarray(point, dtype=np.float64).reshape(3)
    lower, upper = bound_single_regime_prepared(p, planes, q, q)
    return PointValueBound(
        lower=float(lower),
        upper=float(upper),
        point=tuple(float(v) for v in q),
    )


def _face_patch_split_at_knots(
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    fixed_axis: int,
    size: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    lo = np.asarray(lo, dtype=np.float64).reshape(3)
    hi = np.asarray(hi, dtype=np.float64).reshape(3)
    if not (0 <= fixed_axis < 3):
        raise ValueError("INVALID_FIXED_AXIS")
    if abs(float(hi[fixed_axis] - lo[fixed_axis])) > 1e-15:
        raise ValueError("FACE_AXIS_NOT_FIXED")

    knots = interpolation_knots(size)
    axes_segments: dict[int, list[tuple[float, float]]] = {}
    tol = 4e-15
    for axis in range(3):
        if axis == fixed_axis:
            continue
        if not hi[axis] > lo[axis]:
            raise ValueError("FACE_VARYING_AXIS_NOT_POSITIVE")
        points = [float(lo[axis])]
        points.extend(
            float(k)
            for k in knots
            if lo[axis] + tol < k < hi[axis] - tol
        )
        points.append(float(hi[axis]))
        points = sorted(set(points))
        axes_segments[axis] = [
            (points[i], points[i + 1])
            for i in range(len(points) - 1)
        ]

    varying = [a for a in range(3) if a != fixed_axis]
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for s0, s1 in itertools.product(
        axes_segments[varying[0]],
        axes_segments[varying[1]],
    ):
        a = lo.copy()
        b = hi.copy()
        a[varying[0]], b[varying[0]] = s0
        a[varying[1]], b[varying[1]] = s1
        a[fixed_axis] = b[fixed_axis] = lo[fixed_axis]
        out.append((a, b))
    return out


def _split_face_quads(
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    fixed_axis: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    lo = np.asarray(lo, dtype=np.float64).reshape(3)
    hi = np.asarray(hi, dtype=np.float64).reshape(3)
    varying = [a for a in range(3) if a != fixed_axis]
    mid = (lo + hi) * 0.5
    out = []
    for bits in itertools.product((0, 1), repeat=2):
        a = lo.copy()
        b = hi.copy()
        for bit, axis in zip(bits, varying):
            if bit == 0:
                b[axis] = mid[axis]
            else:
                a[axis] = mid[axis]
        a[fixed_axis] = b[fixed_axis] = lo[fixed_axis]
        out.append((a, b))
    return out


def certify_face_sign_prepared(
    p: PreparedField,
    planes: torch.Tensor,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    fixed_axis: int,
    role: str,
    desired_sign: int,
    max_micro_depth: int = 2,
) -> FaceSignCertificate:
    if desired_sign not in (-1, 1):
        raise ValueError("DESIRED_SIGN_MUST_BE_PLUS_OR_MINUS_ONE")
    if max_micro_depth < 0:
        raise ValueError("NEGATIVE_MICRO_DEPTH")

    patches = _face_patch_split_at_knots(
        lo, hi, fixed_axis=fixed_axis, size=int(planes.shape[-1])
    )
    stack = [(a, b, 0) for a, b in patches]
    evaluated = 0
    certified = 0
    unresolved = 0
    min_margin = math.inf

    while stack:
        a, b, depth = stack.pop()
        lower, upper = bound_single_regime_prepared(p, planes, a, b)
        evaluated += 1

        if desired_sign > 0:
            ok = lower > 0.0
            margin = float(lower)
        else:
            ok = upper < 0.0
            margin = float(-upper)

        if ok:
            certified += 1
            min_margin = min(min_margin, margin)
            continue

        if depth < max_micro_depth:
            stack.extend(
                (c, d, depth + 1)
                for c, d in _split_face_quads(
                    a, b, fixed_axis=fixed_axis
                )
            )
        else:
            unresolved += 1

    state = "PROVEN_FACE_SIGN" if unresolved == 0 else "UNKNOWN"
    return FaceSignCertificate(
        axis=int(fixed_axis),
        coordinate=float(lo[fixed_axis]),
        role=str(role),
        desired_sign=int(desired_sign),
        state=state,
        margin=None if unresolved else float(min_margin),
        evaluated_box_count=int(evaluated),
        certified_leaf_count=int(certified),
        unresolved_leaf_count=int(unresolved),
    )


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


def certify_oriented_crossing_c2_prepared(
    p: PreparedField,
    planes: torch.Tensor,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    c1_state: str,
    c1_direction: np.ndarray,
    c0_state: str,
    face_micro_depth: int = 2,
) -> C2CrossingCertificate:
    lo = np.asarray(lo, dtype=np.float64).reshape(3)
    hi = np.asarray(hi, dtype=np.float64).reshape(3)
    d = orient_increasing_direction(c1_state, c1_direction)

    entry_pt, exit_pt = center_chord_endpoints(lo, hi, d)
    entry_bound = point_value_bound_prepared(p, planes, entry_pt)
    exit_bound = point_value_bound_prepared(p, planes, exit_pt)
    center_entry_ok = entry_bound.upper < 0.0
    center_exit_ok = exit_bound.lower > 0.0
    center_ok = bool(center_entry_ok and center_exit_ok)

    face_certs: list[FaceSignCertificate] = []
    for axis in range(3):
        comp = float(d[axis])
        if comp == 0.0:
            continue

        entry_coord = float(lo[axis] if comp > 0.0 else hi[axis])
        exit_coord = float(hi[axis] if comp > 0.0 else lo[axis])

        elo, ehi = _face_box(lo, hi, axis, entry_coord)
        xlo, xhi = _face_box(lo, hi, axis, exit_coord)

        face_certs.append(
            certify_face_sign_prepared(
                p,
                planes,
                elo,
                ehi,
                fixed_axis=axis,
                role="INFLOW",
                desired_sign=-1,
                max_micro_depth=face_micro_depth,
            )
        )
        face_certs.append(
            certify_face_sign_prepared(
                p,
                planes,
                xlo,
                xhi,
                fixed_axis=axis,
                role="OUTFLOW",
                desired_sign=1,
                max_micro_depth=face_micro_depth,
            )
        )

    full_ok = bool(face_certs) and all(
        fc.state == "PROVEN_FACE_SIGN" for fc in face_certs
    )
    c0_exists = c0_state == "PROVEN_ZERO_EXISTS"

    if full_ok:
        state = "PROVEN_FULL_ORIENTED_CROSSING"
    elif center_ok:
        state = "PROVEN_CENTER_CHORD_CROSSING"
    elif c0_exists:
        state = "PROVEN_C0_EXISTENCE_PLUS_LINE_UNIQUENESS"
    else:
        state = "REGULAR_ONLY"

    return C2CrossingCertificate(
        state=state,
        oriented_direction=tuple(float(v) for v in d),
        center_entry=entry_bound,
        center_exit=exit_bound,
        center_chord_certified=center_ok,
        center_entry_margin=float(-entry_bound.upper) if center_entry_ok else None,
        center_exit_margin=float(exit_bound.lower) if center_exit_ok else None,
        full_oriented_crossing_certified=full_ok,
        face_certificates=tuple(face_certs),
        c0_zero_exists=bool(c0_exists),
        evaluated_box_count=(
            2 + sum(fc.evaluated_box_count for fc in face_certs)
        ),
    )


def certify_oriented_crossing_c2(
    field,
    planes: torch.Tensor,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    c1_state: str,
    c1_direction: np.ndarray,
    c0_state: str,
    face_micro_depth: int = 2,
) -> C2CrossingCertificate:
    return certify_oriented_crossing_c2_prepared(
        prepare_field(field),
        prepare_planes(planes),
        lo,
        hi,
        c1_state=c1_state,
        c1_direction=c1_direction,
        c0_state=c0_state,
        face_micro_depth=face_micro_depth,
    )
