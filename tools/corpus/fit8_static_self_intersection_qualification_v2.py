from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from tools.corpus.fit8_static_self_intersection_qualification_v1 import (
    SCHEMA_VERSION as V1_SCHEMA_VERSION,
    StaticSelfIntersectionPolicyV1,
    StaticSelfIntersectionQualificationV1,
    _minimum_triangle_altitude,
)

SCHEMA_VERSION = "RealSaS.StaticSelfIntersectionQualification.v2"
NARROWPHASE_AUTHORITY = "VECTORIZED_SAT_EQUIVALENT_TO_COMPILER_DYNAMIC_GEOMETRY_INTEGRITY_V2"
BROADPHASE_AUTHORITY = "EXACT_AABB_OVERLAP_SUPERSET_BLOCKWISE"


def _sat_intersects_batch(
    triangle_a: np.ndarray,
    triangle_b: np.ndarray,
    *,
    tolerance: float,
) -> np.ndarray:
    """Batch form of compiler `triangles_intersect_sat` with identical axes.

    Degenerate triangles are rejected instead of silently skipped. Zero-length
    candidate axes are ignored exactly as the scalar Compiler implementation does.
    """
    a = np.asarray(triangle_a, dtype=np.float64)
    b = np.asarray(triangle_b, dtype=np.float64)
    if a.ndim != 3 or a.shape[1:] != (3, 3) or b.shape != a.shape:
        raise ValueError("STATIC_SELF_INTERSECTION_V2_TRIANGLE_BATCH_INVALID")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("STATIC_SELF_INTERSECTION_V2_TRIANGLE_NONFINITE")
    if not isfinite(float(tolerance)) or tolerance < 0.0:
        raise ValueError("STATIC_SELF_INTERSECTION_V2_TOLERANCE_INVALID")
    if len(a) == 0:
        return np.zeros((0,), dtype=bool)

    ea = np.stack((a[:, 1] - a[:, 0], a[:, 2] - a[:, 1], a[:, 0] - a[:, 2]), axis=1)
    eb = np.stack((b[:, 1] - b[:, 0], b[:, 2] - b[:, 1], b[:, 0] - b[:, 2]), axis=1)
    na = np.cross(ea[:, 0], ea[:, 1])
    nb = np.cross(eb[:, 0], eb[:, 1])
    if np.any(np.linalg.norm(na, axis=1) <= 1.0e-14) or np.any(np.linalg.norm(nb, axis=1) <= 1.0e-14):
        raise ValueError("STATIC_SELF_INTERSECTION_V2_DEGENERATE_TRIANGLE")

    axes = [na, nb]
    axes.extend(np.cross(ea[:, i], eb[:, j]) for i in range(3) for j in range(3))
    axes.extend(np.cross(na, ea[:, i]) for i in range(3))
    axes.extend(np.cross(nb, eb[:, i]) for i in range(3))
    axis = np.stack(axes, axis=1)
    norm = np.linalg.norm(axis, axis=2)
    valid = norm > 1.0e-14
    unit = np.divide(axis, norm[:, :, None], out=np.zeros_like(axis), where=valid[:, :, None])

    projection_a = np.einsum("pvc,pac->pav", a, unit)
    projection_b = np.einsum("pvc,pac->pav", b, unit)
    separated = (
        (projection_a.max(axis=2) < projection_b.min(axis=2) - float(tolerance))
        | (projection_b.max(axis=2) < projection_a.min(axis=2) - float(tolerance))
    )
    separated &= valid
    return ~np.any(separated, axis=1)


def _aabb_candidate_pairs_blockwise(
    triangles: np.ndarray,
    *,
    tolerance: float,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if block_size < 1:
        raise ValueError("STATIC_SELF_INTERSECTION_V2_BLOCK_SIZE_INVALID")
    mins = triangles.min(axis=1)
    maxs = triangles.max(axis=1)
    n = len(triangles)
    left_rows: list[np.ndarray] = []
    right_rows: list[np.ndarray] = []
    for start in range(0, n, block_size):
        end = min(n, start + block_size)
        overlap = np.all(
            (maxs[start:end, None, :] >= mins[None, :, :] - float(tolerance))
            & (maxs[None, :, :] >= mins[start:end, None, :] - float(tolerance)),
            axis=2,
        )
        local_left, right = np.nonzero(overlap)
        left = local_left + start
        keep = left < right
        if np.any(keep):
            left_rows.append(left[keep].astype(np.int32, copy=False))
            right_rows.append(right[keep].astype(np.int32, copy=False))
    if not left_rows:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)
    return np.concatenate(left_rows), np.concatenate(right_rows)


def _unexpected_intersection_pairs_v2(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    tolerance: float,
    shared_contact_exclusion_fraction: float,
    block_size: int = 256,
) -> tuple[tuple[int, int], ...]:
    xyz = np.asarray(vertices, dtype=np.float64)
    tri_index = np.asarray(faces, dtype=np.int64)
    triangles = xyz[tri_index]
    left, right = _aabb_candidate_pairs_blockwise(
        triangles,
        tolerance=float(tolerance),
        block_size=block_size,
    )
    if not len(left):
        return ()

    hit = _sat_intersects_batch(
        triangles[left],
        triangles[right],
        tolerance=float(tolerance),
    )
    left = left[hit]
    right = right[hit]
    if not len(left):
        return ()

    shared = np.any(
        tri_index[left, :, None] == tri_index[right, None, :],
        axis=(1, 2),
    )
    rows: list[np.ndarray] = []
    if np.any(~shared):
        rows.append(np.stack((left[~shared], right[~shared]), axis=1))

    if np.any(shared):
        a = triangles[left[shared]]
        b = triangles[right[shared]]
        ca = np.mean(a, axis=1, keepdims=True)
        cb = np.mean(b, axis=1, keepdims=True)
        scale = 1.0 - float(shared_contact_exclusion_fraction)
        inner_hit = _sat_intersects_batch(
            ca + scale * (a - ca),
            cb + scale * (b - cb),
            tolerance=float(tolerance),
        )
        if np.any(inner_hit):
            rows.append(
                np.stack(
                    (left[shared][inner_hit], right[shared][inner_hit]),
                    axis=1,
                )
            )

    if not rows:
        return ()
    pairs = np.concatenate(rows, axis=0).astype(np.int64, copy=False)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    canonical = pairs[order]
    return tuple((int(a), int(b)) for a, b in canonical.tolist())


def qualify_static_self_intersection_v2(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    policy: StaticSelfIntersectionPolicyV1,
    block_size: int = 256,
) -> StaticSelfIntersectionQualificationV1:
    """Fast exact-equivalent version of the scale-aware v1 qualification court."""
    policy.validate()
    xyz = np.asarray(vertices, dtype=np.float64)
    tri = np.asarray(faces, dtype=np.int64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("STATIC_SELF_INTERSECTION_V2_VERTICES_INVALID")
    if tri.ndim != 2 or tri.shape[1] != 3 or not len(tri) or np.any(tri < 0) or np.any(tri >= len(xyz)):
        raise ValueError("STATIC_SELF_INTERSECTION_V2_FACES_INVALID")

    minimum_altitude = _minimum_triangle_altitude(xyz, tri)
    shrink_clearance = minimum_altitude * float(policy.shared_contact_exclusion_fraction)
    maximum_tolerance = shrink_clearance * float(policy.tolerance_fraction_of_shrink_clearance)
    tolerances = tuple(float(multiplier) * maximum_tolerance for multiplier in policy.sweep_multipliers)
    pair_sets = tuple(
        _unexpected_intersection_pairs_v2(
            vertices=xyz,
            faces=tri,
            tolerance=tolerance,
            shared_contact_exclusion_fraction=policy.shared_contact_exclusion_fraction,
            block_size=block_size,
        )
        for tolerance in tolerances
    )
    stable = all(rows == pair_sets[0] for rows in pair_sets[1:])
    result = StaticSelfIntersectionQualificationV1(
        minimum_triangle_altitude=minimum_altitude,
        maximum_test_tolerance=maximum_tolerance,
        tested_tolerances=tolerances,
        pair_sets=pair_sets,
        stable_across_tolerance_sweep=stable,
        self_intersection_free=bool(stable and len(pair_sets[0]) == 0),
        qualified=bool(stable),
        provenance=(
            f"{policy.provenance};{BROADPHASE_AUTHORITY};{NARROWPHASE_AUTHORITY};"
            f"parent_policy_schema={V1_SCHEMA_VERSION}"
        ),
    )
    result.validate()
    return result


__all__ = [
    "BROADPHASE_AUTHORITY",
    "NARROWPHASE_AUTHORITY",
    "SCHEMA_VERSION",
    "qualify_static_self_intersection_v2",
]
