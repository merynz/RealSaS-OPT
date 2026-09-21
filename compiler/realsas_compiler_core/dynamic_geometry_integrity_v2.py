from __future__ import annotations

"""Subject-free dynamic mesh integrity primitives for RealSaS V2."""

import numpy as np

from .types import QualificationError


def _aabb(triangle: np.ndarray):
    tri = np.asarray(triangle, dtype=np.float64)
    if tri.shape != (3, 3) or not np.isfinite(tri).all():
        raise QualificationError("DYNAMIC_GEOMETRY_TRIANGLE_INVALID")
    return tri.min(axis=0), tri.max(axis=0)


def _separated_on_axis(
    a: np.ndarray,
    b: np.ndarray,
    axis: np.ndarray,
    *,
    tolerance: float,
) -> bool:
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-14:
        return False
    unit = axis / norm
    pa = a @ unit
    pb = b @ unit
    return bool(
        float(np.max(pa)) < float(np.min(pb)) - float(tolerance)
        or float(np.max(pb)) < float(np.min(pa)) - float(tolerance)
    )


def triangles_intersect_sat(
    triangle_a: np.ndarray,
    triangle_b: np.ndarray,
    *,
    tolerance: float = 1.0e-9,
) -> bool:
    a = np.asarray(triangle_a, dtype=np.float64)
    b = np.asarray(triangle_b, dtype=np.float64)
    if a.shape != (3, 3) or b.shape != (3, 3):
        raise QualificationError("DYNAMIC_GEOMETRY_TRIANGLE_SHAPE_INVALID")
    ea = (a[1] - a[0], a[2] - a[1], a[0] - a[2])
    eb = (b[1] - b[0], b[2] - b[1], b[0] - b[2])
    na = np.cross(ea[0], ea[1])
    nb = np.cross(eb[0], eb[1])
    if float(np.linalg.norm(na)) <= 1.0e-14 or float(np.linalg.norm(nb)) <= 1.0e-14:
        raise QualificationError("DYNAMIC_GEOMETRY_DEGENERATE_TRIANGLE")

    axes = [na, nb]
    axes.extend(np.cross(x, y) for x in ea for y in eb)
    # Coplanar separation requires in-plane axes as well.
    axes.extend(np.cross(na, edge) for edge in ea)
    axes.extend(np.cross(nb, edge) for edge in eb)
    for axis in axes:
        if _separated_on_axis(a, b, axis, tolerance=float(tolerance)):
            return False
    return True


def nonadjacent_intersection_pairs(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    tolerance: float = 1.0e-9,
) -> tuple[tuple[int, int], ...]:
    xyz = np.asarray(vertices, dtype=np.float64)
    tri = np.asarray(faces, dtype=np.int64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise QualificationError("DYNAMIC_GEOMETRY_VERTEX_MATRIX_INVALID")
    if tri.ndim != 2 or tri.shape[1] != 3:
        raise QualificationError("DYNAMIC_GEOMETRY_FACE_MATRIX_INVALID")
    if np.any(tri < 0) or np.any(tri >= len(xyz)):
        raise QualificationError("DYNAMIC_GEOMETRY_FACE_INDEX_INVALID")

    triangles = xyz[tri]
    mins = triangles.min(axis=1)
    maxs = triangles.max(axis=1)
    order = np.argsort(mins[:, 0], kind="mergesort")
    active: list[int] = []
    pairs = []
    for raw in order:
        i = int(raw)
        x_min = float(mins[i, 0])
        active = [
            j
            for j in active
            if float(maxs[j, 0]) >= x_min - float(tolerance)
        ]
        face_i = set(map(int, tri[i]))
        for j in active:
            if face_i.intersection(map(int, tri[j])):
                continue
            if (
                float(maxs[i, 1]) < float(mins[j, 1]) - tolerance
                or float(maxs[j, 1]) < float(mins[i, 1]) - tolerance
                or float(maxs[i, 2]) < float(mins[j, 2]) - tolerance
                or float(maxs[j, 2]) < float(mins[i, 2]) - tolerance
            ):
                continue
            if triangles_intersect_sat(
                triangles[i],
                triangles[j],
                tolerance=float(tolerance),
            ):
                pairs.append((min(i, j), max(i, j)))
        active.append(i)
    return tuple(sorted(set(pairs)))
