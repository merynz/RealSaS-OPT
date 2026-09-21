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


def unexpected_intersection_pairs(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    tolerance: float = 1.0e-9,
    shared_contact_exclusion_fraction: float = 1.0e-6,
) -> tuple[tuple[int, int], ...]:
    """Return triangle pairs whose contact exceeds declared mesh adjacency.

    Ordinary shared-vertex/shared-edge contact is topologically expected and must
    not be reported as self-intersection.  For pairs sharing vertices, both
    triangles are contracted infinitesimally toward their centroids; expected
    boundary-only contact disappears, while any non-zero interior overlap
    remains.  Pairs with no shared vertices use the exact SAT census.
    """
    xyz = np.asarray(vertices, dtype=np.float64)
    tri = np.asarray(faces, dtype=np.int64)
    fraction = float(shared_contact_exclusion_fraction)
    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
        or not np.isfinite(xyz).all()
        or tri.ndim != 2
        or tri.shape[1] != 3
        or np.any(tri < 0)
        or np.any(tri >= len(xyz))
    ):
        raise QualificationError("DYNAMIC_GEOMETRY_TOPOLOGY_CENSUS_INVALID")
    if not np.isfinite(fraction) or not (0.0 < fraction < 0.01):
        raise QualificationError(
            "DYNAMIC_GEOMETRY_SHARED_CONTACT_EXCLUSION_INVALID"
        )

    triangles = xyz[tri]
    mins = triangles.min(axis=1)
    maxs = triangles.max(axis=1)
    order = np.argsort(mins[:, 0], kind="mergesort")
    active: list[int] = []
    pairs: list[tuple[int, int]] = []

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
            if (
                float(maxs[i, 1]) < float(mins[j, 1]) - tolerance
                or float(maxs[j, 1]) < float(mins[i, 1]) - tolerance
                or float(maxs[i, 2]) < float(mins[j, 2]) - tolerance
                or float(maxs[j, 2]) < float(mins[i, 2]) - tolerance
            ):
                continue
            a = triangles[i]
            b = triangles[j]
            if not triangles_intersect_sat(
                a,
                b,
                tolerance=float(tolerance),
            ):
                continue

            shared = face_i.intersection(map(int, tri[j]))
            if not shared:
                pairs.append((min(i, j), max(i, j)))
                continue

            # Remove only the expected boundary/simplex contact.  A real interior
            # penetration survives this contraction and is still detected by SAT.
            a_centroid = np.mean(a, axis=0)
            b_centroid = np.mean(b, axis=0)
            scale = 1.0 - fraction
            a_inner = a_centroid + scale * (a - a_centroid)
            b_inner = b_centroid + scale * (b - b_centroid)
            if triangles_intersect_sat(
                a_inner,
                b_inner,
                tolerance=float(tolerance),
            ):
                pairs.append((min(i, j), max(i, j)))

        active.append(i)

    return tuple(sorted(set(pairs)))


def projected_orientation_flip(
    *,
    rest_screen_triangle: np.ndarray,
    posed_screen_triangle: np.ndarray,
    min_projected_double_area_px2: float,
) -> bool:
    rest = np.asarray(rest_screen_triangle, dtype=np.float64)
    posed = np.asarray(posed_screen_triangle, dtype=np.float64)
    minimum = float(min_projected_double_area_px2)
    if (
        rest.shape != (3, 2)
        or posed.shape != (3, 2)
        or not np.isfinite(rest).all()
        or not np.isfinite(posed).all()
        or not np.isfinite(minimum)
        or minimum <= 0.0
    ):
        raise QualificationError("DYNAMIC_GEOMETRY_ORIENTATION_INPUT_INVALID")

    def signed_area2(triangle: np.ndarray) -> float:
        return float(
            (triangle[1, 0] - triangle[0, 0])
            * (triangle[2, 1] - triangle[0, 1])
            - (triangle[1, 1] - triangle[0, 1])
            * (triangle[2, 0] - triangle[0, 0])
        )

    rest_area2 = signed_area2(rest)
    posed_area2 = signed_area2(posed)
    if abs(rest_area2) < minimum or abs(posed_area2) < minimum:
        return False
    return bool(rest_area2 * posed_area2 < 0.0)


def dynamic_visibility_load_gate(
    *,
    maximum_frame_micro_visible_pixel_fraction: float,
    unmeasurable_consequential_visible_face_count: int,
    max_micro_visible_pixel_fraction_per_frame: float,
    max_unmeasurable_consequential_visible_face_count: int,
) -> dict:
    micro = float(maximum_frame_micro_visible_pixel_fraction)
    unmeasurable = int(unmeasurable_consequential_visible_face_count)
    micro_limit = float(max_micro_visible_pixel_fraction_per_frame)
    unmeasurable_limit = int(max_unmeasurable_consequential_visible_face_count)
    if (
        not np.isfinite(micro)
        or not np.isfinite(micro_limit)
        or micro < 0.0
        or micro_limit < 0.0
        or unmeasurable < 0
        or unmeasurable_limit < 0
    ):
        raise QualificationError("DYNAMIC_GEOMETRY_VISIBILITY_LOAD_INPUT_INVALID")
    micro_passed = micro <= micro_limit
    unmeasurable_passed = unmeasurable <= unmeasurable_limit
    return {
        "micro_visible_face_load_passed": bool(micro_passed),
        "unmeasurable_consequential_face_load_passed": bool(
            unmeasurable_passed
        ),
        "passed": bool(micro_passed and unmeasurable_passed),
    }
