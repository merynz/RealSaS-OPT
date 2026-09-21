from __future__ import annotations

"""Subject-free dynamic mesh integrity primitives for RealSaS V2."""

import numpy as np

from .types import QualificationError


def interior_shared_edge_projection_continuity(
    *,
    faces: np.ndarray,
    projected_face_vertices: np.ndarray,
) -> dict:
    """Prove exact screen-space agreement on topology-owned interior edges.

    Only an edge referenced by exactly two faces through the same canonical vertex
    indices is continuity authority. Geometrically nearby but topologically separate
    faces/components are deliberately outside this census, so legitimate articulated
    negative space cannot be misclassified as a crack.
    """
    tri = np.asarray(faces, dtype=np.int64)
    projected = np.asarray(projected_face_vertices, dtype=np.float64)
    if (
        tri.ndim != 2
        or tri.shape[1] != 3
        or projected.shape != (len(tri), 3, 2)
        or np.any(tri < 0)
        or not np.isfinite(projected).all()
    ):
        raise QualificationError(
            "DYNAMIC_GEOMETRY_SHARED_EDGE_PROJECTION_INPUT_INVALID"
        )

    incidence: dict[tuple[int, int], list[tuple[int, dict[int, np.ndarray]]]] = {}
    for face_index, face in enumerate(tri):
        for ia, ib in ((0, 1), (1, 2), (2, 0)):
            a = int(face[ia])
            b = int(face[ib])
            key = (min(a, b), max(a, b))
            incidence.setdefault(key, []).append(
                (
                    int(face_index),
                    {
                        a: projected[face_index, ia].copy(),
                        b: projected[face_index, ib].copy(),
                    },
                )
            )

    interior = 0
    mismatch = 0
    max_error = 0.0
    worst_edge = None
    nonmanifold_excluded = 0
    for edge, rows in sorted(incidence.items()):
        if len(rows) > 2:
            nonmanifold_excluded += 1
            continue
        if len(rows) != 2:
            continue
        interior += 1
        left = rows[0][1]
        right = rows[1][1]
        endpoint_error = 0.0
        exact = True
        for vertex_index in edge:
            if vertex_index not in left or vertex_index not in right:
                raise QualificationError(
                    "DYNAMIC_GEOMETRY_SHARED_EDGE_ENDPOINT_BINDING_DRIFT"
                )
            a = np.asarray(left[vertex_index], dtype=np.float64)
            b = np.asarray(right[vertex_index], dtype=np.float64)
            exact = exact and bool(np.array_equal(a, b))
            endpoint_error = max(
                endpoint_error,
                float(np.max(np.abs(a - b))),
            )
        if not exact:
            mismatch += 1
            if endpoint_error >= max_error:
                max_error = endpoint_error
                worst_edge = [int(edge[0]), int(edge[1])]

    return {
        "mode": "TOPOLOGY_OWNED_INTERIOR_SHARED_EDGE_EXACT_PROJECTION_V1",
        "interior_shared_edge_count": int(interior),
        "mismatched_interior_shared_edge_count": int(mismatch),
        "maximum_projected_endpoint_error_px": float(max_error),
        "worst_edge_vertex_indices": worst_edge,
        "nonmanifold_edge_count_excluded_from_continuity_authority": int(
            nonmanifold_excluded
        ),
        "passed": bool(mismatch == 0),
        "cross_component_or_geometrically_near_edges_inferred": False,
    }


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


INTERSECTION_PERSISTENCE_BINARY_STEPS = 24
INTERSECTION_PERSISTENCE_MIN_SCALE = 2.0 ** -12
INTERSECTION_PERSISTENCE_NUMERICAL_SLACK = 4.0 * (2.0 ** -24)


def triangle_intersection_persistence_severity(
    triangle_a: np.ndarray,
    triangle_b: np.ndarray,
    *,
    binary_steps: int = INTERSECTION_PERSISTENCE_BINARY_STEPS,
    minimum_scale: float = INTERSECTION_PERSISTENCE_MIN_SCALE,
) -> float:
    """Return normalized interior-overlap persistence under homothetic shrink.

    Both triangles are shrunk about their own centroids by the same scale. Since
    each shrunken triangle is nested inside its larger-scale version, intersection
    is monotone in scale and the first intersecting scale can be found by binary
    search. Severity is 1 - critical_scale: shallow overlap tends toward zero,
    while deep overlap survives stronger shrink and produces a larger value.

    This construction is invariant to a shared rigid transform and uniform scale;
    it is therefore suitable for rest->posed non-regression rather than an
    arbitrary world-distance penetration threshold.
    """
    a = np.asarray(triangle_a, dtype=np.float64)
    b = np.asarray(triangle_b, dtype=np.float64)
    steps = int(binary_steps)
    floor = float(minimum_scale)
    if (
        a.shape != (3, 3)
        or b.shape != (3, 3)
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or steps < 8
        or steps > 64
        or not np.isfinite(floor)
        or not (0.0 < floor < 0.1)
    ):
        raise QualificationError("DYNAMIC_GEOMETRY_INTERSECTION_SEVERITY_INPUT_INVALID")

    ca = np.mean(a, axis=0)
    cb = np.mean(b, axis=0)

    def intersects(scale: float) -> bool:
        aa = ca + float(scale) * (a - ca)
        bb = cb + float(scale) * (b - cb)
        return triangles_intersect_sat(aa, bb, tolerance=0.0)

    if not intersects(1.0):
        return 0.0
    if intersects(floor):
        return 1.0 - floor

    lo = floor
    hi = 1.0
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if intersects(mid):
            hi = mid
        else:
            lo = mid
    severity = 1.0 - hi
    if not np.isfinite(severity) or severity < -1.0e-12 or severity > 1.0:
        raise QualificationError("DYNAMIC_GEOMETRY_INTERSECTION_SEVERITY_INVALID")
    return float(max(0.0, min(1.0, severity)))


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
