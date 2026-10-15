from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from math import isfinite
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

from tools.corpus.fit8_surface_topology_invariants_v1 import SurfaceTopologyReportV1

SCHEMA_VERSION = "RealSaS.MeshFieldTeacher.v1"
NEGATIVE_INSIDE = "NEGATIVE_INSIDE"

DEFAULT_RAY_DIRECTIONS = (
    (1.0, 0.317, 0.581),
    (-0.431, 1.0, 0.229),
    (0.197, -0.647, 1.0),
    (-1.0, 0.283, 0.419),
    (0.337, -1.0, 0.719),
)


@dataclass(frozen=True)
class MeshFieldTeacherPolicyV1:
    spatial_tolerance: float
    minimum_triangle_area2: float
    distance_tie_tolerance: float
    ray_parallel_sine_epsilon: float = 1.0e-12
    barycentric_boundary_epsilon: float = 1.0e-10
    minimum_nonambiguous_votes: int = 3
    require_unanimous_votes: bool = True
    bvh_leaf_size: int = 8
    query_batch_size: int = 512
    ray_directions: tuple[tuple[float, float, float], ...] = DEFAULT_RAY_DIRECTIONS
    provenance: str = ""
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("MESH_FIELD_TEACHER_POLICY_SCHEMA_INVALID")
        for name in ("spatial_tolerance", "minimum_triangle_area2", "distance_tie_tolerance"):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"MESH_FIELD_TEACHER_POLICY_INVALID:{name}")
        if not isfinite(self.ray_parallel_sine_epsilon) or self.ray_parallel_sine_epsilon <= 0.0:
            raise ValueError("MESH_FIELD_TEACHER_RAY_PARALLEL_EPS_INVALID")
        if not isfinite(self.barycentric_boundary_epsilon) or not (0.0 < self.barycentric_boundary_epsilon < 0.5):
            raise ValueError("MESH_FIELD_TEACHER_BARY_EPS_INVALID")
        if self.minimum_nonambiguous_votes < 1:
            raise ValueError("MESH_FIELD_TEACHER_VOTE_COUNT_INVALID")
        if self.bvh_leaf_size < 1 or self.query_batch_size < 1:
            raise ValueError("MESH_FIELD_TEACHER_BATCH_POLICY_INVALID")
        if len(self.ray_directions) < self.minimum_nonambiguous_votes:
            raise ValueError("MESH_FIELD_TEACHER_RAY_COUNT_LT_REQUIRED_VOTES")
        normalized = []
        for row in self.ray_directions:
            value = np.asarray(row, dtype=np.float64)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError("MESH_FIELD_TEACHER_RAY_DIRECTION_INVALID")
            norm = float(np.linalg.norm(value))
            if norm <= 0.0 or np.any(np.abs(value / norm) <= 1.0e-15):
                raise ValueError("MESH_FIELD_TEACHER_RAY_DIRECTION_DEGENERATE")
            normalized.append(tuple((value / norm).tolist()))
        if len(set(normalized)) != len(normalized):
            raise ValueError("MESH_FIELD_TEACHER_RAY_DIRECTION_DUPLICATE")
        if not self.provenance:
            raise ValueError("MESH_FIELD_TEACHER_POLICY_PROVENANCE_MISSING")

    def sha256(self) -> str:
        self.validate()
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class GlobalSignAdmissionV1:
    admitted: bool
    reasons: tuple[str, ...]
    source_component_count: int
    self_intersection_free: bool
    containment_semantics_qualified: bool
    sign_convention: str
    provenance: str
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("MESH_FIELD_TEACHER_SIGN_ADMISSION_SCHEMA_INVALID")
        if self.source_component_count < 1:
            raise ValueError("MESH_FIELD_TEACHER_SIGN_COMPONENT_COUNT_INVALID")
        if self.sign_convention != NEGATIVE_INSIDE:
            raise ValueError("MESH_FIELD_TEACHER_SIGN_CONVENTION_INVALID")
        if tuple(sorted(set(self.reasons))) != self.reasons:
            raise ValueError("MESH_FIELD_TEACHER_SIGN_REASONS_NOT_CANONICAL")
        if self.admitted != (len(self.reasons) == 0):
            raise ValueError("MESH_FIELD_TEACHER_SIGN_ADMISSION_REASON_MISMATCH")
        if not self.provenance:
            raise ValueError("MESH_FIELD_TEACHER_SIGN_PROVENANCE_MISSING")


def qualify_single_component_global_sign_v1(
    topology: SurfaceTopologyReportV1,
    *,
    self_intersection_free: bool,
    containment_semantics_qualified: bool,
    provenance: str,
) -> GlobalSignAdmissionV1:
    """Admit the first global signed-field court only on one qualified closed component.

    Multi-component nesting/union semantics are deliberately deferred. Open,
    non-manifold or non-orientable surfaces still receive exact unsigned-distance
    truth from this module, but their global sign remains invalid.
    """
    topology.validate()
    reasons: list[str] = []
    if topology.component_count != 1:
        reasons.append("MULTI_COMPONENT_COMPOSITION_UNQUALIFIED")
    else:
        component = topology.components[0]
        if component.boundary_edge_count != 0:
            reasons.append("OPEN_BOUNDARY")
        if component.nonmanifold_edge_count != 0:
            reasons.append("NONMANIFOLD_INCIDENCE")
        if not component.orientability_valid or component.orientable is not True:
            reasons.append("ORIENTABILITY_UNQUALIFIED")
    if not self_intersection_free:
        reasons.append("SELF_INTERSECTION_UNQUALIFIED")
    if not containment_semantics_qualified:
        reasons.append("CONTAINMENT_SEMANTICS_UNQUALIFIED")
    result = GlobalSignAdmissionV1(
        admitted=not reasons,
        reasons=tuple(sorted(set(reasons))),
        source_component_count=topology.component_count,
        self_intersection_free=bool(self_intersection_free),
        containment_semantics_qualified=bool(containment_semantics_qualified),
        sign_convention=NEGATIVE_INSIDE,
        provenance=provenance,
    )
    result.validate()
    return result


@dataclass(frozen=True)
class MeshFieldTeacherResultV1:
    query_xyz: np.ndarray
    unsigned_distance: np.ndarray
    nearest_triangle_id: np.ndarray
    nearest_barycentric: np.ndarray
    nearest_address_tie_count: np.ndarray
    oriented_level_set: np.ndarray
    orientation_valid: np.ndarray
    inside: np.ndarray
    nonambiguous_vote_count: np.ndarray
    ambiguous_ray_count: np.ndarray
    policy_sha256: str
    sign_admission: GlobalSignAdmissionV1
    provenance: str
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("MESH_FIELD_TEACHER_RESULT_SCHEMA_INVALID")
        self.sign_admission.validate()
        q = np.asarray(self.query_xyz)
        n = len(q)
        if q.ndim != 2 or q.shape[1] != 3 or not np.isfinite(q).all():
            raise ValueError("MESH_FIELD_TEACHER_QUERY_INVALID")
        for name in ("unsigned_distance", "nearest_triangle_id", "nearest_address_tie_count", "oriented_level_set", "orientation_valid", "inside", "nonambiguous_vote_count", "ambiguous_ray_count"):
            value = np.asarray(getattr(self, name))
            if value.shape != (n,):
                raise ValueError(f"MESH_FIELD_TEACHER_RESULT_SHAPE_INVALID:{name}")
        bary = np.asarray(self.nearest_barycentric)
        if bary.shape != (n, 3):
            raise ValueError("MESH_FIELD_TEACHER_BARY_SHAPE_INVALID")
        if not np.isfinite(self.unsigned_distance).all() or np.any(self.unsigned_distance < 0.0):
            raise ValueError("MESH_FIELD_TEACHER_DISTANCE_INVALID")
        if np.any(self.nearest_triangle_id < 0) or np.any(self.nearest_address_tie_count < 1):
            raise ValueError("MESH_FIELD_TEACHER_ADDRESS_INVALID")
        if not np.isfinite(bary).all() or np.any(bary < -1.0e-9) or np.any(bary > 1.0 + 1.0e-9):
            raise ValueError("MESH_FIELD_TEACHER_BARY_INVALID")
        if np.max(np.abs(bary.sum(axis=1) - 1.0), initial=0.0) > 1.0e-8:
            raise ValueError("MESH_FIELD_TEACHER_BARY_SUM_INVALID")
        valid = np.asarray(self.orientation_valid, dtype=bool)
        oriented = np.asarray(self.oriented_level_set)
        if np.any(~np.isfinite(oriented[valid])):
            raise ValueError("MESH_FIELD_TEACHER_VALID_ORIENTED_NONFINITE")
        if np.any(np.isfinite(oriented[~valid])):
            raise ValueError("MESH_FIELD_TEACHER_INVALID_ORIENTED_MUST_BE_NAN")
        if len(self.policy_sha256) != 64:
            raise ValueError("MESH_FIELD_TEACHER_POLICY_SHA_INVALID")
        if not self.provenance:
            raise ValueError("MESH_FIELD_TEACHER_RESULT_PROVENANCE_MISSING")


class _BVHNode:
    __slots__ = ("lo", "hi", "left", "right", "triangle_ids")

    def __init__(self, ids: np.ndarray, tri_lo: np.ndarray, tri_hi: np.ndarray, centroids: np.ndarray, leaf_size: int):
        self.lo = tri_lo[ids].min(axis=0)
        self.hi = tri_hi[ids].max(axis=0)
        self.left = None
        self.right = None
        self.triangle_ids = None
        if len(ids) <= leaf_size:
            self.triangle_ids = np.asarray(ids, dtype=np.int32)
            return
        span = np.ptp(centroids[ids], axis=0)
        axis = int(np.argmax(span))
        order = ids[np.argsort(centroids[ids, axis], kind="mergesort")]
        middle = len(order) // 2
        self.left = _BVHNode(order[:middle], tri_lo, tri_hi, centroids, leaf_size)
        self.right = _BVHNode(order[middle:], tri_lo, tri_hi, centroids, leaf_size)


def _closest_points_barycentric_batch(points: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized Ericson closest-point regions for [Q] points x [K] triangles."""
    p = points[:, None, :]
    a = triangles[None, :, 0, :]
    b = triangles[None, :, 1, :]
    c = triangles[None, :, 2, :]
    ab, ac, ap = b - a, c - a, p - a
    d1 = np.sum(ab * ap, axis=2)
    d2 = np.sum(ac * ap, axis=2)
    bp = p - b
    d3 = np.sum(ab * bp, axis=2)
    d4 = np.sum(ac * bp, axis=2)
    cpv = p - c
    d5 = np.sum(ab * cpv, axis=2)
    d6 = np.sum(ac * cpv, axis=2)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    shape = d1.shape
    bary = np.zeros(shape + (3,), dtype=np.float64)
    assigned = np.zeros(shape, dtype=bool)

    def assign(mask: np.ndarray, values: tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float]) -> None:
        active = mask & ~assigned
        if not np.any(active):
            return
        for i, value in enumerate(values):
            array = np.broadcast_to(value, shape) if np.ndim(value) == 0 else value
            bary[..., i][active] = array[active]
        assigned[active] = True

    assign((d1 <= 0.0) & (d2 <= 0.0), (1.0, 0.0, 0.0))
    assign((d3 >= 0.0) & (d4 <= d3), (0.0, 1.0, 0.0))
    denom = d1 - d3
    v = np.divide(d1, denom, out=np.zeros_like(d1), where=np.abs(denom) > 0.0)
    assign((vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0), (1.0 - v, v, 0.0))
    assign((d6 >= 0.0) & (d5 <= d6), (0.0, 0.0, 1.0))
    denom = d2 - d6
    w = np.divide(d2, denom, out=np.zeros_like(d2), where=np.abs(denom) > 0.0)
    assign((vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0), (1.0 - w, 0.0, w))
    denom = (d4 - d3) + (d5 - d6)
    wbc = np.divide(d4 - d3, denom, out=np.zeros_like(d1), where=np.abs(denom) > 0.0)
    assign((va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0), (0.0, 1.0 - wbc, wbc))

    remaining = ~assigned
    denom = va + vb + vc
    vface = np.divide(vb, denom, out=np.zeros_like(vb), where=np.abs(denom) > 0.0)
    wface = np.divide(vc, denom, out=np.zeros_like(vc), where=np.abs(denom) > 0.0)
    bary[..., 0][remaining] = (1.0 - vface - wface)[remaining]
    bary[..., 1][remaining] = vface[remaining]
    bary[..., 2][remaining] = wface[remaining]

    closest = (
        bary[..., 0, None] * a
        + bary[..., 1, None] * b
        + bary[..., 2, None] * c
    )
    return closest, bary


def _bbox_distance2(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    delta = np.maximum(np.maximum(lo[None, :] - points, points - hi[None, :]), 0.0)
    return np.sum(delta * delta, axis=1)


def _exact_closest_surface(
    queries: np.ndarray,
    triangles: np.ndarray,
    *,
    policy: MeshFieldTeacherPolicyV1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tri_lo = triangles.min(axis=1)
    tri_hi = triangles.max(axis=1)
    centroids = triangles.mean(axis=1)
    ids = np.arange(len(triangles), dtype=np.int32)
    root = _BVHNode(ids, tri_lo, tri_hi, centroids, policy.bvh_leaf_size)
    centroid_tree = cKDTree(centroids)

    distance = np.empty(len(queries), dtype=np.float64)
    triangle_id = np.empty(len(queries), dtype=np.int32)
    barycentric = np.empty((len(queries), 3), dtype=np.float64)
    tie_count = np.empty(len(queries), dtype=np.int32)
    eps = float(policy.distance_tie_tolerance)

    for start in range(0, len(queries), policy.query_batch_size):
        points = queries[start : start + policy.query_batch_size]
        count = len(points)
        _, seed_ids = centroid_tree.query(points, k=1)
        seed_tri = triangles[np.asarray(seed_ids, dtype=np.int32)]
        cp, bary = _closest_points_barycentric_batch(points, seed_tri)
        cp = cp[:, 0, :]
        bary = bary[:, 0, :]
        best_d2 = np.sum((cp - points) ** 2, axis=1)
        best_id = np.asarray(seed_ids, dtype=np.int32).copy()
        best_bary = bary.copy()
        best_ties = np.ones(count, dtype=np.int32)

        stack: list[tuple[_BVHNode, np.ndarray]] = [(root, np.arange(count, dtype=np.int32))]
        while stack:
            node, query_ids = stack.pop()
            if not len(query_ids):
                continue
            best_distance = np.sqrt(best_d2[query_ids])
            lower2 = _bbox_distance2(points[query_ids], node.lo, node.hi)
            keep = query_ids[lower2 <= (best_distance + eps) ** 2]
            if not len(keep):
                continue
            if node.triangle_ids is None:
                stack.append((node.left, keep))
                stack.append((node.right, keep))
                continue

            leaf_ids = node.triangle_ids
            closest, leaf_bary = _closest_points_barycentric_batch(points[keep], triangles[leaf_ids])
            d2 = np.sum((closest - points[keep, None, :]) ** 2, axis=2)
            local_min_index = np.argmin(d2, axis=1)
            local_min2 = d2[np.arange(len(keep)), local_min_index]
            local_min_distance = np.sqrt(local_min2)
            current_distance = np.sqrt(best_d2[keep])

            for local_row, query_id in enumerate(keep.tolist()):
                local_distance = float(local_min_distance[local_row])
                current = float(current_distance[local_row])
                if local_distance < current - eps:
                    tied = np.flatnonzero(np.abs(np.sqrt(d2[local_row]) - local_distance) <= eps)
                    tied_ids = leaf_ids[tied]
                    chosen_pos = int(tied[np.argmin(tied_ids)])
                    best_d2[query_id] = d2[local_row, chosen_pos]
                    best_id[query_id] = int(leaf_ids[chosen_pos])
                    best_bary[query_id] = leaf_bary[local_row, chosen_pos]
                    best_ties[query_id] = int(len(tied))
                elif abs(local_distance - current) <= eps:
                    tied = np.flatnonzero(np.abs(np.sqrt(d2[local_row]) - current) <= eps)
                    if len(tied):
                        best_ties[query_id] += int(len(tied))
                        tied_ids = leaf_ids[tied]
                        chosen_pos = int(tied[np.argmin(tied_ids)])
                        if int(leaf_ids[chosen_pos]) < int(best_id[query_id]):
                            best_id[query_id] = int(leaf_ids[chosen_pos])
                            best_bary[query_id] = leaf_bary[local_row, chosen_pos]

        distance[start : start + count] = np.sqrt(best_d2)
        triangle_id[start : start + count] = best_id
        barycentric[start : start + count] = best_bary
        tie_count[start : start + count] = best_ties

    return distance, triangle_id, barycentric, tie_count


def _ray_aabb_mask(points: np.ndarray, direction: np.ndarray, node: _BVHNode, spatial_tolerance: float) -> np.ndarray:
    inverse = 1.0 / direction
    t0 = (node.lo[None, :] - points) * inverse[None, :]
    t1 = (node.hi[None, :] - points) * inverse[None, :]
    entry = np.max(np.minimum(t0, t1), axis=1)
    exit_ = np.min(np.maximum(t0, t1), axis=1)
    return exit_ >= np.maximum(entry, spatial_tolerance)


def _classify_ray_direction(
    points: np.ndarray,
    direction: np.ndarray,
    triangles: np.ndarray,
    root: _BVHNode,
    policy: MeshFieldTeacherPolicyV1,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(points)
    hit_lists: list[list[float]] = [[] for _ in range(n)]
    ambiguous = np.zeros(n, dtype=bool)
    stack: list[tuple[_BVHNode, np.ndarray]] = [(root, np.arange(n, dtype=np.int32))]
    spatial = float(policy.spatial_tolerance)
    bary_eps = float(policy.barycentric_boundary_epsilon)
    angular_eps = float(policy.ray_parallel_sine_epsilon)

    while stack:
        node, query_ids = stack.pop()
        if not len(query_ids):
            continue
        keep = query_ids[_ray_aabb_mask(points[query_ids], direction, node, spatial)]
        if not len(keep):
            continue
        if node.triangle_ids is None:
            stack.append((node.left, keep))
            stack.append((node.right, keep))
            continue

        ids = node.triangle_ids
        tri = triangles[ids]
        a = tri[:, 0]
        e1 = tri[:, 1] - a
        e2 = tri[:, 2] - a
        normal = np.cross(e1, e2)
        area2 = np.linalg.norm(normal, axis=1)
        h = np.cross(np.broadcast_to(direction, e2.shape), e2)
        det = np.sum(e1 * h, axis=1)
        nonparallel = np.abs(det) > angular_eps * area2
        if not np.any(nonparallel):
            continue
        a = a[nonparallel]
        e1 = e1[nonparallel]
        e2 = e2[nonparallel]
        h = h[nonparallel]
        det = det[nonparallel]

        s = points[keep, None, :] - a[None, :, :]
        u = np.sum(s * h[None, :, :], axis=2) / det[None, :]
        q = np.cross(s, e1[None, :, :])
        v = np.sum(direction[None, None, :] * q, axis=2) / det[None, :]
        t = np.sum(e2[None, :, :] * q, axis=2) / det[None, :]
        w = 1.0 - u - v
        hit = (t > spatial) & (u >= -bary_eps) & (v >= -bary_eps) & (w >= -bary_eps)
        boundary = hit & ((u <= bary_eps) | (v <= bary_eps) | (w <= bary_eps))
        ambiguous[keep] |= np.any(boundary, axis=1)
        strict = hit & ~boundary
        for local_q, local_t in zip(*np.nonzero(strict)):
            hit_lists[int(keep[local_q])].append(float(t[local_q, local_t]))

    inside = np.zeros(n, dtype=bool)
    for query_id, hits in enumerate(hit_lists):
        if ambiguous[query_id]:
            continue
        hits.sort()
        if any((b - a) <= spatial for a, b in zip(hits[:-1], hits[1:])):
            ambiguous[query_id] = True
            continue
        inside[query_id] = (len(hits) % 2) == 1
    return inside, ambiguous


def build_mesh_field_teacher_v1(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    query_xyz: np.ndarray,
    sign_admission: GlobalSignAdmissionV1,
    policy: MeshFieldTeacherPolicyV1,
    provenance: str,
) -> MeshFieldTeacherResultV1:
    policy.validate()
    sign_admission.validate()
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    queries = np.asarray(query_xyz, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError("MESH_FIELD_TEACHER_VERTICES_INVALID")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("MESH_FIELD_TEACHER_FACES_INVALID")
    if queries.ndim != 2 or queries.shape[1] != 3 or not np.isfinite(queries).all():
        raise ValueError("MESH_FIELD_TEACHER_QUERIES_INVALID")
    triangles = vertices[faces]
    area2 = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    if np.any(area2 <= policy.minimum_triangle_area2):
        raise ValueError("MESH_FIELD_TEACHER_DEGENERATE_SOURCE_TRIANGLE")

    unsigned, triangle_id, barycentric, tie_count = _exact_closest_surface(queries, triangles, policy=policy)
    oriented = np.full(len(queries), np.nan, dtype=np.float64)
    orientation_valid = np.zeros(len(queries), dtype=bool)
    inside = np.zeros(len(queries), dtype=bool)
    nonambiguous_votes = np.zeros(len(queries), dtype=np.int16)
    ambiguous_count = np.zeros(len(queries), dtype=np.int16)

    if sign_admission.admitted and len(queries):
        tri_lo = triangles.min(axis=1)
        tri_hi = triangles.max(axis=1)
        centroids = triangles.mean(axis=1)
        root = _BVHNode(np.arange(len(triangles), dtype=np.int32), tri_lo, tri_hi, centroids, policy.bvh_leaf_size)
        directions = []
        for row in policy.ray_directions:
            value = np.asarray(row, dtype=np.float64)
            directions.append(value / np.linalg.norm(value))

        on_surface = unsigned <= policy.spatial_tolerance
        oriented[on_surface] = 0.0
        orientation_valid[on_surface] = True

        off_ids = np.flatnonzero(~on_surface)
        for start in range(0, len(off_ids), policy.query_batch_size):
            ids = off_ids[start : start + policy.query_batch_size]
            points = queries[ids]
            votes: list[np.ndarray] = []
            ambiguities: list[np.ndarray] = []
            for direction in directions:
                vote, ambiguous = _classify_ray_direction(points, direction, triangles, root, policy)
                votes.append(vote)
                ambiguities.append(ambiguous)
            vote_matrix = np.stack(votes, axis=1)
            ambiguous_matrix = np.stack(ambiguities, axis=1)
            for local, query_id in enumerate(ids.tolist()):
                good = ~ambiguous_matrix[local]
                values = vote_matrix[local, good]
                nonambiguous_votes[query_id] = int(len(values))
                ambiguous_count[query_id] = int(np.count_nonzero(~good))
                if len(values) < policy.minimum_nonambiguous_votes:
                    continue
                if policy.require_unanimous_votes and not np.all(values == values[0]):
                    continue
                if not policy.require_unanimous_votes:
                    true_count = int(np.count_nonzero(values))
                    false_count = int(len(values) - true_count)
                    if true_count == false_count:
                        continue
                    value = true_count > false_count
                else:
                    value = bool(values[0])
                inside[query_id] = value
                orientation_valid[query_id] = True
                oriented[query_id] = -unsigned[query_id] if value else unsigned[query_id]

    result = MeshFieldTeacherResultV1(
        query_xyz=queries,
        unsigned_distance=unsigned,
        nearest_triangle_id=triangle_id,
        nearest_barycentric=barycentric,
        nearest_address_tie_count=tie_count,
        oriented_level_set=oriented,
        orientation_valid=orientation_valid,
        inside=inside,
        nonambiguous_vote_count=nonambiguous_votes,
        ambiguous_ray_count=ambiguous_count,
        policy_sha256=policy.sha256(),
        sign_admission=sign_admission,
        provenance=provenance,
    )
    result.validate()
    return result


__all__ = [
    "DEFAULT_RAY_DIRECTIONS",
    "GlobalSignAdmissionV1",
    "MeshFieldTeacherPolicyV1",
    "MeshFieldTeacherResultV1",
    "NEGATIVE_INSIDE",
    "SCHEMA_VERSION",
    "build_mesh_field_teacher_v1",
    "qualify_single_component_global_sign_v1",
]
