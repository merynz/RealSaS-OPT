from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from tools.corpus.fit8_mesh_field_teacher_v1 import (
    GlobalSignAdmissionV1,
    MeshFieldTeacherPolicyV1,
    _BVHNode,
    _bbox_distance2,
    _classify_ray_direction,
)

SCHEMA_VERSION = "RealSaS.MeshFieldTeacher.v2"
DISTANCE_AUTHORITY = "EXACT_AABB_BVH_ERICSON_PAIRWISE_SEED_AND_TIE_QUALIFIED"
SIGN_AUTHORITY = "MULTIRAY_PARITY_UNANIMOUS_OR_ABSTAIN"


@dataclass(frozen=True)
class MeshFieldTeacherResultV2:
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
    distance_authority: str
    sign_authority: str
    provenance: str
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("MESH_FIELD_TEACHER_V2_SCHEMA_INVALID")
        self.sign_admission.validate()
        if self.distance_authority != DISTANCE_AUTHORITY:
            raise ValueError("MESH_FIELD_TEACHER_V2_DISTANCE_AUTHORITY_INVALID")
        if self.sign_authority != SIGN_AUTHORITY:
            raise ValueError("MESH_FIELD_TEACHER_V2_SIGN_AUTHORITY_INVALID")
        q = np.asarray(self.query_xyz)
        n = len(q)
        if q.ndim != 2 or q.shape[1] != 3 or not np.isfinite(q).all():
            raise ValueError("MESH_FIELD_TEACHER_V2_QUERY_INVALID")
        for name in (
            "unsigned_distance", "nearest_triangle_id", "nearest_address_tie_count",
            "oriented_level_set", "orientation_valid", "inside",
            "nonambiguous_vote_count", "ambiguous_ray_count",
        ):
            if np.asarray(getattr(self, name)).shape != (n,):
                raise ValueError(f"MESH_FIELD_TEACHER_V2_SHAPE_INVALID:{name}")
        bary = np.asarray(self.nearest_barycentric)
        if bary.shape != (n, 3):
            raise ValueError("MESH_FIELD_TEACHER_V2_BARY_SHAPE_INVALID")
        if not np.isfinite(self.unsigned_distance).all() or np.any(self.unsigned_distance < 0.0):
            raise ValueError("MESH_FIELD_TEACHER_V2_DISTANCE_INVALID")
        if np.any(self.nearest_triangle_id < 0) or np.any(self.nearest_address_tie_count < 1):
            raise ValueError("MESH_FIELD_TEACHER_V2_ADDRESS_INVALID")
        if not np.isfinite(bary).all() or np.max(np.abs(bary.sum(axis=1) - 1.0), initial=0.0) > 1.0e-8:
            raise ValueError("MESH_FIELD_TEACHER_V2_BARY_INVALID")
        if np.any(bary < -1.0e-9) or np.any(bary > 1.0 + 1.0e-9):
            raise ValueError("MESH_FIELD_TEACHER_V2_BARY_RANGE_INVALID")
        valid = np.asarray(self.orientation_valid, dtype=bool)
        oriented = np.asarray(self.oriented_level_set)
        if np.any(~np.isfinite(oriented[valid])):
            raise ValueError("MESH_FIELD_TEACHER_V2_VALID_ORIENTED_NONFINITE")
        if np.any(np.isfinite(oriented[~valid])):
            raise ValueError("MESH_FIELD_TEACHER_V2_INVALID_ORIENTED_MUST_BE_NAN")
        if len(self.policy_sha256) != 64:
            raise ValueError("MESH_FIELD_TEACHER_V2_POLICY_SHA_INVALID")
        if not self.provenance:
            raise ValueError("MESH_FIELD_TEACHER_V2_PROVENANCE_MISSING")


def _closest_point_pairs(points: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ericson closest-point regions for exactly paired [Q] points and [Q] triangles."""
    p = np.asarray(points, dtype=np.float64)
    tri = np.asarray(triangles, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or tri.shape != (len(p), 3, 3):
        raise ValueError("MESH_FIELD_TEACHER_V2_PAIR_SHAPE_INVALID")
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.sum(ab * ap, axis=1), np.sum(ac * ap, axis=1)
    bp = p - b
    d3, d4 = np.sum(ab * bp, axis=1), np.sum(ac * bp, axis=1)
    cpv = p - c
    d5, d6 = np.sum(ab * cpv, axis=1), np.sum(ac * cpv, axis=1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    bary = np.zeros((len(p), 3), dtype=np.float64)
    assigned = np.zeros(len(p), dtype=bool)

    def take(mask: np.ndarray, x: np.ndarray | float, y: np.ndarray | float, z: np.ndarray | float) -> None:
        active = mask & ~assigned
        if not np.any(active):
            return
        for column, value in enumerate((x, y, z)):
            if np.ndim(value) == 0:
                bary[active, column] = float(value)
            else:
                bary[active, column] = np.asarray(value)[active]
        assigned[active] = True

    take((d1 <= 0.0) & (d2 <= 0.0), 1.0, 0.0, 0.0)
    take((d3 >= 0.0) & (d4 <= d3), 0.0, 1.0, 0.0)
    den = d1 - d3
    v = np.divide(d1, den, out=np.zeros_like(d1), where=np.abs(den) > 0.0)
    take((vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0), 1.0 - v, v, 0.0)
    take((d6 >= 0.0) & (d5 <= d6), 0.0, 0.0, 1.0)
    den = d2 - d6
    w = np.divide(d2, den, out=np.zeros_like(d2), where=np.abs(den) > 0.0)
    take((vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0), 1.0 - w, 0.0, w)
    den = (d4 - d3) + (d5 - d6)
    wbc = np.divide(d4 - d3, den, out=np.zeros_like(d1), where=np.abs(den) > 0.0)
    take((va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0), 0.0, 1.0 - wbc, wbc)

    remaining = ~assigned
    den = va + vb + vc
    vface = np.divide(vb, den, out=np.zeros_like(vb), where=np.abs(den) > 0.0)
    wface = np.divide(vc, den, out=np.zeros_like(vc), where=np.abs(den) > 0.0)
    bary[remaining, 0] = (1.0 - vface - wface)[remaining]
    bary[remaining, 1] = vface[remaining]
    bary[remaining, 2] = wface[remaining]
    closest = bary[:, 0, None] * a + bary[:, 1, None] * b + bary[:, 2, None] * c
    return closest, bary


def _closest_cartesian(points: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ericson closest point for [Q] points against common [K] triangles."""
    q, k = len(points), len(triangles)
    repeated_points = np.repeat(points, k, axis=0)
    tiled_triangles = np.tile(triangles, (q, 1, 1))
    closest, bary = _closest_point_pairs(repeated_points, tiled_triangles)
    return closest.reshape(q, k, 3), bary.reshape(q, k, 3)


def _exact_closest_surface_v2(
    queries: np.ndarray,
    triangles: np.ndarray,
    *,
    policy: MeshFieldTeacherPolicyV1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tri_lo = triangles.min(axis=1)
    tri_hi = triangles.max(axis=1)
    centroids = triangles.mean(axis=1)
    root = _BVHNode(
        np.arange(len(triangles), dtype=np.int32),
        tri_lo,
        tri_hi,
        centroids,
        policy.bvh_leaf_size,
    )
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
        seed_ids = np.asarray(seed_ids, dtype=np.int32).reshape(-1)
        seed_cp, seed_bary = _closest_point_pairs(points, triangles[seed_ids])
        best_d2 = np.sum((seed_cp - points) ** 2, axis=1)
        best_id = seed_ids.copy()
        best_bary = seed_bary.copy()
        best_ties = np.zeros(count, dtype=np.int32)

        stack: list[tuple[Any, np.ndarray]] = [(root, np.arange(count, dtype=np.int32))]
        while stack:
            node, query_ids = stack.pop()
            if not len(query_ids):
                continue
            current_distance = np.sqrt(best_d2[query_ids])
            lower2 = _bbox_distance2(points[query_ids], node.lo, node.hi)
            keep = query_ids[lower2 <= (current_distance + eps) ** 2]
            if not len(keep):
                continue
            if node.triangle_ids is None:
                stack.append((node.left, keep))
                stack.append((node.right, keep))
                continue

            leaf_ids = node.triangle_ids
            closest, leaf_bary = _closest_cartesian(points[keep], triangles[leaf_ids])
            distances = np.linalg.norm(closest - points[keep, None, :], axis=2)
            local_min = distances.min(axis=1)
            current = np.sqrt(best_d2[keep])
            for local_row, query_id in enumerate(keep.tolist()):
                d = float(local_min[local_row])
                if d < float(current[local_row]) - eps:
                    tied = np.flatnonzero(np.abs(distances[local_row] - d) <= eps)
                    tied_ids = leaf_ids[tied]
                    pos = int(tied[np.argmin(tied_ids)])
                    best_d2[query_id] = distances[local_row, pos] ** 2
                    best_id[query_id] = int(leaf_ids[pos])
                    best_bary[query_id] = leaf_bary[local_row, pos]
                    best_ties[query_id] = int(len(tied))
                elif abs(d - float(current[local_row])) <= eps:
                    tied = np.flatnonzero(np.abs(distances[local_row] - float(current[local_row])) <= eps)
                    if not len(tied):
                        continue
                    best_ties[query_id] += int(len(tied))
                    tied_ids = leaf_ids[tied]
                    pos = int(tied[np.argmin(tied_ids)])
                    if int(leaf_ids[pos]) < int(best_id[query_id]):
                        best_id[query_id] = int(leaf_ids[pos])
                        best_bary[query_id] = leaf_bary[local_row, pos]

        if np.any(best_ties < 1):
            raise RuntimeError("MESH_FIELD_TEACHER_V2_TIE_QUALIFICATION_INCOMPLETE")
        distance[start : start + count] = np.sqrt(best_d2)
        triangle_id[start : start + count] = best_id
        barycentric[start : start + count] = best_bary
        tie_count[start : start + count] = best_ties

    return distance, triangle_id, barycentric, tie_count


def build_mesh_field_teacher_v2(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    query_xyz: np.ndarray,
    sign_admission: GlobalSignAdmissionV1,
    policy: MeshFieldTeacherPolicyV1,
    provenance: str,
) -> MeshFieldTeacherResultV2:
    policy.validate()
    sign_admission.validate()
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    queries = np.asarray(query_xyz, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError("MESH_FIELD_TEACHER_V2_VERTICES_INVALID")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("MESH_FIELD_TEACHER_V2_FACES_INVALID")
    if queries.ndim != 2 or queries.shape[1] != 3 or not np.isfinite(queries).all():
        raise ValueError("MESH_FIELD_TEACHER_V2_QUERIES_INVALID")
    triangles = vertices[faces]
    area2 = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    if np.any(area2 <= policy.minimum_triangle_area2):
        raise ValueError("MESH_FIELD_TEACHER_V2_DEGENERATE_SOURCE_TRIANGLE")

    unsigned, triangle_id, barycentric, tie_count = _exact_closest_surface_v2(
        queries,
        triangles,
        policy=policy,
    )
    oriented = np.full(len(queries), np.nan, dtype=np.float64)
    orientation_valid = np.zeros(len(queries), dtype=bool)
    inside = np.zeros(len(queries), dtype=bool)
    nonambiguous_votes = np.zeros(len(queries), dtype=np.int16)
    ambiguous_count = np.zeros(len(queries), dtype=np.int16)

    if sign_admission.admitted and len(queries):
        tri_lo = triangles.min(axis=1)
        tri_hi = triangles.max(axis=1)
        centroids = triangles.mean(axis=1)
        root = _BVHNode(
            np.arange(len(triangles), dtype=np.int32),
            tri_lo,
            tri_hi,
            centroids,
            policy.bvh_leaf_size,
        )
        directions = []
        for row in policy.ray_directions:
            value = np.asarray(row, dtype=np.float64)
            directions.append(value / np.linalg.norm(value))

        on_surface = unsigned <= policy.spatial_tolerance
        oriented[on_surface] = 0.0
        orientation_valid[on_surface] = True
        nonambiguous_votes[on_surface] = len(directions)

        off_ids = np.flatnonzero(~on_surface)
        for start in range(0, len(off_ids), policy.query_batch_size):
            ids = off_ids[start : start + policy.query_batch_size]
            points = queries[ids]
            votes, ambiguities = [], []
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
                if policy.require_unanimous_votes:
                    if not np.all(values == values[0]):
                        continue
                    value = bool(values[0])
                else:
                    true_count = int(np.count_nonzero(values))
                    false_count = int(len(values) - true_count)
                    if true_count == false_count:
                        continue
                    value = true_count > false_count
                inside[query_id] = value
                orientation_valid[query_id] = True
                oriented[query_id] = -unsigned[query_id] if value else unsigned[query_id]

    result = MeshFieldTeacherResultV2(
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
        distance_authority=DISTANCE_AUTHORITY,
        sign_authority=SIGN_AUTHORITY,
        provenance=provenance,
    )
    result.validate()
    return result


__all__ = [
    "DISTANCE_AUTHORITY",
    "MeshFieldTeacherResultV2",
    "SCHEMA_VERSION",
    "SIGN_AUTHORITY",
    "build_mesh_field_teacher_v2",
]
