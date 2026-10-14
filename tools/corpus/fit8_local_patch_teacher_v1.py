from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = "RealSaS.FIT8LocalPatchTeacher.v1"
CUBE_EDGE_CORNERS = (
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
CUBE_CORNER_SIGNS = (
    (-1, -1, -1), (1, -1, -1), (-1, 1, -1), (1, 1, -1),
    (-1, -1, 1), (1, -1, 1), (-1, 1, 1), (1, 1, 1),
)


def _v3(value: Sequence[float]) -> np.ndarray:
    a = np.asarray(value, dtype=np.float64)
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        raise ValueError("FIT8_LOCAL_PATCH_VECTOR_INVALID")
    return a


@dataclass(frozen=True)
class CrossingCanonicalizationPolicyV1:
    distance_tolerance: float
    provenance: str
    preserve_identity_sets: bool = True

    def validate(self) -> None:
        if not np.isfinite(self.distance_tolerance) or self.distance_tolerance <= 0.0:
            raise ValueError("FIT8_LOCAL_PATCH_CROSSING_DISTANCE_TOLERANCE_INVALID")
        if not self.provenance:
            raise ValueError("FIT8_LOCAL_PATCH_CROSSING_POLICY_PROVENANCE_MISSING")
        if not self.preserve_identity_sets:
            raise ValueError("FIT8_LOCAL_PATCH_CROSSING_IDENTITY_PRESERVATION_REQUIRED")


@dataclass(frozen=True)
class ExtractionCellV1:
    center_xyz: tuple[float, float, float]
    half_extent: float
    pitch: float

    def validate(self) -> None:
        _v3(self.center_xyz)
        if not np.isfinite(self.half_extent) or self.half_extent <= 0.0:
            raise ValueError("FIT8_LOCAL_PATCH_CELL_HALF_EXTENT_INVALID")
        if not np.isfinite(self.pitch) or self.pitch <= 0.0:
            raise ValueError("FIT8_LOCAL_PATCH_CELL_PITCH_INVALID")

    def corners(self) -> np.ndarray:
        self.validate()
        c = _v3(self.center_xyz)
        h = float(self.half_extent)
        return np.asarray([[c[0] + sx*h, c[1] + sy*h, c[2] + sz*h]
                           for sx, sy, sz in CUBE_CORNER_SIGNS], dtype=np.float64)


@dataclass(frozen=True)
class SourceTriangleV1:
    triangle_id: int
    vertices_xyz: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    geometric_component_id: int
    source_face_id: int | None = None
    owner_region_id: int | None = None
    material_region_id: int | None = None
    mechanical_owner_id: int | None = None

    def vertices(self) -> np.ndarray:
        v = np.asarray(self.vertices_xyz, dtype=np.float64)
        if v.shape != (3, 3) or not np.all(np.isfinite(v)):
            raise ValueError("FIT8_LOCAL_PATCH_TRIANGLE_INVALID")
        return v


@dataclass(frozen=True)
class EdgeCrossingV1:
    edge_index: int
    t: float
    xyz: tuple[float, float, float]
    source_triangle_ids: tuple[int, ...]
    geometric_component_ids: tuple[int, ...]
    owner_region_ids: tuple[int, ...]
    material_region_ids: tuple[int, ...]
    mechanical_owner_ids: tuple[int, ...]
    normal_xyz: tuple[float, float, float] | None
    normal_valid: bool

    def validate(self) -> None:
        if self.edge_index < 0 or self.edge_index >= len(CUBE_EDGE_CORNERS):
            raise ValueError("FIT8_LOCAL_PATCH_EDGE_INDEX_INVALID")
        if not np.isfinite(self.t) or self.t < -1e-9 or self.t > 1.0 + 1e-9:
            raise ValueError("FIT8_LOCAL_PATCH_EDGE_T_INVALID")
        _v3(self.xyz)
        if not self.source_triangle_ids:
            raise ValueError("FIT8_LOCAL_PATCH_EDGE_SOURCE_EMPTY")
        if self.normal_valid:
            if self.normal_xyz is None:
                raise ValueError("FIT8_LOCAL_PATCH_EDGE_NORMAL_MISSING")
            n = _v3(self.normal_xyz)
            if abs(float(np.linalg.norm(n)) - 1.0) > 1e-5:
                raise ValueError("FIT8_LOCAL_PATCH_EDGE_NORMAL_NOT_UNIT")


@dataclass(frozen=True)
class LocalPatchLowerBoundV1:
    patch_local_id: int
    source_triangle_ids: tuple[int, ...]
    geometric_component_ids: tuple[int, ...]


@dataclass(frozen=True)
class ExtractionCellTeacherV1:
    schema_version: str
    cell: ExtractionCellV1
    active: bool
    intersecting_triangle_ids: tuple[int, ...]
    local_patch_lower_bound: tuple[LocalPatchLowerBoundV1, ...]
    edge_crossings: tuple[EdgeCrossingV1, ...]
    coplanar_edge_support: tuple[tuple[int, tuple[int, ...]], ...]
    max_distinct_crossings_per_edge: int
    crossing_canonicalization: CrossingCanonicalizationPolicyV1

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("FIT8_LOCAL_PATCH_SCHEMA_VERSION_INVALID")
        self.cell.validate()
        self.crossing_canonicalization.validate()
        if self.active != bool(self.intersecting_triangle_ids):
            raise ValueError("FIT8_LOCAL_PATCH_ACTIVE_MISMATCH")
        if tuple(sorted(set(self.intersecting_triangle_ids))) != self.intersecting_triangle_ids:
            raise ValueError("FIT8_LOCAL_PATCH_TRIANGLE_IDS_NOT_CANONICAL")
        ids = [p.patch_local_id for p in self.local_patch_lower_bound]
        if ids != list(range(len(ids))):
            raise ValueError("FIT8_LOCAL_PATCH_IDS_NOT_CONTIGUOUS")
        for crossing in self.edge_crossings:
            crossing.validate()
        by_edge: dict[int, int] = {}
        for crossing in self.edge_crossings:
            by_edge[crossing.edge_index] = by_edge.get(crossing.edge_index, 0) + 1
        if max(by_edge.values(), default=0) != self.max_distinct_crossings_per_edge:
            raise ValueError("FIT8_LOCAL_PATCH_MAX_CROSSING_MISMATCH")


def triangle_aabb_intersects_v1(vertices_xyz: np.ndarray, center_xyz: np.ndarray, half_extent: float,
                                eps: float = 1e-12) -> bool:
    """Exact SAT test for a triangle against an axis-aligned cube."""
    v = np.asarray(vertices_xyz, dtype=np.float64) - np.asarray(center_xyz, dtype=np.float64)[None, :]
    h = float(half_extent)
    edges = (v[1] - v[0], v[2] - v[1], v[0] - v[2])
    axes = [np.array((1.0, 0.0, 0.0)), np.array((0.0, 1.0, 0.0)), np.array((0.0, 0.0, 1.0))]
    normal = np.cross(edges[0], edges[1])
    test_axes = axes + [normal]
    for e in edges:
        for a in axes:
            test_axes.append(np.cross(e, a))
    for axis in test_axes:
        norm = float(np.linalg.norm(axis))
        if norm <= eps:
            continue
        p = v @ axis
        r = h * float(np.abs(axis).sum())
        if float(p.min()) > r + eps or float(p.max()) < -r - eps:
            return False
    return True


def _triangle_normal(v: np.ndarray, eps: float) -> np.ndarray | None:
    n = np.cross(v[1] - v[0], v[2] - v[0])
    length = float(np.linalg.norm(n))
    if length <= eps:
        return None
    return n / length


def _segment_triangle_crossing(p0: np.ndarray, p1: np.ndarray, tri: np.ndarray,
                               eps: float) -> tuple[str, float | None, np.ndarray | None]:
    """Return CROSSING, COPLANAR, or NONE using Moller-Trumbore on the cell edge."""
    d = p1 - p0
    e1 = tri[1] - tri[0]
    e2 = tri[2] - tri[0]
    h = np.cross(d, e2)
    a = float(np.dot(e1, h))
    n = np.cross(e1, e2)
    nlen = float(np.linalg.norm(n))
    if abs(a) <= eps:
        if nlen > eps:
            dist0 = abs(float(np.dot(n, p0 - tri[0]))) / nlen
            dist1 = abs(float(np.dot(n, p1 - tri[0]))) / nlen
            if dist0 <= eps and dist1 <= eps:
                tri_min, tri_max = tri.min(axis=0), tri.max(axis=0)
                seg_min, seg_max = np.minimum(p0, p1), np.maximum(p0, p1)
                if np.all(tri_max + eps >= seg_min) and np.all(seg_max + eps >= tri_min):
                    return "COPLANAR", None, None
        return "NONE", None, None
    inv = 1.0 / a
    ss = p0 - tri[0]
    u = inv * float(np.dot(ss, h))
    if u < -eps or u > 1.0 + eps:
        return "NONE", None, None
    q = np.cross(ss, e1)
    v = inv * float(np.dot(d, q))
    if v < -eps or u + v > 1.0 + eps:
        return "NONE", None, None
    t = inv * float(np.dot(e2, q))
    if t < -eps or t > 1.0 + eps:
        return "NONE", None, None
    t = min(1.0, max(0.0, t))
    return "CROSSING", t, p0 + t * d


def _canonical_optional_ids(values: Iterable[int | None]) -> tuple[int, ...]:
    return tuple(sorted({int(v) for v in values if v is not None}))


def _patch_lower_bound(intersecting_ids: Sequence[int], triangles: Mapping[int, SourceTriangleV1],
                       triangle_adjacency: Mapping[int, Sequence[int]]) -> tuple[LocalPatchLowerBoundV1, ...]:
    """Connected components of source-triangle adjacency restricted to cell triangles.

    This is deliberately named LOWER_BOUND: global source adjacency can connect
    pieces whose shared edge lies outside this cell. A future exact clipped-patch
    qualifier may split such groups further; this function must never overclaim
    final local sheet truth.
    """
    allowed = set(map(int, intersecting_ids))
    unseen = set(allowed)
    groups: list[list[int]] = []
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        stack = [root]
        group = []
        while stack:
            cur = stack.pop()
            group.append(cur)
            for nxt in triangle_adjacency.get(cur, ()):
                nxt = int(nxt)
                if nxt in unseen and nxt in allowed:
                    unseen.remove(nxt)
                    stack.append(nxt)
        groups.append(sorted(group))
    groups.sort(key=lambda g: (g[0], len(g)))
    out = []
    for pi, ids in enumerate(groups):
        comps = tuple(sorted({triangles[i].geometric_component_id for i in ids}))
        out.append(LocalPatchLowerBoundV1(pi, tuple(ids), comps))
    return tuple(out)


def extract_cell_teacher_v1(cell: ExtractionCellV1,
                            triangles: Sequence[SourceTriangleV1],
                            triangle_adjacency: Mapping[int, Sequence[int]] | None = None,
                            *,
                            crossing_canonicalization: CrossingCanonicalizationPolicyV1,
                            eps: float = 1e-9) -> ExtractionCellTeacherV1:
    """Build representation-neutral local support/patch/crossing teacher evidence.

    No fixed local-patch cardinality is imposed. No global SDF/sign assumption is
    made. Coplanar edge support is retained explicitly instead of fabricating a
    crossing. Crossing deduplication is performed in spatial coordinates under an
    explicit provenance-carrying policy; merged rows preserve all source identities.
    """
    cell.validate()
    if eps <= 0.0:
        raise ValueError("FIT8_LOCAL_PATCH_INTERSECTION_EPS_INVALID")
    crossing_canonicalization.validate()
    tri_map: dict[int, SourceTriangleV1] = {}
    for tri in triangles:
        if tri.triangle_id in tri_map:
            raise ValueError("FIT8_LOCAL_PATCH_DUPLICATE_TRIANGLE_ID")
        tri.vertices()
        tri_map[int(tri.triangle_id)] = tri
    center = _v3(cell.center_xyz)
    intersecting = []
    for tid in sorted(tri_map):
        if triangle_aabb_intersects_v1(tri_map[tid].vertices(), center, cell.half_extent, eps=eps):
            intersecting.append(tid)
    adjacency = triangle_adjacency or {}
    patches = _patch_lower_bound(intersecting, tri_map, adjacency)
    corners = cell.corners()
    crossings: list[EdgeCrossingV1] = []
    coplanar_rows: list[tuple[int, tuple[int, ...]]] = []
    for ei, (ci0, ci1) in enumerate(CUBE_EDGE_CORNERS):
        p0, p1 = corners[ci0], corners[ci1]
        raw: list[tuple[float, np.ndarray, SourceTriangleV1, np.ndarray | None]] = []
        coplanar_ids = []
        for tid in intersecting:
            tri_rec = tri_map[tid]
            tri = tri_rec.vertices()
            kind, t, xyz = _segment_triangle_crossing(p0, p1, tri, eps)
            if kind == "COPLANAR":
                coplanar_ids.append(tid)
            elif kind == "CROSSING":
                assert t is not None and xyz is not None
                raw.append((float(t), xyz, tri_rec, _triangle_normal(tri, eps)))
        if coplanar_ids:
            coplanar_rows.append((ei, tuple(sorted(set(coplanar_ids)))))
        raw.sort(key=lambda row: row[0])
        groups: list[list[tuple[float, np.ndarray, SourceTriangleV1, np.ndarray | None]]] = []
        for row in raw:
            if not groups:
                groups.append([row])
            else:
                ref_xyz = np.mean([g[1] for g in groups[-1]], axis=0)
                if float(np.linalg.norm(row[1] - ref_xyz)) > crossing_canonicalization.distance_tolerance:
                    groups.append([row])
                else:
                    groups[-1].append(row)
        for group in groups:
            t = float(sum(r[0] for r in group) / len(group))
            xyz = p0 + t * (p1 - p0)
            tri_ids = tuple(sorted({r[2].triangle_id for r in group}))
            comp_ids = tuple(sorted({r[2].geometric_component_id for r in group}))
            owner_ids = _canonical_optional_ids(r[2].owner_region_id for r in group)
            material_ids = _canonical_optional_ids(r[2].material_region_id for r in group)
            mech_ids = _canonical_optional_ids(r[2].mechanical_owner_id for r in group)
            normals = [r[3] for r in group if r[3] is not None]
            normal_valid = False
            normal_xyz = None
            if normals:
                ref = normals[0]
                aligned = [n if float(np.dot(n, ref)) >= 0.0 else -n for n in normals]
                mean = np.sum(aligned, axis=0)
                length = float(np.linalg.norm(mean))
                if length > eps:
                    mean = mean / length
                    normal_valid = True
                    normal_xyz = tuple(map(float, mean))
            crossings.append(EdgeCrossingV1(
                edge_index=ei,
                t=t,
                xyz=tuple(map(float, xyz)),
                source_triangle_ids=tri_ids,
                geometric_component_ids=comp_ids,
                owner_region_ids=owner_ids,
                material_region_ids=material_ids,
                mechanical_owner_ids=mech_ids,
                normal_xyz=normal_xyz,
                normal_valid=normal_valid,
            ))
    crossings.sort(key=lambda c: (c.edge_index, c.t, c.source_triangle_ids))
    by_edge: dict[int, int] = {}
    for crossing in crossings:
        by_edge[crossing.edge_index] = by_edge.get(crossing.edge_index, 0) + 1
    result = ExtractionCellTeacherV1(
        schema_version=SCHEMA_VERSION,
        cell=cell,
        active=bool(intersecting),
        intersecting_triangle_ids=tuple(intersecting),
        local_patch_lower_bound=patches,
        edge_crossings=tuple(crossings),
        coplanar_edge_support=tuple(coplanar_rows),
        max_distinct_crossings_per_edge=max(by_edge.values(), default=0),
        crossing_canonicalization=crossing_canonicalization,
    )
    result.validate()
    return result
