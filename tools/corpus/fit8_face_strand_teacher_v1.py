from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from tools.corpus.fit8_local_patch_teacher_v1 import ExtractionCellV1, SourceTriangleV1
from tools.corpus.fit8_patch_face_support_v1 import CELL_FACE_AXIS_SIDE

SCHEMA_VERSION = "RealSaS.FIT8FaceStrandTeacher.v1"
STRAND_KINDS = {
    "BOUNDARY_TO_BOUNDARY_ARC",
    "CLOSED_LOOP",
    "OPEN_INTERIOR_ENDPOINT",
    "BRANCH_OR_NONMANIFOLD",
    "DEGENERATE_OR_COMPLEX",
}


@dataclass(frozen=True)
class FaceSegmentV1:
    triangle_id: int
    p0_uv: tuple[float, float]
    p1_uv: tuple[float, float]


@dataclass(frozen=True)
class FaceStrandV1:
    strand_local_id: int
    source_triangle_ids: tuple[int, ...]
    segments: tuple[FaceSegmentV1, ...]
    kind: str
    boundary_endpoint_count: int
    interior_endpoint_count: int
    max_graph_degree: int


@dataclass(frozen=True)
class FaceStrandTeacherV1:
    schema_version: str
    face_index: int
    strands: tuple[FaceStrandV1, ...]
    coplanar_triangle_ids: tuple[int, ...]
    point_degeneracy_triangle_ids: tuple[int, ...]

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("FIT8_FACE_STRAND_SCHEMA_INVALID")
        if self.face_index < 0 or self.face_index >= 6:
            raise ValueError("FIT8_FACE_STRAND_FACE_INVALID")
        if [strand.strand_local_id for strand in self.strands] != list(range(len(self.strands))):
            raise ValueError("FIT8_FACE_STRAND_IDS_INVALID")
        for strand in self.strands:
            if strand.kind not in STRAND_KINDS:
                raise ValueError("FIT8_FACE_STRAND_KIND_INVALID")
            if not strand.source_triangle_ids or not strand.segments:
                raise ValueError("FIT8_FACE_STRAND_EMPTY")


def _clip_segment_rect(
    p0: np.ndarray,
    p1: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    t0, t1 = 0.0, 1.0
    delta = p1 - p0
    for axis in range(2):
        if abs(float(delta[axis])) <= eps:
            if p0[axis] < lo[axis] - eps or p0[axis] > hi[axis] + eps:
                return None
            continue
        a = float((lo[axis] - p0[axis]) / delta[axis])
        b = float((hi[axis] - p0[axis]) / delta[axis])
        t0 = max(t0, min(a, b))
        t1 = min(t1, max(a, b))
        if t0 > t1 + eps:
            return None
    q0 = p0 + t0 * delta
    q1 = p0 + t1 * delta
    if float(np.linalg.norm(q1 - q0)) <= eps:
        return None
    return q0, q1


def _triangle_face_segment(
    triangle: np.ndarray,
    axis: int,
    plane: float,
    uv_axes: Sequence[int],
    lo: np.ndarray,
    hi: np.ndarray,
    eps: float,
) -> tuple[str, tuple[np.ndarray, np.ndarray] | None]:
    distance = triangle[:, axis] - plane
    if np.all(distance > eps) or np.all(distance < -eps):
        return "NONE", None
    if np.all(np.abs(distance) <= eps):
        return "COPLANAR", None

    points: list[np.ndarray] = []
    for i in range(3):
        if abs(float(distance[i])) <= eps:
            points.append(triangle[i, uv_axes])
    for i, j in ((0, 1), (1, 2), (2, 0)):
        di, dj = float(distance[i]), float(distance[j])
        if di * dj < -(eps * eps):
            t = di / (di - dj)
            points.append((triangle[i] + t * (triangle[j] - triangle[i]))[uv_axes])

    unique: list[np.ndarray] = []
    for point in points:
        point = np.asarray(point, dtype=np.float64)
        if not any(float(np.linalg.norm(point - prior)) <= eps for prior in unique):
            unique.append(point)
    if len(unique) < 2:
        return ("POINT", None) if unique else ("NONE", None)

    _, p0, p1 = max(
        (
            (float(np.linalg.norm(unique[i] - unique[j])), unique[i], unique[j])
            for i in range(len(unique))
            for j in range(i + 1, len(unique))
        ),
        key=lambda row: row[0],
    )
    clipped = _clip_segment_rect(p0, p1, lo, hi, eps)
    return ("SEGMENT", clipped) if clipped is not None else ("NONE", None)


def _source_adjacent(
    a_triangle_id: int,
    b_triangle_id: int,
    triangle_adjacency: Mapping[int, Sequence[int]],
) -> bool:
    return (
        b_triangle_id in set(map(int, triangle_adjacency.get(a_triangle_id, ())))
        or a_triangle_id in set(map(int, triangle_adjacency.get(b_triangle_id, ())))
    )


def derive_face_strands_v1(
    cell: ExtractionCellV1,
    face_index: int,
    triangles: Sequence[SourceTriangleV1],
    triangle_adjacency: Mapping[int, Sequence[int]],
    *,
    geometry_tolerance: float = 1e-9,
) -> FaceStrandTeacherV1:
    """Derive representation-neutral authored contour strands on one extraction-cell face.

    Strands are connected only through qualified source-triangle adjacency plus
    coincident face-intersection endpoints. Geometric crossing of unrelated sheets
    does not silently merge their identities.
    """
    if geometry_tolerance <= 0.0:
        raise ValueError("FIT8_FACE_STRAND_TOLERANCE_INVALID")
    cell.validate()
    if face_index < 0 or face_index >= 6:
        raise ValueError("FIT8_FACE_STRAND_FACE_INVALID")
    triangle_map = {int(triangle.triangle_id): triangle for triangle in triangles}
    if len(triangle_map) != len(triangles):
        raise ValueError("FIT8_FACE_STRAND_DUPLICATE_TRIANGLE_ID")

    center = np.asarray(cell.center_xyz, dtype=np.float64)
    axis, side = CELL_FACE_AXIS_SIDE[face_index]
    plane = float(center[axis] + side * cell.half_extent)
    uv_axes = [i for i in range(3) if i != axis]
    lo = center[uv_axes] - cell.half_extent
    hi = center[uv_axes] + cell.half_extent

    segments: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    coplanar: list[int] = []
    point_degeneracy: list[int] = []
    for triangle_id in sorted(triangle_map):
        kind, segment = _triangle_face_segment(
            triangle_map[triangle_id].vertices(),
            axis,
            plane,
            uv_axes,
            lo,
            hi,
            geometry_tolerance,
        )
        if kind == "SEGMENT":
            assert segment is not None
            segments[triangle_id] = segment
        elif kind == "COPLANAR":
            coplanar.append(triangle_id)
        elif kind == "POINT":
            point_degeneracy.append(triangle_id)

    unseen = set(segments)
    groups: list[list[int]] = []
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        stack = [root]
        group: list[int] = []
        while stack:
            current = stack.pop()
            group.append(current)
            current_segment = segments[current]
            for candidate in sorted(list(unseen)):
                if not _source_adjacent(current, candidate, triangle_adjacency):
                    continue
                candidate_segment = segments[candidate]
                endpoint_distance = min(
                    float(np.linalg.norm(a - b))
                    for a in current_segment
                    for b in candidate_segment
                )
                if endpoint_distance <= geometry_tolerance:
                    unseen.remove(candidate)
                    stack.append(candidate)
        groups.append(sorted(group))

    strands: list[FaceStrandV1] = []
    for strand_local_id, group in enumerate(groups):
        points: list[np.ndarray] = []
        graph_edges: list[tuple[int, int]] = []
        rows: list[FaceSegmentV1] = []

        def node_id(point: np.ndarray) -> int:
            for index, prior in enumerate(points):
                if float(np.linalg.norm(point - prior)) <= geometry_tolerance:
                    return index
            points.append(np.asarray(point, dtype=np.float64))
            return len(points) - 1

        for triangle_id in group:
            p0, p1 = segments[triangle_id]
            a = node_id(p0)
            b = node_id(p1)
            graph_edges.append((a, b))
            rows.append(
                FaceSegmentV1(
                    triangle_id=triangle_id,
                    p0_uv=tuple(map(float, p0)),
                    p1_uv=tuple(map(float, p1)),
                )
            )

        degree = [0] * len(points)
        for a, b in graph_edges:
            if a != b:
                degree[a] += 1
                degree[b] += 1
        degree_one = [index for index, value in enumerate(degree) if value == 1]

        def on_perimeter(point: np.ndarray) -> bool:
            return any(
                abs(float(point[k] - lo[k])) <= geometry_tolerance
                or abs(float(point[k] - hi[k])) <= geometry_tolerance
                for k in range(2)
            )

        boundary_endpoints = sum(on_perimeter(points[index]) for index in degree_one)
        interior_endpoints = len(degree_one) - boundary_endpoints
        max_degree = max(degree, default=0)
        if max_degree > 2:
            kind = "BRANCH_OR_NONMANIFOLD"
        elif len(degree_one) == 0:
            kind = "CLOSED_LOOP"
        elif len(degree_one) == 2 and boundary_endpoints == 2:
            kind = "BOUNDARY_TO_BOUNDARY_ARC"
        elif len(degree_one) == 2 and interior_endpoints > 0:
            kind = "OPEN_INTERIOR_ENDPOINT"
        else:
            kind = "DEGENERATE_OR_COMPLEX"

        strands.append(
            FaceStrandV1(
                strand_local_id=strand_local_id,
                source_triangle_ids=tuple(group),
                segments=tuple(rows),
                kind=kind,
                boundary_endpoint_count=int(boundary_endpoints),
                interior_endpoint_count=int(interior_endpoints),
                max_graph_degree=int(max_degree),
            )
        )

    result = FaceStrandTeacherV1(
        schema_version=SCHEMA_VERSION,
        face_index=face_index,
        strands=tuple(strands),
        coplanar_triangle_ids=tuple(coplanar),
        point_degeneracy_triangle_ids=tuple(point_degeneracy),
    )
    result.validate()
    return result
