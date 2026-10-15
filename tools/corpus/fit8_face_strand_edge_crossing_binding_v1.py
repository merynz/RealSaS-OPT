from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from tools.corpus.fit8_face_strand_teacher_v1 import FaceStrandTeacherV1, FaceStrandV1
from tools.corpus.fit8_local_patch_teacher_v1 import (
    CUBE_CORNER_SIGNS,
    CUBE_EDGE_CORNERS,
    ExtractionCellTeacherV1,
)
from tools.corpus.fit8_patch_face_support_v1 import CELL_FACE_AXIS_SIDE

SCHEMA_VERSION = "RealSaS.FaceStrandEdgeCrossingBinding.v1"
EXACT = "EXACT"
CORNER_MULTI_EDGE = "CORNER_MULTI_EDGE"
STATUSES = {EXACT, CORNER_MULTI_EDGE}


@dataclass(frozen=True)
class FaceStrandCrossingBindingPolicyV1:
    distance_tolerance: float
    provenance: str

    def validate(self) -> None:
        if not np.isfinite(self.distance_tolerance) or self.distance_tolerance <= 0.0:
            raise ValueError("FACE_STRAND_CROSSING_BINDING_TOLERANCE_INVALID")
        if not self.provenance:
            raise ValueError("FACE_STRAND_CROSSING_BINDING_PROVENANCE_MISSING")


@dataclass(frozen=True)
class BoundaryEndpointCrossingBindingV1:
    endpoint_local_id: int
    xyz: tuple[float, float, float]
    source_triangle_ids: tuple[int, ...]
    cube_edge_indices: tuple[int, ...]
    edge_crossing_indices: tuple[int, ...]
    status: str

    def validate(self) -> None:
        if self.endpoint_local_id < 0:
            raise ValueError("FACE_STRAND_CROSSING_BINDING_ENDPOINT_ID_INVALID")
        xyz = np.asarray(self.xyz, dtype=np.float64)
        if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
            raise ValueError("FACE_STRAND_CROSSING_BINDING_XYZ_INVALID")
        if not self.source_triangle_ids:
            raise ValueError("FACE_STRAND_CROSSING_BINDING_SOURCE_EMPTY")
        if tuple(sorted(set(self.source_triangle_ids))) != self.source_triangle_ids:
            raise ValueError("FACE_STRAND_CROSSING_BINDING_SOURCE_NOT_CANONICAL")
        if not self.cube_edge_indices or tuple(sorted(set(self.cube_edge_indices))) != self.cube_edge_indices:
            raise ValueError("FACE_STRAND_CROSSING_BINDING_CUBE_EDGES_INVALID")
        if any(index < 0 or index >= len(CUBE_EDGE_CORNERS) for index in self.cube_edge_indices):
            raise ValueError("FACE_STRAND_CROSSING_BINDING_CUBE_EDGE_ID_INVALID")
        if not self.edge_crossing_indices or tuple(sorted(set(self.edge_crossing_indices))) != self.edge_crossing_indices:
            raise ValueError("FACE_STRAND_CROSSING_BINDING_CROSSINGS_INVALID")
        if self.status not in STATUSES:
            raise ValueError("FACE_STRAND_CROSSING_BINDING_STATUS_INVALID")
        if self.status == EXACT and len(self.edge_crossing_indices) != 1:
            raise ValueError("FACE_STRAND_CROSSING_BINDING_EXACT_MULTIPLICITY_INVALID")
        if self.status == CORNER_MULTI_EDGE and len(self.cube_edge_indices) < 2:
            raise ValueError("FACE_STRAND_CROSSING_BINDING_CORNER_EDGE_COUNT_INVALID")


@dataclass(frozen=True)
class StrandCrossingBindingV1:
    strand_local_id: int
    boundary_endpoints: tuple[BoundaryEndpointCrossingBindingV1, ...]

    def validate(self, expected_boundary_endpoint_count: int) -> None:
        if self.strand_local_id < 0:
            raise ValueError("FACE_STRAND_CROSSING_BINDING_STRAND_ID_INVALID")
        if len(self.boundary_endpoints) != expected_boundary_endpoint_count:
            raise ValueError("FACE_STRAND_CROSSING_BINDING_ENDPOINT_COUNT_MISMATCH")
        if [row.endpoint_local_id for row in self.boundary_endpoints] != list(range(len(self.boundary_endpoints))):
            raise ValueError("FACE_STRAND_CROSSING_BINDING_ENDPOINT_IDS_INVALID")
        for row in self.boundary_endpoints:
            row.validate()


@dataclass(frozen=True)
class FaceStrandEdgeCrossingBindingV1:
    schema_version: str
    face_index: int
    policy: FaceStrandCrossingBindingPolicyV1
    strands: tuple[StrandCrossingBindingV1, ...]

    def validate(self, face: FaceStrandTeacherV1) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("FACE_STRAND_CROSSING_BINDING_SCHEMA_INVALID")
        if self.face_index != face.face_index:
            raise ValueError("FACE_STRAND_CROSSING_BINDING_FACE_MISMATCH")
        self.policy.validate()
        if [row.strand_local_id for row in self.strands] != list(range(len(self.strands))):
            raise ValueError("FACE_STRAND_CROSSING_BINDING_STRAND_IDS_INVALID")
        if len(self.strands) != len(face.strands):
            raise ValueError("FACE_STRAND_CROSSING_BINDING_STRAND_COUNT_MISMATCH")
        for row, strand in zip(self.strands, face.strands):
            row.validate(int(strand.boundary_endpoint_count))


def _canonical_nodes(
    strand: FaceStrandV1,
    tolerance: float,
) -> tuple[list[np.ndarray], list[set[int]], list[int]]:
    points: list[np.ndarray] = []
    source_ids: list[set[int]] = []
    degree: list[int] = []

    def node_id(raw: Sequence[float], triangle_id: int) -> int:
        point = np.asarray(raw, dtype=np.float64)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError("FACE_STRAND_CROSSING_BINDING_POINT_INVALID")
        for index, prior in enumerate(points):
            if float(np.linalg.norm(point - prior)) <= tolerance:
                source_ids[index].add(int(triangle_id))
                return index
        points.append(point)
        source_ids.append({int(triangle_id)})
        degree.append(0)
        return len(points) - 1

    for segment in strand.segments:
        a = node_id(segment.p0_uv, int(segment.triangle_id))
        b = node_id(segment.p1_uv, int(segment.triangle_id))
        if a != b:
            degree[a] += 1
            degree[b] += 1
    return points, source_ids, degree


def _world_point(cell, face_index: int, point_uv: np.ndarray) -> np.ndarray:
    axis, side = CELL_FACE_AXIS_SIDE[face_index]
    uv_axes = [index for index in range(3) if index != int(axis)]
    center = np.asarray(cell.center_xyz, dtype=np.float64)
    world = center.copy()
    world[int(axis)] = center[int(axis)] + int(side) * float(cell.half_extent)
    world[uv_axes] = np.asarray(point_uv, dtype=np.float64)
    return world


def _edge_lies_on_face(edge_index: int, face_index: int) -> bool:
    axis, side = CELL_FACE_AXIS_SIDE[face_index]
    a, b = CUBE_EDGE_CORNERS[edge_index]
    return (
        int(CUBE_CORNER_SIGNS[int(a)][int(axis)]) == int(side)
        and int(CUBE_CORNER_SIGNS[int(b)][int(axis)]) == int(side)
    )


def _point_on_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray, tolerance: float) -> bool:
    delta = b - a
    length2 = float(np.dot(delta, delta))
    if length2 <= tolerance * tolerance:
        return float(np.linalg.norm(point - a)) <= tolerance
    t = float(np.dot(point - a, delta) / length2)
    if t < -tolerance or t > 1.0 + tolerance:
        return False
    closest = a + min(1.0, max(0.0, t)) * delta
    return float(np.linalg.norm(point - closest)) <= tolerance


def _cube_edge_candidates(cell, face_index: int, world: np.ndarray, tolerance: float) -> tuple[int, ...]:
    center = np.asarray(cell.center_xyz, dtype=np.float64)
    rows = []
    for edge_index, (corner_a, corner_b) in enumerate(CUBE_EDGE_CORNERS):
        if not _edge_lies_on_face(edge_index, face_index):
            continue
        a = center + np.asarray(CUBE_CORNER_SIGNS[int(corner_a)], dtype=np.float64) * float(cell.half_extent)
        b = center + np.asarray(CUBE_CORNER_SIGNS[int(corner_b)], dtype=np.float64) * float(cell.half_extent)
        if _point_on_segment(world, a, b, tolerance):
            rows.append(edge_index)
    return tuple(sorted(rows))


def qualify_face_strand_edge_crossing_binding_v1(
    cell_teacher: ExtractionCellTeacherV1,
    face: FaceStrandTeacherV1,
    *,
    policy: FaceStrandCrossingBindingPolicyV1,
) -> FaceStrandEdgeCrossingBindingV1:
    """Bind each face-strand boundary endpoint to exact cell EdgeCrossingV1 evidence.

    Cube-edge identity alone is insufficient because one edge may carry multiple
    authored crossings. This qualifier uses world-space endpoint position and exact
    source-triangle lineage. A true cube-corner endpoint may bind equivalent crossing
    rows on multiple incident cube edges; that case remains one endpoint with status
    `CORNER_MULTI_EDGE` rather than being split into anonymous nodes.
    """
    cell_teacher.validate()
    face.validate()
    policy.validate()
    tolerance = float(policy.distance_tolerance)
    rows: list[StrandCrossingBindingV1] = []

    for strand in face.strands:
        points, source_ids, degree = _canonical_nodes(strand, tolerance)
        endpoint_payloads = []
        for node_index, value in enumerate(degree):
            if value != 1:
                continue
            world = _world_point(cell_teacher.cell, face.face_index, points[node_index])
            cube_edges = _cube_edge_candidates(cell_teacher.cell, face.face_index, world, tolerance)
            if not cube_edges:
                continue  # interior endpoint; not a cell-boundary crossing node
            endpoint_sources = tuple(sorted(source_ids[node_index]))
            source_set = set(endpoint_sources)
            candidates = []
            for crossing_index, crossing in enumerate(cell_teacher.edge_crossings):
                if int(crossing.edge_index) not in cube_edges:
                    continue
                if float(np.linalg.norm(np.asarray(crossing.xyz, dtype=np.float64) - world)) > tolerance:
                    continue
                if not (source_set & set(map(int, crossing.source_triangle_ids))):
                    continue
                candidates.append(crossing_index)
            if not candidates:
                raise ValueError(
                    "FACE_STRAND_CROSSING_BINDING_UNBOUND_ENDPOINT:"
                    f"face={face.face_index}:strand={strand.strand_local_id}:xyz={tuple(map(float, world))}"
                )
            candidate_edges = {
                int(cell_teacher.edge_crossings[index].edge_index)
                for index in candidates
            }
            if len(candidates) == 1:
                status = EXACT
            else:
                if len(candidate_edges) < 2 or len(cube_edges) < 2:
                    raise ValueError("FACE_STRAND_CROSSING_BINDING_NONCORNER_AMBIGUITY")
                if any(
                    float(np.linalg.norm(
                        np.asarray(cell_teacher.edge_crossings[index].xyz, dtype=np.float64) - world
                    )) > tolerance
                    for index in candidates
                ):
                    raise ValueError("FACE_STRAND_CROSSING_BINDING_CORNER_POSITION_MISMATCH")
                status = CORNER_MULTI_EDGE
            endpoint_payloads.append((world, endpoint_sources, cube_edges, tuple(sorted(candidates)), status))

        endpoint_payloads.sort(key=lambda row: (tuple(map(float, row[0])), row[1], row[2], row[3]))
        bound = tuple(
            BoundaryEndpointCrossingBindingV1(
                endpoint_local_id=index,
                xyz=tuple(map(float, world)),
                source_triangle_ids=source_triangles,
                cube_edge_indices=cube_edges,
                edge_crossing_indices=crossings,
                status=status,
            )
            for index, (world, source_triangles, cube_edges, crossings, status) in enumerate(endpoint_payloads)
        )
        rows.append(StrandCrossingBindingV1(int(strand.strand_local_id), bound))

    result = FaceStrandEdgeCrossingBindingV1(
        schema_version=SCHEMA_VERSION,
        face_index=int(face.face_index),
        policy=policy,
        strands=tuple(rows),
    )
    result.validate(face)
    return result
