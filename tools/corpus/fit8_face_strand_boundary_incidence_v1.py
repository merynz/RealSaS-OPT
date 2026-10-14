from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from tools.corpus.fit8_face_strand_teacher_v1 import FaceStrandTeacherV1, FaceStrandV1
from tools.corpus.fit8_local_patch_teacher_v1 import CUBE_CORNER_SIGNS, CUBE_EDGE_CORNERS, ExtractionCellV1
from tools.corpus.fit8_patch_face_support_v1 import CELL_FACE_AXIS_SIDE

SCHEMA_VERSION = "RealSaS.FaceStrandBoundaryIncidenceQualification.v1"


@dataclass(frozen=True)
class FaceBoundaryIncidencePolicyV1:
    distance_tolerance: float
    provenance: str

    def validate(self) -> None:
        if not np.isfinite(self.distance_tolerance) or self.distance_tolerance <= 0.0:
            raise ValueError("FACE_STRAND_BOUNDARY_INCIDENCE_TOLERANCE_INVALID")
        if not self.provenance:
            raise ValueError("FACE_STRAND_BOUNDARY_INCIDENCE_PROVENANCE_MISSING")


@dataclass(frozen=True)
class StrandBoundaryIncidenceV1:
    strand_local_id: int
    boundary_endpoint_cube_edge_sets: tuple[tuple[int, ...], ...]

    def validate(self) -> None:
        if self.strand_local_id < 0:
            raise ValueError("FACE_STRAND_BOUNDARY_INCIDENCE_STRAND_ID_INVALID")
        for edge_set in self.boundary_endpoint_cube_edge_sets:
            if not edge_set:
                raise ValueError("FACE_STRAND_BOUNDARY_INCIDENCE_EDGE_SET_EMPTY")
            if tuple(sorted(set(edge_set))) != edge_set:
                raise ValueError("FACE_STRAND_BOUNDARY_INCIDENCE_EDGE_SET_NOT_CANONICAL")
            if any(edge < 0 or edge >= len(CUBE_EDGE_CORNERS) for edge in edge_set):
                raise ValueError("FACE_STRAND_BOUNDARY_INCIDENCE_EDGE_ID_INVALID")


@dataclass(frozen=True)
class FaceStrandBoundaryIncidenceQualificationV1:
    schema_version: str
    face_index: int
    policy: FaceBoundaryIncidencePolicyV1
    strand_incidence: tuple[StrandBoundaryIncidenceV1, ...]

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("FACE_STRAND_BOUNDARY_INCIDENCE_SCHEMA_INVALID")
        if self.face_index < 0 or self.face_index >= len(CELL_FACE_AXIS_SIDE):
            raise ValueError("FACE_STRAND_BOUNDARY_INCIDENCE_FACE_INVALID")
        self.policy.validate()
        if [row.strand_local_id for row in self.strand_incidence] != list(range(len(self.strand_incidence))):
            raise ValueError("FACE_STRAND_BOUNDARY_INCIDENCE_STRAND_IDS_INVALID")
        for row in self.strand_incidence:
            row.validate()


def _canonical_nodes_from_strand(strand: FaceStrandV1, tolerance: float) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    points: list[np.ndarray] = []
    graph_edges: list[tuple[int, int]] = []

    def node_id(raw: Sequence[float]) -> int:
        point = np.asarray(raw, dtype=np.float64)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError("FACE_STRAND_BOUNDARY_INCIDENCE_POINT_INVALID")
        for index, prior in enumerate(points):
            if float(np.linalg.norm(point - prior)) <= tolerance:
                return index
        points.append(point)
        return len(points) - 1

    for segment in strand.segments:
        a = node_id(segment.p0_uv)
        b = node_id(segment.p1_uv)
        if a != b:
            graph_edges.append((a, b))
    return points, graph_edges


def _edge_lies_on_face(edge_index: int, face_index: int) -> bool:
    axis, side = CELL_FACE_AXIS_SIDE[face_index]
    a, b = CUBE_EDGE_CORNERS[edge_index]
    return (
        int(CUBE_CORNER_SIGNS[int(a)][int(axis)]) == int(side)
        and int(CUBE_CORNER_SIGNS[int(b)][int(axis)]) == int(side)
    )


def _point_on_segment_2d(point: np.ndarray, a: np.ndarray, b: np.ndarray, tolerance: float) -> bool:
    delta = b - a
    length2 = float(np.dot(delta, delta))
    if length2 <= tolerance * tolerance:
        return float(np.linalg.norm(point - a)) <= tolerance
    t = float(np.dot(point - a, delta) / length2)
    if t < -tolerance or t > 1.0 + tolerance:
        return False
    t = min(1.0, max(0.0, t))
    closest = a + t * delta
    return float(np.linalg.norm(point - closest)) <= tolerance


def _cube_edge_candidates_for_face_point(
    cell: ExtractionCellV1,
    face_index: int,
    point_uv: np.ndarray,
    tolerance: float,
) -> tuple[int, ...]:
    axis, _ = CELL_FACE_AXIS_SIDE[face_index]
    uv_axes = [index for index in range(3) if index != int(axis)]
    center = np.asarray(cell.center_xyz, dtype=np.float64)
    candidates = []
    for edge_index, (corner_a, corner_b) in enumerate(CUBE_EDGE_CORNERS):
        if not _edge_lies_on_face(edge_index, face_index):
            continue
        sign_a = np.asarray(CUBE_CORNER_SIGNS[int(corner_a)], dtype=np.float64)
        sign_b = np.asarray(CUBE_CORNER_SIGNS[int(corner_b)], dtype=np.float64)
        world_a = center + sign_a * float(cell.half_extent)
        world_b = center + sign_b * float(cell.half_extent)
        a_uv = world_a[uv_axes]
        b_uv = world_b[uv_axes]
        if _point_on_segment_2d(point_uv, a_uv, b_uv, tolerance):
            candidates.append(edge_index)
    return tuple(sorted(candidates))


def qualify_face_strand_boundary_incidence_v1(
    cell: ExtractionCellV1,
    face: FaceStrandTeacherV1,
    *,
    policy: FaceBoundaryIncidencePolicyV1,
) -> FaceStrandBoundaryIncidenceQualificationV1:
    """Bind strand boundary endpoints to cube-edge identity under explicit policy.

    A face-corner endpoint legitimately belongs to two cube edges; the artifact
    preserves that set instead of selecting one arbitrary edge. Interior endpoints
    produce no boundary edge set. The qualified boundary-endpoint count must agree
    with the raw strand teacher or the court fails closed.
    """
    cell.validate()
    face.validate()
    policy.validate()
    tolerance = float(policy.distance_tolerance)
    rows: list[StrandBoundaryIncidenceV1] = []

    for strand in face.strands:
        points, graph_edges = _canonical_nodes_from_strand(strand, tolerance)
        degree = [0] * len(points)
        for a, b in graph_edges:
            degree[a] += 1
            degree[b] += 1
        degree_one = [index for index, value in enumerate(degree) if value == 1]
        boundary_sets = []
        for node_index in degree_one:
            edge_set = _cube_edge_candidates_for_face_point(
                cell,
                face.face_index,
                points[node_index],
                tolerance,
            )
            if edge_set:
                boundary_sets.append(edge_set)
        boundary_sets = sorted(boundary_sets)
        if len(boundary_sets) != int(strand.boundary_endpoint_count):
            raise ValueError(
                "FACE_STRAND_BOUNDARY_INCIDENCE_COUNT_MISMATCH:"
                f"strand={strand.strand_local_id}:"
                f"raw={strand.boundary_endpoint_count}:qualified={len(boundary_sets)}"
            )
        rows.append(
            StrandBoundaryIncidenceV1(
                strand_local_id=int(strand.strand_local_id),
                boundary_endpoint_cube_edge_sets=tuple(boundary_sets),
            )
        )

    result = FaceStrandBoundaryIncidenceQualificationV1(
        schema_version=SCHEMA_VERSION,
        face_index=int(face.face_index),
        policy=policy,
        strand_incidence=tuple(rows),
    )
    result.validate()
    return result
