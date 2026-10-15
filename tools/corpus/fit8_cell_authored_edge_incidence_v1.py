from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from tools.corpus.fit8_local_patch_teacher_v1 import ExtractionCellV1

SCHEMA_VERSION = "RealSaS.CellAuthoredEdgeIncidence.v1"
MANIFOLD_INTERIOR = "MANIFOLD_INTERIOR"
AUTHORED_OPEN_BOUNDARY = "AUTHORED_OPEN_BOUNDARY"
AUTHORED_NONMANIFOLD = "AUTHORED_NONMANIFOLD"
EDGE_KINDS = {MANIFOLD_INTERIOR, AUTHORED_OPEN_BOUNDARY, AUTHORED_NONMANIFOLD}
OUTSIDE = "OUTSIDE"
POINT_TOUCH = "POINT_TOUCH"
POSITIVE_LENGTH = "POSITIVE_LENGTH"
SUPPORT_KINDS = {OUTSIDE, POINT_TOUCH, POSITIVE_LENGTH}


@dataclass(frozen=True)
class QualifiedSurfaceEdgeV1:
    edge_id: int
    endpoint0_xyz: tuple[float, float, float]
    endpoint1_xyz: tuple[float, float, float]
    incident_triangle_ids: tuple[int, ...]
    provenance: str

    def validate(self, tolerance: float) -> None:
        if self.edge_id < 0:
            raise ValueError("CELL_AUTHORED_EDGE_ID_INVALID")
        p0 = np.asarray(self.endpoint0_xyz, dtype=np.float64)
        p1 = np.asarray(self.endpoint1_xyz, dtype=np.float64)
        if p0.shape != (3,) or p1.shape != (3,) or not np.all(np.isfinite(p0)) or not np.all(np.isfinite(p1)):
            raise ValueError("CELL_AUTHORED_EDGE_ENDPOINT_INVALID")
        if float(np.linalg.norm(p1 - p0)) <= tolerance:
            raise ValueError("CELL_AUTHORED_EDGE_DEGENERATE")
        if not self.incident_triangle_ids:
            raise ValueError("CELL_AUTHORED_EDGE_TRIANGLE_INCIDENCE_EMPTY")
        if tuple(sorted(set(self.incident_triangle_ids))) != self.incident_triangle_ids:
            raise ValueError("CELL_AUTHORED_EDGE_TRIANGLE_INCIDENCE_NOT_CANONICAL")
        if not self.provenance:
            raise ValueError("CELL_AUTHORED_EDGE_PROVENANCE_MISSING")

    @property
    def topology_kind(self) -> str:
        count = len(self.incident_triangle_ids)
        if count == 1:
            return AUTHORED_OPEN_BOUNDARY
        if count == 2:
            return MANIFOLD_INTERIOR
        return AUTHORED_NONMANIFOLD


@dataclass(frozen=True)
class CellAuthoredEdgeIncidencePolicyV1:
    distance_tolerance: float
    provenance: str
    edge_set_complete: bool = True

    def validate(self) -> None:
        if not np.isfinite(self.distance_tolerance) or self.distance_tolerance <= 0.0:
            raise ValueError("CELL_AUTHORED_EDGE_TOLERANCE_INVALID")
        if not self.provenance:
            raise ValueError("CELL_AUTHORED_EDGE_POLICY_PROVENANCE_MISSING")
        if not self.edge_set_complete:
            raise ValueError("CELL_AUTHORED_EDGE_SET_INCOMPLETE")


@dataclass(frozen=True)
class CellAuthoredEdgeSupportV1:
    edge_id: int
    topology_kind: str
    support_kind: str
    clipped_endpoint0_xyz: tuple[float, float, float] | None
    clipped_endpoint1_xyz: tuple[float, float, float] | None
    incident_triangle_ids: tuple[int, ...]

    def validate(self) -> None:
        if self.edge_id < 0 or self.topology_kind not in EDGE_KINDS or self.support_kind not in SUPPORT_KINDS:
            raise ValueError("CELL_AUTHORED_EDGE_SUPPORT_INVALID")
        if self.support_kind == OUTSIDE:
            if self.clipped_endpoint0_xyz is not None or self.clipped_endpoint1_xyz is not None:
                raise ValueError("CELL_AUTHORED_EDGE_OUTSIDE_HAS_GEOMETRY")
        else:
            if self.clipped_endpoint0_xyz is None or self.clipped_endpoint1_xyz is None:
                raise ValueError("CELL_AUTHORED_EDGE_SUPPORTED_GEOMETRY_MISSING")


@dataclass(frozen=True)
class CellAuthoredEdgeIncidenceV1:
    schema_version: str
    policy: CellAuthoredEdgeIncidencePolicyV1
    rows: tuple[CellAuthoredEdgeSupportV1, ...]

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("CELL_AUTHORED_EDGE_SCHEMA_INVALID")
        self.policy.validate()
        if [row.edge_id for row in self.rows] != sorted(row.edge_id for row in self.rows):
            raise ValueError("CELL_AUTHORED_EDGE_ROWS_NOT_CANONICAL")
        if len({row.edge_id for row in self.rows}) != len(self.rows):
            raise ValueError("CELL_AUTHORED_EDGE_DUPLICATE_ID")
        for row in self.rows:
            row.validate()


def _clip_segment_to_cell(
    p0: np.ndarray,
    p1: np.ndarray,
    cell: ExtractionCellV1,
    tolerance: float,
) -> tuple[str, np.ndarray | None, np.ndarray | None]:
    center = np.asarray(cell.center_xyz, dtype=np.float64)
    lo = center - float(cell.half_extent)
    hi = center + float(cell.half_extent)
    direction = p1 - p0
    t0, t1 = 0.0, 1.0
    for axis in range(3):
        if abs(float(direction[axis])) <= tolerance:
            if p0[axis] < lo[axis] - tolerance or p0[axis] > hi[axis] + tolerance:
                return OUTSIDE, None, None
            continue
        a = float((lo[axis] - p0[axis]) / direction[axis])
        b = float((hi[axis] - p0[axis]) / direction[axis])
        t0 = max(t0, min(a, b))
        t1 = min(t1, max(a, b))
        if t0 > t1 + tolerance:
            return OUTSIDE, None, None
    q0 = p0 + t0 * direction
    q1 = p0 + t1 * direction
    if float(np.linalg.norm(q1 - q0)) <= tolerance:
        point = (q0 + q1) * 0.5
        return POINT_TOUCH, point, point
    return POSITIVE_LENGTH, q0, q1


def qualify_cell_authored_edge_incidence_v1(
    cell: ExtractionCellV1,
    edges: Sequence[QualifiedSurfaceEdgeV1],
    *,
    policy: CellAuthoredEdgeIncidencePolicyV1,
) -> CellAuthoredEdgeIncidenceV1:
    """Classify qualified source-carrier edge incidence inside one extraction cell.

    `AUTHORED_OPEN_BOUNDARY` and `AUTHORED_NONMANIFOLD` are topology evidence,
    not material/owner semantics. Lower-dimensional POINT_TOUCH is retained but is
    not equivalent to a positive-length local boundary segment.
    """
    cell.validate()
    policy.validate()
    tolerance = float(policy.distance_tolerance)
    seen: set[int] = set()
    rows = []
    for edge in sorted(edges, key=lambda row: row.edge_id):
        edge.validate(tolerance)
        if edge.edge_id in seen:
            raise ValueError("CELL_AUTHORED_EDGE_DUPLICATE_INPUT_ID")
        seen.add(edge.edge_id)
        p0 = np.asarray(edge.endpoint0_xyz, dtype=np.float64)
        p1 = np.asarray(edge.endpoint1_xyz, dtype=np.float64)
        support, q0, q1 = _clip_segment_to_cell(p0, p1, cell, tolerance)
        rows.append(
            CellAuthoredEdgeSupportV1(
                edge_id=int(edge.edge_id),
                topology_kind=edge.topology_kind,
                support_kind=support,
                clipped_endpoint0_xyz=None if q0 is None else tuple(map(float, q0)),
                clipped_endpoint1_xyz=None if q1 is None else tuple(map(float, q1)),
                incident_triangle_ids=tuple(map(int, edge.incident_triangle_ids)),
            )
        )
    result = CellAuthoredEdgeIncidenceV1(SCHEMA_VERSION, policy, tuple(rows))
    result.validate()
    return result
