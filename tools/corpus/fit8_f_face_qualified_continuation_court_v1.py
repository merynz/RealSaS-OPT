from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from tools.corpus.fit8_f_patch_continuation_court_v1 import (
    PatchContinuationEdgeV1,
    evaluate_neighbor_patch_continuation_v1,
)
from tools.corpus.fit8_local_patch_teacher_v1 import ExtractionCellTeacherV1
from tools.corpus.fit8_patch_face_support_v1 import PatchFaceSupportReportV1

SCHEMA_VERSION = "RealSaS.FFaceQualifiedPatchContinuationCourt.v1"


@dataclass(frozen=True)
class FFaceQualifiedPatchContinuationReportV1:
    schema_version: str
    neighbor_axis: int
    a_shared_face_index: int
    b_shared_face_index: int
    a_face_patch_ids: tuple[int, ...]
    b_face_patch_ids: tuple[int, ...]
    continuation_edges: tuple[PatchContinuationEdgeV1, ...]
    unmatched_a_patch_ids: tuple[int, ...]
    unmatched_b_patch_ids: tuple[int, ...]
    branching_a_patch_ids: tuple[int, ...]
    branching_b_patch_ids: tuple[int, ...]
    one_to_one_edge_count: int
    max_patch_degree: int

    @property
    def continuation_edge_count(self) -> int:
        return len(self.continuation_edges)

    @property
    def one_to_one_fraction(self) -> float:
        return self.one_to_one_edge_count / max(1, self.continuation_edge_count)

    @property
    def continuation_complete(self) -> bool:
        return not self.unmatched_a_patch_ids and not self.unmatched_b_patch_ids

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("F_FACE_CONTINUATION_SCHEMA_INVALID")
        if self.neighbor_axis not in (0, 1, 2):
            raise ValueError("F_FACE_CONTINUATION_AXIS_INVALID")
        if self.a_shared_face_index < 0 or self.a_shared_face_index >= 6:
            raise ValueError("F_FACE_CONTINUATION_FACE_INVALID")
        if self.b_shared_face_index < 0 or self.b_shared_face_index >= 6:
            raise ValueError("F_FACE_CONTINUATION_FACE_INVALID")
        for edge in self.continuation_edges:
            edge.validate()


def _shared_face_indices(
    a: ExtractionCellTeacherV1,
    b: ExtractionCellTeacherV1,
    axis: int,
    tolerance: float,
) -> tuple[int, int]:
    delta = float(b.cell.center_xyz[axis] - a.cell.center_xyz[axis])
    if abs(abs(delta) - a.cell.pitch) > tolerance:
        raise ValueError("F_FACE_CONTINUATION_NEIGHBOR_DIRECTION_INVALID")
    if delta > 0.0:
        return 2 * axis + 1, 2 * axis
    return 2 * axis, 2 * axis + 1


def evaluate_face_qualified_neighbor_patch_continuation_v1(
    a: ExtractionCellTeacherV1,
    b: ExtractionCellTeacherV1,
    a_face_support: PatchFaceSupportReportV1,
    b_face_support: PatchFaceSupportReportV1,
    triangle_adjacency: Mapping[int, Sequence[int]],
    *,
    geometry_tolerance: float = 1e-9,
) -> FFaceQualifiedPatchContinuationReportV1:
    """Evaluate continuation only for local patches that actually reach the shared face.

    The unqualified continuation court answers whether source-lineage connections
    exist between cell-local patches. This court additionally asks whether every
    patch that physically reaches the shared face has a qualified continuation.
    Interior patches are excluded rather than being mislabeled as stitch failures.
    """
    if geometry_tolerance <= 0.0:
        raise ValueError("F_FACE_CONTINUATION_TOLERANCE_INVALID")
    a_face_support.validate()
    b_face_support.validate()
    if len(a_face_support.rows) != len(a.local_patch_lower_bound):
        raise ValueError("F_FACE_CONTINUATION_SUPPORT_CARDINALITY_MISMATCH")
    if len(b_face_support.rows) != len(b.local_patch_lower_bound):
        raise ValueError("F_FACE_CONTINUATION_SUPPORT_CARDINALITY_MISMATCH")

    base = evaluate_neighbor_patch_continuation_v1(
        a,
        b,
        triangle_adjacency,
        geometry_tolerance=geometry_tolerance,
    )
    a_face, b_face = _shared_face_indices(a, b, base.neighbor_axis, geometry_tolerance)
    a_ids = tuple(row.patch_local_id for row in a_face_support.rows if a_face in row.cell_face_indices)
    b_ids = tuple(row.patch_local_id for row in b_face_support.rows if b_face in row.cell_face_indices)
    a_set, b_set = set(a_ids), set(b_ids)
    edges = tuple(
        edge
        for edge in base.continuation_edges
        if edge.a_patch_local_id in a_set and edge.b_patch_local_id in b_set
    )

    degree_a = {patch_id: 0 for patch_id in a_ids}
    degree_b = {patch_id: 0 for patch_id in b_ids}
    for edge in edges:
        degree_a[edge.a_patch_local_id] += 1
        degree_b[edge.b_patch_local_id] += 1

    report = FFaceQualifiedPatchContinuationReportV1(
        schema_version=SCHEMA_VERSION,
        neighbor_axis=base.neighbor_axis,
        a_shared_face_index=a_face,
        b_shared_face_index=b_face,
        a_face_patch_ids=a_ids,
        b_face_patch_ids=b_ids,
        continuation_edges=edges,
        unmatched_a_patch_ids=tuple(i for i, degree in degree_a.items() if degree == 0),
        unmatched_b_patch_ids=tuple(i for i, degree in degree_b.items() if degree == 0),
        branching_a_patch_ids=tuple(i for i, degree in degree_a.items() if degree > 1),
        branching_b_patch_ids=tuple(i for i, degree in degree_b.items() if degree > 1),
        one_to_one_edge_count=sum(
            1
            for edge in edges
            if degree_a[edge.a_patch_local_id] == 1 and degree_b[edge.b_patch_local_id] == 1
        ),
        max_patch_degree=max((*degree_a.values(), *degree_b.values()), default=0),
    )
    report.validate()
    return report
