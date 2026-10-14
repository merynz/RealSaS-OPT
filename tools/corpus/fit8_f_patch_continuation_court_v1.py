from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from tools.corpus.fit8_local_patch_teacher_v1 import ExtractionCellTeacherV1

SCHEMA_VERSION = "RealSaS.FPatchContinuationCourt.v1"


@dataclass(frozen=True)
class PatchContinuationEdgeV1:
    a_patch_local_id: int
    b_patch_local_id: int
    evidence: str
    shared_triangle_ids: tuple[int, ...]
    adjacent_triangle_pairs: tuple[tuple[int, int], ...]

    def validate(self) -> None:
        if self.a_patch_local_id < 0 or self.b_patch_local_id < 0:
            raise ValueError("F_PATCH_CONTINUATION_PATCH_ID_INVALID")
        if self.evidence not in {"SHARED_SOURCE_TRIANGLE", "SOURCE_TRIANGLE_ADJACENCY"}:
            raise ValueError("F_PATCH_CONTINUATION_EVIDENCE_INVALID")
        if self.evidence == "SHARED_SOURCE_TRIANGLE" and not self.shared_triangle_ids:
            raise ValueError("F_PATCH_CONTINUATION_SHARED_TRIANGLE_MISSING")
        if self.evidence == "SOURCE_TRIANGLE_ADJACENCY" and not self.adjacent_triangle_pairs:
            raise ValueError("F_PATCH_CONTINUATION_ADJACENCY_MISSING")


@dataclass(frozen=True)
class FPatchContinuationReportV1:
    schema_version: str
    neighbor_axis: int
    a_patch_count: int
    b_patch_count: int
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

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("F_PATCH_CONTINUATION_SCHEMA_INVALID")
        if self.neighbor_axis not in (0, 1, 2):
            raise ValueError("F_PATCH_CONTINUATION_AXIS_INVALID")
        if self.a_patch_count < 0 or self.b_patch_count < 0 or self.max_patch_degree < 0:
            raise ValueError("F_PATCH_CONTINUATION_NEGATIVE_COUNT")
        for edge in self.continuation_edges:
            edge.validate()
        for value, upper, code in (
            (self.unmatched_a_patch_ids, self.a_patch_count, "A_UNMATCHED"),
            (self.unmatched_b_patch_ids, self.b_patch_count, "B_UNMATCHED"),
            (self.branching_a_patch_ids, self.a_patch_count, "A_BRANCH"),
            (self.branching_b_patch_ids, self.b_patch_count, "B_BRANCH"),
        ):
            if any(i < 0 or i >= upper for i in value):
                raise ValueError(f"F_PATCH_CONTINUATION_{code}_ID_INVALID")


def _neighbor_axis(a: ExtractionCellTeacherV1, b: ExtractionCellTeacherV1, tol: float) -> int:
    a.cell.validate()
    b.cell.validate()
    if abs(a.cell.pitch - b.cell.pitch) > tol:
        raise ValueError("F_PATCH_CONTINUATION_PITCH_MISMATCH")
    if abs(a.cell.half_extent - b.cell.half_extent) > tol:
        raise ValueError("F_PATCH_CONTINUATION_HALF_EXTENT_MISMATCH")
    if abs(2.0 * a.cell.half_extent - a.cell.pitch) > tol:
        raise ValueError("F_PATCH_CONTINUATION_CELL_PITCH_INCONSISTENT")
    delta = [abs(float(b.cell.center_xyz[i] - a.cell.center_xyz[i])) for i in range(3)]
    axes = [i for i, d in enumerate(delta) if abs(d - a.cell.pitch) <= tol]
    if len(axes) != 1:
        raise ValueError("F_PATCH_CONTINUATION_NOT_FACE_NEIGHBORS")
    axis = axes[0]
    if any(delta[i] > tol for i in range(3) if i != axis):
        raise ValueError("F_PATCH_CONTINUATION_NOT_FACE_NEIGHBORS")
    return axis


def _adjacency_pairs(
    a_ids: Sequence[int],
    b_ids: Sequence[int],
    triangle_adjacency: Mapping[int, Sequence[int]],
) -> tuple[tuple[int, int], ...]:
    b_set = set(map(int, b_ids))
    pairs: set[tuple[int, int]] = set()
    for a_tid in map(int, a_ids):
        for b_tid in triangle_adjacency.get(a_tid, ()):
            b_tid = int(b_tid)
            if b_tid in b_set:
                pairs.add((a_tid, b_tid))
    a_set = set(map(int, a_ids))
    for b_tid in map(int, b_ids):
        for a_tid in triangle_adjacency.get(b_tid, ()):
            a_tid = int(a_tid)
            if a_tid in a_set:
                pairs.add((a_tid, b_tid))
    return tuple(sorted(pairs))


def evaluate_neighbor_patch_continuation_v1(
    a: ExtractionCellTeacherV1,
    b: ExtractionCellTeacherV1,
    triangle_adjacency: Mapping[int, Sequence[int]],
    *,
    geometry_tolerance: float = 1e-9,
) -> FPatchContinuationReportV1:
    """Qualify teacher-local patch continuation across one shared cell face.

    Local patch IDs are intentionally cell-local and carry no cross-cell identity.
    Continuation is derived only from exact source-triangle lineage:

    1. the same source triangle intersects both cells; or
    2. source triangles from the two local patches are adjacent in qualified
       source/teacher topology.

    Equality of component, material, owner, bone or local patch ID is never used
    as continuation authority. Branching is retained explicitly rather than
    silently forcing a one-to-one stitch.
    """
    if geometry_tolerance <= 0.0:
        raise ValueError("F_PATCH_CONTINUATION_TOLERANCE_INVALID")
    a.validate()
    b.validate()
    axis = _neighbor_axis(a, b, geometry_tolerance)

    edges: list[PatchContinuationEdgeV1] = []
    for pa in a.local_patch_lower_bound:
        a_ids = set(map(int, pa.source_triangle_ids))
        for pb in b.local_patch_lower_bound:
            b_ids = set(map(int, pb.source_triangle_ids))
            shared = tuple(sorted(a_ids & b_ids))
            if shared:
                edges.append(PatchContinuationEdgeV1(
                    pa.patch_local_id,
                    pb.patch_local_id,
                    "SHARED_SOURCE_TRIANGLE",
                    shared,
                    (),
                ))
                continue
            pairs = _adjacency_pairs(pa.source_triangle_ids, pb.source_triangle_ids, triangle_adjacency)
            if pairs:
                edges.append(PatchContinuationEdgeV1(
                    pa.patch_local_id,
                    pb.patch_local_id,
                    "SOURCE_TRIANGLE_ADJACENCY",
                    (),
                    pairs,
                ))

    edges.sort(key=lambda e: (e.a_patch_local_id, e.b_patch_local_id, e.evidence))
    degree_a = {p.patch_local_id: 0 for p in a.local_patch_lower_bound}
    degree_b = {p.patch_local_id: 0 for p in b.local_patch_lower_bound}
    for edge in edges:
        degree_a[edge.a_patch_local_id] += 1
        degree_b[edge.b_patch_local_id] += 1

    unmatched_a = tuple(sorted(i for i, degree in degree_a.items() if degree == 0))
    unmatched_b = tuple(sorted(i for i, degree in degree_b.items() if degree == 0))
    branching_a = tuple(sorted(i for i, degree in degree_a.items() if degree > 1))
    branching_b = tuple(sorted(i for i, degree in degree_b.items() if degree > 1))
    one_to_one = sum(
        1 for edge in edges
        if degree_a[edge.a_patch_local_id] == 1 and degree_b[edge.b_patch_local_id] == 1
    )
    report = FPatchContinuationReportV1(
        schema_version=SCHEMA_VERSION,
        neighbor_axis=axis,
        a_patch_count=len(a.local_patch_lower_bound),
        b_patch_count=len(b.local_patch_lower_bound),
        continuation_edges=tuple(edges),
        unmatched_a_patch_ids=unmatched_a,
        unmatched_b_patch_ids=unmatched_b,
        branching_a_patch_ids=branching_a,
        branching_b_patch_ids=branching_b,
        one_to_one_edge_count=one_to_one,
        max_patch_degree=max((*degree_a.values(), *degree_b.values()), default=0),
    )
    report.validate()
    return report
