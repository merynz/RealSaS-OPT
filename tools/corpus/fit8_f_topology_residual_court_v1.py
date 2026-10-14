from __future__ import annotations

from dataclasses import dataclass

from tools.corpus.fit8_face_strand_teacher_v1 import FaceStrandTeacherV1
from tools.corpus.fit8_local_patch_teacher_v1 import ExtractionCellTeacherV1

SCHEMA_VERSION = "RealSaS.FTopologyResidualCourt.v1"


@dataclass(frozen=True)
class FTopologyResidualReportV1:
    schema_version: str
    face_index: int
    strand_count: int
    patch_strand_counts: tuple[tuple[int, int], ...]
    topology_residual_reasons: tuple[str, ...]
    canonicalization_reasons: tuple[str, ...]

    @property
    def topology_residual_required(self) -> bool:
        return bool(self.topology_residual_reasons)

    @property
    def canonicalization_required(self) -> bool:
        return bool(self.canonicalization_reasons)

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("F_TOPOLOGY_RESIDUAL_SCHEMA_INVALID")
        if self.face_index < 0 or self.face_index >= 6:
            raise ValueError("F_TOPOLOGY_RESIDUAL_FACE_INVALID")


def evaluate_face_topology_residual_v1(
    cell: ExtractionCellTeacherV1,
    face: FaceStrandTeacherV1,
) -> FTopologyResidualReportV1:
    """Decide whether one face needs explicit topology beyond the F-B scalar core.

    The base assumption is deliberately narrow: one local sheet-conditioned
    bilinear face may carry ordinary boundary-to-boundary arcs, with at most two
    such strands per local patch. Interior loops, interior endpoints, branch/
    non-manifold topology, a strand spanning multiple local patches, or more than
    two strands for one local patch activate the explicit topology-residual court.

    Point/coplanar degeneracies are kept separate as teacher-canonicalization
    concerns rather than being silently promoted to model-capacity requirements.
    """
    cell.validate()
    face.validate()

    triangle_to_patch: dict[int, int] = {}
    for patch in cell.local_patch_lower_bound:
        for triangle_id in patch.source_triangle_ids:
            if (
                triangle_id in triangle_to_patch
                and triangle_to_patch[triangle_id] != patch.patch_local_id
            ):
                raise ValueError("F_TOPOLOGY_RESIDUAL_TRIANGLE_MULTI_PATCH")
            triangle_to_patch[triangle_id] = patch.patch_local_id

    topology_reasons: set[str] = set()
    canonicalization_reasons: set[str] = set()
    patch_strand_counts: dict[int, int] = {}

    for strand in face.strands:
        patch_ids = {
            triangle_to_patch[triangle_id]
            for triangle_id in strand.source_triangle_ids
            if triangle_id in triangle_to_patch
        }
        if not patch_ids:
            raise ValueError("F_TOPOLOGY_RESIDUAL_STRAND_PATCH_UNRESOLVED")
        if len(patch_ids) > 1:
            topology_reasons.add("STRAND_MULTI_PATCH")
        for patch_id in patch_ids:
            patch_strand_counts[patch_id] = patch_strand_counts.get(patch_id, 0) + 1
        if strand.kind != "BOUNDARY_TO_BOUNDARY_ARC":
            topology_reasons.add(strand.kind)

    if any(count > 2 for count in patch_strand_counts.values()):
        topology_reasons.add("PATCH_GT2_STRANDS")
    if face.coplanar_triangle_ids:
        canonicalization_reasons.add("COPLANAR_FACE_SUPPORT")
    if face.point_degeneracy_triangle_ids:
        canonicalization_reasons.add("POINT_DEGENERACY")

    result = FTopologyResidualReportV1(
        schema_version=SCHEMA_VERSION,
        face_index=face.face_index,
        strand_count=len(face.strands),
        patch_strand_counts=tuple(sorted(patch_strand_counts.items())),
        topology_residual_reasons=tuple(sorted(topology_reasons)),
        canonicalization_reasons=tuple(sorted(canonicalization_reasons)),
    )
    result.validate()
    return result
