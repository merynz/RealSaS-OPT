from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from tools.corpus.fit8_cell_supported_patch_qualification_v1 import (
    CellSupportedPatchQualificationV1,
)
from tools.corpus.fit8_local_patch_teacher_v1 import ExtractionCellTeacherV1

SCHEMA_VERSION = "RealSaS.FRepresentationCapacityCourt.v2"


@dataclass(frozen=True)
class QualifiedCellCapacityInputV2:
    teacher: ExtractionCellTeacherV1
    patch_qualification: CellSupportedPatchQualificationV1

    def validate(self) -> None:
        self.teacher.validate()
        self.patch_qualification.validate(self.teacher)


@dataclass(frozen=True)
class FRepresentationCapacityReportV2:
    schema_version: str
    cell_count: int
    active_cell_count: int
    teacher_crossing_count: int
    qualified_patch_crossing_membership_count: int
    f_a_capacity_crossing_count: int
    f_b_capacity_patch_crossing_count: int
    f_a_unmatched_crossings: int
    f_b_unmatched_patch_crossings: int
    cells_requiring_more_than_f_a: int
    cells_requiring_f_r_edge_residual: int
    f_r_explicit_extra_crossing_payload: int
    f_c_explicit_crossing_payload: int
    max_teacher_crossings_per_edge: int
    max_teacher_crossings_per_qualified_patch_edge: int

    @property
    def f_a_recall_ceiling(self) -> float:
        return self.f_a_capacity_crossing_count / max(1, self.teacher_crossing_count)

    @property
    def f_b_recall_ceiling(self) -> float:
        return self.f_b_capacity_patch_crossing_count / max(
            1, self.qualified_patch_crossing_membership_count
        )

    @property
    def f_r_local_recall_ceiling_with_perfect_activation(self) -> float:
        # F-R is defined here as F-B plus an explicit variable-cardinality residual
        # on exactly the qualified patch-edge memberships that exceed F-B capacity.
        return 1.0

    @property
    def f_r_explicit_payload_fraction_of_f_c(self) -> float:
        return self.f_r_explicit_extra_crossing_payload / max(1, self.f_c_explicit_crossing_payload)

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("F_REPRESENTATION_COURT_V2_SCHEMA_INVALID")
        for name in (
            "cell_count",
            "active_cell_count",
            "teacher_crossing_count",
            "qualified_patch_crossing_membership_count",
            "f_a_capacity_crossing_count",
            "f_b_capacity_patch_crossing_count",
            "f_a_unmatched_crossings",
            "f_b_unmatched_patch_crossings",
            "cells_requiring_more_than_f_a",
            "cells_requiring_f_r_edge_residual",
            "f_r_explicit_extra_crossing_payload",
            "f_c_explicit_crossing_payload",
            "max_teacher_crossings_per_edge",
            "max_teacher_crossings_per_qualified_patch_edge",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"F_REPRESENTATION_COURT_V2_NEGATIVE:{name}")
        if self.active_cell_count > self.cell_count:
            raise ValueError("F_REPRESENTATION_COURT_V2_ACTIVE_GT_TOTAL")
        if self.f_a_capacity_crossing_count > self.teacher_crossing_count:
            raise ValueError("F_REPRESENTATION_COURT_V2_FA_OVERCLAIM")
        if self.f_b_capacity_patch_crossing_count > self.qualified_patch_crossing_membership_count:
            raise ValueError("F_REPRESENTATION_COURT_V2_FB_OVERCLAIM")
        if self.f_r_explicit_extra_crossing_payload != self.f_b_unmatched_patch_crossings:
            raise ValueError("F_REPRESENTATION_COURT_V2_FR_PAYLOAD_MISMATCH")
        if self.f_c_explicit_crossing_payload != self.teacher_crossing_count:
            raise ValueError("F_REPRESENTATION_COURT_V2_FC_PAYLOAD_MISMATCH")


def evaluate_f_representation_capacity_v2(
    cells: Iterable[QualifiedCellCapacityInputV2],
) -> FRepresentationCapacityReportV2:
    """Measure local edge-crossing capacity under qualified cell-supported sheets.

    Representation meanings in this court:

    F-A
        one unconditioned crossing per cell edge.

    F-B
        one crossing per *qualified cell-supported local sheet* per cell edge.
        This deliberately does not use ExtractionCellTeacherV1.local_patch_lower_bound,
        which is only a coarse diagnostic and can over-merge sheets whose shared
        authored edge lies outside the cell.

    F-R
        F-B plus explicit extra crossing rows only where one qualified local sheet
        crosses one cell edge more than once. Local recall reaches F-C only under
        perfect residual activation; learned activation is a separate fail-closed
        court and must not inherit this oracle ceiling as a model PASS.

    F-C
        full explicit variable crossing set on every cell edge. This court measures
        only local crossing-set capacity; face topology, cross-cell assembly,
        geometry fidelity, dynamics and CAO remain separate courts.
    """
    cell_count = 0
    active_count = 0
    teacher_crossings = 0
    qualified_memberships = 0
    fa_capacity = 0
    fb_capacity = 0
    fa_unmatched = 0
    fb_unmatched = 0
    cells_more_fa = 0
    cells_need_fr = 0
    max_edge = 0
    max_patch_edge = 0

    for record in cells:
        record.validate()
        teacher = record.teacher
        qualification = record.patch_qualification
        cell_count += 1
        active_count += int(teacher.active)

        tri_to_patch: dict[int, int] = {}
        for patch in qualification.patches:
            for triangle_id in patch.source_triangle_ids:
                if triangle_id in tri_to_patch and tri_to_patch[triangle_id] != patch.patch_local_id:
                    raise ValueError("F_REPRESENTATION_COURT_V2_TRIANGLE_MULTI_PATCH")
                tri_to_patch[int(triangle_id)] = int(patch.patch_local_id)

        by_edge: dict[int, list] = {}
        for crossing in teacher.edge_crossings:
            by_edge.setdefault(int(crossing.edge_index), []).append(crossing)

        cell_more_fa = False
        cell_need_fr = False
        for edge_crossings in by_edge.values():
            crossing_count = len(edge_crossings)
            teacher_crossings += crossing_count
            fa_capacity += min(1, crossing_count)
            fa_unmatched += max(0, crossing_count - 1)
            max_edge = max(max_edge, crossing_count)
            if crossing_count > 1:
                cell_more_fa = True

            patch_counts: dict[int, int] = {}
            for crossing in edge_crossings:
                patch_ids = {
                    tri_to_patch[int(triangle_id)]
                    for triangle_id in crossing.source_triangle_ids
                    if int(triangle_id) in tri_to_patch
                }
                if not patch_ids:
                    raise ValueError("F_REPRESENTATION_COURT_V2_CROSSING_PATCH_UNRESOLVED")
                for patch_id in patch_ids:
                    patch_counts[patch_id] = patch_counts.get(patch_id, 0) + 1

            for count in patch_counts.values():
                qualified_memberships += count
                fb_capacity += min(1, count)
                fb_unmatched += max(0, count - 1)
                max_patch_edge = max(max_patch_edge, count)
                if count > 1:
                    cell_need_fr = True

        cells_more_fa += int(cell_more_fa)
        cells_need_fr += int(cell_need_fr)

    report = FRepresentationCapacityReportV2(
        schema_version=SCHEMA_VERSION,
        cell_count=cell_count,
        active_cell_count=active_count,
        teacher_crossing_count=teacher_crossings,
        qualified_patch_crossing_membership_count=qualified_memberships,
        f_a_capacity_crossing_count=fa_capacity,
        f_b_capacity_patch_crossing_count=fb_capacity,
        f_a_unmatched_crossings=fa_unmatched,
        f_b_unmatched_patch_crossings=fb_unmatched,
        cells_requiring_more_than_f_a=cells_more_fa,
        cells_requiring_f_r_edge_residual=cells_need_fr,
        f_r_explicit_extra_crossing_payload=fb_unmatched,
        f_c_explicit_crossing_payload=teacher_crossings,
        max_teacher_crossings_per_edge=max_edge,
        max_teacher_crossings_per_qualified_patch_edge=max_patch_edge,
    )
    report.validate()
    return report


__all__ = [
    "FRepresentationCapacityReportV2",
    "QualifiedCellCapacityInputV2",
    "SCHEMA_VERSION",
    "evaluate_f_representation_capacity_v2",
]
