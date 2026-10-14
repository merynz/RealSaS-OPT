from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from tools.corpus.fit8_local_patch_teacher_v1 import ExtractionCellTeacherV1

SCHEMA_VERSION = "RealSaS.FRepresentationCapacityCourt.v1"


@dataclass(frozen=True)
class FRepresentationCapacityReportV1:
    schema_version: str
    cell_count: int
    active_cell_count: int
    teacher_crossing_count: int
    f_a_capacity_crossing_count: int
    f_b_patch_crossing_count: int
    f_b_capacity_crossing_count: int
    f_c_capacity_crossing_count: int
    f_a_unmatched_crossings: int
    f_b_unmatched_patch_crossings: int
    cells_requiring_more_than_f_a: int
    cells_requiring_more_than_f_b: int
    max_teacher_crossings_per_edge: int
    max_teacher_crossings_per_patch_edge: int

    @property
    def f_a_recall_ceiling(self) -> float:
        return self.f_a_capacity_crossing_count / max(1, self.teacher_crossing_count)

    @property
    def f_b_recall_ceiling(self) -> float:
        return self.f_b_capacity_crossing_count / max(1, self.f_b_patch_crossing_count)

    @property
    def f_c_recall_ceiling(self) -> float:
        return self.f_c_capacity_crossing_count / max(1, self.teacher_crossing_count)

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("F_REPRESENTATION_COURT_SCHEMA_INVALID")
        for name in (
            "cell_count", "active_cell_count", "teacher_crossing_count",
            "f_a_capacity_crossing_count", "f_b_patch_crossing_count",
            "f_b_capacity_crossing_count", "f_c_capacity_crossing_count",
            "f_a_unmatched_crossings", "f_b_unmatched_patch_crossings",
            "cells_requiring_more_than_f_a", "cells_requiring_more_than_f_b",
            "max_teacher_crossings_per_edge", "max_teacher_crossings_per_patch_edge",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"F_REPRESENTATION_COURT_NEGATIVE:{name}")
        if self.active_cell_count > self.cell_count:
            raise ValueError("F_REPRESENTATION_COURT_ACTIVE_GT_TOTAL")
        if self.f_a_capacity_crossing_count > self.teacher_crossing_count:
            raise ValueError("F_REPRESENTATION_COURT_FA_OVERCLAIM")
        if self.f_b_capacity_crossing_count > self.f_b_patch_crossing_count:
            raise ValueError("F_REPRESENTATION_COURT_FB_OVERCLAIM")
        if self.f_c_capacity_crossing_count != self.teacher_crossing_count:
            raise ValueError("F_REPRESENTATION_COURT_FC_LOCAL_SET_MISMATCH")


def evaluate_f_representation_capacity_v1(
    cells: Iterable[ExtractionCellTeacherV1],
) -> FRepresentationCapacityReportV1:
    """Measure local crossing representability without training a scene model.

    F-A: one unconditioned crossing per cell edge.
    F-B: one crossing per local-patch hypothesis per cell edge.
    F-C: explicit variable crossing set; local set capacity only. Cross-cell
         assembly/continuation is deliberately outside this court.

    Local patch membership is recovered only from source-triangle bindings in
    ExtractionCellTeacherV1. No component/material/owner equality is assumed.
    """
    cell_count = active_count = 0
    teacher_crossings = fa_capacity = 0
    fb_patch_crossings = fb_capacity = 0
    fa_unmatched = fb_unmatched = 0
    cells_more_fa = cells_more_fb = 0
    max_edge = max_patch_edge = 0

    for cell in cells:
        cell.validate()
        cell_count += 1
        if cell.active:
            active_count += 1

        tri_to_patch: dict[int, int] = {}
        for patch in cell.local_patch_lower_bound:
            for tid in patch.source_triangle_ids:
                if tid in tri_to_patch and tri_to_patch[tid] != patch.patch_local_id:
                    raise ValueError("F_REPRESENTATION_COURT_TRIANGLE_MULTI_PATCH")
                tri_to_patch[tid] = patch.patch_local_id

        by_edge: dict[int, list] = {}
        for crossing in cell.edge_crossings:
            by_edge.setdefault(crossing.edge_index, []).append(crossing)

        cell_more_fa = False
        cell_more_fb = False
        for edge_crossings in by_edge.values():
            m = len(edge_crossings)
            teacher_crossings += m
            fa_capacity += min(1, m)
            fa_unmatched += max(0, m - 1)
            max_edge = max(max_edge, m)
            if m > 1:
                cell_more_fa = True

            patch_counts: dict[int, int] = {}
            for crossing in edge_crossings:
                patch_ids = {
                    tri_to_patch[tid]
                    for tid in crossing.source_triangle_ids
                    if tid in tri_to_patch
                }
                if not patch_ids:
                    raise ValueError("F_REPRESENTATION_COURT_CROSSING_PATCH_UNRESOLVED")
                for pid in patch_ids:
                    patch_counts[pid] = patch_counts.get(pid, 0) + 1
            for k in patch_counts.values():
                fb_patch_crossings += k
                fb_capacity += min(1, k)
                fb_unmatched += max(0, k - 1)
                max_patch_edge = max(max_patch_edge, k)
                if k > 1:
                    cell_more_fb = True

        cells_more_fa += int(cell_more_fa)
        cells_more_fb += int(cell_more_fb)

    report = FRepresentationCapacityReportV1(
        schema_version=SCHEMA_VERSION,
        cell_count=cell_count,
        active_cell_count=active_count,
        teacher_crossing_count=teacher_crossings,
        f_a_capacity_crossing_count=fa_capacity,
        f_b_patch_crossing_count=fb_patch_crossings,
        f_b_capacity_crossing_count=fb_capacity,
        f_c_capacity_crossing_count=teacher_crossings,
        f_a_unmatched_crossings=fa_unmatched,
        f_b_unmatched_patch_crossings=fb_unmatched,
        cells_requiring_more_than_f_a=cells_more_fa,
        cells_requiring_more_than_f_b=cells_more_fb,
        max_teacher_crossings_per_edge=max_edge,
        max_teacher_crossings_per_patch_edge=max_patch_edge,
    )
    report.validate()
    return report
