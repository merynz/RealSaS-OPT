from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from tools.corpus.fit8_f_representation_capacity_court_v1 import FRepresentationCapacityReportV1
from tools.corpus.fit8_f_topology_residual_court_v1 import FTopologyResidualReportV1

SCHEMA_VERSION = "RealSaS.FResidualBudgetCourt.v1"


@dataclass(frozen=True)
class FResidualBudgetReportV1:
    schema_version: str
    active_cell_count: int
    edge_residual_cells: int
    edge_extra_crossings: int
    teacher_patch_edge_crossings: int
    face_count: int
    topology_residual_faces: int
    canonicalization_faces: int
    teacher_face_strands: int
    residual_face_strands: int

    @property
    def edge_residual_cell_fraction(self) -> float:
        return self.edge_residual_cells / max(1, self.active_cell_count)

    @property
    def edge_extra_crossing_fraction(self) -> float:
        return self.edge_extra_crossings / max(1, self.teacher_patch_edge_crossings)

    @property
    def topology_residual_face_fraction(self) -> float:
        return self.topology_residual_faces / max(1, self.face_count)

    @property
    def face_residual_strand_fraction(self) -> float:
        return self.residual_face_strands / max(1, self.teacher_face_strands)

    @property
    def explicit_topology_record_fraction(self) -> float:
        numerator = self.edge_extra_crossings + self.residual_face_strands
        denominator = self.teacher_patch_edge_crossings + self.teacher_face_strands
        return numerator / max(1, denominator)

    @property
    def canonicalization_face_fraction(self) -> float:
        return self.canonicalization_faces / max(1, self.face_count)

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("F_RESIDUAL_BUDGET_SCHEMA_INVALID")
        for name in (
            "active_cell_count",
            "edge_residual_cells",
            "edge_extra_crossings",
            "teacher_patch_edge_crossings",
            "face_count",
            "topology_residual_faces",
            "canonicalization_faces",
            "teacher_face_strands",
            "residual_face_strands",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"F_RESIDUAL_BUDGET_NEGATIVE:{name}")
        if self.edge_residual_cells > self.active_cell_count:
            raise ValueError("F_RESIDUAL_BUDGET_EDGE_CELL_OVERCLAIM")
        if self.topology_residual_faces > self.face_count:
            raise ValueError("F_RESIDUAL_BUDGET_TOPOLOGY_FACE_OVERCLAIM")
        if self.canonicalization_faces > self.face_count:
            raise ValueError("F_RESIDUAL_BUDGET_CANONICALIZATION_FACE_OVERCLAIM")
        if self.residual_face_strands > self.teacher_face_strands:
            raise ValueError("F_RESIDUAL_BUDGET_STRAND_OVERCLAIM")


def evaluate_f_residual_budget_v1(
    edge_capacity: FRepresentationCapacityReportV1,
    face_reports: Iterable[FTopologyResidualReportV1],
) -> FResidualBudgetReportV1:
    """Keep model-capacity residuals separate from teacher canonicalization work.

    `explicit_topology_record_fraction` is only a representation-count diagnostic:
    one edge-crossing record and one face-strand record are counted equally. It is
    not a byte-accurate memory or runtime estimate.
    """
    edge_capacity.validate()
    faces = list(face_reports)
    for face in faces:
        face.validate()
    result = FResidualBudgetReportV1(
        schema_version=SCHEMA_VERSION,
        active_cell_count=edge_capacity.active_cell_count,
        edge_residual_cells=edge_capacity.cells_requiring_more_than_f_b,
        edge_extra_crossings=edge_capacity.f_b_unmatched_patch_crossings,
        teacher_patch_edge_crossings=edge_capacity.f_b_patch_crossing_count,
        face_count=len(faces),
        topology_residual_faces=sum(face.topology_residual_required for face in faces),
        canonicalization_faces=sum(face.canonicalization_required for face in faces),
        teacher_face_strands=sum(face.strand_count for face in faces),
        residual_face_strands=sum(
            face.strand_count for face in faces if face.topology_residual_required
        ),
    )
    result.validate()
    return result
