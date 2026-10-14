from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from tools.corpus.fit8_surface_topology_invariants_v1 import SurfaceTopologyReportV1

SCHEMA_VERSION = "RealSaS.SurfaceTopologyComparisonCourt.v1"


@dataclass(frozen=True, order=True)
class ComponentCorrespondenceV1:
    source_component_id: int
    candidate_component_id: int

    def validate(self) -> None:
        if self.source_component_id < 0 or self.candidate_component_id < 0:
            raise ValueError("SURFACE_TOPOLOGY_COMPARISON_COMPONENT_ID_INVALID")


@dataclass(frozen=True)
class TopologyInvariantMismatchV1:
    source_component_id: int
    candidate_component_id: int
    field: str
    source_value: object
    candidate_value: object


@dataclass(frozen=True)
class SurfaceTopologyComparisonReportV1:
    schema_version: str
    source_component_count: int
    candidate_component_count: int
    correspondence_count: int
    mismatches: tuple[TopologyInvariantMismatchV1, ...]

    @property
    def exact_topology_invariants(self) -> bool:
        return (
            self.source_component_count == self.candidate_component_count
            and self.correspondence_count == self.source_component_count
            and not self.mismatches
        )

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("SURFACE_TOPOLOGY_COMPARISON_SCHEMA_INVALID")
        for name in ("source_component_count", "candidate_component_count", "correspondence_count"):
            if getattr(self, name) < 0:
                raise ValueError(f"SURFACE_TOPOLOGY_COMPARISON_NEGATIVE:{name}")


def compare_surface_topology_invariants_v1(
    source: SurfaceTopologyReportV1,
    candidate: SurfaceTopologyReportV1,
    correspondence: Sequence[ComponentCorrespondenceV1],
) -> SurfaceTopologyComparisonReportV1:
    """Compare topology while allowing legitimate retessellation.

    Vertex/edge/face counts are intentionally not required to match: a candidate
    may retessellate one authored surface while preserving its topology. The court
    instead compares topology-bearing invariants and their qualification state.

    Validity/abstention state is itself part of the contract. If source genus or
    orientability is not authoritative, the court does not compare the numerical
    value, but a candidate becoming qualified where the source was unqualified is
    still a topology/precondition change and is recorded rather than silently
    accepted.
    """
    source.validate()
    candidate.validate()
    rows = tuple(correspondence)
    for row in rows:
        row.validate()
    source_ids = [row.source_component_id for row in rows]
    candidate_ids = [row.candidate_component_id for row in rows]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("SURFACE_TOPOLOGY_COMPARISON_SOURCE_MAPPING_NOT_INJECTIVE")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("SURFACE_TOPOLOGY_COMPARISON_CANDIDATE_MAPPING_NOT_INJECTIVE")
    if any(i >= source.component_count for i in source_ids):
        raise ValueError("SURFACE_TOPOLOGY_COMPARISON_SOURCE_MAPPING_OOB")
    if any(i >= candidate.component_count for i in candidate_ids):
        raise ValueError("SURFACE_TOPOLOGY_COMPARISON_CANDIDATE_MAPPING_OOB")

    mismatches: list[TopologyInvariantMismatchV1] = []

    def record(src_id: int, cand_id: int, field: str, src_value: object, cand_value: object) -> None:
        if src_value != cand_value:
            mismatches.append(
                TopologyInvariantMismatchV1(src_id, cand_id, field, src_value, cand_value)
            )

    for row in rows:
        src = source.components[row.source_component_id]
        cand = candidate.components[row.candidate_component_id]
        sid, cid = row.source_component_id, row.candidate_component_id

        record(sid, cid, "euler_characteristic", src.euler_characteristic, cand.euler_characteristic)
        record(sid, cid, "boundary_edge_count", src.boundary_edge_count, cand.boundary_edge_count)
        record(
            sid,
            cid,
            "boundary_graph_component_count",
            src.boundary_graph_component_count,
            cand.boundary_graph_component_count,
        )
        record(sid, cid, "nonmanifold_edge_count", src.nonmanifold_edge_count, cand.nonmanifold_edge_count)

        src_boundary_valid = src.boundary_loop_count is not None
        cand_boundary_valid = cand.boundary_loop_count is not None
        record(sid, cid, "boundary_loop_valid", src_boundary_valid, cand_boundary_valid)
        if src_boundary_valid and cand_boundary_valid:
            record(sid, cid, "boundary_loop_count", src.boundary_loop_count, cand.boundary_loop_count)

        record(sid, cid, "orientability_valid", src.orientability_valid, cand.orientability_valid)
        if src.orientability_valid and cand.orientability_valid:
            record(sid, cid, "orientable", src.orientable, cand.orientable)

        record(sid, cid, "genus_valid", src.genus_valid, cand.genus_valid)
        if src.genus_valid and cand.genus_valid:
            record(sid, cid, "genus", src.genus, cand.genus)

    if source.component_count != candidate.component_count:
        mismatches.append(
            TopologyInvariantMismatchV1(
                -1,
                -1,
                "component_count",
                source.component_count,
                candidate.component_count,
            )
        )
    if len(rows) != source.component_count or len(rows) != candidate.component_count:
        mismatches.append(
            TopologyInvariantMismatchV1(
                -1,
                -1,
                "component_correspondence_coverage",
                (source.component_count, candidate.component_count),
                len(rows),
            )
        )

    report = SurfaceTopologyComparisonReportV1(
        schema_version=SCHEMA_VERSION,
        source_component_count=source.component_count,
        candidate_component_count=candidate.component_count,
        correspondence_count=len(rows),
        mismatches=tuple(mismatches),
    )
    report.validate()
    return report
