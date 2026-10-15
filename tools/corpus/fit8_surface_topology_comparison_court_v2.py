from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from tools.corpus.fit8_surface_topology_invariants_v1 import SurfaceTopologyReportV1

SCHEMA_VERSION = "RealSaS.SurfaceTopologyComparisonCourt.v2"


@dataclass(frozen=True, order=True)
class ComponentCorrespondenceV2:
    source_component_id: int
    candidate_component_id: int

    def validate(self) -> None:
        if self.source_component_id < 0 or self.candidate_component_id < 0:
            raise ValueError("SURFACE_TOPOLOGY_COMPARISON_V2_COMPONENT_ID_INVALID")


@dataclass(frozen=True)
class TopologyInvariantMismatchV2:
    source_component_id: int
    candidate_component_id: int
    field: str
    source_value: object
    candidate_value: object


@dataclass(frozen=True)
class SurfaceTopologyComparisonReportV2:
    schema_version: str
    source_component_count: int
    candidate_component_count: int
    correspondence_count: int
    mismatches: tuple[TopologyInvariantMismatchV2, ...]
    requires_boundary_incidence_court: bool
    requires_nonmanifold_incidence_court: bool

    @property
    def qualified_topology_invariants_match(self) -> bool:
        return (
            self.source_component_count == self.candidate_component_count
            and self.correspondence_count == self.source_component_count
            and not self.mismatches
        )

    @property
    def exact_topology_invariants(self) -> bool:
        return (
            self.qualified_topology_invariants_match
            and not self.requires_boundary_incidence_court
            and not self.requires_nonmanifold_incidence_court
        )

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("SURFACE_TOPOLOGY_COMPARISON_V2_SCHEMA_INVALID")
        for name in ("source_component_count", "candidate_component_count", "correspondence_count"):
            if getattr(self, name) < 0:
                raise ValueError(f"SURFACE_TOPOLOGY_COMPARISON_V2_NEGATIVE:{name}")


def compare_surface_topology_invariants_v2(
    source: SurfaceTopologyReportV1,
    candidate: SurfaceTopologyReportV1,
    correspondence: Sequence[ComponentCorrespondenceV2],
) -> SurfaceTopologyComparisonReportV2:
    """Compare topology while allowing legitimate retessellation.

    Discretization counts are deliberately not topology authority. In particular,
    vertex/edge/face counts, raw boundary-edge counts, and raw non-manifold-edge
    counts may change under legal subdivision/retessellation.

    The court compares topology-bearing invariants instead:

    - connected component correspondence;
    - Euler characteristic;
    - presence/absence of boundary;
    - qualified boundary-component/loop topology;
    - edge-manifold versus non-manifold state;
    - orientability qualification/value;
    - genus qualification/value.

    If a component is non-manifold, this court cannot establish exact singular-set
    equivalence from a raw edge count. A separate authored/source-lineage incidence
    court is required and `exact_topology_invariants` remains false until that
    external proof exists.

    Likewise, a boundary graph whose loop topology is unqualified requires a
    separate boundary-incidence court rather than a segment-count comparison.
    """
    source.validate()
    candidate.validate()
    rows = tuple(correspondence)
    for row in rows:
        row.validate()

    source_ids = [row.source_component_id for row in rows]
    candidate_ids = [row.candidate_component_id for row in rows]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("SURFACE_TOPOLOGY_COMPARISON_V2_SOURCE_MAPPING_NOT_INJECTIVE")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("SURFACE_TOPOLOGY_COMPARISON_V2_CANDIDATE_MAPPING_NOT_INJECTIVE")
    if any(i >= source.component_count for i in source_ids):
        raise ValueError("SURFACE_TOPOLOGY_COMPARISON_V2_SOURCE_MAPPING_OOB")
    if any(i >= candidate.component_count for i in candidate_ids):
        raise ValueError("SURFACE_TOPOLOGY_COMPARISON_V2_CANDIDATE_MAPPING_OOB")

    mismatches: list[TopologyInvariantMismatchV2] = []
    requires_boundary_incidence_court = False
    requires_nonmanifold_incidence_court = False

    def record(src_id: int, cand_id: int, field: str, src_value: object, cand_value: object) -> None:
        if src_value != cand_value:
            mismatches.append(
                TopologyInvariantMismatchV2(src_id, cand_id, field, src_value, cand_value)
            )

    for row in rows:
        src = source.components[row.source_component_id]
        cand = candidate.components[row.candidate_component_id]
        sid, cid = row.source_component_id, row.candidate_component_id

        record(sid, cid, "euler_characteristic", src.euler_characteristic, cand.euler_characteristic)

        src_has_boundary = src.boundary_edge_count > 0
        cand_has_boundary = cand.boundary_edge_count > 0
        record(sid, cid, "has_boundary", src_has_boundary, cand_has_boundary)

        src_boundary_valid = src.boundary_loop_count is not None
        cand_boundary_valid = cand.boundary_loop_count is not None
        record(sid, cid, "boundary_loop_valid", src_boundary_valid, cand_boundary_valid)
        if src_boundary_valid and cand_boundary_valid:
            record(
                sid,
                cid,
                "boundary_graph_component_count",
                src.boundary_graph_component_count,
                cand.boundary_graph_component_count,
            )
            record(sid, cid, "boundary_loop_count", src.boundary_loop_count, cand.boundary_loop_count)
        elif src_has_boundary or cand_has_boundary:
            requires_boundary_incidence_court = True

        src_edge_manifold = src.nonmanifold_edge_count == 0
        cand_edge_manifold = cand.nonmanifold_edge_count == 0
        record(sid, cid, "edge_manifold", src_edge_manifold, cand_edge_manifold)
        if not src_edge_manifold or not cand_edge_manifold:
            requires_nonmanifold_incidence_court = True

        record(sid, cid, "orientability_valid", src.orientability_valid, cand.orientability_valid)
        if src.orientability_valid and cand.orientability_valid:
            record(sid, cid, "orientable", src.orientable, cand.orientable)

        record(sid, cid, "genus_valid", src.genus_valid, cand.genus_valid)
        if src.genus_valid and cand.genus_valid:
            record(sid, cid, "genus", src.genus, cand.genus)

    if source.component_count != candidate.component_count:
        mismatches.append(
            TopologyInvariantMismatchV2(
                -1,
                -1,
                "component_count",
                source.component_count,
                candidate.component_count,
            )
        )
    if len(rows) != source.component_count or len(rows) != candidate.component_count:
        mismatches.append(
            TopologyInvariantMismatchV2(
                -1,
                -1,
                "component_correspondence_coverage",
                (source.component_count, candidate.component_count),
                len(rows),
            )
        )

    report = SurfaceTopologyComparisonReportV2(
        schema_version=SCHEMA_VERSION,
        source_component_count=source.component_count,
        candidate_component_count=candidate.component_count,
        correspondence_count=len(rows),
        mismatches=tuple(mismatches),
        requires_boundary_incidence_court=requires_boundary_incidence_court,
        requires_nonmanifold_incidence_court=requires_nonmanifold_incidence_court,
    )
    report.validate()
    return report


__all__ = [
    "ComponentCorrespondenceV2",
    "SCHEMA_VERSION",
    "SurfaceTopologyComparisonReportV2",
    "TopologyInvariantMismatchV2",
    "compare_surface_topology_invariants_v2",
]
