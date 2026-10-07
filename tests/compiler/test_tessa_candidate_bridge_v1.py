from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mechanical_partition_v1 import build_structural_partition
from compiler.realsas_compiler_core.product_authority_v1 import (
    ComponentCarrierDecisionIR,
    build_component_carrier_policy,
)
from compiler.realsas_compiler_core.tessa_candidate_bridge_v1 import (
    build_tessa_candidate_bridge_v1,
    tessa_candidate_bridge_evidence_hash_v1,
    validate_tessa_candidate_bridge_evidence_v1,
)
from compiler.realsas_compiler_core.tessa_surface_field_v1 import (
    build_tessa_material_support_field_v1,
)
from compiler.realsas_compiler_core.types import (
    QualificationError,
    RiggingSurfaceIR,
    SurfaceNode,
    SurfaceRelation,
)
from models.tessa.v1.contracts_v1 import TESSAMeshProposalV1, TESSAVertexProposalV1


def _node(surface_id: str, p):
    return SurfaceNode(
        surface_id=surface_id,
        P=tuple(map(float, p)),
        support_views=(0, 1),
        provenance_refs=(f"prov:{surface_id}",),
        source_observation_ids=(f"obs:{surface_id}",),
        derived_normal=(0.0, 0.0, 1.0),
        validity_flags=("OBSERVED_SIGNED_ZERO_SURFACE",),
    )


def _surface() -> RiggingSurfaceIR:
    nodes = (
        _node("a", (0.0, 0.0, 0.0)),
        _node("b", (1.0, 0.0, 0.0)),
        _node("c", (0.0, 1.0, 0.0)),
    )
    relations = (
        SurfaceRelation("ab", "a", "b", "SIGNED_ZERO_SURFACE_TOPOLOGY_NEIGHBOR", 1.0),
        SurfaceRelation("bc", "b", "c", "SIGNED_ZERO_SURFACE_TOPOLOGY_NEIGHBOR", 1.0),
        SurfaceRelation("ca", "c", "a", "SIGNED_ZERO_SURFACE_TOPOLOGY_NEIGHBOR", 1.0),
    )
    return RiggingSurfaceIR(
        surface_nodes=nodes,
        local_relations=relations,
        geometry_lineage_hash="surface-v1",
    )


def _context():
    surface = _surface()
    partition = build_structural_partition(surface)
    decisions = tuple(
        ComponentCarrierDecisionIR(
            component.component_id,
            "MESH",
            ("TEST_CONSERVATIVE_MESH",),
        )
        for component in partition.components
    )
    carrier = build_component_carrier_policy(
        partition=partition,
        decisions=decisions,
        metadata={"test": True},
    )
    return surface, partition, carrier


def _proposal(surface: RiggingSurfaceIR) -> TESSAMeshProposalV1:
    # The learned geometry deliberately sits away from the GSA plane.  This is the
    # exact case the bridge must preserve without fabricating convex-lift geometry.
    vertices = (
        TESSAVertexProposalV1("v0", (0.15, 0.15, 0.20), "a", "decoded:0"),
        TESSAVertexProposalV1("v1", (0.75, 0.10, 0.18), "b", "decoded:0"),
        TESSAVertexProposalV1("v2", (0.10, 0.75, 0.22), "c", "decoded:0"),
    )
    return TESSAMeshProposalV1(
        vertices=vertices,
        faces=(("v0", "v1", "v2"),),
        source_geometry_lineage_hash=surface.geometry_lineage_hash,
        model_provenance="checkpoint:test",
        topology_sequence_hash="topology:test",
        metadata={"authority_class": "LEARNED_PROPOSAL"},
    )


def _field(proposal, surface, partition):
    vertices = np.asarray([vertex.P for vertex in proposal.vertices], dtype=np.float64)
    return build_tessa_material_support_field_v1(
        vertices_world=vertices,
        decoded_component_indices=np.zeros(len(vertices), dtype=np.int64),
        surface=surface,
        partition=partition,
        topology_sequence_hash=proposal.topology_sequence_hash,
        proposal_vertex_ids=tuple(vertex.proposal_vertex_id for vertex in proposal.vertices),
        max_support_nodes=3,
        max_graph_hops=2,
    )


def test_bridge_preserves_learned_xyz_and_keeps_material_support_non_geometric():
    surface, partition, carrier = _context()
    proposal = _proposal(surface)
    field = _field(proposal, surface, partition)

    candidate, evidence = build_tessa_candidate_bridge_v1(
        proposal=proposal,
        support_field=field,
        surface=surface,
        partition=partition,
        carrier_policy=carrier,
        producer_policy_hash=content_sha256({"policy": "test"}),
    )

    by_proposal_id = {
        vertex.metadata["proposal_vertex_id"]: vertex for vertex in candidate.vertices
    }
    for proposed in proposal.vertices:
        candidate_vertex = by_proposal_id[proposed.proposal_vertex_id]
        assert candidate_vertex.P == proposed.P
        assert candidate_vertex.refinement is None
        assert candidate_vertex.support_binding.metadata["authority_class"] == "MATERIAL_SUPPORT_ONLY"
        assert candidate_vertex.metadata["material_support_is_not_geometry_support"] is True
        assert candidate_vertex.metadata["geometry_position_derived_from_material_support"] is False

    # Learned Z remains nonzero even though every GSA support node lies at z=0.
    assert min(vertex.P[2] for vertex in candidate.vertices) > 0.0
    assert candidate.metadata["learned_xyz_preserved_exactly"] is True
    assert candidate.metadata["legacy_g1_convex_lift_claimed"] is False
    assert candidate.metadata["product_geometry_authority_claimed"] is False
    assert evidence.metadata["requires_stage19_static_qualification"] is True
    assert evidence.metadata["distance_metrics_are_diagnostic_not_acceptance_thresholds"] is True


def test_bridge_fails_closed_on_source_geometry_lineage_drift():
    surface, partition, carrier = _context()
    proposal = replace(_proposal(surface), source_geometry_lineage_hash="wrong-surface")
    field = _field(replace(proposal, source_geometry_lineage_hash=surface.geometry_lineage_hash), surface, partition)

    with pytest.raises(QualificationError, match="TESSA_CANDIDATE_SOURCE_GEOMETRY_LINEAGE_MISMATCH"):
        build_tessa_candidate_bridge_v1(
            proposal=proposal,
            support_field=field,
            surface=surface,
            partition=partition,
            carrier_policy=carrier,
            producer_policy_hash="policy-hash",
        )


def test_bridge_fails_closed_on_topology_sequence_drift():
    surface, partition, carrier = _context()
    proposal = _proposal(surface)
    field = _field(proposal, surface, partition)
    bad_field = replace(field, topology_sequence_hash="other", field_lineage_hash="")
    from compiler.realsas_compiler_core.tessa_surface_field_v1 import (
        tessa_material_support_field_lineage_hash,
    )
    bad_field = replace(
        bad_field,
        field_lineage_hash=tessa_material_support_field_lineage_hash(bad_field),
    )

    with pytest.raises(QualificationError, match="TESSA_CANDIDATE_TOPOLOGY_SEQUENCE_MISMATCH"):
        build_tessa_candidate_bridge_v1(
            proposal=proposal,
            support_field=bad_field,
            surface=surface,
            partition=partition,
            carrier_policy=carrier,
            producer_policy_hash="policy-hash",
        )


def test_bridge_evidence_hash_is_fail_closed():
    surface, partition, carrier = _context()
    proposal = _proposal(surface)
    field = _field(proposal, surface, partition)
    _, evidence = build_tessa_candidate_bridge_v1(
        proposal=proposal,
        support_field=field,
        surface=surface,
        partition=partition,
        carrier_policy=carrier,
        producer_policy_hash="policy-hash",
    )
    bad = replace(evidence, nearest_surface_distance_max=evidence.nearest_surface_distance_max + 1.0)
    assert tessa_candidate_bridge_evidence_hash_v1(bad) != bad.evidence_hash
    with pytest.raises(QualificationError, match="TESSA_CANDIDATE_EVIDENCE_HASH_MISMATCH"):
        validate_tessa_candidate_bridge_evidence_v1(
            bad,
            proposal=proposal,
            support_field=field,
            surface=surface,
            partition=partition,
            carrier_policy=carrier,
        )
