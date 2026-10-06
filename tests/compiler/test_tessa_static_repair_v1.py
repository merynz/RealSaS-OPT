from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from compiler.realsas_compiler_core.mechanical_partition_v1 import build_structural_partition
from compiler.realsas_compiler_core.product_authority_v1 import (
    CarrierCoverageThresholdIR,
    ComponentCarrierDecisionIR,
    MeshQualificationPolicyIR,
    build_component_carrier_policy,
)
from compiler.realsas_compiler_core.tessa_candidate_bridge_v1 import (
    build_tessa_candidate_bridge_v1,
)
from compiler.realsas_compiler_core.tessa_static_repair_v1 import (
    repair_tessa_candidate_static_v1,
    validate_tessa_reference_binding_v1,
    validate_tessa_static_repair_evidence_v1,
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


def _surface() -> RiggingSurfaceIR:
    nodes = (
        SurfaceNode("a", (0.0, 0.0, 0.0), (0, 1), ("pa",), ("oa",), derived_normal=(0.0, 0.0, 1.0)),
        SurfaceNode("b", (1.0, 0.0, 0.0), (0, 1), ("pb",), ("ob",), derived_normal=(0.0, 0.0, 1.0)),
        SurfaceNode("c", (0.0, 1.0, 0.0), (0, 1), ("pc",), ("oc",), derived_normal=(0.0, 0.0, 1.0)),
    )
    relations = (
        SurfaceRelation("ab", "a", "b", "SIGNED_ZERO_SURFACE_TOPOLOGY_NEIGHBOR", 1.0),
        SurfaceRelation("bc", "b", "c", "SIGNED_ZERO_SURFACE_TOPOLOGY_NEIGHBOR", 1.0),
        SurfaceRelation("ca", "c", "a", "SIGNED_ZERO_SURFACE_TOPOLOGY_NEIGHBOR", 1.0),
    )
    return RiggingSurfaceIR(nodes, relations, "surface-v1")


def _candidate_context():
    surface = _surface()
    partition = build_structural_partition(surface)
    carrier = build_component_carrier_policy(
        partition=partition,
        decisions=tuple(
            ComponentCarrierDecisionIR(component.component_id, "MESH", ("TEST",))
            for component in partition.components
        ),
    )
    proposal = TESSAMeshProposalV1(
        vertices=(
            TESSAVertexProposalV1("v0", (0.10, 0.10, 0.20), "a", "decoded:0"),
            TESSAVertexProposalV1("v1", (0.85, 0.10, 0.20), "b", "decoded:0"),
            TESSAVertexProposalV1("v2", (0.10, 0.85, 0.20), "c", "decoded:0"),
        ),
        faces=(("v0", "v1", "v2"),),
        source_geometry_lineage_hash=surface.geometry_lineage_hash,
        model_provenance="checkpoint:test",
        topology_sequence_hash="topology:test",
    )
    field = build_tessa_material_support_field_v1(
        vertices_world=np.asarray([vertex.P for vertex in proposal.vertices], dtype=np.float64),
        decoded_component_indices=np.zeros(3, dtype=np.int64),
        surface=surface,
        partition=partition,
        topology_sequence_hash=proposal.topology_sequence_hash,
        proposal_vertex_ids=("v0", "v1", "v2"),
        max_support_nodes=3,
        max_graph_hops=2,
    )
    candidate, _ = build_tessa_candidate_bridge_v1(
        proposal=proposal,
        support_field=field,
        surface=surface,
        partition=partition,
        carrier_policy=carrier,
        producer_policy_hash="proposal-policy-hash",
    )
    return candidate, partition


def _policy() -> MeshQualificationPolicyIR:
    return MeshQualificationPolicyIR(
        0.125,
        0.25,
        7.5,
        16.0,
        (
            CarrierCoverageThresholdIR("MESH", 0.999, 0.999, 0.00025, 0.0005),
            CarrierCoverageThresholdIR("PLANAR", 0.99, 0.995, 0.0025, 0.0025),
        ),
        "mesh-policy-hash",
    )


def test_static_repair_seals_non_circular_candidate_evidence_chain():
    candidate, partition = _candidate_context()
    repaired, evidence, reference = repair_tessa_candidate_static_v1(
        candidate=candidate,
        policy=_policy(),
        partition=partition,
    )

    assert repaired.producer_id == candidate.producer_id
    assert repaired.metadata["compiler_static_repair_applied"] is True
    assert repaired.metadata["learned_proposal_is_immutable_reference"] is True
    assert repaired.metadata["learned_proposal_xyz_preserved_exactly"] is False
    assert repaired.metadata["learned_xyz_preserved_exactly"] is True
    assert repaired.metadata["learned_xyz_preservation_scope"] == "POST_COMPILER_REPAIR_CANDIDATE_TO_DOWNSTREAM"
    assert repaired.metadata["material_support_is_not_geometry_support"] is True
    assert repaired.metadata["teacher_geometry_used_to_repair"] is False
    assert repaired.metadata["threshold_relaxation_performed"] is False
    assert repaired.metadata["tessa_reference_binding_hash"] == reference.binding_hash
    assert repaired.metadata["tessa_static_repair_evidence_hash"] == evidence.evidence_hash

    assert reference.output_candidate_lineage_hash == repaired.candidate_lineage_hash
    assert evidence.output_candidate_lineage_hash == repaired.candidate_lineage_hash
    assert evidence.reference_binding_hash == reference.binding_hash
    assert evidence.final_policy_violating_face_count == 0
    assert len(reference.rows) == len(repaired.vertices)
    assert all(abs(sum(weight for _, weight in row.coefficients) - 1.0) < 1e-12 for row in reference.rows)

    validate_tessa_reference_binding_v1(reference, candidate=repaired)
    validate_tessa_static_repair_evidence_v1(
        evidence,
        candidate=repaired,
        reference_binding=reference,
    )


def test_static_repair_reverse_links_are_fail_closed():
    candidate, partition = _candidate_context()
    repaired, evidence, reference = repair_tessa_candidate_static_v1(
        candidate=candidate,
        policy=_policy(),
        partition=partition,
    )

    with pytest.raises(QualificationError, match="TESSA_REFERENCE_BINDING_CANDIDATE_MISMATCH"):
        validate_tessa_reference_binding_v1(
            replace(reference, output_candidate_lineage_hash="wrong"),
            candidate=repaired,
        )

    with pytest.raises(QualificationError, match="TESSA_STATIC_REPAIR_EVIDENCE_CANDIDATE_MISMATCH"):
        validate_tessa_static_repair_evidence_v1(
            replace(evidence, output_candidate_lineage_hash="wrong"),
            candidate=repaired,
            reference_binding=reference,
        )


def test_static_repair_rejects_non_tessa_input_authority():
    candidate, partition = _candidate_context()
    bad = replace(
        candidate,
        metadata={**candidate.metadata, "geometry_authority_class": "WRONG"},
    )
    with pytest.raises(QualificationError, match="TESSA_STATIC_REPAIR_INPUT_AUTHORITY_INVALID"):
        repair_tessa_candidate_static_v1(
            candidate=bad,
            policy=_policy(),
            partition=partition,
        )
