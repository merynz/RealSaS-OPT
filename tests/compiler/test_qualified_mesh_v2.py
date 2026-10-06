from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from compiler.realsas_compiler_core.canonical_mesh_candidate_v1 import (
    build_canonical_relation_candidate,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mechanical_partition_v1 import build_structural_partition
from compiler.realsas_compiler_core.product_authority_v1 import (
    CarrierCoverageThresholdIR,
    ComponentCarrierDecisionIR,
    DeformationCapabilityEnvelopeIR,
    build_component_carrier_policy,
    build_mesh_qualification_policy,
    deformation_envelope_lineage_hash,
    qualify_canonical_mesh_candidate,
)
from compiler.realsas_compiler_core.qualified_mesh_v2 import (
    qualify_canonical_mesh_candidate_v2,
    validate_tessa_qualified_mesh_v2,
)
from compiler.realsas_compiler_core.surface_addressing_v1 import (
    StaticCanonicalMeshQualificationIR,
    static_mesh_qualification_hash,
)
from compiler.realsas_compiler_core.tessa_candidate_bridge_v1 import (
    build_tessa_candidate_bridge_v1,
)
from compiler.realsas_compiler_core.tessa_geometry_qualification_v1 import (
    bind_tessa_to_static_carrier_v1,
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
        support_views=tuple(range(8)),
        provenance_refs=(f"prov:{surface_id}",),
        source_observation_ids=(f"obs:{surface_id}",),
        derived_normal=(0.0, 0.0, 1.0),
        validity_flags=("OBSERVED_SIGNED_ZERO_SURFACE",),
    )


def _surface():
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
    return RiggingSurfaceIR(nodes, relations, "surface-v2-test")


def _common():
    surface = _surface()
    partition = build_structural_partition(surface)
    carrier = build_component_carrier_policy(
        partition=partition,
        decisions=tuple(
            ComponentCarrierDecisionIR(
                component.component_id,
                "MESH",
                ("TEST_CONSERVATIVE_MESH",),
            )
            for component in partition.components
        ),
    )
    policy = build_mesh_qualification_policy(
        g1_max_normal_refinement_ratio=0.125,
        g1_max_tangential_to_normal_ratio=0.25,
        g3_min_angle_deg=7.5,
        g3_max_aspect_longest_over_min_altitude=16.0,
        coverage_thresholds=(
            CarrierCoverageThresholdIR("MESH", 0.0, 0.0, 1.0, 1.0),
            CarrierCoverageThresholdIR("PLANAR", 0.0, 0.0, 1.0, 1.0),
        ),
    )
    envelope = DeformationCapabilityEnvelopeIR(
        skeleton_lineage_hash="skeleton-v1",
        joint_ranges=(),
        camera_binding_hashes=tuple(f"camera-{i}" for i in range(8)),
        allowed_attachment_state_hashes=(),
        axis_contract_hash="axis-v1",
        probe_plan_hash="probe-v1",
        envelope_lineage_hash="",
    )
    envelope = replace(
        envelope,
        envelope_lineage_hash=deformation_envelope_lineage_hash(envelope),
    )
    return surface, partition, carrier, policy, envelope


def _dynamic_report(partition, carrier, envelope):
    component_id = partition.components[0].component_id
    return {
        "gates": {
            "G1_SUPPORT_LINEAGE": "PASS",
            "G2_TOPOLOGY": "PASS",
            "G3_DEFORMATION": "PASS",
            "G3B_SKIN_TOPOLOGY_COMPATIBILITY": "PASS",
            "G4_COMPONENT_BOUNDARY": "PASS",
            "G5_MULTIVIEW_COVERAGE": "PASS",
        },
        "single_aggregate_score_authority": False,
        "view_component_coverage_matrix_complete": True,
        "consequential_unknown_boundary_count": 0,
        "unknown_boundary_analysis_hash": content_sha256({"unknown": []}),
        "g3_envelope_binding_hash": envelope.envelope_lineage_hash,
        "g3_motion_capability_claimed": False,
        "g3_stress_probe_hash": content_sha256({"g3": "pass"}),
        "g3_stress_probe_status": "PASS",
        "skin_topology_compatibility_report_hash": content_sha256({"g3b": "pass"}),
        "skin_topology_compatibility_status": "PASS",
        "skin_topology_weight_mutation": False,
        "carrier_policy_hash": carrier.carrier_policy_lineage_hash,
        "g5_evidence_hash": content_sha256({"g5": "pass"}),
        "view_component_coverage": [
            {
                "view_index": view_index,
                "component_id": component_id,
                "carrier_class": "MESH",
                "recall": 1.0,
                "precision": 1.0,
                "largest_coherent_hole_fraction": 0.0,
                "interior_uncovered_fraction": 0.0,
                "status": "PASS",
            }
            for view_index in range(8)
        ],
    }


def _tessa_context():
    surface, partition, carrier, policy, envelope = _common()
    proposal = TESSAMeshProposalV1(
        vertices=(
            TESSAVertexProposalV1("v0", (0.0, 0.0, 0.20), "a", "decoded:0"),
            TESSAVertexProposalV1("v1", (1.0, 0.0, 0.20), "b", "decoded:0"),
            TESSAVertexProposalV1("v2", (0.0, 1.0, 0.20), "c", "decoded:0"),
        ),
        faces=(("v0", "v1", "v2"),),
        source_geometry_lineage_hash=surface.geometry_lineage_hash,
        model_provenance="checkpoint:test",
        topology_sequence_hash="topology:test",
        metadata={"product_authority_claimed": False},
    )
    field = build_tessa_material_support_field_v1(
        vertices_world=np.asarray([row.P for row in proposal.vertices], dtype=np.float64),
        decoded_component_indices=np.zeros(3, dtype=np.int64),
        surface=surface,
        partition=partition,
        topology_sequence_hash=proposal.topology_sequence_hash,
        proposal_vertex_ids=tuple(row.proposal_vertex_id for row in proposal.vertices),
        max_support_nodes=3,
        max_graph_hops=2,
    )
    candidate, bridge = build_tessa_candidate_bridge_v1(
        proposal=proposal,
        support_field=field,
        surface=surface,
        partition=partition,
        carrier_policy=carrier,
        producer_policy_hash=content_sha256({"policy": "tessa-test"}),
    )
    static_report = {
        "status": "PASS_STATIC_CANONICAL_CARRIER",
        "static_quality_policy_passed": True,
        "stage13_policy_replayed_on_actual_candidate_mesh": True,
        "actual_candidate_source_fidelity_passed": True,
        "source_fidelity_qualification_passed": True,
        "demo_geometry_lineage": False,
        "product_authority_claimed": False,
        "actual_candidate_source_fidelity_views": [
            {"view_index": view_index, "passed": True}
            for view_index in range(8)
        ],
    }
    static = StaticCanonicalMeshQualificationIR(
        candidate_mesh_binding_hash=candidate.candidate_lineage_hash,
        surface_addressing_binding_hash="addressing-v1",
        appearance_domain_binding_hash="appearance-v1",
        geometry_gate_binding_hash="geometry-v1",
        partition_binding_hash=partition.partition_lineage_hash,
        qualification_report=static_report,
        qualification_hash="",
    )
    static = replace(static, qualification_hash=static_mesh_qualification_hash(static))
    binding = bind_tessa_to_static_carrier_v1(
        candidate=candidate,
        bridge_evidence=bridge,
        static_mesh=static,
    )
    return (
        surface,
        partition,
        carrier,
        policy,
        envelope,
        candidate,
        bridge,
        static,
        binding,
    )


def test_v2_dispatcher_is_byte_exact_legacy_for_deterministic_candidate():
    surface, partition, carrier, policy, envelope = _common()
    candidate = build_canonical_relation_candidate(
        surface,
        partition,
        carrier,
        producer_policy_hash=content_sha256({"policy": "legacy-test"}),
    )
    report = _dynamic_report(partition, carrier, envelope)
    legacy = qualify_canonical_mesh_candidate(
        candidate,
        surface=surface,
        partition=partition,
        carrier_policy=carrier,
        envelope=envelope,
        policy=policy,
        qualification_report=report,
    )
    v2 = qualify_canonical_mesh_candidate_v2(
        candidate,
        surface=surface,
        partition=partition,
        carrier_policy=carrier,
        envelope=envelope,
        policy=policy,
        qualification_report=report,
    )
    assert v2.to_dict() == legacy.to_dict()


def test_tessa_v2_mints_exact_learned_xyz_only_with_stage19_binding():
    (
        surface,
        partition,
        carrier,
        policy,
        envelope,
        candidate,
        bridge,
        static,
        binding,
    ) = _tessa_context()
    report = _dynamic_report(partition, carrier, envelope)
    mesh = qualify_canonical_mesh_candidate_v2(
        candidate,
        surface=surface,
        partition=partition,
        carrier_policy=carrier,
        envelope=envelope,
        policy=policy,
        qualification_report=report,
        bridge_evidence=bridge,
        static_mesh=static,
        static_binding=binding,
    )
    assert min(float(vertex.P[2]) for vertex in mesh.vertices) > 0.0
    assert mesh.metadata["geometry_authority"] == "EXISTING_STAGE19_STATIC_CANONICAL_CARRIER"
    assert mesh.metadata["material_support_is_geometry_authority"] is False
    assert mesh.metadata["legacy_g1_convex_lift_applicable"] is False
    assert mesh.metadata["learned_xyz_preserved_exactly"] is True
    assert mesh.metadata["motion_capability_claimed"] is False
    assert mesh.qualification_report["stage19_static_binding_hash"] == binding.binding_hash


def test_tessa_v2_rejects_missing_stage19_evidence():
    surface, partition, carrier, policy, envelope, candidate, *_ = _tessa_context()
    with pytest.raises(QualificationError, match="TESSA_QUALIFIED_MESH_STAGE19_EVIDENCE_REQUIRED"):
        qualify_canonical_mesh_candidate_v2(
            candidate,
            surface=surface,
            partition=partition,
            carrier_policy=carrier,
            envelope=envelope,
            policy=policy,
            qualification_report=_dynamic_report(partition, carrier, envelope),
        )


def test_tessa_v2_rejects_failed_dynamic_gate():
    (
        surface,
        partition,
        carrier,
        policy,
        envelope,
        candidate,
        bridge,
        static,
        binding,
    ) = _tessa_context()
    report = _dynamic_report(partition, carrier, envelope)
    report["gates"] = {**report["gates"], "G3_DEFORMATION": "FAIL"}
    with pytest.raises(QualificationError, match="TESSA_QUALIFIED_MESH_REQUIRES_ALL_SIX_GATES_PASS"):
        qualify_canonical_mesh_candidate_v2(
            candidate,
            surface=surface,
            partition=partition,
            carrier_policy=carrier,
            envelope=envelope,
            policy=policy,
            qualification_report=report,
            bridge_evidence=bridge,
            static_mesh=static,
            static_binding=binding,
        )


def test_tessa_v2_rejects_tampered_stage19_binding():
    (
        surface,
        partition,
        carrier,
        policy,
        envelope,
        candidate,
        bridge,
        static,
        binding,
    ) = _tessa_context()
    mesh = qualify_canonical_mesh_candidate_v2(
        candidate,
        surface=surface,
        partition=partition,
        carrier_policy=carrier,
        envelope=envelope,
        policy=policy,
        qualification_report=_dynamic_report(partition, carrier, envelope),
        bridge_evidence=bridge,
        static_mesh=static,
        static_binding=binding,
    )
    bad_binding = replace(binding, static_mesh_qualification_hash="drift")
    with pytest.raises(QualificationError, match="TESSA_STATIC_BINDING_STAGE19_BINDING_MISMATCH"):
        validate_tessa_qualified_mesh_v2(
            mesh,
            candidate=candidate,
            bridge_evidence=bridge,
            static_mesh=static,
            static_binding=bad_binding,
            surface=surface,
            partition=partition,
            carrier_policy=carrier,
            envelope=envelope,
            policy=policy,
        )
