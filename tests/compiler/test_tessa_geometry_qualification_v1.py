from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from compiler.realsas_compiler_core.mechanical_partition_v1 import build_structural_partition
from compiler.realsas_compiler_core.product_authority_v1 import (
    ComponentCarrierDecisionIR,
    build_component_carrier_policy,
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
    tessa_static_carrier_binding_hash_v1,
    validate_tessa_static_carrier_binding_v1,
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


_RESEARCH_RULE = (
    "GSA_BARYCENTRIC_ADJOINT_ANCHOR__COMPONENT_HARMONIC__"
    "EXACT_RIGID_FALLBACK_V1"
)


def _context():
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
    surface = RiggingSurfaceIR(nodes, relations, "surface-v1")
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
            TESSAVertexProposalV1("v0", (0.15, 0.15, 0.20), "a", "decoded:0"),
            TESSAVertexProposalV1("v1", (0.75, 0.10, 0.18), "b", "decoded:0"),
            TESSAVertexProposalV1("v2", (0.10, 0.75, 0.22), "c", "decoded:0"),
        ),
        faces=(("v0", "v1", "v2"),),
        source_geometry_lineage_hash=surface.geometry_lineage_hash,
        model_provenance="checkpoint:test",
        topology_sequence_hash="topology:test",
    )
    field = build_tessa_material_support_field_v1(
        vertices_world=np.asarray([v.P for v in proposal.vertices], dtype=np.float64),
        decoded_component_indices=np.zeros(3, dtype=np.int64),
        surface=surface,
        partition=partition,
        topology_sequence_hash=proposal.topology_sequence_hash,
        proposal_vertex_ids=("v0", "v1", "v2"),
        max_support_nodes=3,
        max_graph_hops=2,
    )
    candidate, bridge = build_tessa_candidate_bridge_v1(
        proposal=proposal,
        support_field=field,
        surface=surface,
        partition=partition,
        carrier_policy=carrier,
        producer_policy_hash="policy-hash",
    )
    return candidate, bridge


def _stage19(candidate, *, source_pass=True, status="PASS_STATIC_CANONICAL_CARRIER"):
    report = {
        "status": status,
        "static_quality_policy_passed": True,
        "stage13_policy_replayed_on_actual_candidate_mesh": True,
        "actual_candidate_source_fidelity_passed": bool(source_pass),
        "source_fidelity_qualification_passed": bool(source_pass),
        "actual_candidate_source_fidelity_views": tuple(
            {"view_index": view_index, "passed": bool(source_pass)}
            for view_index in range(8)
        ),
        "demo_geometry_lineage": False,
        "product_authority_claimed": False,
    }
    provisional = StaticCanonicalMeshQualificationIR(
        candidate_mesh_binding_hash=candidate.candidate_lineage_hash,
        surface_addressing_binding_hash="addressing-hash",
        appearance_domain_binding_hash="appearance-domain-hash",
        geometry_gate_binding_hash="stage13-geometry-hash",
        partition_binding_hash=candidate.partition_binding_hash,
        qualification_report=report,
        qualification_hash="",
    )
    return replace(
        provisional,
        qualification_hash=static_mesh_qualification_hash(provisional),
    )


def test_tessa_binding_reuses_exact_stage19_authority_without_parallel_metric():
    candidate, bridge = _context()
    static_mesh = _stage19(candidate)
    value = bind_tessa_to_static_carrier_v1(
        candidate=candidate,
        bridge_evidence=bridge,
        static_mesh=static_mesh,
    )
    assert value.candidate_lineage_hash == candidate.candidate_lineage_hash
    assert value.bridge_evidence_hash == bridge.evidence_hash
    assert value.static_mesh_qualification_hash == static_mesh.qualification_hash
    assert value.proposal_geometry_hash == bridge.proposal_geometry_hash
    assert value.material_support_field_hash == bridge.material_support_field_hash
    assert value.metadata["authority"] == "EXISTING_STAGE19_STATIC_CANONICAL_CARRIER"
    assert value.metadata["parallel_source_fidelity_metric_created"] is False
    assert value.metadata["material_support_is_geometry_authority"] is False
    assert value.metadata["motion_capability_claimed"] is False
    assert value.metadata["dynamic_carrier_field_binding_required_before_qualified_mesh"] is True
    assert value.metadata["dynamic_carrier_field_binding_product_qualified"] is False
    assert value.metadata["research_fit1_binding_rule"] == _RESEARCH_RULE
    assert value.metadata["research_fit1_binding_is_product_authority"] is False

    # Stage19 is allowed to seal static geometry while the downstream dynamic
    # carrier-field transport remains deliberately unresolved.
    validate_tessa_static_carrier_binding_v1(
        value,
        candidate=candidate,
        bridge_evidence=bridge,
        static_mesh=static_mesh,
        require_dynamic_carrier_field_binding=False,
    )


def test_tessa_static_binding_blocks_dynamic_use_until_product_field_transport_is_qualified():
    candidate, bridge = _context()
    static_mesh = _stage19(candidate)
    value = bind_tessa_to_static_carrier_v1(
        candidate=candidate,
        bridge_evidence=bridge,
        static_mesh=static_mesh,
    )
    with pytest.raises(
        QualificationError,
        match="TESSA_MIRA_CARRIER_FIELD_BINDING_NOT_PRODUCT_QUALIFIED",
    ):
        validate_tessa_static_carrier_binding_v1(
            value,
            candidate=candidate,
            bridge_evidence=bridge,
            static_mesh=static_mesh,
        )


def test_tessa_binding_rejects_stage19_source_fidelity_failure():
    candidate, bridge = _context()
    with pytest.raises(QualificationError, match="TESSA_STATIC_BINDING_STAGE19_SOURCE_FIDELITY_NOT_PASS"):
        bind_tessa_to_static_carrier_v1(
            candidate=candidate,
            bridge_evidence=bridge,
            static_mesh=_stage19(candidate, source_pass=False),
        )


def test_tessa_binding_rejects_demo_or_nonpass_stage19():
    candidate, bridge = _context()
    demo = _stage19(candidate)
    demo_report = {**demo.qualification_report, "demo_geometry_lineage": True}
    demo = replace(demo, qualification_report=demo_report, qualification_hash="")
    demo = replace(demo, qualification_hash=static_mesh_qualification_hash(demo))
    with pytest.raises(QualificationError, match="TESSA_STATIC_BINDING_DEMO_LINEAGE_FORBIDDEN"):
        bind_tessa_to_static_carrier_v1(
            candidate=candidate,
            bridge_evidence=bridge,
            static_mesh=demo,
        )

    with pytest.raises(QualificationError, match="TESSA_STATIC_BINDING_STAGE19_STATUS_NOT_PASS"):
        bind_tessa_to_static_carrier_v1(
            candidate=candidate,
            bridge_evidence=bridge,
            static_mesh=_stage19(candidate, status="DEMO_ONLY_MEASURED_STATIC_CANONICAL_CARRIER__P999_FAIL"),
        )


def test_tessa_binding_is_hash_and_candidate_fail_closed():
    candidate, bridge = _context()
    static_mesh = _stage19(candidate)
    value = bind_tessa_to_static_carrier_v1(
        candidate=candidate,
        bridge_evidence=bridge,
        static_mesh=static_mesh,
    )
    bad = replace(value, proposal_geometry_hash="wrong")
    bad = replace(bad, binding_hash=tessa_static_carrier_binding_hash_v1(bad))
    with pytest.raises(QualificationError, match="TESSA_STATIC_BINDING_PROPOSAL_GEOMETRY_MISMATCH"):
        validate_tessa_static_carrier_binding_v1(
            bad,
            candidate=candidate,
            bridge_evidence=bridge,
            static_mesh=static_mesh,
            require_dynamic_carrier_field_binding=False,
        )

    drifted_static = replace(static_mesh, candidate_mesh_binding_hash="wrong", qualification_hash="")
    drifted_static = replace(
        drifted_static,
        qualification_hash=static_mesh_qualification_hash(drifted_static),
    )
    with pytest.raises(QualificationError, match="TESSA_STATIC_BINDING_STAGE19_CANDIDATE_MISMATCH"):
        bind_tessa_to_static_carrier_v1(
            candidate=candidate,
            bridge_evidence=bridge,
            static_mesh=drifted_static,
        )
