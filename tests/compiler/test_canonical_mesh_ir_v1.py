from __future__ import annotations

from dataclasses import replace
import json

import pytest

from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
    canonical_mesh_from_dict_v1,
    mint_canonical_mesh_v1,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mechanical_carrier_evidence_v1 import (
    MechanicalCarrierEvidenceIR,
    mechanical_carrier_evidence_hash,
)
from compiler.realsas_compiler_core.product_authority_v1 import (
    CanonicalMeshCandidateIR,
    CanonicalMeshVertexCandidateIR,
)
from compiler.realsas_compiler_core.static_mesh_qualification_v2 import (
    StaticMeshQualificationIRV2,
    static_mesh_qualification_hash_v2,
)
from compiler.realsas_compiler_core.surface_addressing_v1 import (
    StaticCanonicalMeshQualificationIR,
    static_mesh_qualification_hash,
)
from compiler.realsas_compiler_core.types import QualificationError, SurfaceSupportBinding


def _fixture():
    support = SurfaceSupportBinding(
        mode="MATERIAL_SUPPORT_ONLY",
        coefficients=(("s0", 1.0),),
        metadata={"geometry_authority": False},
    )
    vertices = (
        CanonicalMeshVertexCandidateIR("v0", support, "c0", (0.0, 0.0, 0.0)),
        CanonicalMeshVertexCandidateIR("v1", support, "c0", (1.0, 0.0, 0.0)),
        CanonicalMeshVertexCandidateIR("v2", support, "c0", (0.0, 1.0, 0.0)),
    )
    candidate = CanonicalMeshCandidateIR(
        vertices=vertices,
        faces=(("v0", "v1", "v2"),),
        edges=(("v0", "v1"), ("v1", "v2"), ("v0", "v2")),
        surface_binding_hash="surface-evidence-binding",
        partition_binding_hash="partition-binding",
        carrier_policy_binding_hash="carrier-policy-binding",
        producer_id="test.proposal.bridge",
        producer_policy_hash="producer-policy-hash",
        candidate_lineage_hash="c" * 64,
    )
    static = StaticCanonicalMeshQualificationIR(
        candidate_mesh_binding_hash=candidate.candidate_lineage_hash,
        surface_addressing_binding_hash="addressing-binding",
        appearance_domain_binding_hash="appearance-domain-binding",
        geometry_gate_binding_hash="geometry-gate-binding",
        partition_binding_hash=candidate.partition_binding_hash,
        qualification_report={
            "status": "PASS_STATIC_CANONICAL_CARRIER",
            "static_quality_policy_passed": True,
            "source_fidelity_qualification_passed": True,
            "demo_geometry_lineage": False,
        },
        qualification_hash="",
    )
    static = replace(static, qualification_hash=static_mesh_qualification_hash(static))

    ordered_ids = ("v0", "v1", "v2")
    positions = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    normals = ((0.0, 0.0, 1.0),) * 3
    normal_valid = (True, True, True)
    face_indices = ((0, 1, 2),)
    topology_hash = content_sha256(
        {
            "schema": "RealSaS.MechanicalCarrierTopologyBasis.v1",
            "ordered_vertex_ids": ordered_ids,
            "face_vertex_indices": face_indices,
        }
    )
    geometry_hash = content_sha256(
        {
            "schema": "RealSaS.MechanicalCarrierGeometryBasis.v1",
            "ordered_vertex_ids": ordered_ids,
            "positions": positions,
            "normals": normals,
            "normal_valid": normal_valid,
        }
    )
    carrier = MechanicalCarrierEvidenceIR(
        candidate_mesh_binding_hash=candidate.candidate_lineage_hash,
        static_mesh_qualification_binding_hash=static.qualification_hash,
        surface_addressing_binding_hash=static.surface_addressing_binding_hash,
        source_surface_binding_hash=candidate.surface_binding_hash,
        ordered_vertex_ids=ordered_ids,
        positions=positions,
        normals=normals,
        normal_valid=normal_valid,
        face_vertex_indices=face_indices,
        topology_hash=topology_hash,
        geometry_hash=geometry_hash,
        carrier_evidence_hash="",
        metadata={"authority": "STATIC_QUALIFIED_MECHANICAL_CARRIER"},
    )
    carrier = replace(carrier, carrier_evidence_hash=mechanical_carrier_evidence_hash(carrier))
    return candidate, static, carrier


def _static_v2(candidate, static_v1, carrier_v1):
    value = StaticMeshQualificationIRV2(
        candidate_mesh_binding_hash=candidate.candidate_lineage_hash,
        surface_addressing_binding_hash=static_v1.surface_addressing_binding_hash,
        geometry_gate_binding_hash=static_v1.geometry_gate_binding_hash,
        partition_binding_hash=candidate.partition_binding_hash,
        qualification_report={
            "status": "PASS_STATIC_CANONICAL_CARRIER",
            "static_quality_policy_passed": True,
            "source_fidelity_qualification_passed": True,
            "demo_geometry_lineage": False,
            "presentation_domain_required_for_mesh_admission": False,
        },
        qualification_hash="",
        metadata={"presentation_direction_binding": "FORBIDDEN_AT_MESH_BIRTH"},
    )
    value = replace(value, qualification_hash=static_mesh_qualification_hash_v2(value))
    carrier = replace(
        carrier_v1,
        static_mesh_qualification_binding_hash=value.qualification_hash,
        carrier_evidence_hash="",
    )
    carrier = replace(carrier, carrier_evidence_hash=mechanical_carrier_evidence_hash(carrier))
    return value, carrier


def test_canonical_mesh_is_generic_static_authority_and_roundtrips():
    candidate, static, carrier = _fixture()
    mesh = mint_canonical_mesh_v1(
        candidate,
        static_qualification=static,
        carrier_evidence=carrier,
        proposal_evidence_refs=("shape-proposal-evidence-sha",),
    )
    assert mesh.metadata["authority"] == "COMPILER_CANONICAL_MESH"
    assert mesh.metadata["dynamic_qualification_claimed"] is False
    assert mesh.metadata["learned_family_embedded_in_schema"] is False
    assert mesh.carrier_topology_hash == carrier.topology_hash
    assert mesh.carrier_geometry_hash == carrier.geometry_hash
    assert mesh.proposal_evidence_refs == ("shape-proposal-evidence-sha",)
    assert "TESSA" not in json.dumps(mesh.to_dict(), sort_keys=True).upper()
    decoded = canonical_mesh_from_dict_v1(mesh.to_dict())
    assert decoded == mesh


def test_canonical_mesh_mints_from_presentation_independent_static_v2():
    candidate, static_v1, carrier_v1 = _fixture()
    static_v2, carrier_v2 = _static_v2(candidate, static_v1, carrier_v1)
    mesh = mint_canonical_mesh_v1(
        candidate,
        static_qualification=static_v2,
        carrier_evidence=carrier_v2,
        proposal_evidence_refs=("shape-v2-content-sha",),
    )
    assert mesh.static_qualification_hash == static_v2.qualification_hash
    assert mesh.metadata["authority"] == "COMPILER_CANONICAL_MESH"
    assert "appearance_domain" not in json.dumps(static_v2.to_dict(), sort_keys=True).lower()
    assert static_v2.qualification_report["presentation_domain_required_for_mesh_admission"] is False


def test_canonical_mesh_rejects_carrier_basis_drift_even_when_bindings_match():
    candidate, static, carrier = _fixture()
    drifted = replace(
        carrier,
        positions=((0.0, 0.0, 0.0), (9.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        carrier_evidence_hash="",
    )
    drifted = replace(drifted, carrier_evidence_hash=mechanical_carrier_evidence_hash(drifted))
    with pytest.raises(QualificationError, match="CARRIER_POSITION_DRIFT"):
        mint_canonical_mesh_v1(
            candidate,
            static_qualification=static,
            carrier_evidence=drifted,
        )


def test_canonical_mesh_rejects_static_source_fidelity_failure():
    candidate, static, carrier = _fixture()
    report = dict(static.qualification_report)
    report["source_fidelity_qualification_passed"] = False
    failed = replace(static, qualification_report=report, qualification_hash="")
    failed = replace(failed, qualification_hash=static_mesh_qualification_hash(failed))
    with pytest.raises(QualificationError, match="SOURCE_FIDELITY_NOT_PASS"):
        mint_canonical_mesh_v1(
            candidate,
            static_qualification=failed,
            carrier_evidence=carrier,
        )
