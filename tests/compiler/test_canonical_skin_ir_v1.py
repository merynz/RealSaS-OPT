from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
    CanonicalMeshIR,
    CanonicalMeshVertexIR,
    canonical_mesh_lineage_hash_v1,
)
from compiler.realsas_compiler_core.canonical_rig_ir_v1 import (
    CanonicalRigIR,
    CanonicalRigJointIR,
    canonical_rig_lineage_hash_v1,
)
from compiler.realsas_compiler_core.canonical_skin_ir_v1 import (
    canonical_skin_from_dict_v1,
    mint_canonical_skin_v1,
    qualify_canonical_skin_v1,
)
from compiler.realsas_compiler_core.carrier_skin_v1 import (
    CarrierSkinInfluenceProposal,
    CarrierSkinProposalIR,
    qualify_carrier_skin_v1,
)
from compiler.realsas_compiler_core.types import QualificationError, SurfaceSupportBinding


def _support(surface_id: str) -> SurfaceSupportBinding:
    return SurfaceSupportBinding(
        mode="EXACT_SURFACE_NODE",
        coefficients=((surface_id, 1.0),),
        metadata={"test_fixture": True},
    )


def _mesh() -> CanonicalMeshIR:
    vertices = tuple(
        CanonicalMeshVertexIR(
            canonical_mesh_vertex_id=f"MV:{i}",
            support_binding=_support(f"S:{i}"),
            component_id="body",
            P=position,
            source_candidate_vertex_id=f"C:{i}",
        )
        for i, position in enumerate(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        )
    )
    value = CanonicalMeshIR(
        vertices=vertices,
        faces=(("MV:0", "MV:1", "MV:2"),),
        edges=(("MV:0", "MV:1"), ("MV:0", "MV:2"), ("MV:1", "MV:2")),
        source_evidence_binding_hash="1" * 64,
        partition_binding_hash="2" * 64,
        carrier_policy_binding_hash="3" * 64,
        static_qualification_hash="4" * 64,
        carrier_evidence_hash="5" * 64,
        carrier_topology_hash="6" * 64,
        carrier_geometry_hash="7" * 64,
        proposal_evidence_refs=("8" * 64,),
        source_candidate_lineage_hash="9" * 64,
        mesh_lineage_hash="",
        metadata={"authority": "COMPILER_CANONICAL_MESH", "learned_family_embedded_in_schema": False, "dynamic_qualification_claimed": False},
    )
    return replace(value, mesh_lineage_hash=canonical_mesh_lineage_hash_v1(value))


def _rig(mesh: CanonicalMeshIR) -> CanonicalRigIR:
    value = CanonicalRigIR(
        joints=(
            CanonicalRigJointIR("J:0", (0.0, 0.0, 0.0), None, "P:0"),
            CanonicalRigJointIR("J:1", (0.0, 1.0, 0.0), "J:0", "P:1"),
        ),
        root_joint_ids=("J:0",),
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        mesh_carrier_evidence_hash=mesh.carrier_evidence_hash,
        mesh_carrier_topology_hash=mesh.carrier_topology_hash,
        mesh_carrier_geometry_hash=mesh.carrier_geometry_hash,
        feature_evidence_binding_hash=mesh.source_evidence_binding_hash,
        proposal_evidence_hash="a" * 64,
        structure_qualification_hash="b" * 64,
        rig_lineage_hash="",
        metadata={
            "authority": "COMPILER_CANONICAL_RIG",
            "canonical_root_semantics": "NON_EMPTY_DEFORM_ROOT_SET",
            "learned_family_embedded_in_schema": False,
            "frame_observability_claimed": False,
        },
    )
    return replace(value, rig_lineage_hash=canonical_rig_lineage_hash_v1(value))


def _proposal(mesh: CanonicalMeshIR, rig: CanonicalRigIR, *, row_sum: float = 1.0) -> CarrierSkinProposalIR:
    influences = []
    for index, vertex in enumerate(mesh.vertices):
        if index == 0:
            weights = (("J:0", row_sum * 0.75), ("J:1", row_sum * 0.25))
        else:
            weights = (("J:0", row_sum * 0.25), ("J:1", row_sum * 0.75))
        for joint_id, weight in weights:
            influences.append(
                CarrierSkinInfluenceProposal(
                    carrier_vertex_id=vertex.source_candidate_vertex_id,
                    canonical_joint_id=joint_id,
                    weight=weight,
                )
            )
    return CarrierSkinProposalIR(
        influences=tuple(influences),
        carrier_evidence_hash=mesh.carrier_evidence_hash,
        carrier_topology_hash=mesh.carrier_topology_hash,
        carrier_geometry_hash=mesh.carrier_geometry_hash,
        skeleton_binding_hash=rig.rig_lineage_hash,
        model_provenance="RealSaS.MIRA.DirectCarrierSimplex.Axis541.v5_5",
        metadata={
            "output_domain": "CANONICAL_MESH_CARRIER",
            "direct_simplex_readout": True,
            "semantic_skin_transfer_performed": False,
            "gsa_skin_field_authority_minted": False,
            "source_skin_runtime_authority": False,
            "mesh_mutation_invalidates_prediction": True,
        },
    )


def test_canonical_skin_maps_legacy_carrier_ids_to_stable_mesh_ids_and_mints():
    mesh = _mesh()
    rig = _rig(mesh)
    qualification = qualify_canonical_skin_v1(
        mesh,
        rig,
        _proposal(mesh, rig),
        max_simplex_repair_l1=1e-6,
        max_total_correction_l1=1e-5,
        negative_tolerance=1e-9,
    )
    assert tuple(row.canonical_mesh_vertex_id for row in qualification.rows) == (
        "MV:0", "MV:1", "MV:2"
    )
    assert all(not row.canonical_mesh_vertex_id.startswith("C:") for row in qualification.rows)
    skin = mint_canonical_skin_v1(mesh, rig, qualification)
    assert skin.mesh_lineage_hash == mesh.mesh_lineage_hash
    assert skin.rig_lineage_hash == rig.rig_lineage_hash
    assert skin.metadata["authority"] == "COMPILER_CANONICAL_SKIN"
    assert skin.metadata["feature_evidence_is_canonical_parent"] is False
    assert skin.metadata["post_bind_frame_observability_claimed"] is False
    assert skin.metadata["mechanics_authority_claimed"] is False
    assert skin.metadata["learned_family_embedded_in_schema"] is False
    assert canonical_skin_from_dict_v1(skin.to_dict()) == skin


def test_canonical_and_legacy_skin_use_same_numeric_simplex_court():
    mesh = _mesh()
    rig = _rig(mesh)
    proposal = _proposal(mesh, rig, row_sum=1.000001)
    canonical = qualify_canonical_skin_v1(
        mesh,
        rig,
        proposal,
        max_simplex_repair_l1=1e-5,
        max_total_correction_l1=1e-4,
        negative_tolerance=1e-9,
    )
    legacy_proposal = replace(
        proposal,
        skeleton_binding_hash="legacy-rig",
        metadata={**proposal.metadata, "output_domain": "MECHANICAL_CARRIER_M"},
    )
    carrier = SimpleNamespace(
        ordered_vertex_ids=tuple(v.source_candidate_vertex_id for v in mesh.vertices),
        carrier_evidence_hash=mesh.carrier_evidence_hash,
        topology_hash=mesh.carrier_topology_hash,
        geometry_hash=mesh.carrier_geometry_hash,
    )
    skeleton = SimpleNamespace(
        skeleton_lineage_hash="legacy-rig",
        joints=tuple(SimpleNamespace(canonical_joint_id=j.canonical_joint_id) for j in rig.joints),
    )
    legacy = qualify_carrier_skin_v1(
        carrier,
        skeleton,
        legacy_proposal,
        max_simplex_repair_l1=1e-5,
        max_total_correction_l1=1e-4,
        negative_tolerance=1e-9,
    )
    assert [row.influences for row in canonical.rows] == [row.influences for row in legacy.rows]
    assert canonical.qualification_report["max_row_correction_l1"] == legacy.qualification_report["max_row_correction_l1"]
    assert canonical.qualification_report["total_correction_l1"] == legacy.qualification_report["total_correction_l1"]


def test_canonical_skin_rejects_unknown_carrier_vertex():
    mesh = _mesh()
    rig = _rig(mesh)
    proposal = _proposal(mesh, rig)
    bad = replace(
        proposal,
        influences=(
            replace(proposal.influences[0], carrier_vertex_id="C:UNKNOWN"),
            *proposal.influences[1:],
        ),
    )
    with pytest.raises(QualificationError, match="UNKNOWN_CARRIER_VERTEX"):
        qualify_canonical_skin_v1(mesh, rig, bad, max_simplex_repair_l1=1e-6, max_total_correction_l1=1e-5, negative_tolerance=1e-9)


def test_canonical_skin_rejects_rig_or_carrier_drift():
    mesh = _mesh()
    rig = _rig(mesh)
    proposal = _proposal(mesh, rig)
    with pytest.raises(QualificationError, match="PROPOSAL_RIG_BINDING_DRIFT"):
        qualify_canonical_skin_v1(mesh, rig, replace(proposal, skeleton_binding_hash="wrong"), max_simplex_repair_l1=1e-6, max_total_correction_l1=1e-5, negative_tolerance=1e-9)
    with pytest.raises(QualificationError, match="PROPOSAL_CARRIER_BINDING_DRIFT"):
        qualify_canonical_skin_v1(mesh, rig, replace(proposal, carrier_geometry_hash="wrong"), max_simplex_repair_l1=1e-6, max_total_correction_l1=1e-5, negative_tolerance=1e-9)


def test_canonical_skin_rejects_repair_budget_overrun():
    mesh = _mesh()
    rig = _rig(mesh)
    proposal = _proposal(mesh, rig, row_sum=1.1)
    with pytest.raises(QualificationError, match="ROW_REPAIR_BUDGET_EXCEEDED"):
        qualify_canonical_skin_v1(mesh, rig, proposal, max_simplex_repair_l1=1e-3, max_total_correction_l1=1.0, negative_tolerance=1e-9)
