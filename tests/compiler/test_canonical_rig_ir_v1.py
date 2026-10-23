from __future__ import annotations

from dataclasses import replace

import pytest

from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
    CanonicalMeshIR,
    CanonicalMeshVertexIR,
    canonical_mesh_lineage_hash_v1,
)
from compiler.realsas_compiler_core.canonical_rig_ir_v1 import (
    CanonicalRigIR,
    CanonicalRigJointIR,
    canonical_rig_from_dict_v1,
    canonical_rig_lineage_hash_v1,
    mint_canonical_rig_v1,
)
from compiler.realsas_compiler_core.types import (
    QualifiedJoint,
    QualifiedSkeletonIR,
    QualificationError,
    SkeletonProposalEdge,
    SkeletonProposalIR,
    SkeletonProposalJoint,
    SurfaceSupportBinding,
)

SURFACE_HASH = "1" * 64


def _support(surface_id: str) -> SurfaceSupportBinding:
    return SurfaceSupportBinding(
        mode="EXACT_SURFACE_NODE",
        coefficients=((surface_id, 1.0),),
        metadata={"test_fixture": True},
    )


def _mesh() -> CanonicalMeshIR:
    value = CanonicalMeshIR(
        vertices=(
            CanonicalMeshVertexIR(
                canonical_mesh_vertex_id="MV:0",
                support_binding=_support("S:0"),
                P=(0.0, 0.0, 0.0),
                component_id="body",
                source_candidate_vertex_id="C:0",
            ),
            CanonicalMeshVertexIR(
                canonical_mesh_vertex_id="MV:1",
                support_binding=_support("S:1"),
                P=(1.0, 0.0, 0.0),
                component_id="body",
                source_candidate_vertex_id="C:1",
            ),
            CanonicalMeshVertexIR(
                canonical_mesh_vertex_id="MV:2",
                support_binding=_support("S:2"),
                P=(0.0, 1.0, 0.0),
                component_id="body",
                source_candidate_vertex_id="C:2",
            ),
        ),
        faces=(("MV:0", "MV:1", "MV:2"),),
        edges=(("MV:0", "MV:1"), ("MV:0", "MV:2"), ("MV:1", "MV:2")),
        source_evidence_binding_hash=SURFACE_HASH,
        partition_binding_hash="2" * 64,
        carrier_policy_binding_hash="3" * 64,
        static_qualification_hash="4" * 64,
        carrier_evidence_hash="5" * 64,
        carrier_topology_hash="6" * 64,
        carrier_geometry_hash="7" * 64,
        proposal_evidence_refs=("8" * 64,),
        source_candidate_lineage_hash="9" * 64,
        mesh_lineage_hash="",
        metadata={"authority": "COMPILER_CANONICAL_MESH"},
    )
    return replace(value, mesh_lineage_hash=canonical_mesh_lineage_hash_v1(value))


def _proposal() -> SkeletonProposalIR:
    return SkeletonProposalIR(
        joints=(
            SkeletonProposalJoint(
                proposal_id="P:0",
                position=(0.0, 0.0, 0.0),
                root_score=1.0,
                support_surface_ids=("S:0",),
            ),
            SkeletonProposalJoint(
                proposal_id="P:1",
                position=(0.0, 1.0, 0.0),
                support_surface_ids=("S:1",),
            ),
        ),
        edges=(
            SkeletonProposalEdge(
                edge_id="E:0:1",
                parent_proposal_id="P:0",
                child_proposal_id="P:1",
                score=1.0,
                hard_required=True,
                reason="AXIS_V541_HARD_CAUSAL_PARENT_AUTHORITY",
            ),
        ),
        surface_binding_hash=SURFACE_HASH,
        model_provenance="RealSaS.AXIS41.BFSHardParent.OrderedDiscreteXYZ256.FrozenCausal.v5_4_1",
        metadata={
            "hard_causal_parent_authority": True,
            "compiler_role": "VALIDATE_AND_MATERIALIZE",
            "compiler_parent_reselection_allowed": False,
            "geometry_authority": "ORDERED_256_BIN_XYZ_ARGMAX",
        },
    )


def _qualified(mesh: CanonicalMeshIR) -> QualifiedSkeletonIR:
    return QualifiedSkeletonIR(
        joints=(
            QualifiedJoint(
                canonical_joint_id="J:0",
                position=(0.0, 0.0, 0.0),
                parent_canonical_id=None,
                support_surface_ids=("S:0",),
                source_proposal_id="P:0",
            ),
            QualifiedJoint(
                canonical_joint_id="J:1",
                position=(0.0, 1.0, 0.0),
                parent_canonical_id="J:0",
                support_surface_ids=("S:1",),
                source_proposal_id="P:1",
            ),
        ),
        root_id="J:0",
        qualification_report={
            "status": "PASS_REQUIRED_TREE_VALIDATED",
            "selection_search_performed": False,
            "parent_reselection_performed": False,
            "authority_bindings": {
                "canonical_mesh": mesh.mesh_lineage_hash,
                "mechanical_carrier_evidence": mesh.carrier_evidence_hash,
                "mechanical_carrier_topology": mesh.carrier_topology_hash,
                "mechanical_carrier_geometry": mesh.carrier_geometry_hash,
            },
        },
        skeleton_lineage_hash="a" * 64,
    )


def test_canonical_rig_mints_on_exact_mesh_without_frame_authority():
    mesh = _mesh()
    rig = mint_canonical_rig_v1(
        mesh,
        proposal=_proposal(),
        qualified_skeleton=_qualified(mesh),
        feature_evidence_binding_hash=SURFACE_HASH,
    )
    assert rig.mesh_lineage_hash == mesh.mesh_lineage_hash
    assert rig.root_joint_ids == ("J:0",)
    assert rig.joints[1].parent_canonical_joint_id == "J:0"
    assert rig.metadata["authority"] == "COMPILER_CANONICAL_RIG"
    assert rig.metadata["feature_evidence_is_canonical_parent"] is False
    assert rig.metadata["frame_observability_claimed"] is False
    assert rig.metadata["post_bind_frame_observability_required"] is True
    assert rig.metadata["learned_family_embedded_in_schema"] is False
    assert rig.metadata["canonical_root_semantics"] == "NON_EMPTY_DEFORM_ROOT_SET"
    assert rig.metadata["current_axis_root_semantics"] == "SINGLE_HARD_CAUSAL_TREE_IMPLEMENTATION_DETAIL"
    assert canonical_rig_from_dict_v1(rig.to_dict()) == rig


def test_canonical_rig_schema_accepts_generic_multi_root_forest():
    value = CanonicalRigIR(
        joints=(
            CanonicalRigJointIR("J:A", (0.0, 0.0, 0.0), None, "P:A"),
            CanonicalRigJointIR("J:A1", (0.0, 1.0, 0.0), "J:A", "P:A1"),
            CanonicalRigJointIR("J:B", (2.0, 0.0, 0.0), None, "P:B"),
        ),
        root_joint_ids=("J:A", "J:B"),
        mesh_lineage_hash="1" * 64,
        mesh_carrier_evidence_hash="2" * 64,
        mesh_carrier_topology_hash="3" * 64,
        mesh_carrier_geometry_hash="4" * 64,
        feature_evidence_binding_hash="5" * 64,
        proposal_evidence_hash="6" * 64,
        structure_qualification_hash="7" * 64,
        rig_lineage_hash="",
        metadata={
            "authority": "COMPILER_CANONICAL_RIG",
            "canonical_root_semantics": "NON_EMPTY_DEFORM_ROOT_SET",
            "learned_family_embedded_in_schema": False,
            "frame_observability_claimed": False,
        },
    )
    sealed = replace(value, rig_lineage_hash=canonical_rig_lineage_hash_v1(value))
    assert canonical_rig_from_dict_v1(sealed.to_dict()) == sealed


def test_canonical_rig_schema_rejects_disconnected_or_mismatched_root_set():
    value = CanonicalRigIR(
        joints=(
            CanonicalRigJointIR("J:A", (0.0, 0.0, 0.0), None, "P:A"),
            CanonicalRigJointIR("J:B", (1.0, 0.0, 0.0), None, "P:B"),
        ),
        root_joint_ids=("J:A",),
        mesh_lineage_hash="1" * 64,
        mesh_carrier_evidence_hash="2" * 64,
        mesh_carrier_topology_hash="3" * 64,
        mesh_carrier_geometry_hash="4" * 64,
        feature_evidence_binding_hash="5" * 64,
        proposal_evidence_hash="6" * 64,
        structure_qualification_hash="7" * 64,
        rig_lineage_hash="",
        metadata={
            "authority": "COMPILER_CANONICAL_RIG",
            "canonical_root_semantics": "NON_EMPTY_DEFORM_ROOT_SET",
            "learned_family_embedded_in_schema": False,
            "frame_observability_claimed": False,
        },
    )
    sealed = replace(value, rig_lineage_hash=canonical_rig_lineage_hash_v1(value))
    with pytest.raises(QualificationError, match="ROOT_SET_DRIFT"):
        canonical_rig_from_dict_v1(sealed.to_dict())


def test_canonical_rig_rejects_parent_reselection():
    mesh = _mesh()
    qualified = _qualified(mesh)
    report = dict(qualified.qualification_report)
    report["parent_reselection_performed"] = True
    with pytest.raises(QualificationError, match="PARENT_RESELECTION_FORBIDDEN"):
        mint_canonical_rig_v1(
            mesh,
            proposal=_proposal(),
            qualified_skeleton=replace(qualified, qualification_report=report),
            feature_evidence_binding_hash=SURFACE_HASH,
        )


def test_canonical_rig_rejects_mesh_binding_drift():
    mesh = _mesh()
    qualified = _qualified(mesh)
    bindings = dict(qualified.qualification_report["authority_bindings"])
    bindings["canonical_mesh"] = "f" * 64
    report = {**qualified.qualification_report, "authority_bindings": bindings}
    with pytest.raises(QualificationError, match="AUTHORITY_BINDING_DRIFT:canonical_mesh"):
        mint_canonical_rig_v1(
            mesh,
            proposal=_proposal(),
            qualified_skeleton=replace(qualified, qualification_report=report),
            feature_evidence_binding_hash=SURFACE_HASH,
        )


def test_canonical_rig_rejects_feature_evidence_becoming_second_parent():
    mesh = _mesh()
    with pytest.raises(QualificationError, match="PROPOSAL_EVIDENCE_BINDING_DRIFT"):
        mint_canonical_rig_v1(
            mesh,
            proposal=_proposal(),
            qualified_skeleton=_qualified(mesh),
            feature_evidence_binding_hash="e" * 64,
        )
