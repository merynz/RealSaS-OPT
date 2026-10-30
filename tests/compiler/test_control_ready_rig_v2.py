from __future__ import annotations

from dataclasses import replace

import pytest

from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
    CanonicalMeshIR,
    CanonicalMeshVertexIR,
    canonical_mesh_lineage_hash_v1,
)
from compiler.realsas_compiler_core.canonical_rig_ir_v2 import (
    canonical_rig_from_dict_v2,
    mint_canonical_rig_v2,
)
from compiler.realsas_compiler_core.rig_program_ir_v1 import (
    RigProgramAttachmentAnchorProposalIR,
    RigProgramChainProposalIR,
    RigProgramConstraintProposalIR,
    RigProgramControlProposalIR,
    RigProgramJointProposalIR,
    RigProgramLocalFrameProposalIR,
    RigProgramProposalIR,
    qualify_rig_program_v1,
    seal_rig_program_proposal_v1,
)
from compiler.realsas_compiler_core.skin_token_ir_v1 import (
    JointSkinTokenRowIR,
    SkinTokenProposalIR,
    seal_skin_token_proposal_v1,
    validate_skin_token_proposal_v1,
)
from compiler.realsas_compiler_core.types import QualificationError, SurfaceSupportBinding

SURFACE_HASH = "1" * 64


def _support(surface_id: str) -> SurfaceSupportBinding:
    return SurfaceSupportBinding(mode="EXACT_SURFACE_NODE", coefficients=((surface_id, 1.0),))


def _mesh() -> CanonicalMeshIR:
    value = CanonicalMeshIR(
        vertices=(
            CanonicalMeshVertexIR("MV:0", _support("S:0"), (0.0, 0.0, 0.0), "body", "C:0"),
            CanonicalMeshVertexIR("MV:1", _support("S:1"), (0.0, 1.0, 0.0), "body", "C:1"),
            CanonicalMeshVertexIR("MV:2", _support("S:2"), (0.2, 2.0, 0.0), "body", "C:2"),
        ),
        faces=(("MV:0", "MV:1", "MV:2"),),
        edges=(("MV:0", "MV:1"), ("MV:1", "MV:2"), ("MV:0", "MV:2")),
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


def _proposal(mesh: CanonicalMeshIR) -> RigProgramProposalIR:
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    value = RigProgramProposalIR(
        joints=(
            RigProgramJointProposalIR("Jp:root", (0.0, 0.0, 0.0), None, ("S:0",), 0),
            RigProgramJointProposalIR("Jp:upper", (0.0, 1.0, 0.0), "Jp:root", ("S:1",), 1),
            RigProgramJointProposalIR("Jp:end", (0.2, 2.0, 0.0), "Jp:upper", ("S:2",), 2),
        ),
        local_frames=(
            RigProgramLocalFrameProposalIR("Jp:root", identity, ("ROT_Z",)),
            RigProgramLocalFrameProposalIR("Jp:upper", identity, ("ROT_Z",)),
            RigProgramLocalFrameProposalIR("Jp:end", identity, ("ROT_Z",)),
        ),
        controls=(
            RigProgramControlProposalIR("Cp:root", "ROOT_CONTROL", ("Jp:root",), None, (0.0, 0.0, 0.0)),
            RigProgramControlProposalIR("Cp:eff", "IK_EFFECTOR", ("Jp:end",), "Cp:root", (0.2, 2.0, 0.0)),
            RigProgramControlProposalIR("Cp:pole", "POLE_CONTROL", ("Jp:upper",), "Cp:root", (1.0, 1.0, 0.0)),
        ),
        chains=(RigProgramChainProposalIR("Kp:limb", ("Jp:root", "Jp:upper", "Jp:end")),),
        constraints=(
            RigProgramConstraintProposalIR(
                "Qp:ik", "IK_CHAIN_SOLVE_V1", ("Cp:eff", "Cp:pole"),
                ("Jp:upper", "Jp:end"), "Kp:limb",
            ),
        ),
        attachment_anchors=(
            RigProgramAttachmentAnchorProposalIR("Ap:grip", parent_proposal_control_id="Cp:eff"),
        ),
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        feature_evidence_binding_hash=SURFACE_HASH,
        model_provenance="RealSaS.AXIS.RigProgram.TokenCausal.v6",
        metadata={"compiler_role": "VALIDATE_AND_MATERIALIZE_ONLY"},
    )
    return seal_rig_program_proposal_v1(value)


def test_rig_program_v2_preserves_learned_program_without_compiler_synthesis():
    mesh = _mesh()
    proposal = _proposal(mesh)
    qualified = qualify_rig_program_v1(proposal)
    report = qualified.qualification_report
    assert report["parent_reselection_performed"] is False
    assert report["control_synthesis_performed"] is False
    assert report["constraint_synthesis_performed"] is False
    assert report["constraint_solver_executed"] is False

    rig = mint_canonical_rig_v2(mesh, qualified_program=qualified)
    assert rig.metadata["constraint_execution_authority"] == "MECHANICS"
    assert rig.metadata["skin_binding_target"] == "DEFORM_STRUCTURE_HASH"
    assert len(rig.controls) == 3
    assert len(rig.constraints) == 1
    assert len(rig.attachment_anchors) == 1
    assert rig.deform_structure_hash != rig.control_program_hash
    assert canonical_rig_from_dict_v2(rig.to_dict()) == rig


def test_rig_program_v2_rejects_noncontiguous_chain_instead_of_repairing_it():
    mesh = _mesh()
    p = _proposal(mesh)
    bad = replace(p, chains=(RigProgramChainProposalIR("Kp:bad", ("Jp:root", "Jp:end")),), proposal_lineage_hash="")
    bad = seal_rig_program_proposal_v1(bad)
    with pytest.raises(QualificationError, match="CHAIN_NOT_PARENT_CONTIGUOUS"):
        qualify_rig_program_v1(bad)


def test_skin_tokens_are_bound_to_exact_rig_proposal_and_never_repaired():
    mesh = _mesh()
    p = _proposal(mesh)
    value = SkinTokenProposalIR(
        rows=tuple(JointSkinTokenRowIR(j.proposal_joint_id, (1, 3, 7, 2)) for j in p.joints),
        rig_program_proposal_hash=p.proposal_lineage_hash,
        carrier_evidence_hash=mesh.carrier_evidence_hash,
        carrier_topology_hash=mesh.carrier_topology_hash,
        carrier_geometry_hash=mesh.carrier_geometry_hash,
        codec_architecture_id="RealSaS.MIRA.FSQSkinCodec.v6",
        codec_checkpoint_hash="a" * 64,
        codebook_size=16,
        tokens_per_joint=4,
        model_provenance="RealSaS.AXIS.RigProgram.TokenCausal.v6",
    )
    value = seal_skin_token_proposal_v1(value)
    validate_skin_token_proposal_v1(
        value,
        expected_proposal_joint_ids=tuple(j.proposal_joint_id for j in p.joints),
        expected_rig_program_proposal_hash=p.proposal_lineage_hash,
    )

    bad_rows = list(value.rows)
    bad_rows[0] = replace(bad_rows[0], token_ids=(1, 3, 99, 2))
    bad = seal_skin_token_proposal_v1(replace(value, rows=tuple(bad_rows), proposal_lineage_hash=""))
    with pytest.raises(QualificationError, match="TOKEN_ID_OUT_OF_DOMAIN"):
        validate_skin_token_proposal_v1(
            bad,
            expected_proposal_joint_ids=tuple(j.proposal_joint_id for j in p.joints),
            expected_rig_program_proposal_hash=p.proposal_lineage_hash,
        )
