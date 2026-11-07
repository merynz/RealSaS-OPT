"""AXIS V6 research adapter; structural qualification != training/IK PASS."""
from __future__ import annotations

from dataclasses import replace

import pytest

from tools.corpus.derive_corpus4008_rigprogram_prior_v1 import discover
from models.axis.experimental_derived_control_adapter_v1 import (
    adapt_blind_prior_to_rigprogram_parts,
)
from compiler.realsas_compiler_core.rig_program_ir_v1 import (
    RigProgramJointProposalIR,
    RigProgramLocalFrameProposalIR,
    RigProgramProposalIR,
    seal_rig_program_proposal_v1,
    qualify_rig_program_v1,
)


def bones():
    return [{
        "bone_id":str(i),
        "parent_bone_id":str(i-1) if i else None,
        "head_source":[float(i),0.,0.],
        "tail_source":[float(i+1),0.,0.],
        "use_deform":True,
        "skin_weight_mass":float(3+i),
        "skin_influenced_vertices":12+i,
        "source_motion_curve_count":2*i,
    } for i in range(4)]


def fixture():
    b=bones()
    p=discover(b,max_depth=4,max_candidates=8)
    values=adapt_blind_prior_to_rigprogram_parts(
        p,
        source_to_proposal_joint_id={str(i):f"P:J:{i:04d}" for i in range(4)},
        source_joint_parent_id={str(i):str(i-1) if i else None
                                for i in range(4)},
        source_joint_tail_P={str(i):(i+1.,0.,0.) for i in range(4)},
        normalization_receipt_sha256="f"*64,
        joint_coordinate_domain="IMPORTED_SOURCE_WORLD",
        max_controls=2,
    )
    return p,values


def test_typed_axis_program_without_native_constraint_or_compiler_repair():
    prior,parts=fixture()
    assert prior["validated_working_derived_IK_count"]==0
    assert 1<=parts["provider_control_proposal_count"]<=2
    assert not parts["training_authorized"]
    assert not parts["original_native_ik_constraints_copied"]
    assert all(c.control_kind=="IK_EFFECTOR" for c in parts["controls"])
    assert all(c.operator_id=="IK" for c in parts["constraints"])
    assert all(c.metadata["authority"]=="INFERRED_IK_PRIOR"
               and not c.metadata["solver_functionality_proven"]
               and c.metadata["normalization_receipt_sha256"]=="f"*64
               for c in parts["controls"])
    joints=tuple(
        RigProgramJointProposalIR(
            proposal_joint_id=f"P:J:{i:04d}",
            P=(float(i),0.,0.),
            parent_proposal_joint_id=(f"P:J:{i-1:04d}" if i else None),
            sequence_index=i,
        ) for i in range(4))
    identity=((1.,0.,0.),(0.,1.,0.),(0.,0.,1.))
    frames=tuple(RigProgramLocalFrameProposalIR(
        proposal_joint_id=j.proposal_joint_id,
        basis=identity,
        dof_labels=("RX","RY","RZ"),
    ) for j in joints)
    program=seal_rig_program_proposal_v1(RigProgramProposalIR(
        joints=joints,local_frames=frames,
        controls=parts["controls"],
        chains=parts["chains"],
        constraints=parts["constraints"],
        attachment_anchors=(),
        mesh_lineage_hash="1"*64,
        feature_evidence_binding_hash="2"*64,
        model_provenance="RESEARCH_SOURCE_ONLY_PRIOR_NOT_LEARNED_AX_V6",
    ))
    q=qualify_rig_program_v1(program)
    assert q.qualification_report["constraint_solver_executed"] is False
    assert q.qualification_report["control_synthesis_performed"] is False
    assert q.proposal.controls[0].metadata["native_ik_controller_proven"] is False


def test_adapter_rejects_wrong_coordinate_domain_and_parent_graph():
    p,_=fixture()
    args=dict(
        source_to_proposal_joint_id={str(i):f"J:{i}" for i in range(4)},
        source_joint_parent_id={str(i):(str(i-1) if i else None)
                                for i in range(4)},
        source_joint_tail_P={str(i):(i+1.,0.,0.) for i in range(4)},
        normalization_receipt_sha256="f"*64,
        joint_coordinate_domain="IMPORTED_SOURCE_WORLD",
        max_controls=2)
    with pytest.raises(ValueError,match="MIXED_SOURCE_AND_NORMALIZED_COORDINATES"):
        adapt_blind_prior_to_rigprogram_parts(
            p,**{**args,"joint_coordinate_domain":"IRIS_NORMALIZED"})
    with pytest.raises(ValueError,match="NORMALIZATION_RECEIPT_REQUIRED"):
        adapt_blind_prior_to_rigprogram_parts(
            p,**{**args,"normalization_receipt_sha256":"unknown"})
    with pytest.raises(ValueError,match="CHAIN_NOT_CAUSAL_CONTIGUOUS"):
        bad={**args,"source_joint_parent_id":{**args["source_joint_parent_id"],"3":"0"}}
        adapt_blind_prior_to_rigprogram_parts(p,**bad)


def test_tampered_oracle_control_cannot_become_provider_proposal():
    prior,_=fixture()
    prior["proposals"][0]["operator_proposal"]="IK_NATIVE_TEACHER"
    with pytest.raises(ValueError,match="BLIND_PRIOR_SHA"):
        adapt_blind_prior_to_rigprogram_parts(
            prior,
            source_to_proposal_joint_id={str(i):f"J:{i}" for i in range(4)},
            source_joint_parent_id={str(i):str(i-1) if i else None
                                    for i in range(4)},
            source_joint_tail_P={str(i):(i+1.,0.,0.) for i in range(4)},
            normalization_receipt_sha256="f"*64,
            joint_coordinate_domain="IMPORTED_SOURCE_WORLD",
        )
