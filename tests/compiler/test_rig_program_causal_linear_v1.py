from __future__ import annotations

from dataclasses import replace

import pytest

from compiler.realsas_compiler_core.rig_program_ir_v1 import (
    RigProgramControlProposalIR,
    RigProgramJointProposalIR,
    RigProgramLocalFrameProposalIR,
    RigProgramProposalIR,
    qualify_rig_program_v1,
    seal_rig_program_proposal_v1,
)
from compiler.realsas_compiler_core.types import QualificationError


IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _proposal(count: int = 4) -> RigProgramProposalIR:
    joints = tuple(
        RigProgramJointProposalIR(
            proposal_joint_id=f"J:{index}",
            P=(0.0, float(index), 0.0),
            parent_proposal_joint_id=None if index == 0 else f"J:{index - 1}",
            sequence_index=index,
        )
        for index in range(count)
    )
    frames = tuple(
        RigProgramLocalFrameProposalIR(j.proposal_joint_id, IDENTITY, ("RX",))
        for j in joints
    )
    value = RigProgramProposalIR(
        joints=joints,
        local_frames=frames,
        controls=(),
        chains=(),
        constraints=(),
        attachment_anchors=(),
        mesh_lineage_hash="1" * 64,
        feature_evidence_binding_hash="2" * 64,
        model_provenance="causal-linear-test",
    )
    return seal_rig_program_proposal_v1(value)


def test_long_causal_joint_chain_qualifies_without_per_node_root_walks():
    # Large enough to catch accidental recursive/depth-limited implementations;
    # no wall-clock threshold is used because CI timing is not scientific truth.
    proposal = _proposal(2048)
    qualified = qualify_rig_program_v1(proposal)
    assert qualified.qualification_report["joint_count"] == 2048
    assert qualified.qualification_report["deform_root_count"] == 1
    assert (
        qualified.qualification_report["qualification_complexity_contract"]
        == "LINEAR_IN_DECLARED_RIG_PROGRAM_SIZE"
    )


def test_noncausal_joint_parent_is_rejected_instead_of_graph_search_or_repair():
    proposal = _proposal(3)
    joints = list(proposal.joints)
    joints[1] = replace(joints[1], parent_proposal_joint_id="J:2")
    bad = seal_rig_program_proposal_v1(
        replace(proposal, joints=tuple(joints), proposal_lineage_hash="")
    )
    with pytest.raises(QualificationError, match="JOINT_PARENT_NOT_CAUSAL"):
        qualify_rig_program_v1(bad)


def test_noncausal_control_parent_is_rejected_in_single_pass():
    proposal = _proposal(1)
    controls = (
        RigProgramControlProposalIR(
            "C:child",
            "FK",
            ("J:0",),
            "C:parent",
            (0.0, 0.0, 0.0),
        ),
        RigProgramControlProposalIR(
            "C:parent",
            "ROOT",
            ("J:0",),
            None,
            (0.0, 0.0, 0.0),
        ),
    )
    bad = seal_rig_program_proposal_v1(
        replace(proposal, controls=controls, proposal_lineage_hash="")
    )
    with pytest.raises(QualificationError, match="CONTROL_PARENT_NOT_CAUSAL"):
        qualify_rig_program_v1(bad)
