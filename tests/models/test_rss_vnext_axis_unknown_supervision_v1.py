from __future__ import annotations

"""FBX lacks authored controller != source proves there are zero controllers."""
import pytest
import torch

from models.axis.rig_program_model_v6 import (
    RigTokenKindV6 as K, RigProgramEventV6 as E, AXISSequenceV6,
    qualified_axis_v6_kind_mask_v1,
    qualified_axis_rig_skin_sequence_loss_v6,
)
from tests.models.test_rss_vnext_axis_structural_chain_training_v1 import _setup


@pytest.fixture(autouse=True)
def fixed():
    torch.manual_seed(20261119)
    torch.set_num_threads(1)


def _index(seq, kind):
    idx=torch.where(seq.kind_ids[0]==int(kind))[0]
    assert len(idx)==1
    return int(idx[0])


def _no_chain():
    return AXISSequenceV6.from_events([
        E(K.BOS),
        E(K.JOINT,pointer_a=-1),
        E(K.FRAMES_START),
        E(K.FRAME,pointer_a=0,frame_6d=(1.,0.,0.,0.,1.,0.)),
        E(K.CONTROLS_START),
        E(K.CHAINS_START),
        E(K.CONSTRAINTS_START),
        E(K.ANCHORS_START),
        E(K.SKIN_START),
        *(E(K.SKIN_CODE,skin_code_id=i) for i in range(4)),
        E(K.EOS),
    ],device=torch.device("cpu"))


def test_unknown_control_chain_and_anchor_are_not_negative_kind_labels():
    seq=_no_chain()
    masked=qualified_axis_v6_kind_mask_v1(
        seq,control_consequences_qualified=False,
        structural_chains_qualified=False,
    )
    assert masked.shape==(1,seq.kind_ids.shape[1]-1)
    for kind in (K.CHAINS_START,K.CONSTRAINTS_START,K.ANCHORS_START,K.SKIN_START):
        # With unqualified missing sections, each observed empty section
        # transition is UNKNOWN rather than a witnessed negative.
        assert not bool(masked[0,_index(seq,kind)-1])
    joint_indices=torch.where(seq.kind_ids[0]==int(K.JOINT))[0]
    assert all(bool(masked[0,int(i)-1]) for i in joint_indices)
    assert bool(masked[0,_index(seq,K.FRAMES_START)-1])


def test_real_chain_receipt_does_not_claim_authored_controls_are_absent():
    _,seq=_setup()
    masked=qualified_axis_v6_kind_mask_v1(
        seq,control_consequences_qualified=False,
        structural_chains_qualified=True,
    )
    assert bool(masked[0,_index(seq,K.CONSTRAINTS_START)-1])
    assert not bool(masked[0,_index(seq,K.CHAINS_START)-1])
    assert not bool(masked[0,_index(seq,K.ANCHORS_START)-1])


def test_explicit_source_verified_absence_trains_negative_without_guessing():
    seq=_no_chain()
    all_known=qualified_axis_v6_kind_mask_v1(
        seq,control_consequences_qualified=False,
        control_absence_qualified=True,
        chain_absence_qualified=True,
    )
    assert all_known.all()
    with pytest.raises(ValueError,match="SECTION_AUTHORITY_INVALID"):
        qualified_axis_v6_kind_mask_v1(
            seq,control_consequences_qualified=True,
            control_absence_qualified=True,
        )


def test_source_native_frame_does_not_claim_unknown_dof_limits():
    model,seq=_setup()
    def run(dof):
        return qualified_axis_rig_skin_sequence_loss_v6(
            model,torch.zeros(1,4,32),seq,
            tokens_per_joint=4,
            executable_constraint_operators=("IK",),
            control_intervention_qualified=False,
            structural_chains_qualified=True,
            frame_dof_qualified=dof,
        )
    loss,parts=run(False)
    assert torch.isfinite(loss)
    assert "frame" in parts
    assert "dof" not in parts
    loss.backward()
    assert model.kind_head.weight.grad is not None
    _,parts=run(True)
    assert "dof" in parts
