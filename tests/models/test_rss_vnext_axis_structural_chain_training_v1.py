from __future__ import annotations

"""Structural source-native CHAIN labels require no invented IK intervention."""
import pytest
import torch

from models.axis.rig_program_model_v6 import (
    AXISRigProgramConfigV6, AXISRigProgramModelV6, AXISSequenceV6,
    RigProgramEventV6 as E, RigTokenKindV6 as K,
    qualified_axis_rig_skin_sequence_loss_v6,
)


def _setup(*, with_control: bool = False):
    torch.manual_seed(1123)
    torch.set_num_threads(1)
    cfg=AXISRigProgramConfigV6(
        scene_dim=32,hidden_dim=32,layers=1,heads=4,
        max_sequence_length=64,max_pointer_entities=16,skin_vocab_size=16,
    )
    model=AXISRigProgramModelV6(cfg)
    events=[
        E(K.BOS),
        E(K.JOINT,coord_indices=(1,2,3),pointer_a=-1),
        E(K.JOINT,coord_indices=(4,5,6),pointer_a=0),
        E(K.FRAMES_START),
        E(K.FRAME,pointer_a=0,frame_6d=(1.,0.,0.,0.,1.,0.)),
        E(K.FRAME,pointer_a=1,frame_6d=(1.,0.,0.,0.,1.,0.)),
        E(K.CONTROLS_START),
    ]
    if with_control:
        events.append(E(K.CONTROL,coord_indices=(3,3,3),pointer_a=-1,aux_type_id=1))
        events.append(E(K.CONTROL_TARGET,pointer_a=1))
    events.extend([
        E(K.CHAINS_START),
        E(K.CHAIN,pointer_a=0,pointer_b=1),
        E(K.CONSTRAINTS_START),
        E(K.ANCHORS_START),
        E(K.SKIN_START),
    ])
    events.extend(E(K.SKIN_CODE,skin_code_id=i%16) for i in range(8))
    events.append(E(K.EOS))
    return model, AXISSequenceV6.from_events(events,device=torch.device("cpu"))


def _loss(model,seq,*,structural_chains_qualified=False,control_intervention_qualified=False):
    return qualified_axis_rig_skin_sequence_loss_v6(
        model,torch.zeros(1,4,32),seq,
        tokens_per_joint=4,executable_constraint_operators=("IK",),
        structural_chains_qualified=structural_chains_qualified,
        control_intervention_qualified=control_intervention_qualified,
    )[0]


def test_native_chain_tokens_train_without_control_intervention():
    model,seq=_setup()
    with pytest.raises(ValueError,match="STRUCTURAL_CHAIN_PROOF_REQUIRED"):
        _loss(model,seq)
    loss=_loss(model,seq,structural_chains_qualified=True)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.kind_head.weight.grad is not None
    assert torch.count_nonzero(model.kind_head.weight.grad)>0


def test_structural_chain_receipt_cannot_license_ik_control():
    model,seq=_setup(with_control=True)
    with pytest.raises(ValueError,match="CONTROL_INTERVENTION_PROOF_REQUIRED"):
        _loss(model,seq,structural_chains_qualified=True)
    assert torch.isfinite(_loss(
        model,seq,structural_chains_qualified=True,
        control_intervention_qualified=True,
    ))


def test_product_token_budget_can_hold_real_auxiliary_rigs():
    from models.rss_vnext.contracts import RSSvNextConfig
    cfg=RSSvNextConfig()
    # 190 joints, 190 rest frames, four hard skin code tokens per joint,
    # plus section sentinels, twenty auxiliary chains, and thirty controller
    # events. This only proves syntactic capacity, NOT training or latency.
    source_scale_minimum=190+190+190*cfg.skin_tokens_per_joint+8+20+30
    assert source_scale_minimum == 1198
    assert cfg.control_max_seq >= source_scale_minimum
    assert cfg.axis_config().max_sequence_length >= cfg.control_max_seq


def test_causal_generation_has_only_descendant_chain_targets():
    from models.axis.rig_program_model_v6 import causal_chain_descendants_v6
    parents=(-1,0,1,0,3)
    assert causal_chain_descendants_v6(parents,0)==(0,1,2,3,4)
    assert causal_chain_descendants_v6(parents,1)==(1,2)
    assert causal_chain_descendants_v6(parents,3)==(3,4)
    with pytest.raises(ValueError,match="CHAIN_START_OUT_OF_RANGE"):
        causal_chain_descendants_v6(parents,5)
    with pytest.raises(ValueError,match="CHAIN_PARENT_GRAPH_INVALID"):
        causal_chain_descendants_v6((-1,2,0),0)


def test_invalid_chain_teacher_rejected_even_when_source_claims_proof():
    model,seq=_setup()
    chain_token=int(torch.where(seq.kind_ids[0]==int(K.CHAIN))[0][0])
    seq.pointer_a[0,chain_token]=1
    seq.pointer_b[0,chain_token]=0
    with pytest.raises(ValueError,match="NONANCESTRAL_CHAIN"):
        _loss(model,seq,structural_chains_qualified=True)
