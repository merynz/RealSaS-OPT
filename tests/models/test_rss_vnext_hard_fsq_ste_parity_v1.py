from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from models.axis.rig_program_model_v6 import RigTokenKindV6
from tests.models.test_rss_vnext_unified_character_model import (
    model as make_model, sequence_two_joints,
)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(764)
    torch.set_num_threads(1)


@pytest.mark.parametrize("mode", ("DOT_PRODUCT_BASELINE", "POINT_QUERY_CROSS_ATTENTION"))
def test_differentiable_skin_forward_uses_exact_product_hard_fs_q_ids(mode):
    from models.rss_vnext import RSSvNextConfig
    cfg=replace(RSSvNextConfig.tiny_for_test(),skin_decoder_mode=mode)
    net=make_model(cfg).eval()
    sequence=sequence_two_joints()
    v=9
    local=torch.randn(1,v,cfg.iris.local_feature_dim)
    rest=torch.randn(1,v,3)
    normals=F.normalize(torch.randn_like(rest),dim=-1)
    state=torch.randn(1,sequence.kind_ids.shape[1],cfg.control_hidden_dim)
    vocab=net.mira.codec.quantizer.vocab_size
    # Nontrivial logits are not one-hot: the old soft expectation differs
    # from the hard token sequence that product actually exports.
    logits=torch.randn(1,sequence.kind_ids.shape[1],vocab,requires_grad=True)
    weights,valid,joints=net.mira(
        local,rest,state,logits,sequence,vertex_normals=normals,
    )
    joint_positions=torch.where(
        sequence.kind_ids[0]==int(RigTokenKindV6.JOINT),
    )[0]
    code_positions=torch.where(
        sequence.kind_ids[0]==int(RigTokenKindV6.SKIN_CODE),
    )[0]
    hard_ids=logits[:,code_positions-1,:].argmax(-1).reshape(
        1,len(joint_positions),cfg.skin_tokens_per_joint,
    )
    independent=net.mira.decode_discrete_skin_v1(
        local,rest,state[:,joint_positions-1],hard_ids,
        vertex_normals=normals,
    )
    assert bool(valid.all())
    assert weights.shape==independent.shape
    assert float((weights-independent).abs().max())<=1e-5
    # Product-hard forward does not turn the code head into a no-grad argmax.
    error=((weights[...,0]-.8)**2).mean()
    error.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum())>0


def test_hard_ste_token_policy_does_not_depend_on_current_teacher_skin_embedding():
    net=make_model().eval()
    seq=sequence_two_joints()
    cfg=net.cfg
    local=torch.randn(1,5,cfg.iris.local_feature_dim)
    rest=torch.randn(1,5,3)
    hidden=torch.randn(1,seq.kind_ids.shape[1],cfg.control_hidden_dim)
    logits=torch.randn(
        1,seq.kind_ids.shape[1],net.mira.codec.quantizer.vocab_size,
    )
    first=net.mira(local,rest,hidden,logits,seq)[0]
    edited=replace(seq,skin_code_ids=torch.randint(
        0,net.axis.config.skin_vocab_size,seq.skin_code_ids.shape,
    ))
    second=net.mira(local,rest,hidden,logits,edited)[0]
    assert torch.equal(first,second)
