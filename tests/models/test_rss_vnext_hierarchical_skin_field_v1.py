from __future__ import annotations

"""AXIS-causal two-level MIRA skin: learned gate, exact identity, hard replay."""
from dataclasses import replace
from dataclasses import asdict

import pytest
import torch

from models.axis.rig_program_model_v6 import (
    AXISSequenceV6, RigProgramEventV6 as E, RigTokenKindV6 as K,
)
from models.rss_vnext import RSSvNextConfig
from models.rss_vnext.hierarchical_skin_field_v1 import (
    RSSvNextLearnedChainMixtureV1,
    axis_joint_group_ids_v1, axis_joint_group_ids_from_cas_events_v1,
)
from models.rss_vnext.unified_character_model import RSSvNextSkinCoupler
from compiler.realsas_compiler_core.rss_vnext_discrete_skin_replay_v1 import (
    replay_rss_vnext_discrete_skin_candidate_v1,
)
from compiler.realsas_compiler_core.types import QualificationError


@pytest.fixture(autouse=True)
def fixed_cpu():
    torch.manual_seed(20261119)
    torch.set_num_threads(1)


def _branched_events():
    # A branching mechanical rig is NOT one ancestor -> end chain.
    # The common shoulder is UNKNOWN; branches have unique membership.
    return [
        E(K.BOS),
        E(K.JOINT,pointer_a=-1),  # shoulder
        E(K.JOINT,pointer_a=0),   # left wing elbow
        E(K.JOINT,pointer_a=1),   # left wing tip
        E(K.JOINT,pointer_a=0),   # right wing elbow
        E(K.JOINT,pointer_a=3),   # right wing tip
        E(K.CHAINS_START),
        E(K.CHAIN,pointer_a=0,pointer_b=2),
        E(K.CHAIN,pointer_a=0,pointer_b=4),
        E(K.CONSTRAINTS_START), E(K.ANCHORS_START),
        E(K.SKIN_START), E(K.EOS),
    ]


def test_branches_have_distinct_group_gate_but_shared_root_stays_unknown():
    events=_branched_events()
    seq=AXISSequenceV6.from_events(events,device=torch.device("cpu"))
    expected=torch.tensor([-1,0,0,1,1])
    torch.testing.assert_close(axis_joint_group_ids_v1(seq),expected,rtol=0,atol=0)
    raw=[{**asdict(e),"kind":int(e.kind)} for e in events]
    torch.testing.assert_close(
        axis_joint_group_ids_from_cas_events_v1(raw),expected,rtol=0,atol=0,
    )
    raw[8]["pointer_b"]=99
    with pytest.raises(ValueError,match="CHAIN_PATH_JOINT_POINTER_INVALID"):
        axis_joint_group_ids_from_cas_events_v1(raw)


def test_learned_group_gate_is_not_posthoc_flat_weight_summation():
    field=RSSvNextLearnedChainMixtureV1(
        vertex_feature_dim=9,joint_feature_dim=11,hidden_dim=16,
    )
    logits=torch.randn(8,5,requires_grad=True)
    vertex=torch.randn(8,9)
    joints=torch.randn(5,11)
    ids=torch.tensor([-1,0,0,1,1])
    output=field(
        joint_logits=logits,vertex_features=vertex,
        joint_features=joints,joint_group_ids=ids,
    )
    assert output.shape==logits.shape
    torch.testing.assert_close(output.sum(-1),torch.ones(8),atol=1e-6,rtol=0)
    assert (output > 0).all()
    # For positive group+joint targets gradients reach actual learned
    # group gate, not only the old joint-pair skin score.
    truth=torch.zeros_like(output);truth[:,3]=1.
    loss=(output-truth).square().mean()
    loss.backward()
    assert logits.grad is not None and float(logits.grad.abs().sum())>0
    assert any(p.grad is not None and float(p.grad.abs().sum())>0
               for p in field.group_pair.parameters())


def test_no_source_chain_is_qualified_unknown_bucket_not_a_fake_label():
    field=RSSvNextLearnedChainMixtureV1(
        vertex_feature_dim=3,joint_feature_dim=5,hidden_dim=8,
    )
    logits=torch.randn(6,55)
    result=field(
        joint_logits=logits,vertex_features=torch.randn(6,3),
        joint_features=torch.randn(55,5),
        joint_group_ids=torch.full((55,),-1,dtype=torch.long),
    )
    torch.testing.assert_close(
        result,logits.softmax(-1),rtol=0.,atol=0.,
    )


@pytest.mark.parametrize("mode",["DOT_PRODUCT_BASELINE","POINT_QUERY_CROSS_ATTENTION"])
def test_hard_fsq_55_joint_group_replay_exact_and_sparse(mode):
    cfg=replace(
        RSSvNextConfig.tiny_for_test(),skin_decoder_mode=mode,
        skin_max_influences=8,
    )
    net=RSSvNextSkinCoupler(cfg)
    local=torch.randn(1,6,cfg.iris.local_feature_dim)
    xyz=torch.randn(1,6,3)
    normals=torch.nn.functional.normalize(torch.randn(1,6,3),dim=-1)
    joints=torch.randn(1,55,cfg.control_hidden_dim)
    ids=torch.randint(0,net.codec.quantizer.vocab_size,
                      (1,55,cfg.skin_tokens_per_joint))
    groups=torch.tensor([0]*27+[1]*28,dtype=torch.long)[None]
    net.train()
    result=net.decode_discrete_skin_v1(
        local,xyz,joints,ids,vertex_normals=normals,
        joint_group_ids=groups,
    )
    assert (result.detach()>0).sum(-1).max().item()==8
    assert torch.allclose(
        result.detach().sum(-1),torch.ones(1,6),atol=1e-6,
    )
    ((result[...,0]-1.)**2).sum().backward()
    assert any(
        p.grad is not None and float(p.grad.abs().sum())>0
        for p in net.chain_mixture.parameters()
    )
    net.eval()
    with torch.no_grad():
        replay=net.decode_discrete_skin_v1(
            local,xyz,joints,ids,vertex_normals=normals,
            joint_group_ids=groups,
        )
    torch.testing.assert_close(result.detach(),replay,rtol=0,atol=0)
    # Crucial: ignoring the causal group evidence is NOT equivalent.
    with torch.no_grad():
        flat=net.decode_discrete_skin_v1(
            local,xyz,joints,ids,vertex_normals=normals,
        )
    assert float((flat-replay).abs().max())>1e-6


def test_exact_group_replay_from_compiler_cas_joint_chain_events():
    from tests.models.test_rss_vnext_compiler_discrete_skin_replay_v1 import (
        _source, _replay,
    )
    net,cfg,mesh,dynamic,_=_source("DOT_PRODUCT_BASELINE")
    # Source mini-rig has two independent causal joints and one chain on
    # only the second, providing two real groups: UNKNOWN and CHAIN-0.
    events=[
        {"kind":int(K.JOINT),"pointer_a":-1},
        {"kind":int(K.JOINT),"pointer_a":-1},
        {"kind":int(K.CHAIN),"pointer_a":1,"pointer_b":1},
    ] + [
        {"kind":int(K.SKIN_CODE),"skin_code_id":c}
        for c in (
            [0]*cfg.skin_tokens_per_joint+
            [cfg.axis_config().skin_vocab_size-1]*cfg.skin_tokens_per_joint
        )
    ]
    tensor=torch.tensor(axis_joint_group_ids_from_cas_events_v1(events))[None]
    with torch.no_grad():
        hard=net.mira.decode_discrete_skin_v1(
            torch.tensor(mesh["local_features"])[None],
            torch.tensor(mesh["rest_xyz"])[None],
            torch.tensor(dynamic["skin_joint_hidden_states"])[None],
            torch.tensor([[
                [0]*cfg.skin_tokens_per_joint,
                [cfg.axis_config().skin_vocab_size-1]*cfg.skin_tokens_per_joint,
            ]]),vertex_normals=torch.tensor(mesh["normals_xyz"])[None],
            joint_group_ids=tensor,
        )
    dynamic=dict(dynamic)
    dynamic["discrete_skin_weights"]=hard[0].tolist()
    difference=(
        torch.tensor(dynamic["weights"])-hard[0]
    ).abs()
    # Independent evidence validator checks the measured soft/hard error
    # before reading the replay result. This metric must match these bytes.
    dynamic["soft_to_discrete_max_abs_difference"]=float(difference.max())
    dynamic["soft_to_discrete_mean_abs_difference"]=float(difference.mean())
    receipt=_replay(net,cfg,mesh,dynamic,events)
    assert receipt.dense_weight_max_abs_replay_error<1e-6
    # A semantically different, even structurally legal, group program is
    # not equal to the original frozen hard-FSQ candidate's representation.
    erased=[e for e in events if e["kind"]!=int(K.CHAIN)]
    with pytest.raises(QualificationError,match="DISCRETE_WEIGHT_MISMATCH"):
        _replay(net,cfg,mesh,dynamic,erased)
