from __future__ import annotations

"""Production MIRA support: actual zeros from training to hard-FSQ replay."""
import pytest
import torch

from models.rss_vnext import RSSvNextConfig
from models.rss_vnext.unified_character_model import RSSvNextSkinCoupler
from models.rss_vnext.sparse_skin_support_v1 import sparse_skin_support_v1
from models.rss_vnext.experts.mira.conditioning_v6 import (
    MIRAConditioningV6, MIRACarrierViewV2,
    MIRADeformSkeletonViewV2, _SkeletonJointViewV2,
    carrier_skin_proposal_from_decoded_v6,
)
from compiler.realsas_compiler_core.carrier_skin_v1 import (
    qualify_carrier_skin_v1,
)


@pytest.fixture(autouse=True)
def fixed_cpu():
    torch.manual_seed(7062)
    torch.set_num_threads(1)


def test_source_positive_joint_outside_support_gets_training_gradient():
    logits=torch.linspace(-3.,3.,55).reshape(1,1,55).requires_grad_()
    dense=logits.softmax(-1)
    soft=sparse_skin_support_v1(
        dense,max_influences=8,straight_through=True,
    )
    hard=sparse_skin_support_v1(
        dense.detach(),max_influences=8,straight_through=False,
    )
    torch.testing.assert_close(soft.detach(),hard,rtol=0,atol=0)
    assert int((soft.detach()>0).sum())==8
    assert int((soft.detach()==0).sum())==47
    assert float(soft.detach().sum())==pytest.approx(1,abs=1.e-6)
    assert float(soft.detach()[0,0,0])==0.
    # Source ground truth says small-probability joint zero SHOULD be active.
    target=torch.zeros_like(soft)
    target[...,0]=1.
    (soft-target).square().sum().backward()
    assert logits.grad is not None
    assert float(logits.grad[...,0].abs().sum())>0


def test_stable_joint_id_tie_break_and_fail_closed_on_invalid_support():
    dense=torch.full((1,2,55),1/55)
    one=sparse_skin_support_v1(dense,max_influences=8)
    two=sparse_skin_support_v1(dense,max_influences=8)
    torch.testing.assert_close(one,two,atol=0,rtol=0)
    assert (one[..., :8]>0).all()
    assert (one[..., 8:]==0).all()
    with pytest.raises(ValueError,match="SUPPORT_DOMAIN_INVALID"):
        sparse_skin_support_v1(dense,max_influences=42)
    with pytest.raises(ValueError,match="SUPPORT_DOMAIN_INVALID"):
        sparse_skin_support_v1(torch.zeros_like(dense),max_influences=8)
    with pytest.raises(ValueError,match="SUPPORT_DOMAIN_INVALID"):
        bad=dense.clone();bad[0,0,0]=float("nan")
        sparse_skin_support_v1(bad,max_influences=8)


@pytest.mark.parametrize("mode",["DOT_PRODUCT_BASELINE","POINT_QUERY_CROSS_ATTENTION"])
def test_55_joint_embedded_fsq_decode_is_sparse_and_still_trainable(mode):
    from dataclasses import replace
    cfg=replace(RSSvNextConfig.tiny_for_test(),
                skin_decoder_mode=mode,skin_max_influences=8)
    model=RSSvNextSkinCoupler(cfg)
    local=torch.randn(1,3,cfg.iris.local_feature_dim)
    xyz=torch.randn(1,3,3)
    normals=torch.nn.functional.normalize(torch.randn(1,3,3),dim=-1)
    joints=torch.randn(1,55,cfg.control_hidden_dim)
    ids=torch.randint(0,model.codec.quantizer.vocab_size,
                      (1,55,cfg.skin_tokens_per_joint))
    model.train()
    out=model.decode_discrete_skin_v1(
        local,xyz,joints,ids,vertex_normals=normals,
    )
    assert out.shape==(1,3,55)
    assert (out.detach()>0).sum(-1).max().item()==8
    assert torch.allclose(out.detach().sum(-1),torch.ones(1,3),atol=1.e-6)
    (out[...,0]-1.).square().sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum().item()>0
               for p in model.parameters() if p.requires_grad)
    model.eval()
    with torch.no_grad():
        replay=model.decode_discrete_skin_v1(
            local,xyz,joints,ids,vertex_normals=normals,
        )
    torch.testing.assert_close(replay,out.detach(),atol=0,rtol=0)


def test_sparse_decoded_proposal_serializes_only_real_support():
    j=55
    dense=torch.full((3,j),1/j)
    sparse=sparse_skin_support_v1(dense,max_influences=8)
    names=tuple(f"J:{i}" for i in range(j))
    carrier=MIRACarrierViewV2(
        ordered_vertex_ids=("V:0","V:1","V:2"),
        carrier_evidence_hash="a"*64,topology_hash="b"*64,
        geometry_hash="c"*64,
    )
    skeleton=MIRADeformSkeletonViewV2(
        joints=tuple(_SkeletonJointViewV2(n) for n in names),
        skeleton_lineage_hash="d"*64,
    )
    conditioning=MIRAConditioningV6(
        vertex_features=torch.zeros(1,3,7),
        joint_features=torch.zeros(1,j,25),
        carrier=carrier,skeleton=skeleton,joint_ids=names,
    )
    proposal=carrier_skin_proposal_from_decoded_v6(
        sparse,conditioning,model_provenance="test:model:fsq",
        max_influences=8,
    )
    assert len(proposal.influences)==3*8
    assert proposal.metadata["zero_weights_not_serialized"] is True
    qualified=qualify_carrier_skin_v1(
        carrier,skeleton,proposal,
        max_simplex_repair_l1=1.e-5,max_total_correction_l1=1.e-4,
        negative_tolerance=1.e-8,max_influences=41,
    )
    assert len(qualified.rows)==3
    assert all(len(row.influences)==8 for row in qualified.rows)
    with pytest.raises(RuntimeError,match="SPARSE_SIMPLEX_INVALID"):
        carrier_skin_proposal_from_decoded_v6(
            dense,conditioning,model_provenance="test",max_influences=8,
        )
