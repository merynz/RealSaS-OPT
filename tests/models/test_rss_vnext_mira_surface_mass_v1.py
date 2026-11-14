from __future__ import annotations

import pytest
import torch

from models.rss_vnext.skin_surface_mass_v1 import (
    POOLING_ABI, tessa_vertex_surface_mass_v1,
)
from models.mira.skin_fsq_codec_v6 import (
    MIRASkinFSQCodecConfig, MIRASkinFSQCodecV6,
)


def test_surface_mean_of_linear_field_is_stable_under_nonuniform_tessellation():
    # Both triangulations cover the SAME right triangle. The added interior
    # vertex near (0,0) would bias unweighted vertex pooling.
    coarse=torch.tensor([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]])
    coarse_faces=torch.tensor([[0,1,2]])
    fine=torch.tensor([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[.1,.1,0.]]])
    fine_faces=torch.tensor([[0,1,3],[1,2,3],[2,0,3]])
    a=tessa_vertex_surface_mass_v1(coarse,coarse_faces)
    b=tessa_vertex_surface_mass_v1(fine,fine_faces)
    assert POOLING_ABI.endswith(".v2")
    assert torch.allclose(a.sum(-1),torch.ones(1))
    assert torch.allclose(b.sum(-1),torch.ones(1))
    assert torch.allclose((a*coarse[...,0]).sum(),(b*fine[...,0]).sum(),atol=1e-6)
    assert abs(float((b*fine[...,0]).sum())-1./3.)<1e-6
    assert abs(float(fine[...,0].mean())-1./3.)>.04


def test_zero_surface_mass_vertex_cannot_change_causal_fsq_labels():
    # An unreferenced vertex has no drawable surface and should have no
    # authority over a per-joint FSQ label, even with extreme features.
    torch.manual_seed(212)
    codec=MIRASkinFSQCodecV6(
        MIRASkinFSQCodecConfig(
            7,25,hidden_dim=32,tokens_per_joint=4,levels=(3,5,3),
        )
    )
    coarse=torch.tensor([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]])
    face=torch.tensor([[0,1,2]])
    expanded=torch.cat((
        coarse,torch.tensor([[[100.,-100.,7.]]])
    ),dim=1)
    base_mass=tessa_vertex_surface_mass_v1(coarse,face)
    expanded_mass=tessa_vertex_surface_mass_v1(expanded,face)
    assert expanded_mass[0,-1].item()==0.
    features=torch.randn(1,3,7)
    extraneous=torch.full((1,1,7),500.)
    joints=torch.randn(1,2,25)
    weights=torch.tensor([[[1.,0.],[0.35,.65],[0.,1.]]])
    extra_weights=torch.tensor([[[0.,1.]]])
    base_ids,_,_=codec.encode_tokens(
        weights,features,joints,surface_vertex_mass=base_mass,
    )
    expanded_ids,_,_=codec.encode_tokens(
        torch.cat((weights,extra_weights),dim=1),
        torch.cat((features,extraneous),dim=1),
        joints,surface_vertex_mass=expanded_mass,
    )
    assert torch.equal(base_ids,expanded_ids)
    base_decode=codec.decode_tokens(base_ids,features,joints)
    expanded_decode=codec.decode_tokens(
        expanded_ids,torch.cat((features,extraneous),dim=1),joints,
    )
    # Different GEMM shapes may differ by a few FP32 ulps. Hard token
    # identity is exact; decoded weights must agree numerically.
    assert torch.allclose(base_decode,expanded_decode[:,:3],atol=1e-6,rtol=1e-6)
    # Real surface vertices must still influence the actual codec encoder.
    model_features=features.detach().clone().requires_grad_(True)
    fit=codec(weights,model_features,joints,surface_vertex_mass=base_mass)
    gradient=torch.autograd.grad(fit["quantization_loss"],model_features)[0]
    assert torch.isfinite(gradient).all()


def test_source_surface_weights_are_detached_and_invalid_mesh_is_rejected():
    rest=torch.tensor([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]],requires_grad=True)
    faces=torch.tensor([[0,1,2]])
    mass=tessa_vertex_surface_mass_v1(rest,faces)
    assert not mass.requires_grad
    with pytest.raises(ValueError,match="MESH_INVALID"):
        tessa_vertex_surface_mass_v1(rest,torch.tensor([[0,1,7]]))
    with pytest.raises(ValueError,match="NO_VALID_AREA"):
        tessa_vertex_surface_mass_v1(
            torch.zeros_like(rest),faces,
        )


def test_codec_and_causal_token_teacher_use_same_hard_surface_mass_codes():
    torch.manual_seed(211)
    codec=MIRASkinFSQCodecV6(
        MIRASkinFSQCodecConfig(
            7,25,hidden_dim=32,tokens_per_joint=4,levels=(3,5,3),
        )
    )
    rest=torch.tensor([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[.1,.1,0.]]])
    faces=torch.tensor([[0,1,3],[1,2,3],[2,0,3]])
    mass=tessa_vertex_surface_mass_v1(rest,faces)
    features=torch.randn(1,4,7)
    joints=torch.randn(1,2,25)
    source_skin=torch.softmax(torch.randn(1,4,2),dim=-1)
    result=codec(source_skin,features,joints,surface_vertex_mass=mass)
    ids,_,_=codec.encode_tokens(
        source_skin,features,joints,surface_vertex_mass=mass,
    )
    assert torch.equal(result["token_ids"],ids)
    assert torch.equal(
        result["reconstructed_weights"],
        codec.decode_tokens(ids,features,joints),
    )
    result["loss"].backward()
    assert codec.latent_head.weight.grad is not None
    assert float(codec.latent_head.weight.grad.abs().sum())>0
    with pytest.raises(ValueError,match="SURFACE_MASS_INVALID"):
        codec.encode_tokens(
            source_skin,features,joints,
            surface_vertex_mass=torch.ones((1,3)),
        )
    with pytest.raises(ValueError,match="SURFACE_MASS_INVALID"):
        codec.encode_tokens(
            source_skin,features,joints,
            surface_vertex_mass=-torch.ones((1,4)),
        )
