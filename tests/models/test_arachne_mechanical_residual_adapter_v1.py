from __future__ import annotations

import math
import torch

from models.arachne.v4.mechanical_residual_adapter_v1 import (
    MechanicalResidualAdapterV1,
)
from models.arachne.v4.mechanical_consequence_loss_v1 import (
    mechanical_consequence_loss_v1,
)


def test_mechanical_residual_adapter_is_identity_at_initialization_and_simplex():
    torch.manual_seed(1)
    model=MechanicalResidualAdapterV1()
    B,N,J=1,5,3
    base=torch.rand(B,N,J)
    base=base/base.sum(-1,keepdim=True)
    geom=torch.randn(B,N,7)
    pair=torch.randn(B,N,J,10)
    sm=torch.ones(B,N,dtype=torch.bool)
    jm=torch.ones(B,J,dtype=torch.bool)
    out,delta=model(
        base_weights=base,
        surface_geometry7=geom,
        pair_geometry=pair,
        surface_mask=sm,
        joint_mask=jm,
    )
    assert model.parameter_count==1_582_849
    assert torch.equal(delta,torch.zeros_like(delta))
    assert torch.allclose(out,base,atol=2e-7,rtol=0.0)
    assert torch.allclose(out.sum(-1),torch.ones(B,N),atol=1e-7,rtol=0.0)
    assert bool((out>=0).all())


def test_mechanical_residual_adapter_receives_gradient_from_product_space_mechanics():
    torch.manual_seed(2)
    model=MechanicalResidualAdapterV1()
    B,N,J=1,3,2
    base=torch.tensor([[[1.0,0.0],[0.0,1.0],[1.0,0.0]]],dtype=torch.float32)
    geom=torch.randn(B,N,7)
    pair=torch.randn(B,N,J,10)
    sm=torch.ones(B,N,dtype=torch.bool)
    jm=torch.ones(B,J,dtype=torch.bool)
    out,_=model(
        base_weights=base,
        surface_geometry7=geom,
        pair_geometry=pair,
        surface_mask=sm,
        joint_mask=jm,
    )
    support_idx=torch.arange(N).reshape(1,N,1)
    support_coeff=torch.ones(1,N,1)
    rest=torch.tensor([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]])
    faces=torch.tensor([[0,1,2]],dtype=torch.long)
    transforms=torch.eye(4).reshape(1,1,1,4,4).repeat(1,1,J,1,1)
    theta=math.pi/2
    transforms[0,0,1,0,0]=math.cos(theta)
    transforms[0,0,1,0,1]=-math.sin(theta)
    transforms[0,0,1,1,0]=math.sin(theta)
    transforms[0,0,1,1,1]=math.cos(theta)
    loss=mechanical_consequence_loss_v1(
        out,
        candidate_support_indices=support_idx,
        candidate_support_coefficients=support_coeff,
        candidate_rest_vertices=rest,
        candidate_faces=faces,
        probe_transforms=transforms,
        base_surface_weights=base,
        max_edge_ratio=1.05,
        min_area_ratio=0.8,
        max_area_ratio=1.25,
        max_condition_number=1.25,
        teacher_weight=0.0,
        trust_weight=0.01,
    )
    assert torch.isfinite(loss["total"])
    assert float(loss["mechanical"])>0.0
    loss["total"].backward()
    final=model.residual[-1]
    assert final.weight.grad is not None
    assert float(final.weight.grad.abs().sum())>0.0


def test_product_space_loss_is_zero_for_rigid_single_joint_identity():
    weights=torch.ones(1,3,1)
    support_idx=torch.arange(3).reshape(1,3,1)
    support_coeff=torch.ones(1,3,1)
    rest=torch.tensor([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]])
    faces=torch.tensor([[0,1,2]],dtype=torch.long)
    transforms=torch.eye(4).reshape(1,1,1,4,4)
    loss=mechanical_consequence_loss_v1(
        weights,
        candidate_support_indices=support_idx,
        candidate_support_coefficients=support_coeff,
        candidate_rest_vertices=rest,
        candidate_faces=faces,
        probe_transforms=transforms,
        max_edge_ratio=4.0,
        min_area_ratio=0.05,
        max_area_ratio=20.0,
        max_condition_number=16.0,
        teacher_weight=0.0,
        trust_weight=0.0,
    )
    assert abs(float(loss["mechanical"]))<1e-12


def test_mechanical_residual_adapter_cannot_mint_masked_joint_support():
    torch.manual_seed(3)
    model=MechanicalResidualAdapterV1()
    B,N,J=1,2,3
    base=torch.tensor([[
        [0.75,0.25,0.0],
        [0.0,1.0,0.0],
    ]],dtype=torch.float32)
    geom=torch.randn(B,N,7)
    pair=torch.randn(B,N,J,10)
    sm=torch.ones(B,N,dtype=torch.bool)
    jm=torch.ones(B,J,dtype=torch.bool)
    row_mask=torch.tensor([[
        [True,True,False],
        [False,True,False],
    ]])
    # Make a nonzero residual deliberately; masked supports must remain exact zero.
    with torch.no_grad():
        model.residual[-1].bias.fill_(2.0)
    out,_=model(
        base_weights=base,
        surface_geometry7=geom,
        pair_geometry=pair,
        surface_mask=sm,
        joint_mask=jm,
        row_joint_mask=row_mask,
    )
    assert torch.equal(out[...,2],torch.zeros_like(out[...,2]))
    assert float(out[0,1,0])==0.0
    assert float(out[0,1,1])==1.0
    assert torch.allclose(out.sum(-1),torch.ones(B,N),atol=1e-7,rtol=0.0)
