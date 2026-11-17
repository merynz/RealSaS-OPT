from __future__ import annotations

"""Large mesh sampling is SOURCE-LOSS ONLY, not mesh simplification."""

from dataclasses import replace

import pytest
import torch

from models.rss_vnext.seeded_surface_sampling_v1 import (
    SeededSurfaceSamplingPolicyV1,
    component_spatial_sample_indices_v1,
)


def test_component_cape_not_erased_by_large_body_and_checkpoint_step_rotates():
    torch.manual_seed(88)
    body=torch.randn(5000,3)*0.5
    cape=torch.randn(24,3)*0.04+torch.tensor([0.,0.,0.7])
    xyz=torch.cat((body,cape))
    component=torch.cat((
        torch.zeros(5000,dtype=torch.long),
        torch.ones(24,dtype=torch.long),
    ))
    def sample(step):
        return component_spatial_sample_indices_v1(
            xyz,component,budget=256,step=step,
        )
    a=sample(0)
    b=sample(1)
    torch.testing.assert_close(a,sample(0))
    assert len(a)==256
    assert len(torch.unique(a))==len(a)
    assert int((component[a]==1).sum())>=3
    assert int((component[b]==1).sum())>=3
    assert not torch.equal(a,b)
    assert int((a>=5000).sum())>=3
    assert not torch.allclose(xyz[a],xyz[b])


def test_all_source_vertices_kept_when_below_budget():
    xyz=torch.randn(10,3,requires_grad=True)
    comp=torch.tensor([0,0,0,0,0,1,1,1,1,1])
    ids=component_spatial_sample_indices_v1(xyz,comp,budget=256,step=3)
    torch.testing.assert_close(ids,torch.arange(10))
    xyz[ids].square().sum().backward()
    assert xyz.grad is not None and bool((xyz.grad!=0).any())


def test_many_small_components_fail_instead_of_disappearing():
    xyz=torch.arange(500,dtype=torch.float32)[:,None].expand(500,3).contiguous()
    comp=torch.arange(500,dtype=torch.long)
    with pytest.raises(ValueError,match="COMPONENT_COUNT_EXCEEDS_LOSS_BUDGET"):
        component_spatial_sample_indices_v1(
            xyz,comp,budget=256,step=0,minimum=3,
        )


def test_source_conforming_noncontiguous_components_and_policy_identity():
    policy=SeededSurfaceSamplingPolicyV1()
    policy.validate()
    assert policy.digest()!=replace(policy,max_student_vertices=4096).digest()
    xyz=torch.randn(900,3)
    comp=torch.arange(900,dtype=torch.long)%5
    ids=component_spatial_sample_indices_v1(
        xyz,comp,budget=300,step=20,minimum=3,
    )
    assert len(ids)==300
    assert set(comp[ids].tolist())==set(comp.tolist())
    assert len(torch.unique(ids))==len(ids)
    with pytest.raises(ValueError,match="SAMPLER_STEP|SAMPLE_INPUT_DOMAIN"):
        component_spatial_sample_indices_v1(
            xyz,comp,budget=300,step=-1,
        )
