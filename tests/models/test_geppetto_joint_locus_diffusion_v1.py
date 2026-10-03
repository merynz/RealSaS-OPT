from __future__ import annotations

import torch

from models.geppetto.v2.joint_locus_diffusion_v1 import (
    ConditionalDiffusionLocusConfigV1,
    ConditionalDiffusionLocusHeadV1,
)


def _head():
    torch.manual_seed(101)
    return ConditionalDiffusionLocusHeadV1(
        ConditionalDiffusionLocusConfigV1(
            condition_dim=16,
            hidden_dim=32,
            time_dim=16,
            train_steps=64,
            sample_steps=8,
            sample_count=3,
            position_scale=1.25,
        )
    )


def test_diffusion_locus_sampling_is_exact_seed_replay_and_bounded():
    head=_head().eval()
    cond=torch.randn(4,16)
    a=head.sample_hypotheses(cond,seed=77)
    b=head.sample_hypotheses(cond,seed=77)
    assert torch.equal(a,b)
    assert torch.isfinite(a).all()
    assert float(a.abs().max())<=1.25+1e-7


def test_diffusion_locus_is_condition_sensitive_at_fixed_seed():
    head=_head().eval()
    base=torch.linspace(-1.0,1.0,16)
    c0=torch.stack([base,base.flip(0)],dim=0)
    c1=torch.stack([base.roll(3),base.roll(-5)],dim=0)
    a=head.sample_hypotheses(c0,seed=91)
    b=head.sample_hypotheses(c1,seed=91)
    assert not torch.equal(a,b)
    assert float((a-b).abs().max())>1e-7


def test_diffusion_distribution_representative_is_real_medoid_sample():
    head=_head().eval()
    cond=torch.randn(3,16)
    dist=head.distribution(cond,seed=5)
    assert dist.modes_normalized.shape==(3,3,3)
    assert dist.representative_position.shape==(3,3)
    for bi in range(3):
        idx=int(dist.representative_mode_index[bi])
        assert torch.equal(
            dist.representative_position[bi],
            dist.modes_normalized[bi,idx],
        )
    assert torch.isfinite(dist.mode_logits).all()
    assert torch.equal(
        torch.argsort(
            dist.mode_logits,dim=-1,descending=True,stable=True
        )[:,0],
        dist.representative_mode_index,
    )
    assert torch.isfinite(dist.representative_log_sigma).all()


def test_diffusion_training_loss_is_finite_and_head_trainable():
    head=_head().train()
    cond=torch.randn(5,16)
    target=torch.randn(5,3).clamp(-1.0,1.0)
    timesteps=torch.tensor([0,5,12,31,63],dtype=torch.long)
    noise=torch.randn(5,3)
    out=head.training_loss(
        cond,target,timesteps=timesteps,noise=noise
    )
    assert torch.isfinite(out["total"])
    assert torch.isfinite(out["x0_reconstruction_mse"])
    out["total"].backward()
    grad=sum(
        float(p.grad.abs().sum())
        for p in head.parameters()
        if p.grad is not None
    )
    assert grad>0.0


def test_diffusion_loss_does_not_require_parent_or_identity_inputs():
    head=_head()
    cond=torch.randn(2,16)
    target=torch.randn(2,3).clamp(-1.0,1.0)
    out=head.training_loss(cond,target)
    assert "total" in out
    # The D1 locus head's public API is deliberately only latent condition + XYZ.
    assert head.config.condition_dim==16
