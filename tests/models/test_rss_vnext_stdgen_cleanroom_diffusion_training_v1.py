from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from models.rss_vnext.stdgen_cleanroom_diffusion_training_v1 import (
    STAGE_CANONICAL, STAGE_MULTIVIEW,
    CleanroomDiffusionRenderRecordV1, validate_render_record_v1,
    validate_cleanroom_render_corpus_v1, AttentionLoRALinearV1,
    install_attention_lora_v1, PairedSourceDiffusionConditionerV1,
    cleanroom_diffusion_noise_microstep_v1,
)


def _render(asset="a", split="train", group="c", poses=True, count=6):
    def h(x): return (x*64)[:64]
    return CleanroomDiffusionRenderRecordV1(
        source_asset_lineage_sha256=h(asset),
        dedup_family_sha256=h(group),
        rights_admission_sha256=h("r"),
        morphology_id="insect_six_legged",
        neutral_pose_definition_sha256=h("p"),
        split=split,
        input_pose_ids=("pose-0","pose-1") if poses else ("static","static"),
        input_scene_rgb_sha256=(h("1"),h("2")),
        canonical_target_rgb_sha256=(h("3"),h("4")),
        multiview_target_rgb_sha256=tuple(h(str(i+10)) for i in range(count)),
        multiview_target_normal_sha256=tuple(h(str(i+20)) for i in range(count)),
        multiview_target_alpha_sha256=tuple(h(str(i+30)) for i in range(count)),
        multiview_target_depth_sha256=tuple(h(str(i+40)) for i in range(count)),
        renderer_version_sha256=h("z"),
        source_pose_truth_available=poses,
    )


def test_render_truth_requires_admitted_asset_and_multimodal_3d_gt():
    a=_render()
    validate_render_record_v1(a)
    assert validate_cleanroom_render_corpus_v1((a,))["counts_by_split"]["train"]==1
    assert validate_cleanroom_render_corpus_v1((a,))[
        "canonical_pose_training_eligible_by_split"]["train"]==1
    with pytest.raises(ValueError,match="TARGET_MISSING"):
        validate_render_record_v1(replace(a,multiview_target_normal_sha256=()))
    with pytest.raises(ValueError,match="SOURCE_OR_RIGHTS_INVALID"):
        validate_render_record_v1(replace(
            a,source_asset_training_rights_independently_admitted=False,
        ))
    with pytest.raises(ValueError,match="UNRIGGED_POSE_TARGET"):
        validate_render_record_v1(replace(a,source_pose_truth_available=False))
    assert validate_cleanroom_render_corpus_v1((_render(poses=False),))[
        "canonical_pose_training_eligible_by_split"]["train"]==0


def test_train_val_unseen_dedup_family_is_held_out_not_random_render_frame():
    train=_render(asset="a",group="c")
    unseen=_render(asset="b",group="c",split="unseen")
    with pytest.raises(ValueError,match="TRAIN_UNSEEN_LINEAGE_LEAK"):
        validate_cleanroom_render_corpus_v1((train,unseen))
    with pytest.raises(ValueError,match="DUPLICATE_TRAINING_EXAMPLE"):
        validate_cleanroom_render_corpus_v1((train,train))
    safe=_render(asset="d",group="f",split="unseen")
    result=validate_cleanroom_render_corpus_v1((train,safe))
    assert result["counts_by_split"]["train"]==1
    assert result["counts_by_split"]["unseen"]==1
    assert not result["product_quality_or_unseen_proof"]


class _MiniAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q=nn.Linear(8,8)
        self.to_k=nn.Linear(8,8)
        self.to_out=nn.Sequential(nn.Linear(8,8),nn.Identity())
        self.unrelated=nn.Linear(8,8)
    def forward(self,x):
        return self.to_out(self.to_q(x)+self.to_k(x))+self.unrelated(x)


def test_lora_initially_matches_frozen_pretrained_attention_then_learns():
    torch.manual_seed(17)
    net=_MiniAttention()
    input=torch.randn(3,8)
    old=net(input).detach().clone()
    patched=install_attention_lora_v1(net,rank=2,alpha=2.)
    assert patched==("to_q","to_k","to_out.0")
    assert isinstance(net.to_q,AttentionLoRALinearV1)
    assert torch.allclose(net(input),old,atol=1e-7)
    assert not any(p.requires_grad for p in net.unrelated.parameters())
    assert not any(p.requires_grad for p in net.to_q.base.parameters())
    opt=torch.optim.SGD([p for p in net.parameters() if p.requires_grad],lr=.4)
    loss=(net(input)**2).mean()
    loss.backward()
    assert net.to_q.lora_B.grad.abs().sum()>0
    opt.step()
    assert not torch.allclose(net(input),old)
    assert not net.to_q.base.weight.requires_grad


class _FakeVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.config=SimpleNamespace(scaling_factor=1.)
    def encode(self,x):
        return SimpleNamespace(latent_dist=SimpleNamespace(
            mode=lambda : F.avg_pool2d(x,2),
        ))


class _FakeScheduler:
    config=SimpleNamespace(num_train_timesteps=100,prediction_type="epsilon")
    def add_noise(self,x,noise,t):
        return x+noise*.1


class _FakeDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q=nn.Linear(8,8)
        self.to_out=nn.Sequential(nn.Linear(8,3),nn.Identity())
    def forward(self,latents,t,encoder_hidden_states):
        x=self.to_out(self.to_q(encoder_hidden_states.mean(dim=1)))
        pred=x[:,:,None,None].expand_as(latents)
        return SimpleNamespace(sample=pred)


def _microcase(stage):
    torch.manual_seed(4)
    count=2 if stage==STAGE_CANONICAL else 12
    src=torch.randn(1,2,4,8)
    targets=torch.rand((1,count,3,16,16))
    mask=torch.ones((1,count,1,16,16))
    cameras=torch.zeros((1,count,13))
    cameras[:,:,3]=1.
    cameras[:,:,7]=1.
    cameras[:,:,11]=1.
    cameras[:,:,12]=1.
    mods=torch.zeros((1,count),dtype=torch.long)
    if stage==STAGE_MULTIVIEW:
        mods[0,count//2:]=1
    condition=PairedSourceDiffusionConditionerV1(8,8,heads=2)
    denoiser=_FakeDenoiser()
    install_attention_lora_v1(denoiser,rank=2,alpha=2.)
    return dict(source_tokens=src,rendered_target_rgb=targets,
        rendered_target_validity_mask=mask,target_cameras=cameras,
        target_modalities=mods,vae=_FakeVAE(),scheduler=_FakeScheduler(),
        denoiser=denoiser,conditioner=condition,stage=stage)


@pytest.mark.parametrize("stage",[STAGE_CANONICAL,STAGE_MULTIVIEW])
def test_one_step_cleanroom_2source_conditioned_noise_training_is_differentiable(stage):
    kw=_microcase(stage)
    condition=kw["conditioner"]
    with torch.no_grad():
        hidden=condition(kw["source_tokens"],kw["target_cameras"],kw["target_modalities"])
        alternate=kw["source_tokens"].clone()
        alternate[:,1]+=3.
        altered=condition(alternate,kw["target_cameras"],kw["target_modalities"])
    assert hidden.shape[0]==kw["target_cameras"].shape[1]
    assert not torch.allclose(hidden,altered)
    loss=cleanroom_diffusion_noise_microstep_v1(**kw)
    assert torch.isfinite(loss) and loss.item()>0
    loss.backward()
    assert kw["denoiser"].to_q.lora_B.grad.abs().sum()>0
    assert condition.source_projection.weight.grad.abs().sum()>0
    assert not kw["denoiser"].to_q.base.weight.requires_grad


def test_training_step_does_not_accept_missing_teacher_pixels_or_wrong_stage():
    kw=_microcase(STAGE_CANONICAL)
    kw["rendered_target_validity_mask"].zero_()
    with pytest.raises(ValueError,match="RENDER_SUPERVISION_DOMAIN"):
        cleanroom_diffusion_noise_microstep_v1(**kw)
    kw=_microcase(STAGE_CANONICAL)
    kw["target_modalities"][:]=1
    with pytest.raises(ValueError,match="RENDER_SUPERVISION_DOMAIN"):
        cleanroom_diffusion_noise_microstep_v1(**kw)
    kw=_microcase(STAGE_MULTIVIEW)
    kw["scheduler"].config.prediction_type="v_prediction"
    with pytest.raises(ValueError,match="EPSILON_SCHEDULER_REQUIRED"):
        cleanroom_diffusion_noise_microstep_v1(**kw)
