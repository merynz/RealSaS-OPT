from __future__ import annotations

"""CPU microcourts for V2 Stage A/B math. No claim of 300-source GPU fit."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from models.rss_vnext.paired_sdxl_lora_prior_v1 import (
    RSSvNextTwoImageConditionerV1,
)
from models.rss_vnext.realsas_sdxl_stage_a_source_training_v2 import (
    SDXLCanonicalAlphaHeadV2,SDXLSourceStageABatchV2,
    source_sdxl_canonical_stage_a_loss_v2,
)
from models.rss_vnext.realsas_mvadapter_camera_bridge_v2 import (
    mv_adapter_controls_from_13d_virtual_cameras_v2,
    fixed_sdxl_mvadapter_two_plus_six_cameras_v2,
)
from models.rss_vnext.realsas_mvadapter_dual_reference_v2 import (
    MVAdapterRGBNormalTaskHeadsV2,
)
from models.rss_vnext.realsas_mvadapter_source_training_v2 import (
    MVAdapterSourceStageBBatchV2,mv_adapter_stage_b_loss_v2,
)


class DummyPosterior:
    def __init__(self,sample): self.sample=sample
    def mode(self): return self.sample


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy=nn.Parameter(torch.zeros(1),requires_grad=False)
        self.config=SimpleNamespace(scaling_factor=.18)
    @property
    def dtype(self):
        return self.dummy.dtype
    def encode(self,colors):
        lat=F.avg_pool2d(colors,8)
        return SimpleNamespace(latent_dist=DummyPosterior(
            torch.cat([lat,lat[:,0:1]],dim=1)
        ))
    def decode(self,latent,return_dict=True):
        x=F.interpolate(latent[:,:3],scale_factor=8,mode="bilinear")
        return SimpleNamespace(sample=x)


class Scheduler:
    def __init__(self):
        self.config=SimpleNamespace(
            prediction_type="epsilon",num_train_timesteps=10,
        )
        self.alphas_cumprod=torch.linspace(.99,.1,10)
        self.init_noise_sigma=1.
        self.timesteps=[]
    def add_noise(self,clean,noise,timesteps):
        a=self.alphas_cumprod[timesteps].reshape(-1,1,1,1)
        return a.sqrt()*clean+(1.-a).sqrt()*noise
    def set_timesteps(self,n,device):
        self.timesteps=list(range(n-1,-1,-1))
    def scale_model_input(self,x,t): return x
    def step(self,noise,t,latents,return_dict=True):
        return SimpleNamespace(prev_sample=latents-.1*noise)


class FakeSDXLUnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_q=nn.Parameter(torch.tensor(.1))
        self.frozen_base=nn.Parameter(torch.tensor(.2),requires_grad=False)
    def forward(self,noisy,timestep,encoder_hidden_states,
                added_cond_kwargs,return_dict=True):
        influence=encoder_hidden_states.mean(
            dim=(1,2),keepdim=True,
        ).reshape(-1,1,1,1)
        return SimpleNamespace(sample=noisy*self.lora_q+influence)


class FakeMVUnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_mv=nn.Parameter(torch.tensor(.1))
        self.attn_ref=nn.Parameter(torch.tensor(.2))
        self.base=nn.Parameter(torch.tensor(.3),requires_grad=False)
    def forward(self,latent,timestep,encoder_hidden_states,
                added_cond_kwargs,cross_attention_kwargs,
                return_dict=True,down_intrablock_additional_residuals=None):
        if "cache_hidden_states" in cross_attention_kwargs:
            cross_attention_kwargs["cache_hidden_states"]["layer"] = (
                latent.mean(dim=(-1,-2))[:,None,:]
            )
            return (latent,)
        ref=cross_attention_kwargs["ref_hidden_states"]["layer"]
        refval=ref.mean(dim=(1,2)).reshape(-1,1,1,1)
        textval=encoder_hidden_states[:,-1,:].mean(dim=-1).view(-1,1,1,1)
        down=down_intrablock_additional_residuals[0].mean(
            dim=(1,2,3),keepdim=True,
        )
        return SimpleNamespace(
            sample=latent*self.attn_mv+refval*self.attn_ref+textval+down,
        )


class FakeControl(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Conv2d(6,4,1)
    def forward(self,x):
        return [self.conv(x)]


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(72)
    torch.set_num_threads(1)


def test_camera_translation_matches_native_mv_adapter_ortho_semantics():
    canonical,six,control=fixed_sdxl_mvadapter_two_plus_six_cameras_v2(
        1,device="cpu",dtype=torch.float32,image_size=64,
    )
    assert canonical.shape==(1,2,13)
    assert six.shape==(1,6,13)
    assert control.shape==(1,6,6,64,64)
    actual=control[0,:, :,0,0]*2.-1.
    assert torch.allclose(actual[:,:3],six[0,:,9:12],atol=1e-6)
    assert torch.allclose(
        actual[:,3:],F.normalize(six[0,:,:3],dim=-1),atol=1e-6,
    )
    assert torch.allclose(control[...,0,0],control[...,-1,-1])
    with pytest.raises(ValueError,match="CAMERA_BANK_NOT_V1"):
        mv_adapter_controls_from_13d_virtual_cameras_v2(
            six+1.,height=64,width=64,
        )


def test_paired_stage_a_supervision_reaches_lora_conditioner_and_alpha():
    unet=FakeSDXLUnet()
    conditioner=RSSvNextTwoImageConditionerV1(
        source_dim=8,pose_dim=4,hidden_dim=16,output_dim=2048,
        queries=2,heads=4,
    )
    head=SDXLCanonicalAlphaHeadV2()
    source=torch.rand(1,2,4,8)
    valid=torch.ones(1,2,4,dtype=torch.bool)
    canon_cam,six,control=fixed_sdxl_mvadapter_two_plus_six_cameras_v2(
        1,device="cpu",dtype=torch.float32,image_size=64,
    )
    batch=SDXLSourceStageABatchV2(
        source_tokens=source,source_token_valid=valid,
        target_canonical_rgba=torch.rand(1,2,4,64,64),
        target_valid=torch.ones(1,2,1,64,64),
        target_cameras=canon_cam,
        canonical_pose_queries=torch.zeros(1,2,4),
        prompt_embeds=torch.zeros(1,2,2048),
        pooled_text_embeds=torch.zeros(1,1280),
        empty_prompt_embeds=torch.zeros(1,2,2048),
        empty_pooled_text_embeds=torch.zeros(1,1280),
        time_ids=torch.zeros(1,6),
        source_render_qualification_key="fixture-not-real-source",
    )
    loss=source_sdxl_canonical_stage_a_loss_v2(
        unet=unet,vae=VAE(),noise_scheduler=Scheduler(),
        conditioner=conditioner,alpha_head=head,batch=batch,
    )
    assert torch.isfinite(loss.total_loss)
    loss.total_loss.backward()
    assert unet.lora_q.grad is not None
    assert conditioner.output[1].weight.grad is not None
    assert head.project[0].weight.grad is not None
    assert loss.qualified_pixels==2*64*64


def test_native_dual_reference_stage_b_source_loss_and_rgb_normal_alpha():
    pipe=SimpleNamespace(
        unet=FakeMVUnet(),cond_encoder=FakeControl(),
        vae=VAE(),
    )
    heads=MVAdapterRGBNormalTaskHeadsV2()
    canon_cam,six,control=fixed_sdxl_mvadapter_two_plus_six_cameras_v2(
        1,device="cpu",dtype=torch.float32,image_size=64,
    )
    batch=MVAdapterSourceStageBBatchV2(
        canonical_rgba=torch.rand(1,2,4,64,64),
        rgb=torch.rand(1,6,3,64,64),
        normal_encoded_rgb=torch.rand(1,6,3,64,64),
        alpha=torch.rand(1,6,1,64,64),
        valid=torch.ones(1,6,1,64,64),
        plucker_controls=control,target_cameras=six,
        prompt_embeds=torch.zeros(1,2,2048),
        pooled_text_embeds=torch.zeros(1,1280),
        empty_prompt_embeds=torch.zeros(1,2,2048),
        empty_pooled_text_embeds=torch.zeros(1,1280),
        time_ids=torch.zeros(1,6),
        source_render_qualification_key="fixture-not-source",
    )
    loss=mv_adapter_stage_b_loss_v2(
        pipe=pipe,task_heads=heads,source=batch,
        noise_scheduler=Scheduler(),
    )
    assert torch.isfinite(loss.total_loss)
    assert loss.rgb_epsilon_loss>0
    assert loss.normal_epsilon_loss>0
    assert loss.alpha_loss>0
    loss.total_loss.backward()
    assert pipe.unet.attn_mv.grad is not None
    assert pipe.unet.attn_ref.grad is not None
    assert pipe.cond_encoder.conv.weight.grad is not None
    assert heads.modality_token.weight.grad is not None
    assert heads.alpha_head[0].weight.grad is not None


def test_stage_b_camera_target_cannot_use_another_virtual_bank():
    _,six,control=fixed_sdxl_mvadapter_two_plus_six_cameras_v2(
        1,device="cpu",dtype=torch.float32,image_size=64,
    )
    batch=MVAdapterSourceStageBBatchV2(
        canonical_rgba=torch.rand(1,2,4,64,64),
        rgb=torch.rand(1,6,3,64,64),
        normal_encoded_rgb=torch.rand(1,6,3,64,64),
        alpha=torch.rand(1,6,1,64,64),
        valid=torch.ones(1,6,1,64,64),
        plucker_controls=control,target_cameras=six.flip(1),
        prompt_embeds=torch.zeros(1,2,2048),
        pooled_text_embeds=torch.zeros(1,1280),
        empty_prompt_embeds=torch.zeros(1,2,2048),
        empty_pooled_text_embeds=torch.zeros(1,1280),
        time_ids=torch.zeros(1,6),
        source_render_qualification_key="fixture-not-source",
    )
    with pytest.raises(ValueError,match="CAMERA_BANK_NOT_V1"):
        batch.validate()
