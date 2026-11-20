from __future__ import annotations

"""Shape/gradient/authority checks, not evidence of trained diffusion quality."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models.rss_vnext.realsas_mvadapter_dual_reference_v2 import (
    capture_native_two_reference_cache_v2,
    MVAdapterRGBNormalTaskHeadsV2,
    native_mv_adapter_rgb_normal_denoise_v2,
    official_mv_adapter_trainable_parameters_v2,
)


class NativeMVFakeUnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base=nn.Parameter(torch.tensor(2.))
        self.attn_mv=nn.Parameter(torch.tensor(.2))
        self.attn_ref=nn.Parameter(torch.tensor(.3))

    def forward(self, latent, steps, *, encoder_hidden_states,
                added_cond_kwargs, cross_attention_kwargs, return_dict,
                down_intrablock_additional_residuals=None):
        if "cache_hidden_states" in cross_attention_kwargs:
            # Written by the official pipeline's attention processor; this
            # fixture only approximates its write interface, never its quality.
            out = cross_attention_kwargs["cache_hidden_states"]
            feature = latent.mean(dim=(-1,-2)).unsqueeze(1)
            out["layer0"] = feature
            out["layer1"] = 2*feature
            assert not cross_attention_kwargs["use_mv"]
            assert not cross_attention_kwargs["use_ref"]
            return (latent,)
        refs=cross_attention_kwargs["ref_hidden_states"]
        assert cross_attention_kwargs["num_views"]==6
        assert refs["layer0"].shape[0]==latent.shape[0]
        source=refs["layer0"].mean(dim=(1,2)).view(-1,1,1,1)
        control=down_intrablock_additional_residuals[0].mean(
            dim=(1,2,3),keepdim=True,
        )
        cond=encoder_hidden_states[:,-1,:].mean(dim=-1).view(-1,1,1,1)
        result=latent*1.+self.attn_mv*control+self.attn_ref*source+cond
        return SimpleNamespace(sample=result)


class CondEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Conv2d(6,4,kernel_size=1)

    def forward(self,controls):
        return [self.conv(controls)]


def prompt(b=1):
    return (torch.zeros(b,2,2048),
            torch.zeros(b,1280),
            torch.zeros(b,6))


def native_pipeline():
    unet=NativeMVFakeUnet()
    pipe=SimpleNamespace(unet=unet,cond_encoder=CondEncoder(),
                         vae=nn.Conv2d(3,3,1),
                         text_encoder=nn.Linear(2,2),
                         text_encoder_2=nn.Linear(2,2))
    return pipe


def test_reference_features_preserve_both_canonical_images_no_average():
    pipe=native_pipeline()
    refs=torch.zeros((1,2,4,4,4))
    refs[:,0]=1.
    refs[:,1]=3.
    text,pooled,timeids=prompt()
    cache=capture_native_two_reference_cache_v2(
        unet=pipe.unet,canonical_latents=refs,
        prompt_embeds=text,pooled_text_embeds=pooled,
        time_ids=timeids,modalities=2,
    )
    assert cache.modalities==2 and cache.num_views==6
    assert cache.features["layer0"].shape==(12,2,4)
    assert torch.all(cache.features["layer0"][:,0,:]==1)
    assert torch.all(cache.features["layer0"][:,1,:]==3)
    assert not cache.features["layer0"].requires_grad
    cache.validate()


def test_real_stage_b_task_groups_are_six_rgb_and_six_normal():
    pipe=native_pipeline()
    refs=torch.stack([torch.ones(4,4,4),torch.ones(4,4,4)*2])[None]
    text,pooled,ts=prompt()
    cache=capture_native_two_reference_cache_v2(
        unet=pipe.unet,canonical_latents=refs,
        prompt_embeds=text,pooled_text_embeds=pooled,time_ids=ts,
    )
    heads=MVAdapterRGBNormalTaskHeadsV2()
    latent=torch.zeros((1,2,6,4,4,4),requires_grad=True)
    controls=torch.ones((1,2,6,6,4,4))
    noise=native_mv_adapter_rgb_normal_denoise_v2(
        pipe=pipe,task_heads=heads,noisy_latents=latent,
        timesteps=torch.zeros((1,2,6),dtype=torch.long),
        cache=cache,prompt_embeds=text,pooled_text_embeds=pooled,
        time_ids=ts,control_plucker=controls,
    )
    assert noise.shape==(1,2,6,4,4,4)
    loss=noise.square().sum()
    loss.backward()
    assert pipe.unet.attn_mv.grad is not None
    assert pipe.unet.attn_ref.grad is not None
    assert heads.modality_token.weight.grad is not None
    assert latent.grad is not None


def test_trainability_matches_upstream_not_full_unet_finetune():
    pipe=native_pipeline()
    active=official_mv_adapter_trainable_parameters_v2(pipe)
    names=[x for x,_ in active]
    assert any("_mv" in x for x in names)
    assert any("_ref" in x for x in names)
    assert any("cond_encoder" in x for x in names)
    assert not pipe.unet.base.requires_grad
    assert pipe.unet.attn_mv.requires_grad
    assert pipe.unet.attn_ref.requires_grad
    assert not any(x.requires_grad for x in pipe.vae.parameters())
    assert not any(x.requires_grad for x in pipe.text_encoder.parameters())


def test_unequal_modality_or_ref_cache_must_fail():
    pipe=native_pipeline()
    text,pooled,timeids=prompt()
    with pytest.raises(ValueError,match="SOURCE_ABI_INVALID"):
        capture_native_two_reference_cache_v2(
            unet=pipe.unet,
            canonical_latents=torch.rand(1,1,4,4,4),
            prompt_embeds=text,pooled_text_embeds=pooled,time_ids=timeids,
        )
    cache=capture_native_two_reference_cache_v2(
        unet=pipe.unet,canonical_latents=torch.ones(1,2,4,4,4),
        prompt_embeds=text,pooled_text_embeds=pooled,time_ids=timeids,
        modalities=1,
    )
    with pytest.raises(ValueError,match="PAIRED_RGB_NORMAL_BATCH_ABI"):
        native_mv_adapter_rgb_normal_denoise_v2(
            pipe=pipe,task_heads=MVAdapterRGBNormalTaskHeadsV2(),
            noisy_latents=torch.zeros((1,2,6,4,4,4)),
            timesteps=torch.zeros((1,2,6),dtype=torch.long),
            cache=cache,prompt_embeds=text,pooled_text_embeds=pooled,
            time_ids=timeids,control_plucker=torch.zeros((1,2,6,6,4,4)),
        )


def test_alpha_head_is_trainable_and_is_not_a_synthetic_normal_teacher():
    heads=MVAdapterRGBNormalTaskHeadsV2()
    noisy=torch.randn(6,4,8,8,requires_grad=True)
    alpha_logits=heads.alpha_logits(noisy,(64,64))
    assert alpha_logits.shape==(6,1,64,64)
    alpha_logits.mean().backward()
    assert heads.alpha_head[0].weight.grad is not None
    assert noisy.grad is not None
