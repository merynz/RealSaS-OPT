from __future__ import annotations

from types import SimpleNamespace
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from models.rss_vnext.stdgen_era3d_unclip_microstep_v1 import (
    expand_pretrained_unclip_input_conv_v1,
    era3d_aligned_unclip_noise_microstep_v1,
)


def test_4to8_pretrained_conv_preserves_original_output_with_zero_condition():
    torch.manual_seed(9)
    original=nn.Conv2d(4,8,3,padding=1)
    upgraded=expand_pretrained_unclip_input_conv_v1(original)
    x=torch.randn((2,4,8,8))
    zeros=torch.zeros_like(x)
    assert upgraded.in_channels==8
    assert torch.equal(upgraded.weight[:,:4],original.weight)
    assert torch.count_nonzero(upgraded.weight[:,4:])==0
    assert torch.allclose(upgraded(torch.cat((x,zeros),dim=1)),original(x),atol=1e-6)
    with pytest.raises(ValueError,match="EXPECT_PRETRAINED_FOUR_CHANNEL_CONV"):
        expand_pretrained_unclip_input_conv_v1(upgraded)


class _VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy=nn.Parameter(torch.ones(1),requires_grad=False)
        self.config=SimpleNamespace(scaling_factor=1.)
    def encode(self,x):
        # output four latent channels at 1/2 spatial resolution
        latent=F.avg_pool2d(x,2)
        latent=torch.cat((latent,latent[:,:1]),dim=1)
        return SimpleNamespace(latent_dist=SimpleNamespace(
            mode=lambda:latent,
            sample=lambda:latent,
        ))


class _Clip(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy=nn.Parameter(torch.ones(1),requires_grad=False)
    def forward(self,x):
        return SimpleNamespace(image_embeds=x.mean(dim=(-1,-2)).repeat(1,2))


class _Norm:
    def scale(self,x): return x
    def unscale(self,x): return x


class _Scheduler:
    config=SimpleNamespace(prediction_type="epsilon",num_train_timesteps=1000)
    alphas_cumprod=torch.linspace(.999,.02,1000)
    def add_noise(self,x,noise,t): return x+.2*noise


class _UNet(nn.Module):
    rowwise_attention_in_unet_features=True
    config=SimpleNamespace(in_channels=8)
    def __init__(self):
        super().__init__()
        # Each synthetic model owns its config; one negative test must not
        # mutate the class-wide configuration of subsequent tests.
        self.config=SimpleNamespace(in_channels=8)
        self.rowwise_attention_in_unet_features=True
        self.conv=nn.Conv2d(8,4,1)
        self.rowwise_view_features=nn.Conv2d(4,4,1)
    def forward(self,x,t,encoder_hidden_states,class_labels):
        # Explicitly consume all UnCLIP condition carriers.
        pred=self.conv(x)
        pred=self.rowwise_view_features(pred)
        pred=pred+encoder_hidden_states.mean(dim=(1,2))[:,None,None,None]*.001
        pred=pred+class_labels.mean(dim=1)[:,None,None,None]*.001
        return SimpleNamespace(sample=pred)


def _case():
    torch.manual_seed(12)
    b=1
    return dict(
        reference_rgb=torch.rand((b,3,16,16)),
        target_rgb=torch.rand((b,6,3,16,16)),
        target_normal_encoded_rgb=torch.rand((b,6,3,16,16)),
        target_rgb_valid=torch.ones((b,6,1,16,16)),
        target_normal_valid=torch.ones((b,6,1,16,16)),
        reference_clip_preprocessed=torch.rand((b,3,4,4)),
        prompt_rgb=torch.randn((b,6,4,8)),
        prompt_normal=torch.randn((b,6,4,8)),
        vae=_VAE(),clip_image_encoder=_Clip(),
        image_normalizer=_Norm(),image_noise_scheduler=_Scheduler(),
        diffusion_scheduler=_Scheduler(),unet=_UNet(),
        object_timesteps=torch.tensor([10,20],dtype=torch.long),
    )


def test_real_era3d_carriers_six_view_rgb_normal_unclip_are_differentiable():
    kwargs=_case()
    model=kwargs["unet"]
    loss=era3d_aligned_unclip_noise_microstep_v1(**kwargs)
    assert torch.isfinite(loss),loss
    loss.backward()
    assert model.conv.weight.grad.abs().sum()>0
    assert model.rowwise_view_features.weight.grad.abs().sum()>0
    assert kwargs["vae"].dummy.grad is None
    assert kwargs["clip_image_encoder"].dummy.grad is None


def test_wrong_4channel_unet_or_external_latent_mixer_cannot_pass():
    kwargs=_case()
    kwargs["unet"].config.in_channels=4
    with pytest.raises(ValueError,match="EIGHT_CHANNEL_UNET_REQUIRED"):
        era3d_aligned_unclip_noise_microstep_v1(**kwargs)
    kwargs=_case()
    kwargs["unet"].rowwise_attention_in_unet_features=False
    with pytest.raises(ValueError,match="UNET_FEATURE_ROW_ATTENTION_REQUIRED"):
        era3d_aligned_unclip_noise_microstep_v1(**kwargs)


def test_no_original_source_pixels_are_coerced_to_training_gt():
    kwargs=_case()
    kwargs["target_normal_valid"].zero_()
    with pytest.raises(ValueError,match="NO_RENDERER_SUPERVISION"):
        era3d_aligned_unclip_noise_microstep_v1(**kwargs)
    kwargs=_case()
    kwargs["target_rgb"]=torch.rand((1,8,3,16,16))
    with pytest.raises(ValueError,match="SIX_RGB_AND_NORMAL_VIEWS_REQUIRED"):
        era3d_aligned_unclip_noise_microstep_v1(**kwargs)


def test_noise_schedule_repeats_per_object_and_modality_exactly_six_times():
    kwargs=_case()
    class _LogScheduler(_Scheduler):
        def __init__(self): self.steps=[]
        def add_noise(self,x,noise,t):
            self.steps.append(t.clone())
            return super().add_noise(x,noise,t)
    s=_LogScheduler()
    kwargs["diffusion_scheduler"]=s
    loss=era3d_aligned_unclip_noise_microstep_v1(**kwargs)
    assert torch.isfinite(loss)
    assert torch.equal(s.steps[0],torch.tensor([10]*6+[20]*6))
