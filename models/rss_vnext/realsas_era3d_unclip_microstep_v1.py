from __future__ import annotations

"""Independently written Era3D-style SD2.1 UnCLIP training forward.

The reference is Era3D's published train-unclip-512-6view configuration and
training algorithm; the CVPR 2025 reference §3.1 adopts and retrains this approach with
progressive 1024 resolution. Load the ancestral base from SD2.1 UnCLIP,
NOT an Era3D or RealSaS task-specific weight. This is a task training step;
it cannot infer canonical geometry and does not manufacture source truth.

Training input must be independent 3D renderer truth. A cleanroom UNet with
in-UNet feature row-wise attention + 8-channel input is required. Base
encoder, CLIP image encoder and VAE stay frozen; *full UNet* fine-tunes.
Only baseline one-canonical-source mode is included here; paired-source
is a separate product ablation, never silently replacing source pixels.
"""

import math
from typing import Any
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def expand_pretrained_unclip_input_conv_v1(conv: nn.Conv2d) -> nn.Conv2d:
    """Era3D zero-init the four extra source conditioning channels.

    The original SD2.1 4-channel denoiser output is preserved at step zero
    for all zero extra latents, regardless of bias and convolution padding.
    """
    if type(conv) is not nn.Conv2d or conv.in_channels != 4 or conv.groups != 1:
        raise ValueError("RSS_VNEXT_ERA3D_EXPECT_PRETRAINED_FOUR_CHANNEL_CONV")
    conv8=nn.Conv2d(
        in_channels=8,out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,stride=conv.stride,
        padding=conv.padding,dilation=conv.dilation,
        padding_mode=conv.padding_mode,bias=conv.bias is not None,
        device=conv.weight.device,dtype=conv.weight.dtype,
    )
    with torch.no_grad():
        conv8.weight.zero_()
        conv8.weight[:,:4].copy_(conv.weight)
        if conv.bias is not None:
            conv8.bias.copy_(conv.bias)
    return conv8


def _unclip_timestep_embedding(timesteps: Tensor, width: int) -> Tensor:
    """Fixed frequency embedding matching SD2.1 UnCLIP convention."""
    if timesteps.ndim != 1 or width < 4:
        raise ValueError("RSS_VNEXT_ERA3D_IMAGE_NOISE_LEVEL")
    half=width//2
    frequencies=torch.exp(
        -math.log(10000.0)*torch.arange(
            half,device=timesteps.device,dtype=torch.float32,
        )/max(half,1)
    )
    angle=timesteps.float()[:,None]*frequencies[None,:]
    embedding=torch.cat((angle.cos(),angle.sin()),dim=-1)
    return F.pad(embedding,(0,width-embedding.shape[-1]))


def _latent(vae: Any, images: Tensor, *, mode: bool) -> Tensor:
    dist=vae.encode(images*2.-1.).latent_dist
    value=dist.mode() if mode else dist.sample()
    value=value*float(vae.config.scaling_factor)
    if value.ndim != 4 or not bool(torch.isfinite(value).all()):
        raise ValueError("RSS_VNEXT_ERA3D_INVALID_VAE_LATENT")
    return value


def _safe_image_embedding(
    image_features: Tensor, *,
    normalizer: Any, scheduler: Any,
) -> Tensor:
    """UnCLIP CLIP image embeddings + fixed level-zero noise embedding."""
    if image_features.ndim != 2 or not bool(torch.isfinite(image_features).all()):
        raise ValueError("RSS_VNEXT_ERA3D_IMAGE_EMBEDDING_INVALID")
    count,width=image_features.shape
    time=torch.zeros((count,),device=image_features.device,dtype=torch.long)
    with torch.no_grad():
        normalized=normalizer.scale(image_features)
        # level 0 is the reference configuration, not independent task noise
        noised=scheduler.add_noise(normalized,torch.zeros_like(normalized),time)
        noised=normalizer.unscale(noised)
    return torch.cat((noised,_unclip_timestep_embedding(time,width).to(noised.dtype)),dim=-1)


def era3d_aligned_unclip_noise_microstep_v1(
    *,
    reference_rgb: Tensor, target_rgb: Tensor, target_normal_encoded_rgb: Tensor,
    target_rgb_valid: Tensor, target_normal_valid: Tensor,
    reference_clip_preprocessed: Tensor,
    prompt_rgb: Tensor, prompt_normal: Tensor,
    vae: Any, clip_image_encoder: Any, image_normalizer: Any,
    image_noise_scheduler: Any, diffusion_scheduler: Any,
    unet: nn.Module, noise: Tensor | None = None,
    object_timesteps: Tensor | None = None,
    snr_gamma: float | None = 5.0,
) -> Tensor:
    """Loss with Era3D's six simultaneous RGB+normal 8-channel UNet inputs.

    Normal maps must already be *encoded to RGB [0,1]* by the independent
    3D renderer; do not use a generated normal map as ground truth.
    Prompt embeddings must be locally derived from the pinned CLIP text
    encoder, not the RealSaS released fixed prompt tensor.
    """
    if reference_rgb.ndim != 4 or reference_rgb.shape[1] != 3:
        raise ValueError("RSS_VNEXT_ERA3D_SINGLE_CANONICAL_REFERENCE_REQUIRED")
    b,_,h,w=reference_rgb.shape
    if b<1 or b>8 or min(h,w)<16 or max(h,w)>1024:
        raise ValueError("RSS_VNEXT_ERA3D_SOURCE_IMAGE_BUDGET")
    if target_rgb.shape!=(b,6,3,h,w) or target_normal_encoded_rgb.shape!=(b,6,3,h,w):
        raise ValueError("RSS_VNEXT_ERA3D_SIX_RGB_AND_NORMAL_VIEWS_REQUIRED")
    if (
        target_rgb_valid.shape!=(b,6,1,h,w)
        or target_normal_valid.shape!=(b,6,1,h,w)
        or prompt_rgb.ndim!=4 or prompt_rgb.shape[:2]!=(b,6)
        or prompt_normal.shape!=prompt_rgb.shape
        or reference_clip_preprocessed.shape[0]!=b
    ):
        raise ValueError("RSS_VNEXT_ERA3D_MODALITY_OR_CLIP_CONTRACT")
    for x in (reference_rgb,target_rgb,target_normal_encoded_rgb,
              target_rgb_valid,target_normal_valid,reference_clip_preprocessed,
              prompt_rgb,prompt_normal):
        if not torch.is_floating_point(x) or not bool(torch.isfinite(x).all()):
            raise ValueError("RSS_VNEXT_ERA3D_NONFINITE_TRAINING_INPUT")
    for x in (reference_rgb,target_rgb,target_normal_encoded_rgb,
              target_rgb_valid,target_normal_valid):
        if bool((x<0).any() or (x>1).any()):
            raise ValueError("RSS_VNEXT_ERA3D_TRAINING_IMAGE_RANGE")
    if not bool((target_rgb_valid>0).any()) or not bool((target_normal_valid>0).any()):
        raise ValueError("RSS_VNEXT_ERA3D_NO_RENDERER_SUPERVISION")
    if snr_gamma is not None and (not math.isfinite(snr_gamma) or snr_gamma<=0):
        raise ValueError("RSS_VNEXT_ERA3D_SNR_GAMMA_INVALID")
    if getattr(unet,"rowwise_attention_in_unet_features",False) is not True:
        raise ValueError("RSS_VNEXT_ERA3D_UNET_FEATURE_ROW_ATTENTION_REQUIRED")
    if int(unet.config.in_channels)!=8:
        raise ValueError("RSS_VNEXT_ERA3D_EIGHT_CHANNEL_UNET_REQUIRED")
    if getattr(diffusion_scheduler.config,"prediction_type",None)!="epsilon":
        raise ValueError("RSS_VNEXT_ERA3D_EPSILON_SCHEDULER_REQUIRED")
    if any(p.requires_grad for m in (vae,clip_image_encoder) for p in m.parameters()):
        raise ValueError("RSS_VNEXT_ERA3D_FROZEN_ENCODERS_REQUIRED")
    if not any(p.requires_grad for p in unet.parameters()):
        raise ValueError("RSS_VNEXT_ERA3D_FINETUNABLE_UNET_REQUIRED")
    if any(not p.requires_grad for name,p in unet.named_parameters()
           if "lora_" not in name and "rowwise_" not in name):
        raise ValueError("RSS_VNEXT_ERA3D_REJECT_LORA_ONLY_FINETUNING")

    with torch.no_grad():
        ref_emb=_latent(vae,reference_rgb,mode=True)
        images=torch.cat((target_normal_encoded_rgb,target_rgb),dim=0)
        gt_images=images.reshape(2*b*6,3,h,w)
        gt_latent=_latent(vae,gt_images,mode=False)
        clip_features=clip_image_encoder(reference_clip_preprocessed).image_embeds
        if clip_features.shape[0]!=b:
            raise ValueError("RSS_VNEXT_ERA3D_CLIP_BATCH_DRIFT")
        image_condition=_safe_image_embedding(
            clip_features,normalizer=image_normalizer,
            scheduler=image_noise_scheduler,
        )
    _,lat_c,lat_h,lat_w=ref_emb.shape
    if (
        gt_latent.shape!=(2*b*6,lat_c,lat_h,lat_w)
        or lat_c != 4 or image_condition.shape[0]!=b
    ):
        raise ValueError("RSS_VNEXT_ERA3D_LATENT_DOMAIN")
    # Order [NORMAL:B,V][RGB:B,V]; each group of six shares an object
    # timestep, but independent RGB/normal noise per view is retained.
    condition=ref_emb[:,None].expand(b,6,4,lat_h,lat_w)
    condition=torch.cat((condition,condition),dim=0)
    condition=condition.reshape(2*b*6,4,lat_h,lat_w)
    clip_condition=image_condition[:,None].expand(b,6,-1)
    clip_condition=torch.cat((clip_condition,clip_condition),dim=0)
    clip_condition=clip_condition.reshape(2*b*6,-1)
    prompts=torch.cat((prompt_normal,prompt_rgb),dim=0)
    prompts=prompts.reshape(2*b*6,*prompt_rgb.shape[2:])
    if object_timesteps is None:
        object_timesteps=torch.randint(
            0,int(diffusion_scheduler.config.num_train_timesteps),
            (2*b,),device=gt_latent.device,
        )
    if (
        object_timesteps.shape!=(2*b,) or object_timesteps.dtype!=torch.long
        or bool((object_timesteps<0).any())
        or bool((object_timesteps>=int(diffusion_scheduler.config.num_train_timesteps)).any())
    ):
        raise ValueError("RSS_VNEXT_ERA3D_SHARED_VIEW_TIMESTEP_INVALID")
    timesteps=object_timesteps.repeat_interleave(6)
    if noise is None: noise=torch.randn_like(gt_latent)
    if noise.shape!=gt_latent.shape or not bool(torch.isfinite(noise).all()):
        raise ValueError("RSS_VNEXT_ERA3D_LATENT_NOISE_INVALID")
    noisy=diffusion_scheduler.add_noise(gt_latent,noise,timesteps)
    model_input=torch.cat((noisy,condition),dim=1)
    pred=unet(
        model_input,timesteps,encoder_hidden_states=prompts,
        class_labels=clip_condition,
    )
    guess=pred.sample if hasattr(pred,"sample") else pred[0].sample
    if guess.shape!=noise.shape or not bool(torch.isfinite(guess).all()):
        raise ValueError("RSS_VNEXT_ERA3D_UNET_OUTPUT_INVALID")
    valid=torch.cat((target_normal_valid,target_rgb_valid),dim=0)
    valid=valid.reshape(2*b*6,1,h,w)
    valid=F.interpolate(valid,size=(lat_h,lat_w),mode="nearest")
    weighted=((guess.float()-noise.float())**2)*valid.float()
    denom=valid.sum(dim=(1,2,3)).clamp(min=1.)
    # Per-sample normalized MSE, avoid background/no-target dominating.
    values=weighted.sum(dim=(1,2,3))/(denom*4.)
    if snr_gamma is not None:
        alphas=diffusion_scheduler.alphas_cumprod.to(timesteps.device)[timesteps]
        snr=alphas/(1.-alphas).clamp(min=1e-6)
        values=values*torch.minimum(snr,torch.full_like(snr,snr_gamma))/snr.clamp(min=1e-6)
    return values.mean()


__all__=[
    "expand_pretrained_unclip_input_conv_v1",
    "era3d_aligned_unclip_noise_microstep_v1",
]
