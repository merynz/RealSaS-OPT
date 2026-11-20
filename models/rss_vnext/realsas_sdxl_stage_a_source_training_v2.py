from __future__ import annotations

"""Real SDXL paired-image Stage A: source-rendered canonical pose supervision.

The official SDXL base UNet remains pretrained; train only its installed LoRA,
a two-original-image reference conditioner and a learned canonical alpha head.
Input: TWO original RGBA images -> frozen encoder -> independent image token
streams. Output teachers: TWO independently source-rendered canonical pose
RGBA targets, not student predictions and not guessed original camera poses.
"""

from dataclasses import dataclass
from typing import Any
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .paired_sdxl_lora_prior_v1 import (
    RSSvNextTwoImageConditionerV1,
    predict_sdxl_unet_noise_from_two_images_v1,
)

STAGE="SDXL_TWO_SOURCE_TO_TWO_CANONICAL_RGBA_V2"
AUTHORITY="SOURCE_3D_RENDERED_CANONICAL_TRUTH"


class SDXLCanonicalAlphaHeadV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.project=nn.Sequential(
            nn.Conv2d(4,32,3,padding=1),
            nn.SiLU(),
            nn.Conv2d(32,1,3,padding=1),
        )

    def forward(self, x:Tensor, image_size:tuple[int,int])->Tensor:
        if x.ndim!=4 or x.shape[1]!=4 or min(image_size)<1:
            raise ValueError("RSS_VNEXT_SDXL_STAGE_A_ALPHA_INPUT_INVALID")
        return F.interpolate(
            self.project(x.float()),size=image_size,
            mode="bilinear",align_corners=False,
        )


@dataclass(frozen=True)
class SDXLSourceStageABatchV2:
    # Target canonical renders come from SOURCE 3D, never A predictions.
    source_tokens:Tensor            # B,2,T,D frozen source encoder
    source_token_valid:Tensor       # B,2,T boolean
    target_canonical_rgba:Tensor    # B,2,4,H,W source truth
    target_valid:Tensor             # B,2,1,H,W independent valid pixels
    target_cameras:Tensor           # B,2,13 virtual canonical cameras
    canonical_pose_queries:Tensor  # B,2,P generic canonical pose code
    prompt_embeds:Tensor           # B,T,2048 frozen text encoders
    pooled_text_embeds:Tensor      # B,1280 frozen text
    time_ids:Tensor                # B,6 SDXL resolution/crop IDs
    source_render_qualification_key:str
    evidence_authority:str=AUTHORITY

    def validate(self,conditioner:RSSvNextTwoImageConditionerV1)->None:
        src=self.source_tokens
        if (
            type(conditioner) is not RSSvNextTwoImageConditionerV1
            or src.ndim!=4 or src.shape[1]!=2 or src.shape[2]<1
            or src.shape[-1]!=conditioner.source_dim
            or self.source_token_valid.shape!=src.shape[:3]
            or self.source_token_valid.dtype!=torch.bool
            or not bool(self.source_token_valid.any(dim=2).all())
            or src.requires_grad
            or self.target_canonical_rgba.ndim!=5
        ):
            raise ValueError("RSS_VNEXT_SDXL_STAGE_A_SOURCE_ENCODER_ABI")
        b,v,c,h,w=self.target_canonical_rgba.shape
        if (
            b!=src.shape[0] or v!=2 or c!=4
            or h%8 or w%8 or min(h,w)<64
            or self.target_valid.shape!=(b,2,1,h,w)
            or self.target_cameras.shape!=(b,2,13)
            or self.canonical_pose_queries.shape!=(b,2,conditioner.pose_dim)
            or self.prompt_embeds.ndim!=3
            or self.prompt_embeds.shape[0]!=b
            or self.prompt_embeds.shape[-1]!=2048
            or self.pooled_text_embeds.shape!=(b,1280)
            or self.time_ids.shape!=(b,6)
            or self.evidence_authority!=AUTHORITY
            or type(self.source_render_qualification_key) is not str
            or not self.source_render_qualification_key
        ):
            raise ValueError("RSS_VNEXT_SDXL_STAGE_A_RENDER_ABI")
        checks=(
            src,self.target_canonical_rgba,self.target_valid,
            self.target_cameras,self.canonical_pose_queries,
            self.prompt_embeds,self.pooled_text_embeds,self.time_ids,
        )
        if (
            any(
                not torch.is_floating_point(t)
                or not bool(torch.isfinite(t).all())
                or t.device!=src.device
                for t in checks
            )
            or any(bool(((t<0)|(t>1)).any()) for t in (
                self.target_canonical_rgba,self.target_valid,
            ))
            or not bool((self.target_valid>0).any())
        ):
            raise ValueError("RSS_VNEXT_SDXL_STAGE_A_SOURCE_RENDER_INVALID")


def two_source_sdxl_conditioning_v2(
    *,conditioner:RSSvNextTwoImageConditionerV1,
    source_tokens:Tensor,source_token_valid:Tensor,
    target_cameras:Tensor,canonical_pose_queries:Tensor,
)->Tensor:
    """Explicitly UNKNOWN original pose and camera — no training leakage."""
    b,v,_,_=source_tokens.shape
    if (
        v!=2 or target_cameras.shape!=(b,2,13)
        or canonical_pose_queries.shape!=(b,2,conditioner.pose_dim)
    ):
        raise ValueError("RSS_VNEXT_SDXL_STAGE_A_CANONICAL_TARGET_LAYOUT")
    contexts=[]
    for target in range(2):
        context=conditioner(
            observed_tokens=source_tokens,
            observed_token_valid=source_token_valid,
            original_cameras=source_tokens.new_zeros((b,2,13)),
            original_camera_known=torch.zeros(
                b,2,dtype=torch.bool,device=source_tokens.device,
            ),
            original_poses=source_tokens.new_zeros(
                (b,2,conditioner.pose_dim),
            ),
            original_pose_known=torch.zeros(
                b,2,dtype=torch.bool,device=source_tokens.device,
            ),
            target_camera=target_cameras[:,target],
            target_pose=canonical_pose_queries[:,target],
        )
        contexts.append(context)
    return torch.stack(contexts,dim=1).flatten(0,1)


@dataclass
class SDXLCanonicalStageALossV2:
    total_loss:Tensor
    rgb_epsilon_loss:Tensor
    alpha_loss:Tensor
    qualified_pixels:int


def source_sdxl_canonical_stage_a_loss_v2(
    *, unet:nn.Module,vae:Any,noise_scheduler:Any,
    conditioner:RSSvNextTwoImageConditionerV1,
    alpha_head:SDXLCanonicalAlphaHeadV2,
    batch:SDXLSourceStageABatchV2,
    noise:Tensor|None=None,timesteps:Tensor|None=None,
    alpha_weight:float=.1,
)->SDXLCanonicalStageALossV2:
    """Differentiable pretrained UNet LoRA + 2-image conditioner + alpha loss."""
    batch.validate(conditioner)
    if (
        type(alpha_head) is not SDXLCanonicalAlphaHeadV2
        or type(alpha_weight) not in (int,float)
        or not math.isfinite(alpha_weight) or alpha_weight<=0
        or getattr(noise_scheduler.config,"prediction_type",None)!="epsilon"
    ):
        raise ValueError("RSS_VNEXT_SDXL_STAGE_A_TRAIN_POLICY")
    b=batch.source_tokens.shape[0]
    h,w=batch.target_canonical_rgba.shape[-2:]
    rgba=batch.target_canonical_rgba
    rgb=rgba[:,:,:3]*rgba[:,:,3:4]+.5*(1.-rgba[:,:,3:4])
    with torch.no_grad():
        clean=vae.encode(
            rgb.reshape(b*2,3,h,w).to(vae.dtype)*2.-1.
        ).latent_dist.mode().float()*float(vae.config.scaling_factor)
    if clean.shape!=(b*2,4,h//8,w//8):
        raise ValueError("RSS_VNEXT_SDXL_STAGE_A_VAE_LATENTS")
    if noise is None: noise=torch.randn_like(clean)
    if noise.shape!=clean.shape or not bool(torch.isfinite(noise).all()):
        raise ValueError("RSS_VNEXT_SDXL_STAGE_A_NOISE_DRIFT")
    if timesteps is None:
        timesteps=torch.randint(
            0,int(noise_scheduler.config.num_train_timesteps),
            (b*2,),device=clean.device,dtype=torch.long,
        )
    if (
        timesteps.shape!=(b*2,) or timesteps.dtype!=torch.long
        or bool((timesteps<0).any())
        or bool((timesteps>=noise_scheduler.config.num_train_timesteps).any())
    ):
        raise ValueError("RSS_VNEXT_SDXL_STAGE_A_TIMESTEPS_INVALID")
    noisy=noise_scheduler.add_noise(clean,noise,timesteps)
    paired=two_source_sdxl_conditioning_v2(
        conditioner=conditioner,source_tokens=batch.source_tokens,
        source_token_valid=batch.source_token_valid,
        target_cameras=batch.target_cameras,
        canonical_pose_queries=batch.canonical_pose_queries,
    )
    predicted=predict_sdxl_unet_noise_from_two_images_v1(
        unet=unet,
        noisy_latents=noisy.to(dtype=next(unet.parameters()).dtype),
        timestep=timesteps,
        text_context=batch.prompt_embeds.repeat_interleave(2,dim=0).to(next(unet.parameters()).dtype),
        paired_image_context=paired.to(next(unet.parameters()).dtype),
        pooled_text_embedding=batch.pooled_text_embeds.repeat_interleave(2,dim=0).to(next(unet.parameters()).dtype),
        time_ids=batch.time_ids.repeat_interleave(2,dim=0),
    )
    mask=F.interpolate(
        batch.target_valid.reshape(b*2,1,h,w),
        size=(h//8,w//8),mode="nearest",
    )
    denom=mask.sum()*4
    if not bool(denom>0):
        raise ValueError("RSS_VNEXT_SDXL_STAGE_A_NO_VISIBLE_TARGET")
    epsilon=((predicted.float()-noise.float()).square()*mask).sum()/denom
    a=noise_scheduler.alphas_cumprod.to(
        device=noisy.device,dtype=noisy.dtype,
    )[timesteps].view(b*2,1,1,1)
    x0=(noisy-(1.-a).sqrt()*predicted)/a.sqrt().clamp_min(1.e-5)
    logits=alpha_head(x0,(h,w)).reshape(b,2,1,h,w)
    alpha=F.binary_cross_entropy_with_logits(
        logits,rgba[:,:,3:4].float(),
    )
    total=epsilon+alpha*float(alpha_weight)
    if not bool(torch.isfinite(total).all()) or not total.requires_grad:
        raise RuntimeError("RSS_VNEXT_SDXL_STAGE_A_LOSS_DISCONNECTED")
    return SDXLCanonicalStageALossV2(
        total_loss=total,rgb_epsilon_loss=epsilon,
        alpha_loss=alpha,
        qualified_pixels=int((batch.target_valid>0).sum()),
    )


__all__=[
    "STAGE","AUTHORITY","SDXLCanonicalAlphaHeadV2",
    "SDXLSourceStageABatchV2","SDXLCanonicalStageALossV2",
    "two_source_sdxl_conditioning_v2",
    "source_sdxl_canonical_stage_a_loss_v2",
]
