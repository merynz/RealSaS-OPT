from __future__ import annotations

"""Source-qualified RGB+normal joint training for official SDXL MV-Adapter v2.

No scratch multiview UNet: train the real pretrained MV-Adapter _mv/_ref
projections, T2I camera encoder and newly introduced BUSTED modality/alpha
heads. Each subject has 2 canonical references and six paired RGB/normal
targets. The two source references remain separate in native reference cache.
This step is differentiable; no GPU fit/quality is claimed by importing it.
"""

from dataclasses import dataclass
from typing import Any
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .realsas_mvadapter_guidance_policy_v1 import (
    OfficialMVAdapterGuidancePolicyV1,
    draw_official_mvadapter_dropout_v1,
    select_official_mvadapter_conditions_v1,
)

from .realsas_mvadapter_dual_reference_v2 import (
    MVAdapterRGBNormalTaskHeadsV2,
    capture_native_two_reference_cache_v2,
    native_mv_adapter_rgb_normal_denoise_v2,
)

STAGE = "SDXL_MVADAPTER_TWO_CANONICAL_TO_SIX_RGB_NORMAL_V2"
AUTHORITY = "SOURCE_3D_RENDERED_NOT_GENERATED_TEACHER"


def _in_unit_interval(t: Tensor) -> bool:
    return bool(torch.isfinite(t).all()) and not bool(
        ((t < 0) | (t > 1)).any()
    )


@dataclass(frozen=True)
class MVAdapterSourceStageBBatchV2:
    canonical_rgba: Tensor       # B,2,4,H,W — source 3D rendered canonical
    rgb: Tensor                  # B,6,3,H,W — independently rendered
    normal_encoded_rgb: Tensor   # B,6,3,H,W — independently rendered
    alpha: Tensor                # B,6,1,H,W — independent source alpha
    valid: Tensor                # B,6,1,H,W — qualified pixels, NOT fake masks
    plucker_controls: Tensor     # B,6,6,H,W — actual same six virtual cameras
    target_cameras: Tensor       # B,6,13 authoritative RSS v1 virtual bank
    prompt_embeds: Tensor        # B,T,2048 — frozen native text encoder outputs
    pooled_text_embeds: Tensor   # B,1280 — frozen native text encoder output
    empty_prompt_embeds: Tensor   # B,T,2048 — pinned empty native text encoders
    empty_pooled_text_embeds: Tensor # B,1280 — pinned empty pooled text
    time_ids: Tensor             # B,6 SDXL actual 6 scalar time/size ids
    source_render_qualification_key: str
    evidence_authority: str = AUTHORITY

    def validate(self) -> None:
        source=self.canonical_rgba
        if not isinstance(source,Tensor) or source.ndim!=5:
            raise ValueError("RSS_VNEXT_MVADAPTER_STAGE_B_SOURCE_RGBA_REQUIRED")
        b,v,c,h,w=source.shape
        if (
            b < 1 or v!=2 or c!=4 or h%8 or w%8 or min(h,w)<64
            or self.rgb.shape!=(b,6,3,h,w)
            or self.normal_encoded_rgb.shape!=(b,6,3,h,w)
            or self.alpha.shape!=(b,6,1,h,w)
            or self.valid.shape!=(b,6,1,h,w)
            or self.plucker_controls.shape!=(b,6,6,h,w)
            or self.target_cameras.shape!=(b,6,13)
            or self.prompt_embeds.ndim!=3
            or self.prompt_embeds.shape[0]!=b
            or self.prompt_embeds.shape[-1]!=2048
            or self.pooled_text_embeds.shape!=(b,1280)
            or self.empty_prompt_embeds.shape!=self.prompt_embeds.shape
            or self.empty_pooled_text_embeds.shape!=(b,1280)
            or self.time_ids.shape!=(b,6)
            or type(self.source_render_qualification_key) is not str
            or not self.source_render_qualification_key
            or self.evidence_authority!=AUTHORITY
        ):
            raise ValueError("RSS_VNEXT_MVADAPTER_STAGE_B_SOURCE_RENDER_ABI")
        tensors=(
            source,self.rgb,self.normal_encoded_rgb,self.alpha,self.valid,
            self.plucker_controls,self.target_cameras,
            self.prompt_embeds,self.pooled_text_embeds,
            self.empty_prompt_embeds,self.empty_pooled_text_embeds,
            self.time_ids,
        )
        if any(
            not torch.is_floating_point(t) or t.device!=source.device
            or not bool(torch.isfinite(t).all())
            for t in tensors
        ) or any(not _in_unit_interval(t) for t in (
            source,self.rgb,self.normal_encoded_rgb,self.alpha,self.valid,
            self.plucker_controls,
        )) or not bool((self.valid>0).any()):
            raise ValueError("RSS_VNEXT_MVADAPTER_STAGE_B_PIXELS_OR_SOURCE_NONFINITE")
        from .realsas_mvadapter_camera_bridge_v2 import (
            mv_adapter_controls_from_13d_virtual_cameras_v2,
        )
        actual=mv_adapter_controls_from_13d_virtual_cameras_v2(
            self.target_cameras,height=h,width=w,
        )
        if not torch.allclose(
            self.plucker_controls,actual,atol=1e-6,rtol=0,
        ):
            raise ValueError("RSS_VNEXT_MVADAPTER_STAGE_B_CAMERA_CONTROL_MISMATCH")


@dataclass
class MVAdapterPairedStageBLossV2:
    total_loss: Tensor
    rgb_epsilon_loss: Tensor
    normal_epsilon_loss: Tensor
    alpha_loss: Tensor
    active_rgb_pixels: int
    active_normal_pixels: int
    prompt_dropped_subjects: int = 0
    reference_dropped_subjects: int = 0
    joint_dropped_subjects: int = 0


@torch.no_grad()
def _encode_rgb_to_vae_latents(vae:Any, color:Tensor)->Tensor:
    """RGB in [0,1] -> real VAE latents; never optimize against JPEG proxies."""
    if color.ndim!=4 or color.shape[1]!=3:
        raise ValueError("RSS_VNEXT_MVADAPTER_STAGE_B_VAE_COLOR_SHAPE")
    result=[]
    for chunk in color.split(6,dim=0):
        posterior=vae.encode(
            (chunk.to(dtype=vae.dtype)*2.-1.)
        ).latent_dist
        result.append(
            posterior.mode().float()*float(vae.config.scaling_factor)
        )
    return torch.cat(result,dim=0)


def mv_adapter_stage_b_loss_v2(
    *, pipe:Any, task_heads:MVAdapterRGBNormalTaskHeadsV2,
    source:MVAdapterSourceStageBBatchV2,
    noise_scheduler:Any,
    noise:Tensor|None=None,
    timesteps:Tensor|None=None,
    alpha_weight:float=.1,
    guidance_policy:OfficialMVAdapterGuidancePolicyV1=(
        OfficialMVAdapterGuidancePolicyV1()
    ),
)->MVAdapterPairedStageBLossV2:
    """True source-rendered epsilon+normal+alpha, real native MV attention.

    Batch order [B,2 modalities,6 views]. We use identical timestep within
    each 6-view group (as in upstream MV-Adapter training). Different RGB and
    normal groups can use different noise draws but share the *same camera*.
    """
    source.validate()
    guidance_policy.validate()
    if (
        type(task_heads) is not MVAdapterRGBNormalTaskHeadsV2
        or type(alpha_weight) not in (int,float)
        or not math.isfinite(alpha_weight) or alpha_weight<=0
        or getattr(noise_scheduler.config,"prediction_type",None)!="epsilon"
    ):
        raise ValueError("RSS_VNEXT_MVADAPTER_STAGE_B_TRAINING_POLICY")
    b=source.canonical_rgba.shape[0]
    h,w=source.canonical_rgba.shape[-2:]
    # Upstream encodes the reference UNet with the SAME prompt dropout mask
    # used by target denoising. Dropping the target text but encoding
    # reference keys with positive text would silently leak the prompt.
    dropped=draw_official_mvadapter_dropout_v1(
        b,device=source.canonical_rgba.device,policy=guidance_policy,
    )
    reference_prompt=torch.where(
        dropped.prompt[:,None,None],
        source.empty_prompt_embeds,source.prompt_embeds,
    )
    reference_pooled=torch.where(
        dropped.prompt[:,None],
        source.empty_pooled_text_embeds,source.pooled_text_embeds,
    )
    # Official reference preprocessing (gray under transparent source).
    rgba=source.canonical_rgba
    canonical_rgb=rgba[:,:,:3]*rgba[:,:,3:4]+.5*(1.-rgba[:,:,3:4])
    ref_latents=_encode_rgb_to_vae_latents(
        pipe.vae,canonical_rgb.reshape(b*2,3,h,w)
    ).reshape(b,2,4,h//8,w//8)
    cache=capture_native_two_reference_cache_v2(
        unet=pipe.unet,canonical_latents=ref_latents,
        prompt_embeds=reference_prompt,
        pooled_text_embeds=reference_pooled,
        time_ids=source.time_ids,modalities=2,
    )
    target=torch.stack(
        (source.rgb,source.normal_encoded_rgb),dim=1,
    )   # B,2,6,3,H,W
    clean=_encode_rgb_to_vae_latents(
        pipe.vae,target.reshape(b*12,3,h,w),
    ).reshape(b,2,6,4,h//8,w//8)
    if noise is None: noise=torch.randn_like(clean)
    if noise.shape!=clean.shape or not bool(torch.isfinite(noise).all()):
        raise ValueError("RSS_VNEXT_MVADAPTER_STAGE_B_NOISE_SHAPE_DRIFT")
    if timesteps is None:
        # Same noise level across the SIX views in each row-wise group.
        schedule=torch.randint(
            0,int(noise_scheduler.config.num_train_timesteps),
            (b,2,1),device=clean.device,
        )
        timesteps=schedule.expand(b,2,6).clone()
    if (
        timesteps.shape!=(b,2,6) or timesteps.dtype!=torch.long
        or bool((timesteps<0).any())
        or bool((timesteps>=noise_scheduler.config.num_train_timesteps).any())
        or not bool((timesteps==timesteps[:,:,:1]).all())
    ):
        raise ValueError("RSS_VNEXT_MVADAPTER_STAGE_B_GROUP_TIMESTEP_DRIFT")
    noisy=noise_scheduler.add_noise(
        clean.reshape(b*12,4,h//8,w//8),
        noise.reshape(b*12,4,h//8,w//8),
        timesteps.reshape(b*12),
    ).reshape_as(clean)
    # Official I2MV dropout is one decision per *source identity*, shared
    # by all six cameras and RGB/normal channels. Empty text is a genuine
    # frozen SDXL empty-prompt encoding, NOT a zero-valued text surrogate.
    text,pooled,conditioned_cache=select_official_mvadapter_conditions_v1(
        conditional_prompt=source.prompt_embeds,
        conditional_pooled=source.pooled_text_embeds,
        empty_prompt=source.empty_prompt_embeds,
        empty_pooled=source.empty_pooled_text_embeds,
        reference_cache=cache,dropout=dropped,
    )
    cameras=source.plucker_controls[:,None].expand(-1,2,-1,-1,-1,-1)
    pred=native_mv_adapter_rgb_normal_denoise_v2(
        pipe=pipe,task_heads=task_heads,
        noisy_latents=noisy.to(dtype=next(pipe.unet.parameters()).dtype),
        timesteps=timesteps,
        cache=conditioned_cache,prompt_embeds=text,
        pooled_text_embeds=pooled,
        time_ids=source.time_ids,
        control_plucker=cameras,
    )
    mask=F.interpolate(
        source.valid.reshape(b*6,1,h,w),size=(h//8,w//8),
        mode="nearest",
    ).reshape(b,6,1,h//8,w//8)
    valid=mask[:,None].expand(-1,2,-1,-1,-1,-1)
    square=(pred.float()-noise.float()).square()
    weighted=(square*valid).sum(dim=(2,3,4,5))
    denom=valid.sum(dim=(2,3,4,5))*4
    if not bool((denom>0).all()):
        raise ValueError("RSS_VNEXT_MVADAPTER_STAGE_B_MODALITY_TRUTH_EMPTY")
    losses=weighted/denom
    rgb=losses[:,0].mean()
    normal=losses[:,1].mean()
    # Epsilon -> denoised x0 in the SD noise schedule, used to train a
    # genuinely learned visibility channel rather than thresholding RGB.
    alphabar=noise_scheduler.alphas_cumprod.to(
        device=noisy.device,dtype=noisy.dtype,
    )[timesteps[:,0].reshape(b*6)].reshape(b,6,1,1,1)
    reconstructed=(
        noisy[:,0] - (1.-alphabar).sqrt()*pred[:,0]
    )/alphabar.sqrt().clamp_min(1.e-5)
    alpha_pred=task_heads.alpha_logits(
        reconstructed.reshape(b*6,4,h//8,w//8),
        (h,w),
    ).reshape(b,6,1,h,w)
    # Source-qualified pixels only; unknown alpha cannot supervise a
    # fabricated transparent/opaque negative.
    alpha_raw=F.binary_cross_entropy_with_logits(
        alpha_pred,source.alpha.float(),reduction="none",
    )
    alpha_valid=source.valid.float()
    alpha=(alpha_raw*alpha_valid).sum()/alpha_valid.sum().clamp_min(1.)
    loss=rgb+normal+float(alpha_weight)*alpha
    if not bool(torch.isfinite(loss).all()) or not loss.requires_grad:
        raise RuntimeError("RSS_VNEXT_MVADAPTER_STAGE_B_LOSS_DISCONNECTED")
    return MVAdapterPairedStageBLossV2(
        total_loss=loss,rgb_epsilon_loss=rgb,
        normal_epsilon_loss=normal,alpha_loss=alpha,
        active_rgb_pixels=int((source.valid>0).sum()),
        active_normal_pixels=int((source.valid>0).sum()),
        prompt_dropped_subjects=int(dropped.prompt.sum()),
        reference_dropped_subjects=int(dropped.reference.sum()),
        joint_dropped_subjects=int(dropped.joint.sum()),
    )


__all__=[
    "STAGE","AUTHORITY","MVAdapterSourceStageBBatchV2",
    "MVAdapterPairedStageBLossV2","mv_adapter_stage_b_loss_v2",
]
