from __future__ import annotations

"""MV-Adapter SDXL dual-reference cache and trainable modality extension.

This is a clean-room extension to the OFFICIAL pretrained MV-Adapter I2MV
UNet/cond_encoder/attention processors. Unlike the old external-latent-row
denoiser, it feeds both canonical reference images through the actual native
reference UNet, concatenating reference attention keys/values *by tokens*
without altering trained _ref/_mv attention projection weights.

Batch layout: B subjects x M modalities x V=6 views. Each modality is its
own six-view row-attention group, not twelve competing camera views.
No source/native teacher is read by inference and no pretrained RGB checkpoint
is falsely claimed to have already learned normal maps.
"""

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

SCHEMA = "RealSaS.RSSvNext.NativeMVAdapterDualReferenceRGBNormal.v2"
MODALITIES = ("RGB", "NORMAL")
VIEWS = 6


@dataclass(frozen=True)
class PairedReferenceCacheV2:
    """Immutable container; tensors keep the exact B,M,V grouping."""
    features: dict[str, Tensor]  # each [B*M*6, 2*T, C]
    batch_size: int
    modalities: int
    num_views: int
    reference_count: int = 2

    def validate(self) -> None:
        if (
            type(self.features) is not dict or not self.features
            or type(self.batch_size) is not int or self.batch_size < 1
            or self.modalities not in (1, 2) or self.num_views != VIEWS
            or self.reference_count != 2
        ):
            raise ValueError("RSS_VNEXT_MVADAPTER_PAIRED_CACHE_DOMAIN")
        expected = self.batch_size * self.modalities * self.num_views
        for key, x in self.features.items():
            if (
                type(key) is not str or not key
                or not isinstance(x, Tensor) or x.ndim != 3
                or x.shape[0] != expected or x.shape[1] < 2
                or x.shape[1] % 2 != 0 or x.shape[-1] < 1
                or not bool(torch.isfinite(x).all())
            ):
                raise ValueError("RSS_VNEXT_MVADAPTER_PAIRED_CACHE_FEATURE_DOMAIN")


def official_mv_adapter_trainable_parameters_v2(
    pipe: Any, *, train_reference: bool = True,
) -> tuple[tuple[str, nn.Parameter], ...]:
    """Match *published MV-Adapter SDXL recipe*: _mv/_ref and cond_encoder.

    Do not train a random new multiview module or silently unfreeze entire
    SDXL. A real loaded reference attention checkpoint is mandatory.
    """
    if (
        not hasattr(pipe, "unet") or not isinstance(pipe.unet, nn.Module)
        or not hasattr(pipe, "cond_encoder")
        or not isinstance(pipe.cond_encoder, nn.Module)
    ):
        raise TypeError("RSS_VNEXT_MVADAPTER_OFFICIAL_MODULES_REQUIRED")
    result: list[tuple[str, nn.Parameter]] = []
    for name, parameter in pipe.unet.named_parameters():
        enabled = "_mv" in name or (train_reference and "_ref" in name)
        parameter.requires_grad_(enabled)
        parameter.grad = None
        if enabled:
            result.append(("unet."+name, parameter))
    for name, parameter in pipe.cond_encoder.named_parameters():
        parameter.requires_grad_(True)
        parameter.grad = None
        result.append(("cond_encoder."+name, parameter))
    if (
        not any("_mv" in name for name, _ in result)
        or not any("_ref" in name for name, _ in result)
        or not any(name.startswith("cond_encoder.") for name, _ in result)
    ):
        raise RuntimeError("RSS_VNEXT_MVADAPTER_TRAINED_ATTENTION_NOT_PRESENT")
    if hasattr(pipe, "vae"):
        pipe.vae.requires_grad_(False)
    if hasattr(pipe, "text_encoder"):
        pipe.text_encoder.requires_grad_(False)
    if hasattr(pipe, "text_encoder_2"):
        pipe.text_encoder_2.requires_grad_(False)
    return tuple(result)


@torch.no_grad()
def capture_native_two_reference_cache_v2(
    *, unet: nn.Module, canonical_latents: Tensor,
    prompt_embeds: Tensor, pooled_text_embeds: Tensor,
    time_ids: Tensor, modalities: int = 2,
) -> PairedReferenceCacheV2:
    """Native official reference feature capture with TWO canonical views.

    The actual upstream attention processor writes to cache_hidden_states.
    Concatenation along the TOKEN dimension preserves both image histories,
    whereas averaging would erase distinct unseen surfaces. K/V projections
    remain the exact trained _ref adapter from the official checkpoint.
    """
    if (
        not isinstance(unet, nn.Module)
        or not isinstance(canonical_latents, Tensor)
        or canonical_latents.ndim != 5 or canonical_latents.shape[1:3] != (2,4)
        or modalities not in (1,2)
        or prompt_embeds.ndim != 3
        or prompt_embeds.shape[0] != canonical_latents.shape[0]
        or prompt_embeds.shape[2] != 2048
        or pooled_text_embeds.ndim != 2
        or pooled_text_embeds.shape[0] != canonical_latents.shape[0]
        or time_ids.shape != (canonical_latents.shape[0],6)
        or any(not bool(torch.isfinite(t).all()) for t in (
            canonical_latents,prompt_embeds,pooled_text_embeds,time_ids,
        ))
    ):
        raise ValueError("RSS_VNEXT_MVADAPTER_DUAL_REF_SOURCE_ABI_INVALID")
    b = canonical_latents.shape[0]
    native: dict[str, Tensor] = {}
    timesteps = torch.zeros(
        b*2, device=canonical_latents.device, dtype=torch.long,
    )
    unet(
        canonical_latents.reshape(b*2,*canonical_latents.shape[2:]),
        timesteps,
        encoder_hidden_states=prompt_embeds.repeat_interleave(2,dim=0),
        added_cond_kwargs={
            "text_embeds":pooled_text_embeds.repeat_interleave(2,dim=0),
            "time_ids":time_ids.repeat_interleave(2,dim=0),
        },
        cross_attention_kwargs={
            "cache_hidden_states":native,
            "use_mv":False,
            "use_ref":False,
        },
        return_dict=False,
    )
    if not native:
        raise RuntimeError("RSS_VNEXT_MVADAPTER_UPSTREAM_REF_CACHE_NOT_WRITTEN")
    features={}
    for name, hidden in native.items():
        if (
            not isinstance(hidden, Tensor) or hidden.ndim != 3
            or hidden.shape[0] != b*2 or hidden.shape[1] < 1
            or not bool(torch.isfinite(hidden).all())
        ):
            raise RuntimeError("RSS_VNEXT_MVADAPTER_REF_FEATURE_DIMENSIONS_DRIFT")
        # [B,2,T,C] -> [B,2T,C] -> [B,M,6,2T,C] -> [B*M*6,2T,C].
        paired=hidden.reshape(b,2,*hidden.shape[1:]).flatten(1,2)
        features[name]=paired.repeat_interleave(modalities*VIEWS,dim=0)
    out=PairedReferenceCacheV2(
        features=features,batch_size=b,modalities=modalities,num_views=VIEWS,
    )
    out.validate()
    return out


class MVAdapterRGBNormalTaskHeadsV2(nn.Module):
    """Explicit NEW trainable task parameters; never claimed upstream weights.

    Embedding distinguishes the two 6-view diffusion modalities. A separate
    alpha head is trained against genuine rendered source masks; RGB checkpoints
    by themselves do not generate reliable alpha or normal maps.
    """
    def __init__(self, hidden_dim: int=2048):
        super().__init__()
        if hidden_dim != 2048:
            raise ValueError("RSS_VNEXT_MVADAPTER_REQUIRES_SDXL_CONTEXT_2048")
        self.modality_token=nn.Embedding(2,hidden_dim)
        nn.init.zeros_(self.modality_token.weight)
        self.alpha_head=nn.Sequential(
            nn.Conv2d(4,32,3,padding=1),
            nn.SiLU(),
            nn.Conv2d(32,1,3,padding=1),
        )

    def context_for_modalities(
        self,prompt_embeds:Tensor,*,modalities:int=2,
    )->Tensor:
        """B x text tokens -> (B*M*6) x (tokens+1) x 2048."""
        if (
            prompt_embeds.ndim!=3 or prompt_embeds.shape[-1]!=2048
            or modalities not in (1,2)
        ):
            raise ValueError("RSS_VNEXT_MVADAPTER_MODAL_TEXT_CONTEXT_ABI")
        b=prompt_embeds.shape[0]
        repeated=prompt_embeds[:,None,None,:,:].expand(
            b,modalities,VIEWS,-1,-1,
        )
        labels=torch.arange(modalities,device=prompt_embeds.device)
        embed=self.modality_token(labels).to(prompt_embeds.dtype)
        new_token=embed[None,:,None,None,:].expand(b,-1,VIEWS,1,-1)
        return torch.cat((repeated,new_token),dim=-2).reshape(
            b*modalities*VIEWS,prompt_embeds.shape[1]+1,2048,
        )

    def alpha_logits(self,estimated_clean_latents:Tensor,
                     output_size:tuple[int,int])->Tensor:
        if (
            estimated_clean_latents.ndim!=4
            or estimated_clean_latents.shape[1]!=4
            or len(output_size)!=2
            or min(output_size)<1
        ):
            raise ValueError("RSS_VNEXT_MVADAPTER_ALPHA_INPUT_INVALID")
        return F.interpolate(
            self.alpha_head(estimated_clean_latents.float()),
            size=output_size,mode="bilinear",align_corners=False,
        )


def native_mv_adapter_rgb_normal_denoise_v2(
    *,pipe:Any,task_heads:MVAdapterRGBNormalTaskHeadsV2,
    noisy_latents:Tensor,timesteps:Tensor,
    cache:PairedReferenceCacheV2,prompt_embeds:Tensor,
    pooled_text_embeds:Tensor,time_ids:Tensor,
    control_plucker:Tensor,
) -> Tensor:
    """Exact upstream MV row-attn + _ref + T2IAdapter; two modalities grouped.

    All tensors are owned by the caller's training/inference stage. The
    controls must be generated from qualified virtual cameras, never source
    camera guesses. No guidance/CFG in this core conditional training pass.
    """
    cache.validate()
    b,m,v=cache.batch_size,cache.modalities,cache.num_views
    if (
        type(task_heads) is not MVAdapterRGBNormalTaskHeadsV2
        or noisy_latents.ndim!=6
        or noisy_latents.shape[:4]!=(b,m,VIEWS,4)
        or timesteps.shape!=(b,m,VIEWS) or timesteps.dtype!=torch.long
        or prompt_embeds.shape[0]!=b
        or pooled_text_embeds.shape[0]!=b
        or time_ids.shape!=(b,6)
        or control_plucker.ndim!=6
        or control_plucker.shape[:4]!=(b,m,VIEWS,6)
        or any(not bool(torch.isfinite(x).all()) for x in (
            noisy_latents,control_plucker,prompt_embeds,pooled_text_embeds,time_ids,
        ))
    ):
        raise ValueError("RSS_VNEXT_MVADAPTER_PAIRED_RGB_NORMAL_BATCH_ABI")
    latent=noisy_latents.reshape(b*m*VIEWS,*noisy_latents.shape[-3:])
    steps=timesteps.reshape(b*m*VIEWS)
    controls=control_plucker.reshape(
        b*m*VIEWS,*control_plucker.shape[-3:],
    )
    target_context=task_heads.context_for_modalities(
        prompt_embeds,modalities=m,
    )
    pooled=pooled_text_embeds.repeat_interleave(m*VIEWS,dim=0)
    times=time_ids.repeat_interleave(m*VIEWS,dim=0)
    down=pipe.cond_encoder(controls.to(dtype=latent.dtype))
    output=pipe.unet(
        latent,steps,
        encoder_hidden_states=target_context.to(dtype=latent.dtype),
        added_cond_kwargs={"text_embeds":pooled,"time_ids":times},
        cross_attention_kwargs={
            "ref_hidden_states":cache.features,
            "num_views":VIEWS,
        },
        down_intrablock_additional_residuals=down,
        return_dict=True,
    )
    predicted=output.sample
    if predicted.shape!=latent.shape or not bool(torch.isfinite(predicted).all()):
        raise RuntimeError("RSS_VNEXT_MVADAPTER_EPSILON_OUTPUT_SHAPE_DRIFT")
    return predicted.reshape_as(noisy_latents)


__all__=[
    "SCHEMA","MODALITIES","VIEWS","PairedReferenceCacheV2",
    "official_mv_adapter_trainable_parameters_v2",
    "capture_native_two_reference_cache_v2",
    "MVAdapterRGBNormalTaskHeadsV2",
    "native_mv_adapter_rgb_normal_denoise_v2",
]
