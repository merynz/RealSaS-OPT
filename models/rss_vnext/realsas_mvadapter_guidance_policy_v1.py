from __future__ import annotations

"""Official MV-Adapter SDXL classifier-free training/inference conditioning.

Upstream I2MV SDXL uses 10% prompt, 10% reference, 10% combined dropout
and eval guidance scale 3.0. BUSTED extends these SAME learned native
_mv/_ref projection weights to two distinct canonical references and two
separate six-view RGB/normal groups. The dropout mask is per original
identity, never different for its six cameras or RGB/normal partners.
"""

from dataclasses import dataclass,asdict
from hashlib import sha256
import json
import math

import torch
from torch import Tensor

from .realsas_mvadapter_dual_reference_v2 import (
    PairedReferenceCacheV2,VIEWS,
)

SCHEMA="RealSaS.RSSvNext.NativeMVAdapterSDXLGuidance.v1"


@dataclass(frozen=True)
class OfficialMVAdapterGuidancePolicyV1:
    prompt_drop_prob:float=.1
    reference_drop_prob:float=.1
    combined_drop_prob:float=.1
    inference_guidance_scale:float=3.

    def validate(self)->None:
        values=(
            self.prompt_drop_prob,self.reference_drop_prob,
            self.combined_drop_prob,self.inference_guidance_scale,
        )
        if (
            any(type(x) not in (float,int) or not math.isfinite(x)
                for x in values)
            or any(x<0 or x>=1 for x in values[:3])
            or not 1.<=values[3]<=20.
            or sum(values[:3])>1.
        ):
            raise ValueError("RSS_VNEXT_MVADAPTER_GUIDANCE_POLICY_INVALID")

    @property
    def digest(self)->str:
        self.validate()
        return sha256(json.dumps(
            {"schema":SCHEMA,**asdict(self)},
            sort_keys=True,separators=(",",":"),
        ).encode()).hexdigest()


@dataclass(frozen=True)
class MVAdapterSubjectDropoutV1:
    prompt:Tensor      # [B] bool, including joint dropout
    reference:Tensor   # [B] bool, including joint dropout
    joint:Tensor       # [B] bool

    def validate(self,b:int)->None:
        if (
            any(not isinstance(x,Tensor) or x.dtype!=torch.bool
                or x.shape!=(b,) for x in (
                    self.prompt,self.reference,self.joint,
                ))
            or not bool((~self.joint | (
                self.prompt & self.reference
            )).all())
        ):
            raise ValueError("RSS_VNEXT_MVADAPTER_SUBJECT_DROPOUT_INVALID")


def draw_official_mvadapter_dropout_v1(
    batch_size:int,*,device:torch.device,
    policy:OfficialMVAdapterGuidancePolicyV1,
    generator:torch.Generator|None=None,
)->MVAdapterSubjectDropoutV1:
    policy.validate()
    if type(batch_size) is not int or batch_size<1:
        raise ValueError("RSS_VNEXT_MVADAPTER_DROPOUT_BATCH_INVALID")
    a=torch.rand(batch_size,device=device,generator=generator)
    b=torch.rand(batch_size,device=device,generator=generator)
    c=torch.rand(batch_size,device=device,generator=generator)
    joint=c<policy.combined_drop_prob
    out=MVAdapterSubjectDropoutV1(
        prompt=(a<policy.prompt_drop_prob)|joint,
        reference=(b<policy.reference_drop_prob)|joint,
        joint=joint,
    )
    out.validate(batch_size)
    return out


def zero_native_mvadapter_reference_cache_v1(
    cache:PairedReferenceCacheV2,
)->PairedReferenceCacheV2:
    cache.validate()
    out=PairedReferenceCacheV2(
        features={k:torch.zeros_like(v) for k,v in cache.features.items()},
        batch_size=cache.batch_size,
        modalities=cache.modalities,
        num_views=cache.num_views,
        reference_count=cache.reference_count,
    )
    out.validate()
    return out


def select_official_mvadapter_conditions_v1(
    *,conditional_prompt:Tensor,conditional_pooled:Tensor,
    empty_prompt:Tensor,empty_pooled:Tensor,
    reference_cache:PairedReferenceCacheV2,
    dropout:MVAdapterSubjectDropoutV1,
)->tuple[Tensor,Tensor,PairedReferenceCacheV2]:
    """Never mutate native references; dropout is identical for 12 targets.

    This is the published conditional dropout strategy expressed in
    BUSTED's two-canonical/12-target layout. Empty text comes from the
    pinned frozen SDXL tokenizer/encoders, not zeroed embedding tensors.
    """
    reference_cache.validate()
    b=reference_cache.batch_size
    dropout.validate(b)
    if (
        conditional_prompt.ndim!=3
        or conditional_prompt.shape[0]!=b
        or conditional_prompt.shape[-1]!=2048
        or empty_prompt.shape!=conditional_prompt.shape
        or conditional_pooled.shape!=(b,1280)
        or empty_pooled.shape!=conditional_pooled.shape
        or any(not bool(torch.isfinite(v).all()) for v in (
            conditional_prompt,conditional_pooled,
            empty_prompt,empty_pooled,
        ))
        or any(v.device!=conditional_prompt.device for v in (
            conditional_pooled,empty_prompt,empty_pooled,
            dropout.prompt,dropout.reference,dropout.joint,
        ))
    ):
        raise ValueError("RSS_VNEXT_MVADAPTER_GUIDANCE_CONDITION_ABI")
    p=torch.where(
        dropout.prompt[:,None,None],empty_prompt,conditional_prompt,
    )
    pooled=torch.where(
        dropout.prompt[:,None],empty_pooled,conditional_pooled,
    )
    m,v=reference_cache.modalities,reference_cache.num_views
    output={}
    for key,hidden in reference_cache.features.items():
        shape=hidden.shape
        condition=(
            ~dropout.reference
        ).reshape(b,1,1,1,1).to(hidden.dtype)
        output[key]=(
            hidden.reshape(b,m,v,shape[1],shape[2])*condition
        ).reshape(shape)
    returned=PairedReferenceCacheV2(
        features=output,batch_size=b,modalities=m,num_views=v,
        reference_count=reference_cache.reference_count,
    )
    returned.validate()
    return p,pooled,returned


__all__=[
    "SCHEMA","OfficialMVAdapterGuidancePolicyV1",
    "MVAdapterSubjectDropoutV1",
    "draw_official_mvadapter_dropout_v1",
    "zero_native_mvadapter_reference_cache_v1",
    "select_official_mvadapter_conditions_v1",
]
