from __future__ import annotations

"""Trained paired SDXL Stage A checkpoint and teacher-free two-view sampler.

Same LoRA+two-image conditioner+alpha-head ABI as source training. Does not
silently fall back to SDXL text-to-image, does not read source mesh/poses, and
refuses to generate from a newly initialized LoRA.
"""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .realsas_sdxl_stage_a_source_training_v2 import (
    SDXLCanonicalAlphaHeadV2,two_source_sdxl_conditioning_v2,
)
from .paired_sdxl_lora_prior_v1 import (
    RSSvNextTwoImageConditionerV1,
    predict_sdxl_unet_noise_from_two_images_v1,
)

SCHEMA="RealSaS.RSSvNext.PairedSDXLCanonicalTaskWeights.v2"


def _sha(x:object)->bool:
    return type(x) is str and len(x)==64 and all(
        c in "0123456789abcdef" for c in x
    )


def _file_sha(path:Path)->str:
    digest=sha256()
    with path.open("rb") as file:
        for block in iter(lambda:file.read(4*1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SDXLCanonicalTaskWeightsV2:
    checkpoint_path:str
    checkpoint_sha256:str
    pretrained_unet_sha256:str
    qualified_corpus_sha256:str
    trained_steps:int
    schema:str=SCHEMA
    paired_conditioner_trained:bool=True
    alpha_trained:bool=True

    def validate(self)->None:
        path=Path(self.checkpoint_path)
        if (
            self.schema!=SCHEMA
            or any(not _sha(v) for v in (
                self.checkpoint_sha256,
                self.pretrained_unet_sha256,
                self.qualified_corpus_sha256,
            ))
            or type(self.trained_steps) is not int
            or self.trained_steps<1
            or self.paired_conditioner_trained is not True
            or self.alpha_trained is not True
            or path.is_symlink() or not path.is_file()
        ):
            raise RuntimeError("RSS_VNEXT_SDXL_STAGE_A_REAL_TRAINED_WEIGHTS_REQUIRED")
        if _file_sha(path)!=self.checkpoint_sha256:
            raise RuntimeError("RSS_VNEXT_SDXL_STAGE_A_TASK_CHECKPOINT_SHA_DRIFT")


def _lora_params(unet:nn.Module)->dict[str,nn.Parameter]:
    items={name:parameter for name,parameter in unet.named_parameters()
           if "lora_" in name.lower()}
    if not items:
        raise RuntimeError("RSS_VNEXT_SDXL_STAGE_A_PRETRAINED_LORA_MODULES_REQUIRED")
    return items


def write_stage_a_task_weights_v2(
    path:str|Path,*,unet:nn.Module,
    conditioner:RSSvNextTwoImageConditionerV1,
    alpha_head:SDXLCanonicalAlphaHeadV2,
    trained_steps:int,pretrained_unet_sha256:str,
    qualified_corpus_sha256:str,
)->SDXLCanonicalTaskWeightsV2:
    if (
        type(trained_steps) is not int or trained_steps<1
        or not _sha(pretrained_unet_sha256)
        or not _sha(qualified_corpus_sha256)
    ):
        raise RuntimeError("RSS_VNEXT_SDXL_STAGE_A_UNTRAINED_SAVE_FORBIDDEN")
    data={
        "schema":SCHEMA,
        "trained_steps":trained_steps,
        "pretrained_unet_sha256":pretrained_unet_sha256,
        "qualified_corpus_sha256":qualified_corpus_sha256,
        "lora":{k:v.detach().cpu().contiguous()
                for k,v in _lora_params(unet).items()},
        "conditioner":{
            k:v.detach().cpu().contiguous()
            for k,v in conditioner.state_dict().items()
        },
        "alpha":{
            k:v.detach().cpu().contiguous()
            for k,v in alpha_head.state_dict().items()
        },
    }
    dest=Path(path)
    dest.parent.mkdir(parents=True,exist_ok=True)
    import tempfile,os
    with tempfile.NamedTemporaryFile(
        dir=dest.parent,prefix="rss-sdxl-a-",suffix=".pt",delete=False,
    ) as file:
        tmp=Path(file.name)
    try:
        torch.save(data,tmp)
        os.replace(tmp,dest)
    finally:
        tmp.unlink(missing_ok=True)
    return SDXLCanonicalTaskWeightsV2(
        checkpoint_path=str(dest),checkpoint_sha256=_file_sha(dest),
        pretrained_unet_sha256=pretrained_unet_sha256,
        qualified_corpus_sha256=qualified_corpus_sha256,
        trained_steps=trained_steps,
    )


def load_exact_stage_a_task_weights_v2(
    *,unet:nn.Module,conditioner:RSSvNextTwoImageConditionerV1,
    alpha_head:SDXLCanonicalAlphaHeadV2,receipt:SDXLCanonicalTaskWeightsV2,
)->None:
    receipt.validate()
    data=torch.load(receipt.checkpoint_path,map_location="cpu",weights_only=True)
    if (
        type(data) is not dict or data.get("schema")!=SCHEMA
        or data.get("trained_steps")!=receipt.trained_steps
        or data.get("pretrained_unet_sha256")!=receipt.pretrained_unet_sha256
        or data.get("qualified_corpus_sha256")!=receipt.qualified_corpus_sha256
        or type(data.get("lora")) is not dict
        or type(data.get("conditioner")) is not dict
        or type(data.get("alpha")) is not dict
    ):
        raise RuntimeError("RSS_VNEXT_SDXL_STAGE_A_TASK_CHECKPOINT_CONTENT_DRIFT")
    expected=_lora_params(unet)
    if set(data["lora"])!=set(expected):
        raise RuntimeError("RSS_VNEXT_SDXL_STAGE_A_LORA_KEY_DRIFT")
    for name,parameter in expected.items():
        item=data["lora"][name]
        if not isinstance(item,Tensor) or item.shape!=parameter.shape:
            raise RuntimeError("RSS_VNEXT_SDXL_STAGE_A_LORA_SHAPE_DRIFT")
    unet.load_state_dict(data["lora"],strict=False)
    conditioner.load_state_dict(data["conditioner"],strict=True)
    alpha_head.load_state_dict(data["alpha"],strict=True)
    unet.eval()
    conditioner.eval()
    alpha_head.eval()
    unet._rss_vnext_stage_a_task_sha256=receipt.checkpoint_sha256
    conditioner._rss_vnext_stage_a_task_sha256=receipt.checkpoint_sha256
    alpha_head._rss_vnext_stage_a_task_sha256=receipt.checkpoint_sha256


@dataclass(frozen=True)
class PairedSDXLCanonicalPredictionV2:
    rgba:Tensor  # B,2,4,H,W — lower-than-source authority
    trained_checkpoint_sha256:str
    source_observation_count:int=2
    output_canonical_count:int=2
    original_observation_authority:bool=False
    source_teacher_authority:bool=False
    schema:str=SCHEMA


@torch.no_grad()
def sample_trained_stage_a_canonical_v2(
    *,unet:nn.Module,vae:Any,scheduler:Any,
    conditioner:RSSvNextTwoImageConditionerV1,
    alpha_head:SDXLCanonicalAlphaHeadV2,
    receipt:SDXLCanonicalTaskWeightsV2,
    observed_tokens:Tensor,observed_token_valid:Tensor,
    target_cameras:Tensor,canonical_pose_queries:Tensor,
    prompt_embeds:Tensor,pooled_text_embeds:Tensor,
    time_ids:Tensor,height:int,width:int,steps:int,seed:int,
)->PairedSDXLCanonicalPredictionV2:
    receipt.validate()
    if any(
        getattr(module,"_rss_vnext_stage_a_task_sha256",None)
        !=receipt.checkpoint_sha256
        or module.training
        for module in (unet,conditioner,alpha_head)
    ):
        raise RuntimeError("RSS_VNEXT_SDXL_STAGE_A_UNTRAINED_MODEL_NOT_ADMITTED")
    if (
        observed_tokens.ndim!=4 or observed_tokens.shape[1]!=2
        or observed_tokens.shape[-1]!=conditioner.source_dim
        or observed_token_valid.shape!=observed_tokens.shape[:3]
        or observed_token_valid.dtype!=torch.bool
        or not bool(observed_token_valid.any(dim=2).all())
        or height<64 or width<64 or height%8 or width%8
        or type(steps) is not int or not 1<=steps<=100
        or type(seed) is not int or not 0<=seed<2**63
    ):
        raise ValueError("RSS_VNEXT_SDXL_STAGE_A_TWO_SOURCE_INFERENCE_ABI")
    b=observed_tokens.shape[0]
    context=two_source_sdxl_conditioning_v2(
        conditioner=conditioner,source_tokens=observed_tokens,
        source_token_valid=observed_token_valid,
        target_cameras=target_cameras,
        canonical_pose_queries=canonical_pose_queries,
    )
    prompt=prompt_embeds.repeat_interleave(2,dim=0)
    pooled=pooled_text_embeds.repeat_interleave(2,dim=0)
    tids=time_ids.repeat_interleave(2,dim=0)
    unet_dtype=next(unet.parameters()).dtype
    generator=torch.Generator(device=observed_tokens.device).manual_seed(seed)
    latents=torch.randn(
        b*2,4,height//8,width//8,generator=generator,
        device=observed_tokens.device,dtype=next(unet.parameters()).dtype,
    )
    scheduler.set_timesteps(steps,device=observed_tokens.device)
    latents*=float(scheduler.init_noise_sigma)
    for t in scheduler.timesteps:
        noisy=scheduler.scale_model_input(latents,t)
        noise=predict_sdxl_unet_noise_from_two_images_v1(
            unet=unet,noisy_latents=noisy,timestep=t,
            text_context=prompt.to(unet_dtype),
            paired_image_context=context.to(unet_dtype),
            pooled_text_embedding=pooled.to(unet_dtype),time_ids=tids,
        )
        latents=scheduler.step(
            noise,t,latents,return_dict=True,
        ).prev_sample
    rgb=(vae.decode(
        latents.to(dtype=vae.dtype)/float(vae.config.scaling_factor),
    ).sample.float()+1.)*.5
    if rgb.shape!=(b*2,3,height,width):
        raise RuntimeError("RSS_VNEXT_SDXL_STAGE_A_VAE_DECODE_SHAPE")
    alpha=alpha_head(latents,(height,width)).sigmoid()
    rgba=torch.cat((rgb.clamp(0,1),alpha),dim=1).reshape(
        b,2,4,height,width,
    )
    if not bool(torch.isfinite(rgba).all()):
        raise RuntimeError("RSS_VNEXT_SDXL_STAGE_A_GENERATED_NONFINITE")
    return PairedSDXLCanonicalPredictionV2(
        rgba=rgba,trained_checkpoint_sha256=receipt.checkpoint_sha256,
    )


__all__=[
    "SCHEMA","SDXLCanonicalTaskWeightsV2",
    "write_stage_a_task_weights_v2",
    "load_exact_stage_a_task_weights_v2",
    "PairedSDXLCanonicalPredictionV2",
    "sample_trained_stage_a_canonical_v2",
]
