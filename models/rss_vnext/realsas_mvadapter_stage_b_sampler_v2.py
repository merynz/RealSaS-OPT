from __future__ import annotations

"""Versioned two-source SDXL/MV-Adapter Stage B inference and checkpoint gate.

This actually runs the same native _mv/_ref + T2IAdapter pathway as the v2
training objective. It is not an interchangeable implementation of
PairedEpsilon4ChannelLatentRow.v1. A real locally trained checkpoint and
source/normal/alpha qualification are mandatory before calling sample().
"""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import math

import torch
from torch import Tensor

from .realsas_mvadapter_dual_reference_v2 import (
    VIEWS, MVAdapterRGBNormalTaskHeadsV2,
    capture_native_two_reference_cache_v2,
    native_mv_adapter_rgb_normal_denoise_v2,
)
from .realsas_mvadapter_guidance_policy_v1 import (
    OfficialMVAdapterGuidancePolicyV1,
    zero_native_mvadapter_reference_cache_v1,
)

SCHEMA="RealSaS.RSSvNext.MVAdapterPairedTaskCheckpoint.v2"


def _sha(value:object)->bool:
    return type(value) is str and len(value)==64 and all(
        c in "0123456789abcdef" for c in value
    )


def _file_sha(path:str|Path)->str:
    digest=sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class MVAdapterPairedTaskCheckpointV2:
    checkpoint_file:str
    checkpoint_sha256:str
    pretrained_base_unet_sha256:str
    pretrained_mv_adapter_sha256:str
    training_noise_schedule_sha256:str
    guidance_policy_sha256:str
    vae_checkpoint_sha256:str
    qualified_corpus_sha256:str
    trained_steps:int
    teacher_scope:str="SOURCE_3D_RENDERED_NOT_GENERATED_TEACHER"
    source_pair_trained:bool=True
    rgb_normal_pair_trained:bool=True
    alpha_trained:bool=True
    schema:str=SCHEMA

    def validate(self)->None:
        if (
            self.schema!=SCHEMA
            or any(not _sha(v) for v in (
                self.checkpoint_sha256,self.pretrained_base_unet_sha256,
                self.pretrained_mv_adapter_sha256,
                self.training_noise_schedule_sha256,
                self.guidance_policy_sha256,
                self.vae_checkpoint_sha256,self.qualified_corpus_sha256,
            ))
            or self.teacher_scope!="SOURCE_3D_RENDERED_NOT_GENERATED_TEACHER"
            or type(self.trained_steps) is not int or self.trained_steps<=0
            or self.source_pair_trained is not True
            or self.rgb_normal_pair_trained is not True
            or self.alpha_trained is not True
            or not Path(self.checkpoint_file).is_file()
        ):
            raise RuntimeError("RSS_VNEXT_MVADAPTER_STAGE_B_RELEASE_NOT_TRAINED")
        if _file_sha(self.checkpoint_file)!=self.checkpoint_sha256:
            raise RuntimeError("RSS_VNEXT_MVADAPTER_STAGE_B_CHECKPOINT_SHA_DRIFT")


def load_exact_stage_b_task_checkpoint_v2(
    *,pipe:Any,task_heads:MVAdapterRGBNormalTaskHeadsV2,
    receipt:MVAdapterPairedTaskCheckpointV2,
) -> None:
    """Load only exact trained _mv/_ref/cond_encoder/head tensors, no pickle code."""
    receipt.validate()
    if (
        getattr(pipe.unet,"_rss_vnext_pretrained_sdxl_unet_sha256",None)
            !=receipt.pretrained_base_unet_sha256
        or getattr(pipe,"_rss_vnext_pretrained_sdxl_unet_sha256",None)
            !=receipt.pretrained_base_unet_sha256
        or getattr(pipe,"_rss_vnext_pretrained_mv_adapter_sha256",None)
            !=receipt.pretrained_mv_adapter_sha256
    ):
        raise RuntimeError("RSS_VNEXT_MVADAPTER_STAGE_B_VERIFIED_ANCESTOR_REQUIRED")
    data=torch.load(receipt.checkpoint_file,map_location="cpu",weights_only=True)
    if (
        type(data) is not dict or data.get("schema")!=SCHEMA
        or data.get("trained_steps")!=receipt.trained_steps
        or data.get("pretrained_base_unet_sha256")
            !=receipt.pretrained_base_unet_sha256
        or data.get("pretrained_mv_adapter_sha256")
            !=receipt.pretrained_mv_adapter_sha256
        or data.get("training_noise_schedule_sha256")
            !=receipt.training_noise_schedule_sha256
        or data.get("guidance_policy_sha256")!=receipt.guidance_policy_sha256
        or data.get("vae_checkpoint_sha256")!=receipt.vae_checkpoint_sha256
        or getattr(pipe.vae,"_rss_vnext_vae_ancestor_sha256",None)
            !=receipt.vae_checkpoint_sha256
        or data.get("qualified_corpus_sha256")!=receipt.qualified_corpus_sha256
        or type(data.get("mv_ref_adapter")) is not dict
        or type(data.get("cond_encoder")) is not dict
        or type(data.get("task_heads")) is not dict
    ):
        raise RuntimeError("RSS_VNEXT_MVADAPTER_STAGE_B_CHECKPOINT_CONTENT_DRIFT")
    native=dict(pipe.unet.named_parameters())
    trainable={k for k in native if "_mv" in k or "_ref" in k}
    actual=set(data["mv_ref_adapter"])
    if not trainable or trainable!=actual:
        raise RuntimeError("RSS_VNEXT_MVADAPTER_STAGE_B_ATTENTION_KEY_DRIFT")
    # This is safe if the full native module has the exact pretrained
    # parameters already installed and every task key above matches shapes.
    for name,parameter in native.items():
        if name in actual:
            tensor=data["mv_ref_adapter"][name]
            if not isinstance(tensor,Tensor) or tensor.shape!=parameter.shape:
                raise RuntimeError("RSS_VNEXT_MVADAPTER_STAGE_B_ATTENTION_SHAPE_DRIFT")
    pipe.unet.load_state_dict(data["mv_ref_adapter"],strict=False)
    pipe.cond_encoder.load_state_dict(data["cond_encoder"],strict=True)
    task_heads.load_state_dict(data["task_heads"],strict=True)
    pipe.unet.eval()
    pipe.cond_encoder.eval()
    task_heads.eval()
    # Only the exact weights-loaded pipeline is admitted for sampling.
    pipe._rss_vnext_stage_b_checkpoint_sha256=receipt.checkpoint_sha256
    task_heads._rss_vnext_stage_b_checkpoint_sha256=receipt.checkpoint_sha256


def write_stage_b_task_weights_v2(
    path:str|Path,*,pipe:Any,
    task_heads:MVAdapterRGBNormalTaskHeadsV2,
    trained_steps:int,pretrained_base_unet_sha256:str,
    pretrained_mv_adapter_sha256:str,vae_checkpoint_sha256:str,
    training_noise_schedule_sha256:str,
    guidance_policy_sha256:str,qualified_corpus_sha256:str,
)->MVAdapterPairedTaskCheckpointV2:
    """Weights-only output after REAL optimizer steps; no ModelRelease claim.

    The caller must bind an independently qualified stage-runner receipt to
    the training process. We do not take an arbitrary file 'PASS' string.
    """
    if (
        type(trained_steps) is not int or trained_steps<=0
        or any(not _sha(x) for x in (
            pretrained_base_unet_sha256,pretrained_mv_adapter_sha256,
            vae_checkpoint_sha256,training_noise_schedule_sha256,
            guidance_policy_sha256,qualified_corpus_sha256,
        ))
    ):
        raise ValueError("RSS_VNEXT_MVADAPTER_STAGE_B_UNTRAINED_SAVE_FORBIDDEN")
    native={
        name:parameter.detach().cpu().contiguous()
        for name,parameter in pipe.unet.named_parameters()
        if "_mv" in name or "_ref" in name
    }
    if not native:
        raise RuntimeError("RSS_VNEXT_MVADAPTER_STAGE_B_NO_REAL_ADAPTER")
    payload={
        "schema":SCHEMA,
        "trained_steps":trained_steps,
        "pretrained_base_unet_sha256":pretrained_base_unet_sha256,
        "pretrained_mv_adapter_sha256":pretrained_mv_adapter_sha256,
        "vae_checkpoint_sha256":vae_checkpoint_sha256,
        "training_noise_schedule_sha256":training_noise_schedule_sha256,
        "guidance_policy_sha256":guidance_policy_sha256,
        "qualified_corpus_sha256":qualified_corpus_sha256,
        "mv_ref_adapter":native,
        "cond_encoder":{
            k:v.detach().cpu().contiguous()
            for k,v in pipe.cond_encoder.state_dict().items()
        },
        "task_heads":{
            k:v.detach().cpu().contiguous()
            for k,v in task_heads.state_dict().items()
        },
    }
    dest=Path(path)
    dest.parent.mkdir(parents=True,exist_ok=True)
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(
        dir=dest.parent,prefix="rss-mvadapter-b-",
        suffix=".pt",delete=False,
    ) as file:
        tmp=Path(file.name)
    try:
        torch.save(payload,tmp)
        os.replace(tmp,dest)
    finally:
        tmp.unlink(missing_ok=True)
    return MVAdapterPairedTaskCheckpointV2(
        checkpoint_file=str(dest),
        checkpoint_sha256=_file_sha(dest),
        pretrained_base_unet_sha256=pretrained_base_unet_sha256,
        pretrained_mv_adapter_sha256=pretrained_mv_adapter_sha256,
        vae_checkpoint_sha256=vae_checkpoint_sha256,
        training_noise_schedule_sha256=training_noise_schedule_sha256,
        guidance_policy_sha256=guidance_policy_sha256,
        qualified_corpus_sha256=qualified_corpus_sha256,
        trained_steps=trained_steps,
    )


@dataclass(frozen=True)
class MVAdapterPairedGeneratedSceneV2:
    rgba:Tensor           # B,6,4,H,W, generated not observed
    normals:Tensor        # B,6,3,H,W, generated vector [-1,1]
    canonical_reference_count:int
    constructed_view_count:int
    trained_checkpoint_sha256:str
    observation_authority:bool=False
    source_teacher_authority:bool=False
    schema:str=SCHEMA


@torch.no_grad()
def sample_stage_b_paired_rgb_normal_v2(
    *,pipe:Any,task_heads:MVAdapterRGBNormalTaskHeadsV2,
    receipt:MVAdapterPairedTaskCheckpointV2,
    canonical_rgba:Tensor,plucker_controls:Tensor,
    prompt_embeds:Tensor,pooled_text_embeds:Tensor,time_ids:Tensor,
    empty_prompt_embeds:Tensor,empty_pooled_text_embeds:Tensor,
    steps:int,seed:int,
) -> MVAdapterPairedGeneratedSceneV2:
    """Native paired MV-Adapter inference, with separately learned alpha.

    Six RGB and six normal maps share identical camera controls. No source
    ground-truth mesh/rays/poses enter this function. The outputs remain
    hypotheses, never native teachers.
    """
    receipt.validate()
    guidance=OfficialMVAdapterGuidancePolicyV1()
    if receipt.guidance_policy_sha256!=guidance.digest:
        raise RuntimeError("RSS_VNEXT_MVADAPTER_STAGE_B_GUIDANCE_POLICY_DRIFT")
    from .realsas_mvadapter_shift_snr_training_v2 import (
        schedule_alphas_sha256_v1,
    )
    if schedule_alphas_sha256_v1(pipe.scheduler)!=receipt.training_noise_schedule_sha256:
        raise RuntimeError("RSS_VNEXT_MVADAPTER_STAGE_B_INFERENCE_NOISE_SCHEDULE_DRIFT")
    if (
        getattr(pipe.vae,"_rss_vnext_vae_ancestor_sha256",None)
            !=receipt.vae_checkpoint_sha256
        or getattr(pipe,"_rss_vnext_stage_b_checkpoint_sha256",None)
        !=receipt.checkpoint_sha256
        or getattr(task_heads,"_rss_vnext_stage_b_checkpoint_sha256",None)
        !=receipt.checkpoint_sha256
        or pipe.unet.training or pipe.cond_encoder.training
        or task_heads.training
    ):
        raise RuntimeError("RSS_VNEXT_MVADAPTER_STAGE_B_EXACT_TASK_WEIGHTS_NOT_LOADED")
    if (
        canonical_rgba.ndim!=5 or canonical_rgba.shape[1:3]!=(2,4)
        or not 1<=steps<=100 or not 0<=seed<2**63
        or not bool(torch.isfinite(canonical_rgba).all())
        or bool(((canonical_rgba<0)|(canonical_rgba>1)).any())
    ):
        raise ValueError("RSS_VNEXT_MVADAPTER_STAGE_B_INFERENCE_SOURCE")
    b=canonical_rgba.shape[0]
    h,w=canonical_rgba.shape[-2:]
    if (
        h%8 or w%8
        or plucker_controls.shape!=(b,6,6,h,w)
        or prompt_embeds.shape[0]!=b
        or pooled_text_embeds.shape!=(b,1280)
        or empty_prompt_embeds.shape!=prompt_embeds.shape
        or empty_pooled_text_embeds.shape!=(b,1280)
        or time_ids.shape!=(b,6)
    ):
        raise ValueError("RSS_VNEXT_MVADAPTER_STAGE_B_INFERENCE_CAMERA_ABI")
    # The official source uses gray compositing and aspect-preserving
    # reference images; callers of this tensor ABI supply normalized square
    # canonical images from the already trained A stage.
    canon=canonical_rgba
    reference_rgb=canon[:,:,:3]*canon[:,:,3:4]+.5*(1.-canon[:,:,3:4])
    vae=pipe.vae
    with torch.no_grad():
        ref=vae.encode(
            reference_rgb.reshape(b*2,3,h,w).to(vae.dtype)*2.-1.
        ).latent_dist.mode()*float(vae.config.scaling_factor)
    cache=capture_native_two_reference_cache_v2(
        unet=pipe.unet,canonical_latents=ref.reshape(b,2,4,h//8,w//8),
        prompt_embeds=prompt_embeds,pooled_text_embeds=pooled_text_embeds,
        time_ids=time_ids,modalities=2,
    )
    generator=torch.Generator(device=canonical_rgba.device).manual_seed(seed)
    latent=torch.randn(
        (b,2,6,4,h//8,w//8),
        generator=generator,device=canonical_rgba.device,
        dtype=next(pipe.unet.parameters()).dtype,
    )
    pipe.scheduler.set_timesteps(steps,device=canonical_rgba.device)
    latent*=float(pipe.scheduler.init_noise_sigma)
    controls=plucker_controls[:,None].expand(-1,2,-1,-1,-1,-1)
    no_reference_cache=zero_native_mvadapter_reference_cache_v1(cache)
    for timestep in pipe.scheduler.timesteps:
        input_latents=pipe.scheduler.scale_model_input(
            latent.reshape(b*12,4,h//8,w//8),timestep,
        ).reshape_as(latent)
        t=torch.full(
            (b,2,6),int(timestep),device=canonical_rgba.device,
            dtype=torch.long,
        )
        # Exact upstream classifier-free guidance: unconditional text is
        # the frozen SDXL EMPTY prompt embedding, and reference K/V is
        # separately zeroed. No invented zero token or random image prior.
        negative=native_mv_adapter_rgb_normal_denoise_v2(
            pipe=pipe,task_heads=task_heads,
            noisy_latents=input_latents,timesteps=t,
            cache=no_reference_cache,prompt_embeds=empty_prompt_embeds,
            pooled_text_embeds=empty_pooled_text_embeds,time_ids=time_ids,
            control_plucker=controls,
        )
        conditional=native_mv_adapter_rgb_normal_denoise_v2(
            pipe=pipe,task_heads=task_heads,
            noisy_latents=input_latents,timesteps=t,
            cache=cache,prompt_embeds=prompt_embeds,
            pooled_text_embeds=pooled_text_embeds,time_ids=time_ids,
            control_plucker=controls,
        )
        prediction=negative+guidance.inference_guidance_scale*(
            conditional-negative
        )
        next_latents=pipe.scheduler.step(
            prediction.reshape(b*12,4,h//8,w//8),
            timestep,latent.reshape(b*12,4,h//8,w//8),
            return_dict=True,
        ).prev_sample
        latent=next_latents.reshape_as(latent)
    decoded=vae.decode(
        (latent/float(vae.config.scaling_factor)).reshape(
            b*12,4,h//8,w//8,
        ).to(dtype=vae.dtype),
        return_dict=True,
    ).sample.float().reshape(b,2,6,3,h,w)
    images=(decoded+1.)*.5
    rgb=images[:,0].clamp(0,1)
    normal=images[:,1].clamp(0,1)*2.-1.
    normal=normal/normal.norm(dim=2,keepdim=True).clamp_min(1.e-6)
    alpha=task_heads.alpha_logits(
        latent[:,0].reshape(b*6,4,h//8,w//8),(h,w),
    ).sigmoid().reshape(b,6,1,h,w)
    rgba=torch.cat((rgb,alpha),dim=2)
    if not bool(torch.isfinite(rgba).all()) or not bool(torch.isfinite(normal).all()):
        raise RuntimeError("RSS_VNEXT_MVADAPTER_STAGE_B_GENERATED_SCENE_NONFINITE")
    return MVAdapterPairedGeneratedSceneV2(
        rgba=rgba,normals=normal,
        canonical_reference_count=2,constructed_view_count=VIEWS,
        trained_checkpoint_sha256=receipt.checkpoint_sha256,
    )


__all__=[
    "SCHEMA","MVAdapterPairedTaskCheckpointV2",
    "load_exact_stage_b_task_checkpoint_v2",
    "write_stage_b_task_weights_v2",
    "MVAdapterPairedGeneratedSceneV2",
    "sample_stage_b_paired_rgb_normal_v2",
]
