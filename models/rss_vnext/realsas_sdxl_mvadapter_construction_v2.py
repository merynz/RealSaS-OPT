from __future__ import annotations

"""Teacher-free production candidate: two source RGBA -> SDXL A -> MV B -> H(x).

This is the V2 (SDXL native _ref/_mv) execution ABI. It deliberately does NOT
alias or silently override PairedEpsilon4ChannelLatentRow.v1: checkpoint
loading and validation are distinct, and the two canonical *RGBA images*
are passed to Stage B's native reference UNet, not discarded as DINO tokens.

This function compiles no ground truth into its outputs and creates no
ProductPASS. At release time the independent Compiler can ABSTAIN.
"""

from dataclasses import dataclass
from hashlib import sha256
from typing import Any
import json

import torch
from torch import Tensor

from .unified_character_model import RSSvNextUnifiedCharacterModel
from .realsas_sdxl_stage_a_sampler_v2 import (
    SDXLCanonicalTaskWeightsV2,
    sample_trained_stage_a_canonical_v2,
)
from .realsas_mvadapter_stage_b_sampler_v2 import (
    MVAdapterPairedTaskCheckpointV2,
    sample_stage_b_paired_rgb_normal_v2,
)
from .realsas_mvadapter_camera_bridge_v2 import (
    fixed_sdxl_mvadapter_two_plus_six_cameras_v2,
)

SCHEMA="RealSaS.RSSvNext.SDXLMVAdapterPairedConstruction.v2"


def _sha(x:object)->bool:
    return type(x) is str and len(x)==64 and all(
        c in "0123456789abcdef" for c in x
    )


def two_stage_sdxl_mv_binding_v2(
    *,stage_a_checkpoint_sha256:str,stage_b_checkpoint_sha256:str,
    source_binding_sha256:str,pose_program_sha256:str,
)->str:
    parts=(
        stage_a_checkpoint_sha256,stage_b_checkpoint_sha256,
        source_binding_sha256,pose_program_sha256,
    )
    if any(not _sha(x) for x in parts):
        raise ValueError("RSS_VNEXT_SDXL_MV_V2_GENERATOR_PIN_DOMAIN")
    return sha256(json.dumps({
        "abi":SCHEMA,"stage_a":parts[0],"stage_b":parts[1],
        "source_pair":parts[2],"canonical_pose_program":parts[3],
        "six_view_bank":"RealSaS.RSSvNext.OrthographicSixTarget.v1",
    },sort_keys=True,separators=(",",":")).encode()).hexdigest()


@dataclass(frozen=True)
class TwoSourceSDXLMVAdapterConstructionV2:
    proposal:dict[str,Any]
    canonical_rgba:Tensor
    generated_rgba:Tensor
    generated_normals:Tensor
    generated_cameras:Tensor
    stage_a_task_checkpoint_sha256:str
    stage_b_task_checkpoint_sha256:str
    generation_checkpoint_sha256:str
    original_observed_view_count:int=2
    constructed_view_count:int=6
    inferred_source_camera:bool=False
    generated_is_source_truth:bool=False
    product_pass:bool=False
    schema:str=SCHEMA


@torch.no_grad()
def construct_two_source_sdxl_mv_puppet_v2(
    *,model:RSSvNextUnifiedCharacterModel,
    observed_rgba:Tensor,
    source_binding_sha256:str,
    pose_program_sha256:str,
    stage_a_unet:Any,stage_a_vae:Any,stage_a_scheduler:Any,
    stage_a_conditioner:Any,stage_a_alpha_head:Any,
    stage_a_receipt:SDXLCanonicalTaskWeightsV2,
    stage_b_pipeline:Any,stage_b_task_heads:Any,
    stage_b_receipt:MVAdapterPairedTaskCheckpointV2,
    canonical_pose_queries:Tensor,
    prompt_embeds:Tensor,pooled_text_embeds:Tensor,time_ids:Tensor,
    empty_prompt_embeds:Tensor,empty_pooled_text_embeds:Tensor,
    image_size:int,stage_a_steps:int,stage_b_steps:int,
    stage_a_seed:int,stage_b_seed:int,
    flexicubes_extractor:Any,
    flexicubes_layer:Any=None,
    flexicubes_layers:tuple|None=None,
    hybrid_open_policy:Any=None,
    generate_controls:bool=False,
)->TwoSourceSDXLMVAdapterConstructionV2:
    """No mesh, rig, skin, camera truth or extra image enters student input.

    caller loads A/B checkpoint bytes using their exact V2 loaders. The
    structural model is loaded separately by its own checkpoint contract.
    """
    if (
        type(model) is not RSSvNextUnifiedCharacterModel
        or model.training
        or observed_rgba.ndim!=5 or observed_rgba.shape[1:3]!=(2,4)
        or observed_rgba.shape[0]!=1
        or image_size!=observed_rgba.shape[-1]
        or observed_rgba.shape[-2]!=image_size
        or not bool(torch.isfinite(observed_rgba).all())
        or bool(((observed_rgba<0)|(observed_rgba>1)).any())
        or flexicubes_extractor is None
        or not _sha(source_binding_sha256)
        or not _sha(pose_program_sha256)
    ):
        raise RuntimeError("RSS_VNEXT_SDXL_MV_V2_TWO_SOURCE_STUDENT_POLICY")
    b=observed_rgba.shape[0]
    if canonical_pose_queries.shape!=(
        b,2,stage_a_conditioner.pose_dim,
    ):
        raise ValueError("RSS_VNEXT_SDXL_MV_V2_CANONICAL_POSE_QUERY_DOMAIN")
    canonical_cam,six_cam,plucker=(
        fixed_sdxl_mvadapter_two_plus_six_cameras_v2(
            b,device=str(observed_rgba.device),
            dtype=observed_rgba.dtype,image_size=image_size,
        )
    )
    raw_tokens,_xy,_early=model._observed_pixel_tokens_without_camera(
        observed_rgba,
    )
    original_valid=torch.ones(
        raw_tokens.shape[:3],dtype=torch.bool,
        device=observed_rgba.device,
    )
    canonical=sample_trained_stage_a_canonical_v2(
        unet=stage_a_unet,vae=stage_a_vae,
        scheduler=stage_a_scheduler,
        conditioner=stage_a_conditioner,
        alpha_head=stage_a_alpha_head,
        receipt=stage_a_receipt,
        observed_tokens=raw_tokens,
        observed_token_valid=original_valid,
        target_cameras=canonical_cam,
        canonical_pose_queries=canonical_pose_queries,
        prompt_embeds=prompt_embeds,
        pooled_text_embeds=pooled_text_embeds,time_ids=time_ids,
        height=image_size,width=image_size,
        steps=stage_a_steps,seed=stage_a_seed,
    )
    generated=sample_stage_b_paired_rgb_normal_v2(
        pipe=stage_b_pipeline,task_heads=stage_b_task_heads,
        receipt=stage_b_receipt,canonical_rgba=canonical.rgba,
        plucker_controls=plucker,prompt_embeds=prompt_embeds,
        pooled_text_embeds=pooled_text_embeds,time_ids=time_ids,
        empty_prompt_embeds=empty_prompt_embeds,
        empty_pooled_text_embeds=empty_pooled_text_embeds,
        steps=stage_b_steps,seed=stage_b_seed,
    )
    if (
        canonical.rgba.shape!=(b,2,4,image_size,image_size)
        or generated.rgba.shape!=(b,6,4,image_size,image_size)
        or generated.normals.shape!=(b,6,3,image_size,image_size)
        or generated.observation_authority
        or generated.source_teacher_authority
    ):
        raise RuntimeError("RSS_VNEXT_SDXL_MV_V2_GENERATOR_OUTPUT_DRIFT")
    pin=two_stage_sdxl_mv_binding_v2(
        stage_a_checkpoint_sha256=canonical.trained_checkpoint_sha256,
        stage_b_checkpoint_sha256=generated.trained_checkpoint_sha256,
        source_binding_sha256=source_binding_sha256,
        pose_program_sha256=pose_program_sha256,
    )
    # Same shared H(x), learned seeded M*, AXIS and MIRA. No native
    # teacher/rig/skin source enters this function.
    generated_kwargs={
        "generated_rgba":generated.rgba,
        "generated_cameras":six_cam,
        "generated_normals":generated.normals,
        "generation_checkpoint_sha256":pin,
        "source_binding_sha256":source_binding_sha256,
        "flexicubes_extractor":flexicubes_extractor,
        "generate_controls":generate_controls,
    }
    if flexicubes_layer is not None:
        generated_kwargs["flexicubes_layer"]=flexicubes_layer
    if flexicubes_layers is not None:
        generated_kwargs["flexicubes_layers"]=flexicubes_layers
    if hybrid_open_policy is not None:
        generated_kwargs["hybrid_open_policy"]=hybrid_open_policy
    if not generate_controls:
        # Only compiler-authorized execution may enable solver controls.
        generated_kwargs["executable_constraint_operators"]=None
    else:
        from .realsas_two_source_construction_v1 import (
            _allowed_product_constraint_operators_v1,
        )
        generated_kwargs["executable_constraint_operators"]=(
            _allowed_product_constraint_operators_v1()
        )
    proposal=model.propose_teacher_free(
        observed_rgba,None,**generated_kwargs,
    )
    state=proposal["state"]
    if (
        state.scene_conditioning_authority!="GENERATED_SIX_VIEW_SHARED_SCENE"
        or state.generated_view_count!=6
        or state.camera_features is not None
        or state.source_camera_authority
            !="UNKNOWN_NO_ORIGINAL_CAMERA_PROJECTION"
    ):
        raise RuntimeError("RSS_VNEXT_SDXL_MV_V2_SOURCE_AUTHORITY_VIOLATION")
    return TwoSourceSDXLMVAdapterConstructionV2(
        proposal=proposal,canonical_rgba=canonical.rgba,
        generated_rgba=generated.rgba,
        generated_normals=generated.normals,
        generated_cameras=six_cam,
        stage_a_task_checkpoint_sha256=canonical.trained_checkpoint_sha256,
        stage_b_task_checkpoint_sha256=generated.trained_checkpoint_sha256,
        generation_checkpoint_sha256=pin,
    )


__all__=[
    "SCHEMA","TwoSourceSDXLMVAdapterConstructionV2",
    "two_stage_sdxl_mv_binding_v2",
    "construct_two_source_sdxl_mv_puppet_v2",
]
