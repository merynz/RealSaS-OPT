from __future__ import annotations

"""Admitted Graphics source RGB and ray RGB/alpha losses in product seeded H.

Source witnesses are LOSS ONLY. The student gets the same two original
views + pinned six generated hypotheses. Source cameras, source ray colors,
masks and source 3D meshes are never model observations. A first-hit visible
RGB label is not a proof of hidden RGB or alpha: independent source-3D ray
rendered Stage1/2 witnesses supervise transmittance, alpha and occlusion.

Stage3 requires an actual differentiable M* rasterizer and is NOT silently
approximated by a NeRF field. These losses cannot by themselves qualify
Graphics release, source visibility or native motion.

All receipts are launcher-owned and bind source asset, dataset manifest,
original observed input, generated scene checkpoint, and the exact query
tensor hashes. No loss is activated from an unregistered batch string.
"""

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor,nn
from torch.nn import functional as F

from .realsas_joint_slrm_render_v1 import (
    RealSaSJointTargetV1, STAGE_1, STAGE_2,
    render_realsas_joint_from_shared_scene_v1,
    realsas_joint_stage_loss_v1,
)


@dataclass(frozen=True)
class ProductGraphicsLossesV1:
    visible_rgb: Tensor | None
    joint_ray: Tensor | None
    visible_query_count: int
    joint_ray_count: int
    joint_parts: dict[str,Tensor] | None


def product_source_graphics_loss_v1(
    *,model:nn.Module,state,source_rgb:Tensor|None,
    source_directions:Tensor|None,source_mask:Tensor|None,
    admission_key:str,receipt_registry:Mapping[str,Mapping[str,object]],
    dataset_manifest_sha256:str,source_qualification_key:str,
    source_asset_sha256:str,original_rgba:Tensor,
    original_cameras:Tensor,generated_source_binding_sha256:str,
    generation_checkpoint_sha256:str,tensor_digest,
) -> tuple[Tensor|None,int]:
    requested=(
        any(t is not None for t in (
            source_rgb,source_directions,source_mask,
        )) or bool(admission_key)
    )
    if not requested:
        return None,0
    if (
        any(not isinstance(t,Tensor) for t in (
            source_rgb,source_directions,source_mask,
        ))
        or type(admission_key) is not str or not admission_key
        or state.scene_conditioning_authority
            != "GENERATED_SIX_VIEW_SHARED_SCENE"
        or state.camera_features is not None
    ):
        raise ValueError("RSS_VNEXT_PRODUCT_GRAPHICS_RGB_SOURCE_REQUIRED")
    h=state.local_features
    original_logits=state.graphics_logits
    b,q=h.shape[:2]
    if (
        b!=1 or q>65536
        or source_rgb.shape!=(b,q,3)
        or source_directions.shape!=(b,q,3)
        or source_mask.shape!=(b,q)
        or source_mask.dtype!=torch.bool
        or not bool(source_mask.any())
        or any(x.device!=h.device or x.dtype!=h.dtype
               for x in (source_rgb,source_directions))
        or source_mask.device!=h.device
        or original_logits.shape!=(b,q,5)
        or any(not bool(torch.isfinite(x).all())
               for x in (source_rgb,source_directions))
        or bool(((source_rgb<0)|(source_rgb>1)).any())
        or not bool(torch.allclose(
            source_directions.norm(dim=-1)[source_mask],
            torch.ones_like(
                source_directions.norm(dim=-1)[source_mask],
            ),atol=1.e-3,rtol=0,
        ))
    ):
        raise ValueError("RSS_VNEXT_PRODUCT_GRAPHICS_RGB_DOMAIN")
    receipt=receipt_registry.get(admission_key)
    expected={
        "admission_status":"QUALIFIED_SOURCE_VISIBLE_RGB_QUERIES",
        "dataset_manifest_sha256":dataset_manifest_sha256,
        "source_qualification_key":source_qualification_key,
        "source_asset_sha256":source_asset_sha256,
        "query_xyz_sha256":tensor_digest(state.query_xyz),
        "rgb_target_sha256":tensor_digest(source_rgb),
        "source_ray_direction_sha256":tensor_digest(source_directions),
        "rgb_mask_sha256":tensor_digest(source_mask),
        "original_rgba_sha256":tensor_digest(original_rgba),
        "original_camera_sha256":tensor_digest(original_cameras),
        "source_binding_sha256":generated_source_binding_sha256,
        "combined_generation_checkpoint_sha256":generation_checkpoint_sha256,
        "generated_views_as_source_truth":False,
        "source_model_observation_count":2,
        "teacher_semantics":
            "ORIGINAL_OPAQUE_FIRST_HIT_RGB_NOT_TEXTURE_OR_DEPTH",
        "visibility_mechanism":"SOURCE_ZBUFFER_FIRST_HIT_OPAQUE",
    }
    if not isinstance(receipt,Mapping) or any(
        receipt.get(k)!=v for k,v in expected.items()
    ):
        raise ValueError("RSS_VNEXT_PRODUCT_GRAPHICS_RGB_RECEIPT_DRIFT")
    latent=model.constructed_view_renderer.core(
        torch.cat((h,source_directions),dim=-1)
    )
    predicted=torch.sigmoid(
        model.constructed_view_renderer.rgb(latent)
        + original_logits[...,:3]
    )
    return F.smooth_l1_loss(
        predicted[source_mask],source_rgb[source_mask],
    ),int(source_mask.sum().item())


def product_source_joint_ray_loss_v1(
    *,model:nn.Module,state,
    ray_xyz:Tensor|None,ray_delta:Tensor|None,
    ray_depth:Tensor|None,ray_direction:Tensor|None,
    keep_roles:Tensor|None,target:RealSaSJointTargetV1|None,
    admission_key:str,receipt_registry:Mapping[str,Mapping[str,object]],
    dataset_manifest_sha256:str,source_qualification_key:str,
    source_asset_sha256:str,original_rgba:Tensor,
    original_cameras:Tensor,generated_source_binding_sha256:str,
    generation_checkpoint_sha256:str,tensor_digest,
) -> tuple[Tensor|None,int,dict[str,Tensor]|None]:
    requested=(
        bool(admission_key) or any(x is not None for x in (
            ray_xyz,ray_delta,ray_depth,ray_direction,keep_roles,target,
        ))
    )
    if not requested:
        return None,0,None
    if (
        any(not isinstance(t,Tensor) for t in (
            ray_xyz,ray_delta,ray_depth,ray_direction,keep_roles,
        ))
        or not isinstance(target,RealSaSJointTargetV1)
        or type(admission_key) is not str or not admission_key
        or state.scene_conditioning_authority
            != "GENERATED_SIX_VIEW_SHARED_SCENE"
        or state.camera_features is not None
        or target.stage not in (STAGE_1,STAGE_2)
    ):
        raise ValueError("RSS_VNEXT_PRODUCT_GRAPHICS_JOINT_SOURCE_REQUIRED")
    b,r,s,_=ray_xyz.shape if ray_xyz.ndim==4 else (0,0,0,0)
    k=model.cfg.semantic_role_schema().count
    if (
        b!=1 or r<1 or s<2 or r*s>65536
        or ray_delta.shape!=(b,r,s)
        or ray_depth.shape!=(b,r,s)
        or ray_direction.shape!=(b,r,3)
        or keep_roles.shape!=(b,r,k)
        or keep_roles.dtype!=torch.bool
        or (target.stage==STAGE_1 and not bool(keep_roles.all()))
        or (target.stage==STAGE_2 and bool(keep_roles.all()))
        or any(x.device!=state.scene_triplanes.device
               for x in (ray_xyz,ray_delta,ray_depth,
                         ray_direction,keep_roles))
        or any(not bool(torch.isfinite(x).all())
               for x in (ray_xyz,ray_delta,ray_depth,ray_direction))
    ):
        raise ValueError("RSS_VNEXT_PRODUCT_GRAPHICS_JOINT_RAY_DOMAIN")
    receipt=receipt_registry.get(admission_key)
    expected={
        "admission_status":"QUALIFIED_SOURCE_RENDERED_3D_JOINT_SCENE",
        "dataset_manifest_sha256":dataset_manifest_sha256,
        "source_qualification_key":source_qualification_key,
        "source_asset_sha256":source_asset_sha256,
        "source_binding_sha256":generated_source_binding_sha256,
        "combined_generation_checkpoint_sha256":generation_checkpoint_sha256,
        "source_model_observation_count":2,
        "generated_views_as_source_truth":False,
        "observed_rgba_sha256":tensor_digest(original_rgba),
        "observed_camera_sha256":tensor_digest(original_cameras),
        "ray_xyz_sha256":tensor_digest(ray_xyz),
        "ray_delta_sha256":tensor_digest(ray_delta),
        "ray_depth_sha256":tensor_digest(ray_depth),
        "ray_direction_sha256":tensor_digest(ray_direction),
        "keep_roles_sha256":tensor_digest(keep_roles),
        "target_rgb_sha256":tensor_digest(target.rgb_white),
        "target_alpha_sha256":tensor_digest(target.alpha),
        "target_labels_sha256":tensor_digest(target.role_indices),
        "target_rgb_mask_sha256":tensor_digest(target.qualified_rgb_mask),
        "target_alpha_mask_sha256":tensor_digest(target.qualified_mask_mask),
        "target_role_mask_sha256":tensor_digest(target.qualified_role_mask),
        "semantic_role_schema_id":model.cfg.semantic_role_schema_id,
        "training_stage":target.stage,
        "semantic_layer_name":target.layer_name,
        "source_3d_truth_binding_sha256":
            target.source_3d_truth_binding_sha256,
        "generated_diffusion_truth":False,
        "independent_source_3d_render":True,
        "drawable_or_attachment_teacher":False,
    }
    if not isinstance(receipt,Mapping) or any(
        receipt.get(k)!=v for k,v in expected.items()
    ):
        raise ValueError("RSS_VNEXT_PRODUCT_GRAPHICS_JOINT_RECEIPT_DRIFT")
    predicted=render_realsas_joint_from_shared_scene_v1(
        iris=model.iris,scene_triplanes=state.scene_triplanes,
        sample_xyz=ray_xyz,ray_delta=ray_delta,ray_depth=ray_depth,
        ray_direction=ray_direction,keep_roles=keep_roles,
        shared_graphics_head=model.graphics,
        shared_semantic_role_head=model.presentation_role_head,
    )
    parts=realsas_joint_stage_loss_v1(predicted,target)
    return parts["total"],int(
        (target.qualified_rgb_mask |
         target.qualified_mask_mask |
         target.qualified_role_mask).sum().item()
    ),parts


__all__=[
    "ProductGraphicsLossesV1",
    "product_source_graphics_loss_v1",
    "product_source_joint_ray_loss_v1",
]
