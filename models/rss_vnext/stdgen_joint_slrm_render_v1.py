from __future__ import annotations

"""StdGEN CVPR 2025 §3.2 clean-room joint S-LRM ray rendering and losses.

The *same* 3D H(x) is decoded into geometry-density, RGB, and semantic
probabilities. Outer-to-inner training selectively removes covering semantic
layers by alpha reweighting, NOT by making a second hidden-surface scene.
This is original implementation of equations (4)-(9) with ray-grid inputs.

This is NOT evidence that hidden surfaces exist in an unseen input:
  - targets must be independently rendered from admitted 3D sources;
  - ray xyz/cameras are deterministic rendering coordinates, not source XYZ;
  - diffuse generated RGB/normal cannot be training truth;
  - Stage3 normals/depth require a *separate differentiable mesh renderer*
    on semantic-equivalent SDF + FlexiCubes, never this NeRF proxy.

Semantic roles are anonymous/morphology-generic BUSTED presentation roles,
not an invented rigid limb/attachment/drawable ontology.
"""

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

STAGE_1 = "STDGEN_SLRM_NERF_SINGLE_LAYER"
STAGE_2 = "STDGEN_SLRM_NERF_OUTSIDE_IN_MULTILAYER"
STAGE_3 = "STDGEN_SLRM_FLEXICUBES_MULTILAYER"
_SOURCE_TRUTH = "ADMITTED_SOURCE_3D_RENDER_NOT_GENERATED_COMPLETION"


@dataclass(frozen=True)
class StdGENJointRayOutputV1:
    rgb_premultiplied: Tensor   # [B,R,3] from shared scene
    rgb_white_composite: Tensor # [B,R,3], expected white background
    alpha: Tensor               # [B,R,1] same geometry/semantic weights
    semantic_mass: Tensor       # [B,R,K], same geometry weights
    semantic_posterior: Tensor  # [B,R,K], normalized conditioned on opacity
    expected_depth: Tensor      # [B,R,1] conditional on hit, NOT source depth
    weights: Tensor             # [B,R,S], for cross-head gradients
    authority: str = "PREDICTED_SHARED_TRIPLANE_JOINT_FIELDS_NOT_SOURCE_TRUTH"


def _fields(
    *, density_logit: Tensor, rgb_logit: Tensor, role_logits: Tensor,
    ray_delta: Tensor, ray_depth: Tensor, inside_cube: Tensor,
    keep_roles: Tensor,
) -> tuple[int,int,int,int]:
    if density_logit.ndim!=3:
        raise ValueError("RSS_VNEXT_STDGEN_JOINT_DENSITY_RANK")
    b,r,s=density_logit.shape
    if (
        b<1 or r<1 or s<2 or r*s>65536
        or rgb_logit.shape!=(b,r,s,3)
        or role_logits.ndim!=4 or role_logits.shape[:3]!=(b,r,s)
        or role_logits.shape[-1]<2 or role_logits.shape[-1]>16
        or ray_delta.shape!=(b,r,s) or ray_depth.shape!=(b,r,s)
        or inside_cube.shape!=(b,r,s) or inside_cube.dtype!=torch.bool
        or keep_roles.shape!=(b,r,role_logits.shape[-1])
        or keep_roles.dtype!=torch.bool
        or not bool(keep_roles.any(dim=-1).all())
        or not bool(torch.isfinite(ray_delta).all())
        or bool((ray_delta<=0).any())
        or not bool(torch.isfinite(ray_depth).all())
        or bool((ray_depth[...,1:]-ray_depth[...,:-1]<0).any())
    ):
        raise ValueError("RSS_VNEXT_STDGEN_JOINT_RAY_GEOMETRY_OR_ROLE_DOMAIN")
    if not all(
        bool(torch.isfinite(t).all()) for t in
        (density_logit,rgb_logit,role_logits)
    ):
        raise ValueError("RSS_VNEXT_STDGEN_JOINT_FIELD_NONFINITE")
    return b,r,s,role_logits.shape[-1]


def render_stdgen_joint_rays_v1(
    *, density_logit: Tensor, rgb_logit: Tensor, role_logits: Tensor,
    ray_delta: Tensor, ray_depth: Tensor, inside_cube: Tensor,
    keep_roles: Tensor,
) -> StdGENJointRayOutputV1:
    """Joint Eq(4,6,7,8): transmittance uses per-semantic keep mass.

    Stage1: keep all K roles. Stage2: selected outer layers removed and
    back surfaces become visible along the SAME scene rays. Geometry,
    appearance, and semantics get coupled derivatives from every layer.
    """
    b,r,s,k=_fields(
        density_logit=density_logit,rgb_logit=rgb_logit,
        role_logits=role_logits,ray_delta=ray_delta,
        ray_depth=ray_depth,inside_cube=inside_cube,keep_roles=keep_roles,
    )
    dtype=density_logit.dtype
    # Positive NeRF density is a geometry head conditioned by shared H(x).
    density=F.softplus(density_logit.float())
    geom_alpha=-torch.expm1(-density*ray_delta.float())
    geometry_mask=inside_cube.float()
    sem=F.softmax(role_logits.float(),dim=-1)
    role_mass=(sem*keep_roles[:, :, None].float()).sum(dim=-1)
    effective_alpha=geom_alpha*geometry_mask*role_mass
    survival=torch.cat((
        torch.ones_like(effective_alpha[...,:1]),
        (1.-effective_alpha+1.e-10).cumprod(dim=-1)[...,:-1],
    ),dim=-1)
    weights=effective_alpha*survival
    alpha=weights.sum(-1,keepdim=True).clamp(0.,1.)
    rgb=(torch.sigmoid(rgb_logit.float())*weights[...,None]).sum(dim=-2)
    # StdGEN Eq(7) uses the *original* semantic distribution p_i,
    # weighted by the peeled transmittance T_P and alpha*P_keep.
    # Do not silently re-normalize roles at samples; that would change
    # the paper's volume semantic objective and conceal uncertainty.
    sem_mass=(sem*weights[...,None]).sum(-2)
    posterior=sem_mass/alpha.clamp_min(1.e-8)
    posterior=posterior/posterior.sum(-1,keepdim=True).clamp_min(1.e-8)
    expected=(weights*ray_depth.float()).sum(-1,keepdim=True)/alpha.clamp_min(1.e-8)
    return StdGENJointRayOutputV1(
        rgb_premultiplied=rgb.to(dtype),
        rgb_white_composite=(rgb+1.-alpha).clamp(0.,1.).to(dtype),
        alpha=alpha.to(dtype),semantic_mass=sem_mass.to(dtype),
        semantic_posterior=posterior.to(dtype),
        expected_depth=expected.to(dtype),weights=weights.to(dtype),
    )


def render_stdgen_joint_from_shared_scene_v1(
    *, iris: nn.Module, scene_triplanes: Tensor,
    sample_xyz: Tensor, ray_delta: Tensor, ray_depth: Tensor,
    ray_direction: Tensor, keep_roles: Tensor,
    shared_graphics_head: nn.Module, shared_semantic_role_head: nn.Module,
) -> StdGENJointRayOutputV1:
    """Live model path, not synthetic logits: query the exact IRIS H(x).

    Renders geometric occupancy/opacity, Graphics RGB and semantic class
    labels from one common neural scene and camera ray. No extra observation
    or teacher mesh is exposed to the model.
    """
    if sample_xyz.ndim!=4 or sample_xyz.shape[-1]!=3:
        raise ValueError("RSS_VNEXT_STDGEN_JOINT_SAMPLE_XYZ_SHAPE")
    b,r,s,_=sample_xyz.shape
    if (
        ray_direction.shape!=(b,r,3)
        or scene_triplanes.shape[0]!=b
        or not bool(torch.isfinite(sample_xyz).all())
        or not bool(torch.isfinite(ray_direction).all())
    ):
        raise ValueError("RSS_VNEXT_STDGEN_JOINT_SCENE_CAMERA_DOMAIN")
    h=iris.query(scene_triplanes,sample_xyz.reshape(b,r*s,3))["H"]
    graphics=shared_graphics_head(h).reshape(b,r,s,-1)
    semantic=shared_semantic_role_head(h).reshape(b,r,s,-1)
    if graphics.shape[-1]!=5:
        raise ValueError("RSS_VNEXT_STDGEN_JOINT_GRAPHICS_FIVE_CHANNELS")
    if semantic.shape[-1]!=keep_roles.shape[-1]:
        raise ValueError("RSS_VNEXT_STDGEN_JOINT_SEMANTIC_ROLE_WIDTH")
    # Geometry-density logit, RGB and semantic role logits are ALL queried
    # from the same spatial H(x); a ray direction is an auxiliary texture
    # modifier, not a new source ray / geometry teacher.
    return render_stdgen_joint_rays_v1(
        density_logit=graphics[...,3],
        rgb_logit=graphics[...,:3],
        role_logits=semantic,
        ray_delta=ray_delta,ray_depth=ray_depth,
        inside_cube=(sample_xyz.abs()<=1.).all(-1),
        keep_roles=keep_roles,
    )


@dataclass(frozen=True)
class StdGENJointTargetV1:
    rgb_white: Tensor              # [B,R,3], independently rendered
    alpha: Tensor                  # [B,R,1], source-layer coverage
    role_indices: Tensor           # [B,R], signed int, -1 means unlabeled
    qualified_rgb_mask: Tensor     # [B,R] bool
    qualified_mask_mask: Tensor    # [B,R] bool
    qualified_role_mask: Tensor    # [B,R] bool
    source_3d_truth_binding_sha256: str
    layer_name: str
    stage: str
    teacher_authority: str = _SOURCE_TRUTH
    generated_completion_used_as_truth: bool = False


@dataclass(frozen=True)
class StdGENFlexiCubesLayerRasterV1:
    """Actual differentiable single-layer mesh RASTER output, not NeRF rays.

    This output must be produced from the SAME H(x) via qualified
    semantic-equivalent signed SDF + pinned FlexiCubes + differentiable
    2D triangle rasterizer. It NEVER proves a source mesh or observed depth.
    """
    rgb_white_composite: Tensor
    alpha: Tensor
    semantic_posterior: Tensor
    normal_xyz: Tensor
    depth: Tensor
    flexicubes_deformation_regularizer: Tensor
    source_scene_prediction_sha256: str
    mesh_surface_prediction_sha256: str
    semantic_layer_name: str
    rasterizer_identity: str
    authority: str = "PREDICTED_FLEXICUBES_MESH_RASTER_NOT_SOURCE_OBSERVATION"


def stdgen_joint_stage_loss_v1(
    predicted: StdGENJointRayOutputV1 | None, target: StdGENJointTargetV1,
    *,
    lpips_loss_fn: Callable[[Tensor,Tensor],Tensor] | None = None,
    mesh_raster: StdGENFlexiCubesLayerRasterV1 | None = None,
    target_normal: Tensor | None = None,
    target_depth: Tensor | None = None,
    qualified_normal_mask: Tensor | None = None,
    qualified_depth_mask: Tensor | None = None,
) -> dict[str,Tensor]:
    """Paper §3.2 L1,L2,L3; stage3 uses REAL mesh renderer outputs.

    LPIPS is a separate optional term; it is *never silently approximated*
    by RGB MSE. Stage3 normals/depth/deform only valid with mesh renderer,
    not NeRF expected-depth or predicted diffusion normals as source truth.
    """
    if (
        type(target) is not StdGENJointTargetV1
        or target.teacher_authority!=_SOURCE_TRUTH
        or target.generated_completion_used_as_truth is not False
        or type(target.source_3d_truth_binding_sha256) is not str
        or len(target.source_3d_truth_binding_sha256)!=64
        or any(ch not in "0123456789abcdef" for ch in target.source_3d_truth_binding_sha256)
        or not target.layer_name
        or target.stage not in (STAGE_1,STAGE_2,STAGE_3)
    ):
        raise ValueError("RSS_VNEXT_STDGEN_JOINT_TEACHER_NOT_ADMITTED_3D_RENDER")
    if target.stage==STAGE_3:
        if mesh_raster is None:
            raise ValueError("RSS_VNEXT_STDGEN_JOINT_STAGE3_REQUIRES_FLEXICUBES_MESH_RENDER")
        b,r,_=mesh_raster.rgb_white_composite.shape
        k=mesh_raster.semantic_posterior.shape[-1]
    else:
        if type(predicted) is not StdGENJointRayOutputV1:
            raise ValueError("RSS_VNEXT_STDGEN_JOINT_NERF_STAGE_RAYS_REQUIRED")
        b,r,_=predicted.rgb_white_composite.shape
        k=predicted.semantic_mass.shape[-1]
    if target.stage==STAGE_3:
        if (
            type(mesh_raster) is not StdGENFlexiCubesLayerRasterV1
            or mesh_raster.authority != "PREDICTED_FLEXICUBES_MESH_RASTER_NOT_SOURCE_OBSERVATION"
            or mesh_raster.semantic_layer_name != target.layer_name
            or len(mesh_raster.source_scene_prediction_sha256)!=64
            or len(mesh_raster.mesh_surface_prediction_sha256)!=64
            or not mesh_raster.rasterizer_identity
        ):
            raise ValueError("RSS_VNEXT_STDGEN_JOINT_STAGE3_REQUIRES_FLEXICUBES_MESH_RENDER")
        # Stage3 must use MESH-rendered RGB, mask and semantic maps, not
        # the Stage2 NeRF predictions plus a substituted mesh depth/normal.
        active_rgb=mesh_raster.rgb_white_composite
        active_alpha=mesh_raster.alpha
        active_sem=mesh_raster.semantic_posterior
    else:
        if mesh_raster is not None:
            raise ValueError("RSS_VNEXT_STDGEN_JOINT_MESH_TARGET_ON_NERF_STAGE")
        active_rgb=predicted.rgb_white_composite
        active_alpha=predicted.alpha
        active_sem=predicted.semantic_posterior
    if (
        active_rgb.shape!=(b,r,3) or active_alpha.shape!=(b,r,1)
        or active_sem.shape!=(b,r,k)
        or not all(bool(torch.isfinite(x).all()) for x in (
            active_rgb,active_alpha,active_sem,
        ))
        or bool((active_alpha<0).any() or (active_alpha>1).any())
        or bool((active_sem<0).any() or (active_sem>1).any())
    ):
        raise ValueError("RSS_VNEXT_STDGEN_JOINT_ACTIVE_STAGE_RENDER_DOMAIN")
    maskfields=(target.qualified_rgb_mask,target.qualified_mask_mask,target.qualified_role_mask)
    if (
        target.rgb_white.shape!=(b,r,3)
        or target.alpha.shape!=(b,r,1)
        or target.role_indices.shape!=(b,r) or target.role_indices.dtype!=torch.long
        or any(x.shape!=(b,r) or x.dtype!=torch.bool for x in maskfields)
        or not all(bool(torch.isfinite(x).all()) for x in (
            target.rgb_white,target.alpha,
            active_rgb,active_alpha,active_sem,
        ))
        or bool((target.rgb_white<0).any() or (target.rgb_white>1).any())
        or bool((target.alpha<0).any() or (target.alpha>1).any())
    ):
        raise ValueError("RSS_VNEXT_STDGEN_JOINT_TARGET_DOMAIN")
    if not bool(any(x.any() for x in maskfields)):
        raise ValueError("RSS_VNEXT_STDGEN_JOINT_NO_TARGET_PIXELS")
    zero=active_alpha.sum()*0.
    rgb_mse=(
        F.mse_loss(active_rgb[target.qualified_rgb_mask].float(),
                   target.rgb_white[target.qualified_rgb_mask].float())
        if bool(target.qualified_rgb_mask.any()) else zero
    )
    mask_mse=(
        F.mse_loss(active_alpha[...,0][target.qualified_mask_mask].float(),
                   target.alpha[...,0][target.qualified_mask_mask].float())
        if bool(target.qualified_mask_mask.any()) else zero
    )
    labelled=target.qualified_role_mask
    if bool(labelled.any()):
        y=target.role_indices[labelled]
        if bool(((y<0)|(y>=k)).any()):
            raise ValueError("RSS_VNEXT_STDGEN_JOINT_SEMANTIC_LABEL_INVALID")
        sem_prob=active_sem[labelled].float()
        semantic_ce=F.nll_loss(sem_prob.clamp_min(1.e-8).log(),y)
    else:
        semantic_ce=zero
    terms={"rgb_mse":rgb_mse,"mask_mse":mask_mse,"semantic_ce":semantic_ce}
    if lpips_loss_fn is not None:
        # LPIPS applies to image maps; scalar-ray microbatches cannot
        # invent a 2D layout. Caller must provide a conforming image layout.
        if int(r**.5)**2!=r:
            raise ValueError("RSS_VNEXT_STDGEN_JOINT_LPIPS_REQUIRES_SQUARE_RAY_GRID")
        hw=int(r**.5)
        a=active_rgb.permute(0,2,1).reshape(b,3,hw,hw)
        z=target.rgb_white.permute(0,2,1).reshape(b,3,hw,hw)
        terms["lpips"]=lpips_loss_fn(a*2.-1.,z*2.-1.)
    if target.stage==STAGE_3:
        if any(x is None for x in (
            target_normal,target_depth,qualified_normal_mask,qualified_depth_mask,
        )):
            raise ValueError("RSS_VNEXT_STDGEN_JOINT_STAGE3_REQUIRES_FLEXICUBES_MESH_RENDER")
        if (
            mesh_raster.normal_xyz.shape!=(b,r,3)
            or mesh_raster.depth.shape!=(b,r,1)
            or target_normal.shape!=(b,r,3)
            or target_depth.shape!=(b,r,1)
            or qualified_normal_mask.shape!=(b,r)
            or qualified_depth_mask.shape!=(b,r)
            or qualified_normal_mask.dtype!=torch.bool
            or qualified_depth_mask.dtype!=torch.bool
            or mesh_raster.flexicubes_deformation_regularizer.ndim!=0
            or not all(bool(torch.isfinite(v).all()) for v in (
                mesh_raster.normal_xyz,mesh_raster.depth,
                target_normal,target_depth,
                mesh_raster.flexicubes_deformation_regularizer,
            ))
        ):
            raise ValueError("RSS_VNEXT_STDGEN_JOINT_MESH_TARGET_DOMAIN")
        if not bool(qualified_normal_mask.any()) or not bool(qualified_depth_mask.any()):
            raise ValueError("RSS_VNEXT_STDGEN_JOINT_NO_MESH_SUPERVISION")
        n=F.normalize(mesh_raster.normal_xyz[qualified_normal_mask].float(),dim=-1)
        truth=F.normalize(target_normal[qualified_normal_mask].float(),dim=-1)
        terms["normal_cosine"]=(1.-(n*truth).sum(-1)).mean()
        terms["depth_l1"]=F.l1_loss(
            mesh_raster.depth[...,0][qualified_depth_mask].float(),
            target_depth[...,0][qualified_depth_mask].float(),
        )
        terms["flexicubes_deformation"]=mesh_raster.flexicubes_deformation_regularizer
    elif any(x is not None for x in (
        target_normal,target_depth,qualified_normal_mask,qualified_depth_mask,
    )):
        raise ValueError("RSS_VNEXT_STDGEN_JOINT_MESH_TARGET_ON_NERF_STAGE")
    terms["total"]=sum(terms.values())
    if not bool(torch.isfinite(terms["total"])):
        raise ValueError("RSS_VNEXT_STDGEN_JOINT_OBJECTIVE_NONFINITE")
    return terms


__all__=[
    "STAGE_1","STAGE_2","STAGE_3",
    "StdGENJointRayOutputV1","StdGENJointTargetV1",
    "StdGENFlexiCubesLayerRasterV1",
    "render_stdgen_joint_rays_v1",
    "render_stdgen_joint_from_shared_scene_v1",
    "stdgen_joint_stage_loss_v1",
]
