from __future__ import annotations

"""Independent inside-out mesh refinement, with CVPR 2025 §3.3 as reference.

Unlike S-LRM Stage3 training, this is an inference-time, per-character
differentiable mesh optimization. The GENERATED six-view normal/mask maps
are guidance only (NEVER source-truth). Each inner layer is finalized and
held fixed while an outer semantic layer is refined. A nearest-inner
vertex penetration loss avoids pushing the outer surface THROUGH the
frozen inner normal.

No anatomy is hard-coded: the compiler/representation policy chooses
layer order; this optimization does not infer joint or drawable identity.
The referenced paper additionally employs edge collapse/split/flip remeshing,
high-resolution backprojection coloring and ExplicitTarget optimization;
these remain independent operations and are NOT silently approximated.
"""

from dataclasses import dataclass,replace
from hashlib import sha256
import math
from typing import Callable,Sequence

import torch
from torch import Tensor,nn
import torch.nn.functional as F

from .realsas_joint_slrm_render_v1 import RealSaSFlexiCubesLayerRasterV1
from .realsas_flexicubes_mesh_render_v1 import render_realsas_layered_flexicubes_mesh_v1
from .flexicubes_surface_evidence import RSSvNextFlexiCubesSurfaceEvidence


def _sha(v:object)->bool:
    return type(v) is str and len(v)==64 and all(c in "0123456789abcdef" for c in v)


@dataclass(frozen=True)
class RealSaSGeneratedMeshGuidanceV1:
    expected_mask:Tensor        # [1,R,1] 0..1
    expected_normal_world:Tensor # [1,R,3] normalized in world frame
    known_mask:Tensor           # [1,R] bool
    normal_valid_mask:Tensor    # [1,R] bool from GENERATED normal confidence
    generation_checkpoint_sha256:str
    original_source_binding_sha256:str
    normal_frame:str="WORLD_XYZ_FROM_GENERATED_VIEW_FRAME"
    authority:str="GENERATED_MULTIVIEW_REFINEMENT_PRIOR_NOT_SOURCE_TRUTH"

    def validate(self)->None:
        if (
            not _sha(self.generation_checkpoint_sha256)
            or not _sha(self.original_source_binding_sha256)
            or self.normal_frame!="WORLD_XYZ_FROM_GENERATED_VIEW_FRAME"
            or self.authority!="GENERATED_MULTIVIEW_REFINEMENT_PRIOR_NOT_SOURCE_TRUTH"
            or self.expected_mask.ndim!=3 or self.expected_mask.shape[0]!=1
            or self.expected_mask.shape[-1]!=1
        ):
            raise ValueError("RSS_VNEXT_REALSAS_REFINEMENT_GENERATED_PRIOR_REQUIRED")
        r=self.expected_mask.shape[1]
        if (
            r<1 or r>6*1024*1024
            or self.expected_normal_world.shape!=(1,r,3)
            or self.known_mask.shape!=(1,r) or self.known_mask.dtype!=torch.bool
            or self.normal_valid_mask.shape!=(1,r)
            or self.normal_valid_mask.dtype!=torch.bool
            or not bool(self.known_mask.any())
            or not bool(self.normal_valid_mask.any())
            or not bool(torch.all(self.normal_valid_mask<=self.known_mask))
            or not all(bool(torch.isfinite(v).all()) for v in (
                self.expected_mask,self.expected_normal_world,
            ))
            or bool((self.expected_mask<0).any())
            or bool((self.expected_mask>1).any())
        ):
            raise ValueError("RSS_VNEXT_REALSAS_REFINEMENT_GUIDANCE_DOMAIN")
        lengths=self.expected_normal_world[self.normal_valid_mask].norm(dim=-1)
        if not bool(torch.allclose(lengths,torch.ones_like(lengths),atol=1.e-3,rtol=0)):
            raise ValueError("RSS_VNEXT_REALSAS_REFINEMENT_WORLD_NORMALS_NOT_UNIT")


def realsas_outer_inner_collision_loss_v1(
    outer_xyz:Tensor,inner_xyz:Tensor,inner_normal_xyz:Tensor,
    *,outer_chunk:int=256,max_pairwise_elements:int=4_000_000,
)->Tensor:
    """Exact RealSaS Eq13: mean relu((v_inner-v_outer) dot n_inner)^3."""
    if (
        outer_xyz.ndim!=2 or outer_xyz.shape[1]!=3
        or inner_xyz.ndim!=2 or inner_xyz.shape[1]!=3
        or inner_normal_xyz.shape!=inner_xyz.shape
        or outer_xyz.shape[0]<1 or inner_xyz.shape[0]<1
        or outer_xyz.shape[0]>250_000 or inner_xyz.shape[0]>250_000
        or outer_chunk<1 or outer_chunk>4096
        or max_pairwise_elements<1
        or any(x.device!=outer_xyz.device for x in (inner_xyz,inner_normal_xyz))
        or not all(bool(torch.isfinite(x).all()) for x in (
            outer_xyz,inner_xyz,inner_normal_xyz,
        ))
        or not bool(torch.allclose(
            inner_normal_xyz.norm(dim=-1),
            torch.ones_like(inner_normal_xyz[:,0]),atol=1.e-3,rtol=0.,
        ))
    ):
        raise ValueError("RSS_VNEXT_REALSAS_REFINEMENT_COLLISION_MESH_DOMAIN")
    inner=inner_xyz.detach()
    normals=inner_normal_xyz.detach()
    if inner.shape[0]>max_pairwise_elements:
        raise ValueError("RSS_VNEXT_REALSAS_REFINEMENT_COLLISION_RESOURCE_BUDGET")
    size=min(outer_chunk,max_pairwise_elements//inner.shape[0])
    if size<1:
        raise ValueError("RSS_VNEXT_REALSAS_REFINEMENT_COLLISION_RESOURCE_BUDGET")
    losses=[]
    for start in range(0,len(outer_xyz),size):
        v=outer_xyz[start:start+size]
        idx=torch.cdist(v[None].float(),inner[None].float())[0].argmin(dim=-1)
        neighbor=inner[idx]
        n=normals[idx]
        gap=((neighbor-v)*n).sum(-1)
        losses.append(gap.clamp_min(0).pow(3))
    return torch.cat(losses).mean()


def realsas_generated_refinement_objective_v1(
    *,raster:RealSaSFlexiCubesLayerRasterV1,
    guidance:RealSaSGeneratedMeshGuidanceV1,
    current_outer_xyz:Tensor,
    fixed_inner_xyz:Tensor|None=None,
    fixed_inner_normals:Tensor|None=None,
    mask_weight:float=1.,
    normal_weight:float=1.,
    collision_weight:float=1.,
)->dict[str,Tensor]:
    """Paper Eq12+13, no diffusion generated maps become source targets."""
    guidance.validate()
    if (
        type(raster) is not RealSaSFlexiCubesLayerRasterV1
        or raster.authority!="PREDICTED_FLEXICUBES_MESH_RASTER_NOT_SOURCE_OBSERVATION"
        or raster.rgb_white_composite.shape[:2]!=guidance.expected_mask.shape[:2]
        or raster.alpha.shape!=guidance.expected_mask.shape
        or raster.normal_xyz.shape!=guidance.expected_normal_world.shape
        or not all(bool(torch.isfinite(x).all()) for x in (
            raster.alpha,raster.normal_xyz,current_outer_xyz,
        ))
        or any(not isinstance(w,(int,float)) or not math.isfinite(w) or w<0 for w in (
            mask_weight,normal_weight,collision_weight,
        ))
        or (fixed_inner_xyz is None)!=(fixed_inner_normals is None)
    ):
        raise ValueError("RSS_VNEXT_REALSAS_REFINEMENT_RENDER_OR_POLICY_CONTRACT")
    zero=raster.alpha.sum()*0.
    mask_loss=F.mse_loss(
        raster.alpha[guidance.known_mask].float(),
        guidance.expected_mask[guidance.known_mask].float(),
    )
    norm_pred=F.normalize(raster.normal_xyz[guidance.normal_valid_mask].float(),dim=-1)
    norm_target=guidance.expected_normal_world[guidance.normal_valid_mask].float()
    normal_loss=(norm_pred-norm_target).square().sum(-1).mean()
    collision=(
        realsas_outer_inner_collision_loss_v1(
            current_outer_xyz,fixed_inner_xyz,fixed_inner_normals,
        ) if fixed_inner_xyz is not None else zero
    )
    total=mask_weight*mask_loss+normal_weight*normal_loss+collision_weight*collision
    if not bool(torch.isfinite(total)):
        raise ValueError("RSS_VNEXT_REALSAS_REFINEMENT_OBJECTIVE_NONFINITE")
    return {"mask_mse":mask_loss,"normal_l2":normal_loss,
            "outer_inner_penetration":collision,"total":total}


@dataclass(frozen=True)
class RealSaSRefinedLayerV1:
    layer_name:str
    refined_xyz:Tensor
    face_indices:Tensor
    source_binding_sha256:str
    guidance_checkpoint_sha256:str
    optimizer_steps:int
    final_loss:float
    authority:str="OPTIMIZED_GENERATED_LAYER_GEOMETRY_NOT_OBSERVED_SOURCE"


def refine_realsas_one_layer_v1(
    *,layer_name:str,
    initial_vertices_xyz:Tensor,face_indices:Tensor,
    guidance:RealSaSGeneratedMeshGuidanceV1,
    rasterize_vertices:Callable[[Tensor],RealSaSFlexiCubesLayerRasterV1],
    fixed_inner_xyz:Tensor|None=None,
    fixed_inner_normals:Tensor|None=None,
    steps:int=100,lr:float=.01,
    mask_weight:float=1.,normal_weight:float=1.,collision_weight:float=1.,
    max_vertex_displacement:float|None=None,
) -> RealSaSRefinedLayerV1:
    """One inner-to-outer optimization stage with previous layers frozen."""
    guidance.validate()
    if (
        not layer_name or type(steps) is not int or not 1<=steps<=300
        or not math.isfinite(lr) or not 0<lr<=.1
        or initial_vertices_xyz.ndim!=2
        or initial_vertices_xyz.shape[1]!=3
        or initial_vertices_xyz.shape[0]<3
        or not bool(torch.isfinite(initial_vertices_xyz).all())
        or face_indices.ndim!=2 or face_indices.shape[-1]!=3
        or not face_indices.numel()
        or face_indices.dtype not in (torch.long,torch.int32)
        or int(face_indices.min())<0
        or int(face_indices.max())>=len(initial_vertices_xyz)
        or (max_vertex_displacement is not None and (
            not math.isfinite(max_vertex_displacement)
            or max_vertex_displacement<=0
        ))
    ):
        raise ValueError("RSS_VNEXT_REALSAS_REFINEMENT_LAYER_GEOMETRY_OR_OPTIMIZER")
    initial=initial_vertices_xyz.detach().float()
    inner=fixed_inner_xyz.detach().float() if fixed_inner_xyz is not None else None
    normals=fixed_inner_normals.detach().float() if fixed_inner_normals is not None else None
    vertex=nn.Parameter(initial.clone())
    optimizer=torch.optim.Adam([vertex],lr=lr)
    score=float("inf")
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        raster=rasterize_vertices(vertex)
        if raster.semantic_layer_name!=layer_name:
            raise ValueError("RSS_VNEXT_REALSAS_REFINEMENT_LAYER_RASTER_MISMATCH")
        terms=realsas_generated_refinement_objective_v1(
            raster=raster,guidance=guidance,current_outer_xyz=vertex,
            fixed_inner_xyz=inner,fixed_inner_normals=normals,
            mask_weight=mask_weight,normal_weight=normal_weight,
            collision_weight=collision_weight,
        )
        grad=torch.autograd.grad(terms["total"],vertex,allow_unused=False)[0]
        if not bool(torch.isfinite(grad).all()):
            raise ValueError("RSS_VNEXT_REALSAS_REFINEMENT_NONFINITE_VERTEX_GRADIENT")
        vertex.grad=grad
        torch.nn.utils.clip_grad_norm_([vertex],1.,error_if_nonfinite=True)
        optimizer.step()
        if max_vertex_displacement is not None:
            with torch.no_grad():
                direction=vertex-initial
                distance=direction.norm(dim=-1,keepdim=True)
                vertex.copy_(
                    initial+direction*(max_vertex_displacement/distance.clamp_min(1.e-8)).clamp(max=1.)
                )
        score=float(terms["total"].detach())
    return RealSaSRefinedLayerV1(
        layer_name=layer_name,refined_xyz=vertex.detach().clone(),
        face_indices=face_indices.detach().clone(),
        source_binding_sha256=guidance.original_source_binding_sha256,
        guidance_checkpoint_sha256=guidance.generation_checkpoint_sha256,
        optimizer_steps=steps,final_loss=score,
    )


def realsas_witness_refinement_renderer_v1(
    *,vertex_xyz:Tensor,witness:RSSvNextFlexiCubesSurfaceEvidence,
    iris:nn.Module,scene_triplanes:Tensor,
    graphics_head:nn.Module,semantic_role_head:nn.Module,
    cameras:Tensor,height:int,width:int,
    source_scene_prediction_sha256:str,
)->RealSaSFlexiCubesLayerRasterV1:
    """Bridge to the LIVE NVIDIA mesh raster, not a synthetic proxy."""
    if vertex_xyz.ndim!=2 or vertex_xyz.shape!=witness.xyz[0].shape:
        raise ValueError("RSS_VNEXT_REALSAS_REFINEMENT_WITNESS_VERTEX_DRIFT")
    updated=replace(witness,xyz=vertex_xyz[None])
    return render_realsas_layered_flexicubes_mesh_v1(
        witness=updated,iris=iris,scene_triplanes=scene_triplanes,
        graphics_head=graphics_head,semantic_role_head=semantic_role_head,
        cameras=cameras,height=height,width=width,
        source_scene_prediction_sha256=source_scene_prediction_sha256,
    )


__all__=[
    "RealSaSGeneratedMeshGuidanceV1","RealSaSRefinedLayerV1",
    "realsas_outer_inner_collision_loss_v1",
    "realsas_generated_refinement_objective_v1",
    "refine_realsas_one_layer_v1",
    "realsas_witness_refinement_renderer_v1",
]
