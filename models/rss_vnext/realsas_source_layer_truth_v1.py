from __future__ import annotations

"""Actual geometry-derived RealSaS S-LRM source layer teachers (opaque subset).

Rasterized RGB, mask, semantic front-surface role, normal and *source* depth
derive from the same independent triangulated source geometry. Holding out
outer roles exposes internal original geometry; it is NOT manufactured from
image diffusion. This admissible subset handles already-resolved per-face
linear RGB and opaque triangles, NOT Blender textures, translucent source-
over, multi-material semantic unknown faces or arbitrary scene occlusion.
Those must use the existing Blender Truth Foundry measured raster path.

This utility is for source-truth preparation, NEVER model inference.
A SHA receipt tracks determinism but is NOT rights or source qualification.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
import numpy as np
import torch

from .realsas_joint_slrm_render_v1 import (
    RealSaSJointTargetV1,STAGE_1,STAGE_2,STAGE_3,
)

_SCHEMA="RealSaS.RSSvNext.RealSaSSource3DLayerRaster.v1"
_PIXEL_BUDGET=128*128
_FACE_TEST_BUDGET=1_000_000


def _hash(t:np.ndarray) -> str:
    a=np.ascontiguousarray(t)
    hdr=json.dumps({"dtype":str(a.dtype),"shape":list(a.shape)},
        sort_keys=True,separators=(",",":")).encode()
    return sha256(hdr+b"\0"+a.tobytes()).hexdigest()


def _sha(s:object) -> bool:
    return type(s) is str and len(s)==64 and all(c in "0123456789abcdef" for c in s)


@dataclass(frozen=True,eq=False)
class RealSaSSourceOpaqueTriangleSurfaceV1:
    source_mesh_xyz:np.ndarray    # [V,3] float32; source pose resolved
    source_faces:np.ndarray       # [F,3] int64, exact source triangles
    source_face_rgb_linear:np.ndarray  # [F,3] float32, independently decoded source
    source_face_role:np.ndarray  # [F] int64, independently annotated
    semantic_role_count:int
    source_mesh_truth_sha256:str
    source_rights_admission_sha256:str
    renderer_version_sha256:str
    source_colors_from_artist_materials:bool
    source_roles_independently_assigned:bool
    source_surface_is_opaque:bool
    source_scene_contains_unmodeled_occluders:bool=False

    def validate(self) -> None:
        n=len(self.source_mesh_xyz);f=len(self.source_faces)
        if (
            not all(_sha(x) for x in (
                self.source_mesh_truth_sha256,
                self.source_rights_admission_sha256,
                self.renderer_version_sha256,
            ))
            or type(self.semantic_role_count) is not int
            or not 2<=self.semantic_role_count<=16
            or n<3 or n>100000 or f<1 or f>20000
            or type(self.source_mesh_xyz) is not np.ndarray
            or self.source_mesh_xyz.shape!=(n,3)
            or self.source_mesh_xyz.dtype!=np.float32
            or type(self.source_faces) is not np.ndarray
            or self.source_faces.shape!=(f,3)
            or self.source_faces.dtype!=np.int64
            or type(self.source_face_rgb_linear) is not np.ndarray
            or self.source_face_rgb_linear.shape!=(f,3)
            or self.source_face_rgb_linear.dtype!=np.float32
            or type(self.source_face_role) is not np.ndarray
            or self.source_face_role.shape!=(f,)
            or self.source_face_role.dtype!=np.int64
            or not np.isfinite(self.source_mesh_xyz).all()
            or not np.isfinite(self.source_face_rgb_linear).all()
            or np.any(self.source_face_rgb_linear<0)
            or np.any(self.source_face_rgb_linear>1)
            or np.any(self.source_faces<0)
            or np.any(self.source_faces>=n)
            or np.any(self.source_face_role<0)
            or np.any(self.source_face_role>=self.semantic_role_count)
            or self.source_colors_from_artist_materials is not True
            or self.source_roles_independently_assigned is not True
            or self.source_surface_is_opaque is not True
            or self.source_scene_contains_unmodeled_occluders is not False
        ):
            raise ValueError("RSS_VNEXT_REALSAS_SOURCE_TRIANGLE_TRUTH_NOT_QUALIFIED")
        triangle=self.source_mesh_xyz[self.source_faces]
        norms=np.linalg.norm(np.cross(
            triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0],
        ),axis=-1)
        if np.any(norms<1.e-8):
            raise ValueError("RSS_VNEXT_REALSAS_SOURCE_DEGENERATE_TRIANGLE")


@dataclass(frozen=True,eq=False)
class RealSaSSourceOpaqueLayerRasterV1:
    rgb_white:np.ndarray          # [H,W,3] linear [0,1]
    alpha:np.ndarray              # [H,W,1] binary
    role_indices:np.ndarray       # [H,W] int64, -1 for background
    metric_depth:np.ndarray       # [H,W,1] true source depth inside valid
    normal_xyz:np.ndarray         # [H,W,3] exact source face normal
    visible_valid_mask:np.ndarray # [H,W] source hit, not inferred
    unambiguous_mask:np.ndarray   # [H,W] known front/back pixel
    scene_source_binding_sha256:str
    layer_source_render_sha256:str
    role_subset:tuple[int,...]
    camera_13d:tuple[float,...]
    render_contract:str=_SCHEMA
    is_generated_diffusion_prediction:bool=False
    source_is_watertight_signed_mesh:bool=False

    def as_training_target(
        self,*,stage:str,layer_name:str,
    )->RealSaSJointTargetV1:
        if stage not in (STAGE_1,STAGE_2,STAGE_3):
            raise ValueError("RSS_VNEXT_REALSAS_LAYER_STAGE_INVALID")
        if not layer_name:
            raise ValueError("RSS_VNEXT_REALSAS_LAYER_NAME_REQUIRED")
        h,w=self.role_indices.shape
        return RealSaSJointTargetV1(
            rgb_white=torch.from_numpy(self.rgb_white.reshape(1,h*w,3).copy()),
            alpha=torch.from_numpy(self.alpha.reshape(1,h*w,1).copy()),
            role_indices=torch.from_numpy(self.role_indices.reshape(1,h*w).copy()),
            qualified_rgb_mask=torch.from_numpy(self.unambiguous_mask.reshape(1,h*w).copy()),
            qualified_mask_mask=torch.from_numpy(self.unambiguous_mask.reshape(1,h*w).copy()),
            qualified_role_mask=torch.from_numpy(self.visible_valid_mask.reshape(1,h*w).copy()),
            source_3d_truth_binding_sha256=self.layer_source_render_sha256,
            layer_name=layer_name,stage=stage,
        )


def render_realsas_source_opaque_peeled_layer_v1(
    *,source:RealSaSSourceOpaqueTriangleSurfaceV1,
    camera_13d:np.ndarray,width:int,height:int,
    role_subset:tuple[int,...],
)->RealSaSSourceOpaqueLayerRasterV1:
    """Top-left pixel-centre barycentrics, per-pixel true source depth.

    For each pixel, first-hit = smallest camera-forward ray distance.
    Exact depth ties between DIFFERENT faces are marked ambiguous and never
    become positive color/semantic training labels. No mean/average z per
    triangle, no generated hidden depth, no source over translucency.
    """
    source.validate()
    if (
        type(camera_13d) is not np.ndarray or camera_13d.dtype!=np.float32
        or camera_13d.shape!=(13,) or not np.isfinite(camera_13d).all()
        or camera_13d[12]<=0
        or type(role_subset) is not tuple or len(role_subset)<1
        or len(set(role_subset))!=len(role_subset)
        or any(type(x) is not int or not 0<=x<source.semantic_role_count for x in role_subset)
        or type(width) is not int or type(height) is not int
        or width<8 or height<8 or width*height>_PIXEL_BUDGET
        or width*height*len(source.source_faces)>_FACE_TEST_BUDGET
    ):
        raise ValueError("RSS_VNEXT_REALSAS_SOURCE_LAYER_CAMERA_OR_BUDGET")
    axes=np.stack((camera_13d[3:6],camera_13d[6:9],camera_13d[9:12]))
    if not np.allclose(axes@axes.T,np.eye(3),atol=1.e-3,rtol=0.) or np.linalg.det(axes)<.99:
        raise ValueError("RSS_VNEXT_REALSAS_SOURCE_LAYER_CAMERA_BASIS")
    delta=source.source_mesh_xyz.astype(np.float64)-camera_13d[:3].astype(np.float64)
    u=(delta@camera_13d[3:6])/float(camera_13d[12])
    v=(delta@camera_13d[6:9])/float(camera_13d[12])
    depths=delta@camera_13d[9:12]
    xy=np.stack(((u+1)*width*.5,(1-v)*height*.5),axis=-1)
    x=np.arange(width,dtype=np.float64)+.5
    y=np.arange(height,dtype=np.float64)+.5
    py,px=np.meshgrid(y,x,indexing="ij")
    zbuffer=np.full((height,width),np.inf,dtype=np.float64)
    face_id=np.full((height,width),-1,dtype=np.int64)
    ambiguous=np.zeros((height,width),dtype=bool)
    for f,tri in enumerate(source.source_faces):
        if int(source.source_face_role[f]) not in role_subset:
            continue
        pts=xy[tri]
        d0=(pts[1,1]-pts[2,1])*(pts[0,0]-pts[2,0])+(pts[2,0]-pts[1,0])*(pts[0,1]-pts[2,1])
        if abs(d0)<1.e-10:
            # Degenerate in this projection, not a source-mesh failure.
            continue
        wa=((pts[1,1]-pts[2,1])*(px-pts[2,0])+(pts[2,0]-pts[1,0])*(py-pts[2,1]))/d0
        wb=((pts[2,1]-pts[0,1])*(px-pts[2,0])+(pts[0,0]-pts[2,0])*(py-pts[2,1]))/d0
        wc=1.-wa-wb
        hit=(wa>=-1.e-9)&(wb>=-1.e-9)&(wc>=-1.e-9)
        z=wa*depths[tri[0]]+wb*depths[tri[1]]+wc*depths[tri[2]]
        hit &= z>=0
        closer=hit&(z<zbuffer-1.e-6)
        tie=hit&(np.abs(z-zbuffer)<=1.e-6)&(face_id!=f)
        ambiguous[closer]=False
        ambiguous[tie]=True
        zbuffer[closer]=z[closer]
        face_id[closer]=f
    valid=(face_id>=0)&~ambiguous
    alpha=(face_id>=0).astype(np.float32)[...,None]
    rgb_white=np.ones((height,width,3),dtype=np.float32)
    indices=np.full((height,width),-1,dtype=np.int64)
    depth=np.zeros((height,width,1),dtype=np.float32)
    normal=np.zeros((height,width,3),dtype=np.float32)
    for f,tri in enumerate(source.source_faces):
        mask=(face_id==f)&valid
        if not np.any(mask): continue
        rgb_white[mask]=source.source_face_rgb_linear[f]
        indices[mask]=source.source_face_role[f]
        depth[mask,0]=zbuffer[mask].astype(np.float32)
        v0,v1,v2=source.source_mesh_xyz[tri]
        n=np.cross(v1-v0,v2-v0)
        normal[mask]=(n/np.linalg.norm(n)).astype(np.float32)
    # Ambiguous pixels: no color or depth supervision; they are not
    # guessed to be one class. Alpha is still 1 if geometry projects here.
    payload=dict(
        scene_source_binding_sha256=source.source_mesh_truth_sha256,
        rights_admission_sha256=source.source_rights_admission_sha256,
        role_subset=role_subset,width=width,height=height,
        camera_sha256=_hash(camera_13d),
        xyz_sha256=_hash(source.source_mesh_xyz),
        face_sha256=_hash(source.source_faces),
        role_sha256=_hash(source.source_face_role),
        rgb_source_sha256=_hash(source.source_face_rgb_linear),
        output_rgb_sha256=_hash(rgb_white),
        output_depth_sha256=_hash(depth),
        output_role_sha256=_hash(indices),
        output_mask_sha256=_hash(alpha),
    )
    receipt=sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return RealSaSSourceOpaqueLayerRasterV1(
        rgb_white=rgb_white,alpha=alpha,role_indices=indices,
        metric_depth=depth,normal_xyz=normal,visible_valid_mask=valid,
        unambiguous_mask=~ambiguous,
        scene_source_binding_sha256=source.source_mesh_truth_sha256,
        layer_source_render_sha256=receipt,role_subset=role_subset,
        camera_13d=tuple(float(x) for x in camera_13d),
    )


__all__=[
    "RealSaSSourceOpaqueTriangleSurfaceV1",
    "RealSaSSourceOpaqueLayerRasterV1",
    "render_realsas_source_opaque_peeled_layer_v1",
]
