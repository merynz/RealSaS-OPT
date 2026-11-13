from __future__ import annotations

"""Live RealSaS stage3: FlexiCubes layered mesh -> nvdiffrast RGB/semantic.

Reference: CVPR 2025 §3.2 Stage3 Eq(10) uses semantic-equivalent SDF,
FlexiCubes and mesh RGB/semantic/normal/depth raster supervision. This
independently written adapter consumes the EXISTING PINNED NVIDIA
FlexiCubes witness, queries the SAME IRIS H(x) for appearance and roles,
then uses nvdiffrast's GPU differentiation primitives. It never copies
released research code/weights and never mints source-surface ownership.

Prerequisites for actual training: local trusted NVIDIA nvdiffrast runtime
with GPU support, aligned source-render teacher with exact camera and
metric-depth/normal-space convention, and source-signed qualification.
CPU unit tests use a labelled fake raster API; they do NOT prove the
native CUDA kernel or high-resolution fidelity.
"""

from hashlib import sha256
import json
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from models.iris.vnext.flexicubes_v1 import vertex_normals_v1
from .flexicubes_surface_evidence import RSSvNextFlexiCubesSurfaceEvidence
from .realsas_joint_slrm_render_v1 import RealSaSFlexiCubesLayerRasterV1

RASTER_ID="NVLABS_NVDIFFRAST_FLEXICUBES_SHARED_H_STAGE3_V1"


def _mesh_digest(vertices: Tensor,faces: Tensor) -> str:
    h=sha256()
    for t in (vertices,faces):
        a=t.detach().contiguous().cpu()
        h.update(json.dumps({
            "dtype":str(a.dtype),"shape":list(a.shape)
        },sort_keys=True,separators=(",",":")).encode("utf8")+b"\0")
        h.update(a.numpy().tobytes())
    return h.hexdigest()


def _camera_check(cameras: Tensor) -> None:
    if (
        cameras.ndim!=3 or cameras.shape[0]!=1
        or cameras.shape[1] not in (1,6)
        or cameras.shape[2]!=13
        or not torch.is_floating_point(cameras)
        or not bool(torch.isfinite(cameras).all())
        or bool((cameras[:,:,12]<=0).any())
    ):
        raise ValueError("RSS_VNEXT_REALSAS_STAGE3_CAMERA_DOMAIN")
    xyz=torch.stack((
        cameras[:,:,3:6],cameras[:,:,6:9],cameras[:,:,9:12],
    ),dim=-2)
    if not bool(torch.allclose(
        xyz@xyz.transpose(-1,-2),
        torch.eye(3,device=xyz.device,dtype=xyz.dtype),
        atol=1.e-3,rtol=0.,
    )) or bool((torch.linalg.det(xyz)<.99).any()):
        raise ValueError("RSS_VNEXT_REALSAS_STAGE3_CAMERA_ORTHONORMAL")


def render_realsas_layered_flexicubes_mesh_v1(
    *,
    witness: RSSvNextFlexiCubesSurfaceEvidence,
    iris: nn.Module,
    scene_triplanes: Tensor,
    graphics_head: nn.Module,
    semantic_role_head: nn.Module,
    cameras: Tensor,
    height: int,width: int,
    source_scene_prediction_sha256: str,
    raster_backend: Any = None,
    raster_context: Any = None,
    query_chunk: int = 65536,
) -> RealSaSFlexiCubesLayerRasterV1:
    """Differentiable mesh-stage producer for L3, from signed H(x) witness.

    Per-view z-buffer/first-hit computed by NVIDIA rasterization; NDC camera
    vertical axis is flipped for top-left torch image convention.
    No average-z / whole-face single-depth shortcut is used. Full rendered
    RGB/mask/semantics/normal/depth are bound to exactly the witness mesh.
    """
    if (
        type(witness) is not RSSvNextFlexiCubesSurfaceEvidence
        or witness.source_policy!="SIGNED_HYPOTHESIS_ONLY"
        or witness.evidence_authority!="UNQUALIFIED_FLEXICUBES_SURFACE_WITNESS"
        or not witness.layer_name
        or len(source_scene_prediction_sha256)!=64
        or any(x not in "0123456789abcdef" for x in source_scene_prediction_sha256)
    ):
        raise ValueError("RSS_VNEXT_REALSAS_STAGE3_WITNESS_AUTHORITY")
    k=witness.presentation_role_logits.shape[-1]
    d=witness.local_features.shape[-1]
    witness.validate(carrier_dim=d,role_count=k)
    _camera_check(cameras)
    v=cameras.shape[1]
    n=witness.xyz.shape[1]
    nf=witness.faces.shape[0]
    if (
        height<8 or width<8 or height>1024 or width>1024
        or v*height*width>6*1024*1024
        or n>250000 or nf>250000
        or query_chunk<128 or query_chunk>262144
    ):
        raise ValueError("RSS_VNEXT_REALSAS_STAGE3_RESOURCE_BUDGET")
    if (
        scene_triplanes.ndim!=5 or scene_triplanes.shape[:2]!=(1,3)
        or scene_triplanes.device!=witness.xyz.device
        or cameras.device!=witness.xyz.device
        or not bool(torch.isfinite(scene_triplanes).all())
    ):
        raise ValueError("RSS_VNEXT_REALSAS_STAGE3_SHARED_SCENE_DRIFT")
    if witness.xyz.device.type!="cuda" and raster_backend is None:
        raise RuntimeError("RSS_VNEXT_REALSAS_STAGE3_NVDIFFRAST_GPU_REQUIRED")
    if raster_backend is None:
        try:
            import nvdiffrast.torch as dr
        except ImportError as exc:
            raise RuntimeError("RSS_VNEXT_REALSAS_STAGE3_NVDIFFRAST_INSTALL_REQUIRED") from exc
    else:
        dr=raster_backend
    if raster_context is None:
        raster_context=dr.RasterizeCudaContext()
    verts=witness.xyz[0].float()
    tri=witness.faces.to(torch.int32).contiguous()
    normals=vertex_normals_v1(witness.xyz[0],witness.faces).float()
    per_view=[]
    for vi in range(v):
        cam=cameras[0,vi].float()
        delta=verts-cam[None,:3]
        right=delta@cam[3:6]/cam[12]
        up=delta@cam[6:9]/cam[12]
        forward=delta@cam[9:12]
        with torch.no_grad():
            lower=forward.min()-.05
            upper=forward.max()+.05
        if float((upper-lower).detach())<=1.e-5:
            raise ValueError("RSS_VNEXT_REALSAS_STAGE3_DEGENERATE_DEPTH_RANGE")
        z=(forward-lower)*2./(upper-lower)-1.
        clip=torch.stack((
            right,up,z,torch.ones_like(right),
        ),dim=-1).contiguous()[None]
        rast,_db=dr.rasterize(
            raster_context,clip,tri,resolution=(height,width),
        )
        # NVIDIA nvdiffrast raster image origin is lower-left. Model and
        # admitted PNG teachers are top-left: explicit y axis reversal.
        coverage=(rast[...,3:4]>0).to(verts.dtype)
        sampled_position,_=dr.interpolate(
            verts[None].contiguous(),rast,tri,
        )
        sampled_normal,_=dr.interpolate(
            normals[None].contiguous(),rast,tri,
        )
        points=sampled_position.reshape(1,height*width,3)
        rgba_chunks=[]
        role_chunks=[]
        for start in range(0,height*width,query_chunk):
            h=iris.query(
                scene_triplanes,points[:,start:start+query_chunk].to(scene_triplanes.dtype),
            )["H"]
            graphics=graphics_head(h)
            logits=semantic_role_head(h)
            if graphics.shape[-1]!=5 or logits.shape[-1]!=k:
                raise ValueError("RSS_VNEXT_REALSAS_STAGE3_SHARED_HEAD_CONTRACT")
            rgba_chunks.append(torch.sigmoid(graphics[...,0:3].float()))
            role_chunks.append(F.softmax(logits.float(),dim=-1))
        rgb=torch.cat(rgba_chunks,dim=1).reshape(1,height,width,3)
        role=torch.cat(role_chunks,dim=1).reshape(1,height,width,k)
        rgba=torch.cat((
            rgb*coverage,coverage,
        ),dim=-1).contiguous()
        # Use the official nvdiffrast triangle-edge antialias operation.
        # Never blurrily average depths, which would invalidate z ordering.
        smooth=dr.antialias(rgba,rast,clip,tri).clamp(0.,1.)
        alpha=smooth[...,3:4]
        whitened=(smooth[...,:3]+1.-alpha).clamp(0.,1.)
        normal=F.normalize(sampled_normal,dim=-1,eps=1.e-8)*coverage
        metric_depth=((sampled_position-cam[None,None,None,:3])*
                      cam[None,None,None,9:12]).sum(-1,keepdim=True)*coverage
        # Explicitly convert from GL bottom-left raster orientation.
        values=[whitened,alpha,role*coverage,normal,metric_depth]
        per_view.append(tuple(x.flip(dims=(1,)).reshape(1,height*width,-1) for x in values))
    grouped=[
        torch.cat([row[i] for row in per_view],dim=1)
        for i in range(5)
    ]
    rgb,alpha,semantic,normal,depth=grouped
    return RealSaSFlexiCubesLayerRasterV1(
        rgb_white_composite=rgb,alpha=alpha,
        semantic_posterior=semantic,
        normal_xyz=normal,depth=depth,
        flexicubes_deformation_regularizer=witness.regularizer.float().mean(),
        source_scene_prediction_sha256=source_scene_prediction_sha256,
        mesh_surface_prediction_sha256=_mesh_digest(witness.xyz,witness.faces),
        semantic_layer_name=witness.layer_name,
        rasterizer_identity=RASTER_ID,
    )


__all__=["RASTER_ID","render_realsas_layered_flexicubes_mesh_v1"]
