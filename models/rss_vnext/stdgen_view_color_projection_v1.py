from __future__ import annotations

"""Camera/depth-qualified generated-view vertex-color candidates.

StdGEN post-refinement transfers color from six generated views onto its
newly refined mesh. This implementation deliberately produces only a
PREDICTED appearance proposal; it is not original-image pixel ownership,
CAA, an observed hidden surface, a texture atlas, or a qualified puppet.

The source mesh must be independently validated before any promotion.
Any vertex/face remeshing changes its exact parent hash and invalidates this
proposal. Unseen vertices remain UNKNOWN rather than borrowing an invented
color. This is also not the final UV/fragment-space color bake.
"""
from dataclasses import dataclass
from hashlib import sha256
import json

import numpy as np
import torch

from .stdgen_flexicubes_mesh_render_v1 import RASTER_ID, _mesh_digest


@dataclass(frozen=True, eq=False)
class StdGENGeneratedViewColorCandidatesV1:
    vertex_rgb_linear: np.ndarray
    vertex_valid: np.ndarray
    selected_view: np.ndarray
    view_confidence: np.ndarray
    mesh_parent_sha256: str
    generated_views_sha256: str
    source_binding_sha256: str
    generating_checkpoint_sha256: str
    authority: str = "PREDICTED_GENERATED_VIEW_COLOR_CANDIDATE_NOT_SOURCE_PIXELS"


def _sha(v: object) -> bool:
    return type(v) is str and len(v) == 64 and all(x in "0123456789abcdef" for x in v)


def _digest_array(a: np.ndarray) -> str:
    hdr=json.dumps({"shape":list(a.shape),"dtype":str(a.dtype)},
                   sort_keys=True,separators=(",",":")).encode("utf8")
    return sha256(hdr+b"\0"+np.ascontiguousarray(a).tobytes()).hexdigest()


def project_stdgen_generated_view_colors_v1(
    *, vertices: np.ndarray, faces: np.ndarray, vertex_normals: np.ndarray,
    generated_srgb: np.ndarray, generated_alpha: np.ndarray,
    generated_camera13: np.ndarray, mesh_raster_depth: np.ndarray,
    mesh_raster_alpha: np.ndarray,
    mesh_parent_sha256: str, raster_mesh_parent_sha256: str,
    rasterizer_identity: str,
    source_binding_sha256: str, generating_checkpoint_sha256: str,
    depth_tolerance: float = 0.01, min_facing: float = 0.1,
    min_alpha: float = 0.95,
) -> StdGENGeneratedViewColorCandidatesV1:
    """Transfer only z-visible pixels from *generated* RGB images to mesh.

    Cameras use the orthographic source contract [origin3,right3,up3,
    forward3,half_extent1] and top-left pixel center. This function requires
    *mesh-derived* depth/alpha (not diffusion depth guesses) from Stage3.
    Front-facing is dot(-forward, outward normal) > min_facing. Donor is the
    best view by facing*alpha; a confident observed source donor still wins
    at the separate Compiler/CAA authority boundary.
    """
    if (
        not all(_sha(v) for v in (
            mesh_parent_sha256,raster_mesh_parent_sha256,
            source_binding_sha256,generating_checkpoint_sha256,
        ))
        or rasterizer_identity != RASTER_ID
        or not isinstance(vertices,np.ndarray) or vertices.dtype != np.float32
        or vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices)<3
        or not isinstance(faces,np.ndarray) or faces.dtype != np.int64
        or faces.ndim != 2 or faces.shape[1] != 3 or len(faces)<1
        or not np.isfinite(vertices).all()
        or np.any(faces<0) or np.any(faces>=len(vertices))
        or not isinstance(vertex_normals,np.ndarray)
        or vertex_normals.shape != vertices.shape
        or vertex_normals.dtype != np.float32
        or not np.isfinite(vertex_normals).all()
        or not isinstance(generated_srgb,np.ndarray)
        or generated_srgb.dtype != np.float32
        or generated_srgb.ndim != 4 or generated_srgb.shape[0] != 6
        or generated_srgb.shape[-1] != 3
        or not isinstance(generated_alpha,np.ndarray)
        or generated_alpha.dtype != np.float32
        or generated_alpha.shape != generated_srgb.shape[:-1]
        or not isinstance(mesh_raster_depth,np.ndarray)
        or mesh_raster_depth.dtype != np.float32
        or mesh_raster_depth.shape != generated_alpha.shape
        or not isinstance(mesh_raster_alpha,np.ndarray)
        or mesh_raster_alpha.dtype != np.float32
        or mesh_raster_alpha.shape != generated_alpha.shape
        or not isinstance(generated_camera13,np.ndarray)
        or generated_camera13.dtype != np.float32
        or generated_camera13.shape != (6,13)
        or not all(np.isfinite(x).all() for x in (
            generated_srgb,generated_alpha,mesh_raster_depth,
            mesh_raster_alpha,generated_camera13,
        ))
        or not (0.0 < depth_tolerance <= .25)
        or not (0.0 <= min_facing < 1.0)
        or not (0.0 < min_alpha <= 1.0)
        or np.any(generated_srgb<0) or np.any(generated_srgb>1)
        or np.any(generated_alpha<0) or np.any(generated_alpha>1)
        or np.any(mesh_raster_alpha<0) or np.any(mesh_raster_alpha>1)
        or np.any(mesh_raster_depth<0)
    ):
        raise ValueError("RSS_VNEXT_STDGEN_VIEW_COLOR_SOURCE_OR_RENDER_DOMAIN")
    h,w=generated_srgb.shape[1:3]
    if min(h,w)<8 or max(h,w)>2048 or h*w*6>6*2048*2048:
        raise ValueError("RSS_VNEXT_STDGEN_VIEW_COLOR_RESOURCE_BUDGET")
    camera_axes=np.stack((generated_camera13[:,3:6],
                          generated_camera13[:,6:9],
                          generated_camera13[:,9:12]),axis=-2)
    gram=camera_axes@np.swapaxes(camera_axes,-1,-2)
    if (not np.allclose(gram,np.eye(3),atol=1.e-3,rtol=0)
        or np.any(np.linalg.det(camera_axes)<.99)
        or np.any(generated_camera13[:,12]<=0)):
        raise ValueError("RSS_VNEXT_STDGEN_VIEW_COLOR_CAMERA_ORTHONORMAL")
    actual_mesh=_mesh_digest(torch.from_numpy(vertices[None].copy()),
                             torch.from_numpy(faces.copy()))
    if mesh_parent_sha256!=actual_mesh or raster_mesh_parent_sha256!=actual_mesh:
        raise ValueError("RSS_VNEXT_STDGEN_VIEW_COLOR_MESH_PARENT_DRIFT")
    normal_length=np.linalg.norm(vertex_normals,axis=-1)
    if np.any(normal_length<1.e-6) or np.any(abs(normal_length-1)>1.e-3):
        raise ValueError("RSS_VNEXT_STDGEN_VIEW_COLOR_NORMAL_DOMAIN")
    n=len(vertices)
    best=np.full(n,-1,dtype=np.int64)
    scores=np.zeros(n,dtype=np.float32)
    rgb=np.zeros((n,3),dtype=np.float32)
    for v in range(6):
        cam=generated_camera13[v]
        delta=vertices-cam[:3]
        x=(delta@cam[3:6]/cam[12]+1.)*w*.5-.5
        y=(1.-delta@cam[6:9]/cam[12])*h*.5-.5
        ix=np.rint(x).astype(np.int64)
        iy=np.rint(y).astype(np.int64)
        within=(ix>=0)&(ix<w)&(iy>=0)&(iy<h)
        ix=np.clip(ix,0,w-1);iy=np.clip(iy,0,h-1)
        depth=delta@cam[9:12]
        normal_facing=-vertex_normals@cam[9:12]
        image_alpha=generated_alpha[v,iy,ix]
        mesh_alpha=mesh_raster_alpha[v,iy,ix]
        mesh_depth=mesh_raster_depth[v,iy,ix]
        visible=(within&(depth>0)&(normal_facing>min_facing)
                 &(image_alpha>=min_alpha)&(mesh_alpha>=.99)
                 &(np.abs(depth-mesh_depth)<=depth_tolerance))
        confidence=np.where(visible,normal_facing*image_alpha,0.).astype(np.float32)
        update=visible&(confidence>scores)
        if not np.any(update):
            continue
        encoded=generated_srgb[v,iy[update],ix[update]]
        linear=np.where(encoded<=.04045,encoded/12.92,
                        ((encoded+.055)/1.055)**2.4)
        rgb[update]=linear
        best[update]=v
        scores[update]=confidence[update]
    digest=_digest_array(generated_srgb)
    return StdGENGeneratedViewColorCandidatesV1(
        vertex_rgb_linear=rgb,vertex_valid=best>=0,selected_view=best,
        view_confidence=scores,mesh_parent_sha256=actual_mesh,
        generated_views_sha256=digest,
        source_binding_sha256=source_binding_sha256,
        generating_checkpoint_sha256=generating_checkpoint_sha256,
    )


__all__=["StdGENGeneratedViewColorCandidatesV1",
         "project_stdgen_generated_view_colors_v1"]
