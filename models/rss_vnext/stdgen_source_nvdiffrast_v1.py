from __future__ import annotations

"""NVIDIA source-geometry layer raster candidate, not an admission court.

The existing CPU source-layer raster is a bounded reference, not a 1024px
corpus renderer. This GPU adapter consumes the EXACT SAME independently
annotated per-triangle 3D source, runs a real first-hit raster, and emits
candidate RGB, semantic, mask, normal, metric depth and camera provenance.

It never converts diffusion pixels to teacher truth. GPU results are explicitly
UNQUALIFIED until compared to the source Blender/Truth Foundry receipts:
ordinary nvdiffrast cannot disambiguate coincident coplanar surfaces.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .stdgen_source_layer_truth_v1 import (
    StdGENSourceOpaqueTriangleSurfaceV1, _hash,
)

RASTER_ID = "REAL_SOURCE3D_NVDIFFRAST_FIRST_HIT_CANDIDATE_V1"


@dataclass(frozen=True, eq=False)
class StdGENSourceLayerGPUCandidateV1:
    rgb_white_linear: np.ndarray
    alpha: np.ndarray
    role_indices: np.ndarray
    metric_depth: np.ndarray
    normal_xyz: np.ndarray
    visible_hit: np.ndarray
    source_mesh_truth_sha256: str
    source_rights_admission_sha256: str
    camera_sha256: str
    raster_receipt_sha256: str
    role_subset: tuple[int, ...]
    identity: str = RASTER_ID
    authority: str = "GPU_RENDER_CANDIDATE__NOT_SOURCE_TEACHER_ADMITTED"


def render_stdgen_opaque_source_layer_gpu_candidate_v1(
    *, source: StdGENSourceOpaqueTriangleSurfaceV1,
    camera_13d: np.ndarray, height: int, width: int,
    role_subset: tuple[int, ...],
    raster_backend: Any = None, raster_context: Any = None,
) -> StdGENSourceLayerGPUCandidateV1:
    """Opaque per-face source color/roles, real triangle z-buffer, no mean-z.

    Only resolved source materials fit this contract. Textured/material-
    blended/translucent assets MUST use the Blender source renderer instead.
    A parity/qualification court owns any transition from this candidate
    to source-3D training labels.
    """
    source.validate()
    if (type(camera_13d) is not np.ndarray or
            camera_13d.shape != (13,) or camera_13d.dtype != np.float32 or
            not np.isfinite(camera_13d).all() or camera_13d[12] <= 0 or
            type(role_subset) is not tuple or not role_subset or
            len(set(role_subset)) != len(role_subset) or
            any(type(r) is not int or r < 0 or r >= source.semantic_role_count
                for r in role_subset) or
            type(height) is not int or type(width) is not int or
            not 16 <= height <= 2048 or not 16 <= width <= 2048 or
            height * width > 2048 * 2048):
        raise ValueError("RSS_VNEXT_STDGEN_GPU_SOURCE_DOMAIN")
    xyz_axes = np.stack([
        camera_13d[3:6], camera_13d[6:9], camera_13d[9:12],
    ])
    if (not np.allclose(xyz_axes @ xyz_axes.T, np.eye(3), atol=1.e-3, rtol=0.)
            or np.linalg.det(xyz_axes) < .99):
        raise ValueError("RSS_VNEXT_STDGEN_GPU_SOURCE_CAMERA_BASIS")
    role_mask = np.isin(source.source_face_role, np.array(role_subset))
    selected_face_ids = np.flatnonzero(role_mask)
    if not len(selected_face_ids):
        raise ValueError("RSS_VNEXT_STDGEN_GPU_SOURCE_LAYER_EMPTY")
    device = "cuda" if raster_backend is None else "cpu"
    if raster_backend is None:
        if not torch.cuda.is_available():
            raise RuntimeError("RSS_VNEXT_STDGEN_GPU_SOURCE_CUDA_REQUIRED")
        try:
            import nvdiffrast.torch as dr
        except ImportError as exc:
            raise RuntimeError("RSS_VNEXT_STDGEN_GPU_SOURCE_NVDIFFRAST_REQUIRED") from exc
    else:
        dr = raster_backend

    vertices = torch.from_numpy(source.source_mesh_xyz.copy()).to(device=device)
    triangles = torch.from_numpy(
        source.source_faces[selected_face_ids].astype(np.int32)
    ).to(device=device)
    camera = torch.from_numpy(camera_13d.copy()).to(device=device)
    delta = vertices - camera[:3]
    forward = delta @ camera[9:12]
    with torch.no_grad():
        near = forward.min() - .05
        far = forward.max() + .05
    if float(far - near) < 1.e-5 or float(near) < 0:
        raise ValueError("RSS_VNEXT_STDGEN_GPU_SOURCE_DEPTH_DOMAIN")
    right = (delta @ camera[3:6]) / camera[12]
    up = (delta @ camera[6:9]) / camera[12]
    z = 2.*(forward-near)/(far-near) - 1.
    clip = torch.stack((right,up,z,torch.ones_like(z)),dim=-1)[None].contiguous()
    if raster_context is None:
        raster_context = dr.RasterizeCudaContext()
    rast,_ = dr.rasterize(raster_context,clip,triangles,
                          resolution=(height,width))
    # raster origin is bottom-left, canonical source images are top-left
    pix_face = rast[...,3].long().flip(1).squeeze(0) - 1
    coverage = pix_face >= 0
    hit_points,_ = dr.interpolate(vertices[None].contiguous(),rast,triangles)
    hit_points = hit_points.flip(1).squeeze(0)
    depths = ((hit_points-camera[:3]) * camera[9:12]).sum(-1)
    # Face-constant material and independently annotated role, not inferred
    # surface appearance. Per-fragment first-hit chooses the actual triangle.
    face_idx = pix_face.clamp(min=0)
    colors = torch.from_numpy(source.source_face_rgb_linear[selected_face_ids].copy()).to(device)
    roles = torch.from_numpy(source.source_face_role[selected_face_ids].copy()).to(device)
    vertices_face = vertices[triangles.long()]
    fn = torch.linalg.cross(
        vertices_face[:,1]-vertices_face[:,0],
        vertices_face[:,2]-vertices_face[:,0],
    )
    fn = torch.nn.functional.normalize(fn,dim=-1)
    colors_per_pixel = colors[face_idx]
    roles_per_pixel = roles[face_idx]
    normals_per_pixel = fn[face_idx]
    rgb = torch.where(coverage[...,None],colors_per_pixel,
                      torch.ones_like(colors_per_pixel))
    rid = torch.where(coverage,roles_per_pixel,torch.full_like(roles_per_pixel,-1))
    normal = torch.where(coverage[...,None],normals_per_pixel,
                         torch.zeros_like(normals_per_pixel))
    depth = torch.where(coverage,depths,torch.zeros_like(depths))
    alpha = coverage.to(torch.float32)[...,None]
    arrays = [
        t.detach().cpu().contiguous().numpy()
        for t in (rgb,alpha,rid,depth[...,None],normal,coverage)
    ]
    payload = {
        "schema": RASTER_ID,
        "source_mesh_truth_sha256": source.source_mesh_truth_sha256,
        "source_rights_admission_sha256": source.source_rights_admission_sha256,
        "source_role_hash": _hash(source.source_face_role),
        "source_rgb_hash": _hash(source.source_face_rgb_linear),
        "selected_faces": selected_face_ids.tolist(),
        "camera_sha256": _hash(camera_13d),
        "height": height, "width": width, "role_subset": list(role_subset),
        "output_hashes": [_hash(np.ascontiguousarray(arr)) for arr in arrays],
        "authority": "UNQUALIFIED_GPU_RASTER_CANDIDATE",
    }
    receipt = sha256(json.dumps(
        payload,sort_keys=True,separators=(",",":"),
    ).encode()).hexdigest()
    return StdGENSourceLayerGPUCandidateV1(
        rgb_white_linear=arrays[0],alpha=arrays[1],
        role_indices=arrays[2],metric_depth=arrays[3],
        normal_xyz=arrays[4],visible_hit=arrays[5],
        source_mesh_truth_sha256=source.source_mesh_truth_sha256,
        source_rights_admission_sha256=source.source_rights_admission_sha256,
        camera_sha256=_hash(camera_13d),
        raster_receipt_sha256=receipt,role_subset=role_subset,
    )


__all__ = [
    "RASTER_ID","StdGENSourceLayerGPUCandidateV1",
    "render_stdgen_opaque_source_layer_gpu_candidate_v1",
]
