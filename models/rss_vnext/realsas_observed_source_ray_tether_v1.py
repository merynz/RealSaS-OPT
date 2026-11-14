from __future__ import annotations

"""Source-authoritative FRONT/BACK photometric tether for the SAME S-LRM H(x).

Two ORIGINAL cameras/images must have independent provenance. Camera rays
sample H(x); the recorded source pixel supplies ONLY RGB/alpha along the
entire ray. It does not supply 3D depth, a point material, surface semantics,
or a generated six-view source observation. Unknown-camera sources cannot
activate this loss.

The caller's trusted launcher (not this module) qualifies camera calibration,
original image lineage, linear RGB space, source rights and pixel masks.
"""

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F


SCHEMA = "RealSaS.RSSvNext.ObservedTwoSourceRayPhotometry.v1"
CAMERA_ABI = "ORIGINAL_CALIBRATED_ORTHOGRAPHIC_13D_RIGHT_UP_FORWARD_HALF_EXTENT"


@dataclass(frozen=True)
class RSSvNextOriginalObservedRayBatchV1:
    original_linear_rgba: Tensor    # [B,2,4,H,W] native original, straight linear RGB
    original_camera_13d: Tensor     # [B,2,13] independently calibrated camera
    source_pixel_indices: Tensor    # [B,R,3] view index, pixel y, pixel x
    ray_depth: Tensor               # [B,R,S] camera-forward distances (NOT observed depths)
    ray_delta: Tensor               # [B,R,S] positive quadrature intervals
    qualified_rgb_mask: Tensor      # [B,R], source-observed pixels
    qualified_alpha_mask: Tensor    # [B,R], source-observed alpha


@dataclass(frozen=True)
class RSSvNextOriginalObservedRaySamplesV1:
    sample_xyz: Tensor
    ray_delta: Tensor
    ray_depth: Tensor
    ray_direction: Tensor
    rgb_white: Tensor
    alpha: Tensor
    qualified_rgb_mask: Tensor
    qualified_alpha_mask: Tensor
    source_depth_is_truth: bool = False
    source_geometry_is_truth: bool = False


def construct_original_observed_ray_samples_v1(
    batch: RSSvNextOriginalObservedRayBatchV1,
) -> RSSvNextOriginalObservedRaySamplesV1:
    """Construct exactly calibrated per-pixel camera rays, without XYZ teachers.

    Targets are taken directly from original source photo pixels. White
    compositing matches the S-LRM renderer's white background convention.
    Any missing camera, arbitrary rays or forged target tensor is rejected.
    """
    if type(batch) is not RSSvNextOriginalObservedRayBatchV1:
        raise ValueError("RSS_VNEXT_ORIGINAL_RAY_BATCH_ABI")
    rgba = batch.original_linear_rgba
    cam = batch.original_camera_13d
    idx = batch.source_pixel_indices
    depth = batch.ray_depth
    delta = batch.ray_delta
    if (
        not isinstance(rgba, Tensor) or rgba.ndim != 5
        or rgba.shape[1:3] != (2,4)
        or not torch.is_floating_point(rgba)
        or rgba.shape[0] < 1 or rgba.shape[0] > 8
        or min(rgba.shape[-2:]) < 8 or max(rgba.shape[-2:]) > 2048
        or not isinstance(cam, Tensor)
        or cam.shape != (rgba.shape[0],2,13)
        or cam.dtype != rgba.dtype
        or not isinstance(idx, Tensor) or idx.dtype != torch.long
        or idx.ndim != 3 or idx.shape[:1] != rgba.shape[:1]
        or idx.shape[-1] != 3
        or not isinstance(depth, Tensor) or depth.ndim != 3
        or not isinstance(delta, Tensor) or delta.shape != depth.shape
        or depth.shape[:2] != idx.shape[:2]
        or depth.shape[1] < 2 or depth.shape[1] > 8192
        or depth.shape[2] < 2 or depth.shape[2] > 128
        or depth.dtype != rgba.dtype or delta.dtype != rgba.dtype
        or not isinstance(batch.qualified_rgb_mask, Tensor)
        or not isinstance(batch.qualified_alpha_mask, Tensor)
        or batch.qualified_rgb_mask.shape != idx.shape[:2]
        or batch.qualified_alpha_mask.shape != idx.shape[:2]
        or batch.qualified_rgb_mask.dtype != torch.bool
        or batch.qualified_alpha_mask.dtype != torch.bool
        or len({str(v.device) for v in (
            rgba,cam,idx,depth,delta,
            batch.qualified_rgb_mask,batch.qualified_alpha_mask,
        )}) != 1
    ):
        raise ValueError("RSS_VNEXT_ORIGINAL_RAY_DOMAIN")
    b,_,_,h,w = rgba.shape
    if (
        not all(bool(torch.isfinite(t).all()) for t in (rgba,cam,depth,delta))
        or bool(((rgba < 0) | (rgba > 1)).any())
        or bool((depth < 0).any())
        or bool((delta <= 0).any())
        or bool((depth[...,1:] <= depth[...,:-1]).any())
        or bool((idx[...,0] < 0).any()) or bool((idx[...,0] > 1).any())
        or bool((idx[...,1] < 0).any()) or bool((idx[...,1] >= h).any())
        or bool((idx[...,2] < 0).any()) or bool((idx[...,2] >= w).any())
        or not bool((batch.qualified_rgb_mask | batch.qualified_alpha_mask).any())
    ):
        raise ValueError("RSS_VNEXT_ORIGINAL_RAY_NUMERIC_OR_COVERAGE")
    camera_axes = torch.stack((cam[...,3:6],cam[...,6:9],cam[...,9:12]),dim=-2)
    eye = torch.eye(3,device=cam.device,dtype=cam.dtype)
    if (
        not bool(torch.allclose(
            camera_axes@camera_axes.transpose(-1,-2),
            eye,atol=1e-3,rtol=0,
        ))
        or bool((torch.linalg.det(camera_axes) < .99).any())
        or bool((cam[...,12] <= 0).any())
    ):
        raise ValueError("RSS_VNEXT_ORIGINAL_RAY_CAMERA_NOT_CALIBRATED_ORTHONORMAL")
    # Source FRONT/BACK must both supply qualified observations. Repeated
    # pixels could otherwise overpower a minority source angle.
    for bi in range(b):
        addresses = idx[bi].tolist()
        if len(set(map(tuple,addresses))) != len(addresses):
            raise ValueError("RSS_VNEXT_ORIGINAL_RAY_DUPLICATE_PIXEL")
        for view in (0,1):
            selected = (idx[bi,:,0] == view)
            if not bool(
                (selected & (batch.qualified_rgb_mask[bi] | batch.qualified_alpha_mask[bi])).any()
            ):
                raise ValueError("RSS_VNEXT_ORIGINAL_RAY_BOTH_SOURCE_VIEWS_REQUIRED")

    # Source camera and target are DATA, never trainable geometry or a
    # privileged source XYZ. No gradient may revise their calibration.
    with torch.no_grad():
        index = idx.detach()
        bb = torch.arange(b,device=rgba.device)[:,None]
        view = index[...,0]
        yy = index[...,1]
        xx = index[...,2]
        camera = cam.detach()[bb,view]
        pixel = rgba.detach().permute(0,1,3,4,2)[bb,view,yy,xx]
        u = (xx.to(rgba.dtype) + .5) * (2./w) - 1.
        v = 1. - (yy.to(rgba.dtype) + .5) * (2./h)
        half = camera[...,12]
        ray_origin = (
            camera[...,:3]
            + camera[...,3:6]*(u*half)[...,None]
            + camera[...,6:9]*(v*half)[...,None]
        )
        direction = camera[...,9:12]
        xyz = ray_origin[:,:,None,:] + direction[:,:,None,:]*depth.detach()[...,None]
        alpha = pixel[...,3:4].clamp(0,1)
        white = (pixel[...,:3]*alpha + 1.-alpha).clamp(0,1)
    return RSSvNextOriginalObservedRaySamplesV1(
        sample_xyz=xyz,ray_delta=delta.detach(),ray_depth=depth.detach(),
        ray_direction=direction,rgb_white=white,alpha=alpha,
        qualified_rgb_mask=batch.qualified_rgb_mask.detach(),
        qualified_alpha_mask=batch.qualified_alpha_mask.detach(),
    )


def original_observed_source_photometric_loss_v1(
    predicted_rgb_white: Tensor, predicted_alpha: Tensor,
    target: RSSvNextOriginalObservedRaySamplesV1,
) -> dict[str, Tensor]:
    """Photometric source anchor, not unknown surface or novel-view truth."""
    if (
        predicted_rgb_white.shape != target.rgb_white.shape
        or predicted_alpha.shape != target.alpha.shape
        or not bool(torch.isfinite(predicted_rgb_white).all())
        or not bool(torch.isfinite(predicted_alpha).all())
        or target.source_geometry_is_truth or target.source_depth_is_truth
    ):
        raise ValueError("RSS_VNEXT_ORIGINAL_RAY_PHOTOMETRY_ABI")
    zero=predicted_rgb_white.sum()*0.
    rgb = (
        F.mse_loss(
            predicted_rgb_white[target.qualified_rgb_mask].float(),
            target.rgb_white[target.qualified_rgb_mask].float(),
        )
        if bool(target.qualified_rgb_mask.any()) else zero
    )
    mask = (
        F.mse_loss(
            predicted_alpha[...,0][target.qualified_alpha_mask].float(),
            target.alpha[...,0][target.qualified_alpha_mask].float(),
        )
        if bool(target.qualified_alpha_mask.any()) else zero
    )
    return {"observed_source_rgb_mse":rgb,"observed_source_alpha_mse":mask}


__all__ = [
    "SCHEMA","CAMERA_ABI","RSSvNextOriginalObservedRayBatchV1",
    "RSSvNextOriginalObservedRaySamplesV1",
    "construct_original_observed_ray_samples_v1",
    "original_observed_source_photometric_loss_v1",
]
