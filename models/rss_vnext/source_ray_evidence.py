from __future__ import annotations

"""Two observed-view orthographic ray evidence, not invented 3D supervision.

Inspired by StdGEN's source-image appearance/silhouette supervision, but
deliberately avoids StdGEN's named hair/cloth/body-part ontology and forced
global-signed-solid prior.

Important epistemic distinction:
  sampled RGBA means "the pixel on this camera ray has this value", NOT
  "this candidate XYZ is the observed surface point". Without qualified
  source depth, hidden positions along that ray remain unobserved.

This deterministic module is a *non-compressing access path* back to the
original FRONT/BACK images for all shared-carrier experts and mesh vertices.
No source-ray color is assigned canonical material/texture authority here.
"""
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass
class RSSvNextSourceRayEvidence:
    rgba: Tensor                 # [B,N,2,4], source camera pixels
    projection_valid: Tensor     # [B,N,2] under supplied camera
    silhouette_alpha: Tensor     # [B,N,2] image alpha along candidate ray
    source_view_count: int = 2
    ray_depth_authority: str = "UNKNOWN_NO_SOURCE_DEPTH"
    appearance_authority: str = "ORIGINAL_RGBA_CAMERA_RAY_NOT_POINT_TEXTURE"

    def validate(self, *, batch: int, points: int) -> None:
        if self.rgba.shape != (batch, points, 2, 4):
            raise ValueError("RSS_VNEXT_SOURCE_RAY_RGBA_SHAPE")
        if self.projection_valid.shape != (batch, points, 2):
            raise ValueError("RSS_VNEXT_SOURCE_RAY_VALID_SHAPE")
        if self.projection_valid.dtype != torch.bool:
            raise ValueError("RSS_VNEXT_SOURCE_RAY_VALID_DTYPE")
        if self.silhouette_alpha.shape != (batch, points, 2):
            raise ValueError("RSS_VNEXT_SOURCE_RAY_ALPHA_SHAPE")
        if not bool(torch.isfinite(self.rgba).all()):
            raise ValueError("RSS_VNEXT_SOURCE_RAY_NONFINITE")
        if self.source_view_count != 2 or self.ray_depth_authority != "UNKNOWN_NO_SOURCE_DEPTH":
            raise ValueError("RSS_VNEXT_SOURCE_RAY_AUTHORITY_DRIFT")


def sample_front_back_source_rays(
    observed_rgba: Tensor,
    camera_features: Tensor,
    points_xyz: Tensor,
) -> RSSvNextSourceRayEvidence:
    """Project arbitrary 3D queries to original front/back RGBA camera rays.

    Camera 13D contract: origin xyz(3), right xyz(3), up xyz(3),
    forward xyz(3), orthographic half-extent(1). This is a *projection*
    contract, not unqualified inference of camera calibration.

    Screen-outside queries receive zero evidence and a false validity mask.
    The query's actual depth along the ray is NOT source depth, and must not
    qualify surface visibility, contact, material ownership or solidity.
    """
    if observed_rgba.ndim != 5 or observed_rgba.shape[1:3] != (2, 4):
        raise ValueError("RSS_VNEXT_SOURCE_RAY_ONLY_FRONT_BACK_RGBA")
    b, _, _, h, w = observed_rgba.shape
    if points_xyz.ndim != 3 or points_xyz.shape[:1] != (b,) or points_xyz.shape[-1] != 3:
        raise ValueError("RSS_VNEXT_SOURCE_RAY_POINT_SHAPE")
    if camera_features.shape != (b, 2, 13):
        raise ValueError("RSS_VNEXT_SOURCE_RAY_CAMERA_SHAPE")
    if not all(bool(torch.isfinite(x).all()) for x in
               (observed_rgba, camera_features, points_xyz)):
        raise ValueError("RSS_VNEXT_SOURCE_RAY_NONFINITE_INPUT")
    if not all(torch.is_floating_point(x) for x in
               (observed_rgba, camera_features, points_xyz)):
        raise ValueError("RSS_VNEXT_SOURCE_RAY_FLOAT_INPUT_REQUIRED")
    if observed_rgba.device != camera_features.device or observed_rgba.device != points_xyz.device:
        raise ValueError("RSS_VNEXT_SOURCE_RAY_DEVICE_DRIFT")

    camera = camera_features
    right = camera[..., 3:6]
    up = camera[..., 6:9]
    forward = camera[..., 9:12]
    # Do not quietly normalize a zero/invalid basis into a plausible camera.
    axes = torch.stack((right, up, forward), dim=-2)
    norms = axes.norm(dim=-1)
    if bool((norms - 1).abs().max() > .03):
        raise ValueError("RSS_VNEXT_SOURCE_RAY_CAMERA_AXIS_NORM")
    orthogonality = torch.matmul(axes, axes.transpose(-1, -2))
    identity = torch.eye(3, device=axes.device, dtype=axes.dtype)
    if bool((orthogonality - identity).abs().max() > .03):
        raise ValueError("RSS_VNEXT_SOURCE_RAY_CAMERA_ORTHONORMAL")
    half = camera[..., 12]
    if bool((half <= 0).any()):
        raise ValueError("RSS_VNEXT_SOURCE_RAY_HALF_EXTENT")
    relative = points_xyz[:, :, None, :] - camera[:, None, :, :3]
    u = (relative * right[:, None]).sum(dim=-1) / half[:, None]
    v = (relative * up[:, None]).sum(dim=-1) / half[:, None]
    projected = torch.stack((u, v), dim=-1)
    valid = (projected.abs() <= 1).all(dim=-1)

    source = observed_rgba.reshape(b * 2, 4, h, w)
    # grid_sample takes [B,HO,WO,2]. Preserve exact RGB/alpha pixel evidence.
    grid = projected.permute(0, 2, 1, 3).reshape(b * 2, -1, 1, 2)
    sampled = F.grid_sample(
        source, grid, mode="bilinear", align_corners=False,
        padding_mode="zeros",
    )
    n = points_xyz.shape[1]
    rays = sampled.reshape(b, 2, 4, n).permute(0, 3, 1, 2).contiguous()
    rays = torch.where(valid[..., None], rays, torch.zeros_like(rays))
    out = RSSvNextSourceRayEvidence(
        rgba=rays, projection_valid=valid,
        silhouette_alpha=rays[..., 3],
    )
    out.validate(batch=b, points=n)
    return out


__all__ = ["RSSvNextSourceRayEvidence", "sample_front_back_source_rays"]
