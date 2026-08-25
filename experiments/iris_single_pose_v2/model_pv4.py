from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from coords import field_cell_centers
from model import IRISSinglePoseV2, IRISV2Config, camera_basis_from_yaw


ALPHA_THRESHOLD = 0.5


def _bbox_span(mask: torch.Tensor, coordinate: str) -> torch.Tensor:
    """Inclusive foreground span in pixels for mask [B,H,W].

    coordinate='x' collapses rows and measures width; coordinate='y' collapses
    columns and measures height. The explicit name prevents H/W axis ambiguity.
    """
    if mask.ndim != 3:
        raise ValueError(f"mask must be [B,H,W], got {tuple(mask.shape)}")
    if coordinate == "x":
        occupied = mask.any(dim=1)  # [B,W]
    elif coordinate == "y":
        occupied = mask.any(dim=2)  # [B,H]
    else:
        raise ValueError(f"coordinate must be 'x' or 'y', got {coordinate!r}")
    if not occupied.any(dim=1).all():
        raise RuntimeError("blank alpha support in canonical sheet-scale estimator")
    n = occupied.shape[1]
    idx = torch.arange(n, device=mask.device, dtype=torch.long)[None]
    lo = torch.where(occupied, idx, torch.full_like(idx, n)).min(dim=1).values
    hi = torch.where(occupied, idx, torch.full_like(idx, -1)).max(dim=1).values
    return (hi - lo + 1).to(torch.float32)


def estimate_sheet_half_extent_from_alpha(
    images: torch.Tensor,
    yaw_deg: torch.Tensor | None = None,
    alpha_threshold: float = ALPHA_THRESHOLD,
) -> torch.Tensor:
    """Derive the canonical orthographic sheet scale from RGBA only.

    Canonical P gauge defines max(object bbox extent x/y/z)=1. Ordered V0 and V2
    expose x/y horizontal extents exactly; image height exposes z. Therefore:

      half_extent = 1 / (2 * max(V0 width occupancy, V2 width occupancy, z occupancy))

    No camera metadata or teacher geometry is consumed. Returned shape is [B].
    The operation is intentionally non-differentiable: it is deterministic input
    canonicalization, not a learned latent.
    """
    if images.ndim != 5 or images.shape[1] != 8 or images.shape[2] != 4:
        raise ValueError(f"images must be [B,8,4,H,W], got {tuple(images.shape)}")
    b, _, _, h, w = images.shape
    if h != w:
        raise ValueError(f"square canonical views required, got {h}x{w}")
    if yaw_deg is not None:
        y = yaw_deg.float()
        if y.ndim == 1:
            y = y[None].expand(b, -1)
        expected = torch.arange(8, device=y.device, dtype=y.dtype)[None] * 45.0
        if tuple(y.shape) != (b, 8) or not torch.allclose(y, expected.expand_as(y), atol=1e-4, rtol=0):
            raise RuntimeError("canonical ordered yaw contract required for image-derived sheet scale")

    with torch.no_grad():
        alpha = images[:, :, 3].float()
        fg = alpha >= float(alpha_threshold)
        # Canonical X/Y extents are directly visible as horizontal occupancy in yaw 0/90.
        width_v0 = _bbox_span(fg[:, 0], "x") / float(w)
        width_v2 = _bbox_span(fg[:, 2], "x") / float(w)
        # Z is camera-up for every view; use maximum vertical occupancy for robustness.
        heights = torch.stack([_bbox_span(fg[:, v], "y") / float(h) for v in range(8)], dim=1)
        z_occ = heights.max(dim=1).values
        max_occ = torch.stack([width_v0, width_v2, z_occ], dim=1).max(dim=1).values
        if not torch.isfinite(max_occ).all() or (max_occ <= 0).any():
            raise RuntimeError("invalid alpha occupancy in sheet-scale estimator")
        return (0.5 / max_occ).to(images.device)


def reconstruct_p_from_observable_scale(depth: torch.Tensor, yaw_deg: torch.Tensor, half_extent: torch.Tensor) -> torch.Tensor:
    if half_extent.ndim == 0:
        he = half_extent.reshape(1)
    else:
        he = half_extent
    if he.ndim != 1 or he.shape[0] != depth.shape[0]:
        raise ValueError(f"half_extent must be [B], got {tuple(he.shape)} for depth {tuple(depth.shape)}")
    b, v, _, h, w = depth.shape
    right, up, forward = camera_basis_from_yaw(yaw_deg, batch=b, views=v, dtype=depth.dtype)
    right = right[..., :, None, None]
    up = up[..., :, None, None]
    forward = forward[..., :, None, None]
    grid = field_cell_centers(h, w, device=depth.device, dtype=depth.dtype)
    gx = grid[..., 0][None, None, None]
    gy = grid[..., 1][None, None, None]
    scale = he.to(depth.dtype)[:, None, None, None, None]
    return scale * gx * right - scale * gy * up + depth * forward


class IRISSinglePoseV2PV4(IRISSinglePoseV2):
    """P-V4 candidate: image-only observable sheet scale + learned view depth."""

    def __init__(self, cfg: IRISV2Config = IRISV2Config()):
        super().__init__(cfg)
        c2 = cfg.widths[0]
        # gx, gy, sin(yaw), cos(yaw), observable sheet half-extent.
        self.p_depth_head = nn.Conv2d(c2 + 5, 1, 1)

    def _depth_conditioning_v4(
        self,
        y2: torch.Tensor,
        yaw_deg: torch.Tensor,
        half_extent: torch.Tensor,
        b: int,
        v: int,
    ) -> torch.Tensor:
        _, _, h, w = y2.shape
        grid = field_cell_centers(h, w, device=y2.device, dtype=y2.dtype).permute(2, 0, 1)
        grid = grid[None].expand(b * v, -1, -1, -1)
        yaw = yaw_deg.float()
        if yaw.ndim == 1:
            yaw = yaw[None].expand(b, -1)
        if tuple(yaw.shape) != (b, v):
            raise ValueError(f"yaw_deg must resolve to {(b, v)}, got {tuple(yaw.shape)}")
        t = torch.deg2rad(yaw).reshape(b * v, 1, 1, 1)
        yaw_feat = torch.cat([torch.sin(t), torch.cos(t)], dim=1).to(y2.dtype).expand(-1, -1, h, w)
        scale_feat = half_extent.to(y2.dtype)[:, None, None, None].expand(b, v, h, w).reshape(b * v, 1, h, w)
        return torch.cat([y2, grid, yaw_feat, scale_feat], dim=1)

    def forward(self, images, yaw_deg):
        if images.ndim != 5:
            raise ValueError(f"images must be [B,V,4,H,W], got {tuple(images.shape)}")
        b, v, c, h, w = images.shape
        if v != self.cfg.views or c != 4:
            raise ValueError(f"expected V={self.cfg.views}, RGBA=4; got V={v}, C={c}")
        if h != w or h % 16:
            raise ValueError(f"square resolution divisible by 16 required, got {h}x{w}")

        half_extent = estimate_sheet_half_extent_from_alpha(images, yaw_deg)
        x = images.reshape(b * v, c, h, w)
        f2, f4, f8, f16 = self.encoder(x)
        f16_local = self.within(f16).reshape(b, v, f16.shape[1], f16.shape[2], f16.shape[3])
        f16_global = self.cross(f16_local, yaw_deg)
        fused16 = self.context_fuse(
            f16_local.reshape(b * v, f16.shape[1], f16.shape[2], f16.shape[3]),
            f16_global.reshape(b * v, f16.shape[1], f16.shape[2], f16.shape[3]),
        )
        y8 = self.d8(fused16, f8)
        zc = F.normalize(self.coarse_head(y8), dim=1, eps=1e-8)
        y4 = self.d4(y8, f4)
        y2 = self.d2(y4, f2)
        depth = self.p_depth_head(self._depth_conditioning_v4(y2, yaw_deg, half_extent, b, v))
        depth_v = depth.reshape(b, v, 1, depth.shape[-2], depth.shape[-1])
        p = reconstruct_p_from_observable_scale(depth_v, yaw_deg, half_extent)
        n = F.normalize(self.n_head(y2), dim=1, eps=1e-8)
        u_geo = torch.clamp(self.u_geo_head(y2), -6.0, 3.0)
        zf = F.normalize(self.fine_head(y2), dim=1, eps=1e-8)

        def rv(t):
            return t.reshape(b, v, *t.shape[1:])

        return {
            "P": p,
            "N": rv(n),
            "U_geo": rv(u_geo),
            "Z_coarse": rv(zc),
            "Z_fine": rv(zf),
        }
