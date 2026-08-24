from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from coords import field_cell_centers
from model import IRISSinglePoseV2, IRISV2Config
from model_pv4 import estimate_sheet_half_extent_from_alpha, reconstruct_p_from_observable_scale


def estimate_native_sheet_half_extent(native_images: torch.Tensor, yaw_deg: torch.Tensor) -> torch.Tensor:
    """Image-only canonical gauge, measured once on the original native RGBA sheet.

    This helper intentionally delegates to the frozen V4 alpha-support definition,
    but it changes the *execution boundary*: the value must be computed before any
    learner resize and then transported unchanged. No camera metadata is accepted.
    """
    if native_images.ndim != 5 or native_images.shape[1] != 8 or native_images.shape[2] != 4:
        raise ValueError(f"native_images must be [B,8,4,H,W], got {tuple(native_images.shape)}")
    if native_images.shape[-2:] != (1024, 1024):
        raise ValueError(f"native product sheet must be 1024x1024 per view, got {tuple(native_images.shape[-2:])}")
    return estimate_sheet_half_extent_from_alpha(native_images, yaw_deg).detach()


def validate_observable_sheet_half_extent(sheet_half_extent: torch.Tensor, batch: int, device) -> torch.Tensor:
    h = torch.as_tensor(sheet_half_extent, device=device, dtype=torch.float32)
    if h.ndim == 0:
        h = h.reshape(1)
    if tuple(h.shape) != (batch,):
        raise ValueError(f"sheet_half_extent must be [B], got {tuple(h.shape)} for B={batch}")
    if not torch.isfinite(h).all() or (h <= 0).any():
        raise RuntimeError("sheet_half_extent must be finite and positive")
    if h.requires_grad:
        raise RuntimeError("sheet_half_extent is deterministic preprocessing authority, not a learned latent")
    return h


class IRISSinglePoseV2PV5(IRISSinglePoseV2):
    """P-V5: native-image gauge is computed once, transported through learner resize.

    Overall extractor remains image-only. `sheet_half_extent` is legal only when it
    is produced by `estimate_native_sheet_half_extent` from the original 1024 RGBA
    sheet. Teacher camera half-extent is not an accepted argument anywhere here.
    """

    def __init__(self, cfg: IRISV2Config = IRISV2Config()):
        super().__init__(cfg)
        c2 = cfg.widths[0]
        # gx, gy, sin(yaw), cos(yaw), native-image-derived sheet half-extent.
        self.p_depth_head = nn.Conv2d(c2 + 5, 1, 1)

    def _depth_conditioning_v5(
        self,
        y2: torch.Tensor,
        yaw_deg: torch.Tensor,
        sheet_half_extent: torch.Tensor,
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
        scale_feat = sheet_half_extent.to(y2.dtype)[:, None, None, None].expand(b, v, h, w).reshape(b * v, 1, h, w)
        return torch.cat([y2, grid, yaw_feat, scale_feat], dim=1)

    def forward(self, images, yaw_deg, sheet_half_extent):
        if images.ndim != 5:
            raise ValueError(f"images must be [B,V,4,H,W], got {tuple(images.shape)}")
        b, v, c, h, w = images.shape
        if v != self.cfg.views or c != 4:
            raise ValueError(f"expected V={self.cfg.views}, RGBA=4; got V={v}, C={c}")
        if h != w or h % 16:
            raise ValueError(f"square resolution divisible by 16 required, got {h}x{w}")
        observable_h = validate_observable_sheet_half_extent(sheet_half_extent, b, images.device)

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
        depth = self.p_depth_head(self._depth_conditioning_v5(y2, yaw_deg, observable_h, b, v))
        depth_v = depth.reshape(b, v, 1, depth.shape[-2], depth.shape[-1])
        p = reconstruct_p_from_observable_scale(depth_v, yaw_deg, observable_h)
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
