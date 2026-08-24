from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ConvNormAct, ResidualBlock, IRISV2Config
from model_pv5 import (
    IRISSinglePoseV2PV5,
    validate_observable_sheet_half_extent,
)
from model_pv4 import reconstruct_p_from_observable_scale


class IRISSinglePoseV2PV5R256(IRISSinglePoseV2PV5):
    """P-V5 with a true full-resolution image-conditioned P/depth branch.

    The old V5 path predicts depth on y2 (R/2 for an R input). Merely resizing that
    scalar field to R would preserve the old band-limit. This subclass instead
    consumes raw learner-resolution RGBA through a full-resolution stem, fuses it
    with upsampled y2 features at R, and predicts depth directly on that R grid.
    """

    def __init__(self, cfg: IRISV2Config = IRISV2Config(), full_width: int = 32):
        super().__init__(cfg)
        c2 = cfg.widths[0]
        fw = int(full_width)
        if fw < 8:
            raise ValueError("full_width must be >= 8")
        self.p_s1_stem = nn.Sequential(
            ConvNormAct(4, fw, 3, 1),
            ResidualBlock(fw),
        )
        self.p_s1_fuse = nn.Sequential(
            ConvNormAct(c2 + fw, c2, 3, 1),
            ResidualBlock(c2),
        )
        self.p_depth_head = nn.Conv2d(c2 + 5, 1, 1)

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
        with torch.no_grad():
            zc = F.normalize(self.coarse_head(y8), dim=1, eps=1e-8)
        y4 = self.d4(y8, f4)
        y2 = self.d2(y4, f2)

        p_img_r = self.p_s1_stem(x)
        y2_r = F.interpolate(y2, size=(h, w), mode="bilinear", align_corners=False)
        p_feat_r = self.p_s1_fuse(torch.cat([y2_r, p_img_r], dim=1))
        depth = self.p_depth_head(
            self._depth_conditioning_v5(p_feat_r, yaw_deg, observable_h, b, v)
        )
        if depth.shape[-2:] != (h, w):
            raise RuntimeError(f"R256 depth branch drift: {tuple(depth.shape[-2:])} != {(h,w)}")
        depth_v = depth.reshape(b, v, 1, h, w)
        p = reconstruct_p_from_observable_scale(depth_v, yaw_deg, observable_h)

        with torch.no_grad():
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
            "_P_FULLRES_FEATURE": p_feat_r.reshape(b, v, *p_feat_r.shape[1:]),
        }
