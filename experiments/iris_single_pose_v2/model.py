from __future__ import annotations

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int, cap: int = 16) -> int:
    g = min(cap, channels)
    while channels % g:
        g -= 1
    return g


class ConvNormAct(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 3, s: int = 1, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False)
        self.norm = nn.GroupNorm(_groups(cout), cout)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.dw = ConvNormAct(c, c, 5, 1, groups=c)
        self.pw1 = nn.Conv2d(c, 4 * c, 1)
        self.pw2 = nn.Conv2d(4 * c, c, 1)
        self.norm = nn.GroupNorm(_groups(c), c)

    def forward(self, x):
        y = self.dw(x)
        y = self.pw2(F.gelu(self.pw1(y)))
        return x + self.norm(y)


class SharedImageEncoder(nn.Module):
    def __init__(self, in_ch: int = 4, widths=(48, 96, 160, 256)):
        super().__init__()
        c2, c4, c8, c16 = widths
        self.widths = tuple(widths)
        self.s2 = nn.Sequential(ConvNormAct(in_ch, c2, 5, 2), ResidualBlock(c2), ResidualBlock(c2))
        self.s4 = nn.Sequential(ConvNormAct(c2, c4, 3, 2), ResidualBlock(c4), ResidualBlock(c4))
        self.s8 = nn.Sequential(ConvNormAct(c4, c8, 3, 2), ResidualBlock(c8), ResidualBlock(c8))
        self.s16 = nn.Sequential(ConvNormAct(c8, c16, 3, 2), ResidualBlock(c16), ResidualBlock(c16))

    def forward(self, x):
        f2 = self.s2(x)
        f4 = self.s4(f2)
        f8 = self.s8(f4)
        f16 = self.s16(f8)
        return f2, f4, f8, f16


class AxialWithinViewReasoning(nn.Module):
    """Resolution-preserving row/column context at f16."""
    def __init__(self, c: int, heads: int = 8, depth: int = 1):
        super().__init__()
        if c % heads:
            raise ValueError(f"channels {c} must be divisible by heads {heads}")
        row = nn.TransformerEncoderLayer(c, heads, 2 * c, dropout=0, batch_first=True, norm_first=True, activation="gelu")
        col = nn.TransformerEncoderLayer(c, heads, 2 * c, dropout=0, batch_first=True, norm_first=True, activation="gelu")
        self.row = nn.TransformerEncoder(row, num_layers=depth)
        self.col = nn.TransformerEncoder(col, num_layers=depth)
        self.norm = nn.LayerNorm(c)

    def forward(self, x):
        n, c, h, w = x.shape
        z = x.permute(0, 2, 3, 1).contiguous()
        z = self.row(z.reshape(n * h, w, c)).reshape(n, h, w, c)
        z = self.col(z.permute(0, 2, 1, 3).reshape(n * w, h, c))
        z = z.reshape(n, w, h, c).permute(0, 2, 1, 3)
        return self.norm(z).permute(0, 3, 1, 2).contiguous()


class ResolutionIndependentRowFusion(nn.Module):
    """Known-yaw cross-view context on a fixed pooled grid; never scales with native width."""
    def __init__(self, c: int, views: int = 8, context_hw: int = 16, layers: int = 2, heads: int = 8):
        super().__init__()
        if c % heads:
            raise ValueError(f"channels {c} must be divisible by heads {heads}")
        self.views = int(views)
        self.context_hw = int(context_hw)
        self.view_mlp = nn.Sequential(nn.Linear(2, c), nn.GELU(), nn.Linear(c, c))
        self.x_mlp = nn.Sequential(nn.Linear(6, c), nn.GELU(), nn.Linear(c, c))
        layer = nn.TransformerEncoderLayer(c, heads, 2 * c, dropout=0, batch_first=True, norm_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(layer, num_layers=layers)
        self.out_norm = nn.LayerNorm(c)

    def _x_embedding(self, device, dtype):
        w = self.context_hw
        x = 2.0 * (torch.arange(w, device=device, dtype=dtype) + 0.5) / float(w) - 1.0
        f = torch.stack([
            x,
            x * x,
            torch.sin(math.pi * x),
            torch.cos(math.pi * x),
            torch.sin(2.0 * math.pi * x),
            torch.cos(2.0 * math.pi * x),
        ], dim=-1)
        return self.x_mlp(f)

    def forward(self, f16, yaw_deg):
        b, v, c, h, w = f16.shape
        if v != self.views:
            raise ValueError(f"expected {self.views} views, got {v}")
        pooled = F.adaptive_avg_pool2d(f16.reshape(b * v, c, h, w), (self.context_hw, self.context_hw))
        pooled = pooled.reshape(b, v, c, self.context_hw, self.context_hw)
        if yaw_deg.ndim == 1:
            yaw_deg = yaw_deg[None].expand(b, -1)
        t = torch.deg2rad(yaw_deg.float())
        view_e = self.view_mlp(torch.stack([torch.sin(t), torch.cos(t)], dim=-1)).to(pooled.dtype)
        x_e = self._x_embedding(pooled.device, pooled.dtype)
        z = pooled.permute(0, 3, 1, 4, 2).contiguous()
        z = z + view_e[:, None, :, None, :] + x_e[None, None, None, :, :]
        z = z.reshape(b * self.context_hw, v * self.context_hw, c)
        z = self.out_norm(self.tr(z))
        z = z.reshape(b, self.context_hw, v, self.context_hw, c).permute(0, 2, 4, 1, 3).contiguous()
        up = F.interpolate(z.reshape(b * v, c, self.context_hw, self.context_hw), size=(h, w), mode="bilinear", align_corners=False)
        return up.reshape(b, v, c, h, w)


class ContextFuse(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.proj = ConvNormAct(2 * c, c, 3, 1)
        self.res = ResidualBlock(c)

    def forward(self, local, global_context):
        return self.res(self.proj(torch.cat([local, global_context], dim=1)))


class FuseBlock(nn.Module):
    def __init__(self, cin: int, skip: int, cout: int):
        super().__init__()
        self.proj = ConvNormAct(cin + skip, cout, 3, 1)
        self.res = ResidualBlock(cout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.res(self.proj(torch.cat([x, skip], dim=1)))


@dataclass(frozen=True)
class IRISV2Config:
    views: int = 8
    widths: tuple[int, int, int, int] = (48, 96, 160, 256)
    coarse_dim: int = 64
    fine_dim: int = 32
    context_hw: int = 16
    within_heads: int = 8
    cross_heads: int = 8
    cross_layers: int = 2


class IRISSinglePoseV2(nn.Module):
    """Resolution-safe single-pose 8-view observable geometry frontend.

    Roles are deliberately separated:
      P/N/U_geo -> dense geometry evidence at f2
      Z_coarse  -> global/high-recall persistence at f8
      Z_fine    -> local refinement only at f2
    """
    def __init__(self, cfg: IRISV2Config = IRISV2Config()):
        super().__init__()
        self.cfg = cfg
        c2, c4, c8, c16 = cfg.widths
        self.encoder = SharedImageEncoder(4, cfg.widths)
        self.within = AxialWithinViewReasoning(c16, cfg.within_heads, 1)
        self.cross = ResolutionIndependentRowFusion(c16, cfg.views, cfg.context_hw, cfg.cross_layers, cfg.cross_heads)
        self.context_fuse = ContextFuse(c16)
        self.d8 = FuseBlock(c16, c8, c8)
        self.d4 = FuseBlock(c8, c4, c4)
        self.d2 = FuseBlock(c4, c2, c2)
        self.p_head = nn.Conv2d(c2, 3, 1)
        self.n_head = nn.Conv2d(c2, 3, 1)
        self.u_geo_head = nn.Conv2d(c2, 1, 1)
        self.fine_head = nn.Conv2d(c2, cfg.fine_dim, 1)
        self.coarse_head = nn.Conv2d(c8, cfg.coarse_dim, 1)

    def forward(self, images, yaw_deg):
        if images.ndim != 5:
            raise ValueError(f"images must be [B,V,4,H,W], got {tuple(images.shape)}")
        b, v, c, h, w = images.shape
        if v != self.cfg.views or c != 4:
            raise ValueError(f"expected V={self.cfg.views}, RGBA=4; got V={v}, C={c}")
        if h != w or h % 16:
            raise ValueError(f"square resolution divisible by 16 required, got {h}x{w}")
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
        p = self.p_head(y2)
        n = F.normalize(self.n_head(y2), dim=1, eps=1e-8)
        u_geo = torch.clamp(self.u_geo_head(y2), -6.0, 3.0)
        zf = F.normalize(self.fine_head(y2), dim=1, eps=1e-8)

        def rv(t):
            return t.reshape(b, v, *t.shape[1:])

        return {
            "P": rv(p),
            "N": rv(n),
            "U_geo": rv(u_geo),
            "Z_coarse": rv(zc),
            "Z_fine": rv(zf),
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
