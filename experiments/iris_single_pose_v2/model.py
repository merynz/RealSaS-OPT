from __future__ import annotations

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from coords import field_cell_centers


def _groups(channels: int, cap: int = 16) -> int:
    g = min(cap, channels)
    while channels % g:
        g -= 1
    return g


def camera_basis_from_yaw(yaw_deg: torch.Tensor, *, batch: int | None = None, views: int | None = None, dtype=None):
    """Controlled-camera basis for `realsas.level_orthographic_z_orbit.v1`.

    yaw=0: right=(1,0,0), forward=(0,1,0), up=(0,0,1).
    Positive yaw rotates the camera orbit around +Z.
    """
    yaw = yaw_deg.float()
    if yaw.ndim == 1:
        if batch is None:
            batch = 1
        yaw = yaw[None].expand(int(batch), -1)
    if yaw.ndim != 2:
        raise ValueError(f"yaw_deg must be [V] or [B,V], got {tuple(yaw.shape)}")
    if batch is not None and yaw.shape[0] != int(batch):
        raise ValueError(f"yaw batch mismatch: {tuple(yaw.shape)} vs B={batch}")
    if views is not None and yaw.shape[1] != int(views):
        raise ValueError(f"yaw view mismatch: {tuple(yaw.shape)} vs V={views}")
    t = torch.deg2rad(yaw)
    s, c = torch.sin(t), torch.cos(t)
    z = torch.zeros_like(s)
    right = torch.stack([c, -s, z], dim=-1)
    forward = torch.stack([s, c, z], dim=-1)
    up = torch.stack([z, z, torch.ones_like(z)], dim=-1)
    if dtype is not None:
        right = right.to(dtype)
        forward = forward.to(dtype)
        up = up.to(dtype)
    return right, up, forward


def reconstruct_p_from_view_depth(
    depth: torch.Tensor,
    yaw_deg: torch.Tensor,
    half_extent: float = 0.54,
) -> torch.Tensor:
    """Reconstruct canonical P from exact orthographic screen coordinates + learned view depth.

    depth: [B,V,1,H,W], representing P dot camera_forward.

    For output-cell grid coordinate g=(gx,gy):
      P dot right = half_extent * gx
      P dot up    = -half_extent * gy
      P dot forward = depth

    The first two coordinates are therefore analytic camera geometry, never neural regression.
    """
    if depth.ndim != 5 or depth.shape[2] != 1:
        raise ValueError(f"depth must be [B,V,1,H,W], got {tuple(depth.shape)}")
    b, v, _, h, w = depth.shape
    right, up, forward = camera_basis_from_yaw(
        yaw_deg, batch=b, views=v, dtype=depth.dtype
    )
    right = right[..., :, None, None]
    up = up[..., :, None, None]
    forward = forward[..., :, None, None]
    grid = field_cell_centers(h, w, device=depth.device, dtype=depth.dtype)
    gx = grid[..., 0][None, None, None]
    gy = grid[..., 1][None, None, None]
    screen = float(half_extent) * gx * right - float(half_extent) * gy * up
    return screen + depth * forward


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
    orthographic_half_extent: float = 0.54


class IRISSinglePoseV2(nn.Module):
    """Resolution-safe single-pose 8-view observable geometry frontend.

    Roles are deliberately separated:
      P -> analytic screen-plane geometry + one learned camera-forward depth scalar at f2
      N/U_geo -> dense geometry evidence at f2
      Z_coarse -> global/high-recall persistence at f8
      Z_fine -> local refinement only at f2
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
        # P is not a free 3-vector. The head predicts only camera-forward depth.
        # Normalized gx/gy and sin/cos(yaw) are explicit conditioning; screen-plane P is analytic.
        self.p_depth_head = nn.Conv2d(c2 + 4, 1, 1)
        self.n_head = nn.Conv2d(c2, 3, 1)
        self.u_geo_head = nn.Conv2d(c2, 1, 1)
        self.fine_head = nn.Conv2d(c2, cfg.fine_dim, 1)
        self.coarse_head = nn.Conv2d(c8, cfg.coarse_dim, 1)

    def _depth_conditioning(self, y2: torch.Tensor, yaw_deg: torch.Tensor, b: int, v: int) -> torch.Tensor:
        _, _, h, w = y2.shape
        grid = field_cell_centers(h, w, device=y2.device, dtype=y2.dtype).permute(2, 0, 1)
        grid = grid[None].expand(b * v, -1, -1, -1)
        yaw = yaw_deg.float()
        if yaw.ndim == 1:
            yaw = yaw[None].expand(b, -1)
        if tuple(yaw.shape) != (b, v):
            raise ValueError(f"yaw_deg must resolve to {(b, v)}, got {tuple(yaw.shape)}")
        t = torch.deg2rad(yaw).reshape(b * v, 1, 1, 1)
        yaw_feat = torch.cat([torch.sin(t), torch.cos(t)], dim=1).to(y2.dtype)
        yaw_feat = yaw_feat.expand(-1, -1, h, w)
        return torch.cat([y2, grid, yaw_feat], dim=1)

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
        depth = self.p_depth_head(self._depth_conditioning(y2, yaw_deg, b, v))
        depth_v = depth.reshape(b, v, 1, depth.shape[-2], depth.shape[-1])
        p = reconstruct_p_from_view_depth(depth_v, yaw_deg, self.cfg.orthographic_half_extent)
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


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
