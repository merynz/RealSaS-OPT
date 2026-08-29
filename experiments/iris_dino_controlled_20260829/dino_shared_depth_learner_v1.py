from __future__ import annotations

from dataclasses import dataclass, asdict
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


class AxialWithinViewReasoning(nn.Module):
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

    def forward(self, feat, yaw_deg):
        b, v, c, h, w = feat.shape
        if v != self.views:
            raise ValueError(f"expected {self.views} views, got {v}")
        pooled = F.adaptive_avg_pool2d(feat.reshape(b * v, c, h, w), (self.context_hw, self.context_hw))
        pooled = pooled.reshape(b, v, c, self.context_hw, self.context_hw)
        if yaw_deg.ndim == 1:
            yaw_deg = yaw_deg[None].expand(b, -1)
        if tuple(yaw_deg.shape) != (b, v):
            raise ValueError(f"yaw_deg shape drift: {tuple(yaw_deg.shape)}")
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


def fixed_native_patch_offsets(radii=(1, 4, 16)):
    out = [(0, 0)]
    ring = [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]
    for r in radii:
        out.extend((dx * int(r), dy * int(r)) for dx, dy in ring)
    return tuple(out)


def sample_native_rgba_stencil(native_rgba: torch.Tensor, query_xy: torch.Tensor, offsets_px) -> torch.Tensor:
    """Sample a fixed full-native-resolution RGBA stencil at query points.

    native_rgba: [B,V,4,1024,1024], float32 in [0,1]
    query_xy: [B,V,S,2], normalized align_corners=False grid coordinates
    returns: [B,V,S,4*len(offsets_px)]
    """
    if native_rgba.ndim != 5 or native_rgba.shape[2] != 4 or tuple(native_rgba.shape[-2:]) != (1024, 1024):
        raise ValueError(f"native_rgba must be [B,V,4,1024,1024], got {tuple(native_rgba.shape)}")
    if query_xy.ndim != 4 or query_xy.shape[-1] != 2:
        raise ValueError(f"query_xy must be [B,V,S,2], got {tuple(query_xy.shape)}")
    b, v, _, h, w = native_rgba.shape
    if tuple(query_xy.shape[:2]) != (b, v):
        raise ValueError("query/native batch-view mismatch")
    s = query_xy.shape[2]
    base = query_xy.float()
    grids = []
    for dx, dy in offsets_px:
        delta = torch.tensor([2.0 * float(dx) / float(w), 2.0 * float(dy) / float(h)], device=base.device, dtype=base.dtype)
        grids.append(base + delta)
    grid = torch.stack(grids, dim=3).reshape(b * v, s, len(offsets_px), 2)
    x = native_rgba.reshape(b * v, 4, h, w).float()
    samp = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    samp = samp.reshape(b, v, 4, s, len(offsets_px)).permute(0, 1, 3, 4, 2).contiguous()
    return samp.reshape(b, v, s, len(offsets_px) * 4)


def lift_and_normalize_tokens(native_tokens: torch.Tensor, native_dim: int, *, output_dim: int = 1536, eps: float = 1e-12) -> torch.Tensor:
    """Exact preregistered Q_d zero-pad followed by non-affine FP32 token L2 norm."""
    if native_tokens.shape[-1] != int(native_dim):
        raise ValueError(f"native token width drift {native_tokens.shape[-1]} != {native_dim}")
    x = native_tokens.float()
    if native_dim > output_dim:
        raise ValueError("native_dim > output_dim")
    if native_dim < output_dim:
        z = torch.zeros(*x.shape[:-1], output_dim - native_dim, dtype=x.dtype, device=x.device)
        x = torch.cat([x, z], dim=-1)
    return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(float(eps))


@dataclass(frozen=True)
class DINOSharedDepthConfig:
    views: int = 8
    token_hw: int = 37
    token_input_dim: int = 1536
    token_width: int = 256
    context_hw: int = 16
    within_heads: int = 8
    cross_heads: int = 8
    cross_layers: int = 2
    native_patch_radii: tuple[int, int, int] = (1, 4, 16)
    native_patch_hidden: int = 128
    decoder_hidden1: int = 256
    decoder_hidden2: int = 128


class DINOSharedDepthLearnerV1(nn.Module):
    def __init__(self, cfg: DINOSharedDepthConfig = DINOSharedDepthConfig()):
        super().__init__()
        self.cfg = cfg
        c = cfg.token_width
        self.token_stem = nn.Sequential(ConvNormAct(cfg.token_input_dim, c, 1, 1), ResidualBlock(c))
        self.within = AxialWithinViewReasoning(c, cfg.within_heads, 1)
        self.cross = ResolutionIndependentRowFusion(c, cfg.views, cfg.context_hw, cfg.cross_layers, cfg.cross_heads)
        self.context_fuse = ContextFuse(c)
        self.native_offsets_px = fixed_native_patch_offsets(cfg.native_patch_radii)
        native_in = 4 * len(self.native_offsets_px)
        self.native_encoder = nn.Sequential(nn.LayerNorm(native_in), nn.Linear(native_in, cfg.native_patch_hidden), nn.GELU(), nn.Linear(cfg.native_patch_hidden, cfg.native_patch_hidden), nn.GELU())
        qin = c + cfg.native_patch_hidden + 5
        self.depth_decoder = nn.Sequential(nn.LayerNorm(qin), nn.Linear(qin, cfg.decoder_hidden1), nn.GELU(), nn.Linear(cfg.decoder_hidden1, cfg.decoder_hidden2), nn.GELU(), nn.Linear(cfg.decoder_hidden2, 1))

    def config_dict(self):
        d = asdict(self.cfg)
        d["native_offsets_px"] = [list(x) for x in self.native_offsets_px]
        return d

    def encode_tokens(self, lifted_tokens: torch.Tensor, yaw_deg: torch.Tensor) -> torch.Tensor:
        if lifted_tokens.ndim != 4:
            raise ValueError(f"lifted_tokens must be [B,V,T,1536], got {tuple(lifted_tokens.shape)}")
        b, v, t, d = lifted_tokens.shape
        if v != self.cfg.views or t != self.cfg.token_hw * self.cfg.token_hw or d != self.cfg.token_input_dim:
            raise ValueError(f"token contract drift {tuple(lifted_tokens.shape)}")
        z = lifted_tokens.reshape(b * v, self.cfg.token_hw, self.cfg.token_hw, d).permute(0, 3, 1, 2).contiguous()
        local = self.within(self.token_stem(z)).reshape(b, v, self.cfg.token_width, self.cfg.token_hw, self.cfg.token_hw)
        global_ctx = self.cross(local, yaw_deg)
        fused = self.context_fuse(local.reshape(b * v, self.cfg.token_width, self.cfg.token_hw, self.cfg.token_hw), global_ctx.reshape(b * v, self.cfg.token_width, self.cfg.token_hw, self.cfg.token_hw))
        return fused.reshape(b, v, self.cfg.token_width, self.cfg.token_hw, self.cfg.token_hw)

    def query_depth_from_encoded(self, encoded: torch.Tensor, native_rgba: torch.Tensor, query_xy: torch.Tensor, yaw_deg: torch.Tensor, sheet_half_extent: torch.Tensor) -> torch.Tensor:
        b, v, c, h, w = encoded.shape
        if (v, c, h, w) != (self.cfg.views, self.cfg.token_width, self.cfg.token_hw, self.cfg.token_hw):
            raise ValueError(f"encoded contract drift {tuple(encoded.shape)}")
        if query_xy.ndim != 4 or tuple(query_xy.shape[:2]) != (b, v) or query_xy.shape[-1] != 2:
            raise ValueError(f"query_xy drift {tuple(query_xy.shape)}")
        s = query_xy.shape[2]
        grid = query_xy.reshape(b * v, s, 1, 2).to(encoded.dtype)
        tok = F.grid_sample(encoded.reshape(b * v, c, h, w), grid, mode="bilinear", padding_mode="border", align_corners=False)
        tok = tok.squeeze(-1).transpose(1, 2).reshape(b, v, s, c)
        native = self.native_encoder(sample_native_rgba_stencil(native_rgba, query_xy, self.native_offsets_px))
        if yaw_deg.ndim == 1:
            yaw_deg = yaw_deg[None].expand(b, -1)
        yaw = torch.deg2rad(yaw_deg.float())
        if sheet_half_extent.ndim == 1:
            sheet_half_extent = sheet_half_extent[None].expand(b, -1)
        if tuple(sheet_half_extent.shape) != (b, v):
            raise ValueError(f"half extent drift {tuple(sheet_half_extent.shape)}")
        cam = torch.stack([torch.sin(yaw), torch.cos(yaw), sheet_half_extent.float()], dim=-1)
        cam = cam[:, :, None, :].expand(-1, -1, s, -1)
        q = torch.cat([query_xy.float(), cam], dim=-1).to(tok.dtype)
        z = torch.cat([tok, native.to(tok.dtype), q], dim=-1)
        return self.depth_decoder(z).squeeze(-1)

    def forward(self, lifted_tokens: torch.Tensor, native_rgba: torch.Tensor, query_xy: torch.Tensor, yaw_deg: torch.Tensor, sheet_half_extent: torch.Tensor) -> torch.Tensor:
        return self.query_depth_from_encoded(self.encode_tokens(lifted_tokens, yaw_deg), native_rgba, query_xy, yaw_deg, sheet_half_extent)


def depth_truth_from_points(points: torch.Tensor, camera_forward: torch.Tensor) -> torch.Tensor:
    return torch.sum(points.float() * camera_forward.float(), dim=-1)


def depth_loss(pred_d: torch.Tensor, truth_d: torch.Tensor, beta: float = 0.01) -> torch.Tensor:
    return F.smooth_l1_loss(pred_d.float(), truth_d.float(), beta=float(beta), reduction="mean")
