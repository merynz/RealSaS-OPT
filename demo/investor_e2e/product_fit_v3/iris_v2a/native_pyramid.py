from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


def _groups(c: int, cap: int = 16) -> int:
    g = min(int(cap), int(c))
    while c % g:
        g -= 1
    return g


class ConvGNAct(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 3, stride: int = 1, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups, bias=False)
        self.norm = nn.GroupNorm(_groups(cout), cout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class NativeResidual(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.dw = ConvGNAct(c, c, 5, 1, groups=c)
        self.pw1 = nn.Conv2d(c, 4 * c, 1)
        self.pw2 = nn.Conv2d(4 * c, c, 1)
        self.norm = nn.GroupNorm(_groups(c), c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)
        y = self.pw2(F.gelu(self.pw1(y)))
        return x + self.norm(y)


class NativeStage(nn.Module):
    def __init__(self, cin: int, cout: int, blocks: int):
        super().__init__()
        self.down = ConvGNAct(cin, cout, 3, 2)
        self.blocks = nn.Sequential(*[NativeResidual(cout) for _ in range(int(blocks))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.down(x))


@dataclass(frozen=True)
class NativePyramidConfig:
    channels: tuple[int, int, int, int] = (32, 64, 128, 192)
    blocks: tuple[int, int, int, int] = (2, 2, 3, 3)
    views: int = 8
    native_resolution: int = 1024


@dataclass
class NativePyramidOutput:
    f2: torch.Tensor     # [B,V,C2,512,512]
    f4: torch.Tensor     # [B,V,C4,256,256]
    f8: torch.Tensor     # [B,V,C8,128,128]
    f16: torch.Tensor    # [B,V,C16,64,64]

    def levels(self) -> tuple[torch.Tensor, ...]:
        return (self.f2, self.f4, self.f8, self.f16)


class Native1024SharedPyramid(nn.Module):
    """Shared high-resolution appearance/detail path for all eight views.

    No view-specific parameters exist.  The module sees authoritative native RGBA
    and retains sub-DINO-patch detail through the 1/2 and 1/4 levels.
    """

    def __init__(self, cfg: NativePyramidConfig = NativePyramidConfig()):
        super().__init__()
        self.cfg = cfg
        c2, c4, c8, c16 = cfg.channels
        b2, b4, b8, b16 = cfg.blocks
        self.s2 = NativeStage(4, c2, b2)
        self.s4 = NativeStage(c2, c4, b4)
        self.s8 = NativeStage(c4, c8, b8)
        self.s16 = NativeStage(c8, c16, b16)

    def forward(self, rgba: torch.Tensor) -> NativePyramidOutput:
        if rgba.ndim != 5:
            raise ValueError(f"NATIVE_RGBA_RANK:{rgba.ndim}")
        b, v, c, h, w = rgba.shape
        if (v, c, h, w) != (self.cfg.views, 4, self.cfg.native_resolution, self.cfg.native_resolution):
            raise ValueError(f"NATIVE_RGBA_CONTRACT:{tuple(rgba.shape)}")
        if not torch.isfinite(rgba).all():
            raise ValueError("NATIVE_RGBA_NONFINITE")
        x = rgba.reshape(b * v, c, h, w)
        f2 = self.s2(x)
        f4 = self.s4(f2)
        f8 = self.s8(f4)
        f16 = self.s16(f8)

        def pack(z: torch.Tensor) -> torch.Tensor:
            return z.reshape(b, v, z.shape[1], z.shape[2], z.shape[3])

        return NativePyramidOutput(pack(f2), pack(f4), pack(f8), pack(f16))


def sample_multiview_feature_map(feature: torch.Tensor, grid_xy: torch.Tensor) -> torch.Tensor:
    """Exact bilinear sampling of one [B,V,C,H,W] map at q projections.

    grid_xy: [B,V,Q,2], align_corners=False normalized coordinates.
    returns: [B,V,Q,C]
    """
    if feature.ndim != 5 or grid_xy.ndim != 4 or grid_xy.shape[-1] != 2:
        raise ValueError("FEATURE_OR_GRID_SHAPE")
    b, v, c, h, w = feature.shape
    if grid_xy.shape[:2] != (b, v):
        raise ValueError("FEATURE_GRID_BATCH_VIEW_MISMATCH")
    q = grid_xy.shape[2]
    g = grid_xy.reshape(b * v, q, 1, 2).to(feature.dtype)
    y = F.grid_sample(
        feature.reshape(b * v, c, h, w),
        g,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return y.squeeze(-1).transpose(1, 2).reshape(b, v, q, c)


def sample_native_pyramid(pyr: NativePyramidOutput, grid_xy: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return tuple(sample_multiview_feature_map(level, grid_xy) for level in pyr.levels())


def fixed_native_patch_offsets(radii: tuple[int, ...] = (1, 4, 16)) -> tuple[tuple[int, int], ...]:
    out: list[tuple[int, int]] = [(0, 0)]
    ring = ((-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1))
    for radius in radii:
        r = int(radius)
        out.extend((dx * r, dy * r) for dx, dy in ring)
    return tuple(out)


def sample_native_rgba_stencil(
    native_rgba: torch.Tensor,
    grid_xy: torch.Tensor,
    offsets_px: tuple[tuple[int, int], ...] | None = None,
) -> torch.Tensor:
    """Preserve the sealed learner's exact-native local refinement evidence.

    native_rgba [B,V,4,1024,1024], grid_xy [B,V,Q,2]
    -> [B,V,Q,4*len(offsets)]
    """
    if offsets_px is None:
        offsets_px = fixed_native_patch_offsets()
    if native_rgba.ndim != 5 or native_rgba.shape[2:] != (4, 1024, 1024):
        raise ValueError(f"NATIVE_RGBA_STENCIL_CONTRACT:{tuple(native_rgba.shape)}")
    b, v, _, h, w = native_rgba.shape
    if grid_xy.shape[:2] != (b, v) or grid_xy.ndim != 4 or grid_xy.shape[-1] != 2:
        raise ValueError("NATIVE_STENCIL_GRID_MISMATCH")
    q = grid_xy.shape[2]
    grids = []
    base = grid_xy.float()
    for dx, dy in offsets_px:
        delta = base.new_tensor([2.0 * float(dx) / float(w), 2.0 * float(dy) / float(h)])
        grids.append(base + delta)
    g = torch.stack(grids, dim=3).reshape(b * v, q, len(offsets_px), 2)
    samp = F.grid_sample(
        native_rgba.reshape(b * v, 4, h, w).float(),
        g,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    samp = samp.reshape(b, v, 4, q, len(offsets_px)).permute(0, 1, 3, 4, 2).contiguous()
    return samp.reshape(b, v, q, 4 * len(offsets_px))
