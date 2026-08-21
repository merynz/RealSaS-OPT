from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F


def _gn(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualDescriptorBlock(nn.Module):
    """Small spatial block; deliberately original and dependency-free."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = _gn(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = _gn(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.gelu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.gelu(x + y)


@dataclass(frozen=True)
class DescriptorV2Config:
    coarse_in_dim: int = 128
    fine_in_dim: int = 64
    hidden_dim: int = 96
    coarse_dim: int = 64
    fine_dim: int = 32
    coarse_blocks: int = 1
    fine_blocks: int = 1


class DescriptorV2Head(nn.Module):
    """Coarse identity + fine localization descriptor head.

    Inputs are flattened image features [M,C,H,W], where M can be B*P*V.
    `coarse_feature` is the current pair-aware dense f4 feature (32x32 in SEES).
    `fine_feature` is an optional higher-resolution encoder feature (e.g. f2/stem, 64x64).

    The head keeps semantic/global context in the coarse branch and injects that
    context into a genuinely higher-resolution fine branch. This directly avoids
    asking bilinear interpolation of a 32x32 descriptor map to resolve 2px choices.
    """

    def __init__(self, cfg: DescriptorV2Config = DescriptorV2Config()):
        super().__init__()
        self.cfg = cfg

        self.coarse_in = nn.Sequential(
            nn.Conv2d(cfg.coarse_in_dim, cfg.hidden_dim, 3, padding=1, bias=False),
            _gn(cfg.hidden_dim),
            nn.GELU(),
        )
        self.coarse_body = nn.Sequential(
            *[ResidualDescriptorBlock(cfg.hidden_dim) for _ in range(cfg.coarse_blocks)]
        )
        self.coarse_desc = nn.Conv2d(cfg.hidden_dim, cfg.coarse_dim, 1)
        self.coarse_log_sigma = nn.Conv2d(cfg.hidden_dim, 1, 1)

        self.fine_in = nn.Sequential(
            nn.Conv2d(cfg.fine_in_dim, cfg.hidden_dim // 2, 3, padding=1, bias=False),
            _gn(cfg.hidden_dim // 2),
            nn.GELU(),
        )
        self.context_proj = nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim // 2, 1, bias=False)
        self.fine_fuse = nn.Sequential(
            nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim, 3, padding=1, bias=False),
            _gn(cfg.hidden_dim),
            nn.GELU(),
            *[ResidualDescriptorBlock(cfg.hidden_dim) for _ in range(cfg.fine_blocks)],
        )
        self.fine_desc = nn.Conv2d(cfg.hidden_dim, cfg.fine_dim, 1)
        self.fine_log_sigma = nn.Conv2d(cfg.hidden_dim, 1, 1)

    def forward(
        self,
        coarse_feature: torch.Tensor,
        fine_feature: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        c = self.coarse_body(self.coarse_in(coarse_feature))
        out: Dict[str, torch.Tensor] = {
            "descriptor_coarse": F.normalize(self.coarse_desc(c), dim=1, eps=1e-6),
            "descriptor_coarse_log_sigma": self.coarse_log_sigma(c).clamp(-6, 2),
        }
        if fine_feature is None:
            return out

        f = self.fine_in(fine_feature)
        ctx = F.interpolate(
            self.context_proj(c), size=f.shape[-2:], mode="bilinear", align_corners=False
        )
        ff = self.fine_fuse(torch.cat([f, ctx], dim=1))
        out.update(
            {
                "descriptor_fine": F.normalize(self.fine_desc(ff), dim=1, eps=1e-6),
                "descriptor_fine_log_sigma": self.fine_log_sigma(ff).clamp(-6, 2),
            }
        )
        return out


def reshape_bpv(field: torch.Tensor, B: int, P: int = 2, V: int = 8) -> torch.Tensor:
    """[B*P*V,C,H,W] -> [B,P,V,C,H,W]."""
    if field.ndim != 4 or field.shape[0] != B * P * V:
        raise ValueError((field.shape, B, P, V))
    return field.view(B, P, V, *field.shape[1:])
