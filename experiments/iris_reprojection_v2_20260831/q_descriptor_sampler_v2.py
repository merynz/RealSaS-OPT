from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

from .q_domain_v2 import RayHypothesisDomainV2


def _groups(c: int) -> int:
    for g in range(min(16, c), 0, -1):
        if c % g == 0:
            return g
    return 1


class _ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False), nn.GroupNorm(_groups(cout), cout), nn.GELU(), nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(_groups(cout), cout), nn.GELU())
    def forward(self, x): return self.net(x)


class NativeResolutionPyramidV2(nn.Module):
    def __init__(self, in_channels: int = 4, widths: tuple[int, ...] = (24, 40, 64)):
        super().__init__(); self.l0=_ConvBlock(in_channels,widths[0],1); self.l1=_ConvBlock(widths[0],widths[1],2); self.l2=_ConvBlock(widths[1],widths[2],2); self.output_dim=sum(widths)
    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if images.ndim != 5 or images.shape[1] != 8 or images.shape[2] != 4: raise ValueError("images must be [B,8,4,H,W]")
        B,V,C,H,W=images.shape; x=images.reshape(B*V,C,H,W); f0=self.l0(x); f1=self.l1(f0); f2=self.l2(f1)
        return tuple(f.reshape(B,V,f.shape[1],f.shape[2],f.shape[3]) for f in (f0,f1,f2))


def sample_multiview_map(feature_map: torch.Tensor, domain: RayHypothesisDomainV2) -> torch.Tensor:
    if feature_map.ndim != 5 or feature_map.shape[1] != 8: raise ValueError("feature map must be [B,8,C,H,W]")
    B,V,C,H,W=feature_map.shape
    if B != domain.projected_grid.shape[0]: raise ValueError("feature/domain batch mismatch")
    Q,D=domain.projected_grid.shape[1:3]; fmap=feature_map.reshape(B*V,C,H,W); grid=domain.projected_grid.permute(0,3,1,2,4).reshape(B*V,Q*D,1,2)
    sampled=F.grid_sample(fmap,grid,mode="bilinear",padding_mode="zeros",align_corners=False)
    return sampled.squeeze(-1).transpose(1,2).reshape(B,V,Q,D,C).permute(0,2,3,1,4)


@dataclass(frozen=True)
class SampledQDescriptorsV2:
    descriptors: torch.Tensor
    valid_views: torch.Tensor
    descriptor_dim: int


class QDescriptorSamplerV2(nn.Module):
    def __init__(self, foundation_dims: tuple[int, ...], native: NativeResolutionPyramidV2 | None = None):
        super().__init__(); self.foundation_dims=tuple(int(x) for x in foundation_dims)
        if not self.foundation_dims or min(self.foundation_dims)<=0: raise ValueError("foundation feature dims required")
        self.native=native or NativeResolutionPyramidV2(); self.descriptor_dim=sum(self.foundation_dims)+self.native.output_dim
    def forward(self, images: torch.Tensor, frozen_foundation_maps: tuple[torch.Tensor, ...], domain: RayHypothesisDomainV2) -> SampledQDescriptorsV2:
        if len(frozen_foundation_maps)!=len(self.foundation_dims): raise ValueError("foundation level count mismatch")
        parts=[]
        for fmap,dim in zip(frozen_foundation_maps,self.foundation_dims):
            if fmap.shape[2]!=dim: raise ValueError("foundation channel mismatch")
            parts.append(sample_multiview_map(fmap.detach(),domain))
        parts.extend(sample_multiview_map(fmap,domain) for fmap in self.native(images)); desc=torch.cat(parts,dim=-1)
        return SampledQDescriptorsV2(desc,domain.in_frame & domain.candidate_valid[...,None],int(desc.shape[-1]))
