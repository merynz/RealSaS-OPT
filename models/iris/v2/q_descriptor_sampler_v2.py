from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .q_domain_v2 import RayHypothesisDomainV2


def _groups(c: int) -> int:
    for g in range(min(16, c), 0, -1):
        if c % g == 0:
            return g
    return 1


class _ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_groups(cout), cout),
            nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(cout), cout),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NativeResolutionPyramidV2(nn.Module):
    """Learned RGB-only native-image pyramid preserving the five-scale path.

    Source observations may be stored as RGBA for raster/provenance authority, but
    renderer alpha is not learner evidence. Production and source-test learner
    inputs are therefore exactly three RGB channels.
    """

    DEFAULT_WIDTHS = (32, 48, 64, 96, 128)
    INPUT_CHANNELS = 3

    def __init__(self, in_channels: int = INPUT_CHANNELS, widths: tuple[int, ...] = DEFAULT_WIDTHS):
        super().__init__()
        if int(in_channels) != self.INPUT_CHANNELS:
            raise ValueError("IRIS V2 learned native path is RGB-only")
        widths = tuple(int(x) for x in widths)
        if len(widths) != 5 or min(widths) <= 0:
            raise ValueError("IRIS V2 native pyramid requires five positive widths")
        blocks = []
        cin = self.INPUT_CHANNELS
        for level, cout in enumerate(widths):
            blocks.append(_ConvBlock(cin, cout, stride=1 if level == 0 else 2))
            cin = cout
        self.blocks = nn.ModuleList(blocks)
        self.widths = widths
        self.output_dim = int(sum(widths))

    def forward_flat(self, images_flat: torch.Tensor, *, checkpoint_blocks: bool = False) -> tuple[torch.Tensor, ...]:
        if images_flat.ndim != 4 or images_flat.shape[1] != self.INPUT_CHANNELS:
            raise ValueError("native flat images must be [N,3,H,W] RGB")
        if min(images_flat.shape[-2:]) < 16:
            raise ValueError("native pyramid input too small for five-scale path")
        x = images_flat
        out = []
        for block in self.blocks:
            if checkpoint_blocks and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
            out.append(x)
        return tuple(out)

    def forward(self, images: torch.Tensor, *, checkpoint_blocks: bool = False) -> tuple[torch.Tensor, ...]:
        if images.ndim != 5 or images.shape[1] != 8 or images.shape[2] != self.INPUT_CHANNELS:
            raise ValueError("images must be [B,8,3,H,W] RGB")
        B, V, C, H, W = images.shape
        levels = self.forward_flat(images.reshape(B * V, C, H, W), checkpoint_blocks=checkpoint_blocks)
        return tuple(x.reshape(B, V, x.shape[1], x.shape[2], x.shape[3]) for x in levels)


def sample_multiview_map(feature_map: torch.Tensor, domain: RayHypothesisDomainV2) -> torch.Tensor:
    if feature_map.ndim != 5 or feature_map.shape[1] != 8:
        raise ValueError("feature map must be [B,8,C,H,W]")
    B, V, C, H, W = feature_map.shape
    if B != domain.projected_grid.shape[0]:
        raise ValueError("feature/domain batch mismatch")
    Q, D = domain.projected_grid.shape[1:3]
    fmap = feature_map.reshape(B * V, C, H, W)
    grid = domain.projected_grid.permute(0, 3, 1, 2, 4).reshape(B * V, Q * D, 1, 2)
    sampled = F.grid_sample(fmap, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return sampled.squeeze(-1).transpose(1, 2).reshape(B, V, Q, D, C).permute(0, 2, 3, 1, 4)


def _sample_view_chunk_map(feature_map_flat: torch.Tensor, domain: RayHypothesisDomainV2, view_start: int, view_end: int) -> torch.Tensor:
    """Sample [B*Vc,C,H,W] features for a contiguous view chunk into [B,Q,D,Vc,C]."""
    B = domain.projected_grid.shape[0]
    Vc = int(view_end - view_start)
    if Vc < 1 or feature_map_flat.ndim != 4 or feature_map_flat.shape[0] != B * Vc:
        raise ValueError("native view-chunk feature shape mismatch")
    C, H, W = feature_map_flat.shape[1:]
    Q, D = domain.projected_grid.shape[1:3]
    grid = domain.projected_grid[..., view_start:view_end, :].permute(0, 3, 1, 2, 4).reshape(B * Vc, Q * D, 1, 2)
    sampled = F.grid_sample(feature_map_flat, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return sampled.squeeze(-1).transpose(1, 2).reshape(B, Vc, Q, D, C).permute(0, 2, 3, 1, 4)


@dataclass(frozen=True)
class SampledQDescriptorsV2:
    descriptors: torch.Tensor
    valid_views: torch.Tensor
    descriptor_dim: int


class QDescriptorSamplerV2(nn.Module):
    def __init__(
        self,
        foundation_dims: tuple[int, ...],
        native: NativeResolutionPyramidV2 | None = None,
        *,
        native_view_chunk: int = 1,
        checkpoint_native_blocks: bool = True,
    ):
        super().__init__()
        self.foundation_dims = tuple(int(x) for x in foundation_dims)
        if not self.foundation_dims or min(self.foundation_dims) <= 0:
            raise ValueError("foundation feature dims required")
        if not (1 <= int(native_view_chunk) <= 8):
            raise ValueError("native_view_chunk must be 1..8")
        self.native = native or NativeResolutionPyramidV2()
        self.native_view_chunk = int(native_view_chunk)
        self.checkpoint_native_blocks = bool(checkpoint_native_blocks)
        self.descriptor_dim = int(sum(self.foundation_dims) + self.native.output_dim)

    def sample_native_streamed(self, images: torch.Tensor, domain: RayHypothesisDomainV2) -> torch.Tensor:
        """Mathematically full five-scale RGB path with bounded view materialization."""
        if images.ndim != 5 or images.shape[1] != 8 or images.shape[2] != NativeResolutionPyramidV2.INPUT_CHANNELS:
            raise ValueError("images must be [B,8,3,H,W] RGB")
        B, V, C, H, W = images.shape
        chunks = []
        for start in range(0, V, self.native_view_chunk):
            end = min(V, start + self.native_view_chunk)
            vc = end - start
            flat = images[:, start:end].reshape(B * vc, C, H, W)
            levels = self.native.forward_flat(flat, checkpoint_blocks=self.checkpoint_native_blocks)
            sampled_levels = [_sample_view_chunk_map(level, domain, start, end) for level in levels]
            chunks.append(torch.cat(sampled_levels, dim=-1))
        return torch.cat(chunks, dim=-2)

    def forward(
        self,
        images: torch.Tensor,
        frozen_foundation_maps: tuple[torch.Tensor, ...],
        domain: RayHypothesisDomainV2,
    ) -> SampledQDescriptorsV2:
        if len(frozen_foundation_maps) != len(self.foundation_dims):
            raise ValueError("foundation level count mismatch")
        parts = []
        for fmap, dim in zip(frozen_foundation_maps, self.foundation_dims):
            if fmap.shape[2] != dim:
                raise ValueError("foundation channel mismatch")
            parts.append(sample_multiview_map(fmap.detach(), domain))
        parts.append(self.sample_native_streamed(images, domain))
        desc = torch.cat(parts, dim=-1)
        valid = domain.in_frame & domain.candidate_valid[..., None]
        return SampledQDescriptorsV2(desc, valid, int(desc.shape[-1]))
