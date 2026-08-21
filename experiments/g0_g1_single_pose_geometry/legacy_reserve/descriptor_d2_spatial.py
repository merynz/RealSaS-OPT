from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import torch
from torch import nn
import torch.nn.functional as F

from realsas_iris_sees.model import IRISSEESN1, SEESConfig
from descriptor_d1_objective import observation_loss_d1, descriptor_diagnostics


def _gn(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualFineBlock(nn.Module):
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
class D2FineConfig:
    f2_dim: int = 64
    context_dim: int = 128
    local_dim: int = 64
    hidden_dim: int = 96
    descriptor_dim: int = 32


class FineDescriptorHead(nn.Module):
    """Genuine 64x64 localization field with pair-aware 32x32 context."""
    def __init__(self, cfg: D2FineConfig = D2FineConfig()):
        super().__init__()
        self.cfg = cfg
        self.local = nn.Sequential(
            nn.Conv2d(cfg.f2_dim, cfg.local_dim, 3, padding=1, bias=False),
            _gn(cfg.local_dim),
            nn.GELU(),
        )
        self.context = nn.Sequential(
            nn.Conv2d(cfg.context_dim, cfg.local_dim, 1, bias=False),
            _gn(cfg.local_dim),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(cfg.local_dim * 2, cfg.hidden_dim, 3, padding=1, bias=False),
            _gn(cfg.hidden_dim),
            nn.GELU(),
            ResidualFineBlock(cfg.hidden_dim),
        )
        self.desc = nn.Conv2d(cfg.hidden_dim, cfg.descriptor_dim, 1)
        self.conf = nn.Conv2d(cfg.hidden_dim, 1, 1)

    def forward(self, f2: torch.Tensor, dense32: torch.Tensor) -> Dict[str, torch.Tensor]:
        if f2.ndim != 4 or dense32.ndim != 4:
            raise ValueError((tuple(f2.shape), tuple(dense32.shape)))
        a = self.local(f2)
        c = self.context(dense32)
        c = F.interpolate(c, size=a.shape[-2:], mode="bilinear", align_corners=False)
        h = self.fuse(torch.cat([a, c], dim=1))
        return {
            "descriptor_fine": F.normalize(self.desc(h), dim=1, eps=1e-6),
            "descriptor_fine_log_sigma": self.conf(h).clamp(-6, 2),
        }


class IRISSEESN1D2(nn.Module):
    """Canonical D1 model + an additive fine descriptor branch.

    The canonical 32x32 descriptor and every geometry/differential output remain
    byte-schema compatible in `base`. A persistent forward hook exposes the encoder
    stem's genuine 64x64 f2 feature without editing frozen N1D source.
    """
    def __init__(self, cfg: SEESConfig, fine_cfg: D2FineConfig = D2FineConfig()):
        super().__init__()
        self.cfg = cfg
        self.base = IRISSEESN1(cfg)
        self.fine_head = FineDescriptorHead(fine_cfg)
        self._f2_cache = None
        self._stem_hook = self.base.encoder.stem.register_forward_hook(self._capture_f2)

    def _capture_f2(self, module, inputs, output):
        self._f2_cache = output
        return None

    def load_d1_state(self, state: Dict[str, torch.Tensor]):
        result = self.base.load_state_dict(state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError((result.missing_keys, result.unexpected_keys))
        return result

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._f2_cache = None
        out = self.base(images)
        f2 = self._f2_cache
        if f2 is None:
            raise RuntimeError("encoder stem hook did not capture f2")
        B, P, V = images.shape[:3]
        dense = out["dense_feature"]
        if tuple(dense.shape[:3]) != (B, P, V):
            raise RuntimeError((tuple(dense.shape), B, P, V))
        flat_dense = dense.reshape(B * P * V, dense.shape[3], dense.shape[4], dense.shape[5])
        fine = self.fine_head(f2, flat_dense)
        Hf, Wf = fine["descriptor_fine"].shape[-2:]
        out["descriptor_fine"] = fine["descriptor_fine"].view(B, P, V, -1, Hf, Wf)
        out["descriptor_fine_log_sigma"] = fine["descriptor_fine_log_sigma"].view(B, P, V, 1, Hf, Wf)
        return out


def sample_dense_native(field: torch.Tensor, xy_px: torch.Tensor, image_size: int = 256) -> torch.Tensor:
    """Sample [B,V,C,H,W] at [B,V,N,2] native coordinates -> [B,V,N,C]."""
    if field.ndim != 5 or xy_px.ndim != 4:
        raise ValueError((tuple(field.shape), tuple(xy_px.shape)))
    B, V, C, H, W = field.shape
    if tuple(xy_px.shape[:2]) != (B, V) or xy_px.shape[-1] != 2:
        raise ValueError((tuple(field.shape), tuple(xy_px.shape)))
    N = xy_px.shape[2]
    grid = xy_px.to(dtype=field.dtype).clone()
    grid[..., 0] = grid[..., 0] / float(image_size - 1) * 2.0 - 1.0
    grid[..., 1] = grid[..., 1] / float(image_size - 1) * 2.0 - 1.0
    y = F.grid_sample(
        field.reshape(B * V, C, H, W),
        grid.reshape(B * V, N, 1, 2),
        mode="bilinear", padding_mode="border", align_corners=True,
    )
    return y[:, :, :, 0].permute(0, 2, 1).reshape(B, V, N, C)


def _offset_grid(radius_px: int, step_px: int, device, dtype) -> torch.Tensor:
    vals = torch.arange(-radius_px, radius_px + 1, step_px, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(vals, vals, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


def local_fine_matching_loss(
    outputs: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    image_size: int = 256,
    radius_px: int = 8,
    step_px: int = 2,
    temperature: float = 0.05,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    fine = outputs["descriptor_fine"]
    B, P, V, D, H, W = fine.shape
    if P != 2:
        raise ValueError(tuple(fine.shape))
    xyA = target["XY_A"].to(fine.device)
    xyB = target["XY_B"].to(fine.device)
    vA = target["V_A"].bool().to(fine.device)
    vB = target["V_B"].bool().to(fine.device)
    q = F.normalize(sample_dense_native(fine[:, 0], xyA, image_size), dim=-1, eps=1e-6)
    offsets = _offset_grid(radius_px, step_px, fine.device, xyB.dtype)
    K = offsets.shape[0]
    zero = ((offsets[:, 0] == 0) & (offsets[:, 1] == 0)).nonzero(as_tuple=False).flatten()
    if zero.numel() != 1:
        raise RuntimeError((radius_px, step_px, offsets))
    zero_idx = int(zero.item())
    cand = xyB[..., None, :] + offsets.view(1, 1, 1, K, 2)
    inb = ((cand[..., 0] >= 0) & (cand[..., 0] <= image_size - 1) &
           (cand[..., 1] >= 0) & (cand[..., 1] <= image_size - 1))
    z = sample_dense_native(fine[:, 1], cand.reshape(B, V, -1, 2).clamp(0, image_size - 1), image_size)
    z = F.normalize(z.view(B, V, xyB.shape[2], K, D), dim=-1, eps=1e-6)
    logits = (q[..., None, :] * z).sum(dim=-1) / temperature
    logits = logits.masked_fill(~inb, -1e4)
    valid = vA & vB & inb[..., zero_idx]
    if not valid.any():
        zero_loss = fine.sum() * 0.0
        return zero_loss, {
            "D2_local_fine_top1": zero_loss.detach(),
            "D2_local_fine_expected_error_px": zero_loss.detach(),
            "D2_local_fine_candidate_n": torch.tensor(K, device=fine.device),
        }
    label = torch.full(logits.shape[:-1], zero_idx, device=fine.device, dtype=torch.long)
    loss = F.cross_entropy(logits[valid], label[valid])
    pred = logits.argmax(dim=-1)
    top1 = (pred[valid] == zero_idx).float().mean()
    probs = F.softmax(logits, dim=-1)
    expected_offset = torch.einsum("bvnk,kd->bvnd", probs, offsets.to(probs.dtype))
    exp_err = expected_offset.norm(dim=-1)[valid].mean()
    return loss, {
        "D2_local_fine_top1": top1.detach(),
        "D2_local_fine_expected_error_px": exp_err.detach(),
        "D2_local_fine_candidate_n": torch.tensor(K, device=fine.device),
    }


def observation_loss_c2(outputs, target, image_size: int = 256):
    return observation_loss_d1(outputs, target, image_size)


def observation_loss_d2(outputs, target, image_size: int = 256, fine_weight: float = 0.10):
    d1_total, parts = observation_loss_d1(outputs, target, image_size)
    fine_loss, fine_diag = local_fine_matching_loss(outputs, target, image_size=image_size)
    total = d1_total + float(fine_weight) * fine_loss
    out = dict(parts)
    out["D2_local_fine_loss"] = fine_loss
    out.update(fine_diag)
    out["D2_fine_weight"] = torch.as_tensor(float(fine_weight), device=fine_loss.device)
    out["total"] = total
    return total, out
