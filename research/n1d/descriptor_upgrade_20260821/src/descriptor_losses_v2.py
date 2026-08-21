from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def sample_dense_native(field: torch.Tensor, xy_px: torch.Tensor, image_size: int = 256) -> torch.Tensor:
    """Sample [B,V,C,H,W] at [B,V,N,2] native-pixel coordinates -> [B,V,N,C]."""
    if field.ndim != 5 or xy_px.ndim != 4:
        raise ValueError((field.shape, xy_px.shape))
    B, V, C, H, W = field.shape
    if xy_px.shape[:2] != (B, V) or xy_px.shape[-1] != 2:
        raise ValueError((field.shape, xy_px.shape))
    N = xy_px.shape[2]
    grid = xy_px.to(field.dtype).clone()
    grid[..., 0] = grid[..., 0] / float(image_size - 1) * 2.0 - 1.0
    grid[..., 1] = grid[..., 1] / float(image_size - 1) * 2.0 - 1.0
    y = F.grid_sample(
        field.reshape(B * V, C, H, W),
        grid.reshape(B * V, N, 1, 2),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return y[:, :, :, 0].permute(0, 2, 1).reshape(B, V, N, C)


def _mean_or_zero(values: list[torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
    if not values:
        return ref.sum() * 0.0
    return torch.stack(values).mean()


def same_view_dual_softmax_loss(
    zA: torch.Tensor,
    zB: torch.Tensor,
    vA: torch.Tensor,
    vB: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Directly train same-view A carrier i -> B carrier i matching."""
    losses: list[torch.Tensor] = []
    B, V, N, D = zA.shape
    for b in range(B):
        for v in range(V):
            valid = vA[b, v].bool() & vB[b, v].bool()
            if int(valid.sum()) < 2:
                continue
            a = F.normalize(zA[b, v, valid], dim=-1, eps=1e-6)
            c = F.normalize(zB[b, v, valid], dim=-1, eps=1e-6)
            logits = (a @ c.T) / temperature
            labels = torch.arange(logits.shape[0], device=logits.device)
            losses.append(0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)))
    return _mean_or_zero(losses, zA)


def observation_supervised_contrastive_loss(
    zA: torch.Tensor,
    zB: torch.Tensor,
    vA: torch.Tensor,
    vB: torch.Tensor,
    temperature: float = 0.08,
) -> torch.Tensor:
    """Multi-positive SupCon over observations, never averaging views first."""
    family_losses: list[torch.Tensor] = []
    B, V, N, D = zA.shape
    for b in range(B):
        feats: list[torch.Tensor] = []
        ids: list[torch.Tensor] = []
        for z, vis in ((zA[b], vA[b]), (zB[b], vB[b])):
            idx = vis.bool().nonzero(as_tuple=False)
            if idx.numel() == 0:
                continue
            feats.append(z[idx[:, 0], idx[:, 1]])
            ids.append(idx[:, 1])
        if not feats:
            continue
        f = F.normalize(torch.cat(feats, dim=0), dim=-1, eps=1e-6)
        carrier = torch.cat(ids, dim=0)
        if f.shape[0] < 3:
            continue
        sim = (f @ f.T) / temperature
        eye = torch.eye(f.shape[0], device=f.device, dtype=torch.bool)
        pos = carrier[:, None].eq(carrier[None, :]) & ~eye
        valid_anchor = pos.any(dim=1)
        if not valid_anchor.any():
            continue
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()
        exp_sim = torch.exp(sim) * (~eye)
        denom = exp_sim.sum(dim=1).clamp_min(1e-12)
        numer = (exp_sim * pos).sum(dim=1).clamp_min(1e-12)
        family_losses.append((-torch.log(numer / denom))[valid_anchor].mean())
    return _mean_or_zero(family_losses, zA)


def hard_negative_margin_loss(
    zA: torch.Tensor,
    zB: torch.Tensor,
    vA: torch.Tensor,
    vB: torch.Tensor,
    margin: float = 0.15,
) -> torch.Tensor:
    """Mine the highest-scoring wrong carrier in the same view as hard negative."""
    losses: list[torch.Tensor] = []
    B, V, N, D = zA.shape
    for b in range(B):
        for v in range(V):
            valid = vA[b, v].bool() & vB[b, v].bool()
            if int(valid.sum()) < 2:
                continue
            a = F.normalize(zA[b, v, valid], dim=-1, eps=1e-6)
            c = F.normalize(zB[b, v, valid], dim=-1, eps=1e-6)
            sim = a @ c.T
            pos = sim.diag()
            eye = torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
            neg = sim.masked_fill(eye, -1e9).max(dim=1).values
            losses.append(F.relu(margin + neg - pos).mean())
    return _mean_or_zero(losses, zA)


def _offset_grid(radius_px: int, step_px: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    vals = torch.arange(-radius_px, radius_px + 1, step_px, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(vals, vals, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


def local_fine_matching_loss(
    fieldA: torch.Tensor,
    fieldB: torch.Tensor,
    xyA: torch.Tensor,
    xyB: torch.Tensor,
    vA: torch.Tensor,
    vB: torch.Tensor,
    image_size: int = 256,
    radius_px: int = 8,
    step_px: int = 2,
    temperature: float = 0.05,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Fine localization classification on a native-pixel lattice around truth B."""
    q = F.normalize(sample_dense_native(fieldA, xyA, image_size=image_size), dim=-1, eps=1e-6)
    offsets = _offset_grid(radius_px, step_px, fieldA.device, xyA.dtype)
    zero_idx = int(((offsets[:, 0] == 0) & (offsets[:, 1] == 0)).nonzero(as_tuple=False)[0])
    per: list[torch.Tensor] = []
    for off in offsets:
        xy = xyB + off.view(1, 1, 1, 2)
        xy_clamped = xy.clamp(0, image_size - 1)
        z = F.normalize(sample_dense_native(fieldB, xy_clamped, image_size=image_size), dim=-1, eps=1e-6)
        per.append((q * z).sum(dim=-1))
    logits = torch.stack(per, dim=-1) / temperature
    valid = vA.bool() & vB.bool()
    if valid.any():
        target = torch.full(logits.shape[:-1], zero_idx, device=logits.device, dtype=torch.long)
        loss = F.cross_entropy(logits[valid], target[valid])
        top1 = logits.argmax(dim=-1)
        acc = (top1[valid] == zero_idx).float().mean()
    else:
        loss = logits.sum() * 0.0
        acc = logits.sum() * 0.0
    return loss, {
        "fine_local_top1": acc.detach(),
        "fine_local_candidate_n": torch.tensor(logits.shape[-1], device=logits.device),
    }


@dataclass(frozen=True)
class DescriptorV2LossWeights:
    observation_supcon: float = 1.0
    same_view_dual: float = 1.0
    hard_negative: float = 0.5
    local_fine: float = 1.0


def descriptor_v2_objective(
    outputs: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    image_size: int = 256,
    weights: DescriptorV2LossWeights = DescriptorV2LossWeights(),
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Training-only objective; no target/sidecar field is required at inference."""
    d = outputs["descriptor_coarse"]
    xyA, xyB = target["XY_A"], target["XY_B"]
    vA, vB = target["V_A"].bool(), target["V_B"].bool()
    zA = sample_dense_native(d[:, 0], xyA, image_size=image_size)
    zB = sample_dense_native(d[:, 1], xyB, image_size=image_size)

    parts: Dict[str, torch.Tensor] = {}
    parts["observation_supcon"] = observation_supervised_contrastive_loss(zA, zB, vA, vB)
    parts["same_view_dual"] = same_view_dual_softmax_loss(zA, zB, vA, vB)
    parts["hard_negative"] = hard_negative_margin_loss(zA, zB, vA, vB)

    if "descriptor_fine" in outputs:
        fine = outputs["descriptor_fine"]
        parts["local_fine"], diag = local_fine_matching_loss(
            fine[:, 0], fine[:, 1], xyA, xyB, vA, vB, image_size=image_size
        )
        parts.update(diag)
    else:
        parts["local_fine"] = d.sum() * 0.0

    total = (
        weights.observation_supcon * parts["observation_supcon"]
        + weights.same_view_dual * parts["same_view_dual"]
        + weights.hard_negative * parts["hard_negative"]
        + weights.local_fine * parts["local_fine"]
    )
    parts["total"] = total
    return total, parts
