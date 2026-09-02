from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F

from .model_v2 import IrisReprojectionOutputV2
from .q_domain_v2 import RayHypothesisDomainV2
from .world_regularizer_v2 import isotropic_world_regularizer_v2


@dataclass(frozen=True)
class IrisV2LossWeights:
    depth_mode: float = 1.0
    refined_depth: float = 2.0
    uncertainty_nll: float = 0.5
    support: float = 0.5
    world_regularizer: float = 0.02


def iris_v2_loss(
    output: IrisReprojectionOutputV2,
    domain: RayHypothesisDomainV2,
    teacher_depth: torch.Tensor,
    teacher_support: torch.Tensor,
    *,
    weights: IrisV2LossWeights = IrisV2LossWeights(),
) -> dict[str, torch.Tensor]:
    """Depth/support-only supervision. P/N truth losses are deliberately absent."""
    if teacher_depth.shape != domain.anchor_view.shape or teacher_support.shape != domain.anchor_view.shape:
        raise ValueError("teacher target shape mismatch")
    td = teacher_depth.to(domain.depth_values.dtype)
    ts = teacher_support.to(dtype=torch.bool)
    dist = (domain.depth_values - td[..., None]).abs()
    nearest = dist.argmin(dim=-1)
    logits = output.field.score_logits
    mode_ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), nearest.reshape(-1), reduction="none").reshape_as(td)
    mode_ce = (mode_ce * ts.to(mode_ce.dtype)).sum() / ts.sum().clamp_min(1)

    if output.refined_depth.shape[-1] == 0:
        raise ValueError("no output modes")
    pred = output.refined_depth[..., 0]
    residual = pred - td
    refined = (F.smooth_l1_loss(pred, td, reduction="none") * ts.to(pred.dtype)).sum() / ts.sum().clamp_min(1)
    log_sigma = output.depth_output.log_sigma[..., 0]
    nll = (0.5 * torch.exp(-2.0 * log_sigma) * residual.square() + log_sigma)
    nll = (nll * ts.to(nll.dtype)).sum() / ts.sum().clamp_min(1)
    support_logits = output.depth_output.support_logits[..., 0]
    support = F.binary_cross_entropy_with_logits(support_logits, ts.to(support_logits.dtype))
    regularizer = isotropic_world_regularizer_v2(domain.q_points, logits, valid_mask=domain.candidate_valid)
    total = weights.depth_mode * mode_ce + weights.refined_depth * refined + weights.uncertainty_nll * nll + weights.support * support + weights.world_regularizer * regularizer
    return {"total": total, "depth_mode": mode_ce, "refined_depth": refined, "uncertainty_nll": nll, "support": support, "world_regularizer": regularizer}


def train_step(model, optimizer, batch: dict) -> dict[str, float]:
    model.train(); optimizer.zero_grad(set_to_none=True)
    output = model(batch["images"], batch["foundation_maps"], batch["domain"])
    losses = iris_v2_loss(output, batch["domain"], batch["teacher_depth"], batch["teacher_support"])
    losses["total"].backward(); optimizer.step()
    return {k: float(v.detach().cpu()) for k, v in losses.items()}
