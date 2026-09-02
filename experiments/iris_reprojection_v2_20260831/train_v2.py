from __future__ import annotations

from dataclasses import dataclass
import math
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
    tail_depth: float = 0.5
    world_regularizer: float = 0.02
    tail_fraction: float = 0.10

    def validate(self) -> None:
        numeric = (self.depth_mode, self.refined_depth, self.uncertainty_nll, self.support, self.tail_depth, self.world_regularizer)
        if min(numeric) < 0 or not (0.0 < self.tail_fraction <= 1.0):
            raise ValueError("invalid IRIS V2 loss weights")


def _teacher_modes(
    teacher_depth: torch.Tensor,
    teacher_support: torch.Tensor,
    domain: RayHypothesisDomainV2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize legacy single-depth and generic multimodal supervision.

    Returns depth [B,Q,M], mode-valid [B,Q,M], q-supported [B,Q]. NaN/Inf
    teacher modes are always invalid. A rank-2 support mask broadcasts over modes;
    a rank-3 support mask can identify individual valid intersections.
    """
    if teacher_depth.ndim == 2:
        if teacher_depth.shape != domain.anchor_view.shape:
            raise ValueError("teacher depth shape mismatch")
        depth = teacher_depth[..., None]
    elif teacher_depth.ndim == 3 and teacher_depth.shape[:2] == domain.anchor_view.shape:
        depth = teacher_depth
    else:
        raise ValueError("teacher_depth must be [B,Q] or [B,Q,M]")
    if teacher_support.ndim == 2:
        if teacher_support.shape != domain.anchor_view.shape:
            raise ValueError("teacher support shape mismatch")
        mode_valid = teacher_support.bool()[..., None].expand_as(depth)
    elif teacher_support.ndim == 3 and teacher_support.shape == depth.shape:
        mode_valid = teacher_support.bool()
    else:
        raise ValueError("teacher_support must be [B,Q] or match multimodal depth")
    finite = torch.isfinite(depth)
    mode_valid = mode_valid & finite
    depth = torch.where(finite, depth, torch.zeros_like(depth))
    q_supported = mode_valid.any(dim=-1)
    return depth.to(domain.depth_values.dtype), mode_valid, q_supported


def _multitarget_mode_nll(logits: torch.Tensor, depth_values: torch.Tensor, teacher_depth: torch.Tensor, teacher_valid: torch.Tensor) -> torch.Tensor:
    dist = (depth_values[..., None] - teacher_depth[:, :, None, :]).abs()
    nearest = dist.argmin(dim=2)  # [B,Q,M]
    logp = torch.log_softmax(logits, dim=-1)
    selected = torch.gather(logp, -1, nearest)
    valid = teacher_valid & torch.isfinite(selected)
    if not valid.any():
        return logits.sum() * 0.0
    return -selected[valid].mean()


def _mode_matching_errors(
    output: IrisReprojectionOutputV2,
    teacher_depth: torch.Tensor,
    teacher_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = output.refined_depth
    pred_valid = output.modes.mode_valid.bool()
    diff = (pred[..., :, None] - teacher_depth[..., None, :]).abs()  # [B,Q,K,M]
    inf = torch.full_like(diff, float("inf"))
    diff_for_teacher = torch.where(pred_valid[..., :, None], diff, inf)
    teacher_error, teacher_to_pred = diff_for_teacher.min(dim=-2)
    teacher_ok = teacher_valid & torch.isfinite(teacher_error)
    diff_for_pred = torch.where(teacher_valid[..., None, :], diff, inf)
    pred_error, _ = diff_for_pred.min(dim=-1)
    pred_ok = pred_valid & torch.isfinite(pred_error)
    return teacher_error, teacher_to_pred, teacher_ok, pred_error, pred_ok


def _top_fraction(values: torch.Tensor, fraction: float) -> torch.Tensor:
    if values.numel() == 0:
        raise ValueError("tail objective requires supported teacher modes")
    k = max(1, int(math.ceil(float(fraction) * values.numel())))
    return torch.topk(values.reshape(-1), k=k, largest=True).values.mean()


def iris_v2_loss(
    output: IrisReprojectionOutputV2,
    domain: RayHypothesisDomainV2,
    teacher_depth: torch.Tensor,
    teacher_support: torch.Tensor,
    *,
    weights: IrisV2LossWeights = IrisV2LossWeights(),
) -> dict[str, torch.Tensor]:
    """Multimodal depth/support supervision with metric-aligned hard-tail pressure.

    P/N truth losses remain deliberately absent. Rank-2 teacher depth remains a
    supported compatibility lane, but the objective no longer assumes that the
    representation itself has only one admissible ray mode.
    """
    weights.validate()
    td, tv, q_supported = _teacher_modes(teacher_depth, teacher_support, domain)
    logits = output.field.score_logits
    mode_nll = _multitarget_mode_nll(logits, domain.depth_values, td, tv)

    teacher_error, teacher_to_pred, teacher_ok, pred_error, pred_ok = _mode_matching_errors(output, td, tv)
    if not teacher_ok.any():
        raise ValueError("IRIS V2 loss requires at least one supported teacher mode")
    coverage = teacher_error[teacher_ok].mean()
    # With a complete multimodal teacher, also penalize unsupported extra modes.
    # The legacy rank-2 lane intentionally does not collapse alternative modes.
    multimodal_teacher = td.shape[-1] > 1
    if multimodal_teacher and pred_ok.any():
        coverage = 0.5 * (coverage + pred_error[pred_ok].mean())

    # Calibrate uncertainty on the predicted mode nearest each supported teacher mode.
    gathered_sigma = torch.gather(output.depth_output.log_sigma, -1, teacher_to_pred.clamp_min(0))
    residual = torch.gather(output.refined_depth, -1, teacher_to_pred.clamp_min(0)) - td
    nll = 0.5 * torch.exp(-2.0 * gathered_sigma) * residual.square() + gathered_sigma
    uncertainty_nll = nll[teacher_ok].mean()

    # Mode-level support: a predicted peak is positive iff it is close to at least
    # one supported teacher intersection. Unsupported q rays make all modes negative.
    K = output.refined_depth.shape[-1]
    if domain.depth_values.shape[-1] > 1:
        spacing = (domain.depth_values[..., 1:] - domain.depth_values[..., :-1]).abs().median().clamp_min(1e-6)
    else:
        spacing = torch.tensor(1e-3, device=logits.device, dtype=logits.dtype)
    nearest_teacher = torch.where(tv[..., None, :], (output.refined_depth[..., :, None] - td[..., None, :]).abs(), float("inf")).min(dim=-1).values
    support_target = (nearest_teacher <= 1.5 * spacing) & q_supported[..., None] & output.modes.mode_valid.bool()
    support_logits = output.depth_output.support_logits
    support_mask = output.modes.mode_valid.bool() | (~q_supported[..., None]).expand_as(output.modes.mode_valid)
    if support_mask.any():
        support = F.binary_cross_entropy_with_logits(support_logits[support_mask], support_target.to(support_logits.dtype)[support_mask])
    else:
        support = support_logits.sum() * 0.0

    tail = _top_fraction(teacher_error[teacher_ok], weights.tail_fraction)
    regularizer = isotropic_world_regularizer_v2(
        domain.q_points,
        logits,
        valid_mask=domain.candidate_valid,
        neighbor_indices=output.field.neighbor_indices,
        neighbor_mask=output.field.neighbor_mask,
    )
    total = (
        weights.depth_mode * mode_nll
        + weights.refined_depth * coverage
        + weights.uncertainty_nll * uncertainty_nll
        + weights.support * support
        + weights.tail_depth * tail
        + weights.world_regularizer * regularizer
    )
    return {
        "total": total,
        "depth_mode": mode_nll,
        "refined_depth": coverage,
        "uncertainty_nll": uncertainty_nll,
        "support": support,
        "tail_depth": tail,
        "world_regularizer": regularizer,
    }


def train_step(model, optimizer, batch: dict) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(batch["images"], batch["foundation_maps"], batch["domain"])
    losses = iris_v2_loss(output, batch["domain"], batch["teacher_depth"], batch["teacher_support"])
    losses["total"].backward()
    optimizer.step()
    return {k: float(v.detach().cpu()) for k, v in losses.items()}
