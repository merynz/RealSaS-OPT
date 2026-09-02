from __future__ import annotations

import math
import torch


def row_l1_cvar_v1(
    predicted_weights: torch.Tensor,
    teacher_weights: torch.Tensor,
    surface_mask: torch.Tensor,
    *,
    tail_fraction: float = 0.10,
) -> torch.Tensor:
    """Generic row-level hard-tail objective used only after a frozen A1 p95 failure.

    It contains no family, joint-name, source-id, or semantic special case. The
    tail is defined over per-surface simplex-row L1 reconstruction error.
    """
    if predicted_weights.shape != teacher_weights.shape:
        raise ValueError("Arachne tail remediation weight shape mismatch")
    if surface_mask.shape != predicted_weights.shape[:2]:
        raise ValueError("Arachne tail remediation surface mask mismatch")
    if not (0.0 < float(tail_fraction) <= 1.0):
        raise ValueError("tail_fraction must be in (0,1]")
    row_l1 = (predicted_weights - teacher_weights).abs().sum(dim=-1)
    valid = row_l1[surface_mask.bool()]
    if valid.numel() == 0:
        raise ValueError("Arachne tail remediation requires valid surface rows")
    k = max(1, int(math.ceil(float(tail_fraction) * valid.numel())))
    return torch.topk(valid, k=k, largest=True).values.mean()


def mean_field_remediation_loss_v1(
    *,
    predicted_latent_mean: torch.Tensor,
    teacher_latents: torch.Tensor,
    predicted_weights: torch.Tensor,
    teacher_weights: torch.Tensor,
    surface_mask: torch.Tensor,
    reconstruction_loss: torch.Tensor,
    deformation_mse: torch.Tensor,
    tail_fraction: float = 0.10,
    tail_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Metric-aligned A1 remediation after the frozen base ladder fails on p95.

    Uncertainty calibration is intentionally excluded from this stage. The
    caller freezes the uncertainty head, stabilizes the latent mean/field, then
    runs a separate heteroscedastic calibration stage with the mean path frozen.
    """
    if tail_weight < 0.0:
        raise ValueError("tail_weight must be non-negative")
    if predicted_latent_mean.shape != teacher_latents.shape:
        raise ValueError("latent shape mismatch")
    latent_mse = (predicted_latent_mean - teacher_latents).square().mean()
    tail = row_l1_cvar_v1(
        predicted_weights,
        teacher_weights,
        surface_mask,
        tail_fraction=tail_fraction,
    )
    total = latent_mse + reconstruction_loss + deformation_mse + float(tail_weight) * tail
    if not torch.isfinite(total):
        raise ValueError("non-finite Arachne tail remediation loss")
    return {
        "total": total,
        "latent_mse": latent_mse,
        "row_l1_cvar": tail,
        "reconstruction": reconstruction_loss,
        "deformation_mse": deformation_mse,
    }


def heteroscedastic_calibration_loss_v1(
    predicted_latent_mean: torch.Tensor,
    predicted_latent_log_sigma: torch.Tensor,
    teacher_latents: torch.Tensor,
    joint_mask: torch.Tensor,
) -> torch.Tensor:
    """Second-stage uncertainty calibration with mean path held fixed by caller."""
    if predicted_latent_mean.shape != predicted_latent_log_sigma.shape or predicted_latent_mean.shape != teacher_latents.shape:
        raise ValueError("Arachne uncertainty calibration latent shape mismatch")
    valid = joint_mask[..., None].bool().expand_as(predicted_latent_mean)
    if not valid.any():
        raise ValueError("Arachne uncertainty calibration requires valid joints")
    residual = (predicted_latent_mean - teacher_latents).detach()
    ls = predicted_latent_log_sigma
    nll = (0.5 * torch.exp(-2.0 * ls) * residual.square() + ls)[valid].mean()
    if not torch.isfinite(nll):
        raise ValueError("non-finite Arachne uncertainty calibration loss")
    return nll
