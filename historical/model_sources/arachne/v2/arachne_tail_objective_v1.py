from __future__ import annotations

import math
import torch


def row_l1_error_v1(decoded_weights: torch.Tensor, teacher_weights: torch.Tensor, surface_mask: torch.Tensor) -> torch.Tensor:
    """Per-surface simplex-row L1 error over valid admitted surface rows."""
    if decoded_weights.shape != teacher_weights.shape:
        raise ValueError("decoded/teacher weight shape mismatch")
    if surface_mask.shape != decoded_weights.shape[:2]:
        raise ValueError("surface mask shape mismatch")
    rows = (decoded_weights - teacher_weights).abs().sum(dim=-1)
    return rows[surface_mask.bool()]


def top_fraction_row_l1_tail_v1(
    decoded_weights: torch.Tensor,
    teacher_weights: torch.Tensor,
    surface_mask: torch.Tensor,
    *,
    fraction: float = 0.10,
) -> torch.Tensor:
    """Generic hard-tail objective for skin-field reconstruction.

    Selects the largest fraction of *current* valid row-L1 errors. The rule uses
    no family, semantic slot, joint name, source index, or product ID and is
    therefore compatible with the generic RealSaS claim. It is a training-only
    objective; qualification thresholds remain external and unchanged.
    """
    if not (0.0 < float(fraction) <= 1.0):
        raise ValueError("tail fraction must lie in (0,1]")
    row = row_l1_error_v1(decoded_weights, teacher_weights, surface_mask)
    if row.numel() == 0:
        raise ValueError("tail objective requires at least one valid surface row")
    k = max(1, int(math.ceil(float(fraction) * row.numel())))
    return torch.topk(row, k, largest=True).values.mean()


def arachne_field_remediation_loss_v1(
    *,
    latent_mean: torch.Tensor,
    teacher_latent: torch.Tensor,
    decoded_weights: torch.Tensor,
    teacher_weights: torch.Tensor,
    surface_mask: torch.Tensor,
    joint_mask: torch.Tensor,
    reconstruction_loss: torch.Tensor,
    deformation_mse: torch.Tensor,
    tail_fraction: float = 0.10,
    tail_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Metric-aligned generic A1 field remediation objective.

    The uncertainty head is intentionally excluded here and should be calibrated
    in a separate stage after the decoded field is frozen. This prevents sigma
    optimization from compensating for field-reconstruction error.
    """
    if tail_weight < 0:
        raise ValueError("tail_weight must be non-negative")
    valid_latent = joint_mask[..., None].bool().expand_as(latent_mean)
    if latent_mean.shape != teacher_latent.shape or not valid_latent.any():
        raise ValueError("latent shape/mask mismatch")
    latent_mse = (latent_mean - teacher_latent).square()[valid_latent].mean()
    tail = top_fraction_row_l1_tail_v1(
        decoded_weights,
        teacher_weights,
        surface_mask,
        fraction=tail_fraction,
    )
    total = latent_mse + reconstruction_loss + float(tail_weight) * tail + deformation_mse
    if not torch.isfinite(total):
        raise ValueError("non-finite Arachne field remediation loss")
    return {
        "total": total,
        "latent_mse": latent_mse,
        "tail_row_l1": tail,
        "reconstruction": reconstruction_loss,
        "deformation_mse": deformation_mse,
    }
