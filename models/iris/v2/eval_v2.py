from __future__ import annotations

import math
import torch

from .model_v2 import IrisReprojectionOutputV2
from .q_domain_v2 import RayHypothesisDomainV2
from .train_v2 import _teacher_modes, _mode_matching_errors
from .world_regularizer_v2 import isotropic_world_regularizer_v2


def depth_metrics_v2(predicted_depth: torch.Tensor, teacher_depth: torch.Tensor, support: torch.Tensor) -> dict[str, float]:
    """Legacy single-depth compatibility metrics."""
    mask = support.bool()
    if not mask.any():
        return {"count": 0, "mae": float("nan"), "rmse": float("nan"), "p95_abs": float("nan")}
    err = (predicted_depth - teacher_depth)[mask].detach().float().abs().cpu()
    return {
        "count": int(err.numel()),
        "mae": float(err.mean()),
        "rmse": float(torch.sqrt((err.square()).mean())),
        "p95_abs": float(torch.quantile(err, 0.95)),
    }


def _top_fraction_mean(values: torch.Tensor, fraction: float) -> float:
    values = values.detach().float().reshape(-1).cpu()
    if values.numel() == 0:
        return float("nan")
    k = max(1, int(math.ceil(float(fraction) * values.numel())))
    return float(torch.topk(values, k=k, largest=True).values.mean())


def iris_v2_scientific_metrics(
    output: IrisReprojectionOutputV2,
    domain: RayHypothesisDomainV2,
    teacher_depth: torch.Tensor,
    teacher_support: torch.Tensor,
    *,
    support_probability_threshold: float = 0.5,
    tail_fraction: float = 0.10,
) -> dict[str, float]:
    """Consumer-aligned evaluation for multimodal IRIS evidence.

    Reports teacher-mode coverage, hard-tail error, unsupported predicted modes,
    support classification, uncertainty calibration and the same sparse world-space
    consistency term used by training. This evaluator never collapses a multimodal
    teacher to one scalar target and never requires P/N truth.
    """
    if not (0.0 < support_probability_threshold < 1.0):
        raise ValueError("support_probability_threshold must be in (0,1)")
    if not (0.0 < tail_fraction <= 1.0):
        raise ValueError("tail_fraction must be in (0,1]")

    td, tv, q_supported = _teacher_modes(teacher_depth, teacher_support, domain)
    teacher_error, teacher_to_pred, teacher_ok, pred_error, pred_ok = _mode_matching_errors(output, td, tv)
    if not teacher_ok.any():
        raise ValueError("scientific metrics require at least one supported teacher mode")

    err = teacher_error[teacher_ok].detach().float()
    matched_pred = torch.gather(output.refined_depth, -1, teacher_to_pred.clamp_min(0))
    matched_log_sigma = torch.gather(output.depth_output.log_sigma, -1, teacher_to_pred.clamp_min(0))
    residual = (matched_pred - td).abs()
    sigma = torch.exp(matched_log_sigma).clamp_min(1e-8)
    z = residual / sigma
    z_valid = z[teacher_ok].detach().float()

    if domain.depth_values.shape[-1] > 1:
        spacing = (domain.depth_values[..., 1:] - domain.depth_values[..., :-1]).abs().median().clamp_min(1e-6)
    else:
        spacing = torch.tensor(1e-3, device=td.device, dtype=td.dtype)
    tolerance = 1.5 * spacing

    nearest_teacher = torch.where(
        tv[..., None, :],
        (output.refined_depth[..., :, None] - td[..., None, :]).abs(),
        float("inf"),
    ).min(dim=-1).values
    predicted_mode_valid = output.modes.mode_valid.bool()
    support_target = (nearest_teacher <= tolerance) & q_supported[..., None] & predicted_mode_valid
    support_pred = (output.depth_output.support_probability >= float(support_probability_threshold)) & predicted_mode_valid
    tp = (support_pred & support_target).sum().float()
    fp = (support_pred & ~support_target).sum().float()
    fn = (~support_pred & support_target).sum().float()
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)

    mode_recall = (err <= tolerance).float().mean()
    false_mode_error = pred_error[pred_ok].detach().float() if pred_ok.any() else torch.empty(0)
    world = isotropic_world_regularizer_v2(
        domain.q_points,
        output.field.score_logits,
        valid_mask=domain.candidate_valid,
        neighbor_indices=output.field.neighbor_indices,
        neighbor_mask=output.field.neighbor_mask,
    ).detach().float()

    return {
        "teacher_mode_count": int(teacher_ok.sum().item()),
        "predicted_valid_mode_count": int(predicted_mode_valid.sum().item()),
        "coverage_mae": float(err.mean()),
        "coverage_rmse": float(torch.sqrt(err.square().mean())),
        "coverage_p95_abs": float(torch.quantile(err.cpu(), 0.95)),
        "coverage_tail_mean": _top_fraction_mean(err, tail_fraction),
        "teacher_mode_recall_at_1p5_spacing": float(mode_recall),
        "predicted_mode_nearest_teacher_mae": float(false_mode_error.mean()) if false_mode_error.numel() else float("nan"),
        "support_precision": float(precision),
        "support_recall": float(recall),
        "support_f1": float(f1),
        "uncertainty_mean_abs_z": float(z_valid.mean()),
        "uncertainty_coverage_1sigma": float((z_valid <= 1.0).float().mean()),
        "uncertainty_coverage_2sigma": float((z_valid <= 2.0).float().mean()),
        "world_consistency_regularizer": float(world),
        "depth_spacing": float(spacing.detach().cpu()),
        "support_tolerance": float(tolerance.detach().cpu()),
    }
