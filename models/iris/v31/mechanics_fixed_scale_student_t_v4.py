"""Identifiable robust mechanics objective and shortcut courts for IRIS V3.1 Truth139.

The historical heteroscedastic objective allowed the learner to predict both a location and a
per-query scale.  That is useful for conditional-density modelling but makes a strong NLL alone
insufficient evidence that scene-conditioned mechanics was learned.  Truth139 therefore freezes
one robust scale per mechanics channel from TRAIN rows only and optimizes location only.

The existing four-channel head is retained for checkpoint/architecture compatibility.  Channels
1 and 3 (historical log-scale outputs) are intentionally unused by this objective and receive zero
gradient.  A future architecture cleanup may remove them after this scientific boundary closes.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .presentation_objective_v1 import MECHANICAL_CHANNELS_V31_PRESENTATION


OBJECTIVE_ID_V4 = "FIXED_SCALE_STUDENT_T_LOCATION_ONLY__LOG1P__DF3"


def _means(raw: torch.Tensor) -> torch.Tensor:
    if raw.shape[-1] != len(MECHANICAL_CHANNELS_V31_PRESENTATION):
        raise ValueError("IRIS_V31_MECHANICS_V4_RAW_SHAPE")
    return torch.stack((raw[..., 0], raw[..., 2]), dim=-1)


def robust_fixed_scale_v4(
    transformed_target: torch.Tensor | np.ndarray,
    valid: torch.Tensor | np.ndarray,
    *,
    minimum_scale: float,
) -> tuple[float, float]:
    """Return TRAIN-only channel scales in log1p target coordinates using 1.4826*MAD."""
    if not math.isfinite(float(minimum_scale)) or float(minimum_scale) <= 0.0:
        raise ValueError("IRIS_V31_MECHANICS_V4_SCALE_FLOOR")
    target = np.asarray(
        transformed_target.detach().cpu().numpy()
        if isinstance(transformed_target, torch.Tensor)
        else transformed_target,
        dtype=np.float64,
    )
    mask = np.asarray(
        valid.detach().cpu().numpy() if isinstance(valid, torch.Tensor) else valid,
        dtype=bool,
    )
    if target.ndim < 2 or target.shape[-1] != 2 or mask.shape != target.shape[:-1]:
        raise ValueError("IRIS_V31_MECHANICS_V4_SCALE_SHAPE")
    values = target[mask]
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("IRIS_V31_MECHANICS_V4_SCALE_EMPTY_OR_NONFINITE")
    if np.any(values < 0.0):
        raise ValueError("IRIS_V31_MECHANICS_V4_LOG1P_TARGET_NEGATIVE")
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median[None, :]), axis=0)
    scale = np.maximum(1.4826 * mad, float(minimum_scale))
    return float(scale[0]), float(scale[1])


def fixed_scale_student_t_mechanical_loss_v4(
    raw: torch.Tensor,
    transformed_target: torch.Tensor,
    valid: torch.Tensor,
    *,
    fixed_scale: tuple[float, float] | list[float] | torch.Tensor,
    degrees_of_freedom: float = 3.0,
) -> torch.Tensor:
    """Student-t NLL where only the conditional location is learnable.

    `transformed_target` is the existing log1p(non-negative magnitude) target.  `fixed_scale`
    must be sealed from TRAIN rows before the optimizer is created.  No per-query uncertainty
    output participates in this loss.
    """
    mean = _means(raw)
    if transformed_target.shape != mean.shape or valid.shape != mean.shape[:-1]:
        raise ValueError("IRIS_V31_MECHANICS_V4_LOSS_SHAPE")
    if not math.isfinite(float(degrees_of_freedom)) or float(degrees_of_freedom) <= 2.0:
        raise ValueError("IRIS_V31_MECHANICS_V4_DF")
    mask = valid.bool()
    if not bool(mask.any()):
        raise ValueError("IRIS_V31_MECHANICS_V4_LOSS_EMPTY")
    if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(transformed_target).all()):
        raise ValueError("IRIS_V31_MECHANICS_V4_NONFINITE")

    scale = torch.as_tensor(fixed_scale, device=mean.device, dtype=mean.dtype).reshape(-1)
    if scale.shape != (2,) or not bool(torch.isfinite(scale).all()) or bool(torch.any(scale <= 0.0)):
        raise ValueError("IRIS_V31_MECHANICS_V4_FIXED_SCALE_INVALID")

    target = transformed_target.to(mean)
    residual = target - mean
    nu = mean.new_tensor(float(degrees_of_freedom))
    constant = (
        0.5 * torch.log(nu * mean.new_tensor(math.pi))
        + torch.lgamma(0.5 * nu)
        - torch.lgamma(0.5 * (nu + 1.0))
    )
    scaled_square = residual.square() / (nu * scale.square())
    nll = constant + torch.log(scale) + 0.5 * (nu + 1.0) * torch.log1p(scaled_square)
    return nll[mask].mean()


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        average = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average
        start = end
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return 0.0
    ra, rb = _average_ranks(a), _average_ranks(b)
    sa, sb = float(ra.std()), float(rb.std())
    if sa <= 0.0 or sb <= 0.0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def mechanics_regression_metrics_v4(
    raw: torch.Tensor | np.ndarray,
    transformed_target: torch.Tensor | np.ndarray,
    valid: torch.Tensor | np.ndarray,
    *,
    tail_quantile: float = 0.90,
) -> dict:
    """Metrics that cannot be made good solely by inflating predicted uncertainty."""
    raw_np = np.asarray(raw.detach().cpu().numpy() if isinstance(raw, torch.Tensor) else raw)
    target = np.asarray(
        transformed_target.detach().cpu().numpy()
        if isinstance(transformed_target, torch.Tensor)
        else transformed_target,
        dtype=np.float64,
    )
    mask = np.asarray(valid.detach().cpu().numpy() if isinstance(valid, torch.Tensor) else valid, dtype=bool)
    if raw_np.shape[-1] != 4 or target.shape != raw_np.shape[:-1] + (2,) or mask.shape != target.shape[:-1]:
        raise ValueError("IRIS_V31_MECHANICS_V4_METRIC_SHAPE")
    if not 0.5 <= float(tail_quantile) < 1.0:
        raise ValueError("IRIS_V31_MECHANICS_V4_TAIL_QUANTILE")

    prediction = np.stack((raw_np[..., 0], raw_np[..., 2]), axis=-1).astype(np.float64)[mask]
    truth = target[mask]
    if not len(truth) or not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError("IRIS_V31_MECHANICS_V4_METRIC_EMPTY_OR_NONFINITE")
    raw_prediction = np.maximum(np.expm1(prediction), 0.0)
    raw_truth = np.expm1(truth)

    channels = []
    for channel in range(2):
        residual = prediction[:, channel] - truth[:, channel]
        raw_residual = raw_prediction[:, channel] - raw_truth[:, channel]
        threshold = float(np.quantile(raw_truth[:, channel], float(tail_quantile)))
        tail = raw_truth[:, channel] >= threshold
        channels.append(
            {
                "log1p_mae": float(np.mean(np.abs(residual))),
                "log1p_rmse": float(np.sqrt(np.mean(residual * residual))),
                "raw_mae": float(np.mean(np.abs(raw_residual))),
                "spearman": _spearman(prediction[:, channel], truth[:, channel]),
                "tail_quantile": float(tail_quantile),
                "tail_threshold_raw": threshold,
                "tail_count": int(np.count_nonzero(tail)),
                "tail_raw_mae": float(np.mean(np.abs(raw_residual[tail]))),
            }
        )
    return {
        "count": int(len(truth)),
        "channels": channels,
        "mean_log1p_mae": float(np.mean([row["log1p_mae"] for row in channels])),
        "mean_log1p_rmse": float(np.mean([row["log1p_rmse"] for row in channels])),
        "mean_raw_mae": float(np.mean([row["raw_mae"] for row in channels])),
        "mean_spearman": float(np.mean([row["spearman"] for row in channels])),
        "mean_tail_raw_mae": float(np.mean([row["tail_raw_mae"] for row in channels])),
    }


def constant_prior_raw_v4(train_transformed_target: torch.Tensor, train_valid: torch.Tensor, shape) -> torch.Tensor:
    """Build a train-only constant median predictor in the existing four-channel layout."""
    target = train_transformed_target.float()
    mask = train_valid.bool()
    if target.shape[-1] != 2 or mask.shape != target.shape[:-1] or not bool(mask.any()):
        raise ValueError("IRIS_V31_MECHANICS_V4_PRIOR_SHAPE")
    values = target[mask]
    median = torch.quantile(values, 0.5, dim=0)
    out = torch.zeros(tuple(shape), dtype=target.dtype, device=target.device)
    if out.shape[-1] != 4:
        raise ValueError("IRIS_V31_MECHANICS_V4_PRIOR_OUTPUT_SHAPE")
    out[..., 0] = median[0]
    out[..., 2] = median[1]
    return out


def conditioning_gain_report_v4(model: dict, train_prior: dict, shuffled_scene: dict) -> dict:
    """Report shortcut resistance without inventing a scientific promotion threshold."""
    for name, row in (("model", model), ("train_prior", train_prior), ("shuffled_scene", shuffled_scene)):
        for key in ("mean_log1p_mae", "mean_raw_mae", "mean_tail_raw_mae"):
            value = float(row[key])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("IRIS_V31_MECHANICS_V4_COURT_INVALID:" + name + ":" + key)

    def gain(baseline: dict, key: str) -> float:
        denom = float(baseline[key])
        if denom <= 0.0:
            return 0.0
        return float((denom - float(model[key])) / denom)

    return {
        "model": model,
        "train_prior": train_prior,
        "shuffled_scene": shuffled_scene,
        "relative_gain_vs_train_prior_log1p_mae": gain(train_prior, "mean_log1p_mae"),
        "relative_gain_vs_shuffled_scene_log1p_mae": gain(shuffled_scene, "mean_log1p_mae"),
        "relative_gain_vs_train_prior_tail_raw_mae": gain(train_prior, "mean_tail_raw_mae"),
        "relative_gain_vs_shuffled_scene_tail_raw_mae": gain(shuffled_scene, "mean_tail_raw_mae"),
        "qualification_threshold_applied": False,
        "pass_claimed": False,
    }
