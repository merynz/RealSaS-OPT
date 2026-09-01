from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .geppetto_demo_v1 import GeppettoRawOutput


IRIS_RMS_MAX = 0.00250
IRIS_P95_MAX = 0.00539
GEPPETTO_PCK_RADIUS_FRAC = 0.05
GEPPETTO_RMS_MAX_FRAC = 0.01
ARACHNE_MEAN_ROW_L1_MAX = 0.02
ARACHNE_P95_ROW_L1_MAX = 0.05
ARACHNE_SIMPLEX_P95_MAX = 0.02


@dataclass(frozen=True)
class FitVerdict:
    passed: bool
    metrics: dict
    blockers: tuple[str, ...]


def iris_fit_loss(
    depth_pred: torch.Tensor,
    support_logits: torch.Tensor,
    log_sigma: torch.Tensor,
    teacher_depth: torch.Tensor,
    teacher_support: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Demo-only IRIS fit loss using geometry/support truth, never consumer labels."""
    if depth_pred.shape != teacher_depth.shape:
        raise ValueError("IRIS depth shape mismatch")
    if support_logits.shape != teacher_support.shape or log_sigma.shape != teacher_depth.shape:
        raise ValueError("IRIS support/uncertainty shape mismatch")
    mask = teacher_support.bool()
    if not bool(mask.any()):
        raise ValueError("IRIS teacher has no supported target samples")
    err = depth_pred[mask] - teacher_depth[mask]
    sigma_term = torch.exp(-2.0 * log_sigma[mask])
    depth_nll = (0.5 * sigma_term * err.square() + log_sigma[mask]).mean()
    depth_l1 = err.abs().mean()
    support_bce = F.binary_cross_entropy_with_logits(support_logits, teacher_support.float())
    total = depth_nll + 0.25 * depth_l1 + 0.5 * support_bce
    return total, {"depth_nll": depth_nll, "depth_l1": depth_l1, "support_bce": support_bce}


@torch.no_grad()
def evaluate_iris_fit(
    depth_pred: torch.Tensor,
    support_logits: torch.Tensor,
    teacher_depth: torch.Tensor,
    teacher_support: torch.Tensor,
    *,
    support_threshold: float = 0.5,
) -> FitVerdict:
    if depth_pred.shape != teacher_depth.shape or support_logits.shape != teacher_support.shape:
        raise ValueError("IRIS evaluator shape mismatch")
    finite = torch.isfinite(depth_pred) & torch.isfinite(support_logits)
    target = teacher_support.bool()
    pred_support = torch.sigmoid(support_logits) >= support_threshold
    target_count = int(target.sum().item())
    if target_count == 0:
        raise ValueError("IRIS evaluator target support is empty")
    visible_err = (depth_pred - teacher_depth).abs()[target]
    rms = float(torch.sqrt(torch.mean(visible_err.square())).item())
    p95 = float(torch.quantile(visible_err.float(), 0.95).item())
    recall = float((pred_support & target).sum().item() / target_count)
    invented = int((pred_support & ~target).sum().item())
    finite_fraction = float(finite.float().mean().item())
    blockers = []
    if finite_fraction != 1.0: blockers.append("IRIS_NONFINITE")
    if recall != 1.0: blockers.append("IRIS_REQUIRED_SUPPORT_RECALL")
    if invented != 0: blockers.append("IRIS_INVENTED_UNKNOWN_SUPPORT")
    if rms > IRIS_RMS_MAX: blockers.append("IRIS_DEPTH_RMS")
    if p95 > IRIS_P95_MAX: blockers.append("IRIS_DEPTH_P95")
    return FitVerdict(
        passed=not blockers,
        metrics={
            "finite_output_fraction": finite_fraction,
            "required_target_support_recall": recall,
            "invented_unknown_support_count": invented,
            "depth_rms": rms,
            "depth_abs_p95": p95,
            "target_support_count": target_count,
        },
        blockers=tuple(blockers),
    )


def _normalize_to_surface(points: torch.Tensor, surface_points: torch.Tensor) -> torch.Tensor:
    pmin = surface_points.amin(dim=0, keepdim=True)
    pmax = surface_points.amax(dim=0, keepdim=True)
    center = 0.5 * (pmin + pmax)
    scale = (pmax - pmin).amax().clamp_min(1e-6)
    return (points - center) / scale


def _hungarian(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if pred.ndim != 2 or target.ndim != 2 or pred.shape[1] != 3 or target.shape[1] != 3:
        raise ValueError("joint matrices must be [N,3]")
    if len(target) > len(pred):
        raise ValueError("teacher joint count exceeds Geppetto query capacity")
    cost = torch.cdist(pred, target).detach().cpu().numpy()
    row, col = linear_sum_assignment(cost)
    order = np.argsort(col)
    row = row[order]
    col = col[order]
    return (
        torch.tensor(row, dtype=torch.long, device=pred.device),
        torch.tensor(col, dtype=torch.long, device=pred.device),
    )


def geppetto_fit_loss(
    raw: GeppettoRawOutput,
    surface_points: torch.Tensor,
    teacher_positions_world: torch.Tensor,
    teacher_parent_index: torch.Tensor,
    teacher_root_index: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]]:
    """Permutation-free single-specimen fit objective over generic anonymous teacher controls."""
    if raw.position_norm.shape[0] != 1:
        raise ValueError("demo Geppetto fitter currently requires B=1")
    if teacher_parent_index.shape != (len(teacher_positions_world),):
        raise ValueError("teacher parent shape mismatch")
    teacher_norm = _normalize_to_surface(teacher_positions_world, surface_points)
    pred = raw.position_norm[0]
    matched_q, matched_t = _hungarian(pred, teacher_norm)

    existence_target = torch.zeros_like(raw.existence_logits[0])
    existence_target[matched_q] = 1.0
    existence_loss = F.binary_cross_entropy_with_logits(raw.existence_logits[0], existence_target)
    position_loss = F.smooth_l1_loss(pred[matched_q], teacher_norm[matched_t])

    root_target = torch.zeros_like(raw.root_logits[0])
    teacher_to_query = {int(t): int(q) for q, t in zip(matched_q.tolist(), matched_t.tolist())}
    root_target[teacher_to_query[int(teacher_root_index)]] = 1.0
    root_loss = F.binary_cross_entropy_with_logits(raw.root_logits[0], root_target)

    active_q = matched_q
    edge_pred = raw.edge_logits[0][active_q][:, active_q]
    edge_target = torch.zeros_like(edge_pred)
    query_to_local = {int(q): i for i, q in enumerate(active_q.tolist())}
    for child_t, parent_t in enumerate(teacher_parent_index.tolist()):
        if parent_t < 0:
            continue
        parent_q = teacher_to_query[int(parent_t)]
        child_q = teacher_to_query[int(child_t)]
        edge_target[query_to_local[parent_q], query_to_local[child_q]] = 1.0
    diag = torch.eye(len(active_q), dtype=torch.bool, device=edge_target.device)
    edge_loss = F.binary_cross_entropy_with_logits(edge_pred[~diag], edge_target[~diag]) if len(active_q) > 1 else edge_pred.sum() * 0.0

    total = 4.0 * position_loss + existence_loss + root_loss + 2.0 * edge_loss
    return total, {
        "position": position_loss,
        "existence": existence_loss,
        "root": root_loss,
        "edge": edge_loss,
        "matching": (matched_q, matched_t),
    }


@torch.no_grad()
def evaluate_geppetto_fit(
    predicted_positions_world: torch.Tensor,
    predicted_parent_index: torch.Tensor,
    predicted_root_index: int,
    teacher_positions_world: torch.Tensor,
    teacher_parent_index: torch.Tensor,
    teacher_root_index: int,
    *,
    subject_bbox_min: torch.Tensor,
    subject_bbox_max: torch.Tensor,
) -> FitVerdict:
    if len(predicted_positions_world) != len(teacher_positions_world):
        return FitVerdict(False, {"predicted_count": len(predicted_positions_world), "teacher_count": len(teacher_positions_world)}, ("GEPPETTO_COUNT_MISMATCH",))
    pred_i, teacher_i = _hungarian(predicted_positions_world, teacher_positions_world)
    error = torch.linalg.norm(predicted_positions_world[pred_i] - teacher_positions_world[teacher_i], dim=-1)
    diag = torch.linalg.norm(subject_bbox_max - subject_bbox_min).clamp_min(1e-8)
    error_frac = error / diag
    rms_frac = float(torch.sqrt(torch.mean(error_frac.square())).item())
    pck = float((error_frac <= GEPPETTO_PCK_RADIUS_FRAC).float().mean().item())

    teacher_to_pred = {int(t): int(p) for p, t in zip(pred_i.tolist(), teacher_i.tolist())}
    root_ok = int(predicted_root_index) == teacher_to_pred[int(teacher_root_index)]
    edge_total = max(0, len(teacher_positions_world) - 1)
    edge_correct = 0
    for child_t, parent_t in enumerate(teacher_parent_index.tolist()):
        if parent_t < 0:
            continue
        child_p = teacher_to_pred[int(child_t)]
        parent_p = teacher_to_pred[int(parent_t)]
        if int(predicted_parent_index[child_p].item()) == parent_p:
            edge_correct += 1
    edge_accuracy = 1.0 if edge_total == 0 else edge_correct / edge_total
    blockers = []
    if pck != 1.0: blockers.append("GEPPETTO_PCK")
    if rms_frac > GEPPETTO_RMS_MAX_FRAC: blockers.append("GEPPETTO_RMS")
    if not root_ok: blockers.append("GEPPETTO_ROOT")
    if edge_accuracy != 1.0: blockers.append("GEPPETTO_PARENT_EDGES")
    return FitVerdict(
        passed=not blockers,
        metrics={
            "joint_count": len(predicted_positions_world),
            "pck_at_0.05D": pck,
            "joint_rms_over_D": rms_frac,
            "root_accuracy": float(root_ok),
            "directed_parent_edge_accuracy": edge_accuracy,
            "max_joint_error_over_D": float(error_frac.max().item()) if len(error_frac) else 0.0,
        },
        blockers=tuple(blockers),
    )


def arachne_fit_loss(logits: torch.Tensor, teacher_weights: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if logits.shape != teacher_weights.shape or logits.ndim != 2:
        raise ValueError("Arachne logits/teacher must match [N,J]")
    if (teacher_weights < 0).any():
        raise ValueError("teacher weights contain negatives")
    denom = teacher_weights.sum(dim=-1, keepdim=True)
    if (denom <= 1e-12).any():
        raise ValueError("teacher contains zero skin row")
    teacher = teacher_weights / denom
    logp = torch.log_softmax(logits, dim=-1)
    pred = torch.softmax(logits, dim=-1)
    soft_ce = -(teacher * logp).sum(dim=-1).mean()
    l1 = (pred - teacher).abs().sum(dim=-1).mean()
    total = soft_ce + 2.0 * l1
    return total, {"soft_cross_entropy": soft_ce, "mean_row_l1": l1}


@torch.no_grad()
def evaluate_arachne_weights(predicted_weights: torch.Tensor, teacher_weights: torch.Tensor) -> FitVerdict:
    if predicted_weights.shape != teacher_weights.shape or predicted_weights.ndim != 2:
        raise ValueError("Arachne evaluator matrices must match [N,J]")
    finite = bool(torch.isfinite(predicted_weights).all())
    negative = int((predicted_weights < 0).sum().item())
    row_sum = predicted_weights.sum(dim=-1)
    simplex = (row_sum - 1.0).abs()
    teacher = teacher_weights / teacher_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    row_l1 = (predicted_weights - teacher).abs().sum(dim=-1)
    mean_l1 = float(row_l1.mean().item())
    p95_l1 = float(torch.quantile(row_l1.float(), 0.95).item())
    simplex_p95 = float(torch.quantile(simplex.float(), 0.95).item())
    blockers = []
    if not finite: blockers.append("ARACHNE_NONFINITE")
    if negative: blockers.append("ARACHNE_NEGATIVE_WEIGHT")
    if mean_l1 > ARACHNE_MEAN_ROW_L1_MAX: blockers.append("ARACHNE_MEAN_ROW_L1")
    if p95_l1 > ARACHNE_P95_ROW_L1_MAX: blockers.append("ARACHNE_P95_ROW_L1")
    if simplex_p95 > ARACHNE_SIMPLEX_P95_MAX: blockers.append("ARACHNE_SIMPLEX_P95")
    return FitVerdict(
        passed=not blockers,
        metrics={
            "row_count": int(predicted_weights.shape[0]),
            "finite": finite,
            "negative_weight_count": negative,
            "mean_row_l1": mean_l1,
            "row_l1_p95": p95_l1,
            "simplex_residual_p95": simplex_p95,
        },
        blockers=tuple(blockers),
    )
