from __future__ import annotations

from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F

from .model_v2 import IrisReprojectionOutputV2
from .q_domain_v2 import RayHypothesisDomainV2
from .resource_contract_v2 import require_cuda_forward_live_set_v2
from .world_regularizer_v2 import isotropic_world_regularizer_v2


DENSE_SOURCE_COVERAGE_POLICY_V2 = {
    "schema": "RealSaS.IRISDenseSourceCoveragePolicy.v2",
    "target_authority": "QUALIFIED_OBSERVATION_FOREGROUND_MASK_ONLY",
    "target_resolution": 1024,
    "bce_weight": 1.0,
    "soft_dice_weight": 1.0,
    "boundary_multiplier": 4.0,
    "boundary_radius_px": 1,
    "mask_is_forward_input": False,
    "hard_stage13_raster_gate_replaced": False,
}


def _dense_source_coverage_loss(
    output: IrisReprojectionOutputV2,
    domain: RayHypothesisDomainV2,
    source_foreground_masks: torch.Tensor,
) -> torch.Tensor:
    """Differentiable dense source-mask surrogate for the exact Stage13 hard raster gate.

    Source masks are training targets only. They never enter the learner forward/Q-domain.
    Per-ray surface probability is reconstructed from predicted mode-support probability,
    reshaped onto the sealed camera-only production lattice, then evaluated at native
    1024x1024 resolution. Final product qualification remains the exact decoded-surface
    Stage13 raster gate; this objective is only its direct dense training pressure.
    """
    if source_foreground_masks.ndim != 4 or source_foreground_masks.shape[1:] != (8, 1024, 1024):
        raise ValueError("dense source coverage targets must be [B,8,1024,1024]")
    if source_foreground_masks.shape[0] != domain.anchor_view.shape[0]:
        raise ValueError("dense source coverage batch mismatch")
    if domain.construction_authority != "RGB_CAMERA_FULL_FRAME_LATTICE_V1":
        raise ValueError("dense source coverage requires production camera-only Q domain")
    if domain.anchor_stride_px is None or int(domain.anchor_stride_px) <= 0:
        raise ValueError("dense source coverage requires sealed anchor stride")

    B, Q = domain.anchor_view.shape
    anchor_view = domain.anchor_view[:, 0]
    if not torch.equal(domain.anchor_view, anchor_view[:, None].expand_as(domain.anchor_view)):
        raise ValueError("dense source coverage requires one analytic anchor view per batch item")

    side = int(round(math.sqrt(Q)))
    if side * side != Q:
        raise ValueError("dense source coverage requires square full-frame anchor lattice")

    probability = output.depth_output.support_probability
    if probability.ndim != 3 or probability.shape[:2] != (B, Q):
        raise ValueError("dense source coverage support probability shape mismatch")
    probability = probability.clamp(0.0, 1.0)
    ray_foreground = 1.0 - torch.prod(1.0 - probability, dim=-1)
    coarse = ray_foreground.reshape(B, 1, side, side)
    dense = F.interpolate(
        coarse,
        size=(1024, 1024),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1).clamp(1e-6, 1.0 - 1e-6)

    b = torch.arange(B, device=dense.device)
    target = source_foreground_masks.to(device=dense.device, dtype=dense.dtype)[
        b, anchor_view
    ]

    radius = int(DENSE_SOURCE_COVERAGE_POLICY_V2["boundary_radius_px"])
    kernel = 2 * radius + 1
    target4 = target[:, None]
    dilated = F.max_pool2d(target4, kernel_size=kernel, stride=1, padding=radius)
    eroded = -F.max_pool2d(-target4, kernel_size=kernel, stride=1, padding=radius)
    boundary = (dilated - eroded).abs().squeeze(1).clamp(0.0, 1.0)
    pixel_weight = 1.0 + (
        float(DENSE_SOURCE_COVERAGE_POLICY_V2["boundary_multiplier"]) - 1.0
    ) * boundary

    bce = F.binary_cross_entropy(dense, target, reduction="none")
    weighted_bce = (bce * pixel_weight).sum() / pixel_weight.sum().clamp_min(1.0)

    intersection = (dense * target).sum(dim=(1, 2))
    denominator = dense.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0))
    return (
        float(DENSE_SOURCE_COVERAGE_POLICY_V2["bce_weight"]) * weighted_bce
        + float(DENSE_SOURCE_COVERAGE_POLICY_V2["soft_dice_weight"]) * dice_loss.mean()
    )


@dataclass(frozen=True)
class IrisV2LossWeights:
    depth_mode: float = 1.0
    refined_depth: float = 2.0
    uncertainty_nll: float = 0.5
    support: float = 0.5
    tail_depth: float = 0.5
    world_regularizer: float = 0.02
    dense_source_coverage: float = 1.0
    tail_fraction: float = 0.10

    def validate(self) -> None:
        numeric = (
            self.depth_mode,
            self.refined_depth,
            self.uncertainty_nll,
            self.support,
            self.tail_depth,
            self.world_regularizer,
            self.dense_source_coverage,
        )
        if min(numeric) < 0 or not (0.0 < self.tail_fraction <= 1.0):
            raise ValueError("invalid IRIS V2 loss weights")


def _teacher_modes(
    teacher_depth: torch.Tensor,
    teacher_support: torch.Tensor,
    domain: RayHypothesisDomainV2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    nearest = dist.argmin(dim=2)
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = output.refined_depth
    pred_valid = output.modes.mode_valid.bool()
    diff = (pred[..., :, None] - teacher_depth[..., None, :]).abs()
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
    source_foreground_masks: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Multimodal depth/support supervision with metric-aligned hard-tail pressure."""
    weights.validate()
    td, tv, q_supported = _teacher_modes(teacher_depth, teacher_support, domain)
    logits = output.field.score_logits
    mode_nll = _multitarget_mode_nll(logits, domain.depth_values, td, tv)

    teacher_error, teacher_to_pred, teacher_ok, pred_error, pred_ok = _mode_matching_errors(output, td, tv)
    if not teacher_ok.any():
        raise ValueError("IRIS V2 loss requires at least one supported teacher mode")
    coverage = teacher_error[teacher_ok].mean()
    multimodal_teacher = td.shape[-1] > 1
    if multimodal_teacher and pred_ok.any():
        coverage = 0.5 * (coverage + pred_error[pred_ok].mean())

    gathered_sigma = torch.gather(output.depth_output.log_sigma, -1, teacher_to_pred.clamp_min(0))
    residual = torch.gather(output.refined_depth, -1, teacher_to_pred.clamp_min(0)) - td
    nll = 0.5 * torch.exp(-2.0 * gathered_sigma) * residual.square() + gathered_sigma
    uncertainty_nll = nll[teacher_ok].mean()

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
    dense_source_coverage = (
        logits.sum() * 0.0
        if source_foreground_masks is None
        else _dense_source_coverage_loss(
            output,
            domain,
            source_foreground_masks,
        )
    )
    total = (
        weights.depth_mode * mode_nll
        + weights.refined_depth * coverage
        + weights.uncertainty_nll * uncertainty_nll
        + weights.support * support
        + weights.tail_depth * tail
        + weights.world_regularizer * regularizer
        + weights.dense_source_coverage * dense_source_coverage
    )
    return {
        "total": total,
        "depth_mode": mode_nll,
        "refined_depth": coverage,
        "uncertainty_nll": uncertainty_nll,
        "support": support,
        "tail_depth": tail,
        "world_regularizer": regularizer,
        "dense_source_coverage": dense_source_coverage,
    }


def train_step_production_v2(apparatus, optimizer, batch: dict) -> dict[str, float]:
    """Scientific training entrypoint: foundation maps cannot be injected by caller."""
    if "foundation_maps" in batch:
        raise ValueError("production IRIS training forbids caller-supplied foundation_maps")
    if "source_foreground_masks" not in batch:
        raise ValueError("production IRIS training requires dense source foreground targets")
    if not hasattr(apparatus, "runtime_seal") or not hasattr(apparatus, "source_contract_hash"):
        raise TypeError("production IRIS training requires exact foundation-bound apparatus")
    resource = require_cuda_forward_live_set_v2(apparatus, batch["domain"])
    apparatus.train()
    optimizer.zero_grad(set_to_none=True)
    output = apparatus(batch["images"], batch["domain"])
    losses = iris_v2_loss(
        output,
        batch["domain"],
        batch["teacher_depth"],
        batch["teacher_support"],
        source_foreground_masks=batch["source_foreground_masks"],
    )
    losses["total"].backward()
    optimizer.step()
    result = {k: float(v.detach().cpu()) for k, v in losses.items()}
    result["resource_forward_live_lower_bound_gib"] = resource.forward_live_lower_bound_gib
    return result


def train_step_injected_foundation_source_test_v2(model, optimizer, batch: dict) -> dict[str, float]:
    """Synthetic/source-test lane only; never a scientific fit entrypoint."""
    if "foundation_maps" not in batch:
        raise ValueError("injected source-test lane requires explicit foundation_maps")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(batch["images"], batch["foundation_maps"], batch["domain"])
    losses = iris_v2_loss(
        output,
        batch["domain"],
        batch["teacher_depth"],
        batch["teacher_support"],
        source_foreground_masks=batch.get("source_foreground_masks"),
    )
    losses["total"].backward()
    optimizer.step()
    return {k: float(v.detach().cpu()) for k, v in losses.items()}
