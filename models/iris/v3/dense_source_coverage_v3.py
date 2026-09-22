from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DenseSourceCoveragePolicyV3:
    """Subject-free V3 source-coverage surrogate constants."""

    reference_decoder_resolution: int = 256
    minimum_feature_width_reference_voxels: float = 0.75
    maximum_depth_step_fraction_of_min_feature: float = 0.5
    occupancy_beta_fraction_of_min_feature: float = 0.125
    boundary_multiplier: float = 4.0
    bce_weight: float = 1.0
    soft_dice_weight: float = 1.0
    top_level_loss_weight: float = 1.0

    def validate(self) -> None:
        if self.reference_decoder_resolution < 2:
            raise ValueError("reference decoder resolution must be >=2")
        if self.minimum_feature_width_reference_voxels <= 0.0:
            raise ValueError("minimum feature width must be positive")
        if not (0.0 < self.maximum_depth_step_fraction_of_min_feature <= 1.0):
            raise ValueError("maximum depth step fraction must be in (0,1]")
        if self.occupancy_beta_fraction_of_min_feature <= 0.0:
            raise ValueError("occupancy beta fraction must be positive")
        if self.boundary_multiplier < 1.0:
            raise ValueError("boundary multiplier must be >=1")
        if min(self.bce_weight, self.soft_dice_weight, self.top_level_loss_weight) < 0.0:
            raise ValueError("loss weights must be non-negative")

    @property
    def reference_voxel_normalized(self) -> float:
        return 2.0 / float(self.reference_decoder_resolution - 1)

    @property
    def minimum_feature_width_normalized(self) -> float:
        return self.minimum_feature_width_reference_voxels * self.reference_voxel_normalized

    @property
    def maximum_depth_step_normalized(self) -> float:
        return self.maximum_depth_step_fraction_of_min_feature * self.minimum_feature_width_normalized

    @property
    def occupancy_beta_normalized(self) -> float:
        return self.occupancy_beta_fraction_of_min_feature * self.minimum_feature_width_normalized


def historical_sparse_objective_v3(
    local_sdf: torch.Tensor,
    local_target: torch.Tensor,
    background_sdf: torch.Tensor,
    certified_outside_sdf: torch.Tensor,
    teacher_surface_sdf: torch.Tensor,
    *,
    positive_margin: float = 0.04,
) -> dict[str, torch.Tensor]:
    """Exact 2026-09-19 STRIDE8 sparse objective, preserved for causal Arm A."""

    margin = float(positive_margin)
    if not math.isfinite(margin) or margin <= 0.0:
        raise ValueError("positive_margin must be finite and positive")
    if local_sdf.shape != local_target.shape:
        raise ValueError("local_sdf/local_target shape mismatch")
    tensors = (
        local_sdf,
        local_target,
        background_sdf,
        certified_outside_sdf,
        teacher_surface_sdf,
    )
    if any(value.numel() == 0 for value in tensors):
        raise ValueError("historical sparse objective requires non-empty terms")
    if any(not torch.isfinite(value).all() for value in tensors):
        raise ValueError("historical sparse objective requires finite tensors")

    local = F.smooth_l1_loss(local_sdf.float(), local_target.float(), beta=0.02)
    background = F.relu(margin - background_sdf.float()).mean()
    certified_outside = F.relu(margin - certified_outside_sdf.float()).mean()
    teacher_surface_zero = F.smooth_l1_loss(
        teacher_surface_sdf.float(),
        torch.zeros_like(teacher_surface_sdf.float()),
        beta=0.01,
    )
    total = local + 0.25 * background + 0.25 * certified_outside + 0.50 * teacher_surface_zero
    return {
        "total": total,
        "local": local,
        "background": background,
        "certified_outside": certified_outside,
        "teacher_surface_zero": teacher_surface_zero,
    }


def deterministic_phase_fraction(
    fit_seed: int,
    training_step: int,
    opposite_view_pair: tuple[int, int],
) -> float:
    """Stable [0,1) phase derived only from frozen causal identifiers."""

    a, b = map(int, opposite_view_pair)
    if a < 0 or b < 0 or a == b:
        raise ValueError("invalid opposite-view pair")
    payload = f"{int(fit_seed)}:{int(training_step)}:{a}:{b}".encode("ascii")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)
    return float(value) / float(1 << 64)


def phase_jittered_depth_lattice(
    depth_min: float,
    depth_max: float,
    *,
    policy: DenseSourceCoveragePolicyV3 = DenseSourceCoveragePolicyV3(),
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return an equally-spaced lattice with one random global phase."""

    policy.validate()
    lo = float(depth_min)
    hi = float(depth_max)
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        raise ValueError("depth range must be finite and increasing")
    span = hi - lo
    interval_count = max(1, int(math.ceil(span / policy.maximum_depth_step_normalized)))
    spacing = span / float(interval_count)
    phase = torch.rand((), generator=generator, device=device, dtype=dtype) * spacing
    index = torch.arange(interval_count, device=device, dtype=dtype)
    return torch.as_tensor(lo, device=device, dtype=dtype) + phase + index * spacing


def ray_box_intersections_unit_cube(
    ray_origins_normalized: torch.Tensor,
    ray_directions: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact forward-ray slab intersection against canonical [-1,1]^3."""

    if (
        ray_origins_normalized.ndim != 2
        or ray_origins_normalized.shape[-1] != 3
        or ray_directions.shape != ray_origins_normalized.shape
    ):
        raise ValueError("ray origins/directions must both be [R,3]")
    if not torch.isfinite(ray_origins_normalized).all() or not torch.isfinite(ray_directions).all():
        raise ValueError("ray origins/directions must be finite")
    norm = torch.linalg.vector_norm(ray_directions, dim=-1)
    if torch.any(norm <= float(epsilon)):
        raise ValueError("ray direction must be nonzero")

    o = ray_origins_normalized
    d = ray_directions / norm[:, None]
    moving = torch.abs(d) > float(epsilon)
    parallel_outside = (~moving) & ((o < -1.0) | (o > 1.0))
    safe_d = torch.where(moving, d, torch.ones_like(d))
    t1 = (-1.0 - o) / safe_d
    t2 = (1.0 - o) / safe_d
    ninf = torch.full_like(t1, float("-inf"))
    pinf = torch.full_like(t1, float("inf"))
    near = torch.where(moving, torch.minimum(t1, t2), ninf)
    far = torch.where(moving, torch.maximum(t1, t2), pinf)
    entry = torch.max(near, dim=-1).values
    exit = torch.min(far, dim=-1).values
    entry = torch.maximum(entry, torch.zeros_like(entry))
    valid = (~parallel_outside.any(dim=-1)) & torch.isfinite(entry) & torch.isfinite(exit) & (exit >= entry)
    return entry, exit, valid


def phase_jittered_unit_cube_ray_points(
    ray_origins_normalized: torch.Tensor,
    ray_directions: torch.Tensor,
    *,
    phase_fraction: float,
    policy: DenseSourceCoveragePolicyV3 = DenseSourceCoveragePolicyV3(),
) -> torch.Tensor:
    """Sample admitted forward ray segments with bounded spacing and explicit endpoints."""

    policy.validate()
    phase = float(phase_fraction)
    if not math.isfinite(phase) or not (0.0 <= phase < 1.0):
        raise ValueError("phase_fraction must be in [0,1)")
    entry, exit, valid = ray_box_intersections_unit_cube(
        ray_origins_normalized,
        ray_directions,
    )
    if not bool(torch.all(valid)):
        raise ValueError("all dense-loss rays must intersect normalization cube")
    direction = ray_directions / torch.linalg.vector_norm(ray_directions, dim=-1, keepdim=True)
    span = exit - entry
    maximum_span = float(torch.max(span).detach().cpu())
    interval_count = max(
        1,
        int(math.ceil(maximum_span / policy.maximum_depth_step_normalized)),
    )
    spacing = span / float(interval_count)
    k = torch.arange(
        interval_count,
        device=ray_origins_normalized.device,
        dtype=ray_origins_normalized.dtype,
    )
    interior_t = entry[:, None] + (phase + k[None, :]) * spacing[:, None]
    t = torch.cat([entry[:, None], interior_t, exit[:, None]], dim=1)
    points = ray_origins_normalized[:, None, :] + t[:, :, None] * direction[:, None, :]
    if not torch.isfinite(points).all():
        raise ValueError("dense-loss ray points became nonfinite")
    gaps = t[:, 1:] - t[:, :-1]
    if float(torch.max(gaps).detach().cpu()) > policy.maximum_depth_step_normalized + 1e-6:
        raise RuntimeError("DENSE_RAY_SPACING_CONTRACT_VIOLATED")
    return points


def ray_foreground_logit_from_sdf_samples(
    sdf_samples: torch.Tensor,
    *,
    policy: DenseSourceCoveragePolicyV3 = DenseSourceCoveragePolicyV3(),
) -> torch.Tensor:
    """Stable foreground logit from the exact hard signed minimum.

    Training must use this logit with BCEWithLogits. Converting to probability and then
    clamping can erase the gradient on severely wrong rays, exactly where correction is
    most needed.
    """

    policy.validate()
    if sdf_samples.ndim < 1 or sdf_samples.shape[-1] < 1:
        raise ValueError("sdf_samples must have a non-empty depth axis")
    if not torch.isfinite(sdf_samples).all():
        raise ValueError("sdf_samples must be finite")
    ray_min = torch.amin(sdf_samples, dim=-1)
    return -ray_min / policy.occupancy_beta_normalized


def ray_foreground_probability_from_sdf_samples(
    sdf_samples: torch.Tensor,
    *,
    policy: DenseSourceCoveragePolicyV3 = DenseSourceCoveragePolicyV3(),
) -> torch.Tensor:
    """Probability view for diagnostics/soft Dice; not the BCE training interface."""

    return torch.sigmoid(ray_foreground_logit_from_sdf_samples(sdf_samples, policy=policy))


def _query_ray_sdf_v3(
    model,
    scene_planes: torch.Tensor,
    ray_points_normalized: torch.Tensor,
    *,
    query_chunk: int,
) -> torch.Tensor:
    if ray_points_normalized.ndim != 4 or ray_points_normalized.shape[-1] != 3:
        raise ValueError("ray_points_normalized must be [B,R,D,3]")
    if ray_points_normalized.shape[0] != scene_planes.shape[0]:
        raise ValueError("ray points/scene plane batch mismatch")
    if int(query_chunk) <= 0:
        raise ValueError("query_chunk must be positive")
    b, r, d, _ = ray_points_normalized.shape
    flat = ray_points_normalized.reshape(b, r * d, 3)
    sdf_parts = []
    for start in range(0, flat.shape[1], int(query_chunk)):
        sdf_parts.append(
            model.query(scene_planes, flat[:, start : start + int(query_chunk)])["sdf"]
        )
    return torch.cat(sdf_parts, dim=1).reshape(b, r, d)


def query_ray_foreground_logit_v3(
    model,
    scene_planes: torch.Tensor,
    ray_points_normalized: torch.Tensor,
    *,
    policy: DenseSourceCoveragePolicyV3 = DenseSourceCoveragePolicyV3(),
    query_chunk: int = 131072,
) -> torch.Tensor:
    """Backpropagating stable occupancy logit through the actual V3 signed field."""

    sdf = _query_ray_sdf_v3(
        model,
        scene_planes,
        ray_points_normalized,
        query_chunk=query_chunk,
    )
    return ray_foreground_logit_from_sdf_samples(sdf, policy=policy)


def query_ray_foreground_probability_v3(
    model,
    scene_planes: torch.Tensor,
    ray_points_normalized: torch.Tensor,
    *,
    policy: DenseSourceCoveragePolicyV3 = DenseSourceCoveragePolicyV3(),
    query_chunk: int = 131072,
) -> torch.Tensor:
    """Probability diagnostic wrapper around the stable V3 logit query."""

    return torch.sigmoid(
        query_ray_foreground_logit_v3(
            model,
            scene_planes,
            ray_points_normalized,
            policy=policy,
            query_chunk=query_chunk,
        )
    )


def _validate_group_inputs(
    value: torch.Tensor,
    target_foreground: torch.Tensor,
    component_id: torch.Tensor,
    boundary: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        value.shape != target_foreground.shape
        or value.shape != component_id.shape
        or value.shape != boundary.shape
    ):
        raise ValueError("value/target/component_id/boundary shapes must match")
    if value.numel() == 0:
        raise ValueError("coverage loss requires at least one sampled ray")
    if not torch.isfinite(value).all():
        raise ValueError("coverage values must be finite")
    target = target_foreground.to(dtype=value.dtype)
    if torch.any((target != 0) & (target != 1)):
        raise ValueError("target_foreground must be binary")
    cid = component_id.to(dtype=torch.long)
    bnd = boundary.to(dtype=value.dtype).clamp(0.0, 1.0)
    if torch.any((target > 0.5) & (cid < 0)):
        raise ValueError("foreground samples require non-negative component_id")
    if torch.any((target < 0.5) & (cid >= 0)):
        raise ValueError("background samples require component_id=-1")
    return target, cid, bnd


def _balanced_group_mean(
    per_ray_loss: torch.Tensor,
    cid: torch.Tensor,
    ray_weight: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    group_losses: list[torch.Tensor] = []
    background_mask = cid < 0
    if torch.any(background_mask):
        w = ray_weight[background_mask]
        group_losses.append(
            (per_ray_loss[background_mask] * w).sum() / w.sum().clamp_min(1e-12)
        )
    positive_ids = torch.unique(cid[cid >= 0], sorted=True)
    for group_id in positive_ids:
        mask = cid == group_id
        w = ray_weight[mask]
        group_losses.append((per_ray_loss[mask] * w).sum() / w.sum().clamp_min(1e-12))
    if not group_losses:
        raise ValueError("no valid source-coverage groups")
    return torch.stack(group_losses).mean(), len(group_losses)


def component_balanced_source_coverage_loss_from_logits(
    foreground_logit: torch.Tensor,
    target_foreground: torch.Tensor,
    component_id: torch.Tensor,
    boundary: torch.Tensor,
    *,
    policy: DenseSourceCoveragePolicyV3 = DenseSourceCoveragePolicyV3(),
) -> dict[str, torch.Tensor]:
    """Numerically stable training loss with equal BCE authority per source component."""

    policy.validate()
    target, cid, bnd = _validate_group_inputs(
        foreground_logit,
        target_foreground,
        component_id,
        boundary,
    )
    per_ray_bce = F.binary_cross_entropy_with_logits(
        foreground_logit.float(),
        target.float(),
        reduction="none",
    )
    ray_weight = 1.0 + (policy.boundary_multiplier - 1.0) * bnd.float()
    balanced_bce, group_count = _balanced_group_mean(per_ray_bce, cid, ray_weight)

    probability = torch.sigmoid(foreground_logit.float())
    target_float = target.float()
    intersection = torch.sum(probability * target_float)
    denominator = torch.sum(probability) + torch.sum(target_float)
    soft_dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    total = policy.bce_weight * balanced_bce + policy.soft_dice_weight * soft_dice
    return {
        "total": total,
        "component_balanced_bce": balanced_bce,
        "soft_dice": soft_dice,
        "group_count": torch.as_tensor(group_count, device=foreground_logit.device, dtype=torch.int64),
    }


def component_balanced_source_coverage_loss(
    probability: torch.Tensor,
    target_foreground: torch.Tensor,
    component_id: torch.Tensor,
    boundary: torch.Tensor,
    *,
    policy: DenseSourceCoveragePolicyV3 = DenseSourceCoveragePolicyV3(),
) -> dict[str, torch.Tensor]:
    """Probability-domain compatibility helper for non-saturated diagnostics/tests.

    Shipping/training code must use component_balanced_source_coverage_loss_from_logits.
    """

    policy.validate()
    target, cid, bnd = _validate_group_inputs(
        probability,
        target_foreground,
        component_id,
        boundary,
    )
    if torch.any((probability <= 0.0) | (probability >= 1.0)):
        raise ValueError("probability-domain helper requires strict probabilities in (0,1)")
    per_ray_bce = F.binary_cross_entropy(probability, target, reduction="none")
    ray_weight = 1.0 + (policy.boundary_multiplier - 1.0) * bnd
    balanced_bce, group_count = _balanced_group_mean(per_ray_bce, cid, ray_weight)
    intersection = torch.sum(probability * target)
    denominator = torch.sum(probability) + torch.sum(target)
    soft_dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    total = policy.bce_weight * balanced_bce + policy.soft_dice_weight * soft_dice
    return {
        "total": total,
        "component_balanced_bce": balanced_bce,
        "soft_dice": soft_dice,
        "group_count": torch.as_tensor(group_count, device=probability.device, dtype=torch.int64),
    }
