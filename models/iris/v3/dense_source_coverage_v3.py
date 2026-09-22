from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DenseSourceCoveragePolicyV3:
    """Subject-free V3 source-coverage surrogate constants.

    The physical scale is frozen from the VF-11 feature-survival calibration rather than
    tuned on Knight. The smallest required control is 0.75 of an R256 reference voxel.
    """

    reference_decoder_resolution: int = 256
    minimum_feature_width_reference_voxels: float = 0.75
    maximum_depth_step_fraction_of_min_feature: float = 0.5
    occupancy_beta_fraction_of_min_feature: float = 0.125
    boundary_multiplier: float = 4.0
    bce_weight: float = 1.0
    soft_dice_weight: float = 1.0

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
        if min(self.bce_weight, self.soft_dice_weight) < 0.0:
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


def phase_jittered_depth_lattice(
    depth_min: float,
    depth_max: float,
    *,
    policy: DenseSourceCoveragePolicyV3 = DenseSourceCoveragePolicyV3(),
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return an equally-spaced lattice with one random global phase.

    Independent per-bin jitter is deliberately not used: it can double the largest
    inter-sample gap. A global phase shift preserves the preregistered maximum spacing
    while preventing one fixed phase from systematically missing thin geometry.
    """

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


def ray_foreground_probability_from_sdf_samples(
    sdf_samples: torch.Tensor,
    *,
    policy: DenseSourceCoveragePolicyV3 = DenseSourceCoveragePolicyV3(),
) -> torch.Tensor:
    """Map V3 signed samples on each ray to foreground occupancy probability.

    Uses the exact hard minimum rather than log-sum-exp soft-min, avoiding the
    multiplicity-dependent tau*log(N) silhouette bias. torch.amin is differentiable
    almost everywhere and distributes gradients across exact ties.
    """

    policy.validate()
    if sdf_samples.ndim < 1 or sdf_samples.shape[-1] < 1:
        raise ValueError("sdf_samples must have a non-empty depth axis")
    if not torch.isfinite(sdf_samples).all():
        raise ValueError("sdf_samples must be finite")
    ray_min = torch.amin(sdf_samples, dim=-1)
    return torch.sigmoid(-ray_min / policy.occupancy_beta_normalized)


def component_balanced_source_coverage_loss(
    probability: torch.Tensor,
    target_foreground: torch.Tensor,
    component_id: torch.Tensor,
    boundary: torch.Tensor,
    *,
    policy: DenseSourceCoveragePolicyV3 = DenseSourceCoveragePolicyV3(),
) -> dict[str, torch.Tensor]:
    """BCE + silhouette Dice with equal total BCE authority per source component.

    component_id is -1 for background and >=0 for foreground connected components.
    Every represented foreground component contributes one group mean, regardless of
    area; background contributes one additional group. Boundary pixels are emphasized
    inside each group and then renormalized, so tiny components cannot acquire an
    unbounded raw inverse-area weight.
    """

    policy.validate()
    if (
        probability.shape != target_foreground.shape
        or probability.shape != component_id.shape
        or probability.shape != boundary.shape
    ):
        raise ValueError("probability/target/component_id/boundary shapes must match")
    if probability.numel() == 0:
        raise ValueError("coverage loss requires at least one sampled ray")
    if not torch.isfinite(probability).all():
        raise ValueError("probability must be finite")
    p = probability.clamp(1e-6, 1.0 - 1e-6)
    target = target_foreground.to(dtype=p.dtype)
    if torch.any((target != 0) & (target != 1)):
        raise ValueError("target_foreground must be binary")
    cid = component_id.to(dtype=torch.long)
    bnd = boundary.to(dtype=p.dtype).clamp(0.0, 1.0)
    if torch.any((target > 0.5) & (cid < 0)):
        raise ValueError("foreground samples require non-negative component_id")
    if torch.any((target < 0.5) & (cid >= 0)):
        raise ValueError("background samples require component_id=-1")

    per_ray_bce = F.binary_cross_entropy(p, target, reduction="none")
    ray_weight = 1.0 + (policy.boundary_multiplier - 1.0) * bnd

    group_losses: list[torch.Tensor] = []
    background_mask = cid < 0
    if torch.any(background_mask):
        w = ray_weight[background_mask]
        group_losses.append(
            (per_ray_bce[background_mask] * w).sum() / w.sum().clamp_min(1e-12)
        )

    positive_ids = torch.unique(cid[cid >= 0], sorted=True)
    for group_id in positive_ids:
        mask = cid == group_id
        w = ray_weight[mask]
        group_losses.append((per_ray_bce[mask] * w).sum() / w.sum().clamp_min(1e-12))

    if not group_losses:
        raise ValueError("no valid source-coverage groups")
    balanced_bce = torch.stack(group_losses).mean()

    intersection = torch.sum(p * target)
    denominator = torch.sum(p) + torch.sum(target)
    soft_dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    total = policy.bce_weight * balanced_bce + policy.soft_dice_weight * soft_dice
    return {
        "total": total,
        "component_balanced_bce": balanced_bce,
        "soft_dice": soft_dice,
        "group_count": torch.as_tensor(len(group_losses), device=p.device, dtype=torch.int64),
    }
