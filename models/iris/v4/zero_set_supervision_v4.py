from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class ZeroSetSupervisionPolicyV4:
    """Training-only source-pixel supervision of the learned zero level set.

    This is not a second geometry authority and does not extract or repair a mesh.
    Foreground rays must approach f=0; foreground-interior rays must also contain at
    least one adjacent sampled sign transition. Boundary rays are exempt from the
    sign-transition requirement because a silhouette-tangent ray may touch f=0 without
    entering a numerically resolvable negative interval.
    """

    reference_decoder_resolution: int = 256
    minimum_feature_width_reference_voxels: float = 0.75
    boundary_multiplier: float = 4.0
    near_zero_weight: float = 1.0
    interior_crossing_weight: float = 1.0

    def validate(self) -> None:
        if self.reference_decoder_resolution < 2:
            raise ValueError("reference_decoder_resolution must be >=2")
        if self.minimum_feature_width_reference_voxels <= 0:
            raise ValueError("minimum feature width must be positive")
        if self.boundary_multiplier < 1.0:
            raise ValueError("boundary_multiplier must be >=1")
        if min(self.near_zero_weight, self.interior_crossing_weight) < 0:
            raise ValueError("zero-set weights must be non-negative")

    @property
    def minimum_feature_width_normalized(self) -> float:
        return (
            self.minimum_feature_width_reference_voxels
            * 2.0
            / float(self.reference_decoder_resolution - 1)
        )

    @property
    def crossing_scale_normalized(self) -> float:
        return 0.25 * self.minimum_feature_width_normalized


def _foreground_component_balanced_mean(
    values: torch.Tensor,
    component_id: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    positive_ids = torch.unique(component_id[component_id >= 0], sorted=True)
    if positive_ids.numel() == 0:
        return values.sum() * 0.0
    rows = []
    for group_id in positive_ids:
        mask = component_id == group_id
        w = weights[mask]
        rows.append((values[mask] * w).sum() / w.sum().clamp_min(1e-12))
    return torch.stack(rows).mean()


def source_foreground_zero_set_loss_v4(
    sdf_samples: torch.Tensor,
    target_foreground: torch.Tensor,
    component_id: torch.Tensor,
    boundary: torch.Tensor,
    *,
    policy: ZeroSetSupervisionPolicyV4 = ZeroSetSupervisionPolicyV4(),
) -> dict[str, torch.Tensor]:
    """Directly constrain source-foreground camera rays to contain learned f=0."""

    policy.validate()
    if sdf_samples.ndim != 3 or sdf_samples.shape[-1] < 2:
        raise ValueError("sdf_samples must be [B,R,D] with D>=2")
    expected = sdf_samples.shape[:2]
    if (
        target_foreground.shape != expected
        or component_id.shape != expected
        or boundary.shape != expected
    ):
        raise ValueError("zero-set target/component/boundary must be [B,R]")
    if not torch.isfinite(sdf_samples).all():
        raise ValueError("zero-set SDF samples must be finite")
    target = target_foreground.to(dtype=sdf_samples.dtype)
    cid = component_id.to(dtype=torch.long)
    bnd = boundary.to(dtype=sdf_samples.dtype).clamp(0.0, 1.0)
    if torch.any((target != 0) & (target != 1)):
        raise ValueError("target_foreground must be binary")
    foreground = target > 0.5
    if torch.any(foreground & (cid < 0)):
        raise ValueError("foreground zero-set rays require component ids")
    if not torch.any(foreground):
        zero = sdf_samples.sum() * 0.0
        return {
            "total": zero,
            "near_zero": zero,
            "interior_crossing": zero,
            "foreground_ray_count": torch.zeros((), dtype=torch.int64, device=sdf_samples.device),
            "interior_ray_count": torch.zeros((), dtype=torch.int64, device=sdf_samples.device),
            "boundary_ray_count": torch.zeros((), dtype=torch.int64, device=sdf_samples.device),
        }

    min_abs = torch.amin(torch.abs(sdf_samples), dim=-1)
    near_weights = 1.0 + (float(policy.boundary_multiplier) - 1.0) * bnd
    near_zero = _foreground_component_balanced_mean(
        min_abs[foreground],
        cid[foreground],
        near_weights[foreground],
    )

    interior = foreground & (bnd < 0.5)
    if torch.any(interior):
        pair_product = sdf_samples[..., :-1] * sdf_samples[..., 1:]
        best_pair_product = torch.amin(pair_product, dim=-1)
        crossing_violation = torch.relu(best_pair_product) / float(
            policy.crossing_scale_normalized
        )
        interior_crossing = _foreground_component_balanced_mean(
            crossing_violation[interior],
            cid[interior],
            torch.ones_like(crossing_violation[interior]),
        )
    else:
        interior_crossing = sdf_samples.sum() * 0.0

    total = (
        float(policy.near_zero_weight) * near_zero
        + float(policy.interior_crossing_weight) * interior_crossing
    )
    return {
        "total": total,
        "near_zero": near_zero,
        "interior_crossing": interior_crossing,
        "foreground_ray_count": foreground.sum().to(dtype=torch.int64),
        "interior_ray_count": interior.sum().to(dtype=torch.int64),
        "boundary_ray_count": (foreground & (bnd >= 0.5)).sum().to(dtype=torch.int64),
    }
