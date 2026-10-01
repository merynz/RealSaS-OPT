from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class NarrowBandEikonalPolicyV41:
    band_radius_normalized: float = 0.02
    finite_difference_epsilon_normalized: float = 2.0 / 511.0
    maximum_points_per_step: int = 512
    top_level_weight: float = 0.02

    def validate(self) -> None:
        if not math.isfinite(self.band_radius_normalized) or self.band_radius_normalized <= 0:
            raise ValueError("band_radius_normalized must be finite and positive")
        if (
            not math.isfinite(self.finite_difference_epsilon_normalized)
            or self.finite_difference_epsilon_normalized <= 0
        ):
            raise ValueError("finite_difference_epsilon_normalized must be finite and positive")
        if int(self.maximum_points_per_step) <= 0:
            raise ValueError("maximum_points_per_step must be positive")
        if not math.isfinite(self.top_level_weight) or self.top_level_weight <= 0:
            raise ValueError("top_level_weight must be finite and positive")


def finite_difference_narrow_band_eikonal_v41(
    *,
    model,
    scene_planes: torch.Tensor,
    local_points_normalized: torch.Tensor,
    local_target_sdf: torch.Tensor,
    policy: NarrowBandEikonalPolicyV41 = NarrowBandEikonalPolicyV41(),
) -> dict[str, torch.Tensor]:
    """First-order finite-difference Eikonal regularity in the existing FIT local band.

    We intentionally avoid autograd-of-grid_sample second derivatives. The selected
    points are already part of the FIT-only local metric-SDF supervision; this term does
    not mint new geometry authority. It only penalizes oscillatory signed fields between
    observed constraints by encouraging |grad f| ~= 1 near the current source-supported
    surface neighborhood. Finite-difference field queries run explicitly in FP32 so the
    small epsilon is not quantized by the surrounding BF16 training autocast context.
    """

    policy.validate()
    if local_points_normalized.ndim != 3 or local_points_normalized.shape[-1] != 3:
        raise ValueError("local_points_normalized must be [B,N,3]")
    if local_target_sdf.shape != local_points_normalized.shape[:2]:
        raise ValueError("local_target_sdf must be [B,N]")

    mask = torch.abs(local_target_sdf.detach().float()) <= float(policy.band_radius_normalized)
    flat_points = local_points_normalized[mask]
    if flat_points.numel() == 0:
        zero = scene_planes.sum() * 0.0
        return {
            "total": zero,
            "sample_count": torch.zeros((), dtype=torch.int64, device=scene_planes.device),
            "gradient_norm_mean": torch.full((), float("nan"), device=scene_planes.device),
            "gradient_norm_p95": torch.full((), float("nan"), device=scene_planes.device),
            "gradient_norm_abs_error_mean": torch.full((), float("nan"), device=scene_planes.device),
        }

    limit = min(int(policy.maximum_points_per_step), int(flat_points.shape[0]))
    points = flat_points[:limit]
    eps = float(policy.finite_difference_epsilon_normalized)
    points = torch.clamp(points.float(), min=-1.0 + eps, max=1.0 - eps)
    planes = scene_planes.float()
    points = points.to(device=planes.device, dtype=torch.float32)

    derivatives: list[torch.Tensor] = []
    with torch.autocast(device_type=planes.device.type, enabled=False):
        for axis in range(3):
            delta = torch.zeros_like(points)
            delta[:, axis] = eps
            plus = model.query(planes, (points + delta).unsqueeze(0))["sdf"].squeeze(0).float()
            minus = model.query(planes, (points - delta).unsqueeze(0))["sdf"].squeeze(0).float()
            derivatives.append((plus - minus) / (2.0 * eps))
    gradient = torch.stack(derivatives, dim=-1)
    norm = torch.linalg.vector_norm(gradient, dim=-1)
    error = torch.abs(norm - 1.0)
    total = torch.mean((norm - 1.0) ** 2)
    p95_index = max(0, int(math.ceil(0.95 * int(norm.numel()))) - 1)
    p95 = torch.sort(norm).values[p95_index]
    return {
        "total": total,
        "sample_count": torch.as_tensor(limit, dtype=torch.int64, device=planes.device),
        "gradient_norm_mean": norm.mean(),
        "gradient_norm_p95": p95,
        "gradient_norm_abs_error_mean": error.mean(),
    }


@dataclass(frozen=True, order=True)
class SourceContractCheckpointScoreV41:
    """Lexicographic score; every field is lower-is-better.

    Hard sign violations come before soft margin/regularity and before training loss, so
    checkpoint selection cannot hide a source-contract regression behind a lower scalar
    objective.
    """

    max_background_zero_crossing_fraction: float
    max_foreground_miss_fraction: float
    source_exterior_negative_fraction: float
    max_background_margin_violation_fraction: float
    eikonal_abs_error_mean: float
    scalar_loss: float

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"invalid V4.1 checkpoint score field {name}")


def make_checkpoint_score_v41(
    *,
    max_background_zero_crossing_fraction: float,
    max_foreground_miss_fraction: float,
    source_exterior_negative_fraction: float,
    max_background_margin_violation_fraction: float,
    eikonal_abs_error_mean: float,
    scalar_loss: float,
) -> SourceContractCheckpointScoreV41:
    score = SourceContractCheckpointScoreV41(
        max_background_zero_crossing_fraction=float(max_background_zero_crossing_fraction),
        max_foreground_miss_fraction=float(max_foreground_miss_fraction),
        source_exterior_negative_fraction=float(source_exterior_negative_fraction),
        max_background_margin_violation_fraction=float(max_background_margin_violation_fraction),
        eikonal_abs_error_mean=float(eikonal_abs_error_mean),
        scalar_loss=float(scalar_loss),
    )
    score.validate()
    return score
