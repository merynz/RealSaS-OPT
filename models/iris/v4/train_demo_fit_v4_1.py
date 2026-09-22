from __future__ import annotations

import torch

from models.iris.v3.dense_source_coverage_v3 import DenseSourceCoveragePolicyV3
from models.iris.v4.regularity_v4_1 import (
    NarrowBandEikonalPolicyV41,
    finite_difference_narrow_band_eikonal_v41,
)
from models.iris.v4.train_demo_fit_v4 import (
    SourceConstraintRayBatchV4,
    compute_v4_demo_geometry_objective,
)


def compute_v41_demo_geometry_objective(
    *,
    model,
    scene_planes: torch.Tensor,
    local_points_normalized: torch.Tensor,
    local_target_sdf: torch.Tensor,
    teacher_surface_zero_points_normalized: torch.Tensor,
    source_exterior_points_normalized: torch.Tensor,
    source_exterior_margin_normalized: torch.Tensor,
    source_view_batches: tuple[SourceConstraintRayBatchV4, ...],
    coverage_policy: DenseSourceCoveragePolicyV3 = DenseSourceCoveragePolicyV3(),
    eikonal_policy: NarrowBandEikonalPolicyV41 = NarrowBandEikonalPolicyV41(),
    teacher_surface_zero_weight: float = 0.50,
    foreground_existence_weight: float = 1.0,
    negative_space_weight: float = 1.0,
    source_exterior_weight: float = 1.0,
    maximum_background_margin_normalized: float = 0.04,
    query_chunk: int = 131072,
) -> dict[str, object]:
    """V4 source objective + narrow-band field regularity; architecture is unchanged."""

    base = compute_v4_demo_geometry_objective(
        model=model,
        scene_planes=scene_planes,
        local_points_normalized=local_points_normalized,
        local_target_sdf=local_target_sdf,
        teacher_surface_zero_points_normalized=teacher_surface_zero_points_normalized,
        source_exterior_points_normalized=source_exterior_points_normalized,
        source_exterior_margin_normalized=source_exterior_margin_normalized,
        source_view_batches=source_view_batches,
        coverage_policy=coverage_policy,
        teacher_surface_zero_weight=teacher_surface_zero_weight,
        foreground_existence_weight=foreground_existence_weight,
        negative_space_weight=negative_space_weight,
        source_exterior_weight=source_exterior_weight,
        maximum_background_margin_normalized=maximum_background_margin_normalized,
        query_chunk=query_chunk,
    )
    eikonal = finite_difference_narrow_band_eikonal_v41(
        model=model,
        scene_planes=scene_planes,
        local_points_normalized=local_points_normalized,
        local_target_sdf=local_target_sdf,
        policy=eikonal_policy,
    )
    total = base["total"] + float(eikonal_policy.top_level_weight) * eikonal["total"]
    out = dict(base)
    out["base_total_before_eikonal"] = base["total"]
    out["eikonal_total"] = eikonal["total"]
    out["eikonal_weight"] = float(eikonal_policy.top_level_weight)
    out["eikonal_sample_count"] = eikonal["sample_count"]
    out["eikonal_gradient_norm_mean"] = eikonal["gradient_norm_mean"]
    out["eikonal_gradient_norm_p95"] = eikonal["gradient_norm_p95"]
    out["eikonal_gradient_norm_abs_error_mean"] = eikonal["gradient_norm_abs_error_mean"]
    out["total"] = total
    return out
