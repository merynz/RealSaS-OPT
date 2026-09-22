from __future__ import annotations

import torch
import torch.nn.functional as F

from models.iris.v3.dense_source_coverage_v3 import DenseSourceCoveragePolicyV3
from models.iris.v3.negative_space_barrier_v3 import query_dense_ray_sdf_v3
from models.iris.v4.regularity_v4_2 import (
    GlobalEikonalPolicyV42,
    RayLipschitzPolicyV42,
    finite_difference_global_eikonal_v42,
    ray_directional_lipschitz_v42,
    sign_transition_diagnostic_v42,
)
from models.iris.v4.source_exterior_v4 import source_exterior_metric_barrier_v4
from models.iris.v4.train_demo_fit_v4 import (
    SourceConstraintRayBatchV4,
    component_balanced_foreground_existence_hinge_v4,
    source_negative_space_barrier_v4,
)


def compute_v42_tp64_geometry_objective(
    *,
    model,
    scene_planes: torch.Tensor,
    local_points_normalized: torch.Tensor,
    local_target_sdf: torch.Tensor,
    teacher_surface_zero_points_normalized: torch.Tensor,
    source_exterior_points_normalized: torch.Tensor,
    source_exterior_margin_normalized: torch.Tensor,
    source_view_batches: tuple[SourceConstraintRayBatchV4, ...],
    fit_seed: int,
    training_step: int,
    coverage_policy: DenseSourceCoveragePolicyV3 = DenseSourceCoveragePolicyV3(),
    global_eikonal_policy: GlobalEikonalPolicyV42 = GlobalEikonalPolicyV42(),
    ray_lipschitz_policy: RayLipschitzPolicyV42 = RayLipschitzPolicyV42(),
    teacher_surface_zero_weight: float = 0.50,
    foreground_existence_weight: float = 1.0,
    negative_space_weight: float = 1.0,
    source_exterior_weight: float = 1.0,
    maximum_background_margin_normalized: float = 0.04,
    query_chunk: int = 131072,
) -> dict[str, object]:
    """V4.2 TP64 source objective with volume regularity and ray Lipschitz control.

    Geometry/source authority is unchanged. The only new training signals are regularity
    assumptions of a signed-distance-like field. No teacher depth, global teacher SDF,
    visual-hull clipping, mesh repair, or raster-derived geometry target is introduced.
    """

    coverage_policy.validate()
    global_eikonal_policy.validate()
    ray_lipschitz_policy.validate()
    if int(training_step) <= 0:
        raise ValueError("training_step must be positive")
    if int(query_chunk) <= 0:
        raise ValueError("query_chunk must be positive")
    if len(source_view_batches) != 8:
        raise ValueError("V4.2 TP64 requires all eight source views every step")
    view_indices = [int(batch.view_index) for batch in source_view_batches]
    if set(view_indices) != set(range(8)) or len(set(view_indices)) != 8:
        raise ValueError("source-view batches must be unique V0..V7")

    local_sdf = model.query(scene_planes, local_points_normalized)["sdf"]
    teacher_sdf = model.query(scene_planes, teacher_surface_zero_points_normalized)["sdf"]
    exterior_sdf = model.query(scene_planes, source_exterior_points_normalized)["sdf"]
    if local_sdf.shape != local_target_sdf.shape:
        raise ValueError("local target shape mismatch")

    local = F.smooth_l1_loss(local_sdf.float(), local_target_sdf.float(), beta=0.02)
    teacher_zero = F.smooth_l1_loss(
        teacher_sdf.float(),
        torch.zeros_like(teacher_sdf.float()),
        beta=0.01,
    )
    exterior = source_exterior_metric_barrier_v4(
        exterior_sdf.float(),
        source_exterior_margin_normalized.float(),
    )
    structural = local + float(teacher_surface_zero_weight) * teacher_zero

    beta = float(coverage_policy.occupancy_beta_normalized)
    boundary_inside_margin = 0.5 * beta
    interior_inside_margin = beta
    foreground_terms: list[torch.Tensor] = []
    negative_terms: list[torch.Tensor] = []
    lipschitz_terms: list[torch.Tensor] = []
    rows: list[dict[str, object]] = []

    for batch in sorted(source_view_batches, key=lambda row: int(row.view_index)):
        batch.validate()
        ray_sdf = query_dense_ray_sdf_v3(
            model,
            scene_planes,
            batch.ray_points_normalized,
            query_chunk=query_chunk,
        )
        foreground = component_balanced_foreground_existence_hinge_v4(
            ray_sdf,
            batch.target_foreground,
            batch.component_id,
            batch.boundary,
            boundary_inside_margin_normalized=boundary_inside_margin,
            interior_inside_margin_normalized=interior_inside_margin,
        )
        negative = source_negative_space_barrier_v4(
            ray_sdf,
            batch.target_foreground,
            batch.distance_to_foreground_px,
            camera_half_extent_normalized=batch.camera_half_extent_normalized,
            raster_width=batch.raster_width,
            maximum_margin_normalized=maximum_background_margin_normalized,
        )
        lipschitz = ray_directional_lipschitz_v42(
            ray_sdf,
            batch.ray_points_normalized,
            policy=ray_lipschitz_policy,
        )
        transitions = sign_transition_diagnostic_v42(ray_sdf)
        foreground_terms.append(foreground["total"])
        negative_terms.append(negative["total"])
        lipschitz_terms.append(lipschitz["total"])
        rows.append({
            "view_index": int(batch.view_index),
            "foreground_existence_total": foreground["total"],
            "foreground_component_group_count": foreground["group_count"],
            "foreground_satisfied_fraction": foreground["satisfied_fraction"],
            "foreground_zero_crossing_fraction": foreground["zero_crossing_fraction"],
            "foreground_minimum_ray_sdf": foreground["minimum_ray_sdf"],
            "foreground_maximum_ray_sdf": foreground["maximum_ray_sdf"],
            "negative_space_total": negative["total"],
            "negative_space_violating_point_fraction": negative["violating_point_fraction"],
            "negative_space_zero_crossing_fraction": negative["zero_crossing_fraction"],
            "minimum_background_sdf": negative["minimum_background_sdf"],
            "minimum_margin": negative["minimum_margin"],
            "maximum_margin": negative["maximum_margin"],
            "ray_lipschitz_total": lipschitz["total"],
            "ray_lipschitz_violating_fraction": lipschitz["violating_fraction"],
            "ray_lipschitz_maximum_excess": lipschitz["maximum_excess"],
            "ray_directional_slope_mean": lipschitz["directional_slope_mean"],
            "ray_directional_slope_p95": lipschitz["directional_slope_p95"],
            "sign_transition_mean_diagnostic": transitions["transition_mean"],
            "sign_transition_p95_diagnostic": transitions["transition_p95"],
            "sign_transition_max_diagnostic": transitions["transition_max"],
            "sign_transition_gt2_fraction_diagnostic": transitions["rays_gt2_fraction"],
            "sign_transition_gt4_fraction_diagnostic": transitions["rays_gt4_fraction"],
            "hard_replay_ray_count": int(batch.hard_replay.sum().detach().cpu()),
            "ray_count": int(batch.target_foreground.numel()),
        })

    foreground_total = torch.stack(foreground_terms).mean()
    negative_total = torch.stack(negative_terms).mean()
    ray_lipschitz_total = torch.stack(lipschitz_terms).mean()
    global_eikonal = finite_difference_global_eikonal_v42(
        model=model,
        scene_planes=scene_planes,
        fit_seed=int(fit_seed),
        training_step=int(training_step),
        policy=global_eikonal_policy,
    )

    base_total = (
        structural
        + float(foreground_existence_weight) * foreground_total
        + float(negative_space_weight) * negative_total
        + float(source_exterior_weight) * exterior["total"]
    )
    total = (
        base_total
        + float(global_eikonal_policy.top_level_weight) * global_eikonal["total"]
        + float(ray_lipschitz_policy.top_level_weight) * ray_lipschitz_total
    )
    return {
        "total": total,
        "base_total_before_v42_regularity": base_total,
        "structural_total": structural,
        "local": local,
        "teacher_surface_zero": teacher_zero,
        "foreground_existence_total": foreground_total,
        "negative_space_total": negative_total,
        "source_exterior_total": exterior["total"],
        "source_exterior_violating_fraction": exterior["violating_fraction"],
        "source_exterior_negative_fraction": exterior["negative_fraction"],
        "global_eikonal_total": global_eikonal["total"],
        "global_eikonal_weight": float(global_eikonal_policy.top_level_weight),
        "global_eikonal_sample_count": global_eikonal["sample_count"],
        "global_eikonal_gradient_norm_mean": global_eikonal["gradient_norm_mean"],
        "global_eikonal_gradient_norm_p95": global_eikonal["gradient_norm_p95"],
        "global_eikonal_gradient_norm_abs_error_mean": global_eikonal["gradient_norm_abs_error_mean"],
        "ray_lipschitz_total": ray_lipschitz_total,
        "ray_lipschitz_weight": float(ray_lipschitz_policy.top_level_weight),
        "source_views": tuple(rows),
    }
