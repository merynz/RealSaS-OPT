from __future__ import annotations

from dataclasses import dataclass

import torch

from models.iris.v3.dense_source_coverage_v3 import (
    DenseSourceCoveragePolicyV3,
    component_balanced_source_coverage_loss_from_logits,
    historical_sparse_objective_v3,
    ray_foreground_logit_from_sdf_samples,
)
from models.iris.v3.negative_space_barrier_v3 import (
    background_full_ray_empty_space_barrier,
    query_dense_ray_sdf_v3,
)


@dataclass(frozen=True)
class DenseRayTrainingBatchV3:
    ray_points_normalized: torch.Tensor
    target_foreground: torch.Tensor
    component_id: torch.Tensor
    boundary: torch.Tensor
    view_index: int

    def validate(self) -> None:
        if self.ray_points_normalized.ndim != 4 or self.ray_points_normalized.shape[-1] != 3:
            raise ValueError("ray_points_normalized must be [B,R,D,3]")
        expected = self.ray_points_normalized.shape[:2]
        for name, value in (
            ("target_foreground", self.target_foreground),
            ("component_id", self.component_id),
            ("boundary", self.boundary),
        ):
            if tuple(value.shape) != tuple(expected):
                raise ValueError(f"{name} must be [B,R]")
        if int(self.ray_points_normalized.shape[1]) <= 0 or int(self.ray_points_normalized.shape[2]) <= 0:
            raise ValueError("dense ray batch must contain rays and depth samples")
        if not (0 <= int(self.view_index) < 8):
            raise ValueError("view_index must be in [0,7]")


def compute_v3_dense_fit_objective(
    *,
    model,
    scene_planes: torch.Tensor,
    local_points_normalized: torch.Tensor,
    local_target_sdf: torch.Tensor,
    background_points_normalized: torch.Tensor,
    certified_outside_points_normalized: torch.Tensor,
    teacher_surface_zero_points_normalized: torch.Tensor,
    dense_view_batches: tuple[DenseRayTrainingBatchV3, ...],
    policy: DenseSourceCoveragePolicyV3 = DenseSourceCoveragePolicyV3(),
    positive_margin: float = 0.04,
    query_chunk: int = 131072,
    enforce_full_ray_background_empty_space: bool = True,
) -> dict[str, torch.Tensor | tuple[dict[str, torch.Tensor | int], ...]]:
    """Historical STRIDE8 sparse objective plus source-raster constraints.

    Foreground rays retain the Arm-A occupancy semantics: some negative/inside field
    evidence must occur along the admitted ray. Source-background rays carry stronger
    authority: the whole admitted segment is certified empty, so every sampled SDF
    point is additionally constrained to stay positive by the historical 0.04 margin.

    The full-ray barrier is one-sided negative-space supervision. It does not use a
    teacher mesh, does not make the source mask a forward input, and does not replace
    the learned field with a visual hull.
    """

    policy.validate()
    if not dense_view_batches:
        raise ValueError("dense_view_batches must be non-empty")
    if int(query_chunk) <= 0:
        raise ValueError("query_chunk must be positive")

    local_sdf = model.query(scene_planes, local_points_normalized)["sdf"]
    background_sdf = model.query(scene_planes, background_points_normalized)["sdf"]
    outside_sdf = model.query(scene_planes, certified_outside_points_normalized)["sdf"]
    teacher_surface_sdf = model.query(scene_planes, teacher_surface_zero_points_normalized)["sdf"]
    sparse = historical_sparse_objective_v3(
        local_sdf,
        local_target_sdf,
        background_sdf,
        outside_sdf,
        teacher_surface_sdf,
        positive_margin=positive_margin,
    )

    dense_rows: list[dict[str, torch.Tensor | int]] = []
    dense_totals: list[torch.Tensor] = []
    empty_space_totals: list[torch.Tensor] = []
    seen_views: set[int] = set()
    for batch in dense_view_batches:
        batch.validate()
        view_index = int(batch.view_index)
        if view_index in seen_views:
            raise ValueError(f"duplicate dense scheduled view:{view_index}")
        seen_views.add(view_index)

        ray_sdf = query_dense_ray_sdf_v3(
            model,
            scene_planes,
            batch.ray_points_normalized,
            query_chunk=query_chunk,
        )
        logit = ray_foreground_logit_from_sdf_samples(ray_sdf, policy=policy)
        coverage = component_balanced_source_coverage_loss_from_logits(
            logit,
            batch.target_foreground,
            batch.component_id,
            batch.boundary,
            policy=policy,
        )
        if enforce_full_ray_background_empty_space:
            empty = background_full_ray_empty_space_barrier(
                ray_sdf,
                batch.target_foreground,
                positive_margin=positive_margin,
            )
        else:
            zero = ray_sdf.sum() * 0.0
            empty = {
                "total": zero,
                "background_ray_count": torch.zeros((), dtype=torch.int64, device=ray_sdf.device),
                "violating_point_fraction": zero,
                "minimum_background_sdf": torch.full((), float("nan"), dtype=ray_sdf.dtype, device=ray_sdf.device),
            }

        dense_totals.append(coverage["total"])
        empty_space_totals.append(empty["total"])
        dense_rows.append(
            {
                "view_index": view_index,
                "total": coverage["total"],
                "component_balanced_bce": coverage["component_balanced_bce"],
                "soft_dice": coverage["soft_dice"],
                "group_count": coverage["group_count"],
                "background_full_ray_empty_space": empty["total"],
                "background_ray_count": empty["background_ray_count"],
                "background_violating_point_fraction": empty["violating_point_fraction"],
                "minimum_background_sdf": empty["minimum_background_sdf"],
            }
        )

    dense_total = torch.stack(dense_totals).mean()
    empty_space_total = torch.stack(empty_space_totals).mean()
    total = (
        sparse["total"]
        + float(policy.top_level_loss_weight) * dense_total
        + empty_space_total
    )
    return {
        "total": total,
        "sparse_total": sparse["total"],
        "dense_total": dense_total,
        "background_full_ray_empty_space": empty_space_total,
        "local": sparse["local"],
        "background": sparse["background"],
        "certified_outside": sparse["certified_outside"],
        "teacher_surface_zero": sparse["teacher_surface_zero"],
        "dense_views": tuple(dense_rows),
    }
