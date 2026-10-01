from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.iris.v3.dense_source_coverage_v3 import (
    DenseSourceCoveragePolicyV3,
    component_balanced_source_coverage_loss_from_logits,
    ray_foreground_logit_from_sdf_samples,
)
from models.iris.v3.negative_space_barrier_v3 import query_dense_ray_sdf_v3
from models.iris.v3.negative_space_upper_bound_v3 import graded_background_full_ray_barrier
from models.iris.v3.train_dense_fit_v3 import DenseRayTrainingBatchV3


@dataclass(frozen=True)
class GradedNegativeSpaceBatchV3:
    ray_points_normalized: torch.Tensor
    target_margin_normalized: torch.Tensor
    view_index: int

    def validate(self) -> None:
        if self.ray_points_normalized.ndim != 4 or self.ray_points_normalized.shape[-1] != 3:
            raise ValueError("ray_points_normalized must be [B,R,D,3]")
        if self.target_margin_normalized.shape != self.ray_points_normalized.shape[:2]:
            raise ValueError("target_margin_normalized must be [B,R]")
        if not torch.isfinite(self.ray_points_normalized).all() or not torch.isfinite(self.target_margin_normalized).all():
            raise ValueError("graded negative-space batch must be finite")
        if torch.any(self.target_margin_normalized <= 0.0):
            raise ValueError("graded negative-space margins must be positive")
        if not (0 <= int(self.view_index) < 8):
            raise ValueError("view_index must be in [0,7]")


def compute_v3_owner_separation_upper_bound_objective(
    *,
    model,
    scene_planes: torch.Tensor,
    local_points_normalized: torch.Tensor,
    local_target_sdf: torch.Tensor,
    teacher_surface_zero_points_normalized: torch.Tensor,
    dense_view_batches: tuple[DenseRayTrainingBatchV3, ...],
    graded_negative_space_batches: tuple[GradedNegativeSpaceBatchV3, ...],
    policy: DenseSourceCoveragePolicyV3 = DenseSourceCoveragePolicyV3(),
    query_chunk: int = 131072,
) -> dict[str, torch.Tensor | tuple[dict[str, torch.Tensor | int], ...]]:
    """Owner-separation diagnostic, not a product recipe.

    Keep the metric local-SDF and teacher-surface terms from historical STRIDE8, keep
    the Arm-A dense source occupancy term, but deliberately remove the historical
    fixed +0.04 sparse background/certified-outside terms because their margin is not
    metric-consistent near the source silhouette. Replace negative-space authority by
    explicit source-distance-derived full-ray barriers supplied by the caller.

    Interpretation is binary: if the current V3 representation cannot satisfy dense
    source occupancy plus metric-consistent systematically-covered negative space, the
    representation becomes the remaining owner. Passing only shows that sampling /
    objective design can explain the prior source-support failure.
    """
    policy.validate()
    if not dense_view_batches or not graded_negative_space_batches:
        raise ValueError("upper-bound objective requires dense foreground and graded negative-space batches")
    if int(query_chunk) <= 0:
        raise ValueError("query_chunk must be positive")

    local_sdf = model.query(scene_planes, local_points_normalized)["sdf"]
    teacher_surface_sdf = model.query(scene_planes, teacher_surface_zero_points_normalized)["sdf"]
    if local_sdf.shape != local_target_sdf.shape:
        raise ValueError("local_sdf/local_target_sdf shape mismatch")
    local = F.smooth_l1_loss(local_sdf.float(), local_target_sdf.float(), beta=0.02)
    teacher_surface_zero = F.smooth_l1_loss(
        teacher_surface_sdf.float(),
        torch.zeros_like(teacher_surface_sdf.float()),
        beta=0.01,
    )
    structural_sparse = local + 0.50 * teacher_surface_zero

    dense_rows: list[dict[str, torch.Tensor | int]] = []
    dense_totals: list[torch.Tensor] = []
    dense_seen: set[int] = set()
    for batch in dense_view_batches:
        batch.validate()
        view_index = int(batch.view_index)
        if view_index in dense_seen:
            raise ValueError(f"duplicate dense scheduled view:{view_index}")
        dense_seen.add(view_index)
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
        dense_totals.append(coverage["total"])
        dense_rows.append({
            "view_index": view_index,
            "total": coverage["total"],
            "component_balanced_bce": coverage["component_balanced_bce"],
            "soft_dice": coverage["soft_dice"],
            "group_count": coverage["group_count"],
        })

    negative_rows: list[dict[str, torch.Tensor | int]] = []
    negative_totals: list[torch.Tensor] = []
    negative_seen: set[int] = set()
    for batch in graded_negative_space_batches:
        batch.validate()
        view_index = int(batch.view_index)
        if view_index in negative_seen:
            raise ValueError(f"duplicate graded negative-space view:{view_index}")
        negative_seen.add(view_index)
        ray_sdf = query_dense_ray_sdf_v3(
            model,
            scene_planes,
            batch.ray_points_normalized,
            query_chunk=query_chunk,
        )
        barrier = graded_background_full_ray_barrier(ray_sdf, batch.target_margin_normalized)
        negative_totals.append(barrier["total"])
        negative_rows.append({
            "view_index": view_index,
            "total": barrier["total"],
            "violating_point_fraction": barrier["violating_point_fraction"],
            "minimum_background_sdf": barrier["minimum_background_sdf"],
            "minimum_target_margin": barrier["minimum_target_margin"],
            "maximum_target_margin": barrier["maximum_target_margin"],
        })

    dense_total = torch.stack(dense_totals).mean()
    negative_total = torch.stack(negative_totals).mean()
    total = structural_sparse + float(policy.top_level_loss_weight) * dense_total + negative_total
    return {
        "total": total,
        "structural_sparse_total": structural_sparse,
        "local": local,
        "teacher_surface_zero": teacher_surface_zero,
        "dense_total": dense_total,
        "graded_negative_space_total": negative_total,
        "dense_views": tuple(dense_rows),
        "graded_negative_space_views": tuple(negative_rows),
    }
