from __future__ import annotations

from dataclasses import dataclass

import torch

from models.iris.v3.dense_source_coverage_v3 import (
    DenseSourceCoveragePolicyV3,
    component_balanced_source_coverage_loss_from_logits,
    historical_sparse_objective_v3,
    query_ray_foreground_logit_v3,
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
) -> dict[str, torch.Tensor | tuple[dict[str, torch.Tensor | int], ...]]:
    """Historical STRIDE8 V3 sparse objective plus one frozen dense source term.

    The historical sparse terms are intentionally unchanged. The only added scientific
    variable is the mean dense source-coverage loss over the supplied scheduled views,
    multiplied by the frozen subject-free top-level weight.
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
    seen_views: set[int] = set()
    for batch in dense_view_batches:
        batch.validate()
        view_index = int(batch.view_index)
        if view_index in seen_views:
            raise ValueError(f"duplicate dense scheduled view:{view_index}")
        seen_views.add(view_index)
        logit = query_ray_foreground_logit_v3(
            model,
            scene_planes,
            batch.ray_points_normalized,
            policy=policy,
            query_chunk=query_chunk,
        )
        row = component_balanced_source_coverage_loss_from_logits(
            logit,
            batch.target_foreground,
            batch.component_id,
            batch.boundary,
            policy=policy,
        )
        dense_totals.append(row["total"])
        dense_rows.append(
            {
                "view_index": view_index,
                "total": row["total"],
                "component_balanced_bce": row["component_balanced_bce"],
                "soft_dice": row["soft_dice"],
                "group_count": row["group_count"],
            }
        )

    dense_total = torch.stack(dense_totals).mean()
    total = sparse["total"] + float(policy.top_level_loss_weight) * dense_total
    return {
        "total": total,
        "sparse_total": sparse["total"],
        "dense_total": dense_total,
        "local": sparse["local"],
        "background": sparse["background"],
        "certified_outside": sparse["certified_outside"],
        "teacher_surface_zero": sparse["teacher_surface_zero"],
        "dense_views": tuple(dense_rows),
    }
