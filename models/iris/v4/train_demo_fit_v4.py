from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from models.iris.v3.dense_source_coverage_v3 import (
    DenseSourceCoveragePolicyV3,
    component_balanced_source_coverage_loss_from_logits,
    ray_foreground_logit_from_sdf_samples,
)
from models.iris.v3.negative_space_barrier_v3 import query_dense_ray_sdf_v3


@dataclass(frozen=True)
class SourceConstraintRayBatchV4:
    ray_points_normalized: torch.Tensor
    target_foreground: torch.Tensor
    component_id: torch.Tensor
    boundary: torch.Tensor
    distance_to_foreground_px: torch.Tensor
    hard_replay: torch.Tensor
    camera_half_extent_normalized: float
    raster_width: int
    view_index: int

    def validate(self) -> None:
        if self.ray_points_normalized.ndim != 4 or self.ray_points_normalized.shape[-1] != 3:
            raise ValueError("ray_points_normalized must be [B,R,D,3]")
        expected = self.ray_points_normalized.shape[:2]
        for name, value in (
            ("target_foreground", self.target_foreground),
            ("component_id", self.component_id),
            ("boundary", self.boundary),
            ("distance_to_foreground_px", self.distance_to_foreground_px),
            ("hard_replay", self.hard_replay),
        ):
            if tuple(value.shape) != tuple(expected):
                raise ValueError(f"{name} must be [B,R]")
        if not (0 <= int(self.view_index) < 8):
            raise ValueError("view_index must be in [0,7]")
        if int(self.raster_width) <= 0:
            raise ValueError("raster_width must be positive")
        half = float(self.camera_half_extent_normalized)
        if not math.isfinite(half) or half <= 0.0:
            raise ValueError("camera_half_extent_normalized must be finite and positive")
        target = self.target_foreground
        if torch.any((target != 0) & (target != 1)):
            raise ValueError("target_foreground must be binary")
        background = target < 0.5
        if torch.any(self.distance_to_foreground_px[background] <= 0):
            raise ValueError("source-background rays require positive source distance")
        if torch.any(self.distance_to_foreground_px[~background] != 0):
            raise ValueError("source-foreground rays require zero source distance")


def metric_background_margin_v4(
    distance_px: torch.Tensor,
    *,
    camera_half_extent_normalized: float,
    raster_width: int,
    maximum_margin_normalized: float = 0.04,
) -> torch.Tensor:
    half = float(camera_half_extent_normalized)
    width = int(raster_width)
    cap = float(maximum_margin_normalized)
    if not math.isfinite(half) or half <= 0 or width <= 0:
        raise ValueError("invalid camera metric for V4 source margin")
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError("maximum_margin_normalized must be finite and positive")
    pixel_scale = 2.0 * half / float(width)
    return torch.clamp(distance_px.float() * pixel_scale, min=0.0, max=cap)


def source_negative_space_barrier_v4(
    sdf_samples: torch.Tensor,
    target_foreground: torch.Tensor,
    distance_to_foreground_px: torch.Tensor,
    *,
    camera_half_extent_normalized: float,
    raster_width: int,
    maximum_margin_normalized: float = 0.04,
) -> dict[str, torch.Tensor]:
    if sdf_samples.ndim != 3:
        raise ValueError("sdf_samples must be [B,R,D]")
    if target_foreground.shape != sdf_samples.shape[:2]:
        raise ValueError("target_foreground must be [B,R]")
    if distance_to_foreground_px.shape != target_foreground.shape:
        raise ValueError("distance_to_foreground_px must be [B,R]")
    background = target_foreground < 0.5
    if not torch.any(background):
        zero = sdf_samples.sum() * 0.0
        return {
            "total": zero,
            "violating_point_fraction": zero,
            "zero_crossing_fraction": zero,
            "minimum_background_sdf": torch.full((), float("nan"), device=sdf_samples.device),
            "minimum_margin": torch.full((), float("nan"), device=sdf_samples.device),
            "maximum_margin": torch.full((), float("nan"), device=sdf_samples.device),
        }
    margin = metric_background_margin_v4(
        distance_to_foreground_px,
        camera_half_extent_normalized=camera_half_extent_normalized,
        raster_width=raster_width,
        maximum_margin_normalized=maximum_margin_normalized,
    )
    bg_sdf = sdf_samples[background]
    bg_margin = margin[background]
    violation = F.relu(bg_margin[:, None] - bg_sdf)
    return {
        "total": violation.mean(),
        "violating_point_fraction": (bg_sdf < bg_margin[:, None]).float().mean(),
        "zero_crossing_fraction": (torch.amin(bg_sdf, dim=-1) <= 0.0).float().mean(),
        "minimum_background_sdf": torch.amin(bg_sdf),
        "minimum_margin": torch.amin(bg_margin),
        "maximum_margin": torch.amax(bg_margin),
    }


def compute_v4_demo_geometry_objective(
    *,
    model,
    scene_planes: torch.Tensor,
    local_points_normalized: torch.Tensor,
    local_target_sdf: torch.Tensor,
    teacher_surface_zero_points_normalized: torch.Tensor,
    source_view_batches: tuple[SourceConstraintRayBatchV4, ...],
    coverage_policy: DenseSourceCoveragePolicyV3 = DenseSourceCoveragePolicyV3(),
    teacher_surface_zero_weight: float = 0.50,
    negative_space_weight: float = 1.0,
    maximum_background_margin_normalized: float = 0.04,
    query_chunk: int = 131072,
) -> dict[str, object]:
    """All-view V4 FIT1 objective with retained near/far source constraints."""

    coverage_policy.validate()
    if int(query_chunk) <= 0:
        raise ValueError("query_chunk must be positive")
    if len(source_view_batches) != 8:
        raise ValueError("V4 demo objective requires all eight source views every step")
    view_indices = [int(batch.view_index) for batch in source_view_batches]
    if set(view_indices) != set(range(8)) or len(set(view_indices)) != 8:
        raise ValueError("V4 source-view batches must be unique V0..V7")

    local_sdf = model.query(scene_planes, local_points_normalized)["sdf"]
    teacher_sdf = model.query(scene_planes, teacher_surface_zero_points_normalized)["sdf"]
    if local_sdf.shape != local_target_sdf.shape:
        raise ValueError("local target shape mismatch")
    local = F.smooth_l1_loss(local_sdf.float(), local_target_sdf.float(), beta=0.02)
    teacher_zero = F.smooth_l1_loss(
        teacher_sdf.float(),
        torch.zeros_like(teacher_sdf.float()),
        beta=0.01,
    )
    structural = local + float(teacher_surface_zero_weight) * teacher_zero

    coverage_terms: list[torch.Tensor] = []
    negative_terms: list[torch.Tensor] = []
    rows: list[dict[str, object]] = []
    for batch in sorted(source_view_batches, key=lambda row: int(row.view_index)):
        batch.validate()
        ray_sdf = query_dense_ray_sdf_v3(
            model,
            scene_planes,
            batch.ray_points_normalized,
            query_chunk=query_chunk,
        )
        logit = ray_foreground_logit_from_sdf_samples(ray_sdf, policy=coverage_policy)
        coverage = component_balanced_source_coverage_loss_from_logits(
            logit,
            batch.target_foreground,
            batch.component_id,
            batch.boundary,
            policy=coverage_policy,
        )
        negative = source_negative_space_barrier_v4(
            ray_sdf,
            batch.target_foreground,
            batch.distance_to_foreground_px,
            camera_half_extent_normalized=batch.camera_half_extent_normalized,
            raster_width=batch.raster_width,
            maximum_margin_normalized=maximum_background_margin_normalized,
        )
        coverage_terms.append(coverage["total"])
        negative_terms.append(negative["total"])
        rows.append({
            "view_index": int(batch.view_index),
            "coverage_total": coverage["total"],
            "component_balanced_bce": coverage["component_balanced_bce"],
            "soft_dice": coverage["soft_dice"],
            "group_count": coverage["group_count"],
            "negative_space_total": negative["total"],
            "negative_space_violating_point_fraction": negative["violating_point_fraction"],
            "negative_space_zero_crossing_fraction": negative["zero_crossing_fraction"],
            "minimum_background_sdf": negative["minimum_background_sdf"],
            "minimum_margin": negative["minimum_margin"],
            "maximum_margin": negative["maximum_margin"],
            "hard_replay_ray_count": int(batch.hard_replay.sum().detach().cpu()),
            "ray_count": int(batch.target_foreground.numel()),
        })

    coverage_total = torch.stack(coverage_terms).mean()
    negative_total = torch.stack(negative_terms).mean()
    total = (
        structural
        + float(coverage_policy.top_level_loss_weight) * coverage_total
        + float(negative_space_weight) * negative_total
    )
    return {
        "total": total,
        "structural_total": structural,
        "local": local,
        "teacher_surface_zero": teacher_zero,
        "coverage_total": coverage_total,
        "negative_space_total": negative_total,
        "source_views": tuple(rows),
    }
