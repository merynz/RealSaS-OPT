from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from models.iris.v3.dense_source_coverage_v3 import DenseSourceCoveragePolicyV3
from models.iris.v3.negative_space_barrier_v3 import query_dense_ray_sdf_v3
from models.iris.v4.source_exterior_v4 import source_exterior_metric_barrier_v4


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


def component_balanced_foreground_existence_hinge_v4(
    sdf_samples: torch.Tensor,
    target_foreground: torch.Tensor,
    component_id: torch.Tensor,
    boundary: torch.Tensor,
    *,
    boundary_inside_margin_normalized: float,
    interior_inside_margin_normalized: float,
) -> dict[str, torch.Tensor]:
    """Saturating source-foreground existence constraint.

    A source-foreground pixel only certifies that the continuous signed field must enter
    the interior somewhere along that admitted ray. Unlike BCE(sigmoid(-minSDF/beta),1),
    this hinge gives no reward for driving an already-satisfied ray deeper negative.
    Equal component means keep small source components consequential without inverse-area
    pixel weights.
    """

    if sdf_samples.ndim != 3:
        raise ValueError("sdf_samples must be [B,R,D]")
    expected = sdf_samples.shape[:2]
    if (
        target_foreground.shape != expected
        or component_id.shape != expected
        or boundary.shape != expected
    ):
        raise ValueError("foreground existence metadata must be [B,R]")
    if not torch.isfinite(sdf_samples).all():
        raise ValueError("foreground existence SDF must be finite")
    boundary_margin = float(boundary_inside_margin_normalized)
    interior_margin = float(interior_inside_margin_normalized)
    if (
        not math.isfinite(boundary_margin)
        or not math.isfinite(interior_margin)
        or boundary_margin <= 0.0
        or interior_margin <= 0.0
        or boundary_margin > interior_margin
    ):
        raise ValueError("invalid foreground inside margins")

    target = target_foreground > 0.5
    cid = component_id.to(dtype=torch.long)
    if torch.any(target & (cid < 0)):
        raise ValueError("foreground rays require non-negative component_id")
    if not torch.any(target):
        raise ValueError("foreground existence requires foreground rays")

    ray_min = torch.amin(sdf_samples, dim=-1)
    margin = torch.where(
        boundary > 0.5,
        torch.as_tensor(boundary_margin, device=ray_min.device, dtype=ray_min.dtype),
        torch.as_tensor(interior_margin, device=ray_min.device, dtype=ray_min.dtype),
    )
    per_ray = F.relu(ray_min + margin)

    groups: list[torch.Tensor] = []
    positive_ids = torch.unique(cid[target], sorted=True)
    for group_id in positive_ids:
        mask = target & (cid == group_id)
        groups.append(per_ray[mask].mean())
    if not groups:
        raise ValueError("foreground existence has no valid component group")

    fg_min = ray_min[target]
    fg_margin = margin[target]
    total = torch.stack(groups).mean()
    return {
        "total": total,
        "group_count": torch.as_tensor(len(groups), device=ray_min.device, dtype=torch.int64),
        "satisfied_fraction": (fg_min <= -fg_margin).to(dtype=ray_min.dtype).mean(),
        "zero_crossing_fraction": (fg_min <= 0.0).to(dtype=ray_min.dtype).mean(),
        "minimum_ray_sdf": torch.amin(fg_min),
        "maximum_ray_sdf": torch.amax(fg_min),
        "boundary_inside_margin": torch.as_tensor(boundary_margin, device=ray_min.device, dtype=ray_min.dtype),
        "interior_inside_margin": torch.as_tensor(interior_margin, device=ray_min.device, dtype=ray_min.dtype),
    }


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
    source_exterior_points_normalized: torch.Tensor,
    source_exterior_margin_normalized: torch.Tensor,
    source_view_batches: tuple[SourceConstraintRayBatchV4, ...],
    coverage_policy: DenseSourceCoveragePolicyV3 = DenseSourceCoveragePolicyV3(),
    teacher_surface_zero_weight: float = 0.50,
    foreground_existence_weight: float = 1.0,
    negative_space_weight: float = 1.0,
    source_exterior_weight: float = 1.0,
    maximum_background_margin_normalized: float = 0.04,
    query_chunk: int = 131072,
) -> dict[str, object]:
    """All-view V4 FIT1 objective with retained 2D and 3D source authority.

    Source foreground is a saturating existence constraint, not a probability objective:
    once a ray contains sufficient interior evidence it receives no further pressure to
    thicken geometry. Source background is stronger and remains pointwise exterior along
    the whole admitted ray. Teacher geometry remains FIT-only local structural aid.
    """

    coverage_policy.validate()
    if int(query_chunk) <= 0:
        raise ValueError("query_chunk must be positive")
    if len(source_view_batches) != 8:
        raise ValueError("V4 demo objective requires all eight source views every step")
    view_indices = [int(batch.view_index) for batch in source_view_batches]
    if set(view_indices) != set(range(8)) or len(set(view_indices)) != 8:
        raise ValueError("V4 source-view batches must be unique V0..V7")
    if (
        source_exterior_points_normalized.ndim != 3
        or source_exterior_points_normalized.shape[-1] != 3
        or source_exterior_points_normalized.shape[:2]
        != source_exterior_margin_normalized.shape
        or source_exterior_points_normalized.numel() == 0
    ):
        raise ValueError("source exterior points must be [B,N,3] with margins [B,N]")

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
        foreground_terms.append(foreground["total"])
        negative_terms.append(negative["total"])
        rows.append({
            "view_index": int(batch.view_index),
            "foreground_existence_total": foreground["total"],
            "foreground_component_group_count": foreground["group_count"],
            "foreground_satisfied_fraction": foreground["satisfied_fraction"],
            "foreground_zero_crossing_fraction": foreground["zero_crossing_fraction"],
            "foreground_minimum_ray_sdf": foreground["minimum_ray_sdf"],
            "foreground_maximum_ray_sdf": foreground["maximum_ray_sdf"],
            "foreground_boundary_inside_margin": foreground["boundary_inside_margin"],
            "foreground_interior_inside_margin": foreground["interior_inside_margin"],
            "negative_space_total": negative["total"],
            "negative_space_violating_point_fraction": negative["violating_point_fraction"],
            "negative_space_zero_crossing_fraction": negative["zero_crossing_fraction"],
            "minimum_background_sdf": negative["minimum_background_sdf"],
            "minimum_margin": negative["minimum_margin"],
            "maximum_margin": negative["maximum_margin"],
            "hard_replay_ray_count": int(batch.hard_replay.sum().detach().cpu()),
            "ray_count": int(batch.target_foreground.numel()),
        })

    foreground_total = torch.stack(foreground_terms).mean()
    negative_total = torch.stack(negative_terms).mean()
    total = (
        structural
        + float(foreground_existence_weight) * foreground_total
        + float(negative_space_weight) * negative_total
        + float(source_exterior_weight) * exterior["total"]
    )
    return {
        "total": total,
        "structural_total": structural,
        "local": local,
        "teacher_surface_zero": teacher_zero,
        "foreground_existence_total": foreground_total,
        "negative_space_total": negative_total,
        "source_exterior_total": exterior["total"],
        "source_exterior_violating_fraction": exterior["violating_fraction"],
        "source_exterior_negative_fraction": exterior["negative_fraction"],
        "source_exterior_minimum_sdf": exterior["minimum_sdf"],
        "source_exterior_minimum_margin": exterior["minimum_margin"],
        "source_exterior_maximum_margin": exterior["maximum_margin"],
        "source_views": tuple(rows),
    }
