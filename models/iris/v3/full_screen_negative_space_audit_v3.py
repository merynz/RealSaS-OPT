from __future__ import annotations

import numpy as np
import torch

from models.iris.v3.dense_source_coverage_v3 import (
    DenseSourceCoveragePolicyV3,
    phase_jittered_unit_cube_ray_points,
)
from models.iris.v3.dense_source_sampling_v3 import camera_pixel_ray_origins_normalized
from models.iris.v3.negative_space_barrier_v3 import query_dense_ray_sdf_v3
from models.iris.v3.negative_space_upper_bound_v3 import (
    background_distance_to_source_px,
    distance_binned_violation_summary,
    metric_consistent_background_margin_normalized,
)


@torch.no_grad()
def audit_full_screen_negative_space_v3(
    *,
    model,
    scene_planes: torch.Tensor,
    foreground_mask: np.ndarray,
    admitted_mask: np.ndarray,
    camera_origin_normalized: np.ndarray,
    camera_right: np.ndarray,
    camera_screen_up: np.ndarray,
    camera_forward: np.ndarray,
    camera_half_extent_normalized: float,
    phase_fraction: float = 0.5,
    pixel_chunk: int = 512,
    query_chunk: int = 131072,
    policy: DenseSourceCoveragePolicyV3 = DenseSourceCoveragePolicyV3(),
) -> dict[str, object]:
    """Evaluate every admitted source-background pixel against the learned V3 field.

    Returns raster-aligned ray-min SDF and metric-margin violation maps plus distance-
    binned summaries. This is diagnostic only and never clips or repairs the field.
    """
    foreground = np.asarray(foreground_mask, dtype=bool)
    admitted = np.asarray(admitted_mask, dtype=bool)
    if foreground.ndim != 2 or admitted.shape != foreground.shape:
        raise ValueError("foreground/admitted masks must be matching 2D arrays")
    h, w = map(int, foreground.shape)
    background = (~foreground) & admitted
    flat = np.flatnonzero(background.reshape(-1)).astype(np.int64)
    if flat.size == 0:
        raise ValueError("no admitted source-background pixels to audit")
    if int(pixel_chunk) <= 0:
        raise ValueError("pixel_chunk must be positive")

    distance = background_distance_to_source_px(foreground)
    margin_map = metric_consistent_background_margin_normalized(
        distance,
        camera_half_extent_normalized=float(camera_half_extent_normalized),
        raster_width=w,
        maximum_margin_normalized=0.04,
    )
    ray_min_flat = np.full(h * w, np.nan, dtype=np.float32)

    device = scene_planes.device
    dtype = scene_planes.dtype
    for start in range(0, int(flat.size), int(pixel_chunk)):
        index = flat[start : start + int(pixel_chunk)]
        origin_np, direction_np = camera_pixel_ray_origins_normalized(
            index,
            width=w,
            height=h,
            camera_origin_normalized=np.asarray(camera_origin_normalized, dtype=np.float64),
            camera_right=np.asarray(camera_right, dtype=np.float64),
            camera_screen_up=np.asarray(camera_screen_up, dtype=np.float64),
            camera_forward=np.asarray(camera_forward, dtype=np.float64),
            camera_half_extent_normalized=float(camera_half_extent_normalized),
        )
        origin = torch.as_tensor(origin_np, device=device, dtype=dtype)
        direction = torch.as_tensor(direction_np, device=device, dtype=dtype)
        points = phase_jittered_unit_cube_ray_points(
            origin,
            direction,
            phase_fraction=float(phase_fraction),
            policy=policy,
        )[None, ...]
        sdf = query_dense_ray_sdf_v3(
            model,
            scene_planes,
            points,
            query_chunk=int(query_chunk),
        )[0]
        ray_min_flat[index] = torch.amin(sdf, dim=-1).float().cpu().numpy()

    ray_min_map = ray_min_flat.reshape(h, w)
    valid_distance = distance[background]
    valid_min = ray_min_map[background]
    valid_margin = np.asarray(margin_map, dtype=np.float32)[background]
    summary = distance_binned_violation_summary(valid_distance, valid_min, valid_margin)
    zero_cross = np.zeros((h, w), dtype=bool)
    margin_violation = np.zeros((h, w), dtype=bool)
    zero_cross[background] = valid_min <= 0.0
    margin_violation[background] = valid_min < valid_margin
    return {
        "ray_min_sdf": ray_min_map,
        "target_margin_normalized": np.asarray(margin_map, dtype=np.float32),
        "zero_crossing_mask": zero_cross,
        "margin_violation_mask": margin_violation,
        "distance_to_source_px": distance.astype(np.float32),
        "background_pixel_count": int(flat.size),
        "zero_crossing_fraction": float(np.mean(valid_min <= 0.0)),
        "margin_violation_fraction": float(np.mean(valid_min < valid_margin)),
        "distance_binned": summary,
        "phase_fraction": float(phase_fraction),
    }
