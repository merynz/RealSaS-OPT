from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class NegativeSpaceUpperBoundPolicyV3:
    boundary_band_max_px: float = 32.0
    background_rays_per_active_view: int = 256
    maximum_margin_normalized: float = 0.04

    def validate(self) -> None:
        if not math.isfinite(self.boundary_band_max_px) or self.boundary_band_max_px <= 0.0:
            raise ValueError("boundary_band_max_px must be finite and positive")
        if int(self.background_rays_per_active_view) <= 0:
            raise ValueError("background_rays_per_active_view must be positive")
        if not math.isfinite(self.maximum_margin_normalized) or self.maximum_margin_normalized <= 0.0:
            raise ValueError("maximum_margin_normalized must be finite and positive")


def background_distance_to_source_px(foreground_mask: np.ndarray) -> np.ndarray:
    foreground = np.asarray(foreground_mask, dtype=bool)
    if foreground.ndim != 2 or foreground.size == 0 or not np.any(foreground):
        raise ValueError("foreground_mask must be a non-empty 2D foreground mask")
    # EDT of background pixels to the nearest foreground pixel. Foreground itself is zero.
    distance = ndimage.distance_transform_edt(~foreground).astype(np.float32)
    distance[foreground] = 0.0
    return distance


def metric_consistent_background_margin_normalized(
    distance_px: np.ndarray | torch.Tensor,
    *,
    camera_half_extent_normalized: float,
    raster_width: int,
    maximum_margin_normalized: float = 0.04,
):
    half = float(camera_half_extent_normalized)
    width = int(raster_width)
    cap = float(maximum_margin_normalized)
    if not math.isfinite(half) or half <= 0.0:
        raise ValueError("camera_half_extent_normalized must be finite and positive")
    if width <= 0:
        raise ValueError("raster_width must be positive")
    if not math.isfinite(cap) or cap <= 0.0:
        raise ValueError("maximum_margin_normalized must be finite and positive")
    pixel_scale = 2.0 * half / float(width)
    if isinstance(distance_px, torch.Tensor):
        return torch.clamp(distance_px.to(dtype=torch.float32) * pixel_scale, min=0.0, max=cap)
    value = np.asarray(distance_px, dtype=np.float32) * np.float32(pixel_scale)
    return np.clip(value, 0.0, cap).astype(np.float32)


def boundary_band_background_pool(
    foreground_mask: np.ndarray,
    admitted_mask: np.ndarray,
    *,
    maximum_distance_px: float = 32.0,
) -> tuple[np.ndarray, np.ndarray]:
    foreground = np.asarray(foreground_mask, dtype=bool)
    admitted = np.asarray(admitted_mask, dtype=bool)
    if foreground.shape != admitted.shape or foreground.ndim != 2:
        raise ValueError("foreground/admitted masks must be matching 2D arrays")
    distance = background_distance_to_source_px(foreground)
    pool_mask = (~foreground) & admitted & (distance > 0.0) & (distance <= float(maximum_distance_px))
    flat = np.flatnonzero(pool_mask.reshape(-1)).astype(np.int64)
    if flat.size == 0:
        raise ValueError("boundary-band background pool is empty")
    dist = distance.reshape(-1)[flat]
    # Stable nearest-boundary-first order makes the cycle deterministic and inspectable.
    order = np.lexsort((flat, dist))
    return flat[order], dist[order].astype(np.float32)


def cyclic_boundary_band_selection(
    foreground_mask: np.ndarray,
    admitted_mask: np.ndarray,
    *,
    visit_index: int,
    count: int,
    maximum_distance_px: float = 32.0,
) -> tuple[np.ndarray, np.ndarray]:
    pool, distance = boundary_band_background_pool(
        foreground_mask,
        admitted_mask,
        maximum_distance_px=maximum_distance_px,
    )
    n = int(count)
    visit = int(visit_index)
    if n <= 0 or visit < 0:
        raise ValueError("count must be positive and visit_index non-negative")
    start = (visit * n) % int(pool.size)
    take = (start + np.arange(n, dtype=np.int64)) % int(pool.size)
    return pool[take], distance[take]


def graded_background_full_ray_barrier(
    sdf_samples: torch.Tensor,
    target_margin_normalized: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if sdf_samples.ndim != 3:
        raise ValueError("sdf_samples must be [B,R,D]")
    if target_margin_normalized.shape != sdf_samples.shape[:2]:
        raise ValueError("target_margin_normalized must be [B,R]")
    if not torch.isfinite(sdf_samples).all() or not torch.isfinite(target_margin_normalized).all():
        raise ValueError("graded barrier inputs must be finite")
    margin = target_margin_normalized.to(dtype=sdf_samples.dtype)
    if torch.any(margin <= 0.0):
        raise ValueError("every upper-bound background ray requires a positive metric margin")
    violation = F.relu(margin[..., None] - sdf_samples)
    return {
        "total": violation.mean(),
        "violating_point_fraction": (sdf_samples < margin[..., None]).to(dtype=sdf_samples.dtype).mean(),
        "minimum_background_sdf": torch.amin(sdf_samples),
        "minimum_target_margin": torch.amin(margin),
        "maximum_target_margin": torch.amax(margin),
    }


def distance_binned_violation_summary(
    distance_px: np.ndarray,
    ray_min_sdf: np.ndarray,
    target_margin_normalized: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    distance = np.asarray(distance_px, dtype=np.float64).reshape(-1)
    ray_min = np.asarray(ray_min_sdf, dtype=np.float64).reshape(-1)
    margin = np.asarray(target_margin_normalized, dtype=np.float64).reshape(-1)
    if distance.shape != ray_min.shape or distance.shape != margin.shape:
        raise ValueError("distance/ray_min/margin shapes must match")
    if not (np.isfinite(distance).all() and np.isfinite(ray_min).all() and np.isfinite(margin).all()):
        raise ValueError("violation summary inputs must be finite")
    bins = ((0.0, 2.0, "0_2"), (2.0, 8.0, "2_8"), (8.0, 32.0, "8_32"), (32.0, np.inf, "gt_32"))
    out: dict[str, dict[str, float | int]] = {}
    for lo, hi, name in bins:
        mask = (distance > lo) & (distance <= hi)
        count = int(mask.sum())
        if count == 0:
            out[name] = {"count": 0, "zero_crossing_fraction": 0.0, "margin_violation_fraction": 0.0}
            continue
        out[name] = {
            "count": count,
            "zero_crossing_fraction": float(np.mean(ray_min[mask] <= 0.0)),
            "margin_violation_fraction": float(np.mean(ray_min[mask] < margin[mask])),
        }
    return out
