from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SourceExteriorPolicyV4:
    maximum_margin_normalized: float = 0.04
    pixel_quantization_guard_px: float = math.sqrt(2.0)

    def validate(self) -> None:
        if (
            not math.isfinite(self.maximum_margin_normalized)
            or self.maximum_margin_normalized <= 0.0
        ):
            raise ValueError("maximum_margin_normalized must be finite and positive")
        if (
            not math.isfinite(self.pixel_quantization_guard_px)
            or self.pixel_quantization_guard_px < 0.0
        ):
            raise ValueError("pixel_quantization_guard_px must be finite and non-negative")


def source_foreground_distance_fields_v4(
    foreground_masks: np.ndarray,
) -> np.ndarray:
    masks = np.asarray(foreground_masks, dtype=bool)
    if masks.ndim != 3 or masks.shape[0] != 8:
        raise ValueError("foreground_masks must be [8,H,W]")
    if min(masks.shape[1:]) <= 0 or not np.all(np.any(masks, axis=(1, 2))):
        raise ValueError("every source view requires non-empty foreground")
    return np.stack(
        [ndimage.distance_transform_edt(~masks[v]).astype(np.float32) for v in range(8)],
        axis=0,
    )


def certify_source_exterior_points_v4(
    points_normalized: np.ndarray,
    foreground_masks: np.ndarray,
    camera_origins_normalized: np.ndarray,
    camera_right: np.ndarray,
    camera_screen_up: np.ndarray,
    camera_forward: np.ndarray,
    camera_half_extent_normalized: np.ndarray,
    *,
    distance_fields_px: np.ndarray | None = None,
    policy: SourceExteriorPolicyV4 = SourceExteriorPolicyV4(),
) -> dict[str, np.ndarray]:
    """Certify 3D exterior points from exact orthographic source silhouettes.

    For any admitted in-frame view where a projected point lies on source background,
    the 2D distance to source foreground is a lower bound on the 3D Euclidean distance
    to the opaque subject. We subtract a conservative sqrt(2)-pixel raster/lookup guard
    before converting that projected distance into normalized world units. The maximum
    valid per-view lower bound remains a valid lower bound on true exterior distance.

    Out-of-frame projections are UNKNOWN and never provide negative evidence.
    """

    policy.validate()
    p = np.asarray(points_normalized, dtype=np.float64)
    masks = np.asarray(foreground_masks, dtype=bool)
    origins = np.asarray(camera_origins_normalized, dtype=np.float64)
    right = np.asarray(camera_right, dtype=np.float64)
    up = np.asarray(camera_screen_up, dtype=np.float64)
    forward = np.asarray(camera_forward, dtype=np.float64)
    half = np.asarray(camera_half_extent_normalized, dtype=np.float64).reshape(-1)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("points_normalized must be [N,3]")
    if masks.ndim != 3 or masks.shape[0] != 8:
        raise ValueError("foreground_masks must be [8,H,W]")
    if origins.shape != (8, 3) or right.shape != (8, 3) or up.shape != (8, 3) or forward.shape != (8, 3):
        raise ValueError("camera vectors must be [8,3]")
    if half.shape != (8,) or np.any(~np.isfinite(half)) or np.any(half <= 0.0):
        raise ValueError("camera_half_extent_normalized must be positive [8]")
    if not np.isfinite(p).all() or not np.isfinite(origins).all() or not np.isfinite(right).all() or not np.isfinite(up).all() or not np.isfinite(forward).all():
        raise ValueError("source exterior inputs must be finite")

    height, width = map(int, masks.shape[1:])
    fields = (
        source_foreground_distance_fields_v4(masks)
        if distance_fields_px is None
        else np.asarray(distance_fields_px, dtype=np.float32)
    )
    if fields.shape != masks.shape or not np.isfinite(fields).all():
        raise ValueError("distance_fields_px shape/content invalid")

    best_margin = np.zeros(len(p), dtype=np.float64)
    best_distance_px = np.zeros(len(p), dtype=np.float64)
    witness_view = np.full(len(p), -1, dtype=np.int64)
    observed_view_count = np.zeros(len(p), dtype=np.int64)

    for view in range(8):
        f = forward[view]
        fnorm = np.linalg.norm(f)
        rnorm = np.linalg.norm(right[view])
        unorm = np.linalg.norm(up[view])
        if min(fnorm, rnorm, unorm) <= 1e-12:
            raise ValueError("camera basis vector must be nonzero")
        f = f / fnorm
        r = right[view] / rnorm
        u = up[view] / unorm
        rel = p - origins[view][None, :]
        depth = rel @ f
        gx = (rel @ r) / half[view]
        gy = -(rel @ u) / half[view]
        px = (gx + 1.0) * 0.5 * float(width)
        py = (gy + 1.0) * 0.5 * float(height)
        valid = (
            (depth > 0.0)
            & (px >= 0.0)
            & (px < float(width))
            & (py >= 0.0)
            & (py < float(height))
        )
        observed_view_count += valid.astype(np.int64)
        if not np.any(valid):
            continue
        ii = np.flatnonzero(valid)
        ix = np.floor(px[ii]).astype(np.int64).clip(0, width - 1)
        iy = np.floor(py[ii]).astype(np.int64).clip(0, height - 1)
        is_background = ~masks[view, iy, ix]
        if not np.any(is_background):
            continue
        jj = ii[is_background]
        bx = ix[is_background]
        by = iy[is_background]
        raw_distance_px = fields[view, by, bx].astype(np.float64)
        conservative_px = np.maximum(
            0.0,
            raw_distance_px - float(policy.pixel_quantization_guard_px),
        )
        scale = 2.0 * half[view] / float(width)
        margin = np.minimum(
            float(policy.maximum_margin_normalized),
            conservative_px * scale,
        )
        improve = margin > best_margin[jj]
        if np.any(improve):
            kk = jj[improve]
            best_margin[kk] = margin[improve]
            best_distance_px[kk] = conservative_px[improve]
            witness_view[kk] = int(view)

    certified = best_margin > 0.0
    return {
        "certified": certified,
        "margin_normalized": best_margin.astype(np.float32),
        "conservative_projected_distance_px": best_distance_px.astype(np.float32),
        "witness_view": witness_view,
        "observed_view_count": observed_view_count,
    }


def source_exterior_metric_barrier_v4(
    sdf: torch.Tensor,
    margin_normalized: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if sdf.shape != margin_normalized.shape or sdf.numel() == 0:
        raise ValueError("source exterior sdf/margin must be matching non-empty tensors")
    if not torch.isfinite(sdf).all() or not torch.isfinite(margin_normalized).all():
        raise ValueError("source exterior barrier inputs must be finite")
    margin = margin_normalized.to(dtype=sdf.dtype)
    if torch.any(margin <= 0.0):
        raise ValueError("source exterior margin must be strictly positive")
    violation = F.relu(margin - sdf)
    return {
        "total": violation.mean(),
        "violating_fraction": (sdf < margin).to(dtype=sdf.dtype).mean(),
        "negative_fraction": (sdf <= 0.0).to(dtype=sdf.dtype).mean(),
        "minimum_sdf": torch.amin(sdf),
        "minimum_margin": torch.amin(margin),
        "maximum_margin": torch.amax(margin),
    }


def select_source_exterior_hard_negative_bank_v4(
    pool_index: np.ndarray,
    sdf: np.ndarray,
    margin_normalized: np.ndarray,
    *,
    bank_size: int,
) -> np.ndarray:
    index = np.asarray(pool_index, dtype=np.int64).reshape(-1)
    value = np.asarray(sdf, dtype=np.float64).reshape(-1)
    margin = np.asarray(margin_normalized, dtype=np.float64).reshape(-1)
    if index.shape != value.shape or index.shape != margin.shape or index.size == 0:
        raise ValueError("source exterior hard-negative inputs must match and be non-empty")
    if not np.isfinite(value).all() or not np.isfinite(margin).all():
        raise ValueError("source exterior hard-negative values must be finite")
    n = int(bank_size)
    if n <= 0:
        raise ValueError("bank_size must be positive")
    deficit = value - margin
    order = np.lexsort((index, deficit))
    return np.asarray(index[order[: min(n, index.size)]], dtype=np.int64)
