from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.stats import spearmanr

from .source_signed_field_v5 import SourceSignedFieldPolicyV5, source_signed_field_numpy_v5


def source_signed_field_per_view_numpy_v5(
    points_normalized: np.ndarray,
    *,
    signed_2d: np.ndarray,
    center_xyz: np.ndarray,
    normalization_half_extent: float,
    camera_origins: np.ndarray,
    camera_right: np.ndarray,
    camera_screen_up: np.ndarray,
    camera_half_extent: np.ndarray,
    policy: SourceSignedFieldPolicyV5 = SourceSignedFieldPolicyV5(),
) -> dict[str, np.ndarray]:
    """Decompose canonical F* into exact scaled per-view values for diagnosis only."""
    policy.validate()
    field = np.asarray(signed_2d, dtype=np.float32)
    views = int(policy.required_views)
    res = int(policy.required_resolution)
    expected = (views, res, res)
    if field.shape != expected or not np.isfinite(field).all():
        raise ValueError(f"signed_2d must be finite with shape {expected}")

    q = np.asarray(points_normalized, dtype=np.float32)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    if q.ndim != 2 or q.shape[1] != 3 or len(q) == 0 or not np.isfinite(q).all():
        raise ValueError("points_normalized must be finite non-empty [N,3]")

    center = np.asarray(center_xyz, dtype=np.float32).reshape(-1)
    origins = np.asarray(camera_origins, dtype=np.float32)
    right = np.asarray(camera_right, dtype=np.float32)
    up = np.asarray(camera_screen_up, dtype=np.float32)
    cam_half = np.asarray(camera_half_extent, dtype=np.float32).reshape(-1)
    norm_half = float(normalization_half_extent)
    if center.shape != (3,) or origins.shape != (views, 3) or right.shape != (views, 3) or up.shape != (views, 3):
        raise ValueError("camera arrays have invalid shapes")
    if cam_half.shape != (views,) or not np.isfinite(cam_half).all() or np.any(cam_half <= 0.0):
        raise ValueError("camera_half_extent must be finite positive [V]")
    if not math.isfinite(norm_half) or norm_half <= 0.0:
        raise ValueError("normalization_half_extent must be finite positive")

    pw = center[None, :] + q * np.float32(norm_half)
    values = np.full((len(q), views), -np.inf, dtype=np.float32)
    valid = np.zeros((len(q), views), dtype=bool)

    for view in range(views):
        d = pw - origins[view][None, :]
        gx = (d @ right[view]) / cam_half[view]
        gy = -(d @ up[view]) / cam_half[view]
        sx = (gx + 1.0) * np.float32(0.5 * res)
        sy = (gy + 1.0) * np.float32(0.5 * res)
        ok = (sx >= 0.0) & (sx <= res) & (sy >= 0.0) & (sy <= res)
        if not np.any(ok):
            continue
        ids = np.flatnonzero(ok)
        sampled = map_coordinates(
            field[view],
            np.stack([sy[ids] - 0.5, sx[ids] - 0.5], axis=0),
            order=1,
            mode="nearest",
            prefilter=False,
        ).astype(np.float32)
        scale = np.float32(2.0 * float(cam_half[view]) / (float(res) * norm_half))
        values[ids, view] = sampled * scale
        valid[ids, view] = True

    seen = valid.any(axis=1)
    aggregate = np.max(values, axis=1)
    aggregate[~seen] = np.float32(policy.unseen_value)
    aggregate_clamped = np.clip(
        aggregate, -float(policy.field_clamp), float(policy.field_clamp)
    ).astype(np.float32)

    canonical = source_signed_field_numpy_v5(
        q,
        signed_2d=field,
        center_xyz=center,
        normalization_half_extent=norm_half,
        camera_origins=origins,
        camera_right=right,
        camera_screen_up=up,
        camera_half_extent=cam_half,
        policy=policy,
    )
    if not np.allclose(aggregate_clamped, canonical, rtol=0.0, atol=2e-7):
        raise RuntimeError("per-view decomposition does not reproduce canonical F*")

    return {
        "per_view_unclamped": values,
        "valid": valid,
        "aggregate_unclamped": aggregate.astype(np.float32),
        "aggregate_clamped": aggregate_clamped,
    }


def top_two_view_margin_v5(
    per_view_unclamped: np.ndarray,
    valid: np.ndarray,
) -> dict[str, np.ndarray]:
    values = np.asarray(per_view_unclamped, dtype=np.float32)
    ok = np.asarray(valid, dtype=bool)
    if values.ndim != 2 or values.shape != ok.shape or values.shape[1] < 2:
        raise ValueError("per-view values/valid must align as [N,V], V>=2")
    if np.any(ok & ~np.isfinite(values)):
        raise ValueError("valid per-view entries must be finite")

    masked = np.where(ok, values, -np.inf)
    order = np.argsort(masked, axis=1)
    top1_idx = order[:, -1]
    top2_idx = order[:, -2]
    row = np.arange(len(values))
    top1 = masked[row, top1_idx]
    top2 = masked[row, top2_idx]
    valid_count = ok.sum(axis=1)
    has_two = valid_count >= 2
    margin = np.full(len(values), np.inf, dtype=np.float32)
    margin[has_two] = top1[has_two] - top2[has_two]

    return {
        "top1_view": top1_idx.astype(np.int16),
        "top2_view": top2_idx.astype(np.int16),
        "top1_value": top1.astype(np.float32),
        "top2_value": top2.astype(np.float32),
        "margin": margin,
        "valid_view_count": valid_count.astype(np.int16),
        "has_two_views": has_two,
    }


def finite_difference_gradient_numpy_v5(
    points_normalized: np.ndarray,
    query_fn: Callable[[np.ndarray], np.ndarray],
    *,
    epsilon: float,
) -> np.ndarray:
    q = np.asarray(points_normalized, dtype=np.float32)
    if q.ndim != 2 or q.shape[1] != 3 or len(q) == 0 or not np.isfinite(q).all():
        raise ValueError("points_normalized must be finite non-empty [N,3]")
    eps = float(epsilon)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("epsilon must be finite positive")

    grad = np.empty((len(q), 3), dtype=np.float32)
    for axis in range(3):
        qp = q.copy()
        qm = q.copy()
        qp[:, axis] += np.float32(eps)
        qm[:, axis] -= np.float32(eps)
        fp = np.asarray(query_fn(qp), dtype=np.float32).reshape(-1)
        fm = np.asarray(query_fn(qm), dtype=np.float32).reshape(-1)
        if fp.shape != (len(q),) or fm.shape != (len(q),) or not np.isfinite(fp).all() or not np.isfinite(fm).all():
            raise ValueError("query_fn must return finite [N] values")
        grad[:, axis] = (fp - fm) / np.float32(2.0 * eps)
    return grad


def zero_set_conditioning_proxy_v5(
    field_values: np.ndarray,
    gradient_xyz: np.ndarray,
    *,
    pixel_scale_normalized_units: float | None = None,
    gradient_floor: float = 1e-8,
) -> dict[str, np.ndarray]:
    f = np.asarray(field_values, dtype=np.float64).reshape(-1)
    g = np.asarray(gradient_xyz, dtype=np.float64)
    if g.shape != (len(f), 3) or len(f) == 0 or not np.isfinite(f).all() or not np.isfinite(g).all():
        raise ValueError("field_values [N] and finite gradient_xyz [N,3] required")
    floor = float(gradient_floor)
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("gradient_floor must be finite positive")

    norm = np.linalg.norm(g, axis=1)
    stable = norm > floor
    offset = np.full(len(f), np.inf, dtype=np.float64)
    offset[stable] = np.abs(f[stable]) / norm[stable]

    out = {
        "gradient_norm": norm.astype(np.float32),
        "gradient_stable": stable,
        "offset_proxy_normalized": offset.astype(np.float32),
    }
    if pixel_scale_normalized_units is not None:
        px = float(pixel_scale_normalized_units)
        if not math.isfinite(px) or px <= 0.0:
            raise ValueError("pixel_scale_normalized_units must be finite positive")
        out["offset_proxy_pixels"] = (offset / px).astype(np.float32)
    return out


def replay_staleness_report_v5(
    base_residual: np.ndarray,
    current_residual: np.ndarray,
    *,
    bank_size: int,
) -> dict[str, float | int]:
    base = np.asarray(base_residual, dtype=np.float64).reshape(-1)
    cur = np.asarray(current_residual, dtype=np.float64).reshape(-1)
    if base.shape != cur.shape or len(base) == 0 or not np.isfinite(base).all() or not np.isfinite(cur).all():
        raise ValueError("base/current residuals must be finite aligned non-empty vectors")
    k = int(bank_size)
    if k <= 0 or k > len(base):
        raise ValueError("bank_size must be in [1,N]")

    def hard_indices(x: np.ndarray) -> np.ndarray:
        if k == len(x):
            ids = np.arange(k, dtype=np.int64)
        else:
            ids = np.argpartition(x, -k)[-k:].astype(np.int64)
        order = np.lexsort((ids, -x[ids]))
        return ids[order]

    frozen = hard_indices(base)
    current = hard_indices(cur)
    overlap = np.intersect1d(frozen, current, assume_unique=False)
    frozen_mask = np.zeros(len(base), dtype=bool)
    frozen_mask[frozen] = True
    current_captured = int(np.count_nonzero(frozen_mask[current]))
    rho = spearmanr(base, cur).statistic
    if not np.isfinite(rho):
        rho = 0.0
    return {
        "pool_count": int(len(base)),
        "bank_size": k,
        "hard_set_overlap_count": int(len(overlap)),
        "hard_set_overlap_fraction": float(len(overlap) / k),
        "current_hard_set_captured_by_frozen_bank_count": current_captured,
        "current_hard_set_captured_by_frozen_bank_fraction": float(current_captured / k),
        "base_current_residual_spearman": float(rho),
        "frozen_bank_current_residual_mean": float(cur[frozen].mean()),
        "current_hard_residual_mean": float(cur[current].mean()),
        "pool_current_residual_mean": float(cur.mean()),
    }
