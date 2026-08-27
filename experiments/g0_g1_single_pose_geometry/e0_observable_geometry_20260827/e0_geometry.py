#!/usr/bin/env python3
"""RealSaS E0 observable-geometry / persistence apparatus.

E0 premise: view-local visible common-frame P is exact.  The learner is absent.

E0-a may use teacher surface provenance to establish an oracle persistence upper bound.
E0-b receives only ObservableView: P, P-derived normal, pixel provenance and known cameras.
Teacher triangle/barycentric authority is intentionally absent from the E0-b API.

Coordinate conventions are copied semantically from the frozen P-V5/R256 FIT scale ladder:
  * native 1024 raster authority
  * align_corners=False pixel-center grid coordinates
  * yaw=0: right=(1,0,0), forward=(0,1,0), up=(0,0,1)
  * positive yaw rotates around +Z
  * orthographic half-extent is recovered from exact observable P + raster pixel provenance;
    camera.json is not consumed. 0.54 is only the historical default/test value.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np

VIEWS = 8
NATIVE_RESOLUTION = 1024
RENDER_HALF_EXTENT = 0.54


def pixel_center_to_grid(pixel_xy, resolution: int):
    p = np.asarray(pixel_xy, np.float32)
    return (2.0 * (p + 0.5) / float(resolution) - 1.0).astype(np.float32)


def grid_to_pixel_center(grid_xy, resolution: int):
    g = np.asarray(grid_xy, np.float32)
    return ((g + 1.0) * (float(resolution) / 2.0) - 0.5).astype(np.float32)


def grid_distance_in_pixels(a, b, resolution: int):
    return np.linalg.norm(grid_to_pixel_center(a, resolution) - grid_to_pixel_center(b, resolution), axis=-1)


def camera_for_view(view: int, half_extent: float = RENDER_HALF_EXTENT) -> dict:
    if not 0 <= int(view) < VIEWS:
        raise ValueError(view)
    theta = math.radians(float(view) * 45.0)
    c, s = math.cos(theta), math.sin(theta)
    return {
        "view": int(view),
        "yaw_deg": float(view) * 45.0,
        "right": np.asarray([c, -s, 0.0], np.float32),
        "forward": np.asarray([s, c, 0.0], np.float32),
        "up": np.asarray([0.0, 0.0, 1.0], np.float32),
        "half_extent": float(half_extent),
    }


def estimate_half_extent_from_observable(
    view: int,
    pixel_linear_index,
    P,
    resolution: int,
    *,
    huber_k: float = 1.345,
    max_irls_steps: int = 3,
) -> tuple[float, dict]:
    """Recover orthographic half-extent from exact visible P and raster provenance.

    For each visible row, with pixel-center grid g and camera-frame surface coordinate q:
      q_x = dot(P,right) = h * g_x
      q_y = -dot(P,up)   = h * g_y

    Estimate the single slope h through the origin.  This avoids per-coordinate q/g ratios
    and therefore has no arbitrary |grid| cutoff.  Near-center observations naturally
    contribute negligible leverage through g^2.  A deterministic Huber IRLS pass provides
    outlier resistance; native reprojection remains the fail-closed authority.
    """
    pix = np.asarray(pixel_linear_index, np.int64)
    P = np.asarray(P, np.float32)
    if len(pix) != len(P):
        raise ValueError("pixel/P length mismatch")
    if len(pix) < 2:
        raise RuntimeError(f"insufficient visible rows for half-extent recovery: {len(pix)}")

    y = pix // int(resolution)
    x = pix % int(resolution)
    grid32 = pixel_center_to_grid(np.stack([x, y], axis=1), int(resolution))
    unit_cam = camera_for_view(int(view), half_extent=1.0)
    comp32 = np.stack([P @ unit_cam["right"], -(P @ unit_cam["up"])], axis=1)

    g = np.asarray(grid32, np.float64)
    q = np.asarray(comp32, np.float64)
    finite_rows = np.all(np.isfinite(g), axis=1) & np.all(np.isfinite(q), axis=1)
    g = g[finite_rows]
    q = q[finite_rows]
    if len(g) < 2:
        raise RuntimeError(f"insufficient finite rows for half-extent recovery: {len(g)}")

    row_g2 = np.sum(g * g, axis=1)
    information_energy = float(np.sum(row_g2))
    grid_rms = float(np.sqrt(np.mean(row_g2)))
    numerical_floor = float(np.finfo(np.float64).eps * max(1, len(g)) * 64.0)
    if not np.isfinite(information_energy) or information_energy <= numerical_floor:
        raise RuntimeError(
            f"insufficient camera-scale information energy: {information_energy} <= {numerical_floor}"
        )

    weights = np.ones(len(g), np.float64)
    h = float(np.sum(g * q) / information_energy)
    robust_scale = 0.0
    for _ in range(int(max_irls_steps)):
        residual_vec = q - h * g
        residual = np.linalg.norm(residual_vec, axis=1)
        med = float(np.median(residual))
        mad = float(np.median(np.abs(residual - med)))
        robust_scale = 1.4826 * mad
        if not np.isfinite(robust_scale) or robust_scale <= 1e-12:
            weights = np.ones(len(g), np.float64)
        else:
            cutoff = float(huber_k) * robust_scale
            weights = np.ones(len(g), np.float64)
            high = residual > cutoff
            weights[high] = cutoff / np.maximum(residual[high], 1e-30)
        denom = float(np.sum(weights * row_g2))
        if not np.isfinite(denom) or denom <= numerical_floor:
            raise RuntimeError(f"robust camera-scale information collapsed: {denom}")
        h_new = float(np.sum(weights[:, None] * g * q) / denom)
        if abs(h_new - h) <= 1e-12 * max(1.0, abs(h)):
            h = h_new
            break
        h = h_new

    if not (0.05 <= h <= 5.0):
        raise RuntimeError(f"implausible recovered half-extent: {h}")

    axis_energy = np.sum(g * g, axis=0)
    axis_h = [None, None]
    for a in range(2):
        if float(axis_energy[a]) > numerical_floor:
            axis_h[a] = float(np.sum(g[:, a] * q[:, a]) / axis_energy[a])
    if axis_h[0] is not None and axis_h[1] is not None:
        axis_relative_disagreement = abs(axis_h[0] - axis_h[1]) / max(abs(h), 1e-12)
    else:
        axis_relative_disagreement = None

    predicted = project_grid(P, camera_for_view(int(view), h))
    residual_px = grid_distance_in_pixels(predicted, grid32, int(resolution))
    fit_residual = q - h * g
    weighted_rmse = float(
        np.sqrt(np.sum(weights[:, None] * fit_residual * fit_residual) / max(2.0 * np.sum(weights), 1.0))
    )
    stats = {
        "mode": "observable_P_plus_pixel_grid_robust_origin_slope_v2",
        "visible_rows": int(len(pix)),
        "finite_rows": int(len(g)),
        "information_energy": information_energy,
        "grid_rms": grid_rms,
        "effective_weight_sum": float(np.sum(weights)),
        "robust_scale": float(robust_scale),
        "weighted_fit_rmse": weighted_rmse,
        "half_extent": h,
        "half_extent_x": axis_h[0],
        "half_extent_y": axis_h[1],
        "axis_relative_disagreement": axis_relative_disagreement,
        "reprojection_px_p50": float(np.quantile(residual_px, 0.50)),
        "reprojection_px_p95": float(np.quantile(residual_px, 0.95)),
        "reprojection_px_p99": float(np.quantile(residual_px, 0.99)),
    }
    if stats["reprojection_px_p95"] > 1.25:
        raise RuntimeError(f"observable camera recovery residual too large: {stats}")
    if axis_relative_disagreement is not None and axis_relative_disagreement > 0.01:
        raise RuntimeError(f"observable camera recovery axis disagreement too large: {stats}")
    return h, stats


def project_grid(points, camera):
    p = np.asarray(points, np.float32)
    right = np.asarray(camera["right"], np.float32)
    up = np.asarray(camera["up"], np.float32)
    he = float(camera["half_extent"])
    return np.stack([(p @ right) / he, -(p @ up) / he], axis=-1).astype(np.float32)


def grid_to_nearest_pixel(grid, resolution: int):
    g = np.asarray(grid, np.float32)
    x = np.floor((g[..., 0] + 1.0) * 0.5 * resolution).astype(np.int64)
    y = np.floor((g[..., 1] + 1.0) * 0.5 * resolution).astype(np.int64)
    return x, y


def geometric_vertex_normals(vertices, faces):
    v = np.asarray(vertices, np.float32)
    f = np.asarray(faces, np.int64)
    tri = v[f]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]).astype(np.float32)
    out = np.zeros_like(v, dtype=np.float32)
    for j in range(3):
        np.add.at(out, f[:, j], fn)
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-8)
    return out


def bary_weights(uv):
    uv = np.asarray(uv, np.float32)
    return np.stack([uv[..., 0], uv[..., 1], 1.0 - uv[..., 0] - uv[..., 1]], axis=-1)


def reconstruct_surface(vertices, faces, vertex_normals, triangle_id, barycentric_uv):
    tri = np.asarray(triangle_id, np.int64)
    w = bary_weights(barycentric_uv)
    p = (vertices[faces[tri]] * w[..., None]).sum(axis=-2).astype(np.float32)
    n = (vertex_normals[faces[tri]] * w[..., None]).sum(axis=-2).astype(np.float32)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8)
    return p, n


def raster_lookup_near(pixel_linear_index, query_x, query_y, resolution: int, radius: int):
    pix = np.asarray(pixel_linear_index, np.int64)
    qx = np.asarray(query_x, np.int64).reshape(-1)
    qy = np.asarray(query_y, np.int64).reshape(-1)
    if len(pix) and np.any(pix[1:] < pix[:-1]):
        raise ValueError("pixel_linear_index must be sorted")
    out = np.full((len(qx), (2 * radius + 1) ** 2), -1, np.int64)
    col = 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            xx = qx + dx
            yy = qy + dy
            valid = (xx >= 0) & (xx < resolution) & (yy >= 0) & (yy < resolution)
            target = yy * resolution + xx
            pos = np.searchsorted(pix, target)
            if len(pix):
                pc = np.minimum(pos, len(pix) - 1)
                hit = valid & (pos < len(pix)) & (pix[pc] == target)
                out[hit, col] = pos[hit]
            col += 1
    return out


def _row_lookup(pix: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Sorted pixel-id lookup. Missing targets return -1."""
    pos = np.searchsorted(pix, targets)
    out = np.full(targets.shape, -1, np.int64)
    if len(pix):
        pc = np.minimum(pos, len(pix) - 1)
        hit = (pos < len(pix)) & (pix[pc] == targets)
        out[hit] = pos[hit]
    return out


def derive_view_local_normals(pixel_linear_index, P, resolution: int, stride_px: int = 2):
    """Deterministic normal from visible P only; no mesh normal or triangle identity.

    A tangent is built from two-sided image-grid neighbors when available, otherwise one-sided.
    Normal sign is intentionally not made authoritative; E0-b uses sign-invariant continuity.
    """
    pix = np.asarray(pixel_linear_index, np.int64)
    P = np.asarray(P, np.float32)
    if len(pix) != len(P):
        raise ValueError("pix/P length mismatch")
    if len(pix) and np.any(pix[1:] < pix[:-1]):
        raise ValueError("pixel ids must be sorted")
    y = pix // resolution
    x = pix % resolution
    cur = np.arange(len(pix), dtype=np.int64)

    def lookup(dx, dy):
        xx, yy = x + dx, y + dy
        valid = (xx >= 0) & (xx < resolution) & (yy >= 0) & (yy < resolution)
        q = yy * resolution + xx
        r = _row_lookup(pix, q)
        r[~valid] = -1
        return r

    l = lookup(-stride_px, 0); r = lookup(stride_px, 0)
    u = lookup(0, -stride_px); d = lookup(0, stride_px)
    tx = np.zeros_like(P); ty = np.zeros_like(P)
    vx = np.zeros(len(P), bool); vy = np.zeros(len(P), bool)

    both = (l >= 0) & (r >= 0)
    tx[both] = P[r[both]] - P[l[both]]; vx |= both
    only = (~both) & (r >= 0)
    tx[only] = P[r[only]] - P[cur[only]]; vx |= only
    only = (~both) & (l >= 0)
    tx[only] = P[cur[only]] - P[l[only]]; vx |= only

    both = (u >= 0) & (d >= 0)
    ty[both] = P[d[both]] - P[u[both]]; vy |= both
    only = (~both) & (d >= 0)
    ty[only] = P[d[only]] - P[cur[only]]; vy |= only
    only = (~both) & (u >= 0)
    ty[only] = P[cur[only]] - P[u[only]]; vy |= only

    n = np.cross(tx, ty).astype(np.float32)
    nn = np.linalg.norm(n, axis=1)
    good = vx & vy & (nn > 1e-9)
    n[good] /= nn[good, None]
    n[~good] = 0.0
    return n, good
