from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

import numpy as np


class SurfaceState(str, Enum):
    SURFACE_SUPPORTED = "SURFACE_SUPPORTED"
    SURFACE_AMBIGUOUS = "SURFACE_AMBIGUOUS"
    UNKNOWN_OCCLUDED = "UNKNOWN_OCCLUDED"
    UNKNOWN_UNOBSERVED = "UNKNOWN_UNOBSERVED"
    OUTSIDE_ADMISSIBLE_DOMAIN = "OUTSIDE_ADMISSIBLE_DOMAIN"
    RESOLUTION_UNSUPPORTED = "RESOLUTION_UNSUPPORTED"


@dataclass(frozen=True)
class OrthoCamera:
    forward: np.ndarray
    right: np.ndarray
    screen_up: np.ndarray
    half_extent: float
    yaw_deg: float

    @staticmethod
    def from_dict(o: dict) -> "OrthoCamera":
        if o.get("contract") != "realsas.level_orthographic_z_orbit.v1":
            raise ValueError("CAMERA_CONTRACT_DRIFT")
        c = OrthoCamera(
            forward=np.asarray(o["forward"], dtype=np.float64),
            right=np.asarray(o["right"], dtype=np.float64),
            screen_up=np.asarray(o["screen_up"], dtype=np.float64),
            half_extent=float(o["half_extent"]),
            yaw_deg=float(o["yaw_deg"]),
        )
        c.validate()
        return c

    def validate(self, atol: float = 1e-8) -> None:
        if self.half_extent <= 0:
            raise ValueError("CAMERA_HALF_EXTENT_NONPOSITIVE")
        for name, v in (("forward", self.forward), ("right", self.right), ("screen_up", self.screen_up)):
            if v.shape != (3,):
                raise ValueError(f"CAMERA_{name.upper()}_SHAPE")
            if not np.isfinite(v).all():
                raise ValueError(f"CAMERA_{name.upper()}_NONFINITE")
            if abs(float(np.linalg.norm(v)) - 1.0) > atol:
                raise ValueError(f"CAMERA_{name.upper()}_NOT_UNIT")
        basis = np.stack([self.right, self.screen_up, self.forward], axis=0)
        gram = basis @ basis.T
        if not np.allclose(gram, np.eye(3), atol=atol, rtol=0):
            raise ValueError("CAMERA_BASIS_NOT_ORTHONORMAL")


def orbit_camera(yaw_deg: float, half_extent: float = 0.54) -> OrthoCamera:
    """Exact level orthographic Z-orbit camera used by Gate-0 synthetic tests."""
    t = np.deg2rad(float(yaw_deg))
    forward = np.asarray([np.sin(t), np.cos(t), 0.0], dtype=np.float64)
    right = np.asarray([np.cos(t), -np.sin(t), 0.0], dtype=np.float64)
    screen_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    c = OrthoCamera(forward=forward, right=right, screen_up=screen_up, half_extent=float(half_extent), yaw_deg=float(yaw_deg))
    c.validate()
    return c


def project_world(points_xyz: np.ndarray, camera: OrthoCamera) -> tuple[np.ndarray, np.ndarray]:
    """Project canonical world points to grid_sample-style normalized xy and forward depth.

    Grid x=-1/1 are left/right. Grid y=-1/1 are top/bottom, therefore world screen_up is negated.
    """
    p = np.asarray(points_xyz, dtype=np.float64)
    if p.shape[-1] != 3:
        raise ValueError("POINT_SHAPE")
    gx = np.tensordot(p, camera.right, axes=([-1], [0])) / camera.half_extent
    gy = -np.tensordot(p, camera.screen_up, axes=([-1], [0])) / camera.half_extent
    d = np.tensordot(p, camera.forward, axes=([-1], [0]))
    return np.stack([gx, gy], axis=-1), d


def backproject_grid_depth(grid_xy: np.ndarray, depth: np.ndarray, camera: OrthoCamera) -> np.ndarray:
    g = np.asarray(grid_xy, dtype=np.float64)
    d = np.asarray(depth, dtype=np.float64)
    if g.shape[-1] != 2 or g.shape[:-1] != d.shape:
        raise ValueError("GRID_DEPTH_SHAPE")
    x = g[..., 0] * camera.half_extent
    up = -g[..., 1] * camera.half_extent
    return x[..., None] * camera.right + up[..., None] * camera.screen_up + d[..., None] * camera.forward


def grid_to_pixel(grid_xy: np.ndarray, resolution: int = 1024) -> np.ndarray:
    """Nearest pixel index for boolean support tests, align_corners=False convention."""
    g = np.asarray(grid_xy, dtype=np.float64)
    if g.shape[-1] != 2:
        raise ValueError("GRID_SHAPE")
    px = np.floor((g[..., 0] + 1.0) * 0.5 * resolution).astype(np.int64)
    py = np.floor((g[..., 1] + 1.0) * 0.5 * resolution).astype(np.int64)
    return np.stack([px, py], axis=-1)


def make_dense_lattice(bounds_xyz: Sequence[Sequence[float]], spacing: float) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Return points in canonical [Z,Y,X] traversal while coordinates remain XYZ."""
    b = np.asarray(bounds_xyz, dtype=np.float64)
    if b.shape != (3, 2) or not np.all(b[:, 1] > b[:, 0]) or spacing <= 0:
        raise ValueError("LATTICE_BOUNDS_OR_SPACING")
    axes = [np.arange(lo + spacing * 0.5, hi, spacing, dtype=np.float64) for lo, hi in b]
    x, y, z = axes
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    pts = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    return pts, (len(z), len(y), len(x))


def dilate_binary(mask: np.ndarray, pad_px: int) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2 or pad_px < 0:
        raise ValueError("MASK_OR_PADDING")
    if pad_px == 0:
        return m.copy()
    h, w = m.shape
    out = np.zeros_like(m)
    r = int(pad_px)
    # Square dilation is intentionally conservative for a candidate-domain constraint.
    for dy in range(-r, r + 1):
        ys = slice(max(0, dy), min(h, h + dy))
        yt = slice(max(0, -dy), min(h, h - dy))
        for dx in range(-r, r + 1):
            xs = slice(max(0, dx), min(w, w + dx))
            xt = slice(max(0, -dx), min(w, w - dx))
            out[yt, xt] |= m[ys, xs]
    return out


def visual_hull_membership(
    points_xyz: np.ndarray,
    cameras: Sequence[OrthoCamera],
    foreground_masks: Sequence[np.ndarray],
    pad_px: int = 0,
) -> np.ndarray:
    p = np.asarray(points_xyz, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("POINT_MATRIX_SHAPE")
    if len(cameras) != len(foreground_masks) or not cameras:
        raise ValueError("VIEW_COUNT")
    keep = np.ones(len(p), dtype=bool)
    for cam, raw_mask in zip(cameras, foreground_masks):
        mask = dilate_binary(raw_mask, int(pad_px))
        g, _ = project_world(p, cam)
        pix = grid_to_pixel(g, mask.shape[1])
        x, y = pix[:, 0], pix[:, 1]
        inside = (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
        ok = np.zeros(len(p), dtype=bool)
        ids = np.flatnonzero(inside)
        ok[ids] = mask[y[ids], x[ids]]
        keep &= ok
        if not keep.any():
            break
    return keep


def hull_frontier(
    truth_points_xyz: np.ndarray,
    lattice_points_xyz: np.ndarray,
    cameras: Sequence[OrthoCamera],
    foreground_masks: Sequence[np.ndarray],
    paddings_px: Iterable[int] = (0, 1, 2, 4, 8),
) -> list[dict]:
    truth = np.asarray(truth_points_xyz, dtype=np.float64)
    lattice = np.asarray(lattice_points_xyz, dtype=np.float64)
    out = []
    for pad in paddings_px:
        tm = visual_hull_membership(truth, cameras, foreground_masks, int(pad))
        lm = visual_hull_membership(lattice, cameras, foreground_masks, int(pad))
        out.append({
            "pad_px": int(pad),
            "truth_containment": float(tm.mean()) if len(tm) else 1.0,
            "active_lattice_fraction": float(lm.mean()) if len(lm) else 0.0,
            "active_lattice_count": int(lm.sum()),
            "total_lattice_count": int(len(lm)),
        })
    return out


def thickness_cell_strata(thickness_world: np.ndarray, spacing: float) -> dict:
    t = np.asarray(thickness_world, dtype=np.float64).reshape(-1)
    if spacing <= 0 or (t < 0).any():
        raise ValueError("THICKNESS_OR_SPACING")
    c = t / float(spacing)
    bins = {
        "lt_1": int(np.sum(c < 1.0)),
        "ge_1_lt_2": int(np.sum((c >= 1.0) & (c < 2.0))),
        "ge_2_lt_4": int(np.sum((c >= 2.0) & (c < 4.0))),
        "ge_4": int(np.sum(c >= 4.0)),
    }
    return {
        "spacing": float(spacing),
        "count": int(c.size),
        "min_cells": float(c.min()) if c.size else None,
        "p50_cells": float(np.median(c)) if c.size else None,
        "bins": bins,
    }


def robust_multiview_evidence(descriptors: np.ndarray, valid: np.ndarray | None = None, trim_fraction: float = 0.25) -> dict:
    """Deterministic compact evidence; no learned visibility authority.

    descriptors: [V,C]. Returns permutation-invariant support statistics. A minority of view outliers is trimmed.
    """
    x = np.asarray(descriptors, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("DESCRIPTOR_SHAPE")
    if valid is None:
        valid = np.ones(x.shape[0], dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if valid.shape != (x.shape[0],):
        raise ValueError("VALID_SHAPE")
    xv = x[valid]
    if len(xv) < 2:
        return {"support_views": int(len(xv)), "pair_count": 0, "cosine_trimmed_mean": None, "cosine_median": None, "dispersion": None}
    norm = np.linalg.norm(xv, axis=1, keepdims=True)
    xn = xv / np.maximum(norm, 1e-12)
    sim = xn @ xn.T
    iu = np.triu_indices(len(xn), k=1)
    vals = np.sort(sim[iu])
    k = int(np.floor(len(vals) * float(trim_fraction)))
    core = vals[k: len(vals) - k] if k > 0 and 2 * k < len(vals) else vals
    center = np.median(xv, axis=0)
    dispersion = np.median(np.linalg.norm(xv - center[None, :], axis=1))
    return {
        "support_views": int(len(xv)),
        "pair_count": int(len(vals)),
        "cosine_trimmed_mean": float(core.mean()),
        "cosine_median": float(np.median(vals)),
        "dispersion": float(dispersion),
    }


def streaming_chunk_plan(active_candidates: int, views: int, descriptor_dim: int, max_working_bytes: int, bytes_per_scalar: int = 4) -> dict:
    if min(active_candidates, views, descriptor_dim, max_working_bytes, bytes_per_scalar) <= 0:
        raise ValueError("CHUNK_PLAN_ARGUMENT")
    per_candidate = int(views) * int(descriptor_dim) * int(bytes_per_scalar)
    chunk = max(1, min(int(active_candidates), int(max_working_bytes) // max(1, per_candidate)))
    chunks = (int(active_candidates) + chunk - 1) // chunk
    return {
        "active_candidates": int(active_candidates),
        "per_candidate_raw_descriptor_bytes": int(per_candidate),
        "chunk_candidates": int(chunk),
        "chunk_count": int(chunks),
        "max_raw_descriptor_working_bytes": int(chunk * per_candidate),
    }
