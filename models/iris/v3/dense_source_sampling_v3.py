from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class DenseSourceSamplingPolicyV3:
    foreground_component_rays_per_component_per_view: int = 8
    additional_global_foreground_boundary_rays_per_view: int = 16
    additional_global_foreground_interior_rays_per_view: int = 16
    background_rays_per_view: int = 32

    def validate(self) -> None:
        values = (
            self.foreground_component_rays_per_component_per_view,
            self.additional_global_foreground_boundary_rays_per_view,
            self.additional_global_foreground_interior_rays_per_view,
            self.background_rays_per_view,
        )
        if any(int(value) < 0 for value in values):
            raise ValueError("dense source sampling quotas must be non-negative")
        if int(self.foreground_component_rays_per_component_per_view) <= 0:
            raise ValueError("every foreground component requires a positive ray quota")
        if int(self.background_rays_per_view) <= 0:
            raise ValueError("background ray quota must be positive")


@dataclass(frozen=True)
class DenseSourceViewSamplingStateV3:
    width: int
    height: int
    foreground: np.ndarray
    admitted: np.ndarray
    component_id: np.ndarray
    boundary: np.ndarray
    interior: np.ndarray
    component_count: int


@dataclass(frozen=True)
class DenseSourceRaySelectionV3:
    flat_pixel_index: np.ndarray
    target_foreground: np.ndarray
    component_id: np.ndarray
    boundary: np.ndarray

    @property
    def ray_count(self) -> int:
        return int(self.flat_pixel_index.size)


def deterministic_view_seed(fit_seed: int, training_step: int, view_index: int) -> int:
    payload = f"{int(fit_seed)}:{int(training_step)}:{int(view_index)}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _ray_box_valid_numpy(
    origins: np.ndarray,
    directions: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> np.ndarray:
    o = np.asarray(origins, dtype=np.float64)
    d = np.asarray(directions, dtype=np.float64)
    if o.ndim != 2 or o.shape[1] != 3 or d.shape != o.shape:
        raise ValueError("origins/directions must both be [N,3]")
    if not np.isfinite(o).all() or not np.isfinite(d).all():
        raise ValueError("origins/directions must be finite")
    norm = np.linalg.norm(d, axis=1)
    if np.any(norm <= float(epsilon)):
        raise ValueError("ray directions must be nonzero")
    d = d / norm[:, None]
    moving = np.abs(d) > float(epsilon)
    parallel_outside = (~moving) & ((o < -1.0) | (o > 1.0))
    safe_d = np.where(moving, d, 1.0)
    t1 = (-1.0 - o) / safe_d
    t2 = (1.0 - o) / safe_d
    near = np.where(moving, np.minimum(t1, t2), -np.inf)
    far = np.where(moving, np.maximum(t1, t2), np.inf)
    entry = np.maximum(np.max(near, axis=1), 0.0)
    exit = np.min(far, axis=1)
    return (~np.any(parallel_outside, axis=1)) & np.isfinite(entry) & np.isfinite(exit) & (exit >= entry)


def camera_pixel_ray_origins_normalized(
    flat_pixel_index: np.ndarray,
    *,
    width: int,
    height: int,
    camera_origin_normalized: np.ndarray,
    camera_right: np.ndarray,
    camera_screen_up: np.ndarray,
    camera_forward: np.ndarray,
    camera_half_extent_normalized: float,
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.asarray(flat_pixel_index, dtype=np.int64).reshape(-1)
    w = int(width)
    h = int(height)
    if w <= 0 or h <= 0:
        raise ValueError("width/height must be positive")
    if np.any(idx < 0) or np.any(idx >= w * h):
        raise ValueError("flat pixel index out of range")
    origin = np.asarray(camera_origin_normalized, dtype=np.float64).reshape(3)
    right = np.asarray(camera_right, dtype=np.float64).reshape(3)
    up = np.asarray(camera_screen_up, dtype=np.float64).reshape(3)
    forward = np.asarray(camera_forward, dtype=np.float64).reshape(3)
    half_extent = float(camera_half_extent_normalized)
    if not math.isfinite(half_extent) or half_extent <= 0.0:
        raise ValueError("camera_half_extent_normalized must be finite and positive")
    if not np.isfinite(np.concatenate([origin, right, up, forward])).all():
        raise ValueError("camera vectors must be finite")
    if np.linalg.norm(forward) <= 1e-12:
        raise ValueError("camera forward must be nonzero")

    x = idx % w
    y = idx // w
    gx = 2.0 * (x.astype(np.float64) + 0.5) / float(w) - 1.0
    gy = 2.0 * (y.astype(np.float64) + 0.5) / float(h) - 1.0
    origins = (
        origin[None, :]
        + gx[:, None] * half_extent * right[None, :]
        - gy[:, None] * half_extent * up[None, :]
    )
    directions = np.repeat((forward / np.linalg.norm(forward))[None, :], len(idx), axis=0)
    return origins.astype(np.float32), directions.astype(np.float32)


def admitted_pixel_mask_for_camera(
    *,
    width: int,
    height: int,
    camera_origin_normalized: np.ndarray,
    camera_right: np.ndarray,
    camera_screen_up: np.ndarray,
    camera_forward: np.ndarray,
    camera_half_extent_normalized: float,
    chunk_pixels: int = 131072,
) -> np.ndarray:
    w = int(width)
    h = int(height)
    n = w * h
    if int(chunk_pixels) <= 0:
        raise ValueError("chunk_pixels must be positive")
    admitted = np.zeros(n, dtype=bool)
    for start in range(0, n, int(chunk_pixels)):
        stop = min(n, start + int(chunk_pixels))
        index = np.arange(start, stop, dtype=np.int64)
        origins, directions = camera_pixel_ray_origins_normalized(
            index,
            width=w,
            height=h,
            camera_origin_normalized=camera_origin_normalized,
            camera_right=camera_right,
            camera_screen_up=camera_screen_up,
            camera_forward=camera_forward,
            camera_half_extent_normalized=camera_half_extent_normalized,
        )
        admitted[start:stop] = _ray_box_valid_numpy(origins, directions)
    return admitted.reshape(h, w)


def build_dense_source_view_sampling_state(
    foreground_mask: np.ndarray,
    admitted_mask: np.ndarray,
) -> DenseSourceViewSamplingStateV3:
    foreground = np.asarray(foreground_mask, dtype=bool)
    admitted = np.asarray(admitted_mask, dtype=bool)
    if foreground.ndim != 2 or admitted.shape != foreground.shape:
        raise ValueError("foreground/admitted masks must be matching [H,W]")
    if foreground.size == 0 or not np.any(foreground):
        raise ValueError("foreground mask must be non-empty")
    if not np.any(admitted):
        raise ValueError("camera has no rays intersecting the normalization cube")
    if np.any(foreground & ~admitted):
        raise ValueError("SOURCE_FOREGROUND_OUTSIDE_NORMALIZATION_RAY_DOMAIN")

    four_connected = np.asarray(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        dtype=np.uint8,
    )
    labels, component_count = ndimage.label(foreground, structure=four_connected)
    component_id = np.where(foreground, labels.astype(np.int64) - 1, -1)
    interior = ndimage.binary_erosion(
        foreground,
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
    )
    boundary = foreground & ~interior
    return DenseSourceViewSamplingStateV3(
        width=int(foreground.shape[1]),
        height=int(foreground.shape[0]),
        foreground=foreground,
        admitted=admitted,
        component_id=component_id,
        boundary=boundary,
        interior=interior,
        component_count=int(component_count),
    )


def _sample_flat_pool(
    rng: np.random.Generator,
    pool: np.ndarray,
    count: int,
) -> np.ndarray:
    values = np.asarray(pool, dtype=np.int64).reshape(-1)
    n = int(count)
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    if values.size == 0:
        raise ValueError("required dense source sampling pool is empty")
    replace = bool(values.size < n)
    return np.asarray(rng.choice(values, size=n, replace=replace), dtype=np.int64)


def sample_dense_source_view_rays(
    state: DenseSourceViewSamplingStateV3,
    *,
    fit_seed: int,
    training_step: int,
    view_index: int,
    policy: DenseSourceSamplingPolicyV3 = DenseSourceSamplingPolicyV3(),
) -> DenseSourceRaySelectionV3:
    policy.validate()
    rng = np.random.default_rng(
        deterministic_view_seed(fit_seed, training_step, view_index)
    )
    flat_component = state.component_id.reshape(-1)
    flat_foreground = state.foreground.reshape(-1)
    flat_admitted = state.admitted.reshape(-1)
    flat_boundary = state.boundary.reshape(-1)
    flat_interior = state.interior.reshape(-1)

    selected: list[np.ndarray] = []
    for component in range(int(state.component_count)):
        pool = np.flatnonzero(
            (flat_component == component) & flat_admitted
        )
        if pool.size == 0:
            raise ValueError(f"FOREGROUND_COMPONENT_HAS_NO_ADMITTED_RAY:{component}")
        selected.append(
            _sample_flat_pool(
                rng,
                pool,
                policy.foreground_component_rays_per_component_per_view,
            )
        )

    boundary_pool = np.flatnonzero(flat_boundary & flat_admitted)
    interior_pool = np.flatnonzero(flat_interior & flat_admitted)
    boundary_quota = int(policy.additional_global_foreground_boundary_rays_per_view)
    interior_quota = int(policy.additional_global_foreground_interior_rays_per_view)
    if interior_quota > 0 and interior_pool.size == 0:
        boundary_quota += interior_quota
        interior_quota = 0
    selected.append(_sample_flat_pool(rng, boundary_pool, boundary_quota))
    if interior_quota:
        selected.append(_sample_flat_pool(rng, interior_pool, interior_quota))

    background_pool = np.flatnonzero((~flat_foreground) & flat_admitted)
    selected.append(
        _sample_flat_pool(rng, background_pool, policy.background_rays_per_view)
    )
    index = np.concatenate(selected).astype(np.int64, copy=False)
    target = flat_foreground[index].astype(np.float32)
    component = flat_component[index].astype(np.int64)
    boundary = flat_boundary[index].astype(np.float32)
    if np.any((target > 0.5) & (component < 0)) or np.any((target < 0.5) & (component >= 0)):
        raise RuntimeError("DENSE_SOURCE_SELECTION_COMPONENT_TARGET_DRIFT")
    present = set(map(int, component[component >= 0]))
    if present != set(range(int(state.component_count))):
        raise RuntimeError("DENSE_SOURCE_SELECTION_DROPPED_FOREGROUND_COMPONENT")
    return DenseSourceRaySelectionV3(
        flat_pixel_index=index,
        target_foreground=target,
        component_id=component,
        boundary=boundary,
    )
