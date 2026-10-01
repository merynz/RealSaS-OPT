from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np
from scipy import ndimage
import torch

from models.iris.v3.dense_source_coverage_v3 import (
    DenseSourceCoveragePolicyV3,
    deterministic_phase_fraction,
    phase_jittered_unit_cube_ray_points,
)
from models.iris.v3.dense_source_sampling_v3 import (
    camera_pixel_ray_origins_normalized,
)


@dataclass(frozen=True)
class SourceConstraintSamplingPolicyV4:
    foreground_component_rays_per_component_per_view: int = 4
    foreground_boundary_rays_per_view: int = 12
    foreground_interior_rays_per_view: int = 12
    near_background_rays_per_view: int = 32
    far_background_rays_per_view: int = 16
    hard_negative_replay_rays_per_view: int = 16
    hard_negative_refresh_candidates_per_view: int = 1024
    hard_negative_bank_size_per_view: int = 64
    hard_negative_refresh_interval_steps: int = 80
    near_boundary_max_px: float = 32.0

    def validate(self) -> None:
        ints = (
            self.foreground_component_rays_per_component_per_view,
            self.foreground_boundary_rays_per_view,
            self.foreground_interior_rays_per_view,
            self.near_background_rays_per_view,
            self.far_background_rays_per_view,
            self.hard_negative_replay_rays_per_view,
            self.hard_negative_refresh_candidates_per_view,
            self.hard_negative_bank_size_per_view,
            self.hard_negative_refresh_interval_steps,
        )
        if min(map(int, ints)) <= 0:
            raise ValueError("all V4 source-constraint quotas must be positive")
        if not math.isfinite(self.near_boundary_max_px) or self.near_boundary_max_px <= 0:
            raise ValueError("near_boundary_max_px must be finite and positive")
        if self.hard_negative_replay_rays_per_view > self.hard_negative_bank_size_per_view:
            raise ValueError("replay quota cannot exceed hard-negative bank size")


@dataclass(frozen=True)
class SourceConstraintViewStateV4:
    width: int
    height: int
    foreground: np.ndarray
    admitted: np.ndarray
    component_id: np.ndarray
    boundary: np.ndarray
    interior: np.ndarray
    distance_to_foreground_px: np.ndarray
    near_background_flat: np.ndarray
    far_background_flat: np.ndarray
    all_background_flat: np.ndarray
    component_count: int


@dataclass(frozen=True)
class SourceConstraintRaySelectionV4:
    flat_pixel_index: np.ndarray
    target_foreground: np.ndarray
    component_id: np.ndarray
    boundary: np.ndarray
    distance_to_foreground_px: np.ndarray
    hard_replay: np.ndarray

    @property
    def ray_count(self) -> int:
        return int(self.flat_pixel_index.size)


def deterministic_view_seed_v4(
    fit_seed: int,
    training_step: int,
    view_index: int,
    lane: str,
) -> int:
    payload = (
        f"{int(fit_seed)}:{int(training_step)}:{int(view_index)}:{str(lane)}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def build_source_constraint_view_state_v4(
    foreground_mask: np.ndarray,
    admitted_mask: np.ndarray,
    *,
    near_boundary_max_px: float = 32.0,
) -> SourceConstraintViewStateV4:
    foreground = np.asarray(foreground_mask, dtype=bool)
    admitted = np.asarray(admitted_mask, dtype=bool)
    if foreground.ndim != 2 or admitted.shape != foreground.shape:
        raise ValueError("foreground/admitted masks must be matching [H,W]")
    if not np.any(foreground):
        raise ValueError("foreground mask must be non-empty")
    if not np.any(admitted):
        raise ValueError("camera has no admitted normalization-cube rays")
    if np.any(foreground & ~admitted):
        raise ValueError("SOURCE_FOREGROUND_OUTSIDE_NORMALIZATION_RAY_DOMAIN")

    four = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    labels, component_count = ndimage.label(foreground, structure=four)
    component_id = np.where(foreground, labels.astype(np.int64) - 1, -1)
    interior = ndimage.binary_erosion(
        foreground,
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
    )
    boundary = foreground & ~interior
    distance = ndimage.distance_transform_edt(~foreground).astype(np.float32)
    distance[foreground] = 0.0

    background = (~foreground) & admitted
    near = background & (distance > 0.0) & (distance <= float(near_boundary_max_px))
    far = background & (distance > float(near_boundary_max_px))
    all_background = np.flatnonzero(background.reshape(-1)).astype(np.int64)
    near_flat = np.flatnonzero(near.reshape(-1)).astype(np.int64)
    far_flat = np.flatnonzero(far.reshape(-1)).astype(np.int64)
    if near_flat.size == 0 or far_flat.size == 0:
        raise ValueError("V4 requires both near and far source-background pools")

    flat_distance = distance.reshape(-1)
    near_order = np.lexsort((near_flat, flat_distance[near_flat]))
    far_order = np.lexsort((far_flat, flat_distance[far_flat]))
    return SourceConstraintViewStateV4(
        width=int(foreground.shape[1]),
        height=int(foreground.shape[0]),
        foreground=foreground,
        admitted=admitted,
        component_id=component_id,
        boundary=boundary,
        interior=interior,
        distance_to_foreground_px=distance,
        near_background_flat=near_flat[near_order],
        far_background_flat=far_flat[far_order],
        all_background_flat=np.sort(all_background),
        component_count=int(component_count),
    )


def _rng_sample(rng: np.random.Generator, pool: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(pool, dtype=np.int64).reshape(-1)
    n = int(count)
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    if values.size == 0:
        raise ValueError("required V4 source-constraint pool is empty")
    return np.asarray(
        rng.choice(values, size=n, replace=values.size < n),
        dtype=np.int64,
    )


def _cyclic_take(pool: np.ndarray, count: int, visit_index: int) -> np.ndarray:
    values = np.asarray(pool, dtype=np.int64).reshape(-1)
    if values.size == 0:
        raise ValueError("cyclic V4 pool is empty")
    n = int(count)
    visit = int(visit_index)
    if n <= 0 or visit < 0:
        raise ValueError("cyclic count/visit invalid")
    start = (visit * n) % int(values.size)
    take = (start + np.arange(n, dtype=np.int64)) % int(values.size)
    return values[take]


def sample_source_constraint_view_v4(
    state: SourceConstraintViewStateV4,
    *,
    fit_seed: int,
    training_step: int,
    view_index: int,
    policy: SourceConstraintSamplingPolicyV4 = SourceConstraintSamplingPolicyV4(),
    hard_negative_flat_indices: np.ndarray | None = None,
) -> SourceConstraintRaySelectionV4:
    """Sample foreground, near/far background, and persistent hard negatives."""

    policy.validate()
    flat_component = state.component_id.reshape(-1)
    flat_foreground = state.foreground.reshape(-1)
    flat_boundary = state.boundary.reshape(-1)
    flat_interior = state.interior.reshape(-1)
    flat_distance = state.distance_to_foreground_px.reshape(-1)

    selected: list[np.ndarray] = []
    replay_flags: list[np.ndarray] = []
    rng = np.random.default_rng(
        deterministic_view_seed_v4(fit_seed, training_step, view_index, "foreground")
    )
    for component in range(int(state.component_count)):
        pool = np.flatnonzero(flat_component == component)
        row = _rng_sample(
            rng,
            pool,
            policy.foreground_component_rays_per_component_per_view,
        )
        selected.append(row)
        replay_flags.append(np.zeros(len(row), dtype=bool))

    boundary_pool = np.flatnonzero(flat_boundary)
    interior_pool = np.flatnonzero(flat_interior)
    boundary_count = int(policy.foreground_boundary_rays_per_view)
    interior_count = int(policy.foreground_interior_rays_per_view)
    if interior_pool.size == 0:
        boundary_count += interior_count
        interior_count = 0
    row = _rng_sample(rng, boundary_pool, boundary_count)
    selected.append(row)
    replay_flags.append(np.zeros(len(row), dtype=bool))
    if interior_count:
        row = _rng_sample(rng, interior_pool, interior_count)
        selected.append(row)
        replay_flags.append(np.zeros(len(row), dtype=bool))

    visit = int(training_step) - 1
    near = _cyclic_take(
        state.near_background_flat,
        policy.near_background_rays_per_view,
        visit,
    )
    far = _cyclic_take(
        state.far_background_flat,
        policy.far_background_rays_per_view,
        visit,
    )
    selected.extend((near, far))
    replay_flags.extend(
        (np.zeros(len(near), dtype=bool), np.zeros(len(far), dtype=bool))
    )

    bank = np.asarray(
        hard_negative_flat_indices if hard_negative_flat_indices is not None else [],
        dtype=np.int64,
    ).reshape(-1)
    if bank.size:
        valid = bank[
            (bank >= 0)
            & (bank < flat_foreground.size)
            & (~flat_foreground[bank])
        ]
        if valid.size:
            replay = _cyclic_take(
                np.unique(valid),
                policy.hard_negative_replay_rays_per_view,
                visit,
            )
        else:
            replay = _cyclic_take(
                state.far_background_flat,
                policy.hard_negative_replay_rays_per_view,
                visit + 17,
            )
    else:
        replay = _cyclic_take(
            state.far_background_flat,
            policy.hard_negative_replay_rays_per_view,
            visit + 17,
        )
    selected.append(replay)
    replay_flags.append(np.ones(len(replay), dtype=bool))

    index = np.concatenate(selected).astype(np.int64, copy=False)
    hard = np.concatenate(replay_flags).astype(bool, copy=False)
    target = flat_foreground[index].astype(np.float32)
    component = flat_component[index].astype(np.int64)
    boundary = flat_boundary[index].astype(np.float32)
    distance = flat_distance[index].astype(np.float32)
    if np.any((target > 0.5) & (component < 0)):
        raise RuntimeError("V4_FOREGROUND_COMPONENT_TARGET_DRIFT")
    if np.any((target < 0.5) & (component >= 0)):
        raise RuntimeError("V4_BACKGROUND_COMPONENT_TARGET_DRIFT")
    if set(map(int, component[component >= 0])) != set(range(state.component_count)):
        raise RuntimeError("V4_DROPPED_FOREGROUND_COMPONENT")
    return SourceConstraintRaySelectionV4(
        flat_pixel_index=index,
        target_foreground=target,
        component_id=component,
        boundary=boundary,
        distance_to_foreground_px=distance,
        hard_replay=hard,
    )


def sample_hard_negative_refresh_candidates_v4(
    state: SourceConstraintViewStateV4,
    *,
    fit_seed: int,
    refresh_index: int,
    view_index: int,
    count: int,
) -> np.ndarray:
    rng = np.random.default_rng(
        deterministic_view_seed_v4(fit_seed, refresh_index, view_index, "hard-refresh")
    )
    return _rng_sample(rng, state.all_background_flat, int(count))


def select_hard_negative_replay_bank_v4(
    flat_pixel_index: np.ndarray,
    ray_min_sdf: np.ndarray,
    *,
    bank_size: int,
) -> np.ndarray:
    index = np.asarray(flat_pixel_index, dtype=np.int64).reshape(-1)
    sdf = np.asarray(ray_min_sdf, dtype=np.float64).reshape(-1)
    if index.shape != sdf.shape or index.size == 0:
        raise ValueError("hard-negative index/sdf inputs must match and be non-empty")
    if not np.isfinite(sdf).all():
        raise ValueError("hard-negative SDF values must be finite")
    size = int(bank_size)
    if size <= 0:
        raise ValueError("hard-negative bank_size must be positive")
    order = np.lexsort((index, sdf))
    return np.asarray(index[order[: min(size, index.size)]], dtype=np.int64)


def build_ray_points_v4(
    flat_pixel_index: np.ndarray,
    *,
    width: int,
    height: int,
    camera_origin_normalized: np.ndarray,
    camera_right: np.ndarray,
    camera_screen_up: np.ndarray,
    camera_forward: np.ndarray,
    camera_half_extent_normalized: float,
    fit_seed: int,
    training_step: int,
    view_index: int,
    coverage_policy: DenseSourceCoveragePolicyV3 = DenseSourceCoveragePolicyV3(),
) -> torch.Tensor:
    origins_np, directions_np = camera_pixel_ray_origins_normalized(
        flat_pixel_index,
        width=int(width),
        height=int(height),
        camera_origin_normalized=camera_origin_normalized,
        camera_right=camera_right,
        camera_screen_up=camera_screen_up,
        camera_forward=camera_forward,
        camera_half_extent_normalized=camera_half_extent_normalized,
    )
    origins = torch.from_numpy(origins_np)
    directions = torch.from_numpy(directions_np)
    pair = tuple(sorted((int(view_index), int((view_index + 4) % 8))))
    phase = deterministic_phase_fraction(fit_seed, training_step, pair)
    return phase_jittered_unit_cube_ray_points(
        origins,
        directions,
        phase_fraction=phase,
        policy=coverage_policy,
    )
