from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from models.iris.v4.source_constraint_v4 import (
    SourceConstraintRaySelectionV4,
    SourceConstraintViewStateV4,
    _cyclic_take,
    _rng_sample,
    deterministic_view_seed_v4,
)


@dataclass(frozen=True)
class SourceConstraintSamplingPolicyV41:
    """Demo V4.1 sampler: persistent hard positives + distance-banded negatives.

    V4 showed that source-negative error was strongly localized to the 0-2 px band
    while foreground recall regressed. V4.1 therefore keeps the architecture frozen and
    changes only which exact source constraints are retained throughout optimization.
    """

    foreground_component_rays_per_component_per_view: int = 4
    foreground_boundary_rays_per_view: int = 12
    foreground_interior_rays_per_view: int = 12
    hard_positive_replay_rays_per_view: int = 16
    hard_positive_refresh_candidates_per_view: int = 1024
    hard_positive_bank_size_per_view: int = 64

    background_0_2_rays_per_view: int = 16
    background_2_8_rays_per_view: int = 16
    background_8_32_rays_per_view: int = 16
    background_gt32_rays_per_view: int = 16
    hard_negative_replay_rays_per_band_per_view: int = 8
    hard_negative_refresh_candidates_per_band_per_view: int = 512
    hard_negative_bank_size_per_band_per_view: int = 64

    hard_refresh_interval_steps: int = 80

    def validate(self) -> None:
        values = (
            self.foreground_component_rays_per_component_per_view,
            self.foreground_boundary_rays_per_view,
            self.foreground_interior_rays_per_view,
            self.hard_positive_replay_rays_per_view,
            self.hard_positive_refresh_candidates_per_view,
            self.hard_positive_bank_size_per_view,
            self.background_0_2_rays_per_view,
            self.background_2_8_rays_per_view,
            self.background_8_32_rays_per_view,
            self.background_gt32_rays_per_view,
            self.hard_negative_replay_rays_per_band_per_view,
            self.hard_negative_refresh_candidates_per_band_per_view,
            self.hard_negative_bank_size_per_band_per_view,
            self.hard_refresh_interval_steps,
        )
        if min(map(int, values)) <= 0:
            raise ValueError("all V4.1 source-constraint quotas must be positive")
        if self.hard_positive_replay_rays_per_view > self.hard_positive_bank_size_per_view:
            raise ValueError("hard-positive replay quota cannot exceed bank size")
        if self.hard_negative_replay_rays_per_band_per_view > self.hard_negative_bank_size_per_band_per_view:
            raise ValueError("hard-negative replay quota cannot exceed band bank size")


def source_background_band_pools_v41(
    state: SourceConstraintViewStateV4,
) -> dict[str, np.ndarray]:
    flat_distance = state.distance_to_foreground_px.reshape(-1)
    background = state.all_background_flat
    distance = flat_distance[background]
    pools = {
        "0_2": background[(distance > 0.0) & (distance <= 2.0)],
        "2_8": background[(distance > 2.0) & (distance <= 8.0)],
        "8_32": background[(distance > 8.0) & (distance <= 32.0)],
        "gt_32": background[distance > 32.0],
    }
    for name, values in pools.items():
        if values.size == 0:
            raise ValueError(f"V4.1 source-background band {name} is empty")
        d = flat_distance[values]
        order = np.lexsort((values, d))
        pools[name] = np.asarray(values[order], dtype=np.int64)
    return pools


def sample_hard_positive_refresh_candidates_v41(
    state: SourceConstraintViewStateV4,
    *,
    fit_seed: int,
    refresh_index: int,
    view_index: int,
    count: int,
) -> np.ndarray:
    foreground = np.flatnonzero(state.foreground.reshape(-1)).astype(np.int64)
    rng = np.random.default_rng(
        deterministic_view_seed_v4(fit_seed, refresh_index, view_index, "hard-positive-refresh-v41")
    )
    return _rng_sample(rng, foreground, int(count))


def sample_hard_negative_refresh_candidates_by_band_v41(
    state: SourceConstraintViewStateV4,
    *,
    fit_seed: int,
    refresh_index: int,
    view_index: int,
    count_per_band: int,
) -> dict[str, np.ndarray]:
    pools = source_background_band_pools_v41(state)
    out: dict[str, np.ndarray] = {}
    for band, pool in pools.items():
        rng = np.random.default_rng(
            deterministic_view_seed_v4(
                fit_seed, refresh_index, view_index, f"hard-negative-refresh-v41-{band}"
            )
        )
        out[band] = _rng_sample(rng, pool, int(count_per_band))
    return out


def select_hard_positive_replay_bank_v41(
    flat_pixel_index: np.ndarray,
    ray_min_sdf: np.ndarray,
    boundary: np.ndarray,
    *,
    boundary_inside_margin_normalized: float,
    interior_inside_margin_normalized: float,
    bank_size: int,
) -> np.ndarray:
    index = np.asarray(flat_pixel_index, dtype=np.int64).reshape(-1)
    sdf = np.asarray(ray_min_sdf, dtype=np.float64).reshape(-1)
    is_boundary = np.asarray(boundary, dtype=bool).reshape(-1)
    if index.shape != sdf.shape or index.shape != is_boundary.shape or index.size == 0:
        raise ValueError("V4.1 hard-positive inputs must match and be non-empty")
    if not np.isfinite(sdf).all():
        raise ValueError("V4.1 hard-positive SDF must be finite")
    size = int(bank_size)
    if size <= 0:
        raise ValueError("V4.1 hard-positive bank_size must be positive")
    margin = np.where(
        is_boundary,
        float(boundary_inside_margin_normalized),
        float(interior_inside_margin_normalized),
    )
    violation = sdf + margin
    # Highest positive violation is the most badly missed foreground ray.
    order = np.lexsort((index, -violation))
    return np.asarray(index[order[: min(size, index.size)]], dtype=np.int64)


def select_hard_negative_replay_bank_v41(
    flat_pixel_index: np.ndarray,
    ray_min_sdf: np.ndarray,
    *,
    bank_size: int,
) -> np.ndarray:
    index = np.asarray(flat_pixel_index, dtype=np.int64).reshape(-1)
    sdf = np.asarray(ray_min_sdf, dtype=np.float64).reshape(-1)
    if index.shape != sdf.shape or index.size == 0:
        raise ValueError("V4.1 hard-negative inputs must match and be non-empty")
    if not np.isfinite(sdf).all():
        raise ValueError("V4.1 hard-negative SDF must be finite")
    size = int(bank_size)
    if size <= 0:
        raise ValueError("V4.1 hard-negative bank_size must be positive")
    order = np.lexsort((index, sdf))
    return np.asarray(index[order[: min(size, index.size)]], dtype=np.int64)


def _valid_bank(bank: np.ndarray | None, valid_mask: np.ndarray) -> np.ndarray:
    values = np.asarray(bank if bank is not None else [], dtype=np.int64).reshape(-1)
    if not values.size:
        return values
    good = (values >= 0) & (values < valid_mask.size)
    values = values[good]
    values = values[valid_mask[values]]
    return np.unique(values)


def sample_source_constraint_view_v41(
    state: SourceConstraintViewStateV4,
    *,
    fit_seed: int,
    training_step: int,
    view_index: int,
    policy: SourceConstraintSamplingPolicyV41 = SourceConstraintSamplingPolicyV41(),
    hard_positive_flat_indices: np.ndarray | None = None,
    hard_negative_flat_indices_by_band: dict[str, np.ndarray] | None = None,
) -> SourceConstraintRaySelectionV4:
    policy.validate()
    flat_foreground = state.foreground.reshape(-1)
    flat_component = state.component_id.reshape(-1)
    flat_boundary = state.boundary.reshape(-1)
    flat_interior = state.interior.reshape(-1)
    flat_distance = state.distance_to_foreground_px.reshape(-1)
    background_mask = ~flat_foreground
    bands = source_background_band_pools_v41(state)
    visit = int(training_step) - 1

    selected: list[np.ndarray] = []
    replay_flags: list[np.ndarray] = []
    rng = np.random.default_rng(
        deterministic_view_seed_v4(fit_seed, training_step, view_index, "foreground-v41")
    )

    for component in range(int(state.component_count)):
        pool = np.flatnonzero(flat_component == component)
        row = _rng_sample(rng, pool, policy.foreground_component_rays_per_component_per_view)
        selected.append(row)
        replay_flags.append(np.zeros(row.size, dtype=bool))

    boundary_pool = np.flatnonzero(flat_boundary)
    interior_pool = np.flatnonzero(flat_interior)
    boundary_count = int(policy.foreground_boundary_rays_per_view)
    interior_count = int(policy.foreground_interior_rays_per_view)
    if interior_pool.size == 0:
        boundary_count += interior_count
        interior_count = 0
    row = _rng_sample(rng, boundary_pool, boundary_count)
    selected.append(row); replay_flags.append(np.zeros(row.size, dtype=bool))
    if interior_count:
        row = _rng_sample(rng, interior_pool, interior_count)
        selected.append(row); replay_flags.append(np.zeros(row.size, dtype=bool))

    positive_bank = _valid_bank(hard_positive_flat_indices, flat_foreground)
    if positive_bank.size:
        hard_pos = _cyclic_take(positive_bank, policy.hard_positive_replay_rays_per_view, visit)
    else:
        hard_pos = _cyclic_take(boundary_pool, policy.hard_positive_replay_rays_per_view, visit + 11)
    selected.append(hard_pos); replay_flags.append(np.ones(hard_pos.size, dtype=bool))

    band_counts = {
        "0_2": policy.background_0_2_rays_per_view,
        "2_8": policy.background_2_8_rays_per_view,
        "8_32": policy.background_8_32_rays_per_view,
        "gt_32": policy.background_gt32_rays_per_view,
    }
    for band, count in band_counts.items():
        row = _cyclic_take(bands[band], int(count), visit)
        selected.append(row); replay_flags.append(np.zeros(row.size, dtype=bool))

    hard_negative_flat_indices_by_band = hard_negative_flat_indices_by_band or {}
    for offset, band in enumerate(("0_2", "2_8", "8_32", "gt_32")):
        valid = _valid_bank(hard_negative_flat_indices_by_band.get(band), background_mask)
        if valid.size:
            # Ensure stale bank entries cannot migrate between source-distance bands.
            allowed = np.intersect1d(valid, bands[band], assume_unique=False)
        else:
            allowed = valid
        if allowed.size:
            row = _cyclic_take(
                allowed,
                policy.hard_negative_replay_rays_per_band_per_view,
                visit,
            )
        else:
            row = _cyclic_take(
                bands[band],
                policy.hard_negative_replay_rays_per_band_per_view,
                visit + 23 + offset,
            )
        selected.append(row); replay_flags.append(np.ones(row.size, dtype=bool))

    index = np.concatenate(selected).astype(np.int64, copy=False)
    hard = np.concatenate(replay_flags).astype(bool, copy=False)
    target = flat_foreground[index].astype(np.float32)
    component = flat_component[index].astype(np.int64)
    boundary = flat_boundary[index].astype(np.float32)
    distance = flat_distance[index].astype(np.float32)

    if np.any((target > 0.5) & (component < 0)):
        raise RuntimeError("V41_FOREGROUND_COMPONENT_TARGET_DRIFT")
    if np.any((target < 0.5) & (component >= 0)):
        raise RuntimeError("V41_BACKGROUND_COMPONENT_TARGET_DRIFT")
    if set(map(int, component[component >= 0])) != set(range(state.component_count)):
        raise RuntimeError("V41_DROPPED_FOREGROUND_COMPONENT")
    for band, pool in bands.items():
        if not np.any(np.isin(index, pool)):
            raise RuntimeError(f"V41_DROPPED_BACKGROUND_BAND_{band}")

    return SourceConstraintRaySelectionV4(
        flat_pixel_index=index,
        target_foreground=target,
        component_id=component,
        boundary=boundary,
        distance_to_foreground_px=distance,
        hard_replay=hard,
    )
