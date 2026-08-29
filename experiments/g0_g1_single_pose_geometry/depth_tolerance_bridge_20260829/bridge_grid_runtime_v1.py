#!/usr/bin/env python3
"""Cached/vectorized scientific-grid runtime for the structured depth bridge.

This module is an execution optimization and route-semantics correction layer.
It does not replace or rewrite the sealed M0-M3 apparatus sources.

Canonical distinction:
  BASE_DERIVED_MATCHER = direct `derived_match_row` acceptance.
  D2_MUTUAL_P003       = BASE + reciprocal reverse match + cycle_P <= 0.003.

The vectorized matcher preserves the scalar candidate ordering, gates, score,
and first-argmin tie semantics. Clean observable authority and the P-only source
row sampler are prepared once per asset and reused across frozen corruption cells.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from bridge_persistence_v1 import (
    CorruptedPersistenceCarrier,
    _corrupt_observable_views,
    _x36_from_carrier,
)
from depth_corruption_v1 import DepthCorruptionSpec
from e0_geometry import camera_for_view
from surface_builder_e0_v1 import (
    ObservableView,
    deterministic_anchor_rows,
    load_asset_authority,
)

VIEWS = 8
POINT_FEATURE_DIM = 36
MUTUAL_P003 = 0.003


@dataclass(frozen=True)
class PreparedBridgeAsset:
    """Observation-only clean state cached across a corruption grid."""
    asset_id: str
    clean_obs: tuple[ObservableView, ...]
    source_view: np.ndarray
    source_row: np.ndarray
    anchor_count: int


@dataclass(frozen=True)
class CorruptedPersistenceState:
    carrier: CorruptedPersistenceCarrier
    observable_views: tuple[ObservableView, ...]


def prepare_bridge_asset(asset_dir: str | Path, *, anchor_count: int = 512) -> PreparedBridgeAsset:
    asset_dir = Path(asset_dir)
    asset_id = asset_dir.name
    _geom, authority = load_asset_authority(asset_dir)
    source_view, source_row, _clean_anchor_P = deterministic_anchor_rows(
        authority, asset_id, anchor_count=anchor_count
    )
    # Teacher triangle/bary fields are discarded after authoritative observable P
    # reconstruction and the P-only source-row sampler freeze.
    clean_obs = tuple(a.observable for a in authority)
    return PreparedBridgeAsset(
        asset_id=asset_id,
        clean_obs=clean_obs,
        source_view=np.asarray(source_view, np.int16),
        source_row=np.asarray(source_row, np.int64),
        anchor_count=int(anchor_count),
    )


def batch_derived_rows_fixed_source(
    anchor_P: np.ndarray,
    anchor_N: np.ndarray,
    anchor_grid: np.ndarray,
    source_view: int,
    source_half_extent: float,
    target: ObservableView,
    *,
    radius_px: int = 4,
    max_common_frame_error: float = 0.006,
    min_abs_normal_cos: float = np.cos(np.deg2rad(50.0)),
    max_reciprocal_error_px: float = 3.0,
) -> np.ndarray:
    """Vectorized scalar `derived_match_row` contract for one fixed source view."""
    from e0_geometry import project_grid, grid_to_nearest_pixel, raster_lookup_near

    P = np.asarray(anchor_P, np.float32)
    N = np.asarray(anchor_N, np.float32)
    G = np.asarray(anchor_grid, np.float32)
    K = len(P)
    if K == 0:
        return np.zeros(0, np.int64)

    tg = project_grid(P, camera_for_view(target.view, target.half_extent))
    inframe = np.all(np.abs(tg) <= 1.0, axis=1)
    x, y = grid_to_nearest_pixel(tg, target.resolution)
    cand = raster_lookup_near(target.pixel_linear_index, x, y, target.resolution, radius_px)
    cvalid = cand >= 0
    cc = np.clip(cand, 0, max(0, len(target.P) - 1))

    tP = target.P[cc]
    pd = np.linalg.norm(tP - P[:, None, :], axis=2)
    src_norm = np.linalg.norm(N, axis=1)
    tN = target.N_derived[cc]
    tgt_norm = np.linalg.norm(tN, axis=2)
    nv = target.N_valid[cc]
    nvalid = cvalid & (src_norm[:, None] > 0.5) & nv & (tgt_norm > 0.5)

    ndot = np.zeros_like(pd, dtype=np.float32)
    numer = np.abs(np.sum(tN * N[:, None, :], axis=2))
    denom = tgt_norm * src_norm[:, None] + 1e-8
    ndot[nvalid] = (numer / denom)[nvalid]

    back = project_grid(
        tP.reshape(-1, 3), camera_for_view(source_view, source_half_extent)
    ).reshape(K, -1, 2)
    recip = np.linalg.norm((back - G[:, None, :]) * (target.resolution / 2.0), axis=2)

    valid = (
        cvalid
        & inframe[:, None]
        & (pd <= max_common_frame_error)
        & nvalid
        & (ndot >= min_abs_normal_cos)
        & (recip <= max_reciprocal_error_px)
    )
    score = (pd / max_common_frame_error) + (1.0 - ndot) + (recip / max_reciprocal_error_px)
    score[~valid] = np.inf
    j = np.argmin(score, axis=1)
    best = cand[np.arange(K), j]
    ok = np.isfinite(score[np.arange(K), j])
    return np.where(ok, best, -1).astype(np.int64)


def build_base_state(prepared: PreparedBridgeAsset, spec: DepthCorruptionSpec) -> CorruptedPersistenceState:
    """Run the pre-cycle BASE derived matcher from one prepared clean asset."""
    obs_list, corruption_reports = _corrupt_observable_views(
        prepared.asset_id, prepared.clean_obs, spec
    )
    obs = tuple(obs_list)
    src_views = np.asarray(prepared.source_view, np.int64)
    src_rows = np.asarray(prepared.source_row, np.int64)
    K = len(src_views)

    anchor_P = np.zeros((K, 3), np.float32)
    source_grid = np.zeros((K, 2), np.float32)
    source_N = np.zeros((K, 3), np.float32)
    matched = np.full((K, VIEWS), -1, np.int64)
    support = np.zeros((K, VIEWS), np.uint8)
    raster_xy = np.zeros((K, VIEWS, 2), np.float32)

    obs_grid = [o.grid for o in obs]
    for sv in range(VIEWS):
        ids = np.flatnonzero(src_views == sv)
        if not len(ids):
            continue
        sr = src_rows[ids]
        anchor_P[ids] = obs[sv].P[sr]
        source_grid[ids] = obs_grid[sv][sr]
        source_N[ids] = obs[sv].N_derived[sr]
        matched[ids, sv] = sr
        support[ids, sv] = 1
        raster_xy[ids, sv] = source_grid[ids]

        for tv in range(VIEWS):
            if tv == sv:
                continue
            rows = batch_derived_rows_fixed_source(
                anchor_P[ids], source_N[ids], source_grid[ids],
                sv, obs[sv].half_extent, obs[tv],
            )
            matched[ids, tv] = rows
            ok = rows >= 0
            if np.any(ok):
                good = ids[ok]
                support[good, tv] = 1
                raster_xy[good, tv] = obs_grid[tv][rows[ok]]

    carrier = CorruptedPersistenceCarrier(
        asset_id=prepared.asset_id,
        P=anchor_P,
        support=support,
        raster_xy=raster_xy,
        source_view=np.asarray(prepared.source_view, np.int16),
        source_row=np.asarray(prepared.source_row, np.int64),
        matched_row=matched,
        X36=_x36_from_carrier(anchor_P, support, raster_xy),
        per_view_corruption=corruption_reports,
    )
    return CorruptedPersistenceState(carrier=carrier, observable_views=obs)


def build_d2_mutual_p003_from_prepared(prepared: PreparedBridgeAsset, spec: DepthCorruptionSpec):
    """Build historical-D2-compatible X36 after BASE + reciprocal cycle admission."""
    state = build_base_state(prepared, spec)
    carrier = state.carrier
    obs = state.observable_views
    P = np.asarray(carrier.P, np.float32)
    sv = np.asarray(carrier.source_view, np.int64)
    rows = np.asarray(carrier.matched_row, np.int64)
    base = np.asarray(carrier.support, bool)

    support = np.zeros_like(base)
    support[np.arange(len(P)), sv] = True
    obs_grid = [o.grid for o in obs]

    for s in range(VIEWS):
        for tv in range(VIEWS):
            if tv == s:
                continue
            ids = np.flatnonzero((sv == s) & base[:, tv])
            if not len(ids):
                continue
            br = rows[ids, tv]
            rr = batch_derived_rows_fixed_source(
                obs[tv].P[br], obs[tv].N_derived[br], obs_grid[tv][br],
                tv, obs[tv].half_extent, obs[s],
            )
            ok = rr >= 0
            cyc = np.full(len(ids), np.inf, np.float32)
            if np.any(ok):
                cyc[ok] = np.linalg.norm(obs[s].P[rr[ok]] - P[ids[ok]], axis=1)
            support[ids, tv] = ok & (cyc <= MUTUAL_P003)

    admitted_rows = rows.copy()
    admitted_rows[~support] = -1
    grid = np.zeros((len(P), VIEWS, 2), np.float32)
    for v in range(VIEWS):
        idx = np.flatnonzero(support[:, v] & (admitted_rows[:, v] >= 0))
        if len(idx):
            grid[idx, v] = obs_grid[v][admitted_rows[idx, v]]

    depth = np.stack([
        P @ np.asarray(camera_for_view(v, obs[v].half_extent)["forward"], np.float32)
        for v in range(VIEWS)
    ], axis=1).astype(np.float32)
    X = np.concatenate([
        P,
        support.astype(np.float32),
        grid.reshape(len(P), 16),
        (support.sum(axis=1, keepdims=True) / float(VIEWS)).astype(np.float32),
        depth,
    ], axis=1).astype(np.float32)
    if X.shape != (len(P), POINT_FEATURE_DIM):
        raise AssertionError(X.shape)
    return state, support, admitted_rows, grid, X
