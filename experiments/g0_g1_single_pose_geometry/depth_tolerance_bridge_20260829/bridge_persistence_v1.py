#!/usr/bin/env python3
"""Bridge adapter that replays sealed E0 MUTUAL_P003 matching on corrupted view-local P.

Forward path uses only:
- exact observable raster rows;
- corrupted P = P_exact + delta_d F;
- normals re-derived from corrupted P;
- known camera bases / frozen clean half-extent;
- sealed `derived_match_row` matcher.

Teacher triangle/barycentric identity is not passed into the forward persistence
routine. Clean authority is used only to freeze the source row carrier and may be
used later by evaluator-only diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from depth_corruption_v1 import (
    DepthCorruptionSpec,
    corruption_diagnostics,
    corrupt_points_along_ray,
    depth_delta_for_view,
)
from e0_geometry import camera_for_view, derive_view_local_normals
from surface_builder_e0_v1 import (
    ObservableView,
    load_asset_authority,
    deterministic_anchor_rows,
    derived_match_row,
)


@dataclass(frozen=True)
class CorruptedPersistenceCarrier:
    asset_id: str
    P: np.ndarray
    support: np.ndarray
    raster_xy: np.ndarray
    source_view: np.ndarray
    source_row: np.ndarray
    matched_row: np.ndarray
    X36: np.ndarray
    per_view_corruption: tuple[dict, ...]


def _corrupt_observable_views(asset_id: str, clean_obs, spec: DepthCorruptionSpec):
    out = []
    reports = []
    for obs in clean_obs:
        cam = camera_for_view(obs.view, obs.half_extent)
        dd = depth_delta_for_view(
            asset_id,
            obs.view,
            obs.pixel_linear_index,
            resolution=obs.resolution,
            spec=spec,
        )
        ph = corrupt_points_along_ray(obs.P, cam["forward"], dd)
        nd, nv = derive_view_local_normals(obs.pixel_linear_index, ph, obs.resolution)
        out.append(
            ObservableView(
                obs.view,
                obs.resolution,
                obs.pixel_linear_index.copy(),
                ph,
                nd,
                nv,
                obs.half_extent,
                obs.camera_recovery,
            )
        )
        rep = {"view": int(obs.view), **corruption_diagnostics(obs.P, ph, cam["forward"], dd)}
        reports.append(rep)
    return out, tuple(reports)


def _x36_from_carrier(P: np.ndarray, support: np.ndarray, raster_xy: np.ndarray) -> np.ndarray:
    P = np.asarray(P, np.float32)
    support = np.asarray(support, np.uint8)
    raster_xy = np.asarray(raster_xy, np.float32)
    if support.shape != (len(P), 8):
        raise ValueError("support must be [N,8]")
    if raster_xy.shape != (len(P), 8, 2):
        raise ValueError("raster_xy must be [N,8,2]")
    x = np.zeros((len(P), 36), np.float32)
    x[:, 0:3] = P
    x[:, 3:11] = support.astype(np.float32)
    x[:, 11:27] = raster_xy.reshape(len(P), 16)
    x[:, 27] = support.mean(axis=1, dtype=np.float32)
    for v in range(8):
        f = np.asarray(camera_for_view(v, 1.0)["forward"], np.float32)
        x[:, 28 + v] = P @ f
    return x


def build_corrupted_persistence_carrier(
    asset_dir: str | Path,
    spec: DepthCorruptionSpec,
    *,
    anchor_count: int = 512,
) -> CorruptedPersistenceCarrier:
    """Re-run sealed teacher-free persistence after ray-aligned view-local corruption.

    Source-row identity is frozen from the clean P-only deterministic anchor sampler
    before corruption. This prevents the bridge from turning into an anchor-sampler
    intervention while still moving the source anchor P itself by the predicted-depth
    residual of its source view.
    """
    asset_dir = Path(asset_dir)
    asset_id = asset_dir.name
    _geom, authority = load_asset_authority(asset_dir)
    clean_obs = [a.observable for a in authority]

    src_views, src_rows, _clean_anchor_P = deterministic_anchor_rows(
        authority, asset_id, anchor_count=anchor_count
    )

    obs, corruption_reports = _corrupt_observable_views(asset_id, clean_obs, spec)
    K = len(src_views)

    anchor_P = np.zeros((K, 3), np.float32)
    source_grid = np.zeros((K, 2), np.float32)
    source_N = np.zeros((K, 3), np.float32)
    matched = np.full((K, 8), -1, np.int64)
    support = np.zeros((K, 8), np.uint8)
    raster_xy = np.zeros((K, 8, 2), np.float32)

    for i, (sv0, sr0) in enumerate(zip(src_views, src_rows)):
        sv = int(sv0)
        sr = int(sr0)
        anchor_P[i] = obs[sv].P[sr]
        source_grid[i] = obs[sv].grid[sr]
        source_N[i] = obs[sv].N_derived[sr]

        matched[i, sv] = sr
        support[i, sv] = 1
        raster_xy[i, sv] = source_grid[i]

        for tv in range(8):
            if tv == sv:
                continue
            row, _info = derived_match_row(
                anchor_P[i],
                source_N[i],
                source_grid[i],
                sv,
                obs[sv].half_extent,
                obs[tv],
            )
            matched[i, tv] = int(row)
            if row >= 0:
                support[i, tv] = 1
                raster_xy[i, tv] = obs[tv].grid[int(row)]

    x36 = _x36_from_carrier(anchor_P, support, raster_xy)
    return CorruptedPersistenceCarrier(
        asset_id=asset_id,
        P=anchor_P,
        support=support,
        raster_xy=raster_xy,
        source_view=np.asarray(src_views, np.int16),
        source_row=np.asarray(src_rows, np.int64),
        matched_row=matched,
        X36=x36,
        per_view_corruption=corruption_reports,
    )
