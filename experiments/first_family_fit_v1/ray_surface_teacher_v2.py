from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch

from models.iris.v2.observation_contract_v2 import ObservationContractV2
from models.iris.v2.q_domain_v2 import RayHypothesisDomainV2


@dataclass(frozen=True)
class RaySurfaceTeacherV2:
    depth: torch.Tensor
    support: torch.Tensor
    mode_count: torch.Tensor
    triangle_id: torch.Tensor
    telemetry: dict


def _ray_triangle_hits_chunk(
    origins: np.ndarray,
    direction: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return dense t/u/v for one orthographic ray chunk against all triangles.

    Shapes:
      origins   [R,3]
      direction [3]
      vertices  [N,3]
      faces     [F,3]
      returns   t/u/v [R,F], with invalid t set to +inf.
    """
    tri = vertices[faces]
    v0 = tri[:, 0]
    e1 = tri[:, 1] - v0
    e2 = tri[:, 2] - v0
    d = np.asarray(direction, np.float64)

    h = np.cross(np.broadcast_to(d, e2.shape), e2)
    a = np.einsum("fi,fi->f", e1, h)
    nonparallel = np.abs(a) > float(eps)
    inv_a = np.zeros_like(a)
    inv_a[nonparallel] = 1.0 / a[nonparallel]

    s = origins[:, None, :] - v0[None, :, :]
    u = np.einsum("rfi,fi->rf", s, h) * inv_a[None, :]
    q = np.cross(s, e1[None, :, :])
    v = np.einsum("i,rfi->rf", d, q) * inv_a[None, :]
    t = np.einsum("fi,rfi->rf", e2, q) * inv_a[None, :]

    valid = (
        nonparallel[None, :]
        & (u >= -eps)
        & (v >= -eps)
        & ((u + v) <= 1.0 + eps)
        & np.isfinite(t)
        & (t >= -eps)
    )
    t = np.where(valid, t, np.inf)
    return t, u, v


def _dedupe_sorted_hits(
    depths: np.ndarray,
    triangle_ids: np.ndarray,
    *,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    if depths.size == 0:
        return depths.astype(np.float64), triangle_ids.astype(np.int64)
    order = np.argsort(depths, kind="stable")
    d = depths[order]
    f = triangle_ids[order]
    keep_d = [float(d[0])]
    keep_f = [int(f[0])]
    for di, fi in zip(d[1:], f[1:]):
        if abs(float(di) - keep_d[-1]) <= float(tolerance):
            # Shared-edge/shared-vertex triangle hits are one physical depth mode.
            # Preserve the lowest triangle id only as deterministic provenance.
            keep_f[-1] = min(keep_f[-1], int(fi))
            continue
        keep_d.append(float(di))
        keep_f.append(int(fi))
    return np.asarray(keep_d, np.float64), np.asarray(keep_f, np.int64)


def build_exact_anchor_ray_surface_teacher_v2(
    *,
    authority_root: Path,
    contract: ObservationContractV2,
    domain: RayHypothesisDomainV2,
    ray_chunk: int = 128,
    intersection_epsilon: float = 1e-9,
    dedupe_relative_tolerance: float = 1e-5,
) -> RaySurfaceTeacherV2:
    """Exact rendered-teacher depth modes from canonical mesh/ray intersections.

    Crucially, Q creation stays camera-only/full-frame. Mesh truth is used only to
    label the already-created analytic anchor rays. All distinct canonical surface
    intersections along a ray are retained; front-visible raster authority is not
    used to create/delete teacher modes.
    """
    if int(ray_chunk) < 1:
        raise ValueError("ray_chunk must be positive")
    if not (0.0 < float(dedupe_relative_tolerance) < 1e-2):
        raise ValueError("invalid dedupe_relative_tolerance")
    if domain.anchor_view.shape[0] != 1:
        raise ValueError("current exact teacher builder expects one family per batch")
    anchor_views = domain.anchor_view[0].detach().cpu().numpy().astype(np.int64)
    if len(np.unique(anchor_views)) != 1:
        raise ValueError("exact teacher requires one fixed anchor view")
    anchor_view = int(anchor_views[0])
    cam = contract.cameras[anchor_view]

    with np.load(Path(authority_root) / "primary_geometry.npz", allow_pickle=False) as z:
        vertices = np.asarray(z["vertices"], dtype=np.float64)
        faces = np.asarray(z["faces"], dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise RuntimeError("canonical vertices shape drift")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise RuntimeError("canonical faces shape drift")
    if len(vertices) == 0 or len(faces) == 0:
        raise RuntimeError("empty canonical geometry")

    vertex_depth = np.asarray(cam.depth_for_point(vertices), np.float64)
    depth_span = float(vertex_depth.max() - vertex_depth.min())
    dedupe_tol = max(1e-7, depth_span * float(dedupe_relative_tolerance))

    grids = domain.anchor_grid[0].detach().cpu().numpy().astype(np.float64)
    origins = np.asarray(cam.ray_origin_for_grid(grids), np.float64)
    direction = np.asarray(cam.forward, np.float64)
    direction /= np.linalg.norm(direction)

    ray_depths: list[np.ndarray] = []
    ray_faces: list[np.ndarray] = []
    raw_hit_counts = []
    for start in range(0, len(origins), int(ray_chunk)):
        end = min(len(origins), start + int(ray_chunk))
        t, _, _ = _ray_triangle_hits_chunk(
            origins[start:end],
            direction,
            vertices,
            faces,
            eps=float(intersection_epsilon),
        )
        for row in t:
            hit_faces = np.flatnonzero(np.isfinite(row)).astype(np.int64)
            hit_depths = row[hit_faces].astype(np.float64)
            raw_hit_counts.append(int(len(hit_depths)))
            dd, ff = _dedupe_sorted_hits(hit_depths, hit_faces, tolerance=dedupe_tol)
            ray_depths.append(dd)
            ray_faces.append(ff)

    mode_count = np.asarray([len(x) for x in ray_depths], dtype=np.int64)
    max_modes = int(mode_count.max(initial=0))
    storage_modes = max(1, max_modes)
    depth = np.full((1, len(ray_depths), storage_modes), np.nan, dtype=np.float32)
    support = np.zeros((1, len(ray_depths), storage_modes), dtype=bool)
    triangle_id = np.full((1, len(ray_depths), storage_modes), -1, dtype=np.int64)
    for q, (dd, ff) in enumerate(zip(ray_depths, ray_faces)):
        if len(dd) == 0:
            continue
        depth[0, q, : len(dd)] = dd.astype(np.float32)
        support[0, q, : len(dd)] = True
        triangle_id[0, q, : len(ff)] = ff

    supported_rays = int((mode_count > 0).sum())
    if supported_rays < 64:
        raise RuntimeError(f"too few exact teacher-supported full-frame rays: {supported_rays}")

    quantiles = {}
    if len(mode_count):
        for qv in (0.5, 0.9, 0.95, 0.99):
            quantiles[f"p{int(qv * 100):02d}"] = float(np.quantile(mode_count, qv))
    telemetry = {
        "schema": "RealSaS.ExactAnchorRaySurfaceTeacher.v2",
        "teacher_source": "CANONICAL_MESH_EXACT_ANCHOR_RAY_TRIANGLE_INTERSECTIONS",
        "learner_q_selection_uses_teacher_or_alpha": False,
        "anchor_view": anchor_view,
        "q_count": int(len(mode_count)),
        "supported_rays": supported_rays,
        "supported_fraction_full_frame": float((mode_count > 0).mean()),
        "raw_triangle_hit_count_max": int(max(raw_hit_counts, default=0)),
        "unique_surface_mode_count_max": max_modes,
        "unique_surface_mode_count_mean_supported": float(mode_count[mode_count > 0].mean()),
        "unique_surface_mode_count_quantiles_all_rays": quantiles,
        "rays_with_multiple_surface_modes": int((mode_count > 1).sum()),
        "rays_with_more_than_3_surface_modes": int((mode_count > 3).sum()),
        "depth_min": float(vertex_depth.min()),
        "depth_max": float(vertex_depth.max()),
        "depth_span": depth_span,
        "intersection_epsilon": float(intersection_epsilon),
        "dedupe_depth_tolerance_world": float(dedupe_tol),
        "front_visible_raster_used_for_teacher": False,
    }
    return RaySurfaceTeacherV2(
        torch.from_numpy(depth),
        torch.from_numpy(support),
        torch.from_numpy(mode_count[None]),
        torch.from_numpy(triangle_id),
        telemetry,
    )


def teacher_depth_range_for_domain_v2(
    *,
    authority_root: Path,
    contract: ObservationContractV2,
    depth_bins: int,
    margin_fraction: float = 0.08,
) -> torch.Tensor:
    if int(depth_bins) < 2:
        raise ValueError("depth_bins must be >=2")
    with np.load(Path(authority_root) / "primary_geometry.npz", allow_pickle=False) as z:
        vertices = np.asarray(z["vertices"], dtype=np.float64)
    cam = contract.cameras[0]
    depths = np.asarray(cam.depth_for_point(vertices), np.float64)
    dmin = float(depths.min())
    dmax = float(depths.max())
    margin = max((dmax - dmin) * float(margin_fraction), 1e-3)
    return torch.linspace(dmin - margin, dmax + margin, int(depth_bins), dtype=torch.float32)
