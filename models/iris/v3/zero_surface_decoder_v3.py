from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch


@dataclass(frozen=True)
class ZeroSurfaceMeshV3:
    vertices_normalized: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    field_min: float
    field_max: float
    decoder_id: str = "RealSaS.ZeroSurfaceDecoder.MarchingCubes.v3"


@dataclass(frozen=True)
class CompactZeroSurfaceV3:
    vertices_normalized: np.ndarray
    normals: np.ndarray
    relations: np.ndarray
    source_vertex_count: int
    voxel_divisions: int
    normal_k: int
    compactor_id: str = "RealSaS.ZeroSurfaceCompactor.AdaptiveVoxel.v3"
    normal_operator_id: str = "RealSaS.ZeroSurfaceNormal.RobustLocalPCA.v1"


def dense_signed_grid_v3(
    query_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    resolution: int = 256,
    bounds: tuple[float, float] = (-1.0, 1.0),
    chunk_size: int = 262144,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Query a signed field on a deterministic XYZ cube."""
    if resolution < 4 or chunk_size <= 0:
        raise ValueError("invalid dense-grid request")
    lo, hi = map(float, bounds)
    if not np.isfinite([lo, hi]).all() or not lo < hi:
        raise ValueError("invalid bounds")
    dev = torch.device(device) if device is not None else torch.device("cpu")
    axis = np.linspace(lo, hi, int(resolution), dtype=np.float32)
    grid = np.empty((resolution, resolution, resolution), np.float32)
    with torch.no_grad():
        for z0 in range(resolution):
            zz, yy, xx = np.meshgrid(axis[z0 : z0 + 1], axis, axis, indexing="ij")
            p = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
            vals = []
            for start in range(0, len(p), int(chunk_size)):
                q = torch.from_numpy(p[start : start + chunk_size]).to(dev).view(1, -1, 3)
                y = query_fn(q)
                if isinstance(y, dict):
                    y = y["sdf"]
                vals.append(torch.as_tensor(y).detach().float().reshape(-1).cpu().numpy())
            grid[z0] = np.concatenate(vals).reshape(resolution, resolution)
    if not np.isfinite(grid).all():
        raise ValueError("non-finite signed grid")
    return grid


def extract_zero_surface_mesh_v3(
    signed_grid: np.ndarray,
    *,
    bounds: tuple[float, float] = (-1.0, 1.0),
    level: float = 0.0,
) -> ZeroSurfaceMeshV3:
    """Extract the 3D zero level set; no ray/first-hit decision exists here."""
    field = np.asarray(signed_grid, dtype=np.float32)
    if field.ndim != 3 or min(field.shape) < 4 or len(set(field.shape)) != 1:
        raise ValueError("signed_grid must be cubic [R,R,R]")
    if not np.isfinite(field).all():
        raise ValueError("signed_grid contains non-finite values")
    fmin, fmax = float(field.min()), float(field.max())
    if not fmin < float(level) < fmax:
        raise ValueError(f"zero level is not bracketed: min={fmin} max={fmax} level={level}")
    try:
        from skimage import measure
    except ImportError as exc:
        raise RuntimeError("scikit-image is required for zero-surface extraction") from exc
    lo, hi = map(float, bounds)
    r = field.shape[0]
    spacing = ((hi - lo) / (r - 1),) * 3
    v_zyx, faces, n_zyx, _ = measure.marching_cubes(
        field, level=float(level), spacing=spacing, allow_degenerate=False
    )
    vertices = np.stack(
        [lo + v_zyx[:, 2], lo + v_zyx[:, 1], lo + v_zyx[:, 0]], axis=-1
    ).astype(np.float32)
    normals = np.stack([n_zyx[:, 2], n_zyx[:, 1], n_zyx[:, 0]], axis=-1).astype(np.float32)
    denom = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(denom, 1e-12)
    return ZeroSurfaceMeshV3(vertices, faces.astype(np.int64), normals, fmin, fmax)


def robust_zero_surface_normals_v3(
    vertices_normalized: np.ndarray,
    orientation_hints: np.ndarray,
    *,
    k: int = 64,
) -> np.ndarray:
    """Deterministic local-PCA normals over the decoded zero surface.

    The implicit/Marching-Cubes normal is used only to choose sign. Plane authority
    comes from the local zero-surface geometry itself.
    """
    p = np.asarray(vertices_normalized, dtype=np.float64)
    hint = np.asarray(orientation_hints, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or hint.shape != p.shape or len(p) < 4:
        raise ValueError("normal input must be matching [N,3], N>=4")
    if not np.isfinite(p).all() or not np.isfinite(hint).all():
        raise ValueError("normal input must be finite")
    kk = min(max(3, int(k)), len(p) - 1)
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError("scipy is required for robust zero-surface normals") from exc
    tree = cKDTree(p)
    _, nn = tree.query(p, k=kk + 1, workers=-1)
    nn = np.asarray(nn[:, 1:], dtype=np.int64)
    out = np.empty_like(p)
    for start in range(0, len(p), 8192):
        rows = nn[start : start + 8192]
        x = p[rows]
        center = np.median(x, axis=1, keepdims=True)
        dist = np.linalg.norm(x - center, axis=2)
        med = np.median(dist, axis=1)
        mad = np.median(np.abs(dist - med[:, None]), axis=1)
        gate = dist <= (med[:, None] + 3.0 * np.maximum(1.4826 * mad[:, None], 1e-8))
        w = gate.astype(np.float64)
        sw = np.maximum(w.sum(axis=1), 3.0)
        mu = (w[:, :, None] * x).sum(axis=1) / sw[:, None]
        y = x - mu[:, None, :]
        cov = np.einsum("nki,nkj,nk->nij", y, y, w) / sw[:, None, None]
        _, vec = np.linalg.eigh(cov)
        n = vec[:, :, 0]
        h = hint[start : start + len(n)]
        flip = np.sum(n * h, axis=1) < 0.0
        n[flip] *= -1.0
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        out[start : start + len(n)] = n
    return out.astype(np.float32)


def compact_zero_surface_mesh_v3(
    mesh: ZeroSurfaceMeshV3,
    *,
    target_nodes: int = 1024,
    normal_k: int = 64,
) -> CompactZeroSurfaceV3:
    """Compact a dense decoded surface for Geppetto/Arachne without hidden truth.

    Adaptive voxel aggregation is deterministic. Local-PCA normals are first derived
    on the dense decoded surface, then averaged inside each voxel so compaction does
    not erase the local differential estimate.
    """
    v = np.asarray(mesh.vertices_normalized, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    hint = np.asarray(mesh.normals, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 3 or len(v) < 4:
        raise ValueError("mesh vertices must be [N,3], N>=4")
    if f.ndim != 2 or f.shape[1] != 3 or np.any(f < 0) or np.any(f >= len(v)):
        raise ValueError("mesh faces must be valid triangles")
    if hint.shape != v.shape:
        raise ValueError("mesh normals must match vertices")
    if target_nodes < 64:
        raise ValueError("target_nodes must be >=64")

    dense_normals = robust_zero_surface_normals_v3(v, hint, k=normal_k)
    if len(v) <= target_nodes:
        mapped = f
        edges = np.concatenate([mapped[:, [0, 1]], mapped[:, [1, 2]], mapped[:, [2, 0]]], axis=0)
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges[edges[:, 0] != edges[:, 1]], axis=0)
        return CompactZeroSurfaceV3(v.astype(np.float32), dense_normals, edges, len(v), 0, min(normal_k, len(v)-1))

    lo = v.min(axis=0)
    span = np.maximum(v.max(axis=0) - lo, 1e-12)

    def labels_for(divisions: int):
        keys = np.floor((v - lo) / span * divisions).astype(np.int64)
        keys = np.clip(keys, 0, divisions - 1)
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        return unique, inverse

    low, high = 1, 512
    best = None
    while low <= high:
        mid = (low + high) // 2
        unique, inverse = labels_for(mid)
        if len(unique) <= target_nodes:
            best = (mid, unique, inverse)
            low = mid + 1
        else:
            high = mid - 1
    if best is None:
        raise RuntimeError("adaptive voxel compaction failed")
    divisions, unique, inverse = best
    count = len(unique)
    counts = np.bincount(inverse, minlength=count).astype(np.float64)
    points = np.zeros((count, 3), dtype=np.float64)
    normals = np.zeros((count, 3), dtype=np.float64)
    np.add.at(points, inverse, v)
    np.add.at(normals, inverse, dense_normals)
    points /= counts[:, None]
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)

    mapped = inverse[f]
    edges = np.concatenate([mapped[:, [0, 1]], mapped[:, [1, 2]], mapped[:, [2, 0]]], axis=0)
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges[edges[:, 0] != edges[:, 1]], axis=0)
    if len(edges) == 0:
        raise RuntimeError("zero-surface compaction removed all topology")
    return CompactZeroSurfaceV3(
        points.astype(np.float32),
        normals.astype(np.float32),
        edges.astype(np.int64),
        len(v),
        int(divisions),
        min(int(normal_k), len(v) - 1),
    )
