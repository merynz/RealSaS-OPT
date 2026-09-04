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


def dense_signed_grid_v3(
    query_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    resolution: int = 256,
    bounds: tuple[float, float] = (-1.0, 1.0),
    chunk_size: int = 262144,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Query a signed field on a deterministic XYZ cube.

    query_fn receives [1,N,3] points and must return either [1,N] or [N]. The
    returned numpy array is indexed [z,y,x], matching the zero-surface decoder.
    """
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
