from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np

from coords import pixel_center_to_grid


def sha256_file(path: str | Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


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


def pixel_linear_to_grid(pixel_linear_index, resolution):
    pix = np.asarray(pixel_linear_index, np.int64)
    y = pix // resolution
    x = pix % resolution
    return pixel_center_to_grid(np.stack([x, y], axis=-1).astype(np.float32), int(resolution))


def project_grid(points, camera):
    """Project canonical points to exact continuous align_corners=False grid coordinates.

    Raster authority is used to prove visibility/surface consistency; it must not quantize
    the correspondence target back to a raster pixel center after that proof succeeds.
    """
    p = np.asarray(points, np.float32)
    right = np.asarray(camera["right"], np.float32)
    up = np.asarray(camera["up"], np.float32)
    he = float(camera["half_extent"])
    return np.stack([(p @ right) / he, -(p @ up) / he], axis=-1).astype(np.float32)


def grid_to_nearest_pixel(grid, resolution):
    g = np.asarray(grid, np.float32)
    x = np.floor((g[..., 0] + 1.0) * 0.5 * resolution).astype(np.int64)
    y = np.floor((g[..., 1] + 1.0) * 0.5 * resolution).astype(np.int64)
    return x, y


def raster_lookup_near(pixel_linear_index, query_x, query_y, resolution, radius=3):
    pix = np.asarray(pixel_linear_index, np.int64)
    qx = np.asarray(query_x, np.int64).reshape(-1)
    qy = np.asarray(query_y, np.int64).reshape(-1)
    if len(pix) and np.any(pix[1:] < pix[:-1]):
        raise ValueError("pixel_linear_index must be sorted")
    n = len(qx)
    width = (2 * radius + 1) ** 2
    out = np.full((n, width), -1, np.int64)
    col = 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            xx = qx + dx
            yy = qy + dy
            valid = (xx >= 0) & (xx < resolution) & (yy >= 0) & (yy < resolution)
            target = yy * resolution + xx
            pos = np.searchsorted(pix, target)
            pos_clip = np.minimum(pos, max(len(pix) - 1, 0))
            hit = valid & (pos < len(pix))
            if len(pix):
                hit &= pix[pos_clip] == target
            out[hit, col] = pos[hit]
            col += 1
    return out


def choose_visible_correspondence(
    points,
    target_camera,
    target_ra,
    vertices,
    faces,
    vertex_normals,
    radius_px=3,
    max_surface_error=0.003,
):
    """Visibility/surface witness for exact continuous point projection.

    The returned target grid is the exact projection of `points`. A nearby raster-authority
    surface sample is required only as a visibility and same-surface consistency witness.
    This prevents subpixel teacher truth from being unnecessarily quantized to pixel centers.
    """
    points = np.asarray(points, np.float32)
    res = int(np.asarray(target_ra["resolution"]).reshape(-1)[0])
    grid = project_grid(points, target_camera)
    x, y = grid_to_nearest_pixel(grid, res)
    in_frame = (np.abs(grid[:, 0]) <= 1) & (np.abs(grid[:, 1]) <= 1)
    cand = raster_lookup_near(target_ra["pixel_linear_index"], x, y, res, radius_px)
    n, width = cand.shape
    valid = in_frame[:, None] & (cand >= 0)
    errmat = np.full((n, width), np.inf, np.float32)
    flat = np.flatnonzero(valid.ravel())
    if flat.size:
        qi = (flat // width).astype(np.int64)
        rows = cand.ravel()[flat]
        pp, _ = reconstruct_surface(
            vertices,
            faces,
            vertex_normals,
            target_ra["triangle_id"][rows],
            target_ra["barycentric_uv"][rows],
        )
        errmat.ravel()[flat] = np.linalg.norm(pp - points[qi], axis=1).astype(np.float32)
    best_j = np.argmin(errmat, axis=1)
    best_err = errmat[np.arange(n), best_j]
    best_row = cand[np.arange(n), best_j].astype(np.int64)
    ok = best_err <= float(max_surface_error)
    best_row[~ok] = -1
    target_grid = np.zeros((n, 2), np.float32)
    target_grid[ok] = grid[ok]
    return ok, target_grid, best_err, best_row
