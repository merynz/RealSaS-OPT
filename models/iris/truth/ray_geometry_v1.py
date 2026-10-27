"""Complete pixel-center ray geometry over exact source triangles.

This producer measures only geometric ray incidence/order. It does not infer
semantic foreground/background, visibility ownership, material opacity, colour,
or a global layer order. Appearance sampling is a separate qualification
surface.

The camera is the already-qualified exact observation camera. A ray that lands
on a projected triangle boundary abstains because tessellation ownership at that
sample is not unique under this pure geometric raster policy. Interior hits are
sorted by source depth; depth ties remain explicit rather than being broken by
face id.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from compiler.realsas_compiler_core.camera_geometry_v2 import (
    CameraProjectionV3,
    project_points_xyz_v3,
)


@dataclass(frozen=True)
class RayGeometryHitV1:
    face_index: int
    barycentric: tuple[float, float, float]
    depth: float


@dataclass(frozen=True)
class PixelRayGeometryV1:
    view_index: int
    pixel_x: int
    pixel_y: int
    hits: tuple[RayGeometryHitV1, ...]
    complete: bool
    measurement_valid: bool
    tie_groups: tuple[tuple[int, ...], ...]
    reason: str


def _validated_geometry(vertices, faces):
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)
    if v.ndim != 2 or v.shape[1:] != (3,) or not np.isfinite(v).all():
        raise ValueError("RAY_GEOMETRY_FINITE_XYZ_REQUIRED")
    if f.ndim != 2 or f.shape[1:] != (3,) or not np.issubdtype(f.dtype, np.integer):
        raise ValueError("RAY_GEOMETRY_TRIANGLES_REQUIRED")
    if f.size == 0 or np.min(f) < 0 or np.max(f) >= len(v):
        raise ValueError("RAY_GEOMETRY_FACE_INDEX_INVALID")
    return v, f.astype(np.int64, copy=False)


def _validated_pixels(pixels, resolution: int) -> np.ndarray:
    p = np.asarray(tuple(pixels))
    if p.ndim != 2 or p.shape[1:] != (2,) or not np.issubdtype(p.dtype, np.integer):
        raise ValueError("RAY_GEOMETRY_INTEGER_PIXELS_REQUIRED")
    if len(p) == 0:
        raise ValueError("RAY_GEOMETRY_NONEMPTY_PIXEL_SET_REQUIRED")
    if np.min(p) < 0 or np.max(p) >= int(resolution):
        raise ValueError("RAY_GEOMETRY_PIXEL_OUT_OF_RANGE")
    if len({(int(x), int(y)) for x, y in p}) != len(p):
        raise ValueError("RAY_GEOMETRY_DUPLICATE_PIXEL")
    return p.astype(np.int64, copy=False)


def _tie_groups(depths: np.ndarray, epsilon: float) -> tuple[tuple[int, ...], ...]:
    if len(depths) < 2:
        return ()
    groups = []
    start = 0
    for index in range(1, len(depths)):
        if depths[index] - depths[index - 1] > epsilon:
            if index - start > 1:
                groups.append(tuple(range(start, index)))
            start = index
    if len(depths) - start > 1:
        groups.append(tuple(range(start, len(depths))))
    return tuple(groups)


def measure_pixel_ray_geometry_v1(
    vertices,
    faces,
    camera: CameraProjectionV3,
    pixels: Iterable[tuple[int, int]],
    *,
    projected_area_epsilon_px2: float = 1.0e-10,
    boundary_barycentric_epsilon: float = 1.0e-9,
    depth_tie_epsilon_source: float = 1.0e-8,
) -> tuple[PixelRayGeometryV1, ...]:
    """Measure all triangle interiors intersecting declared pixel-center rays.

    ``complete`` means complete under this explicit pixel-center, triangle
    interior raster policy. Projected zero-area triangles have zero interior
    coverage and do not contribute. If the pixel center is on a nondegenerate
    triangle boundary, completeness is withheld rather than assigning the
    boundary to one triangle by an arbitrary face-id rule.
    """
    v, f = _validated_geometry(vertices, faces)
    if not isinstance(camera, CameraProjectionV3):
        raise ValueError("RAY_GEOMETRY_QUALIFIED_CAMERA_REQUIRED")
    p = _validated_pixels(pixels, camera.resolution)
    tolerances = (
        projected_area_epsilon_px2,
        boundary_barycentric_epsilon,
        depth_tie_epsilon_source,
    )
    if any(not np.isfinite(x) or x < 0 for x in tolerances):
        raise ValueError("RAY_GEOMETRY_NONNEGATIVE_FINITE_TOLERANCE_REQUIRED")

    projected = project_points_xyz_v3(v, camera)
    sample = p.astype(np.float64) + 0.5
    hit_rows: list[list[RayGeometryHitV1]] = [[] for _ in range(len(p))]
    boundary_ambiguous = np.zeros(len(p), dtype=bool)

    for face_index, face in enumerate(f):
        tri = projected[face]
        x0, y0 = tri[0, :2]
        x1, y1 = tri[1, :2]
        x2, y2 = tri[2, :2]
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(float(denom)) <= projected_area_epsilon_px2:
            # Zero projected area has no pixel-center interior under this policy.
            continue

        sx = sample[:, 0]
        sy = sample[:, 1]
        b0 = ((y1 - y2) * (sx - x2) + (x2 - x1) * (sy - y2)) / denom
        b1 = ((y2 - y0) * (sx - x2) + (x0 - x2) * (sy - y2)) / denom
        b2 = 1.0 - b0 - b1
        bary = np.stack((b0, b1, b2), axis=1)

        in_closed = np.all(bary >= -boundary_barycentric_epsilon, axis=1) & np.all(
            bary <= 1.0 + boundary_barycentric_epsilon, axis=1
        )
        near_boundary = in_closed & np.any(
            np.abs(bary) <= boundary_barycentric_epsilon, axis=1
        )
        boundary_ambiguous |= near_boundary
        interior = in_closed & ~near_boundary & np.all(
            bary < 1.0 - boundary_barycentric_epsilon, axis=1
        )

        for ray_index in np.flatnonzero(interior):
            weights = bary[ray_index]
            depth = float(np.dot(weights, tri[:, 2]))
            if not np.isfinite(depth):
                raise ValueError("RAY_GEOMETRY_NONFINITE_INTERPOLATED_DEPTH")
            hit_rows[ray_index].append(
                RayGeometryHitV1(
                    face_index=int(face_index),
                    barycentric=tuple(float(x) for x in weights),
                    depth=depth,
                )
            )

    out = []
    for ray_index, (pixel_x, pixel_y) in enumerate(p):
        ordered = tuple(sorted(hit_rows[ray_index], key=lambda hit: hit.depth))
        depths = np.asarray([hit.depth for hit in ordered], dtype=np.float64)
        ties = _tie_groups(depths, float(depth_tie_epsilon_source))
        if boundary_ambiguous[ray_index]:
            complete = False
            valid = False
            reason = "PROJECTED_TRIANGLE_BOUNDARY_AMBIGUOUS"
        elif ties:
            complete = True
            valid = True
            reason = "COMPLETE_WITH_DEPTH_TIE"
        else:
            complete = True
            valid = True
            reason = "COMPLETE"
        out.append(
            PixelRayGeometryV1(
                view_index=int(camera.view_index),
                pixel_x=int(pixel_x),
                pixel_y=int(pixel_y),
                hits=ordered,
                complete=complete,
                measurement_valid=valid,
                tie_groups=ties,
                reason=reason,
            )
        )
    return tuple(out)


def regular_pixel_grid_v1(resolution: int, *, grid_size: int = 32) -> tuple[tuple[int, int], ...]:
    """Deterministic stratified pixel-center probe set, independent of scene labels."""
    if type(resolution) is not int or resolution <= 0:
        raise ValueError("RAY_GEOMETRY_RESOLUTION_INVALID")
    if type(grid_size) is not int or grid_size <= 0 or grid_size > resolution:
        raise ValueError("RAY_GEOMETRY_GRID_SIZE_INVALID")
    bins = np.linspace(0, resolution, grid_size + 1, dtype=np.float64)
    centers = np.floor((bins[:-1] + bins[1:]) * 0.5).astype(np.int64)
    centers = np.clip(centers, 0, resolution - 1)
    return tuple((int(x), int(y)) for y in centers for x in centers)
