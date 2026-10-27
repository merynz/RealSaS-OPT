"""Consumer-neutral geometric ray-order evidence for the Graphics oracle court.

This module measures complete orthographic ray/surface intersections on a fixed
triangulated mesh. It deliberately does *not* accept semantic layer labels,
material names, owner ids, alpha, colour, attachment labels or a global draw
order. Appearance truth remains a separate qualification problem.

The output is suitable for a teacher-oracle court that asks a narrower question:
can measured local ray order drive a fixed Compiler consequence without
fabricating a per-view total layer order?
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
import math

import numpy as np


@dataclass(frozen=True)
class OrthographicCameraV1:
    origin: tuple[float, float, float]
    forward: tuple[float, float, float]
    right: tuple[float, float, float]
    screen_up: tuple[float, float, float]
    half_extent: float
    resolution: int
    view_index: int


@dataclass(frozen=True)
class GeometricRayHitV1:
    face_index: int
    barycentric: tuple[float, float, float]
    depth: float


@dataclass(frozen=True)
class GeometricRayStackV1:
    view_index: int
    pixel_x: int
    pixel_y: int
    hits: tuple[GeometricRayHitV1, ...]
    complete: bool
    has_depth_tie: bool
    tie_epsilon: float


def _finite_xyz(value, code: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if out.shape != (3,) or not np.isfinite(out).all():
        raise ValueError(code)
    return out


def camera_from_mapping(value: Mapping[str, object] | object) -> OrthographicCameraV1:
    """Normalize the frozen product camera payload without depending on its class."""
    def get(name: str):
        if isinstance(value, Mapping):
            return value[name]
        return getattr(value, name)

    origin = _finite_xyz(get("origin"), "GRAPHICS_ORACLE_CAMERA_ORIGIN")
    forward = _finite_xyz(get("forward"), "GRAPHICS_ORACLE_CAMERA_FORWARD")
    right = _finite_xyz(get("right"), "GRAPHICS_ORACLE_CAMERA_RIGHT")
    screen_up = _finite_xyz(get("screen_up"), "GRAPHICS_ORACLE_CAMERA_UP")
    half_extent = float(get("half_extent"))
    resolution = int(get("resolution"))
    view_index = int(get("view_index"))
    if not math.isfinite(half_extent) or half_extent <= 0 or resolution <= 0 or view_index < 0:
        raise ValueError("GRAPHICS_ORACLE_CAMERA_SCALAR_CONTRACT")
    if not (
        np.isclose(np.linalg.norm(forward), 1.0, atol=1e-8, rtol=0)
        and np.isclose(np.linalg.norm(right), 1.0, atol=1e-8, rtol=0)
        and np.isclose(np.linalg.norm(screen_up), 1.0, atol=1e-8, rtol=0)
        and abs(float(np.dot(forward, right))) <= 1e-8
        and abs(float(np.dot(forward, screen_up))) <= 1e-8
        and abs(float(np.dot(right, screen_up))) <= 1e-8
    ):
        raise ValueError("GRAPHICS_ORACLE_CAMERA_ORTHONORMAL_REQUIRED")
    return OrthographicCameraV1(
        origin=tuple(map(float, origin)),
        forward=tuple(map(float, forward)),
        right=tuple(map(float, right)),
        screen_up=tuple(map(float, screen_up)),
        half_extent=half_extent,
        resolution=resolution,
        view_index=view_index,
    )


def _mesh(vertices, faces):
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)
    if v.ndim != 2 or v.shape[1:] != (3,) or not len(v) or not np.isfinite(v).all():
        raise ValueError("GRAPHICS_ORACLE_FINITE_XYZ_MESH_REQUIRED")
    if f.ndim != 2 or f.shape[1:] != (3,) or not len(f) or not np.issubdtype(f.dtype, np.integer):
        raise ValueError("GRAPHICS_ORACLE_TRIANGULATED_MESH_REQUIRED")
    if np.min(f) < 0 or np.max(f) >= len(v):
        raise ValueError("GRAPHICS_ORACLE_FACE_INDEX_INVALID")
    return v, f.astype(np.int64, copy=False)


def _pixel_ray(camera: OrthographicCameraV1, pixel_x: int, pixel_y: int):
    if type(pixel_x) is not int or type(pixel_y) is not int:
        raise ValueError("GRAPHICS_ORACLE_INTEGER_PIXEL_REQUIRED")
    if not (0 <= pixel_x < camera.resolution and 0 <= pixel_y < camera.resolution):
        raise ValueError("GRAPHICS_ORACLE_PIXEL_OUT_OF_RANGE")
    x = ((pixel_x + 0.5) / camera.resolution * 2.0 - 1.0) * camera.half_extent
    # Product image coordinates are top-down; screen_up is geometrically upward.
    y = (1.0 - (pixel_y + 0.5) / camera.resolution * 2.0) * camera.half_extent
    origin = (
        np.asarray(camera.origin)
        + np.asarray(camera.right) * x
        + np.asarray(camera.screen_up) * y
    )
    return origin, np.asarray(camera.forward)


def measure_orthographic_ray_stack_v1(
    vertices,
    faces,
    camera: OrthographicCameraV1 | Mapping[str, object] | object,
    pixel_x: int,
    pixel_y: int,
    *,
    tie_epsilon: float,
    intersection_epsilon: float = 1e-10,
) -> GeometricRayStackV1:
    """Return every forward ray/triangle hit, depth-sorted, with barycentrics.

    No back-face culling is applied. Shared-edge/coincident intersections are
    preserved and marked as ties rather than resolved with face id. ``complete``
    means the declared camera plane is in front of the entire mesh; otherwise the
    court must not claim a complete stack.
    """
    verts, tri_index = _mesh(vertices, faces)
    cam = camera if isinstance(camera, OrthographicCameraV1) else camera_from_mapping(camera)
    if not math.isfinite(tie_epsilon) or tie_epsilon < 0:
        raise ValueError("GRAPHICS_ORACLE_TIE_EPSILON_INVALID")
    if not math.isfinite(intersection_epsilon) or intersection_epsilon <= 0:
        raise ValueError("GRAPHICS_ORACLE_INTERSECTION_EPSILON_INVALID")

    ray_origin, direction = _pixel_ray(cam, pixel_x, pixel_y)
    # A forward-only stack is complete only if the camera plane is strictly
    # before all mesh vertices in the declared forward direction.
    vertex_depth = (verts - ray_origin) @ direction
    complete = bool(np.min(vertex_depth) > intersection_epsilon)

    tri = verts[tri_index]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    pvec = np.cross(np.broadcast_to(direction, e2.shape), e2)
    det = np.einsum("ij,ij->i", e1, pvec)
    nonsingular = np.abs(det) > intersection_epsilon
    inv_det = np.zeros_like(det)
    inv_det[nonsingular] = 1.0 / det[nonsingular]

    tvec = ray_origin[None, :] - tri[:, 0]
    u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
    qvec = np.cross(tvec, e1)
    vv = qvec @ direction * inv_det
    depth = np.einsum("ij,ij->i", e2, qvec) * inv_det
    hit = (
        nonsingular
        & (u >= -intersection_epsilon)
        & (vv >= -intersection_epsilon)
        & (u + vv <= 1.0 + intersection_epsilon)
        & (depth >= intersection_epsilon)
    )
    indices = np.flatnonzero(hit)
    if not len(indices):
        return GeometricRayStackV1(
            view_index=cam.view_index,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            hits=(),
            complete=complete,
            has_depth_tie=False,
            tie_epsilon=float(tie_epsilon),
        )

    order = indices[np.argsort(depth[indices], kind="stable")]
    rows = []
    for face_index in order:
        b1 = float(u[face_index])
        b2 = float(vv[face_index])
        b0 = 1.0 - b1 - b2
        # Numerical boundary noise is clipped only within the accepted
        # intersection tolerance; the normalized barycentric still sums to one.
        bary = np.clip(np.asarray([b0, b1, b2]), 0.0, 1.0)
        bary /= bary.sum()
        rows.append(
            GeometricRayHitV1(
                face_index=int(face_index),
                barycentric=tuple(map(float, bary)),
                depth=float(depth[face_index]),
            )
        )
    has_tie = any(
        b.depth - a.depth <= tie_epsilon
        for a, b in zip(rows, rows[1:])
    )
    return GeometricRayStackV1(
        view_index=cam.view_index,
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        hits=tuple(rows),
        complete=complete,
        has_depth_tie=has_tie,
        tie_epsilon=float(tie_epsilon),
    )


def face_centroid_candidate_pixels_v1(
    vertices,
    faces,
    camera: OrthographicCameraV1 | Mapping[str, object] | object,
    *,
    maximum: int = 256,
) -> tuple[tuple[int, int], ...]:
    """Deterministic geometry-only ray proposal set for an oracle audit.

    Selection is not truth: it only chooses pixels to measure. No material,
    semantic part, owner, source image or depth order is used.
    """
    verts, tri_index = _mesh(vertices, faces)
    cam = camera if isinstance(camera, OrthographicCameraV1) else camera_from_mapping(camera)
    if type(maximum) is not int or maximum < 1:
        raise ValueError("GRAPHICS_ORACLE_MAXIMUM_INVALID")
    centres = verts[tri_index].mean(axis=1)
    delta = centres - np.asarray(cam.origin)
    sx = delta @ np.asarray(cam.right)
    sy = delta @ np.asarray(cam.screen_up)
    px = np.floor(((sx / cam.half_extent + 1.0) * 0.5) * cam.resolution).astype(np.int64)
    py = np.floor(((1.0 - sy / cam.half_extent) * 0.5) * cam.resolution).astype(np.int64)
    inside = (px >= 0) & (px < cam.resolution) & (py >= 0) & (py < cam.resolution)
    candidates = sorted(set((int(x), int(y)) for x, y in zip(px[inside], py[inside])))
    if len(candidates) <= maximum:
        return tuple(candidates)
    # Evenly cover the deterministic sorted candidate list; no outcome-driven
    # threshold or positive-example mining is permitted here.
    pick = np.linspace(0, len(candidates) - 1, maximum, dtype=np.int64)
    return tuple(candidates[int(i)] for i in pick)
