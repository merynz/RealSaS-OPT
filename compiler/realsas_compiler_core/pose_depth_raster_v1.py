from __future__ import annotations

"""Pose-local full-frame geometric visibility for one canonical 3D mesh.

Reuses the exact barycentric + smaller-Z-wins semantics of
inspect_pose_depth_pixel_v1. This is a depth ownership map, NOT an appearance
render, source-ray witness, pose-independent drawable order, or learned
hidden-surface texture. Coplanar first-hit ties remain AMBIGUOUS (-2).
No alpha, CAA, missing backside material or source visibility is invented.
"""

from dataclasses import asdict, dataclass
import math
from typing import Sequence

from .hashing import content_sha256
from .pose_depth_visibility_v1 import _barycentric
from .types import QualificationError


@dataclass(frozen=True)
class PoseDepthRasterV1:
    view_id: str
    graphics_hash: str
    pose_projection_hash: str
    width: int
    height: int
    nearest_depth_by_pixel: tuple[tuple[float | None, ...], ...]
    front_face_by_pixel: tuple[tuple[int, ...], ...]
    front_fragment_by_pixel: tuple[tuple[str | None, ...], ...]
    unique_front_pixel_count: int
    ambiguous_front_pixel_count: int
    empty_pixel_count: int
    face_pixel_test_budget_used: int
    raster_hash: str
    schema: str = "RealSaS.PoseDepthRaster.v1"
    geometric_front_only: bool = True
    source_visibility_qualified: bool = False
    source_alpha_qualified: bool = False
    source_hidden_appearance_qualified: bool = False
    pose_appearance_authority: bool = False
    product_revision_minted: bool = False

    def to_dict(self):
        return asdict(self)


def _screen_bounds(face, width: int, height: int):
    xs, ys = [v[0] for v in face], [v[1] for v in face]
    left = max(0, math.ceil(min(xs) - .5))
    right = min(width-1, math.floor(max(xs) - .5))
    top = max(0, math.ceil(min(ys) - .5))
    bottom = min(height-1, math.floor(max(ys) - .5))
    if left > right or top > bottom:
        return None
    return left, right, top, bottom


def rasterize_pose_depth_v1(
    *, view_id: str, graphics_hash: str,
    projected_faces: Sequence[Sequence[Sequence[float]]],
    face_fragment_ids: Sequence[str],
    width: int, height: int,
    depth_tolerance: float = 1e-6,
    max_face_pixel_tests: int = 4_000_000,
) -> PoseDepthRasterV1:
    if (
        type(view_id) is not str or not view_id
        or type(graphics_hash) is not str or not graphics_hash
        or type(width) is not int or not 1 <= width <= 256
        or type(height) is not int or not 1 <= height <= 256
        or type(max_face_pixel_tests) is not int or not 1 <= max_face_pixel_tests <= 4_000_000
        or type(depth_tolerance) not in (int, float)
        or not math.isfinite(depth_tolerance) or not 0 <= depth_tolerance <= .01
        or len(projected_faces) != len(face_fragment_ids)
        or not 1 <= len(projected_faces) <= 2_000_000
        or any(type(ref) is not str or not ref for ref in face_fragment_ids)
    ):
        raise QualificationError("POSE_DEPTH_RASTER_INPUT_DOMAIN_INVALID")
    faces = []
    for raw in projected_faces:
        if len(raw) != 3:
            raise QualificationError("POSE_DEPTH_RASTER_TRIANGLE_REQUIRED")
        xyz = []
        for vertex in raw:
            if len(vertex) != 3 or any(
                type(v) not in (float, int) or not math.isfinite(v)
                for v in vertex
            ):
                raise QualificationError("POSE_DEPTH_RASTER_VERTEX_DOMAIN_INVALID")
            xyz.append(tuple(float(v) for v in vertex))
        faces.append(tuple(xyz))
    pose_hash = content_sha256({
        "schema": "RealSaS.PoseGraphicsProjection.v1",
        "view_id": view_id,
        "graphics_hash": graphics_hash,
        "triangles": faces,
        "face_fragment_ids": tuple(face_fragment_ids),
    })

    # Check work *before* rasterization. Large full-screen overlapping
    # triangles otherwise make a bounded 256x256 target unexpectedly costly.
    bounds = [_screen_bounds(face, width, height) for face in faces]
    total_tests = sum(
        (right-left+1) * (bottom-top+1)
        for rect in bounds if rect is not None
        for left, right, top, bottom in (rect,)
    )
    if total_tests > max_face_pixel_tests:
        raise QualificationError("POSE_DEPTH_RASTER_FACE_PIXEL_WORK_BUDGET")
    depth = [[math.inf] * width for _ in range(height)]
    for face, rect in zip(faces, bounds):
        if rect is None:
            continue
        left, right, top, bottom = rect
        for y in range(top, bottom+1):
            for x in range(left, right+1):
                weights = _barycentric(x+.5, y+.5, face)
                if weights is None:
                    continue
                z = sum(w * v[2] for w, v in zip(weights, face))
                if not math.isfinite(z):
                    raise QualificationError("POSE_DEPTH_RASTER_INTERPOLATION_NONFINITE")
                if z < depth[y][x]:
                    depth[y][x] = float(z)

    # Resolve ties against the FINAL minimum, not whichever triangle happened
    # to be visited first. Therefore array order is never a z-order policy.
    owner = [[-1] * width for _ in range(height)]
    for fi, (face, rect) in enumerate(zip(faces, bounds)):
        if rect is None:
            continue
        left, right, top, bottom = rect
        for y in range(top, bottom+1):
            for x in range(left, right+1):
                if not math.isfinite(depth[y][x]):
                    continue
                weights = _barycentric(x+.5, y+.5, face)
                if weights is None:
                    continue
                z = sum(w * v[2] for w, v in zip(weights, face))
                if z - depth[y][x] <= depth_tolerance:
                    owner[y][x] = fi if owner[y][x] == -1 else -2
    unique = sum(v >= 0 for row in owner for v in row)
    ambiguous = sum(v == -2 for row in owner for v in row)
    empty = width * height - unique - ambiguous
    payload = {
        "view_id": view_id,
        "graphics_hash": graphics_hash,
        "pose_projection_hash": pose_hash,
        "width": width,
        "height": height,
        "nearest_depth_by_pixel": tuple(
            tuple(None if not math.isfinite(z) else z for z in row) for row in depth
        ),
        "front_face_by_pixel": tuple(tuple(row) for row in owner),
        "front_fragment_by_pixel": tuple(
            tuple(face_fragment_ids[i] if i >= 0 else None for i in row) for row in owner
        ),
        "unique_front_pixel_count": unique,
        "ambiguous_front_pixel_count": ambiguous,
        "empty_pixel_count": empty,
        "face_pixel_test_budget_used": total_tests,
    }
    draft = PoseDepthRasterV1(**payload, raster_hash="")
    return PoseDepthRasterV1(**payload, raster_hash=content_sha256({
        k: v for k, v in draft.to_dict().items() if k != "raster_hash"
    }))


__all__ = ["PoseDepthRasterV1", "rasterize_pose_depth_v1"]
