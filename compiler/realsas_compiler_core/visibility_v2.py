from __future__ import annotations

"""Appearance-independent canonical surface visibility for RealSaS V2."""

from dataclasses import dataclass
import math

import numpy as np

from .hashing import content_sha256
from .mesh.product_coverage_v1 import _covers_pixel_center, _orient2d
from .playback_full_surface_v3 import project_points_xyz_v3
from .types import QualificationError

VISIBILITY_CONTRACT_V2 = {
    "schema": "RealSaS.VisibilityContract.v2",
    "authority": "CANONICAL_POSED_XYZ_PLUS_CAMERA_DEPTH",
    "projection": "FULL_SURFACE_CAMERA_PROJECTION_V3",
    "raster_fill": "HALF_INTEGER_TOP_LEFT",
    "depth": "CAMERA_FORWARD_Z_SMALLER_WINS",
    "exact_depth_tie": "SEALED_FACE_INDEX_ONLY",
    "appearance_input_forbidden": True,
    "source_provenance_tiebreak_forbidden": True,
    "texture_alpha_selects_front_surface": False,
}
VISIBILITY_CONTRACT_V2_HASH = content_sha256(VISIBILITY_CONTRACT_V2)


@dataclass(frozen=True)
class VisibilityRaster:
    owner_face_index: np.ndarray
    depth: np.ndarray
    barycentric: np.ndarray
    projected_vertices: np.ndarray
    contract_hash: str = VISIBILITY_CONTRACT_V2_HASH


def _vertex_id(vertex) -> str:
    for name in ("canonical_mesh_vertex_id", "candidate_vertex_id"):
        value = getattr(vertex, name, None)
        if value is not None:
            return str(value)
    raise QualificationError("VISIBILITY_VERTEX_ID_MISSING")


def rasterize_visible_owner(
    mesh,
    camera,
    *,
    width: int | None = None,
    height: int | None = None,
    positions=None,
) -> VisibilityRaster:
    width = int(camera.resolution if width is None else width)
    height = int(camera.resolution if height is None else height)
    if width <= 0 or height <= 0:
        raise QualificationError("VISIBILITY_DIMENSION_INVALID")

    vertex_ids = tuple(_vertex_id(vertex) for vertex in mesh.vertices)
    if len(vertex_ids) != len(set(vertex_ids)):
        raise QualificationError("VISIBILITY_DUPLICATE_VERTEX_ID")
    xyz = np.asarray(
        [tuple(map(float, vertex.P)) for vertex in mesh.vertices]
        if positions is None
        else positions,
        dtype=np.float64,
    )
    if xyz.shape != (len(vertex_ids), 3) or not np.isfinite(xyz).all():
        raise QualificationError("VISIBILITY_POSITION_MATRIX_INVALID")

    projected = np.asarray(project_points_xyz_v3(xyz, camera), dtype=np.float64)
    if projected.shape != (len(vertex_ids), 3) or not np.isfinite(projected).all():
        raise QualificationError("VISIBILITY_PROJECTED_MATRIX_INVALID")

    by_id = {vertex_ids[i]: projected[i] for i in range(len(vertex_ids))}
    owner = np.full((height, width), -1, dtype=np.int32)
    depth = np.full((height, width), np.inf, dtype=np.float64)
    barycentric = np.full((height, width, 3), np.nan, dtype=np.float32)
    tie = np.full((height, width), np.iinfo(np.int32).max, dtype=np.int32)

    for face_index, face in enumerate(mesh.faces):
        ids = tuple(map(str, face))
        if len(ids) != 3 or any(vertex_id not in by_id for vertex_id in ids):
            raise QualificationError("VISIBILITY_FACE_INVALID")
        a, b, c = (by_id[vertex_id] for vertex_id in ids)
        area = _orient2d(a, b, float(c[0]), float(c[1]))
        if abs(area) <= 1e-12:
            continue

        xs = (float(a[0]), float(b[0]), float(c[0]))
        ys = (float(a[1]), float(b[1]), float(c[1]))
        minx = max(0, int(math.floor(min(xs) - 0.5)))
        maxx = min(width - 1, int(math.ceil(max(xs) - 0.5)))
        miny = max(0, int(math.floor(min(ys) - 0.5)))
        maxy = min(height - 1, int(math.ceil(max(ys) - 0.5)))
        face_key = int(face_index)

        for y in range(miny, maxy + 1):
            for x in range(minx, maxx + 1):
                if not _covers_pixel_center(a, b, c, x, y):
                    continue
                px = float(x) + 0.5
                py = float(y) + 0.5
                w0 = _orient2d(b, c, px, py) / area
                w1 = _orient2d(c, a, px, py) / area
                w2 = _orient2d(a, b, px, py) / area
                z = (
                    w0 * float(a[2])
                    + w1 * float(b[2])
                    + w2 * float(c[2])
                )
                if not math.isfinite(z):
                    raise QualificationError("VISIBILITY_DEPTH_NONFINITE")
                current = float(depth[y, x])
                current_tie = int(tie[y, x])
                if z < current - 1e-12 or (
                    abs(z - current) <= 1e-12
                    and face_key < current_tie
                ):
                    depth[y, x] = z
                    owner[y, x] = int(face_index)
                    barycentric[y, x] = (float(w0), float(w1), float(w2))
                    tie[y, x] = face_key

    return VisibilityRaster(owner, depth, barycentric, projected)


def projected_xy_to_source_texel_xy(projected_xy) -> tuple[float, float]:
    """Map target half-integer raster coordinates to index-centered source texels."""
    return float(projected_xy[0]) - 0.5, float(projected_xy[1]) - 0.5


def nearest_source_texel_index(
    source_xy,
    *,
    width: int,
    height: int,
) -> tuple[int, int] | None:
    x, y = map(float, source_xy)
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    ix = int(round(x))
    iy = int(round(y))
    if ix < 0 or ix >= int(width) or iy < 0 or iy >= int(height):
        return None
    return ix, iy


def sample_is_first_hit(
    visibility: VisibilityRaster,
    *,
    face_index: int,
    projected_xy,
) -> bool:
    source_xy = projected_xy_to_source_texel_xy(projected_xy)
    index = nearest_source_texel_index(
        source_xy,
        width=int(visibility.owner_face_index.shape[1]),
        height=int(visibility.owner_face_index.shape[0]),
    )
    if index is None:
        return False
    x, y = index
    return int(visibility.owner_face_index[y, x]) == int(face_index)
