from __future__ import annotations

"""Appearance-independent canonical surface visibility for RealSaS V2."""

from dataclasses import dataclass
import math

import numpy as np

from .hashing import content_sha256
from .mesh.product_coverage_v1 import _covers_pixel_center, _orient2d
from .camera_geometry_v2 import project_points_xyz_v3
from .types import QualificationError


VISIBILITY_DEPTH_EQUIVALENCE_EPSILON = 1.0e-12

VISIBILITY_CONTRACT_V2 = {
    "schema": "RealSaS.VisibilityContract.v2",
    "authority": "CANONICAL_POSED_XYZ_PLUS_CAMERA_DEPTH",
    "projection": "FULL_SURFACE_CAMERA_PROJECTION_V3",
    "raster_fill": "HALF_INTEGER_TOP_LEFT",
    "shipping_pixel_coverage": "FIXED_2X2_QUARTER_SUBSAMPLES__LINEAR_PM_AVERAGE",
    "depth": "CAMERA_FORWARD_Z_SMALLER_WINS",
    "depth_buffer": "IEEE754_FLOAT64_SOFTWARE_SORT__NO_HARDWARE_Z_QUANTIZATION",
    "depth_equivalence_epsilon_camera_z": VISIBILITY_DEPTH_EQUIVALENCE_EPSILON,
    "exact_depth_tie": "SEALED_FACE_INDEX_ONLY__AMBIGUITY_MUST_BE_QUALIFIED",
    "near_depth_policy": "ABS_DELTA_LE_EPSILON_IS_AMBIGUOUS__OUTSIDE_EPSILON_PHYSICAL_DEPTH_IS_AUTHORITY",
    "layering": "DEPTH_SORTED_K4_GEOMETRY_LAYERS",
    "alpha_composition": "LINEAR_PREMULTIPLIED_FRONT_TO_BACK",
    "layer_overflow": "FORBIDDEN",
    "near_plane": "CAMERA_FORWARD_Z_MUST_BE_POSITIVE",
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
    second_owner_face_index: np.ndarray
    second_depth: np.ndarray
    depth_margin: np.ndarray
    layer_owner_face_index: np.ndarray
    layer_depth: np.ndarray
    layer_barycentric: np.ndarray
    layer_overflow: np.ndarray
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
    max_layers: int = 4,
    coverage_scale: int = 1,
) -> VisibilityRaster:
    base_width = int(camera.resolution if width is None else width)
    base_height = int(camera.resolution if height is None else height)
    coverage_scale = int(coverage_scale)
    width = base_width * coverage_scale
    height = base_height * coverage_scale
    max_layers = int(max_layers)
    if (
        base_width <= 0
        or base_height <= 0
        or coverage_scale <= 0
        or max_layers < 2
    ):
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
    raster_projected = projected.copy()
    raster_projected[:, :2] *= float(coverage_scale)

    by_id = {
        vertex_ids[i]: raster_projected[i]
        for i in range(len(vertex_ids))
    }
    layer_owner = np.full(
        (height, width, max_layers),
        -1,
        dtype=np.int32,
    )
    layer_depth = np.full(
        (height, width, max_layers),
        np.inf,
        dtype=np.float64,
    )
    layer_barycentric = np.full(
        (height, width, max_layers, 3),
        np.nan,
        dtype=np.float32,
    )
    layer_overflow = np.zeros((height, width), dtype=bool)
    eps = VISIBILITY_DEPTH_EQUIVALENCE_EPSILON

    for face_index, face in enumerate(mesh.faces):
        ids = tuple(map(str, face))
        if len(ids) != 3 or any(vertex_id not in by_id for vertex_id in ids):
            raise QualificationError("VISIBILITY_FACE_INVALID")
        a, b, c = (by_id[vertex_id] for vertex_id in ids)
        area = _orient2d(a, b, float(c[0]), float(c[1]))
        if abs(area) <= eps:
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
                if z <= eps:
                    continue

                insert_at = None
                for layer in range(max_layers):
                    current_owner = int(layer_owner[y, x, layer])
                    current_depth = float(layer_depth[y, x, layer])
                    if (
                        current_owner < 0
                        or z < current_depth - eps
                        or (
                            abs(z - current_depth) <= eps
                            and face_key < current_owner
                        )
                    ):
                        insert_at = layer
                        break

                if insert_at is None:
                    layer_overflow[y, x] = True
                    continue
                if int(layer_owner[y, x, max_layers - 1]) >= 0:
                    layer_overflow[y, x] = True
                for layer in range(max_layers - 1, insert_at, -1):
                    layer_owner[y, x, layer] = layer_owner[y, x, layer - 1]
                    layer_depth[y, x, layer] = layer_depth[y, x, layer - 1]
                    layer_barycentric[y, x, layer] = layer_barycentric[
                        y, x, layer - 1
                    ]
                layer_owner[y, x, insert_at] = face_key
                layer_depth[y, x, insert_at] = z
                layer_barycentric[y, x, insert_at] = (
                    float(w0),
                    float(w1),
                    float(w2),
                )

    owner = layer_owner[:, :, 0].copy()
    depth = layer_depth[:, :, 0].copy()
    barycentric = layer_barycentric[:, :, 0].copy()
    second_owner = layer_owner[:, :, 1].copy()
    second_depth = layer_depth[:, :, 1].copy()
    margin = np.full((height, width), np.inf, dtype=np.float64)
    has_second = second_owner >= 0
    margin[has_second] = second_depth[has_second] - depth[has_second]
    if np.any(margin[has_second] < -1e-10):
        raise QualificationError("VISIBILITY_SECOND_DEPTH_ORDER_INVALID")

    return VisibilityRaster(
        owner_face_index=owner,
        depth=depth,
        barycentric=barycentric,
        projected_vertices=projected,
        second_owner_face_index=second_owner,
        second_depth=second_depth,
        depth_margin=margin,
        layer_owner_face_index=layer_owner,
        layer_depth=layer_depth,
        layer_barycentric=layer_barycentric,
        layer_overflow=layer_overflow,
    )

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
