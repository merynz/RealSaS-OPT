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


def _rasterize_visible_owner_legacy(
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
    """Vectorized exact implementation of VisibilityContract.v2.

    Face order remains authoritative. Coverage, barycentrics and K-layer
    insertion preserve the legacy scalar comparisons exactly while replacing
    per-pixel Python loops with NumPy batches inside each face bounding box.
    """
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

    projected = np.asarray(
        project_points_xyz_v3(xyz, camera), dtype=np.float64
    )
    if (
        projected.shape != (len(vertex_ids), 3)
        or not np.isfinite(projected).all()
    ):
        raise QualificationError("VISIBILITY_PROJECTED_MATRIX_INVALID")
    raster_projected = projected.copy()
    raster_projected[:, :2] *= float(coverage_scale)
    by_id = {
        vertex_ids[i]: raster_projected[i]
        for i in range(len(vertex_ids))
    }

    layer_owner = np.full(
        (height, width, max_layers), -1, dtype=np.int32
    )
    layer_depth = np.full(
        (height, width, max_layers), np.inf, dtype=np.float64
    )
    layer_barycentric = np.full(
        (height, width, max_layers, 3), np.nan, dtype=np.float32
    )
    layer_overflow = np.zeros((height, width), dtype=bool)
    eps = VISIBILITY_DEPTH_EQUIVALENCE_EPSILON

    def orient_array(a, b, px, py):
        return (
            (float(b[0]) - float(a[0]))
            * (py - float(a[1]))
            - (float(b[1]) - float(a[1]))
            * (px - float(a[0]))
        )

    def edge_accept_array(edge, top_left: bool):
        return (edge > eps) | (
            (np.abs(edge) <= eps) & bool(top_left)
        )

    for face_index, face in enumerate(mesh.faces):
        ids = tuple(map(str, face))
        if (
            len(ids) != 3
            or any(vertex_id not in by_id for vertex_id in ids)
        ):
            raise QualificationError("VISIBILITY_FACE_INVALID")
        a, b, c0 = (by_id[vertex_id] for vertex_id in ids)
        area = _orient2d(a, b, float(c0[0]), float(c0[1]))
        if abs(area) <= eps:
            continue

        xs = (float(a[0]), float(b[0]), float(c0[0]))
        ys = (float(a[1]), float(b[1]), float(c0[1]))
        minx = max(0, int(math.floor(min(xs) - 0.5)))
        maxx = min(width - 1, int(math.ceil(max(xs) - 0.5)))
        miny = max(0, int(math.floor(min(ys) - 0.5)))
        maxy = min(height - 1, int(math.ceil(max(ys) - 0.5)))
        if minx > maxx or miny > maxy:
            continue

        grid_y, grid_x = np.mgrid[
            miny : maxy + 1,
            minx : maxx + 1,
        ]
        flat_y = grid_y.reshape(-1)
        flat_x = grid_x.reshape(-1)
        px = flat_x.astype(np.float64) + 0.5
        py = flat_y.astype(np.float64) + 0.5

        positive = area > 0.0
        sign = 1.0 if positive else -1.0
        e0 = sign * orient_array(b, c0, px, py)
        e1 = sign * orient_array(c0, a, px, py)
        e2 = sign * orient_array(a, b, px, py)
        tl0 = _is_top_left(b, c0) if positive else _is_top_left(c0, b)
        tl1 = _is_top_left(c0, a) if positive else _is_top_left(a, c0)
        tl2 = _is_top_left(a, b) if positive else _is_top_left(b, a)
        covered = (
            edge_accept_array(e0, tl0)
            & edge_accept_array(e1, tl1)
            & edge_accept_array(e2, tl2)
        )
        if not np.any(covered):
            continue

        yy = flat_y[covered]
        xx = flat_x[covered]
        covered_px = px[covered]
        covered_py = py[covered]
        w0 = orient_array(b, c0, covered_px, covered_py) / area
        w1 = orient_array(c0, a, covered_px, covered_py) / area
        w2 = orient_array(a, b, covered_px, covered_py) / area
        z = (
            w0 * float(a[2])
            + w1 * float(b[2])
            + w2 * float(c0[2])
        )
        if not np.isfinite(z).all():
            raise QualificationError("VISIBILITY_DEPTH_NONFINITE")
        positive_depth = z > eps
        if not np.any(positive_depth):
            continue
        yy = yy[positive_depth]
        xx = xx[positive_depth]
        z = z[positive_depth]
        bary = np.stack(
            (
                w0[positive_depth],
                w1[positive_depth],
                w2[positive_depth],
            ),
            axis=1,
        ).astype(np.float32)

        old_owner = layer_owner[yy, xx].copy()
        old_depth = layer_depth[yy, xx].copy()
        old_bary = layer_barycentric[yy, xx].copy()
        face_key = int(face_index)
        better = (
            (old_owner < 0)
            | (z[:, None] < old_depth - eps)
            | (
                (np.abs(z[:, None] - old_depth) <= eps)
                & (face_key < old_owner)
            )
        )
        has_insert = np.any(better, axis=1)
        insert_at = np.argmax(better, axis=1)
        layer_overflow[yy, xx] |= old_owner[:, -1] >= 0

        if not np.any(has_insert):
            continue
        new_owner = old_owner.copy()
        new_depth = old_depth.copy()
        new_bary = old_bary.copy()
        for layer in range(max_layers):
            take = has_insert & (insert_at == layer)
            if not np.any(take):
                continue
            if layer < max_layers - 1:
                new_owner[take, layer + 1 :] = old_owner[
                    take, layer:-1
                ]
                new_depth[take, layer + 1 :] = old_depth[
                    take, layer:-1
                ]
                new_bary[take, layer + 1 :] = old_bary[
                    take, layer:-1
                ]
            new_owner[take, layer] = face_key
            new_depth[take, layer] = z[take]
            new_bary[take, layer] = bary[take]

        layer_owner[yy, xx] = new_owner
        layer_depth[yy, xx] = new_depth
        layer_barycentric[yy, xx] = new_bary

    owner = layer_owner[:, :, 0].copy()
    depth = layer_depth[:, :, 0].copy()
    barycentric = layer_barycentric[:, :, 0].copy()
    second_owner = layer_owner[:, :, 1].copy()
    second_depth = layer_depth[:, :, 1].copy()
    margin = np.full((height, width), np.inf, dtype=np.float64)
    has_second = second_owner >= 0
    margin[has_second] = (
        second_depth[has_second] - depth[has_second]
    )
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
