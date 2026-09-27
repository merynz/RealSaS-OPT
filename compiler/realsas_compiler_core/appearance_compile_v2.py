from __future__ import annotations

"""Deterministic mesh-domain Complete Appearance Authority compiler."""

import math
from typing import Mapping

import numpy as np
from .appearance_authority_v2 import CAA_PROVENANCE
from .appearance_color_v2 import (
    bilinear_premultiplied_linear_rgba,
    premultiplied_linear_to_straight_srgb_u8,
    straight_srgb_rgba_u8_to_premultiplied_linear,
)
from .appearance_completion_v2 import (
    bounded_surface_harmonic_fill,
    surface_sample_neighbors,
)
from .camera_geometry_v2 import project_points_xyz_v3
from .types import QualificationError
from .hashing import content_sha256
from .visibility_v2 import (
    projected_xy_to_source_texel_xy,
    rasterize_visible_owner,
)


def triangular_barycentric_samples(tile_resolution: int) -> np.ndarray:
    resolution = int(tile_resolution)
    if resolution < 4:
        raise QualificationError("CAA_TILE_RESOLUTION_TOO_SMALL")
    rows = []
    denominator = float(resolution - 1)
    for j in range(resolution):
        for i in range(resolution - j):
            u = float(i) / denominator
            v = float(j) / denominator
            rows.append((1.0 - u - v, u, v))
    value = np.asarray(rows, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3:
        raise QualificationError("CAA_BARYCENTRIC_SAMPLE_BUILD_INVALID")
    if not np.allclose(value.sum(axis=1), 1.0, atol=1e-12):
        raise QualificationError("CAA_BARYCENTRIC_SAMPLE_SIMPLEX_INVALID")
    return value


def face_atlas_layout(
    face_count: int,
    *,
    tile_resolution: int,
    bleed_px: int,
) -> dict:
    faces = int(face_count)
    tile_resolution = int(tile_resolution)
    bleed_px = int(bleed_px)
    if faces <= 0 or tile_resolution < 4 or bleed_px < 1:
        raise QualificationError("CAA_ATLAS_LAYOUT_DIMENSION_INVALID")
    stride = tile_resolution + 2 * bleed_px
    columns = int(math.ceil(math.sqrt(faces)))
    rows = int(math.ceil(faces / columns))
    width = columns * stride
    height = rows * stride
    return {
        "layout": "UNIQUE_FACE_BARYCENTRIC_V1",
        "face_count": faces,
        "tile_resolution": tile_resolution,
        "bleed_px": bleed_px,
        "tile_stride": stride,
        "columns": columns,
        "rows": rows,
        "width": width,
        "height": height,
        "uv_origin": "TOP_LEFT",
        "sampling": "BILINEAR_PREMULTIPLIED_INTERNAL",
    }



def face_atlas_paged_layout(
    face_count: int,
    *,
    tile_resolution: int,
    bleed_px: int,
    max_page_resolution: int,
) -> dict:
    """Deterministic fixed-resolution paging for the unique-face atlas.

    The numerical/art-quality tile resolution is preserved exactly. Paging is
    only a storage/layout operation: every physical page remains within the
    frozen max_page_resolution and no face silently receives a smaller tile.
    """
    faces = int(face_count)
    resolution = int(tile_resolution)
    bleed = int(bleed_px)
    page_resolution = int(max_page_resolution)
    if faces <= 0 or resolution < 4 or bleed < 1 or page_resolution < 1:
        raise QualificationError("CAA_PAGED_ATLAS_LAYOUT_DIMENSION_INVALID")
    stride = resolution + 2 * bleed
    tiles_per_axis = page_resolution // stride
    if tiles_per_axis <= 0:
        raise QualificationError("CAA_PAGED_ATLAS_TILE_EXCEEDS_PAGE")
    faces_per_page = int(tiles_per_axis * tiles_per_axis)
    page_count = int(math.ceil(faces / float(faces_per_page)))
    return {
        "layout": "UNIQUE_FACE_BARYCENTRIC_PAGED_V1",
        "face_count": faces,
        "tile_resolution": resolution,
        "bleed_px": bleed,
        "tile_stride": stride,
        "page_width": page_resolution,
        "page_height": page_resolution,
        "tiles_per_axis": int(tiles_per_axis),
        "faces_per_page": faces_per_page,
        "page_count": page_count,
        "uv_origin": "TOP_LEFT",
        "sampling": "BILINEAR_PREMULTIPLIED_INTERNAL",
        "paging": "FIXED_RESOLUTION_PHYSICAL_PAGES",
    }


def face_paged_uv_array(layout: Mapping[str, int]) -> tuple[np.ndarray, np.ndarray]:
    faces = int(layout["face_count"])
    resolution = int(layout["tile_resolution"])
    bleed = int(layout["bleed_px"])
    stride = int(layout["tile_stride"])
    tiles_per_axis = int(layout["tiles_per_axis"])
    faces_per_page = int(layout["faces_per_page"])
    width = float(layout["page_width"])
    height = float(layout["page_height"])
    if min(tiles_per_axis, faces_per_page) <= 0:
        raise QualificationError("CAA_PAGED_ATLAS_LAYOUT_INVALID")
    uv = np.zeros((faces, 3, 2), dtype=np.float64)
    page_index = np.zeros((faces,), dtype=np.int32)
    for face_index in range(faces):
        page = face_index // faces_per_page
        local_face = face_index % faces_per_page
        tx = (local_face % tiles_per_axis) * stride
        ty = (local_face // tiles_per_axis) * stride
        page_index[face_index] = int(page)
        points = (
            (tx + bleed + 0.5, ty + bleed + 0.5),
            (tx + bleed + resolution - 0.5, ty + bleed + 0.5),
            (tx + bleed + 0.5, ty + bleed + resolution - 0.5),
        )
        for corner, (x, y) in enumerate(points):
            uv[face_index, corner, 0] = float(x / width)
            uv[face_index, corner, 1] = float(y / height)
    return uv, page_index


def adaptive_face_atlas_plan(
    face_tile_resolutions: np.ndarray,
    *,
    bleed_px: int,
    max_page_resolution: int,
) -> dict:
    """Deterministically pack mixed-resolution triangular face tiles.

    Packing order is decreasing tile stride with face index as the exact tie
    break. The returned UV/page addressing is complete runtime authority; the
    per-face resolution remains compile/bake evidence and never changes source
    art or the qualified mesh.
    """
    resolutions = np.asarray(face_tile_resolutions, dtype=np.int32)
    if resolutions.ndim != 1 or len(resolutions) <= 0 or np.any(resolutions < 4):
        raise QualificationError("CAA_ADAPTIVE_ATLAS_RESOLUTION_INVALID")
    bleed = int(bleed_px)
    page_size = int(max_page_resolution)
    if bleed < 1 or page_size < 1:
        raise QualificationError("CAA_ADAPTIVE_ATLAS_POLICY_INVALID")

    tiles = [
        (int(resolution) + 2 * bleed, int(face_index), int(resolution))
        for face_index, resolution in enumerate(resolutions)
    ]
    tiles.sort(key=lambda row: (-row[0], row[1]))
    page = 0
    x = 0
    y = 0
    row_height = 0
    placements: list[tuple[int, int, int, int, int, int]] = []
    for stride, face_index, resolution in tiles:
        if stride > page_size:
            raise QualificationError("CAA_ADAPTIVE_ATLAS_TILE_EXCEEDS_PAGE")
        if x + stride > page_size:
            x = 0
            y += row_height
            row_height = 0
        if y + stride > page_size:
            page += 1
            x = 0
            y = 0
            row_height = 0
        placements.append((face_index, page, x, y, stride, resolution))
        x += stride
        row_height = max(row_height, stride)

    face_count = len(resolutions)
    page_index = np.zeros((face_count,), dtype=np.int32)
    tile_x = np.zeros((face_count,), dtype=np.int32)
    tile_y = np.zeros((face_count,), dtype=np.int32)
    tile_stride = np.zeros((face_count,), dtype=np.int32)
    uv = np.zeros((face_count, 3, 2), dtype=np.float64)
    placement_rows = []
    for face_index, page_index_value, x0, y0, stride, resolution in placements:
        page_index[face_index] = int(page_index_value)
        tile_x[face_index] = int(x0)
        tile_y[face_index] = int(y0)
        tile_stride[face_index] = int(stride)
        points = (
            (x0 + bleed + 0.5, y0 + bleed + 0.5),
            (x0 + bleed + resolution - 0.5, y0 + bleed + 0.5),
            (x0 + bleed + 0.5, y0 + bleed + resolution - 0.5),
        )
        for corner, (px, py) in enumerate(points):
            uv[face_index, corner, 0] = float(px / float(page_size))
            uv[face_index, corner, 1] = float(py / float(page_size))
        placement_rows.append(
            {
                "face_index": int(face_index),
                "page_index": int(page_index_value),
                "x": int(x0),
                "y": int(y0),
                "stride": int(stride),
                "tile_resolution": int(resolution),
            }
        )

    page_count = 1 + int(np.max(page_index, initial=0))
    histogram = {
        str(int(resolution)): int(np.count_nonzero(resolutions == int(resolution)))
        for resolution in sorted(set(map(int, resolutions.tolist())))
    }
    total_tile_area = int(np.sum(tile_stride.astype(np.int64) ** 2))
    layout = {
        "layout": "UNIQUE_FACE_BARYCENTRIC_ADAPTIVE_PAGED_V1",
        "face_count": int(face_count),
        "bleed_px": bleed,
        "page_width": page_size,
        "page_height": page_size,
        "page_count": int(page_count),
        "maximum_tile_resolution": int(np.max(resolutions)),
        "minimum_tile_resolution": int(np.min(resolutions)),
        "selected_resolution_histogram": histogram,
        "total_allocated_tile_area_texels": total_tile_area,
        "packing_efficiency_vs_page_area": float(
            total_tile_area / float(page_count * page_size * page_size)
        ),
        "placement_hash": content_sha256(placement_rows),
        "uv_origin": "TOP_LEFT",
        "sampling": "BILINEAR_PREMULTIPLIED_INTERNAL",
        "paging": "PER_FACE_RESOLUTION_FIXED_PHYSICAL_PAGES_V1",
    }
    return {
        "layout": layout,
        "face_uv": uv,
        "face_page_index": page_index,
        "face_tile_x": tile_x,
        "face_tile_y": tile_y,
        "face_tile_stride": tile_stride,
        "face_tile_resolution": resolutions.copy(),
    }


def adaptive_face_sample_offsets(face_tile_resolutions: np.ndarray) -> np.ndarray:
    resolutions = np.asarray(face_tile_resolutions, dtype=np.int32)
    if resolutions.ndim != 1 or len(resolutions) <= 0 or np.any(resolutions < 4):
        raise QualificationError("CAA_ADAPTIVE_SAMPLE_RESOLUTION_INVALID")
    counts = (
        resolutions.astype(np.int64)
        * (resolutions.astype(np.int64) + 1)
        // 2
    )
    offsets = np.zeros((len(resolutions) + 1,), dtype=np.int64)
    offsets[1:] = np.cumsum(counts, dtype=np.int64)
    return offsets

def face_uv_array(layout: Mapping[str, int]) -> np.ndarray:
    faces = int(layout["face_count"])
    resolution = int(layout["tile_resolution"])
    bleed = int(layout["bleed_px"])
    stride = int(layout["tile_stride"])
    columns = int(layout["columns"])
    width = float(layout["width"])
    height = float(layout["height"])
    uv = np.zeros((faces, 3, 2), dtype=np.float64)
    for face_index in range(faces):
        tx = (face_index % columns) * stride
        ty = (face_index // columns) * stride
        points = (
            (tx + bleed + 0.5, ty + bleed + 0.5),
            (tx + bleed + resolution - 0.5, ty + bleed + 0.5),
            (tx + bleed + 0.5, ty + bleed + resolution - 0.5),
        )
        for corner, (x, y) in enumerate(points):
            uv[face_index, corner, 0] = float(x / width)
            uv[face_index, corner, 1] = float(y / height)
    return uv


def erode_binary_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2:
        raise QualificationError("CAA_SOURCE_MASK_MUST_BE_2D")
    radius = int(radius)
    if radius < 0:
        raise QualificationError("CAA_SOURCE_EROSION_NEGATIVE")
    out = value.copy()
    for _ in range(radius):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        neighbors = [
            padded[dy : dy + out.shape[0], dx : dx + out.shape[1]]
            for dy in range(3)
            for dx in range(3)
        ]
        out = np.logical_and.reduce(neighbors)
    return out


def bilinear_rgba_u8(
    image: np.ndarray,
    xy: np.ndarray,
    *,
    return_premultiplied_linear: bool = False,
):
    """Sample source sRGB RGBA in linear-light premultiplied space.

    The returned transport sample remains straight sRGB RGBA8 for current
    asset compatibility. When requested, the exact pre-quantization linear
    premultiplied value is returned as quality evidence.
    """
    pm = bilinear_premultiplied_linear_rgba(
        np.asarray(image, dtype=np.uint8),
        np.asarray(xy, dtype=np.float64),
        normalized=False,
    )
    straight = premultiplied_linear_to_straight_srgb_u8(pm)
    if return_premultiplied_linear:
        return straight, pm
    return straight

def _candidate_vertex_id(vertex) -> str:
    value = getattr(vertex, "candidate_vertex_id", None)
    if value is None:
        raise QualificationError("CAA_REQUIRES_CANONICAL_MESH_CANDIDATE")
    return str(value)


def _candidate_face_geometry_authority(candidate):
    vertices = {
        _candidate_vertex_id(vertex): np.asarray(vertex.P, dtype=np.float64)
        for vertex in candidate.vertices
    }
    components = {
        _candidate_vertex_id(vertex): str(vertex.component_id)
        for vertex in candidate.vertices
    }
    face_rows = []
    face_component_names = []
    for face in candidate.faces:
        ids = tuple(map(str, face))
        if len(ids) != 3 or any(vertex_id not in vertices for vertex_id in ids):
            raise QualificationError("CAA_CANDIDATE_FACE_INVALID")
        component_set = {components[vertex_id] for vertex_id in ids}
        if len(component_set) != 1:
            raise QualificationError("CAA_FACE_CROSSES_COMPONENT")
        face_rows.append(ids)
        face_component_names.append(next(iter(component_set)))
    component_ids = tuple(sorted(set(face_component_names)))
    if not component_ids:
        raise QualificationError("CAA_COMPONENT_SET_EMPTY")
    component_index = {
        component_id: index for index, component_id in enumerate(component_ids)
    }
    face_component_index = np.asarray(
        [component_index[value] for value in face_component_names],
        dtype=np.int32,
    )
    return (
        vertices,
        tuple(face_rows),
        face_component_index,
        component_ids,
    )


def _surface_sample_geometry(candidate, barycentric: np.ndarray):
    barycentric = np.asarray(barycentric, dtype=np.float64)
    if barycentric.ndim != 2 or barycentric.shape[1] != 3:
        raise QualificationError("CAA_BARYCENTRIC_SAMPLE_SHAPE_INVALID")
    (
        vertices,
        face_rows,
        face_component_index,
        component_ids,
    ) = _candidate_face_geometry_authority(candidate)
    face_count = len(face_rows)
    per_face = len(barycentric)
    sample_count = face_count * per_face
    positions = np.empty((sample_count, 3), dtype=np.float64)
    face_indices = np.empty((sample_count,), dtype=np.int32)
    sample_component_index = np.empty((sample_count,), dtype=np.int32)
    normals = np.empty((face_count, 3), dtype=np.float64)

    for face_index, ids in enumerate(face_rows):
        xyz = np.stack([vertices[vertex_id] for vertex_id in ids], axis=0)
        normal = np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0])
        norm = float(np.linalg.norm(normal))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise QualificationError("CAA_FACE_DEGENERATE_BEFORE_COMPILE")
        base = face_index * per_face
        stop = base + per_face
        positions[base:stop] = barycentric @ xyz
        face_indices[base:stop] = face_index
        sample_component_index[base:stop] = face_component_index[face_index]
        normals[face_index] = normal / norm

    return (
        positions,
        face_indices,
        sample_component_index,
        component_ids,
        normals,
    )


def _surface_sample_geometry_adaptive(
    candidate,
    face_tile_resolutions: np.ndarray,
):
    resolutions = np.asarray(face_tile_resolutions, dtype=np.int32)
    if resolutions.shape != (len(candidate.faces),) or np.any(resolutions < 4):
        raise QualificationError("CAA_ADAPTIVE_GEOMETRY_RESOLUTION_INVALID")
    offsets = adaptive_face_sample_offsets(resolutions)
    sample_count = int(offsets[-1])
    (
        vertices,
        face_rows,
        face_component_index,
        component_ids,
    ) = _candidate_face_geometry_authority(candidate)

    positions = np.empty((sample_count, 3), dtype=np.float64)
    face_indices = np.empty((sample_count,), dtype=np.int32)
    sample_component_index = np.empty((sample_count,), dtype=np.int32)
    normals = np.empty((len(face_rows), 3), dtype=np.float64)
    barycentric_cache: dict[int, np.ndarray] = {}

    for face_index, ids in enumerate(face_rows):
        xyz = np.stack([vertices[vertex_id] for vertex_id in ids], axis=0)
        normal = np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0])
        norm = float(np.linalg.norm(normal))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise QualificationError("CAA_FACE_DEGENERATE_BEFORE_COMPILE")
        resolution = int(resolutions[face_index])
        barycentric = barycentric_cache.get(resolution)
        if barycentric is None:
            barycentric = triangular_barycentric_samples(resolution)
            barycentric_cache[resolution] = barycentric

        base = int(offsets[face_index])
        stop = int(offsets[face_index + 1])
        if stop - base != len(barycentric):
            raise QualificationError(
                "CAA_ADAPTIVE_GEOMETRY_SAMPLE_ACCOUNTING_DRIFT"
            )
        positions[base:stop] = barycentric @ xyz
        face_indices[base:stop] = face_index
        sample_component_index[base:stop] = face_component_index[face_index]
        normals[face_index] = normal / norm

    return (
        positions,
        face_indices,
        sample_component_index,
        component_ids,
        normals,
        offsets,
    )

def projected_tile_resolution_evidence(
    *,
    candidate,
    cameras,
    foreground_mask_by_view: Mapping[int, np.ndarray],
    candidate_resolutions: tuple[int, ...],
    max_source_pixels_per_atlas_texel: float,
    bleed_px: int,
    max_atlas_resolution: int,
) -> dict:
    """Resolve art-quality tile density independently from physical page count.

    max_atlas_resolution is a physical page ceiling, not a reason to lower
    source-preserving tile density. When a single atlas would exceed the cap,
    deterministic fixed-resolution pages are used.
    """
    resolutions = tuple(sorted(set(int(value) for value in candidate_resolutions)))
    if not resolutions or resolutions[0] < 4:
        raise QualificationError("CAA_TILE_CANDIDATES_INVALID")
    limit = float(max_source_pixels_per_atlas_texel)
    page_resolution = int(max_atlas_resolution)
    bleed = int(bleed_px)
    if not math.isfinite(limit) or limit <= 0.0:
        raise QualificationError("CAA_TILE_DENSITY_LIMIT_INVALID")
    if page_resolution < 1 or bleed < 1:
        raise QualificationError("CAA_TILE_PAGE_POLICY_INVALID")

    vertex_ids = [_candidate_vertex_id(vertex) for vertex in candidate.vertices]
    index = {vertex_id: i for i, vertex_id in enumerate(vertex_ids)}
    xyz = np.asarray([vertex.P for vertex in candidate.vertices], dtype=np.float64)
    if len(index) != len(vertex_ids) or xyz.shape != (len(vertex_ids), 3):
        raise QualificationError("CAA_TILE_CANDIDATE_VERTEX_INVALID")

    worst_sigma = 0.0
    worst_view = -1
    worst_face = -1
    visible_face_observation_count = 0
    by_view = {int(camera.view_index): camera for camera in cameras}
    if set(by_view) != set(range(8)):
        raise QualificationError("CAA_TILE_REQUIRES_V0_V7_CAMERAS")

    for view in range(8):
        mask = np.asarray(foreground_mask_by_view[view], dtype=bool)
        camera = by_view[view]
        visibility = rasterize_visible_owner(
            candidate,
            camera,
            width=mask.shape[1],
            height=mask.shape[0],
        )
        owner = visibility.owner_face_index
        face_ids = np.unique(owner[mask & (owner >= 0)])
        if not len(face_ids):
            continue
        projected = np.asarray(project_points_xyz_v3(xyz, camera), dtype=np.float64)
        for face_index in map(int, face_ids):
            face = candidate.faces[face_index]
            try:
                ids = [index[str(vertex_id)] for vertex_id in face]
            except KeyError as exc:
                raise QualificationError("CAA_TILE_FACE_VERTEX_UNKNOWN") from exc
            tri = projected[ids, :2]
            matrix = np.asarray(
                [
                    [tri[1, 0] - tri[0, 0], tri[2, 0] - tri[0, 0]],
                    [tri[1, 1] - tri[0, 1], tri[2, 1] - tri[0, 1]],
                ],
                dtype=np.float64,
            )
            singular = np.linalg.svd(matrix, compute_uv=False)
            sigma = float(np.max(singular))
            if not math.isfinite(sigma):
                raise QualificationError("CAA_TILE_PROJECTED_SCALE_NONFINITE")
            visible_face_observation_count += 1
            if sigma > worst_sigma:
                worst_sigma = sigma
                worst_view = view
                worst_face = face_index

    if visible_face_observation_count <= 0:
        raise QualificationError("CAA_TILE_NO_SOURCE_VISIBLE_FACE")

    rows = []
    selected = None
    selected_layout = None
    face_count = len(candidate.faces)
    for resolution in resolutions:
        source_pixels_per_atlas_texel = worst_sigma / float(resolution - 1)
        single_layout = face_atlas_layout(
            face_count,
            tile_resolution=resolution,
            bleed_px=bleed,
        )
        stride = int(resolution + 2 * bleed)
        tiles_per_axis = page_resolution // stride
        capacity_passed = tiles_per_axis > 0
        page_count = None
        faces_per_page = 0
        if capacity_passed:
            paged = face_atlas_paged_layout(
                face_count,
                tile_resolution=resolution,
                bleed_px=bleed,
                max_page_resolution=page_resolution,
            )
            page_count = int(paged["page_count"])
            faces_per_page = int(paged["faces_per_page"])
        density_passed = source_pixels_per_atlas_texel <= limit
        rows.append(
            {
                "tile_resolution": resolution,
                "worst_source_pixels_per_atlas_texel": source_pixels_per_atlas_texel,
                "density_passed": bool(density_passed),
                "single_page_atlas_width": int(single_layout["width"]),
                "single_page_atlas_height": int(single_layout["height"]),
                "physical_page_width": page_resolution,
                "physical_page_height": page_resolution,
                "page_count": page_count,
                "faces_per_page": faces_per_page,
                "capacity_passed": bool(capacity_passed),
            }
        )
        if selected is None and density_passed and capacity_passed:
            selected = resolution
            selected_layout = paged

    max_supported_face_count = 0
    selected_page_count = 0
    if selected is not None and selected_layout is not None:
        selected_page_count = int(selected_layout["page_count"])
        max_supported_face_count = int(
            selected_layout["faces_per_page"] * selected_layout["page_count"]
        )

    return {
        "mode": "PROJECTED_SOURCE_DENSITY_V1",
        "paging_mode": "FIXED_MAX_RESOLUTION_PAGES_V1",
        "selected_tile_resolution": (
            None if selected is None else int(selected)
        ),
        "selected_page_count": selected_page_count,
        "max_source_pixels_per_atlas_texel": limit,
        "max_atlas_resolution": page_resolution,
        "face_count": int(face_count),
        "worst_projected_barycentric_sigma_px": worst_sigma,
        "worst_view_index": int(worst_view),
        "worst_face_index": int(worst_face),
        "visible_face_observation_count": int(visible_face_observation_count),
        "max_supported_face_count": int(max_supported_face_count),
        "candidates": rows,
    }

def projected_adaptive_face_tile_evidence(
    *,
    candidate,
    cameras,
    foreground_mask_by_view: Mapping[int, np.ndarray],
    candidate_resolutions: tuple[int, ...],
    max_source_pixels_per_atlas_texel: float,
    bleed_px: int,
    max_atlas_resolution: int,
) -> dict:
    """Measure per-face source-density requirements and deterministic page demand.

    This is evidence only. It does not change compile sampling or asset layout.
    Every face receives the smallest frozen candidate resolution whose projected
    source footprint satisfies the same source-pixels-per-texel limit. Faces not
    directly visible in the source observation set receive the minimum candidate
    resolution and are reported separately.
    """
    resolutions=tuple(sorted(set(int(x) for x in candidate_resolutions)))
    if not resolutions or resolutions[0]<4:
        raise QualificationError("CAA_ADAPTIVE_TILE_CANDIDATES_INVALID")
    limit=float(max_source_pixels_per_atlas_texel)
    max_res=int(max_atlas_resolution)
    bleed=int(bleed_px)
    if not math.isfinite(limit) or limit<=0.0 or max_res<256 or bleed<1:
        raise QualificationError("CAA_ADAPTIVE_TILE_POLICY_INVALID")

    vertex_ids=[_candidate_vertex_id(vertex) for vertex in candidate.vertices]
    index={vertex_id:i for i,vertex_id in enumerate(vertex_ids)}
    xyz=np.asarray([vertex.P for vertex in candidate.vertices],dtype=np.float64)
    face_count=len(candidate.faces)
    per_face_sigma=np.zeros(face_count,dtype=np.float64)
    observed=np.zeros(face_count,dtype=bool)

    by_view={int(camera.view_index):camera for camera in cameras}
    if set(by_view)!=set(range(8)):
        raise QualificationError("CAA_ADAPTIVE_TILE_REQUIRES_V0_V7_CAMERAS")
    for view in range(8):
        mask=np.asarray(foreground_mask_by_view[view],dtype=bool)
        camera=by_view[view]
        visibility=rasterize_visible_owner(
            candidate,camera,width=mask.shape[1],height=mask.shape[0]
        )
        owner=visibility.owner_face_index
        face_ids=np.unique(owner[mask & (owner>=0)]).astype(np.int64)
        if not len(face_ids):
            continue
        projected=np.asarray(project_points_xyz_v3(xyz,camera),dtype=np.float64)
        for face_index in face_ids:
            face=candidate.faces[int(face_index)]
            ids=[index[str(vertex_id)] for vertex_id in face]
            tri=projected[ids,:2]
            matrix=np.asarray(
                [
                    [tri[1,0]-tri[0,0],tri[2,0]-tri[0,0]],
                    [tri[1,1]-tri[0,1],tri[2,1]-tri[0,1]],
                ],
                dtype=np.float64,
            )
            sigma=float(np.max(np.linalg.svd(matrix,compute_uv=False)))
            if not math.isfinite(sigma):
                raise QualificationError("CAA_ADAPTIVE_TILE_PROJECTED_SCALE_NONFINITE")
            observed[int(face_index)]=True
            if sigma>per_face_sigma[int(face_index)]:
                per_face_sigma[int(face_index)]=sigma

    selected=np.full(face_count,-1,dtype=np.int32)
    unsatisfied=np.zeros(face_count,dtype=bool)
    minimum=resolutions[0]
    for face_index in range(face_count):
        if not observed[face_index]:
            selected[face_index]=minimum
            continue
        sigma=float(per_face_sigma[face_index])
        choice=None
        for resolution in resolutions:
            if sigma/float(resolution-1)<=limit:
                choice=resolution
                break
        if choice is None:
            unsatisfied[face_index]=True
            selected[face_index]=resolutions[-1]
        else:
            selected[face_index]=int(choice)

    histogram={str(res):int(np.count_nonzero(selected==res)) for res in resolutions}
    plan=adaptive_face_atlas_plan(
        selected,
        bleed_px=bleed,
        max_page_resolution=max_res,
    )
    layout=dict(plan["layout"])
    offsets=adaptive_face_sample_offsets(selected)
    adaptive_samples=int(offsets[-1])
    worst_resolution=int(np.max(selected,initial=minimum))
    uniform_worst_samples=int(
        face_count * worst_resolution * (worst_resolution + 1) // 2
    )
    return {
        "schema":"RealSaS.CAAAdaptiveFaceTileEvidence.v1",
        "mode":"PROJECTED_SOURCE_DENSITY_PER_FACE_V1",
        "face_count":int(face_count),
        "observed_face_count":int(np.count_nonzero(observed)),
        "unobserved_face_count":int(np.count_nonzero(~observed)),
        "unsatisfied_face_count":int(np.count_nonzero(unsatisfied)),
        "candidate_resolutions":list(resolutions),
        "selected_resolution_histogram":histogram,
        "selected_resolution_by_face":[int(value) for value in selected],
        "max_source_pixels_per_atlas_texel":limit,
        "bleed_px":bleed,
        "max_page_resolution":max_res,
        "total_allocated_tile_area_texels":int(
            layout["total_allocated_tile_area_texels"]
        ),
        "page_area_texels":int(max_res*max_res),
        "area_lower_bound_page_count":int(
            math.ceil(
                float(layout["total_allocated_tile_area_texels"])
                / float(max_res*max_res)
            )
        ),
        "deterministic_shelf_page_count":int(layout["page_count"]),
        "packing_efficiency_vs_page_area":float(
            layout["packing_efficiency_vs_page_area"]
        ),
        "maximum_observed_sigma_px":float(np.max(per_face_sigma,initial=0.0)),
        "maximum_required_resolution":worst_resolution,
        "minimum_required_resolution":int(np.min(selected,initial=minimum)),
        "placement_hash":str(layout["placement_hash"]),
        "sample_count_per_direction":adaptive_samples,
        "uniform_worst_case_sample_count_per_direction":uniform_worst_samples,
        "adaptive_sample_fraction_of_uniform_worst_case":float(
            adaptive_samples / float(max(uniform_worst_samples,1))
        ),
    }

def resolve_projected_tile_resolution(
    *,
    candidate,
    cameras,
    foreground_mask_by_view: Mapping[int, np.ndarray],
    candidate_resolutions: tuple[int, ...],
    max_source_pixels_per_atlas_texel: float,
    bleed_px: int,
    max_atlas_resolution: int,
) -> dict:
    evidence = projected_tile_resolution_evidence(
        candidate=candidate,
        cameras=cameras,
        foreground_mask_by_view=foreground_mask_by_view,
        candidate_resolutions=candidate_resolutions,
        max_source_pixels_per_atlas_texel=max_source_pixels_per_atlas_texel,
        bleed_px=bleed_px,
        max_atlas_resolution=max_atlas_resolution,
    )
    if evidence["selected_tile_resolution"] is None:
        raise QualificationError("CAA_TILE_DENSITY_OR_CAPACITY_UNSATISFIED")
    return evidence


def _circular_view_order(target: int) -> tuple[int, ...]:
    return tuple(
        sorted(
            range(8),
            key=lambda view: (
                min(abs(view - target), 8 - abs(view - target)),
                view,
            ),
        )
    )


def select_other_view_donor_by_support(
    *,
    target_view_index: int,
    missing: np.ndarray,
    direct_valid: np.ndarray,
    sample_face_index: np.ndarray,
    face_support_by_view: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Select source donors by geometric support, never circular index proximity.

    Candidate iteration order is cyclic only to make exact-score ties
    deterministic. Any strictly higher camera-forward face support must win,
    regardless of angular/index distance from the target output direction.
    """
    target = int(target_view_index)
    missing = np.asarray(missing, dtype=bool)
    direct_valid = np.asarray(direct_valid, dtype=bool)
    sample_face = np.asarray(sample_face_index, dtype=np.int32)
    support = np.asarray(face_support_by_view, dtype=np.float64)
    if (
        target < 0
        or target >= 8
        or direct_valid.ndim != 2
        or direct_valid.shape[0] != 8
        or direct_valid.shape[1] != len(missing)
        or sample_face.shape != (len(missing),)
        or support.ndim != 2
        or support.shape[0] != 8
        or np.any(sample_face < 0)
        or np.any(sample_face >= support.shape[1])
        or not np.isfinite(support).all()
    ):
        raise QualificationError("CAA_DONOR_SUPPORT_INPUT_INVALID")

    best_score = np.full(len(missing), -1.0, dtype=np.float64)
    best_view = np.full(len(missing), -1, dtype=np.int16)
    for donor in _circular_view_order(target):
        if donor == target:
            continue
        eligible = missing & direct_valid[donor]
        if not np.any(eligible):
            continue
        candidate_support = support[donor, sample_face]
        improve = eligible & (candidate_support > best_score + 1.0e-12)
        if np.any(improve):
            best_score[improve] = candidate_support[improve]
            best_view[improve] = donor
    return best_view, best_score


def compile_deterministic_caa(
    *,
    candidate,
    cameras,
    source_rgba_by_view: Mapping[int, np.ndarray],
    foreground_mask_by_view: Mapping[int, np.ndarray],
    tile_resolution: int | None,
    source_lock_policy: Mapping[str, object],
    completion_quality_policy: Mapping[str, object] | None = None,
    face_tile_resolutions: np.ndarray | None = None,
) -> dict:
    """Compile deterministic CAA on uniform or per-face adaptive lattices.

    Adaptive mode changes only sampling density/layout. Source locking, donor
    selection, bounded harmonic completion and every quality threshold remain
    identical to the uniform path.
    """
    face_count = len(candidate.faces)
    adaptive = face_tile_resolutions is not None
    if adaptive:
        resolutions = np.asarray(face_tile_resolutions, dtype=np.int32)
        if resolutions.shape != (face_count,) or np.any(resolutions < 4):
            raise QualificationError("CAA_ADAPTIVE_COMPILE_RESOLUTION_INVALID")
        (
            positions,
            sample_face,
            sample_component_index,
            component_ids,
            face_normals,
            face_sample_offsets,
        ) = _surface_sample_geometry_adaptive(candidate, resolutions)
        barycentric = None
        barycentric_storage_mode = (
            "RECONSTRUCT_FROM_FACE_RESOLUTION_AND_OFFSETS_V1"
        )
        sample_count = len(positions)
        max_resolution = int(np.max(resolutions))
        max_samples_per_face = max_resolution * (max_resolution + 1) // 2
        sample_count_mode = "PER_FACE_ADAPTIVE_V1"
    else:
        if tile_resolution is None:
            raise QualificationError("CAA_TILE_RESOLUTION_MISSING")
        resolution = int(tile_resolution)
        barycentric = triangular_barycentric_samples(resolution)
        (
            positions,
            sample_face,
            sample_component_index,
            component_ids,
            face_normals,
        ) = _surface_sample_geometry(candidate, barycentric)
        sample_count = len(positions)
        per_face_samples = len(barycentric)
        if sample_count != face_count * per_face_samples:
            raise QualificationError("CAA_SAMPLE_ACCOUNTING_DRIFT")
        resolutions = np.full((face_count,), resolution, dtype=np.int32)
        face_sample_offsets = adaptive_face_sample_offsets(resolutions)
        max_resolution = resolution
        max_samples_per_face = per_face_samples
        sample_count_mode = "UNIFORM_FACE_LATTICE_V1"
        barycentric_storage_mode = "UNIFORM_PATTERN_EXPLICIT_V1"

    min_cos = float(source_lock_policy["min_abs_normal_camera_cos"])
    erosion = int(source_lock_policy["boundary_safe_erosion_px"])
    min_alpha = int(source_lock_policy["min_source_alpha_u8"])
    completion_policy = dict(completion_quality_policy or {})
    max_harmonic_region = int(
        completion_policy.get("max_local_harmonic_region_samples", 16)
    )
    max_harmonic_hops = int(
        completion_policy.get("max_local_harmonic_graph_hops", 2)
    )
    if max_harmonic_region <= 0 or max_harmonic_hops <= 0:
        raise QualificationError("CAA_COMPLETION_POLICY_INVALID")

    direct_valid = np.zeros((8, sample_count), dtype=bool)
    direct_rgba = np.zeros((8, sample_count, 4), dtype=np.uint8)
    # Premultiplied-linear truth is required only where direct source evidence
    # exists. Keep exact float64 values packed in deterministic view-major order
    # instead of allocating a dense 8 x sample_count x 4 tensor dominated by
    # unused zeros.
    direct_pm_linear_chunks: list[np.ndarray] = []
    # Stage24 structured holdout authority consumes source_xy as float32.
    # Store it in that canonical consumer precision instead of carrying a
    # redundant float64 copy across the multi-million-sample compile artifact.
    source_xy = np.full((8, sample_count, 2), np.nan, dtype=np.float32)
    face_support_by_view = np.zeros((8, face_count), dtype=np.float64)

    camera_by_view = {int(camera.view_index): camera for camera in cameras}
    if set(camera_by_view) != set(range(8)):
        raise QualificationError("CAA_DETERMINISTIC_REQUIRES_V0_V7_CAMERAS")

    for view in range(8):
        camera = camera_by_view[view]
        image = np.asarray(source_rgba_by_view[view], dtype=np.uint8)
        mask = np.asarray(foreground_mask_by_view[view], dtype=bool)
        if image.shape[:2] != mask.shape or image.shape[2] != 4:
            raise QualificationError("CAA_SOURCE_IMAGE_MASK_DIMENSION_DRIFT")
        safe_foreground = erode_binary_mask(mask, erosion)
        safe_background = erode_binary_mask(~mask, erosion)
        visibility = rasterize_visible_owner(
            candidate,
            camera,
            width=image.shape[1],
            height=image.shape[0],
        )
        projected = np.asarray(project_points_xyz_v3(positions, camera), dtype=np.float64)
        xy = projected[:, :2] - 0.5
        source_xy[view] = xy

        ix = np.rint(xy[:, 0]).astype(np.int64)
        iy = np.rint(xy[:, 1]).astype(np.int64)
        in_bounds = (
            (ix >= 0)
            & (ix < image.shape[1])
            & (iy >= 0)
            & (iy < image.shape[0])
            & np.isfinite(projected[:, 2])
            & (projected[:, 2] > 0.0)
        )
        foreground_safe = np.zeros(sample_count, dtype=bool)
        background_safe = np.zeros(sample_count, dtype=bool)
        visible = np.zeros(sample_count, dtype=bool)
        alpha_foreground_safe = np.zeros(sample_count, dtype=bool)
        alpha_background_safe = np.zeros(sample_count, dtype=bool)
        valid_index = np.flatnonzero(in_bounds)
        if len(valid_index):
            x = ix[valid_index]
            y = iy[valid_index]
            foreground_safe[valid_index] = safe_foreground[y, x]
            background_safe[valid_index] = safe_background[y, x]
            visible[valid_index] = (
                visibility.owner_face_index[y, x]
                == sample_face[valid_index]
            )
            alpha_foreground_safe[valid_index] = image[y, x, 3] >= min_alpha
            alpha_background_safe[valid_index] = image[y, x, 3] == 0

        forward = np.asarray(camera.forward, dtype=np.float64)
        forward_norm = float(np.linalg.norm(forward))
        if not math.isfinite(forward_norm) or forward_norm <= 1e-12:
            raise QualificationError("CAA_CAMERA_FORWARD_INVALID")
        forward = forward / forward_norm
        face_cos = np.abs(face_normals @ forward)
        face_support_by_view[view] = face_cos
        angle_safe = face_cos[sample_face] >= min_cos

        appearance_support = (
            (foreground_safe & alpha_foreground_safe)
            | (background_safe & alpha_background_safe)
        )
        valid = in_bounds & visible & appearance_support & angle_safe
        direct_valid[view] = valid
        if np.any(valid):
            sampled_rgba, sampled_pm = bilinear_rgba_u8(
                image,
                xy[valid],
                return_premultiplied_linear=True,
            )
            direct_rgba[view, valid] = sampled_rgba
            direct_pm_linear_chunks.append(
                np.asarray(sampled_pm, dtype=np.float64).copy()
            )

    direct_pm_linear_packed = (
        np.concatenate(direct_pm_linear_chunks, axis=0)
        if direct_pm_linear_chunks
        else np.empty((0, 4), dtype=np.float64)
    )
    direct_count = int(np.count_nonzero(direct_valid))
    if direct_pm_linear_packed.shape != (direct_count, 4):
        raise QualificationError("CAA_DIRECT_PM_PACKED_ACCOUNTING_DRIFT")
    direct_pm_linear_chunks.clear()

    rgba = np.zeros_like(direct_rgba)
    provenance = np.full((8, sample_count), 255, dtype=np.uint8)
    source_view = np.full((8, sample_count), -1, dtype=np.int16)
    surface_neighbors = surface_sample_neighbors(
        positions=positions,
        face_count=face_count,
        tile_resolution=None if adaptive else int(max_resolution),
        face_sample_offsets=face_sample_offsets if adaptive else None,
        face_tile_resolutions=resolutions if adaptive else None,
        face_vertex_ids=tuple(
            tuple(map(str, face)) for face in candidate.faces
        ),
    )
    completion_rows = []

    sample_component_index = np.asarray(
        sample_component_index, dtype=np.int32
    )
    if (
        sample_component_index.shape != (sample_count,)
        or np.any(sample_component_index < 0)
        or np.any(sample_component_index >= len(component_ids))
    ):
        raise QualificationError("CAA_SAMPLE_COMPONENT_INDEX_INVALID")
    component_count = len(component_ids)

    for target in range(8):
        direct = direct_valid[target]
        rgba[target, direct] = direct_rgba[target, direct]
        provenance[target, direct] = CAA_PROVENANCE["DIRECT_SOURCE"]
        source_view[target, direct] = target

        missing = ~direct
        best_view, _best_score = select_other_view_donor_by_support(
            target_view_index=target,
            missing=missing,
            direct_valid=direct_valid,
            sample_face_index=sample_face,
            face_support_by_view=face_support_by_view,
        )
        source_take = missing & (best_view >= 0)
        if np.any(source_take):
            indices = np.flatnonzero(source_take)
            donors = best_view[indices].astype(np.int64)
            rgba[target, indices] = direct_rgba[donors, indices]
            provenance[target, indices] = CAA_PROVENANCE["OTHER_VIEW_SOURCE"]
            source_view[target, indices] = donors.astype(np.int16)
            missing[indices] = False

        if not np.any(provenance[target] != 255):
            raise QualificationError("CAA_NO_SOURCE_OBSERVATION_ANYWHERE")

        missing_component_count = np.bincount(
            sample_component_index[missing],
            minlength=component_count,
        )
        observed_component_count = np.bincount(
            sample_component_index[~missing],
            minlength=component_count,
        )
        if np.any(
            (missing_component_count > 0)
            & (observed_component_count == 0)
        ):
            raise QualificationError("CAA_COMPONENT_WITHOUT_SOURCE_OBSERVATION")

        stats = bounded_surface_harmonic_fill(
            rgba=rgba[target],
            provenance=provenance[target],
            source_view=source_view[target],
            missing=missing,
            sample_component=sample_component_index,
            neighbors=surface_neighbors,
            max_region_samples=max_harmonic_region,
            max_graph_hops=max_harmonic_hops,
            abstain_on_policy_violation=True,
            abstain_provenance_code=CAA_PROVENANCE["UNSUPPORTED_ABSTAIN"],
            abstain_source_view_value=-4,
        )
        completion_rows.append({"target_view_index": target, **stats})
        if np.any(provenance[target] == 255):
            raise QualificationError("CAA_PROVENANCE_UNCLASSIFIED_AFTER_COMPILE")
        valid_codes = np.asarray(
            tuple(sorted(CAA_PROVENANCE.values())),
            dtype=np.uint8,
        )
        if np.any(~np.isin(provenance[target], valid_codes)):
            raise QualificationError("CAA_PROVENANCE_CLASS_INVALID_AFTER_COMPILE")

    counts = {
        name: int(np.count_nonzero(provenance == code))
        for name, code in CAA_PROVENANCE.items()
    }
    return {
        "barycentric": barycentric,
        "sample_positions": positions,
        "sample_face_index": sample_face,
        "face_sample_offsets": face_sample_offsets,
        "face_tile_resolutions": resolutions,
        "direct_valid": direct_valid,
        "direct_rgba": direct_rgba,
        "direct_pm_linear_packed": direct_pm_linear_packed,
        "source_xy": source_xy,
        "rgba": rgba,
        "provenance": provenance,
        "source_view": source_view,
        "component_ids": component_ids,
        "sample_component_index": sample_component_index,
        "counts": counts,
        "completion_rows": tuple(completion_rows),
        "unsupported_abstain_source_view_value": -4,
        "face_count": face_count,
        "sample_count_mode": sample_count_mode,
        "sample_count_per_face": int(max_samples_per_face),
        "sample_count_per_direction": int(sample_count),
        "maximum_tile_resolution": int(max_resolution),
        "source_xy_storage_dtype": "float32",
        "direct_pm_linear_storage_dtype": "float64",
        "direct_pm_linear_storage_mode": "PACKED_DIRECT_VALID_VIEW_MAJOR_V1",
        "barycentric_storage_mode": barycentric_storage_mode,
    }

