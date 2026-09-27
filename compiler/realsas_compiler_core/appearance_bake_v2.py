from __future__ import annotations

"""Bake CAA triangular surface samples into unique-face transport atlases."""

import math

import numpy as np

from .appearance_compile_v2 import (
    adaptive_face_atlas_plan,
    adaptive_face_sample_offsets,
    face_atlas_layout,
    face_atlas_paged_layout,
    face_paged_uv_array,
    face_uv_array,
)
from .appearance_color_v2 import (
    bilinear_premultiplied_linear_rgba,
    straight_srgb_rgba_u8_to_premultiplied_linear,
)
from .types import QualificationError


def _sample_index_map(tile_resolution: int) -> dict[tuple[int, int], int]:
    resolution = int(tile_resolution)
    index = {}
    cursor = 0
    for j in range(resolution):
        for i in range(resolution - j):
            index[(i, j)] = cursor
            cursor += 1
    return index


def _nearest_triangle_lattice(
    i: int,
    j: int,
    *,
    tile_resolution: int,
) -> tuple[int, int]:
    resolution = int(tile_resolution)
    max_index = resolution - 1
    i = max(0, min(max_index, int(i)))
    j = max(0, min(max_index, int(j)))
    if i + j <= max_index:
        return i, j
    total = i + j
    if total <= 0:
        return 0, 0
    scale = float(max_index) / float(total)
    ii = int(round(i * scale))
    jj = int(round(j * scale))
    while ii + jj > max_index:
        if ii >= jj and ii > 0:
            ii -= 1
        elif jj > 0:
            jj -= 1
        else:
            break
    return ii, jj


def _iter_face_atlas_lattice_samples(
    *,
    face_count: int,
    tile_resolution: int,
    bleed_px: int,
):
    """Yield the single canonical face-atlas texel -> lattice-sample mapping.

    RGBA, coarse provenance, and exact source-view lineage must consume this
    same mapping so diagnostic lineage can never drift to a different texel.
    """
    layout = face_atlas_layout(
        int(face_count),
        tile_resolution=int(tile_resolution),
        bleed_px=int(bleed_px),
    )
    sample_map = _sample_index_map(int(tile_resolution))
    stride = int(layout["tile_stride"])
    columns = int(layout["columns"])
    bleed = int(layout["bleed_px"])
    for face_index in range(int(face_count)):
        tile_x = (face_index % columns) * stride
        tile_y = (face_index // columns) * stride
        for local_y in range(stride):
            for local_x in range(stride):
                ii, jj = _nearest_triangle_lattice(
                    local_x - bleed,
                    local_y - bleed,
                    tile_resolution=int(tile_resolution),
                )
                yield (
                    face_index,
                    tile_y + local_y,
                    tile_x + local_x,
                    sample_map[(ii, jj)],
                )




def _iter_face_adaptive_paged_atlas_lattice_samples(
    plan: dict,
    face_sample_offsets: np.ndarray,
):
    resolutions = np.asarray(plan["face_tile_resolution"], dtype=np.int32)
    page_index = np.asarray(plan["face_page_index"], dtype=np.int32)
    tile_x = np.asarray(plan["face_tile_x"], dtype=np.int32)
    tile_y = np.asarray(plan["face_tile_y"], dtype=np.int32)
    tile_stride = np.asarray(plan["face_tile_stride"], dtype=np.int32)
    offsets = np.asarray(face_sample_offsets, dtype=np.int64)
    face_count = len(resolutions)
    if (
        page_index.shape != (face_count,)
        or tile_x.shape != (face_count,)
        or tile_y.shape != (face_count,)
        or tile_stride.shape != (face_count,)
        or offsets.shape != (face_count + 1,)
        or offsets[0] != 0
    ):
        raise QualificationError("CAA_ADAPTIVE_BAKE_LAYOUT_INVALID")
    bleed = int(plan["layout"]["bleed_px"])
    sample_map_cache: dict[int, dict[tuple[int, int], int]] = {}
    for face_index in range(face_count):
        resolution = int(resolutions[face_index])
        stride = int(tile_stride[face_index])
        expected_stride = resolution + 2 * bleed
        if stride != expected_stride:
            raise QualificationError("CAA_ADAPTIVE_BAKE_STRIDE_DRIFT")
        sample_map = sample_map_cache.get(resolution)
        if sample_map is None:
            sample_map = _sample_index_map(resolution)
            sample_map_cache[resolution] = sample_map
        base = int(offsets[face_index])
        stop = int(offsets[face_index + 1])
        expected_samples = resolution * (resolution + 1) // 2
        if stop - base != expected_samples:
            raise QualificationError("CAA_ADAPTIVE_BAKE_SAMPLE_ACCOUNTING_DRIFT")
        for local_y in range(stride):
            for local_x in range(stride):
                ii, jj = _nearest_triangle_lattice(
                    local_x - bleed,
                    local_y - bleed,
                    tile_resolution=resolution,
                )
                local_sample = sample_map[(ii, jj)]
                yield (
                    face_index,
                    int(page_index[face_index]),
                    int(tile_y[face_index]) + local_y,
                    int(tile_x[face_index]) + local_x,
                    base + int(local_sample),
                )


def bake_direction_adaptive_atlas_pages(
    *,
    face_sample_rgba: np.ndarray,
    face_sample_provenance: np.ndarray,
    face_tile_resolutions: np.ndarray,
    face_sample_offsets: np.ndarray,
    bleed_px: int,
    max_page_resolution: int,
):
    rgba_samples = np.asarray(face_sample_rgba, dtype=np.uint8)
    provenance_samples = np.asarray(face_sample_provenance, dtype=np.uint8)
    resolutions = np.asarray(face_tile_resolutions, dtype=np.int32)
    offsets = np.asarray(face_sample_offsets, dtype=np.int64)
    plan = adaptive_face_atlas_plan(
        resolutions,
        bleed_px=int(bleed_px),
        max_page_resolution=int(max_page_resolution),
    )
    if offsets.shape != (len(resolutions) + 1,) or offsets[-1] <= 0:
        raise QualificationError("CAA_ADAPTIVE_BAKE_OFFSETS_INVALID")
    total_samples = int(offsets[-1])
    if rgba_samples.shape != (total_samples, 4):
        raise QualificationError("CAA_ADAPTIVE_BAKE_RGBA_SAMPLE_SHAPE_INVALID")
    if provenance_samples.shape != (total_samples,):
        raise QualificationError("CAA_ADAPTIVE_BAKE_PROVENANCE_SAMPLE_SHAPE_INVALID")

    layout = dict(plan["layout"])
    page_count = int(layout["page_count"])
    height = int(layout["page_height"])
    width = int(layout["page_width"])
    pages = np.zeros((page_count, height, width, 4), dtype=np.uint8)
    provenance_pages = np.full((page_count, height, width), 255, dtype=np.uint8)
    allocated = np.zeros((page_count, height, width), dtype=bool)
    for _face, page, atlas_y, atlas_x, sample_index in (
        _iter_face_adaptive_paged_atlas_lattice_samples(plan, offsets)
    ):
        allocated[page, atlas_y, atlas_x] = True
        pages[page, atlas_y, atlas_x] = rgba_samples[sample_index]
        provenance_pages[page, atlas_y, atlas_x] = provenance_samples[sample_index]

    if np.any(provenance_pages[allocated] == 255):
        raise QualificationError("CAA_ADAPTIVE_BAKE_ALLOCATED_TILE_UNDEFINED_TEXEL")
    if np.any((provenance_pages != 255) & ~allocated):
        raise QualificationError("CAA_ADAPTIVE_BAKE_PADDING_CONTAMINATED")
    return (
        pages,
        provenance_pages,
        np.asarray(plan["face_uv"], dtype=np.float64),
        np.asarray(plan["face_page_index"], dtype=np.int32),
        layout,
    )


def bake_direction_adaptive_source_view_atlas_pages(
    *,
    face_sample_source_view: np.ndarray,
    face_tile_resolutions: np.ndarray,
    face_sample_offsets: np.ndarray,
    bleed_px: int,
    max_page_resolution: int,
) -> np.ndarray:
    source_samples = np.asarray(face_sample_source_view, dtype=np.int16)
    resolutions = np.asarray(face_tile_resolutions, dtype=np.int32)
    offsets = np.asarray(face_sample_offsets, dtype=np.int64)
    plan = adaptive_face_atlas_plan(
        resolutions,
        bleed_px=int(bleed_px),
        max_page_resolution=int(max_page_resolution),
    )
    if offsets.shape != (len(resolutions) + 1,) or offsets[-1] <= 0:
        raise QualificationError("CAA_ADAPTIVE_BAKE_SOURCE_OFFSETS_INVALID")
    total_samples = int(offsets[-1])
    if source_samples.shape != (total_samples,):
        raise QualificationError("CAA_ADAPTIVE_BAKE_SOURCE_VIEW_SAMPLE_SHAPE_INVALID")
    valid = (
        ((source_samples >= 0) & (source_samples < 8))
        | (source_samples == -2)
        | (source_samples == -4)
    )
    if not np.all(valid):
        raise QualificationError("CAA_ADAPTIVE_BAKE_SOURCE_VIEW_SAMPLE_VALUE_INVALID")

    layout = dict(plan["layout"])
    padding = np.iinfo(np.int16).min
    pages = np.full(
        (
            int(layout["page_count"]),
            int(layout["page_height"]),
            int(layout["page_width"]),
        ),
        padding,
        dtype=np.int16,
    )
    allocated = np.zeros(pages.shape, dtype=bool)
    for _face, page, atlas_y, atlas_x, sample_index in (
        _iter_face_adaptive_paged_atlas_lattice_samples(plan, offsets)
    ):
        allocated[page, atlas_y, atlas_x] = True
        pages[page, atlas_y, atlas_x] = source_samples[sample_index]
    if np.any(pages[allocated] == padding):
        raise QualificationError("CAA_ADAPTIVE_BAKE_ALLOCATED_SOURCE_VIEW_UNDEFINED")
    if np.any((pages != padding) & ~allocated):
        raise QualificationError("CAA_ADAPTIVE_BAKE_SOURCE_VIEW_PADDING_CONTAMINATED")
    return pages

def _iter_face_paged_atlas_lattice_samples(layout):
    """Yield face/page/texel/sample mapping for fixed-resolution physical pages."""
    face_count = int(layout["face_count"])
    tile_resolution = int(layout["tile_resolution"])
    bleed = int(layout["bleed_px"])
    stride = int(layout["tile_stride"])
    tiles_per_axis = int(layout["tiles_per_axis"])
    faces_per_page = int(layout["faces_per_page"])
    sample_map = _sample_index_map(tile_resolution)
    for face_index in range(face_count):
        page_index = face_index // faces_per_page
        local_face = face_index % faces_per_page
        tile_x = (local_face % tiles_per_axis) * stride
        tile_y = (local_face // tiles_per_axis) * stride
        for local_y in range(stride):
            for local_x in range(stride):
                ii, jj = _nearest_triangle_lattice(
                    local_x - bleed,
                    local_y - bleed,
                    tile_resolution=tile_resolution,
                )
                yield (
                    face_index,
                    page_index,
                    tile_y + local_y,
                    tile_x + local_x,
                    sample_map[(ii, jj)],
                )


def bake_direction_atlas_pages(
    *,
    face_sample_rgba: np.ndarray,
    face_sample_provenance: np.ndarray,
    face_count: int,
    tile_resolution: int,
    bleed_px: int,
    max_page_resolution: int,
):
    """Bake one direction into deterministic fixed-resolution physical pages."""
    rgba_samples = np.asarray(face_sample_rgba, dtype=np.uint8)
    provenance_samples = np.asarray(face_sample_provenance, dtype=np.uint8)
    face_count = int(face_count)
    layout = face_atlas_paged_layout(
        face_count,
        tile_resolution=int(tile_resolution),
        bleed_px=int(bleed_px),
        max_page_resolution=int(max_page_resolution),
    )
    sample_count = int(tile_resolution) * (int(tile_resolution) + 1) // 2
    if rgba_samples.shape != (face_count * sample_count, 4):
        raise QualificationError("CAA_PAGED_BAKE_RGBA_SAMPLE_SHAPE_INVALID")
    if provenance_samples.shape != (face_count * sample_count,):
        raise QualificationError("CAA_PAGED_BAKE_PROVENANCE_SAMPLE_SHAPE_INVALID")

    rgba_samples = rgba_samples.reshape(face_count, sample_count, 4)
    provenance_samples = provenance_samples.reshape(face_count, sample_count)
    page_count = int(layout["page_count"])
    height = int(layout["page_height"])
    width = int(layout["page_width"])
    pages = np.zeros((page_count, height, width, 4), dtype=np.uint8)
    provenance_pages = np.full(
        (page_count, height, width), 255, dtype=np.uint8
    )
    allocated = np.zeros((page_count, height, width), dtype=bool)
    for face_index, page_index, atlas_y, atlas_x, sample_index in (
        _iter_face_paged_atlas_lattice_samples(layout)
    ):
        allocated[page_index, atlas_y, atlas_x] = True
        pages[page_index, atlas_y, atlas_x] = rgba_samples[
            face_index, sample_index
        ]
        provenance_pages[page_index, atlas_y, atlas_x] = provenance_samples[
            face_index, sample_index
        ]

    if np.any(provenance_pages[allocated] == 255):
        raise QualificationError("CAA_PAGED_BAKE_ALLOCATED_TILE_UNDEFINED_TEXEL")
    if np.any((provenance_pages != 255) & ~allocated):
        raise QualificationError("CAA_PAGED_BAKE_PADDING_CONTAMINATED")
    uv, face_page_index = face_paged_uv_array(layout)
    return pages, provenance_pages, uv, face_page_index, layout


def bake_direction_source_view_atlas_pages(
    *,
    face_sample_source_view: np.ndarray,
    face_count: int,
    tile_resolution: int,
    bleed_px: int,
    max_page_resolution: int,
) -> np.ndarray:
    """Bake exact source-view lineage using the identical paged texel mapping."""
    source_samples = np.asarray(face_sample_source_view, dtype=np.int16)
    face_count = int(face_count)
    layout = face_atlas_paged_layout(
        face_count,
        tile_resolution=int(tile_resolution),
        bleed_px=int(bleed_px),
        max_page_resolution=int(max_page_resolution),
    )
    sample_count = int(tile_resolution) * (int(tile_resolution) + 1) // 2
    if source_samples.shape != (face_count * sample_count,):
        raise QualificationError("CAA_PAGED_BAKE_SOURCE_VIEW_SAMPLE_SHAPE_INVALID")
    valid = (
        ((source_samples >= 0) & (source_samples < 8))
        | (source_samples == -2)
        | (source_samples == -4)
    )
    if not np.all(valid):
        raise QualificationError("CAA_PAGED_BAKE_SOURCE_VIEW_SAMPLE_VALUE_INVALID")
    source_samples = source_samples.reshape(face_count, sample_count)
    padding = np.iinfo(np.int16).min
    pages = np.full(
        (
            int(layout["page_count"]),
            int(layout["page_height"]),
            int(layout["page_width"]),
        ),
        padding,
        dtype=np.int16,
    )
    allocated = np.zeros(pages.shape, dtype=bool)
    for face_index, page_index, atlas_y, atlas_x, sample_index in (
        _iter_face_paged_atlas_lattice_samples(layout)
    ):
        allocated[page_index, atlas_y, atlas_x] = True
        pages[page_index, atlas_y, atlas_x] = source_samples[
            face_index, sample_index
        ]
    if np.any(pages[allocated] == padding):
        raise QualificationError("CAA_PAGED_BAKE_ALLOCATED_SOURCE_VIEW_UNDEFINED")
    if np.any((pages != padding) & ~allocated):
        raise QualificationError("CAA_PAGED_BAKE_SOURCE_VIEW_PADDING_CONTAMINATED")
    return pages

def bake_direction_atlas(
    *,
    face_sample_rgba: np.ndarray,
    face_sample_provenance: np.ndarray,
    face_count: int,
    tile_resolution: int,
    bleed_px: int,
):
    rgba_samples = np.asarray(face_sample_rgba, dtype=np.uint8)
    provenance_samples = np.asarray(face_sample_provenance, dtype=np.uint8)
    face_count = int(face_count)
    layout = face_atlas_layout(
        face_count,
        tile_resolution=tile_resolution,
        bleed_px=bleed_px,
    )
    sample_count = tile_resolution * (tile_resolution + 1) // 2
    if rgba_samples.shape != (face_count * sample_count, 4):
        raise QualificationError("CAA_BAKE_RGBA_SAMPLE_SHAPE_INVALID")
    if provenance_samples.shape != (face_count * sample_count,):
        raise QualificationError("CAA_BAKE_PROVENANCE_SAMPLE_SHAPE_INVALID")

    rgba_samples = rgba_samples.reshape(face_count, sample_count, 4)
    provenance_samples = provenance_samples.reshape(face_count, sample_count)
    atlas = np.zeros((layout["height"], layout["width"], 4), dtype=np.uint8)
    provenance_atlas = np.full(
        (layout["height"], layout["width"]),
        255,
        dtype=np.uint8,
    )
    allocated = np.zeros(
        (layout["height"], layout["width"]),
        dtype=bool,
    )
    for face_index, atlas_y, atlas_x, sample_index in (
        _iter_face_atlas_lattice_samples(
            face_count=face_count,
            tile_resolution=tile_resolution,
            bleed_px=bleed_px,
        )
    ):
        allocated[atlas_y, atlas_x] = True
        atlas[atlas_y, atlas_x] = rgba_samples[face_index, sample_index]
        provenance_atlas[atlas_y, atlas_x] = provenance_samples[
            face_index, sample_index
        ]

    if np.any(provenance_atlas[allocated] == 255):
        raise QualificationError("CAA_BAKE_ALLOCATED_TILE_UNDEFINED_TEXEL")
    if np.any((provenance_atlas != 255) & ~allocated):
        raise QualificationError("CAA_BAKE_UNALLOCATED_PADDING_CONTAMINATED")
    uv = face_uv_array(layout)
    return atlas, provenance_atlas, uv, layout


def bake_direction_source_view_atlas(
    *,
    face_sample_source_view: np.ndarray,
    face_count: int,
    tile_resolution: int,
    bleed_px: int,
) -> np.ndarray:
    """Bake compile-time donor/source-view lineage into the exact face atlas.

    Stored texels preserve the Stage21 source-view identity:
      0..7 -> exact source/donor view
      -2   -> compiled local harmonic appearance
      -4   -> compiler abstention on source-unsupported potential surface

    The global unallocated atlas padding uses int16 minimum and is never a
    renderable surface texel. This lineage is diagnostic/provenance authority
    only; it must never select geometry, color, or depth ordering.
    """
    source_samples = np.asarray(face_sample_source_view, dtype=np.int16)
    face_count = int(face_count)
    layout = face_atlas_layout(
        face_count,
        tile_resolution=tile_resolution,
        bleed_px=bleed_px,
    )
    sample_count = tile_resolution * (tile_resolution + 1) // 2
    if source_samples.shape != (face_count * sample_count,):
        raise QualificationError("CAA_BAKE_SOURCE_VIEW_SAMPLE_SHAPE_INVALID")
    valid = (
        ((source_samples >= 0) & (source_samples < 8))
        | (source_samples == -2)
        | (source_samples == -4)
    )
    if not np.all(valid):
        raise QualificationError("CAA_BAKE_SOURCE_VIEW_SAMPLE_VALUE_INVALID")

    source_samples = source_samples.reshape(face_count, sample_count)
    padding = np.iinfo(np.int16).min
    atlas = np.full(
        (layout["height"], layout["width"]),
        padding,
        dtype=np.int16,
    )
    allocated = np.zeros(
        (layout["height"], layout["width"]),
        dtype=bool,
    )
    for face_index, atlas_y, atlas_x, sample_index in (
        _iter_face_atlas_lattice_samples(
            face_count=face_count,
            tile_resolution=tile_resolution,
            bleed_px=bleed_px,
        )
    ):
        allocated[atlas_y, atlas_x] = True
        atlas[atlas_y, atlas_x] = source_samples[face_index, sample_index]

    if np.any(atlas[allocated] == padding):
        raise QualificationError("CAA_BAKE_ALLOCATED_SOURCE_VIEW_UNDEFINED")
    if np.any((atlas != padding) & ~allocated):
        raise QualificationError("CAA_BAKE_SOURCE_VIEW_PADDING_CONTAMINATED")
    return atlas


def straight_rgba_to_premultiplied_float(rgba_u8: np.ndarray) -> np.ndarray:
    """Compatibility wrapper: straight sRGB RGBA8 -> linear PM RGBA."""
    return straight_srgb_rgba_u8_to_premultiplied_linear(rgba_u8)


def conservative_bilinear_provenance(
    provenance_u8: np.ndarray,
    uv: np.ndarray,
) -> np.ndarray:
    """Conservative provenance for the exact bilinear color footprint.

    Provenance codes are ordered by increasing inference risk in CAA V2:
    DIRECT_SOURCE < OTHER_VIEW_SOURCE < COMPILED_LOCAL_HARMONIC
    < UNSUPPORTED_ABSTAIN < 255/padding. Any texel with nonzero
    bilinear weight contributes to the returned risk class.
    """
    source = np.asarray(provenance_u8, dtype=np.uint8)
    points = np.asarray(uv, dtype=np.float64)
    if source.ndim != 2 or points.ndim != 2 or points.shape[1] != 2:
        raise QualificationError("CAA_PROVENANCE_SAMPLE_SHAPE_INVALID")
    height, width = source.shape
    x = np.clip(points[:, 0], 0.0, 1.0) * float(width - 1)
    y = np.clip(points[:, 1], 0.0, 1.0) * float(height - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    tx = x - x0
    ty = y - y0
    weights = np.stack(
        (
            (1.0 - tx) * (1.0 - ty),
            tx * (1.0 - ty),
            (1.0 - tx) * ty,
            tx * ty,
        ),
        axis=1,
    )
    values = np.stack(
        (
            source[y0, x0],
            source[y0, x1],
            source[y1, x0],
            source[y1, x1],
        ),
        axis=1,
    ).astype(np.int16)
    active = weights > 1.0e-12
    masked = np.where(active, values, -1)
    return np.max(masked, axis=1).astype(np.uint8)


def bilinear_premultiplied_rgba(
    straight_rgba_u8: np.ndarray,
    uv: np.ndarray,
) -> np.ndarray:
    """Reference sampling in linear-light premultiplied RGBA."""
    return bilinear_premultiplied_linear_rgba(
        straight_rgba_u8,
        uv,
        normalized=True,
    )
