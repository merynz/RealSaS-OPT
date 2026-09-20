from __future__ import annotations

"""Bake CAA triangular surface samples into unique-face transport atlases."""

import math

import numpy as np

from .appearance_compile_v2 import face_atlas_layout, face_uv_array
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
    sample_map = _sample_index_map(tile_resolution)
    stride = int(layout["tile_stride"])
    columns = int(layout["columns"])
    bleed = int(layout["bleed_px"])

    for face_index in range(face_count):
        tile_x = (face_index % columns) * stride
        tile_y = (face_index // columns) * stride
        for local_y in range(stride):
            for local_x in range(stride):
                lattice_i = local_x - bleed
                lattice_j = local_y - bleed
                ii, jj = _nearest_triangle_lattice(
                    lattice_i,
                    lattice_j,
                    tile_resolution=tile_resolution,
                )
                sample_index = sample_map[(ii, jj)]
                atlas[tile_y + local_y, tile_x + local_x] = rgba_samples[
                    face_index, sample_index
                ]
                provenance_atlas[
                    tile_y + local_y, tile_x + local_x
                ] = provenance_samples[face_index, sample_index]

    if np.any(provenance_atlas == 255):
        raise QualificationError("CAA_BAKE_ATLAS_UNDEFINED_TEXEL")
    uv = face_uv_array(layout)
    return atlas, provenance_atlas, uv, layout


def straight_rgba_to_premultiplied_float(rgba_u8: np.ndarray) -> np.ndarray:
    rgba = np.asarray(rgba_u8, dtype=np.float32) / 255.0
    if rgba.shape[-1] != 4:
        raise QualificationError("CAA_PREMULTIPLY_RGBA_SHAPE_INVALID")
    out = rgba.copy()
    out[..., :3] *= out[..., 3:4]
    return out


def bilinear_premultiplied_rgba(
    straight_rgba_u8: np.ndarray,
    uv: np.ndarray,
) -> np.ndarray:
    """Reference sampling: convert texels to PM, then bilinear interpolate PM."""
    image = straight_rgba_to_premultiplied_float(straight_rgba_u8)
    points = np.asarray(uv, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise QualificationError("CAA_BILINEAR_UV_INVALID")
    height, width, _ = image.shape
    x = np.clip(points[:, 0], 0.0, 1.0) * float(width - 1)
    y = np.clip(points[:, 1], 0.0, 1.0) * float(height - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    tx = (x - x0).reshape(-1, 1)
    ty = (y - y0).reshape(-1, 1)
    p00 = image[y0, x0]
    p10 = image[y0, x1]
    p01 = image[y1, x0]
    p11 = image[y1, x1]
    return (1.0 - ty) * ((1.0 - tx) * p00 + tx * p10) + ty * (
        (1.0 - tx) * p01 + tx * p11
    )
