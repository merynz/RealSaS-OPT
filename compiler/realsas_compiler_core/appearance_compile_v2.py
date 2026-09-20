from __future__ import annotations

"""Deterministic mesh-domain Complete Appearance Authority compiler."""

import math
from typing import Mapping

import numpy as np
from scipy.spatial import cKDTree

from .appearance_authority_v2 import CAA_PROVENANCE
from .playback_full_surface_v3 import project_points_xyz_v3
from .types import QualificationError
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


def face_uv_array(layout: Mapping[str, int]) -> np.ndarray:
    faces = int(layout["face_count"])
    resolution = int(layout["tile_resolution"])
    bleed = int(layout["bleed_px"])
    stride = int(layout["tile_stride"])
    columns = int(layout["columns"])
    width = float(layout["width"])
    height = float(layout["height"])
    uv = np.zeros((faces, 3, 2), dtype=np.float32)
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


def bilinear_rgba_u8(image: np.ndarray, xy: np.ndarray) -> np.ndarray:
    rgba = np.asarray(image, dtype=np.uint8)
    points = np.asarray(xy, dtype=np.float64)
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise QualificationError("CAA_SOURCE_RGBA_INVALID")
    if points.ndim != 2 or points.shape[1] != 2:
        raise QualificationError("CAA_SAMPLE_XY_INVALID")
    h, w, _ = rgba.shape
    x = np.clip(points[:, 0], 0.0, float(w - 1))
    y = np.clip(points[:, 1], 0.0, float(h - 1))
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    tx = (x - x0).reshape(-1, 1)
    ty = (y - y0).reshape(-1, 1)
    p00 = rgba[y0, x0].astype(np.float64)
    p10 = rgba[y0, x1].astype(np.float64)
    p01 = rgba[y1, x0].astype(np.float64)
    p11 = rgba[y1, x1].astype(np.float64)
    value = (1.0 - ty) * ((1.0 - tx) * p00 + tx * p10) + ty * (
        (1.0 - tx) * p01 + tx * p11
    )
    return np.clip(np.floor(value + 0.5), 0.0, 255.0).astype(np.uint8)


def _candidate_vertex_id(vertex) -> str:
    value = getattr(vertex, "candidate_vertex_id", None)
    if value is None:
        raise QualificationError("CAA_REQUIRES_CANONICAL_MESH_CANDIDATE")
    return str(value)


def _surface_sample_geometry(candidate, barycentric: np.ndarray):
    vertices = {
        _candidate_vertex_id(vertex): np.asarray(vertex.P, dtype=np.float64)
        for vertex in candidate.vertices
    }
    components = {
        _candidate_vertex_id(vertex): str(vertex.component_id)
        for vertex in candidate.vertices
    }
    positions = []
    face_indices = []
    component_ids = []
    normals = []
    for face_index, face in enumerate(candidate.faces):
        ids = tuple(map(str, face))
        if len(ids) != 3 or any(vertex_id not in vertices for vertex_id in ids):
            raise QualificationError("CAA_CANDIDATE_FACE_INVALID")
        component_set = {components[vertex_id] for vertex_id in ids}
        if len(component_set) != 1:
            raise QualificationError("CAA_FACE_CROSSES_COMPONENT")
        xyz = np.stack([vertices[vertex_id] for vertex_id in ids], axis=0)
        normal = np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0])
        norm = float(np.linalg.norm(normal))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise QualificationError("CAA_FACE_DEGENERATE_BEFORE_COMPILE")
        normal = normal / norm
        sample_xyz = barycentric @ xyz
        positions.append(sample_xyz)
        face_indices.extend([face_index] * len(barycentric))
        component_ids.extend([next(iter(component_set))] * len(barycentric))
        normals.append(normal)
    return (
        np.concatenate(positions, axis=0).astype(np.float64),
        np.asarray(face_indices, dtype=np.int32),
        tuple(component_ids),
        np.asarray(normals, dtype=np.float64),
    )


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


def compile_deterministic_caa(
    *,
    candidate,
    cameras,
    source_rgba_by_view: Mapping[int, np.ndarray],
    foreground_mask_by_view: Mapping[int, np.ndarray],
    tile_resolution: int,
    source_lock_policy: Mapping[str, object],
) -> dict:
    barycentric = triangular_barycentric_samples(tile_resolution)
    positions, sample_face, sample_component, face_normals = _surface_sample_geometry(
        candidate, barycentric
    )
    sample_count = len(positions)
    per_face_samples = len(barycentric)
    face_count = len(candidate.faces)
    if sample_count != face_count * per_face_samples:
        raise QualificationError("CAA_SAMPLE_ACCOUNTING_DRIFT")

    min_cos = float(source_lock_policy["min_abs_normal_camera_cos"])
    erosion = int(source_lock_policy["boundary_safe_erosion_px"])
    min_alpha = int(source_lock_policy["min_source_alpha_u8"])

    direct_valid = np.zeros((8, sample_count), dtype=bool)
    direct_rgba = np.zeros((8, sample_count, 4), dtype=np.uint8)
    source_xy = np.full((8, sample_count, 2), np.nan, dtype=np.float32)

    camera_by_view = {int(camera.view_index): camera for camera in cameras}
    if set(camera_by_view) != set(range(8)):
        raise QualificationError("CAA_DETERMINISTIC_REQUIRES_V0_V7_CAMERAS")

    samples_per_face = per_face_samples
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
        source_xy[view] = xy.astype(np.float32)

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
        angle_safe = np.repeat(face_cos >= min_cos, samples_per_face)

        appearance_support = (
            (foreground_safe & alpha_foreground_safe)
            | (background_safe & alpha_background_safe)
        )
        valid = in_bounds & visible & appearance_support & angle_safe
        direct_valid[view] = valid
        if np.any(valid):
            direct_rgba[view, valid] = bilinear_rgba_u8(image, xy[valid])

    rgba = np.zeros_like(direct_rgba)
    provenance = np.full(
        (8, sample_count),
        255,
        dtype=np.uint8,
    )
    source_view = np.full((8, sample_count), -1, dtype=np.int16)

    for target in range(8):
        direct = direct_valid[target]
        rgba[target, direct] = direct_rgba[target, direct]
        provenance[target, direct] = CAA_PROVENANCE["DIRECT_SOURCE"]
        source_view[target, direct] = target

        missing = ~direct
        for donor in _circular_view_order(target):
            if donor == target:
                continue
            take = missing & direct_valid[donor]
            if np.any(take):
                rgba[target, take] = direct_rgba[donor, take]
                provenance[target, take] = CAA_PROVENANCE["OTHER_VIEW_SOURCE"]
                source_view[target, take] = donor
                missing[take] = False

        observed_any = provenance[target] != 255
        if not np.any(observed_any):
            raise QualificationError("CAA_NO_SOURCE_OBSERVATION_ANYWHERE")

        component_values = sorted(set(sample_component))
        for component in component_values:
            component_mask = np.asarray(
                [value == component for value in sample_component],
                dtype=bool,
            )
            need = missing & component_mask
            if not np.any(need):
                continue
            donors = observed_any & component_mask
            if not np.any(donors):
                continue
            donor_indices = np.flatnonzero(donors)
            query_indices = np.flatnonzero(need)
            tree = cKDTree(positions[donor_indices])
            _distance, nearest = tree.query(positions[query_indices], k=1)
            nearest_indices = donor_indices[np.asarray(nearest, dtype=np.int64)]
            rgba[target, query_indices] = rgba[target, nearest_indices]
            source_view[target, query_indices] = source_view[target, nearest_indices]
            provenance[target, query_indices] = CAA_PROVENANCE[
                "COMPILED_NEAREST_SURFACE"
            ]
            missing[query_indices] = False

        if np.any(missing):
            donor_indices = np.flatnonzero(provenance[target] != 255)
            query_indices = np.flatnonzero(missing)
            tree = cKDTree(positions[donor_indices])
            _distance, nearest = tree.query(positions[query_indices], k=1)
            nearest_indices = donor_indices[np.asarray(nearest, dtype=np.int64)]
            rgba[target, query_indices] = rgba[target, nearest_indices]
            source_view[target, query_indices] = source_view[target, nearest_indices]
            provenance[target, query_indices] = CAA_PROVENANCE[
                "COMPILED_GLOBAL_SURFACE"
            ]
            missing[query_indices] = False

        if np.any(missing) or np.any(provenance[target] == 255):
            raise QualificationError("CAA_TOTALITY_FAILURE_AFTER_DETERMINISTIC_COMPILE")

    counts = {
        name: int(np.count_nonzero(provenance == code))
        for name, code in CAA_PROVENANCE.items()
    }
    return {
        "barycentric": barycentric.astype(np.float32),
        "sample_positions": positions.astype(np.float32),
        "sample_face_index": sample_face,
        "direct_valid": direct_valid,
        "direct_rgba": direct_rgba,
        "source_xy": source_xy,
        "rgba": rgba,
        "provenance": provenance,
        "source_view": source_view,
        "component_ids": tuple(sorted(set(sample_component))),
        "sample_component": tuple(sample_component),
        "counts": counts,
        "face_count": face_count,
        "sample_count_per_face": per_face_samples,
    }
