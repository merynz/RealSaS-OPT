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
    resolutions = tuple(sorted(set(int(value) for value in candidate_resolutions)))
    if not resolutions or resolutions[0] < 4:
        raise QualificationError("CAA_TILE_CANDIDATES_INVALID")
    limit = float(max_source_pixels_per_atlas_texel)
    if not math.isfinite(limit) or limit <= 0.0:
        raise QualificationError("CAA_TILE_DENSITY_LIMIT_INVALID")

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
    face_count = len(candidate.faces)
    for resolution in resolutions:
        source_pixels_per_atlas_texel = worst_sigma / float(resolution - 1)
        layout = face_atlas_layout(
            face_count,
            tile_resolution=resolution,
            bleed_px=int(bleed_px),
        )
        density_passed = source_pixels_per_atlas_texel <= limit
        capacity_passed = (
            int(layout["width"]) <= int(max_atlas_resolution)
            and int(layout["height"]) <= int(max_atlas_resolution)
        )
        rows.append(
            {
                "tile_resolution": resolution,
                "worst_source_pixels_per_atlas_texel": source_pixels_per_atlas_texel,
                "density_passed": bool(density_passed),
                "atlas_width": int(layout["width"]),
                "atlas_height": int(layout["height"]),
                "capacity_passed": bool(capacity_passed),
            }
        )
        if selected is None and density_passed and capacity_passed:
            selected = resolution

    if selected is None:
        raise QualificationError("CAA_TILE_DENSITY_OR_CAPACITY_UNSATISFIED")
    selected_layout = face_atlas_layout(
        face_count,
        tile_resolution=selected,
        bleed_px=int(bleed_px),
    )
    stride = int(selected_layout["tile_stride"])
    per_axis = int(max_atlas_resolution) // stride
    return {
        "mode": "PROJECTED_SOURCE_DENSITY_V1",
        "selected_tile_resolution": int(selected),
        "max_source_pixels_per_atlas_texel": limit,
        "worst_projected_barycentric_sigma_px": worst_sigma,
        "worst_view_index": int(worst_view),
        "worst_face_index": int(worst_face),
        "visible_face_observation_count": int(visible_face_observation_count),
        "max_supported_face_count": int(per_axis * per_axis),
        "candidates": rows,
    }


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
    completion_quality_policy: Mapping[str, object] | None = None,
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
    direct_pm_linear = np.zeros((8, sample_count, 4), dtype=np.float32)
    source_xy = np.full((8, sample_count, 2), np.nan, dtype=np.float32)
    face_support_by_view = np.zeros((8, face_count), dtype=np.float64)

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
        face_support_by_view[view] = face_cos
        angle_safe = np.repeat(face_cos >= min_cos, samples_per_face)

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
            direct_pm_linear[view, valid] = sampled_pm.astype(np.float32)

    rgba = np.zeros_like(direct_rgba)
    provenance = np.full(
        (8, sample_count),
        255,
        dtype=np.uint8,
    )
    source_view = np.full((8, sample_count), -1, dtype=np.int16)
    surface_neighbors = surface_sample_neighbors(
        positions=positions,
        face_count=face_count,
        tile_resolution=tile_resolution,
    )
    completion_rows = []

    for target in range(8):
        direct = direct_valid[target]
        rgba[target, direct] = direct_rgba[target, direct]
        provenance[target, direct] = CAA_PROVENANCE["DIRECT_SOURCE"]
        source_view[target, direct] = target

        missing = ~direct
        best_score = np.full(sample_count, -1.0, dtype=np.float64)
        best_view = np.full(sample_count, -1, dtype=np.int16)
        for donor in _circular_view_order(target):
            if donor == target:
                continue
            eligible = missing & direct_valid[donor]
            if not np.any(eligible):
                continue
            support = face_support_by_view[donor, sample_face]
            improve = eligible & (support > best_score + 1.0e-12)
            if np.any(improve):
                best_score[improve] = support[improve]
                best_view[improve] = donor
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

        for component_id in sorted(set(sample_component)):
            component_mask = np.asarray(
                [value == component_id for value in sample_component],
                dtype=bool,
            )
            if np.any(missing & component_mask) and not np.any(
                (~missing) & component_mask
            ):
                raise QualificationError(
                    "CAA_COMPONENT_WITHOUT_SOURCE_OBSERVATION"
                )

        stats = bounded_surface_harmonic_fill(
            rgba=rgba[target],
            provenance=provenance[target],
            source_view=source_view[target],
            missing=missing,
            sample_component=sample_component,
            neighbors=surface_neighbors,
            max_region_samples=max_harmonic_region,
            max_graph_hops=max_harmonic_hops,
        )
        completion_rows.append(
            {
                "target_view_index": target,
                **stats,
            }
        )
        if np.any(provenance[target] == 255):
            raise QualificationError("CAA_TOTALITY_FAILURE_AFTER_HARMONIC_COMPILE")

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
        "direct_pm_linear": direct_pm_linear,
        "source_xy": source_xy,
        "rgba": rgba,
        "provenance": provenance,
        "source_view": source_view,
        "component_ids": tuple(sorted(set(sample_component))),
        "sample_component": tuple(sample_component),
        "counts": counts,
        "completion_rows": tuple(completion_rows),
        "face_count": face_count,
        "sample_count_per_face": per_face_samples,
    }
