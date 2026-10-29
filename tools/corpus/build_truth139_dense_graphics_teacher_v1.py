#!/usr/bin/env python3
"""Materialize dense Truth139 Graphics teacher truth from admitted source observations.

This is deliberately a teacher-data producer, not a model forward path.  It enriches the already
independently-auditable dense front-surface G-buffer with canonical addressing, normals, facing,
observation provenance and deterministic cross-view visibility/occlusion labels.

The producer never invents semantic body-part labels, hidden owners or a global draw order.  A
visibility label is a local geometric fact for one canonical surface sample and one camera.  The
current controlled Truth139 lane consists of eight real source observations, but provenance is
explicit so future 1/2-view product lanes may add constructed internal views without granting those
views source authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from tools.corpus.build_truth139_dense_appearance_gbuffer_v1 import sha256

SCHEMA = "RealSaS.Truth139DenseGraphicsTeacher.v1"
VIEW_SCHEMA = "RealSaS.Truth139DenseGraphicsView.v1"
VISIBILITY_SCHEMA = "RealSaS.Truth139VisibilityOcclusionTeacher.v1"
GBUFFER_SCHEMA = "RealSaS.Truth139DenseAppearanceGBuffer.v1"
SNAPSHOT_SCHEMA = "RealSaS.ImportedSourceSceneSnapshot.v2"
EXPECTED_VIEW_COUNT = 8
EXPECTED_RESOLUTION = 1024
GRAPHICS_CONTRACT = "canonical/TRUTH139_DENSE_GRAPHICS_TEACHER_V1.json"

PROVENANCE_CODE = {
    "SOURCE_OBSERVATION": 1,
    "CONSTRUCTED_OBSERVATION": 2,
    "LEARNED_FILL": 3,
}
VISIBILITY_CODE = {
    "OUTSIDE_VIEW": 0,
    "VISIBLE_SAME_SURFACE": 1,
    "OCCLUDED_BY_FRONT_SURFACE": 2,
    "PROJECTED_TO_BACKGROUND_OR_UNSUPPORTED": 3,
    "RASTER_GEOMETRY_DISAGREEMENT_OR_AMBIGUOUS": 4,
}


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _bound_path(ref: Mapping[str, object], code: str) -> Path:
    path = Path(str(ref.get("path") or "")).resolve()
    expected = str(ref.get("sha256") or "")
    if not path.is_file() or len(expected) != 64 or sha256(path) != expected:
        raise ValueError(code)
    return path


def _matrix(value, code: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(code)
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-10, rtol=0.0):
        raise ValueError(code + "_AFFINE")
    if np.linalg.matrix_rank(matrix[:3, :3]) != 3:
        raise ValueError(code + "_SINGULAR")
    return matrix


def _load_snapshot(path: Path) -> tuple[dict, np.ndarray]:
    value = json.loads(path.read_text())
    if value.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError("TRUTH139_GRAPHICS_SNAPSHOT_SCHEMA")
    transform = _matrix(value.get("source_world_to_canonical"), "TRUTH139_GRAPHICS_SOURCE_TO_CANONICAL")
    inverse = _matrix(value.get("canonical_to_source_world"), "TRUTH139_GRAPHICS_CANONICAL_TO_SOURCE")
    if not np.allclose(transform @ inverse, np.eye(4), atol=1.0e-6, rtol=0.0):
        raise ValueError("TRUTH139_GRAPHICS_SNAPSHOT_INVERSE_DRIFT")
    gram = transform[:3, :3].T @ transform[:3, :3]
    if not np.allclose(gram / np.trace(gram), np.eye(3) / 3.0, atol=1.0e-6, rtol=0.0):
        raise ValueError("TRUTH139_GRAPHICS_NONUNIFORM_CANONICAL_TRANSFORM")
    if np.linalg.slogdet(transform[:3, :3])[0] <= 0:
        raise ValueError("TRUTH139_GRAPHICS_REFLECTED_CANONICAL_TRANSFORM")
    return value, transform


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1:] != (3,) or not np.isfinite(p).all():
        raise ValueError("TRUTH139_GRAPHICS_POINT_LAYOUT")
    hom = np.concatenate((p, np.ones((len(p), 1), dtype=np.float64)), axis=1)
    out = hom @ np.asarray(matrix, dtype=np.float64).T
    if not np.allclose(out[:, 3], 1.0, atol=1.0e-8, rtol=0.0):
        raise ValueError("TRUTH139_GRAPHICS_POINT_HOMOGENEOUS_DRIFT")
    return out[:, :3].astype(np.float32)


def transform_normals(normals: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    n = np.asarray(normals, dtype=np.float64)
    if n.ndim != 2 or n.shape[1:] != (3,) or not np.isfinite(n).all():
        raise ValueError("TRUTH139_GRAPHICS_NORMAL_LAYOUT")
    normal_matrix = np.linalg.inv(np.asarray(matrix, dtype=np.float64)[:3, :3]).T
    out = n @ normal_matrix.T
    length = np.linalg.norm(out, axis=1, keepdims=True)
    valid = length[:, 0] > 1.0e-12
    result = np.zeros_like(out)
    result[valid] = out[valid] / length[valid]
    return result.astype(np.float32)


def _bary_weights(uv: np.ndarray) -> np.ndarray:
    value = np.asarray(uv, dtype=np.float64)
    if value.ndim != 2 or value.shape[1:] != (2,) or not np.isfinite(value).all():
        raise ValueError("TRUTH139_GRAPHICS_BARY_LAYOUT")
    return np.stack((value[:, 0], value[:, 1], 1.0 - value[:, 0] - value[:, 1]), axis=1)


def reconstruct_surface(vertices: np.ndarray, faces: np.ndarray, triangle: np.ndarray, uv: np.ndarray) -> np.ndarray:
    tri = np.asarray(triangle, dtype=np.int64).reshape(-1)
    weights = _bary_weights(uv)
    if len(tri) != len(weights) or np.any(tri < 0) or np.any(tri >= len(faces)):
        raise ValueError("TRUTH139_GRAPHICS_SURFACE_ADDRESS")
    return np.sum(np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)[tri]] * weights[..., None], axis=1).astype(np.float32)


def geometric_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    raw = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(raw, axis=1, keepdims=True)
    valid = length[:, 0] > 1.0e-12
    out = np.zeros_like(raw)
    out[valid] = raw[valid] / length[valid]
    return out.astype(np.float32)


def project_grid(points: np.ndarray, camera: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(points, dtype=np.float64)
    origin = np.asarray(camera["origin"], dtype=np.float64)
    right = np.asarray(camera["right"], dtype=np.float64)
    up = np.asarray(camera["screen_up"], dtype=np.float64)
    forward = np.asarray(camera["forward"], dtype=np.float64)
    half = float(camera["half_extent"])
    if p.ndim != 2 or p.shape[1:] != (3,) or any(x.shape != (3,) for x in (origin, right, up, forward)):
        raise ValueError("TRUTH139_GRAPHICS_CAMERA_PROJECTION_LAYOUT")
    if not math.isfinite(half) or half <= 0:
        raise ValueError("TRUTH139_GRAPHICS_CAMERA_HALF_EXTENT")
    relative = p - origin[None]
    grid = np.stack(((relative @ right) / half, -(relative @ up) / half), axis=1)
    depth = relative @ forward
    return grid.astype(np.float32), depth.astype(np.float32)


def grid_to_nearest_pixel(grid: np.ndarray, resolution: int) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=np.float64)
    x = np.floor((g[:, 0] + 1.0) * 0.5 * resolution).astype(np.int64)
    y = np.floor((g[:, 1] + 1.0) * 0.5 * resolution).astype(np.int64)
    return x, y


def lookup_exact_rows(pixel_linear_index: np.ndarray, query_pixel: np.ndarray) -> np.ndarray:
    pixels = np.asarray(pixel_linear_index, dtype=np.int64).reshape(-1)
    query = np.asarray(query_pixel, dtype=np.int64).reshape(-1)
    if len(pixels) and np.any(pixels[1:] <= pixels[:-1]):
        raise ValueError("TRUTH139_GRAPHICS_GBUFFER_PIXELS_NOT_STRICTLY_SORTED")
    out = np.full(len(query), -1, dtype=np.int64)
    if not len(pixels):
        return out
    pos = np.searchsorted(pixels, query)
    clipped = np.minimum(pos, len(pixels) - 1)
    hit = (pos < len(pixels)) & (pixels[clipped] == query)
    out[hit] = pos[hit]
    return out


def lookup_near_rows(pixel_linear_index: np.ndarray, x: np.ndarray, y: np.ndarray, resolution: int, radius: int) -> np.ndarray:
    if radius < 0:
        raise ValueError("TRUTH139_GRAPHICS_LOOKUP_RADIUS")
    pixels = np.asarray(pixel_linear_index, dtype=np.int64).reshape(-1)
    qx = np.asarray(x, dtype=np.int64).reshape(-1)
    qy = np.asarray(y, dtype=np.int64).reshape(-1)
    width = (2 * radius + 1) ** 2
    out = np.full((len(qx), width), -1, dtype=np.int64)
    column = 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            xx = qx + dx
            yy = qy + dy
            valid = (xx >= 0) & (xx < resolution) & (yy >= 0) & (yy < resolution)
            target = yy * resolution + xx
            out[:, column] = lookup_exact_rows(pixels, target)
            out[~valid, column] = -1
            column += 1
    return out


def deterministic_sample_rows(count: int, maximum: int, *, subject_id: str, source_view: int) -> np.ndarray:
    if count < 1 or maximum < 1:
        raise ValueError("TRUTH139_GRAPHICS_SAMPLE_POLICY")
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    seed = int(hashlib.sha256(f"{subject_id}|graphics_visibility|V{source_view}".encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(count, size=maximum, replace=False).astype(np.int64))


def _load_provenance(path: Path | None) -> dict[int, str]:
    if path is None:
        return {i: "SOURCE_OBSERVATION" for i in range(EXPECTED_VIEW_COUNT)}
    value = json.loads(path.resolve().read_text())
    rows = value.get("views") if isinstance(value, dict) else None
    if not isinstance(rows, list):
        raise ValueError("TRUTH139_GRAPHICS_PROVENANCE_SCHEMA")
    result = {}
    for row in rows:
        index = int(row.get("view_index", -1))
        label = str(row.get("provenance") or "")
        if index in result or index < 0 or index >= EXPECTED_VIEW_COUNT or label not in PROVENANCE_CODE:
            raise ValueError("TRUTH139_GRAPHICS_PROVENANCE_ROW")
        result[index] = label
    if set(result) != set(range(EXPECTED_VIEW_COUNT)):
        raise ValueError("TRUTH139_GRAPHICS_PROVENANCE_VIEW_COUNT")
    return result


def _write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".partial.npz")
    np.savez_compressed(staged, **arrays)
    staged.replace(path)


def build(
    *,
    dense_gbuffer_manifest: Path,
    source_scene_snapshot: Path,
    output_dir: Path,
    provenance_json: Path | None = None,
    visibility_samples_per_source_view: int = 8192,
    correspondence_radius_px: int = 2,
    surface_tolerance_relative: float = 3.0e-3,
    depth_tolerance_relative: float = 2.0e-5,
) -> dict:
    manifest_path = dense_gbuffer_manifest.resolve()
    snapshot_path = source_scene_snapshot.resolve()
    output_dir = output_dir.resolve()
    if not manifest_path.is_file() or not snapshot_path.is_file():
        raise FileNotFoundError("TRUTH139_GRAPHICS_INPUT_MISSING")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != GBUFFER_SCHEMA:
        raise ValueError("TRUTH139_GRAPHICS_GBUFFER_SCHEMA")
    if manifest.get("status") != "PASS_DENSE_SURFACE_ADDRESSED_APPEARANCE_GBUFFER__STAGE48_INTERVENTION_WITNESS_STILL_REQUIRED":
        raise ValueError("TRUTH139_GRAPHICS_GBUFFER_NOT_PASS")
    subject_id = str(manifest.get("subject_id") or "")
    if not subject_id:
        raise ValueError("TRUTH139_GRAPHICS_SUBJECT_ID")
    snapshot, source_to_canonical = _load_snapshot(snapshot_path)
    geometry_path = _bound_path(dict(manifest.get("geometry_npz") or {}), "TRUTH139_GRAPHICS_GEOMETRY_DRIFT")
    camera_path = _bound_path(dict(manifest.get("camera_bundle") or {}), "TRUTH139_GRAPHICS_CAMERA_DRIFT")
    if str(snapshot.get("source_sha256") or "") and len(str(snapshot.get("source_sha256"))) != 64:
        raise ValueError("TRUTH139_GRAPHICS_SNAPSHOT_SOURCE_SHA")

    with np.load(geometry_path, allow_pickle=False) as z:
        vertices = np.asarray(z["rest_vertices_source_world"], dtype=np.float32)
        faces = np.asarray(z["faces"], dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or faces.ndim != 2 or faces.shape[1:] != (3,):
        raise ValueError("TRUTH139_GRAPHICS_GEOMETRY_LAYOUT")
    if np.any(faces < 0) or np.any(faces >= len(vertices)) or not np.isfinite(vertices).all():
        raise ValueError("TRUTH139_GRAPHICS_GEOMETRY_INVALID")
    geometry_scale = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    if not math.isfinite(geometry_scale) or geometry_scale <= 0:
        raise ValueError("TRUTH139_GRAPHICS_GEOMETRY_SCALE")
    surface_tolerance = max(geometry_scale * float(surface_tolerance_relative), 1.0e-6)

    camera_payload = json.loads(camera_path.read_text())
    if camera_payload.get("schema") != "RealSaS.CameraProjectionBundle.v1":
        raise ValueError("TRUTH139_GRAPHICS_CAMERA_SCHEMA")
    cameras = sorted(list(camera_payload.get("cameras") or ()), key=lambda row: int(row.get("view_index", -1)))
    if len(cameras) != EXPECTED_VIEW_COUNT or [int(x.get("view_index", -1)) for x in cameras] != list(range(EXPECTED_VIEW_COUNT)):
        raise ValueError("TRUTH139_GRAPHICS_CAMERA_CARDINALITY")
    if any(int(x.get("resolution", -1)) != EXPECTED_RESOLUTION for x in cameras):
        raise ValueError("TRUTH139_GRAPHICS_RESOLUTION_DRIFT")

    provenance = _load_provenance(provenance_json)
    face_normal_source = geometric_face_normals(vertices, faces)
    face_normal_canonical = transform_normals(face_normal_source, source_to_canonical)
    views = sorted(list(manifest.get("views") or ()), key=lambda row: int(row.get("view_index", -1)))
    if len(views) != EXPECTED_VIEW_COUNT:
        raise ValueError("TRUTH139_GRAPHICS_VIEW_COUNT")

    output_dir.mkdir(parents=True, exist_ok=True)
    dense_views: list[dict] = []
    caches: list[dict] = []
    for view in views:
        view_index = int(view["view_index"])
        gbuffer_path = _bound_path(
            {"path": view["gbuffer_path"], "sha256": view["gbuffer_sha256"]},
            "TRUTH139_GRAPHICS_GBUFFER_VIEW_DRIFT",
        )
        with np.load(gbuffer_path, allow_pickle=False) as z:
            pixels = np.asarray(z["pixel_linear_index"], dtype=np.int64).reshape(-1)
            triangle = np.asarray(z["triangle_id"], dtype=np.int64).reshape(-1)
            uv = np.asarray(z["barycentric_uv"], dtype=np.float32)
            depth = np.asarray(z["depth_source_camera"], dtype=np.float32).reshape(-1)
            rgba = np.asarray(z["source_rgba_linear"], dtype=np.float32)
            source_foreground = np.asarray(z["source_foreground"], dtype=np.bool_).reshape(-1)
            direct_opaque = np.asarray(z["direct_opaque_rgba_valid"], dtype=np.bool_).reshape(-1)
            alpha_boundary = np.asarray(z["alpha_boundary"], dtype=np.bool_).reshape(-1)
        if not (
            triangle.shape == pixels.shape == depth.shape == source_foreground.shape == direct_opaque.shape == alpha_boundary.shape
            and uv.shape == (len(pixels), 2)
            and rgba.shape == (len(pixels), 4)
        ):
            raise ValueError("TRUTH139_GRAPHICS_GBUFFER_VIEW_LAYOUT")
        point_source = reconstruct_surface(vertices, faces, triangle, uv)
        point_canonical = transform_points(point_source, source_to_canonical)
        normal_source = face_normal_source[triangle]
        normal_canonical = face_normal_canonical[triangle]
        camera_forward = np.asarray(cameras[view_index]["forward"], dtype=np.float32)
        signed_facing = np.sum(normal_source * (-camera_forward[None]), axis=1).astype(np.float32)
        facing_confidence = np.abs(signed_facing).astype(np.float32)
        provenance_code = np.full(len(pixels), PROVENANCE_CODE[provenance[view_index]], dtype=np.uint8)
        direct_source = direct_opaque & (provenance_code == PROVENANCE_CODE["SOURCE_OBSERVATION"])

        out = output_dir / f"V{view_index}_DENSE_GRAPHICS_TEACHER.npz"
        _write_npz(
            out,
            pixel_linear_index=pixels.astype(np.int32),
            triangle_id=triangle.astype(np.int32),
            barycentric_uv=uv.astype(np.float32),
            surface_xyz_source_world=point_source.astype(np.float32),
            surface_xyz_canonical=point_canonical.astype(np.float32),
            depth_source_camera=depth.astype(np.float32),
            normal_source_world=normal_source.astype(np.float32),
            normal_canonical=normal_canonical.astype(np.float32),
            front_facing_signed_cosine=signed_facing,
            view_facing_confidence=facing_confidence,
            source_rgba_linear=rgba.astype(np.float32),
            source_foreground=source_foreground,
            direct_source_rgba_valid=direct_source.astype(np.bool_),
            alpha_boundary=alpha_boundary,
            observation_provenance_code=provenance_code,
            resolution=np.asarray([EXPECTED_RESOLUTION, EXPECTED_RESOLUTION], dtype=np.int32),
            origin=np.asarray(["TOP_LEFT"]),
        )
        dense_views.append(
            {
                "schema": VIEW_SCHEMA,
                "view_index": view_index,
                "observation_provenance": provenance[view_index],
                "teacher_path": str(out),
                "teacher_sha256": sha256(out),
                "row_count": int(len(pixels)),
                "direct_source_rgba_count": int(np.count_nonzero(direct_source)),
                "alpha_boundary_count": int(np.count_nonzero(alpha_boundary)),
            }
        )
        caches.append(
            {
                "pixels": pixels,
                "triangle": triangle,
                "uv": uv,
                "depth": depth,
                "source_foreground": source_foreground,
                "point_source": point_source,
                "point_canonical": point_canonical,
                "normal_source": normal_source,
            }
        )

    sample_source_view = []
    sample_source_row = []
    sample_source_pixel = []
    sample_triangle = []
    sample_uv = []
    sample_point_source = []
    sample_point_canonical = []
    sample_normal_source = []
    for source_view, cache in enumerate(caches):
        chosen = deterministic_sample_rows(
            len(cache["pixels"]),
            int(visibility_samples_per_source_view),
            subject_id=subject_id,
            source_view=source_view,
        )
        sample_source_view.append(np.full(len(chosen), source_view, dtype=np.int16))
        sample_source_row.append(chosen.astype(np.int32))
        sample_source_pixel.append(cache["pixels"][chosen].astype(np.int32))
        sample_triangle.append(cache["triangle"][chosen].astype(np.int32))
        sample_uv.append(cache["uv"][chosen].astype(np.float32))
        sample_point_source.append(cache["point_source"][chosen].astype(np.float32))
        sample_point_canonical.append(cache["point_canonical"][chosen].astype(np.float32))
        sample_normal_source.append(cache["normal_source"][chosen].astype(np.float32))

    source_view_arr = np.concatenate(sample_source_view)
    source_row_arr = np.concatenate(sample_source_row)
    source_pixel_arr = np.concatenate(sample_source_pixel)
    source_triangle_arr = np.concatenate(sample_triangle)
    source_uv_arr = np.concatenate(sample_uv)
    point_source_arr = np.concatenate(sample_point_source)
    point_canonical_arr = np.concatenate(sample_point_canonical)
    normal_source_arr = np.concatenate(sample_normal_source)
    sample_count = len(source_view_arr)
    if sample_count == 0:
        raise ValueError("TRUTH139_GRAPHICS_NO_VISIBILITY_SAMPLES")

    visibility = np.full((sample_count, EXPECTED_VIEW_COUNT), VISIBILITY_CODE["OUTSIDE_VIEW"], dtype=np.uint8)
    target_pixel = np.full((sample_count, EXPECTED_VIEW_COUNT), -1, dtype=np.int32)
    target_front_triangle = np.full((sample_count, EXPECTED_VIEW_COUNT), -1, dtype=np.int32)
    target_front_row = np.full((sample_count, EXPECTED_VIEW_COUNT), -1, dtype=np.int32)
    target_depth_gap = np.full((sample_count, EXPECTED_VIEW_COUNT), np.nan, dtype=np.float32)
    target_facing = np.zeros((sample_count, EXPECTED_VIEW_COUNT), dtype=np.float32)
    target_query_source_support = np.zeros((sample_count, EXPECTED_VIEW_COUNT), dtype=np.bool_)
    correspondence_match = np.zeros((sample_count, EXPECTED_VIEW_COUNT), dtype=np.bool_)
    correspondence_row = np.full((sample_count, EXPECTED_VIEW_COUNT), -1, dtype=np.int32)
    correspondence_surface_error = np.full((sample_count, EXPECTED_VIEW_COUNT), np.inf, dtype=np.float32)

    for target_view, cache in enumerate(caches):
        camera = cameras[target_view]
        grid, query_depth = project_grid(point_source_arr, camera)
        x, y = grid_to_nearest_pixel(grid, EXPECTED_RESOLUTION)
        in_frame = (np.abs(grid[:, 0]) <= 1.0) & (np.abs(grid[:, 1]) <= 1.0) & (x >= 0) & (x < EXPECTED_RESOLUTION) & (y >= 0) & (y < EXPECTED_RESOLUTION)
        query_pixel = y * EXPECTED_RESOLUTION + x
        exact = lookup_exact_rows(cache["pixels"], query_pixel)
        has_front = in_frame & (exact >= 0)
        target_pixel[in_frame, target_view] = query_pixel[in_frame].astype(np.int32)
        target_front_row[has_front, target_view] = exact[has_front].astype(np.int32)
        target_front_triangle[has_front, target_view] = cache["triangle"][exact[has_front]].astype(np.int32)

        forward = np.asarray(camera["forward"], dtype=np.float32)
        target_facing[:, target_view] = np.abs(np.sum(normal_source_arr * (-forward[None]), axis=1)).astype(np.float32)
        visibility[in_frame & ~has_front, target_view] = VISIBILITY_CODE["PROJECTED_TO_BACKGROUND_OR_UNSUPPORTED"]

        if np.any(has_front):
            ids = np.flatnonzero(has_front)
            front_point = cache["point_source"][exact[ids]]
            surface_error = np.linalg.norm(front_point - point_source_arr[ids], axis=1)
            same = surface_error <= surface_tolerance
            depth_gap = query_depth[ids] - cache["depth"][exact[ids]]
            depth_eps = max(float(camera["half_extent"]) * float(depth_tolerance_relative), 1.0e-6)
            visible_ids = ids[same]
            occluded_ids = ids[(~same) & (depth_gap > depth_eps)]
            ambiguous_ids = ids[(~same) & ~(depth_gap > depth_eps)]
            visibility[visible_ids, target_view] = VISIBILITY_CODE["VISIBLE_SAME_SURFACE"]
            visibility[occluded_ids, target_view] = VISIBILITY_CODE["OCCLUDED_BY_FRONT_SURFACE"]
            visibility[ambiguous_ids, target_view] = VISIBILITY_CODE["RASTER_GEOMETRY_DISAGREEMENT_OR_AMBIGUOUS"]
            target_depth_gap[ids, target_view] = depth_gap.astype(np.float32)
            target_query_source_support[visible_ids, target_view] = cache["source_foreground"][exact[visible_ids]]

        near = lookup_near_rows(cache["pixels"], x, y, EXPECTED_RESOLUTION, int(correspondence_radius_px))
        best_error = np.full(sample_count, np.inf, dtype=np.float32)
        best_row = np.full(sample_count, -1, dtype=np.int64)
        for column in range(near.shape[1]):
            row = near[:, column]
            valid = in_frame & (row >= 0)
            if not np.any(valid):
                continue
            ids = np.flatnonzero(valid)
            error = np.linalg.norm(cache["point_source"][row[ids]] - point_source_arr[ids], axis=1).astype(np.float32)
            better = error < best_error[ids]
            better_ids = ids[better]
            best_error[better_ids] = error[better]
            best_row[better_ids] = row[better_ids]
        matched = best_error <= surface_tolerance
        correspondence_match[matched, target_view] = True
        correspondence_row[matched, target_view] = best_row[matched].astype(np.int32)
        correspondence_surface_error[:, target_view] = best_error

    support_count = np.sum(visibility == VISIBILITY_CODE["VISIBLE_SAME_SURFACE"], axis=1).astype(np.uint8)
    source_support_count = np.sum(target_query_source_support, axis=1).astype(np.uint8)
    visibility_path = output_dir / "VISIBILITY_OCCLUSION_TEACHER.npz"
    _write_npz(
        visibility_path,
        source_view_index=source_view_arr,
        source_surface_row=source_row_arr,
        source_pixel_linear_index=source_pixel_arr,
        source_triangle_id=source_triangle_arr,
        source_barycentric_uv=source_uv_arr,
        surface_xyz_source_world=point_source_arr,
        surface_xyz_canonical=point_canonical_arr,
        target_visibility_class=visibility,
        target_projected_pixel_linear_index=target_pixel,
        target_front_triangle_id=target_front_triangle,
        target_front_row=target_front_row,
        target_depth_gap=target_depth_gap,
        target_view_facing_confidence=target_facing,
        target_query_source_support=target_query_source_support,
        cross_view_same_surface_match=correspondence_match,
        cross_view_match_row=correspondence_row,
        cross_view_match_surface_error=correspondence_surface_error,
        visible_support_count=support_count,
        source_observation_support_count=source_support_count,
        visibility_class_codes=np.asarray([json.dumps(VISIBILITY_CODE, sort_keys=True)]),
        provenance_codes=np.asarray([json.dumps(PROVENANCE_CODE, sort_keys=True)]),
    )

    class_counts = {
        name: int(np.count_nonzero(visibility == code))
        for name, code in VISIBILITY_CODE.items()
    }
    payload = {
        "schema": SCHEMA,
        "date": "2026-10-29",
        "status": "PASS_DENSE_GRAPHICS_TEACHER_CANDIDATE__INDEPENDENT_COURTS_REQUIRED",
        "subject_id": subject_id,
        "graphics_contract": GRAPHICS_CONTRACT,
        "parents": {
            "dense_gbuffer_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
            "dense_gbuffer_authority_identity_sha256": str(manifest.get("authority_identity_sha256") or ""),
            "source_scene_snapshot": {"path": str(snapshot_path), "sha256": sha256(snapshot_path)},
            "geometry_npz": {"path": str(geometry_path), "sha256": sha256(geometry_path)},
            "camera_bundle": {"path": str(camera_path), "sha256": sha256(camera_path)},
        },
        "observation_provenance": {
            "codes": PROVENANCE_CODE,
            "views": [{"view_index": i, "provenance": provenance[i]} for i in range(EXPECTED_VIEW_COUNT)],
            "constructed_views_have_source_authority": False,
        },
        "dense_views": dense_views,
        "visibility_occlusion": {
            "schema": VISIBILITY_SCHEMA,
            "path": str(visibility_path),
            "sha256": sha256(visibility_path),
            "sample_count": int(sample_count),
            "samples_per_source_view_maximum": int(visibility_samples_per_source_view),
            "class_codes": VISIBILITY_CODE,
            "class_counts": class_counts,
            "surface_tolerance": float(surface_tolerance),
            "surface_tolerance_relative_to_bbox_diagonal": float(surface_tolerance_relative),
            "depth_tolerance_relative_to_camera_half_extent": float(depth_tolerance_relative),
            "correspondence_radius_px": int(correspondence_radius_px),
            "semantic_occluder_labels_used": False,
            "global_draw_order_claimed": False,
        },
        "teacher_policy": {
            "graphics_is_parent_domain": True,
            "legacy_caa_is_global_authority": False,
            "source_color_authority_is_partial_and_provenance_bound": True,
            "view_identity_preserved": True,
            "semantic_part_taxonomy_used": False,
            "material_or_owner_ids_exposed_as_student_targets": False,
            "missing_truth_zero_filled": False,
            "training_performed": False,
        },
        "independent_courts_required": [
            "PASS_INDEPENDENT_CPU_RAY_ORACLE_AGREES_WITH_DENSE_GPU_GBUFFER",
            "PASS_INDEPENDENT_CPU_RAY_ORACLE_AGREES_WITH_VISIBILITY_OCCLUSION_TEACHER",
            "STAGE48_RENDERER_INTERVENTION_WITNESS",
        ],
        "qualification_minted": False,
    }
    payload["graphics_teacher_identity_sha256"] = _digest(payload)
    manifest_out = output_dir / "DENSE_GRAPHICS_TEACHER_MANIFEST.json"
    manifest_out.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "subject_id": subject_id,
                "visibility_sample_count": sample_count,
                "visibility_class_counts": class_counts,
                "graphics_teacher_identity_sha256": payload["graphics_teacher_identity_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-gbuffer-manifest", required=True, type=Path)
    parser.add_argument("--source-scene-snapshot", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--provenance-json", type=Path)
    parser.add_argument("--visibility-samples-per-source-view", type=int, default=8192)
    parser.add_argument("--correspondence-radius-px", type=int, default=2)
    parser.add_argument("--surface-tolerance-relative", type=float, default=3.0e-3)
    parser.add_argument("--depth-tolerance-relative", type=float, default=2.0e-5)
    args = parser.parse_args()
    build(
        dense_gbuffer_manifest=args.dense_gbuffer_manifest,
        source_scene_snapshot=args.source_scene_snapshot,
        output_dir=args.output_dir,
        provenance_json=args.provenance_json,
        visibility_samples_per_source_view=args.visibility_samples_per_source_view,
        correspondence_radius_px=args.correspondence_radius_px,
        surface_tolerance_relative=args.surface_tolerance_relative,
        depth_tolerance_relative=args.depth_tolerance_relative,
    )


if __name__ == "__main__":
    main()
