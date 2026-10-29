#!/usr/bin/env python3
"""Independently validate Truth139 visibility/occlusion teacher labels with CPU ray stacks.

The dense Graphics teacher is GPU-raster-derived and therefore cannot validate itself.  This court
samples deterministic surface/view pairs, re-measures the exact orthographic ray against the source
mesh with the NumPy Graphics oracle, and checks the stored local visibility consequence.  RGB,
alpha, material ids, owner ids and semantic labels never participate in the verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from models.iris.truth.graphics_oracle_v1 import camera_from_mapping, measure_orthographic_ray_stack_v1
from tools.corpus.build_truth139_dense_appearance_gbuffer_v1 import sha256

SCHEMA = "RealSaS.Truth139VisibilityOcclusionCourt.v1"
GRAPHICS_SCHEMA = "RealSaS.Truth139DenseGraphicsTeacher.v1"
EXPECTED_STATUS = "PASS_DENSE_GRAPHICS_TEACHER_CANDIDATE__INDEPENDENT_COURTS_REQUIRED"
VISIBLE = 1
OCCLUDED = 2
BACKGROUND = 3
AMBIGUOUS = 4


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _bound_path(ref: dict, code: str) -> Path:
    path = Path(str(ref.get("path") or "")).resolve()
    expected = str(ref.get("sha256") or "")
    if not path.is_file() or len(expected) != 64 or sha256(path) != expected:
        raise ValueError(code)
    return path


def _sample_indices(indices: np.ndarray, maximum: int) -> np.ndarray:
    ids = np.asarray(indices, dtype=np.int64).reshape(-1)
    if len(ids) <= maximum:
        return ids
    positions = np.linspace(0, len(ids) - 1, maximum, dtype=np.int64)
    return ids[positions]


def _projected_pixel(point: np.ndarray, camera: dict) -> int:
    p = np.asarray(point, dtype=np.float64)
    origin = np.asarray(camera["origin"], dtype=np.float64)
    right = np.asarray(camera["right"], dtype=np.float64)
    up = np.asarray(camera["screen_up"], dtype=np.float64)
    half = float(camera["half_extent"])
    resolution = int(camera["resolution"])
    relative = p - origin
    gx = float(relative @ right) / half
    gy = -float(relative @ up) / half
    if abs(gx) > 1.0 or abs(gy) > 1.0:
        return -1
    x = int(math.floor((gx + 1.0) * 0.5 * resolution))
    y = int(math.floor((gy + 1.0) * 0.5 * resolution))
    if not (0 <= x < resolution and 0 <= y < resolution):
        return -1
    return y * resolution + x


def validate(
    manifest: dict,
    *,
    maximum_pairs_per_target_view: int = 128,
    minimum_validated_per_target_view: int = 16,
) -> dict:
    if manifest.get("schema") != GRAPHICS_SCHEMA or manifest.get("status") != EXPECTED_STATUS:
        raise ValueError("TRUTH139_VISIBILITY_COURT_PARENT_NOT_CANDIDATE")
    subject_id = str(manifest.get("subject_id") or "")
    if not subject_id:
        raise ValueError("TRUTH139_VISIBILITY_COURT_SUBJECT")
    if maximum_pairs_per_target_view < 1 or minimum_validated_per_target_view < 1:
        raise ValueError("TRUTH139_VISIBILITY_COURT_SAMPLE_POLICY")

    parents = dict(manifest.get("parents") or {})
    geometry_path = _bound_path(dict(parents.get("geometry_npz") or {}), "TRUTH139_VISIBILITY_COURT_GEOMETRY_DRIFT")
    camera_path = _bound_path(dict(parents.get("camera_bundle") or {}), "TRUTH139_VISIBILITY_COURT_CAMERA_DRIFT")
    visibility_ref = dict(manifest.get("visibility_occlusion") or {})
    visibility_path = _bound_path(
        {"path": visibility_ref.get("path"), "sha256": visibility_ref.get("sha256")},
        "TRUTH139_VISIBILITY_COURT_TEACHER_DRIFT",
    )

    with np.load(geometry_path, allow_pickle=False) as z:
        vertices = np.asarray(z["rest_vertices_source_world"], dtype=np.float64)
        faces = np.asarray(z["faces"], dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or faces.ndim != 2 or faces.shape[1:] != (3,):
        raise ValueError("TRUTH139_VISIBILITY_COURT_GEOMETRY_LAYOUT")
    geometry_scale = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    if not math.isfinite(geometry_scale) or geometry_scale <= 0:
        raise ValueError("TRUTH139_VISIBILITY_COURT_GEOMETRY_SCALE")
    tie_epsilon = max(geometry_scale * 1.0e-7, 1.0e-9)
    surface_tolerance = float(visibility_ref.get("surface_tolerance", 0.0))
    depth_relative = float(visibility_ref.get("depth_tolerance_relative_to_camera_half_extent", 0.0))
    if not math.isfinite(surface_tolerance) or surface_tolerance <= 0:
        raise ValueError("TRUTH139_VISIBILITY_COURT_SURFACE_TOLERANCE")
    if not math.isfinite(depth_relative) or depth_relative <= 0:
        raise ValueError("TRUTH139_VISIBILITY_COURT_DEPTH_TOLERANCE")

    camera_payload = json.loads(camera_path.read_text())
    cameras = sorted(list(camera_payload.get("cameras") or ()), key=lambda row: int(row.get("view_index", -1)))
    if len(cameras) != 8 or [int(row.get("view_index", -1)) for row in cameras] != list(range(8)):
        raise ValueError("TRUTH139_VISIBILITY_COURT_CAMERA_COUNT")

    with np.load(visibility_path, allow_pickle=False) as z:
        point = np.asarray(z["surface_xyz_source_world"], dtype=np.float64)
        classes = np.asarray(z["target_visibility_class"], dtype=np.uint8)
        pixels = np.asarray(z["target_projected_pixel_linear_index"], dtype=np.int64)
        front_triangle = np.asarray(z["target_front_triangle_id"], dtype=np.int64)
        stored_depth_gap = np.asarray(z["target_depth_gap"], dtype=np.float64)
    if point.ndim != 2 or point.shape[1:] != (3,) or classes.shape != (len(point), 8):
        raise ValueError("TRUTH139_VISIBILITY_COURT_ARRAY_LAYOUT")
    if pixels.shape != classes.shape or front_triangle.shape != classes.shape or stored_depth_gap.shape != classes.shape:
        raise ValueError("TRUTH139_VISIBILITY_COURT_MATRIX_LAYOUT")

    target_results = []
    total_validated = 0
    total_tie_skipped = 0
    global_class_validated = {"VISIBLE_SAME_SURFACE": 0, "OCCLUDED_BY_FRONT_SURFACE": 0, "PROJECTED_TO_BACKGROUND_OR_UNSUPPORTED": 0}
    max_visible_surface_error = 0.0
    max_depth_gap_error = 0.0

    for target_view in range(8):
        camera_mapping = cameras[target_view]
        camera = camera_from_mapping(camera_mapping)
        eligible = np.flatnonzero(np.isin(classes[:, target_view], [VISIBLE, OCCLUDED, BACKGROUND]))
        if len(eligible) < minimum_validated_per_target_view:
            raise ValueError(
                f"TRUTH139_VISIBILITY_COURT_TOO_FEW_ELIGIBLE:V{target_view}:{len(eligible)}<{minimum_validated_per_target_view}"
            )
        picked = _sample_indices(eligible, int(maximum_pairs_per_target_view))
        validated = 0
        tie_skipped = 0
        class_validated = {"VISIBLE_SAME_SURFACE": 0, "OCCLUDED_BY_FRONT_SURFACE": 0, "PROJECTED_TO_BACKGROUND_OR_UNSUPPORTED": 0}
        view_surface_error = 0.0
        view_depth_error = 0.0
        depth_epsilon = max(float(camera.half_extent) * depth_relative, 1.0e-6)

        for sample_index in picked:
            label = int(classes[sample_index, target_view])
            stored_pixel = int(pixels[sample_index, target_view])
            independent_pixel = _projected_pixel(point[sample_index], camera_mapping)
            if stored_pixel != independent_pixel:
                raise ValueError(
                    f"TRUTH139_VISIBILITY_COURT_PROJECTED_PIXEL_MISMATCH:V{target_view}:S{int(sample_index)}:"
                    f"stored={stored_pixel}:cpu={independent_pixel}"
                )
            if stored_pixel < 0:
                raise ValueError(f"TRUTH139_VISIBILITY_COURT_ELIGIBLE_PIXEL_MISSING:V{target_view}:S{int(sample_index)}")
            pixel_y, pixel_x = divmod(stored_pixel, camera.resolution)
            stack = measure_orthographic_ray_stack_v1(
                vertices,
                faces,
                camera,
                int(pixel_x),
                int(pixel_y),
                tie_epsilon=tie_epsilon,
            )
            if not stack.complete:
                raise ValueError(f"TRUTH139_VISIBILITY_COURT_CAMERA_NOT_BEFORE_MESH:V{target_view}")
            if stack.has_depth_tie:
                tie_skipped += 1
                continue

            relative = point[sample_index] - np.asarray(camera.origin, dtype=np.float64)
            query_depth = float(relative @ np.asarray(camera.forward, dtype=np.float64))
            stored_front = int(front_triangle[sample_index, target_view])

            if label == BACKGROUND:
                if stack.hits:
                    raise ValueError(
                        f"TRUTH139_VISIBILITY_COURT_GPU_BACKGROUND_CPU_HIT:V{target_view}:S{int(sample_index)}:"
                        f"face={stack.hits[0].face_index}"
                    )
                class_validated["PROJECTED_TO_BACKGROUND_OR_UNSUPPORTED"] += 1
                global_class_validated["PROJECTED_TO_BACKGROUND_OR_UNSUPPORTED"] += 1
                validated += 1
                continue

            if not stack.hits:
                raise ValueError(f"TRUTH139_VISIBILITY_COURT_GPU_FRONT_CPU_MISS:V{target_view}:S{int(sample_index)}")
            hit = stack.hits[0]
            if int(hit.face_index) != stored_front:
                raise ValueError(
                    f"TRUTH139_VISIBILITY_COURT_FRONT_FACE_MISMATCH:V{target_view}:S{int(sample_index)}:"
                    f"gpu={stored_front}:cpu={int(hit.face_index)}"
                )
            face = faces[int(hit.face_index)]
            hit_point = np.sum(vertices[face] * np.asarray(hit.barycentric, dtype=np.float64)[:, None], axis=0)
            surface_error = float(np.linalg.norm(hit_point - point[sample_index]))
            independent_gap = query_depth - float(hit.depth)
            stored_gap = float(stored_depth_gap[sample_index, target_view])
            depth_gap_error = abs(independent_gap - stored_gap) if math.isfinite(stored_gap) else math.inf

            if label == VISIBLE:
                if surface_error > surface_tolerance:
                    raise ValueError(
                        f"TRUTH139_VISIBILITY_COURT_VISIBLE_SURFACE_MISMATCH:V{target_view}:S{int(sample_index)}:{surface_error}"
                    )
                class_validated["VISIBLE_SAME_SURFACE"] += 1
                global_class_validated["VISIBLE_SAME_SURFACE"] += 1
                view_surface_error = max(view_surface_error, surface_error)
                max_visible_surface_error = max(max_visible_surface_error, surface_error)
            elif label == OCCLUDED:
                if independent_gap <= depth_epsilon:
                    raise ValueError(
                        f"TRUTH139_VISIBILITY_COURT_OCCLUSION_DEPTH_NOT_POSITIVE:V{target_view}:S{int(sample_index)}:"
                        f"gap={independent_gap}:eps={depth_epsilon}"
                    )
                if not math.isfinite(depth_gap_error) or depth_gap_error > max(depth_epsilon * 2.0, 1.0e-6):
                    raise ValueError(
                        f"TRUTH139_VISIBILITY_COURT_OCCLUSION_DEPTH_DRIFT:V{target_view}:S{int(sample_index)}:{depth_gap_error}"
                    )
                class_validated["OCCLUDED_BY_FRONT_SURFACE"] += 1
                global_class_validated["OCCLUDED_BY_FRONT_SURFACE"] += 1
                view_depth_error = max(view_depth_error, depth_gap_error)
                max_depth_gap_error = max(max_depth_gap_error, depth_gap_error)
            else:
                raise ValueError("TRUTH139_VISIBILITY_COURT_UNEXPECTED_LABEL")
            validated += 1

        if validated < minimum_validated_per_target_view:
            raise ValueError(
                f"TRUTH139_VISIBILITY_COURT_TOO_FEW_WITNESSES:V{target_view}:{validated}<{minimum_validated_per_target_view}"
            )
        total_validated += validated
        total_tie_skipped += tie_skipped
        target_results.append(
            {
                "target_view_index": target_view,
                "eligible_pair_count": int(len(eligible)),
                "sampled_pair_count": int(len(picked)),
                "validated_pair_count": int(validated),
                "depth_tie_skipped": int(tie_skipped),
                "validated_class_counts": class_validated,
                "maximum_visible_surface_l2_error": view_surface_error,
                "maximum_occlusion_depth_gap_error": view_depth_error,
                "depth_epsilon": depth_epsilon,
            }
        )

    payload = {
        "schema": SCHEMA,
        "date": "2026-10-29",
        "status": "PASS_INDEPENDENT_CPU_RAY_ORACLE_AGREES_WITH_VISIBILITY_OCCLUSION_TEACHER",
        "subject_id": subject_id,
        "graphics_teacher_identity_sha256": str(manifest.get("graphics_teacher_identity_sha256") or ""),
        "policy": {
            "oracle": "NUMPY_COMPLETE_ORTHOGRAPHIC_RAY_STACK",
            "selection": "DETERMINISTIC_EVEN_COVERAGE_PER_TARGET_VIEW",
            "maximum_pairs_per_target_view": int(maximum_pairs_per_target_view),
            "minimum_validated_per_target_view": int(minimum_validated_per_target_view),
            "depth_ties": "SKIP_NOT_FORCE_CLASSIFY",
            "ambiguous_teacher_rows": "EXCLUDED_FROM_PASS_COUNT",
            "rgb_alpha_material_owner_or_semantic_evidence_used": False,
            "global_draw_order_assumed": False,
        },
        "target_view_courts": target_results,
        "validated_class_counts": global_class_validated,
        "total_validated_pair_count": int(total_validated),
        "total_depth_tie_skipped": int(total_tie_skipped),
        "maximum_visible_surface_l2_error": max_visible_surface_error,
        "maximum_occlusion_depth_gap_error": max_depth_gap_error,
        "training_performed": False,
        "qualification_minted": False,
    }
    payload["court_identity_sha256"] = _digest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-graphics-teacher-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--maximum-pairs-per-target-view", type=int, default=128)
    parser.add_argument("--minimum-validated-per-target-view", type=int, default=16)
    args = parser.parse_args()
    manifest = json.loads(args.dense_graphics_teacher_manifest.resolve().read_text())
    result = validate(
        manifest,
        maximum_pairs_per_target_view=args.maximum_pairs_per_target_view,
        minimum_validated_per_target_view=args.minimum_validated_per_target_view,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.with_suffix(output.suffix + ".partial")
    staged.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "subject_id": result["subject_id"],
                "total_validated_pair_count": result["total_validated_pair_count"],
                "validated_class_counts": result["validated_class_counts"],
                "court_identity_sha256": result["court_identity_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
