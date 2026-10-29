#!/usr/bin/env python3
"""Independently validate Truth139 visibility/occlusion teacher relations with the CPU ray oracle.

The court uses only exact source geometry, the frozen camera bundle and primitive teacher claims.
It never consumes RGB, materials, source-owner ids, semantic parts, attachments or layer names.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from models.iris.truth.graphics_oracle_v1 import camera_from_mapping, measure_orthographic_ray_stack_v1
from tools.corpus.build_truth139_dense_graphics_gbuffer_v1 import sha256, transform_points
from tools.corpus.build_truth139_visibility_occlusion_teacher_v1 import VISIBILITY

SCHEMA = "RealSaS.Truth139VisibilityOcclusionCourt.v1"
TEACHER_SCHEMA = "RealSaS.Truth139VisibilityOcclusionTeacher.v1"
GRAPHICS_SCHEMA = "RealSaS.Truth139DenseGraphicsGBuffer.v1"
EXPECTED_STATUS = "PASS_VISIBILITY_OCCLUSION_TEACHER__INDEPENDENT_CPU_ORACLE_COURT_PENDING"
PASS_STATUS = "PASS_INDEPENDENT_CPU_RAY_ORACLE_AGREES_WITH_VISIBILITY_OCCLUSION_TEACHER"


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _bound_path(ref: dict, code: str) -> Path:
    path = Path(str(ref.get("path") or "")).resolve()
    expected = str(ref.get("sha256") or "")
    if not path.is_file() or len(expected) != 64 or sha256(path) != expected:
        raise ValueError(code)
    return path


def _even_rows(count: int, maximum: int) -> np.ndarray:
    if count < 0 or maximum < 1:
        raise ValueError("TRUTH139_VISIBILITY_COURT_SAMPLE_POLICY")
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, maximum, dtype=np.int64)


def _camera_bundle(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "RealSaS.CameraProjectionBundle.v1":
        raise ValueError("TRUTH139_VISIBILITY_COURT_CAMERA_SCHEMA")
    cameras = sorted(list(payload.get("cameras") or ()), key=lambda row: int(row.get("view_index", -1)))
    if len(cameras) != 8 or [int(row.get("view_index", -1)) for row in cameras] != list(range(8)):
        raise ValueError("TRUTH139_VISIBILITY_COURT_CAMERA_COUNT")
    resolution = int(cameras[0].get("resolution", 0))
    if resolution <= 0 or any(int(row.get("resolution", 0)) != resolution for row in cameras):
        raise ValueError("TRUTH139_VISIBILITY_COURT_CAMERA_RESOLUTION")
    return cameras


def _load_teacher(path: Path) -> dict[str, np.ndarray]:
    required = {
        "surface_xyz_source_world",
        "surface_xyz_canonical",
        "target_visibility_class",
        "target_projected_pixel_linear_index",
        "target_front_triangle_id",
        "target_depth_gap",
        "target_view_facing_confidence",
        "target_source_support",
        "cross_view_same_surface_match",
    }
    with np.load(path, allow_pickle=False) as z:
        if not required.issubset(set(z.files)):
            raise ValueError("TRUTH139_VISIBILITY_COURT_TEACHER_FIELDS")
        value = {name: np.asarray(z[name]) for name in required}
    n = len(value["surface_xyz_source_world"])
    if value["surface_xyz_source_world"].shape != (n, 3) or value["surface_xyz_canonical"].shape != (n, 3):
        raise ValueError("TRUTH139_VISIBILITY_COURT_POINT_LAYOUT")
    for name in (
        "target_visibility_class",
        "target_projected_pixel_linear_index",
        "target_front_triangle_id",
        "target_depth_gap",
        "target_view_facing_confidence",
        "target_source_support",
        "cross_view_same_surface_match",
    ):
        if value[name].shape != (n, 8):
            raise ValueError("TRUTH139_VISIBILITY_COURT_TARGET_LAYOUT:" + name)
    if not np.isfinite(value["surface_xyz_source_world"]).all() or not np.isfinite(value["surface_xyz_canonical"]).all():
        raise ValueError("TRUTH139_VISIBILITY_COURT_POINT_NONFINITE")
    return value


def _face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    raw = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(raw, axis=1)
    out = np.zeros_like(raw, dtype=np.float64)
    valid = length > 1.0e-12
    out[valid] = raw[valid] / length[valid, None]
    return out


def validate(
    teacher_manifest: dict,
    *,
    maximum_samples_per_target_view: int = 64,
    minimum_total_ray_witnesses: int = 16,
    depth_gap_abs_tolerance: float = 2.0e-4,
    facing_abs_tolerance: float = 2.0e-4,
) -> dict:
    if teacher_manifest.get("schema") != TEACHER_SCHEMA or teacher_manifest.get("status") != EXPECTED_STATUS:
        raise ValueError("TRUTH139_VISIBILITY_COURT_PARENT_NOT_PENDING_PASS")
    subject_id = str(teacher_manifest.get("subject_id") or "")
    if not subject_id:
        raise ValueError("TRUTH139_VISIBILITY_COURT_SUBJECT")
    if maximum_samples_per_target_view < 1 or minimum_total_ray_witnesses < 1:
        raise ValueError("TRUTH139_VISIBILITY_COURT_SAMPLE_ARGUMENT")

    teacher_path = _bound_path(dict(teacher_manifest.get("teacher_npz") or {}), "TRUTH139_VISIBILITY_COURT_TEACHER_DRIFT")
    graphics_path = _bound_path(dict(teacher_manifest.get("graphics_manifest") or {}), "TRUTH139_VISIBILITY_COURT_GRAPHICS_DRIFT")
    camera_path = _bound_path(dict(teacher_manifest.get("camera_bundle") or {}), "TRUTH139_VISIBILITY_COURT_CAMERA_DRIFT")
    graphics = json.loads(graphics_path.read_text())
    if graphics.get("schema") != GRAPHICS_SCHEMA or str(graphics.get("subject_id") or "") != subject_id:
        raise ValueError("TRUTH139_VISIBILITY_COURT_GRAPHICS_PARENT")

    geometry_path = _bound_path(dict(graphics.get("geometry_npz") or {}), "TRUTH139_VISIBILITY_COURT_GEOMETRY_DRIFT")
    with np.load(geometry_path, allow_pickle=False) as z:
        vertices = np.asarray(z["rest_vertices_source_world"], dtype=np.float64)
        faces = np.asarray(z["faces"], dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or faces.ndim != 2 or faces.shape[1:] != (3,):
        raise ValueError("TRUTH139_VISIBILITY_COURT_GEOMETRY_LAYOUT")
    if len(vertices) == 0 or len(faces) == 0 or np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("TRUTH139_VISIBILITY_COURT_GEOMETRY_INVALID")

    source_to_canonical = np.asarray(graphics["coordinate_contract"]["source_world_to_canonical"], dtype=np.float64)
    if source_to_canonical.shape != (4, 4) or not np.isfinite(source_to_canonical).all():
        raise ValueError("TRUTH139_VISIBILITY_COURT_COORDINATE_CONTRACT")
    canonical_vertices = transform_points(vertices, source_to_canonical).astype(np.float64)
    normals = _face_normals(vertices, faces)
    cameras = _camera_bundle(camera_path)
    teacher = _load_teacher(teacher_path)
    n = len(teacher["surface_xyz_source_world"])
    resolution = int(cameras[0]["resolution"])

    mesh_scale = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    if not math.isfinite(mesh_scale) or mesh_scale <= 0.0:
        raise ValueError("TRUTH139_VISIBILITY_COURT_MESH_SCALE")
    tie_epsilon = max(mesh_scale * 1.0e-7, 1.0e-9)
    tolerances = dict(teacher_manifest.get("tolerances") or {})
    surface_tolerance_pixels = float(tolerances.get("surface_tolerance_pixels", 0.0))
    depth_epsilon_pixels = float(tolerances.get("depth_epsilon_pixels", 0.0))
    canonical_scale = float(tolerances.get("canonical_uniform_scale", 0.0))
    if surface_tolerance_pixels <= 0 or depth_epsilon_pixels <= 0 or canonical_scale <= 0:
        raise ValueError("TRUTH139_VISIBILITY_COURT_TEACHER_TOLERANCE")

    rows = _even_rows(n, int(maximum_samples_per_target_view))
    class_checked = {name: 0 for name in VISIBILITY}
    class_ray_validated = {name: 0 for name in VISIBILITY}
    tie_skipped = 0
    total_ray_validated = 0
    max_gap_error = 0.0
    max_surface_error = 0.0
    max_facing_error = 0.0
    target_rows = []

    for target_view, camera_payload in enumerate(cameras):
        camera = camera_from_mapping(camera_payload)
        pixel_world = 2.0 * float(camera.half_extent) / float(camera.resolution)
        surface_tolerance = surface_tolerance_pixels * pixel_world * canonical_scale
        depth_epsilon = depth_epsilon_pixels * pixel_world
        checked = ray_validated = outside_checked = ambiguous_checked = 0

        for i in rows.tolist():
            cls = int(teacher["target_visibility_class"][i, target_view])
            if cls not in VISIBILITY.values():
                raise ValueError("TRUTH139_VISIBILITY_COURT_UNKNOWN_CLASS")
            name = next(key for key, value in VISIBILITY.items() if value == cls)
            class_checked[name] += 1
            checked += 1
            pixel = int(teacher["target_projected_pixel_linear_index"][i, target_view])
            tri_claim = int(teacher["target_front_triangle_id"][i, target_view])
            same_claim = bool(teacher["cross_view_same_surface_match"][i, target_view])

            if cls == VISIBILITY["OUTSIDE_VIEW"]:
                if pixel != -1 or tri_claim != -1 or same_claim:
                    raise ValueError("TRUTH139_VISIBILITY_COURT_OUTSIDE_ENCODING")
                outside_checked += 1
                continue
            if cls == VISIBILITY["RASTER_GEOMETRY_DISAGREEMENT_OR_AMBIGUOUS"]:
                if same_claim:
                    raise ValueError("TRUTH139_VISIBILITY_COURT_AMBIGUOUS_SAME_SURFACE")
                ambiguous_checked += 1
                continue
            if pixel < 0 or pixel >= resolution * resolution:
                raise ValueError("TRUTH139_VISIBILITY_COURT_PIXEL_RANGE")
            py, px = divmod(pixel, resolution)
            stack = measure_orthographic_ray_stack_v1(vertices, faces, camera, int(px), int(py), tie_epsilon=tie_epsilon)
            if not stack.complete:
                raise ValueError(f"TRUTH139_VISIBILITY_COURT_CAMERA_NOT_BEFORE_MESH:V{target_view}")
            if stack.has_depth_tie:
                tie_skipped += 1
                continue

            if cls == VISIBILITY["PROJECTED_TO_BACKGROUND_OR_UNSUPPORTED"]:
                if stack.hits:
                    raise ValueError(f"TRUTH139_VISIBILITY_COURT_BACKGROUND_CPU_HIT:V{target_view}:P{pixel}")
                if tri_claim != -1 or same_claim:
                    raise ValueError("TRUTH139_VISIBILITY_COURT_BACKGROUND_ENCODING")
                ray_validated += 1
                total_ray_validated += 1
                class_ray_validated[name] += 1
                continue

            if not stack.hits:
                raise ValueError(f"TRUTH139_VISIBILITY_COURT_TEACHER_HIT_CPU_MISS:V{target_view}:P{pixel}")
            hit = stack.hits[0]
            if int(hit.face_index) != tri_claim:
                raise ValueError(
                    f"TRUTH139_VISIBILITY_COURT_FRONT_FACE_MISMATCH:V{target_view}:P{pixel}:"
                    f"teacher={tri_claim}:cpu={int(hit.face_index)}"
                )

            sample_depth = float((teacher["surface_xyz_source_world"][i] - np.asarray(camera.origin)) @ np.asarray(camera.forward))
            cpu_gap = sample_depth - float(hit.depth)
            teacher_gap = float(teacher["target_depth_gap"][i, target_view])
            gap_error = abs(cpu_gap - teacher_gap)
            if not math.isfinite(teacher_gap) or gap_error > max(depth_gap_abs_tolerance, depth_epsilon * 0.05):
                raise ValueError(f"TRUTH139_VISIBILITY_COURT_DEPTH_GAP_MISMATCH:V{target_view}:P{pixel}:{gap_error}")
            max_gap_error = max(max_gap_error, gap_error)

            normal = normals[int(hit.face_index)]
            view_direction = -np.asarray(camera.forward, dtype=np.float64)
            view_direction /= np.linalg.norm(view_direction)
            cpu_facing = abs(float(np.dot(normal, view_direction)))
            teacher_facing = float(teacher["target_view_facing_confidence"][i, target_view])
            facing_error = abs(cpu_facing - teacher_facing)
            if facing_error > facing_abs_tolerance:
                raise ValueError(f"TRUTH139_VISIBILITY_COURT_FACING_MISMATCH:V{target_view}:P{pixel}:{facing_error}")
            max_facing_error = max(max_facing_error, facing_error)

            if cls == VISIBILITY["OCCLUDED_BY_FRONT_SURFACE"]:
                if cpu_gap <= depth_epsilon or same_claim:
                    raise ValueError(f"TRUTH139_VISIBILITY_COURT_OCCLUSION_RELATION:V{target_view}:P{pixel}:{cpu_gap}")
            elif cls == VISIBILITY["VISIBLE_SAME_SURFACE"]:
                if not same_claim:
                    raise ValueError("TRUTH139_VISIBILITY_COURT_VISIBLE_SAME_SURFACE_BIT")
                bary = np.asarray(hit.barycentric, dtype=np.float64)
                source_hit = np.sum(vertices[faces[int(hit.face_index)]] * bary[:, None], axis=0)
                canonical_hit = np.sum(canonical_vertices[faces[int(hit.face_index)]] * bary[:, None], axis=0)
                source_sample = np.asarray(teacher["surface_xyz_source_world"][i], dtype=np.float64)
                canonical_sample = np.asarray(teacher["surface_xyz_canonical"][i], dtype=np.float64)
                source_error = float(np.linalg.norm(source_hit - source_sample))
                canonical_error = float(np.linalg.norm(canonical_hit - canonical_sample))
                if canonical_error > surface_tolerance + max(mesh_scale * 1.0e-6, 1.0e-7):
                    raise ValueError(f"TRUTH139_VISIBILITY_COURT_VISIBLE_SURFACE_MISMATCH:V{target_view}:P{pixel}:{canonical_error}")
                max_surface_error = max(max_surface_error, canonical_error, source_error * canonical_scale)
            else:
                raise AssertionError("TRUTH139_VISIBILITY_COURT_CLASS_SWITCH")

            ray_validated += 1
            total_ray_validated += 1
            class_ray_validated[name] += 1

        target_rows.append({
            "target_view_index": target_view,
            "sampled_teacher_rows": int(checked),
            "ray_validated": int(ray_validated),
            "outside_checked": int(outside_checked),
            "ambiguous_checked": int(ambiguous_checked),
        })

    if total_ray_validated < minimum_total_ray_witnesses:
        raise ValueError(
            f"TRUTH139_VISIBILITY_COURT_TOO_FEW_RAY_WITNESSES:{total_ray_validated}<{minimum_total_ray_witnesses}"
        )

    payload = {
        "schema": SCHEMA,
        "date": "2026-10-29",
        "status": PASS_STATUS,
        "subject_id": subject_id,
        "teacher_identity_sha256": str(teacher_manifest.get("teacher_identity_sha256") or ""),
        "policy": {
            "selection": "DETERMINISTIC_EVEN_COVERAGE_OF_TEACHER_ROWS_PER_TARGET_VIEW",
            "maximum_samples_per_target_view": int(maximum_samples_per_target_view),
            "minimum_total_ray_witnesses": int(minimum_total_ray_witnesses),
            "tie_epsilon": tie_epsilon,
            "rgb_material_owner_semantic_or_attachment_evidence_used": False,
            "ambiguous_teacher_rows_are_abstentions": True,
        },
        "class_checked": class_checked,
        "class_ray_validated": class_ray_validated,
        "target_view_courts": target_rows,
        "total_ray_validated": int(total_ray_validated),
        "depth_tie_skipped": int(tie_skipped),
        "maximum_depth_gap_abs_error": float(max_gap_error),
        "maximum_surface_l2_error_canonical": float(max_surface_error),
        "maximum_facing_abs_error": float(max_facing_error),
        "training_performed": False,
        "qualification_minted": False,
    }
    payload["court_identity_sha256"] = _digest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--maximum-samples-per-target-view", type=int, default=64)
    parser.add_argument("--minimum-total-ray-witnesses", type=int, default=16)
    args = parser.parse_args()
    manifest = json.loads(args.teacher_manifest.resolve().read_text())
    result = validate(
        manifest,
        maximum_samples_per_target_view=args.maximum_samples_per_target_view,
        minimum_total_ray_witnesses=args.minimum_total_ray_witnesses,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.with_suffix(output.suffix + ".partial")
    staged.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(output)
    print(json.dumps({
        "status": result["status"],
        "subject_id": result["subject_id"],
        "total_ray_validated": result["total_ray_validated"],
        "court_identity_sha256": result["court_identity_sha256"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
