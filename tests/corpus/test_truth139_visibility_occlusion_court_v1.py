from __future__ import annotations

import json

import numpy as np

from tools.corpus.build_truth139_dense_graphics_gbuffer_v1 import sha256
from tools.corpus.build_truth139_visibility_occlusion_teacher_v1 import VISIBILITY
from tools.corpus.validate_truth139_visibility_occlusion_teacher_v1 import validate


def test_cpu_court_accepts_exact_visible_and_occluded_relations(tmp_path):
    subject_id = "synthetic_two_layer"
    geometry = tmp_path / "geometry.npz"
    vertices = np.asarray(
        [
            [-2.0, -2.0, 1.0],
            [2.0, -2.0, 1.0],
            [-2.0, 2.0, 1.0],
            [-2.0, -2.0, 0.0],
            [2.0, -2.0, 0.0],
            [-2.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    np.savez_compressed(geometry, rest_vertices_source_world=vertices, faces=faces)

    camera_path = tmp_path / "cameras.json"
    cameras = []
    for view in range(8):
        cameras.append(
            {
                "view_index": view,
                "origin": [0.0, 0.0, 10.0],
                "forward": [0.0, 0.0, -1.0],
                "right": [1.0, 0.0, 0.0],
                "screen_up": [0.0, 1.0, 0.0],
                "half_extent": 2.0,
                "resolution": 4,
            }
        )
    camera_path.write_text(json.dumps({"schema": "RealSaS.CameraProjectionBundle.v1", "cameras": cameras}))

    graphics_path = tmp_path / "graphics_manifest.json"
    graphics_manifest = {
        "schema": "RealSaS.Truth139DenseGraphicsGBuffer.v1",
        "status": "PASS_DENSE_GRAPHICS_FRONT_SURFACE__VISIBILITY_OCCLUSION_COURT_PENDING",
        "subject_id": subject_id,
        "geometry_npz": {"path": str(geometry), "sha256": sha256(geometry)},
        "coordinate_contract": {
            "source_world_to_canonical": np.eye(4, dtype=float).tolist(),
            "point_transform": "AFFINE_HOMOGENEOUS",
            "normal_transform": "INVERSE_TRANSPOSE_THEN_NORMALIZE",
        },
    }
    graphics_path.write_text(json.dumps(graphics_manifest, sort_keys=True))

    teacher_path = tmp_path / "teacher.npz"
    # Pixel x=1,y=2 has ray centre (-0.5,-0.5), safely inside both triangles.
    pixel = 2 * 4 + 1
    source_xyz = np.asarray([[-0.5, -0.5, 1.0], [-0.5, -0.5, 0.0]], dtype=np.float32)
    source_canonical = source_xyz.copy()
    classes = np.stack(
        [
            np.full(8, VISIBILITY["VISIBLE_SAME_SURFACE"], dtype=np.uint8),
            np.full(8, VISIBILITY["OCCLUDED_BY_FRONT_SURFACE"], dtype=np.uint8),
        ],
        axis=0,
    )
    projected = np.full((2, 8), pixel, dtype=np.int32)
    front = np.zeros((2, 8), dtype=np.int32)
    gap = np.stack([np.zeros(8, dtype=np.float32), np.ones(8, dtype=np.float32)], axis=0)
    facing = np.ones((2, 8), dtype=np.float32)
    source_support = np.ones((2, 8), dtype=np.bool_)
    same_surface = np.stack([np.ones(8, dtype=np.bool_), np.zeros(8, dtype=np.bool_)], axis=0)
    np.savez_compressed(
        teacher_path,
        surface_xyz_source_world=source_xyz,
        surface_xyz_canonical=source_canonical,
        target_visibility_class=classes,
        target_projected_pixel_linear_index=projected,
        target_front_triangle_id=front,
        target_depth_gap=gap,
        target_view_facing_confidence=facing,
        target_source_support=source_support,
        cross_view_same_surface_match=same_surface,
    )

    manifest = {
        "schema": "RealSaS.Truth139VisibilityOcclusionTeacher.v1",
        "status": "PASS_VISIBILITY_OCCLUSION_TEACHER__INDEPENDENT_CPU_ORACLE_COURT_PENDING",
        "subject_id": subject_id,
        "graphics_manifest": {"path": str(graphics_path), "sha256": sha256(graphics_path)},
        "camera_bundle": {"path": str(camera_path), "sha256": sha256(camera_path)},
        "teacher_npz": {"path": str(teacher_path), "sha256": sha256(teacher_path)},
        "tolerances": {
            "search_radius_px": 1,
            "surface_tolerance_pixels": 0.5,
            "depth_epsilon_pixels": 0.25,
            "canonical_uniform_scale": 1.0,
        },
        "teacher_identity_sha256": "0" * 64,
    }

    result = validate(
        manifest,
        maximum_samples_per_target_view=2,
        minimum_total_ray_witnesses=16,
    )
    assert result["status"] == "PASS_INDEPENDENT_CPU_RAY_ORACLE_AGREES_WITH_VISIBILITY_OCCLUSION_TEACHER"
    assert result["total_ray_validated"] == 16
    assert result["class_ray_validated"]["VISIBLE_SAME_SURFACE"] == 8
    assert result["class_ray_validated"]["OCCLUDED_BY_FRONT_SURFACE"] == 8
