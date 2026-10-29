from __future__ import annotations

import json

import numpy as np

from models.iris.truth.graphics_oracle_v1 import camera_from_mapping, measure_orthographic_ray_stack_v1
from tools.corpus.build_truth139_dense_graphics_gbuffer_v1 import sha256
from tools.corpus.validate_truth139_dense_graphics_geometry_v1 import PASS_STATUS, validate


def test_dense_graphics_geometry_court_reuses_exact_cpu_measurement_without_legacy_authority(tmp_path):
    subject_id = "synthetic_dense_graphics"
    geometry = tmp_path / "geometry.npz"
    vertices = np.asarray(
        [[-2.0, -2.0, 1.0], [2.0, -2.0, 1.0], [-2.0, 2.0, 1.0]],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int32)
    np.savez_compressed(geometry, rest_vertices_source_world=vertices, faces=faces)

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
    camera_path = tmp_path / "cameras.json"
    camera_path.write_text(json.dumps({"schema": "RealSaS.CameraProjectionBundle.v1", "cameras": cameras}))

    views = []
    pixel = 2 * 4 + 1  # ray centre (-0.5,-0.5), safely interior to the triangle
    for view, payload in enumerate(cameras):
        stack = measure_orthographic_ray_stack_v1(
            vertices,
            faces,
            camera_from_mapping(payload),
            1,
            2,
            tie_epsilon=1.0e-7,
        )
        assert stack.complete and not stack.has_depth_tie and len(stack.hits) == 1
        hit = stack.hits[0]
        gbuffer = tmp_path / f"V{view}_DENSE_GRAPHICS_GBUFFER.npz"
        np.savez_compressed(
            gbuffer,
            pixel_linear_index=np.asarray([pixel], dtype=np.int32),
            triangle_id=np.asarray([hit.face_index], dtype=np.int32),
            barycentric_uv=np.asarray([hit.barycentric[:2]], dtype=np.float32),
            depth_source_camera=np.asarray([hit.depth], dtype=np.float32),
        )
        views.append(
            {
                "view_index": view,
                "gbuffer_path": str(gbuffer),
                "gbuffer_sha256": sha256(gbuffer),
            }
        )

    manifest = {
        "schema": "RealSaS.Truth139DenseGraphicsGBuffer.v1",
        "status": "PASS_DENSE_GRAPHICS_FRONT_SURFACE__VISIBILITY_OCCLUSION_COURT_PENDING",
        "subject_id": subject_id,
        "graphics_identity_sha256": "a" * 64,
        "geometry_npz": {"path": str(geometry), "sha256": sha256(geometry)},
        "camera_bundle": {"path": str(camera_path), "sha256": sha256(camera_path)},
        "views": views,
    }
    result = validate(manifest, maximum_samples_per_view=1, minimum_validated_per_view=1)
    assert result["status"] == PASS_STATUS
    assert result["schema"] == "RealSaS.Truth139DenseGraphicsGeometryCourt.v1"
    assert result["dense_graphics_identity_sha256"] == "a" * 64
    assert result["total_validated_pixel_count"] == 8
    assert result["compatibility_projection"]["kind"] == "VOCABULARY_ONLY__MEASUREMENT_UNCHANGED"
    assert result["compatibility_projection"]["material_rgb_semantic_or_owner_evidence_used"] is False
