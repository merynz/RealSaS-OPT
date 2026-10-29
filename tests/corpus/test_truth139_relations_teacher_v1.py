from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.corpus.build_truth139_dense_appearance_gbuffer_v1 import sha256
from tools.corpus.build_truth139_relations_teacher_v1 import build, exact_face_edge_adjacency
from tools.corpus.validate_truth139_relations_teacher_v1 import validate


def _npz(path: Path, **arrays) -> None:
    np.savez_compressed(path, **arrays)


def test_exact_adjacency_emits_all_face_pairs_on_nonmanifold_edge():
    # Faces 0, 1 and 2 all share edge (0, 1).  Truth must preserve all three
    # unordered face pairs rather than silently choosing two incident faces.
    faces = np.asarray(
        [
            [0, 1, 2],
            [1, 0, 3],
            [0, 1, 4],
        ],
        dtype=np.int32,
    )
    pairs, edges = exact_face_edge_adjacency(faces)
    shared = [tuple(map(int, pair)) for pair, edge in zip(pairs, edges) if tuple(map(int, edge)) == (0, 1)]
    assert shared == [(0, 1), (0, 2), (1, 2)]


def test_relations_teacher_rebinds_graphics_parent_truth(tmp_path: Path):
    geometry = tmp_path / "geometry.npz"
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    _npz(geometry, rest_vertices_source_world=vertices, faces=faces)

    dense_rows = []
    for view in range(8):
        dense_path = tmp_path / f"V{view}_dense.npz"
        tri = np.asarray([1 if view == 1 else 0], dtype=np.int32)
        bary = np.asarray([[0.25, 0.25]], dtype=np.float32)
        point = np.asarray([[0.25, 0.25, 0.0]], dtype=np.float32)
        _npz(
            dense_path,
            triangle_id=tri,
            barycentric_uv=bary,
            surface_xyz_source_world=point,
            surface_xyz_canonical=point,
        )
        dense_rows.append(
            {
                "view_index": view,
                "teacher_path": str(dense_path),
                "teacher_sha256": sha256(dense_path),
            }
        )

    visibility = tmp_path / "visibility.npz"
    classes = np.ones((1, 8), dtype=np.uint8)
    classes[0, 1] = 2  # OCCLUDED_BY_FRONT_SURFACE
    same = np.ones((1, 8), dtype=np.bool_)
    same[0, 1] = False
    match = np.zeros((1, 8), dtype=np.int32)
    match[0, 1] = -1
    match_error = np.zeros((1, 8), dtype=np.float32)
    match_error[0, 1] = np.inf
    front = np.zeros((1, 8), dtype=np.int32)
    gap = np.zeros((1, 8), dtype=np.float32)
    gap[0, 1] = 0.5
    _npz(
        visibility,
        source_view_index=np.asarray([0], dtype=np.int16),
        source_surface_row=np.asarray([0], dtype=np.int32),
        source_triangle_id=np.asarray([0], dtype=np.int32),
        source_barycentric_uv=np.asarray([[0.25, 0.25]], dtype=np.float32),
        target_visibility_class=classes,
        target_front_row=front,
        target_depth_gap=gap,
        cross_view_same_surface_match=same,
        cross_view_match_row=match,
        cross_view_match_surface_error=match_error,
    )

    graphics = tmp_path / "graphics_manifest.json"
    graphics_value = {
        "schema": "RealSaS.Truth139DenseGraphicsTeacher.v1",
        "status": "PASS_DENSE_GRAPHICS_TEACHER_CANDIDATE__INDEPENDENT_COURTS_REQUIRED",
        "subject_id": "fixture-animal",
        "graphics_teacher_identity_sha256": "a" * 64,
        "parents": {
            "geometry_npz": {"path": str(geometry), "sha256": sha256(geometry)},
        },
        "dense_views": dense_rows,
        "visibility_occlusion": {
            "path": str(visibility),
            "sha256": sha256(visibility),
            "surface_tolerance": 1.0e-3,
        },
    }
    graphics.write_text(json.dumps(graphics_value, sort_keys=True) + "\n")

    output_dir = tmp_path / "relations"
    result = build(dense_graphics_teacher_manifest=graphics, output_dir=output_dir)
    assert result["counts"]["local_surface_adjacency"] == 1
    assert result["counts"]["same_canonical_surface_across_views"] == 7
    assert result["counts"]["front_behind_on_shared_camera_ray"] == 1
    assert result["policy"]["absence_is_negative_label"] is False

    court = validate(result)
    assert court["status"] == "PASS_RELATIONS_TEACHER_REBINDS_EXACT_PARENT_TRUTH"
    assert court["validated_counts"]["same_canonical_surface_across_views"] == 7
    assert court["validated_counts"]["front_behind_on_shared_camera_ray"] == 1
