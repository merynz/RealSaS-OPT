from __future__ import annotations

import numpy as np

from models.iris.vnext.topology_teacher_v1 import (
    canonicalize_vnext_topology_teacher_v1,
)


def _same_component_seam_mesh():
    # Faces 1 and 2 are geometrically identical after v1/v3 weld, but source
    # indices differ as a UV/material-seam style split. All faces are edge-connected.
    rest = np.asarray(
        [
            [-0.6, -0.6, 0.0],  # 0
            [0.3, -0.5, 0.0],   # 1
            [-0.4, 0.4, 0.0],   # 2
            [0.3, -0.5, 0.0],   # 3 duplicate of 1
            [0.4, 0.4, 0.0],    # 4
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [2, 1, 4], [2, 3, 4]], dtype=np.int64)
    return rest, faces


def test_motion_equivalent_exact_seam_duplicate_is_training_only_welded():
    rest, faces = _same_component_seam_mesh()
    posed = np.stack(
        (
            rest,
            rest + np.asarray([0.1, 0.0, 0.0]),
            rest + np.asarray([0.0, 0.2, 0.0]),
        ),
        axis=0,
    )
    result = canonicalize_vnext_topology_teacher_v1(
        rest, posed, faces, coordinate_bins=1024
    )
    meta = result.metadata
    assert meta["training_only"] is True
    assert meta["product_geometry_authority"] is False
    assert meta["motion_equivalent_welded_vertex_count"] == 1
    assert meta["motion_distinct_coincident_pair_count"] == 0
    assert meta["duplicate_face_count_removed"] == 1
    assert meta["quantized_collision_census"]["collision_group_count"] == 0
    assert result.vertices_iris.shape[0] == 4
    assert result.faces.shape[0] == 2


def test_coincident_vertices_that_separate_in_authored_motion_are_never_welded():
    rest, faces = _same_component_seam_mesh()
    posed = np.stack((rest, rest.copy()), axis=0)
    posed[1, 3, 1] += 0.05
    result = canonicalize_vnext_topology_teacher_v1(
        rest, posed, faces, coordinate_bins=1024
    )
    meta = result.metadata
    assert meta["motion_equivalent_welded_vertex_count"] == 0
    assert meta["motion_distinct_coincident_pair_count"] >= 1
    assert meta["quantized_collision_census"]["collision_group_count"] >= 1
    assert result.vertices_iris.shape[0] == 5


def test_distinct_nearby_vertices_in_same_tessa_bin_remain_a_fail_closed_census():
    rest = np.asarray(
        [
            [-0.8, -0.7, 0.0],
            [0.10000, -0.4, 0.0],
            [-0.2, 0.2, 0.0],
            [0.10020, -0.4, 0.0],
            [0.4, 0.3, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [2, 1, 4], [2, 3, 4]], dtype=np.int64)
    posed = np.stack((rest, rest + np.asarray([0.0, 0.0, 0.1])), axis=0)
    result = canonicalize_vnext_topology_teacher_v1(
        rest, posed, faces, coordinate_bins=1024
    )
    meta = result.metadata
    assert meta["exact_rest_duplicate_vertex_count"] == 0
    assert meta["motion_equivalent_welded_vertex_count"] == 0
    assert meta["quantized_collision_census"]["collision_group_count"] >= 1


def test_same_coordinate_in_edge_disconnected_components_does_not_create_connection():
    rest = np.asarray(
        [
            [-0.5, -0.5, 0.0],
            [0.0, -0.5, 0.0],
            [-0.5, 0.0, 0.0],
            [0.0, -0.5, 0.0],  # exact coordinate repeated in another component
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    posed = np.stack((rest, rest + np.asarray([0.0, 0.0, 0.05])), axis=0)
    result = canonicalize_vnext_topology_teacher_v1(
        rest, posed, faces, coordinate_bins=1024
    )
    meta = result.metadata
    assert meta["component_count"] == 2
    assert meta["motion_equivalent_welded_vertex_count"] == 0
    assert meta["quantized_collision_census"]["collision_group_count"] == 0
    assert result.vertices_iris.shape[0] == 6
