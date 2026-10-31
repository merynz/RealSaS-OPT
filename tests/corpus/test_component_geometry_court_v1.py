from __future__ import annotations

import numpy as np
import pytest

from tools.corpus.validate_component_geometry_authority_v1 import (
    compare_partition,
    direct_distance,
    independent_component_validity,
    independent_sign_components,
    independent_surface_components,
    partition_sets,
    solid_angle_inside,
)


def tetra():
    vertices = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64
    )
    faces = np.asarray(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32
    )
    return vertices, faces


def test_partition_comparison_is_label_invariant():
    expected = np.asarray([0, 0, 1, 1, -1], dtype=np.int32)
    actual = np.asarray([7, 7, 3, 3, -1], dtype=np.int32)
    compare_partition(expected, actual, "TEST")
    assert partition_sets(expected) == partition_sets(actual)


def test_partition_comparison_rejects_different_connectivity():
    expected = np.asarray([0, 0, 1, 1], dtype=np.int32)
    actual = np.asarray([0, 1, 1, 1], dtype=np.int32)
    with pytest.raises(ValueError, match="TEST_PARTITION"):
        compare_partition(expected, actual, "TEST")


def test_independent_closed_tetra_sign_component_is_valid():
    _, faces = tetra()
    face_valid = np.ones(len(faces), dtype=np.bool_)
    surface = independent_surface_components(faces)
    sign = independent_sign_components(faces, face_valid)
    validity = independent_component_validity(faces, sign, face_valid)
    assert len(partition_sets(surface)) == 1
    assert len(partition_sets(sign)) == 1
    assert validity == {0: True}


def test_independent_open_triangle_is_not_sign_valid():
    faces = np.asarray([[0, 1, 2]], dtype=np.int32)
    face_valid = np.asarray([True])
    sign = independent_sign_components(faces, face_valid)
    assert independent_component_validity(faces, sign, face_valid) == {0: False}


def test_solid_angle_is_independent_inside_outside_court():
    vertices, faces = tetra()
    points = np.asarray([[0.1, 0.1, 0.1], [2.0, 2.0, 2.0]], dtype=np.float64)
    inside = solid_angle_inside(points, vertices, faces)
    assert inside.tolist() == [True, False]


def test_direct_distance_to_triangle_surface():
    vertices, faces = tetra()
    points = np.asarray([[0.0, 0.0, 0.0], [0.25, 0.25, -0.5]], dtype=np.float64)
    distance = direct_distance(points, vertices, faces, np.arange(len(faces)))
    assert np.isclose(distance[0], 0.0, atol=1e-10)
    assert np.isclose(distance[1], 0.5, atol=1e-10)
