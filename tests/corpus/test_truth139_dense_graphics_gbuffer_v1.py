from __future__ import annotations

import numpy as np
import pytest

from tools.corpus.build_truth139_dense_graphics_gbuffer_v1 import (
    provenance_direct_source_authority,
    reconstruct_surface_points,
    transform_normals,
    transform_points,
    view_facing,
)


def test_source_to_canonical_point_transform_is_affine():
    matrix = np.asarray(
        [
            [0.0, -2.0, 0.0, 3.0],
            [2.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 2.0, 5.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    points = np.asarray([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    actual = transform_points(points, matrix)
    assert np.allclose(actual[0], [-1.0, 1.0, 11.0])
    assert np.allclose(actual[1], [3.0, -1.0, 5.0])


def test_normal_transform_uses_inverse_transpose_and_normalizes():
    # Deliberately nonuniform here to test the mathematical helper even though the imported-scene
    # contract currently permits only positive uniform normalization.
    matrix = np.asarray(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    normal = np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32)
    actual = transform_normals(normal, matrix)[0]
    expected = np.asarray([0.5, 1.0, 2.0], dtype=np.float64)
    expected /= np.linalg.norm(expected)
    assert np.allclose(actual, expected, atol=1e-7)
    assert np.isclose(np.linalg.norm(actual), 1.0, atol=1e-7)


def test_surface_reconstruction_uses_full_triangle_barycentrics():
    vertices = np.asarray([[0, 0, 0], [2, 0, 0], [0, 4, 0]], dtype=np.float32)
    faces = np.asarray([[0, 1, 2]], dtype=np.int32)
    tri = np.asarray([0], dtype=np.int32)
    bc = np.asarray([[0.25, 0.25, 0.50]], dtype=np.float32)
    point = reconstruct_surface_points(vertices, faces, tri, bc)
    assert np.allclose(point[0], [0.5, 2.0, 0.0])


def test_constructed_and_fill_views_can_never_mint_direct_source_authority():
    assert provenance_direct_source_authority("SOURCE_OBSERVATION") is True
    assert provenance_direct_source_authority("CONSTRUCTED_OBSERVATION") is False
    assert provenance_direct_source_authority("LEARNED_FILL") is False
    with pytest.raises(ValueError, match="PROVENANCE"):
        provenance_direct_source_authority("SYNTHETIC_BUT_PRETEND_SOURCE")


def test_view_facing_preserves_sign_but_grazing_reduces_confidence():
    normals = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    tri = np.asarray([0, 1, 2], dtype=np.int32)
    camera = {"forward": [0.0, 0.0, -1.0]}
    signed, confidence = view_facing(normals, tri, camera)
    assert np.allclose(signed, [1.0, -1.0, 0.0])
    assert np.allclose(confidence, [1.0, 1.0, 0.0])
