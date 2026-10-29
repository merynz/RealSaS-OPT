from __future__ import annotations

import numpy as np

from tools.corpus.build_truth139_dense_appearance_gbuffer_v1 import (
    geometric_face_normals,
    srgb_to_linear,
    world_to_pytorch3d_ndc,
)


def test_srgb_inverse_transfer_endpoints_and_midpoint():
    x = np.asarray([[0.0, 0.04045, 1.0], [0.5, 0.25, 0.75]], dtype=np.float32)
    y = srgb_to_linear(x)
    assert np.allclose(y[0], [0.0, 0.04045 / 12.92, 1.0], atol=1e-7)
    assert np.isclose(y[1, 0], 0.21404114, atol=1e-6)
    assert np.all(y >= 0.0) and np.all(y <= 1.0)


def test_controlled_camera_world_to_pytorch3d_ndc_screen_orientation():
    camera = {
        "origin": [0.0, 0.0, 10.0],
        "right": [1.0, 0.0, 0.0],
        "screen_up": [0.0, 1.0, 0.0],
        "forward": [0.0, 0.0, -1.0],
        "half_extent": 2.0,
    }
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, -2.0, 0.0],
        ],
        dtype=np.float32,
    )
    ndc = world_to_pytorch3d_ndc(vertices, camera)
    assert np.allclose(ndc[0], [0.0, 0.0, 10.0])
    # Controlled +right is image-right; PyTorch3D image-right is NDC -X.
    assert np.allclose(ndc[1], [-1.0, 0.0, 10.0])
    assert np.allclose(ndc[2], [1.0, 0.0, 10.0])
    assert np.allclose(ndc[3], [0.0, 1.0, 10.0])
    assert np.allclose(ndc[4], [0.0, -1.0, 10.0])


def test_geometric_face_normals_are_finite_and_oriented():
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.asarray([[0, 1, 2]], dtype=np.int32)
    normal = geometric_face_normals(vertices, faces)
    assert normal.shape == (1, 3)
    assert np.allclose(normal[0], [0.0, 0.0, 1.0])
