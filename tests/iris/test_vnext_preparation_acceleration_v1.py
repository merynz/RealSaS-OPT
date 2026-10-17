from __future__ import annotations

import math

import numpy as np
import pytest

from models.iris.vnext import teacher_truth_v1 as truth
from tools.iris_vnext import build_fit8_vnext_shards_v1 as shard_v1
from tools.iris_vnext.build_fit8_vnext_shards_v2 import (
    _RAY_BACKEND,
    _ray_triangle_first_two_parallel_grid,
)


def _layered_mesh(seed: int = 19):
    rng = np.random.default_rng(seed)
    triangles = []
    for depth in (1.5, 2.5, 3.5):
        for _ in range(120):
            center = rng.uniform(-0.95, 0.95, size=2)
            radius = rng.uniform(0.015, 0.07)
            angle = rng.uniform(0.0, 2.0 * math.pi)
            offsets = np.asarray(
                [
                    [math.cos(angle), math.sin(angle)],
                    [math.cos(angle + 2.1), math.sin(angle + 2.1)],
                    [math.cos(angle + 4.2), math.sin(angle + 4.2)],
                ],
                dtype=np.float64,
            ) * radius
            z = depth + rng.normal(scale=0.03, size=(3, 1))
            triangles.append(np.concatenate((center[None, :] + offsets, z), axis=1))
    vertices = np.concatenate(triangles, axis=0)
    faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    return vertices, faces


def test_projected_grid_raycast_matches_legacy_first_and_second_hits():
    vertices, faces = _layered_mesh()
    axis = np.linspace(-1.0, 1.0, 41, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    origins = np.stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)), axis=1)
    directions = np.broadcast_to(
        np.asarray([0.0, 0.0, 1.0], dtype=np.float64), origins.shape
    ).copy()

    legacy = shard_v1._ray_triangle_first_two(
        origins, directions, vertices, faces, chunk=64
    )
    accelerated = _ray_triangle_first_two_parallel_grid(
        origins, directions, vertices, faces, chunk=64
    )

    assert _RAY_BACKEND == "PARALLEL_ORTHOGRAPHIC_PROJECTED_GRID_EXACT_EQUIVALENT_V1"
    assert np.array_equal(accelerated["first_face"], legacy["first_face"])
    assert np.array_equal(accelerated["second_face"], legacy["second_face"])
    np.testing.assert_allclose(
        accelerated["first_depth"], legacy["first_depth"], rtol=0.0, atol=1.0e-12
    )
    np.testing.assert_allclose(
        accelerated["second_depth"], legacy["second_depth"], rtol=0.0, atol=1.0e-12
    )
    np.testing.assert_allclose(
        accelerated["first_point"], legacy["first_point"], rtol=0.0, atol=1.0e-12
    )
    np.testing.assert_allclose(
        accelerated["second_point"], legacy["second_point"], rtol=0.0, atol=1.0e-12
    )


def test_projected_grid_raycast_fails_closed_for_nonparallel_rays():
    vertices, faces = _layered_mesh(seed=23)
    origins = np.zeros((2, 3), dtype=np.float64)
    directions = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.01, 1.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="VNEXT_RAY_ACCEL_REQUIRES_PARALLEL_RAYS"):
        _ray_triangle_first_two_parallel_grid(origins, directions, vertices, faces)


def _legacy_bend_reference(rest_normals, posed_normals, faces, degenerate_mask):
    degenerate = np.asarray(degenerate_mask, dtype=bool)
    out = np.zeros(len(faces), dtype=np.float64)
    out[degenerate] = math.pi
    for a, b in truth._adjacency(np.asarray(faces, dtype=np.int64)):
        if degenerate[a] or degenerate[b]:
            out[a] = max(out[a], math.pi)
            out[b] = max(out[b], math.pi)
            continue
        rest_dot = float(np.clip(np.dot(rest_normals[a], rest_normals[b]), -1, 1))
        posed_dot = float(np.clip(np.dot(posed_normals[a], posed_normals[b]), -1, 1))
        delta = abs(math.acos(posed_dot) - math.acos(rest_dot))
        out[a] = max(out[a], delta)
        out[b] = max(out[b], delta)
    return out


def test_vectorized_bend_matches_legacy_scalar_reduction():
    faces = np.asarray(
        [[0, 1, 2], [2, 1, 3], [2, 3, 4], [4, 3, 5], [4, 5, 6]],
        dtype=np.int64,
    )
    rng = np.random.default_rng(7)
    rest_normals = rng.normal(size=(len(faces), 3))
    rest_normals /= np.linalg.norm(rest_normals, axis=1, keepdims=True)
    posed_normals = rest_normals + rng.normal(scale=0.08, size=rest_normals.shape)
    posed_normals /= np.linalg.norm(posed_normals, axis=1, keepdims=True)
    degenerate = np.asarray([False, False, True, False, False], dtype=bool)

    edge_a, edge_b, rest_angles = truth._bend_precompute(rest_normals, faces)
    accelerated = truth._bend_precomputed(
        posed_normals,
        degenerate,
        len(faces),
        edge_a,
        edge_b,
        rest_angles,
    )
    legacy = _legacy_bend_reference(rest_normals, posed_normals, faces, degenerate)
    np.testing.assert_allclose(accelerated, legacy, rtol=0.0, atol=1.0e-14)
