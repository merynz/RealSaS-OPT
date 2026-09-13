from __future__ import annotations

import numpy as np

from compiler.realsas_compiler_core.mesh.quality import FIT2_PRODUCT_MESH_QUALITY_POLICY_V1
from experiments.mage_full_subject_reclosure_v1.run_fit2_adaptive_boundary_quality_cdt_v1 import (
    ParentTriangle,
    _barycentric_weights,
    _boundary_mask,
    _component_triangle_local,
    _hop_neighborhood,
    _parent_adjacency,
    _quality_admissible,
    _sample_component_pixels,
)


def test_barycentric_support_is_nonnegative_simplex_and_reconstructs_point():
    tri = ((0.0, 0.0), (10.0, 0.0), (0.0, 10.0))
    weights = _barycentric_weights((2.0, 3.0), tri)
    assert weights is not None
    assert all(w >= 0.0 for w in weights)
    assert abs(sum(weights) - 1.0) <= 1.0e-12
    x = sum(w * p[0] for w, p in zip(weights, tri))
    y = sum(w * p[1] for w, p in zip(weights, tri))
    assert abs(x - 2.0) <= 1.0e-12
    assert abs(y - 3.0) <= 1.0e-12


def test_barycentric_rejects_point_outside_parent_triangle():
    tri = ((0.0, 0.0), (10.0, 0.0), (0.0, 10.0))
    assert _barycentric_weights((9.0, 9.0), tri) is None


def test_quality_prefilter_uses_frozen_product_thresholds():
    policy = FIT2_PRODUCT_MESH_QUALITY_POLICY_V1
    assert policy.min_raster_triangle_angle_deg == 0.25
    assert policy.max_raster_triangle_aspect_ratio == 250.0
    assert _quality_admissible(((0.0, 0.0), (10.0, 0.0), (0.0, 10.0)))
    assert not _quality_admissible(((0.0, 0.0), (1000.0, 0.0), (0.01, 0.001)))


def test_boundary_sampling_is_deterministic_and_dense_at_stride_one():
    target = np.zeros((12, 12), dtype=bool)
    target[2:10, 3:9] = True
    boundary = _boundary_mask(target)
    rows1, diag1 = _sample_component_pixels(
        target,
        boundary_stride=1,
        interior_spacing=4,
    )
    rows2, diag2 = _sample_component_pixels(
        target,
        boundary_stride=1,
        interior_spacing=4,
    )
    assert rows1 == rows2
    assert diag1 == diag2
    sampled = {(x, y) for x, y, role in rows1 if role == "ALPHA_SUPPORT_BOUNDARY"}
    expected = {(int(x), int(y)) for y, x in zip(*np.nonzero(boundary))}
    assert sampled == expected


def test_parent_locality_never_crosses_disconnected_component_graph():
    parents = (
        ParentTriangle(0, 0, ("A", "B", "C"), ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), (0.0, 0.0, 1.0, 1.0)),
        ParentTriangle(1, 0, ("B", "C", "D"), ((1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), (0.0, 0.0, 1.0, 1.0)),
        ParentTriangle(2, 0, ("X", "Y", "Z"), ((10.0, 10.0), (11.0, 10.0), (10.0, 11.0)), (10.0, 10.0, 11.0, 11.0)),
    )
    neighbors = _parent_adjacency(parents)
    hops = _hop_neighborhood(neighbors, hops=2)
    assert _component_triangle_local((0, 1, 0), hops)
    assert not _component_triangle_local((0, 1, 2), hops)
