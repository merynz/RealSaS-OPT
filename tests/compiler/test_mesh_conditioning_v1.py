from __future__ import annotations

import math

import numpy as np
import pytest

from compiler.realsas_compiler_core.mesh.conditioning_v1 import (
    triangle_deformation_metric,
    triangle_rest_metric,
)


def test_equilateral_rest_metric_is_well_conditioned():
    height = math.sqrt(3.0) / 2.0
    tri = np.array([[0,0,0],[1,0,0],[0.5,height,0]], dtype=np.float64)
    metric = triangle_rest_metric(tri)
    assert metric["degenerate"] is False
    assert metric["min_angle_deg"] == pytest.approx(60.0)
    assert metric["aspect_longest_over_min_altitude"] == pytest.approx(2.0 / math.sqrt(3.0))


def test_deformation_metric_is_rigid_rotation_invariant():
    tri = np.array([[0,0,0],[1,0,0],[0.2,0.8,0]], dtype=np.float64)
    angle = math.radians(37.0)
    rotation = np.array([
        [math.cos(angle),0,math.sin(angle)],
        [0,1,0],
        [-math.sin(angle),0,math.cos(angle)],
    ])
    posed = (rotation @ tri.T).T + np.array([2.0,-3.0,5.0])
    metric = triangle_deformation_metric(tri, posed)
    assert metric["sigma_max"] == pytest.approx(1.0)
    assert metric["sigma_min"] == pytest.approx(1.0)
    assert metric["area_ratio"] == pytest.approx(1.0)
    assert metric["condition_number"] == pytest.approx(1.0)


def test_deformation_degeneracy_predicate_is_local_scale_invariant():
    base = np.array([[0,0,0],[1,0,0],[0.5,0.02,0]], dtype=np.float64)
    for scale in (1e-3, 1.0, 1e3):
        tri = base * scale
        metric = triangle_deformation_metric(
            tri.astype(np.float32),
            tri.astype(np.float32),
            dtype=np.float32,
        )
        assert metric["sigma_max"] == pytest.approx(1.0, rel=1e-4)
        assert metric["sigma_min"] == pytest.approx(1.0, rel=1e-4)
        assert metric["condition_number"] == pytest.approx(1.0, rel=1e-4)


def test_anisotropic_material_deformation_is_not_hidden_by_area():
    tri = np.array([[0,0,0],[1,0,0],[0.2,0.8,0]], dtype=np.float64)
    transform = np.array([[2.0,0.0],[0.0,0.5],[0.0,0.0]])
    posed = (transform @ tri[:,:2].T).T
    metric = triangle_deformation_metric(tri, posed)
    assert metric["area_ratio"] == pytest.approx(1.0)
    assert metric["sigma_max"] == pytest.approx(2.0)
    assert metric["sigma_min"] == pytest.approx(0.5)
    assert metric["condition_number"] == pytest.approx(4.0)
    assert metric["symmetric_stretch"] == pytest.approx(2.0)
