from __future__ import annotations

import numpy as np
import torch

from tools.iris_vnext.h_geo_p3_efreq_v1 import (
    NormalizedCameraV1,
    angular_change_degrees_v1,
    face_vertex_adjacency_v1,
    nonadjacent_surface_stats_v1,
    project_points_v1,
    radial_highband_fraction_v1,
    select_tail_and_matched_control_v1,
    whitened_rms_delta_v1,
)


def test_projection_and_angle_contract():
    cam = NormalizedCameraV1(
        origin_iris=np.array([0.0, 0.0, 2.0]),
        right=np.array([1.0, 0.0, 0.0]),
        up=np.array([0.0, 1.0, 0.0]),
        forward=np.array([0.0, 0.0, -1.0]),
        half_extent_iris=1.0,
    )
    xy = project_points_v1(np.array([[0.25, -0.5, 0.0]]), cam)
    assert np.allclose(xy, [[0.25, -0.5]])
    assert abs(angular_change_degrees_v1(np.array([1.0,0.0,0.0]), np.array([0.0,1.0,0.0])) - 90.0) < 1e-6


def test_nonadjacent_surface_gap_excludes_one_ring():
    vertices = np.array([
        [0,0,0],[1,0,0],[0,1,0],
        [0,0,0.2],[1,0,0.2],[0,1,0.2],
    ], dtype=np.float64)
    faces = np.array([[0,1,2],[3,4,5]], dtype=np.int64)
    triangles = vertices[faces]
    adjacency = face_vertex_adjacency_v1(faces)
    out = nonadjacent_surface_stats_v1(np.array([0.2,0.2,0.0]), 0, triangles, adjacency)
    assert abs(out["nonadjacent_distance_iris"] - 0.2) < 1e-8
    assert int(out["nonadjacent_face_index"]) == 1


def test_tail_selection_is_deterministic_and_distance_valid():
    n = 1000
    distance = np.linspace(0.0, 0.119, n)
    error = np.linspace(0.0, 1.0, n)
    t1,c1 = select_tail_and_matched_control_v1(error,distance,maximum_per_group=64)
    t2,c2 = select_tail_and_matched_control_v1(error,distance,maximum_per_group=64)
    assert np.array_equal(t1,t2)
    assert np.array_equal(c1,c2)
    assert len(t1) == 64
    assert len(c1) > 0
    assert np.all(distance[t1] <= 0.12)


def test_whitened_delta_and_fft_metrics_are_finite():
    a=torch.zeros(4,8); b=torch.ones(4,8); scale=torch.ones(8)
    d=whitened_rms_delta_v1(a,b,scale)
    assert torch.allclose(d,torch.ones(4))
    planes=torch.zeros(1,3,2,16,16)
    planes[...,::2,::2]=1.0
    value=radial_highband_fraction_v1(planes)
    assert 0.0 <= value <= 1.0
