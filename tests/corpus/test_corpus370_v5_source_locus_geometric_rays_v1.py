from __future__ import annotations

import copy

import numpy as np
import pytest

from tools.corpus.measure_corpus370_v5_source_locus_rays_v1 import sample_rays


def fixture():
    # Two disjoint, overlapping source faces; separate face IDs remain teacher-only.
    vertices = np.array([
        [-1, -1, 0], [1, -1, 0], [0, 1, 0],
        [-1, -1, -1], [1, -1, -1], [0, 1, -1],
    ], np.float64)
    faces = np.array([[0, 1, 2], [3, 4, 5]], np.int64)
    cameras = [{
        "origin": [0., 0., 3.],
        "forward": [0., 0., -1.],
        "right": [1., 0., 0.],
        "screen_up": [0., 1., 0.],
        "half_extent": 2.,
        "resolution": 64,
        "view_index": i,
    } for i in range(8)]
    return vertices, faces, {"schema": "RealSaS.CameraProjectionBundle.v1", "cameras": cameras}


def test_geometric_multihits_have_source_addresses_not_drawables():
    v, f, cam = fixture()
    rays, counts = sample_rays(v, f, cam, max_pixels_per_view=8)
    assert counts["qualified_geometric_rays"] > 0
    assert counts["multihit_geometric_rays"] > 0
    assert counts["qualified_geometric_ordered_hit_pairs"] > 0
    assert {r["view_index"] for r in rays} == set(range(8))
    multi = next(x for x in rays if len(x["hits"]) >= 2 and x["status"].startswith("VALID"))
    assert multi["complete"] is True
    assert multi["source_alpha_compositing_qualified"] is False
    assert multi["drawable_ownership_label_minted"] is False
    assert multi["hits"][0]["depth_source_units"] < multi["hits"][1]["depth_source_units"]
    for hit in multi["hits"]:
        assert sum(hit["source_barycentric_teacher_only"]) == pytest.approx(1.0)
        assert hit["face_index_teacher_only"] in (0, 1)


def test_rejects_camera_basis_mutation():
    v, f, cam = fixture()
    cam["cameras"][0]["right"] = [1., .2, 0.]
    with pytest.raises(ValueError, match="GRAPHICS_ORACLE_CAMERA_ORTHONORMAL_REQUIRED"):
        sample_rays(v, f, cam, max_pixels_per_view=8)


def test_rejects_unreviewed_ray_scan_budget():
    v, f, cam = fixture()
    with pytest.raises(ValueError, match="SOURCE_RAY_BOUNDED_SAMPLING"):
        sample_rays(v, f, cam, max_pixels_per_view=1000000)


def test_rejects_unbounded_view_count():
    v, f, cam = fixture()
    cam["cameras"].pop()
    with pytest.raises(ValueError, match="SOURCE_RAY_EIGHT_FIXED_CAMERAS"):
        sample_rays(v, f, cam, max_pixels_per_view=8)
