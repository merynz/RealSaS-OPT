from __future__ import annotations

import numpy as np

from compiler.realsas_compiler_core.mesh.quality import FIT2_PRODUCT_MESH_QUALITY_POLICY_V1
from experiments.mage_full_subject_reclosure_v1.run_fit2_baseline_preserving_adaptive_patch_cdt_v1 import (
    TREATMENTS,
    _boundary_cycle,
    _connected_key_clusters,
    _expand_face_set,
    _face_adjacency,
    _quality_ok_xy,
    _rasterize_faces,
    _topology_counts,
)


def test_treatment_ladder_is_frozen():
    assert TREATMENTS == (
        {"name": "P1_B2_G10", "repair_hops": 1, "boundary_stride": 2, "interior_spacing": 10},
        {"name": "P2_B1_G8", "repair_hops": 2, "boundary_stride": 1, "interior_spacing": 8},
        {"name": "P2_B1_G6", "repair_hops": 2, "boundary_stride": 1, "interior_spacing": 6},
    )


def test_frozen_quality_thresholds_are_not_relaxed():
    p = FIT2_PRODUCT_MESH_QUALITY_POLICY_V1
    assert p.min_source_alpha_recall == 0.94
    assert p.min_precision_inside_alpha == 0.995
    assert p.min_alpha_iou == 0.935
    assert p.max_largest_uncovered_component_fraction == 0.015
    assert p.min_raster_triangle_angle_deg == 0.25
    assert p.max_raster_triangle_aspect_ratio == 250.0


def test_quality_predicate_accepts_normal_and_rejects_skinny_triangle():
    assert _quality_ok_xy(((0.0, 0.0), (10.0, 0.0), (0.0, 10.0)))
    assert not _quality_ok_xy(((0.0, 0.0), (1000.0, 0.0), (0.01, 0.001)))


def test_face_hop_expansion_and_connected_clusters_are_deterministic():
    faces = [
        ("A", "B", "C"),
        ("B", "D", "C"),
        ("B", "E", "D"),
        ("X", "Y", "Z"),
    ]
    adj = _face_adjacency(faces)
    seed = {tuple(sorted(("A", "B", "C")))}
    one = _expand_face_set(seed, adj, 1)
    assert tuple(sorted(("B", "C", "D"))) in one
    assert tuple(sorted(("B", "D", "E"))) not in one
    clusters = _connected_key_clusters(one | {tuple(sorted(("X", "Y", "Z")))}, adj)
    assert len(clusters) == 2


def test_boundary_cycle_for_two_triangle_disk_has_four_seam_vertices():
    faces = [("A", "B", "C"), ("A", "C", "D")]
    xy = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (1.0, 1.0),
        "D": (0.0, 1.0),
    }
    cycle = _boundary_cycle(faces, xy)
    assert cycle is not None
    assert len(cycle) == 4
    assert set(cycle) == {"A", "B", "C", "D"}


def test_topology_counts_duplicate_and_nonmanifold():
    clean = [("A", "B", "C"), ("A", "C", "D")]
    assert _topology_counts(clean) == {
        "duplicate_faces": 0,
        "degenerate_faces": 0,
        "nonmanifold_edges": 0,
    }
    duplicate = clean + [("C", "B", "A")]
    assert _topology_counts(duplicate)["duplicate_faces"] == 1
    nonmanifold = [
        ("A", "B", "C"),
        ("B", "A", "D"),
        ("A", "B", "E"),
    ]
    assert _topology_counts(nonmanifold)["nonmanifold_edges"] == 1


def test_raster_union_is_monotonic_when_face_is_added():
    xy = {
        "A": (1.0, 1.0),
        "B": (6.0, 1.0),
        "C": (1.0, 6.0),
        "D": (6.0, 6.0),
    }
    before = _rasterize_faces([("A", "B", "C")], xy, (8, 8))
    after = _rasterize_faces([("A", "B", "C"), ("B", "D", "C")], xy, (8, 8))
    assert not np.any(before & ~after)
    assert np.count_nonzero(after) > np.count_nonzero(before)
