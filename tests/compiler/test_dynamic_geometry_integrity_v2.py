from __future__ import annotations

import numpy as np

from compiler.realsas_compiler_core.dynamic_geometry_integrity_v2 import (
    nonadjacent_intersection_pairs,
    triangles_intersect_sat,
    unexpected_intersection_pairs,
)


def test_nonadjacent_crossing_triangles_are_detected():
    a = np.asarray(
        ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    b = np.asarray(
        ((0.0, 0.25, -1.0), (0.0, 0.25, 1.0), (0.0, -1.0, 0.0)),
        dtype=np.float64,
    )
    assert triangles_intersect_sat(a, b)


def test_separated_triangles_are_not_false_positive():
    a = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    b = a + np.asarray((0.0, 0.0, 2.0), dtype=np.float64)
    assert not triangles_intersect_sat(a, b)


def test_nonadjacent_pair_census_excludes_shared_vertex_neighbors():
    vertices = np.asarray(
        (
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.25, -1.0),
            (0.0, 0.25, 1.0),
            (0.0, -1.0, 0.0),
            (2.0, 0.0, 0.0),
        ),
        dtype=np.float64,
    )
    faces = np.asarray(
        (
            (0, 1, 2),
            (3, 4, 5),
            (1, 6, 2),
        ),
        dtype=np.int64,
    )
    pairs = nonadjacent_intersection_pairs(vertices=vertices, faces=faces)
    assert (0, 1) in pairs
    assert all(pair != (0, 2) for pair in pairs)



def test_topology_aware_census_allows_expected_shared_edge_contact():
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
        ),
        dtype=np.float64,
    )
    faces = np.asarray(((0, 1, 2), (1, 3, 2)), dtype=np.int64)
    assert unexpected_intersection_pairs(vertices=vertices, faces=faces) == ()


def test_topology_aware_census_rejects_shared_vertex_penetration_away_from_vertex():
    vertices = np.asarray(
        (
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.25, -1.0),
            (0.0, 0.25, 1.0),
        ),
        dtype=np.float64,
    )
    faces = np.asarray(((0, 1, 2), (0, 3, 4)), dtype=np.int64)
    assert (0, 1) in unexpected_intersection_pairs(
        vertices=vertices,
        faces=faces,
    )


def test_topology_aware_census_rejects_shared_edge_interior_overlap():
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.2, 0.8, 0.0),
        ),
        dtype=np.float64,
    )
    faces = np.asarray(((0, 1, 2), (0, 1, 3)), dtype=np.int64)
    assert (0, 1) in unexpected_intersection_pairs(
        vertices=vertices,
        faces=faces,
    )
