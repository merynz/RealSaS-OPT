from tools.corpus.fit8_polygon_triangulation_qualification_v1 import (
    qualify_polygon_triangulation_topology_v1,
)


def test_good_quad_is_topology_qualified():
    report = qualify_polygon_triangulation_topology_v1(
        [(0, 1, 2, 3)],
        {0: [(0, 1, 2), (0, 2, 3)]},
    )
    assert report.topology_qualified


def test_internal_diagonal_aliasing_other_polygon_boundary_is_rejected():
    polygons = [(0, 1, 2, 3), (0, 2, 4)]
    triangles = {
        0: [(0, 1, 2), (0, 2, 3)],
        1: [(0, 2, 4)],
    }
    report = qualify_polygon_triangulation_topology_v1(polygons, triangles)
    assert (0, 2) in report.internal_diagonal_alias_authored_boundary_keys
    assert not report.topology_qualified


def test_internal_diagonal_reuse_across_polygons_is_rejected():
    polygons = [(0, 1, 2, 3), (0, 4, 2, 5)]
    triangles = {
        0: [(0, 1, 2), (0, 2, 3)],
        1: [(0, 4, 2), (0, 2, 5)],
    }
    report = qualify_polygon_triangulation_topology_v1(polygons, triangles)
    assert (0, 2) in report.internal_diagonal_reused_across_polygons_keys
    assert not report.topology_qualified


def test_boundary_edge_must_appear_once_per_polygon():
    report = qualify_polygon_triangulation_topology_v1(
        [(0, 1, 2, 3)],
        {0: [(0, 1, 2), (1, 2, 3)]},
    )
    assert report.boundary_incidence_violation_polygons == (0,)
    assert not report.topology_qualified


def test_n_minus_two_triangle_count_is_required():
    report = qualify_polygon_triangulation_topology_v1(
        [(0, 1, 2, 3)],
        {0: [(0, 1, 2)]},
    )
    assert report.triangle_count_mismatch_polygons == (0,)
    assert not report.topology_qualified
