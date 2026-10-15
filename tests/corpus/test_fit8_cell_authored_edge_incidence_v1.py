import pytest

from tools.corpus.fit8_cell_authored_edge_incidence_v1 import (
    AUTHORED_NONMANIFOLD,
    AUTHORED_OPEN_BOUNDARY,
    MANIFOLD_INTERIOR,
    OUTSIDE,
    POINT_TOUCH,
    POSITIVE_LENGTH,
    CellAuthoredEdgeIncidencePolicyV1,
    QualifiedSurfaceEdgeV1,
    qualify_cell_authored_edge_incidence_v1,
)
from tools.corpus.fit8_local_patch_teacher_v1 import ExtractionCellV1

CELL = ExtractionCellV1((0.0, 0.0, 0.0), 1.0, 2.0)
POLICY = CellAuthoredEdgeIncidencePolicyV1(1e-8, "SYNTHETIC_AUTHORED_EDGE_TEST")


def _edge(edge_id, a, b, incident):
    return QualifiedSurfaceEdgeV1(edge_id, a, b, tuple(incident), "SYNTHETIC_QUALIFIED_EDGE")


def test_positive_length_open_boundary_is_first_class_local_topology():
    result = qualify_cell_authored_edge_incidence_v1(
        CELL,
        (_edge(0, (-0.5, 0.0, 0.0), (0.5, 0.0, 0.0), (7,)),),
        policy=POLICY,
    )
    row = result.rows[0]
    assert row.topology_kind == AUTHORED_OPEN_BOUNDARY
    assert row.support_kind == POSITIVE_LENGTH


def test_nonmanifold_edge_is_not_rewritten_as_open_boundary():
    result = qualify_cell_authored_edge_incidence_v1(
        CELL,
        (_edge(1, (-0.5, 0.0, 0.0), (0.5, 0.0, 0.0), (1, 2, 3)),),
        policy=POLICY,
    )
    assert result.rows[0].topology_kind == AUTHORED_NONMANIFOLD


def test_two_face_edge_remains_manifold_interior():
    result = qualify_cell_authored_edge_incidence_v1(
        CELL,
        (_edge(2, (-0.5, 0.0, 0.0), (0.5, 0.0, 0.0), (1, 2)),),
        policy=POLICY,
    )
    assert result.rows[0].topology_kind == MANIFOLD_INTERIOR


def test_point_touch_is_retained_separately_from_positive_boundary_support():
    result = qualify_cell_authored_edge_incidence_v1(
        CELL,
        (_edge(3, (1.0, 1.0, 1.0), (2.0, 1.0, 1.0), (5,)),),
        policy=POLICY,
    )
    assert result.rows[0].support_kind == POINT_TOUCH


def test_outside_edge_is_explicit():
    result = qualify_cell_authored_edge_incidence_v1(
        CELL,
        (_edge(4, (2.0, 2.0, 2.0), (3.0, 2.0, 2.0), (5,)),),
        policy=POLICY,
    )
    assert result.rows[0].support_kind == OUTSIDE


def test_incomplete_edge_set_is_forbidden():
    with pytest.raises(ValueError, match="EDGE_SET_INCOMPLETE"):
        CellAuthoredEdgeIncidencePolicyV1(
            1e-8,
            "INCOMPLETE_SYNTHETIC_EDGE_SET",
            edge_set_complete=False,
        ).validate()
