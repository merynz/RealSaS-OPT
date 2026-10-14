from tools.corpus.fit8_face_strand_boundary_incidence_v1 import (
    FaceBoundaryIncidencePolicyV1,
    qualify_face_strand_boundary_incidence_v1,
)
from tools.corpus.fit8_face_strand_teacher_v1 import derive_face_strands_v1
from tools.corpus.fit8_local_patch_teacher_v1 import ExtractionCellV1, SourceTriangleV1

CELL = ExtractionCellV1((0.0, 0.0, 0.0), 1.0, 2.0)
POLICY = FaceBoundaryIncidencePolicyV1(1e-9, "SYNTHETIC_FACE_BOUNDARY_INCIDENCE_TEST_V1")


def _triangle(tid, a, b, c, component=0):
    return SourceTriangleV1(tid, (a, b, c), component)


def test_x_positive_boundary_arc_binds_to_exact_cube_edges():
    triangle = _triangle(0, (0, 0, 0), (2, -2, 0), (2, 2, 0))
    face = derive_face_strands_v1(CELL, 1, [triangle], {})
    result = qualify_face_strand_boundary_incidence_v1(CELL, face, policy=POLICY)
    assert result.policy.provenance == "SYNTHETIC_FACE_BOUNDARY_INCIDENCE_TEST_V1"
    assert result.strand_incidence[0].boundary_endpoint_cube_edge_sets == ((9,), (11,))


def test_closed_loop_has_no_boundary_cube_edge_incidence():
    points = [(1, 0, 0.5), (1, 0.5, 0), (1, 0, -0.5), (1, -0.5, 0)]
    triangles = [
        _triangle(i, points[i], points[(i + 1) % 4], (0, 0, 0))
        for i in range(4)
    ]
    adjacency = {i: [(i - 1) % 4, (i + 1) % 4] for i in range(4)}
    face = derive_face_strands_v1(CELL, 1, triangles, adjacency)
    assert face.strands[0].kind == "CLOSED_LOOP"
    result = qualify_face_strand_boundary_incidence_v1(CELL, face, policy=POLICY)
    assert result.strand_incidence[0].boundary_endpoint_cube_edge_sets == ()


def test_face_corner_endpoint_preserves_both_incident_cube_edges():
    # On x+ face, this triangle-plane segment reaches corner (x=1,y=1,z=1).
    triangle = _triangle(0, (0, 0, 0), (2, 2, 2), (2, -2, 0))
    face = derive_face_strands_v1(CELL, 1, [triangle], {})
    result = qualify_face_strand_boundary_incidence_v1(CELL, face, policy=POLICY)
    edge_sets = result.strand_incidence[0].boundary_endpoint_cube_edge_sets
    assert any(len(edge_set) == 2 for edge_set in edge_sets)
    assert (7, 11) in edge_sets


def test_policy_requires_explicit_provenance():
    policy = FaceBoundaryIncidencePolicyV1(1e-9, "")
    try:
        policy.validate()
    except ValueError as exc:
        assert "PROVENANCE_MISSING" in str(exc)
    else:
        raise AssertionError("missing qualification provenance must fail closed")
