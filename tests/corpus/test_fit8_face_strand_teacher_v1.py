from tools.corpus.fit8_face_strand_teacher_v1 import derive_face_strands_v1
from tools.corpus.fit8_local_patch_teacher_v1 import ExtractionCellV1, SourceTriangleV1

CELL = ExtractionCellV1((0.0, 0.0, 0.0), 1.0, 2.0)


def _triangle(tid, a, b, c, component=0):
    return SourceTriangleV1(tid, (a, b, c), component)


def test_single_boundary_to_boundary_arc():
    triangle = _triangle(0, (0, 0, 0), (2, -2, 0), (2, 2, 0))
    result = derive_face_strands_v1(CELL, 1, [triangle], {})
    assert len(result.strands) == 1
    assert result.strands[0].kind == "BOUNDARY_TO_BOUNDARY_ARC"


def test_closed_face_loop_is_first_class_truth():
    points = [(1, 0, 0.5), (1, 0.5, 0), (1, 0, -0.5), (1, -0.5, 0)]
    triangles = [
        _triangle(i, points[i], points[(i + 1) % 4], (0, 0, 0))
        for i in range(4)
    ]
    adjacency = {i: [(i - 1) % 4, (i + 1) % 4] for i in range(4)}
    result = derive_face_strands_v1(CELL, 1, triangles, adjacency)
    assert len(result.strands) == 1
    assert result.strands[0].kind == "CLOSED_LOOP"


def test_two_unrelated_arcs_remain_separate_even_inside_one_component():
    a = _triangle(0, (0, 0, -0.5), (2, -2, -0.5), (2, 2, -0.5), 7)
    b = _triangle(1, (0, 0, 0.5), (2, -2, 0.5), (2, 2, 0.5), 7)
    result = derive_face_strands_v1(CELL, 1, [a, b], {})
    assert len(result.strands) == 2


def test_interior_open_endpoint_is_not_rewritten_as_boundary_arc():
    triangle = _triangle(0, (0, 0, 0), (2, 0, 0), (2, 2, 0))
    result = derive_face_strands_v1(CELL, 1, [triangle], {})
    assert result.strands[0].kind == "OPEN_INTERIOR_ENDPOINT"
