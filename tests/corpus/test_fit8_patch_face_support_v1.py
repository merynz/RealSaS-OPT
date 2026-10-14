from tools.corpus.fit8_local_patch_teacher_v1 import (
    CrossingCanonicalizationPolicyV1,
    ExtractionCellV1,
    SourceTriangleV1,
    extract_cell_teacher_v1,
)
from tools.corpus.fit8_patch_face_support_v1 import derive_patch_face_support_v1

POLICY = CrossingCanonicalizationPolicyV1(
    distance_tolerance=1e-8,
    provenance="SYNTHETIC_PATCH_FACE_SUPPORT_TEST_V1",
)
CELL = ExtractionCellV1((0.0, 0.0, 0.0), 1.0, 2.0)


def _extract(triangles, adjacency=None):
    return extract_cell_teacher_v1(
        CELL,
        triangles,
        adjacency or {},
        crossing_canonicalization=POLICY,
    )


def test_spanning_triangle_reaches_both_x_faces():
    triangle = SourceTriangleV1(
        0,
        ((-2.0, -2.0, 0.0), (2.0, -2.0, 0.0), (0.0, 2.0, 0.0)),
        0,
    )
    report = derive_patch_face_support_v1(_extract([triangle]), [triangle])
    assert 0 in report.faces_for_patch(0)
    assert 1 in report.faces_for_patch(0)


def test_interior_triangle_reaches_no_cell_face():
    triangle = SourceTriangleV1(
        0,
        ((-0.2, -0.2, 0.0), (0.2, -0.2, 0.0), (0.0, 0.2, 0.0)),
        0,
    )
    report = derive_patch_face_support_v1(_extract([triangle]), [triangle])
    assert report.faces_for_patch(0) == ()


def test_triangle_coplanar_with_positive_x_face_is_supported():
    triangle = SourceTriangleV1(
        0,
        ((1.0, -0.5, -0.5), (1.0, 0.5, -0.5), (1.0, 0.0, 0.5)),
        0,
    )
    report = derive_patch_face_support_v1(_extract([triangle]), [triangle])
    assert 1 in report.faces_for_patch(0)


def test_missing_source_triangle_fails_closed():
    triangle = SourceTriangleV1(
        0,
        ((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0)),
        0,
    )
    cell = _extract([triangle])
    try:
        derive_patch_face_support_v1(cell, [])
    except ValueError as exc:
        assert str(exc) == "FIT8_PATCH_FACE_SUPPORT_TRIANGLE_MISSING"
    else:
        raise AssertionError("missing source triangle must fail closed")
