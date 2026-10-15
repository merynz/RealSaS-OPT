import pytest

from tools.corpus.fit8_cell_supported_patch_qualification_v1 import (
    CellSupportedAdjacencyPolicyV1,
    TriangleSharedEdgeWitnessV1,
    qualify_cell_supported_patches_v1,
)
from tools.corpus.fit8_local_patch_teacher_v1 import (
    CrossingCanonicalizationPolicyV1,
    ExtractionCellV1,
    SourceTriangleV1,
    extract_cell_teacher_v1,
)

CELL = ExtractionCellV1((0.0, 0.0, 0.0), 1.0, 2.0)
CROSSING_POLICY = CrossingCanonicalizationPolicyV1(1e-8, "SYNTHETIC_CELL_SUPPORTED_PATCH_TEST")
PATCH_POLICY = CellSupportedAdjacencyPolicyV1(1e-8, "SYNTHETIC_CELL_SUPPORTED_PATCH_TEST")


def _tri(triangle_id, shared0, shared1, third):
    return SourceTriangleV1(triangle_id, (shared0, shared1, third), 0)


def _teacher(triangles, connected=True):
    adjacency = {0: (1,), 1: (0,)} if connected else {}
    return extract_cell_teacher_v1(
        CELL,
        triangles,
        adjacency,
        crossing_canonicalization=CROSSING_POLICY,
    )


def _witness(p0, p1):
    return TriangleSharedEdgeWitnessV1(0, 1, p0, p1, "SYNTHETIC_SHARED_EDGE")


def test_shared_edge_outside_cell_splits_coarse_lower_bound_patch():
    p0, p1 = (2.0, -0.5, 0.0), (2.0, 0.5, 0.0)
    triangles = (
        _tri(0, p0, p1, (0.0, 0.0, -0.5)),
        _tri(1, p0, p1, (0.0, 0.0, 0.5)),
    )
    teacher = _teacher(triangles)
    assert len(teacher.local_patch_lower_bound) == 1
    result = qualify_cell_supported_patches_v1(
        teacher,
        triangles,
        (_witness(p0, p1),),
        policy=PATCH_POLICY,
    )
    assert len(result.patches) == 2
    assert result.outside_cell_pairs == ((0, 1),)
    assert result.positive_support_pairs == ()


def test_positive_length_shared_edge_keeps_one_cell_supported_patch():
    p0, p1 = (0.0, -0.5, 0.0), (0.0, 0.5, 0.0)
    triangles = (
        _tri(0, p0, p1, (-0.5, 0.0, -0.5)),
        _tri(1, p0, p1, (0.5, 0.0, 0.5)),
    )
    teacher = _teacher(triangles)
    result = qualify_cell_supported_patches_v1(
        teacher,
        triangles,
        (_witness(p0, p1),),
        policy=PATCH_POLICY,
    )
    assert len(result.patches) == 1
    assert result.positive_support_pairs == ((0, 1),)


def test_point_touch_is_retained_but_does_not_connect_local_sheets():
    p0, p1 = (1.0, 1.0, 1.0), (2.0, 1.0, 1.0)
    triangles = (
        _tri(0, p0, p1, (0.0, 0.0, 0.0)),
        _tri(1, p0, p1, (0.0, 0.0, 0.5)),
    )
    teacher = _teacher(triangles)
    result = qualify_cell_supported_patches_v1(
        teacher,
        triangles,
        (_witness(p0, p1),),
        policy=PATCH_POLICY,
    )
    assert len(result.patches) == 2
    assert result.point_touch_pairs == ((0, 1),)


def test_qualifier_cannot_merge_two_lower_bound_patches():
    p0, p1 = (0.0, -0.5, 0.0), (0.0, 0.5, 0.0)
    triangles = (
        _tri(0, p0, p1, (-0.5, 0.0, -0.5)),
        _tri(1, p0, p1, (0.5, 0.0, 0.5)),
    )
    teacher = _teacher(triangles, connected=False)
    with pytest.raises(ValueError, match="WITNESS_MERGES_LOWER_BOUND"):
        qualify_cell_supported_patches_v1(
            teacher,
            triangles,
            (_witness(p0, p1),),
            policy=PATCH_POLICY,
        )


def test_witness_must_be_an_actual_edge_of_both_triangles():
    p0, p1 = (0.0, -0.5, 0.0), (0.0, 0.5, 0.0)
    triangles = (
        _tri(0, p0, p1, (-0.5, 0.0, -0.5)),
        _tri(1, p0, p1, (0.5, 0.0, 0.5)),
    )
    teacher = _teacher(triangles)
    bad = TriangleSharedEdgeWitnessV1(0, 1, (0.0, -0.25, 0.0), p1, "BAD_SYNTHETIC_EDGE")
    with pytest.raises(ValueError, match="NOT_SHARED_TRIANGLE_EDGE"):
        qualify_cell_supported_patches_v1(
            teacher,
            triangles,
            (bad,),
            policy=PATCH_POLICY,
        )


def test_incomplete_witness_set_is_forbidden():
    with pytest.raises(ValueError, match="WITNESS_SET_INCOMPLETE"):
        CellSupportedAdjacencyPolicyV1(
            1e-8,
            "SYNTHETIC_INCOMPLETE_POLICY",
            witness_set_complete=False,
        ).validate()
