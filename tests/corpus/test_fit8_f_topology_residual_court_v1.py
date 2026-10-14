from tools.corpus.fit8_f_topology_residual_court_v1 import evaluate_face_topology_residual_v1
from tools.corpus.fit8_face_strand_teacher_v1 import (
    FaceSegmentV1,
    FaceStrandTeacherV1,
    FaceStrandV1,
    SCHEMA_VERSION as FACE_SCHEMA_VERSION,
)
from tools.corpus.fit8_local_patch_teacher_v1 import (
    CrossingCanonicalizationPolicyV1,
    ExtractionCellTeacherV1,
    ExtractionCellV1,
    LocalPatchLowerBoundV1,
    SCHEMA_VERSION as LOCAL_SCHEMA_VERSION,
)

POLICY = CrossingCanonicalizationPolicyV1(1e-8, "SYNTHETIC_TOPOLOGY_RESIDUAL_TEST_V1")


def _cell(patches):
    rows = tuple(
        LocalPatchLowerBoundV1(i, tuple(triangle_ids), (i,))
        for i, triangle_ids in enumerate(patches)
    )
    ids = tuple(sorted({triangle_id for row in rows for triangle_id in row.source_triangle_ids}))
    value = ExtractionCellTeacherV1(
        LOCAL_SCHEMA_VERSION,
        ExtractionCellV1((0.0, 0.0, 0.0), 1.0, 2.0),
        True,
        ids,
        rows,
        (),
        (),
        0,
        POLICY,
    )
    value.validate()
    return value


def _strand(strand_id, triangle_ids, kind="BOUNDARY_TO_BOUNDARY_ARC"):
    segment = FaceSegmentV1(triangle_ids[0], (-1.0, 0.0), (1.0, 0.0))
    return FaceStrandV1(
        strand_id,
        tuple(triangle_ids),
        (segment,),
        kind,
        2 if kind == "BOUNDARY_TO_BOUNDARY_ARC" else 0,
        0,
        2,
    )


def _face(strands, coplanar=(), points=()):
    return FaceStrandTeacherV1(
        FACE_SCHEMA_VERSION,
        1,
        tuple(strands),
        tuple(coplanar),
        tuple(points),
    )


def test_two_ordinary_arcs_on_one_patch_need_no_residual():
    report = evaluate_face_topology_residual_v1(
        _cell([(1, 2)]),
        _face([_strand(0, (1,)), _strand(1, (2,))]),
    )
    assert not report.topology_residual_required


def test_third_arc_on_one_patch_activates_residual():
    report = evaluate_face_topology_residual_v1(
        _cell([(1, 2, 3)]),
        _face([_strand(0, (1,)), _strand(1, (2,)), _strand(2, (3,))]),
    )
    assert "PATCH_GT2_STRANDS" in report.topology_residual_reasons


def test_closed_loop_activates_residual():
    report = evaluate_face_topology_residual_v1(
        _cell([(1,)]),
        _face([_strand(0, (1,), "CLOSED_LOOP")]),
    )
    assert "CLOSED_LOOP" in report.topology_residual_reasons


def test_coplanar_support_is_canonicalization_not_model_capacity():
    report = evaluate_face_topology_residual_v1(
        _cell([(1,)]),
        _face([_strand(0, (1,))], coplanar=(1,)),
    )
    assert not report.topology_residual_required
    assert report.canonicalization_required


def test_one_strand_spanning_two_local_patches_is_explicit_residual():
    report = evaluate_face_topology_residual_v1(
        _cell([(1,), (2,)]),
        _face([_strand(0, (1, 2))]),
    )
    assert "STRAND_MULTI_PATCH" in report.topology_residual_reasons
