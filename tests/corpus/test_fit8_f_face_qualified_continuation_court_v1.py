from tools.corpus.fit8_f_face_qualified_continuation_court_v1 import (
    evaluate_face_qualified_neighbor_patch_continuation_v1,
)
from tools.corpus.fit8_local_patch_teacher_v1 import (
    CrossingCanonicalizationPolicyV1,
    ExtractionCellTeacherV1,
    ExtractionCellV1,
    LocalPatchLowerBoundV1,
    SCHEMA_VERSION as LOCAL_SCHEMA_VERSION,
)
from tools.corpus.fit8_patch_face_support_v1 import (
    PatchFaceSupportReportV1,
    PatchFaceSupportRowV1,
    SCHEMA_VERSION as FACE_SCHEMA_VERSION,
)

POLICY = CrossingCanonicalizationPolicyV1(
    distance_tolerance=1e-8,
    provenance="SYNTHETIC_FACE_CONTINUATION_TEST_V1",
)


def _cell(center, patch_rows):
    patches = tuple(
        LocalPatchLowerBoundV1(i, tuple(triangles), tuple(components))
        for i, (triangles, components) in enumerate(patch_rows)
    )
    ids = tuple(sorted({tid for patch in patches for tid in patch.source_triangle_ids}))
    value = ExtractionCellTeacherV1(
        schema_version=LOCAL_SCHEMA_VERSION,
        cell=ExtractionCellV1(tuple(center), 1.0, 2.0),
        active=bool(ids),
        intersecting_triangle_ids=ids,
        local_patch_lower_bound=patches,
        edge_crossings=(),
        coplanar_edge_support=(),
        max_distinct_crossings_per_edge=0,
        crossing_canonicalization=POLICY,
    )
    value.validate()
    return value


def _support(*face_rows):
    value = PatchFaceSupportReportV1(
        FACE_SCHEMA_VERSION,
        tuple(PatchFaceSupportRowV1(i, tuple(faces)) for i, faces in enumerate(face_rows)),
    )
    value.validate()
    return value


def test_interior_patch_is_not_counted_as_unmatched():
    a = _cell((0, 0, 0), [((1,), (1,)), ((9,), (9,))])
    b = _cell((2, 0, 0), [((1,), (1,))])
    report = evaluate_face_qualified_neighbor_patch_continuation_v1(
        a,
        b,
        _support((1,), ()),
        _support((0,)),
        {},
    )
    assert report.a_face_patch_ids == (0,)
    assert report.b_face_patch_ids == (0,)
    assert report.continuation_complete
    assert report.one_to_one_fraction == 1.0


def test_face_patch_without_lineage_is_explicitly_unmatched():
    a = _cell((0, 0, 0), [((1,), (1,))])
    b = _cell((2, 0, 0), [((2,), (1,))])
    report = evaluate_face_qualified_neighbor_patch_continuation_v1(
        a,
        b,
        _support((1,)),
        _support((0,)),
        {},
    )
    assert not report.continuation_complete
    assert report.unmatched_a_patch_ids == (0,)
    assert report.unmatched_b_patch_ids == (0,)


def test_reverse_neighbor_direction_selects_opposite_faces():
    a = _cell((2, 0, 0), [((1,), (1,))])
    b = _cell((0, 0, 0), [((1,), (1,))])
    report = evaluate_face_qualified_neighbor_patch_continuation_v1(
        a,
        b,
        _support((0,)),
        _support((1,)),
        {},
    )
    assert (report.a_shared_face_index, report.b_shared_face_index) == (0, 1)
    assert report.continuation_complete
