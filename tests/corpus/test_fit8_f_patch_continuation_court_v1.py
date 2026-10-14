from tools.corpus.fit8_f_patch_continuation_court_v1 import evaluate_neighbor_patch_continuation_v1
from tools.corpus.fit8_local_patch_teacher_v1 import (
    CrossingCanonicalizationPolicyV1,
    ExtractionCellTeacherV1,
    ExtractionCellV1,
    LocalPatchLowerBoundV1,
    SCHEMA_VERSION as LOCAL_SCHEMA_VERSION,
)

POLICY = CrossingCanonicalizationPolicyV1(
    distance_tolerance=1e-8,
    provenance="SYNTHETIC_CONTINUATION_TEST_V1",
)


def _cell(center, patch_rows):
    patches = tuple(
        LocalPatchLowerBoundV1(
            patch_local_id=i,
            source_triangle_ids=tuple(row[0]),
            geometric_component_ids=tuple(row[1]),
        )
        for i, row in enumerate(patch_rows)
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


def test_shared_source_triangle_is_strong_continuation_evidence():
    a = _cell((0, 0, 0), [((1, 2), (7,))])
    b = _cell((2, 0, 0), [((2, 3), (7,))])
    r = evaluate_neighbor_patch_continuation_v1(a, b, {})
    assert r.continuation_edge_count == 1
    assert r.continuation_edges[0].evidence == "SHARED_SOURCE_TRIANGLE"
    assert r.continuation_edges[0].shared_triangle_ids == (2,)
    assert r.one_to_one_fraction == 1.0


def test_source_triangle_adjacency_continues_when_triangle_ids_change():
    a = _cell((0, 0, 0), [((10,), (4,))])
    b = _cell((2, 0, 0), [((11,), (4,))])
    r = evaluate_neighbor_patch_continuation_v1(a, b, {10: [11], 11: [10]})
    assert r.continuation_edge_count == 1
    assert r.continuation_edges[0].evidence == "SOURCE_TRIANGLE_ADJACENCY"
    assert r.continuation_edges[0].adjacent_triangle_pairs == ((10, 11),)


def test_same_component_without_lineage_does_not_force_stitch():
    a = _cell((0, 0, 0), [((20,), (5,))])
    b = _cell((2, 0, 0), [((21,), (5,))])
    r = evaluate_neighbor_patch_continuation_v1(a, b, {})
    assert r.continuation_edge_count == 0
    assert r.unmatched_a_patch_ids == (0,)
    assert r.unmatched_b_patch_ids == (0,)


def test_two_parallel_sheets_remain_two_one_to_one_continuations():
    a = _cell((0, 0, 0), [((1,), (1,)), ((2,), (2,))])
    b = _cell((2, 0, 0), [((3,), (1,)), ((4,), (2,))])
    adjacency = {1: [3], 3: [1], 2: [4], 4: [2]}
    r = evaluate_neighbor_patch_continuation_v1(a, b, adjacency)
    assert r.continuation_edge_count == 2
    assert r.one_to_one_edge_count == 2
    assert r.one_to_one_fraction == 1.0
    assert r.max_patch_degree == 1


def test_branching_is_explicit_not_silently_forced_one_to_one():
    a = _cell((0, 0, 0), [((1,), (1,))])
    b = _cell((2, 0, 0), [((2,), (1,)), ((3,), (1,))])
    r = evaluate_neighbor_patch_continuation_v1(a, b, {1: [2, 3], 2: [1], 3: [1]})
    assert r.continuation_edge_count == 2
    assert r.branching_a_patch_ids == (0,)
    assert r.branching_b_patch_ids == ()
    assert r.one_to_one_edge_count == 0
    assert r.max_patch_degree == 2


def test_non_face_neighbors_fail_closed():
    a = _cell((0, 0, 0), [((1,), (1,))])
    b = _cell((2, 2, 0), [((1,), (1,))])
    try:
        evaluate_neighbor_patch_continuation_v1(a, b, {})
    except ValueError as exc:
        assert str(exc) == "F_PATCH_CONTINUATION_NOT_FACE_NEIGHBORS"
    else:
        raise AssertionError("diagonal cells must not be treated as face neighbors")
