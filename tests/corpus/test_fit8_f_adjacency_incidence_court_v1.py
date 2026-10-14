from tools.corpus.fit8_f_adjacency_incidence_court_v1 import evaluate_oracle_adjacency_incidence_v1
from tools.corpus.fit8_f_oracle_assembly_court_v1 import (
    PatchContinuationV1,
    PatchInstanceKeyV1,
    PatchInstanceV1,
)


def _patch(cell, local_id, *triangles):
    return PatchInstanceV1(PatchInstanceKeyV1(cell, local_id), tuple(sorted(triangles)))


def test_intra_patch_witness_closes_source_adjacency():
    patch = _patch((0, 0, 0), 0, 0, 1)
    report = evaluate_oracle_adjacency_incidence_v1({0: [1], 1: [0]}, [patch], [])
    assert report.exact_incidence_coverage
    assert report.witnessed_intra_patch_count == 1
    assert report.witnessed_cross_cell_count == 0


def test_cross_cell_continuation_witnesses_source_adjacency():
    a = _patch((0, 0, 0), 0, 0)
    b = _patch((1, 0, 0), 0, 1)
    continuation = PatchContinuationV1(a.key, b.key)
    report = evaluate_oracle_adjacency_incidence_v1(
        {0: [1], 1: [0]}, [a, b], [continuation]
    )
    assert report.exact_incidence_coverage
    assert report.witnessed_intra_patch_count == 0
    assert report.witnessed_cross_cell_count == 1


def test_missing_continuation_exposes_missing_source_adjacency():
    a = _patch((0, 0, 0), 0, 0)
    b = _patch((1, 0, 0), 0, 1)
    report = evaluate_oracle_adjacency_incidence_v1({0: [1], 1: [0]}, [a, b], [])
    assert not report.exact_incidence_coverage
    assert report.missing_source_adjacencies == ((0, 1),)


def test_unrelated_continuation_is_fail_closed():
    a = _patch((0, 0, 0), 0, 0)
    b = _patch((1, 0, 0), 0, 2)
    c = _patch((0, 1, 0), 0, 1)
    continuation = PatchContinuationV1(a.key, b.key)
    report = evaluate_oracle_adjacency_incidence_v1(
        {0: [1], 1: [0], 2: []}, [a, b, c], [continuation]
    )
    assert not report.exact_incidence_coverage
    assert report.unjustified_continuations == ((a.key, b.key),)
    assert report.missing_source_adjacencies == ((0, 1),)


def test_shared_triangle_continuation_is_lineage_valid_without_inventing_adjacency():
    a = _patch((0, 0, 0), 0, 0, 1)
    b = _patch((1, 0, 0), 0, 1, 2)
    continuation = PatchContinuationV1(a.key, b.key)
    report = evaluate_oracle_adjacency_incidence_v1(
        {0: [1], 1: [0, 2], 2: [1]}, [a, b], [continuation]
    )
    assert report.exact_incidence_coverage
    assert report.unjustified_continuations == ()
    assert report.source_adjacency_count == 2
