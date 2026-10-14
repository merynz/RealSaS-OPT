from tools.corpus.fit8_f_oracle_assembly_court_v1 import (
    PatchContinuationV1,
    PatchInstanceKeyV1,
    PatchInstanceV1,
    evaluate_oracle_patch_assembly_v1,
)


def _key(cell_xyz, patch_local_id):
    return PatchInstanceKeyV1(tuple(cell_xyz), patch_local_id)


def test_exact_component_partition_closes():
    a = PatchInstanceV1(_key((0, 0, 0), 0), (1, 2))
    b = PatchInstanceV1(_key((1, 0, 0), 0), (2, 3))
    c = PatchInstanceV1(_key((0, 1, 0), 0), (10,))
    report = evaluate_oracle_patch_assembly_v1(
        {1: 0, 2: 0, 3: 0, 10: 1},
        [a, b, c],
        [PatchContinuationV1(a.key, b.key)],
    )
    assert report.exact_component_partition


def test_missing_continuation_detects_split():
    a = PatchInstanceV1(_key((0, 0, 0), 0), (1,))
    b = PatchInstanceV1(_key((1, 0, 0), 0), (2,))
    report = evaluate_oracle_patch_assembly_v1({1: 0, 2: 0}, [a, b], [])
    assert report.split_source_components == 1
    assert not report.exact_component_partition


def test_false_continuation_detects_merge():
    a = PatchInstanceV1(_key((0, 0, 0), 0), (1,))
    b = PatchInstanceV1(_key((1, 0, 0), 0), (2,))
    report = evaluate_oracle_patch_assembly_v1(
        {1: 0, 2: 1},
        [a, b],
        [PatchContinuationV1(a.key, b.key)],
    )
    assert report.false_merge_components == 1
    assert not report.exact_component_partition


def test_missing_source_triangle_detected():
    a = PatchInstanceV1(_key((0, 0, 0), 0), (1,))
    report = evaluate_oracle_patch_assembly_v1({1: 0, 2: 0}, [a], [])
    assert report.missing_source_triangles == 1
    assert not report.exact_component_partition
