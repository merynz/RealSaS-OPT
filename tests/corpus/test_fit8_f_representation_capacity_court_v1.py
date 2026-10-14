from tools.corpus.fit8_f_representation_capacity_court_v1 import evaluate_f_representation_capacity_v1
from tools.corpus.fit8_local_patch_teacher_v1 import ExtractionCellV1, SourceTriangleV1, extract_cell_teacher_v1

CELL = ExtractionCellV1((0.0, 0.0, 0.0), 1.0, 2.0)


def _plane(tid, x, component):
    return SourceTriangleV1(tid, ((x, -2, -2), (x, 2, -2), (x, 0, 2)), component)


def test_fa_fails_parallel_distinct_patches_but_fb_represents_them():
    cell = extract_cell_teacher_v1(CELL, [_plane(0, -0.25, 0), _plane(1, 0.25, 1)])
    report = evaluate_f_representation_capacity_v1([cell])
    assert report.teacher_crossing_count > 0
    assert report.f_a_recall_ceiling < 1.0
    assert report.f_b_recall_ceiling == 1.0
    assert report.cells_requiring_more_than_f_a == 1
    assert report.cells_requiring_more_than_f_b == 0


def test_fb_fails_when_one_local_patch_crosses_same_edge_twice():
    triangles = [_plane(0, -0.25, 7), _plane(1, 0.25, 7)]
    cell = extract_cell_teacher_v1(
        CELL,
        triangles,
        triangle_adjacency={0: [1], 1: [0]},
    )
    report = evaluate_f_representation_capacity_v1([cell])
    assert report.f_b_recall_ceiling < 1.0
    assert report.cells_requiring_more_than_f_b == 1
    assert report.max_teacher_crossings_per_patch_edge == 2
    assert report.f_c_recall_ceiling == 1.0


def test_ordinary_single_sheet_cell_is_representable_by_all_families():
    cell = extract_cell_teacher_v1(CELL, [_plane(0, 0.0, 0)])
    report = evaluate_f_representation_capacity_v1([cell])
    assert report.f_a_recall_ceiling == 1.0
    assert report.f_b_recall_ceiling == 1.0
    assert report.f_c_recall_ceiling == 1.0
    assert report.cells_requiring_more_than_f_a == 0
    assert report.cells_requiring_more_than_f_b == 0


def test_empty_cell_does_not_create_fake_capacity_failures():
    cell = extract_cell_teacher_v1(
        CELL,
        [SourceTriangleV1(0, ((5, 5, 5), (6, 5, 5), (5, 6, 5)), 0)],
    )
    report = evaluate_f_representation_capacity_v1([cell])
    assert report.active_cell_count == 0
    assert report.teacher_crossing_count == 0
    assert report.f_a_unmatched_crossings == 0
    assert report.f_b_unmatched_patch_crossings == 0
