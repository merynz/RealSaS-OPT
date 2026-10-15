from tools.corpus.fit8_cell_supported_patch_qualification_v1 import (
    CellSupportedAdjacencyPolicyV1,
    CellSupportedPatchQualificationV1,
    CellSupportedPatchV1,
)
from tools.corpus.fit8_f_representation_capacity_court_v2 import (
    QualifiedCellCapacityInputV2,
    evaluate_f_representation_capacity_v2,
)
from tools.corpus.fit8_local_patch_teacher_v1 import (
    CrossingCanonicalizationPolicyV1,
    EdgeCrossingV1,
    ExtractionCellTeacherV1,
    ExtractionCellV1,
    LocalPatchLowerBoundV1,
)


def _crossing(t, triangle_id):
    return EdgeCrossingV1(
        edge_index=0,
        t=t,
        xyz=(t - 0.5, -0.5, -0.5),
        source_triangle_ids=(triangle_id,),
        geometric_component_ids=(0,),
        owner_region_ids=(),
        material_region_ids=(),
        mechanical_owner_ids=(),
        normal_xyz=None,
        normal_valid=False,
    )


def _teacher():
    return ExtractionCellTeacherV1(
        schema_version="RealSaS.FIT8LocalPatchTeacher.v1",
        cell=ExtractionCellV1(center_xyz=(0.0, 0.0, 0.0), half_extent=0.5, pitch=1.0),
        active=True,
        intersecting_triangle_ids=(0, 1),
        # Deliberately coarse/over-merged diagnostic parent.
        local_patch_lower_bound=(
            LocalPatchLowerBoundV1(
                patch_local_id=0,
                source_triangle_ids=(0, 1),
                geometric_component_ids=(0,),
            ),
        ),
        edge_crossings=(_crossing(0.25, 0), _crossing(0.75, 1)),
        coplanar_edge_support=(),
        max_distinct_crossings_per_edge=2,
        crossing_canonicalization=CrossingCanonicalizationPolicyV1(
            distance_tolerance=1e-6,
            provenance="synthetic_v2_guard",
        ),
    )


def _qualification(*patch_triangle_groups):
    return CellSupportedPatchQualificationV1(
        schema_version="RealSaS.CellSupportedPatchQualification.v1",
        policy=CellSupportedAdjacencyPolicyV1(
            geometry_tolerance=1e-6,
            provenance="synthetic_v2_guard",
            witness_set_complete=True,
        ),
        patches=tuple(
            CellSupportedPatchV1(
                patch_local_id=patch_id,
                parent_lower_bound_patch_id=0,
                source_triangle_ids=tuple(group),
            )
            for patch_id, group in enumerate(patch_triangle_groups)
        ),
        positive_support_pairs=(),
        point_touch_pairs=(),
        outside_cell_pairs=(),
    )


def test_qualified_split_removes_false_f_b_residual_from_coarse_overmerge():
    teacher = _teacher()
    qualified = _qualification((0,), (1,))
    report = evaluate_f_representation_capacity_v2(
        [QualifiedCellCapacityInputV2(teacher, qualified)]
    )

    assert report.teacher_crossing_count == 2
    assert report.cells_requiring_more_than_f_a == 1
    assert report.cells_requiring_f_r_edge_residual == 0
    assert report.f_b_unmatched_patch_crossings == 0
    assert report.f_b_recall_ceiling == 1.0
    assert report.f_r_explicit_payload_fraction_of_f_c == 0.0


def test_same_qualified_sheet_multi_crossing_requires_f_r_residual():
    teacher = _teacher()
    qualified = _qualification((0, 1),)
    report = evaluate_f_representation_capacity_v2(
        [QualifiedCellCapacityInputV2(teacher, qualified)]
    )

    assert report.teacher_crossing_count == 2
    assert report.cells_requiring_f_r_edge_residual == 1
    assert report.f_b_unmatched_patch_crossings == 1
    assert report.f_r_explicit_extra_crossing_payload == 1
    assert report.f_r_explicit_payload_fraction_of_f_c == 0.5
    assert report.f_r_local_recall_ceiling_with_perfect_activation == 1.0


def test_f_r_oracle_ceiling_is_not_a_learned_activation_pass():
    report = evaluate_f_representation_capacity_v2(
        [QualifiedCellCapacityInputV2(_teacher(), _qualification((0, 1),))]
    )
    # This property is intentionally an oracle capacity statement only. The
    # learned residual detector must later prove recall in a separate court.
    assert report.f_r_local_recall_ceiling_with_perfect_activation == 1.0
    assert report.cells_requiring_f_r_edge_residual == 1
