from experiments.rest_preservation_v1.run_subject_free_calibration import (
    RESOLUTIONS,SHAPES,SHIFT_CASES,run_calibration,
)


def test_calibration_apparatus_is_subject_free_and_has_frozen_case_matrix():
    result=run_calibration()
    assert result["authority"]=="SUBJECT_FREE_SYNTHETIC_ONLY__NO_KNIGHT_OR_MAGE_INPUT"
    assert result["resolutions"]==list(RESOLUTIONS)
    assert result["shape_ids"]==list(SHAPES)
    assert result["case_count"]==len(RESOLUTIONS)*len(SHAPES)*(1+len(SHIFT_CASES)+3)
    assert result["view_measurement_count"]==result["case_count"]*8
    assert result["silhouette_selection"]["selection_rule"]=="MIDPOINT(MAX_IDENTITY_P95,MIN_ONE_PIXEL_SHIFT_P95)"
    assert result["exact_policy_candidates"]["max_overlap_rgba_mismatch_pixel_count"]==0
    assert result["exact_policy_candidates"]["max_cross_view_source_geometry_fraction"]==0.0


def test_calibration_report_never_claims_product_or_subject_pass():
    result=run_calibration()
    assert "product PASS" in result["claim_boundary"]
    assert "Knight" not in str(result["case_rows"])
    assert "Mage" not in str(result["case_rows"])
