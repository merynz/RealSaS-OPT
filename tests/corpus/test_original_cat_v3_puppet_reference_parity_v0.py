"""Independent physical source-vs-reference court: synthetic formula tests only."""
import numpy as np
import pytest
from tools.corpus.court_original_cat_v3_puppet_reference_parity_v0 import (
    compare_original_v3_source_against_puppet,
    TOL,
)


def synthetic():
    predicted = np.zeros((2, 4, 4, 2, 2, 4), dtype=np.float32)
    predicted[..., 0] = 0.2
    predicted[..., 1] = 0.4
    predicted[..., 2] = 0.6
    predicted[..., 3] = 1.0
    source = np.repeat(predicted[:, :, None], 2, axis=2)
    diag = tuple(
        {"pose_ordinal": (0, 14)[pi],
         "view_index": (0, 2, 4, 6)[vi],
         "source_arm_index": ai,
         "layer_overflow_pixels": 0,
         "exact_depth_ambiguity_pixels": 0}
        for pi in range(2) for vi in range(4) for ai in range(4)
    )
    return predicted, source, diag


def test_all_original_physical_scope_comparisons_are_separate():
    p, s, d = synthetic()
    out = compare_original_v3_source_against_puppet(p, s, d)
    assert out["physical_original_full_source_render_comparisons"] == 64
    assert out["predicted_reference_frames"] == 32
    assert out["status"] == "SATISFIED_ORIGINAL_CAT_FINITE_REFERENCE_ONLY"
    assert out["total_source_sample_operation_pixel_mismatches"] == 0
    assert len(out["operation_breakdown"]) == 64
    assert not any(out[k] for k in (
        "graphics_training_authorized", "iris_training_authorized",
        "tessa_training_authorized", "materialization_authorized",
        "original_artist_drawable_owner_authorized"))


def test_independent_physical_source_mismatch_is_not_hidden_by_other_sample():
    p, s, d = synthetic()
    s[1, 2, 1, 3, 0, 1, 0] += 0.1
    out = compare_original_v3_source_against_puppet(p, s, d)
    assert out["status"] == "VIOLATED_PHYSICAL_SOURCE_PARITY"
    assert out["total_source_sample_operation_pixel_mismatches"] == 1
    assert out["maximum_premultiplied_rgba_error"] > TOL
    assert sum(row["mismatched_sample_pixels"] for row in out[
        "operation_breakdown"]) == 1


def test_valid_nonbinary_physical_alpha_compares_premultiplied_rgba():
    p, s, d = synthetic()
    s[..., :3] = 0.6
    s[..., 3] = 0.5
    p[..., :3] = 0.3
    p[..., 3] = 0.5
    out = compare_original_v3_source_against_puppet(p, s, d)
    assert out["total_source_sample_operation_pixel_mismatches"] == 0


def test_visibility_unsupported_does_not_become_physical_pass():
    p, s, d = synthetic()
    dd = [dict(row) for row in d]
    dd[0]["layer_overflow_pixels"] = 1
    out = compare_original_v3_source_against_puppet(p, s, dd)
    assert out["status"] == "UNKNOWN_UNQUALIFIED_VISIBILITY_SUPPORT"
    s[1, 1, 0, 1, 1, 1, 0] += 0.1
    out = compare_original_v3_source_against_puppet(p, s, dd)
    assert out["status"] == "VIOLATED_WITH_UNQUALIFIED_VISIBILITY_SUPPORT"


def test_frame_order_is_bound_to_source_pose_view_arm():
    p, s, d = synthetic()
    dd = [dict(x) for x in d]
    dd[0]["view_index"] = 4
    with pytest.raises(ValueError, match="FRAME_DIAGNOSTICS_ORDER_DRIFT"):
        compare_original_v3_source_against_puppet(p, s, dd)


def test_tolerance_cannot_be_tuned_after_original_experiment():
    p, s, d = synthetic()
    with pytest.raises(ValueError, match="PRECOMMITTED_TOLERANCE_DRIFT"):
        compare_original_v3_source_against_puppet(p, s, d, tolerance=0.2)
