"""Synthetic bottom-left storage to top-left physical-court tests only."""
import numpy as np
import pytest
from tools.corpus.court_original_cat_v3_blender_pixel_origin_amendment_v1 import (
    source_bottom_first_to_top_first, ORIGIN_CONTRACT,
)
from tools.corpus.court_original_cat_v3_puppet_reference_parity_v0 import (
    compare_original_v3_source_against_puppet,
)


def _frame_diagnostics():
    return [
        {"pose_ordinal": (0, 14)[pi], "view_index": (0, 2, 4, 6)[vi],
         "source_arm_index": ai, "layer_overflow_pixels": 0,
         "exact_depth_ambiguity_pixels": 0}
        for pi in range(2) for vi in range(4) for ai in range(4)
    ]


def test_source_storage_origin_reversal_is_fixed_independent_of_pose_arm_view():
    raw = np.zeros((2, 4, 2, 4, 2, 3, 4), dtype=np.float32)
    raw[..., 0, :, 0] = 0.25
    raw[..., 1, :, 0] = 0.75
    raw[..., 3] = 1.0
    normalized = source_bottom_first_to_top_first(raw)
    assert normalized.shape == raw.shape
    assert np.allclose(normalized[..., 0, :, 0], 0.75)
    assert np.allclose(normalized[..., 1, :, 0], 0.25)
    assert np.array_equal(source_bottom_first_to_top_first(normalized), raw)
    assert np.array_equal(raw[..., 0, :, 0], np.full(raw[..., 0, :, 0].shape, 0.25))
    assert "BOTTOM_LEFT" in ORIGIN_CONTRACT and "CAA_TOP_LEFT" in ORIGIN_CONTRACT


def test_no_per_arm_or_pose_alignment_choice():
    pred = np.zeros((2, 4, 4, 2, 2, 4), np.float32)
    pred[..., 0, :, 0] = 0.8
    pred[..., 1, :, 0] = 0.2
    pred[..., 3] = 1.0
    source_top_first = np.repeat(pred[:, :, None], 2, axis=2)
    raw_original_bottom_first = source_top_first[..., ::-1, :, :].copy()
    direct = compare_original_v3_source_against_puppet(
        pred, raw_original_bottom_first, _frame_diagnostics())
    corrected = compare_original_v3_source_against_puppet(
        pred, source_bottom_first_to_top_first(raw_original_bottom_first),
        _frame_diagnostics())
    assert direct["total_source_sample_operation_pixel_mismatches"] > 0
    assert corrected["total_source_sample_operation_pixel_mismatches"] == 0
    assert corrected["physical_original_full_source_render_comparisons"] == 64


def test_wrong_input_axis_domain_is_rejected():
    with pytest.raises(ValueError, match="ARRAY_SHAPE_INVALID"):
        source_bottom_first_to_top_first(np.zeros((2, 4, 3)))
