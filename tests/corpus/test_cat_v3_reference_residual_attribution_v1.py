"""Residual decomposition tests with synthetic data, no Cat target pixels."""
import numpy as np
import pytest
from tools.corpus.court_cat_v3_reference_residual_attribution_v1 import classify_residuals_v1


def fixture():
    p=np.zeros((2,4,4,2,2,4),np.float32)
    p[...,:3]=0.4
    p[...,3]=1
    source=np.repeat(p[:,:,None],2,axis=2)
    return p,source[...,::-1,:,:].copy()


def test_zero_residual_attribution_and_valid_lower_bound():
    p,s=fixture()
    r=classify_residuals_v1(p,s)
    assert r["total_physical_reference_mismatches"]==0
    assert r["alpha_or_coverage_mismatched_pixel_samples"]==0
    assert r["opaque_interior_rgb_mismatched_pixel_samples"]==0
    assert r["deterministic_prediction_mandatory_miss_lower_bound_pixel_pairs"]==0
    assert r["source_artist_ownership"]=="UNKNOWN"
    assert not r["training_authorized"]


def test_two_offsets_more_than_two_tol_force_a_deterministic_miss():
    p,s=fixture()
    # In raw bottom-first source buffer, edit one source sample 0.12 away.
    s[0,0,1,0,0,0,0]+=0.12
    r=classify_residuals_v1(p,s)
    assert r["deterministic_prediction_mandatory_miss_lower_bound_pixel_pairs"]==1
    assert r["source_two_offset_pairs_difference_above_tolerance"]==1
    assert r["total_physical_reference_mismatches"]==1
    assert r["opaque_interior_rgb_mismatched_pixel_samples"]==1
    assert r["alpha_or_coverage_mismatched_pixel_samples"]==0


def test_alpha_mismatch_class_is_not_double_counted_as_rgb_error():
    p,s=fixture()
    s[0,0,0,0,0,0,3]=0.5
    r=classify_residuals_v1(p,s)
    assert r["total_physical_reference_mismatches"]==1
    assert r["alpha_or_coverage_mismatched_pixel_samples"]==1
    assert r["opaque_interior_rgb_mismatched_pixel_samples"]==0
    assert r["unclassified_mismatched_pixel_samples"]==0


def test_independent_edit_consequence_mismatch_is_not_base_error():
    p,s=fixture()
    # Original edit arm has a color effect the reference did not compile.
    s[0,0,0,1,0,0,0]+=0.1
    r=classify_residuals_v1(p,s)
    assert r["per_arm"]["SINGLE_0"]["mismatched_pixel_samples"]==1
    assert r["per_arm"]["BASE"]["mismatched_pixel_samples"]==0
    assert r["edit_delta_consequence"]["SINGLE_0"][
        "mismatched_source_edit_effect_sample_pixels"]==1


def test_retroactive_tolerance_tuning_rejected():
    p,s=fixture()
    with pytest.raises(ValueError,match="TOLERANCE_DRIFT"):
        classify_residuals_v1(p,s,tolerance=0.1)
