"""V18l nonvacuity false-positive and paired pixel-order canaries."""
import numpy as np
import pytest
from tools.corpus.court_corpus370_v18l_source_sample_diversity import compare,SCHEMA,PASS

RED=(.9,.08,.035)
BLUE=(.04,.14,.92)


def frame():
    return np.zeros((8,3,192,192,4),dtype=np.float32)


def test_sample_index_variability_and_real_postfilter_failure():
    a=frame()
    for k in range(8):
        if k<4:
            a[k,0,11,12]=(*RED,1.)
            a[k,1,11,12]=(*RED,1.)
        else:
            a[k,0,11,12]=(*BLUE,1.)
            a[k,2,11,12]=(*BLUE,1.)
    z=compare(a)
    assert z["source_original_full_nonvacuous_different_subsets_pixel_count"]==1
    assert z["real_cycles_subset_index_variability_observed"]
    assert z["source_8_subset_prefilter_order_fail_pixels"]==0
    assert z["source_8_subset_postfilter_order_fail_pixels"]==1
    assert z["source_8_subset_postfilter_fail_but_prefilter_pass_pixels"]==1
    assert z["source_8_subset_prefilter_improves_error_pixels"]==1
    assert z["source_8_subset_prefilter_worsens_error_pixels"]==0
    assert z["source_render_same_index_sample_oracle_fail_instances"]==0


def test_constant_identical_samples_are_vacuous_even_when_all_rgba_pass():
    a=frame()
    a[:,0,11,12]=(*RED,1.)
    a[:,1,11,12]=(*RED,1.)
    z=compare(a)
    assert not z["real_cycles_subset_index_variability_observed"]
    assert z["source_original_full_nonvacuous_different_subsets_pixel_count"]==0
    assert z["source_8_subset_postfilter_order_fail_pixels"]==0
    assert z["source_8_subset_prefilter_order_fail_pixels"]==0


def test_overlapping_front_sides_that_change_across_samples_deny_fixed_order():
    a=frame()
    for k in range(8):
        a[k,0,11,12]=(*(RED if k%2==0 else BLUE),1.)
        a[k,1,11,12]=(*RED,1.)
        a[k,2,11,12]=(*BLUE,1.)
    z=compare(a)
    assert z["source_original_full_nonvacuous_different_subsets_pixel_count"]==1
    assert z["source_8_subset_prefilter_order_fail_pixels"]==1
    assert z["source_8_subset_postfilter_order_fail_pixels"]==1
    assert z["source_render_same_index_sample_oracle_fail_instances"]==0


def test_corrupt_alpha_not_a_valid_source_sample():
    a=frame()
    a[0,0,11,12,3]=1.7
    with pytest.raises(ValueError,match="V18L_DIVERSITY_SOURCE_SUBSET_RGBA_SHAPE"):
        compare(a)


def test_record_is_evidence_not_owner():
    assert SCHEMA.endswith(".v18l")
    assert "DRAWABLE_UNKNOWN" in PASS
