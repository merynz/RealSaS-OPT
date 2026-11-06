"""V18m source depth ordering requires no FULL RGB oracle.

These are controlled synthetic tests. Real-source Blender SHA court is separate.
"""
import numpy as np
import pytest

from tools.corpus.court_corpus370_v18m_depth_ordered_source_compositor import (
    predict_source_only,measure_view,SCHEMA,STATUS,TOL,DEPTH_TIE_EPS,
)

RED=(.9,.08,.035)
BLUE=(.04,.14,.92)


def sample_data():
    a=np.zeros((8,3,192,192,4),dtype=np.float32)
    d=np.zeros((8,2,192,192,4),dtype=np.float32)
    # Both layers physically present, visibility flips by source ray:
    # a fixed 2D drawable order cannot produce the alternating scene colors.
    for i in range(8):
        is_g=i%2==0
        a[i,0,30,40]=(*(RED if is_g else BLUE),1.)
        a[i,1,30,40]=(*RED,1.)
        a[i,2,30,40]=(*BLUE,1.)
        d[i,0,30,40]=(.20 if is_g else .60,0.,0.,1.)
        d[i,1,30,40]=(.60 if is_g else .20,0.,0.,1.)
    return a,d


def test_oracle_free_depth_reconstructs_opposing_source_nearest_from_two_layers():
    a,d=sample_data()
    q=predict_source_only(a[:,1:],d)
    np.testing.assert_allclose(q["predicted_full_premultiplied_samples"],
                               a[:,0],atol=2e-8,rtol=0)
    assert np.count_nonzero(q["both_physically_present_mask"])==8
    assert np.count_nonzero(q["unknown_depth_tie_mask"])==0
    assert np.count_nonzero(q["predicted_g_frontmost_mask"])==4
    assert np.count_nonzero(q["predicted_r_frontmost_mask"])==4
    report=measure_view(a,d)
    assert report["original_source_sample_color_failures_on_qualified"]==0
    assert report["original_source_oracle_free_depth_order_predicted_failures"]==0
    assert report["source_original_blender_fixed_samplefirst_failures_full_domain"]==1
    assert report["source_original_blender_postfilter_failures_full_domain"]==1
    assert report["source_fixed_samplefirst_failures_recovered_by_depth_sort"]==1
    assert report["source_depth_predictor_regresses_previously_fixed_qualified_pixels"]==0
    assert report["full_target_input_used_to_select_frontmost_surface"] is False


def test_full_target_tampering_cannot_change_source_depth_prediction():
    a,d=sample_data()
    out=predict_source_only(a[:,1:],d)["predicted_full_premultiplied_samples"]
    a[:,0,:,:,:]=0.
    out2=predict_source_only(a[:,1:],d)["predicted_full_premultiplied_samples"]
    np.testing.assert_array_equal(out,out2)
    # The independent scoring court DOES detect wrong FULL source targets.
    result=measure_view(a,d)
    assert result["original_source_sample_color_failures_on_qualified"]>0


def test_wrong_depth_order_must_fail_independent_full_source_court():
    a,d=sample_data()
    d[:,0,30,40,0],d[:,1,30,40,0]=d[:,1,30,40,0].copy(),d[:,0,30,40,0].copy()
    report=measure_view(a,d)
    assert report["original_source_oracle_free_depth_order_predicted_failures"]==1
    assert report["original_source_sample_color_failures_on_qualified"]==8
    assert report["source_depth_predictor_regresses_previously_fixed_qualified_pixels"]==0


def test_source_depth_near_tie_abstains_not_fake_positive_or_negative():
    a,d=sample_data()
    d[0,0,30,40,0]=.4500
    d[0,1,30,40,0]=.4501
    p=predict_source_only(a[:,1:],d)
    assert p["unknown_depth_tie_mask"][0,30,40]
    assert not p["sample_valid_mask"][0,30,40]
    report=measure_view(a,d)
    assert report["original_source_depth_tie_abstained_samples"]==1
    assert report["source_pixels_unknown_any_sample"]==1
    assert report["source_all_8_sample_qualified_pixels"]==192*192-1
    assert report["original_source_oracle_free_depth_order_predicted_failures"]==0


def test_source_depth_sampling_alpha_mismatch_fails_closed():
    a,d=sample_data()
    d[0,0,30,40,3]=0.
    with pytest.raises(ValueError,match="V18M_PREDICTOR_DIFFERENT_CYCLES_RAY"):
        predict_source_only(a[:,1:],d)


def test_depth_rgba_outside_original_geometry_normalization_is_unknown():
    a,d=sample_data()
    d[0,0,30,40,0]=2.
    p=predict_source_only(a[:,1:],d)
    assert p["invalid_depth_mask"][0,30,40]
    assert not p["sample_valid_mask"][0,30,40]


def test_simple_nonoverlap_depth_prediction_matches_single_source_layer():
    a,d=sample_data()
    a[:,2,30,40,:]=0.
    d[:,1,30,40,:]=0.
    a[:,0,30,40,:]=a[:,1,30,40,:]
    p=predict_source_only(a[:,1:],d)
    assert not np.any(p["unknown_depth_tie_mask"])
    np.testing.assert_array_equal(p["predicted_full_premultiplied_samples"],a[:,0])
    x=measure_view(a,d)
    assert x["original_source_oracle_free_depth_order_predicted_failures"]==0


@pytest.mark.parametrize("bad",("nan","mismatched_dimensions","nonbinary_1spp_alpha"))
def test_invalid_per_sample_depth_evidence_refused(bad):
    a,d=sample_data()
    if bad=="nan":d[0,1,30,40,0]=float("nan")
    elif bad=="mismatched_dimensions":d=d[:3]
    else:
        a[0,1,30,40,3]=.4
        d[0,0,30,40,3]=.4
    with pytest.raises(ValueError,match="V18M_PREDICTOR_"):
        predict_source_only(a[:,1:],d)


def test_schema_is_measurement_without_artist_or_model_authority():
    assert SCHEMA.endswith(".v18m")
    assert "DRAWABLE_UNKNOWN" in STATUS
    assert 0<DEPTH_TIE_EPS<.001
    assert TOL==.025
