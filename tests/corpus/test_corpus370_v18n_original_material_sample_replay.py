"""V18n material/UV policy source renderer math without artist ownership claims."""
import numpy as np
import pytest

from tools.corpus.court_corpus370_v18n_original_material_sample_replay import (
    SCHEMA,STATUS,TOL,predict_material_from_source_isolates,summary_view,
)

RED=(.90,.08,.035)
BLUE=(.04,.14,.92)


def fixture():
    original=np.zeros((8,4,192,192,4),dtype=np.float32)
    depth=np.zeros((8,2,192,192,4),dtype=np.float32)
    for i in range(8):
        group_first=(i%2==0)
        # G translucent; R opaque. Source sample changes frontmost side.
        g=np.array([*RED,.5],np.float32)
        rr=np.array([*BLUE,1.],np.float32)
        gp=np.array([*RED,.5],np.float32)
        # FULL native/soup is straight RGBA consistent with depth-ordered OVER.
        composed=np.array([*((np.array(RED)*.5+
                     np.array(BLUE)*.5) if group_first else BLUE),1.],np.float32)
        original[i,0,20,30]=composed
        original[i,1,20,30]=composed
        original[i,2,20,30]=g
        original[i,3,20,30]=rr
        depth[i,0,20,30]=(.20 if group_first else .60,0.,0.,1.)
        depth[i,1,20,30]=(.60 if group_first else .20,0.,0.,1.)
    return original,depth


def test_no_full_leak_and_partial_alpha_depth_compositing():
    source,depth=fixture()
    x=predict_material_from_source_isolates(source[:,2:],depth)
    native=np.asarray(source[:,0],np.float64)
    predicted=x["predicted_sample_premultiplied_rgba"]
    np.testing.assert_allclose(predicted,native,atol=1e-6,rtol=0)
    assert x["source_compositor_used_full_or_native_to_decide_frontmost"] is False
    assert np.count_nonzero(x["source_both_geometry_hit_mask"])==8
    assert np.count_nonzero(x["source_material_nonbinary_alpha_mask"])==8
    result=summary_view(source,depth)
    assert result["original_source_material_native_vs_soup_sample_failures_full_domain"]==0
    assert result["original_source_material_depth_compositor_vs_soup_sample_failures_qualified"]==0
    assert result["source_material_fixed_samplefirst_vs_soup_8sample_pixel_failures"]==0
    assert result["source_material_depth_compositor_vs_soup_8sample_pixel_failures_qualified"]==0
    assert result["source_material_postfilter_vs_soup_8sample_pixel_failures"]==1


def test_soup_fidelity_failure_separate_from_sample_depth_prediction():
    a,d=fixture()
    a[:,0,20,30,:3]=0.
    out=summary_view(a,d)
    assert out["original_source_material_native_vs_soup_sample_failures_full_domain"]==8
    assert out["original_source_material_depth_compositor_vs_soup_sample_failures_qualified"]==0
    assert out["original_source_material_depth_compositor_vs_native_sample_failures_qualified"]==8


def test_target_native_full_cannot_change_depth_prediction():
    a,d=fixture()
    x=predict_material_from_source_isolates(a[:,2:],d)[
        "predicted_sample_premultiplied_rgba"].copy()
    a[:,0]=0
    a[:,1]=0
    y=predict_material_from_source_isolates(a[:,2:],d)[
        "predicted_sample_premultiplied_rgba"].copy()
    np.testing.assert_array_equal(x,y)
    report=summary_view(a,d)
    assert report["original_source_material_depth_compositor_vs_native_sample_failures_qualified"]==8


def test_material_alpha_zero_does_not_make_geometric_face_absent():
    a,d=fixture()
    a[:,2,20,30,:]=0
    a[:,0,20,30]=(*BLUE,1.)
    a[:,1,20,30]=(*BLUE,1.)
    pred=predict_material_from_source_isolates(a[:,2:],d)
    assert np.count_nonzero(pred["source_both_geometry_hit_mask"])==8
    assert np.count_nonzero(pred["source_geometry_hit_but_source_alpha_zero_mask"])==8
    np.testing.assert_allclose(pred["predicted_sample_premultiplied_rgba"],
                               a[:,1],atol=1e-6)


def test_unmatched_material_visibility_without_geometry_fails_closed():
    a,d=fixture()
    d[0,0,20,30]=0.
    with pytest.raises(ValueError,
                       match="V18N_SOURCE_MATERIAL_VISIBLE_WHERE_GEOMETRY_NOT_HIT"):
        predict_material_from_source_isolates(a[:,2:],d)


def test_ambiguous_source_depth_abstains_both_rgba_and_pixel():
    a,d=fixture()
    d[0,0,20,30,0]=.4
    d[0,1,20,30,0]=.4001
    p=predict_material_from_source_isolates(a[:,2:],d)
    assert p["source_geometry_depth_tie_mask"][0,20,30]
    assert not p["source_valid_sample_mask"][0,20,30]
    result=summary_view(a,d)
    assert result["source_unknown_tied_sample_instances"]==1
    assert result["source_8_sample_unknown_pixels"]==1
    assert result["source_material_depth_compositor_vs_soup_8sample_pixel_failures_qualified"]==0


@pytest.mark.parametrize("failure",("nan","wrong_shape","depth_alpha","outside_depth","bad_material_alpha"))
def test_material_evidence_fail_or_unknown(failure):
    a,d=fixture()
    if failure=="nan":a[0,2,20,30,0]=np.nan
    elif failure=="wrong_shape":d=d[:4]
    elif failure=="depth_alpha":d[0,0,20,30,3]=.5
    elif failure=="outside_depth":d[0,0,20,30,0]=2.
    else:a[0,2,20,30,3]=1.5
    if failure=="outside_depth":
        x=predict_material_from_source_isolates(a[:,2:],d)
        assert x["source_invalid_depth_mask"][0,20,30]
        assert not x["source_valid_sample_mask"][0,20,30]
    else:
        with pytest.raises(ValueError,match="V18N_SOURCE_"):
            predict_material_from_source_isolates(a[:,2:],d)


def test_status_does_not_mint_artist_or_student_authority():
    assert SCHEMA.endswith(".v18n")
    assert STATUS.endswith("ARTIST_OWNER_UNKNOWN")
    assert TOL==.025
