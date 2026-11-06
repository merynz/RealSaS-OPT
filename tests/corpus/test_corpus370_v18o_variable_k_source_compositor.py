"""Adversarial variable-K real-opacity source compositor math only, NOT artist truth."""
import numpy as np
import pytest

from tools.corpus.court_corpus370_v18o_variable_k_source_compositor import (
    k_compose,independently_score_against_full,TIE_EPS,TOL,MAX_SOURCE_GROUPS,
)


def opaque_layers(k=3):
    a=np.zeros((8,k,2,3,4),np.float64)
    d=np.zeros_like(a)
    colors=[(.9,.1,.1),(.1,.9,.1),(.1,.1,.9)]
    alphas=[.50,.40,.80]
    for sample in range(8):
        # Sample-specific ray crossings: 0<1<2 vs 2<0<1
        depths=(.2,.4,.6) if sample%2==0 else (.4,.6,.2)
        for layer in range(3):
            a[sample,layer,0,0,:3]=colors[layer]
            a[sample,layer,0,0,3]=alphas[layer]
            d[sample,layer,0,0]=[depths[layer],0.,0.,1.]
    return a,d


def over(x,y):
    return x+y*(1.-x[3])


def expected_source_full(a,d):
    source=np.zeros((8,2,3,4),np.float64)
    for i in range(8):
        z=d[i,:,0,0,0]
        for k in np.argsort(z)[::-1]:
            row=a[i,k,0,0]
            pre=np.r_[row[:3]*row[3],row[3]]
            source[i,0,0]=over(pre,source[i,0,0])
    straight=source.copy()
    mask=straight[...,3]>1e-9
    straight[...,:3][mask]/=straight[...,3,None][mask]
    return straight


def test_three_actual_translucent_source_surfaces_are_depth_sorted_and_compose():
    a,d=opaque_layers()
    p=k_compose(a,d)
    assert p["source_K"]==3 and p["source_samples"]==8
    assert np.count_nonzero(p["source_at_least_three_geometry_hits_mask"])==8
    assert np.count_nonzero(p["source_any_nonbinary_material_alpha_mask"])==8
    assert not np.any(p["source_depth_tie_mask"])
    assert p["source_diagnostic_original_full_used_for_sorting"] is False
    original=expected_source_full(a,d)
    q=independently_score_against_full(p,original)
    assert q["qualified_source_sample_rgba_failures"]==0
    assert q["qualified_source_8sample_pixel_rgba_failures"]==0
    assert q["total_source_sample_pixel_instances"]==8*2*3


def test_anonymous_component_permutation_does_not_change_pixels():
    a,d=opaque_layers()
    x=k_compose(a,d)
    for perm in ([2,0,1],[1,2,0]):
        p=k_compose(a[:,perm],d[:,perm])
        np.testing.assert_allclose(x["premultiplied_sample_rgba"],
                                   p["premultiplied_sample_rgba"],rtol=0,atol=1e-13)
        np.testing.assert_array_equal(x["qualified_sample_mask"],
                                      p["qualified_sample_mask"])


def test_more_than_three_components_and_invisible_group_are_allowed():
    a,d=opaque_layers(5)
    a[:,3:,0,0]=0.
    d[:,3:,0,0]=0.
    p=k_compose(a,d)
    assert p["source_K"]==5
    assert p["source_at_least_three_geometry_hits_mask"].sum()==8


def test_material_alpha_zero_still_has_geometric_hit_and_reveal():
    a,d=opaque_layers()
    a[:,0,0,0,3]=0.
    p=k_compose(a,d)
    assert p["source_at_least_three_geometry_hits_mask"].sum()==8
    original=expected_source_full(a,d)
    assert independently_score_against_full(p,original)[
        "qualified_source_sample_rgba_failures"]==0


def test_original_full_target_is_not_predictor_input():
    a,d=opaque_layers()
    x=k_compose(a,d)
    true=expected_source_full(a,d)
    corrupted=true.copy()
    corrupted[:,0,0,:3]=0.
    assert independently_score_against_full(x,true)[
        "qualified_source_sample_rgba_failures"]==0
    assert independently_score_against_full(x,corrupted)[
        "qualified_source_sample_rgba_failures"]==8


def test_depth_tie_is_unknown_and_cannot_spoof_success():
    a,d=opaque_layers()
    d[0,0,0,0,0]=d[0,1,0,0,0]+1e-5
    p=k_compose(a,d)
    assert p["source_depth_tie_mask"][0,0,0]
    assert not p["qualified_sample_mask"][0,0,0]
    q=independently_score_against_full(p,expected_source_full(a,d))
    assert q["unknown_source_sample_pixel_instances"]==1
    assert q["unknown_source_all_8_sample_pixels"]==1
    assert q["qualified_source_all_8_sample_pixels"]==5


@pytest.mark.parametrize("broken",(
    "nan","source_without_depth","depth_not_binary","bad_alpha",
    "k2","k65","wrong_sample_count","depth_out_of_range",
))
def test_bad_measured_k_source_fails_or_marks_unknown(broken):
    a,d=opaque_layers()
    if broken=="nan":a[0,0,0,0,0]=np.nan
    elif broken=="source_without_depth":d[0,0,0,0,:]=0.
    elif broken=="depth_not_binary":d[0,0,0,0,3]=.5
    elif broken=="bad_alpha":a[0,0,0,0,3]=1.5
    elif broken=="k2":a,d=a[:,:2],d[:,:2]
    elif broken=="k65":
        a=np.zeros((8,MAX_SOURCE_GROUPS+1,2,3,4))
        d=np.zeros_like(a)
    elif broken=="wrong_sample_count":a,d=a[:4],d[:4]
    else:d[0,0,0,0,0]=.0001
    if broken=="depth_out_of_range":
        p=k_compose(a,d)
        assert p["source_invalid_depth_mask"][0,0,0]
        assert not p["qualified_sample_mask"][0,0,0]
    else:
        with pytest.raises(ValueError,match="V18O_K_"):
            k_compose(a,d)


def test_proof_contract_no_artist_label_authorization():
    a,d=opaque_layers()
    x=k_compose(a,d)
    assert TIE_EPS==.0002 and TOL==.025
    assert x["source_group_ids_are_anonymous_only"] is True
    assert all(x[k] is False for k in (
        "artist_graphics_owner_authorized","iris_training_authorized",
        "tessa_training_authorized","materialization_authorized",
        "training_authorized"))
