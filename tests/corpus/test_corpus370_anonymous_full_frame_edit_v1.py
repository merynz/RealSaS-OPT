"""Numerical full-frame edit court anti-forgery; no Blender scene in CI."""
import copy
import numpy as np
import pytest

from tools.corpus.court_corpus370_anonymous_full_frame_edit_v1 import (
    ARMS,SCHEMA,evaluate,
)


def fixtures():
    resolution=128
    frames=np.zeros((2,len(ARMS),resolution,resolution,4),np.float32)
    frames[:,:,32:96,30:98,:3]=[.6,.3,.2]
    frames[:,:,32:96,30:98,3]=1.
    # Source original == full soup == no-op.
    # A is a source-wide patch in a square region, B a second independent
    # region; neither change requires a semantic category label.
    frames[:,3,35:60,35:60,:]=0
    frames[:,4,61:87,65:87,:]=0
    frames[:,5]=frames[:,3]
    frames[:,5,61:87,65:87,:]=0
    doc={
        "schema":"RealSaS.Corpus370AnonymousFullFrameEditorRender.v1",
        "status":"SOURCE_FULL_FRAME_REST_ALPHA_EDIT_CANDIDATE__INDEPENDENT_COURT_REQUIRED",
        "views":[0,4],"arms":list(ARMS),"teacher_only":True,
        "original_geometry_reimport_parity_verified":True,
        "full_frame_resolution":resolution,
        "downsampled_relative_to_original":True,
        "training_authorized":False,"graphics_owner_authorized":False,
        "model_training_authorized":False,
        "private_view_details":[
            {"view":0,"selected_slot_a_teacher_only":1,
             "selected_slot_b_teacher_only":2,
             "source_pixel_x":201,"source_pixel_y":260},
            {"view":4,"selected_slot_a_teacher_only":3,
             "selected_slot_b_teacher_only":4,
             "source_pixel_x":320,"source_pixel_y":360},
        ],
    }
    return frames,doc


def test_genuine_full_image_parity_and_separable_group_edits_are_measured():
    p,doc=fixtures()
    d=evaluate(p,doc)
    assert d["original_full_fov_pixels_measured"]==32768
    assert d["bounded_parity_and_separable_views"]==2
    assert d["per_view"][0]["source_wide_a_group_removal_pixel_count"]>10
    assert d["per_view"][0]["source_wide_b_conditional_after_a_group_removal_pixel_count"]>10
    assert d["graphics_owner_family_authorized"] is False
    assert d["training_authorized"] is False


def test_non_separable_hypothesis_is_abstain_not_same_owner_negative():
    p,doc=fixtures()
    doc["private_view_details"][1]["selected_slot_b_teacher_only"]=3
    d=evaluate(p,doc)
    assert d["bounded_parity_and_separable_views"]==1
    assert d["per_view"][1]["status"]=="ABSTAIN_SELECTED_GROUPS_NOT_SEPARABLE"
    assert d["different_drawable_negative_pairs"]==0


def test_wide_area_source_to_triangle_soup_appearance_drift_abstains():
    p,doc=fixtures()
    p[0,1,30:90,30:90,0]+=.3
    d=evaluate(p,doc)
    assert d["per_view"][0]["source_to_soup_bounded_parity_pass"] is False
    assert d["bounded_parity_and_separable_views"]==1
    assert d["appearance_caa_total_authority_minted"] is False


def test_noop_drift_abstains_even_when_source_parity_perfect():
    p,doc=fixtures()
    p[0,2,40,42,1]+=.05
    d=evaluate(p,doc)
    assert d["per_view"][0]["no_op_rerender_parity_pass"] is False
    assert d["bounded_parity_and_separable_views"]==1


def test_noop_hidden_background_rgb_not_a_color_consequence():
    p,doc=fixtures()
    p[0,3,0,0,:3]=[10.,10.,10.]
    d=evaluate(p,doc)
    assert d["per_view"][0]["source_wide_a_group_removal_pixel_count"]>10
    assert d["training_authorized"] is False


def test_bad_source_or_training_authority_rejected():
    p,doc=fixtures()
    doc["training_authorized"]=True
    with pytest.raises(ValueError,match="PRODUCER_SCOPE_OR_AUTHORITY"):
        evaluate(p,doc)
    doc["training_authorized"]=False
    doc["arms"][0]="FAKE_SOURCE"
    with pytest.raises(ValueError,match="PRODUCER_SCOPE_OR_AUTHORITY"):
        evaluate(p,doc)


def test_nonfinite_or_invalid_alpha_rejected():
    p,doc=fixtures()
    p[1,3,2,2,0]=float("nan")
    with pytest.raises(ValueError,match="RGBA_ARRAY"):
        evaluate(p,doc)
    p,doc=fixtures()
    p[0,0,2,2,3]=1.5
    with pytest.raises(ValueError,match="RGBA_ARRAY"):
        evaluate(p,doc)


def test_missing_view_or_geometry_parity_rejected():
    p,doc=fixtures()
    doc["private_view_details"]=doc["private_view_details"][:1]
    with pytest.raises(ValueError,match="VIEW_SOURCE_PIXEL_SCOPE"):
        evaluate(p,doc)
    p,doc=fixtures()
    doc["original_geometry_reimport_parity_verified"]=False
    with pytest.raises(ValueError,match="PRODUCER_SCOPE_OR_AUTHORITY"):
        evaluate(p,doc)


def test_never_claim_native_animation_or_2d_puppet_owner_admission():
    p,doc=fixtures()
    d=evaluate(p,doc)
    assert d["animation_replay_authorized"] is False
    assert d["product_compilation_authorized"] is False
    assert d["same_drawable_positive_pairs"]==0
    assert d["different_drawable_negative_pairs"]==0
