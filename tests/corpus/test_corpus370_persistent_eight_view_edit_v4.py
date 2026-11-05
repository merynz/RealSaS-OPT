"""Eight-view original-source FOV and persistent anonymous slot regression.

No Blender source in synthetic CI; real Blender result is separately required.
"""
import copy
import numpy as np
import pytest

from tools.corpus.court_corpus370_persistent_eight_view_edit_v4 import (
    ARMS,evaluate,
)


def fixture():
    n=128
    p=np.zeros((8, len(ARMS), n, n, 4), np.float32)
    p[:, :, 24:104, 20:108, :3]=[.55,.4,.25]
    p[:, :, 24:104, 20:108, 3]=1.
    p[:,3,30:55,24:61,:]=0
    p[:,4,65:91,70:96,:]=0
    p[:,5]=p[:,3]
    p[:,5,65:91,70:96,:]=0
    pairs=[{"view":i,
            "persistent_pair_anchor_view":4,
            "source_pixel_x":256,"source_pixel_y":512,
            "selected_slot_a_teacher_only":6,
            "selected_slot_b_teacher_only":3} for i in range(8)]
    r={
        "schema":"RealSaS.Corpus370AnonymousFullFrameEditorRender.v4",
        "status":"SOURCE_FULL_FRAME_REST_ALPHA_EDIT_CANDIDATE__INDEPENDENT_COURT_REQUIRED",
        "original_geometry_reimport_parity_verified":True,
        "views":list(range(8)),
        "persistent_pair_anchor_view":4,
        "persistent_selected_slot_ids_teacher_only":[6,3],
        "arms":list(ARMS),
        "teacher_only":True,
        "training_authorized":False,
        "graphics_owner_authorized":False,
        "model_training_authorized":False,
        "full_frame_resolution":n,
        "downsampled_relative_to_original":True,
        "private_view_details":pairs,
    }
    return p,r


def test_persistent_layer_editability_eight_views_full_field():
    rgba,doc=fixture()
    out=evaluate(rgba,doc)
    assert out["original_full_fov_pixels_measured"]==8*128*128
    assert out["bounded_parity_and_separable_views"]==8
    assert out["persistent_group_pair_across_all_views"] is True
    assert out["views_with_visible_a_edit"]==8
    assert out["views_with_visible_b_after_a"]==8
    assert out["independently_measured_view4_group_b_reveal"]>0
    assert out["training_authorized"] is False
    assert out["graphics_owner_family_authorized"] is False
    assert out["same_drawable_positive_pairs"]==0
    assert out["different_drawable_negative_pairs"]==0


def test_occluded_or_offscreen_group_in_one_view_is_not_owner_negative():
    rgba,doc=fixture()
    rgba[7,3]=rgba[7,1]
    rgba[7,4]=rgba[7,1]
    rgba[7,5]=rgba[7,1]
    out=evaluate(rgba,doc)
    assert out["bounded_parity_and_separable_views"]==8
    assert out["views_with_visible_a_edit"]==7
    assert out["views_with_visible_b_after_a"]==7
    assert out["different_drawable_negative_pairs"]==0


def test_view4_must_observe_independent_conditional_hidden_edit_for_admission():
    rgba,doc=fixture()
    rgba[4,5]=rgba[4,3]
    decision=evaluate(rgba,doc)
    assert decision["independently_measured_view4_group_b_reveal"]==0
    assert decision["appearance_caa_total_authority_minted"] is False
    assert decision["training_authorized"] is False


def test_layer_slot_identity_cannot_permute_across_cameras():
    rgba,doc=fixture()
    doc["private_view_details"][6]["selected_slot_b_teacher_only"]=9
    with pytest.raises(ValueError,match="PERSISTENT_SLOT_IDENTITY"):
        evaluate(rgba,doc)


def test_forged_training_or_owner_evidence_rejected():
    rgba,doc=fixture()
    doc["training_authorized"]=True
    with pytest.raises(ValueError,match="PRODUCER_SCOPE_OR_AUTHORITY"):
        evaluate(rgba,doc)
    rgba,doc=fixture()
    doc["persistent_selected_slot_ids_teacher_only"]=[3,3]
    with pytest.raises(ValueError,match="PERSISTENT_SLOT_IDENTITY"):
        evaluate(rgba,doc)


def test_source_reconstruction_drift_rejects_one_view_not_hiding_in_p99():
    rgba,doc=fixture()
    rgba[3,1,23:100,22:98,0]+=.35
    decision=evaluate(rgba,doc)
    assert decision["bounded_parity_and_separable_views"]==7
    assert decision["per_view"][3]["source_to_soup_bounded_parity_pass"] is False


def test_noop_court_flags_one_camera_even_when_others_pass():
    rgba,doc=fixture()
    rgba[5,2,40,42,1]+=.2
    d=evaluate(rgba,doc)
    assert d["bounded_parity_and_separable_views"]==7


def test_empty_source_hit_is_not_a_negative_example():
    rgba,doc=fixture()
    rgba[0,3]=rgba[0,1]
    rgba[0,4]=rgba[0,1]
    rgba[0,5]=rgba[0,1]
    d=evaluate(rgba,doc)
    assert d["views_with_visible_a_edit"]==7
    assert d["different_drawable_negative_pairs"]==0
    assert d["training_authorized"] is False


def test_wrong_raster_shape_and_nonfinite_refuse_release():
    rgba,doc=fixture()
    with pytest.raises(ValueError,match="RGBA_ARRAY"):
        evaluate(rgba[:7],doc)
    rgba,doc=fixture()
    rgba[6,3,20,20,1]=float("nan")
    with pytest.raises(ValueError,match="RGBA_ARRAY"):
        evaluate(rgba,doc)
