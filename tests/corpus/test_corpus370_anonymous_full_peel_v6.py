"""Fail-closed synthetic contracts for full eight-view anonymous slot census.

Synthetic arrays exercise the court interface; never imply that source art,
drawable ownership, CAA, native motion or model targets were independently proved.
"""
import copy
import numpy as np
import pytest

from tools.corpus.court_corpus370_anonymous_full_peel_v6 import (
    build_arms, evaluate,
)


def fixture():
    slots=list(range(8))
    arms=list(build_arms(slots))
    p=np.zeros((8,len(arms),128,128,4),dtype=np.float32)
    p[:,:,:, :, :3]=(.3,.6,.4)
    p[:,:,:, :, 3]=1.
    # Eight disjoint finite visible patches; every one is independently removable.
    for slot in slots:
        x=8+slot%4*28
        y=8+slot//4*50
        p[:,3+slot,y:y+22,x:x+20,:]=0.
    # Full-peel isolates one source group by suppressing ALL the others.
    # The all-clear base has no group alpha; its per-slot contrast must be
    # measured separately from the direct-removal contrast.
    p[:,-2]=0.
    for slot in slots:
        x=8+slot%4*28
        y=8+slot//4*50
        p[:,3+len(slots)+slot]=0.
        p[:,3+len(slots)+slot,y:y+22,x:x+20,:]=p[:,1,y:y+22,x:x+20,:]
    p[:,-1]=p[:,3+0]
    p[:,-1,8:30,36:56,:]=0.  # Remove group 1 after group 0.
    rows=[{
        "view":i, "persistent_pair_anchor_view":4,
        "source_pixel_x":256, "source_pixel_y":512,
        "selected_slot_a_teacher_only":0,
        "selected_slot_b_teacher_only":1,
    } for i in range(8)]
    manifest={
        "schema":"RealSaS.Corpus370AnonymousFullPeelEditorRender.v6",
        "status":"SOURCE_FULL_PEEL_REST_ALPHA_EDIT_CANDIDATE__INDEPENDENT_COURT_REQUIRED",
        "original_geometry_reimport_parity_verified":True,
        "views":list(range(8)), "persistent_pair_anchor_view":4,
        "persistent_selected_slot_ids_teacher_only":[0,1],
        "source_slot_ids_teacher_only":slots,
        "all_source_group_slots_rendered":True,
        "all_source_group_slots_isolated":True,
        "source_group_count":len(slots),
        "arms":arms, "teacher_only":True,
        "training_authorized":False,
        "graphics_owner_authorized":False,
        "model_training_authorized":False,
        "full_frame_resolution":128,
        "downsampled_relative_to_original":True,
        "private_view_details":rows,
    }
    return p,manifest


def test_all_slots_and_all_views_are_independently_courted():
    pixels,manifest=fixture()
    c=evaluate(pixels,manifest)
    assert c["all_source_slot_count"]==8
    assert c["all_source_slots_courted_in_all_views"] is True
    assert c["slot_direct_effect_any_view_count"]==8
    assert c["slot_isolation_effect_any_view_count"]==8
    assert c["directly_unobserved_but_isolated_visible_ids_teacher_only"]==[]
    assert c["unobservable_even_in_full_peel_ids_teacher_only"]==[]
    assert c["slot_direct_effect_unobserved_ids_teacher_only"]==[]
    assert len(c["slot_observability_teacher_only"])==8
    assert all(x["status"]=="OBSERVED_DIRECT_REMOVAL_PIXEL_EFFECT__NOT_DRAWABLE_IDENTITY"
               and x["same_drawable"]=="UNKNOWN"
               and x["different_drawable"]=="UNKNOWN"
               and x["training_authorized"] is False
               for x in c["slot_observability_teacher_only"])
    assert c["bounded_parity_and_separable_views"]==8
    assert c["independently_measured_view4_group_b_reveal"]>0
    assert c["original_full_fov_pixels_measured"]==8*128*128
    assert c["graphics_owner_family_authorized"] is False
    assert c["training_authorized"] is False
    assert c["same_drawable_positive_pairs"]==0
    assert c["different_drawable_negative_pairs"]==0


def test_hidden_or_unseen_group_is_abstain_not_different_drawable():
    p,m=fixture()
    p[:,3+7]=p[:,1]
    c=evaluate(p,m)
    assert c["slot_direct_effect_any_view_count"]==7
    assert c["slot_direct_effect_unobserved_ids_teacher_only"]==[7]
    assert c["directly_unobserved_but_isolated_visible_ids_teacher_only"]==[7]
    assert c["slot_isolation_effect_any_view_count"]==8
    assert c["slot_observability_teacher_only"][7]["status"]=="ABSTAIN_NO_DIRECT_EFFECT_IN_EIGHT_REST_CAMERAS"
    assert c["slot_observability_teacher_only"][7]["same_drawable"]=="UNKNOWN"
    assert c["slot_observability_teacher_only"][7]["different_drawable"]=="UNKNOWN"
    assert c["unobserved_group_is_owner_negative"] is False
    assert c["different_drawable_negative_pairs"]==0


@pytest.mark.parametrize("corruption",[
    "arms_swap","duplicate_ids","noncontiguous_ids","missing_all_slots_flag",
    "missing_all_isolation_flag",
    "forged_training","wrong_teacher_scope","wrong_resolution","view_slot_swap",
    "empty_anchor_pair",
])
def test_invalid_producer_claims_or_census_rejected(corruption):
    p,m=fixture()
    if corruption=="arms_swap":
        m["arms"][4],m["arms"][5]=m["arms"][5],m["arms"][4]
    elif corruption=="duplicate_ids":
        m["source_slot_ids_teacher_only"][6]=5
    elif corruption=="noncontiguous_ids":
        m["source_slot_ids_teacher_only"][-1]=9
    elif corruption=="missing_all_slots_flag":
        m["all_source_group_slots_rendered"]=False
    elif corruption=="missing_all_isolation_flag":
        m["all_source_group_slots_isolated"]=False
    elif corruption=="forged_training":
        m["training_authorized"]=True
    elif corruption=="wrong_teacher_scope":
        m["teacher_only"]=False
    elif corruption=="wrong_resolution":
        m["full_frame_resolution"]=256
    elif corruption=="view_slot_swap":
        m["private_view_details"][4]["selected_slot_b_teacher_only"]=7
    elif corruption=="empty_anchor_pair":
        m["persistent_selected_slot_ids_teacher_only"]=[0,0]
    with pytest.raises((ValueError,IndexError)):
        evaluate(p,m)


def test_noop_pixel_drift_is_measured_not_hidden_by_other_views():
    p,m=fixture()
    p[3,2,40,40,0]=.9
    c=evaluate(p,m)
    assert c["bounded_parity_and_separable_views"]==7
    assert c["per_view"][3]["no_op_rerender_parity_pass"] is False


def test_all_census_slots_must_be_finite_alpha():
    p,m=fixture()
    p[0,3,10,10,0]=np.nan
    with pytest.raises(ValueError):
        evaluate(p,m)


def test_wrong_arm_count_rejected():
    p,m=fixture()
    with pytest.raises(ValueError):
        evaluate(p[:,:-1],m)


def test_directly_hidden_and_isolated_invisible_remain_unqualified():
    pixels,manifest=fixture()
    pixels[:,3+7]=pixels[:,1]  # No effect from deleting slot 7 in original.
    pixels[:,3+8+7]=0.         # Nor after all other groups are suppressed.
    court=evaluate(pixels,manifest)
    assert court["slot_direct_effect_unobserved_ids_teacher_only"]==[7]
    assert court["directly_unobserved_but_isolated_visible_ids_teacher_only"]==[]
    assert court["unobservable_even_in_full_peel_ids_teacher_only"]==[7]
    assert court["different_drawable_negative_pairs"]==0
    assert court["training_authorized"] is False
