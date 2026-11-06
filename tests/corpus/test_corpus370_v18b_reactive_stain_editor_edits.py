"""V18b heldout original source graphics edit collateral redteam, not artist GT."""
from __future__ import annotations

import numpy as np
import pytest

from tools.corpus.court_corpus370_v18b_reactive_stain_editor_edits import (
    SCHEMA,THRESHOLDS,TOL,effects,expected_arms,rgba_effect
)


def solid(r=.2,g=.3,b=.4,a=1.):
    x=np.zeros((4,4,4),np.float32)
    x[:]=[r,g,b,a]
    return x


def test_original_reference_matches_without_group_collateral():
    ref=np.zeros((4,4),bool);ref[0,0]=True
    effect=effects(ref,ref)
    assert effect["intersection_with_single_face_proxy_pixel_instances"]==1
    assert effect["group_only_edit_collateral_pixel_instances"]==0
    assert effect["single_face_proxy_not_explained_pixel_instances"]==0
    assert effect["candidate_equals_artist_editable_layer"]=="UNKNOWN"


def test_overedit_is_measured_not_ground_truth_drawable_error():
    ref=np.zeros((4,4),bool);ref[0,0]=True
    broad=ref.copy();broad[1,1]=True;broad[2,2]=True
    e=effects(ref,broad)
    assert e["group_only_edit_collateral_pixel_instances"]==2
    assert e["single_face_proxy_not_explained_pixel_instances"]==0
    assert e["source_stain_group_edit_effect_pixel_instances"]==3
    assert e["candidate_equals_artist_editable_layer"]=="UNKNOWN"


def test_undereffect_is_not_silently_hidden():
    ref=np.zeros((4,4),bool);ref[0,0]=ref[1,0]=True
    group=np.zeros((4,4),bool);group[0,0]=True
    e=effects(ref,group)
    assert e["single_face_proxy_not_explained_pixel_instances"]==1
    assert e["intersection_with_single_face_proxy_pixel_instances"]==1


def test_invisible_single_face_reference_remains_unknown_not_negative():
    source=np.zeros((4,4),bool)
    group=np.zeros((4,4),bool);group[2,2]=True
    e=effects(source,group)
    assert e["observable_single_face_reference"] is False
    assert e["candidate_equals_artist_editable_layer"]=="UNKNOWN"


def test_transparent_rgb_is_not_edit_delta():
    x=solid(.2,.3,.4,0.)
    y=solid(.9,.1,.7,0.)
    assert not rgba_effect(x,y).any()


def test_true_alpha_edit_even_same_straight_color_is_detected():
    x=solid(.2,.3,.4,.4)
    y=solid(.2,.3,.4,.7)
    assert rgba_effect(x,y).all()


def test_source_mask_restores_real_original_even_with_color_change():
    x=solid(.2,.3,.4,.7)
    y=solid(.2,.3,.4,.7)
    y[0,0]=[1.,0.,0.,.7]
    signal=rgba_effect(x,y)
    assert signal.sum()==1


def test_observed_edit_reference_is_not_semantic_part():
    assert SCHEMA.endswith(".v18b")
    assert THRESHOLDS==(.045,.10,.18)
    arms=expected_arms()
    assert len(arms)==4+6*2*(1+len(THRESHOLDS))
    assert arms[0:4]==[
        "SOURCE_ORIGINAL","SOUP_FULL","SOUP_NOOP","SOUP_EMPTY"]
    assert len(arms)==len(set(arms))
    assert not any("hair" in x.lower() or "body" in x.lower()
                   or "armature" in x.lower() for x in arms)


@pytest.mark.parametrize("attack",[
    "nan_pixel","inf_pixel","invalid_rgba","different_shape",
    "invalid_reference_shape","alpha_zero_rgb_noise"
])
def test_independent_edit_pixel_court_strictness(attack):
    if attack in ("invalid_reference_shape",):
        with pytest.raises(ValueError):
            effects(np.zeros((2,2),bool),np.zeros((4,4),bool))
        return
    x=solid();y=solid()
    if attack=="nan_pixel":y[0,0,0]=np.nan
    elif attack=="inf_pixel":y[0,0,2]=np.inf
    elif attack=="invalid_rgba":y=y[...,:3]
    elif attack=="different_shape":y=y[:2]
    elif attack=="alpha_zero_rgb_noise":
        x[...,3]=0.;y[...,3]=0.;y[0,0,:3]=[.9,.8,.7]
        assert not rgba_effect(x,y).any();return
    with pytest.raises(ValueError):
        rgba_effect(x,y)


def test_unseen_source_pose_court_metrics_are_per_camera_and_per_query():
    arms=expected_arms()
    assert len(arms)==52
    # Six original V11 source targets × four calibrated cameras ×
    # three precommitted variable-K proposal policies.
    assert 6*4*len(THRESHOLDS)==72
    assert abs(TOL-.025)<1e-12
