"""V16 actual paired source controls: false drawable/compositor authority redteam.

Test doubles are deliberately NOT original artist sources, nor admission data.
The independent court must not import Blender, source rig or artist semantics.
"""
from __future__ import annotations
import numpy as np
import pytest

from tools.corpus.court_corpus370_dual_native_interaction_v16 import (
    SCHEMA, PAIRS, TOL, measure, premultiplied
)


def img(r=0.,g=0.,b=0.,a=1.):
    return np.array([[[r,g,b,a]]],np.float32)


def apply(base,ra=0.,ga=0.,ba=0.,aa=0.):
    z=base.copy()
    z[...,:3]+=np.array([ra,ga,ba])
    z[...,3]+=aa
    return z


def test_exact_original_common_baseline_simple_additive_dual_control():
    b=img(.1,.2,.3)
    a=apply(b,ra=.3)
    c=apply(b,ga=.25)
    ab=apply(b,ra=.3,ga=.25)
    iso_base=b.copy()
    m=measure(b,a,b,c,ab,iso_base,ab)
    assert m["single_a_effect_pixel_instances"]==1
    assert m["single_b_effect_pixel_instances"]==1
    assert m["joint_ab_effect_pixel_instances"]==1
    assert m["joint_nonadditive_rgba_response_pixel_instances"]==0
    assert m["joint_isolated_active_face_effect_pixel_instances"]==1
    assert m["joint_full_and_isolated_effect_overlap_pixel_instances"]==1
    assert m["joint_isolation_only_effect_pixel_instances__unresolved_visibility_or_compositing"]==0
    assert m["proves_artist_drawable_equivalence"] is False


def test_true_joint_source_nonadditive_visual_response_does_not_prove_drawable():
    base=img()
    a=img(.2)
    b=img(g=.2)
    joint=img(b=.7)
    iso_base=img()
    iso=joint
    m=measure(base,a,base,b,joint,iso_base,iso)
    assert m["joint_nonadditive_rgba_response_pixel_instances"]==1
    assert m["incremental_a_given_b_pixel_instances"]==1
    assert m["incremental_b_given_a_pixel_instances"]==1
    assert m["source_drawable_same_as_control"] is False
    assert m["proves_artist_drawable_equivalence"] is False


def test_noop_sham_all_zero_and_no_fake_interaction():
    b=img(.1,.1,.1,.75)
    m=measure(b,b,b,b,b,b,b)
    assert all(m[k]==0 for k in (
        "single_a_effect_pixel_instances","single_b_effect_pixel_instances",
        "joint_ab_effect_pixel_instances",
        "joint_nonadditive_rgba_response_pixel_instances",
        "joint_isolation_only_effect_pixel_instances__unresolved_visibility_or_compositing",
        "joint_full_only_effect_pixel_instances__contextual",
        "joint_alpha_effect_pixel_instances",
    ))


def test_isolation_only_is_unknown_not_occlusion_or_author_identity():
    full=img(0.1)
    nofull=full.copy()
    isolated_base=img(0.1)
    isolated_changed=img(0.7)
    m=measure(full,nofull,full,nofull,full,isolated_base,isolated_changed)
    assert m["joint_ab_effect_pixel_instances"]==0
    assert m["joint_isolated_active_face_effect_pixel_instances"]==1
    assert m["joint_isolation_only_effect_pixel_instances__unresolved_visibility_or_compositing"]==1
    assert m["joint_full_only_effect_pixel_instances__contextual"]==0
    assert m["proves_artist_drawable_equivalence"] is False


def test_full_only_is_contextual_not_new_drawable():
    full=img()
    joint=img(.5)
    isolated_base=img()
    isolated_joint=isolated_base.copy()
    m=measure(full,full,full,full,joint,isolated_base,isolated_joint)
    assert m["joint_full_only_effect_pixel_instances__contextual"]==1
    assert m["joint_isolated_active_face_effect_pixel_instances"]==0
    assert m["proves_artist_drawable_equivalence"] is False


def test_no_color_signal_under_fully_transparent_rgba():
    base=img(.01,.01,.01,0.)
    arbitrary=img(1.,.2,.7,0.)
    m=measure(base,arbitrary,base,arbitrary,arbitrary,base,arbitrary)
    assert m["joint_ab_effect_pixel_instances"]==0
    assert m["joint_nonadditive_rgba_response_pixel_instances"]==0
    assert m["joint_isolated_active_face_effect_pixel_instances"]==0
    assert premultiplied(base).tolist()==premultiplied(arbitrary).tolist()


def test_alpha_only_effect_counted_separately():
    b=img(.2,.1,.3,.4)
    changed=img(.2,.1,.3,.9)
    m=measure(b,b,b,b,changed,b,changed)
    assert m["joint_alpha_effect_pixel_instances"]==1
    assert m["joint_ab_effect_pixel_instances"]==1
    assert m["joint_isolated_active_face_effect_pixel_instances"]==1


def test_simultaneous_intervention_can_cancel_visible_single_effect():
    baseline=img()
    a=img(.5)
    b=img(-.5)
    m=measure(baseline,a,baseline,b,baseline,baseline,baseline)
    assert m["single_effect_union_pixel_instances"]==1
    assert m["joint_ab_effect_pixel_instances"]==0
    assert m["single_affected_pixels_disappearing_under_joint"]==1


def test_simultaneous_intervention_can_expose_nonobserved_single_effect():
    b=img()
    m=measure(b,b,b,b,img(.6),b,img(.6))
    assert m["single_effect_union_pixel_instances"]==0
    assert m["joint_ab_effect_pixel_instances"]==1
    assert m["joint_affected_pixels_beyond_single_union"]==1


def test_swapping_a_b_preserves_commutative_joint_nonadditivity():
    base=img(.2,.2,.2)
    a=img(.7,.2,.2)
    b=img(.2,.6,.2)
    joint=img(.7,.6,.6)
    m1=measure(base,a,base,b,joint,base,joint)
    m2=measure(base,b,base,a,joint,base,joint)
    assert m1["joint_nonadditive_rgba_response_pixel_instances"]==m2[
        "joint_nonadditive_rgba_response_pixel_instances"]
    assert m1["joint_ab_effect_pixel_instances"]==m2[
        "joint_ab_effect_pixel_instances"]


@pytest.mark.parametrize("attack",["wrong_base","illegal_tolerance_zero",
                                  "illegal_tolerance_large","nan_rgba",
                                  "inf_rgba","rgba_missing_alpha"])
def test_strict_visual_court_rejects_deviant_source_or_unsafe_math(attack):
    base=img()
    a=img(.2)
    other=base.copy()
    tol=TOL
    if attack=="wrong_base":other[0,0,0]=.5
    elif attack=="illegal_tolerance_zero":tol=0.
    elif attack=="illegal_tolerance_large":tol=TOL+1e-3
    elif attack=="nan_rgba":a[0,0,1]=float("nan")
    elif attack=="inf_rgba":a[0,0,2]=float("inf")
    elif attack=="rgba_missing_alpha":a=a[...,:3]
    with pytest.raises((ValueError,AssertionError,IndexError)):
        measure(base,a,other,base,base,base,base,tol=tol)


def test_v16_support_slots_are_anonymous_not_fixed_artist_parts():
    assert SCHEMA=="RealSaS.Corpus370IndependentOriginalDualNativeControlRGBAInteractionCourt.v16"
    assert PAIRS==("PLUS_PLUS","PLUS_MINUS","MINUS_PLUS","MINUS_MINUS")
    assert TOL==.025
    m=measure(img(),img(.4),img(),img(g=.4),img(.4,.4),img(),img(.4,.4))
    assert all("bone" not in k and "material_name" not in k and "artist_part" not in k
               for k in m)
    assert m["proves_artist_drawable_equivalence"] is False
