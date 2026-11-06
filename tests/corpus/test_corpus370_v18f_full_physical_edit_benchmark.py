"""Frozen physical-edit benchmark contracts; no train/artist label minting."""
import numpy as np
import pytest
from tools.corpus.court_corpus370_v18f_full_physical_edit_benchmark import (
    SCHEMA,RENDER,PASS,TOL,VIEWS,TARGET_INDICES,THRESHOLDS,
    expected_arms,rgba_effect,effects
)

def blank():
    return np.zeros((192,192,4),np.float32)

def test_all_precommitted_arms_unmodified():
    arms=expected_arms()
    assert len(arms)==52 and len(set(arms))==52
    assert arms[:4]==["SOURCE_ORIGINAL","SOUP_FULL","SOUP_NOOP","SOUP_EMPTY"]
    assert TARGET_INDICES==(0,2,4,6,8,10)
    assert THRESHOLDS==(.045,.1,.18)
    assert VIEWS==(0,2,4,6)
    assert TOL==.025

def test_alpha_edit_is_observed_and_invalid_editor_mismatch_caught():
    a=blank();b=blank()
    b[3:8,4:9,3]=1.
    mask=rgba_effect(a,b)
    assert mask.sum()==25
    e=effects(mask,mask)
    assert e["group_minus_single_face_proxy_effect_pixels"]==0
    assert e["single_face_proxy_minus_group_effect_pixels"]==0
    assert e["observable_single_face_reference"] is True
    assert e["candidate_equals_artist_editable_layer"]=="UNKNOWN"

def test_extra_group_effect_is_not_artist_collateral_gold():
    ref=np.zeros((4,4),bool)
    grp=np.ones((4,4),bool)
    r=effects(ref,grp)
    assert r["group_minus_single_face_proxy_effect_pixels"]==16
    assert not r["observable_single_face_reference"]
    assert r["candidate_equals_artist_editable_layer"]=="UNKNOWN"

@pytest.mark.parametrize("bad",["nan","inf","shape","channel"])
def test_corrupted_physical_rgba_rejected(bad):
    a=blank();b=a.copy()
    if bad=="nan":b[0,0,0]=np.nan
    if bad=="inf":b[0,0,1]=np.inf
    if bad=="shape":b=b[:4]
    if bad=="channel":b=b[...,:3]
    with pytest.raises(ValueError):
        rgba_effect(a,b)

def test_teacher_only_diagnostic_never_qualified_owner():
    assert SCHEMA.endswith(".v18f")
    assert RENDER.endswith(".v18f")
    assert "ARTIST_DRAWABLE_UNKNOWN" in PASS
