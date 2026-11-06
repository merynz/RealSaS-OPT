"""Adversarial synthetic tests of independent physical pixel consequence gate."""
import numpy as np
import pytest
from tools.corpus.court_corpus370_operational_palette_pixel_consequence_v1 import judge_edit_rgba


def scene():
    baseline=np.zeros((12,12,4),dtype=np.float32)
    baseline[:,:,:3]=0.4
    baseline[:,:,3]=1.0
    edited=baseline.copy()
    edited[:,:6,:3]=0.8
    samples=[{"x":x,"y":y,"color_index":0 if x<6 else 1}
             for y in range(12) for x in range(12)]
    return baseline,edited,samples


def test_selected_changes_and_collateral_stays_identical():
    a,b,s=scene()
    r=judge_edit_rgba(a,b,s,0)
    assert r['qualified'] is True
    assert r['selected_count']==72
    assert r['unselected_count']==72
    assert r['collateral_count']==0


def test_extra_collateral_pixel_fails():
    a,b,s=scene()
    b[8,10,1]=0.2
    r=judge_edit_rgba(a,b,s,0)
    assert r['status']=='FAIL_PIXEL_LOCALITY'
    assert r['collateral_count']==1


def test_inert_edit_fails_even_when_no_collateral():
    a,b,s=scene()
    b=a.copy()
    r=judge_edit_rgba(a,b,s,0)
    assert r['selected_unchanged_count']==72
    assert r['qualified'] is False


def test_altered_alpha_is_rejected_globally():
    a,b,s=scene()
    b[3,4,3]=0.6
    with pytest.raises(ValueError,match='ALPHA_CHANGED'):
        judge_edit_rgba(a,b,s,0)


def test_insufficient_samples_cannot_be_counted_as_pass():
    a,b,s=scene()
    r=judge_edit_rgba(a,b,s[:2],0)
    assert r['status']=='INSUFFICIENT_INTERIOR_SAMPLES'
    assert r['qualified'] is False


def test_unknown_interior_sample_is_illegal():
    a,b,s=scene()
    s[0]['color_index']=-1
    with pytest.raises(ValueError,match='UNKNOWN_SAMPLE'):
        judge_edit_rgba(a,b,s,0)


def test_invalid_dimensions_rejected():
    a,b,s=scene()
    with pytest.raises(ValueError,match='LAYOUT_INVALID'):
        judge_edit_rgba(a[:,:,:3],b,s,0)


def test_mislabeled_source_ray_membership_is_detectable():
    a,b,s=scene()
    s[0]['color_index']=1
    r=judge_edit_rgba(a,b,s,0)
    assert r['qualified'] is False
    assert r['collateral_count']==1


def test_per_color_locality_without_minting_owner():
    a,b,s=scene()
    r=judge_edit_rgba(a,b,s,0)
    assert 'same_drawable' not in r
    assert 'artist_owner' not in r
