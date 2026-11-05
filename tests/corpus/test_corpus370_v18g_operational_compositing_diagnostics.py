"""Adversarial falsifiers for independent V18g compositing, no owner truth."""
import numpy as np
import pytest
from tools.corpus.court_corpus370_v18g_operational_compositing_diagnostics import (
    TOL, SCHEMA, STATUS, compositor_probe, packed_rgba,
    over_premul, validate_rgba,
)

def empty():
    return np.zeros((192,192,4), np.float32)

def rectangle(rgb, alpha, rows, cols):
    a=empty()
    a[rows,cols,:3]=rgb
    a[rows,cols,3]=alpha
    return a

def from_premul(p):
    a=p.copy()
    mask=p[...,3]>1e-7
    a[mask,:3]=p[mask,:3]/p[mask,3:4]
    return a

def test_nonoverlapping_layers_reconstruct_but_are_not_order_evidence():
    a=rectangle((.9,.3,.1),1.,slice(5,20),slice(10,40))
    b=rectangle((.1,.4,.9),1.,slice(50,80),slice(10,40))
    src=from_premul(over_premul(a,b))
    d=compositor_probe(src,a,b)
    assert d["both_layers_alpha_overlap_pixels"]==0
    assert d["order_discriminating_overlap_pixels"]==0
    assert not d["draw_order_observable"]
    assert not d["draw_order_one_fixed_global_direction_possible"]
    assert d["per_pixel_order_oracle"]["pixel_count_above_original_tolerance"]==0

def test_zero_complement_is_vacuous_even_if_perfect_reconstruction():
    a=rectangle((.8,.2,.1),1.,slice(3,27),slice(3,27))
    d=compositor_probe(a,a,empty())
    assert d["remainder_alone_alpha_visible_pixels"]==0
    assert d["group_over_remainder"]["pixel_count_above_original_tolerance"]==0
    assert d["order_discriminating_overlap_pixels"]==0
    assert not d["draw_order_observable"]

def test_over_and_under_discriminate_real_occlusion():
    top=rectangle((1.,.1,.1),.75,slice(20,30),slice(20,30))
    bottom=rectangle((.1,.1,1.),.70,slice(20,30),slice(20,30))
    src=from_premul(over_premul(top,bottom))
    d=compositor_probe(src,top,bottom)
    assert d["order_discriminating_overlap_pixels"]==100
    assert d["informative_pixels_support_group_over_only"]==100
    assert d["informative_pixels_support_remainder_over_only"]==0
    assert d["informative_pixels_both_orders_fail"]==0
    assert d["group_over_remainder"]["pixel_count_above_original_tolerance"]==0
    assert d["remainder_over_group"]["pixel_count_above_original_tolerance"]==100
    assert d["draw_order_one_fixed_global_direction_possible"]

def test_per_pixel_oracle_fails_for_unrepresentable_source_and_does_not_make_fake_order():
    top=rectangle((1.,0.,0.),.75,slice(20,30),slice(20,30))
    bottom=rectangle((0.,0.,1.),.75,slice(20,30),slice(20,30))
    source=rectangle((0.,1.,0.),.9375,slice(20,30),slice(20,30))
    d=compositor_probe(source,top,bottom)
    assert d["informative_pixels_both_orders_fail"]==100
    assert d["per_pixel_order_oracle"]["pixel_count_above_original_tolerance"]==100
    assert not d["draw_order_one_fixed_global_direction_possible"]

def test_opposite_occlusions_are_possible_in_two_separate_pixel_zones():
    a=rectangle((1.,0.,0.),.8,slice(30,50),slice(30,80))
    b=rectangle((0.,0.,1.),.8,slice(30,50),slice(30,80))
    left=over_premul(a,b);right=over_premul(b,a)
    source_p=left.copy()
    source_p[:,55:,:]=right[:,55:,:]
    source=from_premul(source_p)
    d=compositor_probe(source,a,b)
    assert d["per_pixel_order_oracle"]["pixel_count_above_original_tolerance"]==0
    assert d["group_over_remainder"]["pixel_count_above_original_tolerance"]>0
    assert d["remainder_over_group"]["pixel_count_above_original_tolerance"]>0
    assert not d["draw_order_one_fixed_global_direction_possible"]

@pytest.mark.parametrize("corrupt",["nan","inf","wrong_shape","negative_alpha","over_alpha"])
def test_invalid_original_rgba_never_admitted(corrupt):
    frame=empty()
    if corrupt=="nan":frame[0,0,0]=np.nan
    elif corrupt=="inf":frame[0,0,0]=np.inf
    elif corrupt=="wrong_shape":frame=frame[...,:3]
    elif corrupt=="negative_alpha":frame[0,0,3]=-.5
    elif corrupt=="over_alpha":frame[0,0,3]=1.5
    with pytest.raises(ValueError,match="V18G_SOURCE_RGBA_NOT_VALID"):
        validate_rgba(frame)

def test_status_and_threshold_retain_unknown_instead_of_training_gold():
    assert SCHEMA.endswith(".v18g")
    assert "NO_OWNER_TRUTH" in STATUS
    assert TOL==.025
