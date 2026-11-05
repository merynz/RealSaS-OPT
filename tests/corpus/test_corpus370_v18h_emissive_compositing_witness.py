"""V18h original-source emissive layer-composition redteam without drawable gold."""
import numpy as np
import pytest
from tools.corpus.court_corpus370_v18h_emissive_compositing_witness import (
    arms,THRESHOLDS,SCHEMA,STATUS,TOL,compositor_probe,
)
def blank():
    return np.zeros((192,192,4),np.float32)
def solid(rgb,alpha=1.):
    a=blank();a[35:60,35:60,:3]=rgb;a[35:60,35:60,3]=alpha
    return a
def unpremul(value):
    out=value.copy();inside=out[...,3]>0
    out[inside,:3]=out[inside,:3]/out[inside,3:4]
    return out
def test_frozen_source_groups_and_color_probes():
    assert THRESHOLDS==(.045,.1,.18)
    assert arms()==["EMISSIVE_EMPTY",
      "EMISSIVE_0p045_FULL","EMISSIVE_0p045_ISOLATE","EMISSIVE_0p045_REMOVE",
      "EMISSIVE_0p1_FULL","EMISSIVE_0p1_ISOLATE","EMISSIVE_0p1_REMOVE",
      "EMISSIVE_0p18_FULL","EMISSIVE_0p18_ISOLATE","EMISSIVE_0p18_REMOVE"]
    assert "NO_DRAWABLE_AUTHORITY" in STATUS
    assert SCHEMA.endswith(".v18h")
    assert TOL==.025
def test_emissive_disjoint_components_are_not_a_draw_order_label():
    a=solid((.9,.08,.035));z=blank();z[65:90,65:90,:3]=(.04,.14,.92);z[65:90,65:90,3]=1.
    full=a+z
    d=compositor_probe(full,a,z)
    assert d["per_pixel_order_oracle"]["pixel_count_above_original_tolerance"]==0
    assert d["order_discriminating_overlap_pixels"]==0
    assert not d["draw_order_observable"]
def test_emissive_opaque_overlap_changes_layer_order():
    a=solid((.9,.08,.035),1.)
    z=solid((.04,.14,.92),1.)
    d=compositor_probe(a,a,z)
    assert d["order_discriminating_overlap_pixels"]==625
    assert d["group_over_remainder"]["pixel_count_above_original_tolerance"]==0
    assert d["remainder_over_group"]["pixel_count_above_original_tolerance"]==625
def test_emissive_mixed_pixel_depth_order_requires_something_beyond_one_constant_layer():
    a=solid((.9,.08,.035),.8)
    z=solid((.04,.14,.92),.8)
    top= a[..., :3]*.8 + z[..., :3]*.8*.2
    bottom=z[..., :3]*.8 + a[..., :3]*.8*.2
    src=blank()
    src[35:60,35:60,3]=.96
    src[35:60,35:47,:3]=top[35:60,35:47]/.96
    src[35:60,47:60,:3]=bottom[35:60,47:60]/.96
    d=compositor_probe(src,a,z)
    assert d["per_pixel_order_oracle"]["pixel_count_above_original_tolerance"]==0
    assert d["group_over_remainder"]["pixel_count_above_original_tolerance"]>0
    assert d["remainder_over_group"]["pixel_count_above_original_tolerance"]>0
    assert not d["draw_order_one_fixed_global_direction_possible"]
@pytest.mark.parametrize("attack",["nan","alpha_negative","alpha_gt_one"])
def test_no_corrupted_frame_can_be_called_reconstructible(attack):
    a=blank()
    if attack=="nan":a[0,0,0]=np.nan
    elif attack=="alpha_negative":a[0,0,3]=-.5
    else:a[0,0,3]=1.8
    with pytest.raises(ValueError):
        compositor_probe(a,blank(),blank())
