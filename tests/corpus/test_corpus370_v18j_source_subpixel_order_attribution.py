"""Adversarial V18j source-posed 4x4 geometric rays; no artist labels."""
from __future__ import annotations

import numpy as np
import pytest

from models.iris.truth.graphics_oracle_v1 import OrthographicCameraV1
from tools.corpus.measure_corpus370_v18j_source_subpixel_order_attribution import (
    MAX_PIXELS_PER_VIEW_STRATUM,SAMPLE_COUNT,STRATA,SCHEMA,STATUS,
    pick_pixels,ray_probe,stratum_masks,
)

def image(rgb=(0.,0.,0.),alpha=0.):
    x=np.zeros((192,192,4),np.float32)
    x[20:30,30:40,:3]=rgb
    x[20:30,30:40,3]=alpha
    return x

def test_alpha_filtering_versus_postfilter_over_is_a_retrospective_alpha_stratum():
    full=image((.47,.11,.48),1.)
    group=image((.9,.08,.035),.5)
    rem=image((.04,.14,.92),.5)
    masks=stratum_masks(full,group,rem)
    assert masks["ALPHA_FAIL"].sum()==100
    assert not masks["RGB_ONLY_FAIL"].any()
    assert not masks["VISIBLE_PASS"].any()
    assert set(masks)==set(STRATA)

def test_rgb_order_failure_without_alpha_mismatch_stays_distinct():
    full=image((.5,.5,.5),1.)
    group=image((.9,.08,.035),1.)
    rem=image((.04,.14,.92),1.)
    masks=stratum_masks(full,group,rem)
    assert masks["RGB_ONLY_FAIL"].sum()==100
    assert masks["ALPHA_FAIL"].sum()==0

def test_visible_success_is_visible_not_background():
    source=image((.9,.08,.035),1.)
    masks=stratum_masks(source,source,image())
    assert masks["VISIBLE_PASS"].sum()==100
    assert masks["ALPHA_FAIL"].sum()==0

def test_even_spaced_original_framebuffer_pixel_selection_is_repeatable_and_bounded():
    m=np.ones((192,192),dtype=bool)
    expected=pick_pixels(m)
    assert expected==pick_pixels(m)
    assert len(expected)==MAX_PIXELS_PER_VIEW_STRATUM==5
    assert len(set(expected))==5
    assert expected[0]==(0,0)
    assert expected[-1]==(191,191)
    small=np.zeros((192,192),bool)
    small[0,1]=True
    assert pick_pixels(small)==[(0,1)]

def camera(z=3.,forward=(0.,0.,-1.)):
    return OrthographicCameraV1(
        origin=(0.,0.,z),forward=forward,
        right=(1.,0.,0.),screen_up=(0.,1.,0.),
        half_extent=1.,resolution=192,view_index=0)

def stacked_triangles():
    # Front/back occupy exactly same geometric pixel support, at distinct depths.
    xy=np.array([[-1.,-1.],[1.,-1.],[0.,1.]])
    vertices=np.concatenate((
        np.column_stack((xy,np.full(3,.5))),
        np.column_stack((xy,np.full(3,-.5))),
    ),axis=0)
    faces=np.array([[0,1,2],[3,4,5]],np.int64)
    return vertices,faces

def test_original_ray_stack_keeps_true_depth_and_bottom_up_coordinate_transform():
    vertices,faces=stacked_triangles()
    labels=np.array([7,8],np.int32)
    e=ray_probe(vertices,faces,camera(),95,96,labels,7,1e-8)
    assert e["source_framebuffer_pixel_y_bottom_up"]==95
    assert e["source_geometry_pixel_y_top_down"]==96
    assert e["qualified_subsample_count"]==SAMPLE_COUNT==16
    assert e["unknown_incomplete_or_tied_subsamples"]==0
    assert e["qualified_first_group_subsamples"]==16
    assert e["qualified_first_remainder_subsamples"]==0
    assert e["qualified_with_both_sides_in_depth_stack"]==16
    assert not e["opposite_nearest_source_sides_across_subpixels"]
    assert e["fully_qualified_4x4_coverage_fractions"][
        "geometric_joint_any_subsample_fraction"]==1.
    assert all(z["depth_sorted_source_hits_private"][0][
        "anonymized_candidate_side"]=="G" for z in e["geometric_subsamples"])

def test_camera_inside_mesh_denies_negative_coverage():
    vertices,faces=stacked_triangles()
    e=ray_probe(vertices,faces,camera(z=-3.),95,96,
                np.array([7,8],np.int32),7,1e-8)
    assert e["qualified_subsample_count"]==0
    assert e["unknown_incomplete_or_tied_subsamples"]==16
    assert e["fully_qualified_4x4_coverage_fractions"] is None
    assert all(z["first_visible_geometric_side"]=="UNKNOWN"
               for z in e["geometric_subsamples"])

@pytest.mark.parametrize("attack",["nan","alpha_below_zero","alpha_over_one","wrong_shape"])
def test_bad_renderer_input_fails_closed(attack):
    a=image((.1,.1,.1),.7)
    if attack=="nan":a[20,30,0]=np.nan
    if attack=="alpha_below_zero":a[20,30,3]=-.2
    if attack=="alpha_over_one":a[20,30,3]=1.4
    if attack=="wrong_shape":a=a[:10]
    with pytest.raises(ValueError,match="V18J_SOURCE_EMISSIVE_RGBA"):
        stratum_masks(a,image(),image())

def test_no_gold_authority_and_fixed_v17b_offsets():
    assert SCHEMA.endswith(".v18j")
    assert "NO_LAYER_AUTHORITY" in STATUS
    assert STRATA==("ALPHA_FAIL","RGB_ONLY_FAIL","VISIBLE_PASS")
    assert SAMPLE_COUNT==16
