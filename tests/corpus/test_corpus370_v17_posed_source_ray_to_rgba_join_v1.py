"""Red-team V17 pose-aware source-ray and observed RGBA join; no artist labels."""
from __future__ import annotations

import copy
import numpy as np
import pytest

from models.iris.truth.graphics_oracle_v1 import (
    OrthographicCameraV1, measure_orthographic_ray_stack_v1
)
from tools.corpus.measure_corpus370_v17_posed_source_ray_to_causal_rgba_v1 import (
    SCHEMA as PROPOSAL_SCHEMA,SAMPLES_PER_CASE,SIGNS,TOL,pixel_effect,premul
)
from tools.corpus.court_corpus370_v17_posed_source_ray_to_rgba_join_v1 import (
    SCHEMA as COURT_SCHEMA,verify_hit_world_address,
    verify_row_geometry_and_rgba
)


def fixture():
    cam=OrthographicCameraV1(
        origin=(0.,0.,3.),forward=(0.,0.,-1.),
        right=(1.,0.,0.),screen_up=(0.,1.,0.),
        half_extent=1.,resolution=4,view_index=0
    )
    xyz=np.array([
        [-1.,-1.,0.],[1.,-1.,0.],[0.,1.,0.]
    ],np.float64)
    face=np.array([[0,1,2]],np.int64)
    px,py=1,2
    ray=measure_orthographic_ray_stack_v1(
        xyz,face,cam,px,py,tie_epsilon=1e-7)
    assert ray.complete and not ray.has_depth_tie and len(ray.hits)==1
    h=ray.hits[0]
    hit=dict(face_index_teacher_only=h.face_index,
             barycentric_teacher_only=list(h.barycentric),
             forward_depth_source_world_units=h.depth)
    return cam,xyz,face,px,py,hit


def images(px,py,camera):
    shape=(camera.resolution,camera.resolution,4)
    baseline=np.zeros(shape,np.float32)
    source_a=baseline.copy()
    source_b=baseline.copy()
    joint=baseline.copy()
    by=camera.resolution-1-py
    baseline[by,px]=[.2,.2,.2,1.]
    source_a[by,px]=[.5,.2,.2,1.]
    source_b[by,px]=[.2,.5,.2,1.]
    joint[by,px]=[.9,.9,.2,1.]
    return baseline,source_a,source_b,joint


def row_for(cam,xyz,faces,px,py):
    ray=measure_orthographic_ray_stack_v1(
        xyz,faces,cam,px,py,tie_epsilon=1e-7)
    hits=[dict(face_index_teacher_only=h.face_index,
         barycentric_teacher_only=list(h.barycentric),
         forward_depth_source_world_units=h.depth) for h in ray.hits]
    return {
        "ray_image_pixel_topdown_x":px,
        "ray_image_pixel_topdown_y":py,
        "original_blender_framebuffer_y":cam.resolution-1-py,
        "geometric_stack_complete":ray.complete,
        "geometric_stack_has_depth_tie":ray.has_depth_tie,
        "source_geometric_hit_count":len(hits),
        "source_geometric_all_hits_ordered_near_to_far":hits,
        "raster_original_source_alpha":1.,
        "real_original_source_joint_ab_visible_rgba_effect":True,
        "real_original_source_joint_ab_nonadditive_rgba_effect":True,
        "source_render_order_of_hits_qualified":False,
        "source_material_alpha_per_hit":"UNKNOWN",
        "actual_artist_drawable_relation":"UNKNOWN",
        "status":"SOURCE_GEOMETRIC_RAY_JOIN_OBSERVED__NOT_ALPHA_OCCLUSION_PROOF"
    }


def test_a_real_barycentric_world_hit_and_caused_original_pixel():
    cam,xyz,faces,px,py,hit=fixture()
    verify_hit_world_address(xyz,faces,cam,px,py,hit,2.)
    base,a,b,joint=images(px,py,cam)
    result=verify_row_geometry_and_rgba(
        row_for(cam,xyz,faces,px,py),xyz,faces,cam,base,a,b,joint,2.,1e-7
    )
    assert result==(True,False,True,True,True)


@pytest.mark.parametrize("mutation",[
    "face","bary","offray","depth","depth_nan","bary_bad_sum"
])
def test_wrong_original_source_hit_world_address_is_rejected(mutation):
    cam,xyz,faces,px,py,hit=fixture()
    bad=copy.deepcopy(hit)
    if mutation=="face":bad["face_index_teacher_only"]=2
    elif mutation=="bary":bad["barycentric_teacher_only"]=[1.,0.,0.]
    elif mutation=="offray":bad["barycentric_teacher_only"]=[0.,1.,0.]
    elif mutation=="depth":bad["forward_depth_source_world_units"]+=.1
    elif mutation=="depth_nan":bad["forward_depth_source_world_units"]=float("nan")
    elif mutation=="bary_bad_sum":bad["barycentric_teacher_only"]=[.3,.3,.3]
    with pytest.raises(ValueError):
        verify_hit_world_address(xyz,faces,cam,px,py,bad,2.)


@pytest.mark.parametrize("mutation",[
    "pixel_in_wrong_blender_row","forged_visible_effect","forged_alpha",
    "draw_order_forged","artist_owner_forged","hidden_hit_fabricated",
    "source_face_changed","ray_status_forged","nonadditivity_forged"
])
def test_independent_court_rejects_improper_raster_or_authority(mutation):
    cam,xyz,faces,px,py,hit=fixture()
    row=row_for(cam,xyz,faces,px,py)
    base,a,b,joint=images(px,py,cam)
    if mutation=="pixel_in_wrong_blender_row":
        row["original_blender_framebuffer_y"]=py
    elif mutation=="forged_visible_effect":
        row["real_original_source_joint_ab_visible_rgba_effect"]=False
    elif mutation=="forged_alpha":
        row["raster_original_source_alpha"]=.35
    elif mutation=="draw_order_forged":
        row["source_render_order_of_hits_qualified"]=True
    elif mutation=="artist_owner_forged":
        row["actual_artist_drawable_relation"]="SAME_DRAWABLE"
    elif mutation=="hidden_hit_fabricated":
        row["source_geometric_hit_count"]=2
    elif mutation=="source_face_changed":
        row["source_geometric_all_hits_ordered_near_to_far"][0][
            "face_index_teacher_only"]=4
    elif mutation=="ray_status_forged":
        row["status"]="VISIBLE_TOPMOST_DRAWABLE"
    elif mutation=="nonadditivity_forged":
        row["real_original_source_joint_ab_nonadditive_rgba_effect"]=False
    with pytest.raises(ValueError):
        verify_row_geometry_and_rgba(
            row,xyz,faces,cam,base,a,b,joint,2.,1e-7)


def test_transparent_rgb_is_not_causal_evidence():
    b=np.array([.1,.1,.1,0.],np.float32)
    a=np.array([1.,.5,.7,0.],np.float32)
    assert not bool(pixel_effect(b,a))
    assert np.array_equal(premul(b),premul(a))


def test_alpha_change_without_rgb_change_is_real_effect():
    b=np.array([.1,.1,.1,.1],np.float32)
    a=np.array([.1,.1,.1,.4],np.float32)
    assert bool(pixel_effect(b,a))


def test_nonfinite_rgba_rejected():
    rgba=np.array([.2,.2,.2,1.],np.float32)
    bad=rgba.copy();bad[2]=np.nan
    with pytest.raises(ValueError):pixel_effect(rgba,bad)


def test_unknown_never_promotes_owner_and_no_fixed_semantic_k():
    assert PROPOSAL_SCHEMA.endswith(".v17")
    assert COURT_SCHEMA.endswith(".v17")
    assert SAMPLES_PER_CASE==12
    assert len(SIGNS)==4
    assert TOL==.025
    assert "body" not in SIGNS and "arm" not in SIGNS
