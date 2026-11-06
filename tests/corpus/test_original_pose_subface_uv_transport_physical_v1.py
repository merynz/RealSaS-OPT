"""Red-team tests for source physical pose-indexed UV anchor proof (synthetic).

Only tests geometric/epistemic logic; real source authority requires the
sealed original Cat 64-Cycles-source-frames independent court.
"""
import numpy as np
import pytest
from tools.corpus.measure_original_pose_subface_uv_transport_physical_v1 import (
    physical_source_witnesses,transport_anchor_rows,
)

def row(face=0,edit=0,x=96,y=96):
    return {
        "face_index":face,"source_uv_edit_operation_index":edit,
        "pixel_x":x,"pixel_y_top_first":y
    }

def source_sample():
    arr=np.zeros((2,4,192,192,4),np.float32)
    arr[...,3]=1
    arr[:,1,95,96,0]=.2
    return arr

def test_two_original_same_cycles_source_pixels_non_target_arms_absent():
    a=physical_source_witnesses([row()],source_sample(),0,0)
    assert a["source_face_pixel_count"]==1
    assert a["source_positive_sample_count"]==2
    assert a["source_other_edit_absence_count"]==4
    assert a["source_edit_pixel_census"]==[1,0,0]

def test_wrong_original_uv_edit_breaks_independent_source_control():
    a=source_sample()
    a[:,2,95,96,1]=.2
    with pytest.raises(ValueError,match="NON_TARGET_EDIT_FALSE_POSITIVE"):
        physical_source_witnesses([row()],a,14,0)

def test_unobservable_at_exact_original_pixel_cannot_be_minted_as_positive():
    a=source_sample()
    a[:,1,95,96,0]=0
    with pytest.raises(ValueError,match="NOT_PHYSICALLY_OBSERVED"):
        physical_source_witnesses([row()],a,0,0)

def test_wrong_blender_bottom_first_row_rejected():
    a=source_sample()
    a[:,1,95,96,0]=0
    a[:,1,96,96,0]=.3
    with pytest.raises(ValueError,match="NOT_PHYSICALLY_OBSERVED"):
        physical_source_witnesses([row()],a,0,0)

def test_original_alpha_drift_is_fail_closed():
    a=source_sample()
    a[:,1,95,96,3]=.5
    with pytest.raises(ValueError,match="ALPHA_NOT_OPAQUE"):
        physical_source_witnesses([row()],a,0,0)

def synthetic_projected_triangles():
    tri=np.array([
        [[55.,55.],[139.,55.],[97.,139.]],
        [[151.,151.],[160.,151.],[155.,160.]]
    ],np.float64)
    posed=tri.copy()
    posed[0,:,0]+=2.0
    return tri,posed

def test_same_barycentric_source_anchor_transports_between_original_poses():
    p0,p14=synthetic_projected_triangles()
    original=[row()]
    moved=[row(x=98)]
    found=transport_anchor_rows(original,moved,p0,p14,view=0)
    assert len(found)==1
    x=found[0]
    assert x["pose0_pixel_xy"]==[96,96]
    assert x["pose14_qualified_pixel_xy"]==[98,96]
    assert np.isclose(x["screen_displacement_px"],2.0)
    assert np.isclose(sum(x["pose0_center_source_barycentric"]),1.0)
    assert x["source_pose0_and_pose14_independent_physical_positive"] is True
    assert x["exact_Cycles_sample_ray_UV_correspondence"]=="UNKNOWN_NOT_ASSERTED"
    assert x["original_artist_drawable_owner"]=="UNKNOWN"

def test_unwitnessed_pose14_face_is_not_hidden_or_negative():
    p0,p14=synthetic_projected_triangles()
    assert transport_anchor_rows([row()],[],p0,p14,view=0)==[]
    # No original pose14 witness ≠ proof of occlusion, art partition,
    # or source edit impossibility; this is an explicitly incomplete test.

def test_different_original_face_cannot_steal_transported_source_anchor():
    p0,p14=synthetic_projected_triangles()
    assert transport_anchor_rows([row()], [row(face=1,x=98)],p0,p14,view=0)==[]

def test_independent_source_edit_id_is_necessary_for_transport():
    p0,p14=synthetic_projected_triangles()
    assert transport_anchor_rows([row(edit=0)],[row(edit=1,x=98)],p0,p14,view=0)==[]

def test_zero_motion_not_falsely_claimed_as_moving_source():
    p0,_=synthetic_projected_triangles()
    out=transport_anchor_rows([row()],[row()],p0,p0,view=0)
    assert len(out)==1 and np.isclose(out[0]["screen_displacement_px"],0.0)

def test_missing_pose0_face_coverage_cannot_anchor_a_pixel():
    p0,p14=synthetic_projected_triangles()
    bad=np.zeros_like(p0)
    with pytest.raises(ValueError,match="ORIGINAL_POSE0_CERTIFIED_CENTER_OUTSIDE"):
        transport_anchor_rows([row()],[row(x=98)],bad,p14,view=0)
