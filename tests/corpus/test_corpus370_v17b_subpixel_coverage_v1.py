"""V17b 4x4 physical orthographic subpixel ray redteam.

Synthetic only; source art and real agent/editor owner labels never appear.
"""
from __future__ import annotations
import copy
import numpy as np
import pytest

from models.iris.truth.graphics_oracle_v1 import OrthographicCameraV1,measure_orthographic_ray_stack_v1
from tools.corpus.measure_corpus370_v17b_original_subpixel_coverage_v1 import (
    SCHEMA, X_OFFSETS,Y_OFFSETS,N_SAMPLES,offset_camera,ray_sample
)
from tools.corpus.court_corpus370_v17b_subpixel_coverage_v1 import (
    SCHEMA as COURT,validate_source_subpixel_world_point
)


def camera():
    return OrthographicCameraV1(
        origin=(0.,0.,3.),forward=(0.,0.,-1.),
        right=(1.,0.,0.),screen_up=(0.,1.,0.),
        half_extent=1.,resolution=4,view_index=0
    )


def thin_source():
    # Pixel centre (-.25,-.25) misses; a deterministic subpixel
    # (-.375,+.125) ray falls inside a real tiny projected triangle.
    world=np.array([
        [-.49,-.36,0.],[-.38,-.36,0.],[-.435,-.255,0.]
    ],np.float64)
    return world,np.array([[0,1,2]],np.int64)


def test_subpixel_grid_is_prepromised_not_response_conditioned():
    assert SCHEMA.endswith(".v17b") and COURT.endswith(".v17b")
    assert X_OFFSETS==(-.375,-.125,.125,.375)
    assert Y_OFFSETS==(-.375,-.125,.125,.375)
    assert len({(x,y) for y in Y_OFFSETS for x in X_OFFSETS})==N_SAMPLES==16


def test_camera_shift_same_source_pixel_offsets_preserves_depth_and_direction():
    cam=camera()
    moved=offset_camera(cam,-.375,+.125)
    assert moved.right==cam.right
    assert moved.screen_up==cam.screen_up
    assert moved.forward==cam.forward
    assert abs(moved.origin[0] + .1875)<1e-12
    assert abs(moved.origin[1] + .0625)<1e-12
    assert moved.origin[2]==cam.origin[2]


def test_real_geometric_subpixel_hit_recovers_center_missed_triangle():
    cam=camera();verts,faces=thin_source()
    center=measure_orthographic_ray_stack_v1(
        verts,faces,cam,1,2,tie_epsilon=1e-7)
    assert not center.hits and center.complete
    rows=[ray_sample(verts,faces,cam,1,2,dx,dy,1e-7)
          for dy in Y_OFFSETS for dx in X_OFFSETS]
    assert len(rows)==16
    hits=[r for r in rows if r["qualified_subpixel_geometric_hit"]]
    assert len(hits)>=1
    for h in hits:
        assert h["first_source_face_ordinal_teacher_only"]==0
        validate_source_subpixel_world_point(
            verts,faces,cam,1,2,h["subpixel_offset_x"],
            h["subpixel_offset_y"],
            h["first_source_face_ordinal_teacher_only"],
            h["first_hit_barycentric_teacher_only"],
            h["first_hit_world_depth"],2.)


@pytest.mark.parametrize("attack",[
    "invalid_face","wrong_depth","nonfinite_depth","wrong_barycentric",
    "bary_outside","bad_bary_sum","wrong_subpixel_x","wrong_subpixel_y"
])
def test_court_rejects_forged_subpixel_3d_source_hit(attack):
    cam=camera();verts,faces=thin_source()
    correct=ray_sample(verts,faces,cam,1,2,-.375,+.125,1e-7)
    assert correct["qualified_subpixel_geometric_hit"]
    row=copy.deepcopy(correct)
    if attack=="invalid_face":
        row["first_source_face_ordinal_teacher_only"]=1
    elif attack=="wrong_depth":
        row["first_hit_world_depth"]+=.2
    elif attack=="nonfinite_depth":
        row["first_hit_world_depth"]=float("nan")
    elif attack=="wrong_barycentric":
        row["first_hit_barycentric_teacher_only"]=[1.,0.,0.]
    elif attack=="bary_outside":
        row["first_hit_barycentric_teacher_only"]=[-1.,1.,1.]
    elif attack=="bad_bary_sum":
        row["first_hit_barycentric_teacher_only"]=[.3,.3,.3]
    elif attack=="wrong_subpixel_x":
        row["subpixel_offset_x"]=.125
    elif attack=="wrong_subpixel_y":
        row["subpixel_offset_y"]=-.375
    with pytest.raises(ValueError):
        validate_source_subpixel_world_point(
            verts,faces,cam,1,2,
            row["subpixel_offset_x"],row["subpixel_offset_y"],
            row["first_source_face_ordinal_teacher_only"],
            row["first_hit_barycentric_teacher_only"],
            row["first_hit_world_depth"],2.)


@pytest.mark.parametrize("x,y",[(-.5001,0.),(.5001,0.),(0.,-.501),(0.,.501)])
def test_offset_cannot_escape_source_pixel_cell(x,y):
    with pytest.raises(ValueError):
        offset_camera(camera(),x,y)


def test_fixed_4x4_subrays_do_not_claim_artist_or_alpha():
    cam=camera();verts,faces=thin_source()
    rows=[ray_sample(verts,faces,cam,1,2,x,y,1e-7)
          for y in Y_OFFSETS for x in X_OFFSETS]
    assert all("artist" not in str(k).lower() and "material_name" not in k
               for r in rows for k in r)
    assert all("alpha" not in k for r in rows for k in r)
    assert any(r["qualified_subpixel_geometric_hit"] for r in rows)
