"""Adversarial full-pixel source-face UV causality proof tests, synthetic only."""
import numpy as np
import pytest

from tools.corpus.measure_original_source_face_pixel_uv_causality_v1 import (
    project_original, triangle_corner_barycentric, guarded_corners,
    certified_face_uv_pixel_rows, physically_court_witness,
)
from tools.corpus.court_original_source_face_pixel_uv_causality_v1 import (
    direct_row_geometry_check,
)


def scene():
    # Orthographic original-camera-shaped pixel geometry: candidate front
    # triangle fully covers central (96,96) pixel with large safety margin.
    vertices=np.array([
        (-.8,-.8,1),(.8,-.8,1),(0,.8,1),
        (-.8,-.8,2),(.8,-.8,2),(0,.8,2),
    ],np.float64)
    faces=np.array([[0,1,2],[3,4,5]],np.int64)
    uv=np.array([
        [[.79,.79],[.80,.79],[.80,.80]],
        [[.79,.79],[.80,.79],[.80,.80]],
    ],np.float64)
    camera={
        "view_index":1,
        "schema_version":"RealSaS.FullSurfaceCameraProjection.v3",
        "origin":[0.,0.,0.],
        "right":[1.,0.,0.],
        "screen_up":[0.,1.,0.],
        "forward":[0.,0.,1.],
        "half_extent":1.,"clip_start":.1,"clip_end":4.0,
    }
    rects=[(384,384,448,448),(384,448,448,512),(448,384,512,448)]
    return vertices,faces,uv,camera,rects


def test_source_face_interior_and_original_uv_support_producer_and_independent():
    vertices,faces,uv,cam,rects=scene()
    rows,census=certified_face_uv_pixel_rows(vertices,faces,uv,cam,rects)
    selected=[r for r in rows if r["pixel_x"]==96 and
              r["pixel_y_top_first"]==96]
    assert len(selected)==1
    assert selected[0]["face_index"]==0
    assert selected[0]["source_uv_edit_operation_index"]==0
    assert census["whole_pixel_unique_nearest_source_face"]>0
    assert direct_row_geometry_check(vertices[faces],uv,cam,rects[0],0,96,96)


def test_closer_original_face_falsifies_the_wrong_front_surface():
    vertices,faces,uv,cam,rects=scene()
    # Previously back triangle moved in front.
    vertices[3:,2]=.5
    with pytest.raises(ValueError,match="ANOTHER_SOURCE_FACE_MIGHT_OCCLUDE"):
        direct_row_geometry_check(vertices[faces],uv,cam,rects[0],0,96,96)
    rows,_=certified_face_uv_pixel_rows(vertices,faces,uv,cam,rects)
    central=[r for r in rows if r["pixel_x"]==96 and
             r["pixel_y_top_first"]==96]
    assert len(central)==1 and central[0]["face_index"]==1


def test_source_uv_boundary_is_not_a_certified_same_atlas_edit():
    vertices,faces,uv,cam,rects=scene()
    uv[0,:,0]=.874  # near source tile right edge .875 <1 texel guard
    with pytest.raises(ValueError,match="SOURCE_FACE_UV_CROSSES_EDIT_BOUNDARY"):
        direct_row_geometry_check(vertices[faces],uv,cam,rects[0],0,96,96)


def test_pixel_only_point_center_covered_does_not_imply_entire_pixel():
    vertices,faces,uv,cam,rects=scene()
    # Make face much smaller than 1 physical pixel.
    vertices[0]=(-.003,-.003,1)
    vertices[1]=(.003,-.003,1)
    vertices[2]=(0,.003,1)
    with pytest.raises(ValueError,match="CANDIDATE_DOES_NOT_COVER_ENLARGED_PIXEL"):
        direct_row_geometry_check(vertices[faces],uv,cam,rects[0],0,96,96)
    rows,_=certified_face_uv_pixel_rows(vertices,faces,uv,cam,rects)
    assert not any(r["face_index"]==0 and r["pixel_x"]==96 and
                   r["pixel_y_top_first"]==96 for r in rows)


def test_physical_original_rgba_edited_face_observed_all_samples_and_no_other_edits():
    samples=np.zeros((8,8,192,192,4),np.float32)
    samples[...,3]=1.0
    samples[:,1,95,96,0]=0.2  # Blender bottom-first y = 191-top(96)
    row={"face_index":0,"pixel_y_top_first":96,"pixel_x":96,
         "source_uv_edit_operation_index":0}
    audit=physically_court_witness([row],samples,view_ordinal=1)
    assert audit["independently_edited_sample_incidents"]==8
    assert audit["absent_other_edit_effect_obligations"]==16
    assert audit["source_uv_edit_pixel_counts"]==[1,0,0]


def test_independently_changed_wrong_original_edit_is_a_real_contradiction():
    samples=np.zeros((8,8,192,192,4),np.float32)
    samples[...,3]=1.0
    samples[:,1,95,96,0]=0.2
    samples[:,2,95,96,0]=0.2  # the wrong operation, same pixel
    row={"face_index":0,"pixel_y_top_first":96,"pixel_x":96,
         "source_uv_edit_operation_index":0}
    with pytest.raises(ValueError,match="PHYSICAL_SOURCE_FACE_PIXEL_INTERVENTION_CONTRADICTION"):
        physically_court_witness([row],samples,view_ordinal=1)


def test_source_pixel_origin_reversal_detects_false_sample_read():
    samples=np.zeros((8,8,192,192,4),np.float32)
    samples[...,3]=1.0
    # Wrong row 96 instead of original EXR bottom-first row 95.
    samples[:,1,96,96,0]=0.2
    row={"face_index":0,"pixel_y_top_first":96,"pixel_x":96,
         "source_uv_edit_operation_index":0}
    with pytest.raises(ValueError,match="PHYSICAL_SOURCE_FACE_PIXEL_INTERVENTION_CONTRADICTION"):
        physically_court_witness([row],samples,view_ordinal=1)


def test_wrong_source_camera_basis_and_invalid_uv_are_rejected():
    vertices,faces,uv,cam,rects=scene()
    cam["right"]=[0,0,0]
    with pytest.raises(ValueError,match="SOURCE_CAMERA_ORTHONORMAL_FRAME"):
        project_original(vertices,faces,cam)
    cam["right"]=[1,0,0]
    uv[0,0,1]=float("nan")
    with pytest.raises(ValueError,match="GLTF_SOURCE_FACE_CORNER_UV_DOMAIN"):
        certified_face_uv_pixel_rows(vertices,faces,uv,cam,rects)
