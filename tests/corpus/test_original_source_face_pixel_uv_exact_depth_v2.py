"""Exact projected source local depth proof adversarial tests (synthetic only)."""
import numpy as np
import pytest
from shapely.geometry import Polygon, box

from tools.corpus.measure_original_source_face_pixel_uv_exact_depth_v2 import (
    affine_source_depth, prepare_visible_source_faces,
    possible_source_occluder_intersection, exact_depth_original_face_uv_rows,
)
from tools.corpus.court_original_source_face_pixel_uv_exact_depth_v2 import (
    clip_triangle_against_guarded_pixel, direct_source_triangles,
    verify_one_row,
)
from tools.corpus.measure_original_source_face_pixel_uv_causality_v1 import (
    certified_face_uv_pixel_rows,
)


def camera():
    return {
        "view_index": 1,
        "schema_version": "RealSaS.FullSurfaceCameraProjection.v3",
        "origin": [0.0,0.0,0.0],
        "right": [1.0,0.0,0.0],
        "screen_up": [0.0,1.0,0.0],
        "forward": [0.0,0.0,1.0],
        "half_extent": 1.0,
        "clip_start": .1, "clip_end": 3.,
    }


def source_geometry():
    # Front triangle includes central pixel completely. Back triangle same
    # projected shape, different source depth, no artist IDs.
    xyz=np.array([
        (-.8,-.8,1),(.8,-.8,1),(0,.8,1),
        (-.8,-.8,2),(.8,-.8,2),(0,.8,2)
    ],np.float64)
    face=np.array([[0,1,2],[3,4,5]],np.int64)
    uv=np.array([[[.79,.79],[.80,.79],[.80,.80]]]*2,np.float64)
    return xyz,face,uv,[(384,384,448,448)]


def test_v2_original_face_certification_preserves_v1_geometry_certificate():
    xyz,face,uv,rect=source_geometry()
    v1_rows,_=certified_face_uv_pixel_rows(xyz,face,uv,camera(),rect)
    v2_rows,_=exact_depth_original_face_uv_rows(xyz,face,uv,camera(),rect)
    k=lambda rows:{(r["face_index"],r["pixel_y_top_first"],
                    r["pixel_x"],r["source_uv_edit_operation_index"])
                   for r in rows}
    assert k(v1_rows)<=k(v2_rows)
    assert any(r["face_index"]==0 and r["pixel_x"]==96 and
               r["pixel_y_top_first"]==96 for r in v2_rows)


def test_nearer_source_triangle_occludes_entire_original_pixel():
    xyz,face,uv,rect=source_geometry()
    xyz[3:,2]=.4
    rows,_=exact_depth_original_face_uv_rows(xyz,face,uv,camera(),rect)
    selected=[r for r in rows if r["pixel_x"]==96 and
              r["pixel_y_top_first"]==96]
    assert len(selected)==1 and selected[0]["face_index"]==1


def test_degenerate_projected_source_face_is_conservatively_excluded():
    tri=np.array([
        [[0,0],[192,0],[96,192]],
        [[96,95],[96,97],[96,96]],
    ],np.float64)
    depth=np.array([[1,1,1],[.5,.5,.5]],np.float64)
    tree,polys,index,deg=prepare_visible_source_faces(tri)
    assert list(deg)==[1]
    assert not possible_source_occluder_intersection(
        0,tri,depth,tree,polys,index,deg,pixel_x=96,pixel_y=96)


def test_projected_occluder_only_counts_when_strictly_nearer_on_actual_overlap():
    # Sloping other triangle has its globally nearest vertex OUTSIDE Q,
    # but its only local pixel contribution is behind original source face.
    front=np.array([[80.,80.],[113.,80.],[96.,113.]])
    other=np.array([[90.,90.],[107.,90.],[107.,107.]])
    tri=np.array([front,other])
    depth=np.array([[1.,1.,1.],[.5,2.,2.]])
    tree,polys,index,deg=prepare_visible_source_faces(tri)
    assert depth[1].min()<depth[0].max()  # V1 global envelope rejects
    assert possible_source_occluder_intersection(
        0,tri,depth,tree,polys,index,deg,pixel_x=96,pixel_y=96)
    # Exact direct independent clipping agrees for this candidate.
    scene={"face_index":0,"pixel_x":96,"pixel_y_top_first":96,
           "source_uv_edit_operation_index":0}
    rects=[[384,384,448,448]]
    uv=np.array([[[.8,.8]]*3,[[.8,.8]]*3],np.float64)
    # Origin camera projection irrelevant to direct 2D proof; test custom
    # clip independently instead of artificially inventing source XYZ.
    clip=clip_triangle_against_guarded_pixel(other,95.95,97.05,95.95,97.05)
    assert len(clip)>=3
    points=np.asarray(clip)
    from tools.corpus.court_original_source_face_pixel_uv_exact_depth_v2 import affine_depth
    z0,_=affine_depth(front,depth[0],points)
    z1,_=affine_depth(other,depth[1],points)
    assert np.min(z1-z0)>1e-5


def test_independent_direct_rectangle_clip_matches_shapely_random_convex_geometry():
    rng=np.random.default_rng(79)
    for _ in range(200):
        tri=rng.uniform(-4,4,(3,2))
        x,y=rng.uniform(-1,1,2)
        clipped=clip_triangle_against_guarded_pixel(
            tri,x-.05,x+1.05,y-.05,y+1.05)
        exact=Polygon(tri).intersection(box(x-.05,y-.05,x+1.05,y+1.05))
        expected_area=float(exact.area)
        measured_area=float(Polygon(clipped).area) if len(clipped)>=3 else 0.
        assert np.isclose(measured_area,expected_area,atol=1e-8,rtol=0)


def test_source_depth_truth_cannot_be_warped_to_fit_physical_samples():
    triangle=np.array([[0.,0.],[1.,0.],[0.,1.]])
    z=np.array([.5,1.5,2.5])
    samples=np.array([[.25,.25],[.1,.4]])
    a=affine_source_depth(samples,triangle,z)
    assert np.allclose(a,[1.25,1.4])
    with pytest.raises(ValueError,match="DEGENERATE_PROJECTED"):
        affine_source_depth(samples,np.zeros((3,2)),z)
