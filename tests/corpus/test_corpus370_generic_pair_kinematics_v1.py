"""Generic source-topology + observational pose identifiability regressions.

A connected *single deforming drawable* must not accidentally create a
different-drawable negative. A disconnected pair carried by a common root
must not create a same-drawable positive. Privileged IDs are addresses only.
"""
from __future__ import annotations

import numpy as np
import pytest

from tools.corpus.diagnose_corpus370_generic_pair_kinematics_v1 import analyze


def source():
    # Two connected coplanar triangles sharing a true edge.
    vertices = np.array([
        [0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.],
    ])
    faces = np.array([[0,1,2],[2,1,3]],dtype=np.int32)
    return vertices,faces


def pair(face_a=0,face_b=1,view=0,pixel_x=4):
    return {
        "locus_a":{"face_index_teacher_only":face_a,
                   "barycentric_teacher_only":[.2,.4,.4]},
        "locus_b":{"face_index_teacher_only":face_b,
                   "barycentric_teacher_only":[.4,.2,.4]},
        "view_index":view,"pixel_x":pixel_x,"pixel_y":7,
    }


def measure(vertices,faces,posed,pairs=None,**kw):
    return analyze(
        vertices,faces,posed,[pair()] if pairs is None else pairs,
        source_sha256="a"*64,rest_geometry_sha256="b"*64,
        authored_teacher_sha256="c"*64,independent_visibility_sha256="d"*64,
        authored_rest_vertices=kw.pop("authored_rest_vertices",vertices),
        authored_faces=kw.pop("authored_faces",faces),
        **kw,
    )


def test_common_global_rigid_rotation_cannot_mint_same_drawable():
    v,f=source()
    angle=np.pi/2
    R=np.array([[np.cos(angle),-np.sin(angle),0.],
                [np.sin(angle),np.cos(angle),0.],
                [0.,0.,1.]])
    posed=np.stack([v, v @ R.T + [3.,-2.,1.]])
    result=measure(v,f,posed)
    entry=result["rows"][0]
    assert entry["topology"]["shared_vertex_count"]==2
    assert entry["topology"]["manifold_edge_connected"] is True
    assert entry["authored_kinematics"]["relative_deformation_observed"] is False
    assert result["qualified_same_drawable_positives"]==0
    assert result["qualified_different_drawable_negatives"]==0
    assert entry["same_drawable"]=="UNKNOWN"


def test_single_connected_skinned_surface_can_bend_without_different_owner():
    v,f=source()
    moved=v.copy()
    moved[3]+=[0.,0.,.8]
    result=measure(v,f,np.stack([v,moved]))
    entry=result["rows"][0]
    assert entry["topology"]["surface_edge_connected"] is True
    assert entry["authored_kinematics"]["relative_deformation_observed"] is True
    assert entry["authored_kinematics"]["status"].startswith("OBSERVED_")
    # One skinned drawable may bend. Relative motion proves no drawable split.
    assert entry["different_drawable"]=="UNKNOWN"
    assert entry["mechanical_owner"]=="UNKNOWN"


def test_disconnected_sources_carrying_same_global_rig_remain_unknown():
    v,f=source()
    vertices=np.concatenate([v,v+[3.,0.,0.]])
    faces=np.array([[0,1,2],[4,5,6]],dtype=np.int32)
    pose=np.stack([vertices,vertices+[2.,0.,3.]])
    result=measure(vertices,faces,pose)
    row=result["rows"][0]
    assert row["topology"]["vertex_connected"] is False
    assert row["topology"]["manifold_edge_connected"] is False
    assert row["authored_kinematics"]["relative_deformation_observed"] is False
    assert row["same_drawable"]=="UNKNOWN"


def test_point_only_contact_is_not_an_edge():
    v,f=source()
    vertices=np.concatenate([v,[[0.,-1.,0.],[1.,-1.,0.]]])
    faces=np.array([[0,1,2],[0,4,5]],dtype=np.int32)
    result=measure(vertices,faces,np.stack([vertices,vertices]))
    row=result["rows"][0]["topology"]
    assert row["shared_vertex_count"]==1
    assert row["vertex_connected"] is True
    assert row["surface_edge_connected"] is False


def test_nonmanifold_shared_edge_is_not_manifold_connection():
    vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],
                       [0.,-1.,0.],[0.,0.,1.]])
    faces=np.array([[0,1,2],[1,0,3],[0,1,4]],dtype=np.int32)
    result=measure(vertices,faces,np.stack([vertices,vertices]))
    row=result["rows"][0]["topology"]
    assert row["surface_edge_connected"] is True
    assert row["manifold_edge_connected"] is False


def test_teacher_vertex_order_drift_fails_closed():
    v,f=source()
    teacher=v.copy()
    teacher[[0,1]]=teacher[[1,0]]
    with pytest.raises(ValueError,match="TEACHER_REST_OR_FACE_ORDER_DRIFT"):
        measure(v,f,np.stack([v,v]),authored_rest_vertices=teacher)


def test_teacher_face_order_drift_fails_closed():
    v,f=source()
    swapped=f[::-1]
    with pytest.raises(ValueError,match="TEACHER_REST_OR_FACE_ORDER_DRIFT"):
        measure(v,f,np.stack([v,v]),authored_faces=swapped)


def test_nonfinite_poses_rejected():
    v,f=source()
    posed=np.stack([v,v])
    posed[1,1,2]=np.nan
    with pytest.raises(ValueError,match="POSE_SHAPE_OR_NONFINITE"):
        measure(v,f,posed)


def test_bad_barycentric_and_source_face_rejected():
    v,f=source()
    invalid=pair()
    invalid["locus_a"]["barycentric_teacher_only"]=[.2,.2,.2]
    with pytest.raises(ValueError,match="BARYCENTRIC_INVALID"):
        measure(v,f,np.stack([v,v]),[invalid])
    invalid=pair()
    invalid["locus_a"]["face_index_teacher_only"]=999
    with pytest.raises(ValueError,match="FACE_ADDRESS_RANGE"):
        measure(v,f,np.stack([v,v]),[invalid])


def test_duplicate_ray_pixel_and_self_face_rejected():
    v,f=source()
    with pytest.raises(ValueError,match="DUPLICATE_SOURCE_PIXEL"):
        measure(v,f,np.stack([v,v]),[pair(),pair()])
    with pytest.raises(ValueError,match="SELF_FACE"):
        measure(v,f,np.stack([v,v]),[pair(face_a=0,face_b=0)])


def test_degenerate_pose_causes_abstain_not_negative_drawable():
    v,f=source()
    degenerate=v.copy()
    degenerate[3]=degenerate[1]
    result=measure(v,f,np.stack([v,degenerate]))
    assert result["rows"][0]["authored_kinematics"]["status"]=="ABSTAIN_DEGENERATE_POSED_TRIANGLE"
    assert result["degenerate_abstain_count"]==1
    assert result["qualified_different_drawable_negatives"]==0


def test_missing_motion_cannot_make_negative():
    v,f=source()
    with pytest.raises(ValueError,match="POSE_SHAPE"):
        measure(v,f,v[None])


def test_invalid_sha_binding_rejected():
    v,f=source()
    with pytest.raises(ValueError,match="PAIR_PARENT_SHA"):
        analyze(v,f,np.stack([v,v]),[pair()],source_sha256="bad",
                rest_geometry_sha256="b"*64,
                authored_teacher_sha256="c"*64,
                independent_visibility_sha256="d"*64,
                authored_rest_vertices=v,authored_faces=f)
