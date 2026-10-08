import numpy as np
import pytest

from compiler.realsas_compiler_core.camera_geometry_v2 import qualify_camera_v3
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_core.visual_motion_blend_v1 import (
    build_motion_blend_coefficients, evaluate_motion_blend, canonical_pose_palette_residual,
)


def fixture():
    p = np.array([[-1.,-1],[1,-1],[1,1],[-1,1],[0,0]])
    faces = np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]])
    binding = {"rest_positions":p,"vertex_region_id":np.zeros(5,dtype=int),
        "domain_id":np.zeros(5,dtype=int),"anchor_vertex":np.arange(4),
        "anchor_mechanical_vertices":np.tile([0,1,2],(4,1)),
        "anchor_barycentric":np.array([[1.,0,0],[0,1,0],[0,1,0],[1,0,0]]),
        "mechanical_rest_xyz":np.array([[-1.,-1,0],[1,-1,0],[-1,1,0]])}
    weights = np.array([[1.,0],[0,1],[.5,.5]])
    camera = qualify_camera_v3(dict(origin=[0,0,-2],right=[1,0,0],screen_up=[0,-1,0],
        forward=[0,0,1],half_extent=4,resolution=8),view_id="V0",view_index=0)
    return binding,faces,weights,camera


def test_harmonic_motion_ownership_is_convex_and_does_not_modify_canonical_skin():
    b,f,w,_ = fixture();original=w.copy()
    blend=build_motion_blend_coefficients(b,visual_faces=f,mechanical_weights=w)
    np.testing.assert_allclose(blend,[[1,0],[0,1],[0,1],[1,0],[.5,.5]])
    np.testing.assert_array_equal(w,original)


def test_blended_pose_uses_shared_bone_rotation_and_translation_with_exact_depth():
    b,f,w,c=fixture();blend=build_motion_blend_coefficients(b,visual_faces=f,mechanical_weights=w)
    matrices=np.tile(np.eye(4),(2,1,1));matrices[0,:3,3]=[1,0,0];matrices[1,:3,3]=[0,1,0]
    field=np.c_[b['rest_positions']+100,np.arange(5)+2.]
    out=evaluate_motion_blend(field,rest_source_xy=b['rest_positions'],coefficients=blend,
        axis_positions_source=np.zeros((2,3)),skin_matrices_source=matrices,camera=c)
    np.testing.assert_allclose(out[:,:2],b['rest_positions']+[[1,0],[0,1],[0,1],[1,0],[.5,.5]])
    np.testing.assert_array_equal(out[:,2],field[:,2])


def test_out_of_plane_rotation_is_not_a_collapse_of_authored_2d_rigid_art():
    b,f,w,c=fixture();blend=np.tile([1.,0],(5,1));m=np.tile(np.eye(4),(2,1,1))
    m[0,:3,:3]=[[1,0,0],[0,0,-1],[0,1,0]]
    out=evaluate_motion_blend(np.c_[b['rest_positions']*0,np.ones(5)],rest_source_xy=b['rest_positions'],
        coefficients=blend,axis_positions_source=np.zeros((2,3)),skin_matrices_source=m,camera=c)
    np.testing.assert_allclose(out[:,:2],b['rest_positions'])


def test_palette_must_reconstruct_the_sealed_mechanical_witness():
    b,f,w,_=fixture();m=np.tile(np.eye(4),(2,1,1));m[0,:3,3]=[1,0,0];m[1,:3,3]=[0,1,0]
    rest=b['mechanical_rest_xyz'];posed=rest+np.c_[w[:,0],w[:,1],np.zeros(3)]
    assert canonical_pose_palette_residual(rest_xyz=rest,posed_xyz=posed,mechanical_weights=w,skin_matrices_source=m)==0
    m[0,0,3]+=1
    assert canonical_pose_palette_residual(rest_xyz=rest,posed_xyz=posed,mechanical_weights=w,skin_matrices_source=m)==1


@pytest.mark.parametrize('bad', ['negative','nonfinite','sum'])
def test_invalid_canonical_skin_cannot_become_visual_motion_authority(bad):
    b,f,w,_=fixture()
    if bad=='negative':w[0]=[-1,2]
    elif bad=='nonfinite':w[0,0]=np.nan
    else:w[0,0]=.5
    with pytest.raises(QualificationError,match='CANONICAL_COEFFICIENTS_INVALID'):
        build_motion_blend_coefficients(b,visual_faces=f,mechanical_weights=w)
