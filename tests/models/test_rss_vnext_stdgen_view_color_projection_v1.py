from __future__ import annotations
import numpy as np
import pytest
import torch
from models.rss_vnext.stdgen_view_color_projection_v1 import (
    project_stdgen_generated_view_colors_v1,RASTER_ID,_mesh_digest,
)


def fixtures():
    vertices=np.array([[-.5,-.5,0.],[.5,-.5,0.],[0.,.5,0.],[0.,0.,0.]],dtype=np.float32)
    faces=np.array([[0,1,3],[1,2,3],[2,0,3]],dtype=np.int64)
    normals=np.repeat(np.array([[0.,0.,-1.]],dtype=np.float32),4,axis=0)
    rgb=np.zeros((6,8,8,3),dtype=np.float32)
    rgb[:]=np.array([0.2,0.3,0.4],dtype=np.float32)
    rgb[0]=np.array([.5,.2,.1],dtype=np.float32)
    alpha=np.ones((6,8,8),dtype=np.float32)
    mesh_alpha=np.ones((6,8,8),dtype=np.float32)
    mesh_depth=np.full((6,8,8),2,dtype=np.float32)
    cams=np.zeros((6,13),dtype=np.float32)
    for i in range(6):
        theta=i*np.pi/3
        forward=np.array([np.sin(theta),0.,np.cos(theta)])
        right=np.array([np.cos(theta),0.,-np.sin(theta)])
        cams[i,:3]=-forward*2
        cams[i,3:6]=right
        cams[i,6:9]=[0,1,0]
        cams[i,9:12]=forward
        cams[i,12]=1
    digest=_mesh_digest(torch.from_numpy(vertices[None]),torch.from_numpy(faces))
    return dict(vertices=vertices,faces=faces,vertex_normals=normals,
        generated_srgb=rgb,generated_alpha=alpha,generated_camera13=cams,
        mesh_raster_depth=mesh_depth,mesh_raster_alpha=mesh_alpha,
        mesh_parent_sha256=digest,raster_mesh_parent_sha256=digest,
        rasterizer_identity=RASTER_ID,source_binding_sha256='a'*64,
        generating_checkpoint_sha256='b'*64)


def test_best_facing_donor_srgb_linear_and_mesh_identity():
    f=fixtures()
    out=project_stdgen_generated_view_colors_v1(**f)
    assert out.vertex_valid[3]
    assert out.selected_view[3]==0
    assert np.isclose(out.vertex_rgb_linear[3,0],0.21404114,atol=1e-6)
    assert out.vertex_rgb_linear.dtype==np.float32
    assert out.mesh_parent_sha256==f['mesh_parent_sha256']
    assert out.authority=='PREDICTED_GENERATED_VIEW_COLOR_CANDIDATE_NOT_SOURCE_PIXELS'


def test_occluded_unknown_and_remesh_forces_rebinding():
    f=fixtures()
    f['mesh_raster_alpha'][:]=0
    out=project_stdgen_generated_view_colors_v1(**f)
    assert not out.vertex_valid.any()
    assert np.all(out.selected_view==-1)
    assert not out.vertex_rgb_linear.any()
    f=fixtures();f['vertices'][3,2]+=.1
    with pytest.raises(ValueError,match='MESH_PARENT_DRIFT'):
        project_stdgen_generated_view_colors_v1(**f)


def test_generated_image_cannot_rewrite_provenance_or_lie_about_raster():
    f=fixtures();f['rasterizer_identity']='UNVERIFIED'
    with pytest.raises(ValueError,match='SOURCE_OR_RENDER_DOMAIN'):
        project_stdgen_generated_view_colors_v1(**f)
    f=fixtures();f['mesh_raster_depth'][:]=3.4
    out=project_stdgen_generated_view_colors_v1(**f)
    assert not out.vertex_valid.any()
    f=fixtures();f['generated_camera13'][0,11]=2.0
    with pytest.raises(ValueError,match='CAMERA_ORTHONORMAL'):
        project_stdgen_generated_view_colors_v1(**f)
