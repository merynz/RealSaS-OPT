from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest
import torch

from models.rss_vnext.stdgen_source_layer_truth_v1 import (
    StdGENSourceOpaqueTriangleSurfaceV1,
    render_stdgen_source_opaque_peeled_layer_v1,
)
from models.rss_vnext.stdgen_joint_slrm_render_v1 import STAGE_1,STAGE_2


def source():
    # Identical projected triangles, red outer surface at z=.4 and
    # blue inner source geometry at z=0. Removing role0 reveals role1.
    verts=np.array([
        [-.6,-.6,.4],[.6,-.6,.4],[0,.7,.4],
        [-.6,-.6,0.],[.6,-.6,0.],[0,.7,0.],
    ],dtype=np.float32)
    return StdGENSourceOpaqueTriangleSurfaceV1(
        source_mesh_xyz=verts,
        source_faces=np.array([[0,1,2],[3,4,5]],dtype=np.int64),
        source_face_rgb_linear=np.array([[1,0,0],[0,0,1]],dtype=np.float32),
        source_face_role=np.array([0,1],dtype=np.int64),
        semantic_role_count=2,
        source_mesh_truth_sha256="1"*64,
        source_rights_admission_sha256="2"*64,
        renderer_version_sha256="3"*64,
        source_colors_from_artist_materials=True,
        source_roles_independently_assigned=True,
        source_surface_is_opaque=True,
    )


def camera():
    cam=np.zeros(13,dtype=np.float32)
    cam[:3]=[0,0,2]
    cam[3:6]=[-1,0,0]
    cam[6:9]=[0,1,0]
    cam[9:12]=[0,0,-1]
    cam[12]=1.
    return cam


def test_real_source_zbuffer_reveals_occluded_inner_geometry_after_peeling():
    src=source()
    full=render_stdgen_source_opaque_peeled_layer_v1(
        source=src,camera_13d=camera(),width=16,height=16,role_subset=(0,1),
    )
    back=render_stdgen_source_opaque_peeled_layer_v1(
        source=src,camera_13d=camera(),width=16,height=16,role_subset=(1,),
    )
    hit=np.argwhere(full.visible_valid_mask & back.visible_valid_mask)
    assert len(hit)>10
    iy,ix=map(int,hit[len(hit)//2])
    assert full.role_indices[iy,ix]==0
    assert back.role_indices[iy,ix]==1
    assert np.array_equal(full.rgb_white[iy,ix],np.array([1,0,0],dtype=np.float32))
    assert np.array_equal(back.rgb_white[iy,ix],np.array([0,0,1],dtype=np.float32))
    assert full.metric_depth[iy,ix,0]<back.metric_depth[iy,ix,0]
    assert full.layer_source_render_sha256!=back.layer_source_render_sha256
    t1=full.as_training_target(stage=STAGE_1,layer_name="all")
    t2=back.as_training_target(stage=STAGE_2,layer_name="without_cover")
    assert t1.rgb_white.shape==(1,256,3)
    assert t1.qualified_role_mask.sum()>10
    assert t2.role_indices[t2.qualified_role_mask].eq(1).all()


def test_source_qualifications_and_3d_triangle_budget_fail_closed():
    src=source()
    with pytest.raises(ValueError,match="TRIANGLE_TRUTH_NOT_QUALIFIED"):
        render_stdgen_source_opaque_peeled_layer_v1(
            source=replace(src,source_colors_from_artist_materials=False),
            camera_13d=camera(),width=16,height=16,role_subset=(0,),
        )
    with pytest.raises(ValueError,match="CAMERA_OR_BUDGET"):
        render_stdgen_source_opaque_peeled_layer_v1(
            source=src,camera_13d=camera(),width=1024,height=1024,role_subset=(1,),
        )
    with pytest.raises(ValueError,match="CAMERA_OR_BUDGET"):
        render_stdgen_source_opaque_peeled_layer_v1(
            source=src,camera_13d=camera(),width=16,height=16,role_subset=(7,),
        )


def test_generated_views_do_not_create_source_surface_truth():
    src=source()
    a=render_stdgen_source_opaque_peeled_layer_v1(
        source=src,camera_13d=camera(),width=16,height=16,role_subset=(1,),
    )
    assert a.is_generated_diffusion_prediction is False
    assert a.source_is_watertight_signed_mesh is False
    assert a.scene_source_binding_sha256==src.source_mesh_truth_sha256
    assert torch.isfinite(a.as_training_target(stage=STAGE_2,layer_name="peel").rgb_white).all()


def test_unmodelled_occluder_or_transparency_is_not_silently_flattened():
    src=source()
    for changed in (
        replace(src,source_surface_is_opaque=False),
        replace(src,source_scene_contains_unmodeled_occluders=True),
        replace(src,source_roles_independently_assigned=False),
    ):
        with pytest.raises(ValueError,match="TRIANGLE_TRUTH_NOT_QUALIFIED"):
            render_stdgen_source_opaque_peeled_layer_v1(
                source=changed,camera_13d=camera(),width=16,height=16,role_subset=(0,1),
            )
