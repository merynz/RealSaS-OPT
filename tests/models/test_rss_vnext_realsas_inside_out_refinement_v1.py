from __future__ import annotations

import pytest
import torch

from models.rss_vnext.realsas_joint_slrm_render_v1 import RealSaSFlexiCubesLayerRasterV1
from models.rss_vnext.realsas_inside_out_refinement_v1 import (
    RealSaSGeneratedMeshGuidanceV1,
    realsas_outer_inner_collision_loss_v1,
    realsas_generated_refinement_objective_v1,
    refine_realsas_one_layer_v1,
)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(30)
    torch.set_num_threads(1)


def _guidance():
    return RealSaSGeneratedMeshGuidanceV1(
        expected_mask=torch.ones((1,4,1)),
        expected_normal_world=torch.tensor([[[0.,0.,1.]]*4]),
        known_mask=torch.ones((1,4),dtype=torch.bool),
        normal_valid_mask=torch.ones((1,4),dtype=torch.bool),
        generation_checkpoint_sha256="a"*64,
        original_source_binding_sha256="b"*64,
    )


def _raster(vertices,layer="outer"):
    depth=vertices[:,2].mean()
    normal=torch.stack((torch.zeros_like(depth),torch.zeros_like(depth),
                        torch.ones_like(depth)))[None,None].expand(1,4,3)
    return RealSaSFlexiCubesLayerRasterV1(
        rgb_white_composite=torch.ones((1,4,3)),
        alpha=vertices[:,2].mean().sigmoid().expand(1,4,1),
        semantic_posterior=torch.ones((1,4,2))*.5,
        normal_xyz=normal,
        depth=depth.expand(1,4,1),
        flexicubes_deformation_regularizer=vertices.square().mean(),
        source_scene_prediction_sha256="c"*64,
        mesh_surface_prediction_sha256="d"*64,
        semantic_layer_name=layer,
        rasterizer_identity="CPU_TEST_RENDERER_NOT_PRODUCT_GPU",
    )


def test_exact_paper_outer_inside_collision_is_zero_when_outside_and_positive_when_penetrating():
    inside=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    normal=torch.tensor([[0.,0.,1.]]*3)
    outer_inside=inside.clone()
    outer_inside[:,2]=-.2
    outer_inside.requires_grad_(True)
    score=realsas_outer_inner_collision_loss_v1(outer_inside,inside,normal)
    assert torch.allclose(score,torch.tensor(.008),atol=1.e-6)
    score.backward()
    assert outer_inside.grad[:,2].lt(0).all()
    outside=inside.clone()
    outside[:,2]=.2
    assert realsas_outer_inner_collision_loss_v1(outside,inside,normal)==0.


def test_refinement_optimizes_outer_vertices_but_cannot_change_fixed_inner():
    inner=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    frozen=inner.clone()
    normal=torch.tensor([[0.,0.,1.]]*3)
    initial=inner.clone()
    initial[:,2]=-.2
    refined=refine_realsas_one_layer_v1(
        layer_name="outer",initial_vertices_xyz=initial,
        face_indices=torch.tensor([[0,1,2]]),guidance=_guidance(),
        rasterize_vertices=_raster,
        fixed_inner_xyz=inner,fixed_inner_normals=normal,
        steps=20,lr=.02,mask_weight=0.,normal_weight=0.,collision_weight=10.,
    )
    assert refined.refined_xyz[:,2].mean()>initial[:,2].mean()
    assert torch.equal(inner,frozen)
    assert refined.authority=="OPTIMIZED_GENERATED_LAYER_GEOMETRY_NOT_OBSERVED_SOURCE"


def test_generated_normal_mask_objective_is_differentiable_but_not_source_truth():
    verts=torch.randn((3,3),requires_grad=True)
    raster=_raster(verts)
    terms=realsas_generated_refinement_objective_v1(
        raster=raster,guidance=_guidance(),current_outer_xyz=verts,
    )
    terms["total"].backward()
    assert verts.grad is not None and verts.grad.abs().sum()>0
    assert "normal_l2" in terms and "mask_mse" in terms
    assert _guidance().authority.endswith("NOT_SOURCE_TRUTH")


def test_invalid_source_authority_or_unknown_normal_frame_rejected():
    from dataclasses import replace
    g=_guidance()
    with pytest.raises(ValueError,match="GENERATED_PRIOR_REQUIRED"):
        replace(g,authority="OBSERVED_SOURCE_GEOMETRY").validate()
    with pytest.raises(ValueError,match="GENERATED_PRIOR_REQUIRED"):
        replace(g,normal_frame="CAMERA_LOCAL_NOT_TRANSFORMED").validate()


def test_malformed_layer_raster_never_optimizes_other_identity():
    verts=torch.ones((3,3))
    with pytest.raises(ValueError,match="LAYER_RASTER_MISMATCH"):
        refine_realsas_one_layer_v1(
            layer_name="outer",initial_vertices_xyz=verts,
            face_indices=torch.tensor([[0,1,2]]),
            guidance=_guidance(),
            rasterize_vertices=lambda v:_raster(v,layer="not-the-outer"),
            steps=1,
        )
