from __future__ import annotations

import torch
import pytest

from models.rss_vnext.realsas_joint_slrm_render_v1 import (
    STAGE_1,STAGE_2,STAGE_3,RealSaSJointTargetV1,
    render_realsas_joint_rays_v1,render_realsas_joint_from_shared_scene_v1,
    RealSaSFlexiCubesLayerRasterV1,
    realsas_joint_stage_loss_v1,
)


def _rays(*,remove_cover=False):
    # Two aligned samples: near outer red and far inner blue.
    # Strong density; removing outer semantic exposes the rear color.
    density=torch.tensor([[[4.,4.]]],requires_grad=True)
    rgb=torch.tensor([[[[6.,-6.,-6.],[-6.,-6.,6.]]]],requires_grad=True)
    roles=torch.tensor([[[[8.,-8.],[-8.,8.]]]],requires_grad=True)
    keep=torch.tensor([[[not remove_cover,True]]],dtype=torch.bool)
    render=render_realsas_joint_rays_v1(
        density_logit=density,rgb_logit=rgb,role_logits=roles,
        ray_delta=torch.ones((1,1,2)),
        ray_depth=torch.tensor([[[.2,.8]]]),
        inside_cube=torch.ones((1,1,2),dtype=torch.bool),
        keep_roles=keep,
    )
    return render,density,rgb,roles


def _target(stage=STAGE_1):
    return RealSaSJointTargetV1(
        rgb_white=torch.rand((1,1,3)),
        alpha=torch.ones((1,1,1)),
        role_indices=torch.tensor([[1]],dtype=torch.long),
        qualified_rgb_mask=torch.ones((1,1),dtype=torch.bool),
        qualified_mask_mask=torch.ones((1,1),dtype=torch.bool),
        qualified_role_mask=torch.ones((1,1),dtype=torch.bool),
        source_3d_truth_binding_sha256="a"*64,
        layer_name="all" if stage==STAGE_1 else "peeled",
        stage=stage,
    )


def test_occluded_surface_peeling_exposes_inner_rgb_and_semantic_and_geometry_gradients():
    all_view,density,rgb,roles=_rays()
    peeled,_,_,_=_rays(remove_cover=True)
    assert all_view.rgb_premultiplied[0,0,0]>all_view.rgb_premultiplied[0,0,2]
    assert peeled.rgb_premultiplied[0,0,2]>peeled.rgb_premultiplied[0,0,0]
    assert peeled.semantic_posterior[0,0,1]>all_view.semantic_posterior[0,0,1]
    losses=realsas_joint_stage_loss_v1(all_view,_target())
    losses["total"].backward()
    assert density.grad.abs().sum()>0
    assert rgb.grad.abs().sum()>0
    assert roles.grad.abs().sum()>0


def test_nerf_multilayer_loss_never_borrows_mesh_stage_normals_depth():
    render,*_=_rays(remove_cover=True)
    loss=realsas_joint_stage_loss_v1(render,_target(STAGE_2))
    assert set(loss)=={"rgb_mse","mask_mse","semantic_ce","total"}
    assert torch.isfinite(loss["total"])
    with pytest.raises(ValueError,match="MESH_TARGET_ON_NERF_STAGE"):
        realsas_joint_stage_loss_v1(
            render,_target(STAGE_2),
            mesh_raster=object(),
        )


def test_stage3_requires_independent_mesh_render_and_all_3_losses():
    render,*_=_rays()
    with pytest.raises(ValueError,match="STAGE3_REQUIRES_FLEXICUBES_MESH_RENDER"):
        realsas_joint_stage_loss_v1(render,_target(STAGE_3))
    mesh=RealSaSFlexiCubesLayerRasterV1(
        rgb_white_composite=torch.tensor([[[.8,.2,.2]]],requires_grad=True),
        alpha=torch.ones((1,1,1)),
        semantic_posterior=torch.tensor([[[.2,.8]]],requires_grad=True),
        normal_xyz=torch.tensor([[[0.,0.,1.]]],requires_grad=True),
        depth=torch.tensor([[[.6]]],requires_grad=True),
        flexicubes_deformation_regularizer=torch.tensor(.2,requires_grad=True),
        source_scene_prediction_sha256="f"*64,
        mesh_surface_prediction_sha256="e"*64,
        semantic_layer_name="peeled",rasterizer_identity="test-fake-not-prod",
    )
    params=dict(
        mesh_raster=mesh,
        target_normal=torch.tensor([[[0.,1.,0.]]]),
        target_depth=torch.tensor([[[.3]]]),
        qualified_normal_mask=torch.ones((1,1),dtype=torch.bool),
        qualified_depth_mask=torch.ones((1,1),dtype=torch.bool),
    )
    loss=realsas_joint_stage_loss_v1(render,_target(STAGE_3),**params)
    assert torch.allclose(loss["normal_cosine"],torch.tensor(1.))
    assert torch.allclose(loss["depth_l1"],torch.tensor(.3))
    assert torch.allclose(loss["flexicubes_deformation"],torch.tensor(.2))
    loss["total"].backward()
    assert mesh.normal_xyz.grad is not None
    assert mesh.depth.grad is not None
    assert mesh.rgb_white_composite.grad.abs().sum()>0
    assert mesh.semantic_posterior.grad.abs().sum()>0


def test_mesh_stage3_does_not_require_any_nerf_ray_prediction():
    from dataclasses import replace
    mesh=RealSaSFlexiCubesLayerRasterV1(
        rgb_white_composite=torch.ones((1,1,3),requires_grad=True)*.4,
        alpha=torch.ones((1,1,1),requires_grad=True)*.8,
        semantic_posterior=torch.tensor([[[.2,.8]]],requires_grad=True),
        normal_xyz=torch.tensor([[[0.,0.,1.]]],requires_grad=True),
        depth=torch.tensor([[[.6]]],requires_grad=True),
        flexicubes_deformation_regularizer=torch.tensor(.1,requires_grad=True),
        source_scene_prediction_sha256="a"*64,mesh_surface_prediction_sha256="b"*64,
        semantic_layer_name="peeled",rasterizer_identity="nvdiffrast-cpu-standin",
    )
    terms=realsas_joint_stage_loss_v1(
        None,_target(STAGE_3),mesh_raster=mesh,
        target_normal=torch.tensor([[[0.,0.,1.]]]),
        target_depth=torch.tensor([[[.5]]]),
        qualified_normal_mask=torch.ones((1,1),dtype=torch.bool),
        qualified_depth_mask=torch.ones((1,1),dtype=torch.bool),
    )
    terms["total"].backward()
    assert torch.isfinite(terms["total"])
    assert mesh.semantic_posterior.grad.abs().sum()>0
    with pytest.raises(ValueError,match="NERF_STAGE_RAYS_REQUIRED"):
        realsas_joint_stage_loss_v1(None,_target(STAGE_1))


def test_generated_pseudotruth_and_bad_role_mask_rejected():
    from dataclasses import replace
    render,*_=_rays()
    with pytest.raises(ValueError,match="TEACHER_NOT_ADMITTED_3D_RENDER"):
        realsas_joint_stage_loss_v1(
            render,replace(_target(),generated_completion_used_as_truth=True))
    with pytest.raises(ValueError,match="JOINT_RAY_GEOMETRY_OR_ROLE_DOMAIN"):
        render_realsas_joint_rays_v1(
            density_logit=torch.ones((1,1,2)),
            rgb_logit=torch.zeros((1,1,2,3)),
            role_logits=torch.ones((1,1,2,2)),
            ray_delta=torch.ones((1,1,2)),ray_depth=torch.ones((1,1,2)),
            inside_cube=torch.ones((1,1,2),dtype=torch.bool),
            keep_roles=torch.zeros((1,1,2),dtype=torch.bool),
        )


class _MiniIRIS:
    def query(self,planes,points):
        return {"H":points+planes.mean()*0.}


def test_joint_render_from_one_shared_H_not_separate_semantic_graphics_scene():
    import torch.nn as nn
    graphics=nn.Linear(3,5)
    roles=nn.Linear(3,2)
    coords=torch.tensor([[[[.1,.1,.1],[.2,.1,.1]]]],requires_grad=True)
    out=render_realsas_joint_from_shared_scene_v1(
        iris=_MiniIRIS(),
        scene_triplanes=torch.zeros((1,3,2,2,2)),
        sample_xyz=coords,
        ray_delta=torch.full((1,1,2),.25),
        ray_depth=torch.tensor([[[.2,.8]]]),
        ray_direction=torch.tensor([[[0.,0.,1.]]]),
        keep_roles=torch.ones((1,1,2),dtype=torch.bool),
        shared_graphics_head=graphics,
        shared_semantic_role_head=roles,
    )
    (out.rgb_white_composite.sum()+out.semantic_posterior[...,0].sum()).backward()
    assert graphics.weight.grad.abs().sum()>0
    assert roles.weight.grad.abs().sum()>0


def test_no_unknown_generated_pixel_can_gain_source_authority():
    render,*_=_rays()
    assert render.authority=="PREDICTED_SHARED_TRIPLANE_JOINT_FIELDS_NOT_SOURCE_TRUTH"



def test_realsas_stage3_mesh_semantic_graphics_must_not_use_nerf_color():
    from dataclasses import replace
    nerf,*_=_rays()
    mesh=RealSaSFlexiCubesLayerRasterV1(
        rgb_white_composite=torch.tensor([[[.3,.8,.1]]],requires_grad=True),
        alpha=torch.tensor([[[.8]]],requires_grad=True),
        semantic_posterior=torch.tensor([[[.1,.9]]],requires_grad=True),
        normal_xyz=torch.tensor([[[0.,0.,1.]]],requires_grad=True),
        depth=torch.tensor([[[.6]]],requires_grad=True),
        flexicubes_deformation_regularizer=torch.tensor(.1,requires_grad=True),
        source_scene_prediction_sha256="e"*64,
        mesh_surface_prediction_sha256="f"*64,
        semantic_layer_name="peeled",rasterizer_identity="nvdiffrast-standin-cpu",
    )
    target=_target(STAGE_3)
    kwargs=dict(mesh_raster=mesh,target_normal=torch.tensor([[[0.,0.,1.]]]),
        target_depth=torch.tensor([[[.6]]]),
        qualified_normal_mask=torch.ones((1,1),dtype=torch.bool),
        qualified_depth_mask=torch.ones((1,1),dtype=torch.bool))
    loss=realsas_joint_stage_loss_v1(nerf,target,**kwargs)
    changed=replace(nerf,rgb_white_composite=torch.zeros_like(nerf.rgb_white_composite),
        alpha=torch.zeros_like(nerf.alpha),
        semantic_posterior=torch.zeros_like(nerf.semantic_posterior))
    unchanged=realsas_joint_stage_loss_v1(changed,target,**kwargs)
    assert torch.allclose(loss["total"],unchanged["total"])
    loss["total"].backward()
    assert mesh.rgb_white_composite.grad.abs().sum()>0
    assert mesh.alpha.grad.abs().sum()>0
    assert mesh.semantic_posterior.grad.abs().sum()>0
