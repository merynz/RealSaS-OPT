from __future__ import annotations

"""Actual unified product trainer's Graphics loss, gradients and provenance."""
from dataclasses import replace

import pytest
import torch

from models.rss_vnext.training_engine import (
    RSSvNextStructuralTrainer,RSSvNextStructuralTrainingBatch,
    source_surface_tensor_sha256,
)
from models.rss_vnext.training_seeded_tessa_v1 import SeededTESSALossWeightsV1
from models.rss_vnext.product_multitask_gradient_control_v1 import (
    ProductMultitaskGradientPolicyV1,
)
from models.rss_vnext.realsas_joint_slrm_render_v1 import (
    STAGE_1,STAGE_2,RealSaSJointTargetV1,
)
from tests.models.test_rss_vnext_product_seeded_training_route_v1 import _scenario


def _appearance(monkeypatch):
    net,batch,make_seeded,native,generated=_scenario(monkeypatch)
    base=make_seeded()
    state=net.encode_generated_multiview(
        batch.rgba,None,
        generated_rgba=batch.product_generated_rgba,
        generated_cameras=batch.product_generated_cameras,
        generated_normals=batch.product_generated_normals,
        generation_checkpoint_sha256=batch.product_generation_checkpoint_sha256,
        source_binding_sha256=batch.product_source_binding_sha256,
    )
    q=state.query_xyz.shape[1]
    target_rgb=torch.full((1,q,3),.65,dtype=batch.rgba.dtype)
    directions=torch.zeros_like(target_rgb)
    directions[...,2]=1.
    mask=torch.zeros((1,q),dtype=torch.bool)
    mask[:,:min(q,8)]=True
    rgb_key="source-visible-rgb-verified-asset"
    common={
        "dataset_manifest_sha256":"f"*64,
        "source_qualification_key":batch.source_qualification_key,
        "source_asset_sha256":
            batch.seeded_tessa_source_teacher.source_asset_sha256,
        "original_rgba_sha256":source_surface_tensor_sha256(batch.rgba),
        "original_camera_sha256":source_surface_tensor_sha256(
            batch.camera_features,
        ),
        "source_binding_sha256":batch.product_source_binding_sha256,
        "combined_generation_checkpoint_sha256":
            batch.product_generation_checkpoint_sha256,
        "generated_views_as_source_truth":False,
        "source_model_observation_count":2,
    }
    rgb_registry={rgb_key:{
        **common,
        "admission_status":"QUALIFIED_SOURCE_VISIBLE_RGB_QUERIES",
        "query_xyz_sha256":source_surface_tensor_sha256(state.query_xyz),
        "rgb_target_sha256":source_surface_tensor_sha256(target_rgb),
        "source_ray_direction_sha256":source_surface_tensor_sha256(directions),
        "rgb_mask_sha256":source_surface_tensor_sha256(mask),
        "teacher_semantics":
            "ORIGINAL_OPAQUE_FIRST_HIT_RGB_NOT_TEXTURE_OR_DEPTH",
        "visibility_mechanism":"SOURCE_ZBUFFER_FIRST_HIT_OPAQUE",
    }}
    r,s=4,3
    ray_xyz=torch.zeros((1,r,s,3))
    ray_xyz[:,:,:,2]=torch.tensor([-.5,.0,.5])
    ray_delta=torch.full((1,r,s),.3)
    ray_depth=torch.tensor([.0,.1,.2]).repeat(1,r,1)
    ray_dir=torch.zeros((1,r,3))
    ray_dir[...,2]=1.
    k=net.cfg.semantic_role_schema().count
    keep=torch.ones((1,r,k),dtype=torch.bool)
    truth=RealSaSJointTargetV1(
        rgb_white=torch.full((1,r,3),.6),
        alpha=torch.full((1,r,1),.4),
        role_indices=torch.zeros((1,r),dtype=torch.long),
        qualified_rgb_mask=torch.ones((1,r),dtype=torch.bool),
        qualified_mask_mask=torch.ones((1,r),dtype=torch.bool),
        qualified_role_mask=torch.ones((1,r),dtype=torch.bool),
        source_3d_truth_binding_sha256="e"*64,
        layer_name="all",stage=STAGE_1,
    )
    ray_key="source-rgb-alpha-3d-render-verified"
    joint_registry={ray_key:{
        **common,
        "admission_status":"QUALIFIED_SOURCE_RENDERED_3D_JOINT_SCENE",
        "observed_rgba_sha256":common["original_rgba_sha256"],
        "observed_camera_sha256":common["original_camera_sha256"],
        "ray_xyz_sha256":source_surface_tensor_sha256(ray_xyz),
        "ray_delta_sha256":source_surface_tensor_sha256(ray_delta),
        "ray_depth_sha256":source_surface_tensor_sha256(ray_depth),
        "ray_direction_sha256":source_surface_tensor_sha256(ray_dir),
        "keep_roles_sha256":source_surface_tensor_sha256(keep),
        "target_rgb_sha256":source_surface_tensor_sha256(truth.rgb_white),
        "target_alpha_sha256":source_surface_tensor_sha256(truth.alpha),
        "target_labels_sha256":source_surface_tensor_sha256(truth.role_indices),
        "target_rgb_mask_sha256":source_surface_tensor_sha256(
            truth.qualified_rgb_mask,
        ),
        "target_alpha_mask_sha256":source_surface_tensor_sha256(
            truth.qualified_mask_mask,
        ),
        "target_role_mask_sha256":source_surface_tensor_sha256(
            truth.qualified_role_mask,
        ),
        "semantic_role_schema_id":net.cfg.semantic_role_schema_id,
        "training_stage":STAGE_1,
        "semantic_layer_name":"all",
        "source_3d_truth_binding_sha256":"e"*64,
        "generated_diffusion_truth":False,
        "independent_source_3d_render":True,
        "drawable_or_attachment_teacher":False,
    }}
    with_rgb=replace(
        batch,source_graphics_rgb_target=target_rgb,
        source_graphics_view_directions=directions,
        source_graphics_rgb_mask=mask,
        source_graphics_rgb_admission_key=rgb_key,
        realsas_joint_ray_xyz=ray_xyz,
        realsas_joint_ray_delta=ray_delta,
        realsas_joint_ray_depth=ray_depth,
        realsas_joint_ray_direction=ray_dir,
        realsas_joint_keep_roles=keep,
        realsas_joint_target=truth,
        realsas_joint_teacher_admission_key=ray_key,
    )
    def make(*,rgb_weight=.6,ray_weight=.7,
             graphics_registry=rgb_registry,joint_registry_override=None,
             gradient_route_policy=ProductMultitaskGradientPolicyV1()):
        opt=torch.optim.AdamW(net.parameters(),lr=1.e-3)
        return RSSvNextStructuralTrainer(
            net,opt,dataset_manifest_sha256="f"*64,
            scene_conditioning_policy="PRODUCT_GENERATED_SIX_VIEW",
            training_profile="PRODUCT_SEEDED_IRIS_TESSA_V1",
            seeded_tessa_weights=replace(
                SeededTESSALossWeightsV1(),boundary=0.,
            ),
            trusted_seeded_tessa_registry=native,
            trusted_generated_conditioning_registry=generated,
            trusted_signed_source_registry={
                k:dict(v) for k,v in
                base.trusted_signed_source_registry.items()
            },
            trusted_graphics_rgb_registry=graphics_registry,
            trusted_realsas_joint_scene_registry=(
                joint_registry if joint_registry_override is None
                else joint_registry_override
            ),
            product_graphics_rgb_weight=rgb_weight,
            product_joint_ray_weight=ray_weight,
            product_gradient_route_policy=gradient_route_policy,
        )
    return net,with_rgb,make,rgb_registry,joint_registry


def test_graphics_source_rgb_and_alpha_both_train_same_product_H(monkeypatch,tmp_path):
    net,batch,make,*_=_appearance(monkeypatch)
    trainer=make()
    before=net.graphics[1].weight.detach().clone()
    result=trainer.train_step(batch)
    assert result.seeded_tessa_loss is not None
    assert result.graphics_source_rgb_loss is not None
    assert result.realsas_joint_slrm_loss is not None
    assert result.product_graphics_joint_loss_parts["mask_mse"]>=0
    assert result.eligible_graphics_source_rgb_queries>0
    assert result.eligible_realsas_joint_rays==4
    assert not torch.equal(before,net.graphics[1].weight.detach())
    assert net.graphics[1].weight.grad is not None
    assert net.graphics[1].weight.grad.abs().sum()>0
    assert net.tessa_seeded_head[-1].weight.grad is not None
    assert torch.isfinite(result.total_loss)
    path=tmp_path/"graphics.pt"
    trainer.save_checkpoint(path,sampler_cursor=6)
    assert trainer.load_checkpoint(path)==6


def test_source_rgb_tamper_and_generated_teacher_masquerade_fail_closed(monkeypatch):
    net,batch,make,registry,_=_appearance(monkeypatch)
    trainer=make()
    forged=batch.source_graphics_rgb_target.clone()
    forged[0,0,0]=.3
    with pytest.raises(ValueError,match="GRAPHICS_RGB_RECEIPT_DRIFT"):
        trainer.objective(replace(batch,source_graphics_rgb_target=forged))
    modified={k:dict(v) for k,v in registry.items()}
    next(iter(modified.values()))["generated_views_as_source_truth"]=True
    forbidden=make(graphics_registry=modified)
    with pytest.raises(ValueError,match="GRAPHICS_RGB_RECEIPT_DRIFT"):
        forbidden.objective(batch)


def test_joint_alpha_receipt_tamper_and_stage3_not_replaced_with_proxy(monkeypatch):
    _,batch,make,_,registry=_appearance(monkeypatch)
    trainer=make()
    bad={k:dict(v) for k,v in registry.items()}
    next(iter(bad.values()))["target_alpha_sha256"]="0"*64
    invalid=make(joint_registry_override=bad)
    with pytest.raises(ValueError,match="GRAPHICS_JOINT_RECEIPT_DRIFT"):
        invalid.objective(batch)
    stage3=replace(batch.realsas_joint_target,
                   stage="REALSAS_SLRM_FLEXICUBES_MULTILAYER")
    with pytest.raises(ValueError,match="GRAPHICS_JOINT_SOURCE_REQUIRED"):
        trainer.objective(replace(batch,realsas_joint_target=stage3))


def test_graphics_weights_and_missing_teacher_do_not_silently_pass(monkeypatch):
    _,batch,make,*_=_appearance(monkeypatch)
    with pytest.raises(ValueError,match="GRAPHICS_PHASE_POLICY_INVALID"):
        make(rgb_weight=float("nan"))
    with pytest.raises(ValueError,match="GRAPHICS_PHASE_POLICY_INVALID"):
        make(ray_weight=-.5)
    no_appearance=replace(
        batch,
        source_graphics_rgb_target=None,
        source_graphics_view_directions=None,
        source_graphics_rgb_mask=None,
        source_graphics_rgb_admission_key="",
        realsas_joint_ray_xyz=None,
        realsas_joint_ray_delta=None,
        realsas_joint_ray_depth=None,
        realsas_joint_ray_direction=None,
        realsas_joint_keep_roles=None,
        realsas_joint_target=None,
        realsas_joint_teacher_admission_key="",
    )
    with pytest.raises(ValueError,match="GRAPHICS_ENABLED_NO_SOURCE_TRUTH"):
        make().objective(no_appearance)
    with pytest.raises(ValueError,match="UNCONSUMED_SOURCE_TEACHER"):
        make(rgb_weight=0.,ray_weight=.7).objective(batch)


def test_checkpoint_source_rgb_registry_and_loss_weights_are_pinned(monkeypatch,tmp_path):
    _,batch,make,registry,_=_appearance(monkeypatch)
    trainer=make()
    path=tmp_path/"product_graphics.pt"
    trainer.save_checkpoint(path,sampler_cursor=1)
    modified={k:dict(v) for k,v in registry.items()}
    next(iter(modified.values()))["source_binding_sha256"]="0"*64
    changed=make(graphics_registry=modified)
    with pytest.raises(ValueError,match="GRAPHICS_RGB_REGISTRY_DRIFT"):
        changed.load_checkpoint(path)
    different=make(rgb_weight=.3)
    with pytest.raises(ValueError,match="OBJECTIVE_POLICY_DRIFT"):
        different.load_checkpoint(path)
