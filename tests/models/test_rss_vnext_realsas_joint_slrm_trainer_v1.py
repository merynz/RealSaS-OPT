from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import pytest
import torch
from torch import nn

from models.rss_vnext import (
    RSSvNextConfig,RSSvNextPatchEncoderForTests,RSSvNextUnifiedCharacterModel,
)
from models.rss_vnext.training_engine import (
    RSSvNextStructuralTrainingBatch,RSSvNextStructuralTrainer,
    RSSvNextTrainingWeights,source_surface_tensor_sha256,
)
from models.rss_vnext.realsas_joint_slrm_render_v1 import (
    STAGE_1,STAGE_2,STAGE_3,RealSaSJointTargetV1,
)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(6)
    torch.set_num_threads(1)


def sample(stage=STAGE_1):
    cfg=RSSvNextConfig.tiny_for_test()
    net=RSSvNextUnifiedCharacterModel(
        cfg, vision_encoder=RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim),
    )
    net.train()
    rgba=torch.rand((1,2,4,8,8))
    cameras=torch.zeros((1,2,13))
    cameras[:,:,3]=cameras[:,:,7]=cameras[:,:,11]=cameras[:,:,12]=1
    b,r,s=1,4,3
    xyz=torch.empty((b,r,s,3)).uniform_(-.5,.5)
    delta=torch.full((b,r,s),.25)
    depth=torch.arange(s,dtype=torch.float32)[None,None,:].expand(b,r,-1).contiguous()
    directions=torch.zeros((b,r,3))
    directions[...,2]=1.
    keep=torch.ones((b,r,cfg.semantic_role_schema().count),dtype=torch.bool)
    if stage==STAGE_2:
        keep[...,0]=False
    target=RealSaSJointTargetV1(
        rgb_white=torch.rand((b,r,3)),
        alpha=torch.rand((b,r,1)),
        role_indices=torch.tensor([[1,2,1,3]],dtype=torch.long),
        qualified_rgb_mask=torch.ones((b,r),dtype=torch.bool),
        qualified_mask_mask=torch.ones((b,r),dtype=torch.bool),
        qualified_role_mask=torch.ones((b,r),dtype=torch.bool),
        source_3d_truth_binding_sha256="a"*64,
        layer_name="all" if stage==STAGE_1 else "without_secondary",
        stage=stage,
    )
    batch=RSSvNextStructuralTrainingBatch(
        rgba=rgba,camera_features=cameras,
        source_qualification_key="sample-asset-qualified-for-test-only",
        admission_status="QUALIFIED_FOR_THIS_SUPERVISION_FAMILY",
        realsas_joint_ray_xyz=xyz,
        realsas_joint_ray_delta=delta,
        realsas_joint_ray_depth=depth,
        realsas_joint_ray_direction=directions,
        realsas_joint_keep_roles=keep,
        realsas_joint_target=target,
        realsas_joint_teacher_admission_key="verified-3d-target-fixture",
    )
    h=source_surface_tensor_sha256
    receipt=dict(
        admission_status="QUALIFIED_SOURCE_RENDERED_3D_JOINT_SCENE",
        dataset_manifest_sha256="f"*64,
        source_qualification_key=batch.source_qualification_key,
        observed_rgba_sha256=h(rgba),observed_camera_sha256=h(cameras),
        ray_xyz_sha256=h(xyz),ray_delta_sha256=h(delta),
        ray_depth_sha256=h(depth),ray_direction_sha256=h(directions),
        keep_roles_sha256=h(keep),
        target_rgb_sha256=h(target.rgb_white),
        target_alpha_sha256=h(target.alpha),
        target_labels_sha256=h(target.role_indices),
        target_rgb_mask_sha256=h(target.qualified_rgb_mask),
        target_alpha_mask_sha256=h(target.qualified_mask_mask),
        target_role_mask_sha256=h(target.qualified_role_mask),
        semantic_role_schema_id=cfg.semantic_role_schema_id,
        training_stage=target.stage,semantic_layer_name=target.layer_name,
        source_3d_truth_binding_sha256=target.source_3d_truth_binding_sha256,
        generated_diffusion_truth=False,independent_source_3d_render=True,
        drawable_or_attachment_teacher=False,
    )
    return net,batch,receipt


@pytest.mark.parametrize("stage",[STAGE_1,STAGE_2])
def test_training_uses_actual_same_H_for_joint_geo_graphics_semantic(stage):
    net,batch,receipt=sample(stage)
    trainer=RSSvNextStructuralTrainer(
        net,torch.optim.AdamW(net.parameters(),lr=1e-3),
        dataset_manifest_sha256="f"*64,
        weights=RSSvNextTrainingWeights(
            topology=0.,motion_preference=0.,source_semantics=0.,
            flexicubes_surface=0.,realsas_joint_slrm=1.,
        ),
        trusted_realsas_joint_scene_registry={"verified-3d-target-fixture":receipt},
    )
    result=trainer.train_step(batch)
    assert result.realsas_joint_slrm_loss is not None
    assert torch.isfinite(result.realsas_joint_slrm_loss)
    assert result.eligible_realsas_joint_rays==4
    assert trainer.global_step==1
    assert net.graphics[-1].weight.grad.abs().sum()>0
    assert net.presentation_role_head[-1].weight.grad.abs().sum()>0
    assert net.iris.view_bank.image_projection.weight.grad.abs().sum()>0


def test_untrusted_peeled_gt_or_fake_3d_receipt_cannot_train():
    net,batch,receipt=sample(STAGE_2)
    weights=RSSvNextTrainingWeights(realsas_joint_slrm=1.)
    bad=receipt.copy()
    bad["generated_diffusion_truth"]=True
    trainer=RSSvNextStructuralTrainer(
        net,torch.optim.AdamW(net.parameters()),dataset_manifest_sha256="f"*64,
        weights=weights,
        trusted_realsas_joint_scene_registry={"verified-3d-target-fixture":bad},
    )
    with pytest.raises(ValueError,match="JOINT_RENDER_TEACHER_RECEIPT_DRIFT"):
        trainer.objective(batch)
    good=RSSvNextStructuralTrainer(
        net,torch.optim.AdamW(net.parameters()),dataset_manifest_sha256="f"*64,
        weights=weights,
        trusted_realsas_joint_scene_registry={"verified-3d-target-fixture":receipt},
    )
    batch.realsas_joint_keep_roles[0,0,0]=True
    with pytest.raises(ValueError,match="JOINT_RENDER_TEACHER_RECEIPT_DRIFT"):
        good.objective(batch)


def test_no_nerf_ray_can_mint_flexicubes_mesh_teacher():
    net,batch,receipt=sample(STAGE_3)
    trainer=RSSvNextStructuralTrainer(
        net,torch.optim.AdamW(net.parameters()),
        dataset_manifest_sha256="f"*64,
        weights=RSSvNextTrainingWeights(realsas_joint_slrm=1.),
        trusted_realsas_joint_scene_registry={"verified-3d-target-fixture":receipt},
    )
    with pytest.raises(ValueError,match="FLEXICUBES_STAGE_MESH_RENDER_REQUIRED"):
        trainer.objective(batch)


def test_checkpoint_refuses_changed_source_joint_render_registry(tmp_path):
    net,batch,receipt=sample()
    trainer=RSSvNextStructuralTrainer(
        net,torch.optim.AdamW(net.parameters()),dataset_manifest_sha256="f"*64,
        trusted_realsas_joint_scene_registry={"verified-3d-target-fixture":receipt},
    )
    file=tmp_path/"checkpoint.pt"
    trainer.save_checkpoint(file,sampler_cursor=4)
    updated=receipt.copy()
    updated["target_alpha_sha256"]="1"*64
    other=RSSvNextStructuralTrainer(
        net,torch.optim.AdamW(net.parameters()),dataset_manifest_sha256="f"*64,
        trusted_realsas_joint_scene_registry={"verified-3d-target-fixture":updated},
    )
    with pytest.raises(ValueError,match="CHECKPOINT_REALSAS_JOINT_REGISTRY_DRIFT"):
        other.load_checkpoint(file)
