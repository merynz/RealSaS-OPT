from __future__ import annotations

from dataclasses import replace
import torch
import pytest

from models.rss_vnext import (
    RSSvNextConfig,RSSvNextPatchEncoderForTests,RSSvNextUnifiedCharacterModel,
)
from models.rss_vnext.stdgen_slrm_training_driver_v1 import (
    StdGENSLRMBatchV1,StdGENSLRMTrainingLossWeightsV1,
    StdGENSLRMTrainerV1,_digest,
)
from models.rss_vnext.stdgen_scene_conditioning_v1 import (
    CANONICAL_REFERENCE,SIX_VIEW_PRIOR,
)
from models.rss_vnext.stdgen_joint_slrm_render_v1 import (
    STAGE_1,STAGE_2,STAGE_3,StdGENJointTargetV1,
)


@pytest.fixture(autouse=True)
def cpu_determinism():
    torch.manual_seed(55)
    torch.set_num_threads(1)


def _batch(stage=STAGE_1,view_count=1):
    cfg=RSSvNextConfig.tiny_for_test()
    model=RSSvNextUnifiedCharacterModel(
        cfg,vision_encoder=RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim),
    ).train()
    b,r,s,k=1,4,3,cfg.semantic_role_schema().count
    views=torch.rand((b,view_count,4,16,16))
    cams=torch.zeros((b,view_count,13))
    import math
    for i in range(view_count):
        a=math.radians((-90,-45,0,45,90,180)[i]) if view_count==6 else 0.
        cams[:,i,3]=math.cos(a)
        cams[:,i,5]=-math.sin(a)
        cams[:,i,7]=1.
        cams[:,i,9]=math.sin(a)
        cams[:,i,11]=math.cos(a)
        cams[:,i,12]=1.
    xyz=torch.rand((b,r,s,3))-.5
    delta=torch.full((b,r,s),.3)
    depth=torch.tensor([.1,.4,.7])[None,None,:].expand(b,r,-1).contiguous()
    direc=torch.zeros((b,r,3));direc[...,2]=1.
    keep=torch.ones((b,r,k),dtype=torch.bool)
    if stage!=STAGE_1:keep[...,0]=False
    gt=StdGENJointTargetV1(
        rgb_white=torch.rand((b,r,3)),alpha=torch.rand((b,r,1)),
        role_indices=torch.tensor([[1,2,0,1]]),
        qualified_rgb_mask=torch.ones((b,r),dtype=torch.bool),
        qualified_mask_mask=torch.ones((b,r),dtype=torch.bool),
        qualified_role_mask=torch.ones((b,r),dtype=torch.bool),
        source_3d_truth_binding_sha256="a"*64,
        layer_name="all" if stage==STAGE_1 else "without_secondary",
        stage=stage,
    )
    auth=CANONICAL_REFERENCE if view_count==1 else SIX_VIEW_PRIOR
    batch=StdGENSLRMBatchV1(
        conditioning_rgba=views,conditioning_cameras=cams,
        conditioning_authority=auth,source_binding_sha256="b"*64,
        original_observed_view_count=1,
        generating_checkpoint_sha256="c"*64,
        ray_xyz=xyz,ray_deltas=delta,ray_depths=depth,
        ray_directions=direc,keep_roles=keep,
        source_3d_render_target=gt,admitted_3d_render_key="admitted",
    )
    receipt=dict(
        admission_status="QUALIFIED_SOURCE_3D_RENDER_MULTILAYER",
        dataset_manifest_sha256="f"*64,
        original_source_binding_sha256=batch.source_binding_sha256,
        original_observed_view_count=1,
        conditioning_authority=auth,
        conditioning_checkpoint_sha256="c"*64,
        conditioning_rgba_sha256=_digest(views),
        conditioning_cameras_sha256=_digest(cams),
        ray_xyz_sha256=_digest(xyz),ray_deltas_sha256=_digest(delta),
        ray_depths_sha256=_digest(depth),ray_directions_sha256=_digest(direc),
        keep_roles_sha256=_digest(keep),
        target_rgb_sha256=_digest(gt.rgb_white),
        target_alpha_sha256=_digest(gt.alpha),
        target_role_sha256=_digest(gt.role_indices),
        target_rgb_mask_sha256=_digest(gt.qualified_rgb_mask),
        target_alpha_mask_sha256=_digest(gt.qualified_mask_mask),
        target_role_mask_sha256=_digest(gt.qualified_role_mask),
        training_stage=stage,semantic_layer_name=gt.layer_name,
        source_3d_truth_binding_sha256=gt.source_3d_truth_binding_sha256,
        generated_2d_as_3d_teacher=False,
        source_rights_independently_admitted=True,
        role_visibility_from_source_3d=True,
    )
    return model,batch,receipt


def _trainer(model,receipt):
    return StdGENSLRMTrainerV1(
        model,torch.optim.AdamW(model.parameters(),lr=.001),
        dataset_manifest_sha256="f"*64,
        weights=StdGENSLRMTrainingLossWeightsV1(lpips=0.),
        trusted_source_3d_render_registry={"admitted":receipt},
    )


@pytest.mark.parametrize("stage,views",[(STAGE_1,1),(STAGE_2,6)])
def test_stdgen_slrm_actual_1or6_multiview_conditioned_train_step(stage,views):
    net,batch,receipt=_batch(stage,views)
    trainer=_trainer(net,receipt)
    initial=net.iris.view_bank.image_projection.weight.clone().detach()
    losses=trainer.train_step(batch)
    assert trainer.global_step==1
    assert torch.isfinite(losses["total"]) and losses["total"].item()>0
    assert not torch.equal(initial,net.iris.view_bank.image_projection.weight.detach())
    assert net.graphics[-1].weight.grad.abs().sum()>0
    assert net.presentation_role_head[-1].weight.grad.abs().sum()>0


def test_generated_prior_not_become_3d_teacher_even_with_registry_relabel():
    net,batch,receipt=_batch(STAGE_2,6)
    malformed=receipt.copy()
    malformed["generated_2d_as_3d_teacher"]=True
    with pytest.raises(ValueError,match="RENDER_RECEIPT_DRIFT"):
        _trainer(net,malformed).objective(batch)
    valid=_trainer(net,receipt)
    batch.conditioning_rgba[0,0,0,0,0]+=0.1
    with pytest.raises(ValueError,match="RENDER_RECEIPT_DRIFT"):
        valid.objective(batch)


def test_mesh_stage3_cannot_use_nerf_proxy_or_unadmitted_source_sign():
    net,batch,receipt=_batch(STAGE_3,6)
    trainer=_trainer(net,receipt)
    with pytest.raises(ValueError,match="STAGE3_MESH_RENDER_REQUIRED"):
        trainer.objective(batch)


def test_stage3_missing_mesh_targets_fail_before_model_forward():
    net,batch,receipt=_batch(STAGE_3,6)
    trainer=_trainer(net,receipt)
    with pytest.raises(ValueError,match="STAGE3_MESH_RENDER_REQUIRED"):
        trainer.objective(batch)


def test_stage3_source_camera_normal_depth_are_individually_bound_to_receipt():
    from models.rss_vnext.semantic_geometry_prior import RSSvNextSemanticLayerSpec
    net,batch,receipt=_batch(STAGE_3,6)
    k=net.cfg.semantic_role_schema().count
    roles=net.cfg.semantic_role_schema().roles
    batch=replace(batch,
        stage3_flexicubes_extractor=object(),
        stage3_semantic_layer=RSSvNextSemanticLayerSpec(
            "without_secondary",roles[1:],
        ),
        stage3_camera_features=batch.conditioning_cameras.clone(),
        stage3_hw=(8,8),
        stage3_target_normal=torch.zeros((1,6*64,3)),
        stage3_target_depth=torch.ones((1,6*64,1)),
        stage3_normal_mask=torch.ones((1,6*64),dtype=torch.bool),
        stage3_depth_mask=torch.ones((1,6*64),dtype=torch.bool),
    )
    # A 4-ray teacher cannot masquerade as a six-camera 8x8 mesh teacher.
    with pytest.raises(ValueError,match="SOURCE_RENDER_DOMAIN"):
        _trainer(net,receipt).objective(batch)
    # Match the RGB teacher's six camera pixels, then make the receipts
    # complete and prove that even one changed target is caught before fit.
    size=6*64
    target=replace(batch.source_3d_render_target,
        rgb_white=torch.rand((1,size,3)),
        alpha=torch.ones((1,size,1)),
        role_indices=torch.zeros((1,size),dtype=torch.long),
        qualified_rgb_mask=torch.ones((1,size),dtype=torch.bool),
        qualified_mask_mask=torch.ones((1,size),dtype=torch.bool),
        qualified_role_mask=torch.ones((1,size),dtype=torch.bool),
    )
    batch=replace(batch,source_3d_render_target=target)
    receipt=receipt.copy()
    receipt.update({
        "target_rgb_sha256":_digest(target.rgb_white),
        "target_alpha_sha256":_digest(target.alpha),
        "target_role_sha256":_digest(target.role_indices),
        "target_rgb_mask_sha256":_digest(target.qualified_rgb_mask),
        "target_alpha_mask_sha256":_digest(target.qualified_mask_mask),
        "target_role_mask_sha256":_digest(target.qualified_role_mask),
        "independently_qualified_closed_signed_source":True,
        "stage3_source_camera_sha256":_digest(batch.stage3_camera_features),
        "stage3_normal_target_sha256":_digest(batch.stage3_target_normal),
        "stage3_depth_target_sha256":_digest(batch.stage3_target_depth),
        "stage3_normal_mask_sha256":_digest(batch.stage3_normal_mask),
        "stage3_depth_mask_sha256":_digest(batch.stage3_depth_mask),
        "stage3_render_height":8,"stage3_render_width":8,
        "stage3_world_normal_convention":"SOURCE_GEOMETRIC_NORMAL_WORLD_XYZ",
        "stage3_metric_depth_convention":"SOURCE_CAMERA_FORWARD_FIRST_HIT_WORLD_UNITS",
        "stage3_visibility_convention":"SOURCE_3D_ZBUFFER_VISIBLE_ONLY",
        "stage3_semantic_role_schema_id":net.cfg.semantic_role_schema_id,
        "stage3_layer_keep_roles":list(roles[1:]),
    })
    good=_trainer(net,receipt)
    # Intentionally do not execute CUDA; _verify_receipt is the precise
    # launcher boundary, separately tested from actual mesh raster.
    good._verify_receipt(batch)
    corrupt=replace(batch,stage3_target_depth=batch.stage3_target_depth+.01)
    with pytest.raises(ValueError,match="RENDER_RECEIPT_DRIFT"):
        good._verify_receipt(corrupt)
    corrupt=replace(batch,stage3_target_normal=batch.stage3_target_normal+.01)
    with pytest.raises(ValueError,match="RENDER_RECEIPT_DRIFT"):
        good._verify_receipt(corrupt)
    corrupt=replace(batch,stage3_camera_features=batch.stage3_camera_features+.001)
    with pytest.raises(ValueError,match="RENDER_RECEIPT_DRIFT"):
        good._verify_receipt(corrupt)
    missing=dict(receipt)
    missing.pop("stage3_metric_depth_convention")
    with pytest.raises(ValueError,match="RENDER_RECEIPT_DRIFT"):
        _trainer(net,missing)._verify_receipt(batch)


def test_source_truth_registry_resume_refuses_changes(tmp_path):
    net,batch,receipt=_batch()
    trainer=_trainer(net,receipt)
    trainer.train_step(batch)
    filename=tmp_path/"resume.pt"
    trainer.save_checkpoint(filename,sampler_cursor=8)
    same=_trainer(net,receipt)
    assert same.load_checkpoint(filename)==8
    assert same.global_step==1
    changed=dict(receipt,target_rgb_sha256="d"*64)
    with pytest.raises(ValueError,match="CHECKPOINT_RENDER_TRUTH_DRIFT"):
        _trainer(net,changed).load_checkpoint(filename)


def test_lpips_paper_term_is_required_unless_explicitly_disabled():
    net,batch,receipt=_batch()
    with pytest.raises(ValueError,match="LPIPS_REQUIRED_NOT_MSE_SURROGATE"):
        StdGENSLRMTrainerV1(
            net,torch.optim.AdamW(net.parameters()),
            dataset_manifest_sha256="f"*64,
            trusted_source_3d_render_registry={"admitted":receipt},
        )
