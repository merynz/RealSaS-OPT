from __future__ import annotations

from dataclasses import replace
import copy

import pytest
import torch

from models.rss_vnext.realsas_observed_source_ray_tether_v1 import (
    CAMERA_ABI, RSSvNextOriginalObservedRayBatchV1,
    construct_original_observed_ray_samples_v1,
    original_observed_source_photometric_loss_v1,
)
from models.rss_vnext.realsas_virtual_camera_policy_v1 import (
    target_canonical_and_six_cameras_v1,
)
from models.rss_vnext.realsas_joint_slrm_render_v1 import STAGE_1,STAGE_2,STAGE_3
from models.rss_vnext.realsas_slrm_training_driver_v1 import (
    RealSaSSLRMTrainerV1,RealSaSSLRMTrainingLossWeightsV1,
    _digest,REALSAS_SIX_VIEW_TRAINING_PROFILE,
)
from tests.models.test_rss_vnext_realsas_slrm_training_driver_v1 import _batch


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(735)
    torch.set_num_threads(1)


def source_case(stage=STAGE_1):
    model,batch,receipt=_batch(stage,6)
    cameras,_=target_canonical_and_six_cameras_v1(batch_size=1)
    indices=torch.tensor([[
        [0,7,8],[0,8,8],[1,7,8],[1,8,8],
    ]],dtype=torch.long)
    depth=torch.linspace(1.,3.,6)[None,None,:].expand(1,4,6).clone()
    delta=torch.full_like(depth,.4)
    source=RSSvNextOriginalObservedRayBatchV1(
        original_linear_rgba=torch.rand(1,2,4,16,16),
        original_camera_13d=cameras,
        source_pixel_indices=indices,
        ray_depth=depth,ray_delta=delta,
        qualified_rgb_mask=torch.ones((1,4),dtype=torch.bool),
        qualified_alpha_mask=torch.ones((1,4),dtype=torch.bool),
    )
    batch=replace(
        batch,original_observed_view_count=2,
        original_source_rgba=source.original_linear_rgba,
        source_observed_rays=source,source_observed_admission_key="native-source-pair",
    )
    receipt={
        **receipt,"original_observed_view_count":2,
        "original_source_rgba_sha256":_digest(source.original_linear_rgba),
    }
    signed={
        "admission_status":"QUALIFIED_NATIVE_TWO_SOURCE_CAMERA_PHOTOMETRY",
        "dataset_manifest_sha256":"f"*64,
        "original_source_binding_sha256":batch.source_binding_sha256,
        "original_observed_view_count":2,
        "original_linear_rgba_sha256":_digest(source.original_linear_rgba),
        "original_camera_13d_sha256":_digest(source.original_camera_13d),
        "source_pixel_indices_sha256":_digest(source.source_pixel_indices),
        "ray_depth_sha256":_digest(source.ray_depth),
        "ray_delta_sha256":_digest(source.ray_delta),
        "source_rgb_mask_sha256":_digest(source.qualified_rgb_mask),
        "source_alpha_mask_sha256":_digest(source.qualified_alpha_mask),
        "original_camera_abi":CAMERA_ABI,
        "original_cameras_independently_calibrated":True,
        "original_source_linear_rgb_straight_alpha":True,
        "source_pixels_are_native_not_generated":True,
        "source_rays_supply_3d_depth_truth":False,
        "generated_completion_used_as_source_photometry":False,
    }
    return model,batch,receipt,signed


def make_trainer(model,receipt,signed,*,rgb=1.,alpha=1.):
    return RealSaSSLRMTrainerV1(
        model,torch.optim.AdamW(model.parameters(),lr=.001),
        dataset_manifest_sha256="f"*64,
        trusted_source_3d_render_registry={"admitted":receipt},
        trusted_original_observed_ray_registry={"native-source-pair":signed},
        weights=RealSaSSLRMTrainingLossWeightsV1(
            lpips=0.,observed_source_rgb=rgb,observed_source_alpha=alpha,
        ),
        input_profile=REALSAS_SIX_VIEW_TRAINING_PROFILE,
    )


def test_two_actual_source_photometric_rays_train_same_six_view_scene():
    net,batch,receipt,signed=source_case(STAGE_1)
    trainer=make_trainer(net,receipt,signed)
    outcomes=trainer.train_step(batch)
    assert trainer.global_step==1
    assert torch.isfinite(outcomes["total"])
    assert float(outcomes["observed_source_rgb_mse"])>0
    assert float(outcomes["observed_source_alpha_mse"])>0
    assert any(
        p.grad is not None and bool(torch.isfinite(p.grad).all())
        and float(p.grad.abs().sum())>0
        for p in net.graphics.parameters()
    )
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in net.iris.parameters())


def test_pixel_center_projection_returns_original_pixels_not_generated_completion():
    _,batch,_,_=source_case()
    r=construct_original_observed_ray_samples_v1(batch.source_observed_rays)
    src=batch.source_observed_rays
    pix=src.original_linear_rgba.permute(0,1,3,4,2)
    index=src.source_pixel_indices[0]
    raw=pix[0,index[:,0],index[:,1],index[:,2]]
    assert torch.equal(r.alpha,raw[None,:,3:4])
    assert torch.allclose(
        r.rgb_white,raw[None,:,:3]*raw[None,:,3:4]+1-raw[None,:,3:4],
    )
    assert not r.source_geometry_is_truth and not r.source_depth_is_truth
    assert r.sample_xyz.shape==(1,4,6,3)
    assert not r.sample_xyz.requires_grad


@pytest.mark.parametrize("stage",(STAGE_1,STAGE_2))
def test_source_tether_preserves_full_observed_color_during_layer_peeling(stage):
    net,batch,receipt,signed=source_case(stage)
    trainer=make_trainer(net,receipt,signed)
    outcome=trainer.objective(batch)
    assert float(outcome["observed_source_rgb_mse"])>0
    assert float(outcome["observed_source_alpha_mse"])>0
    assert torch.isfinite(outcome["total"])


@pytest.mark.parametrize("field",(
    "original_linear_rgba","original_camera_13d","source_pixel_indices",
    "ray_depth","ray_delta","qualified_rgb_mask","qualified_alpha_mask",
))
def test_independently_bound_source_field_cannot_be_replaced(field):
    net,batch,receipt,signed=source_case()
    original=batch.source_observed_rays
    changed=getattr(original,field).clone()
    if changed.dtype==torch.bool:
        changed[0,0]=~changed[0,0]
    elif field=="source_pixel_indices":
        changed[0,0,2]+=1
    else:
        changed.reshape(-1)[0]+=(.1 if changed.is_floating_point() else 1)
    forged=replace(batch,source_observed_rays=replace(original,**{field:changed}))
    expected = (
        "SOURCE_PIXEL_IDENTITY_DRIFT" if field == "original_linear_rgba"
        else "ORIGINAL_SOURCE_RECEIPT_DRIFT"
    )
    with pytest.raises(ValueError,match=expected):
        make_trainer(net,receipt,signed).objective(forged)


def test_invalid_source_camera_basis_cannot_be_admitted_even_if_bytes_resigned():
    net,batch,receipt,signed=source_case()
    source=batch.source_observed_rays
    invalid=source.original_camera_13d.clone()
    invalid[0,0,3:6]*=2
    source=replace(source,original_camera_13d=invalid)
    signed={**signed,"original_camera_13d_sha256":_digest(invalid)}
    with pytest.raises(ValueError,match="CAMERA_NOT_CALIBRATED_ORTHONORMAL"):
        make_trainer(net,receipt,signed).objective(
            replace(batch,source_observed_rays=source),
        )


def test_missing_calibrated_pair_and_untrusted_source_registry_fail_closed():
    net,batch,receipt,signed=source_case()
    trainer=make_trainer(net,receipt,signed)
    with pytest.raises(ValueError,match="CALIBRATED_PAIR_REQUIRED"):
        trainer.objective(replace(batch,source_observed_rays=None))
    with pytest.raises(ValueError,match="ORIGINAL_SOURCE_RECEIPT_DRIFT"):
        make_trainer(
            net,receipt,{**signed,"original_cameras_independently_calibrated":False},
        ).objective(batch)
    with pytest.raises(ValueError,match="ORIGINAL_SOURCE_REGISTRY_REQUIRED"):
        RealSaSSLRMTrainerV1(
            net,torch.optim.AdamW(net.parameters(),lr=.001),
            dataset_manifest_sha256="f"*64,
            trusted_source_3d_render_registry={"admitted":receipt},
            weights=RealSaSSLRMTrainingLossWeightsV1(
                lpips=0.,observed_source_rgb=1.,
            ),
        )


def test_tether_cannot_turn_source_photometry_into_stage3_nerf_mesh_teacher():
    net,batch,receipt,signed=source_case(STAGE_1)
    bad=replace(batch,source_3d_render_target=replace(
        batch.source_3d_render_target,stage=STAGE_3,
    ))
    with pytest.raises(ValueError,match="CALIBRATED_PAIR_REQUIRED|MESH_RENDER_REQUIRED"):
        make_trainer(net,receipt,signed).objective(bad)


def test_source_registry_and_loss_weights_pinned_in_resumable_checkpoint(tmp_path):
    net,batch,receipt,signed=source_case()
    trainer=make_trainer(net,receipt,signed)
    trainer.train_step(batch)
    dest=tmp_path/"source-ray-shared-h.pt"
    trainer.save_checkpoint(dest,sampler_cursor=7)
    clone=make_trainer(net,receipt,signed)
    assert clone.load_checkpoint(dest)==7
    with pytest.raises(ValueError,match="CHECKPOINT_ORIGINAL_SOURCE_DRIFT"):
        make_trainer(net,receipt,{
            **signed,"original_cameras_independently_calibrated":False,
        }).load_checkpoint(dest)
    with pytest.raises(ValueError,match="CHECKPOINT_OBJECTIVE_POLICY_DRIFT"):
        make_trainer(net,receipt,signed,rgb=.5).load_checkpoint(dest)


def test_gradient_only_flows_to_student_not_source_geometry_or_image():
    net,batch,receipt,signed=source_case()
    original=batch.source_observed_rays
    rgb=original.original_linear_rgba.detach().clone().requires_grad_(True)
    cameras=original.original_camera_13d.detach().clone().requires_grad_(True)
    target=replace(original,original_linear_rgba=rgb,original_camera_13d=cameras)
    engine=make_trainer(net,receipt,signed)
    outcome=engine.train_step(replace(batch,source_observed_rays=target))
    assert float(outcome["observed_source_rgb_mse"])>0
    assert rgb.grad is None and cameras.grad is None


def test_original_camera_pixels_cannot_be_duplicate_or_one_sided():
    _,batch,_,_=source_case()
    source=batch.source_observed_rays
    duplicated=source.source_pixel_indices.clone()
    duplicated[0,1]=duplicated[0,0]
    with pytest.raises(ValueError,match="DUPLICATE_PIXEL"):
        construct_original_observed_ray_samples_v1(
            replace(source,source_pixel_indices=duplicated),
        )
    one_sided=source.source_pixel_indices.clone()
    one_sided[0,2:,0]=0
    one_sided[0,2:,1]=torch.tensor([9,10])
    with pytest.raises(ValueError,match="BOTH_SOURCE_VIEWS_REQUIRED"):
        construct_original_observed_ray_samples_v1(
            replace(source,source_pixel_indices=one_sided),
        )


def _real_stage3_source_case():
    # Contract-level raster replay with a synthetic witness only.
    # It exercises the real first-hit differentiable mesh renderer and H(x)
    # colors; it is NOT a production FlexiCubes extraction quality test.
    from torch import nn
    from models.rss_vnext.flexicubes_surface_evidence import (
        RSSvNextFlexiCubesSurfaceEvidence,
    )
    from models.rss_vnext.semantic_geometry_prior import (
        RSSvNextSemanticLayerSpec,
    )
    net,batch,receipt,signed=source_case(STAGE_3)
    roles=net.cfg.semantic_role_schema().roles
    k=len(roles)
    count=6*8*8
    target=replace(
        batch.source_3d_render_target,
        rgb_white=torch.rand((1,count,3)),
        alpha=torch.ones((1,count,1)),
        role_indices=torch.zeros((1,count),dtype=torch.long),
        qualified_rgb_mask=torch.ones((1,count),dtype=torch.bool),
        qualified_mask_mask=torch.ones((1,count),dtype=torch.bool),
        qualified_role_mask=torch.ones((1,count),dtype=torch.bool),
    )
    batch=replace(
        batch,source_3d_render_target=target,
        stage3_flexicubes_extractor=object(),
        stage3_semantic_layer=RSSvNextSemanticLayerSpec(
            "without_secondary",roles[1:],
        ),
        stage3_camera_features=batch.conditioning_cameras.clone(),
        stage3_hw=(8,8),
        stage3_target_normal=torch.zeros((1,count,3)),
        stage3_target_depth=torch.ones((1,count,1)),
        stage3_normal_mask=torch.ones((1,count),dtype=torch.bool),
        stage3_depth_mask=torch.ones((1,count),dtype=torch.bool),
    )
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
    class SyntheticWitnessBridge(nn.Module):
        def __init__(self):
            super().__init__()
            self.call_layers=[]
        def forward(self,scene_triplanes,*,extractor,role_head,schema,layer,field_policy):
            assert field_policy=="SIGNED_HYPOTHESIS_ONLY"
            assert schema.roles==roles
            self.call_layers.append(layer.name)
            xyz=torch.tensor(
                [[[-.35,-.35,0.],[.35,-.35,0.],[0.,.35,0.]]],
                requires_grad=True,device=scene_triplanes.device,
            )
            return RSSvNextFlexiCubesSurfaceEvidence(
                xyz=xyz,faces=torch.tensor([[0,1,2]],dtype=torch.long),
                local_features=torch.zeros((1,3,4)),
                presentation_role_logits=torch.zeros((1,3,k)),
                regularizer=xyz.square().mean(),
                sample_grid_xyz=torch.zeros((1,8,3)),
                signed_field_grid=torch.tensor([[-1.,1.,-1.,1.,-1.,1.,-1.,1.]]),
                cube_parameters=torch.zeros((1,21)),
                source_policy=field_policy,
                layer_name=layer.name,
            )
    bridge=SyntheticWitnessBridge()
    net.fc_surface_bridge=bridge
    return net,batch,receipt,signed,bridge


def test_stage3_preserves_two_real_views_using_full_role_mesh_raster_not_nerf():
    from unittest.mock import patch
    net,batch,receipt,signed,bridge=_real_stage3_source_case()
    engine=make_trainer(net,receipt,signed)
    with patch(
        "models.rss_vnext.realsas_slrm_training_driver_v1.render_realsas_joint_from_shared_scene_v1",
        side_effect=AssertionError("Stage3 source must not use NeRF"),
    ):
        outputs=engine.objective(batch)
    assert bridge.call_layers==[
        "without_secondary","original_source_all_visible_roles",
    ]
    assert torch.isfinite(outputs["total"])
    assert float(outputs["observed_source_rgb_mse"])>0
    assert float(outputs["observed_source_alpha_mse"])>0
    outputs["total"].backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in net.graphics.parameters())



def test_product_two_originals_train_shared_H_without_any_camera_ray_teacher():
    # Product does not get calibrated source cameras; the original pixels
    # themselves must condition H(x) in the SAME S-LRM reconstruction graph.
    from tests.models.test_rss_vnext_realsas_slrm_training_driver_v1 import _trainer
    net,batch,receipt = _batch(STAGE_1,6)
    originals = torch.rand(1,2,4,16,16)
    batch = replace(
        batch,original_observed_view_count=2,
        original_source_rgba=originals,
    )
    receipt = {
        **receipt,"original_observed_view_count":2,
        "original_source_rgba_sha256":_digest(originals),
    }
    assert batch.source_observed_rays is None
    trainer = _trainer(net,receipt)
    before = net.iris.scene.scene_queries.detach().clone()
    result = trainer.train_step(batch)
    assert torch.isfinite(result["total"])
    assert net.iris.scene.scene_queries.grad is not None
    assert net.iris.scene.scene_queries.grad.abs().sum()>0
    assert not torch.equal(before,net.iris.scene.scene_queries.detach())
    assert trainer.global_step==1


def test_product_two_originals_hard_fail_when_missing_or_byte_modified():
    from tests.models.test_rss_vnext_realsas_slrm_training_driver_v1 import _trainer
    net,batch,receipt = _batch(STAGE_1,6)
    originals = torch.rand(1,2,4,16,16)
    batch = replace(
        batch,original_observed_view_count=2,original_source_rgba=originals,
    )
    receipt = {
        **receipt,"original_observed_view_count":2,
        "original_source_rgba_sha256":_digest(originals),
    }
    trainer = _trainer(net,receipt)
    with pytest.raises(ValueError,match="TWO_CAMERA_FREE_SOURCE_PIXELS_REQUIRED"):
        trainer.objective(replace(batch,original_source_rgba=None))
    tampered=originals.clone()
    tampered[0,1,0,0,0]+=.02
    with pytest.raises(ValueError,match="RENDER_RECEIPT_DRIFT"):
        trainer.objective(replace(batch,original_source_rgba=tampered))


def test_camera_ray_teacher_and_original_scene_tokens_must_be_same_pixels():
    net,batch,receipt,signed=source_case()
    altered=batch.original_source_rgba.clone()
    altered[0,0,0,0,0]+=.01
    receipt={
        **receipt,
        "original_source_rgba_sha256":_digest(altered),
    }
    with pytest.raises(ValueError,match="SOURCE_PIXEL_IDENTITY_DRIFT"):
        make_trainer(net,receipt,signed).objective(
            replace(batch,original_source_rgba=altered),
        )


def test_slrm_checkpoint_seals_architecture_and_cuda_rng_field(tmp_path):
    from tests.models.test_rss_vnext_realsas_slrm_training_driver_v1 import _trainer
    net,batch,receipt=_batch(STAGE_1,6)
    trainer=_trainer(net,receipt)
    trainer.train_step(batch)
    path=tmp_path/"slrm-arch.pt"
    trainer.save_checkpoint(path,sampler_cursor=11)
    saved=torch.load(path,map_location="cpu",weights_only=True)
    assert saved["architecture_id"]==net.architecture_id
    assert len(saved["architecture_digest"])==64
    assert "torch_cuda_rng" in saved
    assert _trainer(net,receipt).load_checkpoint(path)==11
    corrupt={**saved,"architecture_digest":"f"*64}
    torch.save(corrupt,path)
    with pytest.raises(ValueError,match="CHECKPOINT_ARCHITECTURE_DRIFT"):
        _trainer(net,receipt).load_checkpoint(path)
