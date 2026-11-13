from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from models.rss_vnext.contracts import RSSvNextConfig
from models.rss_vnext.unified_character_model import (
    RSSvNextUnifiedCharacterModel,RSSvNextPatchEncoderForTests,
)
from models.rss_vnext.dual_spatial_support_research_v1 import (
    RSSvNextDualSpatialSupportResearchV1,
    source_qualified_bone_support_loss_v1,
)
from models.rss_vnext.training_engine import source_surface_tensor_sha256


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(311)
    torch.set_num_threads(1)


def _scene():
    cfg=RSSvNextConfig.tiny_for_test()
    m=RSSvNextUnifiedCharacterModel(
        cfg,vision_encoder=RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim),
    )
    rgba=torch.rand((1,2,4,8,8))
    camera=torch.zeros((1,2,13))
    camera[:,:,3]=camera[:,:,7]=camera[:,:,11]=camera[:,:,12]=1.
    state,_,_=m.encode_observations(rgba,camera)
    return m,state,rgba,camera


def _apply(model,state,mode="LEARNED_INTERIOR_SUPPORT",count=4):
    candidate=RSSvNextDualSpatialSupportResearchV1(
        model.cfg.iris.local_feature_dim,support_count=count,heads=4,
    )
    out=candidate(iris=model.iris,scene_planes=state.scene_triplanes,
        grid_xyz=state.query_xyz,grid_h=state.local_features,
        mode=mode)
    return candidate,out


def _teacher_receipt(source,mask):
    return {"test-source-native":{
        "admission_status":"QUALIFIED_SOURCE_NATIVE_BONE_PATH",
        "source_asset_sha256":"a"*64,
        "dataset_manifest_sha256":"b"*64,
        "bone_xyz_sha256":source_surface_tensor_sha256(source),
        "bone_valid_sha256":source_surface_tensor_sha256(mask),
        "source_semantics":"NATIVE_RIG_BONE_CENTERLINE_SAMPLES_NOT_INFERENCE",
    }}


def test_two_view_real_iris_field_preserved_both_support_arms_same_at_init():
    model,state,rgba,cameras=_scene()
    research=RSSvNextDualSpatialSupportResearchV1(32,support_count=4,heads=4)
    common=dict(iris=model.iris,scene_planes=state.scene_triplanes,
        grid_xyz=state.query_xyz,grid_h=state.local_features)
    baseline=research(**common,mode="FIXED_VOLUME_SUPPORT")
    learned=research(**common,mode="LEARNED_INTERIOR_SUPPORT")
    assert baseline.support_xyz.shape==(1,4,3)
    assert learned.interior_tokens.shape==(1,4,32)
    assert learned.surface_tokens.shape==state.local_features.shape
    assert learned.mode=="LEARNED_INTERIOR_SUPPORT"
    assert torch.equal(baseline.base_query_indices,learned.base_query_indices)
    assert torch.allclose(baseline.support_xyz,learned.support_xyz,atol=0,rtol=0)
    assert torch.allclose(baseline.interior_tokens,learned.interior_tokens,atol=1e-6)
    assert torch.allclose(baseline.surface_tokens,learned.surface_tokens,atol=1e-6)
    assert learned.authority.startswith("RESEARCH_ONLY")
    assert model.cfg.observed_views==2 and cameras.shape[1]==2


def test_interior_coordinates_are_student_predictions_and_h_gradient_is_live():
    model,state,_,_=_scene()
    research,out=_apply(model,state)
    assert out.support_xyz.requires_grad
    with torch.no_grad():
        research.position_head[-1].bias.fill_(0.4)
    shifted=research(
        iris=model.iris,scene_planes=state.scene_triplanes,
        grid_xyz=state.query_xyz,grid_h=state.local_features,
        mode="LEARNED_INTERIOR_SUPPORT",
    )
    assert torch.any((shifted.support_xyz-out.support_xyz).abs()>1e-5)
    state.scene_triplanes.retain_grad()
    objective=shifted.interior_tokens.square().mean()
    objective=objective+shifted.surface_tokens.square().mean()
    objective=objective+0.03*shifted.support_xyz.square().mean()
    objective.backward()
    assert state.scene_triplanes.grad is not None
    assert torch.isfinite(state.scene_triplanes.grad).all()
    assert state.scene_triplanes.grad.abs().sum()>0
    assert research.position_head[-1].weight.grad is not None
    assert research.position_head[-1].weight.grad.abs().sum()>0
    assert research.interior_from_surface.in_proj_weight.grad is not None
    assert research.interior_from_surface.in_proj_weight.grad.abs().sum()>0
    assert research.surface_from_interior.in_proj_weight.grad is not None
    assert research.surface_from_interior.in_proj_weight.grad.abs().sum()>0


def test_teacher_centerlines_are_only_used_in_separate_receipted_loss():
    model,state,_,_=_scene()
    research,out=_apply(model,state)
    bone=torch.tensor([[[-.6,0.,0.],[0.,0.,0.],[.4,.2,0.],
                        [float("nan"),0.,0.]]])
    mask=torch.tensor([[True,True,True,False]])
    rec=_teacher_receipt(bone,mask)
    result=source_qualified_bone_support_loss_v1(
        out,source_bone_xyz=bone,source_bone_valid=mask,
        source_asset_sha256="a"*64,source_admission_key="test-source-native",
        dataset_manifest_sha256="b"*64,trusted_source_bone_registry=rec,
    )
    assert result.eligible_bone_samples==3
    assert torch.isfinite(result.loss)
    assert result.bone_to_proposal_mean_distance>=0
    assert result.proposal_to_bone_mean_distance>=0
    result.loss.backward()
    assert research.position_head[-1].weight.grad is not None
    assert research.position_head[-1].weight.grad.abs().sum()>0
    assert model.iris.scene.upsample.weight.grad is not None
    assert model.iris.scene.upsample.weight.grad.abs().sum()>0
    assert bone.grad is None  # GT is not a student forward condition


def test_source_teacher_receipts_are_not_self_awarded_or_silent_zero_negatives():
    model,state,_,_=_scene()
    _,out=_apply(model,state)
    bone=torch.zeros(1,2,3);mask=torch.tensor([[True,False]])
    kw=dict(source_bone_xyz=bone,source_bone_valid=mask,
        source_asset_sha256="a"*64,source_admission_key="test-source-native",
        dataset_manifest_sha256="b"*64,
        trusted_source_bone_registry=_teacher_receipt(bone,mask))
    with pytest.raises(ValueError,match="SOURCE_ADMISSION_REQUIRED"):
        source_qualified_bone_support_loss_v1(
            out,**{**kw,"trusted_source_bone_registry":{}},
        )
    with pytest.raises(ValueError,match="SOURCE_BONE_RECEIPT_DRIFT"):
        source_qualified_bone_support_loss_v1(
            out,**{**kw,"source_asset_sha256":"c"*64},
        )
    with pytest.raises(ValueError,match="SOURCE_BONE_RECEIPT_DRIFT"):
        source_qualified_bone_support_loss_v1(
            out,**{**kw,"source_bone_xyz":bone+0.1},
        )
    with pytest.raises(ValueError,match="SOURCE_BONE_DOMAIN"):
        source_qualified_bone_support_loss_v1(
            out,**{**kw,"source_bone_valid":torch.zeros_like(mask)},
        )


def test_no_teacher_geometry_allowed_in_forward_and_invalid_mode_fails():
    model,state,_,_=_scene()
    research=RSSvNextDualSpatialSupportResearchV1(32,support_count=4,heads=4)
    kw=dict(iris=model.iris,scene_planes=state.scene_triplanes,
        grid_xyz=state.query_xyz,grid_h=state.local_features)
    with pytest.raises(ValueError,match="MODE_INVALID"):
        research(**kw,mode="SOURCE_TEACHER_BONES")
    with pytest.raises(TypeError):
        research(**kw,mode="LEARNED_INTERIOR_SUPPORT",
                 source_bone_xyz=torch.zeros((1,3,3)))
    with pytest.raises(ValueError,match="CARRIER_DOMAIN"):
        research(**{**kw,"grid_h":state.local_features[:,:,:-1]},
                 mode="LEARNED_INTERIOR_SUPPORT")
