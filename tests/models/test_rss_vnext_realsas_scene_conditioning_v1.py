from __future__ import annotations

import math
import pytest
import torch

from models.rss_vnext import (
    RSSvNextConfig,RSSvNextPatchEncoderForTests,RSSvNextUnifiedCharacterModel,
)
from models.rss_vnext.realsas_scene_conditioning_v1 import (
    CANONICAL_REFERENCE,SIX_VIEW_PRIOR,ONE_MEASURED_SOURCE,
    encode_realsas_conditioning_into_shared_H_v1,
)
from models.rss_vnext.realsas_joint_slrm_render_v1 import (
    render_realsas_joint_from_shared_scene_v1,
)


def model():
    torch.manual_seed(14)
    cfg=RSSvNextConfig.tiny_for_test()
    return RSSvNextUnifiedCharacterModel(
        cfg,vision_encoder=RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim),
    )


def camera_bank(v):
    result=torch.zeros((1,v,13))
    if v==1:
        result[:,0,3]=result[:,0,7]=result[:,0,11]=result[:,0,12]=1.
    else:
        for i,deg in enumerate((-90,-45,0,45,90,180)):
            a=math.radians(deg)
            result[:,i,3]=math.cos(a)
            result[:,i,5]=-math.sin(a)
            result[:,i,7]=1.
            result[:,i,9]=math.sin(a)
            result[:,i,11]=math.cos(a)
            result[:,i,12]=1.
    return result


@pytest.mark.parametrize("views,auth",[ (1,CANONICAL_REFERENCE),(6,SIX_VIEW_PRIOR)])
def test_srlm_single_and_six_view_reuse_same_iris_H_graphics_semantic_weights(views,auth):
    net=model()
    images=torch.rand((1,views,4,16,16))
    cams=camera_bank(views)
    result=encode_realsas_conditioning_into_shared_H_v1(
        net,conditioning_rgba=images,conditioning_cameras=cams,
        source_binding_sha256="a"*64,original_observed_view_count=1,
        conditioning_authority=auth,generating_checkpoint_sha256="b"*64,
    )
    assert result.scene_triplanes.shape[1]==3
    assert result.conditioning_view_count==views
    assert result.original_observed_view_count==1
    assert not result.generated_views_became_source_observation
    assert not result.scene_geometry_is_source_truth
    rays=render_realsas_joint_from_shared_scene_v1(
        iris=net.iris,scene_triplanes=result.scene_triplanes,
        sample_xyz=torch.rand((1,2,3,3))*.5,
        ray_delta=torch.ones((1,2,3))*.25,
        ray_depth=torch.tensor([[[.1,.2,.3],[.1,.2,.3]]]),
        ray_direction=torch.tensor([[[0.,0.,1.],[0.,0.,1.]]]),
        keep_roles=torch.ones((1,2,net.cfg.semantic_role_schema().count),dtype=torch.bool),
        shared_graphics_head=net.graphics,
        shared_semantic_role_head=net.presentation_role_head,
    )
    (rays.rgb_white_composite.sum()+rays.semantic_posterior[...,0].sum()).backward()
    assert net.iris.view_bank.image_projection.weight.grad.abs().sum()>0
    assert net.graphics[-1].weight.grad.abs().sum()>0
    assert net.presentation_role_head[-1].weight.grad.abs().sum()>0


def test_original_two_view_public_input_is_still_strictly_separate():
    net=model()
    one=torch.rand((1,1,4,16,16))
    six=torch.rand((1,6,4,16,16))
    with pytest.raises(ValueError,match="ONLY_TWO_SOURCE_RGBA"):
        net.encode_observations(one,camera_bank(1))
    with pytest.raises(ValueError,match="ONLY_TWO_SOURCE_RGBA"):
        net.encode_observations(six,camera_bank(6))
    with pytest.raises(ValueError,match="TEST_PATCH_INPUT_INVALID"):
        net.vision(one)


def test_six_views_not_source_and_duplicate_or_wrong_camera_rejected():
    net=model()
    images=torch.rand((1,6,4,16,16))
    cams=camera_bank(6)
    with pytest.raises(ValueError,match="SIX_VIEWS_MUST_BE_GENERATED_PRIOR"):
        encode_realsas_conditioning_into_shared_H_v1(
            net,conditioning_rgba=images,conditioning_cameras=cams,
            source_binding_sha256="a"*64,original_observed_view_count=2,
            conditioning_authority=ONE_MEASURED_SOURCE,
        )
    cams[:,2]=cams[:,1]
    with pytest.raises(ValueError,match="DUPLICATE_PREDICTED_CAMERA"):
        encode_realsas_conditioning_into_shared_H_v1(
            net,conditioning_rgba=images,conditioning_cameras=cams,
            source_binding_sha256="a"*64,original_observed_view_count=2,
            conditioning_authority=SIX_VIEW_PRIOR,generating_checkpoint_sha256="b"*64,
        )


def test_cleanroom_one_measured_source_cannot_mint_second_observation():
    net=model()
    result=encode_realsas_conditioning_into_shared_H_v1(
        net,conditioning_rgba=torch.rand((1,1,4,16,16)),
        conditioning_cameras=camera_bank(1),
        source_binding_sha256="a"*64,
        original_observed_view_count=1,
        conditioning_authority=ONE_MEASURED_SOURCE,
    )
    assert result.original_observed_view_count==1
    assert result.generated_multiview_checkpoint_sha256 is None
    assert not result.source_camera_calibration_minted
    assert not result.generated_views_became_source_observation



def test_single_reference_rejects_invalid_camera_and_fake_source_count():
    net=model()
    image=torch.rand((1,1,4,16,16))
    cam=camera_bank(1)
    cam[:,:,3]=0.
    with pytest.raises(ValueError,match="CANONICAL_CAMERA_BASIS_INVALID"):
        encode_realsas_conditioning_into_shared_H_v1(
            net,conditioning_rgba=image,conditioning_cameras=cam,
            source_binding_sha256="a"*64,original_observed_view_count=1,
            conditioning_authority=ONE_MEASURED_SOURCE,
        )
    with pytest.raises(ValueError,match="ONE_VIEW_PROVENANCE_INVALID"):
        encode_realsas_conditioning_into_shared_H_v1(
            net,conditioning_rgba=image,conditioning_cameras=camera_bank(1),
            source_binding_sha256="a"*64,original_observed_view_count=2,
            conditioning_authority=ONE_MEASURED_SOURCE,
        )
