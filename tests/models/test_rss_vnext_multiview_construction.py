from __future__ import annotations

"""RSS.vNext two-source 2->6/8 total camera and predicted-view contract court.

This is an experimental post-scene baseline, NOT StdGEN multi-view diffusion
parity. Tests exercise forward/backward and source authority. Toy tensors do
NOT imply generated unseen images are accurate.
"""
from dataclasses import replace
import math

import pytest
import torch

from models.rss_vnext import (
    RSSvNextConfig,
    RSSvNextPatchEncoderForTests,
    RSSvNextUnifiedCharacterModel,
)
from models.rss_vnext.multiview_construction import (
    canonical_novel_camera_ring,
    qualified_novel_view_loss,
)


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(44)
    torch.set_num_threads(1)


def source_cameras(batch=1):
    camera = torch.zeros(batch, 2, 13)
    camera[:, 0, 2] = 2
    camera[:, 1, 2] = -2
    camera[:, 0, 3] = 1
    camera[:, 1, 3] = -1
    camera[:, :, 7] = 1
    camera[:, 0, 11] = -1
    camera[:, 1, 11] = 1
    camera[:, :, 12] = 1
    return camera


def model(total=6):
    cfg = replace(RSSvNextConfig.tiny_for_test(), constructed_total_views=total)
    return RSSvNextUnifiedCharacterModel(
        cfg, vision_encoder=RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim),
    )


@pytest.mark.parametrize("total,novel", [(6,4), (8,6)])
def test_exact_two_observed_plus_novel_total_cameras(total, novel):
    source = source_cameras(batch=2)
    original = source.clone()
    ring = canonical_novel_camera_ring(source, total)
    assert ring.novel_cameras.shape == (2, novel, 13)
    assert ring.novel_angles_rad.shape == (novel,)
    assert torch.equal(ring.source_cameras, original)
    assert ring.source_authority == ("SOURCE_FRONT", "SOURCE_BACK")
    assert ring.novel_authority == "PREDICTED_CAMERA_TARGETS_ONLY"
    # No generated front/back duplicate (0°, pi) is allowed in the bank.
    assert not torch.isclose(ring.novel_angles_rad, torch.zeros_like(ring.novel_angles_rad)).any()
    assert not torch.isclose(ring.novel_angles_rad, torch.full_like(ring.novel_angles_rad, math.pi)).any()
    rot = ring.novel_cameras[..., 3:12].reshape(2, novel, 3, 3)
    gram = rot @ rot.transpose(-1, -2)
    assert torch.allclose(gram, torch.eye(3), atol=2e-5)
    assert torch.allclose(ring.novel_cameras[..., 12], torch.ones(2, novel))
    assert torch.isfinite(ring.novel_cameras).all()


def test_invalid_cameras_and_wrong_count_fail_closed():
    camera = source_cameras()
    for bad in (2,4,10):
        with pytest.raises(ValueError, match="TOTAL"):
            canonical_novel_camera_ring(camera, bad)
    wrong_back = camera.clone()
    wrong_back[:, 1, 0] = .3
    with pytest.raises(ValueError, match="ANTIPODAL"):
        canonical_novel_camera_ring(wrong_back, 6)
    wrong_rot = camera.clone()
    wrong_rot[:, 1, 7] = 0
    with pytest.raises(ValueError, match="ORTHONORMAL"):
        canonical_novel_camera_ring(wrong_rot, 6)
    with pytest.raises(ValueError, match="SOURCE_CAMERA_DOMAIN"):
        canonical_novel_camera_ring(camera[:, :1], 6)


def test_camera_conditioned_latents_affect_same_structural_transformer():
    net = model(6)
    rgba = torch.rand(1, 2, 4, 8, 8)
    source = source_cameras()
    out = net(rgba, source, topology_prefix=torch.full((1,4), net.cfg.tessa.BOS))
    assert out.state.constructed_camera_features.shape == (1, 4, 13)
    assert out.auxiliary_view_latents.shape == (1, 4, net.cfg.iris.image_token_dim)
    assert out.state.observed_image_tokens.shape[1] == 2
    assert out.state.source_view_authority == (
        "SOURCE_OBSERVED_FRONT","SOURCE_OBSERVED_BACK",
    )
    out.structural_logits.square().mean().backward()
    assert net.aux_views.target_camera_embed[-1].weight.grad is not None
    assert net.aux_views.target_camera_embed[-1].weight.grad.abs().sum() > 0
    assert net.aux_to_scene.weight.grad is not None
    assert net.iris.scene.scene_queries.grad is not None
    assert net.vision.patch.weight.grad is not None


@pytest.mark.parametrize("total,novel", [(6,4),(8,6)])
def test_experimental_render_outputs_actual_rgb_alpha_normal_and_gradients(total, novel):
    net = model(total)
    rgba = torch.rand(1, 2, 4, 8, 8)
    source = source_cameras()
    state, _, _ = net.encode_observations(rgba, source)
    assert state.observed_rgba is rgba
    with pytest.raises(ValueError, match="NOT_IMPLEMENTED"):
        net.construct_novel_views(state,height=4,width=4,method="STDGEN_PRE_SCENE_DIFFUSION")
    rendered = net.construct_novel_views(
        state, height=4, width=4, method="EXPERIMENTAL_POST_SCENE_H_RENDER",
    )
    assert rendered.rgba.shape == (1,novel,4,4,4)
    assert rendered.normal.shape == (1,novel,3,4,4)
    assert rendered.depth.shape == (1,novel,1,4,4)
    assert rendered.log_uncertainty.shape == (1,novel,1,4,4)
    assert rendered.source_view_count == 2
    assert rendered.authority == "PREDICTED_FROM_TWO_VIEWS_NO_NEW_SOURCE_EVIDENCE"
    assert torch.all((rendered.rgba >= 0) & (rendered.rgba <= 1))
    # Loss goes through novel renderer to shared two-source triplanes.
    loss = rendered.rgba[:, :, :3].square().mean() + rendered.rgba[:, :, 3].mean()
    loss.backward()
    assert net.constructed_view_renderer.rgb.weight.grad is not None
    assert net.constructed_view_renderer.rgb.weight.grad.abs().sum() > 0
    assert net.iris.scene.scene_queries.grad is not None
    assert net.vision.patch.weight.grad is not None


def test_generated_views_are_never_student_observations_and_teacher_provenance_is_strict():
    net = model(6)
    rgba = torch.rand(1,2,4,8,8)
    cam = source_cameras()
    state, _, _ = net.encode_observations(rgba,cam)
    novel = net.construct_novel_views(
        state,height=4,width=4,method="EXPERIMENTAL_POST_SCENE_H_RENDER",
    )
    with pytest.raises(ValueError,match="TWO_SOURCE_RGBA"):
        net.encode_observations(
            torch.cat((rgba,torch.nn.functional.interpolate(
                novel.rgba.reshape(4,4,4,4),size=(8,8),
                mode="bilinear",align_corners=False,
            ).reshape(1,4,4,8,8)),dim=1),cam,
        )
    target_rgba = novel.rgba.detach().clone()
    target_rgba[:, :, 0] = 0
    target_normal = novel.normal.detach().clone()
    visible = torch.ones((1,4,4,4),dtype=torch.bool)
    losses = qualified_novel_view_loss(
        novel,target_rgba,target_normal,
        rgb_qualified_mask=visible,normal_qualified_mask=visible,
        source_render_provenance="QUALIFIED_SOURCE_RENDER:synthetic-unit-test",
    )
    assert torch.isfinite(losses["total"])
    with pytest.raises(ValueError,match="PROVENANCE_REQUIRED"):
        qualified_novel_view_loss(
            novel,target_rgba,target_normal,
            rgb_qualified_mask=visible,normal_qualified_mask=visible,
            source_render_provenance="PREDICTED_VIEW_NOT_SOURCE",
        )
