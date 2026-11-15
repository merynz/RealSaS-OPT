from __future__ import annotations

"""Product six-view H: both observed source images directly condition it.

Synthetic wiring tests only; neither generated images nor these targets are
source 3D truth, trained quality or a ProductRevision.
"""

from dataclasses import replace

import pytest
import torch

from models.rss_vnext import (
    RSSvNextConfig, RSSvNextPatchEncoderForTests,
    RSSvNextUnifiedCharacterModel,
)
from models.rss_vnext.inference_checkpoint_v1 import _load_config
from models.rss_vnext.realsas_scene_conditioning_v1 import (
    SIX_VIEW_PRIOR, encode_realsas_conditioning_into_shared_H_v1,
)
from models.rss_vnext.realsas_virtual_camera_policy_v1 import (
    target_canonical_and_six_cameras_v1,
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(942)
    torch.set_num_threads(1)


def _case():
    cfg = RSSvNextConfig.tiny_for_test()
    model = RSSvNextUnifiedCharacterModel(
        cfg,
        vision_encoder=RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim),
    ).eval()
    original = torch.rand((1, 2, 4, 16, 16), requires_grad=True)
    generated = torch.rand((1, 6, 4, 16, 16))
    normals = torch.rand((1, 6, 3, 16, 16)) * 2 - 1
    _, cameras = target_canonical_and_six_cameras_v1(batch_size=1)
    return model, original, generated, normals, cameras


def _scene(model, original, generated, normals, cameras):
    return model.encode_generated_multiview(
        original, None,
        generated_rgba=generated,
        generated_normals=normals,
        generated_cameras=cameras,
        generation_checkpoint_sha256="a" * 64,
        source_binding_sha256="b" * 64,
    )


def test_both_original_source_views_directly_reach_one_shared_H_and_graphics():
    net, originals, generated, normals, cameras = _case()
    state = _scene(net, originals, generated, normals, cameras)
    assert state.scene_triplanes.shape[1] == 3
    assert state.generated_view_count == 6
    assert state.camera_features is None
    assert state.source_camera_authority == "UNKNOWN_NO_ORIGINAL_CAMERA_PROJECTION"
    assert state.source_view_authority == (
        "SOURCE_OBSERVED_UNCALIBRATED_0", "SOURCE_OBSERVED_UNCALIBRATED_1",
    )
    for i in (0, 1):
        objective = state.local_features.square().mean() + state.graphics_logits.square().mean()
        gradient, = torch.autograd.grad(
            objective, originals, retain_graph=i == 0,
        )
        assert gradient[:, i].abs().sum() > 0, i
    assert net.iris.view_bank.original_view_slot_embedding.weight.requires_grad
    assert state.scene_conditioning_authority == "GENERATED_SIX_VIEW_SHARED_SCENE"


def test_changed_source_pixels_change_H_even_if_generated_six_are_identical():
    net, originals, generated, normals, cameras = _case()
    state1 = _scene(net, originals, generated, normals, cameras)
    mutated = originals.detach().clone()
    mutated[:, 1, :3] = 1. - mutated[:, 1, :3]
    state2 = _scene(net, mutated, generated, normals, cameras)
    assert not torch.allclose(
        state1.scene_triplanes, state2.scene_triplanes, atol=1e-7, rtol=0,
    )
    assert not torch.allclose(
        state1.local_features, state2.local_features, atol=1e-7, rtol=0,
    )


def test_generated_six_cannot_bypass_direct_original_memory_contract():
    net, originals, generated, normals, cameras = _case()
    with pytest.raises(ValueError, match="ORIGINAL_SOURCE_MEMORY_REQUIRED"):
        encode_realsas_conditioning_into_shared_H_v1(
            net, conditioning_rgba=generated,
            conditioning_cameras=cameras,
            conditioning_normals=normals,
            source_binding_sha256="b" * 64,
            original_observed_view_count=2,
            conditioning_authority=SIX_VIEW_PRIOR,
            generating_checkpoint_sha256="a" * 64,
        )


def test_original_source_memory_has_no_imaginary_camera_gradient_or_projection():
    net, originals, generated, normals, cameras = _case()
    source, xy, early = net._observed_pixel_tokens_without_camera(originals)
    observed = net.iris.view_bank.forward_uncalibrated_original(
        early.conditioned_image_tokens, xy,
    )
    observed.square().mean().backward()
    assert net.iris.view_bank.image_projection.weight.grad is not None
    assert net.iris.view_bank.image_projection.weight.grad.abs().sum() > 0
    assert net.iris.view_bank.camera_projection[0].weight.grad is None
    assert source.shape[1] == 2


def test_old_checkpoint_config_cannot_silently_become_source_fused_model():
    cfg = RSSvNextConfig.tiny_for_test()
    from dataclasses import asdict
    payload = asdict(cfg)
    del payload["original_source_direct_h_conditioning"]
    with pytest.raises(RuntimeError, match="CHECKPOINT_CONFIG_KEYS"):
        _load_config(payload)
    with pytest.raises(ValueError, match="DIRECT_ORIGINAL_SOURCE_H_REQUIRED"):
        replace(cfg, original_source_direct_h_conditioning=False).validate()
