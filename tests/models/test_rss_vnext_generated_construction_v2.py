from __future__ import annotations

from dataclasses import replace
import math

import pytest
import torch

from models.rss_vnext import (
    RSSvNextConfig, RSSvNextPatchEncoderForTests,
    RSSvNextUnifiedCharacterModel,
)
from models.rss_vnext.realsas_slrm_stage1_lora_v1 import (
    install_rss_realsas_stage1_attention_lora_v1,
)
from models.rss_vnext.inference_checkpoint_v1 import (
    save_inference_checkpoint_v2, load_inference_checkpoint_v1,
)


def _source_cameras():
    cameras = torch.zeros(1, 2, 13)
    cameras[0, 0, 2] = 2.
    cameras[0, 1, 2] = -2.
    cameras[0, 0, 3] = 1.
    cameras[0, 1, 3] = -1.
    cameras[0, :, 7] = 1.
    cameras[0, 0, 11] = -1.
    cameras[0, 1, 11] = 1.
    cameras[0, :, 12] = 1.
    return cameras


def _six_cameras():
    cameras = torch.zeros(1, 6, 13)
    for i in range(6):
        angle = 2 * math.pi * i / 6
        radial = (math.sin(angle), 0., math.cos(angle))
        right = (-math.cos(angle), 0., math.sin(angle))
        cameras[0, i, :3] = torch.tensor(radial) * 2.
        cameras[0, i, 3:6] = torch.tensor(right)
        cameras[0, i, 7] = 1.
        cameras[0, i, 9:12] = -torch.tensor(radial)
        cameras[0, i, 12] = 1.
    return cameras


def _model():
    cfg = replace(RSSvNextConfig.tiny_for_test(), constructed_total_views=0)
    return RSSvNextUnifiedCharacterModel(
        cfg, vision_encoder=RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim),
    )


def test_six_predicted_views_and_normals_build_shared_scene_for_structural_heads():
    torch.manual_seed(31)
    torch.set_num_threads(1)
    model = _model()
    observed = torch.rand(1, 2, 4, 16, 16)
    generated = torch.rand(1, 6, 4, 16, 16, requires_grad=True)
    normals = torch.rand(1, 6, 3, 16, 16, requires_grad=True) * 2 - 1
    output = model.forward_generated_multiview(
        observed, _source_cameras(),
        generated_rgba=generated, generated_normals=normals,
        generated_cameras=_six_cameras(),
        generation_checkpoint_sha256="a" * 64,
        source_binding_sha256="b" * 64,
    )
    state = output.state
    assert state.scene_conditioning_authority == "GENERATED_SIX_VIEW_SHARED_SCENE"
    assert state.generated_view_count == 6
    assert state.observed_rgba is observed
    assert state.observed_image_tokens.shape[1] == 2
    assert state.scene_triplanes.shape[1] == 3
    assert output.structural_memory.shape[0] == 1
    assert state.presentation_role_logits.shape[:2] == state.local_features.shape[:2]
    loss = output.structural_memory.square().mean() + state.graphics_logits.square().mean()
    loss.backward()
    assert model.iris.scene.scene_queries.grad is not None
    assert model.normal_condition_proj.weight.grad is not None
    assert model.normal_condition_proj.weight.grad.abs().sum() > 0
    assert generated.grad is not None
    assert generated.grad.abs().sum() > 0
    assert model.tessa is not None and model.axis is not None and model.mira is not None


def test_generated_scene_validation_fails_on_invented_or_invalid_normal_cameras():
    model = _model()
    source = torch.rand(1, 2, 4, 16, 16)
    views = torch.rand(1, 6, 4, 16, 16)
    cams = _six_cameras()
    bad = cams.clone()
    bad[:, 4] = bad[:, 0]
    with pytest.raises(ValueError, match="DUPLICATE_PREDICTED_CAMERA"):
        model.encode_generated_multiview(
            source, _source_cameras(), generated_rgba=views,
            generated_normals=torch.rand(1, 6, 3, 16, 16),
            generated_cameras=bad,
            generation_checkpoint_sha256="a" * 64,
            source_binding_sha256="b" * 64,
        )
    with pytest.raises(ValueError, match="PREDICTED_NORMAL_CONDITIONING_INVALID"):
        model.encode_generated_multiview(
            source, _source_cameras(), generated_rgba=views,
            generated_normals=torch.full((1, 6, 3, 16, 16), 10.),
            generated_cameras=cams,
            generation_checkpoint_sha256="a" * 64,
            source_binding_sha256="b" * 64,
        )


def test_lora_reconstruction_v2_checkpoint_rebuilds_identical_inference_module(tmp_path):
    torch.manual_seed(37)
    torch.set_num_threads(1)
    model = _model()
    receipt = install_rss_realsas_stage1_attention_lora_v1(model, rank=2, alpha=3.)
    assert receipt.rank == 2
    path = tmp_path / "reconstruction-v2.pt"
    sha = save_inference_checkpoint_v2(path, model)
    original = {
        k: v.detach().cpu().clone() for k,v in model.state_dict().items()
    }
    restored = load_inference_checkpoint_v1(
        path, expected_sha256=sha,
        vision_encoder=RSSvNextPatchEncoderForTests(model.cfg.iris.image_token_dim),
        product_mode=False,
    )
    assert restored.construction_requires_generated_multiview
    assert restored._realsas_lora_config == (2, 3.)
    assert set(restored.state_dict()) == set(original)
    for key, value in restored.state_dict().items():
        assert torch.equal(value, original[key]), key
    with pytest.raises(RuntimeError, match="REQUIRES_CANONICAL_MULTIVIEW"):
        restored.propose_teacher_free(
            torch.rand(1, 2, 4, 16, 16), _source_cameras(),
            max_topology_tokens=8,
        )
    with pytest.raises(RuntimeError, match="THREE_STAGE_BUNDLE_REQUIRED"):
        load_inference_checkpoint_v1(
            path, expected_sha256=sha,
            vision_encoder=RSSvNextPatchEncoderForTests(model.cfg.iris.image_token_dim),
            product_mode=True,
        )
