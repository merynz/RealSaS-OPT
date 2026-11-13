from __future__ import annotations

from dataclasses import replace
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from models.rss_vnext import (
    RSSvNextConfig, RSSvNextPatchEncoderForTests, RSSvNextUnifiedCharacterModel,
)
from models.rss_vnext.realsas_cleanroom_diffusion_training_v1 import (
    STAGE_CANONICAL, STAGE_MULTIVIEW, PairedSourceDiffusionConditionerV1,
    RowWiseRGBNormalViewAttentionV1,
)
from models.rss_vnext.realsas_two_stage_sampler_v1 import (
    RSSvNextPinnedDiffusionStageV1, admitted_alpha_head_loss_v1,
)
from models.rss_vnext.realsas_two_source_construction_v1 import (
    run_two_source_neural_construction_v1,
)


class FakeScheduleConfig(dict):
    __getattr__ = dict.get


class FakeScheduler:
    init_noise_sigma = 1.

    def __init__(self):
        self.config = FakeScheduleConfig(prediction_type="epsilon", num_train_timesteps=8)
        self.timesteps = []

    def set_timesteps(self, steps, *, device=None):
        self.timesteps = torch.arange(steps - 1, -1, -1, device=device)

    def step(self, prediction, timestep, latents):
        return SimpleNamespace(prev_sample=latents - .05 * prediction)


class FakeVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(scaling_factor=1.)

    def decode(self, latent):
        values = F.interpolate(latent[:, :3], scale_factor=8, mode="bilinear", align_corners=False)
        return SimpleNamespace(sample=torch.tanh(values))


class FakeDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Conv2d(4, 4, 1)

    def forward(self, latent, timestep, *, encoder_hidden_states):
        assert encoder_hidden_states.shape[0] == latent.shape[0]
        return SimpleNamespace(sample=self.net(latent))


def source_cameras():
    cam = torch.zeros(1, 2, 13)
    cam[0, 0, 2] = 2.
    cam[0, 1, 2] = -2.
    cam[0, 0, 3] = 1.
    cam[0, 1, 3] = -1.
    cam[0, :, 7] = 1.
    cam[0, 0, 11] = -1.
    cam[0, 1, 11] = 1.
    cam[0, :, 12] = 1.
    return cam


def six_cameras():
    cam = torch.zeros(1, 6, 13)
    for i in range(6):
        t = 2 * math.pi * i / 6
        radial = torch.tensor((math.sin(t), 0., math.cos(t)))
        cam[0, i, :3] = radial * 2.
        cam[0, i, 3:6] = torch.tensor((-math.cos(t), 0., math.sin(t)))
        cam[0, i, 7] = 1.
        cam[0, i, 9:12] = -radial
        cam[0, i, 12] = 1.
    return cam


def stage(name):
    return RSSvNextPinnedDiffusionStageV1(
        stage=name,
        denoiser=FakeDenoiser(),
        vae=FakeVAE(),
        conditioner=PairedSourceDiffusionConditionerV1(
            source_width=32, conditioning_width=32, heads=4,
        ),
        alpha_head=nn.Conv2d(3, 1, 1),
        scheduler=FakeScheduler(),
        rowwise_attention=(
            RowWiseRGBNormalViewAttentionV1(
                latent_channels=4, hidden_width=8, heads=2, row_chunk=2,
            )
            if name == STAGE_MULTIVIEW else None
        ),
    )


def model():
    cfg = replace(RSSvNextConfig.tiny_for_test(), constructed_total_views=0)
    return RSSvNextUnifiedCharacterModel(
        cfg, vision_encoder=RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim),
    )


def test_pinned_canonical_then_six_rgb_normal_stage_enters_same_scene(tmp_path):
    torch.manual_seed(78)
    torch.set_num_threads(1)
    canonical_stage = stage(STAGE_CANONICAL)
    multiview_stage = stage(STAGE_MULTIVIEW)
    with pytest.raises(RuntimeError, match="UNPINNED"):
        canonical_stage.generate(
            source_tokens=torch.randn(1, 2, 3, 32),
            target_cameras=six_cameras()[:, :2],
            target_modalities=torch.zeros(1, 2, dtype=torch.long),
            height=16, width=16, steps=1, seed=1,
        )
    for name, item in (("canonical", canonical_stage), ("multiview", multiview_stage)):
        dest = tmp_path / (name + ".pt")
        digest = item.save_training_weights(dest)
        item.load_pinned_task_weights(dest, expected_sha256=digest)
        assert all(not p.requires_grad for module in item._modules().values()
                   for p in module.parameters())
    original = torch.rand(1, 2, 4, 16, 16)
    net = model().eval()
    evidence = run_two_source_neural_construction_v1(
        model=net, observed_rgba=original,
        original_camera_features=source_cameras(),
        source_binding_sha256="b" * 64,
        canonical_cameras=six_cameras()[:, [0, 3]],
        six_view_cameras=six_cameras(),
        canonical_sampler=canonical_stage,
        multiview_sampler=multiview_stage,
        image_size=16, denoise_steps=1,
        canonical_seed=12, multiview_seed=34,
        max_topology_tokens=6,
    )
    assert evidence.original_observed_view_count == 2
    assert evidence.generated_view_count == 6
    assert not evidence.generated_is_observed
    assert not evidence.product_revision_minted
    assert evidence.canonical_rgba.shape == (1, 2, 4, 16, 16)
    assert evidence.generated_rgba.shape == (1, 6, 4, 16, 16)
    assert evidence.generated_normals.shape == (1, 6, 3, 16, 16)
    assert evidence.proposal["state"].observed_rgba is original
    assert evidence.proposal["state"].scene_conditioning_authority == "GENERATED_SIX_VIEW_SHARED_SCENE"
    assert evidence.proposal["state"].observed_image_tokens.shape[1] == 2
    assert evidence.proposal["qualification_status"] == "UNQUALIFIED_NEURAL_PROPOSAL"


def test_stage_checkpoint_hash_and_pairing_fail_closed(tmp_path):
    net = stage(STAGE_MULTIVIEW)
    path = tmp_path / "stage.pt"
    sha = net.save_training_weights(path)
    with pytest.raises(RuntimeError, match="PIN_OR_BYTES_DRIFT"):
        net.load_pinned_task_weights(path, expected_sha256="0" * 64)
    net.load_pinned_task_weights(path, expected_sha256=sha)
    camera = torch.cat((six_cameras(),six_cameras()),dim=1)
    modality = torch.tensor([[0] * 6 + [1] * 6], dtype=torch.long)
    camera[0, 7, 0] = 99.
    with pytest.raises(ValueError, match="PAIRING_REQUIRED"):
        net.generate(
            source_tokens=torch.randn(1, 2, 3, 32),
            target_cameras=camera, target_modalities=modality,
            height=16, width=16, steps=1, seed=3,
        )


def test_alpha_head_has_independent_training_gradient():
    head = nn.Conv2d(3, 1, 1)
    rgb = torch.rand(2, 3, 16, 16)
    alpha = torch.rand(2, 1, 16, 16)
    valid = torch.ones_like(alpha)
    loss = admitted_alpha_head_loss_v1(
        head, renderer_rgb=rgb, renderer_alpha=alpha, trusted_valid_mask=valid,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert head.weight.grad is not None
    assert head.weight.grad.abs().sum() > 0
