from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import io
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from compiler.realsas_compiler_core.rss_vnext_uncalibrated_upload_v1 import (
    RSSvNextUncalibratedTwoImageSourceV1,
    encode_uncalibrated_two_image_source_v1,
)
from models.rss_vnext import (
    RSSvNextConfig, RSSvNextPatchEncoderForTests,
    RSSvNextUnifiedCharacterModel,
)
from models.rss_vnext.realsas_cleanroom_diffusion_training_v1 import (
    STAGE_CANONICAL, STAGE_MULTIVIEW,
    PairedSourceDiffusionConditionerV1, RowWiseRGBNormalViewAttentionV1,
)
from models.rss_vnext.realsas_two_stage_sampler_v1 import (
    RSSvNextPinnedDiffusionStageV1,
)
from models.rss_vnext.realsas_product_neural_bundle_v1 import (
    RSSvNextPinnedNeuralBundleV1,
)
from models.rss_vnext.realsas_raw_two_image_inference_v1 import (
    run_uncalibrated_two_image_neural_scene_v1,
)


class _ScheduleConfig(dict):
    __getattr__ = dict.get


class _Scheduler:
    init_noise_sigma = 1.
    def __init__(self):
        self.config = _ScheduleConfig(
            prediction_type="epsilon", num_train_timesteps=12,
        )
    def set_timesteps(self, steps, *, device=None):
        self.timesteps = torch.arange(
            steps - 1, -1, -1, device=device,
        )
    def step(self, noise, timestep, latents):
        return SimpleNamespace(prev_sample=latents - .05 * noise)


class _VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(scaling_factor=1.)
    def decode(self, latents):
        values = F.interpolate(
            latents[:, :3], scale_factor=8, mode="bilinear",
            align_corners=False,
        )
        return SimpleNamespace(sample=torch.tanh(values))


class _Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Conv2d(4, 4, 1)
    def forward(self, latents, timestep, *, encoder_hidden_states):
        assert encoder_hidden_states.shape[0] == latents.shape[0]
        return SimpleNamespace(sample=self.layer(latents))


def _stage(name, temp):
    stage = RSSvNextPinnedDiffusionStageV1(
        stage=name,
        denoiser=_Denoiser(),
        vae=_VAE(),
        conditioner=PairedSourceDiffusionConditionerV1(
            source_width=32, conditioning_width=32, heads=4,
        ),
        alpha_head=nn.Conv2d(3, 1, 1),
        scheduler=_Scheduler(),
        rowwise_attention=(
            RowWiseRGBNormalViewAttentionV1(
                latent_channels=4, hidden_width=8, heads=2,
            ) if name == STAGE_MULTIVIEW else None
        ),
    )
    path = temp / (name + ".pt")
    digest = stage.save_training_weights(path)
    stage.load_pinned_task_weights(path, expected_sha256=digest)
    return stage


def _bundle(tmp_path):
    cfg = replace(RSSvNextConfig.tiny_for_test(), constructed_total_views=0)
    model = RSSvNextUnifiedCharacterModel(
        cfg, vision_encoder=RSSvNextPatchEncoderForTests(
            cfg.iris.image_token_dim,
        ),
    ).eval().requires_grad_(False)
    model.construction_requires_generated_multiview = True
    return RSSvNextPinnedNeuralBundleV1(
        model=model,
        canonical=_stage(STAGE_CANONICAL, tmp_path),
        multiview=_stage(STAGE_MULTIVIEW, tmp_path),
        checkpoint_sha256="b" * 64,
        image_size=64,
        denoise_steps=1,
    )


def _image(color, size):
    image = Image.new("RGBA", size, color)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _raw_cas():
    # The two original observations have different source sizes and poses;
    # neither carries measured source camera, body mask, or 3D ray evidence.
    original = (
        _image((190, 80, 40, 255), (48, 32)),
        _image((25, 90, 210, 255), (24, 56)),
    )
    src = RSSvNextUncalibratedTwoImageSourceV1(
        original_files=original, source_rights_claim_sha256="a" * 64,
        view_claims=("FRONT", "BACK"),  # untrusted labels only
    )
    cas = encode_uncalibrated_two_image_source_v1(src)
    return cas, original


def test_raw_two_original_uploads_reach_shared_three_plane_scene_without_fake_camera(
    tmp_path,
):
    torch.manual_seed(831)
    torch.set_num_threads(1)
    cas, original = _raw_cas()
    bundle = _bundle(tmp_path)
    result = run_uncalibrated_two_image_neural_scene_v1(
        raw_cas_bytes=cas,
        expected_raw_cas_sha256=sha256(cas).hexdigest(),
        neural_bundle=bundle,
        canonical_seed=13, multiview_seed=37,
        max_topology_tokens=6,
    )
    assert result.source.original_files == original
    assert result.source.view_claims == ("FRONT", "BACK")
    assert result.source.source_images[0]["source_camera_calibrated"] is False
    assert result.source.source_images[1]["source_camera_calibrated"] is False
    assert result.source_cas_sha256 == sha256(cas).hexdigest()
    assert len(result.derived_conditioning_sha256) == 64
    assert not result.original_camera_observed
    assert result.original_camera_features is None
    assert not result.source_ray_evidence_qualified
    assert not result.generated_pixels_are_original
    assert not result.virtual_cameras_are_original
    assert not result.product_revision_minted
    construction = result.construction
    assert construction.original_camera_calibrated is False
    assert construction.generated_rgba.shape == (1, 6, 4, 64, 64)
    assert construction.generated_normals.shape == (1, 6, 3, 64, 64)
    scene = construction.proposal["state"]
    assert scene.camera_features is None
    assert scene.source_camera_authority == "UNKNOWN_NO_ORIGINAL_CAMERA_PROJECTION"
    assert scene.source_view_authority == (
        "SOURCE_OBSERVED_UNCALIBRATED_0",
        "SOURCE_OBSERVED_UNCALIBRATED_1",
    )
    assert scene.scene_triplanes.shape[1] == 3
    assert scene.scene_conditioning_authority == "GENERATED_SIX_VIEW_SHARED_SCENE"
    assert scene.generated_view_count == 6
    assert scene.observed_image_tokens.shape[1] == 2
    assert scene.region_relation_logits.shape[0] == 1
    assert scene.graphics_logits.shape[-1] == 5
    if construction.proposal["mesh_candidate"] is not None:
        assert construction.proposal["mesh_candidate"].source_ray_evidence is None


def test_source_cas_tamper_fails_before_inference(tmp_path):
    cas, _ = _raw_cas()
    bundle = _bundle(tmp_path)
    with pytest.raises(RuntimeError, match="SOURCE_CAS_PIN_DRIFT"):
        run_uncalibrated_two_image_neural_scene_v1(
            raw_cas_bytes=cas,
            expected_raw_cas_sha256="0" * 64,
            neural_bundle=bundle,
        )


def test_unknown_source_camera_cannot_enter_legacy_source_ray_or_control_mode(tmp_path):
    torch.manual_seed(833)
    net = _bundle(tmp_path).model
    original = torch.rand(1, 2, 4, 64, 64)
    with pytest.raises(RuntimeError, match="UNCALIBRATED_SOURCE_REQUIRES_GENERATED"):
        net.propose_teacher_free(
            original, None, max_topology_tokens=6,
        )
    with pytest.raises(RuntimeError, match="UNCALIBRATED_CONTROL_PROOF_NOT_READY"):
        net.propose_teacher_free(
            original, None, max_topology_tokens=6,
            generate_controls=True, executable_constraint_operators=("NO_OP",),
        )
